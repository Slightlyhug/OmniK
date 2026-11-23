"""
数据管理模块
负责数据处理、数据集分割和数据加载器创建
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random

# 使用系统路径导入
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .common_utils import custom_collate_fn


class DataManager:
    """数据管理器，负责数据处理和加载器创建"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None  # 用于k-fold的完整数据集
    
    def prepare_data_loaders(self):
        """准备数据加载器"""
        # 1. 处理数据
        self._process_data()
        
        # 2. 分割数据集
        self._split_dataset()
        
        # 3. 创建数据加载器
        train_loader = self._create_train_loader()
        val_loader = self._create_val_loader()
        test_loader = self._create_test_loader()
        
        return train_loader, val_loader, test_loader
    
    def _process_data(self):
        """处理数据"""
        print("处理数据...")
        
        from src.data.unified_data_processor import UnifiedDataProcessor, UnifiedPTMDataset
        
        # 设置数据处理器
        target_ptm_type = str(self.config.get('target_ptm_type', 'phosphorylation')).strip().lower()
        window_size = self.config.get('window_size', 61)
        local_window_size = self.config.get('local_window_size', 31)
        fasta_dir = self.config.get('fasta_dir')
        
        data_processor = UnifiedDataProcessor(
            data_path=self.config['data_path'],
            esm_features_path=self.config.get('features_path', self.config.get('struct_features_path')),
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
            fasta_dir=fasta_dir
        )

        # 准备数据集
        processed_data = data_processor.prepare_dataset()
        residue_info = getattr(data_processor, 'residue_info', None) # UnifiedDataProcessor might not set this on self, but prepare_dataset might. 
        # Actually UnifiedDataProcessor.prepare_dataset returns list of dicts.
        # residue_info is usually resolved during dataset creation if not passed.
        # But UnifiedPTMDataset calculates it if None.
        
        original_dataset = UnifiedPTMDataset(
            processed_data,
            fixed_window_size=window_size,
            fixed_local_size=local_window_size,
            target_ptm_type=target_ptm_type,
            residue_info=residue_info
        )

        # 为k-fold保存原始数据集的独立副本
        self.full_dataset = UnifiedPTMDataset(
            processed_data,
            fixed_window_size=window_size,
            fixed_local_size=local_window_size,
            target_ptm_type=target_ptm_type,
            residue_info=residue_info
        )
        
        # 为常规训练创建另一个实例
        self.dataset = original_dataset

        if hasattr(original_dataset, 'residue_vocab_size'):
            self.config['num_residue_types'] = int(original_dataset.residue_vocab_size)
        if hasattr(original_dataset, 'residue_to_id'):
            self.config['residue_to_id'] = dict(original_dataset.residue_to_id)
        
        # 应用数据增强到self.dataset，但保持self.full_dataset不变
        self._apply_data_augmentation()

        if hasattr(self.dataset, 'ptm_type_to_id'):
            ptm_mapping = dict(self.dataset.ptm_type_to_id)
            self.config.setdefault('num_ptm_types', len(ptm_mapping))
            self.config['ptm_type_to_id'] = ptm_mapping
            print(f"PTM类型映射: {ptm_mapping}")
        
        print(f"数据集大小: {len(self.dataset)}")
        print(f"完整数据集大小: {len(self.full_dataset)}")

    def _apply_data_augmentation(self):
        """应用数据增强"""
        # 应用基础数据增强
        if self.config.get('use_data_augmentation', False):
            print("启用基础数据增强...")
            from models.advanced_components import DataAugmentationWrapper
            self.dataset = DataAugmentationWrapper(
                self.dataset, 
                augment_prob=self.config.get('augment_prob', 0.5),
                noise_level=self.config.get('noise_level', 0.1)
            )
        
        # 应用增强的数据增强到self.dataset，保持self.full_dataset原始
        if self.config.get('use_enhanced_augmentation', True):
            print("启用增强数据增强...")
            from models.advanced_components import EnhancedDataAugmentationWrapper
            self.dataset = EnhancedDataAugmentationWrapper(self.dataset, self.config)
            # 注意：不对self.full_dataset应用增强
    
    def _split_dataset(self):
        """分割数据集"""
        val_ratio = self.config.get('val_ratio', 0.2)
        test_ratio = self.config.get('test_ratio', 0.1)
        use_cv = self.config.get('cross_validation_folds', 0) > 1
        stratified_split = self.config.get('stratified_split', True)
        
        print(f"验证集比例: {val_ratio}, 测试集比例: {test_ratio}")
        print(f"交叉验证: {use_cv}, 分层抽样: {stratified_split}")
        
        # 准备标签信息
        all_labels = np.array([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        
        if use_cv:
            self._split_with_cv(all_labels, val_ratio, test_ratio, stratified_split)
        else:
            self._split_without_cv(all_labels, val_ratio, test_ratio, stratified_split)
        
        # 保存测试索引
        self._save_test_indices()
        
        print(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")
        if self.test_dataset is not None:
            print(f"测试集大小: {len(self.test_dataset)}")
    
    def _split_with_cv(self, all_labels, val_ratio, test_ratio, stratified_split):
        """使用交叉验证分割数据"""
        cv_folds = self.config.get('cross_validation_folds', 5)
        
        if stratified_split:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.args.seed)
            train_idx, val_idx = next(iter(skf.split(np.zeros(len(self.dataset)), all_labels)))
            
            # 如果需要测试集，从训练集中再划分
            if test_ratio > 0:
                test_size = int(len(train_idx) * test_ratio / (1 - val_ratio))
                train_labels = all_labels[train_idx]
                train_idx_new, test_idx = train_test_split(
                    train_idx, 
                    test_size=test_size, 
                    stratify=train_labels,
                    random_state=self.args.seed
                )
                train_idx = train_idx_new
            else:
                test_idx = None
        else:
            # 简单随机分割
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            fold_size = len(indices) // cv_folds
            val_idx = indices[:fold_size]
            remaining = indices[fold_size:]
            
            if test_ratio > 0:
                test_size = int(len(remaining) * test_ratio / (1 - val_ratio))
                test_idx = remaining[:test_size]
                train_idx = remaining[test_size:]
            else:
                train_idx = remaining
                test_idx = None
        
        # 创建数据集子集
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx) if test_idx is not None else None
    
    def _split_without_cv(self, all_labels, val_ratio, test_ratio, stratified_split):
        """不使用交叉验证分割数据"""
        # 检查是否使用蛋白质级别分割
        use_protein_split = self.config.get('use_protein_level_split', True)

        if use_protein_split:
            print("使用蛋白质级别数据分割...")
            train_idx, val_idx, test_idx = protein_level_split(
                self.dataset,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=self.args.seed
            )

            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            self.test_dataset = Subset(self.dataset, test_idx) if test_idx is not None else None

        elif stratified_split:
            if test_ratio > 0:
                # 三路分割
                train_idx, temp_idx, train_labels, temp_labels = train_test_split(
                    range(len(self.dataset)), all_labels,
                    test_size=val_ratio + test_ratio,
                    stratify=all_labels,
                    random_state=self.args.seed
                )

                # 进一步分割验证集和测试集
                relative_test_ratio = test_ratio / (val_ratio + test_ratio)
                val_idx, test_idx, _, _ = train_test_split(
                    temp_idx, temp_labels,
                    test_size=relative_test_ratio,
                    stratify=temp_labels,
                    random_state=self.args.seed
                )

                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
                self.test_dataset = Subset(self.dataset, test_idx)
            else:
                # 两路分割
                train_idx, val_idx, _, _ = train_test_split(
                    range(len(self.dataset)), all_labels,
                    test_size=val_ratio,
                    stratify=all_labels,
                    random_state=self.args.seed
                )

                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
                self.test_dataset = None
        else:
            # 使用随机分割
            if test_ratio > 0:
                dataset_size = len(self.dataset)
                train_size = int((1 - val_ratio - test_ratio) * dataset_size)
                val_size = int(val_ratio * dataset_size)
                test_size = dataset_size - train_size - val_size

                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    self.dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(self.args.seed)
                )
            else:
                train_size = int((1 - val_ratio) * len(self.dataset))
                val_size = len(self.dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.args.seed)
                )
                self.test_dataset = None
    
    def _save_test_indices(self):
        """保存测试集索引"""
        if self.test_dataset is not None:
            output_dir = self.config['output_dir']
            test_indices_path = os.path.join(output_dir, 'test_indices.json')
            
            if hasattr(self.test_dataset, 'indices'):
                test_indices = self.test_dataset.indices
            else:
                test_indices = list(range(len(self.test_dataset)))
            
            with open(test_indices_path, 'w') as f:
                json.dump(test_indices if isinstance(test_indices, list) else test_indices.tolist(), f)
    
    def _create_train_loader(self):
        """创建训练数据加载器"""
        pin_memory = torch.cuda.is_available()
        balanced_sampling = self.config.get('balanced_sampling', True)
        
        if balanced_sampling:
            print("使用平衡采样...")
            sampler = self._create_balanced_sampler()
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 16),
                sampler=sampler,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 16),
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
    
    def _create_balanced_sampler(self):
        """创建平衡采样器"""
        # 获取训练数据的标签
        all_labels = np.array([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        
        if isinstance(self.train_dataset, Subset):
            train_labels = [all_labels[i] for i in self.train_dataset.indices]
        else:
            train_labels = [all_labels[i] for i in range(len(self.train_dataset))]
        
        # 确保标签是整数类型
        train_labels = np.array(train_labels).astype(int)
        
        # 计算类别权重
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        weights = [class_weights[int(label)] for label in train_labels]
        
        print(f"类别计数: {class_counts}")
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def _create_val_loader(self):
        """创建验证数据加载器"""
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
    
    def _create_test_loader(self):
        """创建测试数据加载器"""
        if self.test_dataset is None:
            return None
        
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
    
    def get_full_dataset_for_kfold(self):
        """获取用于k-fold交叉验证的完整数据集"""
        if self.full_dataset is None:
            self._process_data()
        
        # 确保full_dataset不为None
        if self.full_dataset is None:
            print("错误: full_dataset仍为None，使用原始数据集")
            self.full_dataset = self.dataset
            
        print(f"获取k-fold数据集，大小: {len(self.full_dataset)}")
        
        # 直接从数据集获取所有标签
        all_labels = []
        for i in range(len(self.full_dataset)):
            try:
                sample = self.full_dataset[i]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                
                if isinstance(label, torch.Tensor):
                    all_labels.append(label.item())
                else:
                    all_labels.append(int(label))
            except Exception as e:
                print(f"获取样本 {i} 标签时出错: {e}")
                # 使用默认标签
                all_labels.append(0)
        
        # 为k-fold分割创建简单的索引数组
        X_for_split = np.arange(len(self.full_dataset))
        y_numpy = np.array(all_labels)
        
        print(f"k-fold数据: X形状={X_for_split.shape}, y形状={y_numpy.shape}")
        print(f"标签分布: 正样本={np.sum(y_numpy)}, 负样本={len(y_numpy)-np.sum(y_numpy)}")
        
        return X_for_split, y_numpy
    
    def create_kfold_loaders(self, train_indices, val_indices):
        """为k-fold创建训练和验证数据加载器"""
        # 确保数据已处理
        if self.full_dataset is None:
            print("full_dataset为None，重新处理数据...")
            self._process_data()
        
        # 再次检查
        if self.full_dataset is None:
            print("错误: 处理数据后full_dataset仍为None")
            raise ValueError("无法获取完整数据集")
            
        print(f"创建k-fold加载器: 训练索引数={len(train_indices)}, 验证索引数={len(val_indices)}")
        print(f"完整数据集大小: {len(self.full_dataset)}")
        
        # 验证索引范围
        max_train_idx = max(train_indices) if len(train_indices) > 0 else -1
        max_val_idx = max(val_indices) if len(val_indices) > 0 else -1
        max_dataset_idx = len(self.full_dataset) - 1
        
        if max_train_idx > max_dataset_idx or max_val_idx > max_dataset_idx:
            print(f"警告: 索引超出范围! 最大训练索引={max_train_idx}, 最大验证索引={max_val_idx}, 数据集最大索引={max_dataset_idx}")
            # 过滤无效索引
            train_indices = [i for i in train_indices if i <= max_dataset_idx]
            val_indices = [i for i in val_indices if i <= max_dataset_idx]
            print(f"过滤后: 训练索引数={len(train_indices)}, 验证索引数={len(val_indices)}")
        
        # 创建子数据集
        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)
        
        print(f"子数据集创建成功: 训练集={len(train_subset)}, 验证集={len(val_subset)}")
        
        # 为k-fold使用原始数据集，不应用数据增强
        # 注释掉数据增强部分，因为在k-fold中我们希望保持数据的一致性
        # if self.config.get('use_enhanced_augmentation', False):
        #     train_subset = self._apply_augmentation_to_subset(train_subset)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        
        print(f"K-Fold数据加载器创建完成:")
        print(f"  训练集: {len(train_subset)} 样本")
        print(f"  验证集: {len(val_subset)} 样本")
        
        return train_loader, val_loader


def protein_level_split(dataset, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    基于蛋白质ID的数据分割，确保同一蛋白质的所有序列都分配到同一个数据集中

    Args:
        dataset: 数据集，每个样本应包含'uniprot_id'字段
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子

    Returns:
        train_indices, val_indices, test_indices: 训练、验证、测试集的索引
    """
    # 收集所有蛋白质ID和对应的样本索引
    protein_to_indices = defaultdict(list)
    protein_to_labels = defaultdict(list)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        protein_id = sample['uniprot_id']
        label = sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label']

        protein_to_indices[protein_id].append(idx)
        protein_to_labels[protein_id].append(label)

    # 计算每个蛋白质的标签分布（用于分层）
    protein_label_stats = {}
    for protein_id, labels in protein_to_labels.items():
        positive_count = sum(labels)
        total_count = len(labels)
        protein_label_stats[protein_id] = {
            'positive_ratio': positive_count / total_count,
            'total_samples': total_count,
            'positive_count': positive_count
        }

    # 获取所有蛋白质ID列表
    protein_ids = list(protein_to_indices.keys())

    # 为分层分割创建蛋白质级别的标签
    # 使用每个蛋白质的正样本比例作为分层依据
    protein_labels = []
    for protein_id in protein_ids:
        ratio = protein_label_stats[protein_id]['positive_ratio']
        # 将连续的比例转换为离散的类别
        if ratio == 0:
            protein_labels.append(0)  # 全负样本
        elif ratio == 1:
            protein_labels.append(2)  # 全正样本
        else:
            protein_labels.append(1)  # 混合样本

    print(f"蛋白质级别分割统计:")
    print(f"  总蛋白质数: {len(protein_ids)}")
    print(f"  全负样本蛋白质: {protein_labels.count(0)}")
    print(f"  混合样本蛋白质: {protein_labels.count(1)}")
    print(f"  全正样本蛋白质: {protein_labels.count(2)}")

    # 设置随机种子
    random.seed(random_state)
    np.random.seed(random_state)

    # 分割蛋白质ID
    if test_ratio > 0:
        # 三路分割：训练、验证、测试
        try:
            train_proteins, temp_proteins, _, _ = train_test_split(
                protein_ids, protein_labels,
                test_size=val_ratio + test_ratio,
                stratify=protein_labels,
                random_state=random_state
            )

            # 重新计算临时集合的标签
            temp_labels = [protein_labels[protein_ids.index(pid)] for pid in temp_proteins]
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)

            val_proteins, test_proteins, _, _ = train_test_split(
                temp_proteins, temp_labels,
                test_size=relative_test_ratio,
                stratify=temp_labels,
                random_state=random_state
            )
        except ValueError as e:
            print(f"分层分割失败，使用随机分割: {e}")
            # 随机分割作为后备方案
            shuffled_proteins = protein_ids.copy()
            random.shuffle(shuffled_proteins)

            n_proteins = len(shuffled_proteins)
            n_test = int(n_proteins * test_ratio)
            n_val = int(n_proteins * val_ratio)

            test_proteins = shuffled_proteins[:n_test]
            val_proteins = shuffled_proteins[n_test:n_test + n_val]
            train_proteins = shuffled_proteins[n_test + n_val:]
    else:
        # 两路分割：训练、验证
        try:
            train_proteins, val_proteins, _, _ = train_test_split(
                protein_ids, protein_labels,
                test_size=val_ratio,
                stratify=protein_labels,
                random_state=random_state
            )
            test_proteins = []
        except ValueError as e:
            print(f"分层分割失败，使用随机分割: {e}")
            # 随机分割作为后备方案
            shuffled_proteins = protein_ids.copy()
            random.shuffle(shuffled_proteins)

            n_proteins = len(shuffled_proteins)
            n_val = int(n_proteins * val_ratio)

            val_proteins = shuffled_proteins[:n_val]
            train_proteins = shuffled_proteins[n_val:]
            test_proteins = []

    # 收集每个集合的样本索引
    train_indices = []
    val_indices = []
    test_indices = []

    for protein_id in train_proteins:
        train_indices.extend(protein_to_indices[protein_id])

    for protein_id in val_proteins:
        val_indices.extend(protein_to_indices[protein_id])

    for protein_id in test_proteins:
        test_indices.extend(protein_to_indices[protein_id])

    # 统计信息
    print(f"蛋白质级别分割结果:")
    print(f"  训练集: {len(train_proteins)} 蛋白质, {len(train_indices)} 样本")
    print(f"  验证集: {len(val_proteins)} 蛋白质, {len(val_indices)} 样本")
    if test_proteins:
        print(f"  测试集: {len(test_proteins)} 蛋白质, {len(test_indices)} 样本")

    # 验证没有蛋白质重叠
    all_train_proteins = set(train_proteins)
    all_val_proteins = set(val_proteins)
    all_test_proteins = set(test_proteins)

    assert len(all_train_proteins & all_val_proteins) == 0, "训练集和验证集有蛋白质重叠"
    assert len(all_train_proteins & all_test_proteins) == 0, "训练集和测试集有蛋白质重叠"
    assert len(all_val_proteins & all_test_proteins) == 0, "验证集和测试集有蛋白质重叠"

    print("✓ 蛋白质级别分割验证通过，无重叠")

    return train_indices, val_indices, test_indices if test_indices else None
