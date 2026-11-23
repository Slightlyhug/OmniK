#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负样本增强模块 - 提高负样本多样性和质量
包括：Hard Negative Mining、序列相似度去重、结构相似度去重
"""

import numpy as np
import pandas as pd
import h5py
import logging
import os
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import random
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import normalize
from functools import lru_cache

class SequenceAugmentation:
    """序列增强工具 - 用于生成增强的蛋白质序列"""
    def __init__(self):
        self.amino_acid_groups = {
            'hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y'],
            'polar': ['S', 'T', 'N', 'Q'],
            'charged': ['K', 'R', 'H', 'D', 'E'],
            'special': ['C', 'G', 'P']
        }
    
    def conservative_substitution(self, sequence: str, position: int, prob: float = 0.1) -> str:
        """
        保守性氨基酸替换
        
        Args:
            sequence: 原始蛋白质序列
            position: 替换发生的位置 (0-based index)
            prob: 替换发生的概率
            
        Returns:
            增强后的序列
        """
        if not sequence or position < 0 or position >= len(sequence):
            return sequence
            
        if random.random() < prob:
            original_aa = sequence[position]
            for group, aas in self.amino_acid_groups.items():
                if original_aa in aas:
                    # 从同一组中选择不同的氨基酸
                    candidates = [aa for aa in aas if aa != original_aa]
                    if candidates:
                        new_aa = random.choice(candidates)
                        # 替换字符
                        return sequence[:position] + new_aa + sequence[position+1:]
        return sequence

logger = logging.getLogger('negative_enhancement')


class NegativeSampleEnhancer:
    """负样本增强器 - 提高负样本的多样性和质量"""
    
    def __init__(self, esm_features_path: str, positive_data: pd.DataFrame):
        """
        初始化负样本增强器
        
        Args:
            esm_features_path: ESM特征文件路径
            positive_data: 正样本数据
        """
        self.esm_features_path = esm_features_path
        self.positive_data = positive_data
        self.positive_sites = set(zip(positive_data['uniprot_id'], positive_data['position']))
        
    @lru_cache(maxsize=1000)
    def get_protein_features(self, protein_id: str) -> Optional[Dict[str, np.ndarray]]:
        """按需加载蛋白质特征"""
        if not os.path.exists(self.esm_features_path):
            return None
            
        try:
            with h5py.File(self.esm_features_path, 'r') as f:
                if 'proteins' not in f or protein_id not in f['proteins']:
                    return None
                
                protein_group = f['proteins'][protein_id]
                features = {}
                
                # 加载 sequence_features
                if 'sequence_features' in protein_group:
                    features['sequence_features'] = protein_group['sequence_features'][()]
                
                # 加载 structure_features
                if 'structure_features' in protein_group:
                    features['structure_features'] = protein_group['structure_features'][()]
                    
                return features
        except Exception as e:
            logger.error(f"读取蛋白质 {protein_id} 特征失败: {e}")
            return None

class HardNegativeMiner:
    """困难负样本挖掘器 - 挖掘模型预测置信度高但实际为负的样本"""
    
    def __init__(self, model, threshold: float = 0.7, batch_size: int = 32):
        """
        初始化挖掘器
        
        Args:
            model: 训练好的模型
            threshold: 判定为困难负样本的置信度阈值
            batch_size: 批量处理大小
        """
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
        
    def mine_hard_negatives(self, candidates_data: List[Dict], device: str = 'cuda') -> List[Dict]:
        """
        从候选负样本中挖掘困难样本 (模型预测置信度高但实际为负)
        
        Args:
            candidates_data: 候选负样本列表，每个元素为样本字典
            device: 计算设备
            
        Returns:
            困难负样本列表
        """
        hard_negatives = []
        self.model.eval()
        self.model.to(device)
        
        import torch
        from src.utils.common_utils import custom_collate_fn
        
        # 分批处理
        for i in range(0, len(candidates_data), self.batch_size):
            batch_samples = candidates_data[i:i + self.batch_size]
            
            try:
                # 使用 colate_fn 构建 batch
                batch = custom_collate_fn(batch_samples)
                
                # 移动到设备
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = self.model(batch)
                    
                    # 获取预测概率
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('predicted_outcome'))
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                        
                    if isinstance(logits, torch.Tensor):
                        if logits.shape[-1] == 2:
                            probs = torch.softmax(logits, dim=-1)[:, 1]
                        else:
                            probs = torch.sigmoid(logits).squeeze()
                            if probs.dim() == 0:
                                probs = probs.unsqueeze(0)
                        
                        # 筛选困难样本 (预测概率 > 阈值)
                        hard_indices = (probs > self.threshold).nonzero(as_tuple=True)[0]
                        
                        for idx in hard_indices:
                            if idx < len(batch_samples):
                                sample = batch_samples[idx.item()]
                                sample['mining_score'] = probs[idx].item()
                                hard_negatives.append(sample)
                                
            except Exception as e:
                logger.warning(f"Hard negative mining batch failed: {e}")
                continue
            
        return hard_negatives
    
    def select_hard_negatives(self, negative_pool: List[Dict], count: int, processor=None) -> List[Dict]:
        """
        从负样本池中选择最难的样本
        
        Args:
            negative_pool: 负样本池
            count: 需要选择的数量
            processor: 数据处理器，用于将样本转换为模型输入
            
        Returns:
            选中的困难负样本
        """
        if not negative_pool:
            return []
            
        import torch
        self.model.eval()
        device = next(self.model.parameters()).device
        
        scores = []
        batch_size = self.batch_size
        
        with torch.no_grad():
            for i in range(0, len(negative_pool), batch_size):
                batch_samples = negative_pool[i:i + batch_size]
                
                # 使用processor转换数据
                if processor:
                    # 假设processor有一个collate_fn或者类似的方法
                    # 这里我们需要构建一个临时的DataLoader或者直接调用处理函数
                    # 简化处理：假设processor.process_batch可用，或者我们手动构建tensor
                    
                    # 尝试构建 batch input
                    try:
                        # 这是一个简化的假设，实际需要根据 unified_data_processor 的输出结构调整
                        # 假设 batch_samples 已经是处理好的 dict 列表
                        from src.utils.common_utils import custom_collate_fn
                        batch = custom_collate_fn(batch_samples)
                        
                        # 移动到设备
                        if isinstance(batch, dict):
                            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        
                        outputs = self.model(batch)
                        
                        # 获取预测概率
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('predicted_outcome'))
                        else:
                            logits = outputs
                            
                        if logits.shape[-1] == 2:
                            probs = torch.softmax(logits, dim=-1)[:, 1]
                        else:
                            probs = torch.sigmoid(logits).squeeze()
                            
                        scores.extend(probs.cpu().tolist())
                        
                    except Exception as e:
                        logger.warning(f"Hard negative mining batch failed: {e}")
                        # 填充默认分数
                        scores.extend([0.0] * len(batch_samples))
                else:
                     scores.extend([0.0] * len(batch_samples))

        # 根据分数排序 (降序)，分数越高越像正样本（即越难的负样本）
        scored_negatives = list(zip(negative_pool, scores))
        scored_negatives.sort(key=lambda x: x[1], reverse=True)
        
        # 筛选出超过阈值的，或者直接取前 count 个
        hard_ones = [item[0] for item in scored_negatives if item[1] > self.threshold]
        
        # 如果不够，补充置信度最高的剩余样本
        if len(hard_ones) < count:
            remaining_needed = count - len(hard_ones)
            # 从剩下的里面取（已经排过序了）
            # 注意要去重
            existing_ids = {id(x) for x in hard_ones}
            for item, score in scored_negatives:
                if len(hard_ones) >= count:
                    break
                if id(item) not in existing_ids:
                    hard_ones.append(item)
                    
        return hard_ones[:count]

    def get_site_feature(self, protein_id: str, position: int, feature_type: str = 'sequence') -> Optional[np.ndarray]:
        """获取特定位点的特征"""
        features = self.get_protein_features(protein_id)
        if not features:
            return None
            
        key = f"{feature_type}_features"
        if key not in features:
            return None
            
        feat_matrix = features[key]
        pos_idx = int(position) - 1
        
        if 0 <= pos_idx < len(feat_matrix):
            return feat_matrix[pos_idx]
        return None

    
    def hard_negative_mining(self, candidates: List[Dict], 
                            similarity_threshold: float = 0.7,
                            top_k: int = None) -> List[Dict]:
        """
        Hard Negative Mining - 选择与正样本特征相似的负样本
        这些样本更具挑战性，能更好地训练模型
        
        Args:
            candidates: 候选负样本列表
            similarity_threshold: 相似度阈值
            top_k: 返回前k个最相似的样本
            
        Returns:
            筛选后的负样本列表
        """
        logger.info(f"执行Hard Negative Mining，候选样本数: {len(candidates)}")
        
        # 预加载正样本特征
        pos_features = []
        for _, pos_row in self.positive_data.iterrows():
            feat = self.get_site_feature(pos_row['uniprot_id'], pos_row['position'])
            if feat is not None:
                pos_features.append(feat)
        
        if not pos_features:
            logger.warning("无法加载正样本特征，跳过 Hard Mining")
            return candidates
            
        pos_matrix = np.array(pos_features)
        # 归一化正样本特征
        pos_matrix = normalize(pos_matrix, norm='l2')
        
        hard_negatives = []
        
        for candidate in candidates:
            protein_id = candidate['uniprot_id']
            position = candidate['position']
            
            # 获取该位点的ESM特征
            neg_feature = self.get_site_feature(protein_id, position)
            if neg_feature is None:
                continue
            
            # 归一化
            neg_feature = neg_feature.reshape(1, -1)
            neg_feature = normalize(neg_feature, norm='l2')
            
            # 计算与所有正样本的相似度 (矩阵乘法)
            similarities = np.dot(neg_feature, pos_matrix.T)
            max_similarity = np.max(similarities)
            
            # 保留相似度高于阈值的样本（Hard Negatives）
            if max_similarity >= similarity_threshold:
                candidate['similarity_score'] = float(max_similarity)
                hard_negatives.append(candidate)
        
        logger.info(f"Hard Negative Mining 结果: {len(hard_negatives)} 个样本")
        
        # 如果指定了top_k，返回相似度最高的k个
        if top_k and len(hard_negatives) > top_k:
            hard_negatives = sorted(hard_negatives, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
        
        return hard_negatives

class HardNegativeMiner:
    """困难负样本挖掘器 - 挖掘模型预测置信度高但实际为负的样本 (Online Mining)"""
    
    def __init__(self, model, threshold: float = 0.7, batch_size: int = 32):
        """
        初始化挖掘器
        """
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
    
    def mine_hard_negatives(self, candidates_data: List[Dict], device: str = 'cuda') -> List[Dict]:
        """
        从候选负样本中挖掘困难样本
        """
        # Placeholder implementation - real implementation would involve running model inference
        # Since we are implementing OHEM in the loss function, this class is mainly for
        # potential offline analysis or future advanced sampling strategies.
        hard_negatives = []
        # Logic to run model on candidates and select those with p > threshold
        return hard_negatives
    
    def sequence_similarity_deduplication(self, candidates: List[Dict],
                                         similarity_threshold: float = 0.9) -> List[Dict]:
        """
        基于序列相似度的去重 - 移除过于相似的负样本
        
        Args:
            candidates: 候选负样本列表
            similarity_threshold: 相似度阈值（高于此值的样本被认为重复）
            
        Returns:
            去重后的负样本列表
        """
        logger.info(f"执行序列相似度去重，候选样本数: {len(candidates)}")
        
        if len(candidates) == 0:
            return candidates
        
        # 提取ESM特征
        features_list = []
        valid_indices = []
        
        for i, candidate in enumerate(candidates):
            feat = self.get_site_feature(candidate['uniprot_id'], candidate['position'], 'sequence')
            if feat is not None:
                features_list.append(feat)
                valid_indices.append(i)
        
        if len(features_list) == 0:
            return candidates
        
        # 归一化特征
        features_array = np.array(features_list)
        features_normalized = normalize(features_array, norm='l2')
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        
        # 贪心去重
        kept_indices = set()
        removed_indices = set()
        for i in range(len(features_list)):
            if i in removed_indices:
                continue
            
            kept_indices.add(i)
            
            # 移除与当前样本相似度过高的其他样本
            for j in range(i + 1, len(features_list)):
                if j not in removed_indices and similarity_matrix[i, j] > similarity_threshold:
                    removed_indices.add(j)
        
        # 返回去重后的样本
        deduplicated = [candidates[valid_indices[i]] for i in sorted(kept_indices)]
        logger.info(f"序列相似度去重结果: {len(deduplicated)} 个样本")
        
        return deduplicated
    
    def structure_similarity_deduplication(self, candidates: List[Dict],
                                          similarity_threshold: float = 0.85) -> List[Dict]:
        """
        基于结构相似度的去重 - 使用ESM特征计算结构相似度
        
        Args:
            candidates: 候选负样本列表
            similarity_threshold: 相似度阈值
            
        Returns:
            去重后的负样本列表
        """
        logger.info(f"执行结构相似度去重，候选样本数: {len(candidates)}")
        
        if len(candidates) == 0:
            return candidates
        
        # 提取特征
        features_dict = {}
        for i, candidate in enumerate(candidates):
            feat = self.get_site_feature(candidate['uniprot_id'], candidate['position'], 'structure')
            if feat is not None:
                features_dict[i] = feat
        
        if len(features_dict) == 0:
            return candidates
        
        # 聚类去重
        kept_indices = set()
        removed_indices = set()
        
        indices = sorted(features_dict.keys())
        for i in indices:
            if i in removed_indices:
                continue
            
            kept_indices.add(i)
            feat_i = features_dict[i]
            
            # 与其他样本比较
            for j in indices:
                if j <= i or j in removed_indices:
                    continue
                
                feat_j = features_dict[j]
                similarity = 1 - cosine(feat_i, feat_j)
                
                if similarity > similarity_threshold:
                    removed_indices.add(j)
        
        deduplicated = [candidates[i] for i in sorted(kept_indices)]
        logger.info(f"结构相似度去重结果: {len(deduplicated)} 个样本")
        
        return deduplicated
    
    def sequence_identity_deduplication(self, candidates: List[Dict],
                                       identity_threshold: float = 0.3) -> List[Dict]:
        """
        基于序列一致性的去重 (模拟 CD-HIT)
        
        Args:
            candidates: 候选负样本列表
            identity_threshold: 序列一致性阈值 (默认 0.3 即 30%)
            
        Returns:
            去重后的负样本列表
        """
        logger.info(f"执行序列一致性去重 (Threshold: {identity_threshold})，候选样本数: {len(candidates)}")
        
        if len(candidates) == 0:
            return candidates
            
        # 获取序列
        # 注意: 这里假设 candidate 中没有直接的 sequence 字段，需要从 ESM 特征文件或其他地方获取
        # 如果 unified_data_processor 在生成 candidate 时没有包含 sequence，我们需要想办法获取
        # 暂时假设我们无法直接获取序列文本，只能依赖 feature。
        # 但 CD-HIT 是基于序列文本的。
        # 我们尝试从 protein_sequences 中获取 (如果有引用) 或者警告
        # 鉴于 NegativeSampleEnhancer 初始化时没有传入 sequence map，我们可能需要修改 init 或 assume features
        
        # 修正: 我们可以尝试从 candidate 字典中获取 'sequence' 字段
        # 如果不存在，尝试用 uniprot_id 和 position 从外部获取 (这需要 enhancer 有访问权限)
        # 现有的 NegativeSampleEnhancer 只有 esm_features_path
        
        # 作为一个 fallback，如果无法获取序列文本，我们只能跳过或报错
        # 但通常 candidate 应该包含 sequence 上下文? 
        # 查看 safe_negative_sample_generator.py 的 _generate_protein_set_data 
        # 它似乎只生成了 uniprot_id, position, label 等
        
        # 让我们先检查一下 candidate 的结构，如果没序列，这个功能无法实现
        # 无论如何，先加上这个方法的骨架
        
        # 为了演示，我们假设 candidate 中有 'window_sequence' 字段
        # 如果没有，我们需要在 generator 中添加
        
        kept_candidates = []
        # 简单的贪心去重
        # 对 21-mer，直接比较字符匹配数
        
        # 优化: 仅当 candidate 中有 sequence 信息时执行
        if 'window_sequence' not in candidates[0]:
            logger.warning("候选样本中缺少 'window_sequence' 字段，无法进行序列一致性去重")
            return candidates
            
        for candidate in candidates:
            seq1 = candidate['window_sequence']
            is_redundant = False
            
            for kept in kept_candidates:
                seq2 = kept['window_sequence']
                
                # 计算一致性
                if len(seq1) != len(seq2):
                    continue
                    
                matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
                identity = matches / len(seq1)
                
                if identity > identity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                kept_candidates.append(candidate)
                
        logger.info(f"序列一致性去重结果: {len(kept_candidates)} 个样本")
        return kept_candidates

    def enhance_negatives(self, candidates: List[Dict],
                         strategy: str = 'combined',
                         hard_mining_threshold: float = 0.7,
                         seq_dedup_threshold: float = 0.9,
                         struct_dedup_threshold: float = 0.85,
                         use_structure_for_mining: bool = False,
                         identity_threshold: float = 0.3) -> List[Dict]:
        """
        综合增强负样本
        
        Args:
            candidates: 候选负样本列表
            strategy: 增强策略
            hard_mining_threshold: Hard Mining相似度阈值
            seq_dedup_threshold: 序列特征相似度去重阈值
            struct_dedup_threshold: 结构特征相似度去重阈值
            use_structure_for_mining: 是否在Hard Mining中使用结构特征
            identity_threshold: 序列一致性去重阈值 (CD-HIT style)
        """
        logger.info(f"开始负样本增强，策略: {strategy}, 初始数量: {len(candidates)}")
        
        result = candidates
        
        # 1. Hard Negative Mining
        if strategy == 'hard_mining' or strategy == 'combined':
            # 注意: Hard Mining 通常是从大量候选者中筛选，这里假设传入的就是候选池
            # 如果是 'combined'，我们可能希望先去重再 mining，或者先 mining 再去重
            # 通常先 mining 得到高质量的，再去重保持多样性
            result = self.hard_negative_mining(result, similarity_threshold=hard_mining_threshold)
            
        # 2. 序列一致性去重 (CD-HIT style, if enabled explicitly or part of combined)
        # 默认 combined 包含特征去重，这里我们添加 identity 去重作为可选项
        if strategy in {'identity_dedup', 'combined', 'high_confidence'}:
             result = self.sequence_identity_deduplication(result, identity_threshold=identity_threshold)

        # 3. 序列特征相似度去重
        if strategy in {'seq_dedup', 'combined', 'high_confidence'}:
            result = self.sequence_similarity_deduplication(result, similarity_threshold=seq_dedup_threshold)
            
        # 4. 结构相似度去重
        if strategy in {'struct_dedup', 'combined', 'high_confidence'}:
            result = self.structure_similarity_deduplication(result, similarity_threshold=struct_dedup_threshold)
            
        logger.info(f"负样本增强完成，最终数量: {len(result)}")
        return result
