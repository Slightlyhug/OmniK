#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 安全的负样本生成器
防止信息泄露的负样本生成策略 - 优化内存和IO性能
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import h5py
import math
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from functools import lru_cache
import gc

# 导入负样本增强模块
try:
    from negative_sample_enhancement import NegativeSampleEnhancer
except ImportError:
    NegativeSampleEnhancer = None

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('safe_negative_generator')


class FeatureCache:
    """特征缓存管理器 - 减少重复IO操作"""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.max_cache_size = max_cache_size
        self.access_order: List[str] = []
    
    def get(self, protein_id: str, feature_type: str = 'sequence_features') -> Optional[np.ndarray]:
        """从缓存获取特征"""
        if protein_id in self.cache and feature_type in self.cache[protein_id]:
            # 更新访问顺序
            if protein_id in self.access_order:
                self.access_order.remove(protein_id)
            self.access_order.append(protein_id)
            return self.cache[protein_id][feature_type]
        return None
    
    def set(self, protein_id: str, feature_type: str, features: np.ndarray):
        """设置缓存"""
        if protein_id not in self.cache:
            self.cache[protein_id] = {}
        
        self.cache[protein_id][feature_type] = features
        
        # 更新访问顺序
        if protein_id in self.access_order:
            self.access_order.remove(protein_id)
        self.access_order.append(protein_id)
        
        # 如果超过缓存大小,移除最久未使用的
        if len(self.cache) > self.max_cache_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
        gc.collect()


class SafeNegativeSampleGenerator:
    """安全的负样本生成器 - 防止信息泄露,降低资源消耗"""

    AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, positive_data_path: str, esm_features_path: Optional[str] = None,
                 target_residues: Optional[Set[str]] = None,
                 negative_ptm_type: str = 'non_phosphorylation',
                 non_target_residues: Optional[Set[str]] = None,
                 non_target_ratio: float = 0.0,
                 enable_similarity_analysis: bool = False,
                 max_negatives_for_analysis: int = 10000,
                 feature_cache_size: int = 100,
                 sequence_path: Optional[str] = None,
                 window_size: int = 61):
        """
        初始化安全负样本生成器
        
        Args:
            positive_data_path: 正样本数据路径
            esm_features_path: ESM特征文件路径 (可选)
            target_residues: 目标残基类型；None 时使用默认 {'S', 'T', 'Y'}
            negative_ptm_type: 为负样本写入的 PTM 类型标记
            enable_similarity_analysis: 是否启用相似度分析(会消耗大量资源)
            max_negatives_for_analysis: 相似度分析时使用的最大负样本数
            feature_cache_size: 特征缓存大小
            sequence_path: 序列文件路径 (目录或文件)，如果提供则优先从中加载序列
            window_size: 序列窗口大小
        """
        self.positive_data_path = positive_data_path
        self.esm_features_path = esm_features_path
        self.target_residues = set(res.upper() for res in (target_residues or {'S', 'T', 'Y'}))
        self.negative_ptm_type = negative_ptm_type
        self.non_target_ratio = max(non_target_ratio, 0.0)
        self.enable_similarity_analysis = enable_similarity_analysis
        self.max_negatives_for_analysis = max_negatives_for_analysis
        self.sequence_path = sequence_path
        self.window_size = window_size

        if non_target_residues is not None:
            self.non_target_residues = set(res.upper() for res in non_target_residues if res)
        else:
            self.non_target_residues = self.AMINO_ACIDS - self.target_residues

        if self.non_target_ratio > 0 and not self.non_target_residues:
            logger.warning("指定了非目标残基比例，但非目标残基集合为空，将回退为仅使用目标残基")
            self.non_target_ratio = 0.0
        
        # 初始化特征缓存
        self.feature_cache = FeatureCache(max_cache_size=feature_cache_size)
        
        # 加载正样本数据
        self.positive_data = pd.read_csv(positive_data_path)
        logger.info(f"加载正样本数据: {len(self.positive_data)} 个样本")
        
        # 创建正样本位点集合
        self.positive_sites = set()
        self.protein_positive_sites = defaultdict(set)
        
        for _, row in self.positive_data.iterrows():
            site = (row['uniprot_id'], row['position'])
            self.positive_sites.add(site)
            self.protein_positive_sites[row['uniprot_id']].add(row['position'])
        
        # 加载蛋白质序列信息
        self.protein_sequences = self._load_protein_sequences()

        # 校验正样本位点与序列的一致性
        self.alignment_report = self._validate_positive_sites_against_sequences()

        # 初始化负样本增强器
        self.enhancer = None
        if NegativeSampleEnhancer is not None:
            try:
                self.enhancer = NegativeSampleEnhancer(esm_features_path, self.positive_data)
                logger.info("负样本增强器初始化成功")
            except Exception as e:
                logger.warning(f"负样本增强器初始化失败: {e}")

        logger.info(f"正样本位点数量: {len(self.positive_sites)}")
        logger.info(f"涉及蛋白质数量: {len(self.protein_positive_sites)}")
        logger.info(f"相似度分析: {'启用' if self.enable_similarity_analysis else '禁用'}")
    
    def _load_protein_sequences(self) -> Dict[str, str]:
        """加载蛋白质序列信息"""
        sequences = {}
        
        # 1. 优先从 sequence_path 加载 (FASTA)
        if self.sequence_path:
            logger.info(f"从 FASTA 加载序列: {self.sequence_path}")
            try:
                import glob
                from Bio import SeqIO
                
                path = Path(self.sequence_path)
                fasta_files = []
                
                if path.is_file():
                    fasta_files = [path]
                elif path.is_dir():
                    fasta_files = list(path.glob("**/*.fasta")) + list(path.glob("**/*.fa"))
                
                logger.info(f"发现 {len(fasta_files)} 个 FASTA 文件")
                
                for fasta_file in fasta_files:
                    try:
                        for record in SeqIO.parse(fasta_file, "fasta"):
                            # 提取 UniProt ID (通常是 ID 的一部分，或者直接是 ID)
                            # 假设 ID 格式为 "sp|P12345|..." 或直接 "P12345"
                            # 这里简单使用 record.id，如果需要解析 sp|ID|Name 格式需要额外处理
                            pid = record.id.split('|')[1] if '|' in record.id else record.id
                            sequences[pid] = str(record.seq)
                    except Exception as e:
                        logger.warning(f"读取 FASTA 文件 {fasta_file} 失败: {e}")
                        
            except ImportError:
                logger.error("缺少 Biopython，无法加载 FASTA 文件。请安装 biopython")
            except Exception as e:
                logger.error(f"加载 FASTA 序列失败: {e}")

        # 2. 如果 ESM 特征文件存在，从中补充
        if self.esm_features_path and os.path.exists(self.esm_features_path):
            try:
                with h5py.File(self.esm_features_path, 'r') as f:
                    if 'proteins' in f:
                        proteins_group = f['proteins']
                        for protein_id in proteins_group.keys():
                            if protein_id not in sequences: # 仅补充不存在的
                                protein_group = proteins_group[protein_id]
                                if 'sequence' in protein_group.attrs:
                                    sequence = protein_group.attrs['sequence']
                                    if isinstance(sequence, bytes):
                                        sequences[protein_id] = sequence.decode('utf-8')
                                    else:
                                        sequences[protein_id] = str(sequence)
                                elif 'sequence' in protein_group:
                                    sequence = protein_group['sequence'][()]
                                    if isinstance(sequence, bytes):
                                        sequences[protein_id] = sequence.decode('utf-8')
                                    else:
                                        sequences[protein_id] = str(sequence)
            except Exception as e:
                logger.error(f"从 ESM H5 加载序列失败: {e}")

        logger.info(f"总共加载了 {len(sequences)} 个蛋白质序列")
        return sequences

    def _validate_positive_sites_against_sequences(self) -> Dict[str, object]:
        """校验正样本位点与 H5 序列的一致性"""
        missing_proteins: Set[str] = set()
        out_of_range: List[Tuple[str, int, int]] = []
        mismatched_residues: List[Tuple[str, int, str, str]] = []

        for row in self.positive_data.itertuples(index=False):
            protein_id = getattr(row, 'uniprot_id')
            position = int(getattr(row, 'position'))
            expected_residue = getattr(row, 'residue')

            sequence = self.protein_sequences.get(protein_id)
            if not sequence:
                missing_proteins.add(protein_id)
                continue

            if position <= 0 or position > len(sequence):
                out_of_range.append((protein_id, position, len(sequence)))
                continue

            actual_residue = sequence[position - 1]
            if actual_residue != expected_residue:
                mismatched_residues.append((protein_id, position, expected_residue, actual_residue))

        report = {
            'missing_protein_count': len(missing_proteins),
            'missing_proteins': sorted(missing_proteins)[:50],
            'out_of_range_count': len(out_of_range),
            'out_of_range_examples': out_of_range[:20],
            'mismatched_count': len(mismatched_residues),
            'mismatched_examples': mismatched_residues[:20]
        }

        if report['missing_protein_count'] or report['out_of_range_count'] or report['mismatched_count']:
            logger.warning(
                "正样本与 H5 序列存在潜在错位: 缺失蛋白 %d, 越界位点 %d, 残基不一致 %d",
                report['missing_protein_count'],
                report['out_of_range_count'],
                report['mismatched_count']
            )
        else:
            logger.info("正样本位点与 H5 序列校验通过，未发现错位")

        return report

    def _get_site_feature_cached(self, protein_id: str, position: int) -> Optional[np.ndarray]:
        """使用缓存获取位点特征"""
        # 先尝试从缓存获取
        cached_feature = self.feature_cache.get(protein_id, 'sequence_features')
        if cached_feature is not None:
            idx = int(position) - 1
            if 0 <= idx < len(cached_feature):
                return cached_feature[idx]
            return None
        
        # 缓存未命中,从H5文件加载
        try:
            with h5py.File(self.esm_features_path, 'r') as f:
                if 'proteins' not in f or protein_id not in f['proteins']:
                    return None
                
                protein_group = f['proteins'][protein_id]
                if 'sequence_features' not in protein_group:
                    return None
                
                # 加载完整特征并缓存
                seq_features = protein_group['sequence_features'][()]
                self.feature_cache.set(protein_id, 'sequence_features', seq_features)
                
                # 返回指定位点特征
                idx = int(position) - 1
                if 0 <= idx < len(seq_features):
                    return seq_features[idx]
                return None
        except Exception as e:
            logger.error(f"读取蛋白质 {protein_id} 特征失败: {e}")
            return None

    def _collect_vectors_batched_optimized(self, df: pd.DataFrame, batch_size: int = 100) -> List[np.ndarray]:
        """优化的批量收集特征向量,使用缓存减少IO"""
        vectors = []
        if df.empty:
            return vectors
            
        # 按蛋白质分组
        grouped = df.groupby('uniprot_id')
        
        for protein_id, protein_df in grouped:
            positions = protein_df['position'].values
            
            for pos in positions:
                feature = self._get_site_feature_cached(protein_id, int(pos))
                if feature is not None:
                    vectors.append(feature)
        
        return vectors

    def analyze_similarity_distribution(self, data_df: pd.DataFrame, dataset_name: str = "Data"):
        """
        分析正负样本的相似度分布
        计算负样本与正样本中心（基于位点特征）的相似度
        """
        if not self.enable_similarity_analysis:
            logger.info(f"相似度分析已禁用,跳过 {dataset_name} 的分析")
            return
        
        logger.info(f"正在分析 {dataset_name} 的相似度分布...")
        
        positives = data_df[data_df['label'] == 1]
        negatives = data_df[data_df['label'] == 0]
        
        if len(positives) == 0 or len(negatives) == 0:
            logger.warning(f"{dataset_name} 缺少正样本或负样本，无法进行相似度分析")
            return

        # 1. 计算正样本中心
        logger.info("正在计算正样本中心...")
        pos_vectors = self._collect_vectors_batched_optimized(positives, batch_size=100)
        
        if not pos_vectors:
            logger.warning("无法提取正样本特征，跳过相似度分析")
            return
            
        pos_matrix = np.stack(pos_vectors)
        pos_norm = np.linalg.norm(pos_matrix, axis=1, keepdims=True)
        pos_matrix = pos_matrix / (pos_norm + 1e-8)
        
        pos_centroid = np.mean(pos_matrix, axis=0)
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-8)
        
        # 2. 计算负样本相似度 - 抽样以减少内存消耗
        if len(negatives) > self.max_negatives_for_analysis:
            negatives = negatives.sample(n=self.max_negatives_for_analysis, random_state=42)
            logger.info(f"负样本过多，抽样 {self.max_negatives_for_analysis} 条进行分布分析")
            
        logger.info("正在收集负样本特征...")
        neg_vectors = self._collect_vectors_batched_optimized(negatives, batch_size=100)
        
        if not neg_vectors:
            logger.warning("无法提取负样本特征")
            return
            
        neg_matrix = np.stack(neg_vectors)
        neg_norm = np.linalg.norm(neg_matrix, axis=1, keepdims=True)
        neg_matrix = neg_matrix / (neg_norm + 1e-8)
        
        # 批量计算相似度
        similarities = np.dot(neg_matrix, pos_centroid)
        
        if len(similarities) > 0:
            logger.info(f"=== {dataset_name} 负样本相似度分布统计 (相对于正样本中心) ===")
            logger.info(f"样本数: {len(similarities)}")
            logger.info(f"均值: {np.mean(similarities):.4f}")
            logger.info(f"中位数: {np.median(similarities):.4f}")
            logger.info(f"25% 分位数: {np.percentile(similarities, 25):.4f}")
            logger.info(f"75% 分位数: {np.percentile(similarities, 75):.4f}")
            logger.info(f"最大值: {np.max(similarities):.4f}")
            logger.info(f"最小值: {np.min(similarities):.4f}")
            logger.info("==========================================================")
        
        # 清理内存
        del pos_matrix, neg_matrix, similarities
        gc.collect()
    
    def generate_safe_negatives(self,
                               strategy: str = "protein_split",
                               negative_ratio: float = 1.0,
                               min_distance: int = 10,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               enhancement_strategy: Optional[str] = None,
                               hard_mining_threshold: float = 0.7,
                               seq_dedup_threshold: float = 0.9,
                               struct_dedup_threshold: float = 0.85,
                               use_structure_for_mining: bool = False,
                               identity_dedup_threshold: float = 0.3) -> Dict[str, pd.DataFrame]:
        """
        生成安全的负样本，防止信息泄露

        Args:
            strategy: 生成策略
                - "protein_split": 蛋白质级别分割
                - "cross_species": 跨物种负样本
                - "temporal_split": 时间分割
                - "conservative": 保守策略
            negative_ratio: 负样本比例
            min_distance: 最小距离
            test_size: 测试集比例
            random_state: 随机种子
            enhancement_strategy: 负样本增强策略
                - None: 不增强
                - "hard_mining": Hard Negative Mining
                - "seq_dedup": 序列相似度去重
                - "struct_dedup": 结构相似度去重
                - "identity_dedup": 序列一致性去重 (CD-HIT style)
                - "combined": 综合增强
            hard_mining_threshold: Hard Mining相似度阈值
            seq_dedup_threshold: 序列去重相似度阈值
            struct_dedup_threshold: 结构去重相似度阈值
            use_structure_for_mining: 是否在Hard Mining中使用结构特征
            identity_dedup_threshold: 序列一致性去重阈值

        Returns:
            包含训练和测试数据的字典
        """
        logger.info(f"使用安全策略生成负样本: {strategy}")

        if strategy == "protein_split":
            result = self._protein_split_strategy(negative_ratio, min_distance, test_size, random_state)
        elif strategy == "cross_species":
            result = self._cross_species_strategy(negative_ratio, min_distance, test_size, random_state)
        elif strategy == "temporal_split":
            result = self._temporal_split_strategy(negative_ratio, min_distance, test_size, random_state)
        elif strategy == "conservative":
            result = self._conservative_strategy(negative_ratio, min_distance, test_size, random_state)
        else:
            raise ValueError(f"未知的安全策略: {strategy}")

        # 初始生成结果分析
        if 'train' in result:
            self.analyze_similarity_distribution(result['train'], "初始训练集")

        # 应用负样本增强
        if enhancement_strategy and self.enhancer:
            result = self._apply_enhancement(
                result, enhancement_strategy,
                hard_mining_threshold, seq_dedup_threshold, struct_dedup_threshold,
                use_structure_for_mining, identity_dedup_threshold
            )
            
            # 增强后分析
            if 'train' in result:
                self.analyze_similarity_distribution(result['train'], "增强后训练集")

        # 清理缓存
        self.feature_cache.clear()
        
        return result
    
    def _apply_enhancement(self, result: Dict, enhancement_strategy: str,
                          hard_mining_threshold: float, seq_dedup_threshold: float,
                          struct_dedup_threshold: float,
                          use_structure_for_mining: bool = False,
                          identity_dedup_threshold: float = 0.3) -> Dict:
        """应用负样本增强策略"""
        logger.info(f"应用负样本增强策略: {enhancement_strategy}")

        if 'train' in result:
            train_negatives = result['train'][result['train']['label'] == 0].to_dict('records')
            enhanced_negatives = self.enhancer.enhance_negatives(
                train_negatives,
                strategy=enhancement_strategy,
                hard_mining_threshold=hard_mining_threshold,
                seq_dedup_threshold=seq_dedup_threshold,
                struct_dedup_threshold=struct_dedup_threshold,
                use_structure_for_mining=use_structure_for_mining,
                identity_threshold=identity_dedup_threshold
            )

            # 重建训练集
            train_positives = result['train'][result['train']['label'] == 1]
            enhanced_negatives_df = pd.DataFrame(enhanced_negatives)
            result['train'] = pd.concat([train_positives, enhanced_negatives_df], ignore_index=True)
            result['train'] = result['train'].sample(frac=1, random_state=42).reset_index(drop=True)

            logger.info(f"训练集增强完成: {len(result['train'])} 个样本")

        if 'test' in result:
            test_negatives = result['test'][result['test']['label'] == 0].to_dict('records')
            enhanced_negatives = self.enhancer.enhance_negatives(
                test_negatives,
                strategy=enhancement_strategy,
                hard_mining_threshold=hard_mining_threshold,
                seq_dedup_threshold=seq_dedup_threshold,
                struct_dedup_threshold=struct_dedup_threshold,
                use_structure_for_mining=use_structure_for_mining,
                identity_threshold=identity_dedup_threshold
            )

            # 重建测试集
            test_positives = result['test'][result['test']['label'] == 1]
            enhanced_negatives_df = pd.DataFrame(enhanced_negatives)
            result['test'] = pd.concat([test_positives, enhanced_negatives_df], ignore_index=True)
            result['test'] = result['test'].sample(frac=1, random_state=42).reset_index(drop=True)

            logger.info(f"测试集增强完成: {len(result['test'])} 个样本")

        return result

    def _protein_split_strategy(self, negative_ratio: float, min_distance: int,
                               test_size: float, random_state: int) -> Dict[str, pd.DataFrame]:
        """蛋白质级别分割策略 - 最安全的方法"""
        logger.info("使用蛋白质级别分割策略")
        
        # 获取所有有正样本的蛋白质
        proteins_with_positives = list(self.protein_positive_sites.keys())
        
        # 按蛋白质分割
        train_proteins, test_proteins = train_test_split(
            proteins_with_positives, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"训练蛋白质: {len(train_proteins)}, 测试蛋白质: {len(test_proteins)}")
        
        # 分别为训练集和测试集生成数据
        train_data = self._generate_protein_set_data(train_proteins, negative_ratio, min_distance)
        test_data = self._generate_protein_set_data(test_proteins, negative_ratio, min_distance)
        
        return {
            'train': train_data,
            'test': test_data,
            'train_proteins': train_proteins,
            'test_proteins': test_proteins
        }
    
    def _cross_species_strategy(self, negative_ratio: float, min_distance: int,
                               test_size: float, random_state: int) -> Dict[str, pd.DataFrame]:
        """跨物种策略 - 使用不同物种的蛋白质作为负样本源"""
        logger.info("跨物种策略需要物种信息，当前使用蛋白质分割策略")
        return self._protein_split_strategy(negative_ratio, min_distance, test_size, random_state)
    
    def _temporal_split_strategy(self, negative_ratio: float, min_distance: int,
                                test_size: float, random_state: int) -> Dict[str, pd.DataFrame]:
        """时间分割策略 - 基于发现时间分割数据"""
        logger.info("时间分割策略需要时间信息，当前使用蛋白质分割策略")
        return self._protein_split_strategy(negative_ratio, min_distance, test_size, random_state)
    
    def _conservative_strategy(self, negative_ratio: float, min_distance: int,
                              test_size: float, random_state: int) -> Dict[str, pd.DataFrame]:
        """保守策略 - 只使用高置信度的负样本"""
        logger.info("使用保守策略生成负样本")
        
        # 使用更严格的距离限制
        strict_min_distance = max(min_distance, 15)
        proteins_with_positives = list(self.protein_positive_sites.keys())
        
        # 只从没有任何正样本的蛋白质中选择负样本
        proteins_without_positives = []
        for protein_id in self.protein_sequences.keys():
            if protein_id not in self.protein_positive_sites:
                proteins_without_positives.append(protein_id)
        
        logger.info(f"找到 {len(proteins_without_positives)} 个没有正样本的蛋白质")
        
        if len(proteins_without_positives) == 0:
            logger.warning("没有找到无正样本的蛋白质，使用蛋白质分割策略")
            return self._protein_split_strategy(negative_ratio, min_distance, test_size, random_state)

        pos_train_proteins, pos_test_proteins = train_test_split(
            proteins_with_positives,
            test_size=test_size,
            random_state=random_state
        )
        neg_train_proteins, neg_test_proteins = train_test_split(
            proteins_without_positives,
            test_size=test_size,
            random_state=random_state
        )

        train_positives = self.positive_data[
            self.positive_data['uniprot_id'].isin(pos_train_proteins)
        ].copy()
        test_positives = self.positive_data[
            self.positive_data['uniprot_id'].isin(pos_test_proteins)
        ].copy()

        train_negatives = self._generate_negative_pool_from_proteins(
            proteins=neg_train_proteins,
            target_goal=int(round(len(train_positives) * negative_ratio)),
            diverse_goal=int(math.ceil(len(train_positives) * negative_ratio * self.non_target_ratio)) if self.non_target_ratio > 0 else 0,
            min_distance=strict_min_distance,
        )
        test_negatives = self._generate_negative_pool_from_proteins(
            proteins=neg_test_proteins,
            target_goal=int(round(len(test_positives) * negative_ratio)),
            diverse_goal=int(math.ceil(len(test_positives) * negative_ratio * self.non_target_ratio)) if self.non_target_ratio > 0 else 0,
            min_distance=strict_min_distance,
        )

        train_positives['label'] = 1
        test_positives['label'] = 1
        if not train_negatives.empty:
            train_negatives['label'] = 0
        if not test_negatives.empty:
            test_negatives['label'] = 0

        train_data = pd.concat([train_positives, train_negatives], ignore_index=True)
        test_data = pd.concat([test_positives, test_negatives], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        return {
            'train': train_data,
            'test': test_data
        }

    def _generate_negative_pool_from_proteins(
        self,
        proteins: List[str],
        target_goal: int,
        diverse_goal: int,
        min_distance: int,
    ) -> pd.DataFrame:
        """从给定蛋白质集合中构建负样本池，不与正样本蛋白混用。"""
        target_samples: List[Dict] = []
        diverse_samples: List[Dict] = []

        for protein_id in proteins:
            if len(target_samples) >= target_goal and len(diverse_samples) >= diverse_goal:
                break

            sequence = self.protein_sequences.get(protein_id)
            if not sequence:
                continue

            for i, residue in enumerate(sequence):
                position = i + 1
                if len(target_samples) < target_goal and residue in self.target_residues:
                    target_samples.append({
                        'uniprot_id': protein_id,
                        'protein_name': f'NEG_{protein_id}',
                        'position': position,
                        'residue': residue,
                        'ptm_type': self.negative_ptm_type,
                        'window_sequence': self._extract_window(sequence, position)
                    })
                elif (
                    self.non_target_ratio > 0
                    and len(diverse_samples) < diverse_goal
                    and residue in self.non_target_residues
                ):
                    diverse_samples.append({
                        'uniprot_id': protein_id,
                        'protein_name': f'NEG_{protein_id}',
                        'position': position,
                        'residue': residue,
                        'ptm_type': self.negative_ptm_type,
                        'window_sequence': self._extract_window(sequence, position)
                    })

                if len(target_samples) >= target_goal and len(diverse_samples) >= diverse_goal:
                    break

        if len(target_samples) > target_goal:
            target_samples = random.sample(target_samples, target_goal)
        if self.non_target_ratio > 0 and len(diverse_samples) > diverse_goal:
            diverse_samples = random.sample(diverse_samples, diverse_goal)

        return pd.DataFrame(target_samples + diverse_samples)
    
    def _generate_protein_set_data(self, proteins: List[str], negative_ratio: float,
                                  min_distance: int) -> pd.DataFrame:
        """为指定的蛋白质集合生成正负样本数据"""
        # 获取这些蛋白质的正样本
        protein_positives = self.positive_data[
            self.positive_data['uniprot_id'].isin(proteins)
        ].copy()

        target_negative_goal = int(round(len(protein_positives) * negative_ratio))

        target_negative_samples: List[Dict] = []
        diverse_negative_samples: List[Dict] = []
        used_sites: Set[Tuple[str, int]] = set()

        for protein_id in proteins:
            if protein_id not in self.protein_sequences:
                continue

            protein_positive_count = len(protein_positives[protein_positives['uniprot_id'] == protein_id])
            protein_target_goal = int(round(protein_positive_count * negative_ratio))
            protein_diverse_goal = 0
            if self.non_target_ratio > 0 and protein_target_goal > 0:
                protein_diverse_goal = int(math.ceil(protein_target_goal * self.non_target_ratio))

            negatives = self._generate_protein_negatives(
                protein_id,
                protein_target_goal,
                protein_diverse_goal,
                min_distance,
                used_sites
            )
            target_negative_samples.extend(negatives['target'])
            diverse_negative_samples.extend(negatives['diverse'])

        if len(target_negative_samples) < target_negative_goal:
            additional_needed = target_negative_goal - len(target_negative_samples)
            additional_negatives = self._generate_additional_negatives(
                proteins,
                additional_needed,
                0,
                min_distance,
                used_sites
            )
            target_negative_samples.extend(additional_negatives['target'])
            diverse_negative_samples.extend(additional_negatives['diverse'])

        if self.non_target_ratio > 0 and len(target_negative_samples) > 0:
            desired_diverse_total = int(math.ceil(len(target_negative_samples) * self.non_target_ratio))
            if len(diverse_negative_samples) < desired_diverse_total:
                diverse_needed = desired_diverse_total - len(diverse_negative_samples)
                additional_diverse = self._generate_additional_negatives(
                    proteins,
                    0,
                    diverse_needed,
                    min_distance,
                    used_sites
                )
                target_negative_samples.extend(additional_diverse['target'])
                diverse_negative_samples.extend(additional_diverse['diverse'])

        negative_samples = target_negative_samples + diverse_negative_samples

        # 转换为DataFrame
        negative_df = pd.DataFrame(negative_samples)
        
        # 添加标签
        protein_positives['label'] = 1
        if len(negative_df) > 0:
            negative_df['label'] = 0
            
            # 合并数据
            combined_data = pd.concat([protein_positives, negative_df], ignore_index=True)
        else:
            combined_data = protein_positives
        
        # 打乱数据
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return combined_data
    
    def _extract_window(self, sequence: str, position: int, window_size: Optional[int] = None) -> str:
        """提取序列窗口"""
        if window_size is None:
            window_size = self.window_size
            
        half_window = window_size // 2
        start = position - 1 - half_window
        end = position - 1 + half_window + 1
        
        # 处理边界 padding
        padding_left = 0
        padding_right = 0
        
        if start < 0:
            padding_left = abs(start)
            start = 0
            
        if end > len(sequence):
            padding_right = end - len(sequence)
            end = len(sequence)
            
        subseq = sequence[start:end]
        return "_" * padding_left + subseq + "_" * padding_right

    def _generate_protein_negatives(self, protein_id: str, target_count: int,
                                   diverse_count: int, min_distance: int,
                                   used_sites: Set[Tuple[str, int]]) -> Dict[str, List[Dict]]:
        """为特定蛋白质生成负样本"""
        if protein_id not in self.protein_sequences:
            return {'target': [], 'diverse': []}

        sequence = self.protein_sequences[protein_id]
        positive_positions = self.protein_positive_sites[protein_id]

        target_candidates: List[Dict] = []
        diverse_candidates: List[Dict] = []

        for i, residue in enumerate(sequence):
            position = i + 1

            if position in positive_positions:
                continue

            if not self._check_distance_from_positives(position, positive_positions, min_distance):
                continue

            site = (protein_id, position)
            if site in used_sites:
                continue
            
            # 提取窗口序列
            window_seq = self._extract_window(sequence, position)

            if residue in self.target_residues and target_count > 0:
                target_candidates.append({
                    'uniprot_id': protein_id,
                    'protein_name': f'NEG_{protein_id}',
                    'position': position,
                    'residue': residue,
                    'ptm_type': self.negative_ptm_type,
                    'window_sequence': window_seq
                })
            elif (self.non_target_ratio > 0 and diverse_count > 0
                  and residue in self.non_target_residues):
                diverse_candidates.append({
                    'uniprot_id': protein_id,
                    'protein_name': f'NEG_{protein_id}',
                    'position': position,
                    'residue': residue,
                    'ptm_type': self.negative_ptm_type,
                    'window_sequence': window_seq
                })

        selected_target: List[Dict] = []
        if target_count > 0 and target_candidates:
            target_sample_size = min(target_count, len(target_candidates))
            selected_target = random.sample(target_candidates, target_sample_size)

        selected_diverse: List[Dict] = []
        if diverse_count > 0 and diverse_candidates:
            diverse_sample_size = min(diverse_count, len(diverse_candidates))
            selected_diverse = random.sample(diverse_candidates, diverse_sample_size)

        for item in selected_target + selected_diverse:
            used_sites.add((item['uniprot_id'], item['position']))

        return {'target': selected_target, 'diverse': selected_diverse}
    
    def _generate_additional_negatives(self, proteins: List[str], needed_target_count: int,
                                     needed_diverse_count: int, min_distance: int,
                                     used_sites: Set[Tuple[str, int]]) -> Dict[str, List[Dict]]:
        """生成额外的负样本"""
        additional_target: List[Dict] = []
        additional_diverse: List[Dict] = []

        target_remaining = max(needed_target_count, 0)
        diverse_remaining = max(needed_diverse_count, 0)

        for protein_id in proteins:
            if target_remaining <= 0 and diverse_remaining <= 0:
                break

            generated = self._generate_protein_negatives(
                protein_id,
                target_remaining,
                diverse_remaining,
                min_distance,
                used_sites
            )

            if generated['target']:
                additional_target.extend(generated['target'])
                target_remaining = max(0, target_remaining - len(generated['target']))

            if generated['diverse']:
                additional_diverse.extend(generated['diverse'])
                diverse_remaining = max(0, diverse_remaining - len(generated['diverse']))

        return {
            'target': additional_target[:needed_target_count] if needed_target_count > 0 else [],
            'diverse': additional_diverse[:needed_diverse_count] if needed_diverse_count > 0 else []
        }
    
    def _check_distance_from_positives(self, position: int, positive_positions: Set[int], 
                                     min_distance: int) -> bool:
        """检查位点与正样本的距离"""
        for pos_position in positive_positions:
            if abs(position - pos_position) < min_distance:
                return False
        return True
    
    def evaluate_leakage_risk(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """评估信息泄露风险"""
        logger.info("评估信息泄露风险...")
        
        train_proteins = set(train_data['uniprot_id'].unique())
        test_proteins = set(test_data['uniprot_id'].unique())
        
        # 蛋白质重叠
        protein_overlap = len(train_proteins & test_proteins)
        protein_overlap_ratio = protein_overlap / len(test_proteins) if test_proteins else 0
        
        # 位点重叠（不应该有）
        train_sites = set(zip(train_data['uniprot_id'], train_data['position']))
        test_sites = set(zip(test_data['uniprot_id'], test_data['position']))
        site_overlap = len(train_sites & test_sites)
        
        risk_assessment = {
            'protein_overlap_count': protein_overlap,
            'protein_overlap_ratio': protein_overlap_ratio,
            'site_overlap_count': site_overlap,
            'train_proteins': len(train_proteins),
            'test_proteins': len(test_proteins),
            'risk_level': 'LOW' if protein_overlap_ratio == 0 else 'HIGH'
        }
        
        logger.info(f"泄露风险评估: {risk_assessment}")
        return risk_assessment


def main():
    """测试优化的安全负样本生成器"""
    generator = SafeNegativeSampleGenerator(
        positive_data_path="phosphorylation_data_small_batch/acetylation_data_3000.csv",
        esm_features_path="phosphorylation_data_small_batch/features/esm_features.h5",
        enable_similarity_analysis=False  # 禁用相似度分析以节省资源
    )
    
    # 使用蛋白质分割策略
    result = generator.generate_safe_negatives(
        strategy="protein_split",
        negative_ratio=1.0,
        min_distance=10,
        test_size=0.2
    )
    
    # 评估泄露风险
    risk = generator.evaluate_leakage_risk(result['train'], result['test'])
    
    print(f"✅ 优化的安全负样本生成完成!")
    print(f"训练集: {len(result['train'])} 样本")
    print(f"测试集: {len(result['test'])} 样本")
    print(f"泄露风险等级: {risk['risk_level']}")


if __name__ == "__main__":
    main()
