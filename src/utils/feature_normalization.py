#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征归一化工具模块。

提供在训练前对预提取的 ESM 特征进行标准化的功能，
确保不同蛋白质之间的特征尺度一致，便于模型稳定训练。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class _RunningStats:
    """按特征维度累积均值和方差的工具。"""

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: Optional[np.ndarray] = None
        self.sumsq: Optional[np.ndarray] = None

    def update(self, batch: np.ndarray) -> None:
        """使用新的样本批次更新统计量。

        Args:
            batch: 形状为 (N, D) 的二维数组。
        """
        if batch.size == 0:
            return

        if batch.ndim != 2:
            raise ValueError(f"期望二维数组，收到形状 {batch.shape}")

        # 使用 float32 输入以减少临时内存开销，同时在求和时保留 float64 精度
        batch = np.asarray(batch, dtype=np.float32)
        batch_sum = np.add.reduce(batch, axis=0, dtype=np.float64)
        batch_sumsq = np.einsum("ij,ij->j", batch, batch, dtype=np.float64)

        if self.sum is None:
            self.sum = batch_sum
            self.sumsq = batch_sumsq
            self.count = batch.shape[0]
        else:
            self.sum += batch_sum
            self.sumsq += batch_sumsq
            self.count += batch.shape[0]

    def finalize(self, epsilon: float) -> Optional[Dict[str, np.ndarray]]:
        if self.count == 0 or self.sum is None or self.sumsq is None:
            return None

        mean = self.sum / max(self.count, 1)

        if self.count > 1:
            variance = (self.sumsq - (np.square(self.sum) / self.count)) / (self.count - 1)
        else:
            variance = np.zeros_like(mean)

        variance = np.clip(variance, epsilon ** 2, None)
        std = np.sqrt(variance)
        std = np.where(std < epsilon, 1.0, std)

        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": np.array([self.count], dtype=np.int64),
        }


class FeatureNormalizer:
    """负责计算并应用特征归一化的工具类。"""

    TARGET_KEYS: List[str] = [
        "sequence_features",
        "local_features",
        "global_features",
        "structure_features",
        "structure_local",
        "structure_global",
    ]

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._aggregators: Optional[Dict[str, _RunningStats]] = None

    def fit(self, samples: Iterable[Dict]) -> None:
        """基于训练样本估计均值和标准差。"""
        aggregators = {key: _RunningStats() for key in self.TARGET_KEYS}
        total_samples = 0

        for sample in samples:
            total_samples += 1
            for key, aggregator in aggregators.items():
                value = sample.get(key)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                elif array.ndim != 2:
                    raise ValueError(f"{key} 期望 1D 或 2D 数组，收到形状 {array.shape}")
                aggregator.update(array)

        if total_samples == 0:
            logger.warning("fit 时未收到任何样本，跳过特征归一化参数计算")
            self.stats = {}
            return

        self.stats = {}
        for key, aggregator in aggregators.items():
            result = aggregator.finalize(self.epsilon)
            if result is not None:
                self.stats[key] = result

        if not self.stats:
            logger.warning("未能计算任何特征的归一化参数。")
        else:
            logger.info("已计算以下特征的归一化参数: %s", ", ".join(self.stats.keys()))

    def transform(self, samples: List[Dict], inplace: bool = True) -> List[Dict]:
        """对样本列表中的特征进行归一化。"""
        if not self.stats:
            logger.debug("未检测到归一化参数，跳过 transform")
            return samples

        target_samples = samples if inplace else [dict(sample) for sample in samples]

        for sample in target_samples:
            for key, stat in self.stats.items():
                if key not in sample:
                    continue
                value = sample[key]
                if value is None:
                    continue

                array = np.asarray(value, dtype=np.float32)
                mean = stat["mean"].astype(np.float32)
                std = stat["std"].astype(np.float32)

                if array.ndim == 1:
                    normalized = (array - mean) / std
                elif array.ndim == 2:
                    normalized = (array - mean) / std
                else:
                    raise ValueError(f"{key} 期望 1D 或 2D 数组，收到形状 {array.shape}")

                sample[key] = normalized.astype(np.float32)

        return target_samples

    def transform_stream(self, samples: Iterable[Dict], inplace: bool = True) -> Iterable[Dict]:
        """以生成器形式归一化样本序列，避免一次性加载到内存。

        Args:
            samples: 任意可迭代的样本序列。
            inplace: 是否直接修改原始样本。

        Yields:
            逐个归一化后的样本。
        """
        if not self.stats:
            yield from samples
            return

        for sample in samples:
            target = sample if inplace else dict(sample)
            for key, stat in self.stats.items():
                value = target.get(key)
                if value is None:
                    continue
                array = np.asarray(value, dtype=np.float32)
                mean = stat["mean"].astype(np.float32)
                std = stat["std"].astype(np.float32)
                if array.ndim == 1:
                    normalized = (array - mean) / std
                elif array.ndim == 2:
                    normalized = (array - mean) / std
                else:
                    raise ValueError(f"{key} 期望 1D 或 2D 数组，收到形状 {array.shape}")
                target[key] = normalized.astype(np.float32)
            yield target

    def fit_transform(self, samples: List[Dict]) -> List[Dict]:
        """便捷方法：先拟合后原地归一化。"""
        self.fit(samples)
        return self.transform(samples)

    def fit_incremental(self, samples: Iterable[Dict]) -> None:
        """增量方式更新统计量，适用于流式处理大数据集。
        
        调用此方法多次以累积统计量，最后调用 finalize_stats() 完成计算。
        """
        if self._aggregators is None:
            self._aggregators = {key: _RunningStats() for key in self.TARGET_KEYS}
        
        for sample in samples:
            for key, aggregator in self._aggregators.items():
                value = sample.get(key)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                elif array.ndim != 2:
                    logger.warning(f"{key} 期望 1D 或 2D 数组，收到形状 {array.shape}，跳过")
                    continue
                aggregator.update(array)

    def finalize_stats(self) -> None:
        """完成增量统计量的计算并生成最终的归一化参数。
        
        必须在调用 fit_incremental() 之后调用此方法。
        """
        if self._aggregators is None:
            logger.warning("finalize_stats 被调用但未进行任何增量更新")
            self.stats = {}
            return
        
        self.stats = {}
        for key, aggregator in self._aggregators.items():
            result = aggregator.finalize(self.epsilon)
            if result is not None:
                self.stats[key] = result
        
        if not self.stats:
            logger.warning("未能计算任何特征的归一化参数。")
        else:
            logger.info("增量统计完成，已计算以下特征的归一化参数: %s", ", ".join(self.stats.keys()))
        
        # 清理聚合器释放内存
        self._aggregators = None

    def save(self, path: Path) -> None:
        """将归一化参数保存到磁盘。"""
        if not self.stats:
            logger.warning("没有可保存的归一化参数")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}
        metadata = {"counts": {}, "epsilon": self.epsilon}
        for key, stat in self.stats.items():
            arrays[f"{key}_mean"] = stat["mean"]
            arrays[f"{key}_std"] = stat["std"]
            metadata["counts"][key] = int(stat["count"][0])

        np.savez_compressed(path, **arrays)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("归一化参数已保存到 %s", path)

    def load(self, path: Path) -> None:
        """从磁盘加载归一化参数。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"归一化参数文件不存在: {path}")

        data = np.load(path)
        stats: Dict[str, Dict[str, np.ndarray]] = {}

        for key in self.TARGET_KEYS:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in data and std_key in data:
                stats[key] = {
                    "mean": data[mean_key].astype(np.float32),
                    "std": np.maximum(data[std_key].astype(np.float32), self.epsilon),
                    "count": np.array([0], dtype=np.int64),
                }

        if not stats:
            raise ValueError(f"在 {path} 中未找到任何归一化系数")

        self.stats = stats

        logger.info("已从 %s 加载归一化参数", path)

    def get_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """获取当前的统计参数副本。"""
        return {key: {k: v.copy() for k, v in stat.items()} for key, stat in self.stats.items()}
