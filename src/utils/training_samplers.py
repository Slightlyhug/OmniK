"""Batch sampling, label distribution monitoring and early stopping."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

from src.utils.training_config import sanitize_name


def _positive_label_matrix(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.float32, copy=False)
    return (labels > 0.0).astype(np.float32, copy=False)


def _known_label_matrix(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.float32, copy=False)
    return (labels >= 0.0).astype(np.float32, copy=False)


def _positive_label_distribution(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        width = int(labels.shape[1]) if labels.ndim == 2 else 0
        return np.zeros((width,), dtype=np.float32)
    return _positive_label_matrix(labels).mean(axis=0).astype(np.float32, copy=False)


def compute_multilabel_sample_weights(
    dataset: MultiLabelPTMJsonDataset,
    alpha: float = 0.5,
    all_zero_weight: float = 0.3,
    min_pos_freq: float = 1e-4,
    min_weight: float = 1e-3,
) -> torch.DoubleTensor:
    """Build per-sample weights for multi-label weighted sampling.

    Unknown PU labels use -1 and must not count as negative labels when
    estimating positive frequencies or all-zero samples.
    """
    y_raw = dataset.get_label_matrix()
    if y_raw.shape[0] == 0:
        return torch.DoubleTensor([])

    y = _positive_label_matrix(y_raw)
    known = _known_label_matrix(y_raw)
    pos_freq = np.clip(y.mean(axis=0), min_pos_freq, 1.0)
    inv_freq = np.power(1.0 / pos_freq, float(alpha))

    sample_pos = (y * inv_freq.reshape(1, -1)).sum(axis=1)
    all_zero_mask = (y.sum(axis=1) == 0) & (known.sum(axis=1) > 0)
    weights = sample_pos + np.where(all_zero_mask, float(all_zero_weight), 0.0)
    weights = np.clip(weights, float(min_weight), None)
    return torch.as_tensor(weights, dtype=torch.double)


class DistributionAlignedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: MultiLabelPTMJsonDataset,
        batch_size: int,
        alpha: float = 1.0,
        all_zero_weight: float = 1.0,
        label_boost_by_label: Optional[Dict[str, float]] = None,
        min_label_ratio_by_label: Optional[Dict[str, float]] = None,
        max_label_ratio_by_label: Optional[Dict[str, float]] = None,
        target_refine_max_iters: int = 32,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.alpha = max(0.0, float(alpha))
        self.all_zero_weight = max(0.0, float(all_zero_weight))
        self.label_boost_by_label = {
            str(k): max(0.0, float(v))
            for k, v in (label_boost_by_label or {}).items()
            if float(v) > 0.0
        }
        self.min_label_ratio_by_label = {
            str(k): float(np.clip(v, 0.0, 1.0))
            for k, v in (min_label_ratio_by_label or {}).items()
            if float(v) > 0.0
        }
        self.max_label_ratio_by_label = {
            str(k): float(np.clip(v, 0.0, 1.0))
            for k, v in (max_label_ratio_by_label or {}).items()
            if float(v) < 1.0
        }
        self.target_refine_max_iters = max(1, int(target_refine_max_iters))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.raw_labels = dataset.get_label_matrix().astype(np.float32, copy=False)
        self.labels = _positive_label_matrix(self.raw_labels).astype(np.int32)
        self.known_labels = _known_label_matrix(self.raw_labels).astype(np.int32)
        self.ptm_index = {str(name): idx for idx, name in enumerate(dataset.ptm_types)}

        # 1. Group indices by positive-label signature. Unknown labels (-1) are ignored here.
        self.signature_to_indices: Dict[Tuple[int, ...], List[int]] = {}
        for idx, row in enumerate(self.labels):
            signature = tuple(int(v) for v in row.tolist())
            self.signature_to_indices.setdefault(signature, []).append(idx)

        self.signatures = list(self.signature_to_indices.keys())
        self.signature_index_pools = [np.asarray(self.signature_to_indices[sig], dtype=np.int64) for sig in self.signatures]
        self.num_samples = len(dataset)

        # 2. Calculate Smoothed Weights for each signature
        counts = np.asarray([len(self.signature_to_indices[sig]) for sig in self.signatures], dtype=np.float64)
        self.signature_matrix = (
            np.asarray(self.signatures, dtype=np.float32)
            if self.signatures
            else np.zeros((0, len(dataset.ptm_types)), dtype=np.float32)
        )

        # Apply inverse-frequency smoothing so larger alpha more directly boosts minority signatures.
        smoothed_weights = np.power(np.clip(counts, 1.0, None), -self.alpha)

        # 3. Apply explicit all-zero downweighting if provided
        all_zero_sig = tuple([0] * len(dataset.ptm_types))
        if all_zero_sig in self.signatures:
            az_idx = self.signatures.index(all_zero_sig)
            smoothed_weights[az_idx] *= self.all_zero_weight

        # 4. Apply explicit label-aware boosts on signatures containing the target labels.
        if self.signature_matrix.size and self.label_boost_by_label:
            for label_name, boost in self.label_boost_by_label.items():
                label_idx = self.ptm_index.get(label_name)
                if label_idx is None:
                    continue
                mask = self.signature_matrix[:, label_idx] > 0.5
                if np.any(mask):
                    smoothed_weights[mask] *= float(boost)

        # 5. Enforce explicit target marginal ratios for selected labels.
        smoothed_weights = self._apply_label_ratio_constraints(smoothed_weights)

        if smoothed_weights.size == 0 or smoothed_weights.sum() <= 0:
            self.signature_probs = np.zeros((0,), dtype=np.float64)
        else:
            self.signature_probs = smoothed_weights / smoothed_weights.sum()
        self.dataset_label_distribution = _positive_label_distribution(self.raw_labels)
        self.dataset_known_distribution = (
            self.known_labels.mean(axis=0).astype(np.float32, copy=False)
            if self.known_labels.size
            else np.zeros((len(dataset.ptm_types),), dtype=np.float32)
        )
        self.label_distribution = (
            np.matmul(self.signature_probs.astype(np.float32), self.signature_matrix).astype(np.float32)
            if self.signature_matrix.size and self.signature_probs.size
            else self.dataset_label_distribution.copy()
        )
        self._epoch = 0

    def _solve_scale_for_ratio(self, current_mass: float, target_mass: float) -> Optional[float]:
        current_mass = float(np.clip(current_mass, 0.0, 1.0))
        target_mass = float(np.clip(target_mass, 0.0, 1.0))
        if current_mass <= 0.0 or current_mass >= 1.0:
            return None
        if target_mass <= 0.0 or target_mass >= 1.0:
            return None
        numerator = target_mass * (1.0 - current_mass)
        denominator = current_mass * (1.0 - target_mass)
        if denominator <= 0.0:
            return None
        return float(max(numerator / denominator, 0.0))

    def _apply_label_ratio_constraints(self, weights: np.ndarray) -> np.ndarray:
        adjusted = np.clip(weights.astype(np.float64, copy=True), 1e-12, None)
        if adjusted.size == 0 or self.signature_matrix.size == 0:
            return adjusted
        for _ in range(self.target_refine_max_iters):
            probs = adjusted / max(adjusted.sum(), 1e-12)
            marginals = np.matmul(probs.astype(np.float32), self.signature_matrix).astype(np.float64)
            updated = False

            for label_name, min_ratio in self.min_label_ratio_by_label.items():
                label_idx = self.ptm_index.get(label_name)
                if label_idx is None:
                    continue
                current_ratio = float(marginals[label_idx])
                if current_ratio + 1e-6 >= float(min_ratio):
                    continue
                mask = self.signature_matrix[:, label_idx] > 0.5
                if not np.any(mask):
                    continue
                scale = self._solve_scale_for_ratio(current_ratio, float(min_ratio))
                if scale is None or scale <= 1.0:
                    continue
                adjusted[mask] *= scale
                updated = True

            if updated:
                probs = adjusted / max(adjusted.sum(), 1e-12)
                marginals = np.matmul(probs.astype(np.float32), self.signature_matrix).astype(np.float64)

            for label_name, max_ratio in self.max_label_ratio_by_label.items():
                label_idx = self.ptm_index.get(label_name)
                if label_idx is None:
                    continue
                current_ratio = float(marginals[label_idx])
                if current_ratio <= float(max_ratio) + 1e-6:
                    continue
                mask = self.signature_matrix[:, label_idx] > 0.5
                if not np.any(mask):
                    continue
                scale = self._solve_scale_for_ratio(current_ratio, float(max_ratio))
                if scale is None:
                    continue
                scale = min(scale, 1.0)
                adjusted[mask] *= max(scale, 1e-6)
                updated = True

            if not updated:
                break
        return adjusted

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    @staticmethod
    def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
        if total <= 0 or weights.size == 0:
            return np.zeros_like(weights, dtype=np.int64)
        safe = np.clip(weights.astype(np.float64), 0.0, None)
        if safe.sum() <= 0:
            safe = np.ones_like(safe, dtype=np.float64)
        raw = safe / safe.sum() * float(total)
        counts = np.floor(raw).astype(np.int64)
        remainder = int(total - counts.sum())
        if remainder > 0:
            frac = raw - counts
            for idx in np.argsort(-frac)[:remainder]:
                counts[idx] += 1
        return counts

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.signature_probs.size == 0 or not self.signatures:
            return

        total_batches = len(self)
        # Initialize permutations for coverage + oversampling mixed mode
        pool_perms = [rng.permutation(len(pool)) for pool in self.signature_index_pools]
        pool_pointers = [0] * len(self.signatures)

        for batch_idx in range(total_batches):
            if self.drop_last:
                current_batch_size = self.batch_size
            else:
                remaining_items = self.num_samples - batch_idx * self.batch_size
                current_batch_size = min(self.batch_size, max(0, remaining_items))
            if current_batch_size <= 0:
                break

            # Sample signature composition from the alpha-smoothed distribution
            group_counts = rng.multinomial(current_batch_size, self.signature_probs)
            batch: List[int] = []
            for sig_idx, count in enumerate(group_counts):
                take = int(count)
                if take <= 0:
                    continue
                
                pool = self.signature_index_pools[sig_idx]
                perm = pool_perms[sig_idx]
                ptr = pool_pointers[sig_idx]
                
                picked = []
                while take > 0:
                    available = len(pool) - ptr
                    if available >= take:
                        picked.extend(pool[perm[ptr:ptr+take]])
                        pool_pointers[sig_idx] += take
                        take = 0
                    else:
                        picked.extend(pool[perm[ptr:]])
                        take -= available
                        # reshuffle and reset pointer
                        perm = rng.permutation(len(pool))
                        pool_perms[sig_idx] = perm
                        pool_pointers[sig_idx] = 0
                        ptr = 0
                
                batch.extend(int(x) for x in picked)

            rng.shuffle(batch)
            if batch:
                yield batch


class MinimumLabelCoverageBatchSampler(Sampler[List[int]]):
    """Shuffle once per epoch, with minimal replacements to keep target labels present.

    This preserves the dataset's natural label distribution far better than inverse-frequency
    signature sampling. It only replaces a sample when a configured label is completely absent
    from the current batch, which is mainly useful for small batch or sanity runs.
    """

    def __init__(
        self,
        dataset: MultiLabelPTMJsonDataset,
        batch_size: int,
        coverage_labels: Optional[List[str]] = None,
        min_positives: int = 1,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.min_positives = max(1, int(min_positives))
        self.raw_labels = dataset.get_label_matrix().astype(np.float32, copy=False)
        self.labels = _positive_label_matrix(self.raw_labels).astype(np.int32)
        self.num_samples = len(dataset)
        self.ptm_index = {sanitize_name(str(name)): idx for idx, name in enumerate(dataset.ptm_types)}
        requested = [sanitize_name(str(x)) for x in (coverage_labels or []) if str(x).strip()]
        self.coverage_label_indices = [self.ptm_index[x] for x in requested if x in self.ptm_index]
        self.coverage_label_names = [str(dataset.ptm_types[idx]) for idx in self.coverage_label_indices]
        self.label_to_indices: Dict[int, np.ndarray] = {}
        for label_idx in self.coverage_label_indices:
            pool = np.where(self.labels[:, label_idx] > 0)[0].astype(np.int64)
            if pool.size > 0:
                self.label_to_indices[label_idx] = pool
        self.dataset_label_distribution = _positive_label_distribution(self.raw_labels)
        self.label_distribution = self.dataset_label_distribution.copy()
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _next_positive_index(
        self,
        label_idx: int,
        rng: np.random.Generator,
        pool_perms: Dict[int, np.ndarray],
        pool_ptrs: Dict[int, int],
        used: Set[int],
    ) -> Optional[int]:
        pool = self.label_to_indices.get(label_idx)
        if pool is None or pool.size == 0:
            return None
        if label_idx not in pool_perms:
            pool_perms[label_idx] = rng.permutation(pool.size)
            pool_ptrs[label_idx] = 0
        for _ in range(max(1, pool.size * 2)):
            ptr = pool_ptrs[label_idx]
            if ptr >= pool.size:
                pool_perms[label_idx] = rng.permutation(pool.size)
                pool_ptrs[label_idx] = 0
                ptr = 0
            candidate = int(pool[pool_perms[label_idx][ptr]])
            pool_ptrs[label_idx] = ptr + 1
            if candidate not in used:
                return candidate
        return int(pool[int(rng.integers(0, pool.size))])

    def _replacement_slot(self, batch_arr: np.ndarray, rng: np.random.Generator, protected_slots: Set[int]) -> int:
        if batch_arr.size <= 1:
            return 0
        batch_labels = self.labels[batch_arr]
        coverage_sum = (
            batch_labels[:, self.coverage_label_indices].sum(axis=1)
            if self.coverage_label_indices
            else np.zeros((batch_arr.size,), dtype=np.int32)
        )
        all_positive_sum = batch_labels.sum(axis=1)
        candidates = [
            idx for idx in range(batch_arr.size)
            if idx not in protected_slots and coverage_sum[idx] == 0 and all_positive_sum[idx] == 0
        ]
        if not candidates:
            candidates = [idx for idx in range(batch_arr.size) if idx not in protected_slots and coverage_sum[idx] == 0]
        if not candidates:
            candidates = [idx for idx in range(batch_arr.size) if idx not in protected_slots]
        if not candidates:
            candidates = list(range(batch_arr.size))
        return int(candidates[int(rng.integers(0, len(candidates)))])

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.num_samples <= 0:
            return
        order = rng.permutation(self.num_samples).astype(np.int64)
        pool_perms: Dict[int, np.ndarray] = {}
        pool_ptrs: Dict[int, int] = {}
        total_batches = len(self)
        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_arr = order[start:end].copy()
            if batch_arr.size < self.batch_size and self.drop_last:
                continue
            if batch_arr.size == 0:
                continue
            if self.coverage_label_indices:
                used = {int(x) for x in batch_arr.tolist()}
                protected_slots: Set[int] = set()
                for label_idx in self.coverage_label_indices:
                    current = int(self.labels[batch_arr, label_idx].sum())
                    needed = max(0, self.min_positives - current)
                    for _ in range(needed):
                        candidate = self._next_positive_index(label_idx, rng, pool_perms, pool_ptrs, used)
                        if candidate is None:
                            break
                        slot = self._replacement_slot(batch_arr, rng, protected_slots)
                        used.discard(int(batch_arr[slot]))
                        batch_arr[slot] = int(candidate)
                        used.add(int(candidate))
                        protected_slots.add(slot)
            rng.shuffle(batch_arr)
            yield [int(x) for x in batch_arr.tolist()]


class BatchDistributionMonitor:
    def __init__(
        self,
        ptm_types: List[str],
        target_distribution: np.ndarray,
        log_interval: int = 50,
        relative_tolerance: float = 0.25,
        min_abs_tolerance: float = 0.02,
        enable_batch_logging: bool = False,
    ) -> None:
        self.ptm_types = list(ptm_types)
        self.target_distribution = np.asarray(target_distribution, dtype=np.float32)
        self.log_interval = max(1, int(log_interval))
        self.relative_tolerance = max(0.0, float(relative_tolerance))
        self.min_abs_tolerance = max(0.0, float(min_abs_tolerance))
        self.enable_batch_logging = bool(enable_batch_logging)
        self.total_batches = 0
        self.warn_batches = 0
        self.max_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float32)
        self.mean_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float64)
        self._epoch_total_batches = 0
        self._epoch_warn_batches = 0
        self._epoch_max_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float32)
        self._epoch_sum_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float64)

    def update(self, labels: torch.Tensor, logger: logging.Logger, step: int, stage: str = "train") -> None:
        if labels.ndim != 2 or labels.numel() == 0:
            return
        positive = (labels.detach().float() > 0.0).float()
        actual = positive.mean(dim=0).cpu().numpy().astype(np.float32)
        abs_delta = np.abs(actual - self.target_distribution)
        tolerance = np.maximum(self.min_abs_tolerance, self.target_distribution * self.relative_tolerance)
        violated = abs_delta > tolerance
        self.total_batches += 1
        self._epoch_total_batches += 1
        self.max_abs_delta = np.maximum(self.max_abs_delta, abs_delta)
        self.mean_abs_delta += abs_delta
        self._epoch_max_abs_delta = np.maximum(self._epoch_max_abs_delta, abs_delta)
        self._epoch_sum_abs_delta += abs_delta
        should_log = self.enable_batch_logging and (step % self.log_interval == 0)
        if np.any(violated):
            self.warn_batches += 1
            self._epoch_warn_batches += 1
        if should_log:
            parts = []
            for idx, name in enumerate(self.ptm_types):
                parts.append(
                    f"{name}:actual={actual[idx]:.4f},target={self.target_distribution[idx]:.4f},delta={abs_delta[idx]:.4f}"
                )
            level = logger.warning if np.any(violated) else logger.info
            level("BatchDistribution[%s] step=%d | %s", stage, step, " | ".join(parts))

    def flush_epoch_summary(self, logger: logging.Logger, stage: str, epoch: int) -> None:
        if self._epoch_total_batches <= 0:
            return
        mean_abs_delta = self._epoch_sum_abs_delta / float(self._epoch_total_batches)
        parts = []
        for idx, name in enumerate(self.ptm_types):
            parts.append(
                f"{name}:mean_delta={mean_abs_delta[idx]:.4f},max_delta={self._epoch_max_abs_delta[idx]:.4f},target={self.target_distribution[idx]:.4f}"
            )
        logger.info(
            "BatchDistributionEpoch[%s] epoch=%d batches=%d warn_batches=%d | %s",
            stage,
            int(epoch),
            int(self._epoch_total_batches),
            int(self._epoch_warn_batches),
            " | ".join(parts),
        )
        self._epoch_total_batches = 0
        self._epoch_warn_batches = 0
        self._epoch_max_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float32)
        self._epoch_sum_abs_delta = np.zeros_like(self.target_distribution, dtype=np.float64)

    def summary(self) -> Dict[str, Any]:
        denom = max(1, self.total_batches)
        return {
            "target_distribution": {name: float(self.target_distribution[idx]) for idx, name in enumerate(self.ptm_types)},
            "mean_abs_delta": {name: float(self.mean_abs_delta[idx] / denom) for idx, name in enumerate(self.ptm_types)},
            "max_abs_delta": {name: float(self.max_abs_delta[idx]) for idx, name in enumerate(self.ptm_types)},
            "warn_batches": int(self.warn_batches),
            "total_batches": int(self.total_batches),
        }


@dataclass
class EarlyStoppingState:
    best_score: Optional[float] = None
    bad_epochs: int = 0


class EarlyStopping:
    def __init__(self, mode: str = "max", patience: int = 10, min_delta: float = 0.0) -> None:
        self.mode = str(mode).lower()
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.state = EarlyStoppingState()

    def step(self, value: float) -> Tuple[bool, bool]:
        if math.isnan(value):
            self.state.bad_epochs += 1
            return False, self.state.bad_epochs >= self.patience
        if self.state.best_score is None:
            self.state.best_score = float(value)
            self.state.bad_epochs = 0
            return True, False
        if self.mode == "min":
            improved = float(value) < float(self.state.best_score) - self.min_delta
        else:
            improved = float(value) > float(self.state.best_score) + self.min_delta
        if improved:
            self.state.best_score = float(value)
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1
        return improved, self.state.bad_epochs >= self.patience
