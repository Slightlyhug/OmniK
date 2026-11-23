"""Uncertainty-Guided Memory Bank for Exemplar Selection.

Based on "From Uncertainty to Clarity" (IEEE JBHI 2024), tracks cumulative
entropy across training epochs to identify the most informative samples
for replay/prototype update in imbalanced PTM prediction.

Key ideas (论文 Sec. IV-E, Eq. 14-18):
1. Cumulative entropy H_cumulative(x_i) = ∑_{t=1}^T H^(t)(x_i)
   measures sample uncertainty trajectory across all training epochs
2. High-entropy samples sit near decision boundaries → most informative
3. Select top-m samples per task as exemplars for memory bank
4. Use exemplars to update DynamicCosineClassifier prototypes

Addresses:
- Catastrophic forgetting (new negatives harm weak tasks)
- Redundant negative sampling (current 1:5 ratio has many trivial negatives)
- Class imbalance (Khib 7%, Kla/Kcr long-tail)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


class UncertaintyTrajectoryAnalyzer:
    """Tracks per-sample cumulative entropy across training epochs.

    During training:
    1. After each forward pass, compute H^(t)(x_i) for each sample
    2. Accumulate into cumulative_entropy[sample_id]
    3. At epoch end, select top-K samples with highest entropy as exemplars

    Args:
        num_tasks: Number of PTM tasks (8 for lysine).
        memory_size_per_task: Max exemplars to store per task.
        epoch_threshold: Start cumulative entropy after this many epochs
            (论文 hyperparameter: skip initial 80 epochs for warm-up).
        entropy_normalization: How to normalize probability distributions.
            - "softmax": H = -∑ p_i log(p_i), where p = softmax(logits)
            - "l2": L2 normalization before entropy (论文 Eq. 19)
        device: Where to store exemplar features (CPU to save GPU memory).
    """

    def __init__(
        self,
        num_tasks: int,
        memory_size_per_task: int = 200,
        epoch_threshold: int = 0,
        entropy_normalization: str = "l2",
        device: str = "cpu",
        feature_dim: int = 512,
        cache_features: bool = False,
    ) -> None:
        self.num_tasks = int(num_tasks)
        self.memory_size_per_task = int(memory_size_per_task)
        self.epoch_threshold = int(epoch_threshold)
        self.entropy_normalization = str(entropy_normalization).lower()
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.cache_features = bool(cache_features)

        # Core tracking structures
        self.cumulative_entropy: Dict[str, np.ndarray] = {}  # sample_id → [num_tasks]
        self.sample_metadata: Dict[str, Dict[str, Any]] = {}  # sample_id → {labels, ...}
        self.current_epoch = 0
        self._seen_this_epoch: set[str] = set()

        # Memory bank: task_idx → List[(sample_id, cumulative_entropy, features, labels)]
        self.memory_bank: Dict[int, List[Tuple[str, float, torch.Tensor, torch.Tensor]]] = \
            {task_idx: [] for task_idx in range(num_tasks)}

        # Statistics for monitoring
        self.epoch_sample_counts: List[int] = []
        self.epoch_avg_entropy: List[float] = []

    def compute_sample_entropy(
        self,
        logits: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute per-sample entropy from logits.

        Args:
            logits: [B, num_tasks] raw model outputs.
            method: Override self.entropy_normalization if provided.

        Returns:
            entropy: [B, num_tasks] per-task entropy for each sample.
        """
        method = method or self.entropy_normalization

        if method == "l2":
            # Paper Eq. 19: L2-normalize the probability distribution.
            probs = torch.sigmoid(logits)  # [B, T]
            # L2 normalization: M_norm = M / sqrt(sum(M^2))
            prob_l2_norm = probs / (probs.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-8)
            entropy = -torch.sum(
                prob_l2_norm * torch.log(prob_l2_norm.clamp(min=1e-8)),
                dim=0,
            )  # [T] → broadcast to [B, T]
            # Actually we want per-sample entropy:
            entropy = -(probs * torch.log(probs.clamp(min=1e-8)) +
                        (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))  # [B, T]
        else:  # softmax (but for multi-label, we use binary entropy)
            probs = torch.sigmoid(logits)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8)) +
                        (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))  # [B, T]

        return entropy

    def update(
        self,
        sample_ids: List[str],
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate entropy for samples in this batch.

        Args:
            sample_ids: List of unique sample identifiers (e.g., "P12345_K123").
            logits: [B, num_tasks] model predictions.
            labels: [B, num_tasks] ground truth (-1=unknown, 0=neg, 1=pos).
            features: [B, D] optional normalized features for prototype updates.
        """
        if self.current_epoch < self.epoch_threshold:
            return  # Skip warm-up epochs

        batch_size = logits.size(0)
        assert len(sample_ids) == batch_size, \
            f"sample_ids length {len(sample_ids)} != batch_size {batch_size}"

        # Compute per-sample, per-task entropy
        with torch.no_grad():
            entropy = self.compute_sample_entropy(logits)  # [B, num_tasks]
            entropy_np = entropy.cpu().numpy()
            labels_np = labels.cpu().numpy()

        # Accumulate into cumulative entropy dictionary
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self._seen_this_epoch:
                continue
            self._seen_this_epoch.add(sample_id)
            if sample_id not in self.cumulative_entropy:
                self.cumulative_entropy[sample_id] = np.zeros(self.num_tasks, dtype=np.float32)
                self.sample_metadata[sample_id] = {
                    "labels": labels_np[i].copy(),
                    "first_seen_epoch": self.current_epoch,
                }

            # Paper Eq. 18: H_cumulative(x_i) = sum_t H^(t)(x_i).
            self.cumulative_entropy[sample_id] += entropy_np[i]

            # Update metadata (in case labels changed, though shouldn't in PTM)
            self.sample_metadata[sample_id]["labels"] = labels_np[i].copy()
            self.sample_metadata[sample_id]["last_seen_epoch"] = self.current_epoch

            # Store features if provided (for prototype initialization)
            if features is not None and self.cache_features:
                self.sample_metadata[sample_id]["features"] = \
                    features[i].detach().cpu().clone()

    def select_exemplars_per_epoch(self) -> Dict[int, List[str]]:
        """Select top-m samples per task based on cumulative entropy.

        Returns:
            selected: {task_idx: [sample_ids]} of high-entropy samples.
        """
        if not self.cumulative_entropy:
            return {task_idx: [] for task_idx in range(self.num_tasks)}

        selected: Dict[int, List[str]] = {task_idx: [] for task_idx in range(self.num_tasks)}

        for task_idx in range(self.num_tasks):
            # Collect (sample_id, cumulative_entropy) for samples with this task label
            candidates: List[Tuple[str, float]] = []
            for sample_id, cum_ent_vec in self.cumulative_entropy.items():
                meta = self.sample_metadata[sample_id]
                label = meta["labels"][task_idx]

                # Only select positive or reliable negative samples (exclude unknowns)
                if label < 0:
                    continue

                cum_ent = float(cum_ent_vec[task_idx])
                candidates.append((sample_id, cum_ent))

            # Sort by descending cumulative entropy
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Select top-m
            top_m = candidates[:self.memory_size_per_task]
            selected[task_idx] = [sid for sid, _ in top_m]

            logger.info(
                f"Task {task_idx}: selected {len(top_m)} / {len(candidates)} "
                f"exemplars (avg entropy: {np.mean([e for _, e in top_m]):.4f})"
            )

        return selected

    def update_memory_bank(
        self,
        selected: Dict[int, List[str]],
        model: torch.nn.Module,
        dataset: Any,
        collate_fn: Optional[Any] = None,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """Extract features for selected exemplars and update memory bank.

        Args:
            selected: {task_idx: [sample_ids]} from select_exemplars_per_epoch.
            model: Trained model to extract features (set to eval mode).
            dataset: Dataset used to re-encode selected exemplars when features
                are not retained in the trajectory metadata.
        """
        was_training = bool(model.training)
        model.eval()

        feature_by_id: Dict[str, torch.Tensor] = {}
        missing_ids = sorted({sid for ids in selected.values() for sid in ids if sid not in self.sample_metadata or "features" not in self.sample_metadata[sid]})
        if missing_ids and dataset is not None and hasattr(dataset, "get_exemplar_features"):
            feature_by_id.update(
                dataset.get_exemplar_features(
                    model,
                    missing_ids,
                    device or next(model.parameters()).device,
                    self.feature_dim,
                )
            )
        index_to_id: Dict[int, str] = {}
        for sample_id in missing_ids:
            try:
                index_to_id[int(sample_id)] = sample_id
            except (TypeError, ValueError):
                continue

        if index_to_id and dataset is not None and not feature_by_id:
            from src.utils.training_data import to_device

            target_device = device or next(model.parameters()).device
            model.eval()
            indices = list(index_to_id.keys())
            loader = DataLoader(
                dataset,
                sampler=indices,
                batch_size=max(1, int(batch_size)),
                num_workers=0,
                collate_fn=collate_fn or default_collate,
            )
            offset = 0
            try:
                with torch.no_grad():
                    for batch in loader:
                        count = len(batch.get("sample_indices", [])) if isinstance(batch, dict) else 0
                        if count <= 0:
                            count = min(len(indices) - offset, int(batch["label"].shape[0]))
                        try:
                            batch_device = to_device(batch, target_device)
                        except (KeyError, TypeError):
                            def _move(value: Any) -> Any:
                                if isinstance(value, torch.Tensor):
                                    return value.to(target_device, non_blocking=True)
                                if isinstance(value, dict):
                                    return {key: _move(item) for key, item in value.items()}
                                if isinstance(value, list):
                                    return [_move(item) for item in value]
                                if isinstance(value, tuple):
                                    return tuple(_move(item) for item in value)
                                return value
                            batch_device = _move(batch)
                        outputs = model(batch_device)
                        features = outputs.get("shared_representation") if isinstance(outputs, dict) else None
                        if not isinstance(features, torch.Tensor) or features.ndim != 2:
                            raise RuntimeError("UTA exemplar re-encoding requires model outputs['shared_representation'] [B, D].")
                        batch_ids = batch.get("sample_indices", []) if isinstance(batch, dict) else []
                        for row in range(count):
                            if row < len(batch_ids) and batch_ids[row] is not None:
                                sample_id = str(batch_ids[row])
                            else:
                                sample_id = index_to_id[indices[offset + row]]
                            feature_by_id[sample_id] = features[row].detach().to(self.device).float()
                        offset += count
            finally:
                if was_training:
                    model.train()

        for task_idx, sample_ids in selected.items():
            exemplars: List[Tuple[str, float, torch.Tensor, torch.Tensor]] = []
            for sample_id in sample_ids:
                meta = self.sample_metadata.get(sample_id)
                if meta is None:
                    continue
                features = meta.get("features")
                if features is None:
                    features = feature_by_id.get(sample_id)
                if features is None:
                    continue
                labels = torch.as_tensor(meta["labels"], dtype=torch.float32)
                cum_ent = float(self.cumulative_entropy[sample_id][task_idx])
                exemplars.append((sample_id, cum_ent, features.to(self.device), labels))
            self.memory_bank[task_idx] = exemplars
            logger.info("Task %d: memory bank updated with %d exemplars", task_idx, len(exemplars))

        if was_training:
            model.train()

    def get_exemplar_features_labels(
        self,
        task_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve exemplar features and labels for prototype update.

        Args:
            task_idx: If provided, return exemplars for this task only.
                Otherwise return all exemplars.

        Returns:
            features: [N, D] stacked features.
            labels: [N, num_tasks] stacked labels.
        """
        if task_idx is not None:
            exemplars = self.memory_bank[task_idx]
        else:
            exemplars = []
            for task_exemplars in self.memory_bank.values():
                exemplars.extend(task_exemplars)

        if not exemplars:
            return (
                torch.empty(0, self.feature_dim, device=self.device),
                torch.empty(0, self.num_tasks, device=self.device),
            )

        features = torch.stack([feat for _, _, feat, _ in exemplars], dim=0)
        labels = torch.stack([lab for _, _, _, lab in exemplars], dim=0)

        return features, labels

    def on_epoch_start(self, epoch: int) -> None:
        """Call at the start of each epoch."""
        self.current_epoch = epoch
        self._seen_this_epoch.clear()
        logger.info(f"UncertaintyTrajectoryAnalyzer: starting epoch {epoch}")

    def on_epoch_end(self) -> Dict[str, Any]:
        """Call at the end of each epoch to compute statistics."""
        if not self.cumulative_entropy:
            return {}

        all_entropy = np.array([e.sum() for e in self.cumulative_entropy.values()])
        stats = {
            "num_tracked_samples": len(self.cumulative_entropy),
            "avg_cumulative_entropy": float(all_entropy.mean()),
            "max_cumulative_entropy": float(all_entropy.max()),
            "min_cumulative_entropy": float(all_entropy.min()),
        }

        self.epoch_sample_counts.append(stats["num_tracked_samples"])
        self.epoch_avg_entropy.append(stats["avg_cumulative_entropy"])

        logger.info(
            f"Epoch {self.current_epoch}: tracked {stats['num_tracked_samples']} samples, "
            f"avg entropy {stats['avg_cumulative_entropy']:.4f}"
        )

        return stats

    def save(self, path: Path) -> None:
        """Save cumulative entropy and memory bank to disk."""
        state = {
            "feature_dim": self.feature_dim,
            "cumulative_entropy": {k: v.tolist() for k, v in self.cumulative_entropy.items()},
            "sample_metadata": self.sample_metadata,
            "memory_bank": {
                task_idx: [
                    (sid, ent, feat.cpu(), lab.cpu())
                    for sid, ent, feat, lab in exemplars
                ]
                for task_idx, exemplars in self.memory_bank.items()
            },
            "current_epoch": self.current_epoch,
            "epoch_sample_counts": self.epoch_sample_counts,
            "epoch_avg_entropy": self.epoch_avg_entropy,
        }
        torch.save(state, path)
        logger.info(f"Saved UncertaintyTrajectoryAnalyzer to {path}")

    def load(self, path: Path) -> None:
        """Load cumulative entropy and memory bank from disk."""
        state = torch.load(path, map_location="cpu")
        self.cumulative_entropy = {
            k: np.array(v, dtype=np.float32)
            for k, v in state["cumulative_entropy"].items()
        }
        self.sample_metadata = state["sample_metadata"]
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))
        self.memory_bank = {
            task_idx: [
                (sid, ent, feat.to(self.device), lab.to(self.device))
                for sid, ent, feat, lab in exemplars
            ]
            for task_idx, exemplars in state["memory_bank"].items()
        }
        self.current_epoch = state["current_epoch"]
        self.epoch_sample_counts = state["epoch_sample_counts"]
        self.epoch_avg_entropy = state["epoch_avg_entropy"]
        logger.info(f"Loaded UncertaintyTrajectoryAnalyzer from {path}")


class ExemplarReplayBatchSampler:
    """Mix exemplars into training batches for continual learning.

    Each batch contains:
    - (1 - replay_ratio) * batch_size samples from current training data
    - replay_ratio * batch_size samples from memory bank exemplars

    Args:
        base_sampler: Original BatchSampler for training data.
        memory_bank: UncertaintyTrajectoryAnalyzer with filled memory_bank.
        replay_ratio: Fraction of batch to fill with exemplars (default 0.1).
        balance_tasks: If True, sample exemplars uniformly across tasks.
    """

    def __init__(
        self,
        base_sampler: Any,
        memory_bank: UncertaintyTrajectoryAnalyzer,
        replay_ratio: float = 0.1,
        balance_tasks: bool = True,
        dataset_size: Optional[int] = None,
    ) -> None:
        self.base_sampler = base_sampler
        self.memory_bank = memory_bank
        self.replay_ratio = float(replay_ratio)
        self.balance_tasks = bool(balance_tasks)
        self.dataset_size = int(dataset_size) if dataset_size is not None else None

        logger.info(
            "ExemplarReplayBatchSampler initialized with replay_ratio=%.3f; "
            "exemplar indices are refreshed from the memory bank per epoch.",
            replay_ratio,
        )

    def __iter__(self):
        """Yield batches of valid dataset indices mixed with selected exemplars."""
        exemplar_indices = []
        for task_exemplars in self.memory_bank.memory_bank.values():
            for sample_id, _, _, _ in task_exemplars:
                try:
                    index = int(sample_id)
                except (TypeError, ValueError):
                    continue
                if self.dataset_size is not None and not 0 <= index < self.dataset_size:
                    continue
                if index not in exemplar_indices:
                    exemplar_indices.append(index)
        for base_batch in self.base_sampler:
            base_batch = list(base_batch)
            batch_size = len(base_batch)
            num_replay = min(batch_size, int(round(batch_size * self.replay_ratio)))
            num_train = batch_size - num_replay
            if num_replay == 0 or not exemplar_indices:
                yield base_batch
                continue
            replay = np.random.choice(exemplar_indices, size=num_replay, replace=True).astype(int).tolist()
            yield base_batch[:num_train] + replay

    def __len__(self) -> int:
        return len(self.base_sampler)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)
        sampler = getattr(self.base_sampler, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


def build_replay_dataloader(
    loader: DataLoader,
    memory_bank: UncertaintyTrajectoryAnalyzer,
    replay_ratio: float,
) -> DataLoader:
    """Rebuild a loader with the original worker setup and replay batch sampler."""
    batch_sampler = ExemplarReplayBatchSampler(
        base_sampler=loader.batch_sampler,
        memory_bank=memory_bank,
        replay_ratio=replay_ratio,
        dataset_size=len(loader.dataset),
    )
    kwargs: Dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_sampler": batch_sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "generator": loader.generator,
        "persistent_workers": loader.persistent_workers,
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["multiprocessing_context"] = loader.multiprocessing_context
    return DataLoader(**kwargs)
