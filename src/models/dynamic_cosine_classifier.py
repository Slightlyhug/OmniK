"""Dynamic Cosine Classifier with Prototype Update.

Based on "From Uncertainty to Clarity" (IEEE JBHI 2024) for mitigating
classification bias in imbalanced and long-tailed PTM prediction tasks.

Key innovations:
1. L2-normalized features and prototypes on unit hypersphere (Eq. 2-3 in paper)
2. Temperature-scaled cosine similarity for logits (angular distance metric)
3. Momentum-based prototype update using informative exemplars (Eq. 4, 6)
4. Per-task adaptive class representations that evolve during training

Replaces the traditional linear classifier + sigmoid in DualBranchFusionPredictor.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicCosineClassifier(nn.Module):
    """Cosine similarity-based multi-label classifier with dynamic prototypes.

    Unlike traditional linear classifiers where feature magnitudes dominate
    (causing bias toward majority classes in imbalanced data), this module:
    - Normalizes features to unit sphere → only direction matters
    - Updates task prototypes as running average of positive samples
    - Uses temperature-scaled cosine similarity for stable training

    Args:
        feature_dim: Input feature dimensionality (e.g., 512 from fusion module).
        num_tasks: Number of PTM prediction tasks (8 for lysine PTMs).
        temperature: Scaling factor for cosine similarities (default 0.07).
            Lower = sharper boundaries, higher = softer. Typical range [0.05, 0.1].
        momentum: EMA coefficient for prototype updates (default 0.9).
            Higher = more stable, lower = faster adaptation.
        learnable_temperature: If True, temperature becomes a learnable parameter.
        init_prototypes: Optional initial prototypes [num_tasks, feature_dim].
            If None, initialized with Xavier uniform.
        freeze_prototypes: If True, prototypes remain fixed (no momentum updates).
    """

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        temperature: float = 0.07,
        momentum: float = 0.9,
        learnable_temperature: bool = False,
        init_prototypes: Optional[torch.Tensor] = None,
        freeze_prototypes: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_tasks = int(num_tasks)
        self.momentum = float(momentum)
        self.freeze_prototypes = bool(freeze_prototypes)

        # Task prototypes: [num_tasks, feature_dim]
        if init_prototypes is not None:
            assert init_prototypes.shape == (num_tasks, feature_dim), \
                f"init_prototypes shape {init_prototypes.shape} != ({num_tasks}, {feature_dim})"
            self.prototypes = nn.Parameter(init_prototypes.clone(), requires_grad=False)
        else:
            prototypes = torch.empty(num_tasks, feature_dim)
            nn.init.xavier_uniform_(prototypes)
            self.prototypes = nn.Parameter(prototypes, requires_grad=False)

        # Temperature: controls sharpness of cosine similarities
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

        # Track update statistics for debugging/monitoring
        self.register_buffer('update_counts', torch.zeros(num_tasks, dtype=torch.long))
        self.register_buffer('cumulative_cosine_sim', torch.zeros(num_tasks))

    @property
    def temperature(self) -> float:
        """Current temperature value (clamped to [0.01, 1.0] for stability)."""
        return torch.clamp(torch.exp(self.log_temperature), 0.01, 1.0).item()

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_prototypes: bool = True,
        return_similarities: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute cosine similarities as logits.

        Args:
            features: [B, feature_dim] fused features from encoder.
            labels: [B, num_tasks] multi-label targets (0/1, or -1 for unknown).
                Used for prototype momentum update during training.
            update_prototypes: If True and training, update prototypes with
                positive samples. Set False during validation/test.
            return_similarities: If True, return (logits, cosine_similarities).

        Returns:
            logits: [B, num_tasks] temperature-scaled cosine similarities.
            cosine_similarities (optional): [B, num_tasks] raw cosine values in [-1, 1].
        """
        # L2-normalize features to unit hypersphere
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, D]

        # L2-normalize prototypes (in case of manual initialization)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)  # [T, D]

        # Compute cosine similarities: dot product of normalized vectors
        cosine_similarities = features_norm @ prototypes_norm.T  # [B, T]

        # Scale by temperature for stable gradients
        temp = torch.exp(self.log_temperature).clamp(0.01, 1.0)
        logits = cosine_similarities / temp

        # Momentum update prototypes during training
        if self.training and labels is not None and update_prototypes and not self.freeze_prototypes:
            self._update_prototypes(features_norm, labels, cosine_similarities)

        if return_similarities:
            return logits, cosine_similarities
        return logits

    @torch.no_grad()
    def _update_prototypes(
        self,
        features_norm: torch.Tensor,
        labels: torch.Tensor,
        cosine_similarities: torch.Tensor,
    ) -> None:
        """Momentum-based prototype update using positive samples (Eq. 4 in paper).

        For each task, average the normalized features of positive samples and
        blend with current prototype:
            p_new = momentum * p_old + (1 - momentum) * mean(positive_features)

        Args:
            features_norm: [B, D] L2-normalized features.
            labels: [B, T] multi-label targets.
            cosine_similarities: [B, T] for monitoring update quality.
        """
        batch_size, num_tasks = labels.shape
        device = features_norm.device

        for task_idx in range(num_tasks):
            # Find positive samples for this task (label == 1)
            pos_mask = labels[:, task_idx] == 1
            num_pos = pos_mask.sum().item()

            if num_pos == 0:
                continue  # No positive samples in this batch

            # Average normalized features of positive samples
            pos_features = features_norm[pos_mask]  # [num_pos, D]
            pos_mean = pos_features.mean(dim=0)  # [D]

            # Momentum update: p_t = m * p_{t-1} + (1-m) * pos_mean
            old_prototype = self.prototypes[task_idx]
            new_prototype = self.momentum * old_prototype + (1 - self.momentum) * pos_mean

            # Re-normalize to keep on unit sphere
            self.prototypes[task_idx] = F.normalize(new_prototype, p=2, dim=0)

            # Update statistics
            self.update_counts[task_idx] += num_pos
            avg_sim = cosine_similarities[pos_mask, task_idx].mean().item()
            self.cumulative_cosine_sim[task_idx] += avg_sim * num_pos

    def get_prototype_stats(self) -> Dict[str, torch.Tensor]:
        """Return prototype update statistics for monitoring."""
        avg_sim_per_task = torch.where(
            self.update_counts > 0,
            self.cumulative_cosine_sim / self.update_counts.float(),
            torch.zeros_like(self.cumulative_cosine_sim)
        )
        return {
            'update_counts': self.update_counts.clone(),
            'avg_cosine_similarity': avg_sim_per_task,
            'prototype_norms': torch.norm(self.prototypes, p=2, dim=-1),
        }

    def reset_stats(self) -> None:
        """Reset update counters (call at epoch start)."""
        self.update_counts.zero_()
        self.cumulative_cosine_sim.zero_()

    def initialize_prototypes_from_data(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialize prototypes as mean of positive samples per task.

        Call this before training with a representative sample of training data.

        Args:
            features: [N, feature_dim] extracted features from training set.
            labels: [N, num_tasks] training labels.
        """
        with torch.no_grad():
            features_norm = F.normalize(features, p=2, dim=-1)
            for task_idx in range(self.num_tasks):
                pos_mask = labels[:, task_idx] == 1
                if pos_mask.sum() > 0:
                    pos_mean = features_norm[pos_mask].mean(dim=0)
                    self.prototypes[task_idx] = F.normalize(pos_mean, p=2, dim=0)
                else:
                    logger.warning(
                        f"Task {task_idx}: no positive samples for prototype init, "
                        "keeping random initialization."
                    )

    @torch.no_grad()
    def update_prototypes_from_exemplars(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        momentum: Optional[float] = None,
        min_positive_samples: int = 1,
    ) -> torch.Tensor:
        """Blend positive exemplar means into prototypes without resetting them."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected exemplar features [N, {self.feature_dim}], got {tuple(features.shape)}")
        if labels.ndim != 2 or labels.shape != (features.shape[0], self.num_tasks):
            raise ValueError(
                f"Expected exemplar labels [{features.shape[0]}, {self.num_tasks}], got {tuple(labels.shape)}"
            )
        if not 0.0 <= float(self.momentum if momentum is None else momentum) < 1.0:
            raise ValueError("Prototype momentum must be in [0, 1).")

        blend = float(self.momentum if momentum is None else momentum)
        features = F.normalize(features.to(self.prototypes.device, dtype=self.prototypes.dtype), dim=-1)
        labels = labels.to(self.prototypes.device)
        updated_counts = torch.zeros(self.num_tasks, dtype=torch.long, device=self.prototypes.device)
        for task_idx in range(self.num_tasks):
            positive = labels[:, task_idx] > 0
            count = int(positive.sum().item())
            if count < int(min_positive_samples):
                continue
            exemplar_mean = F.normalize(features[positive].mean(dim=0), dim=0)
            old = F.normalize(self.prototypes[task_idx], dim=0)
            self.prototypes[task_idx].copy_(F.normalize(blend * old + (1.0 - blend) * exemplar_mean, dim=0))
            self.update_counts[task_idx] += count
            updated_counts[task_idx] = count
        return updated_counts


class AdaptiveTemperatureCosineClassifier(DynamicCosineClassifier):
    """Extended version with per-task learnable temperatures.

    Useful when different PTM tasks have different confidence scales
    (e.g., Khib with 7% positives vs Kcr with 21%).
    """

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        temperature_init: float = 0.07,
        momentum: float = 0.9,
        init_prototypes: Optional[torch.Tensor] = None,
        freeze_prototypes: bool = False,
    ) -> None:
        # Initialize base class with learnable_temperature=False
        super().__init__(
            feature_dim=feature_dim,
            num_tasks=num_tasks,
            temperature=temperature_init,
            momentum=momentum,
            learnable_temperature=False,
            init_prototypes=init_prototypes,
            freeze_prototypes=freeze_prototypes,
        )

        # Override with per-task temperatures
        self.log_temperatures = nn.Parameter(
            torch.full((num_tasks,), math.log(temperature_init))
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_prototypes: bool = True,
        return_similarities: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward with per-task temperature scaling."""
        features_norm = F.normalize(features, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        cosine_similarities = features_norm @ prototypes_norm.T  # [B, T]

        # Per-task temperature scaling
        temps = torch.exp(self.log_temperatures).clamp(0.01, 1.0)  # [T]
        logits = cosine_similarities / temps.unsqueeze(0)  # [B, T] / [1, T]

        if self.training and labels is not None and update_prototypes and not self.freeze_prototypes:
            self._update_prototypes(features_norm, labels, cosine_similarities)

        if return_similarities:
            return logits, cosine_similarities
        return logits
