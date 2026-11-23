"""Multi-label objectives used by the training pipeline."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.training_config import sanitize_name


def compute_pos_weight(dataset: Any) -> torch.Tensor:
    y = dataset.get_label_matrix()
    known = y >= 0.0
    pos = np.logical_and(y > 0.0, known).sum(axis=0)
    neg = np.logical_and(y == 0.0, known).sum(axis=0)
    weights = (neg + 1.0) / (pos + 1.0)
    return torch.tensor(weights, dtype=torch.float32)


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label long-tail learning."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(max(0.0, clip))
        self.eps = float(max(1e-12, eps))
        self.reduction = str(reduction)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)

        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = probs
        if self.clip > 0:
            neg_probs = (neg_probs - self.clip).clamp(min=0.0, max=1.0)
        neg_probs = neg_probs.clamp(min=self.eps, max=1.0 - self.eps)

        # Ensure bases are strictly > 0 for pow/log gradients
        p_term = (1.0 - pos_probs).clamp(min=self.eps)
        n_term = neg_probs.clamp(min=self.eps)
        
        pos_loss = -targets * torch.pow(p_term, self.gamma_pos) * torch.log(pos_probs.clamp(min=self.eps))
        neg_loss = -(1.0 - targets) * torch.pow(n_term, self.gamma_neg) * torch.log((1.0 - neg_probs).clamp(min=self.eps))

        if self.pos_weight is not None:
            pos_loss = pos_loss * self.pos_weight.view(1, -1).to(pos_loss.device)

        loss = pos_loss + neg_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class AdaptiveTaskWeightedLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        num_tasks: int,
        pos_weight: Optional[torch.Tensor] = None,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        task_weights: Optional[List[float]] = None,
        use_homoscedastic: bool = False,
        use_rank_loss: bool = True,
        rank_loss_margin: float = 0.3,
        rank_loss_weight: float = 0.5,
        rank_loss_task_weights: Optional[Dict[str, float]] = None,
        rank_loss_hard_negative_labels: Optional[List[str]] = None,
        rank_loss_hard_negative_top_fraction: float = 1.0,
        ptm_types: Optional[List[str]] = None,
        pu_task_labels: Optional[List[str]] = None,
        pu_positive_priors: Optional[List[float]] = None,
        pu_reliable_negative_weight: float = 0.35,
        pu_unlabeled_weight: float = 1.0,
        pu_nn_beta: float = 0.0,
        use_label_state_loss: bool = False,
        state_reliable_negative_weight: float = 1.0,
        state_observed_negative_weight: float = 0.7,
        state_unknown_weight: float = 0.0,
        state_soft_positive_target: float = 0.7,
        state_soft_positive_weight: float = 0.5,
        state_conflict_weight: float = 0.0,
        log_var_min: float = float("-inf"),
        log_var_max: float = float("inf"),
        log_var_freeze_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.num_tasks = int(num_tasks)
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(max(0.0, clip))
        self.eps = float(max(1e-12, eps))
        self.use_rank_loss = bool(use_rank_loss)
        self.rank_loss_margin = float(max(0.0, rank_loss_margin))
        self.rank_loss_weight = float(max(0.0, rank_loss_weight))
        self.rank_loss_hard_negative_labels = {sanitize_name(x) for x in (rank_loss_hard_negative_labels or [])}
        self.rank_loss_hard_negative_top_fraction = float(np.clip(rank_loss_hard_negative_top_fraction, 0.05, 1.0))
        self.ptm_types = [sanitize_name(x) for x in (ptm_types or [])]
        rank_weights = np.ones((self.num_tasks,), dtype=np.float32)
        if isinstance(rank_loss_task_weights, dict) and self.ptm_types:
            for idx, name in enumerate(self.ptm_types[: self.num_tasks]):
                if name in rank_loss_task_weights:
                    rank_weights[idx] = max(0.0, float(rank_loss_task_weights[name]))
        self.register_buffer("rank_loss_task_weights", torch.tensor(rank_weights, dtype=torch.float32).view(-1))
        pu_labels = {sanitize_name(x) for x in (pu_task_labels or [])}
        pu_mask = np.zeros((self.num_tasks,), dtype=np.float32)
        if pu_labels and self.ptm_types:
            for idx, name in enumerate(self.ptm_types[: self.num_tasks]):
                if name in pu_labels:
                    pu_mask[idx] = 1.0
        self.register_buffer("pu_task_mask", torch.tensor(pu_mask, dtype=torch.float32).view(-1))
        priors = np.asarray(pu_positive_priors if pu_positive_priors is not None else [0.05] * self.num_tasks, dtype=np.float32)
        if priors.shape[0] != self.num_tasks:
            priors = np.full((self.num_tasks,), 0.05, dtype=np.float32)
        priors = np.clip(priors, 1e-4, 0.95)
        self.register_buffer("pu_positive_priors", torch.tensor(priors, dtype=torch.float32).view(-1))
        self.pu_reliable_negative_weight = float(np.clip(pu_reliable_negative_weight, 0.0, 1.0))
        self.pu_unlabeled_weight = float(max(0.0, pu_unlabeled_weight))
        self.pu_nn_beta = float(max(0.0, pu_nn_beta))
        self.use_label_state_loss = bool(use_label_state_loss)
        self.state_reliable_negative_weight = float(max(0.0, state_reliable_negative_weight))
        self.state_observed_negative_weight = float(max(0.0, state_observed_negative_weight))
        self.state_unknown_weight = float(max(0.0, state_unknown_weight))
        self.state_soft_positive_target = float(np.clip(state_soft_positive_target, 0.01, 0.99))
        self.state_soft_positive_weight = float(max(0.0, state_soft_positive_weight))
        self.state_conflict_weight = float(max(0.0, state_conflict_weight))
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().float().view(1, -1))
        else:
            self.pos_weight = None
        if task_weights is None:
            task_weights = [1.0] * self.num_tasks
        weights = np.asarray(task_weights, dtype=np.float32)
        if weights.shape[0] != self.num_tasks:
            weights = np.ones((self.num_tasks,), dtype=np.float32)
        weights = np.clip(weights, 1e-4, None)
        self.register_buffer("task_weights", torch.tensor(weights, dtype=torch.float32).view(-1))
        self.use_homoscedastic = bool(use_homoscedastic)
        self.log_var_min = float(min(log_var_min, log_var_max))
        self.log_var_max = float(max(log_var_min, log_var_max))
        self.log_var_freeze_epochs = max(0, int(log_var_freeze_epochs))
        if self.use_homoscedastic:
            init_log_vars = np.log(1.0 / weights).astype(np.float32)
            self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        else:
            self.log_vars = None
        self.last_task_losses: Optional[torch.Tensor] = None
        self.supports_sample_weight = True
        self.current_epoch = 0
        self.max_epochs = 100

    def _effective_log_vars(self) -> Optional[torch.Tensor]:
        if self.log_vars is None:
            return None
        with torch.no_grad():
            self.log_vars.clamp_(self.log_var_min, self.log_var_max)
        bounded = self.log_vars.clamp(self.log_var_min, self.log_var_max)
        if self.current_epoch <= self.log_var_freeze_epochs:
            bounded = bounded.detach()
        return bounded

    def _unreduced_asl_components(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logits.float()
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = probs
        if self.clip > 0:
            neg_probs = (neg_probs - self.clip).clamp(min=0.0, max=1.0)
        neg_probs = neg_probs.clamp(min=self.eps, max=1.0 - self.eps)

        # Ensure bases are strictly > 0 for pow/log gradients
        p_term = (1.0 - pos_probs).clamp(min=self.eps)
        n_term = neg_probs.clamp(min=self.eps)

        pos_unit = -torch.pow(p_term, self.gamma_pos) * torch.log(pos_probs.clamp(min=self.eps))
        neg_unit = -torch.pow(n_term, self.gamma_neg) * torch.log((1.0 - neg_probs).clamp(min=self.eps))
        return pos_unit, neg_unit, probs

    def _unreduced_asl(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_unit, neg_unit, probs = self._unreduced_asl_components(logits)
        pos_loss = targets * pos_unit
        neg_loss = (1.0 - targets) * neg_unit

        # Progressive PU Learning False-Negative Trust Region
        # Only apply PU masking after warming up (e.g. > 50% of training)
        if self.current_epoch > self.max_epochs * 0.5:
            # Do not strongly penalize samples with label 0 but predicted prob > 0.9 (increased from 0.8)
            false_negative_mask = (targets == 0.0) & (probs > 0.9)
            # Soften the penalty (10% remaining) instead of completely zeroing out
            neg_loss = torch.where(false_negative_mask, neg_loss * 0.1, neg_loss)
        
        if self.pos_weight is not None:
            pos_loss = pos_loss * self.pos_weight.to(pos_loss.device)
        return pos_loss + neg_loss

    def _unreduced_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight.to(logits.device).view(-1))
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    def _pairwise_ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        task_rank_weights = self.rank_loss_task_weights.to(logits.device)
        for t in range(logits.size(1)):
            known = targets[:, t] >= 0.0
            pos_logits = logits[known & (targets[:, t] == 1.0), t]
            neg_logits = logits[known & (targets[:, t] == 0.0), t]
            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                task_name = self.ptm_types[t] if t < len(self.ptm_types) else f"task_{t}"
                if task_name in self.rank_loss_hard_negative_labels and self.rank_loss_hard_negative_top_fraction < 1.0:
                    k = max(1, int(math.ceil(float(neg_logits.numel()) * self.rank_loss_hard_negative_top_fraction)))
                    neg_logits = torch.topk(neg_logits, k=k, largest=True, sorted=False).values
                diff = margin - (pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0))
                losses.append(torch.relu(diff).mean())
                weights.append(task_rank_weights[t].clamp(min=0.0))
        if not losses:
            return torch.tensor(0.0, device=logits.device)
        loss_tensor = torch.stack(losses)
        weight_tensor = torch.stack(weights).to(loss_tensor.device)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp(min=1e-6)

    def _state_aware_task_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
        label_state: torch.Tensor,
        label_reliability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        state = label_state.to(device=logits.device).long()
        reliability = (
            label_reliability.float().to(logits.device).clamp(0.0, 1.0)
            if label_reliability is not None and label_reliability.shape == logits.shape
            else torch.ones_like(logits, dtype=torch.float32)
        )
        weights = (
            sample_weight.float().to(logits.device).clamp(0.0, 10.0)
            if sample_weight is not None and sample_weight.shape == logits.shape
            else torch.ones_like(logits, dtype=torch.float32)
        )
        pos_unit, neg_unit, _ = self._unreduced_asl_components(logits) if self.loss_type in {"asl", "pu_asl"} else (
            F.binary_cross_entropy_with_logits(logits.float(), torch.ones_like(logits), reduction="none"),
            F.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits), reduction="none"),
            torch.sigmoid(logits.float()),
        )
        if self.pos_weight is not None:
            pos_unit = pos_unit * self.pos_weight.to(pos_unit.device)

        # State ids: unknown=0, positive=1, observed_negative=2, reliable_negative=3, soft_positive=4, conflict=5.
        pos_mask = (state == 1) | ((state == 0) & (targets == 1.0))
        observed_neg_mask = (state == 2) | ((state == 0) & (targets == 0.0))
        reliable_neg_mask = state == 3
        soft_pos_mask = state == 4
        conflict_mask = state == 5
        unknown_mask = (state == 0) & (targets < 0.0)

        soft_target = torch.full_like(logits, self.state_soft_positive_target)
        soft_pos_loss = F.binary_cross_entropy_with_logits(logits.float(), soft_target, reduction="none")

        loss = torch.zeros_like(logits, dtype=torch.float32)
        item_weight = torch.zeros_like(logits, dtype=torch.float32)
        loss = torch.where(pos_mask, pos_unit, loss)
        item_weight = torch.where(pos_mask, weights * reliability, item_weight)
        loss = torch.where(observed_neg_mask, neg_unit, loss)
        item_weight = torch.where(observed_neg_mask, weights * reliability * self.state_observed_negative_weight, item_weight)
        loss = torch.where(reliable_neg_mask, neg_unit, loss)
        item_weight = torch.where(reliable_neg_mask, weights * reliability * self.state_reliable_negative_weight, item_weight)
        loss = torch.where(soft_pos_mask, soft_pos_loss, loss)
        item_weight = torch.where(soft_pos_mask, weights * reliability * self.state_soft_positive_weight, item_weight)
        if self.state_unknown_weight > 0.0:
            loss = torch.where(unknown_mask, neg_unit, loss)
            item_weight = torch.where(unknown_mask, weights * reliability * self.state_unknown_weight, item_weight)
        if self.state_conflict_weight > 0.0:
            conflict_target = targets.clamp(0.0, 1.0)
            conflict_loss = F.binary_cross_entropy_with_logits(logits.float(), conflict_target, reduction="none")
            loss = torch.where(conflict_mask, conflict_loss, loss)
            item_weight = torch.where(conflict_mask, weights * reliability * self.state_conflict_weight, item_weight)

        task_losses = (loss * item_weight).sum(dim=0) / item_weight.sum(dim=0).clamp(min=1.0)
        return task_losses

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        label_state: Optional[torch.Tensor] = None,
        label_reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        known_mask = (targets >= 0.0).float()
        if self.use_label_state_loss and label_state is not None and label_state.shape == logits.shape:
            task_losses = self._state_aware_task_losses(logits, targets, sample_weight, label_state, label_reliability)
            weighted = task_losses * self.task_weights.to(task_losses.device)
            if self.use_homoscedastic and self.log_vars is not None:
                effective_log_vars = self._effective_log_vars()
                precision = torch.exp(-effective_log_vars)
                total = torch.sum(precision * weighted + effective_log_vars)
            else:
                total = weighted.mean()
            if self.use_rank_loss and self.rank_loss_weight > 0:
                total = total + self.rank_loss_weight * self._pairwise_ranking_loss(logits, targets, margin=self.rank_loss_margin)
            self.last_task_losses = task_losses.detach()
            return total
        safe_targets = targets.clamp(min=0.0, max=1.0)
        if self.loss_type in {"asl", "pu_asl"}:
            per_item = self._unreduced_asl(logits, safe_targets)
        else:
            per_item = self._unreduced_bce(logits, safe_targets)
        if sample_weight is not None:
            weights = sample_weight.float().to(per_item.device)
            if weights.shape == per_item.shape:
                per_item = per_item * torch.clamp(weights, min=0.0, max=10.0)

        if self.loss_type == "pu_asl" and bool(torch.any(self.pu_task_mask > 0)):
            pos_unit, neg_unit, _ = self._unreduced_asl_components(logits)
            if self.pos_weight is not None:
                pos_unit = pos_unit * self.pos_weight.to(pos_unit.device)
            if sample_weight is not None:
                weights = sample_weight.float().to(pos_unit.device)
                if weights.shape == pos_unit.shape:
                    weights = torch.clamp(weights, min=0.0, max=10.0)
                    pos_unit = pos_unit * weights
                    neg_unit = neg_unit * weights

            task_loss_values: List[torch.Tensor] = []
            pu_task_mask = self.pu_task_mask.to(logits.device)
            priors = self.pu_positive_priors.to(logits.device)
            for task_idx in range(logits.shape[1]):
                if task_idx < int(pu_task_mask.numel()) and float(pu_task_mask[task_idx].item()) > 0.0:
                    pos_mask = targets[:, task_idx] == 1.0
                    neg_mask = targets[:, task_idx] == 0.0
                    unlabeled_mask = targets[:, task_idx] < 0.0

                    zero = logits[:, task_idx].sum() * 0.0
                    pi = priors[task_idx].clamp(min=1e-4, max=0.95)
                    pos_risk = pi * pos_unit[pos_mask, task_idx].mean() if bool(pos_mask.any()) else zero
                    pos_as_neg_risk = pi * neg_unit[pos_mask, task_idx].mean() if bool(pos_mask.any()) else zero
                    if bool(unlabeled_mask.any()):
                        unlabeled_neg_risk = neg_unit[unlabeled_mask, task_idx].mean()
                    elif bool(neg_mask.any()):
                        unlabeled_neg_risk = neg_unit[neg_mask, task_idx].mean()
                    else:
                        unlabeled_neg_risk = zero
                    nn_neg_risk = unlabeled_neg_risk - pos_as_neg_risk
                    if self.pu_nn_beta > 0.0:
                        nn_neg_risk = torch.clamp(nn_neg_risk, min=-self.pu_nn_beta)
                    nn_neg_risk = torch.clamp(nn_neg_risk, min=0.0)
                    reliable_neg_risk = (
                        neg_unit[neg_mask, task_idx].mean() * self.pu_reliable_negative_weight
                        if bool(neg_mask.any())
                        else zero
                    )
                    task_loss_values.append(pos_risk + self.pu_unlabeled_weight * nn_neg_risk + reliable_neg_risk)
                else:
                    known = known_mask[:, task_idx]
                    task_loss_values.append((per_item[:, task_idx] * known).sum() / known.sum().clamp(min=1.0))
            task_losses = torch.stack(task_loss_values)
        else:
            per_item = per_item * known_mask
            task_den = known_mask.sum(dim=0).clamp(min=1.0)
            task_losses = per_item.sum(dim=0) / task_den
        
        # --- Problem 2: Mutual Exclusion Penalty (Crosstalk) ---
        # DEPRECATED: In macroscopic samples, multiple modifications occurring 
        # in the same sequence window is normal. 
        # We now allow independent multi-label predictions without any penalty.
        crosstalk_penalty = 0.0

        weighted = task_losses * self.task_weights.to(task_losses.device)
        if self.use_homoscedastic and self.log_vars is not None:
            effective_log_vars = self._effective_log_vars()
            precision = torch.exp(-effective_log_vars)
            total = torch.sum(precision * weighted + effective_log_vars)
        else:
            total = weighted.mean()
            
        if self.use_rank_loss and self.rank_loss_weight > 0:
            rank_loss = self._pairwise_ranking_loss(logits, targets, margin=self.rank_loss_margin)
            total = total + self.rank_loss_weight * rank_loss
            
        self.last_task_losses = task_losses.detach()
        return total
