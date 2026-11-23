from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        rank = max(8, min(int(rank), int(dim)))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PTMDisentangledSeparator(nn.Module):
    """Build task-private PTM features from shared and LaST task context.

    The separator returns one private feature and one logit per PTM task. It also
    emits auxiliary losses that encourage task-private features to separate in
    representation space while preserving intra-task positive compactness.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        num_tasks: int,
        task_names: List[str],
        context_dim: Optional[int] = None,
        specialist_dim: int = 0,
        adapter_rank: int = 64,
        dropout: float = 0.1,
        cka_lambda: float = 0.0,
        cka_target_max: float = 0.75,
        cov_lambda: float = 0.0,
        cov_target_max: float = 0.08,
        orthogonal_lambda: float = 0.0,
        orthogonal_target_max: float = 0.15,
        metric_lambda: float = 0.0,
        metric_margin: float = 0.25,
        metric_min_positives: int = 2,
        ranking_lambda: float = 0.0,
        ranking_margin: float = 0.2,
        ranking_target_tasks: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.base_dim = int(base_dim)
        self.num_tasks = int(num_tasks)
        self.task_names = [str(name).lower() for name in task_names]
        self.specialist_dim = int(max(0, specialist_dim))

        context_dim = self.base_dim if context_dim is None else int(context_dim)
        self.context_proj = nn.Identity() if context_dim == self.base_dim else nn.Linear(context_dim, self.base_dim)
        self.specialist_proj = (
            nn.Linear(self.specialist_dim, self.base_dim) if self.specialist_dim > 0 else None
        )

        self.task_embeddings = nn.Parameter(torch.empty(self.num_tasks, self.base_dim))
        nn.init.normal_(self.task_embeddings, mean=0.0, std=0.02)

        self.input_norm = nn.LayerNorm(self.base_dim)
        self.adapters = nn.ModuleList(
            [_ResidualAdapter(self.base_dim, adapter_rank, dropout) for _ in range(self.num_tasks)]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.base_dim * 4, self.base_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.base_dim, self.base_dim),
                )
                for _ in range(self.num_tasks)
            ]
        )
        self.output_norms = nn.ModuleList([nn.LayerNorm(self.base_dim) for _ in range(self.num_tasks)])
        self.heads = nn.ModuleList([nn.Linear(self.base_dim, 1) for _ in range(self.num_tasks)])

        self.cka_lambda = float(cka_lambda)
        self.cka_target_max = float(cka_target_max)
        self.cov_lambda = float(cov_lambda)
        self.cov_target_max = float(cov_target_max)
        self.orthogonal_lambda = float(orthogonal_lambda)
        self.orthogonal_target_max = float(orthogonal_target_max)
        self.metric_lambda = float(metric_lambda)
        self.metric_margin = float(metric_margin)
        self.metric_min_positives = max(1, int(metric_min_positives))
        self.ranking_lambda = float(ranking_lambda)
        self.ranking_margin = float(ranking_margin)
        targets = ranking_target_tasks or []
        self.ranking_target_tasks = {str(name).lower() for name in targets}

    @staticmethod
    def _offdiag_mask(task_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(task_count, device=device, dtype=dtype)
        return 1.0 - eye

    @staticmethod
    def _linear_cka_matrix(features: torch.Tensor) -> torch.Tensor:
        task_count = int(features.shape[1])
        out = features.new_zeros((task_count, task_count), dtype=torch.float32)
        centered = features.float() - features.float().mean(dim=0, keepdim=True)
        for i in range(task_count):
            xi = centered[:, i, :]
            k_xx = torch.matmul(xi.t(), xi)
            norm_x = (k_xx * k_xx).sum().clamp(min=1e-12)
            for j in range(i, task_count):
                xj = centered[:, j, :]
                k_yy = torch.matmul(xj.t(), xj)
                cross = torch.matmul(xi.t(), xj)
                denom = torch.sqrt(norm_x * (k_yy * k_yy).sum().clamp(min=1e-12))
                value = (cross * cross).sum() / denom.clamp(min=1e-12)
                out[i, j] = value
                out[j, i] = value
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _covariance_matrix(features: torch.Tensor) -> torch.Tensor:
        task_count = int(features.shape[1])
        out = features.new_zeros((task_count, task_count), dtype=torch.float32)
        centered = features.float() - features.float().mean(dim=0, keepdim=True)
        for i in range(task_count):
            xi = centered[:, i, :]
            for j in range(i, task_count):
                xj = centered[:, j, :]
                value = (xi * xj).mean(dim=0).abs().mean()
                out[i, j] = value
                out[j, i] = value
        return out.clamp(min=0.0)

    def _align_task_context(self, task_context: Optional[torch.Tensor], shared: torch.Tensor) -> torch.Tensor:
        bsz = int(shared.shape[0])
        if not isinstance(task_context, torch.Tensor):
            return shared.new_zeros((bsz, self.num_tasks, self.base_dim))
        ctx = task_context.to(device=shared.device, dtype=shared.dtype)
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(1).expand(-1, self.num_tasks, -1)
        if ctx.shape[1] != self.num_tasks:
            if ctx.shape[1] > self.num_tasks:
                ctx = ctx[:, : self.num_tasks, :]
            else:
                pad = ctx.new_zeros((ctx.shape[0], self.num_tasks - ctx.shape[1], ctx.shape[2]))
                ctx = torch.cat([ctx, pad], dim=1)
        return self.context_proj(ctx)

    def _metric_losses(
        self,
        private_features: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            return None, None, {}
        y = labels.to(device=private_features.device).float()
        if y.shape[1] != self.num_tasks:
            return None, None, {}

        features = F.normalize(private_features.float(), dim=-1)
        metric_terms = []
        ranking_terms = []
        prototypes = []
        prototype_task_ids = []
        positive_counts = private_features.new_zeros((self.num_tasks,), dtype=torch.float32)
        metric_active_tasks = 0
        ranking_active_tasks = 0

        for task_idx in range(self.num_tasks):
            known_pos = y[:, task_idx] > 0.5
            known_neg = y[:, task_idx] == 0.0
            pos_count = int(known_pos.sum().detach().item())
            neg_count = int(known_neg.sum().detach().item())
            positive_counts[task_idx] = float(pos_count)
            task_features = features[:, task_idx, :]
            if pos_count >= self.metric_min_positives:
                metric_active_tasks += 1
                proto = F.normalize(task_features[known_pos].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                prototypes.append(proto)
                prototype_task_ids.append(task_idx)
                pos_pull = 1.0 - (task_features[known_pos] * proto).sum(dim=-1)
                metric_terms.append(pos_pull.mean())
                if neg_count > 0:
                    neg_sim = (task_features[known_neg] * proto).sum(dim=-1)
                    metric_terms.append(F.relu(neg_sim + self.metric_margin).mean())

            should_rank = not self.ranking_target_tasks or self.task_names[task_idx] in self.ranking_target_tasks
            if should_rank and pos_count > 0 and neg_count > 0:
                ranking_active_tasks += 1
                pos_logits = logits[known_pos, task_idx].float()
                neg_logits = logits[known_neg, task_idx].float()
                pair_loss = F.relu(self.ranking_margin - (pos_logits[:, None] - neg_logits[None, :]))
                ranking_terms.append(pair_loss.mean())

        if len(prototypes) > 1:
            proto_tensor = torch.stack(prototypes, dim=0)
            proto_sim = torch.matmul(proto_tensor, proto_tensor.t())
            offdiag = self._offdiag_mask(proto_tensor.shape[0], proto_tensor.device, proto_tensor.dtype)
            metric_terms.append((F.relu(proto_sim - self.orthogonal_target_max) * offdiag).sum() / offdiag.sum().clamp(min=1.0))

        metric_logs = {
            "separator_metric_positive_counts": positive_counts.detach(),
            "separator_metric_active_tasks": private_features.new_tensor(float(metric_active_tasks)),
            "separator_ranking_active_tasks": private_features.new_tensor(float(ranking_active_tasks)),
            "separator_metric_min_positives": private_features.new_tensor(float(self.metric_min_positives)),
        }
        metric_loss = torch.stack(metric_terms).mean() if metric_terms else None
        ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else None
        return metric_loss, ranking_loss, metric_logs

    def _specialist_tensor(
        self,
        specialist_features: Optional[Dict[str, torch.Tensor]],
        shared: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.specialist_proj is None:
            return None
        bsz = int(shared.shape[0])
        spec = shared.new_zeros((bsz, self.num_tasks, self.specialist_dim))
        if isinstance(specialist_features, dict):
            for task_idx, name in enumerate(self.task_names):
                value = specialist_features.get(name)
                if isinstance(value, torch.Tensor):
                    value = value.to(device=shared.device, dtype=shared.dtype)
                    width = min(value.shape[-1], self.specialist_dim)
                    spec[:, task_idx, :width] = value[:, :width]
        return self.specialist_proj(spec)

    def forward(
        self,
        shared: torch.Tensor,
        *,
        task_context: Optional[torch.Tensor] = None,
        specialist_features: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        shared = shared.float()
        bsz = int(shared.shape[0])
        shared_task = shared.unsqueeze(1).expand(-1, self.num_tasks, -1)
        context = self._align_task_context(task_context, shared)
        prompt = self.task_embeddings.unsqueeze(0).expand(bsz, -1, -1)
        specialist = self._specialist_tensor(specialist_features, shared)
        if specialist is None:
            specialist = shared.new_zeros((bsz, self.num_tasks, self.base_dim))

        seed = self.input_norm(shared_task + context + prompt + specialist)
        private_features = []
        logits = []
        gate_values = []
        for task_idx in range(self.num_tasks):
            private = self.adapters[task_idx](seed[:, task_idx, :])
            gate_input = torch.cat(
                [
                    shared_task[:, task_idx, :],
                    context[:, task_idx, :],
                    prompt[:, task_idx, :],
                    private,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.gates[task_idx](gate_input))
            out = self.output_norms[task_idx](shared + gate * private + 0.2 * context[:, task_idx, :])
            private_features.append(out)
            logits.append(self.heads[task_idx](out).squeeze(-1))
            gate_values.append(gate.mean(dim=-1))

        private_tensor = torch.stack(private_features, dim=1)
        logits_tensor = torch.stack(logits, dim=1)
        gates_tensor = torch.stack(gate_values, dim=1)

        aux_losses: Dict[str, torch.Tensor] = {}
        aux_logs: Dict[str, torch.Tensor] = {}
        contributions: Dict[str, torch.Tensor] = {
            "separator_gate_mean_per_task": gates_tensor.detach().mean(dim=0),
            "separator_private_norm_per_task": private_tensor.detach().norm(dim=-1).mean(dim=0),
        }

        if self.num_tasks > 1:
            offdiag = self._offdiag_mask(self.num_tasks, private_tensor.device, private_tensor.dtype)
            cka = self._linear_cka_matrix(private_tensor)
            cka_offdiag = (cka * offdiag).sum() / offdiag.sum().clamp(min=1.0)
            contributions["separator_cka_matrix"] = cka.detach()
            contributions["separator_cka_offdiag_mean"] = cka_offdiag.detach().view(1)
            if self.cka_lambda > 0:
                aux_losses["separator_cka"] = (
                    F.relu(cka - self.cka_target_max).mul(offdiag).sum() / offdiag.sum().clamp(min=1.0)
                ) * self.cka_lambda
                aux_logs["separator_cka"] = cka_offdiag.detach()

            cov = self._covariance_matrix(private_tensor)
            cov_offdiag = (cov * offdiag).sum() / offdiag.sum().clamp(min=1.0)
            contributions["separator_cov_matrix"] = cov.detach()
            contributions["separator_cov_offdiag_mean"] = cov_offdiag.detach().view(1)
            if self.cov_lambda > 0:
                aux_losses["separator_cov"] = (
                    F.relu(cov - self.cov_target_max).mul(offdiag).sum() / offdiag.sum().clamp(min=1.0)
                ) * self.cov_lambda
                aux_logs["separator_cov"] = cov_offdiag.detach()

            prompt_norm = F.normalize(self.task_embeddings.float(), dim=-1)
            prompt_sim = torch.matmul(prompt_norm, prompt_norm.t()).abs()
            prompt_offdiag = (prompt_sim * offdiag).sum() / offdiag.sum().clamp(min=1.0)
            contributions["separator_prompt_cosine_offdiag_mean"] = prompt_offdiag.detach().view(1)
            if self.orthogonal_lambda > 0:
                aux_losses["separator_prompt_orthogonal"] = (
                    F.relu(prompt_sim - self.orthogonal_target_max).mul(offdiag).sum()
                    / offdiag.sum().clamp(min=1.0)
                ) * self.orthogonal_lambda
                aux_logs["separator_prompt_orthogonal"] = prompt_offdiag.detach()

        metric_loss, ranking_loss, metric_logs = self._metric_losses(private_tensor, logits_tensor, labels)
        if metric_logs:
            aux_logs.update(metric_logs)
            contributions.update(metric_logs)
        if metric_loss is not None and self.metric_lambda > 0:
            aux_losses["separator_metric"] = metric_loss * self.metric_lambda
            aux_logs["separator_metric"] = metric_loss.detach()
        if ranking_loss is not None and self.ranking_lambda > 0:
            aux_losses["separator_ranking"] = ranking_loss * self.ranking_lambda
            aux_logs["separator_ranking"] = ranking_loss.detach()

        return {
            "logits": logits_tensor,
            "private_features": private_tensor.to(dtype=shared.dtype),
            "auxiliary_losses": aux_losses,
            "auxiliary_logs": aux_logs,
            "module_contributions": contributions,
        }
