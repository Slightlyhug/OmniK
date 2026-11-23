from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TaskDescription:
    task_names: List[str]
    num_tasks: int
    sequence_length: int
    sequence_dim: int
    has_structure_context: bool
    has_graph_data: bool
    apply_to_structure_graph: bool
    device: torch.device
    dtype: torch.dtype


@dataclass
class TaskContext:
    task_description: TaskDescription
    sequence_enhanced: torch.Tensor
    task_repr: torch.Tensor
    projected_task_context: Optional[torch.Tensor]
    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    auxiliary_logs: Dict[str, torch.Tensor] = field(default_factory=dict)
    module_contributions: Dict[str, torch.Tensor] = field(default_factory=dict)
    raw_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureGraphContext:
    task_description: TaskDescription
    task_context: Optional[TaskContext]
    graph_data: Optional[Dict[str, Any]]
    output_dtype: torch.dtype
    device: torch.device


@dataclass
class StructureGraphResult:
    structure_global: Optional[torch.Tensor]
    structure_global_raw: Optional[torch.Tensor]
    module_contributions: Dict[str, torch.Tensor] = field(default_factory=dict)
    node_embeddings: Optional[torch.Tensor] = None
    node_batch_index: Optional[torch.Tensor] = None
    node_token_index: Optional[torch.Tensor] = None


class AbstractTaskParser(ABC):
    @abstractmethod
    def parse(
        self,
        *,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> TaskDescription:
        raise NotImplementedError


class AbstractTaskContextGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        task_description: TaskDescription,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
    ) -> TaskContext:
        raise NotImplementedError


class AbstractStructureGraphBuilder(ABC):
    @abstractmethod
    def build(self, context: StructureGraphContext) -> StructureGraphResult:
        raise NotImplementedError


class ConfigDrivenTaskParser(AbstractTaskParser):
    def __init__(self, task_names: List[str], apply_to_structure_graph: bool = False) -> None:
        self.task_names = [str(name) for name in task_names]
        self.apply_to_structure_graph = bool(apply_to_structure_graph)

    def parse(
        self,
        *,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> TaskDescription:
        _ = seq_mask
        return TaskDescription(
            task_names=list(self.task_names),
            num_tasks=len(self.task_names),
            sequence_length=int(sequence_hidden.shape[1]),
            sequence_dim=int(sequence_hidden.shape[2]),
            has_structure_context=isinstance(structure_hidden, torch.Tensor),
            has_graph_data=isinstance(graph_data, dict),
            apply_to_structure_graph=bool(self.apply_to_structure_graph),
            device=sequence_hidden.device,
            dtype=sequence_hidden.dtype,
        )




class TaskSpecificContextProjector(nn.Module):
    """Project LaST task representations while preserving task identity.

    A shared linear projector alone maps nearly-identical task representations to
    nearly-identical contexts. This module adds three anti-collapse operations:
    task embeddings, per-task low-rank residual adapters, and optional removal of
    the across-task shared component.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tasks: int,
        adapter_rank: int = 64,
        dropout: float = 0.05,
        task_embed_scale: float = 0.25,
        remove_shared_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tasks = max(1, int(num_tasks))
        self.task_embed_scale = float(task_embed_scale)
        self.remove_shared_scale = float(remove_shared_scale)
        rank = max(4, min(int(adapter_rank), self.output_dim))

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.task_embeddings = nn.Parameter(torch.empty(self.num_tasks, self.output_dim))
        if self.output_dim >= self.num_tasks:
            nn.init.orthogonal_(self.task_embeddings)
        else:
            nn.init.normal_(self.task_embeddings, mean=0.0, std=0.02)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.output_dim),
                    nn.Linear(self.output_dim, rank, bias=False),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(rank, self.output_dim, bias=False),
                )
                for _ in range(self.num_tasks)
            ]
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, task_repr: torch.Tensor) -> torch.Tensor:
        if task_repr.ndim != 3:
            return self.shared(task_repr)
        base = self.shared(task_repr)
        task_count = min(int(base.shape[1]), self.num_tasks)
        pieces = []
        for idx in range(task_count):
            task_base = base[:, idx, :]
            task_delta = self.adapters[idx](task_base)
            task_embed = self.task_embeddings[idx].to(device=base.device, dtype=base.dtype).view(1, -1)
            pieces.append(task_base + task_delta + self.task_embed_scale * task_embed)
        if task_count < int(base.shape[1]):
            pieces.append(base[:, task_count:, :])
        out = torch.cat([p.unsqueeze(1) if p.ndim == 2 else p for p in pieces], dim=1)
        if self.remove_shared_scale > 0.0 and out.shape[1] > 1:
            shared = out.mean(dim=1, keepdim=True)
            out = out - self.remove_shared_scale * shared
        return self.output_norm(out)

class LaSTTaskContextGenerator(AbstractTaskContextGenerator):
    def __init__(
        self,
        *,
        foreground_block: nn.Module,
        task_context_projector: Optional[nn.Module],
        mask_sparsity_lambda: float,
        mask_entropy_lambda: float,
        mask_diversity_lambda: float,
        mask_iou_lambda: float,
        task_repr_contrastive_lambda: float,
        context_cka_lambda: float = 0.0,
        context_cov_lambda: float = 0.0,
        context_cka_target_max: float = 0.75,
        context_cov_target_max: float = 0.10,
    ) -> None:
        self.foreground_block = foreground_block
        self.task_context_projector = task_context_projector
        self.mask_sparsity_lambda = float(mask_sparsity_lambda)
        self.mask_entropy_lambda = float(mask_entropy_lambda)
        self.mask_diversity_lambda = float(mask_diversity_lambda)
        self.mask_iou_lambda = float(mask_iou_lambda)
        self.task_repr_contrastive_lambda = float(task_repr_contrastive_lambda)
        self.context_cka_lambda = float(context_cka_lambda)
        self.context_cov_lambda = float(context_cov_lambda)
        self.context_cka_target_max = float(context_cka_target_max)
        self.context_cov_target_max = float(context_cov_target_max)
        self.prompt_orthogonal_lambda = float(getattr(foreground_block, "prompt_orthogonal_lambda", 0.0))
        self.prompt_mi_lambda = float(getattr(foreground_block, "prompt_mi_lambda", 0.0))
        self.attention_sparse_lambda = float(getattr(foreground_block, "attention_sparse_lambda", 0.0))
        self.attention_entropy_target = float(getattr(foreground_block, "attention_entropy_target", -1.0))
        self.attention_decorrelation_lambda = float(getattr(foreground_block, "attention_decorrelation_lambda", 0.0))
        self.attention_tau_lambda = float(getattr(foreground_block, "attention_tau_lambda", 0.0))
        self.attention_focus_lambda = float(getattr(foreground_block, "attention_focus_lambda", 0.0))
        self.token_prior_diversity_lambda = float(getattr(foreground_block, "token_prior_diversity_lambda", 0.0))
        self.foreground_attention_consistency_lambda = float(
            getattr(foreground_block, "foreground_attention_consistency_lambda", 0.0)
        )

    @staticmethod
    def _linear_cka_matrix(features: torch.Tensor) -> torch.Tensor:
        task_count = int(features.shape[1])
        cka = features.new_zeros((task_count, task_count), dtype=torch.float32)
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
                cka[i, j] = value
                cka[j, i] = value
        return cka.clamp(0.0, 1.0)

    @staticmethod
    def _task_covariance_matrix(features: torch.Tensor) -> torch.Tensor:
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

    def generate(
        self,
        *,
        task_description: TaskDescription,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
    ) -> TaskContext:
        last_out = self.foreground_block(
            sequence_hidden=sequence_hidden,
            seq_mask=seq_mask,
            structure_hidden=structure_hidden,
        )
        sequence_enhanced = last_out["sequence_enhanced"]
        task_repr = last_out["task_repr"]
        projected_task_context = self.task_context_projector(task_repr) if self.task_context_projector is not None else None

        auxiliary_losses: Dict[str, torch.Tensor] = {}
        auxiliary_logs: Dict[str, torch.Tensor] = {}
        module_contributions: Dict[str, torch.Tensor] = {
            "last_mask_mean_per_task": last_out["mask_mean_per_task"].detach(),
            "last_mask_center_to_bg_ratio": last_out["mask_center_to_bg_ratio"].detach(),
        }

        if self.mask_sparsity_lambda > 0:
            fg_sparse = last_out["fg_sparse_loss"]
            auxiliary_losses["last_fg_sparse"] = fg_sparse * self.mask_sparsity_lambda
            auxiliary_logs["last_fg_sparse"] = fg_sparse.detach()
        if self.mask_entropy_lambda > 0:
            fg_entropy = last_out["fg_entropy_loss"]
            auxiliary_losses["last_fg_entropy"] = fg_entropy * self.mask_entropy_lambda
            auxiliary_logs["last_fg_entropy"] = fg_entropy.detach()
            if "mask_mean_loss" in last_out:
                auxiliary_losses["last_fg_mask_mean"] = last_out["mask_mean_loss"] * max(0.2, self.mask_entropy_lambda)
                auxiliary_logs["last_fg_mask_mean"] = last_out["mask_mean_loss"].detach()
            # Bipolar loss to push mask away from 0.5
            if "fg_bipolar_loss" in last_out:
                auxiliary_losses["last_fg_bipolar"] = last_out["fg_bipolar_loss"] * max(0.1, self.mask_entropy_lambda * 0.5)
                auxiliary_logs["last_fg_bipolar"] = last_out["fg_bipolar_loss"].detach()
            # Mask variance loss to push different tasks to have different masks
            if "fg_mask_variance_loss" in last_out:
                auxiliary_losses["last_fg_mask_variance"] = last_out["fg_mask_variance_loss"] * max(0.2, self.mask_entropy_lambda)
                auxiliary_logs["last_fg_mask_variance"] = last_out["fg_mask_variance_loss"].detach()
        if self.mask_diversity_lambda > 0:
            fg_diversity = last_out["fg_task_diversity_loss"]
            auxiliary_losses["last_fg_diversity"] = fg_diversity * self.mask_diversity_lambda
            auxiliary_logs["last_fg_diversity"] = fg_diversity.detach()
        if self.mask_iou_lambda > 0:
            fg_iou = last_out["fg_mask_iou_loss"]
            auxiliary_losses["last_fg_iou"] = fg_iou * self.mask_iou_lambda
            auxiliary_logs["last_fg_iou"] = last_out["fg_mask_iou_mean"].detach()
        if self.task_repr_contrastive_lambda > 0:
            repr_loss = last_out["task_repr_contrastive_loss"]
            auxiliary_losses["last_task_repr_contrastive"] = repr_loss * self.task_repr_contrastive_lambda
            auxiliary_logs["last_task_repr_inter_similarity"] = last_out["task_repr_inter_similarity"].detach()
        if self.prompt_orthogonal_lambda > 0 and "prompt_orthogonal_loss" in last_out:
            auxiliary_losses["last_prompt_orthogonal"] = last_out["prompt_orthogonal_loss"] * self.prompt_orthogonal_lambda
            auxiliary_logs["last_prompt_orthogonal"] = last_out["prompt_orthogonal_loss"].detach()
        if self.attention_decorrelation_lambda > 0 and "attention_decorrelation_loss" in last_out:
            auxiliary_losses["last_attention_decorrelation"] = last_out["attention_decorrelation_loss"] * self.attention_decorrelation_lambda
            auxiliary_logs["last_attention_decorrelation"] = last_out["attention_decorrelation_loss"].detach()
        if self.token_prior_diversity_lambda > 0 and "token_prior_diversity_loss" in last_out:
            auxiliary_losses["last_token_prior_diversity"] = (
                last_out["token_prior_diversity_loss"] * self.token_prior_diversity_lambda
            )
            auxiliary_logs["last_token_prior_diversity"] = last_out["token_prior_diversity_loss"].detach()
        if self.foreground_attention_consistency_lambda > 0 and "foreground_attention_consistency_loss" in last_out:
            auxiliary_losses["last_foreground_attention_consistency"] = (
                last_out["foreground_attention_consistency_loss"] * self.foreground_attention_consistency_lambda
            )
            auxiliary_logs["last_foreground_attention_consistency"] = last_out["foreground_attention_consistency_loss"].detach()
        if self.prompt_mi_lambda > 0 and "task_mutual_information_loss" in last_out:
            auxiliary_losses["last_task_mutual_information"] = last_out["task_mutual_information_loss"] * self.prompt_mi_lambda
            auxiliary_logs["last_task_mutual_information"] = last_out["task_mutual_information_loss"].detach()
        if self.attention_sparse_lambda > 0 and "attention_sparse_loss" in last_out:
            auxiliary_losses["last_attention_sparse"] = last_out["attention_sparse_loss"] * self.attention_sparse_lambda
            auxiliary_logs["last_attention_sparse"] = last_out["attention_sparse_loss"].detach()
        if self.attention_tau_lambda > 0 and "attention_tau_loss" in last_out:
            auxiliary_losses["last_attention_tau"] = last_out["attention_tau_loss"] * self.attention_tau_lambda
            auxiliary_logs["last_attention_tau"] = last_out["attention_tau_loss"].detach()
        if self.attention_focus_lambda > 0 and "attention_focus_loss" in last_out:
            auxiliary_losses["last_attention_focus"] = last_out["attention_focus_loss"] * self.attention_focus_lambda
            auxiliary_logs["last_attention_focus"] = last_out["attention_focus_loss"].detach()
        if self.attention_entropy_target >= 0.0 and "attention_entropy_loss" in last_out:
            auxiliary_losses["last_attention_entropy"] = last_out["attention_entropy_loss"]
            auxiliary_logs["last_attention_entropy"] = last_out["attention_entropy_loss"].detach()

        if "fg_mask_iou_mean" in last_out:
            module_contributions["last_mask_iou_mean"] = last_out["fg_mask_iou_mean"].detach().view(1)
        if "task_repr_inter_similarity" in last_out:
            module_contributions["last_task_inter_similarity"] = last_out["task_repr_inter_similarity"].detach().view(1)
        if "last_stability_base_mean" in last_out:
            module_contributions["last_stability_base_mean"] = last_out["last_stability_base_mean"].detach()
        if "last_stability_base_peak" in last_out:
            module_contributions["last_stability_base_peak"] = last_out["last_stability_base_peak"].detach()
        if "ptm_attention_peak_per_task" in last_out:
            module_contributions["ptm_attention_peak_per_task"] = last_out["ptm_attention_peak_per_task"].detach()
        if "ptm_attention_entropy_per_task" in last_out:
            module_contributions["ptm_attention_entropy_per_task"] = last_out["ptm_attention_entropy_per_task"].detach()
        if "ptm_attention_focus_ratio_per_task" in last_out:
            module_contributions["ptm_attention_focus_ratio_per_task"] = last_out["ptm_attention_focus_ratio_per_task"].detach()
        if "task_prompt_cosine_offdiag_mean" in last_out:
            module_contributions["task_prompt_cosine_offdiag_mean"] = last_out["task_prompt_cosine_offdiag_mean"].detach().view(1)
        if "task_mutual_information_proxy" in last_out:
            module_contributions["task_mutual_information_proxy"] = last_out["task_mutual_information_proxy"].detach().view(1)
        if "token_prior_peak_per_task" in last_out:
            module_contributions["token_prior_peak_per_task"] = last_out["token_prior_peak_per_task"].detach()
        if "token_prior_mass_per_task" in last_out:
            module_contributions["token_prior_mass_per_task"] = last_out["token_prior_mass_per_task"].detach()
        if "token_prior_offdiag_mean" in last_out:
            module_contributions["token_prior_offdiag_mean"] = last_out["token_prior_offdiag_mean"].detach().view(1)
        if "foreground_attention_kl" in last_out:
            module_contributions["foreground_attention_kl"] = last_out["foreground_attention_kl"].detach().view(1)
        if projected_task_context is not None:
            module_contributions["last_task_context_norm"] = projected_task_context.detach().norm(dim=-1).mean(dim=0)
            if projected_task_context.ndim == 3 and int(projected_task_context.shape[1]) > 1:
                task_count = int(projected_task_context.shape[1])
                eye = torch.eye(task_count, device=projected_task_context.device, dtype=projected_task_context.dtype)
                offdiag = 1.0 - eye
                context_cka_matrix = self._linear_cka_matrix(projected_task_context)
                context_cka_offdiag = (context_cka_matrix * offdiag).sum() / offdiag.sum().clamp(min=1.0)
                context_cov_matrix = self._task_covariance_matrix(projected_task_context)
                context_cov_offdiag = (context_cov_matrix * offdiag).sum() / offdiag.sum().clamp(min=1.0)
                module_contributions["last_task_context_cka_matrix"] = context_cka_matrix.detach()
                module_contributions["last_task_context_cka_offdiag_mean"] = context_cka_offdiag.detach().view(1)
                module_contributions["last_task_context_cov_matrix"] = context_cov_matrix.detach()
                module_contributions["last_task_context_cov_offdiag_mean"] = context_cov_offdiag.detach().view(1)
                if self.context_cka_lambda > 0:
                    context_cka_loss = torch.relu(context_cka_matrix - self.context_cka_target_max) * offdiag
                    auxiliary_losses["last_task_context_cka"] = context_cka_loss.sum() / offdiag.sum().clamp(min=1.0) * self.context_cka_lambda
                    auxiliary_logs["last_task_context_cka"] = context_cka_offdiag.detach()
                if self.context_cov_lambda > 0:
                    context_cov_loss = torch.relu(context_cov_matrix - self.context_cov_target_max) * offdiag
                    auxiliary_losses["last_task_context_cov"] = context_cov_loss.sum() / offdiag.sum().clamp(min=1.0) * self.context_cov_lambda
                    auxiliary_logs["last_task_context_cov"] = context_cov_offdiag.detach()
            
        if "adv_task_logits" in last_out:
            # We want uniform distribution (entropy maximization) over tasks from the discriminator
            adv_logits = last_out["adv_task_logits"]
            adv_prob = F.softmax(adv_logits, dim=-1)
            uniform_prob = torch.ones_like(adv_prob) / adv_prob.shape[-1]
            adv_loss = F.kl_div(F.log_softmax(adv_logits, dim=-1), uniform_prob, reduction="batchmean")
            if "auxiliary_losses" not in locals():
                auxiliary_losses = {}
            auxiliary_losses["shared_adversarial_entropy"] = adv_loss
            if "auxiliary_logs" not in locals():
                auxiliary_logs = {}
            auxiliary_logs["shared_adversarial_entropy"] = adv_loss.detach()

        return TaskContext(
            task_description=task_description,
            sequence_enhanced=sequence_enhanced,
            task_repr=task_repr,
            projected_task_context=projected_task_context,
            auxiliary_losses=auxiliary_losses,
            auxiliary_logs=auxiliary_logs,
            module_contributions=module_contributions,
            raw_outputs={k: v for k, v in last_out.items() if isinstance(v, torch.Tensor)},
        )


class DefaultStructureGraphBuilder(AbstractStructureGraphBuilder):
    def __init__(
        self,
        *,
        structure_graph_tower: nn.Module,
        graph_vote_mode: str,
        graph_topk_ratio: float,
        graph_sigma: float,
        graph_enhance_scale: float,
        logger: Optional[logging.Logger] = None,
        return_graph_nodes: bool = False,
    ) -> None:
        self.structure_graph_tower = structure_graph_tower
        self.return_graph_nodes = bool(return_graph_nodes)
        self.graph_vote_mode = str(graph_vote_mode).strip().lower()
        self.graph_topk_ratio = float(graph_topk_ratio)
        self.graph_sigma = float(graph_sigma)
        self.graph_enhance_scale = float(graph_enhance_scale)
        self.logger = logger or logging.getLogger(__name__)
        self._graph_node_dim_warning_emitted = False
        self._graph_edge_dim_warning_emitted = False

    def _apply_graph_enhancement(self, x: torch.Tensor, batch_index: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 1:
            return x, {}
        if batch_index.ndim != 1 or batch_index.shape[0] != x.shape[0]:
            return x, {}

        eps = 1e-6
        with torch.amp.autocast(device_type="cuda", enabled=False) if x.is_cuda else contextlib.nullcontext():
            x32 = x.float()
            order = torch.argsort(batch_index)
            x_sorted = x32[order]
            b_sorted = batch_index[order]
            bsz = int(b_sorted.max().item()) + 1 if b_sorted.numel() else 0
            counts = torch.bincount(b_sorted, minlength=max(1, bsz)).tolist() if bsz > 0 else []
            out = x32.clone()

            freq = torch.fft.fftfreq(x32.shape[1], device=x32.device, dtype=torch.float32)
            sigma_safe = max(1e-6, self.graph_sigma)
            lowpass = torch.exp(-0.5 * (freq / sigma_safe) ** 2)
            lowpass = (lowpass - lowpass.min()) / (lowpass.max() - lowpass.min() + 1e-12)
            lowpass = lowpass.to(dtype=torch.complex64)

            mean_scores: List[torch.Tensor] = []
            peak_scores: List[torch.Tensor] = []
            sparsity_scores: List[torch.Tensor] = []
            offset = 0
            for count in counts:
                if count <= 0:
                    continue
                idx = order[offset : offset + count]
                xb = x_sorted[offset : offset + count]
                offset += count

                if xb.shape[0] <= 1:
                    score = xb.new_ones((xb.shape[0],))
                else:
                    x_fft = torch.fft.fft(xb, dim=-1)
                    x_hat = torch.fft.ifft(x_fft * lowpass.view(1, -1), dim=-1).real
                    scores = x_hat / (torch.abs(x_hat - xb) + eps)
                    if self.graph_vote_mode == "channel":
                        k = max(1, min(int(xb.shape[0]), int(round(int(xb.shape[0]) * self.graph_topk_ratio))))
                        idx_top = torch.topk(scores, k=k, dim=0, largest=True, sorted=False).indices
                        idx_flat = idx_top.reshape(-1)
                        votes = torch.zeros((int(xb.shape[0]),), device=xb.device, dtype=torch.float32)
                        votes.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=votes.dtype))
                        score = votes / votes.max().clamp(min=1.0)
                    else:
                        token_score = scores.mean(dim=-1)
                        token_score = (token_score - token_score.mean()) / (token_score.std() + 1e-6)
                        score = torch.sigmoid(token_score)

                out[idx] = xb * (1.0 + self.graph_enhance_scale * score.unsqueeze(-1))
                mean_scores.append(score.mean())
                peak_scores.append(score.max())
                sparsity_scores.append((score < 0.5).float().mean())

            metrics: Dict[str, torch.Tensor] = {}
            if mean_scores:
                metrics["last_graph_score_mean"] = torch.stack(mean_scores).mean().detach()
                metrics["last_graph_score_peak"] = torch.stack(peak_scores).mean().detach()
                metrics["last_graph_score_sparsity"] = torch.stack(sparsity_scores).mean().detach()
            return out.to(dtype=x.dtype), metrics

    def build(self, context: StructureGraphContext) -> StructureGraphResult:
        graph_data = context.graph_data
        if not isinstance(graph_data, dict):
            return StructureGraphResult(structure_global=None, structure_global_raw=None, module_contributions={})

        graph_x = graph_data.get("x")
        graph_pos = graph_data.get("pos")
        graph_esm_if = graph_data.get("esm_if")
        graph_esm_if_mask = graph_data.get("esm_if_mask")
        graph_center_mask = graph_data.get("center_mask")
        edge_index = graph_data.get("edge_index")
        edge_attr = graph_data.get("edge_attr")
        batch_index = graph_data.get("batch_index")
        if graph_x is None or graph_pos is None or edge_index is None or edge_attr is None or batch_index is None:
            return StructureGraphResult(structure_global=None, structure_global_raw=None, module_contributions={})

        graph_x = graph_x.to(context.device)
        graph_pos = graph_pos.to(context.device)
        if graph_esm_if is not None:
            esm_if = graph_esm_if.to(context.device, dtype=graph_x.dtype)
            if graph_esm_if_mask is None:
                esm_mask = (esm_if.abs().sum(dim=-1, keepdim=True) > 0).to(graph_x.dtype)
            else:
                esm_mask = graph_esm_if_mask.to(context.device, dtype=graph_x.dtype).reshape(-1, 1)
            esm_if = esm_if * esm_mask
            graph_x = torch.cat([graph_x, esm_if, esm_mask], dim=-1)
        edge_index = edge_index.to(context.device)
        edge_attr = edge_attr.to(context.device)
        batch_index = batch_index.to(context.device)

        expected_node_dim = int(getattr(self.structure_graph_tower.embed, "in_features", graph_x.shape[-1]))
        if graph_x.ndim == 2 and int(graph_x.shape[-1]) != expected_node_dim:
            if not self._graph_node_dim_warning_emitted:
                self.logger.warning(
                    "Graph node dim mismatch: got=%d expected=%d. Applying pad/truncate before EGNN.",
                    int(graph_x.shape[-1]),
                    expected_node_dim,
                )
                self._graph_node_dim_warning_emitted = True
            if int(graph_x.shape[-1]) > expected_node_dim:
                graph_x = graph_x[:, :expected_node_dim]
            else:
                graph_x = F.pad(graph_x, (0, expected_node_dim - int(graph_x.shape[-1])))

        expected_edge_dim = int(getattr(self.structure_graph_tower, "edge_dim", edge_attr.shape[-1]))
        if edge_attr.ndim == 2 and int(edge_attr.shape[-1]) != expected_edge_dim:
            if not self._graph_edge_dim_warning_emitted:
                self.logger.warning(
                    "Graph edge dim mismatch: got=%d expected=%d. Applying pad/truncate before EGNN.",
                    int(edge_attr.shape[-1]),
                    expected_edge_dim,
                )
                self._graph_edge_dim_warning_emitted = True
            if int(edge_attr.shape[-1]) > expected_edge_dim:
                edge_attr = edge_attr[:, :expected_edge_dim]
            else:
                edge_attr = F.pad(edge_attr, (0, expected_edge_dim - int(edge_attr.shape[-1])))

        if edge_attr.ndim == 2 and edge_index.ndim == 2 and edge_index.shape[0] == 2 and edge_attr.shape[0] != edge_index.shape[1]:
            edge_count = int(edge_index.shape[1])
            if int(edge_attr.shape[0]) > edge_count:
                edge_attr = edge_attr[:edge_count]
            else:
                pad_rows = torch.zeros(
                    (edge_count - int(edge_attr.shape[0]), int(edge_attr.shape[1])),
                    device=edge_attr.device,
                    dtype=edge_attr.dtype,
                )
                edge_attr = torch.cat([edge_attr, pad_rows], dim=0)

        module_contributions: Dict[str, torch.Tensor] = {}
        if context.task_description.apply_to_structure_graph and context.task_context is not None:
            graph_x, graph_metrics = self._apply_graph_enhancement(graph_x, batch_index=batch_index)
            module_contributions.update(graph_metrics)

        node_embeddings = None
        if self.return_graph_nodes:
            structure_global, node_embeddings = self.structure_graph_tower(
                graph_x, graph_pos, edge_index, edge_attr, batch_index,
                center_mask=graph_center_mask, return_nodes=True,
            )
            node_embeddings = node_embeddings.to(dtype=context.output_dtype)
        else:
            structure_global = self.structure_graph_tower(
                graph_x, graph_pos, edge_index, edge_attr, batch_index,
                center_mask=graph_center_mask,
            )
        structure_global = structure_global.to(dtype=context.output_dtype)
        token_index = graph_data.get("token_index")
        if isinstance(token_index, torch.Tensor):
            token_index = token_index.to(device=context.device, dtype=torch.long).reshape(-1)
        else:
            token_index = None
        return StructureGraphResult(
            structure_global=structure_global,
            structure_global_raw=structure_global,
            module_contributions=module_contributions,
            node_embeddings=node_embeddings,
            node_batch_index=batch_index,
            node_token_index=token_index,
        )
