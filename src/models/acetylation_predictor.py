"""PTM dual-branch predictors.

The training pipeline uses :class:`DualBranchFusionPredictor`. Legacy
acetylation-only models live in ``legacy_predictors`` and are loaded lazily
through ``__getattr__`` to keep the production import path lightweight.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_branch_modules import (
    AdaptiveGating,
    CrossModalFusionLayer,
    DepthAttentionResidual,
    ProjectionHead,
    SequenceTower,
    StructureTower,
    masked_mean,
)
from .enhanced_components import (
    EnhancedAttentionModule,
    MultiScaleFusionNetwork,
    DynamicFeatureSelector,
)
from .task_separation import (
    AbstractStructureGraphBuilder,
    AbstractTaskContextGenerator,
    AbstractTaskParser,
    ConfigDrivenTaskParser,
    DefaultStructureGraphBuilder,
    LaSTTaskContextGenerator,
    StructureGraphContext,
    TaskSpecificContextProjector,
)
from .ptm_disentangled_separator import PTMDisentangledSeparator
from .learnable_task_attention import LearnableTaskAttentionPool

try:
    from .esm_lora import ESM2LoRAWrapper
except Exception:  # optional inference/training dependency
    ESM2LoRAWrapper = None

try:
    from .egnn_3d_graph import PocketEGNN
except Exception:  # optional graph dependency
    PocketEGNN = None

try:
    from .se3_transformer_graph import SE3TransformerPocketEncoder
except Exception:  # optional graph dependency
    SE3TransformerPocketEncoder = None

logger = logging.getLogger("acetylation_predictor")


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)


def normalize_sequence_strings(sequence_strings: Any) -> List[str]:
    if sequence_strings is None:
        return []
    if isinstance(sequence_strings, str):
        return [sequence_strings]
    if isinstance(sequence_strings, (list, tuple)):
        return [str(s) if s is not None else "" for s in sequence_strings]
    return [str(sequence_strings)]


def compute_non_empty_sequence_indices(sequence_strings: Any) -> Tuple[List[str], List[int]]:
    seq_list = normalize_sequence_strings(sequence_strings)
    non_empty_idx = [idx for idx, seq in enumerate(seq_list) if len(seq.strip()) > 0]
    return seq_list, non_empty_idx


def merge_non_empty_lora_features(
    base_features: torch.Tensor,
    non_empty_indices: List[int],
    encoded_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if not isinstance(base_features, torch.Tensor):
        raise TypeError("base_features must be a torch.Tensor")
    if encoded_features is None or len(non_empty_indices) == 0:
        return base_features
    merged = base_features.clone()
    for row_idx, base_idx in enumerate(non_empty_indices):
        merged[base_idx] = encoded_features[row_idx]
    return merged


# Keep compatibility imports lazy: legacy evaluation code can still import the
# historical names without forcing optional ESM/plotting dependencies at startup.
_LEGACY_NAMES = {
    "AcetylationPredictor",
    "AblationWrapper",
    "AcetylationPredictionModel",
    "MultiTaskLoss",
    "InterpretabilityModule",
}


def __getattr__(name: str):
    if name in _LEGACY_NAMES:
        from . import legacy_predictors

        return getattr(legacy_predictors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SpectralForegroundBlock(nn.Module):
    """Task-conditioned low-rank spectral foreground selector for PTM sites."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        rank_k: int = 8,
        task_conditioned: bool = True,
        center_prior_weight: float = 0.8,
        access_prior_weight: float = 0.4,
        force_fp32: bool = True,
        sequence_enhance_mode: str = "full",
        detach_backbone_for_mask: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = max(1, int(num_tasks))
        self.rank_k = max(1, int(rank_k))
        self.task_conditioned = bool(task_conditioned)
        self.center_prior_weight = float(center_prior_weight)
        self.access_prior_weight = float(access_prior_weight)
        self.force_fp32 = bool(force_fp32)
        self.sequence_enhance_mode = str(sequence_enhance_mode).strip().lower()
        self.detach_backbone_for_mask = bool(detach_backbone_for_mask)

        hidden_k = max(8, self.rank_k * 2)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(self.rank_k, hidden_k),
            nn.GELU(),
            nn.Linear(hidden_k, self.rank_k),
        )
        self.task_query = nn.Parameter(torch.randn(self.num_tasks, self.rank_k) * 0.1)
        self.access_mlp = nn.Sequential(
            nn.Linear(self.input_dim, max(16, self.input_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, self.input_dim // 8), 1),
        )

    @staticmethod
    def _center_prior(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype)
        center = (float(length) - 1.0) / 2.0
        sigma = max(1.0, float(length) / 6.0)
        prior = torch.exp(-((pos - center) ** 2) / (2.0 * sigma * sigma))
        prior = (prior - prior.mean()) / (prior.std() + 1e-6)
        return prior

    def _align_eigvals(self, eigvals_topk: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # eigvals_topk: [B, k_used]
        bsz, used_k = eigvals_topk.shape
        if used_k == self.rank_k:
            return eigvals_topk, used_k
        aligned = eigvals_topk.new_zeros((bsz, self.rank_k))
        aligned[:, :used_k] = eigvals_topk
        return aligned, used_k

    def forward(
        self,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # sequence_hidden: [B, L, D]
        h = torch.nan_to_num(sequence_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        bsz, seq_len, dim = h.shape

        autocast_off_ctx = contextlib.nullcontext()
        if self.force_fp32 and h.is_cuda:
            autocast_off_ctx = torch.amp.autocast(device_type="cuda", enabled=False)

        with autocast_off_ctx:
            h32 = h.float()
            # [Mathematical Fix] Preserve end-to-end gradients by avoiding detach.
            # We use a purely continuous graph and apply Diagonal Jitter to prevent
            # collisions between eigenvalues that cause NaN gradients in eigh.
            spectral_source = h32.detach() if self.detach_backbone_for_mask else h32

            corr = torch.matmul(spectral_source, spectral_source.transpose(1, 2)) / float(max(1, dim))
            if seq_mask is not None:
                valid = (~seq_mask).float()
                corr = corr * valid.unsqueeze(1) * valid.unsqueeze(2)
            corr = 0.5 * (corr + corr.transpose(1, 2))
            
            # [Anti-Roll Bar] Break eigenvalue degeneracy to ensure stability.
            # Padding regions or low-rank data cause identical 0 eigenvalues. The backward 
            # pass of eigh divides by (eig_i - eig_j), so identical eigenvalues cause NaN/Inf.
            # A linearly spaced diagonal jitter uniquely shifts every node, guaranteeing distinct eigenvalues.
            jitter_vals = torch.linspace(1e-4, 2e-4, steps=corr.size(1), device=corr.device, dtype=corr.dtype)
            jitter = torch.diag(jitter_vals).unsqueeze(0)  # Shape: [1, L, L]
            corr = corr + jitter

            eigvals, eigvecs = torch.linalg.eigh(corr)
            k_used = min(self.rank_k, seq_len)
            top_vals = torch.clamp(eigvals[:, -k_used:], min=0.0)
            top_vecs = eigvecs[:, :, -k_used:]

            aligned_vals, k_used = self._align_eigvals(top_vals)
            alpha_base = self.spectral_mlp(aligned_vals)[:, :k_used]

            if self.task_conditioned:
                task_q = self.task_query[:, :k_used]
                alpha = alpha_base.unsqueeze(1) * task_q.unsqueeze(0)
            else:
                alpha = alpha_base.unsqueeze(1).expand(-1, self.num_tasks, -1)

            logits = torch.einsum("blk,btk->btl", top_vecs, alpha) / math.sqrt(float(max(1, k_used)))

            center_prior = self._center_prior(seq_len, device=h.device, dtype=logits.dtype)
            logits = logits + self.center_prior_weight * center_prior.view(1, 1, seq_len)

            use_struct_for_access = (
                isinstance(structure_hidden, torch.Tensor)
                and structure_hidden.ndim == 3
                and structure_hidden.shape[-1] == self.input_dim
            )
            if use_struct_for_access:
                access_source = torch.nan_to_num(structure_hidden, nan=0.0, posinf=0.0, neginf=0.0).float()
                if self.detach_backbone_for_mask:
                    access_source = access_source.detach()
            else:
                access_source = spectral_source
            access_logit = self.access_mlp(access_source)
            access_prior = torch.sigmoid(access_logit).squeeze(-1).to(dtype=logits.dtype)
            logits = logits + self.access_prior_weight * access_prior.unsqueeze(1)

            if seq_mask is not None:
                logits = logits.masked_fill(seq_mask.unsqueeze(1), -12.0)

            fg_mask = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
            
            # [Soft Masking] Compress mask from [0, 1] to [0.05, 1.0] to prevent 
            # absolute zeroing of background which causes "false purity" but destroys context
            soft_mask = fg_mask * 0.95 + 0.05
            
            task_hidden = h32.unsqueeze(1) * soft_mask.unsqueeze(-1)

            if seq_mask is not None:
                valid = (~seq_mask).float().unsqueeze(1).unsqueeze(-1)
                denom = valid.sum(dim=2).clamp(min=1.0)
                task_repr = (task_hidden * valid).sum(dim=2) / denom
            else:
                task_repr = task_hidden.mean(dim=2)

            # Use soft mask for sequence enhancements to preserve background residual
            shared_mask = soft_mask.mean(dim=1)
            if self.sequence_enhance_mode == "off":
                sequence_enhanced = h32
            elif self.sequence_enhance_mode == "mean_only":
                # Mean-only enhancement avoids over-constraining fine-grained center tokens.
                shared_gain = shared_mask.mean(dim=1, keepdim=True).unsqueeze(-1)
                sequence_enhanced = h32 * (1.0 + shared_gain)
            else:
                sequence_enhanced = h32 * (1.0 + shared_mask.unsqueeze(-1))

            center_start = max(0, seq_len // 2 - 2)
            center_end = min(seq_len, seq_len // 2 + 3)
            center_region = fg_mask[:, :, center_start:center_end]
            center_mean = center_region.mean(dim=(0, 2))
            edge_mask = torch.ones((seq_len,), dtype=torch.bool, device=fg_mask.device)
            edge_mask[center_start:center_end] = False
            if bool(edge_mask.any()):
                bg_mean = fg_mask[:, :, edge_mask].mean(dim=(0, 2))
            else:
                bg_mean = torch.ones_like(center_mean)
            center_ratio = center_mean / (bg_mean + 1e-6)

            sparse_loss = fg_mask.mean()
            entropy_loss = -(fg_mask * torch.log(fg_mask) + (1.0 - fg_mask) * torch.log(1.0 - fg_mask)).mean()

            if self.num_tasks > 1:
                task_masks = fg_mask.reshape(bsz, self.num_tasks, -1)
                task_masks = F.normalize(task_masks, dim=-1)
                sim = torch.matmul(task_masks, task_masks.transpose(1, 2))
                offdiag = 1.0 - torch.eye(self.num_tasks, device=sim.device, dtype=sim.dtype)
                diversity_loss = (sim * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))
            else:
                diversity_loss = fg_mask.new_zeros(())

        return {
            "sequence_enhanced": sequence_enhanced.to(dtype=sequence_hidden.dtype),
            "task_repr": task_repr.to(dtype=sequence_hidden.dtype),
            "fg_mask": fg_mask,
            "fg_sparse_loss": sparse_loss,
            "fg_entropy_loss": entropy_loss,
            "fg_task_diversity_loss": diversity_loss,
            "mask_mean_per_task": fg_mask.mean(dim=(0, 2)),
            "mask_center_to_bg_ratio": center_ratio,
        }


class LaSTStabilityForegroundBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        topk_ratio: float = 0.5,
        sigma: float = 0.12,
        vote_mode: str = "token",
        task_conditioned: bool = True,
        center_prior_weight: float = 0.8,
        access_prior_weight: float = 0.4,
        eps: float = 1e-6,
        force_fp32: bool = True,
        sequence_enhance_mode: str = "full",
        detach_backbone_for_mask: bool = True,
        task_center_sigmas: Optional[List[float]] = None,
        task_center_offsets: Optional[List[float]] = None,
        mask_iou_target_max: float = 1.0,
        mask_iou_lambda: float = 0.0,
        task_repr_inter_target: float = 1.0,
        task_repr_contrastive_lambda: float = 0.0,
        grl_lambda: float = 1.0,
        use_prompt_attention: bool = True,
        prompt_orthogonal_lambda: float = 0.0,
        prompt_mi_lambda: float = 0.0,
        attention_sparse_lambda: float = 0.0,
        attention_entropy_target: float = -1.0,
        attention_topk_ratio: float = 0.0,
        attention_guidance_mix: float = 0.0,
        attention_decorrelation_lambda: float = 0.0,
        attention_tau_lambda: float = 0.0,
        attention_tau_target_max: float = 0.20,
        attention_focus_lambda: float = 0.0,
        attention_focus_target: float = 0.18,
        attention_topk_min_mix: float = 0.5,
        mask_mean_target: float = 0.25,
        mask_mean_tolerance: float = 0.10,
        mask_variance_target: float = 0.05,
        use_token_prior_for_prompt_attention: bool = True,
        token_prior_guidance_mix: float = 0.8,
        token_prior_blend_ratio: float = 0.35,
        token_prior_diversity_lambda: float = 0.0,
        foreground_attention_consistency_lambda: float = 0.0,
        token_prior_logit_clip: float = 4.0,
        task_channel_gate_strength: float = 0.0,
        mask_logit_source: str = "relative_log_prior",
        token_prior_mask_logit_mix: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = max(1, int(num_tasks))
        self.topk_ratio = float(topk_ratio)
        self.sigma = float(sigma)
        self.vote_mode = str(vote_mode).strip().lower()
        self.task_conditioned = bool(task_conditioned)
        self.center_prior_weight = float(center_prior_weight)
        self.access_prior_weight = float(access_prior_weight)
        self.eps = float(eps)
        self.force_fp32 = bool(force_fp32)
        self.sequence_enhance_mode = str(sequence_enhance_mode).strip().lower()
        self.detach_backbone_for_mask = bool(detach_backbone_for_mask)
        self.mask_iou_target_max = float(mask_iou_target_max)
        self.mask_iou_lambda = float(mask_iou_lambda)
        self.task_repr_inter_target = float(task_repr_inter_target)
        self.task_repr_contrastive_lambda = float(task_repr_contrastive_lambda)
        self.prompt_orthogonal_lambda = float(prompt_orthogonal_lambda)
        self.prompt_mi_lambda = float(prompt_mi_lambda)
        self.attention_sparse_lambda = float(attention_sparse_lambda)
        self.attention_entropy_target = float(attention_entropy_target)
        self.attention_topk_ratio = float(max(0.0, min(1.0, attention_topk_ratio)))
        self.attention_guidance_mix = float(max(0.0, min(1.0, attention_guidance_mix)))
        self.attention_decorrelation_lambda = float(attention_decorrelation_lambda)
        self.attention_tau_lambda = float(attention_tau_lambda)
        self.attention_tau_target_max = float(max(0.01, attention_tau_target_max))
        self.attention_focus_lambda = float(attention_focus_lambda)
        self.attention_focus_target = float(max(0.0, min(1.0, attention_focus_target)))
        self.attention_topk_min_mix = float(max(0.0, min(1.0, attention_topk_min_mix)))
        self.mask_mean_target = float(max(0.0, min(1.0, mask_mean_target)))
        self.mask_mean_tolerance = float(max(1e-3, min(0.49, mask_mean_tolerance)))
        self.mask_variance_target = float(max(0.0, min(1.0, mask_variance_target)))
        self.use_token_prior_for_prompt_attention = bool(use_token_prior_for_prompt_attention)
        self.token_prior_guidance_mix = float(max(0.0, min(2.0, token_prior_guidance_mix)))
        self.token_prior_blend_ratio = float(max(0.0, min(1.0, token_prior_blend_ratio)))
        self.token_prior_diversity_lambda = float(max(0.0, token_prior_diversity_lambda))
        self.foreground_attention_consistency_lambda = float(max(0.0, foreground_attention_consistency_lambda))
        self.token_prior_logit_clip = float(max(0.5, token_prior_logit_clip))
        self.task_channel_gate_strength = float(max(0.0, task_channel_gate_strength))
        self.mask_logit_source = str(mask_logit_source).strip().lower()
        self.token_prior_mask_logit_mix = float(max(0.0, min(1.0, token_prior_mask_logit_mix)))

        # GRL Discriminator to enforce task-agnostic shared representations
        self.grl_lambda = float(grl_lambda)
        self.grl = GradientReversalLayer(lambda_=self.grl_lambda)
        self.task_discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, self.num_tasks)
        )

        # Task-specific prompts for Cross-Attention based masking
        self.use_prompt_attention = bool(use_prompt_attention)
        self.task_prompts = nn.Parameter(torch.empty(self.num_tasks, self.input_dim))
        nn.init.normal_(self.task_prompts, mean=0.0, std=0.02)
        self.task_channel_gates = nn.Parameter(torch.empty(self.num_tasks, self.input_dim))
        nn.init.normal_(self.task_channel_gates, mean=0.0, std=0.15)
        self.prompt_query_norm = nn.LayerNorm(self.input_dim)
        self.prompt_key_norm = nn.LayerNorm(self.input_dim)
        self.prompt_value_norm = nn.LayerNorm(self.input_dim)
        
        # Find a valid num_heads for MultiheadAttention
        heads = 8
        while self.input_dim % heads != 0 and heads > 1:
            heads -= 1
            
        self.prompt_attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=heads,
            batch_first=True
        )
        nn.init.xavier_uniform_(self.prompt_attention.in_proj_weight, gain=0.5)
        if self.prompt_attention.in_proj_bias is not None:
            nn.init.zeros_(self.prompt_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.prompt_attention.out_proj.weight, gain=0.5)
        if self.prompt_attention.out_proj.bias is not None:
            nn.init.zeros_(self.prompt_attention.out_proj.bias)
        self.attn_temperature = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))

        self.task_bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim // 4, bias=False),
                nn.LayerNorm(self.input_dim // 4),
                nn.GELU(),
                nn.Linear(self.input_dim // 4, self.input_dim, bias=False),
                nn.LayerNorm(self.input_dim)
            ) for _ in range(self.num_tasks)
        ])

        if self.task_conditioned:
            self.task_scale = nn.Parameter(torch.ones(self.num_tasks) * 0.5)
            self.task_bias = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.task_scale = nn.Parameter(torch.tensor(0.5))
            self.task_bias = nn.Parameter(torch.tensor(0.0))
        self._register_prompt_grad_sanitizers()

        self.access_mlp = nn.Sequential(
            nn.Linear(self.input_dim, max(16, self.input_dim // 8)),
            nn.GELU(),
            nn.Linear(max(16, self.input_dim // 8), 1),
        )
        sigma_values = task_center_sigmas or [1.0] * self.num_tasks
        offset_values = task_center_offsets or [0.0] * self.num_tasks
        if len(sigma_values) != self.num_tasks:
            sigma_values = (list(sigma_values) + [1.0] * self.num_tasks)[: self.num_tasks]
        if len(offset_values) != self.num_tasks:
            offset_values = (list(offset_values) + [0.0] * self.num_tasks)[: self.num_tasks]
        self.register_buffer("task_center_sigmas", torch.tensor(sigma_values, dtype=torch.float32))
        self.register_buffer("task_center_offsets", torch.tensor(offset_values, dtype=torch.float32))

    @staticmethod
    def _center_prior(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype)
        center = (float(length) - 1.0) / 2.0
        sigma = max(1.0, float(length) / 6.0)
        prior = torch.exp(-((pos - center) ** 2) / (2.0 * sigma * sigma))
        prior = (prior - prior.mean()) / (prior.std() + 1e-6)
        return prior

    @staticmethod
    def _sanitize_prompt_grad(grad: Optional[torch.Tensor], clip_value: float = 0.5) -> Optional[torch.Tensor]:
        if grad is None:
            return grad
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad.clamp(min=-clip_value, max=clip_value)

    def _register_prompt_grad_sanitizers(self) -> None:
        prompt_params = [
            self.task_prompts,
            self.task_scale,
            self.task_channel_gates,
            self.attn_temperature,
            self.prompt_query_norm.weight,
            self.prompt_query_norm.bias,
            self.prompt_key_norm.weight,
            self.prompt_key_norm.bias,
            self.prompt_value_norm.weight,
            self.prompt_value_norm.bias,
        ]
        attn = self.prompt_attention
        prompt_params.extend(
            [attn.in_proj_weight, attn.in_proj_bias, attn.out_proj.weight, attn.out_proj.bias]
        )
        for param in prompt_params:
            if isinstance(param, torch.Tensor) and getattr(param, "requires_grad", False):
                param.register_hook(self._sanitize_prompt_grad)

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl_lambda = float(lambda_)
        self.grl.set_lambda(self.grl_lambda)

    def _task_center_prior(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype).view(1, -1)
        center = ((float(length) - 1.0) / 2.0) + self.task_center_offsets.to(device=device, dtype=dtype).view(-1, 1)
        sigma = self.task_center_sigmas.to(device=device, dtype=dtype).abs().clamp(min=1.0).view(-1, 1)
        prior = torch.exp(-((pos - center) ** 2) / (2.0 * sigma * sigma))
        prior = (prior - prior.mean(dim=1, keepdim=True)) / (prior.std(dim=1, keepdim=True) + 1e-6)
        return prior

    def _gaussian_lowpass(self, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        d = int(d)
        f = torch.fft.fftfreq(d, device=device, dtype=torch.float32)
        sigma = max(1e-6, float(self.sigma))
        g = torch.exp(-0.5 * (f / sigma) ** 2)
        g = (g - g.min()) / (g.max() - g.min() + 1e-12)
        return g.to(dtype=torch.complex64).to(device=device)

    def _token_scores(self, h: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, dim = h.shape
        g = self._gaussian_lowpass(dim, device=h.device, dtype=h.dtype)
        x_fft = torch.fft.fft(h, dim=-1)
        x_hat = torch.fft.ifft(x_fft * g.view(1, 1, -1), dim=-1).real
        s = x_hat / (torch.abs(x_hat - h) + self.eps)
        if self.vote_mode == "channel":
            k = max(1, min(seq_len, int(round(seq_len * float(self.topk_ratio)))))
            idx = torch.topk(s, k=k, dim=1, largest=True, sorted=False).indices
            idx_flat = idx.permute(0, 2, 1).reshape(bsz * dim, k)
            votes_flat = torch.zeros((bsz * dim, seq_len), device=h.device, dtype=torch.float32)
            votes_flat.scatter_add_(1, idx_flat, torch.ones_like(idx_flat, dtype=votes_flat.dtype))
            votes = votes_flat.view(bsz, dim, seq_len).sum(dim=1)
            scores = votes / votes.max(dim=1, keepdim=True).values.clamp(min=1.0)
        else:
            scores = s.mean(dim=-1)
            scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
            scores = torch.sigmoid(scores)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, 0.0)
        return scores

    def _normalize_token_prior(self, scores: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        prior = scores.float().clamp(min=1e-6)
        if valid_mask is not None:
            prior = prior.masked_fill(~valid_mask.unsqueeze(1), 0.0)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return prior

    def _stable_logit(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(min=1e-4, max=1.0 - 1e-4)
        logits = torch.log(probs) - torch.log1p(-probs)
        return logits.clamp(min=-self.token_prior_logit_clip, max=self.token_prior_logit_clip)

    def _relative_token_logits(
        self,
        probs: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = torch.log(probs.float().clamp(min=1e-6))
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(1).to(dtype=logits.dtype)
            denom = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            center = (logits * valid).sum(dim=-1, keepdim=True) / denom
            var = ((logits - center).pow(2) * valid).sum(dim=-1, keepdim=True) / denom
            logits = (logits - center) / var.sqrt().clamp(min=1e-3)
            logits = logits.masked_fill(~valid_mask.unsqueeze(1), -self.token_prior_logit_clip)
        else:
            center = logits.mean(dim=-1, keepdim=True)
            scale = logits.std(dim=-1, keepdim=True).clamp(min=1e-3)
            logits = (logits - center) / scale
        return logits.clamp(min=-self.token_prior_logit_clip, max=self.token_prior_logit_clip)

    def forward(
        self,
        sequence_hidden: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        structure_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = torch.nan_to_num(sequence_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        bsz, seq_len, dim = h.shape

        autocast_off_ctx = contextlib.nullcontext()
        if self.force_fp32 and h.is_cuda:
            autocast_off_ctx = torch.amp.autocast(device_type="cuda", enabled=False)

        with autocast_off_ctx:
            h32 = h.float()
            spectral_source = h32.detach() if self.detach_backbone_for_mask else h32

            valid = None
            if seq_mask is not None and seq_mask.ndim == 2:
                valid = (~seq_mask).to(dtype=torch.bool)

            # Adversarial Task Discrimination
            grl_source = self.grl(spectral_source.mean(dim=1)) # [bsz, dim]
            adv_logits = self.task_discriminator(grl_source) # [bsz, num_tasks]

            task_bases = []
            for i in range(self.num_tasks):
                task_view = spectral_source
                if self.task_channel_gate_strength > 0.0 and self.task_conditioned:
                    channel_gate = torch.tanh(self.task_channel_gates[i]).to(device=h32.device, dtype=h32.dtype).view(1, 1, -1)
                    task_view = spectral_source * (1.0 + self.task_channel_gate_strength * channel_gate)
                t_source = self.task_bottlenecks[i](task_view)
                b_i = self._token_scores(t_source, valid_mask=valid).to(dtype=h32.dtype)
                task_bases.append(b_i)
            token_prior_scores = torch.stack(task_bases, dim=1)  # [bsz, num_tasks, seq_len]
            token_prior = self._normalize_token_prior(token_prior_scores, valid_mask=valid)
            token_prior_logits = self._stable_logit(token_prior)
            values = self.prompt_value_norm(spectral_source)

            if self.use_prompt_attention:
                prompts = self.task_prompts.unsqueeze(0).expand(bsz, -1, -1) # [bsz, num_tasks, dim]
                keys = self.prompt_key_norm(spectral_source) # [bsz, seq_len, dim]
                queries = self.prompt_query_norm(prompts)    # [bsz, num_tasks, dim]
                keys_norm = F.normalize(keys, p=2, dim=-1, eps=1e-6)
                queries_norm = F.normalize(queries, p=2, dim=-1, eps=1e-6)
                cos_sim = torch.bmm(queries_norm, keys_norm.transpose(1, 2))
                tau = torch.exp(self.attn_temperature).clamp(min=0.01, max=10.0)
                attn_logits = cos_sim / tau

                task_center_prior = self._task_center_prior(seq_len, device=h.device, dtype=attn_logits.dtype)
                attn_logits = attn_logits + self.attention_guidance_mix * task_center_prior.unsqueeze(0)
                if self.use_token_prior_for_prompt_attention:
                    attn_logits = attn_logits + self.token_prior_guidance_mix * token_prior_logits.to(dtype=attn_logits.dtype)

                if seq_mask is not None:
                    attn_logits = attn_logits.masked_fill(seq_mask.unsqueeze(1), -10000.0)

                attn_weights = F.softmax(attn_logits, dim=-1)
                if self.attention_topk_ratio > 0.0:
                    k = max(1, int(round(seq_len * self.attention_topk_ratio)))
                    top_vals, top_idx = torch.topk(attn_weights, k=k, dim=-1, largest=True, sorted=False)
                    sparse_attn = torch.zeros_like(attn_weights)
                    sparse_attn.scatter_(-1, top_idx, top_vals)
                    sparse_attn = sparse_attn / sparse_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    topk_mix = max(self.attention_guidance_mix, self.attention_topk_min_mix)
                    attn_weights = (1.0 - topk_mix) * attn_weights + topk_mix * sparse_attn
                    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

                base = attn_weights
                if self.use_token_prior_for_prompt_attention:
                    base = (1.0 - self.token_prior_blend_ratio) * attn_weights + self.token_prior_blend_ratio * token_prior.to(dtype=attn_weights.dtype)
                    base = base / base.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                base = base.to(dtype=h32.dtype)
                context_from_attn = torch.bmm(base, values)
            else:
                tau = h32.new_tensor(self.attention_tau_target_max)
                attn_weights = token_prior.to(dtype=h32.dtype)
                base = attn_weights
                context_from_attn = torch.bmm(base, spectral_source)

            base_relative_logits = self._relative_token_logits(base, valid_mask=valid).to(dtype=h32.dtype)
            prior_relative_logits = self._relative_token_logits(token_prior, valid_mask=valid).to(dtype=h32.dtype)
            if self.mask_logit_source in {"token_prior", "prior"}:
                mask_source_logits = prior_relative_logits
            elif self.mask_logit_source in {"attention", "attn"}:
                mask_source_logits = base_relative_logits
            else:
                prior_mix = self.token_prior_mask_logit_mix if self.use_token_prior_for_prompt_attention else 0.0
                mask_source_logits = (1.0 - prior_mix) * base_relative_logits + prior_mix * prior_relative_logits

            if self.task_conditioned:
                scale = torch.tanh(self.task_scale).view(1, self.num_tasks, 1)
                bias = self.task_bias.view(1, self.num_tasks, 1)
                logits = mask_source_logits * scale + bias
            else:
                logits = mask_source_logits * torch.tanh(self.task_scale) + self.task_bias

            if self.task_conditioned:
                center_prior = self._task_center_prior(seq_len, device=h.device, dtype=logits.dtype)
                logits = logits + self.center_prior_weight * center_prior.unsqueeze(0)
            else:
                center_prior = self._center_prior(seq_len, device=h.device, dtype=logits.dtype)
                logits = logits + self.center_prior_weight * center_prior.view(1, 1, seq_len)

            use_struct_for_access = (
                isinstance(structure_hidden, torch.Tensor)
                and structure_hidden.ndim == 3
                and structure_hidden.shape[-1] == self.input_dim
            )
            if use_struct_for_access:
                access_source = torch.nan_to_num(structure_hidden, nan=0.0, posinf=0.0, neginf=0.0).float()
                if self.detach_backbone_for_mask:
                    access_source = access_source.detach()
            else:
                access_source = spectral_source

            access_logit = self.access_mlp(access_source)
            access_prior = torch.sigmoid(access_logit).squeeze(-1).to(dtype=logits.dtype)
            logits = logits + self.access_prior_weight * access_prior.unsqueeze(1)

            if seq_mask is not None:
                logits = logits.masked_fill(seq_mask.unsqueeze(1), -12.0)

            fg_mask = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
            mask_mean_per_task = fg_mask.mean(dim=(0, 2))
            soft_mask = fg_mask * 0.95 + 0.05
            task_hidden = h32.unsqueeze(1) * soft_mask.unsqueeze(-1)

            if valid is not None:
                valid_f = valid.float().unsqueeze(1).unsqueeze(-1)
                denom = valid_f.sum(dim=2).clamp(min=1.0)
                task_repr = (task_hidden * valid_f).sum(dim=2) / denom
            else:
                task_repr = task_hidden.mean(dim=2)
            task_repr = task_repr + 0.2 * context_from_attn

            shared_mask = soft_mask.mean(dim=1)
            if self.sequence_enhance_mode == "off":
                sequence_enhanced = h32
            elif self.sequence_enhance_mode == "mean_only":
                shared_gain = shared_mask.mean(dim=1, keepdim=True).unsqueeze(-1)
                sequence_enhanced = h32 * (1.0 + shared_gain)
            else:
                sequence_enhanced = h32 * (1.0 + shared_mask.unsqueeze(-1))

            center_start = max(0, seq_len // 2 - 2)
            center_end = min(seq_len, seq_len // 2 + 3)
            center_region = fg_mask[:, :, center_start:center_end]
            center_mean = center_region.mean(dim=(0, 2))
            edge_mask = torch.ones((seq_len,), dtype=torch.bool, device=fg_mask.device)
            edge_mask[center_start:center_end] = False
            if bool(edge_mask.any()):
                bg_mean = fg_mask[:, :, edge_mask].mean(dim=(0, 2))
            else:
                bg_mean = torch.ones_like(center_mean)
            center_ratio = center_mean / (bg_mean + 1e-6)

            sparse_loss = fg_mask.mean()
            entropy_loss = -(fg_mask * torch.log(fg_mask) + (1.0 - fg_mask) * torch.log(1.0 - fg_mask)).mean()
            attn_entropy = -(attn_weights.clamp(min=1e-6) * torch.log(attn_weights.clamp(min=1e-6))).sum(dim=-1)
            attn_peak = attn_weights.max(dim=-1).values
            attn_focus_ratio = attn_weights.topk(k=max(1, min(seq_len, max(1, int(round(seq_len * 0.05))))), dim=-1).values.sum(dim=-1)
            if self.attention_entropy_target >= 0.0:
                attention_entropy_loss = torch.relu(attn_entropy.mean() - self.attention_entropy_target)
            else:
                attention_entropy_loss = fg_mask.new_zeros(())
            attention_sparse_loss = (1.0 - attn_peak).mean()
            attention_tau_loss = torch.relu(tau - self.attention_tau_target_max)
            attention_focus_loss = torch.relu(self.attention_focus_target - attn_focus_ratio.mean())

            mask_mean_lower = max(0.0, self.mask_mean_target - self.mask_mean_tolerance)
            mask_mean_upper = min(1.0, self.mask_mean_target + self.mask_mean_tolerance)
            mask_mean_loss = (
                torch.relu(mask_mean_per_task - mask_mean_upper)
                + torch.relu(mask_mean_lower - mask_mean_per_task)
            ).mean()

            if self.num_tasks > 1:
                task_masks = fg_mask.reshape(bsz, self.num_tasks, -1)
                task_masks = F.normalize(task_masks, dim=-1)
                sim = torch.matmul(task_masks, task_masks.transpose(1, 2))
                offdiag = 1.0 - torch.eye(self.num_tasks, device=sim.device, dtype=sim.dtype)
                diversity_loss = (sim * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))
                raw_masks = fg_mask.reshape(bsz, self.num_tasks, -1)
                pair_i = raw_masks.unsqueeze(2)
                pair_j = raw_masks.unsqueeze(1)
                intersection = torch.minimum(pair_i, pair_j).sum(dim=-1)
                union = torch.maximum(pair_i, pair_j).sum(dim=-1).clamp(min=1e-6)
                iou = intersection / union
                iou_loss = torch.relu(iou - self.mask_iou_target_max)
                iou_loss = (iou_loss * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))

                task_repr_norm = F.normalize(task_repr, dim=-1)
                task_repr_sim = torch.matmul(task_repr_norm, task_repr_norm.transpose(1, 2))
                repr_contrastive_loss = torch.relu(task_repr_sim - self.task_repr_inter_target)
                repr_contrastive_loss = (repr_contrastive_loss * offdiag.unsqueeze(0)).sum() / (
                    offdiag.sum() * float(max(1, bsz))
                )
                inter_task_similarity = (task_repr_sim * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))

                prompt_norm = F.normalize(self.task_prompts.float(), dim=-1)
                prompt_sim = torch.matmul(prompt_norm, prompt_norm.transpose(0, 1))
                prompt_orthogonal_loss = (prompt_sim.pow(2) * offdiag).sum() / offdiag.sum().clamp(min=1.0)
                task_repr_centered = task_repr.float() - task_repr.float().mean(dim=0, keepdim=True)
                cov = torch.matmul(task_repr_centered.transpose(1, 2), task_repr_centered).mean(dim=0)
                cov = cov / float(max(1, bsz))
                diag_cov = torch.diag(torch.diag(cov))
                task_mi_proxy = ((cov - diag_cov).pow(2).mean()).clamp(min=0.0)
                prior_norm = F.normalize(token_prior.reshape(bsz, self.num_tasks, -1), dim=-1)
                prior_sim = torch.matmul(prior_norm, prior_norm.transpose(1, 2))
                token_prior_diversity_loss = (prior_sim * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))
            else:
                diversity_loss = fg_mask.new_zeros(())
                iou = fg_mask.new_zeros((bsz, 1, 1))
                iou_loss = fg_mask.new_zeros(())
                repr_contrastive_loss = fg_mask.new_zeros(())
                inter_task_similarity = fg_mask.new_zeros(())
                prompt_orthogonal_loss = fg_mask.new_zeros(())
                task_mi_proxy = fg_mask.new_zeros(())
                token_prior_diversity_loss = fg_mask.new_zeros(())

            base_mean = base.mean(dim=(0, 2))
            base_peak = base.max(dim=2).values.mean(dim=0)
            
            # [Principled Constraint 1] Cross-Task Mask Variance Maximization
            # Ensure different tasks do not collapse to the exact same mask (like 0.5 everywhere)
            if self.num_tasks > 1:
                mask_var_across_tasks = fg_mask.var(dim=1) # [bsz, seq_len]
                # Encourage variance to exceed a configurable target
                mask_variance_loss = torch.relu(self.mask_variance_target - mask_var_across_tasks.mean())
            else:
                mask_variance_loss = fg_mask.new_zeros(())
                
            # [Principled Constraint 2] Attention Decorrelation (Orthogonality)
            # Force the attention maps for different tasks to focus on different regions
            if self.num_tasks > 1 and self.use_prompt_attention:
                attn_norm = F.normalize(attn_weights, p=2, dim=-1) # [bsz, num_tasks, seq_len]
                attn_sim = torch.matmul(attn_norm, attn_norm.transpose(1, 2))
                attention_decorrelation_loss = (attn_sim * offdiag.unsqueeze(0)).sum() / (offdiag.sum() * float(max(1, bsz)))
                prior_target = token_prior.detach().clamp(min=1e-6)
                foreground_attention_consistency_loss = (
                    prior_target * (torch.log(prior_target) - torch.log(attn_weights.clamp(min=1e-6)))
                ).sum(dim=-1).mean()
            else:
                attention_decorrelation_loss = fg_mask.new_zeros(())
                foreground_attention_consistency_loss = fg_mask.new_zeros(())
            
            # Keep Bipolar Penalty as a lightweight push away from 0.5, but rely primarily on variance
            fg_bipolar_loss = (fg_mask * (1.0 - fg_mask)).mean()

        return {
            "sequence_enhanced": sequence_enhanced.to(dtype=sequence_hidden.dtype),
            "task_repr": task_repr.to(dtype=sequence_hidden.dtype),
            "fg_mask": fg_mask,
            "fg_sparse_loss": sparse_loss,
            "fg_entropy_loss": entropy_loss,
            "fg_bipolar_loss": fg_bipolar_loss,
            "fg_mask_variance_loss": mask_variance_loss,
            "fg_task_diversity_loss": diversity_loss,
            "fg_mask_iou_loss": iou_loss,
            "fg_mask_iou_mean": iou.detach().mean(),
            "task_repr_contrastive_loss": repr_contrastive_loss,
            "task_repr_inter_similarity": inter_task_similarity.detach(),
            "prompt_orthogonal_loss": prompt_orthogonal_loss,
            "attention_decorrelation_loss": attention_decorrelation_loss,
            "token_prior_diversity_loss": token_prior_diversity_loss,
            "foreground_attention_consistency_loss": foreground_attention_consistency_loss,
            "task_mutual_information_loss": task_mi_proxy,
            "task_mutual_information_proxy": task_mi_proxy.detach(),
            "attention_entropy_loss": attention_entropy_loss,
            "attention_sparse_loss": attention_sparse_loss,
            "attention_tau_loss": attention_tau_loss,
            "attention_focus_loss": attention_focus_loss,
            "mask_mean_loss": mask_mean_loss,
            "mask_mean_per_task": mask_mean_per_task,
            "mask_center_to_bg_ratio": center_ratio,
            "last_stability_base_mean": base_mean.detach(),
            "last_stability_base_peak": base_peak.detach(),
            "ptm_attention_peak_per_task": attn_peak.detach().mean(dim=0),
            "ptm_attention_entropy_per_task": attn_entropy.detach().mean(dim=0),
            "ptm_attention_focus_ratio_per_task": attn_focus_ratio.detach().mean(dim=0),
            "token_prior_peak_per_task": token_prior.detach().max(dim=-1).values.mean(dim=0),
            "token_prior_mass_per_task": token_prior.detach().mean(dim=(0, 2)),
            "token_prior_offdiag_mean": token_prior_diversity_loss.detach().view(1),
            "foreground_attention_kl": foreground_attention_consistency_loss.detach().view(1),
            "token_prior": token_prior.to(dtype=sequence_hidden.dtype),
            "attention_weights": attn_weights.to(dtype=sequence_hidden.dtype),
            "task_prompt_cosine_offdiag_mean": (prompt_sim.detach() * (1.0 - torch.eye(self.num_tasks, device=prompt_sim.device, dtype=prompt_sim.dtype))).sum() / max(1.0, float(self.num_tasks * max(1, self.num_tasks - 1))),
            "adv_task_logits": adv_logits,
        }


class StructureReliabilityGate(nn.Module):
    """Estimate structure reliability and down-weight noisy structure signals."""

    def __init__(self, micro_env_dim: int, num_tasks: int, task_conditioned: bool = True) -> None:
        super().__init__()
        self.micro_env_dim = max(0, int(micro_env_dim))
        self.num_tasks = max(1, int(num_tasks))
        self.task_conditioned = bool(task_conditioned)
        out_dim = self.num_tasks if self.task_conditioned else 1
        in_dim = 2 + self.micro_env_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, max(16, in_dim * 4)),
            nn.GELU(),
            nn.Linear(max(16, in_dim * 4), out_dim),
        )

    def _align_micro_env(self, micro_env: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if not isinstance(micro_env, torch.Tensor):
            return torch.zeros((batch_size, self.micro_env_dim), device=device)
        x = micro_env.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[0] != batch_size:
            x = x[:batch_size]
            if x.shape[0] < batch_size:
                pad_rows = batch_size - x.shape[0]
                x = torch.cat([x, torch.zeros((pad_rows, x.shape[1]), device=device, dtype=x.dtype)], dim=0)
        if x.shape[1] > self.micro_env_dim:
            x = x[:, :self.micro_env_dim]
        elif x.shape[1] < self.micro_env_dim:
            pad_cols = self.micro_env_dim - x.shape[1]
            x = torch.cat([x, torch.zeros((batch_size, pad_cols), device=device, dtype=x.dtype)], dim=1)
        return x.to(device=device)

    def forward(
        self,
        struct_global: Optional[torch.Tensor],
        struct_mask: Optional[torch.Tensor],
        micro_env: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if struct_global is None:
            return None
        bsz = struct_global.shape[0]
        device = struct_global.device
        available = (struct_global.detach().abs().sum(dim=1, keepdim=True) > 0).float()
        if isinstance(struct_mask, torch.Tensor):
            mask_ratio = struct_mask.float().mean(dim=1, keepdim=True)
        else:
            mask_ratio = torch.zeros((bsz, 1), device=device)
        micro = self._align_micro_env(micro_env, batch_size=bsz, device=device)
        features = torch.cat([available, 1.0 - mask_ratio, micro], dim=1)
        reliability = torch.sigmoid(self.mlp(features))
        if reliability.shape[1] == 1:
            reliability = reliability.expand(-1, self.num_tasks)
        return reliability




class PTMMultimodalCrossAttention(nn.Module):
    """Per-PTM cross-attention over sequence, structure and local context tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_tasks: int,
        task_names: Optional[List[str]] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        target_tasks: Optional[List[str]] = None,
        focus_lambda: float = 0.0,
        focus_target: float = 0.06,
        entropy_lambda: float = 0.0,
        entropy_target: float = 3.80,
        decorrelation_lambda: float = 0.0,
        decorrelation_target_max: float = 0.80,
        guidance_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = max(1, int(num_tasks))
        heads = max(1, int(num_heads))
        while self.hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.task_names = [str(x).strip().lower() for x in (task_names or [])]
        target_set = {str(x).strip().lower() for x in (target_tasks or [])}
        target_mask = []
        for idx in range(self.num_tasks):
            name = self.task_names[idx] if idx < len(self.task_names) else f"task_{idx}"
            target_mask.append(1.0 if name in target_set else 0.0)
        self.register_buffer("target_task_mask", torch.tensor(target_mask, dtype=torch.float32).view(1, self.num_tasks, 1))
        self.focus_lambda = float(max(0.0, focus_lambda))
        self.focus_target = float(max(0.0, focus_target))
        self.entropy_lambda = float(max(0.0, entropy_lambda))
        self.entropy_target = float(max(0.0, entropy_target))
        self.decorrelation_lambda = float(max(0.0, decorrelation_lambda))
        self.decorrelation_target_max = float(max(0.0, min(1.0, decorrelation_target_max)))
        self.guidance_lambda = float(max(0.0, guidance_lambda))
        self.task_embed = nn.Parameter(torch.zeros(self.num_tasks, self.hidden_dim))
        nn.init.normal_(self.task_embed, mean=0.0, std=0.02)
        self.seq_attn = nn.MultiheadAttention(self.hidden_dim, heads, dropout=float(dropout), batch_first=True)
        self.struct_attn = nn.MultiheadAttention(self.hidden_dim, heads, dropout=float(dropout), batch_first=True)
        self.local_attn = nn.MultiheadAttention(self.hidden_dim, heads, dropout=float(dropout), batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 3),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_drop = nn.Dropout(float(dropout))

    @staticmethod
    def _safe_key_padding_mask(mask: Optional[torch.Tensor], tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[:2] != tokens.shape[:2]:
            return None
        key_mask = mask.to(dtype=torch.bool, device=tokens.device)
        if key_mask.all(dim=1).any():
            key_mask = key_mask.clone()
            key_mask[key_mask.all(dim=1)] = False
        return key_mask

    @staticmethod
    def _attention_entropy(weights: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not isinstance(weights, torch.Tensor):
            return None
        prob = weights.float().clamp(min=1e-8)
        return -(prob * torch.log(prob)).sum(dim=-1).mean(dim=0)

    @staticmethod
    def _attention_offdiag_cosine(weights: torch.Tensor) -> torch.Tensor:
        if not isinstance(weights, torch.Tensor) or weights.ndim != 3 or weights.shape[1] <= 1:
            return weights.new_tensor(0.0)
        normed = F.normalize(weights.float(), dim=-1, eps=1e-6)
        sim = torch.bmm(normed, normed.transpose(1, 2))
        task_count = int(sim.shape[-1])
        offdiag = sim[:, ~torch.eye(task_count, dtype=torch.bool, device=sim.device)]
        return offdiag.mean() if offdiag.numel() else weights.new_tensor(0.0)

    def _attention_regularization_losses(
        self,
        seq_w: Optional[torch.Tensor],
        struct_w: Optional[torch.Tensor],
        seq_guidance: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        active = [w for w in (seq_w, struct_w) if isinstance(w, torch.Tensor)]
        if active and self.focus_lambda > 0.0:
            focus_vals = [w.float().max(dim=-1).values.mean() for w in active]
            focus = torch.stack(focus_vals).mean()
            losses["ptm_mm_attention_focus"] = torch.relu(focus.new_tensor(self.focus_target) - focus) * self.focus_lambda
        if active and self.entropy_lambda > 0.0:
            entropy_vals = []
            for w in active:
                ent = self._attention_entropy(w)
                if isinstance(ent, torch.Tensor):
                    entropy_vals.append(ent.mean())
            if entropy_vals:
                entropy = torch.stack(entropy_vals).mean()
                losses["ptm_mm_attention_entropy"] = torch.relu(entropy - entropy.new_tensor(self.entropy_target)) * self.entropy_lambda
        if active and self.decorrelation_lambda > 0.0:
            corr_vals = [self._attention_offdiag_cosine(w) for w in active]
            corr = torch.stack(corr_vals).mean()
            losses["ptm_mm_attention_decorrelation"] = torch.relu(corr - corr.new_tensor(self.decorrelation_target_max)) * self.decorrelation_lambda
        if self.guidance_lambda > 0.0 and isinstance(seq_w, torch.Tensor) and isinstance(seq_guidance, torch.Tensor):
            guide = seq_guidance.float()
            if guide.ndim == 3 and guide.shape == seq_w.shape:
                guide = guide.clamp(min=0.0)
                guide = guide / guide.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                pred = seq_w.float().clamp(min=1e-8)
                # Cross-entropy to the LaST foreground distribution; detach guide so it acts as a teacher.
                guidance_loss = -(guide.detach() * torch.log(pred)).sum(dim=-1).mean()
                losses["ptm_mm_seq_guidance"] = guidance_loss * self.guidance_lambda
        return losses

    def forward(
        self,
        task_query: Optional[torch.Tensor],
        seq_context: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor] = None,
        local_tokens: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        struct_mask: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        seq_guidance: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not isinstance(seq_context, torch.Tensor) or seq_context.ndim != 3:
            return None, {}, {}
        bsz = int(seq_context.shape[0])
        device = seq_context.device
        dtype = seq_context.dtype
        if not isinstance(task_query, torch.Tensor) or task_query.ndim != 3:
            query = seq_context.new_zeros((bsz, self.num_tasks, self.hidden_dim))
        else:
            query = task_query
            if query.shape[-1] != self.hidden_dim:
                if query.shape[-1] > self.hidden_dim:
                    query = query[..., : self.hidden_dim]
                else:
                    query = F.pad(query, (0, self.hidden_dim - query.shape[-1]))
            if query.shape[1] != self.num_tasks:
                if query.shape[1] > self.num_tasks:
                    query = query[:, : self.num_tasks, :]
                else:
                    pad = query.new_zeros((bsz, self.num_tasks - query.shape[1], self.hidden_dim))
                    query = torch.cat([query, pad], dim=1)
        query = query + self.task_embed.to(device=device, dtype=dtype).unsqueeze(0)

        seq_key_mask = self._safe_key_padding_mask(seq_mask, seq_context)
        seq_out, seq_w = self.seq_attn(query, seq_context, seq_context, key_padding_mask=seq_key_mask, need_weights=True)

        if isinstance(struct_context, torch.Tensor) and struct_context.ndim == 3:
            struct_key_mask = self._safe_key_padding_mask(struct_mask, struct_context)
            struct_out, struct_w = self.struct_attn(query, struct_context, struct_context, key_padding_mask=struct_key_mask, need_weights=True)
        else:
            struct_out = torch.zeros_like(seq_out)
            struct_w = None

        if isinstance(local_tokens, torch.Tensor) and local_tokens.ndim == 3:
            local_out, local_w = self.local_attn(query, local_tokens, local_tokens, need_weights=True)
        else:
            local_out = torch.zeros_like(seq_out)
            local_w = None

        if isinstance(reliability, torch.Tensor):
            rel = reliability.to(device=device, dtype=dtype)
            if rel.ndim == 1:
                rel = rel.view(-1, 1)
            rel = rel.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        else:
            rel = torch.ones((bsz, 1), device=device, dtype=dtype)
        struct_out = struct_out * rel.unsqueeze(1)

        target_flag = self.target_task_mask.to(device=device, dtype=dtype).expand(bsz, -1, -1)
        rel_flag = rel.unsqueeze(1).expand(-1, self.num_tasks, -1)
        gate_in = torch.cat([query, seq_out, struct_out, local_out, rel_flag, target_flag], dim=-1)
        gate_logits = self.gate(gate_in.float()).to(dtype=dtype)
        if isinstance(struct_context, torch.Tensor):
            gate_logits[..., 1] = gate_logits[..., 1] + torch.log(rel_flag.squeeze(-1).clamp(min=1e-4))
        else:
            gate_logits[..., 1] = -10000.0
        gates = F.softmax(gate_logits, dim=-1)
        fused = gates[..., 0:1] * seq_out + gates[..., 1:2] * struct_out + gates[..., 2:3] * local_out
        out = self.out_norm(query + self.out_drop(fused))

        metrics: Dict[str, torch.Tensor] = {
            "ptm_mm_gate_seq_per_task": gates[..., 0].detach().mean(dim=0),
            "ptm_mm_gate_struct_per_task": gates[..., 1].detach().mean(dim=0),
            "ptm_mm_gate_local_per_task": gates[..., 2].detach().mean(dim=0),
            "ptm_mm_context_norm_per_task": out.detach().norm(dim=-1).mean(dim=0),
        }
        if isinstance(seq_w, torch.Tensor):
            metrics["ptm_mm_seq_attention_peak_per_task"] = seq_w.detach().max(dim=-1).values.mean(dim=0)
            ent = self._attention_entropy(seq_w.detach())
            if isinstance(ent, torch.Tensor):
                metrics["ptm_mm_seq_attention_entropy_per_task"] = ent
        if isinstance(struct_w, torch.Tensor):
            metrics["ptm_mm_struct_attention_peak_per_task"] = struct_w.detach().max(dim=-1).values.mean(dim=0)
            ent = self._attention_entropy(struct_w.detach())
            if isinstance(ent, torch.Tensor):
                metrics["ptm_mm_struct_attention_entropy_per_task"] = ent
        if isinstance(local_w, torch.Tensor):
            metrics["ptm_mm_local_attention_peak_per_task"] = local_w.detach().max(dim=-1).values.mean(dim=0)
        losses = self._attention_regularization_losses(seq_w, struct_w, seq_guidance)
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.detach()
        return out.to(dtype=dtype), metrics, losses

class DualBranchFusionPredictor(nn.Module):
    """Dual-branch PTM predictor with adaptive fusion between sequence and structure features."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Core hyper-parameters
        self.seq_input_dim = int(self.config.get("seq_feature_dim", self.config.get("seq_dim", 1280)))
        self.local_input_dim = self.seq_input_dim  # Default to seq_input_dim
        self.global_input_dim = self.seq_input_dim # Default to seq_input_dim
        self.struct_input_dim = int(self.config.get("struct_feature_dim", self.config.get("struct_dim", 512)))
        self.struct_global_input_dim = int(self.config.get("struct_global_feature_dim", self.config.get("struct_feature_dim", self.config.get("struct_dim", 512))))
        self.use_structure_global_context = bool(self.config.get("use_structure_global_context", True))
        self.branch_hidden_dim = int(self.config.get("branch_hidden_dim", 256))
        self.fusion_hidden_dim = int(self.config.get("fusion_hidden_dim", 384))
        self.dropout = float(self.config.get("dropout", 0.1))
        self.modality_dropout = float(self.config.get("modality_dropout", 0.1))
        self.fusion_gate_bipolar_lambda = float(self.config.get("fusion_gate_bipolar_lambda", 0.0))
        self.use_structure = bool(self.config.get("use_structure", True))
        self.use_esm_lora = bool(self.config.get("use_esm_lora", False))
        self.esm_lora_max_len = int(self.config.get("esm_lora_max_len", self.config.get("window_size", 73)))
        self.sequence_encoder: Optional[nn.Module] = None
        self.branch_warmup_use_branch_logits = bool(self.config.get("branch_warmup_use_branch_logits", False))
        self.branch_warmup_freeze_fusion = bool(self.config.get("branch_warmup_freeze_fusion", False))
        self.branch_warmup_aux_multiplier = float(self.config.get("branch_warmup_aux_multiplier", 1.0))
        self.use_legacy_fusion = bool(self.config.get("use_legacy_fusion", False))
        self.ptm_types = self.config.get("multitask_ptm_types", [])
        self.num_tasks = int(self.config.get("num_tasks", len(self.ptm_types) if isinstance(self.ptm_types, list) and self.ptm_types else 6))
        self.use_micro_environment = bool(self.config.get("use_micro_environment", True))
        self.micro_env_feature_dim = int(self.config.get("micro_env_feature_dim", 6))
        if not self.use_micro_environment:
            self.micro_env_feature_dim = 0

        self.use_structure_graph = bool(self.config.get("use_structure_graph", False))
        self.use_dense_structure_tower_with_graph = bool(
            self.config.get("use_dense_structure_tower_with_graph", False)
        )
        if self.use_structure_graph and not self.use_dense_structure_tower_with_graph:
            # The graph tower is the structural branch for graph-only routing.
            self.use_structure = False
        self.graph_node_dim = int(self.config.get("graph_node_dim", 0))
        self.graph_edge_dim = int(self.config.get("graph_edge_dim", 0))
        self.graph_hidden_dim = int(self.config.get("graph_hidden_dim", self.branch_hidden_dim))
        self.graph_layers = int(self.config.get("graph_layers", 2))
        self.graph_tower_type = str(self.config.get("graph_tower_type", "egnn")).strip().lower()
        self.use_token_aligned_graph_fusion = bool(
            self.config.get("use_token_aligned_graph_fusion", False)
        )

        if self.use_esm_lora:
            if ESM2LoRAWrapper is None:
                raise ImportError("use_esm_lora=True but ESM2LoRAWrapper is unavailable. Install transformers and peft.")
            self.sequence_encoder = ESM2LoRAWrapper(
                model_name_or_path=str(self.config.get("esm_lora_model_name", "facebook/esm2_t33_650M_UR50D")),
                r=int(self.config.get("esm_lora_rank", 8)),
                lora_alpha=int(self.config.get("esm_lora_alpha", 32)),
                lora_dropout=float(self.config.get("esm_lora_dropout", 0.1)),
            )
            if getattr(self.sequence_encoder, "output_dim", 0):
                lora_dim = int(self.sequence_encoder.output_dim)
                if lora_dim > 0 and lora_dim != self.seq_input_dim:
                    self.seq_input_dim = lora_dim
                    self.local_input_dim = lora_dim
                    self.global_input_dim = lora_dim
                    logger.info("DualBranchFusionPredictor: Overriding seq dims to LoRA output_dim=%d", lora_dim)

        # LaSt-ViT inspired controls
        self.use_last_foreground = bool(self.config.get("use_last_foreground", False))
        self.last_foreground_method = str(self.config.get("last_foreground_method", "spectral")).strip().lower()
        self.last_rank_k = int(self.config.get("last_rank_k", 8))
        self.last_task_conditioned = bool(self.config.get("last_task_conditioned", True))
        self.last_prior_center_weight = float(self.config.get("last_prior_center_weight", 0.8))
        self.last_prior_access_weight = float(self.config.get("last_prior_access_weight", 0.4))
        self.last_stability_sigma = float(self.config.get("last_stability_sigma", 0.12))
        self.last_stability_topk_ratio = float(self.config.get("last_stability_topk_ratio", 0.5))
        self.last_stability_vote_mode = str(self.config.get("last_stability_vote_mode", "token")).strip().lower()
        self.last_mask_sparsity_lambda = float(self.config.get("last_mask_sparsity_lambda", 1e-4))
        self.last_mask_entropy_lambda = float(self.config.get("last_mask_entropy_lambda", 1e-4))
        self.last_mask_diversity_lambda = float(self.config.get("last_mask_diversity_lambda", 0.0))
        self.last_mask_variance_target = float(self.config.get("last_mask_variance_target", 0.05))
        self.last_context_scale = float(self.config.get("last_task_context_scale", 0.2))
        self.last_force_fp32 = bool(self.config.get("last_force_fp32", True))
        self.last_detach_backbone_for_mask = bool(self.config.get("last_detach_backbone_for_mask", False))
        self.last_grl_lambda = float(self.config.get("grl_lambda", 1.0))
        self.last_use_prompt_attention = bool(self.config.get("use_prompt_attention", True))
        self.freeze_explicit_embeddings_during_foreground_warmup = bool(
            self.config.get("freeze_explicit_embeddings_during_foreground_warmup", False)
        )
        self.last_sequence_enhance_mode = str(self.config.get("last_sequence_enhance_mode", "full")).strip().lower()
        self.use_last_structure_reliability = bool(self.config.get("use_last_structure_reliability", False))
        self.last_structure_task_conditioned = bool(self.config.get("last_structure_task_conditioned", True))
        self.last_apply_to_structure_graph = bool(self.config.get("last_apply_to_structure_graph", False))
        self.last_graph_sigma = float(self.config.get("last_graph_sigma", self.last_stability_sigma))
        self.last_graph_topk_ratio = float(self.config.get("last_graph_topk_ratio", self.last_stability_topk_ratio))
        self.last_graph_vote_mode = str(self.config.get("last_graph_vote_mode", self.last_stability_vote_mode)).strip().lower()
        self.last_graph_enhance_scale = float(self.config.get("last_graph_enhance_scale", 1.0))
        self.last_mask_iou_lambda = float(self.config.get("last_mask_iou_lambda", 0.0))
        self.last_mask_iou_target_max = float(self.config.get("last_mask_iou_target_max", 1.0))
        self.last_task_repr_contrastive_lambda = float(self.config.get("last_task_repr_contrastive_lambda", 0.0))
        self.last_task_repr_inter_target = float(self.config.get("last_task_repr_inter_target", 1.0))
        self.last_prompt_orthogonal_lambda = float(self.config.get("last_prompt_orthogonal_lambda", 0.0))
        self.last_prompt_mi_lambda = float(self.config.get("last_prompt_mi_lambda", 0.0))
        self.last_attention_sparse_lambda = float(self.config.get("last_attention_sparse_lambda", 0.0))
        self.last_attention_entropy_target = float(self.config.get("last_attention_entropy_target", -1.0))
        self.last_attention_topk_ratio = float(self.config.get("last_attention_topk_ratio", 0.0))
        self.last_attention_guidance_mix = float(self.config.get("last_attention_guidance_mix", 0.0))
        self.last_attention_decorrelation_lambda = float(self.config.get("last_attention_decorrelation_lambda", self.last_prompt_orthogonal_lambda))
        self.last_attention_tau_lambda = float(self.config.get("last_attention_tau_lambda", self.last_attention_sparse_lambda))
        self.last_attention_tau_target_max = float(self.config.get("last_attention_tau_target_max", 0.20))
        self.last_attention_focus_lambda = float(self.config.get("last_attention_focus_lambda", self.last_attention_sparse_lambda))
        self.last_attention_focus_target = float(self.config.get("last_attention_focus_target", 0.18))
        self.last_attention_topk_min_mix = float(self.config.get("last_attention_topk_min_mix", 0.5))
        self.last_mask_mean_target = float(self.config.get("last_mask_mean_target", 0.25))
        self.last_mask_mean_tolerance = float(self.config.get("last_mask_mean_tolerance", 0.10))
        self.last_mask_logit_source = str(self.config.get("last_mask_logit_source", "relative_log_prior")).strip().lower()
        self.token_prior_mask_logit_mix = float(self.config.get("token_prior_mask_logit_mix", 0.25))
        self.use_token_prior_for_prompt_attention = bool(self.config.get("use_token_prior_for_prompt_attention", True))
        self.token_prior_guidance_mix = float(self.config.get("token_prior_guidance_mix", 0.8))
        self.token_prior_blend_ratio = float(self.config.get("token_prior_blend_ratio", 0.35))
        self.token_prior_diversity_lambda = float(self.config.get("token_prior_diversity_lambda", self.last_mask_diversity_lambda))
        self.foreground_attention_consistency_lambda = float(
            self.config.get("foreground_attention_consistency_lambda", self.last_attention_decorrelation_lambda * 0.25)
        )
        self.token_prior_logit_clip = float(self.config.get("token_prior_logit_clip", 4.0))
        self.last_task_channel_gate_strength = float(self.config.get("last_task_channel_gate_strength", 0.0))
        self.last_task_context_projector_type = str(
            self.config.get("last_task_context_projector_type", "shared")
        ).strip().lower()
        self.last_task_context_projector_rank = int(self.config.get("last_task_context_projector_rank", 64))
        self.last_task_context_projector_dropout = float(self.config.get("last_task_context_projector_dropout", 0.05))
        self.last_task_context_embed_scale = float(self.config.get("last_task_context_embed_scale", 0.25))
        self.last_task_context_remove_shared_scale = float(
            self.config.get("last_task_context_remove_shared_scale", 0.75)
        )
        self.moe_load_balance_lambda = float(self.config.get("moe_load_balance_lambda", 0.0))
        self.last_context_cka_lambda = float(self.config.get("last_context_cka_lambda", self.last_task_repr_contrastive_lambda))
        self.last_context_cov_lambda = float(self.config.get("last_context_cov_lambda", self.last_prompt_mi_lambda))
        self.last_context_cka_target_max = float(self.config.get("last_context_cka_target_max", 0.75))
        self.last_context_cov_target_max = float(self.config.get("last_context_cov_target_max", 0.10))
        sigma_by_label = self.config.get("last_task_mask_sigma_by_label", {})
        offset_by_label = self.config.get("last_task_mask_offset_by_label", {})
        self.last_task_mask_sigmas = [
            float(sigma_by_label.get(ptm, self.config.get("last_task_mask_sigma_default", 1.0)))
            for ptm in self.ptm_types
        ]
        self.last_task_mask_offsets = [
            float(offset_by_label.get(ptm, self.config.get("last_task_mask_offset_default", 0.0)))
            for ptm in self.ptm_types
        ]

        # [NEW] Explicit Feature Embeddings for Dual Branch
        self.use_explicit_features = bool(self.config.get("use_explicit_features", True))
        self.residue_embedding_dim = int(self.config.get("aa_embedding_dim", 64))
        self.position_embedding_dim = int(self.config.get("position_embedding_dim", 32))
        
        if self.use_explicit_features:
            # 25 for standard AA + special tokens
            self.residue_embedding = nn.Embedding(25, self.residue_embedding_dim)
            # 512 for relative position (0..window_size or similar)
            self.position_embedding = nn.Embedding(512, self.position_embedding_dim)
            self._residue_embedding_repaired = False
            self._position_embedding_repaired = False
            
            # Increase input dimension for sequence tower
            self.seq_input_dim += (self.residue_embedding_dim + self.position_embedding_dim)
            
            # Also increase structure input dimension for position embedding if structure is used
            if self.use_structure:
                # [Modified] Add residue embedding (64) + position embedding (32) + micro-env (6) to structure branch
                self.struct_input_dim += (self.residue_embedding_dim + self.position_embedding_dim + self.micro_env_feature_dim)
                
            logger.info(f"DualBranchFusionPredictor: Enabled explicit features. Seq dim increased to {self.seq_input_dim}, Struct dim increased to {self.struct_input_dim}")

        seq_layers = int(self.config.get("seq_transformer_layers", 2))
        struct_layers = int(self.config.get("struct_transformer_layers", 2))
        seq_heads = int(self.config.get("seq_attention_heads", 4))
        struct_heads = int(self.config.get("struct_attention_heads", 4))
        cross_heads = int(self.config.get("cross_attention_heads", 4))
        cross_bidirectional = bool(self.config.get("cross_attention_bidirectional", True))
        use_attention_pooling = bool(self.config.get("use_attention_pooling", True))
        window_radius = int(self.config.get("center_window_radius", 3))
        use_block_attn_res = bool(self.config.get("use_block_attn_res", False))
        attn_res_block_size = int(self.config.get("attn_res_block_size", 2))
        use_gradient_checkpointing = bool(self.config.get("use_gradient_checkpointing", False))
        sequence_gradient_checkpointing = bool(
            self.config.get("sequence_gradient_checkpointing", use_gradient_checkpointing)
        )
        structure_gradient_checkpointing = bool(
            self.config.get("structure_gradient_checkpointing", use_gradient_checkpointing)
        )
        use_moe = bool(self.config.get("use_moe", False))
        num_experts = int(self.config.get("num_experts", 4))
        top_k_experts = int(self.config.get("top_k_experts", 2))
        if use_moe and use_block_attn_res:
            logger.info("use_moe=true and use_block_attn_res=true: AttnResMoEEncoderLayer will be used.")

        # [Enhanced Components - Attention]
        self.use_enhanced_attention = bool(self.config.get("use_enhanced_attention", False))
        seq_tower_input_dim = self.seq_input_dim
        struct_tower_input_dim = self.struct_input_dim
        
        if self.use_enhanced_attention:
            self.enhanced_dim = int(self.config.get("enhanced_dim", 256))
            logger.info(f"DualBranchFusionPredictor: Enabling Enhanced Attention Module (dim={self.enhanced_dim})")
            
            self.seq_enhanced_proj = nn.Linear(self.seq_input_dim, self.enhanced_dim)
            self.seq_enhanced_norm = nn.LayerNorm(self.enhanced_dim)
            
            self.enhanced_seq_attention = EnhancedAttentionModule(
                input_dim=self.enhanced_dim,
                num_heads=int(self.config.get("enhanced_attention_heads", 4)),
                num_layers=int(self.config.get("enhanced_attention_layers", 2)),
                dropout=self.dropout
            )
            seq_tower_input_dim = self.enhanced_dim
            
            if self.use_structure:
                self.struct_enhanced_proj = nn.Linear(self.struct_input_dim, self.enhanced_dim)
                self.struct_enhanced_norm = nn.LayerNorm(self.enhanced_dim)
                
                self.enhanced_struct_attention = EnhancedAttentionModule(
                    input_dim=self.enhanced_dim,
                    num_heads=int(self.config.get("enhanced_attention_heads", 4)),
                    num_layers=int(self.config.get("enhanced_attention_layers", 2)),
                    dropout=self.dropout
                )
                struct_tower_input_dim = self.enhanced_dim

        self.last_foreground_block = None
        self.last_task_context_proj = None
        self.task_parser: AbstractTaskParser = ConfigDrivenTaskParser(
            task_names=self.ptm_types,
            apply_to_structure_graph=self.last_apply_to_structure_graph,
        )
        self.task_context_generator: Optional[AbstractTaskContextGenerator] = None
        if self.use_last_foreground:
            if self.last_foreground_method in ("stability", "stability_vote", "stability_token", "last", "last_vit"):
                vote_mode = self.last_stability_vote_mode
                if self.last_foreground_method == "stability_vote":
                    vote_mode = "channel"
                elif self.last_foreground_method == "stability_token":
                    vote_mode = "token"
                self.last_foreground_block = LaSTStabilityForegroundBlock(
                    input_dim=seq_tower_input_dim,
                    num_tasks=self.num_tasks,
                    topk_ratio=self.last_stability_topk_ratio,
                    sigma=self.last_stability_sigma,
                    vote_mode=vote_mode,
                    task_conditioned=self.last_task_conditioned,
                    center_prior_weight=self.last_prior_center_weight,
                    access_prior_weight=self.last_prior_access_weight,
                    force_fp32=self.last_force_fp32,
                    sequence_enhance_mode=self.last_sequence_enhance_mode,
                    detach_backbone_for_mask=self.last_detach_backbone_for_mask,
                    task_center_sigmas=self.last_task_mask_sigmas,
                    task_center_offsets=self.last_task_mask_offsets,
                    mask_iou_target_max=self.last_mask_iou_target_max,
                    mask_iou_lambda=self.last_mask_iou_lambda,
                    task_repr_inter_target=self.last_task_repr_inter_target,
                    task_repr_contrastive_lambda=self.last_task_repr_contrastive_lambda,
                    grl_lambda=self.last_grl_lambda,
                    use_prompt_attention=self.last_use_prompt_attention,
                    prompt_orthogonal_lambda=self.last_prompt_orthogonal_lambda,
                    prompt_mi_lambda=self.last_prompt_mi_lambda,
                    attention_sparse_lambda=self.last_attention_sparse_lambda,
                    attention_entropy_target=self.last_attention_entropy_target,
                    attention_topk_ratio=self.last_attention_topk_ratio,
                    attention_guidance_mix=self.last_attention_guidance_mix,
                    attention_decorrelation_lambda=self.last_attention_decorrelation_lambda,
                    attention_tau_lambda=self.last_attention_tau_lambda,
                    attention_tau_target_max=self.last_attention_tau_target_max,
                    attention_focus_lambda=self.last_attention_focus_lambda,
                    attention_focus_target=self.last_attention_focus_target,
                    attention_topk_min_mix=self.last_attention_topk_min_mix,
                    mask_mean_target=self.last_mask_mean_target,
                    mask_mean_tolerance=self.last_mask_mean_tolerance,
                    mask_variance_target=self.last_mask_variance_target,
                    use_token_prior_for_prompt_attention=self.use_token_prior_for_prompt_attention,
                    token_prior_guidance_mix=self.token_prior_guidance_mix,
                    token_prior_blend_ratio=self.token_prior_blend_ratio,
                    token_prior_diversity_lambda=self.token_prior_diversity_lambda,
                    foreground_attention_consistency_lambda=self.foreground_attention_consistency_lambda,
                    token_prior_logit_clip=self.token_prior_logit_clip,
                    task_channel_gate_strength=self.last_task_channel_gate_strength,
                    mask_logit_source=self.last_mask_logit_source,
                    token_prior_mask_logit_mix=self.token_prior_mask_logit_mix,
                )
            else:
                self.last_foreground_block = SpectralForegroundBlock(
                    input_dim=seq_tower_input_dim,
                    num_tasks=self.num_tasks,
                    rank_k=self.last_rank_k,
                    task_conditioned=self.last_task_conditioned,
                    center_prior_weight=self.last_prior_center_weight,
                    access_prior_weight=self.last_prior_access_weight,
                    force_fp32=self.last_force_fp32,
                    sequence_enhance_mode=self.last_sequence_enhance_mode,
                    detach_backbone_for_mask=self.last_detach_backbone_for_mask,
                )
            if self.last_task_context_projector_type in {"task_specific", "task_specific_residual", "anti_collapse"}:
                self.last_task_context_proj = TaskSpecificContextProjector(
                    input_dim=seq_tower_input_dim,
                    output_dim=self.fusion_hidden_dim // 2,
                    num_tasks=self.num_tasks,
                    adapter_rank=self.last_task_context_projector_rank,
                    dropout=self.last_task_context_projector_dropout,
                    task_embed_scale=self.last_task_context_embed_scale,
                    remove_shared_scale=self.last_task_context_remove_shared_scale,
                )
            else:
                self.last_task_context_proj = nn.Sequential(
                    nn.Linear(seq_tower_input_dim, self.fusion_hidden_dim // 2),
                    nn.LayerNorm(self.fusion_hidden_dim // 2),
                )
            self.task_context_generator = LaSTTaskContextGenerator(
                foreground_block=self.last_foreground_block,
                task_context_projector=self.last_task_context_proj,
                mask_sparsity_lambda=self.last_mask_sparsity_lambda,
                mask_entropy_lambda=self.last_mask_entropy_lambda,
                mask_diversity_lambda=self.last_mask_diversity_lambda,
                mask_iou_lambda=self.last_mask_iou_lambda,
                task_repr_contrastive_lambda=self.last_task_repr_contrastive_lambda,
                context_cka_lambda=self.last_context_cka_lambda,
                context_cov_lambda=self.last_context_cov_lambda,
                context_cka_target_max=self.last_context_cka_target_max,
                context_cov_target_max=self.last_context_cov_target_max,
            )

        self.structure_reliability_gate = None
        if self.use_last_structure_reliability:
            self.structure_reliability_gate = StructureReliabilityGate(
                micro_env_dim=self.micro_env_feature_dim,
                num_tasks=self.num_tasks,
                task_conditioned=self.last_structure_task_conditioned,
            )

        self.sequence_tower = SequenceTower(
            input_dim=seq_tower_input_dim,
            hidden_dim=self.branch_hidden_dim,
            num_layers=seq_layers,
            num_heads=seq_heads,
            dropout=self.dropout,
            use_attention_pooling=use_attention_pooling,
            window_radius=window_radius,
            use_block_attn_res=use_block_attn_res,
            attn_res_block_size=attn_res_block_size,
            use_gradient_checkpointing=sequence_gradient_checkpointing,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
        )

        # [New] Learnable post-tower task attention (config-gated; LaST path untouched).
        self.use_learnable_task_attention = bool(self.config.get("use_learnable_task_attention", False))
        self.learnable_task_attention = None
        if self.use_learnable_task_attention:
            pos_prior = None
            prior_path = self.config.get("learnable_task_attention_pos_prior_path")
            if prior_path:
                import numpy as _np
                prior_np = _np.load(str(prior_path)).astype("float32")
                if prior_np.shape[0] != self.num_tasks:
                    raise ValueError(
                        f"pos prior has {prior_np.shape[0]} tasks, model has {self.num_tasks}"
                    )
                pos_prior = torch.from_numpy(prior_np) * float(
                    self.config.get("learnable_task_attention_pos_prior_scale", 1.0)
                )
                logger.info("LearnableTaskAttention: loaded PSSM position prior %s shape=%s", prior_path, tuple(pos_prior.shape))
            self.learnable_task_attention = LearnableTaskAttentionPool(
                context_dim=self.branch_hidden_dim,
                num_tasks=self.num_tasks,
                output_dim=self.fusion_hidden_dim // 2,
                query_dim=int(self.config.get("learnable_task_attention_query_dim", 128)),
                dropout=float(self.config.get("learnable_task_attention_dropout", 0.0)),
                pos_bias_init=pos_prior,
                temperature_init=float(self.config.get("learnable_task_attention_temperature_init", 0.1)),
            )
            self.learnable_task_attention_scale = float(self.config.get("learnable_task_attention_scale", 1.0))

        self.structure_tower = None
        self.structure_graph_tower = None
        self.structure_graph_builder: Optional[AbstractStructureGraphBuilder] = None
        self.graph_projection = None
        self.tri_modal_reliability_gate = None
        self._graph_node_dim_warning_emitted = False
        self._graph_edge_dim_warning_emitted = False
        if self.use_structure_graph:
            graph_input_dim = self.graph_node_dim if self.graph_node_dim > 0 else struct_tower_input_dim
            if self.graph_tower_type in {"se3", "se3_transformer", "se3-transformer"}:
                if SE3TransformerPocketEncoder is None:
                    raise ImportError("graph_tower_type=se3_transformer but SE3TransformerPocketEncoder is unavailable.")
                self.structure_graph_tower = SE3TransformerPocketEncoder(
                    input_dim=graph_input_dim,
                    hidden_dim=self.graph_hidden_dim,
                    layers=self.graph_layers,
                    edge_dim=max(1, int(self.graph_edge_dim)),
                    heads=int(self.config.get("graph_heads", 8)),
                    vector_channels=int(self.config.get("graph_vector_channels", 16)),
                    dropout=float(self.config.get("graph_dropout", 0.1)),
                )
            else:
                if PocketEGNN is None:
                    raise ImportError("use_structure_graph=True but PocketEGNN is unavailable.")
                self.structure_graph_tower = PocketEGNN(
                    input_dim=graph_input_dim,
                    hidden_dim=self.graph_hidden_dim,
                    layers=self.graph_layers,
                    edge_dim=max(1, int(self.graph_edge_dim)),
                )
            self.structure_graph_builder = DefaultStructureGraphBuilder(
                structure_graph_tower=self.structure_graph_tower,
                graph_vote_mode=self.last_graph_vote_mode,
                graph_topk_ratio=self.last_graph_topk_ratio,
                graph_sigma=self.last_graph_sigma,
                graph_enhance_scale=self.last_graph_enhance_scale,
                logger=logger,
                return_graph_nodes=self.use_token_aligned_graph_fusion,
            )
            if self.graph_hidden_dim == self.branch_hidden_dim:
                self.graph_projection = nn.Identity()
            else:
                self.graph_projection = nn.Sequential(
                    nn.Linear(self.graph_hidden_dim, self.branch_hidden_dim),
                    nn.LayerNorm(self.branch_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                )
            if self.use_token_aligned_graph_fusion:
                if self.graph_hidden_dim == self.branch_hidden_dim:
                    self.graph_token_proj = nn.Identity()
                else:
                    self.graph_token_proj = nn.Linear(self.graph_hidden_dim, self.branch_hidden_dim)
                self.graph_token_norm = nn.LayerNorm(self.branch_hidden_dim)
                # Start with a small but non-zero contribution so a 10-epoch
                # pilot can actually exercise the fusion path.
                self.graph_token_scale = nn.Parameter(torch.full((1,), 0.1))
                self.graph_token_global_scale = nn.Parameter(torch.full((1,), 0.1))
        if self.use_structure and (not self.use_structure_graph or self.use_dense_structure_tower_with_graph):
            self.structure_tower = StructureTower(
                input_dim=struct_tower_input_dim,
                hidden_dim=self.branch_hidden_dim,
                dropout=self.dropout,
                num_layers=struct_layers,
                num_heads=struct_heads,
                use_attention_pooling=use_attention_pooling,
                window_radius=window_radius,
                use_block_attn_res=use_block_attn_res,
                attn_res_block_size=attn_res_block_size,
                use_gradient_checkpointing=structure_gradient_checkpointing,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
            )
        if self.use_dense_structure_tower_with_graph:
            self.tri_modal_reliability_gate = nn.Sequential(
                nn.Linear(
                    self.branch_hidden_dim * 3 + self.micro_env_feature_dim + 2,
                    self.branch_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.branch_hidden_dim, 3),
            )

        self.cross_modal = None
        if self.structure_tower is not None:
            self.cross_modal = CrossModalFusionLayer(
                hidden_dim=self.branch_hidden_dim,
                num_heads=cross_heads,
                dropout=self.dropout,
                bidirectional=cross_bidirectional,
            )

        self.gating = AdaptiveGating(self.branch_hidden_dim)
        self.use_global_context_gate = bool(self.config.get("use_global_context_gate", True))
        self.global_context_fallback_scale = float(self.config.get("global_context_fallback_scale", 0.2))
        self.fusion_mode = str(self.config.get("fusion_mode", "gated")).strip().lower()
        self.use_adaptive_fusion_gate = bool(
            self.config.get("use_adaptive_fusion_gate", self.fusion_mode not in {"late", "concat", "static"})
        )
        self.fusion_feature_set = str(self.config.get("fusion_feature_set", "extended")).strip().lower()
        if self.fusion_feature_set not in {"compact", "extended"}:
            raise ValueError("fusion_feature_set must be 'compact' or 'extended'")

        # Projections for auxiliary features
        self.local_projection = nn.Sequential(
            nn.Linear(self.local_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        # Explicit centre-Lys and multi-scale context pooling (±3/±7/±15).
        self.local_center_scale_logits = nn.Parameter(torch.tensor([1.8, 1.2, 0.7, 0.3]))
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.structure_global_projection = nn.Sequential(
            nn.Linear(self.struct_global_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.global_context_gate = nn.Sequential(
            nn.Linear(self.branch_hidden_dim * 2 + 1, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Linear(self.branch_hidden_dim, 1),
        )
        self.seq_only_projection = nn.Sequential(
            nn.Linear(self.branch_hidden_dim * 3 + self.micro_env_feature_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.use_ptm_multimodal_cross_attention = bool(self.config.get("use_ptm_multimodal_cross_attention", False))
        self.ptm_multimodal_context_scale = float(self.config.get("ptm_multimodal_context_scale", 0.18))
        self.ptm_multimodal_target_context_scale = float(self.config.get("ptm_multimodal_target_context_scale", 0.32))
        self.ptm_multimodal_target_tasks = [
            str(x).strip().lower()
            for x in self.config.get("ptm_multimodal_target_tasks", ["methylation", "crotonylation", "succinylation"])
        ]
        self.ptm_multimodal_attention = None
        self.ptm_multimodal_micro_projection = None
        self.ptm_multimodal_to_task_context = None
        self.ptm_multimodal_to_shared = None
        if self.use_ptm_multimodal_cross_attention:
            self.ptm_multimodal_attention = PTMMultimodalCrossAttention(
                hidden_dim=self.branch_hidden_dim,
                num_tasks=self.num_tasks,
                task_names=self.ptm_types,
                num_heads=int(self.config.get("ptm_multimodal_attention_heads", self.config.get("cross_attention_heads", 4))),
                dropout=float(self.config.get("ptm_multimodal_attention_dropout", self.dropout)),
                target_tasks=self.ptm_multimodal_target_tasks,
                focus_lambda=float(self.config.get("ptm_mm_attention_focus_lambda", 0.0)),
                focus_target=float(self.config.get("ptm_mm_attention_focus_target", 0.06)),
                entropy_lambda=float(self.config.get("ptm_mm_attention_entropy_lambda", 0.0)),
                entropy_target=float(self.config.get("ptm_mm_attention_entropy_target", 3.85)),
                decorrelation_lambda=float(self.config.get("ptm_mm_attention_decorrelation_lambda", 0.0)),
                decorrelation_target_max=float(self.config.get("ptm_mm_attention_decorrelation_target_max", 0.82)),
                guidance_lambda=float(self.config.get("ptm_mm_seq_guidance_lambda", 0.0)),
            )
            self.ptm_multimodal_micro_projection = nn.Sequential(
                nn.Linear(self.micro_env_feature_dim, self.branch_hidden_dim),
                nn.LayerNorm(self.branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            self.ptm_multimodal_local_window_projection = nn.Sequential(
                nn.Linear(self.local_input_dim, self.branch_hidden_dim),
                nn.LayerNorm(self.branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            self.ptm_multimodal_local_window_radius = int(self.config.get("ptm_multimodal_local_window_radius", 4))
            self.ptm_multimodal_to_task_context = nn.Sequential(
                nn.Linear(self.branch_hidden_dim, self.fusion_hidden_dim // 2),
                nn.LayerNorm(self.fusion_hidden_dim // 2),
            )
            self.ptm_multimodal_to_shared = nn.Sequential(
                nn.Linear(self.branch_hidden_dim, self.fusion_hidden_dim // 2),
                nn.LayerNorm(self.fusion_hidden_dim // 2),
            )

        if self.use_legacy_fusion:
            self.bilinear_interaction = nn.Bilinear(
                self.branch_hidden_dim,
                self.branch_hidden_dim,
                self.branch_hidden_dim,
            )
            fusion_input_dim = self.branch_hidden_dim * 8
        elif self.fusion_feature_set == "compact":
            self.bilinear_interaction = None
            fusion_input_dim = self.branch_hidden_dim * 4 + self.micro_env_feature_dim
        else:
            self.bilinear_interaction = nn.Bilinear(
                self.branch_hidden_dim,
                self.branch_hidden_dim,
                self.branch_hidden_dim,
            )
            fusion_input_dim = self.branch_hidden_dim * 8 + self.micro_env_feature_dim
            
        # [NEW] 采用残差结构的融合网络，更好地保留分支特征并促进梯度流动
        self.fusion_proj = nn.Linear(fusion_input_dim, self.fusion_hidden_dim)
        self.fusion_residual_block = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
        )
        self.fusion_residual_mixer = DepthAttentionResidual(self.fusion_hidden_dim, dropout=self.dropout)
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
            nn.LayerNorm(self.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout / 2.0),
        )
        self.classifier = nn.Linear(self.fusion_hidden_dim // 2, 1)

        # Auxiliary objectives
        self.alignment_weight = float(self.config.get("alignment_weight", 0.0))
        self.contrastive_weight = float(self.config.get("contrastive_weight", 0.0))
        self.similarity_weight = float(self.config.get("structure_similarity_weight", 0.0))
        self.aux_cls_weight = float(self.config.get("aux_cls_weight", 0.0))  # New: Auxiliary classification weight
        self.temperature = float(self.config.get("contrastive_temperature", 0.07))
        self.contrastive_label_smoothing = float(self.config.get("contrastive_label_smoothing", 0.05))
        self.contrastive_hard_negative_ratio = float(self.config.get("contrastive_hard_negative_ratio", 0.3))
        self.contrastive_target_ptm = str(self.config.get("contrastive_target_ptm", "acetylation")).strip()

        enable_branch_cls_default = self.aux_cls_weight > 0 or self.branch_warmup_use_branch_logits
        self.enable_branch_classifiers = bool(self.config.get("enable_branch_classifiers", enable_branch_cls_default))

        if self.enable_branch_classifiers:
            self.seq_classifier = nn.Linear(self.branch_hidden_dim, 1)
            if self.use_structure:
                self.struct_classifier = nn.Linear(self.branch_hidden_dim, 1)
            else:
                self.struct_classifier = None
        else:
            self.seq_classifier = None
            self.struct_classifier = None

        if self.contrastive_weight > 0:
            projection_dim = int(self.config.get("contrastive_projection_dim", 128))
            self.projection_head = ProjectionHead(
                input_dim=self.fusion_hidden_dim // 2,
                projection_dim=projection_dim,
                dropout=self.dropout,
            )
            # Use Advanced Contrastive Loss
            from src.utils.advanced_contrastive_loss import AdvancedSupConLoss
            self.contrastive_criterion = AdvancedSupConLoss(
                temperature=self.temperature, 
                label_smoothing=self.contrastive_label_smoothing,
                hard_negative_ratio=self.contrastive_hard_negative_ratio,
            )
        else:
            self.projection_head = None
            self.contrastive_criterion = None

        # Learning rate scaling for stage-wise optimization
        self.seq_lr_scale = float(self.config.get("lr_scale_seq", 0.3))
        self.struct_lr_scale = float(self.config.get("lr_scale_struct", 1.0))
        self.fusion_lr_scale = float(self.config.get("lr_scale_fusion", 1.0))
        self.training_stage = "joint"
        self.branch_warmup_active = False
        self._structure_all_mask_warning_emitted = False

        self._init_center_regularization_dual()

        # [NEW] Integrate PTMSpecialistExtractors for long-tail PTMs
        from .ptm_specialists import PTMSpecialistExtractors
        self.use_ptm_specialists = bool(
            self.config.get("use_base_ptm_specialists", self.config.get("use_ptm_specialists", True))
        )
        self.specialist_extractors = None
        specialist_feature_dim = self.seq_input_dim // 4
        specialist_concat_dim = specialist_feature_dim * 3
        if self.use_ptm_specialists:
            logger.info("DualBranchFusionPredictor: Enabling PTM Specialists for long-tail classes.")
            self.specialist_extractors = PTMSpecialistExtractors(self.seq_input_dim)

        self.use_ptm_feature_separator = bool(self.config.get("use_ptm_feature_separator", False))
        self.separator_metric_use_loss_weight_mask = bool(
            self.config.get("separator_metric_use_loss_weight_mask", False)
        )
        self.separator_metric_min_sample_weight = float(
            self.config.get("separator_metric_min_sample_weight", 0.75)
        )
        self.separator_logit_fusion = str(
            self.config.get("separator_logit_fusion", "separator")
        ).strip().lower()
        self.separator_logit_base_weight = float(self.config.get("separator_logit_base_weight", 0.35))
        self.separator_logit_private_weight = float(self.config.get("separator_logit_private_weight", 0.65))
        self.separator_aux_logit_loss_weight = float(self.config.get("separator_aux_logit_loss_weight", 0.0))
        self.base_aux_logit_loss_weight = float(self.config.get("base_aux_logit_loss_weight", 0.0))
        self.separator_logit_gate = None
        if self.separator_logit_fusion in {"learned_gate", "gate", "gated"}:
            init_private = float(self.config.get("separator_logit_initial_private_weight", 0.65))
            init_private = min(0.95, max(0.05, init_private))
            init_logit = math.log(init_private / (1.0 - init_private))
            self.separator_logit_gate = nn.Parameter(torch.full((self.num_tasks,), init_logit))
        self.ptm_separator = None
        if self.use_ptm_feature_separator:
            logger.info("DualBranchFusionPredictor: Enabling PTMDisentangledSeparator.")
            self.ptm_separator = PTMDisentangledSeparator(
                base_dim=self.fusion_hidden_dim // 2,
                num_tasks=self.num_tasks,
                task_names=self.ptm_types,
                context_dim=self.fusion_hidden_dim // 2,
                specialist_dim=specialist_feature_dim if self.use_ptm_specialists else 0,
                adapter_rank=int(self.config.get("separator_adapter_rank", self.config.get("task_adapter_dim", 64))),
                dropout=float(self.config.get("separator_gate_dropout", self.dropout / 2.0)),
                cka_lambda=float(self.config.get("separator_cka_lambda", 0.0)),
                cka_target_max=float(self.config.get("separator_cka_target_max", 0.75)),
                cov_lambda=float(self.config.get("separator_cov_lambda", 0.0)),
                cov_target_max=float(self.config.get("separator_cov_target_max", 0.08)),
                orthogonal_lambda=float(self.config.get("separator_orthogonal_lambda", 0.0)),
                orthogonal_target_max=float(self.config.get("separator_orthogonal_target_max", 0.15)),
                metric_lambda=float(self.config.get("separator_metric_lambda", 0.0)),
                metric_margin=float(self.config.get("separator_metric_margin", 0.25)),
                metric_min_positives=int(self.config.get("separator_metric_min_positives", 2)),
                ranking_lambda=float(self.config.get("separator_ranking_lambda", self.config.get("ranking_loss_weight", 0.0))),
                ranking_margin=float(self.config.get("separator_ranking_margin", self.config.get("ranking_margin", 0.2))),
                ranking_target_tasks=self.config.get(
                    "separator_ranking_target_tasks",
                    self.config.get("ranking_target_tasks", []),
                ),
            )
            self.classifier = nn.Linear(self.fusion_hidden_dim // 2, self.num_tasks)
        elif self.use_ptm_specialists:
            self.classifier = nn.Linear((self.fusion_hidden_dim // 2) + specialist_concat_dim, self.num_tasks)
        else:
            self.classifier = nn.Linear(self.fusion_hidden_dim // 2, self.num_tasks)

        # [Enhanced Components Initialization]
        self.use_multiscale_fusion = bool(self.config.get("use_multiscale_fusion", False))
        self.use_dynamic_feature_selection = bool(self.config.get("use_dynamic_feature_selection", False))
        if self.fusion_feature_set == "compact" and (
            self.use_multiscale_fusion or self.use_dynamic_feature_selection
        ):
            raise ValueError(
                "compact fusion is incompatible with multiscale fusion and dynamic feature selection"
            )

        if self.use_multiscale_fusion:
            logger.info("DualBranchFusionPredictor: Enabling Multi-Scale Fusion Network")
            # MultiScaleFusionNetwork expects sequence inputs, so we use branch_hidden_dim (output of towers)
            self.multiscale_fusion_net = MultiScaleFusionNetwork(
                seq_dim=self.branch_hidden_dim,
                struct_dim=self.branch_hidden_dim,
                hidden_dims=[512, 256, 128], # Configurable?
                dropout=self.dropout
            )
            # Update fusion dimension because MultiScaleFusionNetwork returns hidden_dims[0] (512)
            # We will use this INSTEAD OF or IN ADDITION TO the standard gated fusion?
            # Let's add it to the fusion inputs.
            # fusion_input_dim calculation needs to be updated in forward or here.
            # But self.fusion_proj is already initialized above. We might need to adjust it.
            # To avoid breaking existing weights if not used, we'll handle dimension mismatch dynamically or re-init if needed.
            # Actually, better to re-init fusion_proj here if we change the input size.
            
            ms_out_dim = 512 # hidden_dims[0]
            # We will replace 'fused_global' (branch_hidden_dim) with ms_out_dim or append it?
            # Let's append it.
            # New fusion_input_dim = old_fusion_input_dim + ms_out_dim
            # But wait, fusion_proj is already created. 
            # I should re-create fusion_proj if this is enabled.
            
            # Recalculate fusion_input_dim
            # Standard: fused(256) + seq(256) + struct(256) + diff(256) + inter(256) + local(256) + global(256) + micro(6) = 1798
            # With MS Fusion: + 512 = 2310
            
            current_dim = self.fusion_proj.in_features
            new_dim = current_dim + ms_out_dim
            
            self.fusion_proj = nn.Linear(new_dim, self.fusion_hidden_dim)
            logger.info(f"DualBranchFusionPredictor: Re-initialized fusion_proj with dim {new_dim}")

        if self.use_dynamic_feature_selection:
            logger.info("DualBranchFusionPredictor: Enabling Dynamic Feature Selection")
            # We will select from: Seq, Struct, Local, Global, (optional MS Fusion)
            # Let's define the feature dims
            feature_dims = {
                'seq': self.branch_hidden_dim,
                'struct': self.branch_hidden_dim,
                'local': self.branch_hidden_dim,
                'global': self.branch_hidden_dim,
                'struct_global': self.branch_hidden_dim,
            }
            if self.use_multiscale_fusion:
                feature_dims['multiscale'] = 512 # ms_out_dim
            else:
                feature_dims['gated'] = self.branch_hidden_dim # standard gated fusion
                
            self.feature_selector = DynamicFeatureSelector(
                feature_dims=feature_dims,
                hidden_dim=256
            )
            
            # If using selector, the fusion_proj input dim becomes sum(feature_dims.values()) + others?
            # Or we replace the concatenation logic.
            # The selector returns fused features (concatenated weighted).
            # Then we pass this to fusion_proj.
            # So fusion_input_dim = sum(feature_dims.values()) + others (like diff, interaction, micro_env)
            
            # Let's assume we keep diff, interaction, micro_env outside of selection (or include them?)
            # Report says: "Dynamic Feature Selector" for "sequence, structure, local, global".
            # So we select these, and then concat with interaction terms.
            
            selector_out_dim = sum(feature_dims.values())
            # Interaction terms: diff(256) + inter(256) + micro(6) = 518
            # Total = selector_out_dim + 518
            
            new_dim_selector = selector_out_dim + self.branch_hidden_dim * 2 + 6
            if self.use_legacy_fusion:
                 # Legacy didn't have micro_env? 
                 pass
                 
            self.fusion_proj = nn.Linear(new_dim_selector, self.fusion_hidden_dim)
            logger.info(f"DualBranchFusionPredictor: Re-initialized fusion_proj for Feature Selector with dim {new_dim_selector}")

    def _init_center_regularization_dual(self) -> None:
        self.center_mask_prob = float(self.config.get("center_mask_prob", 0.0))
        self.center_mask_noise_std = float(self.config.get("center_mask_noise_std", 0.0))
        self.center_mask_token_seq = None
        self.center_mask_token_struct = None
        if self.center_mask_prob > 0.0:
            self.center_mask_token_seq = nn.Parameter(torch.zeros(self.seq_input_dim))
            nn.init.normal_(self.center_mask_token_seq, mean=0.0, std=0.02)
            if self.use_structure:
                self.center_mask_token_struct = nn.Parameter(torch.zeros(self.struct_input_dim))
                nn.init.normal_(self.center_mask_token_struct, mean=0.0, std=0.02)

        self.center_curriculum_cfg = self.config.get("center_curriculum", {}) or {}
        self.center_curriculum_enabled = bool(self.center_curriculum_cfg)
        default_alpha = float(
            self.center_curriculum_cfg.get(
                "initial_alpha",
                0.0 if self.center_curriculum_enabled else 1.0,
            )
        )
        self.center_curriculum_eval_alpha = float(self.center_curriculum_cfg.get("eval_alpha", 1.0))
        self._active_center_alpha = default_alpha if self.center_curriculum_enabled else 1.0

    def update_center_curriculum(self, epoch: float) -> None:
        if not self.center_curriculum_enabled:
            return
        self._active_center_alpha = self._compute_center_alpha_dual(float(epoch))

    def set_center_alpha(self, alpha: float) -> None:
        self._active_center_alpha = float(alpha)

    def _resolve_center_alpha(self, epoch_override=None, alpha_override=None) -> float:
        if alpha_override is not None:
            self.set_center_alpha(alpha_override)
        elif epoch_override is not None:
            self.update_center_curriculum(epoch_override)
        elif not self.training and self.center_curriculum_enabled:
            self._active_center_alpha = self.center_curriculum_eval_alpha
        return float(self._active_center_alpha)

    def _compute_center_alpha_dual(self, epoch_value: float) -> float:
        cfg = self.center_curriculum_cfg
        start_alpha = float(cfg.get("start_alpha", cfg.get("initial_alpha", 0.0)))
        target_alpha = float(cfg.get("target_alpha", cfg.get("alpha_target", 1.0)))
        warmup = max(0, int(cfg.get("warmup_epochs", 0)))
        ramp = max(1, int(cfg.get("ramp_epochs", 1)))
        if epoch_value < warmup:
            return start_alpha
        progress = min(1.0, (epoch_value - warmup) / float(ramp))
        return start_alpha + (target_alpha - start_alpha) * progress

    def _apply_center_mask_if_needed_dual(
        self,
        seq_tensor: torch.Tensor,
        local_tensor: Optional[torch.Tensor],
        struct_tensor: Optional[torch.Tensor],
    ):
        # [NaN Debug] Check inputs
        if torch.isnan(seq_tensor).any():
            logger.warning("NaN detected in seq_tensor BEFORE center mask application. Applying nan_to_num.")
            seq_tensor = torch.nan_to_num(seq_tensor)

        if (
            not self.training
            or self.center_mask_prob <= 0.0
            or seq_tensor is None
            or not isinstance(seq_tensor, torch.Tensor)
            or seq_tensor.dim() < 3
        ):
            return seq_tensor, local_tensor, struct_tensor

        # Ensure mask probability is valid
        prob = max(0.0, min(1.0, self.center_mask_prob))
        
        mask_flags = torch.rand(seq_tensor.size(0), device=seq_tensor.device) < prob
        if not mask_flags.any():
            return seq_tensor, local_tensor, struct_tensor

        seq_tensor = self._mask_center_tensor_dual(seq_tensor, mask_flags, self.center_mask_token_seq)
        if isinstance(local_tensor, torch.Tensor) and local_tensor.dim() >= 3:
            local_tensor = self._mask_center_tensor_dual(local_tensor, mask_flags, self.center_mask_token_seq)
        if (
            isinstance(struct_tensor, torch.Tensor)
            and struct_tensor.dim() >= 3
            and self.center_mask_token_struct is not None
        ):
            struct_tensor = self._mask_center_tensor_dual(struct_tensor, mask_flags, self.center_mask_token_struct)

        # [NaN Debug] Check outputs
        if torch.isnan(seq_tensor).any():
            logger.warning("NaN detected in seq_tensor AFTER center mask application.")
            
        return seq_tensor, local_tensor, struct_tensor

    @staticmethod
    def _mask_center_tensor_dual(
        tensor: torch.Tensor,
        mask_flags: torch.Tensor,
        mask_token: Optional[torch.nn.Parameter],
    ) -> torch.Tensor:
        if mask_token is None:
            return tensor
        center_idx = tensor.size(1) // 2
        masked = tensor.clone()
        token = mask_token.to(dtype=masked.dtype, device=masked.device)
        masked[mask_flags, center_idx, :] = token
        return masked

    def _extract_masks(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor.abs().sum(dim=-1) == 0)

    def _safe_bilinear_interaction(self, seq_global: torch.Tensor, struct_global: torch.Tensor) -> torch.Tensor:
        """Numerically stable bilinear interaction to reduce NaN/Inf gradients under AMP."""
        if self.bilinear_interaction is None:
            return seq_global.new_zeros(seq_global.shape)

        in_clip = float(self.config.get("interaction_input_clip", 50.0))
        out_clip = float(self.config.get("interaction_output_clip", 50.0))

        seq_safe = torch.nan_to_num(seq_global, nan=0.0, posinf=in_clip, neginf=-in_clip).clamp(-in_clip, in_clip)
        struct_safe = torch.nan_to_num(struct_global, nan=0.0, posinf=in_clip, neginf=-in_clip).clamp(-in_clip, in_clip)

        if seq_safe.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                interaction = self.bilinear_interaction(seq_safe.float(), struct_safe.float())
            interaction = interaction.to(dtype=seq_global.dtype)
        else:
            interaction = self.bilinear_interaction(seq_safe, struct_safe)

        interaction = torch.nan_to_num(interaction, nan=0.0, posinf=out_clip, neginf=-out_clip).clamp(-out_clip, out_clip)
        return interaction

    def _collect_moe_load_balance_loss(self) -> Optional[torch.Tensor]:
        losses: List[torch.Tensor] = []
        for module in self.modules():
            loss = getattr(module, "last_load_balance_loss", None)
            if isinstance(loss, torch.Tensor):
                if loss.numel() == 0:
                    continue
                if torch.isfinite(loss).all():
                    losses.append(loss.mean())
        if not losses:
            return None
        return torch.stack(losses, dim=0).mean()

    def _encode_sequences_with_lora(
        self,
        sequences: List[str],
        device: torch.device,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.sequence_encoder is None:
            raise RuntimeError("Sequence encoder is not initialized for LoRA.")
        if next(self.sequence_encoder.parameters()).device != device:
            self.sequence_encoder.to(device)
        seq_list = [str(s) if s is not None else "" for s in sequences]
        if target_len is not None and int(target_len) > 0:
            max_len = int(target_len)
        else:
            configured_max_len = int(self.esm_lora_max_len) if int(self.esm_lora_max_len) > 0 else 0
            observed_max_len = max(1, max(len(s) for s in seq_list)) if seq_list else 1
            max_len = min(configured_max_len, observed_max_len) if configured_max_len > 0 else observed_max_len
        token_max_len = max_len + 2
        embeddings, attn_mask = self.sequence_encoder(seq_list, max_len=token_max_len, return_mask=True)
        embeddings = embeddings.to(device)
        if isinstance(attn_mask, torch.Tensor):
            attn_mask = attn_mask.to(device)

        trimmed_list: List[torch.Tensor] = []
        for i, seq in enumerate(seq_list):
            emb = embeddings[i]
            seq_len = len(seq)
            valid_len = int(attn_mask[i].sum().item()) if isinstance(attn_mask, torch.Tensor) else emb.size(0)
            if seq_len <= 0:
                trimmed = emb[:0]
            else:
                start = 1 if valid_len >= seq_len + 1 else 0
                if valid_len >= seq_len + 2:
                    start = 1
                end = min(start + seq_len, emb.size(0))
                trimmed = emb[start:end]
            trimmed_list.append(trimmed)

        out = embeddings.new_zeros((len(trimmed_list), max_len, embeddings.size(-1)))
        for i, trimmed in enumerate(trimmed_list):
            if trimmed.numel() == 0:
                continue
            length = min(max_len, trimmed.size(0))
            out[i, :length] = trimmed[:length]
        return out

    def _apply_last_to_graph_x(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 1:
            return x, {}
        if not isinstance(batch_index, torch.Tensor) or batch_index.ndim != 1 or batch_index.shape[0] != x.shape[0]:
            return x, {}
        vote_mode = str(self.last_graph_vote_mode).strip().lower()
        topk_ratio = float(self.last_graph_topk_ratio)
        sigma = float(self.last_graph_sigma)
        eps = 1e-6
        enhance_scale = float(self.last_graph_enhance_scale)

        with torch.amp.autocast(device_type="cuda", enabled=False) if x.is_cuda else contextlib.nullcontext():
            x32 = x.float()
            order = torch.argsort(batch_index)
            x_sorted = x32[order]
            b_sorted = batch_index[order]
            bsz = int(b_sorted.max().item()) + 1 if b_sorted.numel() else 0
            counts = torch.bincount(b_sorted, minlength=max(1, bsz)).tolist() if bsz > 0 else []
            out = x32.clone()

            g = torch.fft.fftfreq(x32.shape[1], device=x32.device, dtype=torch.float32)
            sigma_safe = max(1e-6, sigma)
            lowpass = torch.exp(-0.5 * (g / sigma_safe) ** 2)
            lowpass = (lowpass - lowpass.min()) / (lowpass.max() - lowpass.min() + 1e-12)
            lowpass = lowpass.to(dtype=torch.complex64)

            mean_scores: List[torch.Tensor] = []
            peak_scores: List[torch.Tensor] = []
            sparsity_scores: List[torch.Tensor] = []
            offset = 0
            for c in counts:
                if c <= 0:
                    continue
                idx = order[offset : offset + c]
                xb = x32[idx]
                offset += c

                if xb.shape[0] <= 1:
                    score = xb.new_ones((xb.shape[0],))
                else:
                    x_fft = torch.fft.fft(xb, dim=-1)
                    x_hat = torch.fft.ifft(x_fft * lowpass.view(1, -1), dim=-1).real
                    s = x_hat / (torch.abs(x_hat - xb) + eps)
                    if vote_mode == "channel":
                        k = max(1, min(int(xb.shape[0]), int(round(int(xb.shape[0]) * topk_ratio))))
                        idx_top = torch.topk(s, k=k, dim=0, largest=True, sorted=False).indices
                        idx_flat = idx_top.reshape(-1)
                        votes = torch.zeros((int(xb.shape[0]),), device=xb.device, dtype=torch.float32)
                        votes.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=votes.dtype))
                        score = votes / votes.max().clamp(min=1.0)
                    else:
                        token_score = s.mean(dim=-1)
                        token_score = (token_score - token_score.mean()) / (token_score.std() + 1e-6)
                        score = torch.sigmoid(token_score)

                out[idx] = xb * (1.0 + enhance_scale * score.unsqueeze(-1))
                mean_scores.append(score.mean())
                peak_scores.append(score.max())
                sparsity_scores.append((score < 0.5).float().mean())

            metrics: Dict[str, torch.Tensor] = {}
            if mean_scores:
                metrics["last_graph_score_mean"] = torch.stack(mean_scores).mean().detach()
                metrics["last_graph_score_peak"] = torch.stack(peak_scores).mean().detach()
                metrics["last_graph_score_sparsity"] = torch.stack(sparsity_scores).mean().detach()
            return out.to(dtype=x.dtype), metrics

    @staticmethod
    def _task_weighted_summary(
        context: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(context, torch.Tensor) or not isinstance(weights, torch.Tensor):
            return None
        context_f = context.float()
        weights_f = weights.float()
        if weights_f.ndim != 3 or context_f.ndim != 3:
            return None
        if padding_mask is not None and padding_mask.ndim == 2:
            weights_f = weights_f.masked_fill(padding_mask.unsqueeze(1), 0.0)
        weights_f = weights_f / weights_f.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        summary = torch.bmm(weights_f, context_f)
        return summary.to(dtype=context.dtype), weights_f.to(dtype=context.dtype)

    def _fuse_aligned_graph_tokens(
        self,
        seq_context: torch.Tensor,
        seq_global: torch.Tensor,
        seq_mask,
        node_embeddings: torch.Tensor,
        node_batch_index: torch.Tensor,
        node_token_index: torch.Tensor,
    ):
        """Mean-pool graph nodes onto their 73-window tokens and add them to the sequence tokens."""
        if node_embeddings is None or node_embeddings.numel() == 0:
            return seq_context, seq_global
        batch, tokens, dim = seq_context.shape
        device = seq_context.device
        node_embeddings = node_embeddings.to(device=device, dtype=seq_context.dtype)
        node_batch_index = node_batch_index.to(device=device, dtype=torch.long)
        node_token_index = node_token_index.to(device=device, dtype=torch.long)
        valid = (
            (node_token_index >= 0) & (node_token_index < tokens)
            & (node_batch_index >= 0) & (node_batch_index < batch)
        )
        if not bool(valid.any()):
            return seq_context, seq_global
        node_embeddings = node_embeddings[valid]
        flat_index = node_batch_index[valid] * tokens + node_token_index[valid]
        aligned = node_embeddings.new_zeros((batch * tokens, dim))
        counts = node_embeddings.new_zeros((batch * tokens, 1))
        aligned.index_add_(0, flat_index, node_embeddings)
        counts.index_add_(0, flat_index, node_embeddings.new_ones((node_embeddings.shape[0], 1)))
        aligned = (aligned / counts.clamp_min(1.0)).view(batch, tokens, dim)
        mask = (counts.view(batch, tokens, 1) > 0).to(dtype=seq_context.dtype)
        if isinstance(seq_mask, torch.Tensor) and seq_mask.ndim == 2:
            mask = mask * seq_mask.to(device=device).unsqueeze(-1).to(dtype=mask.dtype)
        aligned = self.graph_token_norm(self.graph_token_proj(aligned))
        seq_context = seq_context + self.graph_token_scale * aligned * mask
        token_global = (aligned * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        seq_global = seq_global + self.graph_token_global_scale * token_global
        return seq_context, seq_global

    def _build_task_aware_multimodal_context(
        self,
        *,
        seq_context: Optional[torch.Tensor],
        struct_context: Optional[torch.Tensor],
        fg_mask: Optional[torch.Tensor],
        seq_mask: Optional[torch.Tensor] = None,
        struct_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        seq_summary_out = self._task_weighted_summary(seq_context, fg_mask, padding_mask=seq_mask)
        if seq_summary_out is None:
            return None
        seq_task_summary, seq_task_weights = seq_summary_out
        if not isinstance(struct_context, torch.Tensor):
            focus = seq_task_weights.max(dim=-1).values.mean(dim=-1, keepdim=True)
            zeros = seq_task_summary.new_zeros((seq_task_summary.shape[0], 1))
            task_stats = torch.cat([focus, zeros, zeros, zeros], dim=-1)
            return {
                "seq_task_summary": seq_task_summary,
                "struct_task_summary": None,
                "seq_task_weights": seq_task_weights,
                "struct_task_attention": None,
                "task_stats": task_stats,
            }

        seq_task_norm = F.normalize(seq_task_summary.float(), dim=-1, eps=1e-6)
        struct_norm = F.normalize(struct_context.float(), dim=-1, eps=1e-6)
        struct_logits = torch.bmm(seq_task_norm, struct_norm.transpose(1, 2))
        if isinstance(struct_mask, torch.Tensor) and struct_mask.ndim == 2:
            struct_logits = struct_logits.masked_fill(struct_mask.unsqueeze(1), -10000.0)
        struct_task_attention = F.softmax(struct_logits, dim=-1)
        struct_task_summary = torch.bmm(struct_task_attention, struct_context.float()).to(dtype=seq_context.dtype)

        struct_task_norm = F.normalize(struct_task_summary.float(), dim=-1, eps=1e-6)
        seq_context_norm = F.normalize(seq_context.float(), dim=-1, eps=1e-6)
        seq_logits = torch.bmm(struct_task_norm, seq_context_norm.transpose(1, 2))
        if isinstance(seq_mask, torch.Tensor) and seq_mask.ndim == 2:
            seq_logits = seq_logits.masked_fill(seq_mask.unsqueeze(1), -10000.0)
        seq_refine_attention = F.softmax(seq_logits, dim=-1)
        seq_task_refined = torch.bmm(seq_refine_attention, seq_context.float()).to(dtype=seq_context.dtype)
        seq_task_summary = 0.5 * seq_task_summary + 0.5 * seq_task_refined

        alignment = F.cosine_similarity(seq_task_summary.float(), struct_task_summary.float(), dim=-1).mean(dim=-1, keepdim=True)
        seq_focus = seq_task_weights.max(dim=-1).values.mean(dim=-1, keepdim=True)
        struct_focus = struct_task_attention.max(dim=-1).values.mean(dim=-1, keepdim=True)
        norm_gap = (
            seq_task_summary.float().norm(dim=-1).mean(dim=-1, keepdim=True)
            - struct_task_summary.float().norm(dim=-1).mean(dim=-1, keepdim=True)
        ).abs()
        task_stats = torch.cat([seq_focus, struct_focus, alignment, norm_gap], dim=-1)
        return {
            "seq_task_summary": seq_task_summary,
            "struct_task_summary": struct_task_summary,
            "seq_task_weights": seq_task_weights.to(dtype=seq_context.dtype),
            "struct_task_attention": struct_task_attention.to(dtype=seq_context.dtype),
            "task_stats": task_stats.to(dtype=seq_context.dtype),
        }

    def _known_label_bce(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        loss_weight: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            return None
        if logits.shape != labels.shape:
            return None
        targets = labels.to(device=logits.device, dtype=logits.dtype)
        known = targets >= 0.0
        if not bool(known.any()):
            return None
        targets = targets.clamp(0.0, 1.0)
        raw = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        weights = known.to(dtype=raw.dtype)
        if isinstance(loss_weight, torch.Tensor) and loss_weight.shape == labels.shape:
            weights = weights * loss_weight.to(device=logits.device, dtype=raw.dtype).clamp(min=0.0)
        denom = weights.sum().clamp(min=1.0)
        return (raw * weights).sum() / denom

    def _fuse_separator_logits(
        self,
        base_logits: torch.Tensor,
        separator_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        loss_weight: Optional[torch.Tensor],
        auxiliary_losses: Dict[str, torch.Tensor],
        auxiliary_logs: Dict[str, torch.Tensor],
        module_contributions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        base_logits = torch.nan_to_num(base_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        separator_logits = torch.nan_to_num(separator_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        mode = getattr(self, "separator_logit_fusion", "separator")
        if mode == "base":
            fused = base_logits
        elif mode == "residual":
            fused = base_logits + float(getattr(self, "separator_logit_private_weight", 0.65)) * separator_logits
        elif mode in {"weighted_sum", "weighted", "static_gate"}:
            base_w = max(0.0, float(getattr(self, "separator_logit_base_weight", 0.35)))
            sep_w = max(0.0, float(getattr(self, "separator_logit_private_weight", 0.65)))
            norm = max(1e-6, base_w + sep_w)
            fused = (base_w * base_logits + sep_w * separator_logits) / norm
        elif mode in {"learned_gate", "gate", "gated"} and isinstance(getattr(self, "separator_logit_gate", None), nn.Parameter):
            gate = torch.sigmoid(self.separator_logit_gate).to(device=base_logits.device, dtype=base_logits.dtype).view(1, -1)
            fused = (1.0 - gate) * base_logits + gate * separator_logits
            module_contributions["separator_logit_gate_per_task"] = gate.detach().squeeze(0)
            auxiliary_logs["separator_logit_gate_mean"] = gate.detach().mean()
        else:
            fused = separator_logits

        sep_aux_weight = float(getattr(self, "separator_aux_logit_loss_weight", 0.0))
        if sep_aux_weight > 0.0:
            sep_aux = self._known_label_bce(separator_logits, labels, loss_weight)
            if sep_aux is not None and torch.isfinite(sep_aux):
                auxiliary_losses["separator_logit_aux"] = sep_aux * sep_aux_weight
                auxiliary_logs["separator_logit_aux"] = sep_aux.detach()

        base_aux_weight = float(getattr(self, "base_aux_logit_loss_weight", 0.0))
        if base_aux_weight > 0.0:
            base_aux = self._known_label_bce(base_logits, labels, loss_weight)
            if base_aux is not None and torch.isfinite(base_aux):
                auxiliary_losses["base_logit_aux"] = base_aux * base_aux_weight
                auxiliary_logs["base_logit_aux"] = base_aux.detach()

        module_contributions["base_logit_mean_per_task"] = base_logits.detach().mean(dim=0)
        module_contributions["separator_logit_mean_per_task"] = separator_logits.detach().mean(dim=0)
        module_contributions["fused_logit_mean_per_task"] = fused.detach().mean(dim=0)
        return fused.to(dtype=separator_logits.dtype)

    def _classifier_is_representation_passthrough(self) -> bool:
        return isinstance(getattr(self, "classifier", None), nn.Identity)

    def forward(self, batch, return_importance=False):
        sequence_block = batch.get('sequence', {})
        if isinstance(sequence_block, dict):
            seq_window = sequence_block.get('window_features')
        else:
            seq_window = sequence_block
            sequence_block = {}

        if seq_window is None:
            seq_window = batch.get('sequence_window')
        if seq_window is None:
            seq_window = batch.get('sequence_features')

        sequence_strings = None
        if isinstance(sequence_block, dict):
            sequence_strings = sequence_block.get('window_sequence') or sequence_block.get('window_sequence_str')
        if sequence_strings is None:
            sequence_strings = batch.get('sequence_window_str') or batch.get('window_sequence')

        if self.use_esm_lora and sequence_strings is not None:
            target_len = int(seq_window.shape[1]) if isinstance(seq_window, torch.Tensor) and seq_window.ndim >= 2 else None
            if isinstance(sequence_strings, (list, tuple)):
                device = seq_window.device if isinstance(seq_window, torch.Tensor) else next(self.parameters()).device
                seq_list, non_empty_idx = compute_non_empty_sequence_indices(sequence_strings)
                if non_empty_idx:
                    encoded = self._encode_sequences_with_lora(
                        [seq_list[idx] for idx in non_empty_idx],
                        device=device,
                        target_len=target_len,
                    )
                    if not isinstance(seq_window, torch.Tensor):
                        seq_window = encoded.new_zeros((len(seq_list), encoded.size(1), encoded.size(2)))
                    seq_window = merge_non_empty_lora_features(seq_window, non_empty_idx, encoded)
            elif isinstance(sequence_strings, str):
                device = seq_window.device if isinstance(seq_window, torch.Tensor) else next(self.parameters()).device
                if len(sequence_strings.strip()) > 0:
                    seq_window = self._encode_sequences_with_lora(
                        [sequence_strings],
                        device=device,
                        target_len=target_len,
                    )[:1]

        # [Fix] Extract micro-environment features early
        micro_env = batch.get('micro_env_features')
        if not self.use_micro_environment:
            micro_env = (
                seq_window.new_zeros((seq_window.size(0), 0))
                if isinstance(seq_window, torch.Tensor)
                else None
            )
        elif micro_env is None:
            if seq_window is not None and self.micro_env_feature_dim > 0:
                micro_env = torch.zeros(seq_window.size(0), self.micro_env_feature_dim, device=seq_window.device)
            else:
                micro_env = None
        elif isinstance(micro_env, torch.Tensor) and seq_window is not None and micro_env.device != seq_window.device:
            micro_env = micro_env.to(seq_window.device)

        if seq_window is None:
            raise ValueError("Batch缺少 sequence['window_features'] 特征")

        # Offline Tensor LMDB stores dense features as FP16. Some projection
        # paths are outside AMP, so normalize floating inputs at the model
        # boundary while retaining FP16 disk/PCIe transfer.
        model_dtype = next(self.parameters()).dtype
        if isinstance(seq_window, torch.Tensor) and seq_window.is_floating_point():
            seq_window = seq_window.to(dtype=model_dtype)

        labels = batch.get('label', None)

        center_epoch = batch.get('center_curriculum_epoch')
        center_alpha_override = batch.get('center_alpha')
        active_center_alpha = self._resolve_center_alpha(center_epoch, center_alpha_override)

        struct_tensor = batch.get('structure')
        if struct_tensor is None:
            struct_tensor = batch.get('structure_features')
        structure_available = batch.get('structure_available')
        graph_data = batch.get('structure_graph')
        graph_available = batch.get('structure_graph_available')
        global_context_available = None
        if isinstance(sequence_block, dict):
            global_context_available = sequence_block.get('global_context_available')
        if global_context_available is None:
            global_context_available = batch.get('global_context_available')
        structure_global_features = batch.get('structure_global_features')
        structure_global_available = batch.get('structure_global_available')
        
        if not self.use_structure:
            struct_tensor = None
            structure_global_features = None
            if isinstance(structure_available, torch.Tensor):
                structure_available = torch.zeros_like(structure_available)
            if isinstance(structure_global_available, torch.Tensor):
                structure_global_available = torch.zeros_like(structure_global_available)
        if self.use_structure_graph and not self.use_dense_structure_tower_with_graph:
            struct_tensor = None

        # [Optimization] Extract distance matrix for structure branch
        distance_matrices = batch.get('distance_matrix')
        if distance_matrices is None:
             sequence_block = batch.get('sequence', {})
             if isinstance(sequence_block, dict):
                 distance_matrices = sequence_block.get('distance_matrix')
        
        # Ensure distance matrix is on correct device if present
        if distance_matrices is not None and isinstance(distance_matrices, torch.Tensor):
            distance_matrices = distance_matrices.to(seq_window.device)

        # [Fix] Calculate masks BEFORE injecting explicit features (which make padding non-zero)
        seq_mask = self._extract_masks(seq_window)
        struct_mask = None
        if struct_tensor is not None:
            struct_mask = self._extract_masks(struct_tensor)

        raw_seq_window = seq_window

        # [NEW] Inject Explicit Features (Residue + Position)
        if self.use_explicit_features:
            # 1. Residue Embeddings
            residue_types = sequence_block.get('residue_types')
            if residue_types is None:
                residue_types = batch.get('residue_types')
            if residue_types is not None:
                if residue_types.device != seq_window.device:
                    residue_types = residue_types.to(seq_window.device)
                if residue_types.dtype not in (torch.int32, torch.int64):
                    residue_types = torch.nan_to_num(residue_types.float(), nan=0.0, posinf=0.0, neginf=0.0).long()
                # Ensure valid range [0, 24]
                residue_types = residue_types.clamp(0, 24)

                if not torch.isfinite(self.residue_embedding.weight).all():
                    if not self._residue_embedding_repaired:
                        logger.warning("Non-finite values detected in residue_embedding weights. Repairing with nan_to_num.")
                        self._residue_embedding_repaired = True
                    with torch.no_grad():
                        self.residue_embedding.weight.data = torch.nan_to_num(
                            self.residue_embedding.weight.data,
                            nan=0.0,
                            posinf=1.0,
                            neginf=-1.0,
                        )
                res_emb = self.residue_embedding(residue_types)
                # Check for NaNs in residue embedding
                if torch.isnan(res_emb).any():
                    logger.warning("NaN detected in res_emb. Applying nan_to_num.")
                    res_emb = torch.nan_to_num(res_emb)
            else:
                # Fallback: zeros
                res_emb = torch.zeros(
                    seq_window.size(0), seq_window.size(1), self.residue_embedding_dim,
                    device=seq_window.device, dtype=seq_window.dtype
                )
            
            # 2. Position Embeddings
            # Use simple 0..L-1 position encoding
            pos_ids = torch.arange(seq_window.size(1), device=seq_window.device).expand(seq_window.size(0), -1)
            # Clamp to embedding size limit
            pos_ids = pos_ids.clamp(0, 511)

            if not torch.isfinite(self.position_embedding.weight).all():
                if not self._position_embedding_repaired:
                    logger.warning("Non-finite values detected in position_embedding weights. Repairing with nan_to_num.")
                    self._position_embedding_repaired = True
                with torch.no_grad():
                    self.position_embedding.weight.data = torch.nan_to_num(
                        self.position_embedding.weight.data,
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0,
                    )
            pos_emb = self.position_embedding(pos_ids)
            # Check for NaNs in position embedding
            if torch.isnan(pos_emb).any():
                logger.warning("NaN detected in pos_emb. Applying nan_to_num.")
                pos_emb = torch.nan_to_num(pos_emb)
            
            # 3. Concatenate
            # Check seq_window before concat
            if torch.isnan(seq_window).any():
                 logger.warning("NaN detected in seq_window BEFORE concat. Applying nan_to_num.")
                 seq_window = torch.nan_to_num(seq_window)
                 
            seq_window = torch.cat([seq_window, res_emb, pos_emb], dim=-1)

            # 4. Inject Position Embedding into Structure (Moved here to fix dimension check)
            if struct_tensor is not None:
                # Check struct_tensor before concat
                if torch.isnan(struct_tensor).any():
                     logger.warning("NaN detected in struct_tensor BEFORE concat. Applying nan_to_num.")
                     struct_tensor = torch.nan_to_num(struct_tensor)
                
                # [Modified] Add residue embedding (64) + position embedding (32) + micro-env (3) to structure branch
                # micro_env: [B, 3] -> expand to [B, L, 3]
                if micro_env is not None:
                    micro_env_expanded = micro_env.unsqueeze(1).expand(-1, struct_tensor.size(1), -1)
                    struct_tensor = torch.cat([struct_tensor, res_emb, pos_emb, micro_env_expanded], dim=-1)
                else:
                    # Fallback if micro_env is somehow missing
                    dummy_micro_env = torch.zeros(
                        struct_tensor.size(0),
                        struct_tensor.size(1),
                        self.micro_env_feature_dim,
                        device=struct_tensor.device,
                        dtype=struct_tensor.dtype,
                    )
                    struct_tensor = torch.cat([struct_tensor, res_emb, pos_emb, dummy_micro_env], dim=-1)
                
                # [FIX] Zero out padding regions again because embeddings made them non-zero
                # struct_mask is True for padding (zeros). We want to keep valid data (False).
                if struct_mask is not None:
                    # struct_mask: [B, L] -> True for padding
                    # We want 1.0 for valid, 0.0 for padding
                    valid_mask = (~struct_mask).unsqueeze(-1).to(dtype=struct_tensor.dtype)
                    struct_tensor = struct_tensor * valid_mask

        # Dynamic dimension check
        struct_tensor_check = struct_tensor
        struct_dim = struct_tensor_check.shape[-1] if struct_tensor_check is not None else None
        
        local_features = sequence_block.get('local_features')
        if local_features is None:
            local_features = batch.get('local_features')
        if local_features is None and isinstance(raw_seq_window, torch.Tensor):
            local_window = int(self.config.get("local_window_size", 21))
            if local_window <= 0 or local_window >= raw_seq_window.size(1):
                local_features = raw_seq_window
            else:
                center = raw_seq_window.size(1) // 2
                half = local_window // 2
                start = max(0, center - half)
                end = min(raw_seq_window.size(1), start + local_window)
                local_features = raw_seq_window[:, start:end, :]
        local_dim = local_features.shape[-1] if local_features is not None else None
        
        global_features = sequence_block.get('global_features')
        if global_features is None:
            global_features = batch.get('global_features')
        if global_features is None and isinstance(raw_seq_window, torch.Tensor):
            global_features = raw_seq_window.mean(dim=1)
        global_dim = global_features.shape[-1] if global_features is not None else None
        
        for name in ('local_features', 'global_features', 'struct_tensor', 'micro_env', 'structure_global_features'):
            value = locals().get(name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                value = value.to(dtype=model_dtype)
                if name == 'local_features':
                    local_features = value
                elif name == 'global_features':
                    global_features = value
                elif name == 'struct_tensor':
                    struct_tensor = value
                elif name == 'micro_env':
                    micro_env = value
                elif name == 'structure_global_features':
                    structure_global_features = value

        self._check_and_fix_projection(seq_window.shape[-1], struct_dim, local_dim, global_dim)

        seq_window, local_features, struct_tensor = self._apply_center_mask_if_needed_dual(
            seq_window,
            local_features,
            struct_tensor,
        )

        auxiliary_losses = {}
        auxiliary_logs = {}
        module_contributions: Dict[str, torch.Tensor] = {}
        last_task_context = None

        # [Enhanced Components] Apply Enhanced Attention if enabled
        if getattr(self, 'use_enhanced_attention', False):
            # Project inputs first
            seq_window = self.seq_enhanced_norm(self.seq_enhanced_proj(seq_window))
            seq_window = self.enhanced_seq_attention(seq_window, mask=seq_mask)
            
            if struct_tensor is not None and getattr(self, 'use_structure', False):
                struct_tensor = self.struct_enhanced_norm(self.struct_enhanced_proj(struct_tensor))
                struct_tensor = self.enhanced_struct_attention(struct_tensor, mask=struct_mask)

        task_description = self.task_parser.parse(
            sequence_hidden=seq_window,
            seq_mask=seq_mask,
            structure_hidden=struct_tensor,
            graph_data=graph_data if isinstance(graph_data, dict) else None,
        )
        task_context = None
        task_raw_outputs: Dict[str, torch.Tensor] = {}
        if self.task_context_generator is not None:
            task_context = self.task_context_generator.generate(
                task_description=task_description,
                sequence_hidden=seq_window,
                seq_mask=seq_mask,
                structure_hidden=struct_tensor,
            )
            seq_window = task_context.sequence_enhanced
            last_task_context = task_context.projected_task_context
            auxiliary_losses.update(task_context.auxiliary_losses)
            auxiliary_logs.update(task_context.auxiliary_logs)
            module_contributions.update(task_context.module_contributions)
            task_raw_outputs = dict(task_context.raw_outputs)

        # seq_mask is already calculated
        seq_weights = None
        if return_importance:
            seq_context, seq_global, seq_weights = self.sequence_tower(seq_window, seq_mask, center_alpha=active_center_alpha, return_weights=True)
        else:
            seq_context, seq_global = self.sequence_tower(seq_window, seq_mask, center_alpha=active_center_alpha)

        # [New] Learnable task attention on the contextual tokens from the tower.
        # Only used when LaST is off, so the two mechanisms never stack.
        if self.learnable_task_attention is not None and not self.use_last_foreground:
            la_out = self.learnable_task_attention(seq_context=seq_context, seq_mask=seq_mask)
            last_task_context = la_out["task_context"] * self.learnable_task_attention_scale
            module_contributions.update(la_out["module_diag"])

        labels_flat = None

        labels_column = None
        if isinstance(labels, torch.Tensor):
            labels_flat = labels.to(seq_global.device).float()
            if labels_flat.ndim == 1:
                labels_column = labels_flat.view(-1, 1)
            elif labels_flat.ndim == 2 and labels_flat.shape[1] == 1:
                labels_column = labels_flat

        seq_context_before_structure = seq_context
        seq_global_before_structure = seq_global
        struct_context = None
        struct_global = None
        graph_global = None
        struct_mask = None
        struct_all_masked = None
        seq_all_masked = None

        if seq_mask is not None and seq_mask.ndim == 2:
            seq_all_masked = seq_mask.all(dim=1)

        if self.structure_tower is not None and struct_tensor is not None:
            if struct_mask is None:
                struct_mask = self._extract_masks(struct_tensor)

            struct_all_masked = struct_mask.all(dim=1) if struct_mask.ndim == 2 else None
            if struct_all_masked is not None and struct_all_masked.any():
                if not self._structure_all_mask_warning_emitted:
                    logger.warning(
                        "检测到 %d 个样本的结构分支全部被掩蔽，将跳过跨模态交互，直接使用零向量回退，避免噪声/NaN",
                        int(struct_all_masked.sum().item()),
                    )
                    self._structure_all_mask_warning_emitted = True
                struct_mask = struct_mask.clone()
                struct_mask[struct_all_masked] = False

            if return_importance:
                struct_context, struct_global, struct_weights = self.structure_tower(
                    struct_tensor,
                    struct_mask,
                    center_alpha=active_center_alpha,
                    return_weights=True,
                    distance_matrices=distance_matrices,
                )
            else:
                struct_context, struct_global = self.structure_tower(
                    struct_tensor,
                    struct_mask,
                    center_alpha=active_center_alpha,
                    distance_matrices=distance_matrices,
                )
            seq_global_raw = seq_global
            struct_global_raw = struct_global

            if self.cross_modal is not None:
                use_task_aware_cross_modal = isinstance(task_raw_outputs.get("fg_mask"), torch.Tensor) or isinstance(
                    task_raw_outputs.get("token_prior"), torch.Tensor
                )
                if use_task_aware_cross_modal or (struct_all_masked is not None and struct_all_masked.all()):
                    seq_context_fused, struct_context_fused = seq_context, struct_context
                else:
                    seq_context_fused, struct_context_fused = self.cross_modal(
                        seq_context,
                        struct_context,
                        seq_mask,
                        struct_mask,
                    )
                seq_context = seq_context_fused
                struct_context = struct_context_fused
                seq_global = masked_mean(seq_context, seq_mask)
                struct_global = masked_mean(struct_context, struct_mask)

            if struct_all_masked is not None and struct_all_masked.any():
                struct_context = struct_context.clone()
                struct_global = struct_global.clone()
                struct_global_raw = struct_global_raw.clone()
                struct_context[struct_all_masked] = 0.0
                struct_global[struct_all_masked] = 0.0
                struct_global_raw[struct_all_masked] = 0.0
        else:
            seq_global_raw = seq_global
            struct_global_raw = None

        if self.structure_graph_builder is not None and graph_data is not None:
            graph_result = self.structure_graph_builder.build(
                StructureGraphContext(
                    task_description=task_description,
                    task_context=task_context,
                    graph_data=graph_data if isinstance(graph_data, dict) else None,
                    output_dtype=seq_window.dtype,
                    device=seq_window.device,
                )
            )
            graph_global = graph_result.structure_global
            module_contributions.update(graph_result.module_contributions)
            if graph_global is not None and self.graph_projection is not None:
                graph_global = self.graph_projection(graph_global)
            if (
                self.use_token_aligned_graph_fusion
                and graph_result.node_embeddings is not None
                and graph_result.node_token_index is not None
                and seq_context is not None
            ):
                seq_context, seq_global = self._fuse_aligned_graph_tokens(
                    seq_context,
                    seq_global,
                    seq_mask,
                    graph_result.node_embeddings,
                    graph_result.node_batch_index,
                    graph_result.node_token_index,
                )
                module_contributions["graph_token_scale"] = self.graph_token_scale.detach().view(1)
            if not self.use_dense_structure_tower_with_graph:
                struct_global = graph_global
                struct_global_raw = graph_global
                struct_context = None
                struct_mask = None
                struct_all_masked = None
        if struct_global is None:
            struct_global = seq_global.new_zeros(seq_global.shape)
            struct_global_raw = seq_global.new_zeros(seq_global.shape)
        if graph_global is None:
            graph_global = seq_global.new_zeros(seq_global.shape)

        # 若序列分支也出现全掩蔽，显式置零，避免噪声
        if seq_all_masked is not None and seq_all_masked.any():
            seq_context = seq_context.clone()
            seq_global = seq_global.clone()
            seq_global_raw = seq_global_raw.clone()
            seq_context[seq_all_masked] = 0.0
            seq_global[seq_all_masked] = 0.0
            seq_global_raw[seq_all_masked] = 0.0

        # Modality Dropout: 随机丢弃某一模态以增强鲁棒性 (仅在训练模式下)
        if isinstance(structure_available, torch.Tensor):
            structure_reliability_score = structure_available.unsqueeze(-1).to(dtype=seq_global.dtype, device=seq_global.device)
        elif struct_tensor is not None or graph_data is not None:
            structure_reliability_score = torch.ones((seq_global.size(0), 1), device=seq_global.device, dtype=seq_global.dtype)
        else:
            structure_reliability_score = torch.zeros((seq_global.size(0), 1), device=seq_global.device, dtype=seq_global.dtype)

        # The legacy graph-only route uses graph availability as structure reliability.
        if self.use_structure_graph and not self.use_dense_structure_tower_with_graph and isinstance(graph_available, torch.Tensor):
            structure_reliability_score = graph_available.unsqueeze(-1).to(dtype=seq_global.dtype, device=seq_global.device)

        graph_reliability_score = (
            graph_available.unsqueeze(-1).to(dtype=seq_global.dtype, device=seq_global.device)
            if isinstance(graph_available, torch.Tensor)
            else torch.zeros_like(structure_reliability_score)
        )
        reliability_scale = structure_reliability_score
        if self.use_dense_structure_tower_with_graph:
            reliability_scale = torch.maximum(structure_reliability_score, graph_reliability_score)

        if self.structure_reliability_gate is not None:
            reliability = self.structure_reliability_gate(
                struct_global=struct_global,
                struct_mask=struct_mask,
                micro_env=micro_env,
            )
            if isinstance(reliability, torch.Tensor):
                if reliability.ndim == 1:
                    reliability = reliability.unsqueeze(-1)
                gate_reliability_scale = reliability.mean(dim=1, keepdim=True).to(dtype=seq_global.dtype)
                
                if self.use_dense_structure_tower_with_graph:
                    structure_reliability_score = structure_reliability_score * gate_reliability_scale
                    reliability_scale = torch.maximum(structure_reliability_score, graph_reliability_score)
                else:
                    reliability_scale = reliability_scale * gate_reliability_scale
                module_contributions["structure_reliability_mean_per_task"] = reliability.detach().mean(dim=0)

        reliability_scale = reliability_scale.clamp(min=0.0, max=1.0)
        dense_reliability_scale = structure_reliability_score.clamp(min=0.0, max=1.0)
        struct_global = struct_global * (dense_reliability_scale if self.use_dense_structure_tower_with_graph else reliability_scale)
        struct_global_raw = struct_global_raw * (dense_reliability_scale if self.use_dense_structure_tower_with_graph else reliability_scale)
        graph_global = graph_global * graph_reliability_score
        module_contributions["structure_reliability_score_mean"] = reliability_scale.detach().mean()
        module_contributions["structure_available_rate"] = (structure_reliability_score > 0).float().detach().mean()

        structure_keep = torch.ones_like(reliability_scale)
        if self.training and self.modality_dropout > 0:
            available = (reliability_scale > 0).to(dtype=seq_global.dtype)
            structure_keep = 1.0 - (
                (torch.rand_like(available) < self.modality_dropout).to(dtype=seq_global.dtype) * available
            )
            reliability_scale = reliability_scale * structure_keep
            struct_global = struct_global * structure_keep
            struct_global_raw = struct_global_raw * structure_keep
            graph_global = graph_global * structure_keep
            if isinstance(struct_context, torch.Tensor):
                struct_context = struct_context * structure_keep.unsqueeze(1)
            seq_global = (
                seq_global * structure_keep
                + seq_global_before_structure * (1.0 - structure_keep)
            )
            seq_context = (
                seq_context * structure_keep.unsqueeze(1)
                + seq_context_before_structure * (1.0 - structure_keep).unsqueeze(1)
            )
            module_contributions["structure_modality_keep_rate"] = structure_keep.detach().mean()

        fg_mask_for_fusion = None
        if isinstance(task_raw_outputs.get("fg_mask"), torch.Tensor):
            fg_mask_for_fusion = task_raw_outputs["fg_mask"].to(device=seq_context.device, dtype=seq_context.dtype)
        elif isinstance(task_raw_outputs.get("token_prior"), torch.Tensor):
            fg_mask_for_fusion = task_raw_outputs["token_prior"].to(device=seq_context.device, dtype=seq_context.dtype)

        task_aware_outputs = None
        task_seq_summary = None
        task_struct_summary = None
        task_stats = None
        if fg_mask_for_fusion is not None:
            task_aware_outputs = self._build_task_aware_multimodal_context(
                seq_context=seq_context,
                struct_context=struct_context,
                fg_mask=fg_mask_for_fusion,
                seq_mask=seq_mask,
                struct_mask=struct_mask,
            )
            if isinstance(task_aware_outputs, dict):
                task_seq_summary = task_aware_outputs.get("seq_task_summary")
                task_struct_summary = task_aware_outputs.get("struct_task_summary")
                task_stats = task_aware_outputs.get("task_stats")
                if isinstance(task_seq_summary, torch.Tensor) and isinstance(task_struct_summary, torch.Tensor):
                    task_struct_summary = task_struct_summary * structure_keep.unsqueeze(1)
                    if self.cross_modal is not None:
                        task_seq_before_cross = task_seq_summary
                        task_seq_summary, task_struct_summary = self.cross_modal(
                            task_seq_summary,
                            task_struct_summary,
                            None,
                            None,
                        )
                        task_seq_summary = (
                            task_seq_summary * structure_keep.unsqueeze(1)
                            + task_seq_before_cross * (1.0 - structure_keep).unsqueeze(1)
                        )
                        task_struct_summary = task_struct_summary * structure_keep.unsqueeze(1)
                    task_strength = fg_mask_for_fusion.float().mean(dim=-1)
                    task_strength = task_strength / task_strength.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    seq_global = 0.7 * seq_global + 0.3 * (task_seq_summary.float() * task_strength.unsqueeze(-1)).sum(dim=1).to(dtype=seq_global.dtype)
                    struct_global = (
                        0.7 * struct_global
                        + 0.3 * (task_struct_summary.float() * task_strength.unsqueeze(-1)).sum(dim=1).to(dtype=struct_global.dtype)
                    ) * structure_keep
                    token_struct_alignment = F.cosine_similarity(
                        task_seq_summary.float(),
                        task_struct_summary.float(),
                        dim=-1,
                    ).mean()
                    module_contributions["task_aware_seq_struct_alignment"] = token_struct_alignment.detach().view(1)
                    module_contributions["task_aware_seq_focus"] = task_strength.detach().mean(dim=0)
                    if isinstance(task_aware_outputs.get("struct_task_attention"), torch.Tensor):
                        struct_task_attention = task_aware_outputs["struct_task_attention"]
                        module_contributions["task_aware_struct_focus"] = struct_task_attention.detach().max(dim=-1).values.mean(dim=0)

        if self.tri_modal_reliability_gate is not None:
            dense_available = dense_reliability_scale * structure_keep
            graph_available_scale = graph_reliability_score * structure_keep
            micro_for_gate = micro_env if micro_env is not None else seq_global.new_zeros(
                (seq_global.size(0), self.micro_env_feature_dim)
            )
            gate_input = torch.cat(
                [seq_global, struct_global, graph_global, micro_for_gate, dense_available, graph_available_scale],
                dim=-1,
            )
            gate_logits = self.tri_modal_reliability_gate(gate_input)
            modality_available = torch.cat(
                [torch.ones_like(dense_available), dense_available, graph_available_scale],
                dim=-1,
            )
            tri_gate_weights = torch.softmax(gate_logits.masked_fill(modality_available <= 0, -1e4), dim=-1)
            fused_global = (
                tri_gate_weights[:, :1] * seq_global
                + tri_gate_weights[:, 1:2] * struct_global
                + tri_gate_weights[:, 2:] * graph_global
            )
            gate_weights = tri_gate_weights[:, :1].expand_as(seq_global)
            module_contributions["tri_gate_sequence_mean"] = tri_gate_weights[:, 0].detach().mean()
            module_contributions["tri_gate_dense_structure_mean"] = tri_gate_weights[:, 1].detach().mean()
            module_contributions["tri_gate_graph_mean"] = tri_gate_weights[:, 2].detach().mean()
        elif self.use_adaptive_fusion_gate:
            fused_global, gate_weights = self.gating(
                seq_global,
                struct_global,
                reliability=reliability_scale,
                seq_task_summary=task_seq_summary,
                struct_task_summary=task_struct_summary,
                task_stats=task_stats,
            )
        else:
            gate_weights = torch.full_like(seq_global, 0.5)
            fused_global = 0.5 * seq_global + 0.5 * struct_global
        module_contributions["gate_sequence_mean"] = gate_weights.detach().mean()
        module_contributions["gate_structure_mean"] = (1.0 - gate_weights.detach()).mean()
        module_contributions["fusion_gate_enabled"] = torch.tensor(
            [1.0 if self.use_adaptive_fusion_gate else 0.0],
            device=fused_global.device,
            dtype=fused_global.dtype,
        )
        
        # Optional gate polarization regularizer.
        # Force the gate away from 0.5 to make a definitive sequence/structure routing choice
        if self.use_adaptive_fusion_gate and self.fusion_gate_bipolar_lambda > 0:
            gate_bipolar_loss = (gate_weights * (1.0 - gate_weights)).mean()
            auxiliary_losses["fusion_gate_bipolar"] = (
                gate_bipolar_loss * self.fusion_gate_bipolar_lambda
            )
            auxiliary_logs["fusion_gate_bipolar"] = gate_bipolar_loss.detach()

        if local_features is not None:
            # [Modified] Use average pooling over the local window instead of just center pixel
            # This captures the local motif context better than a single residue
            center = int(local_features.shape[1]) // 2
            local_scales = [local_features[:, center, :]]
            for radius in (3, 7, 15):
                lo = max(0, center - radius)
                hi = min(int(local_features.shape[1]), center + radius + 1)
                local_scales.append(local_features[:, lo:hi, :].mean(dim=1))
            local_scale_weights = torch.softmax(self.local_center_scale_logits, dim=0).to(local_features.dtype)
            local_summary = sum(weight * value for weight, value in zip(local_scale_weights, local_scales))
            local_repr = self.local_projection(local_summary)
        else:
            local_repr = seq_global.new_zeros(seq_global.shape)

        if global_features is not None:
            global_repr = self.global_projection(global_features)
        else:
            global_repr = seq_global.new_zeros(seq_global.shape)
        if isinstance(global_context_available, torch.Tensor):
            global_context_scale = global_context_available.unsqueeze(-1).to(dtype=seq_global.dtype, device=seq_global.device)
        else:
            global_context_scale = torch.ones((seq_global.size(0), 1), device=seq_global.device, dtype=seq_global.dtype)
        global_context_scale = global_context_scale + (1.0 - global_context_scale) * self.global_context_fallback_scale
        global_repr = global_repr * global_context_scale
        if self.use_global_context_gate:
            global_gate = torch.sigmoid(
                self.global_context_gate(torch.cat([seq_global, global_repr, global_context_scale], dim=-1))
            )
            global_repr = global_gate * global_repr + (1.0 - global_gate) * seq_global
            module_contributions["global_context_gate_mean"] = global_gate.detach().mean()
        module_contributions["global_context_scale_mean"] = global_context_scale.detach().mean()

        if self.use_structure_global_context and isinstance(structure_global_features, torch.Tensor):
            structure_global_features = structure_global_features.to(device=seq_global.device, dtype=seq_global.dtype)
            if structure_global_features.shape[-1] != self.struct_global_input_dim:
                if structure_global_features.shape[-1] > self.struct_global_input_dim:
                    structure_global_features = structure_global_features[..., : self.struct_global_input_dim]
                else:
                    pad = self.struct_global_input_dim - structure_global_features.shape[-1]
                    structure_global_features = F.pad(structure_global_features, (0, pad))
            struct_global_repr = self.structure_global_projection(structure_global_features)
        else:
            struct_global_repr = seq_global.new_zeros(seq_global.shape)
        if isinstance(structure_global_available, torch.Tensor):
            struct_global_context_scale = structure_global_available.unsqueeze(-1).to(dtype=seq_global.dtype, device=seq_global.device)
        else:
            struct_global_context_scale = torch.ones((seq_global.size(0), 1), device=seq_global.device, dtype=seq_global.dtype)
        struct_global_repr = struct_global_repr * struct_global_context_scale * reliability_scale
        module_contributions["structure_global_context_scale_mean"] = struct_global_context_scale.detach().mean()

        seq_only_anchor = self.seq_only_projection(torch.cat([seq_global, local_repr, global_repr, micro_env], dim=-1))
        conditioned_fused_global = reliability_scale * fused_global + (1.0 - reliability_scale) * seq_only_anchor
        module_contributions["seq_only_anchor_norm"] = seq_only_anchor.detach().norm(dim=-1).mean()

        ptm_multimodal_shared_context = None
        if self.ptm_multimodal_attention is not None:
            task_query = task_seq_summary if isinstance(task_seq_summary, torch.Tensor) else None
            micro_token = self.ptm_multimodal_micro_projection(micro_env).unsqueeze(1)
            local_tokens = torch.stack(
                [
                    local_repr,
                    global_repr,
                    struct_global_repr,
                    conditioned_fused_global,
                ],
                dim=1,
            )
            if isinstance(local_features, torch.Tensor) and local_features.ndim == 3:
                radius = max(0, int(getattr(self, "ptm_multimodal_local_window_radius", 4)))
                center = int(local_features.shape[1]) // 2
                lo = max(0, center - radius)
                hi = min(int(local_features.shape[1]), center + radius + 1)
                local_window = local_features[:, lo:hi, :]
                if local_window.shape[-1] != self.local_input_dim:
                    if local_window.shape[-1] > self.local_input_dim:
                        local_window = local_window[..., : self.local_input_dim]
                    else:
                        local_window = F.pad(local_window, (0, self.local_input_dim - local_window.shape[-1]))
                local_window_tokens = self.ptm_multimodal_local_window_projection(local_window)
                local_tokens = torch.cat([local_window_tokens, local_tokens], dim=1)
            local_tokens = torch.cat([local_tokens, micro_token], dim=1)
            seq_guidance = fg_mask_for_fusion if isinstance(fg_mask_for_fusion, torch.Tensor) else None
            ptm_mm_context, ptm_mm_metrics, ptm_mm_losses = self.ptm_multimodal_attention(
                task_query=task_query,
                seq_context=seq_context,
                struct_context=struct_context,
                local_tokens=local_tokens,
                seq_mask=seq_mask,
                struct_mask=struct_mask,
                reliability=reliability_scale,
                seq_guidance=seq_guidance,
            )
            if isinstance(ptm_mm_metrics, dict):
                module_contributions.update(ptm_mm_metrics)
            if isinstance(ptm_mm_losses, dict):
                auxiliary_losses.update(ptm_mm_losses)
                for loss_name, loss_value in ptm_mm_losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        auxiliary_logs[loss_name] = loss_value.detach()
            if isinstance(ptm_mm_context, torch.Tensor):
                target_mask = self.ptm_multimodal_attention.target_task_mask.to(
                    device=ptm_mm_context.device,
                    dtype=ptm_mm_context.dtype,
                )
                scale = self.ptm_multimodal_context_scale + target_mask * (
                    self.ptm_multimodal_target_context_scale - self.ptm_multimodal_context_scale
                )
                task_delta = self.ptm_multimodal_to_task_context(ptm_mm_context) * scale
                if isinstance(last_task_context, torch.Tensor) and last_task_context.shape == task_delta.shape:
                    last_task_context = last_task_context + task_delta
                else:
                    last_task_context = task_delta
                weighted_context = (ptm_mm_context * scale).sum(dim=1) / scale.sum(dim=1).clamp(min=1e-6)
                ptm_multimodal_shared_context = self.ptm_multimodal_to_shared(weighted_context)
                module_contributions["ptm_mm_task_context_delta_norm"] = task_delta.detach().norm(dim=-1).mean(dim=0)
                module_contributions["ptm_mm_shared_context_norm"] = ptm_multimodal_shared_context.detach().norm(dim=-1).mean()

        # [Enhanced Components] Multi-Scale Fusion
        ms_fused = None
        if getattr(self, 'use_multiscale_fusion', False) and struct_context is not None:
             ms_fused = self.multiscale_fusion_net(
                 seq_context,
                 struct_context,
                 seq_mask=seq_mask,
                 struct_mask=struct_mask
             )

        if self.use_legacy_fusion:
            struct_confidence = reliability_scale if not self.use_adaptive_fusion_gate else (1.0 - gate_weights) * reliability_scale
            interaction = self._safe_bilinear_interaction(seq_global, struct_global) * struct_confidence
            abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
            
            fusion_inputs = torch.cat(
                [
                    conditioned_fused_global,  # Reliability-conditioned fusion
                    seq_global,    # Sequence features
                    struct_global, # Structure features
                    abs_diff,      # Difference
                    interaction,   # Interaction
                    local_repr,    # Local sequence context
                    global_repr,   # Global protein context
                    struct_global_repr, # Global structure context
                ],
                dim=-1,
            )
        elif self.fusion_feature_set == "compact":
            fusion_inputs = torch.cat(
                [
                    conditioned_fused_global,
                    local_repr,
                    global_repr,
                    struct_global_repr,
                    micro_env,
                ],
                dim=-1,
            )
        elif getattr(self, 'use_dynamic_feature_selection', False):
            # [Enhanced Components] Dynamic Feature Selection
            features_to_select = {
                 'seq': seq_global,
                 'struct': struct_global,
                 'local': local_repr,
                 'global': global_repr,
                 'struct_global': struct_global_repr
            }
            if ms_fused is not None:
                 features_to_select['multiscale'] = ms_fused
            else:
                 features_to_select['gated'] = conditioned_fused_global
            
            fused_selected, selection_weights = self.feature_selector(features_to_select)
            
            # Recalculate interaction terms
            struct_confidence = reliability_scale if not self.use_adaptive_fusion_gate else (1.0 - gate_weights) * reliability_scale
            interaction = self._safe_bilinear_interaction(seq_global, struct_global) * struct_confidence
            abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
            
            fusion_inputs = torch.cat(
                [
                    fused_selected,
                    abs_diff,
                    interaction,
                    micro_env
                ],
                dim=-1
            )
        else:
            # Enhanced: Added explicit interaction and difference to modern fusion path
            # [Fix] Apply gating to interaction and difference to suppress noise from missing/unreliable structure
            # gate_weights: [B, H], higher means more focus on sequence (less on structure)
            struct_confidence = reliability_scale if not self.use_adaptive_fusion_gate else (1.0 - gate_weights) * reliability_scale
            interaction = self._safe_bilinear_interaction(seq_global, struct_global) * struct_confidence
            abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
            
            fusion_list = [
                    conditioned_fused_global,  # Reliability-conditioned fusion
                    seq_global,    # Sequence features (enhanced by structure attention)
                    struct_global, # Structure features
                    abs_diff,      # Explicit difference (weighted)
                    interaction,   # Explicit bilinear interaction (weighted)
                    local_repr,    # Local sequence context
                    global_repr,   # Global protein sequence context
                    struct_global_repr, # Global protein structure context
                    micro_env,     # [NEW] Micro-environment geometric features
            ]
            
            if ms_fused is not None:
                fusion_list.append(ms_fused)

            fusion_inputs = torch.cat(fusion_list, dim=-1)

        # [NEW] 使用残差结构的融合前向传播
        x = self.fusion_proj(fusion_inputs)
        residual = x
        x = self.fusion_residual_block(x)
        x = self.fusion_residual_mixer([residual, x])
        fused_representation = self.fusion_bottleneck(x)
        if isinstance(ptm_multimodal_shared_context, torch.Tensor):
            fused_representation = fused_representation + ptm_multimodal_shared_context.to(dtype=fused_representation.dtype)
        if isinstance(last_task_context, torch.Tensor) and not getattr(self, "use_ptm_feature_separator", False):
            fused_representation = fused_representation + self.last_context_scale * last_task_context.mean(dim=1)

        specialist_feats = None
        if getattr(self, "use_ptm_specialists", False) and self.specialist_extractors is not None:
            specialist_feats = self.specialist_extractors(seq_window)

        separator_private_features = None
        base_logits = None
        separator_logits = None
        if getattr(self, "use_ptm_feature_separator", False) and self.ptm_separator is not None:
            separator_labels = labels_flat
            loss_weight = batch.get("loss_weight")
            if (
                getattr(self, "separator_metric_use_loss_weight_mask", False)
                and isinstance(labels_flat, torch.Tensor)
            ):
                if isinstance(loss_weight, torch.Tensor):
                    metric_loss_weight = loss_weight.to(device=labels_flat.device, dtype=labels_flat.dtype)
                    if metric_loss_weight.shape == labels_flat.shape:
                        separator_labels = labels_flat.clone()
                        clean_mask = metric_loss_weight >= float(getattr(self, "separator_metric_min_sample_weight", 0.75))
                        separator_labels = torch.where(
                            clean_mask,
                            separator_labels,
                            torch.full_like(separator_labels, -1.0),
                        )
            separator_out = self.ptm_separator(
                fused_representation,
                task_context=last_task_context,
                specialist_features=specialist_feats,
                labels=separator_labels,
            )
            base_logits = self.classifier(fused_representation)
            separator_logits = separator_out["logits"]
            separator_private_features = separator_out.get("private_features")
            auxiliary_losses.update(separator_out.get("auxiliary_losses", {}))
            auxiliary_logs.update(separator_out.get("auxiliary_logs", {}))
            module_contributions.update(separator_out.get("module_contributions", {}))
            if self._classifier_is_representation_passthrough():
                # MultiTaskIndependentHead replaces the base classifier with Identity
                # and consumes shared_post_bottleneck itself. In that wrapper mode,
                # base_logits is the [B, H] representation, not [B, num_tasks] logits.
                sep_aux_weight = float(getattr(self, "separator_aux_logit_loss_weight", 0.0))
                if sep_aux_weight > 0.0:
                    sep_aux = self._known_label_bce(
                        separator_logits,
                        labels_flat,
                        loss_weight if isinstance(loss_weight, torch.Tensor) else None,
                    )
                    if sep_aux is not None and torch.isfinite(sep_aux):
                        auxiliary_losses["separator_logit_aux"] = sep_aux * sep_aux_weight
                        auxiliary_logs["separator_logit_aux"] = sep_aux.detach()
                module_contributions["separator_logit_mean_per_task"] = separator_logits.detach().mean(dim=0)
                base_logits = None
                logits = fused_representation
            else:
                logits = self._fuse_separator_logits(
                    base_logits=base_logits,
                    separator_logits=separator_logits,
                    labels=labels_flat,
                    loss_weight=loss_weight if isinstance(loss_weight, torch.Tensor) else None,
                    auxiliary_losses=auxiliary_losses,
                    auxiliary_logs=auxiliary_logs,
                    module_contributions=module_contributions,
                )
        elif specialist_feats is not None:
            concat_specialist = torch.cat([
                specialist_feats["crotonylation"],
                specialist_feats["methylation"],
                specialist_feats["sumoylation"]
            ], dim=-1)
            final_features = torch.cat([fused_representation, concat_specialist], dim=-1)
            logits = self.classifier(final_features).squeeze(-1)
        else:
            logits = self.classifier(fused_representation).squeeze(-1)

        if self.alignment_weight > 0 and struct_tensor is not None:
            alignment_raw = F.mse_loss(seq_global, struct_global, reduction='none').mean(dim=-1)
            # Use average of gate_weights (which is feature-wise weight for sequence branch)
            # 1 - gate_weights is the weight for the structure branch.
            # If structure is deemed unreliable (low weight), we reduce the alignment constraint.
            struct_gate_weight = (1.0 - gate_weights).mean(dim=-1)
            alignment_loss = (alignment_raw * struct_gate_weight).mean()
            
            auxiliary_losses['alignment'] = alignment_loss * self.alignment_weight
            auxiliary_logs['alignment'] = alignment_loss.detach()

        if self.similarity_weight > 0 and struct_tensor is not None:
            # [Fix] Avoid NaN in cosine similarity when structure is fully masked (zero vector)
            if struct_all_masked is not None and struct_all_masked.any():
                valid_mask = ~struct_all_masked
                if valid_mask.sum() > 0:
                    cosine_sim = F.cosine_similarity(seq_global[valid_mask], struct_global[valid_mask], dim=-1)
                    similarity_loss = (1.0 - cosine_sim).mean()
                else:
                    similarity_loss = torch.tensor(0.0, device=seq_global.device)
            else:
                cosine_sim = F.cosine_similarity(seq_global, struct_global, dim=-1)
                similarity_loss = (1.0 - cosine_sim).mean()
                
            auxiliary_losses['structure_similarity'] = similarity_loss * self.similarity_weight
            auxiliary_logs['structure_similarity'] = similarity_loss.detach()

        if self.contrastive_weight > 0 and self.projection_head is not None and labels_flat is not None:
            target_idx = None
            if labels_flat.ndim == 2:
                if hasattr(self, 'ptm_types') and self.contrastive_target_ptm in self.ptm_types:
                    target_idx = self.ptm_types.index(self.contrastive_target_ptm)
                elif labels_flat.shape[1] > 0:
                    target_idx = min(labels_flat.shape[1] - 1, 0)
                labels_target = labels_flat[:, target_idx]
            else:
                labels_target = labels_flat

            proj_features = self.projection_head(fused_representation)
            total_con_loss = self.contrastive_criterion(proj_features, labels_target)

            if total_con_loss is not None and torch.isfinite(total_con_loss):
                auxiliary_losses['contrastive'] = total_con_loss * self.contrastive_weight
                auxiliary_logs['contrastive'] = total_con_loss.detach()
                with torch.no_grad():
                    pos_pairs = torch.eq(labels_target.view(-1, 1), labels_target.view(1, -1))
                    pos_pairs.fill_diagonal_(False)
                    auxiliary_logs['contrastive_pos_pairs'] = pos_pairs.float().sum()
                    auxiliary_logs['contrastive_target_idx'] = torch.tensor(
                        -1 if target_idx is None else int(target_idx),
                        device=proj_features.device,
                        dtype=proj_features.dtype,
                    )

        aux_seq_logits = None
        aux_struct_logits = None

        if self.seq_classifier is not None:
            aux_seq_logits = self.seq_classifier(seq_global_raw)
            if labels_column is not None and self.aux_cls_weight > 0:
                aux_seq_loss = F.binary_cross_entropy_with_logits(
                    aux_seq_logits,
                    labels_column.to(aux_seq_logits.device),
                )
                auxiliary_losses['aux_seq'] = aux_seq_loss * self.aux_cls_weight
                auxiliary_logs['aux_seq'] = auxiliary_losses['aux_seq'].detach()

        if self.struct_classifier is not None and struct_global_raw is not None:
            aux_struct_logits = self.struct_classifier(struct_global_raw)
            if labels_column is not None and self.aux_cls_weight > 0:
                aux_struct_loss = F.binary_cross_entropy_with_logits(
                    aux_struct_logits,
                    labels_column.to(aux_struct_logits.device),
                )
                auxiliary_losses['aux_struct'] = aux_struct_loss * self.aux_cls_weight
                auxiliary_logs['aux_struct'] = auxiliary_losses['aux_struct'].detach()

        if self.branch_warmup_active and self.branch_warmup_aux_multiplier > 1.0:
            for aux_key in ('aux_seq', 'aux_struct'):
                if aux_key in auxiliary_losses:
                    auxiliary_losses[aux_key] = auxiliary_losses[aux_key] * self.branch_warmup_aux_multiplier
                    auxiliary_logs[aux_key] = auxiliary_losses[aux_key].detach()

        if self.branch_warmup_active and self.branch_warmup_use_branch_logits:
            branch_logits = []
            if aux_seq_logits is not None:
                branch_logits.append(aux_seq_logits.squeeze(-1))
            if aux_struct_logits is not None:
                branch_logits.append(aux_struct_logits.squeeze(-1))
            if branch_logits:
                logits = torch.stack(branch_logits, dim=0).mean(dim=0)

        if self.moe_load_balance_lambda > 0:
            moe_balance_loss = self._collect_moe_load_balance_loss()
            if isinstance(moe_balance_loss, torch.Tensor):
                auxiliary_losses['moe_load_balance'] = moe_balance_loss * self.moe_load_balance_lambda
                auxiliary_logs['moe_load_balance'] = moe_balance_loss.detach()
                module_contributions['moe_load_balance'] = moe_balance_loss.detach()

        outputs = {
            'logits': logits,
            'fusion_weights': gate_weights,
            'sequence_global': seq_global,
            'structure_global': struct_global,
            'shared_encoder_layers': {
                'shared_pre_bottleneck': x,
                'shared_post_bottleneck': fused_representation,
            },
            'auxiliary_losses': auxiliary_losses,
            'auxiliary_logs': auxiliary_logs,
            'aux_seq_logits': aux_seq_logits,
            'aux_struct_logits': aux_struct_logits,
            'struct_masked_count': struct_all_masked.sum().detach() if struct_all_masked is not None else torch.tensor(0.0).to(logits.device),
            'last_task_context': last_task_context,
            'learnable_task_context': last_task_context if (self.learnable_task_attention is not None and not self.use_last_foreground) else None,
            'separator_private_features': separator_private_features,
            'base_logits': base_logits,
            'separator_logits': separator_logits,
            'module_contributions': module_contributions,
        }
        
        if return_importance:
            return logits, {
                'window_importance': seq_weights if seq_weights is not None else torch.zeros(logits.shape[0], seq_window.shape[1]),
                'structure_importance': struct_weights if 'struct_weights' in locals() and struct_weights is not None else None,
                'gate_weights': gate_weights
            }
            
        return outputs

    def _supervised_contrastive(self, features: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        labels = labels.contiguous().view(-1)
        if labels.numel() < 2:
            return None

        device = features.device
        temperature = max(self.temperature, 1e-6)

        sim_matrix = torch.div(torch.matmul(features, features.t()), temperature)
        logits_max = torch.max(sim_matrix, dim=1, keepdim=True).values.detach()
        logits = sim_matrix - logits_max

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives = mask.sum(dim=1)
        valid_mask = positives > 0
        if valid_mask.sum() == 0:
            return None

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(positives, min=1.0)
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss

    def _set_requires_grad(self, module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def set_last_module_trainable(self, trainable: bool) -> None:
        self._set_requires_grad(self.last_foreground_block, bool(trainable))
        self._set_requires_grad(self.last_task_context_proj, bool(trainable))
        if bool(getattr(self, "use_explicit_features", False)) and bool(
            getattr(self, "freeze_explicit_embeddings_during_foreground_warmup", True)
        ):
            self._set_requires_grad(getattr(self, "residue_embedding", None), bool(trainable))
            self._set_requires_grad(getattr(self, "position_embedding", None), bool(trainable))

    def set_last_stage(self, stage: str, grl_lambda: Optional[float] = None) -> None:
        stage_name = str(stage).strip().lower()
        if stage_name in {"separation", "pretrain", "stage1"}:
            self.set_last_module_trainable(True)
            if hasattr(self, "last_foreground_block") and self.last_foreground_block is not None:
                if hasattr(self.last_foreground_block, "set_grl_lambda"):
                    self.last_foreground_block.set_grl_lambda(0.0 if grl_lambda is None else grl_lambda)
        elif stage_name in {"adversarial", "joint", "stage2"}:
            self.set_last_module_trainable(True)
            if hasattr(self, "last_foreground_block") and self.last_foreground_block is not None:
                if hasattr(self.last_foreground_block, "set_grl_lambda"):
                    target = self.last_grl_lambda if grl_lambda is None else float(grl_lambda)
                    self.last_foreground_block.set_grl_lambda(target)

    def _apply_branch_warmup_freeze(self) -> None:
        freeze_targets = [
            self.cross_modal,
            self.gating,
            self.fusion_proj,
            self.fusion_residual_block,
            self.fusion_residual_mixer,
            self.fusion_bottleneck,
            self.classifier,
            self.local_projection,
            self.global_projection,
            self.projection_head,
            self.last_foreground_block,
            self.last_task_context_proj,
            self.structure_reliability_gate,
        ]
        if self.use_legacy_fusion:
            freeze_targets.append(self.bilinear_interaction)
            
        for module in freeze_targets:
            self._set_requires_grad(module, False)

    def get_grouped_parameters(self, base_lr, weight_decay):
        param_groups = []

        def _collect_trainable_params(module: Optional[nn.Module]) -> List[nn.Parameter]:
            if module is None:
                return []
            return [p for p in module.parameters() if p.requires_grad]

        seen_params: set = set()

        def _append_group(params: List[nn.Parameter], lr: float) -> None:
            unique_params = []
            for p in params:
                pid = id(p)
                if pid in seen_params:
                    continue
                seen_params.add(pid)
                unique_params.append(p)
            if unique_params:
                param_groups.append({
                    'params': unique_params,
                    'lr': float(lr),
                    'weight_decay': weight_decay,
                })
        
        # Sequence tower
        _append_group(_collect_trainable_params(self.sequence_tower), base_lr * self.seq_lr_scale)
        if self.sequence_encoder is not None:
            _append_group(_collect_trainable_params(self.sequence_encoder), base_lr * self.seq_lr_scale)

        # Structure tower
        if self.structure_tower is not None:
            _append_group(_collect_trainable_params(self.structure_tower), base_lr * self.struct_lr_scale)
        if self.structure_graph_tower is not None:
            _append_group(_collect_trainable_params(self.structure_graph_tower), base_lr * self.struct_lr_scale)

        last_module_lr_scale = float(self.config.get("lr_scale_last_modules", 0.1))
        last_module_params = []
        for module in [self.last_foreground_block, self.last_task_context_proj, self.structure_reliability_gate]:
            last_module_params.extend(_collect_trainable_params(module))
        _append_group(last_module_params, base_lr * last_module_lr_scale)

        explicit_embed_lr_scale = float(self.config.get("lr_scale_explicit_embeddings", 0.2))
        explicit_embed_params: List[nn.Parameter] = []
        if bool(getattr(self, "use_explicit_features", False)):
            for module in [getattr(self, "residue_embedding", None), getattr(self, "position_embedding", None)]:
                explicit_embed_params.extend(_collect_trainable_params(module))
        _append_group(explicit_embed_params, base_lr * explicit_embed_lr_scale)

        bilinear_lr_scale = float(self.config.get("lr_scale_bilinear_interaction", 0.2))
        _append_group(_collect_trainable_params(self.bilinear_interaction), base_lr * bilinear_lr_scale)

        fusion_modules = [
            self.cross_modal,
            self.gating,
            self.fusion_proj,
            self.fusion_residual_block,
            self.fusion_residual_mixer,
            self.fusion_bottleneck,
            self.classifier,
            self.local_projection,
            self.global_projection,
            self.projection_head,
            self.seq_classifier,
            self.struct_classifier,
        ]
        fusion_params = []
        for module in fusion_modules:
            fusion_params.extend(_collect_trainable_params(module))

        _append_group(fusion_params, base_lr * self.fusion_lr_scale)

        if getattr(self, "use_token_aligned_graph_fusion", False):
            token_fusion_params: List[nn.Parameter] = []
            for module in [getattr(self, "graph_token_proj", None), getattr(self, "graph_token_norm", None)]:
                token_fusion_params.extend(_collect_trainable_params(module))
            token_fusion_params.extend([self.graph_token_scale, self.graph_token_global_scale])
            _append_group(token_fusion_params, base_lr)

        # Safety net: ensure no trainable parameter is omitted from optimizer groups.
        remaining_params = [p for p in self.parameters() if p.requires_grad and id(p) not in seen_params]
        _append_group(remaining_params, base_lr * self.fusion_lr_scale)

        return param_groups if param_groups else [{'params': self.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]

    def set_training_stage(self, stage: str):
        """
        Set the training stage and freeze/unfreeze parameters accordingly.
        Supported stages: 'structure_warmup', 'joint', 'full_finetune'
        """
        self.training_stage = stage
        self.branch_warmup_active = stage == 'structure_warmup'
        logger.info(f"Setting training stage to: {stage}")

        # Always unfreeze everything first to ensure clean state
        for param in self.parameters():
            param.requires_grad = True

        if self.branch_warmup_active and self.branch_warmup_freeze_fusion:
            logger.info("Stage 'structure_warmup': Freezing fusion/gating modules for branch warmup")
            self._apply_branch_warmup_freeze()

        if stage == 'structure_warmup':
            # Branch warmup lets towers specialize before joint fusion updates.
            return

        elif stage == 'joint':
            # In joint stage, check config for freezing
            freeze_struct = self.config.get('joint_freeze_structure', False)
            freeze_seq = self.config.get('joint_freeze_sequence', False)
            
            if freeze_struct:
                if self.structure_tower is not None:
                    logger.info("Stage 'joint': Freezing structure tower")
                    for param in self.structure_tower.parameters():
                        param.requires_grad = False
                if self.structure_graph_tower is not None:
                    logger.info("Stage 'joint': Freezing structure graph tower")
                    for param in self.structure_graph_tower.parameters():
                        param.requires_grad = False
            
            if freeze_seq:
                logger.info("Stage 'joint': Freezing sequence tower")
                for param in self.sequence_tower.parameters():
                    param.requires_grad = False
                    
        elif stage == 'full_finetune':
            # Unfreeze everything (already done)
            logger.info("Stage 'full_finetune': Unfreezing all parameters")
    
    def _check_and_fix_projection(self, seq_dim, struct_dim, local_dim=None, global_dim=None):
        expected = {
            "sequence": (seq_dim, self.seq_input_dim),
            "local": (local_dim, self.local_input_dim),
            "global": (global_dim, self.global_input_dim),
        }
        if self.use_structure and (not self.use_structure_graph or self.use_dense_structure_tower_with_graph):
            expected["structure"] = (struct_dim, self.struct_input_dim)
        mismatches = [
            f"{name}={actual} (expected {configured})"
            for name, (actual, configured) in expected.items()
            if actual is not None and actual != configured
        ]
        if mismatches:
            raise ValueError(
                "DualBranchFusionPredictor input dimension mismatch: "
                + ", ".join(mismatches)
                + ". Resolve dimensions before optimizer construction; modules are never rebuilt in forward."
            )



