"""Multitask output heads and model assembly."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dual_branch_predictor import DualBranchFusionPredictor
from src.utils.training_config import canonical_ptm_name, sanitize_name


class PTMAwareCrossAttention(nn.Module):
    """Task-conditioned cross-attention over sequence windows for PTM-specific cues."""

    def __init__(
        self,
        model_dim: int,
        token_dim: int,
        num_tasks: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.token_dim = int(token_dim)
        self.num_tasks = max(1, int(num_tasks))
        heads = max(1, int(num_heads))
        while self.model_dim % heads != 0 and heads > 1:
            heads -= 1
        self.token_proj = nn.Linear(self.token_dim, self.model_dim)
        self.task_embed = nn.Parameter(torch.zeros(self.num_tasks, self.model_dim))
        nn.init.normal_(self.task_embed, mean=0.0, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.out_drop = nn.Dropout(float(dropout))

    def forward(
        self,
        task_features: torch.Tensor,
        seq_tokens: Optional[torch.Tensor],
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not isinstance(seq_tokens, torch.Tensor)
            or seq_tokens.ndim != 3
            or not isinstance(task_features, torch.Tensor)
            or task_features.ndim != 3
        ):
            return None, None

        tokens = seq_tokens
        if int(tokens.shape[-1]) != self.token_dim:
            if int(tokens.shape[-1]) > self.token_dim:
                tokens = tokens[..., : self.token_dim]
            else:
                tokens = F.pad(tokens, (0, self.token_dim - int(tokens.shape[-1])))
        tokens = self.token_proj(tokens.float()).to(dtype=task_features.dtype)
        key_padding_mask = None
        if isinstance(seq_mask, torch.Tensor) and seq_mask.ndim == 2 and seq_mask.shape[:2] == tokens.shape[:2]:
            key_padding_mask = seq_mask.to(dtype=torch.bool, device=tokens.device)

        outputs: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        for idx in range(self.num_tasks):
            query = task_features[:, idx, :] + self.task_embed[idx].view(1, -1).to(task_features.dtype)
            attn_out, attn_w = self.cross_attn(
                query.unsqueeze(1),
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            outputs.append(self.out_norm(query + self.out_drop(attn_out.squeeze(1))))
            weights.append(attn_w.squeeze(1))
        return torch.stack(outputs, dim=1), torch.stack(weights, dim=1)


class MultiTaskIndependentHead(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_tasks: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_task_adapters: bool = False,
        adapter_dim: int = 128,
        adapter_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.base_model = base_model
        self.num_tasks = int(num_tasks)
        base_cfg = getattr(base_model, "config", {}) if isinstance(getattr(base_model, "config", {}), dict) else {}
        self.base_cfg = base_cfg
        ptm_names = [sanitize_name(x) for x in base_cfg.get("multitask_ptm_types", []) if str(x).strip()]
        self.use_last_task_context = bool(base_cfg.get("use_last_foreground", False))
        self.last_task_context_scale = float(base_cfg.get("last_task_context_scale", 1.0))  # Changed from 0.2 to 1.0 to let LaST work
        if not self.use_last_task_context and bool(base_cfg.get("use_learnable_task_attention", False)):
            self.last_task_context_scale = float(base_cfg.get("learnable_task_attention_head_scale", 0.5))
        self.use_label_residual_layer = bool(
            base_cfg.get("use_label_residual_layer", base_cfg.get("use_crosstalk_graph_head", False))
        )
        self.label_residual_l1_lambda = float(
            base_cfg.get("label_residual_l1_lambda", base_cfg.get("crosstalk_l1_lambda", 1e-4))
        )
        in_dim = int(getattr(base_model.classifier, "in_features", hidden_dim))
        self.shared_dim = in_dim
        self.base_model.classifier = nn.Identity()
        self.use_task_adapters = bool(use_task_adapters) or bool(base_cfg.get("use_task_adapters", False))
        self.task_adapter_residual_scale = float(base_cfg.get("task_adapter_residual_scale", 0.1))
        if self.use_task_adapters:
            adapter_dim = max(1, int(base_cfg.get("task_adapter_dim", adapter_dim)))
            self.task_adapters = nn.ModuleDict(
                {
                    f"task_{idx}": nn.Sequential(
                        nn.Linear(in_dim, adapter_dim),
                        nn.GELU(),
                        nn.Dropout(float(adapter_dropout)),
                        nn.Linear(adapter_dim, in_dim),
                    )
                    for idx in range(self.num_tasks)
                }
            )
        else:
            self.task_adapters = None
            
        self.use_task_specific_gates = bool(base_cfg.get("use_task_specific_gates", False))
        if self.use_task_specific_gates:
            self.task_gates = nn.ModuleDict(
                {
                    f"task_{idx}": nn.Sequential(
                        nn.Linear(in_dim, in_dim),
                        nn.Sigmoid()
                    )
                    for idx in range(self.num_tasks)
                }
            )
        else:
            self.task_gates = None
            
        self.task_heads = nn.ModuleDict(
            {
                f"task_{idx}": nn.Sequential(
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), 1),
                )
                for idx in range(self.num_tasks)
            }
        )
        self.use_ptm_specialists = bool(base_cfg.get("use_ptm_specialists", False))
        # V3: DynamicCosineClassifier replaces the per-task linear heads when enabled.
        self.use_dynamic_cosine = bool(base_cfg.get("use_dynamic_cosine_classifier", False))
        self.cosine_classifier = None
        if self.use_dynamic_cosine:
            from src.models.dynamic_cosine_classifier import DynamicCosineClassifier

            cosine_temp = float(base_cfg.get("cosine_temperature", 0.07))
            cosine_momentum = float(base_cfg.get("cosine_momentum", 0.9))
            self.cosine_classifier = DynamicCosineClassifier(
                feature_dim=int(in_dim),
                num_tasks=int(self.num_tasks),
                temperature=cosine_temp,
                momentum=cosine_momentum,
                learnable_temperature=bool(base_cfg.get("cosine_learnable_temperature", False)),
                freeze_prototypes=bool(base_cfg.get("cosine_freeze_prototypes", False)),
            )
            # Per-task learnable residual gate for the cosine logits. Starts small so
            # random early prototypes cannot swamp the trained task heads.
            self.cosine_logit_gate = nn.Parameter(
                torch.full((int(self.num_tasks),), float(base_cfg.get("cosine_logit_scale", 0.1)))
            )
            self.logger.info(
                "V3: DynamicCosineClassifier enabled (feature_dim=%d, temperature=%.4f, momentum=%.4f, gate_init=%.3f)",
                int(in_dim), cosine_temp, cosine_momentum, float(base_cfg.get("cosine_logit_scale", 0.1)),
            )
        # V3: supervised contrastive loss on the shared representation (auxiliary term).
        self.contrastive_loss_fn = None
        self.contrastive_weight = 0.0
        if bool(base_cfg.get("use_contrastive_joint_loss", False)):
            from src.utils.contrastive_task_joint_loss import SupervisedContrastiveLoss

            self.contrastive_weight = float(base_cfg.get("contrastive_weight", 0.1))
            self.contrastive_loss_fn = SupervisedContrastiveLoss(
                temperature=float(base_cfg.get("contrastive_temperature", 0.1))
            )
            self.logger.info(
                "V3: supervised contrastive auxiliary loss enabled (weight=%.3f, temperature=%.3f)",
                self.contrastive_weight, float(base_cfg.get("contrastive_temperature", 0.1)),
            )
        self.ptm_specialist_residual_scale = float(base_cfg.get("ptm_specialist_residual_scale", 0.1))
        self.ptm_specialists = None
        self.ptm_specialist_tasks: Set[int] = set()
        self._specialist_missing_warned: Set[str] = set()
        self._specialist_missing_counts: Dict[str, int] = {}
        if self.use_ptm_specialists:
            target_names = [canonical_ptm_name(x) for x in base_cfg.get("ptm_specialist_targets", []) if str(x).strip()]
            self.logger.info(f"AdvancedModelWrapper: Enabling PTM Specialists for targets: {target_names}")
            if target_names and ptm_names:
                self.ptm_specialist_tasks = {
                    idx for idx, name in enumerate(ptm_names)
                    if canonical_ptm_name(name) in target_names
                }
            else:
                self.ptm_specialist_tasks = set(range(self.num_tasks))
            
            # Use the specialized extractors from ptm_specialists.py if available
            try:
                from src.models.ptm_specialists import PTMSpecialistExtractors
                # The input sequence window is typically 1280 dim (ESM-2)
                seq_dim = int(base_cfg.get("sequence_hidden_dim", 1280))
                self.specialist_extractors = PTMSpecialistExtractors(
                    seq_dim,
                    task_names=ptm_names,
                    enhanced_targets=base_cfg.get(
                        "ptm_specialist_enhanced_targets",
                        ["crotonylation", "2hydroxyisobutyryl", "lactylation"],
                    ),
                )
                self.logger.info("Successfully loaded biological property-based PTMSpecialistExtractors")
                
                # Projection layers to map from specialist feature space back to in_dim
                specialist_hidden = int(base_cfg.get("ptm_specialist_hidden_dim", hidden_dim))
                # PTMSpecialistExtractors returns Dim/4, where Dim is seq_dim (e.g. 1280)
                extracted_dim = seq_dim // 4
                self.ptm_specialists = nn.ModuleDict(
                    {
                        f"task_{idx}": nn.Sequential(
                            nn.Linear(extracted_dim, specialist_hidden),
                            nn.GELU(),
                            nn.Dropout(float(base_cfg.get("ptm_specialist_dropout", dropout))),
                            nn.Linear(specialist_hidden, in_dim)
                        )
                        for idx in self.ptm_specialist_tasks
                    }
                )
            except ImportError:
                self.logger.warning("Could not import PTMSpecialistExtractors, falling back to basic MLPs")
                self.specialist_extractors = None
                specialist_hidden = int(base_cfg.get("ptm_specialist_hidden_dim", hidden_dim))
                specialist_layers = max(1, int(base_cfg.get("ptm_specialist_layers", 2)))
                specialist_dropout = float(base_cfg.get("ptm_specialist_dropout", dropout))

                def _build_specialist() -> nn.Sequential:
                    layers: List[nn.Module] = []
                    dim = in_dim
                    for _ in range(specialist_layers):
                        layers.append(nn.Linear(dim, specialist_hidden))
                        layers.append(nn.GELU())
                        layers.append(nn.Dropout(specialist_dropout))
                        dim = specialist_hidden
                    layers.append(nn.Linear(dim, in_dim))
                    return nn.Sequential(*layers)

                self.ptm_specialists = nn.ModuleDict(
                    {f"task_{idx}": _build_specialist() for idx in range(self.num_tasks)}
                )
        self.sequence_hidden_dim = int(base_cfg.get("sequence_hidden_dim", 1280))
        self.use_ptm_cross_attention = bool(base_cfg.get("use_ptm_cross_attention", False))
        self.ptm_attention_scale = float(base_cfg.get("ptm_cross_attention_scale", 0.35))
        if self.use_ptm_cross_attention:
            self.ptm_cross_attention = PTMAwareCrossAttention(
                model_dim=in_dim,
                token_dim=self.sequence_hidden_dim,
                num_tasks=self.num_tasks,
                num_heads=int(base_cfg.get("ptm_cross_attention_heads", base_cfg.get("cross_attention_heads", 4))),
                dropout=float(base_cfg.get("ptm_cross_attention_dropout", dropout)),
            )
        else:
            self.ptm_cross_attention = None

        self.private_cka_lambda = float(base_cfg.get("private_cka_lambda", 0.0))
        self.private_cka_target_max = float(base_cfg.get("private_cka_target_max", 0.3))
        self.private_mmd_lambda = float(base_cfg.get("private_mmd_lambda", 0.0))
        self.private_mmd_target_min = float(base_cfg.get("private_mmd_target_min", 0.8))
        self.private_sparsity_lambda = float(base_cfg.get("private_sparsity_lambda", 0.0))
        self.private_sparsity_min = float(base_cfg.get("private_sparsity_min", 0.6))
        self.private_sparsity_max = float(base_cfg.get("private_sparsity_max", 0.8))
        self.gate_diversity_lambda = float(base_cfg.get("gate_diversity_lambda", 0.0))
        self.use_dynamic_loss_weighting = bool(base_cfg.get("use_dynamic_loss_weighting", False))
        self.dynamic_loss_scale_min = float(base_cfg.get("dynamic_loss_scale_min", 0.5))
        self.dynamic_loss_scale_max = float(base_cfg.get("dynamic_loss_scale_max", 2.0))
        self.dynamic_shared_loss_scale = float(base_cfg.get("dynamic_shared_loss_scale_init", 1.0))
        self.dynamic_private_loss_scale = float(base_cfg.get("dynamic_private_loss_scale_init", 1.0))

        self.use_motif_bias = bool(base_cfg.get("use_motif_bias", False))
        if self.use_motif_bias:
            init_scale = float(base_cfg.get("motif_bias_scale", 0.2))
            self.logger.info(f"AdvancedModelWrapper: Enabling Motif Bias with scale={init_scale}")
            self.motif_bias_scale = nn.Parameter(torch.full((self.num_tasks,), init_scale))
        else:
            self.motif_bias_scale = None
        if self.use_label_residual_layer:
            self.crosstalk_adj = nn.Parameter(torch.zeros(self.num_tasks, self.num_tasks))
            nn.init.normal_(self.crosstalk_adj, mean=0.0, std=0.02)
        else:
            self.crosstalk_adj = None

    def _fuse_separator_decision_logits(
        self,
        head_logits: torch.Tensor,
        base_outputs: Any,
        module_contrib: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(base_outputs, dict):
            return head_logits
        separator_logits = base_outputs.get("separator_logits")
        if not isinstance(separator_logits, torch.Tensor) or separator_logits.shape != head_logits.shape:
            return head_logits

        mode = str(self.base_cfg.get("separator_logit_fusion", "separator")).strip().lower()
        separator_logits = torch.nan_to_num(
            separator_logits.to(device=head_logits.device, dtype=head_logits.dtype),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        if mode == "base":
            fused = head_logits
        elif mode == "residual":
            sep_w = float(self.base_cfg.get("separator_logit_private_weight", 0.65))
            fused = head_logits + sep_w * separator_logits
        elif mode in {"weighted_sum", "weighted", "static_gate"}:
            base_w = max(0.0, float(self.base_cfg.get("separator_logit_base_weight", 0.35)))
            sep_w = max(0.0, float(self.base_cfg.get("separator_logit_private_weight", 0.65)))
            norm = max(1e-6, base_w + sep_w)
            fused = (base_w * head_logits + sep_w * separator_logits) / norm
        elif mode in {"learned_gate", "gate", "gated"} and isinstance(
            getattr(self.base_model, "separator_logit_gate", None),
            nn.Parameter,
        ):
            gate = torch.sigmoid(self.base_model.separator_logit_gate).to(
                device=head_logits.device,
                dtype=head_logits.dtype,
            ).view(1, -1)
            fused = (1.0 - gate) * head_logits + gate * separator_logits
            module_contrib["separator_decision_gate_per_task"] = gate.detach().squeeze(0)
            module_contrib["separator_decision_gate_mean"] = gate.detach().mean().view(1)
        else:
            fused = separator_logits

        module_contrib["independent_head_logit_mean_per_task"] = head_logits.detach().mean(dim=0)
        module_contrib["separator_decision_logit_mean_per_task"] = separator_logits.detach().mean(dim=0)
        module_contrib["separator_fused_decision_logit_mean_per_task"] = fused.detach().mean(dim=0)
        return fused

    def _extract_shared_representation(self, base_outputs: Any) -> torch.Tensor:
        if not isinstance(base_outputs, dict):
            return base_outputs

        shared_layers = base_outputs.get("shared_encoder_layers")
        if isinstance(shared_layers, dict):
            candidate = shared_layers.get("shared_post_bottleneck")
            if isinstance(candidate, torch.Tensor) and candidate.ndim >= 2 and int(candidate.shape[-1]) == self.shared_dim:
                return candidate

        for key in ("shared_representation", "features", "fused_representation"):
            candidate = base_outputs.get(key)
            if isinstance(candidate, torch.Tensor) and candidate.ndim >= 2 and int(candidate.shape[-1]) == self.shared_dim:
                return candidate

        logits = base_outputs.get("logits")
        if isinstance(logits, torch.Tensor):
            if logits.ndim >= 2 and int(logits.shape[-1]) == self.shared_dim:
                return logits
            if logits.ndim >= 2 and int(logits.shape[-1]) == self.num_tasks and self.shared_dim != self.num_tasks:
                raise RuntimeError(
                    "MultiTaskIndependentHead expected a shared representation with "
                    f"last dim {self.shared_dim}, but base model returned task logits "
                    f"with last dim {int(logits.shape[-1])}. "
                    "Check that DualBranchFusionPredictor returns shared_encoder_layers['shared_post_bottleneck']."
                )
            return logits

        raise RuntimeError("Base model output does not contain a tensor shared representation or logits.")

    def set_dynamic_loss_scales(self, shared_scale: float, private_scale: float) -> None:
        lo = min(self.dynamic_loss_scale_min, self.dynamic_loss_scale_max)
        hi = max(self.dynamic_loss_scale_min, self.dynamic_loss_scale_max)
        self.dynamic_shared_loss_scale = float(np.clip(shared_scale, lo, hi))
        self.dynamic_private_loss_scale = float(np.clip(private_scale, lo, hi))

    def get_cosine_classifier(self) -> Optional[nn.Module]:
        """Return the classifier owned by this wrapper, independent of nesting."""
        return getattr(self, "cosine_classifier", None)

    @staticmethod
    def _linear_cka_matrix(features: torch.Tensor) -> torch.Tensor:
        task_count = int(features.shape[1])
        cka = features.new_zeros((task_count, task_count))
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
    def _pairwise_mmd_matrix(features: torch.Tensor, max_samples: int = 128) -> torch.Tensor:
        task_count = int(features.shape[1])
        feats = F.normalize(features.float(), dim=-1)
        # MMD uses an O(B^2) distance matrix.  Keep the diagnostic bounded for
        # the requested large batches while preserving a deterministic sample.
        max_samples = max(1, int(max_samples))
        if feats.shape[0] > max_samples:
            indices = torch.linspace(
                0, feats.shape[0] - 1, steps=max_samples, device=feats.device
            ).round().long()
            feats = feats.index_select(0, indices)
        out = feats.new_zeros((task_count, task_count))
        for i in range(task_count):
            xi = feats[:, i, :]
            for j in range(i, task_count):
                xj = feats[:, j, :]
                d_xx = torch.cdist(xi, xi, p=2).pow(2)
                d_yy = torch.cdist(xj, xj, p=2).pow(2)
                d_xy = torch.cdist(xi, xj, p=2).pow(2)
                sigma = torch.cat([d_xx.reshape(-1), d_yy.reshape(-1), d_xy.reshape(-1)], dim=0).detach().median().clamp(min=1e-3)
                k_xx = torch.exp(-d_xx / (2.0 * sigma))
                k_yy = torch.exp(-d_yy / (2.0 * sigma))
                k_xy = torch.exp(-d_xy / (2.0 * sigma))
                mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
                out[i, j] = mmd
                out[j, i] = mmd
        return out.clamp(min=0.0)

    @staticmethod
    def _feature_utilization(features: torch.Tensor) -> torch.Tensor:
        abs_feat = features.float().abs()
        threshold = abs_feat.mean(dim=-1, keepdim=True) + 0.25 * abs_feat.std(dim=-1, keepdim=True)
        return (abs_feat > threshold).float().mean(dim=-1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.base_model, "ptm_types"):
            self.base_model.ptm_types = self.base_cfg.get(
                "multitask_ptm_types",
                ["ubiquitination", "sumoylation", "succinylation", "methylation", "crotonylation", "acetylation"],
            )

        base_outputs = self.base_model(batch)
        shared = self._extract_shared_representation(base_outputs)
        if shared.ndim == 1:
            shared = shared.unsqueeze(-1)

        task_context = None
        if isinstance(base_outputs, dict):
            if self.use_last_task_context:
                candidate = base_outputs.get("last_task_context")
            else:
                candidate = base_outputs.get("learnable_task_context")
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 3 and candidate.shape[1] == self.num_tasks:
                if candidate.shape[2] == shared.shape[-1]:
                    task_context = candidate

        module_contrib: Dict[str, torch.Tensor] = {}
        if task_context is not None:
            module_contrib["last_task_context_norm"] = task_context.detach().norm(dim=-1).mean(dim=0)

        seq_block = batch.get("sequence", {}) if isinstance(batch, dict) else {}
        seq_features = seq_block.get("window_features") if isinstance(seq_block, dict) else None
        seq_mask = seq_block.get("mask") if isinstance(seq_block, dict) else None

        specialist_feats = {}
        extracted_dim = self.sequence_hidden_dim // 4
        if getattr(self, "specialist_extractors", None) is not None and isinstance(seq_features, torch.Tensor):
            specialist_feats = self.specialist_extractors(seq_features, mask=seq_mask)

        private_feature_list: List[torch.Tensor] = []
        gate_means: List[torch.Tensor] = []
        for idx in range(self.num_tasks):
            task_input = shared
            if task_context is not None:
                task_input = task_input + self.last_task_context_scale * task_context[:, idx, :]
            if getattr(self, "use_task_specific_gates", False) and self.task_gates is not None:
                gate_weights = self.task_gates[f"task_{idx}"](task_input)
                task_input = task_input * gate_weights
                module_contrib[f"task_{idx}_gate_mean"] = gate_weights.detach().mean()
                if self.gate_diversity_lambda > 0:
                    gate_means.append(gate_weights.mean(dim=0))
            if self.task_adapters is not None:
                adapter_delta = self.task_adapters[f"task_{idx}"](task_input)
                adapter_delta = self.task_adapter_residual_scale * adapter_delta
                module_contrib[f"task_{idx}_adapter_delta_norm"] = adapter_delta.detach().norm(dim=-1).mean()
                task_input = task_input + adapter_delta
            if self.ptm_specialists is not None and idx in self.ptm_specialist_tasks:
                if getattr(self, "specialist_extractors", None) is not None:
                    task_name = getattr(self.base_model, "ptm_types", [""] * self.num_tasks)[idx] if hasattr(self.base_model, "ptm_types") else ""
                    feat = specialist_feats.get(canonical_ptm_name(task_name))
                    if feat is None:
                        key = sanitize_name(task_name)
                        self._specialist_missing_counts[key] = self._specialist_missing_counts.get(key, 0) + 1
                        if key not in self._specialist_missing_warned:
                            self.logger.warning(
                                "Specialist feature unavailable for task=%s; skipping this batch's specialist residual. "
                                "Training continues without dummy features.",
                                task_name,
                            )
                            self._specialist_missing_warned.add(key)
                    else:
                        task_input = task_input + self.ptm_specialist_residual_scale * self.ptm_specialists[f"task_{idx}"](feat)
                else:
                    task_input = task_input + self.ptm_specialist_residual_scale * self.ptm_specialists[f"task_{idx}"](task_input)
            private_feature_list.append(task_input)

        private_features = torch.stack(private_feature_list, dim=1)
        module_contrib["private_feature_norm_per_task"] = private_features.detach().norm(dim=-1).mean(dim=0)

        if self.ptm_cross_attention is not None:
            attended_features, attention_weights = self.ptm_cross_attention(private_features, seq_features, seq_mask=seq_mask)
            if attended_features is not None:
                private_features = private_features + self.ptm_attention_scale * attended_features
                module_contrib["ptm_attention_feature_norm"] = attended_features.detach().norm(dim=-1).mean(dim=0)
            if attention_weights is not None:
                module_contrib["ptm_attention_peak_per_task"] = attention_weights.detach().max(dim=-1).values.mean(dim=0)
                attn_prob = attention_weights.detach().clamp(min=1e-8)
                attn_entropy = -(attn_prob * torch.log(attn_prob)).sum(dim=-1).mean(dim=0)
                module_contrib["ptm_attention_entropy_per_task"] = attn_entropy

        task_logits = [self.task_heads[f"task_{idx}"](private_features[:, idx, :]) for idx in range(self.num_tasks)]
        head_logits_raw = torch.cat(task_logits, dim=-1)

        aux_losses: Dict[str, torch.Tensor] = {}

        if self.cosine_classifier is not None:
            # V3: prototype-based cosine logits on the shared representation.
            # The cosine term is fused as a residual so the learned task heads
            # keep their capacity while the prototypes inject a class-balanced,
            # long-tail-robust prior.
            cos_labels = None
            if self.training and isinstance(batch, dict):
                cand = batch.get("label")
                if isinstance(cand, torch.Tensor) and cand.ndim == 2 and cand.shape[1] == self.num_tasks:
                    cos_labels = cand.to(device=shared.device, dtype=shared.dtype)
            cos_logits = self.cosine_classifier(
                shared,
                labels=cos_labels,
                update_prototypes=self.training,
            )
            cos_logits = torch.nan_to_num(cos_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            fusion_mode = str(self.base_cfg.get("cosine_fusion_mode", "residual")).strip().lower()
            gate = self.cosine_logit_gate.to(device=cos_logits.device, dtype=cos_logits.dtype).view(1, -1)
            if fusion_mode == "replace":
                head_logits_raw = cos_logits
            else:  # residual: task heads + learnable per-task gate * cosine logits
                head_logits_raw = head_logits_raw + gate * cos_logits
            module_contrib["cosine_logit_abs_mean"] = cos_logits.detach().abs().mean().view(1)
            module_contrib["cosine_logit_gate_per_task"] = self.cosine_logit_gate.detach().clone()

        logits_raw = self._fuse_separator_decision_logits(head_logits_raw, base_outputs, module_contrib)
        logits = logits_raw

        if self.num_tasks > 1:
            eye = torch.eye(self.num_tasks, dtype=private_features.dtype, device=private_features.device)
            offdiag = (1.0 - eye)

            if self.gate_diversity_lambda > 0 and len(gate_means) == self.num_tasks:
                gate_stack = torch.stack(gate_means, dim=0)
                gate_norm = F.normalize(gate_stack, dim=-1)
                gate_sim = torch.matmul(gate_norm, gate_norm.t())
                gate_sim_offdiag = (gate_sim * offdiag).sum() / offdiag.sum().clamp(min=1.0)
                aux_losses["gate_diversity"] = gate_sim_offdiag * self.gate_diversity_lambda
                module_contrib["gate_diversity_offdiag_mean"] = gate_sim_offdiag.detach().view(1)

            cka_matrix = self._linear_cka_matrix(private_features)
            cka_offdiag = (cka_matrix * offdiag).sum() / offdiag.sum().clamp(min=1.0)
            module_contrib["private_cka_matrix"] = cka_matrix.detach()
            module_contrib["private_cka_offdiag_mean"] = cka_offdiag.detach().view(1)
            if self.private_cka_lambda > 0:
                cka_loss = torch.relu(cka_matrix - self.private_cka_target_max) * offdiag
                cka_mean = cka_loss.sum() / offdiag.sum().clamp(min=1.0)
                aux_losses["private_cka_orth"] = cka_mean * self.private_cka_lambda
                module_contrib["private_cka_orth_raw"] = cka_mean.detach().view(1)

            if self.private_mmd_lambda > 0:
                mmd_matrix = self._pairwise_mmd_matrix(
                    private_features,
                    max_samples=int(self.base_cfg.get("private_mmd_max_samples", 128)),
                )
            else:
                # Do not allocate O(B^2) matrices when MMD is disabled.
                mmd_matrix = private_features.new_zeros((self.num_tasks, self.num_tasks))
            mmd_offdiag = (mmd_matrix * offdiag).sum() / offdiag.sum().clamp(min=1.0)
            module_contrib["private_mmd_matrix"] = mmd_matrix.detach()
            module_contrib["private_mmd_offdiag_mean"] = mmd_offdiag.detach().view(1)
            if self.private_mmd_lambda > 0:
                mmd_loss = torch.relu(self.private_mmd_target_min - mmd_matrix) * offdiag
                aux_losses["private_mmd_separation"] = mmd_loss.sum() / offdiag.sum().clamp(min=1.0)

        utilization = self._feature_utilization(private_features)
        utilization_mean_per_task = utilization.detach().mean(dim=0)
        utilization_mean = utilization_mean_per_task.mean()
        module_contrib["private_feature_utilization_per_task"] = utilization_mean_per_task
        module_contrib["private_feature_utilization_mean"] = utilization_mean.view(1)
        if self.private_sparsity_lambda > 0:
            sparsity_loss = torch.relu(self.private_sparsity_min - utilization) + torch.relu(utilization - self.private_sparsity_max)
            aux_losses["private_feature_sparsity"] = sparsity_loss.mean()

        label_residual_aux_loss = None
        if self.crosstalk_adj is not None:
            eye = torch.eye(self.num_tasks, dtype=logits_raw.dtype, device=logits_raw.device)
            adj = self.crosstalk_adj * (1.0 - eye)
            logits = logits_raw + torch.matmul(logits_raw, adj.t())
            module_contrib["label_residual_delta_norm"] = (logits - logits_raw).detach().abs().mean()
            module_contrib["label_residual_adj_l1"] = adj.detach().abs().mean()
            if self.label_residual_l1_lambda > 0:
                label_residual_aux_loss = adj.abs().mean() * self.label_residual_l1_lambda

        if self.use_motif_bias and isinstance(batch, dict):
            motif_bias = batch.get("motif_bias")
            if isinstance(motif_bias, torch.Tensor) and motif_bias.ndim == 2 and motif_bias.shape[1] == self.num_tasks:
                motif_bias = motif_bias.to(dtype=logits.dtype, device=logits.device)
                if isinstance(self.motif_bias_scale, torch.Tensor):
                    scale = self.motif_bias_scale.to(dtype=logits.dtype, device=logits.device)
                    logits = logits + motif_bias * scale
                else:
                    logits = logits + motif_bias
                module_contrib["motif_bias_mean"] = motif_bias.detach().mean(dim=0)

        # V3: supervised contrastive loss on the shared representation.
        if self.contrastive_loss_fn is not None and shared.size(0) > 1:
            contrast_labels = None
            if isinstance(batch, dict):
                cand = batch.get("label")
                if isinstance(cand, torch.Tensor) and cand.ndim == 2 and cand.shape[1] == self.num_tasks:
                    contrast_labels = cand.to(device=shared.device, dtype=shared.dtype)
            if contrast_labels is not None:
                scl = self.contrastive_loss_fn(shared.float(), contrast_labels.float())
                if isinstance(scl, torch.Tensor) and torch.isfinite(scl):
                    aux_losses["contrastive_task_joint"] = scl * float(self.contrastive_weight)
                    module_contrib["contrastive_loss_raw"] = scl.detach()
                    module_contrib["contrastive_loss_weighted"] = (scl.detach() * float(self.contrastive_weight))
                    for stat_name, stat_value in self.contrastive_loss_fn.last_stats.items():
                        module_contrib[f"contrastive_{stat_name}"] = stat_value.detach().to(shared.device)

        if isinstance(base_outputs, dict):
            base_aux_losses = base_outputs.get("auxiliary_losses")
            if not isinstance(base_aux_losses, dict):
                base_aux_losses = {}
                base_outputs["auxiliary_losses"] = base_aux_losses
            base_aux_losses.update(aux_losses)
            if label_residual_aux_loss is not None:
                base_aux_losses["label_residual_l1"] = label_residual_aux_loss

            if self.use_dynamic_loss_weighting:
                for key, value in list(base_aux_losses.items()):
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key.startswith("last_"):
                        base_aux_losses[key] = value * float(self.dynamic_shared_loss_scale)
                    elif key.startswith("private_"):
                        base_aux_losses[key] = value * float(self.dynamic_private_loss_scale)
                module_contrib["dynamic_shared_loss_scale"] = torch.tensor(
                    [self.dynamic_shared_loss_scale],
                    device=logits.device,
                    dtype=logits.dtype,
                )
                module_contrib["dynamic_private_loss_scale"] = torch.tensor(
                    [self.dynamic_private_loss_scale],
                    device=logits.device,
                    dtype=logits.dtype,
                )

            existing_contrib = base_outputs.get("module_contributions")
            if isinstance(existing_contrib, dict):
                existing_contrib.update(module_contrib)
                base_outputs["module_contributions"] = existing_contrib
            elif module_contrib:
                base_outputs["module_contributions"] = module_contrib

            base_outputs["shared_representation"] = shared
            base_outputs["private_feature_tensor"] = private_features
            base_outputs["independent_head_logits"] = head_logits_raw
            base_outputs["raw_task_logits"] = logits_raw
            base_outputs["logits"] = logits
            return base_outputs
        return {
            "logits": logits,
            "shared_representation": shared,
            "private_feature_tensor": private_features,
            "independent_head_logits": head_logits_raw,
            "raw_task_logits": logits_raw,
            "module_contributions": module_contrib,
            "auxiliary_losses": aux_losses,
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        Dynamically freeze or unfreeze the backbone (ESM Sequence Tower & Structure Graph Tower).
        Leaves the LaST foreground blocks, adapters, and task heads always trainable.
        This enables the Three-Stage Training strategy.
        """
        for name, param in self.base_model.named_parameters():
            # Keywords indicating layers that should ALWAYS be trainable (LaST, adapters, gates, heads)
            if any(k in name for k in ["last_foreground", "task_adapters", "task_gates", "ptm_specialists", "task_heads"]):
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def set_last_module_trainable(self, trainable: bool) -> None:
        if hasattr(self.base_model, "set_last_module_trainable"):
            self.base_model.set_last_module_trainable(bool(trainable))

    def get_grouped_parameters(self, base_lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        seen: set = set()

        def _append(params: List[nn.Parameter], lr: float) -> None:
            uniq = []
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                uniq.append(p)
            if uniq:
                groups.append({"params": uniq, "lr": float(lr), "weight_decay": float(weight_decay)})

        if hasattr(self.base_model, "get_grouped_parameters"):
            base_groups = self.base_model.get_grouped_parameters(base_lr, weight_decay)
            for g in base_groups:
                params = [p for p in g.get("params", []) if isinstance(p, nn.Parameter)]
                _append(params, float(g.get("lr", base_lr)))
        else:
            _append([p for p in self.base_model.parameters() if p.requires_grad], float(base_lr))

        head_lr = base_lr * float(self.base_cfg.get("lr_scale_fusion", 1.0))
        head_params: List[nn.Parameter] = []
        head_params.extend([p for p in self.task_heads.parameters() if p.requires_grad])
        if self.cosine_classifier is not None:
            head_params.extend([p for p in self.cosine_classifier.parameters() if p.requires_grad])
        if self.task_adapters is not None:
            head_params.extend([p for p in self.task_adapters.parameters() if p.requires_grad])
        if self.ptm_specialists is not None:
            head_params.extend([p for p in self.ptm_specialists.parameters() if p.requires_grad])
        if self.ptm_cross_attention is not None:
            head_params.extend([p for p in self.ptm_cross_attention.parameters() if p.requires_grad])
        if isinstance(self.crosstalk_adj, nn.Parameter) and self.crosstalk_adj.requires_grad:
            head_params.append(self.crosstalk_adj)
        if isinstance(self.motif_bias_scale, nn.Parameter) and self.motif_bias_scale.requires_grad:
            head_params.append(self.motif_bias_scale)
        _append(head_params, head_lr)

        # The cosine gate is a single 8-vector that decides how much the
        # prototype branch contributes.  Give it its own LR group so it can
        # move on the same timescale as the task heads instead of being
        # effectively frozen by the tiny backbone LR.
        if self.cosine_classifier is not None and self.cosine_logit_gate.requires_grad:
            gate_lr = head_lr * float(self.base_cfg.get("cosine_gate_lr_scale", 1.0))
            gate_id = id(self.cosine_logit_gate)
            if gate_id not in seen:
                seen.add(gate_id)
                groups.append({
                    "params": [self.cosine_logit_gate],
                    "lr": float(gate_lr),
                    "weight_decay": 0.0,
                })

        return groups


def build_model(config: Dict[str, Any], num_tasks: int) -> nn.Module:
    model_cfg = copy.deepcopy(config)
    model_cfg["num_tasks"] = int(num_tasks)

    # Disable binary-only auxiliary objectives that assume scalar labels.
    model_cfg["aux_cls_weight"] = 0.0
    model_cfg["enable_branch_classifiers"] = False

    model = DualBranchFusionPredictor(model_cfg)

    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Linear):
        raise RuntimeError("DualBranchFusionPredictor.classifier is not a linear layer.")
    if bool(config.get("use_task_specific_heads", True)):
        return MultiTaskIndependentHead(
            base_model=model,
            num_tasks=num_tasks,
            hidden_dim=int(config.get("task_head_hidden_dim", 256)),
            dropout=float(config.get("task_head_dropout", config.get("output_dropout", 0.1))),
            use_task_adapters=bool(config.get("use_task_adapters", False)),
            adapter_dim=int(config.get("task_adapter_dim", 128)),
            adapter_dropout=float(config.get("task_adapter_dropout", config.get("output_dropout", 0.1))),
        )
    in_dim = int(model.classifier.in_features)
    model.classifier = nn.Linear(in_dim, num_tasks)
    return model
