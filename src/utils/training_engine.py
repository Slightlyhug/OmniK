"""Single-epoch training and evaluation engine."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.training_config import sanitize_name
from src.utils.training_data import to_device
from src.utils.training_runtime import summarize_epoch_performance
from src.utils.training_samplers import BatchDistributionMonitor
from src.utils.training_state import EpochOptions


def _broadcast_ddp_state(model: nn.Module, criterion: Optional[nn.Module] = None) -> None:
    """Keep all ranks bit-identical when DDP has unused parameters.

    find_unused_parameters=True does not synchronize parameters that were not
    used on every rank, so broadcast rank-0 values after each optimizer step.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if os.environ.get("DDP_BROADCAST_PARAMS", "1") != "1":
        return
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for buf in model.buffers():
        torch.distributed.broadcast(buf.data, src=0)
    if criterion is not None:
        for param in criterion.parameters():
            torch.distributed.broadcast(param.data, src=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    device: torch.device,
    options: EpochOptions,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    is_train = optimizer is not None
    model.train(is_train)
    grad_accum_steps = options.grad_accum_steps
    max_batches = options.max_batches
    grad_norm_log_interval = options.grad_norm_log_interval
    grad_norm_topk = options.grad_norm_topk
    grad_norm_warn_threshold = options.grad_norm_warn_threshold
    scaler = options.scaler
    use_amp = options.use_amp
    amp_dtype = options.amp_dtype or torch.float16
    distribution_monitor = options.distribution_monitor
    stage_name = options.stage_name
    grad_clip_norm = options.grad_clip_norm
    log_interval = options.log_interval
    scheduler_step_unit = str(options.scheduler_step_unit or "step").lower()
    gradient_conflict_log_interval = options.gradient_conflict_log_interval
    teacher_distill_weight = options.teacher_distill_weight
    teacher_distill_temperature = options.teacher_distill_temperature
    teacher_distill_min_confidence = options.teacher_distill_min_confidence
    teacher_distill_unknown_only = options.teacher_distill_unknown_only

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_structure_available: List[np.ndarray] = []
    running_loss = 0.0
    total = 0
    non_finite_loss_batches = 0
    non_finite_grad_skip_steps = 0
    module_contrib_sums: Dict[str, np.ndarray] = {}
    module_contrib_counts: Dict[str, int] = {}
    aux_log_sums: Dict[str, float] = {}
    aux_log_counts: Dict[str, int] = {}

    if is_train:
        optimizer.zero_grad(set_to_none=True)
    logger = logging.getLogger("multitask_lysine")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    epoch_start = time.perf_counter()
    model_cfg = getattr(model, "base_cfg", None)
    if not isinstance(model_cfg, dict) and hasattr(model, "base_model"):
        candidate_cfg = getattr(model.base_model, "config", None)
        model_cfg = candidate_cfg if isinstance(candidate_cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    shared_grad_penalty_lambda = float(model_cfg.get("shared_grad_penalty_lambda", 0.0))
    shared_grad_target_min = float(model_cfg.get("shared_grad_target_min", 0.1))
    shared_grad_target_max = float(model_cfg.get("shared_grad_target_max", 0.5))
    use_dynamic_loss_weighting = bool(model_cfg.get("use_dynamic_loss_weighting", False))

    def _iter_checked_params(m: nn.Module, opt: Optional[torch.optim.Optimizer]) -> List[torch.nn.Parameter]:
        if opt is None:
            return [p for p in m.parameters() if p.requires_grad]
        checked: List[torch.nn.Parameter] = []
        seen: set = set()
        for group in opt.param_groups:
            for p in group.get("params", []):
                if not isinstance(p, torch.nn.Parameter) or not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                checked.append(p)
        return checked

    def _has_non_finite_grad(m: nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> bool:
        for p in _iter_checked_params(m, opt):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def _non_finite_grad_param_names(
        m: nn.Module,
        opt: Optional[torch.optim.Optimizer] = None,
        max_items: int = 5,
    ) -> List[str]:
        bad: List[str] = []
        id_to_name = {id(p): n for n, p in m.named_parameters()}
        for p in _iter_checked_params(m, opt):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad.append(id_to_name.get(id(p), "<unnamed_param>"))
                if len(bad) >= max_items:
                    break
        return bad

    def _repair_non_finite_state(
        m: nn.Module,
        opt: Optional[torch.optim.Optimizer],
        param_clip_value: float = 25.0,
    ) -> Dict[str, Any]:
        id_to_name = {id(p): n for n, p in m.named_parameters()}
        repaired_params: List[str] = []
        clipped_params = 0
        reset_state_params = 0

        checked = _iter_checked_params(m, opt)
        with torch.no_grad():
            for p in checked:
                name = id_to_name.get(id(p), "<unnamed_param>")
                grad_bad = p.grad is not None and not torch.isfinite(p.grad).all()
                data_finite = torch.isfinite(p.data).all()
                data_bad = not data_finite

                if data_bad:
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=param_clip_value, neginf=-param_clip_value)
                    p.data.clamp_(-param_clip_value, param_clip_value)
                    repaired_params.append(name)
                else:
                    max_abs = float(p.data.detach().abs().max().item()) if p.data.numel() > 0 else 0.0
                    if max_abs > param_clip_value:
                        p.data.clamp_(-param_clip_value, param_clip_value)
                        clipped_params += 1

                if grad_bad and p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

                if opt is not None and (grad_bad or data_bad) and p in opt.state and len(opt.state[p]) > 0:
                    opt.state[p].clear()
                    reset_state_params += 1

        return {
            "repaired_params": repaired_params,
            "clipped_params": int(clipped_params),
            "reset_state_params": int(reset_state_params),
        }

    def _sanitize_embedding_grads(
        m: nn.Module,
        opt: Optional[torch.optim.Optimizer],
        grad_abs_clip: float = 5.0,
    ) -> List[str]:
        id_to_name = {id(p): n for n, p in m.named_parameters()}
        sanitized: List[str] = []
        checked = _iter_checked_params(m, opt)
        with torch.no_grad():
            for p in checked:
                if p.grad is None:
                    continue
                name = id_to_name.get(id(p), "")
                if "embedding" not in name:
                    continue
                g = p.grad.data
                has_non_finite = not torch.isfinite(g).all()
                max_abs = float(g.detach().abs().max().item()) if g.numel() > 0 else 0.0
                needs_clip = grad_abs_clip > 0 and max_abs > grad_abs_clip
                if not has_non_finite and not needs_clip:
                    continue
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                if grad_abs_clip > 0:
                    g = g.clamp(-grad_abs_clip, grad_abs_clip)
                p.grad.data.copy_(g)
                sanitized.append(name)
        return sanitized

    def _cosine_gate_param(m: nn.Module) -> Optional[nn.Parameter]:
        for owner in (m, getattr(m, "base_model", None)):
            if owner is None:
                continue
            gate = getattr(owner, "cosine_logit_gate", None)
            if isinstance(gate, nn.Parameter):
                return gate
        return None

    def _module_grad_norms(m: nn.Module) -> Dict[str, float]:
        norm_sq: Dict[str, float] = {}
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                continue
            module_name = name.split(".", 1)[0] if "." in name else name
            g_norm = float(torch.linalg.vector_norm(g).item())
            norm_sq[module_name] = norm_sq.get(module_name, 0.0) + g_norm * g_norm
        return {k: math.sqrt(v) for k, v in norm_sq.items()}

    def _estimate_task_gradient_conflict(outputs_any: Any, targets: torch.Tensor) -> Optional[Dict[str, float]]:
        if not isinstance(outputs_any, dict):
            return None
        raw_task_logits = outputs_any.get("raw_task_logits")
        shared_rep = outputs_any.get("shared_representation")
        if not isinstance(raw_task_logits, torch.Tensor) or not isinstance(shared_rep, torch.Tensor):
            return None
        if raw_task_logits.ndim != 2 or shared_rep.ndim < 2 or not shared_rep.requires_grad:
            return None
        if targets.ndim != 2 or targets.shape != raw_task_logits.shape:
            return None

        grad_vecs: List[torch.Tensor] = []
        for task_idx in range(int(raw_task_logits.shape[1])):
            known = targets[:, task_idx] >= 0.0
            if not bool(known.any()):
                continue
            task_loss = F.binary_cross_entropy_with_logits(
                raw_task_logits[known, task_idx].float(),
                targets[known, task_idx].float().clamp(min=0.0, max=1.0),
                reduction="mean",
            )
            grad = torch.autograd.grad(task_loss, shared_rep, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                continue
            grad_vec = grad.detach().float().reshape(grad.shape[0], -1).mean(dim=0)
            if grad_vec.numel() == 0:
                continue
            norm = torch.linalg.vector_norm(grad_vec)
            if (not torch.isfinite(norm)) or float(norm.item()) <= 0.0:
                continue
            grad_vecs.append(grad_vec / norm.clamp_min(1e-12))

        if len(grad_vecs) < 2:
            return None

        sims: List[float] = []
        neg_count = 0
        for i in range(len(grad_vecs)):
            for j in range(i + 1, len(grad_vecs)):
                sim = float(torch.dot(grad_vecs[i], grad_vecs[j]).detach().cpu().item())
                sims.append(sim)
                if sim < 0.0:
                    neg_count += 1
        if not sims:
            return None
        return {
            "gradient_conflict_mean_cosine": float(np.mean(sims)),
            "gradient_conflict_min_cosine": float(np.min(sims)),
            "gradient_conflict_negative_pair_ratio": float(neg_count / len(sims)),
            "gradient_conflict_pair_count": float(len(sims)),
            "gradient_conflict_task_count": float(len(grad_vecs)),
        }

    def _tensor_summary(values: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(values, torch.Tensor) or values.numel() == 0:
            return None
        clean = values.detach().float().reshape(-1)
        clean = clean[torch.isfinite(clean)]
        if clean.numel() == 0:
            return None
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=clean.device, dtype=clean.dtype)
        q = torch.quantile(clean, quantiles)
        return torch.stack([clean.mean(), clean.std(unbiased=False), q[0], q[1], q[2], clean.max()])

    def _collect_separation_gradient_stats(
        outputs_any: Any,
        total_loss: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Tuple[float, float]]]:
        if not is_train or not isinstance(outputs_any, dict):
            return None, {}, {}, None

        shared_layers = outputs_any.get("shared_encoder_layers")
        private_features = outputs_any.get("private_feature_tensor")
        tracked: List[torch.Tensor] = []
        tracked_names: List[str] = []
        if isinstance(shared_layers, dict):
            for key in ("shared_pre_bottleneck", "shared_post_bottleneck"):
                tensor = shared_layers.get(key)
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tracked.append(tensor)
                    tracked_names.append(f"shared_grad_{key}")
        if isinstance(private_features, torch.Tensor) and private_features.requires_grad:
            tracked.append(private_features)
            tracked_names.append("private_grad_features")
        if not tracked:
            return None, {}, {}, None

        grads = torch.autograd.grad(
            total_loss,
            tracked,
            retain_graph=True,
            create_graph=shared_grad_penalty_lambda > 0.0,
            allow_unused=True,
        )

        local_aux: Dict[str, torch.Tensor] = {}
        local_contrib: Dict[str, torch.Tensor] = {}
        penalty_terms: List[torch.Tensor] = []
        shared_mean = None
        private_mean = None

        for name, grad in zip(tracked_names, grads):
            if grad is None:
                continue
            grad_flat = grad.float().reshape(grad.shape[0], -1)
            norms = torch.linalg.vector_norm(grad_flat, dim=1)
            summary = _tensor_summary(norms)
            if summary is None:
                continue
            local_contrib[f"{name}_l2_summary"] = summary
            local_aux[f"{name}_l2_mean"] = summary[0]
            local_aux[f"{name}_l2_p90"] = summary[4]
            if name.startswith("shared_grad_"):
                penalty_terms.append(torch.relu(shared_grad_target_min - norms) + torch.relu(norms - shared_grad_target_max))
                shared_mean = float(summary[0].detach().cpu().item()) if shared_mean is None else shared_mean
            elif name == "private_grad_features":
                private_mean = float(summary[0].detach().cpu().item())

        if (
            "shared_grad_shared_pre_bottleneck_l2_summary" in local_contrib
            and "shared_grad_shared_post_bottleneck_l2_summary" in local_contrib
        ):
            pre_mean = local_contrib["shared_grad_shared_pre_bottleneck_l2_summary"][0]
            post_mean = local_contrib["shared_grad_shared_post_bottleneck_l2_summary"][0]
            local_contrib["shared_grad_stability_ratio"] = (post_mean / pre_mean.clamp(min=1e-6)).view(1)
            local_aux["shared_grad_stability_ratio"] = post_mean / pre_mean.clamp(min=1e-6)

        penalty = None
        if penalty_terms and shared_grad_penalty_lambda > 0.0:
            penalty = torch.stack([term.mean() for term in penalty_terms]).mean() * shared_grad_penalty_lambda
            local_aux["shared_grad_penalty"] = penalty.detach()

        if shared_mean is not None and private_mean is not None and use_dynamic_loss_weighting and hasattr(model, "set_dynamic_loss_scales"):
            ratio = math.sqrt(max(private_mean, 1e-6) / max(shared_mean, 1e-6))
            dyn_min = float(getattr(model, "dynamic_loss_scale_min", 0.5))
            dyn_max = float(getattr(model, "dynamic_loss_scale_max", 2.0))
            dyn_lo = min(dyn_min, dyn_max)
            dyn_hi = max(dyn_min, dyn_max)
            momentum = float(getattr(model, "dynamic_loss_ema_momentum", 0.8))
            jitter = float(getattr(model, "dynamic_loss_scale_jitter", 0.0))
            prev_shared = float(getattr(model, "dynamic_shared_loss_scale", 1.0))
            prev_private = float(getattr(model, "dynamic_private_loss_scale", 1.0))
            target_shared = float(np.clip(ratio, dyn_lo, dyn_hi))
            target_private = float(np.clip(1.0 / max(target_shared, 1e-6), dyn_lo, dyn_hi))
            shared_scale = (momentum * prev_shared) + ((1.0 - momentum) * target_shared)
            private_scale = (momentum * prev_private) + ((1.0 - momentum) * target_private)
            if jitter > 0.0 and is_train:
                phase = float(step + 1)
                shared_scale += jitter * math.sin(phase / 13.0)
                private_scale += jitter * math.cos(phase / 17.0)
            shared_scale = float(np.clip(shared_scale, dyn_lo, dyn_hi))
            private_scale = float(np.clip(private_scale, dyn_lo, dyn_hi))
            model.set_dynamic_loss_scales(shared_scale, private_scale)
            local_aux["dynamic_shared_loss_scale"] = torch.tensor(shared_scale, device=total_loss.device, dtype=total_loss.dtype)
            local_aux["dynamic_private_loss_scale"] = torch.tensor(private_scale, device=total_loss.device, dtype=total_loss.dtype)
            local_aux["dynamic_shared_target_scale"] = torch.tensor(target_shared, device=total_loss.device, dtype=total_loss.dtype)
            local_aux["dynamic_private_target_scale"] = torch.tensor(target_private, device=total_loss.device, dtype=total_loss.dtype)
            return penalty, local_aux, local_contrib, (shared_scale, private_scale)

        return penalty, local_aux, local_contrib, None

    def _accumulate_module_contributions(contrib: Any) -> None:
        if not isinstance(contrib, dict):
            return
        for key, value in contrib.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().float()
                # Do not average matrices if they are intended to be matrices
                if arr.ndim > 1 and not key.endswith("_matrix"):
                    arr = arr.mean(dim=0)
                arr_np = arr.cpu().numpy().astype(np.float64, copy=False)
            elif isinstance(value, (float, int, np.floating, np.integer)):
                arr_np = np.asarray(float(value), dtype=np.float64)
            elif isinstance(value, (list, tuple)):
                try:
                    arr_np = np.asarray(value, dtype=np.float64)
                except Exception:
                    continue
            else:
                continue

            if key not in module_contrib_sums:
                module_contrib_sums[key] = np.array(arr_np, dtype=np.float64, copy=True)
                module_contrib_counts[key] = 1
            else:
                module_contrib_sums[key] = module_contrib_sums[key] + arr_np
                module_contrib_counts[key] = module_contrib_counts[key] + 1

    def _accumulate_aux_logs(aux_logs: Any) -> None:
        if not isinstance(aux_logs, dict):
            return
        for key, value in aux_logs.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                scalar = float(value.detach().float().mean().cpu().item())
            elif isinstance(value, (float, int, np.floating, np.integer)):
                scalar = float(value)
            else:
                continue
            aux_log_sums[key] = aux_log_sums.get(key, 0.0) + scalar
            aux_log_counts[key] = aux_log_counts.get(key, 0) + 1

    optimizer_step_idx = 0
    batch_counter = 0
    embedding_sanitize_events = 0

    # V3: uncertainty trajectory analyzer (attached to the model by the training loop).
    uta = getattr(model, "_uta", None)
    if uta is not None and is_train:
        uta.on_epoch_start(int(getattr(model, "_uta_epoch", 0)))

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch_counter += 1
        batch_dev = to_device(batch, device)
        labels = batch_dev["label"]
        if distribution_monitor is not None:
            distribution_monitor.update(labels, logger, step + 1, stage=stage_name)

        with torch.set_grad_enabled(is_train):
            amp_enabled = bool(use_amp and device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype)
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype)
            )
            with autocast_context:
                outputs = model(batch_dev)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                if isinstance(outputs, dict):
                    _accumulate_module_contributions(outputs.get("module_contributions"))
                    _accumulate_aux_logs(outputs.get("auxiliary_logs"))
                    if uta is not None and is_train:
                        # V3: track per-sample cumulative entropy on the shared features.
                        shared_feat = outputs.get("shared_representation")
                        sample_ids = batch.get("sample_indices") if isinstance(batch, dict) else None
                        if isinstance(sample_ids, list) and len(sample_ids) == int(logits.shape[0]):
                            fallback_uids = batch.get("uniprot_ids", [])
                            fallback_positions = batch.get("positions", [])
                            fallback_residues = batch.get("residues", [])
                            sample_ids = [
                                str(sample_id)
                                if sample_id is not None
                                else f"{fallback_uids[i] if i < len(fallback_uids) else 'unknown'}_"
                                f"{fallback_positions[i] if i < len(fallback_positions) else i}_"
                                f"{fallback_residues[i] if i < len(fallback_residues) else 'K'}"
                                for i, sample_id in enumerate(sample_ids)
                            ]
                        if isinstance(sample_ids, list) and len(sample_ids) == int(logits.shape[0]):
                            feat_for_uta = shared_feat
                            if not isinstance(feat_for_uta, torch.Tensor):
                                feat_for_uta = None
                            uta.update(
                                [str(s) for s in sample_ids],
                                logits.detach().float(),
                                labels.detach().float(),
                                features=feat_for_uta.detach().float() if feat_for_uta is not None else None,
                            )
                    if is_train and gradient_conflict_log_interval > 0 and ((step + 1) % gradient_conflict_log_interval == 0):
                        conflict_logs = _estimate_task_gradient_conflict(outputs, labels)
                        if isinstance(conflict_logs, dict):
                            _accumulate_aux_logs(conflict_logs)
                            logger.info(
                                "GradConflict step=%d | mean_cos=%.4f min_cos=%.4f neg_ratio=%.4f tasks=%d",
                                step + 1,
                                float(conflict_logs["gradient_conflict_mean_cosine"]),
                                float(conflict_logs["gradient_conflict_min_cosine"]),
                                float(conflict_logs["gradient_conflict_negative_pair_ratio"]),
                                int(conflict_logs["gradient_conflict_task_count"]),
                            )

                sample_weight = batch_dev.get("loss_weight")
                label_state = batch_dev.get("label_state")
                label_reliability = batch_dev.get("label_reliability")
                if sample_weight is not None and bool(getattr(criterion, "supports_sample_weight", False)):
                    loss = criterion(
                        logits,
                        labels,
                        sample_weight=sample_weight,
                        label_state=label_state,
                        label_reliability=label_reliability,
                    )
                else:
                    loss = criterion(logits, labels, label_state=label_state, label_reliability=label_reliability)
                teacher_weight_value = float(teacher_distill_weight) if is_train else 0.0
                if teacher_weight_value > 0.0 and all(k in batch_dev for k in ("teacher_probs", "teacher_mask")):
                    teacher_probs = batch_dev["teacher_probs"].float().to(logits.device)
                    teacher_mask = batch_dev["teacher_mask"].float().to(logits.device)
                    if "teacher_weight" in batch_dev:
                        teacher_mask = teacher_mask * torch.clamp(batch_dev["teacher_weight"].float().to(logits.device), min=0.0, max=10.0)
                    teacher_probs = torch.clamp(teacher_probs, min=1e-4, max=1.0 - 1e-4)
                    if teacher_distill_min_confidence > 0.0:
                        confidence = torch.abs(teacher_probs - 0.5) * 2.0
                        teacher_mask = teacher_mask * (confidence >= float(teacher_distill_min_confidence)).float()
                    if teacher_distill_unknown_only:
                        teacher_mask = teacher_mask * (labels < 0.0).float()
                    if teacher_mask.shape == logits.shape and bool((teacher_mask > 0).any()):
                        temperature = max(1e-3, float(teacher_distill_temperature))
                        student_logits = logits.float() / temperature
                        per_teacher = F.binary_cross_entropy_with_logits(student_logits, teacher_probs, reduction="none")
                        teacher_loss = (per_teacher * teacher_mask).sum() / teacher_mask.sum().clamp(min=1.0)
                        teacher_loss = teacher_loss * (temperature ** 2) * teacher_weight_value
                        loss = loss + teacher_loss
                        _accumulate_aux_logs({
                            "teacher_distill_loss": teacher_loss.detach(),
                            "teacher_distill_mask_fraction": (teacher_mask > 0).float().mean().detach(),
                            "teacher_distill_prob_mean": teacher_probs[teacher_mask > 0].mean().detach(),
                            "teacher_distill_abs_gap": torch.abs(torch.sigmoid(logits.detach().float()) - teacher_probs)[teacher_mask > 0].mean(),
                        })
                if hasattr(criterion, "last_forward_stats"):
                    _accumulate_aux_logs(getattr(criterion, "last_forward_stats"))
                weak_cfg = getattr(model, "config", {}) if isinstance(getattr(model, "config", {}), dict) else {}
                weak_labels = [sanitize_name(x) for x in weak_cfg.get("weak_task_precision_labels", [])]
                weak_penalty_weight = float(weak_cfg.get("weak_task_precision_penalty", 0.0))
                hard_ratio = float(weak_cfg.get("weak_task_hard_negative_ratio", 0.2))
                if weak_labels and weak_penalty_weight > 0.0 and logits.ndim == 2 and labels.ndim == 2:
                    ptm_names = [sanitize_name(x) for x in getattr(model, "ptm_types", [])]
                    weak_penalties: List[torch.Tensor] = []
                    probs_for_precision = torch.sigmoid(logits.float())
                    for weak_name in weak_labels:
                        if weak_name not in ptm_names:
                            continue
                        idx = ptm_names.index(weak_name)
                        neg_mask = labels[:, idx] <= 0.0
                        neg_probs = probs_for_precision[neg_mask, idx]
                        if neg_probs.numel() == 0:
                            continue
                        k = max(1, int(round(float(neg_probs.numel()) * hard_ratio)))
                        hard_neg = torch.topk(neg_probs, k=min(k, int(neg_probs.numel())), largest=True, sorted=False).values
                        weak_penalties.append((hard_neg ** 2).mean())
                    if weak_penalties:
                        weak_precision_loss = torch.stack(weak_penalties).mean() * weak_penalty_weight
                        loss = loss + weak_precision_loss
                        _accumulate_aux_logs({"weak_task_precision_penalty": weak_precision_loss.detach()})
                if isinstance(outputs, dict) and isinstance(outputs.get("auxiliary_losses"), dict):
                    for aux_name, aux_value in outputs["auxiliary_losses"].items():
                        if isinstance(aux_value, torch.Tensor):
                            loss = loss + aux_value
                            _accumulate_aux_logs({f"aux_loss/{aux_name}": aux_value.detach()})
                if isinstance(outputs, dict):
                    grad_penalty, grad_aux_logs, grad_contrib, dynamic_scales = _collect_separation_gradient_stats(outputs, loss)
                    if grad_penalty is not None:
                        loss = loss + grad_penalty
                    if grad_aux_logs:
                        _accumulate_aux_logs(grad_aux_logs)
                    if grad_contrib:
                        _accumulate_module_contributions(grad_contrib)
                    if dynamic_scales is not None and (step + 1) % max(1, grad_accum_steps) == 0:
                        logger.info(
                            "DynamicLoss step=%d | shared_scale=%.4f private_scale=%.4f",
                            step + 1,
                            dynamic_scales[0],
                            dynamic_scales[1],
                        )
            if is_train:
                if not torch.isfinite(loss):
                    logger.warning("Skip batch %d due to non-finite loss: %s", step, str(loss.detach().item()))
                    non_finite_loss_batches += 1
                    repair = _repair_non_finite_state(model, optimizer)
                    if repair["repaired_params"] or repair["reset_state_params"] or repair["clipped_params"] > 0:
                        first_names = ", ".join(repair["repaired_params"][:3]) if repair["repaired_params"] else "<none>"
                        logger.warning(
                            "Applied non-finite recovery after loss anomaly at batch %d | repaired=%d clipped=%d reset_state=%d | first_repaired=%s",
                            step,
                            len(repair["repaired_params"]),
                            int(repair["clipped_params"]),
                            int(repair["reset_state_params"]),
                            first_names,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaled_loss = loss / max(1, grad_accum_steps)
                scaler_active = bool(scaler is not None and amp_enabled and hasattr(scaler, "is_enabled") and scaler.is_enabled())
                if scaler is not None and amp_enabled and not hasattr(scaler, "is_enabled"):
                    scaler_active = True
                if scaler_active:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (step + 1) % max(1, grad_accum_steps) == 0:
                    if scaler_active:
                        try:
                            scaler.unscale_(optimizer)
                        except Exception as exc:
                            logger.warning("GradScaler unscale failed at batch %d: %s", step, str(exc))
                    sanitized_names = _sanitize_embedding_grads(model, optimizer)
                    if sanitized_names:
                        embedding_sanitize_events += 1
                        if embedding_sanitize_events <= 3 or embedding_sanitize_events % 50 == 0:
                            logger.warning(
                                "Sanitized embedding gradients at batch %d. First params: %s",
                                step,
                                ", ".join(sanitized_names[:3]),
                            )
                    if _has_non_finite_grad(model, optimizer):
                        bad_names = _non_finite_grad_param_names(model, optimizer)
                        logger.warning(
                            "Detected non-finite gradients at batch %d; skipping the complete optimizer step. First bad params: %s",
                            step,
                            ", ".join(bad_names) if bad_names else "<unknown>",
                        )
                        non_finite_grad_skip_steps += 1
                        if scaler_active:
                            scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    optimizer_step_idx += 1
                    if grad_norm_log_interval > 0 and (optimizer_step_idx % grad_norm_log_interval == 0):
                        module_norms = _module_grad_norms(model)
                        if module_norms:
                            topk = max(1, int(grad_norm_topk))
                            top_items = sorted(module_norms.items(), key=lambda kv: kv[1], reverse=True)[:topk]
                            top_str = ", ".join([f"{k}:{v:.4f}" for k, v in top_items])
                            logger.info("GradNorm step=%d | top_modules=%s", optimizer_step_idx, top_str)

                            if grad_norm_warn_threshold > 0:
                                alert_items = [(k, v) for k, v in top_items if v >= grad_norm_warn_threshold]
                                if alert_items:
                                    alert_str = ", ".join([f"{k}:{v:.4f}" for k, v in alert_items])
                                    logger.warning(
                                        "GradNormAlert step=%d | threshold=%.4f | modules=%s",
                                        optimizer_step_idx,
                                        grad_norm_warn_threshold,
                                        alert_str,
                                    )

                        gate = _cosine_gate_param(model)
                        if gate is not None and gate.grad is not None:
                            g = gate.grad.detach().float()
                            if torch.isfinite(g).all():
                                logger.info(
                                    "CosineGateGrad step=%d | grad_norm=%.6g grad_absmean=%.6g "
                                    "gate_mean=%.6f gate_min=%.6f gate_max=%.6f",
                                    optimizer_step_idx,
                                    float(torch.linalg.vector_norm(g).item()),
                                    float(g.abs().mean().item()),
                                    float(gate.detach().float().mean().item()),
                                    float(gate.detach().float().min().item()),
                                    float(gate.detach().float().max().item()),
                                )

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max(0.0, float(grad_clip_norm)))
                    if scaler_active:
                        try:
                            scaler.step(optimizer)
                        except AssertionError as exc:
                            logger.warning("GradScaler step failed at batch %d: %s", step, str(exc))
                            optimizer.step()
                        except Exception as exc:
                            logger.warning("GradScaler step error at batch %d: %s", step, str(exc))
                            optimizer.step()
                        try:
                            scaler.update()
                        except Exception as exc:
                            logger.warning("GradScaler update failed at batch %d: %s", step, str(exc))
                    else:
                        optimizer.step()
                    _broadcast_ddp_state(model, criterion)
                    if scheduler is not None and scheduler_step_unit == "step":
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(logits)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).detach().float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().float().cpu().numpy())
        all_structure_available.append(batch_dev["structure_available"].detach().float().cpu().numpy())

        bsz = labels.size(0)
        running_loss += float(loss.detach().item()) * bsz
        total += bsz
        if log_interval > 0 and (step + 1) % log_interval == 0:
            logger.info("%s batch=%d loss=%.5f", stage_name, step + 1, float(loss.detach().item()))

    if is_train and total > 0 and len(loader) % max(1, grad_accum_steps) != 0:
        amp_enabled = bool(use_amp and device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
        scaler_active = bool(scaler is not None and amp_enabled and hasattr(scaler, "is_enabled") and scaler.is_enabled())
        if scaler is not None and amp_enabled and not hasattr(scaler, "is_enabled"):
            scaler_active = True
        if scaler_active:
            try:
                scaler.unscale_(optimizer)
            except Exception as exc:
                logger.warning("GradScaler unscale failed at final step: %s", str(exc))
        sanitized_names = _sanitize_embedding_grads(model, optimizer)
        if sanitized_names:
            embedding_sanitize_events += 1
            if embedding_sanitize_events <= 3 or embedding_sanitize_events % 50 == 0:
                logger.warning(
                    "Sanitized embedding gradients before final step. First params: %s",
                    ", ".join(sanitized_names[:3]),
                )
        if _has_non_finite_grad(model, optimizer):
            bad_names = _non_finite_grad_param_names(model, optimizer)
            logger.warning(
                "Detected non-finite gradients at final step; skipping the complete optimizer step. First bad params: %s",
                ", ".join(bad_names) if bad_names else "<unknown>",
            )
            non_finite_grad_skip_steps += 1
            if scaler_active:
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max(0.0, float(grad_clip_norm)))
            if scaler_active:
                try:
                    scaler.step(optimizer)
                except AssertionError as exc:
                    logger.warning("GradScaler step failed at final step: %s", str(exc))
                    optimizer.step()
                except Exception as exc:
                    logger.warning("GradScaler step error at final step: %s", str(exc))
                    optimizer.step()
                try:
                    scaler.update()
                except Exception as exc:
                    logger.warning("GradScaler update failed at final step: %s", str(exc))
            else:
                optimizer.step()
            _broadcast_ddp_state(model, criterion)
            if scheduler is not None and scheduler_step_unit == "step":
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_idx += 1

    mean_loss = running_loss / max(1, total)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 1), dtype=np.float32)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, 1), dtype=np.float32)
    y_struct_avail = np.concatenate(all_structure_available, axis=0) if all_structure_available else np.zeros((0,), dtype=np.float32)
    perf = summarize_epoch_performance(
        elapsed_time=time.perf_counter() - epoch_start,
        sample_count=total,
        batch_count=batch_counter,
        optimizer_steps=optimizer_step_idx,
        device=device,
    )
    perf["non_finite_loss_batches"] = int(non_finite_loss_batches)
    perf["non_finite_grad_skip_steps"] = int(non_finite_grad_skip_steps)
    perf["non_finite_total_skips"] = int(non_finite_loss_batches + non_finite_grad_skip_steps)
    perf["embedding_grad_sanitize_events"] = int(embedding_sanitize_events)
    if module_contrib_sums:
        aggregated: Dict[str, Any] = {}
        for key, summed in module_contrib_sums.items():
            count = max(1, int(module_contrib_counts.get(key, 1)))
            mean_val = summed / float(count)
            if np.isscalar(mean_val) or getattr(mean_val, "ndim", 0) == 0:
                aggregated[key] = float(mean_val)
            else:
                aggregated[key] = np.asarray(mean_val, dtype=np.float64).tolist()
        perf["module_contributions"] = aggregated
    if aux_log_sums:
        perf["auxiliary_logs"] = {
            key: float(aux_log_sums[key] / max(1, aux_log_counts.get(key, 1)))
            for key in sorted(aux_log_sums.keys())
        }
    perf["stage"] = stage_name
    return mean_loss, y_true, y_prob, y_struct_avail, perf
