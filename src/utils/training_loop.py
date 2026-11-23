"""Epoch orchestration for multi-task model training."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.utils.training_state import (
    EpochOptions,
    LoaderBundle,
    RuntimeDeviceAmp,
    TrainingComponents,
    TrainingLoopServices,
    TrainingState,
)
from src.utils.training_stages import apply_epoch_training_stage


def run_training_loop(
    config: Dict[str, Any],
    ptm_types: List[str],
    loaders: LoaderBundle,
    components: TrainingComponents,
    state: TrainingState,
    runtime_amp: RuntimeDeviceAmp,
    use_mlflow: bool,
    logger: logging.Logger,
    services: TrainingLoopServices,
) -> TrainingState:
    collect_prediction_rate_diagnostics = services.collect_prediction_rate_diagnostics
    evaluate_multilabel = services.evaluate_multilabel
    format_lr_summary = services.format_lr_summary
    format_pair_metric_summary = services.format_pair_metric_summary
    format_per_label_summary = services.format_per_label_summary
    format_prediction_rate_summary = services.format_prediction_rate_summary
    get_lr_diagnostics = services.get_lr_diagnostics
    get_validation_monitor_value = services.get_validation_monitor_value
    resolve_monitored_label_pairs = services.resolve_monitored_label_pairs
    run_epoch = services.run_epoch
    sanitize_name = services.sanitize_name
    save_checkpoint = services.save_checkpoint
    search_optimal_thresholds = services.search_optimal_thresholds
    mlflow = services.mlflow_api
    model = components.model
    criterion = components.criterion
    optimizer = components.optimizer
    rank = int(os.environ.get("RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    scheduler = components.scheduler
    writer = components.writer
    stage1_name = str(config.get("stage1_name", "last_separation")).strip().lower()
    stage2_name = str(config.get("stage2_name", "adversarial_joint")).strip().lower()
    stage1_end_epoch = max(0, int(config.get("stage1_epochs", config.get("stage_a_epochs", 0))))
    stage1_min_epochs = max(1, int(config.get("stage1_min_epochs", min(stage1_end_epoch if stage1_end_epoch > 0 else 1, 2))))
    stage2_grl_lambda = float(config.get("grl_lambda", 1.0))
    stage1_grl_lambda = float(config.get("stage1_grl_lambda", 0.0))
    stage_transition_metric = str(config.get("stage_transition_metric", "val_contrib_last_task_context_cka_offdiag_mean")).strip().lower()
    stage_transition_cka_max = float(config.get("stage_transition_cka_max", 0.65))
    stage_transition_attention_entropy_max = float(config.get("stage_transition_attention_entropy_max", 3.2))
    dynamic_saturation_patience = max(1, int(config.get("dynamic_scale_saturation_patience", 3)))
    dynamic_restart_mix = float(config.get("dynamic_scale_restart_mix", 0.5))
    foreground_warmup_epochs = int(getattr(components, "foreground_warmup_epochs", 0))
    hard_freeze_backbone_epochs = max(0, int(config.get("hard_freeze_backbone_epochs", config.get("backbone_freeze_epochs", 0))))
    freeze_backbone_during_stage1 = bool(config.get("freeze_backbone_during_stage1", False))
    backbone_trainable_state: Optional[bool] = None
    monitored_pairs = resolve_monitored_label_pairs(config.get("cooccurrence_monitor_pairs", []), ptm_types)
    if hard_freeze_backbone_epochs > 0 or freeze_backbone_during_stage1:
        logger.info(
            "Hard backbone freeze enabled: hard_freeze_backbone_epochs=%d freeze_backbone_during_stage1=%s",
            hard_freeze_backbone_epochs,
            str(freeze_backbone_during_stage1),
        )

    def _log_last_status(tag: str) -> None:
        base = getattr(model, "base_model", model)
        use_last = bool(getattr(base, "use_last_foreground", False))
        last_block = getattr(base, "last_foreground_block", None)
        last_proj = getattr(base, "last_task_context_proj", None)
        if not use_last:
            logger.warning("LaST status (%s): use_last_foreground disabled", tag)
            return
        if last_block is None:
            logger.warning("LaST status (%s): last_foreground_block missing", tag)
            return
        block_trainable = any(p.requires_grad for p in last_block.parameters())
        proj_trainable = None
        if last_proj is not None:
            proj_trainable = any(p.requires_grad for p in last_proj.parameters())
        logger.info("LaST status (%s): block_trainable=%s context_proj_trainable=%s", tag, block_trainable, proj_trainable)

    _log_last_status("init")

    # V3: UncertaintyTrajectoryAnalyzer wiring.
    uta = None
    if bool(config.get("use_uncertainty_memory_bank", False)):
        from src.utils.uncertainty_guided_memory import (
            UncertaintyTrajectoryAnalyzer,
            build_replay_dataloader,
        )

        uta = UncertaintyTrajectoryAnalyzer(
            num_tasks=len(ptm_types),
            memory_size_per_task=int(config.get("memory_size_per_task", 200)),
            epoch_threshold=int(config.get("memory_epoch_threshold", 5)),
            entropy_normalization=str(config.get("memory_entropy_normalization", "l2")),
            feature_dim=int(getattr(getattr(model, "cosine_classifier", None), "feature_dim", 512)),
        )
        uta_state_path = components.ckpt_last.parent / "uta_memory.pt"
        if state.start_epoch > 1 and uta_state_path.exists():
            uta.load(uta_state_path)
            logger.info("Restored UTA memory state from %s", str(uta_state_path))
        try:
            model._uta = uta
        except Exception:
            pass
        replay_ratio = max(0.0, min(0.9, float(config.get("exemplar_replay_ratio", 0.1))))
        if replay_ratio > 0.0 and not bool(
            getattr(loaders.train_loader.dataset, "packed_batch_dataset", False)
        ):
            loaders.train_loader = build_replay_dataloader(
                loader=loaders.train_loader,
                memory_bank=uta,
                replay_ratio=replay_ratio,
            )
            logger.info("V3 UTA exemplar replay enabled (replay_ratio=%.3f)", replay_ratio)
        elif replay_ratio > 0.0:
            logger.warning(
                "V3 UTA exemplar replay requested but dataset %s does not support sample-index replay; replay skipped.",
                type(loaders.train_loader.dataset).__name__,
            )
        logger.info(
            "V3: UncertaintyTrajectoryAnalyzer enabled (memory_size_per_task=%d, epoch_threshold=%d)",
            int(config.get("memory_size_per_task", 200)),
            int(config.get("memory_epoch_threshold", 5)),
        )

    for epoch in range(state.start_epoch, components.num_epochs + 1):
        if hasattr(criterion, "current_epoch"):
            criterion.current_epoch = epoch
        if hasattr(criterion, "max_epochs"):
            criterion.max_epochs = components.num_epochs
        if uta is not None:
            try:
                model._uta_epoch = int(epoch)
            except Exception:
                pass

        current_stage, backbone_trainable_state = apply_epoch_training_stage(
            model=model,
            state=state,
            epoch=epoch,
            stage1_name=stage1_name,
            stage2_name=stage2_name,
            stage1_end_epoch=stage1_end_epoch,
            stage1_grl_lambda=stage1_grl_lambda,
            stage2_grl_lambda=stage2_grl_lambda,
            foreground_warmup_epochs=foreground_warmup_epochs,
            hard_freeze_backbone_epochs=hard_freeze_backbone_epochs,
            freeze_backbone_during_stage1=freeze_backbone_during_stage1,
            backbone_trainable_state=backbone_trainable_state,
            logger=logger,
            log_last_status=_log_last_status,
        )

        if loaders.train_batch_sampler is not None and hasattr(loaders.train_batch_sampler, "set_epoch"):
            loaders.train_batch_sampler.set_epoch(epoch)
        train_loss, y_train, p_train, train_struct_avail, train_perf = run_epoch(
            model=model,
            loader=loaders.train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=runtime_amp.device,
            options=EpochOptions(
                grad_accum_steps=components.grad_accum_steps,
                max_batches=components.sanity_max_train_batches,
                grad_norm_log_interval=components.grad_norm_log_interval,
                grad_norm_topk=components.grad_norm_topk,
                grad_norm_warn_threshold=components.grad_norm_warn_threshold,
                scaler=runtime_amp.scaler,
                use_amp=runtime_amp.use_amp,
                amp_dtype=runtime_amp.amp_dtype,
                distribution_monitor=loaders.train_distribution_monitor,
                stage_name="train",
                grad_clip_norm=components.grad_clip_norm,
                log_interval=components.log_every,
                scheduler_step_unit=components.scheduler_step_unit,
                gradient_conflict_log_interval=components.gradient_conflict_log_interval,
                teacher_distill_weight=float(config.get("teacher_distill_weight", 0.0)),
                teacher_distill_temperature=float(config.get("teacher_distill_temperature", 1.0)),
                teacher_distill_min_confidence=float(config.get("teacher_distill_min_confidence", 0.0)),
                teacher_distill_unknown_only=bool(config.get("teacher_distill_unknown_only", False)),
            ),
        )
        train_perf["epoch"] = epoch
        state.epoch_performance.append(train_perf)
        loaders.train_distribution_monitor.flush_epoch_summary(logger, stage="train", epoch=epoch)
        train_eval_thresholds = np.full((len(ptm_types),), 0.5, dtype=np.float32)
        train_rate_diag = collect_prediction_rate_diagnostics(
            y_train,
            p_train,
            ptm_types,
            thresholds=train_eval_thresholds,
        )
        should_validate = (epoch % components.validate_every_n_epochs == 0) or (epoch == components.num_epochs)
        if should_validate:
            val_loss, y_val, p_val, val_struct_avail, val_perf = run_epoch(
                model=model,
                loader=loaders.val_loader,
                optimizer=None,
                scheduler=None,
                criterion=criterion,
                device=runtime_amp.device,
                options=EpochOptions(
                    max_batches=components.sanity_max_val_batches,
                    use_amp=runtime_amp.use_amp,
                    amp_dtype=runtime_amp.amp_dtype,
                    stage_name="val",
                    grad_clip_norm=components.grad_clip_norm,
                    log_interval=components.log_every,
                ),
            )
            val_perf["epoch"] = epoch
            state.epoch_performance.append(val_perf)

            if y_val.shape[0] > 0:
                fixed_val_thresholds = np.full((len(ptm_types),), 0.5, dtype=np.float32)
                fixed_val_metrics = evaluate_multilabel(
                    y_val,
                    p_val,
                    ptm_types,
                    thresholds=fixed_val_thresholds,
                    monitored_pairs=monitored_pairs,
                )
                val_macro_f1 = float(fixed_val_metrics["macro_f1"])
                val_macro_auc = float(fixed_val_metrics.get("macro_roc_auc", float("nan")))
                if bool(config.get("search_validation_thresholds_each_epoch", False)):
                    val_thresholds, _ = search_optimal_thresholds(
                        y_true=y_val,
                        y_prob=p_val,
                        ptm_types=ptm_types,
                        metric=components.threshold_metric,
                        max_candidates=int(config.get("validation_threshold_max_candidates", 200)),
                    )
                    val_metrics = evaluate_multilabel(
                        y_val, p_val, ptm_types, thresholds=val_thresholds,
                        monitored_pairs=monitored_pairs,
                    )
                    val_macro_f1_opt = float(val_metrics["macro_f1"])
                    logger.info(
                        "Epoch %d validation threshold search | macro_f1@0.5=%.5f macro_f1_opt=%.5f thresholds=%s",
                        epoch, val_macro_f1, val_macro_f1_opt,
                        ",".join(f"{x:.4f}" for x in val_thresholds.tolist()),
                    )
                else:
                    val_thresholds = fixed_val_thresholds
                    val_metrics = fixed_val_metrics
                    val_macro_f1_opt = val_macro_f1
            else:
                val_thresholds = state.best_thresholds
                val_metrics = {}
                val_macro_f1 = float("nan")
                val_macro_auc = float("nan")
                val_macro_f1_opt = float("nan")
            val_rate_diag = collect_prediction_rate_diagnostics(
                y_val,
                p_val,
                ptm_types,
                thresholds=val_thresholds,
            )
        else:
            val_loss = float("nan")
            val_thresholds = state.best_thresholds
            val_metrics = {}
            val_macro_f1 = float("nan")
            val_macro_auc = float("nan")
            val_macro_f1_opt = float("nan")
            val_rate_diag = None
            val_struct_avail = np.zeros((0,), dtype=np.float32)
            val_perf = {
                "stage": "val_skipped",
                "epoch": epoch,
                "elapsed_time_sec": 0.0,
                "sample_count": 0,
                "batch_count": 0,
                "optimizer_steps": 0,
                "samples_per_sec": 0.0,
                "max_gpu_memory_mb": None,
            }

        state.history["train_loss"].append(float(train_loss))
        state.history["val_loss"].append(float(val_loss))
        state.history["val_macro_f1"].append(val_macro_f1)
        state.history["val_macro_auc"].append(val_macro_auc)
        if isinstance(val_metrics, dict) and isinstance(val_metrics.get("per_label"), dict):
            for label_name, label_metrics in val_metrics["per_label"].items():
                if not isinstance(label_metrics, dict):
                    continue
                if "precision" in label_metrics:
                    state.history.setdefault(f"val_precision_{sanitize_name(label_name)}", []).append(float(label_metrics["precision"]))
                if "recall" in label_metrics:
                    state.history.setdefault(f"val_recall_{sanitize_name(label_name)}", []).append(float(label_metrics["recall"]))
                if "f1" in label_metrics:
                    state.history.setdefault(f"val_f1_{sanitize_name(label_name)}", []).append(float(label_metrics["f1"]))
        if isinstance(val_metrics, dict) and isinstance(val_metrics.get("pair_metrics"), dict):
            for pair_name, pair_metrics in val_metrics["pair_metrics"].items():
                if not isinstance(pair_metrics, dict):
                    continue
                if "joint_f1" in pair_metrics:
                    state.history.setdefault(f"val_pair_f1_{sanitize_name(pair_name)}", []).append(float(pair_metrics["joint_f1"]))
                if "joint_recall" in pair_metrics:
                    state.history.setdefault(f"val_pair_recall_{sanitize_name(pair_name)}", []).append(float(pair_metrics["joint_recall"]))
        state.history.setdefault("train_epoch_time_sec", []).append(float(train_perf["elapsed_time_sec"]))
        state.history.setdefault("val_epoch_time_sec", []).append(float(val_perf["elapsed_time_sec"]))
        state.history.setdefault("train_samples_per_sec", []).append(float(train_perf["samples_per_sec"]))
        state.history.setdefault("train_max_gpu_memory_mb", []).append(
            float(train_perf["max_gpu_memory_mb"]) if train_perf["max_gpu_memory_mb"] is not None else float("nan")
        )
        state.history.setdefault("train_non_finite_total_skips", []).append(float(train_perf.get("non_finite_total_skips", 0.0)))
        state.history.setdefault("train_non_finite_loss_batches", []).append(float(train_perf.get("non_finite_loss_batches", 0.0)))
        state.history.setdefault("train_non_finite_grad_skip_steps", []).append(float(train_perf.get("non_finite_grad_skip_steps", 0.0)))
        train_aux_logs = train_perf.get("auxiliary_logs") if isinstance(train_perf, dict) else None
        if isinstance(train_aux_logs, dict):
            for key, value in train_aux_logs.items():
                state.history.setdefault(f"train_aux_{key}", []).append(float(value))
        train_dyn_shared = float(train_aux_logs.get("dynamic_shared_loss_scale", float("nan"))) if isinstance(train_aux_logs, dict) else float("nan")
        train_dyn_private = float(train_aux_logs.get("dynamic_private_loss_scale", float("nan"))) if isinstance(train_aux_logs, dict) else float("nan")
        dyn_min = float(getattr(model, "dynamic_loss_scale_min", config.get("dynamic_loss_scale_min", 0.5)))
        dyn_max = float(getattr(model, "dynamic_loss_scale_max", config.get("dynamic_loss_scale_max", 2.0)))
        near_sat = (
            np.isfinite(train_dyn_shared)
            and np.isfinite(train_dyn_private)
            and (
                abs(train_dyn_shared - dyn_min) < 1e-4
                or abs(train_dyn_shared - dyn_max) < 1e-4
                or abs(train_dyn_private - dyn_min) < 1e-4
                or abs(train_dyn_private - dyn_max) < 1e-4
            )
        )
        state.dynamic_scale_saturation_streak = state.dynamic_scale_saturation_streak + 1 if near_sat else 0
        if state.dynamic_scale_saturation_streak >= dynamic_saturation_patience and hasattr(model, "set_dynamic_loss_scales"):
            mixed_shared = (1.0 - dynamic_restart_mix) * train_dyn_shared + dynamic_restart_mix * 1.0
            mixed_private = (1.0 - dynamic_restart_mix) * train_dyn_private + dynamic_restart_mix * 1.0
            model.set_dynamic_loss_scales(mixed_shared, mixed_private)
            logger.info(
                "Dynamic scale restart triggered at epoch %d | old_shared=%.4f old_private=%.4f new_shared=%.4f new_private=%.4f",
                epoch,
                train_dyn_shared,
                train_dyn_private,
                mixed_shared,
                mixed_private,
            )
            state.dynamic_scale_saturation_streak = 0
        train_contrib = train_perf.get("module_contributions") if isinstance(train_perf, dict) else None
        if isinstance(train_contrib, dict):
            for key, value in train_contrib.items():
                arr = np.asarray(value, dtype=np.float64)
                if arr.size > 0 and np.isfinite(arr).any():
                    state.history.setdefault(f"train_contrib_{key}", []).append(float(np.nanmean(arr)))
                    if key in {
                        "cosine_logit_abs_mean",
                        "cosine_logit_gate_per_task",
                        "contrastive_loss_raw",
                        "contrastive_loss_weighted",
                        "contrastive_valid_anchor_count",
                        "contrastive_positive_pair_count",
                        "contrastive_candidate_pair_count",
                    }:
                        scalar = float(np.nanmean(arr))
                        if writer is not None:
                            writer.add_scalar(f"v3/train_{key}", scalar, epoch)
                        if use_mlflow and mlflow is not None:
                            mlflow.log_metric(f"v3/train_{key}", scalar, step=epoch)
        val_contrib_hist = val_perf.get("module_contributions") if isinstance(val_perf, dict) else None
        if isinstance(val_contrib_hist, dict):
            for key, value in val_contrib_hist.items():
                arr = np.asarray(value, dtype=np.float64)
                if arr.size > 0 and np.isfinite(arr).any():
                    state.history.setdefault(f"val_contrib_{key}", []).append(float(np.nanmean(arr)))

        if use_mlflow:
            mlflow.log_metric("train_stage_transition_epoch", float(state.stage_transition_epoch), step=epoch)
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            if should_validate:
                mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                if not np.isnan(val_macro_f1):
                    mlflow.log_metric("val_macro_f1", float(val_macro_f1), step=epoch)
                if not np.isnan(val_macro_auc):
                    mlflow.log_metric("val_macro_auc", float(val_macro_auc), step=epoch)
            mlflow.log_metric("train_samples_per_sec", float(train_perf["samples_per_sec"]), step=epoch)
            mlflow.log_metric("train_non_finite_total_skips", float(train_perf.get("non_finite_total_skips", 0.0)), step=epoch)
            if isinstance(train_aux_logs, dict):
                for key, value in train_aux_logs.items():
                    mlflow.log_metric(f"train_aux/{key}", float(value), step=epoch)
            if isinstance(val_metrics, dict) and isinstance(val_metrics.get("per_label"), dict):
                for label_name, label_metrics in val_metrics["per_label"].items():
                    if isinstance(label_metrics, dict) and "precision" in label_metrics:
                        mlflow.log_metric(f"weak_task_precision/{sanitize_name(label_name)}", float(label_metrics["precision"]), step=epoch)
                    if isinstance(label_metrics, dict) and "recall" in label_metrics:
                        mlflow.log_metric(f"weak_task_recall/{sanitize_name(label_name)}", float(label_metrics["recall"]), step=epoch)
                    if isinstance(label_metrics, dict) and "f1" in label_metrics:
                        mlflow.log_metric(f"weak_task_f1/{sanitize_name(label_name)}", float(label_metrics["f1"]), step=epoch)
            if isinstance(val_metrics, dict) and isinstance(val_metrics.get("pair_metrics"), dict):
                for pair_name, pair_metrics in val_metrics["pair_metrics"].items():
                    if not isinstance(pair_metrics, dict):
                        continue
                    if "joint_f1" in pair_metrics:
                        mlflow.log_metric(f"cooccurrence_f1/{sanitize_name(pair_name)}", float(pair_metrics["joint_f1"]), step=epoch)
                    if "joint_recall" in pair_metrics:
                        mlflow.log_metric(f"cooccurrence_recall/{sanitize_name(pair_name)}", float(pair_metrics["joint_recall"]), step=epoch)
                    if "pred_lift" in pair_metrics:
                        mlflow.log_metric(f"cooccurrence_pred_lift/{sanitize_name(pair_name)}", float(pair_metrics["pred_lift"]), step=epoch)

        if scheduler is not None and components.scheduler_step_unit == "epoch":
            scheduler.step()

        lr_diag = get_lr_diagnostics(optimizer, scheduler)
        overfit_gap = float(val_loss - train_loss) if not np.isnan(val_loss) else float("nan")
        if (not np.isnan(overfit_gap)) and overfit_gap > components.overfit_gap_threshold:
            logger.warning("Potential overfitting detected at epoch %d: val_loss-train_loss=%.5f", epoch, overfit_gap)

        if should_validate:
            target_metric_key = stage_transition_metric.replace("val_contrib_", "")
            val_cka_offdiag = float(np.nanmean(np.asarray(val_perf.get("module_contributions", {}).get(target_metric_key, [float("nan")]), dtype=np.float64))) if isinstance(val_perf, dict) else float("nan")
            val_attn_entropy = float(np.nanmean(np.asarray(val_perf.get("module_contributions", {}).get("ptm_attention_entropy_per_task", [float("nan")]), dtype=np.float64))) if isinstance(val_perf, dict) else float("nan")
            if current_stage == stage1_name and np.isfinite(val_cka_offdiag) and np.isfinite(val_attn_entropy):
                transition_ready = (val_cka_offdiag <= stage_transition_cka_max) and (val_attn_entropy <= stage_transition_attention_entropy_max)
                logger.info(
                    "Stage1 gate check | metric=%s cka_offdiag=%.4f entropy=%.4f ready=%s",
                    stage_transition_metric,
                    val_cka_offdiag,
                    val_attn_entropy,
                    str(bool(transition_ready)),
                )
                if transition_ready and epoch >= stage1_min_epochs and epoch < stage1_end_epoch:
                    stage1_end_epoch = epoch
                    logger.info("Stage transition condition satisfied at epoch %d; stage1_end_epoch updated to %d", epoch, stage1_end_epoch)
                    if components.save_checkpoints:
                        transition_ckpt = components.ckpt_best.parent / f"checkpoint_stage1_epoch_{epoch}.pt"
                        save_checkpoint(
                            transition_ckpt,
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            state.best_monitor_value,
                            state.early_stopping_metric,
                            state.best_checkpoint_val_macro_f1,
                            state.best_checkpoint_val_macro_auc,
                            state.best_thresholds,
                            criterion=criterion,
                            scaler=runtime_amp.scaler,
                        )
            logger.info(
                "Epoch %d/%d | train_loss=%.5f val_loss=%.5f val_macro_f1@0.5=%.5f val_macro_auc=%.5f train_time=%.2fs train_samples_per_sec=%.2f",
                epoch,
                components.num_epochs,
                train_loss,
                val_loss,
                val_macro_f1,
                val_macro_auc,
                train_perf["elapsed_time_sec"],
                train_perf["samples_per_sec"],
            )
            logger.info(
                "Epoch %d lr | step_unit=%s %s",
                epoch,
                components.scheduler_step_unit,
                format_lr_summary(lr_diag),
            )
            logger.info(
                "Epoch %d train label-mix | sampled_all_zero=%.4f predicted_all_zero@0.5=%.4f avg_true_labels=%.4f avg_pred_labels@0.5=%.4f",
                epoch,
                float(train_rate_diag.get("sampled_all_zero_rate", float("nan"))),
                float(train_rate_diag.get("predicted_all_zero_rate", float("nan"))),
                float(train_rate_diag.get("avg_true_positive_labels", float("nan"))),
                float(train_rate_diag.get("avg_pred_positive_labels", float("nan"))),
            )
            logger.info("Epoch %d train prediction-rates | %s", epoch, format_prediction_rate_summary(train_rate_diag, ptm_types))
            logger.info(
                "Epoch %d val label-mix | sampled_all_zero=%.4f predicted_all_zero=%.4f avg_true_labels=%.4f avg_pred_labels=%.4f",
                epoch,
                float(val_rate_diag.get("sampled_all_zero_rate", float("nan"))) if isinstance(val_rate_diag, dict) else float("nan"),
                float(val_rate_diag.get("predicted_all_zero_rate", float("nan"))) if isinstance(val_rate_diag, dict) else float("nan"),
                float(val_rate_diag.get("avg_true_positive_labels", float("nan"))) if isinstance(val_rate_diag, dict) else float("nan"),
                float(val_rate_diag.get("avg_pred_positive_labels", float("nan"))) if isinstance(val_rate_diag, dict) else float("nan"),
            )
            if isinstance(val_rate_diag, dict):
                logger.info("Epoch %d val prediction-rates | %s", epoch, format_prediction_rate_summary(val_rate_diag, ptm_types))
            per_label_summary = format_per_label_summary(val_metrics.get("per_label", {})) if isinstance(val_metrics, dict) else ""
            if per_label_summary:
                logger.info("Epoch %d per-task | %s", epoch, per_label_summary)
            pair_summary = format_pair_metric_summary(val_metrics.get("pair_metrics", {})) if isinstance(val_metrics, dict) else ""
            if pair_summary:
                logger.info("Epoch %d cooccurrence | %s", epoch, pair_summary)
        else:
            logger.info(
                "Epoch %d/%d | train_loss=%.5f val=skipped train_time=%.2fs train_samples_per_sec=%.2f",
                epoch,
                components.num_epochs,
                train_loss,
                train_perf["elapsed_time_sec"],
                train_perf["samples_per_sec"],
            )
            logger.info(
                "Epoch %d lr | step_unit=%s %s",
                epoch,
                components.scheduler_step_unit,
                format_lr_summary(lr_diag),
            )
            logger.info(
                "Epoch %d train label-mix | sampled_all_zero=%.4f predicted_all_zero@0.5=%.4f avg_true_labels=%.4f avg_pred_labels@0.5=%.4f",
                epoch,
                float(train_rate_diag.get("sampled_all_zero_rate", float("nan"))),
                float(train_rate_diag.get("predicted_all_zero_rate", float("nan"))),
                float(train_rate_diag.get("avg_true_positive_labels", float("nan"))),
                float(train_rate_diag.get("avg_pred_positive_labels", float("nan"))),
            )
            logger.info("Epoch %d train prediction-rates | %s", epoch, format_prediction_rate_summary(train_rate_diag, ptm_types))
        logger.info(
            "Epoch %d stability | train_non_finite_total=%d (loss=%d grad=%d)",
            epoch,
            int(train_perf.get("non_finite_total_skips", 0)),
            int(train_perf.get("non_finite_loss_batches", 0)),
            int(train_perf.get("non_finite_grad_skip_steps", 0)),
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            if not np.isnan(val_macro_f1):
                writer.add_scalar("metrics/val_macro_f1", val_macro_f1, epoch)
            if not np.isnan(val_macro_auc):
                writer.add_scalar("metrics/val_macro_auc", val_macro_auc, epoch)
            if isinstance(val_metrics, dict) and isinstance(val_metrics.get("per_label"), dict):
                for label_name, label_metrics in val_metrics["per_label"].items():
                    if not isinstance(label_metrics, dict):
                        continue
                    key = sanitize_name(label_name)
                    if "precision" in label_metrics:
                        writer.add_scalar(f"metrics_precision/{key}", float(label_metrics["precision"]), epoch)
                    if "recall" in label_metrics:
                        writer.add_scalar(f"metrics_recall/{key}", float(label_metrics["recall"]), epoch)
                    if "f1" in label_metrics:
                        writer.add_scalar(f"metrics_f1/{key}", float(label_metrics["f1"]), epoch)
            if isinstance(val_metrics, dict) and isinstance(val_metrics.get("pair_metrics"), dict):
                for pair_name, pair_metrics in val_metrics["pair_metrics"].items():
                    if not isinstance(pair_metrics, dict):
                        continue
                    key = sanitize_name(pair_name)
                    if "joint_f1" in pair_metrics:
                        writer.add_scalar(f"cooccurrence_f1/{key}", float(pair_metrics["joint_f1"]), epoch)
                    if "joint_recall" in pair_metrics:
                        writer.add_scalar(f"cooccurrence_recall/{key}", float(pair_metrics["joint_recall"]), epoch)
                    if "pred_lift" in pair_metrics:
                        writer.add_scalar(f"cooccurrence_pred_lift/{key}", float(pair_metrics["pred_lift"]), epoch)
            writer.add_scalar("performance/train_epoch_time_sec", train_perf["elapsed_time_sec"], epoch)
            writer.add_scalar("performance/train_samples_per_sec", train_perf["samples_per_sec"], epoch)
            writer.add_scalar("training/stage_transition_epoch", float(state.stage_transition_epoch), epoch)
            writer.add_scalar("training/dynamic_scale_saturation_streak", float(state.dynamic_scale_saturation_streak), epoch)
            writer.add_scalar("stability/train_non_finite_total_skips", float(train_perf.get("non_finite_total_skips", 0.0)), epoch)
            writer.add_scalar("stability/train_non_finite_loss_batches", float(train_perf.get("non_finite_loss_batches", 0.0)), epoch)
            writer.add_scalar("stability/train_non_finite_grad_skip_steps", float(train_perf.get("non_finite_grad_skip_steps", 0.0)), epoch)
            if train_perf["max_gpu_memory_mb"] is not None:
                writer.add_scalar("performance/train_max_gpu_memory_mb", train_perf["max_gpu_memory_mb"], epoch)
            if isinstance(train_aux_logs, dict):
                for key, value in train_aux_logs.items():
                    writer.add_scalar(f"auxiliary/train_{key}", float(value), epoch)
            train_contrib = train_perf.get("module_contributions")
            if isinstance(train_contrib, dict):
                for key in [
                    "last_stability_base_peak",
                    "last_mask_center_to_bg_ratio",
                    "last_mask_mean_per_task",
                    "last_graph_score_peak",
                    "last_graph_score_mean",
                    "last_graph_score_sparsity",
                    "last_task_context_cka_matrix",
                    "last_task_context_cka_offdiag_mean",
                    "private_mmd_matrix",
                    "private_mmd_offdiag_mean",
                    "private_feature_utilization_mean",
                    "ptm_attention_peak_per_task",
                    "ptm_attention_entropy_per_task",
                    "ptm_attention_focus_ratio_per_task",
                    "task_prompt_cosine_offdiag_mean",
                    "task_mutual_information_proxy",
                ]:
                    val = train_contrib.get(key)
                    if isinstance(val, list):
                        if val:
                            writer.add_scalar(f"last/train_{key}_mean", float(np.mean(val)), epoch)
                    elif isinstance(val, (float, int, np.floating, np.integer)):
                        writer.add_scalar(f"last/train_{key}", float(val), epoch)
            val_contrib = val_perf.get("module_contributions") if should_validate and isinstance(val_perf, dict) else None
            if isinstance(val_contrib, dict):
                for key in [
                    "last_stability_base_peak",
                    "last_mask_center_to_bg_ratio",
                    "last_mask_mean_per_task",
                    "last_graph_score_peak",
                    "last_graph_score_mean",
                    "last_graph_score_sparsity",
                    "last_task_context_cka_matrix",
                    "last_task_context_cka_offdiag_mean",
                    "private_mmd_matrix",
                    "private_mmd_offdiag_mean",
                    "private_feature_utilization_mean",
                    "ptm_attention_peak_per_task",
                    "ptm_attention_entropy_per_task",
                    "ptm_attention_focus_ratio_per_task",
                    "task_prompt_cosine_offdiag_mean",
                    "task_mutual_information_proxy",
                ]:
                    val = val_contrib.get(key)
                    if isinstance(val, list):
                        if val:
                            writer.add_scalar(f"last/val_{key}_mean", float(np.mean(val)), epoch)
                    elif isinstance(val, (float, int, np.floating, np.integer)):
                        writer.add_scalar(f"last/val_{key}", float(val), epoch)

        if should_validate and (not np.isnan(val_macro_f1)):
            state.best_val_macro_f1 = max(state.best_val_macro_f1, val_macro_f1)
        if should_validate and (not np.isnan(val_macro_auc)):
            state.best_val_macro_auc = max(state.best_val_macro_auc, val_macro_auc)

        monitor_value = get_validation_monitor_value(
            state.early_stopping_metric, float(val_loss), val_macro_f1,
            val_macro_auc, val_macro_f1_opt,
        )
        if state.early_stopping_mode == "min":
            improved = should_validate and (not np.isnan(monitor_value)) and (monitor_value < state.best_monitor_value)
        else:
            improved = should_validate and (not np.isnan(monitor_value)) and (monitor_value > state.best_monitor_value)
        if improved:
            state.best_monitor_value = monitor_value
            state.best_checkpoint_val_macro_f1 = val_macro_f1
            state.best_checkpoint_val_macro_auc = val_macro_auc
            state.best_thresholds = val_thresholds

        if uta is not None:
            # Cumulative-entropy exemplars refine prototypes with the same EMA
            # momentum used by minibatch updates; they never replace prototypes.
            try:
                stats = uta.on_epoch_end()
                if stats:
                    logger.info(
                        "V3 UTA epoch %d: tracked=%d avg_cum_entropy=%.4f",
                        epoch,
                        int(stats.get("num_tracked_samples", 0)),
                        float(stats.get("avg_cumulative_entropy", 0.0)),
                    )
                selected = uta.select_exemplars_per_epoch()
                uta.update_memory_bank(
                    selected,
                    model,
                    loaders.train_loader.dataset,
                    collate_fn=loaders.train_loader.collate_fn,
                    batch_size=int(config.get("uta_feature_batch_size", 64)),
                    device=runtime_amp.device,
                )
                # The dynamic classifier is owned by the wrapper model. Do not
                # inspect only ``base_model``: its classifier is Identity here.
                cosine = None
                owner = model
                if hasattr(owner, "get_cosine_classifier"):
                    cosine = owner.get_cosine_classifier()
                if cosine is None:
                    cosine = getattr(owner, "cosine_classifier", None)
                if cosine is None:
                    nested = getattr(owner, "base_model", None)
                    if nested is not None:
                        cosine = getattr(nested, "cosine_classifier", None)
                if cosine is not None:
                    feats, labs = uta.get_exemplar_features_labels()
                    if feats.numel() > 0 and feats.shape[1] == int(cosine.feature_dim):
                        min_exemplars = int(config.get("cosine_uta_min_positive_exemplars", 2))
                        counts = cosine.update_prototypes_from_exemplars(
                            feats,
                            labs,
                            momentum=float(config.get("cosine_uta_momentum", cosine.momentum)),
                            min_positive_samples=max(1, min_exemplars),
                        )
                        logger.info(
                            "V3 UTA: momentum-refined cosine prototypes from %d exemplars across %d tasks",
                            int(labs.shape[0]),
                            int((counts > 0).sum().item()),
                        )
                else:
                    logger.warning(
                        "V3 UTA: cosine classifier not found on model or base_model; "
                        "prototype refresh skipped (model_type=%s)", type(model).__name__,
                    )
            except Exception as exc:  # never let the memory bank break training
                logger.warning("V3 UTA update failed at epoch %d: %s", epoch, exc)

        if components.save_checkpoints and (not distributed or rank == 0):
            if improved:
                save_checkpoint(
                    components.ckpt_best,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    state.best_monitor_value,
                    state.early_stopping_metric,
                    state.best_checkpoint_val_macro_f1,
                    state.best_checkpoint_val_macro_auc,
                    state.best_thresholds,
                    criterion=criterion,
                    scaler=runtime_amp.scaler,
                )
                torch.save((model.module if hasattr(model, "module") else model).state_dict(), components.best_model_path)
            if (epoch % components.checkpoint_interval_epochs == 0) or (epoch == components.num_epochs):
                save_checkpoint(
                    components.ckpt_last,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    state.best_monitor_value,
                    state.early_stopping_metric,
                    state.best_checkpoint_val_macro_f1,
                    state.best_checkpoint_val_macro_auc,
                    state.best_thresholds,
                    criterion=criterion,
                    scaler=runtime_amp.scaler,
                )
            if uta is not None and (
                improved
                or (epoch % components.checkpoint_interval_epochs == 0)
                or (epoch == components.num_epochs)
            ):
                uta.save(components.ckpt_last.parent / "uta_memory.pt")

        should_stop = False
        if components.early_stopping is not None and should_validate and (not distributed or rank == 0):
            _, should_stop = components.early_stopping.step(monitor_value)
            if should_stop:
                logger.info("Early stopping triggered at epoch %d based on %s=%.5f", epoch, state.early_stopping_metric, monitor_value)
        if distributed:
            should_stop_obj = [bool(should_stop) if rank == 0 else False]
            torch.distributed.broadcast_object_list(should_stop_obj, src=0)
            should_stop = bool(should_stop_obj[0])
        if should_stop:
            break

        if distributed:
            for param in criterion.parameters():
                torch.distributed.broadcast(param.data, src=0)
        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")
        state.summary_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": float(val_macro_auc),
                "val_f1": float(val_macro_f1),
                "lr": current_lr,
                "elapsed_time_sec": float(train_perf["elapsed_time_sec"]),
                "train_non_finite_total_skips": int(train_perf.get("non_finite_total_skips", 0)),
            }
        )

    return state
