"""Post-training evaluation helpers extracted from the training entrypoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.utils.training_state import EpochOptions

FloatArray = NDArray[np.float32]


def _resolve_monitored_pairs(config: Dict[str, Any], ptm_types: List[str], sanitize_name_fn: Callable[[str], str]) -> List[Tuple[str, str]]:
    raw_pairs = config.get("cooccurrence_monitor_pairs", [])
    if not isinstance(raw_pairs, list):
        return []
    valid = {sanitize_name_fn(name): str(name) for name in ptm_types}
    resolved: List[Tuple[str, str]] = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left_key = sanitize_name_fn(str(pair[0]))
        right_key = sanitize_name_fn(str(pair[1]))
        if left_key == right_key or left_key not in valid or right_key not in valid:
            continue
        canonical = (valid[left_key], valid[right_key])
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def run_post_training_evaluation(
    *,
    args: Any,
    config: Dict[str, Any],
    ptm_types: List[str],
    val_loader: Any,
    test_loader: Any,
    criterion: Any,
    runtime_amp: Any,
    components: Any,
    state: Any,
    model: Any,
    logger: Any,
    output_dir: Path,
    run_epoch_fn: Callable[..., Tuple[float, FloatArray, FloatArray, FloatArray, Dict[str, Any]]],
    load_checkpoint_fn: Callable[..., Tuple[Any, ...]],
    search_optimal_thresholds_fn: Callable[..., Tuple[FloatArray, Dict[str, Any]]],
    evaluate_multilabel_fn: Callable[..., Dict[str, Any]],
    export_metrics_fn: Callable[[Path, Dict[str, Any]], None],
    export_structure_effect_significance_fn: Callable[..., None],
    sanitize_name_fn: Callable[[str], str],
) -> Dict[str, Any]:
    monitored_pairs = _resolve_monitored_pairs(config, ptm_types, sanitize_name_fn)
    eval_checkpoint = str(config.get("eval_checkpoint", "best")).lower().strip()
    if args.eval_checkpoint:
        eval_checkpoint = str(args.eval_checkpoint).lower().strip()
    eval_ckpt_path = components.ckpt_best if eval_checkpoint == "best" else components.ckpt_last
    eval_loaded_epoch: Optional[int] = None
    if components.save_checkpoints and eval_ckpt_path.exists():
        loaded_epoch, loaded_best_monitor, loaded_best_f1, loaded_best_auc, loaded_monitor_metric, thr = load_checkpoint_fn(
            eval_ckpt_path, model, optimizer=None, scheduler=None, scaler=None
        )
        eval_loaded_epoch = int(loaded_epoch) if loaded_epoch is not None else None
        if not math.isnan(float(loaded_best_monitor)):
            state.best_monitor_value = float(loaded_best_monitor)
        if not math.isnan(float(loaded_best_f1)):
            state.best_checkpoint_val_macro_f1 = float(loaded_best_f1)
            state.best_val_macro_f1 = max(state.best_val_macro_f1, state.best_checkpoint_val_macro_f1)
        if not math.isnan(float(loaded_best_auc)):
            state.best_checkpoint_val_macro_auc = float(loaded_best_auc)
            state.best_val_macro_auc = max(state.best_val_macro_auc, state.best_checkpoint_val_macro_auc)
        if thr is not None and thr.shape[0] == len(ptm_types):
            state.best_thresholds = thr
        logger.info(
            "Loaded %s checkpoint for post-training evaluation: %s (monitor=%s value=%.5f)",
            eval_checkpoint,
            str(eval_ckpt_path),
            loaded_monitor_metric,
            state.best_monitor_value,
        )
    else:
        logger.warning("Requested eval_checkpoint=%s but checkpoint not found: %s", eval_checkpoint, str(eval_ckpt_path))

    val_loss_post, y_val_post, p_val_post, val_struct_avail, val_post_perf = run_epoch_fn(
        model=model,
        loader=val_loader,
        optimizer=None,
        scheduler=None,
        criterion=criterion,
        device=runtime_amp.device,
        options=EpochOptions(
            max_batches=components.sanity_max_val_batches,
            use_amp=runtime_amp.use_amp,
            amp_dtype=runtime_amp.amp_dtype,
            stage_name="val_post_threshold_search",
        ),
    )
    val_post_perf["epoch"] = len(state.history["train_loss"])
    state.epoch_performance.append(val_post_perf)

    if y_val_post.shape[0] > 0:
        state.best_thresholds, threshold_report = search_optimal_thresholds_fn(
            y_true=y_val_post,
            y_prob=p_val_post,
            ptm_types=ptm_types,
            metric=components.threshold_metric,
            max_candidates=int(config.get("threshold_search_max_candidates", 200)),
            min_recall_by_label={
                sanitize_name_fn(k): float(v) for k, v in config.get("threshold_min_recall_by_label", {}).items()
            }
            if isinstance(config.get("threshold_min_recall_by_label"), dict)
            else {},
            max_threshold_by_label={
                sanitize_name_fn(k): float(v) for k, v in config.get("threshold_max_by_label", {}).items()
            }
            if isinstance(config.get("threshold_max_by_label"), dict)
            else {},
        )
        val_threshold_metrics = evaluate_multilabel_fn(
            y_val_post,
            p_val_post,
            ptm_types,
            thresholds=state.best_thresholds,
            monitored_pairs=monitored_pairs,
        )
        val_threshold_metrics["val_loss"] = float(val_loss_post)
        logger.info(
            "Post-training validation threshold search complete: macro_f1=%.5f thresholds=%s",
            float(val_threshold_metrics["macro_f1"]),
            ", ".join([f"{ptm}:{state.best_thresholds[idx]:.4f}" for idx, ptm in enumerate(ptm_types)]),
        )
    else:
        threshold_report = {"metric": components.threshold_metric, "num_samples": 0, "per_label": {}, "selected_thresholds": {}}
        val_threshold_metrics = {"val_loss": float(val_loss_post)}

    if isinstance(val_post_perf.get("module_contributions"), dict):
        export_metrics_fn(output_dir / "module_contributions_val.json", val_post_perf["module_contributions"])

    test_loss, y_test, p_test, test_struct_avail, test_perf = run_epoch_fn(
        model=model,
        loader=test_loader,
        optimizer=None,
        scheduler=None,
        criterion=criterion,
        device=runtime_amp.device,
        options=EpochOptions(
            max_batches=components.sanity_max_test_batches,
            use_amp=runtime_amp.use_amp,
            amp_dtype=runtime_amp.amp_dtype,
            stage_name="test",
        ),
    )
    test_perf["epoch"] = len(state.history["train_loss"])
    state.epoch_performance.append(test_perf)
    test_metrics = evaluate_multilabel_fn(
        y_test,
        p_test,
        ptm_types,
        thresholds=state.best_thresholds,
        monitored_pairs=monitored_pairs,
    )
    test_metrics["test_loss"] = float(test_loss)
    test_metrics["best_val_macro_f1"] = float(state.best_val_macro_f1)
    test_metrics["best_val_macro_auc"] = float(state.best_val_macro_auc)
    test_metrics["best_checkpoint_monitor_metric"] = str(state.early_stopping_metric)
    test_metrics["best_checkpoint_monitor_value"] = float(state.best_monitor_value)
    test_metrics["best_checkpoint_val_macro_f1"] = float(state.best_checkpoint_val_macro_f1)
    test_metrics["best_checkpoint_val_macro_auc"] = float(state.best_checkpoint_val_macro_auc)
    test_metrics["post_threshold_val_macro_f1"] = float(val_threshold_metrics.get("macro_f1", float("nan")))
    test_metrics["eval_checkpoint"] = str(eval_checkpoint)
    test_metrics["eval_checkpoint_path"] = str(eval_ckpt_path)
    test_metrics["eval_checkpoint_epoch"] = int(eval_loaded_epoch) if eval_loaded_epoch is not None else None
    if isinstance(test_perf.get("module_contributions"), dict):
        export_metrics_fn(output_dir / "module_contributions_test.json", test_perf["module_contributions"])

    structure_available_mask = (test_struct_avail >= 0.5) if test_struct_avail.size > 0 else np.zeros((0,), dtype=bool)
    if structure_available_mask.size > 0:
        test_metrics["structure_available_samples"] = int(structure_available_mask.sum())
        test_metrics["structure_missing_samples"] = int((~structure_available_mask).sum())
        subset_metrics = {
            "with_structure": evaluate_multilabel_fn(
                y_test[structure_available_mask],
                p_test[structure_available_mask],
                ptm_types,
                thresholds=state.best_thresholds,
                monitored_pairs=monitored_pairs,
            )
            if int(structure_available_mask.sum()) > 0
            else {},
            "without_structure": evaluate_multilabel_fn(
                y_test[~structure_available_mask],
                p_test[~structure_available_mask],
                ptm_types,
                thresholds=state.best_thresholds,
                monitored_pairs=monitored_pairs,
            )
            if int((~structure_available_mask).sum()) > 0
            else {},
        }
        export_metrics_fn(output_dir / "test_metrics_by_structure.json", subset_metrics)
        export_structure_effect_significance_fn(
            output_dir=output_dir,
            y_true=y_test,
            y_prob=p_test,
            thresholds=state.best_thresholds,
            ptm_types=ptm_types,
            structure_available_mask=structure_available_mask,
        )

    logger.info("Test macro_f1=%.5f macro_roc_auc=%.5f", test_metrics["macro_f1"], test_metrics["macro_roc_auc"])
    return {
        "state": state,
        "threshold_report": threshold_report,
        "val_threshold_metrics": val_threshold_metrics,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "p_test": p_test,
        "val_structure_available": val_struct_avail,
        "test_structure_available": test_struct_avail,
        "test_perf": test_perf,
    }
