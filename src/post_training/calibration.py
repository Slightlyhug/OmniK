"""Calibration pipeline helpers extracted from the training entrypoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def _resolve_monitored_pairs(config: Dict[str, Any], ptm_types: List[str], sanitize_name_fn: Callable[[str], str]) -> List[tuple[str, str]]:
    raw_pairs = config.get("cooccurrence_monitor_pairs", [])
    if not isinstance(raw_pairs, list):
        return []
    valid = {sanitize_name_fn(name): str(name) for name in ptm_types}
    resolved: List[tuple[str, str]] = []
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


def run_calibration_pipeline(
    *,
    config: Dict[str, Any],
    ptm_types: List[str],
    val_loader: Any,
    test_loader: Any,
    runtime_amp: Any,
    components: Any,
    state: Any,
    model: Any,
    output_dir: Path,
    logger: Any,
    collect_logits_labels_fn: Callable[..., Any],
    sigmoid_np_fn: Callable[[FloatArray], FloatArray],
    macro_ece_fn: Callable[[FloatArray, FloatArray], float],
    fit_temperature_scaling_fn: Callable[..., float],
    apply_temperature_fn: Callable[[FloatArray, float], FloatArray],
    search_optimal_thresholds_fn: Callable[..., Any],
    evaluate_multilabel_fn: Callable[..., Dict[str, Any]],
    export_metrics_fn: Callable[[Path, Dict[str, Any]], None],
    reliability_curve_fn: Callable[..., Dict[str, Any]],
    fit_platt_scaling_fn: Callable[..., Any],
    apply_platt_fn: Callable[..., FloatArray],
    fit_isotonic_fn: Callable[..., Any],
    apply_isotonic_fn: Callable[..., FloatArray],
    sanitize_name_fn: Callable[[str], str],
    has_sklearn_cal: bool,
) -> Dict[str, Any]:
    calibration_cfg_raw = config.get("calibration")
    calibration_cfg: Dict[str, Any] = calibration_cfg_raw if isinstance(calibration_cfg_raw, dict) else {}
    if not bool(calibration_cfg.get("enabled", True)):
        return {}

    calibration_rethreshold = bool(config.get("calibration_rethreshold", True))
    search_max_candidates = int(config.get("threshold_search_max_candidates", 200))
    min_recall_by_label = (
        {sanitize_name_fn(k): float(v) for k, v in config.get("threshold_min_recall_by_label", {}).items()}
        if isinstance(config.get("threshold_min_recall_by_label"), dict)
        else {}
    )
    max_threshold_by_label = (
        {sanitize_name_fn(k): float(v) for k, v in config.get("threshold_max_by_label", {}).items()}
        if isinstance(config.get("threshold_max_by_label"), dict)
        else {}
    )

    monitored_pairs = _resolve_monitored_pairs(config, ptm_types, sanitize_name_fn)
    val_logits, val_labels = collect_logits_labels_fn(
        model, val_loader, device=runtime_amp.device, use_amp=runtime_amp.use_amp, amp_dtype=runtime_amp.amp_dtype
    )
    test_logits, test_labels = collect_logits_labels_fn(
        model, test_loader, device=runtime_amp.device, use_amp=runtime_amp.use_amp, amp_dtype=runtime_amp.amp_dtype
    )
    np.save(output_dir / "val_logits.npy", val_logits)
    np.save(output_dir / "val_labels.npy", val_labels)
    np.save(output_dir / "test_logits.npy", test_logits)

    val_prob_raw = sigmoid_np_fn(val_logits)
    test_prob_raw = sigmoid_np_fn(test_logits)
    ece_val_pre = macro_ece_fn(val_labels, val_prob_raw)
    ece_test_pre = macro_ece_fn(test_labels, test_prob_raw)
    raw_ranking_metrics = evaluate_multilabel_fn(
        test_labels,
        test_prob_raw,
        ptm_types,
        thresholds=state.best_thresholds,
        monitored_pairs=monitored_pairs,
    )

    temperature = fit_temperature_scaling_fn(
        val_logits=val_logits,
        val_labels=val_labels,
        device=runtime_amp.device,
        max_iter=int(calibration_cfg.get("temperature_max_iter", 2000)),
        lr=float(calibration_cfg.get("temperature_lr", 0.01)),
    )
    val_prob_temp = apply_temperature_fn(val_logits, temperature)
    test_prob_temp = apply_temperature_fn(test_logits, temperature)
    ece_val_post = macro_ece_fn(val_labels, val_prob_temp)
    ece_test_post = macro_ece_fn(test_labels, test_prob_temp)

    temp_thresholds = state.best_thresholds
    if calibration_rethreshold and val_labels.size > 0:
        temp_thresholds, _ = search_optimal_thresholds_fn(
            y_true=val_labels,
            y_prob=val_prob_temp,
            ptm_types=ptm_types,
            metric=components.threshold_metric,
            max_candidates=search_max_candidates,
            min_recall_by_label=min_recall_by_label,
            max_threshold_by_label=max_threshold_by_label,
        )

    calibrated_test_metrics = evaluate_multilabel_fn(
        test_labels,
        test_prob_temp,
        ptm_types,
        thresholds=temp_thresholds,
        monitored_pairs=monitored_pairs,
    )
    raw_auc = float(raw_ranking_metrics.get("macro_roc_auc", float("nan")))
    raw_auprc = float(raw_ranking_metrics.get("macro_auprc", float("nan")))
    temp_auc = float(calibrated_test_metrics.get("macro_roc_auc", float("nan")))
    temp_auprc = float(calibrated_test_metrics.get("macro_auprc", float("nan")))
    auc_delta = float(temp_auc - raw_auc) if not (math.isnan(raw_auc) or math.isnan(temp_auc)) else float("nan")
    auprc_delta = float(temp_auprc - raw_auprc) if not (math.isnan(raw_auprc) or math.isnan(temp_auprc)) else float("nan")

    calibration_report = {
        "temperature": float(temperature),
        "ece_val_pre": float(ece_val_pre),
        "ece_val_post": float(ece_val_post),
        "ece_test_pre": float(ece_test_pre),
        "ece_test_post": float(ece_test_post),
        "ece_val_relative_reduction": float((ece_val_pre - ece_val_post) / max(1e-12, ece_val_pre)) if not math.isnan(ece_val_pre) else float("nan"),
        "ece_test_relative_reduction": float((ece_test_pre - ece_test_post) / max(1e-12, ece_test_pre)) if not math.isnan(ece_test_pre) else float("nan"),
        "raw_macro_roc_auc": raw_auc,
        "temperature_macro_roc_auc": temp_auc,
        "temperature_macro_roc_auc_delta": auc_delta,
        "raw_macro_auprc": raw_auprc,
        "temperature_macro_auprc": temp_auprc,
        "temperature_macro_auprc_delta": auprc_delta,
        "ranking_invariant": bool((math.isnan(auc_delta) or abs(auc_delta) < 1e-6) and (math.isnan(auprc_delta) or abs(auprc_delta) < 1e-6)),
        "rethreshold": bool(calibration_rethreshold),
        "thresholds": temp_thresholds.astype(np.float32, copy=False).tolist() if isinstance(temp_thresholds, np.ndarray) else None,
    }
    if not bool(calibration_report["ranking_invariant"]):
        logger.warning(
            "Temperature scaling changed ranking metrics: macro_auc_delta=%.8f macro_auprc_delta=%.8f",
            auc_delta,
            auprc_delta,
        )
    export_metrics_fn(output_dir / "calibration_report_temperature.json", calibration_report)

    calibrated_test_metrics["test_loss"] = None
    calibrated_test_metrics["calibration_temperature"] = float(temperature)
    export_metrics_fn(output_dir / "test_metrics_temperature.json", calibrated_test_metrics)
    np.save(output_dir / "test_probabilities_temperature.npy", test_prob_temp.astype(np.float32, copy=False))

    y_flat = test_labels.reshape(-1)
    p_flat_pre = test_prob_raw.reshape(-1)
    p_flat_post = test_prob_temp.reshape(-1)
    curve_pre = reliability_curve_fn(y_flat, p_flat_pre, n_bins=int(calibration_cfg.get("reliability_n_bins", 15)))
    curve_post = reliability_curve_fn(y_flat, p_flat_post, n_bins=int(calibration_cfg.get("reliability_n_bins", 15)))
    export_metrics_fn(output_dir / "reliability_curve_pre.json", curve_pre)
    export_metrics_fn(output_dir / "reliability_curve_temperature.json", curve_post)

    baselines: Dict[str, Any] = {}
    if bool(calibration_cfg.get("enable_platt", True)) and has_sklearn_cal:
        platt_models = fit_platt_scaling_fn(val_logits, val_labels)
        if platt_models is not None:
            val_prob_platt = apply_platt_fn(platt_models, val_logits)
            test_prob_platt = apply_platt_fn(platt_models, test_logits)
            platt_thresholds = state.best_thresholds
            if calibration_rethreshold and val_labels.size > 0:
                platt_thresholds, _ = search_optimal_thresholds_fn(
                    y_true=val_labels,
                    y_prob=val_prob_platt,
                    ptm_types=ptm_types,
                    metric=components.threshold_metric,
                    max_candidates=search_max_candidates,
                    min_recall_by_label=min_recall_by_label,
                    max_threshold_by_label=max_threshold_by_label,
                )
            baselines["platt"] = {
                "ece_test": float(macro_ece_fn(test_labels, test_prob_platt)),
                "rethreshold": bool(calibration_rethreshold),
                "thresholds": platt_thresholds.astype(np.float32, copy=False).tolist() if isinstance(platt_thresholds, np.ndarray) else None,
                "metrics": evaluate_multilabel_fn(
                    test_labels,
                    test_prob_platt,
                    ptm_types,
                    thresholds=platt_thresholds,
                    monitored_pairs=monitored_pairs,
                ),
            }
            np.save(output_dir / "test_probabilities_platt.npy", test_prob_platt.astype(np.float32, copy=False))
    if bool(calibration_cfg.get("enable_isotonic", True)) and has_sklearn_cal:
        iso_models = fit_isotonic_fn(val_prob_raw, val_labels)
        if iso_models is not None:
            val_prob_iso = apply_isotonic_fn(iso_models, val_prob_raw)
            test_prob_iso = apply_isotonic_fn(iso_models, test_prob_raw)
            iso_thresholds = state.best_thresholds
            if calibration_rethreshold and val_labels.size > 0:
                iso_thresholds, _ = search_optimal_thresholds_fn(
                    y_true=val_labels,
                    y_prob=val_prob_iso,
                    ptm_types=ptm_types,
                    metric=components.threshold_metric,
                    max_candidates=search_max_candidates,
                    min_recall_by_label=min_recall_by_label,
                    max_threshold_by_label=max_threshold_by_label,
                )
            baselines["isotonic"] = {
                "ece_test": float(macro_ece_fn(test_labels, test_prob_iso)),
                "rethreshold": bool(calibration_rethreshold),
                "thresholds": iso_thresholds.astype(np.float32, copy=False).tolist() if isinstance(iso_thresholds, np.ndarray) else None,
                "metrics": evaluate_multilabel_fn(
                    test_labels,
                    test_prob_iso,
                    ptm_types,
                    thresholds=iso_thresholds,
                    monitored_pairs=monitored_pairs,
                ),
            }
            np.save(output_dir / "test_probabilities_isotonic.npy", test_prob_iso.astype(np.float32, copy=False))
    export_metrics_fn(output_dir / "calibration_report_baselines.json", baselines)

    min_reduction = float(calibration_cfg.get("min_ece_reduction", 0.1))
    if not math.isnan(ece_test_pre) and not math.isnan(ece_test_post):
        achieved = float((ece_test_pre - ece_test_post) / max(1e-12, ece_test_pre))
        if achieved < min_reduction:
            logger.warning("Temperature scaling ECE reduction %.2f%% < target %.2f%%", achieved * 100.0, min_reduction * 100.0)
        else:
            logger.info("Temperature scaling ECE reduction %.2f%% meets target %.2f%%", achieved * 100.0, min_reduction * 100.0)

    return {
        "calibration_report": calibration_report,
        "calibrated_test_metrics": calibrated_test_metrics,
        "baselines": baselines,
    }
