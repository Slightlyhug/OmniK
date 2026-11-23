"""Artifact export helpers extracted from the training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def export_training_artifacts(
    *,
    output_dir: Path,
    state: Any,
    components: Any,
    train_distribution_monitor: Any,
    threshold_report: Dict[str, Any],
    val_threshold_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    y_test: FloatArray,
    p_test: FloatArray,
    ptm_types: List[str],
    val_structure_available: Any,
    test_structure_available: Any,
    profiler: Any,
    logger: Any,
    use_mlflow: bool,
    mlflow_module: Any,
    config_path: Path,
    calibration_result: Dict[str, Any],
    module_contributions: Dict[str, Any],
    export_metrics_fn: Callable[[Path, Dict[str, Any]], None],
    build_performance_report_fn: Callable[..., Dict[str, Any]],
    export_startup_profile_comparison_fn: Callable[..., None],
) -> None:
    def _write_back_thresholds_to_config() -> None:
        nonlocal config_path
        runtime_config = components.config if hasattr(components, "config") and isinstance(components.config, dict) else {}
        if not bool(runtime_config.get("write_back_thresholds_to_config", True)):
            logger.info("Threshold write-back to config disabled by write_back_thresholds_to_config=false.")
            return
        if not bool(runtime_config.get("write_back_calibration_thresholds", True)):
            calibration_result.clear()
        if config_path is None:
            return
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if config_path.suffix.lower() != ".json" or not config_path.exists():
            return
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                return
            threshold_source = None
            threshold_arr = None
            calibration_report = calibration_result.get("calibration_report") if isinstance(calibration_result, dict) else None
            if isinstance(calibration_report, dict) and isinstance(calibration_report.get("thresholds"), list):
                threshold_arr = calibration_report.get("thresholds")
                threshold_source = "temperature_scaling"
            elif isinstance(state.best_thresholds, np.ndarray):
                threshold_arr = state.best_thresholds.astype(np.float32, copy=False).tolist()
                threshold_source = "validation_threshold_search"
            if not isinstance(threshold_arr, list) or len(threshold_arr) != len(ptm_types):
                return
            cfg["best_thresholds_by_label"] = {
                str(ptm): float(threshold_arr[idx]) for idx, ptm in enumerate(ptm_types)
            }
            cfg["best_threshold_source"] = str(threshold_source)
            if isinstance(calibration_report, dict):
                cfg["calibration_selected_temperature"] = float(calibration_report.get("temperature", 1.0))
                cfg["calibration_selected_ece_test"] = float(
                    calibration_report.get("ece_test_post", calibration_report.get("ece_test_pre", float("nan")))
                )
            config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Wrote back thresholds to config: %s", str(config_path))
        except Exception as exc:
            logger.warning("Failed to write thresholds back to config %s: %s", str(config_path), str(exc))

    if use_mlflow:
        for key, value in test_metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)) and not np.isnan(float(value)):
                mlflow_module.log_metric(f"test_{key}", float(value))
        per_label_raw = test_metrics.get("per_label")
        per_label: Dict[str, Any] = per_label_raw if isinstance(per_label_raw, dict) else {}
        for label, metrics in per_label.items():
            if not isinstance(metrics, dict):
                continue
            for metric_key, metric_val in metrics.items():
                if isinstance(metric_val, (int, float, np.floating, np.integer)):
                    mlflow_module.log_metric(f"test_{label}_{metric_key}", float(metric_val))
        pair_metrics_raw = test_metrics.get("pair_metrics")
        pair_metrics: Dict[str, Any] = pair_metrics_raw if isinstance(pair_metrics_raw, dict) else {}
        for pair_name, metrics in pair_metrics.items():
            if not isinstance(metrics, dict):
                continue
            for metric_key, metric_val in metrics.items():
                if isinstance(metric_val, (int, float, np.floating, np.integer)):
                    mlflow_module.log_metric(f"test_pair_{pair_name}_{metric_key}", float(metric_val))
        for name in [
            "test_metrics.json",
            "test_metrics_temperature.json",
            "test_metrics_by_structure.json",
            "train_history.json",
            "val_threshold_metrics.json",
            "val_threshold_search.json",
            "performance_report.json",
        ]:
            artifact_path = output_dir / name
            if artifact_path.exists():
                mlflow_module.log_artifact(str(artifact_path))

    export_metrics_fn(output_dir / "test_metrics.json", test_metrics)
    if state.summary_rows:
        pd.DataFrame(state.summary_rows).to_csv(output_dir / "summary.csv", index=False)
    export_metrics_fn(output_dir / "train_history.json", state.history)
    export_metrics_fn(output_dir / "batch_distribution_monitor.json", train_distribution_monitor.summary())
    export_metrics_fn(output_dir / "val_threshold_search.json", threshold_report)
    export_metrics_fn(output_dir / "val_threshold_metrics.json", val_threshold_metrics)
    export_metrics_fn(
        output_dir / "performance_report.json",
        build_performance_report_fn(
            history=state.history,
            epoch_performance=state.epoch_performance,
            config=components.config if hasattr(components, "config") else None,
            baseline=components.baseline_benchmark or None,
            test_metrics=test_metrics,
            val_threshold_metrics=val_threshold_metrics,
            threshold_report=threshold_report,
            calibration_result=calibration_result,
            module_contributions=module_contributions,
        ),
    )
    np.save(output_dir / "test_probabilities.npy", p_test)
    np.save(output_dir / "test_labels.npy", y_test)
    if val_structure_available is not None:
        np.save(output_dir / "val_structure_available.npy", np.asarray(val_structure_available, dtype=np.float32))
    if test_structure_available is not None:
        np.save(output_dir / "test_structure_available.npy", np.asarray(test_structure_available, dtype=np.float32))
    np.save(output_dir / "best_thresholds.npy", state.best_thresholds)

    profiler.save(output_dir / "startup_profile.json")
    export_startup_profile_comparison_fn(output_dir, logger)
    _write_back_thresholds_to_config()

    if components.writer is not None:
        components.writer.close()
    if use_mlflow:
        mlflow_module.end_run()
