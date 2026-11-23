"""Top-level orchestration for post-training phases."""

from __future__ import annotations

from typing import Any, Dict

from src.post_training.artifact_export import export_training_artifacts
from src.post_training.calibration import run_calibration_pipeline
from src.post_training.post_eval import run_post_training_evaluation


def main(
    *,
    args: Any,
    config: Dict[str, Any],
    ptm_types: list[str],
    val_loader: Any,
    test_loader: Any,
    criterion: Any,
    runtime_amp: Any,
    components: Any,
    state: Any,
    model: Any,
    logger: Any,
    output_dir: Any,
    profiler: Any,
    train_distribution_monitor: Any,
    use_mlflow: bool,
    mlflow_module: Any,
    dependencies: Dict[str, Any],
) -> Dict[str, Any]:
    post_eval_result = run_post_training_evaluation(
        args=args,
        config=config,
        ptm_types=ptm_types,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        runtime_amp=runtime_amp,
        components=components,
        state=state,
        model=model,
        logger=logger,
        output_dir=output_dir,
        run_epoch_fn=dependencies["run_epoch_fn"],
        load_checkpoint_fn=dependencies["load_checkpoint_fn"],
        search_optimal_thresholds_fn=dependencies["search_optimal_thresholds_fn"],
        evaluate_multilabel_fn=dependencies["evaluate_multilabel_fn"],
        export_metrics_fn=dependencies["export_metrics_fn"],
        export_structure_effect_significance_fn=dependencies["export_structure_effect_significance_fn"],
        sanitize_name_fn=dependencies["sanitize_name_fn"],
    )
    calibration_result = run_calibration_pipeline(
        config=config,
        ptm_types=ptm_types,
        val_loader=val_loader,
        test_loader=test_loader,
        runtime_amp=runtime_amp,
        components=components,
        state=post_eval_result["state"],
        model=model,
        output_dir=output_dir,
        logger=logger,
        collect_logits_labels_fn=dependencies["collect_logits_labels_fn"],
        sigmoid_np_fn=dependencies["sigmoid_np_fn"],
        macro_ece_fn=dependencies["macro_ece_fn"],
        fit_temperature_scaling_fn=dependencies["fit_temperature_scaling_fn"],
        apply_temperature_fn=dependencies["apply_temperature_fn"],
        search_optimal_thresholds_fn=dependencies["search_optimal_thresholds_fn"],
        evaluate_multilabel_fn=dependencies["evaluate_multilabel_fn"],
        export_metrics_fn=dependencies["export_metrics_fn"],
        reliability_curve_fn=dependencies["reliability_curve_fn"],
        fit_platt_scaling_fn=dependencies["fit_platt_scaling_fn"],
        apply_platt_fn=dependencies["apply_platt_fn"],
        fit_isotonic_fn=dependencies["fit_isotonic_fn"],
        apply_isotonic_fn=dependencies["apply_isotonic_fn"],
        sanitize_name_fn=dependencies["sanitize_name_fn"],
        has_sklearn_cal=dependencies["has_sklearn_cal"],
    )
    export_training_artifacts(
        output_dir=output_dir,
        state=post_eval_result["state"],
        components=components,
        train_distribution_monitor=train_distribution_monitor,
        threshold_report=post_eval_result["threshold_report"],
        val_threshold_metrics=post_eval_result["val_threshold_metrics"],
        test_metrics=post_eval_result["test_metrics"],
        y_test=post_eval_result["y_test"],
        p_test=post_eval_result["p_test"],
        ptm_types=ptm_types,
        val_structure_available=post_eval_result.get("val_structure_available"),
        test_structure_available=post_eval_result.get("test_structure_available"),
        profiler=profiler,
        logger=logger,
        use_mlflow=use_mlflow,
        mlflow_module=mlflow_module,
        config_path=args.config_path,
        calibration_result=calibration_result,
        module_contributions=post_eval_result["test_perf"].get("module_contributions", {}) if isinstance(post_eval_result.get("test_perf"), dict) else {},
        export_metrics_fn=dependencies["export_metrics_fn"],
        build_performance_report_fn=dependencies["build_performance_report_fn"],
        export_startup_profile_comparison_fn=dependencies["export_startup_profile_comparison_fn"],
    )
    return {
        "state": post_eval_result["state"],
        "post_eval": post_eval_result,
        "calibration": calibration_result,
    }
