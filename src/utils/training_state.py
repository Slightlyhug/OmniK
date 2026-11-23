"""Typed containers shared by multitask training stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class SplitData:
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    feature_sources: Optional[Dict[str, str]] = None
    source_path: Optional[str] = None
    split_policy: Optional[Dict[str, Any]] = None


@dataclass
class RuntimeSetup:
    config_path: Path
    config: Dict[str, Any]
    seed: int
    output_dir: Path
    logger: Any
    profiler: Any
    use_mlflow: bool


@dataclass
class DataPreparationArtifacts:
    ptm_types: List[str]
    splits: SplitData
    report_dir: Path
    structure_cache: Any
    sequence_window_provider: Optional[Any]
    use_sequence_windows: bool
    use_structure_graph: bool
    graph_cache: Optional[Any]


@dataclass
class DataPreparationServices:
    load_splits: Callable[..., SplitData]
    validate_split_integrity: Callable[..., Any]
    filter_records_with_available_pdb: Callable[..., Any]
    filter_records_by_structure_feature_availability: Callable[..., Any]
    repair_or_filter_sequence_windows: Callable[..., Any]
    filter_records_by_feature_quality: Callable[..., Any]
    validate_split_ptm_presence: Callable[..., Any]
    motif_aware_negative_sampling: Callable[..., Any]
    prepare_data_integrity_report: Callable[..., Any]


@dataclass
class DatasetBundle:
    train_set: Any
    val_set: Any
    test_set: Any


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_distribution_monitor: Any
    use_distribution_batch_sampler: bool
    train_batch_sampler: Optional[Any]


@dataclass
class RuntimeDeviceAmp:
    device: torch.device
    use_amp: bool
    amp_dtype: Optional[torch.dtype]
    scaler: Any


@dataclass
class EpochOptions:
    grad_accum_steps: int = 1
    max_batches: Optional[int] = None
    grad_norm_log_interval: int = 0
    grad_norm_topk: int = 5
    grad_norm_warn_threshold: float = 0.0
    scaler: Optional[Any] = None
    use_amp: bool = False
    amp_dtype: Optional[torch.dtype] = None
    distribution_monitor: Optional[Any] = None
    stage_name: str = "train"
    grad_clip_norm: float = 1.0
    log_interval: int = 0
    scheduler_step_unit: str = "step"
    gradient_conflict_log_interval: int = 0
    teacher_distill_weight: float = 0.0
    teacher_distill_temperature: float = 1.0
    teacher_distill_min_confidence: float = 0.0
    teacher_distill_unknown_only: bool = False


@dataclass
class TrainingLoopServices:
    collect_prediction_rate_diagnostics: Callable[..., Any]
    evaluate_multilabel: Callable[..., Any]
    format_lr_summary: Callable[..., Any]
    format_pair_metric_summary: Callable[..., Any]
    format_per_label_summary: Callable[..., Any]
    format_prediction_rate_summary: Callable[..., Any]
    get_lr_diagnostics: Callable[..., Any]
    get_validation_monitor_value: Callable[..., Any]
    resolve_monitored_label_pairs: Callable[..., Any]
    run_epoch: Callable[..., Any]
    sanitize_name: Callable[..., Any]
    save_checkpoint: Callable[..., Any]
    search_optimal_thresholds: Callable[..., Any]
    mlflow_api: Optional[Any] = None


@dataclass
class TrainingComponents:
    config: Dict[str, Any]
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    scheduler_step_unit: str
    writer: Optional[Any]
    early_stopping: Optional[Any]
    baseline_benchmark: Dict[str, Any]
    ckpt_last: Path
    ckpt_best: Path
    best_model_path: Path
    num_epochs: int
    grad_accum_steps: int
    threshold_metric: str
    sanity_max_train_batches: Optional[int]
    sanity_max_val_batches: Optional[int]
    sanity_max_test_batches: Optional[int]
    grad_norm_log_interval: int
    grad_norm_topk: int
    grad_norm_warn_threshold: float
    log_every: int
    overfit_gap_threshold: float
    validate_every_n_epochs: int
    checkpoint_interval_epochs: int
    save_checkpoints: bool
    grad_clip_norm: float
    foreground_warmup_epochs: int
    gradient_conflict_log_interval: int


@dataclass
class TrainingState:
    start_epoch: int
    best_val_macro_f1: float
    best_val_macro_auc: float
    best_checkpoint_val_macro_f1: float
    best_checkpoint_val_macro_auc: float
    best_thresholds: np.ndarray
    early_stopping_metric: str
    early_stopping_mode: str
    best_monitor_value: float
    history: Dict[str, List[float]]
    last_trainable_state: Optional[bool]
    epoch_performance: List[Dict[str, Any]]
    summary_rows: List[Dict[str, Any]]
    current_stage_name: str = "joint"
    dynamic_scale_saturation_streak: int = 0
    stage_transition_epoch: int = 0
