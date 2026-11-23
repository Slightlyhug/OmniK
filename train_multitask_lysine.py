#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-task multi-label training for lysine PTMs.

This script keeps the existing dual-branch architecture and upgrades the
binary output head to one independent sigmoid unit per configured PTM task:
- ubiquitination
- sumoylation
- succinylation
- methylation
- crotonylation
- acetylation
- 2-hydroxyisobutyryl
- lactylation
"""

import argparse
import copy
import json
import logging
import math
import multiprocessing
import os
import random
import time
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
# Keep CUDA allocations in expandable segments before the first CUDA context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from src.orchestrator import main as orchestrate_post_training
from src.models.multitask_heads import (
    MultiTaskIndependentHead,
    PTMAwareCrossAttention,
    build_model,
)
from src.utils.training_config import (
    DEFAULT_MOTIF_TASKS,
    DEFAULT_PTM_TYPES,
    DEFAULT_SUMO_HYDROPHOBIC,
    LABEL_STATE_ALIASES,
    LABEL_STATE_TO_ID,
    _resolve_config_path,
    canonical_ptm_name,
    label_state_id,
    normalize_and_validate_config,
    normalize_label_state,
    sanitize_name,
)
from src.utils.training_runtime import (
    StageProfiler,
    export_startup_profile_comparison,
    flatten_config_for_logging,
    set_seed,
    setup_logger,
)
from src.utils.training_optim import create_optimizer, create_scheduler
from src.utils.training_engine import run_epoch
from src.utils.training_data import to_device
from src.utils.training_runtime import summarize_epoch_performance
from src.utils.training_state import (
    DataPreparationArtifacts,
    DataPreparationServices,
    DatasetBundle,
    EpochOptions,
    LoaderBundle,
    RuntimeDeviceAmp,
    RuntimeSetup,
    SplitData,
    TrainingComponents,
    TrainingLoopServices,
    TrainingState,
)
from src.utils.training_loop import run_training_loop
from src.utils.training_datasets import build_dataset_bundle
from src.utils.training_preparation import prepare_data_artifacts
from src.utils.training_validation import (
    prepare_data_integrity_report,
    summarize_split_records,
    validate_predefined_split_integrity,
    validate_split_ptm_presence,
)
from src.utils.training_quality import (
    filter_records_by_feature_quality,
    filter_records_by_structure_feature_availability,
    filter_records_with_available_pdb,
    repair_or_filter_sequence_windows,
)
from src.utils.training_data import (
    HAS_BIO_PDB,
    HAS_LMDB,
    BlockShuffleBatchSampler,
    GraphFeatureCache,
    GraphFeatureLMDBCache,
    H5DataLoader,
    H5FeatureStoreManager,
    MultiLabelPTMJsonDataset,
    PDBStructureCache,
    PreprocessedTensorLMDBDataset,
    ProteinDedupLMDBDataset,
    _collate_graphs,
    _drop_lmdb_mmap_pages,
    multilabel_collate_fn,
)
from src.utils.training_records import (
    compute_motif_bias_scores,
    SequenceWindowProvider,
    _canonical_ptm_label,
    _extract_teacher_values,
    _labels_to_array,
    _parse_first_fasta_sequence,
    _resolve_ptm_from_record,
    _row_to_multilabel_record,
    _teacher_column_candidates,
    attach_teacher_values,
    build_feature_sources,
    load_json_records,
    load_split_csv,
    split_records_from_json,
)
from src.utils.training_samplers import (
    BatchDistributionMonitor,
    DistributionAlignedBatchSampler,
    EarlyStopping,
    EarlyStoppingState,
    MinimumLabelCoverageBatchSampler,
    _known_label_matrix,
    _positive_label_distribution,
    _positive_label_matrix,
    compute_multilabel_sample_weights,
)
from src.utils.training_losses import (
    AdaptiveTaskWeightedLoss,
    AsymmetricLoss,
    compute_pos_weight,
)
from src.utils.training_calibration import (
    HAS_SKLEARN_CAL,
    apply_isotonic,
    apply_platt,
    apply_temperature,
    fit_isotonic,
    fit_platt_scaling,
    fit_temperature_scaling,
    macro_ece,
    reliability_curve,
    run_calibration_unit_tests,
    sigmoid_np,
)
from src.utils.training_metrics import (
    _cooccurrence_lift,
    _metric_score,
    ece_binary,
    evaluate_multilabel,
    format_pair_metric_summary,
    format_per_label_summary,
    mcc_score,
    optimize_thresholds,
    precision_recall_at_k,
    resolve_monitored_label_pairs,
    safe_auc,
    safe_auprc,
    search_optimal_thresholds,
)
from src.utils.training_checkpoint import (
    export_metrics,
    load_checkpoint,
    load_shape_compatible_state_dict,
    save_checkpoint,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TB = True
except Exception:
    SummaryWriter = None
    HAS_TB = False

try:
    import mlflow

    HAS_MLFLOW = True
except Exception:
    mlflow = None
    HAS_MLFLOW = False

_LMDB_MMAP_SPANS: Optional[List[Tuple[int, int]]] = None
_LIBC_MADVISE = None


def apply_task_loss_weight_csvs(
    records: List[Dict[str, Any]],
    ptm_types: List[str],
    weight_csvs: Any,
    logger: Optional[logging.Logger] = None,
) -> int:
    if not weight_csvs:
        return 0
    if isinstance(weight_csvs, (str, Path)):
        paths = [weight_csvs]
    elif isinstance(weight_csvs, list):
        paths = weight_csvs
    else:
        return 0

    ptm_to_idx = {sanitize_name(ptm): idx for idx, ptm in enumerate(ptm_types)}
    total_updates = 0
    for raw_path in paths:
        path = Path(str(raw_path))
        if not path.exists():
            if logger is not None:
                logger.warning("Task loss weight CSV not found: %s", str(path))
            continue
        df = pd.read_csv(path)
        required = {"row_index", "label", "sample_weight"}
        if not required.issubset(set(df.columns)):
            if logger is not None:
                logger.warning("Task loss weight CSV missing required columns %s: %s", sorted(required), str(path))
            continue
        updates = 0
        for row in df.to_dict(orient="records"):
            try:
                rec_idx = int(row["row_index"])
            except Exception:
                continue
            if rec_idx < 0 or rec_idx >= len(records):
                continue
            label_key = sanitize_name(row.get("label", ""))
            if label_key not in ptm_to_idx:
                continue
            try:
                weight = float(row["sample_weight"])
            except Exception:
                continue
            weights = records[rec_idx].get("loss_weight")
            if not isinstance(weights, list) or len(weights) != len(ptm_types):
                weights = [1.0] * len(ptm_types)
            task_idx = ptm_to_idx[label_key]
            weights[task_idx] = float(np.clip(weight, 0.0, 10.0))
            records[rec_idx]["loss_weight"] = weights

            label_states = records[rec_idx].get("label_state") or records[rec_idx].get("label_states")
            if not isinstance(label_states, list) or len(label_states) != len(ptm_types):
                labels = _labels_to_array(records[rec_idx], ptm_types)
                label_states = [normalize_label_state(None, label_value=float(y)) for y in labels]
            state_raw = row.get("label_state") if "label_state" in row else row.get("noise_bucket", None)
            label_value = row.get("label_value", None)
            label_states[task_idx] = normalize_label_state(state_raw, label_value=label_value)
            records[rec_idx]["label_state"] = label_states

            reliabilities = records[rec_idx].get("label_reliability") or records[rec_idx].get("label_reliability_score")
            if not isinstance(reliabilities, list) or len(reliabilities) != len(ptm_types):
                reliabilities = [1.0] * len(ptm_types)
            raw_rel = row.get("reliability_score", row.get("label_reliability", row.get("sample_weight", 1.0)))
            try:
                rel = float(raw_rel)
                if rel > 1.0:
                    rel = min(1.0, rel / 10.0)
            except Exception:
                rel = 1.0
            reliabilities[task_idx] = float(np.clip(rel, 0.0, 1.0))
            records[rec_idx]["label_reliability"] = reliabilities
            updates += 1
        total_updates += updates
        if logger is not None:
            logger.info("Applied %d task loss weights from %s", updates, str(path))
    return total_updates





def _extract_motif_from_window(window_seq: str, motif_radius: int) -> Optional[str]:
    if not window_seq:
        return None
    w = str(window_seq).strip().upper()
    if not w:
        return None
    center = len(w) // 2
    r = int(motif_radius)
    if r <= 0:
        return w[center:center + 1]
    start = max(0, center - r)
    end = min(len(w), center + r + 1)
    motif = w[start:end]
    if len(motif) != (2 * r + 1):
        motif = motif.ljust(2 * r + 1, "X")
    return motif


def motif_aware_negative_sampling(
    train_records: List[Dict[str, Any]],
    ptm_types: List[str],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    cfg = config.get("motif_aware_negative_sampling") if isinstance(config.get("motif_aware_negative_sampling"), dict) else {}
    if not bool(cfg.get("enabled", False)):
        return train_records

    target_ptms = cfg.get("target_ptm_types", ["crotonylation", "methylation"])
    target_ptms = [sanitize_name(x) for x in (target_ptms if isinstance(target_ptms, list) else [])]
    if not target_ptms:
        return train_records

    neg_ratio = float(cfg.get("neg_ratio", 5.0))
    motif_radius = int(cfg.get("motif_radius", 3))
    seed = int(cfg.get("seed", int(config.get("seed", 42))))
    apply_all_zero_only = bool(cfg.get("apply_to_all_zero_only", True))

    fasta_dirs: List[str] = []
    mapping = config.get("pdb_fasta_mapping") if isinstance(config.get("pdb_fasta_mapping"), dict) else {}
    for m in mapping.values():
        if isinstance(m, dict) and m.get("fasta_dir"):
            fasta_dirs.append(str(m.get("fasta_dir")))
    fasta_dirs = sorted(list({d for d in fasta_dirs if d}))
    provider = SequenceWindowProvider(fasta_dirs=fasta_dirs, window_size=int(config.get("window_size", 73)), logger=logger)
    always_realign_windows = bool(config.get("sequence_window_force_fasta_realign", True))

    ptm_index = {sanitize_name(name): idx for idx, name in enumerate(ptm_types)}
    target_indices = [ptm_index[p] for p in target_ptms if p in ptm_index]
    if not target_indices:
        return train_records

    label_arrays = [_labels_to_array(rec, ptm_types) for rec in train_records]
    is_all_zero = [bool(arr.sum() <= 0) for arr in label_arrays]

    positive_motifs: Dict[str, Set[str]] = {ptm: set() for ptm in target_ptms}
    pos_counts: Dict[str, int] = {ptm: 0 for ptm in target_ptms}

    for rec, arr in zip(train_records, label_arrays):
        uid = rec.get("uniprot_id")
        pos = rec.get("position")
        window_seq = provider.get_window(uid, int(pos)) if always_realign_windows else None
        if window_seq is None:
            window_seq = rec.get("sequence_window_str") or provider.get_window(uid, int(pos))
        motif = _extract_motif_from_window(window_seq, motif_radius) if window_seq else None
        if motif is None:
            continue
        for ptm, idx in zip(target_ptms, [ptm_index.get(p, -1) for p in target_ptms]):
            if idx >= 0 and float(arr[idx]) > 0:
                positive_motifs[ptm].add(motif)
                pos_counts[ptm] += 1

    all_zero_indices = [i for i, flag in enumerate(is_all_zero) if flag]
    if not all_zero_indices:
        logger.info("Motif-aware negative sampling skipped: no all-zero samples in train.")
        return train_records

    hard_indices: List[int] = []
    easy_indices: List[int] = []
    for i in all_zero_indices:
        rec = train_records[i]
        uid = rec.get("uniprot_id")
        pos = rec.get("position")
        window_seq = provider.get_window(uid, int(pos)) if always_realign_windows else None
        if window_seq is None:
            window_seq = rec.get("sequence_window_str") or provider.get_window(uid, int(pos))
        motif = _extract_motif_from_window(window_seq, motif_radius) if window_seq else None
        if motif is None:
            easy_indices.append(i)
            continue
        is_hard = False
        for ptm in target_ptms:
            if motif in positive_motifs.get(ptm, set()):
                is_hard = True
                break
        if is_hard:
            hard_indices.append(i)
        else:
            easy_indices.append(i)

    desired_all_zero_max = 0
    for ptm in target_ptms:
        desired_all_zero_max = max(desired_all_zero_max, int(round(max(1, pos_counts.get(ptm, 1)) * neg_ratio)))
    desired_all_zero_max = max(desired_all_zero_max, int(cfg.get("min_all_zero_keep", 0)))

    rng = np.random.default_rng(seed)
    keep_easy = []
    remaining = max(0, desired_all_zero_max - len(hard_indices))
    if remaining > 0 and easy_indices:
        choose = min(remaining, len(easy_indices))
        keep_easy = rng.choice(np.array(easy_indices, dtype=np.int64), size=choose, replace=False).tolist()

    keep_all_zero_set = set(hard_indices + keep_easy) if apply_all_zero_only else set(hard_indices + keep_easy)

    kept_records: List[Dict[str, Any]] = []
    dropped_easy = 0
    for idx, rec in enumerate(train_records):
        if apply_all_zero_only and idx in all_zero_indices:
            if idx in keep_all_zero_set:
                kept_records.append(rec)
            else:
                dropped_easy += 1
        else:
            kept_records.append(rec)

    logger.info(
        "Motif-aware negative sampling applied. targets=%s motif_radius=%d neg_ratio=%.2f all_zero_before=%d hard=%d easy_kept=%d easy_dropped=%d train_before=%d train_after=%d",
        ",".join(target_ptms),
        motif_radius,
        neg_ratio,
        len(all_zero_indices),
        len(hard_indices),
        len(keep_easy),
        dropped_easy,
        len(train_records),
        len(kept_records),
    )
    return kept_records


def resolve_amp_dtype(config: Dict[str, Any]) -> torch.dtype:
    raw_amp_dtype = config.get("amp_dtype", "float16")
    # JSON null must not silently disable AMP; treat it as the CUDA default.
    amp_dtype = str("float16" if raw_amp_dtype is None else raw_amp_dtype).lower().strip()
    if amp_dtype in {"float32", "fp32", "32", "f32"}:
        return torch.float32
    if amp_dtype in {"auto", ""}:
        amp_dtype = "bfloat16"
    if amp_dtype in {"bfloat16", "bf16"}:
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if amp_dtype in {"float16", "fp16", "16", "f16"}:
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {raw_amp_dtype!r}")


def get_validation_monitor_value(metric_name: str, val_loss: float, val_macro_f1: float, val_macro_auc: float, val_macro_f1_opt: Optional[float] = None) -> float:
    metric = str(metric_name).lower().strip()
    if metric == "val_loss":
        return float(val_loss)
    if metric == "val_macro_auc":
        return float(val_macro_auc)
    if metric in {"val_macro_f1_opt", "val_macro_f1_optimized", "val_threshold_macro_f1"}:
        return float(val_macro_f1_opt if val_macro_f1_opt is not None else val_macro_f1)
    return float(val_macro_f1)
    return torch.float16


def parse_training_log_benchmark(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}
    epoch_entries: List[Tuple[float, int, int]] = []
    pattern = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Epoch (?P<epoch>\d+)/(?P<total>\d+)")
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if not match:
                continue
            ts = time.mktime(time.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"))
            epoch_entries.append((ts, int(match.group("epoch")), int(match.group("total"))))
    if len(epoch_entries) < 2:
        return {}
    durations: List[float] = []
    for prev, curr in zip(epoch_entries[:-1], epoch_entries[1:]):
        prev_ts, prev_epoch, prev_total = prev
        curr_ts, curr_epoch, curr_total = curr
        if curr_total == prev_total and curr_epoch == prev_epoch + 1:
            durations.append(curr_ts - prev_ts)
    if not durations:
        return {}
    return {
        "source_log": str(log_path),
        "avg_epoch_time_sec": float(np.mean(durations)),
        "median_epoch_time_sec": float(np.median(durations)),
        "observed_epoch_intervals": int(len(durations)),
    }


def build_performance_report(
    history: Dict[str, List[float]],
    epoch_performance: List[Dict[str, Any]],
    config: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    val_threshold_metrics: Optional[Dict[str, Any]] = None,
    threshold_report: Optional[Dict[str, Any]] = None,
    calibration_result: Optional[Dict[str, Any]] = None,
    module_contributions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_records = [row for row in epoch_performance if row.get("stage") == "train"]
    target_macro_f1 = float(config.get("target_macro_f1", 0.85)) if isinstance(config, dict) else 0.85
    summary = {
        "train_epoch_time_sec_mean": float(np.mean([row["elapsed_time_sec"] for row in train_records])) if train_records else None,
        "train_samples_per_sec_mean": float(np.mean([row["samples_per_sec"] for row in train_records])) if train_records else None,
        "train_max_gpu_memory_mb_peak": float(np.max([row["max_gpu_memory_mb"] for row in train_records if row["max_gpu_memory_mb"] is not None])) if any(row["max_gpu_memory_mb"] is not None for row in train_records) else None,
        "epochs_completed": int(len(history.get("train_loss", []))),
    }
    depth_layers = int(config.get("seq_transformer_layers", 0)) + int(config.get("struct_transformer_layers", 0))
    block_size = max(1, int(config.get("attn_res_block_size", 2)))
    block_states = int(math.ceil(max(1, depth_layers) / block_size))
    report = {
        "summary": summary,
        "epoch_records": epoch_performance,
        "communication_proxy": {
            "full_depth_states": int(max(1, depth_layers)),
            "block_depth_states": block_states,
            "state_reduction_ratio": float(max(1, depth_layers) / max(1, block_states)),
            "block_size": block_size,
        },
    }
    if isinstance(test_metrics, dict):
        per_label = test_metrics.get("per_label") if isinstance(test_metrics.get("per_label"), dict) else {}
        failing_tasks = []
        passing_tasks = []
        for label, metrics in per_label.items():
            if not isinstance(metrics, dict):
                continue
            f1_value = float(metrics.get("f1", float("nan")))
            task_row = {
                "label": str(label),
                "f1": f1_value,
                "precision": float(metrics.get("precision", float("nan"))),
                "recall": float(metrics.get("recall", float("nan"))),
                "auprc": float(metrics.get("auprc", float("nan"))),
                "ece": float(metrics.get("ece", float("nan"))),
                "threshold": float(metrics.get("threshold", float("nan"))),
                "gap_to_target": float(target_macro_f1 - f1_value) if not math.isnan(f1_value) else float("nan"),
            }
            if not math.isnan(f1_value) and f1_value >= target_macro_f1:
                passing_tasks.append(task_row)
            else:
                failing_tasks.append(task_row)
        failing_tasks = sorted(failing_tasks, key=lambda x: (math.isnan(x["f1"]), x["f1"]))
        passing_tasks = sorted(passing_tasks, key=lambda x: x["f1"], reverse=True)

        module_bottlenecks: List[Dict[str, Any]] = []
        if isinstance(module_contributions, dict):
            last_iou = module_contributions.get("last_mask_iou_mean")
            if isinstance(last_iou, list) and last_iou:
                last_iou_val = float(np.nanmean(np.asarray(last_iou, dtype=np.float64)))
            elif isinstance(last_iou, (float, int, np.floating, np.integer)):
                last_iou_val = float(last_iou)
            else:
                last_iou_val = float("nan")
            task_sim = module_contributions.get("last_task_inter_similarity")
            if isinstance(task_sim, list) and task_sim:
                task_sim_val = float(np.nanmean(np.asarray(task_sim, dtype=np.float64)))
            elif isinstance(task_sim, (float, int, np.floating, np.integer)):
                task_sim_val = float(task_sim)
            else:
                task_sim_val = float("nan")
            graph_score_mean = float(module_contributions.get("last_graph_score_mean", float("nan")))
            graph_sparsity = float(module_contributions.get("last_graph_score_sparsity", float("nan")))

            if not math.isnan(last_iou_val) and last_iou_val > 0.90:
                module_bottlenecks.append(
                    {
                        "module": "last_foreground",
                        "severity": "high",
                        "signal": "task_masks_high_overlap",
                        "value": last_iou_val,
                        "diagnosis": "任务前景掩码几乎重合，说明任务特异性不足，易造成负迁移。",
                    }
                )
            if not math.isnan(task_sim_val) and task_sim_val > 0.95:
                module_bottlenecks.append(
                    {
                        "module": "last_task_context",
                        "severity": "high",
                        "signal": "task_representation_collapse",
                        "value": task_sim_val,
                        "diagnosis": "任务表示相似度过高，说明共享表示塌缩，难以拉开低表现任务。",
                    }
                )
            if not math.isnan(graph_score_mean) and not math.isnan(graph_sparsity) and graph_score_mean > 0.95 and graph_sparsity < 0.01:
                module_bottlenecks.append(
                    {
                        "module": "structure_graph",
                        "severity": "medium",
                        "signal": "graph_attention_saturation",
                        "value": graph_score_mean,
                        "diagnosis": "图分支打分接近常数且无稀疏性，说明结构图信息区分度偏弱。",
                    }
                )

        report["target_assessment"] = {
            "target_macro_f1": float(target_macro_f1),
            "current_macro_f1": float(test_metrics.get("macro_f1", float("nan"))),
            "current_macro_ece": float(test_metrics.get("macro_ece", float("nan"))),
            "meets_target": bool(float(test_metrics.get("macro_f1", float("nan"))) >= target_macro_f1 and len(failing_tasks) == 0),
            "num_tasks_below_target": int(len(failing_tasks)),
            "failing_tasks": failing_tasks,
            "passing_tasks": passing_tasks,
        }
        pair_metrics = test_metrics.get("pair_metrics") if isinstance(test_metrics.get("pair_metrics"), dict) else {}
        if pair_metrics:
            pair_rows = []
            for pair_name, metrics in pair_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                pair_rows.append(
                    {
                        "pair": str(pair_name),
                        "joint_f1": float(metrics.get("joint_f1", float("nan"))),
                        "joint_recall": float(metrics.get("joint_recall", float("nan"))),
                        "pred_lift": float(metrics.get("pred_lift", float("nan"))),
                        "true_lift": float(metrics.get("true_lift", float("nan"))),
                        "lift_gap": float(metrics.get("pred_lift", float("nan")) - metrics.get("true_lift", float("nan"))),
                    }
                )
            pair_rows = sorted(pair_rows, key=lambda x: (math.isnan(x["joint_f1"]), x["joint_f1"]))
            report["cooccurrence_assessment"] = {
                "monitored_pairs": pair_rows,
                "worst_pairs": pair_rows[:2],
                "best_pairs": sorted(pair_rows, key=lambda x: (math.isnan(x["joint_f1"]), -x["joint_f1"]))[:2],
            }
        report["test_metrics"] = test_metrics
        report["val_threshold_metrics"] = val_threshold_metrics or {}
        report["threshold_search"] = threshold_report or {}
        report["calibration"] = calibration_result or {}
        report["module_contributions"] = module_contributions or {}
        report["module_bottlenecks"] = module_bottlenecks
        if failing_tasks:
            weakest_labels = [str(row["label"]) for row in failing_tasks[:3]]
            next_iteration = {
                "data": [
                    f"优先扩充与重采样低表现任务: {', '.join(weakest_labels)}",
                    "执行时间切分+CD-HIT<=30% 去冗余，确保验证/测试无同源泄漏",
                    "对全零样本继续做 motif-aware hard negative mining，并增加窗口滑动增强",
                ],
                "structure": [
                    "重构 LaST 掩码约束，降低 task mask overlap 与 task representation collapse",
                    "为 structure_graph 增加更强稀疏化/归一化，避免图打分饱和为常数",
                    "优先保留对低表现任务有增益的 task-specific heads / adapters / specialists",
                ],
                "training": [
                    "将 early_stopping 固定到 val_macro_f1，并对低表现任务做分阶段单任务微调",
                    "启用分层学习率与最后若干 checkpoint 的 SWA 融合",
                    "补充 R-Drop 与 label smoothing，配合 focal / dynamic_kg 控制长尾任务波动",
                ],
                "post_processing": [
                    "按任务重搜 optimal F1 threshold，并将温度缩放后的阈值回写配置",
                    "对每个任务比较 raw / temperature / platt / isotonic 的 F1 与 ECE 后择优",
                    "若任一任务仍低于 0.85，触发 5-fold soft voting 再训练",
                ],
            }
            report["next_iteration_top3"] = next_iteration
    if baseline:
        report["baseline"] = baseline
        if baseline.get("avg_epoch_time_sec") and summary["train_epoch_time_sec_mean"]:
            report["comparison"] = {
                "epoch_time_speedup": float(baseline["avg_epoch_time_sec"] / summary["train_epoch_time_sec_mean"]),
            }
    return report


def z_test_two_proportions(x1: int, n1: int, x2: int, n2: int) -> float:
    n1 = int(n1)
    n2 = int(n2)
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p1 = float(x1 / n1)
    p2 = float(x2 / n2)
    p = float((x1 + x2) / (n1 + n2))
    denom = float((p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)) ** 0.5)
    if denom <= 0:
        return float("nan")
    z = float((p1 - p2) / denom)
    p_value = float(math.erfc(abs(z) / (2 ** 0.5)))
    return p_value


def export_structure_effect_significance(
    output_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    ptm_types: List[str],
    structure_available_mask: np.ndarray,
) -> None:
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)
    mask = structure_available_mask.astype(bool, copy=False)
    n1 = int(mask.sum())
    n2 = int((~mask).sum())
    if n1 == 0 or n2 == 0:
        logging.getLogger("multitask_lysine").warning(
            "Skipping structure effect significance: n_with_structure=%d n_without_structure=%d",
            n1,
            n2,
        )
        export_metrics(
            output_dir / "structure_effect_significance.json",
            {
                "skipped": True,
                "reason": "Both structure groups are required to compute significance.",
                "n_with_structure": int(n1),
                "n_without_structure": int(n2),
                "per_label": {},
            },
        )
        return
    stats = {"skipped": False, "n_with_structure": int(n1), "n_without_structure": int(n2), "per_label": {}}
    for i, name in enumerate(ptm_types):
        yi = y_true[:, i].astype(np.int32, copy=False)
        bi = y_pred[:, i].astype(np.int32, copy=False)
        err = (bi != yi).astype(np.int32)
        e1 = int(err[mask].sum())
        e2 = int(err[~mask].sum())
        p_value = z_test_two_proportions(e1, n1, e2, n2)
        stats["per_label"][name] = {
            "with_structure_error_rate": float(e1 / max(1, n1)),
            "without_structure_error_rate": float(e2 / max(1, n2)),
            "delta_without_minus_with": float((e2 / max(1, n2)) - (e1 / max(1, n1))),
            "p_value_ztest": float(p_value),
            "n_with_structure": int(n1),
            "n_without_structure": int(n2),
        }
    export_metrics(output_dir / "structure_effect_significance.json", stats)


def collect_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = to_device(batch, device)
            amp_enabled = bool(use_amp and device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
            with torch.autocast(device_type=("cuda" if device.type == "cuda" else device.type), dtype=amp_dtype, enabled=amp_enabled):
                out = model(batch_dev)
                logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
            logits_list.append(logits.detach().float().cpu().numpy())
            labels_list.append(batch_dev["label"].detach().float().cpu().numpy())
    logits_np = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 1), dtype=np.float32)
    labels_np = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0, 1), dtype=np.float32)
    return logits_np.astype(np.float32, copy=False), labels_np.astype(np.float32, copy=False)


def collect_prediction_rate_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ptm_types: List[str],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if y_true.size == 0 or y_prob.size == 0:
        return {
            "num_samples": int(y_true.shape[0]) if y_true.ndim > 0 else 0,
            "sampled_all_zero_rate": float("nan"),
            "predicted_all_zero_rate": float("nan"),
            "avg_true_positive_labels": float("nan"),
            "avg_pred_positive_labels": float("nan"),
            "per_label": {},
        }

    if thresholds is None:
        thresholds = np.full((y_true.shape[1],), 0.5, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    clipped_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (clipped_prob >= thresholds).astype(np.int32)
    known = y_true >= 0.0
    known_true = np.where(known, y_true, 0.0)
    true_label_counts = known_true.sum(axis=1)
    pred_label_counts = y_pred.sum(axis=1)
    per_label: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(ptm_types):
        known_i = known[:, idx]
        true_rate = float(np.mean(y_true[known_i, idx])) if known_i.any() else float("nan")
        pred_rate = float(np.mean(y_pred[known_i, idx])) if known_i.any() else float("nan")
        per_label[name] = {
            "threshold": float(thresholds[0, idx]),
            "known_count": int(known_i.sum()),
            "unknown_count": int((~known_i).sum()),
            "true_positive_rate": true_rate,
            "pred_positive_rate": pred_rate,
            "pred_minus_true_rate": float(pred_rate - true_rate) if np.isfinite(true_rate) and np.isfinite(pred_rate) else float("nan"),
            "prob_mean": float(np.mean(clipped_prob[known_i, idx])) if known_i.any() else float("nan"),
        }
    return {
        "num_samples": int(y_true.shape[0]),
        "known_label_fraction": float(known.mean()) if known.size else float("nan"),
        "sampled_all_zero_rate": float(np.mean(true_label_counts == 0)),
        "predicted_all_zero_rate": float(np.mean(pred_label_counts == 0)),
        "avg_true_positive_labels": float(np.mean(true_label_counts)),
        "avg_pred_positive_labels": float(np.mean(pred_label_counts)),
        "per_label": per_label,
    }


def format_prediction_rate_summary(diag: Dict[str, Any], ptm_types: List[str]) -> str:
    per_label = diag.get("per_label", {}) if isinstance(diag, dict) else {}
    parts: List[str] = []
    for name in ptm_types:
        stats = per_label.get(name)
        if not isinstance(stats, dict):
            continue
        parts.append(
            (
                f"{name}:thr={float(stats.get('threshold', float('nan'))):.3f},"
                f"true_pos={float(stats.get('true_positive_rate', float('nan'))):.4f},"
                f"pred_pos={float(stats.get('pred_positive_rate', float('nan'))):.4f},"
                f"delta={float(stats.get('pred_minus_true_rate', float('nan'))):+.4f}"
            )
        )
    return " | ".join(parts)


def get_lr_diagnostics(
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Dict[str, Any]:
    group_lrs: List[float] = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            group_lrs.append(float(group.get("lr", float("nan"))))
    scheduler_lrs: List[float] = []
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        try:
            scheduler_lrs = [float(v) for v in scheduler.get_last_lr()]
        except Exception:
            scheduler_lrs = []
    return {
        "group_lrs": group_lrs,
        "scheduler_lrs": scheduler_lrs,
        "primary_lr": float(group_lrs[0]) if group_lrs else float("nan"),
    }


def format_lr_summary(lr_diag: Dict[str, Any]) -> str:
    group_lrs = lr_diag.get("group_lrs", []) if isinstance(lr_diag, dict) else []
    scheduler_lrs = lr_diag.get("scheduler_lrs", []) if isinstance(lr_diag, dict) else []
    group_part = ",".join([f"g{idx}={float(lr):.6g}" for idx, lr in enumerate(group_lrs)]) if group_lrs else "<none>"
    scheduler_part = (
        ",".join([f"s{idx}={float(lr):.6g}" for idx, lr in enumerate(scheduler_lrs)])
        if scheduler_lrs
        else "<none>"
    )
    return f"optimizer=[{group_part}] scheduler=[{scheduler_part}]"


class DDPWithAttributes(torch.nn.parallel.DistributedDataParallel):
    """DDP that forwards missing attributes to the wrapped module.

    The training loop inspects attributes such as ``base_model``,
    ``use_last_foreground`` and ``get_cosine_classifier`` on the model.  A
    plain DDP wrapper only exposes its own submodule, so those lookups would
    silently fall back to defaults and disable LaST/UTA bookkeeping.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is not None:
                return getattr(module, name)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task multi-label lysine PTM model")
    parser.add_argument("--config_path", type=str, required=True, help="JSON config path")
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--json_data_path", type=str, default=None, help="Single JSON file with train/val/test or samples")
    parser.add_argument("--train_json_path", type=str, default=None, help="Train JSON file")
    parser.add_argument("--val_json_path", type=str, default=None, help="Validation JSON file")
    parser.add_argument("--test_json_path", type=str, default=None, help="Test JSON file")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path; resumes epoch/optimizer/scheduler state")
    parser.add_argument("--warm_start_from", type=str, default=None, help="Checkpoint path used only for model weights; starts training from epoch 1")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Run N train batches per rank for speed benchmarking, then exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading and paths without training")
    parser.add_argument(
        "--model-smoke-test",
        action="store_true",
        help="With --dry-run, execute one real collated batch through the model without training",
    )
    parser.add_argument("--ptm_types", type=str, default=None, help="Comma separated PTM subset, e.g. ubiquitination,sumoylation")
    parser.add_argument("--validation_split_case", type=str, default="full", choices=["full", "small"], help="Use full or small predefined split")
    parser.add_argument("--calibration_only", action="store_true", help="Run post-hoc calibration on an existing output_dir")
    parser.add_argument("--run_calibration_unit_tests", action="store_true", help="Run lightweight calibration unit tests and exit")
    parser.add_argument("--eval_checkpoint", type=str, default=None, choices=["best", "last"], help="Which checkpoint to use for post-training evaluation")
    parser.add_argument("--scheduler_step_unit", type=str, default=None, choices=["step", "epoch"], help="Whether to step the LR scheduler per optimizer step or per epoch")
    return parser.parse_args()


def apply_stage_overrides(config: Dict[str, Any], config_path: Path, stage: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    if not stage:
        return config
    stage_key = stage
    if stage_key not in config and stage == "small_batch":
        stage_key = "small_batch"
    overrides = config.get(stage_key)
    if not isinstance(overrides, dict):
        return config
    cfg = copy.deepcopy(config)
    stage_cfg = copy.deepcopy(overrides)
    if "max_epochs" in stage_cfg:
        stage_cfg["num_epochs"] = stage_cfg.pop("max_epochs")
    data_dir = config_path.resolve().parent
    train_path = stage_cfg.pop("train_data_path", None)
    val_path = stage_cfg.pop("val_data_path", None)
    test_path = stage_cfg.pop("test_data_path", None)
    cfg.update(stage_cfg)
    if train_path:
        args.train_json_path = _resolve_config_path(train_path, data_dir, must_exist=True)
    if val_path:
        args.val_json_path = _resolve_config_path(val_path, data_dir, must_exist=True)
    if test_path:
        args.test_json_path = _resolve_config_path(test_path, data_dir, must_exist=True)
    return cfg


def _load_splits(args: argparse.Namespace, config: Dict[str, Any]) -> SplitData:
    ptm_types = config.get("multitask_ptm_types", DEFAULT_PTM_TYPES)
    feature_sources = build_feature_sources(config, ptm_types)
    single_task_known_only = bool(config.get("single_task_known_only", False)) and len(ptm_types) == 1

    def load_csv_splits(train_csv: Path, val_csv: Optional[Path], test_csv: Path, source_path: str, split_policy: Dict[str, Any]) -> SplitData:
        train_records = load_split_csv(train_csv, ptm_types, known_only=single_task_known_only)
        apply_task_loss_weight_csvs(
            train_records,
            ptm_types,
            config.get("train_task_loss_weight_csvs") or config.get("train_task_loss_weight_csv"),
            logger=logging.getLogger("multitask_lysine"),
        )
        val_records = load_split_csv(val_csv, ptm_types, known_only=single_task_known_only) if val_csv is not None else []
        test_records = load_split_csv(test_csv, ptm_types, known_only=single_task_known_only)
        return SplitData(
            train=train_records,
            val=val_records,
            test=test_records,
            feature_sources=feature_sources,
            source_path=source_path,
            split_policy=split_policy,
        )

    config_train_csv = config.get("train_data_path")
    config_val_csv = config.get("val_data_path")
    config_test_csv = config.get("test_data_path")
    if config_train_csv and config_test_csv:
        train_path = Path(str(config_train_csv))
        val_path = Path(str(config_val_csv)) if config_val_csv else None
        test_path = Path(str(config_test_csv))
        if train_path.suffix.lower() == ".csv" and test_path.suffix.lower() == ".csv":
            return load_csv_splits(
                train_path,
                val_path if val_path and val_path.suffix.lower() == ".csv" else None,
                test_path,
                source_path="config_train_val_test_data_path",
                split_policy={"source": "config_train_data_path/val_data_path/test_data_path"},
            )

    predefined_splits = config.get("predefined_splits")
    active_split = str(config.get("active_split", "")).strip()
    if isinstance(predefined_splits, dict) and active_split and isinstance(predefined_splits.get(active_split), dict):
        chosen = predefined_splits[active_split]
        train_csv = chosen.get("train_csv")
        val_csv = chosen.get("val_csv")
        test_csv = chosen.get("test_csv")
        if train_csv and val_csv and test_csv:
            return load_csv_splits(
                Path(str(train_csv)),
                Path(str(val_csv)),
                Path(str(test_csv)),
                source_path=f"predefined_splits:{active_split}",
                split_policy={"source": "predefined_splits", "active_split": active_split, "ratio": chosen.get("ratio", "")},
            )

    validation_split_cfg = config.get("validation_split")
    if isinstance(validation_split_cfg, dict):
        case_key = "small" if str(args.validation_split_case).lower().startswith("small") else "full"
        chosen = validation_split_cfg.get(case_key) if isinstance(validation_split_cfg.get(case_key), dict) else validation_split_cfg
        train_csv = chosen.get("train_csv")
        val_csv = chosen.get("val_csv")
        test_csv = chosen.get("test_csv")
        if train_csv and val_csv and test_csv:
            return load_csv_splits(
                Path(train_csv),
                Path(val_csv),
                Path(test_csv),
                source_path=f"validation_split:{case_key}",
                split_policy={"source": "predefined_csv_split", "case": case_key, "ratio": chosen.get("ratio", "8:1:1")},
            )

    if args.train_json_path and args.test_json_path:
        train_path = Path(args.train_json_path)
        test_path = Path(args.test_json_path)
        val_path = Path(args.val_json_path) if args.val_json_path else None
        if train_path.suffix.lower() == ".csv" and test_path.suffix.lower() == ".csv":
            train_raw = load_split_csv(train_path, ptm_types)
            apply_task_loss_weight_csvs(
                train_raw,
                ptm_types,
                config.get("train_task_loss_weight_csvs") or config.get("train_task_loss_weight_csv"),
                logger=logging.getLogger("multitask_lysine"),
            )
            test_raw = load_split_csv(test_path, ptm_types)
            val_raw = load_split_csv(val_path, ptm_types) if val_path and val_path.suffix.lower() == ".csv" else []
        else:
            train_raw = load_json_records(train_path)
            test_raw = load_json_records(test_path)
            val_raw = load_json_records(val_path) if val_path else []

        if not isinstance(train_raw, list):
            raise ValueError("train_json_path must point to a JSON list of samples")
        if not isinstance(test_raw, list):
            raise ValueError("test_json_path must point to a JSON list of samples")
        if val_raw and not isinstance(val_raw, list):
            raise ValueError("val_json_path must point to a JSON list of samples")

        return SplitData(
            train=train_raw,
            val=val_raw,
            test=test_raw,
            feature_sources=feature_sources or None,
            source_path="explicit_train_val_test_json",
            split_policy={"source": "explicit_train_json_path/test_json_path"},
        )

    data_path = args.json_data_path or config.get("json_data_path")
    if not data_path:
        raise ValueError("Provide --json_data_path or --train_json_path/--test_json_path")

    raw = load_json_records(Path(data_path))
    if isinstance(raw, dict) and isinstance(raw.get("train"), list):
        return SplitData(
            train=raw.get("train", []),
            val=raw.get("val", []),
            test=raw.get("test", []),
            feature_sources=feature_sources or raw.get("feature_sources", {}),
            source_path=str(Path(data_path).resolve()),
            split_policy=raw.get("split_policy"),
        )

    split = split_records_from_json(
        raw,
        val_ratio=float(config.get("val_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.1)),
        seed=int(config.get("seed", args.seed)),
    )
    split.feature_sources = feature_sources or None
    split.source_path = str(Path(data_path).resolve())
    split.split_policy = {
        "source": "random_split_from_samples_or_data",
        "seed": int(config.get("seed", args.seed)),
        "val_ratio": float(config.get("val_ratio", 0.1)),
        "test_ratio": float(config.get("test_ratio", 0.1)),
    }
    return split


def _effective_loader_workers(requested_workers: int, logger: logging.Logger) -> int:
    cpu_count = max(1, int(multiprocessing.cpu_count()))
    recommended_cap = max(1, cpu_count - 2)
    effective = int(max(0, min(int(requested_workers), recommended_cap)))
    if effective != int(requested_workers):
        logger.info(
            "Adjusted dataloader_num_workers from %d to %d based on CPU capacity (cpu_count=%d).",
            int(requested_workers),
            int(effective),
            int(cpu_count),
        )
    return effective


def benchmark_loader_first_batch(loader: DataLoader, stage_name: str, enabled: bool, logger: logging.Logger) -> None:
    if not enabled:
        return
    try:
        t0 = time.perf_counter()
        itr = iter(loader)
        _ = next(itr)
        elapsed = float(time.perf_counter() - t0)
        logger.info("PERF dataloader_first_batch stage=%s elapsed=%.3fs", stage_name, elapsed)
    except StopIteration:
        logger.info("PERF dataloader_first_batch stage=%s skipped_empty_loader=true", stage_name)
    except Exception as exc:
        logger.warning("PERF dataloader_first_batch stage=%s failed: %s", stage_name, str(exc))


def load_runtime_setup(args: argparse.Namespace) -> RuntimeSetup:
    config_path = Path(args.config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            raw_config = yaml.safe_load(f)
        else:
            raw_config = json.load(f)
        config = normalize_and_validate_config(raw_config, config_path)
    config = apply_stage_overrides(config, config_path, args.stage, args)

    seed = int(config.get("seed", args.seed))
    cudnn_deterministic = bool(config.get("cudnn_deterministic", False))
    cudnn_benchmark = bool(config.get("cudnn_benchmark", not cudnn_deterministic))
    set_seed(seed, cudnn_deterministic=cudnn_deterministic, cudnn_benchmark=cudnn_benchmark)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(config.get("output_dir", "multitask_lysine_output"))
    logger = setup_logger(output_dir)
    enable_startup_profile = bool(config.get("enable_startup_profile", True))
    profiler = StageProfiler(enabled=enable_startup_profile, logger=logger)
    use_mlflow = initialize_mlflow_run(config, output_dir, logger)

    return RuntimeSetup(
        config_path=config_path,
        config=config,
        seed=seed,
        output_dir=output_dir,
        logger=logger,
        profiler=profiler,
        use_mlflow=use_mlflow,
    )


def initialize_mlflow_run(config: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> bool:
    mlflow_cfg = config.get("mlflow") if isinstance(config.get("mlflow"), dict) else {}
    use_mlflow = bool(mlflow_cfg.get("enabled", False))
    if use_mlflow and not HAS_MLFLOW:
        logger.warning("MLflow enabled in config but mlflow is unavailable. Disable mlflow or install the dependency.")
        return False
    if not use_mlflow:
        return False

    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
    experiment_name = str(mlflow_cfg.get("experiment_name") or config.get("experiment_name", "multitask_lysine"))
    mlflow.set_experiment(experiment_name)
    run_name = str(mlflow_cfg.get("run_name") or f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    tags = mlflow_cfg.get("tags") if isinstance(mlflow_cfg.get("tags"), dict) else {}
    # Guard against accidentally resuming an old run (e.g., inherited MLFLOW_RUN_ID),
    # which can cause "Changing param values is not allowed" failures.
    if mlflow.active_run() is not None:
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.start_run(run_name=run_name, tags=tags)
    for key, value in flatten_config_for_logging(config).items():
        if str(key) == "output_dir":
            continue
        try:
            mlflow.log_param(key, value)
        except Exception:
            try:
                mlflow.log_param(key, str(value))
            except Exception as exc:
                logger.warning("MLflow log_param skipped for key=%s: %s", str(key), str(exc))
    try:
        mlflow.log_param("output_dir", str(output_dir))
    except Exception as exc:
        logger.warning("MLflow log_param skipped for key=output_dir: %s", str(exc))
        mlflow.set_tag("output_dir", str(output_dir))
    return True


def resolve_ptm_types(config: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> List[str]:
    ptm_types = config.get("multitask_ptm_types", DEFAULT_PTM_TYPES)
    if args.ptm_types:
        requested = [sanitize_name(x) for x in str(args.ptm_types).split(",") if str(x).strip()]
        ptm_types = [p for p in ptm_types if sanitize_name(p) in set(requested)]
        if not ptm_types:
            raise ValueError(f"No valid PTM types selected by --ptm_types={args.ptm_types}")
        config["multitask_ptm_types"] = ptm_types
    logger.info("Configured PTM type count=%d.", len(ptm_types))
    return ptm_types


def build_loader_bundle(
    config: Dict[str, Any],
    seed: int,
    ptm_types: List[str],
    datasets: DatasetBundle,
    logger: logging.Logger,
    profiler: StageProfiler,
) -> LoaderBundle:
    # ``batch_size`` is the requested effective batch.  Large SE(3)+Dual
    # batches can exceed a single GPU's activation memory, so an optional
    # loader micro-batch is combined with grad_accum_steps to preserve that
    # effective batch without changing the optimization schedule.
    batch_size = int(config.get("batch_size", 64))
    loader_batch_size = int(config.get("loader_batch_size", batch_size))
    if loader_batch_size <= 0 or loader_batch_size > batch_size:
        loader_batch_size = batch_size
    if loader_batch_size != batch_size:
        logger.info(
            "Using loader micro-batch=%d with effective batch_size=%d; grad_accum_steps=%d",
            loader_batch_size,
            batch_size,
            int(config.get("grad_accum_steps", 1)),
        )
    requested_workers = int(config.get("dataloader_num_workers", 4))
    num_workers = _effective_loader_workers(requested_workers, logger)
    prefetch_factor = int(config.get("dataloader_prefetch_factor", 2))
    persistent_workers = bool(config.get("dataloader_persistent_workers", num_workers > 0))
    pin_memory = bool(config.get("pin_memory", True) and torch.cuda.is_available())
    benchmark_first_batch = bool(config.get("benchmark_dataloader_first_batch", True))

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": multilabel_collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        common_loader_kwargs["persistent_workers"] = persistent_workers

    use_distribution_batch_sampler = bool(config.get("use_distribution_batch_sampler", True))
    use_min_label_coverage_sampler = bool(config.get("use_min_label_coverage_sampler", False))
    if bool(config.get("use_weighted_sampler", False)):
        logger.info("Weighted sampler is disabled. Falling back to configured batch sampling strategy.")
    train_batch_sampler = None
    # ShardedProteinSiteDataset (feature_store_v2) also stores samples contiguously by shard;
    # global shuffle makes every batch a random HDD seek, so use the same block-shuffle sampler.
    use_block_shuffle_sampler = bool(config.get("use_block_shuffle_sampler", False)) and (
        isinstance(datasets.train_set, PreprocessedTensorLMDBDataset)
        or hasattr(datasets.train_set, "site_shard")
    )
    use_gpu_pipeline = bool(config.get("use_gpu_window_pipeline", False))
    gpu_pipeline_device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
    )
    if use_gpu_pipeline:
        import sys as _sys
        _scripts = Path(__file__).resolve().parent / "vendor" / "feature_store"
        if str(_scripts) not in _sys.path:
            _sys.path.insert(0, str(_scripts))
        from gpu_shard_loader import GpuShardBatchLoader
        use_distribution_batch_sampler = False
        use_min_label_coverage_sampler = False
        use_block_shuffle_sampler = False
        train_rank = int(os.environ.get("RANK", "0"))
        train_world = int(os.environ.get("WORLD_SIZE", "1"))
        train_loader = profiler.run(
            "build_train_loader",
            lambda: GpuShardBatchLoader(
                config["feature_store_v2_root"], "train", loader_batch_size,
                gpu_pipeline_device, seed=seed, rank=train_rank, world_size=train_world,
                shuffle=True, dataset=datasets.train_set, log_fn=logger.info,
                prefetch_shards=bool(config.get("gpu_prefetch_shards", True)),
            ),
        )
        train_batch_sampler = train_loader
        logger.info(
            "Using GPU shard loader: shard-resident features, GPU window/graph construction, rank=%d world=%d batch=%d",
            train_rank, train_world, loader_batch_size,
        )
    elif use_block_shuffle_sampler:
        use_distribution_batch_sampler = False
        use_min_label_coverage_sampler = False
        train_batch_sampler = BlockShuffleBatchSampler(
            length=len(datasets.train_set),
            batch_size=loader_batch_size,
            block_size=int(config.get("block_shuffle_size", 8192)),
            seed=seed,
            drop_last=False,
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
        train_loader = profiler.run(
            "build_train_loader",
            lambda: DataLoader(
                datasets.train_set,
                batch_sampler=train_batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=multilabel_collate_fn,
                **({"prefetch_factor": max(1, prefetch_factor), "persistent_workers": persistent_workers} if num_workers > 0 else {}),
            ),
        )
        logger.info(
            "Using block-shuffle sampler: block_size=%d batch_size=%d blocks=%d",
            train_batch_sampler.block_size,
            train_batch_sampler.batch_size,
            (len(datasets.train_set) + train_batch_sampler.block_size - 1) // train_batch_sampler.block_size,
        )
    elif use_distribution_batch_sampler:
        sampler_label_boost_by_label = (
            {str(k): float(v) for k, v in config.get("sampler_label_boost_by_label", {}).items()}
            if isinstance(config.get("sampler_label_boost_by_label"), dict)
            else {}
        )
        sampler_min_label_ratio_by_label = (
            {str(k): float(v) for k, v in config.get("sampler_min_label_ratio_by_label", {}).items()}
            if isinstance(config.get("sampler_min_label_ratio_by_label"), dict)
            else {}
        )
        sampler_max_label_ratio_by_label = (
            {str(k): float(v) for k, v in config.get("sampler_max_label_ratio_by_label", {}).items()}
            if isinstance(config.get("sampler_max_label_ratio_by_label"), dict)
            else {}
        )
        train_batch_sampler = DistributionAlignedBatchSampler(
            dataset=datasets.train_set,
            batch_size=loader_batch_size,
            alpha=float(config.get("sampler_alpha", 1.0)),
            all_zero_weight=float(config.get("sampler_all_zero_weight", 1.0)),
            label_boost_by_label=sampler_label_boost_by_label,
            min_label_ratio_by_label=sampler_min_label_ratio_by_label,
            max_label_ratio_by_label=sampler_max_label_ratio_by_label,
            target_refine_max_iters=int(config.get("sampler_target_refine_max_iters", 32)),
            seed=seed,
            drop_last=False,
        )
        train_loader = profiler.run(
            "build_train_loader",
            lambda: DataLoader(
                datasets.train_set,
                batch_sampler=train_batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=multilabel_collate_fn,
                **({"prefetch_factor": max(1, prefetch_factor), "persistent_workers": persistent_workers} if num_workers > 0 else {}),
            ),
        )
        logger.info(
            "Using distribution-aligned batch sampler with inverse-frequency alpha=%.3f, all_zero_weight=%.3f, label_boosts=%s, min_label_ratios=%s and replacement-based batch composition.",
            train_batch_sampler.alpha,
            train_batch_sampler.all_zero_weight,
            sampler_label_boost_by_label,
            sampler_min_label_ratio_by_label,
        )
        logger.info(
            "Sampler distributions | dataset=%s | target=%s",
            {ptm: float(train_batch_sampler.dataset_label_distribution[idx]) for idx, ptm in enumerate(ptm_types)},
            {ptm: float(train_batch_sampler.label_distribution[idx]) for idx, ptm in enumerate(ptm_types)},
        )
    elif use_min_label_coverage_sampler:
        coverage_labels = list(config.get("min_label_coverage_labels", []))
        if not coverage_labels:
            coverage_labels = list(config.get("minority_task_boost_labels", []))
        train_batch_sampler = MinimumLabelCoverageBatchSampler(
            dataset=datasets.train_set,
            batch_size=loader_batch_size,
            coverage_labels=coverage_labels,
            min_positives=int(config.get("min_label_coverage_min_positives", 1)),
            seed=seed,
            drop_last=False,
        )
        train_loader = profiler.run(
            "build_train_loader",
            lambda: DataLoader(
                datasets.train_set,
                batch_sampler=train_batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=multilabel_collate_fn,
                **({"prefetch_factor": max(1, prefetch_factor), "persistent_workers": persistent_workers} if num_workers > 0 else {}),
            ),
        )
        logger.info(
            "Using minimum-label-coverage batch sampler: labels=%s min_positives=%d. Natural dataset distribution is preserved except for batches missing a requested label.",
            train_batch_sampler.coverage_label_names,
            train_batch_sampler.min_positives,
        )
        logger.info(
            "Sampler distributions | dataset=%s | target=%s",
            {ptm: float(train_batch_sampler.dataset_label_distribution[idx]) for idx, ptm in enumerate(ptm_types)},
            {ptm: float(train_batch_sampler.label_distribution[idx]) for idx, ptm in enumerate(ptm_types)},
        )
    else:
        train_loader = profiler.run(
            "build_train_loader",
            lambda: DataLoader(
                datasets.train_set,
                shuffle=not isinstance(datasets.train_set, (PreprocessedTensorLMDBDataset, ProteinDedupLMDBDataset)),
                batch_size=loader_batch_size,
                **common_loader_kwargs,
            ),
        )
    target_distribution = (
        train_batch_sampler.label_distribution.copy()
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "label_distribution")
        else profiler.run(
            "compute_target_distribution",
            lambda: _positive_label_distribution(datasets.train_set.get_label_matrix())
            if len(datasets.train_set) > 0
            else np.zeros((len(ptm_types),), dtype=np.float32),
        )
    )
    train_distribution_monitor = BatchDistributionMonitor(
        ptm_types=ptm_types,
        target_distribution=target_distribution,
        log_interval=int(config.get("batch_distribution_log_interval", 50)),
        relative_tolerance=float(config.get("batch_distribution_relative_tolerance", 0.25)),
        min_abs_tolerance=float(config.get("batch_distribution_min_abs_tolerance", 0.02)),
        enable_batch_logging=bool(config.get("batch_distribution_step_log", False)),
    )
    if use_gpu_pipeline:
        from gpu_shard_loader import GpuShardBatchLoader
        val_loader = profiler.run(
            "build_val_loader",
            lambda: GpuShardBatchLoader(
                config["feature_store_v2_root"], "val", loader_batch_size,
                gpu_pipeline_device, seed=seed, rank=0, world_size=1,
                shuffle=False, dataset=datasets.val_set, log_fn=logger.info,
                prefetch_shards=bool(config.get("gpu_prefetch_shards", True)),
            ),
        )
        test_loader = profiler.run(
            "build_test_loader",
            lambda: GpuShardBatchLoader(
                config["feature_store_v2_root"], "test", loader_batch_size,
                gpu_pipeline_device, seed=seed, rank=0, world_size=1,
                shuffle=False, dataset=datasets.test_set, log_fn=logger.info,
                prefetch_shards=bool(config.get("gpu_prefetch_shards", True)),
            ),
        )
    else:
        val_loader = profiler.run(
            "build_val_loader",
            lambda: DataLoader(
                datasets.val_set,
                shuffle=False,
                batch_size=loader_batch_size,
                **common_loader_kwargs,
            ),
        )
        test_loader = profiler.run(
            "build_test_loader",
            lambda: DataLoader(
                datasets.test_set,
                shuffle=False,
                batch_size=loader_batch_size,
                **common_loader_kwargs,
            ),
        )
    benchmark_loader_first_batch(train_loader, "train", benchmark_first_batch, logger)
    benchmark_loader_first_batch(val_loader, "val", benchmark_first_batch, logger)

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_distribution_monitor=train_distribution_monitor,
        use_distribution_batch_sampler=use_distribution_batch_sampler,
        train_batch_sampler=train_batch_sampler,
    )


def build_runtime_device_and_amp(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> RuntimeDeviceAmp:
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        if bool(config.get("use_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(config.get("matmul_precision", "high")))

    use_amp = bool(config.get("use_amp", device.type == "cuda"))
    amp_dtype = resolve_amp_dtype(config)
    amp_enabled = bool(use_amp and device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
    requested_scaler = config.get("amp_use_grad_scaler")
    if requested_scaler is None:
        scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    else:
        scaler_enabled = bool(amp_enabled and bool(requested_scaler))

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    logger.info(
        "AMP enabled: %s dtype=%s grad_scaler=%s",
        str(amp_enabled),
        str(amp_dtype).replace("torch.", ""),
        str(scaler_enabled),
    )
    return RuntimeDeviceAmp(device=device, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler)


def build_training_components(
    args: argparse.Namespace,
    config: Dict[str, Any],
    ptm_types: List[str],
    train_loader: DataLoader,
    train_set: "MultiLabelPTMJsonDataset",
    output_dir: Path,
    logger: logging.Logger,
    runtime_amp: RuntimeDeviceAmp,
) -> Tuple[TrainingComponents, TrainingState]:
    model = build_model(config, num_tasks=len(ptm_types)).to(runtime_amp.device)
    compile_requested = bool(config.get("use_torch_compile", False))
    disable_compile_env = str(os.environ.get("DISABLE_TORCH_COMPILE", "")).strip().lower()
    if disable_compile_env in {"1", "true", "yes", "y", "on"}:
        compile_requested = False
        logger.info("Torch compile disabled via DISABLE_TORCH_COMPILE=%s", str(os.environ.get("DISABLE_TORCH_COMPILE")))
    if str(args.stage or "").lower() == "small_batch" and not bool(config.get("torch_compile_small_batch", False)):
        compile_requested = False
        logger.info("Torch compile disabled for stage=small_batch to avoid autotune warmup overhead.")
    if compile_requested and runtime_amp.device.type != "cuda":
        compile_requested = False
        logger.info("Torch compile disabled because device=%s (only enabled on CUDA for stability).", runtime_amp.device)
    if compile_requested and bool(runtime_amp.use_amp and runtime_amp.device.type == "cuda") and runtime_amp.amp_dtype == torch.float16 and not bool(
        config.get("torch_compile_allow_fp16", False)
    ):
        compile_requested = False
        logger.warning(
            "Torch compile disabled because amp_dtype=float16 can segfault with inductor on some systems. "
            "Set torch_compile_allow_fp16=true to force enable."
        )
    if compile_requested and hasattr(torch, "compile"):
        compile_mode = str(config.get("torch_compile_mode", "default"))
        compile_backend = str(config.get("torch_compile_backend", "inductor"))
        if compile_mode == "max-autotune" and not bool(config.get("allow_max_autotune", False)):
            compile_mode = "reduce-overhead"
        try:
            model = torch.compile(model, mode=compile_mode, backend=compile_backend)
            logger.info("Torch compile enabled: backend=%s mode=%s", compile_backend, compile_mode)
        except Exception as exc:
            logger.warning("Torch compile failed, fallback to eager mode: %s", str(exc))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = DDPWithAttributes(
            model,
            device_ids=[runtime_amp.device.index],
            output_device=runtime_amp.device.index,
            find_unused_parameters=bool(config.get("ddp_find_unused_parameters", True)),
            static_graph=bool(config.get("ddp_static_graph", False)),
        )
        logger.info("DDP enabled: rank=%d world_size=%d device=%s",
                    int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")),
                    str(runtime_amp.device))

    pos_weight = compute_pos_weight(train_set).to(runtime_amp.device)
    if isinstance(config.get("pos_weight"), list):
        manual = np.asarray(config["pos_weight"], dtype=np.float32)
        if manual.shape[0] == len(ptm_types):
            pos_weight = torch.tensor(manual, dtype=torch.float32, device=runtime_amp.device)

    loss_type = str(config.get("loss_type", "bce")).lower()
    task_weights = np.ones((len(ptm_types),), dtype=np.float32)
    if isinstance(config.get("task_initial_weights"), list):
        manual_w = np.asarray(config.get("task_initial_weights"), dtype=np.float32)
        if manual_w.shape[0] == len(ptm_types):
            task_weights = np.clip(manual_w, 1e-4, None)
    elif isinstance(config.get("task_initial_weights"), dict):
        for idx, ptm in enumerate(ptm_types):
            if ptm in config["task_initial_weights"]:
                task_weights[idx] = max(1e-4, float(config["task_initial_weights"][ptm]))
    for boosted in config.get("minority_task_boost_labels", ["crotonylation", "methylation"]):
        if sanitize_name(boosted) in [sanitize_name(x) for x in ptm_types]:
            bidx = [sanitize_name(x) for x in ptm_types].index(sanitize_name(boosted))
            task_weights[bidx] = max(task_weights[bidx], float(config.get("minority_task_boost", 1.8)))
    weak_precision_labels = [sanitize_name(x) for x in config.get("weak_task_precision_labels", [])]
    weak_precision_boost = float(config.get("weak_task_precision_boost", 1.0))
    if weak_precision_boost > 1.0:
        for idx, ptm in enumerate(ptm_types):
            if sanitize_name(ptm) in weak_precision_labels:
                task_weights[idx] = max(task_weights[idx], weak_precision_boost)

    if loss_type == "dynamic_kg":
        from src.utils.dynamic_kg_loss import DynamicKGLoss
        criterion = DynamicKGLoss(
            num_tasks=len(ptm_types),
            ptm_types=ptm_types,
            gamma_neg=float(config.get("asl_gamma_neg", 3.0)),
            gamma_pos=float(config.get("asl_gamma_pos", 1.0)),
            clip=float(config.get("asl_clip", 0.05)),
            eps=float(config.get("asl_eps", 1e-8)),
            pos_weight=pos_weight,
            task_weights=task_weights.tolist(),
            kg_weight=float(config.get("dynamic_kg_weight", 0.1)),
            dynamic_weight=float(config.get("dynamic_kg_dynamic_weight", 0.5)),
            update_momentum=float(config.get("dynamic_kg_update_momentum", 0.9)),
            enable_dynamic_updates=bool(config.get("dynamic_kg_enable_updates", True)),
        ).to(runtime_amp.device)
        logger.info("Using advanced DynamicKGLoss (ASL + pos_weight + task weights + Knowledge Graph Constraints)")
    else:
        criterion = AdaptiveTaskWeightedLoss(
            loss_type=loss_type,
            num_tasks=len(ptm_types),
            pos_weight=pos_weight,
            gamma_neg=float(config.get("asl_gamma_neg", 4.0)),
            gamma_pos=float(config.get("asl_gamma_pos", 1.0)),
            clip=float(config.get("asl_clip", 0.05)),
            eps=float(config.get("asl_eps", 1e-8)),
            task_weights=task_weights.tolist(),
            use_homoscedastic=bool(config.get("use_homoscedastic_task_weighting", True)),
            use_rank_loss=bool(config.get("use_rank_loss", False)),
            rank_loss_margin=float(config.get("rank_loss_margin", 0.3)),
            rank_loss_weight=float(config.get("rank_loss_weight", 0.5)),
            rank_loss_task_weights=config.get("rank_loss_task_weights"),
            rank_loss_hard_negative_labels=config.get("rank_loss_hard_negative_labels", []),
            rank_loss_hard_negative_top_fraction=float(config.get("rank_loss_hard_negative_top_fraction", 1.0)),
            ptm_types=ptm_types,
            pu_task_labels=config.get("pu_task_labels", []),
            pu_positive_priors=config.get("pu_positive_priors"),
            pu_reliable_negative_weight=float(config.get("pu_reliable_negative_weight", 0.35)),
            pu_unlabeled_weight=float(config.get("pu_unlabeled_weight", 1.0)),
            pu_nn_beta=float(config.get("pu_nn_beta", 0.0)),
            use_label_state_loss=bool(config.get("use_label_state_loss", False)),
            state_reliable_negative_weight=float(config.get("state_reliable_negative_weight", config.get("pu_reliable_negative_weight", 0.35))),
            state_observed_negative_weight=float(config.get("state_observed_negative_weight", 0.7)),
            state_unknown_weight=float(config.get("state_unknown_weight", 0.0)),
            state_soft_positive_target=float(config.get("state_soft_positive_target", 0.7)),
            state_soft_positive_weight=float(config.get("state_soft_positive_weight", 0.5)),
            state_conflict_weight=float(config.get("state_conflict_weight", 0.0)),
            log_var_min=float(config.get("homoscedastic_log_var_min", float("-inf"))),
            log_var_max=float(config.get("homoscedastic_log_var_max", float("inf"))),
            log_var_freeze_epochs=int(config.get("homoscedastic_freeze_epochs", 0)),
        ).to(runtime_amp.device)
        logger.info("Using adaptive multi-task loss: type=%s homoscedastic=%s", loss_type, str(bool(config.get("use_homoscedastic_task_weighting", True))))
    logger.info(
        "Task initial weights: %s",
        ", ".join([f"{ptm}:{task_weights[idx]:.3f}" for idx, ptm in enumerate(ptm_types)]),
    )
    if getattr(criterion, "log_vars", None) is not None:
        logger.info(
            "Homoscedastic controls: bounds=[%.3f, %.3f] freeze_epochs=%d criterion_lr_scale=%.4f",
            float(criterion.log_var_min), float(criterion.log_var_max),
            int(criterion.log_var_freeze_epochs), float(config.get("criterion_lr_scale", 1.0)),
        )

    optimizer = create_optimizer(config, model, extra_params=list(criterion.parameters()))
    total_steps = max(1, int(config.get("num_epochs", 50)) * math.ceil(max(1, len(train_loader)) / max(1, int(config.get("grad_accum_steps", 1)))))
    scheduler = create_scheduler(config, optimizer, total_steps)
    scheduler_step_unit = str(config.get("scheduler_step_unit", "")).lower().strip()
    if args.scheduler_step_unit:
        scheduler_step_unit = str(args.scheduler_step_unit).lower().strip()
    if not scheduler_step_unit:
        scheduler_name = str(config.get("scheduler", "none")).lower()
        scheduler_step_unit = "step" if scheduler_name == "cosine_with_warmup" else "epoch"
    else:
        scheduler_name = str(config.get("scheduler", "none")).lower()
    warmup_steps = int(config.get("warmup_steps", 0))
    logger.info(
        "Scheduler config: name=%s step_unit=%s total_steps=%d warmup_steps=%d %s",
        scheduler_name,
        scheduler_step_unit,
        total_steps,
        warmup_steps,
        format_lr_summary(get_lr_diagnostics(optimizer, scheduler)),
    )

    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_macro_auc = -1.0
    best_checkpoint_val_macro_f1 = float("nan")
    best_checkpoint_val_macro_auc = float("nan")
    best_thresholds = np.full((len(ptm_types),), 0.5, dtype=np.float32)
    early_stopping_metric = str(config.get("early_stopping_metric", "val_macro_f1")).lower().strip()
    early_stopping_mode = str(config.get("early_stopping_mode", "max")).lower().strip()
    best_monitor_value = float("inf") if early_stopping_mode == "min" else -1.0

    ckpt_last = output_dir / "checkpoint_last.pt"
    ckpt_best = output_dir / "checkpoint_best.pt"
    baseline_log_path = config.get("baseline_log_path")
    baseline_benchmark = parse_training_log_benchmark(Path(baseline_log_path)) if baseline_log_path else {}

    warm_start_from = args.warm_start_from or config.get("warm_start_from")
    if warm_start_from:
        warm_start_path = Path(warm_start_from)
        if warm_start_path.exists():
            checkpoint = torch.load(warm_start_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            load_shape_compatible_state_dict(
                model,
                state_dict,
                logger,
                context=f"Warm-start load from {warm_start_path}",
            )
            logger.info("Warm-started shape-compatible model weights from %s; epoch/optimizer/scheduler reset.", warm_start_path)
        else:
            logger.warning("Warm-start checkpoint not found: %s", warm_start_path)

    resume_from = args.resume_from or config.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            (
                start_epoch,
                best_monitor_value,
                best_checkpoint_val_macro_f1,
                best_checkpoint_val_macro_auc,
                loaded_monitor_metric,
                loaded_thr,
            ) = load_checkpoint(
                resume_path, model, optimizer, scheduler, criterion=criterion, scaler=runtime_amp.scaler
            )
            best_val_macro_f1 = max(best_val_macro_f1, best_checkpoint_val_macro_f1)
            best_val_macro_auc = max(best_val_macro_auc, best_checkpoint_val_macro_auc)
            if loaded_thr is not None and loaded_thr.shape[0] == len(ptm_types):
                best_thresholds = loaded_thr
            logger.info(
                "Resumed from %s at epoch %d with checkpoint monitor=%s value=%.5f",
                resume_path,
                start_epoch,
                loaded_monitor_metric,
                best_monitor_value,
            )
        else:
            logger.warning("Resume checkpoint not found: %s", resume_path)

    writer = None
    if int(os.environ.get("RANK", "0")) == 0 and bool(config.get("use_tensorboard", True)) and HAS_TB:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_macro_auc": []}
    num_epochs = int(config.get("num_epochs", config.get("epochs", 50)))
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    threshold_metric = str(config.get("threshold_metric", "f1"))
    sanity_max_train_batches = config.get("sanity_max_train_batches")
    sanity_max_val_batches = config.get("sanity_max_val_batches")
    sanity_max_test_batches = config.get("sanity_max_test_batches")
    grad_norm_log_interval = int(config.get("grad_norm_log_interval", 0))
    grad_norm_topk = int(config.get("grad_norm_topk", 5))
    grad_norm_warn_threshold = float(config.get("grad_norm_warn_threshold", 0.0))
    log_every = int(config.get("log_every", 0))
    early_stopping = EarlyStopping(
        mode=str(config.get("early_stopping_mode", "max")),
        patience=int(config.get("early_stopping_patience", 12)),
        min_delta=float(config.get("early_stopping_min_delta", 0.0)),
    ) if bool(config.get("use_early_stopping", True)) else None
    overfit_gap_threshold = float(config.get("overfit_gap_threshold", 0.05))
    validate_every_n_epochs = max(1, int(config.get("validate_every_n_epochs", 1)))
    checkpoint_interval_epochs = max(1, int(config.get("checkpoint_interval_epochs", 1)))
    save_checkpoints = bool(config.get("save_checkpoints", True))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    foreground_warmup_epochs = max(0, int(config.get("foreground_warmup_epochs", 0)))
    gradient_conflict_log_interval = max(0, int(config.get("gradient_conflict_log_interval", 0)))

    sanity_max_train_batches = int(sanity_max_train_batches) if sanity_max_train_batches is not None else None
    sanity_max_val_batches = int(sanity_max_val_batches) if sanity_max_val_batches is not None else None
    sanity_max_test_batches = int(sanity_max_test_batches) if sanity_max_test_batches is not None else None
    if any(v is not None for v in [sanity_max_train_batches, sanity_max_val_batches, sanity_max_test_batches]):
        logger.info(
            "Sanity batch limits enabled: train=%s val=%s test=%s",
            str(sanity_max_train_batches),
            str(sanity_max_val_batches),
            str(sanity_max_test_batches),
        )
    if grad_norm_log_interval > 0:
        logger.info(
            "Gradient norm logging enabled: interval=%d topk=%d warn_threshold=%.4f",
            grad_norm_log_interval,
            grad_norm_topk,
            grad_norm_warn_threshold,
        )
    logger.info("Threshold smoothing disabled. Final threshold search will run once on the validation split after training.")
    logger.info(
        "Validation/checkpoint frequency: validate_every_n_epochs=%d checkpoint_interval_epochs=%d grad_clip_norm=%.4f early_stopping_metric=%s",
        validate_every_n_epochs,
        checkpoint_interval_epochs,
        grad_clip_norm,
        early_stopping_metric,
    )
    if not bool(config.get("save_checkpoints", True)):
        logger.info("Checkpoint saving disabled by config (save_checkpoints=false).")
    if foreground_warmup_epochs > 0 and hasattr(model, "set_last_module_trainable"):
        logger.info("Foreground warmup enabled: freeze LaSt modules for first %d epoch(s).", foreground_warmup_epochs)

    components = TrainingComponents(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_unit=scheduler_step_unit,
        writer=writer,
        early_stopping=early_stopping,
        baseline_benchmark=baseline_benchmark,
        ckpt_last=ckpt_last,
        ckpt_best=ckpt_best,
        best_model_path=output_dir / "best_model.pth",
        num_epochs=num_epochs,
        grad_accum_steps=grad_accum_steps,
        threshold_metric=threshold_metric,
        sanity_max_train_batches=sanity_max_train_batches,
        sanity_max_val_batches=sanity_max_val_batches,
        sanity_max_test_batches=sanity_max_test_batches,
        grad_norm_log_interval=grad_norm_log_interval,
        grad_norm_topk=grad_norm_topk,
        grad_norm_warn_threshold=grad_norm_warn_threshold,
        log_every=log_every,
        overfit_gap_threshold=overfit_gap_threshold,
        validate_every_n_epochs=validate_every_n_epochs,
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        save_checkpoints=save_checkpoints,
        grad_clip_norm=grad_clip_norm,
        foreground_warmup_epochs=foreground_warmup_epochs,
        gradient_conflict_log_interval=gradient_conflict_log_interval,
    )
    state = TrainingState(
        start_epoch=start_epoch,
        best_val_macro_f1=best_val_macro_f1,
        best_val_macro_auc=best_val_macro_auc,
        best_checkpoint_val_macro_f1=best_checkpoint_val_macro_f1,
        best_checkpoint_val_macro_auc=best_checkpoint_val_macro_auc,
        best_thresholds=best_thresholds,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        best_monitor_value=best_monitor_value,
        history=history,
        last_trainable_state=None,
        epoch_performance=[],
        summary_rows=[],
    )
    return components, state




def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.distributed.is_available():
            raise RuntimeError("WORLD_SIZE>1 but torch.distributed is unavailable")
        from datetime import timedelta
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        args.gpu = local_rank
        torch.cuda.set_device(local_rank)
    if args.run_calibration_unit_tests:
        run_calibration_unit_tests()
        print("Calibration unit tests passed.")
        return
    runtime = load_runtime_setup(args)
    config = runtime.config
    output_dir = runtime.output_dir
    logger = runtime.logger
    profiler = runtime.profiler
    use_mlflow = runtime.use_mlflow
    seed = runtime.seed

    ptm_types = resolve_ptm_types(config, args, logger)
    data_artifacts = prepare_data_artifacts(
        args,
        config,
        ptm_types,
        output_dir,
        logger,
        profiler,
        services=DataPreparationServices(
            load_splits=_load_splits,
            validate_split_integrity=validate_predefined_split_integrity,
            filter_records_with_available_pdb=filter_records_with_available_pdb,
            filter_records_by_structure_feature_availability=filter_records_by_structure_feature_availability,
            repair_or_filter_sequence_windows=repair_or_filter_sequence_windows,
            filter_records_by_feature_quality=filter_records_by_feature_quality,
            validate_split_ptm_presence=validate_split_ptm_presence,
            motif_aware_negative_sampling=motif_aware_negative_sampling,
            prepare_data_integrity_report=prepare_data_integrity_report,
        ),
    )
    datasets = build_dataset_bundle(config, data_artifacts, profiler)
    train_set = datasets.train_set
    val_set = datasets.val_set
    test_set = datasets.test_set

    if use_mlflow:
        mlflow.log_param("train_samples", len(train_set))
        mlflow.log_param("val_samples", len(val_set))
        mlflow.log_param("test_samples", len(test_set))

    if args.dry_run:
        if args.model_smoke_test:
            if len(train_set) == 0:
                raise ValueError("Model smoke test requires at least one training sample.")
            runtime_amp = build_runtime_device_and_amp(args, config, logger)
            model = build_model(config, num_tasks=len(ptm_types)).to(runtime_amp.device)
            model.eval()
            batch = to_device(multilabel_collate_fn([train_set[0]]), runtime_amp.device)
            parameter_ids_before = {id(param) for param in model.parameters()}
            with torch.no_grad():
                outputs = model(batch)
                model.train()
                base_model = getattr(model, "base_model", model)
                original_modality_dropout = float(getattr(base_model, "modality_dropout", 0.0))
                base_model.modality_dropout = 1.0
                train_outputs = model(batch)
                base_model.modality_dropout = original_modality_dropout
                model.eval()
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
            train_logits = train_outputs.get("logits") if isinstance(train_outputs, dict) else train_outputs
            expected_shape = (1, len(ptm_types))
            for mode, candidate in (("eval", logits), ("train", train_logits)):
                if not isinstance(candidate, torch.Tensor):
                    raise TypeError(f"Model smoke test {mode} forward did not return tensor logits.")
                if tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Model smoke test {mode} logits shape={tuple(candidate.shape)}, expected={expected_shape}."
                    )
                if not torch.isfinite(candidate).all():
                    raise ValueError(f"Model smoke test {mode} forward produced non-finite logits.")
            if parameter_ids_before != {id(param) for param in model.parameters()}:
                raise RuntimeError("Model parameters changed during forward; runtime module replacement is forbidden.")
            if bool(config.get("use_dense_structure_tower_with_graph", False)):
                contributions = outputs.get("module_contributions", {}) if isinstance(outputs, dict) else {}
                gate_keys = ("tri_gate_sequence_mean", "tri_gate_dense_structure_mean", "tri_gate_graph_mean")
                if not all(isinstance(contributions.get(key), torch.Tensor) for key in gate_keys):
                    raise RuntimeError("Dual-structure graph smoke test did not emit all tri-modal gate weights.")
                gate_sum = sum(float(contributions[key].detach().mean().item()) for key in gate_keys)
                if not math.isclose(gate_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError(f"Dual-structure graph gate weights must sum to 1, got {gate_sum:.6f}.")
            logger.info(
                "Model smoke test success: stage=%s logits_shape=%s finite=true parameters_stable=true",
                str(args.stage or "base"),
                str(tuple(logits.shape)),
            )
        logger.info("Dry-run success: config, split, feature source mapping and cache directory validated.")
        logger.info("Feature report directory: %s", str(data_artifacts.report_dir))
        profiler.save(output_dir / "startup_profile.json")
        export_startup_profile_comparison(output_dir, logger)
        if use_mlflow:
            mlflow.end_run()
        return
    loaders = build_loader_bundle(config, seed, ptm_types, datasets, logger, profiler)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader
    train_distribution_monitor = loaders.train_distribution_monitor
    runtime_amp = build_runtime_device_and_amp(args, config, logger)
    components, training_state = build_training_components(
        args=args,
        config=config,
        ptm_types=ptm_types,
        train_loader=train_loader,
        train_set=train_set,
        output_dir=output_dir,
        logger=logger,
        runtime_amp=runtime_amp,
    )
    model = components.model
    criterion = components.criterion
    optimizer = components.optimizer

    if int(getattr(args, "benchmark_steps", 0)) > 0:
        from src.utils.training_state import EpochOptions
        from src.utils.training_engine import run_epoch as _run_epoch
        steps = int(args.benchmark_steps)
        warmup_steps = max(1, min(5, steps))
        local_batch = int(config.get("loader_batch_size", config.get("batch_size", 1)))

        def _bench_epoch(n_batches: int):
            options = EpochOptions(
                grad_accum_steps=components.grad_accum_steps,
                max_batches=int(n_batches),
                scaler=runtime_amp.scaler,
                use_amp=runtime_amp.use_amp,
                amp_dtype=runtime_amp.amp_dtype,
                grad_clip_norm=components.grad_clip_norm,
                scheduler_step_unit=components.scheduler_step_unit,
                stage_name="train",
            )
            return _run_epoch(
                components.model, train_loader, components.optimizer,
                components.scheduler, components.criterion,
                runtime_amp.device, options,
            )

        _bench_epoch(warmup_steps)
        if distributed:
            torch.distributed.barrier()
        started = time.time()
        _bench_epoch(steps)
        elapsed = time.time() - started
        if distributed:
            torch.distributed.barrier()
        total_samples = steps * local_batch * world_size
        throughput = total_samples / elapsed if elapsed > 0 else 0.0
        if rank == 0:
            logger.info(
                "DDP_BENCH steps_per_rank=%d world_size=%d local_batch=%d total_samples=%d elapsed=%.2fs samples_per_s=%.1f",
                steps, world_size, local_batch, total_samples, elapsed, throughput,
            )
            print(f"DDP_BENCH samples_per_s={throughput:.1f} elapsed={elapsed:.2f}s total_samples={total_samples}")
        if distributed:
            torch.distributed.destroy_process_group()
        return

    state = run_training_loop(
        config=config,
        ptm_types=ptm_types,
        loaders=loaders,
        components=components,
        state=training_state,
        runtime_amp=runtime_amp,
        use_mlflow=use_mlflow,
        logger=logger,
        services=TrainingLoopServices(
            collect_prediction_rate_diagnostics=collect_prediction_rate_diagnostics,
            evaluate_multilabel=evaluate_multilabel,
            format_lr_summary=format_lr_summary,
            format_pair_metric_summary=format_pair_metric_summary,
            format_per_label_summary=format_per_label_summary,
            format_prediction_rate_summary=format_prediction_rate_summary,
            get_lr_diagnostics=get_lr_diagnostics,
            get_validation_monitor_value=get_validation_monitor_value,
            resolve_monitored_label_pairs=resolve_monitored_label_pairs,
            run_epoch=run_epoch,
            sanitize_name=sanitize_name,
            save_checkpoint=save_checkpoint,
            search_optimal_thresholds=search_optimal_thresholds,
            mlflow_api=mlflow,
        ),
    )

    if distributed:
        torch.distributed.barrier()
        if os.environ.get("DDP_SKIP_POST", "0") == "1":
            torch.distributed.destroy_process_group()
            return
        if rank != 0:
            torch.distributed.destroy_process_group()
            return

    orchestrate_post_training(
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
        profiler=profiler,
        train_distribution_monitor=train_distribution_monitor,
        use_mlflow=use_mlflow,
        mlflow_module=mlflow,
        dependencies={
            "run_epoch_fn": run_epoch,
            "load_checkpoint_fn": load_checkpoint,
            "search_optimal_thresholds_fn": search_optimal_thresholds,
            "evaluate_multilabel_fn": evaluate_multilabel,
            "export_metrics_fn": export_metrics,
            "export_structure_effect_significance_fn": export_structure_effect_significance,
            "sanitize_name_fn": sanitize_name,
            "collect_logits_labels_fn": collect_logits_labels,
            "sigmoid_np_fn": sigmoid_np,
            "macro_ece_fn": macro_ece,
            "fit_temperature_scaling_fn": fit_temperature_scaling,
            "apply_temperature_fn": apply_temperature,
            "reliability_curve_fn": reliability_curve,
            "fit_platt_scaling_fn": fit_platt_scaling,
            "apply_platt_fn": apply_platt,
            "fit_isotonic_fn": fit_isotonic,
            "apply_isotonic_fn": apply_isotonic,
            "has_sklearn_cal": HAS_SKLEARN_CAL,
            "build_performance_report_fn": build_performance_report,
            "export_startup_profile_comparison_fn": export_startup_profile_comparison,
        },
    )
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
