"""Threshold selection and multilabel evaluation metrics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.utils.training_config import sanitize_name


def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    metric = str(metric).lower()
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    return float(f1_score(y_true, y_pred, zero_division=0))


def search_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ptm_types: List[str],
    metric: str = "f1",
    max_candidates: int = 200,
    min_recall_by_label: Optional[Dict[str, float]] = None,
    max_threshold_by_label: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    metric = str(metric).lower()
    thresholds = np.full((y_true.shape[1],), 0.5, dtype=np.float32)
    report: Dict[str, Any] = {
        "metric": metric,
        "num_samples": int(y_true.shape[0]),
        "per_label": {},
    }

    for j, name in enumerate(ptm_types):
        known = y_true[:, j] >= 0.0
        t = y_true[known, j]
        p = np.clip(y_prob[known, j], 0.0, 1.0)
        if t.size == 0 or len(np.unique(t)) < 2:
            report["per_label"][name] = {
                "best_threshold": 0.5,
                "best_score": float("nan"),
                "candidates": [],
            }
            continue

        cand = np.unique(np.concatenate([p, np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)]))
        if cand.shape[0] > max_candidates:
            cand = np.quantile(p, np.linspace(0.01, 0.99, max_candidates))
        cand = np.unique(np.clip(cand, 0.0, 1.0))

        best_score = -1.0
        best_thr = 0.5
        key_name = sanitize_name(name)
        recall_floor = float((min_recall_by_label or {}).get(key_name, (min_recall_by_label or {}).get(name, 0.0)))
        threshold_cap = float((max_threshold_by_label or {}).get(key_name, (max_threshold_by_label or {}).get(name, 1.0)))
        candidate_rows: List[Dict[str, float]] = []
        fallback_score = -1.0
        fallback_thr = 0.5
        for thr in cand:
            if float(thr) > threshold_cap:
                continue
            pred = (p >= float(thr)).astype(np.int32)
            score = _metric_score(t, pred, metric)
            precision = float(precision_score(t, pred, zero_division=0))
            recall = float(recall_score(t, pred, zero_division=0))
            f1 = float(f1_score(t, pred, zero_division=0))
            candidate_rows.append(
                {
                    "threshold": float(thr),
                    "score": score,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )
            if score > fallback_score:
                fallback_score = score
                fallback_thr = float(thr)
            if recall < recall_floor:
                continue
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        if best_score < 0:
            best_score = fallback_score
            best_thr = fallback_thr

        thresholds[j] = best_thr
        report["per_label"][name] = {
            "best_threshold": float(best_thr),
            "best_score": float(best_score),
            "recall_floor": float(recall_floor),
            "threshold_cap": float(threshold_cap),
            "candidates": candidate_rows,
        }

    report["selected_thresholds"] = {name: float(thresholds[idx]) for idx, name in enumerate(ptm_types)}
    return thresholds, report


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom <= 0:
        return float("nan")
    return float((tp * tn - fp * fn) / denom)


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> Dict[str, float]:
    n = int(len(y_true))
    if n == 0:
        return {"precision_at_k": float("nan"), "recall_at_k": float("nan"), "k": int(k)}
    k = int(min(max(1, int(k)), n))
    order = np.argsort(-y_score, kind="mergesort")
    top = order[:k]
    tp = int(np.sum(y_true[top] == 1))
    total_pos = int(np.sum(y_true == 1))
    precision = float(tp / k)
    recall = float(tp / max(1, total_pos))
    return {"precision_at_k": precision, "recall_at_k": recall, "k": int(k)}


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float32, copy=False)
    y_prob = np.clip(y_prob.astype(np.float32, copy=False), 0.0, 1.0)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    total = float(y_true.shape[0])
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins) - 2 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> np.ndarray:
    thresholds, _ = search_optimal_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        ptm_types=[f"task_{idx}" for idx in range(y_true.shape[1])],
        metric=metric,
    )
    return thresholds


def resolve_monitored_label_pairs(raw_pairs: Any, ptm_types: List[str]) -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []
    valid_names = {sanitize_name(name): str(name) for name in ptm_types}
    if not isinstance(raw_pairs, list):
        return resolved
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left_key = sanitize_name(pair[0])
        right_key = sanitize_name(pair[1])
        if left_key == right_key or left_key not in valid_names or right_key not in valid_names:
            continue
        canon = (valid_names[left_key], valid_names[right_key])
        if canon not in resolved:
            resolved.append(canon)
    return resolved


def _cooccurrence_lift(a: np.ndarray, b: np.ndarray) -> float:
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    pab = float(np.mean(np.logical_and(a == 1, b == 1)))
    denom = max(pa * pb, 1e-8)
    return float(pab / denom) if denom > 0 else float("nan")


def format_per_label_summary(per_label: Dict[str, Any]) -> str:
    if not isinstance(per_label, dict) or not per_label:
        return ""
    parts = []
    for label_name, label_metrics in per_label.items():
        if not isinstance(label_metrics, dict):
            continue
        parts.append(
            (
                str(label_name),
                float(label_metrics.get("f1", float("nan"))),
                float(label_metrics.get("precision", float("nan"))),
                float(label_metrics.get("recall", float("nan"))),
            )
        )
    parts = sorted(parts, key=lambda x: (math.isnan(x[1]), x[1]))
    return " | ".join([f"{name}:f1={f1:.4f},p={prec:.4f},r={rec:.4f}" for name, f1, prec, rec in parts])


def format_pair_metric_summary(pair_metrics: Dict[str, Any]) -> str:
    if not isinstance(pair_metrics, dict) or not pair_metrics:
        return ""
    parts = []
    for pair_name, metrics in pair_metrics.items():
        if not isinstance(metrics, dict):
            continue
        parts.append(
            f"{pair_name}:joint_f1={float(metrics.get('joint_f1', float('nan'))):.4f},"
            f"joint_recall={float(metrics.get('joint_recall', float('nan'))):.4f},"
            f"pred_lift={float(metrics.get('pred_lift', float('nan'))):.3f}"
        )
    return " | ".join(parts)


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ptm_types: List[str],
    thresholds: Optional[np.ndarray] = None,
    top_k_list: Optional[List[int]] = None,
    monitored_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)

    if thresholds is None:
        thresholds = np.full((y_true.shape[1],), 0.5, dtype=np.float32)

    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)
    known_matrix = y_true >= 0.0
    y_true_known = y_true[known_matrix].astype(np.int32, copy=False)
    y_pred_known = y_pred[known_matrix].astype(np.int32, copy=False)

    metrics: Dict[str, Any] = {}
    metrics["known_label_fraction"] = float(known_matrix.mean()) if known_matrix.size else float("nan")
    metrics["micro_f1"] = float(f1_score(y_true_known, y_pred_known, zero_division=0)) if y_true_known.size else float("nan")
    metrics["micro_precision"] = float(precision_score(y_true_known, y_pred_known, zero_division=0)) if y_true_known.size else float("nan")
    metrics["micro_recall"] = float(recall_score(y_true_known, y_pred_known, zero_division=0)) if y_true_known.size else float("nan")

    if top_k_list is None:
        top_k_list = [50, 100, 200]

    try:
        metrics["macro_mcc"] = float(np.nanmean([mcc_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))
    except Exception:
        metrics["macro_mcc"] = float("nan")

    try:
        metrics["macro_ece"] = float(np.nanmean([ece_binary(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])]))
    except Exception:
        metrics["macro_ece"] = float("nan")

    per_label = {}
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    auc_values: List[float] = []
    auprc_values: List[float] = []
    mcc_values: List[float] = []
    ece_values: List[float] = []
    for i, name in enumerate(ptm_types):
        known = y_true[:, i] >= 0.0
        yi = y_true[known, i]
        pi = y_prob[known, i]
        bi = y_pred[known, i]
        if yi.size == 0:
            per_label[name] = {"known_count": 0, "threshold": float(thresholds[i])}
            continue
        cm = confusion_matrix(yi.astype(np.int32, copy=False), bi.astype(np.int32, copy=False), labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        topk_rows = []
        for k in top_k_list:
            topk_rows.append(precision_recall_at_k(yi, pi, int(k)))
        f1_val = float(f1_score(yi, bi, zero_division=0))
        precision_val = float(precision_score(yi, bi, zero_division=0))
        recall_val = float(recall_score(yi, bi, zero_division=0))
        auc_val = safe_auc(yi, pi)
        auprc_val = safe_auprc(yi, pi)
        mcc_val = mcc_score(yi, bi)
        ece_val = ece_binary(yi, pi)
        f1_values.append(f1_val)
        precision_values.append(precision_val)
        recall_values.append(recall_val)
        auc_values.append(auc_val)
        auprc_values.append(auprc_val)
        mcc_values.append(mcc_val)
        ece_values.append(ece_val)
        per_label[name] = {
            "f1": f1_val,
            "precision": precision_val,
            "recall": recall_val,
            "mcc": mcc_val,
            "roc_auc": auc_val,
            "auprc": auprc_val,
            "ece": ece_val,
            "top_k": topk_rows,
            "threshold": float(thresholds[i]),
            "known_count": int(known.sum()),
            "unknown_count": int((~known).sum()),
            "support_positive": int(np.sum(yi == 1)),
            "support_negative": int(np.sum(yi == 0)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    metrics["macro_f1"] = float(np.nanmean(f1_values)) if f1_values else float("nan")
    metrics["macro_precision"] = float(np.nanmean(precision_values)) if precision_values else float("nan")
    metrics["macro_recall"] = float(np.nanmean(recall_values)) if recall_values else float("nan")
    metrics["macro_roc_auc"] = float(np.nanmean(auc_values)) if auc_values else float("nan")
    metrics["macro_auprc"] = float(np.nanmean(auprc_values)) if auprc_values else float("nan")
    metrics["macro_mcc"] = float(np.nanmean(mcc_values)) if mcc_values else float("nan")
    metrics["macro_ece"] = float(np.nanmean(ece_values)) if ece_values else float("nan")
    metrics["per_label"] = per_label
    pair_metrics: Dict[str, Any] = {}
    pair_lookup = {sanitize_name(name): idx for idx, name in enumerate(ptm_types)}
    for left_name, right_name in monitored_pairs or []:
        left_idx = pair_lookup.get(sanitize_name(left_name))
        right_idx = pair_lookup.get(sanitize_name(right_name))
        if left_idx is None or right_idx is None or left_idx == right_idx:
            continue
        known_pair = np.logical_and(y_true[:, left_idx] >= 0.0, y_true[:, right_idx] >= 0.0)
        true_left = y_true[known_pair, left_idx]
        true_right = y_true[known_pair, right_idx]
        pred_left = y_pred[known_pair, left_idx]
        pred_right = y_pred[known_pair, right_idx]
        true_joint = np.logical_and(true_left == 1, true_right == 1).astype(np.int32)
        pred_joint = np.logical_and(pred_left == 1, pred_right == 1).astype(np.int32)
        pair_name = f"{left_name}__{right_name}"
        pair_metrics[pair_name] = {
            "joint_precision": float(precision_score(true_joint, pred_joint, zero_division=0)) if true_joint.size else float("nan"),
            "joint_recall": float(recall_score(true_joint, pred_joint, zero_division=0)) if true_joint.size else float("nan"),
            "joint_f1": float(f1_score(true_joint, pred_joint, zero_division=0)) if true_joint.size else float("nan"),
            "known_pair_count": int(known_pair.sum()),
            "true_joint_rate": float(true_joint.mean()) if true_joint.size else float("nan"),
            "pred_joint_rate": float(pred_joint.mean()) if pred_joint.size else float("nan"),
            "true_lift": _cooccurrence_lift(true_left, true_right) if true_joint.size else float("nan"),
            "pred_lift": _cooccurrence_lift(pred_left, pred_right) if pred_joint.size else float("nan"),
            "support_joint_positive": int(true_joint.sum()),
            "support_joint_predicted": int(pred_joint.sum()),
        }
    metrics["pair_metrics"] = pair_metrics
    return metrics
