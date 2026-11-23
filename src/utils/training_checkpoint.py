"""Checkpoint persistence helpers for the multitask trainer."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


def save_checkpoint(
    ckpt_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    best_monitor_value: float,
    best_monitor_metric: str,
    best_val_macro_f1: float,
    best_val_macro_auc: float,
    thresholds: np.ndarray,
    criterion: Optional[nn.Module] = None,
    scaler: Optional[Any] = None,
) -> None:
    target_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_monitor_value": best_monitor_value,
        "best_monitor_metric": str(best_monitor_metric),
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_macro_auc": best_val_macro_auc,
        "thresholds": thresholds.tolist(),
        "criterion_state_dict": criterion.state_dict() if criterion is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(prefix=f".{ckpt_path.name}.", suffix=".tmp", dir=str(ckpt_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_shape_compatible_state_dict(model: nn.Module, state_dict: Dict[str, Any], logger: logging.Logger, context: str):
    current_state = model.state_dict()
    compatible: Dict[str, Any] = {}
    skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    skipped_unexpected = 0
    for key, value in state_dict.items():
        if key not in current_state:
            skipped_unexpected += 1
            continue
        current_shape = tuple(current_state[key].shape) if hasattr(current_state[key], "shape") else None
        incoming_shape = tuple(value.shape) if hasattr(value, "shape") else None
        if current_shape is not None and incoming_shape is not None and current_shape != incoming_shape:
            skipped_shape.append((key, incoming_shape, current_shape))
            continue
        compatible[key] = value
    if skipped_shape:
        preview = "; ".join(f"{key}: checkpoint={old} current={new}" for key, old, new in skipped_shape[:20])
        logger.warning("%s skipped %d shape-mismatched tensors before load_state_dict. First skipped: %s", context, len(skipped_shape), preview)
    if skipped_unexpected:
        logger.warning("%s skipped %d unexpected tensors not present in current model.", context, skipped_unexpected)
    result = model.load_state_dict(compatible, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning("%s load_state_dict shape-compatible strict=False | missing=%s unexpected=%s", context, result.missing_keys, result.unexpected_keys)
    return result, skipped_shape


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    criterion: Optional[nn.Module] = None,
    scaler: Optional[Any] = None,
) -> Tuple[int, float, float, float, str, Optional[np.ndarray]]:
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(ckpt_path, map_location="cpu")
    logger = logging.getLogger(__name__)
    target_model = model.module if hasattr(model, "module") else model
    load_shape_compatible_state_dict(target_model, data["model_state_dict"], logger, context=f"Checkpoint load from {ckpt_path}")
    criterion_state = data.get("criterion_state_dict")
    if criterion is not None and isinstance(criterion_state, dict) and criterion_state:
        try:
            criterion.load_state_dict(criterion_state)
        except Exception as exc:
            logger.warning("Skipping criterion state load due to mismatch: %s", str(exc))
    elif criterion is not None:
        logger.info("Checkpoint has no criterion state; using a fresh criterion.")
    for obj, key, label in (
        (optimizer, "optimizer_state_dict", "optimizer"),
        (scheduler, "scheduler_state_dict", "scheduler"),
        (scaler, "scaler_state_dict", "scaler"),
    ):
        saved = data.get(key)
        if obj is None or not isinstance(saved, dict) or not saved:
            continue
        try:
            obj.load_state_dict(saved)
        except Exception as exc:
            logger.warning("Skipping %s state load due to mismatch: %s", label, str(exc))
    if scaler is not None and not data.get("scaler_state_dict"):
        logger.info("Checkpoint has no enabled AMP scaler state; using a fresh scaler.")
    best_f1 = float(data.get("best_val_macro_f1", -1.0))
    thresholds = data.get("thresholds")
    return (
        int(data.get("epoch", 0)) + 1,
        float(data.get("best_monitor_value", best_f1)),
        best_f1,
        float(data.get("best_val_macro_auc", -1.0)),
        str(data.get("best_monitor_metric", "val_macro_f1")),
        np.asarray(thresholds, dtype=np.float32) if thresholds is not None else None,
    )


def export_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
