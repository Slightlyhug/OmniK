"""Runtime setup and lightweight profiling for the multitask trainer."""

from __future__ import annotations

import json
import logging
import os
import random
import resource
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multitask_lysine")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    rank = int(os.environ.get("RANK", "0"))
    log_name = "train_multitask.log" if rank == 0 else f"train_multitask.rank{rank}.log"
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / log_name, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int, cudnn_deterministic: bool = False, cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not cudnn_deterministic)


def flatten_config_for_logging(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config_for_logging(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, ensure_ascii=True)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat


class StageProfiler:
    def __init__(self, enabled: bool, logger: logging.Logger) -> None:
        self.enabled = bool(enabled)
        self.logger = logger
        self.rows: List[Dict[str, float]] = []

    @staticmethod
    def _rss_mb() -> float:
        try:
            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return rss / (1024.0 * 1024.0) if os.uname().sysname.lower() == "darwin" else rss / 1024.0
        except Exception:
            return float("nan")

    def run(self, name: str, fn):
        if not self.enabled:
            return fn()
        t0, rss0 = time.perf_counter(), self._rss_mb()
        out = fn()
        elapsed, rss1 = time.perf_counter() - t0, self._rss_mb()
        row = {
            "stage": str(name), "elapsed_sec": float(elapsed),
            "rss_before_mb": float(rss0), "rss_after_mb": float(rss1),
            "rss_delta_mb": float(rss1 - rss0) if np.isfinite(rss0) and np.isfinite(rss1) else float("nan"),
        }
        self.rows.append(row)
        self.logger.info(
            "PERF stage=%s elapsed=%.3fs rss_before=%.1fMB rss_after=%.1fMB rss_delta=%.1fMB",
            row["stage"], row["elapsed_sec"], row["rss_before_mb"], row["rss_after_mb"], row["rss_delta_mb"],
        )
        return out

    def save(self, output_path: Path) -> None:
        if not self.enabled:
            return
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": self.rows,
            "total_elapsed_sec": float(sum(row["elapsed_sec"] for row in self.rows)),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_startup_profile_comparison(output_dir: Path, logger: logging.Logger) -> None:
    current_path = output_dir / "startup_profile.json"
    baseline_path = output_dir / "startup_profile_baseline.json"
    if not current_path.exists() or not baseline_path.exists():
        return
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current_total = float(current.get("total_elapsed_sec", float("nan")))
        baseline_total = float(baseline.get("total_elapsed_sec", float("nan")))
        reduction = float((baseline_total - current_total) / max(1e-12, baseline_total)) if np.isfinite(baseline_total) else float("nan")
        report = {
            "current_total_elapsed_sec": current_total,
            "baseline_total_elapsed_sec": baseline_total,
            "startup_time_reduction_ratio": reduction,
            "startup_time_reduction_percent": float(reduction * 100.0) if np.isfinite(reduction) else float("nan"),
            "target_reduction_ratio": 0.5,
            "target_met": bool(np.isfinite(reduction) and reduction >= 0.5),
        }
        (output_dir / "startup_profile_comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Startup profile comparison exported. reduction=%.2f%% target_met=%s",
            report["startup_time_reduction_percent"], report["target_met"],
        )
    except Exception as exc:
        logger.warning("Failed to export startup profile comparison: %s", str(exc))


def summarize_epoch_performance(
    elapsed_time: float,
    sample_count: int,
    batch_count: int,
    optimizer_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    gpu_memory_mb = None
    if device.type == "cuda":
        gpu_memory_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))
    throughput = float(sample_count / max(elapsed_time, 1e-8))
    return {
        "elapsed_time_sec": float(elapsed_time),
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "optimizer_steps": int(optimizer_steps),
        "samples_per_sec": throughput,
        "max_gpu_memory_mb": gpu_memory_mb,
    }

