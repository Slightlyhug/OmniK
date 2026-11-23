"""Input record parsing and sequence window providers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.training_config import DEFAULT_SUMO_HYDROPHOBIC, canonical_ptm_name, sanitize_name
from src.utils.training_state import SplitData


def _canonical_ptm_label(name: str) -> str:
    return sanitize_name(name).replace("-", "").replace("_", "").replace(" ", "")


def _resolve_ptm_from_record(rec: Dict[str, Any], ptm_types: List[str]) -> str:
    labels = rec.get("labels")
    if isinstance(labels, dict):
        for name in ptm_types:
            value = labels.get(name, labels.get(name.capitalize(), labels.get(name.upper(), 0)))
            try:
                if float(value) > 0:
                    return sanitize_name(name)
            except Exception:
                continue
    if isinstance(labels, list) and len(labels) == len(ptm_types):
        for i, value in enumerate(labels):
            try:
                if float(value) > 0:
                    return sanitize_name(ptm_types[i])
            except Exception:
                continue
    active = rec.get("ptm_types") or rec.get("active_ptms") or rec.get("mods")
    if isinstance(active, list) and active:
        return sanitize_name(active[0])
    if rec.get("ptm_type"):
        return sanitize_name(str(rec["ptm_type"]))
    return sanitize_name(ptm_types[0]) if ptm_types else ""


def load_json_records(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_sources(config: Dict[str, Any], ptm_types: List[str]) -> Dict[str, Any]:
    feature_sources: Dict[str, Any] = {}
    mapping = config.get("pdb_fasta_mapping") if isinstance(config.get("pdb_fasta_mapping"), dict) else {}
    for ptm in ptm_types:
        ptm_key = sanitize_name(ptm)
        mapped = mapping.get(ptm_key, {})
        mapped_esm = mapped.get("esm_features") if isinstance(mapped, dict) else None
        if mapped_esm:
            feature_sources[ptm_key] = [str(p) for p in mapped_esm] if isinstance(mapped_esm, list) else str(mapped_esm)
    if not feature_sources and isinstance(config.get("esm_features_paths"), list):
        paths = config.get("esm_features_paths", [])
        for idx, path in enumerate(paths):
            if idx < len(ptm_types):
                feature_sources[sanitize_name(ptm_types[idx])] = str(path)
    return feature_sources


def _row_to_multilabel_record(row: Dict[str, Any], ptm_types: List[str]) -> Dict[str, Any]:
    row_keys = set(row.keys())
    label_map: Dict[str, float] = {}
    for ptm in ptm_types:
        col_name = ptm
        alt_cols = [ptm, ptm.capitalize(), ptm.upper(), ptm.lower()]
        value = 0.0
        for col in alt_cols:
            if col in row_keys:
                try:
                    value = float(row[col])
                except Exception:
                    value = 0.0
                break
            for existing in row_keys:
                if _canonical_ptm_label(existing) == _canonical_ptm_label(ptm):
                    try:
                        value = float(row[existing])
                    except Exception:
                        value = 0.0
                    break
        label_map[col_name] = float(value)
    rec = {
        "uniprot_id": str(row.get("uniprot_id", "")),
        "position": int(row.get("position", -1)),
        "residue": str(row.get("residue", "K")),
        "labels": label_map,
    }
    if "window_sequence" in row_keys and pd.notna(row.get("window_sequence")):
        rec["sequence_window_str"] = str(row.get("window_sequence"))
    if "source_tasks" in row_keys and pd.notna(row.get("source_tasks")):
        rec["source_tasks"] = str(row.get("source_tasks", ""))
        raw_tasks = [sanitize_name(x) for x in str(row.get("source_tasks", "")).replace(",", ";").split(";") if str(x).strip()]
        ordered_tasks = [sanitize_name(ptm) for ptm in ptm_types if sanitize_name(ptm) in set(raw_tasks)]
        if ordered_tasks:
            rec["active_ptms"] = ordered_tasks
            rec["feature_source"] = ordered_tasks[0]
    for meta_col in ("source_splits", "known_task_count", "positive_task_count", "negative_task_count", "unknown_task_count"):
        if meta_col in row_keys and pd.notna(row.get(meta_col)):
            value = row.get(meta_col)
            try:
                rec[meta_col] = int(value)
            except Exception:
                rec[meta_col] = str(value)
    for source_col in ("feature_source", "source", "ptm_type"):
        if source_col in row_keys and pd.notna(row.get(source_col)) and str(row.get(source_col)).strip():
            rec["feature_source"] = sanitize_name(str(row.get(source_col)))
            break
    return rec


def _teacher_column_candidates(ptm: str) -> List[str]:
    base = sanitize_name(ptm)
    aliases = [ptm, ptm.capitalize(), ptm.upper(), ptm.lower(), base, base.replace("_", ""), base.replace("-", "")]
    candidates: List[str] = []
    for alias in aliases:
        candidates.extend([
            f"teacher_{alias}",
            f"teacher_{alias}_prob",
            f"teacher_{alias}_probability",
            f"teacher_prob_{alias}",
            f"teacher_probability_{alias}",
            f"teacher_{alias}_logit",
            f"teacher_logit_{alias}",
            f"teacher_{alias}_mask",
            f"teacher_mask_{alias}",
            f"teacher_{alias}_weight",
            f"teacher_weight_{alias}",
        ])
    return candidates


def _extract_teacher_values(row: Dict[str, Any], ptm_types: List[str]) -> Tuple[List[float], List[float], List[float]]:
    row_keys = set(row.keys())
    teacher_probs = [-1.0] * len(ptm_types)
    teacher_mask = [0.0] * len(ptm_types)
    teacher_weight = [1.0] * len(ptm_types)

    def find_value(names: List[str]) -> Any:
        for name in names:
            if name in row_keys:
                return row.get(name)
        for existing in row_keys:
            canon_existing = _canonical_ptm_label(existing)
            for name in names:
                if canon_existing == _canonical_ptm_label(name):
                    return row.get(existing)
        return None

    for idx, ptm in enumerate(ptm_types):
        prob_value = find_value([c for c in _teacher_column_candidates(ptm) if "mask" not in c and "weight" not in c])
        if prob_value is not None and pd.notna(prob_value):
            try:
                prob = float(prob_value)
                if 0.0 <= prob <= 1.0:
                    teacher_probs[idx] = float(prob)
                    teacher_mask[idx] = 1.0
            except Exception:
                pass
        mask_value = find_value([c for c in _teacher_column_candidates(ptm) if "mask" in c])
        if mask_value is not None and pd.notna(mask_value):
            try:
                teacher_mask[idx] = float(np.clip(float(mask_value), 0.0, 1.0))
            except Exception:
                pass
        weight_value = find_value([c for c in _teacher_column_candidates(ptm) if "weight" in c])
        if weight_value is not None and pd.notna(weight_value):
            try:
                teacher_weight[idx] = float(np.clip(float(weight_value), 0.0, 10.0))
            except Exception:
                pass
    return teacher_probs, teacher_mask, teacher_weight


def attach_teacher_values(records: List[Dict[str, Any]], df: pd.DataFrame, ptm_types: List[str]) -> int:
    teacher_candidate_canon = {
        _canonical_ptm_label(candidate)
        for ptm in ptm_types
        for candidate in _teacher_column_candidates(ptm)
    }
    df_teacher_cols = {
        col for col in df.columns
        if _canonical_ptm_label(str(col)) in teacher_candidate_canon
    }
    if not df_teacher_cols:
        return 0
    updates = 0
    for rec, row in zip(records, df[list(df_teacher_cols)].to_dict(orient="records")):
        teacher_probs, teacher_mask, teacher_weight = _extract_teacher_values(row, ptm_types)
        if any(float(x) > 0.0 for x in teacher_mask):
            rec["teacher_probs"] = teacher_probs
            rec["teacher_mask"] = teacher_mask
            rec["teacher_weight"] = teacher_weight
            updates += 1
    return updates


def load_split_csv(path: Path, ptm_types: List[str], known_only: bool = False) -> List[Dict[str, Any]]:
    usecols = ["uniprot_id", "position", "residue", "window_sequence", "source_tasks", "source_splits", "feature_source", "source", "ptm_type"]
    for ptm in ptm_types:
        usecols.extend([ptm, ptm.capitalize(), ptm.upper(), ptm.lower()])
        usecols.extend(_teacher_column_candidates(ptm))
    seen = set()
    dedup_cols: List[str] = []
    for c in usecols:
        if c not in seen:
            dedup_cols.append(c)
            seen.add(c)
    canonical_ptms = {_canonical_ptm_label(ptm) for ptm in ptm_types}
    df = pd.read_csv(
        path,
        usecols=lambda c: c in seen or _canonical_ptm_label(c) in canonical_ptms or str(c).startswith("teacher_"),
    )
    records: List[Dict[str, Any]] = []
    append = records.append
    for row in df.to_dict(orient="records"):
        record = _row_to_multilabel_record(row, ptm_types)
        if known_only:
            # Single-task experiments should not retain unknown sites in the
            # in-memory split records used by validation and preparation.
            label = next(iter(record["labels"].values()), -1.0)
            if not np.isfinite(float(label)) or float(label) < 0.0:
                continue
        append(record)
    attach_teacher_values(records, df, ptm_types)
    return records


def _labels_to_array(rec: Dict[str, Any], ptm_types: List[str]) -> np.ndarray:
    labels = rec.get("labels")
    out = np.zeros((len(ptm_types),), dtype=np.float32)
    if isinstance(labels, dict):
        for i, ptm in enumerate(ptm_types):
            try:
                out[i] = float(labels.get(ptm, labels.get(ptm.capitalize(), 0.0)))
            except Exception:
                out[i] = 0.0
    elif isinstance(labels, list) and len(labels) == len(ptm_types):
        for i in range(len(ptm_types)):
            try:
                out[i] = float(labels[i])
            except Exception:
                out[i] = 0.0
    return out


def _parse_first_fasta_sequence(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    seq_parts: List[str] = []
    started = False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if started:
                        break
                    started = True
                    continue
                if started:
                    seq_parts.append(line)
        if not seq_parts:
            return None
        return "".join(seq_parts).replace(" ", "").replace("\t", "").upper()
    except Exception:
        return None


class SequenceWindowProvider:
    def __init__(self, fasta_dirs: List[str], window_size: int, logger: logging.Logger) -> None:
        self.fasta_dirs = [str(Path(d)) for d in fasta_dirs if d]
        self.window_size = int(window_size)
        self.logger = logger
        self._sequence_cache: Dict[str, Optional[str]] = {}
        self._fasta_path_cache: Dict[str, Optional[Path]] = {}

    def _find_fasta_path(self, uniprot_id: str) -> Optional[Path]:
        uid = str(uniprot_id)
        if uid in self._fasta_path_cache:
            return self._fasta_path_cache[uid]
        candidates = []
        for d in self.fasta_dirs:
            root = Path(d)
            if not root.exists():
                continue
            for ext in [".fasta", ".fa", ".faa", ".fna", ".txt"]:
                candidates.append(root / f"{uid}{ext}")
                candidates.append(root / f"{uid.lower()}{ext}")
        for path in candidates:
            if path.exists():
                self._fasta_path_cache[uid] = path
                return path
        for d in self.fasta_dirs:
            root = Path(d)
            if not root.exists():
                continue
            hits = sorted(root.glob(f"{uid}*.fa*"))
            if hits:
                self._fasta_path_cache[uid] = hits[0]
                return hits[0]
        self._fasta_path_cache[uid] = None
        return None

    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        uid = str(uniprot_id)
        if uid in self._sequence_cache:
            return self._sequence_cache[uid]
        path = self._find_fasta_path(uid)
        seq = _parse_first_fasta_sequence(path) if path is not None else None
        self._sequence_cache[uid] = seq
        return seq

    def get_window(self, uniprot_id: str, position_1based: int) -> Optional[str]:
        seq = self.get_sequence(uniprot_id)
        if not seq:
            return None
        pos = int(position_1based) - 1
        if pos < 0:
            return None
        half = self.window_size // 2
        start = max(0, pos - half)
        end = min(len(seq), pos + half + 1)
        window = seq[start:end]
        left_pad = max(0, half - pos)
        right_pad = max(0, self.window_size - len(window) - left_pad)
        if left_pad > 0:
            window = ("X" * left_pad) + window
        if right_pad > 0:
            window = window + ("X" * right_pad)
        if len(window) != self.window_size:
            window = window[: self.window_size].ljust(self.window_size, "X")
        return window


def split_records_from_json(raw: Any, val_ratio: float, test_ratio: float, seed: int) -> SplitData:
    if isinstance(raw, dict):
        if isinstance(raw.get("train"), list):
            train = raw.get("train", [])
            val = raw.get("val", [])
            test = raw.get("test", [])
            return SplitData(train=train, val=val, test=test)
        if isinstance(raw.get("samples"), list):
            records = raw["samples"]
        elif isinstance(raw.get("data"), list):
            records = raw["data"]
        else:
            raise ValueError("JSON must contain one of: train/val/test, samples, data")
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported JSON format")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    n = len(indices)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    return SplitData(train=train, val=val, test=test)


def compute_motif_bias_scores(
    window_seq: Optional[str],
    center_idx: int,
    ptm_types: List[str],
    active_tasks: Optional[List[str]] = None,
    sumo_hydrophobic: Optional[List[str]] = None,
    methyl_radius: int = 7,
    crotonyl_radius: int = 4,
) -> np.ndarray:
    scores = np.zeros((len(ptm_types),), dtype=np.float32)
    if not isinstance(window_seq, str) or not window_seq:
        return scores
    seq = window_seq.upper()
    if center_idx < 0 or center_idx >= len(seq):
        return scores
    task_set = {sanitize_name(x) for x in (active_tasks or ptm_types)}
    sumo_hydro = {str(x).upper() for x in (sumo_hydrophobic or DEFAULT_SUMO_HYDROPHOBIC)}
    methyl_radius = max(1, int(methyl_radius))
    crotonyl_radius = max(1, int(crotonyl_radius))

    for idx, ptm in enumerate(ptm_types):
        name = sanitize_name(ptm)
        if name not in task_set:
            continue
        if name == "sumoylation":
            score = 0.0
            if seq[center_idx] == "K":
                score += 0.4
            if center_idx - 1 >= 0 and seq[center_idx - 1] in sumo_hydro:
                score += 0.3
            if center_idx + 2 < len(seq) and seq[center_idx + 2] == "E":
                score += 0.3
            scores[idx] = min(1.0, score)
        elif name == "methylation":
            left = max(0, center_idx - methyl_radius)
            right = min(len(seq), center_idx + methyl_radius + 1)
            window = [aa for aa in seq[left:right] if aa != "X"]
            if window:
                rk = sum(1 for aa in window if aa in ("R", "K"))
                scores[idx] = rk / float(len(window))
        elif name == "crotonylation":
            left = max(0, center_idx - crotonyl_radius)
            right = min(len(seq), center_idx + crotonyl_radius + 1)
            window = [aa for aa in seq[left:right] if aa != "X"]
            if window:
                de = sum(1 for aa in window if aa in ("D", "E"))
                scores[idx] = de / float(len(window))
    return scores
