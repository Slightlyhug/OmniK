"""Record filtering and feature-quality checks for training splits."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.training_config import sanitize_name
from src.utils.training_data import H5DataLoader, PDBStructureCache
from src.utils.training_records import _resolve_ptm_from_record


def filter_records_with_available_pdb(
    records: List[Dict[str, Any]],
    ptm_types: List[str],
    structure_cache: Optional[PDBStructureCache],
    logger: logging.Logger,
    split_name: str,
) -> List[Dict[str, Any]]:
    if structure_cache is None:
        return records
    kept: List[Dict[str, Any]] = []
    dropped = 0
    resolved: Dict[Tuple[str, str], bool] = {}
    for rec in records:
        uid = str(rec.get("uniprot_id", ""))
        ptm = _resolve_ptm_from_record(rec, ptm_types)
        key = (uid, ptm)
        if key in resolved:
            if resolved[key]:
                kept.append(rec)
            else:
                dropped += 1
            continue
        try:
            structure_cache._resolve_pdb_path(uid, ptm)
            kept.append(rec)
            resolved[key] = True
        except FileNotFoundError as exc:
            structure_cache.missing_files.add(str(exc))
            dropped += 1
            resolved[key] = False
    if dropped > 0:
        logger.warning("Dropped %d samples in %s due to missing PDB files", dropped, split_name)
    return kept


def _candidate_feature_sources_for_record(
    rec: Dict[str, Any],
    ptm_types: List[str],
    feature_sources: Optional[Dict[str, str]],
) -> List[str]:
    candidates: List[str] = []
    for key in ("source", "ptm_type", "label_source"):
        value = rec.get(key)
        if value is not None:
            name = sanitize_name(str(value))
            if name and name not in candidates:
                candidates.append(name)
    ptm = _resolve_ptm_from_record(rec, ptm_types)
    if ptm and ptm not in candidates:
        candidates.append(ptm)
    for ptm_name in ptm_types:
        name = sanitize_name(ptm_name)
        if name and name not in candidates:
            candidates.append(name)
    for name in (feature_sources or {}).keys():
        key = sanitize_name(name)
        if key and key not in candidates:
            candidates.append(key)
    return candidates


def filter_records_by_structure_feature_availability(
    records: List[Dict[str, Any]],
    ptm_types: List[str],
    feature_sources: Optional[Dict[str, str]],
    window_size: int,
    keep_available: bool,
    logger: logging.Logger,
    split_name: str,
) -> List[Dict[str, Any]]:
    if not feature_sources:
        logger.warning(
            "Structure feature filter requested for %s but no feature_sources are configured; keeping all records.",
            split_name,
        )
        return records
    loader = H5DataLoader(feature_sources)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    cached: Dict[Tuple[str, int], bool] = {}
    try:
        for rec in records:
            uid = str(rec.get("uniprot_id", "")).strip()
            try:
                pos = int(rec.get("position"))
            except Exception:
                pos = -1
            cache_key = (uid, pos)
            available = cached.get(cache_key)
            if available is None:
                available = False
                if uid and pos > 0:
                    for source in _candidate_feature_sources_for_record(rec, ptm_types, feature_sources):
                        window = loader.get_window(source, uid, "structure_features", pos, int(window_size))
                        if window is not None and bool(np.any(np.asarray(window, dtype=np.float32) != 0.0)):
                            available = True
                            break
                cached[cache_key] = available
            if bool(available) == bool(keep_available):
                kept.append(rec)
            else:
                dropped += 1
    finally:
        loader.close()
    logger.info(
        "Structure feature filter %s | keep_available=%s before=%d kept=%d dropped=%d",
        split_name,
        str(bool(keep_available)),
        len(records),
        len(kept),
        dropped,
    )
    return kept


def repair_or_filter_sequence_windows(
    records: List[Dict[str, Any]],
    sequence_window_provider: Optional["SequenceWindowProvider"],
    window_size: int,
    strict_center_k: bool,
    drop_unrepairable: bool,
    logger: logging.Logger,
    split_name: str,
) -> List[Dict[str, Any]]:
    if sequence_window_provider is None:
        return records
    center_idx = int(window_size) // 2
    kept: List[Dict[str, Any]] = []
    repaired = 0
    unchanged_valid = 0
    dropped = 0
    missing_fasta = 0
    center_mismatch = 0
    bad_length = 0

    for rec in records:
        uid = str(rec.get("uniprot_id", ""))
        pos = int(rec.get("position", -1))
        original = rec.get("sequence_window_str")
        aligned = sequence_window_provider.get_window(uid, pos)

        if aligned is None:
            missing_fasta += 1
            if drop_unrepairable:
                dropped += 1
                continue
            kept.append(rec)
            continue

        is_valid = len(aligned) == int(window_size)
        if not is_valid:
            bad_length += 1
        if strict_center_k and is_valid and aligned[center_idx] != "K":
            center_mismatch += 1
            is_valid = False

        if not is_valid:
            if drop_unrepairable:
                dropped += 1
                continue
            kept.append(rec)
            continue

        if original == aligned:
            unchanged_valid += 1
            kept.append(rec)
        else:
            updated = dict(rec)
            updated["sequence_window_str"] = aligned
            kept.append(updated)
            repaired += 1

    logger.info(
        "Sequence window FASTA repair [%s]: kept=%d repaired=%d unchanged_valid=%d dropped=%d missing_fasta=%d bad_length=%d center_mismatch=%d",
        split_name,
        len(kept),
        repaired,
        unchanged_valid,
        dropped,
        missing_fasta,
        bad_length,
        center_mismatch,
    )
    return kept


def _record_candidate_sources(
    rec: Dict[str, Any],
    ptm_types: List[str],
    feature_sources: Optional[Dict[str, str]],
) -> List[str]:
    ptm_type = _resolve_ptm_from_record(rec, ptm_types)
    preferred_source = rec.get("feature_source") or rec.get("source")
    candidates: List[str] = []
    for cand in (sanitize_name(str(preferred_source)) if preferred_source else "", sanitize_name(ptm_type)):
        if cand and cand not in candidates:
            candidates.append(cand)
    for cand in [sanitize_name(x) for x in ptm_types]:
        if cand and cand not in candidates:
            candidates.append(cand)
    for cand in (sanitize_name(k) for k in (feature_sources or {}).keys()):
        if cand and cand not in candidates:
            candidates.append(cand)
    return candidates


def _resolve_existing_feature_source(
    rec: Dict[str, Any],
    ptm_types: List[str],
    feature_sources: Optional[Dict[str, str]],
    h5_loader: H5DataLoader,
) -> Tuple[str, List[str]]:
    uid = str(rec.get("uniprot_id", ""))
    candidates = _record_candidate_sources(rec, ptm_types, feature_sources)
    source = candidates[0] if candidates else sanitize_name(_resolve_ptm_from_record(rec, ptm_types))
    if uid:
        for cand in candidates:
            if h5_loader.get_dataset(cand, uid, "sequence_features") is not None:
                return cand, candidates
    return source, candidates


def _window_quality_flags(
    window_seq: Optional[str],
    window_size: int,
    strict_center_k: bool,
) -> Dict[str, bool]:
    seq = str(window_seq).strip().upper() if isinstance(window_seq, str) else ""
    valid_length = len(seq) == int(window_size)
    center_ok = False
    if valid_length:
        center_ok = seq[int(window_size) // 2] == "K"
    return {
        "present": bool(seq),
        "valid_length": bool(valid_length),
        "center_ok": bool(center_ok if strict_center_k else valid_length),
    }


def filter_records_by_feature_quality(
    records: List[Dict[str, Any]],
    ptm_types: List[str],
    feature_sources: Optional[Dict[str, str]],
    window_size: int,
    strict_center_k: bool,
    feature_filter_cfg: Optional[Dict[str, Any]],
    report_dir: Path,
    logger: logging.Logger,
    split_name: str,
) -> List[Dict[str, Any]]:
    cfg = feature_filter_cfg if isinstance(feature_filter_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return records

    drop_invalid_identity = bool(cfg.get("drop_invalid_identity", True))
    drop_missing_sequence_features = bool(cfg.get("drop_missing_sequence_features", True))
    drop_invalid_window_after_repair = bool(cfg.get("drop_invalid_window_after_repair", True))
    example_limit = max(1, int(cfg.get("report_top_n_examples", 20)))

    report_dir.mkdir(parents=True, exist_ok=True)
    h5_loader = H5DataLoader(feature_sources or {})
    stats: Dict[str, Any] = {
        "split": split_name,
        "input_records": int(len(records)),
        "kept_records": 0,
        "dropped_records": 0,
        "drop_reasons": defaultdict(int),
        "examples": defaultdict(list),
    }
    kept: List[Dict[str, Any]] = []

    try:
        for rec in records:
            uid = str(rec.get("uniprot_id", "")).strip()
            pos = int(rec.get("position", -1))
            flags = _window_quality_flags(rec.get("sequence_window_str"), window_size, strict_center_k)
            drop_reason: Optional[str] = None

            if drop_invalid_identity and not uid:
                drop_reason = "empty_uniprot_id"
            elif drop_invalid_identity and pos <= 0:
                drop_reason = "nonpositive_position"
            elif drop_invalid_window_after_repair and not flags["valid_length"]:
                drop_reason = "invalid_window_length"
            elif drop_invalid_window_after_repair and strict_center_k and not flags["center_ok"]:
                drop_reason = "invalid_window_center"
            else:
                resolved_source, _ = _resolve_existing_feature_source(rec, ptm_types, feature_sources, h5_loader)
                seq_ds = h5_loader.get_dataset(resolved_source, uid, "sequence_features") if uid and resolved_source else None
                if drop_missing_sequence_features and seq_ds is None:
                    drop_reason = "missing_sequence_features"

            if drop_reason is not None:
                stats["dropped_records"] += 1
                stats["drop_reasons"][drop_reason] += 1
                if len(stats["examples"][drop_reason]) < example_limit:
                    stats["examples"][drop_reason].append(
                        {
                            "uniprot_id": uid,
                            "position": pos,
                            "window_sequence": rec.get("sequence_window_str"),
                        }
                    )
                continue

            kept.append(rec)

        stats["kept_records"] = int(len(kept))
        with open(report_dir / f"feature_filter_{split_name}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "split": stats["split"],
                    "input_records": stats["input_records"],
                    "kept_records": stats["kept_records"],
                    "dropped_records": stats["dropped_records"],
                    "drop_reasons": dict(stats["drop_reasons"]),
                    "examples": {k: v for k, v in stats["examples"].items()},
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    finally:
        h5_loader.close()

    logger.info(
        "Feature quality filter [%s]: input=%d kept=%d dropped=%d reasons=%s",
        split_name,
        len(records),
        len(kept),
        stats["dropped_records"],
        json.dumps(dict(stats["drop_reasons"]), ensure_ascii=False, sort_keys=True),
    )
    return kept
