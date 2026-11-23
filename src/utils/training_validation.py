"""Split-level validation and feature availability reporting."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.utils.training_config import sanitize_name
from src.utils.training_data import (
    GraphFeatureCache,
    GraphFeatureLMDBCache,
    H5DataLoader,
    PDBStructureCache,
)
from src.utils.training_quality import (
    _resolve_existing_feature_source,
    _window_quality_flags,
)
from src.utils.training_records import _labels_to_array, _resolve_ptm_from_record
from src.utils.training_state import SplitData


def _site_key(rec: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(rec.get("uniprot_id", "")),
        int(rec.get("position", -1)),
        str(rec.get("residue", "K")).upper(),
    )


def summarize_split_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    proteins = {str(r.get("uniprot_id", "")) for r in records if str(r.get("uniprot_id", ""))}
    site_keys = [_site_key(r) for r in records]
    unique_site_keys = set(site_keys)
    return {
        "samples": int(len(records)),
        "proteins": int(len(proteins)),
        "unique_site_keys": int(len(unique_site_keys)),
        "duplicate_site_keys": int(len(site_keys) - len(unique_site_keys)),
    }


def validate_predefined_split_integrity(
    splits: SplitData,
    logger: logging.Logger,
    allow_protein_overlap: bool = False,
    allow_non_lysine_sites: bool = False,
) -> None:
    protein_sets = {
        "train": {str(r.get("uniprot_id", "")) for r in splits.train if str(r.get("uniprot_id", ""))},
        "val": {str(r.get("uniprot_id", "")) for r in splits.val if str(r.get("uniprot_id", ""))},
        "test": {str(r.get("uniprot_id", "")) for r in splits.test if str(r.get("uniprot_id", ""))},
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = protein_sets[left] & protein_sets[right]
        if overlap:
            message = f"Detected protein leakage between {left} and {right}: {len(overlap)} overlapping proteins"
            if allow_protein_overlap:
                logger.warning("%s (allowed by config)", message)
            else:
                raise ValueError(message)

    for split_name, records in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        summary = summarize_split_records(records)
        logger.info(
            "Split[%s] samples=%d proteins=%d unique_site_keys=%d duplicate_site_keys=%d",
            split_name,
            summary["samples"],
            summary["proteins"],
            summary["unique_site_keys"],
            summary["duplicate_site_keys"],
        )
        if summary["duplicate_site_keys"] > 0:
            raise ValueError(f"Detected duplicate site keys inside {split_name}: {summary['duplicate_site_keys']}")
        non_lysine = [
            _site_key(rec)
            for rec in records
            if str(rec.get("residue", "K")).strip().upper() != "K"
        ]
        if non_lysine:
            message = (
                f"Detected non-lysine sites inside {split_name}: {len(non_lysine)} "
                f"(examples: {non_lysine[:5]})"
            )
            if allow_non_lysine_sites:
                logger.warning("%s (allowed by config)", message)
            else:
                raise ValueError(message)


def validate_split_ptm_presence(splits: SplitData, ptm_types: List[str]) -> None:
    for split_name, records in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        positive_counts = np.zeros((len(ptm_types),), dtype=np.int64)
        known_counts = np.zeros((len(ptm_types),), dtype=np.int64)
        for rec in records:
            labels = _labels_to_array(rec, ptm_types)
            positive_counts += (labels > 0.0).astype(np.int64)
            known_counts += (labels >= 0.0).astype(np.int64)
        missing = [ptm_types[i] for i, v in enumerate(positive_counts) if v <= 0]
        no_known = [ptm_types[i] for i, v in enumerate(known_counts) if v <= 0]
        if missing:
            raise ValueError(f"Split {split_name} is missing positive samples for PTM types: {missing}")
        if no_known:
            raise ValueError(f"Split {split_name} is missing known labels for PTM types: {no_known}")


def prepare_data_integrity_report(
    splits: SplitData,
    ptm_types: List[str],
    report_dir: Path,
    structure_cache: Optional[PDBStructureCache],
    feature_sources: Optional[Dict[str, str]],
    graph_dirs_by_ptm: Optional[Dict[str, str]],
    default_graph_dir: Optional[str],
    graph_lmdbs_by_ptm: Optional[Dict[str, str]],
    default_graph_lmdb: Optional[str],
    window_size: int,
    strict_center_k: bool,
    expected_seq_dim: int,
    expected_struct_dim: int,
    logger: logging.Logger,
    enabled: bool = False,
) -> None:
    env_enabled = bool(os.environ.get("RUN_DATA_INTEGRITY_REPORT", "0") in {"1", "true", "TRUE", "yes", "on"})
    config_enabled = bool(enabled)
    if not (env_enabled or config_enabled):
        logger.info("Data integrity report skipped by default for startup performance. Set RUN_DATA_INTEGRITY_REPORT=1 to enable.")
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        "train": list(splits.train),
        "val": list(splits.val),
        "test": list(splits.test),
    }
    all_records = split_records["train"] + split_records["val"] + split_records["test"]
    total = len(all_records)
    h5_loader = H5DataLoader(feature_sources or {})
    missing_files: Set[str] = set()

    graph_name_cache: Dict[str, Set[str]] = {}
    lmdb_cache = (
        GraphFeatureLMDBCache(graph_lmdbs_by_ptm or {}, default_graph_lmdb)
        if (default_graph_lmdb or (graph_lmdbs_by_ptm and any(graph_lmdbs_by_ptm.values())))
        else None
    )

    def _load_graph_names(directory: Path) -> Set[str]:
        key = str(directory.resolve())
        if key in graph_name_cache:
            return graph_name_cache[key]
        names: Set[str] = set()
        if directory.exists():
            for pattern in ("*.pt", "*.npz"):
                for file_path in directory.glob(pattern):
                    names.add(file_path.name)
        graph_name_cache[key] = names
        return names

    def _graph_available(uid: str, pos: int, source_key: str) -> bool:
        if lmdb_cache is not None:
            return bool(lmdb_cache.has_graph(uid, pos, source_key))
        candidate_names = GraphFeatureCache._candidate_names(uid, pos, source_key)
        graph_dirs: List[Path] = []
        source_dir = (graph_dirs_by_ptm or {}).get(sanitize_name(source_key))
        if source_dir:
            graph_dirs.append(Path(str(source_dir)))
        if default_graph_dir:
            shared_dir = Path(str(default_graph_dir))
            if shared_dir not in graph_dirs:
                graph_dirs.append(shared_dir)
        for graph_dir in graph_dirs:
            names = _load_graph_names(graph_dir)
            if any(name in names for name in candidate_names):
                return True
        return False

    split_reports: Dict[str, Dict[str, Any]] = {}
    global_counts: Dict[str, int] = defaultdict(int)
    seen_sites: Set[Tuple[str, str, int, str]] = set()
    for split_name, records in split_records.items():
        counts: Dict[str, int] = defaultdict(int)
        for rec in records:
            uid = str(rec.get("uniprot_id", ""))
            pos = int(rec.get("position", -1))
            ptm_type = _resolve_ptm_from_record(rec, ptm_types)
            source, _ = _resolve_existing_feature_source(rec, ptm_types, feature_sources, h5_loader)
            site_key = (split_name, uid, pos, source)
            if site_key in seen_sites:
                continue
            seen_sites.add(site_key)

            counts["inspected"] += 1
            label_arr = _labels_to_array(rec, ptm_types)
            if float(label_arr.sum()) > 0:
                counts["positive_rows"] += 1
            else:
                counts["all_zero_rows"] += 1

            window_flags = _window_quality_flags(rec.get("sequence_window_str"), window_size, strict_center_k)
            if window_flags["present"]:
                counts["window_present"] += 1
            if window_flags["valid_length"]:
                counts["window_len_ok"] += 1
            if window_flags["center_ok"]:
                counts["window_center_ok"] += 1

            seq_ds = h5_loader.get_dataset(source, uid, "sequence_features") if uid and source else None
            if seq_ds is None:
                counts["sequence_missing"] += 1
                missing_files.add(f"ESM missing: source={source}, uniprot_id={uid}")
                continue

            counts["sequence_available"] += 1
            try:
                seq_feat_dim = int(seq_ds.shape[1]) if seq_ds.ndim == 2 else -1
                seq_len = int(seq_ds.shape[0]) if seq_ds.ndim == 2 else 0
            except Exception:
                seq_feat_dim = -1
                seq_len = 0
            if seq_feat_dim == int(expected_seq_dim):
                counts["sequence_dim_ok"] += 1

            micro_env = h5_loader.get_site_vector(source, uid, "micro_env_features", pos)
            if micro_env is not None:
                counts["micro_env_available"] += 1
                if int(np.asarray(micro_env).reshape(-1).shape[0]) == 6:
                    counts["micro_env_dim_ok"] += 1
            else:
                counts["micro_env_missing"] += 1

            struct_window = h5_loader.get_window(source, uid, "structure_features", pos, 5)
            if struct_window is not None:
                counts["structure_available_h5"] += 1
                if struct_window.ndim == 2 and int(struct_window.shape[1]) == int(expected_struct_dim):
                    counts["structure_dim_ok"] += 1
            elif structure_cache is not None:
                aligned = structure_cache.align_to_sequence(uid, ptm_type, seq_len=max(1, seq_len))
                if aligned is not None:
                    counts["structure_available_pdb"] += 1
                    if aligned.ndim == 2 and int(aligned.shape[1]) == int(expected_struct_dim):
                        counts["structure_dim_ok"] += 1
                else:
                    counts["structure_missing"] += 1
                    missing_files.add(f"PDB missing or parse failed: ptm={ptm_type}, uniprot_id={uid}")
            else:
                counts["structure_missing"] += 1

            if _graph_available(uid, pos, source or ptm_type):
                counts["graph_available"] += 1
            else:
                counts["graph_missing"] += 1

        inspected = max(1, counts["inspected"])
        split_reports[split_name] = {
            "rows": int(counts["inspected"]),
            "positive_rows": int(counts["positive_rows"]),
            "all_zero_rows": int(counts["all_zero_rows"]),
            "window_present": int(counts["window_present"]),
            "window_len_ok": int(counts["window_len_ok"]),
            "window_center_ok": int(counts["window_center_ok"]),
            "sequence_available": int(counts["sequence_available"]),
            "sequence_missing": int(counts["sequence_missing"]),
            "sequence_dim_ok": int(counts["sequence_dim_ok"]),
            "micro_env_available": int(counts["micro_env_available"]),
            "micro_env_missing": int(counts["micro_env_missing"]),
            "micro_env_dim_ok": int(counts["micro_env_dim_ok"]),
            "structure_available_h5": int(counts["structure_available_h5"]),
            "structure_available_pdb": int(counts["structure_available_pdb"]),
            "structure_missing": int(counts["structure_missing"]),
            "structure_dim_ok": int(counts["structure_dim_ok"]),
            "graph_available": int(counts["graph_available"]),
            "graph_missing": int(counts["graph_missing"]),
            "window_len_ok_rate": float(counts["window_len_ok"] / inspected),
            "window_center_ok_rate": float(counts["window_center_ok"] / inspected),
            "sequence_available_rate": float(counts["sequence_available"] / inspected),
            "micro_env_available_rate": float(counts["micro_env_available"] / inspected),
            "structure_available_rate": float((counts["structure_available_h5"] + counts["structure_available_pdb"]) / inspected),
            "graph_available_rate": float(counts["graph_available"] / inspected),
        }
        for key, value in counts.items():
            global_counts[key] += int(value)
    h5_loader.close()
    if lmdb_cache is not None:
        lmdb_cache.close()
    missing_files |= (set(structure_cache.missing_files) if structure_cache is not None else set())
    report = {
        "total_samples": int(total),
        "train_samples": int(len(split_records["train"])),
        "val_samples": int(len(split_records["val"])),
        "test_samples": int(len(split_records["test"])),
        "ptm_types": ptm_types,
        "window_size": int(window_size),
        "strict_center_k": bool(strict_center_k),
        "missing_files": sorted(list(missing_files)),
        "overall": {
            "sequence_available": int(global_counts["sequence_available"]),
            "sequence_missing": int(global_counts["sequence_missing"]),
            "sequence_dim_ok": int(global_counts["sequence_dim_ok"]),
            "micro_env_available": int(global_counts["micro_env_available"]),
            "micro_env_missing": int(global_counts["micro_env_missing"]),
            "micro_env_dim_ok": int(global_counts["micro_env_dim_ok"]),
            "structure_available_h5": int(global_counts["structure_available_h5"]),
            "structure_available_pdb": int(global_counts["structure_available_pdb"]),
            "structure_missing": int(global_counts["structure_missing"]),
            "structure_dim_ok": int(global_counts["structure_dim_ok"]),
            "graph_available": int(global_counts["graph_available"]),
            "graph_missing": int(global_counts["graph_missing"]),
            "window_len_ok": int(global_counts["window_len_ok"]),
            "window_center_ok": int(global_counts["window_center_ok"]),
            "sequence_available_rate": float(global_counts["sequence_available"] / max(1, total)),
            "micro_env_available_rate": float(global_counts["micro_env_available"] / max(1, total)),
            "structure_available_rate": float((global_counts["structure_available_h5"] + global_counts["structure_available_pdb"]) / max(1, total)),
            "graph_available_rate": float(global_counts["graph_available"] / max(1, total)),
            "window_len_ok_rate": float(global_counts["window_len_ok"] / max(1, total)),
            "window_center_ok_rate": float(global_counts["window_center_ok"] / max(1, total)),
        },
        "splits": split_reports,
    }
    with open(report_dir / "data_integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(report_dir / "feature_dimension_validation.log", "w", encoding="utf-8") as f:
        f.write(f"expected_esm_dim={expected_seq_dim}\n")
        f.write(f"expected_structure_dim={expected_struct_dim}\n")
        f.write(f"checked_records={total}\n")
        f.write(f"esm_dim_match={global_counts['sequence_dim_ok']}\n")
        f.write(f"struct_dim_match={global_counts['structure_dim_ok']}\n")
        f.write(f"sequence_available_rate={report['overall']['sequence_available_rate']:.6f}\n")
        f.write(f"structure_available_rate={report['overall']['structure_available_rate']:.6f}\n")
        f.write(f"graph_available_rate={report['overall']['graph_available_rate']:.6f}\n")
        f.write(f"window_len_ok_rate={report['overall']['window_len_ok_rate']:.6f}\n")
        f.write(f"window_center_ok_rate={report['overall']['window_center_ok_rate']:.6f}\n")
        f.write(f"status={'ok' if total > 0 else 'empty'}\n")
    logger.info("Wrote data integrity report to %s", str(report_dir / "data_integrity_report.json"))
