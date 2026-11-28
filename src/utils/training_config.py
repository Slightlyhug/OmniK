"""Configuration and label normalization shared by training entrypoints."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PTM_TYPES = [
    "ubiquitination", "sumoylation", "succinylation", "methylation",
    "crotonylation", "acetylation", "2hydroxyisobutyryl", "lactylation",
]
DEFAULT_MOTIF_TASKS = ["sumoylation", "methylation", "crotonylation"]
DEFAULT_SUMO_HYDROPHOBIC = ["I", "V", "L", "M", "F", "W", "Y"]

LABEL_STATE_TO_ID = {
    "unknown": 0, "positive": 1, "observed_negative": 2,
    "reliable_negative": 3, "soft_positive": 4, "conflict": 5,
}
LABEL_STATE_ALIASES = {
    "": "unknown", "unlabeled": "unknown", "missing": "unknown",
    "pos": "positive", "clean_positive": "positive", "ordinary_positive": "positive",
    "neg": "observed_negative", "negative": "observed_negative",
    "observed_neg": "observed_negative", "clean_negative": "reliable_negative",
    "reliable_neg": "reliable_negative", "ordinary_negative": "observed_negative",
    "suspect_negative": "conflict", "suspect_negative_or_missing_positive": "conflict",
    "suspect_positive": "conflict", "unknown_high_probability": "soft_positive",
    "pseudo_positive": "soft_positive", "soft_pos": "soft_positive",
}


def normalize_label_state(value: Any, label_value: Optional[float] = None) -> str:
    raw = str(value if value is not None else "").strip().lower().replace("-", "_").replace(" ", "_")
    state = LABEL_STATE_ALIASES.get(raw, raw)
    if state not in LABEL_STATE_TO_ID:
        if label_value is not None:
            try:
                y = float(label_value)
            except Exception:
                y = -1.0
            if y > 0.0:
                return "positive"
            if y == 0.0:
                return "observed_negative"
        return "unknown"
    return state


def label_state_id(value: Any, label_value: Optional[float] = None) -> int:
    return int(LABEL_STATE_TO_ID[normalize_label_state(value, label_value=label_value)])


def _resolve_config_path(path_value: Optional[str], config_dir: Path, must_exist: bool = False) -> Optional[str]:
    if path_value in (None, ""):
        return None
    path = Path(os.path.expandvars(str(path_value))).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")
    return str(path)


def sanitize_name(name: str) -> str:
    return str(name).strip().lower()


def canonical_ptm_name(name: str) -> str:
    value = "".join(ch for ch in sanitize_name(name) if ch.isalnum())
    aliases = {
        "2hydroxyisobutyrylation": "2hydroxyisobutyryl",
        "hydroxyisobutyrylation": "2hydroxyisobutyryl",
        "khib": "2hydroxyisobutyryl", "sumo": "sumoylation",
    }
    return aliases.get(value, value)


def normalize_and_validate_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    # Expand paths in nested stage overrides and feature-store settings too.
    def expand(value):
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if "${" in expanded:
                raise ValueError(f"Unset environment variable in config: {value}")
            return expanded
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value
    cfg = expand(cfg)
    config_dir = config_path.resolve().parent
    required = (
        "seq_feature_dim", "struct_feature_dim", "branch_hidden_dim", "fusion_hidden_dim",
        "batch_size", "num_epochs", "grad_accum_steps", "window_size", "local_window_size",
    )
    for key in required:
        if int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"Config field '{key}' must be a positive integer, got {cfg.get(key)}")
    if int(cfg["local_window_size"]) > int(cfg["window_size"]):
        raise ValueError("Config field 'local_window_size' must be <= 'window_size'")
    ptm_types = cfg.get("multitask_ptm_types", DEFAULT_PTM_TYPES)
    if not isinstance(ptm_types, list) or not ptm_types:
        raise ValueError("Config field 'multitask_ptm_types' must be a non-empty list")
    normalized_ptms = [sanitize_name(x) for x in ptm_types]
    if len(set(normalized_ptms)) != len(normalized_ptms):
        raise ValueError(f"Config field 'multitask_ptm_types' contains duplicates: {ptm_types}")
    cfg["multitask_ptm_types"] = normalized_ptms
    shared = {key: cfg.get(key) for key in ("shared_esm_features", "shared_pdb_dir", "shared_fasta_dir")}
    for key in ("dropout", "input_dropout", "hidden_dropout", "output_dropout", "attention_dropout", "modality_dropout"):
        if key in cfg and not 0.0 <= float(cfg[key]) < 1.0:
            raise ValueError(f"Config field '{key}' must be in [0, 1), got {cfg[key]}")
    for key in ("val_ratio", "test_ratio"):
        if key in cfg and not 0.0 <= float(cfg[key]) < 1.0:
            raise ValueError(f"Config field '{key}' must be in [0, 1), got {cfg[key]}")
    if float(cfg.get("val_ratio", 0.0)) + float(cfg.get("test_ratio", 0.0)) >= 1.0:
        raise ValueError("Config fields 'val_ratio' + 'test_ratio' must be < 1.0")
    path_keys = (
        "json_data_path", "baseline_log_path", "resume_from", "output_dir", "graph_feature_dir",
        "graph_feature_lmdb", "graph_feature_lmdb_fallback", "structure_input_dir", "data_report_dir",
    )
    defaults = {"output_dir": "multitask_lysine_output", "data_report_dir": "Lysine_small_validation/processed_multilabel_data"}
    for key in path_keys:
        cfg[key] = _resolve_config_path(cfg.get(key, defaults.get(key)), config_dir, must_exist=(key == "json_data_path" and bool(cfg.get(key))))
    if isinstance(cfg.get("esm_features_paths"), list):
        cfg["esm_features_paths"] = [_resolve_config_path(p, config_dir) for p in cfg["esm_features_paths"]]
    if isinstance(cfg.get("pdb_fasta_mapping"), dict):
        normalized = {}
        for ptm_name, mapping in cfg["pdb_fasta_mapping"].items():
            if not isinstance(mapping, dict):
                continue
            item = {}
            for key in ("pdb_dir", "fasta_dir", "graph_dir", "graph_lmdb"):
                item[key] = _resolve_config_path(mapping.get(key), config_dir)
            features = mapping.get("esm_features")
            item["esm_features"] = [_resolve_config_path(p, config_dir) for p in features] if isinstance(features, list) else _resolve_config_path(features, config_dir)
            normalized[sanitize_name(ptm_name)] = item
        cfg["pdb_fasta_mapping"] = normalized
    mapping = cfg.setdefault("pdb_fasta_mapping", {})
    for ptm in normalized_ptms:
        entry = mapping.setdefault(ptm, {})
        if shared["shared_esm_features"]:
            entry["esm_features"] = [_resolve_config_path(p, config_dir) for p in shared["shared_esm_features"]]
        if shared["shared_pdb_dir"]:
            entry["pdb_dir"] = _resolve_config_path(shared["shared_pdb_dir"], config_dir)
        if shared["shared_fasta_dir"]:
            entry["fasta_dir"] = _resolve_config_path(shared["shared_fasta_dir"], config_dir)
    return cfg
