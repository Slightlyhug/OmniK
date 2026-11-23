"""Dataset selection and construction for prepared training splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

from src.utils.training_config import DEFAULT_MOTIF_TASKS, DEFAULT_SUMO_HYDROPHOBIC, DEFAULT_PTM_TYPES, canonical_ptm_name
from src.utils.training_data import (
    MultiLabelPTMJsonDataset,
    PreprocessedTensorLMDBDataset,
    ProteinDedupLMDBDataset,
    SiteFeatureLMDBDataset,
)
import sys as _sys
from pathlib import Path as _Path
_FS_BUILD = _Path(__file__).resolve().parents[2] / "vendor" / "feature_store"
if str(_FS_BUILD) not in _sys.path:
    _sys.path.insert(0, str(_FS_BUILD))
from sharded_dataset import ShardedProteinSiteDataset
from src.utils.training_runtime import StageProfiler
from src.utils.training_state import DataPreparationArtifacts, DatasetBundle


def build_dataset_bundle(
    config: Dict[str, Any],
    artifacts: DataPreparationArtifacts,
    profiler: StageProfiler,
) -> DatasetBundle:
    dedup = config.get("protein_dedup_lmdb") if isinstance(config.get("protein_dedup_lmdb"), dict) else {}
    site = config.get("site_feature_lmdb") if isinstance(config.get("site_feature_lmdb"), dict) else {}
    v2_root = config.get("feature_store_v2_root")
    if v2_root and all((Path(v2_root) / s / "manifest.json").exists() for s in ("train", "val", "test")):
        # Feature-store labels contain all eight PTMs.  For a single-task
        # experiment explicitly project to the requested column and discard
        # unknown labels before any sampler or DataLoader sees the samples.
        target_label_index = None
        known_only = bool(config.get("single_task_known_only", False))
        if known_only:
            if len(artifacts.ptm_types) != 1:
                raise ValueError(
                    "single_task_known_only requires exactly one multitask_ptm_types entry; "
                    f"got {artifacts.ptm_types}"
                )
            target_name = canonical_ptm_name(artifacts.ptm_types[0])
            label_names = [canonical_ptm_name(x) for x in DEFAULT_PTM_TYPES]
            if target_name not in label_names:
                raise ValueError(f"Cannot map target PTM {artifacts.ptm_types[0]!r} to feature-store labels")
            target_label_index = label_names.index(target_name)
        ds = [
            ShardedProteinSiteDataset(
                v2_root, split, window=int(config.get("window_size", 73)),
                use_structure=bool(config.get("use_structure", True)),
                use_graph=bool(config.get("use_structure_graph", True)),
                structure_mask_token_value=float(config.get("structure_mask_token_value", 1.0)),
                target_label_index=target_label_index,
                known_only=known_only,
            )
            for split in ("train", "val", "test")
        ]
        return DatasetBundle(*ds)
    if site:
        require_site = bool(config.get("require_site_feature_lmdb", True))
        label_overrides = config.get("site_feature_labels_override")
        if not isinstance(label_overrides, dict):
            label_overrides = {}

        def _site_ds(split: str):
            path = site.get(split)
            if path and Path(f"{path}.meta.json").exists():
                meta = json.loads(Path(f"{path}.meta.json").read_text())
                return SiteFeatureLMDBDataset(path, use_structure=bool(config.get("use_structure", True)),
                                              use_graph=bool(config.get("use_structure_graph", True)),
                                              structure_mask_token_value=float(config.get("structure_mask_token_value", 1.0)),
                                              ram_cache=bool(config.get("site_feature_ram_cache", False)),
                                              ram_cache_preload=bool(config.get("site_feature_ram_cache_preload", False)),
                                              ram_cache_max_bytes=int(config.get("site_feature_ram_cache_max_bytes", 0) or 0),
                                              ram_cache_storage=str(config.get("site_feature_ram_cache_storage", "python")),
                                              labels_override_path=label_overrides.get(split))
            if require_site:
                raise FileNotFoundError(f"Required site feature LMDB is missing for {split}: {path}")
            return None
        ds = [_site_ds(s) for s in ("train", "val", "test")]
        if all(x is not None for x in ds):
            return DatasetBundle(*ds)
    prefer_preprocessed = bool(config.get("prefer_preprocessed_tensor_lmdb", False) or config.get("require_preprocessed_tensor_lmdb", False))
    if dedup and not prefer_preprocessed:
        require_dedup = bool(config.get("require_protein_dedup_lmdb", False))
        def _dedup_ds(split: str) -> Dataset:
            path = dedup.get(split)
            protein_path = dedup.get("protein")
            meta_path = f"{path}.meta.json" if path else None
            if path and protein_path and meta_path and Path(meta_path).exists():
                meta = json.loads(Path(meta_path).read_text())
                labels_path = meta.get("labels_path")
                labels = np.load(labels_path, mmap_mode="r") if labels_path else None
                return ProteinDedupLMDBDataset(
                    path, protein_path, int(meta["length"]), labels,
                    window_size=int(config.get("window_size", 73)),
                    local_window_size=int(config.get("local_window_size", 21)),
                    seq_feature_dim=int(config.get("seq_feature_dim", 1280)),
                    struct_feature_dim=int(config.get("struct_feature_dim", 512)),
                    cache_size=int(config.get("protein_feature_cache_size", 256)),
                    structure_mask_token_value=float(config.get("structure_mask_token_value", 1.0)),
                    graph_cache=artifacts.graph_cache,
                    use_structure_graph=artifacts.use_structure_graph,
                )
            if require_dedup:
                raise FileNotFoundError(f"Required protein dedup LMDB is missing for {split}: {path}")
            return None
        ds_train, ds_val, ds_test = _dedup_ds("train"), _dedup_ds("val"), _dedup_ds("test")
        if ds_train is not None and ds_val is not None and ds_test is not None:
            return DatasetBundle(ds_train, ds_val, ds_test)

    pre = config.get("preprocessed_tensor_lmdb") if isinstance(config.get("preprocessed_tensor_lmdb"), dict) else {}
    if pre:
        require_pre = bool(config.get("require_preprocessed_tensor_lmdb", False))
        def _pre_ds(split: str, fallback: List[Dict[str, Any]]) -> Dataset:
            path = pre.get(split)
            meta_path = f"{path}.meta.json" if path else None
            if path and meta_path and Path(meta_path).exists():
                meta = json.loads(Path(meta_path).read_text())
                labels_path = meta.get("labels_path")
                labels = np.load(labels_path, mmap_mode="r") if labels_path else None
                overlays = config.get("preprocessed_tensor_metadata_overlay", {})
                overlay_cfg = overlays.get(split, {}) if isinstance(overlays, dict) else {}
                if not isinstance(overlay_cfg, dict):
                    overlay_cfg = {}
                return PreprocessedTensorLMDBDataset(
                    path,
                    int(meta["length"]),
                    labels,
                    residue_overlay_path=overlay_cfg.get("residue_types"),
                    motif_overlay_path=overlay_cfg.get("motif_bias"),
                )
            if require_pre:
                raise FileNotFoundError(
                    f"Required preprocessed tensor LMDB is missing or incomplete for {split}: "
                    f"{path} (expected metadata {meta_path}). Run scripts/precompute_tensor_lmdb.py first."
                )
            return MultiLabelPTMJsonDataset(records=fallback, ptm_types=artifacts.ptm_types, window_size=int(config.get("window_size", 73)), local_window_size=int(config.get("local_window_size", 21)), seq_feature_dim=int(config.get("seq_feature_dim", 1280)), struct_feature_dim=int(config.get("struct_feature_dim", 512)), feature_sources=artifacts.splits.feature_sources, structure_cache=artifacts.structure_cache, use_sequence_windows=artifacts.use_sequence_windows, sequence_window_provider=artifacts.sequence_window_provider, use_structure_graph=artifacts.use_structure_graph, graph_cache=artifacts.graph_cache, disable_pdb_fallback=bool(config.get("disable_pdb_fallback", False)), use_structure_features=bool(config.get("use_structure", True)), use_micro_environment=bool(config.get("use_micro_environment", True)))
        return DatasetBundle(_pre_ds("train", artifacts.splits.train), _pre_ds("val", artifacts.splits.val), _pre_ds("test", artifacts.splits.test))
    def _build_dataset(records: List[Dict[str, Any]]) -> MultiLabelPTMJsonDataset:
        return MultiLabelPTMJsonDataset(
            records=records,
            ptm_types=artifacts.ptm_types,
            window_size=int(config.get("window_size", 73)),
            local_window_size=int(config.get("local_window_size", 21)),
            seq_feature_dim=int(config.get("seq_feature_dim", 1280)),
            struct_feature_dim=int(config.get("struct_feature_dim", 512)),
            feature_sources=artifacts.splits.feature_sources,
            structure_cache=artifacts.structure_cache,
            lazy_cache_size=int(config.get("lazy_cache_size", 128)),
            enable_structure_mask_token=bool(config.get("enable_structure_mask_token", True)),
            structure_mask_token_value=float(config.get("structure_mask_token_value", 1.0)),
            use_sequence_windows=artifacts.use_sequence_windows,
            sequence_window_provider=artifacts.sequence_window_provider,
            use_structure_graph=artifacts.use_structure_graph,
            graph_cache=artifacts.graph_cache,
            disable_pdb_fallback=bool(config.get("disable_pdb_fallback", False)),
            use_structure_features=bool(config.get("use_structure", True)),
            use_micro_environment=bool(config.get("use_micro_environment", True)),
            graph_node_dim=int(config.get("graph_node_dim", 0)),
            graph_edge_dim=int(config.get("graph_edge_dim", 0)),
            alignment_target_labels=list(config.get("sequence_window_alignment_target_labels", [])),
            always_realign_windows=bool(config.get("sequence_window_force_fasta_realign", True)),
            strict_center_k=bool(config.get("sequence_window_strict_center_k", False)),
            min_valid_flank_aa=int(config.get("sequence_window_min_valid_flank_aa", 0)),
            realign_on_center_mismatch=bool(config.get("sequence_window_realign_on_center_mismatch", False)),
            realign_on_short_context=bool(config.get("sequence_window_realign_on_short_context", False)),
            motif_bias_tasks=list(config.get("motif_bias_tasks", DEFAULT_MOTIF_TASKS)),
            motif_bias_methyl_radius=int(config.get("motif_bias_methyl_radius", 7)),
            motif_bias_crotonyl_radius=int(config.get("motif_bias_crotonyl_radius", 4)),
            motif_bias_sumo_hydrophobic=list(config.get("motif_bias_sumo_hydrophobic", DEFAULT_SUMO_HYDROPHOBIC)),
        )

    return DatasetBundle(
        train_set=profiler.run("build_train_dataset", lambda: _build_dataset(artifacts.splits.train)),
        val_set=profiler.run("build_val_dataset", lambda: _build_dataset(artifacts.splits.val)),
        test_set=profiler.run("build_test_dataset", lambda: _build_dataset(artifacts.splits.test)),
    )
