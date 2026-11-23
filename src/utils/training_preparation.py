"""Preparation and validation of training splits and feature caches."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.utils.training_config import sanitize_name
from src.utils.training_data import GraphFeatureCache, GraphFeatureLMDBCache, PDBStructureCache
from src.utils.training_records import SequenceWindowProvider
from src.utils.training_runtime import StageProfiler
from src.utils.training_state import DataPreparationArtifacts, DataPreparationServices


def prepare_data_artifacts(
    args: argparse.Namespace,
    config: Dict[str, Any],
    ptm_types: List[str],
    output_dir: Path,
    logger: logging.Logger,
    profiler: StageProfiler,
    services: DataPreparationServices,
) -> DataPreparationArtifacts:
    _load_splits = services.load_splits
    validate_predefined_split_integrity = services.validate_split_integrity
    filter_records_with_available_pdb = services.filter_records_with_available_pdb
    filter_records_by_structure_feature_availability = services.filter_records_by_structure_feature_availability
    repair_or_filter_sequence_windows = services.repair_or_filter_sequence_windows
    filter_records_by_feature_quality = services.filter_records_by_feature_quality
    validate_split_ptm_presence = services.validate_split_ptm_presence
    motif_aware_negative_sampling = services.motif_aware_negative_sampling
    prepare_data_integrity_report = services.prepare_data_integrity_report
    if args.json_data_path:
        cfg_data_path = config.get("json_data_path")
        if cfg_data_path and Path(args.json_data_path).resolve() != Path(cfg_data_path).resolve():
            logger.warning(
                "CLI override --json_data_path=%s replaces config json_data_path=%s",
                str(Path(args.json_data_path).resolve()),
                str(Path(cfg_data_path).resolve()),
            )

    fast_start = bool(config.get("fast_start", False))
    if fast_start:
        config["run_data_integrity_report"] = bool(config.get("fast_start_run_data_integrity_report", False))
        config["sequence_window_force_fasta_realign"] = bool(config.get("fast_start_sequence_window_force_fasta_realign", False))
        config["benchmark_dataloader_first_batch"] = bool(config.get("fast_start_benchmark_dataloader_first_batch", False))
        config["disable_pdb_fallback"] = bool(config.get("fast_start_disable_pdb_fallback", True))
        logger.info(
            "Fast-start mode enabled: integrity_report=%s force_fasta_realign=%s first_batch_benchmark=%s disable_pdb_fallback=%s",
            config["run_data_integrity_report"],
            config["sequence_window_force_fasta_realign"],
            config["benchmark_dataloader_first_batch"],
            config["disable_pdb_fallback"],
        )

    splits = profiler.run("load_splits", lambda: _load_splits(args, config))
    if splits.source_path:
        logger.info("Using split source: %s", splits.source_path)
    if splits.split_policy:
        logger.info("Split policy: %s", json.dumps(splits.split_policy, ensure_ascii=False, sort_keys=True))
    if bool(config.get("validate_split_integrity_on_startup", not fast_start)):
        profiler.run(
            "validate_split_integrity",
            lambda: validate_predefined_split_integrity(
                splits,
                logger,
                allow_protein_overlap=bool(config.get("allow_protein_overlap_across_splits", False)),
                allow_non_lysine_sites=bool(config.get("allow_non_lysine_sites", False)),
            ),
        )
    else:
        logger.info("Split integrity validation skipped for startup performance.")
    logger.info("Data split sizes: train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    report_dir = Path(config.get("data_report_dir", output_dir)).resolve()
    cache_dir = report_dir / "cache"
    pdb_mapping = config.get("pdb_fasta_mapping") if isinstance(config.get("pdb_fasta_mapping"), dict) else {}
    use_structure_features = bool(config.get("use_structure", True))
    use_micro_environment = bool(config.get("use_micro_environment", True))
    disable_pdb_fallback = bool(config.get("disable_pdb_fallback", False))
    micro_env_pdb_fallback = bool(config.get("micro_env_pdb_fallback", True))
    if config.get("feature_store_v2_root"):
        structure_cache = None
        logger.info("feature_store_v2 enabled; skipping PDB structure cache.")
    elif disable_pdb_fallback and not micro_env_pdb_fallback and not bool(config.get("drop_samples_with_missing_pdb", False)):
        structure_cache = None
        logger.info("PDB fallback disabled; missing structure_features and micro_env_features will use mask/zero fallback.")
    elif use_structure_features or use_micro_environment:
        structure_cache = profiler.run(
            "build_structure_cache",
            lambda: PDBStructureCache(
                pdb_fasta_mapping=pdb_mapping,
                cache_dir=cache_dir,
                struct_feature_dim=int(config.get("struct_feature_dim", 512)),
                logger=logger,
                missing_pdb_log_limit=int(config.get("missing_pdb_log_limit", 20)),
            ),
        )
        if disable_pdb_fallback:
            logger.info("Structure window PDB fallback disabled; PDB cache kept for real micro_env_features fallback.")
    else:
        structure_cache = None
        logger.info("Structure and micro-environment features disabled; skipping PDB cache.")
    if bool(config.get("drop_samples_with_missing_pdb", False)):
        splits.train = filter_records_with_available_pdb(splits.train, ptm_types, structure_cache, logger, split_name="train")
        splits.val = filter_records_with_available_pdb(splits.val, ptm_types, structure_cache, logger, split_name="val")
        splits.test = filter_records_with_available_pdb(splits.test, ptm_types, structure_cache, logger, split_name="test")
    else:
        logger.info("Keeping samples with missing PDB and using structure-mask token fallback.")

    def _apply_structure_record_filter(split_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mode = str(config.get(f"structure_record_filter_{split_name}", config.get("structure_record_filter", "all"))).strip().lower()
        if mode in {"", "all", "none", "off", "false", "0"}:
            return records
        if mode in {"with_structure", "structure", "available", "has_structure"}:
            return filter_records_by_structure_feature_availability(
                records,
                ptm_types=ptm_types,
                feature_sources=splits.feature_sources,
                window_size=int(config.get("window_size", 73)),
                keep_available=True,
                logger=logger,
                split_name=split_name,
            )
        if mode in {"without_structure", "missing_structure", "no_structure", "sequence_only", "missing"}:
            return filter_records_by_structure_feature_availability(
                records,
                ptm_types=ptm_types,
                feature_sources=splits.feature_sources,
                window_size=int(config.get("window_size", 73)),
                keep_available=False,
                logger=logger,
                split_name=split_name,
            )
        raise ValueError(f"Unsupported structure_record_filter_{split_name}={mode!r}")

    splits.train = _apply_structure_record_filter("train", splits.train)
    splits.val = _apply_structure_record_filter("val", splits.val)
    splits.test = _apply_structure_record_filter("test", splits.test)
    logger.info("Data split sizes after structure filters: train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    use_esm_lora = bool(config.get("use_esm_lora", False))
    use_sequence_windows = bool(config.get("use_sequence_windows", False) or use_esm_lora)
    fasta_dirs: List[str] = []
    mapping = config.get("pdb_fasta_mapping") if isinstance(config.get("pdb_fasta_mapping"), dict) else {}
    for mapped in mapping.values():
        if isinstance(mapped, dict) and mapped.get("fasta_dir"):
            fasta_dirs.append(str(mapped.get("fasta_dir")))
    fasta_dirs = sorted(list({d for d in fasta_dirs if d}))
    sequence_window_provider = None
    if use_sequence_windows and fasta_dirs:
        sequence_window_provider = SequenceWindowProvider(
            fasta_dirs=fasta_dirs,
            window_size=int(config.get("window_size", 73)),
            logger=logger,
        )
        if bool(config.get("sequence_window_force_fasta_realign", True)):
            drop_unrepairable = bool(config.get("sequence_window_drop_unrepairable", True))
            splits.train = profiler.run(
                "repair_train_sequence_windows",
                lambda: repair_or_filter_sequence_windows(
                    splits.train,
                    sequence_window_provider=sequence_window_provider,
                    window_size=int(config.get("window_size", 73)),
                    strict_center_k=bool(config.get("sequence_window_strict_center_k", False)),
                    drop_unrepairable=drop_unrepairable,
                    logger=logger,
                    split_name="train",
                ),
            )
            splits.val = profiler.run(
                "repair_val_sequence_windows",
                lambda: repair_or_filter_sequence_windows(
                    splits.val,
                    sequence_window_provider=sequence_window_provider,
                    window_size=int(config.get("window_size", 73)),
                    strict_center_k=bool(config.get("sequence_window_strict_center_k", False)),
                    drop_unrepairable=drop_unrepairable,
                    logger=logger,
                    split_name="val",
                ),
            )
            splits.test = profiler.run(
                "repair_test_sequence_windows",
                lambda: repair_or_filter_sequence_windows(
                    splits.test,
                    sequence_window_provider=sequence_window_provider,
                    window_size=int(config.get("window_size", 73)),
                    strict_center_k=bool(config.get("sequence_window_strict_center_k", False)),
                    drop_unrepairable=drop_unrepairable,
                    logger=logger,
                    split_name="test",
                ),
            )

    use_structure_graph = bool(config.get("use_structure_graph", False))
    logger.info(f"Structure Graph feature is {'enabled' if use_structure_graph else 'disabled'}.")
    graph_dirs_by_ptm: Dict[str, str] = {}
    default_graph_dir = config.get("graph_feature_dir")
    graph_lmdbs_by_ptm: Dict[str, str] = {}
    default_graph_lmdb = config.get("graph_feature_lmdb")
    if use_structure_graph:
        for ptm in ptm_types:
            key = sanitize_name(ptm)
            mapped = mapping.get(key, {}) if isinstance(mapping, dict) else {}
            if isinstance(mapped, dict) and mapped.get("graph_dir"):
                graph_dirs_by_ptm[key] = str(mapped.get("graph_dir"))
            if isinstance(mapped, dict) and mapped.get("graph_lmdb"):
                graph_lmdbs_by_ptm[key] = str(mapped.get("graph_lmdb"))

    feature_filter_cfg = config.get("feature_filtering") if isinstance(config.get("feature_filtering"), dict) else {}
    strict_center_k = bool(config.get("sequence_window_strict_center_k", False))
    window_size = int(config.get("window_size", 73))
    splits.train = profiler.run(
        "feature_quality_filter_train",
        lambda: filter_records_by_feature_quality(
            splits.train,
            ptm_types=ptm_types,
            feature_sources=splits.feature_sources,
            window_size=window_size,
            strict_center_k=strict_center_k,
            feature_filter_cfg=feature_filter_cfg,
            report_dir=report_dir,
            logger=logger,
            split_name="train",
        ),
    )
    splits.val = profiler.run(
        "feature_quality_filter_val",
        lambda: filter_records_by_feature_quality(
            splits.val,
            ptm_types=ptm_types,
            feature_sources=splits.feature_sources,
            window_size=window_size,
            strict_center_k=strict_center_k,
            feature_filter_cfg=feature_filter_cfg,
            report_dir=report_dir,
            logger=logger,
            split_name="val",
        ),
    )
    splits.test = profiler.run(
        "feature_quality_filter_test",
        lambda: filter_records_by_feature_quality(
            splits.test,
            ptm_types=ptm_types,
            feature_sources=splits.feature_sources,
            window_size=window_size,
            strict_center_k=strict_center_k,
            feature_filter_cfg=feature_filter_cfg,
            report_dir=report_dir,
            logger=logger,
            split_name="test",
        ),
    )

    if bool(config.get("validate_split_ptm_presence_on_startup", True)):
        profiler.run("validate_split_ptm_presence", lambda: validate_split_ptm_presence(splits, ptm_types))
    else:
        logger.info("Split PTM presence validation skipped for startup performance.")
    splits.train = profiler.run(
        "motif_aware_negative_sampling",
        lambda: motif_aware_negative_sampling(splits.train, ptm_types, config, logger),
    )
    profiler.run(
        "prepare_data_integrity_report",
        lambda: prepare_data_integrity_report(
            splits=splits,
            ptm_types=ptm_types,
            report_dir=report_dir,
            structure_cache=structure_cache,
            feature_sources=splits.feature_sources,
            graph_dirs_by_ptm=graph_dirs_by_ptm,
            default_graph_dir=default_graph_dir,
            graph_lmdbs_by_ptm=graph_lmdbs_by_ptm,
            default_graph_lmdb=default_graph_lmdb,
            window_size=window_size,
            strict_center_k=strict_center_k,
            expected_seq_dim=int(config.get("seq_feature_dim", 1280)),
            expected_struct_dim=int(config.get("struct_feature_dim", 512)),
            logger=logger,
            enabled=bool(config.get("run_data_integrity_report", False)),
        ),
    )
    graph_cache = None
    if use_structure_graph and not config.get("feature_store_v2_root"):
        if default_graph_lmdb or graph_lmdbs_by_ptm:
            graph_cache = GraphFeatureLMDBCache(
                graph_lmdbs_by_ptm,
                default_graph_lmdb,
                config.get("graph_feature_lmdb_fallback"),
            )
        else:
            graph_cache = GraphFeatureCache(graph_dirs_by_ptm, default_graph_dir)

    return DataPreparationArtifacts(
        ptm_types=ptm_types,
        splits=splits,
        report_dir=report_dir,
        structure_cache=structure_cache,
        sequence_window_provider=sequence_window_provider,
        use_sequence_windows=use_sequence_windows,
        use_structure_graph=use_structure_graph,
        graph_cache=graph_cache,
    )

