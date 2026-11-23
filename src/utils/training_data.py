"""Feature stores, datasets and batch collation for multitask training."""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pickle
import random
import re
import time
import zlib
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.utils.training_config import DEFAULT_MOTIF_TASKS, DEFAULT_PTM_TYPES, sanitize_name, label_state_id
from src.utils.training_records import _labels_to_array, _resolve_ptm_from_record, compute_motif_bias_scores

try:
    import lmdb
    HAS_LMDB = True
except Exception:
    lmdb = None
    HAS_LMDB = False

try:
    from Bio.PDB import DSSP, PDBParser
    HAS_BIO_PDB = True
except Exception:
    DSSP = None
    PDBParser = None
    HAS_BIO_PDB = False


class H5DataLoader:
    def __init__(self, feature_sources: Dict[str, Any]) -> None:
        self.feature_sources = {
            sanitize_name(k): ([str(p) for p in v] if isinstance(v, list) else [str(v)])
            for k, v in (feature_sources or {}).items()
        }
        self._handles: Dict[str, h5py.File] = {}
        self._lmdb_envs: Dict[str, Any] = {}
        self._lmdb_record_cache: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()
        self._lmdb_record_cache_limit = 64

    def _get_handle(self, source: str) -> Optional[h5py.File]:
        src = sanitize_name(source)
        paths = self.feature_sources.get(src, [])
        path = next((p for p in paths if not Path(p).is_dir()), None)
        if path is None:
            return None
        if src in self._handles:
            return self._handles[src]
        if not os.path.exists(path):
            return None
        try:
            self._handles[src] = h5py.File(
                path,
                "r",
                libver="latest",
                swmr=True,
                rdcc_nbytes=64 * 1024 * 1024,
                rdcc_nslots=131071,
            )
        except Exception:
            self._handles[src] = h5py.File(path, "r")
        return self._handles[src]

    def get_dataset(self, source: str, uniprot_id: str, dataset_name: str) -> Optional[h5py.Dataset]:
        handle = self._get_handle(source)
        if handle is not None and "proteins" in handle and uniprot_id in handle["proteins"]:
            grp = handle["proteins"][uniprot_id]
            if dataset_name in grp:
                return grp[dataset_name]
        if HAS_LMDB and lmdb is not None:
            for path in self.feature_sources.get(sanitize_name(source), []):
                if not Path(path).is_dir():
                    continue
                env = self._lmdb_envs.get(path)
                if env is None:
                    env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=2048)
                    self._lmdb_envs[path] = env
                cache_key = (str(path), str(uniprot_id))
                record = self._lmdb_record_cache.get(cache_key)
                if record is None:
                    with env.begin(buffers=True) as txn:
                        raw = txn.get(str(uniprot_id).encode())
                    if raw is not None:
                        record = pickle.loads(bytes(raw))
                        self._lmdb_record_cache[cache_key] = record
                        self._lmdb_record_cache.move_to_end(cache_key)
                        while len(self._lmdb_record_cache) > self._lmdb_record_cache_limit:
                            self._lmdb_record_cache.popitem(last=False)
                if record is not None and dataset_name in record:
                    return np.asarray(record[dataset_name])
        return None

    def get_vector(self, source: str, uniprot_id: str, dataset_name: str) -> Optional[np.ndarray]:
        ds = self.get_dataset(source, uniprot_id, dataset_name)
        if ds is None:
            return None
        arr = np.asarray(ds[()], dtype=np.float32)
        if arr.size == 0:
            return arr
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=-50.0)
        return np.clip(arr, -50.0, 50.0).astype(np.float32, copy=False)

    def get_site_vector(self, source: str, uniprot_id: str, dataset_name: str, center_pos_1based: int) -> Optional[np.ndarray]:
        ds = self.get_dataset(source, uniprot_id, dataset_name)
        if ds is None:
            return None
        if ds.ndim == 1:
            arr = np.asarray(ds[()], dtype=np.float32)
        elif ds.ndim == 2:
            idx = int(center_pos_1based) - 1
            if idx < 0 or idx >= int(ds.shape[0]):
                return None
            arr = np.asarray(ds[idx], dtype=np.float32)
        else:
            return None
        if arr.size == 0:
            return arr
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=-50.0)
        return np.clip(arr, -50.0, 50.0).astype(np.float32, copy=False)

    def get_window(self, source: str, uniprot_id: str, dataset_name: str, center_pos_1based: int, length: int) -> Optional[np.ndarray]:
        ds = self.get_dataset(source, uniprot_id, dataset_name)
        if ds is None:
            return None
        if ds.ndim != 2:
            arr = np.asarray(ds[()], dtype=np.float32)
            if arr.ndim != 2:
                return None
            seq_len, feat_dim = arr.shape
            idx = int(center_pos_1based) - 1
            half = int(length) // 2
            start = max(0, idx - half)
            end = min(seq_len, idx + half + 1)
            win = arr[start:end]
        else:
            seq_len, feat_dim = ds.shape
            idx = int(center_pos_1based) - 1
            half = int(length) // 2
            start = max(0, idx - half)
            end = min(seq_len, idx + half + 1)
            win = np.asarray(ds[start:end], dtype=np.float32)
        if win.shape[0] < length:
            left_pad = max(0, half - idx)
            right_pad = max(0, length - win.shape[0] - left_pad)
            if left_pad > 0:
                win = np.concatenate([np.zeros((left_pad, feat_dim), dtype=np.float32), win], axis=0)
            if right_pad > 0:
                win = np.concatenate([win, np.zeros((right_pad, feat_dim), dtype=np.float32)], axis=0)
        elif win.shape[0] > length:
            win = win[:length]
        if not np.isfinite(win).all():
            win = np.nan_to_num(win, nan=0.0, posinf=50.0, neginf=-50.0)
        return np.clip(win, -50.0, 50.0).astype(np.float32, copy=False)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        for env in self._lmdb_envs.values():
            env.close()
        self._handles = {}
        self._lmdb_envs = {}
        self._lmdb_record_cache = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_lmdb_envs"] = {}
        state["_lmdb_record_cache"] = OrderedDict()
        return state


class PDBStructureCache:
    def __init__(
        self,
        pdb_fasta_mapping: Dict[str, Dict[str, str]],
        cache_dir: Path,
        struct_feature_dim: int,
        logger: logging.Logger,
        missing_pdb_log_limit: int = 20,
    ) -> None:
        self.mapping = {sanitize_name(k): v for k, v in (pdb_fasta_mapping or {}).items()}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.struct_feature_dim = int(struct_feature_dim)
        self.logger = logger
        self.missing_files: Set[str] = set()
        self.missing_pdb_log_limit = max(0, int(missing_pdb_log_limit))
        self._missing_pdb_logged_count = 0
        self._missing_pdb_suppressed = False
        self._pdb_index_cache: Dict[str, Dict[str, Path]] = {}
        self._pdb_dir_index_cache: Dict[str, Dict[str, Path]] = {}
        self._resolve_cache: Dict[Tuple[str, str], Optional[Path]] = {}
        self._raw_cache: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()
        self._raw_cache_limit = 1024
        self._failed_cache: Set[Tuple[str, str]] = set()

    def _build_pdb_index(self, ptm_type: str) -> Dict[str, Path]:
        key = sanitize_name(ptm_type)
        if key in self._pdb_index_cache:
            return self._pdb_index_cache[key]
        cfg = self.mapping.get(key, {})
        pdb_dir = cfg.get("pdb_dir")
        if not pdb_dir:
            self._pdb_index_cache[key] = {}
            return self._pdb_index_cache[key]
        pdb_root = Path(pdb_dir)
        dir_key = str(pdb_root.resolve())
        if dir_key in self._pdb_dir_index_cache:
            self._pdb_index_cache[key] = self._pdb_dir_index_cache[dir_key]
            return self._pdb_index_cache[key]
        index: Dict[str, Path] = {}
        if pdb_root.exists() and pdb_root.is_dir():
            for path in pdb_root.iterdir():
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in {".pdb", ".ent"}:
                    continue
                stem_key = path.stem.upper()
                if stem_key not in index:
                    index[stem_key] = path
                prefix = stem_key.split("_")[0]
                if prefix and prefix not in index:
                    index[prefix] = path
        self._pdb_dir_index_cache[dir_key] = index
        self._pdb_index_cache[key] = index
        return index

    def _resolve_pdb_path(self, uniprot_id: str, ptm_type: str) -> Optional[Path]:
        cache_key = (str(uniprot_id).upper(), sanitize_name(ptm_type))
        if cache_key in self._resolve_cache:
            return self._resolve_cache[cache_key]
        cfg = self.mapping.get(sanitize_name(ptm_type), {})
        pdb_dir = cfg.get("pdb_dir")
        if not pdb_dir:
            raise FileNotFoundError(f"Missing pdb_dir mapping for ptm_type={ptm_type}")
        pdb_root = Path(pdb_dir)
        pdb_index = self._build_pdb_index(ptm_type)
        index_hit = pdb_index.get(str(uniprot_id).upper()) or pdb_index.get(str(uniprot_id).split("-")[0].upper())
        if index_hit is not None and index_hit.exists():
            self._resolve_cache[cache_key] = index_hit
            return index_hit
        candidates = [
            pdb_root / f"{uniprot_id}.pdb",
            pdb_root / f"{uniprot_id}.ent",
            pdb_root / f"{uniprot_id.lower()}.pdb",
            pdb_root / f"{uniprot_id.lower()}.ent",
        ]
        for path in candidates:
            if path.exists():
                self._resolve_cache[cache_key] = path
                return path
        self._resolve_cache[cache_key] = None
        return None

    def _extract_raw_features(self, pdb_path: Path) -> Dict[str, Any]:
        if not HAS_BIO_PDB:
            raise RuntimeError("Bio.PDB is unavailable")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_path))
        model = next(structure.get_models())
        residues = []
        for chain in model:
            for residue in chain:
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                ca = residue["CA"].get_coord()
                residues.append((chain.id, int(residue.id[1]), np.asarray(ca, dtype=np.float32)))
        if not residues:
            raise RuntimeError(f"No C-alpha residues parsed from {pdb_path}")
        ss_lookup: Dict[Tuple[str, int], np.ndarray] = {}
        rsa_lookup: Dict[Tuple[str, int], float] = {}
        if DSSP is not None:
            try:
                dssp = DSSP(model, str(pdb_path))
                ss_vocab = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7}
                for key in dssp.keys():
                    chain_id, residue_id = key
                    if isinstance(residue_id, tuple):
                        resid = int(residue_id[1])
                    else:
                        resid = int(residue_id)
                    dssp_row = dssp[key]
                    ss_char = str(dssp_row[2]) if len(dssp_row) > 2 else "-"
                    rsa_value = float(dssp_row[3]) if len(dssp_row) > 3 else 0.0
                    ss_onehot = np.zeros((8,), dtype=np.float32)
                    ss_onehot[ss_vocab.get(ss_char, 7)] = 1.0
                    ss_lookup[(chain_id, resid)] = ss_onehot
                    rsa_lookup[(chain_id, resid)] = rsa_value
            except Exception:
                pass
        residue_numbers = []
        feat_rows = []
        for chain_id, resid, ca in residues:
            ss = ss_lookup.get((chain_id, resid), np.zeros((8,), dtype=np.float32))
            rsa = np.asarray([rsa_lookup.get((chain_id, resid), 0.0)], dtype=np.float32)
            row = np.concatenate([ca.astype(np.float32), ss, rsa], axis=0)
            residue_numbers.append(resid)
            feat_rows.append(row.astype(np.float32))
        return {
            "residue_numbers": residue_numbers,
            "structure_features": np.asarray(feat_rows, dtype=np.float32),
        }

    @staticmethod
    def _micro_environment_from_cached(cached: Dict[str, Any], position: int) -> Optional[np.ndarray]:
        residue_numbers = np.asarray(cached.get("residue_numbers", []), dtype=np.int32)
        raw_feat = np.asarray(cached.get("structure_features", []), dtype=np.float32)
        if residue_numbers.size == 0 or raw_feat.ndim != 2 or raw_feat.shape[0] == 0 or raw_feat.shape[1] < 3:
            return None
        matches = np.where(residue_numbers == int(position))[0]
        if matches.size == 0:
            return None
        center_idx = int(matches[0])
        coords = raw_feat[:, :3].astype(np.float32, copy=False)
        center_coord = coords[center_idx]
        distances = np.linalg.norm(coords - center_coord, axis=1)
        valid_mask = np.ones((distances.shape[0],), dtype=bool)
        valid_mask[center_idx] = False
        neighbor_distances = distances[valid_mask]
        neighbor_positions = residue_numbers[valid_mask]
        if neighbor_distances.size == 0:
            return None
        near8_mask = neighbor_distances <= 8.0
        near12_mask = neighbor_distances <= 12.0
        k = min(5, int(neighbor_distances.size))
        nearest = np.partition(neighbor_distances, k - 1)[:k] if k > 0 else np.asarray([], dtype=np.float32)
        seq_offsets = np.abs(neighbor_positions.astype(np.float32) - float(position))
        seq_offsets_12 = seq_offsets[near12_mask]
        features = np.asarray([
            float(np.sum(near8_mask)),
            float(np.sum(near12_mask)),
            float(np.mean(nearest)) if nearest.size else 0.0,
            float(np.std(nearest)) if nearest.size else 0.0,
            float(np.mean(seq_offsets_12)) if seq_offsets_12.size else 0.0,
            float(np.sum(np.exp(-neighbor_distances / 10.0))),
        ], dtype=np.float32)
        return np.nan_to_num(features, nan=0.0, posinf=1e4, neginf=0.0).astype(np.float32, copy=False)

    def get_micro_environment(self, uniprot_id: str, ptm_type: str, position: int) -> Optional[np.ndarray]:
        try:
            cached = self.load_or_create(uniprot_id, ptm_type)
            return self._micro_environment_from_cached(cached, int(position))
        except Exception as exc:
            msg = f"Micro-environment unavailable for {uniprot_id} ({ptm_type}) at {position}: {exc}"
            is_new_missing = msg not in self.missing_files
            self.missing_files.add(msg)
            if is_new_missing:
                if self._missing_pdb_logged_count < self.missing_pdb_log_limit:
                    self.logger.warning(msg)
                    self._missing_pdb_logged_count += 1
                elif not self._missing_pdb_suppressed:
                    self.logger.warning("Too many missing PDB/micro-environment warnings. Remaining entries are suppressed.")
                    self._missing_pdb_suppressed = True
            return None

    def load_or_create(self, uniprot_id: str, ptm_type: str) -> Dict[str, Any]:
        pdb_id = str(uniprot_id)
        cache_key = (pdb_id.upper(), sanitize_name(ptm_type))
        if cache_key in self._failed_cache:
            raise FileNotFoundError(f"PDB previously failed for {uniprot_id} ({ptm_type})")
        cached_raw = self._raw_cache.get(cache_key)
        if cached_raw is not None:
            self._raw_cache.move_to_end(cache_key)
            return cached_raw
        cache_path = self.cache_dir / f"{pdb_id}_{sanitize_name(ptm_type)}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            raw = {
                "residue_numbers": [int(x) for x in data["residue_numbers"]],
                "structure_features": np.asarray(data["structure_features"], dtype=np.float32),
            }
            self._raw_cache[cache_key] = raw
            self._raw_cache.move_to_end(cache_key)
            while len(self._raw_cache) > self._raw_cache_limit:
                self._raw_cache.popitem(last=False)
            return raw
        pdb_path = self._resolve_pdb_path(uniprot_id, ptm_type)
        if pdb_path is None:
            self._failed_cache.add(cache_key)
            cfg = self.mapping.get(sanitize_name(ptm_type), {})
            pdb_dir = cfg.get("pdb_dir", "<unset>")
            raise FileNotFoundError(f"PDB not found for {uniprot_id} in {pdb_dir}")
        try:
            raw = self._extract_raw_features(pdb_path)
        except Exception:
            self._failed_cache.add(cache_key)
            raise
        torch.save(
            {
                "residue_numbers": raw["residue_numbers"],
                "structure_features": torch.tensor(raw["structure_features"], dtype=torch.float32),
            },
            cache_path,
        )
        self._raw_cache[cache_key] = raw
        self._raw_cache.move_to_end(cache_key)
        while len(self._raw_cache) > self._raw_cache_limit:
            self._raw_cache.popitem(last=False)
        return raw

    def align_to_sequence(self, uniprot_id: str, ptm_type: str, seq_len: int) -> Optional[np.ndarray]:
        pdb_id = str(uniprot_id)
        cache_key = (pdb_id.upper(), sanitize_name(ptm_type))
        if cache_key in self._failed_cache:
            return None
        cache_path = self.cache_dir / f"{pdb_id}_{sanitize_name(ptm_type)}.pt"
        try:
            if cache_path.exists():
                data = torch.load(cache_path, map_location="cpu", weights_only=True)
                cached = {
                    "residue_numbers": [int(x) for x in data["residue_numbers"]],
                    "structure_features": np.asarray(data["structure_features"], dtype=np.float32),
                }
            else:
                pdb_path = self._resolve_pdb_path(uniprot_id, ptm_type)
                if pdb_path is None:
                    cfg = self.mapping.get(sanitize_name(ptm_type), {})
                    pdb_dir = cfg.get("pdb_dir", "<unset>")
                    msg = f"PDB not found for {uniprot_id} in {pdb_dir}"
                    is_new_missing = msg not in self.missing_files
                    self.missing_files.add(msg)
                    if is_new_missing:
                        if self._missing_pdb_logged_count < self.missing_pdb_log_limit:
                            self.logger.warning(msg)
                            self._missing_pdb_logged_count += 1
                        elif not self._missing_pdb_suppressed:
                            self.logger.warning("Too many missing PDB warnings. Remaining missing PDB entries are suppressed.")
                            self._missing_pdb_suppressed = True
                    return None
                raw = self._extract_raw_features(pdb_path)
                torch.save(
                    {
                        "residue_numbers": raw["residue_numbers"],
                        "structure_features": torch.tensor(raw["structure_features"], dtype=torch.float32),
                    },
                    cache_path,
                )
                cached = raw
        except Exception as exc:
            self.logger.warning("Failed to parse structure for %s (%s): %s", uniprot_id, ptm_type, str(exc))
            return None
        residue_numbers = cached["residue_numbers"]
        raw_feat = cached["structure_features"]
        base_dim = raw_feat.shape[1]
        aligned = np.zeros((max(1, int(seq_len)), base_dim), dtype=np.float32)
        for i, resid in enumerate(residue_numbers):
            idx = int(resid) - 1
            if 0 <= idx < aligned.shape[0] and i < raw_feat.shape[0]:
                aligned[idx] = raw_feat[i]
        if base_dim < self.struct_feature_dim:
            pad = np.zeros((aligned.shape[0], self.struct_feature_dim - base_dim), dtype=np.float32)
            aligned = np.concatenate([aligned, pad], axis=1)
        elif base_dim > self.struct_feature_dim:
            aligned = aligned[:, : self.struct_feature_dim]
        return aligned.astype(np.float32, copy=False)


class H5FeatureStoreManager:
    def __init__(
        self,
        feature_sources: Dict[str, str],
        cache_size: int = 128,
        structure_cache: Optional[PDBStructureCache] = None,
        ptm_types: Optional[List[str]] = None,
        disable_pdb_fallback: bool = False,
        use_structure_features: bool = True,
        use_micro_environment: bool = True,
    ) -> None:
        self.feature_sources = {sanitize_name(k): str(v) for k, v in (feature_sources or {}).items()}
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[Tuple[str, str, int, int], Dict[str, Optional[np.ndarray]]] = OrderedDict()
        self.h5_loader = H5DataLoader(self.feature_sources)
        self.structure_cache = structure_cache
        self.disable_pdb_fallback = bool(disable_pdb_fallback)
        self.use_structure_features = bool(use_structure_features)
        self.use_micro_environment = bool(use_micro_environment)
        self.ptm_types = [sanitize_name(x) for x in (ptm_types or [])]
        self._uniprot_source_cache: Dict[str, str] = {}

    def _resolve_source(self, record_source: Optional[str], ptm_type: str, uniprot_id: str) -> str:
        uid = str(uniprot_id or "").strip()
        if uid and uid in self._uniprot_source_cache:
            return self._uniprot_source_cache[uid]

        preferred = sanitize_name(record_source) if record_source else ""
        ptm_key = sanitize_name(ptm_type)

        candidates: List[str] = []
        for cand in (preferred, ptm_key):
            if cand and cand not in candidates:
                candidates.append(cand)
        for cand in self.ptm_types:
            if cand and cand not in candidates:
                candidates.append(cand)
        for cand in self.feature_sources.keys():
            if cand and cand not in candidates:
                candidates.append(cand)

        resolved = preferred or ptm_key
        if uid:
            for cand in candidates:
                if self.h5_loader.get_dataset(cand, uid, "sequence_features") is not None:
                    resolved = cand
                    break
            self._uniprot_source_cache[uid] = resolved
        return resolved

    def _load_site(self, source: str, uniprot_id: str, position: int, window_size: int, local_window_size: int, ptm_type: str) -> Dict[str, Optional[np.ndarray]]:
        key = (sanitize_name(source), str(uniprot_id), int(position), int(window_size))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        sequence_window = self.h5_loader.get_window(source, uniprot_id, "sequence_features", position, window_size)
        sequence_local = self.h5_loader.get_window(source, uniprot_id, "sequence_features", position, local_window_size)
        sequence_global = self.h5_loader.get_vector(source, uniprot_id, "sequence_mean")
        structure_global = None
        structure_window = None
        if self.use_structure_features:
            structure_global = self.h5_loader.get_vector(source, uniprot_id, "structure_mean")
            if structure_global is None:
                structure_global = self.h5_loader.get_vector(source, uniprot_id, "structure_cls")
            structure_window = self.h5_loader.get_window(source, uniprot_id, "structure_features", position, window_size)
        micro_env = self.h5_loader.get_site_vector(source, uniprot_id, "micro_env_features", position) if self.use_micro_environment else None
        micro_env_available = micro_env is not None and bool(np.any(np.asarray(micro_env, dtype=np.float32).reshape(-1) != 0.0))
        if self.structure_cache is not None and (micro_env is None or not micro_env_available):
            pdb_micro = self.structure_cache.get_micro_environment(uniprot_id=uniprot_id, ptm_type=ptm_type, position=position)
            if pdb_micro is not None and bool(np.any(np.asarray(pdb_micro, dtype=np.float32).reshape(-1) != 0.0)):
                micro_env = pdb_micro
                micro_env_available = True
        if structure_window is None and self.structure_cache is not None and not self.disable_pdb_fallback:
            seq_dataset = self.h5_loader.get_dataset(source, uniprot_id, "sequence_features")
            seq_len = int(seq_dataset.shape[0]) if seq_dataset is not None and seq_dataset.ndim == 2 else window_size
            aligned_struct = self.structure_cache.align_to_sequence(uniprot_id=uniprot_id, ptm_type=ptm_type, seq_len=seq_len)
            if aligned_struct is not None:
                structure_window = self._extract_window(aligned_struct, position, window_size)
        site = {
            "sequence_window": sequence_window,
            "sequence_local": sequence_local,
            "sequence_global": sequence_global,
            "structure_window": structure_window,
            "structure_global": structure_global,
            "micro_env": micro_env,
            "micro_env_available": np.asarray([1.0 if micro_env_available else 0.0], dtype=np.float32),
        }
        self._cache[key] = site
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return site

    @staticmethod
    def _extract_window(features: Optional[np.ndarray], center_pos_1based: int, length: int) -> Optional[np.ndarray]:
        if features is None:
            return None
        if features.ndim != 2:
            return None

        seq_len, feat_dim = features.shape
        idx = int(center_pos_1based) - 1
        half = length // 2
        start = max(0, idx - half)
        end = min(seq_len, idx + half + 1)
        win = features[start:end]

        if win.shape[0] < length:
            left_pad = max(0, half - idx)
            right_pad = max(0, length - win.shape[0] - left_pad)
            if left_pad > 0:
                win = np.concatenate([np.zeros((left_pad, feat_dim), dtype=features.dtype), win], axis=0)
            if right_pad > 0:
                win = np.concatenate([win, np.zeros((right_pad, feat_dim), dtype=features.dtype)], axis=0)
        elif win.shape[0] > length:
            win = win[:length]
        return win.astype(np.float32, copy=False)

    def get_site_features(
        self,
        source: str,
        uniprot_id: str,
        position: int,
        window_size: int,
        local_window_size: int,
        ptm_type: str,
    ) -> Dict[str, Optional[np.ndarray]]:
        resolved_source = self._resolve_source(source, ptm_type, uniprot_id)
        return self._load_site(
            source=resolved_source,
            uniprot_id=uniprot_id,
            position=position,
            window_size=window_size,
            local_window_size=local_window_size,
            ptm_type=ptm_type,
        )

    def close(self) -> None:
        self.h5_loader.close()


class GraphFeatureCache:
    def __init__(
        self,
        graph_dirs_by_ptm: Dict[str, str],
        default_graph_dir: Optional[str] = None,
    ) -> None:
        self.graph_dirs_by_ptm = {sanitize_name(k): str(v) for k, v in (graph_dirs_by_ptm or {}).items()}
        self.default_graph_dir = str(default_graph_dir) if default_graph_dir else None
        self._dir_cache: Dict[str, Optional[Path]] = {}

    def _resolve_dir(self, ptm_type: str) -> Optional[Path]:
        key = sanitize_name(ptm_type)
        if key in self._dir_cache:
            return self._dir_cache[key]
        candidate = self.graph_dirs_by_ptm.get(key) or self.default_graph_dir
        if not candidate:
            self._dir_cache[key] = None
            return None
        path = Path(candidate)
        if not path.exists():
            self._dir_cache[key] = None
            return None
        self._dir_cache[key] = path
        return path

    @staticmethod
    def _candidate_names(uniprot_id: str, position: int, ptm_type: str) -> List[str]:
        uid = str(uniprot_id)
        pos = int(position)
        ptm = sanitize_name(ptm_type)
        return [
            f"{uid}_{pos}.pt",
            f"{uid}_{pos}_{ptm}.pt",
            f"{uid}-{pos}.pt",
            f"{uid}-{pos}-{ptm}.pt",
            f"{uid}_{pos}.npz",
            f"{uid}_{pos}_{ptm}.npz",
        ]

    def get_graph(self, uniprot_id: str, position: int, ptm_type: str) -> Optional[Dict[str, torch.Tensor]]:
        graph_dir = self._resolve_dir(ptm_type)
        candidate_names = self._candidate_names(uniprot_id, position, ptm_type)
        
        # 1. Try specific PTM directory
        if graph_dir is not None:
            for name in candidate_names:
                path = graph_dir / name
                if path.exists():
                    res = self._load_graph_file(path, uniprot_id, position, ptm_type)
                    if res is not None:
                        return res
                        
        # 2. Fallback to default/shared directory if not found in specific dir
        if self.default_graph_dir:
            shared_dir = Path(self.default_graph_dir)
            if shared_dir.exists() and shared_dir != graph_dir:
                for name in candidate_names:
                    path = shared_dir / name
                    if path.exists():
                        res = self._load_graph_file(path, uniprot_id, position, ptm_type)
                        if res is not None:
                            return res
        return None

    def _load_graph_file(self, path: Path, uniprot_id: str, position: int, ptm_type: str) -> Optional[Dict[str, torch.Tensor]]:
        try:
            if path.suffix.lower() == ".npz":
                data = np.load(path, allow_pickle=False)
                x = torch.from_numpy(np.asarray(data["x"], dtype=np.float32))
                edge_index = torch.from_numpy(np.asarray(data["edge_index"], dtype=np.int64))
                edge_attr = torch.from_numpy(np.asarray(data["edge_attr"], dtype=np.float32))
            else:
                data = torch.load(path, map_location="cpu", weights_only=True)
                x = torch.as_tensor(data.get("x"), dtype=torch.float32)
                edge_index = torch.as_tensor(data.get("edge_index"), dtype=torch.long)
                edge_attr = torch.as_tensor(data.get("edge_attr"), dtype=torch.float32)
        except Exception:
            return None
            
        if x.ndim != 2 or edge_index.ndim != 2:
            return None
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.ndim != 2 or int(edge_attr.shape[0]) != int(edge_index.shape[1]):
            return None
            
        x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        return {
            "x": x.contiguous(),
            "edge_index": edge_index.contiguous(),
            "edge_attr": edge_attr.contiguous(),
        }


class GraphFeatureLMDBCache:
    def __init__(
        self,
        graph_lmdbs_by_ptm: Dict[str, str],
        default_graph_lmdb: Optional[str] = None,
        fallback_graph_lmdb: Optional[str] = None,
        readonly: bool = True,
    ) -> None:
        self.graph_lmdbs_by_ptm = {sanitize_name(k): str(v) for k, v in (graph_lmdbs_by_ptm or {}).items() if v}
        self.default_graph_lmdb = str(default_graph_lmdb) if default_graph_lmdb else None
        self.fallback_graph_lmdb = str(fallback_graph_lmdb) if fallback_graph_lmdb else None
        self.readonly = bool(readonly)
        self._env_cache: Dict[str, Any] = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_env_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._env_cache = {}

    def close(self) -> None:
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_cache.clear()

    def _resolve_path(self, ptm_type: str) -> Optional[str]:
        key = sanitize_name(ptm_type)
        return self.graph_lmdbs_by_ptm.get(key) or self.default_graph_lmdb

    def _paths(self, ptm_type: str) -> List[str]:
        primary = self._resolve_path(ptm_type)
        return [p for p in (primary, self.fallback_graph_lmdb) if p]

    def _get_env(self, path_value: str) -> Optional[Any]:
        if not HAS_LMDB or lmdb is None:
            return None
        key = str(path_value)
        if key in self._env_cache:
            return self._env_cache[key]
        path = Path(key)
        if not path.exists():
            self._env_cache[key] = None
            return None
        env = lmdb.open(
            str(path),
            readonly=self.readonly,
            lock=not self.readonly,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )
        self._env_cache[key] = env
        return env

    @staticmethod
    def _candidate_keys(uniprot_id: str, position: int, ptm_type: str) -> List[str]:
        uid = str(uniprot_id)
        pos = int(position)
        ptm = sanitize_name(ptm_type)
        return [
            f"{uid}_{pos}_{ptm}",
            f"{uid}_{pos}",
            f"{uid}-{pos}-{ptm}",
            f"{uid}-{pos}",
        ]

    def has_graph(self, uniprot_id: str, position: int, ptm_type: str) -> bool:
        keys = self._candidate_keys(uniprot_id, position, ptm_type)
        for path_value in self._paths(ptm_type):
            env = self._get_env(path_value)
            if env is None:
                continue
            try:
                with env.begin(write=False) as txn:
                    if any(txn.get(key.encode("utf-8")) is not None for key in keys):
                        return True
            except Exception:
                continue
        return False

    def get_graph(self, uniprot_id: str, position: int, ptm_type: str) -> Optional[Dict[str, torch.Tensor]]:
        keys = self._candidate_keys(uniprot_id, position, ptm_type)
        raw = None
        for path_value in self._paths(ptm_type):
            env = self._get_env(path_value)
            if env is None:
                continue
            try:
                with env.begin(write=False) as txn:
                    for key in keys:
                        raw = txn.get(key.encode("utf-8"))
                        if raw is not None:
                            break
            except Exception:
                raw = None
            if raw is not None:
                break
        if raw is None:
            return None
        try:
            data = pickle.loads(raw)
        except Exception:
            return None
        try:
            x = torch.as_tensor(data.get("x"), dtype=torch.float32)
            edge_index = torch.as_tensor(data.get("edge_index"), dtype=torch.long)
            edge_attr = torch.as_tensor(data.get("edge_attr"), dtype=torch.float32)
        except Exception:
            return None
        if x.ndim != 2 or edge_index.ndim != 2:
            return None
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.ndim != 2 or int(edge_attr.shape[0]) != int(edge_index.shape[1]):
            return None
        x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        return {"x": x.contiguous(), "edge_index": edge_index.contiguous(), "edge_attr": edge_attr.contiguous()}


class MultiLabelPTMJsonDataset(Dataset):
    """Dataset for multi-label lysine PTM prediction from JSON records."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        ptm_types: List[str],
        window_size: int,
        local_window_size: int,
        seq_feature_dim: int,
        struct_feature_dim: int,
        feature_sources: Optional[Dict[str, str]] = None,
        structure_cache: Optional[PDBStructureCache] = None,
        lazy_cache_size: int = 128,
        enable_structure_mask_token: bool = True,
        structure_mask_token_value: float = 1.0,
        use_sequence_windows: bool = False,
        sequence_window_provider: Optional["SequenceWindowProvider"] = None,
        use_structure_graph: bool = False,
        graph_cache: Optional[GraphFeatureCache] = None,
        disable_pdb_fallback: bool = False,
        use_structure_features: bool = True,
        use_micro_environment: bool = True,
        graph_node_dim: int = 0,
        graph_edge_dim: int = 0,
        alignment_target_labels: Optional[List[str]] = None,
        always_realign_windows: bool = True,
        strict_center_k: bool = False,
        min_valid_flank_aa: int = 0,
        realign_on_center_mismatch: bool = False,
        realign_on_short_context: bool = False,
        motif_bias_tasks: Optional[List[str]] = None,
        motif_bias_methyl_radius: int = 7,
        motif_bias_crotonyl_radius: int = 4,
        motif_bias_sumo_hydrophobic: Optional[List[str]] = None,
    ) -> None:
        self.records = records
        self.ptm_types = [sanitize_name(p) for p in ptm_types]
        self.ptm_to_idx = {p: i for i, p in enumerate(self.ptm_types)}
        self.window_size = int(window_size)
        self.local_window_size = int(local_window_size)
        self.seq_feature_dim = int(seq_feature_dim)
        self.struct_feature_dim = int(struct_feature_dim)
        self.center_idx = self.window_size // 2
        self.enable_structure_mask_token = bool(enable_structure_mask_token)
        self.structure_mask_token_value = float(structure_mask_token_value)
        self.feature_sources = feature_sources or {}
        self.use_sequence_windows = bool(use_sequence_windows)
        self.sequence_window_provider = sequence_window_provider
        self.use_structure_graph = bool(use_structure_graph)
        self.graph_cache = graph_cache
        self.graph_node_dim = max(0, int(graph_node_dim))
        self.graph_edge_dim = max(0, int(graph_edge_dim))
        self.alignment_target_labels = {sanitize_name(x) for x in (alignment_target_labels or [])}
        self.always_realign_windows = bool(always_realign_windows)
        self.strict_center_k = bool(strict_center_k)
        self.min_valid_flank_aa = max(0, int(min_valid_flank_aa))
        self.realign_on_center_mismatch = bool(realign_on_center_mismatch)
        self.realign_on_short_context = bool(realign_on_short_context)
        self.motif_bias_tasks = {sanitize_name(x) for x in (motif_bias_tasks or DEFAULT_MOTIF_TASKS)}
        self.motif_bias_methyl_radius = max(1, int(motif_bias_methyl_radius))
        self.motif_bias_crotonyl_radius = max(1, int(motif_bias_crotonyl_radius))
        self.motif_bias_sumo_hydrophobic = [str(x).upper() for x in (motif_bias_sumo_hydrophobic or DEFAULT_SUMO_HYDROPHOBIC)]
        self.h5_store = (
            H5FeatureStoreManager(
                self.feature_sources,
                cache_size=lazy_cache_size,
                structure_cache=structure_cache,
                ptm_types=self.ptm_types,
                disable_pdb_fallback=bool(disable_pdb_fallback),
                use_structure_features=bool(use_structure_features),
                use_micro_environment=bool(use_micro_environment),
            )
            if self.feature_sources
            else None
        )

        self.residue_to_id = {
            "A": 1,
            "R": 2,
            "N": 3,
            "D": 4,
            "C": 5,
            "Q": 6,
            "E": 7,
            "G": 8,
            "H": 9,
            "I": 10,
            "L": 11,
            "K": 12,
            "M": 13,
            "F": 14,
            "P": 15,
            "S": 16,
            "T": 17,
            "W": 18,
            "Y": 19,
            "V": 20,
            "X": 0,
        }
        self._position_ids_t = torch.from_numpy(np.arange(self.window_size, dtype=np.int64))
        self._rel_pos_t = torch.from_numpy(np.arange(self.window_size, dtype=np.int64) - self.center_idx)
        site_indicator = np.zeros((self.window_size,), dtype=np.float32)
        site_indicator[self.center_idx] = 1.0
        self._site_indicator_t = torch.from_numpy(site_indicator)

    def _compute_motif_bias(self, window_seq: Optional[str]) -> np.ndarray:
        return compute_motif_bias_scores(
            window_seq=window_seq,
            center_idx=self.center_idx,
            ptm_types=self.ptm_types,
            active_tasks=list(self.motif_bias_tasks),
            sumo_hydrophobic=self.motif_bias_sumo_hydrophobic,
            methyl_radius=self.motif_bias_methyl_radius,
            crotonyl_radius=self.motif_bias_crotonyl_radius,
        )

    def _record_requires_alignment(self, rec: Dict[str, Any]) -> bool:
        if not self.alignment_target_labels:
            return True
        labels = rec.get("labels")
        if isinstance(labels, dict):
            for key, value in labels.items():
                if sanitize_name(key) in self.alignment_target_labels and float(value) > 0:
                    return True
        active = rec.get("ptm_types") or rec.get("active_ptms") or rec.get("mods")
        if isinstance(active, list):
            return any(sanitize_name(x) in self.alignment_target_labels for x in active)
        ptm = rec.get("ptm_type")
        return sanitize_name(ptm) in self.alignment_target_labels if ptm is not None else False

    def _normalize_window_sequence(self, window_seq: Optional[str]) -> Optional[str]:
        if not isinstance(window_seq, str):
            return None
        seq = window_seq.strip().upper()
        return seq if seq else None

    def _valid_flank_lengths(self, window_seq: Optional[str]) -> Tuple[int, int]:
        if not isinstance(window_seq, str) or not window_seq:
            return 0, 0
        seq = window_seq.upper()
        center = min(len(seq) - 1, self.center_idx)
        left = 0
        for i in range(center - 1, -1, -1):
            if seq[i] == "X":
                break
            left += 1
        right = 0
        for i in range(center + 1, len(seq)):
            if seq[i] == "X":
                break
            right += 1
        return left, right

    def _should_realign_window(self, rec: Dict[str, Any], window_seq: Optional[str]) -> bool:
        if not self.use_sequence_windows or self.sequence_window_provider is None:
            return False
        if self.always_realign_windows:
            return True
        if not self._record_requires_alignment(rec):
            return False
        if window_seq is None or (isinstance(window_seq, str) and len(window_seq.strip()) == 0):
            return True
        if not isinstance(window_seq, str):
            return True
        seq = window_seq.strip().upper()
        if len(seq) != self.window_size:
            return True
        if self.strict_center_k and seq[self.center_idx] != "K":
            return True
        left_flank, right_flank = self._valid_flank_lengths(seq)
        if self.realign_on_short_context and min(left_flank, right_flank) < self.min_valid_flank_aa:
            return True
        if self.realign_on_center_mismatch and seq[self.center_idx] != "K":
            return True
        return False

    def _resolve_window_sequence(self, rec: Dict[str, Any], uniprot_id: str, position_1based: int) -> Optional[str]:
        raw_window = self._normalize_window_sequence(rec.get("sequence_window_str"))
        if self._should_realign_window(rec, raw_window):
            aligned_window = self.sequence_window_provider.get_window(uniprot_id, position_1based)
            if aligned_window is not None:
                return aligned_window
        return raw_window

    def get_label_matrix(self) -> np.ndarray:
        labels = [self._extract_labels(rec) for rec in self.records]
        if not labels:
            return np.zeros((0, len(self.ptm_types)), dtype=np.float32)
        return np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _as_np(x: Any, dtype=np.float32) -> Optional[np.ndarray]:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            arr = x.astype(dtype, copy=False)
        else:
            arr = np.asarray(x, dtype=dtype)
        if arr.size == 0:
            return arr
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=-50.0)
        return np.clip(arr, -50.0, 50.0).astype(dtype, copy=False)

    def _ensure_2d(self, arr: Optional[np.ndarray], length: int, feat_dim: int) -> np.ndarray:
        if arr is None:
            return np.zeros((length, feat_dim), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, feat_dim) if arr.size % feat_dim == 0 else np.zeros((length, feat_dim), dtype=np.float32)
        if arr.shape[1] != feat_dim:
            if arr.shape[1] > feat_dim:
                arr = arr[:, :feat_dim]
            else:
                pad = np.zeros((arr.shape[0], feat_dim - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)

        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], feat_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > length:
            arr = arr[:length]
        return arr.astype(np.float32, copy=False)

    def _extract_labels(self, rec: Dict[str, Any]) -> np.ndarray:
        labels = np.zeros((len(self.ptm_types),), dtype=np.float32)

        if isinstance(rec.get("labels"), dict):
            for k, v in rec["labels"].items():
                key = sanitize_name(k)
                if key in self.ptm_to_idx:
                    labels[self.ptm_to_idx[key]] = float(v)
            return labels

        if isinstance(rec.get("labels"), list) and len(rec["labels"]) == len(self.ptm_types):
            return np.asarray(rec["labels"], dtype=np.float32)

        active = rec.get("ptm_types") or rec.get("active_ptms") or rec.get("mods")
        if isinstance(active, list):
            for name in active:
                key = sanitize_name(name)
                if key in self.ptm_to_idx:
                    labels[self.ptm_to_idx[key]] = 1.0
            return labels

        ptm = rec.get("ptm_type")
        y = rec.get("label")
        if ptm is not None and y is not None:
            key = sanitize_name(ptm)
            if key in self.ptm_to_idx:
                labels[self.ptm_to_idx[key]] = float(y)

        return labels

    def _extract_loss_weights(self, rec: Dict[str, Any]) -> np.ndarray:
        weights = rec.get("loss_weight")
        if weights is None:
            weights = rec.get("loss_weights")
        if isinstance(weights, dict):
            out = np.ones((len(self.ptm_types),), dtype=np.float32)
            for key, value in weights.items():
                ptm_key = sanitize_name(key)
                if ptm_key in self.ptm_to_idx:
                    try:
                        out[self.ptm_to_idx[ptm_key]] = float(value)
                    except Exception:
                        out[self.ptm_to_idx[ptm_key]] = 1.0
            return np.clip(out, 0.0, 10.0).astype(np.float32, copy=False)
        if isinstance(weights, list) and len(weights) == len(self.ptm_types):
            try:
                return np.clip(np.asarray(weights, dtype=np.float32), 0.0, 10.0)
            except Exception:
                pass
        return np.ones((len(self.ptm_types),), dtype=np.float32)

    def _extract_label_states(self, rec: Dict[str, Any], labels: np.ndarray) -> np.ndarray:
        states = rec.get("label_state")
        if states is None:
            states = rec.get("label_states")
        out = np.zeros((len(self.ptm_types),), dtype=np.int64)
        if isinstance(states, dict):
            for key, value in states.items():
                ptm_key = sanitize_name(key)
                if ptm_key in self.ptm_to_idx:
                    idx = self.ptm_to_idx[ptm_key]
                    out[idx] = label_state_id(value, label_value=float(labels[idx]))
            for idx, y in enumerate(labels):
                if out[idx] == 0 and float(y) >= 0.0:
                    out[idx] = label_state_id(None, label_value=float(y))
            return out
        if isinstance(states, list) and len(states) == len(self.ptm_types):
            for idx, value in enumerate(states):
                out[idx] = label_state_id(value, label_value=float(labels[idx]))
            return out
        for idx, y in enumerate(labels):
            out[idx] = label_state_id(None, label_value=float(y))
        return out

    def _extract_label_reliability(self, rec: Dict[str, Any]) -> np.ndarray:
        return self._extract_vector_field(
            rec,
            ["label_reliability", "label_reliability_score", "reliability_score", "reliability_scores"],
            default=1.0,
            clip_min=0.0,
            clip_max=1.0,
        )

    def _extract_vector_field(self, rec: Dict[str, Any], field_names: List[str], default: float, clip_min: float, clip_max: float) -> np.ndarray:
        value = None
        for field in field_names:
            if field in rec:
                value = rec.get(field)
                break
        if isinstance(value, dict):
            out = np.full((len(self.ptm_types),), default, dtype=np.float32)
            for key, raw in value.items():
                ptm_key = sanitize_name(key)
                if ptm_key in self.ptm_to_idx:
                    try:
                        out[self.ptm_to_idx[ptm_key]] = float(raw)
                    except Exception:
                        out[self.ptm_to_idx[ptm_key]] = default
            return np.clip(out, clip_min, clip_max).astype(np.float32, copy=False)
        if isinstance(value, list) and len(value) == len(self.ptm_types):
            try:
                return np.clip(np.asarray(value, dtype=np.float32), clip_min, clip_max)
            except Exception:
                pass
        return np.full((len(self.ptm_types),), default, dtype=np.float32)

    def _extract_residue_types(self, rec: Dict[str, Any]) -> np.ndarray:
        seq_block = rec.get("sequence", {}) if isinstance(rec.get("sequence"), dict) else {}
        res_ids = seq_block.get("residue_types") or rec.get("residue_types")
        if res_ids is not None:
            arr = np.asarray(res_ids, dtype=np.int64)
            if arr.shape[0] < self.window_size:
                pad = np.zeros((self.window_size - arr.shape[0],), dtype=np.int64)
                arr = np.concatenate([arr, pad])
            elif arr.shape[0] > self.window_size:
                arr = arr[: self.window_size]
            return arr

        residue_arr = np.zeros((self.window_size,), dtype=np.int64)
        seq_str = rec.get("sequence_window_str") or seq_block.get("window_seq")
        if isinstance(seq_str, str) and len(seq_str) > 0:
            seq_str = seq_str[: self.window_size].ljust(self.window_size, "X")
            for i, aa in enumerate(seq_str):
                residue_arr[i] = self.residue_to_id.get(aa.upper(), 0)
            return residue_arr

        center_res = str(rec.get("residue", "X")).upper()
        residue_arr[self.center_idx] = self.residue_to_id.get(center_res, 0)
        return residue_arr

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        uid = str(rec.get("uniprot_id", ""))
        pos = int(rec.get("position", -1))
        seq_block = rec.get("sequence", {}) if isinstance(rec.get("sequence"), dict) else {}

        seq_window = self._as_np(self._first_present(seq_block.get("window_features"), rec.get("sequence_features")))
        seq_local = self._as_np(self._first_present(seq_block.get("local_features"), rec.get("local_features")))
        seq_global = self._as_np(self._first_present(seq_block.get("global_features"), rec.get("global_features")))
        struct = self._as_np(self._first_present(rec.get("structure"), rec.get("structure_features")))
        struct_global = self._as_np(self._first_present(rec.get("structure_global_features"), rec.get("structure_global")))
        structure_available = struct is not None
        structure_global_available = struct_global is not None
        micro = self._as_np(self._first_present(rec.get("micro_env_features"), rec.get("micro_environment")))

        # Check if this is an augmented fake sequence
        is_augmented = uid.startswith("AUG_REV_") or uid.startswith("AUG_HOMO_")

        if (
            self.h5_store is not None
            and seq_window is None
            and seq_local is None
            and seq_global is None
            and not is_augmented
        ):
            source = rec.get("feature_source") or rec.get("source")
            uid = str(rec.get("uniprot_id", ""))
            pos = int(rec.get("position", -1))
            loaded = self.h5_store.get_site_features(
                source=source,
                uniprot_id=uid,
                position=pos,
                window_size=self.window_size,
                local_window_size=self.local_window_size,
                ptm_type=_resolve_ptm_from_record(rec, self.ptm_types),
            )
            seq_window = self._as_np(loaded.get("sequence_window"))
            seq_local = self._as_np(loaded.get("sequence_local"))
            seq_global = self._as_np(loaded.get("sequence_global"))
            struct = self._as_np(loaded.get("structure_window"))
            struct_global = self._as_np(loaded.get("structure_global"))
            structure_available = struct is not None
            structure_global_available = struct_global is not None
            if micro is None:
                micro = self._as_np(loaded.get("micro_env"))

        seq_window = self._ensure_2d(seq_window, self.window_size, self.seq_feature_dim)
        seq_local = self._ensure_2d(seq_local, self.local_window_size, self.seq_feature_dim)

        global_context_available = seq_global is not None
        if seq_global is None:
            seq_global = seq_window.mean(axis=0)
        else:
            seq_global = np.asarray(seq_global, dtype=np.float32).reshape(-1)
            if seq_global.shape[0] > self.seq_feature_dim:
                seq_global = seq_global[: self.seq_feature_dim]
            elif seq_global.shape[0] < self.seq_feature_dim:
                seq_global = np.pad(seq_global, (0, self.seq_feature_dim - seq_global.shape[0]))

        if struct_global is None:
            struct_global = np.zeros((self.struct_feature_dim,), dtype=np.float32)
        else:
            struct_global = np.asarray(struct_global, dtype=np.float32).reshape(-1)
            if struct_global.shape[0] > self.struct_feature_dim:
                struct_global = struct_global[: self.struct_feature_dim]
            elif struct_global.shape[0] < self.struct_feature_dim:
                struct_global = np.pad(struct_global, (0, self.struct_feature_dim - struct_global.shape[0]))

        struct = self._ensure_2d(struct, self.window_size, self.struct_feature_dim)
        if not structure_available and self.enable_structure_mask_token and self.struct_feature_dim > 0:
            struct[:, -1] = self.structure_mask_token_value
        elif not structure_available and is_augmented and self.struct_feature_dim > 0:
            # Force apply structure mask token for augmented fake sequences even if not globally enabled
            struct[:, -1] = self.structure_mask_token_value

        micro_available = False
        if micro is None:
            micro = np.zeros((6,), dtype=np.float32)
        else:
            micro = np.asarray(micro, dtype=np.float32).reshape(-1)
            if micro.shape[0] > 6:
                micro = micro[:6]
            elif micro.shape[0] < 6:
                micro = np.pad(micro, (0, 6 - micro.shape[0]))
            micro = np.nan_to_num(micro, nan=0.0, posinf=1e4, neginf=0.0).astype(np.float32, copy=False)
            micro_available = bool(np.any(micro != 0.0))

        window_seq = self._resolve_window_sequence(rec, uid, pos)

        motif_bias = self._compute_motif_bias(window_seq)

        labels = self._extract_labels(rec)
        loss_weights = self._extract_loss_weights(rec)
        label_states = self._extract_label_states(rec, labels)
        label_reliability = self._extract_label_reliability(rec)
        teacher_probs = self._extract_vector_field(rec, ["teacher_probs", "teacher_probabilities"], default=-1.0, clip_min=-1.0, clip_max=1.0)
        teacher_mask = self._extract_vector_field(rec, ["teacher_mask", "teacher_masks"], default=0.0, clip_min=0.0, clip_max=1.0)
        teacher_weight = self._extract_vector_field(rec, ["teacher_weight", "teacher_weights"], default=1.0, clip_min=0.0, clip_max=10.0)
        residue_types = self._extract_residue_types(rec)

        graph = None
        graph_available = False
        if self.use_structure_graph and self.graph_cache is not None and not is_augmented:
            ptm_type = _resolve_ptm_from_record(rec, self.ptm_types)
            graph = self.graph_cache.get_graph(uid, pos, ptm_type)
            graph_available = graph is not None
        
        if graph is None and self.graph_node_dim > 0:
            edge_dim = self.graph_edge_dim if self.graph_edge_dim > 0 else 1
            graph = {
                "x": torch.zeros((1, self.graph_node_dim), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, edge_dim), dtype=torch.float32),
            }

        return {
            "sample_index": idx,
            "uniprot_id": uid,
            "position": pos,
            "residue": str(rec.get("residue", "K")),
            "sequence_window_str": window_seq,
            "source_tasks": str(rec.get("source_tasks", ";".join(rec.get("active_ptms", [])) if isinstance(rec.get("active_ptms"), list) else "")),
            "source_splits": str(rec.get("source_splits", "")),
            "known_task_count": int(rec.get("known_task_count", -1)) if str(rec.get("known_task_count", "")).strip() else -1,
            "positive_task_count": int(rec.get("positive_task_count", -1)) if str(rec.get("positive_task_count", "")).strip() else -1,
            "negative_task_count": int(rec.get("negative_task_count", -1)) if str(rec.get("negative_task_count", "")).strip() else -1,
            "unknown_task_count": int(rec.get("unknown_task_count", -1)) if str(rec.get("unknown_task_count", "")).strip() else -1,
            "sequence": {
                "window_features": torch.from_numpy(np.ascontiguousarray(seq_window)),
                "local_features": torch.from_numpy(np.ascontiguousarray(seq_local)),
                "global_features": torch.from_numpy(np.ascontiguousarray(seq_global)),
                "global_context_available": torch.tensor(1.0 if global_context_available else 0.0, dtype=torch.float32),
                "residue_types": torch.from_numpy(np.ascontiguousarray(residue_types)),
                "position_ids": self._position_ids_t,
                "relative_position_offsets": self._rel_pos_t,
                "site_indicator": self._site_indicator_t,
            },
            "structure": torch.from_numpy(np.ascontiguousarray(struct)),
            "structure_global_features": torch.from_numpy(np.ascontiguousarray(struct_global)),
            "structure_global_available": torch.tensor(1.0 if structure_global_available else 0.0, dtype=torch.float32),
            "structure_available": torch.tensor(1.0 if structure_available else 0.0, dtype=torch.float32),
            "structure_graph": graph,
            "structure_graph_available": torch.tensor(1.0 if graph_available else 0.0, dtype=torch.float32),
            "micro_env_features": torch.from_numpy(np.ascontiguousarray(micro)),
            "micro_env_available": torch.tensor(1.0 if micro_available else 0.0, dtype=torch.float32),
            "motif_bias": torch.from_numpy(np.ascontiguousarray(motif_bias)),
            "label": torch.from_numpy(np.ascontiguousarray(labels)),
            "loss_weight": torch.from_numpy(np.ascontiguousarray(loss_weights)),
            "label_state": torch.from_numpy(np.ascontiguousarray(label_states)),
            "label_reliability": torch.from_numpy(np.ascontiguousarray(label_reliability)),
            "teacher_probs": torch.from_numpy(np.ascontiguousarray(teacher_probs)),
            "teacher_mask": torch.from_numpy(np.ascontiguousarray(teacher_mask)),
            "teacher_weight": torch.from_numpy(np.ascontiguousarray(teacher_weight)),
        }


class PreprocessedTensorLMDBDataset(Dataset):
    """Reads already-collatable samples in physical LMDB order."""
    def __init__(
        self,
        path: str,
        length: int,
        label_matrix: Optional[np.ndarray] = None,
        residue_overlay_path: Optional[str] = None,
        motif_overlay_path: Optional[str] = None,
    ) -> None:
        if not HAS_LMDB or lmdb is None:
            raise RuntimeError("preprocessed tensor LMDB requires python-lmdb")
        self.path = str(path)
        self.length = int(length)
        self._labels = label_matrix
        self._env = None
        self._txn = None
        # Small sidecars repair metadata that was generated from legacy 31/61-mer
        # CSV windows while the F4 tensors themselves are already 73 positions.
        self.residue_overlay_path = str(residue_overlay_path) if residue_overlay_path else None
        self.motif_overlay_path = str(motif_overlay_path) if motif_overlay_path else None
        self._residue_overlay = None
        self._motif_overlay = None

    def _open(self):
        if self._env is None:
            self._env = lmdb.open(self.path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=2048)
        return self._env

    def _open_txn(self):
        # Keep one read transaction per worker/process.  Opening a transaction
        # for every sample defeats LMDB's sequential-read advantage.
        if self._txn is None:
            self._txn = self._open().begin(buffers=True)
        return self._txn

    def __len__(self) -> int:
        return self.length

    def get_label_matrix(self) -> np.ndarray:
        if self._labels is None:
            self._labels = np.asarray([self[i]["label"].numpy() for i in range(self.length)], dtype=np.float32)
        return self._labels

    def _open_overlays(self) -> None:
        if self.residue_overlay_path and self._residue_overlay is None:
            self._residue_overlay = np.load(self.residue_overlay_path, mmap_mode="r")
            if self._residue_overlay.shape[0] != self.length:
                raise ValueError(
                    f"Residue overlay length mismatch: {self._residue_overlay.shape[0]} != {self.length}"
                )
        if self.motif_overlay_path and self._motif_overlay is None:
            self._motif_overlay = np.load(self.motif_overlay_path, mmap_mode="r")
            if self._motif_overlay.shape[0] != self.length:
                raise ValueError(
                    f"Motif overlay length mismatch: {self._motif_overlay.shape[0]} != {self.length}"
                )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self._open_txn().get(f"{int(idx):09d}".encode())
        if raw is None:
            raise IndexError(idx)
        # lmdb.Buffer implements the buffer protocol; avoid copying every
        # serialized F4 sample into a second bytes object before unpickling.
        record = pickle.loads(raw)
        self._open_overlays()
        if self._residue_overlay is not None or self._motif_overlay is not None:
            # Shallow copies keep the large F4 tensors zero-copy while replacing
            # only the misaligned metadata arrays.
            record = dict(record)
            seq_block = dict(record.get("sequence") or {})
            if self._residue_overlay is not None:
                seq_block["residue_types"] = torch.from_numpy(
                    np.asarray(self._residue_overlay[int(idx)], dtype=np.int64)
                )
            record["sequence"] = seq_block
            if self._motif_overlay is not None:
                record["motif_bias"] = torch.from_numpy(
                    np.asarray(self._motif_overlay[int(idx)], dtype=np.float32)
                )
        return record

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def __del__(self):
        if self._txn is not None:
            try:
                self._txn.abort()
            except Exception:
                pass


class SiteFeatureLMDBDataset(Dataset):
    """Random-access dataset for ``site_feature_lmdb_v1`` records."""

    def __init__(self, path: str, meta_path: Optional[str] = None,
                 labels_path: Optional[str] = None, keys_path: Optional[str] = None,
                 use_structure: bool = True, use_graph: bool = True,
                 structure_mask_token_value: float = 1.0,
                 ram_cache: bool = False, ram_cache_preload: bool = False,
                 ram_cache_max_bytes: int = 0,
                 ram_cache_storage: str = "python",
                 labels_override_path: Optional[str] = None):
        if not HAS_LMDB or lmdb is None:
            raise RuntimeError("site feature LMDB requires python-lmdb")
        self.path = str(path)
        meta = json.loads(Path(meta_path or f"{path}.meta.json").read_text())
        self.length = int(meta["length"])
        labels_path = labels_path or meta.get("labels_path")
        self._labels_override = bool(labels_override_path)
        if labels_override_path:
            labels_path = str(labels_override_path)
        self._labels = np.load(labels_path, mmap_mode="r") if labels_path and Path(labels_path).exists() else None
        if self._labels_override and (self._labels is None or self._labels.shape[0] != self.length):
            raise ValueError(
                f"labels override length mismatch: {labels_override_path} "
                f"({None if self._labels is None else self._labels.shape[0]} != {self.length})"
            )
        self._keys = Path(keys_path or meta.get("keys_path", f"{path}.keys.txt")).read_text().splitlines()
        if len(self._keys) != self.length:
            raise ValueError(f"site LMDB key sidecar length mismatch: {len(self._keys)} != {self.length}")
        self.use_structure, self.use_graph = bool(use_structure), bool(use_graph)
        self.structure_mask_token_value = float(structure_mask_token_value)
        self._env = self._txn = None
        self.ram_cache = bool(ram_cache)
        self.ram_cache_preload = bool(ram_cache_preload)
        self.ram_cache_max_bytes = int(ram_cache_max_bytes or 0)
        self.ram_cache_storage = str(ram_cache_storage or "python").strip().lower()
        if self.ram_cache_storage not in {"python", "page_cache"}:
            raise ValueError(f"Unsupported site LMDB RAM cache storage: {self.ram_cache_storage}")
        self._ram_cache_ready = False
        self._ram_records: Optional[List[bytes]] = None
        self._ram_cache_bytes = 0
        if self.ram_cache and self.ram_cache_preload:
            self._preload_ram_cache()

    def _open(self):
        if self._env is None:
            # Preloading is a sequential pass; readahead helps that pass and is
            # irrelevant after the compressed records are resident in RAM.
            self._env = lmdb.open(self.path, readonly=True, lock=False,
                                  readahead=not bool(self._ram_records),
                                  meminit=False, max_readers=2048)
            self._txn = self._env.begin(buffers=True)
        return self._txn

    def _preload_ram_cache(self) -> None:
        """Read compressed records once so epochs do not touch the LMDB file.

        Keeping the compressed representation avoids expanding hundreds of
        thousands of Python dictionaries in RAM.  Decompression and tensor
        conversion remain batch-local in ``__getitem__``.
        """
        if self._ram_records is not None:
            return
        started = time.time()
        if self.ram_cache_storage == "page_cache":
            # Populate the kernel page cache with one sequential pass. Reading
            # the LMDB data file in large blocks avoids 666k Python cursor
            # iterations while still touching every overflow page. Do not
            # retain Python copies: mmap serves later random accesses from RAM.
            data_path = Path(self.path) / "data.mdb"
            if not data_path.exists():
                data_path = Path(self.path)
            total_bytes = 0
            chunk_size = 64 * 1024 * 1024
            with data_path.open("rb", buffering=0) as stream:
                while True:
                    block = stream.read(chunk_size)
                    if not block:
                        break
                    total_bytes += len(block)
                    if self.ram_cache_max_bytes and total_bytes > self.ram_cache_max_bytes:
                        raise MemoryError(
                            f"site LMDB page-cache warmup exceeded {self.ram_cache_max_bytes} bytes"
                        )
            if self._txn is not None:
                self._txn.abort(); self._txn = None
            if self._env is not None:
                self._env.close(); self._env = None
            self._ram_cache_bytes = total_bytes
            self._ram_cache_ready = True
            logging.info(
                "Warmed site LMDB OS page cache: path=%s file_bytes=%d records=%d elapsed=%.1fs",
                self.path, total_bytes, self.length, time.time() - started,
            )
            return
        records_by_key: Dict[bytes, bytes] = {}
        total_bytes = 0
        txn = self._open()
        with txn.cursor() as cursor:
            for key, value in cursor:
                raw = bytes(value)
                total_bytes += len(raw)
                if self.ram_cache_max_bytes and total_bytes > self.ram_cache_max_bytes:
                    raise MemoryError(
                        f"site LMDB RAM cache needs more than {self.ram_cache_max_bytes} "
                        f"bytes (reached {total_bytes} at key {key!r})"
                    )
                records_by_key[bytes(key)] = raw
            if len(records_by_key) != self.length:
                raise ValueError(
                    f"site LMDB record count mismatch: {len(records_by_key)} != {self.length}"
                )
        missing = [key for key in self._keys if key.encode() not in records_by_key]
        if missing:
            raise KeyError(f"site LMDB keys missing from data: {missing[:3]}")
        records = [records_by_key[key.encode()] for key in self._keys]
        # Release the small key index; the compressed values remain resident.
        del records_by_key
        self._ram_records = records
        self._ram_cache_bytes = total_bytes
        self._ram_cache_ready = True
        # The cache is now independent of LMDB.  Close the mapping so the
        # kernel does not keep a second copy of the 231 GB data file resident
        # while validation and test caches are loaded.
        if self._txn is not None:
            self._txn.abort()
            self._txn = None
        if self._env is not None:
            self._env.close()
            self._env = None
        logging.info(
            "Loaded site LMDB RAM cache: path=%s records=%d compressed_bytes=%d "
            "elapsed=%.1fs",
            self.path, len(records), total_bytes, time.time() - started,
        )

    def __len__(self):
        return self.length

    def get_label_matrix(self):
        return np.asarray(self._labels, dtype=np.float32) if self._labels is not None else np.zeros((self.length, 8), np.float32)

    @staticmethod
    def _graph_token_index(graph: Dict[str, Any], site_position: int) -> Optional[np.ndarray]:
        """Map graph nodes to 73-window token positions via chain:resid keys.

        The sequence window is centered on the site (token 36).  Nodes on the
        center chain inside the window get 0..72; everything else gets -1.
        """
        keys = graph.get("residue_keys")
        if keys is None or site_position < 0:
            return None
        meta = graph.get("meta") if isinstance(graph.get("meta"), dict) else {}
        center_chain = meta.get("center_chain")
        center_mask = graph.get("center_mask")
        if center_chain is None and center_mask is not None:
            mask = torch.as_tensor(center_mask).reshape(-1)
            if mask.numel() and int(mask.argmax().item()) < len(keys):
                raw = keys[int(mask.argmax().item())]
                text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                center_chain = text.split(":", 1)[0] if ":" in text else None
        token_index = np.full(len(keys), -1, dtype=np.int64)
        if center_chain is None:
            return token_index
        window, center = 73, 36
        for i, raw in enumerate(keys):
            text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            chain, sep, rest = text.partition(":")
            if not sep:
                continue
            match = re.match(r"(-?\d+)", rest)
            if match is None or chain != str(center_chain):
                continue
            token = center + (int(match.group(1)) - int(site_position))
            if 0 <= token < window:
                token_index[i] = token
        return token_index

    @staticmethod
    def _tensor(value, dtype=torch.float32):
        return torch.from_numpy(np.asarray(value)).to(dtype=dtype)

    def __getitem__(self, idx):
        index = int(idx)
        if self._ram_records is not None:
            raw = self._ram_records[index]
        else:
            raw = self._open().get(self._keys[index].encode())
        if raw is None:
            raise IndexError(idx)
        rec = pickle.loads(zlib.decompress(raw))
        seq = np.asarray(rec["sequence_window"], dtype=np.float32)
        struct = np.asarray(rec["structure_window"], dtype=np.float32)
        struct_available = bool(rec.get("structure_available", 0))
        if not self.use_structure:
            struct.fill(0.0); struct_available = False
        elif not struct_available and struct.shape[1] > 0:
            struct[:, -1] = self.structure_mask_token_value
        seq_global = np.asarray(rec["sequence_global"], dtype=np.float32)
        struct_global = np.asarray(rec["structure_global"], dtype=np.float32)
        residue = str(rec.get("metadata", {}).get("window_sequence", "X"))
        residue = residue[:73].ljust(73, "X")
        aa = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,"I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,"W":18,"Y":19,"V":20}
        residue_types = torch.tensor([aa.get(c, 0) for c in residue], dtype=torch.long)
        graph = rec.get("graph") if self.use_graph else None
        if graph is not None:
            graph = {k: self._tensor(v, torch.long if k == "edge_index" else torch.float32)
                     if k in ("x", "pos", "edge_index", "edge_attr", "esm_if", "esm_if_mask", "center_mask") else v
                     for k, v in graph.items()}
            if isinstance(graph, dict):
                graph["token_index"] = self._graph_token_index(graph, int(rec.get("position", -1)))
        if self._labels_override and self._labels is not None:
            labels = np.asarray(self._labels[idx], dtype=np.float32)
        else:
            labels = np.asarray(rec.get("labels", self._labels[idx] if self._labels is not None else np.full(8, -1)), dtype=np.float32)
        # State ids: unknown=0, positive=1, observed_negative=2, reliable_negative=3.
        # Known negatives MUST map to 2 (or 3), never 1: the state-aware ASL applies
        # the positive term to state==1, so the old 0->1 mapping pushed every known
        # negative up instead of down.
        label_state = np.where(labels < 0, 0, np.where(labels > 0, 1, 2)).astype(np.int64)
        return {"sample_index": int(rec.get("sample_index", idx)), "uniprot_id": rec["uniprot_id"], "position": int(rec["position"]),
                "residue": rec.get("residue", "K"), "sequence_window_str": rec.get("metadata", {}).get("window_sequence"),
                "sequence": {"window_features": self._tensor(seq), "local_features": self._tensor(seq[21:52]),
                             "global_features": self._tensor(seq_global), "global_context_available": torch.tensor(float(rec.get("sequence_global_available", 1))),
                             "residue_types": residue_types, "position_ids": torch.arange(73), "relative_position_offsets": torch.arange(73)-36,
                             "site_indicator": torch.nn.functional.one_hot(torch.tensor(36), num_classes=73).float()},
                "structure": self._tensor(struct), "structure_global_features": self._tensor(struct_global),
                "structure_available": torch.tensor(float(struct_available)), "structure_global_available": torch.tensor(float(rec.get("structure_global_available", 0))),
                "structure_graph": graph, "structure_graph_available": torch.tensor(float(graph is not None and graph.get("graph_available", 0))),
                "micro_env_features": torch.empty(0), "micro_env_available": torch.tensor(0.),
                "motif_bias": torch.zeros(8), "label": self._tensor(labels), "loss_weight": torch.ones(8),
                "label_state": torch.tensor(label_state), "label_reliability": torch.ones(8),
                "teacher_probs": torch.full((8,), -1.), "teacher_mask": torch.zeros(8), "teacher_weight": torch.ones(8)}

    def __getstate__(self):
        state = self.__dict__.copy(); state["_env"] = state["_txn"] = None; return state

    def __del__(self):
        if self._txn is not None:
            try: self._txn.abort()
            except Exception: pass
        if self._env is not None:
            try: self._env.close()
            except Exception: pass
        if self._env is not None:
            self._env.close()


class BlockShuffleBatchSampler(Sampler[List[int]]):
    """Shuffle physical LMDB blocks while preserving sequential reads inside each block."""

    def __init__(self, length: int, batch_size: int, block_size: int = 8192, seed: int = 42,
                 drop_last: bool = False, rank: int = 0, world_size: int = 1) -> None:
        self.length = int(length)
        self.batch_size = max(1, int(batch_size))
        self.block_size = max(self.batch_size, int(block_size))
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_batches(self):
        block_ids = list(range((self.length + self.block_size - 1) // self.block_size))
        random.Random(self.seed + self.epoch).shuffle(block_ids)
        batches = []
        for block_id in block_ids:
            start = block_id * self.block_size
            stop = min(start + self.block_size, self.length)
            for batch_start in range(start, stop, self.batch_size):
                batch = list(range(batch_start, min(batch_start + self.batch_size, stop)))
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._all_batches()
        if self.world_size > 1:
            # DDP requires every rank to run the same number of optimizer steps.
            per_rank = len(batches) // self.world_size
            batches = batches[self.rank * per_rank:(self.rank + 1) * per_rank]
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            total = self.length // self.batch_size
        else:
            total = (self.length + self.batch_size - 1) // self.batch_size
        if self.world_size > 1:
            return total // self.world_size
        return total


class ProteinDedupLMDBDataset(Dataset):
    """Compact sample LMDB plus protein-level feature LMDB with worker-local cache."""
    def __init__(self, sample_path: str, protein_path: str, length: int,
                 label_matrix: Optional[np.ndarray] = None, window_size: int = 73,
                 local_window_size: int = 21, seq_feature_dim: int = 1280,
                 struct_feature_dim: int = 512, cache_size: int = 256,
                 structure_mask_token_value: float = 1.0, graph_cache: Optional["GraphFeatureLMDBCache"] = None, use_structure_graph: bool = False) -> None:
        if not HAS_LMDB or lmdb is None:
            raise RuntimeError("protein dedup LMDB requires python-lmdb")
        self.sample_path, self.protein_path = str(sample_path), str(protein_path)
        self.length, self._labels = int(length), label_matrix
        self.window_size, self.local_window_size = int(window_size), int(local_window_size)
        self.seq_feature_dim, self.struct_feature_dim = int(seq_feature_dim), int(struct_feature_dim)
        self.structure_mask_token_value, self.cache_size = float(structure_mask_token_value), max(1, int(cache_size))
        self.graph_cache = graph_cache
        self.use_structure_graph = bool(use_structure_graph and graph_cache is not None)
        self._sample_env = self._protein_env = self._sample_txn = self._protein_txn = None
        self._protein_cache = OrderedDict()

    def _open_txns(self):
        read_ahead = not bool(getattr(self, 'disable_readahead', False))
        if self._sample_env is None:
            self._sample_env = lmdb.open(self.sample_path, readonly=True, lock=False, readahead=read_ahead, meminit=False, max_readers=2048)
        if self._protein_env is None:
            self._protein_env = lmdb.open(self.protein_path, readonly=True, lock=False, readahead=read_ahead, meminit=False, max_readers=2048)
        if self._sample_txn is None:
            self._sample_txn = self._sample_env.begin(buffers=True)
        if self._protein_txn is None:
            self._protein_txn = self._protein_env.begin(buffers=True)
        return self._sample_txn, self._protein_txn

    def __len__(self):
        return self.length

    def get_label_matrix(self):
        if self._labels is None:
            self._labels = np.asarray([self[i]["label"].numpy() for i in range(self.length)], dtype=np.float32)
        return self._labels

    @staticmethod
    def _window(full, pos_1based, length, dim):
        # Keep dataset outputs in float32 so dry-run/model-smoke and the
        # regular path agree with Linear weights; LMDB storage remains FP16.
        out = torch.zeros((int(length), int(dim)), dtype=torch.float32)
        if full is None or not isinstance(full, torch.Tensor) or full.ndim != 2 or full.shape[0] <= 0:
            return out
        idx, half = int(pos_1based) - 1, int(length) // 2
        a, b = max(0, idx-half), min(int(full.shape[0]), idx+half+1)
        if b <= a: return out
        dst, width = a-(idx-half), min(b-a, int(length)-(a-(idx-half)))
        if width > 0: out[dst:dst+width, :min(dim, int(full.shape[1]))] = full[a:a+width, :min(dim, int(full.shape[1]))].to(torch.float32)
        return out

    def _get_protein(self, key, txn):
        key = str(key)
        value = self._protein_cache.get(key)
        if value is not None:
            self._protein_cache.move_to_end(key); return value
        raw = txn.get(key.encode())
        value = pickle.loads(bytes(raw)) if raw is not None else {}
        self._protein_cache[key] = value; self._protein_cache.move_to_end(key)
        while len(self._protein_cache) > self.cache_size: self._protein_cache.popitem(last=False)
        return value

    def __getitem__(self, idx):
        stxn, ptxn = self._open_txns()
        raw = stxn.get(f"{int(idx):09d}".encode())
        if raw is None: raise IndexError(idx)
        sample = pickle.loads(bytes(raw)); protein = self._get_protein(sample.get('protein_key', ''), ptxn)
        pos = int(sample.get('position', -1))
        seq_full = protein.get('sequence_features'); struct_full = protein.get('structure_features')
        if not isinstance(seq_full, torch.Tensor) and seq_full is not None: seq_full = torch.as_tensor(seq_full)
        if not isinstance(struct_full, torch.Tensor) and struct_full is not None: struct_full = torch.as_tensor(struct_full)
        seq_win = self._window(seq_full, pos, self.window_size, self.seq_feature_dim)
        seq_local = self._window(seq_full, pos, self.local_window_size, self.seq_feature_dim)
        seq_global = protein.get('sequence_mean')
        seq_global = torch.as_tensor(seq_global, dtype=torch.float32).reshape(-1) if seq_global is not None else seq_win.mean(0)
        seq_global = seq_global[:self.seq_feature_dim]
        if seq_global.numel() < self.seq_feature_dim: seq_global = F.pad(seq_global, (0, self.seq_feature_dim-seq_global.numel()))
        struct_win = self._window(struct_full, pos, self.window_size, self.struct_feature_dim)
        struct_global = protein.get('structure_mean')
        struct_global = torch.as_tensor(struct_global, dtype=torch.float32).reshape(-1) if struct_global is not None else torch.zeros(self.struct_feature_dim, dtype=torch.float32)
        struct_global = struct_global[:self.struct_feature_dim]
        if struct_global.numel() < self.struct_feature_dim: struct_global = F.pad(struct_global, (0, self.struct_feature_dim-struct_global.numel()))
        if not bool(protein.get('structure_available', False)): struct_win[:, -1] = self.structure_mask_token_value
        out, seq = dict(sample), dict(sample.get('sequence', {}))
        seq.update(window_features=seq_win, local_features=seq_local, global_features=seq_global,
                   global_context_available=torch.tensor(1.0 if protein.get('sequence_mean') is not None else 0.0))
        out['sequence'] = seq
        out['structure'] = struct_win; out['structure_global_features'] = struct_global
        out['structure_available'] = torch.tensor(1.0 if protein.get('structure_available', False) else 0.0)
        out['structure_global_available'] = torch.tensor(1.0 if protein.get('structure_mean') is not None else 0.0)
        graph = None
        if self.use_structure_graph and self.graph_cache is not None:
            graph = self.graph_cache.get_graph(str(sample.get('uniprot_id', '')), pos, '')
        out['structure_graph'] = graph
        out['structure_graph_available'] = torch.tensor(1.0 if graph is not None else 0.0)
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ('_sample_env','_protein_env','_sample_txn','_protein_txn'): state[k] = None
        state['_protein_cache'] = OrderedDict(); return state

    def __del__(self):
        for k in ('_sample_txn','_protein_txn'):
            x = getattr(self, k, None)
            if x is not None:
                try: x.abort()
                except Exception: pass
        for k in ('_sample_env','_protein_env'):
            x = getattr(self, k, None)
            if x is not None:
                try: x.close()
                except Exception: pass


def _collate_graphs(graphs: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    node_dim = 0
    edge_dim = 0
    for graph in graphs:
        if graph is None:
            continue
        x = graph.get("x")
        edge_attr = graph.get("edge_attr")
        if isinstance(x, torch.Tensor) and x.ndim == 2:
            node_dim = max(node_dim, int(x.shape[1]))
        if isinstance(edge_attr, torch.Tensor) and edge_attr.ndim >= 2:
            edge_dim = max(edge_dim, int(edge_attr.shape[1]))
    if node_dim <= 0:
        return None
    edge_dim = edge_dim or 1

    xs: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    batch_indices: List[torch.Tensor] = []
    poss: List[torch.Tensor] = []
    esm_ifs: List[torch.Tensor] = []
    esm_masks: List[torch.Tensor] = []
    center_masks: List[torch.Tensor] = []
    token_indices: List[torch.Tensor] = []
    node_offset = 0
    for idx, graph in enumerate(graphs):
        if graph is None:
            x = torch.zeros((1, node_dim), dtype=torch.float32)
            pos = torch.zeros((1, 3), dtype=torch.float32)
            esm_if = torch.zeros((1, 512), dtype=torch.float32)
            esm_mask = torch.zeros((1,), dtype=torch.float32)
            center_mask = torch.ones((1,), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, edge_dim), dtype=torch.float32)
            token_index = torch.full((1,), -1, dtype=torch.long)
        else:
            x = graph.get("x")
            pos = graph.get("pos")
            esm_if = graph.get("esm_if")
            esm_mask = graph.get("esm_if_mask")
            center_mask = graph.get("center_mask")
            edge_index = graph.get("edge_index")
            edge_attr = graph.get("edge_attr")
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                x = torch.zeros((1, node_dim), dtype=torch.float32)
            elif int(x.shape[1]) != node_dim:
                if int(x.shape[1]) > node_dim:
                    x = x[:, :node_dim]
                else:
                    x = F.pad(x, (0, node_dim - int(x.shape[1])))
            n_nodes = int(x.shape[0])
            if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.shape[0] != n_nodes:
                pos = torch.zeros((n_nodes, 3), dtype=torch.float32)
            if not isinstance(esm_if, torch.Tensor) or esm_if.ndim != 2 or esm_if.shape[0] != n_nodes:
                esm_if = torch.zeros((n_nodes, 512), dtype=torch.float32)
            elif int(esm_if.shape[1]) != 512:
                esm_if = esm_if[:, :512] if int(esm_if.shape[1]) > 512 else F.pad(esm_if, (0, 512 - int(esm_if.shape[1])))
            if not isinstance(esm_mask, torch.Tensor) or esm_mask.reshape(-1).shape[0] != n_nodes:
                esm_mask = torch.zeros((n_nodes,), dtype=torch.float32)
            else:
                esm_mask = esm_mask.reshape(-1).to(torch.float32)
            if not isinstance(center_mask, torch.Tensor) or center_mask.reshape(-1).shape[0] != n_nodes:
                center_mask = torch.zeros((n_nodes,), dtype=torch.float32)
            else:
                center_mask = center_mask.reshape(-1).to(torch.float32)
            if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            if edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
            if not isinstance(edge_attr, torch.Tensor) or edge_attr.ndim < 2:
                edge_attr = torch.zeros((edge_index.shape[1], edge_dim), dtype=torch.float32)
            elif int(edge_attr.shape[1]) != edge_dim:
                if int(edge_attr.shape[1]) > edge_dim:
                    edge_attr = edge_attr[:, :edge_dim]
                else:
                    edge_attr = F.pad(edge_attr, (0, edge_dim - int(edge_attr.shape[1])))
            if edge_attr.shape[0] != edge_index.shape[1]:
                current_e = int(edge_index.shape[1])
                if int(edge_attr.shape[0]) > current_e:
                    edge_attr = edge_attr[:current_e]
                else:
                    pad_rows = torch.zeros((current_e - int(edge_attr.shape[0]), edge_dim), dtype=edge_attr.dtype)
                    edge_attr = torch.cat([edge_attr, pad_rows], dim=0)
            token_index = graph.get("token_index")
            if token_index is not None:
                token_index = torch.as_tensor(token_index).reshape(-1).to(torch.long)
                if token_index.shape[0] > n_nodes:
                    token_index = token_index[:n_nodes]
                elif token_index.shape[0] < n_nodes:
                    token_index = torch.cat(
                        [token_index, torch.full((n_nodes - token_index.shape[0],), -1, dtype=torch.long)]
                    )
            else:
                token_index = torch.full((n_nodes,), -1, dtype=torch.long)
        token_indices.append(token_index.to(torch.long))
        xs.append(x)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + node_offset)
            edge_attrs.append(edge_attr)
        batch_indices.append(torch.full((x.shape[0],), idx, dtype=torch.long))
        poss.append(pos.to(torch.float32)); esm_ifs.append(esm_if.to(torch.float32))
        esm_masks.append(esm_mask.to(torch.float32)); center_masks.append(center_mask.to(torch.float32))
        node_offset += x.shape[0]

    if xs:
        x_all = torch.cat(xs, dim=0)
    else:
        x_all = torch.zeros((1, node_dim), dtype=torch.float32)
    if edge_indices:
        edge_index_all = torch.cat(edge_indices, dim=1)
        edge_attr_all = torch.cat(edge_attrs, dim=0)
    else:
        edge_index_all = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_all = torch.zeros((0, edge_dim), dtype=torch.float32)
    batch_index_all = torch.cat(batch_indices, dim=0) if batch_indices else torch.zeros((x_all.shape[0],), dtype=torch.long)
    return {
        "x": x_all,
        "pos": torch.cat(poss, dim=0),
        "esm_if": torch.cat(esm_ifs, dim=0),
        "esm_if_mask": torch.cat(esm_masks, dim=0),
        "center_mask": torch.cat(center_masks, dim=0),
        "edge_index": edge_index_all,
        "edge_attr": edge_attr_all,
        "batch_index": batch_index_all,
        "token_index": torch.cat(token_indices, dim=0) if token_indices else torch.zeros((x_all.shape[0],), dtype=torch.long),
    }


def _drop_lmdb_mmap_pages() -> None:
    """Release pages of the large read-only LMDB mmap after a batch.

    The F4 Tensor LMDB is hundreds of GB while the remote host has much less
    RAM. Linux otherwise keeps pages touched by each DataLoader worker in its
    RSS, causing worker memory to grow linearly through an epoch. This is
    opt-in so legacy H5/LMDB paths are unchanged.
    """
    global _LMDB_MMAP_SPANS, _LIBC_MADVISE
    if os.environ.get("PTM_DROP_LMDB_PAGES", "0") != "1":
        return
    threshold_mb = int(os.environ.get("PTM_DROP_LMDB_PAGES_THRESHOLD_MB", "0"))
    if threshold_mb > 0:
        try:
            rss_pages = int(open("/proc/self/statm", "r", encoding="utf-8").read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            if rss_pages * page_size < threshold_mb * 1024 * 1024:
                return
        except (OSError, ValueError):
            pass
    if _LMDB_MMAP_SPANS is None:
        spans: List[Tuple[int, int]] = []
        try:
            with open("/proc/self/maps", "r", encoding="utf-8") as handle:
                for line in handle:
                    if "data.mdb" not in line or "preprocessed_f4_tensor_lmdb" not in line:
                        continue
                    start_end = line.split(None, 1)[0]
                    start_s, end_s = start_end.split("-", 1)
                    spans.append((int(start_s, 16), int(end_s, 16)))
        except (OSError, ValueError):
            spans = []
        _LMDB_MMAP_SPANS = spans
        if spans:
            _LIBC_MADVISE = ctypes.CDLL(None).madvise
            _LIBC_MADVISE.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            _LIBC_MADVISE.restype = ctypes.c_int
    if _LIBC_MADVISE is not None:
        # MADV_DONTNEED = 4. Tensor values have already been copied out of
        # the mmap before this function is called.
        for start, end in _LMDB_MMAP_SPANS or []:
            _LIBC_MADVISE(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 4)


def multilabel_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    window_size = int(batch[0]["sequence"]["window_features"].shape[0])
    center_idx = window_size // 2
    position_ids = torch.arange(window_size, dtype=torch.long)
    relative_offsets = position_ids - center_idx
    site_indicator = torch.zeros(window_size, dtype=torch.float32)
    site_indicator[center_idx] = 1.0
    empty_structure = torch.empty((window_size, 0), dtype=torch.float32)
    empty_global = torch.empty((0,), dtype=torch.float32)
    empty_micro = torch.empty((0,), dtype=torch.float32)
    graph_list = [x.get("structure_graph") for x in batch]
    graph_batch = _collate_graphs(graph_list) if any(g is not None for g in graph_list) else None
    payload = {
        "sample_indices": [x.get("sample_index") for x in batch],
        "uniprot_ids": [x["uniprot_id"] for x in batch],
        "positions": [x["position"] for x in batch],
        "residues": [x["residue"] for x in batch],
        "sequence_window_str": [x.get("sequence_window_str") for x in batch],
        "source_tasks": [x.get("source_tasks", "") for x in batch],
        "source_splits": [x.get("source_splits", "") for x in batch],
        "known_task_count": [x.get("known_task_count", -1) for x in batch],
        "positive_task_count": [x.get("positive_task_count", -1) for x in batch],
        "negative_task_count": [x.get("negative_task_count", -1) for x in batch],
        "unknown_task_count": [x.get("unknown_task_count", -1) for x in batch],
        "sequence": {
            "window_features": torch.stack([x["sequence"]["window_features"] for x in batch]),
            "local_features": torch.stack([x["sequence"]["local_features"] for x in batch]),
            "global_features": torch.stack([x["sequence"]["global_features"] for x in batch]),
            "global_context_available": torch.stack(
                [x["sequence"].get("global_context_available", torch.tensor(0.0)) for x in batch]
            ),
            "residue_types": torch.stack([x["sequence"]["residue_types"] for x in batch]),
            "position_ids": torch.stack([x["sequence"].get("position_ids", position_ids) for x in batch]),
            "relative_position_offsets": torch.stack([x["sequence"].get("relative_position_offsets", relative_offsets) for x in batch]),
            "site_indicator": torch.stack([x["sequence"].get("site_indicator", site_indicator) for x in batch]),
        },
        "structure": torch.stack([x.get("structure", empty_structure) for x in batch]),
        "structure_global_features": torch.stack([x.get("structure_global_features", empty_global) for x in batch]),
        "structure_global_available": torch.stack([x.get("structure_global_available", torch.tensor(0.0)) for x in batch]),
        "structure_available": torch.stack([x.get("structure_available", torch.tensor(0.0)) for x in batch]),
        "structure_graph_available": torch.stack([x.get("structure_graph_available", torch.tensor(0.0)) for x in batch]),
        "micro_env_features": torch.stack([x.get("micro_env_features", empty_micro) for x in batch]),
        "micro_env_available": torch.stack([x.get("micro_env_available", torch.tensor(0.0)) for x in batch]),
        "motif_bias": torch.stack([x["motif_bias"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        "loss_weight": torch.stack([x.get("loss_weight", torch.ones_like(x["label"])) for x in batch]),
        "label_state": torch.stack([x.get("label_state", torch.zeros_like(x["label"], dtype=torch.long)) for x in batch]),
        "label_reliability": torch.stack([x.get("label_reliability", torch.ones_like(x["label"])) for x in batch]),
        "teacher_probs": torch.stack([x.get("teacher_probs", torch.full_like(x["label"], -1.0)) for x in batch]),
        "teacher_mask": torch.stack([x.get("teacher_mask", torch.zeros_like(x["label"])) for x in batch]),
        "teacher_weight": torch.stack([x.get("teacher_weight", torch.ones_like(x["label"])) for x in batch]),
    }
    if graph_batch is not None:
        payload["structure_graph"] = graph_batch
    _drop_lmdb_mmap_pages()
    return payload


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {
        "sequence": {
            "window_features": batch["sequence"]["window_features"].to(device, non_blocking=True),
            "local_features": batch["sequence"]["local_features"].to(device, non_blocking=True),
            "global_features": batch["sequence"]["global_features"].to(device, non_blocking=True),
            "global_context_available": batch["sequence"]["global_context_available"].to(device, non_blocking=True),
            "residue_types": batch["sequence"]["residue_types"].to(device, non_blocking=True),
            "position_ids": batch["sequence"]["position_ids"].to(device, non_blocking=True),
            "relative_position_offsets": batch["sequence"]["relative_position_offsets"].to(device, non_blocking=True),
            "site_indicator": batch["sequence"]["site_indicator"].to(device, non_blocking=True),
        },
        "structure": batch["structure"].to(device, non_blocking=True),
        "structure_global_features": batch["structure_global_features"].to(device, non_blocking=True),
        "structure_global_available": batch["structure_global_available"].to(device, non_blocking=True),
        "structure_available": batch["structure_available"].to(device, non_blocking=True),
        "micro_env_features": batch["micro_env_features"].to(device, non_blocking=True),
        "micro_env_available": batch.get("micro_env_available", torch.zeros_like(batch["structure_available"])).to(device, non_blocking=True),
        "motif_bias": batch["motif_bias"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }
    if "loss_weight" in batch:
        out["loss_weight"] = batch["loss_weight"].to(device, non_blocking=True)
    for key in ("teacher_probs", "teacher_mask", "teacher_weight"):
        if key in batch:
            out[key] = batch[key].to(device, non_blocking=True)
    graph = batch.get("structure_graph")
    if isinstance(graph, dict):
        out["structure_graph"] = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in graph.items()
        }
    if "structure_graph_available" in batch:
        out["structure_graph_available"] = batch["structure_graph_available"].to(device, non_blocking=True)
    # NOTE: label_state / label_reliability are intentionally not forwarded here.
    # The reported runs in the manuscript use standard ASL with unknown masking
    # (supplement S6.1); the state-aware branch in the criterion is therefore
    # inactive. To enable state-weighted loss, copy those keys here and retrain.
    return out
