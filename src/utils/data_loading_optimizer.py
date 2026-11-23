import os
import sys

# 设置 HDF5 环境变量以避免文件锁定问题
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pandas as pd
import logging
import atexit
import threading
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, cpu_count
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Iterable, Iterator, Generator

import torch
from torch.utils.data import Dataset, Sampler

try:
    from Bio import SeqIO
    import Bio.PDB as bpdb
except ImportError:
    SeqIO = None
    bpdb = None

try:
    import lmdb  # type: ignore
    _LMDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    lmdb = None  # type: ignore
    _LMDB_AVAILABLE = False

from src.utils.ptm_residue_info import resolve_ptm_residue_info

LMDB_CACHE_FILENAME = "protein_features.lmdb"
DEFAULT_LMDB_MAP_SIZE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB 默认容量


def _resolve_lmdb_map_size() -> int:
    """Resolve LMDB map size from environment variables with sane fallback."""

    env_value = os.environ.get("PTM_LMDB_MAP_SIZE")
    if env_value:
        try:
            map_size = int(env_value)
            if map_size > 0:
                return map_size
            logger.warning("PTM_LMDB_MAP_SIZE 必须为正数，当前值 %s 将被忽略", env_value)
        except ValueError:
            logger.warning("无法解析 PTM_LMDB_MAP_SIZE='%s'，将使用默认值", env_value)
    return DEFAULT_LMDB_MAP_SIZE_BYTES

logger = logging.getLogger(__name__)


def _sanitize_ptm_type(raw_value: Any, target_ptm_type: str, override_cache: Optional[Set[str]] = None, context: str = "") -> str:
    """Normalize ptm_type values to avoid leaking label information."""
    canonical_value = str(target_ptm_type).strip()
    normalized_target = canonical_value.lower()

    if raw_value is None:
        raw_str = ""
    else:
        raw_str = str(raw_value).strip()

    raw_lower = raw_str.lower()
    if raw_lower and raw_lower != normalized_target:
        reason = "未知 PTM 类型"
        if raw_lower.startswith("non_") and raw_lower.endswith(normalized_target):
            reason = "检测到非目标 PTM 标记"
        cache_key = f"{context}:{raw_lower}" if context else raw_lower
        if override_cache is not None and cache_key not in override_cache:
            logger.warning(
                "DataLoadingOptimizer: 样本 ptm_type='%s' 将统一映射为 '%s' (%s)",
                raw_str,
                canonical_value,
                reason,
            )
            override_cache.add(cache_key)
        return canonical_value

    return canonical_value


def _normalize_uniprot_id(value: Any) -> Optional[str]:
    """将原始UniProt ID标准化为合法字符串，非法值返回None"""
    if value is None:
        return None

    # 处理缺失值/NaN
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        # 某些自定义对象不支持 pd.isna，忽略此处错误
        pass

    # 转为字符串并去除首尾空白
    normalized = str(value).strip()
    if not normalized:
        return None

    lower = normalized.lower()
    if lower in {"nan", "none", "null"}:
        return None

    return normalized


def normalize_uniprot_id(value: Any) -> Optional[str]:
    """公共接口，包装内部的UniProt ID标准化逻辑。"""
    return _normalize_uniprot_id(value)


def _parse_env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_env_int(key: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return max(value, minimum)
    except ValueError:
        logger.warning("无法解析环境变量 %s='%s'，将使用默认值 %s", key, raw, default)
        return default


def _parse_env_float(key: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
        if minimum is not None:
            value = max(value, minimum)
        return value
    except ValueError:
        logger.warning("无法解析环境变量 %s='%s'，将使用默认值 %.3f", key, raw, default)
        return default


def _extract_ca_coords_fast(pdb_path: str) -> Optional[np.ndarray]:
    """快速从PDB文件中提取CA原子坐标，避免Bio.PDB的沉重开销。"""
    coords = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
        if coords:
            return np.array(coords, dtype=np.float32)
    except Exception:
        pass
    return None


def load_sequence_from_fasta(protein_id: str, fasta_dir: str) -> Optional[str]:
    """从FASTA文件加载序列。"""
    if not fasta_dir or not SeqIO:
        return None
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    if not os.path.exists(fasta_path):
        return None
    try:
        record = SeqIO.read(fasta_path, "fasta")
        return str(record.seq)
    except Exception:
        return None


def _compute_single_protein_micro_env(protein_id: str, pdb_dir: str) -> Tuple[str, Optional[np.ndarray]]:
    """
    计算单个蛋白质的微环境特征。
    必须是顶层函数以便 ProcessPoolExecutor 序列化。
    """
    if not pdb_dir:
        return protein_id, None
    
    pdb_path = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if not os.path.exists(pdb_path):
        return protein_id, None
        
    # 优先使用快速解析器
    coords = _extract_ca_coords_fast(pdb_path)
    
    # 回退到 Bio.PDB
    if coords is None and bpdb:
        try:
            parser = bpdb.PDBParser(QUIET=True)
            structure = parser.get_structure(protein_id, pdb_path)
            coords_list = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords_list.append(residue['CA'].get_coord())
                break
            if coords_list:
                coords = np.array(coords_list, dtype=np.float32)
        except Exception:
            pass
            
    if coords is None or len(coords) == 0:
        return protein_id, None
        
    # 计算特征
    try:
        num_residues = len(coords)
        features = np.zeros((num_residues, 6), dtype=np.float32)
        
        # 向量化计算距离矩阵 (注意内存消耗，对于超大蛋白可能需要分块)
        if num_residues < 2000:
            # 小蛋白使用广播，速度快
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists_matrix = np.linalg.norm(diff, axis=2)
            
            for i in range(num_residues):
                dists = dists_matrix[i]
                
                # 1. 局部密度 (10A内的邻居数)
                radius_10 = 10.0
                mask_10 = (dists < radius_10) & (dists > 1e-6)
                local_density_10 = np.sum(mask_10)
                
                # 2. 局部凹凸性 (到邻居质心的距离)
                if local_density_10 > 0:
                    neighbor_coords = coords[mask_10]
                    centroid = np.mean(neighbor_coords, axis=0)
                    dist_to_centroid = np.linalg.norm(coords[i] - centroid)
                    
                    # 3. 邻居平均距离 (10A)
                    avg_dist_10 = np.mean(dists[mask_10])
                else:
                    dist_to_centroid = 0.0
                    avg_dist_10 = 0.0
                
                # 4. 紧密堆积 (5A内的邻居数)
                radius_5 = 5.0
                mask_5 = (dists < radius_5) & (dists > 1e-6)
                local_density_5 = np.sum(mask_5)
                
                # 5. 紧密平均距离 (5A)
                if local_density_5 > 0:
                    avg_dist_5 = np.mean(dists[mask_5])
                else:
                    avg_dist_5 = 0.0
                    
                # 6. 最近邻距离
                dists_no_self = dists[dists > 1e-6]
                if len(dists_no_self) > 0:
                    min_dist = np.min(dists_no_self)
                else:
                    min_dist = 10.0 # Default large value
                    
                features[i] = [
                    local_density_10 / 20.0,
                    dist_to_centroid / 5.0,
                    avg_dist_10 / 10.0,
                    local_density_5 / 10.0,
                    avg_dist_5 / 5.0,
                    min_dist / 5.0
                ]
        else:
            # 大蛋白循环计算，省内存
            for i in range(num_residues):
                center_coord = coords[i]
                dists = np.linalg.norm(coords - center_coord, axis=1)
                
                # 1. 局部密度 (10A内的邻居数)
                radius_10 = 10.0
                mask_10 = (dists < radius_10) & (dists > 1e-6)
                neighbor_indices_10 = np.where(mask_10)[0]
                local_density_10 = len(neighbor_indices_10)
                
                # 2. 局部凹凸性 (到邻居质心的距离)
                if local_density_10 > 0:
                    neighbor_coords = coords[neighbor_indices_10]
                    centroid = np.mean(neighbor_coords, axis=0)
                    dist_to_centroid = np.linalg.norm(center_coord - centroid)
                    
                    # 3. 邻居平均距离 (10A)
                    avg_dist_10 = np.mean(dists[neighbor_indices_10])
                else:
                    dist_to_centroid = 0.0
                    avg_dist_10 = 0.0
                
                # 4. 紧密堆积 (5A内的邻居数)
                radius_5 = 5.0
                mask_5 = (dists < radius_5) & (dists > 1e-6)
                neighbor_indices_5 = np.where(mask_5)[0]
                local_density_5 = len(neighbor_indices_5)
                
                # 5. 紧密平均距离 (5A)
                if local_density_5 > 0:
                    avg_dist_5 = np.mean(dists[neighbor_indices_5])
                else:
                    avg_dist_5 = 0.0
                    
                # 6. 最近邻距离
                dists_no_self = dists[dists > 1e-6]
                if len(dists_no_self) > 0:
                    min_dist = np.min(dists_no_self)
                else:
                    min_dist = 10.0 # Default large value
                
                features[i] = [
                    local_density_10 / 20.0,
                    dist_to_centroid / 5.0,
                    avg_dist_10 / 10.0,
                    local_density_5 / 10.0,
                    avg_dist_5 / 5.0,
                    min_dist / 5.0
                ]
        
        return protein_id, features
    except Exception:
        return protein_id, None


class LazyProteinEntry:
    """按需从HDF5文件读取窗口特征，避免整蛋白载入内存"""

    __slots__ = (
        "protein_id",
        "_group",
        "_sequence_dataset",
        "_sequence_cls_ds",
        "_sequence_mean_ds",
        "_structure_dataset",
        "_structure_cls_ds",
        "_structure_mean_ds",
        "_sequence_cls_cache",
        "_sequence_mean_cache",
        "_structure_cls_cache",
        "_structure_mean_cache",
        "_structure_coords_cache",
        "_micro_env_cache",
        "_sequence_str",
        "pdb_dir",
    )

    def __init__(self, protein_id: str, protein_group: "h5py.Group", fasta_dir: Optional[str] = None, pdb_dir: Optional[str] = None) -> None:
        self.protein_id = protein_id
        self.pdb_dir = pdb_dir
        self._group = protein_group
        
        self._sequence_dataset = None
        self._sequence_cls_ds = None
        self._sequence_mean_ds = None
        self._structure_dataset = None
        self._structure_cls_ds = None
        self._structure_mean_ds = None
        
        # Try to load sequence string from attributes
        try:
            seq_bytes = self._group.attrs.get("sequence")
            if isinstance(seq_bytes, bytes):
                self._sequence_str = seq_bytes.decode('utf-8')
            else:
                self._sequence_str = str(seq_bytes) if seq_bytes is not None else None
        except Exception:
            self._sequence_str = None

        # Fallback to FASTA if sequence is missing
        if not self._sequence_str and fasta_dir:
            self._sequence_str = self._load_sequence_from_fasta(protein_id, fasta_dir)

        self._sequence_cls_cache: Optional[np.ndarray] = None
        self._sequence_mean_cache: Optional[np.ndarray] = None
        self._structure_cls_cache: Optional[np.ndarray] = None
        self._structure_mean_cache: Optional[np.ndarray] = None
        self._structure_coords_cache: Optional[np.ndarray] = None
        self._micro_env_cache: Optional[np.ndarray] = None

    def _load_sequence_from_fasta(self, protein_id: str, fasta_dir: str) -> Optional[str]:
        if not SeqIO:
            return None
        fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
        if not os.path.exists(fasta_path):
            return None
        try:
            record = SeqIO.read(fasta_path, "fasta")
            return str(record.seq)
        except Exception:
            return None

    @property
    def sequence_dataset(self) -> Optional["h5py.Dataset"]:
        if self._sequence_dataset is None and self._group:
            self._sequence_dataset = self._group.get("sequence_features")
        return self._sequence_dataset

    @property
    def structure_dataset(self) -> Optional["h5py.Dataset"]:
        if self._structure_dataset is None and self._group:
            self._structure_dataset = self._group.get("structure_features")
        return self._structure_dataset

    def _cached_array(self, dataset: Optional["h5py.Dataset"], cache_name: str) -> Optional[np.ndarray]:
        if dataset is None:
            return None
        cached = getattr(self, cache_name)
        if cached is None:
            data = dataset[()]
            array = np.asarray(data, dtype=np.float32)
            setattr(self, cache_name, array)
            return array
        return cached

    @property
    def sequence_cls(self) -> Optional[np.ndarray]:
        if self._sequence_cls_ds is None and self._group:
            self._sequence_cls_ds = self._group.get("sequence_cls")
        return self._cached_array(self._sequence_cls_ds, "_sequence_cls_cache")

    @property
    def sequence_mean(self) -> Optional[np.ndarray]:
        if self._sequence_mean_ds is None and self._group:
            self._sequence_mean_ds = self._group.get("sequence_mean")
        return self._cached_array(self._sequence_mean_ds, "_sequence_mean_cache")

    @property
    def structure_cls(self) -> Optional[np.ndarray]:
        if self._structure_cls_ds is None and self.has_structure:
            self._structure_cls_ds = self._group.get("structure_cls")
        return self._cached_array(self._structure_cls_ds, "_structure_cls_cache")

    @property
    def structure_mean(self) -> Optional[np.ndarray]:
        if self._structure_mean_ds is None and self.has_structure:
            self._structure_mean_ds = self._group.get("structure_mean")
        return self._cached_array(self._structure_mean_ds, "_structure_mean_cache")

    @property
    def sequence_length(self) -> int:
        if self.sequence_dataset is None:
            return 0
        return int(self.sequence_dataset.shape[0])

    @property
    def has_structure(self) -> bool:
        return self.structure_dataset is not None

    def clear_cached_vectors(self) -> None:
        """释放已经缓存的全局向量，帮助长期运行时降低峰值内存。"""
        self._sequence_cls_cache = None
        self._sequence_mean_cache = None
        self._structure_cls_cache = None
        self._structure_mean_cache = None

    def close(self) -> None:
        """断开与底层HDF5对象的关联，供缓存回收时调用。"""
        self.clear_cached_vectors()
        self._group = None
        self._sequence_dataset = None
        self._sequence_cls_ds = None
        self._sequence_mean_ds = None
        self._structure_dataset = None
        self._structure_cls_ds = None
        self._structure_mean_ds = None

    @property
    def sequence(self) -> Optional[str]:
        return self._sequence_str

    @property
    def micro_env_features(self) -> Optional[np.ndarray]:
        """获取或计算全蛋白的微环境特征。"""
        if self._micro_env_cache is not None:
            return self._micro_env_cache

        # 如果没有缓存，尝试从HDF5加载
        if self._group and "micro_env_features" in self._group:
            try:
                self._micro_env_cache = self._group["micro_env_features"][()]
                return self._micro_env_cache
            except Exception:
                pass
            
        # 如果有结构坐标，尝试实时计算
        coords = self.structure_coords
        if coords is not None:
            try:
                # 计算全蛋白的微环境特征（对每个残基）
                # 注意：这可能比较耗时，所以只在确实需要时计算
                # 这里我们实现一个简化的计算，或者调用外部的计算函数
                # 为了避免循环引用，我们在方法内部实现核心逻辑或调用静态方法
                features_list = []
                for i in range(len(coords)):
                    feat = DataLoadingOptimizer._compute_single_residue_micro_env(coords, i)
                    features_list.append(feat)
                
                self._micro_env_cache = np.array(features_list, dtype=np.float32)
                return self._micro_env_cache
            except Exception as e:
                # logger.warning(f"实时计算微环境特征失败: {e}")
                pass
                
        return None

    @micro_env_features.setter
    def micro_env_features(self, value: np.ndarray):
        self._micro_env_cache = value

    @property
    def structure_coords(self) -> Optional[np.ndarray]:
        if self._structure_coords_cache is not None:
            return self._structure_coords_cache
            
        if not self.pdb_dir:
            return None
            
        pdb_path = os.path.join(self.pdb_dir, f"{self.protein_id}.pdb")
        if not os.path.exists(pdb_path):
            return None
            
        # 优先使用快速解析器
        coords = _extract_ca_coords_fast(pdb_path)
        if coords is not None:
            self._structure_coords_cache = coords
            return coords
            
        # 回退到 Bio.PDB
        if bpdb:
            try:
                parser = bpdb.PDBParser(QUIET=True)
                structure = parser.get_structure(self.protein_id, pdb_path)
                coords_list = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                coords_list.append(residue['CA'].get_coord())
                    break
                
                if coords_list:
                    self._structure_coords_cache = np.array(coords_list, dtype=np.float32)
                    return self._structure_coords_cache
            except Exception:
                pass
            
        return None


class DataLoadingOptimizer:
    """数据加载优化器"""
    
    def __init__(self, esm_features_path: str, num_workers: int = None, *, lazy_loading: Optional[bool] = None, fasta_dir: Optional[str] = None, pdb_dir: Optional[str] = None):
        """
        初始化数据加载优化器
        
        Args:
            esm_features_path: ESM特征文件路径
            num_workers: 工作进程数，默认为CPU核心数
            fasta_dir: FASTA文件目录，用于补充缺失的序列信息
            pdb_dir: PDB结构文件目录，用于计算微环境几何特征
        """
        self.esm_features_path = esm_features_path
        self.num_workers = num_workers or min(cpu_count(), 8)  # 限制最大进程数
        self.fasta_dir = fasta_dir
        self.pdb_dir = pdb_dir
        self._ptm_override_cache: Set[str] = set()
        self._missing_micro_env_warning_count = 0
        self._lmdb_map_size = _resolve_lmdb_map_size()
        self.lazy_loading = lazy_loading if lazy_loading is not None else _parse_env_bool("PTM_LAZY_LOADING", True)
        self._h5_cache_bytes = _parse_env_int("PTM_H5_CACHE_BYTES", 128 * 1024 * 1024, minimum=8 * 1024 * 1024)
        self._h5_cache_slots = _parse_env_int("PTM_H5_CACHE_SLOTS", 1_048_575, minimum=1024)
        self._h5_cache_w0 = _parse_env_float("PTM_H5_CACHE_W0", 0.75, minimum=0.01)
        self._h5_file: Optional[h5py.File] = None
        self._proteins_group: Optional[h5py.Group] = None
        self._lazy_cache: Dict[str, Any] = {}

        if self.lazy_loading:
            logger.info(
                "懒加载模式已启用，将按需从 %s 读取窗口特征 (HDF5 cache: %.2f MB, slots=%d)",
                self.esm_features_path,
                self._h5_cache_bytes / 1024 / 1024,
                self._h5_cache_slots,
            )
        else:
            logger.info("懒加载模式已禁用，将使用预加载策略")

        atexit.register(self.close)
        
        logger.info(f"初始化数据加载优化器，使用 {self.num_workers} 个工作进程")

    def _load_sequence_from_fasta(self, protein_id: str) -> Optional[str]:
        if not self.fasta_dir or not SeqIO:
            return None
        fasta_path = os.path.join(self.fasta_dir, f"{protein_id}.fasta")
        if not os.path.exists(fasta_path):
            return None
        try:
            record = SeqIO.read(fasta_path, "fasta")
            return str(record.seq)
        except Exception:
            return None

    @property
    def h5_cache_settings(self) -> Dict[str, float]:
        """返回当前HDF5缓存配置，供外部懒加载单元复用。"""
        return {
            "rdcc_nbytes": self._h5_cache_bytes,
            "rdcc_nslots": self._h5_cache_slots,
            "rdcc_w0": self._h5_cache_w0,
        }

    def _get_h5_file(self, mode: str = "r") -> h5py.File:
        # Check if we need to reopen (either file is None, or mode mismatch)
        need_reopen = self._h5_file is None
        if not need_reopen:
            # If current mode is 'r' but we need 'r+', we must reopen
            if mode == "r+" and self._h5_file.mode == "r":
                need_reopen = True
        
        if need_reopen:
            if self._h5_file is not None:
                try:
                    # CRITICAL: When closing/reopening, all existing LazyProteinEntry objects 
                    # holding references to datasets in the old file handle become invalid.
                    # We must clear the cache to force reloading from the new handle.
                    for entry in self._lazy_cache.values():
                        try:
                            if isinstance(entry, LazyProteinEntry):
                                entry.close()
                        except Exception:
                            pass
                    self._lazy_cache.clear()
                    
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                # CRITICAL: Invalidate the group cache when file is closed
                self._proteins_group = None
            
            try:
                self._h5_file = h5py.File(
                    self.esm_features_path,
                    mode,
                    rdcc_nbytes=self._h5_cache_bytes,
                    rdcc_nslots=self._h5_cache_slots,
                    rdcc_w0=self._h5_cache_w0,
                )
                logger.info(
                    "已打开 HDF5 特征文件 (mode=%s)，使用 rdcc_nbytes=%.2f MB, rdcc_nslots=%d, rdcc_w0=%.2f",
                    mode,
                    self._h5_cache_bytes / 1024 / 1024,
                    self._h5_cache_slots,
                    self._h5_cache_w0,
                )
            except Exception as exc:
                # Ensure state is clean on failure
                self._h5_file = None
                self._proteins_group = None
                raise RuntimeError(f"无法打开特征文件 {self.esm_features_path} (mode={mode}): {exc}") from exc
                
        return self._h5_file

    def _get_proteins_group(self, mode: str = "r") -> "h5py.Group":
        # Check conditions to refresh group:
        # 1. Group is None
        # 2. File is None (should imply 1, but safety first)
        # 3. Mode upgrade needed (r -> r+)
        refresh = (self._proteins_group is None) or (self._h5_file is None)
        
        if not refresh and mode == "r+" and self._h5_file is not None:
             if hasattr(self._h5_file, 'mode') and self._h5_file.mode == "r":
                 refresh = True

        if refresh:
            h5_file = self._get_h5_file(mode)
            if "proteins" not in h5_file:
                if mode == "r+":
                    self._proteins_group = h5_file.create_group("proteins")
                else:
                    raise KeyError(f"特征文件 {self.esm_features_path} 中缺少 'proteins' 分组")
            else:
                self._proteins_group = h5_file["proteins"]
        return self._proteins_group

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception as exc:
                logger.warning("关闭特征文件时出现异常: %s", exc)
            finally:
                self._h5_file = None
                self._proteins_group = None
        
    def create_protein_index(self, output_path: str = None) -> Dict[str, Dict]:
        """
        创建蛋白质索引，用于快速查找
        
        Args:
            output_path: 索引文件保存路径
            
        Returns:
            Dict: 蛋白质索引字典
        """
        logger.info("创建蛋白质索引...")
        
        protein_index = {}
        
        proteins_group = self._get_proteins_group()

        for protein_id in proteins_group.keys():
            protein_group = proteins_group[protein_id]

            # 记录蛋白质基本信息
            seq_features = protein_group['sequence_features']
            seq_len = seq_features.shape[0]
            seq_dim = seq_features.shape[1]

            protein_info = {
                'sequence_length': seq_len,
                'sequence_dim': seq_dim,
                'has_structure': 'structure_features' in protein_group
            }

            if protein_info['has_structure']:
                struct_features = protein_group['structure_features']
                protein_info['structure_dim'] = struct_features.shape[1]

            protein_index[protein_id] = protein_info
                
        logger.info(f"创建了 {len(protein_index)} 个蛋白质的索引")
        
        # 保存索引
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(protein_index, f)
            logger.info(f"蛋白质索引已保存到: {output_path}")
            
        return protein_index
    
    def precompute_micro_environment(self, protein_ids: List[str], cache_dir: str = None, use_lmdb: bool = True, save_to_h5: bool = True) -> Dict[str, np.ndarray]:
        """批量预计算蛋白质的微环境特征。"""
        if not self.pdb_dir:
            logger.warning("未指定 pdb_dir，跳过微环境特征预计算")
            return {}
            
        results = {}
        micro_env_cache_path = None
        lmdb_env = None
        
        # 0. 优先尝试从 LMDB/Pickle 加载 (调整顺序，LMDB 优先)
        remaining_ids = list(protein_ids) # Start with all IDs
        lmdb_loaded_ids = set()
        
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            if use_lmdb and _LMDB_AVAILABLE:
                lmdb_path = cache_path / "micro_env_features.lmdb"
                try:
                    # 使用 map_size 自动扩容策略
                    lmdb_env = lmdb.open(
                        str(lmdb_path),
                        map_size=self._lmdb_map_size,
                        subdir=False,
                        create=True,
                        lock=True, 
                        readahead=False, 
                        meminit=False
                    )
                    # 从 LMDB 加载已有的
                    with lmdb_env.begin(write=False) as txn:
                        cursor = txn.cursor()
                        for pid in list(remaining_ids):
                            payload = cursor.get(pid.encode("utf-8"))
                            if payload is not None:
                                try:
                                    feat = pickle.loads(payload)
                                    # [Fix] Validate feature dimension (must be 6)
                                    if len(feat.shape) == 2 and feat.shape[1] == 6:
                                        results[pid] = feat
                                        lmdb_loaded_ids.add(pid)
                                        remaining_ids.remove(pid)
                                except Exception:
                                    continue
                    if results:
                        logger.info(f"已从 LMDB 缓存加载微环境特征，当前总计 {len(results)} 个")
                except Exception as e:
                    logger.warning(f"打开 micro_env LMDB 失败: {e}，将尝试重建或使用其他方式")
                    if lmdb_env:
                        lmdb_env.close()
                        lmdb_env = None

            if not lmdb_env and remaining_ids:
                micro_env_cache_path = cache_path / "micro_env_cache.pkl"
                if micro_env_cache_path.exists():
                    try:
                        with open(micro_env_cache_path, 'rb') as f:
                            pickle_results = pickle.load(f)
                            for pid in list(remaining_ids):
                                if pid in pickle_results:
                                    feat = pickle_results[pid]
                                    if len(feat.shape) == 2 and feat.shape[1] == 6:
                                        results[pid] = feat
                                        lmdb_loaded_ids.add(pid)
                                        remaining_ids.remove(pid)
                        logger.info(f"已从 Pickle 缓存加载微环境特征，当前总计 {len(results)} 个")
                    except Exception as e:
                        logger.warning(f"加载微环境特征缓存失败: {e}")
        
        # 0.5 如果开启了 save_to_h5，检查 LMDB 加载的数据是否已经在 HDF5 中
        # 如果不在，需要写入 HDF5，以免 LazyProteinEntry 无法加载
        if save_to_h5 and lmdb_loaded_ids:
            try:
                # 检查 HDF5 中是否存在
                proteins_group = self._get_proteins_group("r")
                missing_in_h5 = {}
                for pid in lmdb_loaded_ids:
                    # 检查组是否存在，以及 dataset 是否存在
                    in_h5 = False
                    if pid in proteins_group:
                        pg = proteins_group[pid]
                        if "micro_env_features" in pg:
                            in_h5 = True
                    
                    if not in_h5:
                        missing_in_h5[pid] = results[pid]
                
                if missing_in_h5:
                    logger.info(f"发现 {len(missing_in_h5)} 个特征在 LMDB 中但缺失于 HDF5，正在同步...")
                    # Retry mechanism for HDF5 writing
                    max_retries = 1 # [Optimization] Reduce retries to avoid blocking
                    for attempt in range(max_retries):
                        try:
                            self.save_micro_environment_to_h5(missing_in_h5)
                            break
                        except Exception as e:
                            if "file is already open" in str(e):
                                logger.warning(f"HDF5 文件被锁定，跳过同步 (数据在 LMDB 中安全): {e}")
                                break
                            
                            if attempt < max_retries - 1:
                                logger.warning(f"同步 LMDB 特征到 HDF5 失败 (尝试 {attempt+1}/{max_retries}): {e}，正在重试...")
                                time.sleep(1)
                            else:
                                logger.warning(f"同步 LMDB 特征到 HDF5 最终失败 (非关键错误): {e}")
            except Exception as e:
                logger.error(f"同步 LMDB 特征到 HDF5 过程出错: {e}")

        # 1. 其次尝试从 HDF5 加载 (作为备用，且不强求)
        if remaining_ids and save_to_h5:
            try:
                # 使用 "r" 模式尝试读取，如果失败则忽略
                # 注意：如果之前 HDF5 写入失败，这里可能读不到数据
                proteins_group = self._get_proteins_group("r")
                h5_loaded_count = 0
                for pid in list(remaining_ids):
                    if pid in proteins_group and "micro_env_features" in proteins_group[pid]:
                        try:
                            feat = proteins_group[pid]["micro_env_features"][()]
                            if feat is not None and feat.size > 0 and len(feat.shape) == 2 and feat.shape[1] == 6:
                                results[pid] = feat
                                remaining_ids.remove(pid)
                                h5_loaded_count += 1
                        except Exception:
                            pass # 读取单个失败，忽略
                
                if h5_loaded_count > 0:
                    logger.info(f"已从 HDF5 补充加载 {h5_loaded_count} 个蛋白质的微环境特征")
            except Exception as e:
                logger.warning(f"从 HDF5 检查微环境特征失败 (忽略): {e}")

        # 3. 计算真正剩余的
        if not remaining_ids:
            if lmdb_env:
                lmdb_env.close()
            return results
            
        logger.info(f"开始预计算 {len(remaining_ids)} 个蛋白质的微环境特征 (使用 {self.num_workers} 个进程)...")
        
        start_time = time.time()
        
        # 使用进程池并行计算
        newly_computed = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 分批提交任务以避免内存问题
            batch_size = 500
            for i in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[i:i+batch_size]
                futures = [executor.submit(_compute_single_protein_micro_env, pid, self.pdb_dir) for pid in batch_ids]
                
                current_batch_results = {}
                for future in futures:
                    try:
                        pid, features = future.result()
                        if features is not None:
                            results[pid] = features
                            current_batch_results[pid] = features
                            newly_computed[pid] = features
                    except Exception as e:
                        logger.error(f"计算蛋白质微环境特征失败: {e}")
                
                # 及时写入 LMDB
                lmdb_success = False
                if lmdb_env and current_batch_results:
                    try:
                        with lmdb_env.begin(write=True) as txn:
                            for pid, feat in current_batch_results.items():
                                txn.put(pid.encode("utf-8"), pickle.dumps(feat, protocol=pickle.HIGHEST_PROTOCOL))
                        lmdb_success = True
                    except Exception as e:
                        logger.warning(f"写入 LMDB 失败: {e}")

                # 及时写入 HDF5 (断点续传的关键)
                # [Optimization] 如果 LMDB 写入成功，则 HDF5 写入变为可选 (Best Effort)
                # 避免 HDF5 文件锁定导致的训练卡顿
                if save_to_h5 and current_batch_results:
                    try:
                        self.save_micro_environment_to_h5(current_batch_results)
                    except Exception as e:
                        if lmdb_success:
                            logger.debug(f"HDF5 同步跳过 (已保存至 LMDB): {e}")
                        else:
                            logger.warning(f"HDF5 同步失败且 LMDB 不可用: {e}")
                
                if (i + batch_size) % 2000 == 0 or (i + batch_size) >= len(remaining_ids):
                    logger.info(f"已处理 {min(i + batch_size, len(remaining_ids))}/{len(remaining_ids)} 个蛋白质...")
                    
        duration = time.time() - start_time
        logger.info(f"微环境特征预计算完成，耗时 {duration:.2f} 秒，新增 {len(newly_computed)} 个蛋白质")
        
        # 5. 如果使用的是 Pickle 缓存且有新计算的，保存一下
        if not lmdb_env and micro_env_cache_path and newly_computed:
            try:
                # 重新读写整个 pickle 比较慢，但在没有 LMDB 时是兜底
                with open(micro_env_cache_path, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f"微环境特征已保存到 Pickle 缓存: {micro_env_cache_path}")
            except Exception as e:
                logger.warning(f"保存微环境特征缓存失败: {e}")
        
        if lmdb_env:
            lmdb_env.close()
                
        return results

    def save_micro_environment_to_h5(self, micro_env_dict: Dict[str, np.ndarray]) -> None:
        """将预计算的微环境特征保存回 HDF5 文件中。"""
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        return
        
        if not micro_env_dict:
            return
            
        try:
            proteins_group = self._get_proteins_group("r+")
            count = 0
            missing_pids = []
            for pid, features in micro_env_dict.items():
                if pid in proteins_group:
                    pg = proteins_group[pid]
                    # 如果 pg 是数据集而不是组，说明 HDF5 结构有问题
                    if not isinstance(pg, h5py.Group):
                        logger.warning(f"HDF5 中蛋白质 {pid} 不是一个组，跳过写入")
                        continue
                        
                    if "micro_env_features" in pg:
                        del pg["micro_env_features"]
                    pg.create_dataset(
                        "micro_env_features", 
                        data=features, 
                        compression="gzip", 
                        compression_opts=4
                    )
                    count += 1
                else:
                    # 如果蛋白质在 HDF5 中完全不存在，不创建组，因为缺少 sequence_features 也没用
                    missing_pids.append(pid)
                    # logger.debug(f"蛋白质 {pid} 不在 HDF5 中，跳过微环境特征写入")
            
            if count > 0:
                logger.info(f"成功更新/创建了 {count} 个蛋白质的微环境特征到 HDF5")
            if missing_pids:
                logger.warning(f"有 {len(missing_pids)} 个蛋白质在 HDF5 中不存在且创建失败")
        except Exception as e:
            if "file is already open" in str(e) or "BlockingIOError" in str(e):
                logger.warning(f"保存微环境特征到 HDF5 被锁定 (非关键，已跳过): {e}")
            else:
                logger.error(f"保存微环境特征到 HDF5 失败: {e}")

    def save_micro_environment_features(self, micro_env_dict: Dict[str, np.ndarray], cache_dir: str = None) -> None:
        """
        保存微环境特征到 LMDB (如果可用) 和 HDF5。
        用于 OnDemandPTMDataset 在 fallback 计算后保存结果。
        """
        if not micro_env_dict:
            return

        # 1. Update in-memory cache (so subsequent calls get it fast)
        for pid, feat in micro_env_dict.items():
            self._micro_env_cache_external[pid] = feat

        # 2. Save to HDF5 (Persistent storage)
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        # self.save_micro_environment_to_h5(micro_env_dict)
        
        # 3. Save to LMDB (Fast cache) if available
        # We try to use the cache_dir if provided, or fallback to inferring from internal state if possible
        # However, DataLoadingOptimizer doesn't store cache_dir. 
        # But we can try to look for 'micro_env_features.lmdb' in standard locations if needed.
        # For now, we rely on HDF5 persistence which is robust enough.
        # If the user provided cache_dir to precompute_micro_environment, we might have stored it? No.
        
        # Optimization: If we have an open LMDB env (unlikely here as it's closed after preload), we could use it.
        # Given the complexity of reopening LMDB here without a known path, we skip LMDB write for on-demand fallback.
        # HDF5 write is sufficient for persistence across epochs.
        pass

    def preload_protein_features(self, protein_ids: List[str], 
                               cache_dir: str = None) -> Dict[str, Any]:
        """
        预加载指定蛋白质的特征到内存或缓存文件
        
        Args:
            protein_ids: 需要预加载的蛋白质ID列表
            cache_dir: 缓存目录路径
            
        Returns:
            Dict: 如果是懒加载模式，返回预计算的微环境特征；
                  如果是预加载模式，返回完整的蛋白质特征字典。
        """
        logger.info(f"预加载 {len(protein_ids)} 个蛋白质的特征...")

        start_time = time.time()
        
        # 预计算微环境特征 (统一处理，显著提升后续训练速度)
        micro_env_data = {}
        if self.pdb_dir:
            micro_env_data = self.precompute_micro_environment(protein_ids, cache_dir=cache_dir)

        if self.lazy_loading:
            # 懒加载模式下，需构建 LazyProteinEntry 对象并返回
            # 这样 batch_extract_features 才能通过 entry 获取数据
            proteins_group = self._get_proteins_group("r")
            protein_features = {}
            
            # 使用 self._lazy_cache 避免重复创建对象
            for pid in protein_ids:
                if pid not in proteins_group:
                    continue
                
                pg = proteins_group[pid]
                if "sequence_features" not in pg:
                    continue

                if pid in self._lazy_cache:
                    entry = self._lazy_cache[pid]
                else:
                    entry = LazyProteinEntry(pid, pg, self.fasta_dir, self.pdb_dir)
                    self._lazy_cache[pid] = entry
                
                # 注入预计算的微环境特征
                if pid in micro_env_data:
                    entry.micro_env_features = micro_env_data[pid]
                
                protein_features[pid] = entry

            load_time = time.time() - start_time
            logger.info(
                "懒加载预处理完成，已准备 %d 个蛋白质的 LazyEntry (含 %d 个微环境特征)，耗时 %.2f 秒",
                len(protein_features),
                len(micro_env_data),
                load_time
            )
            return protein_features

        # 预加载模式 (Eager mode)
        protein_features: Dict[str, Any] = {}

        cache_path: Optional[Path] = None
        lmdb_env: Optional["lmdb.Environment"] = None
        use_lmdb_cache = False

        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            if _LMDB_AVAILABLE:
                lmdb_path = cache_path / LMDB_CACHE_FILENAME
                try:
                    lmdb_env = lmdb.open(
                        str(lmdb_path),
                        map_size=self._lmdb_map_size,
                        subdir=False,
                        create=True,
                        lock=True,
                        readahead=True,
                        max_readers=max(64, self.num_workers * 4),
                    )
                    use_lmdb_cache = True
                    logger.debug("使用 LMDB 缓存文件: %s", lmdb_path)
                except Exception as exc:  # pragma: no cover - unexpected environment issues
                    logger.warning("打开 LMDB 缓存失败，将回退到 Pickle 缓存: %s", exc)
                    lmdb_env = None
            else:
                logger.warning("当前环境未安装 lmdb，暂时回退到 Pickle 缓存形式")

        def _write_batch_to_lmdb(env: Optional["lmdb.Environment"], items: Dict[str, Dict[str, Any]]) -> None:
            if env is None or not items:
                return

            serialized = {pid: pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL) for pid, features in items.items()}
            attempts = 0
            while attempts < 4:
                try:
                    with env.begin(write=True) as txn:
                        for pid, payload in serialized.items():
                            txn.put(pid.encode("utf-8"), payload)
                    env.sync()
                    return
                except lmdb.MapFullError as exc:  # pragma: no cover - requires full map
                    attempts += 1
                    current_size = env.info().get("map_size", self._lmdb_map_size)
                    new_size = max(current_size * 2, current_size + sum(len(v) for v in serialized.values()))
                    logger.warning(
                        "LMDB 缓存空间不足 (%.2f GB)，正扩容至 %.2f GB (第 %d 次尝试)",
                        current_size / 1024 ** 3,
                        new_size / 1024 ** 3,
                        attempts,
                    )
                    env.set_mapsize(new_size)
                except Exception as exc:  # pragma: no cover - unexpected write error
                    logger.warning("写入 LMDB 缓存失败: %s", exc)
                    return

        cached_count = 0
        remaining_ids = set(protein_ids)

        try:
            if use_lmdb_cache and lmdb_env is not None and remaining_ids:
                with lmdb_env.begin(write=False) as txn:
                    for protein_id in list(remaining_ids):
                        payload = txn.get(protein_id.encode("utf-8"))
                        if payload is None:
                            continue
                        try:
                            protein_features[protein_id] = pickle.loads(payload)
                            remaining_ids.discard(protein_id)
                            cached_count += 1
                        except Exception as exc:
                            logger.warning("从 LMDB 缓存解码蛋白质 %s 失败，将重新加载: %s", protein_id, exc)

                # 迁移遗留的 Pickle 缓存文件
                legacy_loaded: Dict[str, Dict[str, Any]] = {}
                if cache_path is not None and remaining_ids:
                    for protein_id in list(remaining_ids):
                        legacy_file = cache_path / f"{protein_id}.pkl"
                        if not legacy_file.exists():
                            continue
                        try:
                            with open(legacy_file, "rb") as fh:
                                features = pickle.load(fh)
                            protein_features[protein_id] = features
                            remaining_ids.discard(protein_id)
                            legacy_loaded[protein_id] = features
                            cached_count += 1
                        except Exception as exc:
                            logger.warning("迁移旧缓存文件 %s 失败: %s", legacy_file, exc)

                    if legacy_loaded:
                        _write_batch_to_lmdb(lmdb_env, legacy_loaded)
                        for protein_id in legacy_loaded:
                            legacy_file = cache_path / f"{protein_id}.pkl"
                            try:
                                legacy_file.unlink()
                            except OSError as exc:
                                logger.debug("删除旧缓存文件 %s 失败: %s", legacy_file, exc)

            elif cache_path is not None and remaining_ids:
                # Pickle 缓存回退逻辑
                for protein_id in list(remaining_ids):
                    cache_file = cache_path / f"{protein_id}.pkl"
                    if not cache_file.exists():
                        continue
                    try:
                        with open(cache_file, "rb") as fh:
                            protein_features[protein_id] = pickle.load(fh)
                        remaining_ids.discard(protein_id)
                        cached_count += 1
                    except Exception as exc:
                        logger.warning("加载缓存文件 %s 失败: %s", cache_file, exc)

            if cache_path is not None:
                logger.info("从缓存加载了 %d 个蛋白质特征", cached_count)

            remaining_to_load = list(remaining_ids)
            newly_loaded: Dict[str, Dict[str, Any]] = {}

            if remaining_to_load:
                with h5py.File(self.esm_features_path, "r") as f:
                    proteins_group = f["proteins"]

                    for protein_id in remaining_to_load:
                        if protein_id not in proteins_group:
                            logger.warning("蛋白质 %s 不在特征文件中", protein_id)
                            continue

                        protein_group = proteins_group[protein_id]
                        
                        if "sequence_features" not in protein_group:
                            # logger.warning(f"蛋白质 {protein_id} 缺少 sequence_features，跳过")
                            continue

                        features = {
                            "sequence_features": protein_group["sequence_features"][()],
                            "sequence_cls": protein_group["sequence_cls"][()],
                            "sequence_mean": protein_group["sequence_mean"][()],
                        }
                        
                        # Try to load sequence string
                        try:
                            seq_bytes = protein_group.attrs.get("sequence")
                            if isinstance(seq_bytes, bytes):
                                features["sequence_str"] = seq_bytes.decode('utf-8')
                            else:
                                features["sequence_str"] = str(seq_bytes) if seq_bytes is not None else None
                        except Exception:
                            features["sequence_str"] = None

                        if not features["sequence_str"]:
                            features["sequence_str"] = self._load_sequence_from_fasta(protein_id)

                        # Load structure coords if pdb_dir provided
                        features["structure_coords"] = None
                        if self.pdb_dir and bpdb:
                             pdb_path = os.path.join(self.pdb_dir, f"{protein_id}.pdb")
                             if os.path.exists(pdb_path):
                                 try:
                                     parser = bpdb.PDBParser(QUIET=True)
                                     structure = parser.get_structure(protein_id, pdb_path)
                                     coords = []
                                     for model in structure:
                                         for chain in model:
                                             for residue in chain:
                                                 if 'CA' in residue:
                                                     coords.append(residue['CA'].get_coord())
                                         break
                                     if coords:
                                         features["structure_coords"] = np.array(coords, dtype=np.float32)
                                 except Exception:
                                     pass

                        if "structure_features" in protein_group:
                            features.update(
                                {
                                    "structure_features": protein_group["structure_features"][()],
                                    "structure_cls": protein_group["structure_cls"][()],
                                    "structure_mean": protein_group["structure_mean"][()],
                                }
                            )
                        else:
                            features.update(
                                {
                                    "structure_features": None,
                                    "structure_cls": None,
                                    "structure_mean": None,
                                }
                            )

                        protein_features[protein_id] = features
                        newly_loaded[protein_id] = features

                if newly_loaded:
                    if use_lmdb_cache and lmdb_env is not None:
                        _write_batch_to_lmdb(lmdb_env, newly_loaded)
                    elif cache_path is not None:
                        for protein_id, features in newly_loaded.items():
                            cache_file = cache_path / f"{protein_id}.pkl"
                            try:
                                with open(cache_file, "wb") as fh:
                                    pickle.dump(features, fh)
                            except Exception as exc:
                                logger.warning("保存缓存文件 %s 失败: %s", cache_file, exc)

        finally:
            if lmdb_env is not None:
                lmdb_env.close()

        # 确保所有加载的特征都包含预计算的微环境特征 (eager 模式)
        if micro_env_data:
            for pid, feat in protein_features.items():
                if pid in micro_env_data:
                    if isinstance(feat, dict):
                        if "micro_env_features" not in feat or feat["micro_env_features"] is None:
                            feat["micro_env_features"] = micro_env_data[pid]

        load_time = time.time() - start_time
        if protein_features:
            average_time = load_time / len(protein_features)
            logger.info("预加载完成，耗时 %.2f 秒，平均每个蛋白质 %.3f 秒", load_time, average_time)
        else:
            logger.warning("未能预加载任何蛋白质特征，耗时 %.2f 秒", load_time)

        return protein_features
    
    def batch_extract_features_generator(
        self,
        data: pd.DataFrame,
        protein_features: Dict[str, Any],
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = 'phosphorylation',
    ) -> Generator[Dict, None, None]:
        """
        批量提取特征的生成器版本，逐个返回样本以节省内存。
        """
        logger.info(f"开始流式提取 {len(data)} 个样本的特征...")
        
        half_window = window_size // 2
        local_half_window = local_window_size // 2
        target_ptm_label = str(target_ptm_type).strip()

        count = 0
        for idx, row in data.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']

            entry = protein_features.get(uniprot_id)
            if entry is None:
                logger.warning(f"跳过样本 {uniprot_id}:{position}，蛋白质特征未预加载")
                continue

            try:
                if isinstance(entry, LazyProteinEntry):
                    seq_dataset = entry.sequence_dataset
                    if seq_dataset is None:
                        logger.warning("蛋白质 %s 的懒加载入口不可用，跳过样本", uniprot_id)
                        continue
                    seq_len = entry.sequence_length
                    structure_dataset = entry.structure_dataset
                    struct_mean_raw = entry.structure_mean
                    structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32)
                    struct_cls_raw = entry.structure_cls
                    structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32)
                    structure_coords = entry.structure_coords
                    seq_mean_raw = entry.sequence_mean
                    if seq_mean_raw is None:
                        feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                        sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
                    else:
                        sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32)
                else:
                    features = entry
                    seq_dataset = features['sequence_features']
                    seq_len = len(seq_dataset)
                    structure_dataset = features.get('structure_features')
                    structure_mean = features.get('structure_mean')
                    structure_cls = features.get('structure_cls')
                    sequence_mean = features['sequence_mean']

                pos_idx = position - 1  # 转换为0-indexed

                if pos_idx < 0 or pos_idx >= seq_len:
                    logger.warning(f"位置 {position} 超出蛋白质 {uniprot_id} 序列范围")
                    continue

                # 提取序列窗口特征
                seq_window = self._extract_window_features(
                    seq_dataset, pos_idx, window_size, half_window
                )

                seq_local = self._extract_window_features(
                    seq_dataset, pos_idx, local_window_size, local_half_window
                )

                # 提取结构窗口特征
                struct_window = None
                struct_local = None
                micro_env_features = None

                if structure_dataset is not None:
                    struct_window = self._extract_window_features(
                        structure_dataset, pos_idx, window_size, half_window
                    )

                    struct_local = self._extract_window_features(
                        structure_dataset, pos_idx, local_window_size, local_half_window
                    )
                    
                    # 尝试计算微环境特征
                    micro_env_all = None
                    if isinstance(entry, LazyProteinEntry):
                        micro_env_all = entry.micro_env_features
                    elif isinstance(entry, dict):
                        micro_env_all = entry.get("micro_env_features")
                    
                    if micro_env_all is not None and pos_idx < len(micro_env_all):
                        micro_env_features = micro_env_all[pos_idx]
                    elif structure_coords is not None:
                        # 按需计算
                        if self._missing_micro_env_warning_count < 5:
                            logger.warning(f"蛋白质 {uniprot_id} 缺少预计算的微环境特征，正在实时计算 (这会降低训练速度)...")
                        elif self._missing_micro_env_warning_count == 5:
                            logger.warning("已有 5 个蛋白质缺失微环境特征，后续警告将被抑制...")
                        self._missing_micro_env_warning_count += 1
                        
                        micro_env_features = self._compute_single_residue_micro_env(structure_coords, pos_idx)

                ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else target_ptm_label
                ptm_type = _sanitize_ptm_type(ptm_type_raw, target_ptm_type, self._ptm_override_cache, "batch_extract_features")

                label_value = row['label'] if 'label' in row else None
                has_label = label_value is not None and not pd.isna(label_value)
                target_label = int(label_value) if has_label else 0

                sample = {
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'residue': row['residue'],
                    'ptm_type': ptm_type,
                    'ptm_type_raw': ptm_type_raw,
                    'target_ptm_type': target_ptm_label,
                    'sequence_features': seq_window,
                    'local_features': seq_local,
                    'global_features': sequence_mean,
                    'structure_features': struct_window,
                    'structure_local': struct_local,
                    'structure_global': structure_mean,
                    'structure_cls_vector': structure_cls,
                    'micro_env_features': micro_env_features,
                    'is_target_ptm': target_label,
                    'is_phosphorylated': target_label,
                    'has_label': has_label,
                    'label': target_label if has_label else None,
                    'sequence_length': seq_len
                }

                yield sample
                count += 1

                # 进度报告
                if count % 1000 == 0:
                    logger.info(f"已流式处理 {count}/{len(data)} 个样本")

            except Exception as e:
                logger.error(f"处理样本 {uniprot_id}:{position} 时出错: {e}")
                continue

    def batch_extract_features(
        self,
        data: pd.DataFrame,
    protein_features: Dict[str, Any],
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = 'phosphorylation',
    ) -> List[Dict]:
        """
        批量提取特征
        
        Args:
            data: 包含样本信息的DataFrame
            protein_features: 预加载的蛋白质特征
            window_size: 窗口大小
            local_window_size: 局部窗口大小
            
        Returns:
            List: 处理后的样本列表
        """
        logger.info(f"批量提取 {len(data)} 个样本的特征...")
        
        start_time = time.time()
        processed_samples = []
        
        half_window = window_size // 2
        local_half_window = local_window_size // 2
        target_ptm_label = str(target_ptm_type).strip()

        for idx, row in data.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']

            entry = protein_features.get(uniprot_id)
            if entry is None:
                logger.warning(f"跳过样本 {uniprot_id}:{position}，蛋白质特征未预加载")
                continue

            try:
                if isinstance(entry, LazyProteinEntry):
                    seq_dataset = entry.sequence_dataset
                    if seq_dataset is None:
                        logger.warning("蛋白质 %s 的懒加载入口不可用，跳过样本", uniprot_id)
                        continue
                    seq_len = entry.sequence_length
                    structure_dataset = entry.structure_dataset
                    struct_mean_raw = entry.structure_mean
                    structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32)
                    struct_cls_raw = entry.structure_cls
                    structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32)
                    structure_coords = entry.structure_coords
                    seq_mean_raw = entry.sequence_mean
                    if seq_mean_raw is None:
                        feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                        sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
                    else:
                        sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32)
                else:
                    features = entry
                    seq_dataset = features['sequence_features']
                    seq_len = len(seq_dataset)
                    structure_dataset = features.get('structure_features')
                    structure_mean = features.get('structure_mean')
                    structure_cls = features.get('structure_cls')
                    sequence_mean = features['sequence_mean']

                pos_idx = position - 1  # 转换为0-indexed

                if pos_idx < 0 or pos_idx >= seq_len:
                    logger.warning(f"位置 {position} 超出蛋白质 {uniprot_id} 序列范围")
                    continue

                # 提取序列窗口特征
                seq_window = self._extract_window_features(
                    seq_dataset, pos_idx, window_size, half_window
                )

                seq_local = self._extract_window_features(
                    seq_dataset, pos_idx, local_window_size, local_half_window
                )

                # 提取结构窗口特征
                struct_window = None
                struct_local = None
                micro_env_features = None

                if structure_dataset is not None:
                    struct_window = self._extract_window_features(
                        structure_dataset, pos_idx, window_size, half_window
                    )

                    struct_local = self._extract_window_features(
                        structure_dataset, pos_idx, local_window_size, local_half_window
                    )
                    
                    # 尝试计算微环境特征
                    micro_env_all = None
                    if isinstance(entry, LazyProteinEntry):
                        micro_env_all = entry.micro_env_features
                    elif isinstance(entry, dict):
                        micro_env_all = entry.get("micro_env_features")
                    
                    if micro_env_all is not None and pos_idx < len(micro_env_all):
                        micro_env_features = micro_env_all[pos_idx]
                    elif structure_coords is not None:
                        # 按需计算
                        if self._missing_micro_env_warning_count < 5:
                            logger.warning(f"蛋白质 {uniprot_id} 缺少预计算的微环境特征，正在实时计算 (这会降低训练速度)...")
                        elif self._missing_micro_env_warning_count == 5:
                            logger.warning("已有 5 个蛋白质缺失微环境特征，后续警告将被抑制...")
                        self._missing_micro_env_warning_count += 1
                        
                        micro_env_features = self._compute_single_residue_micro_env(structure_coords, pos_idx)

                ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else target_ptm_label
                ptm_type = _sanitize_ptm_type(ptm_type_raw, target_ptm_type, self._ptm_override_cache, "batch_extract_features")

                label_value = row['label'] if 'label' in row else None
                has_label = label_value is not None and not pd.isna(label_value)
                target_label = int(label_value) if has_label else 0

                sample = {
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'residue': row['residue'],
                    'ptm_type': ptm_type,
                    'ptm_type_raw': ptm_type_raw,
                    'target_ptm_type': target_ptm_label,
                    'sequence_features': seq_window,
                    'local_features': seq_local,
                    'global_features': sequence_mean,
                    'structure_features': struct_window,
                    'structure_local': struct_local,
                    'structure_global': structure_mean,
                    'structure_cls_vector': structure_cls,
                    'micro_env_features': micro_env_features,
                    'is_target_ptm': target_label,
                    'is_phosphorylated': target_label,
                    'has_label': has_label,
                    'label': target_label if has_label else None,
                    'sequence_length': seq_len
                }

                processed_samples.append(sample)

                # 进度报告
                if len(processed_samples) % 1000 == 0:
                    logger.info(f"已处理 {len(processed_samples)}/{len(data)} 个样本")

            except Exception as e:
                logger.error(f"处理样本 {uniprot_id}:{position} 时出错: {e}")
                continue
        
        process_time = time.time() - start_time
        logger.info(f"批量特征提取完成，耗时 {process_time:.2f} 秒，处理了 {len(processed_samples)} 个样本")
        
        return processed_samples
    
    @staticmethod
    def _extract_window_features(features: np.ndarray, center_pos: int,
                                window_size: int, half_window: int) -> np.ndarray:
        """提取窗口特征，避免反复堆叠造成的临时数组"""
        seq_len, feature_dim = features.shape

        start = center_pos - half_window
        end = start + window_size

        src_start = max(start, 0)
        src_end = min(end, seq_len)
        left_pad = max(0, -start)
        right_pad = max(0, end - seq_len)

        window = np.zeros((window_size, feature_dim), dtype=features.dtype)

        if src_start < src_end:
            dest_start = left_pad
            valid_length = src_end - src_start
            dest_end = dest_start + valid_length
            # 直接分块赋值，避免多次拷贝
            window[dest_start:dest_end] = features[src_start:src_end]

        return window


class LazyProteinFeatureStore:
    """按需访问蛋白质特征的轻量存储，兼容多进程加载。"""

    def __init__(
        self,
        features_path: str,
        *,
        lazy_loading: bool = True,
        h5_cache_bytes: int = 512 * 1024 * 1024,
        h5_cache_slots: int = 1_048_575,
        h5_cache_w0: float = 0.75,
        known_proteins: Optional[Iterable[str]] = None,
        structure_map: Optional[Dict[str, bool]] = None,
        sequence_lengths: Optional[Dict[str, int]] = None,
        max_lazy_cache_entries: Optional[int] = None,
        fasta_dir: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        micro_env_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.features_path = str(features_path)
        self.lazy_loading = bool(lazy_loading)
        self._h5_cache_bytes = int(h5_cache_bytes)
        self._h5_cache_slots = int(h5_cache_slots)
        self._h5_cache_w0 = float(h5_cache_w0)
        self._known_proteins = set(known_proteins) if known_proteins is not None else None
        self._structure_map = dict(structure_map) if structure_map else {}
        self._sequence_lengths = dict(sequence_lengths) if sequence_lengths else {}
        self.fasta_dir = fasta_dir
        self.pdb_dir = pdb_dir
        self._micro_env_cache_external = micro_env_cache or {}
        self._h5_file: Optional[h5py.File] = None
        self._proteins_group: Optional[h5py.Group] = None
        if max_lazy_cache_entries is None:
            max_lazy_cache_entries = _parse_env_int("PTM_LAZY_CACHE_MAX_ENTRIES", 512, minimum=32)
        self._max_lazy_cache_entries: Optional[int] = int(max_lazy_cache_entries) if max_lazy_cache_entries else None
        self._lazy_cache: "OrderedDict[str, LazyProteinEntry]" = OrderedDict()
        self._eager_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _get_h5_file(self, mode: str = "r") -> h5py.File:
        # Check if we need to reopen (either file is None, or mode mismatch)
        need_reopen = self._h5_file is None
        if not need_reopen:
            # If current mode is 'r' but we need 'r+', we must reopen
            if mode == "r+" and self._h5_file.mode == "r":
                need_reopen = True
        
        if need_reopen:
            if self._h5_file is not None:
                try:
                    # CRITICAL: Clear stale LazyProteinEntry references when closing file
                    for entry in self._lazy_cache.values():
                        try:
                            if isinstance(entry, LazyProteinEntry):
                                entry.close()
                        except Exception:
                            pass
                    self._lazy_cache.clear()

                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                # CRITICAL: Invalidate the group cache when file is closed
                self._proteins_group = None
            
            try:
                self._h5_file = h5py.File(
                    self.features_path,
                    mode,
                    rdcc_nbytes=self._h5_cache_bytes,
                    rdcc_nslots=self._h5_cache_slots,
                    rdcc_w0=self._h5_cache_w0,
                )
            except Exception as exc:  # pragma: no cover - 硬件/IO异常罕见
                # Ensure state is clean on failure
                self._h5_file = None
                self._proteins_group = None
                raise RuntimeError(f"无法打开特征文件 {self.features_path} (mode={mode}): {exc}") from exc
        return self._h5_file

    def _get_proteins_group(self, mode: str = "r") -> "h5py.Group":
        # Check conditions to refresh group:
        # 1. Group is None
        # 2. File is None (should imply 1, but safety first)
        # 3. Mode upgrade needed (r -> r+)
        refresh = (self._proteins_group is None) or (self._h5_file is None)
        
        if not refresh and mode == "r+" and self._h5_file is not None:
             if hasattr(self._h5_file, 'mode') and self._h5_file.mode == "r":
                 refresh = True

        if refresh:
            h5_file = self._get_h5_file(mode)
            if "proteins" not in h5_file:
                if mode == "r+":
                    self._proteins_group = h5_file.create_group("proteins")
                else:
                    raise KeyError(f"特征文件 {self.features_path} 中缺少 'proteins' 分组")
            else:
                self._proteins_group = h5_file["proteins"]
        return self._proteins_group

    def _get_protein_group(self, protein_id: str) -> Optional["h5py.Group"]:
        group = self._get_proteins_group()
        if protein_id not in group:
            return None
        return group[protein_id]

    def _prune_lazy_cache(self) -> None:
        if self._max_lazy_cache_entries is None:
            return
        while len(self._lazy_cache) > self._max_lazy_cache_entries:
            old_pid, old_entry = self._lazy_cache.popitem(last=False)
            try:
                if isinstance(old_entry, LazyProteinEntry):
                    old_entry.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("回收懒加载蛋白质 %s 时遇到异常: %s", old_pid, exc)

    def has_protein(self, protein_id: str) -> bool:
        if self._known_proteins is not None:
            return protein_id in self._known_proteins
        group = self._get_proteins_group()
        return protein_id in group

    def register_known_proteins(self, protein_ids: Iterable[str]) -> None:
        if protein_ids is None:
            return
        if self._known_proteins is None:
            self._known_proteins = set()
        self._known_proteins.update(str(pid) for pid in protein_ids)

    def has_structure(self, protein_id: str) -> bool:
        if protein_id in self._structure_map:
            return bool(self._structure_map[protein_id])
        protein_group = self._get_protein_group(protein_id)
        if protein_group is None:
            self._structure_map[protein_id] = False
            return False
        has_struct = "structure_features" in protein_group
        self._structure_map[protein_id] = has_struct
        return has_struct

    def get_sequence_length(self, protein_id: str) -> Optional[int]:
        if protein_id in self._sequence_lengths:
            return self._sequence_lengths[protein_id]
        protein_group = self._get_protein_group(protein_id)
        if protein_group is None:
            return None
        seq_dataset = protein_group.get("sequence_features")
        if seq_dataset is None:
            return None
        seq_len = int(seq_dataset.shape[0])
        self._sequence_lengths[protein_id] = seq_len
        return seq_len

    def get_entry(self, protein_id: str) -> Optional[Any]:
        if self.lazy_loading:
            entry = self._lazy_cache.get(protein_id)
            if entry is not None:
                self._lazy_cache.move_to_end(protein_id)
                return entry

            with self._lock:
                entry = self._lazy_cache.get(protein_id)
                if entry is None:
                    protein_group = self._get_protein_group(protein_id)
                    if protein_group is None:
                        return None
                
                    if "sequence_features" not in protein_group:
                        return None
                    
                    entry = LazyProteinEntry(protein_id, protein_group, fasta_dir=self.fasta_dir, pdb_dir=self.pdb_dir)
                    
                    # 注入预计算的微环境特征
                    if protein_id in self._micro_env_cache_external:
                        entry.micro_env_features = self._micro_env_cache_external[protein_id]
                        
                    self._lazy_cache[protein_id] = entry
                    self._lazy_cache.move_to_end(protein_id)
                    self._prune_lazy_cache()
                else:
                    self._lazy_cache.move_to_end(protein_id)
                return entry

        entry = self._eager_cache.get(protein_id)
        if entry is not None:
            return entry

        with self._lock:
            entry = self._eager_cache.get(protein_id)
            if entry is None:
                protein_group = self._get_protein_group(protein_id)
                if protein_group is None:
                    return None
                
                if "sequence_features" not in protein_group:
                    return None
                    
                entry = {
                    "sequence_features": protein_group["sequence_features"][()],
                    "sequence_cls": protein_group["sequence_cls"][()],
                    "sequence_mean": protein_group["sequence_mean"][()],
                }
                
                # Try to load sequence string
                try:
                    seq_bytes = protein_group.attrs.get("sequence")
                    if isinstance(seq_bytes, bytes):
                        entry["sequence_str"] = seq_bytes.decode('utf-8')
                    else:
                        entry["sequence_str"] = str(seq_bytes) if seq_bytes is not None else None
                except Exception:
                    entry["sequence_str"] = None

                if not entry["sequence_str"] and self.fasta_dir:
                    entry["sequence_str"] = load_sequence_from_fasta(protein_id, self.fasta_dir)

                if "structure_features" in protein_group:
                    entry.update(
                        {
                            "structure_features": protein_group["structure_features"][()],
                            "structure_cls": protein_group["structure_cls"][()],
                            "structure_mean": protein_group["structure_mean"][()],
                        }
                    )
                    self._structure_map.setdefault(protein_id, True)
                else:
                    entry.update(
                        {
                            "structure_features": None,
                            "structure_cls": None,
                            "structure_mean": None,
                        }
                    )
                    self._structure_map.setdefault(protein_id, False)
                
                # 注入预计算的微环境特征
                if protein_id in self._micro_env_cache_external:
                    entry["micro_env_features"] = self._micro_env_cache_external[protein_id]
                else:
                    entry["micro_env_features"] = None
            
        self._eager_cache[protein_id] = entry
        return entry

    def save_micro_environment_to_h5(self, micro_env_dict: Dict[str, np.ndarray]) -> None:
        """将预计算的微环境特征保存回 HDF5 文件中。"""
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        return
        
        if not micro_env_dict:
            return
            
        try:
            # 使用 r+ 模式获取组
            proteins_group = self._get_proteins_group("r+")
            count = 0
            missing_pids = []
            for pid, features in micro_env_dict.items():
                if pid in proteins_group:
                    pg = proteins_group[pid]
                    # 如果 pg 是数据集而不是组，说明 HDF5 结构有问题
                    if not isinstance(pg, h5py.Group):
                        logger.warning(f"HDF5 中蛋白质 {pid} 不是一个组，跳过写入")
                        continue
                        
                    if "micro_env_features" in pg:
                        del pg["micro_env_features"]
                    pg.create_dataset(
                        "micro_env_features", 
                        data=features, 
                        compression="gzip", 
                        compression_opts=4
                    )
                    count += 1
                else:
                    # 如果蛋白质在 HDF5 中完全不存在，不创建组
                    missing_pids.append(pid)
            
            if count > 0:
                logger.debug(f"成功更新/创建了 {count} 个蛋白质的微环境特征到 HDF5")
            if missing_pids:
                logger.debug(f"有 {len(missing_pids)} 个蛋白质在 HDF5 中不存在且创建失败")
        except Exception as e:
            logger.error(f"保存微环境特征到 HDF5 失败: {e}")

    def save_micro_environment_features(self, micro_env_dict: Dict[str, np.ndarray], cache_dir: str = None) -> None:
        """
        保存微环境特征到 LMDB (如果可用) 和 HDF5。
        """
        if not micro_env_dict:
            return

        # 1. Update in-memory cache
        for pid, feat in micro_env_dict.items():
             self.update_micro_env_cache(pid, feat)

        # 2. Save to HDF5 (Persistent storage)
        self.save_micro_environment_to_h5(micro_env_dict)

    def update_micro_env_cache(self, protein_id: str, features: np.ndarray) -> None:
        """Update the in-memory micro-environment cache."""
        if features is None:
            return
        self._micro_env_cache_external[protein_id] = features
        
        # Try to update active lazy entry if exists
        if self.lazy_loading:
            if protein_id in self._lazy_cache:
                entry = self._lazy_cache[protein_id]
                if isinstance(entry, LazyProteinEntry):
                    entry.micro_env_features = features
        else:
            if protein_id in self._eager_cache:
                entry = self._eager_cache[protein_id]
                if isinstance(entry, dict):
                    entry["micro_env_features"] = features

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception as exc:  # pragma: no cover - IO异常罕见
                logger.warning("关闭特征文件时出现异常: %s", exc)
            finally:
                self._h5_file = None
                self._proteins_group = None
                for entry in self._lazy_cache.values():
                    try:
                        if isinstance(entry, LazyProteinEntry):
                            entry.close()
                    except Exception as exc:  # pragma: no cover - defensive cleanup
                        logger.debug("释放懒加载缓存时出现异常: %s", exc)
                self._lazy_cache.clear()
                self._eager_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - 析构器不可预测
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state['_h5_file'] = None
        state['_proteins_group'] = None
        state['_lazy_cache'] = OrderedDict()
        state['_lock'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lazy_cache = OrderedDict()
        self._lock = threading.Lock()
        atexit.register(self.close)


class OnDemandPTMDataset(Dataset):
    """按需从HDF5特征文件提取窗口的PTM数据集。"""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        protein_store: LazyProteinFeatureStore,
        *,
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = "phosphorylation",
        residue_info: Optional[Dict[str, Any]] = None,
        feature_reducer: Optional[Any] = None,
        use_structure: bool = True,
    ) -> None:
        self.data_frame = data_frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.fixed_window_size = int(window_size)
        self.fixed_local_size = int(local_window_size)
        self.center_index = self.fixed_window_size // 2
        self.half_window = self.fixed_window_size // 2
        self.local_half_window = self.fixed_local_size // 2
        self.target_ptm_label = str(target_ptm_type).strip()
        self.target_ptm_type = self.target_ptm_label.lower()
        self._ptm_override_cache: Set[str] = set()
        self._normalization_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.feature_reducer = feature_reducer
        self.use_structure = use_structure

        if residue_info is None:
            observed_residues = {
                str(res).strip().upper()
                for res in self.data_frame.get("residue", [])
                if isinstance(res, str) and res.strip()
            }
            residue_info = resolve_ptm_residue_info(self.target_ptm_type, observed_residues)

        self.residue_info = residue_info
        self.residue_to_id = dict(residue_info['residue_to_id'])
        self.other_residue_id = int(residue_info['other_id'])
        self.residue_vocab_size = int(residue_info['vocab_size'])
        self.ptm_type_to_id = {self.target_ptm_label: 0}

    def _compute_micro_environment_features(self, coords: np.ndarray, center_idx: int) -> np.ndarray:
        """计算微环境特征 (密度+凹凸性+平均距离)"""
        if coords is None or len(coords) == 0:
            return np.zeros(3, dtype=np.float32)
        
        if center_idx < 0 or center_idx >= len(coords):
             return np.zeros(3, dtype=np.float32)

        center_coord = coords[center_idx]
        # 计算所有点到中心点的距离
        dists = np.linalg.norm(coords - center_coord, axis=1)
        
        # 1. 局部密度 (10A内的邻居数)
        radius = 10.0
        mask = (dists < radius) & (dists > 1e-6)
        neighbor_indices = np.where(mask)[0]
        local_density = len(neighbor_indices)
        
        # 2. 局部凹凸性 (到邻居质心的距离)
        if local_density > 0:
            neighbor_coords = coords[neighbor_indices]
            centroid = np.mean(neighbor_coords, axis=0)
            dist_to_centroid = np.linalg.norm(center_coord - centroid)
            
            # 3. 邻居平均距离
            avg_dist = np.mean(dists[neighbor_indices])
        else:
            dist_to_centroid = 0.0
            avg_dist = 0.0
            
        # 归一化 (参考 encoders.py 的归一化系数)
        return np.array([
            local_density / 20.0,
            dist_to_centroid / 5.0,
            avg_dist / 10.0
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._build_raw_sample(idx)
        
        if self.feature_reducer:
            self.feature_reducer.transform([sample], inplace=True)
            
        if self._normalization_stats:
            self._apply_normalization(sample)
        return self._to_torch(sample)

    def set_normalization_stats(self, stats: Optional[Dict[str, Dict[str, np.ndarray]]]) -> None:
        if not stats:
            self._normalization_stats = {}
            return

        normalized_stats: Dict[str, Dict[str, np.ndarray]] = {}
        for key, value in stats.items():
            mean = np.array(value.get("mean"), dtype=np.float32, copy=True)
            std = np.array(value.get("std"), dtype=np.float32, copy=True)
            normalized_stats[key] = {"mean": mean, "std": std}
        self._normalization_stats = normalized_stats

    def iter_raw_samples(self) -> Iterator[Dict[str, Any]]:
        """迭代器：逐个生成原始样本，避免全部加载到内存。
        
        警告：对大数据集不要使用 list(iter_raw_samples())，这会导致内存爆炸。
        应该使用流式处理或按批次处理。
        """
        for idx in range(len(self.data_frame)):
            yield self._build_raw_sample(idx)
    
    def iter_raw_samples_batch(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """批量迭代器：按批次生成原始样本，平衡内存和效率。
        
        Args:
            batch_size: 每批次的样本数量
            
        Yields:
            List[Dict]: 一批原始样本
        """
        total = len(self.data_frame)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = []
            for idx in range(start_idx, end_idx):
                try:
                    sample = self._build_raw_sample(idx)
                    batch.append(sample)
                except Exception as e:
                    logger.warning(f"跳过样本 {idx}: {e}")
                    continue
            if batch:
                yield batch

    def _unpack_entry(self, entry: Any, uniprot_id: str):
        if isinstance(entry, LazyProteinEntry):
            seq_dataset = entry.sequence_dataset
            if seq_dataset is None:
                raise RuntimeError(f"蛋白质 {uniprot_id} 的序列特征不可用 (懒加载条目已失效)")
            seq_len = entry.sequence_length
            structure_dataset = entry.structure_dataset

            struct_mean_raw = entry.structure_mean
            structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32).copy()

            struct_cls_raw = entry.structure_cls
            structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32).copy()

            structure_coords = entry.structure_coords

            seq_mean_raw = entry.sequence_mean
            if seq_mean_raw is None:
                feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                if feature_dim == 0:
                    logger.warning("蛋白质 %s 缺少 sequence_mean 且无法推断维度，使用空向量", uniprot_id)
                    sequence_mean = np.zeros((0,), dtype=np.float32)
                else:
                    logger.debug("蛋白质 %s 缺少序列全局特征，使用零向量", uniprot_id)
                    sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
            else:
                sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32).copy()
            sequence_str = entry.sequence
        else:
            seq_dataset = entry['sequence_features']
            seq_len = seq_dataset.shape[0]
            structure_dataset = entry.get('structure_features')
            structure_mean = entry.get('structure_mean')
            if structure_mean is not None:
                structure_mean = np.array(structure_mean, copy=True)
            structure_cls = entry.get('structure_cls')
            if structure_cls is not None:
                structure_cls = np.array(structure_cls, copy=True)
            structure_coords = entry.get('structure_coords')
            sequence_mean = np.array(entry['sequence_mean'], copy=True)
            sequence_str = entry.get('sequence_str')
            
        return seq_dataset, seq_len, structure_dataset, structure_mean, structure_cls, structure_coords, sequence_mean, sequence_str

    def _build_raw_sample(self, idx: int) -> Dict[str, Any]:
        row = self.data_frame.iloc[idx]
        uniprot_id = row['uniprot_id']
        if not isinstance(uniprot_id, str):
            uniprot_id = str(uniprot_id)

        entry = self.protein_store.get_entry(uniprot_id)
        if entry is None:
            raise KeyError(f"蛋白质 {uniprot_id} 未在特征存储中注册")

        # Initial unpack
        (seq_dataset, seq_len, structure_dataset, structure_mean, 
         structure_cls, structure_coords, sequence_mean, sequence_str) = self._unpack_entry(entry, uniprot_id)

        position_value = row['position']
        try:
            position = int(position_value)
        except (TypeError, ValueError):
            raise ValueError(f"样本 {uniprot_id} 的 position='{position_value}' 无法解析为整数")

        pos_idx = position - 1
        if pos_idx < 0 or pos_idx >= seq_len:
            raise IndexError(f"位置 {position} 超出蛋白质 {uniprot_id} 序列范围 ({seq_len})")

        # Calculate micro-environment features
        micro_env_features = None
        if isinstance(entry, LazyProteinEntry):
            if entry.micro_env_features is not None:
                all_micro_env = entry.micro_env_features
                if hasattr(all_micro_env, '__len__') and pos_idx < len(all_micro_env):
                    micro_env_features = all_micro_env[pos_idx]
        elif isinstance(entry, dict):
            # Eager mode
            all_micro_env = entry.get('micro_env_features')
            if all_micro_env is not None and pos_idx < len(all_micro_env):
                micro_env_features = all_micro_env[pos_idx]
        
        if micro_env_features is None:
            # Fallback logic
            reloaded = False
            
            if self.protein_store.pdb_dir:
                try:
                    _, full_features = _compute_single_protein_micro_env(uniprot_id, self.protein_store.pdb_dir)
                    if full_features is not None:
                        self.protein_store.save_micro_environment_features({uniprot_id: full_features})
                        if pos_idx < len(full_features):
                            micro_env_features = full_features[pos_idx]
                        reloaded = True
                except Exception as e:
                    logger.warning(f"Failed to compute full micro-env features for {uniprot_id}: {e}")

            if micro_env_features is None:
                if structure_coords is not None and len(structure_coords) > 0:
                    try:
                        full_features = _compute_micro_env_from_coords(structure_coords)
                        if full_features is not None:
                            self.protein_store.save_micro_environment_features({uniprot_id: full_features})
                            if pos_idx < len(full_features):
                                micro_env_features = full_features[pos_idx]
                            reloaded = True
                    except Exception as e:
                         logger.debug(f"Failed to compute full micro-env features from coords for {uniprot_id}: {e}")

            if micro_env_features is None:
                 logger.debug(f"Protein {uniprot_id} missing micro-env features, computing on-demand (slow)...")
                 micro_env_features = self._compute_micro_environment_features(structure_coords, pos_idx)
            
            if reloaded:
                # Reload entry and datasets because file handle might have changed
                entry = self.protein_store.get_entry(uniprot_id)
                if entry is None:
                     raise KeyError(f"蛋白质 {uniprot_id} 在重新加载后丢失")
                (seq_dataset, seq_len, structure_dataset, structure_mean, 
                 structure_cls, structure_coords, sequence_mean, sequence_str) = self._unpack_entry(entry, uniprot_id)

        seq_window = DataLoadingOptimizer._extract_window_features(seq_dataset, pos_idx, self.fixed_window_size, self.half_window)
        seq_local = DataLoadingOptimizer._extract_window_features(seq_dataset, pos_idx, self.fixed_local_size, self.local_half_window)

        struct_window = None
        struct_local = None
        if self.use_structure and structure_dataset is not None:
            struct_window = DataLoadingOptimizer._extract_window_features(structure_dataset, pos_idx, self.fixed_window_size, self.half_window)
            struct_local = DataLoadingOptimizer._extract_window_features(structure_dataset, pos_idx, self.fixed_local_size, self.local_half_window)

        ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else self.target_ptm_label
        if pd.isna(ptm_type_raw):
            ptm_type_raw = self.target_ptm_label
        ptm_type = _sanitize_ptm_type(ptm_type_raw, self.target_ptm_label, self._ptm_override_cache, "OnDemandPTMDataset")

        label_raw = row['label'] if 'label' in row else None
        has_label = label_raw is not None and not pd.isna(label_raw)
        if has_label:
            try:
                target_label = int(label_raw)
            except (TypeError, ValueError):
                target_label = int(float(label_raw))
        else:
            target_label = 0

        residue_value = row['residue'] if 'residue' in row else None

        return {
            'uniprot_id': uniprot_id,
            'position': position,
            'residue': residue_value,
            'ptm_type': ptm_type,
            'ptm_type_raw': ptm_type_raw,
            'target_ptm_type': self.target_ptm_label,
            'sequence_features': np.array(seq_window, copy=False),
            'local_features': np.array(seq_local, copy=False),
            'global_features': np.array(sequence_mean, copy=True),
            'structure_features': None if struct_window is None else np.array(struct_window, copy=False),
            'structure_local': None if struct_local is None else np.array(struct_local, copy=False),
            'structure_global': None if structure_mean is None else np.array(structure_mean, copy=True),
            'structure_cls_vector': None if structure_cls is None else np.array(structure_cls, copy=True),
            'micro_env_features': micro_env_features,
            'is_target_ptm': target_label,
            'is_phosphorylated': target_label,
            'has_label': has_label,
            'label': target_label if has_label else None,
            'sequence_length': seq_len,
            'sequence': sequence_str,
        }

    def _apply_normalization(self, sample: Dict[str, Any]) -> None:
        for key, stat in self._normalization_stats.items():
            value = sample.get(key)
            if value is None:
                continue
            array = np.array(value, dtype=np.float32, copy=True)
            mean = stat['mean']
            std = stat['std']
            normalized = (array - mean) / std
            sample[key] = normalized.astype(np.float32, copy=False)

    def _to_torch(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        seq_features = torch.from_numpy(np.asarray(sample['sequence_features'], dtype=np.float32))
        local_features = torch.from_numpy(np.asarray(sample['local_features'], dtype=np.float32))
        global_features = torch.from_numpy(np.asarray(sample['global_features'], dtype=np.float32))
        
        micro_env = sample.get('micro_env_features')
        if micro_env is not None:
             micro_env_arr = np.asarray(micro_env, dtype=np.float32)
             # Ensure dimension is 3
             if micro_env_arr.shape[0] > 3:
                 micro_env_arr = micro_env_arr[:3]
             elif micro_env_arr.shape[0] < 3:
                 micro_env_arr = np.pad(micro_env_arr, (0, 3 - micro_env_arr.shape[0]), 'constant')
             micro_env_tensor = torch.from_numpy(micro_env_arr)
        else:
             micro_env_tensor = torch.zeros(3, dtype=torch.float32)

        struct_tensor = None
        if sample['structure_features'] is not None:
            struct_tensor = torch.from_numpy(np.asarray(sample['structure_features'], dtype=np.float32))

        residue_value = sample.get('residue')
        if residue_value is None or (isinstance(residue_value, float) and np.isnan(residue_value)):
            residue_key = None
        else:
            residue_key = str(residue_value).strip().upper()
        residue_type_id = self.residue_to_id.get(residue_key, self.other_residue_id)

        residue_types = torch.full((self.fixed_window_size,), self.other_residue_id, dtype=torch.long)
        
        sequence_str = sample.get('sequence')
        position = int(sample['position'])
        pos_idx = position - 1
        
        if sequence_str:
            seq_len = len(sequence_str)
            start = pos_idx - self.half_window
            end = pos_idx + self.half_window + 1
            
            for i, seq_i in enumerate(range(start, end)):
                if 0 <= seq_i < seq_len:
                    res = sequence_str[seq_i].upper()
                    residue_types[i] = self.residue_to_id.get(res, self.other_residue_id)
        else:
            residue_types[self.center_index] = residue_type_id

        position_ids = torch.arange(self.fixed_window_size, dtype=torch.long) - self.center_index
        relative_offsets = position_ids.clone()

        site_indicator = torch.zeros(self.fixed_window_size, dtype=torch.float32)
        site_indicator[self.center_index] = 1.0

        label_tensor = torch.tensor(float(sample['is_target_ptm']), dtype=torch.float32)

        ptm_type_value = sample['ptm_type']
        ptm_type_id = self.ptm_type_to_id.get(ptm_type_value, 0)

        return {
            'uniprot_id': sample['uniprot_id'],
            'position': sample['position'],
            'residue': residue_value,
            'residue_type_id': residue_type_id,
            'residue_types': residue_types,
            'position_ids': position_ids,
            'relative_position_offsets': relative_offsets,
            'site_indicator': site_indicator,
            'sequence': {
                'window_features': seq_features,
                'local_features': local_features,
                'global_features': global_features,
                'position_ids': position_ids,
                'relative_position_offsets': relative_offsets,
                'site_indicator': site_indicator,
                'residue_types': residue_types,
            },
            'structure': struct_tensor,
            'micro_env_features': micro_env_tensor,
            'label': label_tensor,
            'ptm_type': ptm_type_value,
            'ptm_type_id': ptm_type_id,
        }


def _compute_micro_env_from_coords(coords: np.ndarray) -> np.ndarray:
    """从坐标数组计算微环境特征 (无IO操作)。"""
    seq_len = len(coords)
    features = np.zeros((seq_len, 3), dtype=np.float32)
    
    # 向量化计算所有残基的距离
    for i in range(seq_len):
        center_coord = coords[i]
        dists = np.linalg.norm(coords - center_coord, axis=1)
        
        # 1. 局部密度 (10A内的邻居数)
        radius = 10.0
        mask = (dists < radius) & (dists > 1e-6)
        neighbor_indices = np.where(mask)[0]
        local_density = len(neighbor_indices)
        
        if local_density > 0:
            neighbor_coords = coords[neighbor_indices]
            centroid = np.mean(neighbor_coords, axis=0)
            dist_to_centroid = np.linalg.norm(center_coord - centroid)
            avg_dist = np.mean(dists[neighbor_indices])
        else:
            dist_to_centroid = 0.0
            avg_dist = 0.0
            
        features[i] = [
            local_density / 20.0,
            dist_to_centroid / 5.0,
            avg_dist / 10.0
        ]
    return features

def _compute_single_protein_micro_env(protein_id: str, pdb_dir: str) -> Tuple[str, Optional[np.ndarray]]:
    """为单个蛋白质计算所有残基的微环境特征。"""
    pdb_path = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if not os.path.exists(pdb_path):
        return protein_id, None
        
    coords = _extract_ca_coords_fast(pdb_path)
    if coords is None:
        return protein_id, None
        
    features = _compute_micro_env_from_coords(coords)
    return protein_id, features


def optimize_data_loading(
    train_data_path: str,
    test_data_path: str,
    esm_features_path: str,
    cache_dir: str = None,
    num_workers: int = None,
    target_ptm_type: str = 'phosphorylation',
    window_size: int = 61,
    local_window_size: int = 31,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    """
    优化的数据加载函数
    
    Args:
        train_data_path: 训练数据路径
        test_data_path: 测试数据路径
        esm_features_path: ESM特征文件路径
        cache_dir: 缓存目录
        num_workers: 工作进程数
        target_ptm_type: 目标PTM类型
        window_size: 序列窗口大小
        local_window_size: 局部窗口大小
        
    Returns:
        Tuple: (训练数据, 测试数据, 残基映射信息)
    """
    logger.info("开始优化数据加载...")

    # 创建优化器
    optimizer = DataLoadingOptimizer(esm_features_path, num_workers)

    try:
        # 加载数据
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        def _clean_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
            raw_count = len(df)
            df = df.copy()
            df['uniprot_id'] = df['uniprot_id'].apply(_normalize_uniprot_id)
            invalid_mask = df['uniprot_id'].isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                logger.warning(
                    f"{source} 中发现 {invalid_count} 个无效的UniProt ID (约 {invalid_count / max(raw_count, 1) * 100:.2f}% )，将被忽略"
                )
                df = df.loc[~invalid_mask].copy()
            coerced_count = int((df['uniprot_id'].apply(lambda v: not isinstance(v, str))).sum())
            if coerced_count > 0:
                # 理论上不会发生，但为安全起见再转换一次
                df['uniprot_id'] = df['uniprot_id'].astype(str)
                logger.debug(f"{source} 中有 {coerced_count} 个UniProt ID 被强制转换为字符串")
            return df

        train_data = _clean_dataframe(train_data, "训练数据")
        test_data = _clean_dataframe(test_data, "测试数据")

        logger.info(f"加载训练数据: {len(train_data)} 个样本")
        logger.info(f"加载测试数据: {len(test_data)} 个样本")

        # 收集所有需要的蛋白质ID
        all_protein_ids = set(train_data['uniprot_id'].unique()) | set(test_data['uniprot_id'].unique())
        logger.info(f"需要加载 {len(all_protein_ids)} 个不同的蛋白质特征")

        train_residue_series = train_data['residue'].dropna().astype(str).str.strip()
        observed_residues = {res for res in train_residue_series if res}
        residue_info = resolve_ptm_residue_info(target_ptm_type, observed_residues)
        logger.info(
            "优化加载使用的残基映射仅基于训练集: %s",
            residue_info['residue_to_id'],
        )

        test_residue_series = test_data['residue'].dropna().astype(str).str.strip()
        unseen_test_residues = sorted({res for res in test_residue_series if res} - observed_residues)
        if unseen_test_residues:
            logger.info(
                "测试集中发现训练阶段未出现的残基 %s，将在推理时映射为 Other",
                unseen_test_residues,
            )

        # 预加载蛋白质特征
        protein_features = optimizer.preload_protein_features(list(all_protein_ids), cache_dir)

        # 批量提取特征
        train_processed = optimizer.batch_extract_features(
            train_data,
            protein_features,
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
        )
        test_processed = optimizer.batch_extract_features(
            test_data,
            protein_features,
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
        )

        logger.info("优化数据加载完成")

        return train_processed, test_processed, residue_info
    finally:
        optimizer.close()


class StratifiedBatchSampler(Sampler):
    """
    分层批次采样器
    确保每个批次中正负样本的比例与目标比例一致
    """
    def __init__(self, labels: np.ndarray, batch_size: int, pos_ratio: float, shuffle: bool = True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # 计算每个批次应有的正负样本数
        self.n_pos = int(batch_size * pos_ratio)
        # 确保至少有一个正样本（如果有的话）
        if self.n_pos == 0 and len(self.pos_indices) > 0:
            self.n_pos = 1
        
        self.n_neg = batch_size - self.n_pos
        # 确保至少有一个负样本（如果有的话）
        if self.n_neg == 0 and len(self.neg_indices) > 0:
            self.n_neg = 1
            self.n_pos = batch_size - self.n_neg
            
        # 修复逻辑：基于覆盖多数类的原则计算 num_batches
        # 旧逻辑: self.num_batches = int(np.ceil(len(labels) / batch_size))
        # 这会导致在为了满足 pos_ratio 而减少负样本比例时，丢失部分负样本数据
        
        batches_needed_pos = 0
        if self.n_pos > 0:
            batches_needed_pos = int(np.ceil(len(self.pos_indices) / self.n_pos))
            
        batches_needed_neg = 0
        if self.n_neg > 0:
            batches_needed_neg = int(np.ceil(len(self.neg_indices) / self.n_neg))
            
        # 取最大值，确保所有样本（无论是正还是负）在每个Epoch至少被覆盖一次
        self.num_batches = max(batches_needed_pos, batches_needed_neg)
        
        # 安全检查: 防止空数据集
        if self.num_batches == 0:
             self.num_batches = int(np.ceil(len(labels) / batch_size)) if len(labels) > 0 else 0

        logger.info(f"初始化分层采样器: 批大小={batch_size}, 目标正样本率={pos_ratio:.4f}")
        logger.info(f"每批次: 正样本={self.n_pos}, 负样本={self.n_neg}")
        logger.info(f"Epoch调整: 将运行 {self.num_batches} 个Batches 以确保覆盖所有样本 (原计算值: {int(np.ceil(len(labels) / batch_size))})")
        
    def __iter__(self):
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        
        if self.shuffle:
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            
        pos_ptr = 0
        neg_ptr = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 采样正样本
            for _ in range(self.n_pos):
                if len(pos_indices) > 0:
                    batch_indices.append(pos_indices[pos_ptr % len(pos_indices)])
                    pos_ptr += 1
                
            # 采样负样本
            for _ in range(self.n_neg):
                if len(neg_indices) > 0:
                    batch_indices.append(neg_indices[neg_ptr % len(neg_indices)])
                    neg_ptr += 1
            
            # 再次打乱批次内部顺序
            if self.shuffle:
                np.random.shuffle(batch_indices)
                
            yield batch_indices

    def __len__(self):
        return self.num_batches


class RotatingBalancedSampler(Sampler):
    """
    DeepMVP 风格的迭代式平衡采样器
    
    特点：
    1. 每个Epoch只使用一部分负样本，确保正负样本数量平衡（1:1 或指定比例）
    2. 在后续的 Epoch 中自动切换负样本子集，实现对所有负样本的覆盖
    3. 避免了在同一个 Epoch 中大量重复正样本，减少过拟合风险
    """
    def __init__(self, labels: np.ndarray, batch_size: int, pos_ratio: float = 0.5, shuffle: bool = True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # 确保正负样本都存在
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("正样本或负样本数量为0，无法进行平衡采样")
            
        # 计算每个批次的正负样本数
        self.n_pos_per_batch = int(batch_size * pos_ratio)
        if self.n_pos_per_batch == 0: self.n_pos_per_batch = 1
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch
        if self.n_neg_per_batch == 0: self.n_neg_per_batch = 1
        
        # 计算一个Epoch应该有多少个Batch
        # 策略：以正样本数量为基准，确保每个Epoch遍历一次所有正样本
        # 负样本则按比例截取
        self.num_batches = int(np.ceil(len(self.pos_indices) / self.n_pos_per_batch))
        
        # 负样本指针，用于在不同Epoch间轮转
        self.neg_ptr = 0
        # 预先打乱负样本，确保随机性
        if self.shuffle:
            np.random.shuffle(self.neg_indices)
            
        logger.info(f"初始化旋转平衡采样器 (DeepMVP Strategy):")
        logger.info(f"  总正样本: {len(self.pos_indices)}, 总负样本: {len(self.neg_indices)}")
        logger.info(f"  每批次: 正样本={self.n_pos_per_batch}, 负样本={self.n_neg_per_batch}")
        logger.info(f"  每Epoch Batch数: {self.num_batches} (覆盖 {len(self.pos_indices)} 正样本, {self.num_batches * self.n_neg_per_batch} 负样本)")
        
    def __iter__(self):
        # 正样本：每个Epoch都使用全部，并打乱
        batch_pos_indices = self.pos_indices.copy()
        if self.shuffle:
            np.random.shuffle(batch_pos_indices)
            
        # 负样本：从上次结束的地方开始截取
        batch_neg_indices = []
        needed_neg = self.num_batches * self.n_neg_per_batch
        
        current_ptr = self.neg_ptr
        for _ in range(needed_neg):
            batch_neg_indices.append(self.neg_indices[current_ptr % len(self.neg_indices)])
            current_ptr += 1
            
        # 更新全局指针，供下个Epoch使用
        self.neg_ptr = current_ptr % len(self.neg_indices)
        
        # 构建批次
        pos_ptr = 0
        neg_ptr_local = 0
        
        for _ in range(self.num_batches):
            batch = []
            
            # 添加正样本 (如果不足则循环)
            for _ in range(self.n_pos_per_batch):
                batch.append(batch_pos_indices[pos_ptr % len(batch_pos_indices)])
                pos_ptr += 1
                
            # 添加负样本
            for _ in range(self.n_neg_per_batch):
                batch.append(batch_neg_indices[neg_ptr_local])
                neg_ptr_local += 1
                
            # 打乱批次内顺序
            if self.shuffle:
                np.random.shuffle(batch)
                
            yield batch
            
    def __len__(self):
        return self.num_batches
