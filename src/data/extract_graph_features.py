import os
import json
import csv
import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import h5py
from Bio.PDB import PDBParser

try:
    from Bio.PDB.DSSP import DSSP
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False

logger = logging.getLogger("extract_graph_features")


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# [hydrophobicity, charge, polarity, aromaticity]
AA_PHYSICOCHEM: Dict[str, Tuple[float, float, float, float]] = {
    "A": (0.62, 0.0, 0.0, 0.0),
    "R": (-2.53, 1.0, 1.0, 0.0),
    "N": (-0.78, 0.0, 1.0, 0.0),
    "D": (-0.90, -1.0, 1.0, 0.0),
    "C": (0.29, 0.0, 0.0, 0.0),
    "Q": (-0.85, 0.0, 1.0, 0.0),
    "E": (-0.74, -1.0, 1.0, 0.0),
    "G": (0.48, 0.0, 0.0, 0.0),
    "H": (-0.40, 0.3, 1.0, 1.0),
    "I": (1.38, 0.0, 0.0, 0.0),
    "L": (1.06, 0.0, 0.0, 0.0),
    "K": (-1.50, 1.0, 1.0, 0.0),
    "M": (0.64, 0.0, 0.0, 0.0),
    "F": (1.19, 0.0, 0.0, 1.0),
    "P": (0.12, 0.0, 0.0, 0.0),
    "S": (-0.18, 0.0, 1.0, 0.0),
    "T": (-0.05, 0.0, 1.0, 0.0),
    "W": (0.81, 0.0, 0.0, 1.0),
    "Y": (0.26, 0.0, 1.0, 1.0),
    "V": (1.08, 0.0, 0.0, 0.0),
}


def _aa1_from_resname(resname: str) -> str:
    return AA3_TO_1.get(str(resname).strip().upper(), "X")


def _aa_physicochem(aa1: str) -> np.ndarray:
    values = AA_PHYSICOCHEM.get(aa1.upper(), (0.0, 0.0, 0.0, 0.0))
    # Keep a bounded range for stability.
    hydro = float(np.clip(values[0] / 2.5, -1.0, 1.0))
    charge = float(np.clip(values[1], -1.0, 1.0))
    polarity = float(np.clip(values[2], 0.0, 1.0))
    aromatic = float(np.clip(values[3], 0.0, 1.0))
    return np.asarray([hydro, charge, polarity, aromatic], dtype=np.float32)


class PDBPathResolver:
    def __init__(self, pdb_dir: Path) -> None:
        self.pdb_dir = pdb_dir
        self._cache: Dict[str, Optional[Path]] = {}
        self._exact: Dict[str, Path] = {}
        self._prefix: Dict[str, Path] = {}
        self._file_list: List[Tuple[str, Path]] = []

        if not pdb_dir.exists():
            return

        for ext in ("*.pdb", "*.ent"):
            for path in sorted(pdb_dir.glob(ext)):
                stem_l = path.stem.lower()
                self._file_list.append((stem_l, path))
                self._exact.setdefault(stem_l, path)
                token = stem_l.split("_")[0].split("-")[0]
                self._prefix.setdefault(token, path)

    def resolve(self, uniprot_id: str) -> Optional[Path]:
        uid = str(uniprot_id).strip()
        if not uid:
            return None
        key = uid.lower()
        if key in self._cache:
            return self._cache[key]

        for candidate in (key, uid, uid.lower()):
            found = self._exact.get(str(candidate).lower())
            if found is not None:
                self._cache[key] = found
                return found

        token_found = self._prefix.get(key)
        if token_found is not None:
            self._cache[key] = token_found
            return token_found

        for stem, path in self._file_list:
            if stem.startswith(key):
                self._cache[key] = path
                return path

        self._cache[key] = None
        return None


class StructureEmbeddingStore:
    def __init__(self, h5_path: Optional[Path]) -> None:
        self.h5_path = h5_path if h5_path and h5_path.exists() else None
        self._h5: Optional[h5py.File] = None
        self._dim: Optional[int] = None

    def _open(self) -> Optional[h5py.File]:
        if self.h5_path is None:
            return None
        if self._h5 is None:
            self._h5 = h5py.File(str(self.h5_path), "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
        self._h5 = None

    def infer_dim(self) -> int:
        if self._dim is not None:
            return int(self._dim)
        h5 = self._open()
        if h5 is None or "proteins" not in h5:
            self._dim = 0
            return 0
        proteins = h5["proteins"]
        for uid in proteins.keys():
            grp = proteins[uid]
            emb_key = "structure_features" if "structure_features" in grp else (
                "sequence_features" if "sequence_features" in grp else None)
            if emb_key is not None:
                ds = grp[emb_key]
                if len(ds.shape) == 2 and ds.shape[1] > 0:
                    self._dim = int(ds.shape[1])
                    return int(self._dim)
            if "structure_cls" in grp:
                ds = grp["structure_cls"]
                if len(ds.shape) == 1 and ds.shape[0] > 0:
                    self._dim = int(ds.shape[0])
                    return int(self._dim)
        self._dim = 0
        return 0

    @staticmethod
    def _get_group(proteins: h5py.Group, uid: str) -> Optional[h5py.Group]:
        if uid in proteins:
            return proteins[uid]
        up = uid.upper()
        lo = uid.lower()
        if up in proteins:
            return proteins[up]
        if lo in proteins:
            return proteins[lo]
        return None

    def get_embedding(self, uniprot_id: str, residue_pos_1based: int, center_pos_1based: int, out_dim: int) -> np.ndarray:
        if out_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        h5 = self._open()
        if h5 is None or "proteins" not in h5:
            return np.zeros((out_dim,), dtype=np.float32)
        proteins = h5["proteins"]
        grp = self._get_group(proteins, str(uniprot_id).strip())
        if grp is None:
            return np.zeros((out_dim,), dtype=np.float32)

        emb = None
        emb_key = "structure_features" if "structure_features" in grp else (
            "sequence_features" if "sequence_features" in grp else None)
        if emb_key is not None:
            mat = grp[emb_key]
            if len(mat.shape) == 2 and mat.shape[1] > 0:
                idx = int(residue_pos_1based) - 1
                center_idx = int(center_pos_1based) - 1
                if 0 <= idx < mat.shape[0]:
                    emb = np.asarray(mat[idx], dtype=np.float32)
                elif 0 <= center_idx < mat.shape[0]:
                    emb = np.asarray(mat[center_idx], dtype=np.float32)

        if emb is None and "structure_cls" in grp:
            emb = np.asarray(grp["structure_cls"], dtype=np.float32)

        if emb is None:
            emb = np.zeros((out_dim,), dtype=np.float32)
        if emb.shape[0] > out_dim:
            emb = emb[:out_dim]
        elif emb.shape[0] < out_dim:
            emb = np.pad(emb, (0, out_dim - emb.shape[0]))
        return emb.astype(np.float32, copy=False)


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            if isinstance(raw.get("samples"), list):
                return raw["samples"]
            if isinstance(raw.get("data"), list):
                return raw["data"]
            if isinstance(raw.get("train"), list):
                return list(raw.get("train", [])) + list(raw.get("val", [])) + list(raw.get("test", []))
        if isinstance(raw, list):
            return raw
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)
    return records


def _rbf(distances: np.ndarray, radius: float, bins: int) -> np.ndarray:
    centers = np.linspace(0.0, radius, bins, dtype=np.float32)
    sigma = max(radius / max(bins, 1), 1e-6)
    diff = distances[:, None] - centers[None, :]
    return np.exp(-0.5 * (diff ** 2) / (sigma ** 2)).astype(np.float32)


def _find_residue(model, chain_id: Optional[str], target_resi: int):
    if chain_id:
        try:
            chain = model[str(chain_id)]
            for residue in chain:
                if residue.id[0] == " " and int(residue.id[1]) == int(target_resi):
                    return residue
        except Exception:
            pass

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if int(residue.id[1]) == int(target_resi):
                return residue
    return None


def _lru_get(cache: "OrderedDict[str, Any]", key: str) -> Optional[Any]:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_put(cache: "OrderedDict[str, Any]", key: str, value: Any, max_size: int) -> None:
    if max_size <= 0:
        return
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


def _build_pdb_context(model) -> Dict[str, Any]:
    residues: List[Tuple[str, int]] = []
    coords: List[np.ndarray] = []
    aa1_list: List[str] = []
    index_map: Dict[Tuple[str, int], int] = {}
    resid_to_first_idx: Dict[int, int] = {}

    for chain in model:
        chain_id = str(getattr(chain, "id", "")).strip() or "?"
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            resid = int(residue.id[1])
            coord = residue["CA"].get_coord()
            idx = len(residues)
            residues.append((chain_id, resid))
            coords.append(coord)
            aa1_list.append(_aa1_from_resname(residue.get_resname()))
            index_map[(chain_id, resid)] = idx
            resid_to_first_idx.setdefault(resid, idx)

    if not residues:
        return {"residues": [], "coords": np.zeros((0, 3), dtype=np.float32), "aa1": [], "index_map": {}, "resid_to_first_idx": {}}
    return {
        "residues": residues,
        "coords": np.asarray(coords, dtype=np.float32),
        "aa1": aa1_list,
        "index_map": index_map,
        "resid_to_first_idx": resid_to_first_idx,
    }


def _get_center_index(pdb_ctx: Dict[str, Any], target_chain: Optional[str], target_resi: int) -> Optional[int]:
    if not pdb_ctx.get("residues"):
        return None
    index_map = pdb_ctx.get("index_map", {})
    resid_to_first_idx = pdb_ctx.get("resid_to_first_idx", {})
    if target_chain:
        key = (str(target_chain).strip(), int(target_resi))
        if key in index_map:
            return int(index_map[key])
    if int(target_resi) in resid_to_first_idx:
        return int(resid_to_first_idx[int(target_resi)])
    return None


def _compute_dssp_lookups(model, pdb_file: Path) -> Tuple[Dict[tuple, np.ndarray], Dict[tuple, float]]:
    ss_lookup: Dict[tuple, np.ndarray] = {}
    rsa_lookup: Dict[tuple, float] = {}
    if not HAS_DSSP:
        return ss_lookup, rsa_lookup
    dssp = DSSP(model, str(pdb_file))
    ss_vocab = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7}
    for key in dssp.keys():
        chain_id, residue_id = key
        resid = int(residue_id[1]) if isinstance(residue_id, tuple) else int(residue_id)
        dssp_row = dssp[key]
        ss_char = str(dssp_row[2]) if len(dssp_row) > 2 else "-"
        rsa_value = float(dssp_row[3]) if len(dssp_row) > 3 else 0.0
        ss_onehot = np.zeros((8,), dtype=np.float32)
        ss_onehot[ss_vocab.get(ss_char, 7)] = 1.0
        ss_lookup[(chain_id, resid)] = ss_onehot
        rsa_lookup[(chain_id, resid)] = rsa_value
    return ss_lookup, rsa_lookup


def extract_pocket_graph(
    pdb_file: Path,
    uniprot_id: str,
    target_chain: Optional[str],
    target_resi: int,
    embedding_store: Optional[StructureEmbeddingStore] = None,
    esm_if_dim: int = 0,
    radius: float = 10.0,
    edge_radius: Optional[float] = None,
    rbf_bins: int = 16,
    parser: Optional[PDBParser] = None,
    pdb_context_cache: Optional["OrderedDict[str, Any]"] = None,
    pdb_context_cache_size: int = 128,
    dssp_cache: Optional["OrderedDict[str, Any]"] = None,
    dssp_cache_size: int = 256,
) -> Optional[Dict[str, torch.Tensor]]:
    node_radius = float(radius)
    use_edge_radius = float(edge_radius) if edge_radius is not None else float(radius)
    parser = parser or PDBParser(QUIET=True)

    pdb_key = str(pdb_file.resolve())
    pdb_ctx = None
    model = None
    if pdb_context_cache is not None:
        cached = _lru_get(pdb_context_cache, pdb_key)
        if isinstance(cached, dict) and "model" in cached and "pdb_ctx" in cached:
            model = cached["model"]
            pdb_ctx = cached["pdb_ctx"]
    if pdb_ctx is None or model is None:
        structure = parser.get_structure("struct", str(pdb_file))
        model = next(structure.get_models())
        pdb_ctx = _build_pdb_context(model)
        if pdb_context_cache is not None:
            _lru_put(pdb_context_cache, pdb_key, {"model": model, "pdb_ctx": pdb_ctx}, int(pdb_context_cache_size))

    center_idx = _get_center_index(pdb_ctx, target_chain, int(target_resi))
    if center_idx is None:
        return None
    all_coords = pdb_ctx["coords"]
    center_ca = all_coords[int(center_idx)]

    dist_to_center = np.linalg.norm(all_coords - center_ca[None, :], axis=-1)
    keep = dist_to_center <= float(node_radius)
    if not bool(np.any(keep)):
        return None
    keep_idx = np.where(keep)[0].astype(np.int64)
    residues_all: List[Tuple[str, int]] = pdb_ctx["residues"]
    aa1_all: List[str] = pdb_ctx["aa1"]
    residues = [residues_all[int(i)] for i in keep_idx.tolist()]
    coords = all_coords[keep_idx]
    aa1_list = [aa1_all[int(i)] for i in keep_idx.tolist()]

    if not residues:
        return None

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    edge_mask = (dists <= float(use_edge_radius)) & (dists > 0)
    edge_index = np.vstack(np.where(edge_mask)).astype(np.int64)
    edge_attr = _rbf(dists[edge_mask].reshape(-1), radius=float(use_edge_radius), bins=rbf_bins)

    ss_lookup: Dict[tuple, np.ndarray] = {}
    rsa_lookup: Dict[tuple, float] = {}
    if dssp_cache is not None:
        cached = _lru_get(dssp_cache, pdb_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            ss_lookup, rsa_lookup = cached
    if not ss_lookup and HAS_DSSP:
        try:
            ss_lookup, rsa_lookup = _compute_dssp_lookups(model, pdb_file)
        except Exception:
            ss_lookup, rsa_lookup = {}, {}
        if dssp_cache is not None:
            _lru_put(dssp_cache, pdb_key, (ss_lookup, rsa_lookup), int(dssp_cache_size))

    node_features = []
    for (chain_id, resid), coord, aa1 in zip(residues, coords, aa1_list):
        ss = ss_lookup.get((chain_id, resid), np.zeros((8,), dtype=np.float32))
        rsa = np.asarray([rsa_lookup.get((chain_id, resid), 0.0)], dtype=np.float32)
        rel_coord = ((coord - center_ca) / max(float(node_radius), 1e-6)).astype(np.float32)
        physicochem = _aa_physicochem(aa1)
        esm_if = (
            embedding_store.get_embedding(uniprot_id, resid, target_resi, esm_if_dim)
            if (embedding_store is not None and esm_if_dim > 0)
            else np.zeros((esm_if_dim,), dtype=np.float32)
        )
        feat = np.concatenate([rel_coord, ss, rsa, physicochem, esm_if], axis=0)
        node_features.append(feat)
    x = np.asarray(node_features, dtype=np.float32)

    return {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pocket graphs for EGNN training")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory with PDB files")
    parser.add_argument("--manifest", type=str, required=True, help="CSV or JSON manifest with uniprot_id/position")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write .pt graphs")
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--edge_radius", type=float, default=0.0, help="Edge cutoff in Å (0=use --radius)")
    parser.add_argument("--rbf_bins", type=int, default=16)
    parser.add_argument("--esm_if_h5", type=str, default="", help="Optional H5 path with proteins/<uid>/structure_features")
    parser.add_argument("--esm_if_dim", type=int, default=0, help="Override ESM-IF node embedding dim (0=infer from H5)")
    parser.add_argument("--uniprot_col", type=str, default="uniprot_id")
    parser.add_argument("--position_col", type=str, default="position")
    parser.add_argument("--chain_col", type=str, default="chain")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing graph files")
    parser.add_argument("--max_records", type=int, default=0, help="Max records to process (0=all)")
    parser.add_argument("--shuffle_records", action="store_true")
    parser.add_argument("--pdb_cache_size", type=int, default=128)
    parser.add_argument("--dssp_cache_size", type=int, default=256)
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_manifest(Path(args.manifest))
    if bool(args.shuffle_records):
        rng = np.random.default_rng(42)
        rng.shuffle(records)
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]

    resolver = PDBPathResolver(pdb_dir)
    embedding_store = StructureEmbeddingStore(Path(args.esm_if_h5)) if args.esm_if_h5 else StructureEmbeddingStore(None)
    esm_if_dim = int(args.esm_if_dim)
    if esm_if_dim <= 0:
        esm_if_dim = int(embedding_store.infer_dim())
    logger.info("Node feature layout: rel_coord(3)+DSSP(8)+RSA(1)+physchem(4)+esm_if(%d)", esm_if_dim)

    pdb_parser = PDBParser(QUIET=True)
    pdb_context_cache: "OrderedDict[str, Any]" = OrderedDict()
    dssp_cache: "OrderedDict[str, Any]" = OrderedDict()
    edge_radius = float(args.edge_radius) if float(args.edge_radius) > 0 else None

    written = 0
    skipped_existing = 0
    skipped = 0
    for rec in records:
        uid = str(rec.get(args.uniprot_col, "")).strip()
        if not uid:
            skipped += 1
            continue
        try:
            pos = int(rec.get(args.position_col, -1))
        except Exception:
            skipped += 1
            continue
        chain = rec.get(args.chain_col)
        out_path = output_dir / f"{uid}_{pos}.pt"
        if out_path.exists() and not bool(args.overwrite):
            skipped_existing += 1
            continue

        pdb_path = resolver.resolve(uid)
        if pdb_path is None:
            skipped += 1
            continue
        graph = extract_pocket_graph(
            pdb_file=pdb_path,
            uniprot_id=uid,
            target_chain=str(chain).strip() if chain else None,
            target_resi=pos,
            embedding_store=embedding_store,
            esm_if_dim=esm_if_dim,
            radius=float(args.radius),
            edge_radius=edge_radius,
            rbf_bins=int(args.rbf_bins),
            parser=pdb_parser,
            pdb_context_cache=pdb_context_cache,
            pdb_context_cache_size=int(args.pdb_cache_size),
            dssp_cache=dssp_cache,
            dssp_cache_size=int(args.dssp_cache_size),
        )
        if graph is None:
            skipped += 1
            continue
        graph["meta"] = {
            "node_dim": int(graph["x"].shape[1]),
            "edge_dim": int(graph["edge_attr"].shape[1]) if graph["edge_attr"].ndim == 2 else 0,
            "esm_if_dim": int(esm_if_dim),
            "radius": float(args.radius),
            "edge_radius": float(edge_radius) if edge_radius is not None else float(args.radius),
            "rbf_bins": int(args.rbf_bins),
        }
        torch.save(graph, out_path)
        written += 1

    embedding_store.close()
    logger.info(
        "Graph extraction complete: wrote=%d skipped_missing_or_invalid=%d skipped_existing=%d",
        written,
        skipped,
        skipped_existing,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
