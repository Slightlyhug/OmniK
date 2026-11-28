"""Graph, PDB mapping and micro-environment primitives from the OmniK training pipeline."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
PHYS = {
    "A": (0.248, 0, 0, 0),
    "R": (-1, 1, 1, 0),
    "N": (-0.312, 0, 1, 0),
    "D": (-0.36, -1, 1, 0),
    "C": (0.116, 0, 0, 0),
    "Q": (-0.34, 0, 1, 0),
    "E": (-0.296, -1, 1, 0),
    "G": (0.192, 0, 0, 0),
    "H": (-0.16, 0.3, 1, 1),
    "I": (0.552, 0, 0, 0),
    "L": (0.424, 0, 0, 0),
    "K": (-0.6, 1, 1, 0),
    "M": (0.256, 0, 0, 0),
    "F": (0.476, 0, 0, 1),
    "P": (0.048, 0, 0, 0),
    "S": (-0.072, 0, 1, 0),
    "T": (-0.02, 0, 1, 0),
    "W": (0.324, 0, 0, 1),
    "Y": (0.104, 0, 1, 1),
    "V": (0.432, 0, 0, 0),
}


def parse_pdb(path, parser):
    model = next(parser.get_structure(path.stem, str(path)).get_models())
    chains = {}
    residues = []
    for chain in model:
        cid = str(chain.id).strip() or "?"
        chain_res = []
        for res in chain:
            if res.id[0] != " " or "CA" not in res:
                continue
            aa = AA3_TO_1.get(res.resname.upper(), "X")
            item = {
                "chain": cid,
                "resid": int(res.id[1]),
                "icode": str(res.id[2]).strip(),
                "aa": aa,
                "coord": np.asarray(res["CA"].coord, dtype=np.float32),
            }
            chain_res.append(item)
            residues.append(item)
        chains[cid] = chain_res
    return chains, residues


def build_graph(
    chains, all_residues, chain_id, position, seq_radius, spatial_radius, edge_radius
):
    target_candidates = []
    for cid, chain in chains.items():
        if chain_id and cid != chain_id:
            continue
        target_candidates.extend(
            (cid, i, r) for i, r in enumerate(chain) if r["resid"] == position
        )
    if not target_candidates:
        return None, "center_residue_missing"
    if len(target_candidates) > 1 and not chain_id:
        return None, "ambiguous_chain_for_center"
    cid, center_i, center = target_candidates[0]
    if center["aa"] != "K":
        return None, "center_residue_not_lysine"

    seq_chain = chains[cid]
    seq_ids = set(
        range(
            max(0, center_i - seq_radius),
            min(len(seq_chain), center_i + seq_radius + 1),
        )
    )
    center_xyz = center["coord"]
    spatial_ids = {
        i
        for i, r in enumerate(all_residues)
        if float(np.linalg.norm(r["coord"] - center_xyz)) <= spatial_radius
    }
    selected = [(cid, i, r) for i, r in enumerate(seq_chain) if i in seq_ids]
    selected_keys = {(r["chain"], r["resid"], r["icode"]) for _, _, r in selected}
    selected.extend(
        (r["chain"], i, r)
        for i, r in enumerate(all_residues)
        if i in spatial_ids
        and (r["chain"], r["resid"], r["icode"]) not in selected_keys
    )
    keys = [(r["chain"], r["resid"], r["icode"]) for _, _, r in selected]
    center_key = (cid, center["resid"], center["icode"])
    center_idx = keys.index(center_key)
    coords = np.stack([r["coord"] for _, _, r in selected])
    pos = coords - center_xyz[None, :]

    # Scalar node layout: amino-acid one-hot (20), DSSP one-hot (8), RSA (1),
    # physicochemical descriptors (4), sequence-offset indicator (1).
    x = np.zeros((len(selected), 34), dtype=np.float32)
    seq_index = {(cid, i): i - center_i for i in seq_ids}
    for j, (res_chain, res_i, residue) in enumerate(selected):
        if residue["aa"] in AA_ORDER:
            x[j, AA_ORDER.index(residue["aa"])] = 1.0
        # DSSP and RSA remain zero until a DSSP executable/data source is available.
        x[j, 29] = float(
            np.clip(seq_index.get((res_chain, res_i), 0) / max(seq_radius, 1), -1, 1)
        )
        x[j, 30:34] = PHYS.get(residue["aa"], (0, 0, 0, 0))

    delta = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    edge_pairs = np.argwhere((distances <= edge_radius) & (distances > 1e-6))
    edge_index = edge_pairs.T.astype(np.int64)
    centers = np.linspace(0.0, edge_radius, 16, dtype=np.float32)
    sigma = max(edge_radius / 16.0, 1e-6)
    edge_attr = np.exp(
        -0.5
        * ((distances[edge_pairs[:, 0], edge_pairs[:, 1], None] - centers) / sigma) ** 2
    ).astype(np.float32)
    meta = {
        "center_index": int(center_idx),
        "center_chain": cid,
        "center_resid": int(position),
        "n_nodes": int(len(selected)),
        "n_edges": int(edge_index.shape[1]),
        "dssp_available": False,
        "rsa_available": False,
        "node_feature_layout": "aa_onehot20,dssp_onehot8,rsa1,sequence_offset1,physchem4",
    }
    return {
        "x": x,
        "pos": pos.astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "meta": meta,
        "residue_keys": np.asarray([f"{a}:{b}{c}" for a, b, c in keys], dtype="S32"),
    }, None


AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
    "SEC",
    "PYL",
    "MLY",
    "KCX",
    "LLP",
    "LLY",
    "M3L",
    "ALY",
    "FME",
    "CME",
    "CSO",
    "PCA",
    "NLE",
    "NVA",
    "HYP",
    "PTR",
    "SEP",
    "TPO",
}


def parse_ca(path: Path):
    coords, resids = [], []
    seen = set()
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            if line[17:20].strip().upper() not in AA3:
                continue
            try:
                resid = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (line[21], resid, line[26])
            if key in seen:
                continue
            seen.add(key)
            coords.append(xyz)
            resids.append(resid)
    return np.asarray(coords, dtype=np.float32), np.asarray(resids, dtype=np.int32)


def micro_for_position(coords, resids, position):
    matches = np.where(resids == position)[0]
    if matches.size == 0:
        return np.zeros((6,), np.float32), 0
    center = int(matches[0])
    d = np.linalg.norm(coords - coords[center], axis=1)
    valid = np.ones(d.shape[0], dtype=bool)
    valid[center] = False
    nd = d[valid]
    npos = resids[valid]
    if nd.size == 0:
        return np.zeros((6,), np.float32), 0
    near8 = nd <= 8.0
    near12 = nd <= 12.0
    k = min(5, int(nd.size))
    nearest = np.partition(nd, k - 1)[:k]
    seq_off = np.abs(npos.astype(np.float32) - float(position))
    seq12 = seq_off[near12]
    vals = np.asarray(
        [
            float(np.sum(near8)),
            float(np.sum(near12)),
            float(np.mean(nearest)) if nearest.size else 0.0,
            float(np.std(nearest)) if nearest.size else 0.0,
            float(np.mean(seq12)) if seq12.size else 0.0,
            float(np.sum(np.exp(-nd / 10.0))),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vals, nan=0.0, posinf=1e4, neginf=0.0), 1


NONSTANDARD = {
    "PTR": "Y",
    "SEP": "S",
    "TPO": "T",
    "MSE": "M",
    "KCX": "K",
    "LLP": "K",
    "LLY": "K",
    "MLY": "K",
    "M3L": "K",
    "HYP": "P",
    "ALY": "K",
    "FME": "M",
    "CME": "C",
    "CSO": "C",
    "PCA": "E",
    "NLE": "L",
    "NVA": "V",
}


def decode(value):
    return (
        value.decode("ascii", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )


def residue_key(chain_id, residue_id):
    return f"{chain_id}:{int(residue_id[1])}{str(residue_id[2]).strip()}"


def map_pdb_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure(pdb_path.stem, str(pdb_path)).get_models())
    residue_to_index = {}
    sequence_by_chain = defaultdict(list)
    for chain in model:
        index = 0
        for residue in chain:
            if residue.id[0] != " " and residue.resname not in NONSTANDARD:
                continue
            if not all(atom in residue for atom in ("N", "CA", "C")):
                continue
            aa = AA3_TO_1.get(
                residue.resname.upper(), NONSTANDARD.get(residue.resname.upper())
            )
            if aa is None:
                continue
            residue_to_index[residue_key(chain.id, residue.id)] = index
            sequence_by_chain[chain.id].append(aa)
            index += 1
    return residue_to_index, {c: "".join(s) for c, s in sequence_by_chain.items()}
