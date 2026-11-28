#!/usr/bin/env python3
"""Pack split CSVs, ESM HDF5 and local graphs for the released shard loaders.

Rebuilds features from supplied local structures; historical structure versions
are needed for numerical reproduction of manuscript predictions.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.training_config import DEFAULT_PTM_TYPES, canonical_ptm_name
from feature_building import (
    build_graph,
    parse_pdb,
    parse_ca,
    micro_for_position,
    map_pdb_residues,
    decode,
)


def concatenate(items, shape, dtype, axis=0):
    return (
        np.concatenate(items, axis=axis).astype(dtype)
        if items
        else np.zeros(shape, dtype)
    )


def build_split(args, split, esm):
    df = pd.read_csv(args.data_dir / f"{split}.csv", dtype={"uniprot_id": str})
    if args.limit_proteins:
        ids = sorted(df.uniprot_id.unique())[: args.limit_proteins]
        df = df[df.uniprot_id.isin(ids)]
    columns = {canonical_ptm_name(c): c for c in df.columns}
    tasks = [columns[t] for t in DEFAULT_PTM_TYPES]
    df = df.sort_values(["uniprot_id", "position"], kind="stable").reset_index(
        drop=True
    )
    if df.duplicated(["uniprot_id", "position"]).any():
        raise ValueError(f"{split}: duplicate site identifiers")
    if df.empty:
        raise ValueError(f"{split}: no sites selected")
    if not df.window_sequence.astype(str).str.len().eq(73).all():
        raise ValueError(f"{split}: expected 73-character sequence windows")
    if not np.isin(df[tasks].to_numpy(), [-1, 0, 1]).all():
        raise ValueError(f"{split}: expected labels in -1, 0, 1")
    dest = args.out / split
    if (dest / "manifest.json").exists():
        raise FileExistsError(
            f"Complete store already exists: {dest}; choose a new output directory"
        )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "global").mkdir(exist_ok=True)
    groups = list(df.groupby("uniprot_id", sort=True))
    keys, shards, locals_, labels = [], [], [], []
    coverage = {
        "missing_sequence": 0,
        "missing_structure": 0,
        "graphs_available": 0,
        "dssp_failed_proteins": 0,
        "graph_esm_mapping_failed_proteins": 0,
    }
    protein_start = 0
    for sid, start in enumerate(range(0, len(groups), args.proteins_per_shard)):
        subset = groups[start : start + args.proteins_per_shard]
        arrays = {
            name: []
            for name in (
                "protein_seq_features",
                "protein_struct_features",
                "protein_seq_mean",
                "protein_struct_mean",
                "protein_flags",
                "protein_seq_length",
                "protein_struct_length",
                "site_protein",
                "site_position",
                "site_window_sequence",
                "site_micro_env",
                "site_micro_env_available",
                "site_graph_available",
                "site_center_index",
                "graph_node_features",
                "graph_node_esm_index",
                "graph_node_positions",
                "graph_edge_index",
                "graph_edge_attr",
                "graph_token_index",
            )
        }
        seq_off, struct_off, node_off, edge_off = [0], [0], [0], [0]
        site_start = len(keys)
        for pid, (uid, rows) in enumerate(subset):
            entry = esm.get(uid)
            seq = (
                np.asarray(entry["sequence_features"], np.float16)
                if entry is not None and "sequence_features" in entry
                else np.zeros((0, 1280), np.float16)
            )
            if seq.ndim != 2 or seq.shape[1] != 1280 or not len(seq):
                raise ValueError(
                    f"{uid}: missing or invalid ESM-2 features; do not train on zero sequence embeddings"
                )
            raw = (
                np.asarray(entry["structure_features"], np.float16)
                if "structure_features" in entry
                else None
            )
            valid_struct = raw is not None and raw.shape == (len(seq), 512)
            struct = raw if valid_struct else np.zeros((len(seq), 512), np.float16)
            coverage["missing_structure"] += int(not valid_struct)
            for name, value in [
                ("protein_seq_features", seq),
                ("protein_struct_features", struct),
                ("protein_seq_mean", seq.mean(0, dtype=np.float32)),
                ("protein_struct_mean", struct.mean(0, dtype=np.float32)),
                ("protein_flags", [1, int(valid_struct), 1, int(valid_struct)]),
                ("protein_seq_length", len(seq)),
                ("protein_struct_length", len(struct) if valid_struct else 0),
            ]:
                arrays[name].append(value)
            seq_off.append(seq_off[-1] + len(seq))
            struct_off.append(struct_off[-1] + len(struct))
            pdb = args.pdb_dir / f"{uid}.pdb"
            chains = residues = None
            ca, resids = np.zeros((0, 3)), np.zeros(0)
            esm_map, ss = {}, {}
            if pdb.exists():
                chains, residues = parse_pdb(pdb, PDBParser(QUIET=True))
                ca, resids = parse_ca(pdb)
                if valid_struct:
                    mapping, sequences = map_pdb_residues(pdb)
                    feature_seq = decode(entry.attrs.get("sequence", ""))
                    matching = [
                        c
                        for c, s in sequences.items()
                        if s == feature_seq and len(s) == len(struct)
                    ]
                    if len(matching) == 1:
                        esm_map = {
                            k: v
                            for k, v in mapping.items()
                            if k.startswith(matching[0] + ":")
                        }
                    else:
                        coverage["graph_esm_mapping_failed_proteins"] += 1
                if args.dssp_bin:
                    try:
                        model = next(
                            PDBParser(QUIET=True)
                            .get_structure(uid, str(pdb))
                            .get_models()
                        )
                        dssp = DSSP(
                            model,
                            str(pdb),
                            dssp=args.dssp_bin,
                            file_type="PDB",
                            acc_array="Sander",
                        )
                        ss = {
                            f"{c}:{r[1]}{str(r[2]).strip()}": dssp[(c, r)]
                            for c, r in dssp.keys()
                        }
                    except Exception:
                        coverage["dssp_failed_proteins"] += 1
            for _, row in rows.iterrows():
                pos = int(row.position)
                graph, reason = (
                    build_graph(
                        chains,
                        residues,
                        None,
                        pos,
                        args.seq_radius,
                        args.spatial_radius,
                        args.edge_radius,
                    )
                    if chains
                    else (None, "missing_pdb")
                )
                if graph is None:
                    graph = {
                        "x": np.zeros((1, 34)),
                        "pos": np.zeros((1, 3)),
                        "edge_index": np.zeros((2, 0), dtype=np.int32),
                        "edge_attr": np.zeros((0, 16)),
                        "residue_keys": np.array([b""]),
                        "meta": {"center_index": -1},
                    }
                    available = 0
                else:
                    available = 1
                coverage["graphs_available"] += available
                gkeys = [decode(k).rstrip("\0") for k in graph["residue_keys"]]
                for i, k in enumerate(gkeys):
                    if k in ss:
                        record = ss[k]
                        graph["x"][i, 20:28] = 0
                        graph["x"][
                            i,
                            20
                            + {
                                "H": 0,
                                "B": 1,
                                "E": 2,
                                "G": 3,
                                "I": 4,
                                "T": 5,
                                "S": 6,
                                "-": 7,
                            }.get(record[2], 7),
                        ] = 1
                        if isinstance(record[3], (float, int)) and np.isfinite(
                            record[3]
                        ):
                            graph["x"][i, 28] = np.clip(record[3], 0, 2)
                center_chain = graph["meta"].get("center_chain")
                token = []
                for k in gkeys:
                    chain, _, number = k.partition(":")
                    try:
                        offset = 36 + int(number) - pos
                        token.append(
                            offset if chain == center_chain and 0 <= offset < 73 else -1
                        )
                    except ValueError:
                        token.append(-1)
                micro, ma = (
                    micro_for_position(ca, resids, pos) if len(ca) else (np.zeros(6), 0)
                )
                keys.append(f"{uid}_{pos}")
                shards.append(sid)
                locals_.append(len(arrays["site_position"]))
                labels.append(row[tasks].to_numpy(dtype=np.int8))
                for name, value in [
                    ("site_protein", pid),
                    ("site_position", pos),
                    ("site_window_sequence", str(row.window_sequence)),
                    ("site_micro_env", micro),
                    ("site_micro_env_available", ma),
                    ("site_graph_available", available),
                    ("site_center_index", graph["meta"]["center_index"]),
                    ("graph_node_features", graph["x"]),
                    (
                        "graph_node_esm_index",
                        np.array([esm_map.get(k, -1) for k in gkeys]),
                    ),
                    ("graph_node_positions", graph["pos"]),
                    ("graph_edge_index", graph["edge_index"]),
                    ("graph_edge_attr", graph["edge_attr"]),
                    ("graph_token_index", np.array(token)),
                ]:
                    arrays[name].append(value)
                node_off.append(node_off[-1] + len(graph["x"]))
                edge_off.append(edge_off[-1] + graph["edge_index"].shape[1])
        sd = dest / "shards" / f"shard_{sid:04d}"
        sd.mkdir(parents=True, exist_ok=True)
        concat_names = {
            "protein_seq_features": ((0, 1280), np.float16, 0),
            "protein_struct_features": ((0, 512), np.float16, 0),
            "graph_node_features": ((0, 34), np.float16, 0),
            "graph_node_positions": ((0, 3), np.float32, 0),
            "graph_node_esm_index": ((0,), np.int32, 0),
            "graph_edge_index": ((2, 0), np.int32, 1),
            "graph_edge_attr": ((0, 16), np.float16, 0),
            "graph_token_index": ((0,), np.int16, 0),
        }
        for name, values in arrays.items():
            if name in concat_names:
                shape, dtype, axis = concat_names[name]
                value = concatenate(values, shape, dtype, axis)
            else:
                dtype = (
                    "S73"
                    if name == "site_window_sequence"
                    else (
                        np.float16
                        if name
                        in ("protein_seq_mean", "protein_struct_mean", "site_micro_env")
                        else np.int32
                    )
                )
                value = np.asarray(values, dtype=dtype)
            np.save(sd / f"{name}.npy", value)
        for name, value in [
            ("protein_seq_offsets", seq_off),
            ("protein_struct_offsets", struct_off),
            ("graph_node_offsets", node_off),
            ("graph_edge_offsets", edge_off),
        ]:
            np.save(sd / f"{name}.npy", np.asarray(value, np.int64))
        (sd / "proteins.txt").write_text("\n".join(uid for uid, _ in subset) + "\n")
        meta = {
            "shard_id": sid,
            "site_start": site_start,
            "site_end": len(keys),
            "n_sites": len(keys) - site_start,
            "n_proteins": len(subset),
            "protein_start": protein_start,
            "protein_end": protein_start + len(subset),
            "seq_dim": 1280,
            "struct_dim": 512,
            "graph_node_dim": 34,
            "graph_edge_dim": 16,
            "window_size": 73,
        }
        (sd / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        protein_start += len(subset)
        print(split, "shard", sid, "sites", len(keys), flush=True)
    for name, value in [
        ("site_shard", shards),
        ("site_local", locals_),
        ("labels", labels),
    ]:
        np.save(
            dest / "global" / f"{name}.npy",
            np.asarray(value, np.int8 if name == "labels" else np.int32),
        )
    (dest / "global/site_keys.txt").write_text("\n".join(keys) + "\n")
    manifest = {
        "format": "feature_store_v2",
        "split": split,
        "n_sites": len(keys),
        "n_proteins": len(groups),
        "n_shards": (len(groups) + args.proteins_per_shard - 1)
        // args.proteins_per_shard,
        "seq_dim": 1280,
        "struct_dim": 512,
        "graph_node_dim": 34,
        "graph_model_node_dim": 547,
        "graph_edge_dim": 16,
        "window_size": 73,
        "coverage": coverage,
        "parameters": {
            "seq_radius": args.seq_radius,
            "spatial_radius": args.spatial_radius,
            "edge_radius": args.edge_radius,
            "dssp_bin": args.dssp_bin,
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--esm-h5", required=True, type=Path)
    ap.add_argument("--pdb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    ap.add_argument("--proteins-per-shard", type=int, default=128)
    ap.add_argument(
        "--limit-proteins",
        type=int,
        default=0,
        help="Debug subset only; changes dataset size",
    )
    ap.add_argument("--seq-radius", type=int, default=15)
    ap.add_argument("--spatial-radius", type=float, default=10.0)
    ap.add_argument("--edge-radius", type=float, default=10.0)
    ap.add_argument(
        "--dssp-bin",
        default=None,
        help="mkdssp executable; omission zeros DSSP/RSA channels",
    )
    args = ap.parse_args()
    if args.proteins_per_shard < 1:
        ap.error("--proteins-per-shard must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.esm_h5, "r") as handle:
        for split in args.splits:
            build_split(args, split, handle["proteins"])


if __name__ == "__main__":
    main()
