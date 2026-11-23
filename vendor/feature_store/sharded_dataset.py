#!/usr/bin/env python3
"""ShardedProteinSiteDataset: feature_store_v2 reader compatible with the
SiteFeatureLMDBDataset sample dict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

AA = {"A": 1, "R": 2, "N": 3, "D": 4, "C": 5, "Q": 6, "E": 7, "G": 8,
      "H": 9, "I": 10, "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
      "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20}


def padded_window(mat, center_zero_based, width=73):
    out = np.zeros((width, int(mat.shape[1])), dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return out
    start = int(center_zero_based) - width // 2
    a, b = max(0, start), min(int(mat.shape[0]), start + width)
    if b > a:
        out[a - start:a - start + b - a] = np.asarray(mat[a:b], dtype=np.float32)
    return out


class ShardedProteinSiteDataset(Dataset):
    def __init__(self, root, split, window=73, use_structure=True, use_graph=True,
                 structure_mask_token_value=1.0, target_label_index=None,
                 known_only=False):
        self.root = Path(root) / split
        self.split = split
        self.window = int(window)
        self.use_structure = bool(use_structure)
        self.use_graph = bool(use_graph)
        self.structure_mask_token_value = float(structure_mask_token_value)
        self.target_label_index = None if target_label_index is None else int(target_label_index)
        self.known_only = bool(known_only)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        g = self.root / "global"
        self.site_keys = (g / "site_keys.txt").read_text().splitlines()
        self.site_shard = np.load(g / "site_shard.npy", mmap_mode="r")
        self.site_local = np.load(g / "site_local.npy", mmap_mode="r")
        self.labels = np.load(g / "labels.npy", mmap_mode="r")
        # Keep the feature-store arrays memory mapped, but construct a compact
        # index for single-task runs.  This prevents unknown (-1) labels and
        # samples belonging only to other PTMs from entering the loader.
        if self.target_label_index is not None:
            if self.target_label_index < 0 or self.target_label_index >= int(self.labels.shape[1]):
                raise IndexError(
                    f"target_label_index={self.target_label_index} outside labels shape {self.labels.shape}"
                )
            target = np.asarray(self.labels[:, self.target_label_index])
            self.sample_indices = np.flatnonzero(target >= 0) if self.known_only else np.arange(len(self.site_keys), dtype=np.int64)
        else:
            self.sample_indices = np.arange(len(self.site_keys), dtype=np.int64)
        self._handles = {}

    def __len__(self):
        return int(self.sample_indices.shape[0])

    def get_label_matrix(self):
        if self.target_label_index is not None:
            labels = self.labels[self.sample_indices, self.target_label_index]
            return np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        return np.asarray(self.labels[self.sample_indices], dtype=np.float32)

    def _shard(self, sid: int):
        if sid in self._handles:
            return self._handles[sid]
        d = self.root / "shards" / f"shard_{sid:04d}"
        meta = json.loads((d / "meta.json").read_text())
        h = {"meta": meta, "proteins": (d / "proteins.txt").read_text().splitlines()}
        for name, fn in {
            "protein_seq_features": "protein_seq_features.npy",
            "protein_struct_features": "protein_struct_features.npy",
            "protein_seq_offsets": "protein_seq_offsets.npy",
            "protein_struct_offsets": "protein_struct_offsets.npy",
            "protein_seq_mean": "protein_seq_mean.npy",
            "protein_struct_mean": "protein_struct_mean.npy",
            "protein_flags": "protein_flags.npy",
            "protein_seq_length": "protein_seq_length.npy",
            "protein_struct_length": "protein_struct_length.npy",
            "site_protein": "site_protein.npy",
            "site_position": "site_position.npy",
            "site_window_sequence": "site_window_sequence.npy",
            "site_micro_env": "site_micro_env.npy",
            "site_micro_env_available": "site_micro_env_available.npy",
            "site_graph_available": "site_graph_available.npy",
            "site_center_index": "site_center_index.npy",
            "graph_node_features": "graph_node_features.npy",
            "graph_node_esm_index": "graph_node_esm_index.npy",
            "graph_node_positions": "graph_node_positions.npy",
            "graph_node_offsets": "graph_node_offsets.npy",
            "graph_edge_index": "graph_edge_index.npy",
            "graph_edge_attr": "graph_edge_attr.npy",
            "graph_edge_offsets": "graph_edge_offsets.npy",
            "graph_token_index": "graph_token_index.npy",
        }.items():
            h[name] = np.load(d / fn, mmap_mode="r")
        self._handles[sid] = h
        return h

    def __getitem__(self, idx):
        i = int(idx)
        global_i = int(self.sample_indices[i])
        sid = int(self.site_shard[global_i])
        j = int(self.site_local[global_i])
        sh = self._shard(sid)
        pid = int(sh["site_protein"][j])
        pos = int(sh["site_position"][j])
        seq_len = int(sh["protein_seq_length"][pid])
        struct_len = int(sh["protein_struct_length"][pid])
        flags = sh["protein_flags"][pid]
        a0, a1 = int(sh["protein_seq_offsets"][pid]), int(sh["protein_seq_offsets"][pid + 1])
        s0, s1 = int(sh["protein_struct_offsets"][pid]), int(sh["protein_struct_offsets"][pid + 1])
        seq_mat = sh["protein_seq_features"][a0:a1]
        struct_mat = sh["protein_struct_features"][s0:s1]
        seq = padded_window(seq_mat, pos - 1, self.window)
        struct = padded_window(struct_mat, pos - 1, self.window)
        struct_available = bool(int(flags[1]) and 1 <= pos <= seq_len and struct_len > 0)
        if not self.use_structure:
            struct.fill(0.0)
            struct_available = False
        elif not struct_available:
            struct[:, -1] = self.structure_mask_token_value
        seq_global = np.asarray(sh["protein_seq_mean"][pid], dtype=np.float32)
        struct_global = np.asarray(sh["protein_struct_mean"][pid], dtype=np.float32)
        window = sh["site_window_sequence"][j]
        if isinstance(window, (bytes, np.bytes_)):
            window = bytes(window).decode("ascii", "ignore")
        else:
            window = str(window)
        window = window[: self.window].ljust(self.window, "X")
        residue_types = torch.tensor([AA.get(c, 0) for c in window], dtype=torch.long)

        if self.target_label_index is not None:
            labels = np.asarray([self.labels[global_i, self.target_label_index]], dtype=np.float32)
        else:
            labels = np.asarray(self.labels[global_i], dtype=np.float32)
        # State ids: unknown=0, positive=1, observed_negative=2, reliable_negative=3.
        # Known negatives MUST map to 2 (or 3), never 1: the state-aware ASL applies
        # the positive term to state==1, so the old 0->1 mapping pushed every known
        # negative up instead of down.
        label_state = np.where(labels < 0, 0, np.where(labels > 0, 1, 2)).astype(np.int64)
        micro = np.asarray(sh["site_micro_env"][j], dtype=np.float32)
        micro_available = bool(int(sh["site_micro_env_available"][j]))

        n0, n1 = int(sh["graph_node_offsets"][j]), int(sh["graph_node_offsets"][j + 1])
        e0, e1 = int(sh["graph_edge_offsets"][j]), int(sh["graph_edge_offsets"][j + 1])
        x = np.asarray(sh["graph_node_features"][n0:n1], dtype=np.float32)
        eidx = np.asarray(sh["graph_node_esm_index"][n0:n1], dtype=np.int64)
        gpos = np.asarray(sh["graph_node_positions"][n0:n1], dtype=np.float32)
        gi = np.asarray(sh["graph_edge_index"][:, e0:e1], dtype=np.int64)
        ga = np.asarray(sh["graph_edge_attr"][e0:e1], dtype=np.float32)
        ti = np.asarray(sh["graph_token_index"][n0:n1], dtype=np.int64)
        gav = int(sh["site_graph_available"][j])
        ci = int(sh["site_center_index"][j])
        if x.shape[0] == 0:
            x = np.zeros((1, 34), np.float32)
            eidx = np.full((1,), -1, np.int64)
            gpos = np.zeros((1, 3), np.float32)
            ti = np.full((1,), -1, np.int64)
            gav = 0
            ci = -1
        esm_if = np.zeros((x.shape[0], 512), np.float32)
        valid = (eidx >= 0) & (eidx < struct_mat.shape[0])
        if np.any(valid):
            esm_if[valid] = np.asarray(struct_mat[eidx[valid]], dtype=np.float32)
        esm_mask = valid.astype(np.float32)
        center_mask = np.zeros((x.shape[0],), np.float32)
        if 0 <= ci < x.shape[0]:
            center_mask[ci] = 1.0
        elif not gav:
            center_mask[:] = 1.0
        graph = None
        if self.use_graph and gav:
            graph = {
                "x": torch.from_numpy(np.ascontiguousarray(x)),
                "esm_if": torch.from_numpy(np.ascontiguousarray(esm_if)),
                "esm_if_mask": torch.from_numpy(esm_mask),
                "center_mask": torch.from_numpy(center_mask),
                "pos": torch.from_numpy(np.ascontiguousarray(gpos)),
                "edge_index": torch.from_numpy(np.ascontiguousarray(gi)),
                "edge_attr": torch.from_numpy(np.ascontiguousarray(ga)),
                "token_index": torch.from_numpy(np.ascontiguousarray(ti)),
                "graph_available": np.uint8(1),
            }
        elif self.use_graph:
            graph = {
                "x": torch.zeros((1, 34)), "esm_if": torch.zeros((1, 512)),
                "esm_if_mask": torch.zeros((1,)), "center_mask": torch.ones((1,)),
                "pos": torch.zeros((1, 3)), "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 16)), "token_index": torch.full((1,), -1, dtype=torch.long),
                "graph_available": np.uint8(0),
            }

        return {
            "sample_index": global_i, "uniprot_id": sh["proteins"][pid], "position": pos,
            "residue": "K", "sequence_window_str": window,
            "sequence": {
                "window_features": torch.from_numpy(seq),
                "local_features": torch.from_numpy(seq[21:52]),
                "global_features": torch.from_numpy(seq_global),
                "global_context_available": torch.tensor(float(int(flags[2]))),
                "residue_types": residue_types, "position_ids": torch.arange(self.window),
                "relative_position_offsets": torch.arange(self.window) - self.window // 2,
                "site_indicator": torch.nn.functional.one_hot(torch.tensor(self.window // 2),
                                                              num_classes=self.window).float(),
            },
            "structure": torch.from_numpy(struct),
            "structure_global_features": torch.from_numpy(struct_global),
            "structure_available": torch.tensor(float(struct_available)),
            "structure_global_available": torch.tensor(float(int(flags[3]) and struct_len > 0)),
            "structure_graph": graph,
            "structure_graph_available": torch.tensor(float(gav if self.use_graph else 0)),
            "micro_env_features": torch.from_numpy(micro),
            "micro_env_available": torch.tensor(float(micro_available)),
            "motif_bias": torch.zeros(labels.shape[0]), "label": torch.from_numpy(labels),
            "loss_weight": torch.ones(labels.shape[0]), "label_state": torch.tensor(label_state),
            "label_reliability": torch.ones(labels.shape[0]), "teacher_probs": torch.full(labels.shape, -1.),
            "teacher_mask": torch.zeros(labels.shape), "teacher_weight": torch.ones(labels.shape),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        self._handles.clear()
