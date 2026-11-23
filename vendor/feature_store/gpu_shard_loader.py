#!/usr/bin/env python3
"""GPU-side batch construction from feature_store_v2 shards.

The loader keeps the global labels/aux arrays on GPU, streams one shard's
protein/graph arrays to GPU at a time, and builds the 73-aa windows, ESM-IF
gather and padded graph batch entirely on GPU.  It exposes the small subset of
the DataLoader interface used by the training loop.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "vendor" / "feature_store"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from sharded_dataset import AA  # noqa: E402
from src.utils.training_data import multilabel_collate_fn  # noqa: E402

WINDOW = 73
HALF = WINDOW // 2


class GpuShardBatchLoader:
    def __init__(
        self,
        store_root: str | Path,
        split: str,
        batch_size: int,
        device: torch.device,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        dataset: Any = None,
        log_fn=None,
        prefetch_shards: bool = True,
    ) -> None:
        self.root = Path(store_root) / split
        self.split = split
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.shuffle = bool(shuffle)
        self.dataset = dataset
        self.collate_fn = multilabel_collate_fn
        self.epoch = 0
        self.log_fn = log_fn
        self.prefetch_shards = bool(prefetch_shards)
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetch_shards else None
        self._prefetch_future = None
        self._prefetch_sid = None

        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.labels_gpu = torch.as_tensor(
            np.asarray(np.load(self.root / "global/labels.npy", mmap_mode="r"))
        ).to(self.device)
        self.specs: List[tuple[int, int, int]] = []
        self.shard_meta: Dict[int, dict] = {}
        for shard_dir in sorted((self.root / "shards").glob("shard_*")):
            meta = json.loads((shard_dir / "meta.json").read_text())
            sid = int(meta["shard_id"])
            self.shard_meta[sid] = meta
            n = int(meta["n_sites"])
            for start in range(0, n, self.batch_size):
                self.specs.append((sid, start, min(start + self.batch_size, n)))
        self._g: Optional[Dict[str, Any]] = None
        self._g_sid: Optional[int] = None
        self._residue_types: Optional[torch.Tensor] = None
        self._load_seconds = 0.0
        self._load_count = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.world_size > 1:
            return len(self.specs) // self.world_size
        return len(self.specs)

    def _specs_for_epoch(self) -> List[tuple[int, int, int]]:
        order = list(range(len(self.shard_meta)))
        if self.shuffle:
            import random
            random.Random(self.seed + self.epoch).shuffle(order)
        specs = []
        for sid in order:
            specs.extend([s for s in self.specs if s[0] == sid])
        if self.world_size > 1:
            per_rank = len(specs) // self.world_size
            specs = specs[self.rank * per_rank:(self.rank + 1) * per_rank]
        return specs

    def _load_shard_arrays(self, sid: int, async_copy: bool = False) -> Dict[str, Any]:
        shard_dir = self.root / "shards" / f"shard_{sid:04d}"
        t0 = time.time()
        g: Dict[str, Any] = {}
        stream = torch.cuda.Stream(device=self.device) if async_copy and self.device.type == "cuda" else None
        for name in (
            "protein_seq_features", "protein_struct_features",
            "protein_seq_offsets", "protein_struct_offsets",
            "protein_seq_mean", "protein_struct_mean", "protein_flags",
            "protein_seq_length", "protein_struct_length",
            "site_protein", "site_position", "site_micro_env",
            "site_micro_env_available", "site_graph_available", "site_center_index",
            "graph_node_features", "graph_node_esm_index", "graph_node_positions",
            "graph_node_offsets", "graph_edge_index", "graph_edge_attr",
            "graph_edge_offsets", "graph_token_index",
        ):
            arr = np.load(shard_dir / f"{name}.npy", mmap_mode="r")
            t = torch.as_tensor(np.asarray(arr))
            if async_copy and self.device.type == "cuda":
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                with torch.cuda.stream(stream):
                    t = t.to(self.device, non_blocking=True)
            else:
                t = t.to(self.device, non_blocking=False)
            g[name] = t
        windows = np.load(shard_dir / "site_window_sequence.npy", mmap_mode="r")
        table = np.zeros((len(windows), WINDOW), dtype=np.int64)
        for i, w in enumerate(windows):
            s = bytes(w).decode("ascii", "ignore") if isinstance(w, (bytes, np.bytes_)) else str(w)
            s = s[:WINDOW].ljust(WINDOW, "X")
            table[i] = [AA.get(c, 0) for c in s]
        g["_residue_types"] = torch.as_tensor(table).to(self.device)
        if stream is not None:
            stream.synchronize()
        elapsed = time.time() - t0
        if self.log_fn is not None:
            self.log_fn(f"GPU loader: loaded shard {sid} in {elapsed:.2f}s (prefetch={async_copy})")
        return g

    def _load_shard(self, sid: int) -> None:
        self._g = self._load_shard_arrays(sid, async_copy=False)
        self._g_sid = sid
        self._load_seconds += 0.0
        self._load_count += 1

    def _build_batch(self, sid: int, start: int, end: int) -> Dict[str, Any]:
        g = self._g
        assert g is not None
        local_idx = torch.arange(start, end, device=self.device, dtype=torch.long)
        B = int(local_idx.numel())
        pid = g["site_protein"][local_idx].long()
        pos = g["site_position"][local_idx].long()
        gav = g["site_graph_available"][local_idx].long()
        seq_len = g["protein_seq_length"][pid].long()
        struct_len = g["protein_struct_length"][pid].long()
        s_off = g["protein_seq_offsets"][pid].long()
        st_off = g["protein_struct_offsets"][pid].long()
        flags = g["protein_flags"][pid].long()

        rel = pos[:, None] - 1 - HALF + torch.arange(WINDOW, device=self.device)[None, :]
        valid_seq = (rel >= 0) & (rel < seq_len[:, None])
        valid_struct = (rel >= 0) & (rel < struct_len[:, None])
        seq_max = int(g["protein_seq_features"].shape[0]) - 1
        struct_max = int(g["protein_struct_features"].shape[0]) - 1
        seq_global = (s_off[:, None] + rel.clamp(min=0, max=seq_max)).clamp(max=seq_max)
        struct_global = (st_off[:, None] + rel.clamp(min=0, max=struct_max)).clamp(max=struct_max)
        seq_win = g["protein_seq_features"][seq_global].float() * valid_seq.unsqueeze(-1).float()
        struct_win = g["protein_struct_features"][struct_global].float() * valid_struct.unsqueeze(-1).float()
        struct_available = ((flags[:, 1] > 0) & (pos >= 1) & (pos <= seq_len) & (struct_len > 0))
        struct_win[~struct_available, :, -1] = 1.0

        seq_global_feat = g["protein_seq_mean"][pid].float()
        struct_global_feat = g["protein_struct_mean"][pid].float()
        global_available = flags[:, 2].float()
        struct_global_available = ((flags[:, 3] > 0) & (struct_len > 0)).float()

        n0 = g["graph_node_offsets"][local_idx].long()
        n1 = g["graph_node_offsets"][local_idx + 1].long()
        e0 = g["graph_edge_offsets"][local_idx].long()
        e1 = g["graph_edge_offsets"][local_idx + 1].long()
        nc = torch.where(gav > 0, n1 - n0, torch.ones_like(n0))
        ec = torch.where(gav > 0, e1 - e0, torch.zeros_like(e0))
        node_site = torch.repeat_interleave(torch.arange(B, device=self.device), nc)
        node_start = torch.cumsum(nc, 0) - nc
        node_local = torch.arange(int(nc.sum()), device=self.device) - node_start[node_site]
        avail_node = gav[node_site] > 0
        node_global = torch.where(avail_node, n0[node_site] + node_local, torch.full_like(node_local, -1))
        N = node_global.numel()
        x = torch.zeros((N, int(g["graph_node_features"].shape[1])), device=self.device, dtype=torch.float32)
        if bool(avail_node.any()):
            x[avail_node] = g["graph_node_features"][node_global[avail_node]].float()
        esm_idx = torch.full((N,), -1, device=self.device, dtype=torch.long)
        if bool(avail_node.any()):
            esm_idx[avail_node] = g["graph_node_esm_index"][node_global[avail_node]].long()
        esm_global = st_off[node_site] + esm_idx
        valid_esm = (esm_idx >= 0) & (esm_idx < struct_len[node_site])
        esm_if = torch.zeros((N, int(g["protein_struct_features"].shape[1])), device=self.device, dtype=torch.float32)
        if bool(valid_esm.any()):
            esm_if[valid_esm] = g["protein_struct_features"][esm_global[valid_esm]].float()
        esm_mask = valid_esm.float()

        node_pos = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        token_index = torch.full((N,), -1, device=self.device, dtype=torch.long)
        if bool(avail_node.any()):
            node_pos[avail_node] = g["graph_node_positions"][node_global[avail_node]].float()
            token_index[avail_node] = g["graph_token_index"][node_global[avail_node]].long()

        ci = g["site_center_index"][local_idx].long()
        center_mask = torch.zeros((N,), device=self.device, dtype=torch.float32)
        if bool(avail_node.any()):
            site_of_node = node_site[avail_node]
            ci_ok = (ci[site_of_node] >= 0) & (ci[site_of_node] < nc[site_of_node])
            center_mask[avail_node] = ((node_local[avail_node] == ci[site_of_node]) & ci_ok).float()
        center_mask[~avail_node] = 1.0

        edge_site = torch.repeat_interleave(torch.arange(B, device=self.device), ec)
        edge_start = torch.cumsum(ec, 0) - ec
        edge_local = torch.arange(int(ec.sum()), device=self.device) - edge_start[edge_site]
        edge_global = e0[edge_site] + edge_local
        if edge_global.numel() > 0:
            edge_index = g["graph_edge_index"][:, edge_global].long() + node_start[edge_site][None, :]
            edge_attr = g["graph_edge_attr"][edge_global].float()
        else:
            edge_index = torch.zeros((2, 0), device=self.device, dtype=torch.long)
            edge_attr = torch.zeros((0, int(g["graph_edge_attr"].shape[1])), device=self.device, dtype=torch.float32)

        site_start = int(self.shard_meta[sid]["site_start"])
        gidx = site_start + local_idx
        labels = self.labels_gpu[gidx].float()
        # State ids: unknown=0, positive=1, observed_negative=2, reliable_negative=3.
        # Known negatives MUST map to 2 (or 3), never 1: the state-aware ASL applies
        # the positive term to state==1, so the old 0->1 mapping pushed every known
        # negative up instead of down.
        label_state = torch.where(
            labels < 0,
            torch.zeros_like(labels, dtype=torch.long),
            torch.where(
                labels > 0,
                torch.ones_like(labels, dtype=torch.long),
                torch.full_like(labels, 2, dtype=torch.long),
            ),
        )
        micro = g["site_micro_env"][local_idx].float()
        micro_available = g["site_micro_env_available"][local_idx].float()
        residue_types = g["_residue_types"][local_idx]
        position_ids = torch.arange(WINDOW, device=self.device, dtype=torch.long)
        rel_offsets = position_ids - HALF
        site_indicator = F.one_hot(torch.tensor(HALF, device=self.device), num_classes=WINDOW).float()
        ones8 = torch.ones((B, 8), device=self.device, dtype=torch.float32)
        zeros8 = torch.zeros((B, 8), device=self.device, dtype=torch.float32)

        graph = {
            "x": x, "pos": node_pos, "esm_if": esm_if, "esm_if_mask": esm_mask,
            "center_mask": center_mask, "edge_index": edge_index, "edge_attr": edge_attr,
            "batch_index": node_site, "token_index": token_index,
        }
        return {
            "sequence": {
                "window_features": seq_win,
                "local_features": seq_win[:, 21:52],
                "global_features": seq_global_feat,
                "global_context_available": global_available,
                "residue_types": residue_types,
                "position_ids": position_ids.unsqueeze(0).expand(B, -1),
                "relative_position_offsets": rel_offsets.unsqueeze(0).expand(B, -1),
                "site_indicator": site_indicator.unsqueeze(0).expand(B, -1),
            },
            "structure": struct_win,
            "structure_global_features": struct_global_feat,
            "structure_available": struct_available.float(),
            "structure_global_available": struct_global_available,
            "micro_env_features": micro,
            "micro_env_available": micro_available,
            "structure_graph_available": gav.float(),
            "structure_graph": graph,
            "motif_bias": zeros8,
            "label": labels,
            "loss_weight": ones8,
            "label_state": label_state,
            "label_reliability": ones8,
            "teacher_probs": -ones8,
            "teacher_mask": zeros8,
            "teacher_weight": ones8,
            "sample_indices": gidx.cpu().tolist(),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        specs = self._specs_for_epoch()
        next_sid_by_sid = {}
        for i, (sid, _, _) in enumerate(specs):
            if i + 1 < len(specs) and specs[i + 1][0] != sid:
                next_sid_by_sid[sid] = specs[i + 1][0]
        if self._prefetch_future is not None:
            try:
                self._prefetch_future.result()
            except Exception:
                pass
            self._prefetch_future = None
            self._prefetch_sid = None
        current = None
        for sid, start, end in specs:
            if sid != current:
                if self._prefetch_future is not None and self._prefetch_sid == sid:
                    self._g = self._prefetch_future.result()
                    self._g_sid = sid
                else:
                    self._load_shard(sid)
                self._prefetch_future = None
                self._prefetch_sid = None
                current = sid
                nxt = next_sid_by_sid.get(sid)
                if self.prefetch_shards and self._executor is not None and nxt is not None:
                    self._prefetch_sid = nxt
                    self._prefetch_future = self._executor.submit(self._load_shard_arrays, nxt, True)
            yield self._build_batch(sid, start, end)
        if self._prefetch_future is not None:
            try:
                self._prefetch_future.result()
            except Exception:
                pass
            self._prefetch_future = None
            self._prefetch_sid = None
