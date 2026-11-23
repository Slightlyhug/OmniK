#!/usr/bin/env python3
"""Packed-batch LMDB adapter for train_multitask_lysine.py."""

import json
import pickle
import random
import zlib
from pathlib import Path

import h5py
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import train_multitask_lysine as base


def _select_batch_rows(value, row_indices, original_size):
    """Select sample rows from a recursively collated packed batch."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == original_size:
            return value.index_select(0, row_indices)
        return value
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and int(value.shape[0]) == original_size:
            return value[row_indices.cpu().numpy()]
        return value
    if (
        isinstance(value, dict)
        and {"x", "edge_index", "batch_index"}.issubset(value)
        and isinstance(value["x"], torch.Tensor)
        and isinstance(value["edge_index"], torch.Tensor)
        and isinstance(value["batch_index"], torch.Tensor)
    ):
        graph_batch = value["batch_index"].long()
        graph_map = torch.full(
            (original_size,), -1, dtype=torch.long, device=graph_batch.device
        )
        graph_map[row_indices.to(graph_batch.device)] = torch.arange(
            row_indices.numel(), dtype=torch.long, device=graph_batch.device
        )
        node_keep = graph_map[graph_batch] >= 0
        node_indices = torch.nonzero(node_keep, as_tuple=False).flatten()
        node_map = torch.full(
            (value["x"].shape[0],), -1, dtype=torch.long, device=graph_batch.device
        )
        node_map[node_indices] = torch.arange(
            node_indices.numel(), dtype=torch.long, device=graph_batch.device
        )
        edge_index = value["edge_index"].long()
        edge_keep = node_keep[edge_index[0]] & node_keep[edge_index[1]]
        selected = dict(value)
        for node_key, node_value in value.items():
            if (
                isinstance(node_value, torch.Tensor)
                and node_value.ndim > 0
                and int(node_value.shape[0]) == int(value["x"].shape[0])
            ):
                selected[node_key] = node_value.index_select(0, node_indices)
        selected["batch_index"] = graph_map[graph_batch[node_keep]]
        selected["edge_index"] = node_map[edge_index[:, edge_keep]]
        edge_attr = value.get("edge_attr")
        if isinstance(edge_attr, torch.Tensor):
            selected["edge_attr"] = edge_attr[edge_keep]
        return selected
    if isinstance(value, dict):
        return {
            key: _select_batch_rows(item, row_indices, original_size)
            for key, item in value.items()
        }
    if isinstance(value, list) and len(value) == original_size:
        return [value[int(index)] for index in row_indices.tolist()]
    if isinstance(value, tuple) and len(value) == original_size:
        return tuple(value[int(index)] for index in row_indices.tolist())
    # PyG Batch objects need graph-aware slicing; tensor slicing would leave
    # edge indices and batch assignments inconsistent.
    if hasattr(value, "to_data_list") and hasattr(value.__class__, "from_data_list"):
        data_list = value.to_data_list()
        if len(data_list) == original_size:
            return value.__class__.from_data_list(
                [data_list[int(index)] for index in row_indices.tolist()]
            )
    return value


class PackedTensorLMDBDataset(Dataset):
    packed_batch_dataset = True

    def __init__(self, path, label_overlay_path=None, prune_all_unknown=False,
                 dense_fill_lmdb=None, dense_fill_mask=None, structure_rebuild_lmdb=None,
                 site_graph_h5=None):
        # Optional rebuilt structure store (data/structure_rebuild_v1). When set,
        # it replaces every structure field and the dense fill is not used.
        self.structure_rebuild_lmdb = str(structure_rebuild_lmdb) if structure_rebuild_lmdb else None
        self.site_graph_h5 = str(site_graph_h5) if site_graph_h5 else None
        self._site_graph_handle = None
        self._rb_env = None
        self._rb_txn = None
        # Optional side store with re-extracted ESM-IF dense windows for rows the
        # packed store lacks. Only rows set in dense_fill_mask (bool over logical
        # indices, written by the overlay builder after verification) are filled.
        self.dense_fill_lmdb = str(dense_fill_lmdb) if dense_fill_lmdb else None
        self.dense_fill_mask_path = str(dense_fill_mask) if dense_fill_mask else None
        self._fill_mask = None
        self._fill_env = None
        self._fill_txn = None
        self.path = str(path)
        meta_path = Path(f"{self.path}.meta.json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("format") != "packed_tensor_lmdb_zlib_v1":
            raise ValueError(f"Unsupported packed LMDB format: {self.meta.get('format')}")
        self.num_batches = int(self.meta["num_batches"])
        self.sample_count = int(self.meta["length"])
        self.batch_size = int(self.meta["batch_size"])
        self.local_window_size = int(self.meta.get("local_window_size", 31))
        self.labels_path = str(self.meta["labels_path"])
        self.label_overlay_path = str(label_overlay_path) if label_overlay_path else None
        self.prune_all_unknown = bool(prune_all_unknown)
        self._labels = None
        self._env = None
        self._txn = None

    def _load_site_graphs(self, batch):
        if not self.site_graph_h5:
            return batch
        if self._site_graph_handle is None:
            self._site_graph_handle = h5py.File(
                self.site_graph_h5, "r", libver="latest", swmr=True,
                rdcc_nbytes=32 * 1024 * 1024, rdcc_nslots=65537,
            )
        uids, positions = batch.get("uniprot_ids", []), batch.get("positions", [])
        if len(uids) != len(positions):
            raise ValueError("Packed batch UID/position lengths differ; cannot align site graphs")
        xs, poss, eis, eas, esm, esm_masks, centers, bis = [], [], [], [], [], [], [], []
        graph_available = torch.zeros(len(uids), dtype=torch.float32)
        node_offset = 0
        for row, (uid, position) in enumerate(zip(uids, positions)):
            key = f"{str(uid).strip()}_{int(position)}"
            group = self._site_graph_handle.get(key)
            if group is None:
                # Keep a node for every packed row so graph pooling retains batch order.
                xs.append(torch.zeros((1, 34), dtype=torch.float32))
                poss.append(torch.zeros((1, 3), dtype=torch.float32))
                esm.append(torch.zeros((1, 512), dtype=torch.float32))
                esm_masks.append(torch.zeros((1,), dtype=torch.float32))
                centers.append(torch.ones((1,), dtype=torch.bool))
                bis.append(torch.full((1,), row, dtype=torch.long))
                node_offset += 1
                continue
            x = torch.from_numpy(np.asarray(group["x"], dtype=np.float32))
            pos = torch.from_numpy(np.asarray(group["pos"], dtype=np.float32))
            edge_index = torch.from_numpy(np.asarray(group["edge_index"], dtype=np.int64))
            edge_attr = torch.from_numpy(np.asarray(group["edge_attr"], dtype=np.float32))
            if "esm_if" in group:
                esm_if = torch.from_numpy(np.asarray(group["esm_if"], dtype=np.float32))
                esm_mask = torch.from_numpy(np.asarray(group.get("esm_if_mask", np.ones(x.shape[0])), dtype=np.float32)).reshape(-1)
            else:
                esm_if = torch.zeros((x.shape[0], 512), dtype=torch.float32)
                esm_mask = torch.zeros((x.shape[0],), dtype=torch.float32)
            center_index = int(json.loads(group.attrs.get("meta", "{}")).get("center_index", -1))
            if not 0 <= center_index < x.shape[0]:
                raise ValueError(f"Missing/invalid center_index in site graph {key}")
            center_mask = torch.zeros((x.shape[0],), dtype=torch.bool)
            center_mask[center_index] = True
            if x.ndim != 2 or x.shape[1] != 34 or pos.shape != (x.shape[0], 3):
                raise ValueError(f"Invalid site graph schema for {key}: x={tuple(x.shape)}, pos={tuple(pos.shape)}")
            if esm_if.shape != (x.shape[0], 512) or esm_mask.numel() != x.shape[0]:
                raise ValueError(f"Invalid ESM-IF schema for {key}: features={tuple(esm_if.shape)}, mask={tuple(esm_mask.shape)}")
            xs.append(x)
            poss.append(pos)
            esm.append(esm_if)
            esm_masks.append(esm_mask)
            centers.append(center_mask)
            bis.append(torch.full((x.shape[0],), row, dtype=torch.long))
            if edge_index.numel():
                eis.append(edge_index + node_offset)
                eas.append(edge_attr)
            node_offset += x.shape[0]
            graph_available[row] = 1.0
        batch["structure_graph"] = {
            "x": torch.cat(xs, dim=0),
            "pos": torch.cat(poss, dim=0),
            "esm_if": torch.cat(esm, dim=0),
            "esm_if_mask": torch.cat(esm_masks, dim=0),
            "center_mask": torch.cat(centers, dim=0),
            "edge_index": torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.cat(eas, dim=0) if eas else torch.zeros((0, 16), dtype=torch.float32),
            "batch_index": torch.cat(bis, dim=0),
        }
        batch["structure_graph_available"] = graph_available
        batch["structure_available"] = graph_available.clone()
        batch["structure_global_available"] = torch.zeros_like(graph_available)
        batch["micro_env_available"] = torch.zeros_like(graph_available)
        return batch

    def __len__(self):
        return self.num_batches

    def get_label_matrix(self):
        if self._labels is None:
            path = self.label_overlay_path or self.labels_path
            self._labels = np.load(path, mmap_mode="r")
            if self._labels.shape != (self.sample_count, 8):
                raise ValueError(
                    f"Label matrix shape mismatch for {path}: "
                    f"{self._labels.shape} != {(self.sample_count, 8)}"
                )
        return self._labels

    def _open(self):
        if self._env is None:
            self._env = lmdb.open(
                self.path, readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=2048,
            )
            self._txn = self._env.begin(buffers=True)
        return self._txn

    def _apply_dense_fill(self, batch):
        if self._fill_mask is None:
            self._fill_mask = np.load(self.dense_fill_mask_path, mmap_mode="r")
            if self._fill_mask.shape != (self.sample_count,):
                raise ValueError(f"dense fill mask shape {self._fill_mask.shape} != ({self.sample_count},)")
        logical = np.asarray(batch["sample_indices"], dtype=np.int64)
        avail = batch["structure_available"]
        filled = torch.zeros(len(logical), dtype=torch.bool)
        rows = np.flatnonzero(np.asarray(self._fill_mask[logical]) & (avail.numpy() <= 0))
        if len(rows):
            if self._fill_env is None:
                self._fill_env = lmdb.open(self.dense_fill_lmdb, readonly=True, lock=False,
                                           readahead=False, meminit=False, max_readers=2048)
                self._fill_txn = self._fill_env.begin(buffers=True)
            dense = batch["structure"]
            glob = batch.get("structure_global_features")
            gavail = batch.get("structure_global_available")
            for j in rows:
                raw = self._fill_txn.get(f"{int(logical[j]):09d}".encode())
                if raw is None:
                    raise KeyError(f"dense fill mask set but record missing for logical row {int(logical[j])}")
                rec = pickle.loads(zlib.decompress(raw))
                dense[j] = torch.from_numpy(np.asarray(rec["w"])).to(dense.dtype)
                if isinstance(glob, torch.Tensor):
                    glob[j] = torch.from_numpy(np.asarray(rec["g"])).to(glob.dtype)
                if isinstance(gavail, torch.Tensor):
                    gavail[j] = 1.0
                avail[j] = 1.0
                filled[j] = True
        batch["dense_filled"] = filled
        return batch

    def _apply_structure_rebuild(self, batch):
        """Replace ALL structure-derived fields with the rebuilt store.

        Rows with a rebuilt record get its dense window, global vector,
        micro-environment and pocket graph. Rows without one are marked
        unavailable (zeros, availability 0) -- no value from the old packed
        structure fields is kept, because those mix structure versions and
        normalisations.
        """
        if self._rb_env is None:
            self._rb_env = lmdb.open(self.structure_rebuild_lmdb, readonly=True, lock=False,
                                     readahead=False, meminit=False, max_readers=2048)
            self._rb_txn = self._rb_env.begin(buffers=True)
        logical = np.asarray(batch["sample_indices"], dtype=np.int64)
        n = len(logical)
        dense = batch["structure"]
        glob = batch["structure_global_features"]
        micro = batch["micro_env_features"]
        old_g = batch["structure_graph"]
        x_dt, ea_dt = old_g["x"].dtype, old_g["edge_attr"].dtype
        ei_dt, bi_dt = old_g["edge_index"].dtype, old_g["batch_index"].dtype
        node_dim, edge_dim = int(old_g["x"].shape[1]), int(old_g["edge_attr"].shape[1])
        dense.zero_()
        glob.zero_()
        micro.zero_()
        avail = torch.zeros(n, dtype=torch.float32)
        xs, eis, eas, bis = [], [], [], []
        offset = 0
        for j in range(n):
            raw = self._rb_txn.get(f"{int(logical[j]):09d}".encode())
            if raw is None:
                # placeholder graph (one all-zero node), same as the old store
                xs.append(torch.zeros((1, node_dim), dtype=x_dt))
                bis.append(torch.full((1,), j, dtype=bi_dt))
                offset += 1
                continue
            rec = pickle.loads(zlib.decompress(raw))
            dense[j] = torch.from_numpy(np.asarray(rec["structure"])).to(dense.dtype)
            glob[j] = torch.from_numpy(np.asarray(rec["structure_global_features"])).to(glob.dtype)
            micro[j] = torch.from_numpy(np.asarray(rec["micro_env_features"])).to(micro.dtype)
            gx = torch.from_numpy(np.asarray(rec["graph_x"])).to(x_dt)
            gei = torch.from_numpy(np.asarray(rec["graph_ei"]).astype(np.int64)).to(ei_dt)
            gea = torch.from_numpy(np.asarray(rec["graph_ea"])).to(ea_dt)
            xs.append(gx)
            eis.append(gei + offset)
            eas.append(gea)
            bis.append(torch.full((gx.shape[0],), j, dtype=bi_dt))
            offset += gx.shape[0]
            avail[j] = 1.0
        batch["structure_graph"] = {
            "x": torch.cat(xs, 0),
            "edge_index": torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=ei_dt),
            "edge_attr": torch.cat(eas, 0) if eas else torch.zeros((0, edge_dim), dtype=ea_dt),
            "batch_index": torch.cat(bis, 0),
        }
        for key in ("structure_available", "structure_global_available",
                    "micro_env_available", "structure_graph_available"):
            if isinstance(batch.get(key), torch.Tensor):
                batch[key] = avail.to(batch[key].dtype).clone()
            else:
                batch[key] = avail.clone()
        batch["structure_rebuilt"] = avail.bool()
        return batch

    def __getitem__(self, index):
        compressed = self._open().get(f"{int(index):09d}".encode())
        if compressed is None:
            raise IndexError(index)
        batch = pickle.loads(zlib.decompress(compressed))
        batch = self._load_site_graphs(batch)
        if self.structure_rebuild_lmdb:
            batch = self._apply_structure_rebuild(batch)
        elif self.dense_fill_lmdb and self.dense_fill_mask_path:
            batch = self._apply_dense_fill(batch)
        if self.label_overlay_path:
            logical_indices = np.asarray(batch["sample_indices"], dtype=np.int64)
            labels = np.asarray(self.get_label_matrix()[logical_indices], dtype=np.float32)
            if self.prune_all_unknown:
                keep = np.any(labels >= 0.0, axis=1)
                row_indices = torch.from_numpy(np.flatnonzero(keep).astype(np.int64))
                if row_indices.numel() == 0:
                    raise RuntimeError(f"Packed batch {index} contains no supervised rows")
                batch = _select_batch_rows(batch, row_indices, len(logical_indices))
                logical_indices = logical_indices[keep]
                labels = labels[keep]
        sequence = batch["sequence"]
        window = sequence["window_features"]
        window_size = int(window.shape[1])
        local_start = (window_size - self.local_window_size) // 2
        sequence["local_features"] = window[:, local_start:local_start + self.local_window_size]
        positions = torch.arange(window_size, dtype=torch.long)
        center = window_size // 2
        sequence["position_ids"] = positions.repeat(window.shape[0], 1)
        sequence["relative_position_offsets"] = (positions - center).repeat(window.shape[0], 1)
        site = torch.zeros(window_size, dtype=torch.float32)
        site[center] = 1.0
        sequence["site_indicator"] = site.repeat(window.shape[0], 1)
        if self.label_overlay_path:
            label_tensor = torch.from_numpy(labels.copy())
            known = label_tensor >= 0.0
            state = torch.zeros_like(label_tensor, dtype=torch.long)
            state[label_tensor > 0.0] = 1
            # Cleaned negatives are deliberately selected high-confidence negatives.
            state[label_tensor == 0.0] = 3
            batch["label"] = label_tensor
            batch["label_state"] = state
            batch["loss_weight"] = known.to(torch.float32)
            batch["label_reliability"] = known.to(torch.float32)
            batch["known_task_count"] = known.sum(dim=1).tolist()
            batch["positive_task_count"] = (label_tensor == 1.0).sum(dim=1).tolist()
            batch["negative_task_count"] = (label_tensor == 0.0).sum(dim=1).tolist()
            batch["unknown_task_count"] = (label_tensor < 0.0).sum(dim=1).tolist()
        graph = batch.get("structure_graph")
        if (
            isinstance(graph, dict)
            and isinstance(graph.get("x"), torch.Tensor)
            and isinstance(graph.get("batch_index"), torch.Tensor)
        ):
            graph_batch = graph["batch_index"].long()
            graph_count = int(batch["label"].shape[0])
            graph_has_payload = torch.zeros(graph_count, dtype=torch.bool)
            node_has_payload = torch.any(graph["x"] != 0, dim=1)
            if node_has_payload.any():
                graph_has_payload[graph_batch[node_has_payload]] = True
            # The historical packed store contains placeholder all-zero graphs
            # whose availability flag was incorrectly set. Trust payload truth.
            batch["structure_graph_available"] = graph_has_payload.to(torch.float32)
        return batch

    def get_exemplar_features(self, model, sample_ids, device, feature_dim):
        """Encode selected logical rows without confusing them with packed-batch indices."""
        from src.utils.training_data import to_device

        wanted = {int(sample_id) for sample_id in sample_ids}
        grouped = {}
        for logical_index in wanted:
            batch_index = logical_index // self.batch_size
            if not 0 <= batch_index < len(self):
                continue
            grouped.setdefault(batch_index, set()).add(logical_index)

        was_training = bool(model.training)
        model.eval()
        result = {}
        try:
            with torch.no_grad():
                for batch_index, logical_indices in grouped.items():
                    packed_batch = self[batch_index]
                    row_indices = [
                        row for row, logical in enumerate(packed_batch["sample_indices"])
                        if int(logical) in logical_indices
                    ]
                    if not row_indices:
                        continue
                    batch_size = int(packed_batch["label"].shape[0])
                    selected_batch = _select_batch_rows(
                        packed_batch,
                        torch.tensor(row_indices, dtype=torch.long),
                        batch_size,
                    )
                    outputs = model(to_device(selected_batch, device))
                    features = outputs.get("shared_representation") if isinstance(outputs, dict) else None
                    if not isinstance(features, torch.Tensor) or features.ndim != 2:
                        raise RuntimeError("Packed UTA re-encoding requires shared_representation [B, D].")
                    if int(features.shape[1]) != int(feature_dim):
                        raise RuntimeError(
                            f"Packed UTA feature dimension {features.shape[1]} does not match {feature_dim}."
                        )
                    for row, logical in enumerate(selected_batch["sample_indices"]):
                        result[str(int(logical))] = features[row].detach().cpu().float()
        finally:
            if was_training:
                model.train()
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        state["_labels"] = None
        state["_fill_env"] = None
        state["_fill_txn"] = None
        state["_fill_mask"] = None
        state["_rb_env"] = None
        state["_rb_txn"] = None
        state["_site_graph_handle"] = None
        return state

    def __del__(self):
        if getattr(self, "_txn", None) is not None:
            try:
                self._txn.abort()
            except Exception:
                pass
        if getattr(self, "_env", None) is not None:
            try:
                self._env.close()
            except Exception:
                pass
        graph_handle = getattr(self, "_site_graph_handle", None)
        if graph_handle is not None:
            try:
                graph_handle.close()
            except Exception:
                pass


class PackedBlockShuffleSampler(Sampler):
    def __init__(self, length, block_batches=8, seed=42):
        self.length = int(length)
        self.block_batches = max(1, int(block_batches))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        blocks = list(range((self.length + self.block_batches - 1) // self.block_batches))
        random.Random(self.seed + self.epoch).shuffle(blocks)
        for block in blocks:
            first = block * self.block_batches
            yield from range(first, min(self.length, first + self.block_batches))


def identity_collate(item):
    return item


original_build_dataset_bundle = base.build_dataset_bundle
original_build_loader_bundle = base.build_loader_bundle
original_multilabel_collate_fn = base.multilabel_collate_fn


def packed_aware_collate(items):
    # The base model smoke test wraps train_set[0] in a one-element list and
    # collates it again.  Packed datasets already return a complete batch.
    if (
        isinstance(items, list)
        and len(items) == 1
        and isinstance(items[0], dict)
        and "uniprot_ids" in items[0]
        and "sample_indices" in items[0]
    ):
        # The base smoke-test contract expects exactly one sample. Select one
        # complete row, including a graph-safe node/edge remap.
        packed_batch = items[0]
        batch_size = int(packed_batch["label"].shape[0])
        return _select_batch_rows(
            packed_batch, torch.tensor([0], dtype=torch.long), batch_size
        )
    return original_multilabel_collate_fn(items)


def build_dataset_bundle(config, artifacts, profiler):
    packed = config.get("packed_tensor_lmdb")
    if not isinstance(packed, dict) or not bool(config.get("prefer_packed_tensor_lmdb", False)):
        return original_build_dataset_bundle(config, artifacts, profiler)
    overlays = config.get("packed_tensor_label_overlay", {})
    if not isinstance(overlays, dict):
        overlays = {}
    prune_all_unknown = bool(config.get("packed_tensor_prune_all_unknown", False))
    dense_fill = config.get("packed_tensor_dense_fill") or {}
    if bool(config.get("use_structure_graph", False)) and not bool(config.get("use_dense_structure_tower_with_graph", False)):
        dense_fill = {}
    rebuild = config.get("packed_tensor_structure_rebuild") or {}
    datasets = []
    for split in ("train", "val", "test"):
        path = packed.get(split)
        if not path or not Path(f"{path}.meta.json").exists():
            raise FileNotFoundError(f"Packed tensor LMDB is incomplete for {split}: {path}")
        overlay_path = overlays.get(split)
        if overlay_path and not Path(overlay_path).exists():
            raise FileNotFoundError(f"Packed label overlay is missing for {split}: {overlay_path}")
        fill = dense_fill.get(split) if isinstance(dense_fill, dict) else None
        fill_lmdb = fill.get("lmdb") if isinstance(fill, dict) else None
        fill_mask = fill.get("mask") if isinstance(fill, dict) else None
        if fill_lmdb or fill_mask:
            if not (fill_lmdb and fill_mask and Path(fill_lmdb).exists() and Path(fill_mask).exists()):
                raise FileNotFoundError(f"Dense fill for {split} incomplete: lmdb={fill_lmdb} mask={fill_mask}")
        rb_path = rebuild.get(split) if isinstance(rebuild, dict) else None
        if rb_path and not Path(rb_path).exists():
            raise FileNotFoundError(f"Structure rebuild store missing for {split}: {rb_path}")
        graph_paths = config.get("site_graph_h5", {})
        graph_path = graph_paths.get(split) if isinstance(graph_paths, dict) else None
        if bool(config.get("use_structure_graph", False)):
            if not graph_path or not Path(graph_path).exists():
                raise FileNotFoundError(f"Site graph HDF5 missing for {split}: {graph_path}")
            graph_root = Path(graph_path).parent.parent
            esm_summary = graph_root / "esm_if_summary.json"
            if bool(config.get("require_site_graph_esm_if_complete", False)):
                if not esm_summary.exists():
                    raise RuntimeError(
                        f"ESM-IF graph annotation is not finalized ({esm_summary} missing); "
                        "wait for add_esm_if_to_graphs.py to finish before training."
                    )
                summary = json.loads(esm_summary.read_text(encoding="utf-8"))
                split_summary = summary.get(split, {})
                if float(split_summary.get("node_coverage", 0.0)) < 0.999:
                    raise RuntimeError(
                        f"ESM-IF coverage for {split} is incomplete: "
                        f"{split_summary.get('node_coverage', 0.0):.4f}"
                    )
        datasets.append(PackedTensorLMDBDataset(
            path,
            label_overlay_path=overlay_path,
            prune_all_unknown=prune_all_unknown,
            dense_fill_lmdb=fill_lmdb,
            dense_fill_mask=fill_mask,
            structure_rebuild_lmdb=rb_path,
            site_graph_h5=graph_path,
        ))
    return base.DatasetBundle(*datasets)


def build_loader_bundle(config, seed, ptm_types, datasets, logger, profiler):
    if not isinstance(datasets.train_set, PackedTensorLMDBDataset):
        return original_build_loader_bundle(config, seed, ptm_types, datasets, logger, profiler)
    requested_workers = int(config.get("dataloader_num_workers", 2))
    workers = base._effective_loader_workers(requested_workers, logger)
    prefetch = max(1, int(config.get("dataloader_prefetch_factor", 1)))
    persistent = bool(config.get("dataloader_persistent_workers", workers > 0))
    pin_memory = bool(config.get("pin_memory", True) and torch.cuda.is_available())
    kwargs = {
        "batch_size": None,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": identity_collate,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
        kwargs["persistent_workers"] = persistent

    block_samples = int(config.get("block_shuffle_size", 8192))
    block_batches = max(1, block_samples // datasets.train_set.batch_size)
    sampler = PackedBlockShuffleSampler(len(datasets.train_set), block_batches, seed)
    train_loader = profiler.run(
        "build_train_loader", lambda: DataLoader(datasets.train_set, sampler=sampler, **kwargs)
    )
    val_loader = profiler.run(
        "build_val_loader", lambda: DataLoader(datasets.val_set, shuffle=False, **kwargs)
    )
    test_loader = profiler.run(
        "build_test_loader", lambda: DataLoader(datasets.test_set, shuffle=False, **kwargs)
    )
    target_distribution = profiler.run(
        "compute_target_distribution",
        lambda: base._positive_label_distribution(datasets.train_set.get_label_matrix()),
    )
    monitor = base.BatchDistributionMonitor(
        ptm_types=ptm_types,
        target_distribution=target_distribution,
        log_interval=int(config.get("batch_distribution_log_interval", 50)),
        relative_tolerance=float(config.get("batch_distribution_relative_tolerance", 0.25)),
        min_abs_tolerance=float(config.get("batch_distribution_min_abs_tolerance", 0.02)),
        enable_batch_logging=bool(config.get("batch_distribution_step_log", False)),
    )
    logger.info(
        "Using packed tensor LMDB: train_batches=%d batch_size=%d block_batches=%d workers=%d prefetch=%d prune_all_unknown=%s",
        len(datasets.train_set), datasets.train_set.batch_size, block_batches, workers, prefetch,
        datasets.train_set.prune_all_unknown,
    )
    base.benchmark_loader_first_batch(
        train_loader, "train", bool(config.get("benchmark_dataloader_first_batch", False)), logger
    )
    return base.LoaderBundle(train_loader, val_loader, test_loader, monitor, False, sampler)


base.build_dataset_bundle = build_dataset_bundle
base.build_loader_bundle = build_loader_bundle
base.multilabel_collate_fn = packed_aware_collate


if __name__ == "__main__":
    base.main()
