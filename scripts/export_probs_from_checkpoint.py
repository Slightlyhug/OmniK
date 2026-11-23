#!/usr/bin/env python3
"""Export per-site probabilities from a trained checkpoint for one split."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "feature_store"))

from src.models.multitask_heads import build_model  # noqa: E402
from src.utils.training_data import to_device  # noqa: E402
from gpu_shard_loader import GpuShardBatchLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store", default=None, help="Feature-store root; required unless PTM_FEATURE_STORE_ROOT is set")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    store = args.store or os.environ.get("PTM_FEATURE_STORE_ROOT")
    if not store:
        ap.error("--store or PTM_FEATURE_STORE_ROOT is required")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    config = json.loads(Path(args.config).read_text())
    config["use_torch_compile"] = False
    ptm_types = config["multitask_ptm_types"]
    model = build_model(config, num_tasks=len(ptm_types)).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    print("missing", len(missing), "unexpected", len(unexpected), flush=True)
    if missing:
        print("missing sample", missing[:10], flush=True)
    if unexpected:
        print("unexpected sample", unexpected[:10], flush=True)
    model.eval()
    loader = GpuShardBatchLoader(
        store, args.split, args.batch_size, device,
        seed=42, rank=0, world_size=1, shuffle=False,
        prefetch_shards=True,
    )
    probs, indices, labels = [], [], []
    amp_dtype = torch.bfloat16
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch_dev = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                out = model(batch_dev)
                logits = out["logits"] if isinstance(out, dict) else out
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            indices.extend(int(x) for x in batch["sample_indices"])
            labels.append(batch_dev["label"].float().cpu().numpy())
            if (step + 1) % 100 == 0:
                print("step", step + 1, "samples", len(indices), flush=True)
    p = np.concatenate(probs, 0).astype(np.float32)
    y = np.concatenate(labels, 0).astype(np.float32)
    idx = np.asarray(indices, dtype=np.int64)
    np.save(args.out_prefix + "_probabilities.npy", p)
    np.save(args.out_prefix + "_labels.npy", y)
    np.save(args.out_prefix + "_indices.npy", idx)
    print("saved", p.shape, y.shape, idx.shape, flush=True)


if __name__ == "__main__":
    main()
