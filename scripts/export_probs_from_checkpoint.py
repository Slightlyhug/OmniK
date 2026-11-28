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
from src.utils.training_config import normalize_and_validate_config  # noqa: E402
from gpu_shard_loader import GpuShardBatchLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--store",
        default=None,
        help="Feature-store root; required unless PTM_FEATURE_STORE_ROOT is set",
    )
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    store = args.store or os.environ.get("PTM_FEATURE_STORE_ROOT")
    if not store:
        ap.error("--store or PTM_FEATURE_STORE_ROOT is required")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    config_path = Path(args.config).resolve()
    config = normalize_and_validate_config(
        json.loads(config_path.read_text()), config_path
    )
    config["use_torch_compile"] = False
    ptm_types = config["multitask_ptm_types"]
    model = build_model(config, num_tasks=len(ptm_types)).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    loader = GpuShardBatchLoader(
        store,
        args.split,
        args.batch_size,
        device,
        seed=42,
        rank=0,
        world_size=1,
        shuffle=False,
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
    if not probs:
        raise ValueError(f"No samples found in feature-store split {args.split}")
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    p = np.concatenate(probs, 0).astype(np.float32)
    y = np.concatenate(labels, 0).astype(np.float32)
    idx = np.asarray(indices, dtype=np.int64)
    np.save(args.out_prefix + "_probabilities.npy", p)
    np.save(args.out_prefix + "_labels.npy", y)
    np.save(args.out_prefix + "_indices.npy", idx)
    Path(args.out_prefix + "_metadata.json").write_text(
        json.dumps(
            {
                "tasks": ptm_types,
                "split": args.split,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "probability_transform": "sigmoid (uncalibrated)",
                "index_reference": str(
                    Path(store).resolve() / args.split / "global/site_keys.txt"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("saved", p.shape, y.shape, idx.shape, flush=True)


if __name__ == "__main__":
    main()
