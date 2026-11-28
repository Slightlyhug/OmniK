#!/usr/bin/env python3
"""Write a portable runtime config using absolute local paths."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("configs/local.json"))
    ap.add_argument(
        "--prior", type=Path, help="Training-only PSSM prior; required for training"
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        help="Inference: obtain trained position bias from checkpoint instead of a prior",
    )
    args = ap.parse_args()
    if bool(args.prior) == bool(args.checkpoint):
        ap.error(
            "Provide exactly one of --prior (training) or --checkpoint (inference)"
        )
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/generated/site_feature_full_v3/config.json").read_text()
    )
    feature = args.feature_root.resolve()
    output = args.output_root.resolve()
    if args.checkpoint:
        import numpy as np
        import torch

        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        prior = ck["model_state_dict"][
            "base_model.learnable_task_attention.pos_bias"
        ].numpy()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prior_path = args.out.with_suffix(".position_bias.npy").resolve()
        np.save(prior_path, prior)
    else:
        prior_path = args.prior.resolve()
        if not prior_path.is_file():
            ap.error(f"Prior does not exist: {prior_path}")
    replacements = {
        "${PTM_DATA_ROOT}": str(args.data_dir.resolve()),
        "${PTM_FEATURE_ROOT}": str(feature),
        "${PTM_OUTPUT_ROOT}": str(output),
    }

    def replace(v):
        if isinstance(v, str):
            for old, new in replacements.items():
                v = v.replace(old, new)
            return v
        if isinstance(v, dict):
            return {k: replace(x) for k, x in v.items()}
        if isinstance(v, list):
            return [replace(x) for x in v]
        return v

    config = replace(config)
    for split in ("train", "val", "test"):
        config[f"{split}_data_path"] = str(args.data_dir.resolve() / f"{split}.csv")
    config["feature_store_v2_root"] = str(feature / "feature_store")
    config["graph_feature_dir"] = str(feature / "graphs")
    config["learnable_task_attention_pos_prior_path"] = str(prior_path)
    config["M6_dual_structure_graph"]["learnable_task_attention_pos_prior_path"] = str(
        prior_path
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
