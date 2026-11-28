#!/usr/bin/env python3
"""Run the bundled Q00169 FASTA/PDB through ESM extraction and OmniK inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "demo/inputs"
TASKS = [
    "ubiquitination",
    "sumoylation",
    "succinylation",
    "methylation",
    "crotonylation",
    "acetylation",
    "hydroxyisobutyrylation",
    "lactylation",
]


def verify_inputs():
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    provenance = json.loads((INPUTS / "provenance.json").read_text())
    for item in provenance["files"]:
        data = (INPUTS / item["path"]).read_bytes()
        if (
            len(data) != item["bytes"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"Bundled input failed checksum: {item['path']}")
    records = list(SeqIO.parse(INPUTS / "fasta/Q00169.fasta", "fasta"))
    if len(records) != 1:
        raise ValueError("Expected exactly one FASTA record")
    sequence = str(records[0].seq)
    model = next(
        PDBParser(QUIET=True)
        .get_structure("demo", INPUTS / "pdb/Q00169.pdb")
        .get_models()
    )
    residues = [r for r in model["A"] if r.id[0] == " "]
    if (
        len(list(model)) != 1
        or "".join(seq1(r.resname) for r in residues) != sequence
        or [r.id[1] for r in residues] != list(range(1, len(sequence) + 1))
        or not all(all(a in r for a in ("N", "CA", "C")) for r in residues)
    ):
        raise ValueError("FASTA/PDB sequence or backbone numbering mismatch")
    with (INPUTS / "test.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = [i + 1 for i, residue in enumerate(sequence) if residue == "K"]
    if [int(row["position"]) for row in rows] != expected:
        raise ValueError("Site CSV must contain each lysine exactly once")
    for row in rows:
        pos = int(row["position"])
        window = "".join(
            sequence[j] if 0 <= j < len(sequence) else "X"
            for j in range(pos - 37, pos + 36)
        )
        if (
            row["uniprot_id"] != "Q00169"
            or row["site_id"] != f"Q00169_{pos}"
            or row["residue"] != "K"
            or row["window_sequence"] != window
            or any(row[t] != "-1" for t in TASKS)
        ):
            raise ValueError(f"Invalid inference site: {row['site_id']}")
    if (
        len(sequence) != provenance["sequence_length"]
        or len(rows) != provenance["n_sites"]
    ):
        raise ValueError("Input count differs from provenance")
    print(
        f"Verified Q00169: {len(sequence)} residues, {len(rows)} lysine sites, exact FASTA/PDB match",
        flush=True,
    )
    return provenance


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print("Running:", " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    return round(time.monotonic() - started, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esm-model-dir", type=Path, default=ROOT / "weights/esm")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "weights/OmniK_dense0n_checkpoint_last.pt",
    )
    ap.add_argument("--dssp-bin", default="mkdssp")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runs/demo-Q00169")
    ap.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="Validate bundled files without models or a GPU",
    )
    ap.add_argument(
        "--compare-reference",
        action="store_true",
        help="Compare output with bundled reference predictions",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=0.01,
        help="Absolute probability tolerance for reference comparison",
    )
    args = ap.parse_args()
    provenance = verify_inputs()
    if args.verify_inputs_only:
        return

    import h5py
    import numpy as np
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        ap.error(
            "This demo requires a CUDA GPU with BF16 support; select a GPU with CUDA_VISIBLE_DEVICES"
        )
    dssp = shutil.which(args.dssp_bin)
    if not dssp:
        ap.error("mkdssp not found; install DSSP or pass --dssp-bin /path/to/mkdssp")
    checkpoint = args.checkpoint.resolve()
    esm_dir = args.esm_model_dir.resolve()
    esm2 = esm_dir / "esm2_t33_650M_UR50D.pt"
    esmif = esm_dir / "esm_if1_gvp4_t16_142M_UR50.pt"
    for file in (checkpoint, esm2, esmif):
        if not file.is_file():
            ap.error(
                f"Required weights not found: {file}; see demo/README.md for download links"
            )
    manifest = json.loads((ROOT / "docs/zenodo-manifest.json").read_text())
    released = next(
        f for f in manifest["files"] if f["name"] == "OmniK_dense0n_checkpoint_last.pt"
    )
    digest = hashlib.md5()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if (
        checkpoint.stat().st_size != released["size"]
        or "md5:" + digest.hexdigest() != released["checksum"]
    ):
        ap.error("Checkpoint differs from the versioned Zenodo demo checkpoint")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        ap.error(f"Output directory is not empty: {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    timings = {}
    timings["esm_extraction_seconds"] = run(
        "tools/extract_esm_features_optimized.py",
        "--fasta_dir",
        INPUTS / "fasta",
        "--pdb_dir",
        INPUTS / "pdb",
        "--sequence_model_path",
        esm2,
        "--structure_model_path",
        esmif,
        "--output_dir",
        output,
        "--mode",
        "both",
        "--num_gpus",
        1,
        "--batch_size",
        1,
        "--max_proteins_per_chunk",
        1,
        "--cache_dir",
        output / "cache",
        "--force_recompute",
        "--no_mixed_precision",
        "--output_format",
        "hdf5",
    )
    with h5py.File(output / "esm_features.h5", "r") as handle:
        group = handle["proteins/Q00169"]
        for name, width in (("sequence_features", 1280), ("structure_features", 512)):
            values = np.asarray(group[name])
            if (
                values.shape != (provenance["sequence_length"], width)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"Invalid extracted feature array: {name}, {values.shape}"
                )
    timings["feature_store_seconds"] = run(
        "scripts/build_feature_store.py",
        "--data-dir",
        INPUTS,
        "--esm-h5",
        output / "esm_features.h5",
        "--pdb-dir",
        INPUTS / "pdb",
        "--out",
        output / "feature_store",
        "--splits",
        "test",
        "--proteins-per-shard",
        1,
        "--dssp-bin",
        dssp,
    )
    store = output / "feature_store/test"
    store_meta = json.loads((store / "manifest.json").read_text())
    coverage = store_meta["coverage"]
    if (
        store_meta["n_sites"] != provenance["n_sites"]
        or coverage["graphs_available"] != provenance["n_sites"]
        or any(
            coverage[k]
            for k in (
                "missing_sequence",
                "missing_structure",
                "dssp_failed_proteins",
                "graph_esm_mapping_failed_proteins",
            )
        )
    ):
        raise ValueError(f"Demo requires complete feature coverage: {coverage}")
    timings["config_seconds"] = run(
        "scripts/prepare_config.py",
        "--data-dir",
        INPUTS,
        "--feature-root",
        output,
        "--checkpoint",
        checkpoint,
        "--out",
        output / "inference.json",
        "--output-root",
        output,
    )
    prefix = output / "predictions"
    timings["inference_seconds"] = run(
        "scripts/export_probs_from_checkpoint.py",
        "--config",
        output / "inference.json",
        "--checkpoint",
        checkpoint,
        "--store",
        output / "feature_store",
        "--split",
        "test",
        "--batch-size",
        4,
        "--out-prefix",
        prefix,
    )
    probs = np.load(str(prefix) + "_probabilities.npy")
    indices = np.load(str(prefix) + "_indices.npy")
    labels = np.load(str(prefix) + "_labels.npy")
    keys = (store / "global/site_keys.txt").read_text().splitlines()
    meta = json.loads(Path(str(prefix) + "_metadata.json").read_text())
    sys.path.insert(0, str(ROOT))
    from src.utils.training_config import canonical_ptm_name

    if [canonical_ptm_name(t) for t in meta["tasks"]] != [
        canonical_ptm_name(t) for t in TASKS
    ]:
        raise ValueError("Prediction task order differs from demo task order")
    if (
        probs.shape != (len(keys), 8)
        or not np.isfinite(probs).all()
        or not ((probs >= 0) & (probs <= 1)).all()
        or sorted(indices.tolist()) != list(range(len(keys)))
        or not (labels == -1).all()
    ):
        raise ValueError("Invalid demo probabilities, indices or unknown labels")
    rows = []
    for i in np.argsort(indices):
        key = keys[int(indices[i])]
        uid, pos = key.rsplit("_", 1)
        rows.append(
            {
                "site_id": key,
                "uniprot_id": uid,
                "position": int(pos),
                "residue": "K",
                **{f"p_{t}": float(probs[i, j]) for j, t in enumerate(TASKS)},
            }
        )
    tsv = output / "predictions.tsv"
    with tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protein": "Q00169",
        "sequence_length": provenance["sequence_length"],
        "n_sites": len(rows),
        "tasks": TASKS,
        "probability_transform": "sigmoid (uncalibrated)",
        "esm_precision": "float32 (mixed precision disabled)",
        "inference_precision": "bfloat16 autocast",
        "checkpoint_doi": manifest["doi"],
        "checkpoint_md5": released["checksum"],
        "input_sha256": {f["path"]: f["sha256"] for f in provenance["files"]},
        "feature_coverage": coverage,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "fair_esm": version("fair-esm"),
            "biopython": version("biopython"),
            "numpy": np.__version__,
            "dssp": subprocess.check_output([dssp, "--version"], text=True).strip(),
        },
        "timings": timings,
        "scope": "Inference demonstration on one protein; no labels, accuracy estimate or novelty claim.",
    }
    if args.compare_reference:
        with (ROOT / "demo/expected/predictions.tsv").open(newline="") as handle:
            reference = list(csv.DictReader(handle, delimiter="\t"))
        if [r["site_id"] for r in rows] != [r["site_id"] for r in reference]:
            raise ValueError("Reference site mapping does not match")
        expected = np.array([[float(r[f"p_{t}"]) for t in TASKS] for r in reference])
        observed = np.array([[r[f"p_{t}"] for t in TASKS] for r in rows])
        delta = float(np.max(np.abs(observed - expected)))
        summary["reference_comparison"] = {
            "max_absolute_difference": delta,
            "absolute_tolerance": args.atol,
            "passed": delta <= args.atol,
        }
        if not delta <= args.atol:
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            raise ValueError(
                f"Reference probability difference {delta} exceeds tolerance {args.atol}"
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Demo complete: {len(rows)} lysines × 8 PTM probabilities\nPredictions: {tsv}\nRun summary: {output / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
