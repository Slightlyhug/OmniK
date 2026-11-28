#!/usr/bin/env python3
"""Evaluate raw exported probabilities, masking unknown (-1) labels per task."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def load(prefix):
    p = np.load(prefix + "_probabilities.npy")
    y = np.load(prefix + "_labels.npy")
    meta = json.loads(Path(prefix + "_metadata.json").read_text())
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != len(meta["tasks"]):
        raise ValueError("Prediction, label and task shapes do not agree")
    return p, y, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True)
    ap.add_argument(
        "--val-prefix", help="Choose per-task F1 thresholds on validation only"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p, y, meta = load(args.prefix)
    thresholds = np.full(p.shape[1], 0.5)
    if args.val_prefix:
        vp, vy, vm = load(args.val_prefix)
        if vm["split"] != "val" or vm["tasks"] != meta["tasks"]:
            raise ValueError("Threshold fitting requires the matching validation split")
        for t in range(p.shape[1]):
            known = vy[:, t] >= 0
            truth = vy[known, t]
            prob = vp[known, t]
            if len(np.unique(truth)) < 2:
                continue
            prec, rec, cut = precision_recall_curve(truth, prob)
            score = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
            thresholds[t] = cut[int(np.argmax(score))]
    rows = []
    for t, name in enumerate(meta["tasks"]):
        known = y[:, t] >= 0
        truth = y[known, t]
        prob = p[known, t]
        pred = prob >= thresholds[t]
        both = len(np.unique(truth)) == 2
        rows.append(
            {
                "task": name,
                "n_known": int(known.sum()),
                "n_positive": int((truth == 1).sum()),
                "threshold": float(thresholds[t]),
                "auroc": float(roc_auc_score(truth, prob)) if both else None,
                "auprc": float(average_precision_score(truth, prob)) if both else None,
                "f1": (
                    float(f1_score(truth, pred, zero_division=0))
                    if len(truth)
                    else None
                ),
                "mcc": float(matthews_corrcoef(truth, pred)) if both else None,
            }
        )
    report = {
        "split": meta["split"],
        "probability_transform": meta["probability_transform"],
        "threshold_source": "validation F1" if args.val_prefix else "fixed 0.5",
        "tasks": rows,
        "macro": {
            metric: (
                float(np.mean([r[metric] for r in rows if r[metric] is not None]))
                if any(r[metric] is not None for r in rows)
                else None
            )
            for metric in ["auroc", "auprc", "f1", "mcc"]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
