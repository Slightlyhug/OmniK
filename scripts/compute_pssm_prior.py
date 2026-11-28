"""Compute a per-task position prior for LearnableTaskAttentionPool from TRAIN data only.

For each task t and window position p, compare the residue distribution of
known positives vs known negatives (labels from the overlay .npy, -1 = unknown):

    score[t, p] = KL( P_pos(residue | p) || P_neg(residue | p) )

Positions whose residue composition separates positives from negatives get a
high score. Scores are z-scored per task, clipped, and lightly smoothed. The
result seeds the trainable `pos_bias` (attention logit bias) of the new module.

Train CSV rows align 1:1 with overlay rows (verified on sampled rows against the
packed LMDB: uniprot_id / position / window_sequence all match).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Packed-LMDB residue_types encoding (observed): A=1 R=2 N=3 D=4 C=5 Q=6 E=7 G=8
# H=9 I=10 L=11 K=12 M=13 F=14 P=15 S=16 T=17 W=18 Y=19 V=20, other = 0.
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
N_CLASSES = 21


def encode(windows, L):
    lut = np.zeros(256, dtype=np.int64)
    for i, c in enumerate(AA_ORDER):
        lut[ord(c)] = i + 1
    arr = np.frombuffer(
        "".join(w.ljust(L, "X")[:L] for w in windows).encode("ascii", "replace"),
        dtype=np.uint8,
    )
    return lut[arr].reshape(len(windows), L)


def profile(codes, pseudo=1.0):
    # codes [n, L] -> [L, 20] amino-acid probability with pseudocounts.
    # Class 0 (padding 'X' / unknown) is excluded: padding frequency differs
    # between positives and negatives near the window edge (e.g. Kme positives
    # have ~5x more padding at offset -28), which otherwise shows up as a fake
    # distal "motif" in the KL score.
    L = codes.shape[1]
    counts = np.zeros((L, N_CLASSES), dtype=np.float64)
    for p in range(L):
        counts[p] = np.bincount(codes[:, p], minlength=N_CLASSES)
    counts = counts[:, 1:] + pseudo
    return counts / counts.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument(
        "--store-site-keys", help="Align CSV windows to the feature-store label order"
    )
    ap.add_argument("--ptm_names", required=True)
    ap.add_argument("--window_size", type=int, default=73)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ptm_names = [s.strip() for s in args.ptm_names.split(",")]
    labels = np.load(args.overlay).astype(np.float32)
    columns = (
        ["window_sequence", "uniprot_id", "position"]
        if args.store_site_keys
        else ["window_sequence"]
    )
    df = pd.read_csv(args.train_csv, usecols=columns)
    if args.store_site_keys:
        keys = df["uniprot_id"].astype(str) + "_" + df["position"].astype(str)
        if keys.duplicated().any():
            raise SystemExit("Duplicate site identifiers in training CSV")
        df.index = keys
        store_keys = Path(args.store_site_keys).read_text().splitlines()
        df = df.loc[store_keys]
    if len(df) != labels.shape[0] or labels.shape[1] != len(ptm_names):
        raise SystemExit(
            f"shape mismatch csv={len(df)} overlay={labels.shape} tasks={len(ptm_names)}"
        )
    L = args.window_size
    codes = encode(df["window_sequence"].astype(str).tolist(), L)

    offs = np.arange(L)
    g = np.exp(
        -((offs[:, None] - offs[None, :]) ** 2)
        / (2 * max(args.smooth_sigma, 1e-3) ** 2)
    )
    g /= g.sum(axis=1, keepdims=True)

    prior = np.zeros((len(ptm_names), L), dtype=np.float32)
    stats = {}
    for t, name in enumerate(ptm_names):
        pos = codes[labels[:, t] == 1.0]
        neg = codes[labels[:, t] == 0.0]
        if len(pos) < 50 or len(neg) < 50:
            stats[name] = {
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "skipped": True,
            }
            continue
        pp, pn = profile(pos), profile(neg)
        kl = (pp * np.log(pp / pn)).sum(axis=1)  # [L]
        kl = g @ kl
        z = (kl - kl.mean()) / (kl.std() + 1e-8)
        prior[t] = np.clip(z, -args.clip, args.clip)
        top = np.argsort(-kl)[:5] - L // 2
        stats[name] = {
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "kl_max": float(kl.max()),
            "top_offsets": top.tolist(),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, prior)
    with open(args.out.replace(".npy", ".json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    print("written", args.out, prior.shape)


if __name__ == "__main__":
    main()
