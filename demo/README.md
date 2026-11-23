# Teaching demo

This walkthrough explains the complete path from raw inputs to a prediction without requiring private files to be committed.

1. Prepare a private data directory containing FASTA files, PDB files, the released split CSVs, and compatible ESM checkpoints.
2. Run `tools/extract_esm_features_optimized.py` to create sequence/structure embeddings.
3. Run `src/data/extract_graph_features.py` to create one local graph per protein site. Node features combine relative coordinates, DSSP/RSA descriptors, physicochemical descriptors, and optional ESM-IF vectors.
4. Build or copy the feature-store shards expected by the JSON config and set `PTM_FEATURE_ROOT` and `PTM_DATA_ROOT`.
5. Run the training dry-run. It validates dimensions, split wiring, feature availability, and model construction without updating weights.
6. Train with the released config, then use `scripts/export_probs_from_checkpoint.py` for per-site probabilities.

The smallest code-only checks are:

```bash
python -m compileall -q src scripts tools vendor
python train_multitask_lysine.py --help
python scripts/export_probs_from_checkpoint.py --help
```

The demo deliberately uses placeholders for private data and checkpoints. Replace them with local paths rather than copying them into Git.
