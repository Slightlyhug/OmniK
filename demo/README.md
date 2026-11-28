# Small-scale reproduction workflow

Start with [installation](../README.md#installation) and [downloads](../README.md#download-resources). Use a small, explicitly recorded subset of proteins for which you have full sequences and compatible PDBs. Keep the same proteins' rows in the chosen CSV split; do not move them between benchmark splits.

1. Extract ESM features for those full sequences and structures using the command in the main README.
2. Run the store builder on the subset, for example:

```bash
python scripts/build_feature_store.py \
  --data-dir data/zenodo --esm-h5 features/esm_features.h5 \
  --pdb-dir data/pdb --out features/demo_store \
  --splits test --limit-proteins 2 --proteins-per-shard 1 --dssp-bin mkdssp
```

`--limit-proteins` selects the first two accessions in sorted order from the chosen split, so supply embeddings for those exact accessions or use a separate subset CSV directory. Missing sequence embeddings cause an error. A two-protein subset is a workflow check, not a benchmark evaluation.

3. Generate an inference config with `prepare_config.py --checkpoint` as shown in the main README, and pass `--store features/demo_store --split test --batch-size 1` to the probability exporter.
4. Check that the exporter loads the checkpoint strictly, writes an `[N,8]` probability matrix, and maps every output index to `features/demo_store/test/global/site_keys.txt`.
5. Run `evaluate_predictions.py` for a metric-format check. Ranking metrics can be undefined on tiny subsets.

For a small training check, prepare small train/val/test subsets with at least one positive for every task in every split (the training entry point validates task presence). Build each split, compute a training-only prior with site-key alignment, and run `--dry-run --model-smoke-test`. This executes one real forward pass without updating weights. The test-only demo store above is insufficient for the training entry point.

Code-only checks, run from the repository root:

```bash
python -m compileall -q src scripts tools vendor
python scripts/download_resources.py --group benchmark --list
python train_multitask_lysine.py --help
python scripts/export_probs_from_checkpoint.py --help
```

No synthetic prediction is presented as a biological result. Full benchmark reproduction requires the resources and conditions described in the [reproduction guide](../docs/REPRODUCIBILITY.md).
