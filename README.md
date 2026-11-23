# Protein811 PTM Prediction

Code-only release of the Protein811 multi-task PTM prediction framework. The model combines sequence ESM embeddings, optional structure embeddings, local graph features, task-specific heads, learnable task attention, calibration, and post-training evaluation for eight PTM tasks.

This repository intentionally excludes private datasets, PDB/FASTA files, ESM checkpoints, feature stores, training outputs, and model checkpoints. The published configuration uses environment-variable paths so it can be run on another machine without exposing local filesystem paths.

## Repository map

- `src/models/`: dual-branch model, graph/SE(3) components, task heads, and losses.
- `src/utils/`: data preparation, feature-store readers, training loop, metrics, calibration, and checkpoint helpers.
- `train_multitask_lysine.py`: primary multi-task training entry point.
- `scripts/export_probs_from_checkpoint.py`: inference/export of per-site probabilities.
- `src/data/extract_graph_features.py`: PDB-to-local-graph feature extraction pipeline.
- `tools/extract_esm_features_optimized.py`: ESM-2 and ESM-IF sequence/structure feature extraction pipeline.
- `configs/generated/site_feature_full_v3/`: the released 50-epoch M6/SE3 configuration.
- `demo/`: minimal teaching workflow and smoke-test commands.

## Installation

Use Python 3.10 or newer. Install a PyTorch build that matches your CUDA or CPU environment, then install the project packages:

```bash
python -m pip install -r requirements.txt
```

To recreate the exported PTM Conda environment, use `conda env create -f environment-ptm.yml`. The full Python package lock is also available in `requirements-ptm.txt`. Both files record the Linux x86_64 PTM environment with Python 3.10, PyTorch 2.4.1, and CUDA 12.1; the CUDA-specific PyG wheel may need a different wheel index on another hardware stack. The shorter `requirements.txt` lists the release runtime dependencies.

Install `fair-esm` only when extracting ESM features. DSSP is an external executable and is optional when secondary-structure features are not needed. TensorBoard and MLflow are optional training-monitoring integrations.

The code is imported from the repository root:

```bash
git clone https://github.com/Slightlyhug/protein811-f4-reproducible.git
cd protein811-f4-reproducible
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Data and feature paths

Set these variables to your local, private data layout:

```bash
export PTM_DATA_ROOT=/path/to/private/splits-and-priors
export PTM_FEATURE_ROOT=/path/to/private/esm-and-feature-store
export PTM_OUTPUT_ROOT=$PWD/runs
export PTM_FEATURE_STORE_ROOT=$PTM_FEATURE_ROOT/feature_store_v3_extfix_20261110_species_rebal_20261110
```

The released configuration expects:

- `$PTM_FEATURE_ROOT/esm_features.h5`
- `$PTM_DATA_ROOT/cluster_split/{train,val,test}/sites_with_structure.csv`
- `$PTM_FEATURE_ROOT/feature_store_v3_extfix_20261110_species_rebal_20261110/{train,val,test}/manifest.json`
- `$PTM_DATA_ROOT/pssm_prior_multispecies_rebuild_v1_nopad.npy`

These files are not part of the repository and must be obtained from the project owner or regenerated with the pipelines below.

## Train

Validate configuration and data wiring first:

```bash
python train_multitask_lysine.py \
  --config_path configs/generated/site_feature_full_v3/site_feature_store_v3_extfix_species_rebal_M6_SE3_3l_denseon_learnattn_ddp2_512ga2_neg2p5_50ep.json \
  --dry-run
```

Run training on one GPU after the dry-run passes:

```bash
python train_multitask_lysine.py \
  --config_path configs/generated/site_feature_full_v3/site_feature_store_v3_extfix_species_rebal_M6_SE3_3l_denseon_learnattn_ddp2_512ga2_neg2p5_50ep.json
```

For multi-GPU training, launch the same entry point with `torchrun`; keep checkpoints and logs under `PTM_OUTPUT_ROOT`.

## Inference

Export probabilities for a split from a compatible checkpoint:

```bash
python scripts/export_probs_from_checkpoint.py \
  --config configs/generated/site_feature_full_v3/site_feature_store_v3_extfix_species_rebal_M6_SE3_3l_denseon_learnattn_ddp2_512ga2_neg2p5_50ep.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --store "$PTM_FEATURE_STORE_ROOT" \
  --split test \
  --out-prefix runs/test
```

## Feature extraction

Extract ESM-2/ESM-IF features from FASTA/PDB inputs:

```bash
python tools/extract_esm_features_optimized.py \
  --fasta_dir /path/to/fasta \
  --pdb_dir /path/to/pdb \
  --sequence_model_path /path/to/esm2_t33_650M_UR50D.pt \
  --structure_model_path /path/to/esm_if1_gvp4_t16_142M_UR50.pt \
  --output_dir "$PTM_FEATURE_ROOT/esm_features"
```

Convert PDB structures and optional ESM-IF embeddings into local graph tensors:

```bash
python src/data/extract_graph_features.py \
  --pdb_dir /path/to/pdb \
  --manifest "$PTM_DATA_ROOT/cluster_split/train/sites_with_structure.csv" \
  --output_dir "$PTM_FEATURE_ROOT/graph_features_v2/shared" \
  --esm_if_h5 "$PTM_FEATURE_ROOT/esm_features.h5" \
  --esm_if_dim 512
```

The feature-store reader and GPU shard loader used by training are included in `vendor/feature_store/` and no longer depend on the original machine's absolute paths.

Feature creation utilities for packing the released data into feature-store shards are maintained with the project owner because they operate on private, project-specific datasets. The public training, inference, ESM extraction, and graph extraction paths are included here.

## License and release status

The GitHub repository is initially private for manuscript review. Before making it public, verify that all data, weights, and third-party model licenses permit redistribution, then change the repository visibility in GitHub settings.
