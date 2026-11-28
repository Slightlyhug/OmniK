# OmniK

**Multimodal, multi-task prediction of lysine post-translational modification states.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24533479.svg)](https://doi.org/10.5281/zenodo.24533479)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data and weights: CC BY 4.0](https://img.shields.io/badge/Data%20%26%20weights-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

OmniK predicts eight PTM labels at a lysine site by combining ESM-2 sequence embeddings, ESM-IF1 structure embeddings, local residue graphs, micro-environment features, and task-specific prediction heads with learnable task attention. Labels are partially observed: a site can carry multiple positive PTM labels, and unknown labels are masked during training and evaluation.

The release has three components:

1. **Source code** for feature extraction, training, probability export, and evaluation in this repository.
2. **Benchmark data and pretrained weights** on [Zenodo, version V1.0](https://zenodo.org/records/24533479), including the fixed training, validation, and test CSVs.
3. **LysPTM-StateDB and downstream resources** in the same archive: curated PTM tables, perturbation data, interaction data, and human proteome prediction subsets.

**Data and model DOI: [10.5281/zenodo.24533479](https://doi.org/10.5281/zenodo.24533479).** Use this version-specific DOI to identify the resources used in an experiment.

**Start with the [complete single-protein demo](demo/README.md):** one bundled FASTA/PDB pair, model download links, a one-command workflow, and reference predictions for all 28 lysines of human PITPNA (Q00169).

> Reproduction status: the archive provides site windows, labels, and the model checkpoint. Full-benchmark FASTA files, PDB structures, precomputed ESM features, and feature-store shards are not deposited in this version. GitHub includes one matching FASTA/PDB pair for the demo. The workflow below rebuilds benchmark features from local sequence/structure inputs; exact reproduction of manuscript numbers also requires the historical input structures and preprocessing choices. See the [reproduction guide](docs/REPRODUCIBILITY.md) for the input contract and limitations.

## Contents

- [Supported PTM tasks](#supported-ptm-tasks)
- [Pretrained weights](#pretrained-weights)
- [Datasets](#datasets)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Complete demo](#complete-demo)
- [Download resources](#download-resources)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Human proteome predictions](#human-proteome-predictions)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

## Supported PTM tasks

The checkpoint and exported prediction columns use the following fixed order. The wider database contains additional PTM types; these are not extra model output heads.

| Index (0-based) | PTM | Code | Benchmark CSV column |
|---:|---|---|---|
| 0 | Ubiquitination | Kub | `ubiquitination` |
| 1 | SUMOylation | Ksumo | `sumoylation` |
| 2 | Succinylation | Ksucc | `succinylation` |
| 3 | Methylation | Kme | `methylation` |
| 4 | Crotonylation | Kcr | `crotonylation` |
| 5 | Acetylation | Kac | `acetylation` |
| 6 | 2-hydroxyisobutyrylation | Khib | `hydroxyisobutyrylation` |
| 7 | Lactylation | Kla | `lactylation` |

The configuration's `2-Hydroxyisobutyryl` and `Lactylation` names map to the corresponding lowercase CSV columns. Preserve this order when building feature stores, loading weights, or interpreting arrays.

## Pretrained weights

| Model | File | Size | Description |
|---|---|---:|---|
| OmniK M6 / SE(3), dense structure branch + task attention | [`OmniK_dense0n_checkpoint_last.pt`](https://zenodo.org/records/24533479/files/OmniK_dense0n_checkpoint_last.pt?download=1) | 919.47 MB | Last checkpoint at epoch 50; eight PTM tasks |

The filename contains the digit **0** in `dense0n`; use the exact name. This release contains the **last** checkpoint, not a separately deposited best-validation checkpoint. The corresponding architecture is defined in [`configs/generated/site_feature_full_v3/config.json`](configs/generated/site_feature_full_v3/config.json).

[Download and checksum manifest](docs/zenodo-manifest.json) · [Complete resource catalogue](docs/DATA_AVAILABILITY.md) · [Workflow validation](docs/VALIDATION.md)

## Datasets

### Fixed benchmark splits

| Split | Download | Sites | Proteins | Size |
|---|---|---:|---:|---:|
| Training | [`train.csv`](https://zenodo.org/records/24533479/files/train.csv?download=1) | 1,666,028 | 105,682 | 203.11 MB |
| Validation | [`val.csv`](https://zenodo.org/records/24533479/files/val.csv?download=1) | 210,344 | 13,652 | 25.60 MB |
| Test | [`test.csv`](https://zenodo.org/records/24533479/files/test.csv?download=1) | 209,846 | 13,439 | 25.56 MB |

Protein identifiers do not overlap between these three tables. Retain the published split membership for benchmark comparisons. This export does not independently re-establish sequence or structure similarity separation.

Each row is one site with `site_id`, `uniprot_id`, `position`, `residue`, `window_sequence`, and the eight label columns above. Positions are **1-based**; sequence windows have 73 characters, with the target at position 37 and `X` used for boundary padding or unknown residues.

- `1`: positive label.
- `0`: dataset-assigned negative; this does not establish experimentally verified absence.
- `-1`: unknown; exclude from supervised loss and evaluation for that task.

All source rows are retained, including sites with all eight labels unknown. Do not convert unknown labels to negatives. These CSVs contain site windows, **not full-length protein FASTA sequences**.

### LysPTM-StateDB and additional resources

| Resource | Download | Use |
|---|---|---|
| Integrated database | [`LysPTM-StateDB-all-resours.tsv`](https://zenodo.org/records/24533479/files/LysPTM-StateDB-all-resours.tsv?download=1) | Integrated lysine PTM resource; spelling follows the deposited filename |
| Multi-state sites | [`LysPTM-StateDB-multi-state.tsv`](https://zenodo.org/records/24533479/files/LysPTM-StateDB-multi-state.tsv?download=1) | Sites with multiple PTM states |
| Single-state resource | [`LysPTM-StateDB-Single.csv`](https://zenodo.org/records/24533479/files/LysPTM-StateDB-Single.csv?download=1) | Single-state resource table |
| Perturbation data | [`Perturbation_PTM_data.tsv`](https://zenodo.org/records/24533479/files/Perturbation_PTM_data.tsv?download=1) | Downstream perturbation analyses; 2.43 GB |
| Human interactions | [`PPI_human_all_high_confidence_interactions_score_gt_0.5.csv`](https://zenodo.org/records/24533479/files/PPI_human_all_high_confidence_interactions_score_gt_0.5.csv?download=1) | Interaction resource |

The archive also supplies **20 per-modification TSV files** and **four human proteome prediction tables**. Individual links, file sizes, checksums, and licensing are listed in the [resource catalogue](docs/DATA_AVAILABILITY.md). These additional tables are not required for basic model training or inference.

## Project structure

```text
OmniK/
├── configs/generated/site_feature_full_v3/config.json  # Released architecture/settings
├── src/models/                      # Sequence, structure, graph and task modules
├── src/utils/                       # Data loading, training, losses and calibration
├── tools/extract_esm_features_optimized.py             # ESM-2 / ESM-IF1 extraction
├── scripts/
│   ├── download_resources.py        # Version-pinned downloads with verification
│   ├── build_feature_store.py       # Pack embeddings, graphs and micro-environments
│   ├── feature_building.py          # Graph/PDB feature primitives
│   ├── compute_pssm_prior.py        # Training-only positional attention prior
│   ├── prepare_config.py            # Generate config with absolute local paths
│   ├── export_probs_from_checkpoint.py               # CUDA probability export
│   └── evaluate_predictions.py      # Metrics with unknown-label masking
├── vendor/feature_store/             # CPU dataset reader and GPU shard loader
├── train_multitask_lysine.py          # Primary training entry point
├── docs/                             # Data catalogue and reproduction guide
├── demo/
│   ├── inputs/                       # One matching FASTA/PDB pair + 28-site CSV
│   ├── run_demo.py                   # Raw inputs → fresh features → predictions
│   └── expected/                     # Reference predictions and run summary
├── requirements.txt                  # Runtime dependencies
├── environment-ptm.yml               # Historical Linux environment export
└── CITATION.cff                       # Zenodo resource citation
```

## Installation

The recorded training environment uses **Linux x86_64, Python 3.10, PyTorch 2.4.1, and CUDA 12.1**. The released GPU shard workflow and probability exporter require an NVIDIA CUDA GPU; BF16 is enabled in the reference configuration.

```bash
git clone https://github.com/Slightlyhug/OmniK.git
cd OmniK
conda create -n omnik python=3.10 -y
conda activate omnik
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

For ESM-IF1 extraction, also install its graph dependencies using wheels matching PyTorch/CUDA:

```bash
python -m pip install torch-geometric==2.6.1
python -m pip install biotite==1.2.0 biotraj==1.2.2
python -m pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Install [DSSP / `mkdssp`](https://github.com/PDB-REDO/dssp) for secondary-structure and solvent-accessibility graph channels, for example with `conda install -c conda-forge dssp`. Record its version with your experiment. TensorBoard is an optional dashboard.

[`environment-ptm.yml`](environment-ptm.yml) and [`requirements-ptm.txt`](requirements-ptm.txt) preserve the historical full environment. They include analysis packages and platform-specific CUDA wheels; use the focused installation above for a new runtime and consult the exports when matching the original stack.

## Complete demo

The [demo guide](demo/README.md) includes a full-length [Q00169 FASTA](demo/inputs/fasta/Q00169.fasta), its matching [AlphaFold DB v6 PDB](demo/inputs/pdb/Q00169.pdb), [28 lysine inputs](demo/inputs/test.csv), and [reference predictions](demo/expected/predictions.tsv). The benchmark CSVs are not needed for this example.

After installing the environment and downloading the OmniK/ESM weights using the [demo instructions](demo/README.md#2-download-the-three-models), run:

```bash
CUDA_VISIBLE_DEVICES=0 python demo/run_demo.py --compare-reference
```

This verifies input checksums and sequence/structure alignment, extracts fresh ESM features, builds graphs with DSSP, loads the released checkpoint, and writes `runs/demo-Q00169/predictions.tsv` with **28 sites × eight PTM probabilities**. The [reference run summary](demo/expected/summary.json) records feature coverage, environment and runtime. The demo is an unlabeled inference example; probabilities are uncalibrated and do not measure accuracy.

## Download resources

Run from the repository root. The downloader uses the bundled version-specific [manifest](docs/zenodo-manifest.json) and checks both byte size and Zenodo's MD5 checksum. It needs only Python's standard library.

```bash
# Inspect downloads without fetching them.
python scripts/download_resources.py --group benchmark --list

# About 254 MB of benchmark CSVs and 919 MB of weights.
python scripts/download_resources.py --group benchmark --output-dir data/zenodo
python scripts/download_resources.py --group model --output-dir weights

# Optional downstream data; these groups are independent of model reproduction.
python scripts/download_resources.py --group database --output-dir data/zenodo
python scripts/download_resources.py --group predictions --output-dir data/zenodo
```

Choose individual deposited files with `--files Kac.tsv Kub.tsv`, or download the entire record with `--group all`. Verified existing files are reused. Interrupted downloads have a `.part` suffix and are restarted on the next run. Data and weights are excluded from Git.

## Data preparation

### 1. Full sequences, structures, and ESM models

Obtain the full-length sequences corresponding to split accessions from [UniProt](https://www.uniprot.org/), and compatible structures from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), [RCSB PDB](https://www.rcsb.org/), or a documented prediction workflow. Store proteins as `data/fasta/<uniprot_id>.fasta` and structures as `data/pdb/<uniprot_id>.pdb`.

Verify that the full sequence matches each supplied site window and that PDB residue numbers correspond to the benchmark's 1-based positions. Record sequence/structure versions and any chain mapping. Current database downloads can differ from historical inputs; a 73-residue site window alone cannot reconstruct a full protein or its structure.

The ESM feature extractors use these upstream models:

| Feature | Model | Download |
|---|---|---|
| Sequence, 1280 dimensions | ESM-2 `esm2_t33_650M_UR50D` | [Checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) · [Contact regression](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt) |
| Structure, 512 dimensions | ESM-IF1 `esm_if1_gvp4_t16_142M_UR50` | [Checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt) |

Save these files in `weights/esm/`, keeping the contact regression file beside the ESM-2 checkpoint. See the [official ESM repository](https://github.com/facebookresearch/esm) for model usage and terms.

### 2. ESM features

```bash
python tools/extract_esm_features_optimized.py \
  --fasta_dir data/fasta --pdb_dir data/pdb \
  --sequence_model_path weights/esm/esm2_t33_650M_UR50D.pt \
  --structure_model_path weights/esm/esm_if1_gvp4_t16_142M_UR50.pt \
  --output_dir features --mode both --num_gpus 1 \
  --max_proteins_per_chunk 128 --output_format hdf5
```

This writes `features/esm_features.h5`. Inspect the extraction summary for missing or skipped proteins. The extractor's default maximum length is 10,000 residues; review `--max_length` for longer proteins.

### 3. Feature store and training-only prior

```bash
python scripts/build_feature_store.py \
  --data-dir data/zenodo --esm-h5 features/esm_features.h5 \
  --pdb-dir data/pdb --out features/feature_store \
  --dssp-bin mkdssp --proteins-per-shard 128

python scripts/compute_pssm_prior.py \
  --train_csv data/zenodo/train.csv \
  --overlay features/feature_store/train/global/labels.npy \
  --store-site-keys features/feature_store/train/global/site_keys.txt \
  --ptm_names ubiquitination,sumoylation,succinylation,methylation,crotonylation,acetylation,hydroxyisobutyrylation,lactylation \
  --out features/pssm_prior.npy

python scripts/prepare_config.py \
  --data-dir data/zenodo --feature-root features \
  --prior features/pssm_prior.npy --out configs/local.json
```

The builder preserves all sites and labels, sorts sites by protein and position, and stores the resulting row mapping in `global/site_keys.txt`. It packs 34 scalar graph channels, 512 ESM-IF channels plus a validity flag, and 16 edge channels for the released 547-dimensional graph input. The six micro-environment values are computed from local Cα geometry. The manifest records coverage and graph settings. Missing ESM-2 embeddings stop the build; unavailable structural inputs use explicit fallback masks. Inspect coverage before training.

The prior script aligns CSV windows to the feature store's site keys, then uses **training labels only**. Do not fit it on validation or test data. See [feature details and reproduction limits](docs/REPRODUCIBILITY.md).

## Training

Validate the prepared data and perform one forward pass before a full run:

```bash
python train_multitask_lysine.py --config_path configs/local.json \
  --dry-run --model-smoke-test
```

Single GPU:

```bash
python train_multitask_lysine.py --config_path configs/local.json
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_multitask_lysine.py --config_path configs/local.json
```

The reference configuration uses seed 42, up to 50 epochs, AdamW with learning rate `3e-5` and weight decay `0.02`, 73-residue sequence windows, 31-residue local windows, three SE(3) graph layers, BF16, and gradient accumulation. GPU shard batches use `loader_batch_size=256` and `grad_accum_steps=4`; two GPUs give a nominal 2048 samples per optimizer update, with smaller tail batches possible. Training may stop early after 10 epochs without improvement in `val_macro_f1_opt`.

If memory is insufficient, reduce `loader_batch_size` and the shard size; record any changes because they can affect optimization and throughput. A minimum GPU memory or wall-clock estimate has not been established for this portable rebuild.

To resume a training run, pass `--resume_from runs/.../checkpoint_last.pt`. `--warm_start_from` loads model weights for a new run. Full training also writes validation/test reports and calibration artifacts; outputs are placed under the `output_dir` recorded in the generated config.

## Inference

Use a prepared feature store. The exporter does **not** accept raw FASTA or PDB files directly. For the Zenodo model, generate an inference config using the trained positional bias stored in the checkpoint; this avoids needing to regenerate its initialization prior:

```bash
python scripts/prepare_config.py \
  --data-dir data/zenodo --feature-root features \
  --checkpoint weights/OmniK_dense0n_checkpoint_last.pt \
  --out configs/inference.json

python scripts/export_probs_from_checkpoint.py \
  --config configs/inference.json \
  --checkpoint weights/OmniK_dense0n_checkpoint_last.pt \
  --store features/feature_store --split test \
  --batch-size 64 --out-prefix runs/pretrained/test
```

The model is loaded strictly: incompatible parameter names or tensor dimensions cause an error. The exporter writes:

| File | Meaning |
|---|---|
| `test_probabilities.npy` | `[N, 8]` raw sigmoid probabilities |
| `test_labels.npy` | `[N, 8]` labels, including `-1` for unknown |
| `test_indices.npy` | `[N]` indices into the split's `global/site_keys.txt` |
| `test_metadata.json` | Task order, checkpoint, split, and index reference |

Use `test_indices.npy` to recover site identities; do not assume the exported row order matches the original CSV. Raw sigmoid probabilities have **no post-hoc calibration applied**. The released Zenodo record does not separately provide the fitted temperature-scaling artifacts used for calibrated manuscript analyses.

For new proteins, prepare the same sequence/structure features and a site CSV with the eight task columns set to `-1`, build a store with `--splits test`, and export that split. The model supports partially missing structures, but this does not imply equivalence to predictions made with complete structural features.

## Evaluation

Export both validation and test probabilities with the same checkpoint. Fit thresholds on validation only, then evaluate the held-out test set:

```bash
python scripts/export_probs_from_checkpoint.py \
  --config configs/inference.json \
  --checkpoint weights/OmniK_dense0n_checkpoint_last.pt \
  --store features/feature_store --split val \
  --batch-size 64 --out-prefix runs/pretrained/val

python scripts/evaluate_predictions.py \
  --prefix runs/pretrained/test --val-prefix runs/pretrained/val \
  --out runs/pretrained/test_metrics.json
```

The report includes per-task and macro AUROC, AUPRC, F1, and MCC, with unknown labels masked separately for each task. Omitting `--val-prefix` evaluates F1/MCC at a fixed threshold of 0.5. This standalone evaluation uses raw exported probabilities; calibrated manuscript metrics require the matching calibration procedure and inputs.

## Human proteome predictions

Ready-to-use subsets are available without running the model:

| Probability threshold | Download |
|---:|---|
| 0.5 | [`OmniK_human_proteome_lysine_hubs_ge_0.5.tsv`](https://zenodo.org/records/24533479/files/OmniK_human_proteome_lysine_hubs_ge_0.5.tsv?download=1) |
| 0.6 | [`OmniK_human_proteome_lysine_hubs_ge_0.6.tsv`](https://zenodo.org/records/24533479/files/OmniK_human_proteome_lysine_hubs_ge_0.6.tsv?download=1) |
| 0.7 | [`OmniK_human_proteome_lysine_hubs_ge_0.7.tsv`](https://zenodo.org/records/24533479/files/OmniK_human_proteome_lysine_hubs_ge_0.7.tsv?download=1) |
| 0.8 | [`OmniK_human_proteome_lysine_hubs_ge_0.8.tsv`](https://zenodo.org/records/24533479/files/OmniK_human_proteome_lysine_hubs_ge_0.8.tsv?download=1) |

Each table retains sites predicted to have **at least three PTM states** above its specified threshold. These are prediction subsets, not exhaustive tables of every lysine or experimental evidence of simultaneous modification.

## Acknowledgements

OmniK uses [ESM-2 and ESM-IF1](https://github.com/facebookresearch/esm), [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pyg.org/), [Biopython](https://biopython.org/), and [DSSP](https://github.com/PDB-REDO/dssp). Sequence and structural resources can be obtained from [UniProt](https://www.uniprot.org/), [AlphaFold DB](https://alphafold.ebi.ac.uk/), and [PDB](https://www.rcsb.org/); cite the resources and versions used in your reconstruction.

The Zenodo resource release acknowledges support from the International Research Collaboration Seed Fund of the Zhejiang University International Campus.

## License

Source code: **MIT**, see [LICENSE](LICENSE). Datasets, pretrained OmniK weights, and prediction tables in Zenodo V1.0: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, as specified in the [record](https://zenodo.org/records/24533479). The bundled demo FASTA, PDB and generated demo data are also CC BY 4.0; see their [sources and attribution](demo/README.md#sources-and-license). Upstream ESM models, databases, and third-party packages retain their own licenses and terms.

## Citation

Please cite the versioned resource record when using the data, weights, or prediction tables. This DOI identifies the **resource archive**; it is not a manuscript DOI. GitHub's “Cite this repository” entry is supplied through [CITATION.cff](CITATION.cff).

```bibtex
@dataset{peng2026omnik,
  author = {Peng, Jiahua and Fang, Yingbo and Li, Muxuan and Xiao, Yueming and
            Qian, Qiuting and Wang, Zirui and Qu, Yuan and Liu, Jingliang and
            Fu, Rao and Zhao, Jingjing and Zhao, Make and Chen, Qiuxu and
            Ke, Yuehai and Chi, Ying},
  title = {OmniK: Model weights and benchmark datasets for lysine
           post-translational modification prediction},
  year = {2026},
  version = {V1.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.24533479},
  url = {https://doi.org/10.5281/zenodo.24533479}
}
```

For questions and reproducibility issues, open a [GitHub issue](https://github.com/Slightlyhug/OmniK/issues) with the code commit, resource DOI, environment, and relevant command/log.
