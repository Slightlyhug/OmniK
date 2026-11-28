# Complete single-protein demo

Predict eight lysine PTM probabilities from the **bundled full-length FASTA and PDB**, starting with fresh ESM feature extraction. The example uses human phosphatidylinositol transfer protein alpha (**PITPNA**, UniProt **Q00169**): 270 amino acids and **28 lysine sites**.

| File | Description |
|---|---|
| [`inputs/fasta/Q00169.fasta`](inputs/fasta/Q00169.fasta) | One full-length UniProt sequence |
| [`inputs/pdb/Q00169.pdb`](inputs/pdb/Q00169.pdb) | One AlphaFold DB v6 structure, chain A, residues 1–270 |
| [`inputs/test.csv`](inputs/test.csv) | All 28 lysine sites, 73-residue windows, eight unknown labels (`-1`) |
| [`inputs/provenance.json`](inputs/provenance.json) | Source URLs, versions, input SHA-256 checksums and attribution |
| [`run_demo.py`](run_demo.py) | Input verification, feature extraction, store building, inference and readable export |
| [`expected/predictions.tsv`](expected/predictions.tsv) | Reference run: 28 rows × eight PTM probabilities, with site identifiers |
| [`expected/summary.json`](expected/summary.json) | Reference environment, coverage and runtime |

This is an inference example. The unknown labels do not define negative sites, and the output is not an accuracy estimate or evidence of experimental modification. Q00169 is not claimed to be absent from OmniK training.

## 1. Install the environment

Follow the [main installation instructions](../README.md#installation), including `fair-esm`, the ESM-IF1 graph dependencies, Biotite/Biotraj and DSSP. Use a CUDA GPU supporting BF16 and verify that `mkdssp --version` works. Run all commands below from the repository root.

Input checks alone do not require model weights or a GPU:

```bash
python demo/run_demo.py --verify-inputs-only
```

Expected message:

```text
Verified Q00169: 270 residues, 28 lysine sites, exact FASTA/PDB match
```

## 2. Download the three models

The small example files are included in GitHub; model weights are downloaded separately. Benchmark training/validation/test data are not needed for this demo.

```bash
# Released OmniK checkpoint, approximately 919 MB; size and MD5 verified.
python scripts/download_resources.py --group model --output-dir weights

# Upstream ESM models, approximately 2.6 GB and 1.7 GB, plus contact regression.
mkdir -p weights/esm
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
  --output weights/esm/esm2_t33_650M_UR50D.pt
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt \
  --output weights/esm/esm2_t33_650M_UR50D-contact-regression.pt
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt \
  --output weights/esm/esm_if1_gvp4_t16_142M_UR50.pt
```

Upstream model documentation: [facebookresearch/esm](https://github.com/facebookresearch/esm). The ESM-2 contact head is not used for PTM prediction, but keeping its regression file alongside the model avoids a missing-contact-weight warning from the upstream loader.

## 3. Run the complete workflow

```bash
CUDA_VISIBLE_DEVICES=0 python demo/run_demo.py --compare-reference
```

The runner:

1. Checks the bundled file hashes, complete FASTA/PDB sequence and residue-number correspondence, backbone atoms, and all 28 lysine windows.
2. Verifies the OmniK checkpoint against the versioned Zenodo checksum.
3. Extracts fresh ESM-2 `[270,1280]` and ESM-IF1 `[270,512]` embeddings from the bundled files in FP32 (`--no_mixed_precision`). This reduces variation from low-precision graph reductions in ESM-IF1; OmniK inference still uses the reference BF16 autocast.
4. Computes local graphs, DSSP/RSA and micro-environment features, requiring full feature coverage for this example.
5. Generates an inference config from the checkpoint's learned positional bias, then loads the complete model strictly.
6. Exports raw sigmoid probabilities and a readable TSV, checking site identity and task order.
7. Compares predictions to the bundled reference, using an absolute probability tolerance of 0.01.

The tolerance accommodates numerical differences between CUDA/PyTorch/GPU environments; it is a workflow comparison, not a scientific performance threshold. The actual maximum difference is recorded in `summary.json`. Set `--atol` to change it, or omit `--compare-reference` to generate predictions without comparison.

An independent repeat from fresh FP32 ESM extraction passed this comparison with maximum absolute probability difference **0.00473**; see `repeat_validation` in the [reference summary](expected/summary.json). Results need not be identical across environments.

To use existing local weights, another DSSP executable, or a different output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python demo/run_demo.py \
  --esm-model-dir /path/to/esm-models \
  --checkpoint /path/to/OmniK_dense0n_checkpoint_last.pt \
  --dssp-bin /path/to/mkdssp \
  --output-dir runs/demo-Q00169-another-run \
  --compare-reference
```

The ESM directory must contain the filenames in step 2. The runner uses the exact released OmniK checkpoint. Choose an empty output directory for each run; existing results are not overwritten. The selected physical GPU is mapped to device 0 by `CUDA_VISIBLE_DEVICES`.

## 4. Inspect the output

By default, results are written under `runs/demo-Q00169/`:

```text
runs/demo-Q00169/
├── esm_features.h5              # Fresh sequence and structure embeddings
├── feature_store/test/          # Graph, micro-environment and embedding shards
├── inference.json               # Generated runtime configuration
├── inference.position_bias.npy  # Learned positional bias from the checkpoint
├── predictions_probabilities.npy
├── predictions_labels.npy       # All -1 for this unlabeled input
├── predictions_indices.npy
├── predictions_metadata.json
├── predictions.tsv              # Readable per-site probabilities
└── summary.json                 # Checksums, environment, coverage and timings
```

`predictions.tsv` has `site_id`, `uniprot_id`, 1-based `position`, `residue`, and eight `p_<PTM>` columns in this order: ubiquitination, sumoylation, succinylation, methylation, crotonylation, acetylation, hydroxyisobutyrylation, lactylation. The runner resolves the export indices back to site identifiers. These are **uncalibrated sigmoid probabilities**; no fitted temperature scaling or binary decision threshold is applied.

The [reference predictions](expected/predictions.tsv) and [run summary](expected/summary.json) can be inspected without running the model. The reference run on an NVIDIA RTX 4090 took about **23 seconds** across feature extraction, store building, configuration and inference, excluding model downloads and initial input/checkpoint verification. Model files had already been read on this machine, so filesystem caching can affect this time. All 28 sites had complete features, with zero DSSP or graph mapping failures. Hardware and runtime details come from that particular run; they do not establish minimum memory requirements or general throughput.

## Sources and license

The FASTA was retrieved from [UniProtKB Q00169](https://www.uniprot.org/uniprotkb/Q00169/entry), sequence version 2, and is attributed to **The UniProt Consortium**. The unmodified PDB was retrieved from [AlphaFold DB AF-Q00169-F1, model v6](https://alphafold.ebi.ac.uk/entry/Q00169), chain A, and is attributed to **Google DeepMind and EMBL-EBI**. Both were retrieved on 28 November 2026. Exact download URLs and file hashes are recorded in [provenance.json](inputs/provenance.json).

The bundled FASTA, PDB and generated demo data are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Their upstream attribution is preserved; the source code remains under the repository's [MIT License](../LICENSE). The PDB is a predicted structure, not an experimental measurement. Cite [UniProt](https://doi.org/10.1093/nar/gkac1052), [AlphaFold DB](https://doi.org/10.1093/nar/gkad1011), [AlphaFold](https://doi.org/10.1038/s41586-021-03819-2), and the [OmniK resource DOI](https://doi.org/10.5281/zenodo.24533479) when reusing the corresponding resources.

For full benchmark training and evaluation, use the [main README](../README.md) and [reproduction guide](../docs/REPRODUCIBILITY.md).
