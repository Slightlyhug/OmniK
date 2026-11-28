# Reproducing OmniK

The [main README](../README.md) gives commands in execution order. This guide specifies which resources those commands require and what can be reproduced with the present release.

## Choose a workflow

| Goal | Required resources | Entry point |
|---|---|---|
| Run one protein from raw inputs | Bundled Q00169 FASTA/PDB + OmniK/ESM weights + DSSP + CUDA GPU | [Complete demo](../demo/README.md): `python demo/run_demo.py --compare-reference` |
| Inspect benchmark labels | Zenodo `train.csv`, `val.csv`, `test.csv` | `pandas.read_csv` |
| Use released human prediction subsets | Four Zenodo `OmniK_human_proteome_*.tsv` tables | Direct download; no GPU needed |
| Evaluate the released checkpoint | Checkpoint + benchmark splits + compatible features | `prepare_config.py --checkpoint`, then probability export and evaluation |
| Train a new model | Fixed splits + full sequences + structures + upstream ESM models | Feature extraction → store builder → training prior → config → training |
| Reproduce manuscript numbers | Above, plus historical input versions, graph settings and matching calibration | Requires artifacts/provenance not fully deposited in V1.0 |

## Local input contract

The [bundled demo](../demo/README.md) supplies one checked full-length sequence/structure pair and an all-unknown site CSV. It demonstrates this input contract without requiring benchmark downloads; [provenance](../demo/inputs/provenance.json) records versions, hashes and source URLs. It does not replace the full-benchmark input collection.

Run commands from the repository root. Use one full protein sequence per accession (`data/fasta/<accession>.fasta`) and a PDB named `data/pdb/<accession>.pdb`. Retain isoform identifiers when present. FASTA headers must identify the accession, either directly or in UniProt `sp|accession|...` / `tr|accession|...` format.

The benchmark CSV supplies a *window*, not the full sequence. Check the accession, residue coordinate, center lysine and available flanking residues against the complete sequence. Resolve sequence updates and isoform mappings explicitly. For graph construction, PDB residue numbering must use the benchmark coordinate system. The portable builder rejects ambiguous center chains and non-lysine centers by marking the graph unavailable; it does not infer chain mappings or repair residue numbering. Where an experimental PDB requires alignment, prepare an aligned input and record that transformation.

`features/esm_features.h5` has a `proteins/<accession>` group with `sequence_features` of shape `[L,1280]` and optional `structure_features` of shape `[L,512]`. The extractor writes sequence and pooled metadata too. Structure arrays must be aligned to the protein sequence. The graph ESM-IF mapping additionally requires an unambiguous exact PDB-chain match to the stored sequence, using residues with N/CA/C backbone atoms; unmapped nodes receive a validity mask of zero.

The supplied extractor may skip proteins exceeding `--max_length`; its default is 10,000 residues. Review extraction summaries, supply the needed embeddings and then build the store. The store builder refuses to replace missing sequence embeddings with zeros.

## Rebuilt graph and micro-environment features

The portable store builder reuses graph/PDB primitives recovered from the local OmniK training pipeline. It combines same-chain sequence neighbours (default radius 15 residues) with spatial neighbours (default 10 Å), and connects nodes within 10 Å. These parameters are explicit CLI arguments and recorded in each split manifest. Match the historical parameters before claiming an exact reproduction.

The input layout is:

| Input | Dimensions | Contents |
|---|---:|---|
| Sequence window | 73 × 1280 | ESM-2; zero padding outside protein boundaries |
| Structure window | 73 × 512 | ESM-IF1; missing-structure mask convention |
| Scalar graph node | 34 | Amino-acid one-hot (20), DSSP (8), RSA (1), sequence offset (1), physicochemical values (4) |
| Graph ESM-IF | 512 + 1 | Per-node structure embedding plus validity flag |
| Combined graph model input | 547 | Scalar channels + ESM-IF + validity |
| Edge | 16 | Distance radial-basis channels |
| Micro-environment | 6 | Neighbour counts within 8/12 Å; mean/std distance to five nearest residues; mean sequence distance within 12 Å; exponential distance sum |

Relative Cα coordinates are supplied separately to the SE(3) graph tower. `mkdssp` supplies the DSSP/RSA channels. If it is omitted or fails, the affected channels stay zero; inspect `dssp_failed_proteins`. Also inspect missing structure and graph mapping counts. A masked rebuild is executable but is not equivalent to a complete historical feature store.

`src/data/extract_graph_features.py` is an earlier standalone extractor with a different feature layout. Use `scripts/build_feature_store.py` for the released 547-dimensional shard model. Do not substitute the standalone graph tensors without adapting and validating their representation.

## Store layout and row identity

```text
features/feature_store/<split>/
├── manifest.json
├── global/
│   ├── site_keys.txt       # accession_position in store order
│   ├── labels.npy          # [N,8], values -1/0/1
│   ├── site_shard.npy
│   └── site_local.npy
└── shards/shard_0000/
    ├── meta.json
    ├── proteins.txt
    └── protein_*.npy, site_*.npy, graph_*.npy
```

The builder sorts by accession and residue position and never reassigns split membership. Exported prediction indices refer to `site_keys.txt`, which may differ from original CSV row order. The positional prior script uses `--store-site-keys` to align training windows with those labels. Omitting this option is valid only when CSV and overlay rows already match exactly.

`--limit-proteins` deliberately builds a debug subset. Do not use its metrics as full-benchmark results. Store manifests identify row, protein and shard counts. Choose a new output directory for each complete rebuild. If a build is interrupted, remove or move the incomplete output before retrying.

## Training and model loading

Use `prepare_config.py` to resolve paths in the released reference configuration. For new training, pass `--prior` computed from training data only. For inference, pass `--checkpoint`; this extracts the learned `[8,73]` position bias into a local initialization file, after which strict checkpoint loading restores all weights. The generated bias file is not a calibration artifact.

The reference configuration contains stage/ablation dictionaries. Running without `--stage` uses the top-level dense-on M6 configuration. A stage override or changes to task order, graph representation, hidden dimensions or attention pooling can invalidate checkpoint compatibility. Do not select an ablation stage for the deposited checkpoint.

The reference loader batch is 256 per GPU and gradient accumulation is 4; the nominal effective batch is 2048 on two GPUs. The config also contains legacy batch settings for other loader paths. The 50-epoch limit and validation-based early stopping are both active. The deposited model is the epoch-50 **last** checkpoint; it is not named or advertised as the best-validation checkpoint.

## Evaluation and calibration

Mask `-1` per task before computing losses, fitting thresholds, or evaluating metrics. Retain 0 as the dataset-assigned negative class. Thresholds for F1/MCC must be selected on validation, never test. AUROC/AUPRC do not require binary thresholds. When a task contains only one class in a debug subset, the evaluation script reports undefined ranking metrics as `null` and averages only defined task metrics.

`export_probs_from_checkpoint.py` writes raw sigmoid outputs, labels, indices and task metadata. The evaluation helper computes AUROC, AUPRC, F1 and MCC on those raw probabilities. It does not apply post-hoc temperature scaling. Training includes calibration utilities, but reproducing calibrated historical results requires matching validation predictions, fitted calibration parameters and checkpoint selection. These fitted artifacts are not separately in the Zenodo record.

## Record an experiment

Save the Git commit, version DOI, verified file checksums, generated runtime config, full sequence/structure provenance, ESM and DSSP versions, feature-store manifests, CUDA/PyTorch versions, GPU count, seed, checkpoint epoch, probability transform, and threshold source alongside results. These identify whether a result uses a newly rebuilt input representation or the original historical features.
