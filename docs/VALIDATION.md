# Release workflow validation

Checks performed on 28 November 2026 used the existing Linux PTM environment (Python 3.10, PyTorch 2.4.1, CUDA 12.1), an NVIDIA RTX 4090 GPU, and DSSP 4.5.7. These are workflow checks, not a rerun of the complete benchmark.

| Check | Result |
|---|---|
| Resource inventory | All 34 filenames, sizes, MD5 values and links captured from the public Zenodo record |
| Download verification | Zenodo `README.txt` downloaded and verified by size and MD5 |
| Benchmark identity | Local train/val/test CSV MD5 values matched the deposited files |
| Checkpoint identity | Local epoch-50 checkpoint MD5 `605175807a9f2713ef0f603eeb604714` matched Zenodo |
| Public label schema | Khib CSV alias preserved positive, negative and unknown values |
| Runtime configuration | Nested environment paths expanded; training and checkpoint-based config generation succeeded |
| Feature-store rebuild | Real split subsets of 7 training, 6 validation and 7 test sites built across 4/3/4 shards using existing ESM features and local PDBs |
| Structure annotations | All 20 selected sites received graphs; no DSSP protein failures or graph ESM mapping failures in these subsets |
| Training entry point | `--dry-run --model-smoke-test` passed with finite `[1,8]` logits and unchanged model parameters |
| Checkpoint inference | Strict parameter loading and CUDA export passed on a four-site fixture and a seven-site, four-shard test subset |
| Row and feature schema | Store labels matched the sorted CSV; scalar/ESM graph dimensions were 34/512; exported index mapping was checked |
| Prior alignment | Reversing CSV row order while retaining store keys produced the identical prior on a controlled fixture |
| Evaluation | Validation-only threshold fitting and unknown-label masking passed a controlled fixture; single-class/unknown task metrics were handled on real small exports |
| Documentation and syntax | Local Markdown links, citation metadata, Python compilation and Git whitespace checks passed |

Feature rebuilding used existing ESM embeddings; upstream ESM models were not re-extracted as part of these checks. No full training, complete held-out evaluation, multi-GPU training, clean-environment installation, or historical calibrated metric reproduction was performed. The [reproduction guide](REPRODUCIBILITY.md) describes the required additional inputs and provenance.
