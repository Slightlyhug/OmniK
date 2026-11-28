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

The benchmark-subset checks above used existing ESM embeddings. The separate bundled-demo validation below extracts embeddings afresh. No full training, complete held-out evaluation, multi-GPU training, clean-environment installation, or historical calibrated metric reproduction was performed. The [reproduction guide](REPRODUCIBILITY.md) describes the required additional inputs and provenance.

## Complete FASTA/PDB demo

The [Q00169 demo](../demo/README.md) was run from the bundled raw inputs on 28 November 2026 using the environment above. The FASTA and AlphaFold DB v6 PDB match exactly across all 270 residues, with chain A numbered 1–270 and complete N/CA/C backbone atoms. Input SHA-256 values and the released OmniK checkpoint MD5 were verified.

The run extracted fresh ESM-2 `[270,1280]` and ESM-IF1 `[270,512]` arrays, built all 28 lysine graphs with zero missing embeddings, DSSP failures or graph mapping failures, strictly loaded the released checkpoint, and exported finite `[28,8]` probabilities in `[0,1]`. Export indices were resolved to the input site identities; all labels remained unknown (`-1`).

The [reference predictions](../demo/expected/predictions.tsv) and [portable summary](../demo/expected/summary.json) are included. The recorded subprocess times were 12.869 s for ESM extraction, 2.247 s for store building, 2.054 s for configuration, and 5.661 s for inference (about 23 s total, excluding downloads and initial input/checkpoint verification). Model files had already been read on this machine, so filesystem caching can affect this time. These describe one run, not minimum hardware requirements or general throughput. No predictive accuracy is inferred from this unlabeled example.

An initial mixed-precision ESM extraction comparison exceeded the demo's 0.01 absolute probability tolerance (maximum difference 0.01497). The published demo therefore extracts ESM features in FP32 via `--no_mixed_precision`, while retaining BF16 autocast for OmniK inference. The extractor's existing default remains mixed precision for other workflows.

A second independent complete FP32 extraction/inference run passed `--compare-reference`: maximum absolute probability difference **0.0047259331**, below the 0.01 tolerance. This is recorded under `repeat_validation` in the reference summary. It verifies these two runs in the stated environment; other GPU/software combinations can differ.
