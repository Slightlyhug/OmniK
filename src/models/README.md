# Model architecture

The production training entry point builds `DualBranchFusionPredictor` through
`dual_branch_predictor.py` and combines it with the task heads in
`multitask_heads.py`. Dense-on and dense-off configurations share this model
and selectively enable graph and dense structure features, task adapters,
PTM specialists, task separation, and learnable task attention.

Sequence embeddings are precomputed by `tools/extract_esm_features_optimized.py`.
The runtime model consumes those embeddings and does not download or fine-tune
an ESM language model. Graph features are prepared by
`src/data/extract_graph_features.py`.
