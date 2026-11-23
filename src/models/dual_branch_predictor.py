"""Canonical production import for the multi-task dual-branch predictor.

The implementation remains in ``acetylation_predictor`` for checkpoint and
downstream compatibility.  Keeping this narrow module as the public training
dependency makes it clear that the legacy acetylation-only models are not part
of the multi-task runtime.
"""

from .acetylation_predictor import DualBranchFusionPredictor

__all__ = ["DualBranchFusionPredictor"]
