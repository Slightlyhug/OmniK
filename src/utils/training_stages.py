"""Model stage and trainability transitions for each training epoch."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from src.utils.training_state import TrainingState


def apply_epoch_training_stage(
    *,
    model: object,
    state: TrainingState,
    epoch: int,
    stage1_name: str,
    stage2_name: str,
    stage1_end_epoch: int,
    stage1_grl_lambda: float,
    stage2_grl_lambda: float,
    foreground_warmup_epochs: int,
    hard_freeze_backbone_epochs: int,
    freeze_backbone_during_stage1: bool,
    backbone_trainable_state: Optional[bool],
    logger: logging.Logger,
    log_last_status: Callable[[str], None],
) -> Tuple[str, Optional[bool]]:
    current_stage = stage1_name if epoch <= stage1_end_epoch else stage2_name
    if state.current_stage_name != current_stage:
        state.current_stage_name = current_stage
        state.stage_transition_epoch = epoch
        if hasattr(model, "set_last_stage"):
            grl_lambda = stage1_grl_lambda if current_stage == stage1_name else stage2_grl_lambda
            model.set_last_stage(current_stage, grl_lambda=grl_lambda)
        if hasattr(model, "set_training_stage"):
            model.set_training_stage("structure_warmup" if current_stage == stage1_name else "joint")
            backbone_trainable_state = None
            state.last_trainable_state = None
        logger.info("Epoch %d: switched training stage to %s", epoch, current_stage)

    if hasattr(model, "set_backbone_trainable"):
        freeze_reasons: List[str] = []
        backbone_trainable = True
        if hard_freeze_backbone_epochs > 0 and epoch <= hard_freeze_backbone_epochs:
            backbone_trainable = False
            freeze_reasons.append(f"epoch<={hard_freeze_backbone_epochs}")
        if freeze_backbone_during_stage1 and stage1_end_epoch > 0 and current_stage == stage1_name:
            backbone_trainable = False
            freeze_reasons.append(f"stage={stage1_name}")
        if backbone_trainable_state is None or backbone_trainable_state != backbone_trainable:
            model.set_backbone_trainable(backbone_trainable)
            backbone_trainable_state = backbone_trainable
            reasons = " reasons=" + ",".join(freeze_reasons) if freeze_reasons else ""
            logger.info("Epoch %d: backbone trainable=%s%s", epoch, str(backbone_trainable), reasons)

    if hasattr(model, "set_last_module_trainable"):
        last_trainable = foreground_warmup_epochs <= 0 or epoch > foreground_warmup_epochs
        if state.last_trainable_state is None or state.last_trainable_state != last_trainable:
            model.set_last_module_trainable(last_trainable)
            state.last_trainable_state = last_trainable
            log_last_status("warmup" if not last_trainable else "unfrozen")
            if last_trainable:
                if foreground_warmup_epochs > 0 and epoch == foreground_warmup_epochs + 1:
                    logger.info(
                        "Epoch %d: foreground warmup finished, unfreezing LaST modules for %s stage.",
                        epoch,
                        current_stage,
                    )
                else:
                    logger.info("Epoch %d: LaST modules set to trainable.", epoch)
            else:
                logger.info(
                    "Epoch %d: foreground warmup active, LaST modules frozen (%d/%d).",
                    epoch,
                    epoch,
                    foreground_warmup_epochs,
                )
    return current_stage, backbone_trainable_state
