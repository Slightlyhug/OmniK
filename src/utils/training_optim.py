"""Optimizer and learning-rate scheduler construction."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def create_optimizer(config: Dict[str, Any], model: nn.Module, extra_params: Optional[List[torch.nn.Parameter]] = None) -> torch.optim.Optimizer:
    lr = float(config.get("learning_rate", 3e-5))
    wd = float(config.get("weight_decay", 0.0))
    opt_name = str(config.get("optimizer", "adamw")).lower()

    param_groups: List[Dict[str, Any]] = []
    seen_params: set = set()

    def _append_group(params: List[torch.nn.Parameter], group_lr: float, group_wd: float) -> None:
        unique_params: List[torch.nn.Parameter] = []
        for p in params:
            if p is None or not isinstance(p, torch.nn.Parameter) or not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            unique_params.append(p)
        if unique_params:
            param_groups.append(
                {
                    "params": unique_params,
                    "lr": float(group_lr),
                    "weight_decay": float(group_wd),
                }
            )

    if hasattr(model, "get_grouped_parameters"):
        grouped = model.get_grouped_parameters(base_lr=lr, weight_decay=wd)
        if isinstance(grouped, list):
            for group in grouped:
                if not isinstance(group, dict):
                    continue
                group_params = group.get("params", [])
                if not isinstance(group_params, list):
                    continue
                _append_group(
                    group_params,
                    float(group.get("lr", lr)),
                    float(group.get("weight_decay", wd)),
                )
    else:
        _append_group(list(model.parameters()), lr, wd)

    if extra_params:
        criterion_lr = float(config.get("criterion_learning_rate", lr * float(config.get("criterion_lr_scale", 1.0))))
        criterion_wd = float(config.get("criterion_weight_decay", wd))
        _append_group([p for p in extra_params if p is not None], criterion_lr, criterion_wd)

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer construction.")

    if opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def create_scheduler(config: Dict[str, Any], optimizer: torch.optim.Optimizer, total_steps: int):
    sch = str(config.get("scheduler", "none")).lower()
    if sch == "cosine_with_warmup":
        warmup_steps = int(config.get("warmup_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if sch == "cosine":
        epochs = int(config.get("num_epochs", config.get("epochs", 50)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    return None
