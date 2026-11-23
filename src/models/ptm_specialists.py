from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_ptm_name(name: str) -> str:
    value = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "2hydroxyisobutyryl": "2hydroxyisobutyryl",
        "2hydroxyisobutyrylation": "2hydroxyisobutyryl",
        "khib": "2hydroxyisobutyryl",
        "sumo": "sumoylation",
    }
    return aliases.get(value, value)


class _CenterTaskExpert(nn.Module):
    """Lightweight task query with an explicit centre-Lys positional prior."""

    def __init__(self, hidden_dim: int, enhanced: bool = False):
        super().__init__()
        self.enhanced = bool(enhanced)
        self.query = nn.Parameter(torch.empty(hidden_dim))
        self.distance_scale = nn.Parameter(torch.tensor(0.75 if enhanced else 1.25))
        self.scale_logits = nn.Parameter(torch.tensor([1.8, 1.2, 0.7, 0.3, 0.8]))
        self.local_filter = (
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, 1),
            )
            if enhanced
            else None
        )
        expansion = 2 if enhanced else 1
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )
        nn.init.normal_(self.query, std=hidden_dim ** -0.5)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.local_filter is not None:
            tokens = tokens + self.local_filter(tokens.transpose(1, 2)).transpose(1, 2)
        length = int(tokens.shape[1])
        center = length // 2
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        positions = torch.arange(length, device=tokens.device, dtype=tokens.dtype)
        distance = (positions - float(center)).abs() / max(1.0, float(center))
        scores = torch.einsum("bld,d->bl", F.normalize(tokens.float(), dim=-1), F.normalize(self.query.float(), dim=0))
        scores = scores - F.softplus(self.distance_scale.float()) * distance.float()
        scores = scores.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(scores, dim=1).to(dtype=tokens.dtype)
        attended = torch.sum(tokens * attention.unsqueeze(-1), dim=1)

        pooled = [tokens[:, center, :]]
        for radius in (3, 7, 15):
            lo, hi = max(0, center - radius), min(length, center + radius + 1)
            local_mask = valid[:, lo:hi].to(dtype=tokens.dtype).unsqueeze(-1)
            pooled.append((tokens[:, lo:hi, :] * local_mask).sum(1) / local_mask.sum(1).clamp_min(1.0))
        pooled.append(attended)
        weights = torch.softmax(self.scale_logits, dim=0).to(dtype=tokens.dtype)
        fused = sum(weight * value for weight, value in zip(weights, pooled))
        return self.output(fused)


class PTMSpecialistExtractors(nn.Module):
    """Eight task-specific centre-aware experts; low-data tasks receive extra capacity."""

    DEFAULT_TASKS = (
        "ubiquitination", "sumoylation", "succinylation", "methylation",
        "crotonylation", "acetylation", "2hydroxyisobutyryl", "lactylation",
    )
    DEFAULT_ENHANCED = ("crotonylation", "2hydroxyisobutyryl", "lactylation")

    def __init__(
        self,
        input_dim: int,
        task_names: Optional[Iterable[str]] = None,
        enhanced_targets: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = max(64, self.input_dim // 4)
        names = [canonical_ptm_name(x) for x in (task_names or self.DEFAULT_TASKS)]
        enhanced = {canonical_ptm_name(x) for x in (enhanced_targets or self.DEFAULT_ENHANCED)}
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
        )
        self.experts = nn.ModuleDict({
            name: _CenterTaskExpert(self.output_dim, enhanced=name in enhanced)
            for name in names
        })

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        reference = next(self.parameters())
        x = x.to(dtype=reference.dtype)
        tokens = self.shared_projection(x)
        if mask is None:
            mask = torch.isfinite(x).all(dim=-1) & (x.abs().sum(dim=-1) > 0)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool).clone()
        # The centre is always a real Lys in the repaired 73-mer data.
        mask[:, int(mask.shape[1]) // 2] = True
        return {name: expert(tokens, mask) for name, expert in self.experts.items()}
