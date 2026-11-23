"""Learnable, fully-differentiable task-conditioned attention pooling (replacement foreground).

This module is a drop-in, config-gated alternative to the LaST stability-vote
foreground. Unlike LaST, it:

1. Runs *after* the SequenceTower (on contextualised tokens), so it can attend to
   task-relevant motifs that only emerge from context.
2. Produces a genuinely per-task attention map via softmax over cosine
   similarities (fully differentiable, no top-k/scatter_add dead-end).
3. Emits one task-specific context vector per PTM and feeds it straight into the
   separator per-task (the separator already expects a [B, T, D] context and
   applies it per task; averaging is avoided).

A data-driven position prior (PSSM-style log-odds computed from training
positives) can optionally seed the per-task position bias and remain trainable.

Controlled by ``use_learnable_task_attention`` in the model config. When off, the
model behaviour is identical to the current M6 (LaST untouched).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTaskAttentionPool(nn.Module):
    """Per-task differentiable attention pooling over the sequence context.

    Args:
        context_dim: dimensionality of the SequenceTower contextual tokens.
        num_tasks: number of PTM tasks (8).
        output_dim: dimensionality of the emitted context vectors.
        query_dim: task-query projection dimension (defaults to context_dim).
        dropout: dropout on attention weights.
        pos_bias_init: optional ``[num_tasks, seq_len]`` prior added to attention
            logits before softmax (initialised but still trainable).
        temperature_init: initial attention temperature (sqrt of query_dim by
            default). Lower = sharper attention.
        use_layer_norm: whether to layer-normalise tokens before projection.
    """

    def __init__(
        self,
        context_dim: int,
        num_tasks: int,
        output_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        dropout: float = 0.1,
        pos_bias_init: Optional[torch.Tensor] = None,
        temperature_init: Optional[float] = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.num_tasks = max(1, int(num_tasks))
        self.output_dim = int(output_dim) if output_dim is not None else self.context_dim
        q_dim = int(query_dim) if query_dim is not None else self.context_dim

        self.task_queries = nn.Parameter(torch.empty(self.num_tasks, q_dim))
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)

        self.use_layer_norm = bool(use_layer_norm)
        if self.use_layer_norm:
            self.k_norm = nn.LayerNorm(self.context_dim)
            self.v_norm = nn.LayerNorm(self.context_dim)
        else:
            self.k_norm = nn.Identity()
            self.v_norm = nn.Identity()

        self.key_proj = nn.Linear(self.context_dim, q_dim, bias=False)
        self.value_proj = nn.Linear(self.context_dim, self.output_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        t_init = float(temperature_init) if temperature_init is not None else math.sqrt(q_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(t_init, dtype=torch.float32)))

        if pos_bias_init is not None and pos_bias_init.shape[-1] >= 1:
            self.pos_bias = nn.Parameter(pos_bias_init.to(torch.float32).clone())
        else:
            # One learnable scalar per task; the SequenceTower already injects a
            # learned position encoding, so a constant bias suffices and avoids
            # spurious positional structure.
            self.pos_bias = nn.Parameter(torch.zeros(self.num_tasks, 1))

    def forward(
        self,
        seq_context: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            seq_context: ``[B, L, context_dim]`` contextualised tokens.
            seq_mask: ``[B, L]`` True where padded.
        Returns:
            dict with ``task_context`` ``[B, T, output_dim]`` and diagnostics.
        """
        h = self.k_norm(seq_context)
        v = self.value_proj(self.v_norm(seq_context))   # [B, L, output_dim]
        B, L, _ = h.shape

        k = self.key_proj(h)                          # [B, L, q_dim]
        q = self.task_queries                          # [T, q_dim]
        qn = F.normalize(q, p=2, dim=-1)
        kn = F.normalize(k, p=2, dim=-1)
        logits = torch.einsum("td,bld->btl", qn, kn)   # [B, T, L]
        tau = torch.exp(self.log_temperature).clamp(min=0.05, max=20.0)
        logits = logits / tau

        pb = self.pos_bias
        if pb.shape[-1] != L:
            if pb.shape[-1] < L:
                pad = torch.zeros(self.num_tasks, L - pb.shape[-1], device=pb.device, dtype=pb.dtype)
                pb = torch.cat([pb, pad], dim=-1)
            else:
                pb = pb[:, :L]
        logits = logits + pb.unsqueeze(0)

        if seq_mask is not None and seq_mask.ndim == 2:
            logits = logits.masked_fill(seq_mask.unsqueeze(1), -1e4)

        attn = F.softmax(logits, dim=-1)               # [B, T, L]
        task_context = torch.einsum("btl,bld->btd", self.dropout(attn), v)  # [B, T, output_dim]

        # Diagnostics computed on clean (pre-dropout) attention.
        with torch.no_grad():
            max_ent = math.log(float(L))
            entropy = -(attn.clamp(min=1e-6) * torch.log(attn.clamp(min=1e-6))).sum(dim=-1)
            entropy_ratio = entropy.mean(dim=0) / max_ent            # [T]
            an = F.normalize(attn, p=2, dim=-1)
            sim = torch.matmul(an, an.transpose(1, 2))               # [B, T, T]
            eye = torch.eye(self.num_tasks, device=sim.device, dtype=sim.dtype)
            offdiag = 1.0 - eye
            denom = (offdiag.sum() * B).clamp(min=1.0)
            inter_sim = (sim * offdiag.unsqueeze(0)).sum() / denom    # scalar
            peak = attn.max(dim=-1).values.mean(dim=0)                # [T]

        out: Dict[str, torch.Tensor] = {"task_context": task_context}
        if return_attention:
            out["attention"] = attn
        out["module_diag"] = {
            "la_attention_entropy_ratio": entropy_ratio.detach(),
            "la_inter_task_attention_sim": inter_sim.detach().view(1),
            "la_attention_peak": peak.detach(),
        }
        return out

