"""SE(3)-equivariant graph transformer for residue-centered pocket graphs.

Drop-in alternative to :class:`egnn_3d_graph.PocketEGNN`.  Scalars are
invariant; vector channels are only scaled by invariant weights and added
along relative directions, so the encoder is rotation equivariant and
translation invariant (input ``pos`` is already centered).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over all edges that share the same destination node."""
    if logits.numel() == 0:
        return logits
    max_per = logits.new_full((num_nodes, logits.shape[-1]), float("-inf"))
    max_per = max_per.scatter_reduce(
        0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=True
    )
    exp = torch.exp(logits - max_per[index])
    denom = logits.new_zeros((num_nodes, logits.shape[-1]))
    denom = denom.index_add(0, index, exp)
    return exp / denom[index].clamp_min(1e-6)


class SE3TransformerLayer(nn.Module):
    """Equivariant attention layer over a radius graph.

    ``h`` are invariant scalar features ``[N, C]``; ``v`` are vector features
    ``[N, Cv, 3]``.  Attention logits are functions of invariant features only,
    and the vector update is a scalar-weighted combination of neighbour
    vectors plus a scalar-weighted relative direction, which preserves
    equivariance.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 8,
                 vector_channels: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} is not divisible by heads={heads}")
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.vector_channels = int(vector_channels)
        inv_dim = self.hidden_dim * 2 + self.edge_dim + 1 + self.vector_channels * 2
        self.attn = nn.Sequential(
            nn.Linear(inv_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.heads),
        )
        msg_dim = self.hidden_dim + self.edge_dim + 1
        self.scalar_msg = nn.Sequential(
            nn.Linear(msg_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.vector_weight = nn.Sequential(
            nn.Linear(msg_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.heads * self.vector_channels),
        )
        self.vector_mix = nn.Linear(self.vector_channels, self.vector_channels, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        # Start as an identity residual; the gate opens during training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor):
        row, col = edge_index
        delta = pos[row] - pos[col]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = delta / distance

        vi = v[row]
        vj = v[col]
        inv = torch.cat(
            [h[row], h[col], edge_attr, distance, vi.norm(dim=-1), vj.norm(dim=-1)], dim=-1
        )
        logits = self.attn(inv).float()
        alpha = _scatter_softmax(logits, row, h.shape[0]).to(dtype=h.dtype)
        alpha = self.dropout(alpha)

        msg_in = torch.cat([h[col], edge_attr, distance], dim=-1)
        scalar_msg = self.scalar_msg(msg_in).view(-1, self.heads, self.head_dim)
        scalar_msg = (alpha.unsqueeze(-1) * scalar_msg).reshape(-1, self.hidden_dim)
        scalar_agg = h.new_zeros(h.shape)
        scalar_agg.index_add_(0, row, scalar_msg.to(dtype=scalar_agg.dtype))
        h = self.norm(h + self.dropout(self.out_proj(scalar_agg)))

        beta = self.vector_weight(msg_in).view(-1, self.heads, self.vector_channels)
        beta = (alpha.unsqueeze(-1) * beta).sum(dim=1)
        alpha_vec = alpha.mean(dim=1, keepdim=True)
        mixed = torch.einsum("ab,ebc->eac", self.vector_mix.weight, v[col])
        vector_msg = beta.unsqueeze(-1) * direction.unsqueeze(1) + alpha_vec.unsqueeze(-1) * mixed
        vector_agg = v.new_zeros(v.shape)
        vector_agg.index_add_(0, row, vector_msg.to(dtype=vector_agg.dtype))
        v = v + self.dropout(vector_agg)
        return h, v


class SE3TransformerPocketEncoder(nn.Module):
    """SE(3)-Transformer pocket encoder with invariant center/mean pooling."""

    def __init__(self, input_dim: int = 547, hidden_dim: int = 512, layers: int = 3,
                 edge_dim: int = 16, heads: int = 8, vector_channels: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.embed = nn.Linear(int(input_dim), int(hidden_dim))
        self.layers = nn.ModuleList(
            [
                SE3TransformerLayer(
                    hidden_dim=int(hidden_dim),
                    edge_dim=self.edge_dim,
                    heads=int(heads),
                    vector_channels=int(vector_channels),
                    dropout=float(dropout),
                )
                for _ in range(int(layers))
            ]
        )
        self.pool = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2 + int(vector_channels), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )

    def forward(self, h_nodes, pos, edge_index, edge_attr, batch_index,
                center_mask=None, return_nodes: bool = False):
        dtype = self.embed.weight.dtype
        h_nodes = h_nodes.to(dtype=dtype)
        pos = pos.to(dtype=dtype)
        edge_attr = edge_attr.to(dtype=dtype)
        h = self.embed(h_nodes)
        n_nodes = int(h.shape[0])
        if n_nodes == 0:
            out = h.new_zeros((1, self.pool[0].out_features))
            return (out, h) if return_nodes else out

        v = h.new_zeros((n_nodes, self.layers[0].vector_channels, 3))
        # Self-loops keep every node attended even when it has no neighbours.
        loop = torch.arange(n_nodes, device=h.device, dtype=edge_index.dtype)
        edge_index = torch.cat([edge_index, torch.stack([loop, loop], dim=0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr.new_zeros((n_nodes, self.edge_dim))], dim=0)

        for layer in self.layers:
            h, v = layer(h, v, pos, edge_index, edge_attr)

        count = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
        sums = h.new_zeros((count, h.shape[-1]))
        sums.index_add_(0, batch_index, h)
        counts = h.new_zeros((count, 1))
        counts.index_add_(0, batch_index, h.new_ones((n_nodes, 1)))
        means = sums / counts.clamp_min(1.0)

        v_norm = v.norm(dim=-1)
        v_sums = v_norm.new_zeros((count, v_norm.shape[-1]))
        v_sums.index_add_(0, batch_index, v_norm)
        v_means = v_sums / counts.clamp_min(1.0)

        if center_mask is None:
            center_idx = torch.full((count,), n_nodes, dtype=torch.long, device=h.device)
            center_idx.scatter_reduce_(
                0, batch_index, torch.arange(n_nodes, device=h.device), reduce="amin", include_self=True
            )
            center_mask = torch.arange(n_nodes, device=h.device) == center_idx[batch_index]
        else:
            center_mask = center_mask.to(device=h.device, dtype=torch.bool)
        candidate = torch.where(
            center_mask.bool(),
            torch.arange(n_nodes, device=h.device),
            torch.full((n_nodes,), n_nodes, device=h.device),
        )
        center_idx = torch.full((count,), n_nodes, dtype=torch.long, device=h.device)
        center_idx.scatter_reduce_(0, batch_index, candidate, reduce="amin", include_self=True)
        center_idx = center_idx.clamp_max(n_nodes - 1)
        center_nodes = h[center_idx]

        out = self.pool(torch.cat([center_nodes, means, v_means], dim=-1))
        if return_nodes:
            return out, h
        return out
