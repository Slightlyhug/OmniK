"""SE(3)-equivariant message passing for residue-centered pocket graphs."""

import torch
import torch.nn as nn


class SE3GraphLayer(nn.Module):
    """Scalar/vector message passing with equivariant coordinate directions."""

    def __init__(self, hidden_dim: int, edge_dim: int, vector_channels: int = 16):
        super().__init__()
        self.vector_channels = int(vector_channels)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + vector_channels * 2, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim + vector_channels),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + vector_channels, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_mix = nn.Linear(vector_channels, vector_channels, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, v, pos, edge_index, edge_attr):
        row, col = edge_index
        delta = pos[row] - pos[col]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = delta / distance
        vi, vj = v[row], v[col]
        vi_norm = vi.norm(dim=-1)
        vj_norm = vj.norm(dim=-1)
        msg_out = self.message(torch.cat([h[row], h[col], edge_attr, vi_norm, vj_norm], dim=-1))
        scalar_msg, vector_weight = msg_out[:, :h.shape[-1]], msg_out[:, h.shape[-1]:]
        mixed_vj = torch.einsum("ab,ebc->eac", self.vector_mix.weight, vj)
        vector_msg = vector_weight.unsqueeze(-1) * direction.unsqueeze(1) + mixed_vj
        scalar_agg = h.new_zeros(h.shape)
        vector_agg = v.new_zeros(v.shape)
        # AMP keeps some parameter-backed operations in fp32.  Match the
        # destination dtype before in-place aggregation so bf16/fp16 graph
        # batches do not fail in ``index_add_``.
        scalar_agg.index_add_(0, row, scalar_msg.to(dtype=scalar_agg.dtype))
        vector_agg.index_add_(0, row, vector_msg.to(dtype=vector_agg.dtype))
        degree = h.new_zeros((h.shape[0], 1))
        degree.index_add_(0, row, h.new_ones((row.numel(), 1)))
        scalar_agg = scalar_agg / degree.clamp_min(1.0)
        vector_agg = vector_agg / degree.unsqueeze(-1).clamp_min(1.0)
        h = self.norm(h + self.update(torch.cat([h, scalar_agg, vector_agg.norm(dim=-1)], dim=-1)))
        return h, v + vector_agg


class PocketEGNN(nn.Module):
    """SE(3)-equivariant pocket encoder with invariant center/mean pooling."""

    def __init__(self, input_dim=546, hidden_dim=512, layers=3, edge_dim=16, vector_channels=16):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.embed = nn.Linear(int(input_dim), int(hidden_dim))
        self.layers = nn.ModuleList([
            SE3GraphLayer(hidden_dim, self.edge_dim, vector_channels) for _ in range(int(layers))
        ])
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )

    def forward(self, h_nodes, pos, edge_index, edge_attr, batch_index,
                center_mask=None, return_nodes: bool = False):
        dtype = self.embed.weight.dtype
        h_nodes = h_nodes.to(dtype=dtype)
        pos = pos.to(dtype=dtype)
        edge_attr = edge_attr.to(dtype=dtype)
        h = self.embed(h_nodes)
        v = h.new_zeros((h.shape[0], self.layers[0].vector_channels, 3))
        for layer in self.layers:
            h, v = layer(h, v, pos, edge_index, edge_attr)
        count = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
        sums = h.new_zeros((count, h.shape[-1]))
        sums.index_add_(0, batch_index, h)
        counts = h.new_zeros((count, 1))
        counts.index_add_(0, batch_index, h.new_ones((h.shape[0], 1)))
        means = sums / counts.clamp_min(1.0)
        if center_mask is None:
            center_idx = torch.full((count,), h.shape[0], dtype=torch.long, device=h.device)
            center_idx.scatter_reduce_(
                0, batch_index, torch.arange(h.shape[0], device=h.device), reduce="amin", include_self=True
            )
            center_mask = torch.arange(h.shape[0], device=h.device) == center_idx[batch_index]
        else:
            center_mask = center_mask.to(device=h.device, dtype=torch.bool)
        candidate = torch.where(center_mask.bool(), torch.arange(h.shape[0], device=h.device),
                                torch.full((h.shape[0],), h.shape[0], device=h.device))
        center_idx = torch.full((count,), h.shape[0], dtype=torch.long, device=h.device)
        center_idx.scatter_reduce_(0, batch_index, candidate, reduce="amin", include_self=True)
        center_idx = center_idx.clamp_max(h.shape[0] - 1)
        center_nodes = h[center_idx]
        pooled = self.pool(torch.cat([center_nodes, means], dim=-1))
        if return_nodes:
            return pooled, h
        return pooled
