import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts (MoE) module."""
    
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        """
        Initialize the Sparse MoE module.
        
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            num_experts (int): Number of experts.
            top_k (int): Number of experts to route each token to.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k is not greater than num_experts
        
        # Router for selecting experts
        self.router = nn.Linear(input_dim, num_experts)
        
        # Create experts (each expert is a feed-forward network)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 4 * input_dim),
                nn.GELU(),
                nn.Linear(4 * input_dim, output_dim)
            ) for _ in range(num_experts)
        ])

        self.last_load_balance_loss = None
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim].
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape input to [batch_size * seq_len, input_dim]
        x_flat = x.reshape(-1, self.input_dim)
        
        # Get router logits and probabilities
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts and their probabilities
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        if self.num_experts > 1 and router_probs.numel() > 0:
            topk_assign = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
            load = topk_assign.sum(dim=1).mean(dim=0)
            importance = router_probs.mean(dim=0)
            self.last_load_balance_loss = (
                self.num_experts * (importance * load).sum()
            )
        else:
            self.last_load_balance_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        
        # Initialize output tensor
        final_output = torch.zeros(batch_size * seq_len, self.output_dim, device=x.device)
        
        # Compute outputs for each expert and combine them based on routing probabilities
        for i, expert in enumerate(self.experts):
            # Find indices where this expert is in the top-k
            mask = (top_k_indices == i).any(dim=-1)
            if not mask.any():
                continue
                
            # Get inputs for this expert
            expert_inputs = x_flat[mask]
            
            # Get the probabilities for this expert
            expert_probs = torch.zeros(mask.sum(), device=x.device)
            for j in range(self.top_k):
                expert_probs += top_k_probs[mask, j] * (top_k_indices[mask, j] == i)
                
            # Get expert output
            expert_output = expert(expert_inputs)
            
            # Scale output by expert probabilities
            expert_output = expert_output * expert_probs.unsqueeze(-1)
            
            # Add to final output
            final_output[mask] += expert_output
            
        # Reshape back to [batch_size, seq_len, output_dim]
        final_output = final_output.reshape(batch_size, seq_len, self.output_dim)
        
        return final_output


class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE-based feed-forward network."""
    
    def __init__(self, d_model, nhead, num_experts=4, top_k=2, dropout=0.1):
        """
        Initialize the MoE transformer layer.
        
        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            num_experts (int): Number of experts.
            top_k (int): Number of experts to route each token to.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MoE feed-forward network
        self.feed_forward = SparseMoE(d_model, d_model, num_experts=num_experts, top_k=top_k)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, d_model].
            src_mask: Attention mask.
            src_key_padding_mask: Key padding mask.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Self-attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, 
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        
        # MoE feed-forward
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)

        self.last_load_balance_loss = getattr(self.feed_forward, "last_load_balance_loss", None)
        
        return src

class AttnResMoEEncoderLayer(nn.Module):
    """Transformer layer with both Attention Residuals and MoE-based feed-forward network."""
    
    def __init__(self, hidden_dim: int, num_heads: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        from .dual_branch_modules import DepthAttentionResidual
        self.pre_attn_residual = DepthAttentionResidual(hidden_dim, dropout=dropout)
        self.pre_mlp_residual = DepthAttentionResidual(hidden_dim, dropout=dropout)
        
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        self.feed_forward = SparseMoE(hidden_dim, hidden_dim, num_experts=num_experts, top_k=top_k)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        previous_blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_input = self.attn_norm(self.pre_attn_residual(previous_blocks + [partial_block]))
        attn_out, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        partial_block = partial_block + self.attn_dropout(attn_out)
        
        ffn_input = self.ffn_norm(self.pre_mlp_residual(previous_blocks + [partial_block]))
        ffn_out = self.feed_forward(ffn_input)
        ffn_out = torch.nan_to_num(ffn_out, nan=0.0, posinf=0.0, neginf=0.0)
        partial_block = partial_block + self.ffn_dropout(ffn_out)

        self.last_load_balance_loss = getattr(self.feed_forward, "last_load_balance_loss", None)
        
        return partial_block