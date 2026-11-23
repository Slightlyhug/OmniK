from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer

from .dual_branch_modules import (
    AttentionPooling,
    PositionalEncoding,
    ProjectionHead,
    SequenceTower,
    SpatialGraphModule,
    _sanitize_padding_mask,
    masked_mean,
)


def _infer_layer_count(state_dict: dict, prefix: str, default: int = 4) -> int:
    layer_ids = set()
    marker = f"{prefix}."
    for key in state_dict:
        if not key.startswith(marker):
            continue
        rest = key[len(marker):]
        layer = rest.split(".", 1)[0]
        if layer.isdigit():
            layer_ids.add(int(layer))
    return max(layer_ids) + 1 if layer_ids else default




class OldCrossModalFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.seq_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        if bidirectional:
            self.struct_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.struct_norm = nn.LayerNorm(hidden_dim)
        self.seq_norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_repr, struct_repr, seq_mask=None, struct_mask=None):
        struct_padding, struct_all_masked = _sanitize_padding_mask(struct_mask)
        seq_padding, seq_all_masked = _sanitize_padding_mask(seq_mask)
        seq_context, _ = self.seq_attn(seq_repr, struct_repr, struct_repr, key_padding_mask=struct_padding)
        seq_context = torch.nan_to_num(seq_context, nan=0.0, posinf=0.0, neginf=0.0)
        if struct_all_masked is not None and struct_all_masked.any():
            seq_context = seq_context.clone()
            seq_context[struct_all_masked] = 0.0
        seq_fused = self.seq_norm(seq_repr + seq_context)
        if self.bidirectional:
            struct_context, _ = self.struct_attn(struct_repr, seq_repr, seq_repr, key_padding_mask=seq_padding)
            struct_context = torch.nan_to_num(struct_context, nan=0.0, posinf=0.0, neginf=0.0)
            if seq_all_masked is not None and seq_all_masked.any():
                struct_context = struct_context.clone()
                struct_context[seq_all_masked] = 0.0
            struct_fused = self.struct_norm(struct_repr + struct_context)
        else:
            struct_fused = struct_repr
        return seq_fused, struct_fused


class OldAdaptiveGating(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, seq_global, struct_global):
        if struct_global is None:
            weights = seq_global.new_ones(seq_global.shape)
            return seq_global, weights
        weights = torch.sigmoid(self.gate(torch.cat([seq_global, struct_global], dim=-1)))
        return weights * seq_global + (1.0 - weights) * struct_global, weights


class OldStructureTower(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        use_attention_pooling: bool = True,
        window_radius: int = 3,
    ):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.window_radius = window_radius
        mid_dim = max(hidden_dim, input_dim // 2)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.spatial_gnn = SpatialGraphModule(hidden_dim, num_heads=num_heads, dropout=dropout)
        try:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
                enable_nested_tensor=False,
            )
        except TypeError:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
            )
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.spatial_residual_weight = nn.Parameter(torch.tensor(0.5))
        if self.use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim, max(32, hidden_dim // 4))
            self.global_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
            self.global_norm = nn.LayerNorm(hidden_dim)
            self.dropout_layer = nn.Dropout(dropout)
            self.center_dropout = nn.Dropout(dropout)

    def forward(self, structure_features, mask=None, center_alpha: float = 1.0, return_weights: bool = False, distance_matrices=None):
        structure_features = torch.nan_to_num(structure_features, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        projection_dtype = structure_features.dtype
        with torch.autocast(device_type=structure_features.device.type, enabled=False):
            hidden = self.projection(structure_features.float())
        hidden = self.pos_encoder(hidden.to(dtype=projection_dtype))
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        padding_mask, all_masked = _sanitize_padding_mask(mask)
        spatial_context = self.spatial_gnn(hidden, distance_matrix=distance_matrices, mask=padding_mask)
        curr = spatial_context
        for layer in self.layers:
            curr = layer(curr, src_key_padding_mask=padding_mask)
            curr = torch.nan_to_num(curr, nan=0.0, posinf=0.0, neginf=0.0)
            curr = curr + self.spatial_residual_weight * spatial_context
        contextual = self.post_norm(curr)
        contextual = torch.nan_to_num(contextual, nan=0.0, posinf=0.0, neginf=0.0)
        if all_masked is not None and all_masked.any():
            contextual = contextual.clone()
            contextual[all_masked] = 0.0
        attn_weights = None
        if self.use_attention_pooling:
            pooled_context, weights = self.attention_pool(contextual, padding_mask)
            attn_weights = weights
            center_idx = contextual.size(1) // 2
            center_feature = contextual[:, center_idx, :] if center_idx < contextual.size(1) else pooled_context
            center_feature = self.center_dropout(center_feature)
            if center_alpha < 0.9999:
                center_feature = center_feature * center_alpha
            global_repr = self.global_fusion(torch.cat([center_feature, pooled_context], dim=-1))
            global_repr = self.global_norm(self.dropout_layer(global_repr))
        else:
            global_repr = masked_mean(contextual, padding_mask)
        if return_weights:
            return contextual, global_repr, attn_weights
        return contextual, global_repr


def build_compat_config(config: dict, state_dict: dict) -> dict:
    """Normalize current config files to the architecture stored in old checkpoints."""
    compat = dict(config)
    compat["use_moe"] = False
    compat["use_block_attn_res"] = False
    compat["use_ptm_specialists"] = False
    compat["use_ptm_feature_separator"] = False
    compat["use_structure_global_context"] = False
    compat["num_tasks"] = 1
    compat["multitask_ptm_types"] = []
    compat["micro_env_feature_dim"] = 6

    compat["seq_transformer_layers"] = _infer_layer_count(
        state_dict,
        "sequence_tower.encoder.layers",
        int(compat.get("seq_transformer_layers", 4)),
    )
    compat["struct_transformer_layers"] = _infer_layer_count(
        state_dict,
        "structure_tower.layers",
        int(compat.get("struct_transformer_layers", 4)),
    )
    return compat


class PhosphorylationCheckpointCompatPredictor(nn.Module):
    """Prediction-only model matching the stored phosphorylation checkpoints.

    The training checkpoints in this project predate the current
    DualBranchFusionPredictor implementation. This class preserves the old
    module names and tensor shapes so checkpoint loading can be strict.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.seq_input_dim = int(config.get("seq_feature_dim", 1280))
        self.local_input_dim = self.seq_input_dim
        self.global_input_dim = self.seq_input_dim
        self.struct_input_dim = int(config.get("struct_feature_dim", 512))
        self.branch_hidden_dim = int(config.get("branch_hidden_dim", 512))
        self.fusion_hidden_dim = int(config.get("fusion_hidden_dim", 512))
        self.dropout = float(config.get("dropout", 0.25))
        self.use_structure = bool(config.get("use_structure", True))
        self.use_explicit_features = bool(config.get("use_explicit_features", True))
        self.residue_embedding_dim = int(config.get("aa_embedding_dim", 64))
        self.position_embedding_dim = int(config.get("position_embedding_dim", 32))
        self.micro_env_feature_dim = int(config.get("micro_env_feature_dim", 6))

        if self.use_explicit_features:
            self.residue_embedding = nn.Embedding(25, self.residue_embedding_dim)
            self.position_embedding = nn.Embedding(512, self.position_embedding_dim)
            self.seq_input_dim += self.residue_embedding_dim + self.position_embedding_dim
            if self.use_structure:
                self.struct_input_dim += (
                    self.residue_embedding_dim
                    + self.position_embedding_dim
                    + self.micro_env_feature_dim
                )

        seq_layers = int(config.get("seq_transformer_layers", 4))
        struct_layers = int(config.get("struct_transformer_layers", 4))
        seq_heads = int(config.get("seq_attention_heads", 16))
        struct_heads = int(config.get("struct_attention_heads", 16))
        cross_heads = int(config.get("cross_attention_heads", 16))
        use_attention_pooling = bool(config.get("use_attention_pooling", True))
        window_radius = int(config.get("center_window_radius", 5))

        self.sequence_tower = SequenceTower(
            input_dim=self.seq_input_dim,
            hidden_dim=self.branch_hidden_dim,
            num_layers=seq_layers,
            num_heads=seq_heads,
            dropout=self.dropout,
            use_attention_pooling=use_attention_pooling,
            window_radius=window_radius,
            use_moe=False,
        )
        self.structure_tower = OldStructureTower(
            input_dim=self.struct_input_dim,
            hidden_dim=self.branch_hidden_dim,
            dropout=self.dropout,
            num_layers=struct_layers,
            num_heads=struct_heads,
            use_attention_pooling=use_attention_pooling,
            window_radius=window_radius,
        )
        self.cross_modal = OldCrossModalFusionLayer(
            hidden_dim=self.branch_hidden_dim,
            num_heads=cross_heads,
            dropout=self.dropout,
        )
        self.gating = OldAdaptiveGating(self.branch_hidden_dim)

        self.local_projection = nn.Sequential(
            nn.Linear(self.local_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.bilinear_interaction = nn.Bilinear(
            self.branch_hidden_dim,
            self.branch_hidden_dim,
            self.branch_hidden_dim,
        )

        fusion_input_dim = self.branch_hidden_dim * 7 + self.micro_env_feature_dim
        self.fusion_proj = nn.Linear(fusion_input_dim, self.fusion_hidden_dim)
        self.fusion_residual_block = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
        )
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
            nn.LayerNorm(self.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout / 2.0),
        )
        self.classifier = nn.Linear(self.fusion_hidden_dim // 2, 1)

        self.seq_classifier = nn.Linear(self.branch_hidden_dim, 1)
        self.struct_classifier = nn.Linear(self.branch_hidden_dim, 1)
        self.projection_head = ProjectionHead(
            input_dim=self.fusion_hidden_dim,
            projection_dim=self.fusion_hidden_dim,
            hidden_dim=self.fusion_hidden_dim,
            dropout=self.dropout,
        )

    @staticmethod
    def _extract_mask(features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if features is None or features.ndim < 3:
            return None
        return torch.isclose(features.abs().sum(dim=-1), torch.zeros((), device=features.device))

    def _explicit_features(
        self,
        batch: dict,
        seq_window: torch.Tensor,
        struct_tensor: Optional[torch.Tensor],
        micro_env: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_explicit_features:
            return seq_window, struct_tensor

        sequence_block = batch.get("sequence", {})
        residue_types = sequence_block.get("residue_types") if isinstance(sequence_block, dict) else None
        if residue_types is None:
            residue_types = batch.get("residue_types")
        if residue_types is None:
            residue_types = torch.zeros(
                seq_window.size(0),
                seq_window.size(1),
                device=seq_window.device,
                dtype=torch.long,
            )
        residue_types = residue_types.to(seq_window.device).long().clamp(0, 24)

        res_emb = self.residue_embedding(residue_types)
        pos_ids = torch.arange(seq_window.size(1), device=seq_window.device)
        pos_ids = pos_ids.expand(seq_window.size(0), -1).clamp(0, 511)
        pos_emb = self.position_embedding(pos_ids)

        seq_window = torch.cat([seq_window, res_emb, pos_emb], dim=-1)
        if struct_tensor is not None:
            micro_expanded = micro_env.unsqueeze(1).expand(-1, struct_tensor.size(1), -1)
            struct_tensor = torch.cat([struct_tensor, res_emb, pos_emb, micro_expanded], dim=-1)
        return seq_window, struct_tensor

    def forward(self, batch: dict, return_importance: bool = False):
        sequence_block = batch.get("sequence", {})
        seq_window = sequence_block.get("window_features") if isinstance(sequence_block, dict) else None
        if seq_window is None:
            seq_window = batch.get("sequence_features")
        if seq_window is None:
            raise ValueError("Batch is missing sequence window features")

        struct_tensor = batch.get("structure")
        if struct_tensor is None:
            struct_tensor = batch.get("structure_features")

        micro_env = batch.get("micro_env_features")
        if micro_env is None:
            micro_env = torch.zeros(
                seq_window.size(0),
                self.micro_env_feature_dim,
                device=seq_window.device,
                dtype=seq_window.dtype,
            )
        else:
            micro_env = micro_env.to(seq_window.device, dtype=seq_window.dtype)
            if micro_env.size(-1) < self.micro_env_feature_dim:
                micro_env = F.pad(micro_env, (0, self.micro_env_feature_dim - micro_env.size(-1)))
            elif micro_env.size(-1) > self.micro_env_feature_dim:
                micro_env = micro_env[..., : self.micro_env_feature_dim]

        seq_mask = self._extract_mask(seq_window)
        struct_mask = self._extract_mask(struct_tensor)
        seq_window, struct_tensor = self._explicit_features(batch, seq_window, struct_tensor, micro_env)

        distance_matrices = batch.get("distance_matrix")
        if distance_matrices is None and isinstance(sequence_block, dict):
            distance_matrices = sequence_block.get("distance_matrix")
        if isinstance(distance_matrices, torch.Tensor):
            distance_matrices = distance_matrices.to(seq_window.device)

        if return_importance:
            seq_context, seq_global, seq_weights = self.sequence_tower(
                seq_window,
                seq_mask,
                return_weights=True,
            )
        else:
            seq_context, seq_global = self.sequence_tower(seq_window, seq_mask)
            seq_weights = None

        if struct_tensor is not None:
            if return_importance:
                struct_context, struct_global, struct_weights = self.structure_tower(
                    struct_tensor,
                    struct_mask,
                    return_weights=True,
                    distance_matrices=distance_matrices,
                )
            else:
                struct_context, struct_global = self.structure_tower(
                    struct_tensor,
                    struct_mask,
                    distance_matrices=distance_matrices,
                )
                struct_weights = None
            seq_context, struct_context = self.cross_modal(seq_context, struct_context, seq_mask, struct_mask)
            seq_global = masked_mean(seq_context, seq_mask)
            struct_global = masked_mean(struct_context, struct_mask)
        else:
            struct_global = torch.zeros_like(seq_global)
            struct_weights = None

        fused_global, gate_weights = self.gating(seq_global, struct_global)
        local_features = sequence_block.get("local_features") if isinstance(sequence_block, dict) else None
        if local_features is None:
            local_features = batch.get("local_features")
        global_features = sequence_block.get("global_features") if isinstance(sequence_block, dict) else None
        if global_features is None:
            global_features = batch.get("global_features")

        local_repr = self.local_projection(local_features.mean(dim=1)) if local_features is not None else torch.zeros_like(seq_global)
        global_repr = self.global_projection(global_features) if global_features is not None else torch.zeros_like(seq_global)

        struct_confidence = 1.0 - gate_weights
        interaction = self.bilinear_interaction(seq_global, struct_global) * struct_confidence
        abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
        fusion_inputs = torch.cat(
            [
                fused_global,
                seq_global,
                struct_global,
                abs_diff,
                interaction,
                local_repr,
                global_repr,
                micro_env,
            ],
            dim=-1,
        )

        x = self.fusion_proj(fusion_inputs)
        residual = x
        x = residual + self.fusion_residual_block(x)
        fused_representation = self.fusion_bottleneck(x)
        logits = self.classifier(fused_representation).squeeze(-1)

        if return_importance:
            return logits, {
                "window_importance": seq_weights,
                "structure_importance": struct_weights,
                "gate_weights": gate_weights,
            }
        return {"logits": logits, "fusion_weights": gate_weights}


def load_phosphorylation_checkpoint(model: nn.Module, state_dict: dict, logger=None) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    if logger is not None:
        logger.info("Checkpoint loaded strictly with %d tensors.", len(state_dict))
