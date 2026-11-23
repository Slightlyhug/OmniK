import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSupConLoss(nn.Module):
    """
    Advanced Supervised Contrastive Learning Loss with Label Smoothing 
    and Dynamic Hard Negative Mining.
    """
    def __init__(self, temperature: float = 0.1, label_smoothing: float = 0.05, hard_negative_ratio: float = 0.3):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.label_smoothing = float(max(0.0, min(1.0, label_smoothing)))
        self.hard_negative_ratio = float(max(0.0, min(1.0, hard_negative_ratio)))
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [batch, dim] (e.g., from Projection Head)
        labels: [batch] (binary labels for a specific PTM, e.g., Acetylation)
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D [batch, dim], got shape={tuple(features.shape)}")

        device = features.device
        features = F.normalize(features.float(), dim=1)
        labels = labels.to(device=device).view(-1)
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())
        if labels.numel() != batch_size:
            raise ValueError(
                f"labels must have the same batch size as features, got labels={labels.numel()} features={batch_size}"
            )
        
        # [Batch, Batch] Similarity Matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        logits_max = sim_matrix.max(dim=1, keepdim=True).values.detach()
        sim_matrix = sim_matrix - logits_max
        
        # Mask for self-contrast
        mask_self = torch.eye(batch_size, device=device).bool()
        
        # Mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        
        # Apply label smoothing to positive mask
        if self.label_smoothing > 0:
            mask_pos = mask_pos * (1 - self.label_smoothing) + self.label_smoothing / batch_size
        
        # Mask out self-contrast from positives
        mask_pos = mask_pos.masked_fill(mask_self, 0.0)
        
        # --- Dynamic Hard Negative Mining ---
        # Find negatives (different labels)
        mask_neg = (~torch.eq(labels, labels.T)) & (~mask_self)
        
        # We only want to keep the hardest negatives (those with highest similarity)
        # To do this smoothly, we can weight them or just filter them.
        # For each anchor, get similarities of negative samples
        neg_sim = sim_matrix.masked_fill(~mask_neg, float("-inf"))
        
        # Sort negatives by similarity (descending)
        num_negs = mask_neg.sum(dim=1)
        k_hard = torch.clamp((num_negs * self.hard_negative_ratio).long(), min=1)
        
        # Create a hard negative mask
        hard_neg_mask = torch.zeros_like(mask_neg, dtype=torch.float32)
        for i in range(batch_size):
            if num_negs[i] > 0:
                _, topk_idx = torch.topk(neg_sim[i], k=k_hard[i].item())
                hard_neg_mask[i, topk_idx] = 1.0
                
        # The denominator should include positives and ONLY HARD negatives
        denom_mask = mask_pos.bool() | hard_neg_mask.bool()
        denom_logits = sim_matrix.masked_fill(~denom_mask, float("-inf"))
        denom_logsumexp = torch.logsumexp(denom_logits, dim=1, keepdim=True)
        log_prob = sim_matrix - denom_logsumexp
        
        # Mean log-prob over positives
        mask_pos_sum = mask_pos.sum(1)
        valid_rows = mask_pos_sum > 0
        if not bool(valid_rows.any()):
            return features.new_zeros(())
        mask_pos_sum = torch.where(valid_rows, mask_pos_sum, torch.ones_like(mask_pos_sum))
        
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos_sum
        
        loss = -mean_log_prob_pos[valid_rows].mean()
        return loss.to(dtype=features.dtype)
