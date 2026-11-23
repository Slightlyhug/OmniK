"""
Enhanced Loss Functions for PTM Prediction
增强的损失函数：Focal Loss + 对比学习 + 困难负样本挖掘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    论文: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 平衡因子，用于平衡正负样本
            gamma: 聚焦参数，用于降低易分类样本的权重
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            focal_loss: scalar or [batch_size]
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    论文: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim] 归一化的特征向量
            labels: [batch_size] 类别标签
        Returns:
            contrastive_loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线（自己与自己的相似度）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算exp(similarity)
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        
        # 计算log_prob
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算每个样本的平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class HardNegativeMiningLoss(nn.Module):
    """
    Hard Negative Mining Loss
    动态挖掘困难负样本并增加其权重
    """
    
    def __init__(self, neg_pos_ratio: float = 3.0, hard_ratio: float = 0.5):
        """
        Args:
            neg_pos_ratio: 负样本与正样本的比例
            hard_ratio: 困难负样本的比例
        """
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.hard_ratio = hard_ratio
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            loss: scalar
        """
        # 计算每个样本的损失
        loss_per_sample = F.cross_entropy(predictions, targets, reduction='none')
        
        # 分离正负样本
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_loss = loss_per_sample[pos_mask]
        neg_loss = loss_per_sample[neg_mask]
        
        # 如果没有正样本或负样本，返回平均损失
        if pos_loss.numel() == 0 or neg_loss.numel() == 0:
            return loss_per_sample.mean()
        
        # 计算需要的负样本数量
        num_pos = pos_loss.numel()
        num_neg_total = neg_loss.numel()
        num_neg_keep = min(int(num_pos * self.neg_pos_ratio), num_neg_total)
        
        # 选择困难负样本
        num_hard_neg = int(num_neg_keep * self.hard_ratio)
        num_easy_neg = num_neg_keep - num_hard_neg
        
        # 按损失排序，选择最困难的负样本
        neg_loss_sorted, neg_indices = torch.sort(neg_loss, descending=True)
        hard_neg_loss = neg_loss_sorted[:num_hard_neg]
        
        # 随机选择一些简单负样本
        if num_easy_neg > 0:
            easy_neg_indices = torch.randperm(num_neg_total - num_hard_neg)[:num_easy_neg]
            easy_neg_loss = neg_loss_sorted[num_hard_neg:][easy_neg_indices]
            selected_neg_loss = torch.cat([hard_neg_loss, easy_neg_loss])
        else:
            selected_neg_loss = hard_neg_loss
        
        # 计算最终损失
        total_loss = torch.cat([pos_loss, selected_neg_loss]).mean()
        
        return total_loss


class EnhancedPTMLoss(nn.Module):
    """
    Enhanced PTM Prediction Loss
    整合Focal Loss、对比学习和困难负样本挖掘
    """
    
    def __init__(self, 
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 contrastive_temperature: float = 0.07,
                 contrastive_weight: float = 0.1,
                 use_hard_mining: bool = True,
                 hard_mining_ratio: float = 0.5):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=contrastive_temperature)
        self.hard_mining_loss = HardNegativeMiningLoss(hard_ratio=hard_mining_ratio) if use_hard_mining else None
        
        self.contrastive_weight = contrastive_weight
        self.use_hard_mining = use_hard_mining
        
    def forward(self, predictions: torch.Tensor, features: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            features: [batch_size, feature_dim] 特征向量（用于对比学习）
            targets: [batch_size] class labels
        Returns:
            loss_dict: {'total_loss', 'focal_loss', 'contrastive_loss', 'hard_mining_loss'}
        """
        # Focal Loss
        if self.use_hard_mining and self.hard_mining_loss is not None:
            classification_loss = self.hard_mining_loss(predictions, targets)
            loss_type = 'hard_mining_loss'
        else:
            classification_loss = self.focal_loss(predictions, targets)
            loss_type = 'focal_loss'
        
        # Contrastive Loss
        contrastive = self.contrastive_loss(features, targets)
        
        # Total Loss
        total_loss = classification_loss + self.contrastive_weight * contrastive
        
        loss_dict = {
            'total_loss': total_loss,
            loss_type: classification_loss,
            'contrastive_loss': contrastive
        }
        
        return loss_dict


class AdaptiveWeightScheduler:
    """
    自适应权重调度器
    根据各类别的性能动态调整损失权重
    """
    
    def __init__(self, initial_weights: list = None, adaptation_rate: float = 0.01):
        if initial_weights is None:
            initial_weights = [1.0, 3.0]  # [neg_weight, pos_weight]
        
        self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        
    def update_weights(self, class_accuracies: list) -> torch.Tensor:
        """
        根据各类别准确率动态调整权重
        
        Args:
            class_accuracies: [neg_acc, pos_acc]
        Returns:
            updated_weights: torch.Tensor
        """
        for i, acc in enumerate(class_accuracies):
            if acc < 0.8:  # 准确率低，增加权重
                self.weights[i] *= (1 + self.adaptation_rate)
            elif acc > 0.95:  # 准确率过高，减少权重
                self.weights[i] *= (1 - self.adaptation_rate)
        
        # 归一化权重
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        
        return self.weights.clone()
    
    def get_weights(self) -> torch.Tensor:
        return self.weights.clone()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    防止模型过度自信，提高泛化能力
    """
    
    def __init__(self, num_classes: int = 2, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        
        return loss


# 测试代码
if __name__ == "__main__":
    # 测试增强损失函数
    batch_size = 32
    num_classes = 2
    feature_dim = 128
    
    # 模拟数据
    predictions = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feature_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 测试Focal Loss
    focal_loss = FocalLoss()
    loss1 = focal_loss(predictions, targets)
    print(f"Focal Loss: {loss1.item():.4f}")
    
    # 测试对比学习损失
    contrastive_loss = SupervisedContrastiveLoss()
    loss2 = contrastive_loss(features, targets)
    print(f"Contrastive Loss: {loss2.item():.4f}")
    
    # 测试困难负样本挖掘
    hard_mining_loss = HardNegativeMiningLoss()
    loss3 = hard_mining_loss(predictions, targets)
    print(f"Hard Mining Loss: {loss3.item():.4f}")
    
    # 测试增强PTM损失
    enhanced_loss = EnhancedPTMLoss()
    loss_dict = enhanced_loss(predictions, features, targets)
    print(f"\nEnhanced PTM Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 测试标签平滑
    label_smoothing = LabelSmoothingLoss()
    loss4 = label_smoothing(predictions, targets)
    print(f"\nLabel Smoothing Loss: {loss4.item():.4f}")
