import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import logging

# 设置日志
logger = logging.getLogger("causal")


class CausalEncoder(nn.Module):
    """Encoder for causal inference in protein acetylation prediction."""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, use_layer_norm=False):
        """
        Initialize the causal encoder.
        
        Args:
            input_dim (int): Input dimension.
            latent_dim (int): Latent dimension.
            hidden_dim (int): Hidden dimension.
            use_layer_norm (bool): Whether to use layer normalization.
        """
        super().__init__()
        
        # Store layer normalization flag
        self.use_layer_norm = use_layer_norm
        
        # Encoder network with optional layer normalization
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        if self.use_layer_norm:
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.use_layer_norm:
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance for the latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            tuple: (mu, logvar, z)
        """
        # Encode
        h = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z


class StructuralCausalModel(nn.Module):
    """Structural Causal Model (SCM) for protein acetylation."""
    
    def __init__(self, feature_dim, hidden_dim=128, causal_factors=5, use_layer_norm=False):
        """
        Initialize the SCM.
        
        Args:
            feature_dim (int): Feature dimension.
            hidden_dim (int): Hidden dimension.
            causal_factors (int): Number of causal factors.
            use_layer_norm (bool): Whether to use layer normalization.
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.causal_factors = causal_factors
        self.use_layer_norm = use_layer_norm
        
        # Encoder for causal factors
        self.encoder = CausalEncoder(feature_dim, causal_factors, hidden_dim, use_layer_norm)
        
        # Decoder for reconstruction with optional layer normalization
        decoder_layers = []
        decoder_layers.append(nn.Linear(causal_factors, hidden_dim))
        if self.use_layer_norm:
            decoder_layers.append(nn.LayerNorm(hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.use_layer_norm:
            decoder_layers.append(nn.LayerNorm(hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Treatment effect network with optional layer normalization
        treatment_layers = []
        treatment_layers.append(nn.Linear(causal_factors, hidden_dim))
        if self.use_layer_norm:
            treatment_layers.append(nn.LayerNorm(hidden_dim))
        treatment_layers.append(nn.ReLU())
        treatment_layers.append(nn.Linear(hidden_dim, 1))
        
        self.treatment_effect = nn.Sequential(*treatment_layers)
        
        # Outcome prediction network with optional layer normalization
        outcome_layers = []
        outcome_layers.append(nn.Linear(causal_factors + 1, hidden_dim))  # +1 for treatment
        if self.use_layer_norm:
            outcome_layers.append(nn.LayerNorm(hidden_dim))
        outcome_layers.append(nn.ReLU())
        outcome_layers.append(nn.Linear(hidden_dim, 1))
        
        self.outcome = nn.Sequential(*outcome_layers)
        
    def forward(self, x, treatment=None):
        """
        Forward pass.
        
        Args:
            x: Feature tensor.
            treatment: Treatment indicator (1 for acetylation, 0 for non-acetylation).
            
        Returns:
            dict: Dictionary of model outputs.
        """
        # Encode features to causal factors
        mu, logvar, z = self.encoder(x)
        
        # Reconstruction
        x_recon = self.decoder(z)
        
        # If treatment is not provided, estimate it
        if treatment is None:
            # Estimate treatment effect
            treatment_logits = self.treatment_effect(z)
            treatment_prob = torch.sigmoid(treatment_logits)
        else:
            treatment_prob = treatment
            
        # Combine causal factors with treatment
        z_treatment = torch.cat([z, treatment_prob], dim=1)
        
        # Predict outcome (return logits instead of probabilities)
        outcome_logits = self.outcome(z_treatment)

        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'x_recon': x_recon,
            'treatment_prob': treatment_prob,
            'outcome_logits': outcome_logits,
            'predicted_outcome': outcome_logits  # For compatibility
        }
    
    def compute_causal_effect(self, x):
        """
        Compute the Average Treatment Effect (ATE) for a batch of samples.
        
        Args:
            x: Feature tensor.
            
        Returns:
            torch.Tensor: ATE estimates.
        """
        # Encode features to causal factors
        _, _, z = self.encoder(x)
        
        # Outcome when treatment is 1 (acetylated)
        z_treated = torch.cat([z, torch.ones_like(z[:, :1])], dim=1)
        outcome_treated = torch.sigmoid(self.outcome(z_treated))
        
        # Outcome when treatment is 0 (not acetylated)
        z_control = torch.cat([z, torch.zeros_like(z[:, :1])], dim=1)
        outcome_control = torch.sigmoid(self.outcome(z_control))
        
        # Compute ATE
        ate = outcome_treated - outcome_control
        
        return ate
    
    def loss_function(self, x, outputs, treatment=None, outcome=None, beta=1.0):
        """
        Loss function for the SCM.
        
        Args:
            x: Input features.
            outputs: Model outputs.
            treatment: True treatment indicator.
            outcome: True outcome.
            beta: Weight for KL divergence term.
            
        Returns:
            tuple: (total_loss, losses_dict)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['x_recon'], x)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp(), dim=1).mean()
        
        # Treatment prediction loss (if true treatment is provided)
        treatment_loss = 0
        if treatment is not None:
            treatment_loss = F.binary_cross_entropy(outputs['treatment_prob'], treatment)
            
        # Outcome prediction loss (if true outcome is provided)
        outcome_loss = 0
        if outcome is not None:
            # Ensure consistent shapes
            outcome_prob = outputs['outcome_prob']
            if outcome.shape != outcome_prob.shape:
                if len(outcome.shape) < len(outcome_prob.shape):
                    outcome = outcome.view(outcome.shape[0], 1)
                elif len(outcome.shape) > len(outcome_prob.shape):
                    outcome_prob = outcome_prob.squeeze(-1)
            outcome_loss = F.binary_cross_entropy(outcome_prob, outcome)
            
        # Total loss
        total_loss = recon_loss + beta * kl_loss + treatment_loss + outcome_loss
        
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'treatment': treatment_loss.item() if treatment is not None else 0,
            'outcome': outcome_loss.item() if outcome is not None else 0
        }
        
        return total_loss, losses


class DoWhy(nn.Module):
    """
    Implementation of DoWhy-inspired causal inference for acetylation prediction.
    
    This module implements counterfactual analysis to understand the causal effect
    of different protein features on acetylation.
    """
    
    def __init__(self, feature_dim, hidden_dim=128, use_layer_norm=False):
        """
        Initialize the DoWhy module.
        
        Args:
            feature_dim (int): Feature dimension.
            hidden_dim (int): Hidden dimension.
            use_layer_norm (bool): Whether to use layer normalization.
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_layer_norm = use_layer_norm
        
        # Propensity score model (treatment assignment) with optional layer normalization
        propensity_layers = []
        propensity_layers.append(nn.Linear(feature_dim, hidden_dim))
        if self.use_layer_norm:
            propensity_layers.append(nn.LayerNorm(hidden_dim))
        propensity_layers.append(nn.ReLU())
        propensity_layers.append(nn.Linear(hidden_dim, 1))
        propensity_layers.append(nn.Sigmoid())
        
        self.propensity_model = nn.Sequential(*propensity_layers)
        
        # Outcome models for treatment and control
        treated_layers = []
        treated_layers.append(nn.Linear(feature_dim, hidden_dim))
        if self.use_layer_norm:
            treated_layers.append(nn.LayerNorm(hidden_dim))
        treated_layers.append(nn.ReLU())
        treated_layers.append(nn.Linear(hidden_dim, 1))
        treated_layers.append(nn.Sigmoid())
        
        self.outcome_model_treated = nn.Sequential(*treated_layers)
        
        control_layers = []
        control_layers.append(nn.Linear(feature_dim, hidden_dim))
        if self.use_layer_norm:
            control_layers.append(nn.LayerNorm(hidden_dim))
        control_layers.append(nn.ReLU())
        control_layers.append(nn.Linear(hidden_dim, 1))
        control_layers.append(nn.Sigmoid())
        
        self.outcome_model_control = nn.Sequential(*control_layers)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor.
            
        Returns:
            dict: Dictionary of model outputs.
        """
        # Propensity score (probability of treatment)
        propensity = self.propensity_model(x)
        
        # Potential outcomes
        outcome_treated = self.outcome_model_treated(x)
        outcome_control = self.outcome_model_control(x)
        
        # Individual treatment effect (ITE)
        ite = outcome_treated - outcome_control
        
        # Predicted outcome (weighted average)
        predicted_outcome = propensity * outcome_treated + (1 - propensity) * outcome_control
        
        return {
            'propensity': propensity,
            'outcome_treated': outcome_treated,
            'outcome_control': outcome_control,
            'ite': ite,
            'predicted_outcome': predicted_outcome
        }
    
    def estimate_ate(self, x):
        """
        Estimate the Average Treatment Effect (ATE).
        
        Args:
            x: Feature tensor.
            
        Returns:
            torch.Tensor: Estimated ATE.
        """
        outputs = self.forward(x)
        ate = outputs['ite'].mean()
        return ate
    
    def loss_function(self, x, treatment, outcome):
        """
        Loss function for causal inference.
        
        Args:
            x: Feature tensor.
            treatment: Treatment indicator (1 for acetylation, 0 for non-acetylation).
            outcome: True outcome.
            
        Returns:
            tuple: (total_loss, losses_dict)
        """
        outputs = self.forward(x)
        
        # Ensure consistent shapes for propensity score
        propensity = outputs['propensity']
        if treatment.shape != propensity.shape:
            if len(treatment.shape) < len(propensity.shape):
                treatment = treatment.view(treatment.shape[0], 1)
            elif len(treatment.shape) > len(propensity.shape):
                propensity = propensity.squeeze(-1)
        
        # Propensity score loss
        propensity_loss = F.binary_cross_entropy(propensity, treatment)
        
        # Ensure consistent shapes for outcome models
        outcome_treated = outputs['outcome_treated']
        outcome_control = outputs['outcome_control']
        
        if outcome.shape != outcome_treated.shape:
            if len(outcome.shape) < len(outcome_treated.shape):
                outcome = outcome.view(outcome.shape[0], 1)
            elif len(outcome.shape) > len(outcome_treated.shape):
                outcome_treated = outcome_treated.squeeze(-1)
                outcome_control = outcome_control.squeeze(-1)
        
        # Factual outcome loss (depends on treatment status)
        mask_treated = (treatment == 1).float()
        mask_control = (treatment == 0).float()
        
        # Reshape masks if needed
        if len(mask_treated.shape) < len(outcome_treated.shape):
            mask_treated = mask_treated.view(mask_treated.shape[0], 1)
            mask_control = mask_control.view(mask_control.shape[0], 1)
        
        factual_loss = (
            mask_treated * F.binary_cross_entropy(outcome_treated, outcome, reduction='none') +
            mask_control * F.binary_cross_entropy(outcome_control, outcome, reduction='none')
        ).mean()
        
        # Overlap regularization (encourage propensity to be neither 0 nor 1)
        overlap_loss = -torch.log(propensity * (1 - propensity) + 1e-8).mean()
        
        # Total loss
        total_loss = propensity_loss + factual_loss + 0.1 * overlap_loss
        
        losses = {
            'total': total_loss.item(),
            'propensity': propensity_loss.item(),
            'factual': factual_loss.item(),
            'overlap': overlap_loss.item()
        }
        
        return total_loss, losses


class BERTBiLSTMCausalModel(nn.Module):
    """
    基于BERT+BiLSTM架构的因果推断模型

    专门为蛋白质翻译后修饰(PTMs)的因果分析设计，结合了：
    - BERT的长程依赖建模能力
    - BiLSTM的局部序列特征提取
    - 结构因果模型(SCM)的因果推断机制
    """

    def __init__(self, feature_dim, bert_hidden_dim=768, lstm_hidden_dim=128,
                 causal_factors=8, hidden_dim=256, use_layer_norm=True):
        """
        初始化BERT+BiLSTM因果模型

        Args:
            feature_dim (int): 输入特征维度
            bert_hidden_dim (int): BERT隐藏层维度
            lstm_hidden_dim (int): LSTM隐藏层维度
            causal_factors (int): 因果因子数量
            hidden_dim (int): 隐藏层维度
            use_layer_norm (bool): 是否使用层归一化
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.bert_hidden_dim = bert_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.causal_factors = causal_factors
        self.use_layer_norm = use_layer_norm

        logger.info(f"初始化BERT+BiLSTM因果推断模型")
        logger.info(f"- 特征维度: {feature_dim}")
        logger.info(f"- BERT隐藏维度: {bert_hidden_dim}")
        logger.info(f"- LSTM隐藏维度: {lstm_hidden_dim}")
        logger.info(f"- 因果因子数量: {causal_factors}")

        # 1. 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 2. BERT风格编码器（用于长程依赖）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_hidden_dim,
            nhead=8,
            dim_feedforward=bert_hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.bert_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 3. BiLSTM层（用于局部序列特征）
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # BiLSTM输出维度
        self.lstm_output_dim = lstm_hidden_dim * 2

        # 4. 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_hidden_dim + self.lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 5. 因果编码器（变分自编码器风格）
        self.causal_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU()
        )

        # 因果因子的均值和方差
        self.fc_mu = nn.Linear(hidden_dim // 2, causal_factors)
        self.fc_logvar = nn.Linear(hidden_dim // 2, causal_factors)

        # 6. 因果解码器（重构原始特征）
        self.causal_decoder = nn.Sequential(
            nn.Linear(causal_factors, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 7. 处理效应网络（估计处理对结果的影响）
        self.treatment_effect_network = nn.Sequential(
            nn.Linear(causal_factors, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 8. 结果预测网络 (输出logits而不是概率)
        self.outcome_network = nn.Sequential(
            nn.Linear(causal_factors + 1, hidden_dim // 2),  # +1 for treatment
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
            # 移除Sigmoid，输出logits
        )

        # 9. 混杂因子网络（识别潜在的混杂变量）
        self.confounder_network = nn.Sequential(
            nn.Linear(causal_factors, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, causal_factors // 2),
            nn.Sigmoid()
        )

        logger.info("BERT+BiLSTM因果推断模型初始化完成")

    def encode_features(self, x):
        """
        编码输入特征到因果表示空间

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            tuple: (mu, logvar, z) 因果因子的均值、方差和采样值
        """
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影
        projected_input = self.input_projection(x)  # [batch_size, seq_len, bert_hidden_dim]

        # 2. BERT编码
        bert_features = self.bert_encoder(projected_input)  # [batch_size, seq_len, bert_hidden_dim]

        # 3. BiLSTM编码
        lstm_features, _ = self.bilstm(bert_features)  # [batch_size, seq_len, lstm_output_dim]

        # 4. 特征融合
        combined_features = torch.cat([bert_features, lstm_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, hidden_dim]

        # 5. 全局池化（获取序列级表示）
        pooled_features = fused_features.mean(dim=1)  # [batch_size, hidden_dim]

        # 6. 因果编码
        encoded_features = self.causal_encoder(pooled_features)  # [batch_size, hidden_dim // 2]

        # 7. 获取因果因子的分布参数
        mu = self.fc_mu(encoded_features)  # [batch_size, causal_factors]
        logvar = self.fc_logvar(encoded_features)  # [batch_size, causal_factors]

        # 8. 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [batch_size, causal_factors]

        return mu, logvar, z

    def forward(self, x, treatment=None):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]
            treatment: 处理指示器 [batch_size] (可选)

        Returns:
            dict: 包含各种输出的字典
        """
        batch_size = x.size(0)

        # 1. 编码到因果空间
        mu, logvar, z = self.encode_features(x)

        # 2. 重构原始特征
        x_recon = self.causal_decoder(z)  # [batch_size, feature_dim]

        # 3. 估计处理效应
        if treatment is None:
            treatment_prob = self.treatment_effect_network(z)  # [batch_size, 1]
        else:
            treatment_prob = treatment.view(-1, 1).float()

        # 4. 预测结果 (输出logits)
        z_treatment = torch.cat([z, treatment_prob], dim=1)  # [batch_size, causal_factors + 1]
        outcome_logits = self.outcome_network(z_treatment)  # [batch_size, 1]

        # 5. 识别混杂因子
        confounder_scores = self.confounder_network(z)  # [batch_size, causal_factors // 2]

        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'x_recon': x_recon,
            'treatment_prob': treatment_prob,
            'outcome_logits': outcome_logits,
            'predicted_outcome': outcome_logits,  # For compatibility
            'confounder_scores': confounder_scores
        }

    def compute_causal_effect(self, x):
        """
        计算平均处理效应(ATE)

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            torch.Tensor: ATE估计值 [batch_size, 1]
        """
        # 编码特征
        _, _, z = self.encode_features(x)

        # 处理组结果 (treatment = 1)
        z_treated = torch.cat([z, torch.ones(z.size(0), 1, device=z.device)], dim=1)
        outcome_treated = self.outcome_network(z_treated)

        # 对照组结果 (treatment = 0)
        z_control = torch.cat([z, torch.zeros(z.size(0), 1, device=z.device)], dim=1)
        outcome_control = self.outcome_network(z_control)

        # 计算ATE
        ate = outcome_treated - outcome_control

        return ate

    def loss_function(self, x, outputs, treatment=None, outcome=None,
                     beta=1.0, gamma=0.1):
        """
        因果模型的损失函数

        Args:
            x: 原始输入特征
            outputs: 模型输出
            treatment: 真实处理指示器
            outcome: 真实结果
            beta: KL散度权重
            gamma: 混杂因子正则化权重

        Returns:
            tuple: (总损失, 损失字典)
        """
        # 1. 重构损失
        recon_loss = F.mse_loss(outputs['x_recon'], x.mean(dim=1))  # 使用序列均值

        # 2. KL散度损失
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp(),
            dim=1
        ).mean()

        # 3. 处理预测损失
        treatment_loss = 0
        if treatment is not None:
            treatment_loss = F.binary_cross_entropy(
                outputs['treatment_prob'].squeeze(), treatment.float()
            )

        # 4. 结果预测损失
        outcome_loss = 0
        if outcome is not None:
            outcome_loss = F.binary_cross_entropy(
                outputs['outcome_prob'].squeeze(), outcome.float()
            )

        # 5. 混杂因子正则化（鼓励稀疏性）
        confounder_reg = torch.mean(outputs['confounder_scores'])

        # 总损失
        total_loss = (recon_loss + beta * kl_loss + treatment_loss +
                     outcome_loss + gamma * confounder_reg)

        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'treatment': treatment_loss.item() if treatment is not None else 0,
            'outcome': outcome_loss.item() if outcome is not None else 0,
            'confounder_reg': confounder_reg.item()
        }

        return total_loss, losses