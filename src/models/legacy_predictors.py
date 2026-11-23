import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import contextlib

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

try:
    from src.utils.adaptive_dropout import create_adaptive_dropout_layer
except ImportError:
    create_adaptive_dropout_layer = None
from .dual_branch_modules import (
    SequenceTower,
    StructureTower,
    GatedFusionModule,
    SequenceContextEncoder,
    StructureContextEncoder,
    CrossModalFusionLayer,
    AdaptiveGating,
    ProjectionHead,
    DepthAttentionResidual,
    masked_mean
)
from .enhanced_components import (
    EnhancedAttentionModule,
    MultiScaleFusionNetwork,
    DynamicFeatureSelector
)
from .encoders import (
    MultiModalFusion,
    ProjectionLayer,
    ProteinSequenceEncoder,
    ProteinStructureEncoder,
    AcetylationInfoEncoder
)
from .task_separation import (
    AbstractStructureGraphBuilder,
    AbstractTaskContextGenerator,
    AbstractTaskParser,
    ConfigDrivenTaskParser,
    DefaultStructureGraphBuilder,
    LaSTTaskContextGenerator,
    StructureGraphContext,
    TaskSpecificContextProjector,
)
from .ptm_disentangled_separator import PTMDisentangledSeparator
from .learnable_task_attention import LearnableTaskAttentionPool
try:
    from .esm_lora import ESM2LoRAWrapper
except Exception:
    ESM2LoRAWrapper = None
try:
    from .egnn_3d_graph import PocketEGNN
except Exception:
    PocketEGNN = None
try:
    from .se3_transformer_graph import SE3TransformerPocketEncoder
except Exception:
    SE3TransformerPocketEncoder = None
from src.utils.enhanced_loss import FocalLoss
import numpy as np
import math
import warnings
import logging
from typing import List, Optional

# Import matplotlib and seaborn only when needed to avoid dependency issues
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    plt = None
    sns = None
    logging.warning("Failed to import visualization libraries: %s", e)
except Exception as e:
    HAS_PLOTTING = False
    plt = None
    sns = None
    logging.warning("Error importing visualization libraries: %s", e)

# 设置日志
logger = logging.getLogger("acetylation_predictor")


def normalize_sequence_strings(sequence_strings: Any) -> List[str]:
    if sequence_strings is None:
        return []
    if isinstance(sequence_strings, str):
        return [sequence_strings]
    if isinstance(sequence_strings, (list, tuple)):
        return [str(s) if s is not None else "" for s in sequence_strings]
    return [str(sequence_strings)]


def compute_non_empty_sequence_indices(sequence_strings: Any) -> Tuple[List[str], List[int]]:
    seq_list = normalize_sequence_strings(sequence_strings)
    non_empty_idx = [idx for idx, seq in enumerate(seq_list) if len(seq.strip()) > 0]
    return seq_list, non_empty_idx


def merge_non_empty_lora_features(
    base_features: torch.Tensor,
    non_empty_indices: List[int],
    encoded_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if not isinstance(base_features, torch.Tensor):
        raise TypeError("base_features must be a torch.Tensor")
    if encoded_features is None or len(non_empty_indices) == 0:
        return base_features
    merged = base_features.clone()
    for row_idx, base_idx in enumerate(non_empty_indices):
        merged[base_idx] = encoded_features[row_idx]
    return merged


class InterpretabilityModule(nn.Module):
    """可解释性模块，用于分析模型的预测原因"""
    
    def __init__(self, input_dim, hidden_dim=128):
        """
        初始化可解释性模块
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
        """
        super().__init__()
        
        self.importance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, return_importance=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            return_importance (bool): 是否返回重要性分数
            
        Returns:
            torch.Tensor or tuple: 输出和重要性分数
        """
        # 计算每个位置的重要性分数
        importance = self.importance_network(x).squeeze(-1)  # [batch_size, seq_len]
        
        # 添加注意力机制，赋予每个氨基酸位点权重
        attention_weights = F.softmax(importance, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]
        
        # 加权求和获得上下文向量
        context = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        
        if return_importance:
            return context, attention_weights.squeeze(2)
        else:
            return context


class AcetylationPredictor(nn.Module):
    """
    Modular acetylation prediction model that can use different encoders and architectures.
    基于ProteinGPT优化，整合序列编码器、结构编码器和乙酰化信息编码器
    """
    
    def __init__(self, config=None, seq_encoder_type="esm2_t33_650M_UR50D", struct_encoder_type="esm_if1_gvp4_t16_142M_UR50",
                 core_model_type="bert_bilstm", use_moe=False, use_reinforcement=False, use_meta_learning=False,
                 use_causal=False, use_interpretability=True, use_precomputed_features=True):
        """
        初始化乙酰化预测器

        Args:
            config: 配置对象或字典
            seq_encoder_type: 序列编码器类型
            struct_encoder_type: 结构编码器类型
            core_model_type: 核心模型类型 ('transformer', 'bert_bilstm', 'ptm_bert_bilstm')
                - 'transformer': 标准Transformer架构
                - 'bert_bilstm': BERT+BiLSTM混合架构（推荐用于PTMs预测）
                - 'ptm_bert_bilstm': PTM特异性BERT+BiLSTM架构
            use_moe: 是否使用混合专家模型
            use_reinforcement: 是否使用强化学习
            use_meta_learning: 是否使用元学习
            use_causal: 是否使用因果推理
            use_interpretability: 是否使用可解释性方法
            use_precomputed_features: 是否使用预计算特征（跳过编码器加载）
        """
        super().__init__()
        
        # 检查和打印设备信息
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"AcetylationPredictor初始化在设备: {self.device}")
        
        # 存储配置
        self.config = config or {}
        self.use_moe = use_moe
        self.use_reinforcement = use_reinforcement
        self.use_meta_learning = use_meta_learning
        self.use_causal = use_causal
        self.use_interpretability = use_interpretability
        self.use_precomputed_features = use_precomputed_features
        self.use_multitask_loss = False  # 将在acetylation_transformer模型中设置为True
        
        # [新增] 是否使用One-hot编码拼接 (Hybrid Encoding)
        # 默认为True，如果use_explicit_features也为True
        self.use_one_hot = self.config.get("use_one_hot", self.config.get("use_explicit_features", False))
        if self.use_one_hot:
            logger.info("启用Hybrid Encoding: 将One-hot编码与ESM特征拼接")

        # 获取基本维度配置
        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.projection_dim = self.config.get("projection_dim", 256)
        self.dropout = self.config.get("dropout", 0.1)
        self.bert_hidden_dim = self.config.get("bert_hidden_dim", 384)
        self.config.setdefault("bert_hidden_dim", self.bert_hidden_dim)

        # 统计并记录PTM与残基、位置配置
        ptm_mapping = self.config.get("ptm_type_to_id", {})
        inferred_num_ptm_types = max(1, len(ptm_mapping))
        self.num_ptm_types = max(self.config.get("num_ptm_types", inferred_num_ptm_types), inferred_num_ptm_types)
        self.config["num_ptm_types"] = self.num_ptm_types

        self.num_residue_types = self.config.get("num_residue_types", 35)
        self.config.setdefault("num_residue_types", self.num_residue_types)

        self.max_position_embeddings = self.config.get("max_position_embeddings", 512)
        self.config.setdefault("max_position_embeddings", self.max_position_embeddings)
            
        # 编码器初始化（根据是否使用预计算特征决定）
        if self.use_precomputed_features:
            # 使用预计算特征，跳过编码器加载
            logger.info("使用预计算特征，跳过ESM编码器加载")
            self.seq_encoder = None
            self.struct_encoder = None
            # 设置预计算特征的默认维度（根据ESM-2和ESM-IF的标准维度）
            self.seq_dim = 1280  # ESM-2 t33 650M 的特征维度
            
            # [新增] 如果启用One-hot，增加维度
            if self.use_one_hot:
                self.seq_dim += self.num_residue_types
                logger.info(f"启用Hybrid Encoding，序列特征维度增加至: {self.seq_dim} (1280 + {self.num_residue_types})")

            self.struct_dim = 512  # ESM-IF 的特征维度
            self.seq_projection = ProjectionLayer(self.seq_dim, self.projection_dim)
            self.struct_projection = ProjectionLayer(self.struct_dim, self.projection_dim)
            self.use_structure = True
            logger.info(f"预计算特征维度 - 序列: {self.seq_dim}, 结构: {self.struct_dim}")
        else:
            # 序列编码器
            self.seq_encoder = ProteinSequenceEncoder(model_name=seq_encoder_type, freeze=True)
            self.seq_projection = ProjectionLayer(self.seq_encoder.embed_dim, self.projection_dim)
            self.seq_dim = self.seq_encoder.embed_dim
            logger.info(f"序列编码器初始化完成，维度: {self.seq_dim}")

            # 结构编码器 (可选，仅当提供结构数据时使用)
            try:
                self.struct_encoder = ProteinStructureEncoder(model_name=struct_encoder_type, freeze=True)
                self.struct_projection = ProjectionLayer(self.struct_encoder.embed_dim, self.projection_dim)
                self.struct_dim = self.struct_encoder.embed_dim
                self.use_structure = True
                logger.info(f"结构编码器初始化完成，维度: {self.struct_dim}")
            except Exception as e:
                logger.warning(f"结构编码器初始化失败: {e}")
                self.use_structure = False
                self.struct_dim = 0
        
        # 乙酰化信息编码器
        self.acetylation_encoder = AcetylationInfoEncoder(
            input_dim=21,  # 20个氨基酸 + 1个填充
            hidden_dim=128,
            output_dim=self.projection_dim
        )
        logger.info("乙酰化信息编码器初始化完成")
        
        # 多模态融合层
        if self.use_structure:
            self.fusion = MultiModalFusion(feature_dim=self.projection_dim, num_heads=4)
            logger.info("多模态融合层初始化完成")
        else:
            self.fusion = None
            logger.info("未初始化多模态融合层 (use_structure=False)")

        # 显式特征嵌入层
        self.position_embedding_layer = nn.Embedding(self.max_position_embeddings, self.projection_dim)
        self.residue_type_embedding = nn.Embedding(self.num_residue_types, self.projection_dim)
        # 注意：ptm_type_embedding 保留用于向后兼容性和模型加载，但在 forward 中不再使用
        # 这是为了修复标签泄露问题（见 forward 方法中的注释）
        self.ptm_type_embedding = nn.Embedding(self.num_ptm_types, self.projection_dim)
        self.relative_position_projection = nn.Sequential(
            nn.Linear(1, self.projection_dim),
            nn.Tanh()
        )
        self.site_indicator_projection = nn.Sequential(
            nn.Linear(1, self.projection_dim),
            nn.Sigmoid()
        )
        
        # 局部序列投影层
        # 注意：如果启用了Hybrid Encoding，seq_dim包含了One-hot维度
        # 但我们暂时只对全局序列特征应用One-hot拼接
        local_input_dim = self.seq_dim
        if self.use_one_hot and self.use_precomputed_features:
             local_input_dim = self.seq_dim - self.num_residue_types
             
        self.local_seq_projection = ProjectionLayer(local_input_dim, self.projection_dim)

        # 全局上下文投影（用于融合额外的全局特征）
        # 全局特征通常也是原始ESM特征的聚合，不包含One-hot
        self.global_feature_dim = self.config.get("global_feature_dim", local_input_dim)
        self.global_context_projection = ProjectionLayer(self.global_feature_dim, self.projection_dim)
        
        # 可解释性模块
        if self.use_interpretability:
            self.local_interpretability = InterpretabilityModule(self.projection_dim)
            self.global_interpretability = InterpretabilityModule(self.projection_dim)
            logger.info("可解释性模块初始化完成")
        else:
            self.local_interpretability = None
            self.global_interpretability = None

        # 中心特征正则化（硬遮挡 + 课程调度）
        self._init_center_regularization()
        
        # 核心模型 - 支持多种架构类型
        if core_model_type == "transformer":
            if use_moe:
                # 使用MoE的Transformer
                logger.info("使用MoE Transformer作为核心模型")
                self.core_model = nn.TransformerEncoder(
                    MoETransformerLayer(
                        d_model=self.projection_dim,
                        nhead=self.config.get("num_heads", 8),
                        num_experts=self.config.get("num_experts", 4),
                        top_k=self.config.get("top_k", 2),
                        dropout=self.dropout
                    ),
                    num_layers=self.config.get("num_layers", 6)
                )
            else:
                # 标准Transformer
                logger.info("使用标准Transformer作为核心模型")
                self.core_model = TransformerModel(
                    input_dim=self.projection_dim,
                    nhead=self.config.get("num_heads", 8),
                    num_layers=self.config.get("num_layers", 6),
                    dim_feedforward=self.config.get("dim_feedforward", 2048),
                    dropout=self.dropout
                )
        elif core_model_type == "bert_bilstm":
            # BERT+BiLSTM混合架构
            logger.info("使用BERT+BiLSTM混合架构作为核心模型")
            self.core_model = BERTBiLSTMModel(
                input_dim=self.projection_dim,
                bert_model_name=self.config.get("bert_model_name", "bert-base-uncased"),
                lstm_hidden_dim=self.config.get("lstm_hidden_dim", 256),
                lstm_num_layers=self.config.get("lstm_num_layers", 2),
                fusion_dim=self.config.get("fusion_dim", 512),
                dropout=self.dropout,
                use_bert_pooler=self.config.get("use_bert_pooler", False),
                freeze_bert=self.config.get("freeze_bert", False),
                config=self.config,
                transformer_layers=self.config.get("transformer_layers", 6),
                struct_dim=self.projection_dim if self.use_structure else 0
            )
        elif core_model_type == "ptm_bert_bilstm":
            # PTM特异性BERT+BiLSTM架构
            logger.info("使用PTM特异性BERT+BiLSTM架构作为核心模型")
            self.core_model = PTMSpecificBERTBiLSTM(
                input_dim=self.projection_dim,
                num_ptm_types=self.config.get("num_ptm_types", 1),
                aa_embedding_dim=self.config.get("aa_embedding_dim", 64),
                position_embedding_dim=self.config.get("position_embedding_dim", 32),
                bert_model_name=self.config.get("bert_model_name", "bert-base-uncased"),
                lstm_hidden_dim=self.config.get("lstm_hidden_dim", 256),
                lstm_num_layers=self.config.get("lstm_num_layers", 2),
                fusion_dim=self.config.get("fusion_dim", 512),
                dropout=self.dropout,
                freeze_bert=self.config.get("freeze_bert", False),
                config=self.config,
                transformer_layers=self.config.get("transformer_layers", 6),
                struct_dim=self.projection_dim if self.use_structure else 0
            )
        elif core_model_type in ["acetylation_transformer", "dual_branch_fusion", "dual_branch_attention"]:
            # 乙酰化Transformer架构 / 双分支融合架构 - 专门为乙酰化预测优化
            logger.info(f"使用 {core_model_type} 架构作为核心模型 (映射到 AcetylationPredictionModel)")
            self.core_model = AcetylationPredictionModel(
                seq_feature_dim=self.seq_dim,
                struct_feature_dim=self.struct_dim,
                hidden_dim=self.config.get("hidden_dim", 512),
                context_window=self.config.get("context_window", 15),
                dropout=self.dropout,
                config=self.config
            )
            # 为乙酰化Transformer设置多任务损失函数
            self.multitask_loss = MultiTaskLoss(
                alpha=self.config.get("multitask_alpha", 0.7),
                class_weights=self.config.get("class_weights", [1.0, 3.0]),
                alpha_bounds=self.config.get("multitask_alpha_bounds")
            )
            self.use_multitask_loss = True
            logger.info("核心模型已初始化，启用多任务学习")
        else:
            raise ValueError(f"不支持的核心模型类型: {core_model_type}。支持的类型: 'transformer', 'bert_bilstm', 'ptm_bert_bilstm', 'acetylation_transformer', 'dual_branch_fusion', 'dual_branch_attention'")
            
        # 强化学习分类器 (可选)
        if use_reinforcement:
            logger.info("初始化强化学习分类器")

            # 根据核心模型类型选择合适的RL分类器
            if core_model_type in ["bert_bilstm", "ptm_bert_bilstm"]:
                logger.info("使用BERT+BiLSTM强化学习分类器")
                self.rl_classifier = BERTBiLSTMRLClassifier(
                    input_dim=self.projection_dim,
                    bert_hidden_dim=self.bert_hidden_dim,
                    lstm_hidden_dim=self.config.get("rl_lstm_hidden_dim", 128),
                    num_decision_steps=self.config.get("rl_num_steps", 3),
                    dropout=self.dropout
                )
            else:
                # 传统RL分类器
                self.rl_classifier = RLClassifier(
                    input_dim=self.projection_dim,
                    hidden_dim=self.config.get("rl_hidden_dim", 128),
                    num_steps=self.config.get("rl_num_steps", 5)
                )
            self.use_rl_classifier = True
        else:
            self.use_rl_classifier = False
            
        # 因果模型 (可选)
        if use_causal:
            logger.info("初始化因果模型")

            # 根据核心模型类型选择合适的因果模型
            if core_model_type in ["bert_bilstm", "ptm_bert_bilstm"]:
                logger.info("使用BERT+BiLSTM因果推断模型")
                self.causal_model = BERTBiLSTMCausalModel(
                    feature_dim=self.projection_dim,
                    bert_hidden_dim=self.bert_hidden_dim,
                    lstm_hidden_dim=self.config.get("causal_lstm_hidden_dim", 128),
                    causal_factors=self.config.get("causal_factors", 8),
                    hidden_dim=self.config.get("causal_hidden_dim", 256),
                    use_layer_norm=self.config.get("use_layer_norm", True)
                )
            else:
                # 传统因果模型
                self.causal_model = StructuralCausalModel(
                    feature_dim=self.projection_dim,
                    hidden_dim=self.config.get("causal_hidden_dim", 128),
                    causal_factors=self.config.get("causal_factors", 5)
                )
            self.use_causal_model = True
        else:
            self.use_causal_model = False

        # 元学习模块 (可选)
        if use_meta_learning:
            logger.info("初始化元学习模块")

            # 根据核心模型类型选择合适的元学习模型
            if core_model_type in ["bert_bilstm", "ptm_bert_bilstm"]:
                logger.info("使用BERT+BiLSTM元学习模型")

                # MAML模型
                self.maml_model = BERTBiLSTMMAML(
                    input_dim=self.projection_dim,
                    output_dim=1,
                    bert_hidden_dim=self.bert_hidden_dim,
                    lstm_hidden_dim=self.config.get("meta_lstm_hidden_dim", 128),
                    inner_lr=self.config.get("inner_lr", 0.01),
                    inner_steps=self.config.get("inner_steps", 5)
                )

                # ProtoNet模型
                self.protonet_model = BERTBiLSTMProtoNet(
                    input_dim=self.projection_dim,
                    bert_hidden_dim=self.bert_hidden_dim,
                    lstm_hidden_dim=self.config.get("meta_lstm_hidden_dim", 128),
                    embedding_dim=self.config.get("embedding_dim", 256)
                )
            else:
                # 传统元学习模型
                # 创建基础分类器用于MAML
                base_classifier = nn.Sequential(
                    nn.Linear(self.projection_dim, self.projection_dim // 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.projection_dim // 2, 1)
                )

                self.maml_model = MAML(
                    base_model=base_classifier,
                    inner_lr=self.config.get("inner_lr", 0.01),
                    inner_steps=self.config.get("inner_steps", 5)
                )

                # 创建编码器用于ProtoNet
                encoder = nn.Sequential(
                    nn.Linear(self.projection_dim, self.projection_dim // 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.projection_dim // 2, self.config.get("embedding_dim", 256))
                )

                self.protonet_model = ProtoNet(encoder=encoder)

            self.use_meta_learning_model = True
        else:
            self.use_meta_learning_model = False

        # 全局和局部特征融合层
        # 计算融合维度: Global + Local
        fusion_input_dim = self.projection_dim * 2
        
        # 中心位点聚焦机制
        self.use_center_focus = self.config.get("use_center_focus", True)
        if self.use_center_focus:
            logger.info("启用中心位点聚焦机制 (Center-Focused Fusion)")
            fusion_input_dim += self.projection_dim
            # [新增] 靶向Dropout：专门针对中心特征的高强度Dropout，防止模型死记硬背中心位点
            self.center_dropout = nn.Dropout(self.config.get("center_dropout", 0.5))

        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
            
        # 最终分类头 - 简化版本以提高稳定性
        if not self.use_rl_classifier and not self.use_causal_model:
            self.classifier = nn.Sequential(
                nn.Linear(self.projection_dim, 64),  # 大幅减少中间层大小
                nn.ReLU(),                           # 使用更稳定的ReLU
                nn.Dropout(0.1),                     # 降低dropout
                nn.Linear(64, 1)                     # 输出层
            )

            # 添加稳定的权重初始化
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)  # 小的gain值
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        logger.info("AcetylationPredictor初始化完成")
        
    def _init_center_regularization(self) -> None:
        """Initialize masking tokens and curriculum config for center features."""
        self.center_mask_prob = float(self.config.get("center_mask_prob", 0.0))
        self.center_mask_noise_std = float(self.config.get("center_mask_noise_std", 0.0))
        self.center_mask_token_seq = None
        self.center_mask_token_struct = None
        if self.center_mask_prob > 0.0:
            self.center_mask_token_seq = nn.Parameter(torch.zeros(self.seq_dim))
            nn.init.normal_(self.center_mask_token_seq, mean=0.0, std=0.02)
            if self.use_structure:
                self.center_mask_token_struct = nn.Parameter(torch.zeros(self.struct_dim))
                nn.init.normal_(self.center_mask_token_struct, mean=0.0, std=0.02)

        self.center_curriculum_cfg = self.config.get("center_curriculum", {}) or {}
        self.center_curriculum_enabled = bool(self.center_curriculum_cfg)
        default_alpha = float(
            self.center_curriculum_cfg.get(
                "initial_alpha",
                0.0 if self.center_curriculum_enabled else 1.0,
            )
        )
        self.center_curriculum_eval_alpha = float(self.center_curriculum_cfg.get("eval_alpha", 1.0))
        self.current_center_alpha = default_alpha if self.center_curriculum_enabled else 1.0

    def update_center_curriculum(self, epoch: float) -> None:
        """External hook for training loop to update curriculum epoch."""
        if not self.center_curriculum_enabled:
            return
        self.current_center_alpha = self._compute_center_alpha(float(epoch))

    def set_center_alpha(self, alpha: float) -> None:
        """Allow manual override of the current center feature scaling."""
        self.current_center_alpha = float(alpha)

    def _resolve_center_alpha(self, epoch_override=None, alpha_override=None) -> float:
        if alpha_override is not None:
            self.set_center_alpha(alpha_override)
        elif epoch_override is not None:
            self.update_center_curriculum(epoch_override)
        elif not self.training and self.center_curriculum_enabled:
            self.current_center_alpha = self.center_curriculum_eval_alpha
        return float(self.current_center_alpha)

    def _compute_center_alpha(self, epoch_value: float) -> float:
        cfg = self.center_curriculum_cfg
        start_alpha = float(cfg.get("start_alpha", cfg.get("initial_alpha", 0.0)))
        target_alpha = float(cfg.get("target_alpha", cfg.get("alpha_target", 1.0)))
        warmup = max(0, int(cfg.get("warmup_epochs", 0)))
        ramp = max(1, int(cfg.get("ramp_epochs", 1)))
        if epoch_value < warmup:
            return start_alpha
        progress = min(1.0, (epoch_value - warmup) / float(ramp))
        return start_alpha + (target_alpha - start_alpha) * progress

    @staticmethod
    def _apply_center_alpha(tensor: Optional[torch.Tensor], alpha: float) -> Optional[torch.Tensor]:
        if tensor is None or alpha >= 0.9999:
            return tensor
        return tensor * alpha

    def _apply_center_mask_if_needed(
        self,
        seq_tensor: Optional[torch.Tensor],
        local_tensor: Optional[torch.Tensor],
        struct_tensor: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        relative_offsets: Optional[torch.Tensor],
    ):
        if (
            not self.training
            or self.center_mask_prob <= 0.0
            or seq_tensor is None
            or not isinstance(seq_tensor, torch.Tensor)
            or seq_tensor.dim() < 3
        ):
            return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

        batch_size = seq_tensor.size(0)
        device = seq_tensor.device
        mask_flags = torch.rand(batch_size, device=device) < self.center_mask_prob
        if not mask_flags.any():
            return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

        seq_tensor = self._mask_center_tensor(seq_tensor, mask_flags, self.center_mask_token_seq)
        if isinstance(local_tensor, torch.Tensor) and local_tensor.dim() >= 3:
            local_tensor = self._mask_center_tensor(local_tensor, mask_flags, self.center_mask_token_seq)
        if (
            isinstance(struct_tensor, torch.Tensor)
            and struct_tensor.dim() >= 3
            and self.center_mask_token_struct is not None
        ):
            struct_tensor = self._mask_center_tensor(struct_tensor, mask_flags, self.center_mask_token_struct)

        mask_indices = mask_flags.nonzero(as_tuple=False).view(-1)
        if mask_indices.numel() > 0:
            if isinstance(position_ids, torch.Tensor) and position_ids.dim() >= 2:
                position_ids = position_ids.clone()
                center_idx = min(position_ids.size(1) // 2, position_ids.size(1) - 1)
                if self.center_mask_noise_std > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.center_mask_noise_std,
                        size=(mask_indices.numel(),),
                        device=position_ids.device,
                    ).round().long()
                else:
                    noise = torch.zeros(mask_indices.numel(), device=position_ids.device, dtype=torch.long)
                center_vals = position_ids[mask_indices, center_idx] + noise
                position_ids[mask_indices, center_idx] = center_vals.clamp_(0, self.max_position_embeddings - 1)

            if isinstance(relative_offsets, torch.Tensor) and relative_offsets.dim() >= 2:
                relative_offsets = relative_offsets.clone()
                center_idx = min(relative_offsets.size(1) // 2, relative_offsets.size(1) - 1)
                relative_offsets[mask_indices, center_idx] = 0

        return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

    @staticmethod
    def _mask_center_tensor(
        tensor: torch.Tensor,
        mask_flags: torch.Tensor,
        mask_token: Optional[torch.nn.Parameter],
    ) -> torch.Tensor:
        if mask_token is None or tensor.dim() < 3:
            return tensor
        center_idx = tensor.size(1) // 2
        masked = tensor.clone()
        token = mask_token.to(dtype=masked.dtype, device=masked.device)
        masked[mask_flags, center_idx, :] = token
        return masked

    def forward(self, seq_data=None, local_seq_data=None, struct_data=None,
                acetylation_positions=None, site_positions=None, **kwargs):
        """
        前向传播

        Args:
            seq_data: 序列数据或批次字典
                - 如果是torch.Tensor: 直接使用作为序列特征
                - 如果是dict: 批次字典，包含'sequence', 'structure', 'label'等键
            local_seq_data: 局部序列数据 (可选)
            struct_data: 结构数据 (可选)
            acetylation_positions: 乙酰化位点位置 [batch_size] (可选)
            site_positions: 修饰位点位置 [batch_size] (可选，用于acetylation_transformer)
            **kwargs: 兼容DataLoader额外传入的特征 (例如window_features、local_features等)

        Returns:
            torch.Tensor or tuple: 预测结果
        """
        curriculum_epoch = kwargs.pop('center_curriculum_epoch', None)
        center_alpha_override = kwargs.pop('center_alpha', None)
        if center_alpha_override is None:
            center_alpha_override = kwargs.pop('center_alpha_override', None)
        active_center_alpha = self._resolve_center_alpha(curriculum_epoch, center_alpha_override)

        attention_mask = kwargs.pop('attention_mask', None)
        ptm_type_ids = None
        position_ids = None
        residue_types_tensor = None
        relative_position_offsets = None
        site_indicator = None
        extra_context = None

        # 兼容外部调用传入的关键字特征
        if seq_data is None and 'window_features' in kwargs:
            seq_data = kwargs.pop('window_features')
        if local_seq_data is None and 'local_features' in kwargs:
            local_seq_data = kwargs.pop('local_features')
        if struct_data is None and 'structure' in kwargs:
            struct_data = kwargs.pop('structure')
        if struct_data is None and 'structure_features' in kwargs:
            struct_data = kwargs.pop('structure_features')
        if acetylation_positions is None and 'positions' in kwargs:
            acetylation_positions = kwargs.pop('positions')
        if site_positions is None and 'site_positions' in kwargs:
            site_positions = kwargs.pop('site_positions')
        if 'position_ids' in kwargs and position_ids is None:
            position_ids = kwargs.pop('position_ids')
        if 'ptm_type_ids' in kwargs and ptm_type_ids is None:
            ptm_type_ids = kwargs.pop('ptm_type_ids')
        if 'site_indicator' in kwargs and site_indicator is None:
            site_indicator = kwargs.pop('site_indicator')
        if 'relative_position_offsets' in kwargs and relative_position_offsets is None:
            relative_position_offsets = kwargs.pop('relative_position_offsets')
        if 'global_features' in kwargs:
            extra_context = kwargs.pop('global_features')
        
        micro_env_data = None
        if 'micro_environment' in kwargs:
            micro_env_data = kwargs.pop('micro_environment')

        # 处理批次字典输入
        if isinstance(seq_data, dict):
            batch = seq_data
            # 从批次字典中提取数据
            if 'sequence' in batch:
                if isinstance(batch['sequence'], dict):
                    # 新格式：sequence是字典，包含多种特征
                    if 'window_features' in batch['sequence']:
                        seq_data = batch['sequence']['window_features']
                    elif 'sequence_features' in batch['sequence']:
                        seq_data = batch['sequence']['sequence_features']
                    elif 'local_features' in batch['sequence']:
                        seq_data = batch['sequence']['local_features']
                    else:
                        # 取第一个可用的特征
                        seq_data = list(batch['sequence'].values())[0]

                    # 提取局部序列特征
                    if local_seq_data is None and 'local_features' in batch['sequence']:
                        local_seq_data = batch['sequence']['local_features']
                else:
                    # 旧格式：sequence直接是张量
                    seq_data = batch['sequence']
            elif 'sequence_features' in batch:
                seq_data = batch['sequence_features']
                if local_seq_data is None and 'local_features' in batch:
                    local_seq_data = batch['local_features']
            else:
                raise ValueError("批次字典中未找到序列数据")

            # 提取结构数据
            if struct_data is None:
                if 'structure' in batch:
                    struct_data = batch['structure']
                elif 'structure_features' in batch:
                    struct_data = batch['structure_features']

            # 提取乙酰化位置
            if acetylation_positions is None and 'positions' in batch:
                acetylation_positions = batch['positions']

            # 提取残基类型
            if 'residue_types' in batch:
                residue_types_tensor = batch['residue_types']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'residue_types' in batch['sequence']:
                residue_types_tensor = batch['sequence']['residue_types']

            # 提取位置信息
            if 'position_ids' in batch:
                position_ids = batch['position_ids']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'position_ids' in batch['sequence']:
                position_ids = batch['sequence']['position_ids']

            if 'relative_position_offsets' in batch:
                relative_position_offsets = batch['relative_position_offsets']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'relative_position_offsets' in batch['sequence']:
                relative_position_offsets = batch['sequence']['relative_position_offsets']

            # 提取PTM类型信息
            if 'ptm_type_ids' in batch:
                ptm_type_ids = batch['ptm_type_ids']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'ptm_type_ids' in batch['sequence']:
                ptm_type_ids = batch['sequence']['ptm_type_ids']
            elif 'ptm_type' in batch:
                ptm_type_ids = batch['ptm_type']

            # 目标位点指示器
            if 'site_indicator' in batch:
                site_indicator = batch['site_indicator']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'site_indicator' in batch['sequence']:
                site_indicator = batch['sequence']['site_indicator']

            if extra_context is None and 'global_features' in batch:
                extra_context = batch['global_features']

            if attention_mask is None:
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask']
                elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'attention_mask' in batch['sequence']:
                    attention_mask = batch['sequence']['attention_mask']


        # 归一化PTM类型ID
        if ptm_type_ids is not None and not isinstance(ptm_type_ids, torch.Tensor):
            mapping = self.config.get('ptm_type_to_id') if isinstance(self.config, dict) else None
            if isinstance(ptm_type_ids, (list, tuple)):
                normalized_ids = []
                for value in ptm_type_ids:
                    if isinstance(value, torch.Tensor):
                        normalized_ids.append(int(value.item()))
                    elif isinstance(value, str):
                        if mapping and value in mapping:
                            normalized_ids.append(mapping[value])
                        else:
                            normalized_ids.append(0)
                    else:
                        normalized_ids.append(int(value))
                ptm_type_ids = normalized_ids
            elif isinstance(ptm_type_ids, str):
                if mapping and ptm_type_ids in mapping:
                    ptm_type_ids = [mapping[ptm_type_ids]]
                else:
                    ptm_type_ids = [0]
            else:
                ptm_type_ids = [int(ptm_type_ids)]
            ptm_type_ids = torch.tensor(ptm_type_ids, dtype=torch.long)
        # 确保数据在正确的设备上
        if hasattr(self, 'device'):
            if isinstance(seq_data, torch.Tensor):
                seq_data = seq_data.to(self.device)
            if local_seq_data is not None and isinstance(local_seq_data, torch.Tensor):
                local_seq_data = local_seq_data.to(self.device)
            if struct_data is not None and isinstance(struct_data, torch.Tensor):
                struct_data = struct_data.to(self.device)
            if ptm_type_ids is not None and isinstance(ptm_type_ids, torch.Tensor):
                ptm_type_ids = ptm_type_ids.to(self.device)
            if position_ids is not None and isinstance(position_ids, torch.Tensor):
                position_ids = position_ids.to(self.device)
            if residue_types_tensor is not None and isinstance(residue_types_tensor, torch.Tensor):
                residue_types_tensor = residue_types_tensor.to(self.device)
            if relative_position_offsets is not None and isinstance(relative_position_offsets, torch.Tensor):
                relative_position_offsets = relative_position_offsets.to(self.device)
            if site_indicator is not None and isinstance(site_indicator, torch.Tensor):
                site_indicator = site_indicator.to(self.device)
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
                
        seq_data, local_seq_data, struct_data, position_ids, relative_position_offsets = self._apply_center_mask_if_needed(
            seq_data,
            local_seq_data,
            struct_data,
            position_ids,
            relative_position_offsets,
        )

        # 检查必需输入
        if seq_data is None:
            raise ValueError("forward 调用缺少序列特征：请提供 seq_data 或 window_features")

        # 打印初始推理的设备信息
        if not hasattr(self, '_first_forward_done'):
            logger.info(f"\n===== AcetylationPredictor设备检查 =====")
            if isinstance(seq_data, torch.Tensor):
                logger.info(f"seq_data设备: {seq_data.device}")
            if struct_data is not None:
                if isinstance(struct_data, torch.Tensor):
                    logger.info(f"struct_data设备: {struct_data.device}")
                else:
                    logger.info(f"struct_data类型: {type(struct_data)}")
            logger.info(f"模型参数设备: {next(self.parameters()).device}")
            self._first_forward_done = True
            logger.info("===== 设备检查结束 =====\n")
        
        # 1. 提取序列特征
        if self.use_precomputed_features:
            # 使用预计算特征，seq_data应该已经是特征向量
            if isinstance(seq_data, torch.Tensor):
                per_residue_seq = seq_data  # 直接使用预计算的特征
                
                # [Fix] 提前处理残基类型ID，确保在One-hot编码前ID有效
                if residue_types_tensor is not None:
                     if not isinstance(residue_types_tensor, torch.Tensor):
                        residue_types_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
                     else:
                        residue_types_tensor = residue_types_tensor.to(self.device)
                        if residue_types_tensor.dtype != torch.long:
                             residue_types_tensor = residue_types_tensor.long()
                     
                     if residue_types_tensor.dim() == 1:
                         batch_size, seq_len = per_residue_seq.shape[:2]
                         residue_types_tensor = residue_types_tensor.unsqueeze(1).expand(-1, seq_len)

                     # 动态调整残基类型数量，避免索引越界
                     max_residue_id = self.num_residue_types - 1
                     if residue_types_tensor.max() > max_residue_id:
                        if not hasattr(self, '_warned_residue_clamp'):
                             logger.warning(f"输入残基ID最大值 ({residue_types_tensor.max()}) 超过配置的 num_residue_types-1 ({max_residue_id})，已自动裁剪。建议在配置中增加 num_residue_types。")
                             self._warned_residue_clamp = True
                        residue_types_tensor = torch.clamp(residue_types_tensor, min=0, max=max_residue_id)

                # [Modified] Restore One-hot concatenation if needed (Hybrid Encoding)
                # If the model expects higher dimensionality (e.g. 1315) but we have 1280 (ESM),
                # and residue types are available, we should concatenate One-hot encoding.
                if hasattr(self, 'seq_dim') and per_residue_seq.shape[-1] < self.seq_dim:
                    diff = self.seq_dim - per_residue_seq.shape[-1]
                    # Check if the difference matches num_residue_types (approx 35)
                    # We only apply if residue_types_tensor is available
                    if residue_types_tensor is not None:
                        one_hot = F.one_hot(residue_types_tensor, num_classes=self.num_residue_types).float()
                        # Ensure dimension match
                        if one_hot.shape[-1] != diff:
                            if one_hot.shape[-1] > diff:
                                one_hot = one_hot[:, :, :diff]
                            else:
                                pad_len = diff - one_hot.shape[-1]
                                one_hot = F.pad(one_hot, (0, pad_len))
                        
                        per_residue_seq = torch.cat([per_residue_seq, one_hot], dim=-1)
                        # logger.debug(f"Applied Hybrid Encoding: Concatenated One-hot ({diff} dims) to ESM features.")
                
                seq_repr = per_residue_seq.mean(dim=1)  # 全局表示
            else:
                raise ValueError("使用预计算特征时，seq_data应该是torch.Tensor")
        else:
            # 使用编码器提取特征
            seq_repr, per_residue_seq = self.seq_encoder(seq_data)

        # 对于acetylation_transformer模型，不需要投影
        if self.use_multitask_loss:
            seq_features = per_residue_seq
        else:
            # 严格检查维度，动态重新初始化以适应特征维度的变化
            if hasattr(self, 'seq_projection'):
                input_dim = per_residue_seq.shape[-1]
                expected_dim = self.seq_projection.projection[0].in_features
                if expected_dim != input_dim:
                     logger.warning(f"检测到序列特征维度变化 ({expected_dim} -> {input_dim})，正在重新初始化 seq_projection")
                     device = per_residue_seq.device
                     self.seq_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)
                
                seq_features = self.seq_projection(per_residue_seq)
            else:
                # 理论上不应到达此处，除非没有初始化seq_projection
                pass

        # 2. 提取局部序列特征 (如果可用)
        # 注意：acetylation_transformer模型不需要局部特征处理
        if local_seq_data is not None and not self.use_multitask_loss:
            if self.use_precomputed_features:
                # 使用预计算的局部特征
                if isinstance(local_seq_data, torch.Tensor):
                    local_per_residue_seq = local_seq_data
                else:
                    raise ValueError("使用预计算特征时，local_seq_data应该是torch.Tensor")
            else:
                # 使用编码器提取局部特征
                local_seq_repr, local_per_residue_seq = self.seq_encoder(local_seq_data)

            # 动态检查输入维度并重新初始化投影层（如果需要）
            if hasattr(self, 'local_seq_projection'):
                input_dim = local_per_residue_seq.shape[-1]
                if self.local_seq_projection.projection[0].in_features != input_dim:
                    logger.warning(f"检测到局部序列特征维度变化 ({self.local_seq_projection.projection[0].in_features} -> {input_dim})，正在重新初始化 local_seq_projection")
                    device = local_per_residue_seq.device
                    self.local_seq_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)

            local_features = self.local_seq_projection(local_per_residue_seq)
        else:
            local_features = None

        # 3. 提取结构特征 (如果可用)
        struct_features = None
        per_residue_struct = None  # 初始化per_residue_struct
        if struct_data is not None and self.use_structure:
            if self.use_precomputed_features:
                # 使用预计算的结构特征
                if isinstance(struct_data, torch.Tensor):
                    per_residue_struct = struct_data
                else:
                    raise ValueError("使用预计算特征时，struct_data应该是torch.Tensor")
            else:
                # 使用编码器提取结构特征
                # [Optimization] Pass distance matrix to enable SpatialGNN
                dist_mat = kwargs.get('distance_matrix', None)
                if hasattr(self, 'device') and dist_mat is not None and isinstance(dist_mat, torch.Tensor):
                    dist_mat = dist_mat.to(self.device)
                
                # Check if encoder accepts distance_matrices (it should be StructureContextEncoder)
                try:
                    struct_repr, per_residue_struct = self.struct_encoder(struct_data, distance_matrices=dist_mat)
                except TypeError:
                    # Fallback for encoders that don't support distance_matrices
                    struct_repr, per_residue_struct = self.struct_encoder(struct_data)

            # 对于acetylation_transformer模型，不需要投影
            if self.use_multitask_loss:
                struct_features = per_residue_struct
            else:
                # 动态检查输入维度并重新初始化投影层（如果需要）
                if hasattr(self, 'struct_projection'):
                    input_dim = per_residue_struct.shape[-1]
                    if self.struct_projection.projection[0].in_features != input_dim:
                        logger.warning(f"检测到结构特征维度变化 ({self.struct_projection.projection[0].in_features} -> {input_dim})，正在重新初始化 struct_projection")
                        device = per_residue_struct.device
                        self.struct_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)
                
                struct_features = self.struct_projection(per_residue_struct)

        # 4. 提取乙酰化信息特征
        if isinstance(seq_data, torch.Tensor) and seq_data.dim() > 1:
            if self.use_precomputed_features:
                # 对于预计算特征，暂时跳过乙酰化编码
                acetyl_features = None
            else:
                acetyl_features = self.acetylation_encoder(seq_data, acetylation_positions)
        else:
            acetyl_features = None
            
        # 5. 多模态特征融合
        # 注意：acetylation_transformer模型不需要融合
        if self.use_multitask_loss:
            # 对于acetylation_transformer，直接使用原始特征
            combined_features = seq_features
        else:
            available_features = []

            # 添加序列特征（始终可用）
            available_features.append(seq_features)

            # 添加结构特征（如果可用）
            if struct_features is not None:
                available_features.append(struct_features)

            # 融合特征
            # 优化：如果核心模型是BERTBiLSTM或其变体，它们内部会处理结构特征融合
            # 因此这里只传递序列特征，避免双重融合（Double Fusion）
            if isinstance(self.core_model, (BERTBiLSTMModel, PTMSpecificBERTBiLSTM)):
                # 传递微环境特征给核心模型
                if micro_env_data is not None:
                    # 使用 kwargs 传递，避免修改 core_model 签名
                    # 但是 core_model.__call__ 可能不接受 kwargs，需要检查
                    # 假设 core_model 已经更新为接受 micro_env_features
                    combined_features = seq_features
                    # 我们不能在这里直接调用 core_model，因为这里只是准备 features
                    # 但是下面的代码会调用 core_model
                else:
                    combined_features = seq_features
            else:
                # 对于其他模型，执行早期融合
                if self.fusion is not None:
                    combined_features = self.fusion(*available_features)
                else:
                    combined_features = available_features[0]

        batch_size, seq_len, _ = combined_features.shape

        # 5.1 加入显式位置、残基和PTM类型信息
        # 优化：如果核心模型是PTMSpecificBERTBiLSTM，它内部会处理这些嵌入（通过拼接）
        # 因此这里跳过叠加嵌入，避免双重嵌入（Double Embedding）
        is_ptm_specific_model = isinstance(self.core_model, PTMSpecificBERTBiLSTM)
        
        # 注意：对于acetylation_transformer，不需要添加位置嵌入，因为AcetylationPredictionModel已经处理了
        if position_ids is not None and not self.use_multitask_loss and not is_ptm_specific_model:
            if not isinstance(position_ids, torch.Tensor):
                position_ids_tensor = torch.tensor(position_ids, dtype=torch.long, device=self.device)
            else:
                position_ids_tensor = position_ids.to(self.device)
                if position_ids_tensor.dtype != torch.long:
                    position_ids_tensor = position_ids_tensor.long()
            if position_ids_tensor.dim() == 1:
                position_ids_tensor = position_ids_tensor.unsqueeze(1).expand(-1, seq_len)
            min_val = position_ids_tensor.min()
            if min_val < 0:
                position_ids_tensor = position_ids_tensor - min_val
            max_allowed = self.position_embedding_layer.num_embeddings - 1
            max_val = position_ids_tensor.max()
            if max_val > max_allowed:
                logger.warning(
                    "position_ids 超出嵌入范围，已裁剪至最大值。请考虑增加 max_position_embeddings 配置。"
                )
                position_ids_tensor = torch.clamp(position_ids_tensor, max=max_allowed)
            position_embedding = self.position_embedding_layer(position_ids_tensor)
            combined_features = combined_features + position_embedding
        else:
            position_ids_tensor = None

        # 注意：对于acetylation_transformer，不需要添加残基类型嵌入
        if residue_types_tensor is not None and not self.use_multitask_loss and not is_ptm_specific_model:
            if not isinstance(residue_types_tensor, torch.Tensor):
                residue_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
            else:
                residue_tensor = residue_types_tensor.to(self.device)
                if residue_tensor.dtype != torch.long:
                    residue_tensor = residue_tensor.long()
            if residue_tensor.dim() == 1:
                residue_tensor = residue_tensor.unsqueeze(1).expand(-1, seq_len)
            
            # 动态调整残基类型数量，避免索引越界
            max_residue_id = self.num_residue_types - 1
            if residue_tensor.max() > max_residue_id:
                if not hasattr(self, '_warned_residue_clamp'):
                    logger.warning(f"输入残基ID最大值 ({residue_tensor.max()}) 超过配置的 num_residue_types-1 ({max_residue_id})，已自动裁剪。建议在配置中增加 num_residue_types。")
                    self._warned_residue_clamp = True
                residue_tensor = torch.clamp(residue_tensor, min=0, max=max_residue_id)
                
            residue_embedding = self.residue_type_embedding(residue_tensor)
            combined_features = combined_features + residue_embedding
        else:
            residue_tensor = None

        # 修复标签泄露问题：移除 ptm_type_embedding 的使用
        # ptm_type 现在仅用于任务标识，不应作为模型输入特征
        # 这样可以防止模型通过 ptm_type_id 直接学习正负样本的区分
        ptm_tensor = None
        if ptm_type_ids is not None:
            if not isinstance(ptm_type_ids, torch.Tensor):
                ptm_tensor = torch.tensor(ptm_type_ids, dtype=torch.long, device=self.device)
            else:
                ptm_tensor = ptm_type_ids.to(self.device)
                if ptm_tensor.dtype != torch.long:
                    ptm_tensor = ptm_tensor.long()
            # 注意：不再使用 ptm_type_embedding，以避免标签泄露
            # ptm_embedding = self.ptm_type_embedding(ptm_tensor)
            # combined_features = combined_features + ptm_embedding

        site_indicator_tensor = None
        # 注意：对于acetylation_transformer，不需要添加相对位置和位点指示投影
        if relative_position_offsets is not None and not self.use_multitask_loss:
            if not isinstance(relative_position_offsets, torch.Tensor):
                rel_tensor = torch.tensor(relative_position_offsets, dtype=torch.long, device=self.device)
            else:
                rel_tensor = relative_position_offsets.to(self.device)
                if rel_tensor.dtype != torch.long:
                    rel_tensor = rel_tensor.long()
            if rel_tensor.dim() == 1:
                rel_tensor = rel_tensor.unsqueeze(1).expand(-1, seq_len)
            rel_float = rel_tensor.to(combined_features.dtype).unsqueeze(-1)
            rel_projection = self.relative_position_projection(rel_float)
            combined_features = combined_features + rel_projection
            site_indicator_tensor = (rel_tensor == 0).float()
        if site_indicator is not None and not self.use_multitask_loss:
            if not isinstance(site_indicator, torch.Tensor):
                site_indicator_tensor = torch.tensor(site_indicator, dtype=torch.float32, device=self.device)
            else:
                site_indicator_tensor = site_indicator.to(self.device)
                if site_indicator_tensor.dtype != torch.float32:
                    site_indicator_tensor = site_indicator_tensor.float()
        if site_indicator_tensor is not None and not self.use_multitask_loss:
            if site_indicator_tensor.dim() == 1:
                site_indicator_tensor = site_indicator_tensor.unsqueeze(1).expand(-1, seq_len)
            site_emb_input = site_indicator_tensor.unsqueeze(-1)
            site_embedding = self.site_indicator_projection(site_emb_input)
            combined_features = combined_features + site_embedding

        # 缓存处理后的嵌入，用于解释性输出
        if residue_tensor is not None:
            residue_types_tensor = residue_tensor
        if position_ids_tensor is not None:
            position_ids = position_ids_tensor
        if ptm_tensor is not None:
            ptm_type_ids = ptm_tensor
        if relative_position_offsets is not None and 'rel_tensor' in locals():
            relative_position_offsets = rel_tensor
        if site_indicator_tensor is not None:
            site_indicator = site_indicator_tensor
        
        # 6. 核心模型处理
        core_kwargs = {}

        # 传递结构特征给支持它的核心模型 (如BERTBiLSTMModel)
        if struct_features is not None and self.use_structure:
            core_kwargs['struct_features'] = struct_features
            
        # 传递微环境特征给支持它的核心模型
        if micro_env_data is not None:
            core_kwargs['micro_env_features'] = micro_env_data

        # 检查是否使用acetylation_transformer模型
        if self.use_multitask_loss:
            # acetylation_transformer模型需要原始特征和site_positions
            logger.debug(f"[acetylation_transformer] 使用多任务学习模式")
            logger.debug(f"[acetylation_transformer] per_residue_seq shape: {per_residue_seq.shape}")
            logger.debug(f"[acetylation_transformer] per_residue_struct: {per_residue_struct.shape if per_residue_struct is not None else 'None'}")

            # 需要从site_positions参数中提取位置信息
            if site_positions is None:
                # 如果没有提供site_positions，使用relative_position_offsets推导
                if relative_position_offsets is not None:
                    if isinstance(relative_position_offsets, torch.Tensor):
                        # 找到相对位置为0的位置
                        site_pos = (relative_position_offsets == 0).nonzero(as_tuple=True)[1]
                    else:
                        site_pos = torch.tensor(relative_position_offsets, dtype=torch.long, device=self.device)
                        site_pos = (site_pos == 0).nonzero(as_tuple=True)[1]
                else:
                    # 默认使用序列中点
                    batch_size = per_residue_seq.shape[0]
                    site_pos = torch.full((batch_size,), per_residue_seq.shape[1] // 2, dtype=torch.long, device=self.device)
            else:
                if not isinstance(site_positions, torch.Tensor):
                    site_pos = torch.tensor(site_positions, dtype=torch.long, device=self.device)
                else:
                    site_pos = site_positions.to(self.device).long()

            # 确保site_pos是整数类型
            if site_pos.dtype != torch.long:
                site_pos = site_pos.long()

            # 确保per_residue_struct已定义
            if per_residue_struct is None:
                # 如果没有结构特征，使用零张量
                batch_size, seq_len = per_residue_seq.shape[0], per_residue_seq.shape[1]
                per_residue_struct = torch.zeros(batch_size, seq_len, self.struct_dim, device=self.device, dtype=per_residue_seq.dtype)
                logger.debug(f"[acetylation_transformer] 结构特征为None，使用零张量，shape: {per_residue_struct.shape}")

            logger.debug(f"[acetylation_transformer] site_pos shape: {site_pos.shape}")
            logger.debug(f"[acetylation_transformer] 调用acetylation_transformer模型")

            # 转换 attention_mask 为 padding mask (True 表示 padding)
            padding_mask = None
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
                # 确保 padding_mask 与 sequence features 维度一致
                if padding_mask.shape[1] != per_residue_seq.shape[1]:
                    logger.warning(f"attention_mask length ({padding_mask.shape[1]}) != sequence length ({per_residue_seq.shape[1]}). Truncating/Padding mask.")
                    if padding_mask.shape[1] > per_residue_seq.shape[1]:
                         padding_mask = padding_mask[:, :per_residue_seq.shape[1]]
                    else:
                         pad_len = per_residue_seq.shape[1] - padding_mask.shape[1]
                         padding_mask = F.pad(padding_mask, (0, pad_len), value=True) # Pad with True (padding)

            # 调用acetylation_transformer模型
            core_output = self.core_model(per_residue_seq, per_residue_struct, site_pos, mask=padding_mask)
            logger.debug(f"[acetylation_transformer] 模型输出类型: {type(core_output)}")
            if isinstance(core_output, tuple):
                logger.debug(f"[acetylation_transformer] 模型输出为元组，长度: {len(core_output)}")
                for i, out in enumerate(core_output):
                    if isinstance(out, torch.Tensor):
                        logger.debug(f"[acetylation_transformer] 输出{i} shape: {out.shape}")
            elif isinstance(core_output, torch.Tensor):
                logger.debug(f"[acetylation_transformer] 模型输出shape: {core_output.shape}")
        elif isinstance(self.core_model, PTMSpecificBERTBiLSTM):
            if ptm_type_ids is not None:
                if not isinstance(ptm_type_ids, torch.Tensor):
                    ptm_type_ids = torch.tensor(ptm_type_ids, dtype=torch.long, device=self.device)
                elif ptm_type_ids.dtype != torch.long:
                    ptm_type_ids = ptm_type_ids.long()
                core_kwargs['ptm_type_ids'] = ptm_type_ids

            if position_ids is not None:
                if not isinstance(position_ids, torch.Tensor):
                    position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
                elif position_ids.dtype != torch.long:
                    position_ids = position_ids.long()
                core_kwargs['position_ids'] = position_ids

            if residue_types_tensor is not None:
                if not isinstance(residue_types_tensor, torch.Tensor):
                    residue_types_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
                elif residue_types_tensor.dtype != torch.long:
                    residue_types_tensor = residue_types_tensor.long()
                core_kwargs['residue_types'] = residue_types_tensor

            core_output = self.core_model(combined_features, **core_kwargs)
        else:
            core_output = self.core_model(combined_features, **core_kwargs)
        
        # 7. 处理acetylation_transformer模型的特殊输出
        if self.use_multitask_loss:
            # acetylation_transformer模型返回(acetylation_prob, strength_pred)
            # 直接返回，不进行进一步处理
            return core_output

        # 8. 存储可解释性得分
        importance_scores = {}

        # 9. 处理局部序列特征
        if local_features is not None:
            # 策略升级：如果启用了中心聚焦，局部特征优先取中心 (模拟 DualBranchFusion 的行为)
            # 这样可以保留最原始的中心残基特征，防止深层网络过平滑
            if self.use_center_focus:
                local_center_idx = local_features.size(1) // 2
                local_context = local_features[:, local_center_idx, :]
                # [新增] 对局部中心特征应用强Dropout
                if hasattr(self, 'center_dropout'):
                    local_context = self.center_dropout(local_context)
                local_context = self._apply_center_alpha(local_context, active_center_alpha)
            elif self.use_interpretability:
                local_context, local_importance = self.local_interpretability(local_features, return_importance=True)
                importance_scores['local_importance'] = local_importance
            else:
                local_context = torch.mean(local_features, dim=1)  # 平均池化
        else:
            local_context = None

        # 10. 应用解释性模块到全局特征
        if self.use_interpretability:
            global_context, global_importance = self.global_interpretability(core_output, return_importance=True)
            importance_scores['global_importance'] = global_importance
            if residue_types_tensor is not None:
                importance_scores['residue_types'] = residue_types_tensor
            if position_ids is not None:
                importance_scores['position_ids'] = position_ids
            if relative_position_offsets is not None:
                importance_scores['relative_position_offsets'] = relative_position_offsets
            if ptm_type_ids is not None:
                importance_scores['ptm_type_ids'] = ptm_type_ids
            if site_indicator is not None:
                importance_scores['site_indicator'] = site_indicator
        else:
            global_context = core_output.mean(dim=1)  # 平均池化
            
        # 10. 融合全局和局部特征
        fusion_inputs = [global_context]
        
        # 准备局部特征 (如果缺失则用零填充，以匹配融合层维度)
        if local_context is None:
            local_context = torch.zeros_like(global_context)
        fusion_inputs.append(local_context)
            
        # 提取中心位点特征 (Center-Focused Fusion)
        if self.use_center_focus:
            # core_output shape: [batch_size, seq_len, dim]
            seq_len = core_output.size(1)
            center_idx = seq_len // 2
            center_feature = core_output[:, center_idx, :]
            # [新增] 对全局中心特征应用强Dropout
            if hasattr(self, 'center_dropout'):
                center_feature = self.center_dropout(center_feature)
            center_feature = self._apply_center_alpha(center_feature, active_center_alpha)
            fusion_inputs.append(center_feature)
            
        # 执行融合
        # 注意：context_fusion 期望的输入维度在 __init__ 中已根据配置确定
        if len(fusion_inputs) > 1:
            fused_context = self.context_fusion(torch.cat(fusion_inputs, dim=1))
        else:
            fused_context = global_context
        
        # 11. 合并乙酰化特征（如果可用）
        if acetyl_features is not None:
            fused_context = fused_context + 0.2 * acetyl_features  # 使用加权和融合乙酰化特征

        # 11.a 合并微环境特征（如果可用）
        # 如果启用了显式特征或微环境特征存在，则将其融合
        if micro_env_data is not None:
            # 投影微环境特征
            micro_env_proj = self.micro_env_projection(micro_env_data)
            
            # 如果是3D张量 [batch, seq_len, 6]，取中心位点或池化
            if micro_env_proj.dim() == 3:
                # 假设我们关注中心位点（因为微环境通常是针对特定位点的）
                # 如果 micro_env_data 已经是针对位点的 (batch, 6)，则 dim=2
                # 这里假设输入可能是 (batch, 6) 或 (batch, 1, 6)
                pass 
            
            # 确保维度匹配 [batch, projection_dim]
            if micro_env_proj.dim() == 3:
                 # [batch, seq_len, dim] -> [batch, dim] (取平均或中心)
                 # 这里的微环境特征通常是针对Target Residue的，所以应该已经是 [batch, 6] 或 [batch, 1, 6]
                 micro_env_proj = micro_env_proj.mean(dim=1)
            
            # 融合
            fused_context = fused_context + micro_env_proj
            
        # 11.1 合并外部上下文特征（如果提供）
        if extra_context is not None:
            if isinstance(extra_context, torch.Tensor):
                extra_context = extra_context.to(fused_context.device)
                
                # 动态检查输入维度并重新初始化投影层（如果需要）
                if hasattr(self, 'global_context_projection'):
                    input_dim = extra_context.shape[-1]
                    if self.global_context_projection.projection[0].in_features != input_dim:
                        logger.warning(f"检测到全局上下文特征维度变化 ({self.global_context_projection.projection[0].in_features} -> {input_dim})，正在重新初始化 global_context_projection")
                        device = extra_context.device
                        self.global_context_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)

                if extra_context.dim() == 2:
                    projected_global = self.global_context_projection(extra_context)
                    fused_context = fused_context + projected_global
                elif extra_context.dim() == 3:
                    projected_global = self.global_context_projection(extra_context)
                    fused_context = fused_context + projected_global.mean(dim=1)
                else:
                    logger.warning("global_features 维度不受支持，已忽略: %s", extra_context.shape)
            else:
                logger.warning("global_features 类型不受支持，已忽略: %s", type(extra_context))
            
        # 12. 使用强化学习分类器（如果启用）
        if self.use_rl_classifier:
            rl_output = self.rl_classifier(fused_context)
            if self.use_interpretability:
                return rl_output, importance_scores
            else:
                return rl_output
            
        # 13. 使用因果模型（如果启用）
        if self.use_causal_model:
            causal_output = self.causal_model(fused_context)
            if self.use_interpretability:
                return causal_output, importance_scores
            else:
                return causal_output
            
        # 14. 标准分类
        # 对于acetylation_transformer模型，已经在第7步处理过了
        if self.use_multitask_loss:
            # acetylation_transformer模型已经返回了预测结果
            # 这里不需要再进行分类
            if self.use_interpretability:
                return core_output, importance_scores
            else:
                return core_output
        else:
            logits = self.classifier(fused_context)

            # [Fix] Check for NaNs in logits
            if torch.isnan(logits).any():
                logger.warning(f"NaN detected in logits! Batch size: {logits.size(0)}. Replacing with zeros.")
                logits = torch.nan_to_num(logits, nan=0.0)

            if self.use_interpretability:
                return logits, importance_scores
            else:
                return logits
            
    def extract_features(self, seq_data, struct_data=None, return_separate=False):
        """
        提取特征而不进行分类
        
        Args:
            seq_data: 序列数据
            struct_data: 结构数据 (可选)
            return_separate: 是否返回分离的特征 (默认: False)
            
        Returns:
            torch.Tensor or dict: 提取的特征
        """
        # 确保数据在正确的设备上
        if hasattr(self, 'device'):
            seq_data = seq_data.to(self.device)
            if struct_data is not None:
                struct_data = struct_data.to(self.device)
                
        # 提取序列特征
        seq_repr, per_residue_seq = self.seq_encoder(seq_data)
        seq_features = self.seq_projection(per_residue_seq)
        
        # 提取结构特征 (如果可用)
        if struct_data is not None and self.use_structure:
            struct_repr, per_residue_struct = self.struct_encoder(struct_data)
            struct_features = self.struct_projection(per_residue_struct)
        else:
            struct_features = None
            
        if return_separate:
            features = {
                'seq_repr': seq_repr,
                'seq_features': seq_features
            }
            if struct_features is not None:
                features['struct_repr'] = struct_repr
                features['struct_features'] = struct_features
            return features
        
        # 融合特征
        if struct_features is not None:
            # 使用多模态融合
            fused_features = self.fusion(seq_features, struct_features)
        else:
            fused_features = seq_features
            
        # 使用核心模型处理
        return self.core_model(fused_features).mean(dim=1)  # 全局池化
            
    def visualize_importance(
        self,
        local_sequence,
        importance_scores,
        save_path=None,
        title: Optional[str] = None,
        residue_offsets: Optional[List[int]] = None,
        cmap: str = "YlOrRd",
        global_sequence: Optional[str] = None,
        global_importance: Optional[torch.Tensor] = None,
        global_offsets: Optional[List[int]] = None,
        global_top_k: int = 20,
    ):
        """
        可视化氨基酸重要性
        
        Args:
            local_sequence (str): 局部氨基酸序列
            importance_scores (torch.Tensor): 重要性分数 [seq_len]
            save_path (str, optional): 保存路径
            title (str, optional): 图标题
            residue_offsets (List[int], optional): 局部序列相对中心偏移
            cmap (str): 热力图使用的颜色映射
            global_sequence (str, optional): 全局序列窗口，用于显示全局贡献
            global_importance (torch.Tensor, optional): 全局重要性分数
            global_offsets (List[int], optional): 全局序列相对中心偏移
            global_top_k (int): 全局贡献图展示的Top位置数量
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        if not HAS_PLOTTING:
            logger.warning("未安装可视化依赖，无法生成重要性图")
            return None

        if local_sequence is None:
            raise ValueError("local_sequence 不能为空")

        # 处理局部重要性分数
        seq_len = len(local_sequence)
        if isinstance(importance_scores, torch.Tensor):
            scores = importance_scores[:seq_len].detach().cpu().numpy()
        else:
            scores = np.asarray(importance_scores)[:seq_len]

        if scores.shape[-1] != seq_len:
            raise ValueError(f"重要性分数长度 {scores.shape[-1]} 与局部序列长度 {seq_len} 不一致")

        center_pos = seq_len // 2

        if residue_offsets is None:
            residue_offsets = list(range(-center_pos, seq_len - center_pos))
        if len(residue_offsets) != seq_len:
            residue_offsets = list(range(-center_pos, seq_len - center_pos))

        # 处理全局信息
        has_global = False
        global_fig_offsets: Optional[List[int]] = None
        global_scores: Optional[np.ndarray] = None
        if global_sequence is not None and global_importance is not None:
            global_seq = str(global_sequence)
            if isinstance(global_importance, torch.Tensor):
                global_scores = global_importance[:len(global_seq)].detach().cpu().numpy()
            else:
                global_scores = np.asarray(global_importance)[:len(global_seq)]

            if global_scores.size == len(global_seq) and global_scores.size > 0:
                has_global = True
                if global_offsets is not None and len(global_offsets) == len(global_seq):
                    global_fig_offsets = list(global_offsets)
                else:
                    center = len(global_seq) // 2
                    global_fig_offsets = list(range(-center, len(global_seq) - center))
            else:
                logger.debug(
                    "跳过全局贡献图: global_sequence长度=%s, global_scores长度=%s",
                    len(global_seq),
                    None if global_scores is None else global_scores.size,
                )
                has_global = False

        base_width = max(14.0, seq_len * 0.5)
        local_height = 3.8

        # 创建图形
        if has_global:
            available_global = len(global_scores) if global_scores is not None else 0
            top_count = max(1, min(global_top_k if global_top_k > 0 else available_global, available_global))
            global_height = max(2.8, 0.32 * top_count)
            fig = plt.figure(figsize=(base_width, local_height + global_height))
            gs = fig.add_gridspec(2, 1, height_ratios=[global_height, local_height], hspace=0.5)
            ax_global = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
        else:
            fig, ax = plt.subplots(figsize=(base_width, local_height))
            ax_global = None

        fig.subplots_adjust(left=0.12, right=0.97)
        
        # 绘制热力图
        sns.heatmap(
            [scores],
            cmap=cmap,
            annot=np.array([list(local_sequence)]),
            fmt='',
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            cbar_kws={'label': 'Importance weight'},
            ax=ax
        )
        
        # 设置标题和标签
        display_title = title or 'Local residue importance'
        ax.set_title(display_title)
        ax.set_yticks([])
        tick_positions = np.arange(seq_len) + 0.5
        tick_labels = [f"{res}\n{offset:+d}" for res, offset in zip(local_sequence, residue_offsets)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        ax.set_xlabel('Relative position (offset from center)')
        
        # 在中心位置添加标记
        ax.add_patch(plt.Rectangle((center_pos, 0), 1, 1, fill=False, edgecolor='black', lw=2))
        
        if ax_global is not None and global_scores is not None and global_fig_offsets is not None:
            abs_scores = np.abs(global_scores)
            if global_top_k > 0 and global_top_k < len(abs_scores):
                top_indices = np.argsort(abs_scores)[-global_top_k:][::-1]
            else:
                top_indices = np.argsort(abs_scores)[::-1]
            residues = [
                f"{global_sequence[i]} ({global_fig_offsets[i]:+d})"
                for i in top_indices
            ]
            values = abs_scores[top_indices]
            signed_values = global_scores[top_indices]

            cmap_obj = plt.get_cmap(cmap)
            if values.size:
                vmin = float(values.min())
                vmax = float(values.max())
                if math.isclose(vmax, vmin):
                    vmax = vmin + 1e-6
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                colors = [cmap_obj(norm(v)) for v in values]
            else:
                colors = None

            bar_colors = colors[::-1] if colors else None
            ax_global.barh(residues[::-1], values[::-1], color=bar_colors, height=0.8)
            ax_global.invert_yaxis()
            ax_global.set_xlabel('Absolute importance')
            ax_global.set_title('Top global residues by importance')
            ax_global.tick_params(axis='y', labelsize=11)
            ax_global.margins(y=0.02)
            ax_global.grid(axis='x', linestyle='--', alpha=0.3)

            if values.size:
                max_val = values.max()
                ax_global.set_xlim(0, max_val * 1.2 if max_val > 0 else 1.0)
                offset = max(max_val * 0.03, 0.01)
                for idx, (val, signed) in enumerate(zip(values[::-1], signed_values[::-1])):
                    ax_global.text(val + offset, idx, f"{signed:.4f}", va='center', fontsize=9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None

        return fig


class AblationWrapper(nn.Module):
    """Wrapper for ablation studies on different components of the acetylation predictor."""
    
    def __init__(self, base_model_config, ablation_configs=None):
        """
        Initialize the ablation wrapper.
        
        Args:
            base_model_config: Configuration for the base model.
            ablation_configs (dict): Dictionary of configurations for ablation variants.
        """
        super().__init__()
        
        # Initialize base model
        self.base_model = AcetylationPredictor(**base_model_config)
        
        # Initialize ablation variants
        self.ablation_models = nn.ModuleDict()
        
        if ablation_configs:
            for name, config in ablation_configs.items():
                self.ablation_models[name] = AcetylationPredictor(**config)
                
    def forward(self, seq_data, local_seq_data=None, struct_data=None, model_name=None):
        """
        Forward pass.
        
        Args:
            seq_data: 全局序列数据
            local_seq_data: 局部序列数据
            struct_data: 结构数据 (可选)
            model_name (str): 使用的ablation模型名称 (None表示基础模型)
            
        Returns:
            torch.Tensor or dict: 模型输出
        """
        if model_name is None:
            return self.base_model(seq_data, local_seq_data, struct_data)
        else:
            return self.ablation_models[model_name](seq_data, local_seq_data, struct_data)
            
    def evaluate_all(self, seq_data, local_seq_data=None, struct_data=None):
        """
        评估所有模型变体
        
        Args:
            seq_data: 全局序列数据
            local_seq_data: 局部序列数据
            struct_data: 结构数据 (可选)
            
        Returns:
            dict: 所有模型的结果字典
        """
        results = {
            'base': self.base_model(seq_data, local_seq_data, struct_data)
        }
        
        for name in self.ablation_models:
            results[name] = self.ablation_models[name](seq_data, local_seq_data, struct_data)
            
        return results


class AcetylationPredictionModel(nn.Module):
    """完整的乙酰化预测模型 (Legacy Implementation)"""
    
    def __init__(self,
                 seq_feature_dim=1280,
                 struct_feature_dim=512,
                 hidden_dim=512,
                 context_window=15,
                 dropout=0.1,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.use_adaptive_dropout = bool(
            self.config.get("adaptive_dropout", False) and create_adaptive_dropout_layer is not None
        )

        self.seq_encoder = SequenceContextEncoder(
            input_dim=seq_feature_dim,
            hidden_dim=hidden_dim,
            context_window=context_window
        )

        self.struct_encoder = StructureContextEncoder(
            input_dim=struct_feature_dim,
            hidden_dim=hidden_dim // 2,
            context_window=context_window,
            adaptive_config=self.config if self.use_adaptive_dropout else None,
            use_adaptive_dropout=self.use_adaptive_dropout
        )

        seq_out_dim = self.seq_encoder.output_dim
        struct_out_dim = self.struct_encoder.output_dim

        self.fusion_module = GatedFusionModule(
            seq_dim=seq_out_dim,
            struct_dim=struct_out_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2,
            dropout=self.config.get("fusion_dropout", 0.2),
            adaptive_config=self.config if self.use_adaptive_dropout else None,
            use_adaptive_dropout=self.use_adaptive_dropout
        )
        
        if self.use_adaptive_dropout and create_adaptive_dropout_layer is not None:
            classifier_dropout = create_adaptive_dropout_layer(self.config, 'output')
        else:
            classifier_dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            classifier_dropout,
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self._warned_missing_structures = False
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                sequences: torch.Tensor,
                structures: torch.Tensor,
                site_positions: torch.Tensor,
                distance_matrices: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        seq_site_feat, seq_context_feat, seq_padding_mask = self.seq_encoder(sequences, site_positions, mask=mask)

        if structures is None:
            if not self._warned_missing_structures:
                logger.warning("AcetylationPredictionModel: 未提供结构特征，使用零向量并跳过距离信息。")
                self._warned_missing_structures = True
            batch_size, seq_len, _ = sequences.shape
            structures = sequences.new_zeros(batch_size, seq_len, self.struct_encoder.input_dim)
            distance_matrices = None
        elif distance_matrices is None:
            with torch.no_grad():
                distance_matrices = torch.cdist(structures.detach(), structures.detach(), p=2)

        struct_site_feat, struct_context_feat, struct_padding_mask = self.struct_encoder(
            structures,
            site_positions,
            distance_matrices
        )

        fused_features = self.fusion_module(
            seq_site_feat, 
            struct_site_feat,
            seq_context=seq_context_feat,
            struct_context=struct_context_feat,
            seq_padding_mask=seq_padding_mask,
            struct_padding_mask=struct_padding_mask
        )

        acetylation_logits = self.classifier(fused_features)
        strength_logits = self.regression_head(fused_features)

        return acetylation_logits, strength_logits


class MultiTaskLoss(nn.Module):
    """多任务损失函数，支持动态权重更新和Focal Loss"""

    def __init__(self, alpha: float = 0.7, class_weights=None, alpha_bounds: Optional[Tuple[float, float]] = None, use_focal_loss: bool = True):
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.register_buffer('class_weights', class_weights)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_bounds[0]) if alpha_bounds else float(alpha)
        self.alpha_max = float(alpha_bounds[1]) if alpha_bounds else float(alpha)

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            # Focal Loss parameters can be tuned or passed via config if needed
            self.classification_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.mse_loss = nn.MSELoss(reduction='mean')

    def set_alpha(self, new_alpha: float):
        self.alpha = float(new_alpha)

    def configure_alpha_schedule(self, min_alpha: float, max_alpha: float):
        self.alpha_min = float(min_alpha)
        self.alpha_max = float(max(self.alpha_min, max_alpha))

    def update_alpha_by_progress(self, progress: float):
        progress_tensor = torch.clamp(torch.tensor(progress, dtype=torch.float32), 0.0, 1.0)
        progress = float(progress_tensor.item())
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * progress

    def set_class_weights(self, class_weights):
        if not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=self.class_weights.dtype, device=self.class_weights.device)
        class_weights = class_weights.to(self.class_weights.device).to(self.class_weights.dtype)
        self.class_weights.copy_(class_weights)
        
        # Update CrossEntropyLoss weights if not using Focal Loss
        if not self.use_focal_loss:
             self.classification_loss.weight = self.class_weights

    def forward(self,
                predictions: Tuple[torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_pred, strength_pred = predictions
        class_target, strength_target = targets

        if self.use_focal_loss:
            # Focal Loss handles class imbalance internally via alpha parameter or we can pass weights if modified
            class_loss = self.classification_loss(class_pred, class_target.long())
        else:
            class_weights = self.class_weights.to(class_pred.device)
            class_loss = F.cross_entropy(class_pred, class_target.long(), weight=class_weights)

        mask = (class_target == 1).float()
        if mask.sum().item() > 0:
            strength_pred = strength_pred.squeeze(-1)
            reg_loss = self.mse_loss(strength_pred * mask, strength_target * mask)
        else:
            reg_loss = torch.zeros(1, device=class_pred.device, dtype=class_pred.dtype).squeeze()

        total_loss = self.alpha * class_loss + (1 - self.alpha) * reg_loss
        return total_loss, class_loss, reg_loss

