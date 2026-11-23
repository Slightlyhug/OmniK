"""
模型管理模块
负责模型创建、优化器和调度器设置
"""

import logging
import torch

# 使用系统路径导入
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.acetylation_predictor import AcetylationPredictor
from src.models.dual_branch_predictor import DualBranchFusionPredictor
from .optimizer_utils import create_scheduler


class ModelManager:
    """模型管理器，负责模型和优化器的创建"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def create_model(self):
        """创建模型"""
        logger = logging.getLogger("model_manager")

        core_model_type = self.config.get('core_model_type', 'ptm_bert_bilstm')
        if core_model_type == 'dual_branch_fusion':
            logger.info("创建DualBranchFusionPredictor模型用于K-Fold训练流程")
            model = DualBranchFusionPredictor(config=self.config)
        else:
            logger.info("创建AcetylationPredictor模型(core_model_type=%s)用于K-Fold训练流程", core_model_type)
            seq_encoder_type = self.config.get('seq_encoder_type', 'esm2_t33_650M_UR50D')
            struct_encoder_type = self.config.get('struct_encoder_type', 'esm_if1_gvp4_t16_142M_UR50')
            model = AcetylationPredictor(
                config=self.config,
                seq_encoder_type=seq_encoder_type,
                struct_encoder_type=struct_encoder_type,
                core_model_type=core_model_type,
                use_moe=self.config.get('use_moe', False),
                use_reinforcement=self.config.get('use_reinforcement', False),
                use_meta_learning=self.config.get('use_meta_learning', False),
                use_causal=self.config.get('use_causal', False),
                use_interpretability=self.config.get('use_interpretability', True),
                use_precomputed_features=self.config.get('use_precomputed_features', True)
            )

        # 移动模型到设备
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型创建完成，总参数: {total_params:,}，可训练参数: {trainable_params:,}")
        self._print_model_info()

        return model
    
    def _print_model_info(self):
        """打印模型信息"""
        print("模型初始化完成，使用以下高级模块:")
        print(f"- 强化学习: {self.config.get('use_reinforcement', False)}")
        print(f"- 因果推理: {self.config.get('use_causal', False)}")
        print(f"- 元学习: {self.config.get('use_meta_learning', False)}")
        print(f"- 专家混合: {self.config.get('use_moe', False)}")
        print(f"- 集成方法: {self.config.get('integration_method', 'serial')}")
        print(f"- 层标准化: {self.config.get('use_layer_norm', False)}")
        print(f"- Dropout率: {self.config.get('dropout', 0.3)}")
    
    def create_optimizer_and_scheduler(self, model, train_loader):
        """创建优化器和调度器"""
        print("创建优化器和学习率调度器...")
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, train_loader)
        
        return optimizer, scheduler
    
    def _create_optimizer(self, model):
        """创建优化器"""
        # 分组参数 - 对不同层应用不同的学习率
        param_groups = self._create_parameter_groups(model)
        
        optimizer_type = self.config.get('optimizer_type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2)
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2),
                momentum=0.9
            )
        else:
            print(f"未知优化器类型: {optimizer_type}，使用默认AdamW")
            optimizer = torch.optim.AdamW(param_groups)
        
        print(f"优化器: {optimizer_type}, 参数组数量: {len(param_groups)}")
        return optimizer
    
    def _create_parameter_groups(self, model):
        """创建参数组"""
        return [{'params': model.parameters()}]
    
    def _create_scheduler(self, optimizer, train_loader):
        """创建学习率调度器"""
        scheduler, scheduler_type = create_scheduler(
            optimizer,
            self.config,
            train_loader=train_loader,
            grad_accum_steps=self.config.get('grad_accum_steps', 1)
        )

        if scheduler_type and scheduler is not None:
            print(f"学习率调度器: {scheduler_type}")
        elif scheduler_type:
            print(f"学习率调度器配置为 {scheduler_type}，但未创建实例")
        else:
            print("未使用学习率调度器")

        # 保留归一化调度器名称供后续使用
        self.config['scheduler_normalized'] = scheduler_type

        return scheduler
