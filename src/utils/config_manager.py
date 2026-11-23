"""
配置管理模块
负责配置文件的加载、验证、合并和保存
"""

import os
import json
import argparse
from typing import Dict, Any, Optional


class ConfigManager:
    """配置文件管理器，用于加载、验证和动态调整配置参数"""
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None):
        self.config_path = config_path
        self.args = args
        self.config = self._load_config()
        self._validate_config()
        self._merge_args()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and os.path.exists(self.config_path):
            print(f"从配置文件加载参数: {self.config_path}")
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            print("使用默认配置参数")
            config = self._get_default_config()

        # 提取嵌套配置到顶级
        if 'model_config' in config and isinstance(config['model_config'], dict):
            for key, value in config['model_config'].items():
                if key not in config:  # 避免覆盖已有的顶级配置
                    config[key] = value

        if 'training_config' in config and isinstance(config['training_config'], dict):
            for key, value in config['training_config'].items():
                if key not in config:
                    config[key] = value

        if 'scheduler_config' in config and isinstance(config['scheduler_config'], dict):
            for key, value in config['scheduler_config'].items():
                if key not in config:
                    config[key] = value

        if 'early_stopping_config' in config and isinstance(config['early_stopping_config'], dict):
            for key, value in config['early_stopping_config'].items():
                if key not in config:
                    config[key] = value

        if 'regularization_config' in config and isinstance(config['regularization_config'], dict):
            for key, value in config['regularization_config'].items():
                if key not in config:
                    config[key] = value

        if 'data_config' in config and isinstance(config['data_config'], dict):
            for key, value in config['data_config'].items():
                if key not in config:
                    config[key] = value

        if 'augmentation_config' in config and isinstance(config['augmentation_config'], dict):
            for key, value in config['augmentation_config'].items():
                if key not in config:
                    config[key] = value

        if 'evaluation_config' in config and isinstance(config['evaluation_config'], dict):
            for key, value in config['evaluation_config'].items():
                if key not in config:
                    config[key] = value

        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 模型结构参数
            "hidden_dim": 512,
            "projection_dim": 512,
            "bert_hidden_dim": 768,
            "force_simple_bert": False,
            "dropout": 0.9,
            "use_layer_norm": True,
            
            # 训练参数 - 优化学习率策略
            "learning_rate": 2e-5,  # 调整初始学习率适配CosineAnnealingLR
            "weight_decay": 0.3,   # 降低权重衰减
            "epochs": 200,
            "batch_size": 64,
            "grad_accum_steps": 2,
            
            # 优化器和调度器参数 - 支持多种调度器
            "optimizer": "adamw",  # adamw, adam, sgd
            "scheduler": "cosine_with_warmup",  # cosine, cosine_with_warmup, cosine_restarts, step, plateau, exponential, none
            "warmup_steps": 500,   # 预热步数（用于cosine_with_warmup）
            "scheduler_cycles": 0.5,  # 余弦调度器的周期数
            "scheduler_T_max": 150,  # CosineAnnealingLR的T_max，None时使用epochs
            "scheduler_eta_min": 1e-6,  # CosineAnnealingLR的最小学习率
            "scheduler_T_0": 50,  # CosineAnnealingWarmRestarts的初始重启周期
            "scheduler_T_mult": 2,  # CosineAnnealingWarmRestarts的周期倍增因子
            "scheduler_step_size": 30,  # StepLR的步长
            "scheduler_gamma": 0.1,  # StepLR/ExponentialLR的衰减因子
            "scheduler_patience": 15,  # ReduceLROnPlateau的耐心值
            "scheduler_factor": 0.5,  # ReduceLROnPlateau的衰减因子
            "scheduler_threshold": 1e-4,  # ReduceLROnPlateau的阈值
            "scheduler_mode": "max",  # ReduceLROnPlateau的模式 (min/max)
            "scheduler_min_lr": 1e-7,  # ReduceLROnPlateau的最小学习率
            
            # 优化器参数
            "optimizer_betas": [0.9, 0.999],  # Adam/AdamW的beta参数（调整为标准值）
            "optimizer_eps": 1e-8,  # Adam/AdamW的epsilon参数（调整为标准值）
            "optimizer_momentum": 0.9,
            "optimizer_nesterov": True,
            
            # 正则化参数 - 针对CosineAnnealingLR优化
            "gradient_clip_val": 1.1,  # 增加梯度裁剪值
            "use_early_stopping": True,  # 启用早停
            "early_stopping_patience": 15,  # 增加早停patience
            "early_stopping_min_delta": 0.0001,  # 调整最小变化量
            "early_stopping_monitor": "val_loss",  # 监控指标：val_loss, val_acc, val_f1, val_auc
            "early_stopping_mode": "min",  # min表示越小越好，max表示越大越好
            "restore_best_weights": True,  # 早停时是否恢复最佳权重
            "early_stopping_verbose": True,  # 是否输出详细的早停信息

            # Dropout配置
            "input_dropout": 0.2,  # 输入层dropout
            "hidden_dropout": 0.3,  # 隐藏层dropout
            "output_dropout": 0.1,  # 输出层dropout
            "attention_dropout": 0.1,  # 注意力层dropout
            "lstm_dropout": 0.2,  # LSTM层dropout
            "adaptive_dropout": True,  # 是否使用自适应dropout
            "dropout_schedule": "linear",  # dropout调度：linear, exponential, cosine
            
            # 数据增强参数
            "use_enhanced_augmentation": True,
            "augment_prob": 0.7,
            "feature_noise_level": 0.25,
            "feature_mask_prob": 0.35,
            
            # 损失函数参数
            "use_focal_loss": True,
            "focal_alpha": 0.3,
            "focal_gamma": 2.0,
            "use_label_smoothing": True,
            "label_smoothing": 0.1,
            "loss_combination_weight": 0.8,
            
            # ================== 高级模块参数 ==================
            
            # 基础高级模块开关
            "use_reinforcement": False,
            "use_causal": False,
            "use_meta_learning": False,
            "use_moe": False,
            "integration_method": "ensemble",  # serial, ensemble, dynamic

            # 强化学习模块配置
            "reinforcement_learning": {
                "reward_type": "performance_based",  # performance_based, confidence_based, hybrid
                "exploration_epsilon": 0.1,
                "policy_learning_rate": 1e-5,
                "discount_factor": 0.95,
                "experience_buffer_size": 10000,
                "ppo_clip_param": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "num_decision_steps": 3
            },

            # 因果推断模块配置
            "causal_inference": {
                "causal_method": "structural",  # structural, instrumental, counterfactual
                "intervention_strength": 0.1,
                "causal_regularization": 0.01,
                "causal_discovery_method": "pc",  # pc, ges, lingam
                "use_backdoor_adjustment": True,
                "latent_dim": 32,
                "causal_hidden_dim": 128
            },

            # 元学习模块配置
            "meta_learning": {
                "meta_algorithm": "maml",  # maml, protonet
                "inner_learning_rate": 0.01,
                "num_inner_steps": 5,
                "support_set_size": 10,
                "query_set_size": 15,
                "embedding_dim": 256,
                "distance_metric": "euclidean"  # euclidean, cosine
            },

            # 专家混合模块配置
            "mixture_of_experts": {
                "num_experts": 8,
                "top_k": 2,
                "routing_method": "top_k",  # top_k, sparse
                "expert_type": "feed_forward",  # feed_forward, transformer
                "expert_hidden_dim": 256,
                "load_balancing_loss_coef": 0.01
            },
            
            # 多模块集成参数
            "module_integration": {
                "fusion_method": "attention",  # attention, concatenation, addition, gating
                "attention_heads": 8,  # 注意力头数
                "fusion_hidden_dim": 512,  # 融合隐藏维度
                "module_weights": [1.0, 1.0, 1.0, 1.0],  # 各模块权重
                "adaptive_weighting": True,  # 是否使用自适应权重
                "weight_learning_rate": 1e-5,  # 权重学习率
                "temperature": 1.0,  # 软最大温度
                "use_residual_connection": True,  # 是否使用残差连接
                "dropout_integration": 0.1,  # 集成dropout
                "cross_module_attention": True,  # 是否使用跨模块注意力
                "synchronization_method": "gradual",  # gradual, alternating, simultaneous
                "module_communication": True,  # 模块间通信
            },
            
            # 数据处理参数
            "use_structure": True,
            "use_interpretability": True,
            "use_class_weights": True,
            "pos_weight": 1.0,  # 正样本权重
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "stratified_split": True,
            "balanced_sampling": True,
            "optimize_threshold": True,
            "threshold_metric": "f1",
            "threshold_min_precision": 0.0,
            "threshold_min_specificity": 0.0,
            "threshold_min_recall": 0.0,
            "target_ptm_type": "phosphorylation",
            
            # ================== K-Fold交叉验证参数 ==================
            "use_kfold": True,  # 是否使用k-fold交叉验证
            "kfold_splits": 5,  # k-fold的k值
            "kfold_stratified": True,  # 是否使用分层k-fold
            "kfold_shuffle": True,  # 是否在k-fold前打乱数据
            "kfold_random_state": 42,  # k-fold随机种子
            "kfold_save_models": True,  # 是否保存每个fold的模型
            "kfold_ensemble_method": "average",  # 集成方法: average, voting, weighted
            "kfold_early_stopping": True,  # 是否在k-fold中使用早停
            "kfold_parallel": False,  # 是否并行执行k-fold（实验性功能）
            "kfold_save_predictions": True,  # 是否保存每个fold的预测结果
            "kfold_detailed_metrics": True,  # 是否计算详细的统计指标
        }
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        print("验证配置参数...")
        
        # 验证学习率参数
        if self.config.get('max_learning_rate', 0) <= self.config.get('learning_rate', 0):
            print("警告: max_learning_rate 应该大于 learning_rate，自动调整")
            self.config['max_learning_rate'] = self.config.get('learning_rate', 1e-4) * 5
        
        # 验证dropout范围
        dropout = self.config.get('dropout', 0.2)
        if not 0 <= dropout <= 1:
            print(f"警告: dropout值 {dropout} 超出范围[0,1]，重置为0.2")
            self.config['dropout'] = 0.2
        
        # 验证批次大小
        batch_size = self.config.get('batch_size', 32)
        if batch_size <= 0:
            print(f"警告: batch_size {batch_size} 无效，重置为32")
            self.config['batch_size'] = 32
        
        # 验证epoch数量
        epochs = self.config.get('epochs', 50)
        if epochs <= 0:
            print(f"警告: epochs {epochs} 无效，重置为50")
            self.config['epochs'] = 50
        
        print("配置参数验证完成")
    
    def _merge_args(self):
        """合并命令行参数到配置中"""
        if self.args is None:
            return
            
        specified_args = getattr(self.args, '_specified_args', None)
        # 优先使用命令行参数覆盖配置文件参数
        arg_to_config_mapping = {
            # 基础参数
            'epochs': 'epochs',  # 修复：添加epochs参数映射
            'batch_size': 'batch_size',
            'grad_accum_steps': 'grad_accum_steps',
            'optimizer': 'optimizer',
            'scheduler': 'scheduler',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'warmup_steps': 'warmup_steps',
            'use_reinforcement': 'use_reinforcement',
            'use_causal': 'use_causal',
            'use_meta_learning': 'use_meta_learning',
            'use_moe': 'use_moe',
            'integration_method': 'integration_method',
            'target_ptm_type': 'target_ptm_type',
            'core_model_type': 'core_model_type',  # 添加core_model_type映射
            'seq_encoder_type': 'seq_encoder_type',  # 添加seq_encoder_type映射
            'struct_encoder_type': 'struct_encoder_type',  # 添加struct_encoder_type映射

            # K-Fold交叉验证参数 - 新增
            'use_kfold': 'use_kfold',
            'kfold_splits': 'kfold_splits',
            'kfold_stratified': 'kfold_stratified',
            
            # 强化学习模块参数
            'reward_type': ['reinforcement_learning', 'reward_type'],
            'exploration_epsilon': ['reinforcement_learning', 'exploration_epsilon'],
            'policy_learning_rate': ['reinforcement_learning', 'policy_learning_rate'],
            'discount_factor': ['reinforcement_learning', 'discount_factor'],
            
            # 因果推理模块参数
            'causal_method': ['causal_inference', 'causal_method'],
            'intervention_strength': ['causal_inference', 'intervention_strength'],
            'causal_regularization': ['causal_inference', 'causal_regularization'],
            'causal_discovery_method': ['causal_inference', 'causal_discovery_method'],
            
            # 元学习模块参数
            'meta_algorithm': ['meta_learning', 'meta_algorithm'],
            'inner_learning_rate': ['meta_learning', 'inner_learning_rate'],
            'num_inner_steps': ['meta_learning', 'num_inner_steps'],
            'support_set_size': ['meta_learning', 'support_set_size'],
            'query_set_size': ['meta_learning', 'query_set_size'],
            
            # 专家混合模块参数
            'num_experts': ['mixture_of_experts', 'num_experts'],
            'top_k': ['mixture_of_experts', 'top_k'],
            'routing_method': ['mixture_of_experts', 'routing_method'],
            'expert_type': ['mixture_of_experts', 'expert_type'],
            'load_balancing_weight': ['mixture_of_experts', 'load_balancing_weight'],
            
            # 模块集成参数
            'fusion_method': ['module_integration', 'fusion_method'],
            'attention_heads': ['module_integration', 'attention_heads'],
            'adaptive_weighting': ['module_integration', 'adaptive_weighting'],
        }
        
        for arg_name, config_key in arg_to_config_mapping.items():
            if not hasattr(self.args, arg_name):
                continue
            if specified_args is not None and arg_name not in specified_args:
                continue

            arg_value = getattr(self.args, arg_name)
            if arg_value is None:
                continue

            if isinstance(config_key, list):
                # 嵌套配置
                parent_key, child_key = config_key
                if parent_key not in self.config or not isinstance(self.config[parent_key], dict):
                    self.config[parent_key] = {}

                old_value = self.config[parent_key].get(child_key)
                if old_value != arg_value:
                    print(f"命令行参数覆盖: {parent_key}.{child_key} {old_value} -> {arg_value}")
                    self.config[parent_key][child_key] = arg_value
            else:
                # 普通配置
                old_value = self.config.get(config_key)
                if old_value != arg_value:
                    print(f"命令行参数覆盖: {config_key} {old_value} -> {arg_value}")
                    self.config[config_key] = arg_value

    def get_config(self) -> Dict[str, Any]:
        """获取最终配置"""
        return self.config
    
    def print_config(self):
        """打印配置信息"""
        print("="*80)
        print("最终配置参数:")
        print("="*80)
        
        # 按类别分组打印
        categories = {
            "模型结构": ["hidden_dim", "projection_dim", "bert_hidden_dim", "force_simple_bert", "dropout", "use_layer_norm"],
            "训练超参数": ["learning_rate", "weight_decay", "epochs", "batch_size", "grad_accum_steps"],
            "优化器": ["optimizer", "optimizer_betas", "optimizer_eps", "optimizer_momentum"],
            "调度器参数": ["scheduler", "warmup_steps", "scheduler_cycles", "scheduler_T_max"],
            "正则化": ["gradient_clip_val", "use_early_stopping", "early_stopping_patience"],
            "数据增强": ["use_enhanced_augmentation", "augment_prob", "feature_noise_level"],
            "损失函数": ["use_focal_loss", "focal_alpha", "focal_gamma", "use_label_smoothing"],
            "高级模块开关": ["use_reinforcement", "use_causal", "use_meta_learning", "use_moe", "integration_method"],
            "数据处理": ["use_structure", "use_interpretability", "val_ratio", "test_ratio", "target_ptm_type"]
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if key in self.config:
                    print(f"  {key}: {self.config[key]}")
        
        # 打印高级模块详细配置
        advanced_modules = ["reinforcement_learning", "causal_inference", "meta_learning", 
                          "mixture_of_experts", "module_integration"]
        
        for module_name in advanced_modules:
            if module_name in self.config:
                print(f"\n{module_name.replace('_', ' ').title()} 模块参数:")
                module_config = self.config[module_name]
                for key, value in module_config.items():
                    print(f"  {key}: {value}")
        
        print("="*80)
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        # 创建可序列化的配置副本，排除不可序列化的对象
        serializable_config = {}
        non_serializable_keys = ['tensorboard_writer']  # 不可序列化的键
        
        for key, value in self.config.items():
            if key not in non_serializable_keys:
                # 检查值是否可JSON序列化
                try:
                    json.dumps(value)
                    serializable_config[key] = value
                except (TypeError, ValueError):
                    # 如果不可序列化，记录但跳过
                    print(f"跳过不可序列化的配置项: {key}")
        
        with open(output_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        print(f"配置已保存到: {output_path}")


def demonstrate_config_usage():
    """演示配置文件的使用方法"""
    print("\n" + "="*80)
    print("高级模块配置文件使用方法演示")
    print("="*80)
    
    print("\n1. 基本使用方法 (使用AdamW + Cosine with Warmup):")
    print("   python train_advanced_model_refactored.py")
    
    print("\n2. 启用强化学习模块:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_reinforcement \\")
    print("          --reward_type performance_based \\")
    print("          --exploration_epsilon 0.1 \\")
    print("          --policy_learning_rate 1e-5")
    
    print("\n3. 启用因果推理模块:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_causal \\")
    print("          --causal_method structural \\")
    print("          --intervention_strength 0.1 \\")
    print("          --causal_regularization 0.01")
    
    print("\n4. 启用元学习模块:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_meta_learning \\")
    print("          --meta_algorithm maml \\")
    print("          --inner_learning_rate 0.01 \\")
    print("          --num_inner_steps 5 \\")
    print("          --support_set_size 10")
    
    print("\n5. 启用专家混合模块:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_moe \\")
    print("          --num_experts 8 \\")
    print("          --top_k 2 \\")
    print("          --routing_method top_k \\")
    print("          --expert_type feed_forward")
    
    print("\n6. 多模块组合训练 (完整示例):")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_moe --use_causal --use_reinforcement \\")
    print("          --integration_method ensemble \\")
    print("          --fusion_method attention \\")
    print("          --attention_heads 8 \\")
    print("          --adaptive_weighting \\")
    print("          --num_experts 6 \\")
    print("          --top_k 2 \\")
    print("          --causal_method structural \\")
    print("          --reward_type hybrid \\")
    print("          --optimizer adamw \\")
    print("          --scheduler cosine_with_warmup \\")
    print("          --learning_rate 1e-4")
    
    print("\n7. 使用配置文件进行高级训练:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --config config/advanced_modules_config.json \\")
    print("          --num_experts 12 --top_k 3")  # 覆盖配置文件中的参数

    print("\n8. 快速测试配置:")
    print("   python train_advanced_model_refactored.py \\")
    print("          --use_moe --num_experts 4 --top_k 2 \\")
    print("          --batch_size 16 --epochs 10")

    print("\n" + "="*80)
    print("高级模块参数详细说明:")
    print("="*80)
    
    module_descriptions = {
        "强化学习模块": {
            "reward_type": "奖励类型：performance_based(基于性能)、confidence_based(基于置信度)、hybrid(混合)",
            "exploration_epsilon": "探索概率：控制随机探索的程度，通常0.01-0.3",
            "policy_learning_rate": "策略网络学习率：通常比主网络学习率小1-2个数量级",
            "discount_factor": "折扣因子：未来奖励的权重，通常0.9-0.99",
            "experience_buffer_size": "经验回放缓冲区大小：存储历史经验，影响学习稳定性"
        },
        "因果推理模块": {
            "causal_method": "因果方法：structural(结构化)、instrumental(工具变量)、counterfactual(反事实)",
            "intervention_strength": "干预强度：控制因果干预的力度，通常0.01-0.5",
            "causal_regularization": "因果正则化：约束模型学习因果关系，通常0.001-0.1",
            "causal_discovery_method": "因果发现方法：pc(PC算法)、ges(GES算法)、lingam(LiNGAM算法)",
            "use_backdoor_adjustment": "后门调整：控制混淆变量的影响"
        },
        "元学习模块": {
            "meta_algorithm": "元学习算法：maml(MAML)、reptile(Reptile)、fomaml(First-Order MAML)、prototypical(原型网络)",
            "inner_learning_rate": "内循环学习率：任务适应时的学习率，通常0.001-0.1",
            "num_inner_steps": "内循环步数：每个任务的适应步数，通常1-10",
            "support_set_size": "支持集大小：用于快速适应的样本数，通常5-50",
            "use_second_order": "二阶梯度：是否使用二阶导数信息，提高但增加计算成本"
        },
        "专家混合模块": {
            "num_experts": "专家数量：并行的专家网络数目，通常4-64",
            "routing_method": "路由方法：top_k(选择top-k)、sparse(稀疏)、soft(软选择)、switch(开关)",
            "top_k": "选择专家数：激活的专家数量，通常1-4",
            "load_balancing_weight": "负载平衡权重：确保专家使用均衡，通常0.001-0.1",
            "expert_specialization": "专家专业化：domain(领域)、feature(特征)、task(任务)、adaptive(自适应)"
        },
        "模块集成": {
            "fusion_method": "融合方法：attention(注意力)、concatenation(拼接)、addition(相加)、gating(门控)",
            "attention_heads": "注意力头数：多头注意力的头数，通常4-16",
            "adaptive_weighting": "自适应权重：动态调整各模块权重",
            "temperature": "软最大温度：控制权重分布的尖锐程度"
        }
    }
    
    for module, params in module_descriptions.items():
        print(f"\n{module}:")
        for param, desc in params.items():
            print(f"  • {param}: {desc}")

    print("\n" + "="*80)
    print("参数调优建议:")
    print("="*80)
    print("\n• 初学者建议: 先使用默认配置，逐步调整单个模块")
    print("• 计算资源限制: 减少num_experts、batch_size，增加grad_accum_steps")
    print("• 性能优化: 调整learning_rate、weight_decay、scheduler参数")
    print("• 模块组合: 建议先单模块训练，再尝试多模块集成")
    print("• 超参数搜索: 使用配置文件批量测试不同参数组合")
    print("\n" + "="*80)
