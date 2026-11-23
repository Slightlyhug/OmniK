import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc
from collections import defaultdict

# 导入必要的模块
from .reinforcement import PPOTrainer
from .causal import StructuralCausalModel, DoWhy
from .meta_learning import MAML, ProtoNet, MetaTaskSampler

# 设置日志
logger = logging.getLogger("advanced_trainer")

class AdvancedTrainer:
    """
    高级学习模块的专用训练器，支持强化学习、因果推断和元学习。
    """
    
    def __init__(self, model, config, train_loader, val_loader):
        """
        初始化高级训练器
        
        Args:
            model: 预测模型
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 获取模型的设备
        self.device = next(model.parameters()).device
        logger.info(f"高级训练器使用设备: {self.device}")
        
        # 创建标准优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 创建损失函数
        if config.get('use_class_weights', True):
            # 使用正样本权重来处理类别不平衡
            pos_weight = torch.tensor([config.get('pos_weight', 1.5)]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(f"使用加权BCE损失，正样本权重: {pos_weight.item()}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("使用标准BCE损失（无权重）")
        
        # 创建学习率调度器
        scheduler_type = config.get('scheduler_type', 'reduce_on_plateau')
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=config.get('lr_factor', 0.5),
                patience=config.get('lr_patience', 5),
                verbose=True
            )
            logger.info(f"使用ReduceLROnPlateau调度器，因子: {config.get('lr_factor', 0.5)}, 耐心: {config.get('lr_patience', 5)}")
        elif scheduler_type == 'cosine':
            # 余弦退火调度，适合长期训练
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 50),
                eta_min=config.get('min_lr', 1e-6)
            )
            logger.info(f"使用CosineAnnealingLR调度器，T_max: {config.get('epochs', 50)}, 最小学习率: {config.get('min_lr', 1e-6)}")
        elif scheduler_type == 'one_cycle':
            # One Cycle Policy，适合快速训练
            steps_per_epoch = len(train_loader)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.get('learning_rate', 1e-4) * 10,
                total_steps=steps_per_epoch * config.get('epochs', 50),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4
            )
            logger.info(f"使用OneCycleLR调度器，最大学习率: {config.get('learning_rate', 1e-4) * 10}")
        else:
            # 默认使用固定学习率
            self.scheduler = None
            logger.info("不使用学习率调度器")
        
        # 特定模块的训练配置
        self.rl_config = {
            'ppo_epochs': config.get('ppo_epochs', 5),
            'clip_param': config.get('clip_param', 0.2),
            'value_coef': config.get('value_coef', 0.5),
            'entropy_coef': config.get('entropy_coef', 0.01)
        }
        
        self.causal_config = {
            'beta': config.get('causal_beta', 1.0)  # KL损失权重
        }
        
        self.meta_config = {
            'inner_lr': config.get('inner_lr', 0.01),
            'inner_steps': config.get('inner_steps', 5),
            'n_way': config.get('n_way', 2),
            'k_shot': config.get('k_shot', 5),
            'n_query': config.get('n_query', 10)
        }
        
        # 如果使用强化学习，创建PPO训练器
        if model.use_rl:
            self.ppo_trainer = PPOTrainer(
                model.actor_critic,
                optim.Adam(model.actor_critic.parameters(), lr=config.get('rl_lr', 5e-5)),
                clip_param=self.rl_config['clip_param'],
                value_coef=self.rl_config['value_coef'],
                entropy_coef=self.rl_config['entropy_coef']
            )
        
        # 梯度累积步数 - 这允许使用更大的有效批次大小
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        
        # 自动批次大小调整
        self.enable_auto_batch_size = config.get('enable_auto_batch_size', False)
        self.initial_batch_size = config.get('batch_size', 32)
        self.max_batch_size = config.get('max_batch_size', 128)
        
        # 如果启用自动批次大小调整，添加GPU内存监视
        if self.enable_auto_batch_size and torch.cuda.is_available():
            self.auto_adjust_batch_size()
        
        logger.info("高级训练器初始化完成")
    
    def auto_adjust_batch_size(self):
        """自动调整批次大小以最大化GPU内存使用"""
        logger.info("尝试自动调整批次大小以最大化GPU内存使用...")
        
        # 获取当前GPU内存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # 保留20%的内存作为缓冲
        target_memory = total_memory * 0.8
        
        # 获取当前已分配内存
        current_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        
        # 如果当前内存使用率低于目标的20%
        if current_memory < (target_memory * 0.2):
            logger.info(f"当前内存使用率较低: {current_memory:.2f}GB / {target_memory:.2f}GB")
            
            # 尝试增加batch_size
            new_batch_size = min(self.initial_batch_size * 2, self.max_batch_size)
            
            if new_batch_size > self.initial_batch_size:
                logger.info(f"调整批次大小: {self.initial_batch_size} -> {new_batch_size}")
                
                # 更新配置
                self.config['batch_size'] = new_batch_size
                
                # 这里需要重新创建数据加载器，但我们可能没有原始数据集的引用
                # 因此只记录建议，实际调整需要在外部实现
                logger.info(f"建议: 重新创建DataLoader，使用批次大小 {new_batch_size}")
        else:
            logger.info(f"当前内存使用率正常: {current_memory:.2f}GB / {target_memory:.2f}GB")
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch索引
            
        Returns:
            dict: 训练指标
        """
        self.model.train()
        metrics = defaultdict(float)
        
        # 根据模型的模块组合选择相应的训练方法
        if self.model.use_causal and not (self.model.use_rl or self.model.use_meta):
            # 只有因果模块
            return self.train_causal_epoch(epoch)
        elif self.model.use_rl and not (self.model.use_causal or self.model.use_meta):
            # 只有强化学习模块
            return self.train_rl_epoch(epoch)
        elif self.model.use_meta and not (self.model.use_rl or self.model.use_causal):
            # 只有元学习模块
            return self.train_meta_epoch(epoch)
        elif self.model.integration_method == 'serial':
            # 串行整合模式需要特殊处理
            return self.train_serial_integration_epoch(epoch)
        else:
            # 其他整合模式使用标准训练过程
            total_samples = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                # 将批次数据移动到正确设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                # 前向传播
                outputs = self.model(batch)
                labels = batch['label']
                
                # [Fix] Capture dictionary outputs for auxiliary losses
                raw_outputs = outputs if isinstance(outputs, dict) else None
                
                # 处理张量形状不匹配
                if isinstance(outputs, dict):
                    if 'predicted_outcome' in outputs or 'outcome_prob' in outputs:
                        outputs = outputs.get('predicted_outcome', outputs.get('outcome_prob', None))
                    elif 'action_logits' in outputs:
                        action_logits = outputs.get('action_logits')
                        if len(action_logits.shape) == 3:  # [batch_size, seq_len, num_classes]
                            action_logits = action_logits[:, -1, :]
                        probs = F.softmax(action_logits, dim=-1)
                        if probs.size(-1) > 1:
                            outputs = probs[:, 1].unsqueeze(1)
                        else:
                            outputs = probs.squeeze(-1)
                    else:
                        for possible_key in ['logits', 'probabilities', 'predictions', 'output']:
                            if possible_key in outputs:
                                outputs = outputs[possible_key]
                                break
                
                if isinstance(outputs, torch.Tensor):
                    if outputs.shape != labels.shape:
                        if len(outputs.shape) > len(labels.shape):
                            outputs = outputs.squeeze(-1)
                        else:
                            labels = labels.view(labels.shape[0], 1)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                
                # [Fix] Add auxiliary losses if present
                if raw_outputs is not None and 'auxiliary_losses' in raw_outputs:
                    aux_losses = raw_outputs['auxiliary_losses']
                    if aux_losses:
                        for name, aux_loss in aux_losses.items():
                            # Ensure aux_loss is a scalar or matches batch size (mean it if needed)
                            if aux_loss.numel() > 1:
                                aux_loss = aux_loss.mean()
                            loss += aux_loss
                            metrics[f'loss_{name}'] += aux_loss.item() * batch_size
                
                # 反向传播
                loss.backward()
                
                # 应用梯度裁剪，防止梯度爆炸
                max_norm = self.config.get('grad_clip', 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
                
                # 如果使用OneCycleLR调度器，在每个批次后更新
                if hasattr(self, 'scheduler') and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                # 更新指标
                batch_size = labels.size(0)
                total_samples += batch_size
                metrics['loss'] += loss.item() * batch_size
                
                # 计算准确率
                if outputs.min() >= 0 and outputs.max() <= 1:
                    preds = (outputs > 0.5).float()
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                
                accuracy = (preds == labels).float().mean().item()
                metrics['accuracy'] += accuracy * batch_size
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                              f"Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
            
            # 计算平均指标
            if total_samples > 0:
                metrics['loss'] /= total_samples
                metrics['accuracy'] /= total_samples
                
            logger.info(f"Epoch {epoch} 训练完成: Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
            return metrics
    
    def train_causal_epoch(self, epoch):
        """使用因果模块训练一个epoch"""
        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取融合特征
            fused_features = self.model.get_fused_features(batch)
            
            # 获取标签
            labels = batch['label']
            batch_size = labels.size(0)
            total_samples += batch_size
            
            # 前向传播通过因果模型
            causal_outputs = self.model.causal_module(fused_features)
            
            # 如果使用SCM
            if isinstance(self.model.causal_module, StructuralCausalModel):
                # 计算因果损失
                causal_loss, loss_components = self.model.causal_module.loss_function(
                    x=fused_features,
                    outputs=causal_outputs,
                    treatment=None,  # 使用模型估计的处理效应
                    outcome=labels,
                    beta=self.causal_config['beta']
                )
                
                # 反向传播
                causal_loss.backward()
                self.optimizer.step()
                
                # 计算ATE (因果效应)
                ate = self.model.causal_module.compute_causal_effect(fused_features).mean()
                
                # 计算预测准确率
                if 'predicted_outcome' in causal_outputs:
                    pred_outcome = causal_outputs['predicted_outcome']
                    if pred_outcome.shape != labels.shape:
                        if len(pred_outcome.shape) > len(labels.shape):
                            pred_outcome = pred_outcome.squeeze(-1)
                        else:
                            labels = labels.view(labels.shape[0], 1)
                    
                    # 转换为二分类预测
                    preds = (pred_outcome > 0.5).float()
                    accuracy = (preds == labels).float().mean().item()
                else:
                    accuracy = 0.0
                
                # 更新指标
                metrics['causal_loss'] += causal_loss.item() * batch_size
                metrics['recon_loss'] += loss_components['recon'] * batch_size
                metrics['kl_loss'] += loss_components['kl'] * batch_size
                metrics['outcome_loss'] += loss_components['outcome'] * batch_size
                metrics['ate'] += ate.item() * batch_size
                metrics['accuracy'] += accuracy * batch_size
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Causal Loss: {causal_loss.item():.4f}, ATE: {ate.item():.4f}, Acc: {accuracy:.4f}")
        
        # 计算平均指标
        if total_samples > 0:
            for key in metrics:
                metrics[key] /= total_samples
            
            # 添加标准化键名
            metrics['loss'] = metrics['causal_loss']  # 使用因果损失作为主损失
            
        logger.info(f"Epoch {epoch} 因果训练完成: Causal Loss: {metrics['causal_loss']:.4f}, ATE: {metrics['ate']:.4f}, Acc: {metrics['accuracy']:.4f}")
        return metrics
    
    def train_rl_epoch(self, epoch):
        """使用强化学习模块训练一个epoch"""
        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 在检查优化器相关问题时，最好简单一些，先尝试标准的训练流程
            # 获取融合特征
            fused_features = self.model.get_fused_features(batch)
            
            # 获取标签
            labels = batch['label'].to(self.device)
            
            # 转换为适合处理的形式
            batch_size = fused_features.shape[0]
            total_samples += batch_size
            
            # 训练指标累计
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            
            # 单步训练策略
            for batch_iter in range(1):  # 简化为单次更新，后续再扩展到PPO
                # 前向传播计算
                outputs = self.model.actor_critic(fused_features)
                action_logits = outputs['action_logits']
                values = outputs['state_value'].squeeze(-1)
                
                # 计算动作概率
                probs = F.softmax(action_logits, dim=-1)
                log_probs = F.log_softmax(action_logits, dim=-1)
                
                # 使用标签作为目标动作（二分类问题: 0=负, 1=正）
                # 这里简化问题，直接使用标签作为目标
                target_actions = (labels > 0.5).long()
                
                # 计算策略损失 (负对数似然)
                policy_loss = F.nll_loss(log_probs, target_actions)
                
                # 计算值函数损失 (MSE)
                value_loss = F.mse_loss(values, labels.float())
                
                # 计算熵以促进探索
                entropy = -(probs * log_probs).sum(-1).mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 清空梯度
                self.optimizer.zero_grad()
                
                # 反向传播
                loss.backward()
                
                # 应用梯度裁剪，防止梯度爆炸
                max_norm = self.config.get('grad_clip', 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                # 更新参数
                self.optimizer.step()
                
                # 累加损失
                epoch_policy_loss += policy_loss.item() * batch_size
                epoch_value_loss += value_loss.item() * batch_size
                epoch_entropy += entropy.item() * batch_size
            
            # 计算预测精度
            preds = (probs[:, 1] > 0.5).float()
            accuracy = (preds == (labels > 0.5).float()).float().mean().item()
            
            # 更新训练指标
            metrics['policy_loss'] += epoch_policy_loss
            metrics['value_loss'] += epoch_value_loss 
            metrics['entropy'] += epoch_entropy
            metrics['accuracy'] += accuracy * batch_size
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Policy Loss: {policy_loss.item():.4f}, "
                          f"Value Loss: {value_loss.item():.4f}, "
                          f"Acc: {accuracy:.4f}")
            
        # 计算平均损失
        if total_samples > 0:
            for key in metrics:
                metrics[key] /= total_samples
            
            # 添加总损失作为标准化的loss键
            metrics['loss'] = metrics['policy_loss'] + 0.5 * metrics['value_loss'] - 0.01 * metrics['entropy']
        
        logger.info(f"Epoch {epoch} RL训练完成: "
                   f"Policy Loss: {metrics['policy_loss']:.4f}, "
                   f"Value Loss: {metrics['value_loss']:.4f}, "
                   f"Loss: {metrics['loss']:.4f}, "
                   f"Acc: {metrics['accuracy']:.4f}")
                   
        return metrics
    
    def train_meta_epoch(self, epoch):
        """使用元学习模块训练一个epoch"""
        # 设置为训练模式
        self.model.train()
        metrics = defaultdict(float)
        
        # 创建任务采样器（此处简化，实际实现需根据具体训练集创建）
        task_sampler = MetaTaskSampler(
            self.train_loader.dataset,
            n_way=self.meta_config['n_way'],
            k_shot=self.meta_config['k_shot'],
            n_query=self.meta_config['n_query'],
            n_tasks=100  # 每个epoch的任务数
        )
        
        # 创建元学习优化器
        meta_optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('meta_lr', 1e-3))
        
        # 记录处理的任务数
        num_processed_tasks = 0
        
        # 对每个元学习任务
        for task_idx, (support_batch, query_batch) in enumerate(task_sampler):
            meta_optimizer.zero_grad()
            
            # 将批次数据移动到正确设备
            for key in support_batch:
                if isinstance(support_batch[key], torch.Tensor):
                    support_batch[key] = support_batch[key].to(self.device)
                elif isinstance(support_batch[key], dict):
                    for subkey in support_batch[key]:
                        if isinstance(support_batch[key][subkey], torch.Tensor):
                            support_batch[key][subkey] = support_batch[key][subkey].to(self.device)
                            
            for key in query_batch:
                if isinstance(query_batch[key], torch.Tensor):
                    query_batch[key] = query_batch[key].to(self.device)
                elif isinstance(query_batch[key], dict):
                    for subkey in query_batch[key]:
                        if isinstance(query_batch[key][subkey], torch.Tensor):
                            query_batch[key][subkey] = query_batch[key][subkey].to(self.device)
            
            # 提取支持集特征
            support_features = self.model.get_fused_features(support_batch)
            support_labels = support_batch['label']
            
            # 提取查询集特征
            query_features = self.model.get_fused_features(query_batch)
            query_labels = query_batch['label']
            
            # 确保标签形状正确
            if support_labels.dim() == 1:
                support_labels = support_labels.view(-1, 1)
            if query_labels.dim() == 1:
                query_labels = query_labels.view(-1, 1)
            
            # 确保特征有梯度信息
            support_features_with_grad = support_features.detach().clone().requires_grad_(True)
            
            try:
                # 使用梯度上下文进行适配
                with torch.enable_grad():
                    # 适应模型到支持集
                    adapted_classifier = self.model.meta_module.adapt_to_task(support_features_with_grad, support_labels)
                    
                    # 保存适配后的分类器到模型，供后续使用
                    self.model.adapted_classifier = adapted_classifier
                    
                    # 在查询集上评估
                    query_preds = adapted_classifier(query_features)
                    
                    # 处理形状不匹配
                    if query_preds.shape != query_labels.shape:
                        if len(query_preds.shape) > len(query_labels.shape):
                            query_preds = query_preds.squeeze(-1)
                        else:
                            query_labels = query_labels.view(query_labels.shape[0], 1)
                    
                    # 检查预测输出的范围
                    if query_preds.min() >= 0 and query_preds.max() <= 1:
                        # 预测已经是概率，使用BCE损失
                        query_loss = F.binary_cross_entropy(query_preds, query_labels)
                    else:
                        # 预测是logits，使用BCEWithLogits损失
                        query_loss = F.binary_cross_entropy_with_logits(query_preds, query_labels)
                    
                    # 反向传播
                    query_loss.backward()
                    
                    # 梯度裁剪以提高稳定性
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    # 更新参数
                    meta_optimizer.step()
                    
                    # 计算查询准确率
                    with torch.no_grad():
                        if query_preds.min() >= 0 and query_preds.max() <= 1:
                            # 已经是概率
                            pred_probs = query_preds
                        else:
                            # 转换logits为概率
                            pred_probs = torch.sigmoid(query_preds)
                        
                        query_acc = ((pred_probs > 0.5) == query_labels).float().mean().item()
                    
                    # 更新指标
                    metrics['meta_loss'] += query_loss.item()
                    metrics['query_acc'] += query_acc
                    
                    # 增加处理的任务计数
                    num_processed_tasks += 1
                    
                    if task_idx % 10 == 0:
                        logger.info(f"Epoch {epoch}, Task {task_idx}/100, "
                                f"Meta Loss: {query_loss.item():.4f}, Query Acc: {query_acc:.4f}")
            
            except Exception as e:
                logger.error(f"元学习任务训练失败: {e}")
                continue
        
        # 计算平均指标，使用处理的任务数而不是task_sampler的长度
        if num_processed_tasks > 0:
            metrics['meta_loss'] /= num_processed_tasks
            metrics['query_acc'] /= num_processed_tasks
            
            # 添加标准化的指标键名
            metrics['loss'] = metrics['meta_loss']  # 使用元学习损失作为主损失
            metrics['accuracy'] = metrics['query_acc']  # 使用查询准确率作为主准确率
        
        logger.info(f"Epoch {epoch} 元学习训练完成: Meta Loss: {metrics['meta_loss']:.4f}, "
                   f"Query Acc: {metrics['query_acc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def train_serial_integration_epoch(self, epoch):
        """
        使用串行整合训练一个epoch
        
        Args:
            epoch (int): 当前epoch
            
        Returns:
            dict: 训练指标
        """
        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch)
            labels = batch['label']
            batch_size = labels.size(0)
            total_samples += batch_size
            
            # [Fix] Capture dictionary outputs for auxiliary losses
            raw_outputs = outputs if isinstance(outputs, dict) else None
            
            # 处理字典类型的输出
            if isinstance(outputs, dict):
                # 如果是因果模块的输出字典
                if 'predicted_outcome' in outputs or 'outcome_prob' in outputs:
                    outputs = outputs.get('predicted_outcome', outputs.get('outcome_prob', None))
                    if outputs is None:
                        logger.error(f"训练时因果模块输出缺少预测结果")
                        continue
                # 如果是RL模块的输出字典
                elif 'action_logits' in outputs:
                    action_logits = outputs.get('action_logits')
                    # 检查并处理额外的维度
                    if len(action_logits.shape) == 3:  # [batch_size, seq_len, num_classes]
                        # 对于序列输出，使用最后一个时间步
                        action_logits = action_logits[:, -1, :]
                    
                    # 使用softmax获取概率值
                    probs = F.softmax(action_logits, dim=-1)
                    # 通常第二个值（索引1）对应正类的概率
                    if probs.size(-1) > 1:
                        # 只提取正类（索引1）的概率作为输出
                        outputs = probs[:, 1].unsqueeze(1)
                    else:
                        outputs = probs.squeeze(-1)
                else:
                    # 尝试查找其他可能的字典键
                    for possible_key in ['logits', 'probabilities', 'predictions', 'output']:
                        if possible_key in outputs:
                            outputs = outputs[possible_key]
                            break
                    else:
                        # 如果没有找到任何已知键，记录错误
                        logger.error(f"训练时遇到未知类型的字典输出: {outputs.keys()}")
                        continue
            elif isinstance(outputs, torch.Tensor):
                # 检查并处理可能的三维输出
                if len(outputs.shape) == 3:  # [batch_size, seq_len, num_classes]
                    if outputs.shape[2] == 2:  # 二分类情况
                        # 使用最后一个时间步的输出
                        outputs = outputs[:, -1, :]
                
                # 处理二维情况
                if len(outputs.shape) == 2 and outputs.shape[1] == 2:
                    # 如果是直接的[batch_size, 2]形式的张量输出，只取第二列
                    outputs = outputs[:, 1].unsqueeze(1)
            
            # 处理张量形状不匹配
            if outputs.shape != labels.shape:
                if len(outputs.shape) > len(labels.shape):
                    outputs = outputs.squeeze(-1)
                else:
                    labels = labels.view(labels.shape[0], 1)
            
            # 计算损失
            loss = self.criterion(outputs, labels)
            
            # [Fix] Add auxiliary losses if present
            if raw_outputs is not None and 'auxiliary_losses' in raw_outputs:
                aux_losses = raw_outputs['auxiliary_losses']
                if aux_losses:
                    for name, aux_loss in aux_losses.items():
                        # Ensure aux_loss is a scalar or matches batch size
                        if aux_loss.numel() > 1:
                            aux_loss = aux_loss.mean()
                        loss += aux_loss
                        metrics[f'integration_loss_{name}'] += aux_loss.item() * batch_size
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新指标
            metrics['integration_loss'] += loss.item() * batch_size
            
            # 计算准确率
            if outputs.min() >= 0 and outputs.max() <= 1:
                preds = (outputs > 0.5).float()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
            accuracy = (preds == labels).float().mean().item()
            metrics['integration_acc'] += accuracy * batch_size
        
        # 计算平均整合指标
        if total_samples > 0:
            metrics['integration_loss'] /= total_samples
            metrics['integration_acc'] /= total_samples
            
            # 添加标准化的指标键名，与其他训练方法保持一致
            metrics['loss'] = metrics['integration_loss']
            metrics['accuracy'] = metrics['integration_acc']
            
        logger.info(f"Epoch {epoch} 串行整合训练完成: "
                   f"Integration Loss: {metrics['integration_loss']:.4f}, "
                   f"Acc: {metrics['integration_acc']:.4f}")
        return metrics
        
    def validate(self):
        """
        在验证集上评估模型
        
        Returns:
            dict: 验证指标
        """
        self.model.eval()
        metrics = defaultdict(float)
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                # 前向传播
                outputs = self.model(batch)
                labels = batch['label']
                
                # 处理不同模块可能输出的字典类型
                if isinstance(outputs, dict):
                    if 'predicted_outcome' in outputs or 'outcome_prob' in outputs:
                        outputs = outputs.get('predicted_outcome', outputs.get('outcome_prob', None))
                        if outputs is None:
                            print("警告: 因果模块输出缺少预测结果，跳过此批次")
                            continue
                    elif 'action_logits' in outputs:
                        action_logits = outputs.get('action_logits')
                        if len(action_logits.shape) == 3:  # [batch_size, seq_len, num_classes]
                            action_logits = action_logits[:, -1, :]
                        probs = F.softmax(action_logits, dim=-1)
                        if probs.size(-1) > 1:
                            outputs = probs[:, 1].unsqueeze(1)
                        else:
                            outputs = probs.squeeze(-1)
                    else:
                        for possible_key in ['logits', 'probabilities', 'predictions', 'output']:
                            if possible_key in outputs:
                                outputs = outputs[possible_key]
                                break
                
                # 处理形状不匹配
                if outputs.shape != labels.shape:
                    if len(outputs.shape) > len(labels.shape):
                        outputs = outputs.squeeze(-1)
                    else:
                        labels = labels.view(labels.shape[0], 1)
                
                # 计算损失
                val_loss = self.criterion(outputs, labels)
                metrics['val_loss'] += val_loss.item() * labels.size(0)
                
                # 保存预测结果
                if outputs.min() >= 0 and outputs.max() <= 1:
                    probs = outputs
                else:
                    probs = torch.sigmoid(outputs)
                
                preds = (probs > 0.5).float()
                
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # Calculate metrics
        val_acc = accuracy_score(all_labels, all_preds)
        metrics['val_acc'] = val_acc
        
        # Only compute precision, recall, F1 if there are positive predictions
        if all_preds.sum() > 0:
            metrics['val_precision'] = precision_score(all_labels, all_preds, zero_division=0)
            metrics['val_recall'] = recall_score(all_labels, all_preds, zero_division=0)
            metrics['val_f1'] = f1_score(all_labels, all_preds, zero_division=0)
        else:
            metrics['val_precision'] = 0.0
            metrics['val_recall'] = 0.0
            metrics['val_f1'] = 0.0
        
        # Only compute AUC if there are both positive and negative samples
        if len(np.unique(all_labels)) > 1:
            metrics['val_auc'] = roc_auc_score(all_labels, all_probs)
            
            # 计算Matthews相关系数 (MCC)
            metrics['val_mcc'] = matthews_corrcoef(all_labels, all_preds)
            
            # 计算精确率-召回率曲线下面积 (AUPRC)
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            metrics['val_auprc'] = auc(recall, precision)
        else:
            metrics['val_auc'] = 0.5
            metrics['val_mcc'] = 0.0
            metrics['val_auprc'] = 0.0
        
        # Average loss
        metrics['val_loss'] /= len(all_labels)
        
        # 更新学习率调度器
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics['val_loss'])
            elif isinstance(self.scheduler, (optim.lr_scheduler.CosineAnnealingLR, 
                                            optim.lr_scheduler.OneCycleLR)):
                self.scheduler.step()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"当前学习率: {current_lr:.8f}")
        
        # 如果是因果模块，添加因果效应评估
        if self.model.use_causal:
            # 添加ATE指标
            try:
                # 首先检查estimate_ate方法是否存在
                if hasattr(self.model.causal_module, 'estimate_ate'):
                    metrics['val_ate'] = self.model.causal_module.estimate_ate(all_labels, all_probs)
                # 如果不存在，尝试使用compute_causal_effect方法
                elif hasattr(self.model.causal_module, 'compute_causal_effect'):
                    # 这里使用fused_features而不是labels和probs
                    # 首先获取一个批次的特征用于计算
                    batch = next(iter(self.val_loader))
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)
                        elif isinstance(batch[key], dict):
                            for subkey in batch[key]:
                                if isinstance(batch[key][subkey], torch.Tensor):
                                    batch[key][subkey] = batch[key][subkey].to(self.device)
                    
                    fused_features = self.model.get_fused_features(batch)
                    ate = self.model.causal_module.compute_causal_effect(fused_features).mean()
                    metrics['val_ate'] = ate.item()
                else:
                    # 如果两种方法都不存在，设置一个默认值
                    logger.warning("因果模块没有实现estimate_ate或compute_causal_effect方法，使用默认值0.0")
                    metrics['val_ate'] = 0.0
            except Exception as e:
                logger.warning(f"计算ATE时出错: {e}")
                metrics['val_ate'] = 0.0
        
        # 记录验证结果
        logger.info(f"验证结果: Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_acc']:.4f}, "
                   f"Precision: {metrics.get('val_precision', 0):.4f}, Recall: {metrics.get('val_recall', 0):.4f}, "
                   f"F1: {metrics.get('val_f1', 0):.4f}, AUC: {metrics.get('val_auc', 0):.4f}, "
                   f"MCC: {metrics.get('val_mcc', 0):.4f}, AUPRC: {metrics.get('val_auprc', 0):.4f}")
        
        return metrics 