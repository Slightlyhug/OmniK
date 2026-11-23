"""
优化器和学习率调度器工具模块
提供各种优化器和调度器的创建和配置功能
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR
)
import math


def create_optimizer(
    model,
    config=None,
    optimizer_name=None,
    learning_rate=None,
    weight_decay=None,
    betas=None,
    eps=None,
    **kwargs
):
    """
    创建优化器
    
    Args:
        model: 模型
        config: 配置字典
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    config = dict(config or {})
    if kwargs:
        config.update(kwargs)

    optimizer_type = (optimizer_name or config.get('optimizer', config.get('optimizer_name', 'adamw'))).lower()
    learning_rate = learning_rate if learning_rate is not None else config.get('learning_rate', 2e-5)
    weight_decay = weight_decay if weight_decay is not None else config.get('weight_decay', 0.01)
    betas = betas if betas is not None else config.get('optimizer_betas', [0.9, 0.999])
    eps = eps if eps is not None else config.get('optimizer_eps', 1e-8)
    
    # 获取模型参数或参数组
    if hasattr(model, 'get_optimizer_groups') and callable(getattr(model, 'get_optimizer_groups')):
        params = model.get_optimizer_groups(base_lr=learning_rate, weight_decay=weight_decay)
    elif config.get('use_custom_param_groups', False):
        params = get_parameter_groups(model, config)
    else:
        params = model.parameters()
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get('optimizer_momentum', 0.9),
            nesterov=config.get('optimizer_nesterov', config.get('use_nesterov', True))
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get('optimizer_momentum', 0.9),
            alpha=config.get('rmsprop_alpha', 0.99)
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer,
    config=None,
    train_loader=None,
    grad_accum_steps=1,
    **kwargs
):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典
        train_loader: 训练数据加载器（用于计算总步数）
        
    Returns:
        torch.optim.lr_scheduler: 学习率调度器
    """
    config = dict(config or {})
    if kwargs:
        # 将旧参数映射到配置键
        arg_mapping = {
            'scheduler_name': 'scheduler',
            'epochs': 'epochs',
            'warmup_steps': 'warmup_steps',
            'num_cycles': 'scheduler_cycles',
            'T_max': 'scheduler_T_max',
            'step_size': 'scheduler_step_size',
            'gamma': 'scheduler_gamma',
            'patience': 'scheduler_patience',
            'factor': 'scheduler_factor',
            'threshold': 'scheduler_threshold',
            'mode': 'scheduler_mode',
            'min_lr': 'min_lr',
            'max_lr': 'max_learning_rate',
            'total_steps': 'total_steps',
            'pct_start': 'onecycle_pct_start',
            'div_factor': 'onecycle_div_factor',
            'final_div_factor': 'onecycle_final_div_factor'
        }
        for arg, value in kwargs.items():
            if value is None:
                continue
            key = arg_mapping.get(arg, arg)
            config[key] = value

    scheduler_type = str(config.get('scheduler', 'cosine')).lower()

    # 统一常见别名
    scheduler_aliases = {
        'cosineannealing': 'cosine',
        'cosineannealinglr': 'cosine',
        'cosine_warm_restart': 'cosine_warm_restarts',
        'cosine_warm_restarts': 'cosine_warm_restarts',
        'cosinewarmrestarts': 'cosine_warm_restarts',
        'cosine_with_warmup': 'warmup_cosine',
        'cosine-warmup': 'warmup_cosine',
        'cosinewarmup': 'warmup_cosine',
        'cosine_warmup': 'warmup_cosine',
        'one_cycle': 'onecycle',
        'onecyclelr': 'onecycle'
    }
    scheduler_type = scheduler_aliases.get(scheduler_type, scheduler_type)
    
    if scheduler_type == 'cosine':
        T_max = config.get('scheduler_T_max', config.get('epochs', 50))
        eta_min = config.get('min_lr', 1e-7)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    
    elif scheduler_type == 'cosine_warm_restarts':
        T_0 = config.get('scheduler_T_0', 10)
        T_mult = config.get('scheduler_T_mult', 2)
        eta_min = config.get('min_lr', 1e-7)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    
    elif scheduler_type == 'step':
        step_size = config.get('scheduler_step_size', 10)
        gamma = config.get('scheduler_gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_type == 'multistep':
        milestones = config.get('scheduler_milestones', [30, 60, 90])
        gamma = config.get('scheduler_gamma', 0.1)
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif scheduler_type == 'exponential':
        gamma = config.get('scheduler_gamma', 0.95)
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    elif scheduler_type == 'plateau' or scheduler_type == 'reduce_on_plateau':
        mode = config.get('scheduler_mode', 'min')
        factor = config.get('scheduler_factor', 0.5)
        patience = config.get('scheduler_patience', 10)
        threshold = config.get('scheduler_threshold', 1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            verbose=True
        )
    
    elif scheduler_type == 'warmup_cosine' or scheduler_type == 'cosine_with_warmup':
        # 带预热的余弦调度器
        warmup_steps = config.get('warmup_steps', 1000)
        total_steps = config.get('total_steps')

        if total_steps is None and train_loader is not None:
            epochs = config.get('epochs', 50)
            steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
            total_steps = steps_per_epoch * epochs

        if total_steps is None:
            raise ValueError("使用warmup_cosine调度器时需要提供total_steps或train_loader")

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=config.get('min_lr', 1e-7)
        )

    elif scheduler_type == 'onecycle':
        epochs = config.get('epochs', 50)
        total_steps = config.get('total_steps')

        if total_steps is None and train_loader is not None:
            steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
            total_steps = steps_per_epoch * epochs

        if total_steps is None:
            raise ValueError("使用onecycle调度器时需要提供total_steps或train_loader")

        max_lr = config.get('max_learning_rate', config.get('learning_rate', 2e-5) * 5)
        pct_start = config.get('onecycle_pct_start', 0.3)
        div_factor = config.get('onecycle_div_factor', 25.0)
        final_div_factor = config.get('onecycle_final_div_factor', 1e4)
        anneal_strategy = config.get('onecycle_anneal_strategy', 'cos')
        cycle_momentum = config.get('onecycle_cycle_momentum', True)
        base_momentum = config.get('onecycle_base_momentum', 0.85)
        max_momentum = config.get('onecycle_max_momentum', 0.95)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum
        )
    
    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = None
    
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return scheduler, scheduler_type


class WarmupCosineScheduler:
    """带预热的余弦学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        """更新学习率"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 预热阶段：线性增长
            lr_scale = self.step_count / self.warmup_steps
        else:
            # 余弦退火阶段
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = self.eta_min + (1 - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


def get_parameter_groups(model, config):
    """
    创建参数组，为不同层设置不同的学习率
    
    Args:
        model: 模型
        config: 配置字典
        
    Returns:
        list: 参数组列表
    """
    base_lr = config.get('learning_rate', 2e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    # 默认参数组
    param_groups = []
    
    # 特征提取层 - 较低学习率
    feature_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ['projection', 'fusion_transformer', 'feature_fusion']):
            feature_params.append(param)
    
    if feature_params:
        param_groups.append({
            'params': feature_params,
            'lr': base_lr * 0.5,
            'weight_decay': weight_decay
        })
    
    # 高级模块 - 正常学习率
    advanced_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ['rl_module', 'causal_module', 'meta_module', 'classifier']):
            advanced_params.append(param)
    
    if advanced_params:
        param_groups.append({
            'params': advanced_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    # 其他参数
    other_params = []
    feature_param_ids = set(id(p) for p in feature_params)
    advanced_param_ids = set(id(p) for p in advanced_params)
    
    for param in model.parameters():
        if id(param) not in feature_param_ids and id(param) not in advanced_param_ids:
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    return param_groups if param_groups else [{'params': model.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]
