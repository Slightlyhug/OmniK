"""
自适应Dropout模块
根据训练进度和验证性能动态调整dropout率
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class AdaptiveDropout(nn.Module):
    """
    自适应Dropout层，可以根据训练进度动态调整dropout率
    """
    
    def __init__(self, initial_dropout=0.3, min_dropout=0.1, max_dropout=0.5, 
                 schedule='linear', warmup_epochs=10, total_epochs=50):
        """
        初始化自适应Dropout
        
        Args:
            initial_dropout (float): 初始dropout率
            min_dropout (float): 最小dropout率
            max_dropout (float): 最大dropout率
            schedule (str): 调度策略 ('linear', 'exponential', 'cosine', 'adaptive')
            warmup_epochs (int): 预热轮数
            total_epochs (int): 总训练轮数
        """
        super().__init__()
        self.initial_dropout = initial_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # 当前dropout率
        self.current_dropout = initial_dropout
        self.current_epoch = 0
        
        # 性能历史（用于自适应调整）
        self.performance_history = []
        self.patience_counter = 0
        self.best_performance = float('inf')
        
        logger.info(f"初始化自适应Dropout: 初始={initial_dropout}, 范围=[{min_dropout}, {max_dropout}], 策略={schedule}")
    
    def update_epoch(self, epoch, val_loss=None):
        """
        更新当前epoch和dropout率
        
        Args:
            epoch (int): 当前epoch
            val_loss (float): 验证损失（用于自适应调整）
        """
        self.current_epoch = epoch
        
        if self.schedule == 'adaptive' and val_loss is not None:
            self._adaptive_update(val_loss)
        else:
            self._scheduled_update()
    
    def _scheduled_update(self):
        """根据预定义策略更新dropout率"""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：从最大值线性降到初始值
            warmup_progress = self.current_epoch / self.warmup_epochs
            self.current_dropout = self.max_dropout - (self.max_dropout - self.initial_dropout) * warmup_progress
        else:
            # 主训练阶段
            adjusted_progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            adjusted_progress = min(adjusted_progress, 1.0)
            
            if self.schedule == 'linear':
                # 线性衰减
                self.current_dropout = self.initial_dropout - (self.initial_dropout - self.min_dropout) * adjusted_progress
            elif self.schedule == 'exponential':
                # 指数衰减
                decay_factor = math.exp(-3 * adjusted_progress)  # 衰减因子
                self.current_dropout = self.min_dropout + (self.initial_dropout - self.min_dropout) * decay_factor
            elif self.schedule == 'cosine':
                # 余弦衰减
                self.current_dropout = self.min_dropout + (self.initial_dropout - self.min_dropout) * \
                                     (1 + math.cos(math.pi * adjusted_progress)) / 2
            else:
                # 保持初始值
                self.current_dropout = self.initial_dropout
        
        # 确保在有效范围内
        self.current_dropout = max(self.min_dropout, min(self.max_dropout, self.current_dropout))
    
    def _adaptive_update(self, val_loss):
        """基于验证性能自适应更新dropout率"""
        self.performance_history.append(val_loss)

        # 检查性能是否改进
        if val_loss < self.best_performance:
            self.best_performance = val_loss
            self.patience_counter = 0
            # 性能改进，可以适当降低dropout
            self.current_dropout = max(self.min_dropout, self.current_dropout * 0.95)
        else:
            self.patience_counter += 1
            # 性能没有改进，增加dropout以防止过拟合
            if self.patience_counter >= 3:
                # 增加dropout，但确保不超过0.95（留出安全边界）
                self.current_dropout = min(0.95, self.current_dropout * 1.05)
                self.patience_counter = 0

        # 确保在有效范围内 [0, 1]
        self.current_dropout = max(0.0, min(1.0, self.current_dropout))
        # 再次确保在配置的范围内
        self.current_dropout = max(self.min_dropout, min(self.max_dropout, self.current_dropout))
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            return nn.functional.dropout(x, p=self.current_dropout, training=True)
        else:
            return x
    
    def get_current_dropout(self):
        """获取当前dropout率"""
        return self.current_dropout
    
    def get_dropout_info(self):
        """获取dropout信息"""
        return {
            'current_dropout': self.current_dropout,
            'current_epoch': self.current_epoch,
            'schedule': self.schedule,
            'performance_history_length': len(self.performance_history),
            'best_performance': self.best_performance if self.performance_history else None
        }


class DropoutScheduler:
    """
    Dropout调度器，管理模型中所有的自适应dropout层
    """
    
    def __init__(self, model, config):
        """
        初始化Dropout调度器
        
        Args:
            model: 包含AdaptiveDropout层的模型
            config: 配置字典
        """
        self.model = model
        self.config = config
        self.adaptive_dropouts = []
        
        # 查找模型中的所有AdaptiveDropout层
        self._find_adaptive_dropouts()
        
        logger.info(f"找到 {len(self.adaptive_dropouts)} 个自适应Dropout层")
    
    def _find_adaptive_dropouts(self):
        """查找模型中的所有AdaptiveDropout层"""
        for name, module in self.model.named_modules():
            if isinstance(module, AdaptiveDropout):
                self.adaptive_dropouts.append((name, module))
    
    def step(self, epoch, val_loss=None):
        """
        更新所有自适应dropout层
        
        Args:
            epoch (int): 当前epoch
            val_loss (float): 验证损失
        """
        for name, dropout_layer in self.adaptive_dropouts:
            dropout_layer.update_epoch(epoch, val_loss)
            
            if epoch % 10 == 0:  # 每10个epoch记录一次
                info = dropout_layer.get_dropout_info()
                logger.info(f"Dropout层 {name}: 当前率={info['current_dropout']:.4f}")
    
    def get_all_dropout_rates(self):
        """获取所有dropout层的当前率"""
        rates = {}
        for name, dropout_layer in self.adaptive_dropouts:
            rates[name] = dropout_layer.get_current_dropout()
        return rates
    
    def set_dropout_rates(self, rates_dict):
        """手动设置dropout率"""
        for name, dropout_layer in self.adaptive_dropouts:
            if name in rates_dict:
                dropout_layer.current_dropout = rates_dict[name]


def create_adaptive_dropout_layer(config, layer_name="default"):
    """
    创建自适应dropout层的工厂函数
    
    Args:
        config: 配置字典
        layer_name: 层名称，用于获取特定配置
        
    Returns:
        AdaptiveDropout: 自适应dropout层
    """
    # 获取基础配置
    base_dropout = config.get('dropout', 0.3)
    
    # 根据层名称获取特定配置
    layer_configs = {
        'input': {
            'initial_dropout': config.get('input_dropout', base_dropout * 0.7),
            'min_dropout': 0.05,
            'max_dropout': base_dropout
        },
        'hidden': {
            'initial_dropout': config.get('hidden_dropout', base_dropout),
            'min_dropout': 0.1,
            'max_dropout': base_dropout * 1.5
        },
        'output': {
            'initial_dropout': config.get('output_dropout', base_dropout * 0.5),
            'min_dropout': 0.02,
            'max_dropout': base_dropout * 0.8
        },
        'attention': {
            'initial_dropout': config.get('attention_dropout', base_dropout * 0.3),
            'min_dropout': 0.01,
            'max_dropout': base_dropout * 0.5
        },
        'lstm': {
            'initial_dropout': config.get('lstm_dropout', base_dropout * 0.7),
            'min_dropout': 0.05,
            'max_dropout': base_dropout
        }
    }
    
    # 获取层特定配置或使用默认配置
    layer_config = layer_configs.get(layer_name, {
        'initial_dropout': base_dropout,
        'min_dropout': 0.1,
        'max_dropout': min(0.95, base_dropout * 1.1)  # 确保max_dropout不超过0.95
    })

    # 确保max_dropout不超过1.0
    layer_config['max_dropout'] = min(0.95, layer_config['max_dropout'])

    return AdaptiveDropout(
        initial_dropout=layer_config['initial_dropout'],
        min_dropout=layer_config['min_dropout'],
        max_dropout=layer_config['max_dropout'],
        schedule=config.get('dropout_schedule', 'linear'),
        warmup_epochs=config.get('warmup_epochs', 10),
        total_epochs=config.get('epochs', 50)
    )
