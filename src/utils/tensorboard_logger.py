"""
TensorBoard日志记录器
提供统一的训练过程日志记录接口
"""

import os
from typing import Dict, Any, Optional
import torch


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, output_dir: str, writer):
        self.output_dir = output_dir
        self.writer = writer
        self.step = 0
        
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """记录多个标量值"""
        if step is None:
            step = self.step
        for tag, value in tag_scalar_dict.items():
            self.writer.add_scalar(tag, value, step)
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """记录直方图"""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_model_parameters(self, model: torch.nn.Module, step: Optional[int] = None):
        """记录模型参数分布"""
        if step is None:
            step = self.step
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'parameters/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """记录学习率"""
        if step is None:
            step = self.step
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
            
    def log_training_metrics(self, metrics: Dict[str, float], epoch: int, step: Optional[int] = None):
        """记录训练指标"""
        if step is None:
            step = self.step
        
        # 记录训练指标
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'train/{metric_name}', value, step)
            
    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int, step: Optional[int] = None):
        """记录验证指标"""
        if step is None:
            step = self.step
        
        # 记录验证指标
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'val/{metric_name}', value, step)
            
    def log_epoch_summary(self, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], epoch: int):
        """记录epoch总结"""
        # 记录epoch级别的指标
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'epoch/train_{metric_name}', value, epoch)
            
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'epoch/val_{metric_name}', value, epoch)
    
    def increment_step(self):
        """增加步数计数器"""
        self.step += 1
        
    def set_step(self, step: int):
        """设置当前步数"""
        self.step = step
        
    def close(self):
        """关闭writer"""
        if self.writer:
            self.writer.close()
            
    def flush(self):
        """刷新缓冲区"""
        if self.writer:
            self.writer.flush()
