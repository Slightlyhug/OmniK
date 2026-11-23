"""
简单日志记录器
当TensorBoard不可用时提供基本的日志记录功能
"""

import os
import json
from typing import Dict, Any, Optional
import torch


class SimpleLogger:
    """简单文件日志记录器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, 'training_log.txt')
        self.metrics_file = os.path.join(output_dir, 'metrics_history.json')
        self.step = 0
        self.metrics_history = []
        
        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write("=== 训练日志 ===\n")
            
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量值"""
        if step is None:
            step = self.step
        
        log_entry = f"Step {step} - {tag}: {value:.6f}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
            
    def log_scalars(self, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """记录多个标量值"""
        if step is None:
            step = self.step
            
        log_entries = [f"Step {step}:"]
        for tag, value in tag_scalar_dict.items():
            log_entries.append(f"  {tag}: {value:.6f}")
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """记录直方图（简化版本，只记录统计信息）"""
        if step is None:
            step = self.step
            
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            
        stats = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max())
        }
        
        log_entry = f"Step {step} - {tag} stats: {stats}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
            
    def log_model_parameters(self, model: torch.nn.Module, step: Optional[int] = None):
        """记录模型参数分布（简化版本）"""
        if step is None:
            step = self.step
            
        log_entries = [f"Step {step} - Model Parameters:"]
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats = {
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'norm': float(param.data.norm())
                }
                log_entries.append(f"  {name}: {param_stats}")
                
                if param.grad is not None:
                    grad_stats = {
                        'mean': float(param.grad.mean()),
                        'std': float(param.grad.std()),
                        'norm': float(param.grad.norm())
                    }
                    log_entries.append(f"  {name}_grad: {grad_stats}")
        
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
            
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """记录学习率"""
        if step is None:
            step = self.step
            
        log_entries = [f"Step {step} - Learning Rates:"]
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            log_entries.append(f"  group_{i}: {lr:.8f}")
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
            
    def log_training_metrics(self, metrics: Dict[str, float], epoch: int, step: Optional[int] = None):
        """记录训练指标"""
        if step is None:
            step = self.step
            
        # 添加到历史记录
        metrics_entry = {
            'epoch': epoch,
            'step': step,
            'type': 'train',
            'metrics': metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # 写入日志文件
        log_entries = [f"Epoch {epoch}, Step {step} - Training Metrics:"]
        for metric_name, value in metrics.items():
            log_entries.append(f"  {metric_name}: {value:.6f}")
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
            
    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int, step: Optional[int] = None):
        """记录验证指标"""
        if step is None:
            step = self.step
            
        # 添加到历史记录
        metrics_entry = {
            'epoch': epoch,
            'step': step,
            'type': 'validation',
            'metrics': metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # 写入日志文件
        log_entries = [f"Epoch {epoch}, Step {step} - Validation Metrics:"]
        for metric_name, value in metrics.items():
            log_entries.append(f"  {metric_name}: {value:.6f}")
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
            
    def log_epoch_summary(self, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], epoch: int):
        """记录epoch总结"""
        log_entries = [f"=== Epoch {epoch} Summary ==="]
        
        log_entries.append("Training Metrics:")
        for metric_name, value in train_metrics.items():
            log_entries.append(f"  {metric_name}: {value:.6f}")
            
        log_entries.append("Validation Metrics:")
        for metric_name, value in val_metrics.items():
            log_entries.append(f"  {metric_name}: {value:.6f}")
            
        log_entries.append("=" * 30)
        log_entries.append("")  # 空行
        
        with open(self.log_file, 'a') as f:
            f.write("\n".join(log_entries) + "\n")
    
    def increment_step(self):
        """增加步数计数器"""
        self.step += 1
        
    def set_step(self, step: int):
        """设置当前步数"""
        self.step = step
        
    def close(self):
        """保存历史记录并关闭"""
        # 保存指标历史到JSON文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        with open(self.log_file, 'a') as f:
            f.write("=== 训练结束 ===\n")
            
    def flush(self):
        """刷新缓冲区（文件系统会自动处理）"""
        pass
