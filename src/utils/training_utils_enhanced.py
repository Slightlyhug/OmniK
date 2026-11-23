"""
增强训练管理模块
负责训练循环、验证、早停等功能
采用与原始脚本一致的详细日志格式
"""

import os
import time
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import TensorBoard only when available to avoid dependency issues
HAS_TENSORBOARD = False
SummaryWriter = None

SCHEDULER_ALIASES = {
    'cosineannealing': 'cosine',
    'cosineannealinglr': 'cosine',
    'cosine_warm_restart': 'cosine_warm_restarts',
    'cosine_warm_restarts': 'cosine_warm_restarts',
    'cosinewarmrestarts': 'cosine_warm_restarts',
    'cosine_with_warmup': 'warmup_cosine',
    'cosine-warmup': 'warmup_cosine',
    'cosinewarmup': 'warmup_cosine',
    'cosine_warmup': 'warmup_cosine',
    'one_cycle': 'onecycle'
}

def _try_import_tensorboard():
    """尝试导入TensorBoard，如果失败则返回None"""
    global HAS_TENSORBOARD, SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter as TB_SummaryWriter
        HAS_TENSORBOARD = True
        SummaryWriter = TB_SummaryWriter
        return True
    except Exception as e:
        HAS_TENSORBOARD = False
        SummaryWriter = None
        return False
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score,
    matthews_corrcoef, precision_recall_curve, auc, confusion_matrix,
    balanced_accuracy_score
)

# 导入自适应dropout调度器
try:
    from .adaptive_dropout import DropoutScheduler
    HAS_DROPOUT_SCHEDULER = True
except ImportError:
    HAS_DROPOUT_SCHEDULER = False
    DropoutScheduler = None


def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"


class TensorRangeObserver:
    """Collect tensor statistics to help locate NaN/Inf sources during training."""

    def __init__(self, model, logger, output_dir, config=None):
        self.logger = logger
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.enabled = bool(self.config.get('enable_nan_debugging', False))
        self.global_step = 0
        self.context_stack = []
        self.handles = []
        self.module_counters = defaultdict(int)
        self.batch_counter = 0
        self.saved_tensor_count = 0

        # Early exit if disabled, but keep attributes callable.
        if not self.enabled or model is None:
            return

        self.frequency = max(1, int(self.config.get('nan_debug_frequency', 50)))
        self.log_finite = bool(self.config.get('nan_debug_log_finite', False))
        self.break_on_error = bool(self.config.get('nan_debug_break_on_error', False))
        self.save_on_issue = bool(self.config.get('nan_debug_save_tensors_on_anomaly', True))
        self.max_saved_tensors = max(0, int(self.config.get('nan_debug_max_saved_tensors', 10)))
        self.batch_check_enabled = bool(self.config.get('nan_debug_check_batches', False))
        self.batch_frequency = max(1, int(self.config.get('nan_debug_batch_frequency', 1)))
        self.max_records_per_call = max(0, int(self.config.get('nan_debug_max_tensors_per_call', 6)))

        self.targets = self._normalize_targets(self.config.get('nan_debug_targets'))
        self.watch_all = self.targets is None

        save_dir_cfg = self.config.get('nan_debug_save_dir')
        self.save_dir = Path(save_dir_cfg) if save_dir_cfg else (self.output_dir / 'nan_debug')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.save_dir / 'nan_events.jsonl'

        self.logger.info(
            "NaN调试已启用: watch_all=%s, targets=%s, log_finite=%s, frequency=%s, save_dir=%s",
            self.watch_all,
            self.targets if not self.watch_all else 'ALL',
            self.log_finite,
            self.frequency,
            self.save_dir,
        )

        self._register_hooks(model)

    # ------------------------------------------------------------------
    # Public helper API
    # ------------------------------------------------------------------
    def update_global_step(self, step):
        if not self.enabled:
            return
        self.global_step = int(step)

    def push_context(self, **context):
        if not self.enabled:
            return
        merged = dict(context)
        merged.setdefault('global_step', self.global_step)
        self.context_stack.append(merged)

    def pop_context(self):
        if not self.enabled or not self.context_stack:
            return
        self.context_stack.pop()

    def inspect_batch(self, batch, *, phase, epoch=None, batch_idx=None, global_step=None):
        if not self.enabled or not self.batch_check_enabled:
            return

        self.batch_counter += 1
        if self.batch_counter % self.batch_frequency != 0:
            return

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'batch',
        }

        log_snapshot = self.log_finite and (self.batch_counter % self.frequency == 0)
        logged_snapshots = 0

        for tensor_index, (name, tensor) in enumerate(self._iter_named_tensors(batch)):
            stats = self._tensor_stats(tensor)
            if stats is None:
                continue

            stats.update({
                'module': name,
                'kind': 'batch_input',
                'tensor_index': tensor_index,
                'call_index': self.batch_counter,
            })

            snapshot_allowed = log_snapshot and (
                self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
            )

            logged = self._handle_stats(
                stats,
                context,
                tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                log_snapshot=snapshot_allowed,
                anomaly_event='batch_anomaly',
            )

            if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                logged_snapshots += 1

    def observe_tensor(
        self,
        name,
        tensor,
        *,
        phase=None,
        epoch=None,
        batch_idx=None,
        global_step=None,
        reason='value',
        log_snapshot=False,
    ):
        if not self.enabled:
            return

        stats = self._tensor_stats(tensor)
        if stats is None:
            return

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'observe_tensor',
            'module_name': name,
        }

        stats.update({
            'module': name,
            'kind': reason,
            'tensor_index': 0,
            'call_index': None,
        })

        self._handle_stats(
            stats,
            context,
            tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
            log_snapshot=log_snapshot,
        )

    def observe_scalar(
        self,
        name,
        value,
        *,
        phase=None,
        epoch=None,
        batch_idx=None,
        global_step=None,
    ):
        if not self.enabled:
            return

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            scalar = float(value.detach().cpu().item())
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                return

        stats = {
            'module': name,
            'kind': 'scalar',
            'tensor_index': 0,
            'call_index': None,
            'dtype': 'float',
            'shape': [],
            'numel': 1,
            'min': scalar,
            'max': scalar,
            'mean': scalar,
            'std': 0.0,
            'abs_max': abs(scalar),
            'has_nan': math.isnan(scalar),
            'has_inf': math.isinf(scalar),
            'device': 'cpu',
            'requires_grad': False,
        }

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'scalar',
            'module_name': name,
        }

        self._handle_stats(
            stats,
            context,
            tensor=None,
            log_snapshot=False,
            anomaly_event='scalar_anomaly',
        )

    def close(self):
        if not self.handles:
            return
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                continue
        self.handles.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_targets(self, targets):
        if targets is None:
            return []
        if isinstance(targets, str):
            text = targets.strip()
            if not text:
                return []
            if text.lower() in {'*', 'all', 'any'}:
                return None
            return [text.lower()]
        normalized = []
        watch_all = False
        for item in targets if isinstance(targets, (list, tuple, set)) else [targets]:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            if text.lower() in {'*', 'all', 'any'}:
                watch_all = True
            else:
                normalized.append(text.lower())
        if watch_all:
            return None
        return normalized

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            module_name = name or module.__class__.__name__
            if not self._should_watch(module_name, module):
                continue
            try:
                handle = module.register_forward_hook(self._make_hook(module_name))
                self.handles.append(handle)
            except Exception as exc:
                self.logger.warning("注册NaN调试钩子失败: module=%s, error=%s", module_name, exc)

        if not self.handles:
            self.logger.warning("NaN调试未注册任何模块钩子，请检查nan_debug_targets配置")

    def _should_watch(self, module_name, module):
        if self.watch_all:
            return True
        if not self.targets:
            # 默认仅监控最终模型输出
            return module_name in {'model', module.__class__.__name__}

        label = module_name.lower()
        class_name = module.__class__.__name__.lower()
        for target in self.targets:
            if target in label or target in class_name:
                return True
        return False

    def _make_hook(self, module_name):
        def hook(module, inputs, output):
            if not self.enabled:
                return

            call_index = self.module_counters[module_name] + 1
            self.module_counters[module_name] = call_index

            context = dict(self._current_context())
            context.setdefault('scope', 'forward_hook')
            context['module_name'] = module_name
            context['call_index'] = call_index

            log_snapshot = self.log_finite and (call_index % self.frequency == 0)
            logged_snapshots = 0

            for tensor_index, tensor in enumerate(self._iter_tensors(inputs)):
                stats = self._tensor_stats(tensor)
                if stats is None:
                    continue
                stats.update({
                    'module': module_name,
                    'kind': 'input',
                    'tensor_index': tensor_index,
                    'call_index': call_index,
                })
                snapshot_allowed = log_snapshot and (
                    self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
                )
                logged = self._handle_stats(
                    stats,
                    context,
                    tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                    log_snapshot=snapshot_allowed,
                )
                if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                    logged_snapshots += 1

            for tensor_index, tensor in enumerate(self._iter_tensors(output)):
                stats = self._tensor_stats(tensor)
                if stats is None:
                    continue
                stats.update({
                    'module': module_name,
                    'kind': 'output',
                    'tensor_index': tensor_index,
                    'call_index': call_index,
                })
                snapshot_allowed = log_snapshot and (
                    self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
                )
                logged = self._handle_stats(
                    stats,
                    context,
                    tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                    log_snapshot=snapshot_allowed,
                )
                if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                    logged_snapshots += 1

        return hook

    def _current_context(self):
        if self.context_stack:
            return self.context_stack[-1]
        return {'global_step': self.global_step}

    def _iter_tensors(self, value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._iter_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._iter_tensors(item)

    def _iter_named_tensors(self, value, prefix=None):
        if isinstance(value, torch.Tensor):
            yield prefix or 'tensor', value
        elif isinstance(value, dict):
            for key, item in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_named_tensors(item, new_prefix)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                yield from self._iter_named_tensors(item, new_prefix)

    def _tensor_stats(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return None

        with torch.no_grad():
            data = tensor.detach()

            is_nested = bool(getattr(data, "is_nested", False))
            if is_nested:
                try:
                    # Attempt to convert to padded tensor for stats
                    # Use 0.0 as padding value which is standard for stats
                    padded_data = data.to_padded_tensor(0.0)
                    
                    # Recursively call _tensor_stats on the padded version
                    # But mark it as nested in the result
                    nested_stats = self._tensor_stats(padded_data)
                    if nested_stats:
                        nested_stats['is_nested'] = True
                        nested_stats['notes'] = 'nested_tensor_converted_to_padded'
                        return nested_stats
                        
                except Exception as exc:
                    self.logger.warning(
                        "NaN调试: NestedTensor 无法转换为稠密张量 (%s)，记录简化统计",
                        exc,
                    )
                    return {
                        'dtype': str(data.dtype),
                        'shape': None, # Shape might be ragged
                        'numel': None,
                        'device': str(data.device),
                        'requires_grad': bool(tensor.requires_grad),
                        'has_nan': False, # Cannot determine easily without conversion
                        'has_inf': False,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None,
                        'abs_max': None,
                        'notes': 'nested_tensor_conversion_failed',
                    }

            stats = {
                'dtype': str(data.dtype),
                'shape': list(data.shape),
                'numel': int(data.numel()),
                'device': str(data.device),
                'requires_grad': bool(tensor.requires_grad),
                'is_nested': is_nested,
            }

            if data.numel() == 0:
                stats.update({
                    'has_nan': False,
                    'has_inf': False,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'abs_max': None,
                })
                return stats

            try:
                if data.is_floating_point() or data.dtype in {torch.float16, torch.bfloat16}:
                    has_nan = bool(torch.isnan(data).any().item())
                    has_inf = bool(torch.isinf(data).any().item())
                    finite_mask = torch.isfinite(data)
                    if finite_mask.any():
                        finite_values = data[finite_mask].float()
                        stats['min'] = float(finite_values.min().item())
                        stats['max'] = float(finite_values.max().item())
                        stats['mean'] = float(finite_values.mean().item())
                        stats['abs_max'] = float(torch.max(torch.abs(finite_values)).item())
                        if finite_values.numel() > 1:
                            stats['std'] = float(finite_values.std(unbiased=False).item())
                        else:
                            stats['std'] = 0.0
                    else:
                        stats['min'] = stats['max'] = stats['mean'] = stats['std'] = stats['abs_max'] = None
                    stats['has_nan'] = has_nan
                    stats['has_inf'] = has_inf
                else:
                    data_fp = data.float()
                    stats['min'] = float(data_fp.min().item())
                    stats['max'] = float(data_fp.max().item())
                    stats['mean'] = float(data_fp.mean().item())
                    stats['abs_max'] = float(torch.max(torch.abs(data_fp)).item())
                    if data_fp.numel() > 1:
                        stats['std'] = float(data_fp.std(unbiased=False).item())
                    else:
                        stats['std'] = 0.0
                    stats['has_nan'] = False
                    stats['has_inf'] = False
            except Exception as exc:
                stats.update({
                    'has_nan': False,
                    'has_inf': False,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'abs_max': None,
                    'error': str(exc),
                })

            return stats

    def _handle_stats(self, stats, context, tensor, log_snapshot, *, anomaly_event='anomaly', snapshot_event='snapshot'):
        logged = False
        has_nan = bool(stats.get('has_nan'))
        has_inf = bool(stats.get('has_inf'))

        if has_nan or has_inf:
            self._emit_event(stats, context, anomaly_event, tensor=tensor)
            logged = True
        elif log_snapshot and self.log_finite:
            self._emit_event(stats, context, snapshot_event, tensor=None)
            logged = True

        return logged

    def _emit_event(self, stats, context, event_type, tensor=None):
        event = {
            'event_type': event_type,
            'timestamp': time.time(),
            'module': stats.get('module'),
            'kind': stats.get('kind'),
            'tensor_index': stats.get('tensor_index'),
            'call_index': stats.get('call_index'),
            'shape': stats.get('shape'),
            'dtype': stats.get('dtype'),
            'numel': stats.get('numel'),
            'min': stats.get('min'),
            'max': stats.get('max'),
            'mean': stats.get('mean'),
            'std': stats.get('std'),
            'abs_max': stats.get('abs_max'),
            'has_nan': stats.get('has_nan'),
            'has_inf': stats.get('has_inf'),
            'device': stats.get('device'),
            'requires_grad': stats.get('requires_grad'),
            'context': context,
        }

        message = (
            f"[NaN Debug][{event_type.upper()}] module={event['module']} kind={event['kind']} "
            f"shape={event['shape']} dtype={event['dtype']} min={event['min']} max={event['max']} "
            f"mean={event['mean']} std={event['std']} abs_max={event['abs_max']} "
            f"context=({self._format_context(context)})"
        )

        if str(event_type).lower().endswith('anomaly'):
            self.logger.error(message)
        else:
            self.logger.info(message)

        if tensor is not None and self.save_on_issue and self.saved_tensor_count < self.max_saved_tensors:
            try:
                file_name = self._build_tensor_filename(event)
                file_path = self.save_dir / file_name
                torch.save(tensor.detach().cpu(), file_path)
                event['tensor_path'] = str(file_path)
                self.saved_tensor_count += 1
            except Exception as exc:
                self.logger.warning("保存NaN调试张量失败: %s", exc)

        try:
            with open(self.events_path, 'a', encoding='utf-8') as fp:
                fp.write(json.dumps(self._to_serializable(event), ensure_ascii=False) + '\n')
        except Exception as exc:
            self.logger.warning("记录NaN调试事件失败: %s", exc)

        if str(event_type).lower().endswith('anomaly') and self.break_on_error:
            raise RuntimeError(message)

    def _format_context(self, context):
        keys = ['phase', 'epoch', 'batch_idx', 'global_step', 'call_index', 'scope', 'module_name']
        parts = []
        for key in keys:
            value = context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return ', '.join(parts)

    def _build_tensor_filename(self, event):
        module = str(event.get('module') or 'tensor').replace('.', '_')
        kind = str(event.get('kind') or 'value')
        call_index = event.get('call_index')
        tensor_index = event.get('tensor_index')
        global_step = event.get('context', {}).get('global_step', 'na')
        phase = event.get('context', {}).get('phase', 'unknown')
        return f"{phase}_step{global_step}_call{call_index}_idx{tensor_index}_{module}_{kind}.pt"

    def _to_serializable(self, event):
        def convert(value):
            if isinstance(value, (int, float, str)) or value is None:
                return value
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            try:
                return float(value)
            except Exception:
                return str(value)

        return convert(event)

class EnhancedLoss(nn.Module):
    """增强损失函数：Focal Loss + 对比学习"""
    def __init__(self, alpha=0.25, gamma=2.0, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        
    def focal_loss(self, predictions, targets):
        """Focal Loss处理类别不平衡"""
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # Calculate BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
    def contrastive_loss(self, features, labels):
        """对比学习增强特征表示"""
        if features is None:
            return torch.tensor(0.0, device=labels.device)
            
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # 计算特征相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建正负样本掩码
        labels = labels.float().view(-1, 1)
        positive_mask = (labels == labels.T).float()
        
        # Mask out self-similarity
        mask_diag = torch.eye(labels.shape[0], device=labels.device)
        positive_mask = positive_mask - mask_diag
        
        negative_mask = 1 - positive_mask - mask_diag
        
        # 计算对比损失
        # Log-Sum-Exp trick for numerical stability could be used, but following report logic:
        exp_sim = torch.exp(similarity_matrix)
        
        # Mask out self-similarity from exp_sim
        exp_sim = exp_sim * (1 - mask_diag)
        
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = (exp_sim * negative_mask).sum(dim=1)
        
        # Avoid division by zero
        denominator = positive_sum + negative_sum + 1e-8
        
        # Only compute loss for samples that have positives (other than themselves)
        has_positives = positive_mask.sum(dim=1) > 0
        
        if has_positives.sum() == 0:
             return torch.tensor(0.0, device=features.device)

        loss = -torch.log((positive_sum[has_positives] + 1e-8) / denominator[has_positives])
        return loss.mean()

class AdaptiveWeightScheduler:
    """自适应权重调度器"""
    def __init__(self, initial_weights, adaptation_rate=0.01):
        self.weights = initial_weights if isinstance(initial_weights, torch.Tensor) else torch.tensor(initial_weights)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def update_weights(self, class_accuracies):
        """根据各类别准确率动态调整权重"""
        # Ensure weights are on the same device as before or CPU
        device = self.weights.device
        
        for i, acc in enumerate(class_accuracies):
            if i >= len(self.weights):
                break
                
            if acc < 0.8:  # 如果某类别准确率低，增加权重
                self.weights[i] *= (1 + self.adaptation_rate)
            elif acc > 0.95:  # 如果某类别准确率过高，减少权重
                self.weights[i] *= (1 - self.adaptation_rate)
        
        # 归一化权重
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.weights = self.weights.to(device)
        return self.weights

class EnhancedTrainingManager:
    """增强训练管理器，负责整个训练过程，采用原始脚本的详细日志格式"""
    
    def __init__(self, model, config, train_loader, val_loader, optimizer, scheduler, device, output_dir):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 设置TensorBoard
        self.tb_writer = self._setup_tensorboard()
        
        # 损失函数配置
        self._setup_loss_config()
        
        # 早停配置
        self._setup_early_stopping()
        
        # 训练历史
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [], 'val_mcc': [], 'val_auprc': []
        }

        # 初始化dropout调度器
        self.dropout_scheduler = None
        if HAS_DROPOUT_SCHEDULER and self.config.get('adaptive_dropout', True):
            try:
                self.dropout_scheduler = DropoutScheduler(self.model, self.config)
                self.logger.info("已初始化自适应Dropout调度器")
            except Exception as e:
                self.logger.warning(f"初始化Dropout调度器失败: {e}")
                self.dropout_scheduler = None

        # 缓存标准Dropout层以支持阶段性正则调节
        self._dropout_layers = []
        self._dropout_base_rates = {}
        self._cache_dropout_layers()

        # NaN/Inf debugging observer
        self.nan_observer = TensorRangeObserver(
            model=self.model,
            logger=self.logger,
            output_dir=self.output_dir,
            config=self.config,
        )

        # Mixup与随机掩码配置
        self.use_mixup = bool(self.config.get('use_mixup', False))
        self.mixup_alpha = float(self.config.get('mixup_alpha', 0.2))
        self.mixup_prob = float(self.config.get('mixup_prob', 0.0))
        self.random_mask_prob = float(self.config.get('random_mask_prob', 0.0))
        self.random_mask_scale = float(self.config.get('random_mask_scale', 0.0))
        self.random_mask_apply_to_structure = bool(self.config.get('random_mask_apply_to_structure', True))
        self.random_mask_apply_to_local = bool(self.config.get('random_mask_apply_to_local', True))

        # 梯度累积与裁剪配置
        self.grad_accum_steps = max(1, int(self.config.get('grad_accum_steps', 1)))
        
        # 混合精度训练 (AMP) 配置
        self.use_amp = bool(self.config.get('use_amp', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("已启用自动混合精度 (AMP) 训练")

        gradient_clip = self.config.get('gradient_clip_val', None)
        try:
            self.gradient_clip_val = float(gradient_clip) if gradient_clip is not None else None
        except (TypeError, ValueError):
            self.gradient_clip_val = None
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            self.gradient_clip_val = None

        # 权重衰减warmup配置
        self.weight_decay_warmup_steps = int(self.config.get('weight_decay_warmup_steps', 0))
        self.weight_decay_warmup_init = float(self.config.get('weight_decay_warmup_init', 0.0))
        self._target_weight_decays = [group.get('weight_decay', self.config.get('weight_decay', 0.0)) for group in self.optimizer.param_groups]
        self._base_target_weight_decays = list(self._target_weight_decays)
        wd_final_cfg = self.config.get('weight_decay_warmup_final')
        self.weight_decay_warmup_final = float(wd_final_cfg) if wd_final_cfg is not None else None
        if self.weight_decay_warmup_final is None and self.config.get('weight_decay') is not None:
            self.weight_decay_warmup_final = float(self.config.get('weight_decay'))
        if self.weight_decay_warmup_final is None and self._target_weight_decays:
            self.weight_decay_warmup_final = float(self._target_weight_decays[0])
        if self.weight_decay_warmup_final is None:
            self.weight_decay_warmup_final = 0.0
        self._base_weight_decay_final = float(self.weight_decay_warmup_final)
        if self.weight_decay_warmup_steps > 0:
            self.logger.info(
                f"启用权重衰减Warmup: steps={self.weight_decay_warmup_steps}, "
                f"start={self.weight_decay_warmup_init}, target≈{self.weight_decay_warmup_final}"
            )
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx]
                if target == 0.0 and self.weight_decay_warmup_final:
                    target = self.weight_decay_warmup_final
                    self._target_weight_decays[idx] = target
                group['weight_decay'] = self.weight_decay_warmup_init

        # 调度器信息
        scheduler_config_name = self.config.get('scheduler_normalized', self.config.get('scheduler', 'cosine')) if self.config else 'cosine'
        scheduler_name = str(scheduler_config_name).lower()
        self.scheduler_type = SCHEDULER_ALIASES.get(scheduler_name, scheduler_name)
        if self.scheduler is None:
            self.scheduler_type = None
        self.scheduler_step_on_batch = self.scheduler_type in {'onecycle', 'warmup_cosine'}
        self.global_step = 0

        # 阈值调优约束
        self.threshold_min_precision = float(self.config.get('threshold_min_precision', 0.0)) if self.config else 0.0
        self.threshold_min_specificity = float(self.config.get('threshold_min_specificity', 0.0)) if self.config else 0.0
        self.threshold_min_recall = float(self.config.get('threshold_min_recall', 0.0)) if self.config else 0.0

        # 阶段式训练配置
        self.structure_warmup_epochs = int(self.config.get('structure_warmup_epochs', 0)) if self.config else 0
        full_epoch = self.config.get('full_finetune_epoch') if self.config else None
        try:
            self.full_finetune_epoch = int(full_epoch) if full_epoch is not None else None
        except (TypeError, ValueError):
            self.full_finetune_epoch = None
        self._current_stage = None
        
        # 自适应权重调度器
        self.adaptive_weight_scheduler = None
    
    def _setup_logger(self):
        """设置日志记录器，使用与原始脚本一致的格式"""
        logger = logging.getLogger("train_advanced")
        logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 文件处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_formatter = logging.Formatter('%(asctime)s - train_advanced - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')  # 简化控制台输出格式
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_tensorboard(self):
        """设置TensorBoard"""
        # 尝试导入TensorBoard
        if _try_import_tensorboard():
            tb_log_dir = os.path.join(self.output_dir, 'tensorboard', 'version_0')
            os.makedirs(tb_log_dir, exist_ok=True)
            return SummaryWriter(log_dir=tb_log_dir)
        else:
            self.logger.warning("TensorBoard不可用，跳过TensorBoard日志记录")
            return None
    
    def _setup_loss_config(self):
        """设置损失函数配置"""
        self.loss_config = {
            'use_focal_loss': self.config.get('use_focal_loss', True),
            'use_label_smoothing': self.config.get('use_label_smoothing', True),
            'loss_combination_weight': self.config.get('loss_combination_weight', 0.7),
            'focal_alpha': self.config.get('focal_alpha', 0.25),
            'focal_gamma': self.config.get('focal_gamma', 1.5),
            'label_smoothing': self.config.get('label_smoothing', 0.05),
            'use_class_weights': self.config.get('use_class_weights', True),
            'use_mcc_loss': self.config.get('use_mcc_loss', False),
            'mcc_loss_weight': self.config.get('mcc_loss_weight', 0.5),
            'use_contrastive_loss': self.config.get('use_contrastive_loss', False),
            'contrastive_loss_weight': self.config.get('contrastive_loss_weight', 0.1),
            'use_adaptive_weighting': self.config.get('use_adaptive_weighting', False)
        }

        # Initialize EnhancedLoss
        self.enhanced_loss = EnhancedLoss(
            alpha=self.loss_config['focal_alpha'], 
            gamma=self.loss_config['focal_gamma']
        )
        
        # Initialize AdaptiveWeightScheduler
        if self.loss_config['use_adaptive_weighting']:
            # Assume 2 classes for now, or get from config
            # Using simple [1.0, 1.0] initialization, logic will normalize it
            self.adaptive_weight_scheduler = AdaptiveWeightScheduler(
                initial_weights=[1.0, 1.0],
                adaptation_rate=self.config.get('adaptive_weight_rate', 0.01)
            )

        self.pos_weight = None
        if self.loss_config['use_class_weights']:
            pos_weight_value = self.config.get('pos_weight', 1.0)
            try:
                self.pos_weight = torch.tensor(float(pos_weight_value), dtype=torch.float32, device=self.device)
            except (TypeError, ValueError):
                self.logger.warning(f"pos_weight值 {pos_weight_value} 无效，忽略类别权重")
                self.pos_weight = None
        self.loss_config['pos_weight'] = self.pos_weight

        # 记录最终损失相关配置，便于排查传参问题
        pos_weight_display = None
        if isinstance(self.pos_weight, torch.Tensor):
            pos_weight_display = float(self.pos_weight.detach().cpu().item())
        elif self.pos_weight is not None:
            try:
                pos_weight_display = float(self.pos_weight)
            except (TypeError, ValueError):
                pos_weight_display = str(self.pos_weight)

        self.logger.info(
            "损失函数配置: "
            f"focal_loss={self.loss_config['use_focal_loss']} (alpha={self.loss_config['focal_alpha']}, gamma={self.loss_config['focal_gamma']}), "
            f"label_smoothing={self.loss_config['use_label_smoothing']} (ε={self.loss_config['label_smoothing']}), "
            f"mcc_loss={self.loss_config['use_mcc_loss']} (weight={self.loss_config['mcc_loss_weight']}), "
            f"loss_mix_weight={self.loss_config['loss_combination_weight']}, "
            f"pos_weight={pos_weight_display if pos_weight_display is not None else 'None'}"
        )
    
    def _setup_early_stopping(self):
        """设置早停配置"""
        self.early_stopping = {
            'use_early_stopping': self.config.get('use_early_stopping', True),  # 默认启用
            'patience': self.config.get('early_stopping_patience', 10),  # 增加耐心值
            'min_delta': self.config.get('early_stopping_min_delta', 0.001),
            'monitor_metric': self.config.get('early_stopping_monitor', 'val_loss'),  # 监控指标
            'mode': self.config.get('early_stopping_mode', 'min'),  # min表示越小越好
            'best_score': float('inf') if self.config.get('early_stopping_mode', 'min') == 'min' else float('-inf'),
            'counter': 0,
            'restore_best_weights': self.config.get('restore_best_weights', True),  # 是否恢复最佳权重
            'verbose': self.config.get('early_stopping_verbose', True)
        }
        self.monitor_metric = self.early_stopping['monitor_metric']
    
    def _ensure_micro_env_precomputed(self, loader, name):
        """确保数据集的微环境特征已预计算"""
        if not loader or not hasattr(loader, 'dataset'):
            return
            
        dataset = loader.dataset
        # Handle Subset if split was used
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
            
        if hasattr(dataset, 'protein_store') and hasattr(dataset.protein_store, 'precompute_micro_environment'):
            if hasattr(dataset, 'data_frame') and 'uniprot_id' in dataset.data_frame.columns:
                self.logger.info(f"正在检查 {name} 集的微环境特征完整性...")
                try:
                    pids = dataset.data_frame['uniprot_id'].unique().tolist()
                    # Trigger pre-computation (it handles skipping if already done/cached)
                    dataset.protein_store.precompute_micro_environment(pids)
                except Exception as e:
                    self.logger.warning(f"检查 {name} 集微环境特征时出错: {e}")

    def train(self):
        """执行完整训练流程"""
        self.logger.info("=== Start Enhanced Training ===")
        
        # Ensure micro-environment features are pre-computed
        self._ensure_micro_env_precomputed(self.train_loader, "Training")
        self._ensure_micro_env_precomputed(self.val_loader, "Validation")
        
        # 初始化最佳指标
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_mcc = 0.0
        best_val_auprc = 0.0
        best_epoch = 0
        
        early_stopping_counter = 0
        early_stopping_patience = self.early_stopping['patience']
        use_early_stopping = self.early_stopping['use_early_stopping']
        
        # 检查初始GPU状态
        if torch.cuda.is_available():
            self.logger.info("训练开始前的GPU状态:")
            self.logger.info(check_gpu_memory())
        
        self.logger.info(f"梯度累积步数: {self.grad_accum_steps}")
        if self.gradient_clip_val is not None:
            self.logger.info(f"梯度裁剪阈值: {self.gradient_clip_val}")
        else:
            self.logger.info("梯度裁剪: 未启用")
        if self.scheduler_type:
            self.logger.info(f"学习率调度器: {self.scheduler_type}")
        else:
            self.logger.info("学习率调度器: 未使用")

        num_epochs = self.config.get('epochs', 50)
        
        for epoch in range(num_epochs):
            # 输出epoch标题
            self.logger.info("="*80)
            self.logger.info(f"开始训练 Epoch {epoch+1}/{num_epochs}")
            self.logger.info("="*80)
            
            # 检查每个epoch开始时的GPU内存状态
            if torch.cuda.is_available():
                self.logger.info(f"Epoch {epoch+1} 开始时的GPU状态:")
                self.logger.info(check_gpu_memory())
            
            # 阶段训练调度
            self._update_training_stage(epoch)

            # 更新中心特征课程 (如果模型支持)
            if hasattr(self.model, 'update_center_curriculum'):
                self.model.update_center_curriculum(epoch)

            # 训练一个epoch
            train_metrics = self._train_epoch(epoch)

            # 验证
            val_metrics = self._validate_epoch(epoch)

            # 更新自适应权重
            if self.adaptive_weight_scheduler is not None:
                # 使用验证集的准确率（按类别）
                # 由于我们只有整体准确率，或者需要从混淆矩阵推导
                # 简化：使用 recall (positive acc) 和 specificity (negative acc)
                # val_metrics['val_recall'] -> positive class accuracy
                # val_metrics['val_acc'] -> overall accuracy
                # 如果有 specificity 更好，但 val_metrics 可能没有直接返回
                
                # 尝试从 val_metrics 获取 specificity
                if 'val_specificity' in val_metrics: # check if I added it or if it exists
                     neg_acc = val_metrics['val_specificity']
                else:
                     # 粗略估计: 如果 val_recall 很低，提升正类权重
                     # 如果 val_recall 很高，降低正类权重
                     # 但 adaptive_weight_scheduler.update_weights 期望 class_accuracies 列表
                     # [neg_acc, pos_acc]
                     
                     # 我们可以使用 recall 作为 pos_acc
                     pos_acc = val_metrics.get('val_recall', 0.5)
                     
                     # 对于 neg_acc，我们可以用 1 - FPR? 
                     # 或者简单地，如果没有 specificity，就只根据 recall 调整 pos_weight
                     neg_acc = 0.8 # 假定负类通常较好
                     
                class_accuracies = [neg_acc, val_metrics.get('val_recall', 0.5)]
                
                new_weights = self.adaptive_weight_scheduler.update_weights(class_accuracies)
                
                # 更新 loss_config 中的 pos_weight (如果是 BCEWithLogitsLoss 使用的)
                # 注意：AdaptiveWeightScheduler 修改的是 self.adaptive_weight_scheduler.weights
                # 我们在 _compute_basic_loss 中已经直接使用了 self.adaptive_weight_scheduler.weights
                # 所以这里不需要显式更新 self.pos_weight，除非用于日志记录
                
                self.logger.info(f"Epoch {epoch+1}: 自适应权重更新为 {new_weights.tolist()}")

            # 更新自适应dropout调度
            if self.dropout_scheduler is not None:
                self.dropout_scheduler.step(epoch, val_metrics['val_loss'])

                # 每10个epoch记录dropout信息
                if epoch % 10 == 0:
                    dropout_rates = self.dropout_scheduler.get_all_dropout_rates()
                    if dropout_rates:
                        self.logger.info("当前Dropout率:")
                        for layer_name, rate in dropout_rates.items():
                            self.logger.info(f"  {layer_name}: {rate:.4f}")

            # 如果模型有update_dropout_schedule方法，也调用它
            if hasattr(self.model, 'update_dropout_schedule'):
                self.model.update_dropout_schedule(epoch, val_metrics['val_loss'])

            # 更新训练历史
            self._update_history(train_metrics, val_metrics)
            
            # 记录到TensorBoard
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            # 按epoch更新调度器
            self._step_scheduler_epoch(epoch, val_metrics)
            
            # 检查最佳模型和早停
            improvement_flag = ""
            should_stop, improvement_flag = self._check_early_stopping(
                val_metrics, epoch, best_val_loss, best_val_acc, best_val_f1,
                best_val_mcc, best_val_auprc, best_epoch
            )

            # 更新最佳指标（如果有改进）
            if improvement_flag.find("新的最佳模型") != -1:
                best_val_loss = val_metrics['val_loss']
                best_val_acc = val_metrics['val_acc']
                best_val_f1 = val_metrics.get('val_f1', 0.0)
                best_val_mcc = val_metrics.get('val_mcc', 0.0)
                best_val_auprc = val_metrics.get('val_auprc', 0.0)
                best_epoch = epoch

            # 检查是否需要早停
            if should_stop:
                self.logger.info("="*80)
                self.logger.info(f"早停条件满足，在第 {epoch+1} 轮停止训练")
                self.logger.info(f"最佳模型来自第 {best_epoch+1} 轮")
                self.logger.info("="*80)
                break
            
            # 输出epoch总结
            self.logger.info("="*60)
            self.logger.info(f"Epoch {epoch+1} 总结{improvement_flag}")
            self.logger.info(f"训练: Loss={train_metrics.get('loss', 0):.4f}, Acc={train_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"验证: Loss={val_metrics['val_loss']:.4f}, Acc={val_metrics['val_acc']:.4f}, "
                           f"F1={val_metrics.get('val_f1', 0):.4f}")
            self.logger.info(f"当前最佳: Epoch {best_epoch+1}, Loss={best_val_loss:.4f}, "
                           f"Acc={best_val_acc:.4f}, F1={best_val_f1:.4f}")
            self.logger.info("="*60)
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        
        # 关闭TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        # 返回最佳指标
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'best_val_mcc': best_val_mcc,
            'best_val_auprc': best_val_auprc,
            'best_epoch': best_epoch,
            'history': self.training_history
        }
    
    def _update_training_stage(self, epoch: int) -> None:
        if not hasattr(self.model, 'set_training_stage'):
            return

        if self.structure_warmup_epochs > 0 and epoch < self.structure_warmup_epochs:
            stage = 'structure_warmup'
        elif self.full_finetune_epoch is not None and epoch >= self.full_finetune_epoch:
            stage = 'full_finetune'
        else:
            stage = 'joint'

        if stage == self._current_stage:
            return

        stage_changed = False
        if hasattr(self.model, 'set_training_stage'):
            try:
                self.model.set_training_stage(stage)
                self.logger.info(f"切换训练阶段 -> {stage}")
                stage_changed = True
            except Exception as exc:
                self.logger.warning(f"设置训练阶段 {stage} 失败: {exc}")

        self._apply_stage_regularization(stage)
        if not stage_changed:
            self.logger.info(f"应用阶段性正则调整 -> {stage}")
        self._current_stage = stage

    def _train_epoch(self, epoch):
        """训练一个epoch - 详细日志版本"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        
        # 记录开始时间
        epoch_start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        observer = getattr(self, 'nan_observer', None)
        observer_enabled = bool(observer and getattr(observer, 'enabled', False))

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 确保数据在正确设备上
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(self.device)

            # 训练阶段正则化（随机掩码、Mixup等）
            self._apply_batch_regularization(batch)
            
            # 注入当前epoch信息用于中心特征课程学习
            batch['center_curriculum_epoch'] = epoch + (batch_idx / max(1, len(self.train_loader)))
            
            if observer_enabled:
                observer.inspect_batch(
                    batch,
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                )

            context_pushed = False
            if observer_enabled:
                observer.push_context(
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                )
                context_pushed = True

            # 前向传播
            try:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    
                    # 提取辅助损失 (如果存在)
                    batch_aux_loss = 0.0
                    aux_loss_dict = {}
                    if isinstance(outputs, dict) and 'auxiliary_losses' in outputs:
                        for k, v in outputs['auxiliary_losses'].items():
                            if isinstance(v, torch.Tensor):
                                batch_aux_loss += v
                                aux_loss_dict[k] = v.item()
                            else:
                                batch_aux_loss += v
                                aux_loss_dict[k] = v

                    labels = batch['label']

                    # 处理输出格式
                    importance_scores = None
                    is_multitask = False
                    features = None

                    if isinstance(outputs, tuple):
                        # 检查是否是多任务输出 (acetylation_transformer)
                        # 多任务输出: (class_logits, strength_logits)
                        # 可解释性输出: (predictions, importance_scores)
                        if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                            # 检查是否是多任务输出（第一个元素是2D张量，第二个是1D或2D张量）
                            if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                                # 这是多任务输出
                                is_multitask = True
                                class_logits, strength_logits = outputs
                                logits = class_logits  # 使用分类logits作为主要输出
                            else:
                                # 这是可解释性输出
                                logits = outputs[0]
                                importance_scores = outputs[1] if len(outputs) > 1 else None
                        elif len(outputs) == 3:
                             # 可能包含特征 (logits, features, other) 或者 (logits, strength, aux)
                             logits = outputs[0]
                             if isinstance(outputs[1], torch.Tensor) and outputs[1].dim() > 1:
                                 features = outputs[1] # Assume second element is features if it looks like embeddings
                        else:
                            # 其他元组格式
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    elif isinstance(outputs, dict):
                        if 'logits' in outputs:
                            logits = outputs['logits']
                        elif 'predicted_outcome' in outputs:
                            logits = outputs['predicted_outcome']
                        else:
                            logits = outputs[list(outputs.keys())[0]]
                        
                        # 尝试提取特征
                        if 'features' in outputs:
                            features = outputs['features']
                        elif 'embedding' in outputs:
                            features = outputs['embedding']
                        elif 'projection' in outputs:
                            features = outputs['projection']
                    else:
                        logits = outputs

                    # 确保张量维度匹配
                    if logits.dim() > 1 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    if labels.dim() > 1 and labels.size(-1) == 1:
                        labels = labels.squeeze(-1)

                    # 处理多任务输出的特殊情况
                    if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                        # 对于多任务输出，可能需要特殊的损失计算
                        # 但如果使用了自定义的 MultiTaskLoss (如在 AcetylationModel 中)，
                        # 我们应该调用它并传入所有输出。
                        # 这里的 self._compute_loss 通常只是 CrossEntropyLoss 或 FocalLoss。
                        # 如果我们想要利用完整的 MultiTaskLoss，我们需要一种方式来传递额外参数。
                        
                        # HACK: 如果是 MoE 输出 (len(outputs)==3)，我们尝试构造一个包含所有信息的 loss
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                             # outputs: (logits, strength, aux_outputs)
                             # self.criterion 可能是 MultiTaskLoss
                             # 我们尝试调用 self.criterion(logits, strength, labels, strength_targets, aux_outputs)
                             # 但是 self._compute_loss 接口只接受 (logits, labels)
                             
                             # 因此，我们需要在这里直接计算 loss，绕过 self._compute_loss
                             # 或者修改 _compute_loss。
                             # 最简单的方法是检查 self.criterion 是否是 MultiTaskLoss 的实例
                             
                             # 假设 criterion 支持 aux_outputs (我们在 acetylation_model.py 中修改了 MultiTaskLoss)
                             # 但是 training_manager 中 self.criterion 可能是被封装的。
                             
                             # 让我们尝试手动计算
                             pass
                             
                        # 将标签转换为长整型
                        labels_long = labels.long()
                        # loss = F.cross_entropy(logits, labels_long) # OLD logic
                        
                        # NEW LOGIC: Try to use the model's loss if available or construct it
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            logits, strength_pred, aux_outputs = outputs
                            # 假设 strengths 在 batch 中
                            strengths = batch.get('strengths', torch.zeros_like(labels))
                            
                            # 检查 criterion 是否支持这种调用
                            # 由于 Python 的动态特性，我们可以尝试
                            try:
                                loss_dict = self.criterion(logits, strength_pred, labels, strengths, aux_outputs)
                                loss = loss_dict['loss']
                                batch_aux_loss = loss_dict.get('loss_aux', 0.0)
                            except TypeError:
                                # Fallback
                                loss = F.cross_entropy(logits, labels_long)
                        else:
                             loss = F.cross_entropy(logits, labels_long)

                    else:
                        # 检查是否需要转换概率值回logits（更安全的检查）
                        if torch.all((logits >= 0) & (logits <= 1)) and torch.all(logits != 0) and torch.all(logits != 1):
                            # 只有当所有值都在(0,1)范围内且不是边界值时才转换
                            # 使用更稳定的logit转换
                            eps = 1e-7
                            logits = torch.clamp(logits, eps, 1-eps)
                            logits = torch.log(logits / (1 - logits))

                        # 计算损失
                        loss = self._compute_loss(logits, labels)
                        
                        # 添加辅助损失
                        if isinstance(batch_aux_loss, torch.Tensor) or batch_aux_loss > 0:
                            loss = loss + batch_aux_loss
            finally:
                if observer_enabled and context_pushed:
                    observer.pop_context()

            # [Fix] 检查损失是否为NaN
            if torch.isnan(loss):
                self.logger.error(f"严重错误: Loss为NaN (Epoch {epoch}, Batch {batch_idx})")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = float(loss.detach().item())
            total_loss += loss_value

            # 反向传播与梯度累积 (使用GradScaler)
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if observer_enabled and isinstance(logits, torch.Tensor):
                observer.observe_tensor(
                    'train_logits',
                    logits.detach(), # 显式detach
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    reason='logits',
                )

            # 处理多任务输出的统计
            if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                if observer_enabled:
                    observer.observe_scalar(
                        'train_loss',
                        loss.detach(),
                        phase='train',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                # 统计 - 对于多类分类
                probs = torch.softmax(logits.detach(), dim=-1)
                preds = torch.argmax(probs, dim=-1)
                labels_binary = labels.long()
                preds_binary = preds
                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

                # 收集预测和标签用于详细统计
                all_train_probs.extend(probs[:, 1].cpu().numpy())  # 使用正类概率
                all_train_preds.extend(preds_binary.cpu().numpy())
                all_train_labels.extend(labels_binary.cpu().numpy())
            else:
                if observer_enabled:
                    observer.observe_scalar(
                        'train_loss',
                        loss.detach(),
                        phase='train',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                # 统计
                # [Fix] Handle NaN in logits before sigmoid
                if torch.isnan(logits).any():
                     self.logger.warning("NaN detected in logits during training statistics. Replacing with zeros.")
                     logits_for_stats = torch.nan_to_num(logits.detach(), nan=0.0)
                else:
                     logits_for_stats = logits.detach()

                probs = torch.sigmoid(logits_for_stats)
                
                # [Fix] Handle NaN in probs
                if torch.isnan(probs).any():
                     self.logger.warning("NaN detected in probabilities. Replacing with 0.")
                     probs = torch.nan_to_num(probs, nan=0.0)

                preds = probs > 0.5
                labels_binary = (labels > 0.5).long()
                preds_binary = preds.long()
                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

                # 收集预测和标签用于详细统计
                all_train_probs.extend(probs.cpu().numpy())
                all_train_preds.extend(preds_binary.cpu().numpy())
                all_train_labels.extend(labels_binary.cpu().numpy())

            if observer_enabled:
                observer.observe_tensor(
                    'train_probs',
                    probs,
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    reason='probabilities',
                )

            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))
            if should_step:
                # [Fix] Check for NaN/Inf in gradients before update to prevent weight corruption
                # Use scaler to check for invalid gradients if AMP is enabled
                grad_invalid = False
                
                # Unscale gradients before clipping and checking for NaNs
                self.scaler.unscale_(self.optimizer)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_invalid = True
                            self.logger.warning(f"Invalid gradients (NaN/Inf) detected in {name}. Skipping optimization step to protect model weights.")
                            break
                
                if grad_invalid:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update() # Still need to update scaler
                    continue

                if self.gradient_clip_val is not None:
                    try:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
                    except RuntimeError as clip_error:
                        self.logger.warning(f"梯度裁剪失败: {clip_error}")
                
                # scaler.step() first unscales gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)
                
                # Updates the scale for next iteration.
                self.scaler.update()
                
                self.global_step += 1
                self._update_weight_decay()
                if self.scheduler_step_on_batch and self.scheduler is not None:
                    self._step_scheduler_batch()
                self.optimizer.zero_grad(set_to_none=True)

                if observer_enabled:
                    observer.update_global_step(self.global_step)
            
            # 计算批次处理时间
            batch_time = time.time() - batch_start_time
            
            # 每100个批次记录一次详细信息
            if batch_idx % 100 == 0:
                current_lr = self._get_current_lr()
                self.logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, '
                               f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}, '
                               f'Batch Time: {batch_time:.3f}s')
            
            # 每500个批次记录GPU内存状态
            if batch_idx % 500 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f'GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB')
        
        # 计算epoch总时间
        epoch_time = time.time() - epoch_start_time
        
        # 计算训练集详细统计
        all_train_probs = np.array(all_train_probs)
        all_train_labels = np.array(all_train_labels)
        train_binary_preds = (all_train_probs > 0.5).astype(int)
        
        # 计算训练集指标
        train_precision = precision_score(all_train_labels, train_binary_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, train_binary_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, train_binary_preds, zero_division=0)
        train_mcc = matthews_corrcoef(all_train_labels, train_binary_preds)
        
        # 计算AUC
        if len(np.unique(all_train_labels)) > 1:
            # [Fix] Check for NaNs in all_train_probs
            if np.isnan(all_train_probs).any():
                self.logger.warning("NaN detected in all_train_probs before AUC calculation. Replacing with 0.")
                all_train_probs = np.nan_to_num(all_train_probs, nan=0.0)
            
            try:
                train_auc = roc_auc_score(all_train_labels, all_train_probs)
            except ValueError as e:
                self.logger.error(f"Error calculating AUC: {e}")
                train_auc = 0.5
        else:
            train_auc = 0.5
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        
        if total > 0:
            accuracy = correct / total
        else:
            self.logger.warning(f"Epoch {epoch+1}: No samples processed successfully (total=0). This usually indicates NaN loss in all batches.")
            accuracy = 0.0
        
        # 输出详细的训练统计信息
        self.logger.info("="*60)
        self.logger.info(f"Epoch {epoch+1} 训练完成 (耗时: {epoch_time:.2f}s)")
        self.logger.info("训练集统计:")
        
        # 标签和预测分布
        unique_labels, label_counts = np.unique(all_train_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(train_binary_preds, return_counts=True)
        self.logger.info(f"  标签分布: {dict(zip(unique_labels.astype(int), label_counts))}")
        self.logger.info(f"  预测分布: {dict(zip(unique_preds.astype(int), pred_counts))}")
        
        # 概率值统计
        self.logger.info(f"  概率值范围: [{all_train_probs.min():.4f}, {all_train_probs.max():.4f}]")
        self.logger.info(f"  概率值均值: {all_train_probs.mean():.4f}")
        self.logger.info(f"  概率值标准差: {all_train_probs.std():.4f}")

        # 置信度评估指标
        high_confidence_count = np.sum(all_train_probs >= 0.8)
        medium_confidence_count = np.sum((all_train_probs >= 0.5) & (all_train_probs < 0.8))
        low_confidence_count = np.sum(all_train_probs < 0.5)
        total_samples = len(all_train_probs)

        self.logger.info("置信度分布:")
        self.logger.info(f"  高置信度 (≥0.8): {high_confidence_count}/{total_samples} ({high_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  中等置信度 (0.5-0.8): {medium_confidence_count}/{total_samples} ({medium_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  低置信度 (<0.5): {low_confidence_count}/{total_samples} ({low_confidence_count/total_samples*100:.1f}%)")

        # 详细指标
        self.logger.info("训练结果详情:")
        self.logger.info(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        self.logger.info(f"  F1: {train_f1:.4f}, MCC: {train_mcc:.4f}")
        self.logger.info(f"  AUC: {train_auc:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'mcc': train_mcc,
            'auc': train_auc
        }
    
    def _apply_batch_regularization(self, batch):
        """按需对训练批次施加额外正则化"""
        if self.random_mask_prob > 0:
            self._apply_random_mask(batch)

        if self.use_mixup and self.mixup_prob > 0 and batch.get('sequence_features') is not None:
            if batch['sequence_features'].size(0) > 1:
                trigger = float(torch.rand(1, device=self.device).item())
                if trigger < self.mixup_prob:
                    self._apply_mixup(batch)

    def _cache_dropout_layers(self):
        """缓存模型中的标准Dropout层以便阶段性调节"""
        try:
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                    if module not in self._dropout_layers and hasattr(module, 'p'):
                        self._dropout_layers.append(module)
                        self._dropout_base_rates[module] = float(module.p)
            if self._dropout_layers:
                self.logger.info(f"检测到 {len(self._dropout_layers)} 个Dropout层用于阶段性调节")
        except Exception as exc:
            self.logger.warning(f"缓存Dropout层失败: {exc}")

    def _apply_stage_regularization(self, stage: str) -> None:
        """根据训练阶段调整Dropout与权重衰减"""
        if not isinstance(stage, str):
            return

        stage_key = stage.lower().strip()
        dropout_multiplier = float(self.config.get(f"{stage_key}_dropout_multiplier", 1.0) or 1.0)
        weight_decay_multiplier = float(self.config.get(f"{stage_key}_weight_decay_multiplier", 1.0) or 1.0)

        self._set_dropout_multiplier(dropout_multiplier)
        self._set_weight_decay_multiplier(weight_decay_multiplier)

        self.logger.info(
            "阶段正则化: stage=%s, dropout_multiplier=%.3f, weight_decay_multiplier=%.3f",
            stage_key,
            dropout_multiplier,
            weight_decay_multiplier,
        )

    def _set_dropout_multiplier(self, multiplier: float) -> None:
        if not self._dropout_layers:
            return

        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0

        multiplier = max(0.0, multiplier)
        for module in self._dropout_layers:
            base_rate = self._dropout_base_rates.get(module, getattr(module, 'p', 0.0))
            new_rate = max(0.0, min(0.95, base_rate * multiplier))
            module.p = new_rate

    def _set_weight_decay_multiplier(self, multiplier: float) -> None:
        if not self.optimizer or not self.optimizer.param_groups:
            return

        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0

        multiplier = max(0.0, multiplier)

        if not hasattr(self, '_base_target_weight_decays'):
            self._base_target_weight_decays = [group.get('weight_decay', 0.0) for group in self.optimizer.param_groups]

        for idx, group in enumerate(self.optimizer.param_groups):
            base_value = self._base_target_weight_decays[idx] if idx < len(self._base_target_weight_decays) else self.config.get('weight_decay', 0.0)
            target = float(base_value) * multiplier
            if idx < len(self._target_weight_decays):
                self._target_weight_decays[idx] = target
            else:
                self._target_weight_decays.append(target)

        if hasattr(self, '_base_weight_decay_final'):
            self.weight_decay_warmup_final = self._base_weight_decay_final * multiplier

        if self.weight_decay_warmup_steps > 0:
            ratio = 0.0
            if self.weight_decay_warmup_steps > 0:
                ratio = min(1.0, max(0.0, self.global_step / float(self.weight_decay_warmup_steps)))
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else self.weight_decay_warmup_final
                adjusted = self.weight_decay_warmup_init + (target - self.weight_decay_warmup_init) * ratio
                group['weight_decay'] = adjusted
        else:
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else group.get('weight_decay', 0.0)
                group['weight_decay'] = target

    def _apply_random_mask(self, batch):
        """对输入特征施加随机掩码"""
        seq_features = batch.get('sequence_features')
        if not isinstance(seq_features, torch.Tensor):
            return

        if self.random_mask_prob <= 0:
            return

        batch_size, window_len = seq_features.shape[0], seq_features.shape[1]
        mask = torch.rand(batch_size, window_len, device=seq_features.device) < self.random_mask_prob
        if mask.any():
            seq_features = seq_features.masked_fill(mask.unsqueeze(-1), 0.0)
            if self.random_mask_scale > 0:
                noise = torch.randn_like(seq_features) * self.random_mask_scale
                seq_features = seq_features + noise * mask.unsqueeze(-1).float()
            batch['sequence_features'] = seq_features

        if self.random_mask_apply_to_local and isinstance(batch.get('local_features'), torch.Tensor):
            local_features = batch['local_features']
            local_mask = torch.rand(local_features.shape[0], local_features.shape[1], device=local_features.device) < self.random_mask_prob
            if local_mask.any():
                batch['local_features'] = local_features.masked_fill(local_mask.unsqueeze(-1), 0.0)

        if isinstance(batch.get('global_features'), torch.Tensor):
            global_features = batch['global_features']
            global_mask = (torch.rand(global_features.shape[0], 1, device=global_features.device) < self.random_mask_prob).float()
            if global_mask.any():
                batch['global_features'] = global_features * (1 - global_mask)

        if self.random_mask_apply_to_structure and isinstance(batch.get('structure_features'), torch.Tensor):
            structure_features = batch['structure_features']
            if structure_features.shape[:2] == mask.shape:
                structure_features = structure_features.masked_fill(mask.unsqueeze(-1), 0.0)
            else:
                struct_mask = torch.rand(structure_features.shape[0], structure_features.shape[1], device=structure_features.device) < self.random_mask_prob
                structure_features = structure_features.masked_fill(struct_mask.unsqueeze(-1), 0.0)
            batch['structure_features'] = structure_features

    def _apply_mixup(self, batch):
        """在批次上应用Mixup增强"""
        seq_features = batch.get('sequence_features')
        if not isinstance(seq_features, torch.Tensor) or seq_features.size(0) < 2:
            return

        lam = self._sample_mixup_lambda()
        perm = torch.randperm(seq_features.size(0), device=seq_features.device)

        batch['sequence_features'] = lam * seq_features + (1 - lam) * seq_features[perm]

        if isinstance(batch.get('local_features'), torch.Tensor):
            local_features = batch['local_features']
            batch['local_features'] = lam * local_features + (1 - lam) * local_features[perm]

        if isinstance(batch.get('global_features'), torch.Tensor):
            global_features = batch['global_features']
            batch['global_features'] = lam * global_features + (1 - lam) * global_features[perm]

        if isinstance(batch.get('structure_features'), torch.Tensor):
            structure_features = batch['structure_features']
            batch['structure_features'] = lam * structure_features + (1 - lam) * structure_features[perm]

        if isinstance(batch.get('structure_mask'), torch.Tensor):
            structure_mask = batch['structure_mask']
            batch['structure_mask'] = lam * structure_mask + (1 - lam) * structure_mask[perm]

        labels = batch.get('label')
        if isinstance(labels, torch.Tensor):
            batch['label'] = lam * labels + (1 - lam) * labels[perm]

    def _sample_mixup_lambda(self):
        alpha = max(self.mixup_alpha, 1e-3)
        lam = float(np.random.beta(alpha, alpha))
        return max(lam, 1.0 - lam)

    def _update_weight_decay(self):
        """根据warmup策略更新权重衰减"""
        if self.weight_decay_warmup_steps <= 0:
            return

        step = min(self.global_step, self.weight_decay_warmup_steps)
        if step <= 0:
            return

        ratio = step / float(self.weight_decay_warmup_steps)
        for idx, group in enumerate(self.optimizer.param_groups):
            target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else self.weight_decay_warmup_final
            group['weight_decay'] = self.weight_decay_warmup_init + (target - self.weight_decay_warmup_init) * ratio

    def _validate_epoch(self, epoch):
        """验证一个epoch - 详细日志版本"""
        self.model.eval()
        total_loss = 0
        total_loss_with_aux = 0  # 用于记录包含辅助损失的总损失，以便与训练Loss对比
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_aux_seq_probs = []
        all_aux_struct_probs = []
        
        val_start_time = time.time()
        
        observer = getattr(self, 'nan_observer', None)
        observer_enabled = bool(observer and getattr(observer, 'enabled', False))

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 确保数据在正确设备上
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                if observer_enabled:
                    observer.inspect_batch(
                        batch,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                context_pushed = False
                if observer_enabled:
                    observer.push_context(
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )
                    context_pushed = True

                try:
                    outputs = self.model(batch)
                finally:
                    if observer_enabled and context_pushed:
                        observer.pop_context()
                
                # 计算辅助损失总和 (用于Total Loss计算)
                batch_aux_loss_sum = 0.0
                if isinstance(outputs, dict) and 'auxiliary_losses' in outputs:
                    for v in outputs['auxiliary_losses'].values():
                        if isinstance(v, torch.Tensor):
                            batch_aux_loss_sum += v.item()
                        else:
                            batch_aux_loss_sum += v

                # 收集辅助分支概率
                if isinstance(outputs, dict):
                    if 'aux_seq_logits' in outputs and outputs['aux_seq_logits'] is not None:
                        aux_seq = outputs['aux_seq_logits']
                        if aux_seq.dim() > 1 and aux_seq.size(-1) == 1:
                            aux_seq = aux_seq.squeeze(-1)
                        # 检查是否需要转换概率
                        if torch.all((aux_seq >= 0) & (aux_seq <= 1)) and torch.all(aux_seq != 0) and torch.all(aux_seq != 1):
                             # 已经是概率了? 不太可能，通常是logits
                             pass
                        aux_seq_probs = torch.sigmoid(aux_seq).cpu().numpy()
                        all_aux_seq_probs.extend(aux_seq_probs)
                    
                    if 'aux_struct_logits' in outputs and outputs['aux_struct_logits'] is not None:
                        aux_struct = outputs['aux_struct_logits']
                        if aux_struct.dim() > 1 and aux_struct.size(-1) == 1:
                            aux_struct = aux_struct.squeeze(-1)
                        aux_struct_probs = torch.sigmoid(aux_struct).cpu().numpy()
                        all_aux_struct_probs.extend(aux_struct_probs)

                labels = batch['label']

                # 处理输出格式
                importance_scores = None
                is_multitask = False

                if isinstance(outputs, tuple):
                    # 检查是否是多任务输出 (acetylation_transformer)
                    if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                        # 检查是否是多任务输出
                        if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                            # 这是多任务输出
                            is_multitask = True
                            class_logits, strength_logits = outputs
                            logits = class_logits  # 使用分类logits作为主要输出
                        else:
                            # 这是可解释性输出
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    else:
                        # 其他元组格式
                        logits = outputs[0]
                        importance_scores = outputs[1] if len(outputs) > 1 else None
                elif isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'predicted_outcome' in outputs:
                        logits = outputs['predicted_outcome']
                    else:
                        logits = outputs[list(outputs.keys())[0]]
                else:
                    logits = outputs

                # 确保张量维度匹配
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                if observer_enabled and isinstance(logits, torch.Tensor):
                    observer.observe_tensor(
                        'val_logits',
                        logits,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        reason='logits',
                    )

                # 处理多任务输出的特殊情况
                if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                    # 对于多任务输出，使用CrossEntropyLoss
                    labels_long = labels.long()
                    loss = F.cross_entropy(logits, labels_long)
                    total_loss += loss.item()
                    total_loss_with_aux += (loss.item() + batch_aux_loss_sum)

                    if observer_enabled:
                        observer.observe_scalar(
                            'val_loss',
                            loss,
                            phase='val',
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                        )

                    # 获取预测概率
                    probs = torch.softmax(logits, dim=-1)
                    labels_binary = labels_long
                else:
                    # 检查是否需要转换概率为logits，以匹配训练阶段的处理方式
                    if torch.all((logits >= 0) & (logits <= 1)) and torch.all(logits != 0) and torch.all(logits != 1):
                        eps = 1e-7
                        logits = torch.clamp(logits, eps, 1 - eps)
                        logits = torch.log(logits / (1 - logits))

                    # 计算损失
                    loss = self._compute_loss(logits, labels)
                    total_loss += loss.item()
                    total_loss_with_aux += (loss.item() + batch_aux_loss_sum)

                    if observer_enabled:
                        observer.observe_scalar(
                            'val_loss',
                            loss,
                            phase='val',
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                        )

                    # 获取预测概率
                    # [Fix] Handle NaNs in validation logits
                    if torch.isnan(logits).any():
                        logits = torch.nan_to_num(logits, nan=0.0)

                    probs = torch.sigmoid(logits)
                    
                    # [Fix] Handle NaNs in validation probs
                    if torch.isnan(probs).any():
                        probs = torch.nan_to_num(probs, nan=0.0)

                    labels_binary = (labels > 0.5).long()
                
                if observer_enabled:
                    observer.observe_tensor(
                        'val_probs',
                        probs,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        reason='probabilities',
                    )

                if is_multitask:
                    preds_binary = torch.argmax(probs, dim=-1)
                else:
                    preds_binary = (probs > 0.5).long()

                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

                all_preds.extend(preds_binary.cpu().numpy())
                all_labels.extend(labels_binary.cpu().numpy())
                if is_multitask:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # 使用正类概率
                else:
                    all_probs.extend(probs.cpu().numpy())
        
        val_time = time.time() - val_start_time
        
        # 计算额外指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 重新计算最优阈值用于最终统计
        best_threshold = 0.5
        best_score = -np.inf
        metric_name = str(self.config.get('threshold_metric', 'f1')).lower()

        for threshold in np.arange(0.1, 0.9, 0.05):
            temp_binary_preds = (all_probs > threshold).astype(int)
            if len(np.unique(temp_binary_preds)) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(all_labels, temp_binary_preds, labels=[0, 1]).ravel()
            temp_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            temp_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            temp_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            if temp_precision < self.threshold_min_precision:
                continue
            if temp_recall < self.threshold_min_recall:
                continue
            if temp_specificity < self.threshold_min_specificity:
                continue
            if metric_name == 'mcc':
                temp_score = matthews_corrcoef(all_labels, temp_binary_preds)
            elif metric_name in {'balanced_accuracy', 'balanced_acc', 'balanced'}:
                temp_score = balanced_accuracy_score(all_labels, temp_binary_preds)
            elif metric_name in {'recall', 'sensitivity', 'tpr'}:
                temp_score = recall_score(all_labels, temp_binary_preds, zero_division=0)
            elif metric_name in {'precision', 'ppv'}:
                temp_score = precision_score(all_labels, temp_binary_preds, zero_division=0)
            elif metric_name in {'specificity', 'tnr'}:
                tn, fp, fn, tp = confusion_matrix(all_labels, temp_binary_preds, labels=[0, 1]).ravel()
                temp_score = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            elif metric_name.startswith('f') and metric_name[1:].replace('.', '', 1).isdigit():
                try:
                    beta = float(metric_name[1:])
                    if beta <= 0:
                        raise ValueError
                except ValueError:
                    beta = 1.0
                temp_score = fbeta_score(all_labels, temp_binary_preds, beta=beta, zero_division=0)
            else:
                temp_score = f1_score(all_labels, temp_binary_preds, zero_division=0)

            if temp_score > best_score:
                best_score = temp_score
                best_threshold = threshold

        if best_score == -np.inf:
            self.logger.warning(
                "阈值调优: 未找到满足precision/recall/specificity约束的阈值，回退至0.5"
            )
            best_threshold = 0.5
        
        # 使用最优阈值重新计算预测
        binary_preds = (all_probs > best_threshold).astype(int)
        
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        mcc = matthews_corrcoef(all_labels, binary_preds)
        
        if len(np.unique(all_labels)) > 1:
            # [Fix] Handle NaNs in validation all_probs
            if np.isnan(all_probs).any():
                self.logger.warning("NaN detected in validation all_probs. Replacing with 0.")
                all_probs = np.nan_to_num(all_probs, nan=0.0)

            try:
                auc_score = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc_score = 0.5
            
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                auprc = auc(recall_curve, precision_curve)
            except ValueError:
                auprc = 0.0
        else:
            auc_score = 0.5
            auprc = 0.0
        
        # 计算辅助分支指标
        val_seq_mcc = 0.0
        val_struct_mcc = 0.0
        
        self.logger.info("--- 验证集分支独立评估 ---")
        
        if all_aux_seq_probs:
            all_aux_seq_probs = np.array(all_aux_seq_probs)
            
            # [Fix] Handle NaNs in auxiliary sequence probs
            if np.isnan(all_aux_seq_probs).any():
                self.logger.warning("NaN detected in validation all_aux_seq_probs. Replacing with 0.")
                all_aux_seq_probs = np.nan_to_num(all_aux_seq_probs, nan=0.0)

            seq_preds = (all_aux_seq_probs > 0.5).astype(int)
            
            val_seq_acc = accuracy_score(all_labels, seq_preds) if 'accuracy_score' in globals() else (seq_preds == all_labels).mean()
            val_seq_mcc = matthews_corrcoef(all_labels, seq_preds)
            val_seq_f1 = f1_score(all_labels, seq_preds, zero_division=0)
            
            if len(np.unique(all_labels)) > 1:
                val_seq_auc = roc_auc_score(all_labels, all_aux_seq_probs)
            else:
                val_seq_auc = 0.5
                
            self.logger.info(f"  序列分支 (Sequence Branch):")
            self.logger.info(f"    MCC: {val_seq_mcc:.4f}")
            self.logger.info(f"    ACC: {val_seq_acc:.4f}")
            self.logger.info(f"    F1 : {val_seq_f1:.4f}")
            self.logger.info(f"    AUC: {val_seq_auc:.4f}")
            
        if all_aux_struct_probs:
            all_aux_struct_probs = np.array(all_aux_struct_probs)
            
            # [Fix] Handle NaNs in auxiliary structure probs
            if np.isnan(all_aux_struct_probs).any():
                self.logger.warning("NaN detected in validation all_aux_struct_probs. Replacing with 0.")
                all_aux_struct_probs = np.nan_to_num(all_aux_struct_probs, nan=0.0)
                
            struct_preds = (all_aux_struct_probs > 0.5).astype(int)
            
            val_struct_acc = accuracy_score(all_labels, struct_preds) if 'accuracy_score' in globals() else (struct_preds == all_labels).mean()
            val_struct_mcc = matthews_corrcoef(all_labels, struct_preds)
            val_struct_f1 = f1_score(all_labels, struct_preds, zero_division=0)
            
            if len(np.unique(all_labels)) > 1:
                val_struct_auc = roc_auc_score(all_labels, all_aux_struct_probs)
            else:
                val_struct_auc = 0.5

            self.logger.info(f"  结构分支 (Structure Branch):")
            self.logger.info(f"    MCC: {val_struct_mcc:.4f}")
            self.logger.info(f"    ACC: {val_struct_acc:.4f}")
            self.logger.info(f"    F1 : {val_struct_f1:.4f}")
            self.logger.info(f"    AUC: {val_struct_auc:.4f}")

        avg_loss = total_loss / len(self.val_loader)
        avg_total_loss = total_loss_with_aux / len(self.val_loader)
        # 使用全局最优阈值计算的预测结果来计算准确率，保持与混淆矩阵一致
        accuracy = (binary_preds == all_labels).mean()
        
        # 输出详细的验证统计信息
        self.logger.info(f"验证完成 (耗时: {val_time:.2f}s)")
        self.logger.info(f"使用最优阈值: {best_threshold:.3f}")
        self.logger.info("验证集统计:")
        
        # 标签和预测分布
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(binary_preds, return_counts=True)
        self.logger.info(f"  标签分布: {dict(zip(unique_labels.astype(int), label_counts))}")
        self.logger.info(f"  预测分布: {dict(zip(unique_preds.astype(int), pred_counts))}")
        
        # 概率值统计
        self.logger.info(f"  概率值范围: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
        self.logger.info(f"  概率值均值: {all_probs.mean():.4f}")
        self.logger.info(f"  概率值标准差: {all_probs.std():.4f}")

        # 置信度评估指标
        high_confidence_count = np.sum(all_probs >= 0.8)
        medium_confidence_count = np.sum((all_probs >= 0.5) & (all_probs < 0.8))
        low_confidence_count = np.sum(all_probs < 0.5)
        total_samples = len(all_probs)

        self.logger.info("置信度分布:")
        self.logger.info(f"  高置信度 (≥0.8): {high_confidence_count}/{total_samples} ({high_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  中等置信度 (0.5-0.8): {medium_confidence_count}/{total_samples} ({medium_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  低置信度 (<0.5): {low_confidence_count}/{total_samples} ({low_confidence_count/total_samples*100:.1f}%)")

        # 详细指标
        self.logger.info("验证结果详情:")
        self.logger.info(f"  Main Loss: {avg_loss:.4f}")
        self.logger.info(f"  Total Loss (w/ Aux): {avg_total_loss:.4f} (Comparable to Train Loss)")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        self.logger.info(f"  F1: {f1:.4f}, MCC: {mcc:.4f}")
        self.logger.info(f"  AUC: {auc_score:.4f}, AUPRC: {auprc:.4f}")
        
        # 类别平衡分析
        val_specificity = 0.0
        if len(unique_labels) == 2:
            positive_ratio = label_counts[1] / sum(label_counts) if len(label_counts) > 1 else 0
            pred_positive_ratio = pred_counts[1] / sum(pred_counts) if len(pred_counts) > 1 and len(pred_counts) == 2 else 0
            self.logger.info(f"  真实正样本比例: {positive_ratio:.3f}")
            self.logger.info(f"  预测正样本比例: {pred_positive_ratio:.3f}")
            
            # 计算混淆矩阵的各个值
            cm = confusion_matrix(all_labels, binary_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                val_specificity = specificity
                self.logger.info(f"  特异性 (Specificity): {specificity:.4f}")
                self.logger.info(f"  敏感性 (Sensitivity): {sensitivity:.4f}")
                self.logger.info(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            else:
                self.logger.info(f"  混淆矩阵形状异常: {cm.shape}")
                self.logger.info(f"  混淆矩阵: {cm}")
        
        self.logger.info("="*60)
        
        return {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_specificity': val_specificity,
            'val_f1': f1,
            'val_auc': auc_score,
            'val_mcc': mcc,
            'val_auprc': auprc,
            'val_seq_mcc': val_seq_mcc,
            'val_struct_mcc': val_struct_mcc,
            'best_threshold': best_threshold
        }
    
    def _compute_loss(self, logits, labels, features=None):
        """计算损失"""
        if hasattr(self.model, 'compute_enhanced_loss'):
            # TODO: Update model interface to accept features if needed
            return self.model.compute_enhanced_loss(
                logits, labels, 
                use_focal=self.loss_config['use_focal_loss'],
                use_label_smooth=self.loss_config['use_label_smoothing']
            )
        else:
            return self._compute_basic_loss(logits, labels, features)
    
    def _compute_basic_loss(self, logits, labels, features=None):
        """计算基础损失，支持 OHEM (Hard Negative Mining)"""
        loss = None
        
        # 准备 reduction 模式
        # 如果启用了 hard_negative_mining，我们需要 per-sample loss
        use_ohem = self.loss_config.get('use_hard_negative_mining', False)
        reduction = 'none' if use_ohem else 'mean'
        
        # 使用EnhancedLoss中的Focal Loss
        if self.loss_config.get('use_focal_loss', False):
            # ... (自适应权重逻辑，已在之前的 Turn 中处理)
            if self.loss_config.get('use_adaptive_weighting', False) and self.adaptive_weight_scheduler:
                weights = self.adaptive_weight_scheduler.weights
                sample_weights = torch.where(labels == 1, weights[1], weights[0])
                
                alpha = self.loss_config.get('focal_alpha', 0.25)
                gamma = self.loss_config.get('focal_gamma', 2.0)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                pt = torch.exp(-bce_loss)
                alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
                focal_loss_per_sample = alpha_t * (1 - pt) ** gamma * bce_loss
                
                # 应用自适应权重
                per_sample_loss = focal_loss_per_sample * sample_weights
                
                if use_ohem:
                    loss = self._apply_ohem(per_sample_loss, labels)
                else:
                    loss = per_sample_loss.mean()
            else:
                # 标准 Focal Loss
                if use_ohem:
                    # 我们需要 EnhancedLoss.focal_loss 返回 none reduction
                    # 但现在的实现是返回 mean。我们需要修改 EnhancedLoss 或在这里重写
                    # 简单起见，这里重写逻辑以支持 reduction='none'
                    alpha = self.loss_config.get('focal_alpha', 0.25)
                    gamma = self.loss_config.get('focal_gamma', 2.0)
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                    pt = torch.exp(-bce_loss)
                    alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
                    per_sample_loss = alpha_t * (1 - pt) ** gamma * bce_loss
                    loss = self._apply_ohem(per_sample_loss, labels)
                else:
                    loss = self.enhanced_loss.focal_loss(logits, labels)
                
            if self.loss_config.get('use_label_smoothing', False):
                smooth_loss = self._label_smooth_loss(logits, labels)
                weight = self.loss_config.get('loss_combination_weight', 0.5)
                loss = (weight * loss + (1 - weight) * smooth_loss)
        elif self.loss_config.get('use_label_smoothing', False):
            loss = self._label_smooth_loss(logits, labels)
        else:
            # 标准 BCE
            if self.pos_weight is not None:
                # ... (Adaptive weighting check)
                 if self.loss_config.get('use_adaptive_weighting', False) and self.adaptive_weight_scheduler:
                     weights = self.adaptive_weight_scheduler.weights
                     dynamic_pos_weight = weights[1] / (weights[0] + 1e-8)
                     per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=dynamic_pos_weight, reduction='none')
                 else:
                    per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=self.pos_weight, reduction='none')
            else:
                per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            
            if use_ohem:
                loss = self._apply_ohem(per_sample_loss, labels)
            else:
                loss = per_sample_loss.mean()
        
        # 添加MCC Loss (通常作为辅助，不应用 OHEM)
        if self.loss_config.get('use_mcc_loss', False):
            mcc_loss = self._soft_mcc_loss(logits, labels)
            loss = loss + self.loss_config.get('mcc_loss_weight', 0.5) * mcc_loss
            
        # 添加对比损失
        if self.loss_config.get('use_contrastive_loss', False) and features is not None:
            contrastive_loss = self.enhanced_loss.contrastive_loss(features, labels)
            loss = loss + self.loss_config.get('contrastive_loss_weight', 0.1) * contrastive_loss
            
        return loss

    def _apply_ohem(self, per_sample_loss, labels):
        """应用在线困难样本挖掘 (OHEM)"""
        # 策略：保留所有正样本，挖掘困难负样本
        # 或者：保留 top K total loss
        
        # 这里实现：保留 top ratio 的样本
        ratio = self.loss_config.get('hard_negative_mining_ratio', 0.5) # 默认保留50%
        num_samples = per_sample_loss.size(0)
        num_keep = int(num_samples * ratio)
        
        if num_keep < 1:
            return per_sample_loss.mean()
            
        # 获取 loss 最高的样本索引
        top_k_loss, _ = torch.topk(per_sample_loss, num_keep)
        return top_k_loss.mean()
    
    def _focal_loss(self, inputs, targets):
        """
        Focal Loss implementation.
        Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        alpha = self.loss_config.get('focal_alpha', 0.25)
        gamma = self.loss_config.get('focal_gamma', 2.0)
        
        # Calculate BCE with logits, no reduction to keep shape [batch_size, *]
        # We do NOT use pos_weight here because alpha handles class balancing in Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt
        # pt = exp(-bce_loss) is correct because bce_loss = -log(pt)
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha_t
        # alpha for class 1, 1-alpha for class 0
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        
        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
        
        return focal_loss.mean()

    def _soft_mcc_loss(self, inputs, targets):
        """
        Soft Matthews Correlation Coefficient Loss
        Approximates MCC as a differentiable loss function.
        Loss = 1 - MCC
        """
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        
        tp = torch.sum(probs * targets)
        tn = torch.sum((1 - probs) * (1 - targets))
        fp = torch.sum(probs * (1 - targets))
        fn = torch.sum((1 - probs) * targets)
        
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        # Add epsilon to denominator to avoid division by zero
        mcc = numerator / (denominator + 1e-7)
        return 1 - mcc
    
    def _label_smooth_loss(self, inputs, targets):
        """标签平滑损失"""
        smoothing = self.loss_config['label_smoothing']
        targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(inputs, targets_smooth, pos_weight=self.pos_weight)
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)
    
    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics):
        """记录到TensorBoard"""
        if self.tb_writer is not None:
            # 记录训练指标
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key}', value, epoch)

            # 记录验证指标
            for key, value in val_metrics.items():
                if key != 'best_threshold':
                    self.tb_writer.add_scalar(f'Validation/{key}', value, epoch)

            # 记录学习率
            self.tb_writer.add_scalar('Learning_Rate', self._get_current_lr(), epoch)

    def _get_current_lr(self):
        """获取当前学习率"""
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            last_lr = self.scheduler.get_last_lr()
            if isinstance(last_lr, (list, tuple)):
                return float(last_lr[0])
            return float(last_lr)
        return float(self.optimizer.param_groups[0]['lr'])

    def _step_scheduler_batch(self):
        """按批次更新调度器"""
        if not self.scheduler or not self.scheduler_step_on_batch:
            return
        try:
            self.scheduler.step()
        except TypeError:
            self.scheduler.step(self.global_step)

    def _step_scheduler_epoch(self, epoch, val_metrics):
        """按epoch更新调度器"""
        if not self.scheduler or self.scheduler_step_on_batch:
            return

        if self.scheduler_type == 'plateau':
            metric_value = val_metrics.get(self.monitor_metric)
            if metric_value is None:
                metric_value = val_metrics.get('val_loss', 0.0)
            self.scheduler.step(metric_value)
        elif self.scheduler_type == 'cosine_warm_restarts':
            try:
                self.scheduler.step(epoch + 1)
            except TypeError:
                self.scheduler.step()
        else:
            try:
                self.scheduler.step()
            except TypeError:
                self.scheduler.step(epoch + 1)
    
    def _update_history(self, train_metrics, val_metrics):
        """更新训练历史"""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['train_f1'].append(train_metrics['f1'])
        
        self.training_history['val_loss'].append(val_metrics['val_loss'])
        self.training_history['val_acc'].append(val_metrics['val_acc'])
        self.training_history['val_f1'].append(val_metrics['val_f1'])
        self.training_history['val_auc'].append(val_metrics['val_auc'])
        self.training_history['val_precision'].append(val_metrics['val_precision'])
        self.training_history['val_recall'].append(val_metrics['val_recall'])
        self.training_history['val_mcc'].append(val_metrics['val_mcc'])
        self.training_history['val_auprc'].append(val_metrics['val_auprc'])
    
    def evaluate_test_set(self, test_loader):
        """评估测试集"""
        if test_loader is None:
            return None
        
        self.logger.info("=== Testing ===")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 确保数据在正确设备上
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                outputs = self.model(batch)
                labels = batch['label']

                # 处理输出格式
                importance_scores = None
                is_multitask = False

                if isinstance(outputs, tuple):
                    # 检查是否是多任务输出 (acetylation_transformer)
                    if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                        # 检查是否是多任务输出
                        if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                            # 这是多任务输出
                            is_multitask = True
                            class_logits, strength_logits = outputs
                            logits = class_logits  # 使用分类logits作为主要输出
                        else:
                            # 这是可解释性输出
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    else:
                        # 其他元组格式
                        logits = outputs[0]
                        importance_scores = outputs[1] if len(outputs) > 1 else None
                elif isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'predicted_outcome' in outputs:
                        logits = outputs['predicted_outcome']
                    else:
                        logits = outputs[list(outputs.keys())[0]]
                else:
                    logits = outputs

                # 确保张量维度匹配
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                # 处理多任务输出的特殊情况
                if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                    # 对于多任务输出，使用softmax获取概率
                    probs = torch.softmax(logits, dim=-1)
                    # 使用正类概率
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_labels.extend(labels.long().cpu().numpy())
                else:
                    # 获取概率
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # 计算测试指标
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 找到最优阈值
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.05):
            temp_preds = (all_probs > threshold).astype(int)
            if len(np.unique(temp_preds)) > 1:
                temp_f1 = f1_score(all_labels, temp_preds, zero_division=0)
                if temp_f1 > best_f1:
                    best_f1 = temp_f1
                    best_threshold = threshold
        
        binary_preds = (all_probs > best_threshold).astype(int)
        
        test_metrics = {
            'test_accuracy': (binary_preds == all_labels).mean(),
            'test_precision': precision_score(all_labels, binary_preds, zero_division=0),
            'test_recall': recall_score(all_labels, binary_preds, zero_division=0),
            'test_f1': f1_score(all_labels, binary_preds, zero_division=0),
            'test_mcc': matthews_corrcoef(all_labels, binary_preds),
            'test_threshold': best_threshold
        }
        
        if len(np.unique(all_labels)) > 1:
            test_metrics['test_auc'] = roc_auc_score(all_labels, all_probs)
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
            test_metrics['test_auprc'] = auc(recall_curve, precision_curve)
        
        self.logger.info("测试集结果:")
        for key, value in test_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            
        # 计算特异性和敏感性
        tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.logger.info(f"  test_specificity: {specificity:.4f}")
        self.logger.info(f"  test_sensitivity: {sensitivity:.4f}")
        self.logger.info(f"  test_confusion_matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        test_metrics['test_specificity'] = specificity
        test_metrics['test_sensitivity'] = sensitivity
        
        return test_metrics

    def _check_early_stopping(self, val_metrics, epoch, best_val_loss, best_val_acc,
                             best_val_f1, best_val_mcc, best_val_auprc, best_epoch):
        """
        检查早停条件

        Args:
            val_metrics: 验证指标字典
            epoch: 当前epoch
            best_val_*: 当前最佳指标
            best_epoch: 最佳epoch

        Returns:
            tuple: (should_stop, improvement_flag)
        """
        if not self.early_stopping['use_early_stopping']:
            # 如果没有启用早停，仍然检查是否有改进用于保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_model_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"已保存最佳模型到: {best_model_path}")
                return False, " *** 新的最佳模型! ***"
            return False, ""

        monitor_metric = self.early_stopping['monitor_metric']
        mode = self.early_stopping['mode']
        min_delta = self.early_stopping['min_delta']
        patience = self.early_stopping['patience']

        # 获取当前监控指标的值
        current_score = val_metrics.get(monitor_metric, val_metrics['val_loss'])

        # 判断是否有改进
        improved = False
        if mode == 'min':
            improved = current_score < (self.early_stopping['best_score'] - min_delta)
        else:  # mode == 'max'
            improved = current_score > (self.early_stopping['best_score'] + min_delta)

        improvement_flag = ""

        if improved:
            # 有改进，更新最佳分数并重置计数器
            self.early_stopping['best_score'] = current_score
            self.early_stopping['counter'] = 0
            improvement_flag = " *** 新的最佳模型! ***"

            # 保存最佳模型
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)

            if self.early_stopping['verbose']:
                self.logger.info(f"已保存最佳模型到: {best_model_path}")
                self.logger.info(f"监控指标 {monitor_metric} 改进: {current_score:.6f}")
        else:
            # 没有改进，增加计数器
            self.early_stopping['counter'] += 1
            improvement_flag = f" (早停计数: {self.early_stopping['counter']}/{patience})"

            if self.early_stopping['verbose'] and self.early_stopping['counter'] > 0:
                self.logger.info(f"监控指标 {monitor_metric} 没有改进: {current_score:.6f} "
                               f"(最佳: {self.early_stopping['best_score']:.6f})")

        # 检查是否应该停止训练
        should_stop = self.early_stopping['counter'] >= patience

        if should_stop and self.early_stopping['restore_best_weights']:
            # 恢复最佳权重
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                if self.early_stopping['verbose']:
                    self.logger.info("已恢复最佳模型权重")

        return should_stop, improvement_flag
