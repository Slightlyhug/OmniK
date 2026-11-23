"""
通用工具函数模块
包含项目中常用的工具函数，如随机种子设置、数据批处理等
"""

import random
import numpy as np
import torch
import logging

# 设置日志
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")


def custom_collate_fn(batch):
    """
    自定义批处理函数，用于处理预提取特征的数据
    
    该函数处理包含序列特征、结构特征、标签等多种数据类型的批次，
    确保所有数据都能正确地组织成批次张量。
    
    Args:
        batch (list): 包含多个样本的列表，每个样本是一个字典
        
    Returns:
        dict: 批处理后的数据字典，包含以下键：
            - sequence_features: 序列特征张量 [batch_size, seq_len, feature_dim]
            - local_features: 局部特征张量 [batch_size, local_len, feature_dim]  
            - global_features: 全局特征张量 [batch_size, feature_dim]
            - structure_features: 结构特征张量 [batch_size, seq_len, struct_dim] (可选)
            - residue_types: 残基类型张量 [batch_size, seq_len]
            - position_ids: 位置ID张量 [batch_size, seq_len]
            - ptm_type_ids: PTM类型ID张量 [batch_size]
            - label: 标签张量 [batch_size]
            - uniprot_ids: UniProt ID列表
            - positions: 位置列表
    """
    try:
        # 检查批次是否为空
        if not batch:
            raise ValueError("批次数据为空")
        
        # 获取第一个样本以确定数据结构
        first_sample = batch[0]
        
        # 处理序列特征
        if 'sequence' in first_sample and isinstance(first_sample['sequence'], dict):
            # 新格式：sequence是字典，包含多种特征
            sequence_features = []
            local_features = []
            global_features = []
            
            for item in batch:
                seq_data = item['sequence']
                
                # 窗口特征
                if 'window_features' in seq_data:
                    sequence_features.append(seq_data['window_features'])
                elif 'features' in seq_data:
                    sequence_features.append(seq_data['features'])
                else:
                    raise KeyError("序列数据中缺少特征信息")
                
                # 局部特征
                if 'local_features' in seq_data:
                    local_features.append(seq_data['local_features'])
                elif 'local' in seq_data:
                    local_features.append(seq_data['local'])
                else:
                    # 如果没有局部特征，使用窗口特征的子集
                    local_features.append(sequence_features[-1])
                
                # 全局特征
                if 'global_features' in seq_data:
                    global_features.append(seq_data['global_features'])
                elif 'global' in seq_data:
                    global_features.append(seq_data['global'])
                else:
                    # 如果没有全局特征，使用窗口特征的平均值
                    global_features.append(torch.mean(sequence_features[-1], dim=0))
            
            # 转换为张量
            sequence_features = torch.stack(sequence_features)
            local_features = torch.stack(local_features)
            global_features = torch.stack(global_features)
            
        elif 'sequence_features' in first_sample:
            # Flat format (DataLoadingOptimizer output)
            sequence_features = torch.stack([
                torch.from_numpy(item['sequence_features']) if isinstance(item['sequence_features'], np.ndarray) else item['sequence_features'] 
                for item in batch
            ])
            
            if 'local_features' in first_sample:
                local_features = torch.stack([
                    torch.from_numpy(item['local_features']) if isinstance(item['local_features'], np.ndarray) else item['local_features']
                    for item in batch
                ])
            else:
                local_features = sequence_features

            if 'global_features' in first_sample:
                global_features = torch.stack([
                    torch.from_numpy(item['global_features']) if isinstance(item['global_features'], np.ndarray) else item['global_features']
                    for item in batch
                ])
            else:
                global_features = torch.mean(sequence_features, dim=1)

        else:
            # 旧格式：直接使用sequence作为特征
            sequence_features = torch.stack([item['sequence'] for item in batch])
            local_features = sequence_features  # 使用相同的特征
            global_features = torch.mean(sequence_features, dim=1)  # 全局特征为平均值
        
        # 处理结构特征（可选，允许缺失值）
        structure_features = None
        structure_mask = None
        if 'structure' in first_sample:
            structure_values = [item.get('structure') for item in batch]
            tensor_structures = [value for value in structure_values if isinstance(value, torch.Tensor)]

            if tensor_structures:
                reference_tensor = tensor_structures[0]
                padded_structures = []
                mask_values = []

                for value in structure_values:
                    if isinstance(value, torch.Tensor):
                        padded_structures.append(value)
                        mask_values.append(1.0)
                    else:
                        padded_structures.append(torch.zeros_like(reference_tensor))
                        mask_values.append(0.0)

                if tensor_structures and len(tensor_structures) != len(structure_values):
                    missing_count = len(structure_values) - len(tensor_structures)
                    if not getattr(custom_collate_fn, "_structure_warning_issued", False):
                        logger.warning(
                            "custom_collate_fn: 检测到 %d 个样本缺少结构特征，使用零填充并提供structure_mask", missing_count
                        )
                        setattr(custom_collate_fn, "_structure_warning_issued", True)

                structure_features = torch.stack(padded_structures)
                structure_mask = torch.tensor(mask_values, dtype=reference_tensor.dtype)
            else:
                structure_features = None
                structure_mask = None
        elif 'structure_features' in first_sample:
            # Handle flat format from DataLoadingOptimizer
            structure_values = [item.get('structure_features') for item in batch]
            valid_structures = [s for s in structure_values if s is not None]
            
            if valid_structures:
                # Get reference tensor/array for shape
                ref = valid_structures[0]
                if isinstance(ref, np.ndarray):
                    ref_tensor = torch.from_numpy(ref)
                else:
                    ref_tensor = ref
                
                padded_structures = []
                mask_values = []
                
                for s in structure_values:
                    if s is not None:
                        if isinstance(s, np.ndarray):
                            padded_structures.append(torch.from_numpy(s))
                        else:
                            padded_structures.append(s)
                        mask_values.append(1.0)
                    else:
                        padded_structures.append(torch.zeros_like(ref_tensor))
                        mask_values.append(0.0)
                
                structure_features = torch.stack(padded_structures)
                structure_mask = torch.tensor(mask_values, dtype=torch.float32)
        
        # 处理残基类型
        residue_types = None
        if 'residue_types' in first_sample:
            residue_types = torch.stack([item['residue_types'] for item in batch])
        elif 'residue_type' in first_sample:
            residue_types = torch.stack([item['residue_type'] for item in batch])

        # 处理目标位点指示器
        site_indicator = None
        if 'site_indicator' in first_sample:
            site_indicator = torch.stack([item['site_indicator'] for item in batch])

        # 处理位置信息
        position_ids = None
        relative_position_offsets = None
        if 'position_ids' in first_sample:
            position_ids = torch.stack([item['position_ids'] for item in batch])
        if 'relative_position_offsets' in first_sample:
            relative_position_offsets = torch.stack([item['relative_position_offsets'] for item in batch])

        # 处理PTM类型
        ptm_type_ids = None
        if 'ptm_type_id' in first_sample:
            ptm_type_ids = torch.tensor([
                int(item['ptm_type_id'].item()) if isinstance(item['ptm_type_id'], torch.Tensor) else int(item['ptm_type_id'])
                for item in batch
            ], dtype=torch.long)
        elif 'ptm_type_ids' in first_sample:
            ptm_type_ids = torch.tensor([
                int(item['ptm_type_ids']) for item in batch
            ], dtype=torch.long)
        
        # 处理微环境特征
        micro_env_features = None
        if 'micro_env_features' in first_sample:
            # 统一特征维度，以确保stack不会失败
            target_dim = 6
            processed_features = []
            for item in batch:
                raw_feat = item.get('micro_env_features')
                if raw_feat is None:
                     # Default to zeros if missing
                     feat = torch.zeros(target_dim, dtype=torch.float32)
                else:
                    feat = torch.as_tensor(raw_feat, dtype=torch.float32)
                    if feat.shape[0] > target_dim:
                        feat = feat[:target_dim]
                    elif feat.shape[0] < target_dim:
                        padding = torch.zeros(target_dim - feat.shape[0], dtype=feat.dtype)
                        feat = torch.cat([feat, padding])
                processed_features.append(feat)
            
            micro_env_features = torch.stack(processed_features)

        # 处理标签
        labels = []
        for item in batch:
            lbl = item.get('label')
            if lbl is None:
                labels.append(torch.tensor(-1.0, dtype=torch.float32)) # Dummy label
            elif isinstance(lbl, torch.Tensor):
                labels.append(lbl)
            else:
                labels.append(torch.tensor(lbl, dtype=torch.float32))
        labels = torch.stack(labels)
        
        # 处理元数据
        uniprot_ids = [item.get('uniprot_id', '') for item in batch]
        positions = [item.get('position', -1) for item in batch]
        residues = [item.get('residue', '') for item in batch]
        ptm_types = [item.get('ptm_type', '') for item in batch]
        
        # 构建返回字典
        result = {
            'sequence_features': sequence_features,
            'local_features': local_features,
            'global_features': global_features,
            'label': labels,
            'uniprot_ids': uniprot_ids,
            'positions': positions,
            'residues': residues,
            'ptm_types': ptm_types
        }

        if micro_env_features is not None:
            result['micro_env_features'] = micro_env_features
        
        # 添加可选特征
        if structure_features is not None:
            result['structure_features'] = structure_features
            if structure_mask is not None:
                result['structure_mask'] = structure_mask
        
        if residue_types is not None:
            result['residue_types'] = residue_types

        if position_ids is not None:
            result['position_ids'] = position_ids
        if relative_position_offsets is not None:
            result['relative_position_offsets'] = relative_position_offsets
        if site_indicator is not None:
            result['site_indicator'] = site_indicator

        if ptm_type_ids is not None:
            result['ptm_type_ids'] = ptm_type_ids

        # 为兼容需要嵌套"sequence"/"structure"键的模型，提供统一接口
        result['sequence'] = {
            'window_features': sequence_features,
            'local_features': local_features,
            'global_features': global_features
        }

        if structure_features is not None:
            result['structure'] = structure_features
            # 一些模型会同时检查结构掩码，放在sequence字典中便于访问
            if structure_mask is not None:
                result['sequence']['structure_mask'] = structure_mask

        if residue_types is not None:
            result['sequence']['residue_types'] = residue_types
        if position_ids is not None:
            result['sequence']['position_ids'] = position_ids
        if relative_position_offsets is not None:
            result['sequence']['relative_position_offsets'] = relative_position_offsets
        if site_indicator is not None:
            result['sequence']['site_indicator'] = site_indicator
        if ptm_type_ids is not None:
            result['sequence']['ptm_type_ids'] = ptm_type_ids
        
        return result
        
    except Exception as e:
        logger.error(f"批处理函数出错: {str(e)}")
        logger.error(f"批次大小: {len(batch) if batch else 0}")
        if batch:
            logger.error(f"第一个样本的键: {list(first_sample.keys())}")
        raise


def check_gpu_memory():
    """
    检查GPU内存使用情况
    
    Returns:
        str: GPU内存使用情况的描述
    """
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated_memory / (1024**3)
        cached_gb = cached_memory / (1024**3)
        free_gb = total_gb - allocated_gb
        
        return (f"GPU {device}: {allocated_gb:.2f}GB/{total_gb:.2f}GB 已使用, "
                f"{cached_gb:.2f}GB 缓存, {free_gb:.2f}GB 可用")
    except Exception as e:
        return f"无法获取GPU内存信息: {str(e)}"


def get_device(gpu_id=None):
    """
    获取可用的计算设备
    
    Args:
        gpu_id (int, optional): 指定的GPU ID
        
    Returns:
        torch.device: 计算设备
    """
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"使用指定GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return device
        else:
            logger.warning(f"指定的GPU {gpu_id} 不存在，使用默认设备")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
        return device


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.2f}秒"


def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        dict: 包含总参数数和可训练参数数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        loss: 当前损失
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(f"检查点已保存到: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        
    Returns:
        dict: 包含epoch和loss信息的字典
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"检查点已从 {filepath} 加载")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }
