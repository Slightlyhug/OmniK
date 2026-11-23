import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import os
from argparse import Namespace
import logging
import argparse # Import argparse for add_safe_globals
from pathlib import Path

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("protein_encoders")

# 定义模型文件路径
_DEFAULT_ESM_MODEL_PATHS = {
    "esm2_t33_650M_UR50D": "",
    "esm2_t33_650M_UR50D_alphabet": "",
    "esm2_t33_650M_UR50D_contact": "",
    "esm2_t36_3B_UR50D": "",
    "esm_if1_gvp4_t16_142M_UR50": "",
}

_ESM_MODEL_ENV_KEYS = {
    "esm2_t33_650M_UR50D": "ENHANCED_PTMS_ESM2_T33_PATH",
    "esm2_t33_650M_UR50D_alphabet": "ENHANCED_PTMS_ESM2_T33_ALPHABET_PATH",
    "esm2_t33_650M_UR50D_contact": "ENHANCED_PTMS_ESM2_T33_CONTACT_PATH",
    "esm2_t36_3B_UR50D": "ENHANCED_PTMS_ESM2_T36_PATH",
    "esm_if1_gvp4_t16_142M_UR50": "ENHANCED_PTMS_ESM_IF1_PATH",
}


def _resolve_esm_model_paths():
    resolved = {}
    configured_root = os.getenv("ENHANCED_PTMS_ESM_ROOT")
    for name, default_path in _DEFAULT_ESM_MODEL_PATHS.items():
        env_value = os.getenv(_ESM_MODEL_ENV_KEYS.get(name, ""))
        if env_value:
            resolved[name] = env_value
            continue
        if configured_root:
            candidate = Path(configured_root) / Path(default_path).name
            if candidate.exists():
                resolved[name] = str(candidate)
                continue
        resolved[name] = default_path
    return resolved


ESM_MODEL_PATHS = _resolve_esm_model_paths()

class ProteinSequenceEncoder(nn.Module):
    """
    蛋白质序列编码器: 基于ESM-2模型提取蛋白质序列特征
    参考ProteinGPT实现，优化加载和特征提取
    """
    def __init__(self, model_name="esm2_t33_650M_UR50D", freeze=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"ESM模型将加载到设备: {self.device}")
        
        if model_name in ESM_MODEL_PATHS:
            model_path = ESM_MODEL_PATHS[model_name]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型
            self.model, self.alphabet = self._load_model_from_path(model_path, model_name)
            self.model = self.model.to(self.device)
            
            # 确保模型有 embed_dim 属性
            if hasattr(self.model, "embed_dim"):
                self.embed_dim = self.model.embed_dim
            elif hasattr(self.model, "args") and hasattr(self.model.args, "embed_dim"):
                self.embed_dim = self.model.args.embed_dim
            elif hasattr(self.model, "dim"):
                self.embed_dim = self.model.dim
            else:
                # 通过检查模型参数推断嵌入维度
                for name, param in self.model.named_parameters():
                    if 'embed' in name and 'weight' in name:
                        self.embed_dim = param.shape[1]
                        break
                if not hasattr(self, 'embed_dim'):
                    # 使用默认值或从当前架构推断
                    if '3B' in model_name:
                        self.embed_dim = 2560
                    elif '650M' in model_name:
                        self.embed_dim = 1280
                    else:
                        self.embed_dim = 768
                    logger.warning(f"无法直接获取embed_dim，使用推断值: {self.embed_dim}")
            
            # 设置返回隐藏状态的层
            if hasattr(self.model, "num_layers"):
                self.repr_layer_index = self.model.num_layers
            else:
                self.repr_layer_index = -1
                
            # 冻结参数
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("序列编码器模型参数已冻结")
        else:
            raise ValueError(f"没有为模型 {model_name} 配置本地路径，请确保其在ESM_MODEL_PATHS中定义")
    
    def _load_model_from_path(self, model_path, model_name):
        """辅助函数：从路径加载模型，优化加载过程"""
        logger.info(f"加载序列模型: {model_name} 从: {model_path}")

        try:
            # 策略1: 首先尝试使用ESM库的API直接加载
            try:
                if hasattr(esm.pretrained, 'load_model_and_alphabet_local'):
                    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                    logger.info(f"成功使用ESM API加载序列模型: {model_name}")
                    return model, alphabet
                else:
                    logger.warning("当前esm版本缺少 'load_model_and_alphabet_local'，尝试替代方法...")
            except Exception as e:
                logger.warning(f"ESM API加载失败: {str(e)}，尝试替代方法...")
            
            # 策略2: 使用torch直接加载模型数据
            logger.info("使用torch.load加载模型数据...")
            # 添加安全全局变量以支持argparse.Namespace
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            try:
                model_data = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.warning(f"weights_only=True 加载失败: {e}，尝试 weights_only=False")
                model_data = torch.load(model_path, map_location="cpu", weights_only=False)
            logger.info("模型数据加载成功")
            
            # 获取模型名称和目录
            model_location = Path(model_path)
            model_basename = model_location.stem
            logger.info(f"模型文件名: {model_basename}")
            
            # 检查是否需要回归权重
            regression_data = None
            if not ("esm1v" in model_basename or "esm_if" in model_basename or "270K" in model_basename or "500K" in model_basename):
                try:
                    regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
                    if os.path.exists(regression_location):
                        try:
                            regression_data = torch.load(regression_location, map_location="cpu", weights_only=True)
                        except Exception:
                            regression_data = torch.load(regression_location, map_location="cpu", weights_only=False)
                        logger.info("回归数据加载成功")
                except Exception as re:
                    logger.warning(f"无法加载回归数据: {re}，但这不影响特征提取")
            
            # 策略3: 使用ESM库中的核心函数加载模型
            try:
                if hasattr(esm.pretrained, 'load_model_and_alphabet_core'):
                    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                        model_basename, model_data, regression_data
                    )
                    logger.info("使用ESM核心函数成功加载模型")
                    return model, alphabet
            except Exception as core_e:
                logger.warning(f"使用核心函数加载失败: {core_e}，尝试其他方法...")
            
            # 策略4: 直接从model_data提取模型
            if isinstance(model_data, dict):
                if "model" in model_data:
                    model_obj = model_data["model"]
                    alphabet = model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
                    logger.info("从模型字典加载序列模型成功")
                    return model_obj, alphabet
                elif "args" in model_data and "state_dict" in model_data:
                    args = model_data["args"]
                    state_dict = model_data["state_dict"]
                    if hasattr(esm, 'model') and hasattr(esm.model, 'ProteinBertModel'):
                        model = esm.model.ProteinBertModel(args)
                        model.load_state_dict(state_dict)
                        alphabet = esm.data.Alphabet.from_architecture(args.arch)
                        logger.info("从state_dict和args创建序列模型成功")
                        return model, alphabet
                    else:
                        raise ValueError("当前esm版本缺少 'model.ProteinBertModel'")
            elif isinstance(model_data, nn.Module):
                logger.info("直接加载序列模型模块成功")
                return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
            
            # 策略5: 回退到使用预训练模型
            logger.warning(f"无法从文件加载模型，尝试使用预训练模型...")
            if "esm2_t33_650M" in model_basename:
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                logger.info("使用预训练的esm2_t33_650M_UR50D模型")
                return model, alphabet
            elif "esm2_t36_3B" in model_basename:
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                logger.info("使用预训练的esm2_t36_3B_UR50D模型")
                return model, alphabet
            else:
                raise ValueError(f"无法找到匹配的预训练模型: {model_basename}")
                
        except Exception as e:
            logger.error(f"所有加载方法均失败: {str(e)}")
            raise ValueError(f"无法加载序列模型 {model_name}: {str(e)}")

    def tokenize(self, sequences):
        """将蛋白质序列转换为模型输入的token ID"""
        if not hasattr(self, 'alphabet'):
            raise ValueError("模型没有alphabet属性，无法进行标记化")
            
        batch_tokens = []
        for seq in sequences:
            tokens = self.alphabet.encode(seq)
            batch_tokens.append(tokens)
        
        # 填充到相同长度
        max_len = max(len(t) for t in batch_tokens)
        batch_tokens_padded = []
        for tokens in batch_tokens:
            padding = torch.full((max_len - len(tokens),), self.alphabet.padding_idx)
            batch_tokens_padded.append(torch.cat([tokens, padding]))
        
        return torch.stack(batch_tokens_padded)

    def extract_features(self, tokens, return_contacts=False):
        """从序列提取特征，可选择返回接触图"""
        if hasattr(self, 'device'):
            tokens = tokens.to(self.device)
            
        with torch.no_grad():
            # 使用ESM-2的标准API
            if hasattr(self.model, 'extract_features'):
                out = self.model.extract_features(
                    tokens,
                    repr_layers=[self.repr_layer_index],
                    return_contacts=return_contacts
                )
                sequence_repr = out['representations'][self.repr_layer_index]
                
                # 将CLS token分离为全局表示，其余为每个残基的表示
                global_repr = sequence_repr[:, 0]
                per_residue_repr = sequence_repr[:, 1:]
                
                if return_contacts:
                    return global_repr, per_residue_repr, out['contacts']
                return global_repr, per_residue_repr
            
            # 如果没有extract_features方法，使用标准前向传播
            return self._forward_standard(tokens)
                
    def forward(self, tokens):
        """
        前向传播。优先使用extract_features方法，如果不可用则回退到标准前向传播。
        
        Args:
            tokens: ESM标记化的序列。
        
        Returns:
            tuple: (sequence_repr, per_residue_repr)
        """
        if hasattr(self, 'device'):
            tokens = tokens.to(self.device)
            
        # 优先使用extract_features方法，它更一致地处理各种ESM模型版本
        if hasattr(self.model, 'extract_features'):
            return self.extract_features(tokens, return_contacts=False)
            
        return self._forward_standard(tokens)
        
    def _forward_standard(self, tokens):
        """标准前向传播方法，处理各种可能的模型输出格式"""
        with torch.no_grad():
            try:
                # 尝试标准ESM-2接口
                try:
                    logger.debug(f"尝试使用标准ESM-2接口处理输入 shape: {tokens.shape}")
                    repr_layers = [self.repr_layer_index]
                    output = self.model(tokens, repr_layers=repr_layers)
                    
                    if isinstance(output, dict) and "representations" in output:
                        layer_key = list(output["representations"].keys())[-1]
                        sequence_repr = output["representations"][layer_key][:, 0, :]
                        per_residue_repr = output["representations"][layer_key][:, 1:, :]
                        logger.debug(f"成功获取表示，序列表示shape: {sequence_repr.shape}, 每残基表示shape: {per_residue_repr.shape}")
                        return sequence_repr, per_residue_repr
                except Exception as e:
                    logger.warning(f"标准ESM-2接口失败: {e}")
                
                # 尝试简化接口
                try:
                    output = self.model(tokens)
                    
                    # 处理各种可能的输出格式
                    if isinstance(output, dict) and "representations" in output:
                        layer_key = list(output["representations"].keys())[-1]
                        sequence_repr = output["representations"][layer_key][:, 0, :]
                        per_residue_repr = output["representations"][layer_key][:, 1:, :]
                    elif isinstance(output, dict) and "last_hidden_state" in output:
                        sequence_repr = output["last_hidden_state"][:, 0, :]
                        per_residue_repr = output["last_hidden_state"][:, 1:, :]
                    elif isinstance(output, tuple) and len(output) > 0:
                        sequence_repr = output[0][:, 0, :]
                        per_residue_repr = output[0][:, 1:, :]
                    else:
                        sequence_repr = output[:, 0, :]
                        per_residue_repr = output[:, 1:, :]
                    
                    logger.debug(f"成功获取表示，序列表示shape: {sequence_repr.shape}, 每残基表示shape: {per_residue_repr.shape}")
                    return sequence_repr, per_residue_repr
                except Exception as e:
                    logger.warning(f"简化接口失败: {e}")
                
                # 尝试从模型特定方法获取特征
                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(tokens)
                    if isinstance(features, torch.Tensor):
                        sequence_repr = features[:, 0, :]
                        per_residue_repr = features[:, 1:, :]
                        return sequence_repr, per_residue_repr
                
                # 最后尝试直接调用
                logger.warning("无法确定模型接口，使用默认方法...")
                output = self.model(tokens)
                if isinstance(output, torch.Tensor):
                    sequence_repr = output[:, 0, :]
                    per_residue_repr = output[:, 1:, :]
                    return sequence_repr, per_residue_repr
                
                raise RuntimeError("无法从模型获取有效的表示")
                
            except Exception as e:
                raise RuntimeError(f"序列模型前向传播错误: {e}")

class GVP(nn.Module):
    """
    几何向量感知器 (Geometric Vector Perceptron)
    
    处理蛋白质结构中的标量特征和向量特征，保持旋转等变性
    参考ProteinGPT的实现
    """
    def __init__(self, scalar_dim, vector_dim, scalar_out_dim=None, vector_out_dim=None, activation=F.relu):
        super(GVP, self).__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.scalar_out_dim = scalar_out_dim or scalar_dim
        self.vector_out_dim = vector_out_dim or vector_dim
        self.activation = activation
        
        # 标量到标量/向量投影
        self.w_s2s = nn.Linear(scalar_dim, self.scalar_out_dim)
        self.w_s2v = nn.Linear(scalar_dim, 3*self.vector_out_dim) if vector_dim > 0 else None
        
        # 向量到标量/向量投影
        self.w_v2s = nn.Linear(vector_dim*3, self.scalar_out_dim) if vector_dim > 0 else None
        self.w_v2v = nn.Linear(vector_dim*3, 3*self.vector_out_dim) if vector_dim > 0 else None
        
    def forward(self, s, v=None):
        """
        前向传播
        
        Args:
            s: 标量特征 [batch, ..., scalar_dim]
            v: 向量特征 [batch, ..., vector_dim, 3]
            
        Returns:
            s_out: 输出标量特征 [batch, ..., scalar_out_dim]
            v_out: 输出向量特征 [batch, ..., vector_out_dim, 3]
        """
        s_out = self.w_s2s(s)
        
        if self.vector_dim > 0 and v is not None:
            v = v.reshape(v.shape[:-2] + (3*self.vector_dim,))
            
            # 向量特征到标量特征 (通过向量范数)
            v2s = self.w_v2s(v)
            s_out = s_out + v2s
            
            # 标量特征到向量特征
            s2v = self.w_s2v(s).reshape(s.shape[:-1] + (self.vector_out_dim, 3))
            
            # 向量特征到向量特征
            v2v = self.w_v2v(v).reshape(v.shape[:-1] + (self.vector_out_dim, 3))
            
            # 组合所有向量特征
            v_out = s2v + v2v
            
            # 应用激活函数到标量部分
            s_out = self.activation(s_out)
            
            # 使用向量范数调制向量部分以保持等变性
            v_norm = torch.norm(v_out, dim=-1, keepdim=True)
            v_normalized = self.activation(v_norm) * (v_out / (v_norm + 1e-8))
            
            return s_out, v_normalized
        else:
            # 只有标量特征时
            s_out = self.activation(s_out)
            return s_out, None

class GVPConvLayer(nn.Module):
    """
    几何向量卷积层，用于处理蛋白质结构中的空间关系
    """
    def __init__(self, scalar_dim, vector_dim, edge_dim=None, n_message=3):
        super(GVPConvLayer, self).__init__()
        
        # 边特征维度
        edge_dim = edge_dim or scalar_dim
        
        # 消息传递模块
        self.message_funcs = nn.ModuleList([
            GVP(scalar_dim + edge_dim, vector_dim, scalar_dim, vector_dim)
            for _ in range(n_message)
        ])
        
    def forward(self, s, v, edge_index, edge_attr=None):
        """
        Args:
            s: 节点标量特征 [n_nodes, scalar_dim]
            v: 节点向量特征 [n_nodes, vector_dim, 3]
            edge_index: 边连接 [2, n_edges]
            edge_attr: 边属性 [n_edges, edge_dim]
        """
        src, dst = edge_index
        
        for message_func in self.message_funcs:
            # 收集源节点特征
            s_src = s[src]
            v_src = v[src] if v is not None else None
            
            # 合并边特征
            if edge_attr is not None:
                s_src = torch.cat([s_src, edge_attr], dim=-1)
            
            # 通过GVP传递消息
            ds, dv = message_func(s_src, v_src)
            
            # 聚合消息
            ds = scatter_add(ds, dst, dim=0, dim_size=s.size(0))
            if v is not None:
                dv = scatter_add(dv, dst, dim=0, dim_size=v.size(0))
            
            # 更新节点表示
            s = s + ds
            if v is not None:
                v = v + dv
        
        return s, v

def scatter_add(src, index, dim, dim_size):
    """模拟torch_scatter.scatter_add的简化版本"""
    result = torch.zeros(dim_size, *src.shape[1:], device=src.device)
    result.index_add_(dim, index, src)
    return result

class ProteinStructureEncoder(nn.Module):
    """
    蛋白质结构编码器: 基于ESM-IF1模型和几何向量感知器处理蛋白质结构信息
    参考ProteinGPT实现，优化处理3D坐标和空间关系的能力
    """
    
    def __init__(self, model_name="esm_if1_gvp4_t16_142M_UR50", freeze=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"ESM-IF1模型将加载到设备: {self.device}")
        
        if model_name in ESM_MODEL_PATHS:
            model_path = ESM_MODEL_PATHS[model_name]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型
            self.model, self.alphabet = self._load_model_from_path(model_path, model_name)
            self.model = self.model.to(self.device)
            
            # 确保模型有 embed_dim 属性
            if hasattr(self.model, "embed_dim"):
                self.embed_dim = self.model.embed_dim
            elif hasattr(self.model, "args") and hasattr(self.model.args, "embed_dim"):
                self.embed_dim = self.model.args.embed_dim
            else:
                # ESM-IF1的默认嵌入维度
                self.embed_dim = 512
                logger.warning(f"模型 {model_name} 缺少 embed_dim 属性，使用默认值: {self.embed_dim}")
            
            # 设备验证
            model_device = next(self.model.parameters()).device
            if str(model_device) != str(self.device):
                logger.warning(f"模型在设备 {model_device} 上，强制移动到 {self.device}")
                self.model = self.model.to(self.device)
            
            # 增强GVP处理能力
            self.gvp_encoder = self._create_gvp_encoder()
            
            # 冻结参数
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("结构编码器模型参数已冻结")
        else:
            raise ValueError(f"没有为模型 {model_name} 配置本地路径，请确保其在ESM_MODEL_PATHS中定义")
    
    def _create_gvp_encoder(self):
        """创建额外的几何向量处理层增强结构表示能力"""
        return nn.ModuleDict({
            'node_embedding': GVP(self.embed_dim, 1, self.embed_dim, 16),
            'edge_embedding': GVP(self.embed_dim, 1, self.embed_dim, 4), 
            'conv_layers': nn.ModuleList([
                GVPConvLayer(self.embed_dim, 16, self.embed_dim)
                for _ in range(3)
            ]),
            'final_projection': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
        })
    
    def _load_model_from_path(self, model_path, model_name):
        """辅助函数：从路径加载模型"""
        logger.info(f"加载结构模型: {model_name} 从: {model_path}")
        # 结构模型通常不包含 argparse.Namespace，但为保持一致性，也添加
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])

        try:
            # 优先使用weights_only=True
            model_data = torch.load(model_path, map_location="cpu", weights_only=True)

            if isinstance(model_data, dict) and "model" in model_data:
                model_obj = model_data["model"]
                if isinstance(model_obj, nn.Module):
                    logger.info("成功从字典加载结构模型 (weights_only=True)")
                    # 使用 esm.data.Alphabet
                    return model_obj, model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
            elif isinstance(model_data, nn.Module):
                logger.info("成功直接加载结构模型模块 (weights_only=True)")
                # 使用 esm.data.Alphabet
                return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
            else:
                # 尝试使用ESM官方API加载
                try:
                    if hasattr(esm, 'pretrained') and hasattr(esm.pretrained, 'load_model_and_alphabet_local'):
                        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                        logger.info(f"使用ESM API成功加载结构模型: {model_name}")
                        return model, alphabet
                    else:
                        raise AttributeError("ESM API 加载方法不可用")
                except Exception as esm_err:
                    logger.warning(f"ESM API加载失败: {esm_err}，尝试 weights_only=False")
                    raise ValueError("无法解析模型数据 (weights_only=True)")
        except Exception as e:
            logger.error(f"加载结构模型失败 (weights_only=True): {e}")
            logger.warning("尝试使用 weights_only=False 重新加载...")
            try:
                model_data = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(model_data, dict) and "model" in model_data:
                    model_obj = model_data["model"]
                    if isinstance(model_obj, nn.Module):
                        logger.info("成功从字典加载结构模型 (weights_only=False)")
                        # 使用 esm.data.Alphabet
                        return model_obj, model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
                elif isinstance(model_data, nn.Module):
                    logger.info("成功直接加载结构模型模块 (weights_only=False)")
                    # 使用 esm.data.Alphabet
                    return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
                else:
                    raise ValueError("无法解析模型数据 (weights_only=False)")
            except Exception as fallback_e:
                logger.error(f"使用 weights_only=False 重新加载结构模型也失败: {fallback_e}")
                raise ValueError(f"无法加载结构模型 {model_name}: {fallback_e}")

    def _prepare_structure_input(self, coords):
        """
        处理蛋白质结构坐标为ESM-IF1可接受的格式
        
        Args:
            coords: 原子坐标张量 [batch_size, seq_length, n_atoms, 3]
                   通常n_atoms=4，代表N, CA, C, O四个主链原子
        
        Returns:
            dict: ESM-IF1的输入格式
        """
        batch_size, seq_length = coords.shape[0], coords.shape[1]
        
        # 确保坐标在正确的设备上
        if hasattr(self, 'device') and coords.device != self.device:
            coords = coords.to(self.device)
        
        # 创建ESM-IF1需要的输入格式
        structure_input = {
            "coords": {
                "N": coords[:, :, 0, :],    # 氮原子坐标
                "CA": coords[:, :, 1, :],   # α碳原子坐标
                "C": coords[:, :, 2, :],    # 碳原子坐标
            }
        }
        
        # 如果有第四个原子坐标(氧原子)，也添加进去
        if coords.shape[2] >= 4:
            structure_input["coords"]["O"] = coords[:, :, 3, :]
            
        # 添加掩码 - 所有位置都是有效的
        structure_input["mask"] = torch.ones(batch_size, seq_length, device=self.device).bool()
        
        # 为GVP处理准备额外特征
        structure_input["seq_length"] = seq_length
        
        # 计算残基间距离矩阵 (使用CA原子)
        ca_coords = structure_input["coords"]["CA"]  # [batch, seq_len, 3]
        
        # 添加批次维度的距离计算
        distances = []
        for b in range(batch_size):
            # 计算每对CA原子之间的欧氏距离
            ca_b = ca_coords[b]  # [seq_len, 3]
            dist_mat = torch.cdist(ca_b, ca_b)  # [seq_len, seq_len]
            distances.append(dist_mat)
        
        structure_input["distances"] = torch.stack(distances)  # [batch, seq_len, seq_len]
        
        return structure_input

    def _compute_structure_features(self, structure_input):
        """
        计算额外的几何结构特征以增强表示能力
        
        Args:
            structure_input: 包含原子坐标的字典
            
        Returns:
            torch.Tensor: 增强的结构特征
        """
        batch_size = structure_input["coords"]["CA"].shape[0]
        seq_length = structure_input["coords"]["CA"].shape[1]
        
        # 提取坐标
        ca_coords = structure_input["coords"]["CA"]  # [batch, seq_len, 3]
        n_coords = structure_input["coords"]["N"]    # [batch, seq_len, 3]
        c_coords = structure_input["coords"]["C"]    # [batch, seq_len, 3]
        
        all_features = []
        
        for b in range(batch_size):
            ca_b = ca_coords[b]  # [seq_len, 3]
            n_b = n_coords[b]    # [seq_len, 3]
            c_b = c_coords[b]    # [seq_len, 3]
            
            # 计算键角: N-CA-C
            v1 = n_b - ca_b  # N->CA向量
            v2 = c_b - ca_b  # C->CA向量
            
            # 单位化向量
            v1_norm = torch.norm(v1, dim=1, keepdim=True)
            v2_norm = torch.norm(v2, dim=1, keepdim=True)
            v1 = v1 / (v1_norm + 1e-7)
            v2 = v2 / (v2_norm + 1e-7)
            
            # 计算点积得到cos角
            cos_angles = torch.sum(v1 * v2, dim=1)  # [seq_len]
            
            # 计算叉积得到正弦值，用于确定二面角
            cross_products = torch.cross(v1, v2, dim=1)  # [seq_len, 3]
            sin_norms = torch.norm(cross_products, dim=1)  # [seq_len]
            
            # 创建一个本地坐标系，这对捕获残基间关系很有用
            local_frames = torch.zeros(seq_length, 3, 3, device=self.device)
            
            # X轴: CA->C的归一化向量
            x_axis = c_b - ca_b
            x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-7)
            
            # 临时Y轴: CA->N的归一化向量
            temp_y = n_b - ca_b
            temp_y = temp_y / (torch.norm(temp_y, dim=1, keepdim=True) + 1e-7)
            
            # Z轴: X轴和临时Y轴的叉积
            z_axis = torch.cross(x_axis, temp_y, dim=1)
            z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-7)
            
            # 真正的Y轴: Z轴和X轴的叉积
            y_axis = torch.cross(z_axis, x_axis, dim=1)
            
            # 组装局部坐标系
            local_frames[:, :, 0] = x_axis
            local_frames[:, :, 1] = y_axis
            local_frames[:, :, 2] = z_axis
            
            # 收集所有几何特征
            geom_features = torch.cat([
                cos_angles.unsqueeze(1),     # 键角余弦
                sin_norms.unsqueeze(1),      # 叉积范数
                torch.flatten(local_frames, 1, 2)  # 局部坐标系(展平)
            ], dim=1)
            
            all_features.append(geom_features)
        
        # 将所有批次的特征堆叠起来
        return torch.stack(all_features)  # [batch, seq_len, feature_dim]

    def forward(self, structure_data):
        """
        前向传播
        
        Args:
            structure_data: 蛋白质结构数据，可以是坐标张量[batch, seq_len, n_atoms, 3]
                           或已预处理的结构输入字典
        
        Returns:
            tuple: (structure_repr, per_residue_repr) - 全局结构表示和每残基表示
        """
        # 确保输入在正确的设备上
        if hasattr(self, 'device'):
            if isinstance(structure_data, torch.Tensor):
                structure_data = structure_data.to(self.device)
            elif isinstance(structure_data, dict):
                for key in structure_data:
                    if isinstance(structure_data[key], torch.Tensor):
                        structure_data[key] = structure_data[key].to(self.device)
                    elif isinstance(structure_data[key], dict):
                        for subkey in structure_data[key]:
                            if isinstance(structure_data[key][subkey], torch.Tensor):
                                structure_data[key][subkey] = structure_data[key][subkey].to(self.device)
        
        with torch.no_grad():
            try:
                # 准备结构输入
                if isinstance(structure_data, torch.Tensor):
                    structure_input = self._prepare_structure_input(structure_data)
                    logger.debug(f"已准备结构输入，从坐标张量shape: {structure_data.shape}")
                else:
                    structure_input = structure_data
                    logger.debug("使用预处理的结构输入")
                
                # 计算额外的几何特征
                geom_features = self._compute_structure_features(structure_input)
                
                # 使用ESM-IF1处理结构
                try:
                    # 支持多种ESM-IF1版本的接口
                    if hasattr(self.model, 'forward_structure'):
                        # 较新的ESM-IF1接口
                        output = self.model.forward_structure(structure_input)
                    else:
                        # 标准接口
                        repr_layers = [self.model.num_layers] if hasattr(self.model, "num_layers") else [-1]
                        output = self.model(structure_input, repr_layers=repr_layers)
                    
                    if isinstance(output, dict):
                        # 从输出提取表示
                        if "representations" in output:
                            layer_key = list(output["representations"].keys())[-1]
                            structure_repr = output["representations"][layer_key][:, 0, :]
                            per_residue_repr = output["representations"][layer_key][:, 1:, :]
                        elif "s" in output and "z" in output:
                            # ESM-IF1可能返回标量和向量表示
                            structure_repr = output["s"][:, 0, :]
                            per_residue_repr = output["s"][:, 1:, :]
                        else:
                            raise ValueError("无法从模型输出中提取表示")
                    else:
                        # 直接返回张量的情况
                        structure_repr = output[:, 0, :]
                        per_residue_repr = output[:, 1:, :]
                    
                    # 通过GVP增强表示
                    per_residue_repr = self._enhance_with_gvp(per_residue_repr, structure_input, geom_features)
                    
                    logger.debug(f"成功获取表示，结构表示shape: {structure_repr.shape}, 每残基表示shape: {per_residue_repr.shape}")
                    return structure_repr, per_residue_repr
                    
                except Exception as e:
                    logger.error(f"结构编码失败: {e}")
                    raise RuntimeError(f"ESM-IF1前向传播错误: {e}")
                    
            except Exception as e:
                logger.error(f"结构编码器错误: {e}")
                raise RuntimeError(f"结构编码器错误: {e}")
                
    def _enhance_with_gvp(self, per_residue_repr, structure_input, geom_features):
        """
        使用GVP增强每残基表示，并融合局部微环境特征
        
        Args:
            per_residue_repr: ESM-IF1产生的每残基表示 [batch, seq_len, embed_dim]
            structure_input: 结构输入字典
            geom_features: 额外的几何特征 [batch, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: 增强的每残基表示
        """
        batch_size = per_residue_repr.shape[0]
        seq_length = per_residue_repr.shape[1]
        
        # 计算局部微环境特征 (溶剂可及性、凹凸性等)
        micro_env_features = self._compute_micro_environment_features(structure_input)
        
        # 将微环境特征投影到嵌入维度
        if not hasattr(self, 'micro_env_projection'):
            self.micro_env_projection = nn.Sequential(
                nn.Linear(3, self.embed_dim // 4),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 4, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            ).to(per_residue_repr.device)
            
        micro_env_emb = self.micro_env_projection(micro_env_features)

        # 为简化计算，我们将直接合并几何特征到表示中
        enhanced_repr = per_residue_repr.clone()
        
        # 融合微环境特征 (增强局部感知)
        enhanced_repr = enhanced_repr + 0.2 * micro_env_emb
        
        # 如果可用，使用距离信息增强表示
        if "distances" in structure_input:
            # 为每个残基创建距离特征
            distances = structure_input["distances"]  # [batch, seq_len, seq_len]
            
            for b in range(batch_size):
                # 提取这个批次的距离矩阵
                dist_mat = distances[b]  # [seq_len, seq_len]
                
                # 计算距离加权的上下文特征
                # 权重随距离衰减: exp(-dist²/10²)
                weights = torch.exp(-(dist_mat ** 2) / 100).unsqueeze(-1)  # [seq_len, seq_len, 1]
                
                # 矩阵乘法实现加权平均: [seq_len, seq_len, 1] * [seq_len, embed_dim]
                context_features = torch.sum(
                    weights * enhanced_repr[b].unsqueeze(0).expand(seq_length, -1, -1),
                    dim=1
                )  # [seq_len, embed_dim]
                
                # 更新这个批次的表示
                enhanced_repr[b] = enhanced_repr[b] + 0.1 * context_features
        
        return enhanced_repr

    def _compute_micro_environment_features(self, structure_input):
        """
        计算微环境几何特征（溶剂可及性、局部凹凸性等）
        
        Args:
            structure_input: 包含原子坐标的字典
            
        Returns:
            torch.Tensor: 微环境特征 [batch, seq_len, 3]
                - Channel 0: 局部密度 (Local Density) / SASA proxy
                - Channel 1: 局部凹凸性 (Local Convexity/Protrusion)
                - Channel 2: 邻居平均距离 (Local Geometry Variance)
        """
        ca_coords = structure_input["coords"]["CA"] # [batch, seq_len, 3]
        batch_size, seq_len, _ = ca_coords.shape
        device = ca_coords.device

        micro_env_features = []

        for b in range(batch_size):
            coords = ca_coords[b] # [seq_len, 3]
            # 计算距离矩阵
            dist_mat = torch.cdist(coords, coords) # [seq_len, seq_len]
            
            # 1. 局部密度 (近似溶剂可及性) - 10A内的邻居数
            radius = 10.0
            # 排除自身 (距离 > 0)
            mask = (dist_mat < radius) & (dist_mat > 1e-6)
            local_density = mask.sum(dim=1).float().unsqueeze(1) # [seq_len, 1]
            
            # 2. 局部凹凸性 (Local Convexity)
            # 计算邻居的质心
            neighbor_sum = torch.matmul(mask.float(), coords) # [seq_len, 3]
            neighbor_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            centroids = neighbor_sum / neighbor_count # [seq_len, 3]
            
            # 计算自身到邻居质心的距离
            # 突出的残基通常距离其邻居的质心较远，且方向向外
            dist_to_centroid = torch.norm(coords - centroids, dim=1).unsqueeze(1) # [seq_len, 1]
            
            # 3. 邻居平均距离
            avg_dist = (dist_mat * mask.float()).sum(dim=1, keepdim=True) / neighbor_count # [seq_len, 1]
            
            # 归一化特征 (大致范围归一化，便于神经网络学习)
            features = torch.cat([
                local_density / 20.0,      # 密度通常在0-50之间
                dist_to_centroid / 5.0,    # 质心距离通常在0-10A之间
                avg_dist / 10.0            # 平均距离通常在0-10A之间
            ], dim=1)
            
            micro_env_features.append(features)
            
        return torch.stack(micro_env_features)

class AcetylationInfoEncoder(nn.Module):
    """
    乙酰化信息编码器: 处理与乙酰化修饰相关的特殊信息
    参考ProteinGPT架构，专门为乙酰化位点预测设计
    """
    
    def __init__(self, input_dim=21, hidden_dim=128, output_dim=512):
        """
        初始化乙酰化信息编码器
        
        Args:
            input_dim: 输入维度 (氨基酸类型数量 + 1)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，应与其他编码器输出维度匹配
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 氨基酸嵌入层
        self.aa_embedding = nn.Embedding(input_dim, hidden_dim)
        
        # 位置编码
        self.max_seq_len = 2048
        self.position_encoding = nn.Embedding(self.max_seq_len, hidden_dim)
        
        # 乙酰化位点特征提取器
        self.site_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 局部上下文编码器
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, aa_sequence, positions=None):
        """
        前向传播
        
        Args:
            aa_sequence: 氨基酸序列的整数编码 [batch_size, seq_len]
            positions: 潜在的乙酰化位点位置 [batch_size]，默认为序列中间位置
            
        Returns:
            torch.Tensor: 乙酰化相关特征 [batch_size, output_dim]
        """
        batch_size, seq_len = aa_sequence.shape
        device = aa_sequence.device
        
        # 计算默认位置(序列中央)如果未提供
        if positions is None:
            positions = torch.tensor([seq_len // 2] * batch_size, device=device)
        
        # 获取氨基酸嵌入
        aa_embeddings = self.aa_embedding(aa_sequence)  # [batch, seq_len, hidden_dim]
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, max=self.max_seq_len-1)
        pos_embeddings = self.position_encoding(position_ids)  # [batch, seq_len, hidden_dim]
        
        # 组合嵌入
        embeddings = aa_embeddings + pos_embeddings  # [batch, seq_len, hidden_dim]
        
        # 提取中心位点特征
        site_indices = torch.arange(batch_size, device=device)
        center_features = embeddings[site_indices, positions]  # [batch, hidden_dim]
        
        # 编码中心位点特征
        site_features = self.site_encoder(center_features)  # [batch, hidden_dim]
        
        # 创建注意力掩码以突出中心位置
        attn_mask = torch.ones(batch_size, seq_len, device=device)
        for i, pos in enumerate(positions):
            # 中心位置的重要性是1，其余随距离衰减
            distances = torch.abs(torch.arange(seq_len, device=device) - pos)
            attn_mask[i] = torch.exp(-distances.float() / 10)
        
        # 应用上下文编码器，考虑中心位置掩码
        context_features = self.context_encoder(embeddings)  # [batch, seq_len, hidden_dim]
        
        # 加权聚合上下文特征
        weighted_context = context_features * attn_mask.unsqueeze(-1)  # [batch, seq_len, hidden_dim]
        context_sum = weighted_context.sum(dim=1)  # [batch, hidden_dim]
        context_weight_sum = attn_mask.sum(dim=1, keepdim=True)  # [batch, 1]
        aggregated_context = context_sum / context_weight_sum  # [batch, hidden_dim]
        
        # 合并位点和上下文特征
        combined_features = site_features + aggregated_context  # [batch, hidden_dim]
        
        # 投影到输出维度
        output = self.output_projection(combined_features)  # [batch, output_dim]
        
        return output

class ProjectionLayer(nn.Module):
    """
    投影层: 将不同编码器的输出统一到相同的表示空间
    优化后可处理序列、结构和乙酰化信息的融合
    """
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        # [Modified] Enhanced projection with intermediate layer to avoid information bottleneck
        # and increase non-linearity as requested.
        # Strategy: Retain high-dimensional features longer by projecting to an intermediate dimension
        # instead of directly to the small output dimension.
        mid_dim = max(output_dim, input_dim // 2)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """将输入特征投影到统一的表示空间"""
        return self.projection(x)

class MultiModalFusion(nn.Module):
    """
    多模态融合层: 融合序列、结构和乙酰化信息
    使用注意力机制对不同模态进行加权融合
    """
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 多头自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # 注意力池化层 (用于加权融合)
        self.attn_pool = nn.Linear(feature_dim, 1)
        
    def forward(self, *features):
        """
        前向传播: 融合多个特征
        
        Args:
            *features: 可变数量的特征张量，每个形状为 [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, seq_len, feature_dim]
        """
        # 检查有效的特征输入
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            raise ValueError("至少需要提供一种有效的特征")
        
        if len(valid_features) == 1:
            # 只有一种特征时直接返回
            return valid_features[0]
            
        # 堆叠特征 [batch_size, num_features, seq_len, feature_dim]
        stacked = torch.stack(valid_features, dim=1)
        batch_size, num_features, seq_len, feature_dim = stacked.shape
        
        # 重塑为适合注意力层的形状 [batch_size * seq_len, num_features, feature_dim]
        reshaped = stacked.transpose(1, 2).reshape(batch_size * seq_len, num_features, feature_dim)
        
        # 自注意力融合 (模态间交互)
        attn_output, _ = self.attention(reshaped, reshaped, reshaped)
        
        # 残差连接和层归一化
        norm_output = self.layer_norm1(reshaped + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(norm_output)
        
        # 残差连接和层归一化
        output = self.layer_norm2(norm_output + ffn_output)
        
        # [Modified] 使用注意力池化代替平均池化
        # 计算每个模态的权重 score: [batch*seq, num_features, 1]
        scores = self.attn_pool(output)
        weights = F.softmax(scores, dim=1)
        
        # 加权求和
        fused = (output * weights).sum(dim=1)
        
        # 重塑回原始形状 [batch_size, seq_len, feature_dim]
        fused = fused.reshape(batch_size, seq_len, feature_dim)
        
        return fused
