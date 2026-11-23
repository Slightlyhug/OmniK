import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from collections import OrderedDict
import logging

# 设置日志
logger = logging.getLogger("meta_learning")


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation."""
    
    def __init__(self, base_model, inner_lr=0.01, inner_steps=5):
        """
        Initialize MAML.
        
        Args:
            base_model (nn.Module): Base model to be meta-trained.
            inner_lr (float): Inner loop learning rate.
            inner_steps (int): Number of inner loop steps.
        """
        super().__init__()
        
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def forward(self, support_data, query_data):
        """
        Forward pass for meta-learning.
        
        Args:
            support_data (tuple): Support set (inputs, targets).
            query_data (tuple): Query set (inputs, targets).
            
        Returns:
            tuple: (query_predictions, query_loss)
        """
        support_inputs, support_targets = support_data
        query_inputs, query_targets = query_data
        
        # 确保目标标签的形状正确
        if support_targets.dim() == 1:
            support_targets = support_targets.view(-1, 1)
        if query_targets.dim() == 1:
            query_targets = query_targets.view(-1, 1)
        
        # Clone the base model for adaptation
        adapted_model = self.adapt_to_task(support_inputs, support_targets)
        
        # Evaluate on query set
        query_predictions = adapted_model(query_inputs)
        
        # 确保预测和目标的形状匹配
        if query_predictions.shape != query_targets.shape:
            if query_predictions.dim() > query_targets.dim():
                query_predictions = query_predictions.squeeze(-1)
            else:
                query_targets = query_targets.reshape(query_targets.shape[0], 1)
        
        query_loss = F.binary_cross_entropy_with_logits(query_predictions, query_targets)
        
        return query_predictions, query_loss
    
    def adapt_to_task(self, inputs, targets):
        """
        Adapt the model to a specific task.
        
        Args:
            inputs: Task inputs.
            targets: Task targets.
            
        Returns:
            nn.Module: Adapted model.
        """
        # Create a clone of the base model
        adapted_model = copy.deepcopy(self.base_model)
        
        # 确保目标标签的形状与模型输出匹配
        if targets.dim() == 1:
            targets = targets.view(-1, 1)
        
        # 检查输入是否有梯度
        requires_grad = inputs.requires_grad
        
        # 如果输入没有梯度，需要克隆并使其能够计算梯度
        if not requires_grad:
            inputs = inputs.detach().clone().requires_grad_(True)
            
        # 记录原始训练模式
        training_mode = adapted_model.training
        
        # 临时设置为训练模式以进行适配
        if not training_mode:
            adapted_model.train()
        
        try:
            # Create a list of trainable parameters
            adapted_params = list(adapted_model.parameters())
            
            # 使用梯度上下文
            with torch.enable_grad():
                # Inner loop optimization
                for step in range(self.inner_steps):
                    # Forward pass
                    predictions = adapted_model(inputs)
                    
                    # 确保预测和目标的形状匹配
                    if predictions.shape != targets.shape:
                        if predictions.dim() > targets.dim():
                            # 如果预测多一个维度，压缩它
                            predictions = predictions.squeeze(-1)
                        else:
                            # 如果目标需要扩展
                            targets = targets.reshape(targets.shape[0], 1)
                    
                    loss = F.binary_cross_entropy_with_logits(predictions, targets)
                    
                    # Compute gradients
                    gradients = torch.autograd.grad(
                        loss, adapted_params, create_graph=True, allow_unused=True
                    )
                    
                    # Update parameters manually
                    adapted_params = [
                        p - self.inner_lr * g if g is not None else p
                        for p, g in zip(adapted_params, gradients)
                    ]
                    
                    # Update model parameters
                    for param, adapted_param in zip(adapted_model.parameters(), adapted_params):
                        param.data = adapted_param
                        
                # 创建适配后的分类器包装器，保持接口一致性
                class AdaptedModel(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, x):
                        pred = self.model(x)
                        # 确保输出形状一致
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(1)
                        return pred
                
                adapted_classifier = AdaptedModel(adapted_model)
        finally:
            # 恢复原始训练模式
            if not training_mode:
                adapted_model.eval()
        
        return adapted_classifier


class ProtoNet(nn.Module):
    """Prototypical Network implementation."""
    
    def __init__(self, encoder):
        """
        Initialize ProtoNet.
        
        Args:
            encoder (nn.Module): Feature encoder network.
        """
        super().__init__()
        
        self.encoder = encoder
        
    def forward(self, support_data, query_data):
        """
        Forward pass.
        
        Args:
            support_data (tuple): Support set (inputs, targets).
            query_data (tuple): Query set (inputs, targets).
            
        Returns:
            tuple: (query_predictions, query_loss)
        """
        support_inputs, support_targets = support_data
        query_inputs, query_targets = query_data
        
        # 确保目标标签的形状正确
        if support_targets.dim() == 1:
            support_targets = support_targets.view(-1, 1)
        if query_targets.dim() == 1:
            query_targets = query_targets.view(-1, 1)
        
        # Encode support and query inputs
        support_features = self.encoder(support_inputs)
        query_features = self.encoder(query_inputs)
        
        # Get unique classes
        classes = torch.unique(support_targets)
        n_classes = len(classes)
        
        # Compute prototypes
        prototypes = torch.zeros(n_classes, support_features.shape[1], device=support_features.device)
        
        for i, c in enumerate(classes):
            # Get features for class c
            class_features = support_features[support_targets == c]
            # Compute prototype as mean of features
            prototypes[i] = class_features.mean(dim=0)
        
        # Compute distances from query features to prototypes
        dists = torch.cdist(query_features, prototypes)
        
        # Convert distances to probabilities
        logits = -dists
        
        # 确保query_targets的形状与交叉熵计算兼容
        # 对于原型网络，期望target是类别索引的形式
        query_targets_flat = query_targets.squeeze()
        
        # Compute loss
        query_loss = F.cross_entropy(logits, query_targets_flat)
        
        return logits, query_loss
        
    def adapt_to_task(self, support_inputs, support_targets):
        """
        适配模型到当前任务。
        对于ProtoNet，这会返回一个基于原型的分类器
        
        Args:
            support_inputs: 支持集输入
            support_targets: 支持集标签
            
        Returns:
            nn.Module: 适配后的分类器
        """
        # 确保支持集标签形状正确
        if support_targets.dim() == 1:
            support_targets = support_targets.view(-1, 1)
            
        # 记录原始训练状态
        training_mode = self.encoder.training
        
        # 临时设置为eval模式确保编码稳定
        if training_mode:
            self.encoder.eval()
            
        try:
            # 编码支持集
            with torch.no_grad():  # 在eval模式下，使用无梯度上下文提高效率
                support_features = self.encoder(support_inputs)
            
            # 获取唯一类别
            unique_targets = support_targets.squeeze()
            classes = torch.unique(unique_targets)
            
            # 计算每个类的原型
            prototypes = []
            for c in classes:
                # 获取类c的索引
                mask = (unique_targets == c)
                # 获取类c的特征
                class_features = support_features[mask]
                # 如果该类至少有一个样本
                if class_features.size(0) > 0:
                    # 计算原型为特征均值
                    prototype = class_features.mean(dim=0)
                    prototypes.append(prototype)
                
            # 如果没有足够的原型（至少两个类），创建虚拟类
            if len(prototypes) < 2:
                if len(prototypes) == 1:
                    # 添加一个相反的原型（负类）
                    neg_prototype = -prototypes[0]
                    prototypes.append(neg_prototype)
                else:
                    # 如果完全没有原型，创建两个随机原型
                    dim = support_features.size(1) if support_features.size(0) > 0 else 128
                    prototypes = [
                        torch.randn(dim, device=support_inputs.device),
                        torch.randn(dim, device=support_inputs.device)
                    ]
            
            prototypes = torch.stack(prototypes)
            
            # 创建基于原型的分类器
            class ProtoClassifier(nn.Module):
                def __init__(self, encoder, prototypes):
                    super().__init__()
                    self.encoder = encoder
                    self.register_buffer('prototypes', prototypes)
                    
                def forward(self, x):
                    # 获取输入嵌入
                    with torch.no_grad():  # 使用无梯度上下文提高推理效率
                        embeddings = self.encoder(x)
                    # 计算到原型的距离
                    dists = torch.cdist(embeddings, self.prototypes)
                    # 将距离转换为logits（负距离）
                    logits = -dists
                    # 对于二分类，转换为适合BCEWithLogits的形式
                    if self.prototypes.size(0) == 2:
                        # 返回正类的logit，形状为[batch_size, 1]
                        return logits[:, 1].unsqueeze(1)
                    return logits
        
        finally:
            # 恢复原始训练状态
            if training_mode:
                self.encoder.train()
                
        return ProtoClassifier(self.encoder, prototypes)


class MetaTaskSampler:
    """Sampler for generating meta-learning tasks."""
    
    def __init__(self, dataset, n_way=2, k_shot=5, n_query=15, n_tasks=100):
        """
        Initialize the meta-task sampler.
        
        Args:
            dataset: Dataset to sample from.
            n_way (int): Number of classes per task.
            k_shot (int): Number of support examples per class.
            n_query (int): Number of query examples per class.
            n_tasks (int): Number of tasks to generate.
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        
        # Group data by class
        self.class_indices = self._group_by_class()
        
    def _group_by_class(self):
        """Group dataset indices by class."""
        class_indices = {}
        
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            
            # Extract label - handle both dict and custom formats
            if isinstance(item, dict) and 'label' in item:
                label = item['label'].item() if torch.is_tensor(item['label']) else item['label']
            elif hasattr(item, 'label'):
                label = item.label.item() if torch.is_tensor(item.label) else item.label
            else:
                # Default fallback - try to get the second item (common in datasets with (data, label) format)
                try:
                    label = item[1].item() if torch.is_tensor(item[1]) else item[1]
                except:
                    raise ValueError(f"Could not extract label from dataset item: {item}")
            
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
            
        return class_indices
    
    def sample_task(self):
        """
        Sample a meta-learning task.
        
        Returns:
            tuple: (support_batch, query_batch)
        """
        # Sample n_way classes
        all_classes = list(self.class_indices.keys())
        if len(all_classes) < self.n_way:
            raise ValueError(f"Not enough classes in dataset. Found {len(all_classes)}, need {self.n_way}")
            
        classes = np.random.choice(all_classes, self.n_way, replace=False)
        
        support_indices = []
        query_indices = []
        
        for c in classes:
            class_indices = self.class_indices[c]
            if len(class_indices) < (self.k_shot + self.n_query):
                # Fallback: sample with replacement if not enough samples
                indices = np.random.choice(class_indices, self.k_shot + self.n_query, replace=True)
            else:
                indices = np.random.choice(class_indices, self.k_shot + self.n_query, replace=False)
            
            # Split into support and query
            support_indices.extend(indices[:self.k_shot])
            query_indices.extend(indices[self.k_shot:])
        
        # Create support and query sets
        support_data = [self.dataset[i] for i in support_indices]
        query_data = [self.dataset[i] for i in query_indices]
        
        # Create batch dictionaries for consistency
        support_batch = self._create_batch_dict(support_data)
        query_batch = self._create_batch_dict(query_data)
        
        return support_batch, query_batch
    
    def _create_batch_dict(self, data_list):
        """
        Convert a list of data items into a consistent batch dictionary format.
        This allows both dictionary and tuple dataset formats to work with the trainer.
        
        Args:
            data_list: List of data items from the dataset
            
        Returns:
            dict: Batch dictionary with 'sequence', 'structure' (optional), and 'label' keys
        """
        # Handle empty list
        if not data_list:
            return {'sequence': None, 'label': None}
        
        # Check the format of data items
        sample_item = data_list[0]
        
        # If items are already dictionaries with the expected structure
        if isinstance(sample_item, dict) and 'label' in sample_item:
            # Combine individual dicts into a batch dict
            batch = {}
            for key in sample_item.keys():
                if all(key in item for item in data_list):
                    # For tensors, stack them
                    if torch.is_tensor(sample_item[key]):
                        batch[key] = torch.stack([item[key] for item in data_list])
                    elif isinstance(sample_item[key], dict):
                        # Handle nested dicts (like sequence features)
                        batch[key] = {}
                        for subkey in sample_item[key].keys():
                            # 检查嵌套字典项是否为None
                            if all(key in item and item[key] is not None and 
                                   isinstance(item[key], dict) and subkey in item[key] 
                                   for item in data_list):
                                if torch.is_tensor(sample_item[key][subkey]):
                                    batch[key][subkey] = torch.stack([item[key][subkey] for item in data_list])
                                else:
                                    batch[key][subkey] = [item[key][subkey] for item in data_list]
                    else:
                        # For non-tensors, just create a list
                        batch[key] = [item[key] for item in data_list]
            return batch
                
        # For tuple-like items (data, label)
        elif isinstance(sample_item, tuple) or (isinstance(sample_item, list) and len(sample_item) >= 2):
            # Extract data and labels
            features = [item[0] for item in data_list]
            labels = [item[1] for item in data_list]
            
            # Convert to tensors if needed
            if not torch.is_tensor(features[0]):
                features = [torch.tensor(f) for f in features]
            if not torch.is_tensor(labels[0]):
                labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
                
            # Stack tensors
            features_tensor = torch.stack(features)
            labels_tensor = torch.stack(labels)
            
            # Create batch dictionary
            return {
                'sequence': {'window_features': features_tensor},
                'label': labels_tensor
            }
            
        # For custom objects with attributes
        elif hasattr(sample_item, 'features') and hasattr(sample_item, 'label'):
            features = [item.features for item in data_list]
            labels = [item.label for item in data_list]
            
            # Convert to tensors if needed
            if not torch.is_tensor(features[0]):
                features = [torch.tensor(f) for f in features]
            if not torch.is_tensor(labels[0]):
                labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
                
            # Stack tensors
            features_tensor = torch.stack(features)
            labels_tensor = torch.stack(labels)
            
            # Create batch dictionary
            return {
                'sequence': {'window_features': features_tensor},
                'label': labels_tensor
            }
            
        # Fallback for unrecognized formats
        else:
            raise ValueError(f"Unsupported dataset item format: {type(sample_item)}")
    
    def __iter__(self):
        """Iterate over tasks."""
        for _ in range(self.n_tasks):
            yield self.sample_task()


class BERTBiLSTMMetaEncoder(nn.Module):
    """
    基于BERT+BiLSTM架构的元学习编码器

    专门为蛋白质翻译后修饰(PTMs)的元学习任务设计，结合了：
    - BERT的长程依赖建模能力
    - BiLSTM的局部序列特征提取
    - 元学习的快速适应机制
    """

    def __init__(self, input_dim, bert_hidden_dim=768, lstm_hidden_dim=128,
                 output_dim=256, dropout=0.1):
        """
        初始化BERT+BiLSTM元学习编码器

        Args:
            input_dim (int): 输入特征维度
            bert_hidden_dim (int): BERT隐藏层维度
            lstm_hidden_dim (int): LSTM隐藏层维度
            output_dim (int): 输出特征维度
            dropout (float): Dropout概率
        """
        super().__init__()

        self.input_dim = input_dim
        self.bert_hidden_dim = bert_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.output_dim = output_dim

        logger.info(f"初始化BERT+BiLSTM元学习编码器")
        logger.info(f"- 输入维度: {input_dim}")
        logger.info(f"- BERT隐藏维度: {bert_hidden_dim}")
        logger.info(f"- LSTM隐藏维度: {lstm_hidden_dim}")
        logger.info(f"- 输出维度: {output_dim}")

        # 1. 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. BERT风格编码器（用于长程依赖）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_hidden_dim,
            nhead=8,
            dim_feedforward=bert_hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.bert_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. BiLSTM层（用于局部序列特征）
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # 单层LSTM不使用dropout
        )

        # BiLSTM输出维度
        self.lstm_output_dim = lstm_hidden_dim * 2

        # 4. 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_hidden_dim + self.lstm_output_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. 注意力池化层
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 6. 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(bert_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        logger.info("BERT+BiLSTM元学习编码器初始化完成")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: 编码后的特征 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影
        projected_input = self.input_projection(x)  # [batch_size, seq_len, bert_hidden_dim]

        # 2. BERT编码
        bert_features = self.bert_encoder(projected_input)  # [batch_size, seq_len, bert_hidden_dim]

        # 3. BiLSTM编码
        lstm_features, _ = self.bilstm(bert_features)  # [batch_size, seq_len, lstm_output_dim]

        # 4. 特征融合
        combined_features = torch.cat([bert_features, lstm_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, bert_hidden_dim]

        # 5. 注意力池化
        # 使用自注意力进行全局池化
        attended_features, attention_weights = self.attention_pooling(
            fused_features, fused_features, fused_features
        )

        # 全局平均池化
        pooled_features = attended_features.mean(dim=1)  # [batch_size, bert_hidden_dim]

        # 6. 输出投影
        output_features = self.output_projection(pooled_features)  # [batch_size, output_dim]

        return output_features


class BERTBiLSTMMAML(MAML):
    """
    基于BERT+BiLSTM架构的MAML实现

    继承自原始MAML类，但使用BERT+BiLSTM编码器作为基础模型
    """

    def __init__(self, input_dim, output_dim=1, bert_hidden_dim=768,
                 lstm_hidden_dim=128, inner_lr=0.01, inner_steps=5):
        """
        初始化BERT+BiLSTM MAML

        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出维度（通常为1，用于二分类）
            bert_hidden_dim (int): BERT隐藏层维度
            lstm_hidden_dim (int): LSTM隐藏层维度
            inner_lr (float): 内循环学习率
            inner_steps (int): 内循环步数
        """
        # 创建基础模型
        encoder = BERTBiLSTMMetaEncoder(
            input_dim=input_dim,
            bert_hidden_dim=bert_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            output_dim=256  # 编码器输出维度
        )

        # 添加分类头
        classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

        # 组合完整模型
        base_model = nn.Sequential(encoder, classifier_head)

        # 调用父类初始化
        super().__init__(base_model, inner_lr, inner_steps)

        logger.info(f"初始化BERT+BiLSTM MAML模型")
        logger.info(f"- 输入维度: {input_dim}")
        logger.info(f"- 输出维度: {output_dim}")
        logger.info(f"- 内循环学习率: {inner_lr}")
        logger.info(f"- 内循环步数: {inner_steps}")


class BERTBiLSTMProtoNet(ProtoNet):
    """
    基于BERT+BiLSTM架构的原型网络实现

    继承自原始ProtoNet类，但使用BERT+BiLSTM编码器
    """

    def __init__(self, input_dim, bert_hidden_dim=768, lstm_hidden_dim=128,
                 embedding_dim=256):
        """
        初始化BERT+BiLSTM原型网络

        Args:
            input_dim (int): 输入特征维度
            bert_hidden_dim (int): BERT隐藏层维度
            lstm_hidden_dim (int): LSTM隐藏层维度
            embedding_dim (int): 嵌入维度
        """
        # 创建编码器
        encoder = BERTBiLSTMMetaEncoder(
            input_dim=input_dim,
            bert_hidden_dim=bert_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            output_dim=embedding_dim
        )

        # 调用父类初始化
        super().__init__(encoder)

        self.embedding_dim = embedding_dim

        logger.info(f"初始化BERT+BiLSTM原型网络")
        logger.info(f"- 输入维度: {input_dim}")
        logger.info(f"- 嵌入维度: {embedding_dim}")
        logger.info(f"- BERT隐藏维度: {bert_hidden_dim}")
        logger.info(f"- LSTM隐藏维度: {lstm_hidden_dim}")

    def compute_prototypes(self, support_features, support_labels):
        """
        计算类原型

        Args:
            support_features: 支持集特征 [n_support, embedding_dim]
            support_labels: 支持集标签 [n_support]

        Returns:
            torch.Tensor: 类原型 [n_classes, embedding_dim]
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)

        prototypes = torch.zeros(n_classes, self.embedding_dim,
                               device=support_features.device)

        for i, label in enumerate(unique_labels):
            # 获取当前类的所有样本
            class_mask = (support_labels == label)
            class_features = support_features[class_mask]

            # 计算原型（类内样本的均值）
            prototypes[i] = class_features.mean(dim=0)

        return prototypes, unique_labels

    def forward(self, support_data, query_data):
        """
        前向传播（重写以支持BERT+BiLSTM特征）

        Args:
            support_data: 支持集数据 (inputs, targets)
            query_data: 查询集数据 (inputs, targets)

        Returns:
            tuple: (query_predictions, query_loss)
        """
        support_inputs, support_targets = support_data
        query_inputs, query_targets = query_data

        # 确保标签形状正确
        if support_targets.dim() == 1:
            support_targets = support_targets.view(-1, 1)
        if query_targets.dim() == 1:
            query_targets = query_targets.view(-1, 1)

        # 编码支持集和查询集
        support_features = self.encoder(support_inputs)  # [n_support, embedding_dim]
        query_features = self.encoder(query_inputs)      # [n_query, embedding_dim]

        # 计算原型
        prototypes, unique_labels = self.compute_prototypes(
            support_features, support_targets.squeeze()
        )

        # 计算查询样本到原型的距离
        distances = torch.cdist(query_features, prototypes)  # [n_query, n_classes]

        # 转换为logits（负距离）
        logits = -distances

        # 计算损失
        if len(unique_labels) == 2:
            # 二分类情况
            query_loss = F.binary_cross_entropy_with_logits(
                logits[:, 1] - logits[:, 0],
                query_targets.squeeze().float()
            )
            predictions = torch.sigmoid(logits[:, 1] - logits[:, 0])
        else:
            # 多分类情况
            query_loss = F.cross_entropy(logits, query_targets.squeeze().long())
            predictions = F.softmax(logits, dim=1)

        return predictions, query_loss