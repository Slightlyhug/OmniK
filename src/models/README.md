# PTM-GPT-MoE: 高级学习模块

## Production multi-task predictor

The multi-task training pipeline imports `DualBranchFusionPredictor` through
`dual_branch_predictor.py` and wraps it with the task heads in
`multitask_heads.py`. The implementation remains in `acetylation_predictor.py`
to preserve existing checkpoint and external import paths. The acetylation-only
predictors and `MultiTaskLoss` are isolated in `legacy_predictors.py`; old
imports remain available as lazy exports.

The M6 dense-on and dense-off runs rely on their config-selected structure
graph, task separation, PTM specialist, and task attention modules. Those
components remain in dedicated modules and are constructed only when enabled.

本目录包含用于蛋白质翻译后修饰(PTM)预测的高级学习模块，包括强化学习、因果推断和元学习。这些模块与基本的特征融合架构相结合，可以提高预测性能并提供更深入的机理理解。

## 模块概述

### 1. 强化学习模块 (`reinforcement.py`)

强化学习模块使用策略梯度方法，特别是PPO (Proximal Policy Optimization) 算法，优化序列位置决策过程。该模块可以:

- 将预测问题视为序列决策过程
- 学习从蛋白质特征到修饰位点的策略映射
- 通过奖励信号优化预测准确性
- 有效处理稀疏或不平衡数据

### 2. 因果推断模块 (`causal.py`) 

因果推断模块应用因果模型理解氨基酸特征与修饰之间的因果关系。该模块包括:

- 结构因果模型 (SCM)，通过潜变量学习氨基酸特征与修饰之间的因果关系
- DoWhy风格的因果效应估计器，使用反事实推理评估特征对修饰的因果影响
- 通过潜在的混杂因子估计平均处理效应 (ATE)
- 提供特征因果重要性分析

### 3. 元学习模块 (`meta_learning.py`)

元学习模块通过学习快速适应新修饰类型的能力提高泛化性能。该模块实现了:

- 模型无关元学习 (MAML)，用于通过梯度优化快速适应新任务
- 原型网络 (ProtoNet)，用于基于样本间度量学习
- 支持少样本学习场景，适用于数据稀缺的修饰类型
- 改进跨PTM类型的知识转移

## 整合方法

高级学习模块可以通过三种方式与基础模型整合:

1. **串行整合** (Serial Integration)：模块按顺序处理，每个模块的输出传递给下一个模块，建立处理管道。
2. **集成整合** (Ensemble Integration)：各模块独立作出预测，然后通过平均或加权组合生成最终预测。
3. **动态整合** (Dynamic Integration)：使用学习到的选择器机制，根据输入特征动态为每个样本选择最合适的模块。

## 使用方法

### 模型训练

使用 `train_with_rl.py` 脚本训练带有高级学习模块的模型:

```bash
python src/train_with_rl.py \
  --data_path=<数据CSV路径> \
  --seq_features_dir=<序列特征目录> \
  --struct_features_path=<结构特征文件> \
  --output_dir=<输出目录> \
  --use_reinforcement \
  --use_causal \
  --use_meta_learning
```

## 核心架构: 双分支融合 (Dual Branch Fusion)

核心预测模型采用 **双分支融合** 架构 (`DualBranchFusionPredictor`，位于 `src/models/acetylation_predictor.py`)，并行处理蛋白质序列和3D结构：

1. **序列分支**: 使用 `SequenceTower` (基于Transformer) 编码氨基酸序列。
2. **结构分支**: 使用 `StructureTower` (基于Transformer) 编码结构特征 (坐标, 角度)。
3. **融合模块**: 使用交叉注意力 (`CrossModalFusionLayer`) 或门控融合机制结合特征。
4. **混合专家 (MoE)**: (可选) 支持MoE架构，用于专门处理多样化的PTM类型。

## 数据处理

数据处理由 `UnifiedDataProcessor` (`src/data/unified_data_processor.py`) 统一管理，支持:
- 单一数据集或分离的训练/测试数据集。
- HDF5特征缓存，实现高效训练。
- 自动PTM类型标准化和防泄漏机制。
- 支持原始FASTA/PDB作为回退选项。

### 模型比较

使用 `run_advanced_model_comparison.py` 脚本运行多个模型配置并比较它们的性能:

```bash
python PTM/src/run_advanced_model_comparison.py \
  --data_path=<数据CSV路径> \
  --seq_features_dir=<序列特征目录> \
  --struct_features_path=<结构特征文件> \
  --output_dir=<输出目录>
```

默认情况下，该脚本将运行10种不同的模型配置组合，包括基线模型、单模块模型、多模块组合以及不同的整合方法。您可以使用 `--selective` 和 `--configs` 参数选择特定的模型配置运行。

## 理论基础

### 强化学习在PTM预测中的应用

强化学习模型将修饰位点预测视为序列决策问题。与传统的监督学习方法相比，RL能够:
- 考虑长期依赖关系和序列内上下文
- 适应稀疏奖励情况（如罕见的修饰位点）
- 通过探索-利用权衡发现新的模式

### 因果推断的重要性

因果模型超越了简单的相关性，能够:
- 识别混杂因素，如蛋白质家族或细胞位置
- 估计特征的真实因果效应，而不仅仅是统计关联
- 通过反事实推理提供更稳健的解释
- 在数据分布变化时（如进化上不同的蛋白质）保持较好性能

### 元学习的跨PTM泛化

元学习对PTM预测特别有用，因为:
- 不同类型的PTM可以被视为不同但相关的任务
- 元学习允许从数据丰富的修饰类型向数据稀缺的修饰类型转移知识
- 通过共享表示学习实现更高效的少样本学习
- 改进对新蛋白质或新实验数据的适应性

## 可视化和解释

系统提供多种可视化工具，帮助解释预测结果:

1. **因果效应分析**：展示氨基酸特征对修饰的因果影响
2. **特征重要性热图**：显示各特征对预测的贡献
3. **模型性能比较**：比较不同模块组合的预测性能
4. **序列位置决策**：可视化RL模型的决策过程

## 性能基准

根据我们的实验，不同模块组合在乙酰化预测任务上的性能基准如下（结果可能因数据集和超参数选择而变化）:

| 模型配置 | 测试准确率 | 测试F1 | 测试AUC | 备注 |
|---------|---------|-------|---------|------|
| 基线 | ~0.92 | ~0.68 | ~0.85 | 只使用特征融合 |
| +强化学习 | ~0.93 | ~0.70 | ~0.87 | 改进位点选择 |
| +因果推断 | ~0.94 | ~0.72 | ~0.89 | 识别因果特征 |
| +元学习 | ~0.93 | ~0.71 | ~0.88 | 改进泛化能力 |
| 全模块(串行) | ~0.95 | ~0.74 | ~0.91 | 最佳组合 |
| 全模块(集成) | ~0.94 | ~0.73 | ~0.90 | 稳定性更好 |
| 全模块(动态) | ~0.95 | ~0.75 | ~0.92 | 自适应性最佳 |

## 扩展与未来工作

此框架设计为可扩展的，可以进一步增强:

1. 添加更多高级模块，如图神经网络、对比学习
2. 改进模块间的知识蒸馏机制
3. 探索更多PTM类型的迁移学习
4. 整合额外的多模态数据，如进化信息或功能注释

## 引用

如果您使用了这些模块，请引用我们的工作:

```
待添加
``` 



根据代码中的实现，以下是各种模块组合的训练指令:
单一模块训练:

仅使用强化学习模块:
     python src/train_advanced_model.py --use_reinforcement --batch_size 64
Apply to README.md
仅使用因果推断模块:
     python src/train_advanced_model.py --use_causal --batch_size 64
Apply to README.md
仅使用元学习模块:
     python src/train_advanced_model.py --use_meta_learning --batch_size 64
Apply to README.md
仅使用混合专家模块(MoE):
     python src/train_advanced_model.py --use_moe --batch_size 64
Apply to README.md

模块组合训练:
使用串行整合方法(默认):
     python src/train_advanced_model.py --use_reinforcement --use_causal --integration_method serial --batch_size 64
Apply to README.md
使用集成整合方法:
     python src/train_advanced_model.py --use_meta_learning --use_moe --integration_method ensemble --batch_size 64
Apply to README.md
使用动态整合方法:
     python src/train_advanced_model.py --use_reinforcement --use_meta_learning --integration_method dynamic --batch_size 64
Apply to README.md
多模块组合的特定例子:
强化学习+因果推断+元学习(串行):
     python src/train_advanced_model.py --use_reinforcement --use_causal --use_meta_learning --integration_method serial --batch_size 64
Apply to README.md
强化学习+因果推断+元学习+MoE(集成):
     python src/train_advanced_model.py --use_reinforcement --use_causal --use_meta_learning --use_moe --integration_method ensemble --batch_size 64
Apply to README.md
当使用脚本train_advanced.sh时，可以通过以下方式调用这些组合:
Apply to README.md
其中:
combined 表示使用多模块组合
模块名称用逗号分隔(rl, causal, meta, moe)
整合方法可以是 serial, ensemble 或 dynamic
-b 参数设置batch size
这些命令会自动设置相应的参数，并在输出目录名称中反映所使用的模块组合和整合方法。
./train_advanced.sh combined rl,causal serial -b 64
./train_advanced.sh combined meta,moe ensemble -b 64
./train_advanced.sh combined rl,meta,causal,moe dynamic -b 128


根据代码实现，集成整合(ensemble)和串行整合(serial)的主要区别如下：
串行整合(Serial Integration)
工作原理：
模块按顺序依次处理数据
一个模块的输出作为下一个模块的输入
最后一个模块的输出作为最终预测结果
实现方式：
在模型前向传播时，按照预定顺序调用每个模块
例如：先通过因果模块，然后是RL模块，最后是元学习模块
优势：
每个模块可以专注于优化自己的任务
模块间存在明确的依赖关系
便于理解和调试每个模块的贡献
劣势：
早期模块的错误会传播到后续模块
训练比较复杂，需要小心处理模块间的交互
集成整合(Ensemble Integration)
工作原理：
所有模块并行处理相同的输入数据
每个模块独立生成预测结果
最后通过加权组合或投票等方式融合各模块结果
实现方式：
在模型前向传播时，并行调用所有模块
通过某种融合机制（如加权平均）合并各模块输出
返回一个综合的预测结果
优势：
模块间相互独立，错误不会传播
可以结合多种不同类型模型的优势
通常比单一模块具有更好的泛化能力
劣势：
计算成本更高，需要维护多个模型
融合策略的选择会影响最终性能
