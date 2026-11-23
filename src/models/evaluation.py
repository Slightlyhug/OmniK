import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    precision_recall_curve, auc, roc_curve
)


class EvaluationMetrics:
    """用于评估蛋白质乙酰化位点预测模型的评估指标类"""
    
    def __init__(self):
        """初始化评估指标类"""
        pass
        
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
        """
        计算评估指标
        
        Args:
            y_true (numpy.ndarray): 真实标签
            y_pred (numpy.ndarray): 预测标签
            y_scores (numpy.ndarray, optional): 预测概率值
            threshold (float): 分类阈值
            
        Returns:
            dict: 包含多种评估指标的字典
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 如果提供的是分数而非二分类标签，则应用阈值
        if y_scores is not None:
            y_scores = np.array(y_scores)
            binary_preds = (y_scores >= threshold).astype(int)
        else:
            binary_preds = y_pred
            y_scores = y_pred  # 如果没有提供分数，使用预测标签作为分数
        
        # 计算基本指标
        metrics = {
            'accuracy': accuracy_score(y_true, binary_preds),
            'precision': precision_score(y_true, binary_preds, zero_division=0),
            'recall': recall_score(y_true, binary_preds, zero_division=0),
            'f1': f1_score(y_true, binary_preds, zero_division=0),
            'mcc': matthews_corrcoef(y_true, binary_preds),
        }
        
        # 计算AUC (如果提供了概率分数)
        if not np.all(np.isin(y_scores, [0, 1])):
            try:
                metrics['auc'] = roc_auc_score(y_true, y_scores)
                
                # 计算AUPRC
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                metrics['auprc'] = auc(recall, precision)
            except:
                metrics['auc'] = 0.5
                metrics['auprc'] = 0.5
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, binary_preds, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    @staticmethod
    def threshold_optimization(y_true, y_scores, metric='f1'):
        """
        寻找最佳分类阈值
        
        Args:
            y_true (numpy.ndarray): 真实标签
            y_scores (numpy.ndarray): 预测概率值
            metric (str): 优化指标名称 ('f1', 'mcc', 'balanced_accuracy')
            
        Returns:
            float: 最佳阈值
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            # 将概率值转换为二分类预测
            y_pred = (y_scores >= threshold).astype(int)
            
            # 根据所选指标计算分数
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            elif metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            # 更新最佳分数和阈值
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold
    
    @staticmethod
    def bootstrap_confidence_interval(y_true, y_scores, n_bootstraps=1000, ci=0.95, metric='auc'):
        """
        使用Bootstrap方法计算置信区间
        
        Args:
            y_true (numpy.ndarray): 真实标签
            y_scores (numpy.ndarray): 预测概率值
            n_bootstraps (int): Bootstrap样本数量
            ci (float): 置信水平
            metric (str): 评估指标名称
            
        Returns:
            tuple: (下限, 上限) 置信区间
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        bootstrap_scores = []
        rng = np.random.RandomState(42)  # 固定随机种子以确保可重现性
        
        for i in range(n_bootstraps):
            # 随机抽样（有放回）
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                # 跳过只有一个类别的样本
                continue
                
            # 计算度量值
            if metric == 'auc':
                score = roc_auc_score(y_true[indices], y_scores[indices])
            elif metric == 'mcc':
                y_pred = (y_scores[indices] >= 0.5).astype(int)
                score = matthews_corrcoef(y_true[indices], y_pred)
            elif metric == 'f1':
                y_pred = (y_scores[indices] >= 0.5).astype(int)
                score = f1_score(y_true[indices], y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            bootstrap_scores.append(score)
            
        # 计算置信区间
        alpha = (1.0 - ci) / 2.0
        lower_bound = np.percentile(bootstrap_scores, 100 * alpha)
        upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha))
        
        return lower_bound, upper_bound


class VisualizationTool:
    """用于可视化评估结果的工具类"""
    
    def __init__(self, save_dir="./visualizations"):
        """
        初始化可视化工具
        
        Args:
            save_dir (str): 保存可视化结果的目录
        """
        self.save_dir = save_dir
        
    def plot_roc_curve(self, y_true, y_scores, model_names=None, save_path=None, show=True):
        """
        绘制ROC曲线
        
        Args:
            y_true (list or numpy.ndarray): 真实标签列表或单个数组
            y_scores (list or numpy.ndarray): 预测概率值列表或单个数组
            model_names (list, optional): 模型名称列表
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 如果输入是单个数组，则转换为列表
        if not isinstance(y_scores[0], (list, np.ndarray)):
            y_true = [y_true]
            y_scores = [y_scores]
            
        # 如果没有提供模型名称，则生成默认名称
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(y_scores))]
            
        # 绘制每个模型的ROC曲线
        for i, (y_t, y_s, name) in enumerate(zip(y_true, y_scores, model_names)):
            fpr, tpr, _ = roc_curve(y_t, y_s)
            roc_auc = roc_auc_score(y_t, y_s)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
        # 绘制随机猜测的基准线
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('受试者工作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_precision_recall_curve(self, y_true, y_scores, model_names=None, save_path=None, show=True):
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true (list or numpy.ndarray): 真实标签列表或单个数组
            y_scores (list or numpy.ndarray): 预测概率值列表或单个数组
            model_names (list, optional): 模型名称列表
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 如果输入是单个数组，则转换为列表
        if not isinstance(y_scores[0], (list, np.ndarray)):
            y_true = [y_true]
            y_scores = [y_scores]
            
        # 如果没有提供模型名称，则生成默认名称
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(y_scores))]
            
        # 绘制每个模型的PR曲线
        for i, (y_t, y_s, name) in enumerate(zip(y_true, y_scores, model_names)):
            precision, recall, _ = precision_recall_curve(y_t, y_s)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (AUPRC = {pr_auc:.3f})')
            
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线 (PR)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, save_path=None, show=True):
        """
        绘制混淆矩阵
        
        Args:
            y_true (numpy.ndarray): 真实标签
            y_pred (numpy.ndarray): 预测标签
            normalize (bool): 是否归一化
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        # 创建图形
        plt.figure(figsize=(8, 6))
        
        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=['非乙酰化', '乙酰化'],
                   yticklabels=['非乙酰化', '乙酰化'])
        
        # 设置图形属性
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title('混淆矩阵')
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_metrics_comparison(self, metrics_dict, metric_names=None, model_names=None, save_path=None, show=True):
        """
        比较不同模型的多个指标
        
        Args:
            metrics_dict (dict): 包含多个模型多个指标的字典
                格式: {model_name: {metric_name: value}}
            metric_names (list, optional): 要比较的指标名称列表
            model_names (list, optional): 要比较的模型名称列表
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 如果没有提供指标名称，则使用第一个模型的所有指标
        if metric_names is None:
            first_model = list(metrics_dict.keys())[0]
            metric_names = list(metrics_dict[first_model].keys())
            
        # 如果没有提供模型名称，则使用所有模型
        if model_names is None:
            model_names = list(metrics_dict.keys())
            
        # 准备数据
        data = []
        for model in model_names:
            for metric in metric_names:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': metrics_dict[model][metric]
                })
                
        df = pd.DataFrame(data)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 使用seaborn绘制条形图
        sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        
        # 设置图形属性
        plt.title('模型性能指标比较')
        plt.ylabel('指标值')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='模型')
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_ablation_study(self, ablation_results, metric='f1', save_path=None, show=True):
        """
        绘制消融实验结果
        
        Args:
            ablation_results (dict): 消融实验结果字典
                格式: {model_variant: score}
            metric (str): 使用的评估指标
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 排序结果
        sorted_results = sorted(ablation_results.items(), key=lambda x: x[1], reverse=True)
        model_names = [x[0] for x in sorted_results]
        scores = [x[1] for x in sorted_results]
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 绘制水平条形图
        bars = plt.barh(model_names, scores, color=plt.cm.viridis(np.linspace(0, 0.8, len(scores))))
        
        # 在条形上添加数值标签
        for i, (value, bar) in enumerate(zip(scores, bars)):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                    va='center', ha='left', fontsize=10)
        
        # 设置图形属性
        plt.xlabel(f'{metric.upper()} 分数')
        plt.title('消融实验结果')
        plt.xlim(0, max(scores) * 1.15)  # 为数值标签留出空间
        plt.grid(axis='x', alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_learning_curve(self, train_scores, val_scores, epochs, metric='loss', save_path=None, show=True):
        """
        绘制学习曲线
        
        Args:
            train_scores (list): 训练分数列表
            val_scores (list): 验证分数列表
            epochs (list or int): 训练轮数列表或总轮数
            metric (str): 指标名称
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 如果提供的是总轮数，则生成轮数列表
        if isinstance(epochs, int):
            epochs = list(range(1, epochs + 1))
            
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制学习曲线
        plt.plot(epochs, train_scores, label=f'训练 {metric}', marker='o', markersize=4)
        plt.plot(epochs, val_scores, label=f'验证 {metric}', marker='s', markersize=4)
        
        # 设置图形属性
        plt.xlabel('训练轮数')
        plt.ylabel(metric.capitalize())
        plt.title(f'学习曲线 - {metric.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()
        
    def visualize_importance_across_proteins(self, protein_ids, importance_scores, residue_positions, save_path=None, show=True):
        """
        比较不同蛋白质中氨基酸位点的重要性
        
        Args:
            protein_ids (list): 蛋白质ID列表
            importance_scores (list): 每个蛋白质的重要性分数列表
            residue_positions (list): 每个蛋白质中残基位置列表
            save_path (str, optional): 保存路径
            show (bool): 是否显示图形
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 为每个蛋白质绘制散点图
        for i, (protein_id, scores, positions) in enumerate(zip(protein_ids, importance_scores, residue_positions)):
            plt.scatter(positions, scores, label=protein_id, s=30, alpha=0.7)
            
        # 设置图形属性
        plt.xlabel('残基位置')
        plt.ylabel('重要性分数')
        plt.title('不同蛋白质中氨基酸位点重要性比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # 显示图形
        if show:
            plt.show()
            
        return plt.gcf()


def create_results_report(metrics, model_name, dataset_info=None, save_path=None):
    """
    生成评估结果报告
    
    Args:
        metrics (dict): 评估指标字典
        model_name (str): 模型名称
        dataset_info (dict, optional): 数据集信息字典
        save_path (str, optional): 保存路径
        
    Returns:
        str: 评估报告文本
    """
    # 创建报告标题
    report = f"# {model_name} 模型评估报告\n\n"
    
    # 添加数据集信息
    if dataset_info:
        report += "## 数据集信息\n\n"
        report += f"- 数据集名称: {dataset_info.get('name', '未指定')}\n"
        report += f"- 样本数量: {dataset_info.get('num_samples', '未指定')}\n"
        report += f"- 正样本比例: {dataset_info.get('positive_ratio', '未指定')}\n"
        report += f"- 特征数量: {dataset_info.get('num_features', '未指定')}\n\n"
    
    # 添加评估指标
    report += "## 评估指标\n\n"
    report += "| 指标 | 值 |\n"
    report += "| --- | --- |\n"
    
    # 从高到低优先级排序指标
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'auprc', 
                  'specificity', 'balanced_accuracy']
    
    for metric in key_metrics:
        if metric in metrics:
            report += f"| {metric.replace('_', ' ').title()} | {metrics[metric]:.4f} |\n"
    
    # 添加其他未列出的指标
    for metric, value in metrics.items():
        if metric not in key_metrics:
            report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
    
    # 添加总结
    report += "\n## 总结\n\n"
    
    # 根据F1和MCC值给出评价
    f1 = metrics.get('f1', 0)
    mcc = metrics.get('mcc', 0)
    
    if f1 > 0.9 and mcc > 0.8:
        performance = "非常优秀"
    elif f1 > 0.8 and mcc > 0.6:
        performance = "优秀"
    elif f1 > 0.7 and mcc > 0.4:
        performance = "良好"
    elif f1 > 0.6 and mcc > 0.2:
        performance = "中等"
    else:
        performance = "需要改进"
        
    report += f"模型整体表现**{performance}**。\n\n"
    
    # 添加日期
    from datetime import datetime
    report += f"\n\n*生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
    return report 