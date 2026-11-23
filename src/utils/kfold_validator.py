"""K-Fold 交叉验证与模型评估工具模块。

该模块提供了以下能力：
- 支持按蛋白质分组的分层 K Fold（StratifiedGroupKFold）划分，防止信息泄露；
- 训练每个 fold 并执行阈值调优与概率校准；
- 汇总和保存交叉验证结果、预测与详细指标。
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .data_utils import DataManager
from .metrics import calculate_metrics, calculate_optimal_threshold
from .model_utils import ModelManager
from .optimizer_utils import create_optimizer, create_scheduler
from .training_utils_enhanced import EnhancedTrainingManager

try:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - plotting is optional
    HAS_MATPLOTLIB = False
    fm = None
    plt = None

LOGGER = logging.getLogger(__name__)


class KFoldValidator:
    """K-Fold 交叉验证器，实现模型训练与评估流程。"""

    def __init__(self, config: Dict[str, Any], device: torch.device, output_dir: str):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = LOGGER
        self.kfold_dir = os.path.join(output_dir, "kfold_results")
        os.makedirs(self.kfold_dir, exist_ok=True)

        # K-Fold 相关参数
        self.k_splits = int(config.get("kfold_splits", 5))
        self.stratified = bool(config.get("kfold_stratified", True))
        self.shuffle = bool(config.get("kfold_shuffle", True))
        self.random_state = int(config.get("kfold_random_state", 42))
        self.save_models = bool(config.get("kfold_save_models", True))
        self.ensemble_method = config.get("kfold_ensemble_method", "average")
        self.save_predictions = bool(config.get("kfold_save_predictions", True))
        self.detailed_metrics = bool(config.get("kfold_detailed_metrics", True))

        # 运行期状态
        self.fold_results: List[Dict[str, Any]] = []
        self.fold_models: List[Dict[str, torch.Tensor]] = []
        self.fold_predictions: List[Dict[str, Any]] = []
        self.fold_calibration_params: List[Dict[str, Any]] = []
        self.kfold_curve_path: Optional[str] = None
        self.kfold_curve_exports: Dict[str, Dict[str, str]] = {}

        LOGGER.info(
            "初始化 K-Fold 验证器: k=%s, stratified=%s, shuffle=%s, seed=%s",
            self.k_splits,
            self.stratified,
            self.shuffle,
            self.random_state,
        )

        if self.k_splits < 2:
            LOGGER.warning("K 值 %s 太小，重置为 5", self.k_splits)
            self.k_splits = 5
        elif self.k_splits > 20:
            LOGGER.warning("K 值 %s 太大，重置为 10", self.k_splits)
            self.k_splits = 10

    # ------------------------------------------------------------------
    # 数据划分逻辑
    # ------------------------------------------------------------------
    def prepare_kfold_splits(self, data_manager: DataManager) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取交叉验证划分，优先使用按蛋白质的分层分组。"""
        print(f"准备 {self.k_splits}-fold 数据分割...")

        dataset_info = data_manager.get_full_dataset_for_kfold()
        if not isinstance(dataset_info, (tuple, list)) or len(dataset_info) < 2:
            raise ValueError("get_full_dataset_for_kfold 需返回至少 (X, y)")

        X = dataset_info[0]
        y = dataset_info[1]
        groups_raw: Optional[Iterable[Any]] = dataset_info[2] if len(dataset_info) >= 3 else None

        y = np.asarray(y, dtype=np.float32)
        num_samples = int(len(y))

        if X is None or (hasattr(X, "__len__") and len(X) != num_samples):
            X = np.arange(num_samples)
        else:
            X = np.asarray(X)

        groups = self._ensure_group_array(groups_raw, data_manager, num_samples)
        if groups is not None and len(groups) != num_samples:
            LOGGER.warning("分组信息长度与样本数不一致，忽略分组并回退到样本级划分")
            groups = None

        print(f"总样本数: {num_samples}")
        pos_count = float(np.sum(y))
        neg_count = float(num_samples - pos_count)
        pos_ratio = float(pos_count / num_samples) if num_samples else 0.0
        print(f"正样本数: {pos_count}, 负样本数: {neg_count}")
        print(f"正样本比例: {pos_ratio:.4f}")

        if groups is not None:
            unique_groups = np.unique(groups)
            print(f"蛋白质分组数: {len(unique_groups)}")
        else:
            print("未提供可用的蛋白质分组信息，将按样本级分割")

        y_binary = y.astype(int)
        splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        use_groups = groups is not None

        # 检查是否有自定义验证集比例
        custom_val_ratio = self.config.get("val_ratio")
        use_shuffle_split = False
        if custom_val_ratio is not None:
            try:
                custom_val_ratio = float(custom_val_ratio)
                # 如果自定义比例与 K-Fold 默认比例 (1/K) 差异较大，则启用 ShuffleSplit
                if 0 < custom_val_ratio < 1 and abs(custom_val_ratio - 1.0 / self.k_splits) > 0.01:
                    use_shuffle_split = True
            except (ValueError, TypeError):
                pass

        if use_shuffle_split:
            from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit
            print(f"使用自定义验证集比例 {custom_val_ratio} 进行 {self.k_splits} 次随机划分 (ShuffleSplit)")
            
            if use_groups:
                # GroupShuffleSplit 不支持 stratify，但能保证 group 不泄露
                splitter = GroupShuffleSplit(
                    n_splits=self.k_splits,
                    test_size=custom_val_ratio,
                    random_state=self.random_state
                )
                splits = list(splitter.split(X, y_binary, groups))
                print(f"使用按蛋白质分组的 ShuffleSplit (test_size={custom_val_ratio})")
            elif self.stratified:
                splitter = StratifiedShuffleSplit(
                    n_splits=self.k_splits,
                    test_size=custom_val_ratio,
                    random_state=self.random_state
                )
                splits = list(splitter.split(X, y_binary))
                print(f"使用分层 ShuffleSplit (test_size={custom_val_ratio})")
            else:
                splitter = ShuffleSplit(
                    n_splits=self.k_splits,
                    test_size=custom_val_ratio,
                    random_state=self.random_state
                )
                splits = list(splitter.split(X))
                print(f"使用随机 ShuffleSplit (test_size={custom_val_ratio})")

        elif self.stratified and use_groups:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=self.k_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
                splits = list(splitter.split(X, y_binary, groups))
                print(f"使用按蛋白质分层的 {self.k_splits}-fold 交叉验证")
            except ValueError as exc:
                print(f"StratifiedGroupKFold 失败: {exc}，尝试 GroupKFold 保持蛋白质粒度")
                try:
                    splitter = GroupKFold(n_splits=self.k_splits)
                    splits = list(splitter.split(X, y_binary, groups))
                    print(f"使用按蛋白质分组的 {self.k_splits}-fold 交叉验证（未分层）")
                    use_groups = True
                except ValueError as group_exc:
                    print(f"GroupKFold 也失败: {group_exc}，回退到样本级分割")
                    use_groups = False
                    splits = None

        if splits is None:
            if self.stratified:
                unique_labels, counts = np.unique(y_binary, return_counts=True)
                min_count = int(np.min(counts)) if counts.size > 0 else 0
                if min_count < self.k_splits:
                    print(
                        f"警告: 最少类别只有 {min_count} 个样本，少于 k={self.k_splits}，使用标准 KFold"
                    )
                    splitter = KFold(
                        n_splits=self.k_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                    )
                    splits = list(splitter.split(X))
                else:
                    splitter = StratifiedKFold(
                        n_splits=self.k_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                    )
                    splits = list(splitter.split(X, y_binary))
                    print(f"使用样本级分层 {self.k_splits}-fold 交叉验证")
            else:
                splitter = KFold(
                    n_splits=self.k_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
                splits = list(splitter.split(X))
                print(f"使用标准 {self.k_splits}-fold 交叉验证")

        for idx, (train_idx, val_idx) in enumerate(splits):
            train_total = len(train_idx)
            val_total = len(val_idx)
            train_pos = float(np.sum(y_binary[train_idx]))
            val_pos = float(np.sum(y_binary[val_idx]))
            train_ratio = train_pos / train_total if train_total else 0.0
            val_ratio = val_pos / val_total if val_total else 0.0
            info = (
                f"Fold {idx + 1}: 训练集={train_total} (正样本={train_pos}, {train_ratio:.3f}), "
                f"验证集={val_total} (正样本={val_pos}, {val_ratio:.3f})"
            )
            if use_groups and groups is not None:
                train_groups = np.unique(groups[train_idx])
                val_groups = np.unique(groups[val_idx])
                overlap = set(train_groups).intersection(set(val_groups))
                info += (
                    f", 训练蛋白数={len(train_groups)}, 验证蛋白数={len(val_groups)}"
                    + (f"，警告: 存在 {len(overlap)} 个蛋白质重叠" if overlap else "")
                )
            print(info)

        return splits

    # ------------------------------------------------------------------
    # 单个 Fold 训练与记录
    # ------------------------------------------------------------------
    def train_single_fold(
        self,
        fold_idx: int,
        train_loader,
        val_loader,
        data_manager: DataManager,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"训练 Fold {fold_idx + 1}/{self.k_splits}")
        print(f"{'=' * 60}")

        fold_output_dir = os.path.join(self.kfold_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        fold_logger = self._setup_fold_logger(fold_output_dir, fold_idx)
        simple_logger = None
        try:
            fold_config = self._create_fold_config(fold_idx, fold_output_dir)
            self._synchronize_critical_config_flags(fold_config)

            fold_logger.info("同步关键训练配置")
            fold_logger.info(
                "use_early_stopping=%s | monitor=%s | patience=%s",
                fold_config.get("use_early_stopping", False),
                fold_config.get("early_stopping_monitor", "val_f1"),
                fold_config.get("early_stopping_patience", 0),
            )
            fold_logger.info(
                "adaptive_dropout=%s | schedule=%s | input_dropout=%.4f",
                fold_config.get("adaptive_dropout", False),
                fold_config.get("dropout_schedule", "none"),
                fold_config.get("input_dropout", fold_config.get("dropout", 0.0)),
            )

            model_manager = ModelManager(fold_config, self.device)
            model = model_manager.create_model()
            fold_logger.info("模型已实例化并加载至设备")

            optimizer = create_optimizer(
                model=model,
                optimizer_name=fold_config.get("optimizer", "adamw"),
                learning_rate=float(fold_config.get("learning_rate", 1e-4)),
                weight_decay=float(fold_config.get("weight_decay", 0.01)),
                betas=fold_config.get("optimizer_betas", (0.9, 0.999)),
                eps=float(fold_config.get("optimizer_eps", 1e-8)),
            )

            scheduler, scheduler_type = create_scheduler(
                optimizer=optimizer,
                scheduler_name=fold_config.get("scheduler", "cosine_with_warmup"),
                train_loader=train_loader,
                epochs=int(fold_config.get("epochs", 100)),
                warmup_steps=fold_config.get("warmup_steps"),
                num_cycles=float(fold_config.get("scheduler_cycles", 0.5)),
                T_max=fold_config.get("scheduler_T_max"),
                step_size=int(fold_config.get("scheduler_step_size", 30)),
                gamma=float(fold_config.get("scheduler_gamma", 0.1)),
                patience=int(fold_config.get("scheduler_patience", 10)),
            )
            fold_config["scheduler_type"] = scheduler_type

            from .simple_logger import SimpleLogger  # 延迟导入以避免循环

            simple_logger = SimpleLogger(fold_output_dir)
            fold_config["logger"] = simple_logger

            self._inspect_first_batch(train_loader, fold_logger)

            training_manager = EnhancedTrainingManager(
                model=model,
                config=fold_config,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                output_dir=fold_output_dir,
            )

            training_results = training_manager.train()
            fold_logger.info(
                "训练完成: best_epoch=%s | best_val_f1=%.4f | training_time=%.2fs",
                training_results.get("best_epoch"),
                training_results.get("best_val_f1", 0.0),
                training_results.get("training_time", 0.0),
            )

            fold_result: Dict[str, Any] = {
                "fold_idx": fold_idx,
                "best_val_acc": training_results.get("best_val_acc", 0.0),
                "best_val_f1": training_results.get("best_val_f1", 0.0),
                "best_val_precision": training_results.get("best_val_precision", 0.0),
                "best_val_recall": training_results.get("best_val_recall", 0.0),
                "best_val_auc": training_results.get("best_val_auc", 0.0),
                "best_val_auprc": training_results.get("best_val_auprc", 0.0),
                "best_val_mcc": training_results.get("best_val_mcc", 0.0),
                "best_val_specificity": training_results.get("best_val_specificity", 0.0),
                "best_val_sensitivity": training_results.get("best_val_sensitivity", 0.0),
                "training_time": training_results.get("training_time", 0.0),
                "best_epoch": training_results.get("best_epoch", 0),
                "total_epochs": training_results.get("total_epochs", 0),
            }

            val_outputs = self._collect_outputs(model, val_loader, fold_logger)
            calibration_results = self._tune_threshold_and_calibration(
                fold_idx=fold_idx,
                val_outputs=val_outputs,
                fold_config=fold_config,
                fold_output_dir=fold_output_dir,
                fold_logger=fold_logger,
            )

            metrics_map = calibration_results.get("metrics", {})
            if metrics_map:
                fold_result.update(
                    {
                        "best_val_acc": metrics_map.get("val_post_accuracy", fold_result["best_val_acc"]),
                        "best_val_f1": metrics_map.get("val_post_f1", fold_result["best_val_f1"]),
                        "best_val_precision": metrics_map.get(
                            "val_post_precision", fold_result["best_val_precision"]
                        ),
                        "best_val_recall": metrics_map.get("val_post_recall", fold_result["best_val_recall"]),
                        "best_val_auc": metrics_map.get("val_post_roc_auc", fold_result["best_val_auc"]),
                        "best_val_auprc": metrics_map.get("val_post_pr_auc", fold_result["best_val_auprc"]),
                        "best_val_mcc": metrics_map.get("val_post_mcc", fold_result["best_val_mcc"]),
                        "best_val_specificity": metrics_map.get(
                            "val_post_specificity", fold_result["best_val_specificity"]
                        ),
                        "best_val_sensitivity": metrics_map.get(
                            "val_post_recall", fold_result["best_val_sensitivity"]
                        ),
                        "decision_threshold": calibration_results.get("decision_threshold", 0.5),
                        "calibration_method": calibration_results.get("calibration_method", "none"),
                        "temperature": calibration_results.get("temperature"),
                        "platt_coefficients": calibration_results.get("platt_coefficients"),
                        "platt_intercept": calibration_results.get("platt_intercept"),
                        "calibration_summary_path": calibration_results.get("calibration_summary_path"),
                    }
                )

            if self.save_models:
                model_path = os.path.join(fold_output_dir, f"best_model_fold_{fold_idx + 1}.pth")
                torch.save(model.state_dict(), model_path)
                fold_result["model_path"] = model_path
                self.fold_models.append(model.state_dict())
                fold_logger.info("已保存模型权重到 %s", model_path)
            else:
                # 即使不保存到磁盘，也将当前最佳模型权重保存到内存中，以便后续可能的集成评估
                # 注意：这可能会增加内存消耗
                self.fold_models.append(model.state_dict())
                fold_logger.info("kfold_save_models 为 False，仅将模型权重保存在内存中用于集成评估")

            if calibration_results.get("probabilities") is not None:
                fold_predictions = self._build_fold_prediction_payload(fold_idx, calibration_results)
                self.fold_predictions.append(fold_predictions)

                if self.save_predictions:
                    pred_path = os.path.join(fold_output_dir, f"predictions_fold_{fold_idx + 1}.json")
                    with open(pred_path, "w", encoding="utf-8") as file:
                        json.dump(fold_predictions, file, indent=2)
                    fold_result["predictions_path"] = pred_path
                    fold_logger.info("验证集预测结果已保存到 %s", pred_path)

                if self.detailed_metrics:
                    detailed_metrics = self._calculate_detailed_metrics(fold_predictions)
                    fold_result.update(detailed_metrics)
                    fold_logger.info("已计算并记录详细验证指标")

            print(
                "Fold %d 完成:\n  验证准确率: %.4f\n  验证F1分数: %.4f\n  验证精确率: %.4f\n  验证召回率: %.4f\n  验证AUC: %.4f",
                fold_idx + 1,
                fold_result.get("best_val_acc", 0.0),
                fold_result.get("best_val_f1", 0.0),
                fold_result.get("best_val_precision", 0.0),
                fold_result.get("best_val_recall", 0.0),
                fold_result.get("best_val_auc", 0.0),
            )
            fold_logger.info(
                "Fold 指标汇总: Accuracy=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f | AUC=%.4f | MCC=%.4f",
                fold_result.get("best_val_acc", 0.0),
                fold_result.get("best_val_f1", 0.0),
                fold_result.get("best_val_precision", 0.0),
                fold_result.get("best_val_recall", 0.0),
                fold_result.get("best_val_auc", 0.0),
                fold_result.get("best_val_mcc", 0.0),
            )

            return fold_result

        except Exception as exc:  # pylint: disable=broad-except
            import traceback

            error_message = f"Fold {fold_idx + 1} 训练失败: {exc}"
            print(error_message)
            print(traceback.format_exc())
            fold_logger.error(error_message, exc_info=True)
            return {
                "fold_idx": fold_idx,
                "error": str(exc),
                "best_val_acc": 0.0,
                "best_val_f1": 0.0,
                "best_val_precision": 0.0,
                "best_val_recall": 0.0,
                "best_val_auc": 0.0,
            }
        finally:
            if simple_logger and hasattr(simple_logger, "close"):
                simple_logger.close()
            self._teardown_fold_logger(fold_logger)

    def _setup_fold_logger(self, fold_output_dir: str, fold_idx: int) -> logging.Logger:
        logger_name = f"kfold.fold_{fold_idx + 1}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        log_path = os.path.join(fold_output_dir, "fold.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(f"[Fold {fold_idx + 1}] %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def _teardown_fold_logger(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    @staticmethod
    def _inspect_first_batch(loader, fold_logger: logging.Logger) -> None:
        try:
            iterator = iter(loader)
            first_batch = next(iterator)
        except StopIteration:
            fold_logger.warning("训练数据加载器为空，无法检查样本结构")
            return
        except Exception as exc:  # pylint: disable=broad-except
            fold_logger.warning("获取第一个 batch 时出错: %s", exc)
            return

        fold_logger.info("训练数据 batch 类型: %s", type(first_batch))
        if isinstance(first_batch, dict):
            keys = list(first_batch.keys())
            fold_logger.info("Batch keys: %s", keys)
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    fold_logger.info("  %s -> shape=%s, dtype=%s", key, tuple(value.shape), value.dtype)
        else:
            fold_logger.info("Batch 长度: %d", len(first_batch))
            for idx, item in enumerate(first_batch):
                if isinstance(item, torch.Tensor):
                    fold_logger.info("  Item %d -> shape=%s, dtype=%s", idx, tuple(item.shape), item.dtype)

    def _detach_to_cpu(self, value: Any):
        """递归地将张量或张量容器转移到CPU并转换为numpy."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {key: self._detach_to_cpu(sub_value) for key, sub_value in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._detach_to_cpu(item) for item in value]
        if isinstance(value, np.ndarray):
            return value
        # 标量或其他不可转换类型保持原样
        return value

    def _move_to_device(self, value: Any):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(sub_value) for key, sub_value in value.items()}
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        return value

    def _build_model_inputs_from_batch(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        seq_block = batch_dict.get("sequence")
        if isinstance(seq_block, dict):
            for key, value in seq_block.items():
                if value is None:
                    continue
                inputs[key] = value
        else:
            for key in ("window_features", "sequence_features"):
                if key in batch_dict and batch_dict[key] is not None:
                    inputs.setdefault("window_features", batch_dict[key])

        for key in (
            "window_features",
            "local_features",
            "global_features",
            "residue_types",
            "position_ids",
            "relative_position_offsets",
            "site_indicator",
            "ptm_type_ids",
        ):
            if key in batch_dict and batch_dict[key] is not None:
                inputs.setdefault(key, batch_dict[key])

        if "structure" in batch_dict and batch_dict["structure"] is not None:
            inputs["structure"] = batch_dict["structure"]
        elif "structure_features" in batch_dict and batch_dict["structure_features"] is not None:
            inputs["structure"] = batch_dict["structure_features"]

        if "positions" in batch_dict and batch_dict["positions"] is not None:
            inputs.setdefault("positions", batch_dict["positions"])

        return inputs

    # ------------------------------------------------------------------
    # 验证/推理辅助
    # ------------------------------------------------------------------
    def _collect_outputs(
        self,
        model,
        data_loader,
        fold_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, np.ndarray]:
        model.eval()
        logits_list: List[float] = []
        prob_list: List[float] = []
        pred_list: List[float] = []
        label_list: List[float] = []
        struct_masked_count_total = 0
        importance_scores_list: List[Any] = []
        
        # Metadata collection
        metadata_list: Dict[str, List[Any]] = {
            "uniprot_id": [],
            "position": [],
            "residue": [],
            "ptm_type": []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    processed = self._process_batch(model, batch)
                    if processed is None:
                        message = f"Fold batch {batch_idx}: 无有效输入，已跳过"
                        if fold_logger:
                            fold_logger.warning(message)
                        else:
                            print(message)
                        continue

                    if isinstance(processed, (tuple, list)):
                        logits, probabilities, predictions, labels = processed[:4]
                        # Handle new return format [logits, probs, preds, labels, struct_masked_count, importance_scores(optional)]
                        struct_masked_count = 0
                        importance_scores = None
                        
                        if len(processed) > 4:
                            if isinstance(processed[4], (int, float)):
                                struct_masked_count = processed[4]
                                if len(processed) > 5:
                                    importance_scores = processed[5]
                            else:
                                # Legacy fallback: element 4 is importance_scores
                                importance_scores = processed[4]
                                
                        struct_masked_count_total += struct_masked_count
                    else:
                        logits, probabilities, predictions, labels = processed
                        importance_scores = None
                        
                    logits_list.extend(logits.cpu().numpy().reshape(-1).tolist())
                    prob_list.extend(probabilities.cpu().numpy().reshape(-1).tolist())
                    pred_list.extend(predictions.cpu().numpy().reshape(-1).tolist())
                    label_list.extend(labels.cpu().numpy().reshape(-1).tolist())
                    
                    # Extract metadata if available in batch
                    if isinstance(batch, dict):
                        for key in metadata_list.keys():
                            val = batch.get(key)
                            if val is not None:
                                if isinstance(val, torch.Tensor):
                                    metadata_list[key].extend(val.cpu().numpy().tolist())
                                elif isinstance(val, (list, np.ndarray)):
                                    metadata_list[key].extend(list(val))
                                else:
                                    # Fallback for single values repeated for batch size
                                    batch_size = len(labels)
                                    metadata_list[key].extend([val] * batch_size)

                    if importance_scores is not None:
                        importance_scores_list.append(self._detach_to_cpu(importance_scores))
                except Exception as exc:  # pylint: disable=broad-except
                    message = f"处理 batch 时出错: {exc}"
                    if fold_logger:
                        fold_logger.error(message, exc_info=True)
                    else:
                        print(message)
                    continue

        if struct_masked_count_total > 0:
            msg = f"检测到 {int(struct_masked_count_total)} 个样本的结构分支被完全掩蔽 (全零向量)"
            if fold_logger:
                fold_logger.info(msg)
            else:
                print(msg)

        results = {
            "logits": np.array(logits_list, dtype=np.float64),
            "probabilities": np.array(prob_list, dtype=np.float64),
            "predictions": np.array(pred_list, dtype=np.float64),
            "labels": np.array(label_list, dtype=np.float64),
            "struct_masked_count": struct_masked_count_total,
        }
        
        # Add metadata to results if collected
        for key, vals in metadata_list.items():
            if vals and len(vals) == len(label_list):
                results[key] = vals
                
        if importance_scores_list:
            results["importance_scores"] = importance_scores_list
        return results

    def _process_batch(self, model, batch):
        # 增强的模型类型检测
        model_name = model.__class__.__name__
        is_dual_branch = "DualBranchFusionPredictor" in model_name or (hasattr(model, "sequence_tower") and hasattr(model, "gating"))
        is_acetylation_predictor = "AcetylationPredictor" in model_name

        if isinstance(batch, dict):
            batch_device = self._move_to_device(batch)
            labels = batch_device.get("labels", batch_device.get("label"))
            if labels is None:
                return None
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=self.device, dtype=torch.float32)
            else:
                labels = labels.to(self.device)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.view(-1)
            elif labels.dim() == 0:
                labels = labels.unsqueeze(0)
            batch_device["label"] = labels

            if is_dual_branch:
                # DualBranchFusionPredictor 期望完整的 batch 字典
                # 并且它内部会处理 sequence/structure 等键
                outputs = model(batch_device)
            elif is_acetylation_predictor:
                # AcetylationPredictor 也可以直接处理 batch 字典 (作为 seq_data 参数)
                # 这样可以确保它能正确提取内部需要的各种特征 (如 ptm_type_ids, position_ids 等)
                # 同时也支持通过 kwargs 传递
                try:
                    outputs = model(batch_device)
                except TypeError:
                    # 如果直接调用失败，回退到构建 inputs
                    inputs = self._build_model_inputs_from_batch(batch_device)
                    if not inputs:
                        return None
                    outputs = model(**inputs)
            else:
                # 通用模型处理
                inputs = self._build_model_inputs_from_batch(batch_device)
                if not inputs:
                    return None
                if hasattr(model, "get_fused_features"):
                    full_batch = {
                        "sequence": {
                            "window_features": inputs.get("window_features"),
                            "local_features": inputs.get("local_features"),
                            "global_features": inputs.get("global_features"),
                        },
                        "structure": inputs.get("structure"),
                        "label": labels,
                    }
                    # 确保 window_features 存在
                    if full_batch["sequence"]["window_features"] is None:
                        # 尝试从 inputs 中获取其他可能的序列特征
                        if "sequence_features" in inputs:
                            full_batch["sequence"]["window_features"] = inputs["sequence_features"]
                        else:
                            return None
                            
                    fused_features = model.get_fused_features(full_batch)
                    outputs = model.classifier(fused_features)
                else:
                    outputs = model(**inputs)
        else:
            inputs, labels = batch
            inputs = self._move_to_device(inputs)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=self.device, dtype=torch.float32)
            else:
                labels = labels.to(self.device)
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)

        importance_scores = None
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            if len(outputs) > 1:
                # Handle (logits, importance_scores, aux_outputs) or (logits, importance_scores)
                importance_scores = outputs[1]
                # If there are 3 outputs, the 3rd one is usually auxiliary outputs which we don't need for validation metrics
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            if "logits" in outputs:
                logits = outputs["logits"]
            elif "predicted_outcome" in outputs:
                logits = outputs["predicted_outcome"]
            else:
                first_key = next(iter(outputs))
                logits = outputs[first_key]
        else:
            logits = outputs

        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        # [Fix] Handle NaNs in logits
        if torch.isnan(logits).any():
             logits = torch.nan_to_num(logits, nan=0.0)

        probabilities = torch.sigmoid(logits)
        
        # [Fix] Handle NaNs in probabilities
        if torch.isnan(probabilities).any():
             probabilities = torch.nan_to_num(probabilities, nan=0.0)

        predictions = (probabilities > 0.5).float()

        struct_masked_count = 0
        if isinstance(outputs, dict) and "struct_masked_count" in outputs:
            val = outputs["struct_masked_count"]
            if isinstance(val, torch.Tensor):
                struct_masked_count = val.item()
            else:
                struct_masked_count = val

        result = [logits.detach(), probabilities.detach(), predictions.detach(), labels.detach(), struct_masked_count]
        if importance_scores is not None:
            result.append(importance_scores)
        return tuple(result)

    def _get_fold_predictions(self, model, val_loader, fold_idx: int) -> Dict[str, List]:
        outputs = self._collect_outputs(model, val_loader)
        return {
            "fold_idx": fold_idx,
            "predictions": outputs["predictions"].tolist(),
            "probabilities": outputs["probabilities"].tolist(),
            "labels": outputs["labels"].tolist(),
            "logits": outputs["logits"].tolist(),
        }

    def _evaluate_fold_final_metrics(self, model, val_loader) -> Dict[str, float]:
        outputs = self._collect_outputs(model, val_loader)
        if outputs["labels"].size == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "auc": 0.0,
                "mcc": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0,
            }

        y_true = outputs["labels"]
        y_pred = outputs["predictions"]
        y_prob = outputs["probabilities"]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:  # pylint: disable=broad-except
            auc = 0.0

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:  # pylint: disable=broad-except
            mcc = 0.0

        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                specificity = 0.0
                sensitivity = 0.0
        except Exception:  # pylint: disable=broad-except
            specificity = 0.0
            sensitivity = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "mcc": mcc,
            "specificity": specificity,
            "sensitivity": sensitivity,
        }

    # ------------------------------------------------------------------
    # 校准与阈值调优
    # ------------------------------------------------------------------
    def _tune_threshold_and_calibration(
        self,
        fold_idx: int,
        val_outputs: Dict[str, np.ndarray],
        fold_config: Dict[str, Any],
        fold_output_dir: str,
        fold_logger: logging.Logger,
    ) -> Dict[str, Any]:
        logits = val_outputs.get("logits", np.array([]))
        base_probabilities = val_outputs.get("probabilities", np.array([]))
        labels = val_outputs.get("labels", np.array([]))

        if labels.size == 0:
            fold_logger.warning("验证集无标签样本，跳过校准与阈值调优")
            summary = {
                "fold_idx": fold_idx,
                "calibration_method": "none",
                "decision_threshold": fold_config.get("decision_threshold", 0.5),
                "threshold_metric": fold_config.get("threshold_metric", self.config.get("threshold_metric", "f1")),
                "threshold_metric_value": None,
                "temperature": None,
                "platt_coefficients": None,
                "platt_intercept": None,
                "metrics": {},
            }
            self.fold_calibration_params.append(summary)
            return {
                "logits": logits,
                "probabilities": base_probabilities,
                "predictions": val_outputs.get("predictions", np.array([])),
                "labels": labels,
                "decision_threshold": summary["decision_threshold"],
                "calibration_method": "none",
                "metrics": {},
            }

        calibration_method = (
            fold_config.get("calibration_method", self.config.get("calibration_method", "none")) or "none"
        ).lower()
        optimize_threshold = bool(
            fold_config.get("optimize_threshold", self.config.get("optimize_threshold", False))
        )
        threshold_metric = fold_config.get("threshold_metric", self.config.get("threshold_metric", "f1"))
        decision_threshold = float(fold_config.get("decision_threshold", 0.5))

        temperature: Optional[float] = None
        platt_model: Optional[LogisticRegression] = None
        threshold_metric_value: Optional[float] = None

        unique_labels = np.unique(labels)
        if calibration_method == "temperature" and unique_labels.size > 1:
            try:
                temperature = self._fit_temperature_scaling(logits, labels)
                fold_logger.info("温度缩放完成，T=%.4f", temperature)
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("温度缩放失败: %s", exc)
                temperature = None
                calibration_method = "none"
        elif calibration_method == "platt" and unique_labels.size > 1:
            try:
                platt_model = self._fit_platt_scaling(logits, labels)
                fold_logger.info("Platt 缩放完成")
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("Platt 缩放失败: %s", exc)
                platt_model = None
                calibration_method = "none"
        elif calibration_method in {"temperature", "platt"}:
            fold_logger.warning("验证集类别不足，跳过概率校准")
            calibration_method = "none"

        calibrated_probabilities = self._apply_calibration(
            logits, base_probabilities, calibration_method, temperature, platt_model
        )

        if optimize_threshold:
            try:
                decision_threshold, threshold_metric_value = calculate_optimal_threshold(
                    labels, calibrated_probabilities, metric=threshold_metric
                )
                fold_logger.info(
                    "阈值调优(%s) -> threshold=%.4f, score=%.4f",
                    threshold_metric,
                    decision_threshold,
                    threshold_metric_value if threshold_metric_value is not None else float("nan"),
                )
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("阈值调优失败，使用默认阈值 %.4f: %s", decision_threshold, exc)
                threshold_metric_value = None

        predictions = (calibrated_probabilities >= decision_threshold).astype(int)
        metrics = calculate_metrics(
            y_true=labels,
            y_pred=predictions,
            y_prob=calibrated_probabilities,
            threshold=decision_threshold,
            prefix="val_post_",
        )

        fold_logger.info("最终校准方法: %s, 决策阈值: %.4f", calibration_method, decision_threshold)

        calibration_summary = {
            "fold_idx": fold_idx,
            "calibration_method": calibration_method,
            "decision_threshold": decision_threshold,
            "threshold_metric": threshold_metric,
            "threshold_metric_value": threshold_metric_value,
            "temperature": temperature,
            "platt_coefficients": platt_model.coef_.tolist() if platt_model is not None else None,
            "platt_intercept": platt_model.intercept_.tolist() if platt_model is not None else None,
            "metrics": metrics,
        }

        # Handle NaNs in metrics values
        def sanitize_metrics(m):
            sanitized = {}
            for k, v in m.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    sanitized[k] = 0.0
                else:
                    sanitized[k] = v
            return sanitized
            
        calibration_summary["metrics"] = sanitize_metrics(metrics)

        calibration_path = os.path.join(fold_output_dir, "calibration_summary.json")
        try:
            with open(calibration_path, "w", encoding="utf-8") as file:
                json.dump(calibration_summary, file, indent=2)
            fold_logger.info("校准与阈值调优结果已保存到 %s", calibration_path)
        except Exception as e:
            fold_logger.warning(f"无法保存校准结果到 {calibration_path}: {e}")

        calibration_summary["calibration_summary_path"] = calibration_path
        self.fold_calibration_params.append(calibration_summary)

        return {
            "logits": logits,
            "probabilities": calibrated_probabilities,
            "predictions": predictions,
            "labels": labels,
            "decision_threshold": decision_threshold,
            "calibration_method": calibration_method,
            "temperature": temperature,
            "platt_coefficients": calibration_summary["platt_coefficients"],
            "platt_intercept": calibration_summary["platt_intercept"],
            "metrics": metrics,
            "threshold_metric": threshold_metric,
            "threshold_metric_value": threshold_metric_value,
            "calibration_summary_path": calibration_path,
        }
    
    def _build_fold_prediction_payload(self, fold_idx: int, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        logits = calibration_results.get("logits", np.array([]))
        probabilities = calibration_results.get("probabilities", np.array([]))
        predictions = calibration_results.get("predictions", np.array([]))
        labels = calibration_results.get("labels", np.array([]))

        return {
            "fold_idx": fold_idx,
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions),
            "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else list(probabilities),
            "labels": labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
            "logits": logits.tolist() if isinstance(logits, np.ndarray) else list(logits),
            "decision_threshold": calibration_results.get("decision_threshold", 0.5),
            "calibration_method": calibration_results.get("calibration_method", "none"),
            "threshold_metric": calibration_results.get("threshold_metric"),
            "threshold_metric_value": calibration_results.get("threshold_metric_value"),
        }

    def evaluate_ensemble_on_test_set(self, test_loader) -> Dict[str, Any]:
        """
        在独立测试集上评估K-Fold集成模型。
        使用所有Fold模型的平均概率作为最终预测结果。
        """
        if not self.fold_models:
            self.logger.info("尝试从磁盘加载已存在的 Fold 模型...")
            self.discover_and_load_models()

        if not self.fold_models:
            self.logger.warning("没有可用的Fold模型进行集成评估")
            return None

        print(f"\n{'=' * 80}")
        print(f"开始在独立测试集上进行集成评估 (集成 {len(self.fold_models)} 个模型)")
        print(f"{'=' * 80}")

        all_fold_probs = []
        test_labels = None
        test_metadata = {} # Collect metadata once

        # 遍历每个Fold的模型进行预测
        for fold_idx, model_state in enumerate(self.fold_models):
            print(f"正在使用 Fold {fold_idx + 1} 模型进行推理...")
            
            # 创建模型实例
            # 注意：这里我们使用原始配置，但可能需要调整一些推理相关的参数
            fold_config = copy.deepcopy(self.config)
            
            # 确保模型结构参数一致
            model_manager = ModelManager(fold_config, self.device)
            model = model_manager.create_model()
            
            # 加载权重
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            
            # 推理
            fold_outputs = self._collect_outputs(model, test_loader)
            probs = fold_outputs['probabilities']
            
            if test_labels is None:
                test_labels = fold_outputs['labels']
                # 收集元数据
                for key in ["uniprot_id", "position", "residue", "ptm_type"]:
                    if key in fold_outputs:
                        test_metadata[key] = fold_outputs[key]
            else:
                # 简单验证标签一致性
                if len(test_labels) != len(fold_outputs['labels']):
                     self.logger.warning(f"Fold {fold_idx+1} 预测数量与之前的Fold不一致")

            all_fold_probs.append(probs)
        
        if not all_fold_probs:
            return None

        # 计算平均概率
        avg_probs = np.mean(all_fold_probs, axis=0)
        predictions = (avg_probs >= 0.5).astype(int) # 默认阈值 0.5

        # 计算指标
        metrics_prefix = "ensemble_test_"
        metrics = calculate_metrics(
            y_true=test_labels,
            y_pred=predictions,
            y_prob=avg_probs,
            threshold=0.5,
            prefix=metrics_prefix
        )

        print("\n集成模型测试集评估结果:")
        for k, v in metrics.items():
            if 'acc' in k or 'f1' in k or 'auc' in k:
                print(f"  {k}: {v:.4f}")

        # 保存预测结果
        if self.save_predictions:
            # 构造详细的预测结果列表，包含元数据
            rows = []
            for i in range(len(test_labels)):
                row = {
                    'true_label': int(test_labels[i]),
                    'predicted_label': int(predictions[i]),
                    'probability': float(avg_probs[i]),
                }
                # 添加元数据列
                for key in ["uniprot_id", "position", "residue", "ptm_type"]:
                    if key in test_metadata:
                        row[key] = test_metadata[key][i]
                
                rows.append(row)
            
            output_path = os.path.join(self.kfold_dir, "ensemble_test_predictions.csv")
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"集成预测结果已保存至: {output_path}")
            metrics[f'{metrics_prefix}predictions_csv'] = output_path

        return metrics

    def discover_and_load_models(self) -> int:
        """
        在输出目录中搜索并加载已训练好的 Fold 模型。
        
        Returns:
            int: 成功加载的模型数量
        """
        self.fold_models = []
        loaded_count = 0
        
        for fold_idx in range(self.k_splits):
            fold_dir = os.path.join(self.kfold_dir, f"fold_{fold_idx + 1}")
            if not os.path.isdir(fold_dir):
                continue
                
            # 优先尝试加载 best_model.pt (由 EnhancedTrainingManager 保存)
            # 其次尝试加载 best_model_fold_X.pth (由 KFoldValidator 保存)
            possible_paths = [
                os.path.join(fold_dir, "best_model.pt"),
                os.path.join(fold_dir, f"best_model_fold_{fold_idx + 1}.pth"),
                os.path.join(fold_dir, "final_model.pt")
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.fold_models.append(state_dict)
                    loaded_count += 1
                    self.logger.info(f"成功加载 Fold {fold_idx + 1} 模型: {model_path}")
                except Exception as e:
                    self.logger.error(f"加载 Fold {fold_idx + 1} 模型失败 ({model_path}): {e}")
            else:
                self.logger.warning(f"未找到 Fold {fold_idx + 1} 的模型文件")
        
        if loaded_count > 0:
            self.logger.info(f"共成功加载 {loaded_count}/{self.k_splits} 个 Fold 模型")
        else:
            self.logger.warning(f"在 {self.kfold_dir} 中未找到任何有效的 Fold 模型")
            
        return loaded_count

        logits = val_outputs.get("logits", np.array([]))
        base_probabilities = val_outputs.get("probabilities", np.array([]))
        labels = val_outputs.get("labels", np.array([]))

        if labels.size == 0:
            fold_logger.warning("验证集无标签样本，跳过校准与阈值调优")
            summary = {
                "fold_idx": fold_idx,
                "calibration_method": "none",
                "decision_threshold": fold_config.get("decision_threshold", 0.5),
                "threshold_metric": fold_config.get("threshold_metric", self.config.get("threshold_metric", "f1")),
                "threshold_metric_value": None,
                "temperature": None,
                "platt_coefficients": None,
                "platt_intercept": None,
                "metrics": {},
            }
            self.fold_calibration_params.append(summary)
            return {
                "logits": logits,
                "probabilities": base_probabilities,
                "predictions": val_outputs.get("predictions", np.array([])),
                "labels": labels,
                "decision_threshold": summary["decision_threshold"],
                "calibration_method": "none",
                "metrics": {},
            }

        calibration_method = (
            fold_config.get("calibration_method", self.config.get("calibration_method", "none")) or "none"
        ).lower()
        optimize_threshold = bool(
            fold_config.get("optimize_threshold", self.config.get("optimize_threshold", False))
        )
        threshold_metric = fold_config.get("threshold_metric", self.config.get("threshold_metric", "f1"))
        decision_threshold = float(fold_config.get("decision_threshold", 0.5))

        temperature: Optional[float] = None
        platt_model: Optional[LogisticRegression] = None
        threshold_metric_value: Optional[float] = None

        unique_labels = np.unique(labels)
        if calibration_method == "temperature" and unique_labels.size > 1:
            try:
                temperature = self._fit_temperature_scaling(logits, labels)
                fold_logger.info("温度缩放完成，T=%.4f", temperature)
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("温度缩放失败: %s", exc)
                temperature = None
                calibration_method = "none"
        elif calibration_method == "platt" and unique_labels.size > 1:
            try:
                platt_model = self._fit_platt_scaling(logits, labels)
                fold_logger.info("Platt 缩放完成")
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("Platt 缩放失败: %s", exc)
                platt_model = None
                calibration_method = "none"
        elif calibration_method in {"temperature", "platt"}:
            fold_logger.warning("验证集类别不足，跳过概率校准")
            calibration_method = "none"

        calibrated_probabilities = self._apply_calibration(
            logits, base_probabilities, calibration_method, temperature, platt_model
        )

        if optimize_threshold:
            try:
                decision_threshold, threshold_metric_value = calculate_optimal_threshold(
                    labels, calibrated_probabilities, metric=threshold_metric
                )
                fold_logger.info(
                    "阈值调优(%s) -> threshold=%.4f, score=%.4f",
                    threshold_metric,
                    decision_threshold,
                    threshold_metric_value if threshold_metric_value is not None else float("nan"),
                )
            except Exception as exc:  # pylint: disable=broad-except
                fold_logger.error("阈值调优失败，使用默认阈值 %.4f: %s", decision_threshold, exc)
                threshold_metric_value = None

        predictions = (calibrated_probabilities >= decision_threshold).astype(int)
        metrics = calculate_metrics(
            y_true=labels,
            y_pred=predictions,
            y_prob=calibrated_probabilities,
            threshold=decision_threshold,
            prefix="val_post_",
        )

        fold_logger.info("最终校准方法: %s, 决策阈值: %.4f", calibration_method, decision_threshold)

        calibration_summary = {
            "fold_idx": fold_idx,
            "calibration_method": calibration_method,
            "decision_threshold": decision_threshold,
            "threshold_metric": threshold_metric,
            "threshold_metric_value": threshold_metric_value,
            "temperature": temperature,
            "platt_coefficients": platt_model.coef_.tolist() if platt_model is not None else None,
            "platt_intercept": platt_model.intercept_.tolist() if platt_model is not None else None,
            "metrics": metrics,
        }

        calibration_path = os.path.join(fold_output_dir, "calibration_summary.json")
        with open(calibration_path, "w", encoding="utf-8") as file:
            json.dump(calibration_summary, file, indent=2)
        fold_logger.info("校准与阈值调优结果已保存到 %s", calibration_path)

        calibration_summary["calibration_summary_path"] = calibration_path
        self.fold_calibration_params.append(calibration_summary)

        return {
            "logits": logits,
            "probabilities": calibrated_probabilities,
            "predictions": predictions,
            "labels": labels,
            "decision_threshold": decision_threshold,
            "calibration_method": calibration_method,
            "temperature": temperature,
            "platt_coefficients": calibration_summary["platt_coefficients"],
            "platt_intercept": calibration_summary["platt_intercept"],
            "metrics": metrics,
            "threshold_metric": threshold_metric,
            "threshold_metric_value": threshold_metric_value,
            "calibration_summary_path": calibration_path,
        }

    def _apply_calibration(
        self,
        logits: np.ndarray,
        base_probabilities: Optional[np.ndarray],
        method: str,
        temperature: Optional[float],
        platt_model: Optional[LogisticRegression],
    ) -> np.ndarray:
        if logits.size == 0:
            if base_probabilities is not None:
                return np.asarray(base_probabilities, dtype=np.float64)
            return np.array([], dtype=np.float64)

        method = (method or "none").lower()
        if method == "temperature" and temperature is not None:
            calibrated_logits = logits / max(temperature, 1e-6)
            probs = self._sigmoid(calibrated_logits)
        elif method == "platt" and platt_model is not None:
            probs = platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]
        else:
            if base_probabilities is not None and base_probabilities.size == logits.size:
                probs = base_probabilities
            else:
                probs = self._sigmoid(logits)

        return np.clip(probs, 1e-6, 1 - 1e-6)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _fit_temperature_scaling(self, logits: np.ndarray, labels: np.ndarray) -> float:
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device).unsqueeze(1)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

        log_temperature = torch.zeros(1, requires_grad=True, device=self.device)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
        criterion = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_tensor / torch.exp(log_temperature)
            loss = criterion(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(torch.exp(log_temperature).detach().cpu().item())
        return max(temperature, 1e-3)

    @staticmethod
    def _fit_platt_scaling(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        model = LogisticRegression(solver="lbfgs")
        model.fit(logits.reshape(-1, 1), labels.astype(int))
        return model

    # ------------------------------------------------------------------
    # 配置复制与同步
    # ------------------------------------------------------------------
    def _synchronize_critical_config_flags(self, fold_config: Dict[str, Any]):
        critical_keys = [
            "use_early_stopping",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "early_stopping_monitor",
            "early_stopping_mode",
            "restore_best_weights",
            "adaptive_dropout",
            "dropout_schedule",
            "input_dropout",
            "hidden_dropout",
            "output_dropout",
            "dropout",
            "use_reinforcement",
            "use_causal",
            "use_meta_learning",
            "use_moe",
            "integration_method",
            "use_interpretability",
            "use_precomputed_features",
        ]

        for key in critical_keys:
            if key not in fold_config and key in self.config:
                fold_config[key] = self.config[key]

    def _create_fold_config(self, fold_idx: int, fold_output_dir: str) -> Dict[str, Any]:
        non_serializable_keys = {"tensorboard_writer", "logger"}

        fold_config: Dict[str, Any] = {}
        for key, value in self.config.items():
            if key in non_serializable_keys:
                continue
            try:
                if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                    fold_config[key] = value
                elif isinstance(value, dict):
                    fold_config[key] = self._copy_dict_safely(value)
                else:
                    fold_config[key] = copy.copy(value)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("无法复制配置项 %s: %s", key, exc)

        fold_config["output_dir"] = fold_output_dir
        fold_config["fold_idx"] = fold_idx
        return fold_config

    def _copy_dict_safely(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            try:
                if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                    result[key] = value
                elif isinstance(value, dict):
                    result[key] = self._copy_dict_safely(value)
                else:
                    result[key] = copy.copy(value)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("无法复制字典项 %s: %s", key, exc)
        return result

    # ------------------------------------------------------------------
    # 评估指标
    # ------------------------------------------------------------------
    def _calculate_detailed_metrics(self, predictions_data: Dict[str, List]) -> Dict[str, float]:
        y_true = np.array(predictions_data["labels"])
        y_pred = np.array(predictions_data["predictions"])
        y_prob = np.array(predictions_data["probabilities"])

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob)
        except Exception:  # pylint: disable=broad-except
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "detailed_accuracy": float(accuracy),
            "detailed_precision": float(precision),
            "detailed_recall": float(recall),
            "detailed_f1": float(f1),
            "detailed_auc": float(auc),
            "detailed_specificity": float(specificity),
            "detailed_sensitivity": float(sensitivity),
            "detailed_tp": int(tp),
            "detailed_tn": int(tn),
            "detailed_fp": int(fp),
            "detailed_fn": int(fn),
        }

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def _plot_kfold_curves(self) -> None:
        if not HAS_MATPLOTLIB:
            LOGGER.warning("Matplotlib 未安装，跳过K-Fold曲线绘制")
            return
        if not self.fold_predictions:
            return

        valid_records: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for record in self.fold_predictions:
            y_true = np.array(record.get("labels", []), dtype=np.int32)
            y_prob = np.array(record.get("probabilities", []), dtype=np.float64)
            if y_true.size == 0 or np.unique(y_true).size < 2:
                LOGGER.warning("Fold %s 缺少足够的类别用于绘制ROC/PR曲线，已跳过", record.get("fold_idx", "?"))
                continue
            if y_prob.shape[0] != y_true.shape[0]:
                LOGGER.warning("Fold %s 的预测数量与标签不匹配，已跳过绘制", record.get("fold_idx", "?"))
                continue
            valid_records.append((int(record.get("fold_idx", 0)), y_true, y_prob))

        if not valid_records:
            LOGGER.warning("没有可用于绘制的K-Fold预测记录")
            return

        style = {
            "colors": [
                (227 / 255, 141 / 255, 179 / 255),  # pink
                (78 / 255, 172 / 255, 151 / 255),   # teal-green
                (251 / 255, 184 / 255, 142 / 255),  # orange
                (56 / 255, 134 / 255, 194 / 255),   # blue
                (134 / 255, 198 / 255, 184 / 255),  # light green
            ],
            "figsize": (18, 8),
            "dpi": 400,
            "linewidths": {
                "plot_line": 3.4,
                "border": 2.2,
                "tick": 1.8,
                "dash": 1.4,
            },
            "fontsize": {
                "label": 36,
                "title": 36,
                "tick": 20,
                "legend": 22,
            },
        }

        arial_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
        if fm is not None and os.path.exists(arial_path):
            fm.fontManager.addfont(arial_path)
            plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
            plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
        else:
            plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans", "sans-serif"]

        plt.rcParams["axes.linewidth"] = style["linewidths"]["border"]
        plt.rcParams["xtick.major.width"] = style["linewidths"]["tick"]
        plt.rcParams["ytick.major.width"] = style["linewidths"]["tick"]
        plt.rcParams["xtick.direction"] = "out"
        plt.rcParams["ytick.direction"] = "out"

        roc_curves: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        pr_curves: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
        all_labels: List[np.ndarray] = []

        for fold_idx, y_true, y_prob in valid_records:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_curves.append((fold_idx, fpr, tpr, roc_auc_score(y_true, y_prob)))

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_curves.append((fold_idx, recall, precision, average_precision_score(y_true, y_prob)))
            all_labels.append(y_true)

        export_defs = {
            "pdf": {"ext": "pdf", "dpi": 400},
            "png": {"ext": "png", "dpi": 400},
            "svg": {"ext": "svg", "dpi": None},
        }

        def save_figure(fig, base_name: str) -> Dict[str, str]:
            paths: Dict[str, str] = {}
            for fmt, params in export_defs.items():
                ext = params["ext"]
                dpi = params["dpi"]
                path = os.path.join(self.kfold_dir, f"{base_name}.{ext}")
                save_kwargs = {"format": ext, "bbox_inches": "tight"}
                if dpi is not None:
                    save_kwargs["dpi"] = dpi
                fig.savefig(path, **save_kwargs)
                paths[fmt] = path
            return paths

        self.kfold_curve_exports = {}

        # Combined figure
        fig_combined, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=style["figsize"], dpi=style["dpi"])
        for idx, (fold_idx, fpr, tpr, roc_auc) in enumerate(roc_curves):
            color = style["colors"][idx % len(style["colors"])]
            roc_ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=style["linewidths"]["plot_line"],
                label=f"Fold {fold_idx + 1} (AUC={roc_auc:.4f})",
            )

        roc_ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=style["linewidths"]["dash"],
            label="Chance",
        )
        roc_ax.set_xlim(0.0, 1.0)
        roc_ax.set_ylim(0.0, 1.05)
        roc_ax.set_xlabel("False Positive Rate", fontsize=style["fontsize"]["label"], fontname="Arial")
        roc_ax.set_ylabel("True Positive Rate", fontsize=style["fontsize"]["label"], fontname="Arial")
        roc_ax.set_title("K-Fold ROC Curves", fontsize=style["fontsize"]["title"], fontweight="normal", pad=10)
        roc_ax.grid(True, alpha=0.22)
        roc_ax.legend(loc="lower right", fontsize=style["fontsize"]["legend"], frameon=False)

        labels_concat = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int32)
        baseline_value = float(np.mean(labels_concat)) if labels_concat.size > 0 else 0.0
        for idx, (fold_idx, recall, precision, pr_auc) in enumerate(pr_curves):
            color = style["colors"][idx % len(style["colors"])]
            pr_ax.plot(
                recall,
                precision,
                color=color,
                linewidth=style["linewidths"]["plot_line"],
                label=f"Fold {fold_idx + 1} (AUC={pr_auc:.4f})",
            )

        if baseline_value > 0:
            pr_ax.hlines(
                y=baseline_value,
                xmin=0,
                xmax=1,
                colors="gray",
                linestyles="--",
                linewidth=style["linewidths"]["dash"],
                label="Baseline",
            )
        pr_ax.set_xlim(0.0, 1.0)
        pr_ax.set_ylim(0.0, 1.05)
        pr_ax.set_xlabel("Recall", fontsize=style["fontsize"]["label"], fontname="Arial")
        pr_ax.set_ylabel("Precision", fontsize=style["fontsize"]["label"], fontname="Arial")
        pr_ax.set_title("K-Fold Precision-Recall Curves", fontsize=style["fontsize"]["title"], fontweight="normal", pad=10)
        pr_ax.grid(True, alpha=0.22)
        pr_ax.legend(loc="lower left", fontsize=style["fontsize"]["legend"], frameon=False)

        for ax in (roc_ax, pr_ax):
            ax.tick_params(axis="both", labelsize=style["fontsize"]["tick"], length=7)
            for spine in ax.spines.values():
                spine.set_linewidth(style["linewidths"]["border"])

        fig_combined.tight_layout()
        combined_paths = save_figure(fig_combined, "kfold_roc_pr_curves")
        plt.close(fig_combined)
        self.kfold_curve_exports["combined"] = combined_paths

        # Standalone ROC figure
        fig_roc, roc_only_ax = plt.subplots(figsize=(9, 8), dpi=style["dpi"])
        for idx, (fold_idx, fpr, tpr, roc_auc) in enumerate(roc_curves):
            color = style["colors"][idx % len(style["colors"])]
            roc_only_ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=style["linewidths"]["plot_line"],
                label=f"Fold {fold_idx + 1} (AUC={roc_auc:.4f})",
            )
        roc_only_ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=style["linewidths"]["dash"],
            label="Chance",
        )
        roc_only_ax.set_title("K-Fold ROC Curves", fontsize=style["fontsize"]["title"], fontweight="bold", pad=10)
        roc_only_ax.set_xlabel("False Positive Rate", fontsize=style["fontsize"]["label"], fontname="Arial")
        roc_only_ax.set_ylabel("True Positive Rate", fontsize=style["fontsize"]["label"], fontname="Arial")
        roc_only_ax.set_xlim(0.0, 1.0)
        roc_only_ax.set_ylim(0.0, 1.05)
        roc_only_ax.grid(True, alpha=0.22)
        roc_only_ax.legend(loc="lower right", fontsize=style["fontsize"]["legend"], frameon=False)
        roc_only_ax.tick_params(axis="both", labelsize=style["fontsize"]["tick"], length=7)
        for spine in roc_only_ax.spines.values():
            spine.set_linewidth(style["linewidths"]["border"])
        fig_roc.tight_layout()
        roc_paths = save_figure(fig_roc, "kfold_roc_curves")
        plt.close(fig_roc)
        self.kfold_curve_exports["roc"] = roc_paths

        # Standalone PR figure
        fig_pr, pr_only_ax = plt.subplots(figsize=(9, 8), dpi=style["dpi"])
        for idx, (fold_idx, recall, precision, pr_auc) in enumerate(pr_curves):
            color = style["colors"][idx % len(style["colors"])]
            pr_only_ax.plot(
                recall,
                precision,
                color=color,
                linewidth=style["linewidths"]["plot_line"],
                label=f"Fold {fold_idx + 1} (AUC={pr_auc:.4f})",
            )
        if baseline_value > 0:
            pr_only_ax.hlines(
                y=baseline_value,
                xmin=0,
                xmax=1,
                colors="gray",
                linestyles="--",
                linewidth=style["linewidths"]["dash"],
                label="Baseline",
            )
        pr_only_ax.set_title("K-Fold Precision-Recall Curves", fontsize=style["fontsize"]["title"], fontweight="bold", pad=10)
        pr_only_ax.set_xlabel("Recall", fontsize=style["fontsize"]["label"], fontname="Arial")
        pr_only_ax.set_ylabel("Precision", fontsize=style["fontsize"]["label"], fontname="Arial")
        pr_only_ax.set_xlim(0.0, 1.0)
        pr_only_ax.set_ylim(0.0, 1.05)
        pr_only_ax.grid(True, alpha=0.22)
        pr_only_ax.legend(loc="lower left", fontsize=style["fontsize"]["legend"], frameon=False)
        pr_only_ax.tick_params(axis="both", labelsize=style["fontsize"]["tick"], length=7)
        for spine in pr_only_ax.spines.values():
            spine.set_linewidth(style["linewidths"]["border"])
        fig_pr.tight_layout()
        pr_paths = save_figure(fig_pr, "kfold_pr_curves")
        plt.close(fig_pr)
        self.kfold_curve_exports["pr"] = pr_paths

        self.kfold_curve_path = combined_paths.get("pdf")

        print("K-Fold ROC/PR 曲线导出完成:")
        for key, paths in self.kfold_curve_exports.items():
            for fmt, path in paths.items():
                print(f"  {key} ({fmt.upper()}): {path}")

    # ------------------------------------------------------------------
    # 对外主流程
    # ------------------------------------------------------------------
    def run_kfold_validation(self, data_manager: DataManager) -> Dict[str, Any]:
        print(f"\n{'=' * 80}")
        print(f"开始 {self.k_splits}-fold 交叉验证")
        print(f"{'=' * 80}")

        splits = self.prepare_kfold_splits(data_manager)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_loader, val_loader = data_manager.create_kfold_loaders(train_idx, val_idx)
            fold_result = self.train_single_fold(fold_idx, train_loader, val_loader, data_manager)
            self.fold_results.append(fold_result)

            fold_result_path = os.path.join(self.kfold_dir, f"fold_{fold_idx + 1}_results.json")
            with open(fold_result_path, "w", encoding="utf-8") as file:
                json.dump(fold_result, file, indent=2)

        overall_results = self._calculate_overall_statistics()
        self._plot_kfold_curves()
        if self.kfold_curve_path:
            overall_results["kfold_curve_pdf"] = self.kfold_curve_path
        if self.kfold_curve_exports:
            overall_results["kfold_curve_exports"] = self.kfold_curve_exports
        self._save_kfold_results(overall_results)
        self._print_kfold_summary(overall_results)
        return overall_results

    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        valid_results = [result for result in self.fold_results if "error" not in result]
        if not valid_results:
            return {"error": "All folds failed", "valid_folds": 0}

        metrics = [
            "best_val_acc",
            "best_val_f1",
            "best_val_precision",
            "best_val_recall",
            "best_val_auc",
            "best_val_auprc",
            "best_val_mcc",
            "best_val_specificity",
            "best_val_sensitivity",
        ]

        statistics: Dict[str, Any] = {}
        for metric in metrics:
            values = [float(result.get(metric, 0.0)) for result in valid_results]
            if values:
                statistics[f"{metric}_mean"] = float(np.mean(values))
                statistics[f"{metric}_std"] = float(np.std(values))
                statistics[f"{metric}_min"] = float(np.min(values))
                statistics[f"{metric}_max"] = float(np.max(values))
                statistics[f"{metric}_median"] = float(np.median(values))
            else:
                statistics[f"{metric}_mean"] = 0.0
                statistics[f"{metric}_std"] = 0.0
                statistics[f"{metric}_min"] = 0.0
                statistics[f"{metric}_max"] = 0.0
                statistics[f"{metric}_median"] = 0.0

        statistics.update(
            {
                "total_folds": self.k_splits,
                "valid_folds": len(valid_results),
                "failed_folds": self.k_splits - len(valid_results),
                "fold_results": self.fold_results,
                "fold_calibration_params": self.fold_calibration_params,
                "kfold_config": {
                    "k_splits": self.k_splits,
                    "stratified": self.stratified,
                    "shuffle": self.shuffle,
                    "random_state": self.random_state,
                },
            }
        )

        return statistics

    def _save_kfold_results(self, overall_results: Dict[str, Any]):
        results_path = os.path.join(self.kfold_dir, "kfold_overall_results.json")
        serializable_results = copy.deepcopy(overall_results)

        if "fold_results" in serializable_results:
            for fold_result in serializable_results["fold_results"]:
                if isinstance(fold_result, dict):
                    for key in list(fold_result.keys()):
                        if isinstance(fold_result[key], (np.generic,)):
                            fold_result[key] = float(fold_result[key])

        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(serializable_results, file, indent=2)

        self._save_results_summary_csv(overall_results)
        print(f"K-Fold 结果已保存到: {self.kfold_dir}")

    def _save_results_summary_csv(self, overall_results: Dict[str, Any]):
        fold_rows = []
        for idx, result in enumerate(overall_results.get("fold_results", [])):
            if "error" in result:
                continue
            fold_rows.append(
                {
                    "Fold": idx + 1,
                    "Accuracy": result.get("best_val_acc", 0.0),
                    "F1_Score": result.get("best_val_f1", 0.0),
                    "Precision": result.get("best_val_precision", 0.0),
                    "Recall": result.get("best_val_recall", 0.0),
                    "AUC": result.get("best_val_auc", 0.0),
                    "PR_AUC": result.get("best_val_auprc", 0.0),
                    "Best_Epoch": result.get("best_epoch", 0),
                    "Training_Time": result.get("training_time", 0.0),
                }
            )

        if fold_rows:
            df = pd.DataFrame(fold_rows)

            mean_row = {
                "Fold": "Mean",
                "Accuracy": overall_results.get("best_val_acc_mean", 0.0),
                "F1_Score": overall_results.get("best_val_f1_mean", 0.0),
                "Precision": overall_results.get("best_val_precision_mean", 0.0),
                "Recall": overall_results.get("best_val_recall_mean", 0.0),
                "AUC": overall_results.get("best_val_auc_mean", 0.0),
                "PR_AUC": overall_results.get("best_val_auprc_mean", 0.0),
                "Best_Epoch": "-",
                "Training_Time": "-",
            }

            std_row = {
                "Fold": "Std",
                "Accuracy": overall_results.get("best_val_acc_std", 0.0),
                "F1_Score": overall_results.get("best_val_f1_std", 0.0),
                "Precision": overall_results.get("best_val_precision_std", 0.0),
                "Recall": overall_results.get("best_val_recall_std", 0.0),
                "AUC": overall_results.get("best_val_auc_std", 0.0),
                "PR_AUC": overall_results.get("best_val_auprc_std", 0.0),
                "Best_Epoch": "-",
                "Training_Time": "-",
            }

            df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

            csv_path = os.path.join(self.kfold_dir, "kfold_results_summary.csv")
            df.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"结果摘要已保存到: {csv_path}")

    def _print_kfold_summary(self, overall_results: Dict[str, Any]):
        print(f"\n{'=' * 80}")
        print(f"{self.k_splits}-Fold 交叉验证结果摘要")
        print(f"{'=' * 80}")

        if overall_results.get("valid_folds", 0) == 0:
            print("所有 fold 都失败了！")
            return

        valid_folds = overall_results.get("valid_folds", 0)
        failed_folds = overall_results.get("failed_folds", 0)

        print(f"成功完成的 fold 数: {valid_folds}/{self.k_splits}")
        if failed_folds > 0:
            print(f"失败的 fold 数: {failed_folds}")

        print(f"\n主要性能指标 (基于 {valid_folds} 个有效 fold):")
        print("-" * 60)
        metrics = [
            ("准确率", "best_val_acc"),
            ("F1分数", "best_val_f1"),
            ("精确率", "best_val_precision"),
            ("召回率", "best_val_recall"),
            ("AUC", "best_val_auc"),
            ("PR-AUC", "best_val_auprc"),
            ("MCC", "best_val_mcc"),
            ("特异性", "best_val_specificity"),
            ("敏感性", "best_val_sensitivity"),
        ]

        for metric_name, metric_key in metrics:
            mean_val = overall_results.get(f"{metric_key}_mean", 0.0)
            std_val = overall_results.get(f"{metric_key}_std", 0.0)
            min_val = overall_results.get(f"{metric_key}_min", 0.0)
            max_val = overall_results.get(f"{metric_key}_max", 0.0)
            print(
                f"{metric_name:>8}: {mean_val:.4f} ± {std_val:.4f} (范围: {min_val:.4f} - {max_val:.4f})"
            )

        print(f"\n各 fold 详细结果:")
        print("-" * 120)
        print(
            f"{'Fold':>6} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} "
            f"{'AUC':>10} {'PR-AUC':>10} {'MCC':>10} {'Specificity':>12} {'Sensitivity':>12}"
        )
        print("-" * 120)
        for idx, result in enumerate(overall_results.get("fold_results", [])):
            if "error" not in result:
                print(
                    f"{idx + 1:>6} {result.get('best_val_acc', 0.0):>10.4f} {result.get('best_val_f1', 0.0):>10.4f} "
                    f"{result.get('best_val_precision', 0.0):>10.4f} {result.get('best_val_recall', 0.0):>10.4f} "
                    f"{result.get('best_val_auc', 0.0):>10.4f} {result.get('best_val_auprc', 0.0):>10.4f} "
                    f"{result.get('best_val_mcc', 0.0):>10.4f} {result.get('best_val_specificity', 0.0):>12.4f} "
                    f"{result.get('best_val_sensitivity', 0.0):>12.4f}"
                )
            else:
                print(f"{idx + 1:>6} {'FAILED':>10}")
        print("-" * 120)
        curve_exports = overall_results.get("kfold_curve_exports")
        if curve_exports:
            print("ROC/PR 曲线导出文件:")
            for key, paths in curve_exports.items():
                for fmt, path in paths.items():
                    print(f"  {key} ({fmt.upper()}): {path}")
        elif overall_results.get("kfold_curve_pdf"):
            print(f"ROC/PR 曲线汇总PDF: {overall_results['kfold_curve_pdf']}")
        print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # 分组辅助
    # ------------------------------------------------------------------
    def _ensure_group_array(
        self,
        groups: Optional[Iterable[Any]],
        data_manager: DataManager,
        expected_length: int,
    ) -> Optional[np.ndarray]:
        if groups is not None:
            groups_array = np.asarray(list(groups))
            if groups_array.ndim > 1 and groups_array.shape[1] == 1:
                groups_array = groups_array.reshape(-1)
            return groups_array.astype(str)

        dataset = getattr(data_manager, "full_dataset", None)
        if dataset is None:
            return None

        derived_groups: List[str] = []
        for idx in range(expected_length):
            try:
                sample = dataset[idx]
            except Exception:  # pylint: disable=broad-except
                derived_groups.append(f"sample_{idx}")
                continue

            protein_id = self._extract_uniprot_id(sample)
            if protein_id is None:
                protein_id = f"sample_{idx}"
            derived_groups.append(str(protein_id))

        return np.asarray(derived_groups, dtype=str)

    @staticmethod
    def _extract_uniprot_id(sample: Any) -> Optional[str]:
        if isinstance(sample, dict):
            return sample.get("uniprot_id") or sample.get("protein_id")
        if isinstance(sample, tuple) and sample:
            first = sample[0]
            if isinstance(first, dict):
                return first.get("uniprot_id") or first.get("protein_id")
        return None
