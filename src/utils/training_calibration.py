"""Post-training probability calibration helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.training_metrics import ece_binary, safe_auc, safe_auprc

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN_CAL = True
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    HAS_SKLEARN_CAL = False


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def macro_ece(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    vals = []
    for i in range(y_true.shape[1]):
        known = y_true[:, i] >= 0.0
        if np.any(known):
            vals.append(ece_binary(y_true[known, i], y_prob[known, i]))
    return float(np.nanmean(vals)) if vals else float("nan")


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    max_iter: int = 2000,
    lr: float = 0.01,
) -> float:
    if val_logits.size == 0:
        return 1.0
    known = np.isfinite(val_labels) & (val_labels >= 0.0)
    if not np.any(known):
        return 1.0
    logits_np = np.asarray(val_logits, dtype=np.float32)[known]
    labels_np = np.asarray(val_labels, dtype=np.float32)[known]
    logits = torch.tensor(logits_np, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_np, dtype=torch.float32, device=device)
    log_t = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([log_t], lr=float(lr))
    for _ in range(int(max_iter)):
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = F.binary_cross_entropy_with_logits(logits / t, labels)
        loss.backward()
        opt.step()
    t_final = float(torch.exp(log_t).clamp(min=1e-3, max=100.0).detach().cpu().item())
    return t_final


def apply_temperature(val_logits: np.ndarray, temperature: float) -> np.ndarray:
    t = float(max(1e-6, temperature))
    return sigmoid_np(val_logits / t)


def fit_platt_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> Optional[List[Any]]:
    if not HAS_SKLEARN_CAL or LogisticRegression is None:
        return None
    models = []
    for i in range(val_labels.shape[1]):
        # Unknown labels are -1; fitting on them makes a 3-class model whose
        # predict_proba[:, 1] is P(label==0), which inverts the ranking.
        known = np.isfinite(val_labels[:, i]) & (val_labels[:, i] >= 0.0)
        yi = val_labels[known, i].astype(np.int32, copy=False)
        if len(np.unique(yi)) < 2:
            models.append(None)
            continue
        clf = LogisticRegression(solver="lbfgs", max_iter=2000)
        clf.fit(val_logits[known][:, [i]], yi)
        models.append(clf)
    return models


def apply_platt(models: List[Any], logits: np.ndarray) -> np.ndarray:
    out = np.zeros_like(logits, dtype=np.float32)
    for i, clf in enumerate(models):
        if clf is None:
            out[:, i] = sigmoid_np(logits[:, i])
        else:
            out[:, i] = clf.predict_proba(logits[:, [i]])[:, 1].astype(np.float32)
    return out


def fit_isotonic(val_prob: np.ndarray, val_labels: np.ndarray) -> Optional[List[Any]]:
    if not HAS_SKLEARN_CAL or IsotonicRegression is None:
        return None
    models = []
    for i in range(val_labels.shape[1]):
        # Exclude unknown (-1) labels; otherwise the fitted curve is pulled
        # below 0 and collapses to a near-constant after clipping.
        known = np.isfinite(val_labels[:, i]) & (val_labels[:, i] >= 0.0)
        yi = val_labels[known, i].astype(np.float32, copy=False)
        if len(np.unique(yi)) < 2:
            models.append(None)
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_prob[known, i].astype(np.float32), yi)
        models.append(iso)
    return models


def apply_isotonic(models: List[Any], prob: np.ndarray) -> np.ndarray:
    out = np.zeros_like(prob, dtype=np.float32)
    for i, iso in enumerate(models):
        if iso is None:
            out[:, i] = prob[:, i].astype(np.float32)
        else:
            out[:, i] = iso.predict(prob[:, i].astype(np.float32)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    y_true = y_true.astype(np.float32, copy=False)
    y_prob = np.clip(y_prob.astype(np.float32, copy=False), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins) - 2 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            bin_count.append(0)
            continue
        bin_acc.append(float(y_true[mask].mean()))
        bin_conf.append(float(y_prob[mask].mean()))
        bin_count.append(count)
    return {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count, "n_bins": int(n_bins)}


def run_calibration_unit_tests() -> None:
    logits = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    p = sigmoid_np(logits)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    p_hot = sigmoid_np(logits / 0.5)
    p_cold = sigmoid_np(logits / 2.0)
    assert float(np.abs(p_hot - 0.5).mean()) > float(np.abs(p - 0.5).mean())
    assert float(np.abs(p_cold - 0.5).mean()) < float(np.abs(p - 0.5).mean())
    order = np.argsort(logits)
    assert np.all(np.diff(p[order]) >= 0)
    labels = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    raw_auc = safe_auc(labels, p)
    raw_auprc = safe_auprc(labels, p)
    for temperature in (0.2, 0.5, 1.0, 2.0, 10.0):
        calibrated = apply_temperature(logits, temperature)
        assert abs(safe_auc(labels, calibrated) - raw_auc) < 1e-7
        assert abs(safe_auprc(labels, calibrated) - raw_auprc) < 1e-7
    multilabel_logits = np.array([[2.0, -1.0], [-2.0, 0.5], [0.5, 1.5]], dtype=np.float32)
    multilabel_labels = np.array([[1.0, -1.0], [0.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    temperature = fit_temperature_scaling(
        multilabel_logits,
        multilabel_labels,
        device=torch.device("cpu"),
        max_iter=5,
        lr=0.01,
    )
    assert np.isfinite(temperature) and temperature > 0.0
    assert np.isfinite(macro_ece(multilabel_labels, sigmoid_np(multilabel_logits)))
