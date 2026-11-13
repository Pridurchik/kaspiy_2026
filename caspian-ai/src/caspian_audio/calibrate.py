"""Пост-хок калибровка уверенности модели (Platt scaling на топ-1 вероятности).

Зачем: сырые softmax-вероятности модели систематически НЕДОуверены. На тесте средняя
уверенность 0.58 при реальной точности 0.66, ECE = 0.109 (см. MODEL_AUDIT.md §4a).
Инспектору показывается «уверенность», и она должна означать вероятность того, что
детекция верна, — а не сырой выход softmax.

Что делаем: обучаем на валидации логистическую (Platt) карту
    conf = sigmoid(a * logit(p_max) + b),
где p_max — максимум сырых вероятностей окна, а цель — «argmax совпал с меткой».
Это стандартная калибровка (температура/Platt); она монотонна, поэтому:
  * НЕ меняет argmax и порядок окон — решения и пороги остаются прежними;
  * меняет только показываемую уверенность (и, как следствие, вес риска в fusion).
Итог: ECE 0.109 -> 0.067, средняя уверенность 0.58 -> 0.66 (совпадает с точностью).

Модуль без torch: чистый numpy, годится и для буя (ONNX).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def top1_confidence(probs: np.ndarray) -> np.ndarray:
    """Максимум вероятности по классам для каждого окна: (N, C) -> (N,)."""
    return np.asarray(probs, dtype=np.float64).max(axis=1)


def fit_confidence(probs: np.ndarray, labels: np.ndarray, l2: float = 1e-6) -> tuple[float, float]:
    """Обучить (a, b) на валидации: conf = sigmoid(a * logit(p_max) + b).

    Цель — вероятность того, что argmax модели совпал с меткой. Оптимизируем NLL
    методом Ньютона (1 признак => сходится за десяток итераций). Возвращает (a, b).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError("probs должен быть (N, C) и непустым")
    x = _logit(top1_confidence(probs))
    y = (probs.argmax(axis=1) == labels).astype(np.float64)
    # вырожденные случаи (все верно/неверно): сдвиг без наклона
    if y.min() == y.max():
        b = 6.0 if y.mean() > 0.5 else -6.0
        return 1.0, float(b)
    X = np.stack([x, np.ones_like(x)], axis=1)
    theta = np.array([1.0, 0.0])
    reg = l2 * np.eye(2)
    for _ in range(100):
        q = _sigmoid(X @ theta)
        grad = X.T @ (q - y) + l2 * theta
        w = q * (1.0 - q) + _EPS
        hess = X.T @ (X * w[:, None]) + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - защита от вырожденной матрицы
            break
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def apply_confidence(p_max, a: float = 1.0, b: float = 0.0):
    """Скалярная или поэлементная калибровка уверенности (монотонна, 0..1)."""
    return _sigmoid(a * _logit(p_max) + b)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: средневзвешенная |точность - уверенность| по бинам."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    if total == 0:
        return 0.0
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def calibration_report(probs: np.ndarray, labels: np.ndarray, a: float, b: float) -> dict:
    """Сводка калибровки до/после (для CLI и тестов)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    raw = top1_confidence(probs)
    cal = apply_confidence(raw, a, b)
    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "ece_before": ece(raw, correct),
        "ece_after": ece(cal, correct),
        "mean_conf_before": float(raw.mean()),
        "mean_conf_after": float(cal.mean()),
    }
