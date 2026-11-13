"""Метрики классификации на numpy (без scikit-learn)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y, pred, n_classes) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def per_class(y, pred, classes) -> dict:
    cm = confusion_matrix(y, pred, len(classes))
    out = {}
    for k, c in enumerate(classes):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(cm[k].sum())}
    # как в sklearn: усредняем по классам, которые есть в разметке или в предсказаниях
    # (класс без примеров в выборке иначе даёт F1=0 и занижает среднее)
    present = [c for k, c in enumerate(classes) if cm[k].sum() or cm[:, k].sum()]
    out["macro_f1"] = float(np.mean([out[c]["f1"] for c in present])) if present else 0.0
    out["accuracy"] = float(np.trace(cm) / max(cm.sum(), 1))
    return out


def macro_f1(y, pred, n_classes) -> float:
    return per_class(y, pred, list(range(n_classes)))["macro_f1"]


def format_report(rep: dict, classes) -> str:
    lines = [f"{'':>12s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'support':>8s}"]
    for c in classes:
        m = rep[c]
        lines.append(f"{c:>12s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:6.3f} {m['support']:8d}")
    lines.append(f"{'accuracy':>12s} {rep['accuracy']:24.3f}")
    lines.append(f"{'macro F1':>12s} {rep['macro_f1']:24.3f}")
    return "\n".join(lines)
