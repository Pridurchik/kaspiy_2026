"""OOD-детектор «честное не знаю».

Оценка окна — комбинация двух сигналов (ранговая нормализация по валидации):
  1) расстояние эмбеддинга до ближайшего центра класса (далеко = незнакомо);
  2) энтропия softmax (модель «мечется» между классами = незнакомо).
На наших данных комбинация даёт AUROC ~0.81 против ~0.76 у каждого сигнала
поодиночке. Незнакомый звук уходит на проверку человеку вместо уверенного
ложного алерта.

Качество меряем на классах ESC-50, которые НЕ участвуют в обучении (собаки,
сирены, голоса, ...): это готовый набор «незнакомых» звуков -> AUROC.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .audio_io import load_audio
from .dataset import AudioClipDataset
from .prepare import ESC50_MAP, read_manifest


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-9)).sum(axis=1)


class OODScorer:
    def __init__(self, centroids: np.ndarray, ref_dist: np.ndarray, ref_ent: np.ndarray,
                 threshold: float, classes: list[str]):
        self.centroids = centroids        # (C, D), L2-нормированные центры классов
        self.ref_dist = np.sort(ref_dist)  # распределение расстояний на валидации
        self.ref_ent = np.sort(ref_ent)    # распределение энтропий на валидации
        self.threshold = float(threshold)
        self.classes = classes

    def score(self, emb: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """(B, D), (B, C) -> (B,): 0..1, больше = незнакомее."""
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        dist = 1.0 - (e @ self.centroids.T).max(axis=1)
        r1 = np.searchsorted(self.ref_dist, dist) / len(self.ref_dist)
        r2 = np.searchsorted(self.ref_ent, _entropy(probs)) / len(self.ref_ent)
        return (r1 + r2) / 2

    def save(self, path: str | Path):
        np.savez_compressed(path, centroids=self.centroids, ref_dist=self.ref_dist,
                            ref_ent=self.ref_ent, threshold=self.threshold, classes=np.array(self.classes))

    @classmethod
    def load(cls, path: str | Path) -> "OODScorer":
        z = np.load(path, allow_pickle=False)
        return cls(z["centroids"], z["ref_dist"], z["ref_ent"], float(z["threshold"]), [str(c) for c in z["classes"]])


def _embed_rows(predict, cfg, rows, batch=32):
    """Эмбеддинги и вероятности клипов с детерминированной обрезкой (как на инференсе)."""
    ds = AudioClipDataset(rows, cfg, train=False)
    embs, probs, labels = [], [], []
    for i in range(0, len(ds), batch):
        xs = [ds[j][0].numpy() for j in range(i, min(i + batch, len(ds)))]
        p, e = predict(np.stack(xs))
        embs.append(e)
        probs.append(p)
        labels += [ds.rows[j]["label"] for j in range(i, min(i + batch, len(ds)))]
    return np.concatenate(embs), np.concatenate(probs), np.array(labels)


def fit(predict, cfg, quantile: float = 0.95) -> OODScorer:
    rows = read_manifest(cfg["data"]["manifest"])
    emb_tr, _, lab_tr = _embed_rows(predict, cfg, [r for r in rows if r["split"] == "train"])
    e = emb_tr / (np.linalg.norm(emb_tr, axis=1, keepdims=True) + 1e-9)
    cents = []
    for c in cfg["classes"]:
        m = e[lab_tr == c].mean(axis=0)
        cents.append(m / (np.linalg.norm(m) + 1e-9))
    # калибровка на валидации: распределения расстояний и энтропий своих данных
    emb_val, p_val, _ = _embed_rows(predict, cfg, [r for r in rows if r["split"] == "val"])
    ev = emb_val / (np.linalg.norm(emb_val, axis=1, keepdims=True) + 1e-9)
    cents = np.stack(cents)
    scorer = OODScorer(cents, 1.0 - (ev @ cents.T).max(axis=1), _entropy(p_val), 0.0, cfg["classes"])
    # порог: q-квантиль оценок на валидации, т.е. ~(1-q) своих окон уйдут на проверку
    scorer.threshold = float(np.quantile(scorer.score(emb_val, p_val), quantile))
    print(f"[ood] центры: {len(cents)}, калибровка: {len(ev)} вал.-клипов, порог (q={quantile}): {scorer.threshold:.4f}")
    return scorer


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC через ранги (без sklearn). pos — оценки «незнакомых», должны быть выше."""
    x = np.concatenate([neg, pos])
    r = x.argsort().argsort().astype(np.float64) + 1
    r_pos = r[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_ood(predict, cfg, scorer: OODScorer, per_category: int = 8) -> dict:
    rows = read_manifest(cfg["data"]["manifest"])
    emb_in, p_in, _ = _embed_rows(predict, cfg, [r for r in rows if r["split"] == "test"])
    s_in = scorer.score(emb_in, p_in)

    # «незнакомые» звуки: классы ESC-50, не попавшие в обучение
    root = Path(cfg["data"]["raw_dir"]) / "ESC-50-master"
    by_cat: dict[str, list] = {}
    with open(root / "meta" / "esc50.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["category"] not in ESC50_MAP:
                by_cat.setdefault(r["category"], []).append(root / "audio" / r["filename"])
    rng = np.random.default_rng(0)
    sr, n = cfg["sample_rate"], int(cfg["clip_seconds"] * cfg["sample_rate"])
    xs, cats = [], []
    for cat, paths in sorted(by_cat.items()):
        for p in rng.permutation(np.array(paths, dtype=object))[:per_category]:
            x = load_audio(p, sr)
            mid = max(0, (len(x) - n) // 2)
            x = np.pad(x[mid : mid + n], (0, max(0, n - len(x))))[:n]
            xs.append(x)
            cats.append(cat)
    s_out = []
    for i in range(0, len(xs), 32):
        p, e = predict(np.stack(xs[i : i + 32]).astype(np.float32))
        s_out.append(scorer.score(e, p))
    s_out = np.concatenate(s_out)

    res = {
        "auroc": auroc(s_out, s_in),
        "unknown_flagged": float((s_out > scorer.threshold).mean()),
        "known_flagged": float((s_in > scorer.threshold).mean()),
        "n_unknown": len(s_out), "n_known": len(s_in),
        "n_unknown_categories": len(by_cat),
    }
    print(f"[ood] AUROC={res['auroc']:.3f} | незнакомых помечено {res['unknown_flagged']:.1%} "
          f"({res['n_unknown']} клипов, {res['n_unknown_categories']} категорий) | "
          f"своих ошибочно помечено {res['known_flagged']:.1%}")
    worst = {}
    for cat in set(cats):
        m = np.array([c == cat for c in cats])
        worst[cat] = float((s_out[m] <= scorer.threshold).mean())
    missed = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    print("[ood] чаще всего принимаются за знакомые:", ", ".join(f"{c} ({v:.0%})" for c, v in missed if v > 0) or "—")
    return res
