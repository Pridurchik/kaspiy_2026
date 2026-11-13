"""Потоковая детекция: скользящее окно -> вероятности -> события -> алерты.

Модуль не импортирует torch: на буе работает с ONNX-предиктором.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

# (B, T) float32 -> (вероятности (B, C), эмбеддинги (B, D))
PredictFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_predictor(path: str) -> tuple[PredictFn, dict]:
    """`.onnx` (+ рядом .json с конфигом) или `.pt` чекпоинт."""
    if path.endswith(".onnx"):
        import json

        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        cfg = json.loads(open(path[:-5] + ".json", encoding="utf-8").read())
        name = sess.get_inputs()[0].name

        def fn(batch):
            out = sess.run(None, {name: batch.astype(np.float32)})
            emb = out[1] if len(out) > 1 else np.zeros((len(out[0]), 1), np.float32)  # старый экспорт без эмбеддингов
            return _softmax(out[0]), emb

        return fn, cfg

    import torch

    from .model import load_checkpoint

    model, cfg = load_checkpoint(path, "cpu")

    @torch.no_grad()
    def fn(batch):
        logits, emb = model(torch.from_numpy(batch.astype(np.float32)), return_embedding=True)
        return torch.softmax(logits, -1).numpy(), emb.numpy()

    return fn, cfg


def load_ood(model_path: str):
    """OOD-скорер лежит рядом с моделью (runs/audio/ood.npz), если `cli ood` его создал."""
    from pathlib import Path

    from .ood import OODScorer

    p = Path(model_path).with_name("ood.npz")
    return OODScorer.load(p) if p.exists() else None


class StreamingDetector:
    def __init__(self, predict: PredictFn, cfg: dict, batch_size: int = 32, ood=None):
        inf = cfg["inference"]
        self.predict, self.cfg, self.bs = predict, cfg, batch_size
        self.sr = cfg["sample_rate"]
        self.classes = cfg["classes"]
        self.win = int(inf["window_seconds"] * self.sr)
        self.hop = int(inf["hop_seconds"] * self.sr)
        self.thr = inf["thresholds"]
        self.min_consec = inf["min_consecutive"]
        self.ood = ood if inf.get("ood", {}).get("enabled", True) else None
        self.ood_min_consec = inf.get("ood", {}).get("min_consecutive", 2)
        # Порог OOD из конфига имеет приоритет над порогом из ood.npz (0.95-квантиль валидации).
        # Квантиль валидации не отражает полевой поток: на незнакомых звуках он даёт ~84 ложных
        # unknown_sound/ч. Явный порог 0.95 + min_consecutive 3 снижают это до ~2/ч (см. MODEL_AUDIT.md).
        ood_cfg = inf.get("ood", {})
        self.ood_threshold = float(ood_cfg.get("threshold", self.ood.threshold)) if self.ood is not None else None

    def window_probs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x) < self.win:
            x = np.pad(x, (0, self.win - len(x)))
        starts = np.arange(0, len(x) - self.win + 1, self.hop)
        probs, embs = [], []
        for i in range(0, len(starts), self.bs):
            batch = np.stack([x[s : s + self.win] for s in starts[i : i + self.bs]])
            p, e = self.predict(batch)
            probs.append(p)
            embs.append(e)
        return starts / self.sr, np.concatenate(probs), np.concatenate(embs)

    def events(self, times: np.ndarray, probs: np.ndarray) -> list[dict]:
        win_s = self.win / self.sr
        out = []
        for cls, thr in self.thr.items():
            k = self.classes.index(cls)
            active = probs[:, k] >= thr
            i = 0
            while i < len(active):
                if not active[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(active) and active[j + 1]:
                    j += 1
                if j - i + 1 >= self.min_consec.get(cls, 1):
                    seg = probs[i : j + 1, k]
                    out.append({
                        "class": cls,
                        "start": round(float(times[i]), 2),
                        "end": round(float(times[j] + win_s), 2),
                        "confidence": round(float(seg.max()), 3),
                        "mean_confidence": round(float(seg.mean()), 3),
                        "peak_time": round(float(times[i + int(seg.argmax())] + win_s / 2), 2),
                    })
                i = j + 1
        return sorted(out, key=lambda e: e["start"])

    def unknown_events(self, times: np.ndarray, probs: np.ndarray, embs: np.ndarray, events: list[dict]) -> list[dict]:
        """Окна, далёкие от всех классов обучения и не накрытые обычным алертом."""
        if self.ood is None or embs.shape[1] < 2:
            return []
        win_s = self.win / self.sr
        scores = self.ood.score(embs, probs)
        # окно "спокойно знакомое", если какой-то алертный класс уже выше порога
        covered = np.zeros(len(times), dtype=bool)
        for cls, thr in self.thr.items():
            covered |= probs[:, self.classes.index(cls)] >= thr
        active = (scores > self.ood_threshold) & ~covered
        out, i = [], 0
        while i < len(active):
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            if j - i + 1 >= self.ood_min_consec:
                seg = scores[i : j + 1]
                out.append({
                    "class": "unknown_sound",
                    "start": round(float(times[i]), 2),
                    "end": round(float(times[j] + win_s), 2),
                    "confidence": round(float(np.clip((seg.max() - self.ood_threshold) / max(self.ood_threshold, 1e-6), 0, 1)), 3),
                    "mean_confidence": round(float(seg.mean()), 3),
                    "peak_time": round(float(times[i + int(seg.argmax())] + win_s / 2), 2),
                })
            i = j + 1
        return out

    def run(self, x: np.ndarray) -> dict:
        times, probs, embs = self.window_probs(x)
        events = self.events(times, probs)
        events = sorted(events + self.unknown_events(times, probs, embs, events), key=lambda e: e["start"])
        return {"duration": round(len(x) / self.sr, 2), "times": times, "probs": probs, "events": events}


class OnlineDetector:
    """Для живого потока (микрофон/гидрофон): подаём окна по одному, получаем алерт при срабатывании."""

    def __init__(self, predict: PredictFn, cfg: dict, ood=None):
        self.predict, self.cfg = predict, cfg
        self.classes = cfg["classes"]
        self.thr = cfg["inference"]["thresholds"]
        self.min_consec = dict(cfg["inference"]["min_consecutive"])
        self.ood = ood if cfg["inference"].get("ood", {}).get("enabled", True) else None
        if self.ood is not None:
            self.min_consec["unknown_sound"] = cfg["inference"].get("ood", {}).get("min_consecutive", 2)
            self.ood_threshold = float(cfg["inference"].get("ood", {}).get("threshold", self.ood.threshold))
        self.run_len = {c: 0 for c in list(self.thr) + ["unknown_sound"]}
        self.run_max = {c: 0.0 for c in self.run_len}

    def _update(self, cls: str, value: float, thr: float, fired: list):
        if value >= thr:
            self.run_len[cls] += 1
            self.run_max[cls] = max(self.run_max[cls], value)
            if self.run_len[cls] == self.min_consec.get(cls, 1):  # один алерт на непрерывное событие
                fired.append({"class": cls, "confidence": round(self.run_max[cls], 3)})
        else:
            self.run_len[cls], self.run_max[cls] = 0, 0.0

    def push(self, window: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        probs, emb = self.predict(window[None].astype(np.float32))
        p = probs[0]
        fired: list[dict] = []
        any_known = False
        for cls, thr in self.thr.items():
            v = float(p[self.classes.index(cls)])
            any_known |= v >= thr
            self._update(cls, v, thr, fired)
        if self.ood is not None and emb.shape[1] > 1:
            s = float(self.ood.score(emb, probs)[0])
            self._update("unknown_sound", 0.0 if any_known else s, self.ood_threshold + 1e-12, fired)
        return p, fired


def to_alerts(events: list[dict], cfg: dict, sensor_id: str, lat: float | None, lon: float | None,
              recorded_at: datetime | None = None, model_version: str = "audio-cnn-0.1") -> list[dict]:
    """Формат алерта для дашборда инспектора (и для модуля фьюжна сигналов)."""
    inf = cfg["inference"]
    t0 = recorded_at or datetime.now(timezone.utc)
    return [{
        "id": str(uuid.uuid4()),
        "type": "acoustic",
        "category": e["class"],
        "title": inf.get("labels_ru", {}).get(e["class"], e["class"]),
        "severity": inf.get("severity", {}).get(e["class"], "medium"),
        "confidence": e["confidence"],
        "sensor_id": sensor_id,
        "location": {"lat": lat, "lon": lon} if lat is not None and lon is not None else None,
        "start_ts": (t0 + timedelta(seconds=e["start"])).isoformat(),
        "end_ts": (t0 + timedelta(seconds=e["end"])).isoformat(),
        "offset_s": [e["start"], e["end"]],
        "model_version": model_version,
        "status": "new",
    } for e in events]


def score_against_truth(events: list[dict], truth: list[dict], tol: float = 1.0) -> dict:
    """Событийная оценка на размеченной сцене: событие найдено, если пересекается с истиной."""
    res = {}
    for cls in sorted({t["class"] for t in truth} | {e["class"] for e in events}):
        T = [t for t in truth if t["class"] == cls]
        E = [e for e in events if e["class"] == cls]
        hit = lambda a, b: a["start"] - tol < b["end"] and b["start"] - tol < a["end"]
        tp_t = sum(any(hit(t, e) for e in E) for t in T)
        tp_e = sum(any(hit(e, t) for t in T) for e in E)
        res[cls] = {"truth": len(T), "detected": tp_t, "false_alarms": len(E) - tp_e}
    return res
