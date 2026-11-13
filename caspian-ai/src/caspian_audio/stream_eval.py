"""Оценка «как в бою»: потоковая детекция (окна, пороги, N окон подряд, OOD) на реальных записях.

Клиповые метрики из `evaluate` не отвечают на вопрос инспектора «сколько ложных тревог
в час я получу». Здесь считаем:

  1) ложные тревоги в час на реальных звуках, которых модель не видела
     (категории ESC-50 вне обучения — голоса, животные, бытовые шумы);
  2) ложные тревоги в час на реальном фоне из test (фон + классы-буферы без алерта);
  3) recall событий: каждое реальное событие из test вклеивается в 10 с реального фона
     test при SNR 10 и 0 дБ и прогоняется детектором — алерт должен перекрыть событие.

Оба профиля порогов: сервер (`thresholds`) и буй (`thresholds` + `edge_thresholds`).
Синтетика здесь не участвует вообще.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audio_io import active_rms, load_audio
from .detect import StreamingDetector, load_ood, load_predictor
from .prepare import esc50_unknown_categories, read_manifest

TRACK_S, EVENT_AT_S = 10.0, 4.0


def _profiles(cfg: dict) -> dict:
    inf = cfg["inference"]
    return {"server": dict(inf["thresholds"]), "edge": {**inf["thresholds"], **inf.get("edge_thresholds", {})}}


def _detector(predict, cfg, ood, thresholds):
    c = copy.deepcopy(cfg)
    c["inference"]["thresholds"] = thresholds
    return StreamingDetector(predict, c, ood=ood)


def _false_alarms(dets: dict, clips: list[np.ndarray], sr: int) -> dict:
    hours = sum(len(x) for x in clips) / sr / 3600
    out = {"hours": round(hours, 3)}
    for name, det in dets.items():
        cnt = Counter(e["class"] for x in clips for e in det.run(x)["events"])
        out[name] = {c: round(v / hours, 1) for c, v in sorted(cnt.items())}
    return out


def stream_eval(model_path: str, inference: dict | None = None, out: str | Path | None = None, seed: int = 0) -> dict:
    predict, cfg = load_predictor(model_path)
    if inference:  # пороги из актуального YAML
        cfg["inference"] = inference
    ood = load_ood(model_path)
    sr = cfg["sample_rate"]
    train_only = set(cfg["data"].get("train_only_sources", ["synthetic"]))
    alert_classes = list(cfg["inference"]["thresholds"])
    profiles = _profiles(cfg)
    dets = {name: _detector(predict, cfg, ood, thr) for name, thr in profiles.items()}
    rng = np.random.default_rng(seed)

    test = [r for r in read_manifest(cfg["data"]["manifest"]) if r["split"] == "test" and r["source"] not in train_only]
    neg = [r for r in test if r["label"] not in alert_classes]
    neg_clips = [load_audio(r["path"], sr) for r in neg]

    res: dict = {"model": model_path, "profiles": profiles}
    print(f"[stream] модель {model_path}; профили порогов: {profiles}")

    # 1) незнакомые реальные звуки
    unk = esc50_unknown_categories(Path(cfg["data"]["raw_dir"]) / "ESC-50-master")
    unk_clips = [load_audio(p, sr) for ps in unk.values() for p in ps]
    res["unknown_sounds_alerts_per_hour"] = _false_alarms(dets, unk_clips, sr)
    res["unknown_sounds_alerts_per_hour"]["categories"] = len(unk)

    # 2) реальный фон test (+ классы без алерта, например фейерверки)
    res["test_background_alerts_per_hour"] = _false_alarms(dets, neg_clips, sr)
    res["test_background_alerts_per_hour"]["labels"] = dict(Counter(r["label"] for r in neg))

    # 3) recall реальных событий, вклеенных в реальный фон
    bgs = [x for x in neg_clips if np.sqrt(np.mean(x**2)) > 1e-3 and len(x) >= sr]
    n_track = int(TRACK_S * sr)

    def bg_track():
        parts, n = [], 0
        while n < n_track:
            x = bgs[rng.integers(len(bgs))]
            parts.append(x)
            n += len(x)
        return np.concatenate(parts)[:n_track]

    recall: dict = {}
    for cls in alert_classes:
        evs = [load_audio(r["path"], sr) for r in test if r["label"] == cls]
        if not evs:
            continue
        recall[cls] = {"n": len(evs)}
        for snr in (10, 0):
            hits = {name: 0 for name in dets}
            for ev in evs:
                ev = ev[: int((TRACK_S - EVENT_AT_S) * sr)]
                track, i0 = bg_track(), int(EVENT_AT_S * sr)
                # SNR считается как в mix_at_snr: по активной (громкой) части события
                scale = active_rms(ev) / (np.sqrt(np.mean(track**2) + 1e-12) * 10 ** (snr / 20))
                x = track * scale
                x[i0 : i0 + len(ev)] += ev
                x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
                t0, t1 = EVENT_AT_S, EVENT_AT_S + len(ev) / sr
                for name, det in dets.items():
                    evts = det.run(x)["events"]
                    hits[name] += any(e["class"] == cls and e["start"] < t1 + 1 and e["end"] > t0 - 1 for e in evts)
            recall[cls][f"snr{snr}"] = {name: round(h / len(evs), 3) for name, h in hits.items()}
    res["event_recall_real"] = recall

    print(f"[stream] незнакомые звуки ({res['unknown_sounds_alerts_per_hour']['hours']} ч, {len(unk)} категорий), алертов в час:")
    for name in dets:
        print(f"    {name:6s} {res['unknown_sounds_alerts_per_hour'][name]}")
    print(f"[stream] реальный фон test ({res['test_background_alerts_per_hour']['hours']} ч), алертов в час:")
    for name in dets:
        print(f"    {name:6s} {res['test_background_alerts_per_hour'][name]}")
    print("[stream] recall реальных событий на реальном фоне (алерт перекрыл событие):")
    for cls, m in recall.items():
        print(f"    {cls:11s} n={m['n']:4d}  SNR 10 дБ: {m['snr10']}  SNR 0 дБ: {m['snr0']}")

    out = Path(out or Path(model_path).with_name("metrics_stream.json"))
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[stream] -> {out}")
    return res
