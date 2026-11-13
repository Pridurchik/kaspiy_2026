"""Клиент "акустического буя": слушает микрофон/гидрофон (или файл как поток),
детектирует на устройстве и отправляет только алерты на сервер дашборда.

  python -m caspian_audio.edge --model runs/audio/model.onnx --mic
  python -m caspian_audio.edge --model runs/audio/model.onnx --file data/scene.wav --realtime
  ... --post http://localhost:8000/api/alerts   # сервер платформы (inspector.py)

С алертом уходит 8-секундный клип и GPS-время его первого отсчёта: сервер перепроверяет
клип строгими порогами (каскад), а если то же событие прислали ≥3 буя — считает пеленг.
Время: на буе системные часы синхронизированы с GPS (gpsd + chrony, PPS). Для проигрывания
файла как потока время начала задаётся --start-time.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

from .audio_io import load_audio, save_audio
from .detect import OnlineDetector, load_ood, load_predictor, to_alerts


def _send(alert: dict, url: str | None):
    print(json.dumps({k: v for k, v in alert.items() if k != "clip_b64"}, ensure_ascii=False))
    if url:
        req = urllib.request.Request(url, data=json.dumps(alert).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:  # нет связи — алерт остаётся в логе, в проде кладём в очередь
            print(f"[edge] не отправлено: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/audio/model.onnx")
    ap.add_argument("--mic", action="store_true", help="живой вход (нужен pip install sounddevice)")
    ap.add_argument("--file")
    ap.add_argument("--realtime", action="store_true", help="проигрывать файл в реальном времени")
    ap.add_argument("--sensor-id", default="buoy-01")
    ap.add_argument("--lat", type=float, default=45.75)   # пример: дельта Волги
    ap.add_argument("--lon", type=float, default=48.40)
    ap.add_argument("--post")
    ap.add_argument("--save-clips", default="data/alert_clips", help="сохранять аудио-доказательство алерта")
    ap.add_argument("--strict", action="store_true", help="не занижать пороги (без параноидального профиля буя)")
    ap.add_argument("--cooldown", type=float, default=30.0, help="не повторять алерт того же класса чаще, с")
    ap.add_argument("--start-time", help="ISO-время начала файла (GPS) при --file; по умолчанию сейчас")
    a = ap.parse_args()

    predict, cfg = load_predictor(a.model)
    # Каскад: на буе пороги ЗАНИЖЕНЫ (edge_thresholds) — задача не пропустить.
    # Сервер перепроверит присланный клип стандартными, более строгими порогами.
    overrides = cfg["inference"].get("edge_thresholds", {})
    if overrides and not a.strict:
        cfg["inference"]["thresholds"] = {**cfg["inference"]["thresholds"], **overrides}
        print(f"[edge] параноидальный профиль буя: пороги {cfg['inference']['thresholds']}")
    det = OnlineDetector(predict, cfg, ood=load_ood(a.model))
    sr = cfg["sample_rate"]
    win, hop = int(cfg["inference"]["window_seconds"] * sr), int(cfg["inference"]["hop_seconds"] * sr)
    buf = deque(maxlen=win * 4)  # 8 с истории для клипа-доказательства
    t_start = datetime.fromisoformat(a.start_time) if a.start_time else datetime.now(timezone.utc)
    total = [0]  # отсчётов с начала потока: время отсчёта = t_start + i / sr
    last_fire: dict[str, float] = {}  # класс -> время последнего алерта (с от начала потока)

    def handle(chunk: np.ndarray):
        buf.extend(chunk.tolist())
        total[0] += len(chunk)
        if len(buf) < win:
            return
        window = np.array(list(buf)[-win:], dtype=np.float32)
        probs, fired = det.push(window)
        bar = "  ".join(f"{c}={p:.2f}" for c, p in zip(cfg["classes"], probs))
        print(f"\r[edge] {bar}", end="", flush=True)
        for f in fired:
            t_now = total[0] / sr
            if t_now - last_fire.get(f["class"], -1e9) < a.cooldown:
                continue  # тот же источник: не тратим связь на повтор
            last_fire[f["class"]] = t_now
            print()
            clip_t0 = t_start + timedelta(seconds=(total[0] - len(buf)) / sr)  # GPS-время первого отсчёта клипа
            now = t_start + timedelta(seconds=total[0] / sr)
            win_s = cfg["inference"]["window_seconds"]
            alert = to_alerts([{"class": f["class"], "start": -win_s, "end": 0.0, "confidence": f["confidence"]}],
                              cfg, a.sensor_id, a.lat, a.lon, now)[0]
            data = np.array(buf, dtype=np.float32)
            if a.save_clips:
                clip = f"{a.save_clips}/{alert['id']}.wav"
                save_audio(clip, data, sr)
                alert["evidence_audio"] = clip
            if a.post:
                import soundfile as sf
                b = io.BytesIO()
                sf.write(b, data, sr, format="WAV", subtype="PCM_16")
                alert["clip_b64"] = base64.b64encode(b.getvalue()).decode()
                alert["clip_start_ts"] = clip_t0.isoformat()
            _send(alert, a.post)

    if a.mic:
        import sounddevice as sd

        with sd.InputStream(samplerate=sr, channels=1, blocksize=hop, dtype="float32") as stream:
            print("[edge] слушаю микрофон, Ctrl+C для выхода")
            while True:
                chunk, _ = stream.read(hop)
                handle(chunk[:, 0])
    elif a.file:
        x = load_audio(a.file, sr)
        for i in range(0, len(x) - hop + 1, hop):
            handle(x[i : i + hop])
            if a.realtime:
                time.sleep(hop / sr)
        print()
    else:
        ap.error("нужен --mic или --file")


if __name__ == "__main__":
    main()
