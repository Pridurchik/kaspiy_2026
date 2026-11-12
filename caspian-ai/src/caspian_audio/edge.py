"""Клиент "акустического буя": слушает микрофон/гидрофон (или файл как поток),
детектирует на устройстве и отправляет только алерты на сервер дашборда.

  python -m caspian_audio.edge --model runs/audio/model.onnx --mic
  python -m caspian_audio.edge --model runs/audio/model.onnx --file data/scene.wav --realtime
  ... --post http://localhost:8080/api/alerts   # куда слать JSON-алерты
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .audio_io import load_audio, save_audio
from .detect import OnlineDetector, load_ood, load_predictor, to_alerts


def _send(alert: dict, url: str | None):
    print(json.dumps(alert, ensure_ascii=False))
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

    def handle(chunk: np.ndarray):
        buf.extend(chunk.tolist())
        if len(buf) < win:
            return
        window = np.array(list(buf)[-win:], dtype=np.float32)
        probs, fired = det.push(window)
        bar = "  ".join(f"{c}={p:.2f}" for c, p in zip(cfg["classes"], probs))
        print(f"\r[edge] {bar}", end="", flush=True)
        for f in fired:
            print()
            now = datetime.now(timezone.utc)
            alert = to_alerts([{"class": f["class"], "start": 0.0, "end": 0.0, "confidence": f["confidence"]}],
                              cfg, a.sensor_id, a.lat, a.lon, now)[0]
            if a.save_clips:
                clip = f"{a.save_clips}/{alert['id']}.wav"
                save_audio(clip, np.array(buf, dtype=np.float32), sr)
                alert["evidence_audio"] = clip
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
