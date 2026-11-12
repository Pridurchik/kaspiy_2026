"""HTTP API для дашборда: загрузили аудио с буя -> получили события, алерты и таймлайн.

Запуск:  MODEL=runs/audio/model.onnx uvicorn caspian_audio.serve:app --port 8000
"""
from __future__ import annotations

import os
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .audio_io import load_audio_bytes
from .detect import StreamingDetector, load_ood, load_predictor, to_alerts

MODEL = os.environ.get("MODEL", "runs/audio/model.onnx")
if not os.path.exists(MODEL) and os.path.exists("runs/audio/best.pt"):
    MODEL = "runs/audio/best.pt"

app = FastAPI(title="Caspian OOPT — Acoustic AI", version="0.1.0")
_predict, _cfg = load_predictor(MODEL)
_detector = StreamingDetector(_predict, _cfg, ood=load_ood(MODEL))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "classes": _cfg["classes"], "sample_rate": _cfg["sample_rate"]}


@app.post("/v1/acoustic/detect")
async def detect(
    file: UploadFile = File(...),
    sensor_id: str = Form("buoy-01"),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    recorded_at: str | None = Form(None, description="ISO-время начала записи"),
):
    try:
        x = load_audio_bytes(await file.read(), _cfg["sample_rate"])
    except Exception as e:
        raise HTTPException(400, f"не удалось прочитать аудио: {e}")
    res = _detector.run(x)
    t0 = datetime.fromisoformat(recorded_at) if recorded_at else None
    return {
        "sensor_id": sensor_id,
        "duration_s": res["duration"],
        "events": res["events"],
        "alerts": to_alerts(res["events"], _cfg, sensor_id, lat, lon, t0),
        # Таймлайн для графика на дашборде: вероятность каждого класса по окнам
        "timeline": {
            "t": [round(float(t), 2) for t in res["times"]],
            **{c: [round(float(v), 3) for v in res["probs"][:, i]] for i, c in enumerate(_cfg["classes"])},
        },
    }
