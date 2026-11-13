"""HTTP API для дашборда: загрузили аудио с буя -> получили события, алерты и таймлайн.

Запуск:  MODEL=runs/audio/model.onnx uvicorn caspian_audio.serve:app --port 8000

  POST /v1/acoustic/detect   один буй: события, алерты (+ фьюжн: зоны, ночь, AIS если приложен CSV)
  POST /v1/acoustic/array    сеть буёв (WAV с общим GPS-временем): пеленг, трек, «тёмная лодка»
  POST /v1/fusion/assess     готовые алерты + AIS -> опознание и балл риска (для любого источника)
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

from .audio_io import load_audio_bytes, load_config
from .detect import StreamingDetector, load_ood, load_predictor, to_alerts
from .fusion import Fusion

MODEL = os.environ.get("MODEL", "runs/audio/model.onnx")
if not os.path.exists(MODEL) and os.path.exists("runs/audio/best.pt"):
    MODEL = "runs/audio/best.pt"

CONFIG = os.environ.get("CONFIG", "configs/audio.yaml")

app = FastAPI(title="Caspian OOPT — Acoustic AI", version="0.2.0")
_predict, _cfg = load_predictor(MODEL)
_detector = StreamingDetector(_predict, _cfg, ood=load_ood(MODEL))
_full_cfg = load_config(CONFIG)  # секции localization / fusion (зоны, веса риска)


def _ais_rows(data: bytes | None) -> list[dict] | None:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8")))) if data else None


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
    ais: UploadFile | None = File(None, description="CSV AIS: mmsi, ts, lat, lon, name"),
):
    try:
        x = load_audio_bytes(await file.read(), _cfg["sample_rate"])
    except Exception as e:
        raise HTTPException(400, f"не удалось прочитать аудио: {e}")
    res = _detector.run(x)
    t0 = datetime.fromisoformat(recorded_at) if recorded_at else None
    fusion = Fusion(_full_cfg, _ais_rows(await ais.read() if ais else None))
    return {
        "sensor_id": sensor_id,
        "duration_s": res["duration"],
        "events": res["events"],
        "alerts": [fusion.assess(a) for a in to_alerts(res["events"], _cfg, sensor_id, lat, lon, t0)],
        # Таймлайн для графика на дашборде: вероятность каждого класса по окнам
        "timeline": {
            "t": [round(float(t), 2) for t in res["times"]],
            **{c: [round(float(v), 3) for v in res["probs"][:, i]] for i, c in enumerate(_cfg["classes"])},
        },
    }


@app.post("/v1/acoustic/array")
async def array(
    files: list[UploadFile] = File(..., description="WAV каждого буя, общее время начала (GPS)"),
    buoys: str = Form(..., description='JSON [{"id": "buoy-01", "lat": 45.75, "lon": 48.4}, ...] в порядке files'),
    recorded_at: str = Form(..., description="ISO-время начала записи (общее для всех буёв)"),
    ais: UploadFile | None = File(None),
):
    from .array import run_array

    meta_b = json.loads(buoys)
    if len(meta_b) != len(files):
        raise HTTPException(400, "число файлов и буёв не совпадает")
    with tempfile.TemporaryDirectory() as d:
        for b, f in zip(meta_b, files):
            b["file"] = f"{b['id']}.wav"
            Path(d, b["file"]).write_bytes(await f.read())
        Path(d, "array.json").write_text(json.dumps({"recorded_at": recorded_at, "buoys": meta_b}))
        ais_path = None
        if ais:
            ais_path = str(Path(d, "ais.csv"))
            Path(ais_path).write_bytes(await ais.read())
        res = run_array(MODEL, d, _full_cfg, ais_path)
    return {"alerts": [{k: v for k, v in a.items() if k != "_xy"} for a in res["alerts"]]}


@app.post("/v1/fusion/assess")
def fusion_assess(payload: dict = Body(..., examples=[{"alerts": [], "ais": [{"mmsi": "273318450", "ts": "2026-05-19T21:30:00Z",
                                                                             "lat": 45.75, "lon": 48.39, "name": "ВОЛЖСКИЙ-7"}]}])):
    """Алерты из любого источника (буй, камера, патруль) + AIS -> опознание, зоны, балл риска.
    В алерте нужны category, confidence, start_ts, end_ts, location; можно передать `fixes`
    ([{ts, lat, lon, radius_m}]) — точки пеленга."""
    fusion = Fusion(_full_cfg, payload.get("ais"))
    out = []
    for a in payload.get("alerts", []):
        fixes = None
        if a.get("fixes"):
            from .geo import parse_ts
            fixes = [{"t": parse_ts(f["ts"]), **dict(zip(("x", "y"), map(float, fusion.frame.to_xy(f["lat"], f["lon"])))),
                      "radius_m": f.get("radius_m", 50)} for f in a["fixes"]]
        out.append(fusion.assess({k: v for k, v in a.items() if k != "fixes"}, fixes))
    return {"alerts": sorted(out, key=lambda a: -(a["fusion"].get("risk") or 0))}
