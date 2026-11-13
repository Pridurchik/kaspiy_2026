"""«Эко-Щит Каспия» — сервер цифрового помощника инспектора: дашборд + API всех источников.

Запуск:  make demo   (или: python -m caspian_audio.cli demo && uvicorn caspian_audio.inspector:app --port 8000)
Дашборд: http://localhost:8000/

Источники -> единая лента алертов (одинаковый формат, балл риска с причинами):
  акустические буи    POST /api/alerts        алерт + 8-секундный клип; сервер перепроверяет клип
                                              строгими порогами и, если событие прислали ≥3 буя,
                                              считает пеленг (TDOA) по их клипам
  датчики воды        POST /api/water         показания станций -> детектор сбросов
  спутник             (cli demo / satellite)  шлейфы мутности Sentinel-2
  прогноз             GET  /api/forecast      нерест по гидрологии -> окна запрета, план патрулей
Решения инспектора   POST /api/alerts/{id}/status  -> подтверждённые / ложные клипы идут в датасет
                                              дообучения (data/feedback/<класс>/)
Живые обновления      GET  /api/stream         Server-Sent Events
"""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import Body, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audio_io import load_audio, load_audio_bytes, save_audio
from .detect import StreamingDetector
from .fusion import Fusion
from .geo import LocalFrame, iso, parse_ts
from .serve import MODEL, _full_cfg, _predict, _cfg, app  # акустические эндпоинты /v1/... уже в app
from .store import Store

DB = os.environ.get("INSPECTOR_DB", "data/inspector.db")
store = Store(DB)
DASH = Path(__file__).with_name("dashboard")
app.title = "Эко-Щит Каспия — помощник инспектора ООПТ"
app.mount("/static", StaticFiles(directory=DASH), name="static")

_cfg_strict = {**_cfg, "inference": _full_cfg["inference"]}  # сервер: строгие пороги из YAML
_strict = StreamingDetector(_predict, _cfg_strict)
_frame = LocalFrame(*_full_cfg["fusion"]["origin"])

# ---------------------------------------------------------------- live-события (SSE)
_subs: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def publish(evt: dict) -> None:
    if _loop is None:
        return
    data = json.dumps(evt, ensure_ascii=False, default=float)
    for q in list(_subs):
        _loop.call_soon_threadsafe(q.put_nowait, data)


@app.get("/api/stream")
async def stream(request: Request):
    global _loop
    _loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _subs.add(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    yield f"data: {await asyncio.wait_for(q.get(), 15)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _subs.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- дашборд и обзор

@app.get("/", response_class=HTMLResponse)
def index():
    return (DASH / "index.html").read_text(encoding="utf-8")


@app.get("/api/overview")
def overview():
    ov = store.get("overview")
    if not ov:
        raise HTTPException(404, "нет данных: запустите `python -m caspian_audio.cli demo`")
    return {**ov, "feedback": store.feedback_stats()}


@app.get("/api/alerts")
def alerts(status: str | None = None, limit: int = 200):
    return store.alerts(status, limit)


@app.get("/api/alerts/{aid}")
def alert(aid: str):
    a = store.alert(aid)
    if not a:
        raise HTTPException(404)
    return a


@app.get("/api/alerts/{aid}/audio")
def alert_audio(aid: str):
    a = store.alert(aid)
    if not a or not a.get("evidence_audio") or not Path(a["evidence_audio"]).exists():
        raise HTTPException(404, "нет аудио-доказательства")
    return FileResponse(a["evidence_audio"], media_type="audio/wav")


@app.get("/api/alerts/{aid}/spectrogram.png")
def alert_spectrogram(aid: str):
    from .imaging import spectrogram_png

    a = store.alert(aid)
    if not a or not a.get("evidence_audio") or not Path(a["evidence_audio"]).exists():
        raise HTTPException(404)
    return Response(spectrogram_png(load_audio(a["evidence_audio"], _cfg["sample_rate"]), _cfg["sample_rate"]), media_type="image/png")


FEEDBACK_DIR = Path("data/feedback")
STATUSES = {"new", "in_progress", "confirmed", "false_alarm", "closed"}


@app.post("/api/alerts/{aid}/status")
def alert_status(aid: str, body: dict = Body(..., examples=[{"status": "confirmed", "note": "задержана лодка"}])):
    st = body.get("status")
    if st not in STATUSES:
        raise HTTPException(400, f"status из {sorted(STATUSES)}")
    a = store.set_status(aid, st, body.get("note", ""))
    if not a:
        raise HTTPException(404)
    # петля обучения: решение инспектора = разметка. Подтверждённый клип -> его класс,
    # ложная тревога -> «трудный негатив» в фон. Дальше: prepare --sources ... folder:data/feedback
    clip = a.get("evidence_audio")
    if clip and Path(clip).exists() and st in ("confirmed", "false_alarm"):
        label = a["category"] if st == "confirmed" else "background"
        if label in _full_cfg["classes"]:
            dst = FEEDBACK_DIR / label / f"{aid}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(clip, dst)
            a["feedback_clip"] = str(dst)
            store.upsert_alert(a)
    publish({"type": "status", "alert": a})
    return a


# ---------------------------------------------------------------- приём с буёв (каскад + пеленг)

_recent: list[dict] = []  # последние клипы с буёв для пеленга


def _fusion() -> Fusion:
    fu = Fusion(_full_cfg, store.get("ais_rows"))
    fc = store.get("forecast")
    if fc:
        fu.apply_forecast(fc.get("windows", {}))
    return fu


@app.post("/api/alerts")
def ingest_alert(a: dict = Body(...)):
    """Алерт с буя (edge.py --post). Поля: category, confidence, sensor_id, location, start_ts;
    clip_b64 (WAV) и clip_start_ts (GPS-время первого отсчёта клипа) — для перепроверки и пеленга."""
    a = {k: v for k, v in a.items()}
    a.setdefault("id", str(uuid.uuid4()))
    a["type"] = "acoustic"
    sr = _cfg["sample_rate"]
    clip = None
    if a.get("clip_b64"):
        clip = load_audio_bytes(base64.b64decode(a.pop("clip_b64")), sr)
        path = Path("data/inbox") / f"{a['id']}.wav"
        save_audio(path, clip, sr)
        a["evidence_audio"] = str(path)
        # каскад: сервер перепроверяет клип строгими порогами
        ev = [e for e in _strict.run(clip)["events"] if e["class"] == a["category"]]
        a["server_check"] = {"confirmed": bool(ev), "confidence": max((e["confidence"] for e in ev), default=0.0)}
        if a["category"] == "boat_motor":
            from .vessel import profile
            a["vessel"] = profile(clip, sr)
    fu = _fusion()
    fixes = None
    if clip is not None and a.get("clip_start_ts") and a.get("location"):
        fixes = _try_localize(a, clip)
    a = fu.assess(a, fixes)
    if a.get("server_check") and not a["server_check"]["confirmed"]:
        a["fusion"]["reasons"].append({"factor": "server_check", "points": 0, "text": "Сервер не подтвердил клип строгими порогами — проверить вручную"})
        a["fusion"]["priority"] = "low"
    store.upsert_alert(a)
    publish({"type": "alert", "alert": a})
    return a


def _try_localize(a: dict, clip: np.ndarray) -> list[dict] | None:
    """Если то же событие пришло с ≥3 буёв — общий таймлайн по GPS-времени клипов и TDOA."""
    from .localize import kalman_track, locate_span

    sr = _cfg["sample_rate"]
    t_start = parse_ts(a["clip_start_ts"])
    _recent.append({"buoy": a["sensor_id"], "lat": a["location"]["lat"], "lon": a["location"]["lon"], "t": t_start,
                    "x": clip, "category": a["category"], "id": a["id"], "seen": time.time()})
    del _recent[:-50]
    group = {}
    for r in _recent:
        if r["category"] == a["category"] and abs(r["t"] - t_start) < 12 and r["buoy"] not in group:
            group[r["buoy"]] = r
    if len(group) < 3:
        return None
    rs = list(group.values())
    t0 = min(r["t"] for r in rs)
    t1 = max(r["t"] + len(r["x"]) / sr for r in rs)
    n = int((t1 - t0) * sr) + 1
    sigs = []
    for r in rs:
        s = np.zeros(n, np.float32)
        i = int(round((r["t"] - t0) * sr))
        s[i: i + len(r["x"])] = r["x"][: n - i]
        sigs.append(s)
    bxy = np.array([_frame.to_xy(r["lat"], r["lon"]) for r in rs], float)
    # общий участок, где есть звук со всех буёв
    lo = max(r["t"] for r in rs) - t0
    hi = min(r["t"] + len(r["x"]) / sr for r in rs) - t0
    if hi - lo < 1.0:
        return None
    lcfg = _full_cfg["localization"]
    fixes = locate_span(sigs, bxy, sr, lo, hi, a["category"], lcfg)
    if not fixes:
        return None
    impulsive = a["category"] in lcfg.get("impulsive_classes", ["gunshot"])
    track = fixes if impulsive else kalman_track(fixes, lcfg["accel_noise"])
    last = track[-1]
    lat, lon = _frame.to_ll(last["x"], last["y"])
    a["location"] = {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "source": "tdoa", "radius_m": round(last["radius_m"], 1)}
    a["heard_by"] = [r["buoy"] for r in rs]
    a["localization"] = {"method": "tdoa-gcc-phat", "n_fixes": len(fixes),
                         "track": [{"ts": iso(t0 + p["t"]), "lat": round(float(la), 6), "lon": round(float(lo_), 6), "radius_m": round(p["radius_m"], 1)}
                                   for p in track for la, lo_ in [_frame.to_ll(p["x"], p["y"])]]}
    # алерты остальных буёв про это же событие — сливаем в этот
    for r in rs:
        if r["id"] != a["id"]:
            o = store.alert(r["id"])
            if o:
                o["hidden"], o["merged_into"] = True, a["id"]
                store.upsert_alert(o)
                publish({"type": "merged", "id": o["id"], "into": a["id"]})
    return [{"t": t0 + p["t"], "x": p["x"], "y": p["y"], "radius_m": p["radius_m"],
             **({"vx": p["vx"], "vy": p["vy"]} if "vx" in p else {})} for p in track]


# ---------------------------------------------------------------- вода

@app.get("/api/water/{station}")
def water_series(station: str, hours: int = 72):
    from .water import PARAMS

    wd = store.get("water_dir")
    if not wd:
        raise HTTPException(404)
    rows = [r for r in csv.DictReader(open(Path(wd) / "water.csv", encoding="utf-8")) if r["station"] == station]
    rows = rows[-hours * 6:]
    return {"station": station, "ts": [r["ts"] for r in rows], **{p: [float(r[p]) for r in rows] for p in PARAMS},
            "params": {p: {"label": v[0], "unit": v[1]} for p, v in PARAMS.items()}}


@app.post("/api/water")
def water_ingest(body: dict = Body(..., examples=[{"station": "W2", "readings": [{"ts": "2026-05-20T06:10:00+04:00", "temp_c": 18.4,
                                                                                    "ph": 7.8, "do_mg_l": 7.9, "turbidity_ntu": 20,
                                                                                    "cond_us_cm": 425, "level_cm": 221}]}])):
    """Показания станции -> дописываем в ряд, перезапускаем детектор, новые сбросы — в ленту."""
    from .water import PARAMS, detect, load_csv

    wd = store.get("water_dir")
    if not wd:
        raise HTTPException(404)
    path = Path(wd) / "water.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "station", *PARAMS])
        for r in body["readings"]:
            w.writerow({"ts": r["ts"], "station": body["station"], **{p: r.get(p, "") for p in PARAMS}})
    ov = store.get("overview")
    res = detect(load_csv(path), ov["stations"])
    known = {(x.get("sensor_id"), x.get("start_ts"), x["category"]) for x in store.alerts(limit=1000, visible_only=False)}
    new = []
    for al in res["alerts"]:
        if (al.get("sensor_id"), al.get("start_ts"), al["category"]) not in known:
            store.upsert_alert(al)
            publish({"type": "alert", "alert": al})
            new.append(al)
    ov["water_status"] = res["status"]
    store.put("overview", ov)
    publish({"type": "water", "status": res["status"]})
    return {"new_alerts": new, "status": res["status"][body["station"]]}


# ---------------------------------------------------------------- прогноз и спутник

@app.get("/api/forecast")
def get_forecast():
    fc = store.get("forecast")
    if not fc:
        raise HTTPException(404)
    return fc


@app.get("/api/satellite/overlay.png")
def sat_overlay():
    p = Path("data/demo/satellite/overlay.png")
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


# ---------------------------------------------------------------- демо: проигрывание ночи

@app.post("/api/demo/replay")
async def demo_replay(speed: float = 1.0):
    """Скрыть алерты и выдать их заново по одному — как они приходили ночью (для показа жюри)."""
    global _loop
    _loop = asyncio.get_running_loop()
    items = sorted([a for a in store.alerts(limit=1000, visible_only=False) if not a.get("merged_into")],
                   key=lambda a: parse_ts(a["start_ts"]))
    for a in items:
        a["hidden"], a["status"] = True, "new"
        a.pop("history", None)
        store.upsert_alert(a)
    publish({"type": "reset"})

    async def run():
        for a in items:
            await asyncio.sleep(3.5 / speed)
            a["hidden"] = False
            a["received_at"] = time.time()
            store.upsert_alert(a)
            publish({"type": "alert", "alert": a, "replay": True})
        publish({"type": "replay_done"})

    asyncio.create_task(run())
    return {"queued": len(items), "seconds": round(len(items) * 3.5 / speed)}


# ---------------------------------------------------------------- Каспий: миграция, браконьерство, трансграничный обмен

@app.get("/api/caspian")
def caspian_meta():
    m = store.get("caspian")
    if not m:
        raise HTTPException(404, "нет данных: python -m caspian_audio.cli demo")
    return m


@app.get("/api/caspian/layer/{name}")
def caspian_layer(name: str):
    p = Path("data/demo/caspian") / name
    if not name.endswith(".png") or "/" in name or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/transboundary")
def transboundary_list():
    return store.get("transboundary", [])


@app.post("/api/transboundary/{mid}/status")
def transboundary_status(mid: str, body: dict = Body(..., examples=[{"status": "approved", "by": "диспетчер"}])):
    """Решение диспетчера по черновику: approved / rejected. Отправка соседям — по согласованному
    каналу, вне этой системы: здесь ничего не уходит наружу автоматически."""
    from .transboundary import digest

    st = body.get("status")
    if st not in ("approved", "rejected", "draft"):
        raise HTTPException(400, "status: approved | rejected | draft")
    msgs = store.get("transboundary", [])
    for m in msgs:
        if m["id"] == mid:
            if digest(m) != m["sha256"]:
                raise HTTPException(409, "контрольная сумма не совпадает: сообщение изменено")
            m["status"] = st
            m.setdefault("history", []).append({"status": st, "by": body.get("by", "диспетчер"), "ts": time.time()})
            store.put("transboundary", msgs)
            publish({"type": "transboundary", "message": m})
            return m
    raise HTTPException(404)


@app.get("/api/transboundary/{mid}.json")
def transboundary_export(mid: str):
    m = next((m for m in store.get("transboundary", []) if m["id"] == mid), None)
    if not m:
        raise HTTPException(404)
    return Response(json.dumps({k: v for k, v in m.items() if k != "history"}, ensure_ascii=False, indent=2),
                    media_type="application/json", headers={"Content-Disposition": f'attachment; filename="caspian-alert-{mid[:8]}.json"'})


_live = {"proc": None}


@app.post("/api/demo/live")
def demo_live():
    """«Буи в эфире»: три клиента буя в реальном времени проигрывают записанную ночь
    и шлют алерты сюда — видно, как система реагирует на вторжение по мере звука."""
    import subprocess
    import sys

    p = _live["proc"]
    if p is not None and p.poll() is None:
        return {"running": True}
    _live["proc"] = subprocess.Popen([sys.executable, "-m", "caspian_audio.cli", "buoys", "--realtime",
                                      "--url", "http://127.0.0.1:8000/api/alerts"], env={**os.environ, "PYTHONPATH": "src"})
    return {"running": True, "seconds": 150}
