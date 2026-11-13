"""Сборка демо-сценария для дашборда: все источники одной ночи на одном участке дельты.

Ночь на 20 мая 2026 (демо-время 06:00):
  * акустика: сеть из 3 буёв — теплоход с AIS, «тёмная лодка» на нерестилище, выстрел;
  * датчики воды: ночной сброс у рыбоцеха (W2), шлейф дошёл до нерестилища (W3), сбой датчика на W4;
  * спутник Sentinel-2 (11:40 местного предыдущего прохода сравнения — 10 мая): шлейф мутности;
  * прогноз: нерест сазана и осетра идёт, окна запрета и план патрулей на 2 недели.
Все данные синтетические и помечены как демо; модули принимают и реальные данные (см. README).
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import forecast as F
from . import satellite as S
from . import water as W
from .audio_io import load_audio, save_audio
from .fusion import ais_tracks, load_ais_csv
from .geo import LocalFrame, in_zone, iso, load_zones, parse_ts
from .store import Store

DEMO_DIR = Path("data/demo")
SCENARIO_TODAY = date(2026, 5, 20)


def build(cfg: dict, model_path: str, store: Store, demo_dir: str | Path = DEMO_DIR, fresh: bool = False) -> dict:
    from .array import run_array
    from .array_sim import simulate as array_sim

    d = Path(demo_dir)
    tz = timezone(timedelta(hours=cfg["fusion"].get("utc_offset_hours", 4)))
    frame = LocalFrame(*cfg["fusion"]["origin"])
    zones = load_zones(cfg["fusion"]["zones"], frame)
    store.reset()

    # 1) прогноз нереста (нужен раньше всех: задаёт окна запрета для фьюжна)
    hydro = F.simulate_hydrology(SCENARIO_TODAY)
    fc = F.forecast(hydro, SCENARIO_TODAY)
    windows = F.zone_windows(fc, zones)

    # 2) акустика: сеть буёв
    arr = d / "array"
    if fresh or not (arr / "array.json").exists():
        array_sim(arr, cfg)
    res = run_array(model_path, arr, cfg, forecast_windows=windows)
    meta = res["array"]
    clips = d / "clips"
    sr = cfg["sample_rate"]
    for a in res["alerts"]:
        a.pop("_xy", None)
        a["type"] = "acoustic"
        # аудио-доказательство: 8 с с буя, который слышал событие
        b = next(x for x in meta["buoys"] if x["id"] == (a.get("heard_by") or [a["sensor_id"]])[0])
        x = load_audio(arr / b["file"], sr)
        t0 = max(0.0, a["offset_s"][0] - 1.0)
        seg = x[int(t0 * sr): int((t0 + 8) * sr)]
        save_audio(clips / f"{a['id']}.wav", seg, sr)
        a["evidence_audio"] = str(clips / f"{a['id']}.wav")
        store.upsert_alert(a)

    # 3) датчики воды
    wd = d / "water"
    if fresh or not (wd / "water.csv").exists():
        W.simulate(cfg, wd)
    stations = json.loads((wd / "stations.json").read_text(encoding="utf-8"))
    wres = W.detect(W.load_csv(wd / "water.csv"), stations)
    for a in wres["alerts"]:
        store.upsert_alert(a)

    # 4) спутник
    sd = d / "satellite"
    if fresh or not (sd / "s2_2026-05-20.npz").exists():
        S.simulate(cfg, sd)
    sres = S.detect(sd / "s2_2026-05-10.npz", sd / "s2_2026-05-20.npz", cfg)
    S.write_overlay_png(sres, sd / "overlay.png")
    for a in S.to_alerts(sres, wres["alerts"], stations, cfg):
        store.upsert_alert(a)

    # 5) план патрулей с учётом браконьерских алертов в зонах
    all_alerts = store.alerts(limit=1000)
    plan = F.patrol_plan(fc, zones, all_alerts, frame)

    # 6) AIS: треки и последние позиции
    tracks = ais_tracks(load_ais_csv(arr / "ais.csv"), frame)
    ais = []
    for tr in tracks.values():
        lat, lon = frame.to_ll(tr["x"], tr["y"])
        ais.append({"mmsi": tr["mmsi"], "name": tr["name"], "track": [[iso(t), round(float(a), 6), round(float(b), 6)]
                                                                       for t, a, b in zip(tr["t"], lat, lon)]})

    # 7) буи: статус (в демо — заряд и связь)
    buoys = [{**b, "online": True, "battery_pct": p, "last_seen": iso(parse_ts(meta["recorded_at"]) + 150), "sensors": ["гидрофон", "микрофон", "GPS"]}
             for b, p in zip(meta["buoys"], (82, 64, 91))]
    buoys.append({"id": "buoy-04", "lat": round(float(frame.to_ll(-300, 1900)[0]), 6), "lon": round(float(frame.to_ll(-300, 1900)[1]), 6),
                  "online": False, "battery_pct": 7, "last_seen": iso(parse_ts(meta["recorded_at"]) - 5 * 3600), "sensors": ["гидрофон", "GPS"],
                  "note": "нет связи 5 ч, низкий заряд — нужен выезд на обслуживание"})

    # 8) экостатус по зонам: вода + алерты + нерест
    eco = eco_status(zones, frame, wres["status"], stations, store.alerts(limit=1000), windows, SCENARIO_TODAY)

    scenario_now = datetime(2026, 5, 20, 6, 0, tzinfo=tz)
    store.put("overview", {
        "scenario_time": scenario_now.isoformat(), "demo": True, "origin": cfg["fusion"]["origin"],
        "zones": json.loads(Path(cfg["fusion"]["zones"]).read_text(encoding="utf-8")),
        "buoys": buoys, "stations": stations, "water_status": wres["status"], "ais": ais, "eco": eco,
        "satellite": {"date": sres["date"], "baseline": sres["baseline"], "bounds": sres["bounds"], "overlay": "/api/satellite/overlay.png",
                      "cloud_fraction": sres["cloud_fraction"], "plumes": [{k: v for k, v in p.items() if k != "xy"} for p in sres["plumes"]]},
    })
    store.put("forecast", {**fc, "patrol_plan": plan, "windows": windows})
    store.put("ais_rows", load_ais_csv(arr / "ais.csv"))
    store.put("water_dir", str(wd))
    print(f"[demo] алертов: {len(store.alerts(limit=1000))}; экостатус: " + ", ".join(f"{e['name']}={e['level']}" for e in eco))
    return {"alerts": len(store.alerts(limit=1000))}


def eco_status(zones, frame, wstatus, stations, alerts, windows, today) -> list[dict]:
    out = []
    for z in zones:
        reasons, level = [], "green"
        for st in stations:
            x, y = frame.to_xy(st["lat"], st["lon"])
            if in_zone(z, float(x), float(y)):
                ws = wstatus.get(st["id"], {})
                if ws.get("level") in ("red", "yellow"):
                    reasons.append(f"Вода ({st['id']}): " + ("нарушение норматива" if ws.get("violations") else "аномалия качества"))
                    level = max(level, ws["level"], key=["green", "yellow", "red"].index)
        n_hi = 0
        for a in alerts:
            loc = a.get("location")
            if not loc or a.get("category") == "sensor_fault":
                continue
            x, y = frame.to_xy(loc["lat"], loc["lon"])
            if in_zone(z, float(x), float(y)) and (a.get("fusion") or {}).get("priority") in ("critical", "high"):
                n_hi += 1
        if n_hi:
            reasons.append(f"Тревог высокого приоритета: {n_hi}")
            level = "red"
        fcd = windows.get(z["name"], {}).get(today.isoformat())
        if fcd:
            reasons.append(f"Идёт нерест: {fcd['species'].lower()} ({fcd['p']:.0%}) — зона особо уязвима")
            if level == "green":
                level = "yellow"
        out.append({"name": z["name"], "kind": z.get("kind"), "level": level, "reasons": reasons or ["Отклонений нет"]})
    return out
