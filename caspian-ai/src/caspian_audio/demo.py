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
        if a["category"] == "boat_motor":
            from .vessel import profile
            a["vessel"] = profile(seg, sr)  # тип двигателя и направление по гидроакустике
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
    build_caspian(store)
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


def build_caspian(store: Store, today: date = SCENARIO_TODAY, out_dir: str | Path = DEMO_DIR / "caspian", horizon: int = 7) -> dict:
    """Масштаб всего Каспия: миграция осетровых, браконьерские горячие точки, трансграничные уведомления."""
    from . import caspian as C
    from . import migration as M
    from . import poaching as P
    from . import transboundary as T

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = C.build_grid()
    sensors = C.simulate_sensors(g, today)
    env = C.environment(g, today, sensors)
    (out / "temp.png").write_bytes(C.to_png(g, env["temp"] - np.nanmin(env["temp"][g.sea]), alpha_min=0.0))
    (out / "sal.png").write_bytes(C.to_png(g, env["sal"], vmax=14, alpha_min=0.0))

    stages = {}
    for stage in ("spawners", "juvenile"):
        r = M.forecast(g, today, horizon, stage, sensors)
        vmax = float(np.percentile(np.concatenate([m[g.sea] for m in r["maps"]]), 99.5))
        for k, m in enumerate(r["maps"]):
            (out / f"{stage}_d{k}.png").write_bytes(C.to_png(g, m, vmax=vmax))
        stages[stage] = r

    # обучение пригодности на наблюдениях (демо: синтетические съёмки молоди прошлым летом)
    days = [date(2025, 7, 1) + timedelta(days=7 * k) for k in range(8)]
    obs = M.simulate_observations(g, days, "juvenile")
    train = [o for o in obs if o["date"] < "2025-08-10"]
    test = [o for o in obs if o["date"] >= "2025-08-10"]
    hm = M.fit_observations(train)
    pt = M.predict_observations(hm, {k: np.array([o[k] for o in test]) for k in ("temp", "sal", "depth")})
    habitat_fit = {"n_obs": len(obs), "test_auc": round(M.auc(pt, np.array([o["present"] for o in test])), 3),
                   "learned_opt_sal": hm["opt_sal"], "learned_opt_temp": hm["opt_temp"], "synthetic": True}

    # браконьерство: история (синтетика) -> модель -> горячие точки на ближайшие ночи
    fw = P.seasonal_fish(g)
    start = today - timedelta(days=730)
    incidents = P.simulate_history(g, fw, start, 730)
    pm = P.fit(g, incidents, fw, start, 730, 540)
    st = P.Static(g)
    risk_days = []
    for k in range(1, min(horizon, 3) + 1):
        d = today + timedelta(days=k)
        h = P.history_density(st, incidents, d)
        fish = stages["spawners"]["maps"][k]
        r = P.predict(st, pm, fish, h, d)
        (out / f"risk_d{k}.png").write_bytes(C.to_png(g, r, vmax=1.0, alpha_min=0.25))
        risk_days.append({"date": d.isoformat(), "moon": round(P.moon_illumination(d), 2), "wind": round(P.wind(d), 1),
                          "hotspots": P.hotspots(g, st, pm, r, fish, h, d, 8)})

    acoustic = [a for a in store.alerts(limit=1000) if a.get("type") == "acoustic"]
    msgs = T.build(today, "RU", g, stages["spawners"], risk_days[0]["hotspots"], acoustic)

    meta = {
        "today": today.isoformat(), "bounds": g.bounds, "res": g.res, "horizon": horizon,
        "countries": C.COUNTRIES, "rivers": {k: [v[1], v[0]] for k, v in C.RIVERS.items()},
        "sensors": sensors, "env_layers": {"temp": "/api/caspian/layer/temp.png", "sal": "/api/caspian/layer/sal.png"},
        "stages": {s: {"name": r["name"], "source": r["source"], "shift_km_day": r.get("shift_km_day"),
                       "shift_course_deg": r.get("shift_course_deg"), "days": r["days"]} for s, r in stages.items()},
        "habitat_fit": habitat_fit,
        "poaching": {"model": {k: pm[k] for k in ("auc_test", "hit_rate_top5", "n_incidents_train", "n_incidents_test")},
                     "explain": P.explain_model(pm), "days": risk_days, "synthetic_history": True,
                     "incidents_sample": [i for i in incidents if i["date"] >= (today - timedelta(days=365)).isoformat()][::3]},
        "note": "демо: среда, датчики, наблюдения и история инцидентов синтетические; контур моря схематичный",
    }
    store.put("caspian", meta)
    store.put("transboundary", msgs)
    print(f"[caspian] миграция: производители {stages['spawners']['days'][0]['shares']}; "
          f"браконьерство AUC {pm['auc_test']}, топ-5% площади ловит {pm['hit_rate_top5']:.0%} инцидентов; уведомлений: {len(msgs)}")
    return meta
