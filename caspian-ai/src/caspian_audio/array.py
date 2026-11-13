"""Сеть буёв: детекция на каждом -> объединение событий -> пеленг/трек -> фьюжн с AIS и зонами.

Вход — каталог с array.json (буи, координаты, общее время начала записи) и WAV каждого буя.
Выход — алерты с точкой, треком, опознанием и баллом риска + карта (HTML, без интернета).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_io import load_audio
from .detect import StreamingDetector, load_ood, load_predictor, to_alerts
from .fusion import Fusion, load_ais_csv, summary_line
from .geo import iso, parse_ts
from .localize import kalman_track, locate_span


def cluster_events(per_buoy: list[list[dict]], slack: float) -> list[dict]:
    """События одного класса с разных буёв, перекрывающиеся по времени (с запасом на задержку
    распространения), — это один источник."""
    allev = sorted(((e, b) for b, evs in enumerate(per_buoy) for e in evs), key=lambda p: p[0]["start"])
    clusters: list[dict] = []
    for e, b in allev:
        c = next((c for c in clusters if c["class"] == e["class"] and e["start"] <= c["end"] + slack), None)
        if c is None:
            clusters.append({"class": e["class"], "start": e["start"], "end": e["end"], "confidence": e["confidence"],
                             "buoys": {b}, "peak_time": e.get("peak_time", e["start"])})
        else:
            c["end"] = max(c["end"], e["end"])
            if e["confidence"] > c["confidence"]:
                c["confidence"], c["peak_time"] = e["confidence"], e.get("peak_time", e["start"])
            c["buoys"].add(b)
    return clusters


def run_array(model_path: str, array_dir: str | Path, cfg: dict, ais_path: str | None = None,
              zones_path: str | None = None, out_dir: str | Path | None = None,
              forecast_windows: dict | None = None) -> dict:
    array_dir = Path(array_dir)
    meta = json.loads((array_dir / "array.json").read_text(encoding="utf-8"))
    predict, mcfg = load_predictor(model_path)
    mcfg["inference"] = cfg["inference"]  # пороги из актуального YAML
    sr = mcfg["sample_rate"]
    lcfg = cfg["localization"]

    ais_file = ais_path or (array_dir / "ais.csv" if (array_dir / "ais.csv").exists() else None)
    fusion = Fusion(cfg, load_ais_csv(ais_file) if ais_file else None, zones_path)
    if forecast_windows:
        fusion.apply_forecast(forecast_windows)  # окна запрета по прогнозу нереста
    bxy = np.array([fusion.frame.to_xy(b["lat"], b["lon"]) for b in meta["buoys"]], float)
    signals = [load_audio(array_dir / b["file"], sr) for b in meta["buoys"]]
    t0 = parse_ts(meta["recorded_at"])

    det = StreamingDetector(predict, mcfg, ood=load_ood(model_path))
    per_buoy = [det.run(x)["events"] for x in signals]
    for b, evs in zip(meta["buoys"], per_buoy):
        print(f"[array] {b['id']}: " + (", ".join(f"{e['class']} {e['start']:.0f}–{e['end']:.0f}с" for e in evs) or "тихо"))

    span = float(np.max(np.linalg.norm(bxy[:, None] - bxy[None], axis=-1)))
    slack = span / min(lcfg["sound_speed"].values()) + 1.0
    clusters = cluster_events(per_buoy, slack)

    for c in clusters:
        c["fixes"] = locate_span(signals, bxy, sr, c["start"] - 0.5, c["end"] + 0.5, c["class"], lcfg) if len(signals) >= 3 else []
    clusters = merge_continuous(clusters, lcfg)

    alerts = []
    for c in clusters:
        fixes = c["fixes"]
        impulsive = c["class"] in lcfg.get("impulsive_classes", ["gunshot"])
        track = fixes if impulsive else kalman_track(fixes, lcfg["accel_noise"])
        for p in fixes + track:
            p["t_abs"] = t0 + p["t"]
        ev = {"class": c["class"], "start": c["start"], "end": c["end"], "confidence": c["confidence"]}
        sensor = meta["buoys"][min(c["buoys"])]
        alert = to_alerts([ev], mcfg, sensor["id"], sensor["lat"], sensor["lon"], recorded_at=_dt(t0))[0]
        alert["heard_by"] = [meta["buoys"][b]["id"] for b in sorted(c["buoys"])]
        alert["offset_s"] = [c["start"], c["end"]]
        if track:
            last = track[-1]
            lat, lon = fusion.frame.to_ll(last["x"], last["y"])
            alert["location"] = {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "source": "tdoa",
                                 "radius_m": round(last["radius_m"], 1)}
            alert["localization"] = {
                "method": "tdoa-gcc-phat", "n_fixes": len(fixes), "ambiguous": any(f["ambiguous"] for f in fixes),
                "track": [{"ts": iso(p["t_abs"]), "lat": round(float(la), 6), "lon": round(float(lo), 6), "radius_m": round(p["radius_m"], 1)}
                          for p in track for la, lo in [fusion.frame.to_ll(p["x"], p["y"])]],
            }
        else:
            alert["location"]["source"] = "sensor"
        fx = [{"t": p["t_abs"], "x": p["x"], "y": p["y"], "radius_m": p["radius_m"],
               **({"vx": p["vx"], "vy": p["vy"]} if "vx" in p else {})} for p in track] or None
        alert = fusion.assess(alert, fx)
        alert["_xy"] = {"fixes": [(p["x"], p["y"]) for p in fixes], "track": [(p["x"], p["y"]) for p in track],
                        "fix_t": [p["t"] for p in fixes], "track_t": [p["t"] for p in track]}
        alerts.append(alert)

    alerts = absorb_unknown(alerts)
    alerts.sort(key=lambda a: -(a["fusion"].get("risk") or 0))
    print("[array] алерты по приоритету:")
    for a in alerts:
        f = a["fusion"]
        print("   ", summary_line(a))
        for r in f["reasons"]:
            print(f"        {r['points']:+4d}  {r['text']}")
        if a.get("localization"):
            m = f.get("motion", {})
            print(f"        точка: {a['location']['lat']:.5f}, {a['location']['lon']:.5f} ±{a['location']['radius_m']:.0f} м"
                  f" ({a['localization']['n_fixes']} засечек)" + (f", {m['speed_kn']} уз, курс {m['course_deg']:.0f}°" if m else ""))

    res = {"array": meta, "alerts": alerts}
    truth_p = array_dir / "truth.json"
    if truth_p.exists():
        res["accuracy"] = localization_accuracy(alerts, json.loads(truth_p.read_text(encoding="utf-8")), fusion.frame, t0)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in a.items() if k != "_xy"} for a in alerts]
        (out / "array_alerts.json").write_text(json.dumps({**res, "alerts": clean}, ensure_ascii=False, indent=2, default=float))
        from .mapviz import render_map
        render_map(out / "map.html", meta, alerts, fusion, bxy, truth=json.loads(truth_p.read_text()) if truth_p.exists() else None)
        print(f"[array] -> {out / 'array_alerts.json'}, карта: {out / 'map.html'}")
    return res


def merge_continuous(clusters: list[dict], lcfg: dict, max_gap_s: float = 30.0, v_max: float = 20.0) -> list[dict]:
    """Склеить куски одного источника: детектор мог «моргнуть» посреди хода лодки. Куски одного
    класса сливаются, если второй начинается не позже 30 с и там, куда лодка могла доплыть."""
    impulsive = set(lcfg.get("impulsive_classes", ["gunshot"]))
    bridge = {"unknown_sound"}  # OOD срабатывает, когда уверенность в моторе проседает: это тот же источник
    out: list[dict] = []
    for c in sorted(clusters, key=lambda c: c["start"]):
        if c["class"] in impulsive:
            out.append(c)
            continue
        prev = next((p for p in reversed(out) if p["class"] not in impulsive | bridge
                     and (p["class"] == c["class"] or c["class"] in bridge)), None)
        if prev and prev["fixes"] and c["fixes"] and c["start"] - prev["end"] <= max_gap_s:
            b = min(c["fixes"], key=lambda f: f["t"])
            a = min(prev["fixes"], key=lambda f: abs(f["t"] - b["t"]))
            if np.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= v_max * max(abs(b["t"] - a["t"]), 1) + 150:
                prev["end"] = max(prev["end"], c["end"])
                prev["fixes"] = prev["fixes"] + c["fixes"]
                prev["buoys"] |= c["buoys"]
                if c["confidence"] > prev["confidence"]:
                    prev["confidence"] = c["confidence"]
                continue
        out.append(c)
    return out


def absorb_unknown(alerts: list[dict], dist_m: float = 150.0, dt_s: float = 5.0) -> list[dict]:
    """«Неизвестный звук» в той же точке и в то же время, что и опознанный источник, — это его же
    фрагменты (OOD срабатывает, когда уверенность в классе проседает). Отдельный алерт не нужен."""
    known = [a for a in alerts if a["category"] != "unknown_sound"]
    out = []
    for a in alerts:
        if a["category"] != "unknown_sound":
            out.append(a)
            continue
        host = None
        for k in known:
            if a["_xy"]["fixes"] and k["_xy"]["track"]:
                kt, kxy = np.array(k["_xy"]["track_t"]), np.array(k["_xy"]["track"])
                close = [np.any((np.abs(kt - t) <= dt_s) & (np.linalg.norm(kxy - xy, axis=1) <= dist_m))
                         for t, xy in zip(a["_xy"]["fix_t"], np.array(a["_xy"]["fixes"]))]
                if np.mean(close) >= 0.5:
                    host = k
                    break
            elif not a["_xy"]["fixes"] and parse_ts(a["start_ts"]) < parse_ts(k["end_ts"]) and parse_ts(k["start_ts"]) < parse_ts(a["end_ts"]):
                host = k
                break
        if host is None:
            out.append(a)
        else:
            host.setdefault("merged_unknown", []).append([a["start_ts"], a["end_ts"]])
    return out


def _dt(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, timezone.utc)


def localization_accuracy(alerts: list[dict], truth: dict, frame, t0: float) -> dict:
    """Ошибка пеленга относительно истинного положения источника в момент излучения (только симуляция)."""
    out = {}
    for src in truth["sources"]:
        tt, tx, ty = (np.array(src[k], float) for k in ("t", "x", "y"))
        best = None
        for a in alerts:
            if a["category"] != src["class"] or not a["_xy"]["fixes"]:
                continue
            ft = np.array(a["_xy"]["fix_t"])
            if len(tt) > 1 and not (ft.min() < tt.max() and ft.max() > tt.min()):
                continue
            if len(tt) == 1 and np.min(np.abs(ft - tt[0])) > 3:
                continue

            def err(xy, ts):
                xy, ts = np.array(xy), np.array(ts)
                gx, gy = (np.interp(ts, tt, tx), np.interp(ts, tt, ty)) if len(tt) > 1 else (tx[0], ty[0])
                return np.hypot(xy[:, 0] - gx, xy[:, 1] - gy)

            e_fix, e_trk = err(a["_xy"]["fixes"], a["_xy"]["fix_t"]), err(a["_xy"]["track"], a["_xy"]["track_t"])
            ident = a["fusion"]["identification"]
            best = {"alert_id": a["id"], "n_fixes": len(e_fix),
                    "fix_err_m": {"median": round(float(np.median(e_fix)), 1), "p90": round(float(np.percentile(e_fix, 90)), 1)},
                    "track_err_m": {"median": round(float(np.median(e_trk)), 1), "p90": round(float(np.percentile(e_trk, 90)), 1)},
                    "identification": ident.get("status"), "mmsi": ident.get("mmsi")}
            break
        out[src["id"]] = best or {"detected": False}
        b = out[src["id"]]
        if "fix_err_m" in b:
            ok = {"ais_ship": b["identification"] == "ais" and b["mmsi"] == src.get("mmsi"),
                  "dark_boat": b["identification"] == "dark"}.get(src["id"])
            print(f"[accuracy] {src['label']}: засечек {b['n_fixes']}, ошибка точки медиана {b['fix_err_m']['median']} м "
                  f"(p90 {b['fix_err_m']['p90']}), трека {b['track_err_m']['median']} м"
                  + ("" if ok is None else f"; опознание {'верно' if ok else 'НЕВЕРНО'} ({b['identification']})"))
        else:
            print(f"[accuracy] {src['label']}: не найден")
    return out
