"""Фьюжн акустики с AIS и картой зон: «тёмная лодка» и объяснимая оценка риска.

Вопрос инспектора не «шумит ли мотор», а «кто это и надо ли ехать». Поэтому:

  1) сверка с AIS: если в точке, где слышен мотор (по пеленгу или в радиусе слышимости буя),
     есть судно с AIS — это опознанное судно, тревога понижается; судно из списка своих
     (патрульные катера инспекции) — алерт только информационный;
     мотор есть, а AIS нет — «тёмная лодка», тревога повышается;
  2) зоны: заповедное ядро, нерестилища (с сезоном запрета), судовой ход;
  3) время: ночь; курс и ETA к запретной зоне по треку пеленга.

Итог — балл риска 0..100 и список причин с весами: инспектор видит, ПОЧЕМУ алерт важен.

Важно: маломерные лодки не обязаны иметь AIS, поэтому «нет AIS» — не доказательство
нарушения, а признак «судно не опознано». В заповедной зоне или на нерестилище в период
запрета неопознанный мотор — повод для приоритетного выезда.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .geo import LocalFrame, in_zone, iso, load_zones, parse_ts, ray_distance

KNOT = 1.943844  # м/с -> узлы


# ---------------------------------------------------------------- AIS

def load_ais_csv(path: str | Path) -> list[dict]:
    """CSV с колонками mmsi, ts (ISO), lat, lon [, name, sog]. Формат типичных выгрузок AIS."""
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def ais_tracks(rows: list[dict], frame: LocalFrame) -> dict[str, dict]:
    by: dict[str, list] = {}
    names: dict[str, str] = {}
    for r in rows:
        m = str(r["mmsi"])
        by.setdefault(m, []).append((parse_ts(r["ts"]), float(r["lat"]), float(r["lon"])))
        if r.get("name"):
            names[m] = r["name"]
    out = {}
    for m, pts in by.items():
        pts.sort()
        t, lat, lon = map(np.array, zip(*pts))
        x, y = frame.to_xy(lat, lon)
        out[m] = {"mmsi": m, "name": names.get(m, ""), "t": t, "x": x, "y": y}
    return out


def _ais_pos(tr: dict, t: float, max_gap: float):
    """Позиция судна в момент t (линейная интерполяция), если ближайшее сообщение не старше max_gap."""
    i = np.searchsorted(tr["t"], t)
    gap = min(abs(t - tr["t"][max(i - 1, 0)]), abs(tr["t"][min(i, len(tr["t"]) - 1)] - t))
    if gap > max_gap:
        return None
    return float(np.interp(t, tr["t"], tr["x"])), float(np.interp(t, tr["t"], tr["y"]))


def match_ais(fixes: list[dict], tracks: dict[str, dict], fcfg: dict) -> list[dict]:
    """Суда AIS, совпадающие с акустическими точками. fixes: [{t, x, y, radius_m}]."""
    gate0, max_gap = fcfg.get("ais_match_radius_m", 300), fcfg.get("ais_max_gap_s", 180)
    res = []
    for m, tr in tracks.items():
        d, ok = [], 0
        for f in fixes:
            p = _ais_pos(tr, f["t"], max_gap)
            if p is None:
                continue
            dist = float(np.hypot(p[0] - f["x"], p[1] - f["y"]))
            d.append(dist)
            ok += dist <= gate0 + f.get("radius_m", 0)
        if d:
            res.append({"mmsi": m, "name": tr["name"], "distance_m": round(float(np.median(d)), 0),
                        "agree": round(ok / len(d), 2), "n": len(d)})
    return sorted(res, key=lambda r: (-r["agree"], r["distance_m"]))


# ---------------------------------------------------------------- зоны и время

def _season_active(seasons, dt_local: datetime) -> bool:
    if not seasons:
        return True  # круглый год
    md = dt_local.strftime("%m-%d")
    for s in seasons:
        a, b = s["from"], s["to"]
        if (a <= md <= b) if a <= b else (md >= a or md <= b):  # сезон может переходить через Новый год
            return True
    return False


def _forecast_today(zone: dict, dt_local: datetime) -> dict | None:
    """Прогноз нереста (forecast.zone_windows) на этот день, если он подключён к зоне."""
    return zone.get("forecast_days", {}).get(dt_local.date().isoformat())


def _forbidden(zone: dict, category: str, dt_local: datetime) -> bool:
    forbid = zone.get("forbid", [])
    hit = forbid == "all" or category in forbid
    if not hit:
        return False
    if "forecast_days" in zone and not zone.get("seasons"):
        return _forecast_today(zone, dt_local) is not None  # окно запрета — по прогнозу нереста
    return _season_active(zone.get("seasons"), dt_local) or _forecast_today(zone, dt_local) is not None


def _local(t: float, fcfg: dict) -> datetime:
    return datetime.fromtimestamp(t, timezone(timedelta(hours=fcfg.get("utc_offset_hours", 4))))


def _is_night(dt_local: datetime, fcfg: dict) -> bool:
    a, b = fcfg.get("night_hours", [21, 5])
    h = dt_local.hour
    return (h >= a or h < b) if a > b else (a <= h < b)


# ---------------------------------------------------------------- оценка

def _kinematics(fixes: list[dict]) -> dict | None:
    """Скорость и курс по последним точкам трека (если есть vx, vy — берём их)."""
    if not fixes:
        return None
    last = fixes[-1]
    if "vx" in last:
        vx, vy = last["vx"], last["vy"]
    elif len(fixes) >= 2 and fixes[-1]["t"] - fixes[0]["t"] > 1:
        f0 = fixes[0]
        dt = last["t"] - f0["t"]
        vx, vy = (last["x"] - f0["x"]) / dt, (last["y"] - f0["y"]) / dt
    else:
        return None
    v = float(np.hypot(vx, vy))
    return {"speed_ms": v, "speed_kn": round(v * KNOT, 1),
            "course_deg": round(float(np.degrees(np.arctan2(vx, vy)) % 360), 0), "vx": vx, "vy": vy}


def assess(alert: dict, *, frame: LocalFrame, fcfg: dict, zones: list[dict] | None = None,
           tracks: dict | None = None, fixes: list[dict] | None = None) -> dict:
    """Добавить к алерту блок `fusion`: опознание, зоны, причины и балл риска.

    fixes — точки пеленга [{t, x, y, radius_m}] (из localize). Без них берём позицию буя
    и радиус слышимости: опознание грубее, но работает и для одиночного буя.
    """
    cat, conf = alert["category"], float(alert.get("confidence", 1.0))
    t0, t1 = parse_ts(alert["start_ts"]), parse_ts(alert["end_ts"])
    localized = bool(fixes)
    if not fixes:
        loc = alert.get("location")
        if not loc:
            return {**alert, "fusion": {"risk": None, "note": "нет координат буя"}}
        x, y = frame.to_xy(loc["lat"], loc["lon"])
        r = fcfg.get("hearing_range_m", 1500)
        fixes = [{"t": t, "x": float(x), "y": float(y), "radius_m": r} for t in np.linspace(t0, max(t1, t0 + 1), 5)]

    risk = fcfg["risk"]
    reasons: list[dict] = []

    def add(factor, points, text):
        reasons.append({"factor": factor, "points": int(points), "text": text})

    add("category", risk["base"].get(cat, 10), f"Событие: {alert.get('title', cat)}")

    # --- опознание по AIS
    ident = {"status": "n/a"}
    if tracks is not None and cat in fcfg.get("vessel_classes", ["boat_motor"]):
        matches = match_ais(fixes, tracks, fcfg)
        own = set(map(str, fcfg.get("own_fleet_mmsi", [])))
        best = next((m for m in matches if m["agree"] >= 0.5), None)
        if best and best["mmsi"] in own:
            ident = {"status": "own", **best}
            add("own_fleet", risk["own_fleet"], f"Свой патруль: {best['name'] or best['mmsi']} ({best['distance_m']:.0f} м)")
        elif best and not localized:
            # один буй слышит всё в радиусе слышимости: судно с AIS рядом есть, но шумит ли именно оно —
            # без пеленга не сказать (рядом может идти и лодка без AIS)
            ident = {"status": "ais_nearby", **best}
            add("ais_nearby", risk.get("ais_nearby", 0),
                f"Рядом судно с AIS: {best['name'] or best['mmsi']} ({best['distance_m']:.0f} м от буя) — без пеленга источник не установить")
        elif best:
            ident = {"status": "ais", **best}
            add("ais_identified", risk["ais_identified"],
                f"Опознано по AIS: {best['name'] or best['mmsi']}, MMSI {best['mmsi']} ({best['distance_m']:.0f} м)")
        else:
            ident = {"status": "dark", "nearest": matches[:2]}
            where = "в точке пеленга" if localized else f"в радиусе {fixes[0]['radius_m']:.0f} м от буя"
            add("dark_vessel", risk["dark_vessel"], f"«Тёмная лодка»: мотор слышен, судна с AIS {where} нет")
    elif tracks is not None and cat in fcfg.get("suspect_classes", ["gunshot"]):
        # выстрел: сам по себе AIS не опознаётся, но ближайшие суда — подсказка инспектору
        ident = {"status": "n/a", "nearby_ais": [m for m in match_ais(fixes, tracks, {**fcfg, "ais_match_radius_m": 1000})
                                                  if m["agree"] > 0][:3]}

    # --- зоны (по всем точкам трека: зашёл ли хоть раз)
    dt_mid = _local((t0 + t1) / 2, fcfg)
    in_zones, entered = [], set()
    for z in zones or []:
        inside = [in_zone(z, f["x"], f["y"]) for f in fixes]
        if any(inside):
            in_zones.append(z["name"])
            entered.add(z["name"])
            if _forbidden(z, cat, dt_mid):
                fc = _forecast_today(z, dt_mid)
                season = (f" (прогноз: идёт нерест — {fc['species'].lower()}, {fc['p']:.0%})" if fc
                          else " (период запрета)" if z.get("seasons") else "")
                add(f"zone:{z.get('kind', '')}", z.get("risk_points", risk["zone_default"]), f"В зоне «{z['name']}»{season}")
            elif z.get("kind") == "fairway" and cat in fcfg.get("vessel_classes", ["boat_motor"]):
                add("fairway", risk["fairway"], f"На судовом ходу «{z['name']}»: движение судов ожидаемо")

    # --- курс на запретную зону
    kin = _kinematics(fixes) if localized else None
    approach = None
    if kin and kin["speed_ms"] > 0.5:
        p = np.array([fixes[-1]["x"], fixes[-1]["y"]])
        d = np.array([kin["vx"], kin["vy"]]) / kin["speed_ms"]
        for z in zones or []:
            if z["name"] in entered or not _forbidden(z, cat, dt_mid):
                continue
            dist = min((s for s in (ray_distance(p, d, r) for r in z["rings"]) if s is not None), default=None)
            if dist is not None:
                eta = dist / kin["speed_ms"] / 60
                if eta <= fcfg.get("eta_horizon_min", 15) and (approach is None or eta < approach["eta_min"]):
                    approach = {"zone": z["name"], "distance_m": round(dist), "eta_min": round(eta, 1)}
        if approach:
            add("approaching", risk["approaching"], f"Идёт к зоне «{approach['zone']}»: {approach['distance_m']} м, ~{approach['eta_min']} мин")

    if _is_night(dt_mid, fcfg):
        add("night", risk["night"], f"Ночь ({dt_mid:%H:%M} местного времени)")

    raw = sum(r["points"] for r in reasons)
    score = int(np.clip(round(raw * (0.6 + 0.4 * conf)), 0, 100))  # неуверенная детекция весит меньше
    if ident["status"] == "own":
        priority = "info"
    else:
        priority = next((name for name, lo in fcfg["priority"].items() if score >= lo), "low")

    fusion = {
        "risk": score, "priority": priority, "reasons": reasons,
        "identification": ident, "zones": in_zones, "night": _is_night(dt_mid, fcfg),
        "local_time": dt_mid.isoformat(timespec="seconds"),
    }
    if kin:
        fusion["motion"] = {k: kin[k] for k in ("speed_kn", "course_deg")}
    if approach:
        fusion["approach"] = approach
    return {**alert, "fusion": fusion}


class Fusion:
    """Загруженные зоны и AIS для одного участка (опорная точка — из конфига)."""

    def __init__(self, cfg: dict, ais_rows: list[dict] | None = None, zones_path: str | None = None):
        self.fcfg = cfg["fusion"]
        self.frame = LocalFrame(*self.fcfg["origin"])
        zp = zones_path or self.fcfg.get("zones")
        self.zones = load_zones(zp, self.frame) if zp and Path(zp).exists() else []
        self.tracks = ais_tracks(ais_rows, self.frame) if ais_rows is not None else None

    def apply_forecast(self, windows: dict) -> None:
        """Окна нереста из прогноза (forecast.zone_windows) -> динамические сроки запрета зон."""
        for z in self.zones:
            if z["name"] in windows:
                z["forecast_days"] = windows[z["name"]]

    def assess(self, alert: dict, fixes: list[dict] | None = None) -> dict:
        return assess(alert, frame=self.frame, fcfg=self.fcfg, zones=self.zones, tracks=self.tracks, fixes=fixes)


def summary_line(a: dict) -> str:
    f = a["fusion"]
    ident = f["identification"]
    who = {"dark": "НЕ ОПОЗНАНА (нет AIS)", "ais": f"AIS {ident.get('name') or ident.get('mmsi')}",
           "ais_nearby": f"рядом AIS {ident.get('name') or ident.get('mmsi')}, нужен пеленг",
           "own": f"свой патруль {ident.get('name', '')}"}.get(ident["status"], "")
    return f"[{f['priority']:8s} {f['risk']:3d}] {a['title']}" + (f" — {who}" if who else "")


__all__ = ["Fusion", "assess", "load_ais_csv", "ais_tracks", "match_ais", "summary_line", "iso"]
