"""Предиктивный модуль: прогноз нереста по гидрологии -> окна запрета и план патрулирования.

Нерест в дельте Волги управляется тремя факторами:
  * температура воды: у каждого вида своё «окно» (вобла 8–12 °C, судак 10–15, сазан 18–24 ...);
  * созревание: накопленная сумма тепла (градусо-дни) с 1 марта;
  * половодье: фитофильным видам (вобла, лещ, сазан) нужны залитые полои, т.е. расход Волги
    выше порога; график весеннего попуска Волжской ГЭС публикуется заранее — это вход прогноза.

Модель прозрачная (биологические пороги, а не чёрный ящик): интенсивность нереста
I = пригодность_T × созревание × половодье × остаток_нерестового_запаса.
Неопределённость — ансамбль из 50 сценариев температуры: аномалия текущего года затухает
к климатической норме, разброс растёт с горизонтом. На выходе по каждому дню и виду:
вероятность, что нерест идёт, и интенсивность (медиана, 10–90%).

Применение:
  * окна запрета по нерестилищам — не по фиксированным датам, а по прогнозу (фьюжн);
  * план патрулирования: где и когда нерест совпадает с активностью браконьеров.

ВНИМАНИЕ: параметры видов — ориентировочные литературные значения, их нужно сверить
с ихтиологами (КаспНИРХ / ВНИРО) и откалибровать на данных учётных съёмок.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Климатическая норма температуры воды у Астрахани (середины месяцев, °C) и расхода у Волгограда (м³/с)
CLIM_T = [0.3, 0.3, 2.5, 9.5, 17.5, 22.5, 25.0, 24.0, 18.5, 11.0, 4.5, 1.0]
CLIM_Q = [6500, 6500, 7500, 14000, 21000, 12000, 6500, 6000, 6000, 6500, 7000, 6800]

SPECIES = {
    #            T мин, оптимум,  T макс, градусо-дни, нужно половодье
    "vobla": {"name": "Вобла", "t_min": 6, "t_opt": (8, 12), "t_max": 16, "dd": 120, "flood": True},
    "sudak": {"name": "Судак", "t_min": 8, "t_opt": (10, 15), "t_max": 18, "dd": 200, "flood": False},
    "lesch": {"name": "Лещ", "t_min": 10, "t_opt": (12, 18), "t_max": 22, "dd": 300, "flood": True},
    "osetr": {"name": "Русский осётр", "t_min": 10, "t_opt": (13, 18), "t_max": 23, "dd": 650, "flood_q": 12000},
    "sazan": {"name": "Сазан", "t_min": 15, "t_opt": (18, 24), "t_max": 28, "dd": 550, "flood": True},
}
FLOOD_Q = 15000  # м³/с: при таком расходе заливаются полои дельты
STOCK = 12.0     # «запас» нереста в днях полной интенсивности: потом вид отнерестился


def _clim(d: date, table) -> float:
    """Линейная интерполяция по серединам месяцев."""
    doy = d.timetuple().tm_yday
    mids = np.array([15 + 30.4 * m for m in range(12)])
    vals = np.array(table, float)
    return float(np.interp(doy, np.concatenate([mids - 365, mids, mids + 365]), np.concatenate([vals, vals, vals])))


def simulate_hydrology(today: date, anomaly_t: float = 1.5, seed: int = 0) -> dict:
    """Демо-ряд: тёплая весна (+1.5 °C к норме) с 1 января до сегодня + план попуска на 3 недели."""
    rng = np.random.default_rng(seed)
    days = [date(today.year, 1, 1) + timedelta(d) for d in range((today - date(today.year, 1, 1)).days + 1)]
    t, q = [], []
    walk = 0.0
    for d in days:
        walk = 0.85 * walk + rng.normal(0, 0.35)
        t.append(max(0.0, _clim(d, CLIM_T) + anomaly_t * min(1, d.timetuple().tm_yday / 60) + walk))
        q.append(_clim(d, CLIM_Q) * (1 + rng.normal(0, 0.04)))
    plan = [{"date": (today + timedelta(k)).isoformat(), "discharge_m3s": round(_clim(today + timedelta(k), CLIM_Q) * (1.1 if k < 8 else 0.9))}
            for k in range(1, 22)]
    return {"history": [{"date": d.isoformat(), "water_temp_c": round(a, 2), "discharge_m3s": round(b)} for d, a, b in zip(days, t, q)],
            "discharge_plan": plan, "source": "демо: климатическая норма + аномалия +1.5 °C"}


def load_hydrology_csv(path, plan_path=None) -> dict:
    """CSV: date, water_temp_c, discharge_m3s (реальные ряды: Росгидромет, ГМС Астрахань / Волгоград)."""
    with open(path, encoding="utf-8") as f:
        hist = [{"date": r["date"], "water_temp_c": float(r["water_temp_c"]), "discharge_m3s": float(r["discharge_m3s"])} for r in csv.DictReader(f)]
    plan = []
    if plan_path:
        with open(plan_path, encoding="utf-8") as f:
            plan = [{"date": r["date"], "discharge_m3s": float(r["discharge_m3s"])} for r in csv.DictReader(f)]
    return {"history": hist, "discharge_plan": plan, "source": str(path)}


def _suit(t: np.ndarray, sp: dict) -> np.ndarray:
    lo, hi = sp["t_opt"]
    up = np.clip((t - sp["t_min"]) / max(lo - sp["t_min"], 1e-6), 0, 1)
    down = np.clip((sp["t_max"] - t) / max(sp["t_max"] - hi, 1e-6), 0, 1)
    return np.minimum(up, down)


def _intensity(dates: list[date], temp: np.ndarray, q: np.ndarray, sp: dict) -> np.ndarray:
    """Интенсивность нереста по дням (вдоль последней оси, поддерживает ансамбль (M, D))."""
    temp, q = np.atleast_2d(temp), np.atleast_2d(q)
    start = np.array([d >= date(d.year, 3, 1) for d in dates])
    dd = np.cumsum(np.where(start, np.maximum(temp, 0), 0), axis=-1)
    mature = 1 / (1 + np.exp(-(dd - sp["dd"]) / (0.12 * sp["dd"])))
    q_thr = sp.get("flood_q", FLOOD_Q if sp.get("flood") else None)
    flood = 1 / (1 + np.exp(-(q - q_thr) / 1500)) if q_thr else np.ones_like(q)
    raw = _suit(temp, sp) * mature * flood
    out = np.zeros_like(raw)
    spent = np.zeros(raw.shape[0])
    for k in range(raw.shape[1]):  # отнерестившиеся рыбы больше не нерестятся
        out[:, k] = raw[:, k] * np.exp(-spent / STOCK)
        spent += out[:, k]
    return out


def forecast(hydro: dict, today: date, horizon: int = 14, members: int = 50, seed: int = 0, species=None) -> dict:
    rng = np.random.default_rng(seed)
    hist = [h for h in hydro["history"] if date.fromisoformat(h["date"]) <= today]
    hd = [date.fromisoformat(h["date"]) for h in hist]
    ht = np.array([h["water_temp_c"] for h in hist])
    hq = np.array([h["discharge_m3s"] for h in hist])
    fd = [today + timedelta(k) for k in range(1, horizon + 1)]
    # аномалия последней недели затухает к норме (τ = 10 дней) + случайное блуждание
    anom = float(np.mean(ht[-7:] - np.array([_clim(d, CLIM_T) for d in hd[-7:]])))
    base = np.array([_clim(d, CLIM_T) + anom * np.exp(-(k + 1) / 10) for k, d in enumerate(fd)])
    walk = np.cumsum(rng.normal(0, 0.6, (members, horizon)), axis=1)
    ft = np.maximum(base + walk, 0)
    plan = {p["date"]: p["discharge_m3s"] for p in hydro.get("discharge_plan", [])}
    fq0 = np.array([plan.get(d.isoformat(), hq[-1]) for d in fd])
    fq = fq0 * (1 + rng.normal(0, 0.08, (members, 1)))
    dates = hd + fd
    temp = np.concatenate([np.repeat(ht[None], members, 0), ft], axis=1)
    q = np.concatenate([np.repeat(hq[None], members, 0), fq], axis=1)

    out = {"today": today.isoformat(), "horizon_days": horizon, "source": hydro.get("source"),
           "water_temp": {"history": [[d.isoformat(), round(float(v), 2)] for d, v in zip(hd[-30:], ht[-30:])],
                          "forecast": [[d.isoformat(), round(float(np.median(ft[:, k])), 2), round(float(np.percentile(ft[:, k], 10)), 2),
                                        round(float(np.percentile(ft[:, k], 90)), 2)] for k, d in enumerate(fd)]},
           "discharge": {"history": [[d.isoformat(), int(v)] for d, v in zip(hd[-30:], hq[-30:])],
                         "plan": [[d.isoformat(), int(v)] for d, v in zip(fd, fq0)]},
           "temp_anomaly_c": round(anom, 2), "species": {}}
    n_hist = len(hd)
    for key, sp in SPECIES.items():
        if species and key not in species:
            continue
        I = _intensity(dates, temp, q, sp)
        active = I > 0.35
        series = []
        for k in range(n_hist - 30, len(dates)):
            series.append({"date": dates[k].isoformat(), "forecast": k >= n_hist,
                           "intensity": round(float(np.median(I[:, k])), 3),
                           "p10": round(float(np.percentile(I[:, k], 10)), 3), "p90": round(float(np.percentile(I[:, k], 90)), 3),
                           "p_active": round(float(active[:, k].mean()), 2)})
        out["species"][key] = {"name": sp["name"], "series": series, "status": _status(series, today)}
    return out


def _status(series, today) -> str:
    now = next((s for s in series if s["date"] == today.isoformat()), None)
    fut = [s for s in series if s["forecast"]]
    past = [s for s in series if not s["forecast"]]
    if now and now["p_active"] >= 0.5:
        end = next((s["date"] for s in fut if s["p_active"] < 0.3), None)
        return "идёт нерест" + (f", спад ожидается ~{end[8:10]}.{end[5:7]}" if end else ", продолжится весь горизонт прогноза")
    start = next((s for s in fut if s["p_active"] >= 0.5), None)
    if start:
        return f"начало ожидается ~{start['date'][8:10]}.{start['date'][5:7]} (вероятность {start['p_active']:.0%})"
    if any(s["intensity"] > 0.35 for s in past):
        return "нерест в основном завершён"
    return "нерест в горизонте прогноза не ожидается"


def zone_windows(fc: dict, zones: list[dict], p_thr: float = 0.4) -> dict:
    """Нерестилища -> дни, когда по прогнозу идёт нерест (для фьюжна и плана патрулей)."""
    out = {}
    for z in zones:
        sp = z.get("species")
        if z.get("kind") != "spawning" or not sp:
            continue
        days = {}
        for key in sp:
            if key not in fc["species"]:
                continue
            for s in fc["species"][key]["series"]:
                if s["p_active"] >= p_thr:
                    prev = days.get(s["date"])
                    if prev is None or s["p_active"] > prev["p"]:
                        days[s["date"]] = {"p": s["p_active"], "species": fc["species"][key]["name"], "forecast": s["forecast"]}
        out[z["name"]] = days
    return out


def patrol_plan(fc: dict, zones: list[dict], alerts: list[dict] | None = None, frame=None, top: int = 6) -> list[dict]:
    """Где и когда патрулировать: вероятность нереста × давление браконьеров (алерты в зоне)."""
    from .geo import in_zone

    pressure = {}
    for z in zones:
        cnt = 0
        for a in alerts or []:
            loc = a.get("location")
            if frame is None or not loc or a.get("type", "acoustic") != "acoustic":
                continue
            x, y = frame.to_xy(loc["lat"], loc["lon"])
            cnt += in_zone(z, float(x), float(y)) and (a.get("fusion", {}).get("priority") in ("high", "critical"))
        pressure[z["name"]] = cnt
    wins = zone_windows(fc, zones)
    recs = []
    for zname, days in wins.items():
        for d, info in days.items():
            if not info["forecast"] and d != fc["today"]:
                continue
            score = info["p"] * (1 + 0.5 * min(pressure.get(zname, 0), 4))
            recs.append({"zone": zname, "date": d, "score": round(score, 2), "p_spawning": info["p"], "species": info["species"],
                         "poaching_alerts": pressure.get(zname, 0),
                         "why": f"нерест: {info['species']} (вероятность {info['p']:.0%})"
                                + (f"; браконьерских алертов в зоне: {pressure[zname]}" if pressure.get(zname) else ""),
                         "when": "ночь 21:00–05:00 (пик активности браконьеров)"})
    recs.sort(key=lambda r: (-r["score"], r["date"]))
    return recs[:top]
