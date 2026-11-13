"""IoT-датчики воды: приём показаний, детектор сбросов и экологический статус.

Станция (мультипараметрический зонд на буе) раз в 10 минут присылает:
температуру, pH, растворённый кислород, мутность, электропроводность и уровень.

Как отличить сброс от нормы и от сбоя датчика:
  * У воды есть суточный цикл: днём фотосинтез поднимает кислород и pH, ночью они падают.
    Поэтому сравниваем не с общим средним, а с тем же часом суток за прошлые дни
    (медиана и MAD), робастный z-score.
  * Сброс — согласованное отклонение НЕСКОЛЬКИХ параметров (мутность ↑, кислород ↓,
    электропроводность ↑, pH сдвигается), держится ≥ 30 минут; часто потом приходит
    на станцию ниже по течению.
  * Сбой датчика — один параметр: одиночный выброс или «залипшее» значение.
Плюс нормативы (рыбохозяйственные ПДК): кислород не ниже 6 мг/л, pH 6.5–8.5.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .geo import LocalFrame, iso

PARAMS = {  # имя: (подпись, единица, направление «плохо»: +1 рост, -1 падение, 0 любое)
    "temp_c": ("Температура", "°C", 0),
    "ph": ("pH", "", 0),
    "do_mg_l": ("Растворённый O₂", "мг/л", -1),
    "turbidity_ntu": ("Мутность", "NTU", +1),
    "cond_us_cm": ("Электропроводность", "мкСм/см", +1),
    "level_cm": ("Уровень", "см", 0),
}
NORMS = {"do_mg_l": (6.0, None), "ph": (6.5, 8.5)}  # рыбохозяйственные нормативы

# Демо-станции (метры от опорной точки участка); течение — с севера на юг
STATIONS = {
    "W1": {"name": "Протока выше по течению (фон)", "xy": (250, 2300)},
    "W2": {"name": "Ерик у рыбоцеха", "xy": (950, 1750)},
    "W3": {"name": "Нерестилище, центр", "xy": (1100, 1000)},
    "W4": {"name": "Судовой ход", "xy": (-750, 600)},
}


# ---------------------------------------------------------------- симулятор

def simulate(cfg: dict, out: str | Path, end_local: datetime | None = None, days: int = 7, seed: int = 3) -> Path:
    """7 дней показаний 4 станций, шаг 10 мин. Сценарий: ночью сброс у W2, через ~2 ч шлейф
    доходит до нерестилища (W3); на W4 — сбой датчика мутности (залипание + выброс)."""
    rng = np.random.default_rng(seed)
    tz = timezone(timedelta(hours=cfg["fusion"].get("utc_offset_hours", 4)))
    end = end_local or datetime(2026, 5, 20, 6, 0, tzinfo=tz)
    n = days * 144
    ts = np.array([end - timedelta(minutes=10 * (n - 1 - i)) for i in range(n)])
    hour = np.array([t.hour + t.minute / 60 for t in ts])
    day = np.arange(n) / 144
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)  # максимум ~15 ч

    frame = LocalFrame(*cfg["fusion"]["origin"])
    spill_start = end - timedelta(hours=3, minutes=50)  # 02:10 местного
    rows = []
    for k, (sid, st) in enumerate(STATIONS.items()):
        base_t = 17.0 + 0.25 * day + rng.normal(0, 0.1)  # весенний прогрев
        temp = base_t + 1.2 * diurnal + rng.normal(0, 0.08, n)
        do = 9.0 - 0.18 * (temp - 17) + 1.1 * diurnal + rng.normal(0, 0.12, n)
        ph = 7.9 + 0.18 * diurnal + rng.normal(0, 0.03, n)
        turb = 18 + (25 if sid == "W4" else 0) + 3 * np.abs(rng.standard_normal(n)) + 2 * np.sin(2 * np.pi * day / 3)
        cond = 420 + 8 * np.sin(2 * np.pi * day / 5) + rng.normal(0, 4, n)
        level = 180 + 6 * day + 2 * np.sin(2 * np.pi * hour / 12.4) + rng.normal(0, 0.5, n)  # половодье + сгонно-нагонные

        lag_h = {"W2": 0.0, "W3": 2.0}.get(sid)
        if lag_h is not None:  # шлейф сброса: резкий фронт, экспоненциальный спад, у W3 слабее и размытее
            dt_h = np.array([(t - spill_start).total_seconds() / 3600 for t in ts]) - lag_h
            amp = 1.0 if sid == "W2" else 0.55
            width = 0.4 if sid == "W2" else 0.9
            pos = np.maximum(dt_h, 0)
            plume = amp * np.where(dt_h > 0, (1 - np.exp(-pos / 0.2)) * np.exp(-np.maximum(pos - width, 0) / 1.5), 0)
            turb += 140 * plume
            do -= 4.2 * plume
            cond += 380 * plume
            ph -= 0.9 * plume
        if sid == "W4":  # сбой датчика мутности: выброс и 3 часа залипания
            i = n - 40
            turb[i] = 900
            turb[n - 30 : n - 12] = turb[n - 31]
        for i in range(n):
            rows.append({"ts": ts[i].isoformat(), "station": sid, "temp_c": round(temp[i], 2), "ph": round(ph[i], 3),
                         "do_mg_l": round(do[i], 2), "turbidity_ntu": round(turb[i], 1), "cond_us_cm": round(cond[i], 1),
                         "level_cm": round(level[i], 1)})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "water.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    st = []
    for sid, s in STATIONS.items():
        lat, lon = frame.to_ll(*s["xy"])
        st.append({"id": sid, "name": s["name"], "lat": round(float(lat), 6), "lon": round(float(lon), 6)})
    (out / "stations.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"[water-sim] {len(STATIONS)} станции x {days} дн -> {out / 'water.csv'}")
    return out


def load_csv(path) -> dict[str, dict[str, np.ndarray]]:
    """water.csv -> {станция: {"ts": epoch, параметр: массив}}."""
    by: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["station"], []).append(r)
    out = {}
    for sid, rs in by.items():
        rs.sort(key=lambda r: r["ts"])
        d = {"ts": np.array([datetime.fromisoformat(r["ts"]).timestamp() for r in rs]),
             "local": [r["ts"] for r in rs]}
        for p in PARAMS:
            d[p] = np.array([float(r[p]) if r.get(p) not in (None, "") else np.nan for r in rs])
        out[sid] = d
    return out


# ---------------------------------------------------------------- детектор

def _hour_of(local_iso: str) -> int:
    return datetime.fromisoformat(local_iso).hour


def robust_z(series: dict, param: str, history_days: int = 5) -> np.ndarray:
    """z-score относительно того же часа суток за предыдущие дни (медиана/MAD)."""
    x = series[param]
    hours = np.array([_hour_of(s) for s in series["local"]])
    ts = series["ts"]
    z = np.zeros(len(x))
    for i in range(len(x)):
        m = (hours == hours[i]) & (ts < ts[i] - 3 * 3600) & (ts > ts[i] - history_days * 86400 - 3600)
        ref = x[m]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 6:
            continue
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-6
        # пол разброса: у стабильных параметров MAD бывает почти нулевой
        floor = {"temp_c": 0.3, "ph": 0.05, "do_mg_l": 0.3, "turbidity_ntu": 3, "cond_us_cm": 8, "level_cm": 2}[param]
        z[i] = (x[i] - med) / max(mad, floor)
    return z


def detect(series_by_station: dict, stations: list[dict], z_thr: float = 4.0, min_len: int = 3, cfg: dict | None = None) -> dict:
    """Аномалии по всем станциям -> алерты (формат как у акустики) + статус станций."""
    st_by = {s["id"]: s for s in stations}
    alerts, status = [], {}
    episodes = []
    for sid, s in series_by_station.items():
        zs = {p: robust_z(s, p) for p in PARAMS if p != "level_cm"}
        n = len(s["ts"])
        bad = {p: (np.sign(z) * PARAMS[p][2] > 0) & (np.abs(z) > z_thr) if PARAMS[p][2] else np.abs(z) > z_thr for p, z in zs.items()}
        multi = sum(bad[p].astype(int) for p in bad) >= 2
        # эпизоды сброса: ≥2 параметров, ≥ min_len отсчётов подряд (30 мин)
        i = 0
        while i < n:
            if not multi[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and multi[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                seg = slice(i, j + 1)
                params = [p for p in bad if bad[p][seg].mean() > 0.5]
                peak = {p: float(np.nanmax(np.abs(zs[p][seg]))) for p in params}
                episodes.append({"station": sid, "t0": float(s["ts"][i]), "t1": float(s["ts"][j]), "params": params, "peak_z": peak,
                                 "values": {p: [float(np.nanmedian(s[p][max(0, i - 12):i])), float(s[p][i + int(np.argmax(np.abs(zs[p][seg])))])] for p in params}})
            i = j + 1
        # сбои датчика: одиночный огромный выброс или залипание
        for p in bad:
            x = s[p]
            for k in range(1, n - 1):
                if abs(zs[p][k]) > 15 and abs(zs[p][k - 1]) < z_thr and abs(zs[p][k + 1]) < z_thr:
                    alerts.append(_alert(st_by.get(sid), "sensor_fault", float(s["ts"][k]), float(s["ts"][k]), 0.9,
                                         f"Сбой датчика «{PARAMS[p][0]}»: одиночный выброс {x[k]:g} {PARAMS[p][1]}", "info", 5))
            # залипание: ≥ 12 одинаковых значений подряд (2 ч) за последние сутки
            same = np.concatenate([[False], np.abs(np.diff(x)) < 1e-9])
            k = max(1, n - 144)
            while k < n:
                if not same[k]:
                    k += 1
                    continue
                j = k
                while j + 1 < n and same[j + 1]:
                    j += 1
                if j - k + 2 >= 12:
                    alerts.append(_alert(st_by.get(sid), "sensor_fault", float(s["ts"][k - 1]), float(s["ts"][j]), 0.9,
                                         f"Датчик «{PARAMS[p][0]}» залип на {x[k]:g} {PARAMS[p][1]} ({(j - k + 2) * 10} мин): данные не учитываются", "info", 5))
                k = j + 1
        # статус станции по последним значениям
        last = {p: float(s[p][-1]) for p in PARAMS}
        viol = []
        for p, (lo, hi) in NORMS.items():
            if lo is not None and last[p] < lo:
                viol.append(f"{PARAMS[p][0]} {last[p]:g} {PARAMS[p][1]} < норматива {lo:g}")
            if hi is not None and last[p] > hi:
                viol.append(f"{PARAMS[p][0]} {last[p]:g} > норматива {hi:g}")
        recent = [e for e in episodes if e["station"] == sid and e["t1"] >= s["ts"][-1] - 6 * 3600]
        level = "red" if viol or (recent and recent[-1]["t1"] >= s["ts"][-1] - 1800) else "yellow" if recent else "green"
        status[sid] = {"level": level, "last": last, "last_ts": iso(float(s["ts"][-1])), "violations": viol,
                       "z": {p: round(float(zs[p][-1]), 1) for p in zs}}

    # сброс + распространение вниз по течению: эпизоды на разных станциях с задержкой 0.5–6 ч
    episodes.sort(key=lambda e: e["t0"])
    used = set()
    for a, e in enumerate(episodes):
        if a in used:
            continue
        chain = [e]
        for b in range(a + 1, len(episodes)):
            f = episodes[b]
            if f["station"] not in {c["station"] for c in chain} and 1800 <= f["t0"] - chain[-1]["t0"] <= 6 * 3600:
                chain.append(f)
                used.add(b)
        st = st_by.get(e["station"], {"id": e["station"]})
        reasons = []
        for c in chain:
            ch = ", ".join(f"{PARAMS[p][0]} {c['values'][p][0]:.3g}→{c['values'][p][1]:.3g}" for p in c["params"])
            reasons.append(f"{c['station']} ({st_by.get(c['station'], {}).get('name', '')}): {ch}")
        if len(chain) > 1:
            dt = (chain[1]["t0"] - chain[0]["t0"]) / 3600
            reasons.append(f"Шлейф дошёл до {chain[1]['station']} через {dt:.1f} ч — распространяется вниз по течению")
        in_spawn = any("Нерест" in st_by.get(c["station"], {}).get("name", "") for c in chain)
        risk = min(100, 45 + 10 * len(chain[0]["params"]) + (20 if in_spawn else 0) + (10 if len(chain) > 1 else 0))
        al = _alert(st, "water_discharge", e["t0"], max(c["t1"] for c in chain), 0.9,
                    "Вероятный сброс загрязнения", "critical" if risk >= 75 else "high", risk)
        al["fusion"]["reasons"] = [{"factor": "water", "points": 0, "text": r} for r in reasons]
        if in_spawn:
            al["fusion"]["reasons"].append({"factor": "zone", "points": 20, "text": "Шлейф на нерестилище"})
        al["stations"] = [c["station"] for c in chain]
        alerts.append(al)
    return {"alerts": alerts, "status": status, "episodes": episodes}


def _alert(st: dict | None, category: str, t0: float, t1: float, conf: float, title: str, priority: str, risk: int) -> dict:
    import uuid
    return {
        "id": str(uuid.uuid4()), "type": "water", "category": category, "title": title,
        "severity": priority, "confidence": conf, "sensor_id": (st or {}).get("id"),
        "location": {"lat": st["lat"], "lon": st["lon"], "source": "sensor"} if st and "lat" in st else None,
        "start_ts": iso(t0), "end_ts": iso(t1), "status": "new",
        "fusion": {"risk": risk, "priority": priority, "reasons": [{"factor": category, "points": 0, "text": title}]},
    }
