"""Демо-сцена «сеть из 3 буёв»: физически согласованная симуляция для пеленга и фьюжна.

Ночь в период нерестового запрета. Сценарий (150 с):
  0–50 с    теплоход с AIS идёт по судовому ходу — легальное судно, тревога понижается;
  60–150 с  лодка без AIS идёт с юго-запада на нерестилище и сбрасывает ход внутри;
  140 с     выстрел с лодки.

Для каждого буя: звук источника с задержкой r(t)/c (переменная задержка даёт и Доплер),
затухание ~1/r, свой независимый фон. Плюс выгрузка AIS (теплоход + далёкое судно).
Полевые эффекты (по умолчанию включены, `ideal=True` — выключить): эхо от берегов/камыша,
ошибка GPS-координат буёв, рассинхронизация часов, скорость звука отличается от заложенной
в конфиг (температура воды/воздуха). Без них точность пеленга получается нереально идеальной.
Координаты и зоны — демонстрационные, не реальные границы ООПТ.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .audio_io import resample, save_audio
from .geo import LocalFrame
from .synth import background, boat_motor, gunshot

BUOYS = {"buoy-01": (0.0, 0.0), "buoy-02": (1200.0, 300.0), "buoy-03": (500.0, 1100.0)}
T0_LOCAL = datetime(2026, 5, 20, 1, 30, tzinfo=timezone(timedelta(hours=4)))  # 01:30, нерестовый запрет
DURATION = 150.0
R0 = 100.0  # м: на этом расстоянии сигнал источника имеет «паспортный» уровень
UP = 4
BG_LEVEL = 0.12
FIELD = {
    "echoes": 2, "echo_delay_s": (0.005, 0.06), "echo_gain": (0.2, 0.5),  # отражения от берега/камыша
    "gps_sigma_m": 2.0,        # ошибка координат буя (на каждую ось)
    "clock_sigma_s": 50e-6,    # остаточная рассинхронизация (GPS PPS + АЦП)
    "speed_error": -0.008,     # реальная скорость звука на 0.8% ниже заложенной (вода 1468 вместо 1480)
}  # уровень фона: около +10 дБ у мотора в полосе кавитации на 1 км (реально в воде больше)

SOURCES = [
    {"id": "ais_ship", "class": "boat_motor", "label": "Теплоход «Волжский-7» (AIS)", "mmsi": "273318450",
     "level": 0.9, "f0": 24.0, "path": [(0, (-760, -150)), (50, (-720, 270))]},
    {"id": "dark_boat", "class": "boat_motor", "label": "Лодка без AIS", "level": 0.45, "f0": 72.0,
     "path": [(60, (60, 80)), (128, (780, 720)), (150, (830, 780))]},
    {"id": "shot", "class": "gunshot", "label": "Выстрел с лодки", "level": 8.0, "at": 140.0},  # выстрел на 50+ дБ громче мотора в воздухе
]


def _path_xy(path, t):
    ts = np.array([p[0] for p in path], float)
    xs = np.array([p[1][0] for p in path], float)
    ys = np.array([p[1][1] for p in path], float)
    return np.interp(t, ts, xs), np.interp(t, ts, ys)


def real_gunshot(sr: int, manifest: str = "data/manifest.csv"):
    """Реальная запись выстрела из ТЕСТОВОЙ части датасета (модель её не видела). Выбор не по
    оценке модели, а по форме сигнала: самый чёткий импульс (макс. отношение пика к RMS)."""
    from .audio_io import load_audio
    from .prepare import read_manifest

    if not Path(manifest).exists():
        return None
    rows = sorted(r["path"] for r in read_manifest(manifest)
                  if r["split"] == "test" and r["label"] == "gunshot" and r["source"] != "synthetic")
    best, best_c = None, 0.0
    for path in rows:
        x = load_audio(path, sr)
        c = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9))
        if c > best_c:
            best, best_c = (path, x), c
    if best is None:
        return None
    real_gunshot.path = best[0]
    x = best[1][: int(1.5 * sr)]
    return x / (np.max(np.abs(x)) + 1e-9)


def _envelope(n, sr, fade=2.0):
    env = np.ones(n)
    f = min(n // 3, int(fade * sr))
    env[:f], env[n - f :] = np.linspace(0, 1, f), np.linspace(1, 0, f)
    return env


def simulate(out_dir: str | Path, cfg: dict, seed: int = 7, ideal: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    sr = cfg["sample_rate"]
    lcfg = cfg["localization"]
    frame = LocalFrame(*cfg["fusion"]["origin"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(DURATION * sr)
    t = np.arange(n) / sr
    bxy = np.array(list(BUOYS.values()))

    # независимый фон на каждом буе (иначе корреляция фона даст ложный нулевой пик)
    rx = [background(DURATION, sr, rng) * BG_LEVEL for _ in BUOYS]
    fe = {"echoes": 0, "gps_sigma_m": 0.0, "clock_sigma_s": 0.0, "speed_error": 0.0} if ideal else FIELD
    clock = rng.normal(0, fe["clock_sigma_s"], len(BUOYS))
    truth = []
    for src in SOURCES:
        c = lcfg["sound_speed"].get(src["class"], lcfg["sound_speed"]["default"]) * (1 + fe["speed_error"])
        if "path" in src:
            ta, tb = src["path"][0][0], src["path"][-1][0]
            sig = np.zeros(n)
            i0, i1 = int(ta * sr), int(tb * sr)
            sig[i0:i1] = boat_motor(tb - ta, sr, rng, src.get("f0"))[: i1 - i0] * _envelope(i1 - i0, sr)
            pos = lambda te, path=src["path"]: _path_xy(path, te)
            tt = np.arange(ta, tb, 1.0)
            tx, ty = _path_xy(src["path"], tt)
            truth.append({**{k: src[k] for k in ("id", "class", "label")}, "mmsi": src.get("mmsi"),
                          "t": (tt).tolist(), "x": tx.round(1).tolist(), "y": ty.round(1).tolist()})
        else:  # импульс из точки, где в этот момент лодка
            dark = next(s for s in SOURCES if s["id"] == "dark_boat")
            sx, sy = _path_xy(dark["path"], src["at"])
            sig = np.zeros(n)
            g = real_gunshot(sr) if src.get("real", True) else None
            if g is None:
                g = gunshot(1.0, sr, rng, at=0.0)
            src["clip"] = getattr(real_gunshot, "path", "синтетика")
            i0 = int(src["at"] * sr)
            sig[i0 : i0 + len(g)] = g
            pos = lambda te, sx=sx, sy=sy: (np.full_like(te, sx), np.full_like(te, sy))
            truth.append({**{k: src[k] for k in ("id", "class", "label")}, "clip": src["clip"], "t": [src["at"]], "x": [round(float(sx), 1)], "y": [round(float(sy), 1)]})
        sig *= src["level"]
        # дробная задержка точно: сигнал на сетке UP x sr (линейная интерполяция на 16 кГц глушит ВЧ)
        sig_up = resample(sig, sr, sr * UP)
        t_up = np.arange(len(sig_up)) / (sr * UP)
        for k, (bx, by) in enumerate(bxy):
            # приём в момент t: излучено в te, где te + r(te)/c = t (итерации; переменная задержка = Доплер)
            tk = t + clock[k]  # часы буя врут на clock[k]
            te = tk.copy()
            for _ in range(3):
                px, py = pos(te)
                r = np.hypot(px - bx, py - by)
                te = tk - r / c
            g = R0 / np.maximum(r, R0)
            rx[k] = rx[k] + np.interp(te, t_up, sig_up, left=0, right=0) * g
            for _ in range(fe["echoes"]):  # отражённый путь: позже и тише
                d, a = rng.uniform(*fe["echo_delay_s"]), rng.uniform(*fe["echo_gain"])
                rx[k] = rx[k] + a * np.interp(te - d, t_up, sig_up, left=0, right=0) * g

    peak = max(np.max(np.abs(x)) for x in rx)
    for (bid, _), x in zip(BUOYS.items(), rx):
        save_audio(out / f"{bid}.wav", x / peak * 0.9, sr)  # общий масштаб: уровни между буями сохраняются

    reported = bxy + rng.normal(0, fe["gps_sigma_m"], bxy.shape)  # буй знает своё место с ошибкой GPS
    lat, lon = frame.to_ll(reported[:, 0], reported[:, 1])
    meta = {
        "recorded_at": T0_LOCAL.isoformat(), "sample_rate": sr, "note": "демо: координаты и зоны условные" + ("" if ideal else "; эхо, ошибка GPS, рассинхронизация, ошибка скорости звука"),
        "buoys": [{"id": b, "lat": round(float(la), 6), "lon": round(float(lo), 6), "file": f"{b}.wav"}
                  for b, la, lo in zip(BUOYS, lat, lon)],
    }
    (out / "array.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # AIS: теплоход шлёт позицию каждые 10 с (и до, и после записи), далёкий сухогруз — тоже
    t0 = T0_LOCAL.timestamp()
    rows = []
    ship = SOURCES[0]
    (xa, ya), (xb, yb) = ship["path"][0][1], ship["path"][-1][1]
    vx, vy = (xb - xa) / 50, (yb - ya) / 50
    for s in range(-60, 120, 10):
        la, lo = frame.to_ll(xa + vx * s, ya + vy * s)
        rows.append({"mmsi": ship["mmsi"], "ts": datetime.fromtimestamp(t0 + s, timezone.utc).isoformat(),
                     "lat": f"{float(la):.6f}", "lon": f"{float(lo):.6f}", "name": "ВОЛЖСКИЙ-7", "sog": f"{np.hypot(vx, vy) * 1.944:.1f}"})
    for s in range(-60, 200, 10):
        la, lo = frame.to_ll(-2600 + 2 * s, -3000 + 6 * s)
        rows.append({"mmsi": "273449120", "ts": datetime.fromtimestamp(t0 + s, timezone.utc).isoformat(),
                     "lat": f"{float(la):.6f}", "lon": f"{float(lo):.6f}", "name": "ОМСКИЙ-131", "sog": "12.3"})
    with open(out / "ais.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mmsi", "ts", "lat", "lon", "name", "sog"])
        w.writeheader()
        w.writerows(rows)
    (out / "truth.json").write_text(json.dumps({"t0": T0_LOCAL.isoformat(), "sources": truth}, ensure_ascii=False, indent=2))
    print(f"[array-sim] {len(BUOYS)} буя x {DURATION:.0f} с -> {out} (array.json, ais.csv, truth.json)")
    return out
