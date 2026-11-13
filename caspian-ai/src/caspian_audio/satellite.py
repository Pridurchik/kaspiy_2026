"""Спутник: шлейфы мутности (сбросы) по снимкам Sentinel-2.

Индексы по каналам Sentinel-2 L2A (отражение 0..1, пиксель 10 м):
  NDWI = (B3 − B8) / (B3 + B8)  > 0  — вода (маска);
  NDTI = (B4 − B3) / (B4 + B3)        — мутность: взвесь сильнее отражает красный.
Сброс = рост NDTI относительно предыдущего безоблачного снимка (ΔNDTI) на связном
участке воды заметной площади. Облака (яркие во всех каналах) исключаются.
Найденный шлейф сверяется с датчиками воды: совпадение по месту и времени повышает
уверенность — разные источники подтверждают друг друга.

Вход — .npz с массивами B3, B4, B8 и привязкой (lat0, lon0 — верхний левый угол,
pixel_m). Реальные снимки: Copernicus Data Space Browser -> L2A -> экспорт B03/B04/B08
(GeoTIFF) -> перевести в npz (rasterio). В демо — синтетическая сцена дельты.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geo import LocalFrame


def ndwi(b3, b8):
    return (b3 - b8) / (b3 + b8 + 1e-6)


def ndti(b3, b4):
    return (b4 - b3) / (b4 + b3 + 1e-6)


# ---------------------------------------------------------------- синтетическая сцена

def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simulate(cfg: dict, out: str | Path, size_m: int = 4400, pixel_m: int = 10, seed: int = 5) -> Path:
    """Два снимка: 10.05 (фон) и 20.05 11:40 (после ночного сброса у W2). Облако на юго-западе."""
    from .geo import load_zones, point_in_polygon

    rng = np.random.default_rng(seed)
    frame = LocalFrame(*cfg["fusion"]["origin"])
    n = size_m // pixel_m
    x0, y1 = -1500.0, 2900.0  # левый верхний угол в метрах
    xs = x0 + (np.arange(n) + 0.5) * pixel_m
    ys = y1 - (np.arange(n) + 0.5) * pixel_m
    X, Y = np.meshgrid(xs, ys)
    water = (X > -950) & (X < -550)  # судовой ход
    chan = [(250, 3000), (250, 2300), (950, 1750), (1100, 1000), (1300, 300), (1500, -1500)]
    for a, b in zip(chan, chan[1:]):
        water |= _seg_dist(X, Y, a, b) < 70
    for z in load_zones(cfg["fusion"]["zones"], frame):
        if z.get("kind") == "spawning":  # полои и нерестилища залиты в половодье
            for r in z["rings"]:
                water |= np.vectorize(lambda x, y: point_in_polygon(x, y, r))(X[::4, ::4], Y[::4, ::4]).repeat(4, 0).repeat(4, 1)[:n, :n]

    def scene(turb_extra, cloud):
        land_b3, land_b4, land_b8 = 0.07, 0.09, 0.28  # растительность: высокий NIR
        b3 = np.where(water, 0.055, land_b3) + rng.normal(0, 0.004, X.shape)
        b4 = np.where(water, 0.040, land_b4) + rng.normal(0, 0.004, X.shape)
        b8 = np.where(water, 0.020, land_b8) + rng.normal(0, 0.01, X.shape)
        b4 = b4 + np.where(water, turb_extra, 0)
        b3 = b3 + np.where(water, 0.3 * turb_extra, 0)
        if cloud:
            c = np.exp(-(((X + 300) / 450) ** 2 + ((Y + 600) / 300) ** 2))
            b3, b4, b8 = b3 + 0.5 * c, b4 + 0.5 * c, b8 + 0.5 * c
        return {"B3": b3.astype(np.float32), "B4": b4.astype(np.float32), "B8": b8.astype(np.float32)}

    # шлейф: от выпуска у W2 вниз по течению, вытянут вдоль протоки
    d_src = np.hypot(X - 950, Y - 1750)
    along = _seg_dist(X, Y, (950, 1750), (1100, 1000))
    plume = 0.035 * np.exp(-(along / 120) ** 2) * np.exp(-d_src / 900) * (Y < 1850)
    plume += 0.02 * np.exp(-(((X - 1050) / 260) ** 2 + ((Y - 1150) / 300) ** 2))
    lat0, lon0 = frame.to_ll(x0, y1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"lat0": float(lat0), "lon0": float(lon0), "pixel_m": pixel_m, "x0": x0, "y1": y1}
    np.savez_compressed(out / "s2_2026-05-10.npz", **scene(0.0, False), **meta, date="2026-05-10T07:40:00Z")
    np.savez_compressed(out / "s2_2026-05-20.npz", **scene(plume, True), **meta, date="2026-05-20T07:40:00Z")
    print(f"[sat-sim] 2 снимка {n}x{n} пикс по {pixel_m} м -> {out}")
    return out


# ---------------------------------------------------------------- детектор

def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Связные компоненты (4-связность) без scipy."""
    lab = np.zeros(mask.shape, np.int32)
    k = 0
    H, W = mask.shape
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        k += 1
        stack = [(i, j)]
        lab[i, j] = k
        while stack:
            a, b = stack.pop()
            for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= c < H and 0 <= d < W and mask[c, d] and not lab[c, d]:
                    lab[c, d] = k
                    stack.append((c, d))
    return lab, k


def _hull(pts: np.ndarray) -> np.ndarray:
    """Выпуклая оболочка (монотонная цепь Эндрю)."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(P):
        h = []
        for p in P:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    lo, up = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + up[:-1])


def detect(before_path, after_path, cfg: dict, d_thr: float = 0.06, min_area_ha: float = 1.0) -> dict:
    a, b = np.load(before_path), np.load(after_path)
    frame = LocalFrame(*cfg["fusion"]["origin"])
    px = float(b["pixel_m"])
    # облака яркие во всех каналах; края облака и дымка тоже меняют индексы -> порог по росту
    # зелёного канала + буфер 5 пикселей (иначе кромка облака даёт ложные «шлейфы»)
    cloud = ((b["B3"] > 0.2) & (b["B8"] > 0.2)) | ((b["B3"] - a["B3"] > 0.025) & (b["B8"] - a["B8"] > 0.02))
    for _ in range(5):
        cloud = cloud | np.roll(cloud, 1, 0) | np.roll(cloud, -1, 0) | np.roll(cloud, 1, 1) | np.roll(cloud, -1, 1)
    water = (ndwi(a["B3"], a["B8"]) > 0.1) & (ndwi(b["B3"], b["B8"]) > 0.1) & ~cloud
    d = np.where(water, ndti(b["B3"], b["B4"]) - ndti(a["B3"], a["B4"]), 0)
    # NDTI на тёмной воде шумный (σ ~0.05 на пиксель): среднее по окну 5x5 только по воде
    k = 2
    num, den = np.zeros_like(d), np.zeros_like(d)
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            num += np.roll(np.roll(d, di, 0), dj, 1)
            den += np.roll(np.roll(water.astype(float), di, 0), dj, 1)
    d = np.where(water, num / np.maximum(den, 1), 0)
    lab, k = _label(d > d_thr)
    H, W = d.shape
    plumes = []
    for c in range(1, k + 1):
        ii, jj = np.nonzero(lab == c)
        area = len(ii) * px * px / 1e4
        if area < min_area_ha:
            continue
        x = float(b["x0"]) + (jj + 0.5) * px
        y = float(b["y1"]) - (ii + 0.5) * px
        hull = _hull(np.stack([x, y], 1))
        lat, lon = frame.to_ll(hull[:, 0], hull[:, 1])
        clat, clon = frame.to_ll(x.mean(), y.mean())
        plumes.append({"area_ha": round(area, 1), "mean_dndti": round(float(d[ii, jj].mean()), 3),
                       "max_dndti": round(float(d[ii, jj].max()), 3),
                       "centroid": {"lat": round(float(clat), 6), "lon": round(float(clon), 6)},
                       "polygon": [[round(float(lo), 6), round(float(la), 6)] for la, lo in zip(lat, lon)],
                       "xy": [float(x.mean()), float(y.mean())], "length_m": round(float(np.ptp(y) if np.ptp(y) > np.ptp(x) else np.ptp(x))), })
    plumes.sort(key=lambda p: -p["area_ha"])
    lat0, lon0 = float(b["lat0"]), float(b["lon0"])
    lat1, lon1 = frame.to_ll(float(b["x0"]) + W * px, float(b["y1"]) - H * px)
    return {"date": str(b["date"]), "baseline": str(a["date"]), "plumes": plumes,
            "cloud_fraction": round(float(cloud.mean()), 3), "water_fraction": round(float(water.mean()), 3),
            "bounds": [[float(lat1), lon0], [lat0, float(lon1)]], "dndti": d, "water": water, "cloud": cloud}


def to_alerts(res: dict, water_alerts: list[dict] | None = None, stations: list[dict] | None = None, cfg: dict | None = None) -> list[dict]:
    """Шлейф -> алерт; если рядом датчик воды фиксировал сброс в последние 24 ч — подтверждение."""
    import uuid

    from .geo import parse_ts

    out = []
    frame = LocalFrame(*cfg["fusion"]["origin"]) if cfg else None
    t_img = parse_ts(res["date"])
    for p in res["plumes"]:
        reasons = [{"factor": "satellite", "points": 0,
                    "text": f"Sentinel-2 {res['date'][:10]}: мутность выросла (ΔNDTI +{p['mean_dndti']:.2f}) на {p['area_ha']} га, шлейф ~{p['length_m']} м"}]
        risk = 40 + min(20, int(p["area_ha"]))
        confirmed = []
        for wa in water_alerts or []:
            if wa.get("category") != "water_discharge" or not frame:
                continue
            if not 0 <= t_img - parse_ts(wa["start_ts"]) <= 24 * 3600:
                continue
            for sid in wa.get("stations", [wa.get("sensor_id")]):
                st = next((s for s in stations or [] if s["id"] == sid), None)
                if st:
                    sx, sy = frame.to_xy(st["lat"], st["lon"])
                    if np.hypot(sx - p["xy"][0], sy - p["xy"][1]) < 1500:
                        confirmed.append(sid)
        if confirmed:
            risk += 25
            reasons.append({"factor": "confirmed", "points": 25,
                            "text": f"Подтверждено датчиками воды {', '.join(sorted(set(confirmed)))}: сброс зафиксирован за {(t_img - min(parse_ts(w['start_ts']) for w in water_alerts if w.get('category') == 'water_discharge')) / 3600:.0f} ч до снимка"})
        prio = "critical" if risk >= 75 else "high" if risk >= 55 else "medium"
        out.append({"id": str(uuid.uuid4()), "type": "satellite", "category": "turbidity_plume", "title": "Шлейф мутности (спутник)",
                    "severity": prio, "confidence": 0.8, "sensor_id": "sentinel-2",
                    "location": {**p["centroid"], "source": "satellite"}, "start_ts": res["date"], "end_ts": res["date"],
                    "status": "new", "polygon": p["polygon"], "area_ha": p["area_ha"],
                    "fusion": {"risk": min(risk, 100), "priority": prio, "reasons": reasons}})
    return out


def write_overlay_png(res: dict, path: str | Path) -> None:
    """Слой для карты: ΔNDTI (оранжевый — рост мутности), облака серым, суша прозрачная."""
    d, water, cloud = res["dndti"], res["water"], res["cloud"]
    H, W = d.shape
    v = np.clip(d / 0.12, 0, 1)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (200 - 140 * v).astype(np.uint8)
    rgba[..., 2] = (60 * (1 - v)).astype(np.uint8)
    rgba[..., 3] = np.where(water & (d > 0.04), 60 + 180 * v, 0).astype(np.uint8)  # только значимый рост мутности
    rgba[cloud] = (180, 180, 190, 110)
    from .imaging import png_bytes

    png = png_bytes(rgba)
    Path(path).write_bytes(png)


def summary(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("dndti", "water", "cloud")}


__all__ = ["simulate", "detect", "to_alerts", "write_overlay_png", "summary", "json"]
