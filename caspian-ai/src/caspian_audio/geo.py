"""Геометрия на масштабе дельты (единицы км): локальная плоская система координат и полигоны зон.

Без shapely/pyproj: на участке в несколько километров равнопромежуточная проекция
даёт ошибку в сантиметры, этого с запасом хватает для пеленга и проверки зон.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EARTH_R = 6_371_000.0


class LocalFrame:
    """Метры на восток (x) и на север (y) от опорной точки."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_ll(self, x, y):
        return self.lat0 + np.asarray(y) / self.ky, self.lon0 + np.asarray(x) / self.kx


def parse_ts(s) -> float:
    """ISO-время (или уже число) -> секунды UNIX. Время без зоны считаем UTC."""
    if isinstance(s, (int, float)):
        return float(s)
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Чётность пересечений луча (ray casting); poly — (M, 2), без повтора первой точки."""
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cross = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.sum(cross & (x < xi)) % 2)


def ray_distance(p: np.ndarray, d: np.ndarray, poly: np.ndarray) -> float | None:
    """Расстояние вдоль луча p + s·d (d — единичный вектор) до ближайшей границы полигона."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-12
    w = a - p
    s = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1), -1)
    u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1), -1)
    hit = ok & (s > 0) & (u >= 0) & (u <= 1)
    return float(s[hit].min()) if hit.any() else None


def load_zones(path: str | Path, frame: LocalFrame) -> list[dict]:
    """GeoJSON FeatureCollection (Polygon/MultiPolygon) -> зоны с кольцами в метрах.

    properties: name, kind (reserve_core | spawning | fairway | ...), forbid (список
    категорий или "all"), seasons ([{"from": "MM-DD", "to": "MM-DD"}], нет = круглый год),
    risk_points (вес зоны в оценке риска).
    """
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = []
    for f in fc.get("features", []):
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
        rings = []
        for poly in polys:
            ring = np.array(poly[0], dtype=float)  # внешний контур, дырки не поддерживаем
            if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            x, y = frame.to_xy(ring[:, 1], ring[:, 0])
            rings.append(np.stack([x, y], axis=1))
        zones.append({**f.get("properties", {}), "rings": rings, "geometry": g})
    return zones


def in_zone(zone: dict, x: float, y: float) -> bool:
    return any(point_in_polygon(x, y, r) for r in zone["rings"])
