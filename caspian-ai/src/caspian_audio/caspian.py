"""Каспий целиком: сетка 0.1°, схематичный контур моря, прибрежные государства, среда по датчикам.

Масштаб модуля — всё море (миграция осетровых и трансграничный обмен), а не один участок дельты.

  * Контур — СХЕМАТИЧНЫЙ (≈40 точек): годится для карт прогноза с ячейкой ~10 км, не для навигации.
  * Принадлежность ячейки государству — по ближайшему участку побережья. Это НЕ морская граница
    (разграничение дна Каспия определяется двусторонними соглашениями), а только маршрутизация
    уведомлений: «кому из соседей это важно».
  * Среда (температура, солёность, уровень): климатическая основа + поправка по реальным датчикам
    (обратно-взвешенная интерполяция невязок). Если датчиков нет — только климатическая основа.
    В демо и основа, и датчики синтетические.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

COUNTRIES = {
    "RU": {"ru": "Россия", "en": "Russian Federation"},
    "KZ": {"ru": "Казахстан", "en": "Kazakhstan"},
    "TM": {"ru": "Туркменистан", "en": "Turkmenistan"},
    "IR": {"ru": "Иран", "en": "Iran"},
    "AZ": {"ru": "Азербайджан", "en": "Azerbaijan"},
}

# Схематичный контур (lon, lat) по часовой стрелке от устья Самура, с государством участка побережья
COAST = [
    (48.55, 41.85, "RU"), (48.30, 42.06, "RU"), (47.50, 42.98, "RU"), (47.62, 43.50, "RU"), (47.30, 44.40, "RU"),
    (47.40, 45.40, "RU"), (47.80, 45.70, "RU"), (48.50, 45.85, "RU"), (49.25, 46.25, "RU"),
    (49.80, 46.55, "KZ"), (50.60, 46.75, "KZ"), (51.30, 46.95, "KZ"), (51.90, 46.90, "KZ"), (52.80, 46.60, "KZ"),
    (53.15, 46.00, "KZ"), (52.80, 45.45, "KZ"), (51.80, 45.35, "KZ"), (51.10, 45.25, "KZ"), (50.85, 44.85, "KZ"),
    (50.30, 44.55, "KZ"), (50.80, 44.05, "KZ"), (51.20, 43.60, "KZ"), (51.50, 43.15, "KZ"), (52.40, 42.65, "KZ"),
    (52.70, 41.80, "TM"), (52.90, 41.00, "TM"), (52.90, 40.00, "TM"), (53.20, 39.35, "TM"), (53.60, 38.50, "TM"),
    (53.95, 37.40, "IR"), (53.90, 36.90, "IR"), (53.00, 36.75, "IR"), (52.00, 36.62, "IR"), (51.00, 36.75, "IR"),
    (50.20, 37.10, "IR"), (49.50, 37.45, "IR"), (48.90, 38.43, "AZ"), (48.85, 38.80, "AZ"), (49.30, 39.30, "AZ"),
    (49.40, 39.90, "AZ"), (49.85, 40.30, "AZ"), (50.40, 40.45, "AZ"), (49.80, 40.60, "AZ"), (49.40, 40.95, "AZ"),
    (48.65, 41.75, "AZ"),
]
# Устья рек: (lon, lat, вес притока) — нерестовые реки осетровых
RIVERS = {
    "Волга": (48.40, 45.70, 1.0), "Урал (Жайык)": (51.80, 46.85, 0.5), "Терек": (47.60, 43.62, 0.15),
    "Сулак": (47.55, 43.27, 0.1), "Кура": (49.45, 39.35, 0.3), "Сефидруд": (49.95, 37.45, 0.12), "Самур": (48.55, 41.87, 0.05),
}
# Прибрежные посёлки и порты (места выхода лодок) — для модели браконьерских маршрутов
SETTLEMENTS = [
    (48.30, 45.90), (47.37, 45.39), (47.50, 42.98), (48.29, 42.06), (51.90, 47.10), (51.17, 43.65), (50.26, 44.51),
    (52.97, 40.02), (53.00, 39.40), (53.97, 37.47), (54.07, 36.90), (52.65, 36.72), (49.47, 37.47), (48.87, 38.42),
    (48.85, 38.75), (49.25, 39.37), (49.85, 40.38), (49.67, 40.59), (48.60, 41.80), (47.65, 43.60),
]
REF_LEVEL = -28.0  # м БС: уровень, к которому привязаны глубины схемы


def _km(lon, lat, lon0, lat0):
    """Плоское приближение в км (на масштабе Каспия ошибка ~1%)."""
    return (np.asarray(lon) - lon0) * 111.2 * np.cos(np.radians(43.0)), (np.asarray(lat) - lat0) * 111.2


def _in_poly(lon, lat, poly):
    x, y = lon.ravel(), lat.ravel()
    inside = np.zeros(x.shape, bool)
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    for a, b, c, d in zip(px, py, qx, qy):
        cross = (b > y) != (d > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a + (y - b) * (c - a) / (d - b)
        inside ^= cross & (x < xi)
    return inside.reshape(lon.shape)


@dataclass
class Grid:
    lon: np.ndarray      # (H, W) центры ячеек
    lat: np.ndarray
    sea: np.ndarray      # маска моря
    country: np.ndarray  # код государства ближайшего побережья ("" на суше)
    coast_km: np.ndarray
    depth: np.ndarray    # м при REF_LEVEL (схема)
    res: float
    bounds: list         # [[lat_min, lon_min], [lat_max, lon_max]] для карты

    def square(self, lat: float, lon: float, size: float = 0.5) -> str:
        """Квадрат 0.5° для сообщений: буква строки (с севера) + номер столбца, напр. «C7»."""
        r = int((47.5 - lat) // size)
        c = int((lon - 46.5) // size) + 1
        return f"{chr(ord('A') + max(r, 0))}{c}"

    def square_bounds(self, sq: str, size: float = 0.5) -> list:
        r = ord(sq[0]) - ord("A")
        c = int(sq[1:]) - 1
        lat1 = 47.5 - r * size
        lon0 = 46.5 + c * size
        return [round(lat1 - size, 2), round(lon0, 2), round(lat1, 2), round(lon0 + size, 2)]


def build_grid(res: float = 0.1) -> Grid:
    lons = np.arange(46.5 + res / 2, 55.0, res)
    lats = np.arange(47.3 - res / 2, 36.4, -res)
    LON, LAT = np.meshgrid(lons, lats)
    poly = np.array([(a, b) for a, b, _ in COAST])
    sea = _in_poly(LON, LAT, poly)
    # побережье с шагом ~5 км: расстояние до берега и государство ближайшего участка
    pts, tags = [], []
    for (a, b, t), (c, d, _) in zip(COAST, COAST[1:] + COAST[:1]):
        n = max(2, int(np.hypot(*_km(c, d, a, b)) / 5))
        for k in range(n):
            pts.append((a + (c - a) * k / n, b + (d - b) * k / n))
            tags.append(t)
    pts = np.array(pts)
    X, Y = _km(LON, LAT, 46.5, 36.5)
    px, py = _km(pts[:, 0], pts[:, 1], 46.5, 36.5)
    best_d = np.full(LON.shape, np.inf)
    best_i = np.zeros(LON.shape, int)
    for i in range(len(pts)):
        d = np.hypot(X - px[i], Y - py[i])
        m = d < best_d
        best_d[m], best_i[m] = d[m], i
    country = np.where(sea, np.array(tags)[best_i], "")
    # схема глубин: Северный Каспий мелкий (< 10 м), Средний до ~700 м, Южный до ~1000 м
    north = np.clip((LAT - 44.3) / 0.6, 0, 1)
    deep_mid = np.clip(3.2 * best_d ** 1.15, 0, 750)
    deep_south = np.clip(4.0 * best_d ** 1.15, 0, 1000)
    sill = np.exp(-((LAT - 40.4) / 0.35) ** 2)  # Апшеронский порог
    deep = np.where(LAT < 40.4, deep_south, deep_mid) * (1 - 0.7 * sill)
    shallow = 0.4 + 0.11 * best_d
    depth = np.where(sea, north * np.minimum(shallow, 12) + (1 - north) * np.maximum(deep, shallow), 0)
    b = [[float(lats.min() - res / 2), float(lons.min() - res / 2)], [float(lats.max() + res / 2), float(lons.max() + res / 2)]]
    return Grid(LON, LAT, sea, country, np.where(sea, best_d, 0), depth, res, b)


# ---------------------------------------------------------------- среда

def climatology(g: Grid, day: date, volga_q: float = 1.0) -> dict:
    """Климатическая основа температуры и солёности на дату (схема, демо)."""
    doy = day.timetuple().tm_yday
    t_n = 12.5 - 12.5 * np.cos(2 * np.pi * (doy - 20) / 365)   # мелкий север: ~0 °C зимой, ~25 °C в июле
    t_s = 18.5 - 9.0 * np.cos(2 * np.pi * (doy - 40) / 365)    # юг: ~9.5 °C в феврале, ~27.5 °C в августе
    w = np.clip((g.lat - 38.0) / 8.0, 0, 1)
    temp = w * t_n + (1 - w) * t_s
    temp = temp + np.where(g.depth < 5, 0.8 * np.sin(2 * np.pi * (doy - 80) / 365), 0)  # мелководье быстрее прогревается
    sal = np.full(g.lon.shape, 12.8)
    for name, (lon, lat, wgt) in RIVERS.items():
        x, y = _km(g.lon, g.lat, lon, lat)
        dist = np.hypot(x, y)
        radius = {"Волга": 170 * volga_q, "Урал (Жайык)": 70}.get(name, 25 + 60 * wgt)
        sal = sal * (1 - 0.97 * np.exp(-dist / radius) * min(1.0, 0.4 + wgt))
    # мелководные заливы северо-востока солонее за счёт испарения
    x, y = _km(g.lon, g.lat, 52.6, 45.6)
    sal = sal + 4.0 * np.exp(-np.hypot(x, y) / 60)
    return {"temp": np.where(g.sea, temp, np.nan), "sal": np.where(g.sea, np.clip(sal, 0.1, 15), np.nan)}


def simulate_sensors(g: Grid, day: date, n: int = 36, seed: int = 0) -> list[dict]:
    """Сеть существующих датчиков (буи, платформы, береговые станции) — демо: истина = основа + аномалии."""
    rng = np.random.default_rng(seed + day.toordinal())
    truth = _truth(g, day, seed)
    idx = np.argwhere(g.sea)
    pick = idx[rng.choice(len(idx), n, replace=False)]
    out = []
    for k, (i, j) in enumerate(pick):
        out.append({"id": f"S{k + 1:02d}", "lat": round(float(g.lat[i, j]), 2), "lon": round(float(g.lon[i, j]), 2),
                     "country": str(g.country[i, j]), "temp": round(float(truth["temp"][i, j] + rng.normal(0, 0.2)), 2),
                     "sal": round(float(truth["sal"][i, j] + rng.normal(0, 0.1)), 2)})
    return out


def _truth(g: Grid, day: date, seed: int = 0) -> dict:
    """«Истинная» среда для демо: основа + медленные аномалии (тёплое пятно, распреснение)."""
    c = climatology(g, day, volga_q=1.25)  # многоводный год
    rng = np.random.default_rng(seed)
    t, s = c["temp"].copy(), c["sal"].copy()
    for _ in range(4):
        lon0, lat0 = rng.uniform(47.5, 53.5), rng.uniform(37.5, 46.5)
        x, y = _km(g.lon, g.lat, lon0, lat0)
        blob = np.exp(-(x ** 2 + y ** 2) / (2 * rng.uniform(60, 140) ** 2))
        t = t + rng.normal(0, 1.5) * blob
        s = s + rng.normal(0, 0.8) * blob
    return {"temp": t, "sal": np.clip(s, 0.1, 16)}


def environment(g: Grid, day: date, sensors: list[dict] | None = None, level: float = -29.0, power: float = 2.0) -> dict:
    """Поле среды на дату: основа + интерполяция невязок датчиков (IDW, радиус влияния ~200 км)."""
    c = climatology(g, day)
    out = {"temp": c["temp"], "sal": c["sal"]}
    if sensors:
        X, Y = _km(g.lon, g.lat, 46.5, 36.5)
        for var in ("temp", "sal"):
            num = np.zeros(g.lon.shape)
            den = np.full(g.lon.shape, 1e-6)
            for s in sensors:
                i = int(np.argmin(np.abs(g.lat[:, 0] - s["lat"])))
                j = int(np.argmin(np.abs(g.lon[0] - s["lon"])))
                resid = s[var] - c[var][i, j]
                sx, sy = _km(s["lon"], s["lat"], 46.5, 36.5)
                d = np.hypot(X - sx, Y - sy) + 5
                wgt = np.exp(-d / 200) / d ** power
                num += wgt * resid
                den += wgt
            out[var] = c[var] + num / den * (1 - np.exp(-den * 1e4))  # далеко от датчиков — к климату
    eff = g.depth + (level - REF_LEVEL)  # падение уровня осушает мелководье севера
    out["depth"] = np.where(g.sea, np.maximum(eff, 0), 0)
    out["wet"] = g.sea & (eff > 0.3)
    out["level"] = level
    return out


def country_shares(g: Grid, p: np.ndarray) -> dict:
    tot = float(np.nansum(p)) or 1.0
    return {c: round(float(np.nansum(np.where(g.country == c, p, 0)) / tot), 3) for c in COUNTRIES}


def to_png(g: Grid, field: np.ndarray, vmax: float | None = None, alpha_min: float = 0.05) -> bytes:
    """Тепловая карта для слоя карты: палитра + прозрачность там, где значение мало."""
    from .imaging import colormap, png_bytes

    v = np.nan_to_num(field, nan=0.0)
    vmax = vmax or float(np.nanpercentile(v[g.sea], 99.5)) or 1.0
    v = np.clip(v / vmax, 0, 1)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., :3] = colormap(v)
    rgba[..., 3] = np.where(g.sea & (v > alpha_min), (70 + 170 * v), 0).astype(np.uint8)
    return png_bytes(np.ascontiguousarray(rgba))
