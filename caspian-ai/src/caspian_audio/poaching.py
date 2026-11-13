"""Прогноз браконьерских «горячих точек»: где вероятнее всего появятся браконьеры завтра.

Браконьеры идут за рыбой, но не куда угодно: недалеко от берега и мест выхода лодок, в тёмные
ночи и тихую погоду, туда, где их уже ловили. Модель учится на истории задержаний и инцидентов
(выгрузки операций рыбоохраны, например ежегодной казахстанской операции «Бекіре», и наши
акустические тревоги) и сопоставляет её с прогнозом миграции.

Признаки ячейки-дня: ln(плотность рыбы по модели миграции), расстояние до берега, до ближайшего
посёлка/порта, плотность прошлых инцидентов (ядро 15 км, только ДО этого дня — без подглядывания),
тёмность ночи (фаза луны), ветер. Модель — логистическая регрессия: коэффициенты читаются как
«что повышает риск», и для каждой горячей точки видно, какие признаки дали вклад.
Проверка — по времени: обучение на первых 18 месяцах, оценка на последних 6.

ВНИМАНИЕ: в демо история инцидентов СИНТЕТИЧЕСКАЯ (сгенерирована скрытым процессом, похожим на
реальность), поэтому метрики показывают только работу конвейера. Для реальной модели нужна
выгрузка задержаний: дата, координаты, тип нарушения.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .caspian import SETTLEMENTS, Grid, _km
from .migration import auc

FEATURES = ["ln_fish", "coast_km", "ln_settle_km", "history", "moon_dark", "wind"]
# как назвать вклад признака в объяснении: (значение выше среднего, ниже среднего)
FEATURES_WHY = {"ln_fish": ("много рыбы по прогнозу миграции", "мало рыбы"),
                "coast_km": ("далеко от берега", "близко к берегу"),
                "ln_settle_km": ("далеко от посёлков", "рядом посёлки и места выхода лодок"),
                "history": ("здесь уже были инциденты", "инцидентов не было"),
                "moon_dark": ("тёмная ночь (луна)", "светлая ночь"), "wind": ("ветер", "тихая погода")}
FEATURES_RU = {"ln_fish": "плотность рыбы (прогноз миграции)", "coast_km": "удалённость от берега",
               "ln_settle_km": "удалённость от посёлков и портов", "history": "прошлые инциденты рядом",
               "moon_dark": "тёмная ночь (луна)", "wind": "ветер"}


def moon_illumination(d: date) -> float:
    """Освещённость Луны 0..1 (новолуние 6 янв 2000 18:14 UTC, синодический месяц 29.53 сут)."""
    phase = ((d.toordinal() + 0.5 - date(2000, 1, 6).toordinal() - 0.76) / 29.530588) % 1
    return float((1 - np.cos(2 * np.pi * phase)) / 2)


def wind(d: date, seed: int = 0) -> float:
    """ДЕМО-погода: ветер, м/с (для реального — прогноз ветра Росгидромета / ERA5)."""
    r = np.random.default_rng(d.toordinal() * 7 + seed)
    return float(np.clip(5 + 3 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365) + r.gamma(2, 2), 0, 25))


class Static:
    """Неизменные признаки сетки (считаются один раз)."""

    def __init__(self, g: Grid):
        self.g = g
        self.coast = g.coast_km
        X, Y = _km(g.lon, g.lat, 46.5, 36.5)
        d = np.full(g.lon.shape, np.inf)
        for lon, lat in SETTLEMENTS:
            sx, sy = _km(lon, lat, 46.5, 36.5)
            d = np.minimum(d, np.hypot(X - sx, Y - sy))
        self.settle = d
        self.X, self.Y = X, Y


def history_density(st: Static, incidents: list[dict], before: date, window: int = 365, bw: float = 15.0) -> np.ndarray:
    """Плотность прошлых инцидентов (ядро Гаусса 15 км) за год ДО даты."""
    lo = before - timedelta(days=window)
    h = np.zeros(st.g.lon.shape)
    for inc in incidents:
        d = date.fromisoformat(inc["date"])
        if lo <= d < before:
            x, y = _km(inc["lon"], inc["lat"], 46.5, 36.5)
            h += np.exp(-((st.X - x) ** 2 + (st.Y - y) ** 2) / (2 * bw ** 2))
    return h


def features(st: Static, fish: np.ndarray, hist: np.ndarray, d: date, wind_ms: float | None = None) -> np.ndarray:
    w = wind(d) if wind_ms is None else wind_ms
    f = np.stack([np.log(fish * st.g.sea.sum() + 1e-3), st.coast, np.log1p(st.settle), np.log1p(hist),
                  np.full(fish.shape, 1 - moon_illumination(d)), np.full(fish.shape, w)], axis=-1)
    return f


# ---------------------------------------------------------------- синтетическая история (демо)

def simulate_history(g: Grid, fish_by_week: dict, start: date, days: int, seed: int = 11) -> list[dict]:
    """Скрытый «истинный» процесс: рыба^0.8 × близость к берегу и посёлкам × тёмная ночь × штиль."""
    rng = np.random.default_rng(seed)
    st = Static(g)
    base = np.exp(-st.coast / 15) * (0.2 + np.exp(-st.settle / 35))
    out = []
    for k in range(days):
        d = start + timedelta(days=k)
        fish = fish_by_week[_week(d)]
        lam = base * (fish * g.sea.sum() + 0.02) ** 0.8 * (0.35 + (1 - moon_illumination(d))) * (0.15 if wind(d) > 11 else 1.0)
        lam = np.where(g.sea, lam, 0)
        n = rng.poisson(1.6 * lam.sum() / 40)
        if n == 0:
            continue
        flat = lam.ravel() / lam.sum()
        for idx in rng.choice(flat.size, n, p=flat):
            i, j = divmod(int(idx), g.lon.shape[1])
            out.append({"date": d.isoformat(), "lat": round(float(g.lat[i, j] + rng.uniform(-0.05, 0.05)), 3),
                        "lon": round(float(g.lon[i, j] + rng.uniform(-0.05, 0.05)), 3), "country": str(g.country[i, j]),
                        "type": rng.choice(["задержание с уловом осетровых", "изъяты сети", "акустическая тревога"], p=[0.4, 0.45, 0.15]),
                        "synthetic": True})
    return out


def _week(d: date) -> int:
    return min(d.timetuple().tm_yday // 7, 51)


def seasonal_fish(g: Grid, sensors=None) -> dict:
    """Плотность производителей по неделям года (модель миграции, раскрутка от зимовки).
    До начала хода и после нереста — зимовочное распределение."""
    from .caspian import environment
    from .migration import SPECS, forecast, start_distribution

    out = {}
    winter = start_distribution(g, SPECS["spawners"], environment(g, date(2026, 1, 15), None))
    for wk in range(52):
        d = date(2026, 1, 1) + timedelta(days=7 * wk + 3)
        if date(2026, 3, 20) <= d <= date(2026, 7, 10):  # нерестовый ход
            out[wk] = forecast(g, d, 0, "spawners")["maps"][0]
        else:
            out[wk] = winter
    return out


# ---------------------------------------------------------------- обучение и прогноз

def fit(g: Grid, incidents: list[dict], fish_by_week: dict, start: date, days: int, split_day: int,
        neg_per_day: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    st = Static(g)
    sea_idx = np.argwhere(g.sea)
    by_day: dict[str, list] = {}
    for inc in incidents:
        by_day.setdefault(inc["date"], []).append(inc)
    rows, ys, ts = [], [], []
    hist_cache = None
    for k in range(30, days):  # первый месяц — только накопление истории
        d = start + timedelta(days=k)
        if k % 7 == 0 or hist_cache is None:
            hist_cache = history_density(st, incidents, d)
        F = features(st, fish_by_week[_week(d)], hist_cache, d)
        for inc in by_day.get(d.isoformat(), []):
            i = int(np.argmin(np.abs(g.lat[:, 0] - inc["lat"])))
            j = int(np.argmin(np.abs(g.lon[0] - inc["lon"])))
            rows.append(F[i, j]); ys.append(1); ts.append(k)
        for i, j in sea_idx[rng.choice(len(sea_idx), neg_per_day, replace=False)]:
            rows.append(F[i, j]); ys.append(0); ts.append(k)
    X, y, t = np.array(rows), np.array(ys, float), np.array(ts)
    tr, te = t < split_day, t >= split_day
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(2500):
        p = 1 / (1 + np.exp(-(Z[tr] @ w + b)))
        gr = p - y[tr]
        w -= 0.3 * (Z[tr].T @ gr / tr.sum() + 1e-3 * w)
        b -= 0.3 * gr.mean()
    s_te = Z[te] @ w + b
    model = {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(), "features": FEATURES,
             "n_incidents_train": int(y[tr].sum()), "n_incidents_test": int(y[te].sum()),
             "auc_test": round(auc(s_te, y[te]), 3)}
    # доля будущих инцидентов в топ-5% площади моря по прогнозу (стандартная метрика предиктивного патрулирования)
    model["hit_rate_top5"] = _hit_rate(g, st, model, incidents, fish_by_week, start, split_day, days)
    return model


def _hit_rate(g, st, model, incidents, fish_by_week, start, split_day, days, top=0.05):
    hits = tot = 0
    by_day: dict[str, list] = {}
    for inc in incidents:
        by_day.setdefault(inc["date"], []).append(inc)
    for k in range(split_day, days, 3):
        d = start + timedelta(days=k)
        if d.isoformat() not in by_day:
            continue
        r = predict(st, model, fish_by_week[_week(d)], history_density(st, incidents, d), d)
        thr = np.quantile(r[g.sea], 1 - top)
        for inc in by_day[d.isoformat()]:
            i = int(np.argmin(np.abs(g.lat[:, 0] - inc["lat"])))
            j = int(np.argmin(np.abs(g.lon[0] - inc["lon"])))
            hits += r[i, j] >= thr
            tot += 1
    return round(hits / max(tot, 1), 3)


def predict(st: Static, model: dict, fish: np.ndarray, hist: np.ndarray, d: date, wind_ms: float | None = None) -> np.ndarray:
    Z = (features(st, fish, hist, d, wind_ms) - np.array(model["mu"])) / np.array(model["sd"])
    r = 1 / (1 + np.exp(-(Z @ np.array(model["w"]) + model["b"])))
    return np.where(st.g.sea, r, 0)


def hotspots(g: Grid, st: Static, model: dict, risk: np.ndarray, fish: np.ndarray, hist: np.ndarray, d: date, n: int = 6) -> list[dict]:
    """Квадраты 0.5° с максимальным риском + объяснение: какие признаки дали вклад."""
    acc: dict[str, tuple] = {}
    for i, j in np.argwhere(g.sea):
        sq = g.square(g.lat[i, j], g.lon[i, j])
        if sq not in acc or risk[i, j] > acc[sq][0]:
            acc[sq] = (float(risk[i, j]), i, j)
    top = sorted(acc.items(), key=lambda kv: -kv[1][0])[:n]
    F = features(st, fish, hist, d)
    out = []
    for sq, (r, i, j) in top:
        z = (F[i, j] - np.array(model["mu"])) / np.array(model["sd"])
        contrib = z * np.array(model["w"])
        why = [FEATURES_WHY[FEATURES[k]][0 if z[k] > 0 else 1] for k in np.argsort(-contrib)[:3] if contrib[k] > 0.1]
        out.append({"square": sq, "risk": round(r, 3), "lat": round(float(g.lat[i, j]), 2), "lon": round(float(g.lon[i, j]), 2),
                    "country": str(g.country[i, j]), "bounds": g.square_bounds(sq), "why": why})
    return out


def explain_model(model: dict) -> list[str]:
    """Коэффициенты по-русски: знак = повышает/понижает риск."""
    out = []
    for f, w in sorted(zip(model["features"], model["w"]), key=lambda kv: -abs(kv[1])):
        out.append(f"{FEATURES_RU[f]}: {'↑ повышает' if w > 0 else '↓ понижает'} риск ({w:+.2f})")
    return out
