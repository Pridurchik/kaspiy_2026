"""Прогноз миграции осетровых: где рыба будет завтра (тепловая карта по Каспию).

Два слоя модели:
  1) Пригодность среды (HSI) по температуре, солёности и глубине (с учётом уровня моря) для стадии:
       * молодь — предпочтения из постановки задачи: солёность 8–9‰, 24–28 °C (ТРЕБУЕТ проверки
         ихтиологами; параметры в SPECS, их можно заменить);
       * производители весной — нерестовая миграция к устьям рек (Волга, Урал, Кура, ...),
         температура 8–18 °C, притяжение к устью тем сильнее, чем больше сток реки.
     Пригодность можно ОБУЧИТЬ на наблюдениях (учётные траловые съёмки, прилов, мечение):
     логистическая регрессия по T, T², S, S², ln(глубины) — fit_observations().
  2) Движение: рыба за сутки смещается на ограниченное расстояние (молодь ~8 км, взрослые ~25 км)
     в сторону более пригодной среды; масса каждой ячейки сохраняется (ядро перемещения,
     взвешенное пригодностью цели).
Выход: распределение по дням, центр масс и его смещение, доли по водам прибрежных государств.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .caspian import RIVERS, Grid, _km, country_shares, environment

SPECS = {
    "juvenile": {"name": "Молодь осетровых", "speed_km": 8,
                 "temp": (18, 24, 28, 31), "sal": (3, 8, 9, 13), "depth": (0.5, 2, 10, 30),
                 "start": "rivers", "spinup_days": 30, "floor": 0.005,  # молодь держится мелководья
                 "source": "предпочтения из постановки задачи (8–9‰, 24–28 °C) — проверить у ихтиологов"},
    "spawners": {"name": "Производители (нерестовая миграция)", "speed_km": 25,
                 "temp": (6, 9, 16, 21), "sal": (0, 0, 9, 13), "depth": (1, 3, 40, 120),
                 "start": "wintering", "spinup_days": 45, "drift_km": 15,
                 # хоуминг: доля стада, идущая к своей реке (Волго-Уральское стадо — основное)
                 "homing": {"Волга": 0.6, "Урал (Жайык)": 0.25, "Кура": 0.1, "Терек": 0.05},
                 "source": "ориентировочные литературные значения — калибровать по учётным данным"},
}


def _trap(x, a, b, c, d):
    """Трапеция: 0 вне (a, d), 1 на [b, c]."""
    up = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    down = np.clip((d - x) / max(d - c, 1e-6), 0, 1)
    return np.minimum(up, down)


def hsi(g: Grid, env: dict, stage: str, model: dict | None = None) -> np.ndarray:
    sp = SPECS[stage]
    if model:  # обученная модель вместо экспертных трапеций
        p = predict_observations(model, env, g)
    else:
        s = [_trap(env["temp"], *sp["temp"]), _trap(env["sal"], *sp["sal"]), _trap(env["depth"], *sp["depth"])]
        # мягкий нижний порог 0.05: неподходящую воду рыба может пересечь транзитом, но не задерживается
        p = np.prod([np.clip(x, sp.get("floor", 0.05), 1) for x in s], axis=0) ** (1 / 3)
    if sp.get("rivers"):
        # стремление к нерестовой реке: потенциал Σ сток·exp(−расстояние/120 км) — даёт направленный ход
        att = np.zeros(g.lon.shape)
        for lon, lat, w in RIVERS.values():
            x, y = _km(g.lon, g.lat, lon, lat)
            att += w * np.exp(-np.hypot(x, y) / 120)
        p = p * (0.01 + att)
    return np.where(env["wet"], np.nan_to_num(p), 0)


_GEO: dict = {}


def geodesic_next(g: Grid, wet: np.ndarray, river: str):
    """Кратчайший путь по воде до устья реки (Дейкстра, 8 соседей): для каждой ячейки — соседняя
    ячейка на пути к устью. Рыба обходит сушу (Апшерон, Мангышлак), маршруты идут вдоль берегов."""
    import heapq

    key = (id(g), river, int(wet.sum()))
    if key in _GEO:
        return _GEO[key]
    H, W = wet.shape
    lon, lat, _ = RIVERS[river]
    i0 = int(np.argmin(np.abs(g.lat[:, 0] - lat)))
    j0 = int(np.argmin(np.abs(g.lon[0] - lon)))
    cand = np.argwhere(wet)
    i0, j0 = cand[np.argmin((cand[:, 0] - i0) ** 2 + (cand[:, 1] - j0) ** 2)]  # ближайшая морская ячейка к устью
    dy = g.res * 111.2
    dx = g.res * 111.2 * np.cos(np.radians(43))
    dist = np.full((H, W), np.inf)
    dist[i0, j0] = 0
    heap = [(0.0, int(i0), int(j0))]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < H and 0 <= b < W and wet[a, b]:
                    nd = d + np.hypot(di * dy, dj * dx)
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(heap, (nd, a, b))
    nxt = np.arange(H * W).reshape(H, W)
    for i, j in np.argwhere(wet & np.isfinite(dist)):
        best, bi = dist[i, j], (i, j)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W and dist[a, b] < best:
                    best, bi = dist[a, b], (a, b)
        nxt[i, j] = bi[0] * W + bi[1]
    _GEO[key] = (nxt.ravel(), dist)
    return _GEO[key]


def _drift(p: np.ndarray, nxt: np.ndarray, steps: int) -> np.ndarray:
    q = p.ravel()
    for _ in range(steps):
        r = np.zeros_like(q)
        np.add.at(r, nxt, q)
        q = r
    return q.reshape(p.shape)


def start_distribution(g: Grid, sp: dict, env: dict) -> np.ndarray:
    """Где стадия находится в начале сезона: производители — на зимовке (глубже, Средний и Южный
    Каспий), молодь — у устьев нерестовых рек (скат личинок)."""
    if sp["start"] == "wintering":
        p = _trap(env["depth"], 10, 40, 400, 900) * np.clip((44.5 - g.lat) / 3, 0, 1)
    else:
        p = np.zeros(g.lon.shape)
        for lon, lat, w in RIVERS.values():
            x, y = _km(g.lon, g.lat, lon, lat)
            p += w * np.exp(-np.hypot(x, y) / 30)
    p = np.where(env["wet"], p, 0)
    return p / (p.sum() or 1)


def _blur(p: np.ndarray, sigma_cells: float) -> np.ndarray:
    r = max(1, int(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    q = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, p)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, q)


def forecast(g: Grid, today: date, days: int = 7, stage: str = "spawners", sensors: list[dict] | None = None,
             level: float = -29.0, beta: float = 2.0, model: dict | None = None) -> dict:
    """Распределение рыбы на сегодня и следующие дни. Среда: датчики сегодня, далее — их поправка
    затухает к климатической основе (τ ≈ 5 сут)."""
    sp = SPECS[stage]
    env0 = environment(g, today, sensors, level)
    clim0 = environment(g, today, None, level)
    cell_km = g.res * 111.2 * np.cos(np.radians(43))
    sig = sp["speed_km"] / cell_km

    def step(p, env):
        # каждая ячейка-источник раздаёт свою рыбу только в пределах суточной дальности,
        # пропорционально пригодности (масса сохраняется, рыба не «телепортируется» к максимуму):
        # P'(x) = H(x)^β · Σ_y K(x−y) P(y) / Z(y),  Z(y) = Σ_x K(x−y) H(x)^β
        h = np.where(env["wet"], hsi(g, env, stage, model), 0) ** beta + 1e-9 * env["wet"]
        q = h * _blur(p / np.maximum(_blur(h, sig), 1e-12), sig)
        q = np.where(env["wet"], q, 0)
        return q / (q.sum() or 1)

    # раскрутка: стартовое сезонное распределение (зимовка / устья рек) и сутки за сутками до сегодня
    homing = sp.get("homing") or {"": 1.0}
    p0 = start_distribution(g, sp, environment(g, today - timedelta(days=sp["spinup_days"]), None, level))
    subs = {r: w * p0 for r, w in homing.items()}  # субпопуляции по родной реке
    nxts = {r: geodesic_next(g, env0["wet"], r)[0] for r in homing if r}
    steps = int(round(sp.get("drift_km", 0) / cell_km))

    def advance(env):
        for r in subs:
            q = step(subs[r], env) * subs[r].sum()
            if r and steps:
                q = _drift(q, nxts[r], steps)  # направленный ход к своей реке по воде
            subs[r] = q

    def total():
        t = sum(subs.values())
        return t / (t.sum() or 1)

    for k in range(sp["spinup_days"], 0, -1):
        advance(environment(g, today - timedelta(days=k - 1), sensors if k == 1 else None, level))
    maps = [total()]
    for k in range(1, days + 1):
        ck = environment(g, today + timedelta(days=k), None, level)
        # поправка датчиков затухает к климатической основе (τ ≈ 5 сут)
        env = {**ck, **{v: ck[v] + (env0[v] - clim0[v]) * np.exp(-k / 5) for v in ("temp", "sal")}}
        advance(env)
        maps.append(total())
    res = {"stage": stage, "name": sp["name"], "source": sp["source"], "today": today.isoformat(), "maps": maps, "days": []}
    for k, m in enumerate(maps):
        w = m / (m.sum() or 1)
        clat, clon = float((w * g.lat).sum()), float((w * g.lon).sum())
        res["days"].append({"date": (today + timedelta(days=k)).isoformat(), "center": [round(clat, 3), round(clon, 3)],
                            "shares": country_shares(g, m), "top_squares": top_squares(g, m)})
    if days == 0:
        return res
    c0, c1 = res["days"][0]["center"], res["days"][1]["center"]
    dx, dy = _km(c1[1], c1[0], c0[1], c0[0])
    res["shift_km_day"] = round(float(np.hypot(dx, dy)), 1)
    res["shift_course_deg"] = round(float(np.degrees(np.arctan2(dx, dy)) % 360))
    return res


def top_squares(g: Grid, p: np.ndarray, n: int = 5) -> list[dict]:
    """Квадраты 0.5° с наибольшей вероятностью нахождения рыбы."""
    acc: dict[str, float] = {}
    cc: dict[str, str] = {}
    for (i, j) in np.argwhere(p > 0):
        sq = g.square(g.lat[i, j], g.lon[i, j])
        acc[sq] = acc.get(sq, 0) + float(p[i, j])
        cc.setdefault(sq, str(g.country[i, j]))
    tot = sum(acc.values()) or 1
    best = sorted(acc.items(), key=lambda kv: -kv[1])[:n]
    return [{"square": s, "p": round(v / tot, 3), "country": cc[s], "bounds": g.square_bounds(s)} for s, v in best]


# ---------------------------------------------------------------- обучение на наблюдениях

def _features(temp, sal, depth):
    ld = np.log1p(np.maximum(depth, 0))
    return np.stack([temp, temp ** 2 / 10, sal, sal ** 2 / 10, ld, ld ** 2 / 3], axis=-1)


def fit_observations(obs: list[dict], l2: float = 1e-3, iters: int = 3000) -> dict:
    """Логистическая регрессия: наличие рыбы (0/1) по среде в точке наблюдения.
    obs: [{"temp", "sal", "depth", "present"}] — учётные съёмки, прилов, мечение."""
    X = _features(np.array([o["temp"] for o in obs]), np.array([o["sal"] for o in obs]), np.array([o["depth"] for o in obs]))
    y = np.array([o["present"] for o in obs], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):  # градиентный спуск
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= 0.5 * (Z.T @ g / len(y) + l2 * w)
        b -= 0.5 * g.mean()
    model = {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist()}
    # выученные оптимумы: вершина параболы по T и S
    wt = np.array(w) / sd
    model["opt_temp"] = round(float(-wt[0] / (2 * wt[1] / 10)), 1) if wt[1] < 0 else None
    model["opt_sal"] = round(float(-wt[2] / (2 * wt[3] / 10)), 1) if wt[3] < 0 else None
    return model


def predict_observations(model: dict, env: dict, g: Grid | None = None) -> np.ndarray:
    X = _features(np.nan_to_num(env["temp"]), np.nan_to_num(env["sal"]), env["depth"])
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(model["w"]) + model["b"])))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    r = score.argsort().argsort() + 1.0
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def simulate_observations(g: Grid, days: list[date], stage: str = "juvenile", n_per_day: int = 120, seed: int = 0) -> list[dict]:
    """ДЕМО: синтетические учётные съёмки (наличие ~ экспертная пригодность). Только чтобы показать
    конвейер обучения; заменить реальными данными траловых съёмок."""
    from .caspian import _truth

    rng = np.random.default_rng(seed)
    obs = []
    for d in days:
        env = {**_truth(g, d, seed), **environment(g, d, None)}
        env["temp"], env["sal"] = _truth(g, d, seed)["temp"], _truth(g, d, seed)["sal"]
        h = hsi(g, env, stage)
        idx = np.argwhere(env["wet"])
        for i, j in idx[rng.choice(len(idx), n_per_day, replace=False)]:
            obs.append({"date": d.isoformat(), "lat": float(g.lat[i, j]), "lon": float(g.lon[i, j]),
                        "temp": float(env["temp"][i, j]), "sal": float(env["sal"][i, j]), "depth": float(env["depth"][i, j]),
                        "present": int(rng.random() < 0.05 + 0.85 * h[i, j])})
    return obs
