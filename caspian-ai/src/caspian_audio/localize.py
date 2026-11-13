"""Пеленг по нескольким буям: TDOA (GCC-PHAT) -> точка -> трек с курсом и скоростью.

Буи пишут звук с общим временем (GPS, PPS: ошибка синхронизации ~1 мкс). Звук приходит
на ближний буй раньше, на дальний позже; разность времён прихода (TDOA) для пары буёв
задаёт гиперболу, пересечение гипербол — точка источника.

  * GCC-PHAT: взаимная корреляция с выбеливанием спектра. Острый пик даже у шумового
    сигнала (кавитация винта), устойчив к окраске канала.
  * Компенсация Доплера: у идущей лодки разность задержек «плывёт» на миллисекунды
    за окно, широкополосный пик расплывается, и побеждают ложные пики гармоник
    (кратные периоду вращения, ошибка 15–60 м). Перебираем скорость изменения задержки
    и растягиваем сигнал — на симуляции ошибка задержки падает с ~200 мс до ~0.02 мс.
  * 3 буя -> 2 независимые TDOA на 2 неизвестные: решение может быть неоднозначным.
    Добавляем слабый априор по уровню (ближний буй громче): он не двигает точку,
    но выбирает правильную ветвь гипербол.
  * Трек: фильтр Калмана с постоянной скоростью + сглаживание RTS, выбросы отсекаются
    по расстоянию Махаланобиса.

Скорость звука зависит от того, чем слушаем: мотор — гидрофон (вода, ~1480 м/с),
выстрел/пила — микрофон над водой (воздух, ~343 м/с). Задаётся по классам в конфиге.
Ограничения в поле: многолучёвость на мелководье, ветер (в воздухе). Поэтому радиус
неопределённости берём не из идеальной теории, а с запасом (tdoa_sigma_ms).
"""
from __future__ import annotations

import numpy as np


def _per_class(table: dict, cls: str):
    return table.get(cls, table["default"])


def gcc_phat(ref: np.ndarray, other: np.ndarray, sr: int, band=None, interp: int = 8):
    """Корреляция `ref` с более длинным `other`.

    Возвращает (k, psr): k — сдвиг в отсчётах (дробный), при котором ref[n] ≈ other[n + k];
    psr — острота пика (пик / СКО корреляции в окне поиска), мера доверия.
    """
    n = 1 << int(np.ceil(np.log2(len(ref) + len(other))))
    A, B = np.fft.rfft(ref, n), np.fft.rfft(other, n)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-12
    if band:
        f = np.fft.rfftfreq(n, 1 / sr)
        R[(f < band[0]) | (f > band[1])] = 0
    cc = np.fft.irfft(R, n * interp)
    m = (len(other) - len(ref) + 1) * interp  # допустимые сдвиги 0..len(other)-len(ref)
    seg = cc[:m]
    k = int(np.argmax(seg))
    psr = float(seg[k] / (np.std(seg) + 1e-12))
    return k / interp, psr


def gcc_phat_doppler(ref: np.ndarray, other: np.ndarray, sr: int, band, alphas, interp: int = 8):
    """GCC-PHAT с компенсацией движения источника.

    У движущейся лодки разность задержек меняется: τ(t) = τ0 + α·t (|α| до 2v/c ≈ 0.02
    при 15 м/с в воде). За 0.5 с окна это миллисекунды, а когерентность широкополосного
    шума ~0.2 мс: без компенсации пик кавитации расплывается и побеждают ложные пики
    гармоник (кратные периоду вращения). Перебираем α, растягиваем `other` во времени
    вокруг его центра и берём α с самым острым пиком.

    Возвращает (delay_s, psr, alpha): delay_s — задержка `other` относительно `ref`
    в середине окна ref, считая от начала `other` минус начало `ref` (добавить снаружи).
    """
    n = 1 << int(np.ceil(np.log2(len(ref) + len(other))))
    A = np.conj(np.fft.rfft(ref, n))
    f = np.fft.rfftfreq(n, 1 / sr)
    mask = (f >= band[0]) & (f <= band[1]) if band else np.ones_like(f, bool)
    m = len(other) - len(ref) + 1
    u = np.arange(len(other), dtype=float)
    cen = (len(other) - 1) / 2
    best = (-np.inf, 0.0, 0.0)
    for a in alphas:
        y = other if a == 0 else np.interp(cen + (u - cen) * (1 + a), u, other)
        R = np.fft.rfft(y, n) * A
        R = np.where(mask, R / (np.abs(R) + 1e-12), 0)
        cc = np.fft.irfft(R, n)[:m]
        k = int(np.argmax(cc))
        psr = float(cc[k] / (np.std(cc) + 1e-12))
        if psr > best[0]:
            best = (psr, a, k)
    psr, a, k = best
    # уточнение лага с передискретизацией корреляции для лучшего α
    y = other if a == 0 else np.interp(cen + (u - cen) * (1 + a), u, other)
    R = np.fft.rfft(y, n) * A
    R = np.where(mask, R / (np.abs(R) + 1e-12), 0)
    cc = np.fft.irfft(R, n * interp)[: m * interp]
    lo = max(0, k * interp - interp)
    kf = (lo + int(np.argmax(cc[lo : k * interp + interp + 1]))) / interp
    # ref[i] ≈ y[i + kf] = other[cen + (i + kf - cen)(1+a)]; в середине окна ref (i = len(ref)/2):
    i_mid = len(ref) / 2
    pos = cen + (i_mid + kf - cen) * (1 + a)
    return (pos - i_mid) / sr, psr, a


def tdoa_window(signals: list[np.ndarray], buoys_xy: np.ndarray, sr: int, t: float, win_s: float,
                c: float, band, min_psr: float, alphas=(0.0,), ref: int | None = None) -> dict | None:
    """Задержки прихода для окна [t, t+win] относительно опорного (по умолчанию самого громкого) буя."""
    i0, n = int(t * sr), int(win_s * sr)
    if i0 < 0 or i0 + n > min(len(s) for s in signals):
        return None

    def band_rms(x):
        X = np.fft.rfft(x * np.hanning(len(x)))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        m = (f >= band[0]) & (f <= band[1]) if band else slice(None)
        return float(np.sqrt(np.mean(np.abs(X[m]) ** 2)))

    amps = np.array([band_rms(s[i0 : i0 + n]) for s in signals])
    ref = int(np.argmax(amps)) if ref is None else ref
    a = signals[ref][i0 : i0 + n]
    delays = np.full(len(signals), np.nan)
    psr = np.zeros(len(signals))
    delays[ref], psr[ref] = 0.0, np.inf
    for j, s in enumerate(signals):
        if j == ref:
            continue
        max_tau = np.hypot(*(buoys_xy[j] - buoys_xy[ref])) / c + 0.02
        lo, hi = max(0, i0 - int(max_tau * sr)), min(len(s), i0 + n + int(max_tau * sr))
        dd, q, _ = gcc_phat_doppler(a, s[lo:hi], sr, band, alphas)
        d = dd + (lo - i0) / sr
        if q >= min_psr and abs(d) <= max_tau:
            delays[j], psr[j] = d, q
    if np.sum(~np.isnan(delays)) < 3:
        return None
    return {"t": t, "ref": ref, "delays": delays, "psr": psr, "amps": amps}


def solve_position(buoys_xy: np.ndarray, ref: int, delays: np.ndarray, c: float, amps: np.ndarray | None,
                   sigma_t: float, level_sigma: float, search_radius: float, step: float,
                   speed_rel_err: float = 0.0) -> dict:
    """Точка источника по TDOA: перебор по сетке (глобальный минимум) + Гаусс-Ньютон."""
    use = [j for j in range(len(buoys_xy)) if j != ref and not np.isnan(delays[j])]
    meas = c * delays[use]
    # ошибка разности хода: случайная (c·σ_t) + систематическая от неточной скорости звука (ε·|Δr|)
    sd = np.sqrt((c * sigma_t) ** 2 + (speed_rel_err * meas) ** 2)

    def residuals(P):  # P: (..., 2)
        r = np.linalg.norm(P[..., None, :] - buoys_xy, axis=-1)
        res = [(r[..., use] - r[..., [ref]] - meas) / sd]
        if amps is not None and level_sigma:
            # амплитуда ~ 1/r: ln(A_ref/A_j) ≈ ln(r_j/r_ref). Слабый вес — только выбрать ветвь.
            obs = np.log(amps[ref] / np.maximum(amps[use], 1e-12))
            res.append((np.log(np.maximum(r[..., use], 1.0) / np.maximum(r[..., [ref]], 1.0)) - obs) / level_sigma)
        return np.concatenate(res, axis=-1)

    lo, hi = buoys_xy.min(0) - search_radius, buoys_xy.max(0) + search_radius
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    G = np.stack([gx, gy], -1)
    cost = (residuals(G) ** 2).sum(-1)
    k = np.unravel_index(np.argmin(cost), cost.shape)
    P = G[k].astype(float)

    for _ in range(20):  # Гаусс-Ньютон с численным якобианом
        r0 = residuals(P)
        J = np.stack([(residuals(P + e) - r0) / 0.1 for e in (np.array([0.1, 0]), np.array([0, 0.1]))], -1)
        step_p, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        P = P + step_p
        if np.hypot(*step_p) < 0.01:
            break
    r = residuals(P)
    # неопределённость — только по TDOA (априор по уровню выбирает ветвь, но не уточняет точку):
    # вне треугольника буёв гиперболы почти параллельны, и радиус честно растёт
    Jt = J[: len(use)]
    cov = np.linalg.pinv(Jt.T @ Jt + 1e-9 * np.eye(2))
    # неоднозначность: есть ли далеко другая точка почти с той же стоимостью
    far = np.linalg.norm(G - P, axis=-1) > 300
    alt = float(cost[far].min()) if far.any() else np.inf
    best = float((r**2).sum())
    tdoa_rms = float(np.sqrt(np.mean((r[: len(use)] * sd) ** 2)))
    return {
        "x": float(P[0]), "y": float(P[1]),
        "cov": cov, "radius_m": float(2 * np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0))),  # ~95%
        "tdoa_rms_m": tdoa_rms, "ambiguous": bool(alt < best + 4.0), "n_buoys": len(use) + 1,
    }


def locate_span(signals, buoys_xy, sr, t0, t1, cls, lcfg) -> list[dict]:
    """Серия точек пеленга по событию [t0, t1] (время приёма по общим GPS-часам)."""
    c = _per_class(lcfg["sound_speed"], cls)
    band = _per_class(lcfg["band_hz"], cls)
    impulsive = cls in lcfg.get("impulsive_classes", ["gunshot"])
    win = lcfg["impulse_window_seconds"] if impulsive else lcfg["window_seconds"]
    hop = lcfg["hop_seconds"]
    sigma_t = lcfg["tdoa_sigma_ms"] / 1000
    alphas = (0.0,)
    if not impulsive:  # движущийся источник: перебор скорости изменения задержки (см. gcc_phat_doppler)
        amax, step = 2 * lcfg["max_source_speed_ms"] / c, lcfg["alpha_step_ms"] / 1000 / win
        alphas = np.arange(-amax, amax + step / 2, step)
    if impulsive:
        # одно окно на самом громком импульсе (на любом буе), чуть раньше пика. Выстрел мог распознать
        # только дальний буй: ближние услышали его раньше на время пробега между буями — ищем шире.
        spread = float(np.max(np.linalg.norm(buoys_xy[:, None] - buoys_xy[None], axis=-1))) / c
        n0, n1 = int(max(t0 - spread, 0) * sr), int((t1 + spread) * sr)
        peaks = [(np.max(np.abs(s[n0:n1])) if n1 > n0 else 0, int(np.argmax(np.abs(s[n0:n1]))) + n0) for s in signals]
        # опорный — буй с самым громким импульсом: по энергии окна им может оказаться птица у дальнего буя
        ref_buoy = int(np.argmax([p[0] for p in peaks]))
        starts = [peaks[ref_buoy][1] / sr - 0.3 * win]
    else:
        ref_buoy = None
        starts = np.arange(t0, max(t0, t1 - win) + 1e-9, hop)
    fixes = []
    for t in starts:
        w = tdoa_window(signals, buoys_xy, sr, float(t), win, c, band, lcfg["min_psr"], alphas, ref_buoy)
        if w is None:
            continue
        p = solve_position(buoys_xy, w["ref"], w["delays"], c, w["amps"], sigma_t,
                           lcfg["level_sigma"], lcfg["search_radius_m"], lcfg["grid_step_m"],
                           lcfg.get("sound_speed_rel_error", 0.0))
        if p["tdoa_rms_m"] > lcfg["max_tdoa_rms_m"]:
            continue  # гиперболы не сходятся: многолучёвость, два источника, чужой звук
        # момент излучения = приём на опорном буе минус время распространения
        r_ref = float(np.hypot(p["x"] - buoys_xy[w["ref"], 0], p["y"] - buoys_xy[w["ref"], 1]))
        p["t"] = float(t + win / 2 - r_ref / c)
        p["ref"] = w["ref"]
        p["psr_min"] = float(np.min(w["psr"][~np.isnan(w["delays"])]))
        fixes.append(p)
    return fixes


def consistent_fixes(fixes: list[dict], v_max: float = 20.0, horizon: float = 10.0, slack: float = 30.0) -> list[dict]:
    """Отбросить засечки, которые не согласуются с соседями: лодка не может оказаться
    дальше v_max·|dt| + slack. Нужно до Калмана — иначе выброс в начале становится
    стартом трека, и фильтр затем отвергает все верные точки."""
    if len(fixes) < 3:
        return fixes
    t = np.array([f["t"] for f in fixes])
    xy = np.array([[f["x"], f["y"]] for f in fixes])
    keep = []
    for i in range(len(fixes)):
        near = (np.abs(t - t[i]) <= horizon) & (np.arange(len(t)) != i)
        if not near.any():
            continue
        ok = np.linalg.norm(xy[near] - xy[i], axis=1) <= v_max * np.abs(t[near] - t[i]) + slack
        if ok.sum() >= max(1, 0.5 * near.sum()):
            keep.append(fixes[i])
    return keep


def kalman_track(fixes: list[dict], accel_noise: float = 0.5, gate: float = 13.8) -> list[dict]:
    """Постоянная скорость, вперёд + RTS-сглаживание. Возвращает точки с vx, vy и радиусом."""
    fixes = consistent_fixes(sorted(fixes, key=lambda f: f["t"]))
    if not fixes:
        return []
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    x = np.array([fixes[0]["x"], fixes[0]["y"], 0, 0], float)
    P = np.diag([fixes[0]["radius_m"] ** 2, fixes[0]["radius_m"] ** 2, 15.0**2, 15.0**2])
    xs, Ps, xps, Pps, Fs, ts, kept = [], [], [], [], [], [], []
    t_prev = fixes[0]["t"]
    for f in fixes:
        dt = max(f["t"] - t_prev, 1e-3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = accel_noise**2
        Q = q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                          [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])
        xp, Pp = F @ x, F @ P @ F.T + Q
        R = f["cov"] + np.eye(2) * 4.0  # пол 2 м: модель не идеальна
        z = np.array([f["x"], f["y"]])
        S = H @ Pp @ H.T + R
        innov = z - H @ xp
        if len(xs) and innov @ np.linalg.solve(S, innov) > gate:
            continue  # выброс: не обновляем (и не продвигаем время)
        K = Pp @ H.T @ np.linalg.inv(S)
        x, P = xp + K @ innov, (np.eye(4) - K @ H) @ Pp
        xs.append(x), Ps.append(P), xps.append(xp), Pps.append(Pp), Fs.append(F), ts.append(f["t"]), kept.append(f)
        t_prev = f["t"]
    for k in range(len(xs) - 2, -1, -1):  # RTS
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pps[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xps[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pps[k + 1]) @ C.T
    # Систематика (скорость звука, координаты буёв) не усредняется по засечкам, поэтому
    # радиус трека не меньше половины типичного радиуса отдельных засечек рядом по времени.
    kt = np.array([f["t"] for f in kept])
    kr = np.array([f["radius_m"] for f in kept])
    out = []
    for t, st, P in zip(ts, xs, Ps):
        floor = 0.5 * float(np.median(kr[np.abs(kt - t) <= 10]))
        out.append({"t": t, "x": float(st[0]), "y": float(st[1]), "vx": float(st[2]), "vy": float(st[3]),
                    "radius_m": max(float(2 * np.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 0))), floor)})
    return out
