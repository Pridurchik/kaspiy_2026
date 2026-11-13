"""Гидроакустический профиль судна по клипу: тип двигателя и направление движения.

Шум мотора — гармоники частоты вспышек в цилиндрах (f0) плюс широкополосная кавитация винта:
  * подвесной двухтактный мотор маломерной лодки (типичный для браконьеров): f0 ~ 40–100 Гц,
    сильная кавитация на высоких оборотах;
  * стационарный двигатель катера: f0 ~ 18–40 Гц;
  * крупное судно с низкооборотным дизелем: f0 < 18 Гц, мощные низкие тона.
Направление по одному буя: рост уровня — приближается, спад — удаляется; гармоники «съезжают»
вниз при проходе мимо (эффект Доплера). С сетью буёв курс точнее даёт пеленг (localize.py).

ЭТО ЭВРИСТИКА, а не обученный классификатор: пороги ориентировочные, тип — подсказка инспектору.
Обученный классификатор типа судна (как в работах по подводной акустике, напр. ShipsEar, DeepShip)
— следующий шаг, когда будут размеченные записи с гидрофонов.
"""
from __future__ import annotations

import numpy as np

TYPES = [
    (40, "маломерная лодка, подвесной мотор", "small boat, outboard engine"),
    (18, "катер со стационарным двигателем", "motor boat, inboard engine"),
    (0, "крупное судно, низкооборотный дизель", "large vessel, slow-speed diesel"),
]


def _spectrum(x: np.ndarray, sr: int, n_fft: int = 16384):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    hop = n_fft // 2
    frames = [x[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft + 1, hop)]
    S = np.mean([np.abs(np.fft.rfft(f)) for f in frames], axis=0)
    return np.fft.rfftfreq(n_fft, 1 / sr), S


def f0_estimate(x: np.ndarray, sr: int, fmin: float = 8, fmax: float = 150, n_harm: int = 8) -> tuple[float, float]:
    """Частота вспышек по сумме гармоник (лог-спектр минус локальный фон). -> (f0, выраженность)."""
    f, S = _spectrum(x, sr)
    L = np.log(S + 1e-9)
    k = 25
    bg = np.convolve(L, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    E = np.maximum(L - bg, 0)
    cands = np.arange(fmin, fmax, 0.25)
    # сумма на гармониках минус сумма между ними: у ложного f0/2, f0/3 промежуточные позиции пустые,
    # у истинного f0 пустые именно «полугармоники» — так исключаются субгармонические ошибки
    on = np.array([np.mean([np.interp(h * c, f, E) for h in range(1, n_harm + 1)]) for c in cands])
    off = np.array([np.mean([np.interp((h - 0.5) * c, f, E) for h in range(1, n_harm + 1)]) for c in cands])
    score = on - off
    i = int(np.argmax(score))
    return float(cands[i]), float(score[i])


def profile(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, float)
    f0, strength = f0_estimate(x, sr)
    f, S = _spectrum(x, sr)
    lo = S[(f >= 20) & (f < 500)].mean()
    hi = S[(f >= 2000) & (f < 6000)].mean()
    cav = float(20 * np.log10(hi / (lo + 1e-12) + 1e-12))
    typ = next(t for t in TYPES if f0 >= t[0])
    # уровень по полусекундам -> тренд, дБ/с
    n = sr // 2
    lev = np.array([10 * np.log10(np.mean(x[i:i + n] ** 2) + 1e-12) for i in range(0, len(x) - n + 1, n)])
    slope = float(np.polyfit(np.arange(len(lev)) * 0.5, lev, 1)[0]) if len(lev) >= 3 else 0.0
    h = len(x) // 2
    f_a, f_b = f0_estimate(x[:h], sr)[0], f0_estimate(x[h:], sr)[0]
    doppler = (f_b - f_a) / max(f0, 1e-6)
    if slope > 0.8:
        move = ("приближается", "approaching")
    elif slope < -0.8 or doppler < -0.02:
        move = ("удаляется", "receding")
    else:
        move = ("держится рядом / проходит мимо", "loitering / passing")
    return {"type_ru": typ[1], "type_en": typ[2], "f0_hz": round(f0, 1), "harmonic_strength": round(strength, 2),
            "cavitation_db": round(cav, 1), "level_trend_db_s": round(slope, 2), "doppler_rel": round(doppler, 3),
            "movement_ru": move[0], "movement_en": move[1], "method": "эвристика по гармоникам (не обученный классификатор)"}
