"""Синтетические звуки для smoke-теста пайплайна и демо-сцен.

ВНИМАНИЕ: это не замена реальным записям. Синтетика нужна, чтобы весь
пайплайн (данные -> обучение -> инференс -> алерты) запускался за минуты
без скачивания датасетов. Для реальной модели используйте ESC-50,
UrbanSound8K, ShipsEar и собственные записи с гидрофона.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_io import save_audio


def _fft_filter(x, sr, lo=None, hi=None, order=4):
    """Фильтр с АЧХ Баттерворта, применённый в частотной области (numpy only)."""
    X = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(len(x), 1 / sr), 1e-3)
    g = np.ones_like(f)
    if lo:
        g /= np.sqrt(1 + (lo / f) ** (2 * order))
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, len(x))


def _bandpass(x, lo, hi, sr, order=4):
    return _fft_filter(x, sr, lo=lo, hi=min(hi, sr / 2 * 0.95), order=order)


def _lowpass(x, f, sr, order=4):
    return _fft_filter(x, sr, hi=f, order=order)


def pink_noise(n, rng):
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(white) / np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-9)


def background(seconds, sr, rng):
    """Фон: волны (модулированный низкочастотный шум), ветер, птицы, плеск."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = 0.3 * pink_noise(n, rng)
    waves = _lowpass(rng.standard_normal(n), rng.uniform(300, 900), sr)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.08, 0.3) * t + rng.uniform(0, 6.28))
    x += rng.uniform(0.5, 1.5) * waves / (np.std(waves) + 1e-9) * am
    if rng.random() < 0.6:  # порывы ветра
        wind = _bandpass(rng.standard_normal(n), 100, 1500, sr)
        gust = _lowpass(np.abs(rng.standard_normal(n)), 0.5, sr, order=2)
        x += rng.uniform(0.2, 0.8) * wind / (np.std(wind) + 1e-9) * gust / (gust.max() + 1e-9)
    for _ in range(rng.poisson(1.5)):  # птицы: короткие частотные свипы
        dur = rng.uniform(0.05, 0.3)
        start = rng.uniform(0, max(0.01, seconds - dur))
        i0, m = int(start * sr), int(dur * sr)
        tt = np.arange(m) / sr
        f0, f1 = rng.uniform(2000, 5000), rng.uniform(2000, 6000)
        chirp = np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt**2 / (2 * dur)))
        x[i0 : i0 + m] += rng.uniform(0.2, 0.8) * chirp * np.hanning(m)[: len(x[i0 : i0 + m])]
    return (x / (np.max(np.abs(x)) + 1e-9) * rng.uniform(0.2, 0.7)).astype(np.float32)


def boat_motor(seconds, sr, rng, f0=None):
    """Подвесной мотор: гармоники частоты вспышек (20–90 Гц) + кавитация винта."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = f0 or rng.uniform(20, 90)
    drift = 1 + rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * drift) / sr
    x = np.zeros(n)
    decay = rng.uniform(0.6, 1.2)
    for k in range(1, int(min(60, 3000 / f0))):
        x += (1 / k**decay) * np.sin(k * phase + rng.uniform(0, 6.28))
    x /= np.std(x) + 1e-9
    blade_rate = f0 * rng.uniform(0.3, 0.6)
    cav = _bandpass(rng.standard_normal(n), 500, 6000, sr)
    cav *= 0.6 + 0.4 * np.sin(2 * np.pi * blade_rate * t)
    x += rng.uniform(0.2, 0.8) * cav / (np.std(cav) + 1e-9)
    x *= 0.8 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.1, 1.0) * t)
    return (x / (np.max(np.abs(x)) + 1e-9) * rng.uniform(0.3, 0.9)).astype(np.float32)


def gunshot(seconds, sr, rng, at=None):
    """Выстрел: ударная волна + шумовой всплеск с экспоненциальным затуханием + эхо."""
    n = int(seconds * sr)
    x = np.zeros(n)
    n_shots = 1 if rng.random() < 0.7 else 2
    t0 = at if at is not None else rng.uniform(0.1, max(0.15, seconds - 0.8))
    for s in range(n_shots):
        i0 = int((t0 + s * rng.uniform(0.3, 0.9)) * sr)
        if i0 >= n:
            break
        m = min(int(0.6 * sr), n - i0)
        tt = np.arange(m) / sr
        tau = rng.uniform(0.008, 0.04)
        burst = _bandpass(rng.standard_normal(m), rng.uniform(150, 400), rng.uniform(3000, 7500), sr)
        shot = burst * np.exp(-tt / tau)
        shot += 2.0 * np.sin(2 * np.pi * rng.uniform(60, 150) * tt) * np.exp(-tt / (tau * 2))
        for d, a in [(rng.uniform(0.05, 0.2), 0.3), (rng.uniform(0.2, 0.5), 0.15)]:  # эхо от берега
            j = int(d * sr)
            if j < m:
                shot[j:] += a * shot[: m - j]
        x[i0 : i0 + m] += shot
    return (x / (np.max(np.abs(x)) + 1e-9) * rng.uniform(0.5, 0.95)).astype(np.float32)


GENERATORS = {"background": background, "boat_motor": boat_motor, "gunshot": gunshot}


def generate_dataset(out_dir: str | Path, per_class: int, seconds: float, sr: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for cls, gen in GENERATORS.items():
        for i in range(per_class):
            x = gen(seconds * rng.uniform(1.0, 2.0), sr, rng)
            save_audio(out_dir / cls / f"syn_{cls}_{i:04d}.wav", x, sr)
    print(f"[synth] {per_class} клипов x {len(GENERATORS)} классов -> {out_dir}")


def generate_scene(out_wav: str | Path, sr: int, seed: int = 123) -> dict:
    """60-секундная сцена для демо: лодка подходит и уходит, два выстрела, вторая лодка вдали."""
    rng = np.random.default_rng(seed)
    total = 60.0
    x = background(total, sr, rng) * 0.6
    truth = []

    def place(sig, start, gain, label):
        i0 = int(start * sr)
        env = np.ones(len(sig))
        f = min(len(sig) // 3, int(3 * sr))
        env[:f] = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
        x[i0 : i0 + len(sig)] += gain * sig * (env if label == "boat_motor" else 1)
        truth.append({"class": label, "start": start, "end": round(start + len(sig) / sr, 2)})

    place(boat_motor(15, sr, rng), 8.0, 0.8, "boat_motor")
    place(gunshot(1.0, sr, rng, at=0.05), 34.0, 0.9, "gunshot")
    place(gunshot(1.0, sr, rng, at=0.05), 37.5, 0.7, "gunshot")
    place(boat_motor(10, sr, rng), 46.0, 0.45, "boat_motor")
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
    save_audio(out_wav, x, sr)
    Path(out_wav).with_suffix(".truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2))
    print(f"[scene] {out_wav} ({total:.0f} c), разметка: {Path(out_wav).with_suffix('.truth.json')}")
    return {"path": str(out_wav), "truth": truth}
