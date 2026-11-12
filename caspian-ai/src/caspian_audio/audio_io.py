"""Audio loading/saving and small signal helpers (numpy only, no torch)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT-ресемплинг (numpy only): обрезка/дополнение спектра, антиалиасинг встроен."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x.astype(np.float64))
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    m = min(len(X), len(Y))
    Y[:m] = X[:m]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def _to_mono_resampled(x: np.ndarray, file_sr: int, sr: int) -> np.ndarray:
    x = x.mean(axis=1) if x.ndim == 2 else x
    x = resample(x, int(file_sr), int(sr))
    return np.ascontiguousarray(x, dtype=np.float32)


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return _to_mono_resampled(x, file_sr, sr)


def load_audio_bytes(data: bytes, sr: int) -> np.ndarray:
    x, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return _to_mono_resampled(x, file_sr, sr)


def save_audio(path: str | Path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def list_audio(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def frame_energy(x: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if len(x) < frame:
        return np.array([float(np.mean(x**2))])
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.mean(x[idx] ** 2, axis=1)


def active_rms(x: np.ndarray) -> float:
    """RMS по самым громким 25% кадров — чтобы SNR импульсного выстрела считался честно."""
    e = frame_energy(x)
    k = max(1, len(e) // 4)
    top = np.sort(e)[-k:]
    return float(np.sqrt(np.mean(top) + 1e-12))


def mix_at_snr(event: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    rms_e, rms_b = active_rms(event), float(np.sqrt(np.mean(bg**2) + 1e-12))
    scale = rms_e / (rms_b * 10 ** (snr_db / 20))
    out = event + bg * scale
    peak = np.max(np.abs(out)) + 1e-9
    return out / peak * 0.9 if peak > 0.99 else out
