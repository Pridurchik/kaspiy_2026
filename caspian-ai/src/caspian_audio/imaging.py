"""PNG без Pillow: слои для карты и спектрограммы алертов."""
from __future__ import annotations

import struct
import zlib

import numpy as np

# опорные точки палитры (тёмно-синий -> бирюзовый -> жёлтый), как в научных графиках
_CMAP = np.array([[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104],
                  [244, 136, 73], [254, 188, 43], [240, 249, 33]], float)


def colormap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1) * (len(_CMAP) - 1)
    i = np.minimum(v.astype(int), len(_CMAP) - 2)
    f = (v - i)[..., None]
    return (_CMAP[i] * (1 - f) + _CMAP[i + 1] * f).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    """img: (H, W, 3) RGB или (H, W, 4) RGBA, uint8."""
    H, W, C = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(H))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    color = 6 if C == 4 else 2
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def spectrogram_png(x: np.ndarray, sr: int, fmax: float = 8000, height: int = 160, n_fft: int = 1024, hop: int = 256) -> bytes:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(x, (n, n_fft), (x.strides[0] * hop, x.strides[0]))
    S = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    S = S[:, f <= fmax]
    db = 20 * np.log10(S + 1e-6)
    db = (db - (db.max() - 70)) / 70  # динамический диапазон 70 дБ
    # частоты в логарифмическом масштабе: низкие гармоники мотора видны лучше
    fr = f[f <= fmax]
    target = np.geomspace(30, fmax, height)
    idx = np.clip(np.searchsorted(fr, target), 0, len(fr) - 1)
    img = colormap(db[:, idx].T[::-1])
    return png_bytes(np.ascontiguousarray(img))
