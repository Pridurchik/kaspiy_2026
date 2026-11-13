"""Модель: сырое аудио -> log-mel (внутри модели) -> CNN -> логиты классов.

Log-mel реализован через conv1d с фиксированными ядрами DFT, поэтому весь граф
экспортируется в ONNX одним файлом: на буе (Raspberry Pi) нужен только onnxruntime.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax) -> np.ndarray:
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]  # нормировка по площади (slaney)
    return fb


class LogMel(nn.Module):
    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax, eps=1e-6):
        super().__init__()
        self.n_fft, self.hop, self.eps = n_fft, hop_length, eps
        self.n_freq = n_fft // 2 + 1
        win = torch.hann_window(n_fft, periodic=True)
        n = torch.arange(n_fft).float()
        k = torch.arange(self.n_freq).float()
        ang = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        kernel = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)[:, None, :]
        self.register_buffer("kernel", kernel, persistent=False)
        self.register_buffer("mel_fb", torch.from_numpy(mel_filterbank(sr, n_fft, n_mels, fmin, fmax)), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, n_mels, frames)
        x = F.pad(wav.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spec = F.conv1d(x, self.kernel, stride=self.hop)
        re, im = spec[:, : self.n_freq], spec[:, self.n_freq :]
        mel = torch.matmul(self.mel_fb, re * re + im * im)
        return torch.log(mel + self.eps)


def spec_augment(x, n_masks=2, max_f=8, max_t=20):
    B, M, T = x.shape
    fidx = torch.arange(M, device=x.device)[None]
    tidx = torch.arange(T, device=x.device)[None]
    for _ in range(n_masks):
        w = torch.randint(0, max_f + 1, (B, 1), device=x.device)
        f0 = (torch.rand(B, 1, device=x.device) * (M - w)).long()
        x = x.masked_fill(((fidx >= f0) & (fidx < f0 + w))[:, :, None], 0.0)
        w = torch.randint(0, max_t + 1, (B, 1), device=x.device)
        t0 = (torch.rand(B, 1, device=x.device) * (T - w).clamp(min=1)).long()
        x = x.masked_fill(((tidx >= t0) & (tidx < t0 + w))[:, None, :], 0.0)
    return x


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.net(x))


class AudioCNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        fe, m = cfg["features"], cfg["model"]
        # "clip_mean": из log-mel вычитается среднее по окну, т.е. модель не зависит от
        # абсолютной громкости (усиление = сдвиг в лог-шкале). Нормализацию по полосам
        # не делаем: она стёрла бы стационарный спектр мотора. Старые чекпоинты без
        # ключа работают как раньше ("none").
        self.norm = fe.get("normalize", "none")
        eps = 1e-10 if self.norm == "clip_mean" else 1e-6  # иначе тихий сигнал упирается в eps
        self.logmel = LogMel(cfg["sample_rate"], fe["n_fft"], fe["hop_length"], fe["n_mels"], fe["fmin"], fe["fmax"], eps)
        self.in_bn = nn.BatchNorm1d(fe["n_mels"])  # нормализация каждой мел-полосы
        chans = m["channels"]
        self.blocks = nn.Sequential(*[
            ConvBlock(1 if i == 0 else chans[i - 1], c, pool=i < len(chans) - 1) for i, c in enumerate(chans)
        ])
        self.head = nn.Sequential(nn.Dropout(m["dropout"]), nn.Linear(chans[-1] * 2, len(cfg["classes"])))
        self.specaug = m.get("specaugment", True)

    def forward(self, wav: torch.Tensor, return_embedding: bool = False):
        x = self.logmel(wav)
        if self.norm == "clip_mean":
            x = x - x.mean(dim=(1, 2), keepdim=True)
        x = self.in_bn(x)
        if self.training and self.specaug:
            x = spec_augment(x)
        x = self.blocks(x.unsqueeze(1)).mean(dim=2)  # (B, C, T')
        emb = torch.cat([x.mean(-1), x.amax(-1)], dim=1)  # max-пулинг по времени ловит короткие импульсы
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


class ExportWrapper(nn.Module):
    """Для ONNX: отдаёт и логиты, и эмбеддинг (нужен OOD-детектору на буе)."""

    def __init__(self, model: "AudioCNN"):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor):
        return self.model(wav, return_embedding=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AudioCNN(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"]
