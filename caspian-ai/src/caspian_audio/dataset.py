"""PyTorch Dataset с аугментациями под полевые условия."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_io import frame_energy, load_audio, mix_at_snr


class AudioClipDataset(Dataset):
    def __init__(self, rows: list[dict], cfg: dict, train: bool, bg_paths: list[str] | None = None):
        self.rows = rows
        self.cfg = cfg
        self.train = train
        self.sr = cfg["sample_rate"]
        self.n = int(cfg["clip_seconds"] * self.sr)
        self.classes = cfg["classes"]
        self.bg_label = cfg["background_class"]
        self.bg_paths = bg_paths or []
        d = cfg["data"]
        self.mix_prob, self.snr_db, self.gain_db = d["mix_prob"], d["snr_db"], d["gain_db"]
        # с seed из конфига, иначе обучение не воспроизводится
        self.rng = np.random.default_rng(cfg["train"]["seed"] + (0 if train else 1))

    def __len__(self):
        return len(self.rows)

    def labels(self) -> np.ndarray:
        return np.array([self.classes.index(r["label"]) for r in self.rows])

    def _crop(self, x: np.ndarray, guided: bool) -> np.ndarray:
        n = self.n
        if len(x) < n:
            out = np.zeros(n, dtype=np.float32)
            off = self.rng.integers(0, n - len(x) + 1) if self.train else (n - len(x)) // 2
            out[off : off + len(x)] = x
            return out
        if len(x) == n:
            return x
        if guided:
            # Окно обязательно содержит самый громкий участок: иначе короткий выстрел
            # в 4-секундном клипе может не попасть в окно, и метка станет ложной.
            hop = 512
            peak = int(np.argmax(frame_energy(x, 1024, hop))) * hop + 512
            rel = self.rng.uniform(0.15, 0.85) if self.train else 0.5
            start = int(np.clip(peak - rel * n, 0, len(x) - n))
        else:
            start = self.rng.integers(0, len(x) - n + 1) if self.train else (len(x) - n) // 2
        return x[start : start + n]

    def _random_bg(self) -> np.ndarray:
        x = load_audio(self.bg_paths[self.rng.integers(len(self.bg_paths))], self.sr)
        return self._crop(x, guided=False)

    def __getitem__(self, i):
        r = self.rows[i]
        label = r["label"]
        is_event = label != self.bg_label
        x = self._crop(load_audio(r["path"], self.sr), guided=is_event)

        if self.train:
            if self.bg_paths and self.rng.random() < (self.mix_prob if is_event else 0.3):
                snr = self.rng.uniform(*self.snr_db) if is_event else self.rng.uniform(0, 10)
                x = mix_at_snr(x, self._random_bg(), snr)
            x = x * 10 ** (self.rng.uniform(*self.gain_db) / 20)
            if self.rng.random() < 0.5:
                x = -x
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), self.classes.index(label)
