"""Сборка датасета: источники -> ресемплинг 16 кГц моно -> сплит train/val/test -> manifest.csv.

Поддерживаемые источники:
  synthetic           data/raw/synthetic/<class>/*.wav (из `cli synth`)
  esc50               распакованный ESC-50 (meta/esc50.csv + audio/)
  urbansound8k        распакованный UrbanSound8K (metadata/UrbanSound8K.csv + audio/foldN/)
  folder PATH=CLASS   любая папка с аудио, все файлы получают метку CLASS
                      (например ShipsEar моторки, свои записи с гидрофона)
  folder PATH         подпапки PATH называются как классы из конфига
"""
from __future__ import annotations

import csv
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .audio_io import list_audio, load_audio, save_audio

# Маппинг классов внешних датасетов на наши.
# Всё, что похоже на мотор/выстрел, но им не является, идёт в фон как "трудные негативы".
ESC50_MAP = {
    "engine": "boat_motor",
    # отдельные классы техники: раньше были "трудными негативами" в фоне
    # и давали основную массу ложных тревог по мотору
    "helicopter": "aircraft", "airplane": "aircraft",
    "chainsaw": "chainsaw", "hand_saw": "chainsaw",
    # фон: природа
    "sea_waves": "background", "rain": "background", "wind": "background",
    "water_drops": "background", "pouring_water": "background", "thunderstorm": "background",
    "crickets": "background", "insects": "background", "chirping_birds": "background",
    "frog": "background", "crow": "background", "crackling_fire": "background",
    # отдельный класс-буфер без алерта: реальный взрывной звук, похожий на выстрел
    "fireworks": "fireworks",
    # трудные негативы
    "door_wood_knock": "background", "train": "background",
}
# Категории ESC-50, которые есть и в UrbanSound8K (dog_bark, siren, car_horn): если обучались
# на US8K, они уже не «незнакомые», поэтому из оценки OOD/ложных тревог их исключаем всегда.
ESC50_US8K_OVERLAP = {"dog", "siren", "car_horn"}


def esc50_unknown_categories(esc_root) -> dict[str, list[Path]]:
    """Реальные звуки, которых нет в обучении: категории ESC-50 вне ESC50_MAP и вне US8K."""
    by_cat: dict[str, list[Path]] = {}
    with open(Path(esc_root) / "meta" / "esc50.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["category"] not in ESC50_MAP and r["category"] not in ESC50_US8K_OVERLAP:
                by_cat.setdefault(r["category"], []).append(Path(esc_root) / "audio" / r["filename"])
    return by_cat
US8K_MAP = {
    "gun_shot": "gunshot",
    "engine_idling": "boat_motor",
    "air_conditioner": "background", "car_horn": "background", "children_playing": "background",
    "dog_bark": "background", "drilling": "background", "jackhammer": "background",
    "siren": "background", "street_music": "background",
}
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def download_esc50(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath, root = raw_dir / "ESC-50-master.zip", raw_dir / "ESC-50-master"
    if root.exists():
        print(f"[download] уже есть: {root}")
        return root
    if not zpath.exists():  # архив можно скачать вручную и положить в data/raw/ESC-50-master.zip
        print(f"[download] {ESC50_URL} (~600 МБ)...")
        urllib.request.urlretrieve(ESC50_URL, zpath)
    else:
        print(f"[download] распаковываю {zpath}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_dir)
    zpath.unlink()
    print(f"[download] готово: {root}")
    return root


# Каждый источник возвращает список (path, label, source, group).
# group — id исходной записи, чтобы куски одной записи не попали и в train, и в test.

def collect_folder_by_class(root: Path, classes: list[str], source: str):
    items = []
    for cls in classes:
        for p in list_audio(root / cls):
            items.append((p, cls, source, f"{source}:{p.stem}"))
    return items


def collect_folder_label(root: Path, label: str, source: str):
    return [(p, label, source, f"{source}:{p.parent.name}/{p.stem}") for p in list_audio(root)]


def collect_esc50(root: Path):
    items = []
    with open(root / "meta" / "esc50.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = ESC50_MAP.get(r["category"])
            if label:
                items.append((root / "audio" / r["filename"], label, "esc50", f"esc50:{r['src_file']}"))
    return items


def collect_us8k(root: Path):
    items = []
    with open(root / "metadata" / "UrbanSound8K.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = US8K_MAP.get(r["class"])
            if label:
                p = root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
                items.append((p, label, "us8k", f"us8k:{r['fsID']}"))
    return items


def _limit_per_class(items, max_per_class, rng):
    if not max_per_class:
        return items
    out, by_key = [], {}
    for it in items:
        by_key.setdefault((it[2], it[1]), []).append(it)
    for group in by_key.values():
        idx = rng.permutation(len(group))[:max_per_class]
        out.extend(group[i] for i in idx)
    return out


def _split(labels, groups, frac, seed):
    """Стратифицированный групповой сплит: маска выделенной доли.

    Для каждого класса перемешиваем группы (исходные записи) и забираем их целиком,
    пока не наберём долю `frac` клипов этого класса.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    group_label = {}
    for g in np.unique(groups):  # группа относится к своему мажоритарному классу
        vals, cnt = np.unique(labels[groups == g], return_counts=True)
        group_label[g] = vals[np.argmax(cnt)]
    for lab in np.unique(labels):
        target = frac * int((labels == lab).sum())
        gs = [g for g, l in group_label.items() if l == lab]
        taken = 0
        for g in rng.permutation(gs):
            if taken >= target:
                break
            idx = groups == g
            mask |= idx
            taken += int((labels[idx] == lab).sum())
    return mask


def prepare(cfg: dict, sources: list[str], max_per_class: int | None = None) -> Path:
    sr, classes = cfg["sample_rate"], cfg["classes"]
    dcfg = cfg["data"]
    raw, proc = Path(dcfg["raw_dir"]), Path(dcfg["processed_dir"])
    rng = np.random.default_rng(cfg["train"]["seed"])

    items = []
    for s in sources:
        if s == "synthetic":
            items += collect_folder_by_class(raw / "synthetic", classes, "synthetic")
        elif s == "esc50":
            items += collect_esc50(raw / "ESC-50-master")
        elif s == "urbansound8k":
            items += collect_us8k(raw / "UrbanSound8K")
        elif s.startswith("folder:"):
            spec = s[len("folder:"):]
            if "=" in spec:
                path, label = spec.rsplit("=", 1)
                if label not in classes:
                    raise ValueError(f"класс {label} не в конфиге: {classes}")
                items += collect_folder_label(Path(path), label, Path(path).name)
            else:
                items += collect_folder_by_class(Path(spec), classes, Path(spec).name)
        else:
            raise ValueError(f"неизвестный источник: {s}")
    items = _limit_per_class(items, max_per_class, rng)
    if not items:
        raise SystemExit("Нет данных. Сначала: `cli synth` или `cli download esc50`.")

    rows = []
    for i, (src, label, source, group) in enumerate(items):
        dst = proc / source / label / f"{Path(src).stem}.wav"
        if not dst.exists():
            try:
                x = load_audio(src, sr)
            except Exception as e:  # битые файлы встречаются в публичных датасетах
                print(f"[prepare] пропуск {src}: {e}")
                continue
            if len(x) < sr * 0.2 or np.max(np.abs(x)) < 1e-4:
                continue  # слишком короткий или тишина
            save_audio(dst, x, sr)
        rows.append({"path": str(dst), "label": label, "source": source, "group": group})
        if (i + 1) % 500 == 0:
            print(f"[prepare] {i + 1}/{len(items)}")

    split = np.array(["train"] * len(rows), dtype=object)
    # Источники из train_only_sources (синтетика) целиком в train: val/test только реальные,
    # иначе метрики меряют, как модель узнаёт собственный генератор.
    train_only = set(dcfg.get("train_only_sources", []))
    eval_idx = np.array([i for i, r in enumerate(rows) if r["source"] not in train_only], dtype=int)
    if len(eval_idx) == 0:  # только синтетика (make smoke): делим её, метрики — лишь проверка пайплайна
        print("[prepare] ВНИМАНИЕ: реальных записей нет, val/test из синтетики — метрики не отражают реальное качество")
        eval_idx = np.arange(len(rows))
    labels = np.array([rows[i]["label"] for i in eval_idx])
    groups = np.array([rows[i]["group"] for i in eval_idx])
    test = _split(labels, groups, dcfg["test_fraction"], cfg["train"]["seed"])
    split[eval_idx[test]] = "test"
    rest = np.where(~test)[0]
    val_frac = dcfg["val_fraction"] / (1 - dcfg["test_fraction"])
    val = _split(labels[rest], groups[rest], val_frac, cfg["train"]["seed"] + 1)
    split[eval_idx[rest[val]]] = "val"
    for r, s in zip(rows, split):
        r["split"] = s

    manifest = Path(dcfg["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "group", "split"])
        w.writeheader()
        w.writerows(rows)
    stats = Counter((r["split"], r["label"]) for r in rows)
    print(f"[prepare] manifest: {manifest} ({len(rows)} клипов)")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:5s}: " + ", ".join(f"{c}={stats[(sp, c)]}" for c in classes))
    return manifest


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
