"""Обучение и оценка."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import AudioClipDataset
from .metrics import confusion_matrix, format_report, macro_f1, per_class
from .model import AudioCNN, load_checkpoint, pick_device
from .prepare import read_manifest


def _loaders(cfg, splits=("train", "val")):
    rows = read_manifest(cfg["data"]["manifest"])
    bg = [r["path"] for r in rows if r["split"] == "train" and r["label"] == cfg["background_class"]]
    out = {}
    for sp in splits:
        ds = AudioClipDataset([r for r in rows if r["split"] == sp], cfg, train=(sp == "train"), bg_paths=bg)
        out[sp] = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=(sp == "train"),
                             num_workers=cfg["train"]["num_workers"], drop_last=(sp == "train" and len(ds) > cfg["train"]["batch_size"]))
    return out


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)), -1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(cfg: dict, epochs: int | None = None) -> Path:
    tcfg = cfg["train"]
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    device = pick_device()
    loaders = _loaders(cfg)
    y_train = loaders["train"].dataset.labels()
    counts = np.bincount(y_train, minlength=len(cfg["classes"]))
    weights = torch.tensor(len(y_train) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    print(f"[train] device={device}, train={len(y_train)}, val={len(loaders['val'].dataset)}, "
          f"class counts={dict(zip(cfg['classes'], counts.tolist()))}")

    model = AudioCNN(cfg).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    epochs = epochs or tcfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=epochs * len(loaders["train"]))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_loss, best_ep, history = -1.0, float("inf"), 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        yv, pv = predict_loader(model, loaders["val"], device)
        f1 = macro_f1(yv, pv.argmax(1), len(cfg["classes"]))
        vloss = float(-np.mean(np.log(pv[np.arange(len(yv)), yv] + 1e-9)))
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_macro_f1": f1, "val_loss": vloss})
        mark = ""
        # При равном F1 (на лёгкой валидации он быстро упирается в 1.0) берём модель
        # с меньшим val loss: она увереннее и лучше откалибрована для порогов алертов.
        if f1 > best + 1e-9 or (abs(f1 - best) <= 1e-9 and vloss < best_loss):
            best, best_loss, best_ep, mark = f1, vloss, ep, " *"
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": ep,
                        "val_macro_f1": f1, "val_loss": vloss}, out_dir / "best.pt")
        print(f"[train] ep {ep:3d}  loss {tot / max(nb, 1):.4f}  val macro-F1 {f1:.4f}  val loss {vloss:.4f}  ({time.time() - t0:.1f}s){mark}")
        if ep - best_ep >= tcfg["patience"]:
            print(f"[train] early stopping (лучшая эпоха {best_ep})")
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] лучший val macro-F1={best:.4f} -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def _alert_metrics(y, p, classes, thresholds) -> dict:
    """Метрики "как у алерта": порог из конфига, одно событие против всего остального."""
    alert = {}
    for cls, thr in thresholds.items():
        k = classes.index(cls)
        fired, true = p[:, k] >= thr, y == k
        tp = int((fired & true).sum())
        alert[cls] = {"threshold": thr, "precision": tp / max(int(fired.sum()), 1),
                      "recall": tp / max(int(true.sum()), 1), "false_alarms": int((fired & ~true).sum()),
                      "n_positive": int(true.sum())}
    return alert


def _print_alerts(alert: dict):
    for cls, m in alert.items():
        print(f"  {cls:12s} thr={m['threshold']:.2f}  precision={m['precision']:.3f}  recall={m['recall']:.3f}"
              f"  ложных={m['false_alarms']}  (положительных: {m['n_positive']})")


def evaluate(ckpt_path: str | Path, split: str = "test", inference: dict | None = None) -> dict:
    device = pick_device()
    model, cfg = load_checkpoint(ckpt_path, device)
    if inference:  # пороги алертов из актуального YAML, а не из конфига времён обучения
        cfg["inference"] = inference
    loader = _loaders(cfg, splits=(split,))[split]
    y, p = predict_loader(model, loader, device)
    classes = cfg["classes"]
    pred = p.argmax(1)
    rep = per_class(y, pred, classes)
    cm = confusion_matrix(y, pred, len(classes))
    print(format_report(rep, classes))
    print("Confusion matrix (строки — истина, столбцы — предсказание):")
    print("            " + " ".join(f"{c[:10]:>10s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c[:10]:>10s}  " + " ".join(f"{v:10d}" for v in row))

    alert = _alert_metrics(y, p, classes, cfg["inference"]["thresholds"])
    print("Алерты при порогах из конфига (все клипы):")
    _print_alerts(alert)
    metrics = {"split": split, "report": rep, "confusion_matrix": cm.tolist(), "classes": classes, "alerts": alert}

    # Главная цифра — только реальные записи: синтетику модель узнаёт по «почерку» генератора.
    train_only = set(cfg["data"].get("train_only_sources", ["synthetic"]))
    real = np.array([r["source"] not in train_only for r in loader.dataset.rows])
    if real.any() and not real.all():
        yr, pr = y[real], p[real]
        rep_r = per_class(yr, pr.argmax(1), classes)
        metrics["real_only"] = {"n": int(real.sum()), "report": rep_r,
                                "confusion_matrix": confusion_matrix(yr, pr.argmax(1), len(classes)).tolist(),
                                "alerts": _alert_metrics(yr, pr, classes, cfg["inference"]["thresholds"])}
        print(f"Только реальные клипы ({int(real.sum())} из {len(real)}):")
        print(format_report(rep_r, classes))
        _print_alerts(metrics["real_only"]["alerts"])
    elif real.all():
        print("(синтетики в выборке нет — все метрики выше по реальным записям)")

    out = Path(ckpt_path).parent / f"metrics_{split}.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[eval] -> {out}")
    return metrics


def tune_thresholds(ckpt_path: str | Path, beta: float = 0.5) -> dict:
    """Подбор порогов алертов на валидации по F-beta (beta<1 — приоритет у precision:
    инспектору важнее мало ложных тревог). Пороги печатаются, в конфиг их переносим руками."""
    device = pick_device()
    model, cfg = load_checkpoint(ckpt_path, device)
    loader = _loaders(cfg, splits=("val",))["val"]
    y, p = predict_loader(model, loader, device)
    classes = cfg["classes"]
    best = {}
    print(f"[tune] пороги по F{beta} на валидации ({len(y)} клипов):")
    for cls in cfg["inference"]["thresholds"]:
        k = classes.index(cls)
        true = y == k
        rows = []
        for thr in np.arange(0.20, 0.96, 0.05):
            fired = p[:, k] >= thr
            tp = int((fired & true).sum())
            prec = tp / max(int(fired.sum()), 1)
            rec = tp / max(int(true.sum()), 1)
            fb = (1 + beta**2) * prec * rec / max(beta**2 * prec + rec, 1e-9)
            rows.append((fb, thr, prec, rec))
        fb, thr, prec, rec = max(rows)
        best[cls] = {"threshold": round(float(thr), 2), "f_beta": round(fb, 3),
                     "precision": round(prec, 3), "recall": round(rec, 3)}
        cur = cfg["inference"]["thresholds"][cls]
        print(f"  {cls:12s} сейчас {cur:.2f} -> лучший {thr:.2f}  (precision={prec:.3f}, recall={rec:.3f})")
    name = "thresholds_suggested.json" if beta == 0.5 else f"thresholds_suggested_f{beta:g}.json"
    out = Path(ckpt_path).parent / name
    out.write_text(json.dumps(best, indent=2))
    print(f"[tune] -> {out}")
    return best
