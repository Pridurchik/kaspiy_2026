"""Единая точка входа пайплайна.

  python -m caspian_audio.cli synth                 # синтетические данные для smoke-теста
  python -m caspian_audio.cli download esc50        # реальные данные
  python -m caspian_audio.cli prepare --sources synthetic esc50 folder:/path/shipsear_motorboats=boat_motor
  python -m caspian_audio.cli train
  python -m caspian_audio.cli evaluate
  python -m caspian_audio.cli export
  python -m caspian_audio.cli scene                 # демо-сцена 60 с с разметкой
  python -m caspian_audio.cli detect data/scene.wav # события + алерты в JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio_io import load_audio, load_config


def main():
    ap = argparse.ArgumentParser(prog="caspian_audio")
    ap.add_argument("--config", default="configs/audio.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--per-class", type=int, default=200)
    s.add_argument("--seconds", type=float, default=3.0)

    d = sub.add_parser("download")
    d.add_argument("dataset", choices=["esc50"])

    p = sub.add_parser("prepare")
    p.add_argument("--sources", nargs="+", default=["synthetic"])
    p.add_argument("--max-per-class", type=int)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int)

    e = sub.add_parser("evaluate")
    e.add_argument("--ckpt")
    e.add_argument("--split", default="test")

    x = sub.add_parser("export")
    x.add_argument("--ckpt")

    tu = sub.add_parser("tune", help="подбор порогов алертов на валидации")
    tu.add_argument("--ckpt")
    tu.add_argument("--beta", type=float, default=0.5)

    od = sub.add_parser("ood", help="обучить и оценить детектор незнакомых звуков")
    od.add_argument("--model")
    od.add_argument("--quantile", type=float, default=0.95)

    sc = sub.add_parser("scene")
    sc.add_argument("--out", default="data/scene.wav")

    dt = sub.add_parser("detect")
    dt.add_argument("audio")
    dt.add_argument("--model", help=".onnx или .pt (по умолчанию runs/.../model.onnx или best.pt)")
    dt.add_argument("--sensor-id", default="buoy-01")
    dt.add_argument("--lat", type=float, default=45.75)
    dt.add_argument("--lon", type=float, default=48.40)
    dt.add_argument("--out")

    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = Path(cfg["train"]["out_dir"])
    ckpt = getattr(a, "ckpt", None) or str(run_dir / "best.pt")

    if a.cmd == "synth":
        from .synth import generate_dataset
        generate_dataset(Path(cfg["data"]["raw_dir"]) / "synthetic", a.per_class, a.seconds, cfg["sample_rate"], cfg["train"]["seed"])
    elif a.cmd == "download":
        from .prepare import download_esc50
        download_esc50(cfg["data"]["raw_dir"])
    elif a.cmd == "prepare":
        from .prepare import prepare
        prepare(cfg, a.sources, a.max_per_class)
    elif a.cmd == "train":
        from .train import train
        train(cfg, a.epochs)
    elif a.cmd == "evaluate":
        from .train import evaluate
        evaluate(ckpt, a.split, inference=cfg["inference"])
    elif a.cmd == "export":
        from .export import export_onnx
        export_onnx(ckpt, inference=cfg["inference"])
    elif a.cmd == "tune":
        from .train import tune_thresholds
        tune_thresholds(ckpt)
    elif a.cmd == "ood":
        from .detect import load_predictor
        from .ood import evaluate_ood, fit
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        predict, mcfg = load_predictor(model)
        scorer = fit(predict, mcfg, a.quantile)
        scorer.save(run_dir / "ood.npz")
        print(f"[ood] -> {run_dir / 'ood.npz'}")
        evaluate_ood(predict, mcfg, scorer)
    elif a.cmd == "scene":
        from .synth import generate_scene
        generate_scene(a.out, cfg["sample_rate"])
    elif a.cmd == "detect":
        from .detect import StreamingDetector, load_ood, load_predictor, score_against_truth, to_alerts
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        predict, mcfg = load_predictor(model)
        res = StreamingDetector(predict, mcfg, ood=load_ood(model)).run(load_audio(a.audio, mcfg["sample_rate"]))
        out = {"model": model, "duration_s": res["duration"], "events": res["events"],
               "alerts": to_alerts(res["events"], mcfg, a.sensor_id, a.lat, a.lon)}
        truth_path = Path(a.audio).with_suffix(".truth.json")
        if truth_path.exists():
            out["score_vs_truth"] = score_against_truth(res["events"], json.loads(truth_path.read_text()))
        print("События:")
        for ev in res["events"]:
            print(f"  {ev['start']:6.1f}–{ev['end']:6.1f} c  {ev['class']:11s} conf={ev['confidence']:.2f}")
        if "score_vs_truth" in out:
            print("Сверка с разметкой:", json.dumps(out["score_vs_truth"], ensure_ascii=False))
        if a.out:
            Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"[detect] -> {a.out}")


if __name__ == "__main__":
    main()
