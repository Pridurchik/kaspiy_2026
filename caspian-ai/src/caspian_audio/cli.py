"""Единая точка входа пайплайна.

  python -m caspian_audio.cli synth                 # синтетические данные для smoke-теста
  python -m caspian_audio.cli download esc50        # реальные данные
  python -m caspian_audio.cli prepare --sources synthetic esc50 folder:/path/shipsear_motorboats=boat_motor
  python -m caspian_audio.cli train
  python -m caspian_audio.cli evaluate              # клиповые метрики, отдельно — только реальные клипы
  python -m caspian_audio.cli stream-eval           # тревоги в час и recall событий в потоке (реальные записи)
  python -m caspian_audio.cli export
  python -m caspian_audio.cli scene                 # демо-сцена 60 с с разметкой
  python -m caspian_audio.cli detect data/scene.wav # события + алерты в JSON
  python -m caspian_audio.cli detect rec.wav --ais ais.csv   # + опознание по AIS, зоны, балл риска
  python -m caspian_audio.cli array-sim             # демо: 3 буя, теплоход с AIS, «тёмная лодка», выстрел
  python -m caspian_audio.cli array data/array_demo # пеленг + трек + фьюжн -> алерты и карта HTML
  python -m caspian_audio.cli forecast              # прогноз нереста по гидрологии (CSV или демо)
  python -m caspian_audio.cli demo                  # собрать демо-ночь для дашборда (make demo)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .audio_io import load_audio, load_config


def validate_config(cfg: dict) -> None:
    """Ранняя проверка конфига: пороги алертов обязаны ссылаться на существующие классы.

    Именно эта ошибка (порог для класса, которого нет в `classes`) ломает `evaluate`/`tune`
    уже после долгого обучения — лучше упасть сразу и с понятным сообщением.
    """
    classes = list(cfg.get("classes", []))
    if not classes:
        raise SystemExit("config: пустой список classes")
    inf = cfg.get("inference", {})
    for key in ("thresholds", "min_consecutive", "edge_thresholds"):
        bad = [c for c in inf.get(key, {}) if c not in classes]
        if bad:
            raise SystemExit(f"config: inference.{key} ссылается на классы вне classes: {bad}; classes={classes}")
    if cfg.get("background_class") not in classes:
        raise SystemExit(f"config: background_class={cfg.get('background_class')!r} нет в classes")
    cal = inf.get("confidence_calibration")
    if cal is not None and float(cal.get("a", 0.0)) <= 0.0:
        raise SystemExit(f"config: inference.confidence_calibration.a={cal.get('a')!r} должен быть > 0")


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

    cb = sub.add_parser("calibrate", help="калибровка уверенности (Platt) на валидации: ECE до/после")
    cb.add_argument("--ckpt")
    cb.add_argument("--split", default="test", help="контрольный сплит для отчёта ECE (по умолчанию test)")

    od = sub.add_parser("ood", help="обучить и оценить детектор незнакомых звуков")
    od.add_argument("--model")
    od.add_argument("--quantile", type=float, default=0.95)

    se = sub.add_parser("stream-eval", help="ложные тревоги в час и recall событий в потоке, только реальные записи")
    se.add_argument("--model")
    se.add_argument("--out")

    sc = sub.add_parser("scene")
    sc.add_argument("--out", default="data/scene.wav")

    dt = sub.add_parser("detect")
    dt.add_argument("audio")
    dt.add_argument("--model", help=".onnx или .pt (по умолчанию runs/.../model.onnx или best.pt)")
    dt.add_argument("--sensor-id", default="buoy-01")
    dt.add_argument("--lat", type=float, default=45.75)
    dt.add_argument("--lon", type=float, default=48.40)
    dt.add_argument("--out")
    dt.add_argument("--ais", help="CSV AIS (mmsi, ts, lat, lon, name): опознание судов, «тёмная лодка»")
    dt.add_argument("--zones", help="GeoJSON зон (по умолчанию fusion.zones из конфига)")
    dt.add_argument("--recorded-at", help="ISO-время начала записи (для AIS и «ночь/сезон»)")

    asim = sub.add_parser("array-sim", help="демо-сцена для сети из 3 буёв")
    asim.add_argument("--out", default="data/array_demo")
    asim.add_argument("--ideal", action="store_true", help="без полевых эффектов (эхо, ошибки GPS/часов/скорости звука)")

    dm = sub.add_parser("demo", help="собрать демо-ночь для дашборда: акустика, вода, спутник, прогноз")
    dm.add_argument("--model")
    dm.add_argument("--fresh", action="store_true", help="пересоздать симуляции")

    bu = sub.add_parser("buoys", help="3 клиента буя проигрывают демо-сцену и шлют алерты с клипами на сервер")
    bu.add_argument("--url", default="http://localhost:8000/api/alerts")
    bu.add_argument("--dir", default="data/demo/array")
    bu.add_argument("--realtime", action="store_true")

    fo = sub.add_parser("forecast", help="прогноз нереста по гидрологии (CSV или демо-ряд)")
    fo.add_argument("--hydro", help="CSV: date, water_temp_c, discharge_m3s")
    fo.add_argument("--plan", help="CSV графика попуска: date, discharge_m3s")
    fo.add_argument("--today", default="2026-05-20")

    ar = sub.add_parser("array", help="пеленг по сети буёв + фьюжн с AIS и зонами")
    ar.add_argument("dir", nargs="?", default="data/array_demo")
    ar.add_argument("--model")
    ar.add_argument("--ais")
    ar.add_argument("--zones")
    ar.add_argument("--out", default="runs/array_demo")

    a = ap.parse_args()
    cfg = load_config(a.config)
    validate_config(cfg)
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
        tune_thresholds(ckpt, a.beta)
    elif a.cmd == "calibrate":
        from .train import calibrate_confidence
        calibrate_confidence(ckpt, a.split)
    elif a.cmd == "ood":
        from .detect import load_predictor
        from .ood import evaluate_ood, fit
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        predict, mcfg = load_predictor(model)
        scorer = fit(predict, mcfg, a.quantile)
        scorer.save(run_dir / "ood.npz")
        print(f"[ood] -> {run_dir / 'ood.npz'}")
        evaluate_ood(predict, mcfg, scorer)
    elif a.cmd == "stream-eval":
        from .stream_eval import stream_eval
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        stream_eval(model, inference=cfg["inference"], out=a.out)
    elif a.cmd == "scene":
        from .synth import generate_scene
        generate_scene(a.out, cfg["sample_rate"])
    elif a.cmd == "demo":
        from .demo import build
        from .store import Store
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        build(cfg, model, Store("data/inspector.db"), fresh=a.fresh)
    elif a.cmd == "buoys":
        import subprocess
        import sys
        meta = json.loads((Path(a.dir) / "array.json").read_text(encoding="utf-8"))
        procs = [subprocess.Popen([sys.executable, "-m", "caspian_audio.edge", "--file", str(Path(a.dir) / b["file"]),
                                   "--sensor-id", b["id"], "--lat", str(b["lat"]), "--lon", str(b["lon"]),
                                   "--start-time", meta["recorded_at"], "--post", a.url, "--save-clips", ""]
                                  + (["--realtime"] if a.realtime else []), stdout=subprocess.DEVNULL)
                 for b in meta["buoys"]]
        print(f"[buoys] {len(procs)} буя проигрывают {a.dir} -> {a.url}")
        for p in procs:
            p.wait()
        print("[buoys] готово")
    elif a.cmd == "forecast":
        from datetime import date
        from . import forecast as F
        today = date.fromisoformat(a.today)
        hydro = F.load_hydrology_csv(a.hydro, a.plan) if a.hydro else F.simulate_hydrology(today)
        fc = F.forecast(hydro, today)
        print(f"Прогноз нереста на {a.today} (+{fc['horizon_days']} дн), аномалия температуры воды {fc['temp_anomaly_c']:+.1f} °C:")
        for sp in fc["species"].values():
            print(f"  {sp['name']:14s} {sp['status']}")
    elif a.cmd == "array-sim":
        from .array_sim import simulate
        simulate(a.out, cfg, ideal=a.ideal)
    elif a.cmd == "array":
        from .array import run_array
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        run_array(model, a.dir, cfg, a.ais, a.zones, a.out)
    elif a.cmd == "detect":
        from .detect import StreamingDetector, load_ood, load_predictor, score_against_truth, to_alerts
        model = a.model or (str(run_dir / "model.onnx") if (run_dir / "model.onnx").exists() else ckpt)
        predict, mcfg = load_predictor(model)
        res = StreamingDetector(predict, mcfg, ood=load_ood(model)).run(load_audio(a.audio, mcfg["sample_rate"]))
        t0 = datetime.fromisoformat(a.recorded_at) if a.recorded_at else None
        out = {"model": model, "duration_s": res["duration"], "events": res["events"],
               "alerts": to_alerts(res["events"], mcfg, a.sensor_id, a.lat, a.lon, t0)}
        if a.ais or a.zones:
            from .fusion import Fusion, load_ais_csv, summary_line
            fu = Fusion(cfg, load_ais_csv(a.ais) if a.ais else None, a.zones)
            out["alerts"] = [fu.assess(al) for al in out["alerts"]]
            for al in out["alerts"]:
                print("  ", summary_line(al), "|", "; ".join(r["text"] for r in al["fusion"]["reasons"]))
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
