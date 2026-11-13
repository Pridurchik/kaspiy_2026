"""Проверка согласованности артефактов модели (без torch; onnxruntime опционален).

Запуск:  PYTHONPATH=src python scripts/check_artifacts.py

Ловит именно тот класс ошибок, который был найден в аудите: пороги алертов,
ссылающиеся на классы, которых нет в модели, и расхождение размеров выходов ONNX.
Выход с кодом 1 при любой ошибке (годится для CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "audio"

errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    model_json = RUN / "model.json"
    check(model_json.exists(), f"нет {model_json}")
    if errors:
        print("\n".join(errors))
        return 1

    cfg = json.loads(model_json.read_text(encoding="utf-8"))
    classes = cfg["classes"]
    check(len(classes) >= 2, "мало классов")
    check(cfg.get("background_class") in classes, "background_class не входит в classes")

    inf = cfg.get("inference", {})
    for key in ("thresholds", "min_consecutive", "edge_thresholds"):
        for cls in inf.get(key, {}):
            check(cls in classes, f"inference.{key}: класс {cls!r} отсутствует в model.json classes")

    # метрики теста должны быть по тем же классам
    mt = RUN / "metrics_test.json"
    if mt.exists():
        m = json.loads(mt.read_text(encoding="utf-8"))
        check(m.get("classes") == classes, "metrics_test.json classes != model.json classes")
        cm = m.get("confusion_matrix")
        check(isinstance(cm, list) and len(cm) == len(classes), "confusion_matrix: неверный размер")
        check(all(len(row) == len(classes) for row in cm or []), "confusion_matrix: не квадратная")

    # потоковые метрики
    ms = RUN / "metrics_stream.json"
    if ms.exists():
        s = json.loads(ms.read_text(encoding="utf-8"))
        for prof in ("server", "edge"):
            check(prof in s.get("profiles", {}), f"metrics_stream.json: нет профиля {prof}")

    # OOD
    ood = RUN / "ood.npz"
    if ood.exists():
        import numpy as np

        z = np.load(ood, allow_pickle=False)
        check(list(map(str, z["classes"])) == classes, "ood.npz classes != model.json classes")
        check(0.0 <= float(z["threshold"]) <= 1.0, "ood.npz threshold вне [0,1]")

    # ONNX (опционально)
    onnx_path = RUN / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            check(len(inp.shape) == 2 and inp.shape[1] == 32000,
                  f"ONNX input {inp.shape}, ожидалось [batch, 32000]")
            outs = {o.name: o.shape for o in sess.get_outputs()}
            check("logits" in outs and outs["logits"][-1] == len(classes),
                  f"ONNX logits {outs.get('logits')} != {len(classes)} классов")
            check("embedding" in outs, "ONNX без выхода embedding (нужен OOD на буе)")
            print(f"[check] ONNX ok: вход {inp.shape}, выходы {outs}, {onnx_path.stat().st_size/1e6:.2f} МБ")
        except ImportError:
            print("[check] onnxruntime нет — проверка ONNX пропущена")
        except Exception as e:  # noqa: BLE001
            errors.append(f"ONNX не загрузился: {e!r}")

    if errors:
        print("[check] ОШИБКИ:")
        for e in errors:
            print("  -", e)
        return 1
    print(f"[check] артефакты согласованы: {len(classes)} классов, "
          f"{len(inf.get('thresholds', {}))} порогов алертов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
