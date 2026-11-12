"""Экспорт в ONNX для edge-устройства (буй на Raspberry Pi / Jetson)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import ExportWrapper, load_checkpoint


def export_onnx(ckpt_path: str | Path, out_path: str | Path | None = None, inference: dict | None = None) -> Path:
    model, cfg = load_checkpoint(ckpt_path, "cpu")
    if inference:  # пороги/настройки алертов берём из актуального YAML, а не из конфига времён обучения
        cfg["inference"] = inference
    wrapper = ExportWrapper(model).eval()
    out_path = Path(out_path or Path(ckpt_path).with_name("model.onnx"))
    n = int(cfg["inference"]["window_seconds"] * cfg["sample_rate"])
    dummy = torch.randn(2, n) * 0.1
    kwargs = dict(input_names=["waveform"], output_names=["logits", "embedding"], opset_version=17,
                  dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}})
    try:
        torch.onnx.export(wrapper, (dummy,), str(out_path), dynamo=False, **kwargs)
    except TypeError:  # старые версии torch без аргумента dynamo
        torch.onnx.export(wrapper, (dummy,), str(out_path), **kwargs)
    out_path.with_suffix(".json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = (np.random.randn(3, n) * 0.1).astype(np.float32)
    with torch.no_grad():
        ref_logits, ref_emb = model(torch.from_numpy(x), return_embedding=True)
    got_logits, got_emb = sess.run(None, {"waveform": x})
    diff = max(float(np.abs(ref_logits.numpy() - got_logits).max()), float(np.abs(ref_emb.numpy() - got_emb).max()))
    size = out_path.stat().st_size / 1e6
    print(f"[export] {out_path} ({size:.1f} МБ, выходы: logits+embedding), max |torch - onnx| = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("ONNX расходится с torch")
    return out_path
