PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: smoke real serve edge-demo

# Весь пайплайн на синтетике за несколько минут
smoke:
	$(PY) -m caspian_audio.cli synth --per-class 200
	$(PY) -m caspian_audio.cli prepare --sources synthetic
	$(PY) -m caspian_audio.cli train --epochs 20
	$(PY) -m caspian_audio.cli evaluate
	$(PY) -m caspian_audio.cli export
	$(PY) -m caspian_audio.cli scene
	$(PY) -m caspian_audio.cli detect data/scene.wav --out runs/audio/scene_detect.json

# Реальные данные: ESC-50 (+ UrbanSound8K, если распакован в data/raw/UrbanSound8K) + синтетика
real:
	$(PY) -m caspian_audio.cli download esc50
	$(PY) -m caspian_audio.cli prepare --sources synthetic esc50 $(EXTRA)
	$(PY) -m caspian_audio.cli train
	$(PY) -m caspian_audio.cli evaluate
	$(PY) -m caspian_audio.cli export

serve:
	MODEL=runs/audio/model.onnx $(PY) -m uvicorn caspian_audio.serve:app --port 8000

edge-demo:
	$(PY) -m caspian_audio.edge --model runs/audio/model.onnx --file data/scene.wav --realtime
