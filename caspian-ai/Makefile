PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: smoke real serve edge-demo array-demo demo demo-data buoys

# Весь пайплайн на синтетике за несколько минут
smoke:
	$(PY) -m caspian_audio.cli synth --per-class 200
	$(PY) -m caspian_audio.cli prepare --sources synthetic
	$(PY) -m caspian_audio.cli train --epochs 20
	$(PY) -m caspian_audio.cli evaluate
	$(PY) -m caspian_audio.cli export
	$(PY) -m caspian_audio.cli scene
	$(PY) -m caspian_audio.cli detect data/scene.wav --out runs/audio/scene_detect.json

# Реальные данные: ESC-50 + UrbanSound8K (распаковать в data/raw/UrbanSound8K) + синтетика (только в train).
# US8K ограничен 600 клипами на класс, чтобы городской фон не задавил остальное.
US8K := $(if $(wildcard data/raw/UrbanSound8K/metadata/UrbanSound8K.csv),urbansound8k,)
real:
	$(PY) -m caspian_audio.cli download esc50
	$(PY) -m caspian_audio.cli prepare --max-per-class 600 --sources synthetic esc50 $(US8K) $(EXTRA)
	$(PY) -m caspian_audio.cli train
	$(PY) -m caspian_audio.cli tune
	$(PY) -m caspian_audio.cli evaluate
	$(PY) -m caspian_audio.cli export
	$(PY) -m caspian_audio.cli ood
	$(PY) -m caspian_audio.cli stream-eval

serve:
	MODEL=runs/audio/model.onnx $(PY) -m uvicorn caspian_audio.serve:app --port 8000

edge-demo:
	$(PY) -m caspian_audio.edge --model runs/audio/model.onnx --file data/scene.wav --realtime

# Сеть из 3 буёв: пеленг лодки + «тёмная лодка» (фьюжн с AIS и зонами нереста) -> runs/array_demo/map.html
array-demo:
	$(PY) -m caspian_audio.cli array-sim
	$(PY) -m caspian_audio.cli array data/array_demo --out runs/array_demo

# Платформа целиком: демо-ночь (акустика + вода + спутник + прогноз) и дашборд на http://localhost:8000
demo-data:
	$(PY) -m caspian_audio.cli demo

demo: demo-data
	$(PY) -m uvicorn caspian_audio.inspector:app --port 8000

# Три буя шлют алерты с клипами на запущенный сервер (make demo в другом терминале): каскад + пеленг
buoys:
	$(PY) -m caspian_audio.cli buoys --url http://localhost:8000/api/alerts
