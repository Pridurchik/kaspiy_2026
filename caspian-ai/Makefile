PY ?= .venv/bin/python
export PYTHONPATH := src
SMOKE_CFG ?= configs/audio_smoke.yaml

.PHONY: help smoke real test check stream-eval serve edge-demo array-demo demo demo-data buoys

help:
	@echo "make smoke       — весь пайплайн на синтетике, ИЗОЛИРОВАННО (data/smoke, runs/smoke)"
	@echo "make real        — реальные данные (ESC-50 + UrbanSound8K + синтетика в train) -> runs/audio"
	@echo "make test        — быстрые проверки логики (tests/test_core.py, без torch)"
	@echo "make check       — согласованность артефактов модели (classes/пороги/ONNX/OOD)"
	@echo "make stream-eval — ложные тревоги в час и recall событий в потоке (реальные записи)"
	@echo "make demo        — демо-ночь платформы и дашборд на http://localhost:8000"
	@echo "make serve       — только акустический API (FastAPI)"
	@echo "make array-demo  — сеть из 3 буёв: пеленг + «тёмная лодка» -> runs/array_demo/map.html"
	@echo "make buoys       — 3 буя шлют алерты с клипами на запущенный сервер"

# Весь пайплайн на синтетике за несколько минут.
# ВАЖНО: пишет в data/smoke/ и runs/smoke/ и НЕ перезаписывает боевые data/manifest.csv и runs/audio.
smoke:
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) synth --per-class 200
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) prepare --sources synthetic
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) train --epochs 20
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) evaluate
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) export
	$(PY) -m caspian_audio.cli scene --out data/smoke/scene.wav
	$(PY) -m caspian_audio.cli --config $(SMOKE_CFG) detect data/smoke/scene.wav --out runs/smoke/scene_detect.json

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

test:
	$(PY) tests/test_core.py

check:
	$(PY) scripts/check_artifacts.py

stream-eval:
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
