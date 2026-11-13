"""Быстрые проверки ключевой логики (без модели и датасетов, ~10 с).

  PYTHONPATH=src .venv/bin/python tests/test_core.py      # или pytest tests/
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caspian_audio.audio_io import load_config  # noqa: E402

CFG = load_config(ROOT / "configs" / "audio.yaml")
CFG["fusion"]["zones"] = str(ROOT / CFG["fusion"]["zones"])


def test_gcc_phat_known_delay():
    from caspian_audio.localize import gcc_phat_doppler

    rng = np.random.default_rng(0)
    s = rng.standard_normal(40000)
    a = s[10000:18000] + 0.3 * rng.standard_normal(8000)
    b = s[7137:21137] + 0.3 * rng.standard_normal(14000)
    d, psr, _ = gcc_phat_doppler(a, b, 16000, [400, 6000], [0.0])
    # ref[n] ≈ other[n + 2863] -> задержка относительно начала other
    assert abs(d * 16000 - 2863) < 0.5, d * 16000
    assert psr > 10


def test_doppler_compensation_moving_source():
    """Источник, задержка которого плывёт на 4 мс за 0.5 с: без компенсации пик теряется."""
    from caspian_audio.localize import gcc_phat_doppler

    sr, rng = 16000, np.random.default_rng(1)
    t = np.arange(int(3 * sr)) / sr
    src = rng.standard_normal(len(t) * 4)
    tu = np.arange(len(src)) / (sr * 4)
    tau = 0.2 + 0.008 * t  # 8 мс/с
    other = np.interp(t - tau, tu, src)
    ref = np.interp(t, tu, src)
    i0, n = int(1.0 * sr), int(0.5 * sr)
    lo = i0 - int(0.3 * sr)
    alphas = np.arange(-0.02, 0.0201, 0.0006)
    d, psr, a = gcc_phat_doppler(ref[i0:i0 + n], other[lo:i0 + n + int(0.3 * sr)], sr, [400, 6000], alphas)
    # событие ref в момент t_r приходит в other в t_o: t_o − τ(t_o) = t_r  =>  t_o = (t_r + 0.2) / 0.992
    t_r = 1.0 + 0.25
    true = (t_r + 0.2) / (1 - 0.008) - t_r
    assert abs(d + (lo - i0) / sr - true) < 0.3e-3, (d + (lo - i0) / sr, true)


def test_solve_position_inside_array():
    from caspian_audio.localize import solve_position

    b = np.array([[0, 0], [1200, 300], [500, 1100]], float)
    p = np.array([700.0, 600.0])
    c = 1480.0
    r = np.linalg.norm(b - p, axis=1)
    ref = int(np.argmin(r))
    delays = (r - r[ref]) / c
    res = solve_position(b, ref, delays, c, 1 / r, 1e-3, 0.7, 2500, 25)
    assert np.hypot(res["x"] - p[0], res["y"] - p[1]) < 1.0


def _fusion():
    from caspian_audio.fusion import Fusion

    ais = [{"mmsi": "1", "ts": "2026-05-19T21:30:00Z", "lat": "45.75", "lon": "48.39", "name": "SHIP"},
           {"mmsi": "1", "ts": "2026-05-19T21:31:00Z", "lat": "45.7505", "lon": "48.39", "name": "SHIP"}]
    return Fusion(CFG, ais)


def _alert(lat, lon, cat="boat_motor"):
    return {"id": "x", "category": cat, "title": cat, "confidence": 1.0, "start_ts": "2026-05-19T21:30:10Z",
            "end_ts": "2026-05-19T21:30:50Z", "location": {"lat": lat, "lon": lon}}


def test_fusion_dark_vs_ais():
    fu = _fusion()
    x, y = fu.frame.to_xy(45.75, 48.39)
    t = datetime(2026, 5, 19, 21, 30, 30, tzinfo=timezone.utc).timestamp()
    on_ship = fu.assess(_alert(45.75, 48.39), [{"t": t, "x": float(x), "y": float(y), "radius_m": 10}])
    assert on_ship["fusion"]["identification"]["status"] == "ais"
    away = fu.assess(_alert(45.757, 48.4106), [{"t": t, "x": 830.0, "y": 780.0, "radius_m": 10}])
    assert away["fusion"]["identification"]["status"] == "dark"
    assert away["fusion"]["risk"] > on_ship["fusion"]["risk"]
    # один буй без пеленга: судно с AIS в радиусе — честное «нужен пеленг»
    single = fu.assess(_alert(45.7505, 48.395))
    assert single["fusion"]["identification"]["status"] == "ais_nearby"


def test_water_discharge_vs_sensor_fault():
    from caspian_audio import water as W

    with tempfile.TemporaryDirectory() as d:
        W.simulate(CFG, d)
        res = W.detect(W.load_csv(Path(d) / "water.csv"), json.loads((Path(d) / "stations.json").read_text()))
    cats = [a["category"] for a in res["alerts"]]
    assert cats.count("water_discharge") == 1
    dis = next(a for a in res["alerts"] if a["category"] == "water_discharge")
    assert dis["stations"][:2] == ["W2", "W3"]  # шлейф вниз по течению
    faults = [a for a in res["alerts"] if a["category"] == "sensor_fault"]
    assert faults and all(a["sensor_id"] == "W4" for a in faults)  # сбой датчика не стал «сбросом»


def test_forecast_spawning_order():
    from caspian_audio import forecast as F

    today = date(2026, 5, 20)
    fc = F.forecast(F.simulate_hydrology(today), today)
    peak = {k: max(s["series"], key=lambda x: x["intensity"])["date"] for k, s in fc["species"].items()}
    assert peak["vobla"] < peak["sazan"] and peak["sudak"] < peak["sazan"]  # холодолюбивые раньше теплолюбивых
    assert "идёт" in fc["species"]["sazan"]["status"]


def test_satellite_plume_confirmed_by_water():
    from caspian_audio import satellite as S
    from caspian_audio import water as W

    with tempfile.TemporaryDirectory() as d:
        S.simulate(CFG, d)
        r = S.detect(Path(d) / "s2_2026-05-10.npz", Path(d) / "s2_2026-05-20.npz", CFG)
        W.simulate(CFG, d)
        st = json.loads((Path(d) / "stations.json").read_text())
        wa = W.detect(W.load_csv(Path(d) / "water.csv"), st)["alerts"]
    assert len(r["plumes"]) == 1  # кромка облака не даёт ложных шлейфов
    al = S.to_alerts(r, wa, st, CFG)
    assert any(x["factor"] == "confirmed" for x in al[0]["fusion"]["reasons"])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if fails else 0)
