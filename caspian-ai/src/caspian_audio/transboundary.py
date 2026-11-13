"""Трансграничный протокол: уведомления диспетчерам сопредельных государств Каспия.

Основание — Рамочная конвенция по защите морской среды Каспийского моря (Тегеран, 2003) и её
протоколы: стороны договорились сотрудничать и обмениваться информацией. Осетровые не знают
границ: стадо, которое сегодня в водах одной страны, завтра подходит к рекам другой, а браконьеры
идут за ним. Система сама собирает из своих данных черновик уведомления:

  «В квадрате D4 (Россия) зафиксирована активность браконьерских лодок. По модели миграции
   производители идут к квадратам B11, C11 (Казахстан). Рекомендуем усилить патрулирование».

Правила:
  * отправитель — национальный узел, получатели — национальные координационные центры (контакты
    задаются в конфиге; здесь — заглушки);
  * точность — квадрат 0.5°, без точных координат и маршрутов своих патрулей, без персональных
    данных и аудио (минимизация данных);
  * машиночитаемая часть (JSON, схема caspian-eco-alert/0.1) + текст на русском и английском;
  * контрольная сумма SHA-256 канонического JSON — целостность при пересылке;
  * НИЧЕГО НЕ ОТПРАВЛЯЕТСЯ АВТОМАТИЧЕСКИ: черновик → «одобрено диспетчером» → отправка через
    канал, согласованный сторонами (почта, защищённый шлюз), — это решение людей.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone

from .caspian import COUNTRIES

SCHEMA = "caspian-eco-alert/0.1"
FRAMEWORK = {"ru": "Тегеранская конвенция (2003): сотрудничество и обмен информацией",
             "en": "Tehran Convention (2003): cooperation and exchange of information"}
DATA_POLICY = {"ru": "точность — квадрат 0.5°; без координат патрулей, персональных данных и аудио",
               "en": "precision: 0.5° grid squares; no patrol positions, personal data or audio"}


def _sq_list(items, k=3):
    return ", ".join(x["square"] for x in items[:k])


def build(today: date, origin_country: str, grid, migration: dict, hotspots: list[dict], acoustic: list[dict],
          utc_offset: int = 4) -> list[dict]:
    """migration — forecast() производителей; hotspots — прогноз браконьерства на завтра;
    acoustic — свои тревоги (с location) за последние сутки."""
    tz = timezone(timedelta(hours=utc_offset))
    tomorrow = today + timedelta(days=1)
    night = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 18, tzinfo=tz)  # ночь: 18:00–08:00
    valid = {"from": night.isoformat(), "to": (night + timedelta(hours=14)).isoformat()}

    # свои подтверждённые признаки активности (квадраты)
    evidence = []
    for a in acoustic:
        f = a.get("fusion") or {}
        if a.get("type") != "acoustic" or f.get("priority") not in ("critical", "high") or not a.get("location"):
            continue
        sq = grid.square(a["location"]["lat"], a["location"]["lon"])
        dark = (f.get("identification") or {}).get("status") == "dark"
        evidence.append({"type": "acoustic", "square": sq, "country": origin_country, "time": a.get("start_ts"),
                         "lat": a["location"]["lat"], "lon": a["location"]["lon"],
                         "what_ru": {"boat_motor": "лодка без AIS" if dark else "шум мотора", "gunshot": "выстрел"}.get(a["category"], a.get("title", "")),
                         "what_en": {"boat_motor": "unidentified boat (no AIS)" if dark else "boat engine", "gunshot": "gunshot"}.get(a["category"], a["category"])})

    d0, d3 = migration["days"][0]["shares"], migration["days"][min(3, len(migration["days"]) - 1)]["shares"]
    tomorrow_sq = migration["days"][1]["top_squares"] if len(migration["days"]) > 1 else migration["days"][0]["top_squares"]
    out = []
    for c in COUNTRIES:
        if c == origin_country:
            continue
        fish_sq = [s for s in tomorrow_sq if s["country"] == c]
        # дополнительные квадраты этой страны из более широкого списка
        hs = [h for h in hotspots if h["country"] == c]
        border = [h for h in hotspots if h["country"] == origin_country and _near_border(grid, h, c)]
        share, trend = d0.get(c, 0), d3.get(c, 0) - d0.get(c, 0)
        if not fish_sq and not hs and not border:
            continue
        reasons_ru, reasons_en = [], []
        # свои наблюдения важны соседу, если они близко к его водам (то же стадо, те же браконьеры)
        ev_c = [e for e in evidence if _near_border(grid, e, c, km=300)]
        ev_sq = sorted({e["square"] for e in ev_c})
        if ev_c:
            reasons_ru.append(f"В квадрате {', '.join(ev_sq)} ({COUNTRIES[origin_country]['ru']}) зафиксирована активность браконьеров: "
                              + "; ".join(sorted({e['what_ru'] for e in ev_c})) + ".")
            reasons_en.append(f"Poaching activity detected in square {', '.join(ev_sq)} ({COUNTRIES[origin_country]['en']}): "
                              + "; ".join(sorted({e['what_en'] for e in ev_c})) + ".")
        if fish_sq or share > 0.05:
            sq_txt = _sq_list(fish_sq) if fish_sq else "—"
            reasons_ru.append(f"По модели миграции {share:.0%} нерестового стада осетровых находится в ваших водах"
                              + (f", растёт на {trend:+.0%} за 3 сут" if abs(trend) >= 0.01 else "")
                              + (f"; завтра наибольшая плотность — квадраты {sq_txt}." if fish_sq else "."))
            reasons_en.append(f"Migration model: {share:.0%} of the sturgeon spawning stock is in your waters"
                              + (f", {trend:+.0%} over 3 days" if abs(trend) >= 0.01 else "")
                              + (f"; highest density tomorrow in squares {sq_txt}." if fish_sq else "."))
        if hs:
            reasons_ru.append(f"Прогноз браконьерства на ночь: высокий риск в квадратах {_sq_list(hs)} ("
                              + "; ".join(sorted({w for h in hs[:3] for w in h['why'][:2]})) + ").")
            reasons_en.append(f"Poaching forecast for tonight: high risk in squares {_sq_list(hs)}.")
        if border:
            reasons_ru.append(f"Горячие точки у границы с вашими водами: {_sq_list(border)}.")
            reasons_en.append(f"Hotspots near your waters: {_sq_list(border)}.")
        squares = sorted({s["square"] for s in fish_sq} | {h["square"] for h in hs} | {h["square"] for h in border})
        priority = "high" if (hs and hs[0]["risk"] >= 0.8) or (ev_c and fish_sq) else "medium"
        rec_ru = f"Рекомендуем усилить патрулирование в квадратах {', '.join(squares[:5])} в ночь на {tomorrow:%d.%m}."
        rec_en = f"We recommend reinforced patrols in squares {', '.join(squares[:5])} on the night of {tomorrow:%d %b}."
        msg = {
            "id": str(uuid.uuid4()), "schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "valid": valid, "priority": priority, "status": "draft",
            "from": {"country": origin_country, "system": "Эко-Щит Каспия", "node": "национальный узел (демо)"},
            "to": {"country": c, "recipient": f"Национальный координационный центр — {COUNTRIES[c]['ru']} (контакт задаётся в конфиге)"},
            "framework": FRAMEWORK, "data_policy": DATA_POLICY,
            "squares": [{"square": s, "bounds": grid.square_bounds(s)} for s in squares],
            "evidence": [{k: v for k, v in e.items() if k not in ("lat", "lon")} for e in ev_c],  # только квадраты
            "forecast": {"stock_share_today": round(share, 3), "stock_share_trend_3d": round(trend, 3),
                         "fish_squares_tomorrow": fish_sq, "poaching_hotspots": hs[:4], "model_note": migration.get("source")},
            "text": {"ru": " ".join(reasons_ru) + " " + rec_ru, "en": " ".join(reasons_en) + " " + rec_en},
            "recommendation": {"ru": rec_ru, "en": rec_en},
        }
        msg["sha256"] = digest(msg)
        out.append(msg)
    return sorted(out, key=lambda m: (m["priority"] != "high", -m["forecast"]["stock_share_today"]))


def _near_border(grid, h, other: str, km: float = 40) -> bool:
    import numpy as np

    from .caspian import _km
    m = grid.country == other
    if not m.any():
        return False
    x, y = _km(grid.lon[m], grid.lat[m], h["lon"], h["lat"])
    return bool(np.min(np.hypot(x, y)) < km)


def digest(msg: dict) -> str:
    body = {k: v for k, v in msg.items() if k not in ("sha256", "status", "history")}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
