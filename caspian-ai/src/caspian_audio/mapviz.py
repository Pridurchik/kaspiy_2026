"""Карта для демо: один HTML-файл с SVG (без тайлов и интернета) — зоны, буи, AIS, треки пеленга, алерты."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

PRIORITY_COLOR = {"critical": "#d92d20", "high": "#f04438", "medium": "#f79009", "low": "#667085", "info": "#2e90fa"}
ZONE_STYLE = {"reserve_core": ("#12b76a", "Заповедное ядро"), "spawning": ("#7a5af8", "Нерестилище"),
              "fairway": ("#2e90fa", "Судовой ход")}


def render_map(path, meta, alerts, fusion, bxy, truth=None, size=720):
    pts = [bxy]
    for a in alerts:
        if a["_xy"]["track"]:
            pts.append(np.array(a["_xy"]["track"]))
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - 700, allp.max(0) + 700
    span = float(max(hi - lo))
    lo = (lo + hi) / 2 - span / 2
    S = size / span

    def P(x, y):
        return f"{(x - lo[0]) * S:.1f},{size - (y - lo[1]) * S:.1f}"

    svg = []
    for z in fusion.zones:
        col, _ = ZONE_STYLE.get(z.get("kind"), ("#98a2b3", z.get("kind", "")))
        for r in z["rings"]:
            svg.append(f'<polygon points="{" ".join(P(x, y) for x, y in r)}" fill="{col}" fill-opacity=".13" stroke="{col}" stroke-width="1.5"/>')
        c = z["rings"][0].mean(0)
        svg.append(f'<text x="{P(*c).split(",")[0]}" y="{P(*c).split(",")[1]}" class="zl" fill="{col}">{html.escape(z["name"])}</text>')
    if fusion.tracks:
        for tr in fusion.tracks.values():
            xy = np.stack([tr["x"], tr["y"]], 1)
            m = ((xy > lo) & (xy < lo + span)).all(1)
            if m.sum() < 2:
                continue
            svg.append(f'<polyline points="{" ".join(P(x, y) for x, y in xy[m])}" class="ais"/>')
            for x, y in xy[m]:
                svg.append(f'<rect x="{float(P(x, y).split(",")[0]) - 2.5}" y="{float(P(x, y).split(",")[1]) - 2.5}" width="5" height="5" class="aisp"/>')
            x, y = xy[m][-1]
            svg.append(f'<text x="{float(P(x, y).split(",")[0]) + 6}" y="{P(x, y).split(",")[1]}" class="lbl">AIS {html.escape(tr["name"] or tr["mmsi"])}</text>')
    if truth:
        for s in truth["sources"]:
            if len(s["t"]) > 1:
                svg.append(f'<polyline points="{" ".join(P(x, y) for x, y in zip(s["x"], s["y"]))}" class="truth"/>')
    for a in alerts:
        col = PRIORITY_COLOR.get(a["fusion"].get("priority"), "#667085")
        fx, tr = a["_xy"]["fixes"], a["_xy"]["track"]
        for x, y in fx:
            svg.append(f'<circle cx="{P(x, y).split(",")[0]}" cy="{P(x, y).split(",")[1]}" r="2" fill="{col}" opacity=".45"/>')
        if len(tr) > 1:
            svg.append(f'<polyline points="{" ".join(P(x, y) for x, y in tr)}" fill="none" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
        if tr:
            x, y = tr[-1]
            cx, cy = P(x, y).split(",")
            rad = max(a["location"].get("radius_m", 0) * S, 4)
            shape = (f'<path d="M{float(cx)-7},{float(cy)-7} l14,14 m0,-14 l-14,14" stroke="{col}" stroke-width="3"/>'
                     if a["category"] == "gunshot" else f'<circle cx="{cx}" cy="{cy}" r="6" fill="{col}" stroke="#fff" stroke-width="2"/>')
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="{rad:.1f}" fill="{col}" fill-opacity=".12" stroke="{col}" stroke-dasharray="3 3"/>{shape}')
    for b, (x, y) in zip(meta["buoys"], bxy):
        cx, cy = P(x, y).split(",")
        svg.append(f'<polygon points="{cx},{float(cy)-9} {float(cx)+8},{float(cy)+6} {float(cx)-8},{float(cy)+6}" class="buoy"/>'
                   f'<text x="{float(cx)+10}" y="{float(cy)+4}" class="lbl">{html.escape(b["id"])}</text>')
    bar = 500 * S
    svg.append(f'<line x1="16" y1="{size-18}" x2="{16+bar:.1f}" y2="{size-18}" class="bar"/><text x="16" y="{size-24}" class="lbl">500 м</text>')

    cards = []
    for a in alerts:
        f = a["fusion"]
        col = PRIORITY_COLOR.get(f.get("priority"), "#667085")
        rs = "".join(f'<li><b>{r["points"]:+d}</b> {html.escape(r["text"])}</li>' for r in f.get("reasons", []))
        mot = f.get("motion")
        loc = a.get("location", {})
        where = (f'{loc["lat"]:.5f}, {loc["lon"]:.5f} ± {loc.get("radius_m", 0):.0f} м' if loc.get("source") == "tdoa"
                 else f'по бую {html.escape(a["sensor_id"])}')
        cards.append(
            f'<div class="card" style="border-left-color:{col}"><div class="hd"><span class="pr" style="background:{col}">'
            f'{html.escape(str(f.get("priority", "")).upper())} · {f.get("risk", "–")}</span> {html.escape(a["title"])}</div>'
            f'<div class="mt">{html.escape(f.get("local_time", "")[11:19])} · слышали: {", ".join(a.get("heard_by", []))}<br>{where}'
            + (f' · {mot["speed_kn"]} уз, курс {mot["course_deg"]:.0f}°' if mot else "") + f'</div><ul>{rs}</ul></div>')

    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for c, n in ZONE_STYLE.values())
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сеть буёв — пеленг</title><style>
:root{{--bg:#f7f8fa;--fg:#101828;--mut:#667085;--card:#fff;--line:#e4e7ec;--water:#eaf3fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c111d;--fg:#f2f4f7;--mut:#98a2b3;--card:#161b26;--line:#2a3140;--water:#0f1d2e}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
.wrap{{display:flex;gap:20px;padding:16px;flex-wrap:wrap;max-width:1240px;margin:auto}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:10px}}
svg{{width:100%;max-width:{size}px;height:auto;background:var(--water);border:1px solid var(--line);border-radius:10px}}
.side{{flex:1;min-width:300px}} .card{{background:var(--card);border:1px solid var(--line);border-left:5px solid;border-radius:8px;padding:10px 12px;margin-bottom:10px}}
.hd{{font-weight:600}} .pr{{color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:4px}}
.mt{{color:var(--mut);font-size:12px;margin:4px 0}} ul{{margin:4px 0 0 0;padding-left:18px}} li{{margin:1px 0}}
.zl{{font-size:11px;font-weight:600;text-anchor:middle}} .lbl{{font-size:11px;fill:var(--fg)}}
.ais{{fill:none;stroke:#2e90fa;stroke-width:1.5;stroke-dasharray:6 4}} .aisp{{fill:#2e90fa}}
.truth{{fill:none;stroke:var(--mut);stroke-width:1;stroke-dasharray:2 3}} .buoy{{fill:#fdb022;stroke:#7a2e0e;stroke-width:1}}
.bar{{stroke:var(--fg);stroke-width:2}} .lg span{{margin-right:12px;font-size:12px;color:var(--mut)}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}}
</style></head><body><div class="wrap"><div><h1>Сеть буёв: пеленг и фьюжн с AIS</h1>
<div class="sub">{html.escape(meta.get("recorded_at", ""))} · {html.escape(meta.get("note", ""))}</div>
<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">{"".join(svg)}</svg>
<div class="lg">{legend}<span><i style="background:#2e90fa"></i>AIS (пунктир)</span><span><i style="background:#fdb022"></i>буй</span>
<span>точки — засечки, линия — трек, круг — ~95% неопределённость{", серый пунктир — истинный путь" if truth else ""}</span></div></div>
<div class="side"><h1>Алерты</h1><div class="sub">по убыванию риска</div>{"".join(cards)}</div></div></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
