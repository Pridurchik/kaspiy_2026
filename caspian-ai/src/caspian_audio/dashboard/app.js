"use strict";
// Дашборд инспектора: карта + лента тревог + экостатус + прогноз. Данные: /api/*, живые события: /api/stream.

const TZ = "Europe/Astrakhan";
const PRIO = { critical: "КРИТ", high: "ВЫСОК", medium: "СРЕДН", low: "НИЗК", info: "ИНФО" };
const PRIO_COLOR = { critical: "#d92d20", high: "#f04438", medium: "#f79009", low: "#667085", info: "#2e90fa" };
const ZONE_STYLE = { spawning: ["#7a5af8", "Нерестилище"], reserve_core: ["#12b76a", "Заповедное ядро"], fairway: ["#2e90fa", "Судовой ход"] };
const SOURCE = { acoustic: "🎧 Акустика", water: "💧 Датчики воды", satellite: "🛰 Спутник" };
const STATUS_RU = { new: "новая", in_progress: "выезд", confirmed: "подтверждена", false_alarm: "ложная", closed: "закрыта" };
const LEVEL_RU = { green: "норма", yellow: "внимание", red: "тревога" };

const state = { overview: null, alerts: [], forecast: null, selected: null, layers: {}, charts: {} };
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtTime = (iso, withDate) => {
  if (!iso) return "";
  const o = { hour: "2-digit", minute: "2-digit", timeZone: TZ };
  if (withDate) Object.assign(o, { day: "2-digit", month: "2-digit" });
  return new Date(iso).toLocaleString("ru-RU", o);
};
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
};

// ------------------------------------------------------------------ карта
let map;
function initMap(origin) {
  map = L.map("map", { zoomControl: true }).setView(origin, 14);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap" }).addTo(map);
  for (const k of ["zones", "sat", "ais", "stations", "buoys", "alerts"]) state.layers[k] = L.layerGroup().addTo(map);
  L.control.layers(null, {
    "Зоны ООПТ": state.layers.zones, "Спутник: мутность": state.layers.sat, "AIS": state.layers.ais,
    "Датчики воды": state.layers.stations, "Буи": state.layers.buoys, "Тревоги": state.layers.alerts,
  }, { collapsed: true }).addTo(map);
  $("#legend").innerHTML = Object.values(ZONE_STYLE).map(([c, n]) => `<div><i style="background:${c};opacity:.5"></i>${n}</div>`).join("")
    + `<div><span class="buoy-icon">▲</span>акустический буй</div><div><i style="background:#2e90fa;border-radius:50%"></i>датчик воды</div>`
    + `<div><span class="ship-icon">⛴</span>судно с AIS</div><div><i style="background:#ff8a3d"></i>спутник: рост мутности</div><div><i style="background:#b4b4be"></i>облака (исключены)</div>`
    + `<div class="note">Зоны и станции — условные (демо)</div>`;
}

function drawStatic(ov) {
  const Lz = state.layers.zones;
  Lz.clearLayers();
  const eco = Object.fromEntries((ov.eco || []).map((e) => [e.name, e]));
  L.geoJSON(ov.zones, {
    style: (f) => {
      const [c] = ZONE_STYLE[f.properties.kind] || ["#98a2b3"];
      const lvl = eco[f.properties.name]?.level;
      return { color: lvl === "red" ? "#e5484d" : c, weight: lvl === "red" ? 3 : 1.5, fillColor: c, fillOpacity: 0.14, dashArray: f.properties.kind === "fairway" ? "6 4" : null };
    },
    onEachFeature: (f, l) => {
      const e = eco[f.properties.name];
      l.bindTooltip(`<b>${esc(f.properties.name)}</b>${e ? `<br>${LEVEL_RU[e.level]}: ${esc(e.reasons[0])}` : ""}`, { sticky: true });
    },
  }).addTo(Lz);

  const Ls = state.layers.sat;
  Ls.clearLayers();
  if (ov.satellite) {
    L.imageOverlay(ov.satellite.overlay, ov.satellite.bounds, { opacity: 0.85 }).addTo(Ls);
  }

  const La = state.layers.ais;
  La.clearLayers();
  for (const v of ov.ais || []) {
    const pts = v.track.map((p) => [p[1], p[2]]);
    L.polyline(pts, { color: "#2e90fa", weight: 2, dashArray: "6 5" }).addTo(La);
    L.marker(pts[pts.length - 1], { icon: L.divIcon({ className: "", html: '<span class="ship-icon">⛴</span>', iconSize: [18, 18] }) })
      .bindTooltip(`AIS: ${esc(v.name)} · MMSI ${v.mmsi}`).addTo(La);
  }

  const Lw = state.layers.stations;
  Lw.clearLayers();
  for (const st of ov.stations || []) {
    const ws = ov.water_status?.[st.id] || {};
    const col = { green: "#12b76a", yellow: "#f5b30a", red: "#e5484d" }[ws.level] || "#2e90fa";
    L.circleMarker([st.lat, st.lon], { radius: 7, color: "#fff", weight: 2, fillColor: col, fillOpacity: 1 })
      .bindTooltip(`<b>${st.id}</b> ${esc(st.name)}<br>${LEVEL_RU[ws.level] || ""}`)
      .on("click", () => { showTab("eco"); showStation(st.id); }).addTo(Lw);
  }

  const Lb = state.layers.buoys;
  Lb.clearLayers();
  for (const b of ov.buoys || []) {
    L.marker([b.lat, b.lon], { icon: L.divIcon({ className: "", html: `<span class="buoy-icon ${b.online ? "" : "off"}">▲</span>`, iconSize: [20, 20] }) })
      .bindTooltip(`<b>${b.id}</b> · ${b.online ? "на связи" : "НЕТ СВЯЗИ"} · заряд ${b.battery_pct}%`).addTo(Lb);
  }
}

function drawAlerts() {
  const L1 = state.layers.alerts;
  L1.clearLayers();
  for (const a of state.alerts) {
    if (!a.location || a.category === "sensor_fault") continue;
    const col = PRIO_COLOR[a.fusion?.priority] || "#667085";
    const done = ["closed", "false_alarm"].includes(a.status);
    const ll = [a.location.lat, a.location.lon];
    let m;
    if (a.polygon) {
      m = L.polygon(a.polygon.map((p) => [p[1], p[0]]), { color: col, weight: 2, fillOpacity: 0.08, dashArray: "4 3" });
    } else if (a.category === "gunshot") {
      m = L.marker(ll, { icon: L.divIcon({ className: "", html: '<span class="shot-icon">✕</span>', iconSize: [22, 22] }) });
    } else {
      m = L.circleMarker(ll, { radius: 8, color: "#fff", weight: 2, fillColor: col, fillOpacity: done ? 0.4 : 1 });
    }
    m.bindTooltip(`${esc(a.title)} · ${a.fusion?.risk ?? ""}`).on("click", () => openAlert(a.id)).addTo(L1);
    const tr = a.localization?.track;
    if (tr && tr.length > 1) {
      L.polyline(tr.map((p) => [p.lat, p.lon]), { color: col, weight: 4, opacity: done ? 0.35 : 0.9 }).addTo(L1);
    }
    if (a.location.radius_m) L.circle(ll, { radius: a.location.radius_m, color: col, weight: 1, fillOpacity: 0.06, dashArray: "3 3" }).addTo(L1);
  }
}

// ------------------------------------------------------------------ лента тревог
function cardHTML(a) {
  const f = a.fusion || {};
  const id = f.identification || {};
  const idTag = id.status === "dark" ? '<span class="tag dark">тёмная лодка</span>'
    : id.status === "ais" ? `<span class="tag">AIS ${esc(id.name || id.mmsi)}</span>`
    : id.status === "ais_nearby" ? '<span class="tag">рядом AIS, нужен пеленг</span>'
    : id.status === "own" ? '<span class="tag">свой патруль</span>' : "";
  const loc = a.location?.source === "tdoa" ? '<span class="tag">пеленг</span>' : "";
  const st = a.status !== "new" ? `<span class="tag">${STATUS_RU[a.status] || a.status}</span>` : "";
  const top = (f.reasons || []).filter((r) => r.factor !== "category").slice(0, 2);
  return `<div class="card p-${f.priority} ${["closed", "false_alarm"].includes(a.status) ? "done" : ""}" data-id="${a.id}">
    <div class="row"><span class="prio ${f.priority}">${PRIO[f.priority] || ""}</span><span class="risk">${f.risk ?? ""}</span>
      <span class="title">${esc(a.title)}</span></div>
    <div class="meta">${fmtTime(a.start_ts)} · ${SOURCE[a.type] || ""} ${a.sensor_id ? "· " + esc(a.sensor_id) : ""} ${idTag} ${loc} ${st}</div>
    ${top.length ? `<ul class="reasons">${top.map((r) => `<li><b class="${r.points < 0 ? "neg" : ""}">${r.points ? (r.points > 0 ? "+" : "") + r.points : ""}</b><span>${esc(r.text)}</span></li>`).join("")}</ul>` : ""}
  </div>`;
}

function renderAlerts(freshId) {
  const active = state.alerts.filter((a) => !["closed", "false_alarm"].includes(a.status) && a.fusion?.priority !== "info");
  $("#alertCount").textContent = active.filter((a) => ["critical", "high"].includes(a.fusion?.priority)).length;
  $("#alertList").innerHTML = state.alerts.length ? state.alerts.map(cardHTML).join("") : '<div class="empty">Тревог нет</div>';
  $("#alertList").querySelectorAll(".card").forEach((el) => el.addEventListener("click", () => openAlert(el.dataset.id)));
  if (freshId) $(`.card[data-id="${freshId}"]`)?.classList.add("fresh");
  drawAlerts();
}

async function openAlert(id) {
  showTab("alerts");
  const a = await api(`/api/alerts/${id}`);
  state.selected = id;
  const f = a.fusion || {};
  const loc = a.location || {};
  const mot = f.motion;
  const audio = a.evidence_audio ? `<h3>Аудио-доказательство</h3><img class="spec" src="/api/alerts/${id}/spectrogram.png" alt="спектрограмма">
    <audio controls preload="none" src="/api/alerts/${id}/audio"></audio><div class="note">Спектрограмма 30 Гц – 8 кГц (лог. шкала), 8 с вокруг события</div>` : "";
  const sc = a.server_check ? `<span>Проверка сервером</span><span>${a.server_check.confirmed ? "подтверждено" : "НЕ подтверждено"}</span>` : "";
  $("#alertDetail").innerHTML = `<button class="btn back" id="back">← Все тревоги</button>
    <div class="row"><span class="prio ${f.priority}">${PRIO[f.priority] || ""}</span><span class="risk">риск ${f.risk ?? "–"}</span>
    <span class="tag">${STATUS_RU[a.status] || a.status}</span></div>
    <h2>${esc(a.title)}</h2>
    <div class="kv"><span>Время</span><span>${fmtTime(a.start_ts, true)}–${fmtTime(a.end_ts)}</span>
      <span>Источник</span><span>${SOURCE[a.type] || ""} ${esc(a.sensor_id || "")}${a.heard_by ? " · слышали: " + a.heard_by.join(", ") : ""}</span>
      <span>Место</span><span>${loc.lat ? `${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}` : "—"}${loc.radius_m ? ` ± ${Math.round(loc.radius_m)} м` : ""} ${loc.source === "tdoa" ? "(пеленг TDOA)" : loc.source === "sensor" ? "(по датчику)" : ""}</span>
      ${mot ? `<span>Движение</span><span>${mot.speed_kn} уз, курс ${mot.course_deg}°</span>` : ""}
      ${a.vessel ? `<span>Судно (гидроакустика)</span><span>${esc(a.vessel.type_ru)} · f₀ ${a.vessel.f0_hz} Гц · ${esc(a.vessel.movement_ru)}<br><span class="note">${esc(a.vessel.method)}</span></span>` : ""}
      ${f.approach ? `<span>Подход к зоне</span><span>${esc(f.approach.zone)}: ${f.approach.distance_m} м, ~${f.approach.eta_min} мин</span>` : ""}
      ${a.area_ha ? `<span>Площадь</span><span>${a.area_ha} га</span>` : ""}
      ${a.confidence != null ? `<span>Уверенность</span><span>${Math.round(a.confidence * 100)}%</span>` : ""}${sc}
    </div>
    <h3>Почему такой приоритет</h3>
    <ul class="reasons">${(f.reasons || []).map((r) => `<li><b class="${r.points < 0 ? "neg" : ""}">${r.points ? (r.points > 0 ? "+" : "") + r.points : ""}</b><span>${esc(r.text)}</span></li>`).join("")}</ul>
    ${audio}
    <h3>Решение инспектора</h3>
    <div class="actions">
      <button class="btn primary" data-st="in_progress">Выезжаю</button>
      <button class="btn ok" data-st="confirmed">Подтвердить</button>
      <button class="btn bad" data-st="false_alarm">Ложная тревога</button>
      <button class="btn" data-st="closed">Закрыть</button>
    </div>
    <div class="note" style="margin-top:6px">Подтверждения и ложные тревоги с аудио уходят в датасет дообучения модели.</div>`;
  $("#alertList").hidden = true;
  $("#alertDetail").hidden = false;
  $("#back").onclick = closeAlert;
  $("#alertDetail").querySelectorAll("[data-st]").forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/alerts/${id}/status`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: b.dataset.st }) });
    toast(`Статус: ${STATUS_RU[b.dataset.st]}`, "info");
    await refreshAlerts();
    openAlert(id);
  }));
  const tr = a.localization?.track;
  if (tr && tr.length > 1) map.fitBounds(tr.map((p) => [p.lat, p.lon]), { maxZoom: 16, padding: [60, 60] });
  else if (a.polygon) map.fitBounds(a.polygon.map((p) => [p[1], p[0]]), { maxZoom: 16, padding: [60, 60] });
  else if (loc.lat) map.setView([loc.lat, loc.lon], Math.max(map.getZoom(), 15));
}

function closeAlert() {
  state.selected = null;
  $("#alertDetail").hidden = true;
  $("#alertList").hidden = false;
}

// ------------------------------------------------------------------ экостатус и вода
function renderEco() {
  const ov = state.overview;
  const zones = (ov.eco || []).map((e) => `<div class="box"><div class="row"><span class="dot ${e.level}"></span><b>${esc(e.name)}</b>
    <span class="meta">${LEVEL_RU[e.level]}</span></div><ul class="reasons">${e.reasons.map((r) => `<li><span>${esc(r)}</span></li>`).join("")}</ul></div>`).join("");
  const st = (ov.stations || []).map((s) => {
    const w = ov.water_status?.[s.id] || {};
    const l = w.last || {};
    return `<div class="box" style="cursor:pointer" data-station="${s.id}"><div class="row"><span class="dot ${w.level}"></span><b>${s.id}</b> ${esc(s.name)}</div>
      <div class="meta">O₂ ${l.do_mg_l?.toFixed?.(1)} мг/л · мутность ${l.turbidity_ntu?.toFixed?.(0)} NTU · pH ${l.ph?.toFixed?.(2)} · ${l.temp_c?.toFixed?.(1)} °C
      ${w.violations?.length ? `<br><b style="color:var(--crit)">${esc(w.violations.join("; "))}</b>` : ""}</div></div>`;
  }).join("");
  const sat = ov.satellite ? `<div class="box"><b>Sentinel-2</b> · снимок ${fmtTime(ov.satellite.date, true)}, сравнение с ${fmtTime(ov.satellite.baseline, true)}
    <div class="meta">облачность ${Math.round(ov.satellite.cloud_fraction * 100)}% (маскируется) · шлейфов: ${ov.satellite.plumes.length}${ov.satellite.plumes[0] ? ` · крупнейший ${ov.satellite.plumes[0].area_ha} га` : ""}</div></div>` : "";
  $("#panel-eco").innerHTML = `<h3>Зоны ООПТ</h3>${zones}<h3>Датчики воды</h3>${st}<div id="stationChart"></div><h3>Спутник</h3>${sat}
    <div class="note">Демо-данные: зоны, станции и снимки синтетические. Модули принимают реальные ряды и снимки (см. README).</div>`;
  $("#panel-eco").querySelectorAll("[data-station]").forEach((el) => el.addEventListener("click", () => showStation(el.dataset.station)));
}

async function showStation(id) {
  const d = await api(`/api/water/${id}?hours=72`);
  const st = state.overview.stations.find((s) => s.id === id);
  $("#stationChart").innerHTML = `<div class="box"><b>${id}</b> ${esc(st?.name)} · последние 72 ч<div class="chart"><canvas id="cWater"></canvas></div></div>`;
  state.charts.water?.destroy();
  const labels = d.ts.map((t) => fmtTime(t, true));
  state.charts.water = new Chart($("#cWater"), {
    type: "line",
    data: { labels, datasets: [
      { label: "O₂, мг/л", data: d.do_mg_l, borderColor: "#0e7490", pointRadius: 0, borderWidth: 1.5, yAxisID: "y" },
      { label: "Мутность, NTU", data: d.turbidity_ntu, borderColor: "#f79009", pointRadius: 0, borderWidth: 1.5, yAxisID: "y1" },
    ] },
    options: chartOpts({ y: { title: { display: true, text: "O₂" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "NTU" } } }),
  });
  $("#stationChart").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function chartOpts(scales) {
  const fg = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  return {
    responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: fg, boxWidth: 12, font: { size: 11 } } } },
    scales: Object.fromEntries(Object.entries({ x: { ticks: { maxTicksLimit: 6, color: fg, font: { size: 10 } } }, ...scales })
      .map(([k, v]) => [k, { ...v, ticks: { ...(v.ticks || {}), color: fg, font: { size: 10 } } }])),
  };
}

// ------------------------------------------------------------------ прогноз
const SP_COLORS = ["#7a5af8", "#0e7490", "#f79009", "#d92d20", "#12b76a"];
function renderForecast() {
  const fc = state.forecast;
  if (!fc) return;
  const sp = Object.values(fc.species);
  $("#panel-forecast").innerHTML = `<h3>Нерест на ${fmtTime(fc.today + "T12:00:00+04:00", true).split(",")[0]} и +${fc.horizon_days} дн</h3>
    <div class="box">${sp.map((s, i) => `<div class="row"><span class="dot" style="background:${SP_COLORS[i]}"></span><b>${esc(s.name)}</b><span class="meta">${esc(s.status)}</span></div>`).join("")}</div>
    <div class="box"><div class="chart"><canvas id="cSpawn"></canvas></div><div class="note">Интенсивность нереста (медиана ансамбля из 50 сценариев). Справа от «сегодня» — прогноз.</div></div>
    <div class="box"><div class="chart"><canvas id="cTemp"></canvas></div><div class="note">Температура воды: факт, прогноз и 10–90% диапазон. Аномалия к норме: ${fc.temp_anomaly_c > 0 ? "+" : ""}${fc.temp_anomaly_c} °C.</div></div>
    <h3>План патрулирования</h3>
    ${(fc.patrol_plan || []).map((p) => `<div class="box"><div class="row"><b>${p.date.slice(8, 10)}.${p.date.slice(5, 7)}</b><span>${esc(p.zone)}</span><span class="risk" style="margin-left:auto">${p.score}</span></div>
      <div class="meta">${esc(p.why)} · ${esc(p.when)}</div></div>`).join("") || '<div class="empty">Нет рекомендаций</div>'}
    <div class="note">Окна запрета на нерестилищах берутся из прогноза: тревога в зоне получает баллы «идёт нерест».
    Параметры видов ориентировочные, требуют калибровки с ихтиологами. Источник гидрологии: ${esc(fc.source)}.</div>`;
  const labels = sp[0].series.map((s) => s.date.slice(8, 10) + "." + s.date.slice(5, 7));
  const todayIdx = sp[0].series.findIndex((s) => s.date === fc.today);
  const todayLine = {
    id: "today", afterDraw(c) {
      const x = c.scales.x.getPixelForValue(todayIdx);
      const g = c.ctx;
      g.save(); g.strokeStyle = "#98a2b3"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x, c.chartArea.top); g.lineTo(x, c.chartArea.bottom); g.stroke();
      g.fillStyle = "#98a2b3"; g.font = "10px system-ui"; g.fillText("сегодня", x + 3, c.chartArea.top + 10); g.restore();
    },
  };
  state.charts.spawn?.destroy();
  state.charts.spawn = new Chart($("#cSpawn"), {
    type: "line", plugins: [todayLine],
    data: { labels, datasets: sp.map((s, i) => ({ label: s.name, data: s.series.map((x) => x.intensity), borderColor: SP_COLORS[i], pointRadius: 0, borderWidth: 2,
      segment: { borderDash: (ctx) => (ctx.p0DataIndex >= todayIdx ? [5, 4] : undefined) } })) },
    options: chartOpts({ y: { min: 0, max: 1, title: { display: true, text: "интенсивность" } } }),
  });
  const th = fc.water_temp.history, tf = fc.water_temp.forecast;
  const tl = [...th.map((r) => r[0]), ...tf.map((r) => r[0])].map((d) => d.slice(8, 10) + "." + d.slice(5, 7));
  const pad = th.map(() => null);
  state.charts.temp?.destroy();
  state.charts.temp = new Chart($("#cTemp"), {
    type: "line",
    data: { labels: tl, datasets: [
      { label: "факт, °C", data: [...th.map((r) => r[1]), ...tf.map(() => null)], borderColor: "#0e7490", pointRadius: 0, borderWidth: 2 },
      { label: "прогноз", data: [...pad.slice(1), th[th.length - 1][1], ...tf.map((r) => r[1])], borderColor: "#f04438", borderDash: [5, 4], pointRadius: 0, borderWidth: 2 },
      { label: "90%", data: [...pad, ...tf.map((r) => r[3])], borderColor: "transparent", backgroundColor: "rgba(240,68,56,.12)", pointRadius: 0, fill: "+1" },
      { label: "10%", data: [...pad, ...tf.map((r) => r[2])], borderColor: "transparent", pointRadius: 0 },
    ] },
    options: { ...chartOpts({}), plugins: { legend: { labels: { filter: (i) => !["90%", "10%"].includes(i.text), boxWidth: 12, font: { size: 11 } } } } },
  });
}

// ------------------------------------------------------------------ буи и данные
function renderBuoys() {
  const ov = state.overview;
  const fb = ov.feedback || {};
  $("#panel-buoys").innerHTML = `<h3>Акустические буи</h3>${(ov.buoys || []).map((b) => `<div class="box"><div class="row">
      <span class="dot ${b.online ? "green" : "red"}"></span><b>${b.id}</b><span class="meta">${b.online ? "на связи" : "нет связи"} · заряд ${b.battery_pct}%</span></div>
      <div class="meta">${esc(b.sensors.join(", "))} · последний сеанс ${fmtTime(b.last_seen, true)}${b.note ? `<br><b>${esc(b.note)}</b>` : ""}</div></div>`).join("")}
    <h3>Источники данных</h3>
    <div class="box"><div class="kv"><span>🎧 Акустика</span><span>${(ov.buoys || []).filter((b) => b.online).length} буя на связи, детекция на устройстве (ONNX), пеленг на сервере</span>
      <span>💧 Вода</span><span>${(ov.stations || []).length} станции, шаг 10 мин</span>
      <span>🛰 Спутник</span><span>Sentinel-2, каждые ~5 дней (при облачности — пропуск)</span>
      <span>⛴ AIS</span><span>${(ov.ais || []).length} судна в районе</span></div></div>
    <h3>Обучение на решениях инспектора</h3>
    <div class="box"><div class="kv"><span>Подтверждено</span><span>${fb.confirmed || 0}</span><span>Ложных тревог</span><span>${fb.false_alarm || 0}</span>
      <span>Выездов</span><span>${fb.in_progress || 0}</span></div>
      <div class="note">Клипы подтверждённых и ложных тревог сохраняются в data/feedback/ и попадают в следующее дообучение модели.</div></div>`;
}

// ------------------------------------------------------------------ Каспий: миграция, браконьерство, трансграничный обмен
const CC = { RU: "#2e90fa", KZ: "#12b76a", TM: "#f79009", IR: "#d92d20", AZ: "#7a5af8" };
const LAYERS = { spawners: "Производители", juvenile: "Молодь", risk: "Риск браконьерства", temp: "Температура", sal: "Солёность" };
const cas = { meta: null, msgs: [], layer: "spawners", day: 1, lang: "ru", group: null, overlay: null, saved: null, showInc: false };

async function enterCaspian() {
  if (!cas.meta) {
    try { cas.meta = await api("/api/caspian"); } catch (e) { $("#panel-caspian").innerHTML = '<div class="empty">Нет данных Каспия: make demo</div>'; return; }
  }
  cas.msgs = await api("/api/transboundary").catch(() => []);
  cas.saved = { center: map.getCenter(), zoom: map.getZoom() };
  for (const k of ["zones", "sat", "stations", "buoys", "alerts"]) map.removeLayer(state.layers[k]);
  cas.group = cas.group || L.layerGroup();
  cas.group.addTo(map);
  map.fitBounds(cas.meta.bounds, { padding: [10, 10] });
  renderCaspian();
}

function leaveCaspian() {
  if (cas.group) map.removeLayer(cas.group);
  for (const k of ["zones", "sat", "stations", "buoys", "alerts"]) state.layers[k].addTo(map);
  if (cas.saved) map.setView(cas.saved.center, cas.saved.zoom);
}

function layerUrl() {
  if (cas.layer === "risk") return `/api/caspian/layer/risk_d${Math.min(Math.max(cas.day, 1), 3)}.png`;
  if (cas.layer === "temp" || cas.layer === "sal") return `/api/caspian/layer/${cas.layer}.png`;
  return `/api/caspian/layer/${cas.layer}_d${cas.day}.png`;
}

function drawCaspianMap() {
  const m = cas.meta, G = cas.group;
  G.clearLayers();
  L.imageOverlay(layerUrl(), m.bounds, { opacity: 0.85 }).addTo(G);
  for (const [name, ll] of Object.entries(m.rivers))
    L.circleMarker(ll, { radius: 4, color: "#0e7490", weight: 2, fillOpacity: 0 }).bindTooltip(`устье: ${esc(name)}`).addTo(G);
  for (const s of m.sensors)
    L.circleMarker([s.lat, s.lon], { radius: 3, color: "#fff", weight: 1, fillColor: "#475467", fillOpacity: 1 })
      .bindTooltip(`датчик ${s.id} (${s.country}): ${s.temp} °C, ${s.sal}‰`).addTo(G);
  if (cas.layer === "risk") {
    const rd = m.poaching.days[Math.min(Math.max(cas.day, 1), 3) - 1];
    for (const h of rd.hotspots.slice(0, 6)) {
      const b = h.bounds;
      L.rectangle([[b[0], b[1]], [b[2], b[3]]], { color: "#d92d20", weight: 2, fill: false })
        .bindTooltip(`<b>${h.square}</b> (${h.country}) риск ${Math.round(h.risk * 100)}%<br>${esc(h.why.join("; "))}`).addTo(G);
    }
  } else if (cas.layer in m.stages) {
    const d = m.stages[cas.layer].days[cas.day];
    for (const sq of d.top_squares.slice(0, 3)) {
      const b = sq.bounds;
      L.rectangle([[b[0], b[1]], [b[2], b[3]]], { color: "#fff", weight: 1.5, dashArray: "4 3", fill: false })
        .bindTooltip(`${sq.square} (${sq.country}): ${Math.round(sq.p * 100)}% рыбы`).addTo(G);
    }
    const track = m.stages[cas.layer].days.map((x) => x.center);
    L.polyline(track, { color: "#fff", weight: 2 }).addTo(G);
    L.circleMarker(d.center, { radius: 5, color: "#fff", fillColor: "#101828", fillOpacity: 1, weight: 2 }).bindTooltip("центр масс стада").addTo(G);
  }
  if (cas.showInc)
    for (const i of m.poaching.incidents_sample)
      L.circleMarker([i.lat, i.lon], { radius: 2, color: "#f04438", weight: 0, fillOpacity: 0.6 }).bindTooltip(`${i.date}: ${esc(i.type)}`).addTo(G);
}

function sharesBar(sh) {
  return `<div class="bar">${Object.entries(sh).filter(([, v]) => v > 0.005).map(([c, v]) => `<span style="width:${v * 100}%;background:${CC[c]}" title="${c} ${Math.round(v * 100)}%"></span>`).join("")}</div>
    <div class="legend-row">${Object.entries(sh).filter(([, v]) => v > 0.005).map(([c, v]) => `<span><i class="dot" style="background:${CC[c]}"></i> ${cas.meta.countries[c].ru} ${Math.round(v * 100)}%</span>`).join("")}</div>`;
}

function renderCaspian() {
  const m = cas.meta;
  const isStage = cas.layer in m.stages;
  const maxDay = cas.layer === "risk" ? 3 : isStage ? m.horizon : 0;
  if (cas.layer === "risk" && cas.day < 1) cas.day = 1;
  if (cas.day > maxDay) cas.day = maxDay;
  const dayDate = new Date(m.today + "T12:00:00");
  dayDate.setDate(dayDate.getDate() + cas.day);
  const dd = dayDate.toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
  let body = "";
  if (isStage) {
    const st = m.stages[cas.layer], d = st.days[cas.day];
    body = `<div class="box"><b>${esc(st.name)}</b> · ${dd}${cas.day ? ` (+${cas.day} сут)` : " (сегодня)"}
      <div class="meta">Центр масс стада смещается на ${st.shift_km_day} км/сут, курс ${st.shift_course_deg}°</div>
      <h3>Где стадо (доли по водам государств)</h3>${sharesBar(d.shares)}
      <h3>Квадраты с наибольшей плотностью</h3>${d.top_squares.map((q) => `<div class="row"><b>${q.square}</b><span class="meta">${m.countries[q.country].ru}</span><span style="margin-left:auto">${Math.round(q.p * 100)}%</span></div>`).join("")}
      <div class="note" style="margin-top:6px">${esc(st.source)}</div></div>
      <div class="box"><b>Обучение на наблюдениях</b><div class="meta">Логистическая регрессия по T, S, глубине на учётных съёмках: AUC на отложенных неделях ${m.habitat_fit.test_auc}; выученный оптимум солёности ${m.habitat_fit.learned_opt_sal ?? "—"}‰${m.habitat_fit.learned_opt_temp ? `, температуры ${m.habitat_fit.learned_opt_temp} °C` : ""}. ${m.habitat_fit.synthetic ? "Демо: съёмки синтетические — подставьте реальные траловые данные." : ""}</div></div>`;
  } else if (cas.layer === "risk") {
    const rd = m.poaching.days[cas.day - 1], pm = m.poaching.model;
    body = `<div class="box"><b>Горячие точки браконьерства</b> · ночь на ${dd}<div class="meta">Луна ${Math.round(rd.moon * 100)}% · ветер ${rd.wind} м/с</div>
      ${rd.hotspots.slice(0, 6).map((h) => `<div class="row" style="margin-top:4px"><b>${h.square}</b><span class="meta">${m.countries[h.country].ru}</span><span class="risk" style="margin-left:auto">${Math.round(h.risk * 100)}%</span></div><div class="meta">${esc(h.why.join("; "))}</div>`).join("")}</div>
      <div class="box"><b>Модель учится на истории</b><div class="meta">Задержания и инциденты прошлых лет (напр. итоги операции «Бекіре») + прогноз миграции. Проверка по времени: AUC ${pm.auc_test}; 5% площади моря с наибольшим риском покрывают ${Math.round(pm.hit_rate_top5 * 100)}% будущих инцидентов.
      ${m.poaching.synthetic_history ? "<br><b>Демо: история синтетическая</b> — метрики показывают работу конвейера, не реальную точность." : ""}</div>
      <ul class="reasons">${m.poaching.explain.map((x) => `<li><span>${esc(x)}</span></li>`).join("")}</ul>
      <label class="meta"><input type="checkbox" id="showInc" ${cas.showInc ? "checked" : ""}> показать инциденты за год</label></div>`;
  } else {
    body = `<div class="box"><b>${LAYERS[cas.layer]}</b> по данным ${m.sensors.length} датчиков (точки) поверх климатической основы<div class="cbar"></div>
      <div class="legend-row"><span>${cas.layer === "sal" ? "0‰" : "холоднее"}</span><span style="margin-left:auto">${cas.layer === "sal" ? "14‰" : "теплее"}</span></div>
      <div class="note">Датчики снижают ошибку восстановления поля: невязки датчиков интерполируются (IDW), вдали от них — климат.</div></div>`;
  }
  const msgs = cas.msgs.map((x) => `<div class="msg"><div class="row"><span class="prio ${x.priority === "high" ? "high" : "medium"}">${x.priority === "high" ? "ВЫСОК" : "СРЕДН"}</span>
      <b>→ ${m.countries[x.to.country].ru}</b><span class="status ${x.status}" style="margin-left:auto">${{ draft: "черновик", approved: "одобрено", rejected: "отклонено" }[x.status]}</span></div>
      <div class="text">${esc(x.text[cas.lang])}</div>
      <div class="meta">${esc(x.framework[cas.lang])} · ${esc(x.data_policy[cas.lang])} · SHA-256 ${x.sha256.slice(0, 12)}…</div>
      <div class="actions"><button class="btn ok" data-msg="${x.id}" data-st="approved">Одобрить</button><button class="btn bad" data-msg="${x.id}" data-st="rejected">Отклонить</button>
      <a class="btn" href="/api/transboundary/${x.id}.json">JSON</a></div></div>`).join("") || '<div class="empty">Уведомлений нет</div>';
  $("#panel-caspian").innerHTML = `<div class="seg" id="casLayer">${Object.entries(LAYERS).map(([k, v]) => `<button data-l="${k}" class="${k === cas.layer ? "on" : ""}">${v}</button>`).join("")}</div>
    ${maxDay ? `<input type="range" class="slider" id="casDay" min="${cas.layer === "risk" ? 1 : 0}" max="${maxDay}" value="${cas.day}"><div class="legend-row"><span>${cas.layer === "risk" ? "завтра" : "сегодня"}</span><span style="margin-left:auto">+${maxDay} сут</span></div>` : ""}
    ${body}
    <h3>Трансграничный протокол</h3>
    <div class="seg" id="casLang"><button data-lang="ru" class="${cas.lang === "ru" ? "on" : ""}">RU</button><button data-lang="en" class="${cas.lang === "en" ? "on" : ""}">EN</button></div>
    ${msgs}
    <div class="note">Черновики собираются автоматически из акустических тревог, прогноза миграции и браконьерства. Наружу ничего не уходит без решения диспетчера; канал передачи согласуют стороны. ${esc(m.note)}</div>`;
  $("#casLayer").querySelectorAll("button").forEach((b) => (b.onclick = () => { cas.layer = b.dataset.l; renderCaspian(); }));
  $("#casDay")?.addEventListener("input", (e) => { cas.day = +e.target.value; renderCaspian(); });
  $("#casLang").querySelectorAll("button").forEach((b) => (b.onclick = () => { cas.lang = b.dataset.lang; renderCaspian(); }));
  $("#showInc")?.addEventListener("change", (e) => { cas.showInc = e.target.checked; drawCaspianMap(); });
  $("#panel-caspian").querySelectorAll("[data-msg]").forEach((b) => (b.onclick = async () => {
    await api(`/api/transboundary/${b.dataset.msg}/status`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: b.dataset.st }) });
    cas.msgs = await api("/api/transboundary");
    renderCaspian();
  }));
  drawCaspianMap();
}

// ------------------------------------------------------------------ общее
function showTab(t) {
  const was = document.querySelector("#tabs button.on")?.dataset.tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `panel-${t}`));
  if (t === "forecast") renderForecast();
  if (t === "caspian") enterCaspian();
  else if (was === "caspian") leaveCaspian();
}

function renderHeader() {
  const ov = state.overview;
  $("#clock").textContent = `${ov.demo ? "демо · " : ""}${fmtTime(ov.scenario_time, true)}`;
  const n = { red: 0, yellow: 0, green: 0 };
  (ov.eco || []).forEach((e) => n[e.level]++);
  $("#ecoChips").innerHTML = (ov.eco || []).map((e) => `<span class="chip" title="${esc(e.reasons.join("; "))}"><span class="dot ${e.level}"></span>${esc(e.name.replace(" (демо)", ""))}</span>`).join("");
}

function toast(text, prio) {
  const t = $("#toast");
  t.style.borderLeftColor = PRIO_COLOR[prio] || "#667085";
  t.innerHTML = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4500);
}

async function refreshAlerts(freshId) {
  state.alerts = await api("/api/alerts");
  renderAlerts(freshId);
}

async function refreshOverview() {
  state.overview = await api("/api/overview");
  renderHeader();
  drawStatic(state.overview);
  renderEco();
  renderBuoys();
}

function connect() {
  const es = new EventSource("/api/stream");
  es.onopen = () => { $("#live").classList.add("on"); $("#live span").textContent = "онлайн"; };
  es.onerror = () => { $("#live").classList.remove("on"); $("#live span").textContent = "переподключение…"; };
  es.onmessage = async (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "alert") {
      await refreshAlerts(e.alert.id);
      const p = e.alert.fusion?.priority;
      toast(`<b>${PRIO[p] || ""} ${e.alert.fusion?.risk ?? ""}</b> · ${esc(e.alert.title)}<div class="meta">${SOURCE[e.alert.type] || ""} · ${fmtTime(e.alert.start_ts)}</div>`, p);
      if (["critical", "high"].includes(p) && e.alert.location) {
        const tr = e.alert.localization?.track;
        if (tr && tr.length > 1) map.flyToBounds(tr.map((q) => [q.lat, q.lon]), { maxZoom: 16, padding: [60, 60], duration: 0.8 });
        else map.flyTo([e.alert.location.lat, e.alert.location.lon], Math.max(map.getZoom(), 15), { duration: 0.8 });
      }
    } else if (e.type === "reset" || e.type === "merged") {
      closeAlert();
      await refreshAlerts();
    } else if (e.type === "status") {
      await refreshAlerts();
      refreshOverview();
    } else if (e.type === "water") {
      refreshOverview();
    } else if (e.type === "transboundary") {
      if (cas.meta) renderCaspian();
    } else if (e.type === "replay_done") {
      $("#replay").disabled = false;
      toast("Ночь проиграна: все тревоги в ленте", "info");
    }
  };
}

async function main() {
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#live-buoys").onclick = async () => {
    $("#live-buoys").disabled = true;
    showTab("alerts");
    closeAlert();
    await api("/api/demo/live", { method: "POST" });
    toast("Буи в эфире: 3 гидрофона проигрывают ночь в реальном времени (~2.5 мин). Тревоги придут по мере звука.", "info");
    setTimeout(() => ($("#live-buoys").disabled = false), 150000);
  };
  $("#replay").onclick = async () => {
    $("#replay").disabled = true;
    showTab("alerts");
    closeAlert();
    await api("/api/demo/replay", { method: "POST" });
  };
  if (!window.L) {
    $("#map").innerHTML = '<div class="empty">Не загрузилась библиотека карты (нет интернета?)</div>';
    return;
  }
  try {
    state.overview = await api("/api/overview");
  } catch (e) {
    $("#alertList").innerHTML = '<div class="empty">Нет данных. Запустите <code>make demo</code>.</div>';
    return;
  }
  initMap(state.overview.origin);
  await refreshOverview();
  await refreshAlerts();
  state.forecast = await api("/api/forecast").catch(() => null);
  const b = state.overview.buoys.map((x) => [x.lat, x.lon]);
  if (b.length) map.fitBounds(b, { padding: [120, 120] });
  connect();
}
main();
