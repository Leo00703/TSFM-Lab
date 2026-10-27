import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, waitForJob, pct0 } from '../core.js';
import { Card, Button, Tabs, Select, Switch, Spinner, ErrorBox, Notice, Collapse, Eyebrow, Progress } from '../ui.js';
import { Logo } from '../icons.js';
import { Chart, isNarrow, baseOption, axisStyle } from '../charts.js';

const nulls = (n) => Array(n).fill(null);
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const num2 = new Intl.NumberFormat('en', { maximumFractionDigits: 1 });
const fmtV = (v, binary) => (v === null || v === undefined || Number.isNaN(v) ? '–' : binary ? (v >= 0.5 ? 'Yes' : 'No') : Math.abs(v) >= 100 ? compact.format(v) : num2.format(v));
const dayLabel = (v) => { const d = new Date(v); return Number.isNaN(d) ? String(v) : d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }); };
const longDate = (v) => new Date(v).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
/** A number of steps as a plain-language span: hours/days for hourly data, days/weeks for daily, weeks for weekly. */
const span = (sc, n) => sc.freq.endsWith('H') ? (n >= 48 ? Math.round(n / 24) + ' days' : n + ' hours')
  : sc.freq.endsWith('W') ? n + ' weeks' : (n >= 14 ? Math.round(n / 7) + ' weeks' : n + ' days');
const fullDate = (v, hourly) => { const d = new Date(v); return Number.isNaN(d) ? String(v) : d.toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', ...(hourly ? { hour: '2-digit', minute: '2-digit' } : {}) }); };

// Plot margins shared by the main chart and every factor row, so all time axes line up exactly.
const MARGIN = () => ({ left: 12, right: isNarrow() ? 40 : 52 });

// ------------------------------------------------------------------ charts
function mainOption(p, d, showActual) {
  const w = d.window, k = d.k, h = d.horizon, m = MARGIN();
  const lo80 = d.bands['0.1'], hi80 = d.bands['0.9'], lo50 = d.bands['0.25'], hi50 = d.bands['0.75'], med = d.bands['0.5'];
  const band = (lo, hi, key, opacity) => [
    { type: 'line', data: [...nulls(k), ...lo], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
    { type: 'line', data: [...nulls(k), ...hi.map((v, i) => v - lo[i])], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: p.brand, opacity }, silent: true, tooltip: { show: false } },
  ];
  return {
    ...baseOption(p),
    grid: { left: m.left, right: m.right, top: 40, bottom: 28 },
    xAxis: { type: 'category', data: d.x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false }, axisLine: { show: false } }), axisLabel: { color: p.text, fontSize: 12, hideOverlap: true, formatter: dayLabel }, axisPointer: { show: true, label: { show: false } } },
    yAxis: { type: 'value', position: 'right', scale: true, ...axisStyle(p, { axisLine: { show: false }, splitLine: { lineStyle: { color: p.grid, opacity: 0.5 } } }), axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => compact.format(v) } },
    tooltip: { trigger: 'axis', showContent: false, axisPointer: { type: 'line', lineStyle: { color: p.text, opacity: 0.55, width: 1 } } },
    series: [
      { name: 'History', type: 'line', data: [...d.real.slice(0, k), ...nulls(h)], showSymbol: false, lineStyle: { width: 1.7, color: p.text }, itemStyle: { color: p.text }, z: 3,
        markArea: { silent: true, itemStyle: { color: p.brand, opacity: 0.05 }, data: [[{ xAxis: k - 1 }, { xAxis: w - 1 }]] } },
      ...(showActual ? [{ name: 'What really happened', type: 'line', data: [...nulls(k - 1), ...d.real.slice(k - 1)], showSymbol: false, lineStyle: { width: 1.5, color: p.muted, opacity: 0.8 }, itemStyle: { color: p.muted }, z: 4 }] : []),
      ...band(lo80, hi80, 'b80', 0.16), ...band(lo50, hi50, 'b50', 0.28),
      { name: 'Forecast', type: 'line', data: [...nulls(k), ...med], showSymbol: false, lineStyle: { width: 2.4, color: p.brand }, itemStyle: { color: p.brand }, z: 5 },
    ],
  };
}

function sparkOption(p, row, d) {
  const k = d.k, m = MARGIN(), v = row.values;
  const color = row.enabled ? p.text : p.muted, op = row.enabled ? 1 : 0.4;
  const line = (data, opacity) => ({ type: 'line', data, showSymbol: false, step: row.binary ? 'end' : false, lineStyle: { width: row.binary ? 1.6 : 1.2, color, opacity: opacity * 0.9 },
    areaStyle: row.binary ? undefined : { color, opacity: opacity * 0.12 }, silent: true });
  return {
    animation: false, backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', showContent: false, axisPointer: { type: 'line', lineStyle: { color: p.text, opacity: 0.55, width: 1 } } },
    grid: { left: m.left, right: m.right, top: 4, bottom: 4 },
    xAxis: { type: 'category', data: d.x, show: false, boundaryGap: false },
    yAxis: { type: 'value', show: false, scale: true, min: row.binary ? -0.15 : undefined, max: row.binary ? 1.15 : undefined },
    // before the divider = what the model has seen; after it = what it is told about the future (dimmer)
    series: [line([...v.slice(0, k), ...nulls(v.length - k)], op), line([...nulls(k - 1), ...v.slice(k - 1)], op * 0.55)],
  };
}

// ------------------------------------------------------------------ pieces
function Prepare({ sc, onReady }) {
  const [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const go = async () => {
    setBusy(true); setErr(null);
    try { const j = await api(`/api/showcase/${sc.id}/prepare`, { method: 'POST' }); const done = await waitForJob(j.id); if (done.status === 'error') throw new Error(done.error); onReady(); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  };
  return html`<div class="grid gap-4 py-8 md:py-12"><div class="max-w-xl">
    <h3 class="font-mistral text-h3">${sc.title}</h3><p class="mt-2 text-text-muted">${sc.story}</p>
    <p class="mt-4 text-body-small text-text-muted">This scenario uses real data (<span class="font-mono">${sc.source.name}</span>, ${sc.size_mb} MB) from the public <span class="font-mono">${sc.source.repo}</span> collection. It is downloaded once into this project's cache.</p>
    <div class="mt-5 flex items-center gap-3"><${Button} kind="brand" disabled=${busy} onClick=${go}>${busy ? 'Downloading…' : `Download data (${sc.size_mb} MB)`}<//></div>
    ${err && html`<div class="mt-4"><${ErrorBox} error=${err} /></div>`}</div></div>`;
}

function Effect({ sc, impact, loading, progress }) {
  if (loading) return html`<${Progress} done=${progress.done} total=${progress.total} message=${progress.message || 'Measuring on past windows (first time only)…'} />`;
  if (!impact) return null;
  const rows = sc.covariates.map((c) => ({ ...c, gain: impact.gain_each[c.key] }));
  const bar = (g) => `width:${Math.max(0, Math.min(100, (g || 0) * 100 * 1.6))}%`;
  const d = impact.detail;
  return html`<div class="grid gap-5">
    <p class="max-w-3xl text-body-large">Given <b>all</b> the factors, the model's error score ${impact.gain_all >= 0 ? 'drops' : 'goes up'} by <b class=${'num ' + (impact.gain_all >= 0 ? 'text-surface-brand' : '')}>${pct0(Math.abs(impact.gain_all))}</b> compared with using the history alone${impact.gain_all < 0 ? ': with this model the factors do more harm than good here' : ''}.
      Measured on <b class="num">${impact.forecasts}</b> past forecasts (${impact.series} series × the ${impact.windows} most recent windows of ${sc.horizon_label}).</p>
    <div class="grid gap-2">${rows.map((r) => html`<div class="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3 text-body-small"><span>${r.label}</span>
      <span class="h-2 rounded-sm bg-surface-brand-tertiary"><span class="block h-full rounded-sm bg-surface-brand" style=${bar(r.gain)}></span></span><span class="num text-right">${r.gain === null ? '–' : (r.gain >= 0 ? '+' : '') + Math.round(r.gain * 100) + '%'}</span></div>`)}
      <p class="text-body-small text-text-muted">Each bar: the improvement when only that one factor is given.</p></div>
    <p class="text-body-small text-text-muted">The shaded 80% range contained what really happened <b class="num text-text-primary">${pct0(d.all.coverage_80)}</b> of the time with the factors (<b class="num text-text-primary">${pct0(d.none.coverage_80)}</b> without). A trustworthy range would be close to 80%${Math.min(d.all.coverage_80, d.none.coverage_80) < 0.7 ? ': here it is too narrow, so treat the shaded band as optimistic.' : '.'}</p></div>`;
}

// ------------------------------------------------------------------ page
export function Showcase() {
  const { device } = useStore();
  const [list] = useAsync(() => api('/api/showcase'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [reloadKey, setReloadKey] = useState(0);
  const [fresh] = useAsync(() => api('/api/showcase'), [reloadKey]);
  const [tab, setTab] = useState('sales');
  const [series, setSeries] = useState({}), [cuts, setCuts] = useState({}), [on, setOn] = useState({});
  const [model, setModel] = useState(null), [reveal, setReveal] = useState(true), [dragging, setDragging] = useState(false), [hover, setHover] = useState(null);
  const [data, setData] = useState(null), [err, setErr] = useState(null), [busy, setBusy] = useState(false), [wait, setWait] = useState('');
  const [impact, setImpact] = useState(null), [impLoading, setImpLoading] = useState(false), [prog, setProg] = useState({});
  const seq = useRef(0), impJob = useRef(null), box = useRef(null), dataRef = useRef(null);

  const scs = fresh.data || list.data || [];
  const sc = scs.find((s) => s.id === tab);
  const ready = (models.data || []).filter((m) => m.kind === 'foundation' && m.cache.downloaded);
  useEffect(() => { if (!model && ready.length) setModel((ready.find((m) => m.id === 'timestral-28m') || ready[0]).id); }, [models.data]);
  const sid = sc ? (series[sc.id] || sc.default_series) : null;
  const cut = sc ? (cuts[sc.id] ?? sc.k_default) : 0;
  const enabled = sc ? sc.covariates.filter((c) => on[sc.id + c.key] !== false).map((c) => c.key) : [];
  const hourly = !!(sc && sc.freq.endsWith('H'));
  dataRef.current = data;
  const setCut = (k) => sc && setCuts((c) => ({ ...c, [sc.id]: Math.max(sc.k_min, Math.min(sc.k_max, Math.round(k))) }));

  // interactive forecast: follows the divider while it is dragged (short debounce), ignores stale answers, and waits
  // politely while a measurement job holds the compute
  useEffect(() => {
    if (!sc || !sc.ready || !model) return;
    const mine = ++seq.current;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        for (let attempt = 0; attempt < 40; attempt++) {
          try {
            const r = await api('/api/showcase/forecast', { method: 'POST', body: { scenario: sc.id, series: sid, cut, model, device, covariates: enabled } });
            if (mine === seq.current) { setData(r); setErr(null); setWait(''); }
            return;
          } catch (e) {
            if (!/job is running/.test(e.message)) throw e;
            setWait('Measuring the effect of the factors… the chart updates as soon as that finishes.');
            await new Promise((res) => setTimeout(res, 1200));
            if (mine !== seq.current) return;
          }
        }
      } catch (e) { if (mine === seq.current) setErr(e); } finally { if (mine === seq.current) setBusy(false); }
    }, dragging ? 90 : 200);
    return () => clearTimeout(t);
  }, [sc && sc.id, sc && sc.ready, sid, cut, enabled.join(','), model, device]);

  // measured effect over many windows (cached on disk after the first run); starts after the first chart is shown
  const firstShown = !!(data && sc && data.scenario === sc.id);
  useEffect(() => {
    if (!sc || !sc.ready || !model || !firstShown) { setImpact(null); return; }  // never show another scenario's numbers
    let alive = true;
    setImpact(null); setImpLoading(true); setProg({});
    if (impJob.current) api(`/api/jobs/${impJob.current}/cancel`, { method: 'POST' }).catch(() => {});
    api('/api/showcase/impact', { method: 'POST', body: { scenario: sc.id, model, device } }).then(async (j) => {
      impJob.current = j.id;
      const done = await waitForJob(j.id, (x) => alive && setProg({ done: x.done, total: x.total, message: x.message }));
      if (alive && done.status === 'done') setImpact(done.result);
    }).catch(() => {}).finally(() => alive && setImpLoading(false));
    return () => { alive = false; };
  }, [sc && sc.id, sc && sc.ready, model, device, firstShown]);

  // ---- the curtain: the divider moves freely over a FIXED window (left = what really happened, right = what is predicted)
  const fromPointer = (e) => {
    const r = box.current.getBoundingClientRect();
    const idx = Math.round(((e.clientX - r.left) / r.width) * (sc.window - 1));
    setCut(idx + 1);
  };
  const onDown = (e) => { e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); setDragging(true); fromPointer(e); };
  const onMove = (e) => { if (dragging) fromPointer(e); };
  const onUp = () => setDragging(false);
  const onKey = (e) => {
    const weekly = sc.freq.endsWith('W');
    const step = hourly ? 6 : 1, big = hourly ? 24 : weekly ? 4 : 7;
    const d = { ArrowLeft: -step, ArrowDown: -step, ArrowRight: step, ArrowUp: step, PageDown: -big, PageUp: big }[e.key];
    if (d !== undefined) { e.preventDefault(); setCut(cut + d); } else if (e.key === 'Home') { e.preventDefault(); setCut(sc.k_min); } else if (e.key === 'End') { e.preventDefault(); setCut(sc.k_max); }
  };

  // ---- synchronised crosshair + read-out of date and values
  const onChartReady = (c) => {
    c.group = 'showcase';
    c.on('updateAxisPointer', (ev) => { const v = ev.axesInfo && ev.axesInfo[0] && ev.axesInfo[0].value; const d = dataRef.current; if (v === undefined || !d) return; setHover(typeof v === 'number' ? Math.round(v) : d.x.indexOf(v)); });
    c.getZr().on('globalout', () => setHover(null));
    setTimeout(() => window.echarts && echarts.connect('showcase'), 80);
  };
  const valid = !!(data && sc && data.scenario === sc.id);
  const readout = (() => {
    if (hover === null || !valid || hover < 0 || hover >= data.x.length) return null;
    const j = hover - data.k;
    return { when: fullDate(data.x[hover], hourly), future: j >= 0, real: j < 0 ? data.real[hover] : (reveal ? data.real[hover] : null),
      med: j >= 0 ? data.bands['0.5'][j] : null, lo: j >= 0 ? data.bands['0.1'][j] : null, hi: j >= 0 ? data.bands['0.9'][j] : null,
      covs: data.covariates.map((r) => ({ label: r.label, v: r.values[hover], binary: r.binary, on: r.enabled })) };
  })();

  if (list.loading && !scs.length) return html`<${Spinner} />`;
  if (list.error) return html`<div class="pt-10"><${ErrorBox} error=${list.error} /></div>`;
  const allOn = sc && enabled.length === sc.covariates.length;
  const setAll = (v) => setOn((o) => ({ ...o, ...Object.fromEntries(sc.covariates.map((c) => [sc.id + c.key, v])) }));
  const modelOpts = ready.map((m) => ({ value: m.id, label: m.name }));
  const m = MARGIN();
  const pos = sc ? ((cut - 1) / (sc.window - 1)) * 100 : 0;

  return html`<div>
    <section class="pt-10 pb-8 md:pt-20 md:pb-14 border-b border-border-primary">
      <div class="flex items-center gap-4"><${Logo} size=${56} /><${Eyebrow}>Timestral · by Mistral AI<//></div>
      <h1 class="mt-5 max-w-4xl font-mistral text-h1 font-medium md:text-[3.5rem] md:leading-[3.75rem]">A forecasting model that runs on your own machine.</h1>
      <p class="mt-5 max-w-3xl text-body-large text-text-muted">Timestral is a family of small AI models, from 5 million to 209 million parameters, that predict how a number will evolve over time: sales, electricity demand, solar output. It was never trained on the series you give it, and it answers with a range of likely outcomes, not a single guess. This app runs it locally, so you can see how good it really is on real data.</p>
      <p class="mt-4 max-w-3xl text-body-large text-text-muted">Start with the examples below. When you want to look under the hood, the <a class="underline underline-offset-4" href="#/playground">Playground</a> forecasts your own files and the <a class="underline underline-offset-4" href="#/experiments">Experiments</a> page measures accuracy against simple baselines.</p>
    </section>

    <header class="pt-8 pb-6 md:pt-14 md:pb-10 border-b border-border-primary">
      <${Eyebrow}>Showcase · real data, forecast on your machine<//>
      <h2 class="page-title mt-2">See what the model can do</h2>
      <p class="mt-3 max-w-3xl text-body-large text-text-muted">Pick a real situation. Drag the handle: everything on its left is what happened, everything on its right is what Timestral predicts <i>without ever seeing</i> it. Give it the factors that are known in advance, and see how much closer it gets.</p>
    </header>

    <div class="mt-8 rounded-xl border border-border-primary bg-surface-brand-secondary">
      <div class="flex flex-wrap items-center justify-between gap-3 border-b border-border-primary p-4 md:px-8">
        <${Tabs} value=${tab} onChange=${setTab} label="Scenario" items=${scs.map((s) => ({ value: s.id, label: s.tab }))} />
        <div class="flex flex-wrap items-center gap-4">
          ${ready.length > 0 && html`<div class="flex items-center gap-2"><span class="label !mb-0">Model</span><${Select} compact class="w-40" value=${model || ''} onChange=${setModel} options=${modelOpts} aria-label="Model" /></div>`}
          <${Switch} checked=${reveal} onChange=${setReveal} label="Show what really happened" /></div></div>

      ${!sc ? null : !sc.ready ? html`<div class="px-4 md:px-8"><${Prepare} sc=${sc} onReady=${() => setReloadKey((k) => k + 1)} /></div>` : html`
      <div class="p-4 md:p-8">
        ${ready.length === 0 && html`<${Notice} kind="warning" title="No model downloaded">Download a Timestral model on the <a class="underline" href="#/models">Models</a> page to run forecasts.<//>`}
        <div class="flex flex-wrap items-end justify-between gap-4">
          <div><h2 class="font-mistral text-h3">${sc.title}</h2><p class="text-body-small text-text-muted">${sc.subtitle}${sc.unit ? ' · ' + sc.unit : ''}</p></div>
          ${sc.series.length > 1 && html`<div class="flex items-center gap-2"><span class="label !mb-0">${sc.series_label}</span><${Select} compact class="w-44" value=${sid} onChange=${(v) => setSeries({ ...series, [sc.id]: v })} options=${sc.series.map((s) => ({ value: s.id, label: s.label }))} aria-label=${sc.series_label} /></div>`}</div>

        <div class="relative mt-4">
          ${err && html`<${ErrorBox} error=${err} />`}
          ${valid && html`<${Chart} build=${(p) => mainOption(p, data, reveal)} deps=${[data, reveal]} height=${isNarrow() ? 260 : 340} name=${sc.title + ' forecast'} onInstance=${onChartReady} />`}
          ${!valid && !err && html`<${Spinner} label="Forecasting…" />`}
          ${valid && html`<div ref=${box} class="pointer-events-none absolute inset-y-0 z-10" style=${`left:${m.left}px;right:${m.right}px`}>
            <div class="absolute inset-y-0 w-0" style=${`left:${pos}%`}>
              <span class="absolute bottom-7 top-10 w-px -translate-x-1/2 bg-text-primary opacity-60"></span>
              <button type="button" role="slider" tabindex="0" aria-label="Where the forecast starts: drag, or use the arrow keys" aria-valuemin=${sc.k_min} aria-valuemax=${sc.k_max} aria-valuenow=${cut} aria-valuetext=${data.x[Math.min(cut, data.x.length - 1)] ? longDate(data.x[Math.min(cut, data.x.length - 1)]) : ''}
                class="pointer-events-auto absolute left-0 top-1 grid h-8 w-6 -translate-x-1/2 cursor-ew-resize touch-none place-items-center rounded-md bg-text-primary text-text-invert shadow-md"
                onPointerDown=${onDown} onPointerMove=${onMove} onPointerUp=${onUp} onPointerCancel=${onUp} onKeyDown=${onKey}><span class="h-3.5 w-0.5 bg-current"></span></button>
              <span class="pointer-events-none absolute left-4 top-2 whitespace-nowrap text-body-small font-medium text-surface-brand">${dragging && data.x[cut] ? dayLabel(data.x[cut]) : 'Forecast'}</span></div></div>`}
          ${busy && valid && html`<span class="absolute right-2 top-12 rounded-sm bg-surface-brand-tertiary px-2 py-0.5 font-mono text-[0.6875rem] uppercase text-text-muted">updating…</span>`}</div>
        ${wait && html`<p class="mt-2 text-body-small text-text-muted">${wait}</p>`}
        <div class="mt-2 flex flex-wrap items-center gap-x-6 gap-y-2 text-body-small text-text-muted"><span class="inline-flex items-center gap-2"><i class="inline-block h-0.5 w-5 bg-text-primary"></i>History</span><span class="inline-flex items-center gap-2"><i class="inline-block h-0.5 w-5 bg-surface-brand"></i>Forecast${valid ? ' · ' + span(sc, data.horizon) : ''}</span><span class="inline-flex items-center gap-2"><i class="inline-block h-3 w-5 rounded-sm bg-surface-brand opacity-30"></i>Likely range (50% and 80%)</span></div>

        <div class="mt-4 grid gap-x-6 gap-y-1 border-t border-border-primary pt-4 md:grid-cols-[11rem_minmax(0,1fr)]">
          <div><div class="font-medium">Forecast starts</div><div class="num text-body-small text-text-muted">${valid ? longDate(data.start) : ''}</div></div>
          <div class="min-h-[3.25rem] min-w-0 text-body-small">${readout ? html`<div class="flex flex-wrap items-baseline gap-x-5 gap-y-1"><b class="num">${readout.when}</b>
              ${readout.real !== null && readout.real !== undefined && html`<span><span class="text-text-muted">${readout.future ? 'Really happened' : 'Real'}</span> <b class="num">${fmtV(readout.real)}</b></span>`}
              ${readout.future && html`<span class="text-surface-brand"><span>Forecast</span> <b class="num">${fmtV(readout.med)}</b> <span class="num text-text-muted">(likely ${fmtV(readout.lo)}–${fmtV(readout.hi)})</span></span>`}</div>
              <div class="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-text-muted">${readout.covs.map((c) => html`<span class=${c.on ? '' : 'opacity-50'}>${c.label} <b class="num text-text-primary">${fmtV(c.v, c.binary)}</b></span>`)}</div>`
            : html`<span class="text-text-muted">Drag the handle along the chart (or use the arrow keys) to choose where the forecast starts: the further left, the longer the forecast. Move the pointer over the chart to read the date and the values.</span>`}</div></div>

        <div class="mt-4 border-t border-border-primary">
          <div class="flex flex-wrap items-center justify-between gap-3 py-4"><span class="font-medium">Factors known in advance</span>
            <div class="flex items-center gap-4">${impact && html`<span class="text-body-small"><b class=${'num ' + (impact.gain_all >= 0 ? 'text-surface-brand' : '')}>${pct0(Math.abs(impact.gain_all))}</b> <span class="text-text-muted">${impact.gain_all >= 0 ? 'lower' : 'higher'} error with all of them</span></span>`}
              ${impLoading && html`<span class="text-body-small text-text-muted">measuring…</span>`}
              <${Switch} checked=${allOn} onChange=${setAll} label=${allOn ? 'All on' : 'All off'} /></div></div>
          ${sc.covariates.map((c) => { const row = valid ? data.covariates.find((r) => r.key === c.key) : null; const g = impact && impact.gain_each[c.key];
            return html`<div class="border-t border-border-primary py-3">
              <div class="flex items-center justify-between gap-3"><div class="min-w-0"><span class="font-medium">${c.label}</span>
                <span class="ml-2 text-body-small text-text-muted" title=${c.hint}>${g === null || g === undefined ? '' : `alone: ${g >= 0 ? '+' : ''}${Math.round(g * 100)}%`}</span></div>
                <${Switch} checked=${on[sc.id + c.key] !== false} onChange=${(v) => setOn({ ...on, [sc.id + c.key]: v })} label=${c.label} hideLabel /></div>
              <div class="min-w-0">${row && html`<${Chart} build=${(p) => sparkOption(p, row, data)} deps=${[data]} height=${46} name=${c.label} onInstance=${(c2) => { c2.group = 'showcase'; setTimeout(() => window.echarts && echarts.connect('showcase'), 100); }} />`}</div></div>`; })}</div>
      </div>`}
    </div>

    ${sc && sc.ready && html`<div class="mt-10 grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <${Card}><${Eyebrow}>How much do the factors really help?<//><div class="mt-4"><${Effect} sc=${sc} impact=${impact} loading=${impLoading} progress=${prog} /></div>
        <p class="mt-5 border-t border-border-primary pt-4 text-body-small text-text-muted">“Error score” is a quantile loss: it measures how close the forecast is <i>and</i> how honest its shaded range is. Lower is better. It is computed here on this machine for the model and device selected above, on the default ${sc.horizon_label}, and stored so the second time is instant.</p><//>
      <${Card}><${Eyebrow}>Read this before trusting a number<//>
        <ul class="mt-4 list-disc space-y-3 pl-5 text-body-small text-text-muted">${sc.caveats.map((c) => html`<li>${c}</li>`)}
          <li>These are public datasets: Timestral's training data is not disclosed, so it may have seen them. Results on your own data are the real test.</li></ul>
        <div class="mt-4"><${Collapse} title="Data source and license"><p class="text-body-small text-text-muted"><span class="font-mono">${sc.source.repo}</span> · ${sc.source.name}/${sc.source.freq} · revision <span class="font-mono">${sc.source.revision.slice(0, 10)}</span><br />License: ${sc.source.license}.</p><//></div><//>
    </div>`}
  </div>`;
}
