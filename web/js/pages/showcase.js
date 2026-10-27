import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, waitForJob, pct0 } from '../core.js';
import { Card, Button, Tabs, Select, Switch, Spinner, ErrorBox, Notice, Collapse, Eyebrow, Progress } from '../ui.js';
import { Chart, palette, baseOption, axisStyle, isNarrow } from '../charts.js';

const nulls = (n) => Array(n).fill(null);
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const dayLabel = (v) => { const d = new Date(v); return Number.isNaN(d) ? String(v) : d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }); };
const longDate = (v) => new Date(v).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });

// ------------------------------------------------------------------ charts
function mainOption(p, d, showActual) {
  const nh = d.x_history.length, nf = d.x_future.length, x = [...d.x_history, ...d.x_future];
  const lo80 = d.bands['0.1'], hi80 = d.bands['0.9'], lo50 = d.bands['0.25'], hi50 = d.bands['0.75'], med = d.bands['0.5'];
  const band = (lo, hi, key, opacity) => [
    { type: 'line', data: [...nulls(nh), ...lo], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
    { type: 'line', data: [...nulls(nh), ...hi.map((v, i) => v - lo[i])], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: p.brand, opacity }, silent: true, tooltip: { show: false } },
  ];
  const last = d.history[d.history.length - 1];
  return {
    ...baseOption(p),
    grid: { left: 12, right: isNarrow() ? 40 : 52, top: 14, bottom: 26 },
    xAxis: { type: 'category', data: x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false }, axisLine: { show: false } }), axisLabel: { color: p.muted, fontSize: 11, hideOverlap: true, formatter: dayLabel } },
    yAxis: { type: 'value', position: 'right', scale: true, ...axisStyle(p, { axisLine: { show: false }, splitLine: { lineStyle: { color: p.grid, opacity: 0.5 } } }), axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => compact.format(v) } },
    tooltip: { ...baseOption(p).tooltip, valueFormatter: (v) => (v === null || v === undefined ? '' : compact.format(v)) },
    series: [
      { name: 'History', type: 'line', data: [...d.history, ...nulls(nf)], showSymbol: false, lineStyle: { width: 1.6, color: p.text }, itemStyle: { color: p.text }, z: 3,
        markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: p.axis, type: 'solid', opacity: 0.8 }, data: [{ xAxis: nh - 1 }] },
        markArea: { silent: true, itemStyle: { color: p.brand, opacity: 0.04 }, data: [[{ xAxis: nh - 1 }, { xAxis: x.length - 1 }]] } },
      ...(showActual ? [{ name: 'What really happened', type: 'line', data: [...nulls(nh - 1), last, ...d.actual], showSymbol: false, lineStyle: { width: 1.6, color: p.text, type: 'dashed', opacity: 0.85 }, itemStyle: { color: p.text }, z: 4 }] : []),
      ...band(lo80, hi80, 'b80', 0.16), ...band(lo50, hi50, 'b50', 0.28),
      { name: 'Forecast', type: 'line', data: [...nulls(nh), ...med], showSymbol: false, lineStyle: { width: 2.4, color: p.brand }, itemStyle: { color: p.brand }, z: 5 },
    ],
  };
}

function sparkOption(p, row, d) {
  const x = [...d.x_history, ...d.x_future], nh = d.x_history.length;
  const color = row.enabled ? p.text : p.muted;
  return {
    animation: false, backgroundColor: 'transparent', tooltip: { show: false },
    grid: { left: 12, right: isNarrow() ? 40 : 52, top: 4, bottom: 4 },
    xAxis: { type: 'category', data: x, show: false, boundaryGap: false },
    yAxis: { type: 'value', show: false, scale: true, min: row.binary ? -0.15 : undefined, max: row.binary ? 1.15 : undefined },
    series: [{ type: 'line', data: row.values, showSymbol: false, step: row.binary ? 'end' : false, lineStyle: { width: row.binary ? 1.6 : 1.2, color, opacity: row.enabled ? 0.9 : 0.35 },
      areaStyle: row.binary ? undefined : { color, opacity: row.enabled ? 0.12 : 0.05 },
      markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: p.axis, opacity: 0.8 }, data: [{ xAxis: nh - 1 }] } }],
  };
}

// ------------------------------------------------------------------ page
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

function Effect({ sc, impact, loading, progress, covOn, onFocus }) {
  if (loading) return html`<${Progress} done=${progress.done} total=${progress.total} message=${progress.message || 'Measuring on past windows (first time only)…'} />`;
  if (!impact) return null;
  const rows = sc.covariates.map((c) => ({ ...c, gain: impact.gain_each[c.key] }));
  const bar = (g) => `width:${Math.max(0, Math.min(100, (g || 0) * 100 * 1.6))}%`;
  const d = impact.detail;
  return html`<div class="grid gap-5">
    <p class="max-w-3xl text-body-large">Given <b>all</b> the factors, the model's error score drops by <b class="text-surface-brand num">${pct0(impact.gain_all)}</b> compared with using the history alone.
      Measured on <b class="num">${impact.forecasts}</b> past forecasts (${impact.series} ${impact.series === 1 ? 'series' : 'series'} × the ${impact.windows} most recent windows of ${sc.horizon_label}).</p>
    <div class="grid gap-2">${rows.map((r) => html`<div class="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3 text-body-small"><span>${r.label}</span>
      <span class="h-2 rounded-sm bg-surface-brand-tertiary"><span class="block h-full rounded-sm bg-surface-brand" style=${bar(r.gain)}></span></span><span class="num text-right">${r.gain === null ? '–' : (r.gain >= 0 ? '+' : '') + Math.round(r.gain * 100) + '%'}</span></div>`)}
      <p class="text-body-small text-text-muted">Each bar: the improvement when only that one factor is given.</p></div>
    <p class="text-body-small text-text-muted">The shaded 80% range contained what really happened <b class="num text-text-primary">${pct0(d.all.coverage_80)}</b> of the time with the factors (<b class="num text-text-primary">${pct0(d.none.coverage_80)}</b> without). A trustworthy range would be close to 80%${Math.min(d.all.coverage_80, d.none.coverage_80) < 0.7 ? ': here it is too narrow, so treat the shaded band as optimistic.' : '.'}</p></div>`;
}

export function Showcase() {
  const { device } = useStore();
  const [list] = useAsync(() => api('/api/showcase'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [reloadKey, setReloadKey] = useState(0);
  const [tab, setTab] = useState('sales');
  const [series, setSeries] = useState({}), [back, setBack] = useState(0), [on, setOn] = useState({});
  const [model, setModel] = useState(null), [reveal, setReveal] = useState(false);
  const [data, setData] = useState(null), [err, setErr] = useState(null), [busy, setBusy] = useState(false), [wait, setWait] = useState('');
  const [impact, setImpact] = useState(null), [impLoading, setImpLoading] = useState(false), [prog, setProg] = useState({});
  const seq = useRef(0), impJob = useRef(null);
  const [fresh, refetch] = useAsync(() => api('/api/showcase'), [reloadKey]);
  const scs = fresh.data || list.data || [];
  const sc = scs.find((s) => s.id === tab);
  const ready = (models.data || []).filter((m) => m.kind === 'foundation' && m.cache.downloaded);
  useEffect(() => { if (!model && ready.length) setModel((ready.find((m) => m.id === 'timestral-28m') || ready[0]).id); }, [models.data]);
  const sid = sc ? (series[sc.id] || sc.default_series) : null;
  const enabled = sc ? sc.covariates.filter((c) => on[sc.id + c.key] !== false).map((c) => c.key) : [];
  const stepDays = sc && sc.freq.endsWith('H') ? 24 : 1;
  const maxDays = sc ? Math.floor(sc.max_offset_steps / stepDays) : 0;

  // interactive forecast (debounced; stale answers ignored; waits politely while a measurement job holds the compute)
  useEffect(() => {
    if (!sc || !sc.ready || !model) return;
    const mine = ++seq.current;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        for (let attempt = 0; attempt < 40; attempt++) {
          try {
            const r = await api('/api/showcase/forecast', { method: 'POST', body: { scenario: sc.id, series: sid, offset: back * stepDays, model, device, covariates: enabled } });
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
    }, 220);
    return () => clearTimeout(t);
  }, [sc && sc.id, sc && sc.ready, sid, back, enabled.join(','), model, device]);

  // measured effect over many windows (cached on disk after the first run)
  const firstShown = !!(data && sc && data.scenario === sc.id);
  useEffect(() => {
    if (!sc || !sc.ready || !model || !firstShown) return;
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

  if (list.loading && !scs.length) return html`<${Spinner} />`;
  if (list.error) return html`<div class="pt-10"><${ErrorBox} error=${list.error} /></div>`;
  const allOn = sc && enabled.length === sc.covariates.length;
  const setAll = (v) => setOn((o) => ({ ...o, ...Object.fromEntries(sc.covariates.map((c) => [sc.id + c.key, v])) }));
  const modelOpts = ready.map((m) => ({ value: m.id, label: m.name }));

  return html`<div>
    <header class="pt-8 pb-6 md:pt-14 md:pb-10 border-b border-border-primary">
      <${Eyebrow}>Showcase · real data, forecast on your machine<//>
      <h1 class="page-title mt-2">See what the model can do</h1>
      <p class="mt-3 max-w-3xl text-body-large text-text-muted">Pick a real situation. The white line is what happened; the orange line is what Timestral predicts <i>without ever seeing</i> what comes next. Give it the factors that are known in advance, and see how much closer it gets.</p>
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
          ${sc.series.length > 1 && html`<div class="flex items-center gap-2"><span class="label !mb-0">${sc.series_label}</span><${Select} compact class="w-44" value=${sid} onChange=${(v) => { setSeries({ ...series, [sc.id]: v }); }} options=${sc.series.map((s) => ({ value: s.id, label: s.label }))} aria-label=${sc.series_label} /></div>`}</div>

        <div class="mt-4 grid gap-x-4 md:grid-cols-[11rem_minmax(0,1fr)_3.25rem]"><div class="hidden md:block"></div><div class="relative min-w-0">
          ${err && html`<${ErrorBox} error=${err} />`}
          ${data && data.scenario === sc.id && html`<${Chart} build=${(p) => mainOption(p, data, reveal)} deps=${[data, reveal]} height=${isNarrow() ? 230 : 300} name=${sc.title + ' forecast'} />`}
          ${(!data || data.scenario !== sc.id) && !err && html`<${Spinner} label="Forecasting…" />`}
          ${busy && data && html`<span class="absolute right-2 top-1 rounded-sm bg-surface-brand-tertiary px-2 py-0.5 font-mono text-[0.6875rem] uppercase text-text-muted">updating…</span>`}
        ${wait && html`<p class="mt-2 text-body-small text-text-muted">${wait}</p>`}
        <div class="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-body-small text-text-muted"><span class="inline-flex items-center gap-2"><i class="inline-block h-0.5 w-5 bg-text-primary"></i>History</span><span class="inline-flex items-center gap-2"><i class="inline-block h-0.5 w-5 bg-surface-brand"></i>Forecast · ${sc.horizon_label}</span><span class="inline-flex items-center gap-2"><i class="inline-block h-3 w-5 rounded-sm bg-surface-brand opacity-30"></i>Likely range (50% and 80%)</span></div></div><div class="hidden md:block"></div></div>

        <div class="mt-6 grid items-center gap-x-4 gap-y-1 border-t border-border-primary pt-5 md:grid-cols-[11rem_minmax(0,1fr)_3.25rem]">
          <div><div class="font-medium">Forecast starts</div><div class="num text-body-small text-text-muted">${data && data.scenario === sc.id ? longDate(data.start) : ''}</div></div>
          <input type="range" class="h-10 w-full" min="0" max=${maxDays} value=${maxDays - back} aria-label="Forecast start date" onInput=${(e) => setBack(maxDays - Number(e.currentTarget.value))} /><div class="hidden md:block"></div></div>

        <div class="mt-6 grid gap-0 border-t border-border-primary">
          <div class="flex flex-wrap items-center justify-between gap-3 py-4"><span class="font-medium">Factors known in advance</span>
            <div class="flex items-center gap-4">${impact && html`<span class="text-body-small"><b class="text-surface-brand num">${pct0(impact.gain_all)}</b> <span class="text-text-muted">lower error with all of them</span></span>`}
              ${impLoading && html`<span class="text-body-small text-text-muted">measuring…</span>`}
              <${Switch} checked=${allOn} onChange=${setAll} label=${allOn ? 'All on' : 'All off'} /></div></div>
          ${sc.covariates.map((c) => { const row = data && data.scenario === sc.id ? data.covariates.find((r) => r.key === c.key) : null; const g = impact && impact.gain_each[c.key];
            return html`<div class="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-t border-border-primary py-3 md:grid-cols-[11rem_minmax(0,1fr)_3.25rem]">
              <div class="order-1 min-w-0"><div class="font-medium">${c.label}</div><div class="truncate text-body-small text-text-muted" title=${c.hint}>${g === null || g === undefined ? c.hint : `alone: ${g >= 0 ? '+' : ''}${Math.round(g * 100)}%`}</div></div>
              <div class="order-3 col-span-2 min-w-0 md:order-2 md:col-span-1">${row && html`<${Chart} build=${(p) => sparkOption(p, row, data)} deps=${[data]} height=${46} name=${c.label} />`}</div>
              <div class="order-2 justify-self-end md:order-3"><${Switch} checked=${on[sc.id + c.key] !== false} onChange=${(v) => setOn({ ...on, [sc.id + c.key]: v })} label=${c.label} hideLabel /></div></div>`; })}</div>
      </div>`}
    </div>

    ${sc && sc.ready && html`<div class="mt-10 grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <${Card}><${Eyebrow}>How much do the factors really help?<//><div class="mt-4"><${Effect} sc=${sc} impact=${impact} loading=${impLoading} progress=${prog} /></div>
        <p class="mt-5 border-t border-border-primary pt-4 text-body-small text-text-muted">“Error score” is a quantile loss: it measures how close the forecast is <i>and</i> how honest its shaded range is. Lower is better. It is computed here on this machine for the model and device selected above, and stored so the second time is instant.</p><//>
      <${Card}><${Eyebrow}>Read this before trusting a number<//>
        <ul class="mt-4 list-disc space-y-3 pl-5 text-body-small text-text-muted">${sc.caveats.map((c) => html`<li>${c}</li>`)}
          <li>These are public datasets: Timestral's training data is not disclosed, so it may have seen them. Results on your own data are the real test.</li></ul>
        <div class="mt-4"><${Collapse} title="Data source and license"><p class="text-body-small text-text-muted"><span class="font-mono">${sc.source.repo}</span> · ${sc.source.name}/${sc.source.freq} · revision <span class="font-mono">${sc.source.revision.slice(0, 10)}</span><br />License: ${sc.source.license}.</p><//></div><//>
    </div>`}
  </div>`;
}
