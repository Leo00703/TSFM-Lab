import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, num, ms, pct0, deviceLabel } from '../core.js';
import { Card, PageHeader, Section, Button, Field, Select, NumberInput, Check, Notice, Spinner, ErrorBox, Table, DeviceSelect, Tabs, Eyebrow } from '../ui.js';
import { Figure, fanOption, baseOption, axisStyle, isNarrow } from '../charts.js';
import { DatasetPicker, specOf, modelOptions, comparable } from './shared.js';

export function ModelChoice({ models, value, onChange }) {
  return html`<${Select} value=${value} onChange=${onChange} options=${modelOptions(models)} aria-label="Model" />`;
}

const BASELINES = [['seasonal-naive', 'Seasonal naive'], ['naive', 'Naive'], ['drift', 'Drift']];
export const ToggleList = ({ items, value, onChange }) => html`<div class="flex flex-wrap gap-x-5">${items.map(([v, l]) => html`<${Check} checked=${value.includes(v)} onChange=${(c) => onChange(c ? [...value, v] : value.filter((x) => x !== v))}>${l}<//>`)}</div>`;

export function Playground({ route }) {
  const { device } = useStore();
  const [ds] = useAsync(() => api('/api/datasets'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [pick, setPick] = useState({ key: route.query.ds || 'synthetic:multi_seasonality', params: {} });
  const [info, setInfo] = useState(null);
  const [c, setC] = useState({ series: 0, horizon: 48, origin: null, model: 'timestral-5m', baselines: ['seasonal-naive'], context: null, covariates: true, symmetric: true, maskFlat: false, clip: true });
  const [res, setRes] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState(null);
  const [view, setView] = useState({ variate: 0, bands: [0.9, 0.5], hidden: [] });
  const set = (p) => setC((x) => ({ ...x, ...p }));
  const out = useRef(null);

  useEffect(() => {
    let alive = true;
    setInfo(null);
    api('/api/datasets/inspect', { method: 'POST', body: specOf(pick.key, pick.params) }).then((i) => { if (alive) { setInfo(i); set({ series: 0, origin: null }); } }, (e) => alive && setError(e));
    return () => { alive = false; };
  }, [pick.key, JSON.stringify(pick.params)]);

  const len = info ? info.series[Math.min(c.series, info.series.length - 1)].length : 0;
  const origin = c.origin ?? Math.max(16, len - c.horizon);
  const run = async () => {
    setBusy(true); setError(null);
    try {
      const r = await api('/api/forecast', { method: 'POST', body: { dataset: specOf(pick.key, pick.params), series: c.series, origin, horizon: c.horizon, model: c.model, device, baselines: c.baselines,
        context_length: c.context, use_covariates: c.covariates, options: { symmetric_averaging: c.symmetric, mask_flat_covariates: c.maskFlat, clip_nonnegative: c.clip } } });
      setRes(r); setView((v) => ({ ...v, variate: 0 }));
      if (isNarrow()) setTimeout(() => out.current && out.current.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80); // phone: jump to the result
    } catch (e) { setError(e); } finally { setBusy(false); }
  };

  const fanData = res && { xHistory: res.x_history, xFuture: res.x_future, history: res.history, actual: res.actual,
    models: Object.fromEntries(Object.entries(res.models).map(([id, m]) => [id, { quantiles: m.quantiles, levels: m.levels, name: m.name }])) };
  const ids = res ? Object.keys(res.models) : [];
  const shown = ids.filter((i) => !view.hidden.includes(i));
  const hasActual = res && res.actual[0].some((v) => v !== null);
  const buildCov = (p) => { const k = res.known, n = k[0].length, x = [...res.x_history, ...res.x_future].slice(-n);
    return { ...baseOption(p), legend: { top: 0, textStyle: { color: p.text } }, grid: { left: isNarrow() ? 44 : 56, right: 16, top: 36, bottom: 28 },
      xAxis: { type: 'category', data: x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.muted, hideOverlap: true, formatter: (v) => String(v).replace('T', ' ').slice(0, 10) } }, yAxis: { type: 'value', scale: true, ...axisStyle(p) },
      series: k.map((row, i) => ({ name: res.known_names[i], type: 'line', step: 'end', data: row, showSymbol: false, lineStyle: { width: 1.4, color: p.series[(i + 2) % p.series.length] } })) }; };
  const metricCols = ['mase', 'sql', 'bias', 'coverage_80', 'coverage_90'];

  return html`<div><${PageHeader} eyebrow="Single forecast" title="Playground">Choose a series, where the forecast starts, and a model. The chart shows the history, the real values that followed (when they exist) and nested prediction intervals.<//>
    <div class="mt-8 grid gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <${Card} class="!p-5 h-fit"><div class="grid gap-5">
        <${DatasetPicker} datasets=${ds.data || []} value=${pick} onChange=${setPick} />
        ${info && html`<div class="grid gap-4 border-t border-border-primary pt-5">
          <${Field} label=${`Series (${info.n_series})`}><${Select} value=${String(c.series)} onChange=${(v) => set({ series: Number(v), origin: null })} options=${info.series.slice(0, 300).map((s, i) => ({ value: String(i), label: `${s.id} · ${s.length} steps` }))} /><//>
          <div class="grid grid-cols-2 gap-3"><${Field} label="Horizon"><${NumberInput} value=${c.horizon} min=${1} max=${4096} onChange=${(v) => set({ horizon: v || 1, origin: null })} /><//>
            <${Field} label="Context (steps)" hint="blank = all"><${NumberInput} value=${c.context} min=${8} placeholder="all" onChange=${(v) => set({ context: v })} /><//></div>
          <${Field} label=${`Forecast starts at step ${origin} of ${len}`} hint=${origin + c.horizon > len ? 'Beyond the data: no real values to compare with.' : `${len - origin} steps of real data follow.`}>
            <input type="range" class="w-full h-10" min=${Math.min(16, len)} max=${len} value=${origin} onInput=${(e) => set({ origin: Number(e.currentTarget.value) })} aria-label="Forecast origin" /><//></div>`}
        <div class="grid gap-4 border-t border-border-primary pt-5">
          <${Field} label="Model"><${ModelChoice} models=${models.data} value=${c.model} onChange=${(v) => set({ model: v, baselines: c.baselines.filter((b) => b !== v) })} /><//>
          <${DeviceSelect} class="flex-col !items-stretch" />
          <div><span class="label">Compare with</span><${ToggleList} items=${[...BASELINES, ...comparable(models.data, c.model)]} value=${c.baselines} onChange=${(v) => set({ baselines: v })} /></div></div>
        <div class="border-t border-border-primary pt-4">
          ${info && (info.n_known > 0 || info.n_past_covariates > 0) && html`<${Check} checked=${c.covariates} onChange=${(v) => set({ covariates: v })} hint="Known-future and past-only columns of this dataset">Use covariates<//>`}
          <${Check} checked=${c.symmetric} onChange=${(v) => set({ symmetric: v })} hint="Averages the forecast with that of the mirrored series (Timestral; the closest equivalent for TimesFM)">Symmetric averaging<//>
          <${Check} checked=${c.maskFlat} onChange=${(v) => set({ maskFlat: v })} hint="Workaround for a Timestral bug with constant covariates (upstream issue #7)">Mask flat covariates<//>
          <${Check} checked=${c.clip} onChange=${(v) => set({ clip: v })} hint="Clamp at 0 when the history is non-negative">Clip non-negative<//></div>
        <${Button} kind="brand" onClick=${run} disabled=${busy || !info}>${busy ? 'Forecasting…' : 'Run forecast'}<//>
      </div><//>

      <div ref=${out} class="min-w-0 scroll-mt-16 grid grid-cols-[minmax(0,1fr)] content-start gap-4">
        ${error && html`<${ErrorBox} error=${error} />`}
        ${!res && !busy && html`<${Card}><p class="text-text-muted">Set things up on the left and press <b class="text-text-primary">Run forecast</b>. The model will run on <b class="text-text-primary">${deviceLabel(device)}</b>.</p><//>`}
        ${busy && html`<${Card}><${Spinner} label=${`Running on ${deviceLabel(device)}… the first call after loading a model is slower`} /><//>`}
        ${res && html`
          <${Figure} title=${`${res.series_id} · ${res.horizon}-step forecast`} subtitle=${`Context ${res.context_used} steps · ${hasActual ? 'black line = what actually happened' : 'no real values after this point'}`}
            build=${(p) => fanOption(p, fanData, { variate: view.variate, focus: c.model, shown, bands: view.bands, targetName: res.target_names[view.variate] })} deps=${[res, view]} height=${isNarrow() ? 300 : 400} filename=${'forecast-' + res.series_id}
            actions=${res.target_names.length > 1 ? html`<${Select} compact class="w-36" aria-label="Target" value=${String(view.variate)} onChange=${(v) => setView({ ...view, variate: Number(v) })} options=${res.target_names.map((n, i) => ({ value: String(i), label: n }))} />` : null}
            note=${html`<span class="flex flex-wrap items-center gap-x-5">Intervals: ${[0.98, 0.9, 0.8, 0.5].map((b) => html`<label class="inline-flex items-center gap-1.5 min-h-8"><input type="checkbox" class="accent-[var(--color-surface-brand)]" checked=${view.bands.includes(b)} onChange=${(e) => setView({ ...view, bands: e.currentTarget.checked ? [...view.bands, b] : view.bands.filter((x) => x !== b) })} />${Math.round(b * 100)}%</label>`)}
              <span class="text-text-muted">·</span> Models: ${ids.map((i) => html`<label class="inline-flex items-center gap-1.5 min-h-8"><input type="checkbox" class="accent-[var(--color-surface-brand)]" checked=${!view.hidden.includes(i)} onChange=${(e) => setView({ ...view, hidden: e.currentTarget.checked ? view.hidden.filter((x) => x !== i) : [...view.hidden, i] })} />${res.models[i].name}</label>`)}</span>`} />
          ${res.known && html`<${Figure} title="Known covariates" subtitle="Values given to the model for the context and the horizon" build=${buildCov} deps=${[res]} height=${200} filename="covariates" />`}
          <${Card}><${Eyebrow}>Results for this window<//>
            ${hasActual ? html`<div class="mt-3"><${Table} head=${['Model', 'Device', 'MASE', 'SQL', 'Bias', 'Cover 80%', 'Cover 90%', 'Forecast', 'Notes']} rows=${ids.map((i) => { const m = res.models[i], x = m.metrics || {}, t = m.timings;
              return [html`<b>${m.name}</b>`, m.device, num(x.mase), num(x.sql), num(x.bias), pct0(x.coverage_80), pct0(x.coverage_90), ms(t.forecast_ms), t.first_call_after_load ? `first call · load ${t.model_load_s}s` : ''];
            })} /></div><p class="mt-3 text-body-small text-text-muted">One window is anecdote, not evidence: use <a class="underline" href="#/experiments">Experiments</a> for many windows. MASE and SQL are scaled by the history's naive error; lower is better. Bias is positive when the median is too high.</p>`
            : html`<div class="mt-3"><${Table} head=${['Model', 'Device', 'Forecast', 'Notes']} rows=${ids.map((i) => { const m = res.models[i], t = m.timings; return [html`<b>${m.name}</b>`, m.device, ms(t.forecast_ms), t.first_call_after_load ? `first call · load ${t.model_load_s}s` : '']; })} /></div>`}<//>
          ${ids.flatMap((i) => res.models[i].warnings.map((w) => html`<${Notice} kind="warning" title=${res.models[i].name}>${w}<//>`))}`}
      </div></div></div>`;
}
