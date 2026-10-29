import { html, useState } from '/vendor/htm-preact.js';
import { api, useAsync, go, num, pct, pct0, ms, when } from '../core.js';
import { Card, PageHeader, Section, Button, Tabs, Select, Field, Spinner, ErrorBox, Notice, Table, Tag, Eyebrow, PageError } from '../ui.js';
import { Figure, barOption, baseOption, axisStyle, isNarrow } from '../charts.js';

const LETTERS = 'ABCDEF';
const METRICS = { mase: 'MASE', sql: 'SQL', wql: 'WQL' };

function entriesOf(runs) {
  return runs.flatMap((r, ri) => Object.entries(r.models).map(([mid, m]) => ({
    key: `${r.id}:${mid}`, run: r, runLetter: LETTERS[ri], mid, m, label: `${LETTERS[ri]} · ${m.name} · ${m.device}`,
  })));
}

const fmt = (v) => (v === null || v === undefined ? '–' : typeof v === 'object' ? JSON.stringify(v) : String(v));

function ConfigDiff({ runs }) {
  const rows = [
    ['Run', (r) => r.label || r.dataset.name], ['When', (r) => when(r.created)], ['Main model', (r) => (r.request || {}).model],
    ['Device', (r) => (r.request || {}).device], ['Dataset', (r) => `${r.dataset.name} (${r.dataset.n_series} series)`],
    ['Horizon', (r) => r.config.horizon], ['Windows', (r) => r.config.n_windows], ['Stride', (r) => r.config.stride],
    ['Max context', (r) => r.config.max_context], ['Covariates', (r) => r.config.use_covariates], ['Options', (r) => r.config.options],
    ['Baselines', (r) => ((r.request || {}).baselines || []).join(', ')], ['Forecasts per model', (r) => r.items],
  ].map(([k, f]) => { const vals = runs.map((r) => fmt(f(r))); return { k, vals, differs: new Set(vals).size > 1 }; });
  return html`<${Table} head=${['', ...runs.map((_, i) => `Run ${LETTERS[i]}`)]} rows=${rows.map((r) => [html`<span class=${r.differs ? 'font-medium' : 'text-text-muted'}>${r.k}${r.differs ? html` <${Tag}>differs<//>` : ''}</span>`, ...r.vals.map((v) => html`<span class=${r.differs ? 'text-text-primary' : 'text-text-muted'}>${v}</span>`)])} />`;
}

function Paired({ entries, runs }) {
  // default: the main model of run A against the main model of run B
  const [a, setA] = useState(entries[0].key), [b, setB] = useState((entries.find((e) => e.run.id !== entries[0].run.id) || entries[1] || entries[0]).key), [metric, setMetric] = useState('mase');
  const [d] = useAsync(() => api(`/api/compare/paired?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&metric=${metric}`), [a, b, metric]);
  const opts = entries.map((e) => ({ value: e.key, label: e.label }));
  const la = (entries.find((e) => e.key === a) || {}).label, lb = (entries.find((e) => e.key === b) || {}).label;
  const build = (p) => { const x = d.data.a, y = d.data.b, pts = x.map((v, i) => [v, y[i]]).filter(([u, w]) => u !== null && w !== null), hi = Math.max(...pts.flat());
    return { ...baseOption(p), tooltip: { show: false }, grid: { left: isNarrow() ? 48 : 60, right: 20, top: 16, bottom: 48 },
      xAxis: { type: 'value', min: 0, name: la, nameLocation: 'middle', nameGap: 30, nameTextStyle: { color: p.muted }, ...axisStyle(p, { splitLine: { show: false } }) },
      yAxis: { type: 'value', min: 0, name: lb, nameTextStyle: { color: p.muted, align: 'left' }, ...axisStyle(p) },
      series: [{ type: 'scatter', data: pts, symbolSize: 7, itemStyle: { color: p.brand, opacity: 0.6 } }, { type: 'line', data: [[0, 0], [hi, hi]], showSymbol: false, lineStyle: { color: p.axis, type: 'dashed' } }] }; };
  return html`<div class="grid gap-4">
    <div class="grid gap-3 md:grid-cols-3"><${Field} label="X axis"><${Select} value=${a} onChange=${setA} options=${opts} /><//><${Field} label="Y axis"><${Select} value=${b} onChange=${setB} options=${opts} /><//>
      <div class="flex items-end"><${Tabs} value=${metric} onChange=${setMetric} label="Metric" items=${[{ value: 'mase', label: 'MASE' }, { value: 'sql', label: 'SQL' }]} /></div></div>
    ${d.loading ? html`<${Spinner} />` : d.error ? html`<${ErrorBox} error=${d.error} />` : html`
      <${Figure} title=${`${METRICS[metric]} per forecast`} subtitle=${`Each dot is one series window. Below the dashed line, ${lb} is better.`} build=${build} deps=${[d.data]} filename="paired" />
      <${Card} tight><div class="grid grid-cols-3 gap-4 text-center"><div><${Eyebrow}>Windows<//><div class="mt-1 font-mistral text-h4 num">${d.data.n}</div></div>
        <div><${Eyebrow}>Y better on<//><div class="mt-1 font-mistral text-h4 num">${pct0(d.data.b_better_share)}</div></div>
        <div><${Eyebrow}>Typical gain of Y<//><div class="mt-1 font-mistral text-h4 num">${pct(d.data.skill_b_vs_a.skill)}</div></div></div>
        <p class="mt-3 text-body-small text-text-muted">“Typical gain” is 1 − geometric mean of the per-window ratio Y ÷ X (positive = Y has lower error). The share of windows won matters as much as the average: a model can win on average yet lose on most windows.</p><//>`}</div>`;
}

export function Compare({ route }) {
  const ids = (route.query.runs || '').split(',').filter(Boolean);
  const [c] = useAsync(() => (ids.length >= 2 ? api(`/api/compare?runs=${ids.join(',')}`) : Promise.resolve(null)), [route.query.runs]);
  const [metric, setMetric] = useState('mase');
  if (ids.length < 2) return html`<div><${PageHeader} eyebrow="Compare" title="Compare runs">Pick two or more runs in the list to put their results side by side.<//><div class="mt-8"><a class="btn btn-brand" href="#/runs">Choose runs</a></div></div>`;
  if (c.loading) return html`<${Spinner} />`;
  if (c.error) return html`<${PageError} error=${c.error} eyebrow="Compare" title="These runs could not be compared" back="runs" backLabel="All runs" />`;
  const { runs, comparable } = c.data, entries = entriesOf(runs);
  const S = (e, k) => (e.m.summary[k] || {}).mean;
  const rows = entries.map((e) => [html`<b>${e.label}</b>`, num(S(e, 'mase')), num(S(e, 'sql')), num(S(e, 'wql')), pct0(S(e, 'coverage_80')), pct0(S(e, 'coverage_90')), ms(e.m.compute_ms), e.m.warmup_ms == null ? '–' : ms(e.m.warmup_ms), e.m.peak_vram_mb ? `${num(e.m.peak_vram_mb, 0)} MB` : '–']);
  const build = (p) => barOption(p, entries.map((e) => ({ id: e.key, label: e.label, value: S(e, metric) })), { focus: entries[0].key });
  return html`<div><${PageHeader} eyebrow="Compare" title=${`${runs.length} runs side by side`}>Runs are labelled A, B, C… in the order chosen. Differences in configuration are flagged so you know what changed.<//>
    <div class="mt-5 flex flex-wrap gap-2"><${Button} small kind="outline" onClick=${() => go('runs')}>Back to runs<//>${runs.map((r, i) => html`<a class="btn btn-outline btn-small" href=${'#/runs/' + r.id}>Open run ${LETTERS[i]}</a>`)}</div>
    <div class="mt-6 grid gap-3">
      ${!comparable.same_dataset && html`<${Notice} kind="warning" title="Different datasets">The runs used different datasets, so their numbers are not directly comparable.<//>`}
      ${comparable.same_dataset && !comparable.same_horizon && html`<${Notice} kind="warning" title="Different horizons">Errors grow with the horizon: compare only runs with the same horizon.<//>`}
      ${comparable.same_dataset && comparable.same_horizon && !comparable.same_items && html`<${Notice} kind="info" title="Different windows">The runs did not forecast exactly the same series and windows, so the per-forecast comparison is unavailable.<//>`}</div>
    <${Section} title="What changed"><${Card} class="!p-3 md:!p-5"><${ConfigDiff} runs=${runs} /><//><//>
    <${Section} title="Results" aside=${html`<${Tabs} value=${metric} onChange=${setMetric} label="Metric" items=${Object.entries(METRICS).map(([value, label]) => ({ value, label }))} />`}>
      <${Figure} title=${`${METRICS[metric]}, mean over all forecasts`} subtitle="Lower is better ↓ · the first entry is highlighted" build=${build} deps=${[c.data, metric]} height=${Math.max(180, entries.length * 44 + 40)} filename=${'compare-' + metric} />
      <${Card} class="mt-4 !p-3 md:!p-5"><${Table} head=${['Run · model · device', 'MASE', 'SQL', 'WQL', 'Cover 80%', 'Cover 90%', 'Compute', 'Cold start', 'Peak VRAM']} rows=${rows} /><//><//>
    ${comparable.same_items && html`<${Section} title="Forecast by forecast"><${Paired} entries=${entries} runs=${runs} /><//>`}
  </div>`;
}
