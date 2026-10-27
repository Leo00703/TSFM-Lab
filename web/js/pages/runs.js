import { html, useState } from '/vendor/htm-preact.js';
import { api, useAsync, go, num, pct, pct0, ms, when } from '../core.js';
import { Card, PageHeader, Section, Button, Tabs, Select, Field, Spinner, ErrorBox, Notice, Table, KV, Collapse, Tag, Eyebrow, download } from '../ui.js';
import { confirmDialog } from '../dialog.js';
import { Figure, barOption, linesOption, fanOption, baseOption, axisStyle, modelColors, quantileStats, isNarrow } from '../charts.js';

const METRICS = { mase: 'MASE', sql: 'SQL', wql: 'WQL' };

function RunList() {
  const [s, reload] = useAsync(() => api('/api/runs'), []);
  const [sel, setSel] = useState([]);
  const toggle = (id) => setSel((x) => (x.includes(id) ? x.filter((i) => i !== id) : x.length < 6 ? [...x, id] : x));
  const del = async (r) => {
    const ok = await confirmDialog({ title: 'Delete this run?', confirmLabel: 'Delete run', busyLabel: 'Deleting…', run: () => api(`/api/runs/${r.id}`, { method: 'DELETE' }),
      body: 'The run and all of its stored forecasts are deleted. This cannot be undone.' });
    if (ok) reload();
  };
  return html`<div><${PageHeader} eyebrow="History" title="Runs">Every experiment is stored with its configuration, environment and raw forecasts, so any chart can be redrawn and any number traced back.<//>
    <${Section}>${s.loading ? html`<${Spinner} />` : s.error ? html`<${ErrorBox} error=${s.error} />` : !s.data.length ? html`<${Card}><p class="text-text-muted">No runs yet. <a class="underline" href="#/experiments">Start an experiment</a>.</p><//>` : html`
      <div class="mb-3 flex flex-wrap items-center gap-3"><${Button} kind="brand" disabled=${sel.length < 2} onClick=${() => go('compare?runs=' + sel.join(','))}>Compare selected (${sel.length})<//><span class="text-body-small text-text-muted">Tick two or more runs of the same dataset and horizon (up to 6).</span></div>
      <${Card} class="!p-3 md:!p-5"><${Table} head=${['', 'When', 'Dataset', 'Models', 'Device', 'Horizon', 'Items', 'MASE (main model)', '']} rows=${s.data.map((r) => [html`<input type="checkbox" class="size-4 accent-[var(--color-surface-brand)]" aria-label="Select run to compare" checked=${sel.includes(r.id)} onChange=${() => toggle(r.id)} />`, when(r.created), r.dataset_name, r.models.join(', '), r.devices.join(', '), r.horizon, r.n_items,
        num((r.headline[r.models[0]] || {}).mase), html`<span class="flex gap-2"><${Button} small kind="outline" onClick=${() => go('runs/' + r.id)}>Open<//><${Button} small kind="danger" icon="trash" aria-label="Delete run" onClick=${() => del(r)} /></span>`])} /><//>`}<//></div>`;
}

function Distribution({ an, ids, names, focus, metric }) {
  const build = (p) => { const colors = modelColors(ids, focus, p);
    return { ...baseOption(p), tooltip: { trigger: 'item', confine: true, backgroundColor: p.bg, textStyle: { color: p.text } }, grid: { left: isNarrow() ? 44 : 56, right: 16, top: 16, bottom: 36 },
      xAxis: { type: 'category', data: ids.map((i) => names[i]), ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.text, fontSize: 11, interval: 0, width: 70, overflow: 'truncate' } }, yAxis: { type: 'value', min: 0, ...axisStyle(p) },
      series: [{ type: 'boxplot', data: ids.map((i) => ({ value: quantileStats(an.distributions[i][metric] || []), itemStyle: { color: colors[i] + '33', borderColor: colors[i], borderWidth: 1.6 } })) }] }; };
  return html`<${Figure} title=${`${METRICS[metric]} across series and windows`} subtitle="Box = quartiles, whiskers = 1.5×IQR. Outliers are clipped, never hidden: see the items table." build=${build} deps=${[an, metric, focus]} filename=${'distribution-' + metric} />`;
}

function ItemExplorer({ id, meta, focus }) {
  const [items] = useAsync(() => api(`/api/runs/${id}/items`), [id]);
  const [sel, setSel] = useState(null), [metric, setMetric] = useState('mase'), [view] = useState({ bands: [0.9, 0.5] });
  const [item] = useAsync(() => (sel === null ? Promise.resolve(null) : api(`/api/runs/${id}/item/${sel}`)), [id, sel]);
  const rows = (items.data || []).slice().sort((a, b) => ((b.metrics[focus] || {})[metric] ?? -1) - ((a.metrics[focus] || {})[metric] ?? -1)).slice(0, 40);
  const d = item.data;
  const fan = d && { xHistory: d.history[0].map((_, i) => d.origin - d.history[0].length + i), xFuture: d.actual[0].map((_, i) => d.origin + i), history: d.history, actual: d.actual,
    models: Object.fromEntries(Object.entries(d.models).map(([i, m]) => [i, { quantiles: m.quantiles, levels: m.levels, name: m.name }])) };
  return html`<${Section} title="Look at individual forecasts" aside=${html`<span class="text-body-small text-text-muted">worst ${rows.length} by ${METRICS[metric] || metric}, main model</span>`}>
    <div class="mb-4"><${Tabs} value=${metric} onChange=${setMetric} label="Sort by" items=${[{ value: 'mase', label: 'MASE' }, { value: 'sql', label: 'SQL' }, { value: 'bias', label: 'Bias' }]} /></div>
    <div class="grid gap-4 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <${Card} class="!p-3 max-h-[28rem] overflow-y-auto">${items.loading ? html`<${Spinner} />` : html`<table class="data-table"><thead><tr><th>Series</th><th>Win</th><th>${METRICS[metric] || metric}</th><th></th></tr></thead><tbody>
        ${rows.map((r) => html`<tr class=${sel === r.i ? 'bg-action-ghost-1' : ''}><td>${r.series}</td><td class="num">${r.window}</td><td class="num">${num((r.metrics[focus] || {})[metric])}</td><td><${Button} small kind="outline" onClick=${() => setSel(r.i)}>View<//></td></tr>`)}</tbody></table>`}<//>
      <div class="min-w-0">${sel === null ? html`<${Card}><p class="text-text-muted">Select a row to see that window’s forecast with all models.</p><//>` : item.loading || !d ? html`<${Spinner} />` : html`
        <${Figure} title=${`${d.series} · window ${d.window}`} subtitle=${`Forecast starts at step ${d.origin}${d.origin_time ? ' (' + String(d.origin_time).replace('T', ' ').slice(0, 16) + ')' : ''}`}
          build=${(p) => fanOption(p, fan, { variate: 0, focus, shown: Object.keys(fan.models), bands: view.bands })} deps=${[d]} height=${isNarrow() ? 280 : 360} filename=${`item-${d.i}`} />`}</div></div><//>`;
}

function RunDetail({ id }) {
  const [m, reloadMeta] = useAsync(() => api(`/api/runs/${id}`), [id]);
  const meta = m.data;
  const ids = meta ? Object.keys(meta.models) : [];
  const focus = ids[0];
  const [rerun, setRerun] = useState(null);
  const [metric, setMetric] = useState('mase'), [hm, setHm] = useState('mase'), [dm, setDm] = useState('mase'), [aModel, setAModel] = useState(null);
  const model = aModel || focus;
  const [an] = useAsync(() => (meta ? api(`/api/runs/${id}/analysis?model=${model}`) : Promise.resolve(null)), [id, model, !!meta]);
  if (m.loading) return html`<${Spinner} />`;
  if (m.error) return html`<div class="pt-10"><${ErrorBox} error=${m.error} /></div>`;

  const names = Object.fromEntries(ids.map((i) => [i, meta.models[i].name || i]));
  const S = (i, k) => (meta.models[i].summary[k] || {}).mean;
  const baseline = ids.includes('seasonal-naive') ? 'seasonal-naive' : ids.find((i) => meta.models[i].kind === 'baseline');
  const horizonLen = meta.config.horizon;
  const warns = ids.flatMap((i) => [...meta.models[i].warnings.map((w) => [i, w, 'warning']), ...meta.models[i].notes.map((w) => [i, w, 'info'])]);
  const rows = ids.map((i) => { const mm = meta.models[i], sk = baseline && i !== baseline ? (mm.skill[baseline] || {}).mase : null, load = mm.load || {};
    return [html`<b>${names[i]}</b>${i === focus ? html` <${Tag}>main<//>` : ''}`, mm.device, num(S(i, 'mase')), num(S(i, 'sql')), num(S(i, 'wql')), num(S(i, 'bias')), pct0(S(i, 'coverage_80')), pct0(S(i, 'coverage_90')),
      sk ? pct(sk.skill) : '–', ms(mm.batch_ms.reduce((a, b) => a + b, 0)), mm.warmup_ms == null ? '–' : ms(mm.warmup_ms), mm.peak_vram_mb ? `${num(mm.peak_vram_mb, 0)} MB` : '–']; });

  const barBuild = (p) => barOption(p, ids.map((i) => ({ id: i, label: names[i], value: S(i, metric) })), { focus, digits: 3 });
  const horizonBuild = (p) => linesOption(p, { x: Array.from({ length: horizonLen }, (_, k) => k + 1), xName: 'steps ahead', yName: hm.startsWith('coverage') ? 'share inside' : METRICS[hm] || hm, yMin: hm === 'bias' ? undefined : 0, focus,
    series: ids.map((i) => ({ id: i, label: names[i], data: meta.models[i].horizon[hm] })), ...(hm.startsWith('coverage') ? { diagonal: Array(horizonLen).fill(Number(hm.split('_')[1]) / 100) } : {}) });
  const calBuild = (p) => ({ ...linesOption(p, { x: meta.models[focus].levels.map(String), xName: 'predicted quantile', yName: 'share of actuals below', yMin: 0, focus, diagonal: meta.models[focus].levels,
    series: ids.map((i) => ({ id: i, label: names[i], data: meta.models[i].calibration.empirical })) }) });
  const covBuild = (p) => linesOption(p, { x: ['50%', '80%', '90%', '98%'], xName: 'nominal interval', yName: 'share inside', yMin: 0, focus, diagonal: [0.5, 0.8, 0.9, 0.98],
    series: ids.map((i) => ({ id: i, label: names[i], data: [S(i, 'coverage_50'), S(i, 'coverage_80'), S(i, 'coverage_90'), S(i, 'coverage_98')] })) });
  const scatterBuild = (p) => { const s = an.data.scatter, lo = Math.min(...s.actual, ...s.predicted), hi = Math.max(...s.actual, ...s.predicted);
    return { ...baseOption(p), tooltip: { show: false }, grid: { left: isNarrow() ? 48 : 60, right: 20, top: 16, bottom: 44 },
      xAxis: { type: 'value', name: 'actual', nameLocation: 'middle', nameGap: 28, scale: true, ...axisStyle(p, { splitLine: { show: false } }) }, yAxis: { type: 'value', name: 'median forecast', scale: true, nameTextStyle: { color: p.muted, align: 'left' }, ...axisStyle(p) },
      series: [{ type: 'scatter', data: s.actual.map((a, k) => [a, s.predicted[k]]), symbolSize: 4, itemStyle: { color: p.brand, opacity: 0.45 } }, { type: 'line', data: [[lo, lo], [hi, hi]], showSymbol: false, lineStyle: { color: p.axis, type: 'dashed' } }] }; };
  const histBuild = (p) => { const h = an.data.residual_hist, c = h.counts, e = h.edges;
    return { ...baseOption(p), tooltip: { trigger: 'axis', confine: true, backgroundColor: p.bg, textStyle: { color: p.text } }, grid: { left: isNarrow() ? 44 : 56, right: 16, top: 16, bottom: 44 },
      xAxis: { type: 'category', data: c.map((_, k) => ((e[k] + e[k + 1]) / 2).toFixed(2)), name: 'error in units of history scale', nameLocation: 'middle', nameGap: 28, ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.muted, hideOverlap: true } }, yAxis: { type: 'value', min: 0, ...axisStyle(p) },
      series: [{ type: 'bar', data: c, itemStyle: { color: p.brand }, barCategoryGap: '8%' }] }; };

  return html`<div><${PageHeader} eyebrow=${`Run · ${when(meta.created)}`} title=${meta.label || meta.dataset.name}>
      ${meta.dataset.name} · ${meta.dataset.n_series} series · horizon ${horizonLen} · ${meta.config.n_windows} windows · ${meta.items.series.length} forecasts per model<//>
    <div class="mt-5 flex flex-wrap gap-2"><a class="btn btn-outline btn-small" href=${`/api/runs/${id}/export.csv`} download>Export CSV</a><a class="btn btn-outline btn-small" href=${`/api/runs/${id}`} target="_blank" rel="noopener">Raw JSON</a>
      ${meta.request && html`<${Button} small kind="outline" icon="refresh" disabled=${rerun === 'started'} onClick=${async () => { setRerun(true); try { await api('/api/jobs/experiment', { method: 'POST', body: { ...meta.request, label: (meta.label || meta.dataset.name) + ' (re-run)' } }); setRerun('started'); } catch (e) { setRerun(String(e.message)); } }}>Re-run with the same settings<//>${rerun === 'started' ? html`<span class="text-body-small text-text-muted">Started: follow it in the Activity panel (bottom right) and open the run when it finishes.</span>` : rerun ? html`<span class="text-body-small text-text-destructive">${rerun}</span>` : null}`}
      <${Button} small kind="danger" icon="trash" onClick=${async () => { if (await confirmDialog({ title: 'Delete this run?', confirmLabel: 'Delete run', busyLabel: 'Deleting…', run: () => api(`/api/runs/${id}`, { method: 'DELETE' }), body: 'The run and all of its stored forecasts are deleted. This cannot be undone.' })) go('runs'); }}>Delete<//>
      <${Button} small kind="outline" onClick=${() => go('runs')}>All runs<//></div>

    <${Section} title="Summary">
      <${Card} class="!p-3 md:!p-5"><${Table} head=${['Model', 'Device', 'MASE', 'SQL', 'WQL', 'Bias', 'Cover 80%', 'Cover 90%', baseline ? `Skill vs ${names[baseline]}` : 'Skill', 'Compute', 'Cold start', 'Peak VRAM']} rows=${rows} /><//>
      <p class="mt-3 max-w-4xl text-body-small text-text-muted">Lower is better for MASE, SQL and WQL; coverage should match its label (an 80% interval should contain about 80% of the actuals). Skill = 1 − geometric mean of per-item MASE ratios (positive = better than the baseline). “Compute” is the warm model time summed over batches; the cold start (first call: CUDA init, kernel compilation) is reported separately. Definitions on the <a class="underline" href="#/about">About</a> page.</p>
      ${warns.length > 0 && html`<div class="mt-4 grid gap-2">${warns.map(([i, w, k]) => html`<${Notice} kind=${k} title=${names[i]}>${w}<//>`)}</div>`}<//>

    <${Section} title="Accuracy" aside=${html`<${Tabs} value=${metric} onChange=${setMetric} label="Metric" items=${Object.entries(METRICS).map(([v, label]) => ({ value: v, label }))} />`}>
      <${Figure} title=${`${METRICS[metric]}, mean over all forecasts`} subtitle="Lower is better ↓" build=${barBuild} deps=${[meta, metric]} height=${Math.max(160, ids.length * 52 + 40)} filename=${'accuracy-' + metric}
        note="Bars start at zero. Differences between models can be small relative to the spread across series: compare with the distribution below." /><//>

    <${Section} title="Error by forecast step" aside=${html`<${Tabs} value=${hm} onChange=${setHm} label="Horizon metric" items=${[['mase', 'MASE'], ['sql', 'SQL'], ['bias', 'Bias'], ['coverage_90', 'Cover 90%']].map(([value, label]) => ({ value, label }))} />`}>
      <${Figure} title=${hm === 'coverage_90' ? 'Interval coverage by step' : `${METRICS[hm] || 'Bias'} by step ahead`} subtitle=${hm === 'coverage_90' ? 'Dashed line = the nominal 90%' : 'Errors usually grow with the horizon'} build=${horizonBuild} deps=${[meta, hm]} filename=${'horizon-' + hm} /><//>

    <${Section} title="Are the uncertainty estimates honest?">
      <div class="grid gap-4 lg:grid-cols-2">
        <${Figure} title="Calibration" subtitle="Share of actuals below each predicted quantile (dashed = perfect)" build=${calBuild} deps=${[meta]} filename="calibration" note="Above the diagonal at low quantiles and below at high ones means intervals are too narrow (over-confident)." />
        <${Figure} title="Interval coverage" subtitle="Share of actuals inside each central interval (dashed = nominal)" build=${covBuild} deps=${[meta]} filename="coverage" /></div><//>

    <${Section} title="Spread across series" aside=${html`<${Tabs} value=${dm} onChange=${setDm} label="Distribution metric" items=${[{ value: 'mase', label: 'MASE' }, { value: 'sql', label: 'SQL' }]} />`}>
      ${an.data ? html`<${Distribution} an=${an.data} ids=${ids} names=${names} focus=${focus} metric=${dm} />` : html`<${Spinner} />`}<//>

    <${Section} title="Forecast vs actual" aside=${html`<div class="w-56 max-w-full"><${Select} value=${model} onChange=${setAModel} options=${ids.map((i) => ({ value: i, label: names[i] }))} aria-label="Model for the plots" /></div>`}>
      ${an.data ? html`<div class="grid gap-4 lg:grid-cols-2"><${Figure} title="Median forecast vs actual" subtitle=${`${an.data.scatter.n.toLocaleString()} points${an.data.scatter.n > 3000 ? ' (evenly sampled to 3,000)' : ''} · dashed = perfect`} build=${scatterBuild} deps=${[an.data]} filename="scatter" />
        <${Figure} title="Error distribution" subtitle="Median forecast − actual, in units of each series’ history scale" build=${histBuild} deps=${[an.data]} filename="residuals" note="Tails clipped at the 99th percentile of |error| for readability." /></div>` : html`<${Spinner} />`}<//>

    <${ItemExplorer} id=${id} meta=${meta} focus=${focus} />

    <${Section} title="Reproducibility">
      <${Card}><div class="grid gap-6 md:grid-cols-2"><div><${Eyebrow}>Environment<//><div class="mt-3"><${KV} items=${[['OS', meta.environment.os], ['CPU', `${meta.environment.cpu} (${meta.environment.cpu_threads} threads)`], ['RAM', meta.environment.ram_gb + ' GB'], ['Python', meta.environment.python],
          ['PyTorch', `${meta.environment.packages.torch} (CUDA ${meta.environment.cuda_build || 'n/a'})`], ['timestral', `${meta.environment.packages.timestral} @ ${(meta.environment.timestral_commit || '').slice(0, 8)}`], ['tsfm-lab', meta.environment.tsfm_lab_commit || meta.environment.packages['tsfm-lab']]]} /></div></div>
        <div><${Eyebrow}>Models<//><div class="mt-3"><${KV} items=${ids.map((i) => [names[i], `${meta.models[i].device} · revision ${(meta.models[i].load || {}).revision === 'n/a' ? 'n/a' : String((meta.models[i].load || {}).revision || '').slice(0, 8)} · load ${(meta.models[i].load || {}).load_s}s`])} /></div></div></div>
        <${Collapse} title="Run configuration (JSON)"><pre class="overflow-x-auto rounded-md bg-surface-brand-primary p-4 font-mono text-[0.75rem]">${JSON.stringify({ dataset: meta.dataset, config: meta.config }, null, 2)}</pre><//><//><//>
  </div>`;
}

export function Runs({ route }) {
  return route.args[0] ? html`<${RunDetail} id=${route.args[0]} key=${route.args[0]} />` : html`<${RunList} />`;
}
