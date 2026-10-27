import { html, useState, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, go, num } from '../core.js';
import { Card, PageHeader, Section, Tag, Button, Spinner, ErrorBox, Notice, Table, Field, Select, Check, Tabs } from '../ui.js';
import { Figure, axisStyle, baseOption, palette, isNarrow } from '../charts.js';
import { specOf, keyOf } from './shared.js';

function Preview({ ds, onClose }) {
  const spec = specOf(keyOf(ds));
  const [idx, setIdx] = useState(0);
  const [info] = useAsync(() => api('/api/datasets/inspect', { method: 'POST', body: spec }), [ds.id]);
  const [data] = useAsync(() => api(`/api/datasets/series?index=${idx}&max_points=3000`, { method: 'POST', body: spec }), [ds.id, idx]);
  const d = data.data;
  const build = (p) => {
    const lines = d.target.map((row, i) => ({ name: d.target_names[i], type: 'line', data: row, showSymbol: false, lineStyle: { width: 1.5, color: p.series[i % p.series.length] }, itemStyle: { color: p.series[i % p.series.length] } }));
    return { ...baseOption(p), legend: { top: 0, textStyle: { color: p.text }, show: d.target.length > 1 },
      grid: { left: isNarrow() ? 44 : 56, right: 16, top: d.target.length > 1 ? 36 : 20, bottom: isNarrow() ? 30 : 56 },
      xAxis: { type: 'category', data: d.x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.muted, hideOverlap: true, formatter: (v) => String(v).replace('T', ' ').slice(0, 16) } },
      yAxis: { type: 'value', scale: true, ...axisStyle(p) },
      dataZoom: [{ type: 'inside', filterMode: 'none' }, ...(isNarrow() ? [] : [{ type: 'slider', height: 16, bottom: 6, borderColor: p.axis, textStyle: { color: p.muted } }])], series: lines };
  };
  const buildCov = (p) => {
    const rows = [...(d.known || []).map((r, i) => [d.known_names[i], r]), ...(d.past || []).map((r, i) => [d.past_names[i], r])];
    return { ...baseOption(p), legend: { top: 0, textStyle: { color: p.text } }, grid: { left: isNarrow() ? 44 : 56, right: 16, top: 36, bottom: 30 },
      xAxis: { type: 'category', data: d.x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.muted, hideOverlap: true, formatter: (v) => String(v).replace('T', ' ').slice(0, 10) } },
      yAxis: { type: 'value', scale: true, ...axisStyle(p) },
      series: rows.map(([n, r], i) => ({ name: n, type: 'line', step: 'end', data: r, showSymbol: false, lineStyle: { width: 1.4, color: p.series[(i + 2) % p.series.length] } })) };
  };
  return html`<${Section} title=${'Preview · ' + ds.name} aside=${html`<${Button} small kind="outline" onClick=${onClose}>Close<//>`}>
    ${info.loading ? html`<${Spinner} />` : info.error ? html`<${ErrorBox} error=${info.error} />` : html`
      <div class="mb-4 flex flex-wrap items-end gap-4">
        <div class="w-64 max-w-full"><${Field} label=${`Series (${info.data.n_series})`}><${Select} value=${String(idx)} onChange=${(v) => setIdx(Number(v))} options=${info.data.series.slice(0, 200).map((s, i) => ({ value: String(i), label: `${s.id} · ${s.length} steps` }))} /><//></div>
        <p class="pb-2 text-body-small text-text-muted num">freq ${info.data.freq || 'n/a'} · seasonal period ${info.data.seasonal_period} · ${info.data.n_targets} target${info.data.n_targets > 1 ? 's' : ''}${info.data.n_known ? ` · ${info.data.n_known} known covariates` : ''}${info.data.n_past_covariates ? ` · ${info.data.n_past_covariates} past covariates` : ''}</p></div>
      ${data.loading || !d ? html`<${Spinner} />` : html`<div class="grid grid-cols-[minmax(0,1fr)] gap-4">
        <${Figure} title=${d.id} subtitle=${d.downsampled ? `Showing a ${d.x.length}-point downsample of ${d.length} steps` : `${d.length} steps`} build=${build} deps=${[d]} filename=${'dataset-' + ds.id} />
        ${(d.known || d.past) && html`<${Figure} title="Covariates" subtitle="Known-future (and past-only) covariates over the whole series" build=${buildCov} deps=${[d]} height=${200} filename=${'covariates-' + ds.id} />`}</div>`}`}
  <//>`;
}

function Wizard({ onDone }) {
  const [up, setUp] = useState(null), [error, setError] = useState(null), [busy, setBusy] = useState(false), [rec, setRec] = useState(null);
  const [m, setM] = useState(null), [name, setName] = useState('');
  const input = useRef(null);
  const send = async (file) => {
    if (!file) return;
    setError(null); setRec(null); setBusy(true);
    try {
      const r = await api('/api/datasets/upload', { method: 'POST', raw: { data: await file.arrayBuffer(), headers: { 'X-Filename': file.name, 'Content-Type': 'application/octet-stream' } } });
      setUp(r); setName(file.name.replace(/\.[^.]+$/, ''));
      const g = r.guessed_mapping; setM({ timestamp: g.timestamp, id: g.id || '', targets: g.targets, past_covariates: [], known_covariates: [], freq: '' });
    } catch (e) { setError(e); setUp(null); } finally { setBusy(false); }
  };
  const toggle = (key, col) => setM((p) => {
    const has = p[key].includes(col);
    const strip = (k) => p[k].filter((c) => c !== col);
    return { ...p, targets: key === 'targets' ? (has ? strip('targets') : [...p.targets, col]) : strip('targets'),
      past_covariates: key === 'past_covariates' ? (has ? strip('past_covariates') : [...p.past_covariates, col]) : strip('past_covariates'),
      known_covariates: key === 'known_covariates' ? (has ? strip('known_covariates') : [...p.known_covariates, col]) : strip('known_covariates') };
  });
  const register = async () => {
    setError(null); setBusy(true);
    try { const r = await api('/api/datasets/register', { method: 'POST', body: { file_id: up.file_id, name, mapping: { ...m, id: m.id || null, freq: m.freq || null } } }); setRec(r); onDone(); }
    catch (e) { setError(e); } finally { setBusy(false); }
  };
  const cols = up ? up.columns.filter((c) => c !== m.timestamp && c !== m.id) : [];
  return html`<${Card}>
    <div class="grid gap-5">
      <div onDragOver=${(e) => e.preventDefault()} onDrop=${(e) => { e.preventDefault(); send(e.dataTransfer.files[0]); }}
        class="grid place-items-center gap-3 rounded-xl border border-dashed border-border-secondary px-4 py-10 text-center">
        <p class="text-body-large">Drop a CSV, TSV, Parquet or Excel file here</p>
        <p class="max-w-md text-body-small text-text-muted">One row per timestamp (and per series if there is an id column). The file stays on this machine, in <span class="font-mono">data/user/</span>.</p>
        <input ref=${input} type="file" class="sr-only" accept=".csv,.tsv,.txt,.parquet,.xlsx,.xls" onChange=${(e) => send(e.currentTarget.files[0])} />
        <${Button} kind="primary" icon="upload" disabled=${busy} onClick=${() => input.current.click()}>${busy ? 'Reading…' : 'Choose a file'}<//></div>
      ${error && html`<${ErrorBox} error=${error} />`}
      ${up && m && html`<div class="grid gap-5">
        <p class="text-body-small text-text-muted num"><b class="text-text-primary">${up.filename}</b> · ${up.rows.toLocaleString()} rows · ${up.columns.length} columns</p>
        <${Table} head=${up.columns} rows=${up.head.map((r) => up.columns.map((c) => String(r[c] ?? '')))} />
        <div class="grid gap-4 md:grid-cols-3">
          <${Field} label="Timestamp column"><${Select} value=${m.timestamp} onChange=${(v) => setM({ ...m, timestamp: v, targets: m.targets.filter((c) => c !== v) })} options=${up.columns.map((c) => ({ value: c, label: c }))} /><//>
          <${Field} label="Series id column (optional)"><${Select} value=${m.id} onChange=${(v) => setM({ ...m, id: v })} options=${[{ value: '', label: '(single series)' }, ...up.columns.filter((c) => c !== m.timestamp).map((c) => ({ value: c, label: c }))]} /><//>
          <${Field} label="Frequency (optional)" hint="pandas alias, e.g. h, D, 15min. Blank = infer."><input class="input" value=${m.freq} onInput=${(e) => setM({ ...m, freq: e.currentTarget.value })} placeholder="auto" /><//></div>
        <div class="grid gap-4 md:grid-cols-3">
          ${[['targets', 'Targets (forecast these)'], ['known_covariates', 'Known-future covariates'], ['past_covariates', 'Past-only covariates']].map(([k, label]) => html`<div><span class="label">${label}</span>
            <div class="max-h-48 overflow-y-auto rounded-md border border-border-primary bg-surface-brand-primary px-3 py-1">${cols.map((c) => html`<${Check} checked=${m[k].includes(c)} onChange=${() => toggle(k, c)}>${c}<//>`)}</div></div>`)}</div>
        <div class="flex flex-wrap items-end gap-4"><div class="w-64 max-w-full"><${Field} label="Dataset name"><input class="input" value=${name} onInput=${(e) => setName(e.currentTarget.value)} /><//></div>
          <${Button} kind="brand" disabled=${busy || !m.targets.length} onClick=${register}>Register dataset<//></div></div>`}
      ${rec && html`<div class="grid gap-3"><${Notice} kind="success" title=${`Registered “${rec.name}”`}>${rec.summary.n_series} series · freq ${rec.summary.freq || 'n/a'} · seasonal period ${rec.summary.seasonal_period} · data hash <span class="font-mono">${rec.report.data_hash}</span><//>
        ${rec.report.warnings.map((w) => html`<${Notice} kind="warning">${w}<//>`)}
        <${Table} head=${['Series', 'Length', 'Start', 'End', 'Gaps filled', 'Duplicates', 'Target missing %']} rows=${rec.report.series.slice(0, 12).map((s) => [s.id, s.length, s.start.slice(0, 16), s.end.slice(0, 16), s.gaps_filled_with_nan, s.duplicate_timestamps, s.target_missing_pct])} /></div>`}
    </div><//>`;
}

export function Datasets({ route }) {
  const [s, reload] = useAsync(() => api('/api/datasets'), []);
  const [tag, setTag] = useState('all');
  const [prev, setPrev] = useState(null);
  const all = s.data || [];
  const syn = all.filter((d) => d.source === 'synthetic');
  const tags = ['all', ...new Set(syn.flatMap((d) => d.tags))];
  const shown = syn.filter((d) => tag === 'all' || d.tags.includes(tag));
  const local = all.filter((d) => d.source === 'local');
  const del = async (d) => { if (confirm(`Remove “${d.name}” from this tool? The uploaded file in data/user is kept.`)) { await api(`/api/datasets/local/${d.id}`, { method: 'DELETE' }); reload(); } };
  const card = (d) => html`<${Card} tight class="flex flex-col">
    <div class="flex flex-wrap gap-1.5">${d.tags.map((t) => html`<${Tag}>${t}<//>`)}</div>
    <h3 class="mt-3 font-mistral text-h4">${d.name}</h3><p class="mt-1 flex-1 text-body-small text-text-muted">${d.description}</p>
    <div class="mt-4 flex flex-wrap gap-2"><${Button} small kind="outline" onClick=${() => setPrev(d)}>Preview<//>
      <${Button} small kind="outline" onClick=${() => go(`playground?ds=${keyOf(d)}`)}>Forecast<//><${Button} small kind="outline" onClick=${() => go(`experiments?ds=${keyOf(d)}`)}>Backtest<//>
      ${d.source === 'local' && html`<${Button} small kind="danger" icon="trash" aria-label="Remove dataset" onClick=${() => del(d)} />`}</div><//>`;
  return html`<div><${PageHeader} eyebrow="Data" title="Datasets">Bundled synthetic scenarios, your own files, and (next) online benchmarks. Public data may overlap a model's undisclosed training data: scenarios built here are guaranteed unseen, your files are marked “unknown”.<//>
    ${prev && html`<${Preview} ds=${prev} onClose=${() => setPrev(null)} />`}
    <${Section} title="Your data">
      ${local.length > 0 && html`<div class="mb-5 grid gap-4 md:grid-cols-2 lg:grid-cols-3">${local.map(card)}</div>`}<${Wizard} onDone=${reload} /><//>
    <${Section} title="Bundled scenarios" aside=${html`<span class="text-body-small text-text-muted">${syn.length} scenarios · seeded, deterministic</span>`}>
      <div class="mb-5"><${Tabs} value=${tag} onChange=${setTag} label="Filter by tag" items=${tags.map((t) => ({ value: t, label: t }))} /></div>
      ${s.loading ? html`<${Spinner} />` : s.error ? html`<${ErrorBox} error=${s.error} />` : html`<div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">${shown.map(card)}</div>`}<//>
  </div>`;
}
