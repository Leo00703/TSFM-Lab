import { html, useState, useEffect } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, go, bytes, num, when, deviceLabel } from '../core.js';
import { isActive, stopJob } from '../activity.js';
import { Card, PageHeader, Section, Button, Field, NumberInput, Check, Notice, ErrorBox, Progress, DeviceSelect, Table, Tag, Tabs, Spinner } from '../ui.js';
import { confirmDialog } from '../dialog.js';

const SIZE_LIMITS = [['5', 'Under 5 MB'], ['50', 'Under 50 MB'], ['any', 'Any size']];
const params = (n) => (!n ? '–' : n >= 1e6 ? `${Math.round(n / 1e6)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n));
const skillText = (x) => (x === null || x === undefined ? '–' : `${x >= 0 ? '+' : ''}${x.toFixed(1)}%`);

function SkillBar({ value }) {
  if (value === null || value === undefined) return '–';
  const w = Math.min(100, Math.abs(value) * 1.6);
  return html`<span class="flex items-center gap-2"><span class="num w-16 text-right font-semibold">${skillText(value)}</span>
    <span class="h-2 w-24 rounded-sm bg-border-primary" aria-hidden="true"><span class=${'block h-2 rounded-sm ' + (value >= 0 ? 'bg-surface-brand' : 'bg-text-warning')} style=${{ width: `${w}%` }}></span></span></span>`;
}

function Form() {
  const { device, jobs } = useStore();
  const [cat, reloadCat] = useAsync(() => api('/api/catalog'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [c, setC] = useState({ source: 'fev', size: '50', models: null, maxSeries: 20, maxWindows: 3, maxContext: 2048, covariates: true, onlyDownloaded: false, label: '' });
  const [error, setError] = useState(null), [startedId, setStartedId] = useState(() => { try { return sessionStorage.getItem('tsfm-suite-job'); } catch (e) { return null; } });
  const set = (p) => setC((x) => ({ ...x, ...p }));
  const mine = jobs.filter((j) => j.kind === 'suite'), active = mine.find(isActive), job = active || mine.find((j) => j.id === startedId) || null;
  const downloading = jobs.filter((j) => j.kind === 'download' && isActive(j)).length;
  useEffect(() => { if (downloading === 0) reloadCat(); }, [downloading]);
  useEffect(() => {
    if (!job || job.id !== startedId) return;
    if (job.status === 'done' && job.result) { try { sessionStorage.removeItem('tsfm-suite-job'); } catch (e) { /* ignore */ } go(`leaderboard/${job.result.suite_id}`); }
    else if (job.status === 'error') setError(new Error(job.error));
  }, [job && job.status, startedId]);
  if (cat.loading && !cat.data) return html`<${Spinner} />`;
  if (cat.error) return html`<${ErrorBox} error=${cat.error} />`;
  const sources = cat.data.sources, limit = c.size === 'any' ? Infinity : Number(c.size) * 1e6;
  const chosen = cat.data.entries.filter((e) => e.source === c.source && e.size_bytes < limit);
  const missing = chosen.filter((e) => !e.downloaded), ready = chosen.filter((e) => e.downloaded);
  const missingBytes = missing.reduce((n, e) => n + e.size_bytes, 0);
  const ready_models = (models.data || []).filter((m) => m.kind === 'foundation' && m.cache.downloaded);
  const picked = c.models || ready_models.slice(0, 2).map((m) => m.id);
  const run = c.onlyDownloaded ? ready : chosen;
  const blocked = (!c.onlyDownloaded && missing.length > 0) || run.length === 0 || picked.length === 0;
  const fetchMissing = async () => {
    const ok = await confirmDialog({ title: `Download ${missing.length} dataset${missing.length === 1 ? '' : 's'}?`, confirmLabel: `Download ${bytes(missingBytes)}`, busyLabel: 'Starting…',
      run: async () => { for (const e of missing) await api(`/api/catalog/${e.id}/download`, { method: 'POST' }); },
      body: html`${bytes(missingBytes)} will be downloaded from Hugging Face into this project's cache, one dataset after another, and checked against the published SHA-256. ${sources[c.source].license} By downloading you confirm that you will check the original terms before reusing the data.` });
    if (ok) reloadCat();
  };
  const start = async () => {
    setError(null);
    try {
      const j = await api('/api/jobs/suite', { method: 'POST', body: { datasets: run.map((e) => e.id), models: picked, device, max_series: c.maxSeries, max_windows: c.maxWindows, max_context: c.maxContext, use_covariates: c.covariates, label: c.label || null } });
      setStartedId(j.id);
      try { sessionStorage.setItem('tsfm-suite-job', j.id); } catch (e) { /* ignore */ }
    } catch (e) { setError(e); }
  };
  return html`<div class="mt-8 grid gap-4 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
    <${Card} class="!p-5 h-fit"><div class="grid gap-5">
      <${Field} label="Benchmark" hint="Each dataset is evaluated with the horizon and windows of its own benchmark."><${Tabs} value=${c.source} onChange=${(v) => set({ source: v })} label="Benchmark" items=${Object.values(sources).map((x) => ({ value: x.key, label: x.name }))} /><//>
      <${Field} label="Dataset size"><${Tabs} value=${c.size} onChange=${(v) => set({ size: v })} label="Dataset size" items=${SIZE_LIMITS.map(([value, label]) => ({ value, label }))} /><//>
      <div class="rounded-lg border border-border-primary p-3 text-body-small">
        <b class="num">${chosen.length}</b> dataset${chosen.length === 1 ? '' : 's'} selected · <b class="num">${ready.length}</b> downloaded${missing.length ? html` · <b class="num text-text-warning">${missing.length}</b> to download (${bytes(missingBytes)})` : ''}
        ${missing.length > 0 && html`<div class="mt-2 flex flex-wrap items-center gap-3"><${Button} small kind="outline" icon="download" disabled=${downloading > 0} onClick=${fetchMissing}>${downloading > 0 ? 'Downloading…' : `Download ${missing.length} missing`}<//></div>`}
        ${missing.length > 0 && ready.length > 0 && html`<div class="mt-2"><${Check} checked=${c.onlyDownloaded} onChange=${(v) => set({ onlyDownloaded: v })}>Run only on the ${ready.length} already downloaded<//></div>`}</div>
      <div class="border-t border-border-primary pt-5"><span class="label">Models</span>
        ${ready_models.length === 0 ? html`<p class="text-body-small text-text-muted">Download a model on the Models page first.</p>`
          : html`<div class="flex flex-wrap gap-x-5">${ready_models.map((m) => html`<${Check} checked=${picked.includes(m.id)} onChange=${(on) => set({ models: on ? [...picked, m.id] : picked.filter((x) => x !== m.id) })}>${m.name}<//>`)}</div>`}
        <p class="mt-2 text-body-small text-text-muted">Seasonal naive always runs on the same windows as the reference every score is measured against.</p></div>
      <${DeviceSelect} class="flex-col !items-stretch" />
      <div class="grid grid-cols-2 gap-3 border-t border-border-primary pt-5">
        <${Field} label="Series per dataset" hint="seeded sample"><${NumberInput} value=${c.maxSeries} min=${1} max=${200} onChange=${(v) => set({ maxSeries: v || 1 })} /><//>
        <${Field} label="Windows per series" hint="cap"><${NumberInput} value=${c.maxWindows} min=${1} max=${20} onChange=${(v) => set({ maxWindows: v || 1 })} /><//>
        <${Field} label="Max context" hint="same for every model"><${NumberInput} value=${c.maxContext} min=${8} placeholder="all" onChange=${(v) => set({ maxContext: v })} /><//>
        <${Field} label="Label (optional)"><input class="input" value=${c.label} maxlength="80" onInput=${(e) => set({ label: e.currentTarget.value })} /><//></div>
      <${Check} checked=${c.covariates} onChange=${(v) => set({ covariates: v })} hint="Ignored automatically by models that cannot use them">Use covariates<//>
      <div class="flex gap-3"><${Button} kind="brand" disabled=${!!active || blocked} onClick=${start}>${active ? 'Running…' : 'Start leaderboard run'}<//>${active && html`<${Button} kind="danger" onClick=${() => stopJob(active.id)}>Stop<//>`}</div>
      ${blocked && !active && html`<p class="text-body-small text-text-muted">${picked.length === 0 ? 'Pick at least one model.' : run.length === 0 ? 'No dataset matches.' : 'Download the missing datasets first, or run only on the downloaded ones.'}</p>`}
    </div><//>
    <div class="min-w-0 grid grid-cols-[minmax(0,1fr)] content-start gap-4">
      ${error && html`<${ErrorBox} error=${error} />`}
      ${job && html`<${Card}><span class="eyebrow">${job.status}${active ? ' · keeps running if you leave this page' : ''}</span><h3 class="mt-1 font-mistral text-h4">${job.label}</h3><div class="mt-4"><${Progress} done=${job.done} total=${job.total} message=${job.message} /></div>
        ${job.status === 'cancelled' && html`<p class="mt-3 text-text-muted">Cancelled: nothing was saved.</p>`}<//>`}
      <${Card} tight><span class="eyebrow">How it is scored</span>
        <ol class="mt-3 list-decimal space-y-2 pl-5 text-body-small text-text-muted"><li>Every model is loaded once and forecasts exactly the same windows of every dataset, with the horizon of the benchmark's own task.</li><li>Per dataset the error is the mean over series and windows. The score is the geometric mean, over datasets, of the model's error divided by seasonal naive's: 0% matches seasonal naive, +30% means 30% lower error.</li><li>Only datasets that every ranked model completed count towards the ranking, so a model that cannot take multivariate data is not compared on an easier set. Models with different quantile levels are scored on the levels they share.</li></ol><//>
      <${Notice} kind="warning" title="Read it as an indication">Public benchmark data may overlap what a model was trained on, and a sample of ${c.maxSeries} series per dataset is noisy. Use this to compare models with each other on the same footing, not as proof of out-of-sample skill.<//>
    </div></div>`;
}

function History() {
  const [s, reload] = useAsync(() => api('/api/suites'), []);
  const del = async (r) => {
    const ok = await confirmDialog({ title: 'Delete this leaderboard run?', confirmLabel: 'Delete run', busyLabel: 'Deleting…', run: () => api(`/api/suites/${r.id}`, { method: 'DELETE' }),
      body: 'The results of this run are deleted. This cannot be undone.' });
    if (ok) reload();
  };
  return html`<${Section} title="Previous runs">${s.loading ? html`<${Spinner} />` : s.error ? html`<${ErrorBox} error=${s.error} />` : html`<${Card} tight>
    <${Table} head=${['When', 'Label', 'Models', 'Datasets', 'Best (SQL skill)', '']} empty="No leaderboard runs yet." rows=${s.data.map((r) => [when(r.created), r.label || '–', r.models.join(', '), `${r.common} of ${r.datasets}`, r.best ? `${r.best} ${skillText(r.best_skill)}` : '–',
      html`<span class="flex gap-2"><${Button} small kind="outline" onClick=${() => go('leaderboard/' + r.id)}>Open<//><${Button} small kind="danger" icon="trash" aria-label="Delete run" onClick=${() => del(r)} /></span>`])} /><//>`}<//>`;
}

function Detail({ id }) {
  const [s] = useAsync(() => api(`/api/suites/${id}`), [id]);
  const [view, setView] = useState('overall');
  if (s.loading) return html`<${Spinner} />`;
  if (s.error) return html`<${ErrorBox} error=${s.error} />`;
  const r = s.data, board = r.leaderboard, name = (m) => (r.models[m] || {}).name || m;
  const rowsOf = (table) => board.ranked.slice().sort((a, b) => (table[b].sql_skill ?? -1e9) - (table[a].sql_skill ?? -1e9));
  const table = view === 'overall' ? board.overall : board.by_group[view];
  const groups = Object.keys(board.by_group), all = [...board.ranked, 'seasonal-naive'];
  const problems = r.datasets.filter((d) => d.error || Object.keys(d.skipped).length);
  return html`<div><${PageHeader} eyebrow="Leaderboard" title=${r.label || 'Leaderboard run'}>${when(r.created)} · ${deviceLabel(r.device)} · ${board.total_datasets} datasets (${board.common_datasets} counted) · ${Math.round(r.elapsed_s)} s<//>
    <${Section} title="Ranking" aside=${html`<${Button} small kind="ghost" onClick=${() => go('leaderboard')}>Back<//>`}>
      <div class="mb-3"><${Tabs} value=${view} onChange=${setView} label="Group" items=${[{ value: 'overall', label: 'All datasets' }, ...groups.map((g) => ({ value: g, label: g }))]} /></div>
      <${Card} tight><${Table} head=${['#', 'Model', 'SQL skill', 'MASE skill', 'Wins', 'Datasets', 'Time']} empty="Nothing completed." rows=${rowsOf(table).map((m, i) => [i + 1, html`<b>${name(m)}</b>`, html`<${SkillBar} value=${table[m].sql_skill} />`, skillText(table[m].mase_skill), table[m].wins, table[m].datasets, `${table[m].total_s} s`])} /><//>
      <p class="mt-3 max-w-3xl text-body-small text-text-muted">Skill is measured against seasonal naive on the same windows (geometric mean of error ratios over datasets; higher is better). Only the ${board.common_datasets} datasets every ranked model completed are counted. Settings: ${r.config.max_series} series, up to ${r.config.max_windows} windows, context ${r.config.max_context || 'unlimited'}, covariates ${r.config.use_covariates ? 'on' : 'off'}.</p><//>
    ${problems.length > 0 && html`<${Section} title="Not counted"><${Card} tight><ul class="grid gap-1 text-body-small text-text-muted">${problems.map((d) => html`<li><b class="text-text-primary">${d.title}</b> · ${d.error ? d.error : `${Object.keys(d.skipped).map(name).join(', ')} skipped: ${Object.values(d.skipped)[0]}`}</li>`)}</ul><//><//>`}
    <${Section} title="Every dataset" aside=${html`<span class="text-body-small text-text-muted">mean SQL (lower is better); the best model is marked</span>`}>
      <${Card} class="!p-3 md:!p-5"><${Table} head=${['Dataset', 'Horizon × windows', 'Series', ...all.map(name)]} rows=${r.datasets.map((d) => {
        const best = board.ranked.filter((m) => d.models[m]).sort((a, b) => d.models[a].sql - d.models[b].sql)[0];
        return [html`<span>${d.title} <${Tag}>${d.group}<//></span>`, `${d.horizon} × ${d.windows}`, d.series, ...all.map((m) => (d.models[m] ? html`<span class=${m === best ? 'font-semibold text-text-success' : ''}>${num(d.models[m].sql)}</span>` : '–'))];
      })} /><//><//>
    <${Section} title="Models"><${Card} tight><${Table} head=${['Model', 'Parameters', 'Licence', 'Device']} rows=${all.map((m) => [name(m), params(r.models[m].params), r.models[m].license || '–', (r.models[m].load || {}).device || '–'])} /><//><//>
  </div>`;
}

export function Leaderboard({ route }) {
  return route.args[0] ? html`<${Detail} id=${route.args[0]} key=${route.args[0]} />`
    : html`<div><${PageHeader} eyebrow="Compare" title="Leaderboard">Run several models on a whole benchmark (fev-bench, GIFT-Eval or the Monash archive) and rank them on the same footing, scored against seasonal naive the way fev-bench scores.<//><${Form} /><${History} /></div>`;
}
