import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, useFocus, go, bytes } from '../core.js';
import { Card, Button, Tag, Select, Tabs, Spinner, ErrorBox, Progress } from '../ui.js';
import { isActive, stopJob } from '../activity.js';
import { confirmDialog } from '../dialog.js';

const int = (n) => Number(n).toLocaleString('en-US');
const mb = (b) => (b >= 1e8 ? `${Math.round(b / 1e6)} MB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const label = (e) => `${e.title} · ${e.freq_label}`;
const pct = (x) => (x === null || x === undefined ? 'none' : x === 0 ? '0%' : `${x < 0.1 ? '<0.1' : x.toFixed(1)}%`);

function Facts({ items }) {
  return html`<dl class="grid gap-x-6 gap-y-3 text-body-small sm:grid-cols-2">${items.map(([k, v]) => html`<div class="min-w-0"><dt class="eyebrow">${k}</dt><dd class="mt-0.5 break-words">${v}</dd></div>`)}</dl>`;
}

const HORIZON_NOTE = { header: ' (declared in the file)', loader: ' (set by the Hugging Face loader)', default: ' (the file declares none: this is the default of the Hugging Face loader for the frequency)' };
const RULES = { fev: 'fev-bench tasks.yaml', gift: 'GIFT-Eval data.py (horizon and window rules)', monash: 'loader script monash_tsf.py' };
const win = (n) => `${n} window${n === 1 ? '' : 's'}`;

function setup(e) {
  const t = e.task;
  const terms = t.terms && t.terms.length > 1 ? ` Longer terms of the benchmark: ${t.terms.slice(1).map((x) => `${x.term} ${int(x.horizon)} steps × ${win(x.num_windows)}`).join(', ')}.` : '';
  return `forecast ${int(t.horizon)} steps ahead on ${win(t.num_windows)}${t.horizon_source ? HORIZON_NOTE[t.horizon_source] : ''}; seasonal period ${t.seasonality}.${terms}`;
}

function Details({ e, src }) {
  const c = e.checks, L = e.length;
  const verdict = [c.series_with_time_gaps ? `${c.series_with_time_gaps} with time gaps` : 'no time gaps', c.constant_series ? `${c.constant_series} constant` : 'none constant'].join(', ');
  return html`<div class="mt-4 grid gap-5 rounded-lg border border-border-primary bg-surface-brand-primary p-4 md:p-5">
    <${Facts} items=${[
      ['Series', `${int(e.n_series)} · ${L.min === L.max ? `${int(L.min)} steps each` : `${int(L.min)} to ${int(L.max)} steps (median ${int(L.median)})`} · ${int(L.total_points)} points`],
      ['Frequency', `${e.freq_label} (pandas “${e.freq}”)`],
      ['Targets', `${e.targets.length > 3 ? `${e.targets.length} series per row: ` : ''}${e.targets.slice(0, 6).join(', ')}${e.targets.length > 6 ? ', …' : ''}`],
      ['Known in advance', e.known.length ? e.known.join(', ') + (e.text_flags.length ? ` (text columns ${e.text_flags.join(', ')} become on/off flags)` : '') : 'none'],
      ['Past only', e.past.length ? e.past.join(', ') : 'none'],
      ['Ignored (static)', e.static.length ? e.static.join(', ') + ': per-series attributes, not used by the models here' : 'none'],
      [`${src.benchmark} setup`, setup(e)],
      ...(e.description ? [['About', e.description]] : []),
      ...(e.also_in.length ? [['Same data also in', e.also_in.map((o) => `${o.label} (${o.source_name})`).join('; ')]] : []),
    ]} />
    <div>
      <span class="eyebrow">Checked before listing</span>
      <p class="mt-1 text-body-small text-text-muted">${c.sampled_series >= e.n_series ? 'Every series was read.' : `Read on the first ${c.sampled_series} series (lengths are exact for all).`} Missing values: target <b class="num text-text-primary">${pct(c.target_missing_pct)}</b>${e.known.length ? html`, known covariates <b class="num text-text-primary">${pct(c.known_missing_pct)}</b>` : ''}${e.past.length ? html`, past covariates <b class="num text-text-primary">${pct(c.past_missing_pct)}</b>` : ''}. Time axis: ${verdict}.
      ${(c.target_missing_pct || 0) > 5 ? ' A lot of the target is missing: the models treat gaps as missing, but scores on such series deserve caution.' : ''}${L.min < 64 ? ' Some series are very short: they give the model little to learn from.' : ''}</p></div>
    <${Facts} items=${[
      ['Source', html`<span class="font-mono">${src.repo}</span> · ${e.files.length > 1 ? `${e.files.length} files` : e.path.split('/').slice(0, -1).join('/') || e.path} · revision <span class="font-mono">${src.revision.slice(0, 10)}</span>`],
      ['Download', html`${mb(e.size_bytes)}${e.files.length > 1 ? ` in ${e.files.length} files` : ''} · SHA-256 <span class="font-mono">${e.sha256.slice(0, 16)}…</span> (checked after download)`],
      ['Licence', html`${src.license} See the <a class="underline underline-offset-2" href=${src.card_url} target="_blank" rel="noreferrer">dataset card</a>; check the original terms before reusing the data.`],
      ['Task definition', html`<a class="underline underline-offset-2" href=${src.rules_url} target="_blank" rel="noreferrer">${RULES[e.source]}</a> at ${src.key === 'monash' ? 'revision' : 'commit'} <span class="font-mono">${src.rules_ref.slice(0, 10)}</span>`],
    ]} />
    ${e.notes.map((n) => html`<p class="text-body-small text-text-muted">${n}</p>`)}
    <p class="text-body-small text-text-warning">${src.contamination}</p>
  </div>`;
}

function Row({ e, src, active, onPreview, reload }) {
  const [open, setOpen] = useState(false), [ask, setAsk] = useState(false), [error, setError] = useState(null);
  const was = useRef(false);
  useEffect(() => { if (was.current && !active) reload(); was.current = !!active; }, [!!active]);
  const start = async () => { setError(null); setAsk(false); try { await api(`/api/catalog/${e.id}/download`, { method: 'POST' }); } catch (x) { setError(x); } };
  const remove = async () => {
    const ok = await confirmDialog({ title: 'Delete the downloaded file?', confirmLabel: 'Delete file', busyLabel: 'Deleting…', run: () => api(`/api/catalog/${e.id}`, { method: 'DELETE' }),
      body: html`<b class="text-text-primary capitalize">${label(e)}</b> (${mb(e.size_bytes)}) is removed from this project's cache. You can download it again later.` });
    if (ok) reload();
  };
  const key = `hub:${e.id}`;
  const cov = [e.known.length ? `${e.known.length} known` : '', e.past.length ? `${e.past.length} past` : ''].filter(Boolean).join(' + ');
  return html`<li class="py-5" data-focus=${e.id}>
    <div class="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
      <div class="min-w-0 flex-1">
        <h3 class="font-mistral text-h5 capitalize">${label(e)}</h3>
        <p class="mt-1 text-body-small text-text-muted num">${int(e.n_series)} series · ${e.length.min === e.length.max ? int(e.length.min) : `${int(e.length.min)}–${int(e.length.max)}`} steps · ${e.targets.length > 1 ? `${e.targets.length} targets` : '1 target'}${cov ? ` · covariates: ${cov}` : ''} · ${mb(e.size_bytes)}${e.files.length > 1 ? ` · ${e.files.length} files` : ''}</p>
        <div class="mt-2 flex flex-wrap gap-1.5">
          ${e.downloaded && html`<span class="tag !text-text-success">downloaded</span>`}<${Tag}>${src.name}<//>
          ${e.also_in.length > 0 && html`<${Tag}>also in ${[...new Set(e.also_in.map((o) => o.source_name))].join(', ')}<//>`}
          ${e.known.length > 0 && html`<${Tag}>known covariates<//>`}${e.targets.length > 1 && html`<${Tag}>multivariate<//>`}
          ${e.large && html`<${Tag}>large file<//>`}${e.n_series > e.sampled_above && html`<${Tag}>sampled to ${e.sampled_above} series<//>`}</div></div>
      <div class="flex flex-wrap items-center gap-2">
        <${Button} small kind="ghost" onClick=${() => setOpen(!open)} aria-expanded=${open}>${open ? 'Hide details' : 'Details'}<//>
        ${e.downloaded ? html`<${Button} small kind="outline" onClick=${() => onPreview({ source: 'hub', id: e.id, name: label(e) })}>Preview<//>
          <${Button} small kind="outline" onClick=${() => go(`playground?ds=${key}`)}>Forecast<//><${Button} small kind="outline" onClick=${() => go(`experiments?ds=${key}`)}>Backtest<//>
          <${Button} small kind="danger" icon="trash" aria-label="Remove downloaded file" onClick=${remove} />`
          : active ? html`<${Button} small kind="danger" onClick=${() => stopJob(active.id)}>Stop<//>`
          : html`<${Button} small kind="primary" icon="download" onClick=${() => setAsk(!ask)}>Download<//>`}</div></div>
    ${active && html`<div class="mt-3"><${Progress} done=${0} total=${0} message=${active.message || 'Downloading from Hugging Face…'} /></div>`}
    ${ask && !active && html`<div class="mt-4 grid gap-3 rounded-lg border border-border-secondary bg-surface-brand-primary p-4">
      <p class="text-body-small"><b>${mb(e.size_bytes)}</b> will be downloaded from Hugging Face into this project's cache${e.large ? html` <b class="text-text-warning">(a large file: it may take a while and the loaded data uses memory, so only ${e.sampled_above} series are used by default)</b>` : ''}. The file is checked against the SHA-256 published by Hugging Face and discarded if it differs.</p>
      <p class="text-body-small text-text-muted">${src.license} By downloading you confirm that you will check the original terms before reusing the data.</p>
      <div class="flex flex-wrap gap-2"><${Button} small kind="brand" icon="download" onClick=${start}>Download ${mb(e.size_bytes)}<//><${Button} small kind="ghost" onClick=${() => setAsk(false)}>Cancel<//></div></div>`}
    ${error && html`<div class="mt-3"><${ErrorBox} error=${error} /></div>`}
    ${open && html`<${Details} e=${e} src=${src} />`}
  </li>`;
}

export function OnlineCatalog({ onPreview, route }) {
  const { jobs } = useStore();
  const [s, reload] = useAsync(() => api('/api/catalog'), []);
  const [tab, setTab] = useState('all'), [freq, setFreq] = useState('all'), [from, setFrom] = useState('all'), [q, setQ] = useState(''), [limit, setLimit] = useState(20);
  const focus = route && route.query && route.query.focus;
  // a notification can lead to a dataset far down the list: make sure it is on screen before scrolling to it
  useEffect(() => {
    const i = focus && s.data ? s.data.entries.findIndex((e) => e.id === focus) : -1;
    if (i >= limit) setLimit(i + 1);
    if (i >= 0 && from !== 'all' && s.data.entries[i].source !== from) setFrom('all');
  }, [focus, s.data]);
  useFocus(route, !!s.data);
  if (s.loading && !s.data) return html`<${Spinner} />`;
  if (s.error) return html`<${ErrorBox} error=${s.error} />`;
  const { sources } = s.data;
  const entries = s.data.entries.map((e) => ({ ...e, source_name: sources[e.source].name, also_in: e.also_in.map((o) => ({ ...o, source_name: sources[o.source].name })) }));
  const active = (e) => jobs.find((j) => j.kind === 'download' && isActive(j) && j.link === `datasets?focus=${e.id}`);
  const freqs = [...new Set(entries.map((e) => e.freq_label))];
  const count = (key) => entries.filter((e) => e.source === key).length;
  const tests = { all: () => true, downloaded: (e) => e.downloaded, covariates: (e) => e.known.length > 0 || e.past.length > 0, multivariate: (e) => e.targets.length > 1, small: (e) => e.size_bytes < 5e6 };
  const shown = entries.filter((e) => tests[tab](e) && (from === 'all' || e.source === from) && (freq === 'all' || e.freq_label === freq) && (!q.trim() || (label(e) + ' ' + e.family + ' ' + e.targets.join(' ') + ' ' + e.known.join(' ')).toLowerCase().includes(q.trim().toLowerCase())));
  const total = entries.reduce((n, e) => n + (e.downloaded ? e.size_bytes : 0), 0);
  const link = (src, text) => html`<a class="underline underline-offset-2" href=${src.rules_url} target="_blank" rel="noreferrer">${text}</a>`;
  return html`<div>
    <p class="mb-5 max-w-3xl text-body-small text-text-muted">A finite, verified list rather than a search box: <b class="text-text-primary">${entries.length} datasets</b> from three public benchmarks — ${link(sources.fev, 'fev-bench')} (${count('fev')}), ${link(sources.gift, 'GIFT-Eval')} (${count('gift')}) and the ${link(sources.monash, 'Monash archive')} (${count('monash')}) — over files of Hugging Face dataset repositories at fixed revisions. Each was read before it was listed (exact lengths, missing values, time axis, covariates). Datasets that several benchmarks share are marked. Nothing is downloaded until you press Download.${total ? ` Downloaded so far: ${bytes(total)}.` : ''}</p>
    <div class="mb-4 flex flex-wrap items-center gap-3">
      <${Tabs} value=${tab} onChange=${(v) => { setTab(v); setLimit(20); }} label="Filter" items=${[['all', 'All'], ['downloaded', 'Downloaded'], ['covariates', 'With covariates'], ['multivariate', 'Multivariate'], ['small', 'Under 5 MB']].map(([value, l]) => ({ value, label: l }))} />
      <div class="flex flex-1 flex-wrap items-center gap-3 md:justify-end">
        <input class="input !w-56 max-w-full" type="search" placeholder="Search name or column" aria-label="Search the catalog" value=${q} onInput=${(ev) => { setQ(ev.currentTarget.value); setLimit(20); }} />
        <${Select} compact class="w-44" value=${from} onChange=${(v) => { setFrom(v); setLimit(20); }} aria-label="Benchmark" options=${[{ value: 'all', label: 'All benchmarks' }, ...Object.values(sources).map((x) => ({ value: x.key, label: `${x.name} (${count(x.key)})` }))]} />
        <${Select} compact class="w-40" value=${freq} onChange=${(v) => { setFreq(v); setLimit(20); }} aria-label="Frequency" options=${[{ value: 'all', label: 'Any frequency' }, ...freqs.map((f) => ({ value: f, label: f }))]} /></div></div>
    <${Card} tight>${shown.length === 0 ? html`<p class="py-6 text-center text-text-muted">Nothing matches.</p>`
      : html`<ul class="divide-y divide-border-primary">${shown.slice(0, limit).map((e) => html`<${Row} key=${e.id} e=${e} src=${sources[e.source]} active=${active(e)} onPreview=${onPreview} reload=${reload} />`)}</ul>`}<//>
    ${shown.length > limit && html`<div class="mt-4 text-center"><${Button} kind="outline" onClick=${() => setLimit(limit + 20)}>Show 20 more (${shown.length - limit} left)<//></div>`}
  </div>`;
}
