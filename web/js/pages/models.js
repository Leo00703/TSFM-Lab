import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, useFocus, params, bytes, ms, num } from '../core.js';
import { isActive, stopJob } from '../activity.js';
import { Logo, Icon } from '../icons.js';
import { familyName } from './shared.js';
import { Card, PageHeader, Section, Tag, Button, Spinner, ErrorBox, Notice, KV, Progress, Collapse } from '../ui.js';
import { confirmDialog } from '../dialog.js';

function CacheControls({ model, onChange }) {
  const { jobs } = useStore();
  const [error, setError] = useState(null);
  const c = model.cache;
  const active = jobs.find((j) => j.kind === 'download' && isActive(j) && j.label.includes(model.name));  // survives page changes
  const was = useRef(false);
  useEffect(() => { if (was.current && !active) onChange(); was.current = !!active; }, [!!active]);
  const download = async () => {
    setError(null);
    const restriction = model.details.restriction;
    if (restriction) {  // a licence that limits what the weights may be used for must be read before they are fetched
      const ok = await confirmDialog({ title: `${model.name} has a restricted licence`, danger: false, confirmLabel: 'I understand, download',
        body: html`<p>${model.details.restriction_note}</p><p class="mt-3">Read the <a class="underline underline-offset-2" href=${model.license_url} target="_blank" rel="noreferrer">licence</a> before relying on any result. By downloading you confirm that you will use the weights only as it allows.</p>` });
      if (!ok) return;
    }
    try { await api(`/api/models/${model.id}/download`, { method: 'POST' }); } catch (e) { setError(e); }
  };
  const remove = async () => {
    const ok = await confirmDialog({ title: `Delete the weights of ${model.name}?`, confirmLabel: 'Delete weights', busyLabel: 'Deleting…', run: () => api(`/api/models/${model.id}/cache`, { method: 'DELETE' }),
      body: html`${bytes(c.size_bytes)} will be freed from this project's cache. You can download the weights again later.` });
    if (ok) onChange();
  };
  return html`<div class="mt-4 border-t border-border-primary pt-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="text-body-small"><span class="eyebrow">Weights</span><div class="mt-1">${c.downloaded ? html`<span class="text-text-success">Downloaded</span> · <span class="num">${bytes(c.size_bytes)}</span> · revision <span class="font-mono">${c.revision.slice(0, 8)}</span>` : html`<span class="text-text-muted">Not downloaded (≈ ${model.details.weights_mb_fp32} MB)</span>`}</div></div>
      ${c.downloaded ? html`<${Button} kind="danger" small icon="trash" onClick=${remove}>Remove<//>` : html`<div class="flex gap-2"><${Button} kind="primary" small icon="download" disabled=${!!active} onClick=${download}>${active ? 'Downloading…' : 'Download'}<//>${active && html`<${Button} kind="danger" small onClick=${() => stopJob(active.id)}>Stop<//>`}</div>`}
    </div>
    ${active && html`<div class="mt-3"><${Progress} done=${0} total=${0} message=${active.message || 'Downloading from Hugging Face…'} /></div>`}
    ${error && html`<div class="mt-3"><${ErrorBox} error=${error} /></div>`}</div>`;
}

const REPORTED = { gift_eval_mase: 'GIFT-Eval MASE', gift_eval_crps: 'GIFT-Eval CRPS', time_mase: 'TIME MASE', time_crps: 'TIME CRPS', fev_bench_sql_skill: 'fev-bench SQL skill', fev_bench_mase_skill: 'fev-bench MASE skill' };

function Reported({ r }) {
  const rows = r ? Object.entries(REPORTED).filter(([k]) => r[k] !== undefined).map(([k, label]) => [label, k.startsWith('fev_bench') ? r[k] + '%' : r[k]]) : [];
  if (!rows.length) return null;
  return html`<${Collapse} title=${html`Published benchmarks <span class="tag ml-1">not measured here</span>`}>
    <p class="mb-3 text-body-small text-text-muted">Source: ${r.source || 'the vendor'}. Lower is better for MASE/CRPS, higher for fev-bench skill. Compare with <a class="underline" href="#/runs">your own runs</a> before drawing conclusions.</p>
    <${KV} items=${rows} /><//>`;
}

function LocalBench({ b }) {
  if (!b) return html`<p class="mt-4 border-t border-border-primary pt-4 text-body-small text-text-muted">No local benchmark yet. Run one on the <a class="underline" href="#/system">System</a> page to see latency and memory measured on this machine.</p>`;
  return html`<div class="mt-4 border-t border-border-primary pt-4"><span class="eyebrow">Measured on this machine</span>
    <div class="mt-2 grid grid-cols-2 gap-3 text-body-small">${Object.entries(b.devices).map(([dev, d]) => html`<div class="rounded-md bg-surface-brand-primary p-3">
      <div class="font-mono uppercase text-[0.6875rem] text-text-muted">${dev}</div>
      <div class="num font-medium">${ms(d.mistral_median_ms)} <span class="text-text-muted font-normal">warm</span></div>
      <div class="num text-text-muted">cold ${ms(d.cold_first_call_ms)} · load ${d.load_s}s${d.peak_vram_mb ? ` · ${num(d.peak_vram_mb, 0)} MB VRAM` : ''}</div></div>`)}</div>
    <p class="mt-2 text-body-small text-text-muted">1 series, 2016 → 96 steps (Mistral's announced setup).</p></div>`;
}

function ModelCard({ m, reload }) {
  const d = m.details, c = m.capabilities;
  return html`<${Card} class="flex flex-col">
    <div class="flex items-start justify-between gap-3"><div class="flex min-w-0 items-center gap-4">${m.family === 'timestral' ? html`<${Logo} size=${48} />` : html`<span class="grid size-12 shrink-0 place-items-center rounded-md bg-surface-brand-tertiary text-text-secondary"><${Icon} name="models" size=${24} /></span>`}<div class="min-w-0"><span class="eyebrow">${m.family}</span><h3 class="mt-1 font-mistral text-h3">${m.name}</h3></div></div><div class="flex flex-wrap justify-end gap-1.5"><${Tag}>${params(m.params)} params<//></div></div>
    <p class="mt-3 text-body-small text-text-muted">${m.description}</p>
    ${d.restriction && html`<div class="mt-3"><${Notice} kind="warning" title="Non-commercial licence">${d.restriction_note}<//></div>`}
    <div class="mt-5"><${KV} items=${[
      ['Layers × width', `${d.layers} × ${d.d_model}`], ['Attention heads', d.num_heads ?? 'download to read'], ['Patch size', `${d.patch_size} steps`],
      ['Context', `${c.max_context.toLocaleString()} steps`], ['Horizon per pass', `${c.max_horizon_per_pass.toLocaleString()} steps`],
      ['Quantiles', `${c.quantile_levels.length} (${c.quantile_levels[0]} … ${c.quantile_levels.at(-1)})`],
      ['Inputs', [c.multivariate && 'multivariate', c.past_covariates && 'past covariates', c.known_covariates && 'known covariates', c.categorical_covariates && 'categorical'].filter(Boolean).join(' · ')],
      ['Devices', c.device_kinds.join(', ').toUpperCase()],
    ]} /></div>
    <${CacheControls} model=${m} onChange=${reload} />
    <${LocalBench} b=${m.local_benchmark} />
    <div class="mt-4"><${Reported} r=${d.vendor_reported} /><${Collapse} title="Known issues & license"><ul class="list-disc space-y-2 pl-5 text-body-small text-text-muted">${m.notes.map((n) => html`<li>${n}</li>`)}</ul><//></div>
  <//>`;
}

export function Models({ route }) {
  const [s, reload] = useAsync(() => api('/api/models'), []);
  useFocus(route, !s.loading);
  const foundation = (s.data || []).filter((m) => m.kind === 'foundation');
  const families = [...new Set(foundation.map((m) => m.family))];
  const baselines = (s.data || []).filter((m) => m.kind === 'baseline');
  return html`<div><${PageHeader} eyebrow="Model library" title="Models">Specifications come from each checkpoint's own config; benchmark figures from the vendor are labelled as such. Everything under “Measured” was produced on this machine.<//>
    ${s.loading ? html`<${Spinner} />` : s.error ? html`<${ErrorBox} error=${s.error} />` : html`
      ${families.map((f, i) => html`<${Section} title=${familyName(f)} aside=${i === 0 ? html`<span class="text-body-small text-text-muted">New models plug in as adapters</span>` : null}>
        <div class="grid gap-4 lg:grid-cols-2">${foundation.filter((m) => m.family === f).map((m) => html`<div data-focus=${m.id} class="flex flex-col"><${ModelCard} m=${m} reload=${reload} /></div>`)}</div><//>`)}
      <${Section} title="Baselines">
        <p class="mb-4 max-w-3xl text-text-muted">Simple reference methods, always available, used as yardsticks. Their prediction intervals are built from their own in-sample errors.</p>
        <div class="grid gap-4 md:grid-cols-3">${baselines.map((m) => html`<${Card} tight><span class="eyebrow">baseline</span><h3 class="mt-1 font-mistral text-h4">${m.name}</h3><p class="mt-2 text-body-small text-text-muted">${m.description}</p><//>`)}</div><//>`}
  </div>`;
}
