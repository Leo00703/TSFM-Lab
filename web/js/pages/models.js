import { html, useState } from '/vendor/htm-preact.js';
import { api, useAsync, params, bytes, ms, num, waitForJob } from '../core.js';
import { Card, PageHeader, Section, Tag, Button, Spinner, ErrorBox, Notice, KV, Progress, Collapse } from '../ui.js';

function CacheControls({ model, onChange }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const c = model.cache;
  const download = async () => {
    setError(null);
    try {
      const j = await api(`/api/models/${model.id}/download`, { method: 'POST' });
      const done = await waitForJob(j.id, setJob);
      if (done.status === 'error') throw new Error(done.error);
      setJob(null); onChange();
    } catch (e) { setJob(null); setError(e); }
  };
  const remove = async () => {
    if (!confirm(`Delete the downloaded weights of ${model.name} (${bytes(c.size_bytes)})? You can download them again later.`)) return;
    try { await api(`/api/models/${model.id}/cache`, { method: 'DELETE' }); onChange(); } catch (e) { setError(e); }
  };
  return html`<div class="mt-4 border-t border-border-primary pt-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="text-body-small"><span class="eyebrow">Weights</span><div class="mt-1">${c.downloaded ? html`<span class="text-text-success">Downloaded</span> · <span class="num">${bytes(c.size_bytes)}</span> · revision <span class="font-mono">${c.revision.slice(0, 8)}</span>` : html`<span class="text-text-muted">Not downloaded (≈ ${model.details.weights_mb_fp32} MB)</span>`}</div></div>
      ${c.downloaded ? html`<${Button} kind="danger" small icon="trash" onClick=${remove}>Remove<//>` : html`<${Button} kind="primary" small icon="download" disabled=${!!job} onClick=${download}>Download<//>`}
    </div>
    ${job && html`<div class="mt-3"><${Progress} done=${0} total=${0} message=${job.message || 'Downloading from Hugging Face…'} /></div>`}
    ${error && html`<div class="mt-3"><${ErrorBox} error=${error} /></div>`}</div>`;
}

function Reported({ r }) {
  if (!r) return null;
  return html`<${Collapse} title=${html`Vendor-reported benchmarks <span class="tag ml-1">not measured here</span>`}>
    <p class="mb-3 text-body-small text-text-muted">From Mistral's model card. Lower is better for MASE/CRPS, higher for fev-bench skill. Compare with <a class="underline" href="#/runs">your own runs</a> before drawing conclusions.</p>
    <${KV} items=${[['GIFT-Eval MASE', r.gift_eval_mase], ['GIFT-Eval CRPS', r.gift_eval_crps], ['TIME MASE', r.time_mase], ['TIME CRPS', r.time_crps], ['fev-bench SQL skill', r.fev_bench_sql_skill + '%'], ['fev-bench MASE skill', r.fev_bench_mase_skill + '%']]} /><//>`;
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
    <div class="flex items-start justify-between gap-3"><div class="min-w-0"><span class="eyebrow">${m.family}</span><h3 class="mt-1 font-mistral text-h3">${m.name}</h3></div><div class="flex flex-wrap justify-end gap-1.5"><${Tag}>${params(m.params)} params<//></div></div>
    <p class="mt-3 text-body-small text-text-muted">${m.description}</p>
    <div class="mt-5"><${KV} items=${[
      ['Layers × width', `${d.layers} × ${d.d_model}`], ['Attention heads', d.num_heads ?? 'download to read'], ['Patch size', `${d.patch_size} steps`],
      ['Context', `${c.max_context.toLocaleString()} steps`], ['Horizon per pass', `${c.max_horizon_per_pass.toLocaleString()} steps`],
      ['Quantiles', `${c.quantile_levels.length} (${c.quantile_levels[0]} … ${c.quantile_levels.at(-1)})`], ['Inputs', 'multivariate · past & known covariates · categorical'],
      ['Devices', c.device_kinds.join(', ').toUpperCase()],
    ]} /></div>
    <${CacheControls} model=${m} onChange=${reload} />
    <${LocalBench} b=${m.local_benchmark} />
    <div class="mt-4"><${Reported} r=${d.vendor_reported} /><${Collapse} title="Known issues & license"><ul class="list-disc space-y-2 pl-5 text-body-small text-text-muted">${m.notes.map((n) => html`<li>${n}</li>`)}</ul><//></div>
  <//>`;
}

export function Models() {
  const [s, reload] = useAsync(() => api('/api/models'), []);
  const foundation = (s.data || []).filter((m) => m.kind === 'foundation');
  const baselines = (s.data || []).filter((m) => m.kind === 'baseline');
  return html`<div><${PageHeader} eyebrow="Model library" title="Models">Specifications come from each checkpoint's own config; benchmark figures from the vendor are labelled as such. Everything under “Measured” was produced on this machine.<//>
    ${s.loading ? html`<${Spinner} />` : s.error ? html`<${ErrorBox} error=${s.error} />` : html`
      <${Section} title="Timestral" aside=${html`<span class="text-body-small text-text-muted">New models plug in as adapters</span>`}>
        <div class="grid gap-4 lg:grid-cols-2">${foundation.map((m) => html`<${ModelCard} m=${m} reload=${reload} />`)}</div><//>
      <${Section} title="Baselines">
        <p class="mb-4 max-w-3xl text-text-muted">Simple reference methods, always available, used as yardsticks. Their prediction intervals are built from their own in-sample errors.</p>
        <div class="grid gap-4 md:grid-cols-3">${baselines.map((m) => html`<${Card} tight><span class="eyebrow">baseline</span><h3 class="mt-1 font-mistral text-h4">${m.name}</h3><p class="mt-2 text-body-small text-text-muted">${m.description}</p><//>`)}</div><//>`}
  </div>`;
}
