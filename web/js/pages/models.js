import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, useFocus, params, bytes, ms, num, int } from '../core.js';
import { isActive, stopJob } from '../activity.js';
import { Logo, Icon } from '../icons.js';
import { familyName } from './shared.js';
import { Card, PageHeader, Section, Tag, Button, Spinner, ErrorBox, Notice, KV, Progress, Collapse, Select, Check, Field } from '../ui.js';
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

function RemoveCustom({ model, onChange }) {
  const remove = async () => {
    const ok = await confirmDialog({ title: `Remove ${model.name}?`, confirmLabel: 'Remove model', busyLabel: 'Removing…', run: () => api(`/api/models/custom/${model.id}`, { method: 'DELETE' }),
      body: html`The model is taken off this list and its downloaded weights (${bytes(model.cache.size_bytes)}) are deleted. Past runs that used it stay readable.` });
    if (ok) onChange();
  };
  return html`<div class="mt-3 text-right"><${Button} small kind="ghost" icon="trash" onClick=${remove}>Remove this custom model<//></div>`;
}

function AddCustom({ models, onAdded }) {
  const bases = models.filter((m) => m.kind === 'foundation' && !m.details.custom);
  const [repo, setRepo] = useState(''), [base, setBase] = useState(bases[0] ? bases[0].id : ''), [name, setName] = useState('');
  const [found, setFound] = useState(null), [error, setError] = useState(null), [busy, setBusy] = useState(false), [accept, setAccept] = useState(false);
  const reset = () => { setFound(null); setAccept(false); setError(null); };
  const inspect = async () => {
    reset(); setBusy(true);
    try { setFound(await api('/api/models/custom/inspect', { method: 'POST', body: { repo, base } })); } catch (e) { setError(e); }
    setBusy(false);
  };
  const add = async () => {
    setError(null); setBusy(true);
    try { await api('/api/models/custom', { method: 'POST', body: { repo, base, name: name || null, accept } }); setRepo(''); setName(''); reset(); onAdded(); } catch (e) { setError(e); }
    setBusy(false);
  };
  const ok = found && found.blocking.length === 0;
  return html`<${Section} title="Add a checkpoint from Hugging Face" aside=${html`<span class="text-body-small text-text-muted">your own fine-tunes or another checkpoint of a supported model</span>`}>
    <${Card}><div class="grid gap-4 md:grid-cols-2">
      <${Field} label="Repository" hint="organisation/name, public"><input class="input" placeholder="organisation/model-name" value=${repo} onInput=${(e) => { setRepo(e.currentTarget.value); reset(); }} /><//>
      <${Field} label="Runs on the adapter of" hint="its architecture and capabilities are assumed"><${Select} value=${base} onChange=${(v) => { setBase(v); reset(); }} aria-label="Base model" options=${bases.map((m) => ({ value: m.id, label: m.name, group: familyName(m.family) }))} /><//>
      <${Field} label="Name (optional)"><input class="input" maxlength="60" value=${name} onInput=${(e) => setName(e.currentTarget.value)} /><//>
      <div class="flex items-end"><${Button} kind="outline" disabled=${busy || !repo.trim() || !base} onClick=${inspect}>${busy && !found ? 'Checking…' : 'Check the repository'}<//></div></div>
    ${error && html`<div class="mt-4"><${ErrorBox} error=${error} /></div>`}
    ${found && html`<div class="mt-5 grid gap-3 border-t border-border-primary pt-5">
      ${found.blocking.map((b) => html`<${Notice} kind="error" title="Cannot be added">${b}<//>`)}
      ${found.sha && html`<${KV} items=${[['Repository', html`<a class="underline underline-offset-2" href=${found.url} target="_blank" rel="noreferrer">${found.repo}</a> · commit <span class="font-mono">${found.sha.slice(0, 10)}</span>`],
        ['Author', found.author], ['Licence', found.license || 'not declared'], ['Weights', `${bytes(found.size_bytes)}${found.params ? ` · ${params(found.params)} parameters` : ''}`],
        ['Popularity', `${found.downloads ?? '–'} downloads · ${found.likes ?? '–'} likes`], ['Last changed', found.last_modified ? found.last_modified.slice(0, 10) : '–']]} />`}
      ${found.warnings.map((w) => html`<${Notice} kind="warning">${w}<//>`)}
      ${ok && html`<div class="grid gap-3">
        <p class="text-body-small text-text-muted">Only <span class="font-mono">config.json</span> and <span class="font-mono">model.safetensors</span> of this exact commit are downloaded (nothing is executed from the repository), after you press Download on the model's card. Context length, quantile levels and covariate support are taken from the base model: right for a fine-tune of it, wrong for another architecture.</p>
        <${Check} checked=${accept} onChange=${setAccept}>I understand these are unverified third-party weights<//>
        <div><${Button} kind="brand" disabled=${!accept || busy} onClick=${add}>Add to the library<//></div></div>`}</div>`}
  <//><//>`;
}

function ModelCard({ m, reload }) {
  const d = m.details, c = m.capabilities;
  return html`<${Card} class="flex flex-col">
    <div class="flex items-start justify-between gap-3"><div class="flex min-w-0 items-center gap-4">${m.family === 'timestral' ? html`<${Logo} size=${48} />` : html`<span class="grid size-12 shrink-0 place-items-center rounded-md bg-surface-brand-tertiary text-text-secondary"><${Icon} name="models" size=${24} /></span>`}<div class="min-w-0"><span class="eyebrow">${m.family}</span><h3 class="mt-1 font-mistral text-h3">${m.name}</h3></div></div><div class="flex flex-wrap justify-end gap-1.5">${d.custom && html`<${Tag}>custom checkpoint<//>`}<${Tag}>${params(m.params)} params<//></div></div>
    <p class="mt-3 text-body-small text-text-muted">${m.description}</p>
    ${d.restriction && html`<div class="mt-3"><${Notice} kind="warning" title="Non-commercial licence">${d.restriction_note}<//></div>`}
    <div class="mt-5"><${KV} items=${[
      ...(d.custom ? [['Runs on', `the ${d.base} adapter; ${d.repo} @ ${String(d.custom_revision).slice(0, 10)}`]]
        : [['Layers × width', `${d.layers} × ${d.d_model}`], ...(d.num_heads != null || !m.cache.downloaded ? [['Attention heads', d.num_heads ?? 'download to read']] : []), ['Patch size', `${d.patch_size} steps`]]),
      ['Context', `${int(c.max_context)} steps`], ['Horizon per pass', `${int(c.max_horizon_per_pass)} steps`],
      ['Quantiles', `${c.quantile_levels.length} (${c.quantile_levels[0]} … ${c.quantile_levels.at(-1)})`],
      ['Inputs', [c.multivariate && (c.past_covariates || c.known_covariates ? 'multivariate' : 'several variables, one at a time'), c.past_covariates && 'past covariates', c.known_covariates && 'known covariates', c.categorical_covariates && 'categorical'].filter(Boolean).join(' · ')],
      ['Devices', c.device_kinds.join(', ').toUpperCase()],
    ]} /></div>
    ${d.custom && html`<div class="mt-3"><${Notice} kind="warning" title="Unverified third-party checkpoint">Context, quantile levels and covariate support shown here are those of the base model. Only config.json and model.safetensors are downloaded; no code from the repository is ever run.<//></div>`}
    <${CacheControls} model=${m} onChange=${reload} />
    ${d.custom && html`<${RemoveCustom} model=${m} onChange=${reload} />`}
    <${LocalBench} b=${m.local_benchmark} />
    <div class="mt-4"><${Reported} r=${d.vendor_reported} /><${Collapse} title="Known issues & licence"><ul class="list-disc space-y-2 pl-5 text-body-small text-text-muted">${m.notes.map((n) => html`<li>${n}</li>`)}</ul><//></div>
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
      <${AddCustom} models=${s.data} onAdded=${reload} />
      <${Section} title="Baselines">
        <p class="mb-4 max-w-3xl text-text-muted">Simple reference methods, always available, used as yardsticks. Their prediction intervals are built from their own in-sample errors.</p>
        <div class="grid gap-4 md:grid-cols-3">${baselines.map((m) => html`<${Card} tight><span class="eyebrow">baseline</span><h3 class="mt-1 font-mistral text-h4">${m.name}</h3><p class="mt-2 text-body-small text-text-muted">${m.description}</p><//>`)}</div><//>`}
  </div>`;
}
