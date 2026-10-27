import { html, useState, useEffect } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, go, deviceLabel, when } from '../core.js';
import { isActive, stopJob } from '../activity.js';
import { Card, PageHeader, Section, Button, Field, NumberInput, Check, Notice, ErrorBox, Progress, DeviceSelect, Table, Tag } from '../ui.js';
import { DatasetPicker, specOf, comparable } from './shared.js';
import { ModelChoice, ToggleList } from './playground.js';

const BASELINES = [['naive', 'Naive'], ['seasonal-naive', 'Seasonal naive'], ['drift', 'Drift']];

export function Experiments({ route }) {
  const { device } = useStore();
  const [ds] = useAsync(() => api('/api/datasets'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [pick, setPick] = useState({ key: route.query.ds || 'synthetic:multi_seasonality', params: {} });
  const [c, setC] = useState({ model: 'timestral-5m', horizon: 48, windows: 3, stride: null, maxSeries: null, maxContext: null, baselines: ['naive', 'seasonal-naive'], covariates: true, symmetric: true, maskFlat: false, label: '' });
  const { jobs } = useStore();
  const [startedId, setStartedId] = useState(() => { try { return sessionStorage.getItem('tsfm-exp-job'); } catch (e) { return null; } });
  const [error, setError] = useState(null);
  const set = (p) => setC((x) => ({ ...x, ...p }));
  const mine = jobs.filter((j) => j.kind === 'experiment');
  const active = mine.find(isActive);
  const job = active || mine.find((j) => j.id === startedId) || null;
  const running = !!active;

  // when the experiment started from this page finishes while the page is open, go to its run
  useEffect(() => {
    if (!job || job.id !== startedId) return;
    if (job.status === 'done' && job.result) { try { sessionStorage.removeItem('tsfm-exp-job'); } catch (e) { /* ignore */ } go(`runs/${job.result.run_id}`); }
    else if (job.status === 'error') setError(new Error(job.error));
  }, [job && job.status, startedId]);

  const start = async () => {
    setError(null);
    try {
      const j = await api('/api/jobs/experiment', { method: 'POST', body: { model: c.model, dataset: specOf(pick.key, pick.params), device, horizon: c.horizon, windows: c.windows, stride: c.stride, baselines: c.baselines,
        max_series: c.maxSeries, max_context: c.maxContext, use_covariates: c.covariates, label: c.label || null, options: { symmetric_averaging: c.symmetric, mask_flat_covariates: c.maskFlat } } });
      setStartedId(j.id);
      try { sessionStorage.setItem('tsfm-exp-job', j.id); } catch (e) { /* ignore */ }
    } catch (e) { setError(e); }
  };

  return html`<div><${PageHeader} eyebrow="Backtest" title="Experiments">Run a model and the baselines on identical rolling windows of a dataset, then inspect every metric. Timestral runs on <b class="text-text-primary">${deviceLabel(device)}</b>; change the device in the top bar or below.<//>
    <div class="mt-8 grid gap-4 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <${Card} class="!p-5 h-fit"><div class="grid gap-5">
        <${DatasetPicker} datasets=${ds.data || []} value=${pick} onChange=${setPick} />
        <div class="grid gap-4 border-t border-border-primary pt-5">
          <${Field} label="Model"><${ModelChoice} models=${models.data} value=${c.model} onChange=${(v) => set({ model: v, baselines: c.baselines.filter((b) => b !== v) })} /><//>
          <${DeviceSelect} class="flex-col !items-stretch" />
          <div><span class="label">Compare with</span><${ToggleList} items=${[...BASELINES, ...comparable(models.data, c.model)]} value=${c.baselines} onChange=${(v) => set({ baselines: v })} /></div></div>
        <div class="grid grid-cols-2 gap-3 border-t border-border-primary pt-5">
          <${Field} label="Horizon"><${NumberInput} value=${c.horizon} min=${1} max=${4096} onChange=${(v) => set({ horizon: v || 1 })} /><//>
          <${Field} label="Windows per series"><${NumberInput} value=${c.windows} min=${1} max=${50} onChange=${(v) => set({ windows: v || 1 })} /><//>
          <${Field} label="Stride" hint="blank = horizon"><${NumberInput} value=${c.stride} min=${1} placeholder="horizon" onChange=${(v) => set({ stride: v })} /><//>
          <${Field} label="Max series" hint="blank = all"><${NumberInput} value=${c.maxSeries} min=${1} placeholder="all" onChange=${(v) => set({ maxSeries: v })} /><//>
          <${Field} label="Max context" hint="blank = all history"><${NumberInput} value=${c.maxContext} min=${8} placeholder="all" onChange=${(v) => set({ maxContext: v })} /><//>
          <${Field} label="Label (optional)"><input class="input" value=${c.label} maxlength="80" onInput=${(e) => set({ label: e.currentTarget.value })} /><//></div>
        <div class="border-t border-border-primary pt-4">
          <${Check} checked=${c.covariates} onChange=${(v) => set({ covariates: v })} hint="Ignored automatically by models that cannot use them">Use covariates<//>
          <${Check} checked=${c.symmetric} onChange=${(v) => set({ symmetric: v })}>Symmetric averaging (Timestral)<//>
          <${Check} checked=${c.maskFlat} onChange=${(v) => set({ maskFlat: v })} hint="Workaround for upstream issue #7">Mask flat covariates<//></div>
        <div class="flex gap-3"><${Button} kind="brand" disabled=${running} onClick=${start}>${running ? 'Running…' : 'Start experiment'}<//>${running && html`<${Button} kind="danger" onClick=${() => stopJob(active.id)}>Stop<//>`}</div>
      </div><//>

      <div class="min-w-0 grid grid-cols-[minmax(0,1fr)] content-start gap-4">
        ${error && html`<${ErrorBox} error=${error} />`}
        ${job && html`<${Card}><span class="eyebrow">${job.status}${running ? ' · keeps running if you leave this page' : ''}</span><h3 class="mt-1 font-mistral text-h4">${job.label}</h3><div class="mt-4"><${Progress} done=${job.done} total=${job.total} message=${job.message} /></div>
          ${job.status === 'cancelled' && html`<p class="mt-3 text-text-muted">Cancelled.</p>`}<//>`}
        <${Card} tight><span class="eyebrow">What happens</span>
          <ol class="mt-3 list-decimal space-y-2 pl-5 text-body-small text-text-muted"><li>The model is loaded on the chosen device and warmed up once (not timed).</li><li>Each series is cut at ${c.windows} origins, ${c.stride || c.horizon} steps apart, counted back from the end.</li><li>Every model forecasts exactly the same windows; metrics, calibration and timings are stored with the full configuration so the run can be inspected and reproduced.</li></ol><//>
        <${Card} tight><span class="eyebrow">Recent jobs</span>
          <div class="mt-2"><${Table} head=${['Job', 'Status', 'Started']} empty="No jobs yet." rows=${jobs.slice(0, 6).map((j) => [j.label, html`<${Tag}>${j.status}<//>`, j.started ? when(new Date(j.started * 1000).toISOString()) : 'queued'])} /></div><//>
      </div></div></div>`;
}
