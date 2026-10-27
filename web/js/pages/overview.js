import { html } from '/vendor/htm-preact.js';
import { api, useAsync, params, bytes, when, num, useStore } from '../core.js';
import { Logo, Icon } from '../icons.js';
import { Card, Eyebrow, Section, Stat, Spinner, ErrorBox, Notice, DeviceSelect } from '../ui.js';

const Action = ({ href, icon, title, text }) => html`<a href=${href} class="group flex gap-4 rounded-xl border border-border-primary bg-surface-brand-secondary p-5 transition-colors hover:bg-surface-brand-tertiary">
  <span class="grid size-10 shrink-0 place-items-center rounded-md bg-surface-brand text-white"><${Icon} name=${icon} size=${18} /></span>
  <span class="min-w-0"><span class="block font-mistral text-h6">${title}</span><span class="block text-body-small text-text-muted">${text}</span></span></a>`;

export function Overview() {
  const { device } = useStore();
  const [sys] = useAsync(() => api('/api/system'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [runs] = useAsync(() => api('/api/runs'), []);
  const env = sys.data && sys.data.environment;
  const gpu = sys.data && sys.data.devices.find((d) => d.kind === 'cuda' && d.supported);
  const foundation = (models.data || []).filter((m) => m.kind === 'foundation');
  const ready = foundation.filter((m) => m.cache && m.cache.downloaded);
  return html`<div>
    <header class="pt-10 pb-8 md:pt-20 md:pb-14 border-b border-border-primary">
      <div class="flex items-center gap-4"><${Logo} size=${56} /><${Eyebrow}>Local sandbox · time-series foundation models<//></div>
      <h1 class="mt-5 max-w-4xl font-mistral text-h1 font-medium md:text-[3.5rem] md:leading-[3.75rem]">Run Timestral on your machine and judge it with your own data.</h1>
      <p class="mt-5 max-w-2xl text-body-large text-text-muted">Pick a model and a device, forecast any series, and inspect every number behind the result: metrics, calibration, latency, memory, and the exact configuration of each run.</p>
      <div class="mt-7 flex flex-wrap gap-3"><a href="#/playground" class="btn btn-brand">Open the playground</a><a href="#/experiments" class="btn btn-outline">Run an experiment</a></div>
    </header>

    <${Section} title="This machine">
      ${sys.loading ? html`<${Spinner} />` : sys.error ? html`<${ErrorBox} error=${sys.error} />` : html`<${Card} tight>
        <div class="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
          <${Stat} label="CPU" value=${env.cpu_threads + ' threads'} sub=${env.cpu} />
          <${Stat} label="Memory" value=${env.ram_gb + ' GB'} sub=${env.os} />
          <${Stat} label="GPU" value=${gpu ? gpu.name.replace('NVIDIA GeForce ', '') : 'none'} sub=${gpu ? gpu.memory_gb + ' GB · CUDA ' + env.cuda_build : 'CPU only'} />
          <${Stat} label="PyTorch" value=${env.packages.torch} sub=${'Python ' + env.python} />
        </div>
        <div class="mt-5 flex flex-col gap-3 border-t border-border-primary pt-5 sm:flex-row sm:items-center sm:justify-between">
          <p class="text-body-small text-text-muted">Models run on <b class="text-text-primary">${device === 'cpu' ? 'the CPU' : device}</b> by default. You can switch any time, even with a GPU present, and compare both on the System page.</p>
          <${DeviceSelect} /></div><//>`}
    <//>

    <${Section} title="Start here">
      <div class="grid gap-3 md:grid-cols-3">
        <${Action} href="#/playground" icon="play" title="Forecast one series" text="Pick data, origin and horizon; see the fan chart with all quantiles." />
        <${Action} href="#/experiments" icon="chart" title="Backtest on a dataset" text="Rolling windows against baselines, with calibration and per-horizon error." />
        <${Action} href="#/system" icon="chip" title="Benchmark CPU vs GPU" text="Cold start, warm latency, throughput and memory, measured here." />
      </div>
    <//>

    <div class="grid gap-8 md:grid-cols-2">
      <${Section} title="Models" aside=${html`<a class="text-body-small underline underline-offset-4" href="#/models">Manage</a>`}>
        ${models.loading ? html`<${Spinner} />` : html`<${Card} tight><p class="text-body-large"><b class="num">${ready.length}</b> of ${foundation.length} Timestral checkpoints are downloaded.</p>
          <ul class="mt-3 divide-y divide-border-primary">${foundation.map((m) => html`<li class="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 py-2.5"><span>${m.name} <span class="text-text-muted num">· ${params(m.params)}</span></span>
            <span class=${'tag ' + (m.cache.downloaded ? '' : 'opacity-60')}>${m.cache.downloaded ? bytes(m.cache.size_bytes) : 'not downloaded'}</span></li>`)}</ul><//>`}
      <//>
      <${Section} title="Recent runs" aside=${html`<a class="text-body-small underline underline-offset-4" href="#/runs">All runs</a>`}>
        ${runs.loading ? html`<${Spinner} />` : !runs.data.length ? html`<${Card} tight><p class="text-text-muted">No runs yet. Start with an experiment.</p><//>` : html`<${Card} tight>
          <ul class="divide-y divide-border-primary">${runs.data.slice(0, 5).map((r) => html`<li><a class="flex items-center justify-between gap-3 py-2.5 hover:underline underline-offset-4" href=${'#/runs/' + r.id}>
            <span class="min-w-0 truncate">${r.dataset_name} <span class="text-text-muted">· ${r.models[0]}</span></span><span class="shrink-0 text-body-small text-text-muted">${when(r.created)}</span></a></li>`)}</ul><//>`}
      <//>
    </div>
    <div class="mt-10"><${Notice} kind="warning" title="Read the weights license before relying on any result">
      Timestral's weights are released under a "Modified MIT" license that withholds all rights if your company's (or employer's) global consolidated monthly revenue exceeds US$20 million. Details on the <a class="underline underline-offset-2" href="#/about">About</a> page.<//></div>
  </div>`;
}
