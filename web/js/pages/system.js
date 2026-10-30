import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, useAsync, useStore, useFocus, ms, num, bytes, when, deviceLabel, int } from '../core.js';
import { isActive, stopJob } from '../activity.js';
import { modelOptions } from './shared.js';
import { Card, PageHeader, Section, Button, Field, Select, NumberInput, Check, Notice, Spinner, ErrorBox, Table, KV, Progress, Tabs, Tag, Eyebrow } from '../ui.js';
import { Figure, barOption, linesOption } from '../charts.js';

const SCEN = [['mistral', 'Mistral setup (1 series, 2016 → 96)'], ['batch', 'Batch size (1 – 128)'], ['context', 'Context length (256 – 16,384)'], ['horizon', 'Horizon (24 – 1,024)']];
const VENDOR_MS = { 'timestral-5m': 17, 'timestral-209m': 113 };

function Benchmark({ sys, models }) {
  const { jobs } = useStore();
  const [model, setModel] = useState('timestral-5m');
  const gpus = sys.devices.filter((d) => d.kind === 'cuda' && d.supported);
  const [devs, setDevs] = useState(['cpu', ...gpus.map((g) => g.id)]);
  const [repeats, setRepeats] = useState(15), [scen, setScen] = useState(['mistral', 'batch', 'context', 'horizon']);
  const [error, setError] = useState(null);
  const kindsOf = (id) => { const m = models.find((x) => x.id === id); return (m && m.capabilities && m.capabilities.device_kinds) || null; };  // what the adapter can run on
  const offered = (d) => d.supported && (!kindsOf(model) || kindsOf(model).includes(d.kind));
  const pickModel = (id) => { setModel(id); const k = kindsOf(id); if (k) setDevs((cur) => cur.filter((x) => { const d = sys.devices.find((y) => y.id === x); return d && k.includes(d.kind); })); };
  const active = jobs.find((j) => j.kind === 'benchmark' && isActive(j));  // survives page changes and reloads
  const start = async () => {
    setError(null);
    try { await api('/api/jobs/benchmark', { method: 'POST', body: { model, devices: devs, repeats, scenarios: scen } }); }
    catch (e) { setError(e); }
  };
  const opts = modelOptions(models);
  return html`<${Card}><div class="grid gap-5 md:grid-cols-2">
    <div class="grid content-start gap-4"><${Field} label="Model"><${Select} value=${model} onChange=${pickModel} options=${opts} /><//>
      <div class="grid grid-cols-2 gap-3"><${Field} label="Timed repeats" hint="per scenario"><${NumberInput} value=${repeats} min=${1} max=${200} onChange=${(v) => setRepeats(v || 1)} /><//></div></div>
    <div><span class="label">Devices to compare</span>${sys.devices.map((d) => html`<${Check} disabled=${!offered(d)} checked=${devs.includes(d.id)} onChange=${(c) => setDevs(c ? [...devs, d.id] : devs.filter((x) => x !== d.id))} hint=${!d.supported ? d.reason : offered(d) ? null : 'not supported by this model'}>${d.id === 'cpu' ? 'CPU' : deviceLabel(d.id)} <span class="text-text-muted">· ${d.name.slice(0, 40)}</span><//>`)}
      <span class="label mt-3">Scenarios</span>${SCEN.map(([v, l]) => html`<${Check} checked=${scen.includes(v)} onChange=${(c) => setScen(c ? [...scen, v] : scen.filter((x) => x !== v))}>${l}<//>`)}</div></div>
    <div class="mt-5 flex flex-wrap items-center gap-4 border-t border-border-primary pt-5"><${Button} kind="brand" disabled=${!!active || !devs.length || !scen.length} onClick=${start}>${active ? 'Measuring…' : 'Run benchmark'}<//>
      ${active && html`<${Button} kind="danger" onClick=${() => stopJob(active.id)}>Stop<//>`}
      <p class="max-w-xl text-body-small text-text-muted">Each cell: one untimed warm-up call, then timed repeats with the GPU synchronised. Nothing else should be heavy on this machine while it runs. It keeps running if you leave this page.</p></div>
    ${active && html`<div class="mt-4"><span class="eyebrow">${active.label}</span><div class="mt-2"><${Progress} done=${active.done} total=${active.total} message=${active.status === 'queued' ? 'waiting for the current job…' : active.message} /></div></div>`}${error && html`<div class="mt-4"><${ErrorBox} error=${error} /></div>`}<//>`;
}

function Results({ id }) {
  const [b] = useAsync(() => api(`/api/benchmarks/${id}`), [id]);
  const [tab, setTab] = useState('batch');
  if (b.loading) return html`<${Spinner} />`;
  if (b.error) return html`<${ErrorBox} error=${b.error} />`;
  const r = b.data, devs = Object.keys(r.devices), gpu = devs.find((d) => d !== 'cpu');
  const cells = (d, s) => r.devices[d].cells.filter((c) => c.scenario === s || (s === 'batch' && c.scenario === 'mistral' && false));
  const head = (d) => r.devices[d].cells.find((c) => c.scenario === 'mistral');
  const lat = ['mistral', ...(devs.length ? [] : [])].length && devs.flatMap((d) => { const h = head(d); return h ? [{ id: `${d}:warm`, label: `${deviceLabel(d)} warm`, value: h.median_ms }, { id: `${d}:cold`, label: `${deviceLabel(d)} cold`, value: r.devices[d].cold_first_call_ms }] : []; });
  const lineFor = (scenario, key, labelFn) => { const first = cells(devs[0], scenario); return { x: first.map(labelFn), series: devs.map((d) => ({ id: d, label: deviceLabel(d), data: cells(d, scenario).map((c) => c[key]) })) }; };
  const tabs = [['batch', 'Batch'], ['context', 'Context'], ['horizon', 'Horizon']].filter(([s]) => cells(devs[0], s).length);
  const active = tabs.some(([s]) => s === tab) ? tab : (tabs[0] || [])[0];
  const vendor = VENDOR_MS[r.model];
  return html`<div class="grid grid-cols-[minmax(0,1fr)] gap-4">
    <${Card}><div class="flex flex-wrap items-center justify-between gap-3"><div><${Eyebrow}>${r.name} · ${when(r.created)}<//><p class="mt-1 text-body-small text-text-muted">${r.repeats} timed repeats · ${r.environment.cpu} · torch ${r.environment.packages.torch}</p></div></div>
      <div class="mt-4"><${Table} head=${['Device', 'Load', 'Cold first call', 'Warm median', 'p90', 'Warm min', 'Throughput', 'Peak VRAM', 'RAM change at load']} rows=${devs.map((d) => { const x = r.devices[d], h = head(d);
        return [html`<b>${deviceLabel(d)}</b>`, `${x.load_s} s`, ms(x.cold_first_call_ms), h ? ms(h.median_ms) : '–', h ? ms(h.p90_ms) : '–', h ? ms(h.min_ms) : '–', h ? `${num(h.series_per_s, 1)} series/s` : '–', h && h.peak_vram_mb ? `${num(h.peak_vram_mb, 0)} MB` : '–', `${x.process_memory_growth_mb} MB`]; })} /></div>
      ${r.cpu_vs_gpu && html`<p class="mt-4 text-body-small text-text-muted">CPU and ${deviceLabel(r.cpu_vs_gpu.gpu)} forecasts for the same input differ by at most <b class="num text-text-primary">${r.cpu_vs_gpu.max_abs_diff.toExponential(1)}</b> (${r.cpu_vs_gpu.relative_to_series_range.toExponential(1)} of the series range): float32 rounding, not a different answer.</p>`}
      ${vendor && html`<p class="mt-2 text-body-small text-text-muted">For reference, Mistral reports ${vendor} ms for this setup on an 8-thread server CPU. That is a different machine and a vendor measurement: only compare it to the CPU row above as an order of magnitude.</p>`}<//>
    ${lat.length > 0 && html`<${Figure} title="Latency at Mistral's setup" subtitle="1 series · 2016 steps of history · 96 ahead · lower is better ↓" build=${(p) => barOption(p, lat, { focus: lat.find((x) => x.id.startsWith('cuda') && x.id.endsWith('warm')) ? lat.find((x) => x.id.startsWith('cuda') && x.id.endsWith('warm')).id : 'cpu:warm', unit: ' ms', digits: 1 })} deps=${[r]} height=${Math.max(160, lat.length * 44 + 40)} filename="latency"
      note="“Cold” is the first call after loading: CUDA initialisation and kernel compilation. “Warm” is the median of repeated calls. Both matter, for different uses." />`}
    ${tabs.length > 0 && html`<div class="grid grid-cols-[minmax(0,1fr)] gap-3"><${Tabs} value=${active} onChange=${setTab} label="Scaling" items=${tabs.map(([value, label]) => ({ value, label }))} />
      <div class="grid gap-4 lg:grid-cols-2">
        <${Figure} title=${`Latency vs ${active}`} subtitle="Warm median, ms" build=${(p) => { const d = lineFor(active, 'median_ms', (c) => String(c[active])); return linesOption(p, { ...d, xName: active, yName: 'ms', yMin: 0, focus: gpu }); }} deps=${[r, active]} filename=${'latency-' + active} />
        <${Figure} title=${`Throughput vs ${active}`} subtitle="Series forecast per second, log scale" build=${(p) => { const d = lineFor(active, 'series_per_s', (c) => String(c[active])); return linesOption(p, { ...d, xName: active, yName: 'series / s', yLog: true, focus: gpu }); }} deps=${[r, active]} filename=${'throughput-' + active} /></div></div>`}
    ${gpu && html`<${Card} tight><${Eyebrow}>VRAM by scenario<//><div class="mt-2"><${Table} head=${['Scenario', ...devs.filter((d) => d !== 'cpu').map(deviceLabel)]} rows=${cells(gpu, 'batch').concat(cells(gpu, 'context'), cells(gpu, 'horizon')).map((c) => [c.label, c.peak_vram_mb ? `${num(c.peak_vram_mb, 0)} MB` : '–'])} /></div><//>`}
  </div>`;
}

export function System({ route }) {
  const [sys] = useAsync(() => api('/api/system'), []);
  const [disk] = useAsync(() => api('/api/system/disk'), []);
  const [brand] = useAsync(() => api('/api/brand'), []);
  const [models] = useAsync(() => api('/api/models'), []);
  const [list, reload] = useAsync(() => api('/api/benchmarks'), []);
  const [pick, setPick] = useState(null);
  const { jobs } = useStore();
  const seen = useRef(null);
  const doneJob = jobs.find((j) => j.kind === 'benchmark' && j.status === 'done' && j.result);
  useEffect(() => { if (doneJob && seen.current !== doneJob.id) { const first = seen.current === null; seen.current = doneJob.id; if (!first || !list.data || !list.data.length) { setPick(doneJob.result.benchmark_id); } reload(); } }, [doneJob && doneJob.id]);
  const bid = pick || (list.data && list.data[0] && list.data[0].id);
  useFocus(route, !sys.loading && !list.loading);
  if (sys.loading) return html`<${Spinner} />`;
  if (sys.error) return html`<${ErrorBox} error=${sys.error} />`;
  const e = sys.data.environment;
  return html`<div><${PageHeader} eyebrow="This machine" title="System">What the models run on, where the disk space goes, and measured CPU vs GPU performance with cold start reported separately from warm latency.<//>
    <${Section} title="Hardware and software"><div class="grid gap-4 md:grid-cols-2">
      <${Card}><${KV} items=${[['OS', e.os + ' ' + e.machine], ['CPU', e.cpu], ['Cores / threads', `${e.cpu_cores_physical} / ${e.cpu_threads}`], ['Torch threads', e.torch_threads], ['RAM', e.ram_gb + ' GB'], ['Python', e.python],
        ['PyTorch', `${e.packages.torch} · CUDA build ${e.cuda_build || 'n/a'}`], ['timestral', `${e.packages.timestral} @ ${(e.timestral_commit || '').slice(0, 8)}`], ['tsfm-lab', e.tsfm_lab_commit || e.packages['tsfm-lab']]]} /><//>
      <${Card}><${Eyebrow}>Compute devices<//><ul class="mt-3 divide-y divide-border-primary">${sys.data.devices.map((d) => html`<li class="py-3"><div class="flex flex-wrap items-center justify-between gap-2"><b>${d.id === 'cpu' ? 'CPU' : deviceLabel(d.id)}</b><${Tag}>${d.supported ? 'available' : 'unavailable'}<//></div>
        <div class="text-body-small text-text-muted">${d.name}${d.memory_gb ? ` · ${d.memory_gb} GB` : ''}${e.cuda_capability && d.kind === 'cuda' ? ` · compute capability ${e.cuda_capability.join('.')}` : ''}</div>${d.reason && html`<div class="mt-1 text-body-small text-text-warning">${d.reason}</div>`}</li>`)}</ul><//></div><//>
    <${Section} title="Disk" aside=${html`<span class="text-body-small text-text-muted">everything lives inside the project folder</span>`}>
      <${Card} tight>${disk.loading ? html`<${Spinner} label="Measuring disk usage…" />` : disk.error ? html`<${ErrorBox} error=${disk.error} />` : html`<${Table} head=${['Location', 'Files', 'Size']} rows=${disk.data.map((d) => [d.label, int(d.files), bytes(d.bytes)])} />
        <p class="mt-2 text-body-small text-text-muted">The package cache is hard-linked with <span class="font-mono">.venv</span> where the volume allows, so the real total is lower than the sum. Deleting the project folder removes everything.</p>`}<//><//>
    <${Section} title="Benchmark: CPU vs GPU">
      ${models.data && html`<${Benchmark} sys=${sys.data} models=${models.data} />`}
      <div class="mt-6" data-focus="benchmark">${list.data && list.data.length > 1 && html`<div class="mb-4 w-80 max-w-full"><${Field} label="Stored benchmark"><${Select} value=${bid} onChange=${setPick} options=${list.data.map((b) => ({ value: b.id, label: `${b.name} · ${b.devices.join(' + ')} · ${when(b.created)}` }))} /><//></div>`}
        ${bid ? html`<${Results} id=${bid} key=${bid} />` : html`<${Card}><p class="text-text-muted">No benchmark yet. Run one above to see cold start, warm latency, throughput and memory on this machine.</p><//>`}</div><//>
    <${Section} title="Brand assets"><${Card} tight><p class="text-body-small text-text-muted">${brand.data ? (brand.data.fonts ? `ALTMistral font loaded from this machine (${brand.data.fonts} files, local private use).` : 'ALTMistral font not found: the interface falls back to Inter.') : ''} ${brand.data && brand.data.logo ? 'Timestral logo present locally.' : 'Timestral logo not found: a neutral mark is shown.'}</p><//><//>
  </div>`;
}
