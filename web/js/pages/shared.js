import { html } from '/vendor/htm-preact.js';
import { Field, Select, NumberInput } from '../ui.js';

export const keyOf = (d) => `${d.source}:${d.id}`;

/** Model families as the interface names them; an unknown family still gets a readable name. */
const FAMILIES = { timestral: 'Timestral · Mistral AI', chronos: 'Chronos · Amazon', timesfm: 'TimesFM · Google', t0: 't0 · The Forecasting Company' };
export const familyName = (f) => FAMILIES[f] || f.charAt(0).toUpperCase() + f.slice(1);

/** Options for a model menu: foundation models grouped by family (largest first inside a family, as the server lists them). */
export const modelOptions = (models, { ready = false } = {}) => (models || []).filter((m) => m.kind === 'foundation' && (!ready || m.cache.downloaded))
  .map((m) => ({ value: m.id, label: `${m.name}${m.cache.downloaded ? '' : ' (download first)'}`, disabled: !m.cache.downloaded, group: familyName(m.family) }));

/** Downloaded foundation models other than the chosen one: what the chosen model can be compared with. */
export const comparable = (models, selected) => (models || []).filter((m) => m.kind === 'foundation' && m.cache.downloaded && m.id !== selected).map((m) => [m.id, m.name]);
export const specOf = (key, params = {}) => {
  const [source, ...rest] = key.split(':');
  return { source, id: rest.join(':'), params: source === 'synthetic' || source === 'hub' ? params : {} };
};

/** Dataset chooser with the editable generator parameters of synthetic scenarios. */
export function DatasetPicker({ datasets, value, onChange }) {
  const syn = datasets.filter((d) => d.source === 'synthetic'), loc = datasets.filter((d) => d.source === 'local'), hub = datasets.filter((d) => d.source === 'hub');
  const cur = datasets.find((d) => keyOf(d) === value.key);
  const set = (patch) => onChange({ ...value, ...patch });
  return html`<div class="grid gap-4">
    <${Field} label="Dataset" hint=${cur && cur.description}>
      <${Select} value=${value.key} onChange=${(k) => set({ key: k, params: {} })} aria-label="Dataset" options=${[...syn.map((d) => ({ value: keyOf(d), label: d.name, group: 'Bundled scenarios' })), ...loc.map((d) => ({ value: keyOf(d), label: d.name, group: 'Your data' })), ...hub.map((d) => ({ value: keyOf(d), label: d.name, group: 'Online benchmarks (downloaded)' }))]} /><//>
    ${cur && cur.source === 'hub' && html`<div class="grid gap-2"><${Field} label="Max series" hint="Large files are sampled (seeded, so the same series every time)."><${NumberInput} value=${value.params.max_series ?? cur.params.max_series} min=${1} max=${5000} onChange=${(v) => set({ params: { ...value.params, max_series: v ?? cur.params.max_series } })} /><//>
      ${cur.task && html`<p class="text-body-small text-text-muted">${cur.benchmark || 'The benchmark'} evaluates this dataset ${cur.task.horizon} steps ahead on ${cur.task.num_windows} window${cur.task.num_windows === 1 ? '' : 's'}.</p>`}</div>`}
    ${cur && cur.source === 'synthetic' && html`<div class="grid grid-cols-3 gap-3">
      ${['n_series', 'length', 'seed'].filter((k) => k in cur.params).map((k) => html`<${Field} label=${k.replace('_', ' ')}>
        <${NumberInput} value=${value.params[k] ?? cur.params[k]} min=${k === 'seed' ? 0 : 1} onChange=${(v) => set({ params: { ...value.params, [k]: v ?? cur.params[k] } })} /><//>`)}</div>`}
    ${cur && cur.contamination_risk !== 'none' && html`<p class="text-body-small text-text-warning">Possible contamination: ${cur.contamination_risk === 'unknown' ? 'the model’s real training data is not disclosed, so results on data that may be public are not guaranteed to be out-of-sample.' : cur.contamination_risk}</p>`}
  </div>`;
}
