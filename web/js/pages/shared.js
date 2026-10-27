import { html } from '/vendor/htm-preact.js';
import { Field, Select, NumberInput } from '../ui.js';

export const keyOf = (d) => `${d.source}:${d.id}`;
export const specOf = (key, params = {}) => {
  const [source, ...rest] = key.split(':');
  return { source, id: rest.join(':'), params: source === 'synthetic' ? params : {} };
};

/** Dataset chooser with the editable generator parameters of synthetic scenarios. */
export function DatasetPicker({ datasets, value, onChange }) {
  const syn = datasets.filter((d) => d.source === 'synthetic'), loc = datasets.filter((d) => d.source === 'local');
  const cur = datasets.find((d) => keyOf(d) === value.key);
  const set = (patch) => onChange({ ...value, ...patch });
  return html`<div class="grid gap-4">
    <${Field} label="Dataset" hint=${cur && cur.description}>
      <${Select} value=${value.key} onChange=${(k) => set({ key: k, params: {} })} aria-label="Dataset" options=${[...syn.map((d) => ({ value: keyOf(d), label: d.name, group: 'Bundled scenarios' })), ...loc.map((d) => ({ value: keyOf(d), label: d.name, group: 'Your data' }))]} /><//>
    ${cur && cur.source === 'synthetic' && html`<div class="grid grid-cols-3 gap-3">
      ${['n_series', 'length', 'seed'].filter((k) => k in cur.params).map((k) => html`<${Field} label=${k.replace('_', ' ')}>
        <${NumberInput} value=${value.params[k] ?? cur.params[k]} min=${k === 'seed' ? 0 : 1} onChange=${(v) => set({ params: { ...value.params, [k]: v ?? cur.params[k] } })} /><//>`)}</div>`}
    ${cur && cur.contamination_risk !== 'none' && html`<p class="text-body-small text-text-warning">Possible contamination: ${cur.contamination_risk === 'unknown' ? 'the model’s real training data is not disclosed, so results on data that may be public are not guaranteed to be out-of-sample.' : cur.contamination_risk}</p>`}
  </div>`;
}
