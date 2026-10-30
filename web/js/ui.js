// Reusable components in Mistral's visual language (hairlines, 12px cards, mono eyebrows).
import { html, useState } from '/vendor/htm-preact.js';
import { Icon } from './icons.js';
import { Select } from './select.js';
import { useStore, setDevice, deviceLabel, go } from './core.js';

export const Eyebrow = ({ children, class: c = '' }) => html`<div class=${'eyebrow ' + c}>${children}</div>`;

export function PageHeader({ eyebrow, title, children }) {
  return html`<header class="pt-8 pb-6 md:pt-14 md:pb-10 border-b border-border-primary">
    <${Eyebrow}>${eyebrow}<//>
    <h1 class="page-title mt-2">${title}</h1>
    ${children && html`<p class="mt-3 max-w-3xl text-body-large text-text-muted">${children}</p>`}
  </header>`;
}

export const Section = ({ title, aside, children, class: c = '' }) => html`<section class=${'mt-8 md:mt-12 ' + c}>
  ${(title || aside) && html`<div class="mb-4 flex items-end justify-between gap-4 border-b border-border-primary pb-4"><h2 class="section-title">${title}</h2>${aside}</div>`}
  ${children}</section>`;

export const Card = ({ children, class: c = '', tight }) => html`<div class=${(tight ? 'card-tight ' : 'card ') + c}>${children}</div>`;

export function Button({ kind = 'ghost', small, icon, children, ...rest }) {
  return html`<button type="button" class=${`btn btn-${kind} ${small ? 'btn-small' : ''}`} ...${rest}>${icon && html`<${Icon} name=${icon} size=${small ? 14 : 16} />`}${children}</button>`;
}

export function Field({ label, hint, children, class: c = '' }) {
  return html`<label class=${'block ' + c}><span class="label">${label}</span>${children}${hint && html`<span class="mt-1 block text-body-small text-text-muted">${hint}</span>`}</label>`;
}

export { Select } from './select.js';

export function Switch({ checked, onChange, label, hideLabel }) {
  return html`<button type="button" role="switch" aria-checked=${checked} aria-label=${hideLabel ? label : undefined} onClick=${() => onChange(!checked)}
    class="group inline-flex min-h-10 cursor-pointer items-center gap-3 text-body-small">
    ${!hideLabel && html`<span class="text-text-muted group-hover:text-text-primary">${label}</span>`}
    <span class=${'relative h-5 w-9 shrink-0 rounded-full transition-colors ' + (checked ? 'bg-action-primary' : 'bg-surface-brand-tertiary')}>
      <span class=${'absolute top-0.5 size-4 rounded-full transition-all ' + (checked ? 'left-[1.1rem] bg-text-invert' : 'left-0.5 bg-text-muted')}></span></span></button>`;
}

export function NumberInput({ value, onChange, min, max, step = 1, placeholder }) {
  return html`<input class="input num" type="number" inputmode="numeric" value=${value ?? ''} min=${min} max=${max} step=${step} placeholder=${placeholder}
    onInput=${(e) => { const v = e.currentTarget.value; onChange(v === '' ? null : Number(v)); }} />`;
}

export function Check({ checked, onChange, children, hint, disabled }) {
  return html`<label class=${'flex items-start gap-3 min-h-10 py-1.5 cursor-pointer ' + (disabled ? 'opacity-50 pointer-events-none' : '')}>
    <input type="checkbox" class="mt-1 size-4 shrink-0 accent-[var(--color-surface-brand)]" checked=${checked} disabled=${disabled} onChange=${(e) => onChange(e.currentTarget.checked)} />
    <span class="text-body-large leading-6">${children}${hint && html`<span class="block text-body-small text-text-muted">${hint}</span>`}</span></label>`;
}

// `cols` or `fill` make the tabs share the width of a narrow form column in equal cells instead of wrapping raggedly.
export function Tabs({ value, onChange, items, label, cols, fill }) {
  return html`<div class=${cols || fill ? 'tabs tabs-fill' : 'tabs'} style=${cols ? `grid-template-columns:repeat(${cols},minmax(0,1fr))` : undefined} role="tablist" aria-label=${label}>${items.map((it) => html`<button type="button" role="tab" class="tab" aria-selected=${it.value === value} onClick=${() => onChange(it.value)}>${it.label}</button>`)}</div>`;
}

export const Tag = ({ children }) => html`<span class="tag">${children}</span>`;

export function Notice({ kind = 'info', title, children }) {
  const tone = { info: 'bg-surface-informative text-text-informative', warning: 'bg-surface-warning text-text-warning', error: 'bg-surface-destructive text-text-destructive', success: 'bg-surface-success text-text-success' }[kind];
  const icon = { info: 'info', warning: 'warn', error: 'warn', success: 'check' }[kind];
  return html`<div class=${'flex gap-3 rounded-md px-4 py-3 text-body-small ' + tone} role=${kind === 'error' ? 'alert' : 'note'}>
    <${Icon} name=${icon} size=${16} class="mt-0.5 shrink-0" />
    <div class="min-w-0 text-text-primary">${title && html`<div class="font-medium">${title}</div>`}${children}</div></div>`;
}

export function Spinner({ label = 'Loading…' }) {
  return html`<div class="flex items-center gap-3 py-8 text-text-muted"><span class="inline-grid grid-cols-2 gap-0.5" aria-hidden="true">
    ${[0, 1, 2, 3].map((i) => html`<span class="size-2 bg-surface-brand animate-pulse" style=${`animation-delay:${i * 150}ms`}></span>`)}</span>${label}</div>`;
}

export function Progress({ done, total, message }) {
  const p = total ? Math.min(100, (done / total) * 100) : null;
  return html`<div role="progressbar" aria-valuenow=${p ?? undefined} aria-valuemin="0" aria-valuemax="100">
    <div class="relative h-2 w-full overflow-hidden rounded-sm bg-surface-brand-tertiary">${p === null ? html`<div class="sweep absolute inset-y-0 w-2/5 bg-surface-brand"></div>` : html`<div class="h-full bg-surface-brand transition-all" style=${`width:${p}%`}></div>`}</div>
    <div class="mt-2 flex justify-between gap-4 text-body-small text-text-muted"><span class="truncate">${message || ''}</span><span class="num shrink-0">${total ? `${done}/${total}` : ''}</span></div></div>`;
}

export function Table({ head, rows, empty = 'Nothing here yet.', class: c = '' }) {
  if (!rows.length) return html`<p class="py-6 text-text-muted">${empty}</p>`;
  return html`<div class=${'-mx-1 overflow-x-auto ' + c}><table class="data-table"><thead><tr>${head.map((h) => html`<th>${h}</th>`)}</tr></thead>
    <tbody>${rows.map((r) => html`<tr>${r.map((cell) => html`<td class="num">${cell}</td>`)}</tr>`)}</tbody></table></div>`;
}

export function Stat({ label, value, sub }) {
  return html`<div class="min-w-0"><${Eyebrow}>${label}<//><div class="mt-1 font-mistral text-h4 num truncate">${value}</div>${sub && html`<div class="text-body-small text-text-muted truncate">${sub}</div>`}</div>`;
}

export function KV({ items }) {
  return html`<dl class="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-body-small">${items.map(([k, v]) => html`<dt class="text-text-muted">${k}</dt><dd class="num min-w-0 break-words">${v}</dd>`)}</dl>`;
}

export function Collapse({ title, children, open: o = false }) {
  const [open, set] = useState(o);
  return html`<div class="border-t border-border-primary"><button type="button" class="flex w-full items-center justify-between gap-3 py-3 text-left min-h-10" aria-expanded=${open} onClick=${() => set(!open)}>
    <span>${title}</span><span class=${'font-mono text-text-muted transition-transform ' + (open ? 'rotate-90' : '')}>›</span></button>${open && html`<div class="pb-4">${children}</div>`}</div>`;
}

// A page whose data could not be loaded (a run that was deleted, a link with a wrong id): says what happened and leads back.
export function PageError({ error, eyebrow, title, back, backLabel = 'Back' }) {
  return html`<div><${PageHeader} eyebrow=${eyebrow} title=${title}>${error.message || String(error)}<//>
    ${back && html`<div class="mt-6"><${Button} kind="outline" onClick=${() => go(back)}>${backLabel}<//></div>`}</div>`;
}

export function ErrorBox({ error }) {
  return error ? html`<${Notice} kind="error" title="Something went wrong">${error.message || String(error)}<//>` : null;
}

export function download(filename, content, type = 'text/plain') {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([content], { type })), download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** What the compute-device menu says: 'CPU', 'GPU · RTX 5060 Ti', 'GPU · Apple M5'. Technical ids and full specs live on the System page. */
export function friendlyDevice(d, all) {
  if (d.id === 'cpu') return 'CPU';
  const gpus = all.filter((x) => x.kind !== 'cpu');
  const n = gpus.length > 1 ? ` ${gpus.indexOf(d) + 1}` : '';  // only numbered when there are several
  const model = d.name.replace(/^NVIDIA (GeForce )?/, '').replace(/ GPU( \(Metal\))?$/, '').replace(/^CUDA$/, '');
  return `GPU${n}${model ? ' · ' + model : ''}${d.supported ? '' : ' (unavailable)'}`;
}

export function DeviceSelect({ class: c = '', compact }) {
  const { device, devices } = useStore();
  const list = devices.length ? devices : [{ id: 'cpu', name: 'CPU', supported: true }];
  const options = list.map((d) => ({ value: d.id, disabled: !d.supported, text: d.id === 'cpu' ? 'CPU' : 'GPU', label: friendlyDevice(d, list) }));
  return html`<div class=${'flex items-center gap-2 ' + c}><span class=${compact ? 'sr-only' : 'label !mb-0 shrink-0'}>Compute device</span>
    <${Select} compact=${compact} value=${device} onChange=${setDevice} options=${options} aria-label="Compute device" class=${compact ? 'w-52' : 'min-w-0 flex-1'} /></div>`;
}
