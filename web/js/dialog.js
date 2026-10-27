// A confirmation dialog in the app's own design language, replacing the browser's native confirm().
//
//   if (await confirmDialog({ title: 'Delete this run?', body: '…', confirmLabel: 'Delete', danger: true, run: () => api(…) })) reload();
//
// Resolves true when confirmed (after `run` succeeded, if given), false when cancelled. While `run` works the dialog shows
// it and cannot be dismissed; if it fails the error is shown inside the dialog and the action can be retried.
// Accessible: alertdialog, focus starts on the safe choice (Cancel), Tab is kept inside, Escape and a click outside cancel,
// the page behind does not scroll, and focus returns to the button that opened it.
import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { Icon } from './icons.js';

const pending = [];
let host = null, seq = 0;

export function confirmDialog(options) {
  return new Promise((resolve) => { pending.push({ ...options, resolve, n: ++seq }); if (host) host(); });
}

let uid = 0;

function Dialog({ item, done }) {
  const [busy, setBusy] = useState(false), [error, setError] = useState(null);
  const panel = useRef(null), cancel = useRef(null), opener = useRef(document.activeElement), id = useRef('dlg' + ++uid);
  useEffect(() => {
    if (cancel.current) cancel.current.focus();
    const before = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = before; if (opener.current && opener.current.focus && document.contains(opener.current)) opener.current.focus(); };
  }, []);
  const confirm = async () => {
    if (!item.run) return done(true);
    setBusy(true); setError(null);
    try { await item.run(); done(true); } catch (e) { setError(e); setBusy(false); }
  };
  const onKey = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); if (!busy) done(false); return; }
    if (e.key !== 'Tab') return;
    const items = [...panel.current.querySelectorAll('button:not([disabled])')];
    if (!items.length) { e.preventDefault(); return; }
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  const danger = item.danger !== false;
  return html`<div class="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-3 backdrop-blur-[2px] sm:items-center sm:p-4"
    onPointerDown=${(e) => { if (e.target === e.currentTarget && !busy) done(false); }} onKeyDown=${onKey}>
    <div ref=${panel} role="alertdialog" aria-modal="true" aria-labelledby=${id.current + 't'} aria-describedby=${id.current + 'd'}
      class="dialog-in w-full max-w-md rounded-xl border border-border-secondary bg-surface-brand-secondary p-5 shadow-2xl md:p-6">
      <div class="flex gap-4">
        <span class=${'grid size-10 shrink-0 place-items-center rounded-md ' + (danger ? 'bg-surface-destructive text-text-destructive' : 'bg-surface-brand-tertiary text-text-primary')}><${Icon} name=${danger ? 'trash' : 'info'} size=${18} /></span>
        <div class="min-w-0">
          <h2 id=${id.current + 't'} class="font-mistral text-h5">${item.title}</h2>
          <div id=${id.current + 'd'} class="mt-2 text-body-small text-text-muted">${item.body}</div>
          ${error && html`<p class="mt-3 break-words text-body-small text-text-destructive" role="alert">${error.message || String(error)}</p>`}</div></div>
      <div class="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <button ref=${cancel} type="button" class="btn btn-outline" disabled=${busy} onClick=${() => done(false)}>${item.cancelLabel || 'Cancel'}</button>
        <button type="button" class=${'btn ' + (danger ? 'btn-destructive' : 'btn-primary')} disabled=${busy} onClick=${confirm}>${busy ? (item.busyLabel || 'Working…') : error ? 'Try again' : (item.confirmLabel || 'Confirm')}</button></div>
    </div></div>`;
}

/** Mounted once, in the app shell. */
export function DialogHost() {
  const [item, setItem] = useState(null);
  const shown = useRef(null);
  useEffect(() => {
    host = () => { if (!shown.current && pending.length) { shown.current = pending.shift(); setItem(shown.current); } };
    host();
    return () => { host = null; };
  }, []);
  const done = (ok) => { const current = shown.current; shown.current = null; setItem(null); current.resolve(ok); host && host(); };
  return item ? html`<${Dialog} key=${item.n} item=${item} done=${done} />` : null;
}
