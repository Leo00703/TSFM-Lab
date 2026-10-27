// Global activity: every running (or just finished) job is visible on every page, survives page changes and reloads
// (jobs live on the server), and can be stopped. Pages read the same list, so nothing "disappears" when you navigate.
import { html, useState, useEffect } from '/vendor/htm-preact.js';
import { api, setState, useStore, go } from './core.js';
import { Button, Progress } from './ui.js';
import { Icon } from './icons.js';

const ACTIVE = ['queued', 'running'];
export const isActive = (j) => ACTIVE.includes(j.status);

export function useJobPoller() {
  useEffect(() => {
    let alive = true, timer;
    const tick = async () => {
      let delay = 4000;
      try {
        const jobs = await api('/api/jobs');
        if (alive) setState({ jobs, jobsLoaded: true });
        if (jobs.some(isActive)) delay = 1200;
      } catch (e) { delay = 6000; }
      if (alive) timer = setTimeout(tick, delay);
    };
    tick();
    const wake = () => { clearTimeout(timer); tick(); };
    addEventListener('focus', wake);
    return () => { alive = false; clearTimeout(timer); removeEventListener('focus', wake); };
  }, []);
}

export const stopJob = (id) => api(`/api/jobs/${id}/cancel`, { method: 'POST' }).catch(() => {});

const KIND = { benchmark: 'Benchmark', experiment: 'Experiment', download: 'Download', showcase: 'Measuring' };
const WHERE = (j) => (j.status === 'done' && j.kind === 'experiment' && j.result && j.result.run_id ? `runs/${j.result.run_id}`
  : { benchmark: 'system', experiment: 'experiments', download: 'models', showcase: 'showcase' }[j.kind] || 'system');

export function ActivityPanel() {
  const { jobs = [] } = useStore();
  const [dismissed, setDismissed] = useState(() => { try { return JSON.parse(sessionStorage.getItem('tsfm-dismissed') || '[]'); } catch (e) { return []; } });
  const [open, setOpen] = useState(true);
  const now = Date.now() / 1000;
  const recent = jobs.filter((j) => !isActive(j) && !dismissed.includes(j.id) && j.finished && now - j.finished < 180 && j.kind !== 'showcase' && j.status !== 'cancelled');
  const shown = [...jobs.filter(isActive), ...recent];
  if (!shown.length) return null;
  const dismiss = (id) => { const d = [...dismissed, id]; setDismissed(d); try { sessionStorage.setItem('tsfm-dismissed', JSON.stringify(d)); } catch (e) { /* ignore */ } };
  const running = jobs.filter(isActive).length;
  return html`<aside aria-label="Activity" class="fixed inset-x-3 bottom-3 z-40 sm:inset-x-auto sm:right-4 sm:bottom-4 sm:w-[22rem]">
    <div class="overflow-hidden rounded-xl border border-border-secondary bg-surface-brand-secondary shadow-xl">
      <button type="button" class="flex min-h-10 w-full items-center justify-between gap-3 px-4 py-2 text-left" aria-expanded=${open} onClick=${() => setOpen(!open)}>
        <span class="flex items-center gap-2 font-mono text-[0.6875rem] uppercase tracking-[.02em] text-text-muted">
          ${running > 0 && html`<span class="size-2 animate-pulse bg-surface-brand"></span>`}Activity${running ? ` · ${running} running` : ''}</span>
        <${Icon} name="chevron" size=${10} class=${'text-text-muted transition-transform ' + (open ? '' : 'rotate-180')} /></button>
      ${open && html`<ul class="max-h-[60vh] divide-y divide-border-primary overflow-y-auto border-t border-border-primary">${shown.map((j) => html`<li class="p-4">
        <div class="flex items-start justify-between gap-3"><div class="min-w-0"><span class="eyebrow">${KIND[j.kind] || j.kind} · ${j.status}</span>
          <div class="mt-0.5 truncate text-body-small font-medium" title=${j.label}>${j.label}</div></div>
          ${!isActive(j) && html`<button type="button" class="btn btn-ghost btn-small size-8 !p-0" aria-label="Dismiss" onClick=${() => dismiss(j.id)}><${Icon} name="close" size=${12} /></button>`}</div>
        ${isActive(j) ? html`<div class="mt-3"><${Progress} done=${j.done} total=${j.total} message=${j.status === 'queued' ? 'waiting for the current job…' : j.message} /></div>
          <div class="mt-3 flex gap-2"><${Button} small kind="danger" onClick=${() => stopJob(j.id)}>Stop<//><${Button} small kind="outline" onClick=${() => go(WHERE(j))}>Open page<//></div>`
        : j.status === 'error' ? html`<p class="mt-2 text-body-small text-text-destructive">${j.error}</p>`
        : html`<div class="mt-3"><${Button} small kind="outline" onClick=${() => { dismiss(j.id); go(WHERE(j)); }}>${j.kind === 'experiment' ? 'Open the run' : 'Open'}<//></div>`}</li>`)}</ul>`}
    </div></aside>`;
}
