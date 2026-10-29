// Global activity: every running (or just finished) job is visible on every page, survives page changes and reloads
// (jobs live on the server), and can be stopped. Pages read the same list, so nothing "disappears" when you navigate.
//
// The panel shows one notification per job: a status icon, a plain-language title, a progress bar that is honest about
// what is known (a bar and a count when steps are known, a sweep when they are not, elapsed time, never a guessed
// "time left"), and the one action that matters. Finished notifications leave by themselves (pausing while you hover or
// focus them), failures stay until dismissed, and stopping a job is acknowledged immediately.
import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, setState, useStore, useRoute, pathOf, go } from './core.js';
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

// ---------------------------------------------------------------------------------------------- wording
const TITLES = {
  download: { queued: 'Download queued', running: 'Downloading', done: 'Downloaded', error: 'Download failed', cancelled: 'Download stopped' },
  benchmark: { queued: 'Benchmark queued', running: 'Benchmarking', done: 'Benchmark finished', error: 'Benchmark failed', cancelled: 'Benchmark stopped' },
  experiment: { queued: 'Backtest queued', running: 'Running a backtest', done: 'Backtest finished', error: 'Backtest failed', cancelled: 'Backtest stopped' },
  suite: { queued: 'Leaderboard run queued', running: 'Ranking the models', done: 'Leaderboard run finished', error: 'Leaderboard run failed', cancelled: 'Leaderboard run stopped' },
  showcase: { queued: 'Measurement queued', running: 'Measuring the effect of the factors', done: 'Measurement finished', error: 'Measurement failed', cancelled: 'Measurement stopped' },
};
const detail = (j) => (j.kind === 'download' ? j.label.replace(/^Download /, '') : j.kind === 'benchmark' ? j.label.replace(/^Benchmark /, '')
  : j.kind === 'suite' ? j.label.replace(/^Leaderboard: /, '')
  : j.kind === 'showcase' ? j.label.split(' · ').slice(1).join(' · ') : j.label);
// The server says where each job's outcome lives (`link`, e.g. 'models?focus=timestral-209m'); a finished backtest leads to its run.
// A button is only offered when it would take you somewhere else, and it lands on the exact thing, not just the page.
const WHERE = { benchmark: 'system', suite: 'leaderboard', experiment: 'experiments', download: 'models', showcase: 'showcase' };
const target = (j) => (j.kind === 'experiment' && j.status === 'done' && j.result && j.result.run_id ? `runs/${j.result.run_id}` : j.kind === 'suite' && j.status === 'done' && j.result && j.result.suite_id ? `leaderboard/${j.result.suite_id}` : j.link || WHERE[j.kind] || 'system');
const pageOf = (path) => path.split('?')[0];
const OPEN_LABEL = (j) => {
  const done = j.status === 'done', page = pageOf(target(j)).split('/')[0];
  if (j.kind === 'experiment') return done ? 'Open the run' : 'Go to Experiments';
  if (j.kind === 'suite') return done ? 'Open the leaderboard' : 'Go to the leaderboard';
  if (j.kind === 'benchmark') return done ? 'See the results' : 'Go to the benchmark';
  if (j.kind === 'download') return page === 'datasets' ? 'Show the dataset' : page === 'showcase' ? 'Go to Showcase' : 'Show the model';
  return 'Go to Showcase';
};
const clock = (s) => (s < 60 ? `${Math.max(0, Math.floor(s))} s` : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')} min`);

// How long a finished notification stays (seconds); failures stay until dismissed.
const STAY = { done: 12, cancelled: 6 };
const MAX_SHOWN = (typeof innerWidth !== 'undefined' && innerWidth < 640) ? 1 : 3;  // a phone screen is small: one card, the rest behind "more"

function StatusIcon({ status }) {
  if (status === 'running') return html`<span class="inline-grid grid-cols-2 gap-0.5" aria-hidden="true">${[0, 1, 2, 3].map((i) => html`<span class="size-[7px] bg-surface-brand animate-pulse" style=${`animation-delay:${i * 150}ms`}></span>`)}</span>`;
  const [name, tone] = { queued: ['history', 'text-text-muted'], done: ['check', 'text-text-success'], error: ['warn', 'text-text-destructive'], cancelled: ['close', 'text-text-muted'] }[status];
  return html`<${Icon} name=${name} size=${16} class=${tone} />`;
}

function Bar({ job }) {
  const known = job.total > 0, p = known ? Math.min(100, (job.done / job.total) * 100) : null;
  return html`<div role="progressbar" aria-label="Progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${p === null ? undefined : Math.round(p)}
    class="relative h-1.5 w-full overflow-hidden rounded-full bg-surface-brand-tertiary">
    ${known ? html`<div class="h-full rounded-full bg-surface-brand transition-all duration-500" style=${`width:${p}%`}></div>`
      : html`<div class="sweep absolute inset-y-0 w-2/5 rounded-full bg-surface-brand"></div>`}</div>`;
}

function Toast({ job, now, left, stopping, here, onStop, onDismiss, onHover }) {
  const [more, setMore] = useState(false);
  const elsewhere = pageOf(target(job)) !== here;  // already on that page: the button would do nothing
  const live = isActive(job), failed = job.status === 'error';
  const tone = { done: 'bg-surface-success', error: 'bg-surface-destructive' }[job.status] || 'bg-surface-brand-tertiary';
  const started = job.started || job.created;
  const known = job.total > 0;
  const message = job.status === 'queued' ? 'Waiting for the job that is running now…' : stopping ? 'Stopping…' : job.message;
  const stay = STAY[job.status];
  return html`<li class="toast-in group relative overflow-hidden rounded-xl border border-border-secondary bg-surface-brand-secondary shadow-xl"
    role=${failed ? 'alert' : 'status'} onMouseEnter=${() => onHover(job.id, true)} onMouseLeave=${() => onHover(job.id, false)}
    onFocusIn=${() => onHover(job.id, true)} onFocusOut=${() => onHover(job.id, false)}
    onKeyDown=${(e) => { if (e.key === 'Escape' && !live) onDismiss(job.id); }}>
    <div class="flex gap-3 p-3.5">
      <span class=${'grid size-9 shrink-0 place-items-center rounded-md ' + tone}><${StatusIcon} status=${job.status} /></span>
      <div class="min-w-0 flex-1">
        <div class="flex items-start justify-between gap-2">
          <div class="min-w-0"><div class="text-body-small font-medium leading-5">${TITLES[job.kind]?.[job.status] || job.status}</div>
            <div class="truncate text-body-small text-text-muted" title=${job.label}>${detail(job)}</div></div>
          ${!live && html`<button type="button" class="btn btn-ghost size-7 !h-7 !min-w-7 shrink-0 !p-0" aria-label="Dismiss" onClick=${() => onDismiss(job.id)}><${Icon} name="close" size=${10} /></button>`}</div>
        ${live && html`<div class="mt-3">${job.status === 'running' && html`<${Bar} job=${job} />`}
          <div class=${'flex items-baseline justify-between gap-3 text-[0.75rem] leading-4 text-text-muted ' + (job.status === 'running' ? 'mt-1.5' : '')}>
            <span class="truncate">${message}</span>
            <span class="num shrink-0">${known ? `${job.done} of ${job.total}` : ''}${known && job.status === 'running' ? ' · ' : ''}${job.status === 'running' ? clock(now - started) : ''}</span></div></div>`}
        ${failed && html`<p class=${'mt-2 break-words text-body-small text-text-destructive ' + (more ? '' : 'line-clamp-3')}>${job.error || 'Something went wrong.'}</p>
          ${(job.error || '').length > 140 && html`<button type="button" class="mt-1 text-body-small text-text-muted underline underline-offset-2" onClick=${() => setMore(!more)}>${more ? 'Show less' : 'Show all'}</button>`}`}
        ${(live || (elsewhere && job.status !== 'cancelled')) && html`<div class="mt-3 flex flex-wrap items-center gap-2">
          ${live && html`<button type="button" class="btn btn-danger btn-small" disabled=${stopping} onClick=${() => onStop(job.id)}>${stopping ? 'Stopping…' : 'Stop'}</button>`}
          ${job.status === 'done' && elsewhere && html`<button type="button" class="btn btn-primary btn-small" onClick=${() => { onDismiss(job.id); go(target(job)); }}>${OPEN_LABEL(job)}</button>`}
          ${live && elsewhere && html`<button type="button" class="btn btn-ghost btn-small" onClick=${() => go(target(job))}>${OPEN_LABEL(job)}</button>`}
          ${failed && elsewhere && html`<button type="button" class="btn btn-outline btn-small" onClick=${() => go(target(job))}>${OPEN_LABEL(job)}</button>`}</div>`}</div></div>
    ${stay && left !== null && html`<div class="absolute inset-x-0 bottom-0 h-0.5 bg-surface-brand-tertiary"><div class="h-full bg-text-muted opacity-60 transition-[width] duration-1000 ease-linear" style=${`width:${Math.max(0, (left / stay) * 100)}%`}></div></div>`}
  </li>`;
}

export function ActivityPanel() {
  const { jobs = [] } = useStore();
  const here = pathOf(useRoute());
  const read = (k, d) => { try { return JSON.parse(sessionStorage.getItem(k) || d); } catch (e) { return JSON.parse(d); } };
  const [dismissed, setDismissed] = useState(() => read('tsfm-dismissed', '[]'));
  const [folded, setFolded] = useState(() => read('tsfm-activity-folded', 'false'));
  const [showAll, setShowAll] = useState(false);
  const [stopping, setStopping] = useState({});
  const [now, setNow] = useState(Date.now() / 1000);
  const seen = useRef({}), first = useRef(true), hovered = useRef({}), paused = useRef({});  // seen: id -> when this browser saw it finish

  // Which jobs are worth showing. A job this browser watched finishing always gets its moment; one that was already
  // finished when the page opened only if it finished moments ago (the server clock is the reference there).
  const t = Date.now() / 1000;
  for (const j of jobs) {
    if (isActive(j)) { seen.current[j.id] = seen.current[j.id] || { live: true }; continue; }
    const s = seen.current[j.id];
    if (!s) seen.current[j.id] = { at: t, late: first.current && !(j.finished && t - j.finished < 60) };
    else if (s.live) seen.current[j.id] = { at: t, late: false };
  }
  if (jobs.length) first.current = false;
  const left = (j) => { const s = seen.current[j.id], stay = STAY[j.status]; return !stay || !s || s.at === undefined ? null : stay - (now - s.at) + (paused.current[j.id] || 0); };
  const relevant = jobs.filter((j) => {
    if (isActive(j)) return true;
    const s = seen.current[j.id];
    if (!s || s.late || dismissed.includes(j.id) || (j.kind === 'showcase' && j.status === 'done')) return false;
    if (j.status === 'error') return true;
    return STAY[j.status] ? left(j) > 0 : false;
  });
  const order = (j) => (j.status === 'error' ? 0 : isActive(j) ? 1 : 2);
  const sorted = [...relevant].sort((a, b) => order(a) - order(b) || a.created - b.created);
  const needsClock = relevant.length > 0;
  useEffect(() => {
    if (!needsClock) return undefined;
    const id = setInterval(() => {
      for (const jid of Object.keys(hovered.current)) if (hovered.current[jid]) paused.current[jid] = (paused.current[jid] || 0) + 1;  // hovering stops the countdown
      setNow(Date.now() / 1000);
    }, 1000);
    return () => clearInterval(id);
  }, [needsClock]);

  const dismiss = (id) => { setDismissed((d) => { const n = [...d, id]; try { sessionStorage.setItem('tsfm-dismissed', JSON.stringify(n)); } catch (e) { /* ignore */ } return n; }); };
  const stop = (id) => { setStopping((s) => ({ ...s, [id]: true })); stopJob(id); };
  const hover = (id, on) => { hovered.current[id] = on; };
  const fold = (v) => { setFolded(v); try { sessionStorage.setItem('tsfm-activity-folded', JSON.stringify(v)); } catch (e) { /* ignore */ } };
  if (!sorted.length) return null;

  const running = sorted.filter(isActive).length, finished = sorted.length - running;
  const visible = showAll ? sorted : sorted.slice(0, MAX_SHOWN), hidden = sorted.length - visible.length;
  const summary = [running ? `${running} running` : '', finished ? `${finished} finished` : ''].filter(Boolean).join(' · ');
  return html`<aside aria-label="Activity" aria-live="polite" class="fixed inset-x-3 bottom-3 z-40 sm:inset-x-auto sm:bottom-4 sm:right-4 sm:w-[23rem]" style="padding-bottom:env(safe-area-inset-bottom)">
    ${folded ? html`<button type="button" class="toast-in ml-auto flex min-h-10 items-center gap-2.5 rounded-full border border-border-secondary bg-surface-brand-secondary px-4 text-body-small shadow-xl" aria-label=${'Show activity: ' + summary} onClick=${() => fold(false)}>
        ${running > 0 && html`<${StatusIcon} status="running" />`}<span>${summary}</span><${Icon} name="chevron" size=${10} class="rotate-180 text-text-muted" /></button>`
    : html`<div class="flex flex-col gap-2">
      ${sorted.length > 1 && html`<div class="flex items-center justify-between gap-3 rounded-full border border-border-primary bg-surface-brand-secondary py-0.5 pl-4 pr-1.5 text-[0.6875rem] font-mono uppercase tracking-[.02em] text-text-muted shadow-xl">
        <span class="whitespace-nowrap">${summary}</span>
        <span class="flex items-center gap-1">${finished > 0 && html`<button type="button" class="whitespace-nowrap rounded-sm px-1.5 py-1 hover:text-text-primary" onClick=${() => sorted.filter((j) => !isActive(j)).forEach((j) => dismiss(j.id))}>Clear</button>`}
          <button type="button" class="grid size-6 place-items-center rounded-sm hover:text-text-primary" aria-label="Minimise activity" onClick=${() => fold(true)}><${Icon} name="chevron" size=${10} /></button></span></div>`}
      <ul class="flex flex-col gap-2">${visible.map((j) => html`<${Toast} key=${j.id} job=${j} now=${now} left=${left(j)} stopping=${!!stopping[j.id]} here=${here} onStop=${stop} onDismiss=${dismiss} onHover=${hover} />`)}</ul>
      ${hidden > 0 && html`<button type="button" class="self-center rounded-full border border-border-primary bg-surface-brand-secondary px-3 py-1.5 text-body-small text-text-muted hover:text-text-primary" onClick=${() => setShowAll(true)}>${hidden} more</button>`}
      ${showAll && sorted.length > MAX_SHOWN && html`<button type="button" class="self-center rounded-full border border-border-primary bg-surface-brand-secondary px-3 py-1.5 text-body-small text-text-muted hover:text-text-primary" onClick=${() => setShowAll(false)}>Show fewer</button>`}</div>`}
  </aside>`;
}
