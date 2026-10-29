import { html, render, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { api, applyTheme, setState, setDevice, useStore, useRoute } from './core.js';
import { DeviceSelect } from './ui.js';
import { Icon, Logo } from './icons.js';
import { Showcase } from './pages/showcase.js';
import { Models } from './pages/models.js';
import { Datasets } from './pages/datasets.js';
import { Playground } from './pages/playground.js';
import { Experiments } from './pages/experiments.js';
import { Runs } from './pages/runs.js';
import { Leaderboard } from './pages/leaderboard.js';
import { System } from './pages/system.js';
import { About } from './pages/about.js';
import { Compare } from './pages/compare.js';
import { ActivityPanel, useJobPoller } from './activity.js';
import { DialogHost } from './dialog.js';

const PAGES = [
  { id: 'showcase', label: 'Showcase', icon: 'play', C: Showcase },
  { id: 'models', label: 'Models', icon: 'models', C: Models },
  { id: 'datasets', label: 'Datasets', icon: 'data', C: Datasets },
  { id: 'playground', label: 'Playground', icon: 'play', C: Playground },
  { id: 'experiments', label: 'Experiments', icon: 'chart', C: Experiments },
  { id: 'leaderboard', label: 'Leaderboard', icon: 'chart', C: Leaderboard },
  { id: 'runs', label: 'Runs', icon: 'history', C: Runs },
  { id: 'system', label: 'System', icon: 'chip', C: System },
  { id: 'about', label: 'About', icon: 'info', C: About },
  { id: 'compare', label: 'Compare', icon: 'chart', C: Compare, hidden: true },
];

// Top bar: the pages people open every day stand alone, the three ways of testing a model share one menu (a flat row of
// eight did not fit with room to breathe). Cells follow mistral.ai's bar: 48 px, hairlines between cells, 16 px labels, 16 px padding.
const MENU = [
  { id: 'showcase' }, { id: 'models' }, { id: 'datasets' },
  { label: 'Evaluate', items: ['playground', 'experiments', 'leaderboard'] },
  { id: 'runs' }, { id: 'system' },
];
const BLURB = {
  playground: 'Forecast one series and read every quantile',
  experiments: 'Rolling backtests against simple baselines',
  leaderboard: 'Rank the models on a benchmark or your data',
};
const CELL = 'flex h-12 items-center gap-2 px-4 font-mistral text-base font-medium tracking-[0.01em] whitespace-nowrap transition-colors hover:bg-action-ghost-1 cursor-pointer';

function ThemeSwitch() {
  const { theme } = useStore();
  const items = [['system', 'system', 'System theme'], ['light', 'sun', 'Light theme'], ['dark', 'moon', 'Dark theme']];
  return html`<div class="flex items-center gap-1 rounded-md border border-border-primary p-1" role="group" aria-label="Theme">
    ${items.map(([v, icon, label]) => html`<button type="button" aria-label=${label} aria-pressed=${theme === v} title=${label}
      class=${'grid size-8 place-items-center rounded-sm cursor-pointer transition-colors ' + (theme === v ? 'bg-action-primary text-text-invert' : 'text-text-muted hover:bg-action-ghost-1')}
      onClick=${() => applyTheme(v)}><${Icon} name=${icon} size=${14} /></button>`)}</div>`;
}

function NavGroup({ group, page }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null), button = useRef(null);
  const current = group.items.includes(page);
  useEffect(() => setOpen(false), [page]);
  useEffect(() => {
    if (!open) return undefined;
    const away = (e) => { if (root.current && !root.current.contains(e.target)) setOpen(false); };
    const key = (e) => { if (e.key === 'Escape') { setOpen(false); if (button.current) button.current.focus(); } };
    document.addEventListener('pointerdown', away);
    document.addEventListener('keydown', key);
    return () => { document.removeEventListener('pointerdown', away); document.removeEventListener('keydown', key); };
  }, [open]);
  return html`<div ref=${root} class="relative border-r border-border-primary">
    <button ref=${button} type="button" aria-expanded=${open} aria-controls="nav-evaluate" onClick=${() => setOpen(!open)}
      class=${CELL + ' ' + (current || open ? 'bg-surface-brand-secondary' : '')}>${group.label}<${Icon} name="chevron" size=${10} class=${'text-text-muted transition-transform ' + (open ? 'rotate-180' : '')} /></button>
    ${open && html`<div id="nav-evaluate" class="menu-in absolute left-0 top-full z-40 mt-px w-[22rem] rounded-b-xl border border-t-0 border-border-primary bg-surface-brand-secondary p-2 shadow-xl">
      ${group.items.map((id) => { const p = PAGES.find((x) => x.id === id); return html`<a href=${'#/' + id} aria-current=${page === id ? 'page' : undefined}
        class="block rounded-md px-3 py-3 transition-colors hover:bg-action-ghost-1 aria-[current=page]:bg-action-ghost-1">
        <span class="flex items-center gap-2 font-mistral text-base font-medium">${page === id && html`<span class="size-1.5 bg-surface-brand" aria-hidden="true"></span>`}${p.label}</span>
        <span class="mt-0.5 block text-body-small text-text-muted">${BLURB[id]}</span></a>`; })}</div>`}</div>`;
}

function Nav({ page }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [page]);
  const link = (p, cls = '') => html`<a href=${'#/' + p.id} aria-current=${page === p.id ? 'page' : undefined} class=${`${CELL} ${page === p.id ? 'bg-surface-brand-secondary' : ''} ${cls}`}>${p.label}</a>`;
  return html`<nav class="sticky top-0 z-30 border-b border-border-primary bg-surface-brand-primary/95 backdrop-blur" aria-label="Main">
    <div class="mx-auto flex h-12 max-w-6xl items-stretch border-x border-border-primary">
      <a href="#/showcase" class="flex shrink-0 items-center gap-2.5 whitespace-nowrap px-4 font-mistral text-[1.0625rem] font-medium" aria-label="TSFM-Lab home"><${Logo} size=${22} />TSFM-Lab</a>
      <div class="hidden min-w-0 items-stretch border-l border-border-primary xl:flex">${MENU.map((m) => (m.items ? html`<${NavGroup} group=${m} page=${page} />` : html`<div class="border-r border-border-primary">${link(PAGES.find((p) => p.id === m.id))}</div>`))}</div>
      <div class="ml-auto flex shrink-0 items-stretch">
        <div class="hidden items-center border-l border-border-primary px-4 sm:flex"><${DeviceSelect} compact /></div>
        <div class="hidden items-center border-l border-border-primary px-4 md:flex"><${ThemeSwitch} /></div>
        <div class="flex items-center border-l border-border-primary px-4 xl:hidden"><button type="button" class="btn btn-outline" aria-label=${open ? 'Close menu' : 'Open menu'} aria-expanded=${open} onClick=${() => setOpen(!open)}><${Icon} name=${open ? 'close' : 'menu'} size=${16} /></button></div>
      </div>
    </div>
    <div class=${'grid bg-surface-brand-primary transition-[grid-template-rows,opacity] duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none xl:hidden ' + (open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0')} inert=${!open} aria-hidden=${!open}>
      <div class="min-h-0 overflow-hidden"><div class="border-t border-border-primary"><div class="mx-auto max-w-6xl border-x border-border-primary">
        ${PAGES.filter((p) => !p.hidden).map((p) => html`<div class="border-b border-border-primary last:border-0">${link(p, 'w-full')}</div>`)}
        <div class="flex flex-wrap items-center justify-between gap-4 border-t border-border-primary p-4"><${DeviceSelect} class="sm:hidden w-full" /><${ThemeSwitch} /></div></div></div></div></div>
  </nav>`;
}

function Footer() {
  return html`<footer class="border-t border-border-primary"><div class="mx-auto flex max-w-6xl flex-col gap-4 border-x border-border-primary px-4 py-8 text-body-small text-text-muted md:flex-row md:items-center md:justify-between md:px-8">
    <p class="max-w-xl">Independent tool, not affiliated with or endorsed by Mistral AI. Runs entirely on this machine; nothing is uploaded.</p>
    <a class="underline-offset-4 hover:underline" href="#/about">Licences & definitions</a></div></footer>`;
}

function App() {
  useJobPoller();
  const route = useRoute();  // one subscription: two of them could update in different renders and hand a page the previous route
  const { page } = route;
  useEffect(() => {
    api('/api/system').then((s) => {
      const devices = s.devices;
      setState({ devices, busy: s.busy });
      const device = localStorage.getItem('tsfm-device') || 'cpu';
      if (!devices.some((d) => d.id === device && d.supported)) setDevice('cpu');
    }).catch(() => {});
  }, []);
  const current = PAGES.find((p) => p.id === page) || PAGES[0];
  return html`<div class="min-h-dvh flex flex-col"><a href="#main" class="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:bg-surface-brand-primary focus:p-3">Skip to content</a>
    <${Nav} page=${current.id} />
    <main id="main" class="mx-auto w-full max-w-6xl flex-1 border-x border-border-primary px-4 pb-16 md:px-8"><${current.C} route=${route} key=${current.id + (route.query.ds || '') + (route.query.runs || '')} /></main>
    <${Footer} /><${ActivityPanel} /><${DialogHost} /></div>`;
}

applyTheme(localStorage.getItem('tsfm-theme') || 'dark');
// Tab icon: the Timestral icon when present locally (git-ignored asset); the built-in pixel mark otherwise.
const probe = new Image();
probe.onload = () => { const l = document.querySelector('link[rel=icon]'); l.type = 'image/svg+xml'; l.href = probe.src; };
probe.src = '/assets/mistral/logos/timestral-icon-model.svg';
render(html`<${App} />`, document.getElementById('app'));
