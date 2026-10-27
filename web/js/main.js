import { html, render, useState, useEffect } from '/vendor/htm-preact.js';
import { api, applyTheme, setState, setDevice, useStore, useRoute } from './core.js';
import { DeviceSelect } from './ui.js';
import { Icon, Logo } from './icons.js';
import { Overview } from './pages/overview.js';
import { Showcase } from './pages/showcase.js';
import { Models } from './pages/models.js';
import { Datasets } from './pages/datasets.js';
import { Playground } from './pages/playground.js';
import { Experiments } from './pages/experiments.js';
import { Runs } from './pages/runs.js';
import { System } from './pages/system.js';
import { About } from './pages/about.js';
import { Compare } from './pages/compare.js';
import { ActivityPanel, useJobPoller } from './activity.js';
import { DialogHost } from './dialog.js';

const PAGES = [
  { id: 'showcase', label: 'Showcase', icon: 'play', C: Showcase },
  { id: 'overview', label: 'Overview', icon: 'home', C: Overview, hidden: true },
  { id: 'models', label: 'Models', icon: 'models', C: Models },
  { id: 'datasets', label: 'Datasets', icon: 'data', C: Datasets },
  { id: 'playground', label: 'Playground', icon: 'play', C: Playground },
  { id: 'experiments', label: 'Experiments', icon: 'chart', C: Experiments },
  { id: 'runs', label: 'Runs', icon: 'history', C: Runs },
  { id: 'system', label: 'System', icon: 'chip', C: System },
  { id: 'about', label: 'About', icon: 'info', C: About },
  { id: 'compare', label: 'Compare', icon: 'chart', C: Compare, hidden: true },
];

function ThemeSwitch() {
  const { theme } = useStore();
  const items = [['system', 'system', 'System theme'], ['light', 'sun', 'Light theme'], ['dark', 'moon', 'Dark theme']];
  return html`<div class="flex items-center gap-0.5 rounded-md border border-border-primary p-0.5" role="group" aria-label="Theme">
    ${items.map(([v, icon, label]) => html`<button type="button" aria-label=${label} aria-pressed=${theme === v} title=${label}
      class=${'grid size-8 place-items-center rounded-sm cursor-pointer transition-colors ' + (theme === v ? 'bg-action-primary text-text-invert' : 'text-text-muted hover:bg-action-ghost-1')}
      onClick=${() => applyTheme(v)}><${Icon} name=${icon} size=${14} /></button>`)}</div>`;
}

function Nav({ page }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [page]);
  const link = (p, cls = '') => html`<a href=${'#/' + p.id} aria-current=${page === p.id ? 'page' : undefined}
    class=${`flex h-12 items-center gap-2 px-3 text-button-large font-medium transition-colors hover:bg-action-ghost-1 ${page === p.id ? 'bg-surface-brand-secondary' : ''} ${cls}`}>${p.label}</a>`;
  return html`<nav class="sticky top-0 z-30 border-b border-border-primary bg-surface-brand-primary/95 backdrop-blur" aria-label="Main">
    <div class="mx-auto flex h-12 max-w-6xl items-stretch border-x border-border-primary">
      <a href="#/showcase" class="flex shrink-0 items-center gap-2.5 whitespace-nowrap px-4 font-mistral text-[1.0625rem] font-medium" aria-label="TSFM-Lab home"><${Logo} size=${22} />TSFM-Lab</a>
      <div class="hidden min-w-0 items-stretch overflow-hidden border-l border-border-primary xl:flex">${PAGES.filter((p) => p.id !== 'about' && !p.hidden).map((p) => html`<div class="border-r border-border-primary">${link(p)}</div>`)}</div>
      <div class="ml-auto flex shrink-0 items-stretch">
        <div class="hidden items-center border-l border-border-primary px-3 sm:flex"><${DeviceSelect} compact /></div>
        <div class="hidden items-center border-l border-border-primary px-3 md:flex"><${ThemeSwitch} /></div>
        <div class="flex items-center border-l border-border-primary px-3 xl:hidden"><button type="button" class="btn btn-outline btn-small" aria-label=${open ? 'Close menu' : 'Open menu'} aria-expanded=${open} onClick=${() => setOpen(!open)}><${Icon} name=${open ? 'close' : 'menu'} size=${16} /></button></div>
      </div>
    </div>
    ${open && html`<div class="border-t border-border-primary bg-surface-brand-primary xl:hidden"><div class="mx-auto max-w-6xl border-x border-border-primary">
      ${PAGES.filter((p) => !p.hidden).map((p) => html`<div class="border-b border-border-primary last:border-0">${link(p, 'w-full')}</div>`)}
      <div class="flex flex-wrap items-center justify-between gap-3 border-t border-border-primary p-4"><${DeviceSelect} class="sm:hidden w-full" /><${ThemeSwitch} /></div></div></div>`}
  </nav>`;
}

function Footer() {
  return html`<footer class="border-t border-border-primary"><div class="mx-auto flex max-w-6xl flex-col gap-4 border-x border-border-primary px-4 py-8 text-body-small text-text-muted md:flex-row md:items-center md:justify-between md:px-8">
    <p class="max-w-xl">Independent tool, not affiliated with or endorsed by Mistral AI. Runs entirely on this machine; nothing is uploaded.</p>
    <a class="underline-offset-4 hover:underline" href="#/about">Licenses & definitions</a></div></footer>`;
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
