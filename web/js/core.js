// Shared plumbing: tiny store, API client, theme, hash router, formatters.
import { useState, useEffect } from '/vendor/htm-preact.js';

// ---------------------------------------------------------------- store
const state = { device: 'cpu', theme: 'dark', devices: [], busy: false };
const subs = new Set();
try { state.device = localStorage.getItem('tsfm-device') || 'cpu'; state.theme = localStorage.getItem('tsfm-theme') || 'dark'; } catch (e) {}
export function setState(patch) { Object.assign(state, patch); subs.forEach((f) => f()); }
export function useStore() {
  const [, tick] = useState(0);
  useEffect(() => { const f = () => tick((n) => n + 1); subs.add(f); return () => subs.delete(f); }, []);
  return state;
}

// ---------------------------------------------------------------- theme (System / Light / Dark, default Dark)
export function applyTheme(choice) {
  const dark = choice === 'dark' || (choice === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
  const m = document.querySelector('meta[name=theme-color]');
  if (m) m.content = dark ? '#101013' : '#fbfbf8';
  try { localStorage.setItem('tsfm-theme', choice); } catch (e) {}
  setState({ theme: choice });
  window.dispatchEvent(new Event('themechange'));
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => state.theme === 'system' && applyTheme('system'));
export function useThemeTick() {
  const [n, set] = useState(0);
  useEffect(() => { const f = () => set((x) => x + 1); window.addEventListener('themechange', f); return () => window.removeEventListener('themechange', f); }, []);
  return n;
}
export function setDevice(device) { try { localStorage.setItem('tsfm-device', device); } catch (e) {} setState({ device }); }

// ---------------------------------------------------------------- API
export async function api(path, { method = 'GET', body, raw } = {}) {
  const init = { method, headers: {} };
  if (raw !== undefined) { init.body = raw.data; Object.assign(init.headers, raw.headers); }
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers['Content-Type'] = 'application/json'; }
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ') : `Request failed (${res.status})`);
  }
  return data;
}
export async function waitForJob(id, onUpdate, signal) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`);
    onUpdate && onUpdate(job);
    if (['done', 'error', 'cancelled'].includes(job.status)) return job;
    if (signal && signal.aborted) return job;
    await new Promise((r) => setTimeout(r, 500));
  }
}
export function useAsync(fn, deps) {
  const [s, set] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let alive = true;
    set((p) => ({ ...p, loading: true, error: null }));
    fn().then((data) => alive && set({ loading: false, data, error: null }), (error) => alive && set({ loading: false, data: null, error }));
    return () => { alive = false; };
  }, deps);
  return [s, () => fn().then((data) => set({ loading: false, data, error: null }))];
}

// ---------------------------------------------------------------- router (#/page/arg?query)
export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  const [page = '', ...rest] = path.split('/');
  return { page: page || 'overview', args: rest, query: Object.fromEntries(new URLSearchParams(query)) };
}
export function useRoute() {
  const [r, set] = useState(parseHash());
  useEffect(() => { const f = () => { set(parseHash()); window.scrollTo(0, 0); }; addEventListener('hashchange', f); return () => removeEventListener('hashchange', f); }, []);
  return r;
}
export const go = (path) => { location.hash = '#/' + path.replace(/^#?\/?/, ''); };

// ---------------------------------------------------------------- formatters
export const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Math.abs(x) >= 1e5 ? x.toExponential(2) : Number(x).toFixed(d));
export const pct = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '–' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`);
export const pct0 = (x, d = 0) => (x === null || x === undefined ? '–' : `${(x * 100).toFixed(d)}%`);
export const ms = (x) => (x === null || x === undefined ? '–' : x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${x >= 100 ? x.toFixed(0) : x.toFixed(1)} ms`);
export const bytes = (b) => { if (!b) return '0 B'; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let v = b; while (v >= 1024 && i < 4) { v /= 1024; i++; } return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`; };
export const params = (n) => (n === null || n === undefined ? '–' : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : `${n}`);
export const when = (iso) => { try { return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }); } catch (e) { return iso || ''; } };
export const deviceLabel = (d) => (d === 'cpu' ? 'CPU' : d.toUpperCase().replace('CUDA:', 'CUDA:'));
