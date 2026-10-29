// ECharts wrappers styled with the site's chart tokens. Bars start at zero; truncated axes are never silent.
import { html, useEffect, useRef, useMemo } from '/vendor/htm-preact.js';
import { useThemeTick } from './core.js';
import { Icon } from './icons.js';

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export function palette() {
  const series = ['--color-chart-brand-1', '--color-chart-blue', '--color-chart-brand-3', '--color-chart-green', '--color-chart-lavender', '--color-chart-pink', '--color-chart-brand-4'].map(css);
  return {
    text: css('--color-text-primary'), muted: css('--color-chart-label'), grid: css('--color-chart-grid'), axis: css('--color-chart-axis'),
    bg: css('--color-surface-brand-secondary'), brand: css('--color-chart-brand-1'), series,
    neutral: [css('--color-chart-neutral-1'), css('--color-chart-neutral-2'), css('--color-chart-neutral-3')],
    font: css('--font-inter') || 'Inter, sans-serif', mono: css('--font-mono') || 'monospace',
  };
}
// the focus model is orange; baselines/others are neutral or accent colours, like the announcement charts
export function modelColors(ids, focus, p) {
  const accents = [p.series[1], p.series[3], p.series[2], p.series[4]];
  let k = 0, n = 0;
  return Object.fromEntries(ids.map((id) => [id, id === focus ? p.brand : (n++ % 2 === 0 ? p.neutral[Math.min(k++, 2)] : accents[(k++) % accents.length])]));
}

export const isNarrow = () => window.innerWidth < 640;
export function baseOption(p, extra = {}) {
  const narrow = isNarrow();
  return {
    animation: false, backgroundColor: 'transparent', textStyle: { fontFamily: p.font, color: p.text },
    grid: { left: narrow ? 44 : 56, right: narrow ? 12 : 24, top: 28, bottom: 36, containLabel: false },
    tooltip: { trigger: 'axis', confine: true, backgroundColor: p.bg, borderColor: p.axis, textStyle: { color: p.text, fontSize: 12 }, axisPointer: { type: 'line', lineStyle: { color: p.axis } } },
    ...extra,
  };
}
export const axisStyle = (p, extra = {}) => ({ axisLine: { lineStyle: { color: p.axis } }, axisTick: { show: false }, axisLabel: { color: p.muted, fontSize: 11 }, splitLine: { lineStyle: { color: p.grid } }, ...extra });

// ---------------------------------------------------------------- chart + figure
export function Chart({ build, deps = [], height = 320, name = 'chart', onInstance }) {
  const el = useRef(null), inst = useRef(null), tick = useThemeTick(), latest = useRef(build);
  latest.current = build;
  // the builder also gets the width of the box: a legend that wraps onto a second line needs room above the plot
  const make = () => latest.current({ ...palette(), width: el.current ? el.current.clientWidth : 0 });
  useEffect(() => {
    const c = echarts.init(el.current, null, { renderer: 'canvas' });
    inst.current = c; onInstance && onInstance(c);
    let last = el.current.clientWidth;
    const ro = new ResizeObserver(() => {
      c.resize();
      const w = el.current ? el.current.clientWidth : 0;
      if (Math.abs(w - last) > 24) { last = w; c.setOption(make(), true); }  // wide enough a change to re-flow the legend
    });
    ro.observe(el.current);
    return () => { ro.disconnect(); c.dispose(); inst.current = null; };
  }, []);
  useEffect(() => { if (inst.current) inst.current.setOption(make(), true); }, [tick, ...deps]);
  return html`<div ref=${el} role="img" aria-label=${name} style=${`height:${height}px;width:100%`}></div>`;
}

export function Figure({ title, subtitle, note, build, deps, height, actions, filename = 'chart' }) {
  const inst = useRef(null);
  const save = () => {
    if (!inst.current) return;
    const url = inst.current.getDataURL({ pixelRatio: 2, backgroundColor: palette().bg });
    const a = Object.assign(document.createElement('a'), { href: url, download: `${filename}.png` });
    a.click();
  };
  return html`<figure class="flex w-full min-w-0 flex-col gap-4 rounded-xl border border-border-primary bg-surface-brand-secondary p-4 sm:p-5 md:p-8">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0"><h3 class="font-mistral text-h6 text-text-primary">${title}</h3>${subtitle && html`<p class="text-body-small text-text-muted">${subtitle}</p>`}</div>
      <div class="flex shrink-0 items-center gap-2">${actions}
        <button type="button" class="btn btn-outline btn-small size-8 !p-0" aria-label=${`Download ${title} as PNG`} title="Download PNG" onClick=${save}><${Icon} name="download" size=${14} /></button></div></div>
    <${Chart} build=${build} deps=${deps} height=${height || (isNarrow() ? 260 : 340)} name=${title} onInstance=${(c) => { inst.current = c; }} />
    ${note && html`<p class="border-t border-border-primary pt-3 text-body-small text-text-muted">${note}</p>`}
  </figure>`;
}

// ---------------------------------------------------------------- chart option builders
const nulls = (n) => Array(n).fill(null);
const levelIdx = (levels, v) => levels.findIndex((l) => Math.abs(l - v) < 1e-9);

/** Fan chart: history, actuals, median and nested central intervals of the focus model; other models as lines. */
export function fanOption(p, d, { variate = 0, focus, shown, bands, targetName = '', covariate }) {
  const nh = d.xHistory.length, nf = d.xFuture.length, x = [...d.xHistory.map(String), ...d.xFuture.map(String)];
  const hist = [...d.history[variate], ...nulls(nf)];
  const actual = [...nulls(nh), ...d.actual[variate]];
  // connect the actual line to the last history point
  if (d.actual[variate].some((v) => v !== null)) actual[nh - 1] = d.history[variate][nh - 1];
  const series = [
    { name: 'History', type: 'line', data: hist, showSymbol: false, lineStyle: { width: 1.5, color: p.muted }, itemStyle: { color: p.muted }, z: 3 },
    { name: 'Actual', type: 'line', data: actual, showSymbol: false, lineStyle: { width: 1.8, color: p.text }, itemStyle: { color: p.text }, z: 6 },
  ];
  const colors = modelColors(Object.keys(d.models), focus, p);
  const alphas = { 0.98: 0.1, 0.9: 0.16, 0.8: 0.24, 0.5: 0.36 };
  Object.entries(d.models).forEach(([id, m]) => {
    if (!shown.includes(id)) return;
    const q = m.quantiles[variate]; // (H, Q)
    const col = (i) => q.map((row) => row[i]);
    const med = col(levelIdx(m.levels, 0.5));
    if (id === focus) {
      [0.98, 0.9, 0.8, 0.5].filter((c) => bands.includes(c)).forEach((c) => {
        const lo = levelIdx(m.levels, (1 - c) / 2), hi = levelIdx(m.levels, (1 + c) / 2);
        if (lo < 0 || hi < 0) return;
        const a = col(lo), b = col(hi), key = `band${c}`;
        series.push({ name: `${m.name} ${Math.round(c * 100)}%`, type: 'line', data: [...nulls(nh), ...a], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true, z: 1 });
        series.push({ name: `${m.name} ${Math.round(c * 100)}% band`, type: 'line', data: [...nulls(nh), ...b.map((v, i) => v - a[i])], stack: key, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: p.brand, opacity: alphas[c] }, silent: true, tooltip: { show: false }, z: 1 });
      });
    }
    series.push({ name: m.name, type: 'line', data: [...nulls(nh), ...med], showSymbol: false, lineStyle: { width: id === focus ? 2.4 : 1.6, color: colors[id], type: id === focus ? 'solid' : 'dashed' }, itemStyle: { color: colors[id] }, z: id === focus ? 5 : 4 });
  });
  series[0].markLine = { silent: true, symbol: 'none', label: { show: true, formatter: 'forecast start', color: p.muted, position: 'insideEndTop', fontSize: 11 }, lineStyle: { color: p.axis, type: 'dashed' }, data: [{ xAxis: nh - 1 }] };
  const from = Math.max(0, nh - Math.max(4 * nf, 96)); // open on the forecast with recent history; zoom out for more
  return {
    ...baseOption(p),
    grid: { left: isNarrow() ? 44 : 56, right: isNarrow() ? 12 : 24, top: 24, bottom: isNarrow() ? 50 : 76 },
    legend: { show: false },
    xAxis: { type: 'category', data: x, boundaryGap: false, ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.muted, fontSize: 11, hideOverlap: true, formatter: (v) => String(v).replace('T', ' ').slice(0, 16) } },
    yAxis: { type: 'value', scale: true, ...axisStyle(p), name: targetName, nameTextStyle: { color: p.muted, align: 'left' } },
    dataZoom: [{ type: 'inside', filterMode: 'none', startValue: from, endValue: x.length - 1 }, ...(isNarrow() ? [] : [{ type: 'slider', height: 18, bottom: 8, borderColor: p.axis, textStyle: { color: p.muted }, filterMode: 'none', handleSize: 14, startValue: from, endValue: x.length - 1 }])],
    tooltip: { ...baseOption(p).tooltip, valueFormatter: (v) => (v === null || v === undefined ? '' : Number(v).toPrecision(5)) },
    series,
  };
}

/** Horizontal bars from zero, focus model orange, others neutral, value labels (like the announcement charts). */
export function barOption(p, rows, { focus, lowerIsBetter = true, unit = '', digits = 3, refLine }) {
  const sorted = [...rows].sort((a, b) => (lowerIsBetter ? a.value - b.value : b.value - a.value));
  const colors = modelColors(sorted.map((r) => r.id), focus, p);
  return {
    ...baseOption(p), tooltip: { show: false },
    grid: { left: isNarrow() ? 96 : 130, right: 64, top: 8, bottom: 24 },
    xAxis: { type: 'value', min: 0, ...axisStyle(p) },
    yAxis: { type: 'category', inverse: true, data: sorted.map((r) => r.label), ...axisStyle(p, { splitLine: { show: false } }), axisLabel: { color: p.text, fontSize: 12, width: isNarrow() ? 90 : 120, overflow: 'truncate' } },
    series: [{ type: 'bar', data: sorted.map((r) => ({ value: r.value, itemStyle: { color: colors[r.id], borderRadius: [0, 2, 2, 0] } })), barMaxWidth: 22,
      label: { show: true, position: 'right', color: p.text, fontFamily: p.font, fontWeight: 600, formatter: (o) => `${Number(o.value).toFixed(digits)}${unit}` },
      markLine: refLine === undefined ? undefined : { silent: true, symbol: 'none', lineStyle: { color: p.axis, type: 'dashed' }, label: { color: p.muted, formatter: String(refLine) }, data: [{ xAxis: refLine }] } }],
  };
}

export function linesOption(p, { x, series, xName, yName, yMin, yLog, xType = 'category', focus, legend = true, smooth = false, diagonal }) {
  const colors = modelColors(series.map((s) => s.id), focus, p);
  // rows the legend wraps onto at this width (about 7 px a letter plus the key and the gap)
  const legendRows = legend ? Math.max(1, Math.ceil(series.reduce((n, s) => n + String(s.label).length * 7 + 36, 0) / Math.max(200, (p.width || 600) - 24))) : 0;
  return {
    ...baseOption(p), grid: { left: isNarrow() ? 48 : 60, right: isNarrow() ? 12 : 24, top: legendRows * 22 + (legend ? (yName ? 32 : 14) : (yName ? 34 : 24)), bottom: xName ? 52 : 36 },
    legend: legend ? { top: 0, data: series.map((s) => s.label), textStyle: { color: p.text, fontSize: 12 }, icon: 'roundRect', itemWidth: 14, itemHeight: 4 } : { show: false },
    xAxis: { type: xType, data: xType === 'category' ? x : undefined, name: xName, nameLocation: 'middle', nameGap: 30, nameTextStyle: { color: p.muted }, boundaryGap: xType === 'category' ? false : undefined, ...axisStyle(p, { splitLine: { show: false } }), ...(xType === 'log' ? { logBase: 2 } : {}) },
    yAxis: { type: yLog ? 'log' : 'value', name: yName, nameTextStyle: { color: p.muted, align: 'left' }, min: yMin, ...axisStyle(p) },
    series: [...series.map((s) => ({ name: s.label, type: 'line', data: s.data, smooth, showSymbol: s.data.length < 60, symbolSize: 6, lineStyle: { width: s.id === focus ? 2.4 : 1.6, color: s.color || colors[s.id], type: s.dashed ? 'dashed' : 'solid' }, itemStyle: { color: s.color || colors[s.id] } })),
      ...(diagonal ? [{ name: 'ideal', type: 'line', data: diagonal, showSymbol: false, lineStyle: { color: p.axis, type: 'dashed', width: 1 }, tooltip: { show: false } }] : [])],
  };
}

export function quantileStats(values) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return [0, 0, 0, 0, 0];
  const q = (f) => { const i = (v.length - 1) * f, lo = Math.floor(i), hi = Math.ceil(i); return v[lo] + (v[hi] - v[lo]) * (i - lo); };
  const iqr = q(0.75) - q(0.25);
  return [Math.max(v[0], q(0.25) - 1.5 * iqr), q(0.25), q(0.5), q(0.75), Math.min(v[v.length - 1], q(0.75) + 1.5 * iqr)];
}
