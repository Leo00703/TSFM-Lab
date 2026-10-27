// Custom dropdown in the app's own design language (replaces the browser/OS native <select> popup).
// Accessible: combobox button + listbox, arrow/Home/End/Enter/Escape keys and type-ahead.
import { html, useState, useEffect, useRef } from '/vendor/htm-preact.js';
import { Icon } from './icons.js';

let uid = 0;

export function Select({ value, onChange, options, compact, placeholder = 'Select…', class: cls = '', disabled, ...rest }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [pos, setPos] = useState(null);
  const btn = useRef(null), list = useRef(null), typed = useRef({ s: '', t: 0 }), id = useRef('sel' + ++uid);
  const cur = options.find((o) => o.value === value);
  const enabled = options.map((o, i) => (o.disabled ? -1 : i)).filter((i) => i >= 0);

  const show = () => {
    if (disabled) return;
    const r = btn.current.getBoundingClientRect(), below = innerHeight - r.bottom - 12, above = r.top - 12;
    const up = below < 220 && above > below, room = up ? above : below;
    const width = Math.min(Math.max(r.width, 200), innerWidth - 16);
    setPos({ left: Math.max(8, Math.min(r.left, innerWidth - width - 8)), width, maxH: Math.max(120, Math.min(340, room)), ...(up ? { bottom: innerHeight - r.top + 4 } : { top: r.bottom + 4 }) });
    setActive(Math.max(0, options.findIndex((o) => o.value === value)));
    setOpen(true);
  };
  const close = (focus) => { setOpen(false); if (focus && btn.current) btn.current.focus(); };
  const pick = (i) => { const o = options[i]; if (!o || o.disabled) return; close(true); if (o.value !== value) onChange(o.value); };
  const move = (dir) => { const pos0 = enabled.indexOf(active); const next = enabled[Math.max(0, Math.min(enabled.length - 1, (pos0 < 0 ? 0 : pos0 + dir)))]; if (next !== undefined) setActive(next); };

  useEffect(() => {
    if (!open) return;
    const away = (e) => { if (!btn.current.contains(e.target) && !(list.current && list.current.contains(e.target))) setOpen(false); };
    const scroll = (e) => { if (!(list.current && list.current.contains(e.target))) setOpen(false); };
    document.addEventListener('pointerdown', away);
    addEventListener('scroll', scroll, true);
    addEventListener('resize', close);
    return () => { document.removeEventListener('pointerdown', away); removeEventListener('scroll', scroll, true); removeEventListener('resize', close); };
  }, [open]);
  useEffect(() => { if (open && list.current) { const el = list.current.querySelector(`[data-i="${active}"]`); if (el) el.scrollIntoView({ block: 'nearest' }); } }, [open, active]);

  const onKey = (e) => {
    if (!open) { if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) { e.preventDefault(); show(); } return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
    else if (e.key === 'Home') { e.preventDefault(); setActive(enabled[0]); }
    else if (e.key === 'End') { e.preventDefault(); setActive(enabled[enabled.length - 1]); }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(active); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); }
    else if (e.key === 'Tab') close(false);
    else if (e.key.length === 1) {
      const now = Date.now(); typed.current.s = now - typed.current.t > 700 ? e.key : typed.current.s + e.key; typed.current.t = now;
      const hit = enabled.find((i) => String(options[i].text || options[i].label).toLowerCase().startsWith(typed.current.s.toLowerCase()));
      if (hit !== undefined) setActive(hit);
    }
  };

  let lastGroup = null;
  return html`<span class=${'relative block min-w-0 ' + cls}>
    <button ref=${btn} type="button" role="combobox" aria-haspopup="listbox" aria-expanded=${open} aria-controls=${id.current} disabled=${disabled}
      class=${'input flex items-center justify-between gap-2 text-left cursor-pointer transition-colors hover:bg-action-ghost-1 disabled:opacity-50 ' + (compact ? '!h-8 !px-2.5 !text-body-small ' : '') + (open ? 'border-border-secondary' : '')}
      onClick=${() => (open ? close(false) : show())} onKeyDown=${onKey} ...${rest}>
      <span class=${'min-w-0 flex-1 truncate ' + (cur ? '' : 'text-text-muted')}>${cur ? cur.label : placeholder}</span>
      <${Icon} name="chevron" size=${compact ? 10 : 12} class=${'shrink-0 text-text-muted transition-transform ' + (open ? 'rotate-180' : '')} /></button>
    ${open && pos && html`<div ref=${list} id=${id.current} role="listbox" tabindex="-1" style=${`position:fixed;z-index:60;left:${pos.left}px;width:${pos.width}px;max-height:${pos.maxH}px;` + (pos.top !== undefined ? `top:${pos.top}px` : `bottom:${pos.bottom}px`)}
      class="overflow-y-auto rounded-md border border-border-primary bg-surface-brand-secondary p-1 shadow-xl"
      onMouseDown=${(e) => e.preventDefault()} onClick=${(e) => { e.preventDefault(); e.stopPropagation(); }}>
      ${options.map((o, i) => { const header = o.group && o.group !== lastGroup ? html`<div class="eyebrow px-2.5 pb-1 pt-2.5 first:pt-1.5" role="presentation">${o.group}</div>` : null; lastGroup = o.group || lastGroup;
        return html`${header}<div role="option" data-i=${i} aria-selected=${o.value === value} aria-disabled=${o.disabled || undefined}
          class=${'flex min-h-9 cursor-pointer items-center gap-2 rounded-sm px-2.5 py-1.5 text-body-small ' + (o.disabled ? 'cursor-not-allowed text-text-muted opacity-50 ' : 'text-text-primary ') + (i === active && !o.disabled ? 'bg-action-ghost-2 ' : '')}
          onMouseMove=${() => !o.disabled && active !== i && setActive(i)} onClick=${(e) => { e.preventDefault(); e.stopPropagation(); pick(i); }}>
          <span class="grid size-4 shrink-0 place-items-center text-surface-brand">${o.value === value && html`<${Icon} name="check" size=${12} />`}</span>
          <span class="min-w-0 flex-1 truncate">${o.label}</span></div>`; })}</div>`}
  </span>`;
}
