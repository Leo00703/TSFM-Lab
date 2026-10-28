// Theme before first paint: stored choice (system | light | dark); DARK by default, as on mistral.ai.
// A separate file, not an inline script, so the page can run under a Content-Security-Policy that allows no inline script.
(function () {
  var t = 'dark';
  try { t = localStorage.getItem('tsfm-theme') || 'dark'; } catch (e) {}
  var dark = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
  var m = document.querySelector('meta[name=theme-color]');
  if (m) m.content = dark ? '#101013' : '#fbfbf8';
})();
