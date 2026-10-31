// Theme before first paint: stored choice (system | light | dark); SYSTEM by default (the operating system's light or dark setting).
// A separate file, not an inline script, so the page can run under a Content-Security-Policy that allows no inline script.
(function () {
  var t = 'system';
  try { t = localStorage.getItem('tsfm-theme') || 'system'; } catch (e) {}
  var dark = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', dark);
  var m = document.querySelector('meta[name=theme-color]');
  if (m) m.content = dark ? '#101013' : '#fbfbf8';
})();
