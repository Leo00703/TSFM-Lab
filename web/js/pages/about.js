import { html } from '/vendor/htm-preact.js';
import { Card, PageHeader, Section, Notice, KV } from '../ui.js';

const GLOSSARY = [
  ['Quantile / interval', 'The model outputs 21 quantiles (1%, 5%, 10% … 90%, 95%, 99%). An “80% interval” is the range between the 10% and 90% quantiles.'],
  ['MASE', 'Mean absolute error of the median forecast divided by the mean absolute (seasonal) difference of the history. 1 = as good as repeating the last season in the past.'],
  ['SQL', 'Scaled quantile loss: 2 × the average pinball loss over all horizon steps and quantile levels, divided by the same scale as MASE. A discretised, scaled CRPS; it rewards sharp and calibrated intervals.'],
  ['WQL', 'Weighted quantile loss: summed pinball loss over the horizon divided by the sum of absolute actuals, averaged over quantile levels. Undefined when actuals sum to zero.'],
  ['Coverage', 'Share of actual values inside a central interval. A well-calibrated 80% interval contains about 80%. Under-coverage means over-confident intervals.'],
  ['Bias', 'Mean of (median forecast − actual) divided by the history scale. Positive = the model forecasts too high.'],
  ['Skill', '1 − geometric mean of per-item (model MASE ÷ baseline MASE). +30% means a typical 30% lower error than the baseline; negative means worse.'],
  ['Cold start / warm latency', 'Cold = first call after loading (CUDA initialisation, kernel compilation). Warm = median of repeated calls after a warm-up. Both are reported.'],
  ['Backtest window', 'A forecast origin: the model sees only the data before it and is scored on the next “horizon” steps. Windows are counted back from the end of each series.'],
];

export function About() {
  return html`<div><${PageHeader} eyebrow="About" title="Licenses and definitions">This tool is independent and runs entirely on your machine.<//>
    <${Section} title="Not affiliated">
      <${Card}><p class="max-w-3xl">TSFM-Lab is an independent project. It is not affiliated with, endorsed by or sponsored by Mistral AI. “Timestral” and “Mistral” are Mistral AI's names; the visual style follows the public design of mistral.ai, and the logo and typeface, if shown, are loaded from this machine for private use and are never redistributed.</p><//><//>
    <${Section} title="Model weights license">
      <${Notice} kind="warning" title="Timestral weights: “Modified MIT” with a revenue limit">You are not authorized to exercise any rights under this license if the global consolidated <b>monthly</b> revenue of your company (or your employer) exceeds US$20 million for the preceding month. The limit applies to the model and to any derivatives, modifications or combined works, and the text has no exception for research or testing. Companies above the limit can ask Mistral AI for a commercial license (at its discretion) or use its hosted services. Read the full text: <a class="underline" href="https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE" target="_blank" rel="noopener">Hugging Face · LICENSE</a>. This is not legal advice.<//>
      <p class="mt-4 max-w-3xl text-body-small text-text-muted">The <span class="font-mono">timestral</span> Python package is Apache-2.0. Weights are downloaded to your project folder at your request and never uploaded anywhere.</p><//>
    <${Section} title="Metrics, in plain language"><${Card}><dl class="grid gap-x-8 gap-y-5 md:grid-cols-[12rem_1fr]">${GLOSSARY.map(([k, v]) => html`<dt class="font-mistral text-h6">${k}</dt><dd class="text-text-muted">${v}</dd>`)}</dl><//><//>
    <${Section} title="How to read results honestly"><${Card}><ul class="list-disc space-y-2 pl-5 text-text-muted"><li>Compare with a baseline on the same windows; a number alone means little.</li><li>One window is anecdote; look at the spread across many series and windows.</li><li>Foundation models are pre-trained on large, partly undisclosed data: public datasets may overlap with it. Prefer your own recent data.</li><li>Check calibration before using intervals for decisions.</li><li>Vendor benchmark figures are shown only as labelled references; they were not measured here.</li></ul><//><//>
    <${Section} title="Third-party components"><${Card} tight><${KV} items=${[['Apache ECharts 6.1', 'Apache-2.0 · charts'], ['htm 3.1 + Preact 10', 'Apache-2.0 / MIT · rendering'], ['Tailwind CSS 4.3', 'MIT · styling'], ['Inter', 'SIL Open Font License · body text'], ['Space Mono', 'SIL Open Font License · labels'], ['FastAPI · PyTorch · pandas · numpy', 'permissive licenses · backend']]} /><//><//>
  </div>`;
}
