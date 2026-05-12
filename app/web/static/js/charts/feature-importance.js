// Feature importance horizontal bar chart for the Train view (PR 8).
//
// Reuses the lazy `loadPlotly` from charts/pareto.js so we don't ship
// two copies of the CDN-loader closure. Theme tokens match overrides.css
// (oklch palette), kept inline because Plotly only accepts literal
// colour strings — extracting them from CSS at runtime would be a
// tax for cosmetic alignment.

import { loadPlotly } from './pareto.js';

const DEFAULT_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {
    family: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    size: 11,
    color: '#cdd1d3',
  },
  margin: { l: 160, r: 24, t: 12, b: 36 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.06)',
    zerolinecolor: 'rgba(255,255,255,0.10)',
    tickfont: { family: "'JetBrains Mono', monospace", size: 10 },
    title: { text: 'Importance', font: { size: 11 } },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.04)',
    zerolinecolor: 'rgba(255,255,255,0.10)',
    tickfont: { family: "'JetBrains Mono', monospace", size: 10 },
    automargin: true,
  },
  hoverlabel: {
    bgcolor: 'rgba(15, 22, 24, 0.95)',
    bordercolor: 'rgba(255,255,255,0.12)',
    font: {
      family: "'JetBrains Mono', monospace",
      size: 11,
      color: '#cdd1d3',
    },
  },
};

const DEFAULT_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

// Plotly handles 20 features comfortably with automargin; we cap there
// to avoid scroll on fatigue_carbon_steel (24 features). Defensive cap,
// not an aesthetic.
const MAX_BARS = 20;

const COLOR_BAR = 'rgba(124, 178, 255, 0.85)';
const COLOR_BAR_LINE = 'rgba(124, 178, 255, 1)';

/** Render a horizontal feature-importance bar chart.
 *
 * @param {HTMLElement} container  target node; previous content is replaced
 * @param {Array<{feature: string, importance: number}>} items
 *        already-sorted by backend (desc), but we re-sort defensively
 *        so a future caller passing unordered data still renders right.
 * @param {object} [opts]
 * @param {number} [opts.maxBars=20]  hard cap on bars rendered
 * @param {string} [opts.title]
 * @returns {Promise<void>}
 */
export async function renderFeatureImportance(container, items, opts = {}) {
  const Plotly = await loadPlotly();
  const maxBars = Math.max(1, Number(opts.maxBars) || MAX_BARS);

  const data = Array.isArray(items) ? items.slice() : [];
  // Defensive sort desc by importance — backend already does this, but
  // a UI test that swaps in mock data shouldn't have to remember.
  data.sort((a, b) => Number(b.importance || 0) - Number(a.importance || 0));
  const top = data.slice(0, maxBars);

  if (top.length === 0) {
    container.replaceChildren();
    container.appendChild(
      Object.assign(document.createElement('div'), {
        className: 'feature-importance-empty',
        textContent: 'Нет данных feature_importance',
      }),
    );
    return;
  }

  // Plotly orders horizontal bars bottom-up by default — reverse the
  // input so the largest importance ends up on top.
  const reversed = top.slice().reverse();
  const features = reversed.map((d) => d.feature);
  const importance = reversed.map((d) => Number(d.importance || 0));

  const trace = {
    type: 'bar',
    orientation: 'h',
    x: importance,
    y: features,
    marker: {
      color: COLOR_BAR,
      line: { color: COLOR_BAR_LINE, width: 1 },
    },
    text: importance.map((v) => v.toFixed(4)),
    textposition: 'outside',
    textfont: { family: "'JetBrains Mono', monospace", size: 10, color: '#9aa0a6' },
    cliponaxis: false,
    hovertemplate: '<b>%{y}</b><br>importance = %{x:.4f}<extra></extra>',
  };

  const layout = {
    ...DEFAULT_LAYOUT,
    title: opts.title
      ? { text: opts.title, font: { size: 12, color: '#cdd1d3' } }
      : undefined,
    height: Math.max(220, top.length * 22 + 60),
  };

  await Plotly.newPlot(container, [trace], layout, DEFAULT_CONFIG);
}
