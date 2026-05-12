// Pareto chart + Plotly lazy-loader.
//
// PR 6 introduced the lazy CDN loader and a placeholder render with a
// simple {x, y} signature. PR 7 extends `renderParetoChart` to consume
// the NSGA-II candidate shape produced by `app/api/routers/design.py`
// (`{candidates: [{property, cost_eur_per_t, in_spec, is_ood, rank, ...}]}`)
// while keeping the placeholder mode for PR 6 callers — the placeholder
// path is kept first so existing wire-up doesn't regress.

const PLOTLY_CDN_URL = 'https://cdn.plot.ly/plotly-2.35.2.min.js';

let _plotlyPromise = null;

/** Load Plotly from CDN (idempotent — promise is cached at module scope).
 *
 * Returns the global `Plotly` object. Reuses an existing tag if the user
 * already loaded Plotly via index.html (defensive — no harm in double
 * resolution). Rejects if the CDN load fails so callers can fall back
 * to a plain-table view.
 */
export function loadPlotly() {
  // If something else already loaded Plotly into the global, reuse it
  // immediately — no need to inject another <script> tag.
  if (typeof window !== 'undefined' && window.Plotly) {
    return Promise.resolve(window.Plotly);
  }
  if (_plotlyPromise) return _plotlyPromise;

  _plotlyPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = PLOTLY_CDN_URL;
    script.async = true;
    script.onload = () => {
      if (window.Plotly) {
        resolve(window.Plotly);
      } else {
        reject(new Error('Plotly script loaded but window.Plotly is undefined'));
      }
    };
    script.onerror = () => {
      // Reset cache so a later retry can attempt the CDN again.
      _plotlyPromise = null;
      reject(new Error(`Не удалось загрузить Plotly с ${PLOTLY_CDN_URL}`));
    };
    document.head.appendChild(script);
  });

  return _plotlyPromise;
}

/** Default theme tokens — pulled to match css/app.css palette so the chart
 *  doesn't fight the rest of the dashboard. PR 7 extends this with proper
 *  marker styling per candidate state.
 */
const DEFAULT_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {
    family: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    size: 11,
    color: '#cdd1d3',
  },
  margin: { l: 60, r: 16, t: 28, b: 48 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.06)',
    zerolinecolor: 'rgba(255,255,255,0.10)',
    tickfont: { family: "'JetBrains Mono', monospace", size: 10 },
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.06)',
    zerolinecolor: 'rgba(255,255,255,0.10)',
    tickfont: { family: "'JetBrains Mono', monospace", size: 10 },
  },
  legend: {
    orientation: 'h',
    x: 0,
    y: -0.18,
    font: { size: 10.5 },
    bgcolor: 'rgba(0,0,0,0)',
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

// Marker palette — kept in sync with overrides.css colour tokens. We
// don't reach into CSS at runtime because Plotly takes literal strings.
const COLOR_IN_SPEC = 'rgba(108, 209, 132, 0.85)';
const COLOR_IN_SPEC_LINE = 'rgba(108, 209, 132, 1)';
const COLOR_OUT_OF_SPEC = 'rgba(232, 116, 116, 0.78)';
const COLOR_OUT_OF_SPEC_LINE = 'rgba(232, 116, 116, 1)';
const COLOR_OOD = 'rgba(247, 168, 92, 0.85)';
const COLOR_OOD_LINE = 'rgba(247, 168, 92, 1)';
const COLOR_HIGHLIGHT = 'rgba(124, 178, 255, 1)';

// ──────────────────────────────────────────────────────────────────────
// Candidate-shape detection
// ──────────────────────────────────────────────────────────────────────

/** True if ``data`` looks like the NSGA-II candidate array (PR 7 shape).
 *
 * We pick the first row and check for ``property``/``cost_eur_per_t``
 * — both required for the X/Y axes. Defensive: if even one row has the
 * shape we proceed; the other rows fall back to 0 if a key is missing.
 */
function isNsgaCandidates(data) {
  if (!Array.isArray(data) || data.length === 0) return false;
  const first = data[0];
  return (
    first != null &&
    typeof first === 'object' &&
    ('property' in first || 'predicted_mean' in first) &&
    'cost_eur_per_t' in first
  );
}

function fmtNumber(value, decimals = 1) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

function topComposition(comp, k = 3) {
  if (!comp || typeof comp !== 'object') return '';
  // Show k highest-magnitude alloying elements (skip zeros, base) for
  // hover compactness. NSGA-II returns *_pct values so 4 decimals is
  // appropriate.
  const entries = Object.entries(comp)
    .filter(([key, v]) => typeof v === 'number' && Math.abs(v) > 1e-4 && key.endsWith('_pct'))
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
  return entries.map(([key, v]) => `${key}=${v.toFixed(4)}`).join(' · ');
}

function buildHoverText(c, currency) {
  const lines = [];
  lines.push(`<b>Кандидат #${c.rank ?? c.idx ?? '?'}</b>`);
  lines.push(`σт = ${fmtNumber(c.predicted_mean, 1)} ± ${fmtNumber(c.ci_half_width, 1)} МПа`);
  lines.push(`Стоимость = ${fmtNumber(c.cost_eur_per_t, 0)} ${currency || ''}/т`);
  if (c.delta_property_vs_baseline != null) {
    const sign = c.delta_property_vs_baseline >= 0 ? '+' : '';
    lines.push(`Δσт vs baseline = ${sign}${fmtNumber(c.delta_property_vs_baseline, 1)} МПа`);
  }
  if (c.delta_cost_vs_baseline != null) {
    const sign = c.delta_cost_vs_baseline >= 0 ? '+' : '';
    lines.push(`Δcost vs baseline = ${sign}${fmtNumber(c.delta_cost_vs_baseline, 0)} ${currency || ''}/т`);
  }
  const composition = topComposition(c.composition);
  if (composition) lines.push(`<span style="color:#9aa0a6">${composition}</span>`);
  if (c.is_ood) lines.push('<span style="color:#f7a85c">⚠ OOD</span>');
  return lines.join('<br>');
}

/** Group candidates into traces by status (in-spec / out-of-spec / OOD). */
function splitCandidates(candidates) {
  const inSpec = [];
  const outOfSpec = [];
  const ood = [];
  for (const c of candidates) {
    if (c.is_ood) {
      ood.push(c);
    } else if (c.in_spec) {
      inSpec.push(c);
    } else {
      outOfSpec.push(c);
    }
  }
  return { inSpec, outOfSpec, ood };
}

function buildTrace(group, name, color, lineColor, currency, selectedIdx) {
  const sizes = group.map((c) => (c.idx === selectedIdx ? 14 : 8));
  const lineWidths = group.map((c) => (c.idx === selectedIdx ? 2.5 : 1));
  const lineColors = group.map((c) =>
    c.idx === selectedIdx ? COLOR_HIGHLIGHT : lineColor,
  );
  return {
    name,
    x: group.map((c) => Number(c.cost_eur_per_t || 0)),
    y: group.map((c) => Number(c.property ?? c.predicted_mean ?? 0)),
    customdata: group.map((c) => c.idx),
    text: group.map((c) => buildHoverText(c, currency)),
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: sizes,
      color,
      line: { color: lineColors, width: lineWidths },
    },
    hovertemplate: '%{text}<extra></extra>',
  };
}

/** Render either the PR 6 placeholder (legacy `{x, y}` array) or the PR 7
 *  full NSGA-II layout. We auto-detect based on input shape so existing
 *  callers (none today — the design.js view is the first consumer of the
 *  new shape) continue to work.
 *
 * @param {HTMLElement} container  target node
 * @param {Array} data  either `[{x, y, label}]` (legacy) or NSGA-II
 *                      candidate dicts `[{property, cost_eur_per_t, ...}]`
 * @param {object} [opts]
 * @param {string} [opts.title]
 * @param {string} [opts.xLabel]
 * @param {string} [opts.yLabel]
 * @param {string} [opts.currency='EUR']
 * @param {number|null} [opts.selectedIdx]  candidate idx to outline
 * @param {(idx:number, candidate:object)=>void} [opts.onSelect]
 *                                                callback when a marker is
 *                                                clicked
 */
export async function renderParetoChart(container, data, opts = {}) {
  const Plotly = await loadPlotly();

  // --- Legacy path (PR 6 placeholder) --------------------------------
  if (!isNsgaCandidates(data)) {
    const points = Array.isArray(data) ? data : [];
    const trace = {
      x: points.map((d) => (typeof d.x === 'number' ? d.x : 0)),
      y: points.map((d) => (typeof d.y === 'number' ? d.y : 0)),
      text: points.map((d) => d.label || ''),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 7,
        color: COLOR_OOD,
        line: { color: COLOR_OOD_LINE, width: 1 },
      },
      hovertemplate: '%{text}<br>x=%{x}<br>y=%{y}<extra></extra>',
    };
    const layout = {
      ...DEFAULT_LAYOUT,
      title: opts.title
        ? { text: opts.title, font: { size: 12, color: '#cdd1d3' } }
        : undefined,
      xaxis: { ...DEFAULT_LAYOUT.xaxis, title: { text: opts.xLabel || '' } },
      yaxis: { ...DEFAULT_LAYOUT.yaxis, title: { text: opts.yLabel || '' } },
    };
    await Plotly.newPlot(container, [trace], layout, DEFAULT_CONFIG);
    return;
  }

  // --- NSGA-II path (PR 7) -------------------------------------------
  const candidates = data;
  const currency = opts.currency || 'EUR';
  const selectedIdx = opts.selectedIdx ?? null;
  const { inSpec, outOfSpec, ood } = splitCandidates(candidates);

  const traces = [];
  if (inSpec.length > 0) {
    traces.push(
      buildTrace(inSpec, 'In-spec', COLOR_IN_SPEC, COLOR_IN_SPEC_LINE, currency, selectedIdx),
    );
  }
  if (outOfSpec.length > 0) {
    traces.push(
      buildTrace(outOfSpec, 'Out-of-spec', COLOR_OUT_OF_SPEC, COLOR_OUT_OF_SPEC_LINE, currency, selectedIdx),
    );
  }
  if (ood.length > 0) {
    traces.push(
      buildTrace(ood, 'OOD', COLOR_OOD, COLOR_OOD_LINE, currency, selectedIdx),
    );
  }

  const layout = {
    ...DEFAULT_LAYOUT,
    title: opts.title
      ? { text: opts.title, font: { size: 12, color: '#cdd1d3' } }
      : undefined,
    xaxis: {
      ...DEFAULT_LAYOUT.xaxis,
      title: { text: opts.xLabel || `Стоимость, ${currency}/т`, font: { size: 11 } },
    },
    yaxis: {
      ...DEFAULT_LAYOUT.yaxis,
      title: { text: opts.yLabel || 'σт, МПа', font: { size: 11 } },
    },
    showlegend: traces.length > 1,
  };

  await Plotly.newPlot(container, traces, layout, DEFAULT_CONFIG);

  // Wire click handler — Plotly passes the clicked point's customdata
  // (the candidate idx) so the caller can look up the full record in
  // its own state.
  if (typeof opts.onSelect === 'function') {
    container.removeAllListeners?.('plotly_click');
    container.on('plotly_click', (ev) => {
      try {
        const point = ev?.points?.[0];
        if (!point) return;
        const idx = point.customdata;
        const cand = candidates.find((c) => c.idx === idx) || null;
        opts.onSelect(idx, cand);
      } catch (e) {
        console.error('[pareto] click handler failed', e);
      }
    });
  }
}
