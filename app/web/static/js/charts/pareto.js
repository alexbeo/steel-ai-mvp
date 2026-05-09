// Pareto chart + Plotly lazy-loader.
//
// PR 6 of the Streamlit→FastAPI migration. We don't bundle Plotly — at
// 700 KB minified-basic it's a heavy dependency that isn't needed on
// every tab. Instead we lazy-load from CDN the first time a chart is
// requested and cache the promise so subsequent calls reuse the same
// global `window.Plotly`.
//
// PR 7 will replace `renderParetoChart` with the full NSGA-II layout
// (marker shapes for in-spec / OOD / rejected / Top-5 ring, hover with
// composition, axis title with target name + cost units). PR 6 ships
// the lazy-loader pattern + a placeholder render so the import path is
// stable and downstream PRs only touch chart-internal code.

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
 *  doesn't fight the rest of the dashboard. PR 7 will extend this with
 *  proper marker styling per candidate state.
 */
const DEFAULT_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {
    family: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    size: 11,
    color: '#cdd1d3',
  },
  margin: { l: 56, r: 16, t: 28, b: 44 },
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
};

const DEFAULT_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

/** Render a Pareto scatter into `container`.
 *
 * PR 6 placeholder: takes an array of `{x, y, label?}` objects and draws
 * a single scatter trace. PR 7 will overload to accept the NSGA-II
 * candidate shape `{property, cost, in_spec, ood, rank}` and split into
 * multiple traces with shape variants. Keeping the signature
 * forward-compatible (data + opts) so PR 7 callers don't need to migrate.
 *
 * @param {HTMLElement} container  target node, will be wired to Plotly
 * @param {Array<{x:number,y:number,label?:string}>} data
 * @param {object} [opts]
 * @param {string} [opts.title]
 * @param {string} [opts.xLabel]
 * @param {string} [opts.yLabel]
 */
export async function renderParetoChart(container, data, opts = {}) {
  const Plotly = await loadPlotly();

  const points = Array.isArray(data) ? data : [];
  const trace = {
    x: points.map((d) => (typeof d.x === 'number' ? d.x : 0)),
    y: points.map((d) => (typeof d.y === 'number' ? d.y : 0)),
    text: points.map((d) => d.label || ''),
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 7,
      color: 'rgba(247, 168, 92, 0.85)',
      line: { color: 'rgba(247, 168, 92, 0.95)', width: 1 },
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
}
