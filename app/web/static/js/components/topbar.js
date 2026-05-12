// Topbar KPI strip — refresh from /api/system/kpi.
//
// PR 13 of the Streamlit→FastAPI migration. Lives next to the brand block in
// index.html and exposes 5 cells via [data-kpi] attributes:
//   active_model · model_r2 · coverage_90_ci · last_pareto_size · phd_reviews_week
//
// Auto-refreshes every 60 s. We keep this lightweight (single fetch, in-place
// text mutation) so it never competes with view-level work for main-thread
// time. Errors are silently logged — the placeholder dash stays put if the
// API is unreachable, which mirrors the "graceful degradation" rule from
// PR 1's app.js (router failures don't break the shell).

import { apiFetch, ApiError } from '../api.js';
import { qsa } from '../utils/dom.js';

const REFRESH_INTERVAL_MS = 60_000;

// Thresholds match the User decision #2 colour hints. Keep loose:
//   - Coverage 90% CI: 85..95% is the M02 calibration target band.
//     Outside that band → neutral (no warn class) because we do not
//     want to scream "danger" — calibration drift is informational.
//   - Model R²: ≥ 0.7 is "useful for design" per ML conventions in
//     CLAUDE.md. Below that → neutral (still informative, just not
//     greenlit).
const COVERAGE_OK_MIN = 85;
const COVERAGE_OK_MAX = 95;
const R2_OK_MIN = 0.7;

let refreshTimer = null;

/** Truncate a model version for the 12-char-wide topbar cell. */
function shortVersion(version) {
  if (!version) return '';
  // Versions look like "hsla_yield_strength_xgb_20260510_031107". Show the
  // last two underscore-segments — that's the timestamp, which is the only
  // part that distinguishes consecutive trainings of the same target.
  const segs = String(version).split('_');
  if (segs.length < 2) return String(version).slice(0, 12);
  return segs.slice(-2).join('_');
}

/** Format `active_model` payload → "<class name>/<short version>". */
function formatActiveModel(value) {
  if (!value || typeof value !== 'object') return '—';
  const className = value.class_name || value.class_id || '';
  const version = shortVersion(value.version || '');
  if (className && version) return `${className} · ${version}`;
  if (className) return className;
  if (version) return version;
  return '—';
}

/** Format Model R² with 3-digit precision; null → "—". */
function formatR2(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return value.toFixed(3);
}

/** Format coverage as integer percent; null → "—". */
function formatCoverage(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${value}%`;
}

/** Format integer count; null → "—" (we don't show 0 as "—" because 0 is a
 *  meaningful answer for "how many candidates did the last NSGA-II run
 *  produce?" — it tells the user the optimizer failed to find anything).
 */
function formatCount(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return String(Math.trunc(value));
}

/** Decide which CSS class (`ok` / `warn` / none) goes on the value span.
 *  Returns the class name or empty string. */
function colourClassFor(key, value) {
  if (value == null || (typeof value === 'number' && Number.isNaN(value))) {
    return '';
  }
  if (key === 'model_r2' && typeof value === 'number') {
    return value >= R2_OK_MIN ? 'ok' : '';
  }
  if (key === 'coverage_90_ci' && typeof value === 'number') {
    return value >= COVERAGE_OK_MIN && value <= COVERAGE_OK_MAX ? 'ok' : 'warn';
  }
  return '';
}

/** Update one cell in-place. The HTML uses `<span data-kpi="…">—</span>`;
 *  we replace its text and toggle the `ok`/`warn` class without touching the
 *  surrounding meta-cell layout. */
function applyCell(node, key, raw) {
  let text;
  switch (key) {
    case 'active_model':
      text = formatActiveModel(raw);
      break;
    case 'model_r2':
      text = formatR2(raw);
      break;
    case 'coverage_90_ci':
      text = formatCoverage(raw);
      break;
    case 'last_pareto_size':
    case 'phd_reviews_week':
      text = formatCount(raw);
      break;
    default:
      text = '—';
  }
  node.textContent = text;
  // Reset both colour classes, then re-apply at most one. Spans use
  // `meta-value` — the `ok`/`warn` modifiers are scoped to that
  // container in app.css (lines 112-113).
  node.classList.remove('ok', 'warn');
  const cls = colourClassFor(key, raw);
  if (cls) node.classList.add(cls);
}

/** Fetch /api/system/kpi once and write into all `[data-kpi]` cells. */
export async function refreshTopbarKpi() {
  let payload;
  try {
    payload = await apiFetch('/api/system/kpi');
  } catch (err) {
    if (err instanceof ApiError) {
      console.warn(`[topbar] /api/system/kpi failed: ${err.status} ${err.message}`);
    } else {
      console.warn('[topbar] /api/system/kpi failed:', err);
    }
    return; // Leave existing cell text untouched.
  }
  if (!payload || typeof payload !== 'object') return;

  for (const node of qsa('[data-kpi]')) {
    const key = node.dataset.kpi;
    if (!key) continue;
    applyCell(node, key, payload[key]);
  }
}

/** Start the 60 s refresh loop. Calls `refreshTopbarKpi` immediately, then
 *  schedules subsequent runs. Idempotent — calling twice resets the timer
 *  rather than stacking parallel intervals. */
export function startTopbarKpiAutoRefresh() {
  if (refreshTimer != null) clearInterval(refreshTimer);
  refreshTopbarKpi();
  refreshTimer = setInterval(refreshTopbarKpi, REFRESH_INTERVAL_MS);
}

/** Stop the loop — useful for tests / future tear-down hooks. */
export function stopTopbarKpiAutoRefresh() {
  if (refreshTimer != null) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}
