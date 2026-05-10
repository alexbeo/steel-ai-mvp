// Tab 08 — История решений проекта.
//
// Data flow:
//   GET /api/decisions/summary           → {summary_md}
//   GET /api/decisions?phase=&limit=&offset=  → {items, limit, offset, count}
//
// PR 2 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';
import { decisionLogRow } from '../components/decision-log-row.js';

// Phases mirror pattern_library.patterns.Phase. "deoxidation" is a valid
// phase enum value (DX01-DX03 patterns) so it appears in the dropdown
// alongside the standard pipeline phases.
const PHASE_OPTIONS = [
  { value: '', label: 'Все' },
  { value: 'data_acquisition', label: 'data_acquisition' },
  { value: 'preprocessing', label: 'preprocessing' },
  { value: 'feature_engineering', label: 'feature_engineering' },
  { value: 'training', label: 'training' },
  { value: 'inverse_design', label: 'inverse_design' },
  { value: 'validation', label: 'validation' },
  { value: 'reporting', label: 'reporting' },
  { value: 'deoxidation', label: 'deoxidation' },
  { value: 'meta', label: 'meta' },
];

const DEFAULT_LIMIT = 20;

/** Module-scoped state. Re-init() resets it; we only mount this view once
 *  per session in practice (router caches initialised views). */
const state = {
  phase: '',
  limit: DEFAULT_LIMIT,
  offset: 0,
  loading: false,
  total: 0, // running tally for "shown N" line
};

let elements = null; // populated by init()

/** Build the static skeleton — section-head, filter bar, summary block,
 *  list container, "load more" footer. Returned wrapper is appended to the
 *  view container. */
function buildSkeleton() {
  // Section header — mirrors .section-head pattern from app.css.
  const head = el(
    'div',
    { class: 'section-head' },
    el(
      'div',
      {},
      el(
        'div',
        { class: 'breadcrumb' },
        'Аудит',
        el('span', { class: 'sep' }, '/'),
        el('span', { class: 'here' }, 'История решений'),
      ),
      el('h1', { class: 'section-title' }, 'История решений проекта'),
      el(
        'p',
        { class: 'section-sub' },
        'Structured memory — все архитектурные решения с контекстом и reasoning. ' +
          'SQLite база decision_log/decisions.db. Persistent memory проекта.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  // Filter bar — phase select + limit + apply button.
  const phaseSelect = el(
    'select',
    {
      class: 'history-select mono',
      id: 'history-phase-select',
      onChange: (ev) => {
        state.phase = ev.target.value;
      },
    },
    ...PHASE_OPTIONS.map((opt) =>
      el('option', { value: opt.value }, opt.label),
    ),
  );

  const limitSelect = el(
    'select',
    {
      class: 'history-select mono',
      id: 'history-limit-select',
      onChange: (ev) => {
        const next = parseInt(ev.target.value, 10);
        state.limit = Number.isFinite(next) && next > 0 ? next : DEFAULT_LIMIT;
      },
    },
    ...[5, 10, 20, 50, 100].map((n) =>
      el(
        'option',
        n === DEFAULT_LIMIT ? { value: String(n), selected: '' } : { value: String(n) },
        String(n),
      ),
    ),
  );

  const applyBtn = el(
    'button',
    {
      class: 'btn primary',
      type: 'button',
      onClick: () => {
        // New filter or new limit — reset pagination.
        state.offset = 0;
        state.total = 0;
        load({ replace: true });
      },
    },
    'Применить',
  );

  const filterBar = el(
    'div',
    { class: 'history-filter-bar' },
    el(
      'label',
      { class: 'history-filter-cell', for: 'history-phase-select' },
      el('span', { class: 'history-filter-label' }, 'Фильтр по фазе'),
      phaseSelect,
    ),
    el(
      'label',
      { class: 'history-filter-cell', for: 'history-limit-select' },
      el('span', { class: 'history-filter-label' }, 'Записей на страницу'),
      limitSelect,
    ),
    el('div', { class: 'history-filter-spacer' }),
    applyBtn,
  );

  // Summary block — fed by GET /api/decisions/summary.
  const summaryBlock = el(
    'div',
    { class: 'history-summary' },
    el('div', { class: 'history-summary-label' }, 'Сводка проекта'),
    el(
      'pre',
      { class: 'history-summary-body mono', id: 'history-summary-body' },
      'Загрузка…',
    ),
  );

  // List + status + load-more footer.
  const listContainer = el('div', {
    class: 'log-list',
    id: 'history-list',
  });
  const statusLine = el('div', { class: 'history-status' }, '');
  const loadMoreBtn = el(
    'button',
    {
      class: 'btn',
      type: 'button',
      hidden: '',
      onClick: () => {
        state.offset += state.limit;
        load({ replace: false });
      },
    },
    'Загрузить ещё',
  );
  const footer = el(
    'div',
    { class: 'history-footer' },
    statusLine,
    loadMoreBtn,
  );

  // Error banner (hidden until set).
  const errorBanner = el('div', {
    class: 'history-error',
    role: 'alert',
    hidden: '',
  });

  const body = el(
    'div',
    { class: 'history-body' },
    summaryBlock,
    el('div', { class: 'log-panel' }, filterBar, errorBanner, listContainer, footer),
  );

  const root = el('div', { class: 'history-view' }, head, body);

  return {
    root,
    listContainer,
    statusLine,
    loadMoreBtn,
    summaryBody: summaryBlock.querySelector('#history-summary-body'),
    errorBanner,
  };
}

function showError(message) {
  if (!elements) return;
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  if (!elements) return;
  elements.errorBanner.textContent = '';
  elements.errorBanner.hidden = true;
}

function renderRows(items, { replace }) {
  if (!elements) return;
  if (replace) {
    elements.listContainer.replaceChildren();
  }
  if (items.length === 0 && replace) {
    elements.listContainer.replaceChildren(
      el('div', { class: 'log-row history-empty' }, 'Записей не найдено'),
    );
    return;
  }
  for (const d of items) {
    elements.listContainer.append(decisionLogRow(d));
  }
}

function renderStatus() {
  if (!elements) return;
  if (state.loading) {
    elements.statusLine.textContent = 'Загрузка…';
    elements.loadMoreBtn.hidden = true;
    return;
  }
  if (state.total === 0) {
    elements.statusLine.textContent = '';
    elements.loadMoreBtn.hidden = true;
    return;
  }
  elements.statusLine.textContent = `Показано ${state.total} запис.`;
}

/** Toggle "Load more" visibility based on whether the last page was full.
 *  If the server returned fewer than limit rows, we've hit the tail. */
function updateLoadMore(lastPageCount) {
  if (!elements) return;
  elements.loadMoreBtn.hidden = lastPageCount < state.limit;
}

async function loadSummary() {
  try {
    const data = await apiFetch('/api/decisions/summary');
    if (elements && elements.summaryBody) {
      elements.summaryBody.textContent = data.summary_md || '—';
    }
  } catch (err) {
    if (elements && elements.summaryBody) {
      const detail = err instanceof ApiError ? err.message : String(err);
      elements.summaryBody.textContent = `Не удалось загрузить summary: ${detail}`;
    }
  }
}

async function load({ replace }) {
  if (!elements) return;
  state.loading = true;
  clearError();
  renderStatus();
  try {
    const params = new URLSearchParams();
    if (state.phase) params.set('phase', state.phase);
    params.set('limit', String(state.limit));
    params.set('offset', String(state.offset));
    const data = await apiFetch(`/api/decisions?${params.toString()}`);
    const items = Array.isArray(data.items) ? data.items : [];
    if (replace) state.total = 0;
    state.total += items.length;
    renderRows(items, { replace });
    updateLoadMore(items.length);
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка загрузки решений: ${detail}`);
    if (replace) {
      // On a failed initial load wipe stale rows so user isn't confused.
      elements.listContainer.replaceChildren();
    }
  } finally {
    state.loading = false;
    renderStatus();
  }
}

/** Router entry-point. Called once per session with the section[data-view]
 *  container. Replaces placeholder content with the real History view. */
export function init(container) {
  // Reset module state in case init() runs more than once (defensive — router
  // caches initialised views, but tests / hot-reload may re-enter).
  state.phase = '';
  state.limit = DEFAULT_LIMIT;
  state.offset = 0;
  state.loading = false;
  state.total = 0;

  const skeleton = buildSkeleton();
  elements = skeleton;

  // Replace the placeholder div in <section data-view="history"> entirely.
  container.replaceChildren(skeleton.root);

  // Kick off initial loads in parallel.
  loadSummary();
  load({ replace: true });
}
