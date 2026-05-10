// Tab — Следующие эксперименты (cost-weighted Expected Improvement).
//
// Top-K candidates ranked by EI / cost — prefers cheap experiments with
// high expected uplift on the target property.
//
// Data flow:
//   GET  /api/system/models                  → {items, count}
//   GET  /api/system/models/active           → active model (404 if none)
//   GET  /api/system/steel-classes           → {items} for class profile lookup
//   POST /api/active-learning/propose        → {candidates, baseline, model, …}
//
// Class gate: backend rejects non-fatigue_carbon_steel models with 400.
// The view also surfaces a clear banner when the active model isn't
// fatigue_carbon_steel — better UX than waiting for the round-trip.
//
// PR 5 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';
import { buildCsv, downloadCsv } from '../utils/csv.js';

// ──────────────────── module state ────────────────────

const SUPPORTED_STEEL_CLASS = 'fatigue_carbon_steel';

const state = {
  models: [],          // [{version, steel_class, target, …}]
  classes: new Map(),  // class_id -> profile
  selectedVersion: null,
  formValues: {
    n_samples: 2000,
    top_k: 5,
    seed: 42,
  },
  loading: false,
  busy: false,
  lastResult: null,
};

let elements = null;

// ──────────────────── helpers ────────────────────

function activeModel() {
  if (!state.selectedVersion) return null;
  return state.models.find((m) => m.version === state.selectedVersion) || null;
}

function activeProfile() {
  const m = activeModel();
  if (!m) return null;
  return state.classes.get(m.steel_class) || null;
}

function isModelSupported(model) {
  return !!model && model.steel_class === SUPPORTED_STEEL_CLASS;
}

function formatNumber(value, decimals) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

function formatDelta(value, decimals, unit) {
  // ±N delta. Returns plain string (caller decides how to wrap).
  if (value == null || Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(decimals)}${unit ? ` ${unit}` : ''}`;
}

function tsForFilename() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

// ──────────────────── skeleton ────────────────────

function buildSkeleton() {
  const head = el(
    'div',
    { class: 'section-head' },
    el(
      'div',
      {},
      el(
        'div',
        { class: 'breadcrumb' },
        'Рабочий процесс',
        el('span', { class: 'sep' }, '/'),
        el('span', { class: 'here' }, 'Следующие эксперименты'),
      ),
      el(
        'h1',
        { class: 'section-title' },
        'Следующие эксперименты — cost-weighted EI',
      ),
      el(
        'p',
        { class: 'section-sub' },
        'Bayesian active learning. Стохастический LHS-скан над feasible space ' +
          'модели; кандидаты ранжированы по Expected Improvement / cost — ' +
          'предпочитаются дешёвые эксперименты с высоким ожидаемым ' +
          'приростом свойства относительно best-observed-training f*.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'al-error',
    role: 'alert',
    hidden: '',
  });

  const emptyBanner = el(
    'div',
    { class: 'al-empty', hidden: '' },
    el(
      'p',
      {},
      'Нет обученных моделей. Сначала обучите модель во вкладке ',
      el('strong', {}, '«Обучение модели»'),
      '.',
    ),
  );

  const classGateBanner = el(
    'div',
    { class: 'al-empty', hidden: '' },
    el(
      'p',
      {},
      'Active learner работает только на классе ',
      el('strong', {}, 'fatigue_carbon_steel'),
      ' (Agrawal NIMS). Выберите такую модель в селекторе.',
    ),
  );

  // Model selector + meta caption.
  const modelSelect = el('select', {
    class: 'history-select mono',
    id: 'al-model-select',
    onChange: (ev) => {
      state.selectedVersion = ev.target.value;
      state.lastResult = null;
      renderModelMeta();
      renderResult();
      updateFormAvailability();
    },
  });

  const modelMeta = el('div', { class: 'al-model-meta mono' }, '');

  const modelRow = el(
    'div',
    { class: 'al-model-row' },
    el(
      'label',
      { class: 'al-model-cell', for: 'al-model-select' },
      el('span', { class: 'al-model-label' }, 'Активная модель'),
      modelSelect,
    ),
    modelMeta,
  );

  // Inputs grid: n_samples / top_k / seed.
  const nSamplesInput = el('input', {
    type: 'number',
    class: 'al-input',
    id: 'al-n-samples',
    min: '100',
    max: '10000',
    step: '100',
    value: String(state.formValues.n_samples),
    'data-field': 'n_samples',
  });
  const topKInput = el('input', {
    type: 'number',
    class: 'al-input',
    id: 'al-top-k',
    min: '1',
    max: '50',
    step: '1',
    value: String(state.formValues.top_k),
    'data-field': 'top_k',
  });
  const seedInput = el('input', {
    type: 'number',
    class: 'al-input',
    id: 'al-seed',
    min: '0',
    step: '1',
    value: String(state.formValues.seed),
    'data-field': 'seed',
  });

  const formGrid = el(
    'div',
    { class: 'al-form-grid' },
    el(
      'div',
      { class: 'al-field' },
      el(
        'label',
        { class: 'al-field-label', for: 'al-n-samples' },
        'LHS samples (n_samples)',
      ),
      nSamplesInput,
      el('span', { class: 'al-field-hint' }, '500 — стандарт, 5000 — финальный план, 100-10000'),
    ),
    el(
      'div',
      { class: 'al-field' },
      el(
        'label',
        { class: 'al-field-label', for: 'al-top-k' },
        'Top-K кандидатов',
      ),
      topKInput,
      el('span', { class: 'al-field-hint' }, 'Сколько лучших экспериментов вернуть, 1-50'),
    ),
    el(
      'div',
      { class: 'al-field' },
      el('label', { class: 'al-field-label', for: 'al-seed' }, 'Seed'),
      seedInput,
      el('span', { class: 'al-field-hint' }, 'Reproducibility — при одинаковом seed результаты совпадают'),
    ),
  );

  const proposeBtn = el(
    'button',
    {
      class: 'btn primary',
      type: 'button',
      onClick: () => runPropose(),
    },
    'Предложить эксперименты',
  );
  const busyLabel = el(
    'span',
    { class: 'al-busy', hidden: '' },
    'Считаем EI… (~2 c)',
  );
  const actionsRow = el('div', { class: 'al-actions' }, busyLabel, proposeBtn);

  const formPanel = el(
    'div',
    { class: 'al-form-panel' },
    modelRow,
    formGrid,
    actionsRow,
  );

  const resultContainer = el('div', { class: 'al-result-container' });

  const body = el(
    'div',
    { class: 'al-body' },
    errorBanner,
    emptyBanner,
    classGateBanner,
    formPanel,
    resultContainer,
  );

  const root = el('div', { class: 'al-view' }, head, body);

  return {
    root,
    errorBanner,
    emptyBanner,
    classGateBanner,
    modelSelect,
    modelMeta,
    formPanel,
    nSamplesInput,
    topKInput,
    seedInput,
    proposeBtn,
    busyLabel,
    resultContainer,
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

function showEmpty() {
  if (!elements) return;
  elements.emptyBanner.hidden = false;
  elements.classGateBanner.hidden = true;
  elements.formPanel.hidden = true;
  elements.resultContainer.replaceChildren();
}

function showClassGate() {
  if (!elements) return;
  elements.emptyBanner.hidden = true;
  elements.classGateBanner.hidden = false;
  // We keep the form visible (with the model selector active) so the user
  // can switch models without losing the inputs.
  elements.formPanel.hidden = false;
  elements.proposeBtn.disabled = true;
  elements.resultContainer.replaceChildren();
}

function showForm() {
  if (!elements) return;
  elements.emptyBanner.hidden = true;
  elements.classGateBanner.hidden = true;
  elements.formPanel.hidden = false;
  elements.proposeBtn.disabled = state.busy;
}

function updateFormAvailability() {
  // Called after the model select changes. Decides between class-gate
  // banner vs full form.
  const m = activeModel();
  if (!m) {
    showEmpty();
    return;
  }
  if (!isModelSupported(m)) {
    showClassGate();
    return;
  }
  showForm();
}

// ──────────────────── form rendering ────────────────────

function renderModelSelect() {
  if (!elements) return;
  const sel = elements.modelSelect;
  sel.replaceChildren();
  for (const m of state.models) {
    const profile = state.classes.get(m.steel_class);
    const className = profile ? profile.name : m.steel_class;
    const opt = el(
      'option',
      { value: m.version },
      `${m.version} · ${className}`,
    );
    if (m.version === state.selectedVersion) {
      opt.selected = true;
    }
    sel.append(opt);
  }
  renderModelMeta();
}

function renderModelMeta() {
  if (!elements) return;
  const m = activeModel();
  const profile = activeProfile();
  if (!m) {
    elements.modelMeta.textContent = '';
    return;
  }
  const className = profile ? profile.name : m.steel_class;
  elements.modelMeta.textContent = `Класс: ${className} · target: ${m.target}`;
}

function readFormValues() {
  // Pull from inputs, validate ranges, throw user-readable error on bad input.
  // Pydantic on the backend will also reject (422) but we want a fast UI path.
  if (!elements) return null;
  const n = parseInt(elements.nSamplesInput.value, 10);
  const k = parseInt(elements.topKInput.value, 10);
  const s = parseInt(elements.seedInput.value, 10);
  if (Number.isNaN(n) || n < 100 || n > 10000) {
    throw new Error('n_samples должен быть целым числом 100-10000');
  }
  if (Number.isNaN(k) || k < 1 || k > 50) {
    throw new Error('Top-K должен быть целым числом 1-50');
  }
  if (Number.isNaN(s) || s < 0) {
    throw new Error('Seed должен быть неотрицательным целым числом');
  }
  return { n_samples: n, top_k: k, seed: s };
}

// ──────────────────── propose + result ────────────────────

async function runPropose() {
  if (!elements) return;
  if (state.busy) return;
  const model = activeModel();
  if (!model) return;
  if (!isModelSupported(model)) {
    showClassGate();
    return;
  }

  let values;
  try {
    values = readFormValues();
  } catch (err) {
    showError(err.message);
    return;
  }
  if (!values) return;
  state.formValues = values;

  clearError();
  state.busy = true;
  elements.proposeBtn.disabled = true;
  elements.proposeBtn.textContent = 'Считаем…';
  elements.busyLabel.hidden = false;

  try {
    const data = await apiFetch('/api/active-learning/propose', {
      method: 'POST',
      body: { model_version: model.version, ...values },
    });
    state.lastResult = data;
    renderResult();
  } catch (err) {
    state.lastResult = null;
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка active-learning: ${detail}`);
    renderResult();
  } finally {
    state.busy = false;
    elements.proposeBtn.disabled = false;
    elements.proposeBtn.textContent = 'Предложить эксперименты';
    elements.busyLabel.hidden = true;
  }
}

function renderBaselineGrid(baseline, computationTime) {
  return el(
    'div',
    { class: 'al-baseline-grid' },
    el(
      'div',
      { class: 'al-baseline-cell' },
      el('span', { class: 'al-baseline-cell-label' }, 'Baseline σ_pred'),
      el(
        'span',
        { class: 'al-baseline-cell-value' },
        `${formatNumber(baseline.predicted_property, 0)} МПа`,
      ),
    ),
    el(
      'div',
      { class: 'al-baseline-cell' },
      el('span', { class: 'al-baseline-cell-label' }, 'Baseline cost'),
      el(
        'span',
        { class: 'al-baseline-cell-value' },
        `${formatNumber(baseline.cost_per_ton, 2)} €/т`,
      ),
    ),
    el(
      'div',
      { class: 'al-baseline-cell' },
      el('span', { class: 'al-baseline-cell-label' }, 'f* (target)'),
      el(
        'span',
        { class: 'al-baseline-cell-value' },
        `${formatNumber(baseline.f_star, 0)} МПа`,
      ),
    ),
    el(
      'div',
      { class: 'al-baseline-cell' },
      el('span', { class: 'al-baseline-cell-label' }, 'Время расчёта'),
      el(
        'span',
        { class: 'al-baseline-cell-value' },
        `${formatNumber(computationTime, 2)} с`,
      ),
    ),
  );
}

function compositionCell(c) {
  // Compact composition + process_params display: one short line per key.
  const lines = [];
  const comp = c.composition || {};
  const proc = c.process_params || {};
  for (const [k, v] of Object.entries(comp)) {
    lines.push(
      el(
        'span',
        { class: 'al-composition-line' },
        el('span', { class: 'key' }, `${k}:`),
        formatNumber(v, 4),
      ),
    );
  }
  for (const [k, v] of Object.entries(proc)) {
    lines.push(
      el(
        'span',
        { class: 'al-composition-line' },
        el('span', { class: 'key' }, `${k}:`),
        formatNumber(v, 1),
      ),
    );
  }
  return el('div', { class: 'al-composition-cell' }, ...lines);
}

function deltaSpan(value, decimals, unit, invert = false) {
  // ``invert=true`` means lower-is-better (cost) — flip pos/neg colour.
  const text = formatDelta(value, decimals, unit);
  if (value == null || Number.isNaN(value)) return text;
  const positive = invert ? value < 0 : value > 0;
  const cls = positive ? 'al-delta-pos' : 'al-delta-neg';
  return el('span', { class: cls }, text);
}

function oodBadge(isOod) {
  return el(
    'span',
    { class: `al-ood-badge ${isOod ? 'ood' : 'in-spec'}` },
    isOod ? 'OOD' : 'in-spec',
  );
}

function renderTable(candidates) {
  if (!candidates.length) {
    return el(
      'div',
      { class: 'al-empty' },
      'Алгоритм не нашёл feasible-кандидатов с положительной EI. ' +
        'Попробуйте увеличить n_samples или сменить seed.',
    );
  }

  // Highlight top EI + top Δσ rows.
  const maxEi = Math.max(...candidates.map((c) => c.expected_improvement));
  const maxDelta = Math.max(
    ...candidates.map((c) => c.delta_vs_baseline_property),
  );

  const head = el(
    'thead',
    {},
    el(
      'tr',
      {},
      el('th', {}, '#'),
      el('th', {}, 'OOD'),
      el('th', {}, 'Композиция / процесс'),
      el('th', { class: 'right' }, 'σ pred ± CI, МПа'),
      el('th', { class: 'right' }, 'EI / cost'),
      el('th', { class: 'right' }, 'EI, МПа'),
      el('th', { class: 'right' }, 'Δσ vs baseline'),
      el('th', { class: 'right' }, 'Cost, €/т'),
      el('th', { class: 'right' }, 'Δcost vs baseline'),
    ),
  );

  const body = el('tbody', {});
  for (const c of candidates) {
    const isHighlight = c.expected_improvement === maxEi || c.delta_vs_baseline_property === maxDelta;
    const tr = el(
      'tr',
      {
        class: [
          c.is_ood ? 'is-ood' : '',
          isHighlight ? 'highlight-ei' : '',
        ].filter(Boolean).join(' '),
      },
      el('td', {}, el('span', { class: 'al-rank' }, `#${c.rank}`)),
      el('td', {}, oodBadge(!!c.is_ood)),
      el('td', {}, compositionCell(c)),
      el(
        'td',
        { class: 'right' },
        `${formatNumber(c.predicted_property, 0)}`,
        el(
          'div',
          { class: 'al-ci' },
          `[${formatNumber(c.lower_90, 0)}, ${formatNumber(c.upper_90, 0)}]`,
        ),
      ),
      el('td', { class: 'right' }, formatNumber(c.acquisition_score, 4)),
      el('td', { class: 'right' }, formatNumber(c.expected_improvement, 1)),
      el(
        'td',
        { class: 'right' },
        deltaSpan(c.delta_vs_baseline_property, 0, 'МПа'),
      ),
      el('td', { class: 'right' }, formatNumber(c.estimated_cost_eur_per_t, 2)),
      el(
        'td',
        { class: 'right' },
        deltaSpan(c.delta_vs_baseline_cost, 2, '€/т', true),
      ),
    );
    body.append(tr);
  }

  return el(
    'div',
    { class: 'al-table-wrap' },
    el('table', { class: 'al-table' }, head, body),
  );
}

function exportCandidatesCsv(data) {
  // Columns mirror the on-screen table. Composition / process_params are
  // collapsed into one space-separated ``key=value`` cell each (the table
  // shows them as a multi-line cell — flat string is good enough for CSV
  // consumers).
  const headers = [
    'rank', 'id', 'is_ood', 'predicted_property_mpa',
    'lower_90', 'upper_90', 'acquisition_score', 'expected_improvement',
    'delta_vs_baseline_property_mpa', 'cost_per_ton_eur',
    'delta_vs_baseline_cost_eur', 'composition', 'process_params',
  ];
  const rows = (data.candidates || []).map((c) => {
    const composition = Object.entries(c.composition || {})
      .map(([k, v]) => `${k}=${Number(v).toFixed(4)}`)
      .join(' ');
    const process = Object.entries(c.process_params || {})
      .map(([k, v]) => `${k}=${Number(v).toFixed(2)}`)
      .join(' ');
    return [
      c.rank,
      c.id,
      c.is_ood ? '1' : '0',
      Number(c.predicted_property).toFixed(2),
      Number(c.lower_90).toFixed(2),
      Number(c.upper_90).toFixed(2),
      Number(c.acquisition_score).toFixed(6),
      Number(c.expected_improvement).toFixed(4),
      Number(c.delta_vs_baseline_property).toFixed(2),
      Number(c.estimated_cost_eur_per_t).toFixed(4),
      Number(c.delta_vs_baseline_cost).toFixed(4),
      composition,
      process,
    ];
  });
  const csv = buildCsv(headers, rows);
  const filename = `next-experiments-${data.model.version}-${tsForFilename()}.csv`;
  downloadCsv(filename, csv);
}

function renderResult() {
  if (!elements) return;
  const container = elements.resultContainer;
  container.replaceChildren();
  const data = state.lastResult;
  if (!data) return;

  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const baseline = data.baseline || {};
  const computationTime = data.computation_time_s;

  const csvBtn = el(
    'button',
    {
      class: 'btn',
      type: 'button',
      onClick: () => exportCandidatesCsv(data),
      disabled: candidates.length === 0 ? '' : false,
      title: 'Скачать кандидатов в CSV',
    },
    'Скачать CSV',
  );

  const headline = el(
    'div',
    { class: 'al-result-headline' },
    el('span', { class: 'al-result-label' }, 'Top-K кандидатов'),
    el(
      'div',
      {
        style: {
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '14px',
        },
      },
      el(
        'span',
        { class: 'al-result-title' },
        `${candidates.length} ranked экспериментов`,
      ),
      csvBtn,
    ),
  );

  const config = data.config || {};
  const footer = el(
    'div',
    { class: 'al-footer' },
    el(
      'span',
      {},
      `n_samples=${config.n_samples ?? '?'} · seed=${config.seed ?? '?'} · ` +
        `decision vars=${(config.decision_vars || []).length}`,
    ),
    el(
      'span',
      {},
      data.model
        ? `model=${data.model.version}`
        : '',
    ),
  );

  container.append(
    el(
      'div',
      { class: 'al-result' },
      headline,
      renderBaselineGrid(baseline, computationTime),
      renderTable(candidates),
      footer,
    ),
  );
}

// ──────────────────── bootstrap ────────────────────

async function loadAll() {
  if (!elements) return;
  state.loading = true;
  clearError();
  try {
    const [modelsResp, classesResp, activeResp] = await Promise.all([
      apiFetch('/api/system/models'),
      apiFetch('/api/system/steel-classes'),
      apiFetch('/api/system/models/active').catch((err) => {
        // 404 is the legitimate "no models trained" path.
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    ]);

    state.models = Array.isArray(modelsResp.items) ? modelsResp.items : [];
    state.classes.clear();
    for (const c of classesResp.items || []) {
      state.classes.set(c.id, c);
    }

    if (state.models.length === 0) {
      showEmpty();
      return;
    }

    // Default selection: prefer the active model if it's a fatigue model;
    // otherwise pick the most recent fatigue_carbon_steel model so the
    // form is immediately usable. Fall back to whatever active gives us.
    const defaultVersion = activeResp ? activeResp.version : state.models[state.models.length - 1].version;
    const fatigueModel = [...state.models].reverse().find((m) => isModelSupported(m));
    const initialVersion = isModelSupported(activeResp) || !fatigueModel
      ? defaultVersion
      : fatigueModel.version;
    state.selectedVersion = initialVersion;

    renderModelSelect();
    updateFormAvailability();
    renderResult();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить состояние: ${detail}`);
  } finally {
    state.loading = false;
  }
}

/** Router entry-point. Called once per session with the section[data-view]
 *  container. Replaces placeholder content with the real view. */
export function init(container) {
  // Reset module state defensively in case init() runs more than once.
  state.models = [];
  state.classes = new Map();
  state.selectedVersion = null;
  state.formValues = { n_samples: 2000, top_k: 5, seed: 42 };
  state.loading = false;
  state.busy = false;
  state.lastResult = null;

  const skeleton = buildSkeleton();
  elements = skeleton;

  container.replaceChildren(skeleton.root);

  loadAll();
}
