// Tab 02 — Обучение модели. PR 8 wires training job + feature-importance chart.
//
// Flow:
//   GET  /api/system/steel-classes              → {items, count}
//   POST /api/train/run {class_id, n_samples, n_trials, seed}
//                                               → {job_id, config}
//   GET  /api/jobs/{job_id} (polled every 2 s)  → result {version, metrics, ...}
//   DELETE /api/jobs/{job_id}                   → cooperative cancellation
//
// n_samples is sent for every class but the backend silently ignores it
// for fatigue_carbon_steel — the synthetic_generator profile loads the
// fixed 437-record Agrawal NIMS parquet regardless. No client-side
// branching: the input is shown for every class for UX consistency.
//
// PR 8 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { renderFeatureImportance } from '../charts/feature-importance.js';

// Module-scoped state. Re-init() resets it.
const state = {
  classes: [],
  selectedClassId: 'pipe_hsla',
  lastResult: null,
  currentJobId: null,
  pollAbort: null,
  loading: false,
  running: false,
};

let elements = null;

// ──────────────────────────────────────────────────────────────────────
// helpers
// ──────────────────────────────────────────────────────────────────────

function activeProfile() {
  if (!state.selectedClassId) return null;
  return state.classes.find((c) => c.id === state.selectedClassId) || null;
}

function fmt(v, d = 3) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

function fmtPct(v, d = 1) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(Number(v) * 100).toFixed(d)}%`;
}

// ──────────────────────────────────────────────────────────────────────
// skeleton
// ──────────────────────────────────────────────────────────────────────

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
        el('span', { class: 'here' }, 'Обучение модели'),
      ),
      el('h1', { class: 'section-title' }, 'Обучение модели — XGBoost + uncertainty'),
      el(
        'p',
        { class: 'section-sub' },
        'Time-based split + GroupKFold (защита от data leakage), Optuna hyperparameter ' +
          'tuning, quantile regression для 90% CI, GMM OOD-детектор, split-conformal ' +
          'калибровка интервалов. Обучение занимает 1-5 минут.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'train-error',
    role: 'alert',
    hidden: '',
  });

  // Inputs panel.
  const classSelect = el('select', {
    class: 'history-select mono',
    id: 'train-class-select',
    onChange: (ev) => {
      state.selectedClassId = ev.target.value;
      updateClassUI();
    },
  });

  const classDescription = el('div', { class: 'train-class-description mono' }, '');

  const classRow = el(
    'div',
    { class: 'train-form-row' },
    el(
      'label',
      { class: 'train-field', for: classSelect.id },
      el('span', { class: 'train-field-label' }, 'Класс стали'),
      classSelect,
    ),
    classDescription,
  );

  // n_samples — always shown. For fatigue_carbon_steel the backend
  // silently ignores it (the generator loads the fixed 437-record
  // Agrawal NIMS parquet); the slider stays visible regardless of class
  // for UX consistency.
  const nSamplesInput = el('input', {
    type: 'number',
    class: 'train-input mono',
    id: 'train-n-samples',
    value: '800',
    min: '100',
    max: '5000',
    step: '100',
  });
  const nSamplesField = el(
    'label',
    { class: 'train-field', for: nSamplesInput.id },
    el('span', { class: 'train-field-label' }, 'n_samples (synthetic)'),
    nSamplesInput,
  );

  const nTrialsInput = el('input', {
    type: 'number',
    class: 'train-input mono',
    id: 'train-n-trials',
    value: '30',
    min: '5',
    max: '200',
    step: '5',
  });
  const seedInput = el('input', {
    type: 'number',
    class: 'train-input mono',
    id: 'train-seed',
    value: '42',
    min: '0',
    step: '1',
  });

  const formGrid = el(
    'div',
    { class: 'train-form-grid' },
    nSamplesField,
    el(
      'label',
      { class: 'train-field', for: nTrialsInput.id },
      el('span', { class: 'train-field-label' }, 'Optuna trials'),
      nTrialsInput,
    ),
    el(
      'label',
      { class: 'train-field', for: seedInput.id },
      el('span', { class: 'train-field-label' }, 'Random seed'),
      seedInput,
    ),
  );

  const runBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn primary',
      onClick: () => runTrain(),
    },
    'Запустить обучение',
  );

  const actions = el('div', { class: 'train-actions' }, runBtn);
  const progressMount = el('div', { class: 'train-progress-mount' });

  const formPanel = el(
    'div',
    { class: 'train-form-panel' },
    classRow,
    formGrid,
    actions,
    progressMount,
  );

  // Result panel — metrics, critic, feature importance, download.
  const resultHeader = el('div', { class: 'train-result-header' }, '');
  const metricsMount = el('div', { class: 'train-metrics-grid' });
  const criticMount = el('div', { class: 'train-critic-mount' });
  const importanceMount = el('div', { class: 'train-importance-mount' });
  const downloadBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn',
      onClick: () => downloadMetricsJson(),
    },
    'Скачать metrics JSON',
  );
  const resultActions = el('div', { class: 'train-result-actions' }, downloadBtn);
  const resultPanel = el(
    'div',
    { class: 'train-result-panel', hidden: '' },
    resultHeader,
    metricsMount,
    criticMount,
    el(
      'div',
      { class: 'train-importance-section' },
      el('h3', { class: 'train-section-title' }, 'Feature importance (top 20)'),
      importanceMount,
    ),
    resultActions,
  );

  const body = el('div', { class: 'train-body' }, errorBanner, formPanel, resultPanel);
  const root = el('div', { class: 'train-view' }, head, body);

  return {
    root,
    errorBanner,
    classSelect,
    classDescription,
    nSamplesInput,
    nSamplesField,
    nTrialsInput,
    seedInput,
    runBtn,
    progressMount,
    resultPanel,
    resultHeader,
    metricsMount,
    criticMount,
    importanceMount,
    downloadBtn,
  };
}

function showError(msg) {
  if (!elements) return;
  elements.errorBanner.textContent = msg;
  elements.errorBanner.hidden = false;
}

function clearError() {
  if (!elements) return;
  elements.errorBanner.textContent = '';
  elements.errorBanner.hidden = true;
}

// ──────────────────────────────────────────────────────────────────────
// class selector + form toggle
// ──────────────────────────────────────────────────────────────────────

function renderClassSelect() {
  if (!elements) return;
  const sel = elements.classSelect;
  sel.replaceChildren();
  for (const c of state.classes) {
    const opt = el('option', { value: c.id }, `${c.name} (${c.standard})`);
    if (c.id === state.selectedClassId) opt.selected = true;
    sel.append(opt);
  }
  updateClassUI();
}

function updateClassUI() {
  if (!elements) return;
  const profile = activeProfile();
  if (!profile) {
    elements.classDescription.textContent = '';
    return;
  }
  const targetId = profile.target_properties?.[0]?.id || '';
  const targetLabel = profile.target_properties?.[0]?.label || targetId;
  elements.classDescription.textContent =
    `Target: ${targetLabel} · features: ${profile.feature_set.length}`;
}

// ──────────────────────────────────────────────────────────────────────
// run / cancel
// ──────────────────────────────────────────────────────────────────────

function readRequestBody() {
  const classId = state.selectedClassId;
  if (!classId) throw new Error('Класс стали не выбран');
  const nTrials = parseInt(elements.nTrialsInput.value, 10);
  if (!Number.isFinite(nTrials) || nTrials < 5 || nTrials > 200) {
    throw new Error('n_trials должен быть в диапазоне 5..200');
  }
  const seed = parseInt(elements.seedInput.value, 10);
  if (!Number.isFinite(seed) || seed < 0) {
    throw new Error('seed должен быть неотрицательным целым');
  }
  const nSamples = parseInt(elements.nSamplesInput.value, 10);
  if (!Number.isFinite(nSamples) || nSamples < 100 || nSamples > 5000) {
    throw new Error('n_samples должен быть в диапазоне 100..5000');
  }
  // Backend ignores n_samples for classes whose synthetic_generator
  // loads a fixed real-data parquet (e.g. fatigue_carbon_steel uses
  // 437-record Agrawal NIMS), so we always send it.
  return {
    class_id: classId,
    n_trials: nTrials,
    seed,
    n_samples: nSamples,
  };
}

async function runTrain() {
  if (!elements || state.running) return;
  clearError();

  let body;
  try {
    body = readRequestBody();
  } catch (err) {
    showError(err.message);
    return;
  }

  state.running = true;
  elements.runBtn.disabled = true;
  elements.runBtn.textContent = 'Идёт обучение…';
  elements.resultPanel.hidden = true;
  elements.progressMount.replaceChildren();

  const abort = new AbortController();
  state.pollAbort = abort;

  const progress = renderJobProgress(elements.progressMount, {
    label: 'Обучение модели…',
    onCancel: () => cancelTrain(),
  });

  let jobId = null;
  try {
    const submit = await apiFetch('/api/train/run', { method: 'POST', body });
    jobId = submit.job_id;
    state.currentJobId = jobId;
    if (!jobId) throw new Error('Сервер не вернул job_id');
    progress.updateMessage(`job_id=${jobId.slice(0, 8)}…`);

    const result = await pollJob(jobId, {
      interval: 2000,
      signal: abort.signal,
      onProgress: (p) => progress.updateProgress(p),
      onMessage: (m) => progress.updateMessage(m),
    });
    progress.markDone('Готово');
    state.lastResult = result;
    renderResult();
  } catch (err) {
    if (err && err.name === 'AbortError') {
      progress.markError('Отменено пользователем');
    } else {
      const detail = err instanceof ApiError ? err.message : String(err.message || err);
      progress.markError(detail);
      showError(`Ошибка: ${detail}`);
    }
  } finally {
    state.running = false;
    state.pollAbort = null;
    state.currentJobId = null;
    elements.runBtn.disabled = false;
    elements.runBtn.textContent = 'Запустить обучение';
  }
}

async function cancelTrain() {
  // Same pattern as design.js — abort polling immediately on the client,
  // best-effort DELETE on the server. Per PR 8 design doc note:
  // cooperative cancellation only fires between coarse stages, never
  // mid-Optuna. Once the worker is past the "training" stage, DELETE
  // sets the flag but Optuna runs to completion.
  const jobId = state.currentJobId;
  if (state.pollAbort) state.pollAbort.abort();
  if (!jobId) return;
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  } catch (err) {
    console.warn('[train] DELETE /api/jobs failed', err);
  }
}

// ──────────────────────────────────────────────────────────────────────
// result rendering
// ──────────────────────────────────────────────────────────────────────

function renderResult() {
  if (!elements) return;
  const data = state.lastResult;
  if (!data) {
    elements.resultPanel.hidden = true;
    return;
  }
  elements.resultPanel.hidden = false;

  // Header — version + duration + class.
  elements.resultHeader.replaceChildren(
    el(
      'div',
      { class: 'train-result-title' },
      el('span', {}, 'Модель обучена: '),
      el('strong', { class: 'mono' }, data.version || '—'),
    ),
    el(
      'div',
      { class: 'train-result-meta mono' },
      `${data.class_id} · target: ${data.target} · `,
      `длительность: ${fmt(data.duration_s, 1)} с`,
    ),
  );

  // Metrics grid. Coverage gets colour-coded against the [0.85, 0.95]
  // calibration target (M02 in pattern_library — "85-95%" coverage for
  // 90% CI). Outside this range = warning. Inside = success.
  const m = data.metrics || {};
  elements.metricsMount.replaceChildren(
    metricCard('R² test', fmt(m.r2_test, 3), classifyR2(m.r2_test)),
    metricCard('MAE test', fmt(m.mae_test, 2), 'neutral'),
    metricCard('R² train', fmt(m.r2_train, 3), 'neutral'),
    metricCard('Coverage 90% CI', fmtPct(m.coverage_90_ci, 1), classifyCoverage(m.coverage_90_ci)),
    metricCard('RMSE test', fmt(m.rmse_test, 2), 'neutral'),
    metricCard(
      'Conformal Q, МПа',
      fmt(m.conformal_correction_mpa, 1),
      'neutral',
    ),
    metricCard('n_train', String(m.n_train ?? '—'), 'neutral'),
    metricCard('n_test', String(m.n_test ?? '—'), 'neutral'),
  );

  // Critic warnings — pattern_warnings + llm_observations as separate
  // sections. Empty section when both are missing.
  renderCritic(data.critic || {});

  // Feature importance chart.
  const items = Array.isArray(data.feature_importance) ? data.feature_importance : [];
  renderFeatureImportance(elements.importanceMount, items, {
    maxBars: 20,
  }).catch((err) => {
    elements.importanceMount.replaceChildren(
      el(
        'div',
        { class: 'train-chart-fallback' },
        `Не удалось построить chart: ${err.message}`,
      ),
    );
  });
}

function metricCard(label, value, tone) {
  return el(
    'div',
    { class: `train-metric-card train-metric-${tone || 'neutral'}` },
    el('span', { class: 'train-metric-label' }, label),
    el('span', { class: 'train-metric-value mono' }, value),
  );
}

function classifyCoverage(v) {
  // Pattern Library M02: target 85-95% for 90% CI. Outside → warning;
  // inside → green. None / NaN → neutral.
  if (v == null || Number.isNaN(v)) return 'neutral';
  if (v >= 0.85 && v <= 0.95) return 'good';
  return 'warn';
}

function classifyR2(v) {
  if (v == null || Number.isNaN(v)) return 'neutral';
  if (v >= 0.7) return 'good';
  if (v >= 0.5) return 'warn';
  return 'bad';
}

function renderCritic(critic) {
  if (!elements) return;
  const patternWarnings = Array.isArray(critic.pattern_warnings)
    ? critic.pattern_warnings
    : [];
  const llmObservations = critic.llm_observations; // null | array

  const sections = [];

  // Pattern Library section. Title always present so the user knows
  // checks ran; "проблем не обнаружено" message when empty.
  sections.push(
    el(
      'div',
      { class: 'train-critic-section' },
      el('h3', { class: 'train-section-title' }, 'Critic — Pattern Library'),
      patternWarnings.length === 0
        ? el(
            'div',
            { class: 'train-critic-empty' },
            'Pattern Library не нашёл проблем',
          )
        : el(
            'ul',
            { class: 'train-critic-list' },
            ...patternWarnings.map((w) => criticItem(w, 'pattern')),
          ),
    ),
  );

  // LLM-Critic section — null = "не настроен", array = run results.
  if (llmObservations === null) {
    sections.push(
      el(
        'div',
        { class: 'train-critic-section' },
        el('h3', { class: 'train-section-title' }, 'LLM-Critic'),
        el(
          'div',
          { class: 'train-critic-empty muted' },
          'LLM-Critic отключён (нет ANTHROPIC_API_KEY)',
        ),
      ),
    );
  } else {
    sections.push(
      el(
        'div',
        { class: 'train-critic-section' },
        el('h3', { class: 'train-section-title' }, 'LLM-Critic (Claude Sonnet 4.6)'),
        llmObservations.length === 0
          ? el(
              'div',
              { class: 'train-critic-empty' },
              'LLM-Critic не нашёл проблем',
            )
          : el(
              'ul',
              { class: 'train-critic-list' },
              ...llmObservations.map((o) => criticItem(o, 'llm')),
            ),
      ),
    );
  }

  elements.criticMount.replaceChildren(...sections);
}

function criticItem(w, kind) {
  // Pattern shape: {pattern_id, severity, message, suggestion, title?, details?}
  // LLM shape:     {severity, category, message, rationale}
  const severity = String(w.severity || 'LOW').toUpperCase();
  const sevClass = `severity-${severity.toLowerCase()}`;

  const headParts =
    kind === 'pattern'
      ? [
          el('span', { class: 'train-critic-id mono' }, w.pattern_id || '?'),
          el('span', { class: `train-critic-sev ${sevClass}` }, severity),
        ]
      : [
          el(
            'span',
            { class: 'train-critic-id mono' },
            (w.category || 'observation').toLowerCase(),
          ),
          el('span', { class: `train-critic-sev ${sevClass}` }, severity),
        ];

  const detail =
    kind === 'pattern'
      ? el(
          'div',
          { class: 'train-critic-suggestion' },
          el('span', { class: 'muted' }, '→ '),
          w.suggestion || '',
        )
      : el(
          'div',
          { class: 'train-critic-suggestion' },
          el('span', { class: 'muted' }, '→ '),
          w.rationale || '',
        );

  return el(
    'li',
    { class: `train-critic-item ${sevClass}` },
    el('div', { class: 'train-critic-head' }, ...headParts),
    el('div', { class: 'train-critic-message' }, w.message || ''),
    detail,
  );
}

// ──────────────────────────────────────────────────────────────────────
// download
// ──────────────────────────────────────────────────────────────────────

function downloadMetricsJson() {
  const data = state.lastResult;
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json;charset=utf-8',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `model-metrics-${data.version || 'untitled'}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

// ──────────────────────────────────────────────────────────────────────
// bootstrap
// ──────────────────────────────────────────────────────────────────────

async function loadAll() {
  if (!elements) return;
  state.loading = true;
  clearError();
  try {
    const resp = await apiFetch('/api/system/steel-classes');
    state.classes = Array.isArray(resp.items) ? resp.items : [];
    if (state.classes.length === 0) {
      showError('Нет зарегистрированных steel-class профилей');
      return;
    }
    // Default to pipe_hsla if available, else first registered.
    const found = state.classes.find((c) => c.id === 'pipe_hsla');
    state.selectedClassId = found ? found.id : state.classes[0].id;
    renderClassSelect();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить классы стали: ${detail}`);
  } finally {
    state.loading = false;
  }
}

/** Router entry-point. Called once per session with the section[data-view]
 *  container. Replaces placeholder content with the real Train view. */
export function init(container) {
  // Reset module state defensively.
  state.classes = [];
  state.selectedClassId = 'pipe_hsla';
  state.lastResult = null;
  state.currentJobId = null;
  state.pollAbort = null;
  state.loading = false;
  state.running = false;

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadAll();
}
