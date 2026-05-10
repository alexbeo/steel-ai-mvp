// Tab 03 — Прогноз. Class-aware composition form + /api/predict + anomaly explainer.
//
// Streamlit parity reference: app/frontend/app.py lines 685-901.
// Data flow:
//   GET  /api/system/models                  → {items, count}
//   GET  /api/system/models/active           → model meta (404 if none)
//   GET  /api/system/steel-classes           → {items, count}
//   POST /api/predict {model_version, composition}
//                                            → {prediction, ood, derived, model}
//   POST /api/predict/anomaly-explain        → {job_id} → poll /api/jobs/{id}
//                                            → {explanation, evidence, …}
//
// PR 3 of the Streamlit→FastAPI migration introduced the prediction shell
// with a *placeholder* OOD warning (disabled button). PR 12 wires the button
// up to a real anomaly-explainer job and adds the wide-CI heuristic so the
// button also surfaces on non-OOD predictions whose 90% CI is wide relative
// to the target's normal range. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { el } from '../utils/dom.js';

/** Module-scoped state. Re-init() resets it. */
const state = {
  models: [], // [{version, steel_class, target, ...}]
  classes: new Map(), // class_id -> profile
  selectedVersion: null,
  loading: false,
  predicting: false,
  lastResult: null,
  // PR 12 — anomaly explainer state. Decoupled from predict state so a
  // user can submit a prediction, click «Объяснить», cancel mid-flight,
  // and run another prediction without losing the form values.
  explaining: false,
  lastExplain: null, // {explanation, evidence, ...} from /predict/anomaly-explain job
  explainError: null,
  explainAbort: null, // AbortController for in-flight job poll
};

let elements = null;

// -------------------- helpers --------------------

function activeModel() {
  if (!state.selectedVersion) return null;
  return state.models.find((m) => m.version === state.selectedVersion) || null;
}

function activeProfile() {
  const m = activeModel();
  if (!m) return null;
  return state.classes.get(m.steel_class) || null;
}

/** Format target labels — fall back to target id if profile lacks a match. */
function targetLabelFor(model, profile) {
  if (!profile) return model.target || '';
  const tp = (profile.target_properties || []).find((t) => t.id === model.target);
  return tp ? tp.label : model.target;
}

/** Compose a composition dict of midpoint defaults from physical_bounds. */
function defaultComposition(profile) {
  const out = {};
  for (const feat of profile.feature_set) {
    const bounds = profile.physical_bounds[feat];
    if (!Array.isArray(bounds) || bounds.length !== 2) {
      out[feat] = 0;
      continue;
    }
    out[feat] = (bounds[0] + bounds[1]) / 2;
  }
  return out;
}

function decimalsFor(feat) {
  // Streamlit format selector: %.4f for *_pct, %.2f otherwise (line 749).
  return feat.endsWith('_pct') ? 4 : 2;
}

function formatNumber(value, decimals) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

// -------------------- skeleton --------------------

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
        el('span', { class: 'here' }, 'Прогноз'),
      ),
      el('h1', { class: 'section-title' }, 'Прогноз для заданного состава'),
      el(
        'p',
        { class: 'section-sub' },
        'Точечный прогноз свойства по введённой композиции с 90% доверительным интервалом ' +
          '(conformal-corrected) и OOD-флагом — сигналом что состав вне обучающего распределения.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'predict-error',
    role: 'alert',
    hidden: '',
  });

  const emptyBanner = el(
    'div',
    { class: 'predict-empty', hidden: '' },
    el(
      'p',
      {},
      'Нет обученных моделей. Сначала обучите модель во вкладке ',
      el('strong', {}, '«Обучение модели»'),
      '.',
    ),
  );

  // Model selector row (label + select + class/target captions).
  const modelSelect = el('select', {
    class: 'history-select mono',
    id: 'predict-model-select',
    onChange: (ev) => {
      state.selectedVersion = ev.target.value;
      state.lastResult = null;
      renderForm();
      renderResult();
    },
  });

  const modelMeta = el('div', { class: 'predict-model-meta mono' }, '');

  const modelRow = el(
    'div',
    { class: 'predict-model-row' },
    el(
      'label',
      { class: 'predict-model-cell', for: 'predict-model-select' },
      el('span', { class: 'predict-model-label' }, 'Активная модель'),
      modelSelect,
    ),
    modelMeta,
  );

  // Form container — re-rendered on model change.
  const formContainer = el('div', { class: 'predict-form' });

  // Predict button + result panel.
  const predictBtn = el(
    'button',
    {
      class: 'btn primary',
      type: 'button',
      onClick: () => runPredict(),
    },
    'Прогноз',
  );
  const actionsRow = el('div', { class: 'predict-actions' }, predictBtn);

  const resultContainer = el('div', { class: 'predict-result-container' });

  const body = el(
    'div',
    { class: 'predict-body' },
    errorBanner,
    emptyBanner,
    el('div', { class: 'predict-form-panel' }, modelRow, formContainer, actionsRow),
    resultContainer,
  );

  const root = el('div', { class: 'predict-view' }, head, body);

  return {
    root,
    errorBanner,
    emptyBanner,
    modelSelect,
    modelMeta,
    formContainer,
    predictBtn,
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
  // Hide the form panel + result.
  elements.formContainer.replaceChildren();
  elements.resultContainer.replaceChildren();
  elements.predictBtn.disabled = true;
  elements.modelSelect.disabled = true;
}

function hideEmpty() {
  if (!elements) return;
  elements.emptyBanner.hidden = true;
  elements.predictBtn.disabled = false;
  elements.modelSelect.disabled = false;
}

// -------------------- form rendering --------------------

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
  // Update class/target caption.
  const m = activeModel();
  const profile = activeProfile();
  if (m && profile) {
    elements.modelMeta.textContent = `Класс: ${profile.name} · target: ${m.target}`;
  } else if (m) {
    elements.modelMeta.textContent = `target: ${m.target}`;
  } else {
    elements.modelMeta.textContent = '';
  }
}

function renderForm() {
  if (!elements) return;
  const profile = activeProfile();
  elements.formContainer.replaceChildren();
  if (!profile) {
    return;
  }

  const heading = el(
    'div',
    { class: 'predict-form-heading' },
    el('span', {}, 'Композиция и параметры процесса'),
    el(
      'span',
      { class: 'predict-form-hint' },
      'значения по умолчанию — середина physical_bounds',
    ),
  );
  elements.formContainer.append(heading);

  const grid = el('div', { class: 'predict-feature-grid' });
  const defaults = defaultComposition(profile);

  for (const feat of profile.feature_set) {
    const bounds = profile.physical_bounds[feat] || [0, 1];
    const [lo, hi] = bounds;
    const decimals = decimalsFor(feat);
    const step = hi > lo ? (hi - lo) / 100 : 0.01;

    const input = el('input', {
      type: 'number',
      class: 'predict-input mono',
      id: `predict-feat-${feat}`,
      value: String(Number(defaults[feat]).toFixed(decimals)),
      min: String(lo),
      max: String(hi),
      step: String(step),
      'data-feature': feat,
    });

    const cell = el(
      'div',
      { class: 'predict-feature-cell' },
      el('label', { class: 'predict-feature-label', for: input.id }, feat),
      el(
        'div',
        { class: 'predict-feature-input-wrap' },
        input,
        el(
          'span',
          { class: 'predict-feature-bounds mono' },
          `[${formatNumber(lo, decimals)}, ${formatNumber(hi, decimals)}]`,
        ),
      ),
    );
    grid.append(cell);
  }
  elements.formContainer.append(grid);
}

function readComposition() {
  const profile = activeProfile();
  if (!profile) return null;
  const out = {};
  for (const feat of profile.feature_set) {
    const inp = elements.formContainer.querySelector(
      `input[data-feature="${feat}"]`,
    );
    if (!inp) continue;
    const val = parseFloat(inp.value);
    if (Number.isNaN(val)) {
      throw new Error(`Поле «${feat}» содержит некорректное значение`);
    }
    out[feat] = val;
  }
  return out;
}

// -------------------- predict + result --------------------

async function runPredict() {
  if (!elements) return;
  if (state.predicting) return;
  const model = activeModel();
  if (!model) return;

  let composition;
  try {
    composition = readComposition();
  } catch (err) {
    showError(err.message);
    return;
  }
  if (!composition) return;

  clearError();
  state.predicting = true;
  // Reset any prior anomaly-explainer state — running a new prediction
  // implicitly invalidates the previous diagnosis, and a stale explain
  // panel under a fresh prediction is misleading.
  cancelExplainIfRunning();
  state.lastExplain = null;
  state.explainError = null;
  elements.predictBtn.disabled = true;
  elements.predictBtn.textContent = 'Прогнозирование…';
  try {
    const data = await apiFetch('/api/predict', {
      method: 'POST',
      body: { model_version: model.version, composition },
    });
    state.lastResult = data;
    renderResult();
  } catch (err) {
    state.lastResult = null;
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка прогноза: ${detail}`);
    renderResult();
  } finally {
    state.predicting = false;
    elements.predictBtn.disabled = false;
    elements.predictBtn.textContent = 'Прогноз';
  }
}

// -------------------- anomaly explainer --------------------

/** Cancel any in-flight pollJob (no-op when nothing is running). */
function cancelExplainIfRunning() {
  if (state.explainAbort) {
    try { state.explainAbort.abort(); } catch (_e) { /* ignore */ }
    state.explainAbort = null;
  }
  state.explaining = false;
}

/** Build the OOD/wide-CI warning block + anomaly-explain entry point.
 *
 * Visible whenever ``data.ood.is_ood || data.prediction.wide_ci``. The
 * banner copy adapts: pure OOD shows a strong warning; wide-CI-only
 * uses softer "точка близка к границе training distribution" wording.
 *
 * The button toggles a sub-section that holds either the spinner+progress
 * card (during the job), an error message, or the explanation card.
 * State machine driven by ``state.explaining`` / ``state.lastExplain``.
 */
function buildAnomalyTriggerBlock({ isOod, isWideCi }) {
  const title = isOod
    ? 'Внимание: состав вне training distribution'
    : 'Внимание: широкий доверительный интервал';
  const body = isOod
    ? 'Прогноз ненадёжен — точка лежит за пределами области, где модель училась. ' +
      'Рекомендуется скорректировать состав или собрать дополнительные данные.'
    : 'Доверительный интервал шире половины целевого диапазона свойства — модель ' +
      'не уверена в этой точке. PhD-диагностика подскажет, какие признаки рискованны.';

  const explainBtn = el(
    'button',
    {
      class: 'btn primary',
      type: 'button',
      onClick: () => runAnomalyExplain(),
    },
    'Объяснить почему рискованно',
  );

  // Mount point — runAnomalyExplain replaces children with the
  // job-progress card / error / result card.
  const mount = el('div', { class: 'predict-explain-mount' });

  // Restore any previously rendered explanation when re-rendering.
  if (state.explaining) {
    explainBtn.disabled = true;
    explainBtn.textContent = 'PhD-диагностика…';
  } else if (state.lastExplain) {
    mount.append(buildExplanationCard(state.lastExplain));
    explainBtn.textContent = 'Запустить ещё раз';
  } else if (state.explainError) {
    mount.append(
      el(
        'div',
        { class: 'predict-explain-error', role: 'alert' },
        state.explainError,
      ),
    );
  }

  return el(
    'div',
    { class: 'predict-ood-warning' },
    el('div', { class: 'predict-ood-title' }, title),
    el('div', { class: 'predict-ood-body' }, body),
    el(
      'div',
      { class: 'predict-ood-actions' },
      explainBtn,
      el(
        'span',
        { class: 'predict-ood-hint' },
        'Анализ ~30-60 с, ~$0.05-0.07 за вызов.',
      ),
    ),
    mount,
  );
}

/** Submit POST /api/predict/anomaly-explain → poll job → render result. */
async function runAnomalyExplain() {
  if (!elements) return;
  if (state.explaining) return;
  const data = state.lastResult;
  const model = activeModel();
  if (!data || !model) return;

  let composition;
  try {
    composition = readComposition();
  } catch (err) {
    state.explainError = err.message;
    renderResult();
    return;
  }

  state.explaining = true;
  state.lastExplain = null;
  state.explainError = null;
  renderResult();

  // Mount the progress card on the (now re-created) explain mount.
  const mount = elements.resultContainer.querySelector(
    '.predict-explain-mount',
  );
  if (!mount) {
    state.explaining = false;
    return;
  }
  mount.replaceChildren();

  const abort = new AbortController();
  state.explainAbort = abort;

  const progress = renderJobProgress(mount, {
    label: 'PhD-диагностика аномалии',
    onCancel: () => {
      // Best-effort cancel: abort the poll loop AND fire DELETE so the
      // backend cooperative cancellation flag flips.
      cancelExplainIfRunning();
      // Don't await — the poll is already aborted, the DELETE is
      // purely a backend signal; the user gets immediate feedback.
      // We don't have job_id here yet (closure scope), so the abort
      // alone covers the UI side; the worker may still finish in the
      // background but the UI ignores the result.
    },
  });

  try {
    const submit = await apiFetch('/api/predict/anomaly-explain', {
      method: 'POST',
      body: {
        model_version: model.version,
        composition,
        predicted_mean: Number(data.prediction.mean),
        predicted_q05: Number(data.prediction.q05),
        predicted_q95: Number(data.prediction.q95),
        is_ood: Boolean(data.ood && data.ood.is_ood),
        save_to_decision_log: false,
      },
    });
    const jobId = submit.job_id;

    const result = await pollJob(jobId, {
      signal: abort.signal,
      onProgress: progress.updateProgress,
      onMessage: progress.updateMessage,
    });
    progress.markDone('Готово');
    state.lastExplain = result;
  } catch (err) {
    if (err && err.name === 'AbortError') {
      // User-initiated cancel — leave the result panel empty without an error.
      state.explainError = null;
    } else {
      const detail = err instanceof ApiError ? err.message : String(err);
      state.explainError = `Ошибка диагностики: ${detail}`;
      try { progress.markError(detail); } catch (_e) { /* ignore */ }
    }
  } finally {
    state.explaining = false;
    state.explainAbort = null;
    renderResult();
  }
}

/** Build the explanation panel from a completed job result.
 *
 * Mirrors Streamlit lines 869-901 — severity badge, summary, anomalous
 * features list, mechanism concerns, production risks, suggested
 * correction. Falls back to an "Объяснение не получено" notice when
 * ``result.explanation`` is null (LLM call failed silently).
 */
function buildExplanationCard(result) {
  if (!result || !result.explanation) {
    return el(
      'div',
      { class: 'predict-explain-error', role: 'alert' },
      'Объяснение не получено — попробуйте ещё раз.',
    );
  }
  const exp = result.explanation;

  const sevLabel = { LOW: 'НИЗКАЯ', MEDIUM: 'СРЕДНЯЯ', HIGH: 'ВЫСОКАЯ' }[
    exp.severity
  ] || exp.severity;
  const sevClass = `predict-explain-severity sev-${(exp.severity || 'low').toLowerCase()}`;

  const blocks = [
    el('div', { class: sevClass }, `Опасность: ${sevLabel}`),
    el('p', { class: 'predict-explain-summary' }, exp.summary || ''),
  ];

  if (Array.isArray(exp.anomalous_features) && exp.anomalous_features.length > 0) {
    const items = exp.anomalous_features.map((af) =>
      el(
        'li',
        {},
        el('code', {}, af.feature),
        ` = ${formatNumber(af.value, 4)} `,
        el(
          'span',
          { class: 'predict-explain-range mono' },
          `(training [${formatNumber(af.training_range && af.training_range[0], 4)}, ` +
            `${formatNumber(af.training_range && af.training_range[1], 4)}])`,
        ),
        el('div', { class: 'predict-explain-note' }, af.note || ''),
      ),
    );
    blocks.push(
      el('div', { class: 'predict-explain-section-title' }, 'Аномальные параметры'),
      el('ul', { class: 'predict-explain-list' }, ...items),
    );
  }

  if (Array.isArray(exp.mechanism_concerns) && exp.mechanism_concerns.length > 0) {
    blocks.push(
      el(
        'div',
        { class: 'predict-explain-section-title' },
        'Возможные металлургические проблемы',
      ),
      el(
        'ul',
        { class: 'predict-explain-list' },
        ...exp.mechanism_concerns.map((c) => el('li', {}, c)),
      ),
    );
  }

  if (exp.production_risks) {
    blocks.push(
      el('div', { class: 'predict-explain-section-title' }, 'Производственные риски'),
      el('p', { class: 'predict-explain-paragraph' }, exp.production_risks),
    );
  }

  if (exp.suggested_correction) {
    blocks.push(
      el('div', { class: 'predict-explain-section-title' }, 'Рекомендуемая правка'),
      el('p', { class: 'predict-explain-paragraph' }, exp.suggested_correction),
    );
  }

  // Compact duration footer — same style as recipe / hypotheses panels.
  if (typeof result.duration_s === 'number') {
    blocks.push(
      el(
        'div',
        { class: 'predict-explain-footer mono' },
        `Длительность: ${formatNumber(result.duration_s, 1)} с`,
      ),
    );
  }

  return el('div', { class: 'predict-explain-card' }, ...blocks);
}

function renderResult() {
  if (!elements) return;
  const container = elements.resultContainer;
  container.replaceChildren();
  const data = state.lastResult;
  if (!data) return;

  const profile = activeProfile();
  const model = activeModel();
  const targetLabel =
    data.prediction.target_label || targetLabelFor(model, profile) || '';

  const mean = data.prediction.mean;
  const q05 = data.prediction.q05;
  const q95 = data.prediction.q95;
  const half = data.prediction.ci_half_width;

  const headline = el(
    'div',
    { class: 'predict-result-headline' },
    el(
      'div',
      { class: 'predict-result-label' },
      targetLabel ? `${targetLabel}` : 'Прогноз',
    ),
    el(
      'div',
      { class: 'predict-result-mean mono' },
      `${formatNumber(mean, 1)}`,
      el(
        'span',
        { class: 'predict-result-pm' },
        ` ± ${formatNumber(half, 1)}`,
      ),
    ),
    el(
      'div',
      { class: 'predict-result-ci mono' },
      `90% ДИ: [${formatNumber(q05, 1)}, ${formatNumber(q95, 1)}]`,
    ),
  );

  const blocks = [headline];

  // PR 12: show the OOD/wide-CI banner + anomaly-explain button when
  // the prediction is OOD OR the CI is wide vs the target's normal
  // range (Streamlit parity, app.py:793-803). We render the banner as
  // long as either trigger fires; the title/body wording adapts.
  const isOod = Boolean(data.ood && data.ood.is_ood);
  const isWideCi = Boolean(data.prediction && data.prediction.wide_ci);
  if (isOod || isWideCi) {
    blocks.push(buildAnomalyTriggerBlock({ isOod, isWideCi }));
  }

  // Derived HSLA features (cev_iiw / pcm / cen / microalloying_sum).
  const derived = data.derived || {};
  const derivedKeys = Object.keys(derived);
  if (derivedKeys.length > 0) {
    const derivedBlock = el(
      'div',
      { class: 'predict-derived' },
      el('div', { class: 'predict-derived-title' }, 'Производные параметры'),
      el(
        'div',
        { class: 'predict-derived-grid' },
        ...derivedKeys.map((k) =>
          el(
            'div',
            { class: 'predict-derived-cell' },
            el('span', { class: 'predict-derived-label' }, k),
            el('span', { class: 'predict-derived-value mono' }, formatNumber(derived[k], 4)),
          ),
        ),
      ),
    );
    blocks.push(derivedBlock);
  }

  container.append(el('div', { class: 'predict-result' }, ...blocks));
}

// -------------------- bootstrap --------------------

async function loadAll() {
  if (!elements) return;
  state.loading = true;
  clearError();
  try {
    const [modelsResp, classesResp, activeResp] = await Promise.all([
      apiFetch('/api/system/models'),
      apiFetch('/api/system/steel-classes'),
      apiFetch('/api/system/models/active').catch((err) => {
        // 404 is the legitimate "no models trained" path — surface as null.
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
    hideEmpty();

    state.selectedVersion = activeResp ? activeResp.version : state.models[state.models.length - 1].version;
    renderModelSelect();
    renderForm();
    renderResult();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить состояние: ${detail}`);
  } finally {
    state.loading = false;
  }
}

/** Router entry-point. Called once per session with the section[data-view]
 *  container. Replaces placeholder content with the real Predict view. */
export function init(container) {
  // Reset module state defensively in case init() runs more than once.
  state.models = [];
  state.classes = new Map();
  state.selectedVersion = null;
  state.loading = false;
  state.predicting = false;
  state.lastResult = null;
  state.explaining = false;
  state.lastExplain = null;
  state.explainError = null;
  if (state.explainAbort) {
    try { state.explainAbort.abort(); } catch (_e) { /* ignore */ }
    state.explainAbort = null;
  }

  const skeleton = buildSkeleton();
  elements = skeleton;

  container.replaceChildren(skeleton.root);

  loadAll();
}
