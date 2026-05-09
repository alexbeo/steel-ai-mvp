// Tab 03 — Прогноз. Class-aware composition form + /api/predict.
//
// Streamlit parity reference: app/frontend/app.py lines 685-791.
// Data flow:
//   GET  /api/system/models                  → {items, count}
//   GET  /api/system/models/active           → model meta (404 if none)
//   GET  /api/system/steel-classes           → {items, count}
//   POST /api/predict {model_version, composition}
//                                            → {prediction, ood, derived, model}
//
// PR 3 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';

/** Module-scoped state. Re-init() resets it. */
const state = {
  models: [], // [{version, steel_class, target, ...}]
  classes: new Map(), // class_id -> profile
  selectedVersion: null,
  loading: false,
  predicting: false,
  lastResult: null,
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

  // OOD warning + placeholder for PR 12 anomaly explainer.
  if (data.ood && data.ood.is_ood) {
    const oodBlock = el(
      'div',
      { class: 'predict-ood-warning' },
      el(
        'div',
        { class: 'predict-ood-title' },
        'Внимание: состав вне training distribution',
      ),
      el(
        'div',
        { class: 'predict-ood-body' },
        'Прогноз ненадёжен — точка лежит за пределами области, где модель училась. ' +
          'Рекомендуется скорректировать состав или собрать дополнительные данные.',
      ),
      el(
        'button',
        {
          class: 'btn',
          type: 'button',
          disabled: '',
          title: 'Доступно после PR 12 миграции',
        },
        'Объяснить почему рискованно (доступно после PR 12)',
      ),
    );
    blocks.push(oodBlock);
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

  const skeleton = buildSkeleton();
  elements = skeleton;

  container.replaceChildren(skeleton.root);

  loadAll();
}
