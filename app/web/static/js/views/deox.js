// Tab 04 — Раскисление. Three sync sub-tabs: Forward / Inverse / Compare.
//
// Streamlit parity reference: app/frontend/app.py lines 920-1219
// (with tab_deox: → sub_fwd / sub_inv / sub_cmp). The AI advisor sub-tab
// (sub_ai) is intentionally NOT implemented here — that lands in PR 9.
//
// Data flow:
//   GET  /api/deox/models                  → {items, default}
//   GET  /api/system/models/active         → active model meta (404 if none)
//   GET  /api/system/steel-classes         → {items} for target_o_a default
//   POST /api/deox/forward {AlDemandRequest}
//   POST /api/deox/inverse {AlQualityRequest}
//   POST /api/deox/compare {AlDemandRequest}
//
// PR 4 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';

// ──────────────────── module state ────────────────────

const SUBTABS = [
  { id: 'forward', label: 'Сколько Al нужно' },
  { id: 'inverse', label: 'Качество Al по факту' },
  { id: 'compare', label: 'Сравнить модели' },
  { id: 'ai',      label: 'AI советник + критик', disabled: true,
    title: 'Доступно после PR 9 миграции' },
];

const state = {
  thermoModels: [],         // [{id, name, citation, ...}]
  defaultModelId: null,     // id of registry default
  activeClassId: 'pipe_hsla',
  targetOaDefault: 5.0,     // from active class profile, fallback 5 ppm
  selectedModelId: null,    // user pick from dropdown
  subtab: 'forward',
  formValues: {
    forward: null,           // populated on first render
    inverse: null,
    compare: null,
  },
  results: {
    forward: null,
    inverse: null,
    compare: null,
  },
  busy: false,
};

let elements = null;

// ──────────────────── helpers ────────────────────

function formatNumber(value, decimals) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

function thermoModelById(id) {
  return state.thermoModels.find((m) => m.id === id) || null;
}

function defaultForward() {
  return {
    o_a_initial_ppm: 450,
    temperature_C: 1620,
    steel_mass_ton: 180,
    target_o_a_ppm: state.targetOaDefault,
    al_purity_pct: 100,
    burn_off_pct: 20,
    model_id: state.selectedModelId || state.defaultModelId,
  };
}

function defaultInverse() {
  return {
    o_a_before_ppm: 500,
    o_a_after_ppm: 10,
    al_added_kg: 65,
    temperature_C: 1620,
    steel_mass_ton: 180,
    burn_off_pct: 20,
    model_id: state.selectedModelId || state.defaultModelId,
  };
}

function defaultCompare() {
  return {
    o_a_initial_ppm: 450,
    temperature_C: 1620,
    steel_mass_ton: 180,
    target_o_a_ppm: state.targetOaDefault,
    al_purity_pct: 100,
    burn_off_pct: 20,
  };
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
        el('span', { class: 'here' }, 'Раскисление'),
      ),
      el('h1', { class: 'section-title' }, 'Раскисление жидкой стали алюминием'),
      el(
        'p',
        { class: 'section-sub' },
        'Physics-based калькулятор на базе трёх термодинамических моделей ' +
          '(Fruehan 1985, Sigworth-Elliott 1974, Hayashi-Yamamoto 2013). ' +
          'Forward — сколько Al подать. Inverse — эффективная чистота Al ' +
          'по факту. Compare — три формулы рядом для cross-validation.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'deox-error',
    role: 'alert',
    hidden: '',
  });

  // Context strip — active class + target O_a default + thermo model dropdown.
  const contextClass = el('div', { class: 'deox-context-cell' },
    el('span', { class: 'deox-context-label' }, 'Активный класс'),
    el('span', { class: 'deox-context-value', 'data-role': 'active-class' }, '—'),
  );
  const contextTarget = el('div', { class: 'deox-context-cell' },
    el('span', { class: 'deox-context-label' }, 'Target O_a из профиля'),
    el('span', { class: 'deox-context-value', 'data-role': 'target-default' }, '— ppm'),
  );
  const modelSelect = el('select', {
    class: 'deox-select',
    id: 'deox-model-select',
    onChange: (ev) => onModelChange(ev.target.value),
  });
  const contextModel = el('div', {
    class: 'deox-context-cell',
    style: { minWidth: '320px', flex: '1 1 320px' },
  },
    el('label', { class: 'deox-context-label', for: 'deox-model-select' },
      'Термодинамическая модель'),
    modelSelect,
  );
  const contextStrip = el('div', { class: 'deox-context' },
    contextClass, contextTarget, contextModel,
  );

  // Sub-tab nav.
  const subtabStrip = el('div', { class: 'deox-subtab-strip', role: 'tablist' });
  for (const tab of SUBTABS) {
    const btn = el('button', {
      class: 'deox-subtab',
      type: 'button',
      role: 'tab',
      'data-subtab': tab.id,
      ...(tab.disabled ? { disabled: '', title: tab.title } : {}),
      onClick: tab.disabled ? null : () => setSubtab(tab.id),
    }, tab.label);
    subtabStrip.append(btn);
  }

  // Per-subtab containers — only one visible at a time.
  const formContainer = el('div', { class: 'deox-form-panel', 'data-role': 'form' });
  const resultContainer = el('div', { 'data-role': 'result' });

  const body = el(
    'div',
    { class: 'deox-body' },
    errorBanner,
    contextStrip,
    subtabStrip,
    formContainer,
    resultContainer,
  );

  const root = el('div', { class: 'deox-view' }, head, body);

  return {
    root,
    errorBanner,
    contextClass: contextClass.querySelector('[data-role="active-class"]'),
    contextTarget: contextTarget.querySelector('[data-role="target-default"]'),
    modelSelect,
    subtabStrip,
    formContainer,
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

// ──────────────────── form rendering ────────────────────

function buildField(key, label, value, opts = {}) {
  const { step = 1, min, max, decimals = 2 } = opts;
  const id = `deox-field-${key}`;
  const input = el('input', {
    type: 'number',
    class: 'deox-input mono',
    id,
    value: String(Number(value).toFixed(decimals)),
    step: String(step),
    'data-field': key,
    ...(min != null ? { min: String(min) } : {}),
    ...(max != null ? { max: String(max) } : {}),
  });
  return el(
    'div',
    { class: 'deox-field' },
    el('label', { class: 'deox-field-label', for: id }, label),
    input,
  );
}

function readForm(formRoot, schema) {
  const out = {};
  for (const key of Object.keys(schema)) {
    if (key === 'model_id') {
      out[key] = state.selectedModelId || state.defaultModelId;
      continue;
    }
    const inp = formRoot.querySelector(`input[data-field="${key}"]`);
    if (!inp) {
      out[key] = schema[key];
      continue;
    }
    const val = parseFloat(inp.value);
    if (Number.isNaN(val)) {
      throw new Error(`Поле «${key}» содержит некорректное значение`);
    }
    out[key] = val;
  }
  return out;
}

function renderForwardForm() {
  const v = state.formValues.forward || defaultForward();
  state.formValues.forward = v;

  const grid = el(
    'div',
    { class: 'deox-form-grid' },
    buildField('o_a_initial_ppm', 'O_a измерено, ppm', v.o_a_initial_ppm,
      { step: 10, min: 0, max: 2000, decimals: 0 }),
    buildField('target_o_a_ppm', 'Целевой O_a, ppm', v.target_o_a_ppm,
      { step: 1, min: 0.5, max: 1000, decimals: 1 }),
    buildField('temperature_C', 'T расплава, °C', v.temperature_C,
      { step: 5, min: 1400, max: 1700, decimals: 0 }),
    buildField('steel_mass_ton', 'Масса стали, т', v.steel_mass_ton,
      { step: 5, min: 1, max: 500, decimals: 0 }),
    buildField('al_purity_pct', '% активного Al', v.al_purity_pct,
      { step: 1, min: 50, max: 100, decimals: 1 }),
    buildField('burn_off_pct', 'Угар, %', v.burn_off_pct,
      { step: 1, min: 0, max: 50, decimals: 1 }),
  );

  const heading = el('div', { class: 'deox-form-heading' },
    'Параметры плавки (Forward — сколько Al подать)');

  const submitBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    onClick: () => runForward(),
  }, 'Рассчитать');
  const actions = el('div', { class: 'deox-actions' }, submitBtn);

  elements.formContainer.replaceChildren(heading, grid, actions);
}

function renderInverseForm() {
  const v = state.formValues.inverse || defaultInverse();
  state.formValues.inverse = v;

  const grid = el(
    'div',
    { class: 'deox-form-grid' },
    buildField('o_a_before_ppm', 'O_a до, ppm', v.o_a_before_ppm,
      { step: 10, min: 0, max: 2000, decimals: 0 }),
    buildField('o_a_after_ppm', 'O_a после, ppm', v.o_a_after_ppm,
      { step: 1, min: 0, max: 2000, decimals: 1 }),
    buildField('al_added_kg', 'Al добавлено, кг', v.al_added_kg,
      { step: 1, min: 0.1, max: 5000, decimals: 1 }),
    buildField('temperature_C', 'T, °C', v.temperature_C,
      { step: 5, min: 1400, max: 1700, decimals: 0 }),
    buildField('steel_mass_ton', 'Масса стали, т', v.steel_mass_ton,
      { step: 5, min: 1, max: 500, decimals: 0 }),
    buildField('burn_off_pct', 'Угар (допущение), %', v.burn_off_pct,
      { step: 1, min: 0, max: 50, decimals: 1 }),
  );

  const heading = el('div', { class: 'deox-form-heading' },
    'Параметры плавки (Inverse — эффективная чистота Al по факту)');

  const submitBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    onClick: () => runInverse(),
  }, 'Оценить качество');
  const actions = el('div', { class: 'deox-actions' }, submitBtn);

  elements.formContainer.replaceChildren(heading, grid, actions);
}

function renderCompareForm() {
  const v = state.formValues.compare || defaultCompare();
  state.formValues.compare = v;

  const grid = el(
    'div',
    { class: 'deox-form-grid' },
    buildField('o_a_initial_ppm', 'O_a измерено, ppm', v.o_a_initial_ppm,
      { step: 10, min: 0, max: 2000, decimals: 0 }),
    buildField('target_o_a_ppm', 'Целевой O_a, ppm', v.target_o_a_ppm,
      { step: 1, min: 0.5, max: 1000, decimals: 1 }),
    buildField('temperature_C', 'T, °C', v.temperature_C,
      { step: 5, min: 1400, max: 1700, decimals: 0 }),
    buildField('steel_mass_ton', 'Масса, т', v.steel_mass_ton,
      { step: 5, min: 1, max: 500, decimals: 0 }),
    buildField('al_purity_pct', '% Al', v.al_purity_pct,
      { step: 1, min: 50, max: 100, decimals: 1 }),
    buildField('burn_off_pct', 'Угар, %', v.burn_off_pct,
      { step: 1, min: 0, max: 50, decimals: 1 }),
  );

  const heading = el('div', { class: 'deox-form-heading' },
    'Параметры плавки (Compare — три модели на одних входах)');

  const submitBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    onClick: () => runCompare(),
  }, 'Сравнить все 3 модели');
  const actions = el('div', { class: 'deox-actions' }, submitBtn);

  elements.formContainer.replaceChildren(heading, grid, actions);
}

// ──────────────────── result rendering ────────────────────

function renderWarnings(warnings) {
  if (!Array.isArray(warnings) || warnings.length === 0) return null;
  const items = warnings.map((w) => {
    const sevClass = (w.severity || '').toLowerCase();
    return el(
      'div',
      { class: `deox-warning ${sevClass}` },
      el('span', { class: 'deox-warning-id' },
        `[${w.severity || 'INFO'}] ${w.id || ''}:`),
      el('span', {}, ` ${w.message || ''}`),
      w.suggestion
        ? el('span', { class: 'deox-warning-suggestion' },
            `Рекомендация: ${w.suggestion}`)
        : null,
    );
  });
  return el('div', { class: 'deox-warnings' }, ...items);
}

function renderForwardResult() {
  const data = state.results.forward;
  if (!data) {
    elements.resultContainer.replaceChildren();
    return;
  }
  const r = data.result;
  const blocks = [];

  // Headline — Al kg + per ton.
  const headline = el(
    'div',
    { class: 'deox-result-headline' },
    el('div', { class: 'deox-result-label' }, 'Навеска Al'),
    el(
      'div',
      { class: 'deox-result-mean mono' },
      `${formatNumber(r.al_total_kg, 1)} кг`,
      el('span', { class: 'predict-result-pm' },
        ` (${formatNumber(r.al_per_ton, 3)} кг/т)`),
    ),
    el('div', { class: 'deox-result-sub' },
      `Модель: ${thermoModelById(r.model_id)?.name || r.model_id}`),
  );
  blocks.push(headline);

  const grid = el(
    'div',
    { class: 'deox-result-grid' },
    cell('Активный Al на реакцию', `${formatNumber(r.al_active_kg, 1)} кг`),
    cell('Угар', `${formatNumber(r.al_burn_off_kg, 1)} кг`),
    cell('Ожидаемый O_a', `${formatNumber(r.o_a_expected_ppm, 1)} ppm`),
    cell('Стоимость', `${formatNumber(r.cost_eur, 2)} ${r.currency || ''}`),
  );
  blocks.push(grid);

  const wBlock = renderWarnings(data.pattern_warnings);
  if (wBlock) blocks.push(wBlock);

  // Backend-side warnings (string list inside result.warnings) — physics
  // checks such as "T outside model range". Render as info bar.
  if (Array.isArray(r.warnings) && r.warnings.length > 0) {
    const items = r.warnings.map((msg) =>
      el('div', { class: 'deox-warning low' }, msg));
    blocks.push(el('div', { class: 'deox-warnings' }, ...items));
  }

  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result' }, ...blocks),
  );
}

function renderInverseResult() {
  const data = state.results.inverse;
  if (!data) {
    elements.resultContainer.replaceChildren();
    return;
  }
  const r = data.result;
  const blocks = [];

  const headline = el(
    'div',
    { class: 'deox-result-headline' },
    el('div', { class: 'deox-result-label' }, 'Эффективная активная чистота Al'),
    el(
      'div',
      { class: 'deox-result-mean mono' },
      `${formatNumber(r.effective_purity_pct, 1)} %`,
    ),
    el('div', { class: 'deox-result-sub' },
      `Модель: ${thermoModelById(r.model_id)?.name || r.model_id}`),
  );
  blocks.push(headline);

  const grid = el(
    'div',
    { class: 'deox-result-grid' },
    cell('Связал O', `${formatNumber(r.effective_active_kg, 1)} кг`),
    cell('Ожидался при 100% чистоте', `${formatNumber(r.expected_active_kg, 1)} кг`),
    cell('Допущение burn_off', `${formatNumber(r.assumed_burn_off_pct, 0)} %`),
    cell('Δ к 100%', `${formatNumber(r.effective_purity_pct - 100, 1)} %`),
  );
  blocks.push(grid);

  const wBlock = renderWarnings(data.pattern_warnings);
  if (wBlock) blocks.push(wBlock);

  if (Array.isArray(r.warnings) && r.warnings.length > 0) {
    const items = r.warnings.map((msg) =>
      el('div', { class: 'deox-warning low' }, msg));
    blocks.push(el('div', { class: 'deox-warnings' }, ...items));
  }

  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result' }, ...blocks),
  );
}

function renderCompareResult() {
  const data = state.results.compare;
  if (!data) {
    elements.resultContainer.replaceChildren();
    return;
  }
  const blocks = [];

  // Build rows from response.models — preserve registry order.
  const ids = state.thermoModels.map((m) => m.id);
  const rows = ids
    .map((id) => data.models?.[id])
    .filter((r) => r);
  if (rows.length === 0) {
    elements.resultContainer.replaceChildren(
      el('div', { class: 'deox-result' }, 'Нет результатов сравнения.'),
    );
    return;
  }
  const masses = rows.map((r) => r.al_total_kg);
  const maxMass = Math.max(...masses);
  const minMass = Math.min(...masses);

  const head = el('thead', {},
    el('tr', {},
      el('th', {}, 'Модель'),
      el('th', {}, 'Al, кг'),
      el('th', {}, 'Al, кг/т'),
      el('th', {}, 'O_a, ppm'),
      el('th', {}, 'Цена'),
    ),
  );
  const tbody = el('tbody', {},
    ...rows.map((r) => {
      const mClass = r.al_total_kg === maxMass ? 'is-max'
        : r.al_total_kg === minMass ? 'is-min' : '';
      return el('tr', {},
        el('td', { class: 'deox-compare-name' },
          thermoModelById(r.model_id)?.name || r.model_id),
        el('td', { class: mClass }, formatNumber(r.al_total_kg, 2)),
        el('td', {}, formatNumber(r.al_per_ton, 4)),
        el('td', {}, formatNumber(r.o_a_expected_ppm, 1)),
        el('td', {}, `${formatNumber(r.cost_eur, 2)} ${r.currency || ''}`),
      );
    }),
  );
  blocks.push(el('table', { class: 'deox-compare-table' }, head, tbody));

  const spread = data.spread_pct;
  blocks.push(el('div', { class: 'deox-spread-note' },
    `Разброс между моделями: ±${formatNumber(spread, 1)} %. ` +
    `Это ожидаемая неопределённость между академическими ` +
    `термодинамическими формулами; решение принимать по запасу.`));

  const wBlock = renderWarnings(data.pattern_warnings);
  if (wBlock) blocks.push(wBlock);

  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result' }, ...blocks),
  );
}

function cell(label, value) {
  return el(
    'div',
    { class: 'deox-result-cell' },
    el('span', { class: 'deox-result-cell-label' }, label),
    el('span', { class: 'deox-result-cell-value mono' }, value),
  );
}

// ──────────────────── actions ────────────────────

async function runForward() {
  if (state.busy) return;
  let payload;
  try {
    payload = readForm(elements.formContainer, defaultForward());
    state.formValues.forward = payload;
  } catch (err) {
    showError(err.message);
    return;
  }
  await dispatchPost('/api/deox/forward', payload, 'forward', renderForwardResult);
}

async function runInverse() {
  if (state.busy) return;
  let payload;
  try {
    payload = readForm(elements.formContainer, defaultInverse());
    state.formValues.inverse = payload;
  } catch (err) {
    showError(err.message);
    return;
  }
  await dispatchPost('/api/deox/inverse', payload, 'inverse', renderInverseResult);
}

async function runCompare() {
  if (state.busy) return;
  let payload;
  try {
    payload = readForm(elements.formContainer, defaultCompare());
    state.formValues.compare = payload;
  } catch (err) {
    showError(err.message);
    return;
  }
  await dispatchPost('/api/deox/compare', payload, 'compare', renderCompareResult);
}

async function dispatchPost(path, body, key, renderFn) {
  clearError();
  state.busy = true;
  // Show loading marker — replace result section with spinner-like text.
  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result' },
      el('div', { class: 'deox-result-sub' }, 'Считаю…')),
  );
  try {
    const data = await apiFetch(path, { method: 'POST', body });
    state.results[key] = data;
    renderFn();
  } catch (err) {
    state.results[key] = null;
    elements.resultContainer.replaceChildren();
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка расчёта: ${detail}`);
  } finally {
    state.busy = false;
  }
}

// ──────────────────── nav / model change ────────────────────

function setSubtab(id) {
  if (state.subtab === id) return;
  // Flush in-progress edits from the form being left so they survive the
  // re-render. Without this, typing 600 in Forward, switching to Compare
  // and back shows the stale 450 default — submit is the only path that
  // updates state.formValues today (see runForward/runInverse/runCompare).
  // We use the per-tab default schema as the field-key whitelist so a
  // stray ``model_id`` injected by the form doesn't bleed into compare.
  const prevTab = state.subtab;
  if (prevTab && prevTab !== id) {
    const schemaFn = prevTab === 'forward'
      ? defaultForward
      : prevTab === 'inverse'
        ? defaultInverse
        : prevTab === 'compare'
          ? defaultCompare
          : null;
    if (schemaFn && elements && elements.formContainer) {
      try {
        state.formValues[prevTab] = readForm(elements.formContainer, schemaFn());
      } catch {
        // readForm throws on NaN/missing inputs — keep the prior cached
        // values so a half-typed field doesn't wipe the whole form.
      }
    }
  }
  state.subtab = id;
  for (const btn of elements.subtabStrip.querySelectorAll('.deox-subtab')) {
    btn.classList.toggle('active', btn.dataset.subtab === id);
  }
  clearError();
  // Re-render form for the new tab.
  if (id === 'forward') renderForwardForm();
  else if (id === 'inverse') renderInverseForm();
  else if (id === 'compare') renderCompareForm();
  // Restore last result for that tab (or clear).
  if (id === 'forward') renderForwardResult();
  else if (id === 'inverse') renderInverseResult();
  else if (id === 'compare') renderCompareResult();
}

function onModelChange(modelId) {
  state.selectedModelId = modelId;
  // model_id only matters for forward+inverse; compare iterates registry.
  // Update each formValues record so re-render keeps the choice consistent.
  for (const key of ['forward', 'inverse']) {
    if (state.formValues[key]) state.formValues[key].model_id = modelId;
  }
}

// ──────────────────── bootstrap ────────────────────

function renderModelSelect() {
  if (!elements) return;
  const sel = elements.modelSelect;
  sel.replaceChildren();
  for (const m of state.thermoModels) {
    const opt = el('option',
      { value: m.id }, `${m.name} — ${m.citation}`);
    if (m.id === state.selectedModelId) opt.selected = true;
    sel.append(opt);
  }
}

async function loadAll() {
  if (!elements) return;
  clearError();
  try {
    const [thermoResp, classesResp, activeResp] = await Promise.all([
      apiFetch('/api/deox/models'),
      apiFetch('/api/system/steel-classes'),
      apiFetch('/api/system/models/active').catch((err) => {
        // 404 is the "no models trained yet" path — fall back to defaults.
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    ]);

    state.thermoModels = Array.isArray(thermoResp.items) ? thermoResp.items : [];
    state.defaultModelId =
      thermoResp.default ||
      (state.thermoModels[0] && state.thermoModels[0].id) ||
      null;
    state.selectedModelId = state.defaultModelId;

    // Resolve active class + target O_a default from active model meta.
    if (activeResp && activeResp.steel_class) {
      state.activeClassId = activeResp.steel_class;
    }
    const profile = (classesResp.items || []).find(
      (c) => c.id === state.activeClassId,
    );
    if (profile && profile.target_o_activity_ppm != null) {
      state.targetOaDefault = Number(profile.target_o_activity_ppm);
    }

    // Wire context display.
    elements.contextClass.textContent = state.activeClassId;
    elements.contextTarget.textContent = `${state.targetOaDefault} ppm`;
    renderModelSelect();

    // Activate Forward sub-tab by default.
    elements.subtabStrip.querySelector(
      `.deox-subtab[data-subtab="${state.subtab}"]`,
    )?.classList.add('active');
    if (state.subtab === 'forward') renderForwardForm();
    else if (state.subtab === 'inverse') renderInverseForm();
    else if (state.subtab === 'compare') renderCompareForm();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить состояние: ${detail}`);
  }
}

/** Router entry-point. */
export function init(container) {
  // Reset module state defensively on re-init.
  state.thermoModels = [];
  state.defaultModelId = null;
  state.activeClassId = 'pipe_hsla';
  state.targetOaDefault = 5.0;
  state.selectedModelId = null;
  state.subtab = 'forward';
  state.formValues = { forward: null, inverse: null, compare: null };
  state.results = { forward: null, inverse: null, compare: null };
  state.busy = false;

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadAll();
}
