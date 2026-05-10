// Tab 04 — Раскисление. Four sub-tabs: Forward / Inverse / Compare / AI.
//
// The first three sub-tabs are sync calculator forms; the AI sub-tab
// (PR 9) is the long-running PhD advisor + critic cycle (~3 min) over
// the JobStore.
//
// Data flow:
//   GET  /api/deox/models                  → {items, default}
//   GET  /api/system/models/active         → active model meta (404 if none)
//   GET  /api/system/steel-classes         → {items} for target_o_a default
//   POST /api/deox/forward {AlDemandRequest}
//   POST /api/deox/inverse {AlQualityRequest}
//   POST /api/deox/compare {AlDemandRequest}
//   POST /api/deox/ai-cycle {AlAdvisoryRequest}  → {job_id} (PR 9)
//   GET  /api/jobs/{id}                    → poll progress + result
//   DELETE /api/jobs/{id}                  → cooperative cancel
//
// PR 4 + PR 9 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { el } from '../utils/dom.js';

// ──────────────────── module state ────────────────────

const SUBTABS = [
  { id: 'forward', label: 'Сколько Al нужно' },
  { id: 'inverse', label: 'Качество Al по факту' },
  { id: 'compare', label: 'Сравнить модели' },
  { id: 'ai',      label: 'AI советник + критик' },
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
    ai: null,                // AI cycle form snapshot (PR 9)
  },
  results: {
    forward: null,
    inverse: null,
    compare: null,
    ai: null,                // last AI cycle result {advisor, critic, ...}
  },
  busy: false,
  // PR 9 — AI cycle is long-running; track the in-flight job so the
  // user can cancel and so the form stays disabled until completion.
  aiJob: {
    running: false,
    pollAbort: null,
    currentJobId: null,
  },
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

function defaultAi() {
  // Composition defaults to a generic mid-range HSLA-ish heat — the
  // LLM is instructed to use the values as soft hints, not constraints.
  // ``operator_notes`` and ``heat_id`` start empty.
  return {
    o_a_initial_ppm: 280,
    target_o_a_ppm: state.targetOaDefault,
    temperature_C: 1580,
    steel_mass_ton: 100,
    al_purity_pct: 99.7,
    burn_off_pct: 20,
    composition: {
      c_pct: 0.20,
      mn_pct: 0.85,
      si_pct: 0.30,
      s_pct: 0.012,
      p_pct: 0.018,
    },
    slag_feo_pct: 2.5,
    grade_target: 'строительная конструкционная',
    heat_id: '',
    operator_notes: '',
    save_to_decision_log: false,
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

// ──────────────────── AI sub-tab (PR 9) — form ────────────────────

function buildCompField(key, label, value, opts) {
  // Wrapper around buildField that namespaces the data-field with
  // ``comp:`` so readAiForm can demux composition from heat fields.
  const id = `deox-field-comp-${key}`;
  const o = { ...opts };
  const input = el('input', {
    type: 'number',
    class: 'deox-input mono',
    id,
    value: value == null ? '' : String(Number(value).toFixed(o.decimals ?? 2)),
    step: String(o.step ?? 0.01),
    'data-field': `comp:${key}`,
    ...(o.min != null ? { min: String(o.min) } : {}),
    ...(o.max != null ? { max: String(o.max) } : {}),
  });
  return el(
    'div',
    { class: 'deox-field' },
    el('label', { class: 'deox-field-label', for: id }, label),
    input,
  );
}

function renderAiForm() {
  const v = state.formValues.ai || defaultAi();
  state.formValues.ai = v;

  const heatHeading = el('div', { class: 'deox-form-heading' },
    'Параметры плавки');
  const heatGrid = el(
    'div',
    { class: 'deox-form-grid' },
    buildField('o_a_initial_ppm', 'O_a измеренный, ppm', v.o_a_initial_ppm,
      { step: 10, min: 0, max: 2000, decimals: 0 }),
    buildField('target_o_a_ppm', 'Целевой O_a, ppm', v.target_o_a_ppm,
      { step: 0.5, min: 0.5, max: 50, decimals: 1 }),
    buildField('temperature_C', 'T расплава, °C', v.temperature_C,
      { step: 5, min: 1400, max: 1700, decimals: 0 }),
    buildField('steel_mass_ton', 'Масса стали, т', v.steel_mass_ton,
      { step: 5, min: 1, max: 500, decimals: 0 }),
    buildField('al_purity_pct', 'Чистота Al, %', v.al_purity_pct,
      { step: 0.1, min: 50, max: 100, decimals: 1 }),
    buildField('burn_off_pct', 'Угар, %', v.burn_off_pct,
      { step: 1, min: 0, max: 50, decimals: 1 }),
  );

  const compHeading = el('div', { class: 'deox-form-heading' },
    'Композиция (опционально, помогает критику ловить риски)');
  const compGrid = el(
    'div',
    { class: 'deox-form-grid' },
    buildCompField('c_pct',  'C, wt%',  v.composition?.c_pct,
      { step: 0.01, min: 0, max: 1.5,  decimals: 2 }),
    buildCompField('mn_pct', 'Mn, wt%', v.composition?.mn_pct,
      { step: 0.05, min: 0, max: 3.0,  decimals: 2 }),
    buildCompField('si_pct', 'Si, wt%', v.composition?.si_pct,
      { step: 0.05, min: 0, max: 2.5,  decimals: 2 }),
    buildCompField('s_pct',  'S, wt%',  v.composition?.s_pct,
      { step: 0.002, min: 0, max: 0.05, decimals: 3 }),
    buildCompField('p_pct',  'P, wt%',  v.composition?.p_pct,
      { step: 0.002, min: 0, max: 0.05, decimals: 3 }),
    buildField('slag_feo_pct', 'Slag FeO, %', v.slag_feo_pct,
      { step: 0.5, min: 0, max: 15, decimals: 1 }),
  );

  // Free-text grade + operator notes + heat id + save toggle.
  const gradeLabel = el('label',
    { class: 'deox-field-label', for: 'deox-ai-grade' },
    'Целевой grade / задача');
  const gradeInput = el('input', {
    type: 'text',
    class: 'deox-input',
    id: 'deox-ai-grade',
    value: v.grade_target || '',
    'data-field': 'grade_target',
    maxlength: '200',
  });
  const heatIdLabel = el('label',
    { class: 'deox-field-label', for: 'deox-ai-heat-id' },
    'Heat ID (опционально)');
  const heatIdInput = el('input', {
    type: 'text',
    class: 'deox-input',
    id: 'deox-ai-heat-id',
    value: v.heat_id || '',
    'data-field': 'heat_id',
    maxlength: '50',
    placeholder: 'например, 25-04-117',
  });
  const notesLabel = el('label',
    { class: 'deox-field-label', for: 'deox-ai-notes' },
    'Заметки оператора (опционально)');
  const notesInput = el('textarea', {
    class: 'deox-input',
    id: 'deox-ai-notes',
    rows: '3',
    'data-field': 'operator_notes',
    maxlength: '2000',
    placeholder: 'Например: scrap heavy, slag covered, prev heat had high N…',
  }, v.operator_notes || '');
  const grid2 = el(
    'div',
    { class: 'deox-form-grid' },
    el('div', { class: 'deox-field' }, gradeLabel, gradeInput),
    el('div', { class: 'deox-field' }, heatIdLabel, heatIdInput),
    el(
      'div',
      { class: 'deox-field deox-field-wide' },
      notesLabel, notesInput,
    ),
  );

  const saveLabel = el('label',
    { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      'data-field': 'save_to_decision_log',
      ...(v.save_to_decision_log ? { checked: 'checked' } : {}),
    }),
    el('span', {}, 'Сохранить в Decision Log (с тэгом deoxidation_cycle)'),
  );

  const submitBtn = el('button', {
    class: 'btn primary deox-ai-run-btn',
    type: 'button',
    onClick: () => runAiCycle(),
  }, 'Запросить AI совет');
  if (state.aiJob.running) submitBtn.disabled = true;

  const subnote = el('div', { class: 'deox-ai-subnote' },
    'Полный цикл (PhD-советник + adversarial peer-review) занимает ' +
    '~3 минуты, расходует ~$0.20-0.25 на Sonnet API. ' +
    'LLM call нельзя прервать после старта — кнопку «Отменить» лучше ' +
    'нажимать сразу.');

  const actions = el('div', { class: 'deox-actions deox-ai-actions' },
    saveLabel, submitBtn);

  // Mount point for renderJobProgress (filled in runAiCycle).
  const progressMount = el('div', {
    class: 'deox-ai-progress-mount', 'data-role': 'ai-progress-mount',
  });

  elements.formContainer.replaceChildren(
    heatHeading, heatGrid,
    compHeading, compGrid,
    grid2,
    actions, subnote,
    progressMount,
  );
}

function readAiForm(formRoot) {
  // Composition fields use ``comp:<key>``; numeric heat fields use the
  // bare key. Free-text fields (grade_target, heat_id, operator_notes)
  // also live as ``data-field=<key>``. Empty number fields are normalised
  // to undefined so the API treats them as "not provided".
  const out = {
    composition: {},
  };
  for (const inp of formRoot.querySelectorAll('input[data-field], textarea[data-field]')) {
    const key = inp.dataset.field;
    if (!key) continue;

    if (key === 'save_to_decision_log') {
      out.save_to_decision_log = !!inp.checked;
      continue;
    }
    if (key === 'grade_target' || key === 'heat_id' || key === 'operator_notes') {
      out[key] = (inp.value || '').trim();
      continue;
    }

    if (key.startsWith('comp:')) {
      const compKey = key.slice('comp:'.length);
      const raw = (inp.value || '').trim();
      if (raw === '') continue; // optional — drop
      const val = parseFloat(raw);
      if (Number.isNaN(val)) {
        throw new Error(`Поле «${compKey}» содержит некорректное значение`);
      }
      out.composition[compKey] = val;
      continue;
    }

    // Generic numeric heat field.
    const raw = (inp.value || '').trim();
    if (raw === '' && key === 'slag_feo_pct') {
      // Optional — drop so server treats as null.
      continue;
    }
    const val = parseFloat(raw);
    if (Number.isNaN(val)) {
      throw new Error(`Поле «${key}» содержит некорректное значение`);
    }
    out[key] = val;
  }

  // Drop empty heat_id / operator_notes so the request body stays tidy.
  if (out.heat_id === '') delete out.heat_id;
  if (out.operator_notes === '') delete out.operator_notes;
  // Coerce missing booleans (no toggle on form? unlikely) to false.
  if (typeof out.save_to_decision_log !== 'boolean') {
    out.save_to_decision_log = false;
  }

  return out;
}

// ──────────────────── AI sub-tab (PR 9) — result ────────────────────

function verdictLabel(verdict) {
  return ({
    ACCEPT: 'ПРИНЯТО',
    REVISE: 'ТРЕБУЕТ ПРАВОК',
    REJECT: 'ОТКЛОНЕНО',
  })[verdict] || verdict || '—';
}

function verdictModifier(verdict) {
  return ({
    ACCEPT: 'accept',
    REVISE: 'revise',
    REJECT: 'reject',
  })[verdict] || 'unknown';
}

function evidenceMark(verdict) {
  return ({
    VALID: '✓',
    INVALID: '✗',
    UNVERIFIABLE: '?',
  })[verdict] || '•';
}

function renderAiAdvisorCard(advisor) {
  const blocks = [];

  blocks.push(el('div', { class: 'deox-ai-card-title' },
    '🧪 Operator protocol'));
  if (advisor.summary) {
    blocks.push(el('div', { class: 'deox-ai-summary' },
      el('strong', {}, 'Резюме. '), advisor.summary));
  }

  // Headline metrics row.
  const metrics = el(
    'div',
    { class: 'deox-result-grid' },
    cell('Al total, кг', formatNumber(advisor.al_addition_kg, 1)),
    cell('Форма', advisor.al_form || '—'),
    cell('Recovery, %', formatNumber(advisor.expected_recovery_pct, 0)),
    cell(
      'Время до target',
      Array.isArray(advisor.kinetic_timing_min) && advisor.kinetic_timing_min.length === 2
        ? `${formatNumber(advisor.kinetic_timing_min[0], 0)}–${formatNumber(advisor.kinetic_timing_min[1], 0)} мин`
        : '—',
    ),
  );
  blocks.push(metrics);

  if (advisor.addition_strategy) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Стратегия подачи. '), advisor.addition_strategy));
  }
  if (advisor.model_convergence_note) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Сходимость 3 thermo-моделей. '),
      advisor.model_convergence_note));
  }

  if (Array.isArray(advisor.risk_flags) && advisor.risk_flags.length > 0) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, '⚠️ Риски этой плавки:'),
      el('ul', { class: 'deox-ai-list' },
        ...advisor.risk_flags.map((r) => el('li', {}, r)),
      ),
    ));
  }

  if (advisor.inclusion_forecast) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Прогноз включений. '), advisor.inclusion_forecast));
  }

  const pre = Array.isArray(advisor.pre_actions) ? advisor.pre_actions : [];
  const post = Array.isArray(advisor.post_actions) ? advisor.post_actions : [];
  if (pre.length || post.length) {
    blocks.push(el(
      'div',
      { class: 'deox-ai-twocol' },
      el('div', {},
        el('strong', {}, 'До добавки Al:'),
        el('ul', { class: 'deox-ai-list' },
          ...pre.map((a) => el('li', {}, a)),
        ),
      ),
      el('div', {},
        el('strong', {}, 'После добавки Al:'),
        el('ul', { class: 'deox-ai-list' },
          ...post.map((a) => el('li', {}, a)),
        ),
      ),
    ));
  }

  if (Array.isArray(advisor.evidence) && advisor.evidence.length > 0) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Доказательная база:'),
      el('ul', { class: 'deox-ai-list' },
        ...advisor.evidence.map((e) => el('li', {}, e)),
      ),
    ));
  }

  blocks.push(el('div', { class: 'deox-ai-meta' },
    `Уверенность советника: ${advisor.confidence || '—'}`,
    advisor.id ? `  ·  id=${advisor.id}` : '',
  ));

  return el('div', { class: 'deox-ai-card deox-ai-advisor' }, ...blocks);
}

function renderAiCriticCard(critic) {
  if (!critic) {
    return el('div', { class: 'deox-ai-card deox-ai-critic deox-ai-critic-missing' },
      el('div', { class: 'deox-ai-card-title' }, '👨‍🔬 PhD-критик'),
      el('div', { class: 'deox-ai-block' },
        'Критик вернул None — Sonnet API call failed или выдал malformed payload.'),
    );
  }

  const blocks = [];
  const verdictBadge = el(
    'span',
    {
      class: `deox-ai-verdict-badge deox-ai-verdict-${verdictModifier(critic.verdict)}`,
    },
    `👨‍🔬 PhD-критик: ${verdictLabel(critic.verdict)}`,
  );
  const confidence = el('span', { class: 'deox-ai-verdict-confidence' },
    `уверенность ${critic.confidence || '—'}`);
  blocks.push(el('div', { class: 'deox-ai-verdict-row' },
    verdictBadge, confidence));

  if (critic.summary) {
    blocks.push(el('div', { class: 'deox-ai-summary' },
      el('em', {}, critic.summary)));
  }

  if (Array.isArray(critic.evidence_check) && critic.evidence_check.length > 0) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Fact-check доказательной базы:'),
      el('ul', { class: 'deox-ai-list' },
        ...critic.evidence_check.map((ec) => el('li', {},
          `${evidenceMark(ec.verdict)} `,
          el('strong', {}, `${ec.claim || ''}`),
          ' — ',
          ec.note || '',
        )),
      ),
    ));
  }

  const strengths = Array.isArray(critic.strengths) ? critic.strengths : [];
  const weaknesses = Array.isArray(critic.weaknesses) ? critic.weaknesses : [];
  if (strengths.length || weaknesses.length) {
    blocks.push(el(
      'div',
      { class: 'deox-ai-twocol' },
      el('div', {},
        el('strong', {}, 'Сильные стороны'),
        el('ul', { class: 'deox-ai-list' },
          ...strengths.map((s) => el('li', {}, s)),
        ),
      ),
      el('div', {},
        el('strong', {}, 'Слабые стороны'),
        el('ul', { class: 'deox-ai-list' },
          ...weaknesses.map((w) => el('li', {}, w)),
        ),
      ),
    ));
  }

  if (critic.suggested_revision) {
    blocks.push(el('div', { class: 'deox-ai-suggested-revision' },
      el('strong', {}, 'Предложение правки. '), critic.suggested_revision));
  }

  return el('div', { class: 'deox-ai-card deox-ai-critic' }, ...blocks);
}

function renderAiResult() {
  const data = state.results.ai;
  if (!data) {
    elements.resultContainer.replaceChildren();
    return;
  }

  const blocks = [];
  if (data.advisor) blocks.push(renderAiAdvisorCard(data.advisor));
  blocks.push(renderAiCriticCard(data.critic));

  // Pattern Library warnings (DX01/DX02 from sync calc).
  const wBlock = renderWarnings(data.pattern_warnings);
  if (wBlock) blocks.push(wBlock);

  // Decision Log saved indicator (opt-in flow).
  if (data.decision_log_id != null) {
    blocks.push(el('div', { class: 'deox-ai-decision-saved' },
      `✔ Сохранено в Decision Log (id=${data.decision_log_id}, тэг deoxidation_cycle).`));
  }

  // Cycle metadata strip.
  const meta = [];
  if (typeof data.duration_s === 'number') {
    meta.push(`длительность: ${formatNumber(data.duration_s, 1)} с`);
  }
  if (data.thermo_estimates) {
    const parts = Object.entries(data.thermo_estimates)
      .map(([k, v]) => `${k}=${formatNumber(v, 1)} кг`);
    if (parts.length) meta.push(`thermo: ${parts.join(', ')}`);
  }
  if (meta.length) {
    blocks.push(el('div', { class: 'deox-ai-meta' }, meta.join('  ·  ')));
  }

  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result deox-ai-result' }, ...blocks),
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

async function runAiCycle() {
  if (state.aiJob.running) return;
  clearError();

  let body;
  try {
    body = readAiForm(elements.formContainer);
    state.formValues.ai = {
      ...state.formValues.ai,
      ...body,
    };
  } catch (err) {
    showError(err.message);
    return;
  }

  // Add the thermo_model selected at the top of the view — body comes
  // from the form which doesn't include it. The backend defaults if
  // absent, but we send it explicitly for clarity.
  body.thermo_model = state.selectedModelId || state.defaultModelId;

  // Mark busy + freeze the run button. Result panel is cleared so a
  // stale prior result doesn't render alongside the new progress card.
  state.aiJob.running = true;
  state.results.ai = null;
  elements.resultContainer.replaceChildren();
  const runBtn = elements.formContainer.querySelector('.deox-ai-run-btn');
  if (runBtn) runBtn.disabled = true;
  const progressMount = elements.formContainer.querySelector(
    '[data-role="ai-progress-mount"]',
  );
  if (progressMount) progressMount.replaceChildren();

  const abort = new AbortController();
  state.aiJob.pollAbort = abort;

  const progress = renderJobProgress(progressMount || elements.formContainer, {
    label: 'Запрашиваю AI рецензию… (~3 мин)',
    onCancel: () => cancelAiCycle(),
  });

  let jobId = null;
  try {
    const submit = await apiFetch('/api/deox/ai-cycle', { method: 'POST', body });
    jobId = submit.job_id;
    state.aiJob.currentJobId = jobId;
    if (!jobId) throw new Error('Сервер не вернул job_id');
    progress.updateMessage(`job_id=${jobId.slice(0, 8)}…`);

    const result = await pollJob(jobId, {
      interval: 2000,
      signal: abort.signal,
      onProgress: (p) => progress.updateProgress(p),
      onMessage: (m) => progress.updateMessage(m),
    });
    progress.markDone('Готово');
    state.results.ai = result;
    renderAiResult();
  } catch (err) {
    if (err && err.name === 'AbortError') {
      progress.markError('Отменено пользователем');
    } else if (err instanceof ApiError && err.status === 503) {
      // Distinct banner for "AI not configured" — actionable copy.
      progress.markError('AI advisor не настроен');
      showError(
        `AI advisor не настроен (нужен ANTHROPIC_API_KEY и prompts/). ${err.message}`,
      );
    } else {
      const detail = err instanceof ApiError ? err.message : String(err.message || err);
      progress.markError(detail);
      showError(`Ошибка AI цикла: ${detail}`);
    }
  } finally {
    state.aiJob.running = false;
    state.aiJob.pollAbort = null;
    state.aiJob.currentJobId = null;
    if (runBtn) runBtn.disabled = false;
  }
}

async function cancelAiCycle() {
  // Best-effort: abort the polling loop on the client immediately, then
  // DELETE on the server. Cancellation is effective ONLY before the LLM
  // call (the worker raises "Cancelled by user (before advisor LLM
  // call)" then). Once Sonnet messages.create is in flight, the server
  // burns through to completion — see deox.py module docstring.
  const jobId = state.aiJob.currentJobId;
  if (state.aiJob.pollAbort) state.aiJob.pollAbort.abort();
  if (!jobId) return;
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  } catch (err) {
    console.warn('[deox-ai] DELETE /api/jobs failed', err);
  }
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
  // Block switching away from a running AI cycle — the form holds the
  // progress card and DOM continuity matters for the cancel button.
  if (state.aiJob.running) {
    showError(
      'AI цикл запущен — отмените его перед переключением на другую вкладку.',
    );
    return;
  }
  // Flush in-progress edits from the form being left so they survive the
  // re-render. Without this, typing 600 in Forward, switching to Compare
  // and back shows the stale 450 default — submit is the only path that
  // updates state.formValues today (see runForward/runInverse/runCompare).
  // We use the per-tab default schema as the field-key whitelist so a
  // stray ``model_id`` injected by the form doesn't bleed into compare.
  const prevTab = state.subtab;
  if (prevTab && prevTab !== id) {
    if (prevTab === 'ai') {
      try {
        state.formValues.ai = {
          ...state.formValues.ai,
          ...readAiForm(elements.formContainer),
        };
      } catch {
        // Half-typed — keep the cached form values intact.
      }
    } else {
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
  else if (id === 'ai') renderAiForm();
  // Restore last result for that tab (or clear).
  if (id === 'forward') renderForwardResult();
  else if (id === 'inverse') renderInverseResult();
  else if (id === 'compare') renderCompareResult();
  else if (id === 'ai') renderAiResult();
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
    else if (state.subtab === 'ai') renderAiForm();
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
  state.formValues = { forward: null, inverse: null, compare: null, ai: null };
  state.results = { forward: null, inverse: null, compare: null, ai: null };
  state.busy = false;
  state.aiJob = { running: false, pollAbort: null, currentJobId: null };

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadAll();
}
