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
import {
  DEOX_CRITIC_LABELS,
  EVIDENCE_CHECK_LABELS,
  splitKeyedLine,
  labelFor,
} from '../utils/llm_labels.js';

// ──────────────────── module state ────────────────────

const SUBTABS = [
  { id: 'forward',  label: 'Сколько Al нужно' },
  { id: 'inverse',  label: 'Качество Al по факту' },
  { id: 'compare',  label: 'Сравнить модели' },
  { id: 'ai',       label: 'AI советник + критик' },
  { id: 'optimize', label: '🎯 Оптимизация метода' },
  { id: 'eta_calib', label: '🎯 Калибровка η_Al' },
  { id: 'history',  label: '📋 История плавок' },
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
    optimize: null,          // PR 7 — slag-aware optimize form snapshot
  },
  results: {
    forward: null,
    inverse: null,
    compare: null,
    ai: null,                // last AI cycle result {advisor, critic, ...}
    optimize: null,          // PR 7 — OptimizationResponse payload
  },
  busy: false,
  // PR 9 — AI cycle is long-running; track the in-flight job so the
  // user can cancel and so the form stays disabled until completion.
  aiJob: {
    running: false,
    pollAbort: null,
    currentJobId: null,
  },
  // PR 7 — Slag-aware optimization sub-tab. Methods catalog is lazy-loaded
  // on first activation of the tab (GET /api/deox/methods). `save` tracks
  // the in-flight POST /api/deox/optimize/save so the «Сохранить» button
  // can show its own spinner without colliding with the main run flow.
  optimize: {
    methods: [],            // [{id, name, eta_al_typical, ...}]
    defaultMethodId: null,
    methodsLoaded: false,
    methodsLoading: false,  // PR 8 — concurrent-fetch dedup
    saving: false,
  },
  // PR 2 (ASIS-deox calibration) — heats history sub-tab. Reads
  // /api/deox/heats* CRUD. Filters drive the GET list; ``formOpen`` is
  // the «➕ Новая плавка» disclosure; ``editingId`` is the inline
  // outcome-PATCH row (null = none open). Methods catalog is shared
  // with the optimize sub-tab — reloaded if not already cached.
  history: {
    loaded: false,
    loading: false,
    error: null,
    items: [],
    total: 0,
    nextBeforeId: null,
    filters: { plant_id: '', method_id: '', has_outcome: '' },
    plants: [],             // [{plant_id, count}]
    formOpen: false,
    formValues: null,       // lazy-init via defaultHeatForm()
    creating: false,
    editingId: null,
    editValues: {
      o_a_after_ppm: '',
      al_residual_pct: '',
      eta_al_effective: '',
      quality_flag: '',
    },
    patching: false,
  },
  // PR 10 — «🎯 Калибровка η_Al» sub-tab. Lazy-loads the trained ML model
  // status (GET /api/deox/eta-al-model/status) + the plant×method posteriors
  // (GET /api/deox/calibrations) in parallel on first activation. ``running``
  // tracks the in-flight POST /api/deox/calibrations/run so the «Запустить
  // калибровку» button can show a spinner.
  etaCalib: {
    loaded: false,
    loading: false,
    error: null,
    modelStatus: null,    // {model_present, r2_test, coverage_90_ci, ...}
    calibrations: [],     // [{plant_id, method_id, ...}]
    running: false,
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

function defaultOptimize() {
  // Excel base-case (см. spec §6 + test_api_deox._baseline_optimize_payload):
  // 371-т BOF плавка, 657 ppm O_a после tap'а, target 8 ppm, residual
  // [Al]=0.018 %, 2.2 т carry-over шлака с FeO=18 %, T=1600 °C.
  // Эти дефолты позволяют пользователю нажать «Найти оптимальный метод»
  // сразу после открытия таба и увидеть, что выбирается asis_shot.
  return {
    // Block A — heat
    steel_mass_ton: 371,
    o_a_initial_ppm: 657,
    temperature_C: 1600,
    target_o_a_ppm: 8,
    target_al_pct: 0.018,
    // Block B — slag carry-over (optional)
    slag_no_data: false,
    slag_mass_kg: 2200,
    slag_feo_pct: 18,
    slag_mno_pct: 0,
    slag_sio2_pct: 0,
    // Block C — co-deox FeSi (collapsed)
    co_deox_enabled: false,
    co_deox_fesi_kg: 0,
    co_deox_fesi_si_content_pct: 75,
    // Block D — methods (multi-select; null/empty = «все методы»)
    method_ids: [],          // empty array → backend treats as "all"
    user_override_eta_al_enabled: false,
    user_override_eta_al: 0.80,
    // Block E — constraints
    target_n_ppm: null,
    premium_cap_eur_per_kg: null,
    t_drying_c: null,
    // Block F — economics + thermo
    thermo_model_id: null,    // resolved from selectedModelId at submit time
    al_commodity_price_eur_per_kg: 2.40,
    // Block G — multi-objective (PR 7). Backwards compat: 'cost' reproduces
    // pre-PR-7 behavior. 'al_mass' picks min Al pure (carbon footprint).
    // 'pareto' returns the non-dominated frontier with knee chosen.
    objective: 'cost',
    // Block H — η_Al prediction (PR 10). Default off → PR 7 behavior
    // byte-identical. When on, requires plant_id.
    enable_eta_prediction: false,
    plant_id: '',
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
  elements.errorBanner.classList.remove('deox-info');
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

/**
 * Surface a non-error message in the same banner slot — green / info
 * styling instead of red. Used by the «Сохранить рекомендацию» success
 * path so we don't display a successful save as if it were an error.
 *
 * The banner toggles its CSS class via ``deox-info`` (overrides.css);
 * ``showError`` strips that class so the next failure paints red again.
 */
function showInfo(message) {
  if (!elements) return;
  elements.errorBanner.classList.add('deox-info');
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  if (!elements) return;
  elements.errorBanner.classList.remove('deox-info');
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
          ' ',
          el('span', { class: 'deox-ai-evidence-tag' },
            `(${labelFor(ec.verdict, EVIDENCE_CHECK_LABELS)})`),
          ec.note ? ` — ${ec.note}` : '',
        )),
      ),
    ));
  }

  const strengths = Array.isArray(critic.strengths) ? critic.strengths : [];
  const weaknesses = Array.isArray(critic.weaknesses) ? critic.weaknesses : [];
  if (strengths.length || weaknesses.length) {
    // Critic prefixует строки английскими attack_vector ID (см.
    // prompts/deoxidation_critic.md adversarial_mindset[].id); парсим
    // и переводим в русский label.
    blocks.push(el(
      'div',
      { class: 'deox-ai-twocol' },
      el('div', {},
        el('strong', {}, 'Сильные стороны'),
        el('ul', { class: 'deox-ai-list' },
          ...strengths.map((s) =>
            el('li', {}, splitKeyedLine(s, DEOX_CRITIC_LABELS))),
        ),
      ),
      el('div', {},
        el('strong', {}, 'Слабые стороны'),
        el('ul', { class: 'deox-ai-list' },
          ...weaknesses.map((w) =>
            el('li', {}, splitKeyedLine(w, DEOX_CRITIC_LABELS))),
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

// ──────────────────── Optimize sub-tab (PR 7) ────────────────────
//
// Пятый sub-tab «🎯 Оптимизация метода». Покрывает design-doc §8 — форма
// блоков A-F (heat / slag / co-deox / methods / constraints / economics) +
// recommendation card + pareto table + pattern warnings. POST на
// /api/deox/optimize возвращает payload, описанный в test_api_deox.py
// (chosen_method_id, pareto_table sorted ascending, etc.). Save кнопка
// сейчас всегда даёт 501 — PR 8 закроет Decision Log.

function buildOptionalField(key, label, value, opts = {}) {
  // Похож на buildField, но позволяет пустое значение (data-optional=true) —
  // readOptimizeForm дропает поле если raw=''. Используется для блока E
  // (target_n_ppm, premium_cap, t_drying_c) и Block B компонентов %MnO/%SiO₂.
  const { step = 1, min, max, decimals = 2, placeholder } = opts;
  const id = `deox-opt-field-${key}`;
  const inputAttrs = {
    type: 'number',
    class: 'deox-input mono',
    id,
    value: value == null || value === '' ? '' : String(Number(value).toFixed(decimals)),
    step: String(step),
    'data-field': key,
    'data-optional': 'true',
    ...(min != null ? { min: String(min) } : {}),
    ...(max != null ? { max: String(max) } : {}),
    ...(placeholder ? { placeholder } : {}),
  };
  const input = el('input', inputAttrs);
  return el(
    'div',
    { class: 'deox-field' },
    el('label', { class: 'deox-field-label', for: id }, label),
    input,
  );
}

function buildOptimizeField(key, label, value, opts = {}) {
  // Те же поля что buildField, но с префиксом deox-opt-field-* чтобы не
  // коллидировать с forward/inverse/compare формами (если кто-то будет
  // querySelector'ить по id).
  const { step = 1, min, max, decimals = 2 } = opts;
  const id = `deox-opt-field-${key}`;
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

async function loadOptimizeMethods() {
  // Two-flag dedup: ``methodsLoaded`` short-circuits the fully-resolved
  // case; ``methodsLoading`` blocks a second concurrent fetch when the UI
  // re-enters the optimize sub-tab (or the activateSubtab/loadAll race)
  // while the first call is still in-flight. Without the in-flight flag
  // the catalog GET could be hit twice on fast tab clicks (PR 7 nit).
  if (state.optimize.methodsLoaded) return;
  if (state.optimize.methodsLoading) return;
  state.optimize.methodsLoading = true;
  try {
    const resp = await apiFetch('/api/deox/methods');
    state.optimize.methods = Array.isArray(resp.items) ? resp.items : [];
    state.optimize.defaultMethodId = resp.default || null;
    state.optimize.methodsLoaded = true;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить каталог методов подачи Al: ${detail}`);
  } finally {
    state.optimize.methodsLoading = false;
  }
}

function renderOptimizeForm() {
  const v = state.formValues.optimize || defaultOptimize();
  state.formValues.optimize = v;

  // ── Block A — heat ────────────────────────────────────────────────
  const blockA = el('div', { class: 'deox-form-heading' },
    'Блок A — Плавка');
  const gridA = el(
    'div',
    { class: 'deox-form-grid' },
    buildOptimizeField('steel_mass_ton', 'Масса плавки, т', v.steel_mass_ton,
      { step: 10, min: 1, max: 500, decimals: 0 }),
    buildOptimizeField('o_a_initial_ppm', '[O]_init после tap, ppm', v.o_a_initial_ppm,
      { step: 10, min: 10, max: 1500, decimals: 0 }),
    buildOptimizeField('temperature_C', 'T стали, °C', v.temperature_C,
      { step: 5, min: 1500, max: 1700, decimals: 0 }),
    buildOptimizeField('target_o_a_ppm', 'Целевой O_a, ppm', v.target_o_a_ppm,
      { step: 1, min: 1, max: 50, decimals: 1 }),
    buildOptimizeField('target_al_pct', 'Целевой [Al], %', v.target_al_pct,
      { step: 0.001, min: 0.005, max: 0.1, decimals: 3 }),
  );

  // ── Block B — slag carry-over ─────────────────────────────────────
  const slagToggle = el('label', { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      'data-field': 'slag_no_data',
      ...(v.slag_no_data ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        v.slag_no_data = !!ev.target.checked;
        // Re-render to enable/disable Block B fields.
        renderOptimizeForm();
      },
    }),
    el('span', {}, 'Нет данных по шлаку → basic forward (отключить slag-aware)'),
  );
  const slagDisabled = !!v.slag_no_data;
  // ``disabled`` is an HTML attribute valid only on form controls — putting
  // it on the wrapping <div> (PR 7 leftover) is harmless but lints poorly
  // and tricks a11y tools. We only carry the data-attr; the actual
  // disabling lands on the <input> children below.
  const gridB = el(
    'div',
    { class: 'deox-form-grid', 'data-block': 'slag' },
    buildOptionalField('slag_mass_kg', 'M_slag, кг', v.slag_mass_kg,
      { step: 100, min: 0, max: 10000, decimals: 0,
        placeholder: 'например, 2200' }),
    buildOptionalField('slag_feo_pct', '%FeO в шлаке', v.slag_feo_pct,
      { step: 1, min: 0, max: 50, decimals: 1,
        placeholder: 'например, 18' }),
    buildOptionalField('slag_mno_pct', '%MnO (опц.)', v.slag_mno_pct,
      { step: 0.5, min: 0, max: 20, decimals: 1 }),
    buildOptionalField('slag_sio2_pct', '%SiO₂ (опц.)', v.slag_sio2_pct,
      { step: 0.5, min: 0, max: 30, decimals: 1 }),
  );
  if (slagDisabled) {
    // Visually grey-out block B when "нет данных" is checked. Inputs stay
    // in DOM so readOptimizeForm can decide to drop them based on the flag.
    for (const inp of gridB.querySelectorAll('input')) {
      inp.disabled = true;
    }
    gridB.style.opacity = '0.5';
  }
  const blockB = el('div', { class: 'deox-form-heading' },
    'Блок B — Шлак переноса (BOF→LF carry-over)');
  const blockBNote = el('div', { class: 'deox-ai-subnote' },
    'Шлак с FeO=15-20 % после BOF-tap может содержать в 5-10× больше O, ' +
    'чем растворённый O в стали. Slag-aware расчёт учитывает это в O-балансе.');

  // ── Block C — Co-deoxidation (collapsed by toggle) ────────────────
  const coToggle = el('label', { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      'data-field': 'co_deox_enabled',
      ...(v.co_deox_enabled ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        v.co_deox_enabled = !!ev.target.checked;
        renderOptimizeForm();
      },
    }),
    el('span', {}, 'Введены FeSi / SiMn до Al? (Si pre-deox)'),
  );
  const blockC = el('div', { class: 'deox-form-heading' },
    'Блок C — Co-deoxidation (опционально)');
  const coDeoxFields = v.co_deox_enabled
    ? el(
        'div',
        { class: 'deox-form-grid' },
        buildOptionalField('co_deox_fesi_kg', 'm_FeSi подано, кг',
          v.co_deox_fesi_kg, { step: 10, min: 0, max: 5000, decimals: 0 }),
        buildOptionalField('co_deox_fesi_si_content_pct', '%Si в FeSi',
          v.co_deox_fesi_si_content_pct,
          { step: 1, min: 0.1, max: 100, decimals: 1 }),
      )
    : null;

  // ── Block D — Methods + override η_Al ─────────────────────────────
  const blockD = el('div', { class: 'deox-form-heading' },
    'Блок D — Методы подачи Al');
  const methodsItems = Array.isArray(state.optimize.methods)
    ? state.optimize.methods : [];
  const selectedMethodIds = new Set(v.method_ids || []);
  const methodCheckboxes = methodsItems.length === 0
    ? el('div', { class: 'deox-ai-subnote' },
        'Каталог методов загружается… (GET /api/deox/methods)')
    : el(
        'div',
        { class: 'deox-form-grid' },
        ...methodsItems.map((m) => {
          // Default: all selected (empty Set = all). After user toggles
          // any checkbox, the Set tracks the explicit selection.
          const isChecked = selectedMethodIds.size === 0
            ? true
            : selectedMethodIds.has(m.id);
          return el(
            'label',
            {
              class: 'deox-ai-save-toggle',
              style: { gridColumn: 'span 1' },
              title: m.notes || '',
            },
            el('input', {
              type: 'checkbox',
              'data-field': `method:${m.id}`,
              ...(isChecked ? { checked: 'checked' } : {}),
            }),
            el('span', {},
              `${m.name} (η≈${formatNumber(m.eta_al_typical, 2)})`,
              m.premium_eur_per_kg
                ? el('span', { class: 'predict-result-pm' },
                    ` +€${formatNumber(m.premium_eur_per_kg, 2)}/kg`)
                : null,
              m.carrier_gas
                ? el('span', { class: 'predict-result-pm' },
                    ` [${m.carrier_gas}]`)
                : null,
            ),
          );
        }),
      );

  const overrideToggle = el('label', { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      'data-field': 'user_override_eta_al_enabled',
      ...(v.user_override_eta_al_enabled ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        v.user_override_eta_al_enabled = !!ev.target.checked;
        renderOptimizeForm();
      },
    }),
    el('span', {}, 'Override η_Al вручную (например, из исторических плавок)'),
  );
  const overrideField = v.user_override_eta_al_enabled
    ? el(
        'div',
        { class: 'deox-form-grid' },
        buildOptionalField('user_override_eta_al', 'η_Al override (0.1-1.0)',
          v.user_override_eta_al,
          { step: 0.01, min: 0.1, max: 1.0, decimals: 2 }),
      )
    : null;

  // ── Block E — Constraints ─────────────────────────────────────────
  const blockE = el('div', { class: 'deox-form-heading' },
    'Блок E — Ограничения (опционально)');
  const gridE = el(
    'div',
    { class: 'deox-form-grid' },
    buildOptionalField('target_n_ppm', 'Целевой [N], ppm (для DX07)',
      v.target_n_ppm,
      { step: 5, min: 0, max: 500, decimals: 0,
        placeholder: 'например, 30 → исключить N₂-carrier' }),
    buildOptionalField('premium_cap_eur_per_kg', 'Premium cap, €/kg',
      v.premium_cap_eur_per_kg,
      { step: 0.10, min: 0, max: 20, decimals: 2,
        placeholder: 'например, 0.50' }),
    buildOptionalField('t_drying_c', 'T сушки гранул, °C (для DX05)',
      v.t_drying_c,
      { step: 10, min: 0, max: 600, decimals: 0,
        placeholder: 'если applicable' }),
  );

  // ── Block G — Optimization objective (PR 7) ───────────────────────
  // Radio triplet — cost / al_mass / pareto. Mutates ``v.objective`` on
  // change so the next runOptimize() reads the latest selection.
  // We keep the radio inputs ``data-field="objective"`` so an external
  // scraper / e2e test can read the current selection.
  const blockG = el('div', { class: 'deox-form-heading' },
    'Блок G — Критерий оптимизации');
  const makeObjectiveRadio = (value, label) => el('label',
    { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'radio',
      name: 'deox-optimize-objective',
      'data-field': 'objective',
      value,
      ...(v.objective === value ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        if (ev.target.checked) {
          v.objective = value;
        }
      },
    }),
    el('span', {}, label),
  );
  const objectiveGroup = el('div', { class: 'deox-form-grid' },
    makeObjectiveRadio('cost', 'Cost (€/heat) — default'),
    makeObjectiveRadio('al_mass', 'Al pure (кг) — carbon footprint'),
    makeObjectiveRadio('pareto', 'Pareto {Al × €} — ручной выбор'),
  );
  const objectiveNote = el('div', { class: 'deox-ai-subnote' },
    'Cost — стандарт (минимум €/heat). Al pure — минимизация массы Al ' +
    '(прокси CO₂-footprint и Al-инвентарь). Pareto — non-dominated frontier ' +
    'с knee-точкой; выбирается лучший баланс между Al и cost, на графике ' +
    'видна вся frontier для ручного выбора.');

  // ── Block F — Economics + thermo ──────────────────────────────────
  const blockF = el('div', { class: 'deox-form-heading' },
    'Блок F — Экономика и термодинамика');
  const thermoSelectId = 'deox-opt-thermo';
  const thermoSelect = el('select', {
    class: 'deox-select',
    id: thermoSelectId,
    'data-field': 'thermo_model_id',
  });
  const selectedThermo = v.thermo_model_id
    || state.selectedModelId
    || state.defaultModelId;
  for (const m of state.thermoModels) {
    const opt = el('option', { value: m.id },
      `${m.name} — ${m.citation || ''}`);
    if (m.id === selectedThermo) opt.selected = true;
    thermoSelect.append(opt);
  }
  const thermoField = el(
    'div',
    { class: 'deox-field' },
    el('label', { class: 'deox-field-label', for: thermoSelectId },
      'Термодинамическая модель'),
    thermoSelect,
  );
  const gridF = el(
    'div',
    { class: 'deox-form-grid' },
    thermoField,
    buildOptimizeField('al_commodity_price_eur_per_kg',
      'Цена Al commodity, €/kg', v.al_commodity_price_eur_per_kg,
      { step: 0.10, min: 0, max: 20, decimals: 2 }),
  );

  // ── Block H — η_Al prediction (PR 10) ─────────────────────────────
  // Opt-in checkbox + plant_id text input. When enabled, the optimizer
  // threads an EtaAlPredictor (plant Bayesian posterior + global ML) instead
  // of literature η. plant_id is required in that mode (validated client- +
  // server-side). Mutually exclusive with the Block D «Override η_Al».
  const blockH = el('div', { class: 'deox-form-heading' },
    'Блок H — ML-прогноз η_Al (опционально)');
  const etaPredToggle = el('label', { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      'data-field': 'enable_eta_prediction',
      ...(v.enable_eta_prediction ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        v.enable_eta_prediction = !!ev.target.checked;
        renderOptimizeForm();
      },
    }),
    el('span', {}, 'Использовать ML-прогноз η_Al (plant posterior + глобальный ML) ' +
      'вместо литературного η'),
  );
  const plantIdField = v.enable_eta_prediction
    ? el('div', { class: 'deox-field' },
        el('label', { class: 'deox-field-label', for: 'deox-opt-plant-id' },
          'plant_id (обязателен для ML-прогноза)'),
        el('input', {
          type: 'text',
          class: 'deox-input mono',
          id: 'deox-opt-plant-id',
          'data-field': 'plant_id',
          value: v.plant_id || '',
          placeholder: 'например, PLANT_A',
        }),
      )
    : null;
  const blockHNote = v.enable_eta_prediction
    ? el('div', { class: 'deox-ai-subnote' },
        'Калибровки по цехам управляются в табе «🎯 Калибровка η_Al». ' +
        'Если у цеха нет posterior\'а или ML-модель не обучена — predictor ' +
        'мягко деградирует к литературному η. Несовместимо с «Override η_Al».')
    : null;

  // ── Action button ─────────────────────────────────────────────────
  const submitBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    onClick: () => runOptimize(),
    ...(state.busy ? { disabled: 'disabled' } : {}),
  }, '🎯 Найти оптимальный метод');
  const actions = el('div', { class: 'deox-actions' }, submitBtn);

  const subnote = el('div', { class: 'deox-ai-subnote' },
    'Optimizer пробегает по всем выбранным методам, считает Al / cost / scatter ' +
    'через slag-aware O-баланс, фильтрует по constraints (carrier gas / premium cap) ' +
    'и возвращает Pareto-таблицу с минимальным cost-per-heat. ' +
    'Pattern Library проверяет DX04 (полноту slag), DX05 (T сушки гранул), ' +
    'DX06 (override η вне диапазона), DX07 (N₂-carrier для low-N).');

  elements.formContainer.replaceChildren(
    blockA, gridA,
    blockB, blockBNote, slagToggle, gridB,
    blockC, coToggle, ...(coDeoxFields ? [coDeoxFields] : []),
    blockD, methodCheckboxes, overrideToggle,
    ...(overrideField ? [overrideField] : []),
    blockE, gridE,
    blockG, objectiveGroup, objectiveNote,
    blockF, gridF,
    blockH, etaPredToggle,
    ...(plantIdField ? [plantIdField] : []),
    ...(blockHNote ? [blockHNote] : []),
    actions, subnote,
  );
}

function readOptimizeForm(formRoot) {
  // Build OptimizationRequest body. Optional fields are dropped when empty
  // so the backend reads them as None / default. Slag block is suppressed
  // when slag_no_data checkbox is checked.
  const out = {};
  const v = state.formValues.optimize || defaultOptimize();

  // Required Block A fields — always read.
  const requiredKeys = [
    'steel_mass_ton',
    'o_a_initial_ppm',
    'temperature_C',
    'target_o_a_ppm',
    'target_al_pct',
  ];
  for (const key of requiredKeys) {
    const inp = formRoot.querySelector(`input[data-field="${key}"]`);
    if (!inp) {
      throw new Error(`Поле «${key}» не найдено`);
    }
    const val = parseFloat(inp.value);
    if (Number.isNaN(val)) {
      throw new Error(`Поле «${key}» содержит некорректное значение`);
    }
    out[key] = val;
  }

  // Slag block — only included when slag_no_data is unchecked.
  const slagToggleInp = formRoot.querySelector('input[data-field="slag_no_data"]');
  const slagNoData = !!(slagToggleInp && slagToggleInp.checked);
  if (!slagNoData) {
    for (const key of ['slag_mass_kg', 'slag_feo_pct',
      'slag_mno_pct', 'slag_sio2_pct']) {
      const inp = formRoot.querySelector(`input[data-field="${key}"]`);
      if (!inp) continue;
      const raw = (inp.value || '').trim();
      if (raw === '') continue;
      const val = parseFloat(raw);
      if (Number.isNaN(val)) {
        throw new Error(`Поле «${key}» содержит некорректное значение`);
      }
      out[key] = val;
    }
  }

  // Block C — co-deox.
  const coEnabledInp = formRoot.querySelector(
    'input[data-field="co_deox_enabled"]');
  const coEnabled = !!(coEnabledInp && coEnabledInp.checked);
  if (coEnabled) {
    const kgInp = formRoot.querySelector('input[data-field="co_deox_fesi_kg"]');
    const siInp = formRoot.querySelector(
      'input[data-field="co_deox_fesi_si_content_pct"]');
    if (kgInp && kgInp.value && kgInp.value.trim() !== '') {
      const kg = parseFloat(kgInp.value);
      if (!Number.isNaN(kg) && kg > 0) {
        out.co_deox_fesi_kg = kg;
      }
    }
    if (siInp && siInp.value && siInp.value.trim() !== '') {
      const si = parseFloat(siInp.value);
      if (!Number.isNaN(si)) {
        out.co_deox_fesi_si_content_pct = si;
      }
    }
  }

  // Block D — method_ids multi-select. If user kept the default (all
  // checked) we send `null` so backend iterates the full catalog. If some
  // checkboxes are unchecked, send the explicit subset.
  const methodInputs = formRoot.querySelectorAll(
    'input[data-field^="method:"]');
  const totalMethods = methodInputs.length;
  const checkedIds = [];
  for (const inp of methodInputs) {
    if (inp.checked) {
      const id = inp.dataset.field.slice('method:'.length);
      checkedIds.push(id);
    }
  }
  if (totalMethods > 0 && checkedIds.length < totalMethods) {
    if (checkedIds.length === 0) {
      throw new Error('Выберите хотя бы один метод подачи Al');
    }
    out.method_ids = checkedIds;
  }
  // When every method is checked (or the catalog is empty / not loaded yet)
  // we omit ``method_ids`` and the backend iterates the full YAML catalog.
  v.method_ids = totalMethods > 0 && checkedIds.length < totalMethods
    ? checkedIds : [];

  // Override η_Al.
  const overrideToggleInp = formRoot.querySelector(
    'input[data-field="user_override_eta_al_enabled"]');
  if (overrideToggleInp && overrideToggleInp.checked) {
    const inp = formRoot.querySelector(
      'input[data-field="user_override_eta_al"]');
    if (inp && inp.value && inp.value.trim() !== '') {
      const val = parseFloat(inp.value);
      if (!Number.isNaN(val)) {
        out.user_override_eta_al = val;
      }
    }
  }

  // Block E — constraints (all optional).
  for (const key of ['target_n_ppm', 'premium_cap_eur_per_kg', 't_drying_c']) {
    const inp = formRoot.querySelector(`input[data-field="${key}"]`);
    if (!inp) continue;
    const raw = (inp.value || '').trim();
    if (raw === '') continue;
    const val = parseFloat(raw);
    if (!Number.isNaN(val)) {
      out[key] = val;
    }
  }

  // Block F — thermo + economics.
  const thermoSelect = formRoot.querySelector(
    'select[data-field="thermo_model_id"]');
  if (thermoSelect && thermoSelect.value) {
    out.thermo_model_id = thermoSelect.value;
  } else if (state.selectedModelId || state.defaultModelId) {
    out.thermo_model_id = state.selectedModelId || state.defaultModelId;
  }
  const priceInp = formRoot.querySelector(
    'input[data-field="al_commodity_price_eur_per_kg"]');
  if (priceInp && priceInp.value && priceInp.value.trim() !== '') {
    const val = parseFloat(priceInp.value);
    if (!Number.isNaN(val)) {
      out.al_commodity_price_eur_per_kg = val;
    }
  }

  // Block G — multi-objective (PR 7). Read the checked radio; if no radio
  // is present (older HTML or hidden by future refactor) fall back to the
  // form-state value (default 'cost').
  const checkedObjective = formRoot.querySelector(
    'input[data-field="objective"]:checked');
  out.objective = checkedObjective ? checkedObjective.value : (v.objective || 'cost');

  // Block H — η_Al prediction (PR 10). Only emit fields when enabled so the
  // default (disabled) request is byte-identical to PR 7. Client-side guard:
  // checkbox checked without plant_id raises before we POST.
  const etaPredInp = formRoot.querySelector(
    'input[data-field="enable_eta_prediction"]');
  const etaPredEnabled = !!(etaPredInp && etaPredInp.checked);
  let plantIdValue = '';
  if (etaPredEnabled) {
    const plantInp = formRoot.querySelector('input[data-field="plant_id"]');
    plantIdValue = plantInp ? (plantInp.value || '').trim() : '';
    if (plantIdValue === '') {
      throw new Error(
        'ML-прогноз η_Al требует plant_id — укажите цех или снимите галочку.',
      );
    }
    if (out.user_override_eta_al != null) {
      throw new Error(
        'ML-прогноз η_Al несовместим с ручным «Override η_Al» — ' +
        'оставьте только одно.',
      );
    }
    out.enable_eta_prediction = true;
    out.plant_id = plantIdValue;
  }

  // Cache the parsed snapshot back into form values for re-render fidelity.
  // Use Block A keys + optional ones that we successfully parsed; the
  // toggle flags are preserved on the v object via the onChange handlers.
  state.formValues.optimize = {
    ...v,
    steel_mass_ton: out.steel_mass_ton,
    o_a_initial_ppm: out.o_a_initial_ppm,
    temperature_C: out.temperature_C,
    target_o_a_ppm: out.target_o_a_ppm,
    target_al_pct: out.target_al_pct,
    slag_no_data: slagNoData,
    slag_mass_kg: out.slag_mass_kg ?? v.slag_mass_kg,
    slag_feo_pct: out.slag_feo_pct ?? v.slag_feo_pct,
    slag_mno_pct: out.slag_mno_pct ?? v.slag_mno_pct,
    slag_sio2_pct: out.slag_sio2_pct ?? v.slag_sio2_pct,
    co_deox_enabled: coEnabled,
    co_deox_fesi_kg: out.co_deox_fesi_kg ?? v.co_deox_fesi_kg,
    co_deox_fesi_si_content_pct:
      out.co_deox_fesi_si_content_pct ?? v.co_deox_fesi_si_content_pct,
    user_override_eta_al_enabled: !!(
      overrideToggleInp && overrideToggleInp.checked),
    user_override_eta_al:
      out.user_override_eta_al ?? v.user_override_eta_al,
    target_n_ppm: out.target_n_ppm ?? null,
    premium_cap_eur_per_kg: out.premium_cap_eur_per_kg ?? null,
    t_drying_c: out.t_drying_c ?? null,
    thermo_model_id: out.thermo_model_id || null,
    al_commodity_price_eur_per_kg:
      out.al_commodity_price_eur_per_kg
      ?? v.al_commodity_price_eur_per_kg,
    objective: out.objective || v.objective || 'cost',
    enable_eta_prediction: etaPredEnabled,
    plant_id: etaPredEnabled ? plantIdValue : (v.plant_id || ''),
  };

  return out;
}

function renderInlineParetoScatter(container, frontier, chosenId) {
  // PR 7 — inline SVG scatter of the Pareto frontier {al_pure_kg vs
  // cost_per_heat_eur}. We don't reuse charts/pareto.js — that module is
  // shaped around an entirely different domain (NSGA-II inverse design
  // candidates). Inline SVG keeps the dependency surface tight.
  //
  // Layout: 400x250 px, 40 px padding for axis labels. Chosen point is
  // rendered as a larger filled circle (#e63946) with a black stroke;
  // non-chosen points are smaller dark-blue (#1d3557). Each <circle> has
  // a <title> child for the hover tooltip — browser default behaviour.
  const W = 400;
  const H = 250;
  const PAD = 40;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const als = frontier.map((r) => r.al_pure_kg);
  const costs = frontier.map((r) => r.cost_per_heat_eur);
  const alMin = Math.min(...als);
  const alMax = Math.max(...als);
  const costMin = Math.min(...costs);
  const costMax = Math.max(...costs);
  // Guard against zero-range axes (degenerate single-point frontier
  // already short-circuits above, but two identical points would still
  // divide-by-zero without the 1e-9 floor).
  const xScale = (a) => PAD + ((a - alMin) / Math.max(alMax - alMin, 1e-9))
    * (W - 2 * PAD);
  const yScale = (c) => (H - PAD) - ((c - costMin) / Math.max(costMax - costMin, 1e-9))
    * (H - 2 * PAD);

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(H));
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.background = '#fafafa';
  svg.style.border = '1px solid #ddd';
  svg.style.borderRadius = '4px';

  // X-axis baseline.
  const xAxis = document.createElementNS(SVG_NS, 'line');
  xAxis.setAttribute('x1', String(PAD));
  xAxis.setAttribute('y1', String(H - PAD));
  xAxis.setAttribute('x2', String(W - PAD));
  xAxis.setAttribute('y2', String(H - PAD));
  xAxis.setAttribute('stroke', '#333');
  svg.appendChild(xAxis);
  // Y-axis baseline.
  const yAxis = document.createElementNS(SVG_NS, 'line');
  yAxis.setAttribute('x1', String(PAD));
  yAxis.setAttribute('y1', String(PAD));
  yAxis.setAttribute('x2', String(PAD));
  yAxis.setAttribute('y2', String(H - PAD));
  yAxis.setAttribute('stroke', '#333');
  svg.appendChild(yAxis);

  // X-label.
  const xLabel = document.createElementNS(SVG_NS, 'text');
  xLabel.setAttribute('x', String(W / 2));
  xLabel.setAttribute('y', String(H - 8));
  xLabel.setAttribute('text-anchor', 'middle');
  xLabel.setAttribute('font-size', '12');
  xLabel.setAttribute('fill', '#333');
  xLabel.textContent = 'Al pure, кг';
  svg.appendChild(xLabel);
  // Y-label (rotated -90°).
  const yLabel = document.createElementNS(SVG_NS, 'text');
  yLabel.setAttribute('x', '12');
  yLabel.setAttribute('y', String(H / 2));
  yLabel.setAttribute('text-anchor', 'middle');
  yLabel.setAttribute('font-size', '12');
  yLabel.setAttribute('fill', '#333');
  yLabel.setAttribute('transform', `rotate(-90 12 ${H / 2})`);
  yLabel.textContent = 'Cost, €/heat';
  svg.appendChild(yLabel);

  // Axis tick labels — min/max on each axis for orientation.
  const tickAttr = (x, y, text, anchor = 'middle') => {
    const t = document.createElementNS(SVG_NS, 'text');
    t.setAttribute('x', String(x));
    t.setAttribute('y', String(y));
    t.setAttribute('text-anchor', anchor);
    t.setAttribute('font-size', '10');
    t.setAttribute('fill', '#666');
    t.textContent = text;
    svg.appendChild(t);
  };
  tickAttr(PAD, H - PAD + 14, formatNumber(alMin, 0), 'start');
  tickAttr(W - PAD, H - PAD + 14, formatNumber(alMax, 0), 'end');
  tickAttr(PAD - 4, H - PAD + 2, formatNumber(costMin, 0), 'end');
  tickAttr(PAD - 4, PAD + 4, formatNumber(costMax, 0), 'end');

  // Points — chosen rendered last so it sits on top.
  const sortedForRender = [...frontier].sort((a, b) => {
    if (a.method_id === chosenId) return 1;
    if (b.method_id === chosenId) return -1;
    return 0;
  });
  for (const r of sortedForRender) {
    const cx = xScale(r.al_pure_kg);
    const cy = yScale(r.cost_per_heat_eur);
    const isChosen = r.method_id === chosenId;
    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute('cx', String(cx));
    circle.setAttribute('cy', String(cy));
    circle.setAttribute('r', isChosen ? '8' : '5');
    circle.setAttribute('fill', isChosen ? '#e63946' : '#1d3557');
    if (isChosen) {
      circle.setAttribute('stroke', '#000');
      circle.setAttribute('stroke-width', '2');
    }
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent =
      `${r.method_name}: ${formatNumber(r.al_pure_kg, 1)} кг, `
      + `${formatNumber(r.cost_per_heat_eur, 0)} €/heat`;
    circle.appendChild(title);
    svg.appendChild(circle);
  }

  container.appendChild(svg);
}

function renderOptimizeResult() {
  const data = state.results.optimize;
  if (!data) {
    elements.resultContainer.replaceChildren();
    return;
  }

  const blocks = [];

  // ── Recommendation card ───────────────────────────────────────────
  const chosenName = data.chosen_method_name || data.chosen_method_id || '—';
  const chosenCost = data.chosen_cost_eur;
  const inputs = data.inputs || {};
  const steelMass = Number(inputs.steel_mass_ton || 0);
  const costPerTon = steelMass > 0
    ? chosenCost / steelMass : null;

  const headline = el(
    'div',
    { class: 'deox-result-headline' },
    el('div', { class: 'deox-result-label' }, 'Рекомендованный метод'),
    el(
      'div',
      { class: 'deox-result-mean mono' },
      chosenName,
    ),
    el('div', { class: 'deox-result-sub' },
      `Cost: ${formatNumber(chosenCost, 2)} €/heat`
      + (costPerTon != null
        ? ` (${formatNumber(costPerTon, 3)} €/т стали)` : '')
      + (data.thermo_model_used
        ? ` · thermo: ${data.thermo_model_used}` : '')),
  );
  blocks.push(headline);

  if (data.rationale) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Обоснование. '), data.rationale));
  }

  // Runner-up summary.
  if (data.runner_up_method_id) {
    const runnerUp = (data.pareto_table || [])
      .find((r) => r.method_id === data.runner_up_method_id);
    const runnerName = runnerUp
      ? runnerUp.method_name : data.runner_up_method_id;
    const delta = data.runner_up_delta_eur;
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Runner-up: '),
      `${runnerName}`,
      delta != null
        ? ` (+€${formatNumber(delta, 2)} к выбранному)`
        : '',
    ));
  }

  // Constraints active (bullet list).
  if (Array.isArray(data.constraints_active)
      && data.constraints_active.length > 0) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Активные ограничения:'),
      el('ul', { class: 'deox-ai-list' },
        ...data.constraints_active.map((c) => el('li', {}, c)),
      ),
    ));
  }

  // Rejected methods (id + reason).
  if (Array.isArray(data.rejected_methods)
      && data.rejected_methods.length > 0) {
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Исключённые методы:'),
      el('ul', { class: 'deox-ai-list' },
        ...data.rejected_methods.map((rej) => el('li', {},
          el('strong', {}, `${rej.method_id}: `),
          rej.reason || '—',
        )),
      ),
    ));
  }

  // ── Pareto table ──────────────────────────────────────────────────
  const pareto = Array.isArray(data.pareto_table) ? data.pareto_table : [];
  if (pareto.length > 0) {
    const head = el('thead', {},
      el('tr', {},
        el('th', {}, 'Метод'),
        el('th', {}, 'η_Al'),
        el('th', {}, 'Al pure, кг'),
        el('th', {}, 'Al charge, кг'),
        el('th', {}, '€/heat'),
        el('th', {}, '€/т'),
        el('th', {}, 'kg Al/т'),
        el('th', {}, '± scatter'),
        el('th', {}, 'Carrier'),
        el('th', {}, 'Warnings'),
      ),
    );
    const tbody = el('tbody', {},
      ...pareto.map((row) => {
        const isChosen = row.method_id === data.chosen_method_id;
        const trClass = isChosen ? 'is-min' : '';
        const warns = Array.isArray(row.warnings) ? row.warnings : [];
        return el('tr', {},
          el('td', { class: `deox-compare-name ${trClass}` },
            isChosen ? `★ ${row.method_name}` : row.method_name),
          el('td', {}, formatNumber(row.eta_al_used, 2)),
          el('td', {}, formatNumber(row.al_pure_kg, 1)),
          el('td', {}, formatNumber(row.al_charge_kg, 1)),
          el('td', { class: isChosen ? 'is-min' : '' },
            formatNumber(row.cost_per_heat_eur, 2)),
          el('td', {}, formatNumber(row.cost_per_ton_eur, 3)),
          el('td', {}, formatNumber(row.al_specific_kg_per_t, 3)),
          el('td', {}, `±${formatNumber(row.scatter_kg, 1)}`),
          el('td', {}, row.carrier_gas || '—'),
          el('td', {}, warns.length > 0 ? warns.join('; ') : '—'),
        );
      }),
    );
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Pareto-таблица (отсортирована по €/heat ↑):'),
      el('table', { class: 'deox-compare-table' }, head, tbody),
    ));
  }

  // ── Pareto scatter (PR 7) ─────────────────────────────────────────
  // Only render the SVG when objective === "pareto" and the frontier has
  // ≥ 1 point. For cost/al_mass the backend returns pareto_frontier=[]
  // so this block stays inert and the existing pareto_table is the only
  // visualisation.
  if (
    data.objective === 'pareto'
    && Array.isArray(data.pareto_frontier)
    && data.pareto_frontier.length >= 1
  ) {
    const scatterContainer = el('div',
      { class: 'pareto-scatter-container' });
    renderInlineParetoScatter(
      scatterContainer,
      data.pareto_frontier,
      data.chosen_method_id,
    );
    blocks.push(el('div', { class: 'deox-ai-block' },
      el('strong', {}, 'Pareto-frontier {Al pure × €/heat}:'),
      el('div', { class: 'deox-ai-subnote' },
        '🔴 — knee-точка (выбранный метод). Наведите курсор на точку — ' +
        'tooltip покажет имя метода, Al pure и cost.'),
      scatterContainer,
    ));
  }

  // ── Pattern warnings (re-use existing renderer) ───────────────────
  const wBlock = renderWarnings(data.pattern_warnings);
  if (wBlock) blocks.push(wBlock);

  // ── Save button (POST /api/deox/optimize/save → 501 in PR 7) ──────
  const saveBtn = el('button', {
    class: 'btn',
    type: 'button',
    onClick: () => runOptimizeSave(),
    ...(state.optimize.saving ? { disabled: 'disabled' } : {}),
  }, state.optimize.saving
    ? 'Сохранение…'
    : '💾 Сохранить рекомендацию');
  blocks.push(el('div', { class: 'deox-actions' }, saveBtn));

  elements.resultContainer.replaceChildren(
    el('div', { class: 'deox-result' }, ...blocks),
  );
}

async function runOptimize() {
  if (state.busy) return;
  let body;
  try {
    body = readOptimizeForm(elements.formContainer);
  } catch (err) {
    showError(err.message);
    return;
  }
  await dispatchPost(
    '/api/deox/optimize', body, 'optimize', renderOptimizeResult,
  );
}

async function runOptimizeSave() {
  const data = state.results.optimize;
  if (!data) {
    showError('Сначала запустите оптимизацию — нечего сохранять.');
    return;
  }
  if (state.optimize.saving) return;

  // PR 8 — Variant A: backend re-executes the recommendation from the same
  // inputs to keep a single source of truth. We submit the optimize-form
  // payload, augmented with ``heat_id`` and ``author`` — *not* the
  // ``data`` recommendation object (UI-state echo would risk drift if the
  // form was edited after the last run).
  let body;
  try {
    body = readOptimizeForm(elements.formContainer);
  } catch (err) {
    showError(err.message);
    return;
  }
  body.heat_id = null;   // Reserved — heat-id input lives in a future PR
  body.author = 'user';

  state.optimize.saving = true;
  clearError();
  // Re-render so the save button shows the spinner state.
  renderOptimizeResult();

  try {
    const resp = await apiFetch('/api/deox/optimize/save', {
      method: 'POST',
      body,
    });
    const decisionId = resp && resp.decision_id;
    const snapPath = resp && resp.methods_snapshot_path;
    showInfo(
      `Рекомендация сохранена в Decision Log (id=${decisionId}). ` +
      `Snapshot методов: ${snapPath}`,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 501) {
      // Defensive fallback: PR 8 implements the endpoint, but if a stale
      // backend is hit (e.g. mid-rollout) we still surface a sensible note.
      showError(
        err.message
          || 'Сохранение Decision Log пока недоступно на этом backend.',
      );
    } else {
      const detail = err instanceof ApiError ? err.message : String(err);
      showError(`Ошибка сохранения: ${detail}`);
    }
  } finally {
    state.optimize.saving = false;
    renderOptimizeResult();
  }
}

// ──────────────────── History sub-tab (PR 2 — ASIS-deox calibration) ────────────────────
//
// Heats CRUD UI:
//   - Filter bar (plant / method / has_outcome)
//   - «➕ Новая плавка» disclosure form (4 required + collapsible optional)
//   - Table of heats with inline outcome-PATCH and DELETE actions
//   - «Загрузить ещё» button for keyset pagination
//
// All endpoints are under /api/deox/heats* — see app/api/routers/heats.py.
// We reuse the existing optimize-sub-tab methods catalog (GET /api/deox/methods)
// for the method dropdown so the user sees the same list in both places.

function defaultHeatForm() {
  // POST body schema for /api/deox/heats. Required: source / plant_id /
  // steel_mass_ton / o_a_initial_ppm. Optionals start as '' (empty input)
  // so readHeatForm drops them. ``source='manual'`` is hard-coded — the
  // ETL paths (excel_etl, csv_bulk, synthetic) write directly to the DB
  // via PR 3-4 scripts, not through this form.
  return {
    source: 'manual',
    plant_id: '',
    heat_id: '',
    steel_class_id: '',
    steel_mass_ton: '',
    o_a_initial_ppm: '',
    o_a_after_ppm: '',
    t_tap_c: '',
    t_lf_arrival_c: '',
    t_al_addition_c: '',
    al_added_kg: '',
    al_residual_pct: '',
    slag_mass_kg: '',
    carry_over_slag_kg_per_t: '',
    slag_feo_pct: '',
    slag_mno_pct: '',
    slag_sio2_pct: '',
    slag_cao_pct: '',
    slag_mgo_pct: '',
    slag_al2o3_pct: '',
    c_pct: '',
    mn_pct: '',
    si_pct: '',
    s_pct: '',
    p_pct: '',
    method_id: '',
    addition_timing: '',
    carrier_gas: '',
    co_deox_fesi_kg: '',
    dt_to_al_min: '',
    t_drying_c: '',
    ar_stir_nm3: '',
    vacuum_treatment: '',
    refractory_heat_count: '',
    eta_al_effective: '',
    quality_flag: '',
    notes: '',
  };
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatHeatTs(iso) {
  // Render YYYY-MM-DD HH:MM in local TZ (the iso string carries UTC).
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const pad = (n) => String(n).padStart(2, '0');
    return (
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
      `${pad(d.getHours())}:${pad(d.getMinutes())}`
    );
  } catch {
    return iso;
  }
}

async function loadHistoryDataOnce() {
  // Idempotent (loaded flag short-circuits) but a manual refresh path
  // calls loadHistoryList directly to re-fetch list without methods.
  if (state.history.loading) return;
  state.history.loading = true;
  state.history.error = null;
  try {
    // Fire methods catalog load if not already cached — share with optimize tab.
    const methodsPromise = state.optimize.methodsLoaded
      ? Promise.resolve()
      : loadOptimizeMethods();
    const [listResp, plantsResp] = await Promise.all([
      apiFetch(buildHeatsListUrl()),
      apiFetch('/api/deox/heats/plants'),
      methodsPromise,
    ]);
    state.history.items = Array.isArray(listResp.items) ? listResp.items : [];
    state.history.total = Number(listResp.total || 0);
    state.history.nextBeforeId = listResp.next_before_id ?? null;
    state.history.plants = Array.isArray(plantsResp.items) ? plantsResp.items : [];
    state.history.loaded = true;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    state.history.error = `Не удалось загрузить историю плавок: ${detail}`;
  } finally {
    state.history.loading = false;
  }
}

function buildHeatsListUrl(opts = {}) {
  const f = state.history.filters;
  const params = new URLSearchParams();
  params.set('limit', '100');
  if (f.plant_id) params.set('plant_id', f.plant_id);
  if (f.method_id) params.set('method_id', f.method_id);
  if (f.has_outcome === 'true' || f.has_outcome === 'false') {
    params.set('has_outcome', f.has_outcome);
  }
  if (opts.before_id != null) params.set('before_id', String(opts.before_id));
  return `/api/deox/heats?${params.toString()}`;
}

async function loadHistoryList({ append = false } = {}) {
  if (state.history.loading) return;
  state.history.loading = true;
  state.history.error = null;
  try {
    const url = buildHeatsListUrl(
      append && state.history.nextBeforeId != null
        ? { before_id: state.history.nextBeforeId }
        : {},
    );
    const resp = await apiFetch(url);
    const items = Array.isArray(resp.items) ? resp.items : [];
    state.history.items = append ? state.history.items.concat(items) : items;
    state.history.total = Number(resp.total || 0);
    state.history.nextBeforeId = resp.next_before_id ?? null;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    state.history.error = `Ошибка загрузки списка: ${detail}`;
  } finally {
    state.history.loading = false;
    renderHistorySubtab();
  }
}

async function reloadPlantsList() {
  try {
    const resp = await apiFetch('/api/deox/heats/plants');
    state.history.plants = Array.isArray(resp.items) ? resp.items : [];
  } catch {
    // Non-fatal — dropdown stays stale until next reload.
  }
}

function renderHistorySubtab() {
  // The history sub-tab puts everything into formContainer + resultContainer
  // because we already have those two containers from the skeleton. We use
  // formContainer for filters + new-heat form, and resultContainer for the table.
  if (!elements) return;

  // ── Filters + Add button ─────────────────────────────────────────
  const filtersBar = renderHistoryFilters();
  const addToggle = el('button', {
    class: 'btn',
    type: 'button',
    onClick: () => {
      state.history.formOpen = !state.history.formOpen;
      if (state.history.formOpen && !state.history.formValues) {
        state.history.formValues = defaultHeatForm();
      }
      renderHistorySubtab();
    },
  }, state.history.formOpen ? '▾ Скрыть форму' : '➕ Новая плавка');
  const reloadBtn = el('button', {
    class: 'btn',
    type: 'button',
    onClick: () => loadHistoryList({ append: false }),
  }, state.history.loading ? '⏳ Загрузка…' : '🔄 Обновить');
  const toolbar = el(
    'div',
    { class: 'deox-actions', style: { marginTop: '12px', flexWrap: 'wrap' } },
    addToggle,
    reloadBtn,
  );

  // ── Optional create form ─────────────────────────────────────────
  const formBlock = state.history.formOpen ? renderNewHeatForm() : null;

  const parts = [filtersBar, toolbar];
  if (formBlock) parts.push(formBlock);
  elements.formContainer.replaceChildren(...parts);

  // ── Table + load-more ────────────────────────────────────────────
  elements.resultContainer.replaceChildren(renderHistoryTable());
}

function renderHistoryFilters() {
  const f = state.history.filters;
  const plantSelect = el('select', {
    class: 'deox-select',
    'data-field': 'filter_plant_id',
    onChange: (ev) => {
      state.history.filters.plant_id = ev.target.value || '';
      loadHistoryList({ append: false });
    },
  },
    el('option', { value: '' }, '(все площадки)'),
    ...state.history.plants.map((p) => {
      const opt = el('option', { value: p.plant_id },
        `${p.plant_id} (${p.count})`);
      if (p.plant_id === f.plant_id) opt.selected = true;
      return opt;
    }),
  );
  const methodsItems = Array.isArray(state.optimize.methods)
    ? state.optimize.methods : [];
  const methodSelect = el('select', {
    class: 'deox-select',
    'data-field': 'filter_method_id',
    onChange: (ev) => {
      state.history.filters.method_id = ev.target.value || '';
      loadHistoryList({ append: false });
    },
  },
    el('option', { value: '' }, '(все методы)'),
    ...methodsItems.map((m) => {
      const opt = el('option', { value: m.id }, m.name || m.id);
      if (m.id === f.method_id) opt.selected = true;
      return opt;
    }),
  );
  const outcomeSelect = el('select', {
    class: 'deox-select',
    'data-field': 'filter_has_outcome',
    onChange: (ev) => {
      state.history.filters.has_outcome = ev.target.value || '';
      loadHistoryList({ append: false });
    },
  },
    el('option', { value: '' }, '(все статусы)'),
    (() => {
      const o = el('option', { value: 'true' }, 'С outcome');
      if (f.has_outcome === 'true') o.selected = true;
      return o;
    })(),
    (() => {
      const o = el('option', { value: 'false' }, 'In-progress');
      if (f.has_outcome === 'false') o.selected = true;
      return o;
    })(),
  );

  const filterField = (label, ctrl) => el('div', { class: 'deox-context-cell' },
    el('span', { class: 'deox-context-label' }, label),
    ctrl,
  );

  return el('div', { class: 'deox-context', style: { marginTop: '12px' } },
    filterField('Площадка', plantSelect),
    filterField('Метод подачи Al', methodSelect),
    filterField('Outcome', outcomeSelect),
  );
}

function renderNewHeatForm() {
  const v = state.history.formValues || defaultHeatForm();
  state.history.formValues = v;

  const heading = el('div', { class: 'deox-form-heading' },
    'Создание новой плавки (manual entry)');

  // Required fields up top — must be set or POST will 422.
  const requiredGrid = el(
    'div',
    { class: 'deox-form-grid' },
    buildHeatField('plant_id', 'Площадка *', v.plant_id, { kind: 'text', placeholder: 'ASIS_BOF' }),
    buildHeatField('steel_mass_ton', 'Масса стали, т *', v.steel_mass_ton,
      { kind: 'number', step: 10, min: 1, max: 500 }),
    buildHeatField('o_a_initial_ppm', '[O]_a начальный, ppm *', v.o_a_initial_ppm,
      { kind: 'number', step: 10, min: 0, max: 2000 }),
    buildHeatField('heat_id', 'Heat ID (опц.)', v.heat_id, { kind: 'text', placeholder: 'H-12345' }),
  );

  // Disclosure for optional fields.
  const optionalToggleId = 'deox-heat-optional-toggle';
  const optionalChecked = !!v.__optional_open;
  const optionalToggle = el('label', { class: 'deox-ai-save-toggle' },
    el('input', {
      type: 'checkbox',
      id: optionalToggleId,
      ...(optionalChecked ? { checked: 'checked' } : {}),
      onChange: (ev) => {
        v.__optional_open = !!ev.target.checked;
        renderHistorySubtab();
      },
    }),
    el('span', {}, 'Показать дополнительные поля (slag / composition / method / outcome)'),
  );

  const optionalBlock = optionalChecked
    ? renderOptionalHeatFields(v)
    : null;

  const submitBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    ...(state.history.creating ? { disabled: 'disabled' } : {}),
    onClick: () => createHeatHandler(),
  }, state.history.creating ? 'Сохранение…' : '💾 Создать плавку');
  const cancelBtn = el('button', {
    class: 'btn',
    type: 'button',
    onClick: () => {
      state.history.formOpen = false;
      state.history.formValues = null;
      renderHistorySubtab();
    },
  }, 'Отменить');
  const actions = el('div', { class: 'deox-actions' }, submitBtn, cancelBtn);

  const parts = [heading, requiredGrid, optionalToggle];
  if (optionalBlock) parts.push(optionalBlock);
  parts.push(actions);
  return el('div', { class: 'deox-form-panel' }, ...parts);
}

function renderOptionalHeatFields(v) {
  const methodsItems = Array.isArray(state.optimize.methods)
    ? state.optimize.methods : [];

  // Slag block.
  const slag = el('div', { class: 'deox-form-grid' },
    buildHeatField('slag_mass_kg', 'M_slag, кг', v.slag_mass_kg,
      { kind: 'number', step: 100, min: 0, max: 10000 }),
    buildHeatField('slag_feo_pct', '%FeO', v.slag_feo_pct,
      { kind: 'number', step: 1, min: 0, max: 50 }),
    buildHeatField('slag_mno_pct', '%MnO', v.slag_mno_pct,
      { kind: 'number', step: 0.5, min: 0, max: 20 }),
    buildHeatField('slag_sio2_pct', '%SiO₂', v.slag_sio2_pct,
      { kind: 'number', step: 0.5, min: 0, max: 30 }),
    buildHeatField('slag_cao_pct', '%CaO', v.slag_cao_pct,
      { kind: 'number', step: 1, min: 0, max: 70 }),
    buildHeatField('slag_mgo_pct', '%MgO', v.slag_mgo_pct,
      { kind: 'number', step: 0.5, min: 0, max: 25 }),
    buildHeatField('slag_al2o3_pct', '%Al₂O₃', v.slag_al2o3_pct,
      { kind: 'number', step: 1, min: 0, max: 50 }),
    buildHeatField('carry_over_slag_kg_per_t', 'Шлак carry-over, kg/t',
      v.carry_over_slag_kg_per_t,
      { kind: 'number', step: 0.5, min: 0, max: 50 }),
  );

  // Composition.
  const comp = el('div', { class: 'deox-form-grid' },
    buildHeatField('c_pct', '%C', v.c_pct,
      { kind: 'number', step: 0.01, min: 0, max: 1.5, decimals: 3 }),
    buildHeatField('mn_pct', '%Mn', v.mn_pct,
      { kind: 'number', step: 0.1, min: 0, max: 3 }),
    buildHeatField('si_pct', '%Si', v.si_pct,
      { kind: 'number', step: 0.05, min: 0, max: 2.5 }),
    buildHeatField('s_pct', '%S', v.s_pct,
      { kind: 'number', step: 0.001, min: 0, max: 0.05, decimals: 4 }),
    buildHeatField('p_pct', '%P', v.p_pct,
      { kind: 'number', step: 0.001, min: 0, max: 0.05, decimals: 4 }),
    buildHeatField('steel_class_id', 'Класс стали', v.steel_class_id,
      { kind: 'text', placeholder: 'pipe_hsla' }),
  );

  // Temperatures + Al.
  const tempsAndAl = el('div', { class: 'deox-form-grid' },
    buildHeatField('t_tap_c', 'T_tap, °C', v.t_tap_c,
      { kind: 'number', step: 5, min: 1400, max: 1700 }),
    buildHeatField('t_lf_arrival_c', 'T_LF, °C', v.t_lf_arrival_c,
      { kind: 'number', step: 5, min: 1400, max: 1700 }),
    buildHeatField('t_al_addition_c', 'T_Al, °C', v.t_al_addition_c,
      { kind: 'number', step: 5, min: 1400, max: 1700 }),
    buildHeatField('al_added_kg', 'Al подано, кг', v.al_added_kg,
      { kind: 'number', step: 10, min: 0, max: 5000 }),
    buildHeatField('al_residual_pct', '[Al]_остаточный, %', v.al_residual_pct,
      { kind: 'number', step: 0.001, min: 0, max: 0.5, decimals: 4 }),
    buildHeatField('co_deox_fesi_kg', 'FeSi (co-deox), кг', v.co_deox_fesi_kg,
      { kind: 'number', step: 10, min: 0, max: 5000 }),
    buildHeatField('dt_to_al_min', 'Δt до Al, мин', v.dt_to_al_min,
      { kind: 'number', step: 1, min: 0, max: 120 }),
    buildHeatField('t_drying_c', 'T сушки, °C', v.t_drying_c,
      { kind: 'number', step: 10, min: 0, max: 600 }),
    buildHeatField('ar_stir_nm3', 'Ar stirring, Nm³', v.ar_stir_nm3,
      { kind: 'number', step: 1, min: 0, max: 100 }),
    buildHeatField('refractory_heat_count', 'Refractory N плавок',
      v.refractory_heat_count,
      { kind: 'number', step: 1, min: 0, max: 500, integer: true }),
  );

  // Method + enums.
  const methodOpts = [
    el('option', { value: '' }, '(не указан)'),
    ...methodsItems.map((m) => {
      const opt = el('option', { value: m.id }, m.name || m.id);
      if (m.id === v.method_id) opt.selected = true;
      return opt;
    }),
  ];
  const methodSelect = el('select', {
    class: 'deox-select',
    'data-field': 'method_id',
  }, ...methodOpts);
  const methodField = el('div', { class: 'deox-field' },
    el('label', { class: 'deox-field-label' }, 'Метод подачи Al'),
    methodSelect,
  );

  const enumField = (key, label, options, current) => {
    const opts = [el('option', { value: '' }, '(не указано)'),
      ...options.map((v_) => {
        const opt = el('option', { value: v_ }, v_);
        if (v_ === current) opt.selected = true;
        return opt;
      })];
    return el('div', { class: 'deox-field' },
      el('label', { class: 'deox-field-label' }, label),
      el('select', { class: 'deox-select', 'data-field': key }, ...opts),
    );
  };

  const methodAndEnums = el('div', { class: 'deox-form-grid' },
    methodField,
    enumField('addition_timing', 'Addition timing',
      ['in_stream', 'trim_after_lf_arrival', 'split'], v.addition_timing),
    enumField('carrier_gas', 'Carrier gas',
      ['none', 'Ar', 'N2'], v.carrier_gas),
    enumField('vacuum_treatment', 'Vacuum',
      ['none', 'VD', 'RH'], v.vacuum_treatment),
  );

  // Outcome fields (optional at POST — usually set later via PATCH).
  const outcome = el('div', { class: 'deox-form-grid' },
    buildHeatField('o_a_after_ppm', '[O]_a после, ppm', v.o_a_after_ppm,
      { kind: 'number', step: 1, min: 0, max: 2000 }),
    buildHeatField('eta_al_effective', 'η_Al эффективная', v.eta_al_effective,
      { kind: 'number', step: 0.01, min: 0, max: 1.5, decimals: 3 }),
    enumField('quality_flag', 'Quality flag',
      ['accept', 'out_of_spec', 'unknown'], v.quality_flag),
  );

  const notesField = el('div', { class: 'deox-field', style: { gridColumn: '1 / -1' } },
    el('label', { class: 'deox-field-label' }, 'Notes (свободный текст)'),
    el('textarea', {
      class: 'deox-input',
      'data-field': 'notes',
      rows: '2',
      maxlength: '4000',
      style: { width: '100%' },
    }, v.notes || ''),
  );

  return el('div', { style: { marginTop: '8px' } },
    el('div', { class: 'deox-form-heading' }, 'Шлак (опц.)'), slag,
    el('div', { class: 'deox-form-heading' }, 'Состав (опц.)'), comp,
    el('div', { class: 'deox-form-heading' }, 'Температуры / Al / процесс (опц.)'), tempsAndAl,
    el('div', { class: 'deox-form-heading' }, 'Метод и enum-поля (опц.)'), methodAndEnums,
    el('div', { class: 'deox-form-heading' }, 'Outcome (можно ввести позже через PATCH)'), outcome,
    notesField,
  );
}

function buildHeatField(key, label, value, opts = {}) {
  const { kind = 'number', step = 1, min, max, decimals,
    placeholder, integer = false } = opts;
  const id = `deox-heat-field-${key}`;
  const display = value === '' || value == null
    ? ''
    : (kind === 'text' ? String(value)
      : decimals != null ? Number(value).toFixed(decimals) : String(value));
  const input = el('input', {
    type: kind === 'text' ? 'text' : 'number',
    class: 'deox-input mono',
    id,
    value: display,
    ...(kind === 'number'
      ? { step: integer ? '1' : String(step) }
      : {}),
    'data-field': key,
    ...(min != null ? { min: String(min) } : {}),
    ...(max != null ? { max: String(max) } : {}),
    ...(placeholder ? { placeholder } : {}),
  });
  return el('div', { class: 'deox-field' },
    el('label', { class: 'deox-field-label', for: id }, label),
    input,
  );
}

function readNewHeatForm(formRoot) {
  // Build POST /api/deox/heats body. Empty fields are dropped (backend
  // treats absence = None). Required fields throw if blank.
  const v = state.history.formValues || defaultHeatForm();
  const out = { source: 'manual' };

  const readText = (key, { required = false } = {}) => {
    const inp = formRoot.querySelector(`[data-field="${key}"]`);
    if (!inp) return null;
    const val = (inp.value || '').trim();
    if (val === '') {
      if (required) throw new Error(`Поле «${key}» обязательно`);
      return null;
    }
    return val;
  };
  const readNumber = (key, { required = false, integer = false } = {}) => {
    const inp = formRoot.querySelector(`[data-field="${key}"]`);
    if (!inp) return null;
    const raw = (inp.value || '').trim();
    if (raw === '') {
      if (required) throw new Error(`Поле «${key}» обязательно`);
      return null;
    }
    const val = integer ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isNaN(val)) {
      throw new Error(`Поле «${key}» содержит некорректное значение`);
    }
    return val;
  };

  // Required.
  out.plant_id = readText('plant_id', { required: true });
  out.steel_mass_ton = readNumber('steel_mass_ton', { required: true });
  out.o_a_initial_ppm = readNumber('o_a_initial_ppm', { required: true });

  // Optional text.
  for (const k of ['heat_id', 'steel_class_id', 'notes']) {
    const val = readText(k);
    if (val !== null) out[k] = val;
  }
  // Optional enums (also text inputs via select).
  for (const k of [
    'method_id', 'addition_timing', 'carrier_gas',
    'vacuum_treatment', 'quality_flag',
  ]) {
    const val = readText(k);
    if (val !== null) out[k] = val;
  }
  // Optional numbers.
  const numericKeys = [
    't_tap_c', 't_lf_arrival_c', 't_al_addition_c',
    'al_added_kg', 'al_residual_pct',
    'slag_mass_kg', 'carry_over_slag_kg_per_t',
    'slag_feo_pct', 'slag_mno_pct', 'slag_sio2_pct',
    'slag_cao_pct', 'slag_mgo_pct', 'slag_al2o3_pct',
    'c_pct', 'mn_pct', 'si_pct', 's_pct', 'p_pct',
    'co_deox_fesi_kg', 'dt_to_al_min', 't_drying_c',
    'ar_stir_nm3',
    'eta_al_effective', 'o_a_after_ppm',
  ];
  for (const k of numericKeys) {
    const val = readNumber(k);
    if (val !== null) out[k] = val;
  }
  const refractory = readNumber('refractory_heat_count', { integer: true });
  if (refractory !== null) out.refractory_heat_count = refractory;

  // Cache the parsed snapshot back so re-render keeps user input visible.
  state.history.formValues = { ...v, ...out };
  return out;
}

async function createHeatHandler() {
  if (state.history.creating) return;
  if (!elements) return;
  // The form is inside formContainer.
  const formRoot = elements.formContainer;
  let body;
  try {
    body = readNewHeatForm(formRoot);
  } catch (err) {
    showError(err.message || String(err));
    return;
  }
  clearError();
  state.history.creating = true;
  renderHistorySubtab();
  try {
    const resp = await apiFetch('/api/deox/heats', { method: 'POST', body });
    // Prepend the new record to the in-memory list so the operator
    // sees it without an extra round-trip.
    if (resp && resp.heat) {
      state.history.items = [resp.heat, ...state.history.items];
      state.history.total += 1;
    }
    state.history.formOpen = false;
    state.history.formValues = null;
    showInfo(`Плавка #${resp && resp.id} создана.`);
    // Refresh plants dropdown async (a new plant_id may have appeared).
    reloadPlantsList();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка создания плавки: ${detail}`);
  } finally {
    state.history.creating = false;
    renderHistorySubtab();
  }
}

function renderHistoryTable() {
  const items = state.history.items;
  if (state.history.error) {
    return el('div', { class: 'deox-warnings' },
      el('div', { class: 'deox-warning HIGH' }, state.history.error));
  }
  if (state.history.loading && items.length === 0) {
    return el('div', { class: 'deox-result' },
      el('div', { class: 'deox-result-sub' }, 'Загружаю историю плавок…'));
  }
  if (!items || items.length === 0) {
    return el('div', { class: 'deox-result' },
      el('div', { class: 'deox-result-sub' },
        'Пока ни одной плавки. Нажмите «➕ Новая плавка» чтобы добавить первую.'));
  }

  const head = el('thead', {},
    el('tr', {},
      el('th', {}, 'ID'),
      el('th', {}, 'Создано'),
      el('th', {}, 'Площадка'),
      el('th', {}, 'Heat ID'),
      el('th', {}, 'Масса, т'),
      el('th', {}, '[O]_a init→after, ppm'),
      el('th', {}, 'Метод'),
      el('th', {}, 'η_Al'),
      el('th', {}, 'Quality'),
      el('th', {}, 'Действия'),
    ),
  );
  const rows = [];
  for (const it of items) {
    rows.push(renderHistoryRow(it));
    if (state.history.editingId === it.id) {
      rows.push(renderHistoryEditRow(it));
    }
  }
  const table = el('table', { class: 'deox-compare-table' },
    head,
    el('tbody', {}, ...rows),
  );

  // Pagination + summary.
  const summary = el('div', { class: 'deox-ai-subnote' },
    `Показано ${items.length} из ${state.history.total} плавок. ` +
    (state.history.nextBeforeId != null
      ? 'Доступна следующая страница.'
      : 'Конец списка.'));

  const loadMoreBtn = state.history.nextBeforeId != null
    ? el('button', {
        class: 'btn',
        type: 'button',
        ...(state.history.loading ? { disabled: 'disabled' } : {}),
        onClick: () => loadHistoryList({ append: true }),
      }, state.history.loading ? 'Загружаю…' : 'Загрузить ещё 100')
    : null;
  const actions = loadMoreBtn
    ? el('div', { class: 'deox-actions' }, loadMoreBtn)
    : null;

  const parts = [
    el('div', { class: 'deox-ai-block' },
      el('strong', {}, `История плавок (${state.history.total} всего)`)),
    table, summary,
  ];
  if (actions) parts.push(actions);
  return el('div', { class: 'deox-result' }, ...parts);
}

function renderHistoryRow(it) {
  const isEditing = state.history.editingId === it.id;
  const oA = `${it.o_a_initial_ppm}→${it.o_a_after_ppm ?? '?'}`;
  const editBtn = el('button', {
    class: 'btn',
    type: 'button',
    style: { padding: '4px 8px' },
    onClick: () => openEditRow(it),
  }, isEditing ? '× Отменить' : '✎ Outcome');
  const deleteBtn = el('button', {
    class: 'btn',
    type: 'button',
    style: { padding: '4px 8px', marginLeft: '4px' },
    onClick: () => deleteHeatHandler(it.id),
  }, '🗑');
  return el('tr', {},
    el('td', { class: 'mono' }, String(it.id)),
    el('td', { class: 'mono' }, formatHeatTs(it.created_at)),
    el('td', {}, it.plant_id || '—'),
    el('td', { class: 'mono' }, it.heat_id || '—'),
    el('td', {}, formatNumber(it.steel_mass_ton, 0)),
    el('td', { class: 'mono' }, oA),
    el('td', {}, it.method_id || '—'),
    el('td', { class: 'mono' }, it.eta_al_effective != null
      ? formatNumber(it.eta_al_effective, 3) : '—'),
    el('td', {}, it.quality_flag || '—'),
    el('td', {}, editBtn, deleteBtn),
  );
}

function openEditRow(it) {
  if (state.history.editingId === it.id) {
    state.history.editingId = null;
  } else {
    state.history.editingId = it.id;
    state.history.editValues = {
      o_a_after_ppm: it.o_a_after_ppm ?? '',
      al_residual_pct: it.al_residual_pct ?? '',
      eta_al_effective: it.eta_al_effective ?? '',
      quality_flag: it.quality_flag ?? '',
    };
  }
  renderHistorySubtab();
}

function renderHistoryEditRow(it) {
  const v = state.history.editValues;
  const input = (key, opts = {}) => {
    const { step = 1, min, max } = opts;
    return el('input', {
      type: 'number',
      class: 'deox-input mono',
      'data-edit-field': key,
      value: v[key] === '' || v[key] == null ? '' : String(v[key]),
      step: String(step),
      ...(min != null ? { min: String(min) } : {}),
      ...(max != null ? { max: String(max) } : {}),
      style: { width: '90px' },
    });
  };
  const qfOpts = [
    el('option', { value: '' }, '(не менять)'),
    ...['accept', 'out_of_spec', 'unknown'].map((qf) => {
      const opt = el('option', { value: qf }, qf);
      if (qf === v.quality_flag) opt.selected = true;
      return opt;
    }),
  ];
  const qfSelect = el('select', {
    class: 'deox-select',
    'data-edit-field': 'quality_flag',
    style: { width: '120px' },
  }, ...qfOpts);

  const saveBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    style: { padding: '4px 10px' },
    ...(state.history.patching ? { disabled: 'disabled' } : {}),
    onClick: () => patchHeatOutcomeHandler(it.id),
  }, state.history.patching ? 'Сохраняю…' : '💾 Записать outcome');

  const editCell = el('td', { colspan: '10', style: { background: 'rgba(0,0,0,0.04)' } },
    el('div', { class: 'deox-form-grid', style: { padding: '8px' } },
      el('div', { class: 'deox-field' },
        el('label', { class: 'deox-field-label' }, '[O]_a после, ppm'),
        input('o_a_after_ppm', { step: 1, min: 0, max: 2000 })),
      el('div', { class: 'deox-field' },
        el('label', { class: 'deox-field-label' }, '[Al]_residual, %'),
        input('al_residual_pct', { step: 0.001, min: 0, max: 0.5 })),
      el('div', { class: 'deox-field' },
        el('label', { class: 'deox-field-label' }, 'η_Al эффективная'),
        input('eta_al_effective', { step: 0.01, min: 0, max: 1.5 })),
      el('div', { class: 'deox-field' },
        el('label', { class: 'deox-field-label' }, 'Quality flag'),
        qfSelect),
      el('div', { class: 'deox-actions', style: { gridColumn: '1 / -1' } }, saveBtn),
    ),
  );
  return el('tr', {}, editCell);
}

async function patchHeatOutcomeHandler(heatId) {
  if (state.history.patching) return;
  const body = {};
  const root = elements && elements.resultContainer;
  if (!root) return;
  const fields = ['o_a_after_ppm', 'al_residual_pct', 'eta_al_effective'];
  for (const k of fields) {
    const inp = root.querySelector(`[data-edit-field="${k}"]`);
    if (!inp) continue;
    const raw = (inp.value || '').trim();
    if (raw === '') continue;
    const val = parseFloat(raw);
    if (Number.isNaN(val)) {
      showError(`Поле «${k}» содержит некорректное значение`);
      return;
    }
    body[k] = val;
  }
  const qfSel = root.querySelector('[data-edit-field="quality_flag"]');
  if (qfSel && qfSel.value) body.quality_flag = qfSel.value;

  if (Object.keys(body).length === 0) {
    showError('Заполните хотя бы одно поле outcome');
    return;
  }
  clearError();
  state.history.patching = true;
  renderHistorySubtab();
  try {
    const resp = await apiFetch(
      `/api/deox/heats/${encodeURIComponent(heatId)}`,
      { method: 'PATCH', body },
    );
    // Replace the row in state.items in-place.
    if (resp && resp.heat) {
      state.history.items = state.history.items.map(
        (it) => it.id === heatId ? resp.heat : it,
      );
    }
    state.history.editingId = null;
    showInfo(
      `Outcome обновлён для плавки #${heatId}. ` +
      `Decision Log: id=${resp && resp.decision_id}`,
    );
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка обновления outcome: ${detail}`);
  } finally {
    state.history.patching = false;
    renderHistorySubtab();
  }
}

async function deleteHeatHandler(heatId) {
  // eslint-disable-next-line no-alert
  const ok = window.confirm(
    `Удалить плавку #${heatId}? Это действие необратимо.`,
  );
  if (!ok) return;
  clearError();
  try {
    await apiFetch(
      `/api/deox/heats/${encodeURIComponent(heatId)}`,
      { method: 'DELETE' },
    );
    state.history.items = state.history.items.filter((it) => it.id !== heatId);
    state.history.total = Math.max(0, state.history.total - 1);
    if (state.history.editingId === heatId) state.history.editingId = null;
    showInfo(`Плавка #${heatId} удалена.`);
    // Refresh plants list — a plant_id may have lost its last heat.
    reloadPlantsList();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка удаления: ${detail}`);
  } finally {
    renderHistorySubtab();
  }
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

// ──────────────────── η_Al calibration sub-tab (PR 10) ────────────────────
//
// Two read endpoints + one run endpoint feed this tab:
//   GET  /api/deox/eta-al-model/status — trained ML model metrics + plants
//   GET  /api/deox/calibrations        — plant×method Bayesian posteriors
//   POST /api/deox/calibrations/run    — (re)run calibration synchronously
//
// We render two panels: the ML model status block (R² / 90%-CI coverage with
// an amber pill when out of the 85-95% band) and the plant calibrations table
// (one row per plant×method, skipped rows tagged). A «Запустить калибровку»
// button re-runs all plants and re-fetches both endpoints.

async function loadEtaCalibOnce() {
  if (state.etaCalib.loading) return;
  if (state.etaCalib.loaded) return;
  state.etaCalib.loading = true;
  state.etaCalib.error = null;
  try {
    const [statusResp, calibResp] = await Promise.all([
      apiFetch('/api/deox/eta-al-model/status'),
      apiFetch('/api/deox/calibrations'),
    ]);
    state.etaCalib.modelStatus = statusResp || null;
    state.etaCalib.calibrations = Array.isArray(calibResp.items)
      ? calibResp.items : [];
    state.etaCalib.loaded = true;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    state.etaCalib.error = `Не удалось загрузить состояние калибровки: ${detail}`;
  } finally {
    state.etaCalib.loading = false;
  }
}

async function runEtaCalibration() {
  if (state.etaCalib.running) return;
  state.etaCalib.running = true;
  clearError();
  renderEtaCalibSubtab();
  try {
    const resp = await apiFetch('/api/deox/calibrations/run', {
      method: 'POST',
      body: {},   // null plant_id → all plants
    });
    const n = resp && resp.plants_calibrated;
    const written = resp && resp.yaml_written;
    showInfo(
      `Калибровка завершена: ${n} цех(ов), записано YAML: ${written}.`,
    );
    // Re-fetch both endpoints so the table + status reflect new posteriors.
    state.etaCalib.loaded = false;
    await loadEtaCalibOnce();
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Ошибка калибровки: ${detail}`);
  } finally {
    state.etaCalib.running = false;
    renderEtaCalibSubtab();
  }
}

function renderEtaCalibModelPanel() {
  const s = state.etaCalib.modelStatus;
  if (!s || !s.model_present) {
    return el('div', { class: 'deox-result' },
      el('div', { class: 'deox-form-heading' }, 'ML-модель η_Al'),
      el('div', { class: 'deox-warning low' },
        'Модель не обучена. Обучите модель η_Al (deox_calibration class) ' +
        'в вкладке «Обучение», затем калибровка сможет смешивать plant ' +
        'posterior с глобальным ML-прогнозом.'),
    );
  }
  const coverage = s.coverage_90_ci;
  const inTarget = !!s.coverage_in_target;
  const coveragePill = coverage == null
    ? el('span', { class: 'deox-context-value' }, '—')
    : el('span', {
        class: inTarget ? 'deox-context-value' : 'deox-warning medium',
        style: inTarget ? {} : { padding: '2px 8px', borderRadius: '4px' },
      }, `${formatNumber(coverage * 100, 1)} %${inTarget ? '' : ' ⚠ вне 85–95 %'}`);

  const grid = el('div', { class: 'deox-result-grid' },
    cell('Версия модели', s.model_version || '—'),
    cell('R² (test)', s.r2_test == null ? '—' : formatNumber(s.r2_test, 3)),
    cell('N train', s.n_train == null ? '—' : String(s.n_train)),
    cell('Обучена', s.trained_at ? formatHeatTs(s.trained_at) : '—'),
  );
  const coverageRow = el('div', { class: 'deox-result-grid' },
    el('div', { class: 'deox-result-cell' },
      el('div', { class: 'deox-result-cell-label' }, 'Покрытие 90% CI'),
      el('div', { class: 'deox-result-cell-value' }, coveragePill)),
  );
  return el('div', { class: 'deox-result' },
    el('div', { class: 'deox-form-heading' }, 'ML-модель η_Al'),
    grid,
    coverageRow,
  );
}

function renderEtaCalibTable() {
  const rows = state.etaCalib.calibrations;
  if (!Array.isArray(rows) || rows.length === 0) {
    return el('div', { class: 'deox-result' },
      el('div', { class: 'deox-form-heading' }, 'Plant-калибровки (Bayesian posterior)'),
      el('div', { class: 'deox-ai-subnote' },
        'Калибровок пока нет. Нажмите «Запустить калибровку», чтобы посчитать ' +
        'posterior η_Al по историческим плавкам (нужно ≥30 плавок на метод).'),
    );
  }
  const header = el('tr', {},
    el('th', {}, 'Цех'),
    el('th', {}, 'Метод'),
    el('th', {}, 'N плавок'),
    el('th', {}, 'η_post'),
    el('th', {}, 'q05–q95'),
    el('th', {}, 'η_prior'),
    el('th', {}, 'Статус'),
  );
  const body = rows.map((r) => {
    const skipped = r.skipped_reason != null;
    const post = r.posterior_eta_mean;
    const q05 = r.posterior_eta_q05;
    const q95 = r.posterior_eta_q95;
    return el('tr', skipped ? { class: 'deox-row-skipped' } : {},
      el('td', {}, escapeHtml(r.plant_id)),
      el('td', { class: 'mono' }, escapeHtml(r.method_id)),
      el('td', { class: 'mono' }, r.n_heats_used == null ? '—' : String(r.n_heats_used)),
      el('td', { class: 'mono' }, post == null ? '—' : formatNumber(post, 3)),
      el('td', { class: 'mono' },
        (q05 == null || q95 == null)
          ? '—'
          : `${formatNumber(q05, 3)}–${formatNumber(q95, 3)}`),
      el('td', { class: 'mono' },
        r.prior_eta_mean == null ? '—' : formatNumber(r.prior_eta_mean, 3)),
      el('td', {},
        skipped
          ? el('span', { class: 'deox-warning low', style: { padding: '2px 6px' } },
              escapeHtml(r.skipped_reason))
          : el('span', { class: 'deox-context-value' }, 'OK')),
    );
  });
  const table = el('table', { class: 'candidate-table' },
    el('thead', {}, header),
    el('tbody', {}, ...body),
  );
  return el('div', { class: 'deox-result' },
    el('div', { class: 'deox-form-heading' }, 'Plant-калибровки (Bayesian posterior)'),
    table,
  );
}

function renderEtaCalibSubtab() {
  if (!elements) return;

  const heading = el('div', { class: 'deox-form-heading' },
    'Калибровка η_Al — plant posterior + глобальный ML');
  const subnote = el('div', { class: 'deox-ai-subnote' },
    'Bayesian-калибровка обновляет литературный prior η_Al posterior\'ом по ' +
    'историческим плавкам каждого цеха (в logit-пространстве). Глобальная ' +
    'ML-модель добавляет feature-aware прогноз; predictor смешивает оба ' +
    'источника (mixture-of-experts). Включите «Использовать ML-прогноз η_Al» ' +
    'в табе «Оптимизация метода», чтобы применить это вместо литературного η.');
  const runBtn = el('button', {
    class: 'btn primary',
    type: 'button',
    onClick: () => runEtaCalibration(),
    ...(state.etaCalib.running ? { disabled: 'disabled' } : {}),
  }, state.etaCalib.running ? 'Калибрую…' : 'Запустить калибровку');
  const actions = el('div', { class: 'deox-actions' }, runBtn);

  elements.formContainer.replaceChildren(heading, subnote, actions);

  if (state.etaCalib.loading) {
    elements.resultContainer.replaceChildren(
      el('div', { class: 'deox-ai-subnote' }, 'Загрузка…'));
    return;
  }
  if (state.etaCalib.error) {
    elements.resultContainer.replaceChildren(
      el('div', { class: 'deox-warning high' }, state.etaCalib.error));
    return;
  }
  elements.resultContainer.replaceChildren(
    renderEtaCalibModelPanel(),
    renderEtaCalibTable(),
  );
}

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
    } else if (prevTab === 'optimize') {
      try {
        // readOptimizeForm caches state.formValues.optimize itself as a
        // side effect — call inside try so a half-typed numeric field
        // doesn't wipe the whole form snapshot.
        readOptimizeForm(elements.formContainer);
      } catch {
        // ignore — toggles are already preserved via their onChange handlers.
      }
    } else if (prevTab === 'history') {
      // History sub-tab: if the create form is open, flush its inputs
      // into state.history.formValues so the user keeps half-typed data
      // when they switch back. readNewHeatForm throws on missing
      // required fields — swallow that here (we don't submit on tab leave).
      if (state.history.formOpen && elements && elements.formContainer) {
        try {
          readNewHeatForm(elements.formContainer);
        } catch {
          // half-typed required field — keep cached values intact.
        }
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
  else if (id === 'optimize') {
    // Lazy-load /api/deox/methods on first activation, then re-render the
    // form so the multi-select picks up the catalog. Catalog fetch errors
    // surface via showError; the form still renders (catalog list shows
    // «Каталог загружается…» fallback text).
    if (!state.optimize.methodsLoaded) {
      renderOptimizeForm();   // initial render with empty catalog placeholder
      loadOptimizeMethods().then(() => {
        if (state.subtab === 'optimize') renderOptimizeForm();
      });
    } else {
      renderOptimizeForm();
    }
  } else if (id === 'eta_calib') {
    // Render skeleton + fetch status/calibrations on first activation.
    renderEtaCalibSubtab();
    if (!state.etaCalib.loaded) {
      loadEtaCalibOnce().then(() => {
        if (state.subtab === 'eta_calib') renderEtaCalibSubtab();
      });
    }
  } else if (id === 'history') {
    // Render skeleton + fetch data on first activation. Re-renders happen
    // inside loadHistoryDataOnce / loadHistoryList finally{} blocks.
    renderHistorySubtab();
    if (!state.history.loaded) {
      loadHistoryDataOnce().then(() => {
        if (state.subtab === 'history') renderHistorySubtab();
      });
    }
  }
  // Restore last result for that tab (or clear).
  if (id === 'forward') renderForwardResult();
  else if (id === 'inverse') renderInverseResult();
  else if (id === 'compare') renderCompareResult();
  else if (id === 'ai') renderAiResult();
  else if (id === 'optimize') renderOptimizeResult();
  // ``history`` / ``eta_calib`` paint their own resultContainer.
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
    else if (state.subtab === 'optimize') {
      if (!state.optimize.methodsLoaded) {
        renderOptimizeForm();
        loadOptimizeMethods().then(() => {
          if (state.subtab === 'optimize') renderOptimizeForm();
        });
      } else {
        renderOptimizeForm();
      }
    } else if (state.subtab === 'eta_calib') {
      renderEtaCalibSubtab();
      if (!state.etaCalib.loaded) {
        loadEtaCalibOnce().then(() => {
          if (state.subtab === 'eta_calib') renderEtaCalibSubtab();
        });
      }
    } else if (state.subtab === 'history') {
      renderHistorySubtab();
      if (!state.history.loaded) {
        loadHistoryDataOnce().then(() => {
          if (state.subtab === 'history') renderHistorySubtab();
        });
      }
    }
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
  state.formValues = {
    forward: null, inverse: null, compare: null, ai: null, optimize: null,
  };
  state.results = {
    forward: null, inverse: null, compare: null, ai: null, optimize: null,
  };
  state.busy = false;
  state.aiJob = { running: false, pollAbort: null, currentJobId: null };
  state.optimize = {
    methods: [], defaultMethodId: null,
    methodsLoaded: false, methodsLoading: false, saving: false,
  };
  state.history = {
    loaded: false, loading: false, error: null,
    items: [], total: 0, nextBeforeId: null,
    filters: { plant_id: '', method_id: '', has_outcome: '' },
    plants: [],
    formOpen: false, formValues: null, creating: false,
    editingId: null,
    editValues: {
      o_a_after_ppm: '', al_residual_pct: '',
      eta_al_effective: '', quality_flag: '',
    },
    patching: false,
  };

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadAll();
}
