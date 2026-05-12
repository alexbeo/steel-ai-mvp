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
    saving: false,
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
  if (state.optimize.methodsLoaded) return;
  try {
    const resp = await apiFetch('/api/deox/methods');
    state.optimize.methods = Array.isArray(resp.items) ? resp.items : [];
    state.optimize.defaultMethodId = resp.default || null;
    state.optimize.methodsLoaded = true;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить каталог методов подачи Al: ${detail}`);
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
  const slagAttr = slagDisabled ? { disabled: 'disabled' } : {};
  const gridB = el(
    'div',
    { class: 'deox-form-grid', 'data-block': 'slag', ...slagAttr },
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
    blockF, gridF,
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
  } else {
    // All selected (or catalog empty) → omit method_ids → backend uses all.
    // We cache the explicit list in form values for next render's reuse.
  }
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
  };

  return out;
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
  state.optimize.saving = true;
  clearError();
  // Re-render so the save button shows the spinner state.
  renderOptimizeResult();

  try {
    await apiFetch('/api/deox/optimize/save', {
      method: 'POST',
      body: {
        recommendation: data,
        heat_id: null,
        author: 'user',
      },
    });
    // Unreachable in PR 7 — backend always 501. Kept for PR 8 forward-compat.
    showError('Рекомендация сохранена в Decision Log.');
  } catch (err) {
    if (err instanceof ApiError && err.status === 501) {
      // Expected in PR 7 — surface as a friendly info banner, not as error.
      showError(
        err.message
          || 'Сохранение Decision Log будет реализовано в PR 8.',
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
    } else if (prevTab === 'optimize') {
      try {
        // readOptimizeForm caches state.formValues.optimize itself as a
        // side effect — call inside try so a half-typed numeric field
        // doesn't wipe the whole form snapshot.
        readOptimizeForm(elements.formContainer);
      } catch {
        // ignore — toggles are already preserved via their onChange handlers.
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
  }
  // Restore last result for that tab (or clear).
  if (id === 'forward') renderForwardResult();
  else if (id === 'inverse') renderInverseResult();
  else if (id === 'compare') renderCompareResult();
  else if (id === 'ai') renderAiResult();
  else if (id === 'optimize') renderOptimizeResult();
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
    methods: [], defaultMethodId: null, methodsLoaded: false, saving: false,
  };

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadAll();
}
