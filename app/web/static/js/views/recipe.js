// Tab 06 — Подбор рецепта. PR 11 wires recipe_designer + critic loop
// (fatigue_carbon_steel only).
//
// Flow:
//   GET  /api/system/models             → {items, count}        (model dropdown)
//   GET  /api/system/models/active      → active model meta     (default selection)
//   POST /api/recipes/ai-cycle {model_version, task_text, save_to_decision_log}
//                                       → {job_id, config}
//   GET  /api/jobs/{job_id} (polled)    → result {recipes, reviews, baseline, ...}
//   DELETE /api/jobs/{job_id}           → cooperative cancel
//
// The recipe cycle is a single user-visible action ("запросить рецепт")
// that internally fans out to: Sonnet designer (~80 s) → ML+cost
// verification (XGBoost predict + ferroalloy cost compute, ~few s) →
// Sonnet critic with evidence fact-check (~100 s). Cooperative cancel
// works between the designer and critic LLM calls.
//
// Visual contract:
//   • Header + section sub describing what this tab does.
//   • Form panel: model dropdown + task textarea + save toggle + run btn.
//   • Class-gate banner if active model is not fatigue_carbon_steel —
//     cycle is blocked at the API level (400) and we surface the same
//     constraint up-front so the run button stays disabled.
//   • Below: when a job is in flight, the renderJobProgress card.
//   • When a result lands, a metric strip (baseline σf / cost / Δ counts)
//     followed by N recipe cards. Each card carries: rationale, evidence
//     list, expected outcome, ML verification metrics (Δσf/Δcost/CI),
//     critic verdict + fact-check + strengths/weaknesses, and an optional
//     ferroalloy breakdown table.

import { apiFetch, ApiError } from '../api.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { el } from '../utils/dom.js';

const SUPPORTED_STEEL_CLASS = 'fatigue_carbon_steel';

const DEFAULT_TASK_TEXT =
  'Снизить ferroalloy cost vs baseline при сохранении или улучшении ' +
  'fatigue strength. Целевой компромисс: каждый −€10/т имеет ценность ' +
  'если |Δσf| ≤ 30 МПа.';

// ──────────────────── module state ────────────────────

const state = {
  models: [],
  selectedModelVersion: null,
  loadingModels: false,
  job: {
    running: false,
    pollAbort: null,
    currentJobId: null,
  },
  lastResult: null,
  form: {
    task_text: DEFAULT_TASK_TEXT,
    save_to_decision_log: false,
  },
};

let elements = null;

// ──────────────────── label maps ────────────────────

// Reused vocabulary from PR 10 hypotheses view — same Russian glosses.
const NOVELTY_LABEL = { HIGH: 'ВЫСОКАЯ', MEDIUM: 'СРЕДНЯЯ', LOW: 'НИЗКАЯ' };
const VERDICT_LABEL = {
  ACCEPT: 'ПРИНЯТО',
  REVISE: 'ТРЕБУЕТ ПРАВОК',
  REJECT: 'ОТКЛОНЕНО',
};
const CONFIDENCE_LABEL = { HIGH: 'высокая', MEDIUM: 'средняя', LOW: 'низкая' };
const EVIDENCE_MARK = { VALID: '✓', INVALID: '✗', UNVERIFIABLE: '?' };
const EVIDENCE_LABEL = {
  VALID: 'подтверждено',
  INVALID: 'отклонено',
  UNVERIFIABLE: 'требует проверки',
};

function noveltyClass(v) {
  return ({ HIGH: 'high', MEDIUM: 'medium', LOW: 'low' })[v] || 'unknown';
}
function verdictClass(v) {
  return ({ ACCEPT: 'accept', REVISE: 'revise', REJECT: 'reject' })[v] || 'unknown';
}
function evidenceClass(v) {
  return ({ VALID: 'valid', INVALID: 'invalid', UNVERIFIABLE: 'unverifiable' })[v]
    || 'unverifiable';
}

// ──────────────────── helpers ────────────────────

function fmt(v, d = 1) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

function fmtSigned(v, d = 0) {
  if (v == null || Number.isNaN(v)) return '—';
  const n = Number(v);
  const sign = n > 0 ? '+' : (n < 0 ? '' : '±');
  return `${sign}${n.toFixed(d)}`;
}

function activeModel() {
  if (!state.selectedModelVersion) return null;
  return (
    state.models.find((m) => m.version === state.selectedModelVersion) || null
  );
}

function isClassGateOpen() {
  const m = activeModel();
  if (!m) return false;
  return (m.steel_class || '') === SUPPORTED_STEEL_CLASS;
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
        el('span', { class: 'here' }, 'Подбор рецепта'),
      ),
      el(
        'h1',
        { class: 'section-title' },
        '🧪 Подбор рецепта — PhD designer + critic',
      ),
      el(
        'p',
        { class: 'section-sub' },
        'Sonnet-designer формулирует 3-4 production-рецепта с двойной ' +
          'evidence (artifact + classical metallurgical mechanism). XGBoost+cost ' +
          'численно проверяют каждый рецепт. Sonnet-critic уровня journal ' +
          'reviewer #2 делает adversarial review с построчным fact-check ' +
          'evidence (✓ VALID / ✗ INVALID / ? UNVERIFIABLE). ' +
          'Полный цикл ~3 минуты, ~$0.20-0.25.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'recipe-error',
    role: 'alert',
    hidden: '',
  });
  const classBanner = el('div', {
    class: 'recipe-class-banner',
    hidden: '',
  });

  const modelSelect = el('select', {
    class: 'recipe-select mono',
    id: 'recipe-model-select',
    onChange: (ev) => onModelChange(ev.target.value),
  });
  const modelMeta = el('div', { class: 'recipe-model-meta mono' }, '');

  const taskTextarea = el('textarea', {
    class: 'recipe-textarea mono',
    id: 'recipe-task',
    rows: '4',
    maxlength: '2000',
    'data-field': 'task_text',
  });
  taskTextarea.value = state.form.task_text;

  const saveCheckbox = el('input', {
    type: 'checkbox',
    'data-field': 'save_to_decision_log',
    ...(state.form.save_to_decision_log ? { checked: 'checked' } : {}),
  });
  const saveLabel = el(
    'label',
    { class: 'recipe-save-toggle' },
    saveCheckbox,
    el('span', {}, 'Сохранить в Decision Log (тэг recipe_cycle)'),
  );

  const runBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn primary recipe-run-btn',
      onClick: () => runCycle(),
    },
    'Запросить рецепт',
  );

  const subnote = el(
    'div',
    { class: 'recipe-subnote' },
    'Полный цикл (designer + ML+cost truth gate + adversarial peer-review) ' +
      'занимает ~3 минуты, расходует ~$0.20-0.25 на Sonnet API. LLM call ' +
      'нельзя прервать после старта — кнопку «Отменить» лучше нажимать ' +
      'сразу после submit.',
  );

  const formPanel = el(
    'div',
    { class: 'recipe-form-panel' },
    el(
      'div',
      { class: 'recipe-form-row' },
      el(
        'label',
        { class: 'recipe-field', for: modelSelect.id },
        el('span', { class: 'recipe-field-label' }, 'Активная модель'),
        modelSelect,
      ),
      modelMeta,
    ),
    el(
      'label',
      { class: 'recipe-field', for: taskTextarea.id },
      el('span', { class: 'recipe-field-label' }, 'Задача проектирования'),
      taskTextarea,
    ),
    el(
      'div',
      { class: 'recipe-actions' },
      saveLabel,
      runBtn,
    ),
    subnote,
  );

  const progressMount = el('div', {
    class: 'recipe-progress-mount',
    'data-role': 'progress-mount',
  });
  const resultMount = el('div', {
    class: 'recipe-result-mount',
    'data-role': 'result-mount',
  });

  const body = el(
    'div',
    { class: 'recipe-body' },
    errorBanner,
    classBanner,
    formPanel,
    progressMount,
    resultMount,
  );
  const root = el('div', { class: 'recipe-view' }, head, body);

  return {
    root,
    errorBanner,
    classBanner,
    modelSelect,
    modelMeta,
    taskTextarea,
    saveCheckbox,
    runBtn,
    progressMount,
    resultMount,
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

function updateClassBanner() {
  if (!elements) return;
  const m = activeModel();
  if (!m) {
    elements.classBanner.hidden = true;
    elements.classBanner.textContent = '';
    return;
  }
  if ((m.steel_class || '') !== SUPPORTED_STEEL_CLASS) {
    elements.classBanner.hidden = false;
    elements.classBanner.replaceChildren(
      el('strong', {}, 'Доступно только для fatigue моделей. '),
      el(
        'span',
        {},
        `Активный класс — ${m.steel_class || '—'}. ` +
          `Recipe pair работает на ${SUPPORTED_STEEL_CLASS} ` +
          '(Agrawal NIMS). Выберите такую модель в выпадающем списке.',
      ),
    );
  } else {
    elements.classBanner.hidden = true;
    elements.classBanner.textContent = '';
  }
}

// ──────────────────── model loading ────────────────────

function renderModelSelect() {
  if (!elements) return;
  const sel = elements.modelSelect;
  sel.replaceChildren();
  if (!state.models.length) {
    const opt = el('option', { value: '' }, '— нет обученных моделей —');
    sel.append(opt);
    sel.disabled = true;
    elements.runBtn.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const m of state.models) {
    const labelTarget = m.target ? ` · ${m.target}` : '';
    const opt = el(
      'option',
      { value: m.version },
      `${m.version} (${m.steel_class || '—'}${labelTarget})`,
    );
    if (m.version === state.selectedModelVersion) opt.selected = true;
    sel.append(opt);
  }
  updateModelMeta();
  updateClassBanner();
  // Re-enable run button — disabled-state is per-jobstate elsewhere.
  if (!state.job.running) {
    elements.runBtn.disabled = !isClassGateOpen();
  }
}

function updateModelMeta() {
  if (!elements) return;
  const m = activeModel();
  if (!m) {
    elements.modelMeta.textContent = '';
    return;
  }
  const r2 = m.metrics?.r2_test;
  const cov = m.metrics?.coverage_90_ci;
  const parts = [];
  parts.push(`class=${m.steel_class || '—'}`);
  parts.push(`target=${m.target || '—'}`);
  if (r2 != null) parts.push(`R²=${fmt(r2, 3)}`);
  if (cov != null) parts.push(`coverage=${fmt(cov * 100, 1)}%`);
  elements.modelMeta.textContent = parts.join('  ·  ');
}

function onModelChange(version) {
  state.selectedModelVersion = version || null;
  // Stale results carry the prior model name in their metadata strip;
  // wiping them prevents that confusion while the user picks a fresh run.
  state.lastResult = null;
  renderResult();
  updateModelMeta();
  updateClassBanner();
  if (elements && !state.job.running) {
    elements.runBtn.disabled = !isClassGateOpen();
  }
}

// ──────────────────── form read ────────────────────

function readForm() {
  if (!elements) {
    return { ...state.form };
  }
  const out = { ...state.form };
  const taskVal = String(elements.taskTextarea.value || '').trim();
  if (taskVal.length === 0) {
    throw new Error('Задача проектирования не может быть пустой.');
  }
  if (taskVal.length > 2000) {
    throw new Error('Задача проектирования длиннее 2000 символов.');
  }
  out.task_text = taskVal;
  out.save_to_decision_log = !!elements.saveCheckbox.checked;
  state.form = out;
  return out;
}

// ──────────────────── result rendering ────────────────────

function renderResult() {
  if (!elements) return;
  const data = state.lastResult;
  if (!data) {
    elements.resultMount.replaceChildren();
    return;
  }

  const blocks = [];

  const recipes = Array.isArray(data.recipes) ? data.recipes : [];
  const reviewsById = {};
  for (const r of (data.reviews || [])) {
    if (r && r.recipe_id) reviewsById[r.recipe_id] = r;
  }
  const counts = data.verdict_counts || { ACCEPT: 0, REVISE: 0, REJECT: 0 };
  const baseline = data.baseline || {};

  // Headline — N recipes + duration.
  const headlineParts = [`${recipes.length} рецептов`];
  if (typeof data.duration_s === 'number') {
    headlineParts.push(`за ${fmt(data.duration_s, 0)} с`);
  }
  blocks.push(el('div', { class: 'recipe-result-headline' },
    headlineParts.join(' · '),
  ));

  if (data.decision_log_id != null) {
    blocks.push(el('div', { class: 'recipe-decision-saved' },
      `✔ Сохранено в Decision Log (id=${data.decision_log_id}, тэг recipe_cycle).`,
    ));
  }

  // Metric strip — baseline + verdict counts (4 cards).
  const metricsGrid = el(
    'div',
    { class: 'recipe-metrics-grid' },
    metricCard(
      'Baseline σf',
      `${fmt(baseline.predicted_property, 0)} МПа`,
    ),
    metricCard(
      'Baseline cost',
      `${fmt(baseline.cost_per_ton, 2)} €/т`,
    ),
    metricCard('Рецептов', String(recipes.length)),
    metricCard(
      'A / R / X',
      `${counts.ACCEPT || 0} / ${counts.REVISE || 0} / ${counts.REJECT || 0}`,
    ),
  );
  blocks.push(metricsGrid);

  if (recipes.length === 0) {
    blocks.push(el('div', { class: 'recipe-empty' },
      'Designer вернул 0 рецептов. Проверьте логи uvicorn — возможен ' +
      'malformed payload от Sonnet API или таймаут. Можно повторить запуск.',
    ));
  } else {
    for (let i = 0; i < recipes.length; i++) {
      blocks.push(renderRecipeCard(i + 1, recipes[i], reviewsById[recipes[i].id]));
    }
  }

  elements.resultMount.replaceChildren(
    el('div', { class: 'recipe-result' }, ...blocks),
  );
}

function metricCard(label, value, modifier) {
  const cls = modifier
    ? `recipe-metric-card recipe-metric-${modifier}`
    : 'recipe-metric-card';
  return el(
    'div',
    { class: cls },
    el('div', { class: 'recipe-metric-label' }, label),
    el('div', { class: 'recipe-metric-value mono' }, value),
  );
}

function renderRecipeCard(idx, r, review) {
  const ml = r.ml_verification || {};
  const novelty = r.novelty || 'LOW';

  const noveltyBadge = el(
    'span',
    { class: `recipe-badge recipe-novelty-${noveltyClass(novelty)}` },
    `новизна: ${NOVELTY_LABEL[novelty] || novelty}`,
  );

  const titleRow = el(
    'div',
    { class: 'recipe-card-head' },
    el(
      'div',
      { class: 'recipe-card-title' },
      el('span', { class: 'recipe-card-num' }, `${idx}.`),
      el('span', {}, ` ${r.name || '—'}`),
    ),
    el(
      'div',
      { class: 'recipe-card-badges' },
      noveltyBadge,
    ),
  );

  const blocks = [titleRow];

  // ML verification metric strip — Δσf / Δcost / прогноз / OOD.
  const dp = ml.delta_property;
  const dc = ml.delta_cost;
  const ood = !!ml.ood_flag;
  const ml_strip = el(
    'div',
    { class: 'recipe-ml-grid' },
    metricCard(
      'Δσf, МПа',
      `${fmtSigned(dp, 0)}`,
    ),
    metricCard(
      'Δcost, €/т',
      dc != null ? `${fmtSigned(dc, 2)}` : '—',
    ),
    metricCard(
      'Прогноз σf',
      `${fmt(ml.predicted_property, 0)} МПа`,
    ),
    metricCard(
      'OOD',
      ood ? '⚠ ДА' : '✓ нет',
      ood ? 'reject' : 'accept',
    ),
  );
  blocks.push(ml_strip);

  // CI band caption.
  if (ml.lower_90 != null && ml.upper_90 != null) {
    blocks.push(el('div', { class: 'recipe-ci-caption mono' },
      `90% CI: [${fmt(ml.lower_90, 0)}, ${fmt(ml.upper_90, 0)}] МПа`,
    ));
  }

  // Composition + process params (compact two-col display).
  const compEntries = Object.entries(r.composition || {});
  const procEntries = Object.entries(r.process_params || {});
  if (compEntries.length || procEntries.length) {
    blocks.push(el('div', { class: 'recipe-twocol' },
      el(
        'div',
        {},
        el('div', { class: 'recipe-twocol-label' }, 'Композиция'),
        el('pre', { class: 'recipe-json mono' },
          JSON.stringify(Object.fromEntries(compEntries), null, 2),
        ),
      ),
      el(
        'div',
        {},
        el('div', { class: 'recipe-twocol-label' }, 'Процесс'),
        el('pre', { class: 'recipe-json mono' },
          JSON.stringify(Object.fromEntries(procEntries), null, 2),
        ),
      ),
    ));
  }

  if (r.rationale) {
    blocks.push(el('div', { class: 'recipe-block' },
      el('strong', {}, 'Обоснование. '), r.rationale,
    ));
  }

  if (Array.isArray(r.evidence) && r.evidence.length) {
    blocks.push(el('div', { class: 'recipe-block' },
      el('strong', {}, 'Доказательная база.'),
      el('ul', { class: 'recipe-list' },
        ...r.evidence.map((e) => el('li', {}, e)),
      ),
    ));
  }

  if (r.expected_outcome) {
    blocks.push(el('div', { class: 'recipe-block' },
      el('strong', {}, 'Ожидание автора. '), r.expected_outcome,
    ));
  }

  if (r.risk_notes) {
    blocks.push(el('div', { class: 'recipe-block' },
      el('strong', {}, 'Риски. '), r.risk_notes,
    ));
  }

  // Critic verdict — appears below a divider when present.
  if (review) {
    blocks.push(el('div', { class: 'recipe-divider' }));
    blocks.push(renderVerdictBlock(review));
  }

  // Ferroalloy breakdown (collapsed by default — analytical detail only).
  const ferro = Array.isArray(ml.ferroalloy_breakdown) ? ml.ferroalloy_breakdown : [];
  if (ferro.length) {
    blocks.push(renderFerroBreakdown(ferro));
  }

  blocks.push(el('div', { class: 'recipe-card-meta' },
    `id=${r.id || '—'}` + (Array.isArray(r.tags) && r.tags.length
      ? `  ·  теги: ${r.tags.join(', ')}` : ''),
  ));

  return el('div', { class: 'recipe-card' }, ...blocks);
}

function renderVerdictBlock(review) {
  const verdict = review.verdict || 'UNKNOWN';
  const blocks = [];

  const verdictBadge = el(
    'span',
    {
      class: `recipe-verdict-badge recipe-verdict-${verdictClass(verdict)}`,
    },
    `👨‍🔬 PhD-рецензия: ${VERDICT_LABEL[verdict] || verdict}`,
  );
  const conf = el(
    'span',
    { class: 'recipe-verdict-confidence' },
    `уверенность ${CONFIDENCE_LABEL[review.confidence] || review.confidence || '—'}`,
  );
  blocks.push(el('div', { class: 'recipe-verdict-row' }, verdictBadge, conf));

  if (review.summary) {
    blocks.push(el('div', { class: 'recipe-summary' },
      el('em', {}, review.summary),
    ));
  }

  // Evidence fact-check — line-by-line VALID / INVALID / UNVERIFIABLE.
  const ec = Array.isArray(review.evidence_check) ? review.evidence_check : [];
  if (ec.length) {
    blocks.push(el('div', { class: 'recipe-block' },
      el('strong', {}, 'Fact-check доказательной базы.'),
      el('ul', { class: 'recipe-evidence-list' },
        ...ec.map((item) =>
          el('li', { class: `recipe-evidence-${evidenceClass(item.verdict)}` },
            el('span', { class: 'recipe-evidence-mark' },
              EVIDENCE_MARK[item.verdict] || '•'),
            ' ',
            el('strong', {}, item.claim || '—'),
            ' ',
            el('span', { class: 'recipe-evidence-tag' },
              `(${EVIDENCE_LABEL[item.verdict] || item.verdict})`),
            item.note ? el('div', { class: 'recipe-evidence-note' }, item.note) : null,
          ),
        ),
      ),
    ));
  }

  const strengths = Array.isArray(review.strengths) ? review.strengths : [];
  const weaknesses = Array.isArray(review.weaknesses) ? review.weaknesses : [];
  if (strengths.length || weaknesses.length) {
    blocks.push(el(
      'div',
      { class: 'recipe-twocol' },
      el(
        'div',
        {},
        el('strong', {}, 'Сильные стороны'),
        el('ul', { class: 'recipe-list' },
          ...strengths.map((s) => el('li', {}, s)),
        ),
      ),
      el(
        'div',
        {},
        el('strong', {}, 'Слабые стороны'),
        el('ul', { class: 'recipe-list' },
          ...weaknesses.map((w) => el('li', {}, w)),
        ),
      ),
    ));
  }

  if (review.suggested_revision) {
    blocks.push(el('div', { class: 'recipe-suggested-revision' },
      el('strong', {}, 'Предложение правки. '), review.suggested_revision,
    ));
  }

  return el('div', { class: 'recipe-verdict' }, ...blocks);
}

function renderFerroBreakdown(rows) {
  // Open/close via the native <details> so we don't ship state for it.
  const summary = el('summary', { class: 'recipe-ferro-summary' },
    `Расход ферросплавов в рецепте (${rows.length})`,
  );
  const head = el('thead', {},
    el('tr', {},
      el('th', {}, 'material'),
      el('th', { class: 'num' }, 'kg/т'),
      el('th', { class: 'num' }, '€/т'),
    ),
  );
  const body = el('tbody', {},
    ...rows.map((r) =>
      el('tr', {},
        el('td', { class: 'mono' }, String(r.material || '—')),
        el('td', { class: 'num mono' }, String(r.kg_per_ton ?? '—')),
        el('td', { class: 'num mono' }, String(r.eur_per_ton ?? '—')),
      ),
    ),
  );
  return el('details', { class: 'recipe-ferro' },
    summary,
    el('table', { class: 'recipe-ferro-table' }, head, body),
  );
}

// ──────────────────── actions ────────────────────

async function runCycle() {
  if (state.job.running) return;
  clearError();

  if (!state.selectedModelVersion) {
    showError('Сначала выберите активную модель.');
    return;
  }
  if (!isClassGateOpen()) {
    showError(`Recipe pair доступен только для ${SUPPORTED_STEEL_CLASS} моделей.`);
    return;
  }

  let form;
  try {
    form = readForm();
  } catch (err) {
    showError(err.message);
    return;
  }

  const body = {
    model_version: state.selectedModelVersion,
    task_text: form.task_text,
    save_to_decision_log: form.save_to_decision_log,
  };

  // Mark busy + clear stale result so we don't show a prior cycle's
  // verdicts adjacent to the new progress card.
  state.job.running = true;
  state.lastResult = null;
  elements.resultMount.replaceChildren();
  elements.runBtn.disabled = true;
  elements.progressMount.replaceChildren();

  const abort = new AbortController();
  state.job.pollAbort = abort;

  const progress = renderJobProgress(elements.progressMount, {
    label: 'Designer + ML+cost + critic… (~3 мин)',
    onCancel: () => cancelCycle(),
  });

  let jobId = null;
  try {
    const submit = await apiFetch('/api/recipes/ai-cycle', { method: 'POST', body });
    jobId = submit.job_id;
    state.job.currentJobId = jobId;
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
    } else if (err instanceof ApiError && err.status === 503) {
      progress.markError('LLM не настроен');
      showError(
        `LLM не настроен (нужен ANTHROPIC_API_KEY и prompts/). ${err.message}`,
      );
    } else if (err instanceof ApiError && err.status === 400) {
      progress.markError('Класс модели не поддерживается');
      showError(err.message);
    } else if (err instanceof ApiError && err.status === 404) {
      progress.markError('Модель не найдена');
      showError(`Модель не найдена: ${err.message}`);
    } else {
      const detail = err instanceof ApiError ? err.message : String(err.message || err);
      progress.markError(detail);
      showError(`Ошибка цикла: ${detail}`);
    }
  } finally {
    state.job.running = false;
    state.job.pollAbort = null;
    state.job.currentJobId = null;
    if (elements && elements.runBtn) {
      elements.runBtn.disabled = !isClassGateOpen();
    }
  }
}

async function cancelCycle() {
  // Best-effort: abort the polling loop on the client immediately, then
  // DELETE on the server. Cancellation is effective ONLY between the
  // two LLM calls — see recipes.py module docstring.
  const jobId = state.job.currentJobId;
  if (state.job.pollAbort) state.job.pollAbort.abort();
  if (!jobId) return;
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  } catch (err) {
    console.warn('[recipe] DELETE /api/jobs failed', err);
  }
}

// ──────────────────── bootstrap ────────────────────

async function loadModels() {
  if (!elements) return;
  state.loadingModels = true;
  clearError();
  try {
    const [listResp, activeResp] = await Promise.all([
      apiFetch('/api/system/models'),
      apiFetch('/api/system/models/active').catch((err) => {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    ]);
    const items = Array.isArray(listResp.items) ? listResp.items : [];
    state.models = items;

    // Prefer a fatigue model if there's at least one in the list, since
    // the cycle is fatigue-only — otherwise the form opens with a banner
    // and the run button stays disabled until the user picks one.
    const fatigueModels = items.filter(
      (m) => (m.steel_class || '') === SUPPORTED_STEEL_CLASS,
    );
    if (fatigueModels.length) {
      // Use the most recent fatigue model (last in sorted list — same
      // ordering as PR 3 system router).
      state.selectedModelVersion = fatigueModels[fatigueModels.length - 1].version;
    } else if (activeResp && activeResp.version) {
      state.selectedModelVersion = activeResp.version;
    } else if (items.length) {
      state.selectedModelVersion = items[items.length - 1].version;
    } else {
      state.selectedModelVersion = null;
    }
    renderModelSelect();
    if (!items.length) {
      showError(
        'Нет обученных моделей. Сначала обучите модель во вкладке «Обучение».',
      );
    }
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить список моделей: ${detail}`);
  } finally {
    state.loadingModels = false;
  }
}

/** Router entry-point. */
export function init(container) {
  // Reset module state defensively on re-init.
  state.models = [];
  state.selectedModelVersion = null;
  state.loadingModels = false;
  state.job = { running: false, pollAbort: null, currentJobId: null };
  state.lastResult = null;
  state.form = {
    task_text: DEFAULT_TASK_TEXT,
    save_to_decision_log: false,
  };

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadModels();
}
