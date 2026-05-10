// Tab 05 — Гипотезы. PR 10 wires hypothesis_generator + critic loop.
//
// Streamlit parity reference: app/frontend/app.py lines 1474-1764
// (`with tab_hyp:`). The flow:
//   GET  /api/system/models             → {items, count}        (model dropdown)
//   GET  /api/system/models/active      → active model meta     (default selection)
//   POST /api/hypotheses/run {model_version, n_hypotheses, save_to_decision_log}
//                                       → {job_id, config}
//   GET  /api/jobs/{job_id} (polled)    → result {hypotheses, reviews, ...}
//   DELETE /api/jobs/{job_id}           → cooperative cancel
//
// The hypothesis cycle is a single user-visible action ("сгенерировать
// гипотезы") that internally fans out to two LLM calls — Sonnet
// generator (~100 s) followed by Sonnet critic (~45 s). Cooperative
// cancellation works between the LLM calls; once the SDK call is in
// flight the worker burns through to completion (UI stops polling).
//
// Visual contract:
//   • Header + section sub describing what this tab is for.
//   • Topbar-style strip showing the active model + n_hypotheses input
//     + "save to decision log" toggle + run button.
//   • Below: when a job is in flight, the renderJobProgress card.
//   • When a result lands, a verdict-counts strip + N hypothesis cards.
//     Each card carries the statement, rationale, proposed_experiment
//     (fix/sweep), expected_outcome, economic_impact, and the critic's
//     verdict + strengths/weaknesses below a divider.

import { apiFetch, ApiError } from '../api.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { el } from '../utils/dom.js';

// ──────────────────── module state ────────────────────

const state = {
  models: [],            // [{version, steel_class, target, ...}]
  selectedModelVersion: null,
  loadingModels: false,
  // Job state — one cycle in flight at a time.
  job: {
    running: false,
    pollAbort: null,
    currentJobId: null,
  },
  // Last result payload (rendered into the result panel). Cleared on
  // model change so the user doesn't see stale hypotheses for a
  // different model version.
  lastResult: null,
  // Form values held independently from DOM so re-render preserves them.
  form: {
    n_hypotheses: 5,
    save_to_decision_log: false,
  },
};

let elements = null;

// ──────────────────── helpers ────────────────────

function fmt(v, d = 1) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

function activeModel() {
  if (!state.selectedModelVersion) return null;
  return (
    state.models.find((m) => m.version === state.selectedModelVersion) || null
  );
}

// Reuse the verdict / novelty / cost label maps from Streamlit
// (app.py lines 1656-1674) so the badges read identically.
const NOVELTY_LABEL = {
  HIGH: 'ВЫСОКАЯ',
  MEDIUM: 'СРЕДНЯЯ',
  LOW: 'НИЗКАЯ',
};
const COST_LABEL = {
  LOW: 'низкая',
  MEDIUM: 'средняя',
  HIGH: 'высокая',
};
const VERDICT_LABEL = {
  ACCEPT: 'ПРИНЯТО',
  REVISE: 'ТРЕБУЕТ ПРАВОК',
  REJECT: 'ОТКЛОНЕНО',
};
const CONFIDENCE_LABEL = {
  HIGH: 'высокая',
  MEDIUM: 'средняя',
  LOW: 'низкая',
};

function novelyClass(v) {
  return ({ HIGH: 'high', MEDIUM: 'medium', LOW: 'low' })[v] || 'unknown';
}

function verdictClass(v) {
  return ({ ACCEPT: 'accept', REVISE: 'revise', REJECT: 'reject' })[v] || 'unknown';
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
        el('span', { class: 'here' }, 'Гипотезы'),
      ),
      el(
        'h1',
        { class: 'section-title' },
        '💡 Гипотезы — observations от neural model',
      ),
      el(
        'p',
        { class: 'section-sub' },
        'Глубокая нейронная сеть (Sonnet) просматривает обученную модель ' +
          'и формулирует 3-5 testable research hypotheses с оценкой ' +
          'экономического эффекта vs классическая практика. Второй агент — ' +
          'PhD-критик — делает adversarial peer review и выдаёт ACCEPT / ' +
          'REVISE / REJECT с построчной проверкой evidence. ' +
          'Полный цикл ~3 минуты, ~$0.14.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'hypotheses-error',
    role: 'alert',
    hidden: '',
  });

  // Run controls — model dropdown + n_hypotheses + save toggle + button.
  const modelSelect = el('select', {
    class: 'hypotheses-select mono',
    id: 'hypotheses-model-select',
    onChange: (ev) => onModelChange(ev.target.value),
  });
  const modelMeta = el('div', { class: 'hypotheses-model-meta mono' }, '');

  const nInput = el('input', {
    type: 'number',
    class: 'hypotheses-input mono',
    id: 'hypotheses-n',
    value: String(state.form.n_hypotheses),
    min: '1',
    max: '20',
    step: '1',
    'data-field': 'n_hypotheses',
  });

  const saveCheckbox = el('input', {
    type: 'checkbox',
    'data-field': 'save_to_decision_log',
    ...(state.form.save_to_decision_log ? { checked: 'checked' } : {}),
  });
  const saveLabel = el(
    'label',
    { class: 'hypotheses-save-toggle' },
    saveCheckbox,
    el('span', {}, 'Сохранить в Decision Log (тэг hypothesis_cycle)'),
  );

  const runBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn primary hypotheses-run-btn',
      onClick: () => runCycle(),
    },
    'Сгенерировать гипотезы',
  );

  const subnote = el(
    'div',
    { class: 'hypotheses-subnote' },
    'Полный цикл (генератор + adversarial peer-review) занимает ~3 минуты, ' +
      'расходует ~$0.14 на Sonnet API. LLM call нельзя прервать после старта — ' +
      'кнопку «Отменить» лучше нажимать сразу после submit.',
  );

  const formPanel = el(
    'div',
    { class: 'hypotheses-form-panel' },
    el(
      'div',
      { class: 'hypotheses-form-row' },
      el(
        'label',
        { class: 'hypotheses-field', for: modelSelect.id },
        el('span', { class: 'hypotheses-field-label' }, 'Активная модель'),
        modelSelect,
      ),
      el(
        'label',
        { class: 'hypotheses-field', for: nInput.id },
        el('span', { class: 'hypotheses-field-label' }, 'Сколько гипотез'),
        nInput,
      ),
    ),
    modelMeta,
    el(
      'div',
      { class: 'hypotheses-actions' },
      saveLabel,
      runBtn,
    ),
    subnote,
  );

  // Mount points for progress card + result panel.
  const progressMount = el('div', {
    class: 'hypotheses-progress-mount',
    'data-role': 'progress-mount',
  });
  const resultMount = el('div', {
    class: 'hypotheses-result-mount',
    'data-role': 'result-mount',
  });

  const body = el(
    'div',
    { class: 'hypotheses-body' },
    errorBanner,
    formPanel,
    progressMount,
    resultMount,
  );
  const root = el('div', { class: 'hypotheses-view' }, head, body);

  return {
    root,
    errorBanner,
    modelSelect,
    modelMeta,
    nInput,
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
    // Show "<version> · <class> · <target>" in the dropdown so the
    // operator can tell apart multiple HSLA / Q&T / fatigue models.
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
  // Re-enable run button — disabled-state is per-jobstate elsewhere.
  if (!state.job.running) elements.runBtn.disabled = false;
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
}

// ──────────────────── form read ────────────────────

function readForm() {
  if (!elements) {
    return { ...state.form };
  }
  const out = { ...state.form };
  const nVal = parseInt(elements.nInput.value, 10);
  if (!Number.isFinite(nVal) || nVal < 1 || nVal > 20) {
    throw new Error('Поле «Сколько гипотез» должно быть в диапазоне 1..20');
  }
  out.n_hypotheses = nVal;
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

  const hyps = Array.isArray(data.hypotheses) ? data.hypotheses : [];
  const reviewsById = {};
  for (const r of (data.reviews || [])) {
    if (r && r.hypothesis_id) reviewsById[r.hypothesis_id] = r;
  }
  const counts = data.verdict_counts || { ACCEPT: 0, REVISE: 0, REJECT: 0 };

  // Summary header — N hypotheses, duration, decision-log indicator.
  const headlineParts = [
    `${hyps.length} гипотез сгенерировано`,
  ];
  if (typeof data.duration_s === 'number') {
    headlineParts.push(`за ${fmt(data.duration_s, 0)} с`);
  }
  blocks.push(el('div', { class: 'hypotheses-result-headline' },
    headlineParts.join(' · '),
  ));

  if (data.decision_log_id != null) {
    blocks.push(el('div', { class: 'hypotheses-decision-saved' },
      `✔ Сохранено в Decision Log (id=${data.decision_log_id}, тэг hypothesis_cycle).`,
    ));
  }

  // Verdict-counts metric strip. Always rendered — zero-counts make
  // the absence of reviews readable rather than mysterious.
  const metricsGrid = el(
    'div',
    { class: 'hypotheses-metrics-grid' },
    metricCard('Гипотез', String(hyps.length)),
    metricCard('ACCEPT', String(counts.ACCEPT || 0), 'accept'),
    metricCard('REVISE', String(counts.REVISE || 0), 'revise'),
    metricCard('REJECT', String(counts.REJECT || 0), 'reject'),
  );
  blocks.push(metricsGrid);

  if (hyps.length === 0) {
    blocks.push(el('div', { class: 'hypotheses-empty' },
      'Получено 0 гипотез. Проверьте логи uvicorn — возможен malformed payload ' +
      'от Sonnet API или таймаут. Можно повторить запуск.',
    ));
  } else {
    for (let i = 0; i < hyps.length; i++) {
      blocks.push(renderHypothesisCard(i + 1, hyps[i], reviewsById[hyps[i].id]));
    }
  }

  elements.resultMount.replaceChildren(
    el('div', { class: 'hypotheses-result' }, ...blocks),
  );
}

function metricCard(label, value, modifier) {
  const cls = modifier
    ? `hypotheses-metric-card hypotheses-metric-${modifier}`
    : 'hypotheses-metric-card';
  return el(
    'div',
    { class: cls },
    el('div', { class: 'hypotheses-metric-label' }, label),
    el('div', { class: 'hypotheses-metric-value mono' }, value),
  );
}

function renderHypothesisCard(idx, h, review) {
  const novelty = h.novelty || 'LOW';
  const cost = h.experiment_cost_estimate || 'LOW';

  const noveltyBadge = el(
    'span',
    { class: `hypotheses-badge hypotheses-novelty-${novelyClass(novelty)}` },
    `новизна: ${NOVELTY_LABEL[novelty] || novelty}`,
  );
  const costBadge = el(
    'span',
    { class: 'hypotheses-cost' },
    `стоимость: ${COST_LABEL[cost] || cost}`,
  );

  const titleRow = el(
    'div',
    { class: 'hypotheses-card-head' },
    el(
      'div',
      { class: 'hypotheses-card-title' },
      el('span', { class: 'hypotheses-card-num' }, `${idx}.`),
      el('span', {}, ` ${h.statement || '—'}`),
    ),
    el(
      'div',
      { class: 'hypotheses-card-badges' },
      noveltyBadge,
      costBadge,
    ),
  );

  const blocks = [titleRow];

  if (h.rationale) {
    blocks.push(el('div', { class: 'hypotheses-block' },
      el('strong', {}, 'Обоснование. '), h.rationale,
    ));
  }

  // Proposed experiment — fix vs sweep two-column.
  const pe = h.proposed_experiment || {};
  const fix = pe.fix || {};
  const sweep = pe.sweep || {};
  const fixHasKeys = Object.keys(fix || {}).length > 0;
  const sweepHasKeys = Object.keys(sweep || {}).length > 0;
  if (fixHasKeys || sweepHasKeys) {
    blocks.push(el('div', { class: 'hypotheses-experiment' },
      el('strong', {}, 'Предлагаемый эксперимент.'),
      el(
        'div',
        { class: 'hypotheses-twocol' },
        el(
          'div',
          {},
          el('div', { class: 'hypotheses-twocol-label' }, 'Зафиксировать'),
          el('pre', { class: 'hypotheses-json mono' },
            JSON.stringify(fix, null, 2),
          ),
        ),
        el(
          'div',
          {},
          el('div', { class: 'hypotheses-twocol-label' }, 'Варьировать'),
          el('pre', { class: 'hypotheses-json mono' },
            JSON.stringify(sweep, null, 2),
          ),
        ),
      ),
    ));
  }

  if (h.expected_outcome) {
    blocks.push(el('div', { class: 'hypotheses-block' },
      el('strong', {}, 'Ожидаемый результат. '), h.expected_outcome,
    ));
  }

  const ei = h.economic_impact || {};
  if (ei.vs_classical_baseline || ei.estimated_saving || ei.measurement_method) {
    blocks.push(el('div', { class: 'hypotheses-block' },
      el('strong', {}, 'Экономический эффект.'),
      el('ul', { class: 'hypotheses-list' },
        el('li', {},
          el('span', {}, 'Сравнение с классикой: '),
          ei.vs_classical_baseline || '—',
        ),
        el('li', {},
          el('span', {}, 'Оценка экономии: '),
          el('strong', {}, ei.estimated_saving || '—'),
        ),
        el('li', {},
          el('span', {}, 'Метод проверки: '),
          ei.measurement_method || '—',
        ),
      ),
    ));
  }

  // Critic verdict — appears below a divider when present.
  if (review) {
    blocks.push(el('div', { class: 'hypotheses-divider' }));
    blocks.push(renderVerdictBlock(review));
  }

  blocks.push(el('div', { class: 'hypotheses-card-meta' },
    `id=${h.id || '—'}` + (Array.isArray(h.tags) && h.tags.length
      ? `  ·  теги: ${h.tags.join(', ')}` : ''),
  ));

  return el('div', { class: 'hypotheses-card' }, ...blocks);
}

function renderVerdictBlock(review) {
  const verdict = review.verdict || 'UNKNOWN';
  const blocks = [];

  const verdictBadge = el(
    'span',
    {
      class: `hypotheses-verdict-badge hypotheses-verdict-${verdictClass(verdict)}`,
    },
    `👨‍🔬 PhD-рецензия: ${VERDICT_LABEL[verdict] || verdict}`,
  );
  const conf = el(
    'span',
    { class: 'hypotheses-verdict-confidence' },
    `уверенность ${CONFIDENCE_LABEL[review.confidence] || review.confidence || '—'}`,
  );
  blocks.push(el('div', { class: 'hypotheses-verdict-row' }, verdictBadge, conf));

  if (review.summary) {
    blocks.push(el('div', { class: 'hypotheses-summary' },
      el('em', {}, review.summary),
    ));
  }

  const strengths = Array.isArray(review.strengths) ? review.strengths : [];
  const weaknesses = Array.isArray(review.weaknesses) ? review.weaknesses : [];
  if (strengths.length || weaknesses.length) {
    blocks.push(el(
      'div',
      { class: 'hypotheses-twocol' },
      el(
        'div',
        {},
        el('strong', {}, 'Сильные стороны'),
        el('ul', { class: 'hypotheses-list' },
          ...strengths.map((s) => el('li', {}, s)),
        ),
      ),
      el(
        'div',
        {},
        el('strong', {}, 'Слабые стороны'),
        el('ul', { class: 'hypotheses-list' },
          ...weaknesses.map((w) => el('li', {}, w)),
        ),
      ),
    ));
  }

  if (review.suggested_revision) {
    blocks.push(el('div', { class: 'hypotheses-suggested-revision' },
      el('strong', {}, 'Предложение правки. '), review.suggested_revision,
    ));
  }

  return el('div', { class: 'hypotheses-verdict' }, ...blocks);
}

// ──────────────────── actions ────────────────────

async function runCycle() {
  if (state.job.running) return;
  clearError();

  if (!state.selectedModelVersion) {
    showError('Сначала выберите активную модель.');
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
    n_hypotheses: form.n_hypotheses,
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
    label: 'Запрашиваю генератор + критика… (~3 мин)',
    onCancel: () => cancelCycle(),
  });

  let jobId = null;
  try {
    const submit = await apiFetch('/api/hypotheses/run', { method: 'POST', body });
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
    if (elements && elements.runBtn) elements.runBtn.disabled = false;
  }
}

async function cancelCycle() {
  // Best-effort: abort the polling loop on the client immediately, then
  // DELETE on the server. Cancellation is effective ONLY between the
  // two LLM calls — see hypotheses.py module docstring.
  const jobId = state.job.currentJobId;
  if (state.job.pollAbort) state.job.pollAbort.abort();
  if (!jobId) return;
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  } catch (err) {
    console.warn('[hypotheses] DELETE /api/jobs failed', err);
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
        // 404 is the "no models trained yet" path — we render a banner.
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
    ]);
    const items = Array.isArray(listResp.items) ? listResp.items : [];
    state.models = items;
    if (activeResp && activeResp.version) {
      state.selectedModelVersion = activeResp.version;
    } else if (items.length) {
      // Fallback: last entry mirrors the "active" sort order from system.py.
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
  state.form = { n_hypotheses: 5, save_to_decision_log: false };

  const skeleton = buildSkeleton();
  elements = skeleton;
  container.replaceChildren(skeleton.root);
  loadModels();
}
