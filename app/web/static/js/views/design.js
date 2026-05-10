// Tab 01 — Дизайн сплава. NSGA-II inverse design, ~40 s through polling.
//
// Flow:
//   GET  /api/system/models                  → {items, count}
//   GET  /api/system/models/active           → model meta (404 if none)
//   GET  /api/prices/active                  → seed price snapshot
//   POST /api/design/run {model_version, target, ...}
//                                            → {job_id}
//   GET  /api/jobs/{job_id} (polled)         → result {candidates, baseline, ...}
//   DELETE /api/jobs/{job_id}                → cooperative cancellation
//   POST /api/prices/upload (via price-editor) → save user snapshot
//
// PR 7 of the Streamlit→FastAPI migration. See
// docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';
import { pollJob, renderJobProgress } from '../components/job-progress.js';
import { renderParetoChart } from '../charts/pareto.js';
import { priceEditor } from '../components/price-editor.js';
import { buildCsv, downloadCsv } from '../utils/csv.js';

// Module-scoped state. Re-init() resets it.
const state = {
  models: [],
  selectedVersion: null,
  priceSnapshot: null,
  paretoSnapshot: null,
  selectedCandidateIdx: null,
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

function activeModel() {
  if (!state.selectedVersion) return null;
  return state.models.find((m) => m.version === state.selectedVersion) || null;
}

function isHsla(model) {
  return model && model.steel_class === 'pipe_hsla';
}

function fmt(v, d = 1) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(d);
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
        el('span', { class: 'here' }, 'Дизайн сплава'),
      ),
      el('h1', { class: 'section-title' }, 'Дизайн сплава — поиск состава под целевые свойства'),
      el(
        'p',
        { class: 'section-sub' },
        'NSGA-II multi-objective optimization над composition+process пространством. ' +
          'Модель — property predictor; ferroalloy snapshot — €-predictor. ' +
          'Pareto-фронт показывает trade-off между прочностью, стоимостью и неопределённостью прогноза.',
      ),
    ),
    el('div', { class: 'section-actions' }),
  );

  const errorBanner = el('div', {
    class: 'design-error',
    role: 'alert',
    hidden: '',
  });

  const classBanner = el('div', {
    class: 'design-class-banner',
    role: 'note',
    hidden: '',
  });

  const emptyBanner = el(
    'div',
    { class: 'design-empty', hidden: '' },
    el(
      'p',
      {},
      'Нет обученных HSLA-моделей. Сначала обучите модель класса ',
      el('strong', {}, 'pipe_hsla'),
      ' во вкладке «Обучение модели».',
    ),
  );

  // Inputs panel — model selector + target ranges + NSGA-II params + price editor.
  const modelSelect = el('select', {
    class: 'history-select mono',
    id: 'design-model-select',
    onChange: (ev) => {
      state.selectedVersion = ev.target.value;
      updateClassBanner();
    },
  });

  const modelMeta = el('div', { class: 'design-model-meta mono' }, '');

  const modelRow = el(
    'div',
    { class: 'design-model-row' },
    el(
      'label',
      { class: 'design-model-cell', for: 'design-model-select' },
      el('span', { class: 'design-model-label' }, 'Активная модель'),
      modelSelect,
    ),
    modelMeta,
  );

  // Target / constraints inputs.
  const ytMinInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-yt-min',
    value: '485',
    min: '380',
    max: '800',
    step: '5',
  });
  const ytMaxInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-yt-max',
    value: '580',
    min: '400',
    max: '900',
    step: '5',
  });
  const cevMaxInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-cev-max',
    value: '0.43',
    min: '0.30',
    max: '0.60',
    step: '0.01',
  });
  const pcmMaxInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-pcm-max',
    value: '0.22',
    min: '0.15',
    max: '0.35',
    step: '0.01',
  });
  const popSizeInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-pop-size',
    value: '80',
    min: '10',
    max: '200',
    step: '10',
  });
  const nGenInput = el('input', {
    type: 'number',
    class: 'design-input mono',
    id: 'design-n-gen',
    value: '60',
    min: '1',
    max: '200',
    step: '5',
  });
  const includeCostInput = el('input', {
    type: 'checkbox',
    id: 'design-include-cost',
    checked: '',
  });
  const costModeSelect = el(
    'select',
    { class: 'history-select mono', id: 'design-cost-mode' },
    el('option', { value: 'full', selected: '' }, 'full (alloy + scrap)'),
    el('option', { value: 'incremental' }, 'incremental (alloy only)'),
  );

  const targetGrid = el(
    'div',
    { class: 'design-form-grid' },
    el(
      'label',
      { class: 'design-field', for: ytMinInput.id },
      el('span', { class: 'design-field-label' }, 'σт минимум, МПа'),
      ytMinInput,
    ),
    el(
      'label',
      { class: 'design-field', for: ytMaxInput.id },
      el('span', { class: 'design-field-label' }, 'σт максимум, МПа'),
      ytMaxInput,
    ),
    el(
      'label',
      { class: 'design-field', for: cevMaxInput.id },
      el('span', { class: 'design-field-label' }, 'CEV(IIW) max'),
      cevMaxInput,
    ),
    el(
      'label',
      { class: 'design-field', for: pcmMaxInput.id },
      el('span', { class: 'design-field-label' }, 'Pcm max'),
      pcmMaxInput,
    ),
    el(
      'label',
      { class: 'design-field', for: popSizeInput.id },
      el('span', { class: 'design-field-label' }, 'Population size'),
      popSizeInput,
    ),
    el(
      'label',
      { class: 'design-field', for: nGenInput.id },
      el('span', { class: 'design-field-label' }, 'Generations'),
      nGenInput,
    ),
    el(
      'label',
      { class: 'design-field', for: costModeSelect.id },
      el('span', { class: 'design-field-label' }, 'Режим cost'),
      costModeSelect,
    ),
    el(
      'label',
      { class: 'design-field design-field-checkbox', for: includeCostInput.id },
      includeCostInput,
      el('span', { class: 'design-field-label' }, 'Учитывать стоимость в оптимизации'),
    ),
  );

  // Price editor mount point — populated after seed snapshot loads.
  const priceEditorMount = el('div', { class: 'design-price-editor' });
  const priceToggle = el(
    'button',
    {
      type: 'button',
      class: 'design-collapse-toggle',
      'aria-expanded': 'true',
      onClick: () => togglePriceEditor(),
    },
    '▾ Прайс материалов',
  );
  const priceWrap = el(
    'div',
    { class: 'design-price-wrap' },
    priceToggle,
    priceEditorMount,
  );

  const runBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn primary',
      onClick: () => runDesign(),
    },
    'Запустить NSGA-II',
  );

  const actions = el('div', { class: 'design-actions' }, runBtn);

  const progressMount = el('div', { class: 'design-progress-mount' });

  const formPanel = el(
    'div',
    { class: 'design-form-panel' },
    modelRow,
    targetGrid,
    priceWrap,
    actions,
    progressMount,
  );

  // Result panel — Pareto chart + table + breakdown.
  const summaryMount = el('div', { class: 'design-result-summary' });
  const chartMount = el('div', { class: 'design-pareto-chart' });
  const tableMount = el('div', { class: 'design-candidate-table-wrap' });
  const breakdownMount = el('div', { class: 'design-breakdown-panel' });
  const downloadBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn',
      onClick: () => downloadCandidates(),
      hidden: '',
    },
    'Скачать CSV',
  );
  const resultActions = el('div', { class: 'design-result-actions' }, downloadBtn);
  const resultPanel = el(
    'div',
    { class: 'design-result-panel', hidden: '' },
    summaryMount,
    chartMount,
    tableMount,
    breakdownMount,
    resultActions,
  );

  const body = el(
    'div',
    { class: 'design-body' },
    errorBanner,
    classBanner,
    emptyBanner,
    formPanel,
    resultPanel,
  );

  const root = el('div', { class: 'design-view' }, head, body);

  return {
    root,
    errorBanner,
    classBanner,
    emptyBanner,
    modelSelect,
    modelMeta,
    formPanel,
    ytMinInput,
    ytMaxInput,
    cevMaxInput,
    pcmMaxInput,
    popSizeInput,
    nGenInput,
    includeCostInput,
    costModeSelect,
    priceEditorMount,
    priceToggle,
    runBtn,
    progressMount,
    resultPanel,
    summaryMount,
    chartMount,
    tableMount,
    breakdownMount,
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

function showEmpty() {
  if (!elements) return;
  elements.emptyBanner.hidden = false;
  elements.formPanel.hidden = true;
}
function hideEmpty() {
  if (!elements) return;
  elements.emptyBanner.hidden = true;
  elements.formPanel.hidden = false;
}

function togglePriceEditor() {
  if (!elements) return;
  const t = elements.priceToggle;
  const expanded = t.getAttribute('aria-expanded') === 'true';
  if (expanded) {
    elements.priceEditorMount.hidden = true;
    t.textContent = '▸ Прайс материалов';
    t.setAttribute('aria-expanded', 'false');
  } else {
    elements.priceEditorMount.hidden = false;
    t.textContent = '▾ Прайс материалов';
    t.setAttribute('aria-expanded', 'true');
  }
}

// ──────────────────────────────────────────────────────────────────────
// model select rendering + class gate
// ──────────────────────────────────────────────────────────────────────

function renderModelSelect() {
  if (!elements) return;
  const sel = elements.modelSelect;
  sel.replaceChildren();
  for (const m of state.models) {
    const opt = el(
      'option',
      { value: m.version },
      `${m.version} · ${m.steel_class}`,
    );
    if (m.version === state.selectedVersion) opt.selected = true;
    sel.append(opt);
  }
  updateClassBanner();
}

function updateClassBanner() {
  if (!elements) return;
  const m = activeModel();
  if (!m) {
    elements.modelMeta.textContent = '';
    elements.classBanner.hidden = true;
    elements.runBtn.disabled = true;
    return;
  }
  elements.modelMeta.textContent = `Класс: ${m.steel_class} · target: ${m.target}`;
  if (!isHsla(m)) {
    const labels = {
      en10083_qt: 'EN 10083-2 Q&T (carbon)',
      fatigue_carbon_steel: 'Carbon Fatigue (Agrawal NIMS)',
    };
    const label = labels[m.steel_class] || m.steel_class;
    elements.classBanner.replaceChildren(
      el(
        'div',
        {},
        'ℹ️ Inverse design пока работает только для ',
        el('strong', {}, 'Pipe HSLA'),
        '. Активный класс — ',
        el('strong', {}, label),
        ', для него используйте вкладку «Прогноз». ' +
          'Поддержка inverse design для других классов запланирована на v2.',
      ),
    );
    elements.classBanner.hidden = false;
    elements.runBtn.disabled = true;
  } else {
    elements.classBanner.hidden = true;
    elements.runBtn.disabled = false;
  }
}

// ──────────────────────────────────────────────────────────────────────
// run / cancel
// ──────────────────────────────────────────────────────────────────────

function readRequestBody() {
  const m = activeModel();
  if (!m) throw new Error('Модель не выбрана');
  const ytMin = parseFloat(elements.ytMinInput.value);
  const ytMax = parseFloat(elements.ytMaxInput.value);
  if (!(ytMin < ytMax)) {
    throw new Error('σт минимум должен быть меньше максимума');
  }
  const includeCost = !!elements.includeCostInput.checked;
  const body = {
    model_version: m.version,
    target: { min: ytMin, max: ytMax },
    hard_constraints: {
      cev_iiw_max: parseFloat(elements.cevMaxInput.value),
      pcm_max: parseFloat(elements.pcmMaxInput.value),
    },
    population_size: parseInt(elements.popSizeInput.value, 10),
    n_generations: parseInt(elements.nGenInput.value, 10),
    include_cost: includeCost,
    cost_mode: elements.costModeSelect.value,
  };
  if (includeCost && state.priceSnapshot) {
    body.price_snapshot = state.priceSnapshot;
  }
  return body;
}

async function runDesign() {
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
  elements.runBtn.textContent = 'Идёт оптимизация…';
  elements.resultPanel.hidden = true;
  elements.progressMount.replaceChildren();

  const abort = new AbortController();
  state.pollAbort = abort;

  const progress = renderJobProgress(elements.progressMount, {
    label: 'NSGA-II оптимизация…',
    onCancel: () => cancelDesign(),
  });

  let jobId = null;
  try {
    const submit = await apiFetch('/api/design/run', {
      method: 'POST',
      body,
    });
    jobId = submit.job_id;
    state.currentJobId = jobId;
    if (!jobId) {
      throw new Error('Сервер не вернул job_id');
    }
    progress.updateMessage(`job_id=${jobId.slice(0, 8)}…`);

    const result = await pollJob(jobId, {
      interval: 1500,
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
    elements.runBtn.textContent = 'Запустить NSGA-II';
  }
}

async function cancelDesign() {
  // Best-effort cancel — abort polling on client + DELETE /api/jobs/{id}
  // on server. Polling abort is the immediate UX response; server cancel
  // marks the flag (cooperative) and may free a queued worker.
  const jobId = state.currentJobId;
  if (state.pollAbort) state.pollAbort.abort();
  if (!jobId) return;
  try {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  } catch (err) {
    // Cancellation failures are non-fatal — the user already sees the
    // local UI flip; surface to console for debug.
    console.warn('[design] DELETE /api/jobs failed', err);
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

  const cands = data.candidates || [];
  const currency = data.cost_currency || 'EUR';

  // Summary metrics row: four metrics + an expander «Причины отсева»
  // when rejection_summary is non-empty. Plus two baseline cells —
  // useful context for the chart.
  const inSpec = cands.filter((c) => c.in_spec).length;
  const oodN = cands.filter((c) => c.is_ood).length;
  const warnN = cands.filter(
    (c) => (c.validation && c.validation.overall) === 'PASS_WITH_WARNINGS',
  ).length;
  const baseProp = data.baseline?.predicted_property;
  const baseCost = data.baseline?.cost_per_ton;
  const summaryChildren = [
    el(
      'div',
      { class: 'design-summary-grid' },
      summaryCell('Кандидатов', String(cands.length)),
      summaryCell('In-spec', String(inSpec)),
      summaryCell('OOD', String(oodN)),
      summaryCell('С warnings', String(warnN)),
      summaryCell('Отсеяно валидатором', String(data.n_rejected ?? 0)),
      summaryCell('Baseline σт, МПа', fmt(baseProp, 1)),
      summaryCell(`Baseline cost, ${currency}/т`, fmt(baseCost, 0)),
    ),
  ];
  const rejectionPanel = renderRejectionSummary(data.rejection_summary);
  if (rejectionPanel) summaryChildren.push(rejectionPanel);
  elements.summaryMount.replaceChildren(...summaryChildren);

  // Pareto chart.
  state.paretoSnapshot = cands;
  state.selectedCandidateIdx = cands[0]?.idx ?? null;
  elements.chartMount.replaceChildren();
  const chartHost = el('div', { class: 'design-pareto-host' });
  elements.chartMount.append(chartHost);
  renderParetoChart(chartHost, cands, {
    title: 'Pareto-фронт σт × cost',
    xLabel: `Стоимость, ${currency}/т`,
    yLabel: 'σт, МПа',
    currency,
    selectedIdx: state.selectedCandidateIdx,
    onSelect: (idx, _cand) => selectCandidate(idx),
  }).catch((err) => {
    chartHost.replaceChildren(
      el('div', { class: 'design-chart-fallback' }, `Не удалось построить chart: ${err.message}`),
    );
  });

  // Candidate table — Top-K (10) with click-to-select.
  renderCandidateTable(cands.slice(0, 10), currency);

  // Breakdown panel for selected candidate.
  renderBreakdown(cands.find((c) => c.idx === state.selectedCandidateIdx) || cands[0], currency);

  elements.downloadBtn.hidden = cands.length === 0;
}

function summaryCell(label, value) {
  return el(
    'div',
    { class: 'design-summary-cell' },
    el('span', { class: 'design-summary-label' }, label),
    el('span', { class: 'design-summary-value mono' }, value),
  );
}

// If validate_batch reported any rejection reasons, surface them in a
// collapsible panel. Closed by default — the metric "Отсеяно валидатором"
// is the always-visible summary; this expander is the drill-down.
function renderRejectionSummary(summary) {
  if (!summary || typeof summary !== 'object') return null;
  const entries = Object.entries(summary).filter(([, v]) => Number(v) > 0);
  if (entries.length === 0) return null;
  const total = entries.reduce((s, [, v]) => s + Number(v), 0);
  const details = el(
    'details',
    { class: 'design-rejection-summary' },
    el(
      'summary',
      { class: 'design-rejection-summary-header' },
      `Причины отсева (${total})`,
    ),
    el(
      'ul',
      { class: 'design-rejection-list' },
      ...entries.map(([reason, count]) =>
        el(
          'li',
          { class: 'design-rejection-row' },
          el('span', { class: 'design-rejection-reason' }, reason),
          el('span', { class: 'design-rejection-count mono' }, String(count)),
        ),
      ),
    ),
  );
  return details;
}

function renderCandidateTable(cands, currency) {
  if (!elements) return;
  const headers = ['#', 'σт, МПа', '90% CI', `Cost, ${currency}/т`, 'Δ vs baseline', 'OOD', 'Status'];
  const thead = el(
    'thead',
    {},
    el('tr', {}, ...headers.map((h, i) => el('th', { class: i === 1 || i === 3 || i === 4 ? 'right' : '' }, h))),
  );
  const tbody = el('tbody');
  for (const c of cands) {
    const isSelected = c.idx === state.selectedCandidateIdx;
    const tr = el(
      'tr',
      {
        class: 'design-cand-row' + (isSelected ? ' selected' : '') + (c.is_ood ? ' is-ood' : ''),
        'data-idx': String(c.idx),
        onClick: () => selectCandidate(c.idx),
      },
      el('td', { class: 'design-rank mono' }, String(c.rank)),
      el(
        'td',
        { class: 'right mono' },
        `${fmt(c.predicted_mean, 1)} ± ${fmt(c.ci_half_width, 1)}`,
      ),
      el('td', { class: 'mono' }, `[${fmt(c.predicted_q05, 1)}, ${fmt(c.predicted_q95, 1)}]`),
      el('td', { class: 'right mono' }, fmt(c.cost_eur_per_t, 0)),
      el(
        'td',
        { class: 'right mono' },
        c.delta_property_vs_baseline != null
          ? `${c.delta_property_vs_baseline >= 0 ? '+' : ''}${fmt(c.delta_property_vs_baseline, 1)}`
          : '—',
      ),
      el(
        'td',
        {},
        c.is_ood
          ? el('span', { class: 'design-badge ood' }, 'OOD')
          : el('span', { class: 'design-badge ok' }, 'in-range'),
      ),
      el(
        'td',
        {},
        statusBadge(c.validation?.overall, c.in_spec),
      ),
    );
    tbody.append(tr);
  }
  elements.tableMount.replaceChildren(
    el('table', { class: 'design-candidate-table' }, thead, tbody),
  );
}

function statusBadge(overall, inSpec) {
  if (overall === 'FAIL') return el('span', { class: 'design-badge fail' }, 'fail');
  if (overall === 'PASS_WITH_WARNINGS') {
    return el('span', { class: 'design-badge warn' }, 'warn');
  }
  if (overall === 'PASS') return el('span', { class: 'design-badge ok' }, 'pass');
  return el('span', { class: 'design-badge ' + (inSpec ? 'ok' : 'fail') }, inSpec ? 'in-spec' : '—');
}

function selectCandidate(idx) {
  if (!elements) return;
  state.selectedCandidateIdx = idx;
  const cands = state.paretoSnapshot || [];
  const cand = cands.find((c) => c.idx === idx);
  // Re-render chart so highlight outline updates.
  if (cand) {
    const chartHost = elements.chartMount.querySelector('.design-pareto-host');
    if (chartHost) {
      const currency = state.lastResult?.cost_currency || 'EUR';
      renderParetoChart(chartHost, cands, {
        title: 'Pareto-фронт σт × cost',
        xLabel: `Стоимость, ${currency}/т`,
        yLabel: 'σт, МПа',
        currency,
        selectedIdx: idx,
        onSelect: (i) => selectCandidate(i),
      }).catch(() => {});
    }
  }
  // Refresh table classes.
  for (const row of elements.tableMount.querySelectorAll('tr.design-cand-row')) {
    row.classList.toggle('selected', row.dataset.idx === String(idx));
  }
  renderBreakdown(cand, state.lastResult?.cost_currency || 'EUR');
}

function renderBreakdown(cand, currency) {
  if (!elements) return;
  if (!cand) {
    elements.breakdownMount.replaceChildren();
    return;
  }
  const breakdown = cand.breakdown || [];
  const composition = cand.composition || {};
  const processing = cand.processing || {};
  const derived = cand.derived || {};

  const compRows = Object.entries(composition)
    .filter(([_, v]) => Math.abs(v) > 1e-4)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) =>
      el(
        'div',
        { class: 'design-kv-line mono' },
        el('span', { class: 'design-kv-key' }, k),
        el('span', { class: 'design-kv-val' }, fmt(v, 4)),
      ),
    );

  const procRows = Object.entries(processing).map(([k, v]) =>
    el(
      'div',
      { class: 'design-kv-line mono' },
      el('span', { class: 'design-kv-key' }, k),
      el('span', { class: 'design-kv-val' }, fmt(v, 2)),
    ),
  );

  const derivedRows = Object.entries(derived).map(([k, v]) =>
    el(
      'div',
      { class: 'design-kv-line mono' },
      el('span', { class: 'design-kv-key' }, k),
      el('span', { class: 'design-kv-val' }, fmt(v, 4)),
    ),
  );

  let breakdownTable = null;
  if (breakdown.length > 0) {
    const total = breakdown.reduce((s, b) => s + Number(b.contribution_per_ton || 0), 0) || 1;
    const thead = el(
      'thead',
      {},
      el(
        'tr',
        {},
        el('th', {}, 'Материал'),
        el('th', { class: 'right' }, 'Масса, кг/т'),
        el('th', { class: 'right' }, `Цена, ${currency}/кг`),
        el('th', { class: 'right' }, `Вклад, ${currency}/т`),
        el('th', { class: 'right' }, 'Доля, %'),
      ),
    );
    const tbody = el(
      'tbody',
      {},
      ...breakdown.map((b) =>
        el(
          'tr',
          {},
          el('td', { class: 'design-bd-mat mono' }, b.material_id),
          el('td', { class: 'right mono' }, fmt(b.mass_kg_per_ton_steel, 2)),
          el('td', { class: 'right mono' }, fmt(b.price_per_kg, 2)),
          el('td', { class: 'right mono' }, fmt(b.contribution_per_ton, 1)),
          el(
            'td',
            { class: 'right mono' },
            fmt((Number(b.contribution_per_ton || 0) / total) * 100, 1),
          ),
        ),
      ),
    );
    breakdownTable = el('table', { class: 'design-breakdown-table' }, thead, tbody);
  }

  elements.breakdownMount.replaceChildren(
    el(
      'div',
      { class: 'design-breakdown-header' },
      el('span', { class: 'design-breakdown-title' }, `Кандидат #${cand.rank}`),
      el(
        'span',
        { class: 'design-breakdown-meta mono' },
        `σт=${fmt(cand.predicted_mean, 1)} МПа · cost=${fmt(cand.cost_eur_per_t, 0)} ${currency}/т`,
      ),
    ),
    el(
      'div',
      { class: 'design-breakdown-grid' },
      el(
        'div',
        { class: 'design-breakdown-col' },
        el('div', { class: 'design-breakdown-col-title' }, 'Композиция (%)'),
        ...compRows,
      ),
      el(
        'div',
        { class: 'design-breakdown-col' },
        el('div', { class: 'design-breakdown-col-title' }, 'Обработка'),
        ...procRows,
      ),
      el(
        'div',
        { class: 'design-breakdown-col' },
        el('div', { class: 'design-breakdown-col-title' }, 'Производные'),
        ...derivedRows,
      ),
    ),
    breakdownTable
      ? el(
          'div',
          { class: 'design-breakdown-cost' },
          el('div', { class: 'design-breakdown-col-title' }, 'Себестоимость по материалам'),
          breakdownTable,
        )
      : null,
  );
}

// ──────────────────────────────────────────────────────────────────────
// CSV export
// ──────────────────────────────────────────────────────────────────────

function downloadCandidates() {
  const cands = state.paretoSnapshot || [];
  if (cands.length === 0) return;
  const currency = state.lastResult?.cost_currency || 'EUR';

  // Headers: meta columns + composition keys + processing keys.
  const compKeys = Array.from(
    new Set(cands.flatMap((c) => Object.keys(c.composition || {}))),
  );
  const procKeys = Array.from(
    new Set(cands.flatMap((c) => Object.keys(c.processing || {}))),
  );
  const headers = [
    'rank', 'idx',
    'predicted_mean', 'predicted_q05', 'predicted_q95', 'ci_half_width',
    `cost_${currency}_per_t`,
    'delta_property_vs_baseline', 'delta_cost_vs_baseline',
    'in_spec', 'is_ood', 'validation_overall',
    ...compKeys, ...procKeys,
    'cev_iiw', 'pcm', 'cen',
    'breakdown_summary',
  ];

  const rows = cands.map((c) => {
    const breakdownSummary = (c.breakdown || [])
      .map((b) => `${b.material_id}=${Number(b.contribution_per_ton || 0).toFixed(1)}`)
      .join(';');
    const row = [
      c.rank,
      c.idx,
      fmt(c.predicted_mean, 2),
      fmt(c.predicted_q05, 2),
      fmt(c.predicted_q95, 2),
      fmt(c.ci_half_width, 2),
      fmt(c.cost_eur_per_t, 1),
      c.delta_property_vs_baseline != null ? fmt(c.delta_property_vs_baseline, 2) : '',
      c.delta_cost_vs_baseline != null ? fmt(c.delta_cost_vs_baseline, 1) : '',
      c.in_spec ? '1' : '0',
      c.is_ood ? '1' : '0',
      c.validation?.overall || '',
      ...compKeys.map((k) => fmt((c.composition || {})[k], 4)),
      ...procKeys.map((k) => fmt((c.processing || {})[k], 2)),
      fmt((c.derived || {}).cev_iiw, 4),
      fmt((c.derived || {}).pcm, 4),
      fmt((c.derived || {}).cen, 4),
      breakdownSummary,
    ];
    return row;
  });

  const csv = buildCsv(headers, rows);
  const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  downloadCsv(`design-candidates-${ts}.csv`, csv);
}

// ──────────────────────────────────────────────────────────────────────
// bootstrap
// ──────────────────────────────────────────────────────────────────────

async function loadAll() {
  if (!elements) return;
  state.loading = true;
  clearError();
  try {
    const [modelsResp, activeResp, snapResp] = await Promise.all([
      apiFetch('/api/system/models'),
      apiFetch('/api/system/models/active').catch((err) => {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }),
      apiFetch('/api/prices/active'),
    ]);

    state.models = Array.isArray(modelsResp.items) ? modelsResp.items : [];

    if (state.models.length === 0) {
      showEmpty();
      return;
    }
    hideEmpty();

    // Default to active (latest) model; if non-HSLA, the class banner
    // handles the "use Прогноз tab" message.
    state.selectedVersion = activeResp
      ? activeResp.version
      : state.models[state.models.length - 1].version;

    // Prefer an HSLA model on first paint when available. The model
    // list is sorted by name so the user sees the latest model regardless
    // of class; the class banner surfaces the "non-HSLA" guard.
    state.priceSnapshot = snapResp;

    renderModelSelect();

    // Mount price editor against the seed snapshot.
    elements.priceEditorMount.replaceChildren();
    const editor = priceEditor(elements.priceEditorMount, {
      snapshot: snapResp,
      onChange: (snap) => {
        state.priceSnapshot = snap;
      },
      onSave: (savedPath) => {
        // Best-effort UX: surface the saved path; UI banner shown by
        // editor itself.
        console.info('[design] price snapshot saved', savedPath);
      },
    });
    state.priceEditor = editor;
  } catch (err) {
    const detail = err instanceof ApiError ? err.message : String(err);
    showError(`Не удалось загрузить состояние: ${detail}`);
  } finally {
    state.loading = false;
  }
}

/** Router entry-point. Called once per session with the section[data-view]
 *  container. Replaces placeholder content with the real Design view. */
export function init(container) {
  // Reset module state defensively.
  state.models = [];
  state.selectedVersion = null;
  state.priceSnapshot = null;
  state.paretoSnapshot = null;
  state.selectedCandidateIdx = null;
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
