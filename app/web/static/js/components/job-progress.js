// Job progress component + polling helper.
//
// PR 6 of the Streamlit→FastAPI migration. Backend (`app/api/jobs.py`)
// runs long-running ops (training, NSGA-II, LLM cycles) on a thread-pool
// JobStore and exposes them through `GET /api/jobs/{id}`. Frontend
// (PR 7+ views) creates a job, then calls:
//
//     const progress = renderJobProgress(container, { label: 'Обучение' });
//     try {
//       const result = await pollJob(jobId, {
//         onProgress: progress.updateProgress,
//         onMessage:  progress.updateMessage,
//       });
//       progress.markDone();
//       // … render result …
//     } catch (err) {
//       progress.markError(err.message);
//     }
//
// Public API:
//   pollJob(jobId, opts) → Promise<result>
//   renderJobProgress(container, opts) → { updateProgress, updateMessage,
//     markDone, markError, destroy }
//
// All copy is Russian per project convention (CLAUDE.md → "Языковая
// конвенция"). Numeric formatting uses tabular-nums via .mono / .num
// classes already defined in app.css.

import { el } from '../utils/dom.js';
import { ApiError } from '../api.js';

const DEFAULT_POLL_INTERVAL_MS = 2000;

// ──────────────────────────────────────────────────────────────────────
// pollJob — promise-based polling loop with cancellation support
// ──────────────────────────────────────────────────────────────────────

/** Poll a job until it resolves or rejects.
 *
 * @param {string} jobId  job id returned by a long-running endpoint
 * @param {object} [opts]
 * @param {number} [opts.interval=2000]  poll interval in ms
 * @param {(p:number)=>void} [opts.onProgress]  called when progress fraction changes
 * @param {(m:string)=>void} [opts.onMessage]   called when message string changes
 * @param {(j:object)=>void} [opts.onTick]      called on every poll regardless
 * @param {AbortSignal} [opts.signal]  abort to cancel polling (rejects with AbortError)
 * @returns {Promise<*>} resolves with `result` when status==='done',
 *                       rejects with Error(error_msg) when status==='error',
 *                       rejects with DOMException('AbortError') if signal aborts.
 */
export function pollJob(jobId, opts = {}) {
  const interval = Number(opts.interval) > 0 ? Number(opts.interval) : DEFAULT_POLL_INTERVAL_MS;
  const { onProgress, onMessage, onTick, signal } = opts;

  return new Promise((resolve, reject) => {
    let lastProgress = -1;
    let lastMessage = null;
    let timer = null;
    let cancelled = false;

    const cleanup = () => {
      if (timer != null) {
        clearTimeout(timer);
        timer = null;
      }
      if (signal) signal.removeEventListener('abort', onAbort);
    };

    const onAbort = () => {
      cancelled = true;
      cleanup();
      // DOMException with name 'AbortError' is the web-platform convention;
      // fetch() also rejects with it, so callers can `if (err.name === 'AbortError')`.
      reject(new DOMException('Job polling cancelled', 'AbortError'));
    };

    if (signal) {
      if (signal.aborted) { onAbort(); return; }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    const tick = async () => {
      if (cancelled) return;
      let job;
      try {
        const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
          headers: { Accept: 'application/json' },
        });
        if (!resp.ok) {
          // 404 on poll — treat as terminal error so caller doesn't loop forever.
          let detail = `HTTP ${resp.status}`;
          try {
            const payload = await resp.json();
            if (payload && payload.detail) detail = String(payload.detail);
          } catch (_e) { /* ignore */ }
          cleanup();
          reject(new ApiError(detail, resp.status, null));
          return;
        }
        job = await resp.json();
      } catch (err) {
        // Network glitch — keep polling; surface to console for dev.
        // If the caller wants strict "fail on first network err", they can
        // use AbortController and wrap pollJob in a timeout.
        if (cancelled) return;
        console.warn('[job-progress] network error, retrying', err);
        timer = setTimeout(tick, interval);
        return;
      }

      // Progress / message callbacks — debounced via diff so callers
      // don't get woken up for redundant identical values.
      if (typeof onProgress === 'function' && typeof job.progress === 'number'
          && job.progress !== lastProgress) {
        lastProgress = job.progress;
        try { onProgress(job.progress); } catch (e) { console.error(e); }
      }
      if (typeof onMessage === 'function' && job.message !== lastMessage) {
        lastMessage = job.message || '';
        try { onMessage(lastMessage); } catch (e) { console.error(e); }
      }
      if (typeof onTick === 'function') {
        try { onTick(job); } catch (e) { console.error(e); }
      }

      if (job.status === 'done') {
        cleanup();
        resolve(job.result);
      } else if (job.status === 'error') {
        cleanup();
        reject(new Error(job.error || 'Задача завершилась с ошибкой'));
      } else {
        // pending | running — keep going.
        timer = setTimeout(tick, interval);
      }
    };

    // Kick off immediately so progress shows up quickly; subsequent polls
    // honour the interval.
    tick();
  });
}

// ──────────────────────────────────────────────────────────────────────
// renderJobProgress — DOM component
// ──────────────────────────────────────────────────────────────────────

/** Build a progress card with label + bar + message + cancel button.
 *
 * The component owns its DOM only — it does NOT issue API calls. Callers
 * pair it with pollJob and wire the callbacks (or use updateProgress /
 * updateMessage manually). This separation keeps the component reusable
 * for non-job progress (e.g. multi-step client-side imports) too.
 *
 * @param {Element} container  parent element; the component appends itself
 * @param {object} [opts]
 * @param {string} [opts.label='Идёт обработка']  visible label
 * @param {()=>void} [opts.onCancel]  optional cancel callback; if provided,
 *                                    a "Отменить" button appears
 * @returns {{
 *   element: HTMLElement,
 *   updateProgress: (p:number)=>void,
 *   updateMessage: (m:string)=>void,
 *   markDone: (label?:string)=>void,
 *   markError: (msg:string)=>void,
 *   destroy: ()=>void,
 * }}
 */
export function renderJobProgress(container, opts = {}) {
  const label = opts.label || 'Идёт обработка';
  const onCancel = typeof opts.onCancel === 'function' ? opts.onCancel : null;

  const labelEl = el('div', { class: 'job-progress-label' }, label);
  const percentEl = el('span', { class: 'job-progress-percent num' }, '0%');
  const messageEl = el('div', { class: 'job-progress-message' }, '');
  const fillEl = el('div', { class: 'job-progress-fill' });
  const barEl = el('div', { class: 'job-progress-bar' }, fillEl);

  const headerChildren = [labelEl, percentEl];
  let cancelBtn = null;
  if (onCancel) {
    cancelBtn = el(
      'button',
      {
        class: 'btn job-progress-cancel',
        type: 'button',
        onClick: () => {
          cancelBtn.disabled = true;
          try { onCancel(); } catch (e) { console.error(e); }
        },
      },
      'Отменить',
    );
    headerChildren.push(cancelBtn);
  }

  const headerEl = el('div', { class: 'job-progress-header' }, ...headerChildren);
  const root = el(
    'div',
    {
      class: 'job-progress',
      role: 'status',
      'aria-live': 'polite',
      'aria-busy': 'true',
    },
    headerEl,
    barEl,
    messageEl,
  );

  container.appendChild(root);

  let destroyed = false;

  const updateProgress = (p) => {
    if (destroyed) return;
    const pct = Math.max(0, Math.min(100, Math.round(Number(p) * 100)));
    fillEl.style.width = `${pct}%`;
    percentEl.textContent = `${pct}%`;
  };

  const updateMessage = (m) => {
    if (destroyed) return;
    messageEl.textContent = m || '';
  };

  const markDone = (newLabel) => {
    if (destroyed) return;
    fillEl.style.width = '100%';
    percentEl.textContent = '100%';
    root.classList.add('done');
    root.setAttribute('aria-busy', 'false');
    if (newLabel) labelEl.textContent = newLabel;
    if (cancelBtn) cancelBtn.hidden = true;
  };

  const markError = (msg) => {
    if (destroyed) return;
    root.classList.add('error');
    root.setAttribute('aria-busy', 'false');
    labelEl.textContent = 'Ошибка';
    messageEl.textContent = msg || 'Неизвестная ошибка';
    if (cancelBtn) cancelBtn.hidden = true;
  };

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    if (root.parentNode) root.parentNode.removeChild(root);
  };

  return {
    element: root,
    updateProgress,
    updateMessage,
    markDone,
    markError,
    destroy,
  };
}
