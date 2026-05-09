// Price editor component — ferroalloy table with inline edit, YAML upload,
// and snapshot save (POST /api/prices/upload).
//
// PR 7 of the Streamlit→FastAPI migration. Streamlit parity reference:
// app/frontend/app.py lines 261-338 (`with st.expander("💰 Прайс…")`).
// Streamlit uses `st.data_editor` with column_config; we replicate the
// in-place edit affordance with vanilla `<input>` cells + dirty-tracking.
//
// Public API:
//   priceEditor(container, opts) -> { getSnapshot, setSnapshot, destroy }
//
// `opts.snapshot`  — initial snapshot dict (shape from /api/prices/active)
// `opts.onChange(snapshot)` — fired on every successful in-place edit
//                             (on blur/Enter for numeric inputs, change
//                             event for selects/text). Fired AFTER the
//                             local snapshot mirror is updated; safe to read.
// `opts.onSave(savedPath)`  — fired after POST /api/prices/upload returns
//                             OK; receives the on-disk relative path.

import { apiFetch, ApiError } from '../api.js';
import { el } from '../utils/dom.js';

// Single source of truth for material kinds — kept in sync with
// cost_model._validate_material_dict.
const KINDS = ['base', 'ferroalloy', 'pure'];

function elementContentToString(content) {
  if (!content || typeof content !== 'object') return '';
  // Format: "Mn=0.80;Fe=0.20" — matches the column_config hint Streamlit
  // shows so power users have visual continuity.
  return Object.entries(content)
    .map(([k, v]) => `${k}=${Number(v).toFixed(4).replace(/0+$/, '').replace(/\.$/, '')}`)
    .join(';');
}

function parseElementContent(text) {
  // Permissive parse — accept comma OR semicolon, trim spaces. Returns
  // a {Element: fraction} dict or throws Error with a Russian message
  // for the UI banner.
  const out = {};
  if (text == null || String(text).trim() === '') {
    throw new Error('element_content пустой');
  }
  const tokens = String(text).split(/[,;]+/).map((s) => s.trim()).filter(Boolean);
  for (const tok of tokens) {
    const m = tok.match(/^([A-Za-z]+)\s*[=:]\s*([0-9.]+)$/);
    if (!m) throw new Error(`Не распознан токен: «${tok}» (ожидаю «Mn=0.80»)`);
    const elem = m[1];
    const frac = parseFloat(m[2]);
    if (!Number.isFinite(frac) || frac < 0) {
      throw new Error(`${elem}: некорректная доля «${m[2]}»`);
    }
    out[elem] = frac;
  }
  const sum = Object.values(out).reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1.0) > 0.02) {
    throw new Error(
      `сумма element_content = ${sum.toFixed(3)}, должна быть ≈ 1.0 (±0.02)`,
    );
  }
  return out;
}

function cloneSnapshot(snapshot) {
  // Deep-ish clone — materials are flat dicts so JSON round-trip suffices.
  return JSON.parse(JSON.stringify(snapshot || {}));
}

export function priceEditor(container, opts = {}) {
  const onChange = typeof opts.onChange === 'function' ? opts.onChange : null;
  const onSave = typeof opts.onSave === 'function' ? opts.onSave : null;
  let snapshot = cloneSnapshot(opts.snapshot || { materials: {}, currency: 'EUR' });

  // ──────────────────────────────────────────────────────────────────
  // DOM scaffold
  // ──────────────────────────────────────────────────────────────────
  const errorBanner = el('div', {
    class: 'price-editor-error',
    role: 'alert',
    hidden: '',
  });

  const meta = el(
    'div',
    { class: 'price-editor-meta' },
    el(
      'div',
      { class: 'price-editor-meta-cell' },
      el('span', { class: 'price-editor-meta-label' }, 'Валюта'),
      el('span', { class: 'price-editor-meta-value mono', 'data-bind': 'currency' }, snapshot.currency || '—'),
    ),
    el(
      'div',
      { class: 'price-editor-meta-cell' },
      el('span', { class: 'price-editor-meta-label' }, 'Дата'),
      el('span', { class: 'price-editor-meta-value mono', 'data-bind': 'date' }, snapshot.date || '—'),
    ),
    el(
      'div',
      { class: 'price-editor-meta-cell' },
      el('span', { class: 'price-editor-meta-label' }, 'Материалов'),
      el('span', { class: 'price-editor-meta-value mono', 'data-bind': 'count' }, String(Object.keys(snapshot.materials || {}).length)),
    ),
  );

  const tableBody = el('tbody');
  const table = el(
    'table',
    { class: 'price-editor-table' },
    el(
      'thead',
      {},
      el(
        'tr',
        {},
        el('th', {}, 'ID'),
        el('th', {}, 'kind'),
        el('th', { class: 'right' }, 'price/kg'),
        el('th', {}, 'element_content'),
      ),
    ),
    tableBody,
  );

  const fileInput = el('input', {
    type: 'file',
    accept: '.yaml,.yml',
    class: 'price-editor-file',
    onChange: handleYamlUpload,
  });

  const uploadBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn',
      onClick: () => fileInput.click(),
    },
    'Загрузить YAML',
  );

  const saveBtn = el(
    'button',
    {
      type: 'button',
      class: 'btn primary',
      onClick: handleSave,
    },
    'Сохранить snapshot',
  );

  const status = el('span', { class: 'price-editor-status' }, '');

  const actions = el(
    'div',
    { class: 'price-editor-actions' },
    fileInput,
    uploadBtn,
    saveBtn,
    status,
  );

  const root = el(
    'div',
    { class: 'price-editor' },
    errorBanner,
    meta,
    table,
    actions,
  );

  container.appendChild(root);

  // ──────────────────────────────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────────────────────────────

  function setError(msg) {
    errorBanner.textContent = msg;
    errorBanner.hidden = false;
  }
  function clearError() {
    errorBanner.textContent = '';
    errorBanner.hidden = true;
  }

  function setStatus(msg, kind = '') {
    status.textContent = msg || '';
    status.className = 'price-editor-status' + (kind ? ` ${kind}` : '');
  }

  function refreshMeta() {
    root.querySelector('[data-bind="currency"]').textContent =
      snapshot.currency || '—';
    root.querySelector('[data-bind="date"]').textContent =
      snapshot.date || '—';
    root.querySelector('[data-bind="count"]').textContent = String(
      Object.keys(snapshot.materials || {}).length,
    );
  }

  function renderRows() {
    tableBody.replaceChildren();
    const materials = snapshot.materials || {};
    for (const mid of Object.keys(materials)) {
      tableBody.append(makeRow(mid, materials[mid]));
    }
  }

  function makeRow(mid, mat) {
    const idCell = el('td', { class: 'mono price-editor-id' }, mid);
    const kindSelect = el(
      'select',
      {
        class: 'price-input',
        'data-mid': mid,
        'data-field': 'kind',
        onChange: handleEdit,
      },
      ...KINDS.map((k) =>
        el('option', { value: k, selected: k === mat.kind || undefined }, k),
      ),
    );
    const priceInput = el('input', {
      type: 'number',
      class: 'price-input mono',
      step: '0.01',
      min: '0',
      value: String(mat.price_per_kg ?? 0),
      'data-mid': mid,
      'data-field': 'price_per_kg',
      onChange: handleEdit,
    });
    const ecInput = el('input', {
      type: 'text',
      class: 'price-input mono price-input-wide',
      value: elementContentToString(mat.element_content),
      'data-mid': mid,
      'data-field': 'element_content',
      onChange: handleEdit,
      placeholder: 'Mn=0.80;Fe=0.20',
    });
    return el(
      'tr',
      { class: 'price-row', 'data-mid': mid },
      idCell,
      el('td', {}, kindSelect),
      el('td', { class: 'right' }, priceInput),
      el('td', {}, ecInput),
    );
  }

  // ──────────────────────────────────────────────────────────────────
  // Edit handlers
  // ──────────────────────────────────────────────────────────────────

  function applyEdit(input) {
    const mid = input.dataset.mid;
    const field = input.dataset.field;
    if (!mid || !field) return;
    const mat = (snapshot.materials || {})[mid];
    if (!mat) return;

    if (field === 'kind') {
      mat.kind = input.value;
    } else if (field === 'price_per_kg') {
      const val = parseFloat(input.value);
      if (!Number.isFinite(val) || val <= 0) {
        setError(`«${mid}»: price_per_kg должен быть > 0`);
        return;
      }
      mat.price_per_kg = val;
    } else if (field === 'element_content') {
      try {
        mat.element_content = parseElementContent(input.value);
      } catch (err) {
        setError(`«${mid}»: ${err.message}`);
        return;
      }
    }
    clearError();
    if (onChange) {
      try {
        onChange(getSnapshot());
      } catch (e) {
        console.error(e);
      }
    }
  }

  function handleEdit(ev) {
    applyEdit(ev.currentTarget);
  }

  // Numeric inputs rely on `onChange` (fires on blur/Enter) — that's the
  // correct semantics for a price editor: commit on user-confirm, not on
  // every keystroke. A previous attempt at real-time debounced `onInput`
  // was removed because the setTimeout callback received `ev.currentTarget
  // === null` (the browser nulls it after the synchronous handler returns)
  // and `applyEdit(null)` returned early — making the wiring dead code.

  // ──────────────────────────────────────────────────────────────────
  // YAML upload — read file as text, parse minimally, replace snapshot
  // ──────────────────────────────────────────────────────────────────

  function handleYamlUpload(ev) {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result);
        const parsed = parseYamlSnapshot(text);
        setSnapshot(parsed);
        setStatus(`Загружено: ${file.name}`, 'ok');
      } catch (err) {
        setError(`Не удалось разобрать YAML: ${err.message}`);
      } finally {
        // Reset input so re-uploading the same file re-triggers onChange.
        fileInput.value = '';
      }
    };
    reader.onerror = () => setError('Не удалось прочитать файл');
    reader.readAsText(file);
  }

  // ──────────────────────────────────────────────────────────────────
  // Save → POST /api/prices/upload
  // ──────────────────────────────────────────────────────────────────

  async function handleSave() {
    saveBtn.disabled = true;
    setStatus('Сохранение…', '');
    try {
      const body = getSnapshot();
      const resp = await apiFetch('/api/prices/upload', {
        method: 'POST',
        body,
      });
      const savedPath = resp?.saved_path || '';
      setStatus(savedPath ? `Сохранено: ${savedPath}` : 'Сохранено', 'ok');
      if (onSave) {
        try { onSave(savedPath); } catch (e) { console.error(e); }
      }
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : String(err);
      setError(`Не удалось сохранить: ${detail}`);
      setStatus('');
    } finally {
      saveBtn.disabled = false;
    }
  }

  // ──────────────────────────────────────────────────────────────────
  // Public API
  // ──────────────────────────────────────────────────────────────────

  function setSnapshot(next) {
    snapshot = cloneSnapshot(next || { materials: {}, currency: 'EUR' });
    clearError();
    refreshMeta();
    renderRows();
    if (onChange) {
      try { onChange(getSnapshot()); } catch (e) { console.error(e); }
    }
  }

  function getSnapshot() {
    // Return a deep copy so callers can stash without aliasing internal state.
    return cloneSnapshot(snapshot);
  }

  function destroy() {
    if (root.parentNode) root.parentNode.removeChild(root);
  }

  // First render
  renderRows();

  return {
    element: root,
    getSnapshot,
    setSnapshot,
    destroy,
  };
}

// ──────────────────────────────────────────────────────────────────────
// Tiny YAML parser — sufficient for the seed snapshot format only
// ──────────────────────────────────────────────────────────────────────
//
// We don't pull in a YAML library to keep the no-build promise. The
// supported subset:
//   - top-level keys: date, currency, source, notes, materials
//   - materials map: each child has kind / price_per_kg / element_content
//   - element_content: inline-flow `{Mn: 0.80, Fe: 0.20}`
//
// Anything fancier (anchors, multi-line strings, lists) is out of scope —
// in that case the user can paste JSON via the API or use Streamlit.
//
// Throws Error with a human-readable message on parse failure.
function parseYamlSnapshot(text) {
  const lines = String(text).split(/\r?\n/);
  const root = { materials: {} };
  let currentMat = null;
  let currentMatId = null;
  let inMaterials = false;

  const stripComment = (s) => s.replace(/\s+#.*$/, '');

  for (const rawLine of lines) {
    const line = stripComment(rawLine.replace(/\t/g, '  '));
    if (!line.trim() || line.trim().startsWith('#')) continue;

    // Top-level (no leading whitespace)
    if (!line.startsWith(' ')) {
      const m = line.match(/^([a-zA-Z_]+)\s*:\s*(.*)$/);
      if (!m) continue;
      const key = m[1];
      const value = m[2].trim();
      if (key === 'materials') {
        inMaterials = true;
        continue;
      }
      inMaterials = false;
      currentMat = null;
      // Strip surrounding quotes if any.
      const unquoted = value.replace(/^["'](.*)["']$/, '$1');
      root[key] = unquoted;
    } else if (inMaterials) {
      // 2-space indent → material id; 4-space → field of current material.
      const m2 = line.match(/^ {2}([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$/);
      const m4 = line.match(/^ {4}([a-zA-Z_]+)\s*:\s*(.*)$/);
      if (m2) {
        currentMatId = m2[1];
        currentMat = { kind: '', price_per_kg: 0, element_content: {} };
        root.materials[currentMatId] = currentMat;
      } else if (m4 && currentMat) {
        const k = m4[1];
        const v = m4[2].trim();
        if (k === 'kind') {
          currentMat.kind = v.replace(/^["'](.*)["']$/, '$1');
        } else if (k === 'price_per_kg') {
          currentMat.price_per_kg = parseFloat(v);
        } else if (k === 'element_content') {
          // expect inline flow: {Mn: 0.80, Fe: 0.20}
          const m3 = v.match(/^\{(.*)\}$/);
          if (!m3) {
            throw new Error(
              `${currentMatId}.element_content: ожидаю inline-flow {Mn: 0.80, ...}`,
            );
          }
          const inner = m3[1];
          const ec = {};
          for (const tok of inner.split(',').map((s) => s.trim()).filter(Boolean)) {
            const m4i = tok.match(/^([A-Za-z]+)\s*:\s*([0-9.]+)$/);
            if (!m4i) {
              throw new Error(
                `${currentMatId}.element_content: токен «${tok}» (ожидаю «Mn: 0.80»)`,
              );
            }
            ec[m4i[1]] = parseFloat(m4i[2]);
          }
          currentMat.element_content = ec;
        }
      }
    }
  }

  if (!root.currency) throw new Error('отсутствует currency');
  if (!root.materials || Object.keys(root.materials).length === 0) {
    throw new Error('отсутствует materials');
  }
  return root;
}
