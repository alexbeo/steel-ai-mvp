// Tiny DOM helpers — vanilla, no framework. Used by router + future views.

export const qs = (sel, root = document) => root.querySelector(sel);
export const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Build an element with props (incl. dataset/style) and children. */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key === 'class') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value !== false && value != null) {
      node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ─────────────────────────────────────────────────────────────────────
// Loading skeletons — visual polish (R-001 Step 4)
// ─────────────────────────────────────────────────────────────────────
//
// Cell.: вкладки в момент первой загрузки показывают шиммер-плейсхолдер
// вместо «Загрузка…» текста. Соответствующие keyframes описаны в
// overrides.css под селектором `.skeleton`. Helpers тут — DOM-side.

/** A single shimmer rectangle. Use for inline placeholders within a row.
 *
 * @param {object} opts
 * @param {string} [opts.width]  CSS width; default 100 %
 * @param {string} [opts.height] CSS height; default 12 px
 * @param {string} [opts.className] Extra class tokens to merge
 */
export function skeletonBlock(opts = {}) {
  const cls = ['skeleton', 'skeleton-block', opts.className].filter(Boolean).join(' ');
  const style = {};
  if (opts.width) style.width = opts.width;
  if (opts.height) style.height = opts.height;
  return el('span', { class: cls, style, 'aria-hidden': 'true' });
}

/** A skeleton-grid row matching `.log-row` proportions — used for
 *  Decision Log / candidate / experiments tables while the request
 *  is in flight. `cols` controls how many shimmer cells appear.
 *
 * @param {object} opts
 * @param {number} [opts.cols] Number of shimmer cells; default 6
 */
export function skeletonRow(opts = {}) {
  const cols = Number.isFinite(opts.cols) ? opts.cols : 6;
  const children = [];
  for (let i = 0; i < cols; i += 1) {
    children.push(el('span', { class: 'skeleton skeleton-block', 'aria-hidden': 'true' }));
  }
  return el('div', { class: 'skeleton-row', 'aria-busy': 'true', role: 'presentation' }, ...children);
}

/** A vertical stack of `n` shimmer blocks of varying widths — used as
 *  a placeholder for free-form summaries (e.g. project history summary,
 *  predict-result panel) before content arrives. */
export function skeletonStack(n = 3) {
  const widths = ['w-100', 'w-75', 'w-50', 'w-100', 'w-25'];
  const blocks = [];
  for (let i = 0; i < n; i += 1) {
    const w = widths[i % widths.length];
    blocks.push(
      el('span', {
        class: `skeleton skeleton-block ${w}`,
        'aria-hidden': 'true',
      }),
    );
  }
  return el(
    'div',
    { class: 'skeleton-stack', 'aria-busy': 'true', role: 'presentation' },
    ...blocks,
  );
}
