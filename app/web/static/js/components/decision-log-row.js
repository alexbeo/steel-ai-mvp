// Decision log row component — collapsed strip that expands to full details.
//
// Reuses the existing palette classes from app.css:
//   .log-list / .log-row / .tag-mini / .verdict-mini / .pill
// plus a small set of additions in css/overrides.css for the expanded panel.
//
// Public API: `decisionLogRow(decision)` → HTMLElement.
// `decision` is the JSON object returned by /api/decisions — keys: id,
// timestamp, phase, decision, reasoning, alternatives_considered, context,
// tags, author, outcome, revised_at.

import { el } from '../utils/dom.js';

/** Map a Decision Log phase string to the existing severity-pill modifier.
 *  Pure cosmetic — we want the strip rows to read at a glance which phase a
 *  record belongs to without spelling it out.
 */
function phaseToPillClass(phase) {
  switch (phase) {
    case 'training':
    case 'inverse_design':
      return 'amber';
    case 'validation':
    case 'deoxidation':
      return 'warn';
    case 'data_acquisition':
    case 'preprocessing':
    case 'feature_engineering':
    case 'reporting':
    case 'meta':
    default:
      return ''; // green dot (default success colour)
  }
}

/** Short timestamp like 2026-05-09 12:30 (Streamlit shows 10-char date only;
 *  we add hours/minutes for finer ordering since rows are date-sorted DESC).
 */
function fmtTimestamp(iso) {
  if (!iso) return '—';
  // ISO is already in stable lexical sort order; truncate at minute resolution.
  // Tolerates absent 'T' separator.
  const t = String(iso).replace('T', ' ');
  return t.length >= 16 ? t.slice(0, 16) : t;
}

function fmtTagChip(tag) {
  return el('span', { class: 'tag-mini' }, tag);
}

/** Pretty-print the context dict — single source of truth for both the
 *  expanded panel and any future export. JSON.stringify with 2-space indent
 *  matches what Streamlit's `st.json` produces and is mono-font friendly.
 */
function fmtContext(ctx) {
  if (!ctx || (typeof ctx === 'object' && Object.keys(ctx).length === 0)) {
    return '{}';
  }
  try {
    return JSON.stringify(ctx, null, 2);
  } catch (_e) {
    return String(ctx);
  }
}

/** Build the collapsed-row strip used inside the .log-list container. */
function buildRow(d, onToggle) {
  const tags = Array.isArray(d.tags) ? d.tags : [];
  const headDecision = d.decision || '—';

  return el(
    'div',
    {
      class: 'log-row decision-log-row',
      role: 'button',
      tabindex: '0',
      'aria-expanded': 'false',
      'data-decision-id': d.id,
      onClick: onToggle,
      onKeydown: (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          onToggle(ev);
        }
      },
    },
    el('span', { class: 'ts mono' }, fmtTimestamp(d.timestamp)),
    el(
      'span',
      { class: `pill ${phaseToPillClass(d.phase)}` },
      d.phase || '—',
    ),
    el('span', { class: 'desc' }, headDecision),
    el(
      'span',
      { class: 'tags' },
      ...(tags.length ? tags.slice(0, 4).map(fmtTagChip) : [el('span', { class: 'tag-mini' }, '—')]),
    ),
    el('span', { class: 'author mono' }, d.author || 'unknown'),
    el(
      'span',
      { class: 'verdict-mini draft' },
      d.outcome ? 'OUTCOME' : 'OPEN',
    ),
  );
}

/** Build the expanded-detail panel. Layout is in css/overrides.css under
 *  `.decision-log-details`. */
function buildDetails(d) {
  const sections = [];

  // Reasoning is the most important field — full text, no truncation.
  sections.push(
    el(
      'div',
      { class: 'detail-block' },
      el('div', { class: 'detail-label' }, 'Reasoning'),
      el('div', { class: 'detail-body' }, d.reasoning || '—'),
    ),
  );

  const alts = Array.isArray(d.alternatives_considered)
    ? d.alternatives_considered
    : [];
  if (alts.length) {
    sections.push(
      el(
        'div',
        { class: 'detail-block' },
        el('div', { class: 'detail-label' }, 'Альтернативы'),
        el(
          'ul',
          { class: 'detail-alts' },
          ...alts.map((alt) => el('li', {}, String(alt))),
        ),
      ),
    );
  }

  if (d.context && Object.keys(d.context).length) {
    sections.push(
      el(
        'div',
        { class: 'detail-block' },
        el('div', { class: 'detail-label' }, 'Context'),
        el('pre', { class: 'detail-context mono' }, fmtContext(d.context)),
      ),
    );
  }

  const tags = Array.isArray(d.tags) ? d.tags : [];
  if (tags.length) {
    sections.push(
      el(
        'div',
        { class: 'detail-block' },
        el('div', { class: 'detail-label' }, 'Теги'),
        el('div', { class: 'detail-tags' }, ...tags.map(fmtTagChip)),
      ),
    );
  }

  if (d.outcome) {
    sections.push(
      el(
        'div',
        { class: 'detail-block detail-outcome' },
        el('div', { class: 'detail-label' }, 'Outcome'),
        el('div', { class: 'detail-body' }, String(d.outcome)),
      ),
    );
  }

  // Footer line: id, author, revised_at — small, mono, muted.
  const footer = el(
    'div',
    { class: 'detail-footer mono' },
    `id=${d.id} · author=${d.author || 'unknown'}` +
      (d.revised_at ? ` · revised=${fmtTimestamp(d.revised_at)}` : ''),
  );
  sections.push(footer);

  return el(
    'div',
    { class: 'decision-log-details', hidden: '' },
    ...sections,
  );
}

/** Public component: returns a wrapper that contains the collapsed row and
 *  the (initially hidden) details panel. Click on the row toggles the panel.
 */
export function decisionLogRow(decision) {
  const wrap = el('div', { class: 'decision-log-item' });

  const details = buildDetails(decision);
  const onToggle = () => {
    const isOpen = !details.hidden;
    if (isOpen) {
      details.hidden = true;
      row.setAttribute('aria-expanded', 'false');
      wrap.classList.remove('open');
    } else {
      details.hidden = false;
      row.setAttribute('aria-expanded', 'true');
      wrap.classList.add('open');
    }
  };
  const row = buildRow(decision, onToggle);

  wrap.append(row, details);
  return wrap;
}
