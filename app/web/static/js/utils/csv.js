// CSV utilities — RFC 4180 escaping и Blob download.
// Reusable across views: PR 5 (experiments), PR 7 (design candidates),
// PR 11 (recipes). Extracted from views/experiments.js so future CSV
// exports do not duplicate escape/download boilerplate.
//
// References:
//   - RFC 4180 (Common Format and MIME Type for CSV Files)
//   - Excel UTF-8 detection via leading BOM (U+FEFF)

/**
 * Escape one value per RFC 4180.
 *
 * Wraps the value in double-quotes when it contains a comma, double-quote,
 * newline (LF or CR), or has leading/trailing whitespace. Internal quotes
 * are doubled. ``null``/``undefined`` are stringified as empty.
 *
 * @param {*} value
 * @returns {string}
 */
export function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const s = String(value);
  if (/[",\n\r]/.test(s) || /^\s|\s$/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

/**
 * Build a CSV string from header + rows.
 *
 * Each cell is escaped via :func:`csvEscape`. Default end-of-line is
 * ``\r\n`` per RFC 4180; pass ``'\n'`` for Unix-only consumers.
 *
 * @param {string[]} headers — column names (array of strings)
 * @param {Array<Array<*>>} rows — array of row arrays; each row should have
 *   the same length as ``headers`` but no enforcement is done — caller is
 *   responsible for shape consistency.
 * @param {string} [eol='\r\n']
 * @returns {string}
 */
export function buildCsv(headers, rows, eol = '\r\n') {
  const lines = [headers.map(csvEscape).join(',')];
  for (const row of rows) {
    lines.push(row.map(csvEscape).join(','));
  }
  return lines.join(eol);
}

/**
 * Trigger a browser download of CSV via Blob + transient ``<a download>``.
 *
 * Prepends a UTF-8 BOM (``﻿``) so Excel auto-detects encoding when
 * the file is double-clicked. Object URL and DOM node are cleaned up
 * synchronously after click.
 *
 * @param {string} filename — e.g. ``"next-experiments-v1-2026-05-09.csv"``
 * @param {string} csvContent — pre-built CSV string (typically from
 *   :func:`buildCsv`)
 */
export function downloadCsv(filename, csvContent) {
  const BOM = '﻿';
  const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
