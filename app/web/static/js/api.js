// fetch wrapper — same-origin, JSON in/out, normalised errors.

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

export async function apiFetch(path, options = {}) {
  const { method = 'GET', body, headers = {}, signal } = options;
  const init = { method, headers: { Accept: 'application/json', ...headers }, signal };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = typeof body === 'string' ? body : JSON.stringify(body);
  }
  const resp = await fetch(path, init);
  const ct = resp.headers.get('content-type') || '';
  const isJson = ct.includes('application/json');
  const payload = isJson ? await resp.json().catch(() => null) : await resp.text();
  if (!resp.ok) {
    const detail = (payload && payload.detail) || resp.statusText || `HTTP ${resp.status}`;
    throw new ApiError(detail, resp.status, payload);
  }
  return payload;
}
