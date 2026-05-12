// Hash-based router. One section visible at a time; nav-item gets .active.
// View modules are lazy-imported so PR 1's empty stubs don't bloat first paint.
import { qs, qsa } from './utils/dom.js';

const ROUTES = [
  'design', 'train', 'predict', 'deox',
  'hypotheses', 'recipe', 'experiments', 'history',
];
const DEFAULT_ROUTE = 'design';

const viewLoaders = {
  design:      () => import('./views/design.js'),
  train:       () => import('./views/train.js'),
  predict:     () => import('./views/predict.js'),
  deox:        () => import('./views/deox.js'),
  hypotheses:  () => import('./views/hypotheses.js'),
  recipe:      () => import('./views/recipe.js'),
  experiments: () => import('./views/experiments.js'),
  history:     () => import('./views/history.js'),
};

const initialised = new Set();

function parseRoute() {
  const raw = (location.hash || '').replace(/^#\/?/, '');
  const route = raw.split('/')[0];
  return ROUTES.includes(route) ? route : DEFAULT_ROUTE;
}

async function activateView(route) {
  for (const section of qsa('[data-view]')) {
    section.classList.toggle('active', section.dataset.view === route);
  }
  for (const link of qsa('.nav-item[data-route]')) {
    link.classList.toggle('active', link.dataset.route === route);
  }

  if (initialised.has(route)) return;
  const loader = viewLoaders[route];
  if (!loader) return;
  try {
    const mod = await loader();
    const container = qs(`[data-view="${route}"]`);
    if (mod && typeof mod.init === 'function' && container) {
      mod.init(container);
    }
    initialised.add(route);
  } catch (err) {
    // Lazy-import failure is non-fatal — placeholder stays visible. Surface to console for dev.
    console.error(`[router] failed to init view "${route}":`, err);
  }
}

export function mountRouter() {
  if (!location.hash) {
    location.replace(`#/${DEFAULT_ROUTE}`);
  }
  window.addEventListener('hashchange', () => activateView(parseRoute()));
  activateView(parseRoute());
}
