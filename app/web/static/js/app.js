// Entry point — bootstraps the router on DOMContentLoaded.
import { mountRouter } from './router.js';

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mountRouter);
} else {
  mountRouter();
}
