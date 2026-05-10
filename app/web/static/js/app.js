// Entry point — bootstraps the router on DOMContentLoaded and starts
// the topbar KPI auto-refresh loop.
//
// PR 13: Streamlit decommissioned, KPI strip wired to /api/system/kpi.
import { mountRouter } from './router.js';
import { startTopbarKpiAutoRefresh } from './components/topbar.js';

function boot() {
  mountRouter();
  startTopbarKpiAutoRefresh();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
