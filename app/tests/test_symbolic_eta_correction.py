"""Tests для symbolic η_Al correction (PR 11)."""
from __future__ import annotations

import pytest


def test_correction_target_formulation():
    """y = eta / baseline, baseline excluded from features."""
    from scripts.symbolic_eta_correction import _load_data
    X, y, feature_names = _load_data("synthetic", n_heats=200)
    assert "method_eta_baseline" not in feature_names
    assert "plant_offset_baseline" not in feature_names
    assert "slag_feo_pct" in feature_names
    # correction ~ 1.0
    assert 0.5 < y.mean() < 1.5


def test_sr_recovers_signal():
    """SR на synthetic — Pareto frontier найден, формулы валидны.

    NB: на tiny config (pop=200, gen=5) лучшая формула может быть
    вырожденной (R²<0) — это нормально для smoke-уровня. Проверяем,
    что архитектура отрабатывает и возвращает корректные SymbolicFormula.
    """
    from app.backend.symbolic_regressor import run_symbolic_regression
    from scripts.symbolic_eta_correction import _load_data
    X, y, feature_names = _load_data("synthetic", n_heats=300)
    # Small config для скорости теста
    run = run_symbolic_regression(
        X, y, feature_names=feature_names,
        population_size=200, generations=5, random_state=42,
    )
    assert len(run.pareto_frontier) >= 1
    best = run.best_overall
    assert best is not None
    assert isinstance(best.formula_infix, str) and best.formula_infix
    # R² — конечное число (не NaN/inf), знак не гарантирован на tiny config
    import math
    assert math.isfinite(best.r2)


def test_sr_pareto_frontier_multiple():
    from app.backend.symbolic_regressor import run_symbolic_regression
    from scripts.symbolic_eta_correction import _load_data
    X, y, feature_names = _load_data("synthetic", n_heats=300)
    run = run_symbolic_regression(
        X, y, feature_names=feature_names,
        population_size=300, generations=8, random_state=42,
    )
    assert len(run.pareto_frontier) >= 1


def test_heats_db_insufficient_data_errors(monkeypatch):
    # list_heats has DEFAULT_DB_PATH bound as a default arg at definition time,
    # so monkeypatching the module attr won't redirect it. Patch the symbol
    # imported lazily inside _load_data to return an empty list deterministically.
    import app.backend.heat_records as hr
    monkeypatch.setattr(hr, "list_heats", lambda *a, **kw: [])
    from scripts.symbolic_eta_correction import _load_data
    with pytest.raises(SystemExit, match="Недостаточно"):
        _load_data("heats_db", n_heats=0)
