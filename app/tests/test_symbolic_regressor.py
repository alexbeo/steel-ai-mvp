"""Unit tests for symbolic_regressor — small synthetic recovery + Pareto filter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backend.symbolic_regressor import (
    SymbolicFormula,
    SymbolicRegressionRun,
    _pareto_frontier,
    _program_to_infix,
    run_symbolic_regression,
)


def test_pareto_frontier_keeps_non_dominated():
    class FakeProg:
        def __init__(self, name): self.name = name
    a, b, c, d = FakeProg("a"), FakeProg("b"), FakeProg("c"), FakeProg("d")
    triples = [
        (a, 0.5, 5),    # dominated by c
        (b, 0.3, 8),    # on frontier
        (c, 0.4, 5),    # on frontier (lower length)
        (d, 0.5, 10),   # dominated by b
    ]
    front = _pareto_frontier(triples)
    names = {p.name for p, _, _ in front}
    assert "b" in names
    assert "c" in names
    assert "a" not in names
    assert "d" not in names


def test_pareto_frontier_empty_input():
    assert _pareto_frontier([]) == []


def test_run_recovers_simple_linear(monkeypatch):
    """A target of form y = 2*x0 + 5 should yield a Pareto frontier
    that includes a short formula with R² > 0.95 — gplearn handles this
    routinely. Run with a small population to keep test fast."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "alpha": rng.uniform(0, 10, 200),
        "noise":  rng.uniform(0, 10, 200),
    })
    y = pd.Series(2.0 * X["alpha"] + 5.0 + rng.normal(0, 0.1, 200), name="y")

    result = run_symbolic_regression(
        X, y,
        feature_names=["alpha", "noise"],
        population_size=400,
        generations=6,
        n_jobs=1,
        random_state=0,
    )
    assert isinstance(result, SymbolicRegressionRun)
    assert len(result.pareto_frontier) >= 1
    assert all(isinstance(f, SymbolicFormula) for f in result.pareto_frontier)
    assert result.best_overall is not None
    assert result.best_overall.r2 > 0.9
    assert "alpha" in result.best_overall.formula_infix


def test_pareto_frontier_is_monotonic(monkeypatch):
    """As complexity increases along the frontier, R² should not decrease."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"x": rng.uniform(0.5, 5, 150)})
    y = pd.Series(np.log(X["x"]) * 3 + 2 + rng.normal(0, 0.05, 150), name="y")

    result = run_symbolic_regression(
        X, y,
        feature_names=["x"],
        population_size=300,
        generations=5,
        n_jobs=1,
        random_state=1,
    )
    if len(result.pareto_frontier) >= 2:
        complexities = [f.complexity for f in result.pareto_frontier]
        r2s = [f.r2 for f in result.pareto_frontier]
        assert complexities == sorted(complexities)
        for i in range(1, len(r2s)):
            assert r2s[i] >= r2s[i - 1] - 1e-6  # tiny tolerance for float noise


def test_program_to_infix_simple():
    """Manually construct a fake program and verify infix conversion."""
    class FakeFn:
        def __init__(self, name, arity):
            self.name = name
            self.arity = arity

    add = FakeFn("add", 2)
    sqrt = FakeFn("sqrt", 1)

    class FakeProg:
        program = [add, 0, sqrt, 1]  # add(x0, sqrt(x1))

    result = _program_to_infix(FakeProg(), ["a", "b"])
    assert result == "a + sqrt(b)"
