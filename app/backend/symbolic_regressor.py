"""
Symbolic Regressor — B1 from AI integration roadmap.

Runs genetic-programming symbolic regression (via gplearn) over a
training dataset and returns the Pareto-optimal frontier of formulas
trading off complexity vs accuracy.

Output is interpretable closed-form expressions like
    "fatigue ≈ 0.5 * normalizing_temp + log(carburizing_temp_c) - 12"
that can be compared against known empirical metallurgical laws
(Hall-Petch, Hollomon-Jaffe, Pickering solid-solution strengthening,
CEV/Pcm hardenability indices).

Different value than A1: A1 looks for predictive uplift on top of
existing model; B1 looks for interpretable structure that may capture
most of the signal in a far simpler form. Especially valuable for
academic users who need publishable analytical relationships.

Pareto frontier: from the final GP population, each program has
(length, fitness). A program is Pareto-optimal if no other program is
both strictly shorter AND strictly better-fitting. The frontier
gives the user a complexity/accuracy menu rather than a single
"best" answer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolicFormula:
    formula_lisp: str          # gplearn-style: add(mul(...), ...)
    formula_infix: str         # human-readable: a + b * c
    complexity: int            # number of nodes in expression tree
    r2: float                  # on the training fold
    rmse: float
    mae: float


@dataclass
class SymbolicRegressionRun:
    target: str
    feature_names: list[str]
    n_train: int
    pareto_frontier: list[SymbolicFormula] = field(default_factory=list)
    best_overall: SymbolicFormula | None = None


# ---------------------------------------------------------------------------
# lisp → infix converter for gplearn _Program objects
# ---------------------------------------------------------------------------

_BINARY_INFIX_OPS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/",
    "max": None, "min": None,  # keep functional
}
_UNARY_FUNCS = {"sqrt", "log", "abs", "neg", "inv", "sin", "cos", "tan"}


def _format_const(c: float) -> str:
    if abs(c) < 0.01:
        return f"{c:.3g}"
    return f"{c:.3f}".rstrip("0").rstrip(".")


def _program_to_infix(program: Any, feature_names: list[str]) -> str:
    """Convert a gplearn _Program (flat list of Function/int/float) to
    a human-readable infix string."""
    nodes = list(program.program)

    def consume(idx: int) -> tuple[str, int]:
        node = nodes[idx]
        if isinstance(node, (int, np.integer)):
            return feature_names[int(node)], idx + 1
        if isinstance(node, (float, np.floating)):
            return _format_const(float(node)), idx + 1

        name = node.name
        arity = node.arity
        next_i = idx + 1
        args = []
        for _ in range(arity):
            arg, next_i = consume(next_i)
            args.append(arg)

        if arity == 2 and name in _BINARY_INFIX_OPS and _BINARY_INFIX_OPS[name]:
            op = _BINARY_INFIX_OPS[name]
            return f"({args[0]} {op} {args[1]})", next_i
        if arity == 1 and name in _UNARY_FUNCS:
            display = "1/" if name == "inv" else "-" if name == "neg" else name
            if name in {"inv", "neg"}:
                return f"({display}{args[0]})", next_i
            return f"{display}({args[0]})", next_i
        return f"{name}({', '.join(args)})", next_i

    expr, _ = consume(0)
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    return expr


# ---------------------------------------------------------------------------
# Pareto frontier extraction
# ---------------------------------------------------------------------------

def _pareto_frontier(
    programs: list[tuple[Any, float, int]],
) -> list[tuple[Any, float, int]]:
    """Return non-dominated (program, fitness, length) triples.
    Lower fitness is better (gplearn minimises). Lower length is better.
    """
    if not programs:
        return []
    sorted_by_length = sorted(programs, key=lambda p: (p[2], p[1]))
    frontier = []
    best_fitness_seen = float("inf")
    for prog, fit, length in sorted_by_length:
        if fit < best_fitness_seen:
            frontier.append((prog, fit, length))
            best_fitness_seen = fit
    return frontier


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_symbolic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    population_size: int = 1500,
    generations: int = 12,
    parsimony_coefficient: float = 0.01,
    random_state: int = 42,
    n_jobs: int = -1,
) -> SymbolicRegressionRun:
    from gplearn.genetic import SymbolicRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    if feature_names is None:
        feature_names = list(X.columns)

    sr = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs"),
        parsimony_coefficient=parsimony_coefficient,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
        feature_names=feature_names,
    )
    X_arr = X[feature_names].values
    y_arr = y.values
    sr.fit(X_arr, y_arr)

    final_pop = sr._programs[-1]
    triples = [(p, p.fitness_, p.length_) for p in final_pop if p is not None]
    frontier_triples = _pareto_frontier(triples)

    formulas = []
    for prog, _, _ in frontier_triples:
        pred = prog.execute(X_arr)
        formulas.append(SymbolicFormula(
            formula_lisp=str(prog),
            formula_infix=_program_to_infix(prog, feature_names),
            complexity=prog.length_,
            r2=float(r2_score(y_arr, pred)),
            rmse=float(np.sqrt(np.mean((y_arr - pred) ** 2))),
            mae=float(mean_absolute_error(y_arr, pred)),
        ))

    formulas.sort(key=lambda f: f.complexity)

    best = max(formulas, key=lambda f: f.r2) if formulas else None

    return SymbolicRegressionRun(
        target=str(y.name) if y.name else "target",
        feature_names=feature_names,
        n_train=int(len(X)),
        pareto_frontier=formulas,
        best_overall=best,
    )
