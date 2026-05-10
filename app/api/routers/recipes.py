"""Router for /api/recipes/* — paired LLM recipe designer + critic cycle.

PR 11 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Подбор рецепта»).

Streamlit parity reference: ``app/frontend/app.py`` lines 1771-2214
(``with tab_recipe:`` block).

Endpoint:
- ``POST /api/recipes/ai-cycle`` — submit a paired generator+critic cycle
  (Sonnet PhD designer → ML+cost numerical truth gate → Sonnet PhD critic
  with evidence fact-check) as a long-running job. Returns ``{job_id}``;
  UI polls ``GET /api/jobs/{id}`` (PR 6) and reads ``result`` once status
  flips to ``done``.

PR 11 — endpoint convention note:
``ai-cycle`` (paired LLM pattern, mirrors PR 9 ``deox/ai-cycle``) rather
than ``run`` because the user-visible action is the *full bundle* —
designer + critic + ML verification — not just "generate recipes". Per
the design doc:

    «Convention для paired-LLM endpoints: <scope>/ai-cycle (PR 9
    deox/ai-cycle, PR 11 recipes/ai-cycle если применимо). Single-LLM
    endpoints (PR 10 hypotheses) — просто <scope>/run».

Steel class gate:
The recipe pair currently runs only against ``fatigue_carbon_steel``
(real Agrawal NIMS data). Streamlit blocks non-fatigue classes at the
UI level (line 1824); the API surfaces the same restriction as a 400 so
test/automation callers see explicit failure rather than a downstream
NaN cascade.

Cooperative cancellation:
Same single-gate semantics as PR 9 / PR 10 — DELETE before the designer
LLM call raises ``Cancelled by user (before designer LLM call)``. Once
the Anthropic SDK call is in flight we cannot interrupt; ``DELETE``
flips ``cancellation_requested`` and the worker burns through to
completion (UI stops polling). A second gate fires between the designer
and the critic so cancel during the ~80 s designer call still aborts
before the critic burns its ~100 s round-trip.

Risks #3 (SafeJSONResponse): every endpoint declares
``response_class=SafeJSONResponse, response_model=None`` so dataclass +
numpy results from ``recipe_designer`` / ``recipe_critic`` serialise via
the custom encoder instead of Pydantic v2's default revalidation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.jobs import run_as_job
from app.api.llm_gating import check_llm_ready, make_progress_cancel_check
from app.api.responses import SafeJSONResponse
from app.api.routers import system as system_router

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────
# Constants — mirror Streamlit (app/frontend/app.py lines 1880-1890)
# ──────────────────────────────────────────────────────────────────────

# Class gate — same constraint as ``active_learning.py`` (PR 5). Streamlit
# refuses to run the recipe pair for any non-fatigue class (line 1824).
SUPPORTED_STEEL_CLASS = "fatigue_carbon_steel"

# Decision-vars catalogue mirrors ``scripts/design_recipe_with_critic.py``
# and Streamlit (lines 1880-1890). Composition + process knobs that
# downstream LLMs are allowed to recommend.
DESIGN_COMPOSITION = ["si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct"]
DESIGN_PROCESS = [
    "normalizing_temp_c",
    "carburizing_temp_c",
    "carburizing_time_min",
    "tempering_temp_c",
    "tempering_time_min",
    "through_hardening_cooling_rate_c_per_s",
]


_DEFAULT_TASK_TEXT = (
    "Снизить ferroalloy cost vs baseline при сохранении или улучшении "
    "fatigue strength. Целевой компромисс: каждый −€10/т имеет ценность "
    "если |Δσf| ≤ 30 МПа."
)


# ──────────────────────────────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────────────────────────────


class RecipesAiCycleRequest(BaseModel):
    """Body for ``POST /api/recipes/ai-cycle``.

    Streamlit collects only a free-text task description (line 1853) — the
    rest of the context (baseline, training_ranges, feature_importance) is
    derived server-side from the active model. The API mirrors that
    contract: caller passes ``model_version`` + ``task_text``, the worker
    builds the full designer context.

    ``save_to_decision_log`` is opt-in (default False) so test / automation
    flows do not pollute the audit trail; Streamlit always saves (line
    2024-2049).
    """

    model_version: str = Field(
        ...,
        description=(
            "Trained model version slug (matches ``models/<slug>/`` directory)."
            " Validated against the path-traversal regex in ``_safe_version_dir``."
            " Must be a fatigue_carbon_steel model — other classes return 400."
        ),
    )
    task_text: str = Field(
        default=_DEFAULT_TASK_TEXT,
        max_length=2000,
        description=(
            "Free-text design task — passed to the designer prompt as the "
            "high-level objective (cost reduction vs σf trade-off, target "
            "fatigue lift, etc). Default mirrors the Streamlit textarea "
            "default (app/frontend/app.py line 1848)."
        ),
    )
    save_to_decision_log: bool = Field(
        default=False,
        description=(
            "Opt-in audit-trail save with tag 'recipe_cycle'. "
            "Streamlit defaults to true; the API defaults to false to "
            "keep test/automation flows quiet."
        ),
    )

    model_config = {
        # Pydantic v2 reserves ``model_*`` for internal config — same opt-out
        # as the rest of the API surface.
        "protected_namespaces": (),
    }


# ──────────────────────────────────────────────────────────────────────
# Worker — invoked through ``run_as_job``
# ──────────────────────────────────────────────────────────────────────


def _build_designer_context(
    *,
    model_version: str,
    task_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the dict that ``RecipeDesigner.design`` consumes.

    Mirrors Streamlit (``app/frontend/app.py`` lines 1899-1961). Returns a
    ``(designer_ctx, runtime_state)`` tuple where ``runtime_state`` holds
    the shared fixtures (bundle, baseline row, feature_list, snapshot)
    that ``_verify_recipes`` needs without re-loading the model and the
    Agrawal dataset.

    Lazy imports — heavy deps (xgboost, pandas) stay out of the FastAPI
    cold-start path. Tests that monkeypatch the LLM factories don't load
    these unless the worker actually runs.
    """
    import numpy as np
    import pandas as pd

    from app.backend.cost_model import compute_cost, seed_snapshot
    from app.backend.data_curator import load_real_agrawal_fatigue_dataset
    from app.backend.model_trainer import load_model, predict_with_uncertainty

    bundle = load_model(model_version)
    meta = bundle["meta"]
    target = meta["target"]
    feature_list = list(meta["feature_list"])

    df_raw = load_real_agrawal_fatigue_dataset()
    if "sub_class" in df_raw.columns:
        sub = df_raw[df_raw["sub_class"] == "carbon_low_alloy"]
        if len(sub) < 50:
            sub = df_raw
    else:
        sub = df_raw
    baseline = sub.select_dtypes(include=[np.number]).median()

    snapshot = seed_snapshot()
    x_base = pd.DataFrame(
        [[float(baseline[f]) for f in feature_list]],
        columns=feature_list,
    )
    base_pred_row = predict_with_uncertainty(bundle, x_base).iloc[0]
    baseline_predicted = float(base_pred_row["prediction"])

    baseline_comp = {
        k: float(v) for k, v in baseline.items() if k.endswith("_pct")
    }
    baseline_cost = float(
        compute_cost(baseline_comp, snapshot, mode="full").total_per_ton
    )

    avail_comp = [c for c in DESIGN_COMPOSITION if c in feature_list]
    avail_proc = [p for p in DESIGN_PROCESS if p in feature_list]
    baseline_recipe = {
        **{k: float(baseline[k]) for k in avail_comp},
        **{k: float(baseline[k]) for k in avail_proc},
    }

    designer_ctx: dict[str, Any] = {
        "task": task_text,
        "steel_class": meta.get("steel_class"),
        "target": target,
        "data_source": meta.get("data_source"),
        "model_version": model_version,
        "r2_test": meta["metrics"]["r2_test"],
        "mae_test": meta["metrics"]["mae_test"],
        "coverage_90_ci": meta["metrics"]["coverage_90_ci"],
        "conformal_correction_mpa": meta.get("conformal_correction_mpa", 0),
        "feature_importance": meta.get("feature_importance") or {},
        "training_ranges": meta.get("training_ranges") or {},
        "target_distribution": {
            "min": float(df_raw[target].min()),
            "max": float(df_raw[target].max()),
            "mean": float(df_raw[target].mean()),
            "std": float(df_raw[target].std()),
            "n": int(len(df_raw)),
        },
        "baseline_recipe": baseline_recipe,
        "baseline_predicted_property": baseline_predicted,
        "baseline_cost_per_ton": baseline_cost,
        "available_composition": avail_comp,
        "available_process": avail_proc,
    }

    runtime_state: dict[str, Any] = {
        "bundle": bundle,
        "baseline_row": baseline,
        "feature_list": feature_list,
        "snapshot": snapshot,
        "baseline_predicted": baseline_predicted,
        "baseline_cost": baseline_cost,
        "baseline_recipe": baseline_recipe,
    }
    return designer_ctx, runtime_state


def _verify_recipes(
    recipes: list[Any],
    *,
    runtime_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run XGBoost predict + ferroalloy cost on each recipe.

    Mirrors Streamlit (``app/frontend/app.py`` lines 1970-2014). Returns a
    list of asdict-projected recipes with an extra ``ml_verification``
    block — exactly the shape the recipe critic expects (it reads
    ``predicted_property`` / ``lower_90`` / ``cost_per_ton`` /
    ``delta_property`` / ``delta_cost`` / ``ferroalloy_breakdown`` /
    ``ood_flag``).

    Failures in cost compute (e.g. PriceSnapshotIncomplete on a designer-
    proposed alloy not in the seed snapshot) are caught — we set
    ``cost_per_ton=None`` and emit an empty breakdown so the critic still
    has the property delta to peer-review. Streamlit does the same
    (lines 1986-1999).
    """
    import pandas as pd

    from app.backend.cost_model import compute_cost
    from app.backend.model_trainer import predict_with_uncertainty

    bundle = runtime_state["bundle"]
    baseline_row = runtime_state["baseline_row"]
    feature_list = runtime_state["feature_list"]
    snapshot = runtime_state["snapshot"]
    baseline_predicted = runtime_state["baseline_predicted"]
    baseline_cost = runtime_state["baseline_cost"]

    out: list[dict[str, Any]] = []
    for r in recipes:
        # ``baseline_row`` is a pandas Series — copy() so the loop body
        # doesn't mutate the shared baseline used for subsequent recipes.
        row = baseline_row.copy()
        for k, v in r.composition.items():
            if k in row.index:
                row[k] = float(v)
        for k, v in r.process_params.items():
            if k in row.index:
                row[k] = float(v)
        x_r = pd.DataFrame(
            [[float(row[f]) for f in feature_list]],
            columns=feature_list,
        )
        pp = predict_with_uncertainty(bundle, x_r).iloc[0]

        comp = {k: float(v) for k, v in row.items() if k.endswith("_pct")}
        try:
            cb = compute_cost(comp, snapshot, mode="full")
            cost_pt: float | None = float(cb.total_per_ton)
            ferro = [
                {
                    "material": c.material_id,
                    "kg_per_ton": round(float(c.mass_kg_per_ton_steel), 2),
                    "eur_per_ton": round(float(c.contribution_per_ton), 2),
                }
                for c in cb.contributions
                if c.material_id != "scrap"
            ]
        except Exception as exc:  # noqa: BLE001 — cost is best-effort, prediction must survive
            logger.warning(
                "compute_cost failed for recipe %s: %s — skipping cost", r.id, exc,
            )
            cost_pt = None
            ferro = []

        ml = {
            "predicted_property": float(pp["prediction"]),
            "lower_90": float(pp["lower_90"]),
            "upper_90": float(pp["upper_90"]),
            "ood_flag": bool(pp["ood_flag"]),
            "cost_per_ton": cost_pt,
            "delta_property": float(pp["prediction"]) - baseline_predicted,
            "delta_cost": (
                cost_pt - baseline_cost if cost_pt is not None else None
            ),
            "ferroalloy_breakdown": ferro,
        }
        d = asdict(r)
        d["ml_verification"] = ml
        out.append(d)
    return out


def _run_recipe_cycle_job(
    *,
    model_version: str,
    task_text: str,
    save_to_decision_log: bool,
    progress: Any = None,
) -> dict[str, Any]:
    """Background worker for the recipe-cycle pair.

    Stages (all coarse-grained — no intra-LLM progress hooks):

        0.05 → "Загрузка модели и baseline"
        0.20 → "Designer формулирует рецепты (~80 с)"
        0.55 → "ML+cost проверяет рецепты"
        0.65 → "Critic делает PhD-рецензию (~100 с)"
        0.95 → "Сохраняю результаты"
        1.00 → done

    Cancellation gates fire at 0.05 (before designer), 0.55 (between
    designer and critic), and 0.95 (before save). Per the module
    docstring, the gates are effective only outside an active LLM call.

    Returns a dict mirroring the Streamlit display block:
        {
          "model_version": str,
          "baseline": {recipe, predicted_property, cost_per_ton},
          "recipes": [Recipe-asdict + ml_verification, ...],
          "reviews": [RecipeVerdict-asdict, ...],
          "verdict_counts": {"ACCEPT": int, "REVISE": int, "REJECT": int},
          "duration_s": float,
          "decision_log_id": int | None,
          "config": {...echo of inputs...},
        }
    """
    started = time.monotonic()
    is_cancelled = make_progress_cancel_check(progress)

    if callable(progress):
        progress(0.05, "Загрузка модели и baseline…")

    # Path-traversal guard. ``system_router._safe_version_dir`` raises
    # HTTPException(400) on bad input; inside a worker we re-raise as a
    # plain RuntimeError so the JobStore captures it as the job error.
    try:
        version_dir = system_router._safe_version_dir(model_version)
    except HTTPException as exc:
        raise RuntimeError(f"Invalid model_version inside worker: {exc.detail}") from exc
    safe_version = version_dir.name

    if not version_dir.is_dir():
        raise RuntimeError(f"Model version '{safe_version}' not found")

    if is_cancelled():
        raise RuntimeError("Cancelled by user (before designer LLM call)")

    if callable(progress):
        progress(0.10, "Сборка контекста для designer…")

    designer_ctx, runtime_state = _build_designer_context(
        model_version=safe_version,
        task_text=task_text,
    )

    # Lazy imports keep the FastAPI cold-start path light and let tests
    # stub the modules before they're loaded. The factories raise
    # ``PromptNotFoundError`` at import time if prompts/ is missing —
    # the endpoint pre-validates this, so we shouldn't normally hit it.
    if callable(progress):
        progress(0.20, "Designer формулирует рецепты (~80 с)…")
    from app.backend.recipe_designer import make_recipe_designer

    designer = make_recipe_designer()
    if designer is None:
        # Late guard — the endpoint pre-checks the API key so we shouldn't
        # hit this. If we do, surface as a job error (not 503) because the
        # request was already accepted.
        raise RuntimeError(
            "RecipeDesigner unavailable: ANTHROPIC_API_KEY missing or "
            "anthropic SDK not installed."
        )

    recipes = designer.design(designer_ctx)
    # ``design`` swallows API errors and returns []. We don't promote that
    # to a job error — Streamlit shows an error message and bails (line
    # 1965-1967), but the API surfaces the empty list to the caller so the
    # UI can show the same "0 рецептов" panel without a generic 500.

    if is_cancelled():
        raise RuntimeError("Cancelled by user (after designer, before critic)")

    if callable(progress):
        progress(0.55, "ML+cost проверяет рецепты…")

    recipes_with_v: list[dict[str, Any]] = []
    if recipes:
        recipes_with_v = _verify_recipes(recipes, runtime_state=runtime_state)

    review_dicts: list[dict[str, Any]] = []
    if recipes_with_v:
        if callable(progress):
            progress(0.65, "Critic делает PhD-рецензию (~100 с)…")
        from app.backend.recipe_critic import make_recipe_critic

        critic = make_recipe_critic()
        if critic is None:
            raise RuntimeError(
                "RecipeCritic unavailable: ANTHROPIC_API_KEY missing or "
                "anthropic SDK not installed."
            )
        verdicts = critic.review(designer_ctx, recipes_with_v)
        review_dicts = [asdict(v) for v in verdicts]

    # Verdict counts — UI builds the summary strip from this. Streamlit
    # computes the same shape at line 2020-2022; we precompute server-side
    # so the JS reads numbers directly without filtering the reviews list.
    verdict_counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
    for r in review_dicts:
        v = r.get("verdict")
        if v in verdict_counts:
            verdict_counts[v] += 1

    decision_log_id: int | None = None
    if save_to_decision_log and recipes_with_v:
        if callable(progress):
            progress(0.95, "Сохраняю в Decision Log…")
        try:
            from decision_log.logger import log_decision

            decision_log_id = log_decision(
                phase="inverse_design",
                decision=(
                    f"Recipe cycle: {len(recipes_with_v)} рецептов, "
                    f"A={verdict_counts['ACCEPT']} "
                    f"R={verdict_counts['REVISE']} "
                    f"X={verdict_counts['REJECT']}"
                ),
                reasoning=(
                    f"model={safe_version}, "
                    f"baseline σf={runtime_state['baseline_predicted']:.0f} МПа, "
                    f"cost={runtime_state['baseline_cost']:.2f} €/т"
                ),
                context={
                    "model_version": safe_version,
                    "baseline": {
                        "recipe": runtime_state["baseline_recipe"],
                        "predicted_property": runtime_state["baseline_predicted"],
                        "cost_per_ton": runtime_state["baseline_cost"],
                    },
                    "recipes": recipes_with_v,
                    "reviews": review_dicts,
                    "verdict_counts": verdict_counts,
                    "task": task_text,
                },
                author="api",
                tags=["recipe_cycle", "sonnet-4-6"],
            )
        except Exception as exc:  # noqa: BLE001 — audit save is best-effort
            logger.warning("Decision Log save failed: %s", exc)

    if callable(progress):
        progress(1.0, "Готово")

    duration_s = round(time.monotonic() - started, 2)

    return {
        "model_version": safe_version,
        "baseline": {
            "recipe": runtime_state["baseline_recipe"],
            "predicted_property": runtime_state["baseline_predicted"],
            "cost_per_ton": runtime_state["baseline_cost"],
        },
        "recipes": recipes_with_v,
        "reviews": review_dicts,
        "verdict_counts": verdict_counts,
        "duration_s": duration_s,
        "decision_log_id": decision_log_id,
        "config": {
            "model_version": safe_version,
            "task_text": task_text,
            "save_to_decision_log": bool(save_to_decision_log),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/ai-cycle",
    response_class=SafeJSONResponse,
    response_model=None,
)
def ai_cycle(req: RecipesAiCycleRequest) -> dict[str, Any]:
    """Submit a recipe designer + critic cycle and return the job id.

    Validation order:
        1) Pydantic field bounds (task_text length). 422 on miss.
        2) ``_safe_version_dir`` — path traversal guard (CWE-22). 400.
        3) Model directory + meta.json present. 404.
        4) Steel class is fatigue_carbon_steel (Streamlit constraint). 400.
        5) ``check_llm_ready`` — ANTHROPIC_API_KEY + both prompt files.
           503 on miss.
        6) Submit job to the singleton store. Return ``{job_id, config}``.

    The cycle takes ~3 minutes (designer ~80 s + critic ~100 s + ML
    verification ~few s). The frontend polls ``GET /api/jobs/{job_id}``
    until status flips to ``done``, then renders the result.
    ``DELETE /api/jobs/{id}`` sets the cooperative cancellation flag —
    see module docstring for the cancellation semantics.
    """
    # 1-2. Path-traversal validation.
    version_dir = system_router._safe_version_dir(req.model_version)
    safe_version = version_dir.name

    # 3. Model existence.
    if not version_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{safe_version}' not found",
        )
    meta = system_router._read_model_meta(version_dir)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{safe_version}' is missing meta.json",
        )

    # 4. Class gate — fatigue_carbon_steel only.
    steel_class = meta.get("steel_class") or system_router.LEGACY_STEEL_CLASS_FALLBACK
    if steel_class != SUPPORTED_STEEL_CLASS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recipe pair currently supports only '{SUPPORTED_STEEL_CLASS}' "
                f"models (Agrawal NIMS dataset). Got steel_class='{steel_class}'."
            ),
        )

    # 5. Fail-fast LLM gating — return 503 with a clear remediation
    # message instead of a generic "job ended with error" once the
    # worker fails on a missing key or prompt.
    check_llm_ready(["recipe_designer", "recipe_critic"])

    # 6. Submit. We pass the validated ``safe_version`` (not the raw
    # request value) so future code paths cannot accidentally bypass the
    # regex check via direct ``run_as_job`` calls.
    job_id = run_as_job(
        _run_recipe_cycle_job,
        model_version=safe_version,
        task_text=str(req.task_text),
        save_to_decision_log=bool(req.save_to_decision_log),
    )
    return {
        "job_id": job_id,
        "config": {
            "model_version": safe_version,
            "task_text": str(req.task_text),
            "save_to_decision_log": bool(req.save_to_decision_log),
        },
    }
