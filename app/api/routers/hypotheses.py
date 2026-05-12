"""Router for /api/hypotheses/* — neural-network observations on a trained model.

PR 10 of the Streamlit→FastAPI migration. See
``docs/superpowers/specs/2026-05-09_streamlit-to-fastapi-migration.md``
(Endpoint map → Tab «Гипотезы»).

Endpoint:
- ``POST /api/hypotheses/run`` — submit a hypothesis-generation cycle
  (Sonnet PhD generator → Sonnet PhD critic) as a long-running job.
  Returns ``{job_id}``; UI polls ``GET /api/jobs/{id}`` (PR 6) and
  reads ``result`` once status flips to ``done``.

PR 10 — endpoint convention note:
The single-LLM "+ critic peer review" cycle here runs as one paired
generator+critic pair, but the public name is ``run`` (not ``ai-cycle``)
following the convention spelled out in the design doc:

    «Convention для paired-LLM endpoints: <scope>/ai-cycle (PR 9
    deox/ai-cycle, PR 11 recipes/ai-cycle если применимо). Single-LLM
    endpoints (PR 10 hypotheses) — просто <scope>/run или
    <scope>/generate без ai-cycle суффикса.»

The hypotheses cycle has *one* generator-LLM call and *one* critic-LLM
call — but the user-visible action is "сгенерировать гипотезы" (generate
hypotheses), with the critic running automatically as a consistency
check. PR 9's ``deox/ai-cycle`` gates *operator action* on critic
approval (advisor's protocol is the artifact, critic vets it inline);
the hypotheses tab presents the generator's hypotheses as the artifact
and the critic's reviews as adjacent fact-check metadata. Different UX
contract → different endpoint suffix.

Cooperative cancellation:
Same single-gate semantics as PR 9 — DELETE before the generator LLM
call raises ``Cancelled by user (before generator LLM call)``. Once the
Anthropic SDK call is in flight we cannot interrupt; ``DELETE`` flips
``cancellation_requested`` and the worker burns through to completion
(UI stops polling). A second gate fires between the generator and the
critic so a cancel during the ~100 s generator call still aborts before
the critic burns its ~80 s round-trip.

Risks #3 (SafeJSONResponse): every endpoint declares
``response_class=SafeJSONResponse, response_model=None`` so dataclass +
numpy results from the LLM modules serialise via the custom encoder
instead of Pydantic v2's default revalidation.
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
# Request schema
# ──────────────────────────────────────────────────────────────────────


class HypothesesRunRequest(BaseModel):
    """Body for ``POST /api/hypotheses/run``.

    ``n_hypotheses`` mirrors the prompt's ``maxItems: 5`` cap (see
    ``hypothesis_generator.py:_TOOL_SCHEMA``). The schema there is
    immutable per cycle — Sonnet decides how many to emit between 0 and
    5 — but we expose the field so future prompt revisions that lift
    the cap can pass through a higher value without an API change.
    The UI currently fixes the count at 5; we accept 1-20 here so
    tests can exercise the lower edge and developer flows can ask for
    more if/when the prompt schema is widened.
    """

    model_version: str = Field(
        ...,
        description=(
            "Trained model version slug (matches ``models/<slug>/`` directory)."
            " Validated against the path-traversal regex in ``_safe_version_dir``."
        ),
    )
    n_hypotheses: int = Field(
        default=5, ge=1, le=20,
        description=(
            "Maximum hypotheses to request from the generator. "
            "Backend prompt currently caps at 5; values above are honoured "
            "only if the prompt schema is widened in a later release."
        ),
    )
    save_to_decision_log: bool = Field(
        default=False,
        description=(
            "Opt-in audit-trail save with tag 'hypothesis_cycle'. "
            "Default false to keep test/automation flows quiet — the UI "
            "exposes a «Сохранить» toggle to flip it explicitly."
        ),
    )

    model_config = {
        # ``model_version`` collides with Pydantic's ``model_*`` reserved
        # namespace warning; disable to keep the kwarg name stable across
        # the rest of the API surface (predict, design, active-learning).
        "protected_namespaces": (),
    }


# ──────────────────────────────────────────────────────────────────────
# Worker — invoked through ``run_as_job``
# ──────────────────────────────────────────────────────────────────────


def _run_hypothesis_job(
    *,
    model_version: str,
    n_hypotheses: int,
    save_to_decision_log: bool,
    progress: Any = None,
) -> dict[str, Any]:
    """Background worker for the hypothesis-generation cycle.

    Stages (all coarse-grained — no intra-LLM progress hooks):

        0.05 → "Подготовка контекста модели"
        0.15 → "Генератор формулирует гипотезы (~100 с)"
        0.55 → "Критик-PhD рецензирует (~45 с)"
        0.95 → "Сохраняю результаты"
        1.00 → done

    Cancellation gates fire at 0.05 (before generator), 0.55 (between
    generator and critic), and 0.95 (before save). Per the module
    docstring, the gates are effective only outside an active LLM call.

    The ``model_version`` is re-validated inside the worker via
    ``_safe_version_dir`` even though the endpoint already checked it,
    so any future code path that schedules this worker bypassing the
    endpoint (e.g. a future "rerun last cycle" button) still gets the
    path-traversal guard.

    Returns a dict consumed by the hypotheses view:
        {
          "model_version": str,
          "hypotheses": [Hypothesis-asdict, ...],
          "reviews": [CriticVerdict-asdict, ...],
          "verdict_counts": {"ACCEPT": int, "REVISE": int, "REJECT": int},
          "duration_s": float,
          "decision_log_id": int | None,
          "config": {...echo of inputs...},
        }
    """
    started = time.monotonic()

    is_cancelled = make_progress_cancel_check(progress)

    if callable(progress):
        progress(0.05, "Подготовка контекста модели")

    # Path-traversal guard. ``system_router._safe_version_dir`` raises
    # HTTPException(400) on bad input; inside a worker we re-raise as
    # a plain ValueError so the JobStore captures it as the job error.
    try:
        version_dir = system_router._safe_version_dir(model_version)
    except HTTPException as exc:
        # The endpoint already validated, so a re-fail here is unusual —
        # surface clearly so MLOps can grep logs.
        raise RuntimeError(f"Invalid model_version inside worker: {exc.detail}") from exc
    safe_version = version_dir.name

    if not version_dir.is_dir():
        raise RuntimeError(f"Model version '{safe_version}' not found")

    if is_cancelled():
        raise RuntimeError("Cancelled by user (before generator LLM call)")

    # Build the artifact context — same shape the standalone
    # ``scripts/generate_hypotheses_for_model.py`` uses. We reuse
    # ``build_context`` so any future schema change in the generator's
    # input lands in one place.
    if callable(progress):
        progress(0.10, "Загрузка модели и датасета")
    from scripts.generate_hypotheses_for_model import build_context

    ctx = build_context(safe_version)

    # Lazy imports keep the FastAPI cold-start path light and let tests
    # stub the modules before they're loaded. The factories raise
    # ``PromptNotFoundError`` at import time if prompts/ is missing —
    # the endpoint pre-validates this, so we shouldn't normally hit it.
    if callable(progress):
        progress(0.15, "Генератор формулирует гипотезы (~100 с)…")
    from app.backend.hypothesis_generator import make_hypothesis_generator

    generator = make_hypothesis_generator()
    if generator is None:
        # Late guard — the endpoint pre-checks the API key so we
        # shouldn't hit this. If we do, surface as a job error (not 503)
        # because the request was already accepted.
        raise RuntimeError(
            "HypothesisGenerator unavailable: ANTHROPIC_API_KEY missing or "
            "anthropic SDK not installed."
        )

    hypotheses = generator.generate(ctx)
    # ``generate`` swallows API errors and returns []. We don't promote
    # that to a job error — the UI shows a "0 hypotheses" message in
    # the same case.

    if is_cancelled():
        raise RuntimeError("Cancelled by user (after generator, before critic)")

    if callable(progress):
        progress(0.55, "Критик-PhD рецензирует (~45 с)…")

    hyp_dicts = [asdict(h) for h in hypotheses]

    review_dicts: list[dict[str, Any]] = []
    if hypotheses:
        # Only call the critic when there's something to review — the
        # critic's ``review`` is a no-op on empty input (returns []),
        # but skipping the LLM round-trip saves ~$0.06 + ~45 s.
        from app.backend.hypothesis_critic import make_hypothesis_critic

        critic = make_hypothesis_critic()
        if critic is None:
            # Same late-guard rationale as the generator above.
            raise RuntimeError(
                "HypothesisCritic unavailable: ANTHROPIC_API_KEY missing or "
                "anthropic SDK not installed."
            )
        verdicts = critic.review(ctx, hyp_dicts)
        review_dicts = [asdict(v) for v in verdicts]

    # Verdict counts — UI builds the metric strip ("Гипотез / ACCEPT /
    # REVISE / REJECT") from this. We precompute server-side so the
    # JS can read numbers directly without filtering the reviews array.
    verdict_counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
    for r in review_dicts:
        v = r.get("verdict")
        if v in verdict_counts:
            verdict_counts[v] += 1

    # Opt-in Decision Log save. Same "best-effort" envelope as PR 9:
    # an SQLite hiccup must not fail the whole cycle when the LLM
    # outputs are already in the response.
    decision_log_id: int | None = None
    if save_to_decision_log:
        if callable(progress):
            progress(0.95, "Сохраняю в Decision Log…")
        try:
            from decision_log.logger import log_decision

            decision_log_id = log_decision(
                phase="training",
                decision=(
                    f"Hypothesis cycle: {len(hyp_dicts)} гипотез, "
                    f"вердикты A={verdict_counts['ACCEPT']} "
                    f"R={verdict_counts['REVISE']} "
                    f"X={verdict_counts['REJECT']}"
                ),
                reasoning=(
                    f"Model={safe_version}, "
                    f"hypotheses={len(hyp_dicts)}, "
                    f"reviews={len(review_dicts)}"
                ),
                context={
                    "model_version": safe_version,
                    "hypotheses": hyp_dicts,
                    "reviews": review_dicts,
                    "verdict_counts": verdict_counts,
                },
                author="api",
                tags=["hypothesis_cycle", "sonnet-4-6"],
            )
        except Exception as exc:  # noqa: BLE001 — audit save is best-effort
            logger.warning("Decision Log save failed: %s", exc)

    if callable(progress):
        progress(1.0, "Готово")

    duration_s = round(time.monotonic() - started, 2)

    return {
        "model_version": safe_version,
        "hypotheses": hyp_dicts,
        "reviews": review_dicts,
        "verdict_counts": verdict_counts,
        "duration_s": duration_s,
        "decision_log_id": decision_log_id,
        "config": {
            "model_version": safe_version,
            "n_hypotheses": int(n_hypotheses),
            "save_to_decision_log": bool(save_to_decision_log),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/run",
    response_class=SafeJSONResponse,
    response_model=None,
)
def run_hypotheses(req: HypothesesRunRequest) -> dict[str, Any]:
    """Submit a hypothesis-generation cycle and return the job id.

    Validation order:
        1) Pydantic field bounds (n_hypotheses ∈ [1, 20]). 422 on miss.
        2) ``_safe_version_dir`` — path traversal guard (CWE-22). 400.
        3) Model directory + meta.json present. 404 on miss.
        4) ``check_llm_ready`` — ANTHROPIC_API_KEY + both prompt files.
           503 on either miss.
        5) Submit job to the singleton store. Return ``{job_id, config}``.

    The cycle takes ~3 minutes (generator ~100 s + critic ~45 s + IO).
    The frontend polls ``GET /api/jobs/{job_id}`` (PR 6) until status
    flips to ``done``, then renders the result. ``DELETE /api/jobs/{id}``
    sets the cooperative cancellation flag — see module docstring for
    the cancellation semantics (effective only between LLM calls).
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

    # 4. Fail-fast LLM gating — return 503 with a clear remediation
    # message instead of a generic "job ended with error" once the
    # worker fails on a missing key or prompt.
    check_llm_ready(["hypothesis_generator", "hypothesis_critic"])

    # 5. Submit. We pass the validated ``safe_version`` (not the raw
    # request value) so future code paths cannot accidentally bypass
    # the regex check via direct ``run_as_job`` calls.
    job_id = run_as_job(
        _run_hypothesis_job,
        model_version=safe_version,
        n_hypotheses=int(req.n_hypotheses),
        save_to_decision_log=bool(req.save_to_decision_log),
    )
    return {
        "job_id": job_id,
        "config": {
            "model_version": safe_version,
            "n_hypotheses": int(req.n_hypotheses),
            "save_to_decision_log": bool(req.save_to_decision_log),
        },
    }
