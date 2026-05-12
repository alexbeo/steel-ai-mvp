"""Shared LLM-gating + cooperative-cancellation utilities for routers.

PR 10 of the Streamlit→FastAPI migration. Extracted from PR 9
(``app/api/routers/deox.py``) and PR 8 (``app/api/routers/train.py``)
to remove the three duplicated helper copies that grew across the
LLM-backed routers (deox AI cycle, hypotheses cycle, recipes cycle).

Two helpers live here:

1. :func:`check_llm_ready` — fail-fast 503 guard that endpoints call
   *before* submitting a job to ensure the worker won't immediately
   error on missing API key or missing prompt files. Surfaces a 503
   with a remediation message instead of a generic "job ended with
   error" — significantly clearer UX in the frontend.

2. :func:`make_progress_cancel_check` — closure-introspecting predicate
   that walks back from a JobStore-injected progress callback to the
   cooperative cancellation flag. Replaces the per-router
   ``_check_cancelled`` helpers in ``deox.py`` and ``train.py`` with
   a single source of truth so a future ``_make_progress_cb`` refactor
   in ``app/api/jobs.py`` only breaks one place.

Module is intentionally lightweight (no FastAPI app dependency, just
``HTTPException``) so it can be imported by router modules without
worrying about circulars. It does not own router state.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from fastapi import HTTPException

# Re-import lazily inside the function to keep the module's import-cost
# tiny — these are only needed when an endpoint actually performs the
# pre-submit guard.


def check_llm_ready(prompt_files: Iterable[str]) -> None:
    """Validate ANTHROPIC_API_KEY + prompt presence; raise 503 on miss.

    Mirrors the contract established by PR 9 ``deox.py:_llm_ready_or_503``.
    Endpoints call this *synchronously* before scheduling a job so the
    user gets an immediate, actionable banner rather than a job-error
    surfaced 1-2 seconds later through the polling cycle.

    Two checks, in order:

    1. ``ANTHROPIC_API_KEY`` env var present. The Anthropic SDK and
       the project's ``make_*`` factories defer-fail without it
       (returning ``None`` from the factory) — we check up-front for
       a clearer 503 message.
    2. Each name in ``prompt_files`` resolves through
       ``app.backend.prompt_loader.load_prompt`` without raising
       :class:`PromptNotFoundError`. Public clones may be missing the
       prompts (gitignored intellectual property).

    The function imports ``load_prompt`` lazily so a missing
    ``app.backend.prompt_loader`` (highly unusual — it's a shipped
    module) doesn't crash the API at startup.

    Args:
        prompt_files: iterable of prompt names (without the ``.md``
            extension). Examples: ``["hypothesis_generator",
            "hypothesis_critic"]``, ``["deoxidation_advisor",
            "deoxidation_critic"]``.

    Raises:
        HTTPException: 503 with a Russian, actionable detail message.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM не настроен — ANTHROPIC_API_KEY отсутствует в окружении. "
                "Добавьте ключ в .env и перезапустите uvicorn."
            ),
        )

    # Prompt presence check — load_prompt is cached via lru_cache, so
    # this is cheap on the warm path and clear on the cold one.
    from app.backend.prompt_loader import PromptNotFoundError, load_prompt

    for name in prompt_files:
        try:
            load_prompt(name)
        except PromptNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Prompt {name}.md не найден. Промпты — gitignored "
                    f"intellectual property; обратитесь к владельцу проекта. "
                    f"({exc})"
                ),
            ) from exc


def make_progress_cancel_check(progress: Any) -> Callable[[], bool]:
    """Build a zero-arg predicate that reads the job's cancellation flag.

    Background: ``app.api.jobs.JobStore._make_progress_cb`` builds the
    progress callback as a closure over ``job_id``. Long-running ops
    that want to peek at ``Job.cancellation_requested`` can walk back
    through ``progress.__closure__`` to recover the id, then look up
    the live :class:`Job` via the singleton store. This is the price
    of keeping JobStore out of every worker function signature.

    The walk uses ``__code__.co_freevars`` rather than positional
    ``__closure__[0]`` so the helper stays correct even if a future
    refactor adds extra free variables to the closure (which is
    exactly the failure mode flagged by the PR 8/PR 9 ``TODO``s).

    Returns a no-arg callable that returns ``True`` iff cancellation
    was requested. Returns a callable that always returns ``False``
    when:
      * ``progress`` is ``None`` (no JobStore wired — sync path / tests).
      * Closure introspection fails (built-ins, partials with no
        recoverable closure).
      * The job_id resolves but the store no longer has it (evicted).

    The "always-False" fallback is a deliberate silent-failure mode:
    cancellation becomes a no-op rather than crashing the worker if
    the JobStore internals change. The caller sees a regression in
    cancel responsiveness — better than a hard crash mid-training.

    Args:
        progress: the ``progress`` callback that JobStore injected
            into ``fn``, or ``None`` when running outside a job.

    Returns:
        A zero-arg ``() -> bool`` predicate. Safe to call repeatedly
        from inside long-running loops (e.g. Optuna per-trial hooks)
        — each call performs one dict lookup against the singleton
        store, no I/O.
    """
    if progress is None:
        # Sync / test path — return a constant False predicate so callers
        # can pass the result unconditionally without branching.
        return lambda: False

    # Resolve job_id once, then close over it. Doing the introspection
    # per-call would waste a few microseconds on every Optuna trial; the
    # closure layout is fixed once the JobStore creates the callback.
    code = getattr(progress, "__code__", None)
    job_id: str | None = None
    if code is not None and "job_id" in getattr(code, "co_freevars", ()):
        try:
            idx = code.co_freevars.index("job_id")
            job_id = progress.__closure__[idx].cell_contents  # type: ignore[index]
            job_id = str(job_id)
        except (AttributeError, IndexError, TypeError):
            job_id = None

    if job_id is None:
        return lambda: False

    # Defer the JobStore import to first-call time — keeps this module
    # importable without pulling the executor in (helpful for tests
    # that import only the gating helpers).
    from app.api.jobs import get_job_store

    def _is_cancelled() -> bool:
        job = get_job_store().get(job_id)  # type: ignore[arg-type]
        return bool(job and job.cancellation_requested)

    return _is_cancelled
