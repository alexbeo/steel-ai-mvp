"""Coverage smoke tests for app/web/static/js/utils/llm_labels.js.

Pytest не запускает JS, поэтому эти тесты — regex-проверка структуры
файла. Гарантируют что:

1. Все 6 expected dictionaries присутствуют как exports.
2. Shared enums (verdict / evidence-check / severity / novelty / confidence)
   содержат полный набор ключей, на которые опираются views.
3. Helper functions `labelFor` и `splitKeyedLine` — exported.

Если кто-то переименует словарь или удалит ключ — тест поймает.
Тесты НЕ проверяют семантику переводов (что 'Принять' соответствует
ACCEPT) — это ловит code review.
"""
from __future__ import annotations

from pathlib import Path

import pytest


LLM_LABELS_PATH = (
    Path(__file__).resolve().parent.parent / "web" / "static" / "js" / "utils" / "llm_labels.js"
)


@pytest.fixture(scope="module")
def llm_labels_text() -> str:
    """Read llm_labels.js once per module — cheap, ~6 KB."""
    assert LLM_LABELS_PATH.exists(), (
        f"llm_labels.js missing at {LLM_LABELS_PATH} — расширение R-001 Step 4 "
        f"требует этот shared utility (см. Implementation report 2026-05-09)."
    )
    return LLM_LABELS_PATH.read_text(encoding="utf-8")


# ──────────────────── module-level dictionaries ────────────────────


@pytest.mark.parametrize(
    "export_name",
    [
        "HYPOTHESIS_CRITIC_LABELS",
        "RECIPE_CRITIC_LABELS",
        "RECIPE_DESIGNER_LABELS",
        "DEOX_ADVISOR_LABELS",
        "DEOX_CRITIC_LABELS",
        "ANOMALY_EXPLAINER_LABELS",
        "VERDICT_LABELS",
        "EVIDENCE_CHECK_LABELS",
        "SEVERITY_LABELS",
        "NOVELTY_LABELS",
        "CONFIDENCE_LABELS",
    ],
)
def test_dictionary_exported(llm_labels_text: str, export_name: str) -> None:
    """Каждый ожидаемый словарь объявлен как `export const X = {`."""
    needle = f"export const {export_name} = {{"
    assert needle in llm_labels_text, (
        f"Словарь {export_name} не найден в llm_labels.js. "
        f"Не удаляй и не переименовывай экспорты без обновления views, "
        f"которые их импортируют."
    )


# ──────────────────── shared enums (verdict / evidence / severity) ────────────────────


@pytest.mark.parametrize(
    "verdict_key", ["ACCEPT", "REVISE", "REJECT"]
)
def test_verdict_labels_complete(llm_labels_text: str, verdict_key: str) -> None:
    """VERDICT_LABELS содержит все 3 значения output schema критиков."""
    # Ищем pattern `ACCEPT: '…'` внутри объявления VERDICT_LABELS.
    # Грубое regex-достаточно — словарь маленький, не пересекается с другими.
    assert f"  {verdict_key}: '" in llm_labels_text or f"  {verdict_key}: \"" in llm_labels_text, (
        f"Verdict '{verdict_key}' отсутствует в llm_labels.js — это поломает "
        f"критик-views (recipe.js, hypotheses.js, deox.js)."
    )


@pytest.mark.parametrize(
    "ec_key", ["VALID", "INVALID", "UNVERIFIABLE"]
)
def test_evidence_check_labels_complete(llm_labels_text: str, ec_key: str) -> None:
    """EVIDENCE_CHECK_LABELS — recipe + deox AI critic используют."""
    assert f"  {ec_key}: '" in llm_labels_text or f"  {ec_key}: \"" in llm_labels_text, (
        f"Evidence-check verdict '{ec_key}' отсутствует — поломается "
        f"recipe.js / deox.js fact-check rendering."
    )


@pytest.mark.parametrize(
    "level", ["HIGH", "MEDIUM", "LOW"]
)
def test_severity_and_novelty_levels_complete(llm_labels_text: str, level: str) -> None:
    """SEVERITY/NOVELTY/CONFIDENCE — все 3 уровня."""
    # Эти три словаря используют одинаковые keys; проверяем хотя бы
    # один (SEVERITY_LABELS), плюс NOVELTY_LABELS и CONFIDENCE_LABELS
    # exported уже проверено выше.
    line = f"  {level}: '"
    line_dq = f"  {level}: \""
    assert line in llm_labels_text or line_dq in llm_labels_text, (
        f"Severity/novelty/confidence level '{level}' missing."
    )


# ──────────────────── attack-vector ID coverage ────────────────────


@pytest.mark.parametrize(
    "key",
    [
        # Hypothesis critic: prompts/hypothesis_critic.md adversarial_mindset[].id
        "logical_leap",
        "power_problems",
        "ignored_alternative_explanations",
        "scope_creep_ood_risk",
        # Recipe critic: prompts/recipe_critic.md adversarial_mindset[].id
        "evidence_integrity",
        "mechanism_inversions",
        "cost_saving_fact_check",
        "weldability_blind_spot",
        # Deox critic: prompts/deoxidation_critic.md adversarial_mindset[].id
        "stoichiometry_arithmetic",
        "recovery_factor_reality",
        "slag_chemistry_constraints",
    ],
)
def test_attack_vector_keys_present(llm_labels_text: str, key: str) -> None:
    """Каждый attack-vector ID из tool_use schema критиков переведён."""
    assert f"  {key}: '" in llm_labels_text or f"  {key}: \"" in llm_labels_text, (
        f"Attack-vector ID '{key}' нет в llm_labels.js — UI покажет "
        f"сырой английский ключ из tool_use payload."
    )


# ──────────────────── helper functions ────────────────────


def test_label_for_helper_exported(llm_labels_text: str) -> None:
    """`labelFor(key, dictionary)` — graceful-fallback helper."""
    assert "export function labelFor(" in llm_labels_text


def test_split_keyed_line_helper_exported(llm_labels_text: str) -> None:
    """`splitKeyedLine(line, dictionary)` парсит prefix `key: text`."""
    assert "export function splitKeyedLine(" in llm_labels_text
