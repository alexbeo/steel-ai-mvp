"""
Оркестрационный движок платформы Steel AI MVP.

Связывает Orchestrator, executive agents, Critic, Pattern Library и Decision Log.
Запускается либо из CLI (scripts/run_pipeline.py), либо из FastAPI backend.

Главные принципы:
1. Orchestrator держит main loop и принимает решения.
2. Executive agents выполняют задачи (data, feature, train, inverse, validate, report).
3. После каждого шага запускается Critic.
4. Все решения записываются в Decision Log.
5. На HIGH warnings от Critic останавливаемся и эскалируем.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Any, Protocol

# Local imports — PROJECT_ROOT — это /steel-ai-mvp
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pattern_library.patterns import run_all_patterns, Phase, Severity
from decision_log.logger import (
    log_decision, query_decisions, update_outcome, summarize_project_history
)

logger = logging.getLogger(__name__)


def _extract_snapshot_meta(user_request: dict) -> dict | None:
    snap = user_request.get("price_snapshot")
    if snap is None:
        return None
    return {
        "date": snap.date.isoformat(),
        "currency": snap.currency,
        "source": snap.source,
        "n_materials": len(snap.materials),
    }


def _extract_snapshot_materials(user_request: dict) -> list[dict]:
    snap = user_request.get("price_snapshot")
    if snap is None:
        return []
    from dataclasses import asdict
    return [asdict(m) for m in snap.materials.values()]


def _elements_from_bounds(bounds: dict) -> set[str]:
    from app.backend.cost_model import required_elements_for_design
    return required_elements_for_design(bounds)


class Verdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    BLOCK = "BLOCK"


@dataclass
class CriticReport:
    phase: str
    verdict: Verdict
    warnings: list[dict] = field(default_factory=list)
    exploratory_observations: list[dict] = field(default_factory=list)
    requires_human_review: bool = False
    recommended_action: str = ""


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    output: dict
    error: str | None = None
    duration_s: float = 0


@dataclass
class PipelineState:
    """Состояние всего pipeline, передаётся между агентами."""
    dataset: dict = field(default_factory=dict)  # paths, stats
    features: dict = field(default_factory=dict)  # feature set info
    model: dict = field(default_factory=dict)     # model artifact, metrics
    candidates: list[dict] = field(default_factory=list)
    validated_candidates: list[dict] = field(default_factory=list)
    report_paths: dict = field(default_factory=dict)
    
    user_request: dict = field(default_factory=dict)
    critic_reports: list[CriticReport] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "features": self.features,
            "model": self.model,
            "n_candidates": len(self.candidates),
            "n_validated": len(self.validated_candidates),
            "report_paths": self.report_paths,
            "critic_reports_count": len(self.critic_reports),
        }


# =========================================================================
# Executive Agent Protocol
# =========================================================================

class ExecutiveAgent(Protocol):
    name: str
    def run(self, state: PipelineState, task: dict) -> AgentResult: ...


# =========================================================================
# Critic
# =========================================================================

class Critic:
    """
    Runs Pattern Library checks; when use_llm=True and an LLMCritic is
    configured (or ANTHROPIC_API_KEY is in env), also calls Claude Sonnet 4.6
    on the `training` phase for exploratory observations.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client=None,
        llm_critic=None,
    ):
        self.use_llm = use_llm
        self.llm_client = llm_client          # legacy param, retained for API compat
        self.llm_critic = llm_critic
        if self.use_llm and self.llm_critic is None:
            from app.backend.critic_llm import make_llm_critic
            self.llm_critic = make_llm_critic()

    def review(self, phase: str, context: dict) -> CriticReport:
        phase_enum = Phase(phase) if phase in {p.value for p in Phase} else None

        pattern_warnings = run_all_patterns(context, phase=phase_enum)

        high_count = sum(1 for w in pattern_warnings if w["severity"] == "HIGH")
        medium_count = sum(1 for w in pattern_warnings if w["severity"] == "MEDIUM")

        if high_count > 0:
            verdict = Verdict.BLOCK
        elif medium_count > 0:
            verdict = Verdict.PASS_WITH_WARNINGS
        else:
            verdict = Verdict.PASS

        exploratory_raw: list[dict] = []
        if self.use_llm and self.llm_critic and phase == "training":
            from dataclasses import asdict
            observations = self.llm_critic.review_training(context)
            exploratory_raw = [asdict(o) for o in observations]

        requires_human = high_count > 0 or any(
            not w.get("auto_fixable", True) for w in pattern_warnings
        )

        recommendations = [w["suggestion"] for w in pattern_warnings]
        recommended_action = "; ".join(recommendations[:3])

        return CriticReport(
            phase=phase,
            verdict=verdict,
            warnings=pattern_warnings,
            exploratory_observations=exploratory_raw,
            requires_human_review=requires_human,
            recommended_action=recommended_action,
        )


# =========================================================================
# Orchestrator
# =========================================================================

class HumanInTheLoopRequired(Exception):
    """Raised when pipeline needs user input to continue."""
    def __init__(self, question: str, context: dict):
        self.question = question
        self.context = context
        super().__init__(question)


class Orchestrator:
    """
    Главный координатор pipeline.
    
    Usage:
        orch = Orchestrator(agents={...}, critic=Critic())
        state = orch.run_pipeline(user_request={...})
    """
    
    def __init__(
        self,
        agents: dict[str, ExecutiveAgent],
        critic: Critic,
        human_in_the_loop: bool = True,
        on_human_checkpoint: Callable[[str, dict], str | None] | None = None,
    ):
        self.agents = agents
        self.critic = critic
        self.human_in_the_loop = human_in_the_loop
        # Callback для UI — возвращает user response или None для abort
        self.on_human_checkpoint = on_human_checkpoint
    
    def run_pipeline(self, user_request: dict, phases: list[str] | None = None) -> PipelineState:
        """
        Запускает end-to-end pipeline.
        
        user_request: что хочет пользователь
            {
                "task_type": "train" | "predict" | "inverse_design",
                "target_property": "yield_strength_mpa",
                "constraints": {...},
                "targets": {...},
            }
        phases: какие фазы запускать (default все)
        """
        state = PipelineState(user_request=user_request)
        
        # Старт сессии — читаем историю проекта
        history_summary = summarize_project_history()
        logger.info("Project history loaded:\n%s", history_summary[:500])
        
        log_decision(
            phase="meta",
            decision=f"Запуск pipeline для task '{user_request.get('task_type', 'unknown')}'",
            reasoning=f"User request: {json.dumps(user_request, ensure_ascii=False)}",
            author="orchestrator",
            tags=["pipeline_run"],
        )
        
        all_phases = phases or [
            "data_acquisition",
            "preprocessing",
            "feature_engineering",
            "training",
            "inverse_design",
            "validation",
            "reporting",
        ]
        
        for phase in all_phases:
            logger.info("=== Phase: %s ===", phase)
            try:
                self._run_phase(phase, state)
            except HumanInTheLoopRequired as e:
                # Передаём управление наверх (CLI / FastAPI)
                if self.human_in_the_loop and self.on_human_checkpoint:
                    response = self.on_human_checkpoint(e.question, e.context)
                    if response is None or response.lower() in ("abort", "stop", "no"):
                        logger.info("User aborted pipeline at %s", phase)
                        log_decision(
                            phase="meta",
                            decision=f"Pipeline aborted at phase {phase} by user",
                            reasoning="User declined to continue after checkpoint",
                            author="orchestrator",
                            tags=["abort"],
                        )
                        break
                    logger.info("User continued: %s", response)
                    # продолжаем phase (возможно с adjustments в state)
                    state.user_request["last_user_response"] = response
                    self._run_phase(phase, state, skip_checkpoint=True)
                else:
                    raise
        
        return state
    
    def _run_phase(self, phase: str, state: PipelineState, skip_checkpoint: bool = False) -> None:
        agent_for_phase = {
            "data_acquisition": "data_curator",
            "preprocessing": "data_curator",
            "feature_engineering": "feature_eng",
            "training": "model_trainer",
            "inverse_design": "inverse_designer",
            "validation": "validator",
            "reporting": "reporter",
        }
        agent_name = agent_for_phase.get(phase)
        if not agent_name or agent_name not in self.agents:
            logger.warning("No agent for phase %s", phase)
            return
        
        agent = self.agents[agent_name]
        task = self._build_task_for_phase(phase, state)
        result = agent.run(state, task)
        
        if not result.success:
            logger.error("Agent %s failed: %s", agent.name, result.error)
            log_decision(
                phase=phase,
                decision=f"Agent {agent.name} failed",
                reasoning=result.error or "unknown error",
                author="orchestrator",
                tags=["failure"],
            )
            raise RuntimeError(f"Agent {agent.name} failed: {result.error}")
        
        # Обновляем state из результата
        self._merge_result_into_state(phase, result, state)
        
        # Critic review
        context_for_critic = self._build_critic_context(phase, state, result)
        critic_report = self.critic.review(phase, context_for_critic)
        state.critic_reports.append(critic_report)
        
        logger.info(
            "Critic verdict for %s: %s (%d warnings)",
            phase, critic_report.verdict.value, len(critic_report.warnings)
        )
        for w in critic_report.warnings:
            logger.info("  [%s] %s: %s", w["severity"], w["pattern_id"], w["message"][:100])
        
        # Human-in-the-loop checkpoint
        if critic_report.verdict == Verdict.BLOCK and not skip_checkpoint:
            question = self._build_checkpoint_question(phase, critic_report)
            context = {
                "phase": phase,
                "warnings": critic_report.warnings,
                "recommended_action": critic_report.recommended_action,
                "state_summary": state.to_dict(),
            }
            raise HumanInTheLoopRequired(question, context)
    
    def _build_task_for_phase(self, phase: str, state: PipelineState) -> dict:
        """Формирует task-dict для конкретного агента."""
        user_request = state.user_request
        base_task = {"phase": phase, "user_request": user_request}
        
        if phase == "data_acquisition":
            return {**base_task, "operation": "download_nims_hsla"}
        if phase == "preprocessing":
            return {**base_task, "operation": "clean_and_validate"}
        if phase == "feature_engineering":
            return {**base_task, "operation": "compute_pipe_hsla_features",
                    "dataset_path": state.dataset.get("clean_path")}
        if phase == "training":
            return {
                **base_task,
                "operation": "train_xgboost",
                "target": user_request.get("target_property", "yield_strength_mpa"),
                "feature_set": state.features.get("feature_set_name"),
            }
        if phase == "inverse_design":
            return {
                **base_task,
                "operation": "run_nsga2",
                "targets": user_request.get("targets", {}),
                "hard_constraints": user_request.get("constraints", {}),
                "model_version": state.model.get("version"),
                "price_snapshot": user_request.get("price_snapshot"),
                "cost_mode": user_request.get("cost_mode", "full"),
            }
        if phase == "validation":
            return {
                **base_task,
                "operation": "validate_candidates",
                "candidates": state.candidates,
            }
        if phase == "reporting":
            return {
                **base_task,
                "operation": "generate_html_report",
                "candidates": state.validated_candidates,
                "model_info": state.model,
            }
        return base_task
    
    def _merge_result_into_state(self, phase: str, result: AgentResult, state: PipelineState) -> None:
        """Кладёт output агента в нужное поле state."""
        output = result.output
        if phase in ("data_acquisition", "preprocessing"):
            state.dataset.update(output)
        elif phase == "feature_engineering":
            state.features.update(output)
        elif phase == "training":
            state.model.update(output)
        elif phase == "inverse_design":
            state.candidates = output.get("pareto_candidates", [])
        elif phase == "validation":
            state.validated_candidates = output.get("approved", [])
        elif phase == "reporting":
            state.report_paths.update(output)
    
    def _build_critic_context(self, phase: str, state: PipelineState, result: AgentResult) -> dict:
        """Формирует context для Critic, собирая relevant метрики."""
        ctx = {"phase": phase, **result.output}
        if phase == "training":
            steel_class_id = result.output.get("steel_class", "pipe_hsla")
            try:
                from app.backend.steel_classes import load_steel_class
                profile = load_steel_class(steel_class_id)
                ctx["steel_class"] = steel_class_id
                ctx["expected_top_features"] = profile.expected_top_features
                ctx["physical_bounds"] = profile.physical_bounds
            except Exception:
                ctx["steel_class"] = steel_class_id
            ctx.update({
                "has_time_column": state.dataset.get("has_time_column", True),
                "has_groups": state.dataset.get("has_groups", True),
                "split_strategy": result.output.get("split_strategy", "unknown"),
                "cv_strategy": result.output.get("cv_strategy", "unknown"),
                "prediction_has_ci": result.output.get("has_uncertainty", False),
                "ood_detector_configured": result.output.get("has_ood_detector", False),
            })
        if phase == "inverse_design":
            ctx.update({
                "pareto_size": len(state.candidates),
                "objectives_normalized": result.output.get("objectives_normalized", False),
                "n_objectives": result.output.get("n_objectives", 1),
                "variable_bounds": result.output.get("variable_bounds", {}),
                "training_variable_ranges": state.features.get("training_ranges", {}),
                # Cost-optimization context (for C01-C04 patterns):
                "price_snapshot_meta": _extract_snapshot_meta(state.user_request),
                "snapshot_materials": _extract_snapshot_materials(state.user_request),
                "design_required_elements": sorted(
                    _elements_from_bounds(result.output.get("variable_bounds", {}))
                ),
                "cost_breakdown_samples": [
                    c.get("cost") for c in state.candidates[:5] if c.get("cost")
                ],
            })
        return ctx
    
    def _build_checkpoint_question(self, phase: str, critic_report: CriticReport) -> str:
        lines = [f"⚠️  Critic обнаружил проблемы на фазе '{phase}':\n"]
        for w in critic_report.warnings[:5]:
            lines.append(f"  [{w['severity']}] {w['pattern_id']}: {w['message'][:150]}")
            lines.append(f"    → {w['suggestion']}")
            lines.append("")
        lines.append(f"Рекомендованное действие: {critic_report.recommended_action}")
        lines.append("")
        lines.append("Продолжать ли? (yes / no / <custom instructions>)")
        return "\n".join(lines)


# =========================================================================
# Default CLI handler
# =========================================================================

def cli_checkpoint_handler(question: str, context: dict) -> str | None:
    """Простой CLI handler: печатает вопрос, ждёт ввода."""
    print("\n" + "=" * 70)
    print(question)
    print("=" * 70)
    try:
        response = input("> ").strip()
        return response if response else None
    except (EOFError, KeyboardInterrupt):
        return "abort"


if __name__ == "__main__":
    # Демонстрация без реальных агентов (dry run)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    class DummyAgent:
        def __init__(self, name, dummy_output):
            self.name = name
            self.dummy_output = dummy_output
        def run(self, state, task):
            logger.info("DummyAgent %s executing task %s", self.name, task.get("operation"))
            return AgentResult(agent_name=self.name, success=True, output=self.dummy_output)
    
    agents = {
        "data_curator": DummyAgent("data_curator", {
            "clean_path": "/tmp/data.parquet",
            "n_rows": 2847,
            "has_time_column": True,
            "has_groups": True,
        }),
        "feature_eng": DummyAgent("feature_eng", {
            "feature_set_name": "pipe_hsla_v2",
            "n_features": 24,
            "training_ranges": {"c_pct": [0.04, 0.12], "mn_pct": [1.0, 1.8]},
        }),
        "model_trainer": DummyAgent("model_trainer", {
            "version": "hsla_xgb_2026-04-19",
            "r2_train": 0.91,
            "r2_val": 0.86,
            "coverage_90_ci": 0.88,
            "has_uncertainty": True,
            "has_ood_detector": True,
            "feature_importance": {
                "c_pct": 0.22, "mn_pct": 0.18, "nb_pct": 0.14,
                "ti_pct": 0.11, "rolling_finish_temp": 0.10,
            },
            "split_strategy": "time_based",
            "cv_strategy": "group_kfold",
        }),
    }
    
    orch = Orchestrator(
        agents=agents,
        critic=Critic(use_llm=False),
        human_in_the_loop=True,
        on_human_checkpoint=cli_checkpoint_handler,
    )
    
    state = orch.run_pipeline(
        user_request={
            "task_type": "train",
            "target_property": "yield_strength_mpa",
        },
        phases=["data_acquisition", "feature_engineering", "training"],
    )
    
    print("\n" + "=" * 70)
    print("Pipeline finished. Final state:")
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str))
    print(f"\nCritic reports generated: {len(state.critic_reports)}")
    for r in state.critic_reports:
        print(f"  {r.phase}: {r.verdict.value} ({len(r.warnings)} warnings)")
