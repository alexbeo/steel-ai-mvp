"""
Pattern Library — машиночитаемая версия анти-паттернов.

Каждый паттерн содержит:
- id: уникальный код
- phase: на какой фазе pipeline проверять
- severity: HIGH/MEDIUM/LOW
- check: функция проверки (возвращает CheckResult)
- suggestion: что делать при срабатывании

Critic-агент импортирует эту библиотеку и прогоняет релевантные проверки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import numpy as np


class Severity(str, Enum):
    HIGH = "HIGH"      # блокирующая — нельзя выдавать пользователю
    MEDIUM = "MEDIUM"  # серьёзная — предупреждение и документирование
    LOW = "LOW"        # минорная — в отчёт, но не блокирует


class Phase(str, Enum):
    DATA_ACQUISITION = "data_acquisition"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    INVERSE_DESIGN = "inverse_design"
    VALIDATION = "validation"
    REPORTING = "reporting"


@dataclass
class CheckResult:
    triggered: bool
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class Pattern:
    id: str
    title: str
    phase: Phase
    severity: Severity
    description: str
    check: Callable[[dict], CheckResult]
    suggestion: str


# =========================================================================
# D — Data patterns
# =========================================================================

def _check_d01_target_leakage(ctx: dict) -> CheckResult:
    """Детектирует экстремально высокую feature importance на одной фиче."""
    importance = ctx.get("feature_importance", {})
    if not importance:
        return CheckResult(False)
    top_feature, top_val = max(importance.items(), key=lambda x: x[1])
    total = sum(importance.values())
    if total == 0:
        return CheckResult(False)
    top_share = top_val / total
    if top_share > 0.7:
        return CheckResult(
            True,
            message=f"Feature '{top_feature}' накапливает {top_share:.0%} importance. "
                    f"Подозрение на target leakage или trivial predictor.",
            details={"top_feature": top_feature, "share": top_share},
        )
    return CheckResult(False)


def _check_d03_distribution_shift(ctx: dict) -> CheckResult:
    """Детектирует сдвиг распределения target между train и test."""
    y_train = ctx.get("y_train")
    y_test = ctx.get("y_test")
    if y_train is None or y_test is None:
        return CheckResult(False)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    mean_diff = abs(y_train.mean() - y_test.mean())
    std_pooled = np.sqrt((y_train.std() ** 2 + y_test.std() ** 2) / 2)
    if std_pooled == 0:
        return CheckResult(False)
    normalized_diff = mean_diff / std_pooled
    if normalized_diff > 0.5:
        return CheckResult(
            True,
            message=f"Распределение target на train и test сильно различается. "
                    f"Mean diff = {mean_diff:.2f}, что составляет {normalized_diff:.2f}σ.",
            details={"mean_train": y_train.mean(), "mean_test": y_test.mean()},
        )
    return CheckResult(False)


def _check_d04_unit_chaos(ctx: dict) -> CheckResult:
    """Детектирует бимодальность распределения (признак смешанных единиц)."""
    from scipy.stats import gaussian_kde
    df = ctx.get("dataframe")
    col = ctx.get("column_to_check")
    if df is None or col not in df.columns:
        return CheckResult(False)
    values = df[col].dropna().values
    if len(values) < 30:
        return CheckResult(False)
    try:
        kde = gaussian_kde(values)
        x = np.linspace(values.min(), values.max(), 500)
        y = kde(x)
        # Считаем локальные максимумы
        peaks = ((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])).sum()
        if peaks >= 2:
            return CheckResult(
                True,
                message=f"Колонка '{col}' имеет бимодальное распределение. "
                        f"Возможные причины: смешанные единицы (МПа/psi), разные методики, две субпопуляции.",
            )
    except Exception:
        return CheckResult(False)
    return CheckResult(False)


def _check_d06_random_split_in_temporal(ctx: dict) -> CheckResult:
    """Если есть время и использован random split — красный флаг."""
    has_time_column = ctx.get("has_time_column", False)
    split_strategy = ctx.get("split_strategy", "")
    if has_time_column and split_strategy == "random":
        return CheckResult(
            True,
            message="Использован random split, хотя в данных есть временная колонка. "
                    "Это создаёт data leakage через correlated heats.",
        )
    return CheckResult(False)


def _check_d07_physical_bounds(ctx: dict) -> CheckResult:
    """Проверяет базовые физические границы на входных данных."""
    df = ctx.get("dataframe")
    if df is None:
        return CheckResult(False)
    violations = []
    bounds = {
        "c_pct": (0.02, 2.1),
        "mn_pct": (0.0, 20.0),
        "si_pct": (0.0, 5.0),
        "p_pct": (0.0, 0.15),
        "s_pct": (0.0, 0.15),
        "yield_strength_mpa": (100, 3000),
        "tensile_strength_mpa": (150, 3500),
        "elongation_pct": (0, 85),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            out_of_bounds = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_bounds > 0:
                violations.append(f"{col}: {out_of_bounds} значений вне [{lo}, {hi}]")
    if violations:
        return CheckResult(
            True,
            message="Обнаружены значения вне физических границ:\n" + "\n".join(violations),
        )
    return CheckResult(False)


# =========================================================================
# M — Model patterns
# =========================================================================

def _check_m01_overfitting(ctx: dict) -> CheckResult:
    r2_train = ctx.get("r2_train")
    r2_val = ctx.get("r2_val")
    if r2_train is None or r2_val is None:
        return CheckResult(False)
    gap = r2_train - r2_val
    if gap > 0.15:
        return CheckResult(
            True,
            message=f"Overfitting: R² train = {r2_train:.3f}, R² val = {r2_val:.3f}, gap = {gap:.3f}. "
                    f"Модель запомнила train, но не обобщается.",
        )
    return CheckResult(False)


def _check_m02_uncertainty_calibration(ctx: dict) -> CheckResult:
    coverage_90 = ctx.get("coverage_90_ci")
    if coverage_90 is None:
        return CheckResult(False)
    if not (0.85 <= coverage_90 <= 0.95):
        return CheckResult(
            True,
            message=f"Calibration нарушена: 90% CI покрывает {coverage_90:.1%} точек (ожидалось 85-95%). "
                    f"{'Overconfident' if coverage_90 < 0.85 else 'Underconfident'} uncertainty.",
        )
    return CheckResult(False)


def _check_m04_no_uncertainty(ctx: dict) -> CheckResult:
    has_ci = ctx.get("prediction_has_ci", False)
    if not has_ci:
        return CheckResult(
            True,
            message="Модель возвращает только point estimate без CI. "
                    "Для safety-critical применений обязательно prediction interval.",
        )
    return CheckResult(False)


def _check_m05_feature_importance_sanity(ctx: dict) -> CheckResult:
    """
    Для HSLA-сталей ожидаемы в top-10: c_pct, mn_pct, nb_pct, ti_pct, v_pct,
    рулонные температуры. Если чего-то физически маргинального (например, Cu) 
    оказывается в топе — red flag.
    """
    importance = ctx.get("feature_importance", {})
    steel_class = ctx.get("steel_class", "")
    if not importance or steel_class != "pipe_hsla":
        return CheckResult(False)
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    expected_in_top = {"c_pct", "mn_pct", "nb_pct", "ti_pct", "v_pct",
                       "rolling_finish_temp", "cooling_rate_c_per_s",
                       "cev_iiw", "pcm", "cen"}
    top_names = {f[0] for f in top_features}
    overlap = top_names & expected_in_top
    if len(overlap) < 2:
        return CheckResult(
            True,
            message=f"Top-5 feature importance не содержит ожидаемых для HSLA фичей. "
                    f"Top: {[f[0] for f in top_features]}. "
                    f"Ожидалось увидеть минимум 2 из: {sorted(expected_in_top)}.",
        )
    return CheckResult(False)


def _check_m06_ood_detection(ctx: dict) -> CheckResult:
    has_ood_detector = ctx.get("ood_detector_configured", False)
    if not has_ood_detector:
        return CheckResult(
            True,
            message="OOD detector не настроен. Inverse design может возвращать кандидатов "
                    "вне training domain, где модель экстраполирует.",
        )
    return CheckResult(False)


def _check_m07_grouped_cv(ctx: dict) -> CheckResult:
    cv_strategy = ctx.get("cv_strategy", "")
    has_groups = ctx.get("has_groups", False)
    if has_groups and cv_strategy not in ("group_kfold", "stratified_group_kfold", "time_group"):
        return CheckResult(
            True,
            message=f"CV strategy = '{cv_strategy}' не учитывает группы (campaigns). "
                    "Это даёт оптимистично завышенный CV-score.",
        )
    return CheckResult(False)


# =========================================================================
# I — Inverse design patterns
# =========================================================================

def _check_i01_inverse_bounds_within_training(ctx: dict) -> CheckResult:
    bounds = ctx.get("variable_bounds", {})
    training_ranges = ctx.get("training_variable_ranges", {})
    if not bounds or not training_ranges:
        return CheckResult(False)
    violations = []
    for var, (lo_b, hi_b) in bounds.items():
        if var in training_ranges:
            lo_t, hi_t = training_ranges[var]
            margin = (hi_t - lo_t) * 0.1
            if lo_b < lo_t - margin or hi_b > hi_t + margin:
                violations.append(
                    f"{var}: bounds=[{lo_b},{hi_b}], training=[{lo_t},{hi_t}]"
                )
    if violations:
        return CheckResult(
            True,
            message=f"Optimization bounds выходят за training distribution более чем на 10%: "
                    + "; ".join(violations),
        )
    return CheckResult(False)


def _check_i02_normalized_objectives(ctx: dict) -> CheckResult:
    objectives_normalized = ctx.get("objectives_normalized", False)
    n_objectives = ctx.get("n_objectives", 0)
    if n_objectives > 1 and not objectives_normalized:
        return CheckResult(
            True,
            message="Multi-objective optimization без нормализации. Различные порядки величин "
                    "будут создавать bias — один objective может полностью подавить другие.",
        )
    return CheckResult(False)


def _check_i03_empty_pareto(ctx: dict) -> CheckResult:
    pareto_size = ctx.get("pareto_size", None)
    if pareto_size is not None and pareto_size == 0:
        return CheckResult(
            True,
            message="Pareto front пуст. ТЗ несовместимо или constraints слишком жёсткие. "
                    "Нужно вернуться к пользователю с trade-off анализом.",
        )
    if pareto_size is not None and pareto_size < 5:
        return CheckResult(
            True,
            message=f"Pareto front содержит {pareto_size} точек — слишком мало для выбора. "
                    "Возможно, bounds слишком узкие или population size мал.",
        )
    return CheckResult(False)


# =========================================================================
# V — Validation patterns
# =========================================================================

def _check_v01_tenant_validation(ctx: dict) -> CheckResult:
    is_production = ctx.get("is_production", False)
    tenant_config_loaded = ctx.get("tenant_config_loaded", False)
    if is_production and not tenant_config_loaded:
        return CheckResult(
            True,
            message="Validation запускается в production без загрузки tenant_config. "
                    "Generic ограничения не отражают реалии АКОС конкретного клиента.",
        )
    return CheckResult(False)


# =========================================================================
# Библиотека
# =========================================================================

PATTERNS: list[Pattern] = [
    Pattern(
        id="D01",
        title="Target leakage через derived feature",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="Одна фича собрала >70% feature importance — подозрение на leakage",
        check=_check_d01_target_leakage,
        suggestion="Проверить, не вычисляется ли топовая фича ИЗ target. Удалить и переобучить.",
    ),
    Pattern(
        id="D03",
        title="Distribution shift между train и test",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="Распределения target на train и test различаются > 0.5σ",
        check=_check_d03_distribution_shift,
        suggestion="Проверить временные тренды, изменение методики лаборатории, selection bias.",
    ),
    Pattern(
        id="D04",
        title="Unit chaos (бимодальное распределение)",
        phase=Phase.PREPROCESSING,
        severity=Severity.HIGH,
        description="Распределение колонки имеет 2+ пиков",
        check=_check_d04_unit_chaos,
        suggestion="Прогнать unit canonicalization. Проверить raw data на наличие разных единиц.",
    ),
    Pattern(
        id="D06",
        title="Random split в temporal data",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="Temporal data с random split",
        check=_check_d06_random_split_in_temporal,
        suggestion="Переключиться на time-based split: последние X% по дате — в test.",
    ),
    Pattern(
        id="D07",
        title="Физически невозможные значения",
        phase=Phase.PREPROCESSING,
        severity=Severity.HIGH,
        description="Значения вне physical bounds",
        check=_check_d07_physical_bounds,
        suggestion="Удалить либо исправить (опечатки типа 3.45→0.345) с audit log.",
    ),
    Pattern(
        id="M01",
        title="Overfitting",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="R² train − R² val > 0.15",
        check=_check_m01_overfitting,
        suggestion="Снизить сложность (max_depth, n_estimators), добавить регуляризацию, early stopping.",
    ),
    Pattern(
        id="M02",
        title="Плохая calibration uncertainty",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="90% CI покрывает не 85-95% точек",
        check=_check_m02_uncertainty_calibration,
        suggestion="Применить conformal prediction на hold-out для калибровки.",
    ),
    Pattern(
        id="M04",
        title="Отсутствие uncertainty в prediction",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="Модель возвращает только point estimate",
        check=_check_m04_no_uncertainty,
        suggestion="Добавить quantile regression (q05, q95) или ensemble variance.",
    ),
    Pattern(
        id="M05",
        title="Feature importance без физического смысла",
        phase=Phase.TRAINING,
        severity=Severity.MEDIUM,
        description="Top-5 importance не включает ожидаемые для HSLA features",
        check=_check_m05_feature_importance_sanity,
        suggestion="Проверить data leakage, spurious correlation, selection bias.",
    ),
    Pattern(
        id="M06",
        title="Отсутствие OOD detector",
        phase=Phase.INVERSE_DESIGN,
        severity=Severity.HIGH,
        description="OOD detector не настроен",
        check=_check_m06_ood_detection,
        suggestion="Обучить Gaussian Mixture на training composition. Flag на > 3σ от clusters.",
    ),
    Pattern(
        id="M07",
        title="CV без учёта групп",
        phase=Phase.TRAINING,
        severity=Severity.HIGH,
        description="Groups присутствуют, но не используется GroupKFold",
        check=_check_m07_grouped_cv,
        suggestion="Использовать GroupKFold или StratifiedGroupKFold по campaign/month.",
    ),
    Pattern(
        id="I01",
        title="Inverse bounds вне training distribution",
        phase=Phase.INVERSE_DESIGN,
        severity=Severity.HIGH,
        description="Variable bounds выходят за training range > 10%",
        check=_check_i01_inverse_bounds_within_training,
        suggestion="Сузить bounds до training range ± 10%. Модель плохо экстраполирует.",
    ),
    Pattern(
        id="I02",
        title="Objectives не нормализованы",
        phase=Phase.INVERSE_DESIGN,
        severity=Severity.HIGH,
        description="Multi-objective без normalization",
        check=_check_i02_normalized_objectives,
        suggestion="Применить MinMaxScaler к каждому objective перед NSGA-II.",
    ),
    Pattern(
        id="I03",
        title="Пустой или слишком маленький Pareto front",
        phase=Phase.INVERSE_DESIGN,
        severity=Severity.HIGH,
        description="Pareto size < 5",
        check=_check_i03_empty_pareto,
        suggestion="Ослабить constraints, расширить bounds, увеличить population size/generations.",
    ),
    Pattern(
        id="V01",
        title="Validation без tenant config",
        phase=Phase.VALIDATION,
        severity=Severity.MEDIUM,
        description="Production validation без tenant-specific constraints",
        check=_check_v01_tenant_validation,
        suggestion="Загрузить tenant_config. В MVP-демо — явно показать disclaimer.",
    ),
]


def run_all_patterns(ctx: dict, phase: Phase | None = None) -> list[dict]:
    """
    Главная функция для Critic: прогоняет все relevant patterns и возвращает warnings.
    """
    warnings = []
    for pattern in PATTERNS:
        if phase is not None and pattern.phase != phase:
            continue
        try:
            result = pattern.check(ctx)
            if result.triggered:
                warnings.append({
                    "pattern_id": pattern.id,
                    "title": pattern.title,
                    "severity": pattern.severity.value,
                    "message": result.message,
                    "suggestion": pattern.suggestion,
                    "details": result.details,
                })
        except Exception as e:
            warnings.append({
                "pattern_id": pattern.id,
                "title": pattern.title,
                "severity": "LOW",
                "message": f"Pattern check failed: {e}",
                "suggestion": "Check input context",
                "details": {},
            })
    return warnings


if __name__ == "__main__":
    # демо-прогон
    demo_context = {
        "r2_train": 0.95,
        "r2_val": 0.71,
        "coverage_90_ci": 0.62,
        "prediction_has_ci": False,
        "ood_detector_configured": False,
        "has_time_column": True,
        "split_strategy": "random",
        "pareto_size": 2,
    }
    warnings = run_all_patterns(demo_context)
    print(f"Обнаружено {len(warnings)} проблем:")
    for w in warnings:
        print(f"  [{w['severity']}] {w['pattern_id']}: {w['title']}")
        print(f"    {w['message'][:120]}")
        print(f"    → {w['suggestion']}")
        print()
