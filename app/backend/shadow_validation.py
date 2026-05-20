"""Shadow validation comparison engine (Phase 2 S1).

Retrospective: для каждой historical плавки из heats.db вычисляет AI-рекомендацию
Al-дозы (EtaAlPredictor + compute_al_demand_slag_aware) и сравнивает с фактическим
al_added_kg. Comparison в pure-Al-эквиваленте.

Quality gate: AI не раскисляет хуже (target O_a) + фактический residual [Al] в spec +
conservative p90 доза достаточна. НЕ статистика (S2), НЕ report (S3), НЕ UI (S4).

Synthetic-heat tautology: на synthetic AI re-predicts свою η-формулу — реальная
валидация на real heats_db. Smoke демонстрирует механику.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.backend.heat_records import DEFAULT_DB_PATH, HeatRecord, list_heats
from app.backend.slag_aware_deox import (
    SlagState,
    compute_al_demand_slag_aware,
    load_addition_methods,
)

logger = logging.getLogger(__name__)

RESIDUAL_AL_SPEC_DEFAULT = (0.018, 0.040)  # HSLA pipeline residual [Al] %


@dataclass
class ShadowComparison:
    heat_pk: int | None
    heat_id: str | None
    plant_id: str
    method_id: str | None
    al_actual_kg: float | None        # pure-Al-equivalent
    al_ai_p50_kg: float | None
    al_ai_p90_kg: float | None
    delta_pct: float | None           # (ai_p50 - actual) / actual * 100
    delta_p90_pct: float | None
    ai_target_o_a_ppm: float | None
    actual_o_a_ppm: float | None
    residual_in_spec: bool
    dose_sufficient: bool
    quality_pass: bool
    eta_ai_source: str | None
    skip_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _residual_spec_for_class(steel_class_id: str | None) -> tuple[float, float]:
    """Residual [Al] spec range. Forward-compat: future SteelClassProfile field.
    Currently HSLA default for all classes."""
    # TODO: read residual_al_pct_range from SteelClassProfile when added
    return RESIDUAL_AL_SPEC_DEFAULT


def _al_content_pct_for_method(method_id: str | None, methods: dict) -> float:
    """% Al в charge-форме метода. Default 99% (commodity).

    Mirror app.backend.slag_aware_deox._al_content_pct_for_method but keyed by id
    (here we only have method_id + catalog dict, not the AdditionMethod object)."""
    if method_id and method_id in methods:
        return float(methods[method_id].raw.get("al_content_pct", 99.0))
    return 99.0


def _build_features_for_heat(heat: HeatRecord) -> dict:
    """HeatRecord → features dict для predictor. Mirror deox.py _build_eta_features."""
    f: dict[str, float] = {}
    if heat.slag_feo_pct is not None:
        f["slag_feo_pct"] = float(heat.slag_feo_pct)
    if heat.slag_mno_pct is not None:
        f["slag_mno_pct"] = float(heat.slag_mno_pct)
    if heat.slag_sio2_pct is not None:
        f["slag_sio2_pct"] = float(heat.slag_sio2_pct)
    if heat.slag_mass_kg is not None:
        f["slag_mass_kg"] = float(heat.slag_mass_kg)
    if heat.t_al_addition_c is not None:
        f["t_al_addition_c"] = float(heat.t_al_addition_c)
    f["o_a_initial_ppm"] = float(heat.o_a_initial_ppm)
    f["steel_mass_ton"] = float(heat.steel_mass_ton)
    if heat.co_deox_fesi_kg is not None:
        f["co_deox_fesi_kg"] = float(heat.co_deox_fesi_kg)
    if heat.c_pct is not None:
        f["c_pct"] = float(heat.c_pct)
    if heat.mn_pct is not None:
        f["mn_pct"] = float(heat.mn_pct)
    if heat.si_pct is not None:
        f["si_pct"] = float(heat.si_pct)
    return f


def _compare_one_heat(
    heat: HeatRecord,
    predictor: Any,
    methods: dict,
    residual_spec: tuple[float, float],
) -> ShadowComparison:
    """Compare AI recommendation vs actual для одной плавки."""
    base = ShadowComparison(
        heat_pk=heat.id,
        heat_id=heat.heat_id,
        plant_id=heat.plant_id,
        method_id=heat.method_id,
        al_actual_kg=None,
        al_ai_p50_kg=None,
        al_ai_p90_kg=None,
        delta_pct=None,
        delta_p90_pct=None,
        ai_target_o_a_ppm=None,
        actual_o_a_ppm=heat.o_a_after_ppm,
        residual_in_spec=True,
        dose_sufficient=False,
        quality_pass=False,
        eta_ai_source=None,
    )
    # Skip checks
    if heat.o_a_after_ppm is None or heat.al_added_kg is None:
        base.skip_reason = "no_outcome"
        return base
    if heat.al_added_kg == 0:
        base.skip_reason = "zero_al_actual"
        return base
    if heat.method_id is None or heat.method_id not in methods:
        base.skip_reason = "unknown_method"
        return base
    if heat.o_a_after_ppm >= heat.o_a_initial_ppm:
        base.skip_reason = "degenerate_o_balance"
        return base

    # Actual в pure-Al-эквивалент
    al_content = _al_content_pct_for_method(heat.method_id, methods)
    al_actual_pure = float(heat.al_added_kg) * al_content / 100.0
    base.al_actual_kg = al_actual_pure

    # AI recommendation
    lo, hi = residual_spec
    target_al_pct = (lo + hi) / 2.0
    features = _build_features_for_heat(heat)
    slag = None
    if heat.slag_mass_kg is not None and heat.slag_feo_pct is not None:
        slag = SlagState(
            mass_kg=float(heat.slag_mass_kg),
            feo_pct=float(heat.slag_feo_pct),
            mno_pct=float(heat.slag_mno_pct or 0.0),
            sio2_pct=float(heat.slag_sio2_pct or 0.0),
        )
    try:
        result = compute_al_demand_slag_aware(
            steel_mass_ton=float(heat.steel_mass_ton),
            o_a_initial_ppm=float(heat.o_a_initial_ppm),
            target_o_a_ppm=float(heat.o_a_after_ppm),
            target_al_pct=target_al_pct,
            method=heat.method_id,
            slag=slag,
            eta_al_predictor=predictor,
            plant_id=heat.plant_id,
            features_for_eta=features,
        )
    except Exception as exc:  # predictor / physics failure → skip, not crash batch
        base.skip_reason = f"predictor_error: {exc}"
        return base

    base.al_ai_p50_kg = result.al_pure_kg_p50
    base.al_ai_p90_kg = result.al_pure_kg_p90
    base.ai_target_o_a_ppm = float(heat.o_a_after_ppm)
    base.eta_ai_source = (
        result.inputs.get("predicted_eta_metadata") or {}
    ).get("source")

    # Deltas (pure vs pure)
    base.delta_pct = (result.al_pure_kg_p50 - al_actual_pure) / al_actual_pure * 100.0
    base.delta_p90_pct = (
        (result.al_pure_kg_p90 - al_actual_pure) / al_actual_pure * 100.0
    )

    # Quality gate
    # residual_in_spec: фактический al_residual_pct
    if heat.al_residual_pct is not None:
        base.residual_in_spec = lo <= heat.al_residual_pct <= hi
        if not base.residual_in_spec:
            base.notes.append(
                f"actual residual [Al]={heat.al_residual_pct:.3f}% вне spec [{lo},{hi}]"
            )
    else:
        base.residual_in_spec = True
        base.notes.append(
            "actual al_residual_pct отсутствует — residual gate не проверен"
        )

    # dose_sufficient: p90 > 0 и нет underdeox warning
    underdeox = any(
        "недораскисл" in w.lower() or "underdeox" in w.lower()
        for w in result.warnings
    )
    base.dose_sufficient = result.al_pure_kg_p90 > 0 and not underdeox

    # ai_target_o_a <= actual (trivially true под A1, но проверяем явно)
    o_a_ok = base.ai_target_o_a_ppm <= base.actual_o_a_ppm

    base.quality_pass = o_a_ok and base.residual_in_spec and base.dose_sufficient

    # Advisory если сильно меньше Al
    if base.delta_pct is not None and base.delta_pct < -15:
        base.notes.append(
            "AI рекомендует значительно меньше Al — проверьте inclusion flotation "
            "(super-stoich [Al]), Ca/S модификацию, AlN при высоком [N]"
        )
    return base


def run_shadow_comparison(
    heats: list[HeatRecord] | None,
    predictor: Any,
    *,
    steel_class_id: str | None = None,
    db_path: Path | None = None,
) -> list[ShadowComparison]:
    """Run shadow comparison. heats=None → list_heats(has_outcome=True)."""
    if heats is None:
        heats = list_heats(
            has_outcome=True,
            limit=100000,
            db_path=db_path or DEFAULT_DB_PATH,
        )
    methods = load_addition_methods()
    residual_spec = _residual_spec_for_class(steel_class_id)
    return [_compare_one_heat(h, predictor, methods, residual_spec) for h in heats]


__all__ = [
    "ShadowComparison",
    "RESIDUAL_AL_SPEC_DEFAULT",
    "run_shadow_comparison",
]
