"""High-Al cohort diagnoser runner (PR 12 Block IV) — CLI-only discovery tool.

Собирает плавки (synthetic deox calibration dataset ИЛИ heats.db),
находит cohort с аномально высоким удельным расходом Al, строит
deterministic cohort-context (feature_deltas) и опционально зовёт Sonnet
для root-cause диагноза.

Без ANTHROPIC_API_KEY / без prompts/high_al_diagnoser.md — печатает
deterministic cohort + feature_deltas (LLM-слой degrade gracefully).

Run:
    PYTHONPATH=. .venv/bin/python scripts/explain_high_al_heats.py \\
        --source synthetic --method percentile
    PYTHONPATH=. ANTHROPIC_API_KEY=... .venv/bin/python \\
        scripts/explain_high_al_heats.py --source heats_db --method auto
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Целевой O_a после раскисления для synthetic derivation (Al-killed steel ~5 ppm)
_SYNTHETIC_TARGET_O_A_PPM = 5.0


def _derive_synthetic_al(df) -> list[dict]:
    """Synthetic dataset не содержит al_added_kg — деривируем из physics.

    Обратная формула (см. deoxidation.compute_al_demand):
        delta_o_kg   = (o_a_initial − o_a_after) / 1e6 × steel_mass_ton × 1000
        al_active_kg = delta_o_kg × AL_TO_O_MASS_RATIO
        al_added_kg  = al_active_kg / eta_al_effective   # η поглощает burn-off+purity

    o_a_after берём как фиксированный target (~5 ppm, Al-killed steel) —
    в synthetic его нет. Так удельный расход монотонно отражает 1/η, что и
    есть сигнал для outlier-detection.
    """
    from app.backend.deoxidation import AL_TO_O_MASS_RATIO

    heats: list[dict] = []
    for _, row in df.iterrows():
        o_init = float(row["o_a_initial_ppm"])
        mass = float(row["steel_mass_ton"])
        eta = float(row["eta_al_effective"])
        o_after = min(_SYNTHETIC_TARGET_O_A_PPM, o_init * 0.5)
        delta_o_kg = (o_init - o_after) / 1e6 * mass * 1000.0
        al_active_kg = delta_o_kg * AL_TO_O_MASS_RATIO
        al_added_kg = al_active_kg / eta if eta > 0 else None
        h = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in df.columns}
        h["al_added_kg"] = round(al_added_kg, 1) if al_added_kg else None
        h["o_a_after_ppm"] = round(o_after, 2)
        heats.append(h)
    return heats


def _load_synthetic(n_heats: int) -> list[dict]:
    from app.backend.data_curator import generate_synthetic_deox_calibration_dataset

    df = generate_synthetic_deox_calibration_dataset(n_heats=n_heats)
    return _derive_synthetic_al(df)


def _load_heats_db() -> list[dict]:
    """heats.db source — al_added_kg есть напрямую в HeatRecord."""
    from app.backend.heat_records import list_heats

    records = list_heats(has_outcome=True, limit=2000)
    return [r.model_dump() for r in records]


def _resolve_method(method: str) -> str:
    """--method auto → 'prediction' если есть real η_Al ML модель, иначе 'percentile'."""
    if method != "auto":
        return method
    try:
        from app.backend.eta_al_predictor import EtaAlPredictor

        predictor = EtaAlPredictor()
        pred = predictor.predict_eta_al(plant_id="PLANT_A", method_id="asis_shot")
        if pred.source != "literature_fallback":
            logger.info("auto → prediction (model source=%s)", pred.source)
            return "prediction"
    except Exception as exc:
        logger.warning("auto-detect model failed (%s) — fallback to percentile", exc)
    logger.info("auto → percentile (no ML model)")
    return "percentile"


def main() -> None:
    parser = argparse.ArgumentParser(description="High-Al cohort root-cause diagnoser")
    parser.add_argument("--source", choices=["synthetic", "heats_db"], default="synthetic")
    parser.add_argument("--method", choices=["auto", "percentile", "prediction"], default="auto")
    parser.add_argument("--threshold-pct", type=float, default=120.0)
    parser.add_argument("--n-heats", type=int, default=500, help="synthetic only")
    parser.add_argument("--cap-n", type=int, default=40)
    parser.add_argument("--min-cohort", type=int, default=5)
    args = parser.parse_args()

    from app.backend.high_al_diagnoser import (
        build_cohort_context,
        identify_high_al_outliers,
        make_high_al_diagnoser,
    )

    heats = _load_synthetic(args.n_heats) if args.source == "synthetic" else _load_heats_db()
    if not heats:
        logger.error("Нет плавок в source=%s — нечего анализировать.", args.source)
        return
    logger.info("Загружено %d плавок из %s", len(heats), args.source)

    method = _resolve_method(args.method)
    outliers, baseline = identify_high_al_outliers(
        heats, method=method, threshold_pct=args.threshold_pct, min_cohort=args.min_cohort,
    )
    logger.info("Outliers=%d, baseline=%d (method=%s)", len(outliers), len(baseline), method)

    if not outliers:
        logger.warning("Outlier-cohort пуст — нет аномалий или мало данных в стратах.")
        return

    ctx = build_cohort_context(
        outliers, baseline, method=method, threshold_pct=args.threshold_pct, cap_n=args.cap_n,
    )

    print()
    print("=" * 78)
    print(f"High-Al cohort: {ctx['n_outliers']}/{ctx['n_total']} плавок (method={method})")
    print("=" * 78)
    print("\nFeature deltas (outlier vs baseline, sorted by |delta|):")
    for d in ctx["feature_deltas"]:
        print(
            f"  {d['feature']:<28} outlier={d['outlier_mean']:>10.3f}  "
            f"baseline={d['baseline_mean']:>10.3f}  Δ={d['delta_pct']:>+7.1f}%  ({d['direction']})"
        )

    diagnoser = make_high_al_diagnoser()
    if diagnoser is None:
        print("\n[i] LLM-диагноз недоступен (нет ANTHROPIC_API_KEY или prompts/high_al_diagnoser.md).")
        print("    Deterministic cohort выше — главный сигнал в feature_deltas.")
        return

    print("\nВызываю Sonnet HighAlDiagnoser...")
    diagnosis = diagnoser.diagnose(ctx)
    if diagnosis is None:
        logger.error("Diagnoser вернул None (API failure или bad payload).")
        return

    print()
    print("=" * 78)
    print(f"Root-cause диагноз — severity: {diagnosis.cohort_severity}")
    print("=" * 78)
    print(f"\nРезюме: {diagnosis.summary}\n")
    for i, h in enumerate(diagnosis.hypotheses, 1):
        print(f"[{i}] {h.root_cause}  (confidence={h.confidence}, ~{h.est_excess_al_pct:.0f}% excess)")
        print(f"    Механизм: {h.mechanism}")
        print(f"    Доказательства: {', '.join(h.evidence_features)}")
        print(f"    Рекомендация: {h.suggested_action}")
        print()


if __name__ == "__main__":
    main()
