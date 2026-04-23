"""
Al deoxidation calculator — physics-based advisory for ladle furnace.

Three common thermodynamic models for Al-O equilibrium in liquid steel:
  - Fruehan 1985        (classic, default)
  - Sigworth-Elliott 1974 (widely cited via JSPS Sourcebook)
  - Hayashi-Yamamoto 2013 (modern revision for high-Al range)

Forward: required Al mass to reduce O-activity from X to Y ppm.
Inverse: effective Al purity deduced from observed deoxidation depth.

No ML. No feedback loop. Honest advisory — point-estimates with
model-disagreement visualization (compare_all_models).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Atomic masses (g/mol)
M_AL = 26.98
M_O = 16.00
# Stoichiometric ratio: 2 Al + 3 O = Al2O3 →  1 kg O binds 2*M_AL/(3*M_O) kg Al
AL_TO_O_MASS_RATIO = 2.0 * M_AL / (3.0 * M_O)   # ≈ 1.12417


@dataclass(frozen=True)
class ThermoModel:
    id: str
    name: str
    citation: str
    log_k: Callable[[float], float]    # log10(K) as function of T_K
    valid_t_range_k: tuple[float, float]
    expected_accuracy_ppm: float
    al_al_correction: bool = False     # Hayashi quadratic term for [Al]>0.05%


THERMO_MODELS: dict[str, ThermoModel] = {
    "fruehan_1985": ThermoModel(
        id="fruehan_1985",
        name="Fruehan 1985",
        citation="Fruehan R., Ladle Metallurgy, ISS 1985",
        log_k=lambda T_K: 64000.0 / T_K - 20.57,
        valid_t_range_k=(1773.0, 1923.0),
        expected_accuracy_ppm=40.0,
    ),
    "sigworth_elliott_1974": ThermoModel(
        id="sigworth_elliott_1974",
        name="Sigworth-Elliott 1974",
        citation="JSPS Steelmaking Data Sourcebook, 1988 (Sigworth & Elliott 1974)",
        log_k=lambda T_K: 62680.0 / T_K - 20.54,
        valid_t_range_k=(1773.0, 1923.0),
        expected_accuracy_ppm=30.0,
    ),
    "hayashi_2013": ThermoModel(
        id="hayashi_2013",
        name="Hayashi-Yamamoto 2013",
        citation="Hayashi M., Yamamoto T., ISIJ Intl. 53, 2013",
        log_k=lambda T_K: -62780.0 / T_K + 19.18,
        valid_t_range_k=(1823.0, 1973.0),
        expected_accuracy_ppm=20.0,
        al_al_correction=True,
    ),
}

DEFAULT_MODEL_ID = "fruehan_1985"


@dataclass
class DeoxidationResult:
    al_total_kg: float
    al_active_kg: float
    al_burn_off_kg: float
    o_a_expected_ppm: float
    al_per_ton: float
    cost_eur: float
    currency: str
    model_id: str
    inputs: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class AlQualityResult:
    effective_purity_pct: float
    effective_active_kg: float
    expected_active_kg: float
    assumed_burn_off_pct: float
    model_id: str
    inputs: dict
    warnings: list[str] = field(default_factory=list)


def compute_al_demand(
    o_a_initial_ppm: float,
    temperature_C: float,
    steel_mass_ton: float,
    target_o_a_ppm: float,
    al_purity_pct: float = 100.0,
    burn_off_pct: float = 20.0,
    model_id: str = DEFAULT_MODEL_ID,
    al_price_per_kg: float = 2.40,
    currency: str = "EUR",
) -> DeoxidationResult:
    """Forward: how much Al to add to reduce O_a from initial to target."""
    inputs = {
        "o_a_initial_ppm": o_a_initial_ppm,
        "temperature_C": temperature_C,
        "steel_mass_ton": steel_mass_ton,
        "target_o_a_ppm": target_o_a_ppm,
        "al_purity_pct": al_purity_pct,
        "burn_off_pct": burn_off_pct,
        "model_id": model_id,
    }
    warnings: list[str] = []

    if model_id not in THERMO_MODELS:
        raise ValueError(f"Unknown thermo model: {model_id}")
    model = THERMO_MODELS[model_id]

    if not (50.0 <= o_a_initial_ppm <= 800.0):
        warnings.append(
            f"O_a = {o_a_initial_ppm:.0f} ppm вне физического диапазона "
            f"50-800 ppm для LF — проверьте датчик."
        )
    T_K = temperature_C + 273.15
    lo_K, hi_K = model.valid_t_range_k
    if not (lo_K <= T_K <= hi_K):
        warnings.append(
            f"Temperature {temperature_C:.0f}°C вне валидного диапазона модели "
            f"{model.name}: {lo_K-273.15:.0f}-{hi_K-273.15:.0f}°C."
        )
    if target_o_a_ppm >= o_a_initial_ppm:
        warnings.append(
            f"Target O_a ({target_o_a_ppm}) >= initial ({o_a_initial_ppm}) — "
            f"раскисление не требуется."
        )
        return DeoxidationResult(
            al_total_kg=0.0, al_active_kg=0.0, al_burn_off_kg=0.0,
            o_a_expected_ppm=o_a_initial_ppm, al_per_ton=0.0,
            cost_eur=0.0, currency=currency, model_id=model_id,
            inputs=inputs, warnings=warnings,
        )
    if not (0.0 < al_purity_pct <= 100.0):
        raise ValueError(f"al_purity_pct must be in (0, 100], got {al_purity_pct}")
    if not (0.0 <= burn_off_pct < 100.0):
        raise ValueError(f"burn_off_pct must be in [0, 100), got {burn_off_pct}")

    delta_o_kg = (o_a_initial_ppm - target_o_a_ppm) / 1e6 * steel_mass_ton * 1000.0
    al_active_kg = delta_o_kg * AL_TO_O_MASS_RATIO
    al_before_burn_off = al_active_kg / (1.0 - burn_off_pct / 100.0)
    al_burn_off_kg = al_before_burn_off - al_active_kg
    al_total_kg = al_before_burn_off / (al_purity_pct / 100.0)
    o_a_expected_ppm = target_o_a_ppm

    return DeoxidationResult(
        al_total_kg=al_total_kg,
        al_active_kg=al_active_kg,
        al_burn_off_kg=al_burn_off_kg,
        o_a_expected_ppm=o_a_expected_ppm,
        al_per_ton=al_total_kg / steel_mass_ton,
        cost_eur=al_total_kg * al_price_per_kg,
        currency=currency,
        model_id=model_id,
        inputs=inputs,
        warnings=warnings,
    )


def compute_al_quality(
    o_a_before_ppm: float,
    o_a_after_ppm: float,
    al_added_kg: float,
    temperature_C: float,
    steel_mass_ton: float,
    burn_off_pct: float = 20.0,
    model_id: str = DEFAULT_MODEL_ID,
) -> AlQualityResult:
    """Inverse: infer effective Al purity from observed deoxidation depth."""
    inputs = {
        "o_a_before_ppm": o_a_before_ppm,
        "o_a_after_ppm": o_a_after_ppm,
        "al_added_kg": al_added_kg,
        "temperature_C": temperature_C,
        "steel_mass_ton": steel_mass_ton,
        "burn_off_pct": burn_off_pct,
        "model_id": model_id,
    }
    warnings: list[str] = []

    if o_a_after_ppm >= o_a_before_ppm:
        raise ValueError(
            f"O_a after ({o_a_after_ppm}) >= before ({o_a_before_ppm}) — "
            f"no deoxidation observed."
        )
    if al_added_kg <= 0:
        raise ValueError("al_added_kg must be positive")
    if model_id not in THERMO_MODELS:
        raise ValueError(f"Unknown thermo model: {model_id}")

    delta_o_kg = (o_a_before_ppm - o_a_after_ppm) / 1e6 * steel_mass_ton * 1000.0
    effective_active_kg = delta_o_kg * AL_TO_O_MASS_RATIO
    expected_active_kg = al_added_kg * (1.0 - burn_off_pct / 100.0)
    effective_purity_pct = effective_active_kg / expected_active_kg * 100.0

    if effective_purity_pct < 70.0:
        warnings.append(
            f"Эффективная чистота Al = {effective_purity_pct:.1f}% — "
            f"подозрительно низкая (<70%). Проверьте чушку/лигатуру "
            f"или пересмотрите допущение burn_off={burn_off_pct}%."
        )

    return AlQualityResult(
        effective_purity_pct=effective_purity_pct,
        effective_active_kg=effective_active_kg,
        expected_active_kg=expected_active_kg,
        assumed_burn_off_pct=burn_off_pct,
        model_id=model_id,
        inputs=inputs,
        warnings=warnings,
    )


def compare_all_models(
    o_a_initial_ppm: float,
    temperature_C: float,
    steel_mass_ton: float,
    target_o_a_ppm: float,
    al_purity_pct: float = 100.0,
    burn_off_pct: float = 20.0,
    al_price_per_kg: float = 2.40,
    currency: str = "EUR",
) -> list[DeoxidationResult]:
    """Run compute_al_demand against all 3 thermo models — for compare-UI."""
    return [
        compute_al_demand(
            o_a_initial_ppm=o_a_initial_ppm,
            temperature_C=temperature_C,
            steel_mass_ton=steel_mass_ton,
            target_o_a_ppm=target_o_a_ppm,
            al_purity_pct=al_purity_pct,
            burn_off_pct=burn_off_pct,
            model_id=mid,
            al_price_per_kg=al_price_per_kg,
            currency=currency,
        )
        for mid in ("fruehan_1985", "sigworth_elliott_1974", "hayashi_2013")
    ]
