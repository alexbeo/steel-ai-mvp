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
