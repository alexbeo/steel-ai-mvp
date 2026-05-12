"""
Slag-aware advisory для Al-раскисления BOF tap → ladle.

Расширяет термодинамическую модель из app/backend/deoxidation.py учётом:
- O связанного в FeO/MnO/SiO₂ шлака переноса (slag-aware O-balance)
- co-deoxidation by Si (FeSi/SiMn до Al)
- выбора метода подачи Al (catalog в data/deox_methods/al_addition_methods.yaml)

Этот модуль — pure physics + cost. Без LLM, без ML.
Phase: deoxidation (LF-tap, не LF-equilibrium — отличается от deoxidation.py).

Design: docs/superpowers/specs/2026-05-12_asis-slag-aware-deox.md
PR 1 scope: YAML catalog + dataclass AdditionMethod + load_addition_methods().
PR 2 scope (текущий): SlagState/CoDeoxSi dataclasses + compute_o_from_slag,
compute_o_consumed_by_si — стехиометрия O-баланса.
Compute-функции (compute_al_demand_slag_aware, compare_addition_methods,
recommend_optimal_method) — в PR 3-4.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEOX_METHODS_PATH = (
    Path(__file__).parent.parent.parent / "data" / "deox_methods" / "al_addition_methods.yaml"
)


@dataclass(frozen=True)
class AdditionMethod:
    """Один метод подачи Al для раскисления (catalog entry).

    Поля совпадают с YAML-схемой каталога. ``raw`` хранит исходную строку
    из YAML для forward-compat (доступ к опциональным полям вроде
    ``t_drying_max_c``, ``al_content_pct``, ``size_mm`` без жёсткого
    contract в dataclass).
    """

    id: str
    name: str
    eta_al_typical: float
    eta_al_range: tuple[float, float]      # (min, max) literature range
    premium_eur_per_kg: float              # over commodity Al ingot (€/kg Al-equivalent)
    carrier_gas: str | None                # "Ar" | "N2" | None
    surface_m2_per_kg: float
    notes: str
    raw: dict                              # full YAML row for forward-compat


@dataclass(frozen=True)
class SlagState:
    """Состояние шлака переноса BOF→ковш для расчёта O-баланса.

    Хранит общую массу шлака на плавку (kg total, **не** kg/t стали) и
    массовый процент трёх ключевых оксидов. ``mno_pct`` и ``sio2_pct``
    опциональны — в большинстве BOF-плавок шлак переноса в основном FeO,
    остальные оксиды малы. По умолчанию нулевые.
    """

    mass_kg: float       # kg slag total (per heat, not per ton)
    feo_pct: float       # % FeO в шлаке
    mno_pct: float = 0.0
    sio2_pct: float = 0.0


@dataclass(frozen=True)
class CoDeoxSi:
    """Pre-deoxidation by Si (FeSi/SiMn введён ДО Al).

    Опциональный блок: если на плавке ввели FeSi (или SiMn), часть
    кислорода уже связалась в SiO2, и Al-потребность уменьшается.
    ``si_content_pct`` — массовая доля Si в источнике (75% для FeSi-75,
    65% для FeSi-65, ~17% для SiMn). ``eta_si`` — доля Si, ушедшая в
    SiO2 (литература: 0.92-0.97; default 0.95).
    """

    si_source_kg: float           # масса введённого FeSi или SiMn (kg)
    si_content_pct: float = 75.0  # %Si в источнике (75 для FeSi-75; 17 для SiMn)
    eta_si: float = 0.95          # доля Si, выгоревшая в SiO2 (95% по литературе)


# Стехиометрические массовые коэффициенты O в оксидах.
# Молярные массы: Fe=56, Mn=55, Si=28, O=16.
_O_PER_FEO_MASS_FRACTION = 16.0 / 72.0     # = 0.2222 (M(FeO) = 56+16 = 72)
_O_PER_MNO_MASS_FRACTION = 16.0 / 71.0     # = 0.2254 (M(MnO) = 55+16 = 71)
_O_PER_SIO2_MASS_FRACTION = 32.0 / 60.0    # = 0.5333 (2*O в SiO2; M(SiO2) = 28+32 = 60)
_O_PER_SI_IN_SIO2 = 32.0 / 28.0            # = 1.1429 (O₂/Si по массе для SiO2)


def compute_o_from_slag(slag: SlagState) -> float:
    """Масса O (kg), связанного в шлаке переноса как FeO/MnO/SiO2.

    Использует точные стехиометрические массовые отношения. %SiO2 в шлаке
    обычно НЕ считается источником "связанного O" для целей раскисления
    (Si уже окислен, его O недоступен для удаления Al), но включён в
    подсчёт по design-doc для completeness; на практике значения %SiO2
    в этом расчёте обычно ставят 0, если шлак уже работал как pre-deox.

    Args:
        slag: SlagState с массой шлака и % оксидов.

    Returns:
        kg O total bound in slag carry-over.

    Raises:
        ValueError: если масса или %оксидов отрицательные.
    """
    if slag.mass_kg < 0 or slag.feo_pct < 0 or slag.mno_pct < 0 or slag.sio2_pct < 0:
        raise ValueError("SlagState: масса и %оксидов должны быть ≥ 0")
    return slag.mass_kg * (
        _O_PER_FEO_MASS_FRACTION * slag.feo_pct / 100.0
        + _O_PER_MNO_MASS_FRACTION * slag.mno_pct / 100.0
        + _O_PER_SIO2_MASS_FRACTION * slag.sio2_pct / 100.0
    )


def compute_o_consumed_by_si(co_deox: CoDeoxSi) -> float:
    """Масса O (kg), которую связал Si из pre-deox FeSi/SiMn в SiO2.

    Формула::

        Si_kg_total    = source_kg × si_content_pct / 100
        Si_oxidized_kg = Si_kg_total × eta_si
        O_kg_bound     = Si_oxidized_kg × (32 / 28)

    Коэффициент 32/28 = O₂/Si по массе для реакции ``Si + O2 -> SiO2``.

    Args:
        co_deox: CoDeoxSi с массой FeSi/SiMn, %Si и η_Si.

    Returns:
        kg O consumed by Si pre-deoxidation.

    Raises:
        ValueError: при невалидных параметрах
            (si_source_kg<0, si_content_pct вне (0, 100], eta_si вне [0, 1]).
    """
    if co_deox.si_source_kg < 0 or not (0.0 < co_deox.si_content_pct <= 100.0):
        raise ValueError("CoDeoxSi: si_source_kg ≥ 0 и 0 < si_content_pct ≤ 100")
    if not (0.0 <= co_deox.eta_si <= 1.0):
        raise ValueError("CoDeoxSi: eta_si должна быть в [0, 1]")
    si_kg = co_deox.si_source_kg * co_deox.si_content_pct / 100.0
    si_oxidized_kg = si_kg * co_deox.eta_si
    return si_oxidized_kg * _O_PER_SI_IN_SIO2


@lru_cache(maxsize=1)
def _load_addition_methods_cached(path_str: str) -> dict[str, AdditionMethod]:
    """Внутренняя реализация — закешированная по строковому пути.

    lru_cache требует hashable аргументов; Path — hashable, но кеш переживает
    разные Path-объекты к одному файлу. Используем строку для нормализации.
    """
    yaml_path = Path(path_str)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Al addition methods catalog not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    methods_raw = data.get("methods", {})
    if not methods_raw:
        raise ValueError(f"YAML at {yaml_path} has no 'methods' key or it is empty")

    result: dict[str, AdditionMethod] = {}
    required = (
        "name",
        "eta_al_typical",
        "eta_al_range",
        "premium_eur_per_kg",
        "surface_m2_per_kg",
    )
    for method_id, row in methods_raw.items():
        if not isinstance(row, dict):
            raise ValueError(
                f"Method '{method_id}' is not a mapping (got {type(row).__name__})"
            )

        missing = [f for f in required if f not in row]
        if missing:
            raise ValueError(
                f"Method '{method_id}' missing required fields: {missing}"
            )

        eta_range_raw = row["eta_al_range"]
        if not isinstance(eta_range_raw, (list, tuple)) or len(eta_range_raw) != 2:
            raise ValueError(
                f"Method '{method_id}': eta_al_range must be [min, max], "
                f"got {eta_range_raw!r}"
            )
        eta_lo, eta_hi = float(eta_range_raw[0]), float(eta_range_raw[1])
        if not (0.0 < eta_lo <= eta_hi <= 1.0):
            raise ValueError(
                f"Method '{method_id}': eta_al_range must satisfy "
                f"0 < min <= max <= 1, got [{eta_lo}, {eta_hi}]"
            )

        result[method_id] = AdditionMethod(
            id=method_id,
            name=row["name"],
            eta_al_typical=float(row["eta_al_typical"]),
            eta_al_range=(eta_lo, eta_hi),
            premium_eur_per_kg=float(row["premium_eur_per_kg"]),
            carrier_gas=row.get("carrier_gas"),
            surface_m2_per_kg=float(row["surface_m2_per_kg"]),
            notes=row.get("notes", ""),
            raw=dict(row),
        )

    return result


def load_addition_methods(path: Path | None = None) -> dict[str, AdditionMethod]:
    """Загружает каталог методов из YAML.

    Args:
        path: путь к YAML-каталогу. По умолчанию —
            ``data/deox_methods/al_addition_methods.yaml`` в корне проекта.

    Returns:
        dict ``{method_id: AdditionMethod}`` — порядок соответствует YAML.

    Raises:
        FileNotFoundError: если YAML отсутствует по указанному пути.
        ValueError: если YAML невалиден (нет ``methods``, нет required-полей,
            или ``eta_al_range`` нелогичен).
    """
    yaml_path = path or DEOX_METHODS_PATH
    return _load_addition_methods_cached(str(yaml_path.resolve()))


# Алиас для удобства управления кешем извне (тесты, hot-reload UI).
load_addition_methods.cache_clear = _load_addition_methods_cached.cache_clear  # type: ignore[attr-defined]


def list_method_ids(path: Path | None = None) -> list[str]:
    """Удобный helper — возвращает ID всех методов из каталога."""
    return list(load_addition_methods(path).keys())


if __name__ == "__main__":
    # Dry-run demo: показать каталог + примеры O-баланса
    methods = load_addition_methods()
    print(f"Loaded {len(methods)} Al addition methods from catalog:\n")
    for mid, m in methods.items():
        print(f"  {mid}: {m.name}")
        print(
            f"    η_Al = {m.eta_al_typical} "
            f"(range {m.eta_al_range[0]:.2f}-{m.eta_al_range[1]:.2f})"
        )
        print(
            f"    Surface: {m.surface_m2_per_kg} m²/kg | "
            f"Premium: €{m.premium_eur_per_kg:.2f}/kg Al-eq"
        )
        print(f"    Carrier gas: {m.carrier_gas or 'не требуется'}")
        print()

    print("=" * 70)
    print("O-balance examples (PR 2):\n")

    # Example 1: Excel base-case slag (2.2 t шлака переноса, FeO=18%, MnO=SiO2=0)
    base_slag = SlagState(mass_kg=2200.0, feo_pct=18.0)
    o_slag = compute_o_from_slag(base_slag)
    print(
        f"  [Excel base] slag M={base_slag.mass_kg:.0f} kg, "
        f"FeO={base_slag.feo_pct:.1f}% → O bound = {o_slag:.2f} kg "
        f"(≈ 88 kg expected)"
    )

    # Example 2: FeSi-75 pre-deoxidation (100 kg FeSi-75, η=0.95)
    base_codeox = CoDeoxSi(si_source_kg=100.0, si_content_pct=75.0, eta_si=0.95)
    o_si = compute_o_consumed_by_si(base_codeox)
    print(
        f"  [FeSi-75 pre-deox] {base_codeox.si_source_kg:.0f} kg FeSi-75, "
        f"η_Si={base_codeox.eta_si} → O consumed = {o_si:.2f} kg "
        f"(≈ 81.4 kg expected)"
    )
