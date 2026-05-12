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
PR 2 scope: SlagState/CoDeoxSi dataclasses + compute_o_from_slag,
compute_o_consumed_by_si — стехиометрия O-баланса.
PR 3 scope: SlagAwareDemandResult + compute_al_demand_slag_aware —
главная функция, комбинирующая O-баланс + вызов deoxidation.compute_al_demand
+ residual [Al] target + cost-breakdown с premium из YAML.
PR 4 scope (текущий): compare_addition_methods + recommend_optimal_method
с фильтрами по constraints (target_n_ppm<50 → drop N2-carrier, premium_cap),
расширение cost_breakdown полями gas_eur / handling_eur.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from app.backend.deoxidation import (
    AL_TO_O_MASS_RATIO,
    DEFAULT_MODEL_ID,
    compute_al_demand,
)

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


# ---------------------------------------------------------------------------
# PR 3: main slag-aware demand function + result dataclass
# ---------------------------------------------------------------------------


# Default commodity Al-ingot baseline price (€/kg pure Al). Согласован с
# seed_2026-04-23.yaml (Al pure = €2.40/kg). Premium конкретного метода
# подачи добавляется поверх этой baseline (берётся из AdditionMethod).
DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG = 2.40

# Допустимое отклонение eta_al_override от литературного диапазона метода,
# при превышении которого добавляется warning (см. design-doc DX06).
_ETA_OVERRIDE_TOLERANCE = 0.05


@dataclass
class SlagAwareDemandResult:
    """Результат slag-aware расчёта потребности в Al.

    Поля разделены на физический баланс O, потребность Al (pure + charge),
    cost и метаданные. ``cost_breakdown`` содержит ``al_commodity_eur``,
    ``al_premium_eur``, и опциональные ``gas_eur`` + ``handling_eur``
    (заполняются только если у метода задан ``carrier_gas`` с
    ``price_per_nm3`` / ``gas_nm3_per_kg_al`` / ``handling_eur_per_kg_al``
    в YAML-каталоге; иначе остаются 0.0 / None).

    ``inputs`` — snapshot всех входных параметров для reproducibility
    (попадает в Decision Log при opt-in save). ``warnings`` — non-blocking
    advisories (например, η_Al вне литературного диапазона метода).
    """

    al_pure_kg: float                    # Потребность в чистом Al (Al-equivalent)
    al_charge_kg: float                  # Потребность в charge-форме (e.g. FeAl-30)
    al_burn_off_kg: float                # Угар (Al × (1 − η_Al))
    o_dissolved_kg: float                # O в расплаве (для удаления до target)
    o_in_slag_kg: float                  # O в FeO/MnO/SiO2 шлака переноса
    o_consumed_by_si_kg: float           # O связан Si pre-deox (0 если co_deox=None)
    o_total_to_remove_kg: float          # Net O для Al (clipped ≥ 0)
    method_id: str
    method_name: str
    eta_al_used: float                   # Фактически использованный η_Al
    al_specific_kg_per_ton: float        # кг Al-pure на 1 т стали
    charge_specific_kg_per_ton: float    # кг charge-формы на 1 т стали
    cost_eur: float                      # Total cost €/heat (Al + premium)
    cost_breakdown: dict                 # {al_commodity_eur, al_premium_eur}
    cost_per_ton_eur: float
    thermo_model_id: str
    inputs: dict
    warnings: list[str] = field(default_factory=list)


def _resolve_method(method: AdditionMethod | str) -> AdditionMethod:
    """Принимает AdditionMethod или строковый ID, возвращает объект.

    Если строка — резолвит через load_addition_methods().
    """
    if isinstance(method, AdditionMethod):
        return method
    if isinstance(method, str):
        methods = load_addition_methods()
        if method not in methods:
            available = ", ".join(sorted(methods.keys()))
            raise ValueError(
                f"Unknown addition method id: {method!r}. Available: {available}"
            )
        return methods[method]
    raise TypeError(
        f"method must be AdditionMethod or str, got {type(method).__name__}"
    )


def _al_content_pct_for_method(method: AdditionMethod) -> float:
    """% Al в charge-форме метода.

    Для FeAl-30 — 30%, для ASIS-дроби / гранулы / чушки / погружного слитка —
    99% (commodity Al). Хранится в ``method.raw['al_content_pct']`` если
    указан в YAML, иначе default 99%.
    """
    return float(method.raw.get("al_content_pct", 99.0))


def compute_al_demand_slag_aware(
    *,
    steel_mass_ton: float,
    o_a_initial_ppm: float,
    target_o_a_ppm: float,
    target_al_pct: float,
    method: AdditionMethod | str,
    slag: SlagState | None = None,
    co_deox_si: CoDeoxSi | None = None,
    eta_al_override: float | None = None,
    temperature_C: float = 1600.0,
    thermo_model_id: str = DEFAULT_MODEL_ID,
    al_commodity_price_eur_per_kg: float = DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG,
) -> SlagAwareDemandResult:
    """Slag-aware расчёт Al для раскисления BOF tap → ladle.

    Pipeline:

    1. Резолвим ``method`` (строка → AdditionMethod через YAML-каталог).
    2. Считаем O_dissolved (kg) от снижения [O]_a с initial до target:
       ``O_diss = (o_a_initial - target_o_a) × 1e-6 × steel × 1000``.
    3. Если ``slag`` задан → O_in_slag через ``compute_o_from_slag``.
    4. Если ``co_deox_si`` задан → O_consumed_by_si через
       ``compute_o_consumed_by_si``.
    5. Net O для Al = clip(O_diss + O_in_slag − O_si, ≥ 0).
    6. Эта netO переводится в **equivalent_o_a_ppm** — fictitious "initial"
       O activity, который при подаче в существующий ``compute_al_demand``
       с тем же target даст ровно нужную delta. Это позволяет переиспользовать
       basic функцию с её burn_off-математикой без копирования.
    7. ``η_Al`` берётся из ``eta_al_override`` (если задан) или
       ``method.eta_al_typical``. Преобразуется в burn_off_pct = (1−η)·100.
    8. ``compute_al_demand`` возвращает Al для O-баланса.
    9. К полученному добавляется **residual [Al]** в стали:
       ``al_residual_kg = steel × 1000 × target_al_pct/100`` —
       тоже proportionально делится на η_Al (residual Al должен дойти).
    10. ``al_pure_kg`` = total Al-equivalent (= O-binding + residual + burn_off).
    11. ``al_charge_kg`` = al_pure_kg / (al_content_pct/100) — для FeAl-30
        даёт whole-wire mass; для commodity-Al ≈ al_pure_kg / 0.99.
    12. cost = al_pure_kg × (commodity_price + method.premium_eur_per_kg);
        breakdown: ``{"al_commodity_eur": ..., "al_premium_eur": ...}``.
    13. warnings: если η_Al_override далеко от ``eta_al_range`` метода
        (более ±5%), добавляется advisory (отдельный паттерн DX06 в PR 5
        выдаёт это как BLOCK; здесь — soft warning).

    Args:
        steel_mass_ton: масса стали в ковше (т).
        o_a_initial_ppm: текущая [O]_a в расплаве (ppm) после BOF tap.
        target_o_a_ppm: целевая [O]_a после раскисления (ppm).
        target_al_pct: целевая остаточная [Al] в стали (% массовый).
            Для HSLA pipeline-марок типично 0.018-0.040%.
        method: ID метода (e.g. ``"asis_shot"``) или объект AdditionMethod.
        slag: состояние шлака переноса (опционально). Если None — слаг
            не вносит O (минимальная модель).
        co_deox_si: pre-deoxidation by FeSi/SiMn (опционально).
        eta_al_override: ручной override η_Al; если None — берётся
            method.eta_al_typical.
        temperature_C: температура расплава (°C). По умолчанию 1600°C.
        thermo_model_id: ID термодинамической модели для compute_al_demand
            (fruehan_1985 / sigworth_elliott_1974 / hayashi_2013).
        al_commodity_price_eur_per_kg: базовая цена commodity Al (€/kg).
            По умолчанию €2.40/kg — согласовано с seed_2026-04-23.yaml.

    Returns:
        SlagAwareDemandResult со всеми массами, cost-breakdown и warnings.

    Raises:
        ValueError: при невалидных входах (steel_mass<=0, target>=initial и т.д.)
            или unknown method id.
        TypeError: если method не AdditionMethod и не str.
    """
    method_obj = _resolve_method(method)

    if steel_mass_ton <= 0:
        raise ValueError(f"steel_mass_ton must be > 0, got {steel_mass_ton}")
    if target_al_pct < 0:
        raise ValueError(f"target_al_pct must be ≥ 0, got {target_al_pct}")

    inputs: dict = {
        "steel_mass_ton": steel_mass_ton,
        "o_a_initial_ppm": o_a_initial_ppm,
        "target_o_a_ppm": target_o_a_ppm,
        "target_al_pct": target_al_pct,
        "method_id": method_obj.id,
        "slag": {
            "mass_kg": slag.mass_kg,
            "feo_pct": slag.feo_pct,
            "mno_pct": slag.mno_pct,
            "sio2_pct": slag.sio2_pct,
        } if slag is not None else None,
        "co_deox_si": {
            "si_source_kg": co_deox_si.si_source_kg,
            "si_content_pct": co_deox_si.si_content_pct,
            "eta_si": co_deox_si.eta_si,
        } if co_deox_si is not None else None,
        "eta_al_override": eta_al_override,
        "temperature_C": temperature_C,
        "thermo_model_id": thermo_model_id,
        "al_commodity_price_eur_per_kg": al_commodity_price_eur_per_kg,
    }
    warnings: list[str] = []

    # η_Al
    eta_al_used = (
        float(eta_al_override)
        if eta_al_override is not None
        else float(method_obj.eta_al_typical)
    )
    if not (0.0 < eta_al_used < 1.0):
        raise ValueError(
            f"eta_al must be in (0, 1), got {eta_al_used} "
            f"(override={eta_al_override}, typical={method_obj.eta_al_typical})"
        )
    eta_lo, eta_hi = method_obj.eta_al_range
    if eta_al_override is not None and not (
        eta_lo - _ETA_OVERRIDE_TOLERANCE <= eta_al_used <= eta_hi + _ETA_OVERRIDE_TOLERANCE
    ):
        warnings.append(
            f"η_Al override = {eta_al_used:.2f} вне литературного диапазона "
            f"метода {method_obj.id} ({eta_lo:.2f}-{eta_hi:.2f}). Если основано "
            f"на исторических данных — рекомендуется plant-specific калибровка."
        )

    # 1. O-balance
    if target_o_a_ppm >= o_a_initial_ppm:
        # Нечего раскислять (по растворённому O), но residual [Al] всё равно
        # нужно подать. Compute_al_demand вернёт нули в этом случае; обработаем
        # ниже как чистый residual.
        o_dissolved_kg = 0.0
    else:
        o_dissolved_kg = (
            (o_a_initial_ppm - target_o_a_ppm) / 1e6 * steel_mass_ton * 1000.0
        )
    o_in_slag_kg = compute_o_from_slag(slag) if slag is not None else 0.0
    o_consumed_by_si_kg = (
        compute_o_consumed_by_si(co_deox_si) if co_deox_si is not None else 0.0
    )
    o_total_to_remove_kg = max(
        0.0, o_dissolved_kg + o_in_slag_kg - o_consumed_by_si_kg
    )

    # 2. Преобразуем netO в equivalent_o_a_ppm и вызываем basic compute_al_demand.
    # Логика: compute_al_demand берёт delta_O = (o_init - target) и умножает
    # на 1e-6 × steel × 1000. Чтобы он "увидел" наш netO напрямую, мы зафиксируем
    # target_o_a и подберём equivalent_o_a_initial так, что delta даст netO.
    equivalent_o_a_initial_ppm = (
        target_o_a_ppm + o_total_to_remove_kg / steel_mass_ton / 1000.0 * 1e6
    )

    # burn_off_pct = (1 - η_Al) × 100
    burn_off_pct = (1.0 - eta_al_used) * 100.0

    # 3. Вызов existing deoxidation.compute_al_demand для O-binding части.
    # al_purity_pct=100 — Al-equivalent semantics; charge-form накладывается
    # позже через al_content_pct.
    basic = compute_al_demand(
        o_a_initial_ppm=equivalent_o_a_initial_ppm,
        temperature_C=temperature_C,
        steel_mass_ton=steel_mass_ton,
        target_o_a_ppm=target_o_a_ppm,
        al_purity_pct=100.0,
        burn_off_pct=burn_off_pct,
        model_id=thermo_model_id,
        al_price_per_kg=al_commodity_price_eur_per_kg,
    )
    # Прокидываем warnings от basic compute_al_demand (например, T вне диапазона)
    warnings.extend(basic.warnings)

    al_for_o_pure_kg = basic.al_total_kg  # pure Al для O-binding (с burn_off)

    # 4. Residual [Al] в стали — добавляется поверх O-binding.
    # Al, который остаётся в расплаве как dissolved Al — тоже подвержен burn_off
    # (часть угорит до того как растворится). Делим на η_Al.
    al_residual_active_kg = steel_mass_ton * 1000.0 * target_al_pct / 100.0
    if eta_al_used > 0:
        al_residual_pure_kg = al_residual_active_kg / eta_al_used
    else:
        al_residual_pure_kg = 0.0

    al_pure_kg = al_for_o_pure_kg + al_residual_pure_kg
    al_active_total_kg = (
        al_for_o_pure_kg * eta_al_used + al_residual_active_kg
    )  # эквивалент: o-active часть + residual в расплаве
    al_burn_off_kg = al_pure_kg - al_active_total_kg

    # 5. Charge-форма (whole-wire / shot mass)
    al_content_pct = _al_content_pct_for_method(method_obj)
    al_charge_kg = al_pure_kg / (al_content_pct / 100.0)

    # 6. Cost breakdown
    al_commodity_eur = al_pure_kg * al_commodity_price_eur_per_kg
    al_premium_eur = al_pure_kg * method_obj.premium_eur_per_kg

    # Газовая составляющая (опционально): только если у метода задан carrier_gas
    # и в YAML присутствуют ``price_per_nm3`` + ``gas_nm3_per_kg_al``. Иначе 0.0.
    raw = method_obj.raw
    gas_eur: float | None = None
    if method_obj.carrier_gas is not None:
        price_per_nm3 = raw.get("price_per_nm3")
        gas_nm3_per_kg_al = raw.get("gas_nm3_per_kg_al")
        if price_per_nm3 is not None and gas_nm3_per_kg_al is not None:
            gas_eur = float(price_per_nm3) * float(gas_nm3_per_kg_al) * al_pure_kg
        else:
            gas_eur = 0.0

    # Handling cost (опционально): wire-feeder обслуживание, injector wear, etc.
    handling_eur_per_kg_al = raw.get("handling_eur_per_kg_al")
    handling_eur = (
        float(handling_eur_per_kg_al) * al_pure_kg
        if handling_eur_per_kg_al is not None
        else 0.0
    )

    cost_eur = al_commodity_eur + al_premium_eur + (gas_eur or 0.0) + handling_eur
    cost_breakdown = {
        "al_commodity_eur": al_commodity_eur,
        "al_premium_eur": al_premium_eur,
        "gas_eur": gas_eur,
        "handling_eur": handling_eur,
    }

    return SlagAwareDemandResult(
        al_pure_kg=al_pure_kg,
        al_charge_kg=al_charge_kg,
        al_burn_off_kg=al_burn_off_kg,
        o_dissolved_kg=o_dissolved_kg,
        o_in_slag_kg=o_in_slag_kg,
        o_consumed_by_si_kg=o_consumed_by_si_kg,
        o_total_to_remove_kg=o_total_to_remove_kg,
        method_id=method_obj.id,
        method_name=method_obj.name,
        eta_al_used=eta_al_used,
        al_specific_kg_per_ton=al_pure_kg / steel_mass_ton,
        charge_specific_kg_per_ton=al_charge_kg / steel_mass_ton,
        cost_eur=cost_eur,
        cost_breakdown=cost_breakdown,
        cost_per_ton_eur=cost_eur / steel_mass_ton,
        thermo_model_id=thermo_model_id,
        inputs=inputs,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# PR 4: compare_addition_methods + recommend_optimal_method
# ---------------------------------------------------------------------------


# Порог [N] (ppm) ниже которого N2-carrier-методы автоматически отфильтровываются
# (N2 → pickup 5-15 ppm; для марок с target [N]<50 ppm требуется Ar). См. DX07.
_N_PICKUP_HARD_LIMIT_PPM = 50.0


@dataclass(frozen=True)
class MethodCompareRow:
    """Одна строка таблицы сравнения методов подачи Al для одной плавки.

    Используется в ``compare_addition_methods`` и как ``pareto_table`` в
    ``OptimizationRecommendation``. ``scatter_kg`` — диапазон Al-pure
    между ``eta_al_range[0]`` (worst-case η, больше Al) и
    ``eta_al_range[1]`` (best-case η, меньше Al); грубая оценка
    plant-to-plant scatter без plant-specific калибровки.

    Поля упорядочены так, чтобы DataFrame.from_records давал human-friendly
    columns; ``warnings`` агрегирует non-blocking advisories per-method.
    """

    method_id: str
    method_name: str
    eta_al_used: float
    al_pure_kg: float
    al_charge_kg: float
    cost_per_heat_eur: float
    cost_per_ton_eur: float
    al_specific_kg_per_t: float
    carrier_gas: str | None
    scatter_kg: float
    warnings: list[str]


@dataclass(frozen=True)
class OptimizationRecommendation:
    """Результат ``recommend_optimal_method`` — лучший метод + контекст.

    ``rationale`` — 2-3 предложения объяснения выбора (для UI + Decision Log).
    ``constraints_active`` перечисляет применённые фильтры с описанием эффекта
    (например ``"target_n_ppm=30<50 → исключены методы с carrier_gas=N2"``).
    ``rejected_methods`` — методы, отсеянные constraints, с reason; pareto_table
    — все методы, выжившие после фильтров, в порядке возрастания cost.
    ``inputs`` — snapshot входных параметров (reproducibility).
    """

    chosen_method_id: str
    chosen_method_name: str
    chosen_cost_eur: float
    rationale: str
    runner_up_method_id: str | None
    runner_up_cost_eur: float | None
    runner_up_delta_eur: float | None
    constraints_active: list[str]
    rejected_methods: list[dict]
    pareto_table: list[MethodCompareRow]
    inputs: dict


def _compute_scatter_kg(
    *,
    method_obj: AdditionMethod,
    base_inputs: dict,
    slag: SlagState | None,
    co_deox_si: CoDeoxSi | None,
    temperature_C: float,
    thermo_model_id: str,
    al_commodity_price_eur_per_kg: float,
) -> float:
    """± диапазон Al-pure (kg) от литературного eta_al_range метода.

    Считает Al-pure при eta_lo и eta_hi, возвращает abs(delta). Это
    grub-оценка scatter для plant-to-plant variability без калибровки.
    """
    eta_lo, eta_hi = method_obj.eta_al_range
    al_lo = compute_al_demand_slag_aware(
        method=method_obj,
        slag=slag,
        co_deox_si=co_deox_si,
        eta_al_override=eta_lo,
        temperature_C=temperature_C,
        thermo_model_id=thermo_model_id,
        al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
        **base_inputs,
    ).al_pure_kg
    al_hi = compute_al_demand_slag_aware(
        method=method_obj,
        slag=slag,
        co_deox_si=co_deox_si,
        eta_al_override=eta_hi,
        temperature_C=temperature_C,
        thermo_model_id=thermo_model_id,
        al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
        **base_inputs,
    ).al_pure_kg
    return abs(al_lo - al_hi)


def compare_addition_methods(
    *,
    steel_mass_ton: float,
    o_a_initial_ppm: float,
    target_o_a_ppm: float,
    target_al_pct: float,
    slag: SlagState | None = None,
    co_deox_si: CoDeoxSi | None = None,
    temperature_C: float = 1600.0,
    thermo_model_id: str = DEFAULT_MODEL_ID,
    al_commodity_price_eur_per_kg: float = DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG,
    method_ids: list[str] | None = None,
) -> list[MethodCompareRow]:
    """Сравнить все методы подачи Al для одной плавки.

    Запускает ``compute_al_demand_slag_aware`` для каждого метода в каталоге
    (или для подмножества ``method_ids``) с одинаковыми параметрами плавки,
    собирает results в list[MethodCompareRow] **отсортированный по
    cost_per_heat_eur ascending** (cheapest first).

    Args:
        steel_mass_ton, o_a_initial_ppm, target_o_a_ppm, target_al_pct,
        slag, co_deox_si, temperature_C, thermo_model_id,
        al_commodity_price_eur_per_kg: те же параметры, что в
            ``compute_al_demand_slag_aware``.
        method_ids: список ID методов для сравнения. ``None`` (default) =
            все методы из YAML.

    Returns:
        list[MethodCompareRow] sorted ascending by ``cost_per_heat_eur``.

    Raises:
        ValueError: если ``method_ids`` содержит unknown id.
    """
    methods = load_addition_methods()
    if method_ids is None:
        ids = list(methods.keys())
    else:
        unknown = [m for m in method_ids if m not in methods]
        if unknown:
            available = ", ".join(sorted(methods.keys()))
            raise ValueError(
                f"Unknown addition method ids: {unknown}. Available: {available}"
            )
        ids = list(method_ids)

    base_inputs = dict(
        steel_mass_ton=steel_mass_ton,
        o_a_initial_ppm=o_a_initial_ppm,
        target_o_a_ppm=target_o_a_ppm,
        target_al_pct=target_al_pct,
    )

    rows: list[MethodCompareRow] = []
    for mid in ids:
        method_obj = methods[mid]
        res = compute_al_demand_slag_aware(
            method=method_obj,
            slag=slag,
            co_deox_si=co_deox_si,
            temperature_C=temperature_C,
            thermo_model_id=thermo_model_id,
            al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
            **base_inputs,
        )
        scatter_kg = _compute_scatter_kg(
            method_obj=method_obj,
            base_inputs=base_inputs,
            slag=slag,
            co_deox_si=co_deox_si,
            temperature_C=temperature_C,
            thermo_model_id=thermo_model_id,
            al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
        )
        rows.append(
            MethodCompareRow(
                method_id=res.method_id,
                method_name=res.method_name,
                eta_al_used=res.eta_al_used,
                al_pure_kg=res.al_pure_kg,
                al_charge_kg=res.al_charge_kg,
                cost_per_heat_eur=res.cost_eur,
                cost_per_ton_eur=res.cost_per_ton_eur,
                al_specific_kg_per_t=res.al_specific_kg_per_ton,
                carrier_gas=method_obj.carrier_gas,
                scatter_kg=scatter_kg,
                warnings=list(res.warnings),
            )
        )

    rows.sort(key=lambda r: r.cost_per_heat_eur)
    return rows


def recommend_optimal_method(
    *,
    steel_mass_ton: float,
    o_a_initial_ppm: float,
    target_o_a_ppm: float,
    target_al_pct: float,
    slag: SlagState | None = None,
    co_deox_si: CoDeoxSi | None = None,
    temperature_C: float = 1600.0,
    thermo_model_id: str = DEFAULT_MODEL_ID,
    al_commodity_price_eur_per_kg: float = DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG,
    method_ids: list[str] | None = None,
    target_n_ppm: float | None = None,
    premium_cap_eur_per_kg: float | None = None,
) -> OptimizationRecommendation:
    """Найти оптимальный метод подачи Al с учётом constraints.

    Pipeline:

    1. Вызвать ``compare_addition_methods`` → full pareto_table sorted by cost.
    2. Применить фильтры:
       - ``target_n_ppm < 50`` → отсеять методы с ``carrier_gas == "N2"``
         (N₂ pickup 5-15 ppm недопустим для low-N марок; см. DX07).
       - ``premium_cap_eur_per_kg`` → отсеять методы с
         ``method.premium_eur_per_kg > cap``.
    3. Из выживших: chosen = min by ``cost_per_heat_eur``.
    4. Runner-up = следующий по cost (если есть).
    5. Rationale — 2-3 предложения для UI и Decision Log.

    Args:
        ... (см. compare_addition_methods для базовых аргументов).
        target_n_ppm: целевая [N] в стали (ppm). Если < 50 → фильтрует
            N₂-carrier-методы. ``None`` = no [N] constraint.
        premium_cap_eur_per_kg: максимально допустимый premium €/kg Al-eq.
            ``None`` = no cap.

    Returns:
        OptimizationRecommendation с chosen, runner_up, constraints_active,
        rejected_methods и pareto_table выживших методов.

    Raises:
        ValueError: если после фильтров не осталось ни одного метода.
    """
    pareto = compare_addition_methods(
        steel_mass_ton=steel_mass_ton,
        o_a_initial_ppm=o_a_initial_ppm,
        target_o_a_ppm=target_o_a_ppm,
        target_al_pct=target_al_pct,
        slag=slag,
        co_deox_si=co_deox_si,
        temperature_C=temperature_C,
        thermo_model_id=thermo_model_id,
        al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
        method_ids=method_ids,
    )

    methods = load_addition_methods()
    constraints_active: list[str] = []
    rejected: list[dict] = []
    survivors: list[MethodCompareRow] = []

    n_constraint_active = (
        target_n_ppm is not None and target_n_ppm < _N_PICKUP_HARD_LIMIT_PPM
    )
    if n_constraint_active:
        constraints_active.append(
            f"target_n_ppm={target_n_ppm:.0f}<{_N_PICKUP_HARD_LIMIT_PPM:.0f} → "
            f"исключены методы с carrier_gas=N2"
        )
    if premium_cap_eur_per_kg is not None:
        constraints_active.append(
            f"premium_cap={premium_cap_eur_per_kg:.2f}€/kg → исключены "
            f"методы с premium выше cap"
        )

    for row in pareto:
        method_obj = methods[row.method_id]
        # Фильтр 1: N2-carrier при low-N target
        if n_constraint_active and method_obj.carrier_gas == "N2":
            rejected.append({
                "method_id": row.method_id,
                "reason": f"carrier_gas=N2 несовместим с target_n_ppm={target_n_ppm}",
            })
            continue
        # Фильтр 2: premium cap
        if (
            premium_cap_eur_per_kg is not None
            and method_obj.premium_eur_per_kg > premium_cap_eur_per_kg
        ):
            rejected.append({
                "method_id": row.method_id,
                "reason": (
                    f"premium={method_obj.premium_eur_per_kg:.2f}€/kg > "
                    f"cap={premium_cap_eur_per_kg:.2f}€/kg"
                ),
            })
            continue
        survivors.append(row)

    if not survivors:
        raise ValueError(
            "После применения constraints не осталось ни одного метода. "
            f"Active constraints: {constraints_active}. "
            f"Rejected: {[r['method_id'] for r in rejected]}"
        )

    chosen = survivors[0]
    runner_up = survivors[1] if len(survivors) >= 2 else None

    if runner_up is not None:
        delta_eur = runner_up.cost_per_heat_eur - chosen.cost_per_heat_eur
        rationale = (
            f"Выбран '{chosen.method_name}' — наименьший cost "
            f"{chosen.cost_per_heat_eur:.0f}€ при η_Al={chosen.eta_al_used:.2f}. "
            f"Runner-up '{runner_up.method_name}': +{delta_eur:.0f}€ "
            f"({runner_up.cost_per_heat_eur:.0f}€)."
        )
        runner_up_id = runner_up.method_id
        runner_up_cost = runner_up.cost_per_heat_eur
    else:
        delta_eur = None
        rationale = (
            f"Выбран '{chosen.method_name}' — единственный метод, удовлетворяющий "
            f"constraints. Cost {chosen.cost_per_heat_eur:.0f}€ при "
            f"η_Al={chosen.eta_al_used:.2f}."
        )
        runner_up_id = None
        runner_up_cost = None

    inputs = dict(
        steel_mass_ton=steel_mass_ton,
        o_a_initial_ppm=o_a_initial_ppm,
        target_o_a_ppm=target_o_a_ppm,
        target_al_pct=target_al_pct,
        slag={
            "mass_kg": slag.mass_kg,
            "feo_pct": slag.feo_pct,
            "mno_pct": slag.mno_pct,
            "sio2_pct": slag.sio2_pct,
        } if slag is not None else None,
        co_deox_si={
            "si_source_kg": co_deox_si.si_source_kg,
            "si_content_pct": co_deox_si.si_content_pct,
            "eta_si": co_deox_si.eta_si,
        } if co_deox_si is not None else None,
        temperature_C=temperature_C,
        thermo_model_id=thermo_model_id,
        al_commodity_price_eur_per_kg=al_commodity_price_eur_per_kg,
        method_ids=method_ids,
        target_n_ppm=target_n_ppm,
        premium_cap_eur_per_kg=premium_cap_eur_per_kg,
    )

    return OptimizationRecommendation(
        chosen_method_id=chosen.method_id,
        chosen_method_name=chosen.method_name,
        chosen_cost_eur=chosen.cost_per_heat_eur,
        rationale=rationale,
        runner_up_method_id=runner_up_id,
        runner_up_cost_eur=runner_up_cost,
        runner_up_delta_eur=delta_eur,
        constraints_active=constraints_active,
        rejected_methods=rejected,
        pareto_table=survivors,
        inputs=inputs,
    )


# Suppress unused-import lint for AL_TO_O_MASS_RATIO — exposed for tests
# и для downstream-модулей (Pattern Library DX*-checks в PR 5).
__all__ = [
    "AdditionMethod",
    "SlagState",
    "CoDeoxSi",
    "SlagAwareDemandResult",
    "MethodCompareRow",
    "OptimizationRecommendation",
    "AL_TO_O_MASS_RATIO",
    "DEFAULT_AL_COMMODITY_PRICE_EUR_PER_KG",
    "compute_o_from_slag",
    "compute_o_consumed_by_si",
    "compute_al_demand_slag_aware",
    "compare_addition_methods",
    "recommend_optimal_method",
    "load_addition_methods",
    "list_method_ids",
]


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

    print()
    print("=" * 70)
    print("Slag-aware Al demand — Excel base-case (PR 3):\n")
    print("  Plant case: 371 t, [O]_a=657 ppm → target 5 ppm,")
    print("  target [Al]=0.018%, slag carry-over 2.2 t @ FeO=18%, T=1600°C\n")

    base_inputs = dict(
        steel_mass_ton=371.0,
        o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0,
        target_al_pct=0.018,
        temperature_C=1600.0,
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
    )

    for mid in ("ingot", "asis_shot", "cored_wire_feal30"):
        res = compute_al_demand_slag_aware(method=mid, **base_inputs)
        print(
            f"  {mid:>20s} | η={res.eta_al_used:.2f} | "
            f"Al_pure={res.al_pure_kg:>6.1f} kg | "
            f"charge={res.al_charge_kg:>6.1f} kg | "
            f"€{res.cost_eur:>7.1f} (€{res.cost_per_ton_eur:>5.2f}/t)"
        )

    print()
    print("  Excel reference (k=0.89, η≈0.80): ~454 kg Al pure on ASIS-shot.")
    print("  Our model uses literature η_Al per method from YAML catalog;")
    print("  values agree within ±17% of plant calibration constant")
    print("  (residual [Al] target учитывается у нас, но игнорируется Excel).")

    print()
    print("=" * 70)
    print("recommend_optimal_method — Excel base + target_n_ppm=30 (PR 4):\n")
    print("  Constraint: target_n_ppm=30 < 50 → N2-carrier методы отсеиваются.")
    print("  Constraint: premium_cap=1.0 €/kg → cored_wire_feal30 (€2.50) отсеян.\n")

    rec = recommend_optimal_method(
        target_n_ppm=30.0,
        premium_cap_eur_per_kg=1.0,
        **base_inputs,
    )
    print(f"  Chosen: {rec.chosen_method_id} ('{rec.chosen_method_name}')")
    print(f"  Cost:   €{rec.chosen_cost_eur:.0f}/heat")
    print(f"  Rationale: {rec.rationale}")
    print()
    print("  Constraints active:")
    for c in rec.constraints_active:
        print(f"    - {c}")
    print()
    print("  Rejected methods:")
    for rej in rec.rejected_methods:
        print(f"    - {rej['method_id']}: {rej['reason']}")
    print()
    print("  Pareto table (sorted by cost ascending):")
    for r in rec.pareto_table:
        gas = r.carrier_gas or "—"
        print(
            f"    {r.method_id:>22s} | η={r.eta_al_used:.2f} | "
            f"Al_pure={r.al_pure_kg:>6.1f} kg | "
            f"€{r.cost_per_heat_eur:>7.1f}/heat | gas={gas}"
        )
