"""Tests for slag_aware_deox module.

PR 1 — YAML loader + AdditionMethod dataclass.
PR 2 — SlagState/CoDeoxSi dataclasses + O-balance functions.
PR 3 — SlagAwareDemandResult + compute_al_demand_slag_aware + cost_model
       расширение под kind="deox_consumable".
PR 4 — compare_methods, recommend_optimal.
"""
from __future__ import annotations

import pytest
import yaml

from app.backend.deoxidation import AL_TO_O_MASS_RATIO, compute_al_demand
from app.backend.slag_aware_deox import (
    AdditionMethod,
    CoDeoxSi,
    MethodCompareRow,
    OptimizationRecommendation,
    SlagAwareDemandResult,
    SlagState,
    compare_addition_methods,
    compute_al_demand_slag_aware,
    compute_o_consumed_by_si,
    compute_o_from_slag,
    list_method_ids,
    load_addition_methods,
    recommend_optimal_method,
)


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    """Каждый тест начинается с чистого lru_cache."""
    load_addition_methods.cache_clear()
    yield
    load_addition_methods.cache_clear()


def test_load_addition_methods_from_default_path():
    """Каталог из data/deox_methods/ загружается без ошибок."""
    methods = load_addition_methods()
    assert isinstance(methods, dict)
    assert len(methods) >= 5, f"Expected >=5 methods, got {len(methods)}"
    # Обязательные методы из design-doc
    for expected in ["ingot", "asis_shot", "granule_water_quenched", "cored_wire_feal30"]:
        assert expected in methods, f"Method '{expected}' missing from catalog"
    # Каждое значение — AdditionMethod
    for mid, m in methods.items():
        assert isinstance(m, AdditionMethod), f"{mid} is {type(m).__name__}"
        assert m.id == mid


def test_addition_method_dataclass_immutable():
    """AdditionMethod — frozen, нельзя случайно мутировать."""
    methods = load_addition_methods()
    m = next(iter(methods.values()))
    with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
        m.eta_al_typical = 0.99  # type: ignore[misc]


def test_method_eta_ranges_sane():
    """Все η_Al диапазоны — [0,1], typical внутри range."""
    methods = load_addition_methods()
    for mid, m in methods.items():
        lo, hi = m.eta_al_range
        assert 0.0 < lo <= hi <= 1.0, f"{mid}: eta_al_range invalid: {m.eta_al_range}"
        assert lo - 0.05 <= m.eta_al_typical <= hi + 0.05, (
            f"{mid}: eta_al_typical={m.eta_al_typical} far outside range {m.eta_al_range}"
        )


def test_yaml_loader_rejects_missing_required_fields(tmp_path):
    """Loader падает с ValueError если в method не хватает обязательного поля."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            {
                "version": "test",
                "methods": {
                    "broken_method": {
                        "name": "Broken",
                        # Отсутствуют eta_al_typical, eta_al_range,
                        # premium_eur_per_kg, surface_m2_per_kg
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_addition_methods(bad_yaml)


def test_yaml_loader_rejects_invalid_eta_range(tmp_path):
    """Loader падает если eta_al_range нелогичен (max < min)."""
    bad_yaml = tmp_path / "bad_range.yaml"
    bad_yaml.write_text(
        yaml.safe_dump(
            {
                "methods": {
                    "bad_eta": {
                        "name": "Bad",
                        "eta_al_typical": 0.8,
                        "eta_al_range": [0.9, 0.5],  # max < min
                        "premium_eur_per_kg": 0.0,
                        "surface_m2_per_kg": 0.1,
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="eta_al_range"):
        load_addition_methods(bad_yaml)


def test_list_method_ids_returns_all_methods():
    """list_method_ids() возвращает все ID из каталога."""
    ids = list_method_ids()
    assert "asis_shot" in ids
    assert "ingot" in ids
    assert len(ids) == len(load_addition_methods())


# ---------------------------------------------------------------------------
# PR 2: O-balance (compute_o_from_slag + compute_o_consumed_by_si)
# ---------------------------------------------------------------------------


def test_compute_o_from_slag_basic_feo_only():
    """Excel base-case: 2.2 т шлака, FeO=18%, MnO=SiO2=0 → ~88 кг O.

    Точное: 2200 × (16/72) × 0.18 = 88.0 kg.
    Design-doc округляет коэф. до 0.222 и даёт 87.91; мы используем точную
    дробь и допускаем ±0.1 kg окрестность.
    """
    slag = SlagState(mass_kg=2200.0, feo_pct=18.0)
    o_kg = compute_o_from_slag(slag)
    assert o_kg == pytest.approx(88.0, abs=0.1), f"got {o_kg:.3f} kg"


def test_compute_o_from_slag_with_mno_sio2():
    """Все три оксида: проверка слагаемых по точной стехиометрии.

    M_slag=1000 kg, FeO=10%, MnO=5%, SiO2=2%.
    Expected:
        FeO part  = 1000 × (16/72) × 0.10 = 22.222...
        MnO part  = 1000 × (16/71) × 0.05 = 11.2676...
        SiO2 part = 1000 × (32/60) × 0.02 = 10.6667...
        Total ≈ 44.156 kg
    """
    slag = SlagState(mass_kg=1000.0, feo_pct=10.0, mno_pct=5.0, sio2_pct=2.0)
    o_kg = compute_o_from_slag(slag)

    feo_part = 1000.0 * (16.0 / 72.0) * 0.10
    mno_part = 1000.0 * (16.0 / 71.0) * 0.05
    sio2_part = 1000.0 * (32.0 / 60.0) * 0.02
    expected = feo_part + mno_part + sio2_part
    assert o_kg == pytest.approx(expected, abs=1e-9)
    # Sanity check на абсолютное значение
    assert o_kg == pytest.approx(44.156, abs=0.01)


def test_compute_o_from_slag_zero_mass_returns_zero():
    """Шлак нулевой массы → 0 kg O (граничный случай)."""
    slag = SlagState(mass_kg=0.0, feo_pct=18.0)
    assert compute_o_from_slag(slag) == 0.0


def test_compute_o_from_slag_rejects_negative():
    """Отрицательные масса/проценты → ValueError."""
    with pytest.raises(ValueError, match="≥ 0"):
        compute_o_from_slag(SlagState(mass_kg=-100.0, feo_pct=18.0))
    with pytest.raises(ValueError, match="≥ 0"):
        compute_o_from_slag(SlagState(mass_kg=2200.0, feo_pct=-1.0))
    with pytest.raises(ValueError, match="≥ 0"):
        compute_o_from_slag(SlagState(mass_kg=2200.0, feo_pct=18.0, mno_pct=-0.5))
    with pytest.raises(ValueError, match="≥ 0"):
        compute_o_from_slag(SlagState(mass_kg=2200.0, feo_pct=18.0, sio2_pct=-2.0))


def test_compute_o_consumed_by_si_basic():
    """100 kg FeSi-75, η=0.95 → 100·0.75·0.95·(32/28) ≈ 81.43 kg O.

    Detail: Si_kg = 75, Si_oxidized = 71.25, O = 71.25 × 32/28 = 81.4286 kg.
    """
    co = CoDeoxSi(si_source_kg=100.0, si_content_pct=75.0, eta_si=0.95)
    o_kg = compute_o_consumed_by_si(co)
    assert o_kg == pytest.approx(81.4286, abs=0.1)


def test_compute_o_consumed_by_si_simn():
    """200 kg SiMn (17% Si), η_Si=0.95 (default).

    Si_kg = 200 × 0.17 = 34.0
    Si_oxidized = 34.0 × 0.95 = 32.3
    O = 32.3 × 32/28 = 36.9143 kg
    """
    co = CoDeoxSi(si_source_kg=200.0, si_content_pct=17.0)  # default eta_si=0.95
    assert co.eta_si == 0.95
    o_kg = compute_o_consumed_by_si(co)
    expected = 200.0 * 0.17 * 0.95 * (32.0 / 28.0)
    assert o_kg == pytest.approx(expected, abs=1e-9)
    assert o_kg == pytest.approx(36.9143, abs=0.01)


def test_compute_o_consumed_by_si_zero_source_returns_zero():
    """Если FeSi не вводили — 0 kg O (граничный случай)."""
    co = CoDeoxSi(si_source_kg=0.0)
    assert compute_o_consumed_by_si(co) == 0.0


def test_compute_o_consumed_by_si_rejects_invalid():
    """Невалидные параметры → ValueError."""
    # Отрицательная масса
    with pytest.raises(ValueError, match="si_source_kg"):
        compute_o_consumed_by_si(CoDeoxSi(si_source_kg=-10.0))
    # %Si = 0 (выход за нижнюю границу диапазона)
    with pytest.raises(ValueError, match="si_content_pct"):
        compute_o_consumed_by_si(CoDeoxSi(si_source_kg=100.0, si_content_pct=0.0))
    # %Si > 100
    with pytest.raises(ValueError, match="si_content_pct"):
        compute_o_consumed_by_si(CoDeoxSi(si_source_kg=100.0, si_content_pct=110.0))
    # eta_si < 0
    with pytest.raises(ValueError, match="eta_si"):
        compute_o_consumed_by_si(
            CoDeoxSi(si_source_kg=100.0, si_content_pct=75.0, eta_si=-0.01)
        )
    # eta_si > 1
    with pytest.raises(ValueError, match="eta_si"):
        compute_o_consumed_by_si(
            CoDeoxSi(si_source_kg=100.0, si_content_pct=75.0, eta_si=1.5)
        )


def test_slag_state_and_co_deox_si_are_frozen():
    """SlagState и CoDeoxSi — frozen, нельзя случайно мутировать."""
    slag = SlagState(mass_kg=1000.0, feo_pct=15.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        slag.feo_pct = 99.0  # type: ignore[misc]

    co = CoDeoxSi(si_source_kg=100.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        co.eta_si = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PR 3: compute_al_demand_slag_aware + Excel regression
# ---------------------------------------------------------------------------


# Базовый набор входов из Excel-калькулятора пользователя (371 т ASIS).
# Используется в нескольких тестах ниже.
_EXCEL_BASE_INPUTS = dict(
    steel_mass_ton=371.0,
    o_a_initial_ppm=657.0,
    target_o_a_ppm=5.0,
    target_al_pct=0.018,
    temperature_C=1600.0,
)


def test_compute_al_demand_slag_aware_excel_base_case():
    """Excel base-case: 371 т, 657→5 ppm, M_slag=2.2 т @ FeO=18%, target [Al]=0.018%.

    Аналитический ожидаемый расчёт при η_Al=0.82 (asis_shot):
        O_diss     = (657 - 5) × 1e-6 × 371 × 1000 = 241.892 kg
        O_slag     = 2200 × (16/72) × 0.18         =  88.000 kg
        O_net      = 241.892 + 88.000              = 329.892 kg
        Al_active  = 329.892 × 1.12417             ≈ 370.83 kg
        Al_for_O   = 370.83 / 0.82                 ≈ 452.23 kg  (с burn-off)
        Al_residual= 371 × 1000 × 0.018% / 0.82    ≈  81.44 kg  (residual [Al])
        Al_pure    = 452.23 + 81.44                ≈ 533.67 kg

    Excel-калькулятор с k=0.89 даёт ~454 kg — разница ~15-17%
    (Excel игнорирует residual [Al] target и использует empirical k).
    Тест проверяет физическую правильность через аналитическое значение.
    """
    slag = SlagState(mass_kg=2200.0, feo_pct=18.0)
    result = compute_al_demand_slag_aware(
        method="asis_shot",
        slag=slag,
        **_EXCEL_BASE_INPUTS,
    )

    assert isinstance(result, SlagAwareDemandResult)
    assert result.method_id == "asis_shot"
    assert result.eta_al_used == pytest.approx(0.82, abs=1e-9)

    # O-balance компоненты
    assert result.o_dissolved_kg == pytest.approx(241.892, abs=0.01)
    assert result.o_in_slag_kg == pytest.approx(88.0, abs=0.01)
    assert result.o_consumed_by_si_kg == 0.0
    assert result.o_total_to_remove_kg == pytest.approx(329.892, abs=0.01)

    # Аналитический Al_pure
    expected_al_for_o = 329.892 * AL_TO_O_MASS_RATIO / 0.82
    expected_al_residual = 371.0 * 1000.0 * 0.018 / 100.0 / 0.82
    expected_al_pure = expected_al_for_o + expected_al_residual
    assert result.al_pure_kg == pytest.approx(expected_al_pure, rel=1e-3)

    # ~534 kg — финальное число; проверяем что в окрестности Excel ±15%
    assert 450.0 <= result.al_pure_kg <= 620.0, (
        f"al_pure_kg={result.al_pure_kg:.1f} out of expected band 450-620 kg"
    )

    # Charge ≈ pure для ASIS-shot (Al=99%)
    assert result.al_charge_kg == pytest.approx(result.al_pure_kg / 0.99, rel=1e-6)

    # Cost > 0 и breakdown содержит обе позиции
    assert result.cost_eur > 0
    assert "al_commodity_eur" in result.cost_breakdown
    assert "al_premium_eur" in result.cost_breakdown
    # premium = al_pure × 0.30 €/kg (asis_shot)
    assert result.cost_breakdown["al_premium_eur"] == pytest.approx(
        result.al_pure_kg * 0.30, rel=1e-6
    )


def test_compute_al_demand_no_slag_equals_basic_when_no_residual():
    """Без slag, без co_deox, target_al_pct=0 → al_pure_kg совпадает с basic
    compute_al_demand при том же burn_off_pct.

    Это smoke на интеграционную корректность wrapper'а: чистый delegation
    к существующей deoxidation.compute_al_demand.
    """
    inputs = dict(
        steel_mass_ton=100.0,
        o_a_initial_ppm=300.0,
        target_o_a_ppm=10.0,
        target_al_pct=0.0,
        temperature_C=1600.0,
    )
    result = compute_al_demand_slag_aware(method="asis_shot", **inputs)

    # η_Al=0.82 → burn_off = 18%
    basic = compute_al_demand(
        o_a_initial_ppm=300.0,
        temperature_C=1600.0,
        steel_mass_ton=100.0,
        target_o_a_ppm=10.0,
        al_purity_pct=100.0,
        burn_off_pct=18.0,
    )

    assert result.al_pure_kg == pytest.approx(basic.al_total_kg, rel=1e-9)
    assert result.o_in_slag_kg == 0.0
    assert result.o_consumed_by_si_kg == 0.0


def test_co_deox_reduces_al_demand():
    """Pre-deox by FeSi-75 уменьшает Al-потребность.

    Та же плавка с co_deox=CoDeoxSi(100 kg FeSi-75) должна дать al_pure_kg
    меньше, чем без co_deox (Si связывает 81.4 kg O).
    """
    slag = SlagState(mass_kg=2200.0, feo_pct=18.0)
    without_si = compute_al_demand_slag_aware(
        method="asis_shot", slag=slag, **_EXCEL_BASE_INPUTS
    )
    with_si = compute_al_demand_slag_aware(
        method="asis_shot",
        slag=slag,
        co_deox_si=CoDeoxSi(si_source_kg=100.0, si_content_pct=75.0, eta_si=0.95),
        **_EXCEL_BASE_INPUTS,
    )

    assert with_si.al_pure_kg < without_si.al_pure_kg, (
        f"co_deox должен снижать Al: with={with_si.al_pure_kg:.1f}, "
        f"without={without_si.al_pure_kg:.1f}"
    )
    assert with_si.o_consumed_by_si_kg == pytest.approx(81.43, abs=0.1)
    # Ожидаемое снижение: 81.43 kg O × 1.12417 / 0.82 ≈ 111.65 kg Al-pure
    expected_reduction = 81.43 * AL_TO_O_MASS_RATIO / 0.82
    actual_reduction = without_si.al_pure_kg - with_si.al_pure_kg
    assert actual_reduction == pytest.approx(expected_reduction, rel=1e-2)


def test_method_by_id_string_equals_object():
    """method='asis_shot' (строка) и method=AdditionMethod(...) дают идентичный результат."""
    methods = load_addition_methods()
    method_obj = methods["asis_shot"]

    by_string = compute_al_demand_slag_aware(
        method="asis_shot", **_EXCEL_BASE_INPUTS
    )
    by_object = compute_al_demand_slag_aware(
        method=method_obj, **_EXCEL_BASE_INPUTS
    )

    assert by_string.al_pure_kg == pytest.approx(by_object.al_pure_kg, rel=1e-12)
    assert by_string.cost_eur == pytest.approx(by_object.cost_eur, rel=1e-12)
    assert by_string.method_id == by_object.method_id


def test_unknown_method_id_raises():
    """Несуществующий method-id → ValueError со списком доступных."""
    with pytest.raises(ValueError, match="Unknown addition method"):
        compute_al_demand_slag_aware(method="not_a_method", **_EXCEL_BASE_INPUTS)


def test_eta_override_warning_when_far_outside_range():
    """η_Al=0.50 для asis_shot (range 0.75-0.90) → warning в результате."""
    result = compute_al_demand_slag_aware(
        method="asis_shot",
        eta_al_override=0.50,
        **_EXCEL_BASE_INPUTS,
    )
    assert result.eta_al_used == pytest.approx(0.50, abs=1e-9)
    assert any("η_Al" in w or "литературного диапазона" in w for w in result.warnings), (
        f"Expected η_Al override warning, got warnings: {result.warnings}"
    )


def test_eta_override_within_range_no_warning():
    """η_Al=0.85 для asis_shot (range 0.75-0.90) — внутри, без warning."""
    result = compute_al_demand_slag_aware(
        method="asis_shot",
        eta_al_override=0.85,
        **_EXCEL_BASE_INPUTS,
    )
    eta_warnings = [w for w in result.warnings if "η_Al" in w]
    assert eta_warnings == [], f"Unexpected η warnings: {eta_warnings}"


def test_cored_wire_feal30_charge_is_three_times_pure():
    """Для FeAl-30 charge mass = pure / 0.30 (≈3.33× pure)."""
    result = compute_al_demand_slag_aware(
        method="cored_wire_feal30", **_EXCEL_BASE_INPUTS
    )
    assert result.al_charge_kg == pytest.approx(result.al_pure_kg / 0.30, rel=1e-9)
    # Premium для FeAl-30 — €2.50/kg Al-eq
    assert result.cost_breakdown["al_premium_eur"] == pytest.approx(
        result.al_pure_kg * 2.50, rel=1e-9
    )


def test_invalid_steel_mass_raises():
    """steel_mass_ton ≤ 0 → ValueError."""
    bad = dict(_EXCEL_BASE_INPUTS, steel_mass_ton=0)
    with pytest.raises(ValueError, match="steel_mass_ton"):
        compute_al_demand_slag_aware(method="asis_shot", **bad)


def test_target_al_negative_raises():
    """target_al_pct < 0 → ValueError."""
    bad = dict(_EXCEL_BASE_INPUTS, target_al_pct=-0.01)
    with pytest.raises(ValueError, match="target_al_pct"):
        compute_al_demand_slag_aware(method="asis_shot", **bad)


# ---------------------------------------------------------------------------
# PR 3: cost_model — kind="deox_consumable"
# ---------------------------------------------------------------------------


def test_cost_model_accepts_deox_consumable_kind():
    """_validate_material_dict не падает для kind=deox_consumable с
    addition_method_id; падает без addition_method_id.
    """
    from app.backend.cost_model import _validate_material_dict

    # Корректный deox_consumable — FeAl-30 с ссылкой на YAML
    good = {
        "kind": "deox_consumable",
        "price_per_kg": 4.80,
        "element_content": {"Al": 0.30, "Fe": 0.70},  # сумма ≈ 1.0 — ОК
        "addition_method_id": "cored_wire_feal30",
    }
    _validate_material_dict("Al-feal30", good)  # no raise

    # element_content сумма ≠ 1.0 (FeAl-30 без Fe-добивки) — для
    # deox_consumable не валидируется
    relaxed = {
        "kind": "deox_consumable",
        "price_per_kg": 4.80,
        "element_content": {"Al": 0.30},  # сумма 0.30 — обычно zero, но здесь OK
        "addition_method_id": "cored_wire_feal30",
    }
    _validate_material_dict("Al-feal30-relaxed", relaxed)  # no raise

    # Отсутствует addition_method_id — ValueError
    missing_id = {
        "kind": "deox_consumable",
        "price_per_kg": 2.70,
        "element_content": {"Al": 1.0},
    }
    with pytest.raises(ValueError, match="addition_method_id"):
        _validate_material_dict("Al-asis-shot", missing_id)


def test_cost_model_load_snapshot_with_deox_consumable(tmp_path):
    """end-to-end: YAML snapshot с deox_consumable грузится через load_snapshot."""
    from datetime import date

    from app.backend.cost_model import load_snapshot

    snap_yaml = tmp_path / "snap_with_deox.yaml"
    snap_yaml.write_text(
        yaml.safe_dump(
            {
                "date": date(2026, 5, 12).isoformat(),
                "currency": "EUR",
                "source": "test",
                "materials": {
                    "scrap": {
                        "kind": "base",
                        "price_per_kg": 0.30,
                        "element_content": {"Fe": 1.0},
                    },
                    "Al-asis-shot": {
                        "kind": "deox_consumable",
                        "price_per_kg": 2.70,
                        "element_content": {"Al": 1.0},
                        "addition_method_id": "asis_shot",
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    snap = load_snapshot(snap_yaml)
    assert "Al-asis-shot" in snap.materials
    assert snap.materials["Al-asis-shot"].kind == "deox_consumable"
    assert snap.materials["Al-asis-shot"].price_per_kg == 2.70


def test_cost_model_rejects_unknown_kind():
    """kind='nonsense' (не из Literal) → ValueError."""
    from app.backend.cost_model import _validate_material_dict

    bad = {
        "kind": "wonkystuff",
        "price_per_kg": 1.0,
        "element_content": {"Al": 1.0},
    }
    with pytest.raises(ValueError, match="kind must be"):
        _validate_material_dict("bad", bad)


# ---------------------------------------------------------------------------
# PR 4: compare_addition_methods + recommend_optimal_method
# ---------------------------------------------------------------------------


def test_compare_methods_ordered_by_cost():
    """compare_addition_methods возвращает list[MethodCompareRow] sorted ascending
    по cost_per_heat_eur (cheapest first)."""
    rows = compare_addition_methods(
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    assert isinstance(rows, list)
    assert all(isinstance(r, MethodCompareRow) for r in rows)
    assert len(rows) >= 2
    # Sorted ascending
    for i in range(len(rows) - 1):
        assert rows[i].cost_per_heat_eur <= rows[i + 1].cost_per_heat_eur, (
            f"Not sorted: rows[{i}]={rows[i].cost_per_heat_eur} > "
            f"rows[{i+1}]={rows[i+1].cost_per_heat_eur}"
        )


def test_compare_methods_uses_all_catalog_methods():
    """По умолчанию (method_ids=None) сравниваются ВСЕ методы из YAML."""
    rows = compare_addition_methods(
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    catalog_ids = set(load_addition_methods().keys())
    row_ids = {r.method_id for r in rows}
    assert row_ids == catalog_ids, (
        f"Missing methods: {catalog_ids - row_ids}; extra: {row_ids - catalog_ids}"
    )
    # Каждая row имеет валидный scatter_kg ≥ 0
    for r in rows:
        assert r.scatter_kg >= 0, f"{r.method_id}: scatter_kg={r.scatter_kg} < 0"


def test_compare_methods_explicit_subset():
    """method_ids=['asis_shot', 'ingot'] → только эти 2."""
    rows = compare_addition_methods(
        method_ids=["asis_shot", "ingot"],
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    assert {r.method_id for r in rows} == {"asis_shot", "ingot"}


def test_compare_methods_rejects_unknown_id():
    """compare_addition_methods с unknown method_id → ValueError."""
    with pytest.raises(ValueError, match="Unknown addition method"):
        compare_addition_methods(
            method_ids=["asis_shot", "nonexistent"],
            **_EXCEL_BASE_INPUTS,
        )


def test_recommend_filters_by_n_constraint(monkeypatch, tmp_path):
    """target_n_ppm=30 → исключает методы с carrier_gas=='N2'.

    В default YAML только asis_shot имеет carrier_gas (Ar). Чтобы протестировать
    фильтр N2, временно подменяем каталог YAML на тестовый с N2-методом.
    """
    test_yaml = tmp_path / "test_methods.yaml"
    test_yaml.write_text(
        yaml.safe_dump({
            "methods": {
                "asis_shot_ar": {
                    "name": "ASIS Ar",
                    "eta_al_typical": 0.82,
                    "eta_al_range": [0.75, 0.90],
                    "premium_eur_per_kg": 0.30,
                    "carrier_gas": "Ar",
                    "surface_m2_per_kg": 1.7,
                },
                "asis_shot_n2": {
                    "name": "ASIS N2",
                    "eta_al_typical": 0.82,
                    "eta_al_range": [0.75, 0.90],
                    "premium_eur_per_kg": 0.20,  # дешевле — без фильтра выиграл бы
                    "carrier_gas": "N2",
                    "surface_m2_per_kg": 1.7,
                },
                "ingot_no_gas": {
                    "name": "Ingot",
                    "eta_al_typical": 0.58,
                    "eta_al_range": [0.50, 0.65],
                    "premium_eur_per_kg": 0.0,
                    "carrier_gas": None,
                    "surface_m2_per_kg": 0.02,
                },
            }
        }),
        encoding="utf-8",
    )
    # Patch DEOX_METHODS_PATH через monkeypatch на module-level
    from app.backend import slag_aware_deox
    monkeypatch.setattr(slag_aware_deox, "DEOX_METHODS_PATH", test_yaml)
    load_addition_methods.cache_clear()

    rec = recommend_optimal_method(
        target_n_ppm=30.0,  # < 50 → активирует фильтр N2
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )

    # N2-метод должен быть в rejected, не в pareto
    rejected_ids = {r["method_id"] for r in rec.rejected_methods}
    pareto_ids = {r.method_id for r in rec.pareto_table}
    assert "asis_shot_n2" in rejected_ids, (
        f"N2-method should be rejected, got rejected={rejected_ids}"
    )
    assert "asis_shot_n2" not in pareto_ids
    assert any("N2" in c for c in rec.constraints_active)


def test_recommend_filters_by_premium_cap():
    """premium_cap_eur_per_kg=0.20 → отсеивает asis_shot (0.30) и cored_wire (2.50)."""
    rec = recommend_optimal_method(
        premium_cap_eur_per_kg=0.20,
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    rejected_ids = {r["method_id"] for r in rec.rejected_methods}
    assert "cored_wire_feal30" in rejected_ids
    assert "asis_shot" in rejected_ids
    # ingot/submerged_ingot имеют premium ≤ 0.20 → survive
    pareto_ids = {r.method_id for r in rec.pareto_table}
    assert "ingot" in pareto_ids
    # constraints_active содержит описание premium-cap
    assert any("premium" in c.lower() for c in rec.constraints_active)


def test_recommend_returns_runner_up():
    """С полным каталогом без constraints → chosen + runner_up оба заполнены."""
    rec = recommend_optimal_method(
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    assert isinstance(rec, OptimizationRecommendation)
    assert rec.chosen_method_id is not None
    assert rec.chosen_cost_eur > 0
    # Каталог имеет >=5 методов → runner-up должен быть
    assert rec.runner_up_method_id is not None
    assert rec.runner_up_cost_eur is not None
    assert rec.runner_up_delta_eur is not None
    # Runner-up дороже chosen (cost-asc sort)
    assert rec.runner_up_delta_eur >= 0, (
        f"delta {rec.runner_up_delta_eur} < 0 — runner-up should be more expensive"
    )
    assert rec.runner_up_cost_eur == pytest.approx(
        rec.chosen_cost_eur + rec.runner_up_delta_eur, rel=1e-9
    )


def test_recommend_rationale_present():
    """Rationale — non-empty человеческая строка с упоминанием cost и η_Al."""
    rec = recommend_optimal_method(
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    assert isinstance(rec.rationale, str)
    assert len(rec.rationale) > 30, f"rationale too short: {rec.rationale!r}"
    # Должно содержать ключевые слова "cost"/"€" + "η_Al" или название метода
    assert "€" in rec.rationale or "cost" in rec.rationale.lower()
    assert rec.chosen_method_name in rec.rationale


def test_recommend_raises_when_no_valid_methods(monkeypatch, tmp_path):
    """Если все методы отфильтрованы constraints → ValueError."""
    # Каталог только из одного N2-метода → target_n_ppm=10 отсеет его.
    test_yaml = tmp_path / "n2_only.yaml"
    test_yaml.write_text(
        yaml.safe_dump({
            "methods": {
                "only_n2": {
                    "name": "Only N2",
                    "eta_al_typical": 0.80,
                    "eta_al_range": [0.70, 0.90],
                    "premium_eur_per_kg": 0.10,
                    "carrier_gas": "N2",
                    "surface_m2_per_kg": 1.0,
                },
            }
        }),
        encoding="utf-8",
    )
    from app.backend import slag_aware_deox
    monkeypatch.setattr(slag_aware_deox, "DEOX_METHODS_PATH", test_yaml)
    load_addition_methods.cache_clear()

    with pytest.raises(ValueError, match="не осталось"):
        recommend_optimal_method(
            target_n_ppm=10.0,  # < 50 → отсеит only_n2
            **_EXCEL_BASE_INPUTS,
        )


def test_recommend_n_constraint_inactive_when_n_above_50():
    """target_n_ppm=80 (>=50) → фильтр N2 НЕ активирован."""
    rec = recommend_optimal_method(
        target_n_ppm=80.0,
        slag=SlagState(mass_kg=2200.0, feo_pct=18.0),
        **_EXCEL_BASE_INPUTS,
    )
    # Никаких "N2" в constraints_active
    assert not any("N2" in c for c in rec.constraints_active)
    # asis_shot (Ar) и остальные присутствуют в pareto
    pareto_ids = {r.method_id for r in rec.pareto_table}
    assert pareto_ids == set(load_addition_methods().keys())


def test_cost_breakdown_has_gas_and_handling_keys():
    """SlagAwareDemandResult.cost_breakdown содержит gas_eur и handling_eur.

    Для методов без price_per_nm3 в YAML — gas_eur=None (carrier_gas задан, но
    цена не указана → 0.0) или None (carrier_gas отсутствует).
    handling_eur=0.0 если handling_eur_per_kg_al не задан.
    """
    # asis_shot имеет carrier_gas="Ar", но без price_per_nm3 → gas_eur=0.0
    res_asis = compute_al_demand_slag_aware(
        method="asis_shot", **_EXCEL_BASE_INPUTS
    )
    assert "gas_eur" in res_asis.cost_breakdown
    assert "handling_eur" in res_asis.cost_breakdown
    assert res_asis.cost_breakdown["gas_eur"] == 0.0  # carrier_gas, но цены нет
    assert res_asis.cost_breakdown["handling_eur"] == 0.0

    # ingot — carrier_gas=None → gas_eur=None
    res_ingot = compute_al_demand_slag_aware(
        method="ingot", **_EXCEL_BASE_INPUTS
    )
    assert res_ingot.cost_breakdown["gas_eur"] is None
    assert res_ingot.cost_breakdown["handling_eur"] == 0.0


# ── PR 6: η_Al uncertainty propagation ──


def test_posterior_none_collapses_to_point():
    """Без posterior — p10=p50=p90=al_pure_kg."""
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0,
        o_a_initial_ppm=500.0, target_o_a_ppm=5.0, target_al_pct=0.02,
        method="asis_shot",
        eta_al_posterior_logit=None,
    )
    assert res.al_pure_kg_p10 == res.al_pure_kg_p50 == res.al_pure_kg_p90
    assert res.al_pure_kg == res.al_pure_kg_p50  # backwards-compat


def test_posterior_zero_sigma_raises():
    with pytest.raises(ValueError, match="posterior sigma"):
        compute_al_demand_slag_aware(
            steel_mass_ton=100.0, o_a_initial_ppm=500.0,
            target_o_a_ppm=5.0, target_al_pct=0.02,
            method="asis_shot",
            eta_al_posterior_logit=(1.5, 0.0),
        )


def test_posterior_widens_interval_with_sigma():
    """σ > 0 → p10 < p50 < p90, ширина увеличивается с σ."""
    base_kwargs = dict(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
    )
    # μ_post = logit(0.82) ≈ 1.516
    import math
    mu = math.log(0.82 / 0.18)
    res_narrow = compute_al_demand_slag_aware(
        **base_kwargs, eta_al_posterior_logit=(mu, 0.05),
    )
    res_wide = compute_al_demand_slag_aware(
        **base_kwargs, eta_al_posterior_logit=(mu, 0.3),
    )
    assert res_narrow.al_pure_kg_p10 < res_narrow.al_pure_kg_p50 < res_narrow.al_pure_kg_p90
    assert res_wide.al_pure_kg_p10 < res_wide.al_pure_kg_p50 < res_wide.al_pure_kg_p90
    # Wide should have larger ratio p90/p10
    ratio_narrow = res_narrow.al_pure_kg_p90 / res_narrow.al_pure_kg_p10
    ratio_wide = res_wide.al_pure_kg_p90 / res_wide.al_pure_kg_p10
    assert ratio_wide > ratio_narrow


def test_p50_close_to_invlogit_mu_when_posterior_given():
    """p50 = K / invlogit(μ_post)."""
    import math
    mu = math.log(0.85 / 0.15)  # ≈ logit(0.85) = 1.7346
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
        eta_al_posterior_logit=(mu, 0.1),
    )
    # p50 должен быть близок к Al при η=0.85, не Al при η=0.82 (literature)
    res_at_85 = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
        eta_al_override=0.85,
    )
    assert abs(res.al_pure_kg_p50 - res_at_85.al_pure_kg) / res_at_85.al_pure_kg < 0.02


def test_override_wins_over_posterior():
    """Если задан override и posterior — override побеждает + warning."""
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
        eta_al_override=0.90,
        eta_al_posterior_logit=(1.5, 0.2),
    )
    assert res.al_pure_kg_p10 == res.al_pure_kg_p50 == res.al_pure_kg_p90
    assert any("override" in w.lower() and "posterior" in w.lower() for w in res.warnings)


def test_posterior_invalid_format_raises():
    with pytest.raises(ValueError):
        compute_al_demand_slag_aware(
            steel_mass_ton=100.0, o_a_initial_ppm=500.0,
            target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
            eta_al_posterior_logit=(1.5,),  # not 2-tuple
        )


def test_quantile_inverse_monotonic_eta_to_al():
    """Численная проверка: p90_Al = K / η_q10, p10_Al = K / η_q90."""
    import math
    mu, sigma = 1.5, 0.2
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
        eta_al_posterior_logit=(mu, sigma),
    )
    # Recover K from p50 = K / invlogit(mu)
    Z_80 = 1.2816

    def invlogit(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    eta_median = invlogit(mu)
    eta_q10 = invlogit(mu - Z_80 * sigma)
    eta_q90 = invlogit(mu + Z_80 * sigma)
    K = res.al_pure_kg_p50 * eta_median
    assert abs(res.al_pure_kg_p90 - K / eta_q10) < 0.01
    assert abs(res.al_pure_kg_p10 - K / eta_q90) < 0.01


def test_inputs_snapshot_includes_posterior():
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
        eta_al_posterior_logit=(1.5, 0.1),
    )
    assert res.inputs.get("eta_al_posterior_logit") == [1.5, 0.1]


def test_inputs_snapshot_includes_none_posterior():
    res = compute_al_demand_slag_aware(
        steel_mass_ton=100.0, o_a_initial_ppm=500.0,
        target_o_a_ppm=5.0, target_al_pct=0.02, method="asis_shot",
    )
    assert res.inputs.get("eta_al_posterior_logit") is None


# ── PR 7: multi-objective ─────────────────────────────────────────────


def test_recommend_default_objective_is_cost():
    """Без objective kwarg recommend_optimal_method ведёт себя как до PR 7.

    Тот же chosen_method_id, ``objective == "cost"``, и ``pareto_frontier``
    остаётся пустым для не-pareto веток (контракт backwards-compat).
    """
    r1 = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
    )
    r2 = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
        objective="cost",
    )
    assert r1.chosen_method_id == r2.chosen_method_id
    assert r1.objective == "cost"
    assert r1.pareto_frontier == []


def test_recommend_objective_al_mass_picks_min_al():
    """objective='al_mass' → chosen = min al_pure_kg среди survivors."""
    r = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
        objective="al_mass",
    )
    chosen_al = next(
        row.al_pure_kg for row in r.pareto_table
        if row.method_id == r.chosen_method_id
    )
    all_al = [row.al_pure_kg for row in r.pareto_table]
    assert chosen_al == min(all_al)
    assert r.objective == "al_mass"


def test_recommend_objective_pareto_returns_frontier_field():
    """objective='pareto' → pareto_frontier ≥ 1, все non-dominated."""
    r = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
        objective="pareto",
    )
    assert r.objective == "pareto"
    assert len(r.pareto_frontier) >= 1
    # Каждая точка frontier должна быть non-dominated внутри frontier.
    for p in r.pareto_frontier:
        for o in r.pareto_frontier:
            if o is p:
                continue
            dominated = (
                o.al_pure_kg <= p.al_pure_kg
                and o.cost_per_heat_eur <= p.cost_per_heat_eur
                and (
                    o.al_pure_kg < p.al_pure_kg
                    or o.cost_per_heat_eur < p.cost_per_heat_eur
                )
            )
            assert not dominated, (
                f"{p.method_id} dominated by {o.method_id} inside frontier"
            )


def test_recommend_objective_pareto_chosen_is_knee():
    """Chosen — точка с минимальным L2 от утопии в min-max norm space."""
    r = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
        objective="pareto",
    )
    als = [p.al_pure_kg for p in r.pareto_frontier]
    costs = [p.cost_per_heat_eur for p in r.pareto_frontier]
    al_min, al_max = min(als), max(als)
    cost_min, cost_max = min(costs), max(costs)
    al_range = max(al_max - al_min, 1e-9)
    cost_range = max(cost_max - cost_min, 1e-9)

    def l2(p):
        a = (p.al_pure_kg - al_min) / al_range
        c = (p.cost_per_heat_eur - cost_min) / cost_range
        return (a * a + c * c) ** 0.5

    chosen = next(
        p for p in r.pareto_frontier if p.method_id == r.chosen_method_id
    )
    chosen_l2 = l2(chosen)
    for p in r.pareto_frontier:
        assert l2(p) >= chosen_l2 - 1e-9


def test_pareto_filter_removes_dominated():
    """Synthetic 4 точки — 2 dominated, 2 на frontier."""
    from app.backend.slag_aware_deox import MethodCompareRow, _pareto_filter
    rows = [
        MethodCompareRow(
            method_id="A", method_name="A", eta_al_used=0.8,
            al_pure_kg=100.0, al_charge_kg=102.0,
            cost_per_heat_eur=300.0, cost_per_ton_eur=3.0,
            al_specific_kg_per_t=1.0, carrier_gas=None,
            scatter_kg=5.0, warnings=[],
        ),
        MethodCompareRow(
            method_id="B", method_name="B", eta_al_used=0.8,
            al_pure_kg=120.0, al_charge_kg=122.0,
            cost_per_heat_eur=400.0, cost_per_ton_eur=4.0,
            al_specific_kg_per_t=1.2, carrier_gas=None,
            scatter_kg=5.0, warnings=[],
        ),  # dominated by A
        MethodCompareRow(
            method_id="C", method_name="C", eta_al_used=0.8,
            al_pure_kg=80.0, al_charge_kg=82.0,
            cost_per_heat_eur=500.0, cost_per_ton_eur=5.0,
            al_specific_kg_per_t=0.8, carrier_gas=None,
            scatter_kg=5.0, warnings=[],
        ),  # frontier (lower Al than A)
        MethodCompareRow(
            method_id="D", method_name="D", eta_al_used=0.8,
            al_pure_kg=150.0, al_charge_kg=152.0,
            cost_per_heat_eur=600.0, cost_per_ton_eur=6.0,
            al_specific_kg_per_t=1.5, carrier_gas=None,
            scatter_kg=5.0, warnings=[],
        ),  # dominated by A
    ]
    front = _pareto_filter(rows)
    ids = {r.method_id for r in front}
    assert ids == {"A", "C"}


def test_pareto_filter_singleton():
    """Singleton frontier — _pick_knee возвращает (only, None, 0.0)."""
    from app.backend.slag_aware_deox import (
        MethodCompareRow,
        _pareto_filter,
        _pick_knee,
    )
    rows = [
        MethodCompareRow(
            method_id="A", method_name="A", eta_al_used=0.8,
            al_pure_kg=100.0, al_charge_kg=102.0,
            cost_per_heat_eur=300.0, cost_per_ton_eur=3.0,
            al_specific_kg_per_t=1.0, carrier_gas=None,
            scatter_kg=5.0, warnings=[],
        ),
    ]
    front = _pareto_filter(rows)
    assert len(front) == 1
    chosen, runner_up, l2 = _pick_knee(front)
    assert chosen.method_id == "A"
    assert runner_up is None
    assert l2 == 0.0


def test_recommend_unknown_objective_raises():
    """Unknown objective → ValueError на самом раннем шаге."""
    with pytest.raises(ValueError, match="Unknown objective"):
        recommend_optimal_method(
            steel_mass_ton=371.0, o_a_initial_ppm=657.0,
            target_o_a_ppm=5.0, target_al_pct=0.018,
            objective="unknown",
        )


def test_pareto_dominated_methods_in_rejected():
    """Dominated методы попадают в rejected_methods с reason."""
    r = recommend_optimal_method(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
        objective="pareto",
    )
    # full set из compare_addition_methods даёт 5 методов в seed-каталоге.
    # Если хотя бы один метод dominated (frontier < survivors) — он должен
    # появиться в rejected с reason содержащим "dominated".
    full_methods = compare_addition_methods(
        steel_mass_ton=371.0, o_a_initial_ppm=657.0,
        target_o_a_ppm=5.0, target_al_pct=0.018,
    )
    if len(r.pareto_frontier) < len(full_methods):
        assert any(
            "dominated" in str(rej.get("reason", ""))
            for rej in r.rejected_methods
        )
