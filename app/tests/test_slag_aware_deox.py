"""Tests for slag_aware_deox module.

PR 1 — YAML loader + AdditionMethod dataclass.
PR 2 — SlagState/CoDeoxSi dataclasses + O-balance functions.
PR 3-4 — compute_al_demand_slag_aware, compare_methods, recommend_optimal.
"""
from __future__ import annotations

import pytest
import yaml

from app.backend.slag_aware_deox import (
    AdditionMethod,
    CoDeoxSi,
    SlagState,
    compute_o_consumed_by_si,
    compute_o_from_slag,
    list_method_ids,
    load_addition_methods,
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
