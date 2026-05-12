"""Tests for slag_aware_deox module (PR 1 — YAML loader only).

PR 2-4 будут добавлять тесты на compute-функции. Здесь только loader и dataclass.
"""
from __future__ import annotations

import pytest
import yaml

from app.backend.slag_aware_deox import (
    AdditionMethod,
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
