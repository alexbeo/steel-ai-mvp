"""Unit tests for deoxidation calculator."""
from __future__ import annotations

import pytest

from app.backend.deoxidation import (
    AL_TO_O_MASS_RATIO,
    DEFAULT_MODEL_ID,
    THERMO_MODELS,
)


def test_al_to_o_mass_ratio_stoichiometry():
    # 2*26.98 / (3*16.00) = 1.124166...
    assert AL_TO_O_MASS_RATIO == pytest.approx(1.12417, rel=1e-4)


def test_three_thermo_models_registered():
    assert set(THERMO_MODELS) == {
        "fruehan_1985", "sigworth_elliott_1974", "hayashi_2013",
    }
    assert DEFAULT_MODEL_ID == "fruehan_1985"


def test_fruehan_log_k_at_1873K():
    """log_k(T=1873 K, 1600°C) = 64000/1873 - 20.57 ≈ 13.607."""
    model = THERMO_MODELS["fruehan_1985"]
    assert model.log_k(1873.0) == pytest.approx(13.607, abs=0.01)


def test_sigworth_elliott_log_k_at_1873K():
    model = THERMO_MODELS["sigworth_elliott_1974"]
    # 62680/1873 - 20.54 ≈ 12.927
    assert model.log_k(1873.0) == pytest.approx(12.927, abs=0.01)


def test_hayashi_log_k_at_1873K():
    model = THERMO_MODELS["hayashi_2013"]
    # -62780/1873 + 19.18 ≈ -14.337
    assert model.log_k(1873.0) == pytest.approx(-14.337, abs=0.01)
