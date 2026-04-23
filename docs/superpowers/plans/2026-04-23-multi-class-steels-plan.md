# Multi-class Steels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for a second steel class — EN 10083-2 Q&T carbon steels (C22/C35/C45/C60) — alongside existing pipe-HSLA. Class becomes an attribute of each trained model; UI follows the active model's class automatically.

**Architecture:** YAML profiles under `data/steel_classes/` define per-class feature_set, bounds, targets, expected top-features. Python synthetic generators (`generate_synthetic_hsla_dataset`, `generate_synthetic_en10083_qt_dataset`) encode physics. `app/backend/steel_classes.py` is the loader/registry. `TrainedModel.steel_class` is persisted in `meta.json`. Pattern Library M05/D07 read expected_features/bounds from ctx.

**Tech Stack:** Python 3.12, PyYAML (already installed), existing XGBoost+Optuna pipeline, Streamlit. No new libraries.

**Spec:** `docs/superpowers/specs/2026-04-23-multi-class-steels-design.md`

---

## File Structure

### New
- `app/backend/steel_classes.py` — `SteelClassProfile`, `load_steel_class`, `available_steel_classes`, `get_synthetic_generator`, `compute_features_for_class`.
- `data/steel_classes/pipe_hsla.yaml`
- `data/steel_classes/en10083_qt.yaml`
- `app/tests/test_steel_classes.py`

### Modified
- `app/backend/data_curator.py` — add `generate_synthetic_en10083_qt_dataset`, `save_sample_dataset_en10083_qt`.
- `app/backend/model_trainer.py` — `train_model(..., steel_class="pipe_hsla")`; `TrainedModel.steel_class`; version prefix per class; meta persists it.
- `app/backend/engine.py` — `_build_critic_context` for training phase reads per-class profile.
- `pattern_library/patterns.py` — `_check_m05` reads `expected_top_features` from ctx; `_check_d07` reads `physical_bounds` from ctx (fallback to HSLA constants).
- `app/frontend/app.py` — class dropdown in train tab, badge in sidebar, target dropdown filtered by class, conditional predict fields, banner on design tab when Q&T.
- `CLAUDE.md` — one line about multi-class convention.

---

## Task 1: Create YAML profiles + steel_classes.py scaffold

**Files:**
- Create: `data/steel_classes/pipe_hsla.yaml`
- Create: `data/steel_classes/en10083_qt.yaml`
- Create: `app/backend/steel_classes.py`

- [ ] **Step 1: Create pipe_hsla.yaml**

Write `data/steel_classes/pipe_hsla.yaml` with exact content from spec §4.1 `pipe_hsla.yaml`.

- [ ] **Step 2: Create en10083_qt.yaml**

Write `data/steel_classes/en10083_qt.yaml` with exact content from spec §4.1 `en10083_qt.yaml`.

- [ ] **Step 3: Create steel_classes.py**

Write `app/backend/steel_classes.py`:

```python
"""Steel class profiles — loader + registry.

Each profile is a YAML under data/steel_classes/ describing feature set,
physical bounds, target properties, expected top-features for Critic,
and the name of the synthetic data generator function.

Class id is persisted in TrainedModel.meta["steel_class"] so that the
downstream UI can follow the active model's class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEEL_CLASSES_DIR = PROJECT_ROOT / "data" / "steel_classes"
AVAILABLE_CLASS_IDS = ["pipe_hsla", "en10083_qt"]


@dataclass
class TargetProperty:
    id: str
    label: str
    range: list[float]


@dataclass
class SteelClassProfile:
    id: str
    name: str
    standard: str
    target_properties: list[TargetProperty]
    feature_set: list[str]
    physical_bounds: dict[str, list[float]]
    expected_top_features: list[str]
    process_params: list[str]
    synthetic_generator_name: str
    cost_seed_path: str
    feature_engineering: str

    def target_ids(self) -> list[str]:
        return [t.id for t in self.target_properties]


_PROFILE_CACHE: dict[str, SteelClassProfile] = {}


def load_steel_class(class_id: str) -> SteelClassProfile:
    if class_id in _PROFILE_CACHE:
        return _PROFILE_CACHE[class_id]
    path = STEEL_CLASSES_DIR / f"{class_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = SteelClassProfile(
        id=data["id"],
        name=data["name"],
        standard=data["standard"],
        target_properties=[TargetProperty(**t) for t in data["target_properties"]],
        feature_set=data["feature_set"],
        physical_bounds={k: list(v) for k, v in data["physical_bounds"].items()},
        expected_top_features=data["expected_top_features"],
        process_params=data["process_params"],
        synthetic_generator_name=data["synthetic_generator_name"],
        cost_seed_path=data.get("cost_seed_path", ""),
        feature_engineering=data.get("feature_engineering", "passthrough"),
    )
    _PROFILE_CACHE[class_id] = profile
    return profile


def available_steel_classes() -> list[SteelClassProfile]:
    return [load_steel_class(cid) for cid in AVAILABLE_CLASS_IDS]


def get_synthetic_generator(generator_name: str) -> Callable:
    """Lazy import to avoid circular deps with data_curator."""
    from app.backend.data_curator import (
        generate_synthetic_en10083_qt_dataset,
        generate_synthetic_hsla_dataset,
    )
    return {
        "pipe_hsla": generate_synthetic_hsla_dataset,
        "en10083_qt": generate_synthetic_en10083_qt_dataset,
    }[generator_name]


def compute_features_for_class(df, class_id: str):
    profile = load_steel_class(class_id)
    if profile.feature_engineering == "compute_hsla_features":
        from app.backend.feature_eng import compute_hsla_features
        return compute_hsla_features(df)
    return df
```

- [ ] **Step 4: Verify import**

```bash
PYTHONPATH=. .venv/bin/python -c "from app.backend.steel_classes import load_steel_class, available_steel_classes; p = load_steel_class('pipe_hsla'); print(p.name, len(p.feature_set))"
.venv/bin/ruff check app/backend/steel_classes.py
```

Expected: `Pipe HSLA (API 5L X60-X70) 16` and ruff clean. `get_synthetic_generator` will raise ImportError until Task 2 adds `generate_synthetic_en10083_qt_dataset` — don't call it yet.

- [ ] **Step 5: Commit**

```bash
git add app/backend/steel_classes.py data/steel_classes/pipe_hsla.yaml data/steel_classes/en10083_qt.yaml
git commit -m "feat(steel_classes): YAML profiles for pipe_hsla + en10083_qt + loader"
```

---

## Task 2: Synthetic generator for EN 10083-2 Q&T

**Files:**
- Modify: `app/backend/data_curator.py`
- Create: `app/tests/test_steel_classes.py`

- [ ] **Step 1: Write failing tests**

Create `app/tests/test_steel_classes.py`:

```python
"""Unit tests for multi-class steel support."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backend.steel_classes import (
    AVAILABLE_CLASS_IDS,
    SteelClassProfile,
    available_steel_classes,
    compute_features_for_class,
    get_synthetic_generator,
    load_steel_class,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_pipe_hsla_profile():
    p = load_steel_class("pipe_hsla")
    assert p.id == "pipe_hsla"
    assert p.standard.startswith("API 5L")
    assert "c_pct" in p.feature_set
    assert "nb_pct" in p.feature_set
    assert p.physical_bounds["c_pct"] == [0.04, 0.12]
    assert "yield_strength_mpa" in p.target_ids()


def test_load_en10083_profile():
    p = load_steel_class("en10083_qt")
    assert p.id == "en10083_qt"
    assert p.standard == "EN 10083-2"
    assert "tempering_temp" in p.feature_set
    assert "section_thickness_mm" in p.feature_set
    assert "nb_pct" not in p.feature_set   # Q&T does not use microalloying
    assert p.physical_bounds["c_pct"] == [0.18, 0.65]
    assert "hardness_hrc" in p.target_ids()


def test_available_steel_classes_registry():
    profiles = available_steel_classes()
    assert len(profiles) == 2
    ids = {p.id for p in profiles}
    assert ids == set(AVAILABLE_CLASS_IDS)
    for p in profiles:
        assert isinstance(p, SteelClassProfile)


def test_synthetic_generator_en10083_qt_physical_sanity():
    gen = get_synthetic_generator("en10083_qt")
    df = gen(n_samples=500, random_seed=1)

    # Required columns
    for col in ("c_pct", "mn_pct", "tempering_temp",
                "section_thickness_mm", "hardness_hrc",
                "tensile_strength_mpa", "campaign_id", "heat_date"):
        assert col in df.columns, f"missing {col}"

    # Bounds
    assert df["c_pct"].between(0.18, 0.65).all()
    assert df["tempering_temp"].between(150, 650).all()
    assert df["hardness_hrc"].between(15, 65).all()

    # Physics: high-C + low-tempering → high HRC; low-C + high-tempering → low HRC
    hard_mask = (df["c_pct"] > 0.55) & (df["tempering_temp"] < 250)
    soft_mask = (df["c_pct"] < 0.25) & (df["tempering_temp"] > 550)
    if hard_mask.sum() > 10 and soft_mask.sum() > 10:
        assert df.loc[hard_mask, "hardness_hrc"].mean() > \
               df.loc[soft_mask, "hardness_hrc"].mean() + 5


def test_compute_features_for_class_passthrough_for_qt():
    df = pd.DataFrame({"c_pct": [0.4], "mn_pct": [0.6]})
    out = compute_features_for_class(df, "en10083_qt")
    assert list(out.columns) == ["c_pct", "mn_pct"]


def test_compute_features_for_class_adds_derived_for_hsla():
    df = pd.DataFrame({
        "c_pct": [0.08], "mn_pct": [1.5], "si_pct": [0.3],
        "p_pct": [0.015], "s_pct": [0.005], "cr_pct": [0.1],
        "ni_pct": [0.1], "mo_pct": [0.02], "cu_pct": [0.2],
        "al_pct": [0.03], "v_pct": [0.03], "nb_pct": [0.04],
        "ti_pct": [0.02], "n_ppm": [50],
        "rolling_finish_temp": [820], "cooling_rate_c_per_s": [18],
    })
    out = compute_features_for_class(df, "pipe_hsla")
    assert "cev_iiw" in out.columns
    assert "pcm" in out.columns
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v`
Expected: first 3 pass (profiles load), `test_synthetic_generator_en10083_qt_physical_sanity` fails with ImportError on `generate_synthetic_en10083_qt_dataset`.

- [ ] **Step 2: Implement generator in data_curator.py**

Open `app/backend/data_curator.py`. Find the existing `generate_synthetic_hsla_dataset` function. After it, add the new generator (exact code from spec §4.3):

```python
def generate_synthetic_en10083_qt_dataset(
    n_samples: int = 2000, random_seed: int = 42
) -> pd.DataFrame:
    """
    EN 10083-2 Q&T carbon steels (C22/C35/C45/C60).

    Empirical physical model:
    - HRC_quenched = 20 + 85*C + 3*ln(Mn+0.5) − 0.05*thickness_mm
    - HRC_tempered = HRC_quenched − 0.4*((temper_T−150)/10)*ln(1 + temper_t/30)
    - tensile_strength ≈ 34.5*HRC*10 МПа (empirical Rockwell→UTS relation)
    """
    rng = np.random.default_rng(random_seed)
    c  = rng.uniform(0.18, 0.65, n_samples)
    si = rng.uniform(0.15, 0.40, n_samples)
    mn = rng.uniform(0.40, 0.80, n_samples)
    p  = rng.uniform(0.0, 0.035, n_samples)
    s  = rng.uniform(0.0, 0.035, n_samples)
    cr = rng.uniform(0.0, 0.40, n_samples)
    austenit_T = rng.uniform(820.0, 900.0, n_samples)
    temper_T   = rng.uniform(150.0, 650.0, n_samples)
    temper_t   = rng.uniform(30.0, 180.0, n_samples)
    thick_mm   = rng.uniform(10.0, 100.0, n_samples)

    hrc_q = 20 + 85 * c + 3 * np.log(mn + 0.5) - 0.05 * thick_mm
    hrc_q = np.clip(hrc_q, 20, 65)
    temper_loss = ((temper_T - 150) / 10) * np.log1p(temper_t / 30)
    hrc = hrc_q - 0.4 * temper_loss
    hrc = np.clip(hrc + rng.normal(0, 1.5, n_samples), 15, 65)
    tensile = 34.5 * hrc * 10 + rng.normal(0, 50, n_samples)
    tensile = np.clip(tensile, 400, 1600)

    campaign_id = rng.integers(1, 50, n_samples)
    heat_date = pd.date_range("2024-01-01", periods=n_samples, freq="2h")

    return pd.DataFrame({
        "c_pct": c, "si_pct": si, "mn_pct": mn,
        "p_pct": p, "s_pct": s, "cr_pct": cr,
        "austenitizing_temp": austenit_T,
        "tempering_temp": temper_T,
        "tempering_time_min": temper_t,
        "section_thickness_mm": thick_mm,
        "hardness_hrc": hrc,
        "tensile_strength_mpa": tensile,
        "campaign_id": campaign_id,
        "heat_date": heat_date,
    })


def save_sample_dataset_en10083_qt(path: Path | None = None) -> Path:
    """Symmetric to save_sample_dataset() but for Q&T class."""
    path = path or (DATA_DIR / "hsla_en10083_qt_synthetic.parquet")
    df = generate_synthetic_en10083_qt_dataset()
    df.to_parquet(path, index=False)
    return path
```

Imports for `pd`, `np`, `Path`, `DATA_DIR` are already in the file — do not add new ones.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v
.venv/bin/ruff check app/backend/data_curator.py app/backend/steel_classes.py app/tests/test_steel_classes.py
```

Expected: all 6 tests pass; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add app/backend/data_curator.py app/tests/test_steel_classes.py
git commit -m "feat(data_curator): EN 10083-2 Q&T synthetic generator + class-routing tests"
```

---

## Task 3: Persist steel_class in TrainedModel + meta.json

**Files:**
- Modify: `app/backend/model_trainer.py`
- Modify: `app/tests/test_steel_classes.py`

- [ ] **Step 1: Add failing test**

Append to `app/tests/test_steel_classes.py`:

```python
def test_train_model_persists_steel_class(tmp_path, monkeypatch):
    """train_model stores steel_class in meta.json and TrainedModel."""
    from app.backend import model_trainer
    from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET

    monkeypatch.setattr(model_trainer, "MODELS_DIR", tmp_path)

    gen = get_synthetic_generator("pipe_hsla")
    df = gen(n_samples=500, random_seed=1)
    df_feat = compute_hsla_features(df)
    feat = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]

    trained = model_trainer.train_model(
        df_feat, "yield_strength_mpa", feat,
        n_optuna_trials=3, steel_class="pipe_hsla",
    )
    assert trained.steel_class == "pipe_hsla"
    assert trained.version.startswith("hsla_")

    # meta.json has it
    import json
    meta_path = tmp_path / trained.version / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["steel_class"] == "pipe_hsla"

    # load_model returns it
    bundle = model_trainer.load_model(trained.version)
    assert bundle["meta"]["steel_class"] == "pipe_hsla"


def test_train_model_en10083_qt_smoke(tmp_path, monkeypatch):
    """End-to-end Q&T training on synthetic data."""
    from app.backend import model_trainer
    from app.backend.steel_classes import load_steel_class

    monkeypatch.setattr(model_trainer, "MODELS_DIR", tmp_path)

    profile = load_steel_class("en10083_qt")
    gen = get_synthetic_generator("en10083_qt")
    df = gen(n_samples=1500, random_seed=7)

    trained = model_trainer.train_model(
        df, "hardness_hrc", profile.feature_set,
        n_optuna_trials=5, steel_class="en10083_qt",
    )
    assert trained.steel_class == "en10083_qt"
    assert trained.version.startswith("en10083qt_")
    assert trained.metrics.r2_test > 0.6    # synthetic model should be learnable
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py::test_train_model_persists_steel_class -v`
Expected: FAIL — `train_model` does not accept `steel_class`.

- [ ] **Step 2: Modify TrainedModel + train_model**

In `app/backend/model_trainer.py`:

1. Add `steel_class: str` to `TrainedModel` dataclass:

```python
@dataclass
class TrainedModel:
    version: str
    target: str
    feature_list: list[str]
    artifact_path: str
    metrics: TrainingMetrics
    feature_importance: dict[str, float]
    training_ranges: dict[str, list[float]]
    has_uncertainty: bool
    has_ood_detector: bool
    split_strategy: str
    cv_strategy: str
    steel_class: str = "pipe_hsla"    # NEW, default for backward compat
```

2. Modify `train_model` signature:

```python
def train_model(
    df_features: pd.DataFrame,
    target: str,
    feature_list: list[str],
    n_optuna_trials: int = 40,
    random_seed: int = 42,
    steel_class: str = "pipe_hsla",
) -> TrainedModel:
```

3. Inside `train_model`, change the version string — find the existing:

```python
    version = f"hsla_{target.replace('_mpa', '').replace('_j_cm2', '')}_xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

Replace with:

```python
    version_prefix = {"pipe_hsla": "hsla", "en10083_qt": "en10083qt"}.get(steel_class, "model")
    version = f"{version_prefix}_{target.replace('_mpa', '').replace('_j_cm2', '')}_xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

4. Inside `train_model`, where `meta.json` is written, add `steel_class` to the payload. Find:

```python
    with open(artifact_dir / "meta.json", "w") as f:
        json.dump({
            "version": version,
            "target": target,
            ...
```

Add `"steel_class": steel_class,` before `"trained_at"`.

5. Update the final `return TrainedModel(...)` to pass `steel_class=steel_class`.

6. In `load_model(version)`, no change needed — `meta.json` is read as-is; the `steel_class` key is passed through.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v
```

Expected: 8 passed (6 from Task 2 + 2 new). The `test_train_model_en10083_qt_smoke` test actually trains a tiny XGBoost — takes 5-15 seconds, fine.

- [ ] **Step 4: Commit**

```bash
git add app/backend/model_trainer.py app/tests/test_steel_classes.py
git commit -m "feat(model_trainer): persist steel_class in TrainedModel + meta.json; version prefix per class"
```

---

## Task 4: Pattern Library — per-class M05 + D07

**Files:**
- Modify: `pattern_library/patterns.py`
- Modify: `app/tests/test_steel_classes.py`

- [ ] **Step 1: Failing tests**

Append:

```python
def test_pattern_m05_uses_ctx_expected_features_hsla():
    from pattern_library.patterns import run_all_patterns, Phase
    ctx = {
        "steel_class": "pipe_hsla",
        "expected_top_features": [
            "c_pct", "mn_pct", "nb_pct", "ti_pct", "v_pct",
            "rolling_finish_temp", "cooling_rate_c_per_s",
            "cev_iiw", "pcm", "microalloying_sum",
        ],
        "feature_importance": {
            # No HSLA-relevant feature in top-5 — should trigger M05
            "cu_pct": 0.40, "s_pct": 0.20, "n_ppm": 0.15,
            "p_pct": 0.10, "al_pct": 0.05,
        },
    }
    warnings = run_all_patterns(ctx, phase=Phase.TRAINING)
    ids = {w["pattern_id"] for w in warnings}
    assert "M05" in ids


def test_pattern_m05_uses_ctx_expected_features_en10083():
    from pattern_library.patterns import run_all_patterns, Phase
    ctx = {
        "steel_class": "en10083_qt",
        "expected_top_features": [
            "c_pct", "tempering_temp", "austenitizing_temp",
            "mn_pct", "section_thickness_mm",
        ],
        # Good top-5 for Q&T — should NOT trigger M05
        "feature_importance": {
            "c_pct": 0.35, "tempering_temp": 0.25,
            "austenitizing_temp": 0.15, "mn_pct": 0.10,
            "section_thickness_mm": 0.08, "cr_pct": 0.05,
        },
    }
    warnings = run_all_patterns(ctx, phase=Phase.TRAINING)
    ids = {w["pattern_id"] for w in warnings}
    assert "M05" not in ids
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v -k "pattern_m05"`
Expected: first test may pass if existing `_check_m05` triggers on the HSLA-like ctx; second will FAIL because current `_check_m05` is hardcoded to HSLA expected-features and will trigger false-positively on the Q&T ctx whose top is `c_pct, tempering_temp, ...`.

- [ ] **Step 2: Update `_check_m05_feature_importance_sanity`**

In `pattern_library/patterns.py`, find `_check_m05_feature_importance_sanity`. Replace the body with:

```python
def _check_m05_feature_importance_sanity(ctx: dict) -> CheckResult:
    """
    For each steel class, ensure at least 2 of the top-5 features by importance
    match the expected set defined in its YAML profile. Falls back to HSLA
    expected-set for backward compat if no ctx["expected_top_features"] given.
    """
    importance = ctx.get("feature_importance", {})
    if not importance:
        return CheckResult(False)
    expected = set(ctx.get("expected_top_features") or [])
    if not expected:
        # Legacy HSLA-only fallback (for old callers that pass steel_class="pipe_hsla"
        # without the richer ctx).
        if ctx.get("steel_class", "") != "pipe_hsla":
            return CheckResult(False)
        expected = {
            "c_pct", "mn_pct", "nb_pct", "ti_pct", "v_pct",
            "rolling_finish_temp", "cooling_rate_c_per_s",
            "cev_iiw", "pcm", "cen",
        }

    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    top_names = {f[0] for f in top_features}
    overlap = top_names & expected
    if len(overlap) < 2:
        return CheckResult(
            True,
            message=(
                f"Top-5 feature importance не содержит ожидаемых для класса "
                f"{ctx.get('steel_class', '?')}. "
                f"Top: {[f[0] for f in top_features]}. "
                f"Ожидалось минимум 2 из: {sorted(expected)}."
            ),
        )
    return CheckResult(False)
```

- [ ] **Step 3: Update `_check_d07_physical_bounds`**

Find `_check_d07_physical_bounds`. At the top, add:

```python
def _check_d07_physical_bounds(ctx: dict) -> CheckResult:
    df = ctx.get("dataframe")
    if df is None:
        return CheckResult(False)
    violations = []
    bounds = ctx.get("physical_bounds") or {
        "c_pct": (0.02, 2.1),
        "mn_pct": (0.0, 20.0),
        "si_pct": (0.0, 5.0),
        "p_pct": (0.0, 0.15),
        "s_pct": (0.0, 0.15),
        "yield_strength_mpa": (100, 3000),
        "tensile_strength_mpa": (150, 3500),
        "elongation_pct": (0, 85),
    }
    for col, bound in bounds.items():
        lo, hi = (bound[0], bound[1])
        if col in df.columns:
            out_of_bounds = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_bounds > 0:
                violations.append(f"{col}: {out_of_bounds} значений вне [{lo}, {hi}]")
    if violations:
        return CheckResult(
            True,
            message="Обнаружены значения вне физических границ:\n" + "\n".join(violations),
        )
    return CheckResult(False)
```

The only change: `bounds = ctx.get("physical_bounds") or { … }` instead of the hard-coded dict at the top.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py -v
PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"
```

Expected: all pass; previously HSLA pattern tests continue to work via fallback.

- [ ] **Step 5: Commit**

```bash
git add pattern_library/patterns.py app/tests/test_steel_classes.py
git commit -m "feat(patterns): M05 + D07 read expected features / bounds from context (per class)"
```

---

## Task 5: engine.py — per-class context in training review

**Files:**
- Modify: `app/backend/engine.py`
- Modify: `app/tests/test_steel_classes.py`

- [ ] **Step 1: Failing test**

Append:

```python
def test_engine_critic_context_gets_per_class_bounds_and_features(tmp_path, monkeypatch):
    """_build_critic_context populates expected_top_features + physical_bounds per class."""
    from app.backend.engine import Orchestrator, PipelineState, AgentResult, Critic

    state = PipelineState(user_request={"task_type": "train"})
    state.features["training_ranges"] = {}
    agent_result = AgentResult(
        agent_name="model_trainer", success=True,
        output={
            "version": "en10083qt_hardness_hrc_xgb_test",
            "steel_class": "en10083_qt",
            "split_strategy": "time_based",
            "cv_strategy": "group_kfold",
            "has_uncertainty": True,
            "has_ood_detector": True,
            "feature_importance": {
                "c_pct": 0.3, "tempering_temp": 0.2,
                "austenitizing_temp": 0.15, "mn_pct": 0.1,
                "section_thickness_mm": 0.08,
            },
        },
    )
    orch = Orchestrator(agents={}, critic=Critic(use_llm=False))
    ctx = orch._build_critic_context("training", state, agent_result)

    assert ctx["steel_class"] == "en10083_qt"
    assert "c_pct" in ctx["expected_top_features"]
    assert "tempering_temp" in ctx["expected_top_features"]
    assert ctx["physical_bounds"]["c_pct"] == [0.18, 0.65]
```

Run: `PYTHONPATH=. .venv/bin/pytest app/tests/test_steel_classes.py::test_engine_critic_context_gets_per_class_bounds_and_features -v`
Expected: FAIL — engine currently doesn't enrich ctx with per-class data.

- [ ] **Step 2: Modify `_build_critic_context` in engine.py**

Find method `_build_critic_context` in `app/backend/engine.py`. In the `if phase == "training":` branch, add enrichment BEFORE the existing `ctx.update(...)`. Replace the training branch with:

```python
        if phase == "training":
            steel_class_id = result.output.get("steel_class", "pipe_hsla")
            try:
                from app.backend.steel_classes import load_steel_class
                profile = load_steel_class(steel_class_id)
                ctx["steel_class"] = steel_class_id
                ctx["expected_top_features"] = profile.expected_top_features
                ctx["physical_bounds"] = profile.physical_bounds
            except Exception:
                ctx["steel_class"] = steel_class_id
            ctx.update({
                "has_time_column": state.dataset.get("has_time_column", True),
                "has_groups": state.dataset.get("has_groups", True),
                "split_strategy": result.output.get("split_strategy", "unknown"),
                "cv_strategy": result.output.get("cv_strategy", "unknown"),
                "prediction_has_ci": result.output.get("has_uncertainty", False),
                "ood_detector_configured": result.output.get("has_ood_detector", False),
            })
```

(Existing code had `steel_class: "pipe_hsla"` hardcoded — we now take it from agent output.)

- [ ] **Step 3: Update ModelTrainerAgent to pass steel_class**

In `app/backend/model_trainer.py`, find `class ModelTrainerAgent`. In `run()`, the `AgentResult.output` dict must include `"steel_class": trained.steel_class`. Also, `task.get("steel_class", "pipe_hsla")` should be passed into `train_model()`. Add to the existing `output={...}`:

```python
                output={
                    "version": trained.version,
                    "artifact_path": trained.artifact_path,
                    "target": target,
                    "feature_list": feature_list,
                    "steel_class": trained.steel_class,   # NEW
                    ...
```

And the call to `train_model(...)`:

```python
                trained = train_model(
                    df, target=target, feature_list=feature_list,
                    n_optuna_trials=task.get("n_optuna_trials", 40),
                    steel_class=task.get("steel_class", "pipe_hsla"),
                )
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"
```

Expected: all pass.

Also run smoke test without key:

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```

Expected: SMOKE TEST PASSED (HSLA path, no regressions).

- [ ] **Step 5: Commit**

```bash
git add app/backend/engine.py app/backend/model_trainer.py app/tests/test_steel_classes.py
git commit -m "feat(engine): route steel_class through Critic context + ModelTrainerAgent"
```

---

## Task 6: UI — class dropdown, sidebar badge, predict/design conditional

**Files:**
- Modify: `app/frontend/app.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Sidebar badge**

In `app/frontend/app.py`, find the sidebar block where `selected_model` is chosen. After the `st.sidebar.selectbox("Активная модель", …)` line, add:

```python
# Class badge for active model
if selected_model:
    try:
        import json as _json
        _meta_path = PROJECT_ROOT / "models" / selected_model / "meta.json"
        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
        _class_id = _meta.get("steel_class", "pipe_hsla")
        _class_label = {
            "pipe_hsla": "🔩 Pipe HSLA",
            "en10083_qt": "🔨 EN 10083 Q&T",
        }.get(_class_id, _class_id)
        st.sidebar.caption(f"Класс: **{_class_label}**")
        _meta_target = _meta.get("target", "?")
        st.sidebar.caption(f"Target: `{_meta_target}`")
    except Exception:
        pass
```

- [ ] **Step 2: Training tab — class dropdown and filtered target**

Find `with tab_train:`. Replace the existing top of the block (the current `target_col` and `n_trials` columns) with:

```python
with tab_train:
    st.header("Обучение модели")
    st.caption("Обучает XGBoost с quantile regression для uncertainty estimation")

    from app.backend.steel_classes import (
        available_steel_classes, get_synthetic_generator,
        compute_features_for_class, load_steel_class,
    )

    _classes = available_steel_classes()
    _class_opts = {c.id: f"{c.name} ({c.standard})" for c in _classes}
    selected_class_id = st.selectbox(
        "Класс стали",
        options=[c.id for c in _classes],
        format_func=lambda cid: _class_opts[cid],
        key="train_class",
    )
    _profile = load_steel_class(selected_class_id)

    c1, c2 = st.columns(2)
    target_col = c1.selectbox(
        "Target property",
        options=[t.id for t in _profile.target_properties],
        format_func=lambda tid: next(
            t.label for t in _profile.target_properties if t.id == tid
        ),
    )
    n_trials = c2.slider(
        "Optuna trials (чем больше, тем лучше, но медленнее)", 10, 150, 40,
    )

    st.info(
        f"ℹ️ Выбран класс: **{_profile.name}** · стандарт {_profile.standard}. "
        f"Feature set: {len(_profile.feature_set)} колонок. "
        f"Обучение займёт 1-5 минут в зависимости от количества trials."
    )
```

Then replace the actual training button block. Find `if st.button("🤖 Обучить модель", type="primary"):` and update the body:

```python
    if st.button("🤖 Обучить модель", type="primary"):
        with st.spinner("Generating dataset & training..."):
            from app.backend.model_trainer import train_model
            gen = get_synthetic_generator(_profile.synthetic_generator_name)
            df_raw = gen()

            # For Q&T no derived features; for HSLA adds CEV/Pcm/etc.
            df_feat = compute_features_for_class(df_raw, selected_class_id)
            feature_list = [f for f in _profile.feature_set if f in df_feat.columns]

            progress = st.progress(0, text="Запускаю обучение...")
            trained = train_model(
                df_feat, target_col, feature_list,
                n_optuna_trials=n_trials,
                steel_class=selected_class_id,
            )
            progress.progress(100, text="Готово!")

            st.success(f"✅ Модель {trained.version} готова (класс: {_profile.name})")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² test", f"{trained.metrics.r2_test:.3f}")
            m2.metric("MAE test", f"{trained.metrics.mae_test:.2f}")
            m3.metric("R² train", f"{trained.metrics.r2_train:.3f}")
            m4.metric("Coverage 90% CI", f"{trained.metrics.coverage_90_ci:.1%}")

            # Critic — Pattern Library now gets per-class context
            from pattern_library.patterns import run_all_patterns, Phase
            critic_ctx = {
                "r2_train": trained.metrics.r2_train,
                "r2_val": trained.metrics.r2_val,
                "r2_test": trained.metrics.r2_test,
                "mae_test": trained.metrics.mae_test,
                "rmse_test": trained.metrics.rmse_test,
                "coverage_90_ci": trained.metrics.coverage_90_ci,
                "n_train": trained.metrics.n_train,
                "n_val": trained.metrics.n_val,
                "n_test": trained.metrics.n_test,
                "prediction_has_ci": True,
                "has_time_column": True,
                "has_groups": True,
                "split_strategy": "time_based",
                "cv_strategy": "group_kfold",
                "feature_importance": trained.feature_importance,
                "training_ranges": trained.training_ranges,
                "steel_class": selected_class_id,
                "expected_top_features": _profile.expected_top_features,
                "physical_bounds": _profile.physical_bounds,
                "ood_detector_configured": True,
                "target": target_col,
            }
            warnings = run_all_patterns(critic_ctx, phase=Phase.TRAINING)
            if warnings:
                st.subheader("⚠️ Отчёт Critic")
                for w in warnings:
                    sev = w["severity"]
                    msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                    if sev == "HIGH":
                        st.error(msg)
                    elif sev == "MEDIUM":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("✓ Critic не нашёл проблем")

            # LLM-Critic (Claude Sonnet 4.6) — only runs with ANTHROPIC_API_KEY
            from app.backend.critic_llm import make_llm_critic
            from dataclasses import asdict
            _llm = make_llm_critic()
            if _llm is not None:
                with st.spinner("🤖 LLM-Critic проверяет..."):
                    llm_obs = _llm.review_training(critic_ctx)
                    st.session_state["llm_observations"] = [
                        asdict(o) for o in llm_obs
                    ]

            llm_obs_rendered = st.session_state.get("llm_observations", [])
            if llm_obs_rendered:
                st.subheader("🤖 LLM-Critic (Claude Sonnet 4.6)")
                for o in llm_obs_rendered:
                    sev = o["severity"]
                    msg = (f"**[{sev}] {o['category']}:** {o['message']}\n\n"
                           f"💡 {o['rationale']}")
                    if sev == "HIGH":
                        st.error(msg)
                    elif sev == "MEDIUM":
                        st.warning(msg)
                    else:
                        st.info(msg)
            elif _llm is not None:
                st.caption("🤖 LLM-Critic: проблем не обнаружено")

            # Feature importance chart
            st.subheader("Feature importance")
            imp_df = pd.DataFrame(
                sorted(trained.feature_importance.items(), key=lambda x: -x[1])[:15],
                columns=["feature", "importance"],
            )
            st.bar_chart(imp_df.set_index("feature"))
```

- [ ] **Step 3: Predict tab — dynamic fields per active model**

Find `with tab_predict:`. The current tab assumes HSLA feature set (16 fields). Replace the body with:

```python
with tab_predict:
    st.header("Прогноз для заданного состава")
    st.caption("Введите химию и режим — получите прогноз с uncertainty")

    if not selected_model:
        st.warning("Сначала обучите модель")
    else:
        import json as _json
        from app.backend.model_trainer import load_model, predict_with_uncertainty

        _meta_path = PROJECT_ROOT / "models" / selected_model / "meta.json"
        _meta = _json.loads(_meta_path.read_text())
        _class_id = _meta.get("steel_class", "pipe_hsla")
        _profile_p = load_steel_class(_class_id)

        st.caption(f"Класс: **{_profile_p.name}** · target: `{_meta['target']}`")

        # Render inputs based on feature_set and bounds
        row = {}
        cols_per_row = 4
        features_ui = [
            f for f in _profile_p.feature_set
            if f != "n_ppm"
        ]
        for chunk_start in range(0, len(features_ui), cols_per_row):
            chunk = features_ui[chunk_start:chunk_start + cols_per_row]
            cc = st.columns(len(chunk))
            for col_idx, feat in enumerate(chunk):
                lo, hi = _profile_p.physical_bounds.get(feat, (0.0, 1.0))
                default = (lo + hi) / 2
                step = (hi - lo) / 100 if (hi - lo) > 0 else 0.01
                row[feat] = cc[col_idx].number_input(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=float(default), step=float(step),
                    key=f"pred_{feat}",
                    format="%.4f" if feat.endswith("_pct") else "%.2f",
                )
        # Optional n_ppm for HSLA
        if "n_ppm" in _profile_p.feature_set:
            row["n_ppm"] = st.number_input(
                "n_ppm", 20.0, 100.0, 55.0, step=5.0, key="pred_n_ppm",
            )

        if st.button("🔮 Предсказать", type="primary"):
            from app.backend.steel_classes import compute_features_for_class
            df_input = pd.DataFrame([row])
            df_feat = compute_features_for_class(df_input, _class_id)

            bundle = load_model(selected_model)
            pred = predict_with_uncertainty(bundle, df_feat)

            mean = float(pred["prediction"].iloc[0])
            lo = float(pred["lower_90"].iloc[0])
            hi = float(pred["upper_90"].iloc[0])
            ood = bool(pred["ood_flag"].iloc[0])

            _tgt_label = next(
                (t.label for t in _profile_p.target_properties
                 if t.id == _meta["target"]),
                _meta["target"],
            )
            st.subheader(f"{_tgt_label}: **{mean:.1f}** ± {(hi - lo) / 2:.1f}")
            st.caption(f"90% ДИ: [{lo:.1f}, {hi:.1f}]")

            if ood:
                st.error("⚠️ Состав вне training distribution — прогноз ненадёжен!")

            # Derived params only for HSLA
            if _class_id == "pipe_hsla" and {
                "cev_iiw", "pcm", "cen", "microalloying_sum"
            }.issubset(df_feat.columns):
                st.markdown("**Производные параметры:**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CEV(IIW)", f"{df_feat['cev_iiw'].iloc[0]:.3f}")
                c2.metric("Pcm", f"{df_feat['pcm'].iloc[0]:.3f}")
                c3.metric("CEN", f"{df_feat['cen'].iloc[0]:.3f}")
                c4.metric("Микролегирование", f"{df_feat['microalloying_sum'].iloc[0]:.4f}")
```

- [ ] **Step 4: Design tab — banner for Q&T**

Find `with tab_design:`. At the very top of the block (right after `st.header/st.caption`), add:

```python
    # Check active model class — inverse design is HSLA-only in this iteration
    _design_class_id = "pipe_hsla"
    if selected_model:
        try:
            import json as _json
            _meta_path = PROJECT_ROOT / "models" / selected_model / "meta.json"
            _design_class_id = _json.loads(_meta_path.read_text()).get(
                "steel_class", "pipe_hsla"
            )
        except Exception:
            pass

    if _design_class_id == "en10083_qt":
        st.info(
            "ℹ️ Inverse design пока работает только для **Pipe HSLA**. "
            "Для класса EN 10083-2 Q&T используйте вкладку «📊 Прогноз». "
            "Поддержка inverse design для Q&T запланирована на v2."
        )
        st.stop()
```

`st.stop()` halts rendering of the rest of the tab for this run — users on Q&T see only the banner.

- [ ] **Step 5: Update CLAUDE.md**

Find the «### Target feature set» section. Replace it with a broader paragraph:

```markdown
### Target feature set / multi-class

Классы стали описаны YAML-профилями в `data/steel_classes/<id>.yaml` (`id, name, standard, feature_set, physical_bounds, target_properties, expected_top_features, synthetic_generator_name, process_params`). Реестр — `app/backend/steel_classes.py:AVAILABLE_CLASS_IDS`; сейчас: `pipe_hsla` (API 5L) и `en10083_qt` (EN 10083-2 Q&T carbon steels). **Номенклатура европейская** — советских марок нет.

Класс — атрибут каждой обученной модели: `TrainedModel.steel_class` сохраняется в `models/<version>/meta.json`. Downstream-UI (prediction, design) читает активный класс из meta и ведёт себя соответственно: поля ввода в «Прогноз» подстраиваются под `feature_set`, target-label читается из профиля, вкладка «Дизайн» показывает banner для Q&T (inverse design остаётся HSLA-only в этой итерации).

Physical bounds и expected top-features для Critic проверок `D07` / `M05` читаются из профиля через `_build_critic_context` (fallback на HSLA-константы для старых моделей без `meta["steel_class"]`). Synthetic-генераторы живут в `data_curator.py` (`generate_synthetic_hsla_dataset`, `generate_synthetic_en10083_qt_dataset`) и регистрируются в `steel_classes.get_synthetic_generator(name)`.
```

- [ ] **Step 6: Verify Streamlit loads**

```bash
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
sleep 5
curl -sI http://localhost:8501 | head -1
lsof -ti:8501 2>/dev/null | xargs kill 2>/dev/null
```

Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 7: Full test + smoke**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -q -m "not integration"
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```

Expected: all tests pass; smoke test passes (HSLA path).

- [ ] **Step 8: Commit**

```bash
git add app/frontend/app.py CLAUDE.md
git commit -m "feat(ui): class dropdown in train, sidebar badge, predict/design conditional per class"
```

---

## Task 7: Final verification + v0.4 tag

**Files:** None (verification only).

- [ ] **Step 1: Full pytest incl. integration**

```bash
PYTHONPATH=. .venv/bin/pytest app/tests/ -v
```

Expected: 35 (previous) + ~8-10 new test_steel_classes = ~45 tests pass.

- [ ] **Step 2: Smoke test without and with the API key**

```bash
unset ANTHROPIC_API_KEY
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```
Expected: SMOKE TEST PASSED.

- [ ] **Step 3: Streamlit 200 OK**

```bash
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2
PYTHONPATH=. .venv/bin/streamlit run app/frontend/app.py --server.port 8501 --server.headless true &
sleep 5
curl -sI http://localhost:8501 | head -1
```
Expected: `HTTP/1.1 200 OK`. Leave running for manual inspection by the user.

- [ ] **Step 4: Tag v0.4**

```bash
git log --oneline v0.3-llm-critic..HEAD | head -15
git tag -a v0.4-multi-class -m "MVP v0.4: multi-class (pipe HSLA + EN 10083-2 Q&T)"
git tag -l | tail -5
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §4.1 YAML profiles | 1 |
| §4.2 steel_classes.py | 1 |
| §4.3 EN 10083 generator | 2 |
| §4.4 train_model + meta | 3 |
| §4.5 engine context | 5 |
| §4.6 Patterns M05/D07 | 4 |
| §4.7 UI | 6 |
| §6 Acceptance criteria 1-8 | 2, 3, 5, 6, 7 |

All spec sections covered.

**2. Placeholder scan**

No TBDs / "similar to task N" / vague instructions. All steps are concrete commands or code blocks.

**3. Type consistency**

- `SteelClassProfile` fields — same across Tasks 1, 2, 3, 4, 5, 6.
- `TrainedModel.steel_class: str` — declared Task 3, used Tasks 5, 6.
- `ctx["expected_top_features"]` / `ctx["physical_bounds"]` / `ctx["steel_class"]` — same keys in Tasks 4, 5, 6.
- `get_synthetic_generator(name)` — same mapping in Tasks 1 (scaffold), 6 (UI use).
- `compute_features_for_class(df, class_id)` — same signature in Tasks 1 (impl), 2 (test), 6 (UI use).

No drift.

**Plan complete.** Execution handoff happens via subagent-driven-development.
