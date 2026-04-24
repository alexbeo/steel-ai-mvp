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
AVAILABLE_CLASS_IDS = ["pipe_hsla", "en10083_qt", "fatigue_carbon_steel"]


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
    target_o_activity_ppm: float | None = None
    data_source: str | None = None
    data_source_doi: str | None = None

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
        target_o_activity_ppm=data.get("target_o_activity_ppm"),
        data_source=data.get("data_source"),
        data_source_doi=data.get("data_source_doi"),
    )
    _PROFILE_CACHE[class_id] = profile
    return profile


def available_steel_classes() -> list[SteelClassProfile]:
    return [load_steel_class(cid) for cid in AVAILABLE_CLASS_IDS]


def get_synthetic_generator(generator_name: str) -> Callable:
    """Lazy import to avoid circular deps with data_curator.

    Returns a zero-arg callable yielding a DataFrame. Name is historical —
    `fatigue_carbon_steel_real` returns REAL Agrawal NIMS data, not synthetic.
    """
    from app.backend.data_curator import (
        generate_synthetic_en10083_qt_dataset,
        generate_synthetic_hsla_dataset,
        load_real_agrawal_fatigue_dataset,
    )
    return {
        "pipe_hsla": generate_synthetic_hsla_dataset,
        "en10083_qt": generate_synthetic_en10083_qt_dataset,
        "fatigue_carbon_steel_real": load_real_agrawal_fatigue_dataset,
    }[generator_name]


def compute_features_for_class(df, class_id: str):
    profile = load_steel_class(class_id)
    if profile.feature_engineering == "compute_hsla_features":
        from app.backend.feature_eng import compute_hsla_features
        return compute_hsla_features(df)
    return df
