"""
Slag-aware advisory для Al-раскисления BOF tap → ladle.

Расширяет термодинамическую модель из app/backend/deoxidation.py учётом:
- O связанного в FeO/MnO/SiO₂ шлака переноса (slag-aware O-balance)
- co-deoxidation by Si (FeSi/SiMn до Al)
- выбора метода подачи Al (catalog в data/deox_methods/al_addition_methods.yaml)

Этот модуль — pure physics + cost. Без LLM, без ML.
Phase: deoxidation (LF-tap, не LF-equilibrium — отличается от deoxidation.py).

Design: docs/superpowers/specs/2026-05-12_asis-slag-aware-deox.md
PR 1 scope: только YAML catalog + dataclass AdditionMethod + load_addition_methods().
Compute-функции (compute_o_from_slag, compute_al_demand_slag_aware,
compare_addition_methods, recommend_optimal_method) — в PR 2-4.
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
    # Dry-run demo: показать каталог
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
