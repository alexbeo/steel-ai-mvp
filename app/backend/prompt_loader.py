"""Load LLM system prompts from the gitignored prompts/ directory.

Prompts are stored as Markdown files outside Python source because they
constitute project know-how (see prompts/README.md). On a fresh clone
the files are absent — load_prompt raises PromptNotFoundError with
clear instructions instead of silently failing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


class PromptNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise PromptNotFoundError(
            f"Prompt '{name}' not found at {path.relative_to(PROJECT_ROOT)}. "
            "See prompts/README.md — prompts are gitignored intellectual "
            "property; obtain from project owner."
        )
    return path.read_text(encoding="utf-8")


def load_prompt_optional(name: str) -> str | None:
    """Same as load_prompt, but returns None when the file is absent.

    Used at module-import time so backend modules don't crash on a
    fresh clone (e.g. public Streamlit Cloud demo). The corresponding
    make_*() factory must check for None and degrade gracefully.
    """
    try:
        return load_prompt(name)
    except PromptNotFoundError:
        return None


def reset_cache() -> None:
    """Clear the lru_cache — useful when editing prompts during dev."""
    load_prompt.cache_clear()
