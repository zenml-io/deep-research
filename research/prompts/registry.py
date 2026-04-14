"""Prompt registry: scans and loads all .md prompts at import time."""

from __future__ import annotations

from pathlib import Path

from .loader import PromptRecord, load_prompt

_PROMPTS_DIR = Path(__file__).parent

# Populated at import time — one PromptRecord per .md file.
PROMPTS: dict[str, PromptRecord] = {}

for _md_file in sorted(_PROMPTS_DIR.glob("*.md")):
    _record = load_prompt(_md_file)
    PROMPTS[_record.name] = _record


def get_prompt(name: str) -> PromptRecord:
    """Return the PromptRecord for *name*, or raise KeyError."""
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt {name!r}. Available: {sorted(PROMPTS)}"
        ) from None


def get_prompt_hashes() -> dict[str, str]:
    """Return ``{name: sha256}`` for every loaded prompt."""
    return {name: record.sha256 for name, record in PROMPTS.items()}
