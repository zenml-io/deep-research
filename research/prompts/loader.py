"""Prompt loader: reads .md files, parses optional YAML front-matter, computes SHA256."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Matches YAML front-matter block: starts with ---, ends with ---, captures content between.
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Extracts 'version: <value>' from front-matter content.
_VERSION_RE = re.compile(r"^version:\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """Immutable record for a loaded prompt file."""

    name: str
    text: str
    sha256: str
    version: str | None


def load_prompt(filepath: Path) -> PromptRecord:
    """Load a prompt .md file and return a PromptRecord.

    Reads the file, parses optional YAML front-matter (delimited by ``---``)
    for a ``version`` field, computes SHA256 of the **full** file content,
    and returns a :class:`PromptRecord` with the body text (after front-matter).
    """
    raw = filepath.read_text(encoding="utf-8")
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    version: str | None = None
    text = raw

    fm_match = _FRONT_MATTER_RE.match(raw)
    if fm_match:
        front_matter = fm_match.group(1)
        ver_match = _VERSION_RE.search(front_matter)
        if ver_match:
            version = ver_match.group(1).strip()
        # Strip front-matter from text
        text = raw[fm_match.end() :]

    return PromptRecord(
        name=filepath.stem,
        text=text,
        sha256=sha,
        version=version,
    )
