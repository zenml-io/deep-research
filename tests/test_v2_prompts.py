"""Tests for the V2 prompt loader and registry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.prompts import PROMPTS, PromptRecord, get_prompt, get_prompt_hashes
from research.prompts.loader import load_prompt

# The 8 expected prompt names
EXPECTED_PROMPTS = frozenset(
    {
        "scope",
        "planner",
        "supervisor",
        "subagent",
        "generator",
        "reviewer",
        "finalizer",
        "council_judge",
    }
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "research" / "prompts"


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    """Tests for load_prompt()."""

    def test_all_prompt_files_load(self):
        """Every expected .md file loads without error."""
        for name in EXPECTED_PROMPTS:
            filepath = PROMPTS_DIR / f"{name}.md"
            assert filepath.exists(), f"Missing prompt file: {filepath}"
            record = load_prompt(filepath)
            assert isinstance(record, PromptRecord)

    def test_prompt_record_has_required_fields(self):
        """Each PromptRecord has .text, .sha256, .version."""
        filepath = PROMPTS_DIR / "scope.md"
        record = load_prompt(filepath)
        assert isinstance(record.text, str)
        assert len(record.text) > 0
        assert isinstance(record.sha256, str)
        assert len(record.sha256) == 64  # SHA256 hex digest length
        assert record.version is not None  # scope.md has front-matter

    def test_sha256_stable_for_same_content(self):
        """Loading the same file twice produces identical SHA256."""
        filepath = PROMPTS_DIR / "scope.md"
        r1 = load_prompt(filepath)
        r2 = load_prompt(filepath)
        assert r1.sha256 == r2.sha256

    def test_sha256_matches_raw_content(self):
        """SHA256 is computed over the full file content (including front-matter)."""
        filepath = PROMPTS_DIR / "scope.md"
        raw = filepath.read_text(encoding="utf-8")
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        record = load_prompt(filepath)
        assert record.sha256 == expected

    def test_version_parsed_from_front_matter(self):
        """Version field is extracted from YAML front-matter."""
        filepath = PROMPTS_DIR / "scope.md"
        record = load_prompt(filepath)
        assert record.version == "0.1.0"

    def test_prompt_text_excludes_front_matter(self):
        """Prompt text does not include the --- delimiters or front-matter."""
        filepath = PROMPTS_DIR / "scope.md"
        record = load_prompt(filepath)
        assert "---" not in record.text
        assert "version:" not in record.text
        assert record.text.startswith("You are a research scoping agent")

    def test_name_is_file_stem(self):
        """PromptRecord.name is the filename stem (no extension)."""
        filepath = PROMPTS_DIR / "planner.md"
        record = load_prompt(filepath)
        assert record.name == "planner"

    def test_no_front_matter_gives_none_version(self, tmp_path: Path):
        """A file without front-matter has version=None and full text preserved."""
        md = tmp_path / "bare.md"
        md.write_text("Just plain text.\n")
        record = load_prompt(md)
        assert record.version is None
        assert record.text == "Just plain text.\n"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the PROMPTS registry and accessor functions."""

    def test_all_prompts_loaded(self):
        """All 8 expected prompt files are present in the registry."""
        for name in EXPECTED_PROMPTS:
            assert name in PROMPTS, f"Missing prompt in registry: {name}"

    def test_no_duplicate_names(self):
        """Registry keys are unique (enforced by dict, but verify count)."""
        md_files = list(PROMPTS_DIR.glob("*.md"))
        # Number of .md files should match number of entries in PROMPTS
        assert len(md_files) == len(PROMPTS)

    def test_get_prompt_returns_correct_record(self):
        """get_prompt() returns the expected PromptRecord."""
        record = get_prompt("scope")
        assert isinstance(record, PromptRecord)
        assert record.name == "scope"
        assert "research scoping agent" in record.text

    def test_get_prompt_raises_keyerror_for_unknown(self):
        """get_prompt() raises KeyError for an unknown prompt name."""
        with pytest.raises(KeyError, match="Unknown prompt"):
            get_prompt("nonexistent_prompt_xyz")

    def test_get_prompt_hashes_returns_all(self):
        """get_prompt_hashes() returns a dict with all prompt names."""
        hashes = get_prompt_hashes()
        assert isinstance(hashes, dict)
        for name in EXPECTED_PROMPTS:
            assert name in hashes
            assert isinstance(hashes[name], str)
            assert len(hashes[name]) == 64

    def test_get_prompt_hashes_values_match_records(self):
        """Hash values from get_prompt_hashes() match individual records."""
        hashes = get_prompt_hashes()
        for name, sha in hashes.items():
            assert PROMPTS[name].sha256 == sha

    def test_all_records_have_version(self):
        """All placeholder prompts should have version from front-matter."""
        for name in EXPECTED_PROMPTS:
            record = PROMPTS[name]
            assert record.version is not None, f"Prompt {name!r} has no version"
            assert record.version == "0.1.0"

    def test_all_records_text_is_nonempty(self):
        """Every prompt has non-empty text content."""
        for name, record in PROMPTS.items():
            assert len(record.text.strip()) > 0, f"Prompt {name!r} has empty text"
