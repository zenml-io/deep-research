"""Guard tests for V2 architectural invariants.

Fast, deterministic, no LLM calls, no network access.
Protects structural properties of the V2 architecture.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. Provider Topology Tests
# ---------------------------------------------------------------------------


class TestProviderTopology:
    """Every tier's default mapping must include provider-crossing checkpoints."""

    def test_standard_tier_critique_crosses_providers(self) -> None:
        """Generator and reviewer must use different providers (critique crosses)."""
        from research.config.defaults import TIER_DEFAULTS
        from research.config.slots import ModelSlot

        tier = TIER_DEFAULTS["standard"]
        gen_provider = tier.slots[ModelSlot.generator].provider
        rev_provider = tier.slots[ModelSlot.reviewer].provider
        assert gen_provider != rev_provider, (
            f"standard tier generator ({gen_provider}) and reviewer ({rev_provider}) "
            "must use different providers for provider-crossing critique"
        )

    def test_quick_tier_critique_crosses_providers(self) -> None:
        """Quick tier also crosses providers between generator and reviewer."""
        from research.config.defaults import TIER_DEFAULTS
        from research.config.slots import ModelSlot

        tier = TIER_DEFAULTS["quick"]
        gen_provider = tier.slots[ModelSlot.generator].provider
        rev_provider = tier.slots[ModelSlot.reviewer].provider
        assert gen_provider != rev_provider, (
            f"quick tier generator ({gen_provider}) and reviewer ({rev_provider}) "
            "must use different providers"
        )

    def test_deep_tier_critique_crosses_providers(self) -> None:
        """Deep tier crosses providers between generator and reviewer."""
        from research.config.defaults import TIER_DEFAULTS
        from research.config.slots import ModelSlot

        tier = TIER_DEFAULTS["deep"]
        gen_provider = tier.slots[ModelSlot.generator].provider
        rev_provider = tier.slots[ModelSlot.reviewer].provider
        assert gen_provider != rev_provider, (
            f"deep tier generator ({gen_provider}) and reviewer ({rev_provider}) "
            "must use different providers"
        )

    def test_deep_tier_judge_differs_from_generator(self) -> None:
        """Deep tier's judge provider differs from at least one generator provider."""
        from research.config.defaults import TIER_DEFAULTS
        from research.config.slots import ModelSlot

        tier = TIER_DEFAULTS["deep"]
        gen_provider = tier.slots[ModelSlot.generator].provider
        judge_provider = tier.slots[ModelSlot.judge].provider
        assert gen_provider != judge_provider, (
            f"deep tier judge ({judge_provider}) must differ from "
            f"generator ({gen_provider}) for council independence"
        )

    def test_all_tiers_have_at_least_one_provider_crossing(self) -> None:
        """Every tier must have at least one provider-crossing checkpoint."""
        from research.config.defaults import TIER_DEFAULTS
        from research.config.slots import ModelSlot

        for tier_name, tier in TIER_DEFAULTS.items():
            providers = {slot.value: cfg.provider for slot, cfg in tier.slots.items()}
            unique_providers = set(providers.values())
            assert len(unique_providers) >= 2, (
                f"{tier_name} tier uses only {unique_providers} — "
                "needs at least 2 providers for cross-validation"
            )


# ---------------------------------------------------------------------------
# 2. assemble_package Purity Test
# ---------------------------------------------------------------------------


def _stub_kitaru_and_load_assemble():
    """Load assemble.py with kitaru stubbed — no real checkpoint decorator."""

    # Stub kitaru checkpoint as passthrough
    def checkpoint_decorator(fn=None, *, type=None):
        if fn is not None:
            return fn

        def decorator(f):
            return f

        return decorator

    kitaru_mod = types.ModuleType("kitaru")
    kitaru_mod.checkpoint = checkpoint_decorator  # type: ignore[attr-defined]

    saved_kitaru = sys.modules.get("kitaru")
    sys.modules["kitaru"] = kitaru_mod

    # Clear cached modules that depend on kitaru
    modules_to_clear = [k for k in sys.modules if k.startswith("research.checkpoints")]
    saved_modules = {k: sys.modules.pop(k) for k in modules_to_clear}

    try:
        mod = importlib.import_module("research.checkpoints.assemble")
        importlib.reload(mod)
        return mod
    finally:
        # Restore original state
        for k, v in saved_modules.items():
            sys.modules[k] = v
        if saved_kitaru is not None:
            sys.modules["kitaru"] = saved_kitaru
        elif "kitaru" in sys.modules:
            del sys.modules["kitaru"]


class TestAssemblePackagePurity:
    """assemble_package must make ZERO LLM calls — it's pure computation."""

    def test_assemble_makes_no_llm_calls(self) -> None:
        """Patch Agent.run and Agent.run_sync to blow up if called."""
        assemble_mod = _stub_kitaru_and_load_assemble()

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import FinalReport

        metadata = RunMetadata(
            run_id="guard-test",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
        )
        brief = ResearchBrief(topic="test", raw_request="test question")
        plan = ResearchPlan(goal="test", key_questions=["what?"])
        item1 = EvidenceItem(
            evidence_id="ev-001",
            title="Source 1",
            url="https://example.com/1",
            synthesis="snippet 1",
            iteration_added=0,
        )
        item2 = EvidenceItem(
            evidence_id="ev-002",
            title="Source 2",
            url="https://example.com/2",
            synthesis="snippet 2",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item1, item2])

        # Report with citations matching evidence IDs — high grounding density
        final_report = FinalReport(
            content=(
                "This is a substantive sentence with evidence [ev-001]. "
                "Another substantive sentence with source [ev-002]. "
                "A third point referencing evidence [ev-001]."
            ),
        )

        # Patch all LLM call paths — should never be reached
        with (
            patch(
                "pydantic_ai.Agent.run",
                side_effect=RuntimeError("LLM call attempted via Agent.run"),
            ),
            patch(
                "pydantic_ai.Agent.run_sync",
                side_effect=RuntimeError("LLM call attempted via Agent.run_sync"),
            ),
        ):
            package = assemble_mod.assemble_package(
                metadata=metadata,
                brief=brief,
                plan=plan,
                ledger=ledger,
                iterations=[],
                draft=None,
                critique=None,
                final_report=final_report,
            )

        assert package.metadata.run_id == "guard-test"
        assert package.brief.topic == "test"
        assert len(package.ledger.items) == 2


# ---------------------------------------------------------------------------
# 3. Grounding Density Tests
# ---------------------------------------------------------------------------


class TestGroundingDensity:
    """Grounding density gate in assembly must reject ungrounded reports."""

    def test_ungrounded_report_fails_assembly(self) -> None:
        """A report with NO citations must fail grounding check."""
        assemble_mod = _stub_kitaru_and_load_assemble()

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import FinalReport

        metadata = RunMetadata(
            run_id="grounding-fail",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
        )
        brief = ResearchBrief(topic="test", raw_request="test question")
        plan = ResearchPlan(goal="test", key_questions=["what?"])
        item1 = EvidenceItem(
            evidence_id="ev-001",
            title="Source 1",
            url="https://example.com/1",
            synthesis="snippet 1",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item1])

        # Report with NO citations — grounding density = 0.0
        bad_report = FinalReport(
            content=(
                "This is a long enough sentence without any citations whatsoever. "
                "Another sentence that makes claims without evidence. "
                "Yet another ungrounded statement that says nothing useful."
            ),
        )

        with pytest.raises(assemble_mod.GroundingError):
            assemble_mod.assemble_package(
                metadata=metadata,
                brief=brief,
                plan=plan,
                ledger=ledger,
                iterations=[],
                draft=None,
                critique=None,
                final_report=bad_report,
                grounding_min_ratio=0.7,
                strict_grounding=True,
            )

    def test_well_grounded_report_passes_assembly(self) -> None:
        """A report with high citation density passes grounding check."""
        assemble_mod = _stub_kitaru_and_load_assemble()

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import FinalReport

        metadata = RunMetadata(
            run_id="grounding-pass",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
        )
        brief = ResearchBrief(topic="test", raw_request="test question")
        plan = ResearchPlan(goal="test", key_questions=["what?"])
        item1 = EvidenceItem(
            evidence_id="ev-001",
            title="Source 1",
            url="https://example.com/1",
            synthesis="snippet 1",
            iteration_added=0,
        )
        item2 = EvidenceItem(
            evidence_id="ev-002",
            title="Source 2",
            url="https://example.com/2",
            synthesis="snippet 2",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item1, item2])

        good_report = FinalReport(
            content=(
                "This is a substantive sentence with evidence [ev-001]. "
                "Another substantive sentence with source [ev-002]. "
                "A third point referencing evidence [ev-001]."
            ),
        )

        package = assemble_mod.assemble_package(
            metadata=metadata,
            brief=brief,
            plan=plan,
            ledger=ledger,
            iterations=[],
            draft=None,
            critique=None,
            final_report=good_report,
            grounding_min_ratio=0.7,
        )
        assert package.final_report is not None
        assert package.metadata.grounding_density is not None
        assert package.metadata.grounding_density >= 0.7

    def test_ungrounded_report_warns_in_soft_mode(self) -> None:
        """In soft mode (default), low density produces a package with density recorded."""
        assemble_mod = _stub_kitaru_and_load_assemble()

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import FinalReport

        metadata = RunMetadata(
            run_id="soft-mode",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
        )
        brief = ResearchBrief(topic="test", raw_request="test question")
        plan = ResearchPlan(goal="test", key_questions=["what?"])
        item1 = EvidenceItem(
            evidence_id="ev-001",
            title="Source 1",
            url="https://example.com/1",
            synthesis="snippet 1",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item1])

        # Report with NO citations — grounding density = 0.0
        bad_report = FinalReport(
            content=(
                "This is a long enough sentence without any citations whatsoever. "
                "Another sentence that makes claims without evidence. "
                "Yet another ungrounded statement that says nothing useful."
            ),
        )

        # strict_grounding=False (default) — should NOT raise
        package = assemble_mod.assemble_package(
            metadata=metadata,
            brief=brief,
            plan=plan,
            ledger=ledger,
            iterations=[],
            draft=None,
            critique=None,
            final_report=bad_report,
            grounding_min_ratio=0.7,
        )
        assert package.final_report is not None
        assert package.metadata.grounding_density is not None
        assert package.metadata.grounding_density < 0.7

    def test_unresolved_citation_fails_assembly(self) -> None:
        """Citations referencing non-existent evidence IDs must fail."""
        assemble_mod = _stub_kitaru_and_load_assemble()

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import FinalReport

        metadata = RunMetadata(
            run_id="citation-fail",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
        )
        brief = ResearchBrief(topic="test", raw_request="test question")
        plan = ResearchPlan(goal="test", key_questions=["what?"])
        item1 = EvidenceItem(
            evidence_id="ev-001",
            title="Source 1",
            url="https://example.com/1",
            synthesis="snippet 1",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item1])

        # Report cites ev-999 which doesn't exist in ledger
        report = FinalReport(
            content="A substantive sentence citing [ev-999] which does not exist.",
        )

        with pytest.raises(assemble_mod.CitationResolutionError):
            assemble_mod.assemble_package(
                metadata=metadata,
                brief=brief,
                plan=plan,
                ledger=ledger,
                iterations=[],
                draft=None,
                critique=None,
                final_report=report,
            )


# ---------------------------------------------------------------------------
# 4. Ledger Dedup Tests
# ---------------------------------------------------------------------------


class TestLedgerDedup:
    """DOI, arXiv ID, and canonical URL duplicates collapse to one entry."""

    def test_same_doi_produces_same_dedup_key(self) -> None:
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import compute_dedup_key

        item_a = EvidenceItem(
            evidence_id="a",
            title="Paper A",
            url="https://doi.org/10.1234/test",
            synthesis="content",
            doi="10.1234/test",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Paper B",
            url="https://other.com",
            synthesis="content",
            doi="10.1234/test",
            iteration_added=0,
        )
        assert compute_dedup_key(item_a) == compute_dedup_key(item_b)

    def test_same_arxiv_id_produces_same_dedup_key(self) -> None:
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import compute_dedup_key

        item_a = EvidenceItem(
            evidence_id="a",
            title="Paper A",
            url="https://arxiv.org/abs/2301.12345v1",
            synthesis="content",
            arxiv_id="2301.12345v1",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Paper B",
            url="https://arxiv.org/abs/2301.12345v2",
            synthesis="content",
            arxiv_id="2301.12345v2",
            iteration_added=0,
        )
        assert compute_dedup_key(item_a) == compute_dedup_key(item_b)

    def test_same_canonical_url_produces_same_dedup_key(self) -> None:
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import compute_dedup_key

        item_a = EvidenceItem(
            evidence_id="a",
            title="Page A",
            url="https://example.com/page?utm_source=twitter",
            canonical_url="https://example.com/page",
            synthesis="content",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Page B",
            url="https://example.com/page?ref=other",
            canonical_url="https://example.com/page",
            synthesis="content",
            iteration_added=0,
        )
        assert compute_dedup_key(item_a) == compute_dedup_key(item_b)

    def test_no_stable_id_returns_none(self) -> None:
        """Items with no DOI, arXiv ID, or canonical URL get None key."""
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import compute_dedup_key

        item = EvidenceItem(
            evidence_id="a",
            title="Blog post",
            synthesis="content",
            iteration_added=0,
        )
        assert compute_dedup_key(item) is None

    def test_is_duplicate_detects_doi_match(self) -> None:
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import compute_dedup_key, is_duplicate

        item_a = EvidenceItem(
            evidence_id="a",
            title="Paper A",
            synthesis="content",
            doi="10.1234/test",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Paper B",
            synthesis="content",
            doi="10.1234/test",
            iteration_added=0,
        )
        key_a = compute_dedup_key(item_a)
        assert key_a is not None
        existing = {key_a}
        assert is_duplicate(existing, item_b) is True

    def test_is_duplicate_no_stable_id_never_duplicate(self) -> None:
        """Items with no stable identifier are never considered duplicates."""
        from research.contracts.evidence import EvidenceItem
        from research.ledger.dedup import is_duplicate

        item = EvidenceItem(
            evidence_id="a",
            title="Blog post",
            synthesis="content",
            iteration_added=0,
        )
        existing = {"some_key"}
        assert is_duplicate(existing, item) is False

    def test_managed_ledger_collapses_doi_duplicates(self) -> None:
        """ManagedLedger.append with same DOI collapses to one entry."""
        from research.contracts.evidence import EvidenceItem
        from research.ledger.ledger import ManagedLedger

        ml = ManagedLedger()
        item_a = EvidenceItem(
            evidence_id="a",
            title="Paper A",
            synthesis="content A",
            doi="10.1234/test",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Paper B",
            synthesis="content B",
            doi="10.1234/test",
            iteration_added=0,
        )

        added_first = ml.append([item_a])
        added_second = ml.append([item_b])

        assert len(added_first) == 1
        assert len(added_second) == 0  # duplicate collapsed
        assert ml.size == 1

    def test_managed_ledger_collapses_arxiv_duplicates(self) -> None:
        """ManagedLedger.append with same arXiv ID collapses to one entry."""
        from research.contracts.evidence import EvidenceItem
        from research.ledger.ledger import ManagedLedger

        ml = ManagedLedger()
        item_a = EvidenceItem(
            evidence_id="a",
            title="Paper A",
            synthesis="content A",
            arxiv_id="2301.12345v1",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Paper B",
            synthesis="content B",
            arxiv_id="2301.12345v2",
            iteration_added=0,
        )

        ml.append([item_a])
        ml.append([item_b])
        assert ml.size == 1

    def test_managed_ledger_admits_items_without_stable_ids(self) -> None:
        """Items with no stable ID are always admitted, even if similar."""
        from research.contracts.evidence import EvidenceItem
        from research.ledger.ledger import ManagedLedger

        ml = ManagedLedger()
        item_a = EvidenceItem(
            evidence_id="a",
            title="Blog A",
            synthesis="content A",
            iteration_added=0,
        )
        item_b = EvidenceItem(
            evidence_id="b",
            title="Blog B",
            synthesis="content B",
            iteration_added=0,
        )

        ml.append([item_a])
        ml.append([item_b])
        assert ml.size == 2  # both admitted — no stable ID to dedup on


# ---------------------------------------------------------------------------
# 5. run_v2.py Size Threshold Tests
# ---------------------------------------------------------------------------


RUN_V2_PATH = Path(__file__).parent.parent / "run_v2.py"


class TestRunV2SizeThreshold:
    """run_v2.py must stay thin — no embedded reusable helpers."""

    def test_run_v2_under_120_lines(self) -> None:
        lines = RUN_V2_PATH.read_text().splitlines()
        assert len(lines) < 120, f"run_v2.py has {len(lines)} lines, exceeds 120"

    def test_run_v2_only_main_function(self) -> None:
        tree = ast.parse(RUN_V2_PATH.read_text())
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert func_names == ["main"], f"Unexpected functions: {func_names}"

    def test_run_v2_no_classes(self) -> None:
        tree = ast.parse(RUN_V2_PATH.read_text())
        class_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        assert class_names == [], f"Unexpected classes: {class_names}"
