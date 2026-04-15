"""Tests for V2 council flow orchestration.

Uses stub injection for kitaru (@flow, @checkpoint, wait) and
monkeypatching of checkpoint/flow functions on the council module.
Each test is self-contained — no shared conftest.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from research.config.budget import BudgetConfig
from research.config.settings import ResearchConfig
from research.config.slots import ModelSlotConfig
from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.package import (
    CouncilComparison,
    CouncilPackage,
    InvestigationPackage,
    RunMetadata,
)
from research.contracts.plan import ResearchPlan
from research.contracts.reports import DraftReport, FinalReport


# ---------------------------------------------------------------------------
# Stub infrastructure (matches test_v2_flow.py pattern)
# ---------------------------------------------------------------------------

_COUNCIL_MODULES = [
    "research.flows.council",
    "research.flows.deep_research",
    "research.checkpoints.judge",
    "research.checkpoints.metadata",
    "research.checkpoints.scope",
    "research.checkpoints.plan",
    "research.checkpoints.supervisor",
    "research.checkpoints.subagent",
    "research.checkpoints.draft",
    "research.checkpoints.critique",
    "research.checkpoints.finalize",
    "research.checkpoints.assemble",
    "research.checkpoints",
    "research.agents._wrap",
    "research.agents",
    "research.agents.scope",
    "research.agents.planner",
    "research.agents.supervisor",
    "research.agents.subagent",
    "research.agents.generator",
    "research.agents.reviewer",
    "research.agents.finalizer",
    "research.agents.judge",
]


def _install_stubs(monkeypatch):
    """Install kitaru + pydantic_ai stubs for flow tests.

    @flow -> passthrough decorator.
    @checkpoint -> passthrough; also gives each function a .submit() method.
    wait() -> no-op.
    """

    def flow_decorator(fn=None, **kwargs):
        if fn is not None:
            return fn

        def decorator(f):
            return f

        return decorator

    def checkpoint_decorator(fn=None, *, type=None):
        """Checkpoint stub that adds .submit() support."""

        def _add_submit(f):
            def submit(*args, **kw):
                result = f(*args, **kw)
                return types.SimpleNamespace(load=lambda: result)

            f.submit = submit
            f._checkpoint_type = type
            return f

        if fn is not None:
            return _add_submit(fn)

        def decorator(f):
            return _add_submit(f)

        return decorator

    def wait(timeout=None):
        pass

    class FakeAgent:
        def __init__(self, model_name="test", **kwargs):
            pass

    def wrap(agent, **kwargs):
        return agent

    kp_ns = types.SimpleNamespace(wrap=wrap)
    kitaru_mod = types.SimpleNamespace(
        flow=flow_decorator,
        checkpoint=checkpoint_decorator,
        wait=wait,
        adapters=types.SimpleNamespace(pydantic_ai=kp_ns),
    )

    monkeypatch.setitem(sys.modules, "kitaru", kitaru_mod)
    monkeypatch.setitem(
        sys.modules,
        "kitaru.adapters",
        types.SimpleNamespace(pydantic_ai=kp_ns),
    )
    monkeypatch.setitem(sys.modules, "kitaru.adapters.pydantic_ai", kp_ns)
    monkeypatch.setitem(
        sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent)
    )


def _clear_modules():
    for name in _COUNCIL_MODULES:
        sys.modules.pop(name, None)


def _load_council_module(monkeypatch):
    """Install stubs, clear caches, and return the council module."""
    _install_stubs(monkeypatch)
    _clear_modules()
    return importlib.import_module("research.flows.council")


def _load_judge_checkpoint_module(monkeypatch):
    """Install stubs, clear caches, and return the judge checkpoint module."""
    _install_stubs(monkeypatch)
    _clear_modules()
    return importlib.import_module("research.checkpoints.judge")


def make_checkpoint_stub(return_value):
    """Create a stub that works as both direct call and .submit().load()."""

    def stub(*args, **kwargs):
        return return_value

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: return_value)
    return stub


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_council_config(*, include_judge=True):
    """Build a minimal ResearchConfig for council testing."""
    slots = {
        "generator": ModelSlotConfig(provider="anthropic", model="gen"),
        "subagent": ModelSlotConfig(provider="google-gla", model="sub"),
        "reviewer": ModelSlotConfig(provider="openai", model="rev"),
    }
    if include_judge:
        slots["judge"] = ModelSlotConfig(provider="google-gla", model="judge-model")
    return ResearchConfig(
        tier="standard",
        budget=BudgetConfig(soft_budget_usd=1.0),
        slots=slots,
        max_iterations=2,
        max_parallel_subagents=3,
        ledger_window_iterations=3,
        grounding_min_ratio=0.0,
        max_supplemental_loops=1,
        wait_timeout_seconds=3600,
        allow_unfinalized_package=False,
        strict_unknown_model_cost=False,
        sandbox_enabled=False,
        sandbox_backend=None,
        enabled_providers=["arxiv"],
    )


_METADATA_A = RunMetadata(
    run_id="run-a",
    tier="standard",
    started_at="2024-01-01T00:00:00Z",
    total_iterations=1,
    total_cost_usd=0.05,
)
_METADATA_B = RunMetadata(
    run_id="run-b",
    tier="standard",
    started_at="2024-01-01T00:01:00Z",
    total_iterations=2,
    total_cost_usd=0.08,
)

_BRIEF = ResearchBrief(topic="test", raw_request="q")
_PLAN = ResearchPlan(goal="investigate", key_questions=["what?"])
_LEDGER = EvidenceLedger(items=[])
_DRAFT_A = DraftReport(content="# Report A", sections=["A"])
_DRAFT_B = DraftReport(content="# Report B", sections=["B"])
_FINAL_A = FinalReport(content="# Final A", sections=["A"])
_FINAL_B = FinalReport(content="# Final B", sections=["B"])

_PACKAGE_A = InvestigationPackage(
    metadata=_METADATA_A,
    brief=_BRIEF,
    plan=_PLAN,
    ledger=_LEDGER,
    draft=_DRAFT_A,
    final_report=_FINAL_A,
)
_PACKAGE_B = InvestigationPackage(
    metadata=_METADATA_B,
    brief=_BRIEF,
    plan=_PLAN,
    ledger=_LEDGER,
    draft=_DRAFT_B,
    final_report=_FINAL_B,
)

_COMPARISON = CouncilComparison(
    comparison="Generator B produced a more thorough report.",
    generator_scores={"gen_a": 0.7, "gen_b": 0.9},
    recommended_generator="gen_b",
)


# ---------------------------------------------------------------------------
# Judge checkpoint tests
# ---------------------------------------------------------------------------


class TestJudgeCheckpointType:
    """Verify the judge checkpoint has the right type annotation."""

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod = _load_judge_checkpoint_module(monkeypatch)
        assert hasattr(mod.run_judge, "_checkpoint_type")
        assert mod.run_judge._checkpoint_type == "llm_call"


class TestJudgePromptConstruction:
    """Verify run_judge builds a reasonable prompt from package data."""

    def test_serializes_package_data_for_agent(self, monkeypatch):
        mod = _load_judge_checkpoint_module(monkeypatch)

        captured_prompts = []

        class FakeRunResult:
            def __init__(self, output):
                self.output = output

        class FakeWrappedAgent:
            def run_sync(self, prompt, **kw):
                captured_prompts.append(prompt)
                return FakeRunResult(_COMPARISON)

        # Monkeypatch build_judge_agent to return our fake
        monkeypatch.setattr(
            mod, "build_judge_agent", lambda model_name: FakeWrappedAgent()
        )

        packages = {"gen_a": _PACKAGE_A, "gen_b": _PACKAGE_B}
        result = mod.run_judge(packages, "test:judge")

        assert isinstance(result, CouncilComparison)
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Prompt should contain generator names and report content
        assert "gen_a" in prompt
        assert "gen_b" in prompt
        assert "# Final A" in prompt
        assert "# Final B" in prompt

    def test_uses_draft_when_no_final_report(self, monkeypatch):
        """Falls back to draft content when final_report is None."""
        mod = _load_judge_checkpoint_module(monkeypatch)

        captured_prompts = []

        class FakeRunResult:
            def __init__(self, output):
                self.output = output

        class FakeWrappedAgent:
            def run_sync(self, prompt, **kw):
                captured_prompts.append(prompt)
                return FakeRunResult(_COMPARISON)

        monkeypatch.setattr(
            mod, "build_judge_agent", lambda model_name: FakeWrappedAgent()
        )

        pkg_no_final = InvestigationPackage(
            metadata=_METADATA_A,
            brief=_BRIEF,
            plan=_PLAN,
            ledger=_LEDGER,
            draft=_DRAFT_A,
            final_report=None,
        )
        packages = {"gen_a": pkg_no_final}
        mod.run_judge(packages, "test:judge")

        assert "# Report A" in captured_prompts[0]

    def test_no_report_placeholder(self, monkeypatch):
        """Uses placeholder when both draft and final_report are None."""
        mod = _load_judge_checkpoint_module(monkeypatch)

        captured_prompts = []

        class FakeRunResult:
            def __init__(self, output):
                self.output = output

        class FakeWrappedAgent:
            def run_sync(self, prompt, **kw):
                captured_prompts.append(prompt)
                return FakeRunResult(_COMPARISON)

        monkeypatch.setattr(
            mod, "build_judge_agent", lambda model_name: FakeWrappedAgent()
        )

        pkg_empty = InvestigationPackage(
            metadata=_METADATA_A,
            brief=_BRIEF,
            plan=_PLAN,
            ledger=_LEDGER,
            draft=None,
            final_report=None,
        )
        packages = {"gen_a": pkg_empty}
        mod.run_judge(packages, "test:judge")

        assert "(no report produced)" in captured_prompts[0]


# ---------------------------------------------------------------------------
# Provider compromise detection tests
# ---------------------------------------------------------------------------


class TestProviderCompromise:
    """Verify _detect_provider_compromise helper."""

    def test_compromise_when_judge_matches_generator(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }
        judge_slot = ModelSlotConfig(provider="anthropic", model="judge")

        assert mod._detect_provider_compromise(gen_slots, judge_slot) is True

    def test_no_compromise_when_providers_differ(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }
        judge_slot = ModelSlotConfig(provider="google-gla", model="judge")

        assert mod._detect_provider_compromise(gen_slots, judge_slot) is False


# ---------------------------------------------------------------------------
# Council flow tests
# ---------------------------------------------------------------------------


class TestCouncilRunsBothGenerators:
    """Council flow calls deep_research once per generator."""

    def test_calls_deep_research_for_each_generator(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        dr_calls = []

        def mock_deep_research(question, tier="standard", config=None):
            # Track which generator slot was used
            gen_slot = config.slots["generator"] if config else None
            dr_calls.append(gen_slot)
            if gen_slot and gen_slot.model == "model_a":
                return _PACKAGE_A
            return _PACKAGE_B

        mock_deep_research.submit = lambda *a, **kw: types.SimpleNamespace(
            load=lambda: _PACKAGE_A
        )

        monkeypatch.setattr(mod, "deep_research", mock_deep_research)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="model_a"),
            "gen_b": ModelSlotConfig(provider="openai", model="model_b"),
        }

        result = mod.council_research(
            "test question", config=cfg, generator_slots=gen_slots
        )

        assert len(dr_calls) == 2
        assert isinstance(result, CouncilPackage)


class TestCouncilPreservesPackages:
    """Both investigation packages appear in the council output."""

    def test_both_packages_in_output(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        call_index = [0]

        def mock_deep_research(question, tier="standard", config=None):
            call_index[0] += 1
            return _PACKAGE_A if call_index[0] == 1 else _PACKAGE_B

        monkeypatch.setattr(mod, "deep_research", mock_deep_research)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="model_a"),
            "gen_b": ModelSlotConfig(provider="openai", model="model_b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert "gen_a" in result.packages
        assert "gen_b" in result.packages
        assert result.packages["gen_a"].metadata.run_id == "run-a"
        assert result.packages["gen_b"].metadata.run_id == "run-b"


class TestCouncilSetsCanonicalFromJudge:
    """canonical_generator matches the judge's recommendation."""

    def test_canonical_from_recommendation(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)

        comparison = CouncilComparison(
            comparison="B is better",
            generator_scores={"gen_a": 0.6, "gen_b": 0.95},
            recommended_generator="gen_b",
        )
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(comparison))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert result.canonical_generator == "gen_b"

    def test_canonical_none_when_judge_has_no_recommendation(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)

        comparison = CouncilComparison(
            comparison="Tied",
            generator_scores={"gen_a": 0.8, "gen_b": 0.8},
            recommended_generator=None,
        )
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(comparison))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert result.canonical_generator is None


class TestCouncilProviderCompromiseRecorded:
    """Provider compromise flag is correctly recorded in output."""

    def test_compromise_detected_and_recorded(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        # Judge on anthropic, same as gen_a
        cfg = _make_council_config()
        cfg = cfg.model_copy(
            update={
                "slots": {
                    **cfg.slots,
                    "judge": ModelSlotConfig(provider="anthropic", model="judge"),
                }
            }
        )
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert result.council_provider_compromise is True

    def test_no_compromise_when_clean(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()  # judge is google-gla
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert result.council_provider_compromise is False


class TestCouncilRequiresJudgeSlot:
    """Council flow raises CouncilConfigError without a judge slot."""

    def test_raises_without_judge(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)

        cfg = _make_council_config(include_judge=False)
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        with pytest.raises(mod.CouncilConfigError, match="judge"):
            mod.council_research("test", config=cfg, generator_slots=gen_slots)


class TestCouncilSingleGeneratorWarning:
    """Council with < 2 generators logs a warning but proceeds."""

    def test_single_generator_proceeds(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        gen_slots = {
            "gen_only": ModelSlotConfig(provider="anthropic", model="a"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        # Should still produce a CouncilPackage
        assert isinstance(result, CouncilPackage)
        assert "gen_only" in result.packages
        assert len(result.packages) == 1

    def test_no_generator_slots_defaults_to_config_generator(self, monkeypatch):
        """When generator_slots=None, uses the config's generator slot."""
        mod = _load_council_module(monkeypatch)

        dr_calls = []

        def mock_deep_research(question, tier="standard", config=None):
            dr_calls.append(config)
            return _PACKAGE_A

        monkeypatch.setattr(mod, "deep_research", mock_deep_research)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        result = mod.council_research("test", config=cfg, generator_slots=None)

        assert isinstance(result, CouncilPackage)
        assert len(dr_calls) == 1
        assert "generator_a" in result.packages


class TestCouncilPassesConfigToGenerators:
    """Each generator gets a config with its specific generator slot."""

    def test_generator_slot_swapped_per_run(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        received_configs = {}

        def mock_deep_research(question, tier="standard", config=None):
            gen_model = config.slots["generator"].model if config else "unknown"
            received_configs[gen_model] = config
            return _PACKAGE_A

        monkeypatch.setattr(mod, "deep_research", mock_deep_research)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="model_alpha"),
            "gen_b": ModelSlotConfig(provider="openai", model="model_beta"),
        }

        mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert "model_alpha" in received_configs
        assert "model_beta" in received_configs
        # Other slots should be unchanged
        assert (
            received_configs["model_alpha"].slots["reviewer"].model
            == cfg.slots["reviewer"].model
        )
        assert (
            received_configs["model_beta"].slots["subagent"].model
            == cfg.slots["subagent"].model
        )


class TestCouncilSchemaVersion:
    """CouncilPackage has the expected schema version."""

    def test_schema_version_default(self, monkeypatch):
        mod = _load_council_module(monkeypatch)

        monkeypatch.setattr(mod, "deep_research", lambda q, **kw: _PACKAGE_A)
        monkeypatch.setattr(mod, "run_judge", make_checkpoint_stub(_COMPARISON))

        cfg = _make_council_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = mod.council_research("test", config=cfg, generator_slots=gen_slots)

        assert result.schema_version == "1.0"
