"""Tests for V2 checkpoint functions (metadata, scope, plan).

Uses stub injection because kitaru cannot be directly imported in the
test environment. The pattern mirrors test_v2_agents.py but adapts the
checkpoint decorator stub to support both bare ``@checkpoint`` and
parameterized ``@checkpoint(type="llm_call")`` forms.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _clear_modules(*names: str) -> None:
    """Remove named modules so the next import gets a fresh copy."""
    for name in list(names):
        sys.modules.pop(name, None)


def _install_checkpoint_stubs(monkeypatch):
    """Install kitaru + pydantic_ai stubs for checkpoint tests.

    The checkpoint decorator stub supports both forms:
    - ``@checkpoint`` (bare) — tags with ``_checkpoint_type = None``
    - ``@checkpoint(type="llm_call")`` — tags with ``_checkpoint_type = "llm_call"``

    Returns the FakeAgent class for tests that need to mock agent behavior.
    """

    def checkpoint_decorator(fn=None, *, type=None):
        if fn is not None:
            # @checkpoint (bare decorator)
            fn._checkpoint_type = None
            return fn

        # @checkpoint(type="llm_call")
        def decorator(f):
            f._checkpoint_type = type
            return f

        return decorator

    class FakeAgent:
        """Minimal stand-in for ``pydantic_ai.Agent``."""

        def __init__(self, model_name: str = "test", **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    class FakeCapturePolicy:
        def __init__(self, *, tool_capture=None):
            self.tool_capture = tool_capture

    class FakeKitaruAgent:
        """Passthrough stub — returns the agent unchanged for checkpoint tests.

        Delegates attribute access to the wrapped agent so that
        agent.run_sync() works transparently.
        """

        def __init__(self, agent, *, name=None, capture=None):
            self._wrapped = agent

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    kp_ns = types.SimpleNamespace(
        KitaruAgent=FakeKitaruAgent, CapturePolicy=FakeCapturePolicy
    )
    kitaru_mod = types.SimpleNamespace(
        checkpoint=checkpoint_decorator,
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

    return FakeAgent


# ---------------------------------------------------------------------------
# Metadata checkpoint tests
# ---------------------------------------------------------------------------


class TestMetadataCheckpoints:
    """Unit tests for ``research.checkpoints.metadata``."""

    def _load(self, monkeypatch):
        _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.metadata",
            "research.checkpoints",
            "research.agents",
        )
        return importlib.import_module("research.checkpoints.metadata")

    def test_stamp_run_metadata_returns_run_stamp(self, monkeypatch):
        mod = self._load(monkeypatch)
        result = mod.stamp_run_metadata()
        assert isinstance(result, mod.RunStamp)
        assert result.run_id.startswith("run-")
        assert len(result.run_id) > 10  # run-<uuid>

    def test_stamp_run_metadata_has_started_at(self, monkeypatch):
        mod = self._load(monkeypatch)
        result = mod.stamp_run_metadata()
        assert result.started_at.endswith("Z")

    def test_stamp_produces_unique_ids(self, monkeypatch):
        mod = self._load(monkeypatch)
        r1 = mod.stamp_run_metadata()
        r2 = mod.stamp_run_metadata()
        assert r1.run_id != r2.run_id

    def test_snapshot_wall_clock_returns_snapshot(self, monkeypatch):
        mod = self._load(monkeypatch)
        stamp = mod.stamp_run_metadata()
        result = mod.snapshot_wall_clock(stamp.started_at)
        assert isinstance(result, mod.WallClockSnapshot)
        assert result.elapsed_seconds >= 0
        assert result.now_iso.endswith("Z")

    def test_finalize_returns_finalization(self, monkeypatch):
        mod = self._load(monkeypatch)
        stamp = mod.stamp_run_metadata()
        result = mod.finalize_run_metadata(stamp.started_at)
        assert isinstance(result, mod.RunFinalization)
        assert result.elapsed_seconds >= 0
        assert result.completed_at.endswith("Z")

    def test_checkpoint_decorator_applied_to_stamp(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod.stamp_run_metadata, "_checkpoint_type")
        assert mod.stamp_run_metadata._checkpoint_type is None  # bare @checkpoint

    def test_checkpoint_decorator_applied_to_snapshot(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod.snapshot_wall_clock, "_checkpoint_type")
        assert mod.snapshot_wall_clock._checkpoint_type is None

    def test_checkpoint_decorator_applied_to_finalize(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod.finalize_run_metadata, "_checkpoint_type")
        assert mod.finalize_run_metadata._checkpoint_type is None


# ---------------------------------------------------------------------------
# Scope checkpoint tests
# ---------------------------------------------------------------------------


class TestScopeCheckpoint:
    """Unit tests for ``research.checkpoints.scope``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.scope",
            "research.checkpoints",
            "research.agents",
            "research.agents.scope",
        )
        mod = importlib.import_module("research.checkpoints.scope")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_scope._checkpoint_type == "llm_call"

    def test_run_scope_calls_agent(self, monkeypatch):
        """run_scope calls the scope agent with the raw request."""
        mod, FakeAgent = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief

        expected = ResearchBrief(
            topic="RLHF alternatives",
            raw_request="What are RLHF alternatives?",
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock_agent = MockAgent()
        monkeypatch.setattr(mod, "build_scope_agent", lambda model: mock_agent)

        result = mod.run_scope("What are RLHF alternatives?", "test-model")
        assert result is expected
        assert mock_agent.last_prompt == "What are RLHF alternatives?"

    def test_run_scope_passes_model_to_builder(self, monkeypatch):
        """run_scope passes the model_name argument to build_scope_agent."""
        mod, FakeAgent = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief

        captured_model = []

        class MockResult:
            output = ResearchBrief(topic="test", raw_request="test")

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        def fake_build(model_name):
            captured_model.append(model_name)
            return MockAgent()

        monkeypatch.setattr(mod, "build_scope_agent", fake_build)
        mod.run_scope("test", "google-gla:gemini-2.5-flash")
        assert captured_model == ["google-gla:gemini-2.5-flash"]


# ---------------------------------------------------------------------------
# Plan checkpoint tests
# ---------------------------------------------------------------------------


class TestPlanCheckpoint:
    """Unit tests for ``research.checkpoints.plan``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.plan",
            "research.checkpoints",
            "research.agents",
            "research.agents.planner",
        )
        mod = importlib.import_module("research.checkpoints.plan")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_plan._checkpoint_type == "llm_call"

    def test_run_plan_calls_agent_with_brief_json(self, monkeypatch):
        """run_plan serializes the brief to JSON and passes to planner agent."""
        import json

        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan

        mod, FakeAgent = self._load(monkeypatch)

        brief = ResearchBrief(topic="RLHF", raw_request="RLHF alternatives?")
        expected_plan = ResearchPlan(goal="Investigate RLHF", key_questions=["What?"])

        class MockResult:
            output = expected_plan

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock_agent = MockAgent()
        monkeypatch.setattr(mod, "build_planner_agent", lambda model: mock_agent)

        result = mod.run_plan(brief, "test-model")
        assert result is expected_plan
        # Verify the prompt is JSON-serialized brief
        parsed = json.loads(mock_agent.last_prompt)
        assert parsed["topic"] == "RLHF"

    def test_run_plan_passes_model_to_builder(self, monkeypatch):
        """run_plan passes the model_name argument to build_planner_agent."""
        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan

        mod, FakeAgent = self._load(monkeypatch)

        brief = ResearchBrief(topic="test", raw_request="test")
        captured_model = []

        class MockResult:
            output = ResearchPlan(goal="test", key_questions=["q"])

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        def fake_build(model_name):
            captured_model.append(model_name)
            return MockAgent()

        monkeypatch.setattr(mod, "build_planner_agent", fake_build)
        mod.run_plan(brief, "anthropic:claude-sonnet-4-20250514")
        assert captured_model == ["anthropic:claude-sonnet-4-20250514"]

    def test_run_plan_json_contains_all_brief_fields(self, monkeypatch):
        """run_plan serializes the complete brief including optional fields."""
        import json

        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan

        mod, FakeAgent = self._load(monkeypatch)

        brief = ResearchBrief(
            topic="RLHF",
            raw_request="RLHF alternatives?",
            audience="ML practitioners",
            scope="2024 onwards",
            source_preferences=["arxiv", "peer-reviewed"],
        )

        class MockResult:
            output = ResearchPlan(goal="test", key_questions=["q"])

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock_agent = MockAgent()
        monkeypatch.setattr(mod, "build_planner_agent", lambda model: mock_agent)

        mod.run_plan(brief, "test-model")
        parsed = json.loads(mock_agent.last_prompt)
        assert parsed["topic"] == "RLHF"
        assert parsed["audience"] == "ML practitioners"
        assert parsed["scope"] == "2024 onwards"
        assert parsed["source_preferences"] == ["arxiv", "peer-reviewed"]


# ---------------------------------------------------------------------------
# __init__.py re-export tests
# ---------------------------------------------------------------------------


class TestCheckpointsInit:
    """Verify ``research.checkpoints`` re-exports all public names."""

    def _load(self, monkeypatch):
        _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints",
            "research.checkpoints.metadata",
            "research.checkpoints.scope",
            "research.checkpoints.plan",
            "research.checkpoints.supervisor",
            "research.checkpoints.subagent",
            "research.checkpoints.draft",
            "research.checkpoints.critique",
            "research.checkpoints.finalize",
            "research.checkpoints.assemble",
            "research.agents",
            "research.agents.scope",
            "research.agents.planner",
            "research.agents.supervisor",
            "research.agents.subagent",
            "research.agents.generator",
            "research.agents.reviewer",
            "research.agents.finalizer",
        )
        return importlib.import_module("research.checkpoints")

    def test_exports_run_stamp(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod, "RunStamp")

    def test_exports_wall_clock_snapshot(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod, "WallClockSnapshot")

    def test_exports_run_finalization(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert hasattr(mod, "RunFinalization")

    def test_exports_stamp_run_metadata(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.stamp_run_metadata)

    def test_exports_snapshot_wall_clock(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.snapshot_wall_clock)

    def test_exports_finalize_run_metadata(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.finalize_run_metadata)

    def test_exports_run_scope(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_scope)

    def test_exports_run_plan(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_plan)

    def test_all_list_complete(self, monkeypatch):
        mod = self._load(monkeypatch)
        expected = {
            "CitationResolutionError",
            "GroundingError",
            "RunFinalization",
            "RunStamp",
            "WallClockSnapshot",
            "assemble_package",
            "finalize_run_metadata",
            "snapshot_wall_clock",
            "stamp_run_metadata",
            "run_critique",
            "run_draft",
            "run_finalize",
            "run_plan",
            "run_scope",
            "run_supervisor",
            "run_subagent",
            "run_judge",
        }
        assert set(mod.__all__) == expected

    def test_exports_run_supervisor(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_supervisor)

    def test_exports_run_subagent(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_subagent)

    def test_exports_run_draft(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_draft)

    def test_exports_run_critique(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_critique)

    def test_exports_run_finalize(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_finalize)

    def test_exports_assemble_package(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.assemble_package)

    def test_exports_grounding_error(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert issubclass(mod.GroundingError, Exception)

    def test_exports_citation_resolution_error(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert issubclass(mod.CitationResolutionError, Exception)


# ---------------------------------------------------------------------------
# Supervisor checkpoint tests
# ---------------------------------------------------------------------------


class TestSupervisorCheckpoint:
    """Unit tests for ``research.checkpoints.supervisor``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.supervisor",
            "research.checkpoints",
            "research.agents",
            "research.agents.supervisor",
        )
        mod = importlib.import_module("research.checkpoints.supervisor")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_supervisor._checkpoint_type == "llm_call"

    def test_run_supervisor_returns_decision(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan
        from research.contracts.decisions import SupervisorDecision

        brief = ResearchBrief(topic="RLHF", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        expected = SupervisorDecision(
            done=False, rationale="Need more data", gaps=["gap1"]
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_supervisor_agent", lambda model: MockAgent())
        result = mod.run_supervisor(brief, plan, "ledger text", 0.05, 0, "test-model")
        assert result is expected
        assert result.done is False
        assert result.gaps == ["gap1"]

    def test_run_supervisor_passes_model_to_builder(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan
        from research.contracts.decisions import SupervisorDecision

        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        captured = []

        class MockResult:
            output = SupervisorDecision(done=True, rationale="done")

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        def build(model_name):
            captured.append(model_name)
            return MockAgent()

        monkeypatch.setattr(mod, "build_supervisor_agent", build)
        mod.run_supervisor(brief, plan, "", 0.1, 0, "google-gla:gemini-2.5-flash")
        assert captured == ["google-gla:gemini-2.5-flash"]

    def test_run_supervisor_prompt_contains_context(self, monkeypatch):
        """Supervisor prompt JSON contains brief, plan, budget, and iteration."""
        import json

        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan
        from research.contracts.decisions import SupervisorDecision

        brief = ResearchBrief(topic="AI Safety", raw_request="test")
        plan = ResearchPlan(goal="study", key_questions=["how?"])

        class MockResult:
            output = SupervisorDecision(done=True, rationale="r")

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_supervisor_agent", lambda m: mock)
        mod.run_supervisor(brief, plan, "evidence here", 0.05, 2, "m")
        parsed = json.loads(mock.last_prompt)
        assert parsed["brief"]["topic"] == "AI Safety"
        assert parsed["plan"]["goal"] == "study"
        assert parsed["remaining_budget_usd"] == 0.05
        assert parsed["iteration_index"] == 2
        assert parsed["ledger_projection"] == "evidence here"


# ---------------------------------------------------------------------------
# Subagent checkpoint tests
# ---------------------------------------------------------------------------


class TestSubagentCheckpoint:
    """Unit tests for ``research.checkpoints.subagent``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.subagent",
            "research.checkpoints",
            "research.agents",
            "research.agents.subagent",
        )
        mod = importlib.import_module("research.checkpoints.subagent")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_subagent._checkpoint_type == "llm_call"

    def test_run_subagent_returns_findings(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search RLHF", target_subtopic="RLHF")
        expected = SubagentFindings(
            findings=["found stuff"], source_references=["ref1"]
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(
            mod, "build_subagent_agent", lambda model, tools=None: MockAgent()
        )
        result = mod.run_subagent(task, "test-model")
        assert result is expected
        assert result.findings == ["found stuff"]

    def test_run_subagent_passes_tools(self, monkeypatch):
        """Tools are passed through to build_subagent_agent."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search", target_subtopic="t")
        captured = []

        class MockResult:
            output = SubagentFindings(findings=["f"])

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        def build(model_name, tools=None):
            captured.append(tools)
            return MockAgent()

        monkeypatch.setattr(mod, "build_subagent_agent", build)
        fake_tools = ["tool1", "tool2"]
        mod.run_subagent(task, "m", tools=fake_tools)
        assert captured == [fake_tools]

    def test_run_subagent_graceful_degradation(self, monkeypatch):
        """On agent failure, returns degraded SubagentFindings instead of raising."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="failing task", target_subtopic="t")

        def build_failing(model_name, tools=None):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError("LLM provider unavailable")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_subagent_agent", build_failing)
        result = mod.run_subagent(task, "m")

        # Should NOT raise — returns degraded findings
        assert isinstance(result, SubagentFindings)
        assert result.findings == []
        assert "Subagent failed" in result.confidence_notes
        assert "LLM provider unavailable" in result.confidence_notes

    def test_run_subagent_prompt_contains_task(self, monkeypatch):
        """Subagent prompt JSON contains the task details."""
        import json

        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(
            task_description="search for RLHF papers",
            target_subtopic="RLHF",
            search_strategy_hints=["arxiv", "semantic_scholar"],
        )

        class MockResult:
            output = SubagentFindings(findings=["f"])

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_subagent_agent", lambda m, tools=None: mock)
        mod.run_subagent(task, "m")
        parsed = json.loads(mock.last_prompt)
        assert parsed["task_description"] == "search for RLHF papers"
        assert parsed["target_subtopic"] == "RLHF"
        assert parsed["search_strategy_hints"] == ["arxiv", "semantic_scholar"]


# ---------------------------------------------------------------------------
# Draft checkpoint tests
# ---------------------------------------------------------------------------


class TestDraftCheckpoint:
    """Unit tests for ``research.checkpoints.draft``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.draft",
            "research.checkpoints",
            "research.agents",
            "research.agents.generator",
        )
        mod = importlib.import_module("research.checkpoints.draft")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_draft._checkpoint_type == "llm_call"

    def test_run_draft_returns_draft_report(self, monkeypatch):
        """run_draft calls generator agent and returns DraftReport."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        brief = ResearchBrief(topic="RLHF", raw_request="test")
        plan = ResearchPlan(goal="study RLHF", key_questions=["how?"])
        ledger = EvidenceLedger(items=[])

        expected = DraftReport(content="# Report\nBody", sections=["Report"])

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_generator_agent", lambda model: MockAgent())
        result = mod.run_draft(brief, plan, ledger, "test-model")
        assert result is expected
        assert result.content == "# Report\nBody"
        assert result.sections == ["Report"]

    def test_run_draft_prompt_contains_brief_plan_ledger(self, monkeypatch):
        """run_draft prompt JSON includes brief, plan, and ledger."""
        import json

        mod, _ = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        brief = ResearchBrief(topic="AI Safety", raw_request="test")
        plan = ResearchPlan(goal="investigate", key_questions=["what?"])
        item = EvidenceItem(
            evidence_id="ev-1",
            title="Paper A",
            synthesis="Key finding",
            iteration_added=0,
        )
        ledger = EvidenceLedger(items=[item])

        class MockResult:
            output = DraftReport(content="report", sections=[])

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_generator_agent", lambda model: mock)
        mod.run_draft(brief, plan, ledger, "test-model")

        parsed = json.loads(mock.last_prompt)
        assert parsed["brief"]["topic"] == "AI Safety"
        assert parsed["plan"]["goal"] == "investigate"
        assert len(parsed["ledger"]["items"]) == 1
        assert parsed["ledger"]["items"][0]["evidence_id"] == "ev-1"


# ---------------------------------------------------------------------------
# Critique checkpoint tests
# ---------------------------------------------------------------------------


class TestCritiqueCheckpoint:
    """Unit tests for ``research.checkpoints.critique``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.critique",
            "research.checkpoints",
            "research.agents",
            "research.agents.reviewer",
        )
        mod = importlib.import_module("research.checkpoints.critique")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_critique._checkpoint_type == "llm_call"

    def test_single_reviewer_returns_critique(self, monkeypatch):
        """Standard tier: single reviewer, no second_model_name."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=["Report"])
        expected = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.8, explanation="good"
                )
            ],
            require_more_research=False,
            issues=["minor issue"],
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_reviewer_agent", lambda model: MockAgent())
        result = mod.run_critique(draft, "test-model")
        assert result is expected
        assert result.require_more_research is False
        assert result.issues == ["minor issue"]

    def test_dual_reviewer_merges_critiques(self, monkeypatch):
        """Deep tier: two reviewers, merged (averaged scores, union of issues)."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=["Report"])

        critique_a = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.8, explanation="good"
                ),
                CritiqueDimensionScore(
                    dimension="grounding", score=0.6, explanation="ok"
                ),
            ],
            require_more_research=False,
            issues=["issue A"],
        )
        critique_b = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.6, explanation="decent"
                ),
                CritiqueDimensionScore(
                    dimension="source_reliability", score=0.9, explanation="solid"
                ),
            ],
            require_more_research=True,
            issues=["issue B", "issue A"],  # duplicate with A
        )

        call_count = [0]

        def make_agent(model_name):
            class MockResult:
                pass

            class MockAgent:
                def run_sync(self, prompt):
                    idx = call_count[0]
                    call_count[0] += 1
                    r = MockResult()
                    r.output = critique_a if idx == 0 else critique_b
                    return r

            return MockAgent()

        monkeypatch.setattr(mod, "build_reviewer_agent", make_agent)
        result = mod.run_critique(draft, "model-a", second_model_name="model-b")

        # Scores averaged for completeness
        dim_map = {d.dimension: d for d in result.dimensions}
        assert "completeness" in dim_map
        assert dim_map["completeness"].score == pytest.approx(0.7)  # (0.8+0.6)/2

        # Unique dimensions kept
        assert "grounding" in dim_map  # only in A
        assert "source_reliability" in dim_map  # only in B

        # Issues: union, deduplicated
        assert result.issues == ["issue A", "issue B"]

        # require_more_research: True if either is True
        assert result.require_more_research is True

        # Provenance combined
        assert len(result.reviewer_provenance) == 2
        assert "reviewer_1:model-a" in result.reviewer_provenance
        assert "reviewer_2:model-b" in result.reviewer_provenance

    def test_dual_reviewer_tolerates_single_failure(self, monkeypatch):
        """Deep tier: one reviewer fails, other succeeds."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=[])
        good_critique = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.7, explanation="ok"
                )
            ],
            require_more_research=False,
            issues=["one issue"],
        )

        call_count = [0]

        def make_agent(model_name):
            class MockAgent:
                def run_sync(self, prompt):
                    idx = call_count[0]
                    call_count[0] += 1
                    if idx == 0:
                        raise RuntimeError("Provider A down")

                    class MockResult:
                        output = good_critique

                    return MockResult()

            return MockAgent()

        monkeypatch.setattr(mod, "build_reviewer_agent", make_agent)
        result = mod.run_critique(draft, "model-a", second_model_name="model-b")

        # Should return the surviving critique
        assert result.require_more_research is False
        assert result.issues == ["one issue"]
        assert "reviewer_2:model-b" in result.reviewer_provenance

    def test_dual_reviewer_both_fail_raises(self, monkeypatch):
        """Deep tier: both reviewers fail => RuntimeError."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import DraftReport

        draft = DraftReport(content="# Report", sections=[])

        def make_agent(model_name):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError(f"{model_name} down")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_reviewer_agent", make_agent)

        with pytest.raises(RuntimeError, match="Both reviewers failed"):
            mod.run_critique(draft, "model-a", second_model_name="model-b")


# ---------------------------------------------------------------------------
# Finalize checkpoint tests
# ---------------------------------------------------------------------------


class TestFinalizeCheckpoint:
    """Unit tests for ``research.checkpoints.finalize``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.finalize",
            "research.checkpoints",
            "research.agents",
            "research.agents.finalizer",
        )
        mod = importlib.import_module("research.checkpoints.finalize")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_finalize._checkpoint_type == "llm_call"

    def test_run_finalize_returns_final_report(self, monkeypatch):
        """run_finalize calls finalizer agent and returns FinalReport."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
            FinalReport,
        )

        draft = DraftReport(content="# Draft", sections=["Draft"])
        critique = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.7, explanation="ok"
                )
            ],
            require_more_research=False,
            issues=["fix typo"],
        )
        expected = FinalReport(
            content="# Final Report\nRevised body",
            sections=["Final Report"],
            stop_reason="converged",
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_finalizer_agent", lambda model: MockAgent())
        result = mod.run_finalize(draft, critique, "test-model")
        assert result is expected
        assert result.content == "# Final Report\nRevised body"
        assert result.stop_reason == "converged"

    def test_run_finalize_returns_none_on_failure(self, monkeypatch):
        """On agent failure, run_finalize returns None (not crash)."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Draft", sections=[])
        critique = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.5, explanation="poor"
                )
            ],
            require_more_research=True,
        )

        def make_failing(model_name):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError("Finalizer LLM error")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_finalizer_agent", make_failing)
        result = mod.run_finalize(draft, critique, "test-model")
        assert result is None


# ---------------------------------------------------------------------------
# Assemble checkpoint tests
# ---------------------------------------------------------------------------


class TestAssembleCheckpoint:
    """Unit tests for ``research.checkpoints.assemble``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.assemble",
            "research.checkpoints",
            "research.agents",
        )
        mod = importlib.import_module("research.checkpoints.assemble")
        return mod

    def test_checkpoint_type_is_tool_call(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert mod.assemble_package._checkpoint_type == "tool_call"

    def test_produces_valid_package(self, monkeypatch):
        """Assembly produces a valid InvestigationPackage with schema_version 1.0."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import InvestigationPackage, RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        ledger = EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    title="Paper A",
                    synthesis="Found X",
                    iteration_added=0,
                ),
            ]
        )
        draft = DraftReport(
            content="This study found X [ev_001]. The results show Y [ev_001]. Further analysis [ev_001] confirms.",
            sections=["Introduction"],
        )
        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        pkg = mod.assemble_package(
            meta, brief, plan, ledger, [], draft, None, None, grounding_min_ratio=0.5
        )
        assert isinstance(pkg, InvestigationPackage)
        assert pkg.schema_version == "1.0"
        assert pkg.metadata.run_id == "run-123"
        assert pkg.draft is draft
        assert pkg.prompt_hashes  # should have prompt hashes

    def test_unresolved_citations_raise(self, monkeypatch):
        """Assembly fails if report cites evidence IDs not in the ledger."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        ledger = EvidenceLedger(items=[])  # empty ledger
        draft = DraftReport(
            content="This claims X [ev_999] which is not in the ledger. More text here for length.",
            sections=[],
        )
        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        with pytest.raises(mod.CitationResolutionError, match="ev_999"):
            mod.assemble_package(meta, brief, plan, ledger, [], draft, None, None)

    def test_grounding_density_below_threshold_raises(self, monkeypatch):
        """Assembly fails if grounding density is below the threshold."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        ledger = EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    title="A",
                    synthesis="s",
                    iteration_added=0,
                ),
            ]
        )
        # Only 1 of 3 sentences has a citation -> density 0.33
        draft = DraftReport(
            content=(
                "This first sentence has no evidence at all and is quite long. "
                "This second sentence also lacks any references whatsoever. "
                "This third sentence cites [ev_001] properly."
            ),
            sections=[],
        )
        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        with pytest.raises(mod.GroundingError, match="below threshold"):
            mod.assemble_package(
                meta,
                brief,
                plan,
                ledger,
                [],
                draft,
                None,
                None,
                grounding_min_ratio=0.7,
            )

    def test_no_report_skips_grounding_check(self, monkeypatch):
        """Assembly succeeds when no report exists (draft=None, final=None)."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import InvestigationPackage, RunMetadata
        from research.contracts.plan import ResearchPlan

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

        pkg = mod.assemble_package(meta, brief, plan, ledger, [], None, None, None)
        assert isinstance(pkg, InvestigationPackage)

    def test_records_prompt_hashes(self, monkeypatch):
        """Assembly records prompt SHA256 hashes from the registry."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import RunMetadata
        from research.contracts.plan import ResearchPlan

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

        pkg = mod.assemble_package(meta, brief, plan, ledger, [], None, None, None)
        # Should have hashes for all loaded prompts
        assert isinstance(pkg.prompt_hashes, dict)
        assert len(pkg.prompt_hashes) > 0  # at least some prompts loaded
        # All values should be hex strings (SHA256)
        for name, hash_val in pkg.prompt_hashes.items():
            assert isinstance(hash_val, str)
            assert len(hash_val) == 64  # SHA256 hex length

    def test_prefers_final_report_for_grounding(self, monkeypatch):
        """When both draft and final exist, grounding checks use the final report."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import InvestigationPackage, RunMetadata
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport, FinalReport

        ledger = EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    title="A",
                    synthesis="s",
                    iteration_added=0,
                ),
            ]
        )
        # Draft has bad citations, final has good ones
        draft = DraftReport(content="Bad citation [ev_999] throughout.", sections=[])
        final = FinalReport(
            content="Good citation [ev_001] used properly. More text here [ev_001] for density.",
            sections=[],
        )
        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        # Should NOT raise because it checks final, not draft
        pkg = mod.assemble_package(
            meta, brief, plan, ledger, [], draft, None, final, grounding_min_ratio=0.5
        )
        assert isinstance(pkg, InvestigationPackage)
        assert pkg.final_report is final
