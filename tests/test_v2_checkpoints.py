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
        log=lambda **_kwargs: None,
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
# Plan revision checkpoint tests
# ---------------------------------------------------------------------------


class TestPlanRevisionCheckpoint:
    """Unit tests for ``research.checkpoints.replan``."""

    def _load(self, monkeypatch):
        FakeAgent = _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.replan",
            "research.checkpoints",
            "research.agents",
            "research.agents.replanner",
        )
        mod = importlib.import_module("research.checkpoints.replan")
        return mod, FakeAgent

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod, _ = self._load(monkeypatch)
        assert mod.run_plan_revision._checkpoint_type == "llm_call"

    def test_run_plan_revision_calls_agent_with_expected_json(self, monkeypatch):
        """run_plan_revision serializes brief/plan/critique/projection to JSON."""
        import json

        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
        )

        mod, FakeAgent = self._load(monkeypatch)

        brief = ResearchBrief(topic="RLHF", raw_request="RLHF alternatives?")
        plan = ResearchPlan(goal="Investigate RLHF", key_questions=["What?"])
        critique = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.5,
                    explanation="Missing benchmark comparisons",
                )
            ],
            require_more_research=True,
            issues=["Add benchmark coverage"],
        )
        expected_plan = ResearchPlan(
            goal="Investigate RLHF",
            key_questions=["What?", "Which benchmarks matter?"],
        )

        class MockResult:
            output = expected_plan

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock_agent = MockAgent()
        monkeypatch.setattr(mod, "build_replanner_agent", lambda model: mock_agent)

        result = mod.run_plan_revision(
            brief,
            plan,
            critique,
            "ledger projection text",
            "test-model",
        )

        assert result is expected_plan
        parsed = json.loads(mock_agent.last_prompt)
        assert parsed["brief"]["topic"] == "RLHF"
        assert parsed["plan"]["goal"] == "Investigate RLHF"
        assert parsed["critique"]["require_more_research"] is True
        assert parsed["ledger_projection"] == "ledger projection text"

    def test_run_plan_revision_passes_model_to_builder(self, monkeypatch):
        """run_plan_revision passes the model_name argument to build_replanner_agent."""
        from research.contracts.brief import ResearchBrief
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
        )

        mod, FakeAgent = self._load(monkeypatch)

        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        critique = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.5,
                    explanation="gap",
                )
            ],
            require_more_research=True,
            issues=["gap"],
        )
        captured_model = []

        class MockResult:
            output = ResearchPlan(goal="test", key_questions=["q"])

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        def fake_build(model_name):
            captured_model.append(model_name)
            return MockAgent()

        monkeypatch.setattr(mod, "build_replanner_agent", fake_build)
        mod.run_plan_revision(brief, plan, critique, "projection", "test-model")
        assert captured_model == ["test-model"]


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
            "research.checkpoints.replan",
            "research.checkpoints.supervisor",
            "research.checkpoints.subagent",
            "research.checkpoints.draft",
            "research.checkpoints.critique",
            "research.checkpoints.finalize",
            "research.checkpoints.verify",
            "research.checkpoints.assemble",
            "research.agents",
            "research.agents.scope",
            "research.agents.planner",
            "research.agents.replanner",
            "research.agents.supervisor",
            "research.agents.subagent",
            "research.agents.generator",
            "research.agents.reviewer",
            "research.agents.verifier",
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

    def test_exports_run_plan_revision(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_plan_revision)

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
            "record_iteration_spend",
            "resolve_tool_surface",
            "snapshot_wall_clock",
            "stamp_run_metadata",
            "run_critique",
            "run_draft",
            "run_finalize",
            "run_verify",
            "run_plan",
            "run_plan_revision",
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

    def test_exports_run_verify(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert callable(mod.run_verify)

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
        assert "critique_feedback" not in parsed

    def test_run_supervisor_prompt_contains_max_iterations_and_ledger_size(
        self, monkeypatch
    ):
        """max_iterations and ledger_size appear in the supervisor prompt JSON."""
        import json

        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.decisions import SupervisorDecision
        from research.contracts.plan import ResearchPlan

        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        class MockResult:
            output = SupervisorDecision(done=True, rationale="r")

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_supervisor_agent", lambda m: mock)
        mod.run_supervisor(
            brief, plan, "proj", 0.50, 3, "m",
            max_iterations=12,
            ledger_size=42,
        )
        parsed = json.loads(mock.last_prompt)
        assert parsed["max_iterations"] == 12
        assert parsed["ledger_size"] == 42

    def test_run_supervisor_breadth_first_adds_mode(self, monkeypatch):
        """breadth_first=True adds a 'mode' key to the prompt JSON."""
        import json

        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.decisions import SupervisorDecision
        from research.contracts.plan import ResearchPlan

        brief = ResearchBrief(topic="t", raw_request="t")
        plan = ResearchPlan(goal="g", key_questions=["q"])

        class MockResult:
            output = SupervisorDecision(done=True, rationale="r")

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_supervisor_agent", lambda m: mock)

        # Without breadth_first — no mode key
        mod.run_supervisor(brief, plan, "", 0.1, 0, "m", breadth_first=False)
        parsed = json.loads(mock.last_prompt)
        assert "mode" not in parsed

        # With breadth_first — mode key present
        mod.run_supervisor(brief, plan, "", 0.1, 0, "m", breadth_first=True)
        parsed = json.loads(mock.last_prompt)
        assert parsed["mode"] == "breadth_first"

    def test_run_supervisor_threads_critique_feedback_when_provided(
        self, monkeypatch
    ):
        """critique_feedback is included in the prompt only when explicitly passed."""
        import json

        mod, _ = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.decisions import SupervisorDecision
        from research.contracts.plan import ResearchPlan

        brief = ResearchBrief(topic="t", raw_request="t")
        plan = ResearchPlan(goal="g", key_questions=["q"])
        critique_feedback = (
            "Supplemental critique feedback:\n"
            "Issues to address:\n"
            "- Add stronger benchmark evidence"
        )

        class MockResult:
            output = SupervisorDecision(done=False, rationale="r")

        class MockAgent:
            def run_sync(self, prompt):
                self.last_prompt = prompt
                return MockResult()

        mock = MockAgent()
        monkeypatch.setattr(mod, "build_supervisor_agent", lambda m: mock)

        mod.run_supervisor(
            brief,
            plan,
            "proj",
            0.1,
            1,
            "m",
            critique_feedback=critique_feedback,
        )
        parsed = json.loads(mock.last_prompt)
        assert parsed["critique_feedback"] == critique_feedback


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

    def test_run_subagent_tools_reach_agent_constructor(self, monkeypatch):
        """Tools flow from run_subagent through build_subagent_agent into the Agent() constructor."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.decisions import SubagentFindings
        from research.contracts.plan import SubagentTask

        task = SubagentTask(task_description="search", target_subtopic="t")

        # Track what Agent() receives via kwargs
        agent_constructor_kwargs: list[dict] = []

        class MockResult:
            output = SubagentFindings(findings=["f"])

        class SpyAgent:
            def __init__(self, model_name="test", **kwargs):
                agent_constructor_kwargs.append(kwargs)

            def run_sync(self, prompt):
                return MockResult()

        # Patch pydantic_ai.Agent to our spy
        import pydantic_ai

        monkeypatch.setattr(pydantic_ai, "Agent", SpyAgent)

        # Re-import subagent module so _build_agent picks up the spy
        _clear_modules(
            "research.agents._factory",
            "research.agents.subagent",
        )
        importlib.import_module("research.agents._factory")
        mod2 = importlib.import_module("research.checkpoints.subagent")

        fake_tools = [lambda: "search_tool", lambda: "fetch_tool"]
        mod2.run_subagent(task, "test-model", tools=fake_tools)

        assert len(agent_constructor_kwargs) >= 1
        assert agent_constructor_kwargs[0].get("tools") is fake_tools

    def test_run_subagent_no_tools_omits_from_constructor(self, monkeypatch):
        """When tools=None, Agent() is called without a 'tools' kwarg."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.decisions import SubagentFindings
        from research.contracts.plan import SubagentTask

        task = SubagentTask(task_description="search", target_subtopic="t")

        agent_constructor_kwargs: list[dict] = []

        class MockResult:
            output = SubagentFindings(findings=["f"])

        class SpyAgent:
            def __init__(self, model_name="test", **kwargs):
                agent_constructor_kwargs.append(kwargs)

            def run_sync(self, prompt):
                return MockResult()

        import pydantic_ai

        monkeypatch.setattr(pydantic_ai, "Agent", SpyAgent)

        _clear_modules(
            "research.agents._factory",
            "research.agents.subagent",
        )
        importlib.import_module("research.agents._factory")
        mod2 = importlib.import_module("research.checkpoints.subagent")

        mod2.run_subagent(task, "test-model", tools=None)

        assert len(agent_constructor_kwargs) >= 1
        # tools should NOT appear in kwargs when None
        assert "tools" not in agent_constructor_kwargs[0]

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

    def test_run_subagent_retries_on_503(self, monkeypatch):
        """Transient 503 triggers retry — succeeds on second attempt."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search", target_subtopic="t")
        expected = SubagentFindings(findings=["found it"])
        call_count = 0

        class ModelHTTPError(Exception):
            def __init__(self, status_code, model_name, body=None):
                self.status_code = status_code
                self.model_name = model_name
                self.body = body
                super().__init__(f"status_code: {status_code}")

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ModelHTTPError(503, "test-model", {"error": "unavailable"})
                return MockResult()

        monkeypatch.setattr(
            mod, "build_subagent_agent", lambda m, tools=None: MockAgent()
        )
        # Patch sleep to avoid real delay in tests
        monkeypatch.setattr(mod.time, "sleep", lambda _: None)
        result = mod.run_subagent(task, "m")
        assert result is expected
        assert call_count == 2

    def test_run_subagent_retries_on_429(self, monkeypatch):
        """Rate limit 429 triggers retry."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search", target_subtopic="t")
        expected = SubagentFindings(findings=["found it"])
        call_count = 0

        class ModelHTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"status_code: {status_code}")

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ModelHTTPError(429)
                return MockResult()

        monkeypatch.setattr(
            mod, "build_subagent_agent", lambda m, tools=None: MockAgent()
        )
        monkeypatch.setattr(mod.time, "sleep", lambda _: None)
        result = mod.run_subagent(task, "m")
        assert result is expected
        assert call_count == 3

    def test_run_subagent_degrades_after_max_retries(self, monkeypatch):
        """All retries exhausted → degraded findings, no crash."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search", target_subtopic="t")
        call_count = 0

        class ModelHTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"status_code: {status_code}")

        class MockAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                raise ModelHTTPError(503)

        monkeypatch.setattr(
            mod, "build_subagent_agent", lambda m, tools=None: MockAgent()
        )
        monkeypatch.setattr(mod.time, "sleep", lambda _: None)
        result = mod.run_subagent(task, "m")
        assert isinstance(result, SubagentFindings)
        assert result.findings == []
        assert "Subagent failed" in result.confidence_notes
        assert call_count == 3  # _MAX_ATTEMPTS

    def test_run_subagent_no_retry_on_non_retryable(self, monkeypatch):
        """Non-retryable errors (e.g. 400) degrade immediately without retry."""
        mod, _ = self._load(monkeypatch)
        from research.contracts.plan import SubagentTask
        from research.contracts.decisions import SubagentFindings

        task = SubagentTask(task_description="search", target_subtopic="t")
        call_count = 0

        class ModelHTTPError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"status_code: {status_code}")

        class MockAgent:
            def run_sync(self, prompt):
                nonlocal call_count
                call_count += 1
                raise ModelHTTPError(400)

        monkeypatch.setattr(
            mod, "build_subagent_agent", lambda m, tools=None: MockAgent()
        )
        result = mod.run_subagent(task, "m")
        assert isinstance(result, SubagentFindings)
        assert result.findings == []
        assert call_count == 1  # No retry


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
        """run_draft calls generator agent (str output) and wraps in DraftReport."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        brief = ResearchBrief(topic="RLHF", raw_request="test")
        plan = ResearchPlan(goal="study RLHF", key_questions=["how?"])
        ledger = EvidenceLedger(items=[])

        class MockResult:
            output = "## Report\nBody paragraph here."

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_generator_agent", lambda model: MockAgent())
        result = mod.run_draft(brief, plan, ledger, "test-model")
        assert isinstance(result, DraftReport)
        assert result.content == "## Report\nBody paragraph here."
        assert result.sections == ["Report"]

    def test_run_draft_prompt_contains_brief_plan_ledger(self, monkeypatch):
        """run_draft prompt JSON includes brief, plan, and ledger."""
        import json

        mod, _ = self._load(monkeypatch)

        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.plan import ResearchPlan

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
            output = "## Summary\nReport content."

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

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=["Report"])
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()
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
        result = mod.run_critique(draft, plan, ledger, "test-model")
        assert result == expected
        assert result.require_more_research is False
        assert result.issues == ["minor issue"]
        assert result.reviewer_disagreements == []

    def test_single_reviewer_clears_model_supplied_disagreements(self, monkeypatch):
        """Single-reviewer path must not trust model-supplied disagreement metadata."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
            ReviewerDisagreement,
        )

        draft = DraftReport(content="# Report", sections=["Report"])
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()
        bogus = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness", score=0.8, explanation="good"
                )
            ],
            require_more_research=False,
            issues=["minor issue"],
            reviewer_disagreements=[
                ReviewerDisagreement(
                    dimension="completeness",
                    reviewer_1_score=0.2,
                    reviewer_2_score=0.9,
                    delta=0.7,
                )
            ],
        )

        class MockResult:
            output = bogus

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_reviewer_agent", lambda model: MockAgent())
        result = mod.run_critique(draft, plan, ledger, "test-model")
        assert result.reviewer_disagreements == []

    def test_merge_critiques_warns_on_high_delta(self, monkeypatch):
        """Shared dimensions always record disagreement; large deltas also warn."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
        )

        warnings: list[str] = []
        monkeypatch.setattr(
            mod.logger,
            "warning",
            lambda msg, *args: warnings.append(msg % args),
        )

        critique_a = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.9,
                    explanation="comprehensive",
                ),
                CritiqueDimensionScore(
                    dimension="grounding",
                    score=0.6,
                    explanation="fine",
                ),
            ],
            require_more_research=False,
            issues=["issue A"],
            reviewer_provenance=["reviewer_1:model-a"],
        )
        critique_b = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.5,
                    explanation="large gap",
                ),
                CritiqueDimensionScore(
                    dimension="source_reliability",
                    score=0.8,
                    explanation="strong",
                ),
            ],
            require_more_research=True,
            issues=["issue B"],
            reviewer_provenance=["reviewer_2:model-b"],
        )

        result = mod._merge_critiques(
            critique_a, critique_b, disagreement_threshold=0.3
        )

        dim_map = {d.dimension: d for d in result.dimensions}
        assert dim_map["completeness"].score == pytest.approx(0.7)
        assert "grounding" in dim_map
        assert "source_reliability" in dim_map

        assert len(result.reviewer_disagreements) == 1
        disagreement = result.reviewer_disagreements[0]
        assert disagreement.dimension == "completeness"
        assert disagreement.reviewer_1_score == pytest.approx(0.9)
        assert disagreement.reviewer_2_score == pytest.approx(0.5)
        assert disagreement.delta == pytest.approx(0.4)
        assert any("Reviewer disagreement on completeness exceeds threshold 0.30" in w for w in warnings)

    def test_merge_critiques_records_low_delta_without_warning(self, monkeypatch):
        """Low deltas are still recorded but do not emit warnings."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
        )

        warnings: list[str] = []
        monkeypatch.setattr(
            mod.logger,
            "warning",
            lambda msg, *args: warnings.append(msg % args),
        )

        critique_a = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.8,
                    explanation="good",
                ),
                CritiqueDimensionScore(
                    dimension="grounding",
                    score=0.6,
                    explanation="solid",
                ),
            ],
            require_more_research=False,
        )
        critique_b = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.7,
                    explanation="pretty good",
                ),
                CritiqueDimensionScore(
                    dimension="grounding",
                    score=0.55,
                    explanation="acceptable",
                ),
            ],
            require_more_research=False,
        )

        result = mod._merge_critiques(
            critique_a, critique_b, disagreement_threshold=0.3
        )

        disagreements = {d.dimension: d for d in result.reviewer_disagreements}
        assert disagreements["completeness"].delta == pytest.approx(0.1)
        assert disagreements["grounding"].delta == pytest.approx(0.05)
        assert warnings == []

    def test_merge_critiques_only_records_shared_dimensions(self, monkeypatch):
        """Non-overlapping dimensions stay in merged scores but not disagreement list."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
        )

        critique_a = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.75,
                    explanation="good",
                ),
                CritiqueDimensionScore(
                    dimension="grounding",
                    score=0.65,
                    explanation="present only in A",
                ),
            ],
            require_more_research=False,
        )
        critique_b = CritiqueReport(
            dimensions=[
                CritiqueDimensionScore(
                    dimension="completeness",
                    score=0.55,
                    explanation="shared",
                ),
                CritiqueDimensionScore(
                    dimension="source_reliability",
                    score=0.85,
                    explanation="present only in B",
                ),
            ],
            require_more_research=False,
        )

        result = mod._merge_critiques(
            critique_a, critique_b, disagreement_threshold=0.3
        )

        dim_names = {d.dimension for d in result.dimensions}
        assert dim_names == {"completeness", "grounding", "source_reliability"}
        assert [d.dimension for d in result.reviewer_disagreements] == ["completeness"]

    def test_dual_reviewer_merges_critiques(self, monkeypatch):
        """Deep tier: two reviewers, merged (averaged scores, union of issues)."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=["Report"])
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

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
        result = mod.run_critique(draft, plan, ledger, "model-a", second_model_name="model-b")

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
        assert [d.dimension for d in result.reviewer_disagreements] == ["completeness"]

    def test_dual_reviewer_tolerates_single_failure(self, monkeypatch):
        """Deep tier: one reviewer fails, other succeeds."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import (
            CritiqueDimensionScore,
            CritiqueReport,
            DraftReport,
        )

        draft = DraftReport(content="# Report", sections=[])
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()
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
        result = mod.run_critique(draft, plan, ledger, "model-a", second_model_name="model-b")

        # Should return the surviving critique
        assert result.require_more_research is False
        assert result.issues == ["one issue"]
        assert "reviewer_2:model-b" in result.reviewer_provenance
        assert result.reviewer_disagreements == []

    def test_dual_reviewer_both_fail_raises(self, monkeypatch):
        """Deep tier: both reviewers fail => RuntimeError."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        draft = DraftReport(content="# Report", sections=[])
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

        def make_agent(model_name):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError(f"{model_name} down")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_reviewer_agent", make_agent)

        with pytest.raises(RuntimeError, match="Both reviewers failed"):
            mod.run_critique(draft, plan, ledger, "model-a", second_model_name="model-b")


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
        """run_finalize calls finalizer agent (str output) and wraps in FinalReport."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
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
        ledger = EvidenceLedger()

        class MockResult:
            output = "## Final Report\nRevised body"

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_finalizer_agent", lambda model: MockAgent())
        result = mod.run_finalize(
            draft, critique, ledger, "test-model", stop_reason="converged"
        )
        assert isinstance(result, FinalReport)
        assert result.content == "## Final Report\nRevised body"
        assert result.sections == ["Final Report"]
        assert result.stop_reason == "converged"

    def test_run_finalize_returns_none_on_failure(self, monkeypatch):
        """On agent failure, run_finalize returns None (not crash)."""
        mod, _ = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
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
        ledger = EvidenceLedger()

        def make_failing(model_name):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError("Finalizer LLM error")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_finalizer_agent", make_failing)
        result = mod.run_finalize(draft, critique, ledger, "test-model")
        assert result is None


# ---------------------------------------------------------------------------
# Verify checkpoint tests
# ---------------------------------------------------------------------------


class TestVerifyCheckpoint:
    """Unit tests for ``research.checkpoints.verify``."""

    def _load(self, monkeypatch):
        _install_checkpoint_stubs(monkeypatch)
        _clear_modules(
            "research.checkpoints.verify",
            "research.checkpoints",
            "research.agents",
            "research.agents.verifier",
        )
        return importlib.import_module("research.checkpoints.verify")

    def test_checkpoint_type_is_llm_call(self, monkeypatch):
        mod = self._load(monkeypatch)
        assert mod.run_verify._checkpoint_type == "llm_call"

    def test_run_verify_returns_report(self, monkeypatch):
        mod = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.reports import (
            FinalReport,
            VerificationIssue,
            VerificationReport,
        )

        report = FinalReport(content="# Final", sections=["Final"])
        ledger = EvidenceLedger()
        expected = VerificationReport(
            issues=[
                VerificationIssue(
                    claim_excerpt="Claim",
                    evidence_ids=["ev_001"],
                    status="partial",
                )
            ],
            verified_claim_count=4,
            unsupported_claim_count=0,
            needs_revision=False,
        )

        class MockResult:
            output = expected

        class MockAgent:
            def run_sync(self, prompt):
                return MockResult()

        monkeypatch.setattr(mod, "build_verifier_agent", lambda model: MockAgent())
        result = mod.run_verify(report, ledger, "test-model")
        assert result is expected

    def test_run_verify_returns_none_on_failure(self, monkeypatch):
        mod = self._load(monkeypatch)

        from research.contracts.evidence import EvidenceLedger
        from research.contracts.reports import DraftReport

        report = DraftReport(content="# Draft", sections=["Draft"])
        ledger = EvidenceLedger()

        def make_failing(model_name):
            class FailingAgent:
                def run_sync(self, prompt):
                    raise RuntimeError("Verifier LLM error")

            return FailingAgent()

        monkeypatch.setattr(mod, "build_verifier_agent", make_failing)
        result = mod.run_verify(report, ledger, "test-model")
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
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
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
            meta,
            brief,
            plan,
            ledger,
            [],
            draft,
            None,
            None,
            ToolProviderManifest(),
            grounding_min_ratio=0.5,
        )
        assert isinstance(pkg, InvestigationPackage)
        assert pkg.schema_version == "1.0"
        assert pkg.metadata.run_id == "run-123"
        assert pkg.draft is draft
        assert pkg.verification is None
        assert pkg.prompt_hashes  # should have prompt hashes
        assert pkg.metadata.grounding_density is not None
        assert pkg.metadata.grounding_density >= 0.5  # passes threshold

    def test_assemble_preserves_revised_plan_separately(self, monkeypatch):
        """Assembly keeps the approved plan and revised plan in separate fields."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import RunMetadata, ToolProviderManifest
        from research.contracts.plan import ResearchPlan

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        approved_plan = ResearchPlan(goal="approved", key_questions=["q1"])
        revised_plan = ResearchPlan(goal="approved", key_questions=["q1", "q2"])

        pkg = mod.assemble_package(
            meta,
            brief,
            approved_plan,
            EvidenceLedger(),
            [],
            None,
            None,
            None,
            ToolProviderManifest(),
            revised_plan=revised_plan,
        )

        assert pkg.plan == approved_plan
        assert pkg.revised_plan == revised_plan

    def test_unresolved_citations_raise(self, monkeypatch):
        """Assembly fails if strict_grounding=True and report cites evidence IDs not in the ledger."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata, ToolProviderManifest
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        # Non-empty ledger so grounding checks are not skipped,
        # but the draft cites ev_999 which doesn't exist in it.
        ledger = EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    title="Real paper",
                    synthesis="Real finding",
                    iteration_added=0,
                ),
            ]
        )
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
            mod.assemble_package(
                meta,
                brief,
                plan,
                ledger,
                [],
                draft,
                None,
                None,
                ToolProviderManifest(),
                strict_grounding=True,
            )

    def test_unresolved_citations_warn_in_soft_mode(self, monkeypatch, caplog):
        """In soft mode (the default), unresolved citations log a warning
        and the package still ships — LLMs occasionally hallucinate
        citation slugs and we don't want the whole run thrown away at
        the final assembly step."""
        import logging as _logging

        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        ledger = EvidenceLedger(
            items=[
                EvidenceItem(
                    evidence_id="ev_001",
                    title="Real paper",
                    synthesis="Real finding",
                    iteration_added=0,
                ),
            ]
        )
        draft = DraftReport(
            content=(
                "This claims X [investigate-pipeline-run] which is a "
                "hallucinated slug. But this other claim [ev_001] resolves. "
                "More text here for length."
            ),
            sections=[],
        )
        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        with caplog.at_level(_logging.WARNING):
            pkg = mod.assemble_package(
                meta,
                brief,
                plan,
                ledger,
                [],
                draft,
                None,
                None,
                ToolProviderManifest(),
                grounding_min_ratio=0.0,  # don't tangle with the density branch
            )

        assert isinstance(pkg, InvestigationPackage)
        assert any(
            "investigate-pipeline-run" in rec.getMessage() for rec in caplog.records
        )

    def test_grounding_density_below_threshold_raises(self, monkeypatch):
        """Assembly fails if grounding density is below the threshold AND strict_grounding is True."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import RunMetadata, ToolProviderManifest
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
                ToolProviderManifest(),
                grounding_min_ratio=0.7,
                strict_grounding=True,
            )

    def test_grounding_density_below_threshold_warns_by_default(self, monkeypatch):
        """Assembly succeeds with warning when density is below threshold
        and strict_grounding is False (the default)."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
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

        # strict_grounding defaults to False — should NOT raise
        pkg = mod.assemble_package(
            meta,
            brief,
            plan,
            ledger,
            [],
            draft,
            None,
            None,
            ToolProviderManifest(),
            grounding_min_ratio=0.7,
        )
        assert isinstance(pkg, InvestigationPackage)
        # Grounding density should be recorded in metadata
        assert pkg.metadata.grounding_density is not None
        assert 0.0 < pkg.metadata.grounding_density < 0.7

    def test_no_report_skips_grounding_check(self, monkeypatch):
        """Assembly succeeds when no report exists (draft=None, final=None)."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
        from research.contracts.plan import ResearchPlan

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

        pkg = mod.assemble_package(
            meta, brief, plan, ledger, [], None, None, None, ToolProviderManifest()
        )
        assert isinstance(pkg, InvestigationPackage)

    def test_records_prompt_hashes(self, monkeypatch):
        """Assembly records prompt SHA256 hashes from the registry."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import RunMetadata, ToolProviderManifest
        from research.contracts.plan import ResearchPlan

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()

        pkg = mod.assemble_package(
            meta, brief, plan, ledger, [], None, None, None, ToolProviderManifest()
        )
        # Should have hashes for all loaded prompts
        assert isinstance(pkg.prompt_hashes, dict)
        assert len(pkg.prompt_hashes) > 0  # at least some prompts loaded
        # All values should be hex strings (SHA256)
        for name, hash_val in pkg.prompt_hashes.items():
            assert isinstance(hash_val, str)
            assert len(hash_val) == 64  # SHA256 hex length

    def test_accepts_and_persists_verification_report(self, monkeypatch):
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import RunMetadata, ToolProviderManifest
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import VerificationIssue, VerificationReport

        meta = RunMetadata(
            run_id="run-123", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])
        ledger = EvidenceLedger()
        verification = VerificationReport(
            issues=[
                VerificationIssue(
                    claim_excerpt="Claim",
                    evidence_ids=["ev_001"],
                    status="unsupported",
                )
            ],
            verified_claim_count=2,
            unsupported_claim_count=1,
            needs_revision=True,
        )

        pkg = mod.assemble_package(
            meta,
            brief,
            plan,
            ledger,
            [],
            None,
            None,
            None,
            ToolProviderManifest(),
            verification=verification,
        )

        assert pkg.verification is verification

    def test_prefers_final_report_for_grounding(self, monkeypatch):
        """When both draft and final exist, grounding checks use the final report."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
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
            meta,
            brief,
            plan,
            ledger,
            [],
            draft,
            None,
            final,
            ToolProviderManifest(),
            grounding_min_ratio=0.5,
        )
        assert isinstance(pkg, InvestigationPackage)
        assert pkg.final_report is final

    def test_empty_ledger_with_report_skips_grounding(self, monkeypatch):
        """When ledger is empty (e.g. all subagents failed), assembly succeeds
        with a warning instead of raising GroundingError."""
        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        ledger = EvidenceLedger(items=[])  # empty — no evidence collected
        draft = DraftReport(
            content=(
                "This report has no citations because all subagents failed. "
                "The system should still produce a package instead of crashing."
            ),
            sections=["Summary"],
        )
        meta = RunMetadata(
            run_id="run-empty", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        # Should NOT raise — empty ledger means grounding check is skipped
        pkg = mod.assemble_package(
            meta,
            brief,
            plan,
            ledger,
            [],
            draft,
            None,
            None,
            ToolProviderManifest(),
            grounding_min_ratio=0.7,
        )
        assert isinstance(pkg, InvestigationPackage)
        assert pkg.ledger.items == []
        assert pkg.draft is draft

    def test_underlength_warning_for_short_report_with_evidence(self, monkeypatch):
        """Assembly logs warning when report is short but ledger has substantial evidence."""
        import logging

        mod = self._load(monkeypatch)
        from research.contracts.brief import ResearchBrief
        from research.contracts.evidence import EvidenceItem, EvidenceLedger
        from research.contracts.package import (
            InvestigationPackage,
            RunMetadata,
            ToolProviderManifest,
        )
        from research.contracts.plan import ResearchPlan
        from research.contracts.reports import DraftReport

        # Build a ledger with 6 items (above the threshold of 5)
        items = [
            EvidenceItem(
                evidence_id=f"ev_{i:03d}",
                title=f"Paper {i}",
                synthesis=f"Finding {i}",
                iteration_added=0,
            )
            for i in range(6)
        ]
        ledger = EvidenceLedger(items=items)

        # Short report: well under 300 words but all cited
        draft = DraftReport(
            content="Short summary [ev_000] [ev_001] [ev_002].",
            sections=["Summary"],
        )
        meta = RunMetadata(
            run_id="run-short", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        brief = ResearchBrief(topic="test", raw_request="test")
        plan = ResearchPlan(goal="test", key_questions=["q"])

        with monkeypatch.context() as m:
            warnings_logged: list[str] = []
            original_warning = logging.getLogger("research.checkpoints.assemble").warning

            def capture_warning(msg, *args):
                warnings_logged.append(msg % args)

            m.setattr(
                logging.getLogger("research.checkpoints.assemble"),
                "warning",
                capture_warning,
            )

            pkg = mod.assemble_package(
                meta,
                brief,
                plan,
                ledger,
                [],
                draft,
                None,
                None,
                ToolProviderManifest(),
                grounding_min_ratio=0.0,
            )

        # Package should still be produced (non-fatal)
        assert isinstance(pkg, InvestigationPackage)
        # Warning should mention short report
        assert any("short" in w.lower() or "words" in w.lower() for w in warnings_logged)
