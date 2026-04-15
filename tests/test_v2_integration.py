"""Integration tests for V2 flow-level behaviors.

End-to-end scenarios with stubbed checkpoints — no real LLM calls.
Each test is self-contained — no shared conftest, no imports from other test files.
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
from research.contracts.decisions import SubagentFindings, SupervisorDecision
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import (
    CouncilComparison,
    CouncilPackage,
    InvestigationPackage,
    RunMetadata,
)
from research.contracts.plan import ResearchPlan, SubagentTask
from research.contracts.reports import (
    CritiqueDimensionScore,
    CritiqueReport,
    DraftReport,
    FinalReport,
)


# ---------------------------------------------------------------------------
# Stub infrastructure (self-contained, mirrors test_v2_flow.py)
# ---------------------------------------------------------------------------

_FLOW_MODULES = [
    "research.flows.deep_research",
    "research.flows.council",
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

    class FakeCapturePolicy:
        def __init__(self, *, tool_capture=None):
            self.tool_capture = tool_capture

    class FakeKitaruAgent:
        """Passthrough stub — delegates to the wrapped agent."""

        def __init__(self, agent, *, name=None, capture=None):
            self._wrapped = agent

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    kp_ns = types.SimpleNamespace(
        KitaruAgent=FakeKitaruAgent, CapturePolicy=FakeCapturePolicy
    )
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
    for name in _FLOW_MODULES:
        sys.modules.pop(name, None)


def _load_flow_module(monkeypatch):
    """Install stubs, clear caches, and return the deep_research flow module."""
    _install_stubs(monkeypatch)
    _clear_modules()
    return importlib.import_module("research.flows.deep_research")


def _load_council_module(monkeypatch):
    """Install stubs, clear caches, and return the council flow module."""
    _install_stubs(monkeypatch)
    _clear_modules()
    return importlib.import_module("research.flows.council")


def make_checkpoint_stub(return_value):
    """Create a stub that works as both direct call and .submit().load()."""

    def stub(*args, **kwargs):
        return return_value

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: return_value)
    return stub


def with_submit(fn):
    """Add .submit() support to a tracking function.

    Unlike ``make_checkpoint_stub``, this routes .submit().load() through
    the original function so side-effects (call tracking, counters) fire.
    """
    fn.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: fn(*a, **kw))
    return fn


def make_stateful_stub(values):
    """Return a different value on each call, clamping to last."""
    call_count = [0]

    def stub(*args, **kwargs):
        idx = min(call_count[0], len(values) - 1)
        call_count[0] += 1
        return values[idx]

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: stub())
    stub.call_count = call_count
    return stub


def _make_passthrough_assemble():
    """Assembly stub that constructs a real InvestigationPackage from args."""

    def assemble_stub(
        metadata,
        brief,
        plan,
        ledger,
        iterations,
        draft,
        critique,
        final_report,
        grounding_min_ratio=0.7,
    ):
        return InvestigationPackage(
            metadata=metadata,
            brief=brief,
            plan=plan,
            ledger=ledger,
            iterations=iterations,
            draft=draft,
            critique=critique,
            final_report=final_report,
            prompt_hashes={"scope": "abc123", "planner": "def456"},
        )

    assemble_stub.submit = lambda *a, **kw: types.SimpleNamespace(
        load=lambda: assemble_stub(*a, **kw)
    )
    return assemble_stub


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

_STAMP = types.SimpleNamespace(run_id="integ-001", started_at="2024-01-01T00:00:00Z")
_CLOCK = types.SimpleNamespace(now_iso="2024-01-01T00:00:10Z", elapsed_seconds=10)
_FINALIZATION = types.SimpleNamespace(
    completed_at="2024-01-01T00:01:00Z", elapsed_seconds=60
)

_BRIEF = ResearchBrief(topic="integration test", raw_request="test question")
_PLAN = ResearchPlan(goal="investigate", key_questions=["what?"])

_DECISION_DONE = SupervisorDecision(
    done=True,
    rationale="sufficient",
    subagent_tasks=[
        SubagentTask(task_description="search", target_subtopic="t"),
    ],
)
_DECISION_CONTINUE = SupervisorDecision(
    done=False,
    rationale="need more",
    gaps=["gap1"],
    subagent_tasks=[
        SubagentTask(task_description="search more", target_subtopic="t2"),
    ],
)
_FINDINGS = SubagentFindings(findings=["found X"], source_references=["ref1"])
_FINDINGS_DEGRADED = SubagentFindings(
    findings=[],
    source_references=[],
    confidence_notes="Subagent failed: simulated error",
)
_DRAFT = DraftReport(content="# Draft Report", sections=["Introduction"])
_CRITIQUE_OK = CritiqueReport(
    dimensions=[
        CritiqueDimensionScore(dimension="completeness", score=0.8, explanation="good")
    ],
    require_more_research=False,
    issues=[],
)
_CRITIQUE_NEEDS_MORE = CritiqueReport(
    dimensions=[
        CritiqueDimensionScore(dimension="completeness", score=0.5, explanation="gaps")
    ],
    require_more_research=True,
    issues=["gap found"],
)
_FINAL = FinalReport(content="# Final Report", sections=["Conclusion"])


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_iterations=5,
    max_parallel_subagents=3,
    max_supplemental_loops=1,
    allow_unfinalized_package=False,
    soft_budget_usd=1.0,
    spent_usd=0.0,
) -> ResearchConfig:
    """Build a minimal ResearchConfig for integration testing."""
    slots = {
        "generator": ModelSlotConfig(provider="test-a", model="gen"),
        "subagent": ModelSlotConfig(provider="test-b", model="sub"),
        "reviewer": ModelSlotConfig(provider="test-c", model="rev"),
        "judge": ModelSlotConfig(provider="test-d", model="judge"),
    }
    budget = BudgetConfig(soft_budget_usd=soft_budget_usd, spent_usd=spent_usd)
    return ResearchConfig(
        tier="standard",
        budget=budget,
        slots=slots,
        max_iterations=max_iterations,
        max_parallel_subagents=max_parallel_subagents,
        ledger_window_iterations=3,
        grounding_min_ratio=0.0,
        max_supplemental_loops=max_supplemental_loops,
        wait_timeout_seconds=3600,
        allow_unfinalized_package=allow_unfinalized_package,
        strict_unknown_model_cost=False,
        sandbox_enabled=False,
        sandbox_backend=None,
        enabled_providers=["arxiv"],
    )


# ---------------------------------------------------------------------------
# Patching helper
# ---------------------------------------------------------------------------


def _patch_all_checkpoints(
    monkeypatch,
    flow_mod,
    *,
    supervisor_stub=None,
    critique_stub=None,
    subagent_stub=None,
    finalize_stub=None,
):
    """Monkeypatch all checkpoints on the flow module with defaults.

    Override specific checkpoints via keyword args.
    """
    monkeypatch.setattr(flow_mod, "stamp_run_metadata", make_checkpoint_stub(_STAMP))
    monkeypatch.setattr(flow_mod, "snapshot_wall_clock", make_checkpoint_stub(_CLOCK))
    monkeypatch.setattr(
        flow_mod, "finalize_run_metadata", make_checkpoint_stub(_FINALIZATION)
    )
    monkeypatch.setattr(flow_mod, "run_scope", make_checkpoint_stub(_BRIEF))
    monkeypatch.setattr(flow_mod, "run_plan", make_checkpoint_stub(_PLAN))
    monkeypatch.setattr(
        flow_mod,
        "run_supervisor",
        supervisor_stub or make_checkpoint_stub(_DECISION_DONE),
    )
    monkeypatch.setattr(
        flow_mod,
        "run_subagent",
        subagent_stub or make_checkpoint_stub(_FINDINGS),
    )
    monkeypatch.setattr(flow_mod, "run_draft", make_checkpoint_stub(_DRAFT))
    monkeypatch.setattr(
        flow_mod,
        "run_critique",
        critique_stub or make_checkpoint_stub(_CRITIQUE_OK),
    )
    monkeypatch.setattr(
        flow_mod,
        "run_finalize",
        finalize_stub or make_checkpoint_stub(_FINAL),
    )
    monkeypatch.setattr(flow_mod, "assemble_package", _make_passthrough_assemble())


# ---------------------------------------------------------------------------
# Failure stub helpers
# ---------------------------------------------------------------------------


def _make_always_failing_stub():
    """Stub that always raises RuntimeError."""
    calls = [0]

    def stub(*args, **kwargs):
        calls[0] += 1
        raise RuntimeError(f"Simulated failure #{calls[0]}")

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: stub(*a, **kw))
    return stub, calls


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestOneIterationHappyPath:
    """Full pipeline with supervisor returning done=True on first iteration."""

    def test_returns_complete_investigation_package(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)

    def test_metadata_total_iterations_is_one(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.metadata.total_iterations == 1

    def test_final_report_is_present(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.final_report is not None
        assert result.final_report.content == "# Final Report"

    def test_prompt_hashes_populated(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.prompt_hashes is not None
        assert len(result.prompt_hashes) > 0

    def test_schema_version(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.schema_version == "1.0"

    def test_all_fields_populated(self, monkeypatch):
        """Verify brief, plan, ledger, iterations, draft, critique are all present."""
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.brief is not None
        assert result.brief.topic == "integration test"
        assert result.plan is not None
        assert result.plan.goal == "investigate"
        assert result.ledger is not None
        assert result.iterations is not None
        assert len(result.iterations) == 1
        assert result.draft is not None
        assert result.critique is not None
        assert result.metadata.run_id == "integ-001"
        assert result.metadata.tier == "standard"
        assert result.metadata.started_at == "2024-01-01T00:00:00Z"
        assert result.metadata.completed_at == "2024-01-01T00:01:00Z"
        assert result.metadata.stop_reason == "supervisor_done"


class TestMultiIterationResearchLoop:
    """Supervisor returns done=False for first 2 iterations, then done=True."""

    def test_three_iterations_until_done(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        supervisor = make_stateful_stub(
            [
                _DECISION_CONTINUE,
                _DECISION_CONTINUE,
                _DECISION_DONE,
            ]
        )

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor)

        cfg = _make_config(max_iterations=10)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.metadata.total_iterations == 3

    def test_three_iteration_records(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        supervisor = make_stateful_stub(
            [
                _DECISION_CONTINUE,
                _DECISION_CONTINUE,
                _DECISION_DONE,
            ]
        )

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor)

        cfg = _make_config(max_iterations=10)
        result = flow_mod.deep_research("test question", config=cfg)

        assert len(result.iterations) == 3
        # Iteration indices are sequential
        for i, record in enumerate(result.iterations):
            assert record.iteration_index == i

    def test_stop_reason_is_supervisor_done(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        supervisor = make_stateful_stub(
            [
                _DECISION_CONTINUE,
                _DECISION_CONTINUE,
                _DECISION_DONE,
            ]
        )

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor)

        cfg = _make_config(max_iterations=10)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.metadata.stop_reason == "supervisor_done"


class TestReviewerTriggeredSupplementalLoop:
    """Critique returns require_more_research=True, triggering supplemental loop."""

    def test_supplemental_loop_runs_one_extra_iteration(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        # Supervisor always says done
        supervisor_calls = [0]

        def counting_supervisor(*args, **kwargs):
            supervisor_calls[0] += 1
            return _DECISION_DONE

        with_submit(counting_supervisor)

        # First critique: needs more. Second critique: also needs more (should be capped)
        critique = make_stateful_stub([_CRITIQUE_NEEDS_MORE, _CRITIQUE_NEEDS_MORE])

        _patch_all_checkpoints(
            monkeypatch,
            flow_mod,
            supervisor_stub=counting_supervisor,
            critique_stub=critique,
        )

        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test question", config=cfg)

        # 1 main iteration + 1 supplemental = 2 total
        assert result.metadata.total_iterations == 2
        assert len(result.iterations) == 2

    def test_supplemental_capped_at_max(self, monkeypatch):
        """Even with require_more_research=True, supplemental loops are capped."""
        flow_mod = _load_flow_module(monkeypatch)

        critique_calls = [0]

        def always_more(*args, **kwargs):
            critique_calls[0] += 1
            return _CRITIQUE_NEEDS_MORE

        with_submit(always_more)

        _patch_all_checkpoints(monkeypatch, flow_mod, critique_stub=always_more)

        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test question", config=cfg)

        # critique called twice: 1 initial + 1 supplemental
        assert critique_calls[0] == 2
        # Flow completes — doesn't loop forever
        assert isinstance(result, InvestigationPackage)

    def test_flow_completes_when_supplemental_capped(self, monkeypatch):
        """Verify the flow returns a valid package even when supplemental is capped."""
        flow_mod = _load_flow_module(monkeypatch)

        # Always request more research
        critique = make_stateful_stub([_CRITIQUE_NEEDS_MORE, _CRITIQUE_NEEDS_MORE])

        _patch_all_checkpoints(monkeypatch, flow_mod, critique_stub=critique)

        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.final_report is not None
        assert result.draft is not None


class TestDegradedSubagentFailurePath:
    """Subagent returns degraded result (empty findings) — pipeline still completes."""

    def test_pipeline_completes_with_degraded_subagent(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        degraded_stub = make_checkpoint_stub(_FINDINGS_DEGRADED)

        _patch_all_checkpoints(monkeypatch, flow_mod, subagent_stub=degraded_stub)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)

    def test_degraded_result_in_iteration_record(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        degraded_stub = make_checkpoint_stub(_FINDINGS_DEGRADED)

        _patch_all_checkpoints(monkeypatch, flow_mod, subagent_stub=degraded_stub)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert len(result.iterations) >= 1
        first_iter = result.iterations[0]
        for sa_result in first_iter.subagent_results:
            assert sa_result.findings == []
            assert "Subagent failed" in (sa_result.confidence_notes or "")


class TestFinalizerFailureAllowUnfinalized:
    """Finalizer failure with allow_unfinalized_package=True produces package without final_report."""

    def test_package_has_no_final_report(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        failing_finalize, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod, finalize_stub=failing_finalize)

        cfg = _make_config(allow_unfinalized_package=True)
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.final_report is None

    def test_stop_reason_preserved(self, monkeypatch):
        """When iteration loop set a stop_reason, it is preserved even on finalizer failure."""
        flow_mod = _load_flow_module(monkeypatch)

        failing_finalize, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod, finalize_stub=failing_finalize)

        cfg = _make_config(allow_unfinalized_package=True)
        result = flow_mod.deep_research("test question", config=cfg)

        # supervisor_done is set from the iteration loop; finalizer_failed
        # only becomes stop_reason when there was no prior reason
        assert result.metadata.stop_reason == "supervisor_done"

    def test_draft_and_critique_still_present(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        failing_finalize, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod, finalize_stub=failing_finalize)

        cfg = _make_config(allow_unfinalized_package=True)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.draft is not None
        assert result.critique is not None


class TestFinalizerFailureDisallowUnfinalized:
    """Finalizer failure with allow_unfinalized_package=False raises FinalizerError."""

    def test_raises_finalizer_error(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        failing_finalize, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod, finalize_stub=failing_finalize)

        cfg = _make_config(allow_unfinalized_package=False)

        with pytest.raises(flow_mod.FinalizerError):
            flow_mod.deep_research("test question", config=cfg)


class TestCouncilFlowIntegration:
    """Council flow: two generators, judge comparison, canonical selection."""

    def test_council_produces_package_with_both_generators(self, monkeypatch):
        council_mod = _load_council_module(monkeypatch)

        metadata_a = RunMetadata(
            run_id="run-a",
            tier="standard",
            started_at="2024-01-01T00:00:00Z",
            total_iterations=1,
            total_cost_usd=0.05,
        )
        metadata_b = RunMetadata(
            run_id="run-b",
            tier="standard",
            started_at="2024-01-01T00:01:00Z",
            total_iterations=2,
            total_cost_usd=0.08,
        )
        draft_a = DraftReport(content="# Report A", sections=["A"])
        draft_b = DraftReport(content="# Report B", sections=["B"])
        final_a = FinalReport(content="# Final A", sections=["A"])
        final_b = FinalReport(content="# Final B", sections=["B"])

        pkg_a = InvestigationPackage(
            metadata=metadata_a,
            brief=_BRIEF,
            plan=_PLAN,
            ledger=EvidenceLedger(items=[]),
            draft=draft_a,
            final_report=final_a,
        )
        pkg_b = InvestigationPackage(
            metadata=metadata_b,
            brief=_BRIEF,
            plan=_PLAN,
            ledger=EvidenceLedger(items=[]),
            draft=draft_b,
            final_report=final_b,
        )

        call_idx = [0]

        def mock_deep_research(question, tier="standard", config=None):
            call_idx[0] += 1
            return pkg_a if call_idx[0] == 1 else pkg_b

        comparison = CouncilComparison(
            comparison="B is better",
            generator_scores={"gen_a": 0.7, "gen_b": 0.9},
            recommended_generator="gen_b",
        )

        monkeypatch.setattr(council_mod, "deep_research", mock_deep_research)
        monkeypatch.setattr(council_mod, "run_judge", make_checkpoint_stub(comparison))

        cfg = _make_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="model_a"),
            "gen_b": ModelSlotConfig(provider="openai", model="model_b"),
        }

        result = council_mod.council_research(
            "test", config=cfg, generator_slots=gen_slots
        )

        assert isinstance(result, CouncilPackage)
        assert "gen_a" in result.packages
        assert "gen_b" in result.packages

    def test_canonical_generator_from_judge(self, monkeypatch):
        council_mod = _load_council_module(monkeypatch)

        pkg = InvestigationPackage(
            metadata=RunMetadata(
                run_id="run-x", tier="standard", started_at="2024-01-01T00:00:00Z"
            ),
            brief=_BRIEF,
            plan=_PLAN,
            ledger=EvidenceLedger(items=[]),
        )

        monkeypatch.setattr(council_mod, "deep_research", lambda q, **kw: pkg)

        comparison = CouncilComparison(
            comparison="B wins",
            generator_scores={"gen_a": 0.6, "gen_b": 0.95},
            recommended_generator="gen_b",
        )
        monkeypatch.setattr(council_mod, "run_judge", make_checkpoint_stub(comparison))

        cfg = _make_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = council_mod.council_research(
            "test", config=cfg, generator_slots=gen_slots
        )

        assert result.canonical_generator == "gen_b"

    def test_provider_compromise_detected(self, monkeypatch):
        council_mod = _load_council_module(monkeypatch)

        pkg = InvestigationPackage(
            metadata=RunMetadata(
                run_id="run-x", tier="standard", started_at="2024-01-01T00:00:00Z"
            ),
            brief=_BRIEF,
            plan=_PLAN,
            ledger=EvidenceLedger(items=[]),
        )

        monkeypatch.setattr(council_mod, "deep_research", lambda q, **kw: pkg)

        comparison = CouncilComparison(
            comparison="A wins",
            generator_scores={"gen_a": 0.9},
            recommended_generator="gen_a",
        )
        monkeypatch.setattr(council_mod, "run_judge", make_checkpoint_stub(comparison))

        # Judge is anthropic — same as gen_a
        cfg = _make_config()
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

        result = council_mod.council_research(
            "test", config=cfg, generator_slots=gen_slots
        )

        assert result.council_provider_compromise is True

    def test_no_compromise_when_providers_differ(self, monkeypatch):
        council_mod = _load_council_module(monkeypatch)

        pkg = InvestigationPackage(
            metadata=RunMetadata(
                run_id="run-x", tier="standard", started_at="2024-01-01T00:00:00Z"
            ),
            brief=_BRIEF,
            plan=_PLAN,
            ledger=EvidenceLedger(items=[]),
        )

        monkeypatch.setattr(council_mod, "deep_research", lambda q, **kw: pkg)

        comparison = CouncilComparison(
            comparison="A wins",
            generator_scores={"gen_a": 0.9},
            recommended_generator="gen_a",
        )
        monkeypatch.setattr(council_mod, "run_judge", make_checkpoint_stub(comparison))

        # Judge is test-d (default), generators are anthropic and openai — no overlap
        cfg = _make_config()
        gen_slots = {
            "gen_a": ModelSlotConfig(provider="anthropic", model="a"),
            "gen_b": ModelSlotConfig(provider="openai", model="b"),
        }

        result = council_mod.council_research(
            "test", config=cfg, generator_slots=gen_slots
        )

        assert result.council_provider_compromise is False


class TestBudgetExhaustionPath:
    """Budget already exceeded — convergence triggers budget_exhausted stop."""

    def test_stop_reason_is_budget_exhausted(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        # Budget is already exceeded (spent > soft)
        cfg = _make_config(
            max_iterations=5,
            soft_budget_usd=0.01,
            spent_usd=0.02,
        )
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.metadata.stop_reason == "budget_exhausted"

    def test_downstream_phases_still_run(self, monkeypatch):
        """Budget exhaustion exits iteration loop but draft/critique/finalize/assemble still run."""
        flow_mod = _load_flow_module(monkeypatch)

        draft_calls = [0]
        critique_calls = [0]
        finalize_calls = [0]
        assemble_calls = [0]

        def track_draft(*args, **kwargs):
            draft_calls[0] += 1
            return _DRAFT

        with_submit(track_draft)

        def track_critique(*args, **kwargs):
            critique_calls[0] += 1
            return _CRITIQUE_OK

        with_submit(track_critique)

        def track_finalize(*args, **kwargs):
            finalize_calls[0] += 1
            return _FINAL

        with_submit(track_finalize)

        def track_assemble(**kwargs):
            assemble_calls[0] += 1
            return InvestigationPackage(
                metadata=kwargs["metadata"],
                brief=kwargs["brief"],
                plan=kwargs["plan"],
                ledger=kwargs["ledger"],
                iterations=kwargs["iterations"],
                draft=kwargs["draft"],
                critique=kwargs["critique"],
                final_report=kwargs["final_report"],
            )

        with_submit(track_assemble)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_draft", track_draft)
        monkeypatch.setattr(flow_mod, "run_critique", track_critique)
        monkeypatch.setattr(flow_mod, "run_finalize", track_finalize)
        monkeypatch.setattr(flow_mod, "assemble_package", track_assemble)

        cfg = _make_config(
            max_iterations=5,
            soft_budget_usd=0.01,
            spent_usd=0.02,
        )
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.metadata.stop_reason == "budget_exhausted"
        assert draft_calls[0] == 1
        assert critique_calls[0] == 1
        assert finalize_calls[0] == 1
        assert assemble_calls[0] == 1


class TestSupervisorMaxIterationsStop:
    """Supervisor never says done — max_iterations cap stops the loop."""

    def test_stops_after_max_iterations(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        call_count = [0]

        def never_done(*args, **kwargs):
            call_count[0] += 1
            return _DECISION_CONTINUE

        with_submit(never_done)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=never_done)

        cfg = _make_config(max_iterations=2)
        result = flow_mod.deep_research("test question", config=cfg)

        assert call_count[0] == 2
        assert result.metadata.total_iterations == 2

    def test_stop_reason_is_max_iterations(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        def never_done(*args, **kwargs):
            return _DECISION_CONTINUE

        with_submit(never_done)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=never_done)

        cfg = _make_config(max_iterations=2)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.metadata.stop_reason == "max_iterations"

    def test_still_produces_final_report(self, monkeypatch):
        """Even when max_iterations is reached, downstream phases run and produce a report."""
        flow_mod = _load_flow_module(monkeypatch)

        def never_done(*args, **kwargs):
            return _DECISION_CONTINUE

        with_submit(never_done)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=never_done)

        cfg = _make_config(max_iterations=2)
        result = flow_mod.deep_research("test question", config=cfg)

        assert result.final_report is not None
        assert result.draft is not None
        assert result.critique is not None
