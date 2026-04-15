"""Tests for V2 default flow orchestration.

Uses stub injection for kitaru (@flow, @checkpoint, wait) and
monkeypatching of checkpoint functions on the flow module.
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
from research.contracts.decisions import SubagentFindings, SupervisorDecision
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import InvestigationPackage, RunMetadata
from research.contracts.plan import ResearchPlan, SubagentTask
from research.contracts.reports import (
    CritiqueDimensionScore,
    CritiqueReport,
    DraftReport,
    FinalReport,
)


# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------

# Module paths that must be purged before re-importing the flow module.
_FLOW_MODULES = [
    "research.flows.deep_research",
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
]


def _install_stubs(monkeypatch):
    """Install kitaru + pydantic_ai stubs for flow tests.

    @flow → passthrough decorator.
    @checkpoint → passthrough; also gives each function a .submit() method.
    wait() → no-op.
    """

    def flow_decorator(fn=None, **kwargs):
        """Stub @flow that auto-materialises returned submit handles.

        In the real Kitaru runtime, `.wait()` extracts the value from the
        terminal step's OutputArtifact.  Our stub mirrors this by calling
        `.load()` when the flow returns an object with that attribute.
        """

        def _wrap(f):
            def wrapper(*args, **kw):
                result = f(*args, **kw)
                # Simulate Kitaru .wait() extraction: if the flow returned
                # a submit handle (SimpleNamespace with .load), materialise it.
                if hasattr(result, "load") and callable(result.load):
                    return result.load()
                return result

            wrapper.__name__ = getattr(f, "__name__", "flow")
            wrapper.__qualname__ = getattr(f, "__qualname__", "flow")
            wrapper.__module__ = getattr(f, "__module__", __name__)
            return wrapper

        if fn is not None:
            return _wrap(fn)

        def decorator(f):
            return _wrap(f)

        return decorator

    def checkpoint_decorator(fn=None, *, type=None):
        """Checkpoint stub that adds .submit() support."""

        def _add_submit(f):
            def submit(*args, **kw):
                kw.pop("after", None)  # strip DAG-edge param before forwarding
                kw.pop("id", None)
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
    """Install stubs, clear caches, and return the flow module."""
    _install_stubs(monkeypatch)
    _clear_modules()
    return importlib.import_module("research.flows.deep_research")


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
    Strips ``after=`` and ``id=`` kwargs before forwarding to the function.
    """

    def _submit(*a, **kw):
        kw.pop("after", None)
        kw.pop("id", None)
        return types.SimpleNamespace(load=lambda: fn(*a, **kw))

    fn.submit = _submit
    return fn


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_iterations=5,
    max_parallel_subagents=3,
    max_supplemental_loops=1,
) -> ResearchConfig:
    """Build a minimal ResearchConfig for testing."""
    slots = {
        "generator": ModelSlotConfig(provider="test", model="gen"),
        "subagent": ModelSlotConfig(provider="test", model="sub"),
        "reviewer": ModelSlotConfig(provider="test", model="rev"),
    }
    return ResearchConfig(
        tier="standard",
        budget=BudgetConfig(soft_budget_usd=1.0),
        slots=slots,
        max_iterations=max_iterations,
        max_parallel_subagents=max_parallel_subagents,
        ledger_window_iterations=3,
        grounding_min_ratio=0.0,
        max_supplemental_loops=max_supplemental_loops,
        wait_timeout_seconds=3600,
        allow_unfinalized_package=False,
        strict_unknown_model_cost=False,
        sandbox_enabled=False,
        sandbox_backend=None,
        enabled_providers=["arxiv"],
    )


# Canonical return values for each checkpoint.
_STAMP = types.SimpleNamespace(run_id="run-test-001", started_at="2024-01-01T00:00:00Z")
_CLOCK = types.SimpleNamespace(now_iso="2024-01-01T00:00:10Z", elapsed_seconds=10)
_FINALIZATION = types.SimpleNamespace(
    completed_at="2024-01-01T00:01:00Z", elapsed_seconds=60
)

_BRIEF = ResearchBrief(topic="test topic", raw_request="test question")
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
        SubagentTask(task_description="search more", target_subtopic="t"),
    ],
)
_FINDINGS = SubagentFindings(findings=["found X"], source_references=["ref1"])
_DRAFT = DraftReport(content="# Report", sections=["Report"])
_CRITIQUE_OK = CritiqueReport(
    dimensions=[
        CritiqueDimensionScore(dimension="completeness", score=0.8, explanation="good")
    ],
    require_more_research=False,
    issues=[],
)
_CRITIQUE_MORE = CritiqueReport(
    dimensions=[
        CritiqueDimensionScore(dimension="completeness", score=0.5, explanation="gaps")
    ],
    require_more_research=True,
    issues=["needs more evidence"],
)
_FINAL = FinalReport(content="# Final", sections=["Final"])


def _make_passthrough_assemble():
    """Create an assemble_package stub that builds a real InvestigationPackage from args."""

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
        strict_grounding=False,
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
        )

    def _submit_assemble(*a, **kw):
        kw.pop("after", None)
        kw.pop("id", None)
        return types.SimpleNamespace(load=lambda: assemble_stub(*a, **kw))

    assemble_stub.submit = _submit_assemble
    return assemble_stub


def _patch_all_checkpoints(
    monkeypatch, flow_mod, *, supervisor_stub=None, critique_stub=None
):
    """Monkeypatch all checkpoints on the flow module with defaults."""
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
    monkeypatch.setattr(flow_mod, "run_subagent", make_checkpoint_stub(_FINDINGS))
    monkeypatch.setattr(flow_mod, "run_draft", make_checkpoint_stub(_DRAFT))
    monkeypatch.setattr(
        flow_mod,
        "run_critique",
        critique_stub or make_checkpoint_stub(_CRITIQUE_OK),
    )
    monkeypatch.setattr(flow_mod, "run_finalize", make_checkpoint_stub(_FINAL))
    monkeypatch.setattr(flow_mod, "assemble_package", _make_passthrough_assemble())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPathOneIteration:
    """Flow completes in a single iteration when supervisor says done."""

    def test_returns_investigation_package(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)
        assert isinstance(result, InvestigationPackage)

    def test_calls_scope_and_plan(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        scope_calls = []
        plan_calls = []

        def track_scope(*args, **kwargs):
            scope_calls.append(args)
            return _BRIEF

        with_submit(track_scope)

        def track_plan(*args, **kwargs):
            plan_calls.append(args)
            return _PLAN

        with_submit(track_plan)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_scope", track_scope)
        monkeypatch.setattr(flow_mod, "run_plan", track_plan)

        cfg = _make_config()
        flow_mod.deep_research("my question", config=cfg)

        assert len(scope_calls) == 1
        assert scope_calls[0][0] == "my question"
        assert len(plan_calls) == 1

    def test_single_iteration_when_supervisor_done(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        supervisor_calls = []

        def track_supervisor(*args, **kwargs):
            supervisor_calls.append(args)
            return _DECISION_DONE

        with_submit(track_supervisor)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=track_supervisor)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test", config=cfg)

        # Supervisor "done" triggers stop after iteration 0
        assert len(supervisor_calls) == 1
        assert result.metadata.total_iterations == 1


class TestMultiIterationLoop:
    """Flow runs multiple iterations before convergence."""

    def test_multiple_iterations_until_done(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        call_count = [0]

        def counting_supervisor(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                return _DECISION_DONE
            return _DECISION_CONTINUE

        with_submit(counting_supervisor)

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=counting_supervisor
        )

        cfg = _make_config(max_iterations=10)
        result = flow_mod.deep_research("test", config=cfg)

        assert call_count[0] == 3
        assert result.metadata.total_iterations == 3

    def test_stops_at_max_iterations(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        call_count = [0]

        def never_done(*args, **kwargs):
            call_count[0] += 1
            return _DECISION_CONTINUE

        with_submit(never_done)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=never_done)

        cfg = _make_config(max_iterations=3)
        result = flow_mod.deep_research("test", config=cfg)

        assert call_count[0] == 3
        assert result.metadata.total_iterations == 3
        assert result.metadata.stop_reason == "max_iterations"

    def test_stop_reason_recorded(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test", config=cfg)
        # Supervisor says done => stop_reason is supervisor_done
        assert result.metadata.stop_reason == "supervisor_done"


class TestSupplementalLoop:
    """Critique triggers supplemental research, capped at max_supplemental_loops."""

    def test_supplemental_triggered_and_capped(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        critique_calls = [0]

        def critique_stub(*args, **kwargs):
            critique_calls[0] += 1
            # Always request more research
            return _CRITIQUE_MORE

        with_submit(critique_stub)

        supervisor_calls = [0]

        def counting_supervisor(*args, **kwargs):
            supervisor_calls[0] += 1
            return _DECISION_DONE

        with_submit(counting_supervisor)

        _patch_all_checkpoints(
            monkeypatch,
            flow_mod,
            supervisor_stub=counting_supervisor,
            critique_stub=critique_stub,
        )

        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test", config=cfg)

        # 1 main iteration + 1 supplemental = 2 total iterations
        assert result.metadata.total_iterations == 2
        # Critique called twice: once after main draft, once after supplemental draft
        assert critique_calls[0] == 2
        # Supervisor called twice: main iteration + supplemental iteration
        assert supervisor_calls[0] == 2

    def test_no_supplemental_when_critique_ok(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        supervisor_calls = [0]

        def counting_supervisor(*args, **kwargs):
            supervisor_calls[0] += 1
            return _DECISION_DONE

        with_submit(counting_supervisor)

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=counting_supervisor
        )

        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test", config=cfg)

        # Only 1 main iteration, no supplemental
        assert result.metadata.total_iterations == 1
        assert supervisor_calls[0] == 1

    def test_supplemental_capped_at_max(self, monkeypatch):
        """Even if critique keeps requesting more, supplemental loops are capped."""
        flow_mod = _load_flow_module(monkeypatch)

        critique_calls = [0]

        def always_more(*args, **kwargs):
            critique_calls[0] += 1
            return _CRITIQUE_MORE

        with_submit(always_more)

        _patch_all_checkpoints(monkeypatch, flow_mod, critique_stub=always_more)

        # max_supplemental_loops=1 means at most 1 extra loop
        cfg = _make_config(max_iterations=5, max_supplemental_loops=1)
        result = flow_mod.deep_research("test", config=cfg)

        # critique called: 1 (initial) + 1 (supplemental) = 2
        assert critique_calls[0] == 2
        # second require_more_research is recorded in critique but loop exits


class TestSubagentFanOut:
    """Subagent fan-out respects max_parallel_subagents concurrency ceiling."""

    def test_batching_respects_concurrency(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        # Track .submit() calls per batch
        submit_batches: list[list] = []
        current_batch: list[str] = []

        def subagent_stub(*args, **kwargs):
            return _FINDINGS

        def subagent_submit(*args, **kwargs):
            current_batch.append(args[0].task_description if args else "unknown")
            return types.SimpleNamespace(load=lambda: _FINDINGS)

        subagent_stub.submit = subagent_submit

        # Supervisor returns 5 tasks — should be batched at max_parallel=2
        five_tasks = SupervisorDecision(
            done=True,
            rationale="done",
            subagent_tasks=[
                SubagentTask(task_description=f"task-{i}", target_subtopic="t")
                for i in range(5)
            ],
        )

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=make_checkpoint_stub(five_tasks)
        )
        monkeypatch.setattr(flow_mod, "run_subagent", subagent_stub)

        cfg = _make_config(max_iterations=5, max_parallel_subagents=2)
        result = flow_mod.deep_research("test", config=cfg)

        # All 5 tasks should have been submitted
        assert len(current_batch) == 5

    def test_single_batch_when_tasks_within_limit(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        submit_count = [0]

        def subagent_stub(*args, **kwargs):
            return _FINDINGS

        def subagent_submit(*args, **kwargs):
            submit_count[0] += 1
            return types.SimpleNamespace(load=lambda: _FINDINGS)

        subagent_stub.submit = subagent_submit

        two_tasks = SupervisorDecision(
            done=True,
            rationale="done",
            subagent_tasks=[
                SubagentTask(task_description=f"task-{i}", target_subtopic="t")
                for i in range(2)
            ],
        )

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=make_checkpoint_stub(two_tasks)
        )
        monkeypatch.setattr(flow_mod, "run_subagent", subagent_stub)

        cfg = _make_config(max_iterations=5, max_parallel_subagents=3)
        flow_mod.deep_research("test", config=cfg)

        # 2 tasks, max_parallel=3 → single batch of 2
        assert submit_count[0] == 2

    def test_zero_tasks_no_fan_out(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        submit_count = [0]

        def subagent_stub(*args, **kwargs):
            return _FINDINGS

        def subagent_submit(*args, **kwargs):
            submit_count[0] += 1
            return types.SimpleNamespace(load=lambda: _FINDINGS)

        subagent_stub.submit = subagent_submit

        no_tasks = SupervisorDecision(done=True, rationale="done", subagent_tasks=[])

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=make_checkpoint_stub(no_tasks)
        )
        monkeypatch.setattr(flow_mod, "run_subagent", subagent_stub)

        cfg = _make_config(max_iterations=5)
        flow_mod.deep_research("test", config=cfg)

        assert submit_count[0] == 0


class TestFlowConfigDefaults:
    """Flow correctly uses config and tier defaults."""

    def test_uses_provided_config(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)
        _patch_all_checkpoints(monkeypatch, flow_mod)

        cfg = _make_config(max_iterations=1)
        result = flow_mod.deep_research("test", config=cfg)
        assert result.metadata.tier == "standard"

    def test_passes_model_strings_to_checkpoints(self, monkeypatch):
        flow_mod = _load_flow_module(monkeypatch)

        scope_models = []

        def track_scope(*args, **kwargs):
            scope_models.append(args[1] if len(args) > 1 else kwargs.get("model_name"))
            return _BRIEF

        with_submit(track_scope)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_scope", track_scope)

        cfg = _make_config()
        flow_mod.deep_research("test", config=cfg)

        assert scope_models[0] == "test:gen"


# ---------------------------------------------------------------------------
# Failure semantics tests
# ---------------------------------------------------------------------------


def _make_failing_then_succeeding_stub(fail_count, success_value):
    """Stub that fails `fail_count` times then returns success_value."""
    calls = [0]

    def stub(*args, **kwargs):
        calls[0] += 1
        if calls[0] <= fail_count:
            raise RuntimeError(f"Simulated failure #{calls[0]}")
        return success_value

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: stub(*a, **kw))
    return stub, calls


def _make_always_failing_stub():
    """Stub that always raises RuntimeError."""
    calls = [0]

    def stub(*args, **kwargs):
        calls[0] += 1
        raise RuntimeError(f"Simulated failure #{calls[0]}")

    stub.submit = lambda *a, **kw: types.SimpleNamespace(load=lambda: stub(*a, **kw))
    return stub, calls


class TestSupervisorRetry:
    """Supervisor gets one retry on failure; second failure raises SupervisorError."""

    def test_supervisor_retry_on_first_failure(self, monkeypatch):
        """Supervisor fails once, retries, succeeds on second call."""
        flow_mod = _load_flow_module(monkeypatch)

        supervisor_stub, calls = _make_failing_then_succeeding_stub(
            fail_count=1, success_value=_DECISION_DONE
        )

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor_stub)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        # Should succeed — retry worked
        assert isinstance(result, InvestigationPackage)
        # Supervisor was called twice (first fail + retry success)
        assert calls[0] == 2

    def test_supervisor_double_failure_raises(self, monkeypatch):
        """Supervisor fails twice → SupervisorError with ledger preserved."""
        flow_mod = _load_flow_module(monkeypatch)

        supervisor_stub, calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor_stub)

        cfg = _make_config(max_iterations=5)

        with pytest.raises(flow_mod.SupervisorError) as exc_info:
            flow_mod.deep_research("test question", config=cfg)

        # Two attempts were made
        assert calls[0] == 2
        # Error carries the ledger
        assert exc_info.value.ledger is not None
        assert isinstance(exc_info.value.ledger, EvidenceLedger)


class TestFinalizerFailure:
    """Finalizer failure handling based on allow_unfinalized_package config."""

    def test_finalizer_failure_with_allow_unfinalized(self, monkeypatch):
        """run_finalize raises → allow_unfinalized_package=True → package with final_report=None."""
        flow_mod = _load_flow_module(monkeypatch)

        finalizer_stub, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_finalize", finalizer_stub)

        cfg = _make_config()
        # Need to override allow_unfinalized_package — ResearchConfig is frozen,
        # so we rebuild with the flag set.
        cfg = ResearchConfig(
            tier=cfg.tier,
            budget=cfg.budget,
            slots=cfg.slots,
            max_iterations=cfg.max_iterations,
            max_parallel_subagents=cfg.max_parallel_subagents,
            ledger_window_iterations=cfg.ledger_window_iterations,
            grounding_min_ratio=cfg.grounding_min_ratio,
            max_supplemental_loops=cfg.max_supplemental_loops,
            wait_timeout_seconds=cfg.wait_timeout_seconds,
            allow_unfinalized_package=True,
            strict_unknown_model_cost=cfg.strict_unknown_model_cost,
            sandbox_enabled=cfg.sandbox_enabled,
            sandbox_backend=cfg.sandbox_backend,
            enabled_providers=cfg.enabled_providers,
        )

        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.final_report is None
        # stop_reason preserves the iteration loop's reason; finalizer_failed
        # only appears when there was no prior stop reason.
        assert result.metadata.stop_reason == "supervisor_done"
        # Draft and critique should still be present
        assert result.draft is not None
        assert result.critique is not None

    def test_finalizer_failure_without_allow_raises(self, monkeypatch):
        """run_finalize raises → allow_unfinalized_package=False → FinalizerError."""
        flow_mod = _load_flow_module(monkeypatch)

        finalizer_stub, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_finalize", finalizer_stub)

        cfg = _make_config()  # allow_unfinalized_package defaults to False

        with pytest.raises(flow_mod.FinalizerError):
            flow_mod.deep_research("test question", config=cfg)

    def test_finalizer_returns_none_with_allow_unfinalized(self, monkeypatch):
        """run_finalize returns None directly → allow_unfinalized_package=True → package produced."""
        flow_mod = _load_flow_module(monkeypatch)

        none_stub = make_checkpoint_stub(None)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_finalize", none_stub)

        cfg = ResearchConfig(
            tier="standard",
            budget=BudgetConfig(soft_budget_usd=1.0),
            slots={
                "generator": ModelSlotConfig(provider="test", model="gen"),
                "subagent": ModelSlotConfig(provider="test", model="sub"),
                "reviewer": ModelSlotConfig(provider="test", model="rev"),
            },
            max_iterations=5,
            max_parallel_subagents=3,
            ledger_window_iterations=3,
            grounding_min_ratio=0.0,
            max_supplemental_loops=1,
            wait_timeout_seconds=3600,
            allow_unfinalized_package=True,
            strict_unknown_model_cost=False,
            sandbox_enabled=False,
            sandbox_backend=None,
            enabled_providers=["arxiv"],
        )

        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.final_report is None
        # stop_reason preserves the iteration loop's reason when one exists
        assert result.metadata.stop_reason == "supervisor_done"

    def test_finalizer_returns_none_without_allow_raises(self, monkeypatch):
        """run_finalize returns None → allow_unfinalized_package=False → FinalizerError."""
        flow_mod = _load_flow_module(monkeypatch)

        none_stub = make_checkpoint_stub(None)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_finalize", none_stub)

        cfg = _make_config()  # allow_unfinalized_package defaults to False

        with pytest.raises(flow_mod.FinalizerError):
            flow_mod.deep_research("test question", config=cfg)

    def test_finalizer_failed_sets_stop_reason_when_no_prior(self, monkeypatch):
        """When no iteration stop_reason exists, finalizer_failed becomes stop_reason."""
        flow_mod = _load_flow_module(monkeypatch)

        finalizer_stub, _calls = _make_always_failing_stub()

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_finalize", finalizer_stub)

        # max_iterations=0 → no iterations run → stop_reason stays None
        cfg = ResearchConfig(
            tier="standard",
            budget=BudgetConfig(soft_budget_usd=1.0),
            slots={
                "generator": ModelSlotConfig(provider="test", model="gen"),
                "subagent": ModelSlotConfig(provider="test", model="sub"),
                "reviewer": ModelSlotConfig(provider="test", model="rev"),
            },
            max_iterations=0,
            max_parallel_subagents=3,
            ledger_window_iterations=3,
            grounding_min_ratio=0.0,
            max_supplemental_loops=1,
            wait_timeout_seconds=3600,
            allow_unfinalized_package=True,
            strict_unknown_model_cost=False,
            sandbox_enabled=False,
            sandbox_backend=None,
            enabled_providers=["arxiv"],
        )

        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.final_report is None
        assert result.metadata.stop_reason == "finalizer_failed"


class TestBudgetExhaustion:
    """Budget exhaustion sets stop_reason but downstream phases still run."""

    def test_budget_exhaustion_still_produces_deliverable(self, monkeypatch):
        """Budget exceeded after iteration → stop_reason set, deliverable still produced."""
        flow_mod = _load_flow_module(monkeypatch)

        # Track that downstream checkpoints are actually called
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

        # Create config with budget already exhausted
        budget = BudgetConfig(soft_budget_usd=0.01, spent_usd=0.02)
        cfg = ResearchConfig(
            tier="standard",
            budget=budget,
            slots={
                "generator": ModelSlotConfig(provider="test", model="gen"),
                "subagent": ModelSlotConfig(provider="test", model="sub"),
                "reviewer": ModelSlotConfig(provider="test", model="rev"),
            },
            max_iterations=5,
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

        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        assert result.metadata.stop_reason == "budget_exhausted"
        # All downstream phases ran
        assert draft_calls[0] == 1
        assert critique_calls[0] == 1
        assert finalize_calls[0] == 1
        assert assemble_calls[0] == 1


class TestSubagentDegradedResult:
    """Subagent degraded results flow through the iteration record."""

    def test_subagent_degraded_result_in_iteration(self, monkeypatch):
        """Subagent returns degraded (empty) findings — still recorded in iteration."""
        flow_mod = _load_flow_module(monkeypatch)

        degraded_findings = SubagentFindings(
            findings=[],
            source_references=[],
            confidence_notes="Subagent failed: simulated error",
        )

        degraded_stub = make_checkpoint_stub(degraded_findings)

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_subagent", degraded_stub)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test question", config=cfg)

        assert isinstance(result, InvestigationPackage)
        # The degraded subagent result is present in iteration records
        assert len(result.iterations) >= 1
        first_iter = result.iterations[0]
        for sa_result in first_iter.subagent_results:
            assert sa_result.findings == []
            assert "Subagent failed" in (sa_result.confidence_notes or "")


class TestToolSurfaceThreading:
    """Tools are threaded from the flow through to subagent submissions."""

    def test_subagents_receive_tools(self, monkeypatch):
        """run_subagent.submit() receives tools kwarg from the flow."""
        flow_mod = _load_flow_module(monkeypatch)

        received_tools: list = []

        def subagent_stub(*args, **kwargs):
            return _FINDINGS

        def subagent_submit(*args, **kwargs):
            kwargs.pop("after", None)
            kwargs.pop("id", None)
            received_tools.append(kwargs.get("tools"))
            return types.SimpleNamespace(load=lambda: _FINDINGS)

        subagent_stub.submit = subagent_submit

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_subagent", subagent_stub)

        # Patch ProviderRegistry and build_tool_surface to avoid real providers
        fake_tools = [lambda: "search", lambda: "fetch"]
        monkeypatch.setattr(
            flow_mod,
            "ProviderRegistry",
            lambda cfg: None,
        )
        monkeypatch.setattr(
            flow_mod,
            "build_tool_surface",
            lambda cfg, registry: types.SimpleNamespace(
                as_pydantic_tools=lambda: fake_tools
            ),
        )

        cfg = _make_config(max_iterations=5)
        flow_mod.deep_research("test", config=cfg)

        # At least one subagent should have received tools
        assert len(received_tools) >= 1
        assert received_tools[0] is fake_tools

    def test_tool_surface_failure_degrades_gracefully(self, monkeypatch):
        """If tool surface construction fails, subagents get tools=None."""
        flow_mod = _load_flow_module(monkeypatch)

        received_tools: list = []

        def subagent_stub(*args, **kwargs):
            return _FINDINGS

        def subagent_submit(*args, **kwargs):
            kwargs.pop("after", None)
            kwargs.pop("id", None)
            received_tools.append(kwargs.get("tools"))
            return types.SimpleNamespace(load=lambda: _FINDINGS)

        subagent_stub.submit = subagent_submit

        _patch_all_checkpoints(monkeypatch, flow_mod)
        monkeypatch.setattr(flow_mod, "run_subagent", subagent_stub)

        # Make ProviderRegistry raise to simulate failure
        def failing_registry(cfg):
            raise RuntimeError("No providers available")

        monkeypatch.setattr(flow_mod, "ProviderRegistry", failing_registry)

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test", config=cfg)

        # Flow should still complete
        assert isinstance(result, InvestigationPackage)
        # Subagents should have received None for tools
        assert all(t is None for t in received_tools)


class TestPinnedIdsPropagate:
    """Pinned evidence IDs from supervisor decisions propagate to the next projection."""

    def test_pinned_ids_from_previous_decision_passed_to_projection(self, monkeypatch):
        """Supervisor's pinned_evidence_ids affect the next iteration's projection."""
        flow_mod = _load_flow_module(monkeypatch)

        projection_calls: list[str] = []
        original_format = flow_mod.format_projection
        original_project = flow_mod.project_ledger

        call_count = [0]
        pinned_ids_received: list = []

        def tracking_project(ledger, iteration_index, pinned_ids=None, **kwargs):
            pinned_ids_received.append(pinned_ids or [])
            return original_project(ledger, iteration_index, pinned_ids=pinned_ids, **kwargs)

        monkeypatch.setattr(flow_mod, "project_ledger", tracking_project)

        # Supervisor returns pinned IDs on first call, then stops on second
        decision_with_pins = SupervisorDecision(
            done=False,
            rationale="continuing with pins",
            gaps=["gap1"],
            subagent_tasks=[
                SubagentTask(task_description="search", target_subtopic="t"),
            ],
            pinned_evidence_ids=["ev_important_001", "ev_important_002"],
        )

        def supervisor_with_pins(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return decision_with_pins
            return _DECISION_DONE

        with_submit(supervisor_with_pins)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=supervisor_with_pins)

        # Patch out tool surface to avoid real providers
        monkeypatch.setattr(
            flow_mod, "ProviderRegistry", lambda cfg: None
        )
        monkeypatch.setattr(
            flow_mod, "build_tool_surface",
            lambda cfg, registry: types.SimpleNamespace(as_pydantic_tools=lambda: []),
        )

        cfg = _make_config(max_iterations=5)
        flow_mod.deep_research("test", config=cfg)

        # First iteration should have empty pinned_ids (no prior decision)
        assert pinned_ids_received[0] == []
        # Second iteration should have the pins from the first decision
        assert pinned_ids_received[1] == ["ev_important_001", "ev_important_002"]


class TestStalePinnedIdsIgnored:
    """Stale pinned evidence IDs (not in ledger) are safely ignored."""

    def test_stale_pinned_ids_do_not_crash_projection(self, monkeypatch):
        """Supervisor returns pinned IDs that don't exist in the ledger — flow still completes."""
        flow_mod = _load_flow_module(monkeypatch)

        call_count = [0]
        pinned_ids_received: list = []

        original_project = flow_mod.project_ledger

        def tracking_project(ledger, iteration_index, pinned_ids=None, **kwargs):
            pinned_ids_received.append(pinned_ids or [])
            return original_project(
                ledger, iteration_index, pinned_ids=pinned_ids, **kwargs
            )

        monkeypatch.setattr(flow_mod, "project_ledger", tracking_project)

        # Supervisor returns stale pinned IDs on first call
        decision_with_stale_pins = SupervisorDecision(
            done=False,
            rationale="continuing",
            gaps=["gap"],
            subagent_tasks=[
                SubagentTask(task_description="search", target_subtopic="t"),
            ],
            pinned_evidence_ids=["nonexistent_001", "nonexistent_002"],
        )

        def supervisor_stale_pins(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return decision_with_stale_pins
            return _DECISION_DONE

        with_submit(supervisor_stale_pins)

        _patch_all_checkpoints(
            monkeypatch, flow_mod, supervisor_stub=supervisor_stale_pins
        )

        monkeypatch.setattr(flow_mod, "ProviderRegistry", lambda cfg: None)
        monkeypatch.setattr(
            flow_mod,
            "build_tool_surface",
            lambda cfg, registry: types.SimpleNamespace(as_pydantic_tools=lambda: []),
        )

        cfg = _make_config(max_iterations=5)
        result = flow_mod.deep_research("test", config=cfg)

        # Flow completes without error
        assert isinstance(result, InvestigationPackage)
        # Second iteration received the stale IDs — they're just ignored by projection
        assert pinned_ids_received[1] == ["nonexistent_001", "nonexistent_002"]


class TestSupervisorContextEnrichment:
    """Supervisor receives max_iterations and ledger_size in its prompt data."""

    def test_supervisor_receives_max_iterations_and_ledger_size(self, monkeypatch):
        """run_supervisor is called with max_iterations and ledger_size kwargs."""
        flow_mod = _load_flow_module(monkeypatch)

        supervisor_kwargs: list[dict] = []

        def tracking_supervisor(*args, **kwargs):
            supervisor_kwargs.append(kwargs)
            return _DECISION_DONE

        with_submit(tracking_supervisor)

        _patch_all_checkpoints(monkeypatch, flow_mod, supervisor_stub=tracking_supervisor)

        # Patch out tool surface
        monkeypatch.setattr(flow_mod, "ProviderRegistry", lambda cfg: None)
        monkeypatch.setattr(
            flow_mod, "build_tool_surface",
            lambda cfg, registry: types.SimpleNamespace(as_pydantic_tools=lambda: []),
        )

        cfg = _make_config(max_iterations=7)
        flow_mod.deep_research("test", config=cfg)

        assert len(supervisor_kwargs) >= 1
        kw = supervisor_kwargs[0]
        assert kw.get("max_iterations") == 7
        assert "ledger_size" in kw
