"""Tests for V2 agent wrapping infrastructure.

Uses stub injection because kitaru and pydantic_ai may not be directly
importable in the test environment (Python 3.14 / zenml compatibility).
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


def _install_stubs(monkeypatch):
    """Install lightweight pydantic_ai and kitaru stubs into sys.modules.

    Returns the list that records every ``kp.wrap()`` call so tests can
    inspect what was passed.
    """
    wrap_calls: list[dict] = []

    class FakeAgent:
        """Minimal stand-in for ``pydantic_ai.Agent``."""

        def __init__(self, model_name: str = "test", **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    def wrap(agent, *, tool_capture_config=None, name=None):
        record = {
            "agent": agent,
            "tool_capture_config": tool_capture_config,
            "name": name,
        }
        wrap_calls.append(record)
        return record  # return the dict so tests can verify passthrough

    kp_ns = types.SimpleNamespace(wrap=wrap)

    monkeypatch.setitem(
        sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "kitaru",
        types.SimpleNamespace(adapters=types.SimpleNamespace(pydantic_ai=kp_ns)),
    )
    monkeypatch.setitem(
        sys.modules, "kitaru.adapters", types.SimpleNamespace(pydantic_ai=kp_ns)
    )
    monkeypatch.setitem(sys.modules, "kitaru.adapters.pydantic_ai", kp_ns)

    return wrap_calls, FakeAgent


def _load_wrap_module():
    """Re-import ``research.agents._wrap`` from scratch."""
    _clear_modules("research.agents._wrap", "research.agents")
    return importlib.import_module("research.agents._wrap")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWrapAgent:
    """Unit tests for ``wrap_agent``."""

    def test_calls_kp_wrap_with_correct_args(self, monkeypatch):
        """wrap_agent passes tool_capture_config={"mode": "full"} to kp.wrap."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        mod = _load_wrap_module()

        agent = FakeAgent("google-gla:gemini-2.5-flash")
        mod.wrap_agent(agent)

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["agent"] is agent
        assert wrap_calls[0]["tool_capture_config"] == {"mode": "full"}

    def test_passes_name_through(self, monkeypatch):
        """wrap_agent forwards the name keyword to kp.wrap."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        mod = _load_wrap_module()

        agent = FakeAgent("test-model")
        mod.wrap_agent(agent, name="supervisor")

        assert wrap_calls[0]["name"] == "supervisor"

    def test_name_defaults_to_none(self, monkeypatch):
        """When name is omitted, kp.wrap receives None."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        mod = _load_wrap_module()

        agent = FakeAgent("test-model")
        mod.wrap_agent(agent)

        assert wrap_calls[0]["name"] is None

    def test_returns_kp_wrap_result(self, monkeypatch):
        """wrap_agent returns exactly what kp.wrap returns (no alteration)."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        mod = _load_wrap_module()

        agent = FakeAgent("test-model")
        result = mod.wrap_agent(agent, name="writer")

        # Our stub's wrap() returns the record dict
        assert result is wrap_calls[0]
        assert result["agent"] is agent
        assert result["name"] == "writer"

    def test_multiple_wraps_are_independent(self, monkeypatch):
        """Each wrap_agent call creates an independent wrapped agent."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        mod = _load_wrap_module()

        agent_a = FakeAgent("model-a")
        agent_b = FakeAgent("model-b")

        result_a = mod.wrap_agent(agent_a, name="alpha")
        result_b = mod.wrap_agent(agent_b, name="beta")

        assert len(wrap_calls) == 2
        assert result_a["agent"] is agent_a
        assert result_b["agent"] is agent_b
        assert result_a["name"] == "alpha"
        assert result_b["name"] == "beta"
        assert result_a is not result_b

    def test_reexport_from_agents_package(self, monkeypatch):
        """``from research.agents import wrap_agent`` works."""
        _install_stubs(monkeypatch)
        _clear_modules("research.agents._wrap", "research.agents")

        agents_mod = importlib.import_module("research.agents")
        assert hasattr(agents_mod, "wrap_agent")
        assert callable(agents_mod.wrap_agent)


# ---------------------------------------------------------------------------
# Scope agent factory tests
# ---------------------------------------------------------------------------


class TestScopeAgent:
    """Unit tests for ``build_scope_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the scope module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.scope",
        )
        mod = importlib.import_module("research.agents.scope")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_scope_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_research_brief(self, monkeypatch):
        """Factory sets output_type=ResearchBrief on the agent."""
        from research.contracts import ResearchBrief

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_scope_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is ResearchBrief

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the scope prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_scope_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # substantive, not empty
        assert "research scoping agent" in prompt.lower()

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='scope'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_scope_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "scope"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_scope_agent("test-model")

        # Our stub wrap() returns a dict, so result should be that dict
        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Planner agent factory tests
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    """Unit tests for ``build_planner_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the planner module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.planner",
        )
        mod = importlib.import_module("research.agents.planner")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_planner_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_research_plan(self, monkeypatch):
        """Factory sets output_type=ResearchPlan on the agent."""
        from research.contracts import ResearchPlan

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_planner_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is ResearchPlan

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the planner prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_planner_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # substantive, not empty
        assert "research planner" in prompt.lower()

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='planner'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_planner_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "planner"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_planner_agent("test-model")

        # Our stub wrap() returns a dict, so result should be that dict
        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Supervisor agent factory tests
# ---------------------------------------------------------------------------


class TestSupervisorAgent:
    """Unit tests for ``build_supervisor_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the supervisor module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.supervisor",
        )
        mod = importlib.import_module("research.agents.supervisor")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_supervisor_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_supervisor_decision(self, monkeypatch):
        """Factory sets output_type=SupervisorDecision on the agent."""
        from research.contracts.decisions import SupervisorDecision

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_supervisor_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is SupervisorDecision

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the supervisor prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_supervisor_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub
        assert "research supervisor" in prompt.lower()

    def test_prompt_covers_key_concepts(self, monkeypatch):
        """Supervisor prompt addresses budget, gaps, convergence, and delegation."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_supervisor_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        # Must cover the key responsibilities
        assert "budget" in prompt
        assert "gap" in prompt
        assert "subagent" in prompt
        assert "done" in prompt
        assert "pinned" in prompt or "pin" in prompt

    def test_no_tools_passed(self, monkeypatch):
        """Supervisor agent must NOT have any tools — structural guard."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_supervisor_agent("test-model")

        agent = wrap_calls[0]["agent"]
        # FakeAgent stores all kwargs; 'tools' should not be present
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='supervisor'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_supervisor_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "supervisor"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_supervisor_agent("test-model")

        # Our stub wrap() returns a dict, so result should be that dict
        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Subagent factory tests
# ---------------------------------------------------------------------------


class TestSubagent:
    """Unit tests for ``build_subagent_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the subagent module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.subagent",
        )
        mod = importlib.import_module("research.agents.subagent")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_subagent_findings(self, monkeypatch):
        """Factory sets output_type=SubagentFindings on the agent."""
        from research.contracts.decisions import SubagentFindings

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is SubagentFindings

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the subagent prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub
        assert "research subagent" in prompt.lower()

    def test_prompt_covers_key_concepts(self, monkeypatch):
        """Subagent prompt covers search, fetch, findings, confidence, and excerpts."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        assert "search" in prompt
        assert "fetch" in prompt
        assert "findings" in prompt
        assert "confidence" in prompt
        assert "excerpt" in prompt
        assert "doi" in prompt or "arxiv" in prompt

    def test_tools_passed_through(self, monkeypatch):
        """When tools are provided, they are passed to the Agent constructor."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)

        def fake_search():
            pass

        def fake_fetch():
            pass

        tools = [fake_search, fake_fetch]
        mod.build_subagent_agent("test-model", tools=tools)

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["tools"] is tools
        assert len(agent.kwargs["tools"]) == 2

    def test_no_tools_when_none(self, monkeypatch):
        """When tools is None, no tools kwarg is passed to the Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model", tools=None)

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_no_tools_when_empty_list(self, monkeypatch):
        """When tools is an empty list, no tools kwarg is passed to the Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model", tools=[])

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='subagent'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_subagent_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "subagent"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_subagent_agent("test-model")

        # Our stub wrap() returns a dict, so result should be that dict
        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Generator agent factory tests
# ---------------------------------------------------------------------------


class TestGeneratorAgent:
    """Unit tests for ``build_generator_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the generator module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.generator",
        )
        mod = importlib.import_module("research.agents.generator")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_draft_report(self, monkeypatch):
        """Factory sets output_type=DraftReport on the agent."""
        from research.contracts.reports import DraftReport

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is DraftReport

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the generator prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub

    def test_prompt_mentions_citations(self, monkeypatch):
        """Generator prompt must mention citation discipline and [evidence_id] format."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        assert "citation" in prompt
        assert "[evidence_id]" in prompt

    def test_no_tools_passed(self, monkeypatch):
        """Generator agent must NOT have any tools — structural guard."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='generator'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_generator_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "generator"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_generator_agent("test-model")

        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Reviewer agent factory tests
# ---------------------------------------------------------------------------


class TestReviewerAgent:
    """Unit tests for ``build_reviewer_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the reviewer module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.reviewer",
        )
        mod = importlib.import_module("research.agents.reviewer")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("anthropic:claude-sonnet-4-20250514")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "anthropic:claude-sonnet-4-20250514"

    def test_output_type_is_critique_report(self, monkeypatch):
        """Factory sets output_type=CritiqueReport on the agent."""
        from research.contracts.reports import CritiqueReport

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is CritiqueReport

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the reviewer prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub

    def test_prompt_has_three_dimensions(self, monkeypatch):
        """Reviewer prompt must reference all three critique dimensions."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        assert "source_reliability" in prompt
        assert "completeness" in prompt
        assert "grounding" in prompt

    def test_no_tools_passed(self, monkeypatch):
        """Reviewer agent must NOT have any tools — structural guard."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='reviewer'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_reviewer_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "reviewer"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_reviewer_agent("test-model")

        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Finalizer agent factory tests
# ---------------------------------------------------------------------------


class TestFinalizerAgent:
    """Unit tests for ``build_finalizer_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the finalizer module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.finalizer",
        )
        mod = importlib.import_module("research.agents.finalizer")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("google-gla:gemini-2.5-flash")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "google-gla:gemini-2.5-flash"

    def test_output_type_is_final_report(self, monkeypatch):
        """Factory sets output_type=FinalReport on the agent."""
        from research.contracts.reports import FinalReport

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is FinalReport

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the finalizer prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub

    def test_prompt_mentions_critique_and_citation(self, monkeypatch):
        """Finalizer prompt must reference critique handling and citation discipline."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        assert "critique" in prompt
        assert "citation" in prompt

    def test_no_tools_passed(self, monkeypatch):
        """Finalizer agent must NOT have any tools — structural guard."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='finalizer'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_finalizer_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "finalizer"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_finalizer_agent("test-model")

        assert result is wrap_calls[0]


# ---------------------------------------------------------------------------
# Judge agent factory tests
# ---------------------------------------------------------------------------


class TestJudgeAgent:
    """Unit tests for ``build_judge_agent``."""

    def _load(self, monkeypatch):
        """Install stubs, clear module cache, and import the judge module."""
        wrap_calls, FakeAgent = _install_stubs(monkeypatch)
        _clear_modules(
            "research.agents._wrap",
            "research.agents",
            "research.agents.judge",
        )
        mod = importlib.import_module("research.agents.judge")
        return mod, wrap_calls, FakeAgent

    def test_creates_agent_with_correct_model(self, monkeypatch):
        """Factory passes the model_name to Agent."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("openai:gpt-4o-mini")

        agent = wrap_calls[0]["agent"]
        assert isinstance(agent, FakeAgent)
        assert agent.model_name == "openai:gpt-4o-mini"

    def test_output_type_is_council_comparison(self, monkeypatch):
        """Factory sets output_type=CouncilComparison on the agent."""
        from research.contracts.package import CouncilComparison

        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert agent.kwargs["output_type"] is CouncilComparison

    def test_system_prompt_loaded(self, monkeypatch):
        """Factory loads the council_judge prompt and passes it as system_prompt."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("test-model")

        agent = wrap_calls[0]["agent"]
        prompt = agent.kwargs["system_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantive, not a stub

    def test_prompt_covers_key_concepts(self, monkeypatch):
        """Judge prompt covers grounding, coherence, comparison, and objectivity."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("test-model")

        prompt = wrap_calls[0]["agent"].kwargs["system_prompt"].lower()
        assert "grounding" in prompt
        assert "coherence" in prompt
        assert "comparison" in prompt or "comparing" in prompt
        assert "completeness" in prompt
        assert "objective" in prompt

    def test_no_tools_passed(self, monkeypatch):
        """Judge agent must NOT have any tools — structural guard."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("test-model")

        agent = wrap_calls[0]["agent"]
        assert "tools" not in agent.kwargs

    def test_wrapped_with_correct_name(self, monkeypatch):
        """Factory calls wrap_agent with name='council_judge'."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        mod.build_judge_agent("test-model")

        assert len(wrap_calls) == 1
        assert wrap_calls[0]["name"] == "council_judge"

    def test_returns_wrapped_result(self, monkeypatch):
        """Factory returns the result of wrap_agent (not the raw agent)."""
        mod, wrap_calls, FakeAgent = self._load(monkeypatch)
        result = mod.build_judge_agent("test-model")

        assert result is wrap_calls[0]
