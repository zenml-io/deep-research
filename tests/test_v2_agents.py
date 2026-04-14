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
