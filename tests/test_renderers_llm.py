import importlib
import json
import sys
import types

import pytest

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    RenderProse,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
    SelectionItem,
)


def _install_kitaru_checkpoint_stub(monkeypatch):
    """Install a minimal checkpoint decorator so rendering imports stay lightweight."""
    decorated = []

    def checkpoint(*, type):
        def decorator(func):
            func._checkpoint_type = type
            decorated.append((func.__name__, type))
            return func

        return decorator

    monkeypatch.setitem(
        sys.modules, "kitaru", types.SimpleNamespace(checkpoint=checkpoint)
    )
    return decorated


def _import_rendering_module(monkeypatch):
    """Import rendering checkpoints from a clean module cache under test stubs."""
    _install_kitaru_checkpoint_stub(monkeypatch)
    sys.modules.pop("deep_research.checkpoints.rendering", None)
    return importlib.import_module("deep_research.checkpoints.rendering")


def _sample_plan() -> ResearchPlan:
    """Build a compact plan fixture with enough structure for prose synthesis tests."""
    return ResearchPlan(
        goal="Explain the topic",
        key_questions=["What changed?"],
        subtopics=["overview"],
        queries=["topic overview"],
        sections=["Summary"],
        success_criteria=["Cover the basics"],
    )


def _sample_ledger() -> EvidenceLedger:
    """Build a ledger fixture with one selected source and stable citation data."""
    return EvidenceLedger(
        entries=[
            {
                "key": "source-1",
                "title": "Foundational paper",
                "url": "https://example.com/source-1",
                "provider": "arxiv",
                "source_kind": "paper",
                "selected": True,
            }
        ]
    )


def _sample_package() -> InvestigationPackage:
    """Build a minimal package fixture that exercises lazy full-report synthesis."""
    return InvestigationPackage(
        run_summary=RunSummary(
            run_id="run-1",
            brief="Explain the topic",
            tier="standard",
            stop_reason="converged",
            status="completed",
        ),
        research_plan=_sample_plan(),
        evidence_ledger=_sample_ledger(),
        selection_graph=SelectionGraph(
            items=[SelectionItem(candidate_key="source-1", rationale="Start here")]
        ),
        iteration_trace=IterationTrace(),
        renders=[],
    )


def test_render_reading_path_checkpoint_materializes_writer_prose(monkeypatch) -> None:
    module = _import_rendering_module(monkeypatch)
    materialization_module = importlib.import_module(
        "deep_research.renderers.materialization"
    )

    captured = {}

    class FakeAgent:
        def run_sync(self, prompt):
            captured["prompt"] = json.loads(prompt)
            return types.SimpleNamespace(
                output=RenderProse(content_markdown="# Reading Path\n\nUse [1].")
            )

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.writer",
        types.SimpleNamespace(build_writer_agent=lambda model_name: FakeAgent()),
    )
    monkeypatch.setattr(
        materialization_module, "load_prompt", lambda name: f"prompt:{name}"
    )

    result = module.write_reading_path(
        SelectionGraph(
            items=[SelectionItem(candidate_key="source-1", rationale="Start here")]
        ),
        _sample_ledger(),
        _sample_plan(),
        ResearchConfig.for_tier(Tier.STANDARD),
    )

    assert result.render.content_markdown.startswith("# Reading Path")
    assert result.render.citation_map == {"[1]": "source-1"}
    assert captured["prompt"]["trusted_render_guidance"] == "prompt:writer_reading_path"
    assert captured["prompt"]["trusted_context"]["render_name"] == "reading_path"
    assert captured["prompt"]["trusted_context"]["citation_map"] == {"[1]": "source-1"}
    assert captured["prompt"]["untrusted_render_input"]["goal"] == "Explain the topic"
    assert captured["prompt"]["untrusted_render_input"]["key_questions"] == ["What changed?"]
    assert captured["prompt"]["untrusted_render_input"]["gap_coverage_summary"] == []
    assert captured["prompt"]["untrusted_render_input"]["items"][0]["candidate_key"] == "source-1"
    assert captured["prompt"]["untrusted_render_input"]["items"][0]["rationale"] == "Start here"
    assert captured["prompt"]["untrusted_render_input"]["items"][0]["citation"] == "[1]"


def test_render_checkpoint_uses_writer_pricing_for_budget(monkeypatch) -> None:
    module = _import_rendering_module(monkeypatch)
    materialization_module = importlib.import_module(
        "deep_research.renderers.materialization"
    )

    class FakeAgent:
        def run_sync(self, prompt):
            return types.SimpleNamespace(
                output=RenderProse(content_markdown="# Reading Path\n\nUse [1]."),
                usage=lambda: types.SimpleNamespace(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                ),
            )

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.writer",
        types.SimpleNamespace(build_writer_agent=lambda model_name: FakeAgent()),
    )
    monkeypatch.setattr(
        materialization_module, "load_prompt", lambda name: f"prompt:{name}"
    )

    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "writer_pricing": {
                "input_per_million_usd": 100.0,
                "output_per_million_usd": 200.0,
            }
        }
    )

    result = module.write_reading_path(
        SelectionGraph(
            items=[SelectionItem(candidate_key="source-1", rationale="Start here")]
        ),
        _sample_ledger(),
        _sample_plan(),
        config,
    )

    assert result.budget.input_tokens == 11
    assert result.budget.output_tokens == 7
    assert result.budget.estimated_cost_usd == 0.0025


def test_render_checkpoint_rejects_invented_citations(monkeypatch) -> None:
    module = _import_rendering_module(monkeypatch)
    materialization_module = importlib.import_module(
        "deep_research.renderers.materialization"
    )

    class FakeAgent:
        def run_sync(self, prompt):
            return types.SimpleNamespace(
                output=RenderProse(content_markdown="# Report\n\nBad [9].")
            )

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.writer",
        types.SimpleNamespace(build_writer_agent=lambda model_name: FakeAgent()),
    )
    monkeypatch.setattr(
        materialization_module, "load_prompt", lambda name: f"prompt:{name}"
    )

    with pytest.raises(ValueError, match="Unknown citation markers"):
        materialization_module.materialize_render_payload(
            RenderPayload(
                name="reading_path",
                content_markdown="",
                citation_map={"[1]": "source-1"},
                structured_content={"items": []},
            ),
            writer_model="gemini/gemini-2.5-flash",
            prompt_name="writer_reading_path",
            pricing=module.ModelPricing(),
        )


def test_materialize_full_report_uses_full_report_prompt(monkeypatch) -> None:
    module = _import_rendering_module(monkeypatch)
    materialization_module = importlib.import_module(
        "deep_research.renderers.materialization"
    )
    prompt_calls = []
    agent_models = []

    class FakeAgent:
        def run_sync(self, prompt):
            return types.SimpleNamespace(
                output=RenderProse(content_markdown="# Full Report\n\nGrounded [1].")
            )

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.writer",
        types.SimpleNamespace(
            build_writer_agent=lambda model_name: agent_models.append(model_name)
            or FakeAgent()
        ),
    )
    monkeypatch.setattr(
        materialization_module,
        "load_prompt",
        lambda name: prompt_calls.append(name) or f"prompt:{name}",
    )

    result = module.write_full_report(
        _sample_package(),
        ResearchConfig.for_tier(Tier.STANDARD).model_copy(
            update={"writer_model": "gemini/gemini-2.5-flash"}
        ),
    )

    assert result.render.name == "full_report"
    assert prompt_calls == ["writer_full_report"]
    assert agent_models == ["gemini/gemini-2.5-flash"]
