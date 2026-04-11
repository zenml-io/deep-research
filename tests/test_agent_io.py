import types

import pytest
from pydantic import BaseModel

from deep_research.agent_io import (
    ToolResultCollector,
    extract_tool_results_with_stats,
    serialize_prompt_payload,
)
from deep_research.models import RawToolResult


class _DumpablePayload(BaseModel):
    provider: str = "mcp"
    payload: dict = {"source_kind": "paper", "results": []}


def test_serialize_prompt_payload_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="supervisor prompt"):
        serialize_prompt_payload(
            {"bad": {1, 2, 3}},
            label="supervisor prompt payload",
        )


def test_extract_tool_results_with_stats_degrades_without_trace_api() -> None:
    raw_results, stats = extract_tool_results_with_stats(object())

    assert raw_results == []
    assert stats.trace_available is False
    assert stats.warnings == ["result_missing_all_messages"]


def test_extract_tool_results_with_stats_accepts_dumpable_payloads() -> None:
    result = types.SimpleNamespace(
        all_messages=lambda: [
            types.SimpleNamespace(
                parts=[
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content=_DumpablePayload(),
                    )
                ]
            )
        ]
    )

    raw_results, stats = extract_tool_results_with_stats(result)

    assert raw_results == [
        RawToolResult(
            tool_name="search",
            provider="mcp",
            payload={"source_kind": "paper", "results": []},
        )
    ]
    assert stats.trace_available is True
    assert stats.normalized_result_count == 1
    assert stats.dropped_part_count == 0


def test_extract_tool_results_with_stats_reports_unknown_shapes() -> None:
    result = types.SimpleNamespace(
        all_messages=lambda: [
            types.SimpleNamespace(
                parts=[
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content={"provider": "mcp", "unexpected": True},
                    )
                ]
            )
        ]
    )

    raw_results, stats = extract_tool_results_with_stats(result)

    assert raw_results == []
    assert stats.tool_return_part_count == 1
    assert stats.dropped_part_count == 1
    assert stats.warnings == ["unhandled_tool_return_keys:provider,unexpected"]


# --- ToolResultCollector tests ---


def test_tool_result_collector_captures_raw_tool_result_directly() -> None:
    collector = ToolResultCollector()
    raw = RawToolResult(
        tool_name="search",
        provider="brave",
        payload={"results": []},
    )

    collector.hook("search", raw)

    assert collector.results == [raw]
    assert collector.call_count == 1


def test_tool_result_collector_coerces_mapping_with_payload() -> None:
    collector = ToolResultCollector()
    mapping_result = {
        "tool_name": "search",
        "provider": "arxiv",
        "payload": {"source_kind": "paper", "items": []},
        "ok": True,
    }

    collector.hook("search", mapping_result)

    assert len(collector.results) == 1
    assert collector.results[0].tool_name == "search"
    assert collector.results[0].provider == "arxiv"
    assert collector.results[0].payload == {"source_kind": "paper", "items": []}
    assert collector.results[0].ok is True
    assert collector.call_count == 1


def test_tool_result_collector_uses_hook_tool_name_as_fallback() -> None:
    collector = ToolResultCollector()
    mapping_result = {
        "provider": "mcp",
        "payload": {"results": []},
    }

    collector.hook("my_tool", mapping_result)

    assert collector.results[0].tool_name == "my_tool"
    assert collector.results[0].provider == "mcp"


def test_tool_result_collector_tracks_call_count_for_non_collectible_results() -> None:
    collector = ToolResultCollector()

    collector.hook("search", "just a string")
    collector.hook("search", 42)
    collector.hook("search", {"no_payload_key": True})

    assert collector.results == []
    assert collector.call_count == 3


def test_tool_result_collector_handles_multiple_calls() -> None:
    collector = ToolResultCollector()
    raw1 = RawToolResult(tool_name="s1", provider="brave", payload={"r": 1})
    raw2 = RawToolResult(tool_name="s2", provider="arxiv", payload={"r": 2})

    collector.hook("s1", raw1)
    collector.hook("s2", raw2)
    collector.hook("s3", "ignored")

    assert len(collector.results) == 2
    assert collector.results[0] is raw1
    assert collector.results[1] is raw2
    assert collector.call_count == 3


def test_tool_result_collector_handles_flat_shape_results() -> None:
    collector = ToolResultCollector()
    flat_result = {
        "results": [{"title": "test"}],
        "source_kind": "web",
        "provider": "tavily",
    }

    collector.hook("search", flat_result)

    assert len(collector.results) == 1
    assert collector.results[0].tool_name == "search"
    assert collector.results[0].provider == "tavily"
    assert collector.results[0].payload == flat_result
