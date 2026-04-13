from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from deep_research.models import RawToolResult


class ToolTraceExtractionStats(BaseModel):
    """Internal stats for supervisor tool-trace normalization."""

    model_config = ConfigDict(extra="forbid")

    trace_available: bool = False
    message_count: int = 0
    tool_return_part_count: int = 0
    normalized_result_count: int = 0
    dropped_part_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ToolResultCollector:
    """Accumulates RawToolResult objects via PydanticAI after_tool_call hook."""

    def __init__(self) -> None:
        self.results: list[RawToolResult] = []
        self.call_count: int = 0

    def hook(self, tool_name: str, result: object, **kwargs: object) -> None:
        """Hook callback for after_tool_call events."""
        self.call_count += 1
        if isinstance(result, RawToolResult):
            self.results.append(result)
            return
        mapping = _coerce_mapping(result)
        if mapping is not None:
            built = _mapping_to_raw_result(mapping, fallback_name=tool_name)
            if built is not None:
                self.results.append(built)


def serialize_prompt_payload(payload: object, *, label: str = "agent prompt") -> str:
    """Serialize an agent prompt payload with strict JSON semantics."""
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    try:
        return json.dumps(payload, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc


def extract_tool_results_with_stats(
    result: object,
) -> tuple[list[RawToolResult], ToolTraceExtractionStats]:
    """Normalize tool-return parts from a PydanticAI/Kitaru result object.

    Walks the message trace, picks out every tool-return part, and converts
    each to a typed RawToolResult. Unrecognised shapes are counted and
    described in the returned stats so callers can emit structured warnings.
    """
    stats = ToolTraceExtractionStats()
    all_messages = getattr(result, "all_messages", None)
    if not callable(all_messages):
        _append_warning(stats, "result_missing_all_messages")
        return [], stats

    try:
        messages = list(all_messages())
    except Exception as exc:  # pragma: no cover - defensive against adapter drift
        _append_warning(stats, f"all_messages_call_failed:{type(exc).__name__}")
        return [], stats

    stats.trace_available = True
    stats.message_count = len(messages)
    raw_results: list[RawToolResult] = []

    for message in messages:
        for part in _iter_parts(message):
            if _get_field(part, "part_kind") != "tool-return":
                continue
            stats.tool_return_part_count += 1
            normalized = _normalize_tool_return_part(part)
            if normalized is None:
                stats.dropped_part_count += 1
                _append_warning(stats, _describe_unhandled_part(part))
                continue
            raw_results.append(normalized)

    stats.normalized_result_count = len(raw_results)
    return raw_results, stats


def extract_tool_results(result: object) -> list[RawToolResult]:
    """Compatibility wrapper that returns normalized tool results only."""
    raw_results, _ = extract_tool_results_with_stats(result)
    return raw_results


# --- Private helpers ---


def _iter_parts(message: object) -> list[object]:
    """Return message.parts as a list, or empty list if absent/wrong type."""
    parts = _get_field(message, "parts", [])
    if isinstance(parts, list | tuple):
        return list(parts)
    return []


def _normalize_tool_return_part(part: object) -> RawToolResult | None:
    """Convert a single tool-return message part into a RawToolResult."""
    content = _get_field(part, "content")
    if isinstance(content, RawToolResult):
        return content

    content_mapping = _coerce_mapping(content)
    if content_mapping is None:
        return None

    # Fall back to part.tool_name when the content dict omits it.
    fallback_name = str(_get_field(part, "tool_name", "tool"))
    return _mapping_to_raw_result(content_mapping, fallback_name=fallback_name)


def _mapping_to_raw_result(
    mapping: Mapping[str, Any], fallback_name: str
) -> RawToolResult | None:
    """Build a RawToolResult from a flat or nested mapping from an external source.

    Tries a nested payload key first, then flat shapes recognised by results/items/source_kind.
    """
    tool_name = str(mapping.get("tool_name") or fallback_name)
    provider = str(mapping.get("provider") or "mcp")
    ok = bool(mapping.get("ok", True))
    raw_error = mapping.get("error")
    error: str | None = str(raw_error) if raw_error is not None else None

    payload = _coerce_mapping(mapping.get("payload"))
    if payload is not None:
        return RawToolResult(
            tool_name=tool_name,
            provider=provider,
            payload=dict(payload),
            ok=ok,
            error=error,
        )

    if any(key in mapping for key in ("results", "items", "source_kind")):
        return RawToolResult(
            tool_name=tool_name,
            provider=provider,
            payload=dict(mapping),
            ok=ok,
            error=error,
        )

    return None


def _coerce_mapping(value: object) -> Mapping[str, Any] | None:
    """Return value as a Mapping, or convert a BaseModel to one; None otherwise."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        # model_dump(mode="json") always returns dict — cast directly.
        return value.model_dump(mode="json")  # type: ignore[return-value]
    return None


def _describe_unhandled_part(part: object) -> str:
    """Produce a short diagnostic label for an unrecognised tool-return shape."""
    content = _get_field(part, "content")
    mapping = _coerce_mapping(content)
    if mapping is not None:
        keys = ",".join(sorted(str(k) for k in mapping.keys())[:5]) or "<empty>"
        return f"unhandled_tool_return_keys:{keys}"
    if content is None:
        return "unhandled_tool_return_type:none"
    return f"unhandled_tool_return_type:{type(content).__name__}"


def _append_warning(stats: ToolTraceExtractionStats, warning: str) -> None:
    """Add warning to stats, deduplicating and capping at 10 entries."""
    if warning in stats.warnings or len(stats.warnings) >= 10:
        return
    stats.warnings.append(warning)


def _get_field(obj: object, name: str, default: object = None) -> object:
    """Read a named field from a Mapping or arbitrary object (external/untyped source)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)
