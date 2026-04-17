# PydanticAI Usage Audit — deep-research V2

Scope: `research/agents/`, `research/providers/agent_tools.py`, `research/checkpoints/subagent.py`, `research/checkpoints/judge.py`, `research/config/{defaults,slots}.py`.

## Summary

The codebase uses PydanticAI in a clean, idiomatic way: one `Agent(...)` per role, `output_type` for structured Pydantic contracts (or `str` where a tool-call JSON shape would truncate long-form markdown), a single factory (`_build_agent`) that funnels construction through the Kitaru `KitaruAgent` wrapper, and `model_settings` passed as a provider-namespaced dict only where needed (judge's `google_thinking_config`). The agent/tool split correctly confines tools to the subagent slot — supervisor/generator/reviewer/finalizer/judge are all tool-less decision/synthesis agents, which matches PydanticAI's "narrow agent surface" guidance.

No CORRECTNESS BUGs were found — every `Agent(...)` call passes a valid triple of `model_name`, `output_type`, `system_prompt`, plus optional `tools`/`model_settings`. The runtime contract is honoured end-to-end. The findings below are anti-patterns and missed opportunities.

---

## Findings

### 1. `_record_usage` callable fallback is dead code — ANTI-PATTERN
`research/agents/_factory.py:54-60`

```python
usage = getattr(result, "usage", None)
if callable(usage):
    usage = usage()
```

In current PydanticAI (`AgentRunResult.usage()` is a **method** returning a `RunUsage`, see `pydantic_ai.result.AgentRunResult`). So `callable(usage)` will be True and the branch runs — but the initial `getattr(..., None)` + "attribute or callable" duality is confused. Pick one: call `result.usage()` directly. The `None` short-circuit is redundant because `usage` is always present on successful runs; on failure the agent raises. The defensive code hides the actual API shape and will become a silent no-op if the attribute name changes (e.g. to `run_usage`). Recommend: `usage = result.usage()` and let an AttributeError surface loudly in tests.

### 2. Token-field reads use `int(... or 0)` pattern — ANTI-PATTERN
`research/agents/_factory.py:62-63`

`getattr(usage, "input_tokens", 0) or 0` defends against both missing and `None` values. PydanticAI's `RunUsage` guarantees `input_tokens: int` and `output_tokens: int` (not Optional). The `or 0` suppresses any real zero (which is fine) but the extra `int(...)` cast is pointless against a typed int. Drop to `usage.input_tokens`, `usage.output_tokens`. Also: `RunUsage` exposes `request_tokens`, `response_tokens`, `total_tokens`, and `details` — the current extraction silently drops reasoning/cached-token breakdowns that Anthropic/OpenAI now report. For cost accuracy this is a MISSED OPPORTUNITY (see #7).

### 3. Tools lack `RunContext` / `deps_type` injection — MISSED OPPORTUNITY
`research/providers/agent_tools.py:153-206`, `research/checkpoints/subagent.py:71`

`AgentToolSurface.as_pydantic_tools()` returns free-floating async closures that capture `self._sandbox` / `self._registry` via Python closure. For each `run_subagent` call a **new surface** is built and a **new agent** is constructed with new tool closures (see `checkpoints/subagent.py:71`). This works but:

- It prevents Kitaru's capture layer from distinguishing tool identity across runs (tool IDs are tied to closure hashes).
- It rebuilds provider registries on every retry inside the loop — wasteful.
- PydanticAI's intended pattern is `Agent(..., deps_type=AgentToolSurface)` with tools declared as `async def search(ctx: RunContext[AgentToolSurface], queries: list[str]) -> list[dict]` and `agent.run_sync(prompt, deps=surface)`. That gives you tool-level access to the run-scoped surface without closure capture, plus schema docstrings propagate to the tool signature and Logfire gets structured deps spans.

Recommend migrating the subagent factory to `deps_type=AgentToolSurface` and keeping `BudgetAwareAgent` as a thin wrapper; it is orthogonal to deps injection (budget is a process-level concern, deps are a per-run concern).

### 4. Tool return types use bare `dict` / `list[dict]` — ANTI-PATTERN
`research/providers/agent_tools.py:165,193`

```python
async def search(queries: list[str], max_results_per_query: int = 10) -> list[dict]:
async def code_exec(code: str, language: str = "python") -> dict:
```

PydanticAI derives tool schemas from annotations. `list[dict]` produces `{"type": "array", "items": {"type": "object"}}` with **no properties** — the model sees an opaque object and has to guess field names. Define small `TypedDict` or Pydantic models (`SearchToolResult`, `CodeExecToolResult`) so the LLM gets proper JSON schema and your contracts layer gains a single source of truth. Same critique applies to `fetch` returning `str` — fine as-is, but a discriminated `FetchOk | FetchError` would preserve the `None`-on-failure path you currently flatten to empty string (`agent_tools.py:183`).

### 5. `fetch` swallows errors into empty string — CORRECTNESS-ADJACENT ANTI-PATTERN
`research/providers/agent_tools.py:180-183`

`content = await self.fetch(url); return content or ""` turns network errors, non-text content, PDFs, and **valid empty documents** into the same value. The model cannot tell "nothing there" from "failed". PydanticAI supports raising `ModelRetry` from a tool to nudge the model to try a different input; that would be the idiomatic fix. At minimum, return a structured `{ok: bool, content: str, reason: str | None}`.

### 6. `build_subagent_agent` rebuilds the agent on each retry — MISSED OPPORTUNITY
`research/checkpoints/subagent.py:68-73`

The loop lazily builds `agent` only once (`if agent is None`), good. But the retry loop is inside a `@checkpoint(type="llm_call")` — Kitaru's built-in replay + wrapper-level retry could do this, and PydanticAI has its own `ModelRetry` and transient-error retry settings via `ModelSettings(retry=...)`. Duplicating retry logic here + relying on `getattr(exc, "status_code", ...)` duck-typing (`subagent.py:32`) is brittle. PydanticAI exposes `ModelHTTPError` — import it and isinstance-check; keep the duck-type only as last-resort fallback.

### 7. No extraction of cached / reasoning tokens — MISSED OPPORTUNITY
`research/agents/_factory.py:62-69`

Anthropic's prompt caching and OpenAI's reasoning tokens have distinct per-token prices. `RunUsage.details` carries them. Current `BudgetAwareAgent` silently merges all input into one bucket, over- or under-charging. For an app whose CLAUDE.md explicitly lists `DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT` and prices-per-token in `ModelSlotConfig`, this is a visible gap.

### 8. Judge `model_settings` pattern is correct — OK
`research/agents/judge.py:17-27`, `research/config/defaults.py:43-47`

Judge defaults to `google-gla:gemini-3.1-pro-preview` with `{"google_thinking_config": {"thinking_level": "high"}}` — a provider-namespaced setting correctly passed through `_build_agent(... model_settings=...)` into `Agent(model_settings=...)`. Provider distinctness from generator (`anthropic`) and reviewer (`openai`) is enforced by `_detect_provider_compromise` in `flows/council.py`. This is correct per CLAUDE.md's bias-elimination contract.

### 9. Generator/finalizer `output_type=str` — OK (with caveat)
`research/agents/generator.py:20`, `research/agents/finalizer.py:18`

Using `str` for long-form markdown avoids the PydanticAI tool-call JSON envelope, which models frequently truncate. Correct call. Caveat: with `output_type=str`, `result.output` is the raw string and any structured extraction (section headings, citation counts) is done downstream. That's fine — but note PydanticAI also supports `output_type=TextOutput(str)` for explicit docstring-free passthrough; current `str` works identically.

### 10. `tools` argument typing at the factory — MINOR ANTI-PATTERN
`research/agents/_factory.py:84`, `research/agents/subagent.py:19`

`list[Callable[..., Any]] | None` allows anything. PydanticAI accepts `Callable | Tool` and raises at bind time. A Protocol alias like `AgentTool = Callable[..., Awaitable[Any]]` (or reusing `pydantic_ai.Tool`) makes mis-wired tools fail at the factory boundary, not at first LLM turn.

---

## No Issues Found

- `_build_agent` call shape matches PydanticAI's `Agent(model, output_type=, system_prompt=, tools=, model_settings=)`.
- `BudgetAwareAgent.__getattr__` delegation keeps the `KitaruAgent` surface intact (tests in `test_v2_agents.py` depend on this).
- Late-import of `pydantic_ai` / `kitaru` inside `_build_agent` is necessary for the stub-injection test pattern and correctly implemented.
- Model strings (`anthropic:claude-sonnet-4-6`, `openai:gpt-5.4-mini`, `google-gla:gemini-*`) match PydanticAI's `provider:model` convention.

---

## Priority Recommendations

1. **Typed tool returns** (#4) — biggest quality-of-model-call win, low effort.
2. **`deps_type=AgentToolSurface`** (#3) — aligns with idiomatic PydanticAI and simplifies `agent_tools.py`.
3. **Clean up `_record_usage`** (#1, #2, #7) — correct API use + cached/reasoning-token cost accuracy.
4. **Structured fetch result** (#5) — unblocks supervisor reasoning about "nothing found" vs "fetch failed".
