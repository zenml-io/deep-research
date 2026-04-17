# Cleanup Simplify — V2 Code Quality (3-task scope)

Branch: `claude/condescending-golick`
Worktree: `.claude/worktrees/condescending-golick`
Reviewer: team-lead (after simplify-agent went non-responsive)
Scope: 3 of 8 cleanup tasks
- Task 1: Deduplicate / consolidate (DRY where it reduces complexity)
- Task 2: Consolidate type definitions
- Task 5: Remove weak types (`Any`, untyped params)

Baseline before any changes:
- `uv run python -m pytest tests/test_v2_*.py --timeout=60` → **645 passed, 14 failed** (pre-existing `assemble_package` signature mismatch in tests; out of scope).
- 32 occurrences of `Any` across 9 files in `research/`.

---

## Task 1 — Deduplication / DRY findings

### Survey

Searched for repeated patterns:
- 4 search providers (`brave.py`, `exa_provider.py`, `tavily.py`, `semantic_scholar.py`) share a similar shape: `__init__` reads `XXX_API_KEY` from env, `is_available()` returns `bool(self._api_key)`, async `search()` loops queries through `build_async_client()` + `request_with_retry()` + `try/except` per-query.
- 6 agent factories (`planner.py`, `supervisor.py`, `subagent.py`, `judge.py`, `finalizer.py`, `scope.py`, `generator.py`, `reviewer.py`) — already DRY: every one is ~15 lines and delegates to `research.agents._factory._build_agent()`.
- Contracts (`research/contracts/*.py`) — 9 files totalling 590 lines, each focused. All inherit from `StrictBase`. No duplicated shapes.

### Verdict on providers

The 4 web-search providers and 1 academic provider (semantic_scholar) **look** similar but differ in the bodies that matter:
- different endpoints, request shapes (GET vs POST, headers, params vs json body)
- different response shapes (`web.results` for brave, `results` for exa/tavily, `data` for s2)
- different recency-mapping idioms (Brave's "pd/pw/pm/py" string codes vs Tavily's "day/week/month/year" vs Exa's ISO `startPublishedDate` vs S2's per-result post-filter)
- different `raw_metadata` keys per provider

A shared base class would require enough config knobs (request method, body builder, response-list path, recency-encoder) that it would NOT reduce reading complexity — it would scatter logic across base + subclasses. Per global rules: *"three similar lines is better than premature abstraction"*. Honest copy-paste is the right call here.

The agent factories are the OPPOSITE: they already share the right amount via `_build_agent()`. Each provider stays ~15 lines because the only varying inputs are name/prompt/output_type. This is correctly DRY.

**Verdict: no consolidation changes.**

---

## Task 2 — Type consolidation findings

### Survey

- `Literal[...]` enumerations: zero matches in `research/`. No string-literal type duplication.
- All `BaseModel` / `StrictBase` classes are unique — no shape duplication.
- All Pydantic models inherit from `research.contracts.base.StrictBase` (which sets `extra="forbid"`). This is the only base, used uniformly.
- Type aliases scattered across modules: none found.

### Verdict

**Verdict: no consolidation changes.** Type system is already well-factored.

---

## Task 5 — Weak-type findings

### `Any` audit (32 occurrences in 9 files)

Investigated each occurrence. Most are at framework boundaries where `Any` is the correct upstream type:

| Site | Pattern | Verdict |
|------|---------|---------|
| `flows/deep_research.py:68` | `metadata: dict[str, Any] \| None` for `kitaru.wait()` metadata | KEEP — kitaru `wait()` itself takes `dict[str, Any]` |
| `flows/deep_research.py:111,115,147` | `list[Callable[..., Any]] \| None` for PydanticAI tools | KEEP — tools have variable signatures, this is the canonical PydanticAI type |
| `flows/deep_research.py:148,186,240,246,358` | `list[Any]` for kitaru handle lists | KEEP — kitaru's `_CheckpointDefinition.submit()` returns `Any` (verified at `kitaru/checkpoint.py:248`) |
| `agents/_factory.py:38,74` | `*args: Any, **kwargs: Any` and `__getattr__ -> Any` on `BudgetAwareAgent` | KEEP — transparent delegation wrapper requires `Any` |
| `agents/_factory.py:84,98` | `tools: list[Callable[..., Any]]`, `kwargs: dict[str, Any]` for PydanticAI Agent ctor | KEEP — framework boundary |
| `agents/_factory.py:85` | `model_settings: dict[str, Any] \| None` | KEEP — provider-specific opaque dict (anthropic thinking_budget vs openai reasoning_effort) |
| `agents/judge.py:11,19`, `agents/subagent.py:11,19` | Tools/model_settings parameter types | KEEP — same as above |
| `checkpoints/judge.py:5,19`, `checkpoints/supervisor.py:4,49`, `checkpoints/subagent.py:7,42` | Tools, model_settings, prompt_data | KEEP — boundary types |
| `providers/agent_tools.py:18,153,160` | Tool callables and result lists | KEEP — variable callable signatures |
| `contracts/package.py:23` | Comment text (not a type annotation) | KEEP — docstring describes what is replaced |

**Conclusion:** All 32 `Any` occurrences are correctly used at framework boundaries (kitaru handle types, PydanticAI tool callables, provider-specific opaque dicts, transparent delegation). Strengthening any of them would require introducing internal/private upstream types — wrong direction.

### Untyped-parameter audit

Found 4 parameters lacking annotations that would not be controversial to add:

1. **`research/flows/deep_research.py:63`** — `_wait_for_input(*, schema, ...)`: parameter `schema` has no annotation. Kitaru's `wait()` documents schema as "an explicit type (e.g. bool, str, a Pydantic model)". Should be typed `type`.
2. **`research/flows/deep_research.py:69`** — `_wait_for_input(...) -> None`: **REAL BUG**. The function returns `value` at line 84, and the only caller `_await_plan_approval` reads `if approved is not True`. Annotated `-> None` masks this. The single call site passes `schema=bool`, so the actual returned value is a `bool`. Fix: `-> bool`.
3. **`research/flows/deep_research.py:87`** — `_await_plan_approval(question: str, plan, cfg: ResearchConfig)`: `plan` is untyped. Used as `plan.model_dump(mode="json")`. Should be `ResearchPlan` (already imported at `:47`).
4. **`research/flows/deep_research.py:163-174`** — `_run_supervisor_with_retry(brief, plan, ...)`: `brief` and `plan` untyped. Used by `run_supervisor.submit(brief, plan, ...)` whose checkpoint signature is `ResearchBrief, ResearchPlan, ...`. Should be `ResearchBrief` and `ResearchPlan` (both already imported at `:42, :47`).

These four are HIGH-CONFIDENCE fixes — narrow, mechanical, no behavioural change, and one of them is a latent bug.

---

## High-confidence changes (will implement)

1. `research/flows/deep_research.py:69` — change `_wait_for_input` return type from `-> None` to `-> bool` (matches single caller's usage).
2. `research/flows/deep_research.py:63` — annotate `schema: type` parameter.
3. `research/flows/deep_research.py:87` — annotate `plan: ResearchPlan`.
4. `research/flows/deep_research.py:163-174` — annotate `brief: ResearchBrief, plan: ResearchPlan`.

## Speculative (will NOT implement)

1. **Provider base class** — a `WebSearchProvider` ABC for the 4 HTTP search providers would scatter logic and add config knobs without reducing reading complexity. Honest copy-paste is the right call here.
2. **Strengthening `dict[str, Any]` to `dict[str, object]`** in `checkpoints/supervisor.py:49` — cosmetic; values flow into `json.dumps` which accepts both. Skip.
3. **Helper for `os.environ.get("XXX_API_KEY", "")`** repeated across 4 providers — 1 line each, abstraction overhead exceeds benefit.
4. **Generic `_wait_for_input`** with `TypeVar` — single caller, current concrete `bool` return is enough. Adding generics for one site is over-engineering.

---

## Changes Applied

The four high-confidence type fixes were applied to `research/flows/deep_research.py`:

| Line(s) | Before | After | Why |
|---------|--------|-------|-----|
| 63 | `schema,` (no type) | `schema: type,` | Kitaru's `wait()` docs require schema be a type |
| 69 | `) -> None:` | `) -> bool:` | `_wait_for_input` returns `value`; single caller `_await_plan_approval` passes `schema=bool` and uses the return as a bool |
| 87 | `plan,` (no type) | `plan: ResearchPlan,` | Used as `plan.model_dump(mode="json")` |
| 165-166 | `brief,\n    plan,` (no types) | `brief: ResearchBrief,\n    plan: ResearchPlan,` | Forwarded to `run_supervisor.submit(brief, plan, …)` whose checkpoint signature uses these types |

### Test results after changes

- `uv run python -m pytest tests/test_v2_*.py --timeout=60 -q --tb=no` → **645 passed, 14 failed** (identical to baseline; the same 14 pre-existing `TestAssembleCheckpoint` / `TestGroundingDensity` failures unrelated to these changes).
- `uv run python -c "from research.flows.deep_research import deep_research"` — OK.
- `uv run python run_v2.py --help` — OK.
- Net diff: 4 lines changed in 1 file. No behavioural change. One latent typing bug (`_wait_for_input` return) closed.
