# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A research engine that turns questions into structured, evidence-backed investigation packages. Built on **Kitaru** (ZenML's durable execution framework) and **PydanticAI** (structured LLM completions). The engine scopes a brief, plans subtopics, iterates supervisor-driven search cycles with parallel subagents, generates reports with cross-provider critique, and assembles an `InvestigationPackage`.

The V2 codebase lives in the `research/` package. The legacy V1 code (`deep_research/`) is reference material only — V2 does not import or wrap V1 modules.

## Setup

Python >= 3.11 required. Package manager is `uv`.

### 1. Install dependencies

```bash
uv sync                        # core deps
uv sync --extra dev            # adds pytest for running tests
uv sync --extra evals          # adds pydantic-evals for offline eval harness
```

Kitaru is installed from the [PydanticAI integration branch](https://github.com/zenml-io/kitaru/tree/feature/pydanticai-integration) via `pyproject.toml`.

### 2. Configure LLM provider keys

V2 uses three providers by default — Anthropic (generator), OpenAI (reviewer), Gemini (subagent/judge). The cross-provider split is intentional.

**All three providers (recommended):**

```bash
export GEMINI_API_KEY="..."       # subagent, judge, second reviewer
export ANTHROPIC_API_KEY="..."    # generator (scope, plan, draft, finalize)
export OPENAI_API_KEY="..."       # reviewer
```

**Override model slots to use fewer providers:**

```bash
export GEMINI_API_KEY="..."
export RESEARCH_GENERATOR_MODEL="google-gla:gemini-2.5-flash"
export RESEARCH_REVIEWER_MODEL="google-gla:gemini-2.5-flash"
```

### 3. Configure search provider keys (optional)

arXiv and Semantic Scholar work without keys. For broader web search:

```bash
export EXA_API_KEY="..."
export BRAVE_API_KEY="..."
export SEMANTIC_SCHOLAR_API_KEY="..."  # optional, increases rate limits
```

Enable additional providers via:

```bash
export RESEARCH_ENABLED_PROVIDERS="arxiv,semantic_scholar,exa,brave"
```

### 4. Connect Logfire (optional)

Logfire observability is auto-enabled when the `logfire` SDK is installed and a token is present. It instruments PydanticAI agent spans, httpx HTTP calls, and MCP transports.

```bash
uv run logfire auth              # authenticate with Logfire
uv run logfire projects use      # select a project to send traces to
```

To include full LLM prompt/completion content in traces (off by default):

```bash
export DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT=true
```

### 5. Run a research investigation

```bash
uv run python run_v2.py "What are the latest advances in RLHF alternatives?"
uv run python run_v2.py --tier deep "My research question"
uv run python run_v2.py --tier quick --output ./results "Brief overview of X"
```

## Commands

```bash
# Tests (V2)
uv run pytest tests/test_v2_*.py -v       # all V2 tests (554 tests)
uv run pytest tests/test_v2_agents.py -v   # single file
uv run pytest tests/test_v2_agents.py::TestScopeAgent -v  # single class

# All tests (V1 + V2)
uv run pytest tests/ -v

# Evals (separate from pytest — never run as part of default test suite)
uv run python -m evals.runner                          # all suites
uv run python -m evals.runner --suite brief_to_plan    # one suite
uv run python -m evals.runner --write-baseline         # save baseline
uv run python -m evals.runner --use-llm-judge          # enable LLM scoring
```

## Architecture

### Flow -> Checkpoints -> Agents

The system has three layers:

1. **Flow** (`research/flows/deep_research.py`) — The `@flow`-decorated entry point. Orchestrates phases, owns the iteration loop, and enforces convergence. ~268 lines.

2. **Checkpoints** (`research/checkpoints/`) — Each research phase is a `@checkpoint` function (Kitaru). Types are `"llm_call"` or `"tool_call"`. Checkpoints are the replay boundary — on crash recovery, completed checkpoints return cached results.

3. **Agents** (`research/agents/`) — PydanticAI `Agent` factories. Each `build_*_agent(model_name, ...)` function creates an agent, wraps it with `KitaruAgent` from `kitaru.adapters.pydantic_ai`, and returns it. Each factory imports `KitaruAgent` and `CapturePolicy` directly — no indirection layer.

### Key modules

- **`research/contracts/`** — All Pydantic data models. `StrictBase` (`extra="forbid"`) for data contracts, regular `BaseModel` for configs.
- **`research/config/`** — `ResearchSettings` (env vars with `RESEARCH_` prefix), `ResearchConfig` (frozen runtime config), tier defaults. `ResearchConfig.for_tier(tier)` is the canonical factory.
- **`research/ledger/`** — Pure functions, no LLM calls. `ManagedLedger` with append-only dedup. Dedup precedence: DOI > arXiv ID > canonical URL. Windowed projection for context management.
- **`research/providers/`** — `SearchProvider` Protocol + `ProviderRegistry`. Four built-in providers: brave, arxiv, semantic_scholar, exa. Plus `AgentToolSurface` (search/fetch/code_exec as PydanticAI tools).
- **`research/flows/convergence.py`** — Four stop rules in priority order: budget > time > supervisor done > max iterations.
- **`research/prompts/`** — System prompts as `.md` files with SHA-256 hash tracking via `PROMPTS` registry.

### Pipeline data flow

`deep_research()` orchestrates these phases:
1. `run_scope(question)` -> `ResearchBrief`
2. `run_plan(brief)` -> `ResearchPlan`
3. Iteration loop: supervisor -> fan-out subagents -> convergence check
4. `run_draft(brief, plan, ledger)` -> `DraftReport`
5. `run_critique(draft)` -> `CritiqueReport` (with dual-reviewer merge on deep tier)
6. Supplemental loop if reviewer requests more research
7. `run_finalize(draft, critique)` -> `FinalReport`
8. `assemble_package()` -> `InvestigationPackage`

### Non-determinism isolation

UUID generation and wall-clock snapshots are confined to `research/checkpoints/metadata.py`. On Kitaru replay, these checkpoints return their original values — everything else is pure given cached checkpoint results.

## Testing Patterns

- **Kitaru stub injection** — V2 tests inject lightweight stubs for `kitaru` and `pydantic_ai` into `sys.modules` before importing. `FakeKitaruAgent` delegates `.run_sync()` to the wrapped PydanticAI agent.
- **Module re-import** — Test helpers pop module cache entries and re-import under stub context.
- **Monkeypatching checkpoints** — Flow tests monkeypatch individual checkpoint functions to control each phase.
- **`pydantic_ai.TestModel`** — Agent behavior tests use PydanticAI's `TestModel` for offline structured output without real API calls.
- **No shared conftest.py** — Each test file is self-contained.
- **Test timeout** — All tests use `--timeout=60` (configured in `pyproject.toml`).

## LLM Provider Model Strings

PydanticAI uses provider-prefixed format:
- `google-gla:gemini-2.5-flash`
- `anthropic:claude-sonnet-4-20250514`
- `openai:gpt-4o-mini`

## Environment Variables

All `RESEARCH_*` settings are loaded via pydantic-settings (`ResearchSettings`):

- `RESEARCH_DEFAULT_TIER` — Default tier when not specified (default: `standard`)
- `RESEARCH_DEFAULT_COST_BUDGET_USD` — Soft cost ceiling per run (default: `0.10`)
- `RESEARCH_DAILY_COST_LIMIT_USD` — Global daily ceiling (default: `10.00`)
- `RESEARCH_ENABLED_PROVIDERS` — Comma-separated search providers (default: `brave,exa,arxiv,semantic_scholar`)
- `RESEARCH_MAX_PARALLEL_SUBAGENTS` — Concurrent subagents per iteration (default: `3`)
- `RESEARCH_LEDGER_WINDOW_ITERATIONS` — Recent iterations shown in full to agents (default: `3`)
- `RESEARCH_GROUNDING_MIN_RATIO` — Minimum citation density for assembly (default: `0.7`)
- `RESEARCH_MAX_SUPPLEMENTAL_LOOPS` — Extra iterations if reviewer requests more research (default: `1`)
- `RESEARCH_ALLOW_UNFINALIZED_PACKAGE` — Allow output when finalizer fails (default: `false`)
- `RESEARCH_SANDBOX_ENABLED` — Enable sandboxed code execution tool (default: `false`)
- `DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT` — Ship LLM content to Logfire (default: `false`)
