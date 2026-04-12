# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A research engine that turns questions into structured, evidence-backed investigation packages. Built on **Kitaru** (ZenML's durable execution framework) and **PydanticAI** (structured LLM completions). The engine classifies a brief, plans subtopics, iterates search-evaluate-refine cycles, renders reports, and assembles an `InvestigationPackage`.

## Setup

Python ≥ 3.11 required. Package manager is `uv`.

### 1. Install dependencies

```bash
uv sync                    # core deps
uv sync --extra evals      # adds pydantic-evals for offline eval harness
```

### 2. Configure LLM provider keys

Most agents default to Gemini. The `coverage_scorer` (runs every iteration on all tiers) defaults to `openai:gpt-4o-mini`, and the `deep` tier adds Anthropic (reviewer) and more OpenAI (judge).

**Option A: Gemini + Pydantic Gateway (recommended)**

Use a direct Gemini AI Studio key for primary agents, and a [Pydantic Gateway/Codex](https://pydantic.dev/docs/ai/overview/gateway/) key to route OpenAI/Anthropic models (Gateway doesn't support `google-gla`, only `google-vertex`). Model strings use `gateway/<provider>:<model>` format:

```bash
export GEMINI_API_KEY="..."                  # direct — most agents use google-gla:*
export PYDANTIC_AI_GATEWAY_API_KEY="pylf_v..."  # routes openai/anthropic models

# Route non-Gemini models through gateway
export RESEARCH_COVERAGE_SCORER_MODEL="gateway/openai:gpt-4o-mini"
export RESEARCH_REVIEW_MODEL="gateway/anthropic:claude-sonnet-4-20250514"
export RESEARCH_JUDGE_MODEL="gateway/openai:gpt-4o-mini"
```

**Option B: Gemini-only (quick/standard tiers)**

Override the coverage scorer to avoid needing OpenAI:

```bash
export GEMINI_API_KEY="..."
export RESEARCH_COVERAGE_SCORER_MODEL="google-gla:gemini-2.5-flash"
```

**Option C: Direct provider keys**

```bash
export GEMINI_API_KEY="..."        # primary — most agents
export OPENAI_API_KEY=""        # coverage scorer + judge
export ANTHROPIC_API_KEY="..."     # reviewer (deep tier only)
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

Logfire observability is auto-enabled when the `logfire` SDK is installed and a token is present. It instruments PydanticAI (agent spans), httpx (HTTP calls), and MCP transports. The `configure()` call uses `send_to_logfire="if-token-present"`, so it's zero-config locally and only ships traces when authenticated.

```bash
uv run logfire auth              # authenticate with Logfire
uv run logfire projects use      # select a project to send traces to
```

Traces show every checkpoint span (`run_supervisor`, `score_coverage`, etc.) and per-iteration metrics (`iteration_coverage`, `iteration_cost_usd`, `iteration_candidates`).

To include full LLM prompt/completion content in traces (off by default):

```bash
export DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT=true
```

For eval harness traces, use the `--enable-logfire` flag:

```bash
LOGFIRE_TOKEN=... uv run python -m evals.runner --use-llm-judge --enable-logfire
```

Eval traces use service name `deep-research-evals`.

### 5. Run a research investigation

```bash
uv run python run.py "What are the latest advances in RLHF alternatives?"
uv run python run.py --tier deep "My research question"
uv run python run.py --tier deep --output ./results "What is harness for deep research agents and how we can build better deep research agent with proper harness?"
```

## Commands

```bash
# Tests
uv run pytest tests/ -v              # all tests (~297)
uv run pytest tests/test_models.py   # single file
uv run pytest tests/test_models.py::test_name -v  # single test

# Evals (separate from pytest — never run as part of default test suite)
uv run python -m evals.runner                          # all suites
uv run python -m evals.runner --suite brief_to_plan    # one suite
uv run python -m evals.runner --write-baseline         # save baseline
uv run python -m evals.runner --use-llm-judge          # enable LLM scoring
```

## Architecture

### Flow → Checkpoints → Agents

The system has three layers:

1. **Flow** (`deep_research/flow/research_flow.py`) — The `@flow`-decorated entry point. Intentionally thin (~117 lines). Delegates all logic to `_pipeline.py`.

2. **Checkpoints** (`deep_research/checkpoints/`) — Each research phase is a `@checkpoint` function (Kitaru). Checkpoints are called via `.submit().load()` for parallel execution and replay. Types are `"llm_call"` or `"tool_call"`.

3. **Agents** (`deep_research/agents/`) — PydanticAI `Agent` factories. Each is a `build_*_agent(model_name, ...)` function that creates an agent, wraps it with `kp.wrap()` from `kitaru.adapters.pydantic_ai`, and returns it. Agents get typed `output_type` Pydantic models and system prompts from `prompts/*.md`.

### Key modules

- **`models.py`** — All Pydantic data contracts. Strict models (`extra="forbid"`), frozen configs.
- **`config.py`** — `ResearchSettings` (env vars with `RESEARCH_` prefix) and `ResearchConfig` (frozen runtime config). `ResearchConfig.for_tier(tier)` is the canonical factory.
- **`evidence/`** — Pure functions, no LLM calls. `ledger.py` has the ratchet-merge pattern (scores only go up). Dedup precedence: DOI → arXiv ID → canonical URL → title.
- **`providers/search/`** — `SearchProvider` Protocol + `ProviderRegistry`. Four built-in providers: brave, arxiv, semantic_scholar, exa.
- **`flow/convergence.py`** — Seven stop rules checked in priority order: budget → time → converged → loop stall → diminishing returns → max iterations.
- **`prompts/`** — System prompts as `.md` files, bundled as package data.

### Pipeline data flow

`_pipeline.py` orchestrates these phases:
1. `resolve_config_and_classify()` → `RunState`
2. `run_iteration_loop()` → supervisor searches → normalize → score relevance → merge ledger → fetch → score coverage → replan check
3. `render_deliverable()` → reading path / backing report / full report (via writer agent)
4. `run_critique_if_enabled()` → review → revise
5. `run_judges_if_enabled()` → grounding + coherence verification
6. `assemble_final_package()` → `InvestigationPackage`

### Non-determinism isolation

UUID generation and wall-clock snapshots are confined to `checkpoints/metadata.py`. On Kitaru replay, these checkpoints return their original values — everything else is pure given cached checkpoint results.

## Testing Patterns

- **Kitaru stub injection** — Tests in `test_research_flow_unit.py` and `test_checkpoints.py` inject lightweight stubs for `kitaru` and `pydantic_ai` into `sys.modules` before importing. The stub `@checkpoint` adds `.submit()` returning `SimpleNamespace(load=lambda: result)`.
- **Module re-import** — `_load_research_flow_module()` pops module cache entries and re-imports under stub context.
- **Monkeypatching checkpoints** — Flow tests monkeypatch individual checkpoint functions on the `research_flow` module to control each phase.
- **`pydantic_ai.TestModel`** — Agent behavior tests use PydanticAI's `TestModel` for offline structured output without real API calls.
- **No shared conftest.py** — Each test file is self-contained.

## LLM Provider Model Strings

PydanticAI uses provider-prefixed format:
- `google-gla:gemini-2.5-flash`
- `anthropic:claude-sonnet-4-20250514`
- `openai:gpt-4o-mini`

## Environment Variables

All `RESEARCH_*` settings are loaded via pydantic-settings (`ResearchSettings`). Key runtime knobs beyond the API keys documented in Setup:

- `RESEARCH_DEFAULT_TIER` — Default tier when not specified (default: `standard`)
- `RESEARCH_DEFAULT_COST_BUDGET_USD` — Cost ceiling per run (default: `0.10`)
- `RESEARCH_DAILY_COST_LIMIT_USD` — Global daily ceiling (default: `10.00`)
- `RESEARCH_ALLOW_SUPERVISOR_BASH` — Opt-in bash tool for supervisor agent
- `RESEARCH_ENABLED_PROVIDERS` — Comma-separated search providers (default: `arxiv,semantic_scholar`)
- `DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT` — Ship LLM content to Logfire (default: `false`)
