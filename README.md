# Deep Research Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Built with Kitaru](https://img.shields.io/badge/built%20with-Kitaru-22c55e.svg)](https://kitaru.ai)
[![Built with PydanticAI](https://img.shields.io/badge/built%20with-PydanticAI-8b5cf6.svg)](https://ai.pydantic.dev)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

Turn any research question into a structured, evidence-backed investigation package — with full provenance, cross-provider critique, and convergence-driven iteration.

Built on [Kitaru](https://kitaru.ai) (ZenML's durable execution framework) and [PydanticAI](https://ai.pydantic.dev) (structured LLM completions). Every phase boundary is a checkpoint. Every LLM call produces typed output. Every source is traceable.

## Quick Start

```bash
# Clone and install
git clone https://github.com/zenml-io/deep-research.git
cd deep-research
uv sync

# Set LLM provider keys (see Setup below for options)
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run a research investigation
uv run python run_v2.py "What are the latest advances in RLHF alternatives?"
```

Output:

```
run-a1b2c3d4/
├── package.json        # Full serialized package (machine-readable)
├── report.md           # Final report (or draft fallback)
├── evidence/
│   └── ledger.json     # Every candidate: considered, merged, deduped
└── iterations/
    ├── 000.json        # Per-iteration: supervisor decision, subagent findings
    └── 001.json
```

## Setup

### Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Install dependencies

```bash
uv sync                        # core deps
uv sync --extra dev            # adds pytest for running tests
uv sync --extra evals          # adds pydantic-evals for offline eval harness
```

Kitaru is installed automatically from the [PydanticAI integration branch](https://github.com/zenml-io/kitaru/tree/feature/pydanticai-integration).

### 2. Configure LLM provider keys

The engine uses three LLM providers across its agent pipeline. The cross-provider topology is intentional — generation, review, and judging happen on different providers to avoid single-provider blind spots.

**Default model assignments by tier:**

| Role | Quick | Standard | Deep |
|---|---|---|---|
| Generator | `anthropic:claude-sonnet-4-6` | `anthropic:claude-sonnet-4-6` | `anthropic:claude-sonnet-4-6` |
| Scope (override) | — | — | `anthropic:claude-opus-4-6` |
| Subagent | `google-gla:gemini-3.1-flash-lite-preview` | `google-gla:gemini-3.1-flash-lite-preview` | `google-gla:gemini-3.1-flash-lite-preview` |
| Reviewer | `openai:gpt-5.4-mini` | `openai:gpt-5.4-mini` | `openai:gpt-5.4-mini` |
| Judge | — | `google-gla:gemini-3.1-pro-preview` | `google-gla:gemini-3.1-pro-preview` |
| 2nd Reviewer | — | — | `google-gla:gemini-3.1-pro-preview` |

**Option A — All three providers (recommended for deep tier)**

```bash
export GEMINI_API_KEY="..."       # subagent, judge, second reviewer
export ANTHROPIC_API_KEY="..."    # generator (scope, planner, draft, finalize)
export OPENAI_API_KEY="..."       # reviewer
```

**Option B — Gemini + Pydantic Gateway**

Use a [Pydantic Gateway](https://pydantic.dev/docs/ai/overview/gateway/) key to route Anthropic/OpenAI models through a single endpoint:

```bash
export GEMINI_API_KEY="..."
export PYDANTIC_AI_GATEWAY_API_KEY="pylf_v..."
```

**Option C — Override models to use fewer providers**

Override any model slot via environment variables to reduce provider requirements:

```bash
export GEMINI_API_KEY="..."
# Route everything through Gemini
export RESEARCH_GENERATOR_MODEL="google-gla:gemini-2.5-flash"
export RESEARCH_REVIEWER_MODEL="google-gla:gemini-2.5-flash"
```

> Note: cross-provider critique is a load-bearing quality mechanism. Overriding all models to the same provider removes this safeguard.

### 3. Configure search providers (optional)

arXiv and Semantic Scholar work without API keys. For broader web search, add:

```bash
export BRAVE_API_KEY="..."
export EXA_API_KEY="..."
export TAVILY_API_KEY="tvly-..."
export SEMANTIC_SCHOLAR_API_KEY="..."   # optional, increases rate limits
```

Control which providers are active:

```bash
export RESEARCH_ENABLED_PROVIDERS="brave,exa,tavily,arxiv,semantic_scholar"
```

Default: `brave,exa,tavily,arxiv,semantic_scholar` (Brave/Exa/Tavily no-op gracefully when API keys are absent).

### 4. Connect Logfire (optional)

Logfire observability is auto-enabled when the `logfire` SDK is installed and a token is present. It instruments PydanticAI agent spans, httpx HTTP calls, and MCP transports. Zero-config locally — only ships traces when authenticated.

```bash
uv run logfire auth              # authenticate
uv run logfire projects use      # select project
```

Traces show every checkpoint span (`run_supervisor`, `run_scope`, etc.) and per-iteration metrics.

To include full LLM prompt/completion content in traces (off by default):

```bash
export DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT=true
```

## Usage

### CLI

```bash
# Standard tier (default)
uv run python run_v2.py "What are the latest advances in RLHF alternatives?"

# Quick tier — faster, cheaper, no critique/judge
uv run python run_v2.py --tier quick "Brief overview of transformer architectures"

# Deep tier — more iterations, dual reviewer, judge verification
uv run python run_v2.py --tier deep "Comprehensive analysis of protein folding methods"

# Custom output directory
uv run python run_v2.py --output ./results "My research question"

# Allow package output even if finalizer fails
uv run python run_v2.py --allow-unfinalized "My research question"
```

### As a library

```python
from research.config import ResearchConfig
from research.flows.deep_research import deep_research

config = ResearchConfig.for_tier("standard")
package = deep_research(
    question="What are the latest advances in RLHF alternatives?",
    config=config,
)

print(package.metadata.stop_reason)
print(package.final_report or package.draft.report_markdown)
```

## Architecture

### Three layers: Flow -> Checkpoints -> Agents

```
Kitaru @flow / @checkpoint              PydanticAI Agent + KitaruAgent
┌──────────────────────────────┐       ┌─────────────────────────────┐
│ Owns:                        │       │ Owns:                       │
│  - Durable state & replay    │ calls │  - Model call routing       │
│  - Checkpoints & artifacts   │──────>│  - Structured output parse  │
│  - Convergence enforcement   │       │  - Tool execution           │
│  - Cost tracking & budgets   │<──────│  - Response validation      │
│  - Stop rules & time boxes   │returns│                             │
│                              │ typed │ Does NOT own:               │
│                              │result │  - Iteration state          │
│                              │       │  - When to stop             │
│                              │       │  - Checkpointing            │
└──────────────────────────────┘       └─────────────────────────────┘
```

1. **Flow** (`research/flows/deep_research.py`) — The `@flow`-decorated entry point. Orchestrates phases, owns the iteration loop, and enforces convergence. ~268 lines.

2. **Checkpoints** (`research/checkpoints/`) — Each research phase is a `@checkpoint` function. Types are `"llm_call"` or `"tool_call"`. Checkpoints are the replay boundary — on crash recovery, completed checkpoints return cached results.

3. **Agents** (`research/agents/`) — PydanticAI `Agent` factories. Each `build_*_agent()` function creates an agent, wraps it with `KitaruAgent` from `kitaru.adapters.pydantic_ai`, and returns it. Agents produce typed Pydantic output models.

### Research pipeline

```
Question
  │
  ├─ 1. Scope ──────────── classify question → ResearchBrief
  ├─ 2. Plan ───────────── break into subtopics, queries → ResearchPlan
  │
  ├─ 3. Iteration Loop ─── [repeat until convergence]
  │     ├─ Supervisor ───── decide what to search → SupervisorDecision
  │     ├─ Subagents ────── fan-out search + fetch → SubagentFindings[]
  │     └─ Convergence ──── check stop rules → continue or stop
  │
  ├─ 4. Draft ──────────── generate report from evidence → DraftReport
  ├─ 5. Critique ───────── cross-provider review → CritiqueReport
  │     └─ Supplemental ── if reviewer says "need more research" → re-iterate
  ├─ 6. Finalize ───────── incorporate critique → FinalReport
  └─ 7. Assemble ───────── validate & package → InvestigationPackage
```

### Stop rules

Four conditions checked in strict priority order:

| Priority | Rule | Condition |
|---|---|---|
| 1 | Budget exhausted | `spent_usd >= soft_budget_usd` |
| 2 | Time exhausted | `elapsed >= time_limit` |
| 3 | Supervisor done | Supervisor signals `done=True` |
| 4 | Max iterations | Hard cap reached |

Budget exhaustion does NOT prevent deliverable production — the engine still drafts, critiques, and assembles with whatever evidence it has.

### Cross-provider critique

The engine enforces model diversity for quality checks. Generator (Anthropic) and Reviewer (OpenAI) always use different providers. The deep tier adds a second reviewer (Gemini) and a judge. This follows the finding that separating generation from evaluation across providers produces measurably better results.

### Council mode (experimental)

A separate `council_research()` flow runs the full pipeline once per generator model, then uses a judge to compare outputs. Available for the deep tier. Not yet wired into the CLI.

## Tiers

| Tier | Max Iterations | Cost Budget | Critique | Judge | 2nd Reviewer |
|---|---|---|---|---|---|
| **quick** | 2 | $0.05 | Single | No | No |
| **standard** | 5 | $0.10 | Single | Yes | No |
| **deep** | 10 | $1.00 | Dual | Yes | Yes |

All tiers fan out up to 3 parallel subagents per iteration.

## Configuration

### Environment variables

All settings use the `RESEARCH_` prefix, loaded via `pydantic-settings`.

| Variable | Default | Description |
|---|---|---|
| `RESEARCH_DEFAULT_TIER` | `standard` | Default tier when not specified |
| `RESEARCH_DEFAULT_COST_BUDGET_USD` | `0.10` | Soft cost ceiling per run |
| `RESEARCH_DAILY_COST_LIMIT_USD` | `10.00` | Global daily ceiling |
| `RESEARCH_ENABLED_PROVIDERS` | `brave,exa,tavily,arxiv,semantic_scholar` | Comma-separated search providers |
| `RESEARCH_MAX_PARALLEL_SUBAGENTS` | `3` | Concurrent subagents per iteration |
| `RESEARCH_LEDGER_WINDOW_ITERATIONS` | `3` | How many recent iterations shown in full to agents |
| `RESEARCH_GROUNDING_MIN_RATIO` | `0.7` | Minimum citation density for assembly |
| `RESEARCH_MAX_SUPPLEMENTAL_LOOPS` | `1` | Extra iterations if reviewer requests more research |
| `RESEARCH_ALLOW_UNFINALIZED_PACKAGE` | `false` | Allow output when finalizer fails |
| `RESEARCH_SANDBOX_ENABLED` | `false` | Enable sandboxed code execution tool |
| `DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT` | `false` | Ship LLM content to Logfire |

### Provider API keys

| Provider | Variable | Used by |
|---|---|---|
| Google (Gemini) | `GEMINI_API_KEY` | Subagent, Judge, 2nd Reviewer |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Generator (scope, plan, draft, finalize) |
| OpenAI (GPT) | `OPENAI_API_KEY` | Reviewer |
| Brave Search | `BRAVE_API_KEY` | Web search provider |
| Exa | `EXA_API_KEY` | Web search provider |
| Tavily | `TAVILY_API_KEY` | Web search provider |
| Semantic Scholar | `SEMANTIC_SCHOLAR_API_KEY` | Academic search provider (optional, increases rate limits) |

### Model string format

PydanticAI uses `provider:model` format:
- `google-gla:gemini-3.1-flash-lite-preview`
- `anthropic:claude-sonnet-4-6`
- `openai:gpt-5.4-mini`

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run only V2 tests
uv run pytest tests/test_v2_*.py -v

# Run a single test file
uv run pytest tests/test_v2_agents.py -v

# Run a single test
uv run pytest tests/test_v2_agents.py::TestScopeAgent::test_creates_agent_with_correct_model -v
```

554 V2 tests covering agents, checkpoints, config, convergence, contracts, council, flow orchestration, integration, architectural invariants, evidence ledger, package export, prompts, and search providers. All tests run offline using stubs — no API keys or network access required.

### Testing patterns

- **Kitaru stub injection** — Tests inject lightweight stubs for `kitaru` and `pydantic_ai` into `sys.modules` before importing. This avoids needing a real Kitaru runtime.
- **`FakeKitaruAgent`** — Stands in for `KitaruAgent`, delegating `.run_sync()` to the wrapped PydanticAI agent.
- **`pydantic_ai.TestModel`** — Agent behavior tests use PydanticAI's `TestModel` for offline structured output.
- **No shared conftest.py** — Each test file is self-contained.

## Evidence Ledger

The evidence system is pure functions with no LLM calls:

- **Dedup precedence**: DOI > arXiv ID > canonical URL. Items without stable identifiers are always admitted.
- **Append-only**: Scores only ratchet up across iterations. Nothing is deleted.
- **URL canonicalization**: Strips tracking params (`utm_*`, `fbclid`, `gclid`), normalizes scheme/host case, removes fragments.
- **Windowed projection**: Recent iterations shown in full to agents; older items compacted to save context window. Pinned items (high-value evidence) always shown in full.

## Project Layout

```
research/                       # V2 package
├── agents/                     # PydanticAI agent factories (KitaruAgent-wrapped)
│   ├── scope.py                #   Question → ResearchBrief
│   ├── planner.py              #   Brief → ResearchPlan
│   ├── supervisor.py           #   Iteration steering → SupervisorDecision
│   ├── subagent.py             #   Search + fetch execution → SubagentFindings
│   ├── generator.py            #   Evidence → DraftReport
│   ├── reviewer.py             #   Cross-provider critique → CritiqueReport
│   ├── finalizer.py            #   Draft + critique → FinalReport
│   └── judge.py                #   Council comparison → CouncilComparison
├── checkpoints/                # Kitaru @checkpoint functions
│   ├── metadata.py             #   Run stamping, wall clock, finalization
│   ├── scope.py                #   Scope checkpoint (llm_call)
│   ├── plan.py                 #   Plan checkpoint (llm_call)
│   ├── supervisor.py           #   Supervisor checkpoint (llm_call)
│   ├── subagent.py             #   Subagent checkpoint (llm_call)
│   ├── draft.py                #   Draft generation (llm_call)
│   ├── critique.py             #   Review with dual-reviewer merge (llm_call)
│   ├── finalize.py             #   Final report (llm_call)
│   ├── judge.py                #   Council judge (llm_call)
│   └── assemble.py             #   Package assembly + validation (tool_call)
├── config/                     # Configuration
│   ├── settings.py             #   ResearchSettings (env vars)
│   ├── defaults.py             #   Tier defaults (quick/standard/deep)
│   ├── slots.py                #   ModelSlot enum + ModelSlotConfig
│   └── budget.py               #   Mutable budget tracking
├── contracts/                  # Pydantic data models
│   ├── base.py                 #   StrictBase (extra="forbid")
│   ├── brief.py                #   ResearchBrief
│   ├── plan.py                 #   ResearchPlan, Subtopic, SuccessCriteria
│   ├── evidence.py             #   EvidenceItem, EvidenceLedger
│   ├── decisions.py            #   SupervisorDecision, SubagentFindings
│   ├── reports.py              #   DraftReport, CritiqueReport, FinalReport
│   ├── iteration.py            #   IterationRecord
│   └── package.py              #   InvestigationPackage, CouncilPackage
├── flows/                      # Kitaru @flow orchestration
│   ├── deep_research.py        #   Main research flow (~268 lines)
│   ├── council.py              #   Multi-generator council flow
│   ├── convergence.py          #   Stop rule evaluation
│   └── budget.py               #   Token cost estimation
├── ledger/                     # Evidence processing (pure, no LLM)
│   ├── canonical.py            #   DOI/arXiv/URL ID extraction
│   ├── url.py                  #   URL canonicalization
│   ├── dedup.py                #   Dedup key computation
│   ├── ledger.py               #   ManagedLedger (append-only with dedup)
│   └── projection.py           #   Windowed projection for context management
├── providers/                  # Search and tool providers
│   ├── search.py               #   SearchProvider protocol + ProviderRegistry
│   ├── brave.py                #   Brave web search
│   ├── arxiv_provider.py       #   arXiv API
│   ├── semantic_scholar.py     #   Semantic Scholar API
│   ├── exa_provider.py         #   Exa search
│   ├── fetch.py                #   URL fetch + HTML text extraction
│   ├── code_exec.py            #   Sandboxed code execution
│   ├── agent_tools.py          #   AgentToolSurface (search/fetch/exec as tools)
│   └── _http.py                #   Retry-with-backoff HTTP client
├── package/                    # Output materialization
│   ├── assembly.py             #   Package construction + validation
│   └── export.py               #   Filesystem export (JSON + markdown)
├── prompts/                    # System prompts as markdown
│   ├── scope.md
│   ├── planner.md
│   ├── supervisor.md
│   ├── subagent.md
│   ├── generator.md
│   ├── reviewer.md
│   ├── finalizer.md
│   ├── council_judge.md
│   ├── loader.py               #   Load prompt with SHA-256 hash tracking
│   └── registry.py             #   PROMPTS registry for reproducibility
└── mcp/                        #   (reserved for MCP server integration)

run_v2.py                       # CLI entry point
tests/
├── test_v2_agents.py           # Agent factory tests
├── test_v2_checkpoints.py      # Checkpoint behavior tests
├── test_v2_config.py           # Config + tier defaults tests
├── test_v2_convergence.py      # Stop rule tests
├── test_v2_council.py          # Council flow tests
├── test_v2_flow.py             # Flow orchestration tests
├── test_v2_integration.py      # End-to-end integration tests
├── test_v2_invariants.py       # Architectural guard tests
├── test_v2_ledger.py           # Evidence ledger + projection tests
├── test_v2_package.py          # Package export tests
├── test_v2_prompts.py          # Prompt loading tests
├── test_v2_providers.py        # Search provider tests
├── test_v2_imports.py          # Import smoke tests
└── test_v2_run.py              # CLI entry point tests
```

## Design Principles

**Library-first.** The engine is a Python library. The `@flow` is the entry point. Wrap it in any API or CLI.

**Durable everything.** Every phase boundary is a Kitaru checkpoint. Runs survive process crashes and support replay from any completed checkpoint.

**Cross-provider critique.** Generator and reviewer always use different LLM providers. Self-evaluation is unreliable — the engine enforces model diversity for quality checks.

**Immutable evidence.** Evidence only accumulates. Scores ratchet up, snippets merge, nothing is deleted. All contracts use `extra="forbid"`.

**Prompts as markdown.** System prompts live in `research/prompts/*.md`, loaded at agent construction time with SHA-256 tracking for reproducibility. Prompt changes don't require code changes.

**Typed everything.** Every agent produces a Pydantic model. Every checkpoint has a typed return. The `InvestigationPackage` is the canonical, serializable output.

## Contributing

Contributions welcome. This is an active project by the [ZenML](https://zenml.io) team.

- **Issues**: Bug reports and feature requests via [GitHub Issues](https://github.com/zenml-io/deep-research/issues)
- **PRs**: Fork, branch, test (`uv run pytest tests/test_v2_*.py -v`), and open a PR
- **Prompts**: System prompts in `research/prompts/*.md` are high-impact, low-risk contributions

## License

See [LICENSE](LICENSE) for details.
