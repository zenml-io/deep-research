# Deep Research Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Built with Kitaru](https://img.shields.io/badge/built%20with-Kitaru-22c55e.svg)](https://kitaru.ai)
[![Built with PydanticAI](https://img.shields.io/badge/built%20with-PydanticAI-8b5cf6.svg)](https://ai.pydantic.dev)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

Turn any research question into a structured, evidence-backed investigation package — with full provenance, cross-provider critique, and convergence-driven iteration.

**This is the first production application built on [Kitaru](https://kitaru.ai)**, ZenML's durable execution framework for AI workflows. It demonstrates how Kitaru's checkpoints, replay, and wait primitives combine with [PydanticAI](https://ai.pydantic.dev)'s structured completions to build a research system that survives failures, tracks every decision, and knows when to stop.

## Quick Start

```bash
# Install
git clone https://github.com/zenml-io/deep-research.git
cd deep-research
uv sync

# Set at least one LLM provider key
export GEMINI_API_KEY="..."

# Run a research investigation
python -c "
from deep_research.flow.research_flow import research_flow

package = research_flow.run('What are the latest advances in RLHF alternatives?')
print(package.run_summary)
print(package.renders[0].content_markdown[:500])
"
```

Three lines of code. The engine classifies your brief, plans the investigation, iterates through search-evaluate-refine cycles, and hands back a structured `InvestigationPackage` with full evidence provenance.

## What You Get Back

The output isn't a wall of text. It's a structured `InvestigationPackage` with six inspectable layers:

```
run-a1b2c3d4/
├── package.json              # Full serialized package (machine-readable)
├── summary.md                # Run metadata: tier, cost, timing, stop reason
├── plan.json                 # Research plan: subtopics, queries, success criteria
├── evidence/
│   ├── ledger.json           # Every candidate: considered, selected, rejected
│   └── ledger.md             # Human-readable evidence index
├── iterations/
│   ├── 000.json              # Per-iteration: new candidates, coverage delta, tool calls
│   └── 001.json
└── renders/
    ├── reading_path.md       # Ordered reading guide with synthesized prose
    └── backing_report.md     # Evidence rationale and gap analysis
```

Every source is traceable. Every decision is logged. Every iteration records what changed and why.

## Why This Exists

Most "deep research" tools are thin wrappers around search + summarize. They run one query, grab some results, and paste them into a prompt. There's no iteration, no convergence checking, no evidence quality tracking, and no way to inspect what happened.

This engine is different:

- **It iterates until convergence** — not a fixed number of steps, but until coverage metrics plateau or budgets exhaust. Six different stop rules prevent both premature termination and runaway costs.
- **It tracks evidence provenance** — every candidate gets a deterministic key (SHA256 of DOI/arXiv/URL), scores only ratchet up across iterations, and dedup follows DOI > arXiv > URL > title precedence.
- **It critiques its own output** — a different model provider reviews what the generator wrote. Citations are verified against actual evidence. This follows the [Microsoft Critique finding](https://arxiv.org/abs/2310.01851): separating generation from evaluation across providers produces measurably better results.
- **It's durable** — every phase boundary is a Kitaru checkpoint. If the process crashes at iteration 4, replay picks up from the last completed checkpoint. No work is lost.
- **It understands intent** — the classifier extracts structured preferences from natural language ("compare React vs Vue for our team" becomes `planning_mode: comparison`, `comparison_targets: ["React", "Vue"]`, `deliverable_mode: comparison_memo`).

## Built on Kitaru

This project showcases [Kitaru](https://kitaru.ai) as the durable execution backbone. Here's how the two layers split responsibilities:

```
Kitaru @flow / @checkpoint              PydanticAI Agent
┌──────────────────────────────┐       ┌─────────────────────────────┐
│ Owns:                        │       │ Owns:                       │
│  - Durable state & replay    │ calls │  - Model call routing       │
│  - Checkpoints & artifacts   │──────>│  - Structured output parse  │
│  - Wait / resume (HITL)      │       │  - Tool execution           │
│  - Convergence enforcement   │<──────│  - Response validation      │
│  - Cost tracking & budgets   │returns│                             │
│  - Stop rules & time boxes   │ typed │ Does NOT own:               │
│                              │result │  - Iteration state          │
│                              │       │  - When to stop             │
│                              │       │  - Checkpointing            │
└──────────────────────────────┘       └─────────────────────────────┘
```

PydanticAI is the structured completion layer — it calls models and returns typed results. Kitaru is the durable execution layer — it owns checkpoints, waits, replay, and convergence. No competing orchestration. The engine's research loop controls iteration, not an agent SDK.

### Kitaru Primitives Used

| Primitive | How It's Used |
|---|---|
| `@flow` | Top-level research orchestration — the single durable boundary |
| `@checkpoint` | Every phase boundary: classify, plan, search, score, write, critique, verify |
| `wait()` | Human-in-the-loop: clarify ambiguous briefs, approve research plans |
| `log()` | Structured iteration metadata: coverage, cost, tool calls, stop reasons |
| `.submit().load()` | Parallel checkpoint execution (council generators, dual renders, judge pair) |
| `kp.wrap()` | PydanticAI adapter — wraps agents for Kitaru observability |

## The Research Loop

![Deep Research Engine Flow](flow.png)
[Excalidraw URL](https://excalidraw.com/#json=vubTTSYSPtaap6xqmCMQu,14v_EHEIEMOk2RybWmGq7A)

```
classify_request (+ preferences) -> build_plan -> run_supervisor / council
-> execute_searches -> extract_candidates -> score_relevance -> update_ledger
-> enrich_candidates -> score_coverage -> [converged?] -> loop or continue

After convergence:

rank_evidence -> [deliverable_mode?]
  research_package: write_reading_path || write_backing_report
  other modes:      write_full_report
-> critique_reports -> apply_revisions -> verify_grounding || verify_coherence
-> assemble_package -> InvestigationPackage
```

### How It Works

1. **Classify** — Determine audience, freshness, complexity. Extract structured preferences (deliverable mode, source biases, comparison targets). Select a research tier.
2. **Plan** — Break the brief into subtopics, queries, sections, and success criteria. Preferences shape the plan structure.
3. **Iterate** — The supervisor (or council of N parallel supervisors) decides what to search. Built-in providers execute. Results are extracted, scored for relevance, merged into the evidence ledger, enriched with full page text, and coverage is scored. The loop continues until convergence or budget exhaustion.
4. **Render** — Rank evidence into reading order. Based on deliverable mode, write either a reading path + backing report (research package) or a single full report (other modes). Writer prompts adapt per mode.
5. **Critique** — A different model provider reviews the output. Citations are verified against evidence. Coherence is scored against the plan. Revisions are applied.
6. **Assemble** — Everything goes into a canonical `InvestigationPackage` with full provenance.

### Stop Rules

| Rule | Condition |
|---|---|
| Budget exhausted | `estimated_cost >= cost_budget_usd` |
| Time exhausted | `elapsed_seconds >= time_box_seconds` |
| Converged | `coverage >= min_coverage` AND no remaining gaps |
| Loop stall | Zero coverage gain in an iteration |
| Diminishing returns | Low gain + over 50% resource usage |
| Max iterations | Hard cap reached |

### Research Preferences

The classifier extracts a `ResearchPreferences` object from natural language. Two enforcement levels:

| Type | Fields | Enforcement |
|---|---|---|
| **Advisory** | `preferred_source_groups`, `audience`, `freshness`, `cost_bias`, `speed_bias` | Shape LLM decisions via prompt context |
| **Hard constraint** | `excluded_source_groups`, `excluded_providers` | Enforced at the provider registry — excluded providers never execute |

Structural preferences control the research shape:

| Field | Effect |
|---|---|
| `deliverable_mode` | `research_package`, `final_report`, `comparison_memo`, `recommendation_brief`, `answer_only` |
| `planning_mode` | `broad_scan`, `comparison`, `timeline`, `deep_dive`, `decision_support` |
| `comparison_targets` | Ensures balanced coverage across compared items |
| `time_window_days` | Constrains recency in search actions |

## Tiers

| Tier | Max Iterations | Cost Budget | Time Box | Critique | Judge | Council |
|---|---|---|---|---|---|---|
| Quick | 2 | $0.05 | 2 min | No | No | No |
| Standard | 3 | $0.10 | 10 min | No | No | No |
| Deep | 6 | $1.00 | 30 min | Yes | Yes | Available |
| Custom | Configurable | Configurable | Configurable | Configurable | Configurable | Available |

When `tier = "auto"`, the classifier detects complexity from the brief and selects automatically.

### Council Mode (Opt-In)

N parallel generators (default: 3) each produce independent supervisor decisions. The council aggregator merges and dedupes their `SearchAction`s before built-in provider execution. Available on deep and custom tiers.

## Configuration

### LLM Models

All LLM calls route through PydanticAI. Model strings use `provider/model-name` format. Changing a model is a config change, not a code change.

| Setting | Default | Role |
|---|---|---|
| `classifier_model` | `gemini/gemini-2.0-flash-lite` | Request classification |
| `planner_model` | `gemini/gemini-2.5-flash` | Plan generation |
| `supervisor_model` | `gemini/gemini-2.5-flash` | Research cycle supervisor |
| `relevance_scorer_model` | `gemini/gemini-2.5-flash` | Evidence relevance scoring |
| `writer_model` | `gemini/gemini-2.5-flash` | Report composition |
| `review_model` | `anthropic/claude-sonnet-4-20250514` | Cross-provider critique |
| `judge_model` | `openai/gpt-4o-mini` | Grounding + coherence verification |

### Environment Variables

All settings use the `RESEARCH_` prefix and are loaded via Pydantic Settings.

| Variable | Default | Description |
|---|---|---|
| `RESEARCH_DEFAULT_TIER` | `standard` | Default when tier not specified |
| `RESEARCH_DEFAULT_COST_BUDGET_USD` | `0.10` | Default cost ceiling |
| `RESEARCH_DAILY_COST_LIMIT_USD` | `10.00` | Global daily ceiling |
| `RESEARCH_CONVERGENCE_MIN_COVERAGE` | `0.60` | Minimum acceptable coverage |
| `RESEARCH_CONVERGENCE_EPSILON` | `0.05` | Minimum coverage delta to continue |
| `RESEARCH_MAX_TOOL_CALLS_PER_CYCLE` | `5` | Max tool calls per iteration |
| `RESEARCH_ENABLED_PROVIDERS` | `arxiv,semantic_scholar` | Built-in search providers |

### Provider API Keys

At least one LLM provider key must be configured. Built-in search providers (arXiv, Semantic Scholar) run with zero extra keys.

| Provider | Variable | Required |
|---|---|---|
| Google (Gemini) | `GEMINI_API_KEY` | At least one LLM key |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | At least one LLM key |
| OpenAI (GPT) | `OPENAI_API_KEY` | At least one LLM key |
| Brave Search | `BRAVE_API_KEY` | No |
| Exa | `EXA_API_KEY` | No |
| Semantic Scholar | `SEMANTIC_SCHOLAR_API_KEY` | No |

## Testing

```bash
uv run pytest tests/ -v
```

265 tests covering models, evidence pipeline, convergence logic, checkpoints, renderers, agent factories, flow orchestration, and real agent behavior with PydanticAI's `TestModel`.

## Design Principles

**Library-first.** The engine is a Python library. The `@flow` is the entry point. Wrap it in any API or CLI you want.

**Package-canonical.** The `InvestigationPackage` is the canonical output. Reports are rendered views derived from it. Different consumers get different surfaces from the same package.

**Immutable evidence.** Pydantic models use `extra="forbid"` and `model_copy(update=...)` throughout. The ratchet rule ensures evidence only accumulates — scores go up, snippets merge, nothing is lost.

**Prompts as markdown.** System prompts live in `prompts/*.md`, loaded at agent construction time. Prompt changes don't require code changes.

**Durable everything.** Every major phase boundary is a Kitaru checkpoint. Runs survive process crashes, support replay from any checkpoint, and produce inspectable artifacts at every iteration.

**Cross-provider critique.** The reviewer and judges use different providers than the generators. Self-evaluation is unreliable — the engine enforces model diversity for quality checks.

## Requirements

- Python 3.11+
- [Kitaru](https://kitaru.ai) — durable execution for AI workflows
- [PydanticAI](https://ai.pydantic.dev) — structured LLM completions
- Pydantic v2 + pydantic-settings

<details>
<summary><strong>Project Layout</strong></summary>

```
deep_research/
├── agents/                  # PydanticAI agent factories (Kitaru-wrapped)
│   ├── supervisor.py        #   Search supervisor with MCP tools + bash
│   ├── classifier.py        #   Request classification + preference extraction
│   ├── planner.py           #   Research plan generation
│   ├── relevance_scorer.py  #   Evidence relevance scoring
│   ├── reviewer.py          #   Cross-provider critique
│   ├── judge.py             #   Grounding + coherence verification
│   ├── writer.py            #   Report composition
│   └── curator.py           #   Evidence curation
├── checkpoints/             # Kitaru checkpoint functions
│   ├── classify.py          #   classify_request
│   ├── plan.py              #   build_plan
│   ├── supervisor.py        #   run_supervisor — decision + MCP capture
│   ├── council.py           #   run_council_generator — parallel generators
│   ├── search.py            #   execute_searches — built-in provider execution
│   ├── normalize.py         #   extract_candidates
│   ├── relevance.py         #   score_relevance
│   ├── merge.py             #   update_ledger
│   ├── fetch.py             #   enrich_candidates
│   ├── evaluate.py          #   score_coverage
│   ├── select.py            #   rank_evidence
│   ├── rendering.py         #   write_reading_path / write_backing_report / write_full_report
│   ├── review.py            #   critique_reports
│   ├── revise.py            #   apply_revisions
│   ├── grounding.py         #   verify_grounding
│   ├── coherence.py         #   verify_coherence
│   └── assemble.py          #   assemble_package
├── evidence/                # Evidence processing (no LLM calls)
│   ├── dedup.py             #   DOI > arXiv > URL > title precedence
│   ├── ledger.py            #   Ratchet merge (scores only go up)
│   ├── scoring.py           #   Heuristic quality by source kind
│   ├── url.py               #   URL canonicalization
│   └── resolution.py        #   Selected/coverage entry resolution
├── flow/
│   ├── research_flow.py     #   @flow — top-level orchestration
│   ├── convergence.py       #   Stop decision logic
│   └── costing.py           #   Token → USD estimation
├── models.py                # All Pydantic models (strict, immutable)
├── config.py                # ResearchConfig, tier defaults, env settings
├── enums.py                 # StopReason, Tier, SourceKind, DeliverableMode, PlanningMode
├── providers/
│   ├── __init__.py          #   build_supervisor_surface + ProviderRegistry
│   ├── mcp_config.py        #   MCPServerConfig + toolset factory
│   ├── normalization.py     #   Raw tool results → EvidenceCandidate
│   └── search/
│       ├── __init__.py      #   SearchProvider protocol + ProviderRegistry
│       ├── brave.py         #   Brave web search
│       ├── arxiv_provider.py
│       ├── semantic_scholar.py
│       ├── exa_provider.py
│       └── fetcher.py       #   URL fetch + HTML extraction
├── renderers/               # Pure functions: package data → deterministic scaffolds
│   ├── reading_path.py
│   ├── backing_report.py
│   ├── full_report.py
│   └── materialization.py   #   Writer synthesis + deliverable-mode prompt adaptation
├── tools/
│   ├── bash_executor.py     #   Allow-listed bash sandbox
│   └── state_reader.py      #   read_plan, read_gaps tools for supervisor
├── package/
│   ├── assembly.py          #   InvestigationPackage construction
│   └── io.py                #   write_package / read_package (JSON + markdown)
└── prompts/                 #   System prompts as markdown files
    ├── supervisor.md
    ├── planner.md
    ├── classifier.md
    └── ...
```

</details>

## Contributing

Contributions are welcome. This is an active project by the [ZenML](https://zenml.io) team.

- **Issues**: Bug reports and feature requests via [GitHub Issues](https://github.com/zenml-io/deep-research/issues)
- **PRs**: Fork, branch, test (`uv run pytest tests/`), and open a PR
- **Prompts**: System prompts live in `prompts/*.md` — prompt improvements are high-impact, low-risk contributions

## License

See [LICENSE](LICENSE) for details.
