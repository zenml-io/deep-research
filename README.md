# Deep Research

An autonomous research agent that iteratively gathers, scores, and synthesizes evidence to answer complex questions -- built on [Kitaru](https://github.com/zenml-io/kitaru), ZenML's durable execution framework.

## What it does

Deep Research takes a natural-language research brief and produces a comprehensive investigation package: a structured evidence ledger, a curated selection of the most relevant sources, and rendered reports ready for consumption. The agent classifies your request, builds a research plan, then enters an iterative loop where an LLM-powered supervisor searches for evidence, scores it for relevance and quality, and merges findings into a growing knowledge base.

Each iteration measures coverage against the research plan. The loop stops automatically when evidence converges (coverage meets the threshold), returns diminish below epsilon, or budget/time limits are reached. This means the agent does exactly as much work as the question requires -- quick lookups finish in seconds while deep investigations run multiple passes.

For high-stakes research, council mode runs multiple supervisor instances in parallel (potentially with different models) and aggregates their findings, reducing single-model blind spots. The final output includes both a reading-path summary and a detailed backing report with full citations.

## Built on Kitaru

[Kitaru](https://github.com/zenml-io/kitaru) is a durable execution framework by ZenML. Every step in the research pipeline is a **checkpoint** -- if the process crashes or is interrupted, it replays from the last completed checkpoint rather than re-running (and re-paying for) expensive LLM calls.

Key benefits for research workflows:

- **Checkpoint replay** -- LLM calls, evidence normalization, and scoring are all checkpointed. A crash mid-iteration resumes exactly where it left off.
- **Wait points** -- The flow can pause for human input (clarifying ambiguous briefs, approving plans) and resume later without losing state.
- **Crash recovery** -- Long-running deep research (up to 30 minutes, multiple dollars in API costs) is protected against infrastructure failures.
- **Parallel execution** -- Council mode submits multiple supervisor checkpoints concurrently via `submit()` / `load()`.

## Quick start

### Installation

```bash
uv pip install -e .
```

### Basic usage

```python
from deep_research.flow.research_flow import research_flow

# Run with automatic tier selection
package = research_flow(brief="What are the trade-offs between RAG and fine-tuning for domain-specific LLM applications?")

# Access results
print(package.run_summary.stop_reason)
print(package.evidence_ledger.entries)
for render in package.renders:
    print(render.content_markdown)
```

### Specifying a tier

```python
# Quick lookup (2 iterations, $0.05 budget)
package = research_flow(brief="What is RLHF?", tier="quick")

# Deep investigation (6 iterations, $1.00 budget, critique + judge enabled)
package = research_flow(brief="Compare transformer architectures for time-series forecasting", tier="deep")
```

### Custom configuration

```python
from deep_research.config import ResearchConfig
from deep_research.enums import Tier

config = ResearchConfig.for_tier(Tier.DEEP)
# Override defaults via environment variables prefixed with RESEARCH_
# e.g. RESEARCH_SUPERVISOR_MODEL=openai/gpt-4o
```

## Architecture

The research pipeline follows this sequence:

```
classify --> plan --> [iterate] --> select --> render --> assemble
                        |
                        +-- supervisor (or council)
                        +-- normalize
                        +-- score relevance
                        +-- merge into ledger
                        +-- evaluate coverage
                        +-- check convergence
```

Each box is a Kitaru **checkpoint** that is individually durable and replayable.

| Checkpoint | Purpose |
|---|---|
| `classify_request` | Determines tier, audience, freshness needs; flags ambiguous briefs |
| `build_plan` | Generates goal, key questions, subtopics, queries, and success criteria |
| `run_supervisor` | LLM-driven search agent that calls tools to gather raw evidence |
| `run_council_generator` | Parallel supervisor variant for council mode |
| `normalize_evidence` | Converts raw tool results into uniform `EvidenceCandidate` objects |
| `score_relevance` | LLM-scored relevance and quality against the research plan |
| `merge_evidence` | Deduplicates and adds scored candidates to the ledger |
| `evaluate_coverage` | Measures subtopic coverage, source diversity, and evidence density |
| `build_selection_graph` | Curates the best evidence with rationale for each selection |
| `render_reading_path` | Produces a concise reading-path summary |
| `render_backing_report` | Produces a detailed report with full citations |
| `assemble_package` | Bundles everything into the final `InvestigationPackage` |

## Configuration

Research behavior is controlled by `ResearchConfig`, which can be constructed from tier presets or customized directly.

**Environment variables** (prefix `RESEARCH_`):

| Variable | Default | Description |
|---|---|---|
| `RESEARCH_DEFAULT_TIER` | `standard` | Default research tier |
| `RESEARCH_DEFAULT_MAX_ITERATIONS` | `3` | Max iterations for standard/custom tiers |
| `RESEARCH_DEFAULT_COST_BUDGET_USD` | `0.10` | Cost budget for standard/custom tiers |
| `RESEARCH_CONVERGENCE_EPSILON` | `0.05` | Minimum coverage gain to continue |
| `RESEARCH_CONVERGENCE_MIN_COVERAGE` | `0.60` | Coverage threshold to stop (converged) |
| `RESEARCH_COUNCIL_SIZE` | `3` | Number of parallel supervisors in council mode |
| `RESEARCH_SUPERVISOR_MODEL` | `gemini/gemini-2.5-flash` | Model for supervisor agent |
| `RESEARCH_CLASSIFIER_MODEL` | `gemini/gemini-2.0-flash-lite` | Model for request classification |
| `RESEARCH_PLANNER_MODEL` | `gemini/gemini-2.5-flash` | Model for plan generation |

## Tiers

| Setting | QUICK | STANDARD | DEEP | CUSTOM |
|---|---|---|---|---|
| Max iterations | 2 | 3 | 6 | 3 |
| Cost budget (USD) | $0.05 | $0.10 | $1.00 | $0.10 |
| Time box (seconds) | 120 | 600 | 1800 | 600 |
| Critique enabled | No | No | Yes | No |
| Judge enabled | No | No | Yes | No |
| Council allowed | No | No | Yes | Yes |
| Plan approval required | Yes | Yes | Yes | Yes |

The `auto` tier (default) uses the classifier to pick `QUICK`, `STANDARD`, or `DEEP` based on the brief's complexity.

## Human-in-the-loop

The research flow includes two **wait points** that pause execution for human input:

### Clarify brief

If the classifier determines the brief is ambiguous, the flow pauses with a `clarify_brief` wait point. The operator receives the clarification question and provides a refined brief. The flow then re-classifies and continues.

```python
# The wait surfaces as:
# wait(name="clarify_brief", schema=str, question="<clarification question>")
```

### Approve plan

When `require_plan_approval` is enabled (default for all tiers), the flow pauses after plan generation with an `approve_plan` wait point. The operator reviews the plan and approves or rejects it.

```python
# The wait surfaces as:
# wait(name="approve_plan", schema=bool, question="Approve plan for: <goal>?")
```

Both wait points are Kitaru primitives -- the flow state is fully persisted, and execution resumes exactly where it paused regardless of how much time passes.

## Development

### Project structure

```
deep_research/
    agents/          # PydanticAI agent factories
    checkpoints/     # Kitaru checkpoints (one per pipeline step)
    flow/            # Flow orchestration and convergence logic
    models.py        # Pydantic data models
    config.py        # Tier presets and configuration
    enums.py         # Tier, StopReason, SourceKind enums
    prompts/         # System prompts as markdown files
    renderers/       # Report rendering (reading path, backing report)
    tools/           # Search and execution tools
    evidence/        # Evidence scoring utilities
    providers/       # Provider normalization
tests/               # Unit and integration tests
```

### Running tests

```bash
pytest
```

### Requirements

- Python 3.11+
- Dependencies: `kitaru`, `pydantic>=2`, `pydantic-ai`, `pydantic-settings`

## License

Apache 2.0
