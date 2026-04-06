# Deep Research Engine — Implementation Design

**Date:** 2026-04-06
**Status:** Design — pending approval

---

## Context

We are building a standalone deep research engine as a Python library powered by Kitaru (ZenML's durable execution layer for agents) and PydanticAI (structured LLM completions). The engine plans, iterates, evaluates, and synthesizes web research into a structured `InvestigationPackage` with markdown outputs.

This is a greenfield build in an empty repository. The unified specification (§1–§11) defines the full product scope. This design document captures the approved implementation architecture derived from that spec, incorporating decisions made during the design session.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Checkpoint granularity | Fine-grained (one per operation) | Failure replays only the failed operation, not the entire iteration |
| Tool loop ownership | PydanticAI owns tool-calling within checkpoints | Natural agentic behavior; KitaruAgent tracks each call as a child event |
| Observability | Kitaru-native for v1 | KitaruAgent already captures model calls, tool calls, tokens, cost. Langfuse deferred. |
| Service shape | Library-first (no FastAPI/CLI initially) | Kitaru flows are deployable via ZenML. API/CLI added later. |
| Council mode | Included in v1 | Via Kitaru `.submit()` for parallel generators |
| Providers | MCP + Bash (no Python fallback) | MCP servers preferred, CLI tools via bash sandbox. No hardcoded Python search code. At least one search tool must be configured. |
| Host adapters | None (fully standalone) | No Seshat adapter code. Engine is host-independent. |
| Model roles | Full 11 roles per spec | Each role configurable independently |
| CLI | Deferred | Until API layer is built |
| Evidence scoring | Hybrid: LLM relevance + heuristic quality | Follows patterns from GPT Researcher, STORM, DeerFlow |
| State navigation | PydanticAI state-reading tools + bash | LLM navigates accumulated state on demand (Mintlify ChromaFs pattern) |

---

## Architecture

### Single Flow, Many Checkpoints

The engine is a single `@flow` (`research_flow`) that orchestrates fine-grained `@checkpoint` functions. The flow body IS the orchestrator — there is no separate "control plane" service or class.

```
@flow research_flow(brief, tier, config)
  │
  ├── @checkpoint classify_request → RequestClassification
  ├── @checkpoint build_plan → ResearchPlan
  ├── kitaru.wait("approve_plan") — if plan_required
  │
  ├── ITERATION LOOP (until convergence or stop)
  │   ├── @checkpoint run_supervisor → ToolCallResults
  │   │   └── PydanticAI agent with search + state-reading + bash tools
  │   ├── @checkpoint normalize_evidence → NormalizedCandidates
  │   ├── @checkpoint score_relevance → ScoredCandidates (LLM)
  │   ├── @checkpoint merge_evidence → UpdatedLedger
  │   ├── @checkpoint evaluate_coverage → CoverageScore
  │   └── convergence check (in-flow Python, not a checkpoint)
  │
  ├── @checkpoint build_selection_graph → SelectionGraph
  ├── @checkpoint render_reading_path → RenderPayload (.md)
  ├── @checkpoint render_backing_report → RenderPayload (.md)
  ├── @checkpoint critique_output → ReviewedPayloads (if tier allows)
  ├── @checkpoint judge_grounding → GroundingVerdict (cross-provider)
  ├── @checkpoint judge_coherence → CoherenceVerdict (cross-provider)
  └── @checkpoint assemble_package → InvestigationPackage
```

### Three Functional Groups

**Research Checkpoints** — classify, plan, supervise (PydanticAI agent with tools), evaluate coverage. The supervisor agent owns the tool-calling loop within its checkpoint.

**Evidence Layer** — normalize, score relevance (LLM), merge, deduplicate, build selection graph. Search providers are MCP servers registered with PydanticAI; the supervisor agent calls them as tools.

**Output Layer** — render markdown (reading path, backing report, full report), cross-provider critique, grounding/coherence judges, package assembly.

### Kitaru + PydanticAI Layering

```
Kitaru @flow / @checkpoint          PydanticAI Agent (via KitaruAgent)
┌─────────────────────────┐        ┌────────────────────────┐
│ Owns:                   │        │ Owns:                  │
│  - Durable state        │  calls │  - Model call           │
│  - Checkpoints          │ ─────> │  - Tool execution       │
│  - Waits / resume       │        │  - Structured parsing   │
│  - Replay               │ <───── │  - Response validation  │
│  - Convergence checks   │ returns│                        │
│  - Cost enforcement     │ typed  │ Does NOT own:          │
│  - Stop rules           │ result │  - Iteration state      │
│                         │        │  - When to stop         │
│                         │        │  - Checkpointing        │
└─────────────────────────┘        └────────────────────────┘
```

Each PydanticAI agent is wrapped with `KitaruAgent` for automatic tracking of model calls, tool invocations, token usage, and cost as child events.

---

## Project Structure

```
deep_research/
  __init__.py
  config.py                     # ResearchSettings (Pydantic Settings, all env vars)
  models.py                     # InvestigationPackage + all 6 layers
  enums.py                      # StopReason, RunStatus, Tier, etc.

  flow/
    __init__.py
    research_flow.py            # @flow research_flow
    convergence.py              # Convergence checks, stop rules, coverage scoring

  checkpoints/
    __init__.py
    classify.py                 # @checkpoint classify_request
    plan.py                     # @checkpoint build_plan
    supervisor.py               # @checkpoint run_supervisor
    normalize.py                # @checkpoint normalize_evidence
    relevance.py                # @checkpoint score_relevance (LLM)
    merge.py                    # @checkpoint merge_evidence
    evaluate.py                 # @checkpoint evaluate_coverage
    select.py                   # @checkpoint build_selection_graph
    council.py                  # Council mode: parallel generators via .submit()

  agents/
    __init__.py
    planner.py                  # PydanticAI Agent → ResearchPlan
    supervisor.py               # PydanticAI Agent with search + state + bash tools
    classifier.py               # PydanticAI Agent → RequestClassification
    writer.py                   # PydanticAI Agent for report composition
    judge.py                    # PydanticAI Agent for grounding/coherence
    reviewer.py                 # PydanticAI Agent for critique
    curator.py                  # PydanticAI Agent for reading path curation
    aggregator.py               # PydanticAI Agent for council aggregation
    relevance_scorer.py         # PydanticAI Agent for evidence relevance

  providers/
    __init__.py
    mcp_config.py               # MCP server registration and configuration
    normalization.py            # Convert raw tool results → EvidenceCandidate

  evidence/
    __init__.py
    normalization.py            # Candidate normalization
    dedup.py                    # Deduplication (DOI > arXiv ID > URL > title)
    scoring.py                  # Heuristic quality scoring
    ledger.py                   # Evidence ledger operations (append-only)

  renderers/
    __init__.py
    reading_path.py             # Render reading_path.md
    backing_report.py           # Render backing_report.md
    full_report.py              # Render full_report.md (lazy)

  critique/
    __init__.py
    reviewer.py                 # Cross-provider critique
    grounding.py                # Grounding judge
    coherence.py                # Coherence judge
    cross_provider.py           # get_cross_provider_model utility

  package/
    __init__.py
    assembly.py                 # InvestigationPackage assembly
    io.py                       # Read/write package (md + json)

  tools/
    __init__.py
    state_reader.py             # PydanticAI tools: list_evidence, read_evidence, etc.
    bash_executor.py            # Sandboxed bash execution tool

  prompts/                      # System prompts as markdown files (git-tracked)
    supervisor.md               # Research supervisor — tool usage, search strategy
    planner.md                  # Research planner — decomposition, subtopics
    classifier.md               # Request classifier — audience, freshness, tier
    writer.md                   # Report writer — synthesis, citations
    judge_grounding.md          # Grounding judge — citation verification
    judge_coherence.md          # Coherence judge — relevance, flow, completeness
    reviewer.md                 # Cross-provider reviewer — critique dimensions
    curator.md                  # Reading path curator — ordering, bridge notes
    aggregator.md               # Council aggregator — evidence merging
    relevance_scorer.md         # Evidence relevance scorer — subtopic matching
    question_generator.md       # Clarifying question generator

tests/
  ...

pyproject.toml
```

---

## Research Loop Detail

### Flow Body

```python
@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    # 1. Classify request
    classification = classify_request(brief, config)
    resolved_tier = resolve_tier(classification, tier)

    # 2. Build plan
    plan = build_plan(brief, classification, resolved_tier)

    # 3. Approve plan (if required)
    if resolved_tier.requires_plan_approval:
        approval = kitaru.wait(
            schema=PlanApproval,
            question=f"Approve research plan?\n{plan.summary}",
            name="approve_plan",
        )
        if not approval.approved:
            plan = build_plan(brief, classification, resolved_tier,
                            feedback=approval.feedback)

    # 4. Iterate until convergence
    ledger = EvidenceLedger.empty()
    iteration_records = []

    for iteration in range(resolved_tier.max_iterations):
        if config and config.council_mode and resolved_tier.allows_council:
            raw_results = run_council_iteration(plan, ledger, iteration, config)
        else:
            raw_results = run_supervisor(plan, ledger, iteration, resolved_tier)

        candidates = normalize_evidence(raw_results)
        scored = score_relevance(candidates, plan)
        ledger = merge_evidence(scored, ledger)
        coverage = evaluate_coverage(ledger, plan)
        iteration_records.append(coverage.to_record(iteration))

        stop = check_convergence(coverage, iteration, resolved_tier)
        if stop.should_stop:
            kitaru.log(stop_reason=stop.reason)
            break

    # 5. Build selection graph
    selection = build_selection_graph(ledger, plan)

    # 6. Render (eager)
    reading_path = render_reading_path(selection)
    backing_report = render_backing_report(selection, ledger, plan)

    # 7. Critique (if enabled)
    if resolved_tier.critique_enabled:
        reading_path = critique_output(reading_path, ledger)
        backing_report = critique_output(backing_report, ledger)

    # 8. Judge (cross-provider)
    if resolved_tier.judge_enabled:
        grounding = judge_grounding(reading_path, backing_report, ledger)
        coherence = judge_coherence(reading_path, backing_report, plan)
    else:
        grounding, coherence = None, None

    # 9. Assemble
    return assemble_package(
        plan=plan, ledger=ledger, selection=selection,
        renders={"reading_path": reading_path, "backing_report": backing_report},
        iterations=iteration_records,
        grounding=grounding, coherence=coherence,
    )
```

### Convergence Model

Coverage score C(j) at iteration j:
```
C(j) = mean(subtopic_coverage, source_diversity, evidence_density)
```

Stop when: `C(j) - C(j-1) < epsilon AND C(j) >= C_min`

Additional stop rules (priority order):
1. Coverage convergence
2. Diminishing returns (last 2 iterations < 2 new candidates each)
3. Budget exhaustion
4. Time exhaustion
5. Max iterations (hard cap: 10)
6. Loop stall (3 consecutive zero-candidate iterations)

---

## Supervisor Agent & Tools

The supervisor is a PydanticAI agent with three categories of tools:

### Search Tools (MCP + Bash)

Two ways for the agent to search:
1. **MCP servers** (preferred) — registered as PydanticAI tools, discovered at runtime. Structured input/output.
2. **Bash sandbox** — agent uses `run_bash` for any CLI tool or curl command in the environment

No hardcoded Python search functions. At least one search tool must be configured (MCP server or available CLI).

```
Supervisor Agent sees:
  ├── MCP tools (if configured) — discovered at runtime
  ├── run_bash — sandboxed bash for CLI tools, curl, data processing
  ├── State-reading tools — list_evidence, read_evidence, read_plan, etc.
```

Adding a new provider = connecting an MCP server. The agent can also use any CLI via bash.

The supervisor's system prompt includes tool usage guidance:
- When to search web vs academic sources
- How to formulate effective queries per provider
- When to fetch/read URLs vs using title/snippet
- How to check coverage state before searching more
- Budget awareness (remaining cost, iteration count)

### State-Reading Tools (Mintlify-inspired)
- `list_evidence(filter_by)` — list candidates by subtopic/score/provider
- `read_evidence(candidate_key)` — read full content of a specific candidate
- `read_plan()` — read the research plan
- `read_iteration_summary(iteration)` — summary of a specific iteration
- `read_coverage()` — current coverage scores per subtopic
- `read_gaps()` — subtopics still needing evidence

### Execution Tools
- `run_bash(command)` — sandboxed bash execution for data processing

State-reading tools query the accumulated `EvidenceLedger` and `IterationTrace` passed as PydanticAI dependencies. The LLM navigates accumulated research state on demand rather than receiving everything in the prompt.

---

## Council Mode

When `council_mode=true` on deep/custom tiers:

1. Planning phase runs once (shared plan)
2. N parallel generators fan out via `checkpoint.submit()`:
   ```python
   futures = [
       run_council_generator.submit(plan, iteration, model_i, id=f"gen_{i}")
       for i, model_i in enumerate(council_models)
   ]
   ```
3. Each generator runs its own supervisor agent independently
4. Aggregator merges N evidence ledgers: union of candidates, best-score-wins dedup
5. Aggregator model is cross-provider
6. Rendering and critique proceed on merged evidence

Each generator uses a different model when possible (cross-lab diversity). Falls back to same-model with different temperature if only one provider is configured.

---

## Evidence Pipeline

```
MCP tool results (raw, unstructured)
  → normalize (convert to common EvidenceCandidate schema)
  → deduplicate (DOI > arXiv ID > Semantic Scholar ID > URL > title similarity)
  → LLM relevance scoring (PydanticAI agent scores each candidate vs plan subtopics)
  → heuristic quality scoring (source type, authority, freshness)
  → merge into evidence ledger (append-only, ratchet rule)
```

**MCP + Bash providers:**
Search tools are MCP servers registered with PydanticAI, or CLI tools used via the bash sandbox. The supervisor agent decides what to call. Raw results are normalized into `EvidenceCandidate` by the normalization checkpoint.

Adding a new provider = connecting an MCP server. No hardcoded Python search code.

Tool failure never fails the run. Unavailable MCP servers are skipped.

**Quality floor:** Candidates with heuristic quality < 0.3 are excluded from selection.

**Ratchet rule:** Evidence only accumulates. Coverage score monotonically increases. Per-iteration state is a durable artifact.

---

## Renderers & Critique

### Three Markdown Renderers

| Renderer | Timing | Cost | Description |
|----------|--------|------|-------------|
| `reading_path.md` | Eager | Low | Ordered resources with rationale, subtopic coverage, bridge notes |
| `backing_report.md` | Eager | Medium | Selection rationale, collective coverage, exclusions, gaps |
| `full_report.md` | Lazy | High | Full cited synthesis with sections, inline citations, references. Triggered by calling `render_full_report(package)` after the flow completes — not produced during the flow itself. |

### Cross-Provider Critique

- Reviewer model must be from a different provider than the generator (DRACO finding: +7.0 points)
- Reviewer receives: rendered markdown + evidence ledger + quality scores
- Outputs: per-dimension scores + revision suggestions
- Generator revises based on critique (1 round standard, up to 2 rounds deep)
- Reviewer never directly rewrites

### Judges (Cross-Provider)

**Grounding judge:** Verifies each `[citation]` is actually supported by the cited source. Returns `(key, supported, reason)` per citation.

**Coherence judge:** Scores 4 sub-dimensions (relevance, logical flow, completeness, consistency), each 0.0–1.0.

Both judges use the same cross-provider model. Skipped on quick tier.

---

## Evaluation Layer

Five quality dimensions:

| Dimension | Method | Description |
|-----------|--------|-------------|
| Coverage | Heuristic | subtopics with sources / total subtopics |
| Source Authority | Heuristic | Weighted score from source metadata |
| Grounding | LLM judge (cross-provider) | Claims with verified citation support |
| Coherence | LLM judge (cross-provider) | Relevance + flow + completeness + consistency |
| Novelty | Heuristic | Proportion of new sources (1.0 when no host context) |

Composite: `Q = mean(coverage, authority, grounding, coherence, novelty)`

---

## Configuration

All via environment variables with `RESEARCH_` prefix:

```python
class ResearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_")

    # Tier defaults
    default_tier: str = "standard"
    default_max_iterations: int = 3
    default_cost_budget_usd: float = 0.10
    daily_cost_limit_usd: float = 10.00

    # Convergence
    convergence_epsilon: float = 0.05
    convergence_min_coverage: float = 0.60

    # Limits
    max_tool_calls_per_cycle: int = 5
    tool_timeout_sec: int = 20
    source_quality_floor: float = 0.30

    # Council
    council_size: int = 3
    council_cost_budget_usd: float = 2.00

    # 11 Model roles
    supervisor_model: str = "gemini/gemini-2.5-flash"
    critic_model: str = "gemini/gemini-2.5-flash"
    writer_model: str = "gemini/gemini-2.5-flash"
    grounding_model: str = "openai/gpt-4o-mini"
    classifier_model: str = "gemini/gemini-2.0-flash-lite"
    planner_model: str = "gemini/gemini-2.5-flash"
    question_model: str = "gemini/gemini-2.0-flash-lite"
    curator_model: str = "gemini/gemini-2.0-flash-lite"
    aggregator_model: str = "openai/gpt-4o-mini"
    review_model: str = "anthropic/claude-sonnet-4-20250514"
    judge_model: str = "openai/gpt-4o-mini"
```

---

## File Output (Markdown-First)

Each run produces:

```
runs/{run_id}/
  summary.md              # Human-readable run summary
  plan.md                 # Research plan
  plan.json               # Structured ResearchPlan
  evidence/
    ledger.json           # Full evidence ledger
    ledger.md             # Human-readable evidence summary
  renders/
    reading_path.md       # Ordered reading path
    backing_report.md     # Selection rationale
    full_report.md        # Full synthesis (lazy)
  evaluation/
    grounding.json        # Citation verification
    quality_scores.json   # All 5 dimensions
  package.json            # Full InvestigationPackage
  iterations/
    001.json              # Per-iteration trace
    002.json
```

---

## Build Order

### Phase 1: Scaffold & Contracts
- Python package (`pyproject.toml` with `uv`)
- All Pydantic models (InvestigationPackage, 6 layers, enums)
- `ResearchSettings` class
- `SearchProvider` protocol
- Empty module structure for all packages

### Phase 2: Evidence Plane + Tools
- MCP server configuration (connect Brave Search MCP, etc.)
- Bash sandbox tool (sandboxed execution)
- State-reading tools (list_evidence, read_evidence, read_plan, etc.)
- Raw result normalization (any tool output → EvidenceCandidate)
- Deduplication, heuristic quality scoring
- LLM relevance scorer (PydanticAI agent)
- Evidence ledger operations
- System prompts as markdown files in `prompts/`

### Phase 3: Research Loop
- `@flow research_flow`
- Checkpoints: classify, plan, supervisor, normalize, score_relevance, merge, evaluate
- PydanticAI agents: classifier, planner, supervisor (with search + state + bash tools)
- Convergence checks and all stop rules
- `kitaru.wait()` for plan approval
- Council mode via `.submit()`

### Phase 4: Package & Renderers
- InvestigationPackage assembly
- File I/O (markdown + JSON)
- `reading_path` renderer (eager, curator agent)
- `backing_report` renderer (eager, writer agent)
- `full_report` renderer (lazy, writer agent)

### Phase 5: Critique & Evaluation
- Cross-provider model selection utility
- Reviewer agent (critique)
- Grounding judge agent
- Coherence judge agent
- Quality score computation (5 dimensions)
- Critique-revision loop

### Phase 6: Integration & Testing
- End-to-end test with real providers
- Replay verification from multiple checkpoint boundaries
- Cost tracking verification
- Council mode testing
- Markdown output validation

---

## Cross-Design Invariants

1. The canonical output is `InvestigationPackage`. Reports and paths are rendered views.
2. Every claim in rendered output must be grounded in evidence with inspectable provenance.
3. Evaluation uses a different provider than generation (cross-provider judges).
4. Convergence, budget, and stop reasons are explicit and machine-readable.
5. Provider failures never fail the run.
6. Evidence only accumulates (ratchet rule).
7. Cost is tracked and enforced, never unbounded.
8. Every checkpoint produces durable artifacts sufficient for replay.
9. The engine is fully functional without any host context.
10. Markdown is the primary human-facing format.

---

## Deferred from v1

- FastAPI HTTP API
- CLI
- MCP interface
- Frontend UI
- Seshat/host adapters
- Langfuse/external observability
- Fine-tuned models
- Multi-tenancy

## MVP Operations

- Launch: `uv run python -c "from deep_research.flow.research_flow import research_flow; handle = research_flow.run('brief'); print(handle.exec_id)"`
- Provide wait input: `kitaru executions input <exec_id> --value true`
- Resume: `kitaru executions resume <exec_id>`
- Replay from checkpoint: `kitaru executions replay <exec_id> --from build_plan`
