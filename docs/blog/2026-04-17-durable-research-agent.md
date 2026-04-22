---
title: "Building a Durable Research Agent on Kitaru and PydanticAI"
date: 2026-04-17
author: ZenML
tags: [kitaru, pydantic-ai, durable-execution, ai-agents, deep-research]
---

![Figure 01 — The flow, statically](../diagrams/figure-01-flow.png)
*Figure 01 — One question in, one investigation package out. Eight typed artifacts travel left-to-right; the middle band is a loop that runs until the supervisor says stop, or the budget does.*

A research question like *"what are the tradeoffs of MoE versus dense transformers for inference latency"* is the kind of thing a careful engineer investigates over an afternoon — reading a dozen papers, tracking provenance across providers, cross-checking claims, and writing up the findings with citations. [`zenml-io/deep-research`](https://github.com/zenml-io/deep-research) is an open-source agent that turns that question into a structured `InvestigationPackage`: a final report with inline citations, an evidence ledger dedup'd across providers, and per-iteration records of what the supervisor asked and what the subagents found.

Running the agent once is not the hard part. Running it reliably is. A multi-phase research pipeline has three failure modes that compound each other:

- **Agent output is free-form text.** Every next phase has to re-parse it, usually with a second LLM call.
- **Long runs crash.** On a naive pipeline, a minute-9 failure in a ten-minute run throws away the scope, the plan, and every iteration of search that preceded it. Retry is a full re-pay.
- **Self-evaluation is biased.** A reviewer that's the same model as the generator tends to agree with itself more than with ground truth.

The engine addresses each with a framework that takes that slice of the problem seriously. [PydanticAI](https://ai.pydantic.dev) gives every agent a typed `output_type` so contracts fail loudly at the boundary. [Kitaru](https://kitaru.ai) makes every phase a `@checkpoint` with a cached return, so a crash only costs the in-flight phase. Cross-provider review is enforced structurally — generator, reviewer, and judge run on three different LLM providers.

The stack choice is narrow on purpose. A `@flow` is a Python function. A `@checkpoint` is a Python function. `KitaruAgent` wraps a PydanticAI `Agent` without changing its signature. Nothing here is a graph engine, a DAG, or an actor model. Regular research code opts into durability and typing as orthogonal properties.

## Three-layer architecture

The engine separates orchestration, durability, and reasoning into three layers that don't leak into each other.

![Three-layer architecture](../diagrams/architecture.png)

**Flow** — `research/flows/deep_research.py` contains one `@flow`-decorated function. It owns the iteration loop, convergence checks, budget tracking, and phase sequencing. The flow body is the only place where "what happens next" logic lives.

**Checkpoints** — `research/checkpoints/` contains one `@checkpoint` function per research phase, tagged as either `"llm_call"` or `"tool_call"`. Checkpoints are the replay boundary. Kitaru caches a completed checkpoint's result, so on replay the function is not re-executed — the cached output is returned in its place. Checkpoints are the unit at which the engine remembers progress.

**Agents** — `research/agents/` contains pure typed transformers. Each factory builds a PydanticAI `Agent` with an `output_type`, wraps it in `KitaruAgent` from `kitaru.adapters.pydantic_ai`, and returns it. Agents know nothing about iteration state, budget, or when to stop. They take typed input and return typed output.

```python
def _build_agent(
    model_name: str,
    *,
    name: str,
    prompt_name: str,
    output_type: type,
    tools: list[Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> BudgetAwareAgent:
    from pydantic_ai import Agent
    from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
    from research.prompts import get_prompt

    kwargs: dict[str, Any] = {
        "output_type": output_type,
        "system_prompt": get_prompt(prompt_name).text,
    }
    if tools:
        kwargs["tools"] = tools
    if model_settings is not None:
        kwargs["model_settings"] = model_settings

    agent = Agent(model_name, **kwargs)
    kitaru_agent = KitaruAgent(
        agent, name=name, capture=CapturePolicy(tool_capture="full")
    )
    return BudgetAwareAgent(kitaru_agent, model_name=model_name)
```

Each factory is thirty lines. There is no agent base class hierarchy, no registry of "agent roles," no custom runtime. The factory produces an `Agent`, hands it to `KitaruAgent` for durable capture, wraps the result in a `BudgetAwareAgent` to attribute token spend, and returns it to the checkpoint that called it.

## The research pipeline

The flow walks through eight phases in order, each producing a typed artifact.

![Pipeline overview](../diagrams/pipeline.png)

1. **Scope** — `run_scope(question, model)` produces a `ResearchBrief`: the sharpened question, audience, success criteria, and out-of-scope notes.
2. **Plan** — `run_plan(brief, model)` produces a `ResearchPlan`: ranked subtopics, priors, and search angles. On tiers that require it, the flow pauses on a `kitaru.wait()` for operator approval before continuing.
3. **Iteration loop** — the flow enters a bounded loop. On each iteration, a supervisor agent reads the windowed ledger projection and emits a `SupervisorDecision`: a set of subagent tasks with provider hints. The flow fans those tasks out in parallel, merges results into the `ManagedLedger` (append-only, dedup by DOI > arXiv ID > canonical URL), and checks convergence.
4. **Draft** — once the loop exits, `run_draft(brief, plan, ledger, model)` produces a `DraftReport` with inline citations tied back to ledger entries.
5. **Critique** — `run_critique(draft, plan, ledger, ...)` produces a `CritiqueReport`. On `deep` tier, two reviewers run independently and their critiques are merged with per-dimension disagreement tracking.
6. **Supplemental loop** — if the critique sets `require_more_research=True` and the supplemental budget allows, the flow re-enters the iteration loop for a bounded number of extra cycles (default 1).
7. **Finalize** — `run_finalize(draft, critique, model)` produces a `FinalReport` with critique feedback folded in.
8. **Assemble and export** — the flow builds an `InvestigationPackage` and exports it to the output directory.

Convergence is decided by four stop rules in strict priority order, evaluated between iterations: (1) cost budget exhausted, (2) wall-clock time exhausted, (3) supervisor signals `done=True`, (4) max iterations reached. The priority matters — a supervisor that declares victory while the budget allows more work is respected on most tiers but explicitly overridden on `exhaustive`.

```python
@flow
def deep_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
    output_dir: str | None = None,
    require_plan_approval: bool = True,
) -> InvestigationPackage:
    cfg = config or ResearchConfig.for_tier(tier)
    tracker = BudgetTracker(
        budget=cfg.budget,
        strict_unknown_model_cost=cfg.strict_unknown_model_cost,
    )
    tracker_token = set_active_tracker(tracker)
    try:
        stamp_h = stamp_run_metadata.submit()
        stamp = stamp_h.load()

        gen_model = cfg.slots["generator"].model_string
        scope_model = cfg.scope_override.model_string if cfg.scope_override else gen_model
        scope_h = run_scope.submit(question, scope_model)
        brief = scope_h.load()

        plan_h = run_plan.submit(brief, gen_model)
        plan = plan_h.load()

        if require_plan_approval:
            _await_plan_approval(question, plan, cfg)
        # ... iteration loop, draft, critique, finalize, assemble, export
```

Every phase is `.submit()` then `.load()`. The submission is the checkpoint boundary; the load is where the flow observes the result. Between the two, Kitaru has captured enough to replay.

A concrete run looks like this — question in, package out:

```
$ uv run python run_v2.py "Tradeoffs of MoE vs dense transformers for inference latency"
[scope]        ResearchBrief    — 2 subtopics, 4 success criteria
[plan]         ResearchPlan     — 7 queries across arxiv, semantic_scholar, exa
[iter 0]       SupervisorDecision(done=False) → 3 subagents → 12 findings
[iter 1]       SupervisorDecision(done=False) → 3 subagents → 9 findings
[iter 2]       SupervisorDecision(done=True)  → stop: supervisor done
[draft]        DraftReport      — 1840 words, 31 inline citations
[critique]     CritiqueReport   — 6 dimensions, require_more_research=False
[finalize]     FinalReport      — 1912 words, grounding_ratio=0.84
[assemble]     InvestigationPackage  — 47 evidence items after dedup
→ run-a1b2c3d4/
  ├── package.json        # serialized InvestigationPackage
  ├── report.md           # final report, inline citations → ledger
  ├── evidence/ledger.json
  └── iterations/000.json, 001.json, 002.json
```

Every line above is a typed Pydantic model produced by a checkpointed phase. The supervisor's decision, the subagents' findings, and the critique dimensions are all persisted, keyed, and replayable.

## Durability — what survives a crash

Kitaru's durability model is narrow and explicit. Completed `@checkpoint` results are persisted by the runtime and returned on replay — the function body does not run a second time. On a process crash, the flow can be resumed: every completed checkpoint returns its cached value, and only the checkpoint that was in flight re-executes. A nine-minute run that died mid-critique re-runs the critique, not the scope, not the plan, not the four iterations of supervised search that came before it. The durability panel in the diagram above (the timeline under the pipeline) shows the cached-vs-re-run state explicitly.

This only works if the flow body is deterministic given checkpoint outputs. The engine enforces that by confining non-determinism — UUID generation, wall-clock snapshots — to `research/checkpoints/metadata.py`. `stamp_run_metadata` is itself a checkpoint, so its values (run id, start timestamp, git sha, settings snapshot) are cached on first run and returned verbatim on replay. Everything downstream is a pure function of checkpoint outputs. That discipline is not a Kitaru requirement; it is a property the engine adds on top so replay is safe.

The same mechanism carries operator input across crashes. `kitaru.wait()` pauses the flow until a client provides a value, and the wait itself is durable — the flow process can be killed while waiting, the operator can take hours to respond, and the resumed flow still sees every prior checkpoint as cached. The engine uses this in two places: plan approval before research starts, and generator selection in council mode (below). Logfire instrumentation is auto-enabled when the SDK and a token are present and is useful for per-phase latency and cost visibility; it is orthogonal to replay, not a substitute for it.

```python
@checkpoint(type="llm_call")
def run_critique(
    draft: DraftReport,
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    model_name: str,
    second_model_name: str | None = None,
    disagreement_threshold: float = 0.3,
) -> CritiqueReport:
    """On standard tier: single reviewer. On deep tier: two reviewers
    produce independent critiques that are merged."""
    prompt = json.dumps({
        "draft": draft.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
    }, indent=2)

    if second_model_name is None:
        agent = build_reviewer_agent(model_name)
        result = agent.run_sync(prompt).output
        return CritiqueReport(
            dimensions=result.dimensions,
            require_more_research=result.require_more_research,
            issues=result.issues,
            reviewer_provenance=result.reviewer_provenance,
            reviewer_disagreements=[],
        )
```

The checkpoint owns the prompt construction and the agent invocation. The agent owns the typed output. The flow owns whether to call the checkpoint at all. Each layer is replaceable without touching the others.

## Cross-provider critique and typed contracts

The default model slots are deliberately split across providers: generator is `anthropic:claude-sonnet-4-6`, subagent is `google-gla:gemini-3.1-flash-lite-preview`, reviewer is `openai:gpt-5.4-mini`, judge is `google-gla:gemini-3.1-pro-preview`. This is not a performance story — it is a correlation story. Models from the same provider tend to share training data, tokenizers, and failure modes. A reviewer from a different provider catches a different class of errors than a reviewer from the same lineage, and self-evaluation has well-documented pathologies that cross-provider review avoids.

On the `deep` tier, the engine runs two reviewers in parallel — OpenAI and Gemini — and merges their critiques. `CritiqueReport.reviewer_disagreements` captures per-dimension deltas above a threshold, which surfaces the dimensions where the reviewers genuinely disagreed rather than burying them in a scalar average.

![Cross-provider critique](../diagrams/critique.png)

Every artifact crossing an agent boundary is a Pydantic model with `extra="forbid"`. If the LLM returns a field the contract doesn't declare, PydanticAI raises rather than silently dropping it. If it omits a required field, it raises. This is the whole story for structured output in the engine — there is no retry loop for malformed JSON, because PydanticAI handles that, and there is no post-hoc schema massaging, because the schema is the source of truth.

```python
class StrictBase(BaseModel):
    """Base model that rejects unknown fields."""
    model_config = ConfigDict(extra="forbid")


class CritiqueReport(StrictBase):
    """Structured critique of a draft report."""
    dimensions: list[CritiqueDimensionScore]
    require_more_research: bool
    issues: list[str] = []
    reviewer_provenance: list[str] = []
    reviewer_disagreements: list[ReviewerDisagreement] = []
```

The evidence ledger sits behind the same discipline but never calls an LLM. `research/ledger/` is pure functions: append a snippet, dedup against existing entries by DOI > arXiv ID > canonical URL, project a windowed view for the next agent. Recent iterations stay in full context; older iterations compact to summaries. The same ledger instance flows through every phase, and every citation in the final report resolves back to a ledger entry with provider, URL, and fetch timestamp.

## Council mode and the exhaustive tier

Two modes extend the default pipeline in different directions.

**Council mode** (`council_research()` in `research/flows/council.py`) runs the full pipeline once per generator model — Claude, GPT, and Gemini each produce an independent `InvestigationPackage`. A judge agent compares the outputs on a fixed set of dimensions. The flow then pauses on a `kitaru.wait()` for an operator to pick the canonical generator. Selection resumes the flow, which exports the chosen package. Because the wait is durable and every pipeline run behind it is a sequence of completed checkpoints, the operator can take hours to respond, the process can crash, and the flow still resumes with all work intact. This is a durable human-in-the-loop pattern with no custom orchestration on top.

**Exhaustive tier** targets surveys rather than answers. It raises max iterations to 20, parallel subagents to 10, and the budget to $3.00. Critically, it sets `respect_supervisor_done=False` — the supervisor's `done=True` signal is ignored, and the loop runs until budget, time, or max iterations stop it. The supervisor prompt on this tier is biased for breadth: maximum source diversity, varied providers, no early stopping. The other tiers trust the supervisor to know when it has enough; this one deliberately does not.

| Tier | Max iter | Parallel subagents | Budget | Supervisor done |
|------|---------:|-------------------:|-------:|----------------:|
| `quick` | 2 | 3 | $0.10 | respected |
| `standard` | 5 | 3 | $0.10 | respected |
| `deep` | 10 | 3 | $0.10 | respected |
| `exhaustive` | 20 | 10 | $3.00 | ignored |

```bash
# Standard run
uv run python run_v2.py "What are the latest advances in RLHF alternatives?"

# Exhaustive: 20 iterations, 10 parallel subagents, $3 budget
uv run python run_v2.py --tier exhaustive "Comprehensive survey of transformer architectures"

# Council: parallel runs across generator models + operator selection
uv run python run_v2.py --council "My research question"
```

## What the composition buys

The product is the engine. A question goes in, an `InvestigationPackage` comes out, every claim in the final report resolving back to a ledger entry with provider, URL, and fetch timestamp. The interesting engineering is what makes that reliable: typed contracts at every agent edge so malformed completions fail loudly at the boundary; durable phase boundaries so a minute-9 crash costs one phase rather than nine; cross-provider review so generation and evaluation never share a failure mode.

None of the three is novel at its own layer. The engine does not ship a new orchestration primitive, a new agent abstraction, or a new evaluation method. It refuses to write those primitives itself — it delegates each to a framework that already takes that slice of the problem seriously, and keeps the research code pure Python between them. The interesting bits of the system are the boundaries, not the abstractions.

The full source is at [github.com/zenml-io/deep-research](https://github.com/zenml-io/deep-research). Kitaru lives at [kitaru.ai](https://kitaru.ai), and PydanticAI at [ai.pydantic.dev](https://ai.pydantic.dev).
