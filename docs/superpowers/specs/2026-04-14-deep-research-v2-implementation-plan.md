# Deep Research V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Built on kitaru-authoring skill** for all Kitaru checkpoint/flow/adapter patterns.

**Goal:** Implement the V2 deep research system as a clean break from V1, on Kitaru and PydanticAI, with cross-provider critique as a load-bearing quality mechanism and the investigation package as the primary product artifact.

**Architecture:** Three-layer system: `@flow` (thin orchestration) -> `@checkpoint` (durable replay boundaries) -> PydanticAI agents (typed LLM judgment). Temporal-seam module separation: `contracts/`, `config/`, `prompts/`, `agents/`, `checkpoints/`, `flows/`, `ledger/`, `providers/`, `package/`. Evidence ledger is append-only with deterministic windowed projection. Cross-provider critique is mandatory in every flow.

**Tech Stack:** Python 3.11+, Kitaru (durable execution), PydanticAI (structured LLM completions), pydantic-settings (config), httpx (HTTP), uv (package manager)

**Design Spec:** `docs/superpowers/specs/2026-04-13-deep-research-v2-design.md`

**Worktree:** `.worktrees/v2` (branch: `v2`)

---

## Phase 1: Foundation — Scaffolding, Contracts, Config, Prompts

> Create the V2 package skeleton, all Pydantic data contracts, the config system, and the prompt infrastructure. This phase produces no runnable flow but establishes every type boundary that downstream phases depend on.

### Task 1.1: V2 Package Skeleton

**Files:**
- Create: `research/__init__.py`
- Create: `research/contracts/__init__.py`
- Create: `research/config/__init__.py`
- Create: `research/flows/__init__.py`
- Create: `research/checkpoints/__init__.py`
- Create: `research/agents/__init__.py`
- Create: `research/ledger/__init__.py`
- Create: `research/providers/__init__.py`
- Create: `research/package/__init__.py`
- Create: `research/prompts/__init__.py`
- Create: `research/mcp/__init__.py` (empty, deferred)
- Modify: `pyproject.toml` (V2 package config)

- [x] Create the `research/` directory tree with empty `__init__.py` files matching the Runtime Topology from the design spec
- [x] Update `pyproject.toml`: change package name to `research`, update `[tool.setuptools.packages.find]` to `include = ["research*"]`, keep existing deps (kitaru, pydantic>=2, pydantic-ai>=1.75, logfire, pydantic-settings, httpx>=0.27, arxiv>=2.1.3)
- [x] Add `[tool.setuptools.package-data]` entry for `"research.prompts" = ["*.md"]`
- [x] Verify `uv sync` succeeds in the worktree
- [x] Commit: `feat(v2): scaffold V2 package skeleton`

### Task 1.2: Core Data Contracts — Brief, Plan, Evidence

**Files:**
- Create: `research/contracts/brief.py`
- Create: `research/contracts/plan.py`
- Create: `research/contracts/evidence.py`
- Create: `tests/test_contracts.py`

- [x] Define `StrictBase` model (Pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")`)
- [x] Define `ResearchBrief` in `contracts/brief.py`: typed normalization of user request (topic, audience, scope, freshness constraints, etc.)
- [x] Define `ResearchPlan` in `contracts/plan.py`: goal, key questions, subtopics, query strategies, success criteria
- [x] Define `SubagentTask` in `contracts/plan.py`: task description, target subtopic, search strategy hints
- [x] Define `EvidenceItem` in `contracts/evidence.py`: synthesized value, optional verbatim excerpts, canonical ID (DOI/arXiv/URL), source metadata, `confidence_notes: str | None`
- [x] Define `EvidenceLedger` in `contracts/evidence.py`: `items: list[EvidenceItem]`, append/dedupe interface
- [x] Write tests: strict model rejects extra fields, required fields enforced, `schema_version` present on durable types
- [x] Run tests: `uv run pytest tests/test_contracts.py -v`
- [x] Commit: `feat(v2): add brief, plan, and evidence contracts`

### Task 1.3: Core Data Contracts — Decisions, Reports, Package

**Files:**
- Create: `research/contracts/decisions.py`
- Create: `research/contracts/reports.py`
- Create: `research/contracts/package.py`
- Create: `research/contracts/iteration.py`
- Modify: `tests/test_contracts.py`

- [x] Define `SubagentFindings` in `contracts/decisions.py`: synthesized findings, source references, optional excerpts, `confidence_notes: str | None`
- [x] Define `SupervisorDecision` in `contracts/decisions.py`: `done: bool`, `rationale: str`, `gaps: list[str]`, `subagent_tasks: list[SubagentTask]`, `pinned_evidence_ids: list[str]`
- [x] Define `IterationRecord` in `contracts/iteration.py`: iteration index, supervisor decision, subagent results, ledger snapshot size, cost, timing
- [x] Define `DraftReport` in `contracts/reports.py`: markdown content with inline `[evidence_id]` citations, section structure
- [x] Define `CritiqueReport` in `contracts/reports.py`: per-dimension scores (source_reliability, completeness, grounding), `require_more_research: bool`, issues list, per-reviewer provenance for deep tier
- [x] Define `FinalReport` in `contracts/reports.py`: inherits citation discipline from `DraftReport`
- [x] Define `InvestigationPackage` in `contracts/package.py`: `schema_version: str = "1.0"`, brief, plan, ledger, iteration records, draft, critique, final report, supervisor decisions, prompt hashes, run metadata
- [x] Define `CouncilComparison` in `contracts/package.py`: judge output comparing generators
- [x] Define `CouncilPackage` in `contracts/package.py`: `canonical_generator: str | None`, `council_provider_compromise: bool`, both generators' full `InvestigationPackage` outputs
- [x] Write tests: all contract models validate correctly, `schema_version` defaults to `"1.0"`, `CritiqueReport` has all three dimensions, `SupervisorDecision` has all required fields
- [x] Run tests
- [x] Commit: `feat(v2): add decision, report, and package contracts`

### Task 1.4: Config System

**Files:**
- Create: `research/config/budget.py`
- Create: `research/config/slots.py`
- Create: `research/config/settings.py`
- Create: `research/config/defaults.py`
- Create: `tests/test_config.py`

- [x] Define `BudgetConfig` in `config/budget.py`: `soft_budget_usd: float`, `hard_budget_usd: float | None`, `spent_usd: float = 0.0`
- [x] Define `ModelSlot` enum in `config/slots.py`: `generator`, `subagent`, `reviewer`, `judge`
- [x] Define `ModelSlotConfig` in `config/slots.py`: provider, model string, per-token input/output cost
- [x] Define `TierConfig` in `config/defaults.py`: slot-to-model mappings per tier (quick/standard/deep), `max_iterations`, `max_parallel_subagents`, per-provider concurrency sub-ceilings
- [x] Define default tier mappings from spec: quick (Anthropic Sonnet / Gemini Flash / OpenAI lightweight / —), standard (Anthropic Sonnet / Gemini Flash / OpenAI / Gemini Pro), deep (Anthropic Sonnet / Gemini Flash / OpenAI + Gemini Pro reviewer / Gemini Pro)
- [x] Define `ResearchSettings` in `config/settings.py` via pydantic-settings: `RESEARCH_*` env prefix, all model names, API keys, `ledger_window_iterations` (default 3), `grounding_min_ratio` (default 0.7), `max_supplemental_loops` (default 1, hard cap), `wait_timeout_seconds` (default 3600), `allow_unfinalized_package` (default False), `strict_unknown_model_cost` (default False), `sandbox_enabled` (default False), `sandbox_backend: str | None`, `max_parallel_subagents`, per-provider sub-ceilings, `daily_cost_limit_usd`
- [x] Define `ResearchConfig` (frozen) in `config/settings.py`: `.for_tier(tier)` factory that assembles all sub-configs from settings + tier defaults. Analogous to V1's `ResearchConfig`
- [x] Wire `config/__init__.py` to re-export `ResearchConfig`, `ResearchSettings`, `BudgetConfig`
- [x] Write tests: `for_tier()` produces valid config for each tier, budget config tracks spend, settings load from env vars, tier defaults match spec table
- [x] Run tests
- [x] Commit: `feat(v2): add config system with budget, model slots, and tier defaults`

### Task 1.5: Prompts Infrastructure

**Files:**
- Create: `research/prompts/loader.py`
- Create: `research/prompts/registry.py`
- Create: `research/prompts/scope.md` (placeholder)
- Create: `research/prompts/planner.md` (placeholder)
- Create: `research/prompts/supervisor.md` (placeholder)
- Create: `research/prompts/subagent.md` (placeholder)
- Create: `research/prompts/generator.md` (placeholder)
- Create: `research/prompts/reviewer.md` (placeholder)
- Create: `research/prompts/finalizer.md` (placeholder)
- Create: `research/prompts/council_judge.md` (placeholder)
- Create: `tests/test_prompts.py`

- [x] Implement `loader.py`: reads all `.md` files from the `prompts/` directory at import time. Parses optional YAML front-matter for `version`. Computes `sha256` of file content
- [x] Implement `registry.py`: module-level `PROMPTS: dict[str, PromptRecord]` where `PromptRecord` exposes `.text`, `.sha256`, `.version`
- [x] Create placeholder `.md` prompt files for each agent role (scope, planner, supervisor, subagent, generator, reviewer, finalizer, council_judge). Each file should have YAML front-matter with `version: 0.1.0` and a brief description of the role
- [x] Wire `prompts/__init__.py` to expose `PROMPTS` registry
- [x] Write tests: all prompt files load, each has `.text`/`.sha256`/`.version`, SHA256 is stable for same content, no duplicate prompt names
- [x] Run tests
- [x] Commit: `feat(v2): add prompt loader with SHA256 tracking and PROMPTS registry`

---

## Phase 2: Evidence Ledger & Providers

> Build the pure-function evidence layer (no LLM calls) and the search/fetch provider infrastructure. After this phase, subagents will have tools to work with.

### Task 2.1: Canonical ID Resolution & URL Canonicalization

**Files:**
- Create: `research/ledger/canonical.py`
- Create: `research/ledger/url.py`
- Create: `tests/test_ledger.py`

- [x] Implement `canonical.py`: extract DOI, arXiv ID, or canonical URL from evidence metadata. Precedence: DOI -> arXiv ID -> canonical URL. If no stable identifier, retain raw URL
- [x] Implement `url.py`: strip tracking parameters (utm_*, fbclid, etc.), normalize trailing slashes, follow source resolution close to fetch time. Bias toward over-inclusion (avoid incorrect merging)
- [x] Write tests: DOI extraction from various URL formats, arXiv ID extraction, URL canonicalization strips trackers, trailing slash normalization, raw URL fallback
- [x] Run tests
- [x] Commit: `feat(v2): add canonical ID resolution and URL canonicalization`

### Task 2.2: Evidence Ledger — Append, Dedupe, Merge

**Files:**
- Create: `research/ledger/ledger.py`
- Create: `research/ledger/dedup.py`
- Modify: `tests/test_ledger.py`

- [x] Implement `dedup.py`: exact dedup on canonical ID (DOI -> arXiv ID -> canonical URL). No fuzzy matching. Over-include rather than incorrectly merge
- [x] Implement `ledger.py`: `EvidenceLedger` class with `append(items)`, `merge(findings)`, `len`, iteration tracking. Append-only semantics. No novelty scoring, no ratchet scoring, no source priors, no heuristic evidence ranking (all removed from V2 per spec)
- [x] Wire `ledger/__init__.py` to re-export `EvidenceLedger`
- [x] Write tests: append adds items, duplicate DOIs collapse to one entry, duplicate arXiv IDs collapse, duplicate canonical URLs collapse, items with no stable ID treated as unique, ledger is append-only (no removal)
- [x] Run tests
- [x] Commit: `feat(v2): add evidence ledger with append-only dedup`

### Task 2.3: Ledger Projection & Windowing

**Files:**
- Create: `research/ledger/projection.py`
- Modify: `tests/test_ledger.py`

- [x] Implement `projection.py`: pure function `project_ledger(ledger, iteration_index, pinned_ids, window_iterations=3)` that produces the supervisor's compressed view
- [x] Full view for items in last `window_iterations` iterations: title, source type, canonical ID, subagent synthesis text
- [x] Compact view for older items: title + canonical ID + one-line synthesis excerpt
- [x] Items with `evidence_id` in `pinned_evidence_ids` always shown in full regardless of age
- [x] Projection is a pure function of (ledger state, iteration index, prior decision's pins) — fully replayable
- [x] Write tests: items within window shown in full, older items compacted, pinned items always full, projection is deterministic for same inputs, empty ledger returns empty projection
- [x] Run tests
- [x] Commit: `feat(v2): add deterministic ledger projection with windowing and pinning`

### Task 2.4: Search Providers

**Files:**
- Create: `research/providers/search.py`
- Create: `research/providers/brave.py`
- Create: `research/providers/arxiv_provider.py`
- Create: `research/providers/semantic_scholar.py`
- Create: `research/providers/exa_provider.py`
- Create: `research/providers/_http.py`
- Create: `tests/test_providers.py`

- [x] Define `SearchProvider` Protocol in `providers/search.py`: `async search(query, ...) -> list[SearchResult]`
- [x] Define `SearchResult` dataclass: url, title, snippet, provider name, raw metadata
- [x] Define `ProviderRegistry` in `providers/search.py`: register providers by name, resolve enabled providers from config
- [x] Implement `_http.py`: shared httpx async client factory with configurable timeout
- [x] Implement `brave.py`: Brave search provider (no-op when API key absent)
- [x] Implement `arxiv_provider.py`: arXiv search provider (no API key needed)
- [x] Implement `semantic_scholar.py`: Semantic Scholar provider (works without key, higher rate limits with key)
- [x] Implement `exa_provider.py`: Exa search provider (no-op when API key absent)
- [x] Wire `providers/__init__.py` to re-export registry and providers
- [x] Write tests: registry resolves enabled providers, providers no-op gracefully without keys, search returns typed results
- [x] Run tests
- [x] Commit: `feat(v2): add search provider protocol with brave, arxiv, semantic_scholar, exa`

### Task 2.5: Fetch Provider & Agent Tool Surface

**Files:**
- Create: `research/providers/fetch.py`
- Create: `research/providers/code_exec.py`
- Create: `research/providers/agent_tools.py`
- Modify: `tests/test_providers.py`

- [x] Implement `fetch.py`: async content fetcher (httpx-based, with timeout and size limits)
- [x] Implement `code_exec.py`: sandbox execution interface, gated by `config.sandbox_enabled`. When disabled, `code_exec` is omitted from tool surface (not emulated). Safety: ephemeral, time-bounded, no network by default, data passed in explicitly
- [x] Implement `agent_tools.py`: the inward MCP-style tool surface subagents call. Exposes `search`, `fetch`, and conditionally `code_exec`. Only `subagent` slot is bound to this surface (supervisor, generator, reviewer, finalizer, judge have NO tool access)
- [x] Write tests: agent_tools exposes correct tools when sandbox enabled/disabled, tool surface rejects non-subagent access pattern, fetch handles timeout
- [x] Run tests
- [x] Commit: `feat(v2): add fetch, code_exec, and agent tool surface`

---

## Phase 3: Agents — PydanticAI + Kitaru Wrapping

> Build all PydanticAI agent factories. Each agent is wrapped with `kp.wrap()` from `kitaru.adapters.pydantic_ai` following the kitaru-authoring skill pattern: wrap once at module scope, call inside explicit `@checkpoint(type="llm_call")`.

### Task 3.1: Agent Wrapping Infrastructure

**Files:**
- Create: `research/agents/_wrap.py`
- Create: `tests/test_agents.py`

- [x] Implement `_wrap.py`: helper that imports `kitaru.adapters.pydantic_ai as kp` and provides `wrap_agent(agent, *, name=None)` that calls `kp.wrap(agent, tool_capture_config={"mode": "full"})`. This is the single place where Kitaru adapter is imported for agents
- [x] Follow kitaru-authoring pattern: wrap agent at module scope, NOT inside checkpoint function
- [x] Write test: wrapped agent can be called with PydanticAI `TestModel`, wrapping does not alter output type
- [x] Run tests
- [x] Commit: `feat(v2): add agent wrapping infrastructure for Kitaru adapter`

### Task 3.2: Scoping & Planning Agents

**Files:**
- Create: `research/agents/scope.py`
- Create: `research/agents/planner.py`
- Modify: `research/prompts/scope.md`
- Modify: `research/prompts/planner.md`
- Modify: `tests/test_agents.py`

- [x] Write the `scope.md` system prompt: normalize raw user request into a typed `ResearchBrief`. May request clarification
- [x] Implement `scope.py`: `build_scope_agent(model_name)` factory. Output type: `ResearchBrief`. No tools. Wrapped with `kp.wrap()`
- [x] Write the `planner.md` system prompt: produce a typed `ResearchPlan` from the approved brief
- [x] Implement `planner.py`: `build_planner_agent(model_name)` factory. Output type: `ResearchPlan`. No tools. Wrapped with `kp.wrap()`
- [x] Write tests with `TestModel`: scope agent returns valid `ResearchBrief`, planner agent returns valid `ResearchPlan`
- [x] Run tests
- [x] Commit: `feat(v2): add scope and planner agents`

### Task 3.3: Supervisor Agent

**Files:**
- Create: `research/agents/supervisor.py`
- Modify: `research/prompts/supervisor.md`
- Modify: `tests/test_agents.py`

- [x] Write the `supervisor.md` system prompt: this is the highest-leverage prompt in V2. It replaces V1's convergence engine, source priors, novelty logic, and replan heuristics. Reads brief, plan, windowed ledger summary, remaining budget, iteration index. Returns `SupervisorDecision` with done/rationale/gaps/subagent_tasks/pinned_evidence_ids
- [x] Implement `supervisor.py`: `build_supervisor_agent(model_name)` factory. Output type: `SupervisorDecision`. NO tool access (structural guard against supervisor-as-executor creep). Wrapped with `kp.wrap()`
- [x] Write tests: supervisor returns valid `SupervisorDecision`, `done=True` terminates, `done=False` proposes tasks, `pinned_evidence_ids` is a list of strings
- [x] Run tests
- [x] Commit: `feat(v2): add supervisor agent (highest-leverage prompt)`

### Task 3.4: Subagent

**Files:**
- Create: `research/agents/subagent.py`
- Modify: `research/prompts/subagent.md`
- Modify: `tests/test_agents.py`

- [x] Write the `subagent.md` system prompt: per-task research synthesis. Extracts DOI, arXiv ID, or canonical URL while constructing each `EvidenceItem`. Includes `confidence_notes` for uncertainty
- [x] Implement `subagent.py`: `build_subagent_agent(model_name, tools)` factory. Output type: `SubagentFindings`. Has tool access (search, fetch, optionally code_exec from `agent_tools.py`). Wrapped with `kp.wrap()`
- [x] This is the ONLY agent slot bound to the inward tool surface
- [x] Write tests with `TestModel`: subagent returns valid `SubagentFindings`, tools are properly bound
- [x] Run tests
- [x] Commit: `feat(v2): add subagent with tool access (search, fetch, code_exec)`

### Task 3.5: Generator, Reviewer, Finalizer Agents

**Files:**
- Create: `research/agents/generator.py`
- Create: `research/agents/reviewer.py`
- Create: `research/agents/finalizer.py`
- Modify: `research/prompts/generator.md`
- Modify: `research/prompts/reviewer.md`
- Modify: `research/prompts/finalizer.md`
- Modify: `tests/test_agents.py`

- [x] Write `generator.md`: produce first complete report draft from ledger. Inline `[evidence_id]` citations mandatory on substantive claims
- [x] Implement `generator.py`: `build_generator_agent(model_name)`. Output type: `DraftReport`. No tools
- [x] Write `reviewer.md`: DRACO-style rubric critique. Three dimensions: source reliability, completeness, grounding. Produces `require_more_research: bool`. Deep tier: two reviewers on different providers, critiques merged (union of issues, scores averaged)
- [x] Implement `reviewer.py`: `build_reviewer_agent(model_name)`. Output type: `CritiqueReport`. No tools. Provider-crossing checkpoint (different provider than generator)
- [x] Write `finalizer.md`: apply critique while preserving generator's voice and framing. Inherits citation discipline
- [x] Implement `finalizer.py`: `build_finalizer_agent(model_name)`. Output type: `FinalReport`. No tools
- [x] Write tests: all three agents return valid typed output, generator includes citation placeholders, reviewer has all three dimensions, finalizer preserves structure
- [x] Run tests
- [x] Commit: `feat(v2): add generator, reviewer, and finalizer agents`

### Task 3.6: Judge Agent (Council)

**Files:**
- Create: `research/agents/judge.py`
- Modify: `research/prompts/council_judge.md`
- Modify: `tests/test_agents.py`

- [x] Write `council_judge.md`: compare two generators' outputs, produce structured comparison. Should run on provider distinct from both generators
- [x] Implement `judge.py`: `build_judge_agent(model_name)`. Output type: `CouncilComparison`. No tools
- [x] Write tests: judge returns valid `CouncilComparison`
- [x] Run tests
- [x] Commit: `feat(v2): add council judge agent`

---

## Phase 4: Checkpoints — Kitaru Durable Boundaries

> Implement all `@checkpoint` functions. Each checkpoint is a typed replay boundary following kitaru-authoring patterns: no nested checkpoints, no waits inside checkpoints, serializable outputs, agents wrapped at module scope and called inside checkpoints.

### Task 4.1: Scope & Plan Checkpoints

**Files:**
- Create: `research/checkpoints/scope.py`
- Create: `research/checkpoints/plan.py`
- Create: `research/checkpoints/metadata.py`
- Create: `tests/test_checkpoints.py`

- [x] Implement `metadata.py`: isolate non-determinism (UUID generation, wall-clock timestamps). On Kitaru replay, these return cached values. Following V1 pattern from `checkpoints/metadata.py`
- [x] Implement `scope.py`: `@checkpoint(type="llm_call")` that runs the scope agent, returns `ResearchBrief`
- [x] Implement `plan.py`: `@checkpoint(type="llm_call")` that runs the planner agent, returns `ResearchPlan`
- [x] Write tests: scope checkpoint returns valid brief, plan checkpoint returns valid plan, metadata checkpoint returns stable UUIDs on replay
- [x] Run tests
- [x] Commit: `feat(v2): add scope, plan, and metadata checkpoints`

### Task 4.2: Supervisor & Subagent Checkpoints

**Files:**
- Create: `research/checkpoints/supervisor.py`
- Create: `research/checkpoints/subagent.py`
- Modify: `tests/test_checkpoints.py`

- [x] Implement `supervisor.py`: `@checkpoint(type="llm_call")` that runs the supervisor agent with (brief, plan, windowed ledger projection, remaining budget, iteration index), returns `SupervisorDecision`
- [x] Implement `subagent.py`: `@checkpoint(type="llm_call")` that runs the subagent for a single task, returns `SubagentFindings`. Called via Kitaru `.submit().load()` for parallel fan-out. Subagent failures degrade to typed results instead of failing entire run
- [x] Write tests: supervisor checkpoint produces valid decision, subagent checkpoint produces findings, subagent handles tool failure gracefully (typed degraded result)
- [x] Run tests
- [x] Commit: `feat(v2): add supervisor and subagent checkpoints with fan-out`

### Task 4.3: Report Pipeline Checkpoints

**Files:**
- Create: `research/checkpoints/draft.py`
- Create: `research/checkpoints/critique.py`
- Create: `research/checkpoints/finalize.py`
- Modify: `tests/test_checkpoints.py`

- [x] Implement `draft.py`: `@checkpoint(type="llm_call")` — generator agent produces `DraftReport` from ledger
- [x] Implement `critique.py`: `@checkpoint(type="llm_call")` — reviewer agent produces `CritiqueReport`. This is the mandatory provider-crossing checkpoint. Deep tier: run two reviewers on different providers, merge critiques deterministically (union of issues, scores averaged). Single reviewer failure on deep tier is tolerated; both fail = error
- [x] Implement `finalize.py`: `@checkpoint(type="llm_call")` — finalizer agent applies critique to produce `FinalReport`. On failure: draft and critique preserved. Shipping draft-with-critique requires `allow_unfinalized_package` config flag
- [x] Write tests: draft checkpoint produces report with citations, critique checkpoint crosses providers, finalize checkpoint applies critique. Test failure modes: finalizer failure with/without `allow_unfinalized_package`
- [x] Run tests
- [x] Commit: `feat(v2): add draft, critique, and finalize checkpoints`

### Task 4.4: Package Assembly Checkpoint

**Files:**
- Create: `research/checkpoints/assemble.py`
- Modify: `tests/test_checkpoints.py`

- [x] Implement `assemble.py`: `@checkpoint(type="tool_call")` — ZERO LLM calls. Computes derived metadata, runs mechanical checks (grounding density, schema validity, citation ID resolution against ledger), materializes `InvestigationPackage`. Records prompt SHA256 hashes from the PROMPTS registry
- [x] Grounding density check: substantive sentences with valid `[evidence_id]` must cover `config.grounding_min_ratio` (default 0.7). Every referenced ID must resolve to a ledger entry. Failure stops assembly with draft+critique preserved
- [x] Write tests: assembly makes zero LLM calls (patch `Agent.run`/`run_sync` to raise), grounding density below threshold fails, all citation IDs must resolve to ledger entries, assembly produces valid `InvestigationPackage` with `schema_version="1.0"`
- [x] Run tests
- [x] Commit: `feat(v2): add assemble_package checkpoint with grounding enforcement`

---

## Phase 5: Default Flow — Kitaru Orchestration

> Wire all checkpoints into the default `@flow` with the full pipeline: scope -> plan -> supervisor_iteration* -> run_subagent* -> draft -> critique -> optional supplemental loop -> finalize -> assemble.

### Task 5.1: Budget Accounting

**Files:**
- Create: `research/flows/budget.py`
- Create: `tests/test_budget.py`

- [x] Implement budget tracking at the PydanticAI-to-Kitaru adapter boundary: wrapped agent adapter reads token usage from each model response, multiplies by configured per-model rate, increments `BudgetConfig.spent_usd`
- [x] Handle unknown models: record usage, price at $0, emit loud warning, annotate in audit trail. With `strict_unknown_model_cost=True`, hard-fail instead
- [x] Budget checked between iterations, NOT mid-checkpoint. Draft/critique/finalize/assemble run unconditionally on accumulated ledger (some overshoot accepted for always producing a deliverable)
- [x] `hard_budget_usd` (null by default): fails adapter on overshoot regardless of phase
- [x] Write tests: budget increments on model calls, unknown model handled per config, budget check between iterations, hard budget fails adapter
- [x] Run tests
- [x] Commit: `feat(v2): add budget accounting at adapter boundary`

### Task 5.2: Convergence & Stop Rules

**Files:**
- Create: `research/flows/convergence.py`
- Create: `tests/test_convergence.py`

- [x] Implement stop rules checked in priority order: budget exhausted -> time exhausted -> supervisor done -> max iterations. Budget stop records a stop reason and still runs downstream draft/critique/finalize path
- [x] Write tests: each stop rule fires correctly, priority ordering is respected, budget stop still produces deliverable
- [x] Run tests
- [x] Commit: `feat(v2): add convergence stop rules`

### Task 5.3: Default Flow Orchestration

**Files:**
- Create: `research/flows/deep_research.py`
- Create: `tests/test_flow.py`

- [x] Implement `deep_research.py` as a `@flow`-decorated function. Intentionally thin, delegates to checkpoints
- [x] Pipeline phases:
  1. `scope` checkpoint -> `ResearchBrief` (may `wait()` for clarification at flow level)
  2. `plan_research` checkpoint -> `ResearchPlan` (may optionally `wait()` for plan approval)
  3. Iteration loop: `supervisor_iteration` -> fan-out `run_subagent` via `.submit().load()` -> merge findings into ledger -> budget check -> convergence check
  4. `draft_report` checkpoint
  5. `critique_report` checkpoint (provider-crossing)
  6. If `require_more_research` AND supplemental budget not consumed: re-enter iteration loop ONCE (cap at 1), then re-draft, re-critique. Second `require_more_research` recorded but ignored
  7. `finalize_report` checkpoint
  8. `assemble_package` checkpoint -> `InvestigationPackage`
- [x] Flow-level `wait()` calls bounded by `config.wait_timeout_seconds`. On timeout, fail with typed error, last checkpoint replayable. No silent auto-proceed
- [x] Subagent fan-out: at most `max_parallel_subagents` concurrent. Per-provider sub-ceiling. Additional tasks queue within same iteration, never spill across iteration boundaries
- [x] Write tests: happy path one-iteration, multi-iteration loop, supplemental loop triggered and capped, subagent fan-out respects concurrency ceiling, wait timeout behavior
- [x] Run tests
- [x] Commit: `feat(v2): add default flow orchestration`

### Task 5.4: Failure Semantics

**Files:**
- Modify: `research/flows/deep_research.py`
- Modify: `tests/test_flow.py`

- [x] Subagent failures: typed degraded result, supervisor decides if sufficient
- [x] Supervisor failures: one retry with stricter reprompt, second failure = flow error with last valid ledger preserved
- [x] Reviewer failures: no silent fallback to shipping draft. Draft preserved, `critique_report` replayable. Deep tier single reviewer failure tolerated (surviving critique used, failure recorded); both fail = error
- [x] Finalizer failures: draft+critique preserved. With `allow_unfinalized_package`: produce package with `FinalReport=None`, `stop_reason="finalizer_failed"`. Without: run errors, operator replays
- [x] Assembly failures: code/schema bugs, fix-and-replay. Grounding density failures surface here
- [x] Budget exhaustion: recorded stop reason, downstream path still runs
- [x] Wait timeout: fail after `wait_timeout_seconds`, last checkpoint preserved
- [x] Write tests for each failure path
- [x] Run tests
- [x] Commit: `feat(v2): add comprehensive failure semantics`

---

## Phase 6: Package Assembly & run.py

> Materialize packages and build the operator entrypoint.

### Task 6.1: Package Materialization & Export

**Files:**
- Create: `research/package/assembly.py`
- Create: `research/package/export.py`
- Create: `tests/test_package.py`

- [x] Implement `assembly.py`: materialize `InvestigationPackage` from all accumulated data. Compute derived metadata. Run mechanical checks. Record prompt hashes
- [x] Implement `export.py`: serialize package to JSON, write to filesystem. Re-hydration from JSON
- [x] Write tests: round-trip serialization, package schema validation, export produces valid JSON
- [x] Run tests
- [x] Commit: `feat(v2): add package materialization and export`

### Task 6.2: run.py Operator Entrypoint

**Files:**
- Create: `run.py`
- Create: `tests/test_run.py`

- [x] Implement `run.py`: dispatch-only surface. Resolves arguments, loads config, calls appropriate flow. Must NOT contain research policy or business logic
- [x] CLI flags: `--tier` (quick/standard/deep), `--enable-sandbox`, `--allow-unfinalized`, `--council`, `--output` (output directory)
- [x] Keep `run.py` small and replaceable by a future `cli/` package
- [x] Write tests: `run.py` stays below configured line count, does not embed reusable helpers (imports runtime entry functions only), flags map to config correctly
- [x] Run tests
- [x] Commit: `feat(v2): add run.py operator entrypoint`

---

## Phase 7: Council Flow

> Separate opt-in product mode for exposing disagreement across generators.

### Task 7.1: Council Flow Orchestration

**Files:**
- Create: `research/flows/council.py`
- Modify: `tests/test_flow.py`

- [x] Implement `council.py` as a separate `@flow`: runs the full default pipeline once per generator model
- [x] Judge checkpoint: runs on a provider distinct from both generators. If only two providers configured, warn loudly and record `council_provider_compromise: true` in `CouncilPackage`
- [x] After judge emits `CouncilComparison`, flow enters `wait()` for operator selection (which generator's `FinalReport` is canonical)
- [x] Selection stamped into `CouncilPackage.canonical_generator`, chosen `InvestigationPackage` surfaced as primary artifact
- [x] `CouncilPackage` always preserves BOTH generators' full packages
- [x] Write tests: council runs both generators, judge crosses providers, operator selection wait, compromise recording, both packages preserved
- [x] Run tests
- [x] Commit: `feat(v2): add council flow with multi-generator comparison`

---

## Phase 8: Guard Tests & Integration Tests

> Protect architectural invariants. Fast guard tests on every push, integration tests for flow-level behaviors.

### Task 8.1: Fast Guard Tests

**Files:**
- Create: `tests/test_invariants.py`

- [x] **Provider topology test**: assert every flow has at least one provider-crossing checkpoint in its default tier mapping. Standard tier: `critique_report` crosses
- [x] **assemble_package purity test**: patch `pydantic_ai.Agent.run` and `run_sync` to raise, execute `assemble_package` against fixtures — must succeed (zero LLM calls)
- [x] **Grounding density test**: rigged draft with citation ratio below `grounding_min_ratio` fails assembly
- [x] **Ledger dedupe test**: DOI / arXiv / canonical URL duplicates collapse to one entry
- [x] **run.py size threshold test**: `run.py` stays below configured line count, no embedded reusable helpers
- [x] Run all guard tests
- [x] Commit: `test(v2): add fast guard tests for architectural invariants`

### Task 8.2: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

- [x] One-iteration happy path
- [x] Multi-iteration research loop
- [x] Reviewer-triggered supplemental loop (happy path + second-veto-ignored)
- [x] Per-subagent replay
- [x] Finalize-only replay
- [x] Assemble-only replay
- [x] Council comparison generation + operator-selection wait
- [x] Degraded subagent failure path
- [x] Finalizer failure with and without `allow_unfinalized_package`
- [x] Run all integration tests
- [x] Commit: `test(v2): add integration test suite`

---

## Phase 9: Evals Migration

> Update evals/ to test V2 through the flow entrypoint and `InvestigationPackage` only. Evals are a sibling package, not part of runtime.

### Task 9.1: Eval Harness for V2

**Files:**
- Modify: `evals/runner.py` (or create V2-specific runner)
- Modify: `evals/suites/` as needed

- [ ] Update eval runner to drive V2 flow end-to-end
- [ ] Assert on `InvestigationPackage` shape and content
- [ ] Evals must not import V2 runtime internals — test through flow entrypoint only
- [ ] Verify existing eval suites work against V2 package format
- [ ] Run eval suite
- [ ] Commit: `feat(v2): migrate eval harness to V2 flow and package`

---

## Phase 10: Observability & Polish

> Wire up Logfire, finalize prompt engineering, clean up.

### Task 10.1: Logfire Instrumentation

**Files:**
- Create: `research/observability.py`

- [ ] Instrument PydanticAI agents, httpx, and MCP transports with Logfire
- [ ] `configure()` with `send_to_logfire="if-token-present"` (zero-config locally)
- [ ] Checkpoint spans for `run_supervisor`, `score_coverage`, etc.
- [ ] Per-iteration metrics: `iteration_coverage`, `iteration_cost_usd`, `iteration_candidates`
- [ ] Content inclusion gated by `DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT` env var (default: false)
- [ ] Commit: `feat(v2): add Logfire observability instrumentation`

### Task 10.2: Prompt Engineering Pass

**Files:**
- Modify: all `research/prompts/*.md` files

- [ ] Replace placeholder prompts with production-quality system prompts
- [ ] Supervisor prompt is the highest priority — it replaces V1's convergence engine, source priors, novelty logic, and replan heuristics
- [ ] Generator and finalizer prompts must enforce inline `[evidence_id]` citation discipline
- [ ] Reviewer prompt must implement the three-dimension rubric (source reliability, completeness, grounding)
- [ ] All prompts distinguish policy from framing so reviewers can tell whether a change alters behavior or only phrasing
- [ ] Commit: `feat(v2): write production system prompts`

---

## Summary: Phase Dependencies

```
Phase 1 (Foundation)
  └─> Phase 2 (Evidence & Providers)
        └─> Phase 3 (Agents)
              └─> Phase 4 (Checkpoints)
                    └─> Phase 5 (Default Flow)
                          ├─> Phase 6 (Package & run.py)
                          ├─> Phase 7 (Council Flow)
                          └─> Phase 8 (Guard & Integration Tests)
                                └─> Phase 9 (Evals)
                                      └─> Phase 10 (Observability & Polish)
```

## Deferred (Not In This Plan)

Per the design spec, these are explicitly deferred from the first slice:

- [ ] Dedicated `cli/` package
- [ ] Outward MCP control plane (`mcp/`)
- [ ] Memory subsystem
- [ ] Virtual filesystem abstraction
- [ ] V1 compatibility bridge
- [ ] Package-schema migration framework (added only when first non-additive change lands)
- [ ] Broad operator UX beyond `run.py`
