# Deep Research Core Expansion Design

**Date:** 2026-04-06
**Status:** Design - pending user review

---

## Scope

This design covers the next expansion slice for the standalone deep research engine, focused only on the currently partial areas that matter inside the flow-first library:

- richer core research loop and harness
- richer canonical `InvestigationPackage`
- richer provider and evidence plane
- richer durability and inspectability

This design explicitly does **not** cover:

- HTTP API
- CLI
- MCP as a public interface
- host integration surfaces
- critique / judge layer for this phase

The engine remains a Kitaru flow-first Python library. The entrypoint continues to be the flow, invoked directly from Python.

---

## Design Constraints

The approved design constraints for this phase are:

- Keep the existing project structure.
- Keep the implementation direct and readable.
- Avoid introducing a new orchestration/runtime layer.
- Prefer small amounts of explicit code over many helper abstractions.
- Use PydanticAI-native tools, capabilities, toolsets, and MCP integrations where they fit.
- Broaden the provider matrix in this phase, not just one provider per source class.
- Add lazy `full_report` now as part of package/render completeness.

---

## Core Architecture

The repository keeps its current structure and grows in place.

### Orchestration

`deep_research/flow/research_flow.py` remains the only top-level orchestration flow.

The flow body continues to read like the product behavior directly:

1. classify the request
2. build the plan
3. wait for clarification or plan approval when needed
4. run research iterations
5. normalize and merge evidence
6. evaluate convergence and stop conditions
7. build the selection graph
8. render eager outputs
9. assemble the canonical package

There is no new `runtime/`, `engine/`, or `manager/` layer.

### Checkpoints

`deep_research/checkpoints/` remains the durable phase boundary layer.

Each checkpoint performs concrete work for one phase instead of acting as a generic dispatcher. The checkpoint layer stays shallow and explicit.

### Agents

`deep_research/agents/` remains a collection of thin PydanticAI agent builders.

Agent builders are responsible for:

- model selection
- PydanticAI output type
- prompt loading
- direct tool / toolset / capability wiring

Agent builders are not responsible for flow decisions, convergence, or artifact policy.

### Provider Plane

`deep_research/providers/` becomes the place for direct provider execution wiring.

This phase uses:

- PydanticAI-native tools where appropriate
- PydanticAI capabilities such as web search where appropriate
- MCP-backed toolsets for providers that are better represented as MCP servers

The provider plane should not become a large plugin framework.

### Evidence Plane

`deep_research/evidence/` owns:

- normalization support
- heuristic scoring
- deduplication
- ratchet merge behavior
- novelty and provenance inputs

### Package Plane

`deep_research/package/` continues to own:

- canonical package assembly
- package materialization to markdown/json artifacts

### Renderers

`deep_research/renderers/` owns three renderers:

- `reading_path` (eager)
- `backing_report` (eager)
- `full_report` (lazy)

---

## Richer Research Loop

The loop remains a single Kitaru flow, but the iteration body becomes much closer to the unified spec.

### Revised Loop Shape

1. `classify_request`
2. `build_plan`
3. `wait("clarify_brief")` when needed
4. `wait("approve_plan")` when needed
5. iterate until convergence or hard stop:
   - run supervisor with a real provider/tool surface
   - normalize tool outputs into evidence candidates
   - score heuristic quality fields
   - dedupe and ratchet-merge into the ledger
   - compute coverage and uncovered subtopics
   - append iteration record
   - decide continue / stop
6. build selection graph from the accumulated ledger
7. render `reading_path`
8. render `backing_report`
9. assemble package

`full_report` is intentionally not rendered in the main flow. It is generated lazily from the canonical package.

### Supervisor Execution

The supervisor is no longer an empty harness.

The supervisor checkpoint gets a real execution surface composed from configured provider tools and toolsets. It receives current loop state directly:

- plan
- current ledger
- current uncovered subtopics or gap state
- iteration index
- configured call limit
- configured tool timeout

The checkpoint returns structured tool results and iteration cost data.

### Convergence

Convergence remains deterministic and explicit.

This phase improves it by giving it better inputs, not by replacing it with LLM judgment.

The convergence decision should consider:

- total coverage
- change in coverage from the previous iteration
- new candidate count
- elapsed time
- spent cost
- loop stall conditions
- hard iteration cap

Uncovered subtopics should be tracked directly so later iterations are gap-driven instead of only score-driven.

### What We Will Not Add In This Phase

- no second orchestration loop
- no critique/reviewer phase
- no grounding/coherence judges
- no planner re-run every iteration
- no generic runtime abstraction

---

## Richer Package Design

The canonical output remains `InvestigationPackage`, but the package model expands much closer to the unified spec.

### Run Summary

`RunSummary` should grow to include the operational facts already computed by the flow:

- run id
- brief
- tier
- stop reason
- status
- estimated cost
- elapsed seconds
- iteration count
- provider usage summary
- council mode metadata
- timestamps where available

### Research Plan

`ResearchPlan` should expand from the current minimal shape into a richer planning contract:

- goal
- key questions
- subtopics
- grouped queries or query categories
- intended report sections
- source strategy or allowed source groups
- success criteria
- approval status

### Evidence Ledger

The ledger should stop being a single flat list.

It should represent:

- `considered`
- `selected`
- `rejected`
- `dedupe_log`

This lets the engine explain not just what it kept, but what it saw and why it excluded items.

### Selection Graph

The selection graph should include:

- ordered items
- item rationale
- optional bridge note
- matched subtopics
- reading time when available
- ordering rationale
- gap coverage summary

### Iteration Trace

`IterationTrace` should become the main durable summary of loop behavior.

Each `IterationRecord` should include:

- iteration index
- raw result count
- accepted candidate count
- rejected candidate count
- coverage before / after or at least total plus delta
- estimated cost
- stop reason when applicable
- compact tool call summary

### Render Payloads

`RenderPayload` should be strengthened so rendered outputs are explicit package artifacts, not just markdown blobs.

Each render payload should carry:

- stable renderer key
- markdown content
- optional structured content payload
- citation map
- generation timestamp

### Full Package

`InvestigationPackage` remains canonical and contains:

- `run_summary`
- `research_plan`
- `evidence_ledger`
- `selection_graph`
- `iteration_trace`
- render payloads for any generated renders

---

## Render Lifecycle

### Eager Renders

Generated during the main flow:

- `reading_path`
- `backing_report`

### Lazy Render

Generated outside the main flow on demand:

- `full_report`

The lazy render uses the canonical `InvestigationPackage` as input and should not recompute the research loop.

### Renderer Output Expectations

`reading_path` should become an ordered, user-readable path over selected evidence.

`backing_report` should explain:

- why items were selected
- what gaps were covered
- what was rejected or de-prioritized
- the overall support for the path

`full_report` should be the traditional cited synthesis built from the package state.

---

## Package I/O

`deep_research/package/io.py` becomes the single place that materializes the package to disk.

The output layout for this phase should include:

- `package.json`
- `summary.md`
- `plan.json`
- `plan.md`
- `evidence/ledger.json`
- `evidence/ledger.md`
- `iterations/*.json`
- `renders/reading_path.md`
- `renders/backing_report.md`
- `renders/full_report.md` when it exists

`package.json` remains canonical. Markdown files are derived views over the canonical package.

`assemble_package()` remains a thin constructor. It must not absorb file-writing or formatting logic.

---

## Richer Provider Plane

This phase includes a broader provider matrix, but still through a simple direct design.

### Provider Execution Surface

We prefer PydanticAI-native execution surfaces first:

- direct tools
- toolsets
- `WebSearch()` capability where it fits
- MCP-backed toolsets where that fits better

This gives us broader real provider coverage without building a custom provider framework first.

### Provider Scope For This Phase

This phase is designed for multiple web and academic providers, not a single happy path.

The actual matrix should be configured through direct provider/tool wiring in `providers/` and consumed by the supervisor checkpoint.

### Provider Modules

`providers/` should grow through direct modules for:

- provider/toolset configuration
- normalization helpers for distinct raw result shapes
- model/provider selection helpers only when truly needed

What it should not become:

- a large abstract provider hierarchy
- a plugin registry
- a framework of strategy objects

---

## Richer Evidence Plane

The evidence plane becomes the stable boundary between heterogeneous provider outputs and the canonical package.

### Evidence Candidate

`EvidenceCandidate` should be enriched so the engine can support stronger dedup, scoring, and provenance:

- stable key
- title
- URL when present
- provider
- source kind
- snippets
- matched subtopics
- quality score
- relevance score
- authority score
- freshness score when available
- identifiers such as DOI or arXiv ID when available
- provider/raw metadata needed for provenance

### Deduplication

Dedup precedence should become:

1. DOI
2. arXiv ID
3. canonical URL
4. title fallback

### Ratchet Merge

When the same logical evidence item appears again:

- keep one canonical candidate
- ratchet useful scores upward
- merge snippets
- merge metadata
- record the dedupe event

### Selection Policy

The evidence ledger should preserve low-quality and rejected items for inspection, but the selection graph should obey the configured quality floor.

That means:

- normalization persists candidates
- scoring computes quality fields
- merge preserves them in the ledger
- selection excludes or down-ranks below-floor items

This keeps the system auditable without pretending weak evidence is equally useful.

---

## Richer Durability

This phase stays inside the flow-first library boundary, but makes durability more explicit.

### Stable Artifact Boundaries

Durable artifacts should be saved at these major points:

- classification
- plan
- per-iteration supervisor output
- per-iteration normalized candidates
- per-iteration ledger snapshot
- per-iteration coverage snapshot
- selection graph
- eager renders
- final package

### Stable Names

Checkpoint and wait names should remain exported as explicit constants from `research_flow.py` or the relevant modules so replay anchors stay stable and testable.

Council submit IDs should become iteration-qualified, for example including both iteration and council index.

### Replay Contract

Replay continues to work from stable named checkpoints. This phase improves replay confidence by making more state durable and more names explicit.

### What We Will Not Add In This Phase

- no HTTP control plane
- no library-side execution manager
- no artifact framework abstraction
- no custom replay wrapper over Kitaru

---

## File-Level Change Plan

The implementation should stay concentrated in the existing structure.

### Primary files to expand

- `deep_research/models.py`
- `deep_research/config.py`
- `deep_research/flow/research_flow.py`
- `deep_research/flow/convergence.py`
- `deep_research/checkpoints/supervisor.py`
- `deep_research/checkpoints/normalize.py`
- `deep_research/checkpoints/merge.py`
- `deep_research/checkpoints/evaluate.py`
- `deep_research/checkpoints/select.py`
- `deep_research/checkpoints/assemble.py`
- `deep_research/renderers/reading_path.py`
- `deep_research/renderers/backing_report.py`
- `deep_research/renderers/full_report.py`
- `deep_research/package/assembly.py`
- `deep_research/package/io.py`
- `deep_research/providers/*`
- `deep_research/evidence/*`

### Test files to expand

- `tests/test_research_flow_unit.py`
- `tests/test_research_flow_integration.py`
- `tests/test_operator_contract.py`
- `tests/test_package_io.py`
- `tests/test_renderers.py`
- provider/evidence-specific tests

---

## Implementation Principles

The implementation of this phase should follow these rules:

- prefer direct code in the flow over extra orchestration helpers
- keep agent builders thin
- keep checkpoints concrete
- avoid creating new abstraction layers unless a real duplication problem appears
- prefer expanding existing models over introducing wrapper models everywhere
- keep the package canonical and all markdown derived from it

---

## Out Of Scope For This Design Slice

These remain intentionally deferred even if they appear in the broader unified specification:

- HTTP API
- CLI
- MCP as a public external interface
- host adapters
- critique layer
- grounding/coherence judges
- cross-provider review enforcement for package outputs

They may use the richer package and durability foundations from this phase later, but they are not part of this implementation slice.
