# Deep Research Engine Architecture

This document describes the architecture of the deep-research engine -- a Kitaru-based durable workflow that iteratively gathers, scores, and curates evidence to answer a research brief.

---

## 1. High-Level Research Flow

The research flow proceeds through four logical phases: classification of the incoming brief, planning, an iterative search-and-evaluate loop, and final assembly of the investigation package. Two optional wait points allow human-in-the-loop control: one for clarifying ambiguous briefs and one for approving the generated plan.

```mermaid
flowchart TD
    classDef wait fill:#f9e79f,stroke:#d4ac0d,color:#000
    classDef checkpoint fill:#aed6f1,stroke:#2e86c1,color:#000
    classDef decision fill:#fadbd8,stroke:#e74c3c,color:#000
    classDef terminal fill:#a9dfbf,stroke:#27ae60,color:#000

    Brief([brief + tier])

    subgraph Classification
        CL[classify_request]:::checkpoint
        NeedsClarification{needs clarification?}:::decision
        ClarifyWait[/wait: clarify_brief/]:::wait
        ReClassify[classify_request]:::checkpoint
    end

    subgraph Planning
        BP[build_plan]:::checkpoint
        NeedsApproval{require_plan_approval?}:::decision
        ApproveWait[/wait: approve_plan/]:::wait
        Rejected{approved?}:::decision
    end

    subgraph Iteration["Iteration Loop (max N)"]
        direction TB
        SUP[run_supervisor / council]:::checkpoint
        NORM[normalize_evidence]:::checkpoint
        REL[score_relevance]:::checkpoint
        MRG[merge_evidence]:::checkpoint
        EVAL[evaluate_coverage]:::checkpoint
        CONV{convergence check}:::decision
    end

    subgraph Assembly
        SEL[build_selection_graph]:::checkpoint
        RRP[render_reading_path]:::checkpoint
        RBR[render_backing_report]:::checkpoint
        ASM[assemble_package]:::checkpoint
    end

    PKG([InvestigationPackage]):::terminal

    Brief --> CL
    CL --> NeedsClarification
    NeedsClarification -- yes --> ClarifyWait --> ReClassify --> BP
    NeedsClarification -- no --> BP
    BP --> NeedsApproval
    NeedsApproval -- yes --> ApproveWait --> Rejected
    Rejected -- no --> Error([ValueError])
    Rejected -- yes --> SUP
    NeedsApproval -- no --> SUP

    SUP --> NORM --> REL --> MRG --> EVAL --> CONV
    CONV -- not converged --> SUP
    CONV -- converged / budget / time --> SEL

    SEL --> RRP
    SEL --> RBR
    RRP --> ASM
    RBR --> ASM
    ASM --> PKG
```

---

## 2. Checkpoint Dependency Graph

Each checkpoint is a Kitaru-decorated function whose outputs feed into subsequent checkpoints. The main path is sequential, but the council fan-out runs N generators concurrently, and the two renderers execute in parallel after selection.

```mermaid
flowchart LR
    classDef llm fill:#aed6f1,stroke:#2e86c1,color:#000
    classDef tool fill:#d5f5e3,stroke:#27ae60,color:#000
    classDef fan fill:#f5cba7,stroke:#e67e22,color:#000

    classify[classify_request<br/>LLM]:::llm
    plan[build_plan<br/>LLM]:::llm
    sup[run_supervisor<br/>LLM]:::llm
    cg0[council_generator 0<br/>LLM]:::fan
    cg1[council_generator 1<br/>LLM]:::fan
    cgN[council_generator N<br/>LLM]:::fan
    agg[aggregate_council_results]:::tool
    norm[normalize_evidence<br/>tool]:::tool
    rel[score_relevance<br/>LLM]:::llm
    mrg[merge_evidence<br/>tool]:::tool
    eval[evaluate_coverage<br/>tool]:::tool
    sel[build_selection_graph<br/>LLM]:::llm
    rp[render_reading_path<br/>LLM]:::llm
    rb[render_backing_report<br/>LLM]:::llm
    asm[assemble_package<br/>tool]:::tool

    classify -- RequestClassification --> plan
    plan -- ResearchPlan --> sup
    plan -- ResearchPlan --> cg0
    plan -- ResearchPlan --> cg1
    plan -- ResearchPlan --> cgN

    sup -- SupervisorCheckpointResult --> norm
    cg0 -- SupervisorCheckpointResult --> agg
    cg1 -- SupervisorCheckpointResult --> agg
    cgN -- SupervisorCheckpointResult --> agg
    agg -- SupervisorCheckpointResult --> norm

    norm -- "list[EvidenceCandidate]" --> rel
    rel -- RelevanceCheckpointResult --> mrg
    mrg -- EvidenceLedger --> eval
    eval -- CoverageScore --> sel

    sel -- SelectionGraph --> rp
    sel -- SelectionGraph --> rb
    rp -- RenderPayload --> asm
    rb -- RenderPayload --> asm
```

---

## 3. Iteration Loop Detail

Each iteration cycle follows a strict sequence: the supervisor (or council) searches for evidence, results are normalized into candidates, scored for relevance, merged into the running ledger, and coverage is evaluated. The convergence check then decides whether to continue or stop.

```mermaid
sequenceDiagram
    participant Flow as research_flow
    participant Sup as Supervisor / Council
    participant Norm as normalize_evidence
    participant Rel as score_relevance
    participant Mrg as merge_evidence
    participant Eval as evaluate_coverage
    participant Conv as check_convergence

    Flow->>Sup: plan, ledger, iteration, config
    Sup-->>Flow: SupervisorCheckpointResult (raw_results + budget)

    Flow->>Norm: raw_results
    Norm-->>Flow: list[EvidenceCandidate]

    Flow->>Rel: candidates, plan, config
    Rel-->>Flow: RelevanceCheckpointResult (scored candidates + budget)

    Flow->>Mrg: scored candidates, ledger
    Mrg-->>Flow: EvidenceLedger (deduplicated, merged)

    Flow->>Eval: ledger, plan
    Eval-->>Flow: CoverageScore (subtopic, diversity, density, total)

    Flow->>Conv: coverage, history, budget, elapsed, limits
    Conv-->>Flow: StopDecision

    alt should_stop = false
        Flow->>Sup: next iteration
    else should_stop = true
        Flow->>Flow: break to assembly phase
    end
```

---

## 4. Module Structure

The codebase is organized into focused packages. The `flow/` package orchestrates the research loop and convergence logic. `checkpoints/` contains the Kitaru-decorated checkpoint functions. Supporting concerns are separated into `evidence/`, `providers/`, `tools/`, `renderers/`, and `package/`.

```mermaid
flowchart TB
    classDef pkg fill:#d6eaf8,stroke:#2e86c1,color:#000
    classDef core fill:#fdebd0,stroke:#e67e22,color:#000

    subgraph deep_research
        direction TB

        models[models.py<br/>Data models]:::core
        config[config.py<br/>Tiers + settings]:::core
        enums[enums.py<br/>StopReason, Tier, SourceKind]:::core

        subgraph "flow/"
            rf[research_flow.py<br/>Main orchestration]:::pkg
            conv[convergence.py<br/>Stop decision logic]:::pkg
            cost[costing.py<br/>Token cost estimation]:::pkg
        end

        subgraph "checkpoints/"
            ck_classify[classify]:::pkg
            ck_plan[plan]:::pkg
            ck_sup[supervisor]:::pkg
            ck_council[council]:::pkg
            ck_norm[normalize]:::pkg
            ck_rel[relevance]:::pkg
            ck_merge[merge]:::pkg
            ck_eval[evaluate]:::pkg
            ck_sel[select]:::pkg
            ck_asm[assemble]:::pkg
        end

        subgraph "evidence/"
            ev_dedup[dedup.py]:::pkg
            ev_ledger[ledger.py]:::pkg
            ev_score[scoring.py]:::pkg
        end

        subgraph "providers/"
            pr_norm[normalization.py]:::pkg
            pr_mcp[mcp_config.py]:::pkg
        end

        subgraph "tools/"
            tl_bash[bash_executor.py]:::pkg
            tl_state[state_reader.py]:::pkg
        end

        subgraph "renderers/"
            rn_read[reading_path.py]:::pkg
            rn_back[backing_report.py]:::pkg
        end

        subgraph "package/"
            pk_asm[assembly.py]:::pkg
            pk_io[io.py]:::pkg
        end
    end

    rf --> ck_classify & ck_plan & ck_sup & ck_council
    rf --> ck_norm & ck_rel & ck_merge & ck_eval
    rf --> ck_sel & rn_read & rn_back & ck_asm
    rf --> conv & cost
    ck_merge --> ev_ledger --> ev_dedup
    ck_norm --> pr_norm
    ck_asm --> pk_asm
    ck_sup --> models
    ck_rel --> models
    ev_score --> enums
    rf --> config
```

---

## 5. Council Mode vs Single Supervisor

In single-supervisor mode, one LLM call searches for evidence per iteration. In council mode, N generators (each potentially using different models) run concurrently, and their results are aggregated before normalization. Council mode trades higher cost for broader evidence coverage.

```mermaid
flowchart LR
    classDef single fill:#aed6f1,stroke:#2e86c1,color:#000
    classDef council fill:#f5cba7,stroke:#e67e22,color:#000
    classDef shared fill:#d5f5e3,stroke:#27ae60,color:#000

    subgraph "Single Supervisor Path"
        direction TB
        S_IN([plan + ledger + iteration]) --> S_SUP[run_supervisor<br/>1 LLM call]:::single
        S_SUP --> S_OUT[SupervisorCheckpointResult]:::single
    end

    subgraph "Council Mode Path"
        direction TB
        C_IN([plan + ledger + iteration]) --> C_FAN{fan out to N models}
        C_FAN --> C_G0[council_generator 0]:::council
        C_FAN --> C_G1[council_generator 1]:::council
        C_FAN --> C_GN[council_generator N]:::council
        C_G0 --> C_AGG[aggregate_council_results<br/>merge raw_results + budgets]:::council
        C_G1 --> C_AGG
        C_GN --> C_AGG
        C_AGG --> C_OUT[SupervisorCheckpointResult]:::council
    end

    S_OUT --> NORM[normalize_evidence]:::shared
    C_OUT --> NORM
    NORM --> REST[score_relevance -> merge -> evaluate -> ...]:::shared
```

---

## 6. Data Model Relationships

The core data models form a hierarchy rooted in `InvestigationPackage`, which is the final output. The iteration loop populates the `EvidenceLedger` with `EvidenceCandidate` entries, while `SelectionGraph` curates the final set. `RunSummary` and `IterationTrace` capture execution metadata.

```mermaid
classDiagram
    class InvestigationPackage {
        +RunSummary run_summary
        +ResearchPlan research_plan
        +EvidenceLedger evidence_ledger
        +SelectionGraph selection_graph
        +IterationTrace iteration_trace
        +list~RenderPayload~ renders
    }

    class RunSummary {
        +str run_id
        +str brief
        +Tier tier
        +StopReason stop_reason
        +str status
    }

    class ResearchPlan {
        +str goal
        +list~str~ key_questions
        +list~str~ subtopics
        +list~str~ queries
        +list~str~ sections
        +list~str~ success_criteria
    }

    class EvidenceLedger {
        +list~EvidenceCandidate~ entries
    }

    class EvidenceCandidate {
        +str key
        +str title
        +AnyUrl url
        +list~EvidenceSnippet~ snippets
        +str provider
        +SourceKind source_kind
        +float quality_score
        +float relevance_score
        +bool selected
    }

    class EvidenceSnippet {
        +str text
        +str|None source_locator
    }

    class SelectionGraph {
        +list~SelectionItem~ items
    }

    class SelectionItem {
        +str candidate_key
        +str rationale
    }

    class IterationTrace {
        +list~IterationRecord~ iterations
    }

    class IterationRecord {
        +int iteration
        +int new_candidate_count
        +float coverage
    }

    class CoverageScore {
        +float subtopic_coverage
        +float source_diversity
        +float evidence_density
        +float total
    }

    class SupervisorCheckpointResult {
        +list~RawToolResult~ raw_results
        +IterationBudget budget
    }

    class RawToolResult {
        +str tool_name
        +str provider
        +dict payload
        +bool ok
        +str|None error
    }

    class IterationBudget {
        +int input_tokens
        +int output_tokens
        +int total_tokens
        +float estimated_cost_usd
    }

    class RelevanceCheckpointResult {
        +list~EvidenceCandidate~ candidates
        +IterationBudget budget
    }

    class RequestClassification {
        +str audience_mode
        +str freshness_mode
        +Tier recommended_tier
        +bool needs_clarification
        +str|None clarification_question
    }

    class RenderPayload {
        +str name
        +str content_markdown
        +dict~str,str~ citation_map
    }

    InvestigationPackage *-- RunSummary
    InvestigationPackage *-- ResearchPlan
    InvestigationPackage *-- EvidenceLedger
    InvestigationPackage *-- SelectionGraph
    InvestigationPackage *-- IterationTrace
    InvestigationPackage *-- RenderPayload

    EvidenceLedger *-- EvidenceCandidate
    EvidenceCandidate *-- EvidenceSnippet
    SelectionGraph *-- SelectionItem
    IterationTrace *-- IterationRecord
    SupervisorCheckpointResult *-- RawToolResult
    SupervisorCheckpointResult *-- IterationBudget
    RelevanceCheckpointResult *-- EvidenceCandidate
    RelevanceCheckpointResult *-- IterationBudget

    SelectionItem ..> EvidenceCandidate : references by key
```

---

## Convergence Stop Reasons

The convergence check in `flow/convergence.py` evaluates multiple stopping conditions in priority order:

| StopReason | Trigger |
|---|---|
| `BUDGET_EXHAUSTED` | Cumulative cost reaches `cost_budget_usd` |
| `TIME_EXHAUSTED` | Wall-clock time reaches `time_box_seconds` |
| `CONVERGED` | Coverage total meets `convergence_min_coverage` |
| `LOOP_STALL` | Coverage gain is zero or negative |
| `DIMINISHING_RETURNS` | Coverage gain falls below `convergence_epsilon` |
| `MAX_ITERATIONS` | Iteration count reaches `max_iterations` |

## Tier Configuration

| Tier | Max Iterations | Budget (USD) | Time Box (s) | Critique | Judge | Council |
|---|---|---|---|---|---|---|
| `quick` | 2 | 0.05 | 120 | no | no | no |
| `standard` | 3 | 0.10 | 600 | no | no | no |
| `deep` | 6 | 1.00 | 1800 | yes | yes | yes |
| `custom` | 3 | 0.10 | 600 | no | no | yes |
