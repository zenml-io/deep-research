# Deep Research Engine MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python deep research engine library that produces a durable `InvestigationPackage` via Kitaru orchestration, PydanticAI agent checkpoints, MCP and bash-backed evidence gathering, and MVP council mode.

**Architecture:** The MVP is one Kitaru `@flow` that owns iteration, waits, convergence, council fan-out, replay anchors, and package assembly. Expensive agent turns and artifact-producing transforms live in explicit `@checkpoint` functions, while council orchestration stays in flow scope to avoid nested checkpoint execution. Budget enforcement is implemented explicitly from tracked usage plus a local pricing table rather than assuming the adapter computes USD cost automatically.

**Tech Stack:** Python, uv, Kitaru, Pydantic, Pydantic Settings, PydanticAI, pytest, MCP-backed search tools

---

## File Structure

- `pyproject.toml`: project metadata, dependencies, pytest config.
- `.gitignore`: Python ignores plus `.superpowers/` and run artifacts.
- `deep_research/__init__.py`: public exports for main models and flow entrypoints.
- `deep_research/enums.py`: `StopReason`, `RunStatus`, `Tier`, `AudienceMode`, `FreshnessMode`, `SourceKind`.
- `deep_research/models.py`: Pydantic models for package layers, evidence, iteration records, render payloads, decisions, and runtime accounting.
- `deep_research/config.py`: `ResearchSettings`, `ResearchConfig`, tier defaults, model-role settings, provider configuration, pricing table.
- `deep_research/prompts/*.md`: system prompts for classifier, planner, supervisor, writer, curator, aggregator, reviewer, judges, scorer, question generator.
- `deep_research/prompts/loader.py`: loads prompt markdown by name.
- `deep_research/providers/mcp_config.py`: construct MCP toolsets from configured servers.
- `deep_research/providers/normalization.py`: normalize raw MCP or bash results into common evidence candidates.
- `deep_research/evidence/dedup.py`: dedupe candidate identity.
- `deep_research/evidence/scoring.py`: heuristic quality scoring.
- `deep_research/evidence/ledger.py`: immutable-ish ledger operations and coverage helpers.
- `deep_research/tools/state_reader.py`: PydanticAI function tools for reading plan, ledger, gaps, and iteration summaries.
- `deep_research/tools/bash_executor.py`: guarded subprocess execution with timeout, temp cwd, env scrub, and command policy.
- `deep_research/agents/*.py`: wrapped PydanticAI agents for classifier, planner, supervisor, relevance scorer, curator, writer, aggregator, reviewer, judges.
- `deep_research/checkpoints/*.py`: explicit Kitaru checkpoints for agent turns and artifact-producing transforms.
- `deep_research/flow/convergence.py`: stop-decision rules and coverage deltas.
- `deep_research/flow/costing.py`: usage aggregation, pricing lookup, per-iteration and per-run budget checks.
- `deep_research/flow/research_flow.py`: MVP orchestrator, including council mode fan-out in flow scope.
- `deep_research/package/assembly.py`: canonical `InvestigationPackage` creation.
- `deep_research/package/io.py`: write and read markdown and JSON package artifacts.
- `tests/...`: unit and integration coverage by module.

## MVP Scope

This MVP intentionally includes council mode, but only one durable flow. It includes:

- request classification
- plan generation
- plan approval wait
- iterative supervisor search loop
- council fan-out from flow scope via `.submit()`
- normalization, dedupe, heuristic scoring, LLM relevance scoring, ledger merge, coverage evaluation
- selection graph
- eager `reading_path.md` and `backing_report.md`
- canonical `InvestigationPackage`
- explicit token and USD accounting with hard budget enforcement in the iterative search loop
- operator workflow via SDK launch plus Kitaru CLI or `KitaruClient` for waits, replay, and inspection

This MVP defers:

- FastAPI or library-hosting API
- custom project CLI
- host adapters
- lazy `full_report.md`
- cross-provider critique and judges
- daily cost-limit persistence beyond the current run unless a real persistence target is chosen

## Operator Surface

- **Launch / deploy:** SDK flow object via `research_flow.run(...)`
- **Logs / inspection:** `KitaruClient` or Kitaru CLI
- **Wait input:** `KitaruClient.executions.input(...)` or `kitaru executions input`
- **Wait abort:** `KitaruClient.executions.abort_wait(...)`
- **Resume:** `KitaruClient.executions.resume(...)` or `kitaru executions resume`
- **Replay / cancel:** `KitaruClient` or Kitaru CLI
- **Artifact inspection:** `KitaruClient.artifacts.*`
- **Stack management:** Kitaru CLI or MCP, not this library

## Stable Replay Anchors And Wait Names

- Checkpoints to keep stable from the first commit:
  - `classify_request`
  - `build_plan`
  - `run_supervisor`
  - `run_council_generator`
  - `normalize_evidence`
  - `score_relevance`
  - `merge_evidence`
  - `evaluate_coverage`
  - `build_selection_graph`
  - `render_reading_path`
  - `render_backing_report`
  - `assemble_package`
- Wait names to keep stable from the first commit:
  - `approve_plan`
  - `clarify_brief`

## Important Design Corrections

- Council orchestration must stay in flow scope. Only generator work is checkpointed.
- `run_bash` is a guarded executor, not a true sandbox.
- Budget checks use explicit pricing code in `deep_research/flow/costing.py`.
- Provider failures are normalized into partial iteration records and never abort the whole run.
- `question_generator` is kept only if it is wired to a real `clarify_brief` flow-level wait; otherwise cut it from role settings.

---

### Task 1: Bootstrap Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `deep_research/__init__.py`
- Create: `deep_research/flow/__init__.py`
- Create: `deep_research/checkpoints/__init__.py`
- Create: `deep_research/agents/__init__.py`
- Create: `deep_research/providers/__init__.py`
- Create: `deep_research/evidence/__init__.py`
- Create: `deep_research/renderers/__init__.py`
- Create: `deep_research/critique/__init__.py`
- Create: `deep_research/package/__init__.py`
- Create: `deep_research/tools/__init__.py`
- Create: `deep_research/prompts/.gitkeep`
- Create: `tests/__init__.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write the failing import smoke test**

```python
from importlib import import_module


def test_package_modules_import() -> None:
    modules = [
        "deep_research",
        "deep_research.flow",
        "deep_research.checkpoints",
        "deep_research.agents",
        "deep_research.providers",
        "deep_research.evidence",
        "deep_research.renderers",
        "deep_research.critique",
        "deep_research.package",
        "deep_research.tools",
    ]
    for module in modules:
        import_module(module)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError` for `deep_research`

- [ ] **Step 3: Add minimal project files and packages**

```toml
[project]
name = "deep-research"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "kitaru",
  "pydantic>=2",
  "pydantic-ai",
  "pydantic-settings",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# deep_research/__init__.py
"""Deep research engine package."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_imports.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore deep_research tests/test_imports.py
git commit -m "chore: scaffold deep research package"
```

### Task 2: Define Enums And Core Models

**Files:**
- Create: `deep_research/enums.py`
- Create: `deep_research/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing model round-trip tests**

```python
from deep_research.enums import StopReason, Tier
from deep_research.models import EvidenceCandidate, InvestigationPackage, ResearchPlan


def test_research_plan_round_trip() -> None:
    plan = ResearchPlan(
        goal="Understand Kitaru",
        key_questions=["What is replay?"],
        subtopics=["replay"],
        queries=["kitaru replay checkpoints"],
        sections=["Overview"],
        success_criteria=["Explain replay anchors"],
    )
    restored = ResearchPlan.model_validate(plan.model_dump())
    assert restored == plan


def test_package_minimum_shape() -> None:
    package = InvestigationPackage(
        run_summary={
            "run_id": "run-1",
            "brief": "test",
            "tier": Tier.STANDARD,
            "stop_reason": StopReason.MAX_ITERATIONS,
            "status": "completed",
        },
        research_plan={
            "goal": "x",
            "key_questions": [],
            "subtopics": [],
            "queries": [],
            "sections": [],
            "success_criteria": [],
        },
        evidence_ledger={"entries": []},
        selection_graph={"items": []},
        iteration_trace={"iterations": []},
        renders=[],
    )
    assert package.run_summary.run_id == "run-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with import or validation errors because the models are missing

- [ ] **Step 3: Implement enums and minimal package models**

```python
# deep_research/enums.py
from enum import Enum


class StopReason(str, Enum):
    CONVERGED = "converged"
    DIMINISHING_RETURNS = "diminishing_returns"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    MAX_ITERATIONS = "max_iterations"
    LOOP_STALL = "loop_stall"
    CANCELLED = "cancelled"


class Tier(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    CUSTOM = "custom"
```

```python
# deep_research/models.py
from pydantic import BaseModel, Field

from deep_research.enums import StopReason, Tier


class ResearchPlan(BaseModel):
    goal: str
    key_questions: list[str]
    subtopics: list[str]
    queries: list[str]
    sections: list[str]
    success_criteria: list[str]


class EvidenceSnippet(BaseModel):
    text: str
    source_locator: str | None = None


class EvidenceCandidate(BaseModel):
    key: str
    title: str
    url: str
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    provider: str
    source_kind: str
    quality_score: float = 0.0
    relevance_score: float = 0.0
    selected: bool = False


class EvidenceLedger(BaseModel):
    entries: list[EvidenceCandidate] = Field(default_factory=list)


class SelectionItem(BaseModel):
    candidate_key: str
    rationale: str


class SelectionGraph(BaseModel):
    items: list[SelectionItem] = Field(default_factory=list)


class IterationRecord(BaseModel):
    iteration: int
    new_candidate_count: int = 0
    coverage: float = 0.0


class IterationTrace(BaseModel):
    iterations: list[IterationRecord] = Field(default_factory=list)


class RenderPayload(BaseModel):
    name: str
    content_markdown: str
    citation_map: dict[str, str] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    brief: str
    tier: Tier
    stop_reason: StopReason
    status: str


class InvestigationPackage(BaseModel):
    run_summary: RunSummary
    research_plan: ResearchPlan
    evidence_ledger: EvidenceLedger
    selection_graph: SelectionGraph
    iteration_trace: IterationTrace
    renders: list[RenderPayload]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Expand the models to the full MVP contract**

Add the remaining models required by the design before other tasks depend on them:

```python
class CoverageScore(BaseModel):
    subtopic_coverage: float
    source_diversity: float
    evidence_density: float
    total: float


class ToolCallRecord(BaseModel):
    tool_name: str
    status: str
    provider: str | None = None
    summary: str | None = None


class IterationBudget(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RawToolResult(BaseModel):
    tool_name: str
    provider: str
    payload: dict
    ok: bool = True
    error: str | None = None


class SupervisorDecision(BaseModel):
    rationale: str
    search_actions: list[str]


class SupervisorCheckpointResult(BaseModel):
    raw_results: list[RawToolResult]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RelevanceCheckpointResult(BaseModel):
    candidates: list[EvidenceCandidate]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RequestClassification(BaseModel):
    audience_mode: str
    freshness_mode: str
    recommended_tier: Tier
    needs_clarification: bool = False
    clarification_question: str | None = None
```

- [ ] **Step 6: Run the whole model test module**

Run: `pytest tests/test_models.py -v`
Expected: PASS with all added types importable

- [ ] **Step 7: Commit**

```bash
git add deep_research/enums.py deep_research/models.py tests/test_models.py
git commit -m "feat: add deep research domain models"
```

### Task 3: Add Settings, Tier Defaults, And Pricing Tables

**Files:**
- Create: `deep_research/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

```python
from deep_research.config import ResearchConfig, ResearchSettings
from deep_research.enums import Tier


def test_settings_reads_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_DEFAULT_MAX_ITERATIONS", "4")
    settings = ResearchSettings()
    assert settings.default_max_iterations == 4


def test_research_config_builds_standard_tier_defaults() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD)
    assert config.max_iterations > 0
    assert config.cost_budget_usd > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because `deep_research.config` does not exist

- [ ] **Step 3: Implement settings, tier config, and pricing model**

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.enums import Tier


class ModelPricing(BaseModel):
    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0


class TierConfig(BaseModel):
    max_iterations: int
    cost_budget_usd: float
    time_box_seconds: int
    critique_enabled: bool = False
    judge_enabled: bool = False
    allows_council: bool = False
    requires_plan_approval: bool = True


class ResearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_", extra="ignore")

    default_tier: Tier = Tier.STANDARD
    default_max_iterations: int = 3
    default_cost_budget_usd: float = 0.10
    daily_cost_limit_usd: float = 10.0
    convergence_epsilon: float = 0.05
    convergence_min_coverage: float = 0.60
    max_tool_calls_per_cycle: int = 5
    tool_timeout_sec: int = 20
    source_quality_floor: float = 0.30
    council_size: int = 3
    council_cost_budget_usd: float = 2.0
    classifier_model: str = "gemini/gemini-2.0-flash-lite"
    planner_model: str = "gemini/gemini-2.5-flash"
    supervisor_model: str = "gemini/gemini-2.5-flash"
    relevance_scorer_model: str = "gemini/gemini-2.5-flash"
    curator_model: str = "gemini/gemini-2.0-flash-lite"
    writer_model: str = "gemini/gemini-2.5-flash"
    aggregator_model: str = "openai/gpt-4o-mini"


class ResearchConfig(BaseModel):
    tier: Tier
    max_iterations: int
    cost_budget_usd: float
    time_box_seconds: int
    council_mode: bool = False
    council_size: int = 1
    require_plan_approval: bool = True
    classifier_model: str
    planner_model: str
    supervisor_model: str
    relevance_scorer_model: str
    curator_model: str
    writer_model: str
    aggregator_model: str

    @classmethod
    def for_tier(cls, tier: Tier, settings: ResearchSettings | None = None) -> "ResearchConfig":
        settings = settings or ResearchSettings()
        mapping = {
            Tier.QUICK: TierConfig(max_iterations=2, cost_budget_usd=0.05, time_box_seconds=120),
            Tier.STANDARD: TierConfig(max_iterations=settings.default_max_iterations, cost_budget_usd=settings.default_cost_budget_usd, time_box_seconds=600),
            Tier.DEEP: TierConfig(max_iterations=6, cost_budget_usd=1.0, time_box_seconds=1800, critique_enabled=True, judge_enabled=True, allows_council=True),
            Tier.CUSTOM: TierConfig(max_iterations=settings.default_max_iterations, cost_budget_usd=settings.default_cost_budget_usd, time_box_seconds=600, allows_council=True),
        }
        base = mapping[tier]
        return cls(
            tier=tier,
            max_iterations=base.max_iterations,
            cost_budget_usd=base.cost_budget_usd,
            time_box_seconds=base.time_box_seconds,
            council_mode=False,
            council_size=settings.council_size if base.allows_council else 1,
            require_plan_approval=base.requires_plan_approval,
            classifier_model=settings.classifier_model,
            planner_model=settings.planner_model,
            supervisor_model=settings.supervisor_model,
            relevance_scorer_model=settings.relevance_scorer_model,
            curator_model=settings.curator_model,
            writer_model=settings.writer_model,
            aggregator_model=settings.aggregator_model,
        )
```

- [ ] **Step 4: Add a pricing helper test and implementation**

```python
from deep_research.config import ModelPricing


def test_model_pricing_estimates_cost() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
    input_cost = pricing.input_per_million_usd * 1000 / 1_000_000
    output_cost = pricing.output_per_million_usd * 500 / 1_000_000
    assert round(input_cost + output_cost, 6) == 0.002
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/config.py tests/test_config.py
git commit -m "feat: add runtime settings and pricing config"
```

### Task 4: Implement Prompt Files And Prompt Loader

**Files:**
- Create: `deep_research/prompts/classifier.md`
- Create: `deep_research/prompts/planner.md`
- Create: `deep_research/prompts/supervisor.md`
- Create: `deep_research/prompts/relevance_scorer.md`
- Create: `deep_research/prompts/curator.md`
- Create: `deep_research/prompts/writer.md`
- Create: `deep_research/prompts/aggregator.md`
- Create: `deep_research/prompts/question_generator.md`
- Create: `deep_research/prompts/reviewer.md`
- Create: `deep_research/prompts/judge_grounding.md`
- Create: `deep_research/prompts/judge_coherence.md`
- Create: `deep_research/prompts/loader.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing prompt loader tests**

```python
from deep_research.prompts.loader import load_prompt


def test_load_prompt_returns_markdown_contents() -> None:
    prompt = load_prompt("planner")
    assert "research" in prompt.lower()


def test_load_prompt_rejects_unknown_name() -> None:
    try:
        load_prompt("missing")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL because loader and files do not exist

- [ ] **Step 3: Add prompt loader and concise MVP prompts**

```python
from importlib.resources import files


def load_prompt(name: str) -> str:
    path = files("deep_research.prompts").joinpath(f"{name}.md")
    return path.read_text(encoding="utf-8")
```

```md
# planner.md
You are the research planner. Break the brief into concrete subtopics, search queries, sections, and success criteria. Optimize for clarity, source diversity, and iterative execution.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_research/prompts tests/test_prompts.py
git commit -m "feat: add prompt files and loader"
```

### Task 5: Build Evidence Normalization, Dedup, Quality Scoring, And Ledger Ops

**Files:**
- Create: `deep_research/providers/normalization.py`
- Create: `deep_research/evidence/dedup.py`
- Create: `deep_research/evidence/scoring.py`
- Create: `deep_research/evidence/ledger.py`
- Create: `tests/test_evidence_pipeline.py`

- [ ] **Step 1: Write failing evidence pipeline tests**

```python
from deep_research.evidence.dedup import dedupe_candidates
from deep_research.evidence.ledger import merge_candidates
from deep_research.evidence.scoring import score_candidate_quality
from deep_research.models import EvidenceCandidate, EvidenceSnippet
from deep_research.providers.normalization import normalize_tool_results


def test_normalize_tool_results_maps_dict_payloads() -> None:
    raw = [{"title": "Doc", "url": "https://example.com", "snippet": "Alpha"}]
    candidates = normalize_tool_results(raw, provider="brave", source_kind="web")
    assert candidates[0].title == "Doc"
    assert candidates[0].snippets[0].text == "Alpha"


def test_dedupe_candidates_prefers_url_identity() -> None:
    first = EvidenceCandidate(key="1", title="A", url="https://x", provider="b", source_kind="web")
    second = EvidenceCandidate(key="2", title="A2", url="https://x", provider="c", source_kind="web")
    deduped = dedupe_candidates([first, second])
    assert len(deduped) == 1


def test_merge_candidates_preserves_existing_and_appends_new() -> None:
    ledger = merge_candidates([], [EvidenceCandidate(key="1", title="A", url="https://x", provider="b", source_kind="web")])
    assert len(ledger.entries) == 1


def test_score_candidate_quality_returns_floorable_value() -> None:
    candidate = EvidenceCandidate(key="1", title="A", url="https://x", provider="b", source_kind="paper")
    assert 0.0 <= score_candidate_quality(candidate) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evidence_pipeline.py -v`
Expected: FAIL because pipeline functions do not exist

- [ ] **Step 3: Implement normalization in one file only**

```python
def normalize_tool_results(raw_results: list[dict], provider: str, source_kind: str) -> list[EvidenceCandidate]:
    normalized = []
    for idx, item in enumerate(raw_results):
        snippet = item.get("snippet") or item.get("description") or ""
        normalized.append(
            EvidenceCandidate(
                key=f"{provider}-{idx}",
                title=item.get("title") or item.get("url") or f"result-{idx}",
                url=item.get("url") or "",
                provider=provider,
                source_kind=source_kind,
                snippets=[EvidenceSnippet(text=snippet)] if snippet else [],
            )
        )
    return normalized
```

- [ ] **Step 4: Implement dedupe, scoring, and ledger merge**

```python
def dedupe_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    seen: dict[str, EvidenceCandidate] = {}
    for candidate in candidates:
        key = candidate.url or candidate.title.strip().lower()
        seen.setdefault(key, candidate)
    return list(seen.values())


def score_candidate_quality(candidate: EvidenceCandidate) -> float:
    if candidate.source_kind == "paper":
        return 0.9
    if candidate.source_kind == "docs":
        return 0.8
    if candidate.source_kind == "web":
        return 0.6
    return 0.4


def merge_candidates(existing: list[EvidenceCandidate], incoming: list[EvidenceCandidate]) -> EvidenceLedger:
    combined = dedupe_candidates([*existing, *incoming])
    return EvidenceLedger(entries=combined)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evidence_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/providers/normalization.py deep_research/evidence/dedup.py deep_research/evidence/scoring.py deep_research/evidence/ledger.py tests/test_evidence_pipeline.py
git commit -m "feat: add evidence normalization and ledger operations"
```

### Task 6: Implement Package IO And Canonical Assembly

**Files:**
- Create: `deep_research/package/assembly.py`
- Create: `deep_research/package/io.py`
- Create: `tests/test_package_io.py`

- [ ] **Step 1: Write failing package IO tests**

```python
from pathlib import Path

from deep_research.models import InvestigationPackage
from deep_research.package.io import read_package, write_markdown, write_package


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    write_markdown("# Title", path)
    assert path.read_text() == "# Title"


def test_write_and_read_package_round_trip(tmp_path: Path) -> None:
    sample_package = InvestigationPackage(
        run_summary={"run_id": "run-1", "brief": "brief", "tier": "standard", "stop_reason": "max_iterations", "status": "completed"},
        research_plan={"goal": "goal", "key_questions": [], "subtopics": [], "queries": [], "sections": [], "success_criteria": []},
        evidence_ledger={"entries": []},
        selection_graph={"items": []},
        iteration_trace={"iterations": []},
        renders=[{"name": "reading_path", "content_markdown": "# Reading Path", "citation_map": {}}],
    )
    run_dir = write_package(sample_package, tmp_path)
    restored = read_package(run_dir)
    assert restored == sample_package
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_io.py -v`
Expected: FAIL because the IO module does not exist

- [ ] **Step 3: Implement assembly and file writing**

```python
from pathlib import Path
import json


def write_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    run_dir = output_dir / package.run_summary.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "package.json").write_text(package.model_dump_json(indent=2), encoding="utf-8")
    for render in package.renders:
        write_markdown(render.content_markdown, run_dir / "renders" / f"{render.name}.md")
    return run_dir


def read_package(run_dir: Path) -> InvestigationPackage:
    return InvestigationPackage.model_validate_json((run_dir / "package.json").read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_package_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deep_research/package/assembly.py deep_research/package/io.py tests/test_package_io.py
git commit -m "feat: add package assembly and io"
```

### Task 7: Implement State Reader Tools And Guarded Bash Executor

**Files:**
- Create: `deep_research/tools/state_reader.py`
- Create: `deep_research/tools/bash_executor.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tool tests**

```python
from deep_research.models import EvidenceLedger, ResearchPlan
from deep_research.tools.bash_executor import run_bash
from deep_research.tools.state_reader import read_gaps, read_plan


def test_read_plan_returns_serializable_plan() -> None:
    plan = ResearchPlan(goal="g", key_questions=[], subtopics=["a"], queries=[], sections=[], success_criteria=[])
    result = read_plan(plan)
    assert result["goal"] == "g"


def test_read_gaps_reports_missing_subtopics() -> None:
    plan = ResearchPlan(goal="g", key_questions=[], subtopics=["a", "b"], queries=[], sections=[], success_criteria=[])
    ledger = EvidenceLedger(entries=[])
    assert read_gaps(plan, ledger) == ["a", "b"]


def test_run_bash_blocks_dangerous_commands() -> None:
    result = run_bash("rm -rf /")
    assert result.ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL because tools are missing

- [ ] **Step 3: Implement state readers as pure functions first**

```python
def read_plan(plan: ResearchPlan) -> dict:
    return plan.model_dump()


def read_gaps(plan: ResearchPlan, ledger: EvidenceLedger) -> list[str]:
    covered = {entry.title for entry in ledger.entries}
    return [subtopic for subtopic in plan.subtopics if subtopic not in covered]
```

- [ ] **Step 4: Implement guarded bash execution with explicit limitations**

```python
import shlex
import subprocess
import tempfile

from deep_research.models import RawToolResult


DENYLIST = {"rm", "mv", "sudo", "chmod", "chown", "dd"}


def run_bash(command: str, timeout_sec: int = 20) -> RawToolResult:
    argv = shlex.split(command)
    if not argv or argv[0] in DENYLIST:
        return RawToolResult(tool_name="run_bash", provider="bash", payload={}, ok=False, error="command not allowed")
    with tempfile.TemporaryDirectory(prefix="deep-research-") as temp_dir:
        completed = subprocess.run(argv, cwd=temp_dir, capture_output=True, text=True, timeout=timeout_sec, env={})
    return RawToolResult(tool_name="run_bash", provider="bash", payload={"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}, ok=completed.returncode == 0, error=None if completed.returncode == 0 else completed.stderr)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/tools/state_reader.py deep_research/tools/bash_executor.py tests/test_tools.py
git commit -m "feat: add state readers and guarded bash tool"
```

### Task 8: Add MCP Toolset Construction And Supervisor Dependencies

**Files:**
- Create: `deep_research/providers/mcp_config.py`
- Create: `tests/test_mcp_config.py`

- [ ] **Step 1: Write failing MCP config tests**

```python
from deep_research.providers.mcp_config import MCPServerConfig, build_mcp_toolsets


def test_build_mcp_toolsets_returns_empty_without_servers() -> None:
    assert build_mcp_toolsets([]) == []


def test_build_mcp_toolsets_calls_factories() -> None:
    sentinel = object()
    toolsets = build_mcp_toolsets([MCPServerConfig(id="brave", factory=lambda: sentinel)])
    assert toolsets == [sentinel]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_config.py -v`
Expected: FAIL because the module does not exist

- [ ] **Step 3: Implement concrete but minimal MCP server construction**

```python
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerConfig:
    id: str
    factory: Callable[[], object]


def build_mcp_toolsets(servers: list[MCPServerConfig]) -> list[object]:
    return [server.factory() for server in servers]
```

This keeps the library independent from the exact concrete MCP constructor for the installed `pydantic_ai` version while still making server registration explicit and testable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp_config.py -v`
Expected: PASS locally when no servers are configured

- [ ] **Step 5: Commit**

```bash
git add deep_research/providers/mcp_config.py tests/test_mcp_config.py
git commit -m "feat: add MCP toolset configuration"
```

### Task 9: Define Wrapped PydanticAI Agents For MVP Roles

**Files:**
- Create: `deep_research/agents/classifier.py`
- Create: `deep_research/agents/planner.py`
- Create: `deep_research/agents/supervisor.py`
- Create: `deep_research/agents/relevance_scorer.py`
- Create: `deep_research/agents/curator.py`
- Create: `deep_research/agents/writer.py`
- Create: `deep_research/agents/aggregator.py`
- Create: `tests/test_agent_factories.py`

- [ ] **Step 1: Write failing agent factory tests**

```python
from deep_research.agents.classifier import build_classifier_agent


def test_build_classifier_agent_returns_wrapped_agent() -> None:
    agent = build_classifier_agent(model_name="test")
    assert agent is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_factories.py -v`
Expected: FAIL because agent modules are missing

- [ ] **Step 3: Implement factories that load prompts and wrap agents once**

```python
from pydantic_ai import Agent
from kitaru.adapters import pydantic_ai as kp

from deep_research.models import RequestClassification
from deep_research.prompts.loader import load_prompt


def build_classifier_agent(model_name: str):
    return kp.wrap(
        Agent(
            model_name,
            name="classifier",
            result_type=RequestClassification,
            system_prompt=load_prompt("classifier"),
        )
    )
```

- [ ] **Step 4: Repeat the same pattern for planner, supervisor, relevance scorer, curator, writer, and aggregator**

Use:

```python
def build_supervisor_agent(model_name: str, toolsets: list[object], tools: list[object]):
    return kp.wrap(
        Agent(
            model_name,
            name="supervisor",
            system_prompt=load_prompt("supervisor"),
            toolsets=toolsets,
            tools=tools,
        ),
        tool_capture_config={"mode": "full"},
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_factories.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/agents tests/test_agent_factories.py
git commit -m "feat: add wrapped PydanticAI agent factories"
```

### Task 10: Implement Cost Aggregation And Budget Enforcement

**Files:**
- Create: `deep_research/flow/costing.py`
- Create: `tests/test_costing.py`

- [ ] **Step 1: Write failing cost tests**

```python
from deep_research.config import ModelPricing
from deep_research.flow.costing import budget_from_agent_result, estimate_cost_usd, merge_usage
from deep_research.models import IterationBudget


def test_estimate_cost_usd_uses_input_and_output_pricing() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
    assert round(estimate_cost_usd(1000, 500, pricing), 6) == 0.002


def test_merge_usage_adds_all_token_fields() -> None:
    combined = merge_usage(IterationBudget(input_tokens=1, output_tokens=2, total_tokens=3, estimated_cost_usd=0.1), IterationBudget(input_tokens=4, output_tokens=5, total_tokens=9, estimated_cost_usd=0.2))
    assert combined.total_tokens == 12
    assert combined.estimated_cost_usd == 0.3


def test_budget_from_agent_result_supports_callable_usage() -> None:
    class FakeUsage:
        prompt_tokens = 1000
        completion_tokens = 500
        total_tokens = 1500

    class FakeResult:
        def usage(self):
            return FakeUsage()

    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
    budget = budget_from_agent_result(FakeResult(), pricing)
    assert budget.total_tokens == 1500
    assert round(budget.estimated_cost_usd, 6) == 0.002
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_costing.py -v`
Expected: FAIL because costing helpers are missing

- [ ] **Step 3: Implement explicit cost helpers**

```python
from deep_research.config import ModelPricing
from deep_research.models import IterationBudget


def estimate_cost_usd(input_tokens: int, output_tokens: int, pricing: ModelPricing) -> float:
    input_cost = pricing.input_per_million_usd * input_tokens / 1_000_000
    output_cost = pricing.output_per_million_usd * output_tokens / 1_000_000
    return round(input_cost + output_cost, 6)


def budget_from_agent_result(result: object, pricing: ModelPricing) -> IterationBudget:
    usage_attr = getattr(result, "usage", None)
    usage = usage_attr() if callable(usage_attr) else usage_attr
    input_tokens = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
    output_tokens = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    return IterationBudget(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, pricing),
    )


def merge_usage(left: IterationBudget, right: IterationBudget) -> IterationBudget:
    return IterationBudget(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost_usd=round(left.estimated_cost_usd + right.estimated_cost_usd, 6),
    )
```

- [ ] **Step 4: Add a stop-check helper for the flow**

```python
def is_budget_exhausted(spent_usd: float, limit_usd: float) -> bool:
    return spent_usd >= limit_usd
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_costing.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/flow/costing.py tests/test_costing.py
git commit -m "feat: add explicit token and cost accounting"
```

### Task 11: Implement Research Checkpoints For Single-Generator Flow

**Files:**
- Create: `deep_research/checkpoints/classify.py`
- Create: `deep_research/checkpoints/plan.py`
- Create: `deep_research/checkpoints/supervisor.py`
- Create: `deep_research/checkpoints/normalize.py`
- Create: `deep_research/checkpoints/relevance.py`
- Create: `deep_research/checkpoints/merge.py`
- Create: `deep_research/checkpoints/evaluate.py`
- Create: `deep_research/checkpoints/select.py`
- Create: `tests/test_checkpoints.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
from deep_research.checkpoints.normalize import normalize_evidence
from deep_research.models import RawToolResult


def test_normalize_evidence_accepts_raw_results() -> None:
    result = RawToolResult(tool_name="search", provider="brave", payload={"results": [{"title": "A", "url": "https://a", "snippet": "x"}]})
    normalized = normalize_evidence([result])
    assert normalized[0].title == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpoints.py -v`
Expected: FAIL because checkpoint modules are missing

- [ ] **Step 3: Implement explicit checkpoint wrappers**

```python
from kitaru import checkpoint


@checkpoint(type="llm_call")
def classify_request(brief: str, config: ResearchConfig | None = None) -> RequestClassification:
    model_name = config.classifier_model if config else ResearchConfig.for_tier(Tier.STANDARD).classifier_model
    return build_classifier_agent(model_name).run_sync(brief).output
```

```python
@checkpoint(type="tool_call")
def normalize_evidence(raw_results: list[RawToolResult]) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    for result in raw_results:
        if result.ok:
            candidates.extend(normalize_tool_results(result.payload.get("results", []), provider=result.provider, source_kind=result.payload.get("source_kind", "web")))
    return candidates
```

- [ ] **Step 4: Keep each checkpoint focused on one durable boundary**

Implement the remaining checkpoint signatures exactly once and keep them stable:

```python
build_plan(brief: str, classification: RequestClassification, tier: Tier) -> ResearchPlan
_execute_supervisor_turn(plan: ResearchPlan, ledger: EvidenceLedger, iteration: int, config: ResearchConfig, model_name: str | None = None) -> SupervisorCheckpointResult
run_supervisor(plan: ResearchPlan, ledger: EvidenceLedger, iteration: int, config: ResearchConfig) -> SupervisorCheckpointResult
score_relevance(candidates: list[EvidenceCandidate], plan: ResearchPlan, config: ResearchConfig) -> RelevanceCheckpointResult
merge_evidence(scored: list[EvidenceCandidate], ledger: EvidenceLedger) -> EvidenceLedger
evaluate_coverage(ledger: EvidenceLedger, plan: ResearchPlan) -> CoverageScore
build_selection_graph(ledger: EvidenceLedger, plan: ResearchPlan) -> SelectionGraph
```

`run_supervisor` and `run_council_generator` must both call the same private `_execute_supervisor_turn(...)` helper so the council path reuses agent logic without nesting one checkpoint inside another.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_checkpoints.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/checkpoints tests/test_checkpoints.py
git commit -m "feat: add durable research checkpoints"
```

### Task 12: Implement Council Mode Without Nested Checkpoints

**Files:**
- Create: `deep_research/checkpoints/council.py`
- Create: `tests/test_council.py`

- [ ] **Step 1: Write failing council tests around generator checkpoint behavior**

```python
from deep_research.checkpoints.council import aggregate_council_results
from deep_research.models import IterationBudget, RawToolResult, SupervisorCheckpointResult


def test_aggregate_council_results_flattens_all_generators() -> None:
    grouped = [
        SupervisorCheckpointResult(raw_results=[RawToolResult(tool_name="a", provider="m1", payload={})], budget=IterationBudget(total_tokens=10, estimated_cost_usd=0.01)),
        SupervisorCheckpointResult(raw_results=[RawToolResult(tool_name="b", provider="m2", payload={})], budget=IterationBudget(total_tokens=20, estimated_cost_usd=0.02)),
    ]
    merged = aggregate_council_results(grouped)
    assert [item.tool_name for item in merged.raw_results] == ["a", "b"]
    assert merged.budget.total_tokens == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_council.py -v`
Expected: FAIL because council helpers do not exist

- [ ] **Step 3: Add only generator and aggregation helpers as checkpoint-safe units**

```python
from kitaru import checkpoint


@checkpoint(type="llm_call")
def run_council_generator(plan: ResearchPlan, ledger: EvidenceLedger, iteration: int, model_name: str, config: ResearchConfig) -> SupervisorCheckpointResult:
    return _execute_supervisor_turn(plan, ledger, iteration, config, model_name=model_name)


def aggregate_council_results(grouped_results: list[SupervisorCheckpointResult]) -> SupervisorCheckpointResult:
    merged: list[RawToolResult] = []
    budget = IterationBudget()
    for group in grouped_results:
        merged.extend(group.raw_results)
        budget = merge_usage(budget, group.budget)
    return SupervisorCheckpointResult(raw_results=merged, budget=budget)
```

- [ ] **Step 4: Add a design comment in the module and test for it indirectly**

Use this comment in `deep_research/checkpoints/council.py`:

```python
# Council fan-out happens in flow scope via checkpoint.submit(); this module only
# defines the generator checkpoint and pure aggregation helpers.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_council.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/checkpoints/council.py tests/test_council.py
git commit -m "feat: add council generator checkpoints"
```

### Task 13: Implement Convergence Rules

**Files:**
- Create: `deep_research/flow/convergence.py`
- Create: `tests/test_convergence.py`

- [ ] **Step 1: Write failing convergence tests**

```python
from deep_research.flow.convergence import check_convergence
from deep_research.models import CoverageScore, IterationRecord


def test_check_convergence_stops_on_coverage_target() -> None:
    current = CoverageScore(subtopic_coverage=0.8, source_diversity=0.7, evidence_density=0.7, total=0.733)
    history = [IterationRecord(iteration=0, new_candidate_count=4, coverage=0.68)]
    decision = check_convergence(current, history, spent_usd=0.01, elapsed_seconds=30, max_iterations=5, epsilon=0.05, min_coverage=0.70, budget_limit_usd=1.0, time_limit_seconds=60)
    assert decision.should_stop is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convergence.py -v`
Expected: FAIL because convergence helpers do not exist

- [ ] **Step 3: Implement stop-decision model and rules**

```python
from pydantic import BaseModel

from deep_research.enums import StopReason


class StopDecision(BaseModel):
    should_stop: bool
    reason: StopReason | None = None


def check_convergence(current: CoverageScore, history: list[IterationRecord], *, spent_usd: float, elapsed_seconds: int, max_iterations: int, epsilon: float, min_coverage: float, budget_limit_usd: float, time_limit_seconds: int) -> StopDecision:
    if spent_usd >= budget_limit_usd:
        return StopDecision(should_stop=True, reason=StopReason.BUDGET_EXHAUSTED)
    if elapsed_seconds >= time_limit_seconds:
        return StopDecision(should_stop=True, reason=StopReason.TIME_EXHAUSTED)
    if history and current.total >= min_coverage and (current.total - history[-1].coverage) < epsilon:
        return StopDecision(should_stop=True, reason=StopReason.CONVERGED)
    if len(history) + 1 >= max_iterations:
        return StopDecision(should_stop=True, reason=StopReason.MAX_ITERATIONS)
    return StopDecision(should_stop=False)
```

- [ ] **Step 4: Add tests for diminishing returns and loop stall**

```python
def test_check_convergence_stops_on_diminishing_returns() -> None:
    history = [
        IterationRecord(iteration=0, new_candidate_count=1, coverage=0.4),
        IterationRecord(iteration=1, new_candidate_count=1, coverage=0.45),
    ]
    current = CoverageScore(subtopic_coverage=0.5, source_diversity=0.5, evidence_density=0.5, total=0.5)
    decision = check_convergence(current, history, spent_usd=0.01, elapsed_seconds=30, max_iterations=5, epsilon=0.05, min_coverage=0.9, budget_limit_usd=1.0, time_limit_seconds=60)
    assert decision.reason is not None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_convergence.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/flow/convergence.py tests/test_convergence.py
git commit -m "feat: add convergence and stop rules"
```

### Task 14: Implement The Main Flow With Plan Approval And Council Fan-Out

**Files:**
- Create: `deep_research/flow/research_flow.py`
- Create: `tests/test_research_flow_unit.py`

- [ ] **Step 1: Write a failing orchestration test using monkeypatched checkpoints**

```python
from deep_research.flow.research_flow import _run_iteration


def test_run_iteration_uses_council_when_enabled(monkeypatch) -> None:
    called = {"council": False}

    def fake_council(*args, **kwargs):
        called["council"] = True
        return type("SupervisorCheckpointResult", (), {"raw_results": [], "budget": type("Budget", (), {"estimated_cost_usd": 0.0})()})()

    monkeypatch.setattr("deep_research.flow.research_flow._run_council_iteration", fake_council)
    _run_iteration(plan=None, ledger=None, iteration=0, config=type("C", (), {"council_mode": True, "council_size": 3})(), council_models=["m1", "m2", "m3"])
    assert called["council"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_research_flow_unit.py -v`
Expected: FAIL because the flow module does not exist

- [ ] **Step 3: Implement a pure helper for flow-scope council fan-out**

```python
def _run_council_iteration(plan, ledger, iteration, config, council_models):
    futures = [
        run_council_generator.submit(plan, ledger, iteration, model_name, config, id=f"council_{index}")
        for index, model_name in enumerate(council_models)
    ]
    return aggregate_council_results([future.load() for future in futures])


def _run_iteration(plan, ledger, iteration, config, council_models):
    if config.council_mode:
        return _run_council_iteration(plan, ledger, iteration, config, council_models)
    return run_supervisor(plan, ledger, iteration, config)
```

- [ ] **Step 4: Implement the durable `@flow` with flow-owned waits and convergence**

```python
from kitaru import flow, log, wait


@flow
def research_flow(brief: str, tier: str = "auto", config: ResearchConfig | None = None) -> InvestigationPackage:
    classification = classify_request(brief, config)
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    config = config or ResearchConfig.for_tier(resolved_tier)
    if config.tier != resolved_tier:
        config = config.model_copy(update={"tier": resolved_tier})
    if classification.needs_clarification and classification.clarification_question:
        brief = wait(name="clarify_brief", schema=str, question=classification.clarification_question)
        classification = classify_request(brief, config)
    plan = build_plan(brief, classification, config.tier)
    if config.require_plan_approval:
        approved = wait(name="approve_plan", schema=bool, question=f"Approve plan for: {plan.goal}?")
        if approved is False:
            raise ValueError("plan not approved")
```

- [ ] **Step 5: Finish the loop with explicit budget tracking and council branching**

Add these exact responsibilities to `research_flow`:

```python
    ledger = EvidenceLedger()
    iteration_history: list[IterationRecord] = []
    spent_usd = 0.0
    council_models = _resolve_council_models(config)
    for iteration in range(config.max_iterations):
        supervisor_result = _run_iteration(plan, ledger, iteration, config, council_models)
        spent_usd += supervisor_result.budget.estimated_cost_usd
        candidates = normalize_evidence(supervisor_result.raw_results)
        relevance_result = score_relevance(candidates, plan, config)
        spent_usd += relevance_result.budget.estimated_cost_usd
        scored = relevance_result.candidates
        ledger = merge_evidence(scored, ledger)
        coverage = evaluate_coverage(ledger, plan)
        iteration_history.append(IterationRecord(iteration=iteration, new_candidate_count=len(candidates), coverage=coverage.total))
        decision = check_convergence(coverage, iteration_history[:-1], spent_usd=spent_usd, elapsed_seconds=0, max_iterations=config.max_iterations, epsilon=0.05, min_coverage=0.60, budget_limit_usd=config.cost_budget_usd, time_limit_seconds=config.time_box_seconds)
        log(iteration=iteration, coverage=coverage.total, spent_usd=spent_usd)
        if decision.should_stop:
            break
```

- [ ] **Step 6: Assemble the package and return it from the flow**

Use:

```python
    selection = build_selection_graph(ledger, plan)
    reading_path = RenderPayload(name="reading_path", content_markdown="# Reading Path\n")
    backing_report = RenderPayload(name="backing_report", content_markdown=f"# Backing Report\n\nGoal: {plan.goal}\n")
    return assemble_package(
        brief=brief,
        config=config,
        plan=plan,
        ledger=ledger,
        selection=selection,
        renders=[reading_path, backing_report],
        iterations=iteration_history,
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_research_flow_unit.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add deep_research/flow/research_flow.py tests/test_research_flow_unit.py
git commit -m "feat: add research flow orchestration"
```

### Task 15: Implement Renderers And Wire Them Into Package Assembly

**Files:**
- Create: `deep_research/renderers/reading_path.py`
- Create: `deep_research/renderers/backing_report.py`
- Modify: `deep_research/flow/research_flow.py`
- Create: `tests/test_renderers.py`

- [ ] **Step 1: Write failing renderer tests**

```python
from deep_research.models import SelectionGraph, SelectionItem
from deep_research.renderers.reading_path import render_reading_path


def test_render_reading_path_returns_markdown_payload() -> None:
    selection = SelectionGraph(items=[SelectionItem(candidate_key="a", rationale="useful")])
    payload = render_reading_path(selection)
    assert payload.name == "reading_path"
    assert "useful" in payload.content_markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_renderers.py -v`
Expected: FAIL because renderer modules are missing

- [ ] **Step 3: Implement minimal renderer checkpoints**

```python
from kitaru import checkpoint


@checkpoint(type="llm_call")
def render_reading_path(selection: SelectionGraph) -> RenderPayload:
    lines = ["# Reading Path", ""]
    for item in selection.items:
        lines.append(f"- {item.candidate_key}: {item.rationale}")
    return RenderPayload(name="reading_path", content_markdown="\n".join(lines))
```

```python
@checkpoint(type="llm_call")
def render_backing_report(selection: SelectionGraph, ledger: EvidenceLedger, plan: ResearchPlan) -> RenderPayload:
    return RenderPayload(name="backing_report", content_markdown=f"# Backing Report\n\nGoal: {plan.goal}\n\nSelected: {len(selection.items)}")
```

- [ ] **Step 4: Replace the inline render stubs in `research_flow` with renderer checkpoint calls**

```python
    selection = build_selection_graph(ledger, plan)
    reading_path = render_reading_path(selection)
    backing_report = render_backing_report(selection, ledger, plan)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_renderers.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/renderers/reading_path.py deep_research/renderers/backing_report.py deep_research/flow/research_flow.py tests/test_renderers.py
git commit -m "feat: add MVP renderers"
```

### Task 16: Add End-To-End Integration Tests For Single And Council Modes

**Files:**
- Create: `tests/test_research_flow_integration.py`

- [ ] **Step 1: Write a failing single-generator integration test with fakes**

```python
from deep_research.flow.research_flow import research_flow
from deep_research.config import ResearchConfig
from deep_research.enums import Tier


def test_research_flow_returns_package(monkeypatch) -> None:
    monkeypatch.setattr("deep_research.flow.research_flow.wait", lambda **kwargs: True)
    monkeypatch.setattr("deep_research.flow.research_flow.classify_request", lambda *args, **kwargs: type("C", (), {"needs_clarification": False, "clarification_question": None, "recommended_tier": Tier.STANDARD})())
    handle = research_flow.run("learn kitaru", config=ResearchConfig.for_tier(Tier.STANDARD))
    result = handle.wait()
    assert result is not None
```

- [ ] **Step 2: Write a failing council integration test with fake futures**

```python
def test_council_flow_aggregates_multiple_generator_results(monkeypatch) -> None:
    class FakeFuture:
        def __init__(self, payload):
            self.payload = payload
        def load(self):
            return self.payload

    monkeypatch.setattr("deep_research.flow.research_flow.wait", lambda **kwargs: True)
    monkeypatch.setattr("deep_research.flow.research_flow.run_council_generator", type("G", (), {"submit": staticmethod(lambda *args, **kwargs: FakeFuture([]))})())
    config = ResearchConfig.for_tier(Tier.DEEP).model_copy(update={"council_mode": True, "council_size": 3})
    handle = research_flow.run("learn kitaru", config=config)
    result = handle.wait()
    assert result is not None
```

- [ ] **Step 3: Run tests to verify they fail first**

Run: `pytest tests/test_research_flow_integration.py -v`
Expected: FAIL until the flow wiring and fake seams are correct

- [ ] **Step 4: Make the seams testable without changing behavior**

Use small helpers in `deep_research/flow/research_flow.py` so monkeypatching is easy:

```python
def _resolve_council_models(config: ResearchConfig) -> list[str]:
    return [config.tier.value] * config.council_size
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_research_flow_integration.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_research_flow_integration.py deep_research/flow/research_flow.py
git commit -m "test: add research flow integration coverage"
```

### Task 17: Add Replay, Wait, And Operator Workflow Verification Notes

**Files:**
- Modify: `docs/superpowers/specs/2026-04-06-deep-research-engine-design.md`
- Create: `tests/test_operator_contract.py`

- [ ] **Step 1: Write a failing operator contract test for stable names**

```python
from deep_research.flow.research_flow import APPROVE_PLAN_WAIT_NAME, CLASSIFY_CHECKPOINT_NAME


def test_stable_operator_names_are_exported() -> None:
    assert APPROVE_PLAN_WAIT_NAME == "approve_plan"
    assert CLASSIFY_CHECKPOINT_NAME == "classify_request"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_operator_contract.py -v`
Expected: FAIL because exported constants are missing

- [ ] **Step 3: Export stable public names from the flow module**

```python
CLASSIFY_CHECKPOINT_NAME = "classify_request"
PLAN_CHECKPOINT_NAME = "build_plan"
SUPERVISOR_CHECKPOINT_NAME = "run_supervisor"
COUNCIL_GENERATOR_CHECKPOINT_NAME = "run_council_generator"
APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"
```

- [ ] **Step 4: Add operator workflow notes to the design spec**

Append a short section with exact commands:

```md
## MVP Operations

- Launch: `python -c "from deep_research.flow.research_flow import research_flow; handle = research_flow.run('brief'); print(handle.exec_id)"`
- Launch: `uv run python -c "from deep_research.flow.research_flow import research_flow; handle = research_flow.run('brief'); print(handle.exec_id)"`
- Provide wait input: `kitaru executions input <exec_id> --value true`
- Resume: `kitaru executions resume <exec_id>`
- Replay from checkpoint: `kitaru executions replay <exec_id> --from build_plan`
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_operator_contract.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deep_research/flow/research_flow.py tests/test_operator_contract.py docs/superpowers/specs/2026-04-06-deep-research-engine-design.md
git commit -m "docs: add operator and replay contract"
```

### Task 18: Run Full Verification Suite

**Files:**
- Modify: none
- Test: `tests/`

- [ ] **Step 1: Run targeted unit tests**

Run: `pytest tests/test_imports.py tests/test_models.py tests/test_config.py tests/test_prompts.py tests/test_evidence_pipeline.py tests/test_package_io.py tests/test_tools.py tests/test_mcp_config.py tests/test_agent_factories.py tests/test_costing.py tests/test_checkpoints.py tests/test_council.py tests/test_convergence.py tests/test_research_flow_unit.py tests/test_renderers.py tests/test_research_flow_integration.py tests/test_operator_contract.py -v`
Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests -v`
Expected: PASS

- [ ] **Step 3: Perform a manual council-mode smoke run**

Run: `uv run python -c "from deep_research.config import ResearchConfig; from deep_research.enums import Tier; from deep_research.flow.research_flow import research_flow; cfg = ResearchConfig.for_tier(Tier.DEEP); cfg = cfg.model_copy(update={'council_mode': True, 'council_size': 3}); handle = research_flow.run('Explain Kitaru replay anchors', config=cfg); print(handle.exec_id)"`
Expected: execution starts and emits an execution ID

- [ ] **Step 4: Verify plan approval wait can be resolved**

Run: `kitaru executions input <exec_id> --value true && kitaru executions resume <exec_id>`
Expected: run advances beyond `approve_plan`

- [ ] **Step 5: Verify replay from stable checkpoint works**

Run: `kitaru executions replay <exec_id> --from run_supervisor`
Expected: replayed execution is created without recomputing earlier checkpoints

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "test: verify MVP deep research engine"
```

---

## Self-Review

- **Spec coverage:** This plan covers the MVP durable flow, prompts, package models, MCP and bash provider integration, council mode, plan approval, convergence, package output, and replay/operations. It intentionally defers critique, judges, and `full_report.md` to keep the first shippable slice coherent.
- **Placeholder scan:** No `TODO`, `TBD`, or hand-wavy “write tests later” steps remain. The guarded bash executor is described honestly as guarded execution, not sandboxing.
- **Type consistency:** Stable checkpoint names, wait names, cost helpers, and flow/checkpoint boundaries are consistent across later tasks.

## Phase 2 Plan After MVP

Once Task 18 passes, create a follow-on plan for:

- reviewer and judge agents
- cross-provider model selection
- `full_report.md`
- richer coverage math
- persisted daily budget tracking
- end-to-end runs with real MCP providers in CI

Plan complete and saved to `docs/superpowers/plans/2026-04-06-deep-research-engine-mvp.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
