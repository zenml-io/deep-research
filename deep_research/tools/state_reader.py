from deep_research.models import EvidenceLedger, ResearchPlan


def read_plan(plan: ResearchPlan) -> dict:
    """Serialize a research plan to a plain dictionary."""
    return plan.model_dump()


def read_gaps(plan: ResearchPlan, ledger: EvidenceLedger) -> list[str]:
    """Return subtopics from the plan not yet covered by ledger entries."""
    covered = {entry.title for entry in ledger.entries}
    return [subtopic for subtopic in plan.subtopics if subtopic not in covered]
