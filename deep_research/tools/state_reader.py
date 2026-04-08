from deep_research.models import EvidenceLedger, ResearchPlan


def read_plan(plan: ResearchPlan) -> dict:
    """Serialize a research plan to a plain dictionary."""
    return plan.model_dump()


def read_gaps(plan: ResearchPlan, ledger: EvidenceLedger) -> list[str]:
    """Return subtopics from the plan not yet covered by ledger entries."""
    covered: set[str] = set()
    for entry in ledger.entries:
        covered.update(st.lower() for st in entry.matched_subtopics)
    return [
        subtopic
        for subtopic in plan.subtopics
        if subtopic.lower() not in covered
    ]
