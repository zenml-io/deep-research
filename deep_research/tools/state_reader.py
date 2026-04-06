from deep_research.models import EvidenceLedger, ResearchPlan


def read_plan(plan: ResearchPlan) -> dict:
    return plan.model_dump()


def read_gaps(plan: ResearchPlan, ledger: EvidenceLedger) -> list[str]:
    covered = {entry.title for entry in ledger.entries}
    return [subtopic for subtopic in plan.subtopics if subtopic not in covered]
