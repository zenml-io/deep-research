"""Judge checkpoint — compares generator outputs in council mode."""

import json
import logging
from typing import Any

from kitaru import checkpoint

from research.agents.judge import build_judge_agent
from research.contracts.package import CouncilComparison, InvestigationPackage

logger = logging.getLogger(__name__)


@checkpoint(type="llm_call")
def run_judge(
    packages: dict[str, InvestigationPackage],
    model_name: str,
    model_settings: dict[str, Any] | None = None,
) -> CouncilComparison:
    """Compare investigation packages from different generators.

    Produces a CouncilComparison with textual comparison, per-generator
    scores, and a recommendation.  The judge should run on a provider
    distinct from both generators to eliminate model-bias.

    Args:
        packages: Mapping of generator name to its InvestigationPackage.
        model_name: PydanticAI model string for the judge agent.
        model_settings: Optional provider-specific model settings dict.

    Returns:
        A CouncilComparison with comparison text, scores, and recommendation.
    """
    agent = build_judge_agent(model_name, model_settings=model_settings)

    # Build comparison prompt from package data
    summaries: dict[str, dict] = {}
    for gen_name, pkg in packages.items():
        content = ""
        if pkg.final_report:
            content = pkg.final_report.content
        elif pkg.draft:
            content = pkg.draft.content
        else:
            content = "(no report produced)"
        summaries[gen_name] = {
            "generator": gen_name,
            "tier": pkg.metadata.tier,
            "iterations": pkg.metadata.total_iterations,
            "cost_usd": pkg.metadata.total_cost_usd,
            "evidence_items": len(pkg.ledger.items),
            "report_content": content,
        }

    prompt = json.dumps(summaries, indent=2)
    return agent.run_sync(prompt).output
