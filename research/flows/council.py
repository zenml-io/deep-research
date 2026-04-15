"""Council flow — multi-generator comparison mode.

Separate @flow that runs the default pipeline once per generator model,
then runs a judge checkpoint to compare outputs.  Opt-in product mode.
"""

from __future__ import annotations

import logging

from kitaru import flow

from research.checkpoints.judge import run_judge
from research.config.settings import ResearchConfig
from research.config.slots import ModelSlotConfig
from research.contracts.package import CouncilPackage, InvestigationPackage
from research.flows.deep_research import deep_research

logger = logging.getLogger(__name__)


class CouncilConfigError(Exception):
    """Raised when council configuration is invalid."""


def _detect_provider_compromise(
    generator_slots: dict[str, ModelSlotConfig],
    judge_slot: ModelSlotConfig,
) -> bool:
    """Check if the judge shares a provider with any generator."""
    generator_providers = {slot.provider for slot in generator_slots.values()}
    return judge_slot.provider in generator_providers


@flow
def council_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
    generator_slots: dict[str, ModelSlotConfig] | None = None,
) -> CouncilPackage:
    """Run council-mode research with multiple generators.

    Runs the full default pipeline once per generator, then runs a judge
    to compare outputs.  Uses the judge's recommendation as the default
    canonical_generator (operator override via wait() deferred to a
    future slice).

    Args:
        question: The research question.
        tier: Research tier (quick/standard/deep).
        config: Pre-built ResearchConfig, or None to build from tier.
        generator_slots: Mapping of generator name -> ModelSlotConfig.
            Must contain at least two entries for a meaningful comparison.

    Returns:
        A CouncilPackage with per-generator packages and comparison.

    Raises:
        CouncilConfigError: If the config lacks a judge slot.
    """
    cfg = config or ResearchConfig.for_tier(tier)

    # Resolve generator slots — default to the single config generator
    if generator_slots is None:
        default_gen = cfg.slots.get("generator")
        if default_gen is None:
            raise CouncilConfigError("No generator slot in config")
        generator_slots = {"generator_a": default_gen}
        logger.warning(
            "Council mode with single generator — comparison will be trivial"
        )

    if len(generator_slots) < 2:
        logger.warning(
            "Council mode with %d generator(s) — comparison may be trivial",
            len(generator_slots),
        )

    # Resolve judge slot
    judge_slot = cfg.slots.get("judge")
    if judge_slot is None:
        raise CouncilConfigError(
            "Council mode requires a 'judge' model slot in config. "
            "Use a tier that includes a judge (standard or deep)."
        )

    # Check for provider compromise
    compromise = _detect_provider_compromise(generator_slots, judge_slot)
    if compromise:
        logger.warning(
            "Council provider compromise: judge provider '%s' matches a "
            "generator provider.  Recording council_provider_compromise=True.",
            judge_slot.provider,
        )

    # Run default pipeline for each generator
    packages: dict[str, InvestigationPackage] = {}
    for gen_name, gen_slot in generator_slots.items():
        logger.info(
            "Council: running pipeline for generator '%s' (%s)",
            gen_name,
            gen_slot.model_string,
        )
        gen_config = cfg.model_copy(
            update={"slots": {**cfg.slots, "generator": gen_slot}}
        )
        pkg = deep_research(question, tier=tier, config=gen_config)
        packages[gen_name] = pkg

    # Run judge
    judge_model = judge_slot.model_string
    comparison = run_judge(packages, judge_model)

    # Build council package — canonical defaults to judge recommendation
    return CouncilPackage(
        schema_version="1.0",
        council_provider_compromise=compromise,
        comparison=comparison,
        packages=packages,
        canonical_generator=comparison.recommended_generator,
    )
