"""Council flow — multi-generator comparison mode.

Separate @flow that runs the default pipeline once per generator model,
then runs a judge checkpoint to compare outputs and waits for explicit
operator selection of the canonical generator.
"""

import logging

from kitaru import flow, wait

from research.checkpoints.judge import run_judge
from research.config.settings import ResearchConfig
from research.config.slots import ModelSlotConfig
from research.contracts.package import CouncilPackage, InvestigationPackage
from research.flows.deep_research import FlowTimeoutError, _run_deep_research_pipeline

logger = logging.getLogger(__name__)


class CouncilConfigError(Exception):
    """Raised when council configuration is invalid."""


class CouncilSelectionError(Exception):
    """Raised when the operator selects an invalid council winner."""


def _detect_provider_compromise(
    generator_slots: dict[str, ModelSlotConfig],
    judge_slot: ModelSlotConfig,
) -> bool:
    """Check if the judge shares a provider with any generator."""
    generator_providers = {slot.provider for slot in generator_slots.values()}
    return judge_slot.provider in generator_providers


def _default_generator_slots(cfg: ResearchConfig) -> dict[str, ModelSlotConfig]:
    """Choose a production-friendly default generator pair for council mode."""
    default_gen = cfg.slots.get("generator")
    if default_gen is None:
        raise CouncilConfigError("No generator slot in config")

    reviewer_gen = cfg.slots.get("reviewer")
    if reviewer_gen is None:
        logger.warning(
            "Council mode could not derive a second generator slot; running single-generator council"
        )
        return {"generator_a": default_gen}

    return {
        "generator_a": default_gen,
        "generator_b": reviewer_gen,
    }


def _await_council_selection(
    packages: dict[str, InvestigationPackage],
    comparison,
    cfg: ResearchConfig,
) -> str:
    """Pause for explicit operator selection of the canonical generator."""
    choices = sorted(packages)
    recommendation = comparison.recommended_generator or "none"
    try:
        selection = wait(
            schema=str,
            name="select_canonical_generator",
            question=(
                "Select the canonical generator for this council run. "
                f"Choices: {', '.join(choices)}. "
                f"Judge recommendation: {recommendation}."
            ),
            timeout=cfg.wait_timeout_seconds,
            metadata={
                "choices": choices,
                "judge_recommendation": comparison.recommended_generator,
                "comparison": comparison.model_dump(mode="json"),
            },
        )
    except Exception as exc:  # pragma: no cover - exercised by wait stubs/tests
        raise FlowTimeoutError(
            "Timed out waiting for council generator selection after "
            f"{cfg.wait_timeout_seconds} seconds"
        ) from exc

    if selection is None:
        raise FlowTimeoutError(
            "Timed out waiting for council generator selection after "
            f"{cfg.wait_timeout_seconds} seconds"
        )
    if selection not in packages:
        raise CouncilSelectionError(
            f"Invalid council generator selection: {selection!r}. "
            f"Expected one of {choices}."
        )
    return selection


@flow
def council_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
    generator_slots: dict[str, ModelSlotConfig] | None = None,
    output_dir: str | None = None,
) -> CouncilPackage:
    """Run council-mode research with multiple generators."""
    cfg = config or ResearchConfig.for_tier(tier)
    generator_slots = generator_slots or _default_generator_slots(cfg)

    if len(generator_slots) < 2:
        logger.warning(
            "Council mode with %d generator(s) — comparison may be trivial",
            len(generator_slots),
        )

    judge_slot = cfg.slots.get("judge")
    if judge_slot is None:
        raise CouncilConfigError(
            "Council mode requires a 'judge' model slot in config. "
            "Use a tier that includes a judge (standard or deep)."
        )

    compromise = _detect_provider_compromise(generator_slots, judge_slot)
    if compromise:
        logger.warning(
            "Council provider compromise: judge provider '%s' matches a "
            "generator provider. Recording council_provider_compromise=True.",
            judge_slot.provider,
        )

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
        handle = _run_deep_research_pipeline(
            question,
            tier=tier,
            cfg=gen_config,
            output_dir=output_dir,
            require_plan_approval=False,
        )
        packages[gen_name] = handle.load()

    comparison = run_judge(packages, judge_slot.model_string, judge_slot.model_settings)
    canonical_generator = _await_council_selection(packages, comparison, cfg)

    return CouncilPackage(
        schema_version="1.0",
        council_provider_compromise=compromise,
        comparison=comparison,
        packages=packages,
        canonical_generator=canonical_generator,
    )
