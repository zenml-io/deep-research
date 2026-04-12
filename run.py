"""Run a deep research investigation from the command line.

Usage:
    uv run python run.py "What are the latest advances in RLHF alternatives?"
    uv run python run.py --tier standard "My research question"
    uv run python run.py --output ./results "My research question"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_research.flow.research_flow import research_flow
from deep_research.package.io import write_package


def _extract_package(handle):
    """Extract InvestigationPackage from a completed FlowHandle.

    The local orchestrator runs synchronously inside .run(), so the flow is
    already finished when the handle is returned.  The .wait() method fails
    because ZenML sees multiple terminal steps.  Instead we load the result
    directly from the assemble_package step artifact.
    """
    run = handle._run.get_hydrated_version()
    assemble_step = run.steps.get("assemble_package")
    if assemble_step is None:
        raise RuntimeError("Flow completed but assemble_package step not found.")

    output_name = next(iter(assemble_step.regular_outputs))
    return assemble_step.regular_outputs[output_name].load()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deep research investigation.")
    parser.add_argument("brief", help="The research question or brief.")
    parser.add_argument(
        "--tier",
        default="auto",
        help="Research tier: auto, quick, standard, deep (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for the investigation package (default: artifacts/)",
    )
    args = parser.parse_args()

    print(f"Starting research: {args.brief}")
    print(f"Tier: {args.tier}")
    print()

    handle = research_flow.run(args.brief, tier=args.tier)
    package = _extract_package(handle)

    out_path = write_package(package, args.output)
    print(f"\nPackage written to: {out_path}")

    print("\n--- Run Summary ---")
    print(package.run_summary)

    if package.renders:
        print("\n--- Report Preview (first 500 chars) ---")
        print(package.renders[0].content_markdown[:500])


if __name__ == "__main__":
    main()
