"""Run a V2 deep research investigation from the command line.

Usage:
    uv run python run_v2.py "What are the latest advances in RLHF alternatives?"
    uv run python run_v2.py --tier deep "My research question"
    uv run python run_v2.py --output ./results "My research question"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deep research investigation (V2)."
    )
    parser.add_argument("question", help="The research question to investigate.")
    parser.add_argument(
        "--tier",
        default="standard",
        choices=["quick", "standard", "deep", "exhaustive"],
        help="Research tier (default: standard)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for the investigation package (default: artifacts/)",
    )
    parser.add_argument(
        "--enable-sandbox",
        action="store_true",
        help="Enable code execution sandbox",
    )
    parser.add_argument(
        "--allow-unfinalized",
        action="store_true",
        help="Allow producing a package even if the finalizer fails",
    )
    parser.add_argument(
        "--council",
        action="store_true",
        help="Run in council mode (multiple generators)",
    )
    args = parser.parse_args()

    # Lazy imports to keep module fast for testing
    from research.config.settings import ResearchConfig
    from research.flows.deep_research import deep_research
    from research.package.export import write_package

    # Build config from tier defaults
    cfg = ResearchConfig.for_tier(args.tier)

    # Apply CLI overrides via model_copy (ResearchConfig is frozen)
    overrides: dict[str, object] = {}
    if args.enable_sandbox:
        overrides["sandbox_enabled"] = True
    if args.allow_unfinalized:
        overrides["allow_unfinalized_package"] = True
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    if args.council:
        print("Council mode is not yet implemented.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting research: {args.question}")
    print(f"Tier: {args.tier}")
    print()

    package = deep_research.run(args.question, tier=args.tier, config=cfg).wait()

    out_path = write_package(package, args.output)
    print(f"\nPackage written to: {out_path}")

    # Print summary
    print(f"\n--- Run Summary ---")
    print(f"Run ID: {package.metadata.run_id}")
    print(f"Tier: {package.metadata.tier}")
    print(f"Iterations: {package.metadata.total_iterations}")
    print(f"Cost: ${package.metadata.total_cost_usd:.4f}")
    if package.metadata.stop_reason:
        print(f"Stop reason: {package.metadata.stop_reason}")

    if package.final_report:
        preview = package.final_report.content[:500]
        print(f"\n--- Report Preview (first 500 chars) ---")
        print(preview)
    elif package.draft:
        preview = package.draft.content[:500]
        print(f"\n--- Draft Preview (first 500 chars) ---")
        print(preview)


if __name__ == "__main__":
    main()
