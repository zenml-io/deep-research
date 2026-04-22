"""Export checkpoint — materializes a durable package to disk.

NOTE: do NOT add ``from __future__ import annotations`` here. ZenML's
materializer registry inspects the concrete return type of a @checkpoint
at step-registration time and raises if annotations are strings. Same
constraint applies to every module that defines a @checkpoint.
"""

from kitaru import checkpoint

from research.contracts.package import InvestigationPackage
from research.package.export import write_package


@checkpoint(type="tool_call")
def export_package(
    package: InvestigationPackage,
    output_dir: str,
) -> InvestigationPackage:
    """Write *package* to *output_dir* and return it unchanged."""
    write_package(package, output_dir)
    return package
