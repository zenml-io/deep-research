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
