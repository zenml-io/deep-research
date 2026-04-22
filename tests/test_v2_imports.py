"""Smoke test: V2 research package is importable."""


def test_research_package_importable():
    import research

    assert research.__doc__ == "Deep Research V2 runtime package."


def test_research_subpackages_importable():
    import research.contracts
    import research.config
    import research.flows
    import research.checkpoints
    import research.agents
    import research.ledger
    import research.providers
    import research.package
    import research.prompts
    import research.mcp
