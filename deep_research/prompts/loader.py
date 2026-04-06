from importlib.resources import files


def load_prompt(name: str) -> str:
    """Load a prompt template by name from the prompts package directory."""
    path = files("deep_research.prompts").joinpath(f"{name}.md")
    return path.read_text(encoding="utf-8")
