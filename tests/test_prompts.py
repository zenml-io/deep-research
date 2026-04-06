from deep_research.prompts.loader import load_prompt


def test_load_prompt_returns_markdown_contents() -> None:
    prompt = load_prompt("planner")
    assert "research" in prompt.lower()


def test_load_prompt_rejects_unknown_name() -> None:
    try:
        load_prompt("missing")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
