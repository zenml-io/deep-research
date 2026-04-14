"""PydanticAI agent factories and Kitaru wrapping infrastructure."""

__all__ = ["wrap_agent"]


def __getattr__(name: str):
    """Lazy import to avoid pulling in kitaru at package-import time.

    This lets ``import research.agents`` succeed even when kitaru is not
    installed (e.g. in the lightweight test environment on Python 3.14).
    The actual import happens on first access of ``wrap_agent``.
    """
    if name == "wrap_agent":
        from research.agents._wrap import wrap_agent

        return wrap_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
