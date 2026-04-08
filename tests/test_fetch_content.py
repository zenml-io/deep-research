import importlib
import sys
import types

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import EvidenceCandidate, EvidenceLedger, EvidenceSnippet


def _load_fetch_module(monkeypatch):
    def checkpoint(*, type):
        def decorator(func):
            func._checkpoint_type = type
            return func

        return decorator

    monkeypatch.setitem(
        sys.modules, "kitaru", types.SimpleNamespace(checkpoint=checkpoint)
    )
    sys.modules.pop("deep_research.checkpoints.fetch", None)
    return importlib.import_module("deep_research.checkpoints.fetch")


def test_fetch_content_enriches_canonical_ledger_buckets(monkeypatch) -> None:
    module = _load_fetch_module(monkeypatch)
    selected_candidate = EvidenceCandidate(
        key="candidate-1",
        title="Example",
        url="https://example.com/article",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Search snippet")],
        quality_score=0.7,
        selected=True,
    )
    considered_copy = selected_candidate.model_copy(deep=True)
    selected_copy = selected_candidate.model_copy(deep=True)
    rejected_candidate = EvidenceCandidate(
        key="candidate-2",
        title="Rejected",
        url="https://example.com/rejected",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Rejected snippet")],
        quality_score=0.1,
        selected=False,
    )
    ledger = EvidenceLedger.model_validate(
        {
            "considered": [considered_copy, rejected_candidate.model_copy(deep=True)],
            "selected": [selected_copy],
            "rejected": [rejected_candidate.model_copy(deep=True)],
        }
    )

    monkeypatch.setattr(
        "deep_research.checkpoints.fetch.fetch_url_content",
        lambda url, timeout_sec, max_chars: "Fetched body text",
    )

    enriched = module.fetch_content(ledger, ResearchConfig.for_tier(Tier.STANDARD))

    assert enriched.considered[0].snippets[-1].text == "Fetched body text"
    assert enriched.considered[0].snippets[-1].source_locator == "fetched:body"
    assert enriched.selected[0].snippets[-1].text == "Fetched body text"
    assert enriched.selected[0].snippets[-1].source_locator == "fetched:body"
    assert [snippet.text for snippet in enriched.rejected[0].snippets] == [
        "Rejected snippet"
    ]
    assert module.fetch_content._checkpoint_type == "tool_call"


def test_fetch_content_skips_candidates_already_enriched(monkeypatch) -> None:
    module = _load_fetch_module(monkeypatch)
    candidate = EvidenceCandidate(
        key="candidate-1",
        title="Example",
        url="https://example.com/article",
        provider="brave",
        source_kind="web",
        snippets=[
            EvidenceSnippet(text="Fetched body text", source_locator="fetched:body")
        ],
        quality_score=0.7,
        selected=True,
    )
    ledger = EvidenceLedger.model_validate(
        {"considered": [candidate], "selected": [candidate], "rejected": []}
    )

    calls = []
    monkeypatch.setattr(
        "deep_research.checkpoints.fetch.fetch_url_content",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "should not be used",
    )

    enriched = module.fetch_content(ledger, ResearchConfig.for_tier(Tier.STANDARD))

    assert calls == []
    assert enriched == ledger


def test_fetch_content_preserves_and_enriches_canonical_entry_for_divergent_bucket_copies(
    monkeypatch,
) -> None:
    module = _load_fetch_module(monkeypatch)
    canonical_candidate = EvidenceCandidate(
        key="candidate-1",
        title="Canonical Example",
        url="https://example.com/article",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Canonical snippet")],
        matched_subtopics=["impact"],
        authority_score=0.8,
        quality_score=0.7,
        selected=True,
    )
    stale_selected_copy = EvidenceCandidate(
        key="candidate-1",
        title="Stale Example",
        url="https://example.com/article",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Stale selected snippet")],
        quality_score=0.7,
        selected=True,
    )
    ledger = EvidenceLedger.model_validate(
        {
            "considered": [canonical_candidate],
            "selected": [stale_selected_copy],
            "rejected": [],
        }
    )

    monkeypatch.setattr(
        "deep_research.checkpoints.fetch.fetch_url_content",
        lambda url, timeout_sec, max_chars: "Fetched body text",
    )

    enriched = module.fetch_content(ledger, ResearchConfig.for_tier(Tier.STANDARD))

    assert enriched.considered[0].title == "Canonical Example"
    assert enriched.considered[0].matched_subtopics == ["impact"]
    assert [snippet.text for snippet in enriched.considered[0].snippets] == [
        "Canonical snippet",
        "Fetched body text",
    ]
    assert enriched.selected[0] == enriched.considered[0]


def test_fetch_content_isolates_fetch_failures_per_candidate(monkeypatch) -> None:
    module = _load_fetch_module(monkeypatch)
    failing_candidate = EvidenceCandidate(
        key="candidate-1",
        title="Failing Example",
        url="https://example.com/fail",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Failing snippet")],
        quality_score=0.9,
        selected=True,
    )
    succeeding_candidate = EvidenceCandidate(
        key="candidate-2",
        title="Succeeding Example",
        url="https://example.com/succeed",
        provider="brave",
        source_kind="web",
        snippets=[EvidenceSnippet(text="Succeeding snippet")],
        quality_score=0.8,
        selected=True,
    )
    ledger = EvidenceLedger.model_validate(
        {
            "considered": [failing_candidate, succeeding_candidate],
            "selected": [failing_candidate, succeeding_candidate],
            "rejected": [],
        }
    )

    def fetch(url: str, timeout_sec: int, max_chars: int) -> str:
        del timeout_sec, max_chars
        if url.endswith("/fail"):
            raise RuntimeError("boom")
        return "Fetched success body"

    monkeypatch.setattr("deep_research.checkpoints.fetch.fetch_url_content", fetch)

    enriched = module.fetch_content(ledger, ResearchConfig.for_tier(Tier.STANDARD))

    assert [snippet.text for snippet in enriched.considered[0].snippets] == [
        "Failing snippet"
    ]
    assert [snippet.text for snippet in enriched.considered[1].snippets] == [
        "Succeeding snippet",
        "Fetched success body",
    ]
