from deep_research.evidence.dedup import dedupe_candidates
from deep_research.evidence.ledger import merge_candidates
from deep_research.evidence.scoring import score_candidate_quality
from deep_research.models import EvidenceCandidate, EvidenceSnippet
from deep_research.providers.normalization import normalize_tool_results


def test_normalize_tool_results_maps_dict_payloads() -> None:
    raw = [{"title": "Doc", "url": "https://example.com", "snippet": "Alpha"}]

    candidates = normalize_tool_results(raw, provider="brave", source_kind="web")

    assert candidates[0].title == "Doc"
    assert candidates[0].snippets[0].text == "Alpha"


def test_dedupe_candidates_prefers_url_identity() -> None:
    first = EvidenceCandidate(
        key="1",
        title="A",
        url="https://x.example",
        provider="b",
        source_kind="web",
    )
    second = EvidenceCandidate(
        key="2",
        title="A2",
        url="https://x.example",
        provider="c",
        source_kind="web",
    )

    deduped = dedupe_candidates([first, second])

    assert len(deduped) == 1


def test_merge_candidates_preserves_existing_and_appends_new() -> None:
    ledger = merge_candidates(
        [],
        [
            EvidenceCandidate(
                key="1",
                title="A",
                url="https://x.example",
                provider="b",
                source_kind="web",
            )
        ],
    )

    assert len(ledger.entries) == 1


def test_score_candidate_quality_returns_floorable_value() -> None:
    candidate = EvidenceCandidate(
        key="1",
        title="A",
        url="https://x.example",
        provider="b",
        source_kind="paper",
    )

    assert 0.0 <= score_candidate_quality(candidate) <= 1.0


def test_normalize_tool_results_uses_description_as_snippet_and_locator() -> None:
    raw = [
        {
            "title": "Guide",
            "url": "https://docs.example/guide",
            "description": "Beta",
            "source_locator": "#intro",
        }
    ]

    candidates = normalize_tool_results(raw, provider="docs", source_kind="docs")

    assert candidates[0].snippets == [
        EvidenceSnippet(text="Beta", source_locator="#intro")
    ]


def test_merge_candidates_keeps_first_duplicate_and_adds_unique_entries() -> None:
    existing = [
        EvidenceCandidate(
            key="1",
            title="Existing",
            url="https://x.example",
            provider="b",
            source_kind="web",
        )
    ]
    incoming = [
        EvidenceCandidate(
            key="2",
            title="Duplicate",
            url="https://x.example",
            provider="c",
            source_kind="docs",
        ),
        EvidenceCandidate(
            key="3",
            title="New",
            url="https://y.example",
            provider="c",
            source_kind="paper",
        ),
    ]

    ledger = merge_candidates(existing, incoming)

    assert [entry.key for entry in ledger.entries] == ["1", "3"]


def test_score_candidate_quality_assigns_expected_defaults() -> None:
    docs_candidate = EvidenceCandidate(
        key="docs-1",
        title="Docs",
        url="https://docs.example/reference",
        provider="docs",
        source_kind="docs",
    )
    web_candidate = EvidenceCandidate(
        key="web-1",
        title="Web",
        url="https://web.example/article",
        provider="search",
        source_kind="web",
    )
    other_candidate = EvidenceCandidate(
        key="other-1",
        title="Other",
        url="https://other.example/item",
        provider="api",
        source_kind="dataset",
    )

    assert score_candidate_quality(docs_candidate) == 0.8
    assert score_candidate_quality(web_candidate) == 0.6
    assert score_candidate_quality(other_candidate) == 0.4
