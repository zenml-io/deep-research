from deep_research.config import ResearchConfig
from deep_research.evidence.dedup import dedupe_candidates
from deep_research.evidence.ledger import merge_candidates
from deep_research.evidence.scoring import score_candidate_quality
from deep_research.enums import Tier
from deep_research.models import (
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
    RawToolResult,
)
from deep_research.providers.normalization import normalize_tool_results


def test_normalize_tool_results_maps_dict_payloads() -> None:
    raw = [{"title": "Doc", "url": "https://example.com", "snippet": "Alpha"}]

    candidates = normalize_tool_results(raw, provider="brave", source_kind="web")

    assert candidates[0].title == "Doc"
    assert candidates[0].snippets[0].text == "Alpha"


def test_normalize_tool_results_extracts_nested_raw_tool_result_payloads() -> None:
    raw_results = [
        RawToolResult(
            tool_name="search",
            provider="brave",
            payload={
                "results": [
                    {
                        "title": "Nested Doc",
                        "url": "https://nested.example/doc",
                        "snippet": "Nested snippet",
                    }
                ]
            },
        ),
        RawToolResult(
            tool_name="search",
            provider="brave",
            payload={
                "items": [
                    {
                        "title": "Item Doc",
                        "url": "https://nested.example/item",
                        "description": "Item snippet",
                    }
                ]
            },
        ),
    ]

    candidates = normalize_tool_results(
        raw_results, provider="brave", source_kind="web"
    )

    assert [candidate.title for candidate in candidates] == ["Nested Doc", "Item Doc"]
    assert [candidate.snippets[0].text for candidate in candidates] == [
        "Nested snippet",
        "Item snippet",
    ]


def test_normalize_tool_results_generates_stable_provider_url_keys() -> None:
    first_batch = normalize_tool_results(
        [{"title": "One", "url": "https://keys.example/one"}],
        provider="brave",
        source_kind="web",
    )
    second_batch = normalize_tool_results(
        [{"title": "Two", "url": "https://keys.example/two"}],
        provider="brave",
        source_kind="web",
    )
    repeated_batch = normalize_tool_results(
        [{"title": "One again", "url": "https://keys.example/one"}],
        provider="brave",
        source_kind="web",
    )

    assert first_batch[0].key != second_batch[0].key
    assert first_batch[0].key == repeated_batch[0].key


def test_normalize_tool_results_uses_canonicalized_url_for_key_identity() -> None:
    upper_host = normalize_tool_results(
        [{"title": "One", "url": "https://EXAMPLE.com/one"}],
        provider="brave",
        source_kind="web",
    )
    lower_host = normalize_tool_results(
        [{"title": "One", "url": "https://example.com/one"}],
        provider="brave",
        source_kind="web",
    )

    assert str(upper_host[0].url) == str(lower_host[0].url)
    assert upper_host[0].key == lower_host[0].key


def test_normalize_tool_results_captures_identifiers_and_metadata() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Replay Paper",
                "url": "https://example.com/paper",
                "snippet": "Foundational replay paper",
                "doi": "10.1000/replay",
                "matched_subtopics": ["replay"],
                "authority_score": 0.9,
                "freshness_score": 0.3,
                "provider_id": "paper-1",
                "source_locator": "sec-2",
            }
        ],
        provider="arxiv",
        source_kind="paper",
    )

    assert candidates[0].doi == "10.1000/replay"
    assert candidates[0].matched_subtopics == ["replay"]
    assert candidates[0].authority_score == 0.9
    assert candidates[0].freshness_score == 0.3
    assert candidates[0].raw_metadata == {
        "provider_id": "paper-1",
        "source_locator": "sec-2",
    }


def test_normalize_tool_results_uses_identifier_precedence_for_keys() -> None:
    doi_batch = normalize_tool_results(
        [{"title": "Replay", "url": "https://one.example", "doi": "10.1000/replay"}],
        provider="arxiv",
        source_kind="paper",
    )
    doi_repeat = normalize_tool_results(
        [
            {
                "title": "Replay copy",
                "url": "https://two.example",
                "doi": "10.1000/replay",
            }
        ],
        provider="crossref",
        source_kind="paper",
    )
    url_batch = normalize_tool_results(
        [{"title": "Guide", "url": "https://EXAMPLE.com/guide"}],
        provider="docs",
        source_kind="docs",
    )
    url_repeat = normalize_tool_results(
        [{"title": "Guide duplicate", "url": "https://example.com/guide"}],
        provider="web",
        source_kind="web",
    )

    assert doi_batch[0].key == doi_repeat[0].key
    assert url_batch[0].key == url_repeat[0].key


def test_normalize_tool_results_ignores_container_identifier_values_for_identity() -> (
    None
):
    candidates = normalize_tool_results(
        [
            {
                "title": "Doc One",
                "url": "https://example.com/one",
                "doi": {},
            },
            {
                "title": "Doc Two",
                "url": "https://example.com/two",
                "doi": {},
            },
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.key for candidate in candidates] == [
        candidates[0].key,
        candidates[1].key,
    ]
    assert candidates[0].key != candidates[1].key
    assert candidates[0].doi is None
    assert candidates[1].doi is None


def test_normalize_tool_results_treats_bare_origin_and_trailing_slash_as_same_identity() -> (
    None
):
    bare_origin = normalize_tool_results(
        [{"title": "One", "url": "https://example.com"}],
        provider="docs",
        source_kind="docs",
    )
    trailing_slash = normalize_tool_results(
        [{"title": "One slash", "url": "https://example.com/"}],
        provider="web",
        source_kind="web",
    )

    assert bare_origin[0].key == trailing_slash[0].key


def test_normalize_tool_results_treats_none_subtopics_as_empty() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Replay Paper",
                "url": "https://example.com/paper",
                "matched_subtopics": None,
            }
        ],
        provider="arxiv",
        source_kind="paper",
    )

    assert candidates[0].matched_subtopics == []


def test_normalize_tool_results_trims_and_filters_subtopics() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Replay Paper",
                "url": "https://example.com/paper",
                "matched_subtopics": [" replay ", None, ""],
            }
        ],
        provider="arxiv",
        source_kind="paper",
    )

    assert candidates[0].matched_subtopics == ["replay"]


def test_normalize_tool_results_collapses_default_https_ports_in_keys() -> None:
    canonical = normalize_tool_results(
        [{"title": "Guide", "url": "https://example.com/path"}],
        provider="docs",
        source_kind="docs",
    )
    explicit_default_port = normalize_tool_results(
        [{"title": "Guide", "url": "https://example.com:443/path"}],
        provider="web",
        source_kind="web",
    )

    assert canonical[0].key == explicit_default_port[0].key


def test_normalize_tool_results_treats_scalar_subtopic_as_single_topic() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Replay Paper",
                "url": "https://example.com/paper",
                "matched_subtopics": "replay",
            }
        ],
        provider="arxiv",
        source_kind="paper",
    )

    assert candidates[0].matched_subtopics == ["replay"]


def test_normalize_tool_results_handles_invalid_explicit_url_port_defensively() -> None:
    candidates = normalize_tool_results(
        [{"title": "Broken URL", "url": "https://example.com:bad/path"}],
        provider="web",
        source_kind="web",
    )

    assert candidates == []


def test_normalize_tool_results_skips_malformed_urls_without_aborting_batch() -> None:
    candidates = normalize_tool_results(
        [
            {"title": "Broken URL", "url": "notaurl"},
            {"title": "Valid URL", "url": "https://example.com/valid"},
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.title for candidate in candidates] == ["Valid URL"]


def test_normalize_tool_results_skips_non_mapping_top_level_rows_without_aborting_batch() -> (
    None
):
    candidates = normalize_tool_results(
        [
            "bad-row",
            {"title": "Valid URL", "url": "https://example.com/valid"},
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.title for candidate in candidates] == ["Valid URL"]


def test_normalize_tool_results_skips_malformed_numeric_scores_without_aborting_batch() -> (
    None
):
    candidates = normalize_tool_results(
        [
            {
                "title": "Broken Score",
                "url": "https://example.com/broken-score",
                "quality_score": "not-a-number",
            },
            {
                "title": "Valid URL",
                "url": "https://example.com/valid",
                "quality_score": 0.8,
            },
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.title for candidate in candidates] == ["Valid URL"]


def test_normalize_tool_results_skips_container_scores_without_aborting_batch() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Broken Score",
                "url": "https://example.com/broken-score",
                "quality_score": {},
            },
            {
                "title": "Valid URL",
                "url": "https://example.com/valid",
                "quality_score": 0.8,
            },
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.title for candidate in candidates] == ["Valid URL"]


def test_normalize_tool_results_treats_malformed_scalar_subtopics_as_empty() -> None:
    candidates = normalize_tool_results(
        [
            {
                "title": "Malformed Topics",
                "url": "https://example.com/bad-topics",
                "matched_subtopics": 1,
            },
            {
                "title": "Valid URL",
                "url": "https://example.com/valid",
                "matched_subtopics": ["replay"],
            },
        ],
        provider="web",
        source_kind="web",
    )

    assert [candidate.title for candidate in candidates] == [
        "Malformed Topics",
        "Valid URL",
    ]
    assert candidates[0].matched_subtopics == []
    assert candidates[1].matched_subtopics == ["replay"]


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

    deduped, log = dedupe_candidates([first, second])

    assert len(deduped) == 1
    assert [event.match_basis for event in log] == ["canonical_url"]


def test_dedupe_candidates_preserves_url_case_identity() -> None:
    first = EvidenceCandidate(
        key="1",
        title="A",
        url="https://x.example/CaseSensitive",
        provider="b",
        source_kind="web",
    )
    second = EvidenceCandidate(
        key="2",
        title="A2",
        url="https://x.example/casesensitive",
        provider="c",
        source_kind="web",
    )

    deduped, log = dedupe_candidates([first, second])

    assert [candidate.key for candidate in deduped] == ["1", "2"]
    assert log == []


def test_dedupe_candidates_prefers_doi_then_arxiv_then_canonical_url_then_title() -> (
    None
):
    doi_first = EvidenceCandidate(
        key="doi-1",
        title="Replay",
        url="https://a.example/replay",
        provider="arxiv",
        source_kind="paper",
        doi="10.1000/replay",
    )
    doi_second = EvidenceCandidate(
        key="doi-2",
        title="Replay duplicate",
        url="https://b.example/replay",
        provider="crossref",
        source_kind="paper",
        doi="10.1000/replay",
    )
    arxiv_first = EvidenceCandidate(
        key="arxiv-1",
        title="Other",
        url="https://a.example/arxiv",
        provider="arxiv",
        source_kind="paper",
        arxiv_id="1234.5678",
    )
    arxiv_second = EvidenceCandidate(
        key="arxiv-2",
        title="Other duplicate",
        url="https://b.example/arxiv",
        provider="mirror",
        source_kind="paper",
        arxiv_id="1234.5678",
    )
    url_first = EvidenceCandidate(
        key="url-1",
        title="Guide",
        url="https://EXAMPLE.com/guide",
        provider="docs",
        source_kind="docs",
    )
    url_second = EvidenceCandidate(
        key="url-2",
        title="Guide duplicate",
        url="https://example.com/guide",
        provider="web",
        source_kind="web",
    )
    title_first = EvidenceCandidate.model_construct(
        key="title-1",
        title="Fallback Title",
        url="",
        provider="docs",
        source_kind="docs",
    )
    title_second = EvidenceCandidate.model_construct(
        key="title-2",
        title="Fallback Title",
        url="",
        provider="web",
        source_kind="web",
    )

    deduped, log = dedupe_candidates(
        [
            doi_first,
            doi_second,
            arxiv_first,
            arxiv_second,
            url_first,
            url_second,
            title_first,
            title_second,
        ]
    )

    assert [candidate.key for candidate in deduped] == [
        "doi-1",
        "arxiv-1",
        "url-1",
        "title-1",
    ]
    assert [event.match_basis for event in log] == [
        "doi",
        "arxiv_id",
        "canonical_url",
        "title",
    ]


def test_dedupe_candidates_collapses_default_https_port_for_canonical_url() -> None:
    first = EvidenceCandidate(
        key="1",
        title="Guide",
        url="https://example.com/path",
        provider="docs",
        source_kind="docs",
    )
    second = EvidenceCandidate(
        key="2",
        title="Guide duplicate",
        url="https://example.com:443/path",
        provider="web",
        source_kind="web",
    )

    deduped, log = dedupe_candidates([first, second])

    assert [candidate.key for candidate in deduped] == ["1"]
    assert [event.match_basis for event in log] == ["canonical_url"]


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


def test_merge_candidates_cleans_existing_duplicates_even_without_incoming() -> None:
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
            doi="10.1000/replay",
            quality_score=0.8,
        ),
        EvidenceCandidate(
            key="candidate-2",
            title="Replay duplicate",
            url="https://example.com/replay",
            provider="web",
            source_kind="web",
            quality_score=0.2,
        ),
    ]

    ledger = merge_candidates(existing, [], quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == ["candidate-1"]
    assert [event.duplicate_key for event in ledger.dedupe_log] == ["candidate-2"]


def test_merge_candidates_ratchets_existing_duplicates_even_without_incoming() -> None:
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
            doi="10.1000/replay",
            quality_score=0.2,
            relevance_score=0.1,
            snippets=[EvidenceSnippet(text="Initial snippet")],
        ),
        EvidenceCandidate(
            key="candidate-2",
            title="Replay duplicate",
            url="https://example.com/replay",
            provider="paper",
            source_kind="paper",
            doi="10.1000/replay",
            quality_score=0.9,
            relevance_score=0.8,
            authority_score=0.95,
            matched_subtopics=["replay"],
            snippets=[EvidenceSnippet(text="Stronger snippet")],
        ),
    ]

    ledger = merge_candidates(existing, [], quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == ["candidate-1"]
    assert ledger.considered[0].quality_score == 0.9
    assert ledger.considered[0].relevance_score == 0.8
    assert ledger.considered[0].authority_score == 0.95
    assert ledger.considered[0].matched_subtopics == ["replay"]
    assert [snippet.text for snippet in ledger.considered[0].snippets] == [
        "Initial snippet",
        "Stronger snippet",
    ]


def test_merge_candidates_ratchets_scores_and_records_rejected_items() -> None:
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
            doi="10.1000/replay",
            quality_score=0.7,
            relevance_score=0.6,
            authority_score=0.5,
            snippets=[EvidenceSnippet(text="Initial snippet")],
        )
    ]
    incoming = [
        EvidenceCandidate(
            key="candidate-2",
            title="Replay Duplicate",
            url="https://mirror.example/replay",
            provider="arxiv",
            source_kind="paper",
            doi="10.1000/replay",
            quality_score=0.9,
            relevance_score=0.8,
            authority_score=0.95,
            matched_subtopics=["replay"],
            snippets=[EvidenceSnippet(text="New snippet")],
        ),
        EvidenceCandidate(
            key="candidate-3",
            title="Weak source",
            url="https://weak.example",
            provider="blog",
            source_kind="web",
            quality_score=0.1,
            relevance_score=0.2,
        ),
    ]

    ledger = merge_candidates(existing, incoming, quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == [
        "candidate-1",
        "candidate-3",
    ]
    assert ledger.considered[0].quality_score == 0.9
    assert ledger.considered[0].relevance_score == 0.8
    assert ledger.considered[0].authority_score == 0.95
    assert ledger.considered[0].matched_subtopics == ["replay"]
    assert [snippet.text for snippet in ledger.considered[0].snippets] == [
        "Initial snippet",
        "New snippet",
    ]
    assert [candidate.key for candidate in ledger.selected] == ["candidate-1"]
    assert ledger.selected[0].selected is True
    assert [candidate.key for candidate in ledger.rejected] == ["candidate-3"]
    assert ledger.rejected[0].selected is False
    assert ledger.dedupe_log[0].canonical_key == "candidate-1"
    assert ledger.entries == ledger.considered


def test_merge_candidates_reconciles_later_stronger_identifier_with_existing_url_match() -> (
    None
):
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
            quality_score=0.4,
            relevance_score=0.3,
        )
    ]
    incoming = [
        EvidenceCandidate(
            key="candidate-2",
            title="Replay paper",
            url="https://example.com/replay",
            provider="arxiv",
            source_kind="paper",
            doi="10.1000/replay",
            quality_score=0.9,
            relevance_score=0.8,
            authority_score=0.95,
        )
    ]

    ledger = merge_candidates(existing, incoming, quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == ["candidate-1"]
    assert ledger.considered[0].doi == "10.1000/replay"
    assert ledger.considered[0].quality_score == 0.9
    assert [event.duplicate_key for event in ledger.dedupe_log] == ["candidate-2"]
    assert [event.canonical_key for event in ledger.dedupe_log] == ["candidate-1"]


def test_merge_candidates_keeps_unique_incoming_when_existing_already_contains_duplicates() -> (
    None
):
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
        ),
        EvidenceCandidate(
            key="candidate-2",
            title="Replay duplicate",
            url="https://example.com/replay",
            provider="web",
            source_kind="web",
        ),
    ]
    incoming = [
        EvidenceCandidate(
            key="candidate-3",
            title="Unique",
            url="https://example.com/unique",
            provider="arxiv",
            source_kind="paper",
            quality_score=0.8,
            relevance_score=0.7,
        )
    ]

    ledger = merge_candidates(existing, incoming, quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == [
        "candidate-1",
        "candidate-3",
    ]
    assert [event.duplicate_key for event in ledger.dedupe_log] == ["candidate-2"]


def test_merge_candidates_sets_considered_selected_flags_from_quality_floor() -> None:
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Strong",
            url="https://example.com/strong",
            provider="docs",
            source_kind="docs",
            quality_score=0.8,
            relevance_score=0.7,
        ),
        EvidenceCandidate(
            key="candidate-2",
            title="Weak",
            url="https://example.com/weak",
            provider="web",
            source_kind="web",
            quality_score=0.1,
            relevance_score=0.2,
            selected=True,
        ),
    ]

    ledger = merge_candidates(existing, [], quality_floor=0.3)

    assert [(candidate.key, candidate.selected) for candidate in ledger.considered] == [
        ("candidate-1", True),
        ("candidate-2", False),
    ]
    assert [candidate.key for candidate in ledger.selected] == ["candidate-1"]
    assert [candidate.key for candidate in ledger.rejected] == ["candidate-2"]


def test_merge_candidates_cleans_existing_duplicates_when_incoming_is_also_duplicate() -> (
    None
):
    existing = [
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="docs",
            source_kind="docs",
            doi="10.1000/replay",
            quality_score=0.5,
        ),
        EvidenceCandidate(
            key="candidate-2",
            title="Replay duplicate",
            url="https://example.com/replay",
            provider="web",
            source_kind="web",
            quality_score=0.3,
        ),
    ]
    incoming = [
        EvidenceCandidate(
            key="candidate-3",
            title="Replay third copy",
            url="https://example.com/replay",
            provider="arxiv",
            source_kind="paper",
            doi="10.1000/replay",
            quality_score=0.9,
            relevance_score=0.8,
            authority_score=0.95,
        )
    ]

    ledger = merge_candidates(existing, incoming, quality_floor=0.3)

    assert [candidate.key for candidate in ledger.considered] == ["candidate-1"]
    assert ledger.considered[0].quality_score == 0.9
    assert [event.duplicate_key for event in ledger.dedupe_log] == [
        "candidate-2",
        "candidate-3",
    ]


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
