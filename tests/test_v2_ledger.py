"""Tests for v2 evidence ledger: canonical ID resolution and URL canonicalization."""

from __future__ import annotations

import pytest

from research.ledger import (
    canonicalize_url,
    extract_canonical_id,
    strip_tracking_params,
)
from research.ledger.canonical import _normalize_arxiv_id, _normalize_doi


# ---------------------------------------------------------------------------
# DOI normalization
# ---------------------------------------------------------------------------


class TestDOINormalization:
    def test_bare_doi(self):
        assert _normalize_doi("10.1234/foo") == "10.1234/foo"

    def test_doi_with_https_prefix(self):
        assert _normalize_doi("https://doi.org/10.1234/foo") == "10.1234/foo"

    def test_doi_with_http_prefix(self):
        assert _normalize_doi("http://doi.org/10.1234/foo") == "10.1234/foo"

    def test_doi_with_dx_prefix(self):
        assert _normalize_doi("https://dx.doi.org/10.1234/foo") == "10.1234/foo"

    def test_doi_lowercased(self):
        assert _normalize_doi("10.1234/FOO.Bar") == "10.1234/foo.bar"

    def test_doi_whitespace_stripped(self):
        assert _normalize_doi("  10.1234/foo  ") == "10.1234/foo"


# ---------------------------------------------------------------------------
# arXiv ID normalization
# ---------------------------------------------------------------------------


class TestArxivNormalization:
    def test_bare_id_no_version(self):
        assert _normalize_arxiv_id("2301.12345") == "2301.12345"

    def test_bare_id_with_version(self):
        assert _normalize_arxiv_id("2301.12345v1") == "2301.12345"

    def test_bare_id_with_v2(self):
        assert _normalize_arxiv_id("2301.12345v2") == "2301.12345"

    def test_abs_url(self):
        assert _normalize_arxiv_id("https://arxiv.org/abs/2301.12345v2") == "2301.12345"

    def test_pdf_url(self):
        assert _normalize_arxiv_id("https://arxiv.org/pdf/2301.12345v1") == "2301.12345"

    def test_http_url(self):
        assert _normalize_arxiv_id("http://arxiv.org/abs/2301.12345") == "2301.12345"

    def test_whitespace_stripped(self):
        assert _normalize_arxiv_id("  2301.12345v1  ") == "2301.12345"


# ---------------------------------------------------------------------------
# extract_canonical_id
# ---------------------------------------------------------------------------


class TestExtractCanonicalId:
    """Test the main canonical ID extraction with precedence logic."""

    def test_doi_from_full_url(self):
        result = extract_canonical_id(doi="https://doi.org/10.1234/foo")
        assert result == ("doi", "10.1234/foo")

    def test_doi_from_bare_doi(self):
        result = extract_canonical_id(doi="10.1234/foo")
        assert result == ("doi", "10.1234/foo")

    def test_arxiv_from_abs_url(self):
        result = extract_canonical_id(arxiv_id="https://arxiv.org/abs/2301.12345v2")
        assert result == ("arxiv", "2301.12345")

    def test_arxiv_without_url_prefix(self):
        result = extract_canonical_id(arxiv_id="2301.12345v1")
        assert result == ("arxiv", "2301.12345")

    def test_url_fallback(self):
        result = extract_canonical_id(
            url="https://example.com/paper?utm_source=twitter"
        )
        assert result == ("url", "https://example.com/paper")

    def test_no_identifiers(self):
        result = extract_canonical_id()
        assert result == ("none", "")

    def test_none_identifiers(self):
        result = extract_canonical_id(doi=None, arxiv_id=None, url=None)
        assert result == ("none", "")

    def test_empty_string_identifiers(self):
        result = extract_canonical_id(doi="", arxiv_id="", url="")
        assert result == ("none", "")

    def test_whitespace_only_identifiers(self):
        result = extract_canonical_id(doi="  ", arxiv_id="  ", url="  ")
        assert result == ("none", "")

    def test_precedence_doi_over_arxiv(self):
        result = extract_canonical_id(doi="10.1234/foo", arxiv_id="2301.12345")
        assert result == ("doi", "10.1234/foo")

    def test_precedence_doi_over_url(self):
        result = extract_canonical_id(
            doi="10.1234/foo", url="https://example.com/paper"
        )
        assert result == ("doi", "10.1234/foo")

    def test_precedence_arxiv_over_url(self):
        result = extract_canonical_id(
            arxiv_id="2301.12345", url="https://example.com/paper"
        )
        assert result == ("arxiv", "2301.12345")

    def test_precedence_all_present(self):
        """DOI wins when all three identifiers are provided."""
        result = extract_canonical_id(
            doi="10.1234/foo",
            arxiv_id="2301.12345",
            url="https://example.com/paper",
        )
        assert result == ("doi", "10.1234/foo")

    def test_url_fallback_preserves_raw_when_canonicalization_fails(self):
        """When URL has no scheme, canonicalize_url returns empty, so raw URL is kept."""
        result = extract_canonical_id(url="just-a-path/thing")
        assert result == ("url", "just-a-path/thing")


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalizeUrl:
    """Test URL canonicalization: tracking params, trailing slash, case, fragments."""

    def test_strips_utm_source(self):
        result = canonicalize_url("https://example.com/page?utm_source=twitter")
        assert result == "https://example.com/page"

    def test_strips_utm_medium(self):
        result = canonicalize_url("https://example.com/page?utm_medium=email")
        assert result == "https://example.com/page"

    def test_strips_utm_campaign(self):
        result = canonicalize_url("https://example.com/page?utm_campaign=summer")
        assert result == "https://example.com/page"

    def test_strips_utm_term(self):
        result = canonicalize_url("https://example.com/page?utm_term=keyword")
        assert result == "https://example.com/page"

    def test_strips_utm_content(self):
        result = canonicalize_url("https://example.com/page?utm_content=cta")
        assert result == "https://example.com/page"

    def test_strips_fbclid(self):
        result = canonicalize_url("https://example.com/page?fbclid=abc123")
        assert result == "https://example.com/page"

    def test_strips_gclid(self):
        result = canonicalize_url("https://example.com/page?gclid=abc123")
        assert result == "https://example.com/page"

    def test_strips_ref(self):
        result = canonicalize_url("https://example.com/page?ref=homepage")
        assert result == "https://example.com/page"

    def test_strips_source(self):
        result = canonicalize_url("https://example.com/page?source=nav")
        assert result == "https://example.com/page"

    def test_strips_multiple_tracking_params(self):
        result = canonicalize_url(
            "https://example.com/page?utm_source=twitter&utm_medium=email&id=42"
        )
        assert result == "https://example.com/page?id=42"

    def test_preserves_non_tracking_params(self):
        result = canonicalize_url("https://example.com/search?q=test&page=2")
        assert "q=test" in result
        assert "page=2" in result

    def test_strips_trailing_slash(self):
        result = canonicalize_url("https://example.com/page/")
        assert result == "https://example.com/page"

    def test_keeps_trailing_slash_on_domain_root(self):
        result = canonicalize_url("https://example.com/")
        assert result == "https://example.com/"

    def test_lowercases_hostname(self):
        result = canonicalize_url("https://EXAMPLE.COM/Page/Path")
        assert result == "https://example.com/Page/Path"

    def test_lowercases_scheme(self):
        result = canonicalize_url("HTTPS://example.com/page")
        assert result == "https://example.com/page"

    def test_preserves_path_case(self):
        result = canonicalize_url("https://example.com/CamelCase/Path")
        assert "/CamelCase/Path" in result

    def test_strips_fragment(self):
        result = canonicalize_url("https://example.com/page#section-1")
        assert result == "https://example.com/page"

    def test_strips_fragment_with_query(self):
        result = canonicalize_url("https://example.com/page?id=42#section-1")
        assert result == "https://example.com/page?id=42"

    def test_empty_url(self):
        assert canonicalize_url("") == ""

    def test_none_url(self):
        # The function signature accepts str but callers may pass None-like empty
        assert canonicalize_url("") == ""

    def test_whitespace_only_url(self):
        assert canonicalize_url("   ") == ""

    def test_no_scheme_returns_empty(self):
        assert canonicalize_url("example.com/page") == ""

    def test_no_netloc_returns_empty(self):
        assert canonicalize_url("/just/a/path") == ""

    def test_strips_default_http_port(self):
        result = canonicalize_url("http://example.com:80/page")
        assert result == "http://example.com/page"

    def test_strips_default_https_port(self):
        result = canonicalize_url("https://example.com:443/page")
        assert result == "https://example.com/page"

    def test_keeps_non_default_port(self):
        result = canonicalize_url("https://example.com:8080/page")
        assert result == "https://example.com:8080/page"

    def test_complex_url(self):
        """A real-world-ish URL with tracking params, fragment, and trailing slash."""
        result = canonicalize_url(
            "HTTPS://WWW.Example.COM/Research/Paper/?utm_source=google&id=123&fbclid=xyz#abstract"
        )
        assert result == "https://www.example.com/Research/Paper?id=123"


# ---------------------------------------------------------------------------
# strip_tracking_params (standalone)
# ---------------------------------------------------------------------------


class TestStripTrackingParams:
    def test_strips_utm_params(self):
        result = strip_tracking_params(
            "https://example.com/page?utm_source=a&utm_medium=b"
        )
        assert "utm_source" not in result
        assert "utm_medium" not in result

    def test_preserves_other_params(self):
        result = strip_tracking_params("https://example.com/page?q=test&utm_source=a")
        assert "q=test" in result
        assert "utm_source" not in result

    def test_no_query_unchanged(self):
        url = "https://example.com/page"
        assert strip_tracking_params(url) == url

    def test_empty_string(self):
        assert strip_tracking_params("") == ""

    def test_only_tracking_params(self):
        result = strip_tracking_params("https://example.com/page?utm_source=a&fbclid=b")
        assert result == "https://example.com/page"


# ---------------------------------------------------------------------------
# Module re-exports
# ---------------------------------------------------------------------------


class TestModuleReExports:
    """Verify that ledger __init__.py re-exports correctly."""

    def test_extract_canonical_id_importable(self):
        from research.ledger import extract_canonical_id as fn

        assert callable(fn)

    def test_canonicalize_url_importable(self):
        from research.ledger import canonicalize_url as fn

        assert callable(fn)

    def test_strip_tracking_params_importable(self):
        from research.ledger import strip_tracking_params as fn

        assert callable(fn)

    def test_compute_dedup_key_importable(self):
        from research.ledger import compute_dedup_key as fn

        assert callable(fn)

    def test_is_duplicate_importable(self):
        from research.ledger import is_duplicate as fn

        assert callable(fn)

    def test_managed_ledger_importable(self):
        from research.ledger import ManagedLedger as cls

        assert callable(cls)


# ---------------------------------------------------------------------------
# Dedup functions
# ---------------------------------------------------------------------------

from research.contracts.evidence import EvidenceItem
from research.ledger.dedup import compute_dedup_key, is_duplicate


def _make_item(
    evidence_id: str = "ev_test",
    title: str = "Test",
    synthesis: str = "Test synthesis",
    iteration_added: int = 1,
    **kwargs,
) -> EvidenceItem:
    """Helper to build an EvidenceItem with sensible defaults."""
    return EvidenceItem(
        evidence_id=evidence_id,
        title=title,
        synthesis=synthesis,
        iteration_added=iteration_added,
        **kwargs,
    )


class TestComputeDedupKey:
    def test_doi_key(self):
        item = _make_item(doi="10.1234/foo")
        assert compute_dedup_key(item) == "10.1234/foo"

    def test_arxiv_key(self):
        item = _make_item(arxiv_id="2301.12345v2")
        assert compute_dedup_key(item) == "2301.12345"

    def test_canonical_url_key(self):
        item = _make_item(canonical_url="https://example.com/paper?utm_source=x")
        assert compute_dedup_key(item) == "https://example.com/paper"

    def test_falls_back_to_url_when_no_canonical_url(self):
        item = _make_item(url="https://example.com/paper")
        assert compute_dedup_key(item) == "https://example.com/paper"

    def test_canonical_url_takes_precedence_over_url(self):
        item = _make_item(
            url="https://example.com/raw",
            canonical_url="https://example.com/canonical",
        )
        assert compute_dedup_key(item) == "https://example.com/canonical"

    def test_no_identifiers_returns_none(self):
        item = _make_item()
        assert compute_dedup_key(item) is None

    def test_doi_precedence_over_arxiv(self):
        item = _make_item(doi="10.1234/foo", arxiv_id="2301.12345")
        assert compute_dedup_key(item) == "10.1234/foo"


class TestIsDuplicate:
    def test_not_duplicate_empty_set(self):
        item = _make_item(doi="10.1234/foo")
        assert is_duplicate(set(), item) is False

    def test_duplicate_when_key_in_set(self):
        item = _make_item(doi="10.1234/foo")
        assert is_duplicate({"10.1234/foo"}, item) is True

    def test_not_duplicate_different_key(self):
        item = _make_item(doi="10.1234/foo")
        assert is_duplicate({"10.1234/bar"}, item) is False

    def test_no_identifier_never_duplicate(self):
        """Items with no stable identifier are NEVER considered duplicates."""
        item = _make_item()
        assert is_duplicate({"anything"}, item) is False


# ---------------------------------------------------------------------------
# ManagedLedger
# ---------------------------------------------------------------------------

from research.contracts.decisions import SubagentFindings
from research.ledger.ledger import ManagedLedger


class TestManagedLedgerBasics:
    def test_starts_empty(self):
        ml = ManagedLedger()
        assert ml.size == 0
        assert ml.ledger.items == []

    def test_append_adds_items(self):
        ml = ManagedLedger()
        items = [_make_item(evidence_id="ev_1"), _make_item(evidence_id="ev_2")]
        added = ml.append(items)
        assert len(added) == 2
        assert ml.size == 2

    def test_append_returns_added_items(self):
        ml = ManagedLedger()
        item = _make_item(evidence_id="ev_1", doi="10.1234/foo")
        added = ml.append([item])
        assert added == [item]

    def test_size_property(self):
        ml = ManagedLedger()
        ml.append([_make_item(evidence_id="ev_1")])
        assert ml.size == 1
        ml.append([_make_item(evidence_id="ev_2")])
        assert ml.size == 2

    def test_ledger_property_returns_evidence_ledger(self):
        from research.contracts.evidence import EvidenceLedger

        ml = ManagedLedger()
        assert isinstance(ml.ledger, EvidenceLedger)


class TestManagedLedgerDedup:
    def test_dedup_by_doi(self):
        ml = ManagedLedger()
        item1 = _make_item(evidence_id="ev_1", doi="10.1234/foo", title="First")
        item2 = _make_item(evidence_id="ev_2", doi="10.1234/foo", title="Second")
        ml.append([item1])
        added = ml.append([item2])
        assert added == []
        assert ml.size == 1
        assert ml.ledger.items[0].title == "First"

    def test_dedup_by_arxiv_id(self):
        ml = ManagedLedger()
        item1 = _make_item(evidence_id="ev_1", arxiv_id="2301.12345v1")
        item2 = _make_item(evidence_id="ev_2", arxiv_id="2301.12345v2")
        ml.append([item1])
        added = ml.append([item2])
        assert added == []
        assert ml.size == 1

    def test_dedup_by_canonical_url(self):
        ml = ManagedLedger()
        item1 = _make_item(
            evidence_id="ev_1",
            canonical_url="https://example.com/paper",
        )
        item2 = _make_item(
            evidence_id="ev_2",
            canonical_url="https://example.com/paper",
        )
        ml.append([item1])
        added = ml.append([item2])
        assert added == []
        assert ml.size == 1

    def test_no_identifier_always_added(self):
        """Items with no stable identifier are always added (never deduped)."""
        ml = ManagedLedger()
        item1 = _make_item(evidence_id="ev_1", title="A")
        item2 = _make_item(evidence_id="ev_2", title="A")  # same title, no IDs
        ml.append([item1])
        added = ml.append([item2])
        assert len(added) == 1
        assert ml.size == 2

    def test_append_only_no_removal_api(self):
        """Ledger is append-only — no removal method exists."""
        ml = ManagedLedger()
        assert not hasattr(ml, "remove")
        assert not hasattr(ml, "delete")
        assert not hasattr(ml, "pop")

    def test_dedup_within_single_append_call(self):
        """Two items with same DOI in one append() call: only first added."""
        ml = ManagedLedger()
        item1 = _make_item(evidence_id="ev_1", doi="10.1234/foo", title="First")
        item2 = _make_item(evidence_id="ev_2", doi="10.1234/foo", title="Second")
        added = ml.append([item1, item2])
        assert len(added) == 1
        assert added[0].title == "First"
        assert ml.size == 1


class TestManagedLedgerMergeFindings:
    def test_merge_findings_creates_evidence_items(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Finding one", "Finding two"],
            source_references=["https://example.com/1", "https://example.com/2"],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert len(added) == 2
        assert ml.size == 2
        assert added[0].synthesis == "Finding one"
        assert added[1].synthesis == "Finding two"

    def test_merge_findings_generates_unique_ids(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A", "B", "C"],
        )
        added = ml.merge_findings(findings, iteration=1)
        ids = {item.evidence_id for item in added}
        assert len(ids) == 3  # all unique
        for eid in ids:
            assert eid.startswith("ev_")

    def test_merge_findings_sets_iteration(self):
        ml = ManagedLedger()
        findings = SubagentFindings(findings=["A"])
        added = ml.merge_findings(findings, iteration=3)
        assert added[0].iteration_added == 3

    def test_merge_findings_with_confidence_notes(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A"],
            confidence_notes="High confidence",
        )
        added = ml.merge_findings(findings, iteration=1)
        assert added[0].confidence_notes == "High confidence"

    def test_merge_findings_with_matched_excerpts(self):
        """Excerpts with a source prefix matching a ref are kept."""
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A"],
            source_references=["Paper | arxiv:2305.18290"],
            excerpts=['[arxiv:2305.18290] "verbatim quote"'],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert added[0].excerpts == ['[arxiv:2305.18290] "verbatim quote"']

    def test_merge_findings_unmatched_excerpts_dropped(self):
        """Excerpts without a source prefix are dropped."""
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A"],
            excerpts=["verbatim quote without prefix"],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert added[0].excerpts == []


class TestManagedLedgerGetById:
    def test_get_by_id_returns_correct_item(self):
        ml = ManagedLedger()
        item = _make_item(evidence_id="ev_target", doi="10.1234/target")
        ml.append([_make_item(evidence_id="ev_other"), item])
        result = ml.get_by_id("ev_target")
        assert result is not None
        assert result.evidence_id == "ev_target"

    def test_get_by_id_returns_none_for_unknown(self):
        ml = ManagedLedger()
        ml.append([_make_item(evidence_id="ev_1")])
        assert ml.get_by_id("ev_nonexistent") is None

    def test_get_by_id_on_empty_ledger(self):
        ml = ManagedLedger()
        assert ml.get_by_id("ev_1") is None


# ---------------------------------------------------------------------------
# Ledger Projection & Windowing
# ---------------------------------------------------------------------------

from research.contracts.evidence import EvidenceLedger
from research.ledger.projection import (
    ProjectedItem,
    _COMPACT_SYNTHESIS_LIMIT,
    format_projection,
    project_ledger,
)


def _make_ledger(items: list[EvidenceItem]) -> EvidenceLedger:
    """Helper to build an EvidenceLedger from a list of items."""
    return EvidenceLedger(items=items)


class TestProjectLedgerWindowing:
    """Items within the recency window are shown in full."""

    def test_items_within_window_shown_in_full(self):
        """Items added in the last `window_iterations` iterations are full."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_1",
                    title="Recent",
                    synthesis="Full synthesis text",
                    iteration_added=5,
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=6, window_iterations=3)
        assert len(result) == 1
        assert result[0].is_compact is False
        assert result[0].synthesis == "Full synthesis text"

    def test_older_items_compacted(self):
        """Items older than the window are compacted with truncated synthesis."""
        long_synthesis = "A" * 200
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_old",
                    title="Old Item",
                    synthesis=long_synthesis,
                    iteration_added=0,
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=5, window_iterations=3)
        assert len(result) == 1
        assert result[0].is_compact is True
        assert result[0].synthesis == "A" * _COMPACT_SYNTHESIS_LIMIT + "..."

    def test_compact_synthesis_short_no_ellipsis(self):
        """Compact items with short synthesis don't get '...' appended."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_old",
                    title="Old",
                    synthesis="Short",
                    iteration_added=0,
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=5, window_iterations=3)
        assert result[0].is_compact is True
        assert result[0].synthesis == "Short"

    def test_all_items_in_window_no_compaction(self):
        """When all items are within window, none are compacted."""
        ledger = _make_ledger(
            [
                _make_item(evidence_id="ev_1", synthesis="S1", iteration_added=3),
                _make_item(evidence_id="ev_2", synthesis="S2", iteration_added=4),
                _make_item(evidence_id="ev_3", synthesis="S3", iteration_added=5),
            ]
        )
        result = project_ledger(ledger, iteration_index=5, window_iterations=3)
        assert all(not item.is_compact for item in result)

    def test_window_iterations_zero_compacts_all_except_pinned(self):
        """window_iterations=0 means everything is compact (except pinned)."""
        ledger = _make_ledger(
            [
                _make_item(evidence_id="ev_1", synthesis="A" * 200, iteration_added=5),
                _make_item(
                    evidence_id="ev_pinned", synthesis="Pinned text", iteration_added=5
                ),
            ]
        )
        result = project_ledger(
            ledger, iteration_index=5, window_iterations=0, pinned_ids=["ev_pinned"]
        )
        # ev_1 should be compact (window=0 so nothing is in-window)
        assert result[0].is_compact is True
        # ev_pinned should be full (pinned)
        assert result[1].is_compact is False

    def test_exact_window_boundary_is_full(self):
        """Items at the exact window boundary (age == window_iterations - 1) are full.

        Condition: iteration_index - item.iteration_added < window_iterations
        So age = 2, window = 3 -> 2 < 3 -> full (in window).
        """
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_boundary",
                    synthesis="At boundary",
                    iteration_added=3,
                ),
            ]
        )
        # iteration_index=5, iteration_added=3 -> age=2, window=3 -> 2 < 3 -> full
        result = project_ledger(ledger, iteration_index=5, window_iterations=3)
        assert result[0].is_compact is False

    def test_just_outside_window_is_compact(self):
        """Items just outside the window boundary are compact.

        Condition: iteration_index - item.iteration_added >= window_iterations
        So age = 3, window = 3 -> 3 >= 3 -> compact.
        """
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_outside", synthesis="A" * 200, iteration_added=2
                ),
            ]
        )
        # iteration_index=5, iteration_added=2 -> age=3, window=3 -> 3 >= 3 -> compact
        result = project_ledger(ledger, iteration_index=5, window_iterations=3)
        assert result[0].is_compact is True


class TestProjectLedgerPinning:
    """Pinned items are always shown in full regardless of age."""

    def test_pinned_old_item_shown_in_full(self):
        """A pinned item outside the window is still shown in full."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_old_pinned",
                    title="Old Pinned",
                    synthesis="Full text here",
                    iteration_added=0,
                ),
            ]
        )
        result = project_ledger(
            ledger,
            iteration_index=10,
            pinned_ids=["ev_old_pinned"],
            window_iterations=3,
        )
        assert result[0].is_compact is False
        assert result[0].synthesis == "Full text here"

    def test_pinned_ids_none_treated_as_empty(self):
        """pinned_ids=None should not cause errors."""
        ledger = _make_ledger(
            [
                _make_item(evidence_id="ev_1", synthesis="A" * 200, iteration_added=0),
            ]
        )
        result = project_ledger(
            ledger, iteration_index=10, pinned_ids=None, window_iterations=3
        )
        assert result[0].is_compact is True

    def test_pinned_nonexistent_id_ignored(self):
        """Pinning an ID that doesn't exist in the ledger has no effect."""
        ledger = _make_ledger(
            [
                _make_item(evidence_id="ev_1", synthesis="A" * 200, iteration_added=0),
            ]
        )
        result = project_ledger(
            ledger,
            iteration_index=10,
            pinned_ids=["ev_nonexistent"],
            window_iterations=3,
        )
        assert result[0].is_compact is True

    def test_mixed_pinned_and_windowed(self):
        """Mix of pinned, windowed, and compact items."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_old",
                    title="Old",
                    synthesis="A" * 200,
                    iteration_added=0,
                ),
                _make_item(
                    evidence_id="ev_pinned",
                    title="Pinned",
                    synthesis="Pinned synthesis",
                    iteration_added=0,
                ),
                _make_item(
                    evidence_id="ev_recent",
                    title="Recent",
                    synthesis="Recent synthesis",
                    iteration_added=8,
                ),
            ]
        )
        result = project_ledger(
            ledger, iteration_index=9, pinned_ids=["ev_pinned"], window_iterations=3
        )
        assert result[0].is_compact is True  # old, not pinned
        assert result[1].is_compact is False  # old but pinned
        assert result[2].is_compact is False  # recent, in window


class TestProjectLedgerEdgeCases:
    """Edge cases for ledger projection."""

    def test_empty_ledger_returns_empty_list(self):
        ledger = _make_ledger([])
        result = project_ledger(ledger, iteration_index=0)
        assert result == []

    def test_deterministic_same_inputs_same_outputs(self):
        """Projection is deterministic: same inputs produce identical output."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_1",
                    title="A",
                    synthesis="Synth A",
                    iteration_added=0,
                    doi="10.1234/a",
                ),
                _make_item(
                    evidence_id="ev_2",
                    title="B",
                    synthesis="Synth B",
                    iteration_added=2,
                    arxiv_id="2301.12345",
                ),
                _make_item(
                    evidence_id="ev_3",
                    title="C",
                    synthesis="Synth C",
                    iteration_added=4,
                ),
            ]
        )
        pinned = ["ev_1"]
        r1 = project_ledger(
            ledger, iteration_index=5, pinned_ids=pinned, window_iterations=3
        )
        r2 = project_ledger(
            ledger, iteration_index=5, pinned_ids=pinned, window_iterations=3
        )
        assert r1 == r2

    def test_canonical_id_populated_from_doi(self):
        """ProjectedItem.canonical_id is populated from DOI."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_1",
                    synthesis="S",
                    iteration_added=0,
                    doi="10.1234/foo",
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=0, window_iterations=3)
        assert result[0].canonical_id == "10.1234/foo"

    def test_canonical_id_populated_from_arxiv(self):
        """ProjectedItem.canonical_id is populated from arXiv ID."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_1",
                    synthesis="S",
                    iteration_added=0,
                    arxiv_id="2301.12345v2",
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=0, window_iterations=3)
        assert result[0].canonical_id == "2301.12345"

    def test_canonical_id_none_when_no_identifiers(self):
        """ProjectedItem.canonical_id is None when no identifiers exist."""
        ledger = _make_ledger(
            [
                _make_item(evidence_id="ev_1", synthesis="S", iteration_added=0),
            ]
        )
        result = project_ledger(ledger, iteration_index=0, window_iterations=3)
        assert result[0].canonical_id is None

    def test_source_type_carried_through(self):
        """source_type from EvidenceItem is carried through to ProjectedItem."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_1",
                    synthesis="S",
                    iteration_added=0,
                    source_type="preprint",
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=0, window_iterations=3)
        assert result[0].source_type == "preprint"

    def test_preserves_item_order(self):
        """Projected items preserve the order of ledger items."""
        ledger = _make_ledger(
            [
                _make_item(
                    evidence_id="ev_a", title="A", synthesis="S", iteration_added=0
                ),
                _make_item(
                    evidence_id="ev_b", title="B", synthesis="S", iteration_added=1
                ),
                _make_item(
                    evidence_id="ev_c", title="C", synthesis="S", iteration_added=2
                ),
            ]
        )
        result = project_ledger(ledger, iteration_index=3, window_iterations=3)
        assert [r.evidence_id for r in result] == ["ev_a", "ev_b", "ev_c"]


class TestFormatProjection:
    """Tests for format_projection output formatting."""

    def test_empty_projection_returns_empty_string(self):
        assert format_projection([]) == ""

    def test_non_empty_projection_returns_non_empty_string(self):
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="Test",
                source_type="journal",
                canonical_id="10.1234/foo",
                synthesis="A synthesis",
                is_compact=False,
            ),
        ]
        result = format_projection(items)
        assert len(result) > 0

    def test_full_item_includes_source_type(self):
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="Test Paper",
                source_type="journal",
                canonical_id="10.1234/foo",
                synthesis="Full synthesis text",
                is_compact=False,
            ),
        ]
        result = format_projection(items)
        assert "[FULL]" in result
        assert "Test Paper" in result
        assert "journal" in result
        assert "10.1234/foo" in result
        assert "Full synthesis text" in result

    def test_compact_item_format(self):
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="Old Paper",
                source_type="preprint",
                canonical_id="2301.12345",
                synthesis="Truncated...",
                is_compact=True,
            ),
        ]
        result = format_projection(items)
        assert "[COMPACT]" in result
        assert "Old Paper" in result
        assert "2301.12345" in result
        assert "Truncated..." in result
        # Compact items do NOT show source_type
        assert "preprint" not in result

    def test_sections_separated_by_dashes(self):
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="A",
                source_type=None,
                canonical_id=None,
                synthesis="S1",
                is_compact=False,
            ),
            ProjectedItem(
                evidence_id="ev_2",
                title="B",
                source_type=None,
                canonical_id=None,
                synthesis="S2",
                is_compact=True,
            ),
        ]
        result = format_projection(items)
        assert "\n---\n" in result

    def test_full_item_without_source_type_or_canonical_id(self):
        """Full item with no source_type and no canonical_id omits those lines."""
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="No Extras",
                source_type=None,
                canonical_id=None,
                synthesis="Just synthesis",
                is_compact=False,
            ),
        ]
        result = format_projection(items)
        assert "[FULL] No Extras" in result
        assert "Source:" not in result
        assert "ID:" not in result
        assert "Just synthesis" in result

    def test_compact_item_without_canonical_id(self):
        """Compact item with no canonical_id omits the ID line."""
        items = [
            ProjectedItem(
                evidence_id="ev_1",
                title="No ID",
                source_type=None,
                canonical_id=None,
                synthesis="Brief",
                is_compact=True,
            ),
        ]
        result = format_projection(items)
        assert "[COMPACT] No ID" in result
        assert "ID:" not in result
        assert "Brief" in result


class TestProjectionReExports:
    """Verify that projection symbols are re-exported from ledger __init__."""

    def test_project_ledger_importable(self):
        from research.ledger import project_ledger as fn

        assert callable(fn)

    def test_format_projection_importable(self):
        from research.ledger import format_projection as fn

        assert callable(fn)

    def test_projected_item_importable(self):
        from research.ledger import ProjectedItem as cls

        assert cls is not None

    def test_parse_source_reference_importable(self):
        from research.ledger import parse_source_reference as fn

        assert callable(fn)

    def test_parsed_reference_importable(self):
        from research.ledger import ParsedReference as cls

        assert cls is not None


# ---------------------------------------------------------------------------
# Source reference parsing
# ---------------------------------------------------------------------------

from research.ledger.canonical import ParsedReference, parse_source_reference


class TestParseSourceReference:
    """Test extraction of structured identifiers from pipe-separated references."""

    def test_arxiv_only(self):
        ref = "Rafailov et al. (2023) DPO | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"
        parsed = parse_source_reference(ref)
        assert parsed.arxiv_id == "2305.18290"
        assert parsed.url == "https://arxiv.org/abs/2305.18290"
        assert parsed.doi is None

    def test_doi_and_arxiv(self):
        ref = "Author (2023) Title | doi:10.48550/arXiv.2305.18290 | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"
        parsed = parse_source_reference(ref)
        assert parsed.doi == "10.48550/arXiv.2305.18290"
        assert parsed.arxiv_id == "2305.18290"
        assert parsed.url == "https://arxiv.org/abs/2305.18290"

    def test_doi_only(self):
        ref = "Smith et al. (2024) | doi:10.1234/example"
        parsed = parse_source_reference(ref)
        assert parsed.doi == "10.1234/example"
        assert parsed.arxiv_id is None

    def test_url_only(self):
        ref = "Brave Search API Documentation | https://docs.brave.com/api"
        parsed = parse_source_reference(ref)
        assert parsed.url == "https://docs.brave.com/api"
        assert parsed.doi is None
        assert parsed.arxiv_id is None

    def test_empty_string(self):
        parsed = parse_source_reference("")
        assert parsed == ParsedReference()

    def test_none_like_whitespace(self):
        parsed = parse_source_reference("   ")
        assert parsed == ParsedReference()

    def test_no_identifiers(self):
        parsed = parse_source_reference("Just a plain text citation")
        assert parsed.doi is None
        assert parsed.arxiv_id is None
        assert parsed.url is None

    def test_case_insensitive_prefixes(self):
        ref = "Title | DOI:10.1234/foo | ARXIV:2301.12345"
        parsed = parse_source_reference(ref)
        assert parsed.doi == "10.1234/foo"
        assert parsed.arxiv_id == "2301.12345"


# ---------------------------------------------------------------------------
# Improved merge_findings provenance
# ---------------------------------------------------------------------------


class TestMergeFindingsProvenance:
    """Test that merge_findings populates DOI, arXiv ID, and canonical_url."""

    def test_arxiv_populated_from_source_reference(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["DPO achieves comparable performance to PPO"],
            source_references=[
                "Rafailov et al. (2023) DPO | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"
            ],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert len(added) == 1
        assert added[0].arxiv_id == "2305.18290"
        assert added[0].url == "https://arxiv.org/abs/2305.18290"

    def test_doi_populated_from_source_reference(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Key result from the paper"],
            source_references=[
                "Author (2024) Title | doi:10.1234/example | https://doi.org/10.1234/example"
            ],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert len(added) == 1
        assert added[0].doi == "10.1234/example"

    def test_canonical_url_populated(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["API docs show this feature"],
            source_references=[
                "Docs | https://docs.example.com/api?utm_source=google"
            ],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert len(added) == 1
        # canonical_url should strip tracking params
        assert added[0].canonical_url == "https://docs.example.com/api"

    def test_dedup_works_with_parsed_provenance(self):
        """Two findings with the same arXiv ID from different iterations are deduped."""
        ml = ManagedLedger()
        f1 = SubagentFindings(
            findings=["Result A"],
            source_references=["Paper A | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"],
        )
        f2 = SubagentFindings(
            findings=["Result B (same paper)"],
            source_references=["Paper A again | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"],
        )
        added1 = ml.merge_findings(f1, iteration=0)
        added2 = ml.merge_findings(f2, iteration=1)
        assert len(added1) == 1
        assert len(added2) == 0  # deduped by arXiv ID
        assert ml.size == 1

    def test_multiple_findings_paired_with_refs(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Finding about DPO", "Finding about RLHF"],
            source_references=[
                "Paper 1 | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290",
                "Paper 2 | doi:10.1234/rlhf | https://example.com/rlhf",
            ],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added) == 2
        assert added[0].arxiv_id == "2305.18290"
        assert added[1].doi == "10.1234/rlhf"

    def test_findings_without_matching_refs_get_no_provenance(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Finding 1", "Finding 2", "Finding 3"],
            source_references=[
                "Paper 1 | arxiv:2305.18290",
            ],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added) == 3
        assert added[0].arxiv_id == "2305.18290"
        assert added[1].arxiv_id is None
        assert added[2].arxiv_id is None

    def test_raw_url_fallback_for_unparseable_refs(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Finding 1"],
            source_references=["https://example.com/paper"],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added) == 1
        assert added[0].url == "https://example.com/paper"
        assert added[0].canonical_url == "https://example.com/paper"


# ---------------------------------------------------------------------------
# Excerpt bucketing
# ---------------------------------------------------------------------------


class TestMergeFindingsExcerptBucketing:
    """Excerpts are matched to findings by source prefix, not broadcast."""

    def test_excerpts_matched_by_arxiv_prefix(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["DPO result", "RLHF result"],
            source_references=[
                "Paper 1 | arxiv:2305.18290",
                "Paper 2 | arxiv:2301.99999",
            ],
            excerpts=[
                '[arxiv:2305.18290] "DPO achieves 7.8 on MT-Bench"',
                '[arxiv:2301.99999] "RLHF requires 3x more compute"',
            ],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added) == 2
        assert len(added[0].excerpts) == 1
        assert "DPO achieves" in added[0].excerpts[0]
        assert len(added[1].excerpts) == 1
        assert "RLHF requires" in added[1].excerpts[0]

    def test_unmatched_excerpts_are_dropped(self):
        """Unmatched excerpts (no source prefix) are dropped, not broadcast."""
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Finding 1", "Finding 2"],
            source_references=["Paper 1 | arxiv:2305.18290", "Paper 2"],
            excerpts=[
                'Some unmatched excerpt without a bracket prefix',
            ],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added[0].excerpts) == 0
        assert len(added[1].excerpts) == 0

    def test_no_excerpts_no_error(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A finding"],
            source_references=["A ref"],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert added[0].excerpts == []

    def test_excerpts_matched_by_doi_prefix(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Result"],
            source_references=["Paper | doi:10.1234/foo"],
            excerpts=['[doi:10.1234/foo] "The main result is..."'],
        )
        added = ml.merge_findings(findings, iteration=0)
        assert len(added[0].excerpts) == 1
        assert "main result" in added[0].excerpts[0]


# ---------------------------------------------------------------------------
# Source-derived title
# ---------------------------------------------------------------------------


class TestMergeFindingsTitle:
    """merge_findings prefers source-derived titles over finding truncation."""

    def test_title_from_source_reference(self):
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["DPO achieves comparable performance to PPO"],
            source_references=[
                "Rafailov et al. (2023) Direct Preference Optimization | arxiv:2305.18290"
            ],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert added[0].title == "Rafailov et al. (2023) Direct Preference Optimization"

    def test_title_falls_back_to_truncated_finding(self):
        """When source_reference has no human-readable title segment, use finding[:120]."""
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["A finding with no matching ref"],
            source_references=[],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert added[0].title == "A finding with no matching ref"

    def test_title_falls_back_when_ref_is_bare_url(self):
        """A ref that is just a URL produces no title — falls back to finding."""
        ml = ManagedLedger()
        findings = SubagentFindings(
            findings=["Something from example.com"],
            source_references=["https://example.com/paper"],
        )
        added = ml.merge_findings(findings, iteration=1)
        # parse_source_reference sees "https://..." as first segment, so title is None
        assert added[0].title == "Something from example.com"

    def test_title_truncated_at_120_chars_when_from_finding(self):
        """Long finding text is truncated at 120 chars for title."""
        ml = ManagedLedger()
        long_finding = "A" * 200
        findings = SubagentFindings(
            findings=[long_finding],
            source_references=[],
        )
        added = ml.merge_findings(findings, iteration=1)
        assert len(added[0].title) == 120
        assert added[0].title == "A" * 120


# ---------------------------------------------------------------------------
# ParsedReference title extraction
# ---------------------------------------------------------------------------


class TestParsedReferenceTitle:
    """parse_source_reference extracts human-readable title from first pipe segment."""

    def test_title_extracted_from_first_segment(self):
        ref = "Rafailov et al. (2023) DPO | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290"
        parsed = parse_source_reference(ref)
        assert parsed.title == "Rafailov et al. (2023) DPO"

    def test_no_title_when_first_segment_is_url(self):
        ref = "https://example.com/paper | doi:10.1234/foo"
        parsed = parse_source_reference(ref)
        assert parsed.title is None

    def test_no_title_for_empty_ref(self):
        parsed = parse_source_reference("")
        assert parsed.title is None

    def test_title_with_single_segment_no_pipe(self):
        ref = "Just a plain text citation"
        parsed = parse_source_reference(ref)
        assert parsed.title == "Just a plain text citation"

    def test_title_stripped_of_whitespace(self):
        ref = "  Author (2024) Title  | doi:10.1234/foo"
        parsed = parse_source_reference(ref)
        assert parsed.title == "Author (2024) Title"
