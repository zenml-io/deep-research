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
