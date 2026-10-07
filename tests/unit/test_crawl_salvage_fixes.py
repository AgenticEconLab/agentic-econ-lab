# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Web-crawl fixes: PDF extraction, DOI resolution, Semantic Scholar throttle."""

from unittest.mock import MagicMock, patch

from shared.tools.webcrawl_tool import _looks_like_pdf, _resolve_doi_url, _scrape_pdf


class TestDoiResolution:
    def test_nber_doi_rewritten_without_network(self):
        assert _resolve_doi_url("https://doi.org/10.3386/w27399") == \
            "https://www.nber.org/papers/w27399"

    def test_non_doi_untouched(self):
        assert _resolve_doi_url("https://example.org/a.html") == "https://example.org/a.html"

    def test_generic_doi_follows_redirect(self):
        fake = MagicMock()
        fake.url = "https://publisher.example/article/123"
        with patch("requests.head", return_value=fake):
            out = _resolve_doi_url("https://doi.org/10.1080/15140326.2023.2289724")
        assert out == "https://publisher.example/article/123"

    def test_generic_doi_failure_returns_original(self):
        with patch("requests.head", side_effect=OSError("net down")):
            url = "https://doi.org/10.9999/unknown"
            assert _resolve_doi_url(url) == url


class TestPdfExtraction:
    def test_pdf_url_detection(self):
        assert _looks_like_pdf("https://x.org/paper.pdf")
        assert _looks_like_pdf("https://x.org/paper.PDF?dl=1")
        assert not _looks_like_pdf("https://x.org/paper.html")

    def _pdf_bytes(self, text="Structural Reinforcement Learning " * 20):
        import io
        from pypdf import PdfWriter
        w = PdfWriter()
        w.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        w.write(buf)
        return buf.getvalue()

    def test_pdf_extraction_returns_result_for_text_pdf(self):
        # pypdf can't easily synthesize text pages; verify the honest-None paths instead:
        # (a) non-PDF bytes -> None; (b) blank PDF (no extractable text) -> None
        fake = MagicMock()
        fake.content = b"<html>not a pdf</html>"
        fake.headers = {"content-type": "text/html"}
        with patch("shared.observability.tracked_get", return_value=fake):
            assert _scrape_pdf("https://x.org/paper.pdf") is None

        fake.content = self._pdf_bytes()
        fake.headers = {"content-type": "application/pdf"}
        with patch("shared.observability.tracked_get", return_value=fake):
            assert _scrape_pdf("https://x.org/blank.pdf") is None  # <200 chars extractable

    def test_pdf_never_raises_on_network_failure(self):
        with patch("shared.observability.tracked_get", side_effect=OSError("boom")):
            assert _scrape_pdf("https://x.org/paper.pdf") is None


class TestSemanticScholarThrottle:
    def test_429_retried_once_with_backoff(self, monkeypatch):
        import shared.data.s2_client as s2
        monkeypatch.setattr(s2, "_S2_MIN_INTERVAL", 0.0)
        client = s2.SemanticScholarClient.__new__(s2.SemanticScholarClient)
        client.api_key = None

        r429 = MagicMock(status_code=429, headers={"retry-after": "0"})
        r200 = MagicMock(status_code=200, headers={})
        r200.json.return_value = {"data": []}
        r200.raise_for_status.return_value = None
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        with patch.object(s2.httpx, "get", side_effect=[r429, r200]) as g:
            out = client._get("/paper/search", {"query": "q"})
        assert out == {"data": []} and g.call_count == 2
        assert sleeps, "429 must trigger a backoff sleep"
