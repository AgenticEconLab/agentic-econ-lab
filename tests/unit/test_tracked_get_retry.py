# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for tracked_get/tracked_post retry logic in shared.observability."""

from unittest.mock import patch, MagicMock

import requests as _requests
import pytest
from shared.observability import tracked_get, tracked_post, MetricsCollector


def _mock_response(status_code: int, ok: bool = True, json_data=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.json.return_value = json_data or {}
    if not ok:
        exc = _requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = exc
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestTrackedGetRetry:
    """Tests for tracked_get() retry behavior."""

    @patch("requests.get")
    def test_default_retries_zero_raises_on_500(self, mock_get):
        """With default retries=0, 500 raises immediately."""
        mock_get.return_value = _mock_response(500, ok=False)
        with pytest.raises(_requests.HTTPError):
            tracked_get("https://api.example.com/test")

    @patch("requests.get")
    def test_retries_2_succeeds_after_transient_500(self, mock_get):
        """With retries=2, succeeds on third attempt after two 500s."""
        mock_get.side_effect = [
            _mock_response(500, ok=False),
            _mock_response(500, ok=False),
            _mock_response(200, ok=True, json_data={"data": []}),
        ]
        collector = MetricsCollector()
        resp = tracked_get(
            "https://api.example.com/test",
            collector=collector,
            agent="TestAgent",
            retries=2,
            retry_backoff=0.01,  # Fast for testing
        )
        assert resp.status_code == 200
        # Intermediate retries are NOT recorded as failures now — only the final outcome,
        # so a transient error that recovers on retry doesn't pollute success_rate.
        records = collector.tool_calls
        assert len([r for r in records if r.success]) == 1
        assert all("[retry" not in (r.error or "") for r in records)

    @patch("requests.get")
    def test_404_not_retried(self, mock_get):
        """404 is not in retry_on, so should not retry."""
        mock_get.return_value = _mock_response(404, ok=False)
        with pytest.raises(_requests.HTTPError):
            tracked_get(
                "https://api.example.com/test",
                retries=2,
                retry_backoff=0.01,
            )
        assert mock_get.call_count == 1  # No retries

    @patch("requests.get")
    def test_exhausted_retries_raises(self, mock_get):
        """When all retries exhausted, exception is raised."""
        mock_get.return_value = _mock_response(500, ok=False)
        collector = MetricsCollector()
        with pytest.raises(_requests.HTTPError):
            tracked_get(
                "https://api.example.com/test",
                collector=collector,
                agent="TestAgent",
                retries=2,
                retry_backoff=0.01,
            )
        assert mock_get.call_count == 3  # Initial + 2 retries

    @patch("requests.get")
    def test_collector_records_only_final_outcome(self, mock_get):
        """A transient failure that succeeds on retry is recorded ONCE (final success),
        not as multiple failed attempts — so success_rate reflects real outcomes."""
        mock_get.side_effect = [
            _mock_response(500, ok=False),
            _mock_response(200, ok=True, json_data={"ok": True}),
        ]
        collector = MetricsCollector()
        resp = tracked_get(
            "https://api.example.com/test",
            collector=collector,
            agent="RetryAgent",
            retries=1,
            retry_backoff=0.01,
        )
        assert resp.status_code == 200
        records = collector.tool_calls
        # Only the final successful outcome is recorded (no intermediate [retry] failures).
        assert len(records) == 1
        assert records[0].success is True
        assert all("[retry" not in (r.error or "") for r in records)

    @patch("requests.get")
    def test_backward_compat_no_retries(self, mock_get):
        """Without retries param, behaves exactly like before."""
        mock_get.return_value = _mock_response(200, ok=True, json_data={"data": []})
        resp = tracked_get("https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_429_retried_by_default(self, mock_get):
        """429 (rate limit) is in default retry_on."""
        mock_get.side_effect = [
            _mock_response(429, ok=False),
            _mock_response(200, ok=True),
        ]
        resp = tracked_get(
            "https://api.example.com/test",
            retries=1,
            retry_backoff=0.01,
        )
        assert resp.status_code == 200
        assert mock_get.call_count == 2


class TestTrackedPostRetry:
    """Tests for tracked_post() retry behavior."""

    @patch("requests.post")
    def test_post_retries_on_500(self, mock_post):
        """tracked_post also retries on 500."""
        mock_post.side_effect = [
            _mock_response(500, ok=False),
            _mock_response(200, ok=True),
        ]
        collector = MetricsCollector()
        resp = tracked_post(
            "https://api.example.com/test",
            collector=collector,
            retries=1,
            retry_backoff=0.01,
        )
        assert resp.status_code == 200
        assert mock_post.call_count == 2


class TestTrackedTimeout:
    """Every tracked_* call must be time-bounded so a stuck endpoint can't hang the run."""

    @patch("requests.get")
    def test_get_injects_default_timeout(self, mock_get, monkeypatch):
        monkeypatch.delenv("AEL_HTTP_TIMEOUT", raising=False)
        mock_get.return_value = _mock_response(200, ok=True)
        tracked_get("https://api.example.com/test")
        assert mock_get.call_args.kwargs.get("timeout") == 30.0  # default bound applied

    @patch("requests.get")
    def test_get_respects_explicit_timeout(self, mock_get):
        """A caller that passes its own timeout is NOT overridden."""
        mock_get.return_value = _mock_response(200, ok=True)
        tracked_get("https://api.example.com/test", timeout=5)
        assert mock_get.call_args.kwargs.get("timeout") == 5

    @patch("requests.get")
    def test_get_timeout_env_override(self, mock_get, monkeypatch):
        monkeypatch.setenv("AEL_HTTP_TIMEOUT", "12.5")
        mock_get.return_value = _mock_response(200, ok=True)
        tracked_get("https://api.example.com/test")
        assert mock_get.call_args.kwargs.get("timeout") == 12.5

    @patch("requests.post")
    def test_post_injects_default_timeout(self, mock_post, monkeypatch):
        monkeypatch.delenv("AEL_HTTP_TIMEOUT", raising=False)
        mock_post.return_value = _mock_response(200, ok=True)
        tracked_post("https://api.example.com/test")
        assert mock_post.call_args.kwargs.get("timeout") == 30.0
