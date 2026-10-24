"""Tests for the latency/resilience features: TTL cache, compact market tools,
tool-output hook, LLM retry + SSE status events, per-client rate limiting,
and agent/model configuration (tool-call limit, Groq client, telemetry)."""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

import cache

# ═══════════════════════════════════════════════════════════════════════
# TTL cache
# ═══════════════════════════════════════════════════════════════════════


class TestTTLCache:
    def test_make_key_normalises_ticker(self):
        assert cache.make_key("fundamentals", " aapl ") == "fundamentals:AAPL"
        assert cache.make_key("news", "msft", 3) == "news:MSFT:3"

    def test_set_get_and_expiry(self):
        cache.set("k", "v", ttl=60)
        assert cache.get("k") == "v"
        with patch("cache.time.monotonic", return_value=time.monotonic() + 61):
            assert cache.get("k") is None

    def test_default_ttl_is_used(self, monkeypatch):
        monkeypatch.setattr(cache, "DEFAULT_TTL", 5)
        cache.set("k", 1)
        with patch("cache.time.monotonic", return_value=time.monotonic() + 6):
            assert cache.get("k") is None

    def test_zero_ttl_disables_caching(self, monkeypatch):
        monkeypatch.setattr(cache, "DEFAULT_TTL", 0)
        calls = []
        cache.get_or_set("k", lambda: calls.append(1) or "v")
        cache.get_or_set("k", lambda: calls.append(1) or "v")
        assert len(calls) == 2

    def test_get_or_set_caches(self):
        factory = MagicMock(return_value=42)
        assert cache.get_or_set("k", factory) == 42
        assert cache.get_or_set("k", factory) == 42
        assert factory.call_count == 1

    def test_errors_are_not_cached(self):
        factory = MagicMock(side_effect=[ConnectionError("down"), "ok"])
        with pytest.raises(ConnectionError):
            cache.get_or_set("k", factory)
        assert cache.get_or_set("k", factory) == "ok"

    def test_single_flight_under_concurrency(self):
        calls = []
        started = threading.Event()

        def slow():
            calls.append(1)
            started.set()
            time.sleep(0.05)
            return "value"

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(cache.get_or_set("same", slow)))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == ["value"] * 8
        assert len(calls) == 1

    def test_concurrent_writes_respect_max_size(self, monkeypatch):
        monkeypatch.setattr(cache, "MAX_CACHE_SIZE", 50)

        def writer(n):
            for i in range(200):
                cache.set(f"{n}:{i}", i)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cache.stats()["total_entries"] <= 50

    def test_cached_ticker_info_fetches_once(self):
        fake = MagicMock()
        fake.info = {"currentPrice": 1}
        with patch("yfinance.Ticker", return_value=fake) as ticker:
            assert cache.cached_ticker_info("aapl") == {"currentPrice": 1}
            assert cache.cached_ticker_info("AAPL") == {"currentPrice": 1}
        assert ticker.call_count == 1

    def test_cached_ticker_news_fetches_once(self):
        fake = MagicMock()
        fake.news = [{"title": "Big news", "publisher": "Wire"}]
        with patch("yfinance.Ticker", return_value=fake) as ticker:
            assert cache.cached_ticker_news("TSLA") == fake.news
            assert cache.cached_ticker_news("tsla") == fake.news
        assert ticker.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Compact market tools
# ═══════════════════════════════════════════════════════════════════════

_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "regularMarketPrice": 200.0,
    "regularMarketPreviousClose": 190.0,
    "currency": "USD",
    "marketCap": 3e12,
    "trailingPE": 30.5,
    "forwardPE": None,
    "recommendationKey": "buy",
    "numberOfAnalystOpinions": 40,
    "targetMeanPrice": 230.0,
    "companyOfficers": [{"name": "x"} for _ in range(50)],  # never surfaced
}


@pytest.fixture()
def market_tools():
    from tools.market_tools import MarketDataTools

    return MarketDataTools()


class TestMarketDataTools:
    def test_registers_expected_functions(self, market_tools):
        assert set(market_tools.functions) == {
            "get_current_stock_price",
            "get_stock_fundamentals",
            "get_analyst_recommendations",
            "get_company_news",
        }

    def test_price_is_compact(self, market_tools):
        with patch("cache.cached_ticker_info", return_value=_INFO):
            data = json.loads(market_tools.get_current_stock_price("aapl"))
        assert data == {"symbol": "AAPL", "price": 200.0, "currency": "USD", "change_pct": 5.26}

    def test_fundamentals_include_price_and_drop_nulls(self, market_tools):
        with patch("cache.cached_ticker_info", return_value=_INFO):
            out = market_tools.get_stock_fundamentals("AAPL")
        data = json.loads(out)
        assert data["price"] == 200.0 and data["pe_trailing"] == 30.5
        assert "pe_forward" not in data
        assert "companyOfficers" not in out and "\n" not in out

    def test_results_cached_per_ticker_and_function(self, market_tools):
        with patch("cache.cached_ticker_info", return_value=_INFO) as info:
            market_tools.get_stock_fundamentals("AAPL")
            market_tools.get_stock_fundamentals("aapl")
            market_tools.get_current_stock_price("AAPL")
            market_tools.get_stock_fundamentals("MSFT")
        # fundamentals:AAPL once, price:AAPL once, fundamentals:MSFT once
        assert info.call_count == 3

    def test_errors_are_short_and_not_cached(self, market_tools):
        with patch("cache.cached_ticker_info", side_effect=ConnectionError("boom")):
            out = market_tools.get_stock_fundamentals("AAPL")
        assert out.startswith("Error:") and "AAPL" in out
        with patch("cache.cached_ticker_info", return_value=_INFO):
            assert json.loads(market_tools.get_stock_fundamentals("AAPL"))["price"] == 200.0

    def test_recommendations_keep_latest_row_only(self, market_tools):
        table = pd.DataFrame(
            [
                {"period": "0m", "strongBuy": 10, "buy": 20, "hold": 5, "sell": 1, "strongSell": 0},
                {"period": "-1m", "strongBuy": 9, "buy": 19, "hold": 6, "sell": 2, "strongSell": 1},
            ]
        )
        with patch("cache.cached_recommendations", return_value=table), patch(
            "cache.cached_ticker_info", return_value=_INFO
        ):
            data = json.loads(market_tools.get_analyst_recommendations("AAPL"))
        assert data["strongBuy"] == 10 and data["buy"] == 20
        assert data["consensus"] == "buy" and data["target_mean"] == 230.0
        assert "period" not in data

    def test_news_is_truncated_and_limited(self, market_tools):
        news = [
            {"content": {"title": "T" * 500, "pubDate": "2026-10-01T12:00:00Z",
                         "provider": {"displayName": "Wire"}, "body": "x" * 5000}}
            for _ in range(10)
        ]
        with patch("cache.cached_ticker_news", return_value=news):
            out = market_tools.get_company_news("AAPL", num_stories=50)
        stories = json.loads(out)
        assert len(stories) == 5
        assert len(stories[0]["title"]) == 160
        assert stories[0] == {"title": "T" * 160, "publisher": "Wire", "date": "2026-10-01"}

    def test_news_epoch_dates(self, market_tools):
        news = [{"title": "Old style", "publisher": "P", "providerPublishTime": 1_790_000_000}]
        with patch("cache.cached_ticker_news", return_value=news):
            assert json.loads(market_tools.get_company_news("AAPL"))[0]["date"] == "2026-09-21"


# ═══════════════════════════════════════════════════════════════════════
# Tool-output hook
# ═══════════════════════════════════════════════════════════════════════


class TestToolHook:
    def test_long_output_truncated(self, monkeypatch):
        from backend.app.agent import tool_registry
        from config import settings

        fn = MagicMock(return_value="x" * (settings.tool_output_max_chars + 500))
        out = tool_registry.compact_tool_output("create_stock_chart", fn, {"symbol": "AAPL"})
        assert out.endswith("…[truncated]")
        assert len(out) <= settings.tool_output_max_chars + len(" …[truncated]")

    def test_short_and_non_string_output_untouched(self):
        from backend.app.agent.tool_registry import compact_tool_output

        assert compact_tool_output("x", lambda **_: "short", {}) == "short"
        assert compact_tool_output("x", lambda **_: {"a": 1}, {}) == {"a": 1}

    def test_search_results_are_cached(self):
        from backend.app.agent.tool_registry import compact_tool_output

        fn = MagicMock(return_value="results")
        for _ in range(3):
            compact_tool_output("duckduckgo_news", fn, {"query": "AAPL earnings"})
        compact_tool_output("duckduckgo_news", fn, {"query": "MSFT earnings"})
        assert fn.call_count == 2

    def test_file_writing_tools_not_cached(self):
        from backend.app.agent.tool_registry import compact_tool_output

        fn = MagicMock(return_value="aapl_chart.png")
        compact_tool_output("create_stock_chart", fn, {"symbol": "AAPL"})
        compact_tool_output("create_stock_chart", fn, {"symbol": "AAPL"})
        assert fn.call_count == 2

    def test_search_tools_limited(self):
        from backend.app.agent.tool_registry import SEARCH_MAX_RESULTS, build_tools

        ddg = next(t for t in build_tools() if type(t).__name__ == "DuckDuckGoTools")
        assert ddg.fixed_max_results == SEARCH_MAX_RESULTS


# ═══════════════════════════════════════════════════════════════════════
# Agent / model configuration
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_for_unit_tests")


class TestAgentConfig:
    def test_telemetry_defaults_off(self):
        import config  # noqa: F401

        assert os.environ.get("AGNO_TELEMETRY", "").lower() == "false"

    def test_tool_call_limit_and_hook(self, groq_key):
        from backend.app.agent.financial_agent import create_agent
        from backend.app.agent.tool_registry import compact_tool_output
        from config import settings

        agent = create_agent()
        assert agent.tool_call_limit == settings.agent_tool_call_limit == 3
        assert agent.tool_hooks == [compact_tool_output]
        agent.set_monitoring()  # agno applies AGNO_TELEMETRY here at run time
        assert agent.telemetry is False

    def test_groq_model_env_override(self, groq_key, monkeypatch):
        from backend.app.agent.financial_agent import DEFAULT_MODEL_ID, create_agent

        assert create_agent().model.id == DEFAULT_MODEL_ID
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        assert create_agent().model.id == "openai/gpt-oss-20b"
        assert create_agent(model_id="qwen/qwen3.8-27b").model.id == "qwen/qwen3.8-27b"

    def test_groq_sdk_retries_disabled_and_client_shared(self, groq_key):
        from backend.app.agent.financial_agent import create_agent

        a, b = create_agent(), create_agent()
        assert a.model.max_retries == 0
        assert isinstance(a.model.http_client, httpx.Client)
        assert a.model.http_client is b.model.http_client

    def test_model_deepcopy_has_no_httpx_state_error(self, groq_key):
        """agno deep-copies the model per Agent; must not trigger the
        SyncHttpxClientWrapper '_state' AttributeError from __del__."""
        from backend.app.agent.financial_agent import create_agent

        model = create_agent().model
        model.get_client()
        captured = []
        old_hook = sys.unraisablehook
        sys.unraisablehook = captured.append
        try:
            for _ in range(3):
                copy.deepcopy(model)
            gc.collect()
        finally:
            sys.unraisablehook = old_hook
        assert not [u for u in captured if "_state" in str(u.exc_value)]


# ═══════════════════════════════════════════════════════════════════════
# Retry helpers + service behaviour
# ═══════════════════════════════════════════════════════════════════════


def _groq_rate_limit(retry_after: str | None = "3"):
    import groq

    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "https://api.groq.com/x")
    )
    return groq.RateLimitError("Rate limit reached. Please try again in 7.5s.", response=response, body=None)


def _agno_rate_limit():
    from agno.exceptions import ModelProviderError

    try:
        raise ModelProviderError("rate_limit_exceeded", status_code=429) from _groq_rate_limit()
    except ModelProviderError as exc:
        return exc


class TestRetryHelpers:
    def test_detects_rate_limits(self):
        from agno.exceptions import ModelProviderError

        from backend.app.agent.financial_agent import is_rate_limit_error

        assert is_rate_limit_error(_groq_rate_limit())
        assert is_rate_limit_error(_agno_rate_limit())
        assert is_rate_limit_error(RuntimeError("Error code: 429 - Too Many Requests"))
        assert not is_rate_limit_error(ModelProviderError("bad gateway", status_code=502))
        assert not is_rate_limit_error(ValueError("nope"))

    def test_retry_after_header_then_message(self):
        from backend.app.agent.financial_agent import retry_after_seconds

        assert retry_after_seconds(_agno_rate_limit()) == 3.0
        assert retry_after_seconds(_groq_rate_limit(None)) == 7.5
        assert retry_after_seconds(RuntimeError("try again in 1m2.5s")) == 62.5
        assert retry_after_seconds(RuntimeError("other")) is None

    def test_delay_is_capped(self):
        from backend.app.agent.financial_agent import retry_delay
        from config import settings

        assert retry_delay(_groq_rate_limit("3"), 1) == 3
        assert retry_delay(_groq_rate_limit("999"), 1) == settings.llm_retry_max_delay
        assert retry_delay(RuntimeError("Too Many Requests"), 2) == 4  # 2**attempt
        assert retry_delay(ValueError("x"), 1) == 1

    def test_default_retries_from_settings(self):
        from backend.app.agent.financial_agent import FinancialAgentService
        from config import settings

        assert settings.llm_max_retries == 2
        assert FinancialAgentService(agent=MagicMock())._max_retries == 2


def _chunk(text):
    c = MagicMock()
    c.content = text
    c.event = "RunResponseContent"
    return c


class TestServiceRateLimitRetries:
    def test_stream_emits_status_then_recovers(self, _no_retry_sleep):
        from backend.app.agent.financial_agent import FinancialAgentService

        agent = MagicMock()
        agent.run.side_effect = [_agno_rate_limit(), [_chunk("Hello")]]
        chunks = list(FinancialAgentService(agent=agent, max_retries=2).stream("q"))
        assert chunks == ["[STATUS] Rate limit reached — retrying in 3s…", "Hello"]
        _no_retry_sleep.assert_called_once_with(3)

    def test_stream_gives_clear_error_after_max_retries(self, _no_retry_sleep):
        from backend.app.agent.financial_agent import (
            RATE_LIMIT_ERROR_MESSAGE,
            FinancialAgentService,
        )

        agent = MagicMock()
        agent.run.side_effect = _agno_rate_limit()
        chunks = list(FinancialAgentService(agent=agent, max_retries=2).stream("q"))
        assert agent.run.call_count == 3
        assert sum(c.startswith("[STATUS]") for c in chunks) == 2
        assert chunks[-1] == f"[ERROR] {RATE_LIMIT_ERROR_MESSAGE}"

    def test_stream_run_error_event_is_not_shown_as_text(self):
        from backend.app.agent.financial_agent import FinancialAgentService

        err = MagicMock()
        err.event = "RunError"
        err.content = "Error code: 429 rate_limit_exceeded"
        agent = MagicMock()
        agent.run.side_effect = [[err], [_chunk("ok")]]
        chunks = list(FinancialAgentService(agent=agent, max_retries=1).stream("q"))
        assert chunks[0].startswith("[STATUS] Rate limit reached")
        assert chunks[1] == "ok"

    def test_stream_failure_after_output_not_retried(self):
        from backend.app.agent.financial_agent import FinancialAgentService

        def partial(*_a, **_k):
            yield _chunk("partial")
            raise _agno_rate_limit()

        agent = MagicMock()
        agent.run.side_effect = lambda *a, **k: partial()
        chunks = list(FinancialAgentService(agent=agent, max_retries=2).stream("q"))
        assert chunks[0] == "partial" and chunks[-1].startswith("[ERROR]")
        assert agent.run.call_count == 1

    def test_run_raises_rate_limit_error(self, _no_retry_sleep):
        from backend.app.agent.financial_agent import FinancialAgentService, LLMRateLimitError

        agent = MagicMock()
        agent.run.side_effect = _agno_rate_limit()
        with pytest.raises(LLMRateLimitError):
            FinancialAgentService(agent=agent, max_retries=2).run("q")
        assert agent.run.call_count == 3
        assert _no_retry_sleep.call_count == 2

    def test_not_configured_is_not_retried(self):
        from backend.app.agent.financial_agent import (
            AgentNotConfiguredError,
            FinancialAgentService,
        )

        agent = MagicMock()
        agent.run.side_effect = AgentNotConfiguredError("no key")
        with pytest.raises(AgentNotConfiguredError):
            FinancialAgentService(agent=agent, max_retries=2).run("q")
        assert agent.run.call_count == 1


class TestChatEndpointsRateLimitedUpstream:
    def test_non_stream_returns_503_on_llm_rate_limit(self, client, mock_agent):
        from backend.app.agent.financial_agent import RATE_LIMIT_ERROR_MESSAGE, LLMRateLimitError

        mock_agent.run.side_effect = LLMRateLimitError(RATE_LIMIT_ERROR_MESSAGE)
        res = client.post("/api/chat", json={"message": "price of AAPL"})
        assert res.status_code == 503
        assert res.json() == {"detail": RATE_LIMIT_ERROR_MESSAGE}
        assert res.headers["Retry-After"] == "60"


# ═══════════════════════════════════════════════════════════════════════
# Per-client rate limiting + CORS config
# ═══════════════════════════════════════════════════════════════════════


class TestClientRateLimit:
    @pytest.fixture(autouse=True)
    def _limit(self, monkeypatch):
        from backend.app import state

        monkeypatch.setattr(state, "RATE_LIMIT_PER_MINUTE", 3)
        monkeypatch.setattr(state, "RATE_LIMIT_MAX", 100)
        monkeypatch.setattr(state, "IP_RATE_LIMIT_MAX", 100)

    def test_limit_returns_429_json(self, client, mock_agent):
        for _ in range(3):
            assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
        res = client.post("/api/chat/stream", json={"message": "hi"})
        assert res.status_code == 429
        body = res.json()
        assert "3 chat messages per minute" in body["detail"]
        assert int(res.headers["Retry-After"]) >= 1

    def test_limit_is_per_client(self, client, other_client, mock_agent):
        for _ in range(3):
            client.post("/api/chat", json={"message": "hi"})
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 429
        assert other_client.post("/api/chat", json={"message": "hi"}).status_code == 200

    def test_other_endpoints_not_counted(self, client, mock_agent):
        for _ in range(5):
            assert client.get("/api/sessions").status_code == 200
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200

    def test_zero_disables(self, client, mock_agent, monkeypatch):
        from backend.app import state

        monkeypatch.setattr(state, "RATE_LIMIT_PER_MINUTE", 0)
        for _ in range(6):
            assert client.post("/api/chat", json={"message": "hi"}).status_code == 200

    def test_falls_back_to_ip_without_client_id(self):
        from fastapi import HTTPException

        from backend.app.services.sessions import enforce_client_rate_limit
        from backend.app.state import _client_rate_limits

        request = MagicMock()
        request.headers = {}
        request.client.host = "203.0.113.9"
        for _ in range(3):
            enforce_client_rate_limit(request)
        with pytest.raises(HTTPException) as exc:
            enforce_client_rate_limit(request)
        assert exc.value.status_code == 429
        assert "ip:203.0.113.9" in _client_rate_limits


class TestSettingsFromEnv:
    def test_env_values(self, monkeypatch):
        from config import load_settings

        monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "7")
        monkeypatch.setenv("LLM_MAX_RETRIES", "4")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "15")
        s = load_settings()
        assert s.cors_origins == ["https://a.example", "https://b.example"]
        assert (s.rate_limit_per_minute, s.llm_max_retries, s.cache_ttl_seconds) == (7, 4, 15)

    def test_invalid_numbers_fall_back(self, monkeypatch):
        from config import load_settings

        monkeypatch.setenv("LLM_MAX_RETRIES", "lots")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "-5")
        s = load_settings()
        assert s.llm_max_retries == 2
        assert s.cache_ttl_seconds == 0
