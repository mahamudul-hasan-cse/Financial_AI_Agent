"""Tests for the automatic primary → backup model fallback.

Two chains are covered: Groq primary → OpenRouter models, and OpenRouter
primary → OpenRouter backup models → Groq.  Every LLM call is mocked: the
primary agent is a MagicMock whose model passes ``isinstance`` for Groq or
OpenRouter, and backup agents come from an injected ``fallback_factory``.  Keys are fake and set per test, so nothing touches the
network or a developer's real configuration.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from agno.exceptions import ModelProviderError
from agno.models.groq import Groq
from fastapi.testclient import TestClient

from tests.conftest import TEST_CLIENT_ID

PRIMARY_MODEL = "openai/gpt-oss-120b"
# Backups are (provider, model id) pairs; Groq primary tries these in order.
BACKUP_1 = ("openrouter", "openai/gpt-oss-120b")
BACKUP_2 = ("openrouter", "google/gemini-2.5-flash")
GROQ_BACKUP = ("groq", "openai/gpt-oss-120b")
BUSY = "[STATUS] Primary model busy — switched to backup model"


@pytest.fixture()
def openrouter_key(monkeypatch):
    """OpenRouter key set, Groq kept as primary (the chain these tests cover)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key-for-unit-tests")
    monkeypatch.setenv("LLM_PROVIDER", "groq")


@pytest.fixture()
def openrouter_primary(monkeypatch):
    """OpenRouter key set and LLM_PROVIDER left to its default (openrouter)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key-for-unit-tests")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _provider_error(status: int, message: str = "upstream error") -> ModelProviderError:
    try:
        raise ModelProviderError(message, status_code=status)
    except ModelProviderError as exc:
        return exc


def _rate_limited() -> ModelProviderError:
    return _provider_error(429, "rate_limit_exceeded: Please try again in 30s.")


def _chunk(text: str) -> MagicMock:
    c = MagicMock()
    c.content = text
    c.event = "RunResponseContent"
    return c


def _result(text: str) -> MagicMock:
    r = MagicMock()
    r.content = text
    return r


def _groq_agent(side_effect) -> MagicMock:
    agent = MagicMock()
    agent.model = MagicMock(spec=Groq)
    agent.model.id = PRIMARY_MODEL
    agent.run.side_effect = side_effect
    return agent


def _openrouter_agent(side_effect) -> MagicMock:
    from agno.models.openrouter import OpenRouter

    agent = MagicMock()
    agent.model = MagicMock(spec=OpenRouter)
    agent.model.id = PRIMARY_MODEL
    agent.run.side_effect = side_effect
    return agent


class _Backups:
    """``fallback_factory`` stand-in: one mock agent per (provider, model id)."""

    def __init__(self, outcomes=None):
        # outcomes: (provider, model id) -> run side_effect (exception, result, or callable)
        self.agents: dict[tuple[str, str], MagicMock] = {}
        for key, side_effect in (outcomes or {}).items():
            agent = MagicMock()
            agent.run.side_effect = side_effect
            self.agents[key] = agent
        self.calls: list[tuple[str, str]] = []

    def __call__(self, provider, model_id, session_id):
        self.calls.append((provider, model_id))
        return self.agents[(provider, model_id)]


def _service(primary, backups, max_retries=2):
    from backend.app.agent.financial_agent import FinancialAgentService

    return FinancialAgentService(agent=primary, max_retries=max_retries, fallback_factory=backups)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


class TestFallbackConfig:
    def test_disabled_without_key(self):
        from config import get_fallback_provider

        assert get_fallback_provider() == "none"

    def test_defaults_to_openrouter_with_key(self, openrouter_key):
        from config import get_fallback_provider, get_openrouter_models

        assert get_fallback_provider() == "openrouter"
        assert get_openrouter_models() == [BACKUP_1[1], BACKUP_2[1]]

    def test_none_disables(self, openrouter_key, monkeypatch):
        from config import get_fallback_provider

        monkeypatch.setenv("FALLBACK_PROVIDER", "none")
        assert get_fallback_provider() == "none"
        monkeypatch.setenv("FALLBACK_PROVIDER", "bogus")
        assert get_fallback_provider() == "none"

    def test_model_list_from_env(self, monkeypatch):
        from config import get_openrouter_models

        monkeypatch.setenv("OPENROUTER_MODEL", "a/one:free")
        monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", " b/two:free, ,a/one:free,c/three ")
        assert get_openrouter_models() == ["a/one:free", "b/two:free", "c/three"]
        monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "")
        assert get_openrouter_models() == ["a/one:free"]

    def test_fallback_errors(self):
        from backend.app.agent.financial_agent import is_fallback_error

        assert is_fallback_error(_rate_limited())
        assert is_fallback_error(_provider_error(503, "Service Unavailable"))
        assert is_fallback_error(_provider_error(500))
        assert is_fallback_error(httpx.ConnectError("connection refused"))
        assert not is_fallback_error(_provider_error(400, "invalid tool call"))
        assert not is_fallback_error(ValueError("bug"))

    def test_backup_agent_reuses_tools_and_instructions(self, openrouter_key, monkeypatch):
        """The real (unmocked) builder: agno OpenRouter model, same tools/instructions."""
        pytest.importorskip("openai")
        from agno.models.openrouter import OpenRouter

        from backend.app.agent.financial_agent import FinancialAgentService

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_for_unit_tests")
        svc = FinancialAgentService()
        backup = svc._fallback_agent(*BACKUP_1, "sess_1")
        assert isinstance(backup.model, OpenRouter)
        assert backup.model.id == BACKUP_1[1]
        assert backup.model.max_retries == 0
        assert backup.tools == svc.agent.tools
        assert backup.instructions == svc.agent.instructions
        assert backup.tool_call_limit == svc.agent.tool_call_limit
        assert backup.session_id == "sess_1"


# ═══════════════════════════════════════════════════════════════════════
# FinancialAgentService.run
# ═══════════════════════════════════════════════════════════════════════


class TestRunFallback:
    def test_fallback_on_429_without_waiting(self, openrouter_key, _no_retry_sleep):
        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: [_result("Backup says hi")]})
        svc = _service(primary, backups)

        reply = svc.run("q")

        assert reply == "Backup says hi"
        assert (reply.provider, reply.model, reply.fallback) == (*BACKUP_1, True)
        assert primary.run.call_count == 1
        _no_retry_sleep.assert_not_called()

    def test_fallback_on_5xx(self, openrouter_key, _no_retry_sleep):
        primary = _groq_agent(_provider_error(503, "Service Unavailable"))
        backups = _Backups({BACKUP_1: [_result("ok")]})
        reply = _service(primary, backups).run("q")
        assert reply.fallback and reply.provider == "openrouter"
        _no_retry_sleep.assert_not_called()

    def test_fallback_on_connection_error(self, openrouter_key):
        primary = _groq_agent(httpx.ConnectError("connection refused"))
        backups = _Backups({BACKUP_1: [_result("ok")]})
        assert _service(primary, backups).run("q").fallback

    def test_second_backup_model_used_when_first_fails(self, openrouter_key):
        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: _rate_limited(), BACKUP_2: [_result("second")]})
        reply = _service(primary, backups).run("q")
        assert reply == "second" and reply.model == BACKUP_2[1]
        assert backups.calls == [BACKUP_1, BACKUP_2]

    def test_disabled_uses_groq_retry_path(self, openrouter_key, monkeypatch, _no_retry_sleep):
        from backend.app.agent.financial_agent import LLMRateLimitError

        monkeypatch.setenv("FALLBACK_PROVIDER", "none")
        primary = _groq_agent(_rate_limited())
        backups = _Backups()
        with pytest.raises(LLMRateLimitError):
            _service(primary, backups).run("q")
        assert backups.calls == []
        assert primary.run.call_count == 3
        assert _no_retry_sleep.call_count == 2

    def test_no_key_means_no_fallback(self, _no_retry_sleep):
        primary = _groq_agent([_rate_limited(), _result("groq again")])
        backups = _Backups()
        reply = _service(primary, backups).run("q")
        assert reply == "groq again" and reply.provider == "groq"
        assert backups.calls == []

    def test_client_errors_do_not_fall_back(self, openrouter_key):
        primary = _groq_agent([_provider_error(400, "bad request"), _result("fine")])
        backups = _Backups()
        assert _service(primary, backups).run("q") == "fine"
        assert backups.calls == []

    def test_fallback_failing_returns_to_groq_retries(self, openrouter_key, _no_retry_sleep):
        primary = _groq_agent([_rate_limited(), _result("groq recovered")])
        backups = _Backups({BACKUP_1: _rate_limited(), BACKUP_2: _provider_error(502)})
        svc = _service(primary, backups)

        reply = svc.run("q")

        assert reply == "groq recovered"
        assert (reply.provider, reply.model, reply.fallback) == ("groq", PRIMARY_MODEL, False)
        assert backups.calls == [BACKUP_1, BACKUP_2]
        _no_retry_sleep.assert_called_once()

    def test_everything_failing_gives_clear_error_and_tries_backups_once(
        self, openrouter_key, _no_retry_sleep
    ):
        from backend.app.agent.financial_agent import LLMRateLimitError

        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: _rate_limited(), BACKUP_2: _rate_limited()})
        with pytest.raises(LLMRateLimitError):
            _service(primary, backups).run("q")
        assert primary.run.call_count == 3
        assert backups.calls == [BACKUP_1, BACKUP_2]  # not re-tried on every Groq retry

    def test_provider_recorded(self, openrouter_key):
        primary = _groq_agent([_result("from groq"), _rate_limited()])
        backups = _Backups({BACKUP_1: [_result("from backup")]})
        svc = _service(primary, backups)

        assert svc.last_used is None
        svc.run("q1")
        assert svc.last_used["provider"] == "groq"
        assert svc.last_used["model"] == PRIMARY_MODEL
        assert svc.last_used["fallback"] is False
        svc.run("q2")
        assert svc.last_used["provider"] == "openrouter"
        assert svc.last_used["model"] == BACKUP_1[1]
        assert svc.last_used["fallback"] is True
        assert svc.last_used["at"]


# ═══════════════════════════════════════════════════════════════════════
# FinancialAgentService.stream
# ═══════════════════════════════════════════════════════════════════════


def _model_info(chunk: str) -> dict:
    assert chunk.startswith("[MODEL_INFO]")
    return json.loads(chunk[len("[MODEL_INFO]") :])


class TestStreamFallback:
    def test_status_then_backup_text(self, openrouter_key, _no_retry_sleep):
        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: lambda *a, **k: iter([_chunk("Hi "), _chunk("there")])})
        chunks = list(_service(primary, backups).stream("q"))

        assert chunks[:3] == [BUSY, "Hi ", "there"]
        assert _model_info(chunks[3]) == {
            "provider": "openrouter",
            "model": BACKUP_1[1],
            "fallback": True,
        }
        assert len(chunks) == 4
        _no_retry_sleep.assert_not_called()

    def test_5xx_falls_back(self, openrouter_key):
        primary = _groq_agent(_provider_error(502, "Bad Gateway"))
        backups = _Backups({BACKUP_1: lambda *a, **k: iter([_chunk("ok")])})
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[0] == BUSY
        assert _model_info(chunks[-1])["fallback"] is True

    def test_next_backup_announced(self, openrouter_key):
        primary = _groq_agent(_rate_limited())
        backups = _Backups(
            {BACKUP_1: _rate_limited(), BACKUP_2: lambda *a, **k: iter([_chunk("two")])}
        )
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[1] == "[STATUS] Backup model busy — trying another…"
        assert chunks[2] == "two"
        assert _model_info(chunks[3])["model"] == BACKUP_2[1]

    def test_disabled_keeps_retry_status(self, openrouter_key, monkeypatch, _no_retry_sleep):
        monkeypatch.setenv("FALLBACK_PROVIDER", "none")
        primary = _groq_agent([_rate_limited(), [_chunk("Hello")]])
        backups = _Backups()
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[0].startswith("[STATUS] Rate limit reached — retrying in")
        assert chunks[1] == "Hello"
        assert _model_info(chunks[2]) == {
            "provider": "groq",
            "model": PRIMARY_MODEL,
            "fallback": False,
        }
        assert backups.calls == []

    def test_backups_failing_return_to_groq_retries(self, openrouter_key, _no_retry_sleep):
        primary = _groq_agent([_rate_limited(), [_chunk("groq")]])
        backups = _Backups({BACKUP_1: _rate_limited(), BACKUP_2: _rate_limited()})
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[0] == BUSY
        assert chunks[1] == "[STATUS] Backup model busy — trying another…"
        assert chunks[2].startswith("[STATUS] Rate limit reached — retrying in")
        assert chunks[3] == "groq"
        assert _model_info(chunks[4])["provider"] == "groq"
        _no_retry_sleep.assert_called_once()

    def test_everything_failing_ends_with_error(self, openrouter_key):
        from backend.app.agent.financial_agent import RATE_LIMIT_ERROR_MESSAGE

        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: _rate_limited(), BACKUP_2: _rate_limited()})
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[-1] == f"[ERROR] {RATE_LIMIT_ERROR_MESSAGE}"
        assert sum(c == BUSY for c in chunks) == 1
        assert backups.calls == [BACKUP_1, BACKUP_2]

    def test_backup_failing_mid_answer_is_an_error(self, openrouter_key):
        def partial(*_a, **_k):
            yield _chunk("partial")
            raise _provider_error(502)

        primary = _groq_agent(_rate_limited())
        backups = _Backups({BACKUP_1: partial})
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[1] == "partial"
        assert chunks[-1].startswith("[ERROR]")
        assert backups.calls == [BACKUP_1]  # no second backup after partial output
        assert primary.run.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# API: ChatResponse, SSE metadata, /api/health
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def fallback_client(openrouter_key):
    """TestClient whose agent service fails on Groq and answers from a backup."""
    from api import app

    primary = _groq_agent(_rate_limited())
    backups = _Backups(
        {
            BACKUP_1: lambda *a, stream=False, **k: (
                iter([_chunk("Backup answer")]) if stream else _result("Backup answer")
            )
        }
    )
    svc = _service(primary, backups)
    with (
        patch("backend.app.services.chat.get_agent_service", return_value=svc),
        patch("backend.app.routers.health.get_agent_service", return_value=svc),
    ):
        yield TestClient(app, headers={"X-Client-Id": TEST_CLIENT_ID})


def _sse_events(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


class TestApiProviderInfo:
    def test_chat_response_names_backup(self, fallback_client):
        res = fallback_client.post("/api/chat", json={"message": "What is AAPL at?"})
        assert res.status_code == 200
        body = res.json()
        assert body["response"].startswith("Backup answer")
        assert body["llm_provider"] == "openrouter"
        assert body["llm_model"] == BACKUP_1[1]
        assert body["llm_fallback"] is True

    def test_chat_response_with_plain_mock_has_no_provider(self, client):
        body = client.post("/api/chat", json={"message": "hi"}).json()
        assert body["llm_provider"] is None and body["llm_fallback"] is False

    def test_stream_status_and_model_info(self, fallback_client):
        res = fallback_client.post("/api/chat/stream", json={"message": "What is AAPL at?"})
        events = _sse_events(res.text)
        status = [json.loads(e[len("[STATUS]") :]) for e in events if e.startswith("[STATUS]")]
        assert status == [
            {"type": "status", "message": "Primary model busy — switched to backup model"}
        ]
        info = [
            json.loads(e[len("[MODEL_INFO]") :]) for e in events if e.startswith("[MODEL_INFO]")
        ]
        assert info == [{"provider": "openrouter", "model": BACKUP_1[1], "fallback": True}]
        assert events[-1] == "[DONE]"
        # Model info is metadata only: never part of the answer text.
        final = json.loads(next(e for e in events if e.startswith("[FINAL]"))[len("[FINAL]") :])
        assert "MODEL_INFO" not in final

    def test_health_shows_fallback_config_and_last_provider(self, fallback_client):
        before = fallback_client.get("/api/health").json()
        assert before["fallback"] == {
            "provider": "openrouter",
            "configured": True,
            "models": [BACKUP_1[1], BACKUP_2[1]],
            "chain": [
                {"provider": "openrouter", "model": BACKUP_1[1]},
                {"provider": "openrouter", "model": BACKUP_2[1]},
            ],
            "detail": None,
        }
        assert before["llm_provider"] == "groq"
        assert before["last_llm"] is None

        fallback_client.post("/api/chat", json={"message": "hi"})
        last = fallback_client.get("/api/health").json()["last_llm"]
        assert last["provider"] == "openrouter"
        assert last["model"] == BACKUP_1[1]
        assert last["fallback"] is True

    def test_health_without_keys(self, client):
        body = client.get("/api/health").json()
        assert body["fallback"]["provider"] == "none"
        assert body["fallback"]["configured"] is False

    def test_health_flags_missing_openrouter_key(self, client, monkeypatch):
        monkeypatch.setenv("FALLBACK_PROVIDER", "openrouter")
        fallback = client.get("/api/health").json()["fallback"]
        assert fallback["configured"] is False
        assert fallback["detail"] == "OPENROUTER_API_KEY is not set."
        # Health must never echo secrets.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
        assert "sk-or-secret-value" not in client.get("/api/health").text


# ═══════════════════════════════════════════════════════════════════════
# OpenRouter as primary: OPENROUTER_MODEL → OPENROUTER_FALLBACK_MODELS → Groq
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_for_unit_tests")


class TestProviderDefault:
    def test_no_keys_defaults_to_groq(self):
        from config import get_llm_provider

        assert get_llm_provider() == "groq"

    def test_groq_key_only_defaults_to_groq(self, groq_key):
        from config import get_llm_provider

        assert get_llm_provider() == "groq"

    def test_openrouter_key_defaults_to_openrouter(self, openrouter_primary, groq_key):
        from config import get_llm_provider

        assert get_llm_provider() == "openrouter"

    def test_env_still_selects_groq_or_ollama(self, openrouter_primary, monkeypatch):
        from config import get_llm_provider

        for name in ("groq", "ollama", "openrouter"):
            monkeypatch.setenv("LLM_PROVIDER", name)
            assert get_llm_provider() == name
        monkeypatch.setenv("LLM_PROVIDER", "bogus")
        assert get_llm_provider() == "openrouter"

    def test_chain_openrouter_primary(self, openrouter_primary, groq_key):
        from backend.app.agent.financial_agent import fallback_chain, primary_model_id

        assert primary_model_id() == PRIMARY_MODEL
        assert fallback_chain() == [BACKUP_2, GROQ_BACKUP]

    def test_chain_openrouter_primary_without_groq_key(self, openrouter_primary):
        from backend.app.agent.financial_agent import fallback_chain

        assert fallback_chain() == [BACKUP_2]

    def test_chain_uses_env_models(self, openrouter_primary, groq_key, monkeypatch):
        from backend.app.agent.financial_agent import fallback_chain, primary_model_id

        monkeypatch.setenv("OPENROUTER_MODEL", "a/one")
        monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "b/two,c/three")
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        assert primary_model_id() == "a/one"
        assert fallback_chain() == [
            ("openrouter", "b/two"),
            ("openrouter", "c/three"),
            ("groq", "openai/gpt-oss-20b"),
        ]

    def test_chain_disabled(self, openrouter_primary, groq_key, monkeypatch):
        from backend.app.agent.financial_agent import fallback_chain

        monkeypatch.setenv("FALLBACK_PROVIDER", "none")
        assert fallback_chain() == []

    def test_chain_groq_primary_unchanged(self, openrouter_key, groq_key):
        from backend.app.agent.financial_agent import fallback_chain

        assert fallback_chain() == [BACKUP_1, BACKUP_2]

    def test_create_agent_builds_openrouter_primary(self, openrouter_primary):
        pytest.importorskip("openai")
        from agno.models.openrouter import OpenRouter

        from backend.app.agent.financial_agent import _model_info, create_agent

        agent = create_agent(tools=[])
        assert isinstance(agent.model, OpenRouter)
        assert agent.model.id == PRIMARY_MODEL
        assert agent.model.max_retries == 0
        assert _model_info(agent.model) == {
            "provider": "openrouter",
            "model": PRIMARY_MODEL,
            "fallback": False,
        }

    def test_openrouter_primary_without_key_is_not_configured(self, monkeypatch):
        from backend.app.agent.financial_agent import AgentNotConfiguredError, create_agent

        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        with pytest.raises(AgentNotConfiguredError, match="OPENROUTER_API_KEY"):
            create_agent(tools=[])

    def test_groq_backup_agent_builder(self, openrouter_primary, groq_key):
        from backend.app.agent.financial_agent import FinancialAgentService

        svc = FinancialAgentService()
        backup = svc._fallback_agent(*GROQ_BACKUP, "sess_1")
        assert isinstance(backup.model, Groq)
        assert backup.model.id == GROQ_BACKUP[1]
        assert backup.tools == svc.agent.tools
        assert backup.session_id == "sess_1"


class TestOpenRouterPrimary:
    def test_run_falls_back_to_openrouter_backup_then_groq(
        self, openrouter_primary, groq_key, _no_retry_sleep
    ):
        primary = _openrouter_agent(_rate_limited())
        backups = _Backups({BACKUP_2: _provider_error(503), GROQ_BACKUP: [_result("groq")]})
        reply = _service(primary, backups).run("q")
        assert reply == "groq"
        assert (reply.provider, reply.model, reply.fallback) == (*GROQ_BACKUP, True)
        assert backups.calls == [BACKUP_2, GROQ_BACKUP]
        assert primary.run.call_count == 1
        _no_retry_sleep.assert_not_called()

    def test_timeout_switches_immediately(self, openrouter_primary, _no_retry_sleep):
        primary = _openrouter_agent(httpx.ReadTimeout("timed out"))
        backups = _Backups({BACKUP_2: [_result("gemini")]})
        reply = _service(primary, backups).run("q")
        assert (reply.provider, reply.model) == BACKUP_2
        _no_retry_sleep.assert_not_called()

    def test_primary_success_records_openrouter(self, openrouter_primary):
        primary = _openrouter_agent([_result("primary")])
        backups = _Backups()
        svc = _service(primary, backups)
        reply = svc.run("q")
        assert (reply.provider, reply.model, reply.fallback) == ("openrouter", PRIMARY_MODEL, False)
        assert backups.calls == []

    def test_chain_exhausted_gives_clear_error(self, openrouter_primary, groq_key, _no_retry_sleep):
        from backend.app.agent.financial_agent import LLMRateLimitError

        primary = _openrouter_agent(_rate_limited())
        backups = _Backups({BACKUP_2: _rate_limited(), GROQ_BACKUP: _rate_limited()})
        with pytest.raises(LLMRateLimitError):
            _service(primary, backups).run("q")
        assert primary.run.call_count == 3
        assert backups.calls == [BACKUP_2, GROQ_BACKUP]
        assert _no_retry_sleep.call_count == 2

    def test_stream_status_then_groq(self, openrouter_primary, groq_key, _no_retry_sleep):
        primary = _openrouter_agent(_provider_error(502, "Bad Gateway"))
        backups = _Backups(
            {BACKUP_2: _rate_limited(), GROQ_BACKUP: lambda *a, **k: iter([_chunk("groq")])}
        )
        chunks = list(_service(primary, backups).stream("q"))
        assert chunks[:3] == [BUSY, "[STATUS] Backup model busy — trying another…", "groq"]
        assert "Groq" not in chunks[0]
        assert _model_info(chunks[3]) == {
            "provider": "groq",
            "model": GROQ_BACKUP[1],
            "fallback": True,
        }
        _no_retry_sleep.assert_not_called()


class TestHealthPrimary:
    def test_health_no_keys(self, client):
        body = client.get("/api/health").json()
        assert body["llm_provider"] == "groq"
        assert body["llm_configured"] is False
        assert body["openrouter_configured"] is False
        assert body["fallback"]["chain"] == []

    def test_health_groq_only(self, client, groq_key):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert (body["llm_provider"], body["llm_model"]) == ("groq", PRIMARY_MODEL)
        assert body["llm_configured"] is True
        assert body["fallback"]["provider"] == "none"

    def test_health_both_keys(self, client, openrouter_primary, groq_key):
        res = client.get("/api/health")
        body = res.json()
        assert body["status"] == "ok"
        assert (body["llm_provider"], body["llm_model"]) == ("openrouter", PRIMARY_MODEL)
        assert body["llm_configured"] is True
        assert body["groq_configured"] is True and body["openrouter_configured"] is True
        assert body["fallback"] == {
            "provider": "openrouter,groq",
            "configured": True,
            "models": [BACKUP_2[1], GROQ_BACKUP[1]],
            "chain": [
                {"provider": "openrouter", "model": BACKUP_2[1]},
                {"provider": "groq", "model": GROQ_BACKUP[1]},
            ],
            "detail": None,
        }
        assert "sk-or-test-fake-key" not in res.text and "gsk_test_fake" not in res.text

    def test_health_openrouter_selected_without_key(self, client, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["llm_configured"] is False
        assert body["detail"] == "OPENROUTER_API_KEY is not set."

    def test_health_last_provider_after_backup(self, openrouter_primary, groq_key):
        from api import app

        primary = _openrouter_agent(_rate_limited())
        backups = _Backups({BACKUP_2: [_result("ok")]})
        svc = _service(primary, backups)
        with (
            patch("backend.app.services.chat.get_agent_service", return_value=svc),
            patch("backend.app.routers.health.get_agent_service", return_value=svc),
        ):
            c = TestClient(app, headers={"X-Client-Id": TEST_CLIENT_ID})
            c.post("/api/chat", json={"message": "hi"})
            last = c.get("/api/health").json()["last_llm"]
        assert (last["provider"], last["model"], last["fallback"]) == (*BACKUP_2, True)
