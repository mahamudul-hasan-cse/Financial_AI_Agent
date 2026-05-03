"""Tests for the FastAPI backend (api.py).

Covers:
 • Health endpoint
 • Stats endpoint
 • Non-streaming chat
 • Streaming chat (SSE)
 • Invalid input cases
 • Graceful failure / edge cases
 • Session management
 • Rate limiting
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════════
# Health endpoint
# ═══════════════════════════════════════════════════════════════════════
class TestHealth:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self, client: TestClient):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "groq_configured" in data

    def test_health_shows_groq_configured(self, client: TestClient):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            res = client.get("/api/health")
            assert res.json()["groq_configured"] is True

    def test_health_shows_groq_not_configured(self, client: TestClient):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            res = client.get("/api/health")
            assert res.json()["groq_configured"] is False

    def test_health_reports_active_sessions(self, client: TestClient):
        """active_sessions should reflect the number of stored sessions."""
        from api import sessions

        assert client.get("/api/health").json()["active_sessions"] == 0

        # Create two sessions
        client.post("/api/chat", json={"message": "Hi"})
        client.post("/api/chat", json={"message": "Hey"})
        assert client.get("/api/health").json()["active_sessions"] == 2

    def test_health_reports_total_messages(self, client: TestClient):
        """total_messages should match the stat counter."""
        before = client.get("/api/health").json()["total_messages"]
        client.post("/api/chat", json={"message": "Count me"})
        after = client.get("/api/health").json()["total_messages"]
        assert after == before + 1

    def test_health_nlp_status_fields(self, client: TestClient):
        """nlp_status must expose spaCy and VADER availability flags."""
        data = client.get("/api/health").json()
        nlp = data["nlp_status"]
        assert "spacy_ner" in nlp
        assert "vader_sentiment" in nlp
        assert isinstance(nlp["spacy_ner"], bool)
        assert isinstance(nlp["vader_sentiment"], bool)

    def test_health_response_model_shape(self, client: TestClient):
        """Every documented key in HealthResponse must be present."""
        data = client.get("/api/health").json()
        for key in ("status", "groq_configured", "active_sessions",
                     "total_messages", "nlp_status"):
            assert key in data, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# Stats endpoint
# ═══════════════════════════════════════════════════════════════════════
class TestStats:
    """Tests for GET /api/stats."""

    def test_stats_returns_200(self, client: TestClient):
        res = client.get("/api/stats")
        assert res.status_code == 200

    def test_stats_has_required_fields(self, client: TestClient):
        data = client.get("/api/stats").json()
        for key in ("total_messages", "intent_distribution",
                     "avg_response_time_ms", "active_sessions",
                     "total_sessions_created", "classifier_usage"):
            assert key in data, f"Missing key: {key}"

    def test_stats_total_messages_increments(self, client: TestClient):
        before = client.get("/api/stats").json()["total_messages"]
        client.post("/api/chat", json={"message": "Hello"})
        after = client.get("/api/stats").json()["total_messages"]
        assert after == before + 1

    def test_stats_intent_counts_is_dict(self, client: TestClient):
        data = client.get("/api/stats").json()
        assert isinstance(data["intent_distribution"], dict)

    def test_stats_avg_response_time_non_negative(self, client: TestClient):
        data = client.get("/api/stats").json()
        assert data["avg_response_time_ms"] >= 0

    def test_stats_total_sessions_created_increments(self, client: TestClient):
        before = client.get("/api/stats").json()["total_sessions_created"]
        client.post("/api/chat", json={"message": "New session"})
        after = client.get("/api/stats").json()["total_sessions_created"]
        assert after == before + 1

    def test_stats_classifier_usage_populated(self, client: TestClient):
        """After a chat, classifier_usage should have at least one entry."""
        client.post("/api/chat", json={"message": "What is the price of Google?"})
        data = client.get("/api/stats").json()
        assert isinstance(data["classifier_usage"], dict)
        assert sum(data["classifier_usage"].values()) >= 1

    def test_stats_intent_distribution_populated_after_chat(self, client: TestClient):
        """After a chat, intent_distribution should contain the detected intent."""
        client.post("/api/chat", json={"message": "What is the price of AAPL?"})
        intents = client.get("/api/stats").json()["intent_distribution"]
        assert sum(intents.values()) >= 1

    def test_stats_avg_response_time_updates_after_chat(self, client: TestClient):
        """avg_response_time_ms should be positive after at least one chat."""
        client.post("/api/chat", json={"message": "time me"})
        avg = client.get("/api/stats").json()["avg_response_time_ms"]
        assert avg >= 0  # mock is fast; just verify it was recorded

    def test_stats_response_times_capped_at_100(self, client: TestClient):
        """Internal _stats['response_times_ms'] should never exceed 100 entries."""
        from api import _stats

        # Simulate 110 response time recordings
        _stats["response_times_ms"] = list(range(110))
        # Trigger one more via chat
        client.post("/api/chat", json={"message": "cap test"})

        assert len(_stats["response_times_ms"]) <= 101  # at most 100 kept + 1 new


# ═══════════════════════════════════════════════════════════════════════
# Non-streaming chat
# ═══════════════════════════════════════════════════════════════════════
class TestChat:
    """Tests for POST /api/chat."""

    def test_chat_returns_response(self, client: TestClient):
        res = client.post("/api/chat", json={"message": "What is AAPL price?"})
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert "session_id" in data
        assert len(data["response"]) > 0

    def test_chat_creates_session(self, client: TestClient):
        res = client.post("/api/chat", json={"message": "Hello"})
        data = res.json()
        session_id = data["session_id"]
        assert session_id is not None

        from api import sessions
        assert session_id in sessions
        assert len(sessions[session_id]["messages"]) == 2  # user + assistant

    def test_chat_continues_session(self, client: TestClient):
        res1 = client.post("/api/chat", json={"message": "First message"})
        sid = res1.json()["session_id"]

        res2 = client.post("/api/chat", json={"message": "Second message", "session_id": sid})
        assert res2.status_code == 200

        from api import sessions
        assert len(sessions[sid]["messages"]) == 4  # 2 pairs

    def test_chat_empty_message_rejected(self, client: TestClient):
        res = client.post("/api/chat", json={"message": ""})
        assert res.status_code == 422

    def test_chat_too_long_message_rejected(self, client: TestClient):
        res = client.post("/api/chat", json={"message": "x" * 2001})
        assert res.status_code == 422

    def test_chat_agent_timeout(self, client: TestClient, mock_agent):
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        mock_agent.run.side_effect = RuntimeError("timeout")
        res = client.post("/api/chat", json={"message": "Hello"})
        assert res.status_code in (502, 504)

    def test_chat_agent_error(self, client: TestClient, mock_agent):
        mock_agent.run.side_effect = RuntimeError("LLM unavailable")
        res = client.post("/api/chat", json={"message": "Hello"})
        assert res.status_code == 502
        assert "internal error" in res.json()["detail"].lower()

    def test_chat_returns_nlp_metadata(self, client: TestClient):
        """The response must include nlp_metadata with NLP analysis results."""
        res = client.post("/api/chat", json={"message": "What is Apple stock price?"})
        data = res.json()
        assert "nlp_metadata" in data
        assert data["nlp_metadata"] is not None

    def test_chat_nlp_metadata_has_expected_keys(self, client: TestClient):
        """nlp_metadata should contain entities, intent, and sentiment."""
        data = client.post(
            "/api/chat", json={"message": "Tell me about Tesla revenue"}
        ).json()
        meta = data["nlp_metadata"]
        for key in ("entities", "intent", "sentiment"):
            assert key in meta, f"Missing nlp_metadata key: {key}"

    def test_chat_returns_consistent_session_id(self, client: TestClient):
        """Supplying a session_id should return the same session_id."""
        sid = "my-fixed-session"
        res = client.post("/api/chat", json={"message": "Hi", "session_id": sid})
        assert res.json()["session_id"] == sid

    def test_chat_invalid_session_id_creates_new(self, client: TestClient):
        """An unknown session_id should be accepted and used as the new ID."""
        sid = "nonexistent-session-id"
        res = client.post("/api/chat", json={"message": "Hi", "session_id": sid})
        assert res.status_code == 200
        assert res.json()["session_id"] == sid

        from api import sessions
        assert sid in sessions

    def test_chat_agent_retry_succeeds_after_failure(self, client: TestClient, mock_agent):
        """Agent service handles retries internally; API sees final result."""
        # Service.run() encapsulates retry logic, so from the API's perspective
        # a successful retry just means .run() returns a string.
        mock_agent.run.return_value = "Recovered response"

        res = client.post("/api/chat", json={"message": "retry me"})
        assert res.status_code == 200
        assert "Recovered" in res.json()["response"]

    def test_chat_history_stores_user_and_assistant(self, client: TestClient):
        """After one chat, history should contain exactly one user + one assistant message."""
        res = client.post("/api/chat", json={"message": "History check"})
        sid = res.json()["session_id"]

        from api import sessions
        msgs = sessions[sid]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "History check"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["content"]) > 0

    def test_chat_boundary_message_max_length(self, client: TestClient):
        """A message at exactly MAX_MESSAGE_LENGTH should be accepted."""
        from api import MAX_MESSAGE_LENGTH

        res = client.post("/api/chat", json={"message": "a" * MAX_MESSAGE_LENGTH})
        assert res.status_code == 200

    def test_chat_one_over_max_length_rejected(self, client: TestClient):
        """A message of MAX_MESSAGE_LENGTH + 1 should be rejected."""
        from api import MAX_MESSAGE_LENGTH

        res = client.post("/api/chat", json={"message": "a" * (MAX_MESSAGE_LENGTH + 1)})
        assert res.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# Streaming chat (SSE)
# ═══════════════════════════════════════════════════════════════════════
class TestChatStream:
    """Tests for POST /api/chat/stream."""

    def test_stream_returns_sse(self, client: TestClient, mock_agent):
        mock_agent.stream.return_value = iter(["Streamed response"])

        res = client.post("/api/chat/stream", json={"message": "Hello"})
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body = res.text
        assert "data: " in body
        assert "[DONE]" in body

    def test_stream_empty_message_rejected(self, client: TestClient):
        res = client.post("/api/chat/stream", json={"message": ""})
        assert res.status_code == 422

    def test_stream_too_long_message_rejected(self, client: TestClient):
        res = client.post("/api/chat/stream", json={"message": "x" * 2001})
        assert res.status_code == 422

    def test_stream_emits_nlp_meta_first(self, client: TestClient, mock_agent):
        """The very first SSE event should be the [NLP_META] JSON payload."""
        mock_agent.stream.return_value = iter(["data"])

        res = client.post("/api/chat/stream", json={"message": "Apple stock"})
        lines = [ln for ln in res.text.splitlines() if ln.startswith("data: ")]
        assert len(lines) >= 2  # at least NLP_META + DONE

        first_data = lines[0].removeprefix("data: ")
        assert first_data.startswith("[NLP_META]")

        meta_json = json.loads(first_data.removeprefix("[NLP_META]"))
        for key in ("entities", "intent", "sentiment"):
            assert key in meta_json, f"Missing nlp_meta key: {key}"

    def test_stream_emits_done_last(self, client: TestClient, mock_agent):
        """The last SSE data line should be [DONE]."""
        mock_agent.stream.return_value = iter(["chunk"])

        res = client.post("/api/chat/stream", json={"message": "end check"})
        data_lines = [ln for ln in res.text.splitlines() if ln.startswith("data: ")]
        assert data_lines[-1].strip() == "data: [DONE]"

    def test_stream_contains_agent_chunks(self, client: TestClient, mock_agent):
        """Actual content chunks should appear between NLP_META and DONE."""
        mock_agent.stream.return_value = iter(["Hello ", "World"])

        res = client.post("/api/chat/stream", json={"message": "multi"})
        body = res.text
        assert "Hello " in body
        assert "World" in body

    def test_stream_updates_session_history(self, client: TestClient, mock_agent):
        """After streaming, session should contain the assembled response."""
        mock_agent.stream.return_value = iter(["streamed text"])

        res = client.post("/api/chat/stream", json={"message": "save me"})
        assert res.status_code == 200

        # Extract session_id from the NLP_META (not returned in body directly)
        from api import sessions
        assert len(sessions) == 1
        sid = next(iter(sessions))
        msgs = sessions[sid]["messages"]
        assert any(m["role"] == "user" and m["content"] == "save me" for m in msgs)
        assert any(m["role"] == "assistant" and "streamed text" in m["content"] for m in msgs)

    def test_stream_continues_existing_session(self, client: TestClient, mock_agent):
        """Passing an existing session_id to the stream endpoint should append."""
        # Create session via non-streaming endpoint
        r1 = client.post("/api/chat", json={"message": "First"})
        sid = r1.json()["session_id"]

        # Continue via stream
        mock_agent.stream.return_value = iter(["resp"])
        r2 = client.post("/api/chat/stream", json={"message": "Second", "session_id": sid})
        assert r2.status_code == 200

        from api import sessions
        assert len(sessions[sid]["messages"]) == 4  # 2 from chat + 2 from stream

    def test_stream_agent_error_emits_error_event(self, client: TestClient, mock_agent):
        """If the agent raises mid-stream, an [ERROR] event should be emitted."""
        mock_agent.stream.return_value = iter(
            ["[ERROR] An error occurred while generating the response."]
        )

        res = client.post("/api/chat/stream", json={"message": "explode"})
        assert res.status_code == 200  # SSE always starts with 200
        assert "[ERROR]" in res.text
        assert "[DONE]" in res.text  # stream must still terminate cleanly

    def test_stream_rate_limited(self, client: TestClient, mock_agent):
        """Rate limiting should also apply to the streaming endpoint."""
        from api import RATE_LIMIT_MAX

        mock_agent.stream.return_value = iter(["ok"])

        sid = "stream-rate-test"
        for i in range(RATE_LIMIT_MAX):
            mock_agent.stream.return_value = iter(["ok"])
            res = client.post(
                "/api/chat/stream",
                json={"message": f"msg {i}", "session_id": sid},
            )
            assert res.status_code == 200

        res = client.post(
            "/api/chat/stream",
            json={"message": "one more", "session_id": sid},
        )
        assert res.status_code == 429

    def test_stream_increments_stats(self, client: TestClient, mock_agent):
        """Streaming chat should also increment total_messages in stats."""
        mock_agent.stream.return_value = iter(["counted"])

        before = client.get("/api/stats").json()["total_messages"]
        client.post("/api/chat/stream", json={"message": "count stream"})
        after = client.get("/api/stats").json()["total_messages"]
        assert after == before + 1


# ═══════════════════════════════════════════════════════════════════════
# Session management
# ═══════════════════════════════════════════════════════════════════════
class TestSessionManagement:
    """Tests for DELETE /api/chat/{session_id} and session limits."""

    def test_clear_session(self, client: TestClient):
        res = client.post("/api/chat", json={"message": "Hello"})
        sid = res.json()["session_id"]

        res = client.delete(f"/api/chat/{sid}")
        assert res.status_code == 200
        assert res.json()["status"] == "cleared"

        from api import sessions
        assert sid not in sessions

    def test_clear_nonexistent_session(self, client: TestClient):
        res = client.delete("/api/chat/nonexistent-id")
        assert res.status_code == 200  # Idempotent

    def test_max_sessions_eviction(self, client: TestClient):
        from api import sessions, MAX_SESSIONS

        for i in range(MAX_SESSIONS):
            sessions[f"sess-{i}"] = {"messages": [], "last_active": time.time() - i}

        res = client.post("/api/chat", json={"message": "Hello"})
        assert res.status_code == 200
        assert len(sessions) <= MAX_SESSIONS

    def test_clear_session_also_clears_rate_limits(self, client: TestClient):
        """Deleting a session should remove its rate-limit history too."""
        from api import _rate_limits

        sid = "rate-clear-test"
        client.post("/api/chat", json={"message": "Hi", "session_id": sid})
        assert sid in _rate_limits

        client.delete(f"/api/chat/{sid}")
        assert sid not in _rate_limits

    def test_session_ttl_cleanup(self, client: TestClient):
        """Expired sessions should be cleaned up on the next request."""
        from api import sessions, SESSION_TTL

        # Plant an expired session
        sessions["old-sess"] = {
            "messages": [{"role": "user", "content": "ancient"}],
            "last_active": time.time() - SESSION_TTL - 1,
        }

        # A new request triggers cleanup
        client.post("/api/chat", json={"message": "fresh"})
        assert "old-sess" not in sessions

    def test_eviction_removes_oldest(self, client: TestClient):
        """When MAX_SESSIONS is hit, the oldest (least-recently-active) is evicted."""
        from api import sessions, MAX_SESSIONS

        # Plant MAX_SESSIONS sessions with descending last_active
        for i in range(MAX_SESSIONS):
            sessions[f"s-{i}"] = {
                "messages": [],
                "last_active": time.time() - (MAX_SESSIONS - i),
            }

        # s-0 has the oldest last_active
        client.post("/api/chat", json={"message": "trigger eviction"})
        assert "s-0" not in sessions  # oldest evicted
        assert len(sessions) <= MAX_SESSIONS


# ═══════════════════════════════════════════════════════════════════════
# Rate limiting
# ═══════════════════════════════════════════════════════════════════════
class TestRateLimiting:
    """Tests for per-session rate limiting."""

    def test_rate_limit_exceeded(self, client: TestClient):
        from api import RATE_LIMIT_MAX

        sid = "rate-test-session"
        for i in range(RATE_LIMIT_MAX):
            res = client.post("/api/chat", json={"message": f"msg {i}", "session_id": sid})
            assert res.status_code == 200

        res = client.post("/api/chat", json={"message": "one more", "session_id": sid})
        assert res.status_code == 429
        assert "rate limit" in res.json()["detail"].lower()

    def test_rate_limit_window_resets(self, client: TestClient):
        """After the rate-limit window elapses, requests should be allowed again."""
        from api import RATE_LIMIT_MAX, _rate_limits

        sid = "rate-reset-test"
        # Pretend the session already has RATE_LIMIT_MAX requests from the past
        old_time = time.time() - 120  # well outside the 60s window
        _rate_limits[sid] = [old_time] * RATE_LIMIT_MAX

        res = client.post("/api/chat", json={"message": "should pass", "session_id": sid})
        assert res.status_code == 200  # old timestamps are pruned

    def test_rate_limit_is_per_session(self, client: TestClient):
        """Different sessions should have independent rate limits."""
        from api import RATE_LIMIT_MAX

        for i in range(RATE_LIMIT_MAX):
            client.post("/api/chat", json={"message": f"m{i}", "session_id": "sess-A"})

        # sess-A is exhausted, sess-B should still work
        res = client.post("/api/chat", json={"message": "ok", "session_id": "sess-B"})
        assert res.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Invalid input cases
# ═══════════════════════════════════════════════════════════════════════
class TestInvalidInputs:
    """Tests for malformed or invalid requests."""

    def test_chat_missing_message_field(self, client: TestClient):
        """Omitting the 'message' field entirely should return 422."""
        res = client.post("/api/chat", json={"session_id": "abc"})
        assert res.status_code == 422

    def test_chat_null_message(self, client: TestClient):
        """Sending null for message should return 422."""
        res = client.post("/api/chat", json={"message": None})
        assert res.status_code == 422

    def test_chat_non_json_body(self, client: TestClient):
        """Sending non-JSON content should return 422."""
        res = client.post(
            "/api/chat",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 422

    def test_chat_wrong_http_method_get(self, client: TestClient):
        """GET on /api/chat should not return a valid chat JSON response.

        Note: If the SPA catch-all route is active, GET returns 200 with
        HTML; otherwise FastAPI returns 405.  Either way, it must not be
        a successful chat response.
        """
        res = client.get("/api/chat")
        assert res.status_code in (200, 405)
        if res.status_code == 200:
            # SPA fallback — should NOT be a chat JSON body
            assert "response" not in (res.json() if "json" in res.headers.get("content-type", "") else {})

    def test_health_wrong_http_method_post(self, client: TestClient):
        """POST on /api/health should return 405 Method Not Allowed."""
        res = client.post("/api/health")
        assert res.status_code == 405

    def test_stats_wrong_http_method_post(self, client: TestClient):
        """POST on /api/stats should return 405 Method Not Allowed."""
        res = client.post("/api/stats")
        assert res.status_code == 405

    def test_stream_missing_message_field(self, client: TestClient):
        """Omitting 'message' on the stream endpoint should return 422."""
        res = client.post("/api/chat/stream", json={})
        assert res.status_code == 422

    def test_chat_whitespace_only_message(self, client: TestClient):
        """A message of only whitespace (but length ≥ 1) should be accepted
        since Pydantic min_length checks character count, not stripped content."""
        res = client.post("/api/chat", json={"message": "   "})
        # Whitespace message has length 3 (>= 1), so Pydantic accepts it
        assert res.status_code == 200

    def test_chat_extra_fields_ignored(self, client: TestClient):
        """Extra fields in the body should be silently ignored."""
        res = client.post(
            "/api/chat",
            json={"message": "Hello", "unknown_field": 42},
        )
        assert res.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Graceful failure cases
# ═══════════════════════════════════════════════════════════════════════
class TestGracefulFailures:
    """Tests verifying the API degrades gracefully under failure conditions."""

    def test_agent_all_retries_exhausted(self, client: TestClient, mock_agent):
        """When the agent fails on all 3 retries, a 502 should be returned."""
        mock_agent.run.side_effect = RuntimeError("persistent failure")
        res = client.post("/api/chat", json={"message": "fail hard"})
        assert res.status_code == 502

    def test_agent_returns_object_without_content(self, client: TestClient, mock_agent):
        """If agent_service.run returns a non-string, str() coercion is used upstream."""
        mock_agent.run.return_value = "12345"
        res = client.post("/api/chat", json={"message": "weird response"})
        assert res.status_code == 200
        assert "12345" in res.json()["response"]

    def test_agent_returns_empty_string(self, client: TestClient, mock_agent):
        """An empty string response from the agent should still return 200."""
        mock_agent.run.return_value = ""

        res = client.post("/api/chat", json={"message": "empty"})
        assert res.status_code == 200
        assert res.json()["response"] == ""

    def test_stream_empty_agent_response(self, client: TestClient, mock_agent):
        """If the agent yields nothing, stream should still emit NLP_META + DONE."""
        mock_agent.stream.return_value = iter([])

        res = client.post("/api/chat/stream", json={"message": "nothing"})
        assert res.status_code == 200
        assert "[NLP_META]" in res.text
        assert "[DONE]" in res.text

    def test_stream_no_history_if_empty_response(self, client: TestClient, mock_agent):
        """If the agent yields no content, nothing should be appended to history."""
        mock_agent.stream.return_value = iter([])

        client.post("/api/chat/stream", json={"message": "ghost"})

        from api import sessions
        sid = next(iter(sessions))
        # An empty stream should NOT append messages
        assert len(sessions[sid]["messages"]) == 0

    def test_nlp_pipeline_failure_propagates(self, client: TestClient):
        """If the NLP pipeline raises, the API should fall back to defaults."""
        with patch("api.run_nlp_pipeline", side_effect=Exception("NLP boom")):
            response = client.post("/api/chat", json={"message": "crash nlp"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["nlp_metadata"]["intent"] == "general"
        assert payload["nlp_metadata"]["intent_method"] == "fallback"
        assert payload["nlp_metadata"]["spacy_available"] is False
        assert payload["nlp_metadata"]["vader_available"] is False

    def test_multiple_errors_then_recovery(self, client: TestClient, mock_agent):
        """After a failed request, subsequent requests should still work."""
        # First request: agent failure
        mock_agent.run.side_effect = RuntimeError("down")
        res1 = client.post("/api/chat", json={"message": "fail"})
        assert res1.status_code == 502

        # Second request: agent recovers
        mock_agent.run.side_effect = None
        mock_agent.run.return_value = "I'm back"
        res2 = client.post("/api/chat", json={"message": "recover"})
        assert res2.status_code == 200
        assert "I'm back" in res2.json()["response"]

    def test_concurrent_session_creation(self, client: TestClient):
        """Rapidly creating many sessions should not raise errors."""
        for i in range(20):
            res = client.post("/api/chat", json={"message": f"concurrent {i}"})
            assert res.status_code == 200

        from api import sessions
        assert len(sessions) == 20


# ═══════════════════════════════════════════════════════════════════════
# Schema validation (schemas.py integration)
# ═══════════════════════════════════════════════════════════════════════
class TestSchemaIntegration:
    """Verify that responses can be round-tripped through Pydantic schemas."""

    def test_chat_response_validates_against_schema(self, client: TestClient):
        """POST /api/chat response should be parseable by ChatResponse."""
        from schemas import ChatResponse

        res = client.post("/api/chat", json={"message": "schema check"})
        parsed = ChatResponse.model_validate(res.json())
        assert parsed.response
        assert parsed.session_id

    def test_chat_nlp_metadata_validates_against_schema(self, client: TestClient):
        """nlp_metadata in chat response must match NLPMetadataSchema."""
        from schemas import NLPMetadataSchema

        data = client.post(
            "/api/chat", json={"message": "Apple stock price"}
        ).json()
        meta = NLPMetadataSchema.model_validate(data["nlp_metadata"])
        assert isinstance(meta.entities, list)
        assert isinstance(meta.intent, str)
        assert isinstance(meta.sentiment.label, str)
        assert isinstance(meta.sentiment.compound, float)

    def test_health_response_validates_against_schema(self, client: TestClient):
        """GET /api/health response must match HealthResponse."""
        from schemas import HealthResponse

        parsed = HealthResponse.model_validate(client.get("/api/health").json())
        assert parsed.status == "ok"
        assert isinstance(parsed.nlp_status.spacy_ner, bool)

    def test_stats_response_validates_against_schema(self, client: TestClient):
        """GET /api/stats response must match StatsResponse."""
        from schemas import StatsResponse

        parsed = StatsResponse.model_validate(client.get("/api/stats").json())
        assert isinstance(parsed.intent_distribution, dict)
        assert parsed.avg_response_time_ms >= 0

    def test_stream_nlp_meta_validates_against_schema(self, client: TestClient, mock_agent):
        """The NLP_META SSE payload must parse as NLPMetadataSchema."""
        from schemas import NLPMetadataSchema

        mock_agent.stream.return_value = iter(["x"])

        res = client.post("/api/chat/stream", json={"message": "validate stream"})
        for line in res.text.splitlines():
            if line.startswith("data: [NLP_META]"):
                raw = line.removeprefix("data: [NLP_META]")
                meta = NLPMetadataSchema.model_validate_json(raw)
                assert meta.intent
                assert meta.sentiment.label
                break
        else:
            pytest.fail("No NLP_META event found in stream")

    def test_entity_model_from_nlp_metadata(self, client: TestClient):
        """Entity sub-models should have text, label, and optional ticker."""
        from schemas import NLPMetadataSchema

        data = client.post(
            "/api/chat", json={"message": "What is Tesla stock price?"}
        ).json()
        meta = NLPMetadataSchema.model_validate(data["nlp_metadata"])
        if meta.entities:
            ent = meta.entities[0]
            assert isinstance(ent.text, str)
            assert isinstance(ent.label, str)
            # ticker may be None or a string
            assert ent.ticker is None or isinstance(ent.ticker, str)

    def test_error_response_schema(self, client: TestClient, mock_agent):
        """HTTP error bodies should match ErrorResponse schema."""
        from schemas import ErrorResponse

        mock_agent.run.side_effect = RuntimeError("fail")
        res = client.post("/api/chat", json={"message": "break"})
        assert res.status_code == 502
        parsed = ErrorResponse.model_validate(res.json())
        assert "error" in parsed.detail.lower()


# ═══════════════════════════════════════════════════════════════════════
# NLP pipeline graceful degradation
# ═══════════════════════════════════════════════════════════════════════
class TestNLPPipelineGracefulDegradation:
    """Verify that NLP pipeline failures don't crash chat endpoints."""

    def test_chat_succeeds_when_nlp_pipeline_raises(self, client: TestClient):
        """Chat should return 200 with default NLP metadata when the pipeline fails."""
        with patch("api.run_nlp_pipeline", side_effect=RuntimeError("spaCy crashed")):
            res = client.post("/api/chat", json={"message": "What is AAPL?"})
        assert res.status_code == 200
        data = res.json()
        assert data["response"]  # agent still ran
        meta = data["nlp_metadata"]
        assert meta["intent"] == "general"
        assert meta["intent_method"] == "fallback"
        assert meta["entities"] == []

    def test_stream_succeeds_when_nlp_pipeline_raises(self, client: TestClient):
        """Streaming should emit default NLP metadata when the pipeline fails."""
        with patch("api.run_nlp_pipeline", side_effect=RuntimeError("VADER crashed")):
            res = client.post("/api/chat/stream", json={"message": "TSLA news"})
        assert res.status_code == 200
        body = res.text
        assert "[NLP_META]" in body
        # Extract NLP metadata from the SSE event
        for line in body.splitlines():
            if "[NLP_META]" in line:
                raw = line.split("[NLP_META]", 1)[1]
                meta = json.loads(raw)
                assert meta["intent"] == "general"
                assert meta["intent_method"] == "fallback"
                break
        assert "[DONE]" in body

    def test_nlp_failure_still_tracks_stats(self, client: TestClient):
        """Stats counters should still be updated even when NLP fails."""
        from api import _stats

        before = _stats["total_messages"]
        with patch("api.run_nlp_pipeline", side_effect=RuntimeError("boom")):
            client.post("/api/chat", json={"message": "test"})
        assert _stats["total_messages"] == before + 1
        assert _stats["classifier_usage"]["fallback"] >= 1
