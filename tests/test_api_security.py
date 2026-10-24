"""Tests for session ownership, input validation, routing and streaming contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import OTHER_CLIENT_ID, TEST_CLIENT_ID


def _sse_events(body: str) -> list[str]:
    return [ln.removeprefix("data: ") for ln in body.splitlines() if ln.startswith("data: ")]


# ═══════════════════════════════════════════════════════════════════════
# Test isolation
# ═══════════════════════════════════════════════════════════════════════
class TestDatabaseIsolation:
    def test_tests_use_temporary_database(self):
        import persistence

        default_db = Path(__file__).resolve().parent.parent / "data" / "financial_agent.db"
        assert persistence.DB_PATH.resolve() != default_db.resolve()
        assert str(persistence.DB_PATH) == os.environ["DATABASE_PATH"]

    def test_reset_refuses_default_database(self, monkeypatch):
        import persistence

        monkeypatch.setattr(persistence, "DB_PATH", persistence._DEFAULT_DB_PATH)
        with pytest.raises(RuntimeError, match="refused"):
            persistence.reset_for_tests()

    def test_groq_key_not_required_for_import(self):
        assert not os.environ.get("GROQ_API_KEY")
        import api  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Client ownership
# ═══════════════════════════════════════════════════════════════════════
class TestSessionOwnership:
    def test_missing_client_id_rejected(self, mock_agent):
        from api import app

        res = TestClient(app).post("/api/chat", json={"message": "hi"})
        assert res.status_code == 400
        assert "X-Client-Id" in res.json()["detail"]

    def test_invalid_client_id_rejected(self, mock_agent):
        from api import app

        res = TestClient(app, headers={"X-Client-Id": "short"}).get("/api/sessions")
        assert res.status_code == 400

    def test_sessions_listed_only_for_owner(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "Mine"}).json()["session_id"]
        other_client.post("/api/chat", json={"message": "Theirs"})

        mine = client.get("/api/sessions").json()["sessions"]
        theirs = other_client.get("/api/sessions").json()["sessions"]
        assert [s["session_id"] for s in mine] == [sid]
        assert len(theirs) == 1 and theirs[0]["session_id"] != sid

    def test_cannot_read_foreign_session(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "secret"}).json()["session_id"]
        assert client.get(f"/api/sessions/{sid}").status_code == 200
        assert other_client.get(f"/api/sessions/{sid}").status_code == 404

    def test_cannot_rename_or_delete_foreign_session(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "secret"}).json()["session_id"]
        assert other_client.patch(f"/api/sessions/{sid}", json={"title": "pwned"}).status_code == 404
        assert other_client.delete(f"/api/chat/{sid}").status_code == 404
        assert client.get(f"/api/sessions/{sid}").json()["title"] != "pwned"

    def test_cannot_continue_foreign_session(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        res = other_client.post("/api/chat", json={"message": "hijack", "session_id": sid})
        assert res.status_code == 404
        res = other_client.post("/api/chat/stream", json={"message": "hijack", "session_id": sid})
        assert res.status_code == 404

    def test_owner_can_rename(self, client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        res = client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"})
        assert res.status_code == 200
        assert res.json()["title"] == "Renamed"
        assert res.json()["preview"] == "first"

    def test_foreign_alerts_and_watchlist_hidden(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        created = client.post(
            "/api/alerts",
            json={"session_id": sid, "symbol": "AAPL", "alert_type": "daily_digest"},
        )
        assert created.status_code == 200
        alert_id = created.json()["id"]

        assert other_client.get(f"/api/alerts/{sid}").status_code == 404
        assert other_client.get(f"/api/watchlist/{sid}").status_code == 404
        assert other_client.patch(f"/api/alerts/item/{alert_id}", json={"enabled": False}).status_code == 404
        assert other_client.delete(f"/api/alerts/item/{alert_id}").status_code == 404

        res = client.patch(f"/api/alerts/item/{alert_id}", json={"enabled": False})
        assert res.status_code == 200 and res.json()["enabled"] is False


# ═══════════════════════════════════════════════════════════════════════
# Session id validation / PDF export path traversal
# ═══════════════════════════════════════════════════════════════════════
class TestSessionIdValidation:
    @pytest.mark.parametrize("bad", ["/../../x", "../etc", "a/b", "a.b", "x" * 65, "sp ace", ""])
    def test_chat_rejects_bad_session_id(self, client, bad):
        res = client.post("/api/chat", json={"message": "hi", "session_id": bad})
        assert res.status_code == 422

    @pytest.mark.parametrize("bad", ["/../../x", "..", "a.b"])
    def test_export_rejects_bad_session_id(self, client, bad):
        res = client.post(
            "/api/chat/export",
            json={"session_id": bad, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 422

    def test_path_param_rejects_dots(self, client):
        assert client.get("/api/sessions/a.b").status_code == 422

    def test_export_writes_nothing_under_outputs(self, client):
        from config import settings

        before = set(settings.outputs_dir.rglob("*.pdf"))
        res = client.post(
            "/api/chat/export",
            json={"session_id": "sess_abc-123", "messages": [{"role": "user", "content": "<b>hi</b>"}]},
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF")
        assert 'filename="chat_export_sess_abc-123.pdf"' in res.headers["content-disposition"]
        assert set(settings.outputs_dir.rglob("*.pdf")) == before

    def test_export_of_foreign_session_rejected(self, client, other_client):
        sid = client.post("/api/chat", json={"message": "secret"}).json()["session_id"]
        res = other_client.post("/api/chat/export", json={"session_id": sid})
        assert res.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# Routing, CORS, degraded mode
# ═══════════════════════════════════════════════════════════════════════
class TestRoutingAndCors:
    def test_unknown_api_path_is_json_404(self, client):
        res = client.get("/api/does-not-exist")
        assert res.status_code == 404
        assert res.headers["content-type"].startswith("application/json")

    def test_cors_allows_patch_and_client_header(self, client):
        res = client.options(
            "/api/sessions/abc",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "X-Client-Id, Content-Type",
            },
        )
        assert res.status_code == 200
        assert "PATCH" in res.headers["access-control-allow-methods"]
        assert "x-client-id" in res.headers["access-control-allow-headers"].lower()

    def test_chat_returns_503_without_llm(self, monkeypatch):
        """Real (unmocked) service + no GROQ_API_KEY → 503, not a crash."""
        from api import app
        from backend.app.agent.financial_agent import FinancialAgentService

        monkeypatch.setenv("GROQ_API_KEY", "")
        with patch("backend.app.services.chat.get_agent_service", return_value=FinancialAgentService()):
            c = TestClient(app, headers={"X-Client-Id": TEST_CLIENT_ID})
            assert c.post("/api/chat", json={"message": "hi"}).status_code == 503
            assert c.post("/api/chat/stream", json={"message": "hi"}).status_code == 503

    def test_rate_limit_uses_client_ip(self, client):
        from api import IP_RATE_LIMIT_MAX, _ip_rate_limits

        _ip_rate_limits["testclient"] = [__import__("time").time()] * IP_RATE_LIMIT_MAX
        res = client.post("/api/chat", json={"message": "hi"})
        assert res.status_code == 429
        assert "IP rate limit" in res.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════
# Streaming contract
# ═══════════════════════════════════════════════════════════════════════
class TestStreamingContract:
    def test_event_order(self, client, mock_agent):
        mock_agent.stream.return_value = iter(["Hello ", "world"])
        events = _sse_events(client.post("/api/chat/stream", json={"message": "hi"}).text)
        assert events[0].startswith("[NLP_META]")
        assert [json.loads(e) for e in events[1:3]] == ["Hello ", "world"]
        assert events[3].startswith("[FINAL]")
        assert events[4].startswith("[RESEARCH_DATA]")
        assert events[-1] == "[DONE]"

    def test_final_matches_persisted_text(self, client, mock_agent):
        mock_agent.stream.return_value = iter(["Quick Take: price is up"])
        events = _sse_events(client.post("/api/chat/stream", json={"message": "hi"}).text)
        final = json.loads(next(e for e in events if e.startswith("[FINAL]"))[len("[FINAL]"):])
        from api import sessions

        stored = next(iter(sessions.values()))["messages"][-1]["content"]
        assert final == stored
        assert final.startswith("price is up")

    def test_retry_before_output_does_not_duplicate(self, client, mock_agent):
        mock_agent.stream.side_effect = [iter(["[ERROR] boom"]), iter(["Recovered"])]
        events = _sse_events(client.post("/api/chat/stream", json={"message": "hi"}).text)
        chunks = [json.loads(e) for e in events if e.startswith('"')]
        assert chunks == ["Recovered"]
        assert not any(e.startswith("[ERROR]") for e in events)
        assert mock_agent.stream.call_count == 2

    def test_no_retry_after_partial_output(self, client, mock_agent):
        mock_agent.stream.side_effect = [iter(["Partial ", "[ERROR] boom"]), iter(["dup"])]
        events = _sse_events(client.post("/api/chat/stream", json={"message": "hi"}).text)
        chunks = [json.loads(e) for e in events if e.startswith('"')]
        assert chunks == ["Partial "]
        assert any(e.startswith("[ERROR]") for e in events)
        assert mock_agent.stream.call_count == 1
        assert events[-1] == "[DONE]"

    def test_stream_passes_session_id_to_agent(self, client, mock_agent):
        mock_agent.stream.return_value = iter(["x"])
        client.post("/api/chat/stream", json={"message": "hi", "session_id": "sess_one"})
        assert mock_agent.stream.call_args.kwargs["session_id"] == "sess_one"

    def test_clean_response_keeps_inline_words(self):
        from backend.app.services.chat import clean_response

        text = "**Risk:** high\n| Risk: | Summary: |\nThe Summary: of it"
        cleaned = clean_response(text)
        assert cleaned.startswith("high")
        assert "| Risk: | Summary: |" in cleaned
        assert "The Summary: of it" in cleaned


# ═══════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════
class TestPersistence:
    def test_preview_is_latest_user_message(self):
        import persistence

        persistence.upsert_session("sess_p", client_id=TEST_CLIENT_ID)
        persistence.append_message("sess_p", role="user", content="zzz first", timestamp=1.0)
        persistence.append_message("sess_p", role="assistant", content="answer", timestamp=2.0)
        persistence.append_message("sess_p", role="user", content="aaa latest", timestamp=3.0)
        rows = persistence.list_sessions(TEST_CLIENT_ID)
        assert rows[0]["preview"] == "aaa latest"
        assert rows[0]["message_count"] == 3
        assert persistence.list_sessions(OTHER_CLIENT_ID) == []

    def test_update_alert_does_not_deadlock(self):
        import persistence

        persistence.upsert_session("sess_a", client_id=TEST_CLIENT_ID)
        alert = persistence.create_alert("sess_a", "MSFT", "daily_digest", threshold=None, label="x")
        updated = persistence.update_alert(alert["id"], label="renamed")
        assert updated["label"] == "renamed"
