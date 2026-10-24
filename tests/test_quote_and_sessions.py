"""Stock quote in the structured response + sidebar rename/delete flows."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from research_service import build_structured_response


def _sse_events(body: str) -> list[str]:
    return [
        line[len("data: "):]
        for line in body.split("\n")
        if line.startswith("data: ")
    ]


def _ticker_meta():
    entity = SimpleNamespace(text="Apple", label="TICKER", ticker="AAPL")
    return SimpleNamespace(entities=[entity], intent="stock_price", intent_confidence=0.9)


def _market(price=190.5):
    return {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "price": price,
        "prev_close": 188.0,
        "change": 2.5,
        "change_pct": 1.33,
        "week_change_pct": 4.0,
        "currency": "USD",
        "fundamentals": [],
        "headlines": [],
        "sources": [],
        "closes": [
            {"date": "2026-10-01", "close": 180.0},
            {"date": "2026-10-02", "close": 185.0},
        ],
    }


class TestQuote:
    def test_quote_built_from_market_context(self):
        with patch("research_service.get_market_context", return_value=_market()):
            data = build_structured_response("price of apple?", "Apple is up.", _ticker_meta())
        quote = data["quote"]
        assert quote["symbol"] == "AAPL"
        assert quote["price"] == 190.5
        assert quote["change"] == 2.5
        assert quote["change_pct"] == 1.33
        assert [p["close"] for p in quote["closes"]] == [180.0, 185.0]

    def test_no_quote_without_live_price(self):
        with patch("research_service.get_market_context", return_value=_market(price=None)):
            data = build_structured_response("price of apple?", "Apple.", _ticker_meta())
        assert data["quote"] is None

    def test_no_quote_when_market_offline(self):
        # conftest makes get_market_context raise (offline); nothing is invented.
        data = build_structured_response("price of apple?", "Apple.", _ticker_meta())
        assert data["quote"] is None

    def test_quote_sent_in_research_sse_event(self, client, mock_agent):
        mock_agent.stream.return_value = iter(["Apple trades near $190."])
        with patch("research_service.get_market_context", return_value=_market()), patch(
            "backend.app.services.chat.process_nlp_and_track",
            return_value=_nlp_schema(),
        ):
            body = client.post("/api/chat/stream", json={"message": "AAPL price"}).text
        research = next(e for e in _sse_events(body) if e.startswith("[RESEARCH_DATA]"))
        quote = json.loads(research[len("[RESEARCH_DATA]"):])["quote"]
        assert quote["symbol"] == "AAPL" and quote["closes"][-1]["close"] == 185.0

    def test_market_context_exposes_change_and_closes(self):
        import market_service

        history = pd.DataFrame(
            {"Close": [100.0, 102.0, 101.0]},
            index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"]),
        )
        info = {"currentPrice": 101.0, "regularMarketPreviousClose": 102.0, "currency": "USD"}
        with patch("cache.cached_ticker_info", return_value=info), patch(
            "cache.cached_ticker_history", return_value=history
        ), patch("cache.cached_fast_info", return_value=SimpleNamespace()):
            market = market_service.get_market_context("aapl")
        assert market["change"] == -1.0
        assert market["change_pct"] == -0.98
        assert market["currency"] == "USD"
        assert market["closes"] == [
            {"date": "2026-10-01", "close": 100.0},
            {"date": "2026-10-02", "close": 102.0},
            {"date": "2026-10-03", "close": 101.0},
        ]


def _nlp_schema():
    from schemas import NLPMetadataSchema

    return NLPMetadataSchema.model_validate(
        {
            "entities": [{"text": "AAPL", "label": "TICKER", "ticker": "AAPL"}],
            "intent": "stock_price",
            "intent_confidence": 0.9,
            "intent_method": "ml",
            "sentiment": {"label": "neutral", "score": 0.0, "compound": 0.0},
        }
    )


class TestSidebarSessionActions:
    def test_rename_then_list_shows_new_title(self, client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        assert client.patch(f"/api/sessions/{sid}", json={"title": "My AAPL notes"}).status_code == 200
        titles = {s["session_id"]: s["title"] for s in client.get("/api/sessions").json()["sessions"]}
        assert titles[sid] == "My AAPL notes"

    def test_rename_rejects_empty_title(self, client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        assert client.patch(f"/api/sessions/{sid}", json={"title": ""}).status_code == 422

    def test_delete_removes_from_list(self, client):
        sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
        assert client.delete(f"/api/chat/{sid}").status_code == 200
        assert client.get("/api/sessions").json()["sessions"] == []
        assert client.get(f"/api/sessions/{sid}").status_code == 404

    def test_list_is_scoped_to_client(self, client, other_client):
        client.post("/api/chat", json={"message": "mine"})
        assert other_client.get("/api/sessions").json()["sessions"] == []
