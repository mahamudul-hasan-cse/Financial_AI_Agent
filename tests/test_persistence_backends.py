"""Persistence contract tests, run on SQLite and (optionally) PostgreSQL.

The Postgres variant runs only when ``TEST_DATABASE_URL`` points at a
disposable database, e.g.::

    docker run -d -e POSTGRES_PASSWORD=test -e POSTGRES_DB=fa -p 55432:5432 postgres:17
    TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/fa pytest tests/test_persistence_backends.py
"""

import os
import time

import pytest

import persistence

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set"),
        ),
    ]
)
def backend(request):
    if request.param == "postgres":
        persistence.configure(TEST_DATABASE_URL)
        assert persistence.using_postgres()
    persistence.reset_for_tests()
    yield request.param
    if request.param == "postgres":
        persistence.reset_for_tests()
        persistence.configure(None)
        assert not persistence.using_postgres()


def test_init_db_is_idempotent(backend):
    persistence.init_db()
    persistence.init_db()


def test_session_lifecycle(backend):
    persistence.upsert_session("s1", client_id="c1")
    assert persistence.get_session_owner("s1") == (True, "c1")
    assert persistence.get_session_owner("missing") == (False, None)

    persistence.append_message("s1", role="user", content="What is AAPL trading at?")
    persistence.append_message(
        "s1",
        role="assistant",
        content="About $200.",
        nlp_metadata={"intent": "price"},
        structured_response={"ticker": "AAPL"},
    )
    session = persistence.get_session("s1")
    assert session["title"] == "What is AAPL trading at?"
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"]
    assert session["messages"][1]["nlp_metadata"] == {"intent": "price"}
    assert session["messages"][1]["structured_response"] == {"ticker": "AAPL"}

    sessions = persistence.list_sessions("c1")
    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["preview"] == "What is AAPL trading at?"
    assert persistence.list_sessions("c2") == []
    assert persistence.get_session_summary("s1")["session_id"] == "s1"

    assert persistence.rename_session("s1", "Renamed") is True
    assert persistence.rename_session("missing", "x") is False
    assert persistence.get_session("s1")["title"] == "Renamed"

    persistence.upsert_session("s1", touch=False)
    persistence.delete_session("s1")
    assert persistence.get_session("s1") is None


def test_timestamps_keep_full_precision(backend):
    ts = 1761330000.123456
    persistence.append_message("s1", role="user", content="hi", timestamp=ts)
    assert persistence.get_session("s1")["messages"][0]["timestamp"] == ts


def test_watchlist_ignores_duplicates(backend):
    persistence.add_watchlist_symbol("s1", "aapl")
    persistence.add_watchlist_symbol("s1", "AAPL")
    persistence.add_watchlist_symbol("s1", "msft")
    assert persistence.list_watchlist("s1") == ["AAPL", "MSFT"]
    persistence.remove_watchlist_symbol("s1", "aapl")
    assert persistence.list_watchlist("s1") == ["MSFT"]


def test_alerts_crud(backend):
    first = persistence.create_alert("s1", "aapl", "price_above", threshold=250.0, label="A")
    second = persistence.create_alert("s1", "msft", "price_below", threshold=None, label="B")
    assert isinstance(first["id"], int) and second["id"] > first["id"]
    assert first["symbol"] == "AAPL" and first["enabled"] is True

    updated = persistence.update_alert(first["id"], enabled=False, label="A2")
    assert updated["enabled"] is False and updated["label"] == "A2"
    persistence.mark_alert_triggered(first["id"], last_price=251.5)
    assert persistence.get_alert(first["id"])["last_price"] == 251.5
    assert len(persistence.list_alerts("s1")) == 2

    persistence.delete_alert(second["id"])
    with pytest.raises(KeyError):
        persistence.get_alert(second["id"])


def test_prune_old_sessions(backend):
    persistence.upsert_session("old", client_id="c")
    persistence.upsert_session("new", client_id="c")
    removed = persistence.prune_old_sessions(max_sessions=100, session_ttl=-1)
    assert set(removed) == {"old", "new"}

    for i in range(3):
        persistence.upsert_session(f"s{i}", client_id="c")
        time.sleep(0.01)
    removed = persistence.prune_old_sessions(max_sessions=2, session_ttl=3600)
    assert removed == ["s0"]
