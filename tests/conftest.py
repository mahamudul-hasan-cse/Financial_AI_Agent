"""Shared test fixtures for the Financial AI Agent test suite.

Environment is configured *before* any application module is imported:

* ``DATABASE_PATH`` points at a throw-away SQLite file, so tests never touch
  ``data/financial_agent.db``.
* ``GROQ_API_KEY`` is removed, proving the app imports and starts without
  it (tests that build a real agno Agent set a fake key themselves).
"""

import os
import shutil
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

_TEST_DB_DIR = tempfile.mkdtemp(prefix="financial-agent-tests-")
os.environ["DATABASE_PATH"] = os.path.join(_TEST_DB_DIR, "test.db")
os.environ.pop("SQLITE_PATH", None)
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("GROQ_MODEL", None)
os.environ["LLM_PROVIDER"] = "groq"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TEST_CLIENT_ID = "test-client-0000000001"
OTHER_CLIENT_ID = "test-client-0000000002"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Reset in-memory session, rate-limit, and stats stores between tests."""
    from api import _client_rate_limits, _ip_rate_limits, _rate_limits, _stats, sessions
    from persistence import reset_for_tests

    sessions.clear()
    _rate_limits.clear()
    _ip_rate_limits.clear()
    _client_rate_limits.clear()
    reset_for_tests()
    _stats["total_messages"] = 0
    _stats["total_sessions_created"] = 0
    _stats["intent_counts"] = defaultdict(int)
    _stats["response_times_ms"] = []
    _stats["classifier_usage"] = defaultdict(int)
    yield
    sessions.clear()
    _rate_limits.clear()
    _ip_rate_limits.clear()
    _client_rate_limits.clear()
    reset_for_tests()


@pytest.fixture(autouse=True)
def _no_network_headlines():
    """Keep the chat path offline: no headlines, no live market context."""
    import cache

    cache.clear()
    with patch("market_service.get_recent_headlines", return_value=[]), patch(
        "research_service.get_market_context", side_effect=ConnectionError("offline in tests")
    ):
        yield


@pytest.fixture(autouse=True)
def _no_retry_sleep():
    """LLM retry back-off never really sleeps in tests."""
    with patch("backend.app.agent.financial_agent._sleep") as fake_sleep:
        yield fake_sleep


@pytest.fixture()
def mock_agent():
    """Patch the agent service so no real LLM calls are made."""
    mock_service = MagicMock()
    mock_service.run.return_value = "Mocked agent response about stocks."
    mock_service.stream.return_value = iter(["Mocked agent response about stocks."])

    with patch("backend.app.services.chat.get_agent_service", return_value=mock_service):
        yield mock_service


@pytest.fixture()
def client(mock_agent):
    """FastAPI TestClient (agent mocked) identifying as TEST_CLIENT_ID."""
    from api import app

    return TestClient(app, headers={"X-Client-Id": TEST_CLIENT_ID})


@pytest.fixture()
def other_client(mock_agent):
    """A second browser/user with a different anonymous client id."""
    from api import app

    return TestClient(app, headers={"X-Client-Id": OTHER_CLIENT_ID})
