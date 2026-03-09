"""Shared test fixtures for the Financial AI Agent test suite."""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Reset in-memory session, rate-limit, and stats stores between tests."""
    from api import sessions, _rate_limits, _ip_rate_limits, _stats

    sessions.clear()
    _rate_limits.clear()
    _ip_rate_limits.clear()
    _stats["total_messages"] = 0
    _stats["total_sessions_created"] = 0
    _stats["intent_counts"] = defaultdict(int)
    _stats["response_times_ms"] = []
    _stats["classifier_usage"] = defaultdict(int)
    yield
    sessions.clear()
    _rate_limits.clear()
    _ip_rate_limits.clear()


@pytest.fixture()
def mock_agent():
    """Patch the agent service so no real LLM calls are made."""
    mock_service = MagicMock()
    mock_service.run.return_value = "Mocked agent response about stocks."
    mock_service.stream.return_value = iter(["Mocked agent response about stocks."])

    with patch("api.agent_service", mock_service):
        yield mock_service


@pytest.fixture()
def client(mock_agent):
    """FastAPI TestClient with the agent mocked out."""
    from api import app

    return TestClient(app)
