"""Tests for ChartTools (tools/chart_tools.py).

All yfinance network calls are mocked so tests run offline and fast.
Every tool now returns a JSON string conforming to ``ToolResult``.
"""

import json

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from tools.chart_tools import ChartTools, ToolResult


# ── Helpers ──────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    """Parse a tool's JSON response string into a dict."""
    return json.loads(raw)


REQUIRED_KEYS: set[str] = {"status", "tool_name", "message", "data", "artifacts", "sources", "errors"}


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def chart_tools(tmp_path):
    """ChartTools instance writing to a temp directory."""
    return ChartTools(output_dir=str(tmp_path / "charts"))


@pytest.fixture
def fake_history():
    """Realistic 30-day price history DataFrame from yfinance."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D", tz="America/New_York")
    prices = np.linspace(100.0, 120.0, 30)
    return pd.DataFrame(
        {
            "Open":   prices - 0.5,
            "High":   prices + 1.0,
            "Low":    prices - 1.0,
            "Close":  prices,
            "Volume": [5_000_000] * 30,
        },
        index=dates,
    )


# ── Shared response-format tests ────────────────────────────────────

class TestToolResultFormat:
    """Every tool must return valid JSON matching the ToolResult schema."""

    def test_success_response_has_all_keys(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert REQUIRED_KEYS == set(result.keys())

    def test_error_response_has_all_keys(self, chart_tools):
        with patch("cache.cached_ticker_history", return_value=pd.DataFrame()):
            result = _parse(chart_tools.create_stock_chart("FAKE"))
        assert REQUIRED_KEYS == set(result.keys())

    def test_success_status_value(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert result["status"] == "success"

    def test_error_status_value(self, chart_tools):
        with patch("cache.cached_ticker_history", side_effect=ConnectionError("Network")):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert result["status"] == "error"

    def test_success_errors_list_empty(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert result["errors"] == []

    def test_error_errors_list_non_empty(self, chart_tools):
        with patch("cache.cached_ticker_history", side_effect=ConnectionError("Network unavailable")):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert len(result["errors"]) > 0

    def test_success_artifacts_is_list(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert isinstance(result["artifacts"], list)
        assert len(result["artifacts"]) == 1

    def test_error_artifacts_is_empty_list(self, chart_tools):
        with patch("cache.cached_ticker_history", side_effect=ConnectionError("fail")):
            result = _parse(chart_tools.create_stock_chart("AAPL"))
        assert result["artifacts"] == []

    def test_all_tools_return_valid_json(self, chart_tools, fake_history):
        """Smoke test: every registered tool returns parseable JSON with the right keys."""
        mock_info = MagicMock()
        mock_info.last_price = 150.0
        with patch("cache.cached_ticker_history", return_value=fake_history), \
             patch("cache.cached_fast_info", return_value=mock_info):
            for raw in [
                chart_tools.create_stock_chart("AAPL"),
                chart_tools.create_comparison_bar_chart("AAPL,MSFT"),
                chart_tools.generate_line_chart([1, 2], [10, 20]),
                chart_tools.generate_bar_chart(["A", "B"], [1, 2]),
                chart_tools.generate_pie_chart(["X", "Y"], [60, 40]),
            ]:
                result = _parse(raw)
                assert REQUIRED_KEYS == set(result.keys())
                assert result["status"] == "success"


# ── create_stock_chart ───────────────────────────────────────────────

class TestCreateStockChart:
    def test_returns_success_with_filename(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        assert result["status"] == "success"
        assert result["tool_name"] == "create_stock_chart"
        assert "aapl_chart" in result["message"].lower()
        assert ".png" in result["message"]

    def test_artifacts_contain_chart_file(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        assert len(result["artifacts"]) == 1
        artifact = result["artifacts"][0]
        assert artifact["type"] == "chart"
        assert artifact["filename"].endswith(".png")
        assert "aapl_chart" in artifact["filename"]

    def test_data_contains_price_info(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        data = result["data"]
        assert data["symbol"] == "AAPL"
        assert data["period"] == "1mo"
        for key in ("start_price", "end_price", "price_change", "pct_change"):
            assert key in data
            assert isinstance(data[key], (int, float))

    def test_sources_include_yahoo_finance(self, chart_tools, fake_history):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        assert "Yahoo Finance" in result["sources"]

    def test_chart_file_is_created_on_disk(self, chart_tools, fake_history, tmp_path):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            chart_tools.create_stock_chart("MSFT")

        png_files = list((tmp_path / "charts").glob("*.png"))
        assert len(png_files) == 1

    def test_empty_history_returns_error(self, chart_tools):
        with patch("cache.cached_ticker_history", return_value=pd.DataFrame()):
            result = _parse(chart_tools.create_stock_chart("FAKE"))

        assert result["status"] == "error"
        assert "no price data" in result["message"].lower()
        assert result["artifacts"] == []
        assert len(result["errors"]) > 0

    def test_yfinance_exception_returns_error(self, chart_tools):
        with patch("cache.cached_ticker_history", side_effect=ConnectionError("Network unavailable")):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        assert result["status"] == "error"
        assert result["tool_name"] == "create_stock_chart"
        assert len(result["errors"]) > 0

    def test_declining_stock_does_not_crash(self, chart_tools):
        dates = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
        declining = pd.DataFrame({"Close": np.linspace(150, 100, 10)}, index=dates)
        with patch("cache.cached_ticker_history", return_value=declining):
            result = _parse(chart_tools.create_stock_chart("TSLA"))

        assert result["status"] == "success"
        assert result["data"]["price_change"] < 0

    def test_artifact_path_matches_output_dir(self, chart_tools, fake_history, tmp_path):
        with patch("cache.cached_ticker_history", return_value=fake_history):
            result = _parse(chart_tools.create_stock_chart("AAPL"))

        artifact = result["artifacts"][0]
        assert str(tmp_path / "charts") in artifact["path"]


# ── create_comparison_bar_chart ──────────────────────────────────────

class TestCreateComparisonBarChart:
    def test_returns_success_for_valid_symbols(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = 150.0
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("AAPL,MSFT"))

        assert result["status"] == "success"
        assert result["tool_name"] == "create_comparison_bar_chart"
        assert "comparison" in result["message"].lower()
        assert ".png" in result["message"]
        assert len(result["artifacts"]) == 1

    def test_data_contains_symbols_and_prices(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = 150.0
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("AAPL,MSFT"))

        assert "symbols" in result["data"]
        assert "prices" in result["data"]
        assert "Yahoo Finance" in result["sources"]

    def test_single_valid_symbol_still_creates_chart(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = 200.0
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("NVDA"))

        assert result["status"] == "success"
        assert result["artifacts"][0]["filename"].endswith(".png")

    def test_all_invalid_symbols_returns_error(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = None
        mock_info.lastPrice = None
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("FAKE1,FAKE2"))

        assert result["status"] == "error"
        assert result["artifacts"] == []

    def test_handles_spaces_in_symbol_list(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = 100.0
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("AAPL, MSFT, TSLA"))

        assert result["status"] == "success"
        assert len(result["data"]["symbols"]) == 3

    def test_prices_dict_maps_symbol_to_value(self, chart_tools):
        mock_info = MagicMock()
        mock_info.last_price = 175.50
        with patch("cache.cached_fast_info", return_value=mock_info):
            result = _parse(chart_tools.create_comparison_bar_chart("AAPL"))

        prices = result["data"]["prices"]
        assert isinstance(prices, dict)
        assert "AAPL" in prices
        assert isinstance(prices["AAPL"], (int, float))


# ── Generic chart generators ─────────────────────────────────────────

class TestGenerateCharts:
    """Tests for the generic chart generators (no yfinance calls)."""

    def test_line_chart_creates_file(self, chart_tools, tmp_path):
        result = _parse(chart_tools.generate_line_chart([1, 2, 3], [10, 20, 15], title="Rev"))
        assert result["status"] == "success"
        assert result["tool_name"] == "generate_line_chart"
        assert "line_chart" in result["artifacts"][0]["filename"]
        assert len(list((tmp_path / "charts").glob("line_chart*.png"))) == 1

    def test_bar_chart_creates_file(self, chart_tools, tmp_path):
        result = _parse(chart_tools.generate_bar_chart(["Q1", "Q2", "Q3"], [100, 120, 110]))
        assert result["status"] == "success"
        assert result["tool_name"] == "generate_bar_chart"
        assert "bar_chart" in result["artifacts"][0]["filename"]

    def test_pie_chart_creates_file(self, chart_tools, tmp_path):
        result = _parse(chart_tools.generate_pie_chart(["Tech", "Health", "Finance"], [40, 30, 30]))
        assert result["status"] == "success"
        assert result["tool_name"] == "generate_pie_chart"
        assert "pie_chart" in result["artifacts"][0]["filename"]

    def test_line_chart_custom_labels(self, chart_tools):
        result = _parse(chart_tools.generate_line_chart(
            ["Jan", "Feb", "Mar"], [50, 60, 55],
            title="Monthly Avg", x_label="Month", y_label="Price"
        ))
        assert result["status"] == "success"
        assert result["data"]["title"] == "Monthly Avg"

    def test_bar_chart_custom_color(self, chart_tools):
        result = _parse(chart_tools.generate_bar_chart(
            ["A", "B"], [1, 2], color="steelblue"
        ))
        assert result["status"] == "success"

    def test_line_chart_data_has_point_count(self, chart_tools):
        result = _parse(chart_tools.generate_line_chart([1, 2, 3, 4], [10, 20, 15, 25]))
        assert result["data"]["points"] == 4

    def test_bar_chart_data_has_category_count(self, chart_tools):
        result = _parse(chart_tools.generate_bar_chart(["A", "B", "C"], [1, 2, 3]))
        assert result["data"]["categories"] == 3

    def test_pie_chart_data_has_slice_count(self, chart_tools):
        result = _parse(chart_tools.generate_pie_chart(["X", "Y"], [60, 40]))
        assert result["data"]["slices"] == 2

    def test_generic_tools_have_empty_sources(self, chart_tools):
        """Generic charts don't fetch external data, so sources should be empty."""
        result = _parse(chart_tools.generate_line_chart([1, 2], [10, 20]))
        assert result["sources"] == []

    def test_generic_tools_have_empty_errors(self, chart_tools):
        result = _parse(chart_tools.generate_bar_chart(["A"], [1]))
        assert result["errors"] == []
