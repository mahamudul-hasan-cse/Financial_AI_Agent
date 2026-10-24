"""Regression tests: comparison intent needs 2+ tickers or explicit wording."""

import pytest

from nlp_pipeline import _reconcile_comparison_intent, run_nlp_pipeline


@pytest.mark.parametrize(
    "query",
    ["How is Apple stock doing?", "How's Tesla performing?"],
)
def test_single_stock_questions_are_not_comparisons(query):
    meta = run_nlp_pipeline(query)
    assert meta["intent"] != "comparison"


@pytest.mark.parametrize("query", ["Apple vs Microsoft", "Compare AAPL and MSFT"])
def test_explicit_comparisons_stay_comparisons(query):
    meta = run_nlp_pipeline(query)
    assert meta["intent"] == "comparison"


def test_reconcile_downgrades_single_ticker_comparison():
    assert (
        _reconcile_comparison_intent("How's Tesla performing?", "comparison", 0.23, "ml", ["TSLA"])[
            0
        ]
        == "stock_price"
    )


def test_reconcile_upgrades_two_ticker_vs_query():
    assert (
        _reconcile_comparison_intent("AAPL vs MSFT", "stock_price", 0.4, "ml", ["AAPL", "MSFT"])[0]
        == "comparison"
    )


def test_reconcile_keeps_explicit_single_ticker_compare():
    out = _reconcile_comparison_intent(
        "Compare Apple to its peers", "comparison", 0.5, "ml", ["AAPL"]
    )
    assert out[0] == "comparison"
