"""Chart generation tools for the Financial AI Agent.

Provides tools to generate various types of charts (line, bar, pie, etc.)
from financial data and save them as image files.  Includes self-contained
stock chart tools that fetch data automatically via yfinance.

Every public tool returns a **JSON string** conforming to the
``ToolResult`` schema so consumers (LLM agent, API layer, tests) can
rely on a consistent structured format.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from agno.tools import Toolkit

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    raise ImportError("matplotlib is required. Install with: pip install matplotlib")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")

import cache


# ── Structured response types ────────────────────────────────────────

class Artifact(TypedDict):
    """Descriptor for a file produced by a tool."""

    type: str       # e.g. "chart"
    path: str       # e.g. "outputs/charts/aapl_chart_20260307.png"
    filename: str   # e.g. "aapl_chart_20260307.png"


class ToolResult(TypedDict):
    """Canonical response shape returned by every chart tool.

    Fields
    ------
    status      ``"success"`` or ``"error"``.
    tool_name   Name of the tool that produced this result.
    message     Human-readable summary the LLM can relay to the user.
    data        Tool-specific structured payload (prices, symbols, …).
    artifacts   List of generated-file descriptors.
    sources     Data sources consulted (e.g. ``["Yahoo Finance"]``).
    errors      Error messages (empty list on success).
    """

    status: str
    tool_name: str
    message: str
    data: dict[str, Any]
    artifacts: list[Artifact]
    sources: list[str]
    errors: list[str]


# ── Toolkit ──────────────────────────────────────────────────────────

class ChartTools(Toolkit):
    """Toolkit for generating financial charts.

    Every registered tool returns a JSON string matching the
    ``ToolResult`` schema.  On success the ``artifacts`` list contains
    the generated chart file; on error ``errors`` is populated and
    ``artifacts`` is empty.
    """

    def __init__(self, output_dir: str = "outputs/charts") -> None:
        """Initialize ChartTools.

        Args:
            output_dir: Directory to save chart images.
        """
        super().__init__(name="chart_tools")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.register(self.create_stock_chart)
        self.register(self.create_comparison_bar_chart)
        self.register(self.create_comparison_overlay_chart)
        self.register(self.generate_line_chart)
        self.register(self.generate_bar_chart)
        self.register(self.generate_pie_chart)

    # ── Response builders ────────────────────────────────────────────

    def _ok(
        self,
        tool_name: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
        sources: list[str] | None = None,
    ) -> str:
        """Build and serialise a **success** ``ToolResult``."""
        result: ToolResult = {
            "status": "success",
            "tool_name": tool_name,
            "message": message,
            "data": data or {},
            "artifacts": artifacts or [],
            "sources": sources or [],
            "errors": [],
        }
        return json.dumps(result)

    def _fail(
        self,
        tool_name: str,
        message: str,
        *,
        errors: list[str] | None = None,
    ) -> str:
        """Build and serialise an **error** ``ToolResult``."""
        result: ToolResult = {
            "status": "error",
            "tool_name": tool_name,
            "message": message,
            "data": {},
            "artifacts": [],
            "sources": [],
            "errors": errors or [message],
        }
        return json.dumps(result)

    def _artifact(self, filename: str) -> Artifact:
        """Create an ``Artifact`` dict for a generated chart file."""
        return {
            "type": "chart",
            "path": str(self.output_dir / filename),
            "filename": filename,
        }

    # ── Stock chart tools ────────────────────────────────────────────

    def create_stock_chart(self, symbol: str, period: str = "1mo") -> str:
        """Create a stock price chart by fetching data from Yahoo Finance. Use this when the user asks for a stock chart, stock price chart, or price history chart.

        Args:
            symbol: Stock ticker symbol, e.g. AAPL, GOOGL, MSFT, TSLA.
            period: Time period for the chart. Valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max. Default is 1mo.

        Returns:
            JSON string with status, chart filename, and price data.
        """
        tool = "create_stock_chart"
        try:
            hist = cache.cached_ticker_history(symbol, period)

            if hist.empty:
                return self._fail(
                    tool,
                    f"No price data found for {symbol}. Please check the ticker symbol.",
                )

            dates = hist.index.tolist()
            prices = hist["Close"].tolist()

            fig, ax = plt.subplots(figsize=(12, 6))

            color = "green" if prices[-1] >= prices[0] else "red"
            ax.plot(dates, prices, linewidth=2, color=color)
            ax.fill_between(dates, prices, alpha=0.2, color=color)

            ax.set_title(f"{symbol.upper()} Stock Price ({period})", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            price_change = prices[-1] - prices[0]
            pct_change = (price_change / prices[0]) * 100
            ax.annotate(
                f"Change: ${price_change:+.2f} ({pct_change:+.2f}%)",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                fontsize=10,
                verticalalignment="top",
                color=color,
            )

            plt.tight_layout()
            filename = self._save_chart(fig, f"{symbol.lower()}_chart_{self._timestamp()}")

            return self._ok(
                tool,
                f"Chart image saved as {filename}",
                data={
                    "symbol": symbol.upper(),
                    "period": period,
                    "start_price": round(prices[0], 2),
                    "end_price": round(prices[-1], 2),
                    "price_change": round(price_change, 2),
                    "pct_change": round(pct_change, 2),
                },
                artifacts=[self._artifact(filename)],
                sources=["Yahoo Finance"],
            )
        except Exception as e:
            return self._fail(
                tool,
                f"Error creating chart for {symbol}: {e}",
                errors=[str(e)],
            )

    def create_comparison_bar_chart(self, symbols: str) -> str:
        """Create a bar chart comparing current prices of multiple stocks. Use this when the user asks to compare stocks visually.

        Args:
            symbols: Comma-separated stock ticker symbols, e.g. 'AAPL,GOOGL,MSFT' or 'AAPL, GOOGL, MSFT'.

        Returns:
            JSON string with status, chart filename, and price data.
        """
        tool = "create_comparison_bar_chart"
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            prices: list[float] = []
            valid_symbols: list[str] = []

            for sym in symbol_list:
                try:
                    info = cache.cached_fast_info(sym)
                    price = getattr(info, "last_price", None) or getattr(info, "lastPrice", 0)
                    if price:
                        prices.append(float(price))
                        valid_symbols.append(sym)
                except Exception:
                    continue

            if not valid_symbols:
                return self._fail(
                    tool,
                    "Could not fetch prices for any of the provided symbols.",
                )

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set2(range(len(valid_symbols)))
            bars = ax.bar(valid_symbols, prices, color=colors, edgecolor="black", alpha=0.8)
            ax.set_title("Stock Price Comparison", fontsize=14, fontweight="bold")
            ax.set_xlabel("Stock")
            ax.set_ylabel("Price ($)")
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, prices):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"${val:,.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            plt.tight_layout()
            filename = self._save_chart(fig, f"comparison_{self._timestamp()}")

            return self._ok(
                tool,
                f"Chart image saved as {filename}",
                data={
                    "symbols": valid_symbols,
                    "prices": {s: round(p, 2) for s, p in zip(valid_symbols, prices)},
                },
                artifacts=[self._artifact(filename)],
                sources=["Yahoo Finance"],
            )
        except Exception as e:
            return self._fail(
                tool,
                f"Error creating comparison chart: {e}",
                errors=[str(e)],
            )

    def create_comparison_overlay_chart(self, symbols: str, period: str = "3mo") -> str:
        """Create a dual-line overlay chart comparing two or more stocks' price performance over time. Use this when the user asks to compare stock price trends, performance history, or wants to see how stocks moved relative to each other.

        Args:
            symbols: Comma-separated stock ticker symbols, e.g. 'AAPL,GOOGL' or 'NVDA, AMD'.
            period: Time period for the chart. Valid values: 1mo, 3mo, 6mo, 1y, 2y, 5y. Default is 3mo.

        Returns:
            JSON string with status, chart filename, and performance data.
        """
        tool = "create_comparison_overlay_chart"
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if len(symbol_list) < 2:
                return self._fail(tool, "Need at least 2 symbols to compare. Provide comma-separated tickers.")

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ["#4f46e5", "#e11d48", "#16a34a", "#f59e0b", "#8b5cf6"]
            perf_data: dict[str, Any] = {}

            for i, sym in enumerate(symbol_list[:5]):
                try:
                    hist = cache.cached_ticker_history(sym, period)
                    if hist.empty:
                        continue
                    dates = hist.index.tolist()
                    prices = hist["Close"].tolist()
                    # Normalize to percentage change from first day
                    base = prices[0]
                    pct = [(p / base - 1) * 100 for p in prices]
                    color = colors[i % len(colors)]
                    ax.plot(dates, pct, linewidth=2, color=color, label=f"{sym} ({prices[-1]:.2f})")
                    perf_data[sym] = {
                        "start_price": round(base, 2),
                        "end_price": round(prices[-1], 2),
                        "pct_change": round(pct[-1], 2),
                    }
                except Exception:
                    continue

            if not perf_data:
                return self._fail(tool, "Could not fetch data for any of the provided symbols.")

            ax.set_title(f"Price Performance Comparison ({period})", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Change (%)")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = self._save_chart(fig, f"overlay_comparison_{self._timestamp()}")

            return self._ok(
                tool,
                f"Overlay comparison chart saved as {filename}",
                data={"symbols": list(perf_data.keys()), "period": period, "performance": perf_data},
                artifacts=[self._artifact(filename)],
                sources=["Yahoo Finance"],
            )
        except Exception as e:
            return self._fail(tool, f"Error creating overlay chart: {e}", errors=[str(e)])

    # ── Generic chart tools ──────────────────────────────────────────

    def generate_line_chart(
        self,
        x_data: list,
        y_data: list,
        title: str = "Line Chart",
        x_label: str = "X-Axis",
        y_label: str = "Y-Axis",
    ) -> str:
        """Generate a line chart from the provided data.

        Args:
            x_data: List of x-axis values such as dates or labels.
            y_data: List of y-axis values such as prices or amounts.
            title: Title of the chart.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.

        Returns:
            JSON string with status and chart filename.
        """
        tool = "generate_line_chart"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_data, y_data, marker="o", linewidth=2, markersize=4)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = self._save_chart(fig, f"line_chart_{self._timestamp()}")

            return self._ok(
                tool,
                f"Chart image saved as {filename}",
                data={"title": title, "points": len(x_data)},
                artifacts=[self._artifact(filename)],
            )
        except Exception as e:
            return self._fail(tool, f"Error creating line chart: {e}", errors=[str(e)])

    def generate_bar_chart(
        self,
        categories: list,
        values: list,
        title: str = "Bar Chart",
        x_label: str = "Categories",
        y_label: str = "Values",
        color: str = "steelblue",
    ) -> str:
        """Generate a bar chart from the provided data.

        Args:
            categories: List of category names for x-axis.
            values: List of numeric values for each category.
            title: Title of the chart.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            color: Color of the bars.

        Returns:
            JSON string with status and chart filename.
        """
        tool = "generate_bar_chart"
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories, values, color=color, edgecolor="black", alpha=0.8)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:,.2f}" if isinstance(val, float) else str(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            filename = self._save_chart(fig, f"bar_chart_{self._timestamp()}")

            return self._ok(
                tool,
                f"Chart image saved as {filename}",
                data={"title": title, "categories": len(categories)},
                artifacts=[self._artifact(filename)],
            )
        except Exception as e:
            return self._fail(tool, f"Error creating bar chart: {e}", errors=[str(e)])

    def generate_pie_chart(
        self,
        labels: list,
        sizes: list,
        title: str = "Pie Chart",
    ) -> str:
        """Generate a pie chart from the provided data.

        Args:
            labels: List of labels for each slice.
            sizes: List of numeric sizes or values for each slice.
            title: Title of the chart.

        Returns:
            JSON string with status and chart filename.
        """
        tool = "generate_pie_chart"
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(labels)))

            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                explode=[0.02] * len(labels),
            )

            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()

            filename = self._save_chart(fig, f"pie_chart_{self._timestamp()}")

            return self._ok(
                tool,
                f"Chart image saved as {filename}",
                data={"title": title, "slices": len(labels)},
                artifacts=[self._artifact(filename)],
            )
        except Exception as e:
            return self._fail(tool, f"Error creating pie chart: {e}", errors=[str(e)])

    # ── Internal helpers ─────────────────────────────────────────────

    def _save_chart(self, fig: Any, filename: str) -> str:
        """Save *fig* to disk and return the filename (with ``.png`` extension)."""
        full_filename = f"{filename}.png"
        filepath = self.output_dir / full_filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return full_filename

    def _timestamp(self) -> str:
        """Generate a timestamp string for unique filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
