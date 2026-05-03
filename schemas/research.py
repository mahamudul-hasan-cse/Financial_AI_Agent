"""Structured investor research response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchMetric(BaseModel):
    """A single headline metric shown in the structured answer card."""

    label: str
    value: str
    tone: str | None = Field(default=None, description="positive, negative, neutral, or risk")


class ResearchSignal(BaseModel):
    """A short signal card used for news, risks, and next actions."""

    title: str
    detail: str
    tone: str | None = Field(default=None, description="positive, negative, neutral, or risk")


class ResearchSource(BaseModel):
    """A grounded source used to build the response."""

    label: str
    url: str | None = None
    kind: str = "reference"


class ExplainabilityItem(BaseModel):
    """One explainability row for why the app answered the way it did."""

    label: str
    value: str


class ResearchResponse(BaseModel):
    """Structured response returned alongside the raw agent text."""

    title: str
    summary: str
    status: str = "ok"
    symbol: str | None = None
    company_name: str | None = None
    intent: str = "general"
    summary_markdown: str | None = None
    metrics: list[ResearchMetric] = Field(default_factory=list)
    news_signals: list[ResearchSignal] = Field(default_factory=list)
    risks: list[ResearchSignal] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    explainability: list[ExplainabilityItem] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SnapshotHeadline(BaseModel):
    """One recent headline for the snapshot page."""

    title: str
    url: str | None = None
    publisher: str | None = None
    sentiment: str | None = None


class SnapshotPoint(BaseModel):
    """A chart point for the company snapshot trend chart."""

    label: str
    value: float


class CompanySnapshotResponse(BaseModel):
    """Detailed company view for the investor dashboard."""

    symbol: str
    company_name: str
    price: float | None = None
    change_pct: float | None = None
    week_change_pct: float | None = None
    analyst_view: str | None = None
    recommendation: str | None = None
    sentiment_label: str | None = None
    fundamentals: list[ResearchMetric] = Field(default_factory=list)
    headlines: list[SnapshotHeadline] = Field(default_factory=list)
    chart: list[SnapshotPoint] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    """One comparison table row."""

    metric: str
    values: dict[str, str] = Field(default_factory=dict)


class ComparisonResponse(BaseModel):
    """Response for the dedicated stock comparison workflow."""

    symbols: list[str] = Field(default_factory=list)
    headline: str
    winner: str | None = None
    recommendation: str
    rows: list[ComparisonRow] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)

