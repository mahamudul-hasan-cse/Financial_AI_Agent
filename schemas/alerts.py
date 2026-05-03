"""Alert schemas for the Financial AI Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertCreateRequest(BaseModel):
    """Request body for creating a simple watch alert."""

    session_id: str
    symbol: str = Field(..., min_length=1, max_length=10)
    alert_type: str = Field(
        ...,
        description="One of price_above, price_below, daily_digest, breaking_news.",
    )
    threshold: float | None = None
    label: str | None = None


class AlertUpdateRequest(BaseModel):
    """Request body for toggling or renaming an alert."""

    enabled: bool | None = None
    label: str | None = None


class AlertResponse(BaseModel):
    """One persisted alert."""

    id: int
    session_id: str
    symbol: str
    alert_type: str
    threshold: float | None = None
    label: str
    enabled: bool = True
    last_triggered_at: float | None = None
    last_price: float | None = None


class AlertListResponse(BaseModel):
    """List of alerts for a session."""

    session_id: str
    alerts: list[AlertResponse] = Field(default_factory=list)


class AlertCheckItem(BaseModel):
    """One evaluated alert item after checking current prices/news."""

    alert: AlertResponse
    triggered: bool = False
    reason: str | None = None


class AlertCheckResponse(BaseModel):
    """Result of evaluating all alerts for a session."""

    session_id: str
    items: list[AlertCheckItem] = Field(default_factory=list)

