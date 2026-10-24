"""PDF export of a chat transcript.

The PDF is rendered into a private temporary directory under a random UUID
name — user input never becomes part of a filesystem path — and is deleted
after the response has been sent.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from config import settings

logger = logging.getLogger("api.export")

_CHART_REF_RE = re.compile(r"\b([A-Za-z0-9_-]{1,100}\.png)\b")
_MAX_MESSAGE_CHARS = 3000


class ReportlabMissingError(RuntimeError):
    """Raised when the optional reportlab dependency is not installed."""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def download_filename(session_id: str) -> str:
    """Safe ``Content-Disposition`` filename (session id is pre-validated)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id)[:12] or "session"
    return f"chat_export_{safe}.pdf"


def render_chat_pdf(history: list[dict[str, str]], session_id: str) -> Path:
    """Render *history* into a PDF and return its path (inside a temp dir).

    Callers must delete ``path.parent`` once the file has been sent.
    """
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise ReportlabMissingError(str(exc)) from exc

    workdir = Path(tempfile.mkdtemp(prefix="chat-export-"))
    pdf_path = workdir / f"{uuid.uuid4().hex}.pdf"
    charts_dir = (settings.outputs_dir / "charts").resolve()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChatTitle", parent=styles["Title"], fontSize=16,
        textColor=HexColor("#4f46e5"), spaceAfter=12,
    )
    user_style = ParagraphStyle(
        "UserMsg", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#1e3a5f"), leftIndent=10, spaceAfter=4,
    )
    agent_style = ParagraphStyle(
        "AgentMsg", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#1a1d2e"), leftIndent=10, spaceAfter=4,
    )
    role_style = ParagraphStyle(
        "RoleLabel", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#4f46e5"), spaceAfter=2,
    )

    elements: list = [
        Paragraph("Financial AI Agent — Chat Export", title_style),
        Paragraph(
            f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Session: {_escape(session_id[:12])}…",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    for msg in history:
        is_user = msg["role"] == "user"
        elements.append(Paragraph(f"<b>{'You' if is_user else 'Agent'}:</b>", role_style))
        content = _escape(msg["content"])
        if len(content) > _MAX_MESSAGE_CHARS:
            content = content[:_MAX_MESSAGE_CHARS] + "… [truncated]"
        elements.append(Paragraph(content, user_style if is_user else agent_style))

        for chart_file in _CHART_REF_RE.findall(msg["content"]):
            chart_path = (charts_dir / chart_file).resolve()
            if chart_path.parent != charts_dir or not chart_path.is_file():
                continue
            try:
                elements.append(Spacer(1, 6))
                elements.append(RLImage(str(chart_path), width=400, height=200))
            except Exception:
                logger.debug("Could not embed chart %s", chart_file, exc_info=True)
        elements.append(Spacer(1, 8))

    try:
        SimpleDocTemplate(
            str(pdf_path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm
        ).build(elements)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return pdf_path
