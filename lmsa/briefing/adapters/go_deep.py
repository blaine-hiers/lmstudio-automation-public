""""Go deep" links on briefing story cards (#70) -- a default-on extension.

Every story card gets a plain relative link to the dashboard launcher with
`tasks.go_deep --story <briefing>:<story_id>` pre-filled (the dashboard never
runs it by itself; you still press run), and, once a deep report exists for
the story, a link to the newest one. No script is added to the report: the
dashboard's report CSP allows only `HANDLED_JS`.

The launch link is absolute -- `http://127.0.0.1:<DASHBOARD_PORT>/?task=...`
(`lmsa.config.DASHBOARD_PORT`, the dashboard's default port) -- so it works
from a report opened from disk too, as long as the dashboard is running on
that port. The deep-report link is relative to `output/`, so it works
either way.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote

from ...config import DASHBOARD_PORT
from ...report import StoryExtra
from ..sources import register_extension


def launch_href(address: str) -> str:
    """Dashboard URL that opens the go_deep tile with --story pre-filled."""
    return f"http://127.0.0.1:{DASHBOARD_PORT}/?task=go_deep&story={quote(address, safe='')}#task-go_deep"


def latest_report(conn, address: str) -> tuple[str, str] | None:
    """(report file name, created) of the newest deep report for a story, or None."""
    try:
        row = conn.execute(
            "SELECT report_path, created FROM go_deep_reports WHERE story=? ORDER BY created DESC, run_id DESC "
            "LIMIT 1", (address,)).fetchone()
    except sqlite3.Error:  # table not migrated in this db: no link, not a failure
        return None
    if not row or not row[0]:
        return None
    return Path(row[0]).name, str(row[1] or "")


def render(story, ctx) -> list:
    briefing = getattr(getattr(ctx, "briefing", None), "name", None)
    if not briefing or not getattr(story, "story_id", None):
        return []
    address = f"{briefing}:{story.story_id}"
    out = [StoryExtra("go deep", "open in dashboard", href=launch_href(address), tone="info")]
    conn = getattr(ctx, "conn", None)
    found = latest_report(conn, address) if conn is not None else None
    if found:
        name, created = found
        out.append(StoryExtra("deep report", f"research report {created[:10]}".strip(), href=name, tone="acc"))
    return out


register_extension("go_deep", render=render, default=True)
