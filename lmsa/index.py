"""Build an index hub (output/index.html) linking every report.

Scans output/ for timestamped reports, groups by slug, and renders a navigable
index. Updated automatically after every Report.write().

    rebuild(out_dir)  # scan, group, render index
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config import settings
from .report import Card, Report, Section, Tile


def _parse_filename(name: str) -> tuple[str, datetime] | None:
    """Extract (slug, mtime) from a report filename.

    Reports follow the pattern: slug-YYYYMMDD-HHMM.html
    """
    if name == "index.html" or not name.endswith(".html"):
        return None

    # Match: anything-YYYYMMDD-HHMM.html
    match = re.match(r"^(.+)-(\d{8})-(\d{4})\.html$", name)
    if not match:
        return None

    slug, date_str, time_str = match.groups()
    try:
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        return slug, dt
    except ValueError:
        return None


def _read_title(path: Path) -> str | None:
    """Extract and unescape the <title> from an HTML file, or None if not found."""
    try:
        content = path.read_text(encoding="utf-8")
        match = re.search(r"<title>([^<]*)</title>", content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1))
    except Exception:
        pass
    return None


def _scan_reports(out_dir: Path) -> Iterator[tuple[str, str, datetime, str | None]]:
    """Yield (slug, filename, mtime, title) for all reports in out_dir, newest first."""
    reports: list[tuple[str, str, datetime, str | None]] = []

    for file in sorted([*out_dir.glob("*.html"), *out_dir.glob("*/*/*.html")]):  # flat + <day>/<type>/
        parsed = _parse_filename(file.name)
        if parsed:
            slug, dt = parsed
            title = _read_title(file)  # Read title from HTML, fall back to None
            reports.append((slug, file.relative_to(out_dir).as_posix(), dt, title))

    # Sort by datetime, newest first
    for slug, filename, dt, title in sorted(reports, key=lambda x: x[2], reverse=True):
        yield slug, filename, dt, title


def rebuild(out_dir: Path | None = None) -> Path:
    """Scan reports, group by slug, and render index.html.

    Returns the path to the generated index.html.
    """
    out_dir = out_dir or settings().output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group reports by slug
    by_slug: dict[str, list[tuple[str, datetime, str | None]]] = {}
    all_reports: list[tuple[str, datetime]] = []

    for slug, filename, dt, title in _scan_reports(out_dir):
        if slug not in by_slug:
            by_slug[slug] = []
        by_slug[slug].append((filename, dt, title))
        all_reports.append((slug, dt))

    # Count stats
    total_reports = len(all_reports)
    tasks = len(by_slug)
    last_run = max((dt for _, dt in all_reports), default=None)

    # Build the index report
    r = Report(
        "Report Index",
        kicker="lmsa · hub",
        subtitle="Recent reports, grouped by task.",
        sticker=f"{total_reports} reports" if total_reports else "no reports yet",
        sticker_tone="acc" if total_reports else "muted",
    )

    # Tiles showing overview
    tiles = [Tile("Total reports", total_reports, tone="acc")]
    if tasks > 0:
        tiles.append(Tile("Tasks", tasks, tone="info"))
    if last_run:
        tiles.append(Tile("Last run", last_run.strftime("%Y-%m-%d %H:%M"), tone="violet"))

    r.tiles(tiles)

    # One section per task
    for slug in sorted(by_slug.keys()):
        reports = by_slug[slug]
        # Newest first, limit to 10
        recent = reports[:10]

        section = r.section(slug, tone="acc")
        # Build cards with title (or filename fallback), href, and timestamp
        cards = []
        for filename, dt, title in recent:
            # Use title from report if available, otherwise fall back to filename
            display_title = title if title else filename.rsplit("/", 1)[-1]
            cards.append(
                Card(
                    display_title,
                    meta=dt.strftime("%Y-%m-%d %H:%M"),
                    href=filename,
                )
            )
        section.cards(cards)

    # Write with a fixed name to avoid recursion
    index_path = out_dir / "index.html"
    index_path.write_text(r.render(), encoding="utf-8")
    return index_path
