"""The digest contract: how one task's latest run shows up inside a master
report email (#103).

A task *may* expose

    def digest(run: RunRow, max_items: int) -> Digest: ...

at module level (`tasks/<name>.py`). It is optional: a task without one is
bundled with a fallback built from its run's `summary_json` (headline tiles
plus a pointer to the full report -- `tasks.master_report.fallback_digest`).

Rules for a `digest()` hook:
  - Read stored data only (the run row, the store, the task's own saved
    files). **Never call the model** -- the email must go out even when LM
    Studio is down.
  - `run` is the task's latest *successful* run; it may be old (the master
    marks it STALE) -- don't assume it is from today.
  - Return at most `max_items` items. Text is plain and untrusted; the
    renderer escapes it. `DigestItem.url` should be an http(s) URL -- anything
    else is dropped by `normalize` (a local `file://` path is useless on a
    phone).
  - Raising is allowed: the master falls back to the `summary_json` digest
    and notes the hook's exception type.

Whatever a hook returns goes through `normalize` before rendering, which
enforces the size limits below and drops malformed elements, so a
misbehaving hook can't crash the run. (A `full`-mode section shows the
task's saved report instead -- `lmsa.email_report` -- and only falls back to
its digest, uncut, when that file can't be read.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .report import TONES, Tile, _fmt

MAX_TILES = 4
MAX_CALLOUTS = 5
MAX_TITLE = 200
MAX_BLURB = 240
MAX_CALLOUT = 240
MAX_TILE_LABEL = 60
MAX_TILE_VALUE = 24

# Section status kinds, in the order they're explained in the email.
OK, STALE, FAILED, NOT_RUN = "OK", "STALE", "FAILED", "NOT RUN"


@dataclass(frozen=True)
class RunRow:
    """One row of the store's `runs` table (or `briefing_runs`, for a
    briefing section -- then `briefing` is the config name and `id` is its
    run_id). `started` is the ISO timestamp as stored."""
    id: int
    task: str
    started: str
    status: str
    finished: str | None = None
    report_path: str | None = None
    summary_json: str | None = None
    briefing: str | None = None


@dataclass
class DigestItem:
    title: str
    url: str | None = None  # http(s) only; anything else is dropped by normalize
    blurb: str = ""  # one line
    tone: str | None = None  # one of lmsa.report.TONES


@dataclass
class Digest:
    tiles: list[Tile] = field(default_factory=list)  # 1-4 headline numbers
    items: list[DigestItem] = field(default_factory=list)
    callouts: list[str] = field(default_factory=list)  # short strings, e.g. "3 replies overdue"

    def is_empty(self) -> bool:
        return not (self.tiles or self.items or self.callouts)


@dataclass(frozen=True)
class SectionStatus:
    kind: str  # OK | STALE | FAILED | NOT RUN
    age_hours: float | None = None  # age of the last good run, when there is one

    @property
    def label(self) -> str:
        if self.kind == STALE and self.age_hours is not None:
            return f"STALE (last good run {self.age_hours:.0f}h ago)"
        return self.kind


def http_url(url: object) -> str | None:
    """`url` if it is an http(s) URL with a host, else None."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return url


def _one_line(text: object, limit: int) -> str:
    """`text` on one line, cut to `limit` characters (0 = no cap)."""
    s = " ".join(str(text or "").split())
    return s if not limit or len(s) <= limit else s[: limit - 1] + "…"


def _as_list(value: object) -> list:
    """A hook's tiles/items/callouts as a list; None or any other
    non-list (a str isn't a list of callouts) is treated as empty."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _tile_value(value: object) -> int | float | str:
    """A tile value short enough to render: numbers stay numbers unless
    their formatting would be long (then 3 significant digits), anything
    else becomes a length-capped one-line string."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if len(_fmt(value)) <= MAX_TILE_VALUE else f"{value:.3g}"
    return _one_line(value, MAX_TILE_VALUE)


def normalize(d: Digest, max_items: int, *, clip: bool = True) -> tuple[Digest, int, int]:
    """Clamp a digest to the contract: <= 4 tiles with capped label/value,
    <= `max_items` items with one-line, length-capped text and http(s)-only
    urls, <= 5 short callouts.

    `clip=False` (a `full`-mode section showing its digest, #119) keeps the
    one-line and http(s)-only rules but cuts no text: item titles, blurbs
    and callouts stay whole and every callout is kept.

    Tolerant of a malformed hook result: a tiles/items/callouts field that
    isn't a list counts as empty, and an element of the wrong type (not a
    `Tile` / `DigestItem` / `str`) is dropped. Returns (clean digest,
    items cut by `max_items`, malformed elements dropped)."""
    raw_tiles, raw_items, raw_callouts = _as_list(d.tiles), _as_list(d.items), _as_list(d.callouts)
    dropped = sum(not isinstance(value, (list, tuple)) for value in (d.tiles, d.items, d.callouts))
    good_tiles = [t for t in raw_tiles if isinstance(t, Tile)]
    good_items = [i for i in raw_items if isinstance(i, DigestItem)]
    good_callouts = [c for c in raw_callouts if isinstance(c, str)]
    dropped += (len(raw_tiles) - len(good_tiles)) + (len(raw_items) - len(good_items))         + (len(raw_callouts) - len(good_callouts))
    tiles = [
        Tile(_one_line(t.label, MAX_TILE_LABEL), _tile_value(t.value), tone=t.tone if t.tone in TONES else "acc")
        for t in good_tiles[:MAX_TILES]
    ]
    title_cap, blurb_cap, callout_cap = (MAX_TITLE, MAX_BLURB, MAX_CALLOUT) if clip else (0, 0, 0)
    items = [
        DigestItem(
            title=_one_line(i.title, title_cap) or "(untitled)",
            url=http_url(i.url),
            blurb=_one_line(i.blurb, blurb_cap),
            tone=i.tone if i.tone in TONES else None,
        )
        for i in good_items[:max_items]
    ]
    kept_callouts = good_callouts[:MAX_CALLOUTS] if clip else good_callouts
    callouts = [_one_line(c, callout_cap) for c in kept_callouts if c.strip()]
    return Digest(tiles, items, callouts), max(len(good_items) - max_items, 0), dropped


def _parse_ts(value: str | None) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def section_status(last_ok: RunRow | None, latest: RunRow | None, now: datetime,
                   max_age_hours: float) -> SectionStatus:
    """Status chip for one section.

    `last_ok` is the task's latest `status='ok'` run; `latest` its latest
    *finished* run (ok or error -- a row still `running` is ignored, so a
    task that hasn't finished yet today shows yesterday's run as STALE
    rather than failing). Precedence: FAILED (latest finished run errored)
    > NOT RUN (no good run ever) > STALE (last good run older than
    `max_age_hours`, or its timestamp unreadable) > OK.
    """
    age = None
    if last_ok is not None:
        started = _parse_ts(last_ok.started)
        if started is not None:
            age = max((now - started).total_seconds() / 3600, 0.0)
    if latest is not None and latest.status == "error":
        return SectionStatus(FAILED, age)
    if last_ok is None:
        return SectionStatus(NOT_RUN)
    if age is None or age > max_age_hours:
        return SectionStatus(STALE, age)
    return SectionStatus(OK, age)
