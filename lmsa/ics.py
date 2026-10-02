"""Minimal RFC 5545 (iCalendar) writer.

Public API other tasks build on -- keep it small, the travel-itinerary
task (#19) imports this module for its own VEVENTs:

    make_uid(*parts) -> str      -- deterministic UID from stable source identity
    Event                        -- one VEVENT's worth of fields
    render_ics(events) -> str    -- the whole .ics document, as text
    write_ics(events, path)      -- render_ics() written to `path`

Timed events are always emitted in UTC (`...Z`), never with a TZID: it's
the one representation every calendar client parses unambiguously without
also shipping a VTIMEZONE block. A naive `datetime` is treated as *local
system time* before that UTC conversion -- Python's own convention for
`datetime.astimezone()` on a naive value -- so pass an aware datetime
whenever the source timezone isn't the machine running the task.

All-day events use `VALUE=DATE` with an *exclusive* DTEND (the day after
the last day), per RFC 5545 3.6.1: a single all-day event on 2026-01-05
needs no explicit `end` -- one is derived as 2026-01-06.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_FOLD_LIMIT = 75  # octets, per RFC 5545 3.1


def make_uid(*parts: str) -> str:
    """Deterministic UID from stable source identity (e.g. msg_id, title, date).

    Same inputs always produce the same UID, so re-running a task that
    emits the same event updates it on import instead of duplicating it --
    most calendar clients treat an unchanged UID as an update, not a new
    event. A control character joins the parts so that e.g. ("ab", "c")
    and ("a", "bc") can never collide.
    """
    joined = "\x1f".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{digest}@lmsa"


@dataclass(frozen=True)
class Event:
    uid: str
    title: str
    start: date | datetime
    end: date | datetime | None = None
    all_day: bool = False
    location: str | None = None
    description: str | None = None


def _escape_text(value: str) -> str:
    """RFC 5545 3.3.11 TEXT escaping: backslash, semicolon, comma, newline."""
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\n")
        .replace("\r", "\n")  # a bare CR is a line terminator too; never emit it raw
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """Fold a content line at 75 octets (UTF-8 bytes), RFC 5545 3.1.

    Continuation lines are joined by CRLF + a single leading space, which
    itself counts against that line's 75-octet budget. A multi-byte UTF-8
    character is never split across a fold: the chunk boundary backs off
    to the start of the character whenever it would land mid-sequence.
    """
    data = line.encode("utf-8")
    if len(data) <= _FOLD_LIMIT:
        return line
    chunks: list[bytes] = []
    start = 0
    budget = _FOLD_LIMIT
    while start < len(data):
        end = min(start + budget, len(data))
        while end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        if end <= start:  # pathological (budget too small for one char) -- doesn't happen at 74+
            end = start + 1
        chunks.append(data[start:end])
        start = end
        budget = _FOLD_LIMIT - 1  # continuation lines lose one octet to the leading space
    return "\r\n ".join(chunk.decode("utf-8") for chunk in chunks)


def _line(name: str, value: str, params: str = "") -> str:
    return _fold(f"{name}{params}:{value}")


def _dtstamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_timed(value: datetime) -> str:
    """UTC `...Z` form. A naive value is presumed local system time first
    (Python's own `astimezone()` convention), then converted to UTC."""
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _event_lines(event: Event) -> list[str]:
    lines = ["BEGIN:VEVENT", _line("UID", _escape_text(event.uid)), _line("DTSTAMP", _dtstamp_utc())]

    if event.all_day:
        start_date = _as_date(event.start)
        end_date = _as_date(event.end) if event.end is not None else start_date + timedelta(days=1)
        lines.append(_line("DTSTART", _format_date(start_date), ";VALUE=DATE"))
        lines.append(_line("DTEND", _format_date(end_date), ";VALUE=DATE"))
    else:
        if not isinstance(event.start, datetime):
            raise ValueError("Event.start must be a datetime when all_day is False")
        lines.append(_line("DTSTART", _format_timed(event.start)))
        if event.end is not None:
            if not isinstance(event.end, datetime):
                raise ValueError("Event.end must be a datetime when all_day is False")
            lines.append(_line("DTEND", _format_timed(event.end)))

    lines.append(_line("SUMMARY", _escape_text(event.title)))
    if event.location:
        lines.append(_line("LOCATION", _escape_text(event.location)))
    if event.description:
        lines.append(_line("DESCRIPTION", _escape_text(event.description)))

    lines.append("END:VEVENT")
    return lines


def render_ics(events: list[Event], *, prod_id: str = "-//lmsa//lmsa//EN") -> str:
    """The whole .ics document as text: CRLF line endings, folded, escaped."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", _line("PRODID", prod_id), "CALSCALE:GREGORIAN"]
    for event in events:
        lines.extend(_event_lines(event))
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def write_ics(events: list[Event], path: str | Path, *, prod_id: str = "-//lmsa//lmsa//EN") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_ics(events, prod_id=prod_id).encode("utf-8"))
    return path
