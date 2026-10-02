"""Travel itinerary builder: scans recent mail for flight/hotel/car/train
confirmations and groups them into trips with a chronological report and a
per-trip .ics file.

    python -m tasks.travel [--fresh]

Read-only: only reads Gmail over IMAP, never labels/sends/deletes. Each
message is asked (locally, via `lmsa.LMStudio.extract`) for the travel
segments it confirms; a message whose extraction fails is recorded and
skipped rather than aborting the run.

Confirmation numbers are the correlation key across emails: airlines and
hotels routinely send a follow-up email for the same booking (seat change,
itinerary change, cancellation). The newest email (by its own Date header)
for a given (kind, confirmation number) wins over older ones, and a
cancellation email removes that segment entirely -- see `resolve_segments`.

Flight/hotel/car/train times are local to the airport/hotel/branch, not to
whatever machine runs this task. The model is asked for ISO 8601 and told
to include a UTC offset only when the email itself states one; when it
doesn't, the resulting naive datetime is treated the same way `lmsa.ics`
treats a naive value (local-system-time convention) and the report says
plainly that the zone is unknown rather than guessing one.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from lmsa import Card, Link, LMStudio, Report, Store, Tile
from lmsa import layout
from lmsa.config import settings
from lmsa.ics import Event, make_uid, write_ics
from lmsa.report import safe_href
from lmsa.memo import Memo, add_footer, cached, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

SEARCH_QUERY = 'newer_than:120d (flight OR hotel OR itinerary OR reservation OR "rental car" OR train)'
SEARCH_LIMIT = 200
BODY_LIMIT = 4000
MEMO_VERSION = 1  # bump when extract_from_message()'s prompt building or result handling changes
KINDS = ("flight", "hotel", "car", "train")
TRIP_GAP = timedelta(hours=48)  # segments this close or closer chain into one trip
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)  # sentinel for a message with no Date header

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "provider": {"type": "string"},
        "confirmation": {"type": "string"},
        "start": {"type": "string"},
        "end": {"type": ["string", "null"]},
        "from": {"type": ["string", "null"]},
        "to": {"type": ["string", "null"]},
        "address": {"type": ["string", "null"]},
        "cancelled": {"type": "boolean"},
    },
    "required": ["kind", "provider", "confirmation", "start", "end", "from", "to", "address", "cancelled"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["items"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = load_prompt("travel/extract_system")


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


@dataclass
class Segment:
    kind: str
    provider: str
    confirmation: str
    start: datetime
    end: datetime | None
    origin: str | None
    destination: str | None
    address: str | None
    cancelled: bool
    tz_known: bool
    source_msg_id: str
    source_subject: str
    source_date: datetime | None
    source_url: str


@dataclass
class UnparseableItem:
    reason: str
    kind: str
    provider: str
    raw_start: str
    source_subject: str
    source_url: str


@dataclass
class FailedMessage:
    subject: str
    msg_id: str
    error: str


@dataclass
class Trip:
    segments: list[Segment]
    gaps: list[date]


class _BadItem(Exception):
    """A raw extracted item that can't become a `Segment` -- an invalid/missing
    `kind` (never guessed: a mistagged hotel would otherwise skew
    `_find_gaps`'s "no hotel" check) or an unparseable `start`. Recorded as
    an `UnparseableItem` and dropped, same as `deadlines.py`'s `_BadDate`."""

    def __init__(self, reason: str, kind: str, provider: str, raw_start: str):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.provider = provider
        self.raw_start = raw_start


def _truncate(text: str, limit: int = BODY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated, {omitted} more characters omitted ...]"


def extract_from_message(lm: LMStudio, msg: Message, memo: Memo | None = None) -> list[dict]:
    """The raw items the model finds in `msg`. With `memo`, a stored answer
    for this msg_id is reused. The prompt carries only the message's own
    Date header, never "today" -- past/upcoming is decided in Python at
    report time -- so a remembered answer never goes stale with time."""
    prompt = (
        f"Subject: {msg.subject}\n"
        f"From: {msg.from_addr}\n"
        f"Date: {msg.date.isoformat() if msg.date else 'unknown'}\n\n"
        f"{_truncate(msg.text)}"
    )
    result = cached(
        memo,
        msg.msg_id,
        lambda: lm.extract(prompt, SCHEMA, system=EXTRACT_SYSTEM, name="travel"),
        valid=lambda r: isinstance(r, dict) and isinstance(r.get("items"), list),
    )
    return list(result.get("items") or [])


def _parse_when(value: str | None) -> datetime | None:
    """Strict ISO 8601 parse. Returns None (never raises) on anything that
    doesn't parse. A bare date is accepted too (`fromisoformat` treats it
    as midnight); whether the result carries an offset or not is left to
    the caller (that's `tz_known`)."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_segment(raw: dict, msg: Message) -> Segment:
    raw_kind = raw.get("kind")
    provider = str(raw.get("provider") or "(unknown provider)")
    start_raw = raw.get("start") or ""
    if raw_kind not in KINDS:
        raise _BadItem(f"invalid kind {raw_kind!r}", str(raw_kind), provider, start_raw)
    kind = raw_kind
    confirmation = str(raw.get("confirmation") or "").strip()
    start = _parse_when(start_raw)
    if start is None:
        raise _BadItem(f"unparseable start {start_raw!r}", kind, provider, start_raw)
    end = _parse_when(raw.get("end"))
    return Segment(
        kind=kind,
        provider=provider,
        confirmation=confirmation,
        start=start,
        end=end,
        origin=raw.get("from") or None,
        destination=raw.get("to") or None,
        address=raw.get("address") or None,
        cancelled=bool(raw.get("cancelled")),
        tz_known=start.tzinfo is not None,
        source_msg_id=msg.msg_id,
        source_subject=msg.subject,
        source_date=msg.date,
        source_url=msg.web_url,
    )


def gather(
    lm: LMStudio, messages: list[Message], memo: Memo | None = None
) -> tuple[list[Segment], list[UnparseableItem], list[FailedMessage]]:
    """Extract segments from every message, isolating each message's own failure.

    A message whose `extract()` call itself fails (server down, malformed
    JSON, ...) is recorded in `failed` and skipped -- it never aborts the
    rest of the mailbox scan. Within a message's own items, one item with
    an invalid `kind` or an unparseable start goes to `unparseable`, the
    rest still come through.
    """
    segments: list[Segment] = []
    unparseable: list[UnparseableItem] = []
    failed: list[FailedMessage] = []
    for msg in messages:
        try:
            raw_items = extract_from_message(lm, msg, memo)
        except Exception as exc:
            failed.append(FailedMessage(subject=msg.subject, msg_id=msg.msg_id, error=f"{type(exc).__name__}: {exc}"))
            continue
        for raw in raw_items:
            try:
                segments.append(_build_segment(raw, msg))
            except _BadItem as exc:
                unparseable.append(
                    UnparseableItem(
                        reason=exc.reason,
                        kind=exc.kind,
                        provider=exc.provider,
                        raw_start=exc.raw_start,
                        source_subject=msg.subject,
                        source_url=msg.web_url,
                    )
                )
    return segments, unparseable, failed


def _normalize_confirmation(value: str) -> str:
    return "".join(value.split()).upper()


def _source_sort_key(seg: Segment) -> datetime:
    return seg.source_date or _EPOCH


def resolve_segments(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """Correlate segments by (kind, confirmation number) and keep only the
    newest email's data for each -- a later email supersedes an earlier
    one for the same booking, and if the newest email for a booking is a
    cancellation the booking is dropped entirely rather than kept in its
    last non-cancelled state.

    Input order doesn't matter: sorting here by the message's own Date
    header (newest first, ties broken by msg_id for determinism) is what
    makes "newest wins" correct even if the caller's list isn't already in
    that order. A segment with no confirmation number can't be correlated
    with anything else, so it's always kept on its own (unless the email
    that mentioned it was itself a cancellation).

    Two items for the same (kind, confirmation) extracted from the SAME
    message (equal source_date and msg_id) can't be ordered by date at
    all -- that tie is broken by `cancelled` so a cancellation always wins
    over a booking regardless of which order the model's own items array
    happened to list them in; a model that emits the same cancellation
    notice twice must not have a coin-flip chance of writing the booking
    to the .ics.

    Returns (kept, cancelled) -- `cancelled` is for the report only, never
    fed into a trip.
    """
    ordered = sorted(
        segments, key=lambda s: (_source_sort_key(s), s.source_msg_id, s.cancelled), reverse=True
    )
    kept: list[Segment] = []
    cancelled: list[Segment] = []
    seen_keys: set[tuple[str, str]] = set()
    for seg in ordered:
        norm_conf = _normalize_confirmation(seg.confirmation)
        key = (seg.kind, norm_conf) if norm_conf else None
        if key is not None:
            if key in seen_keys:
                continue  # superseded by a newer email already processed
            seen_keys.add(key)
        (cancelled if seg.cancelled else kept).append(seg)
    return kept, cancelled


def _aware(dt: datetime) -> datetime:
    """Naive datetimes are presumed local system time (matches `lmsa.ics`'s
    own convention for a naive `Event.start`), then made aware -- needed
    before any comparison since aware and naive datetimes can't be compared."""
    return dt if dt.tzinfo is not None else dt.astimezone()


def _effective_end(seg: Segment) -> datetime:
    return _aware(seg.end or seg.start)


def drop_past(segments: list[Segment], now: datetime) -> tuple[list[Segment], list[Segment]]:
    """Split into (upcoming, past) by comparing each segment's effective end
    (or start, when no end is given) to `now`."""
    upcoming, past = [], []
    for seg in segments:
        (past if _effective_end(seg) <= now else upcoming).append(seg)
    return upcoming, past


def _find_gaps(segments: list[Segment]) -> list[date]:
    """Nights within the trip's span that no hotel segment covers.

    A hotel segment with a known end covers the nights from its check-in
    date up to (but not including) its check-out date -- standard hotel
    semantics. Only the nights actually needed -- from the trip's first
    start to the night before its last end -- are checked; a hotel with no
    stated check-out date can't be verified to cover anything, so it never
    counts as coverage (better to over-report a gap than hide one).
    """
    trip_start = min(_aware(s.start) for s in segments).date()
    trip_end = max(_effective_end(s) for s in segments).date()
    if trip_end <= trip_start:
        return []
    nights_needed = [trip_start + timedelta(days=i) for i in range((trip_end - trip_start).days)]
    hotel_ranges = [
        (_aware(s.start).date(), _aware(s.end).date())
        for s in segments
        if s.kind == "hotel" and s.end is not None
    ]
    return [
        night
        for night in nights_needed
        if not any(checkin <= night < checkout for checkin, checkout in hotel_ranges)
    ]


def group_trips(segments: list[Segment]) -> list[Trip]:
    """Chain segments into trips: two segments join the same trip when the
    gap between one's effective end and the next's start is <= `TRIP_GAP`
    (48h), transitively, so a multi-city trip whose first and last legs
    are days apart still groups as one trip through its middle legs.

    Sorting happens here, so the result is the same regardless of the
    input order (overlapping or out of chronological order) -- callers
    never need to pre-sort. Overlapping segments (e.g. a hotel night
    spanning two flights) merge correctly because the running trip end is
    the max end seen so far, not just the previous segment's end.
    """
    if not segments:
        return []
    ordered = sorted(
        segments,
        key=lambda s: (_aware(s.start), _effective_end(s), s.kind, _normalize_confirmation(s.confirmation)),
    )
    groups: list[list[Segment]] = [[ordered[0]]]
    current_end = _effective_end(ordered[0])
    for seg in ordered[1:]:
        seg_end = _effective_end(seg)
        if _aware(seg.start) - current_end <= TRIP_GAP:
            groups[-1].append(seg)
            current_end = max(current_end, seg_end)
        else:
            groups.append([seg])
            current_end = seg_end
    return [Trip(segments=g, gaps=_find_gaps(g)) for g in groups]


def trip_key(segments: list[Segment]) -> str:
    """Stable filename-safe identifier for a trip, derived only from its
    segments' own identity (kind, confirmation, start) -- never from the
    wall clock or this run's ordinal position in `trips`.

    A timestamp-plus-index name (`travel-<HHMMSS>-trip1.ics`) collides
    across two runs that land in the same second, silently overwriting an
    unrelated trip's file since the index depends on that run's own
    ordering, not on which trip it is. Keying on content instead means
    two different trips (by definition, different segments) practically
    never collide, and the SAME trip re-run always maps to the same file
    -- the intended idempotent overwrite, same rationale as `to_event`'s
    stable UID. Order of `segments` doesn't matter (sorted before hashing).
    """
    parts = sorted(
        f"{s.kind}\x1f{_normalize_confirmation(s.confirmation)}\x1f{s.start.isoformat()}" for s in segments
    )
    digest = hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def to_event(seg: Segment) -> Event:
    title = f"{seg.kind.title()} — {seg.provider}"
    if seg.origin and seg.destination:
        title += f" ({seg.origin} → {seg.destination})"
    description_lines = [f"Extracted from: {seg.source_subject}"]
    if seg.confirmation:
        description_lines.append(f"Confirmation: {seg.confirmation}")
    if not seg.tz_known:
        description_lines.append("Time zone unknown -- treated as local system time.")
    if seg.source_url:
        description_lines.append(seg.source_url)
    return Event(
        uid=make_uid(seg.confirmation, seg.kind, seg.start.isoformat()),
        title=title,
        start=seg.start,
        end=seg.end,
        all_day=False,
        location=seg.address or seg.destination or seg.origin,
        description="\n".join(description_lines),
    )


def _fmt_when(dt: datetime, tz_known: bool) -> str:
    text = dt.strftime("%Y-%m-%d %H:%M")
    return text if tz_known else f"{text} (zone unknown)"


def _trip_span(trip: Trip) -> tuple[datetime, datetime]:
    return min(_aware(s.start) for s in trip.segments), max(_effective_end(s) for s in trip.segments)


def build_report(
    trips: list[Trip],
    past: list[Segment],
    cancelled: list[Segment],
    unparseable: list[UnparseableItem],
    failed: list[FailedMessage],
    message_count: int,
    ics_paths: list[Path],
    output_dir: Path,
) -> Report:
    total_segments = sum(len(t.segments) for t in trips)
    r = Report(
        "Travel itinerary",
        kicker="lmsa · Gmail, last 120 days",
        subtitle="Upcoming flight, hotel, car, and train confirmations, grouped into trips.",
        sticker=f"{len(trips)} trip(s)" if trips else "no upcoming travel",
        sticker_tone="acc" if trips else "muted",
        source="Gmail IMAP",
    )
    r.tiles(
        [
            Tile("Messages scanned", message_count, tone="info"),
            Tile("Trips", len(trips), tone="acc"),
            Tile("Segments", total_segments, tone="acc"),
            Tile("Past (dropped)", len(past), tone="muted"),
            Tile("Cancelled (removed)", len(cancelled), tone="muted"),
            Tile("Unparseable", len(unparseable), tone="warn" if unparseable else "muted"),
            Tile("Failed messages", len(failed), tone="hot" if failed else "muted"),
        ]
    )

    if not trips:
        r.section("Trips").text("No upcoming travel found.")

    for i, (trip, ics_path) in enumerate(zip(trips, ics_paths), 1):
        start, end = _trip_span(trip)
        title = f"Trip {i}: {start:%b %d} – {end:%b %d, %Y}"
        section = r.section(title, tone="acc")
        try:
            ics_link: str | None = layout.link(ics_path, output_dir)
        except ValueError:
            ics_link = ics_path.name
        section.cards([Card(ics_path.name, meta=f"{len(trip.segments)} segment(s)", href=ics_link, tone="acc")])
        section.table(
            ["When", "Kind", "Provider", "Confirmation", "From → To / Address", "Source"],
            [
                [
                    _fmt_when(_aware(s.start), s.tz_known),
                    s.kind,
                    s.provider,
                    s.confirmation or "—",
                    (f"{s.origin or '?'} → {s.destination}" if s.destination else (s.address or "—")),
                    Link(s.source_subject, href=safe_href(s.source_url)),
                ]
                for s in sorted(trip.segments, key=lambda s: _aware(s.start))
            ],
        )
        if trip.gaps:
            section.callout(
                "No hotel found covering: " + ", ".join(g.isoformat() for g in trip.gaps),
                label="possible gap",
                tone="warn",
            )

    if cancelled:
        r.section("Cancelled (removed from trips)", tone="muted").table(
            ["Kind", "Provider", "Confirmation", "Source"],
            [[s.kind, s.provider, s.confirmation or "—", s.source_subject] for s in cancelled],
        )

    if unparseable:
        r.section("Unparseable segments", tone="warn").callout(
            "\n".join(
                f"{u.kind} · {u.provider} — {u.source_subject!r}: {u.reason}" for u in unparseable
            ),
            label="dropped — invalid kind or start",
            tone="warn",
        )

    if failed:
        r.section("Failed messages", tone="warn").callout(
            "\n".join(f"{f.subject!r}: {f.error}" for f in failed),
            label="extraction failed",
            tone="warn",
        )

    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered extractions and ask the model about every message again",
    )
    args = parser.parse_args(argv)

    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    output_dir = settings().output_dir

    with Gmail() as gmail:
        messages = gmail.search(SEARCH_QUERY, limit=SEARCH_LIMIT)

    with Store() as store:
        last_ok = store.last_successful_run("tasks.travel")
        version = memo_version(
            MEMO_VERSION, prompt_hash("travel/extract_system"), SCHEMA, BODY_LIMIT, model_id(lm)
        )
        memo = Memo(store.conn, "travel", version, fresh=args.fresh)
        with store.record_run("tasks.travel") as run:
            found, unparseable, failed = gather(lm, messages, memo)
            kept, cancelled = resolve_segments(found)
            upcoming, past = drop_past(kept, now)
            trips = group_trips(upcoming)

            ics_paths: list[Path] = []
            for trip in trips:
                events = [to_event(s) for s in sorted(trip.segments, key=lambda s: _aware(s.start))]
                path = write_ics(events, layout.placed(output_dir, f"travel-{trip_key(trip.segments)}.ics"))
                ics_paths.append(path)

            rpt = build_report(trips, past, cancelled, unparseable, failed, len(messages), ics_paths, output_dir)
            if len(messages) >= SEARCH_LIMIT:
                rpt.section("Search limit", tone="warn").callout(
                    f"The search returned its cap of {SEARCH_LIMIT} messages (newest first), "
                    "so older mail in the last 120 days may be missing.",
                    label="truncated",
                )
            add_footer(rpt, memo, last_ok)
            report_path = rpt.write(out_dir=output_dir)

            run.report_path = report_path
            run.summary = {
                "messages": len(messages),
                "segments_found": len(found),
                "trips": len(trips),
                "upcoming_segments": len(upcoming),
                "past_dropped": len(past),
                "cancelled_removed": len(cancelled),
                "unparseable": len(unparseable),
                "failed_messages": len(failed),
                **memo.summary(),
            }
            print(report_path)
            for path in ics_paths:
                print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
