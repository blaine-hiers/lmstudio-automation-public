"""Deadline and event extractor: scans the last 7 days of mail for concrete
deadlines and calendar events, and writes them out as a downloadable .ics.

    python -m tasks.deadlines [--fresh]

Read-only: only reads Gmail over IMAP, never labels/sends/deletes. Each
message is asked (locally, via `lmsa.LMStudio.extract`) for the deadlines
and events it mentions; a message whose extraction fails is recorded and
skipped rather than aborting the run. Extracted dates are untrusted
model output -- parsed strictly, with anything that doesn't parse dropped
into its own report section instead of crashing. Past items (relative to
now) are dropped, and items are deduped on (normalized title, date) since
the same deadline is often mentioned in more than one email.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from lmsa import Card, LMStudio, Report, Store, Tile, handled
from lmsa import layout
from lmsa.config import settings
from lmsa.ics import Event, make_uid, write_ics
from lmsa.report import safe_href
from lmsa.memo import Memo, add_footer, cached, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

SEARCH_QUERY = "newer_than:7d"
SEARCH_LIMIT = 200
BODY_LIMIT = 4000
MEMO_VERSION = 1  # bump when extract_from_message()'s prompt building or result handling changes
KINDS = ("deadline", "event")

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "start": {"type": "string"},
        "end": {"type": ["string", "null"]},
        "all_day": {"type": "boolean"},
        "location": {"type": ["string", "null"]},
    },
    "required": ["title", "kind", "start", "end", "all_day", "location"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["items"],
    "additionalProperties": False,
}


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


EXTRACT_SYSTEM = load_prompt("deadlines/extract_system")


@dataclass
class DeadlineItem:
    title: str
    kind: str
    start: date | datetime
    end: date | datetime | None
    all_day: bool
    location: str | None
    source_msg_id: str
    source_subject: str
    source_url: str


@dataclass
class UnparseableItem:
    title: str
    kind: str
    raw_start: str
    source_subject: str
    source_url: str


@dataclass
class FailedMessage:
    subject: str
    msg_id: str
    error: str


class _BadDate(Exception):
    def __init__(self, title: str, kind: str, raw_start: str):
        super().__init__(f"unparseable start date {raw_start!r}")
        self.title = title
        self.kind = kind
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
        lambda: lm.extract(prompt, SCHEMA, system=EXTRACT_SYSTEM, name="deadlines"),
        valid=lambda r: isinstance(r, dict) and isinstance(r.get("items"), list),
    )
    return list(result.get("items") or [])


def _parse_when(value: str | None, all_day: bool) -> date | datetime | None:
    """Strict ISO 8601 parse. Returns None (never raises) on anything that
    doesn't parse -- the caller turns that into a per-item report entry
    rather than letting one bad model-produced string abort the run.

    All-day items parse a bare date (only the first 10 characters are
    considered, so a date-with-time string still works). Timed items
    parse a full ISO 8601 datetime; a bare date string is also accepted
    there (`datetime.fromisoformat` treats it as midnight) since the
    model is asked for -- but not relied on to always give -- a specific
    time whenever `all_day` is false.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if all_day:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_item(raw: dict, msg: Message) -> DeadlineItem:
    title = str(raw.get("title") or "(untitled)")
    kind = raw.get("kind") if raw.get("kind") in KINDS else "event"
    all_day = bool(raw.get("all_day"))
    start_raw = raw.get("start") or ""
    start = _parse_when(start_raw, all_day)
    if start is None:
        raise _BadDate(title, kind, start_raw)
    end = _parse_when(raw.get("end"), all_day)
    return DeadlineItem(
        title=title,
        kind=kind,
        start=start,
        end=end,
        all_day=all_day,
        location=raw.get("location") or None,
        source_msg_id=msg.msg_id,
        source_subject=msg.subject,
        source_url=msg.web_url,
    )


def gather(
    lm: LMStudio, messages: list[Message], memo: Memo | None = None
) -> tuple[list[DeadlineItem], list[UnparseableItem], list[FailedMessage]]:
    """Extract items from every message, isolating each message's own failure.

    A message whose `extract()` call itself fails (server down, malformed
    JSON, ...) is recorded in `failed` and skipped -- it never aborts the
    rest of the mailbox scan. Within a message's own items, one item with
    an unparseable date goes to `unparseable`, the rest still come through.
    """
    items: list[DeadlineItem] = []
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
                items.append(_build_item(raw, msg))
            except _BadDate as exc:
                unparseable.append(
                    UnparseableItem(
                        title=exc.title,
                        kind=exc.kind,
                        raw_start=exc.raw_start,
                        source_subject=msg.subject,
                        source_url=msg.web_url,
                    )
                )
    return items, unparseable, failed


def _effective_start(item: DeadlineItem, tz: timezone | None = None) -> datetime:
    """`item.start` as an aware datetime, for comparison/sorting only.

    All-day items compare at local midnight of that date. A naive timed
    value is presumed local system time (matches `lmsa.ics`), then made
    aware via `astimezone()` -- Python's own naive-is-local convention.
    `tz` is only for tests that need a fixed zone instead of the machine's.
    """
    start = datetime.combine(item.start, time.min) if item.all_day else item.start
    return start.astimezone(tz)


def _date_key(item: DeadlineItem) -> str:
    d = item.start if item.all_day else item.start.date()
    return d.isoformat()


def _normalize_title(title: str) -> str:
    return " ".join(title.split()).lower()


def _effective_end(item: DeadlineItem, tz: timezone | None = None) -> datetime:
    """When an item stops being upcoming: an all-day item lasts until the end of
    its day (so today's all-day deadline still shows), a timed one at its start."""
    if item.all_day:
        return datetime.combine(item.start + timedelta(days=1), time.min).astimezone(tz)
    return _effective_start(item, tz)


def drop_past(items: list[DeadlineItem], now: datetime) -> tuple[list[DeadlineItem], list[DeadlineItem]]:
    """Split into (upcoming, past) by comparing each item's effective start to `now`."""
    upcoming, past = [], []
    for item in items:
        (past if _effective_end(item) <= now else upcoming).append(item)
    return upcoming, past


def dedupe(items: list[DeadlineItem]) -> tuple[list[DeadlineItem], list[DeadlineItem]]:
    """Drop items sharing a (normalized title, date) key with one already kept.

    `items` is expected newest-message-first (as `Gmail.search` returns
    them), so the first occurrence of a key -- kept -- is the most
    recently mentioned instance of that deadline/event.
    """
    seen: set[tuple[str, str]] = set()
    kept, dupes = [], []
    for item in items:
        key = (_normalize_title(item.title), _date_key(item))
        if key in seen:
            dupes.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, dupes


def to_event(item: DeadlineItem) -> Event:
    description = f"Extracted from: {item.source_subject}"
    if item.source_url:
        description += f"\n{item.source_url}"
    return Event(
        uid=make_uid(item.source_msg_id, _normalize_title(item.title), _date_key(item)),
        title=item.title,
        start=item.start,
        end=item.end,
        all_day=item.all_day,
        location=item.location,
        description=description,
    )


def _fmt_when(item: DeadlineItem) -> str:
    return item.start.isoformat() if item.all_day else item.start.strftime("%Y-%m-%d %H:%M")


def build_report(
    kept: list[DeadlineItem],
    past: list[DeadlineItem],
    duplicates: list[DeadlineItem],
    unparseable: list[UnparseableItem],
    failed: list[FailedMessage],
    message_count: int,
    ics_path: Path,
    output_dir: Path,
    done: list[Message] | None = None,
) -> Report:
    """`done`: messages already checked off (#61) -- not scanned, shown
    only behind the "N handled hidden" toggle."""
    r = Report(
        "Deadlines & events",
        kicker="lmsa · Gmail, last 7 days",
        subtitle="Concrete deadlines and calendar events found in recent mail.",
        sticker=f"{len(kept)} upcoming" if kept else "nothing upcoming",
        sticker_tone="acc" if kept else "muted",
        source="Gmail IMAP",
    )
    r.tiles(
        [
            Tile("Messages scanned", message_count, tone="info"),
            Tile("Upcoming", len(kept), tone="acc"),
            Tile("Past (dropped)", len(past), tone="muted"),
            Tile("Duplicates (dropped)", len(duplicates), tone="muted"),
            Tile("Unparseable dates", len(unparseable), tone="warn" if unparseable else "muted"),
            Tile("Failed messages", len(failed), tone="hot" if failed else "muted"),
        ]
    )

    try:
        ics_link: str | None = layout.link(ics_path, output_dir)
    except ValueError:
        ics_link = ics_path.name
    r.section("Calendar file").cards(
        [Card(f"{ics_path.name}", meta=f"{len(kept)} event(s)", href=ics_link, tone="acc")]
    )

    if kept:
        r.section("Upcoming, chronologically", tone="acc").table(
            ["When", "Kind", "Title", "Location"],
            [[_fmt_when(i), i.kind, i.title, i.location or "—"] for i in kept],
        )
        r.section("Sources").cards(
            [
                Card(
                    i.title,
                    meta=f"{i.kind} · {_fmt_when(i)}",
                    body=i.source_subject,
                    href=safe_href(i.source_url),
                    tone="acc",
                    msg_id=i.source_msg_id or None,
                )
                for i in kept
            ]
        )
    else:
        r.section("Upcoming, chronologically").text("No upcoming deadlines or events found.")

    if done:
        r.section("Checked off", tone="muted").handled_hidden(handled.hidden_cards(done))

    if unparseable:
        r.section("Unparseable dates", tone="warn").callout(
            "\n".join(
                f"{i.title} ({i.kind}) — {i.source_subject!r}: start={i.raw_start!r}" for i in unparseable
            ),
            label="dropped — could not parse start date",
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
        last_ok = store.last_successful_run("tasks.deadlines")
        version = memo_version(
            MEMO_VERSION, prompt_hash("deadlines/extract_system"), SCHEMA, BODY_LIMIT, model_id(lm)
        )
        memo = Memo(store.conn, "deadlines", version, fresh=args.fresh)
        with store.record_run("tasks.deadlines") as run:
            messages, done = handled.split(store.conn, messages)
            found, unparseable, failed = gather(lm, messages, memo)
            upcoming, past = drop_past(found, now)
            kept, duplicates = dedupe(upcoming)
            kept.sort(key=_effective_start)

            ics_file = write_ics([to_event(i) for i in kept], layout.placed(output_dir, f"deadlines-{now:%Y%m%d-%H%M%S}.ics"))
            rpt = build_report(kept, past, duplicates, unparseable, failed, len(messages), ics_file, output_dir,
                               done=done)
            sources = {i.source_msg_id for i in kept}
            handled.seen(store.conn, "deadlines", run.id, [m for m in messages if m.msg_id in sources] + done)
            if len(messages) >= SEARCH_LIMIT:
                rpt.section("Search limit", tone="warn").callout(
                    f"The search returned its cap of {SEARCH_LIMIT} messages (newest first), "
                    "so older mail from the last 7 days may be missing.",
                    label="truncated",
                )
            add_footer(rpt, memo, last_ok)
            report_path = rpt.write(out_dir=output_dir)

            run.report_path = report_path
            run.summary = {
                "messages": len(messages),
                "items_found": len(found),
                "upcoming": len(kept),
                "past_dropped": len(past),
                "duplicates_dropped": len(duplicates),
                "unparseable": len(unparseable),
                "failed_messages": len(failed),
                "handled_hidden": len(done),
                **memo.summary(),
            }
            print(report_path)
            print(ics_file)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
