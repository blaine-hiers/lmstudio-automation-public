"""Shared epoch-second Gmail search-cursor helpers.

Gmail's day-granularity `after:YYYY/MM/DD` search silently drops
time-of-day, so every bound here is a whole-second Unix epoch timestamp
(`after:1727200000`), which Gmail search also accepts.

Shared cursor scheme (`GmailCursor` + `RetrySet`), used by every
Gmail-cursor task (receipts, packages, job_tracker, attachments). It relies
on nothing about UID order or on how a limit cuts off a newest-first list:

1. **Scan the whole window, cheaply.** Search `after:<since - OVERLAP>`
   with no limit and fetch only X-GM-MSGID + INTERNALDATE for every match
   (`Gmail.search_meta`) -- no bodies.
2. **Pick the oldest unprocessed.** Drop msg_ids already in the task's
   applied ledger or its retry set, sort the rest by INTERNALDATE (the
   delivery time Gmail's `after:` filters on -- never the sender's Date
   header), tie-break by msg_id, and fully fetch only the first `limit`.
3. **Advance to what's provably done.** The forward cursor `since` becomes
   the run's start if nothing unprocessed is left in the window, else the
   INTERNALDATE of the oldest message still unprocessed. It never moves
   past an unprocessed message, so an over-limit backlog is simply worked
   through oldest-first over the next runs.

`OVERLAP` makes the next window reach back a little before `since`, which
covers same-second siblings, mail Gmail indexes late (its INTERNALDATE is
before the run's start but it only shows up in search afterwards), and
clock skew; re-seeing a message is harmless because the applied ledger
filters it out before any model call.

Failed items are retried: `RetrySet` keeps the msg_ids of per-item
failures (they are *not* marked applied), refetches them next run by
X-GM-MSGID (`Gmail.fetch_by_msgids`), and gives up after
`MAX_RETRY_ATTEMPTS`, reporting it. Retried and given-up ids are excluded
from the window scan, so they never hold the forward cursor back.

State from the older catch-up scheme (`catchup_before` / `catchup_ceiling`
kv keys) is migrated by keeping `since` = min(stored since, catchup_before)
-- nothing in flight is skipped -- and the old keys are deleted on the next
successful advance.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lmsa.report import Report
    from lmsa.sources.gmail import Message, MessageMeta

MAX_RETRY_ATTEMPTS = 3
# How far before `since` each scan reaches back (same-second siblings,
# Gmail index lag, clock skew). Re-seen messages are filtered by the ledger.
OVERLAP = timedelta(minutes=10)
_LEGACY_KEYS = ("catchup_before", "catchup_ceiling")


def epoch_query_bound(moment: datetime) -> int:
    """Whole-second Unix timestamp for a Gmail `after:`/`before:` search
    bound. `moment` must be an aware datetime."""
    return int(moment.timestamp())


def coerce_aware_moment(raw: str) -> datetime:
    """Parse a stored kv cursor value into an aware datetime, whether it's
    the current epoch-precise format (a full ISO datetime string) or the
    old day-granularity format some tasks used to store (a bare ISO date,
    e.g. "2026-01-15").

    A bare date is migrated to midnight *local* time of that date -- the
    same wall-clock day the old `before:YYYY/MM/DD` query meant, just
    expressed as a precise instant so `epoch_query_bound` can use it.
    Raises `ValueError` if `raw` isn't a valid ISO date or datetime.
    """
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = datetime.combine(parsed.date(), time.min).astimezone()
    return parsed


def parse_stored_moment(raw: Any) -> datetime | None:
    """A stored cursor value -> aware datetime, or None if absent/garbage.
    Accepts the current ISO-datetime format, a legacy bare ISO date (see
    `coerce_aware_moment`), and a legacy integer epoch-seconds value."""
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, timezone.utc)
    if isinstance(raw, str):
        try:
            return coerce_aware_moment(raw)
        except ValueError:
            return None
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _kv_get(conn: sqlite3.Connection, ns: str, key: str) -> Any:
    row = conn.execute("SELECT value_json FROM kv WHERE ns=? AND key=?", (ns, key)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def _kv_set(conn: sqlite3.Connection, ns: str, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(ns, key) DO UPDATE SET value_json=excluded.value_json, updated=excluded.updated",
        (ns, key, json.dumps(value), _now().isoformat()),
    )


@dataclass
class Window:
    """One scan's result: `candidates` are every unprocessed match, oldest
    first; `batch` is the first `limit` of them (this run's work)."""

    candidates: list["MessageMeta"]
    batch: list["MessageMeta"]

    @property
    def pending(self) -> int:
        """Unprocessed matches left for a later run because of the limit."""
        return len(self.candidates) - len(self.batch)


class GmailCursor:
    """kv-backed forward cursor for one task (see the module docstring).

        cursor = GmailCursor(store.conn, "receipts")
        cursor.start()                                  # before the scan
        window = cursor.scan(gmail, QUERY, default_since=..., limit=...,
                             skip=lambda msg_id: applied(msg_id) or msg_id in retry.items)
        messages = cursor.fetch(gmail, window)          # oldest first
        ... process; mark each concluded msg_id applied, or retry.failed() ...
        cursor.advance(window, done=lambda msg_id: applied(msg_id) or msg_id in retry.items)

    `since_key` keeps each task's existing kv key for the cursor.
    """

    def __init__(self, conn: sqlite3.Connection, ns: str, *, since_key: str = "last_run"):
        self.conn = conn
        self.ns = ns
        self.since_key = since_key
        self.since = parse_stored_moment(_kv_get(conn, ns, since_key))
        # Migrate the older catch-up scheme: mail below catchup_before may
        # never have been seen, so start no later than it.
        legacy_before = parse_stored_moment(_kv_get(conn, ns, "catchup_before"))
        if self.since is not None and legacy_before is not None:
            self.since = min(self.since, legacy_before)
        self.started: datetime | None = None

    def start(self) -> datetime:
        """Stamp the run's start. Call immediately *before* the scan."""
        self.started = _now()
        return self.started

    def scan(
        self,
        gmail: Any,
        base_query: str,
        *,
        default_since: datetime,
        limit: int,
        skip: Callable[[str], bool],
        folder: str | None = None,
    ) -> Window:
        """Every match in `[since - OVERLAP, now)` minus `skip`ped msg_ids,
        oldest first by INTERNALDATE; `batch` is the first `limit` (all of
        them if `limit` <= 0). Metadata only -- no bodies are fetched."""
        started = self.started or self.start()
        lower = (self.since or default_since) - OVERLAP
        query = f"{base_query} after:{epoch_query_bound(lower)}"
        metas = gmail.search_meta(query) if folder is None else gmail.search_meta(query, folder)
        seen: set[str] = set()
        candidates = []
        for meta in metas:
            if meta.msg_id in seen or skip(meta.msg_id):
                continue
            seen.add(meta.msg_id)
            candidates.append(meta)
        candidates.sort(key=lambda m: (m.internal_date or started, m.msg_id))
        batch = candidates[:limit] if limit > 0 else list(candidates)
        return Window(candidates=candidates, batch=batch)

    def fetch(self, gmail: Any, window: Window, folder: str | None = None) -> list["Message"]:
        """Full messages for `window.batch`, oldest first."""
        uids = [m.uid for m in window.batch]
        if not uids:
            return []
        return gmail.fetch_uids(uids) if folder is None else gmail.fetch_uids(uids, folder)

    def advance(self, window: Window, *, done: Callable[[str], bool]) -> None:
        """Stage the new `since` (caller commits): the run's start if every
        candidate is `done` (applied, or handed to the retry set), else the
        INTERNALDATE of the oldest candidate that isn't. Clears legacy
        catch-up keys."""
        started = self.started or _now()
        remaining = [m for m in window.candidates if not done(m.msg_id)]
        if remaining:
            self.since = min(m.internal_date or started for m in remaining)
        else:
            self.since = started
        _kv_set(self.conn, self.ns, self.since_key, self.since.isoformat())
        self.conn.execute(
            f"DELETE FROM kv WHERE ns=? AND key IN ({','.join('?' * len(_LEGACY_KEYS))})",
            (self.ns, *_LEGACY_KEYS),
        )


@dataclass
class RetryItem:
    msg_id: str
    subject: str
    error: str
    attempts: int
    gave_up: bool = False


class RetrySet:
    """Per-task persisted set of msg_ids that failed and should be retried.

    `failed()` counts an attempt; after `max_attempts` the item is marked
    "gave up" (kept, so it isn't retried again, and listed in this run's
    report via `add_retry_notes`). `succeeded()` drops it -- call it for any
    item the task reached a conclusion on (including "not relevant").
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        ns: str,
        *,
        key: str = "retry",
        max_attempts: int = MAX_RETRY_ATTEMPTS,
    ):
        self.conn = conn
        self.ns = ns
        self.key = key
        self.max_attempts = max_attempts
        self.items: dict[str, RetryItem] = {}
        raw = _kv_get(conn, ns, key)
        if isinstance(raw, dict):
            for msg_id, v in raw.items():
                if not isinstance(v, dict):
                    continue
                try:
                    attempts = int(v.get("attempts", 0) or 0)
                except (TypeError, ValueError):
                    attempts = 0
                self.items[str(msg_id)] = RetryItem(
                    msg_id=str(msg_id),
                    subject=str(v.get("subject", "")),
                    error=str(v.get("error", "")),
                    attempts=attempts,
                    gave_up=bool(v.get("gave_up", False)),
                )
        self.gave_up_now: list[RetryItem] = []
        self.fetch_error: str | None = None

    def pending(self) -> list[str]:
        return [i.msg_id for i in self.items.values() if not i.gave_up]

    def fetch(self, gmail: Any) -> list["Message"]:
        """Refetch every pending item by X-GM-MSGID. Never raises: a fetch
        failure is kept on `fetch_error` (shown in the report) and costs no
        attempt; an item no longer in the mailbox counts as a failed one."""
        ids = self.pending()
        if not ids:
            return []
        try:
            messages = list(gmail.fetch_by_msgids(ids))
        except Exception as exc:  # noqa: BLE001 - one bad refetch mustn't abort the run
            self.fetch_error = f"{type(exc).__name__}: {exc}"
            return []
        found = {m.msg_id for m in messages}
        for msg_id in ids:
            if msg_id not in found:
                self.failed(msg_id, self.items[msg_id].subject, "no longer found in the mailbox")
        return messages

    def failed(self, msg_id: str, subject: str, error: str) -> None:
        item = self.items.get(msg_id) or RetryItem(msg_id=msg_id, subject=subject, error=error, attempts=0)
        was_given_up = item.gave_up
        item.subject = subject or item.subject
        item.error = error
        item.attempts += 1
        if item.attempts >= self.max_attempts:
            item.gave_up = True
            if not was_given_up:
                self.gave_up_now.append(item)
        self.items[msg_id] = item

    def succeeded(self, msg_id: str) -> None:
        self.items.pop(msg_id, None)

    def save(self) -> None:
        """Stage the set into kv (caller commits)."""
        _kv_set(self.conn, self.ns, self.key, {
            i.msg_id: {"subject": i.subject, "error": i.error, "attempts": i.attempts, "gave_up": i.gave_up}
            for i in self.items.values()
        })


def merge_retried(searched: list["Message"], retried: list["Message"]) -> list["Message"]:
    """This run's work list: the search results, then retried items not
    already among them (a retried msg_id can reappear in the search)."""
    seen = {m.msg_id for m in searched}
    return list(searched) + [m for m in retried if m.msg_id not in seen]


def add_pending_note(report: "Report", window: Window | None) -> None:
    """Say so when the per-run limit left unprocessed mail in the window."""
    if window is None or not window.pending:
        return
    report.section("More pending", tone="warn").callout(
        f"{window.pending} more matching message(s) are pending; this run processed the "
        f"oldest {len(window.batch)} and the next run will continue from there.",
        label="pending", tone="warn",
    )


def add_retry_notes(report: "Report", retry: RetrySet | None) -> None:
    """Report what the retry set is doing: items given up on this run, how
    many failed items are queued for the next run, and a refetch failure."""
    if retry is None:
        return
    if retry.gave_up_now:
        report.section(f"Gave up after {retry.max_attempts} attempts", tone="hot").table(
            ["Subject", "Last error"],
            [[i.subject or i.msg_id, i.error] for i in retry.gave_up_now],
        )
    pending = retry.pending()
    if pending:
        report.section("Retrying next run", tone="warn").text(
            f"{len(pending)} failed item(s) will be refetched and retried on the next run "
            f"(up to {retry.max_attempts} attempts each)."
        )
    if retry.fetch_error:
        report.section("Retry refetch failed", tone="warn").callout(
            f"Refetching earlier failures failed this run ({retry.fetch_error}); they stay queued.",
            label="retry", tone="warn",
        )
