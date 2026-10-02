"""Checked-off ("handled") emails, shared by reports, tasks and the dashboard (#61).

Checking an email off stores it locally and hides it from later reports.
Optionally, `tasks.inbox_actions` also archives it / marks it read in Gmail
-- dry-run by default, every write logged per run before it is sent, and
undoable by run id (the labeler's pattern).

    with Store() as db:
        seen(db.conn, "inbox_brief", run.id, messages)       # cards shown
        open_, done = split(db.conn, messages)               # hide handled
        mark(db.conn, msg_id)                                # dashboard POST

Tables (`lmsa.store._NAMED_MIGRATIONS["handled"]`):
    handled(message_id PK, uid, folder, subject, sender, handled_at,
            source_task, gmail_action NULL, run_id NULL)
    handled_seen(source_task, message_id, run_id, ...) PK(source_task,
            message_id) -- every email card a report showed; a check-off is
            only accepted for a message listed here.
    handled_gmail_log(id, run_id, message_id, action, sender, logged_at,
            applied_at, undone_at) -- Gmail writes, logged before sending.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, TypeVar
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from .report import Card

DEFAULT_FOLDER = "[Gmail]/All Mail"  # lmsa.sources.gmail.DEFAULT_FOLDER (not imported: keeps imapclient optional here)
INBOX_TASKS = ("inbox_brief", "followup_radar", "deadlines")
ACTIONS = {"archive": "unarchive", "mark_read": "mark_unread"}  # action -> its undo
FILTER_THRESHOLD = 3
FILTER_DAYS = 30
MSG_ID_RE = re.compile(r"\d{1,20}")  # Gmail X-GM-MSGID: a 64-bit decimal
_ADDR_RE = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime | None = None) -> str:
    return (when or _now()).isoformat()


def _bare(task: str) -> str:
    return task[len("tasks."):] if task.startswith("tasks.") else task


def _sender(msg: object) -> str:
    """The sender to show / count for filter suggestions. An item with a
    `recipient` instead (follow-up radar: mail *you* sent) shows "to ..." --
    never a valid address, so it can't turn into a filter for yourself."""
    sender = getattr(msg, "from_addr", "") or getattr(msg, "sender", "")
    if not sender and getattr(msg, "recipient", ""):
        return f"to {msg.recipient}"  # type: ignore[attr-defined]
    return str(sender or "")


# -- what reports showed / what is handled ---------------------------------


def seen(conn: sqlite3.Connection, source_task: str, run_id: int | None, messages: Iterable[object]) -> int:
    """Record the email cards a task's report shows (`Message`-like objects:
    `msg_id`, `subject`, `from_addr`, optional `web_url`/`uid`/`folder`).
    Idempotent per (task, message): a rerun just moves it to the newest run.
    Messages without a valid X-GM-MSGID are skipped. Returns rows written."""
    now = _iso()
    n = 0
    for m in messages:
        msg_id = str(getattr(m, "msg_id", "") or "")
        if not MSG_ID_RE.fullmatch(msg_id):
            continue
        conn.execute(
            "INSERT INTO handled_seen (source_task, message_id, run_id, uid, folder, subject, sender, web_url, "
            "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_task, message_id) DO UPDATE SET run_id=excluded.run_id, "
            "subject=excluded.subject, web_url=excluded.web_url, last_seen=excluded.last_seen, "
            # A follow-up (mail you sent) is identified by "to <recipient>";
            # never let a re-record swap that for your own From address.
            "sender=CASE WHEN handled_seen.sender LIKE 'to %' THEN handled_seen.sender ELSE excluded.sender END",
            (
                _bare(source_task), msg_id, run_id, getattr(m, "uid", None),
                getattr(m, "folder", None) or DEFAULT_FOLDER,
                str(getattr(m, "subject", "") or ""), _sender(m),
                str(getattr(m, "web_url", "") or ""), now, now,
            ),
        )
        n += 1
    conn.commit()
    return n


def is_known(conn: sqlite3.Connection, msg_id: object) -> bool:
    """True if some report showed this message as an email card."""
    if not isinstance(msg_id, str) or not MSG_ID_RE.fullmatch(msg_id):
        return False
    return conn.execute("SELECT 1 FROM handled_seen WHERE message_id = ? LIMIT 1", (msg_id,)).fetchone() is not None


def handled_ids(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT message_id FROM handled")}


def split(
    conn: sqlite3.Connection, items: Iterable[T], key: Callable[[T], str] = lambda m: m.msg_id  # type: ignore[attr-defined]
) -> tuple[list[T], list[T]]:
    """(open, handled): `items` partitioned by whether their message id was
    checked off. Order is preserved in both lists."""
    done = handled_ids(conn)
    open_, hidden = [], []
    for item in items:
        (hidden if key(item) in done else open_).append(item)
    return open_, hidden


def mark(conn: sqlite3.Connection, msg_id: str, *, source_task: str | None = None) -> bool:
    """Check a message off. Only a message a report showed (`is_known`) is
    accepted; returns False (and writes nothing) otherwise. Idempotent."""
    if not is_known(conn, msg_id):
        return False
    row = conn.execute(
        "SELECT source_task, uid, folder, subject, sender FROM handled_seen WHERE message_id = ? "
        "ORDER BY (source_task = ?) DESC, last_seen DESC LIMIT 1",
        (msg_id, source_task or ""),
    ).fetchone()
    task, uid, folder, subject, sender = row
    conn.execute(
        "INSERT INTO handled (message_id, uid, folder, subject, sender, handled_at, source_task) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(message_id) DO NOTHING",
        (msg_id, uid, folder, subject, sender, _iso(), task),
    )
    conn.commit()
    return True


def unmark(conn: sqlite3.Connection, msg_id: str) -> bool:
    """Undo a check-off locally (it shows up in reports again). A Gmail
    archive / mark-read already done is *not* reverted here -- that is
    `tasks.inbox_actions --undo RUN_ID`. Returns whether a row was removed."""
    cur = conn.execute("DELETE FROM handled WHERE message_id = ?", (msg_id,))
    conn.commit()
    return cur.rowcount > 0


def hidden_cards(messages: Iterable[object], *, tone: str = "muted") -> list[Card]:
    """Minimal, pre-checked cards for handled messages (subject, sender,
    link) -- what a report's "N handled hidden" toggle shows."""
    return [
        Card(
            title=str(getattr(m, "subject", "") or "(no subject)"),
            meta=str(getattr(m, "from_name", "") or _sender(m)),
            tone=tone,
            href=str(getattr(m, "web_url", "") or "") or None,
            msg_id=str(getattr(m, "msg_id", "")),
            handled=True,
        )
        for m in messages
    ]


# -- dashboard Inbox panel ------------------------------------------------------


def _latest_runs(conn: sqlite3.Connection, tasks: Iterable[str]) -> dict[str, int]:
    out = {}
    for task in tasks:
        row = conn.execute("SELECT MAX(run_id) FROM handled_seen WHERE source_task = ?", (task,)).fetchone()
        if row and row[0] is not None:
            out[task] = row[0]
    return out


def open_items(conn: sqlite3.Connection, tasks: Iterable[str] = INBOX_TASKS) -> list[dict]:
    """Not-yet-handled emails from the latest run of each task, deduplicated
    by message (first task in `tasks` order wins), newest first."""
    out: dict[str, dict] = {}
    for task, run_id in _latest_runs(conn, tasks).items():
        rows = conn.execute(
            "SELECT s.message_id, s.subject, s.sender, s.web_url, s.last_seen FROM handled_seen s "
            "LEFT JOIN handled h ON h.message_id = s.message_id "
            "WHERE s.source_task = ? AND s.run_id = ? AND h.message_id IS NULL ORDER BY s.last_seen DESC",
            (task, run_id),
        ).fetchall()
        for msg_id, subject, sender, web_url, last_seen in rows:
            out.setdefault(msg_id, {
                "msg_id": msg_id, "subject": subject or "(no subject)", "sender": sender or "",
                "web_url": web_url or "", "task": task, "seen": last_seen,
            })
    return list(out.values())


def recent_handled(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT message_id, subject, sender, handled_at, source_task, gmail_action, run_id FROM handled "
        "ORDER BY handled_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"msg_id": r[0], "subject": r[1] or "(no subject)", "sender": r[2] or "", "handled_at": r[3],
         "task": r[4] or "", "gmail_action": r[5], "run_id": r[6]}
        for r in rows
    ]


# -- Gmail archive / mark read (logged, undoable) ------------------------------
#
# Log protocol (the labeler's, per message): an 'intent' row is written
# before each Gmail call and marked right after it -- 'done' when the STORE
# was sent, 'failed' (with the error) when it raised, 'already' when Gmail
# reported it already archived / already read (nothing was written). `undo`
# acts only on 'done' rows, so it reverses exactly what the run changed.
# `pending` skips a message whose latest row for the action is 'already' or
# a not-undone 'done', and gives up on one after MAX_FAILURES failed
# attempts in a row (`gave_up`), so neither can fill --limit forever.


@dataclass
class Outcome:
    msg_id: str
    subject: str
    sender: str
    action: str
    error: str | None = None
    maybe_applied: bool = False
    changed: bool = False  # True: written; False with no error: already so

    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        return "done" if self.changed else "already"


ALREADY = {"archive": "already archived", "mark_read": "already read"}
MAX_FAILURES = 3
GAVE_UP = f"gave up after {MAX_FAILURES} failures -- check Gmail"


def _candidates(conn: sqlite3.Connection, action: str) -> list[tuple[dict, int]]:
    """Handled messages `action` isn't settled for -- the latest log row for
    (message, action) is neither 'already' nor a not-undone 'done' -- with
    the number of failed attempts since the last done/already row. Oldest
    check-off first."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    sql = (
        "SELECT h.message_id, h.subject, h.sender, h.folder, ("
        "  SELECT COUNT(*) FROM handled_gmail_log f WHERE f.message_id = h.message_id AND f.action = ?1 "
        "  AND f.status = 'failed' AND f.id > COALESCE((SELECT MAX(d.id) FROM handled_gmail_log d "
        "  WHERE d.message_id = h.message_id AND d.action = ?1 AND d.status IN ('done', 'already')), 0)"
        ") FROM handled h WHERE NOT EXISTS ("
        "  SELECT 1 FROM handled_gmail_log g WHERE g.id = (SELECT MAX(l.id) FROM handled_gmail_log l "
        "  WHERE l.message_id = h.message_id AND l.action = ?1) "
        "  AND (g.status = 'already' OR (g.status = 'done' AND g.undone_at IS NULL))"
        ") ORDER BY h.handled_at"
    )
    return [
        ({"msg_id": r[0], "subject": r[1] or "", "sender": r[2] or "", "folder": r[3] or DEFAULT_FOLDER}, r[4])
        for r in conn.execute(sql, (action,)).fetchall()
    ]


def pending(conn: sqlite3.Connection, action: str, limit: int | None = None) -> list[dict]:
    """Handled messages `action` still needs applying to, oldest check-off
    first: not already done (or found already so) and not given up on
    (`gave_up`). `limit` counts only these; None means all, and a limit
    below 1 is refused rather than read as "no cap"."""
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    rows = [row for row, failures in _candidates(conn, action) if failures < MAX_FAILURES]
    return rows if limit is None else rows[:limit]


def gave_up(conn: sqlite3.Connection, action: str) -> list[dict]:
    """Handled messages `action` failed on MAX_FAILURES times in a row
    (e.g. gone from All Mail): no longer retried, reported for a manual
    look in Gmail."""
    return [row for row, failures in _candidates(conn, action) if failures >= MAX_FAILURES]


def _call_one(gmail: object, method: str, msg_id: str) -> "bool | Exception":
    """One Gmail write for one message: True (changed), False (already so),
    or the exception -- whether the Gmail method returned it or raised it."""
    try:
        result = getattr(gmail, method)([msg_id])
    except Exception as exc:  # noqa: BLE001 - isolate per message
        return exc
    value = result.get(msg_id, LookupError("no result")) if isinstance(result, dict) else result
    return value if isinstance(value, (bool, Exception)) else bool(value)


def apply(conn: sqlite3.Connection, gmail: object, run_id: int, action: str, rows: list[dict]) -> list[Outcome]:
    """Write `action` ("archive" | "mark_read") to Gmail for `rows` (from
    `pending`), one message at a time, logging intent before and the outcome
    right after each write (see the log protocol above). One failure never
    stops the rest."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    outcomes = []
    for r in rows:
        log_id = conn.execute(
            "INSERT INTO handled_gmail_log (run_id, message_id, action, sender, status, logged_at) "
            "VALUES (?, ?, ?, ?, 'intent', ?)",
            (run_id, r["msg_id"], action, r["sender"], _iso()),
        ).lastrowid
        conn.commit()
        result = _call_one(gmail, action, r["msg_id"])
        o = Outcome(r["msg_id"], r["subject"], r["sender"], action)
        if isinstance(result, Exception):
            o.error = f"{type(result).__name__}: {result}"
            o.maybe_applied = bool(getattr(result, "maybe_applied", False))
            conn.execute("UPDATE handled_gmail_log SET status = 'failed', error = ? WHERE id = ?",
                         (o.error + (" (outcome unknown)" if o.maybe_applied else ""), log_id))
        elif result:
            o.changed = True
            conn.execute("UPDATE handled_gmail_log SET status = 'done', applied_at = ? WHERE id = ?",
                         (_iso(), log_id))
            conn.execute("UPDATE handled SET gmail_action = ?, run_id = ? WHERE message_id = ?",
                         (action, run_id, r["msg_id"]))
        else:
            # Already archived / read: nothing was written, so nothing to
            # undo ('already' is never undone) -- but recorded, so `pending`
            # stops offering it.
            conn.execute("UPDATE handled_gmail_log SET status = 'already' WHERE id = ?", (log_id,))
        conn.commit()
        outcomes.append(o)
    return outcomes


def undo(conn: sqlite3.Connection, gmail: object, run_id: int) -> list[str]:
    """Reverse run `run_id`: unarchive what it archived, mark unread what it
    marked read -- only messages the run actually changed ('done' log rows),
    never anything else. Each undo is logged right after its own write."""
    rows = conn.execute(
        "SELECT id, message_id, action FROM handled_gmail_log "
        "WHERE run_id = ? AND status = 'done' AND undone_at IS NULL ORDER BY id",
        (run_id,),
    ).fetchall()
    if not rows:
        return [f"nothing to undo for run {run_id}"]
    notes = []
    for rid, mid, action in rows:
        inverse = ACTIONS[action]
        result = _call_one(gmail, inverse, mid)
        if isinstance(result, Exception):
            notes.append(f"error ({type(result).__name__}: {result}): {inverse} {mid}")
            continue
        conn.execute("UPDATE handled_gmail_log SET undone_at = ? WHERE id = ?", (_iso(), rid))
        conn.execute("UPDATE handled SET gmail_action = NULL, run_id = NULL WHERE message_id = ? AND run_id = ?",
                     (mid, run_id))
        conn.commit()
        notes.append(f"{inverse} {mid}" + ("" if result else " (already so; nothing to change)"))
    return notes


# -- Gmail filter suggestions (Inbox Zero pattern) -----------------------------


def _own_addresses() -> set[str]:
    try:
        from .config import settings

        user = settings().imap_user
    except Exception:  # noqa: BLE001 - a guard, never a failure
        return set()
    return {user.strip().lower()} if user and user.strip() else set()


def filter_suggestions(
    conn: sqlite3.Connection, *, now: datetime | None = None,
    threshold: int = FILTER_THRESHOLD, days: int = FILTER_DAYS,
    own: Iterable[str] | None = None,
) -> list[tuple[str, int]]:
    """Senders archived (by this tool, not undone) at least `threshold`
    times in the last `days` days: [(address, count)], most first. The
    account's own address (`own`, default LMSA_IMAP_USER) is never offered."""
    since = _iso((now or _now()) - timedelta(days=days))
    mine = {a.strip().lower() for a in own} if own is not None else _own_addresses()
    rows = conn.execute(
        "SELECT lower(sender), COUNT(DISTINCT message_id) FROM handled_gmail_log "
        "WHERE action = 'archive' AND status = 'done' AND undone_at IS NULL AND applied_at >= ? "
        "AND sender IS NOT NULL AND sender != '' GROUP BY lower(sender) HAVING COUNT(DISTINCT message_id) >= ? "
        "ORDER BY 2 DESC, 1",
        (since, threshold),
    ).fetchall()
    return [(s, n) for s, n in rows if _ADDR_RE.fullmatch(s) and s not in mine]


def filter_xml(senders: Iterable[str], *, now: datetime | None = None) -> str:
    """A Gmail filter export (Settings -> Filters and Blocked Addresses ->
    Import filters) with one "from:<sender> -> skip the inbox" filter per
    sender. Gmail filters can't be created over IMAP, so this is the whole
    write: a file the person imports themselves. Invalid addresses are
    skipped."""
    stamp = (now or _now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    for i, sender in enumerate(s for s in senders if _ADDR_RE.fullmatch(s or "")):
        entries.append(
            "  <entry>\n"
            "    <category term='filter'></category>\n"
            "    <title>Mail Filter</title>\n"
            f"    <id>tag:mail.google.com,2008:filter:lmsa{i}</id>\n"
            f"    <updated>{stamp}</updated>\n"
            "    <content></content>\n"
            f"    <apps:property name='from' value={quoteattr(sender)}/>\n"
            "    <apps:property name='shouldArchive' value='true'/>\n"
            "  </entry>\n"
        )
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<feed xmlns='http://www.w3.org/2005/Atom' xmlns:apps='http://schemas.google.com/apps/2006'>\n"
        f"  <title>{xml_escape('Mail Filters')}</title>\n"
        "  <id>tag:mail.google.com,2008:filters:lmsa</id>\n"
        f"  <updated>{stamp}</updated>\n"
        + "".join(entries)
        + "</feed>\n"
    )
