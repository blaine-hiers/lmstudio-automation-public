"""Track shipments from delivery/shipping e-mails.

Searches Gmail (since the last successful run) for messages that look like
order/shipping notifications, asks the model to extract
`{carrier, tracking_number, merchant, status, eta}`, and upserts a
`shipments` table keyed by (normalised) tracking number. Later emails only
ever *advance* a shipment's status; `exception` is the one status that can
land at any point, and is cleared once a normal status's own email is newer
than the most recent exception email (see `merge_status`) -- not by
processing order, since retries and late-indexed mail mean emails aren't
always processed in date order, and a run can be interrupted or re-run.

    python -m tasks.packages
"""

from __future__ import annotations

import re
import sqlite3
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lmsa import LMStudio, Report, Store, Tile
from lmsa.prompts import load as load_prompt
from lmsa.gmail_cursor import (
    GmailCursor,
    RetrySet,
    Window,
    add_pending_note,
    add_retry_notes,
    merge_retried,
)
from lmsa.report import safe_href
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

KV_NS = "packages"
CURSOR_KEY = "since"

SEARCH_LIMIT = 200
BODY_LIMIT = 4000
# No cursor yet (first-ever run): bound the initial scan instead of pulling
# the whole mailbox.
FIRST_RUN_LOOKBACK_DAYS = 30
DELIVERED_WINDOW_DAYS = 7

SEARCH_TERMS = (
    'subject:(track OR tracking OR shipped OR shipment OR "out for delivery" '
    'OR delivered OR "has shipped" OR "on its way")'
)

# Lifecycle order. `exception` is deliberately not in here -- it isn't a
# rung on this ladder, it's a flag with its own timestamp (see merge_status).
LEVELS = {"ordered": 0, "shipped": 1, "out_for_delivery": 2, "delivered": 3}
LEVEL_STATUS = {v: k for k, v in LEVELS.items()}
STATUSES = (*LEVELS, "exception")

# Fixed allowlist of carrier tracking URL templates. `{tracking}` is
# replaced with the normalised, percent-encoded tracking number -- never a
# carrier name or URL taken verbatim from the model or an email.
CARRIER_LINKS = {
    "ups": "https://www.ups.com/track?tracknum={tracking}",
    "fedex": "https://www.fedex.com/fedextrack/?trknbr={tracking}",
    "usps": "https://tools.usps.com/go/TrackConfirmAction?tLabels={tracking}",
    "dhl": "https://www.dhl.com/en/express/tracking.html?AWB={tracking}",
    "ontrac": "https://www.ontrac.com/tracking/?number={tracking}",
    "amazon logistics": "https://track.amazon.com/tracking/{tracking}",
}

# Normalised tracking numbers are uppercase with internal whitespace
# stripped; this is the charset check that runs after that normalisation.
_TRACKING_RE = re.compile(r"^[A-Z0-9-]{6,40}$")


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "carrier": {"type": ["string", "null"]},
        "tracking_number": {"type": ["string", "null"]},
        "merchant": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": list(STATUSES)},
        "eta": {"type": ["string", "null"]},
    },
    "required": ["carrier", "tracking_number", "merchant", "status", "eta"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = load_prompt("packages/extract_system")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _event_date(msg: Message) -> datetime:
    """The email's own Date header, or "now" if it had none/unparseable.

    Status merging is date-driven (see `merge_status`), so every message
    needs *some* aware timestamp; falling back to wall-clock time for the
    rare mail with no usable Date header is the same trade
    the shared cursor makes for undated mail, not a silent correctness gap
    -- an email that genuinely has no date can't be placed in the real
    timeline any more precisely than "just seen". NOT used for the search
    cursor: that runs on INTERNALDATE (see `lmsa.gmail_cursor`).
    """
    return msg.date if msg.date is not None else datetime.now(timezone.utc)


# -- tracking number handling --------------------------------------------------


def normalize_tracking(raw: str | None) -> str | None:
    """Strip internal/edge whitespace and uppercase. `None`/blank -> None."""
    if not raw:
        return None
    stripped = re.sub(r"\s+", "", raw).strip().upper()
    return stripped or None


def valid_tracking(tracking: str) -> bool:
    """Charset/length check on an already-normalised tracking number.

    This runs *after* `normalize_tracking` and before the number is ever
    used to build a URL or a primary key -- the model's output is untrusted
    and this is the gate that keeps garbage (or an injection attempt) out
    of the store and out of any link.
    """
    return bool(_TRACKING_RE.fullmatch(tracking))


def carrier_link(carrier: str | None, tracking: str) -> str | None:
    """A tracking URL for `tracking`, only for a carrier in the fixed
    allowlist, with the tracking number percent-encoded into a fixed
    template -- never a URL or scheme taken from the model. Passed through
    `safe_href` as defense in depth even though the template is fixed.
    """
    if not carrier:
        return None
    template = CARRIER_LINKS.get(carrier.strip().lower())
    if not template:
        return None
    url = template.format(tracking=urllib.parse.quote(tracking, safe=""))
    return safe_href(url)


# -- status merge ----------------------------------------------------------------


def merge_status(
    level: int,
    normal_status_date: datetime | None,
    exception_date: datetime | None,
    incoming_status: str,
    incoming_date: datetime,
) -> tuple[int, datetime | None, datetime | None]:
    """Combine one newly-observed status + its email's own date into a
    shipment's existing state.

    `level` is -1 for a shipment with no *accepted* normal status yet.
    `normal_status_date` is the date of the email that set the current
    `level`; `exception_date` is the date of the most recent `exception`
    email seen, independent of whether it's currently "in force". Returns
    the new `(level, normal_status_date, exception_date)`.

    This merge is order-independent (commutative and idempotent): applying
    the same set of observed `(status, date)` events in any order, or
    re-applying one twice, produces the same result -- required because
    `gmail.search()` returns newest-first and a run can be re-processed
    after a failure. Two rules, both keyed on the email's own date rather
    than the order messages happen to be processed in:

    - A normal status only ever *advances* `level` (`incoming_level >=
      level`); a lower-level status is ignored outright as stale, and
      can't move `normal_status_date` either.
    - `exception` can land at any level -- `exception_date` just tracks the
      newest exception email seen. Whether the shipment is *currently*
      "in exception" is derived (`is_exception`/`displayed_status`) by
      comparing `exception_date` to `normal_status_date`: an exception only
      applies while its email is newer than the last accepted normal
      status's email. So a normal status "clears" an exception precisely
      by virtue of dating newer than it -- no separate clearing logic
      needed, and a normal status *older* than the exception (chronologically
      stale relative to it, however it was processed) doesn't clear it.
    """
    if incoming_status == "exception":
        new_exception_date = (
            incoming_date if exception_date is None else max(exception_date, incoming_date)
        )
        return level, normal_status_date, new_exception_date

    incoming_level = LEVELS[incoming_status]
    if incoming_level < level:
        return level, normal_status_date, exception_date  # stale/lower level, ignore entirely

    new_normal_date = (
        incoming_date if normal_status_date is None else max(normal_status_date, incoming_date)
    )
    return incoming_level, new_normal_date, exception_date


def is_exception(normal_status_date: datetime | None, exception_date: datetime | None) -> bool:
    if exception_date is None:
        return False
    if normal_status_date is None:
        return True
    return exception_date > normal_status_date


def displayed_status(
    level: int, normal_status_date: datetime | None, exception_date: datetime | None
) -> str:
    return "exception" if is_exception(normal_status_date, exception_date) else LEVEL_STATUS[level]


# -- store -------------------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shipments (
            tracking_number TEXT PRIMARY KEY,
            carrier TEXT,
            merchant TEXT,
            status TEXT NOT NULL,
            level INTEGER NOT NULL,
            normal_status_date TEXT,
            exception_date TEXT,
            eta TEXT,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL
        )
        """
    )
    # A per-message ledger of every email this task reached a conclusion on
    # (a shipment folded in, or not a shipment). The shared cursor
    # (lmsa.gmail_cursor) filters these out of every scan -- so a message
    # re-seen in the overlap window never costs a second extract() call --
    # and only moves past messages that are in here (or in the retry set).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS packages_applied (
            msg_id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _is_applied(conn: sqlite3.Connection, msg_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM packages_applied WHERE msg_id = ?", (msg_id,)).fetchone()
    return row is not None


def _mark_applied(conn: sqlite3.Connection, msg_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO packages_applied (msg_id, applied_at) VALUES (?, ?)",
        (msg_id, _now_iso()),
    )
    conn.commit()


def _existing_state(conn: sqlite3.Connection, tracking: str) -> tuple[int, datetime | None, datetime | None]:
    row = conn.execute(
        "SELECT level, normal_status_date, exception_date FROM shipments WHERE tracking_number = ?",
        (tracking,),
    ).fetchone()
    if row is None:
        return -1, None, None
    level, normal_raw, exc_raw = row
    return level, _parse_dt(normal_raw), _parse_dt(exc_raw)


def upsert_shipment(
    conn: sqlite3.Connection,
    *,
    tracking: str,
    carrier: str | None,
    merchant: str | None,
    status: str,
    eta: str | None,
    event_date: datetime,
) -> str:
    """Merge one observed `(status, event_date)` into the `shipments` row
    for `tracking`.

    `carrier`/`merchant`/`eta` fill gaps (a later email missing one of
    these doesn't blank out a value an earlier email supplied) but `status`
    is always recomputed via `merge_status` against whatever's already
    stored, so it can never regress regardless of processing order.
    Returns the resulting displayed status.
    """
    level, normal_status_date, exception_date = _existing_state(conn, tracking)
    new_level, new_normal_date, new_exception_date = merge_status(
        level, normal_status_date, exception_date, status, event_date
    )
    status_out = displayed_status(new_level, new_normal_date, new_exception_date)
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO shipments
            (tracking_number, carrier, merchant, status, level, normal_status_date, exception_date, eta, first_seen, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tracking_number) DO UPDATE SET
            carrier = COALESCE(excluded.carrier, shipments.carrier),
            merchant = COALESCE(excluded.merchant, shipments.merchant),
            status = excluded.status,
            level = excluded.level,
            normal_status_date = excluded.normal_status_date,
            exception_date = excluded.exception_date,
            eta = COALESCE(excluded.eta, shipments.eta),
            last_updated = excluded.last_updated
        """,
        (
            tracking,
            carrier,
            merchant,
            status_out,
            new_level,
            new_normal_date.isoformat() if new_normal_date else None,
            new_exception_date.isoformat() if new_exception_date else None,
            eta,
            now,
            now,
        ),
    )
    conn.commit()
    return status_out


def fetch_shipments(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT tracking_number, carrier, merchant, status, eta, first_seen, last_updated, normal_status_date "
        "FROM shipments ORDER BY last_updated DESC"
    ).fetchall()
    cols = [
        "tracking_number", "carrier", "merchant", "status", "eta",
        "first_seen", "last_updated", "normal_status_date",
    ]
    return [dict(zip(cols, r)) for r in rows]


# -- cursor (kv) -----------------------------------------------------


def _cursor(conn: sqlite3.Connection) -> GmailCursor:
    """This task's shared-scheme cursor (see `lmsa.gmail_cursor`), under its
    existing kv key. Legacy stored formats (bare ISO dates) still load."""
    return GmailCursor(conn, KV_NS, since_key=CURSOR_KEY)


def load_cursor(conn: sqlite3.Connection) -> datetime | None:
    return _cursor(conn).since


# -- per-message extraction ----------------------------------------------------


def _truncate(text: str, limit: int = BODY_LIMIT) -> str:
    return text[:limit]


def process_message(lm: LMStudio, conn: sqlite3.Connection, msg: Message) -> dict:
    """Extract + upsert one message. Never raises: a failure (model call,
    bad JSON, an invalid tracking number) is isolated on the returned dict
    so one bad email can't abort the run for the rest of the mailbox.
    """
    result = {
        "msg_id": msg.msg_id,
        "subject": msg.subject,
        "tracking_number": None,
        "carrier": None,
        "merchant": None,
        "status": None,
        "eta": None,
        "skipped": False,
        "error": None,
    }
    try:
        prompt = f"Subject: {msg.subject}\n\nFrom: {msg.from_addr}\n\nBody:\n{_truncate(msg.text)}"
        extracted = lm.extract(prompt, EXTRACT_SCHEMA, system=EXTRACT_SYSTEM, name="shipment")

        tracking = normalize_tracking(extracted.get("tracking_number"))
        if tracking is None:
            result["skipped"] = True
            return result
        if not valid_tracking(tracking):
            result["error"] = f"tracking number failed charset check: {extracted.get('tracking_number')!r}"
            return result

        status = extracted.get("status")
        if status not in STATUSES:
            result["error"] = f"unexpected status from model: {status!r}"
            return result

        carrier = extracted.get("carrier")
        merchant = extracted.get("merchant")
        eta = extracted.get("eta")
        status_out = upsert_shipment(
            conn,
            tracking=tracking,
            carrier=carrier,
            merchant=merchant,
            status=status,
            eta=eta,
            event_date=_event_date(msg),
        )
        result.update(
            tracking_number=tracking, carrier=carrier, merchant=merchant, status=status_out, eta=eta
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# -- report --------------------------------------------------------------------


def _fmt_status(status: str) -> str:
    return status.replace("_", " ")


def build_report(
    shipments: list[dict],
    results: list[dict],
    now: datetime,
    *,
    window: Window | None = None,
    retry: RetrySet | None = None,
) -> Path:
    counts = {s: 0 for s in STATUSES}
    for s in shipments:
        counts[s["status"]] = counts.get(s["status"], 0) + 1

    cutoff = now - timedelta(days=DELIVERED_WINDOW_DAYS)
    # "Delivered in the last 7 days" is measured against the *delivered
    # email's own date* (normal_status_date, since that's the date of the
    # email that most recently advanced the shipment's level -- exactly
    # the delivered email once status == "delivered"), not when this task
    # happened to process it. Otherwise a first run (or one that's behind)
    # would count a decades-old delivery as "recent" just because today is
    # the first time it was seen.
    delivered_recent = [
        s for s in shipments
        if s["status"] == "delivered"
        and (d := _parse_dt(s["normal_status_date"])) is not None
        and d >= cutoff
    ]
    active = [s for s in shipments if s["status"] != "delivered"]
    errors = [r for r in results if r["error"]]

    r = Report(
        "Package tracker",
        kicker="lmsa · Gmail",
        subtitle=f"{len(shipments)} tracked package(s), {len(results)} email(s) scanned this run.",
        sticker=f"{counts['exception']} exception(s)" if counts["exception"] else "on track",
        sticker_tone="hot" if counts["exception"] else "acc",
        source="Gmail IMAP",
    )
    r.tiles(
        [
            Tile("Ordered", counts["ordered"], tone="info"),
            Tile("Shipped", counts["shipped"], tone="acc"),
            Tile("Out for delivery", counts["out_for_delivery"], tone="violet"),
            Tile("Exceptions", counts["exception"], tone="hot" if counts["exception"] else "muted"),
            Tile(f"Delivered ({DELIVERED_WINDOW_DAYS}d)", len(delivered_recent), tone="acc"),
        ]
    )

    add_pending_note(r, window)

    if active:
        r.section("Active shipments").table(
            ["Tracking #", "Carrier", "Merchant", "Status", "ETA", "Link"],
            [
                [
                    s["tracking_number"],
                    s["carrier"] or "—",
                    s["merchant"] or "—",
                    _fmt_status(s["status"]),
                    s["eta"] or "—",
                    carrier_link(s["carrier"], s["tracking_number"]) or "—",
                ]
                for s in active
            ],
        )
    else:
        r.section("Active shipments").text("Nothing in transit.")

    if errors:
        r.section("Errors", tone="warn").callout(
            "\n".join(f"{e['subject']}: {e['error']}" for e in errors),
            label="extraction errors",
            tone="warn",
        )

    add_retry_notes(r, retry)
    return r.write()


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()

    with Store() as store:
        ensure_schema(store.conn)
        with store.record_run("tasks.packages") as run:
            cursor = _cursor(store.conn)
            retry = RetrySet(store.conn, KV_NS)

            def concluded(msg_id: str) -> bool:
                return _is_applied(store.conn, msg_id) or msg_id in retry.items

            with Gmail() as gmail:
                cursor.start()  # before the scan: mail arriving mid-run is picked up next run
                window = cursor.scan(
                    gmail, SEARCH_TERMS,
                    default_since=_now() - timedelta(days=FIRST_RUN_LOOKBACK_DAYS),
                    limit=SEARCH_LIMIT, skip=concluded,
                )
                fresh = cursor.fetch(gmail, window)  # the oldest SEARCH_LIMIT unprocessed, oldest first
                retried = retry.fetch(gmail)  # earlier failures, refetched by X-GM-MSGID
            messages = merge_retried(fresh, retried)

            results = []
            for m in messages:
                if _is_applied(store.conn, m.msg_id):
                    retry.succeeded(m.msg_id)
                    continue  # already concluded in a prior run -- safe no-op
                r = process_message(lm, store.conn, m)
                results.append(r)
                if r["error"]:
                    retry.failed(m.msg_id, m.subject, r["error"])
                    continue
                retry.succeeded(m.msg_id)
                _mark_applied(store.conn, m.msg_id)  # concluded (a shipment or not)

            # Never past an unprocessed message -- see lmsa.gmail_cursor.
            cursor.advance(window, done=concluded)
            retry.save()
            store.conn.commit()

            shipments = fetch_shipments(store.conn)
            path = build_report(shipments, results, _now(), window=window, retry=retry)
            run.report_path = path
            run.summary = {
                "emails_scanned": len(fresh),
                "retried": len(retried),
                "shipments_updated": sum(1 for r in results if r["tracking_number"] and not r["error"]),
                "skipped_non_shipment": sum(1 for r in results if r["skipped"]),
                "errors": sum(1 for r in results if r["error"]),
                "tracked_total": len(shipments),
                "pending": window.pending,
                "gave_up": len(retry.gave_up_now),
            }
            print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
