"""Send HTML email over Gmail SMTP (#101).

    result = send("Subject", "<p>html</p>", "plain text", dry_run=False, name="morning")

Rules this module keeps:
  - **Dry-run is the default** (repo rule: anything that sends is dry-run
    first). A dry run never imports/instantiates the SMTP class -- it just
    builds the MIME message and writes it to `output/*.eml`, returning that
    path.
  - **STARTTLS, never a bare login.** `smtplib.SMTP(host, port)` ->
    `ehlo()` -> `starttls(context=ssl.create_default_context())` ->
    `ehlo()` -> `login()`. If `starttls()` raises, `login()` is never
    reached -- the credential is never sent over a connection that isn't
    encrypted yet.
  - **The app password never appears** in a log line, exception text,
    return value, `.eml` file, or the `notify()` alert. Failures are
    reported by exception *type* only (`type(exc).__name__`), the same
    pattern `notify.slack` uses; anything else emitted on the failure path
    is additionally run through `tasks.dashboard_setup.redact` as
    defense-in-depth.
  - **Idempotent per (name, local date)** via the `sends` table
    (`lmsa.store._NAMED_MIGRATIONS["mailer"]`): pass `name` to have `send()`
    skip a day that's already gone out, and record the outcome of the ones
    it does send. `force=True` bypasses the skip for a deliberate resend.
    Dry-run never touches `sends`, and a prior failed row never blocks a
    retry. The (name, date) row is claimed *atomically* (`_claim`, an
    INSERT that races safely on the primary key) before the SMTP attempt,
    so two concurrent real sends for the same name can't both deliver --
    one claims it and sends, the other sees it already claimed and is
    skipped. A crash between claiming and finishing leaves the row
    `status='sending'`, which (like `'sent'`) is *not* reclaimable without
    `force=True`; that's a deliberate trade-off against ever double-sending.
  - **`send()` never raises**, on any SMTP or bookkeeping failure. Recording
    a send's outcome and firing the failure `notify()` alert happen after
    the SMTP exception has stopped being "the exception being handled" --
    each is independently wrapped in its own try/except that fails soft
    (logs the exception type and moves on), so a broken store or a broken
    import can't turn one failure into an unhandled exception whose
    traceback chains the original SMTP error (and its password-bearing
    text) onto it.
"""

from __future__ import annotations

import logging
import smtplib
import sqlite3
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Sequence

from .config import settings
from .notify import notify
from .secrets import resolve
from .store import Store

log = logging.getLogger(__name__)

# (filename, content, mime type e.g. "application/pdf")
Attachment = tuple[str, bytes, str]

# Seam for tests: replaced with a fake class so no test ever opens a socket.
_smtp_class = smtplib.SMTP


@dataclass
class SendResult:
    ok: bool
    dry_run: bool
    message_id: str | None = None
    path: Path | None = None  # the .eml a dry run wrote
    error_type: str | None = None  # type(exc).__name__ on a real-send failure
    skipped: bool = False  # already_sent() short-circuited a real send


def _today() -> str:
    return date.today().isoformat()


def _safe(text: str, secret: str | None) -> str:
    """`text` with `secret` (and its common escaped/quoted forms) scrubbed,
    reusing the same redaction `tasks.dashboard_setup` uses for the setup
    checklist. Deferred import: `lmsa` doesn't otherwise depend on `tasks`,
    and this is only needed on the failure path.

    Never raises: if the import itself fails (a broken install, a test that
    knocked the module out of `sys.modules`) this falls back to a plain
    `str.replace` of the secret instead of propagating -- redaction must
    degrade, not blow up the caller it's protecting.
    """
    try:
        from tasks.dashboard_setup import redact

        return redact(text, secret or "", min_len=1)
    except Exception:
        return text.replace(secret, "[redacted]") if secret else text


def already_sent(conn: sqlite3.Connection, name: str, when: str | None = None) -> bool:
    """True if `sends` already has a `status='sent'` row for (`name`,
    `when` or today's local date). A failed row doesn't count, so a
    previous failure never blocks a retry. This is a plain read for a
    caller deciding whether to bother building a report at all -- `send()`
    itself uses the atomic `_claim` below, not this, to decide whether to
    actually dispatch."""
    row = conn.execute(
        "SELECT 1 FROM sends WHERE name=? AND date=? AND status='sent'",
        (name, when or _today()),
    ).fetchone()
    return row is not None


def _claim(conn: sqlite3.Connection, name: str, when: str, *, force: bool) -> bool:
    """Atomically claim the (`name`, `when`) row for a real send by writing
    `status='sending'`, and report whether the claim succeeded.

    A fresh row is claimed by a plain INSERT (the primary key makes that
    atomic on its own). An existing row is only reclaimed -- via a single
    `ON CONFLICT ... DO UPDATE ... WHERE` -- when it's `status='failed'`
    (so a previous failure never blocks a retry) or when `force` is set
    (bypassing that check entirely, for a deliberate resend even of a
    `'sent'` row). There's no read-then-write gap between checking the old
    status and writing the new one: two concurrent callers racing the same
    (name, when) can't both see "claimable" and both proceed, because
    SQLite serializes the two INSERTs against the same primary key and
    only one of them can match the WHERE (or, without `force`, find no
    existing row at all).
    """
    now = datetime.now(timezone.utc).isoformat()
    where = "" if force else "WHERE sends.status = 'failed'"
    cur = conn.execute(
        f"""
        INSERT INTO sends (name, date, message_id, status, error_type, created)
        VALUES (?, ?, NULL, 'sending', NULL, ?)
        ON CONFLICT (name, date) DO UPDATE SET
            status = 'sending', message_id = NULL, error_type = NULL, created = excluded.created
        {where}
        """,
        (name, when, now),
    )
    conn.commit()
    return cur.rowcount > 0


def _finish_safe(name: str, when: str, message_id: str | None, status: str, error_type: str | None) -> None:
    """Write the final outcome of a claimed (name, when) row.

    Best-effort and never raises: for a send that just succeeded, a store
    failure here must not turn a delivered email into an exception out of
    `send()`; for one that just failed, it must not compound the failure.
    Either way the cost of losing this write is bounded -- the row is left
    at `status='sending'` (never reclaimable without `force`), so the
    worst case is one name/day needing a deliberate forced resend, not a
    silent duplicate.
    """
    try:
        with Store() as store:
            store.conn.execute(
                "UPDATE sends SET message_id=?, status=?, error_type=?, created=? WHERE name=? AND date=?",
                (message_id, status, error_type, datetime.now(timezone.utc).isoformat(), name, when),
            )
            store.conn.commit()
    except Exception as exc:
        log.warning("mailer failed to record a %s send for %r (%s)", status, name, type(exc).__name__)


def _build_message(
    subject: str,
    html: str,
    text: str,
    attachments: Sequence[Attachment],
    *,
    sender: str,
    to: str,
) -> EmailMessage:
    """`multipart/alternative` (text + html); attachments (if any) wrap that
    in `multipart/mixed`."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    # An explicit domain: without one, make_msgid() calls socket.getfqdn() on
    # every send (a possible reverse-DNS stall) and leaks the hostname.
    msg["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2] or "localhost")
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    for filename, data, mime_type in attachments:
        maintype, _, subtype = mime_type.partition("/")
        msg.add_attachment(data, maintype=maintype or "application", subtype=subtype or "octet-stream",
                            filename=filename)
    return msg


def _log_failure_safe(error_type: str, password: str | None) -> None:
    """Log the failure by exception type only. Never raises: `_safe` itself
    already falls back safely, but the log call is wrapped too, per the
    same "each step fails on its own" rule as `_notify_failure_safe`."""
    try:
        log.warning(_safe(f"mailer send failed ({error_type})", password))
    except Exception:
        log.warning("mailer send failed (%s)", error_type)  # no secret in this text either way


def _notify_failure_safe(subject: str, host: str, port: int, error_type: str, password: str | None) -> None:
    """Fire the urgent `notify()` alert for a failed send. Never raises: a
    broken toast/Slack channel (already fail-soft on its own) or a broken
    `_safe` must not turn a reported *send* failure into an unhandled
    exception."""
    try:
        notify(
            _safe(f"Email send failed: {subject}", password),
            _safe(f"{error_type} sending via {host}:{port}", password),
            level="urgent",
        )
    except Exception as exc:
        log.warning("mailer failed to send the failure notification (%s)", type(exc).__name__)


def _write_eml(msg: EmailMessage, name: str | None) -> Path:
    out_dir = settings().output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    from . import layout
    path = layout.placed(out_dir, f"mailer-{name or 'send'}-{ts}.eml")
    path.write_bytes(msg.as_bytes())
    return path


def send(
    subject: str,
    html: str,
    text: str,
    attachments: Sequence[Attachment] = (),
    to: str | None = None,
    dry_run: bool = True,
    *,
    name: str | None = None,
    force: bool = False,
) -> SendResult:
    """Build the message and, unless `dry_run` (the default), deliver it
    over Gmail SMTP with STARTTLS.

    `name` opts a caller into the `sends` idempotency table (e.g. a daily
    master-report dispatcher): a real send for a `name` already claimed
    today (sent, or another send in flight) is skipped (`SendResult.skipped`)
    unless `force=True`. Failures never raise -- they come back as a
    failed `SendResult` and fire an urgent `notify()` alert.
    """
    cfg = settings()
    sender = cfg.imap_user
    recipient = to or sender
    # Bad input (a CR/LF in the subject or recipient, a malformed attachment
    # type) or an unwritable output/ comes back as a failed result like any
    # other failure; no secret is involved yet.
    try:
        msg = _build_message(subject, html, text, attachments, sender=sender, to=recipient)
        message_id = msg["Message-ID"]
        if dry_run:
            path = _write_eml(msg, name)
            return SendResult(ok=True, dry_run=True, message_id=message_id, path=path)
    except Exception as exc:
        error_type = type(exc).__name__
        log.warning("mailer could not build the message (%s)", error_type)
        return SendResult(ok=False, dry_run=dry_run, error_type=error_type)

    when = _today()
    if name:
        try:
            with Store() as store:
                claimed = _claim(store.conn, name, when, force=force)
        except Exception as exc:
            error_type = type(exc).__name__
            log.warning("mailer failed to claim the send slot for %r (%s)", name, error_type)
            return SendResult(ok=False, dry_run=False, error_type=error_type)
        if not claimed:
            return SendResult(ok=True, dry_run=False, skipped=True)

    password = None
    smtp = None
    error_type = None
    try:
        password = resolve("imap")
        smtp = _smtp_class(cfg.smtp_host, cfg.smtp_port, timeout=30)
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(sender, password)
        smtp.send_message(msg)
    except Exception as exc:
        # Capture only the type and leave this block immediately -- do
        # nothing else while `exc` is still "the exception being handled".
        # Anything raised further down (a broken deferred import, a locked
        # store, a broken notify channel) would otherwise chain `exc` onto
        # it as __context__, and a formatted traceback of `exc` may quote
        # the password (e.g. an SMTPAuthenticationError's message). The
        # actual failure handling happens below, once we've left `except`.
        error_type = type(exc).__name__
    finally:
        if smtp is not None:
            try:
                smtp.quit()
            except Exception:
                pass

    if error_type is not None:
        if name:
            _finish_safe(name, when, None, "failed", error_type)
        _log_failure_safe(error_type, password)
        _notify_failure_safe(subject, cfg.smtp_host, cfg.smtp_port, error_type, password)
        return SendResult(ok=False, dry_run=False, error_type=error_type)

    if name:
        _finish_safe(name, when, message_id, "sent", None)
    return SendResult(ok=True, dry_run=False, message_id=message_id)
