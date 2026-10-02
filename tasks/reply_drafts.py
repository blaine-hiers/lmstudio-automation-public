"""Reply drafts: draft replies to messages the triage bucket puts in "reply".

Fetches the last day's Gmail (same window as `tasks.inbox_brief`), classifies
each message with `lmsa.triage.classify`, and for every message in the
**reply** bucket asks the chat model to draft a short reply in the owner's
voice. The draft is built as a proper `email.message.EmailMessage` (To,
Subject with "Re: " added once, In-Reply-To/References from the original
following RFC 5322 sec 3.6.4) and, only with `--apply`, appended to Gmail's
Drafts folder via `lmsa.sources.gmail.Gmail.append_draft` -- IMAP APPEND with
the `\\Draft` flag, nothing else. No SMTP anywhere: this drafts, it never
sends.

Deviation from the issue text: the issue describes a `--dry-run` flag that
skips the APPEND. This repo's CLAUDE.md rule ("reads before writes -- a task
that sends, deletes, labels, or pays anything defaults to dry-run") means
dry-run has to be the *default* instead, with an explicit `--apply` to
perform the write -- the same shape `tasks.file_sorter` already uses. So:
`--apply` is what writes; with no flag, this only reports what it would do.

Never drafts a reply to:
- your own mail. `All Mail` includes Sent and this task's own Drafts, so
  without a guard the first `--apply` run would find its own draft (or an
  ordinary Sent message) in the window and draft a reply to it -- a
  feedback loop. Skipped before classification even runs: anything carrying
  a `\\Sent`/`\\Draft`/`\\Drafts` label or flag, or From one of the owner's
  own addresses (`imap_user` plus every sender address seen in a bounded
  Sent-folder scan, same idea as `tasks.followup_radar.own_addresses` --
  reimplemented locally here since that task lives on a sibling branch not
  yet merged into this one's history, so it isn't importable; see NOTES).
- an address that looks automated/no-reply -- checked against From *and*
  every address the draft would actually go to (Reply-To, when present,
  overrides where a reply lands, so a spoofed-looking From with a
  no-reply Reply-To must still be caught).

Idempotency and a partial-write hazard: a message already drafted (recorded
in the store by its Gmail `msg_id`) is never re-drafted. The store records
intent *before* the APPEND (`status='appending'`), then flips to
`'drafted'` only after a *successful* append -- so if the append itself
fails, or it succeeds but the follow-up DB write fails, the row is left at
`'appending'`. A later run treats `'appending'` as "possibly already
drafted": it is never auto-retried (that would risk a duplicate draft for
an append that actually went through), just reported as
`possibly_drafted` so a person can check Gmail Drafts by hand.

Reply-To can carry more than one address; this drafts to all of them
(joined into `To`) rather than silently picking the first. When any
Reply-To address's domain differs from From's, the report flags that
card -- a classic BEC/phishing pattern (a convincing From, replies quietly
redirected) -- without refusing to draft; the person still decides.

    python -m tasks.reply_drafts               # dry run, default newer_than:1d window
    python -m tasks.reply_drafts --hours 6      # narrower window
    python -m tasks.reply_drafts --apply        # actually append the drafts
    python -m tasks.reply_drafts --fresh        # reclassify, ignoring remembered results

Classifications come from the "triage" memo shared with `tasks.inbox_brief`
(`lmsa.triage.open_memo`): a message either task already classified costs no
model call here. Drafting itself is never memoised.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.policy import default as _default_policy

from lmsa import Card, LMStudio, Report, Store, Tile, handled
from lmsa.memo import Memo, add_footer
from lmsa.prompts import load as load_prompt
from imapclient.exceptions import IMAPClientAbortError, IMAPClientError

from lmsa.sources.gmail import Gmail, Message
from lmsa.triage import classify, open_memo

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

MESSAGE_LIMIT = 150
DEFAULT_QUERY = "newer_than:1d"
SENT_ALIAS_LIMIT = 1000  # headers-only scan, just to discover "send as" aliases
MAX_RECIPIENTS = 5

INPUT_LIMIT = 4000  # characters of the original body sent to the model
REPLY_MAX_CHARS = 2000  # characters kept of the model's draft (both body and report)
REPLY_MAX_TOKENS = 600  # bounds generation up front; REPLY_MAX_CHARS is the backstop
ASK_PREVIEW_CHARS = 400  # how much of the original is shown in the report card
REFERENCES_MAX_IDS = 20  # a very long thread's References chain is trimmed to this many ids

# email.policy.default folds a header onto a continuation line once it passes
# the (default 78-char) max_line_length, and re-parsing that fold is where
# Python's email package disagrees with itself: CPython >=3.12 strips the
# fold's leading space back off, 3.11 doesn't (verified against a real
# References header long enough to fold). A few ids already exceed 78
# chars, so raise the limit to RFC 5322's actual hard cap (998 octets, sec
# 2.1.1): ordinary threads then never fold, and both interpreters emit and
# read back identical bytes. A full REFERENCES_MAX_IDS chain of very long
# ids can still pass 998 and fold, as RFC 5322 requires. That fold is valid
# on the wire and Gmail unfolds it correctly, since the draft goes straight
# to IMAP APPEND and nothing here re-parses it. Only a 3.11 re-parse of
# such a header would keep the leading space.
_EML_POLICY = _default_policy.clone(max_line_length=998)

# Local-part heuristics for addresses that shouldn't get a reply drafted at
# all (bulk/automated mail, not a person who'll read a reply).
_AUTOMATED_RE = re.compile(
    r"(no[-._]?reply|do[-._]?not[-._]?reply|mailer-daemon|postmaster|"
    r"notifications?@|automated|bounce)",
    re.IGNORECASE,
)

_RE_PREFIX_RE = re.compile(r"^\s*re\s*:\s*", re.IGNORECASE)
_MSGID_RE = re.compile(r"<[^<>\s]+>")

_OWN_MAIL_MARKERS = {"\\Sent", "\\Draft", "\\Drafts"}

REPLY_SYSTEM_PROMPT = load_prompt("reply_drafts/reply_system_prompt")


def _query(hours: int | None, *, now: float | None = None) -> str:
    """Same windowing as `tasks.inbox_brief._query`: newer_than: has no hour
    unit, so an hour window uses after:<unix seconds> instead."""
    if not hours:
        return DEFAULT_QUERY
    return f"after:{int((time.time() if now is None else now) - hours * 3600)}"


def is_automated_sender(addr: str) -> bool:
    """True for an address that looks like bulk/automated mail, not a person
    -- a reply drafted at it would never be read."""
    return bool(_AUTOMATED_RE.search(addr or ""))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated, {omitted} more characters omitted ...]"


def reply_subject(subject: str) -> str:
    """"Re: " once, never doubled, matched case-insensitively like mail clients do."""
    subject = (subject or "").strip()
    if _RE_PREFIX_RE.match(subject):
        return subject
    return f"Re: {subject}" if subject else "Re:"


# -- own mail / loop guard ------------------------------------------------------


def own_addresses(sent_messages: list[Message], imap_user: str) -> set[str]:
    """Your own address(es), lowercased: `imap_user` plus every sender
    address seen in Sent -- every message there is by definition from you,
    so its `from_addr` is an alias even when it isn't `imap_user` itself
    (e.g. a "send as" address)."""
    user = (imap_user or "").strip().lower()
    addresses = {user} if user else set()
    for m in sent_messages:
        if m.from_addr:
            addresses.add(m.from_addr.strip().lower())
    return addresses


def is_own_mail(msg: Message, own: set[str]) -> bool:
    """True when this message is the owner's own mail: Sent or a Draft
    (both show up in an `All Mail` scan), or simply From one of the
    owner's own addresses. None of these should ever get a reply drafted
    at them -- Sent/From-self is a feedback loop, and Drafts isn't inbox
    mail to begin with."""
    markers = {*(msg.labels or ()), *(msg.flags or ())}
    if markers & _OWN_MAIL_MARKERS:
        return True
    return bool(msg.from_addr) and msg.from_addr.strip().lower() in own


# -- recipients -------------------------------------------------------------------


def resolved_recipients(msg: Message) -> list[str]:
    """Who a reply actually goes to: every Reply-To address when the
    original has any (a header can legally carry more than one -- this
    drafts to all of them rather than silently picking just the first),
    else From."""
    recipients = [a for a in msg.reply_to_all if a]
    if recipients:
        return recipients
    return [msg.from_addr] if msg.from_addr else []


def _domain(addr: str) -> str:
    return addr.rsplit("@", 1)[-1].strip().lower() if addr and "@" in addr else ""


def reply_to_mismatches_sender(msg: Message) -> bool:
    """True when any Reply-To address's domain differs from From's --  a
    classic BEC/phishing pattern (a convincing From, replies quietly
    redirected elsewhere). Only flags it in the report; the draft is still
    built (and, with --apply, still queued) so the person can decide --
    silently refusing to draft would just hide the signal."""
    from_domain = _domain(msg.from_addr)
    if not from_domain:
        return False
    return any(_domain(r) != from_domain for r in msg.reply_to_all if r)


# -- store (idempotency) ------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reply_drafts (
            msg_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def draft_status(conn: sqlite3.Connection, msg_id: str) -> str | None:
    """`'drafted'`, `'appending'` (write started but never confirmed --
    see module docstring), or `None` (never attempted / no msg_id to key
    on)."""
    if not msg_id:
        return None
    row = conn.execute("SELECT status FROM reply_drafts WHERE msg_id = ?", (msg_id,)).fetchone()
    return row[0] if row else None


def _set_status(conn: sqlite3.Connection, msg_id: str, status: str, *, now: datetime) -> None:
    if not msg_id:
        return
    conn.execute(
        "INSERT INTO reply_drafts (msg_id, status, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(msg_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
        (msg_id, status, now.isoformat()),
    )
    conn.commit()


def mark_appending(conn: sqlite3.Connection, msg_id: str, *, now: datetime) -> None:
    """Record intent BEFORE the APPEND. If the process dies between this
    and `mark_drafted` -- the append itself fails, or it succeeds but this
    follow-up write fails -- the row is left at 'appending' rather than
    missing, so a later run treats it as possibly already drafted instead
    of silently re-appending (see module docstring)."""
    _set_status(conn, msg_id, "appending", now=now)


def mark_drafted(conn: sqlite3.Connection, msg_id: str, *, now: datetime) -> None:
    _set_status(conn, msg_id, "drafted", now=now)


def clear_status(conn: sqlite3.Connection, msg_id: str) -> None:
    """Forget a message's draft state: used when an APPEND provably failed,
    and by `--retry <msg_id>` after the person checked Drafts by hand."""
    conn.execute("DELETE FROM reply_drafts WHERE msg_id = ?", (msg_id,))
    conn.commit()


# -- References/In-Reply-To (RFC 5322 sec 3.6.4) -------------------------------


def _normalize_msgid(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if not token.startswith("<"):
        token = f"<{token}"
    if not token.endswith(">"):
        token = f"{token}>"
    return token


def _extract_ids(value: str) -> list[str]:
    if not value:
        return []
    found = _MSGID_RE.findall(value)
    if found:
        return found
    # No <...>-wrapped tokens matched (a malformed header) -- fall back to
    # whitespace-split tokens, each normalised to <...>.
    return [_normalize_msgid(t) for t in value.split() if t.strip()]


def build_references(msg: Message) -> str:
    """RFC 5322 sec 3.6.4: a reply's References is the parent's References
    if it has any, else the parent's In-Reply-To, with the parent's own
    Message-ID appended at the end. IDs are normalised to `<...>` and
    deduplicated in order; a very long chain keeps the first id (the
    thread root, which is what most clients group by) and the most recent
    `REFERENCES_MAX_IDS - 1` -- unbounded growth would eventually make the
    header itself absurd."""
    base = msg.references or msg.in_reply_to
    ids = [_normalize_msgid(i) for i in _extract_ids(base)]
    if msg.message_id:
        mid = _normalize_msgid(msg.message_id)
        if not ids or ids[-1] != mid:
            ids.append(mid)

    seen: set[str] = set()
    deduped = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            deduped.append(i)

    if len(deduped) > REFERENCES_MAX_IDS:
        deduped = [deduped[0], *deduped[-(REFERENCES_MAX_IDS - 1):]]
    return " ".join(deduped)


# -- draft building -------------------------------------------------------------


def draft_reply_text(lm: LMStudio, msg: Message) -> str:
    prompt = (
        f"From: {msg.from_name} <{msg.from_addr}>\n"
        f"Subject: {msg.subject}\n\n"
        f"Body:\n{_truncate(msg.text, INPUT_LIMIT)}"
    )
    text = lm.chat(prompt, system=REPLY_SYSTEM_PROMPT, max_tokens=REPLY_MAX_TOKENS)
    return _truncate(text.strip(), REPLY_MAX_CHARS)


def build_eml(msg: Message, *, from_addr: str, to: str, subject: str, body: str) -> bytes:
    """A proper RFC 5322 message. `to` may be several comma-joined
    addresses (see `resolved_recipients`). Threading headers are set only
    when the original actually had a Message-ID to thread from
    (headers_only fetches, or very old mail, may not)."""
    email_msg = EmailMessage(policy=_EML_POLICY)
    email_msg["From"] = from_addr
    email_msg["To"] = to
    email_msg["Subject"] = subject
    if msg.message_id:
        email_msg["In-Reply-To"] = _normalize_msgid(msg.message_id)
    references = build_references(msg)
    if references:
        email_msg["References"] = references
    email_msg.set_content(body)
    return email_msg.as_bytes()


@dataclass
class Draft:
    msg: Message
    to: str = ""
    subject: str = ""
    body: str | None = None
    # "drafted" (appended), "dry_run" (would append), "skipped_automated",
    # "already_drafted", "possibly_drafted" (see module docstring), or "error"
    status: str = "error"
    error: str | None = None
    reply_to_mismatch: bool = False


def draft_one(gmail: Gmail, conn: sqlite3.Connection, lm: LMStudio, msg: Message, *, apply: bool) -> Draft:
    """Draft (and, if `apply`, append) a reply for one reply-bucket message.

    Never raises: every failure mode (bad address, model call, IMAP append)
    is caught and recorded on the returned Draft so one message's problem
    can't abort the rest of the run.
    """
    recipients = resolved_recipients(msg)
    mismatch = reply_to_mismatches_sender(msg)

    # Checked against From *and* every actual recipient: Reply-To is what
    # decides where the reply lands, so a spoofed-looking From with an
    # automated Reply-To must still be caught.
    if any(is_automated_sender(a) for a in (msg.from_addr, *recipients) if a):
        return Draft(msg=msg, status="skipped_automated", reply_to_mismatch=mismatch)

    status_in_store = draft_status(conn, msg.msg_id)
    if status_in_store == "drafted":
        return Draft(msg=msg, status="already_drafted", reply_to_mismatch=mismatch)
    if status_in_store == "appending":
        return Draft(msg=msg, status="possibly_drafted", reply_to_mismatch=mismatch)

    try:
        if not recipients:
            raise ValueError("message has neither Reply-To nor From to draft a reply to")
        if len(recipients) > MAX_RECIPIENTS:
            # The sender controls Reply-To; don't let one mail fan a draft out widely.
            raise ValueError(f"Reply-To lists {len(recipients)} addresses (limit {MAX_RECIPIENTS}); not drafting")
        to = ", ".join(recipients)
        subject = reply_subject(msg.subject)
        body = draft_reply_text(lm, msg)
        message_bytes = build_eml(msg, from_addr=gmail.cfg.imap_user, to=to, subject=subject, body=body)

        if apply:
            # Resolve Drafts first: a failure here provably wrote nothing, so it
            # must not leave an 'appending' row behind.
            folder = gmail.drafts_folder()
            mark_appending(conn, msg.msg_id, now=datetime.now(timezone.utc))
            try:
                gmail.append_draft(message_bytes, folder)
            except IMAPClientAbortError:
                # Connection dropped mid-APPEND: the server may have stored
                # the draft, so keep the 'appending' row ('possibly drafted',
                # never auto-retried) rather than risk a duplicate.
                raise
            except IMAPClientError:
                # The server answered NO/BAD: definitely not written, so no
                # 'possibly drafted' ambiguity -- clear the intent row.
                clear_status(conn, msg.msg_id)
                raise
            mark_drafted(conn, msg.msg_id, now=datetime.now(timezone.utc))
            status = "drafted"
        else:
            status = "dry_run"
        return Draft(msg=msg, to=to, subject=subject, body=body, status=status, reply_to_mismatch=mismatch)
    except Exception as exc:  # noqa: BLE001 -- per-item isolation, see docstring
        return Draft(msg=msg, status="error", error=f"{type(exc).__name__}: {exc}", reply_to_mismatch=mismatch)


def process_reply_bucket(
    gmail: Gmail,
    conn: sqlite3.Connection,
    lm: LMStudio,
    messages: list[Message],
    own: set[str],
    *,
    apply: bool,
    memo: Memo | None = None,
) -> tuple[list[tuple[Message, str | None, str | None]], list[Draft], int]:
    """Filter out the owner's own mail (never classified -- no point
    spending a model call on something that can never be drafted),
    classify what's left (isolating a failure per-message, like
    `tasks.inbox_brief.classify_all`), then draft only the reply bucket.

    Returns `(classified, drafts, skipped_own)`: `classified` is
    `(msg, bucket, error)` for every non-own message fetched, `drafts` is
    one `Draft` per reply-bucket message, and `skipped_own` is the count
    excluded as the owner's own mail.
    """
    candidates = [m for m in messages if not is_own_mail(m, own)]
    skipped_own = len(messages) - len(candidates)

    classified: list[tuple[Message, str | None, str | None]] = []
    for msg in candidates:
        try:
            bucket = classify(msg, lm, memo=memo)["bucket"]
            classified.append((msg, bucket, None))
        except Exception as exc:  # noqa: BLE001 -- one bad classification can't sink the run
            classified.append((msg, None, str(exc)))

    drafts = [
        draft_one(gmail, conn, lm, msg, apply=apply)
        for msg, bucket, error in classified
        if bucket == "reply"
    ]
    return classified, drafts, skipped_own


def summary_counts(
    classified: list[tuple[Message, str | None, str | None]], drafts: list[Draft], skipped_own: int
) -> dict:
    return {
        "messages": len(classified) + skipped_own,
        "skipped_own_mail": skipped_own,
        "reply_bucket": len(drafts),
        "drafted": sum(1 for d in drafts if d.status == "drafted"),
        "dry_run": sum(1 for d in drafts if d.status == "dry_run"),
        "already_drafted": sum(1 for d in drafts if d.status == "already_drafted"),
        "possibly_drafted": sum(1 for d in drafts if d.status == "possibly_drafted"),
        "skipped_automated": sum(1 for d in drafts if d.status == "skipped_automated"),
        "errors": sum(1 for d in drafts if d.status == "error")
        + sum(1 for _msg, _bucket, error in classified if error),
    }


# -- report ---------------------------------------------------------------------

_TONE = {
    "drafted": "hot",
    "dry_run": "warn",
    "already_drafted": "muted",
    "possibly_drafted": "warn",
    "skipped_automated": "muted",
    "error": "hot",
}

_MISMATCH_TAG = "reply-to differs from sender"
_MULTI_TAG = "multiple recipients"


def _card(d: Draft) -> Card:
    multi = len(resolved_recipients(d.msg)) > 1
    meta = d.msg.from_name or d.msg.from_addr
    if d.to:
        meta = f"{meta} -> {d.to}"
    if d.status == "error":
        body = f"Ask: {_truncate(d.msg.text, ASK_PREVIEW_CHARS)}\n\nError: {d.error}"
    elif d.status == "skipped_automated":
        body = "Skipped -- looks like a no-reply/automated sender."
    elif d.status == "already_drafted":
        body = "Skipped -- a draft for this message already exists."
    elif d.status == "possibly_drafted":
        body = (
            "Needs a manual check -- a previous run may already have appended a draft "
            "for this message but couldn't confirm it went through. Not re-attempted "
            "automatically; check Gmail Drafts."
        )
    else:
        body = f"Ask: {_truncate(d.msg.text, ASK_PREVIEW_CHARS)}\n\nDraft reply:\n{d.body}"
    return Card(
        title=d.subject or d.msg.subject or "(no subject)",
        meta=meta,
        body=body,
        tags=tuple(t for t, on in ((_MISMATCH_TAG, d.reply_to_mismatch), (_MULTI_TAG, multi)) if on),
        # A reply-to mismatch or a many-recipient Reply-To is worth surfacing
        # loudly, regardless of what else is going on with this draft.
        tone="hot" if d.reply_to_mismatch or multi else _TONE[d.status],
        href=d.msg.web_url or None,
        msg_id=d.msg.msg_id or None,
    )


def build_report(
    classified: list[tuple[Message, str | None, str | None]],
    drafts: list[Draft],
    *,
    query: str,
    truncated: bool,
    apply: bool,
    skipped_own: int = 0,
    done: list[Message] | None = None,
) -> Report:
    """`done`: messages already checked off (#61) -- never drafted, shown
    only behind the "N handled hidden" toggle."""
    classify_failed = [(msg, error) for msg, _bucket, error in classified if error]
    drafted = [d for d in drafts if d.status == "drafted"]
    dry_run = [d for d in drafts if d.status == "dry_run"]
    already = [d for d in drafts if d.status == "already_drafted"]
    possibly = [d for d in drafts if d.status == "possibly_drafted"]
    skipped = [d for d in drafts if d.status == "skipped_automated"]
    errored = [d for d in drafts if d.status == "error"]
    written = drafted if apply else dry_run

    r = Report(
        "Reply drafts",
        kicker="lmsa · Gmail",
        subtitle=f"Triaged {len(classified)} message(s) matching `{query}`; "
        f"{len(drafts)} in the reply bucket. Mode: {'applied' if apply else 'dry-run'}.",
        sticker=f"{len(written)} drafted" if apply else f"{len(written)} would draft",
        sticker_tone="hot" if written else "acc",
        source="Gmail IMAP",
    )
    r.tiles([
        Tile("Reply bucket", len(drafts), tone="info"),
        Tile("Drafted" if apply else "Would draft", len(written), tone="hot" if written else "muted"),
        Tile("Already drafted", len(already), tone="muted"),
        Tile("Needs manual check", len(possibly), tone="warn" if possibly else "muted"),
        Tile("Skipped (automated)", len(skipped), tone="muted"),
        Tile("Errors", len(errored) + len(classify_failed), tone="hot" if (errored or classify_failed) else "muted"),
    ])

    if not apply:
        r.section("Note", tone="warn").callout(
            "Dry-run: no drafts were written to Gmail. Re-run with --apply to append them.",
            label="dry-run",
            tone="warn",
        )
    if truncated:
        r.section("Note", tone="warn").callout(
            f"Hit the {MESSAGE_LIMIT}-message cap; there may be more mail in this window "
            "than is shown here. Re-run with a narrower --hours.",
            label="truncated",
            tone="warn",
        )
    if skipped_own:
        r.section("Note", tone="warn").callout(
            f"{skipped_own} message(s) skipped as your own mail (Sent/Drafts, or from one "
            "of your own addresses) -- never a candidate for a reply draft.",
            label="own mail",
            tone="warn",
        )

    if drafts:
        r.section("Reply drafts", tone="hot").cards([_card(d) for d in drafts])
    else:
        r.section("Reply drafts", tone="acc").text("No messages in the reply bucket this run.")

    if done:
        r.section("Checked off", tone="muted").handled_hidden(handled.hidden_cards(done))

    if classify_failed:
        section = r.section("Failed to classify", tone="hot")
        for msg, error in classify_failed:
            section.callout(error, label=msg.subject or msg.from_addr or "(unknown)", tone="hot")

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hours", type=int, default=None,
        help="Look back this many hours instead of the default newer_than:1d window",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually append drafts to Gmail Drafts. Default is dry-run (report only).",
    )
    parser.add_argument(
        "--retry", metavar="MSG_ID",
        help="Clear a 'possibly drafted' message (after checking Gmail Drafts) so the next run drafts it again.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered classifications and ask the model about every message again",
    )
    args = parser.parse_args()

    if args.retry:
        with Store() as db:
            ensure_schema(db.conn)
            clear_status(db.conn, args.retry)
        print(f"cleared draft state for {args.retry}")
        return

    lm = LMStudio()
    lm.ensure_server()
    query = _query(args.hours)

    with Store() as db, db.record_run("tasks.reply_drafts") as run:
        last_ok = db.last_successful_run("tasks.reply_drafts")
        memo = open_memo(db.conn, lm, fresh=args.fresh)
        ensure_schema(db.conn)
        with Gmail() as gmail:
            sent_folder = gmail.sent()
            sent_messages = gmail.search("in:sent", folder=sent_folder, limit=SENT_ALIAS_LIMIT, headers_only=True)
            own = own_addresses(sent_messages, gmail.cfg.imap_user)

            messages = gmail.search(query, limit=MESSAGE_LIMIT)
            truncated = len(messages) >= MESSAGE_LIMIT
            messages, done = handled.split(db.conn, messages)
            classified, drafts, skipped_own = process_reply_bucket(
                gmail, db.conn, lm, messages, own, apply=args.apply, memo=memo
            )

        report = build_report(
            classified, drafts, query=query, truncated=truncated, apply=args.apply, skipped_own=skipped_own,
            done=done,
        )
        handled.seen(db.conn, "reply_drafts", run.id, [d.msg for d in drafts] + done)
        add_footer(report, memo, last_ok)
        path = report.write(out_dir=lm.cfg.output_dir)
        run.report_path = path
        run.summary = {**summary_counts(classified, drafts, skipped_own), "handled_hidden": len(done), **memo.summary()}
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
