"""Follow-up radar: messages you sent that still look unanswered.

Gmail has no search operator for a thread id, so this never queries per
thread. It fetches your Sent messages (last 30 days) and the `All Mail`
headers for the same window once each, groups both by `thread_id`, and
decides "a later message from someone else" by comparing dates and
`from_addr` against your own address(es) -- `settings().imap_user` plus
every address that shows up as the sender of a Sent message (aliases).
Only threads that still look unanswered after that get an `extract()`
call, on the latest sent message in that thread (body truncated), asking
the model whether it expects a reply and, if so, what it's asking for.

Bounces and auto-replies (mailer-daemon, out-of-office) don't count as a
reply. Known limitation: a real reply that Gmail files under a different
thread (the recipient composed fresh or rewrote the subject) isn't seen,
so that thread still shows as waiting.

    python -m tasks.followup_radar [--days 3] [--fresh]

The model's answer is remembered per thread + latest sent msg_id
(`lmsa.memo`): a thread still waiting on the same message costs no model
call on the next run, while a newer message you sent in it changes the key
and is asked about afresh. `--fresh` re-asks for every thread. The prompt
has no "today" in it -- days-waiting is computed in Python each run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from lmsa import LMStudio, Report, Store, Tile, handled
from lmsa.config import settings
from lmsa.memo import Memo, add_footer, cached, has_keys, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.report import Card
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

WINDOW_DAYS = 30
BODY_LIMIT = 4000
HOT_AFTER_DAYS = 7
# Gmail search caps unless a higher limit is passed; say so in the report
# when either fetch hits its cap rather than silently under-counting.
SENT_LIMIT = 1000
ALL_MAIL_LIMIT = 5000
MEMO_VERSION = 1  # bump when classify()'s prompt building or result handling changes

SCHEMA = {
    "type": "object",
    "properties": {
        "expects_reply": {"type": "boolean"},
        "ask_summary": {"type": "string"},
    },
    "required": ["expects_reply", "ask_summary"],
    "additionalProperties": False,
}

SYSTEM = load_prompt("followup_radar/system")


def _truncate(text: str, limit: int = BODY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated, {omitted} more characters omitted ...]"


@dataclass
class Candidate:
    thread_id: str
    sent_message: Message
    recipient: str
    days_waiting: float


@dataclass
class FollowUp:
    thread_id: str
    recipient: str
    subject: str
    days_waiting: float
    web_url: str
    expects_reply: bool = False
    ask_summary: str = ""
    error: str | None = None
    msg_id: str = ""  # the sent message's X-GM-MSGID: the check-off key (#61)


def own_addresses(sent_messages: Iterable[Message], imap_user: str) -> set[str]:
    """Your own address(es), lowercased: `imap_user` plus every sender
    address seen in Sent -- every message there is by definition from you,
    so its `from_addr` is an alias even when it isn't `imap_user` itself
    (e.g. a custom "send as" address)."""
    user = imap_user.strip().lower()
    addresses = {user} if user else set()
    for message in sent_messages:
        if message.from_addr:
            addresses.add(message.from_addr.strip().lower())
    return addresses


def _group_by_thread(messages: Iterable[Message]) -> dict[str, list[Message]]:
    groups: dict[str, list[Message]] = {}
    for message in messages:
        if not message.thread_id:
            continue  # can't group what we can't attribute to a thread
        groups.setdefault(message.thread_id, []).append(message)
    return groups


def _latest_by_thread(messages: Iterable[Message]) -> dict[str, Message]:
    """The most recent message per thread. Messages missing a thread id or
    a parseable date (both required to place them in time) are skipped."""
    latest: dict[str, Message] = {}
    for message in messages:
        if not message.thread_id or message.date is None:
            continue
        current = latest.get(message.thread_id)
        if current is None or message.date > current.date:
            latest[message.thread_id] = message
    return latest


def _has_later_reply(
    thread_messages: list[Message], after: datetime, own: set[str]
) -> bool:
    """True if any message in the thread is newer than `after` (an aware
    datetime -- our own last sent message) and from someone other than us."""
    for message in thread_messages:
        if message.date is None or message.date <= after:
            continue
        if (message.from_addr or "").strip().lower() in own:
            continue
        if _is_automated(message):
            continue  # a bounce or out-of-office is not an answer
        return True
    return False


_AUTOMATED_LOCALPARTS = ("mailer-daemon", "postmaster", "noreply", "no-reply", "do-not-reply", "donotreply")
_AUTOMATED_SUBJECTS = (
    "automatic reply", "auto-reply", "autoreply", "out of office", "out of the office",
    "delivery status notification", "undeliverable", "undelivered mail", "mail delivery failed",
    "returned mail", "delivery failure",
)


def _is_automated(message: Message) -> bool:
    local = (message.from_addr or "").strip().lower().split("@")[0]
    subject = (message.subject or "").strip().lower()
    return local in _AUTOMATED_LOCALPARTS or any(subject.startswith(p) for p in _AUTOMATED_SUBJECTS)


def find_unanswered(
    sent_messages: list[Message],
    all_mail_headers: list[Message],
    own: set[str],
    now: datetime,
    min_days: float,
) -> list[Candidate]:
    """Threads whose latest sent message has no later reply from anyone
    else, and has been waiting at least `min_days`."""
    latest_sent = _latest_by_thread(sent_messages)
    all_by_thread = _group_by_thread(all_mail_headers)

    candidates = []
    for thread_id, sent_message in latest_sent.items():
        if _has_later_reply(all_by_thread.get(thread_id, []), sent_message.date, own):
            continue
        days_waiting = (now - sent_message.date).total_seconds() / 86400
        if days_waiting < min_days:
            continue
        recipient = ", ".join(sent_message.to) or sent_message.reply_to or "unknown"
        candidates.append(
            Candidate(
                thread_id=thread_id,
                sent_message=sent_message,
                recipient=recipient,
                days_waiting=days_waiting,
            )
        )
    return candidates


def memo_key(candidate: Candidate) -> str:
    """Thread id plus the msg_id of the sent message the model is shown.
    Empty (never memoised) when either identity is missing."""
    if not candidate.thread_id or not candidate.sent_message.msg_id:
        return ""
    return f"{candidate.thread_id}:{candidate.sent_message.msg_id}"


def classify(lm: LMStudio, candidate: Candidate, memo: Memo | None = None) -> FollowUp:
    """Ask the model whether the sent message expects a reply. Never
    raises: a failure is recorded on the returned FollowUp so one bad
    thread can't abort the whole run. With `memo`, a stored answer for
    `memo_key(candidate)` is reused."""
    prompt = (
        f"Subject: {candidate.sent_message.subject}\n\n"
        f"Message body:\n{_truncate(candidate.sent_message.text)}"
    )
    try:
        result = cached(
            memo,
            memo_key(candidate),
            lambda: lm.extract(prompt, SCHEMA, system=SYSTEM, name="followup"),
            valid=lambda r: has_keys(r, SCHEMA["required"]),
        )
        return FollowUp(
            thread_id=candidate.thread_id,
            recipient=candidate.recipient,
            subject=candidate.sent_message.subject,
            days_waiting=candidate.days_waiting,
            web_url=candidate.sent_message.web_url,
            msg_id=candidate.sent_message.msg_id,
            expects_reply=bool(result["expects_reply"]),
            ask_summary=str(result["ask_summary"]) if result["expects_reply"] else "",
        )
    except Exception as exc:  # malformed/empty JSON, missing keys, server errors
        return FollowUp(
            thread_id=candidate.thread_id,
            recipient=candidate.recipient,
            subject=candidate.sent_message.subject,
            days_waiting=candidate.days_waiting,
            web_url=candidate.sent_message.web_url,
            msg_id=candidate.sent_message.msg_id,
            error=str(exc),
        )


def build_report(
    waiting: list[FollowUp],
    failures: list[FollowUp],
    *,
    days: int,
    sent_truncated: bool = False,
    all_mail_truncated: bool = False,
    done: list[Candidate] | None = None,
) -> Report:
    """`done`: waiting threads already checked off (#61) -- not asked
    about, shown only behind the "N handled hidden" toggle."""
    r = Report(
        "Follow-up radar",
        kicker="lmsa · Gmail",
        subtitle=f"Sent messages from the last {WINDOW_DAYS} days, unanswered {days}+ day(s).",
        sticker=f"{len(waiting)} waiting" if waiting else "all caught up",
        sticker_tone="hot" if waiting else "acc",
        source="Gmail IMAP (Sent + All Mail headers)",
    )
    oldest = max((f.days_waiting for f in waiting), default=0.0)
    r.tiles(
        [
            Tile("Waiting", len(waiting), tone="hot" if waiting else "muted"),
            Tile(
                "Oldest wait",
                f"{oldest:.1f}d" if waiting else "-",
                tone="hot" if oldest > HOT_AFTER_DAYS else "info",
            ),
        ]
    )

    if sent_truncated or all_mail_truncated:
        which = ", ".join(
            label
            for label, hit in (("Sent", sent_truncated), ("All Mail headers", all_mail_truncated))
            if hit
        )
        r.section("Note", tone="warn").callout(
            f"{which} fetch hit its result cap; there may be more mail in the "
            f"last {WINDOW_DAYS} days than was scanned.",
            label="truncated",
            tone="warn",
        )

    if done:
        r.section("Checked off", tone="muted").handled_hidden([
            Card(
                title=c.sent_message.subject or "(no subject)",
                meta=f"{c.recipient} · waiting {c.days_waiting:.1f}d",
                tone="muted",
                href=c.sent_message.web_url or None,
                msg_id=c.sent_message.msg_id,
                handled=True,
            )
            for c in done
        ])

    if not waiting and not failures:
        return r

    if waiting:
        cards = [
            Card(
                title=f.subject or "(no subject)",
                meta=f"{f.recipient} · waiting {f.days_waiting:.1f}d",
                body=f.ask_summary,
                tone="hot" if f.days_waiting > HOT_AFTER_DAYS else "warn",
                href=f.web_url or None,
                msg_id=f.msg_id or None,
            )
            for f in sorted(waiting, key=lambda f: f.days_waiting, reverse=True)
        ]
        r.section("Waiting on a reply", tone="hot").cards(cards)

    if failures:
        section = r.section("Classification errors", tone="warn")
        for f in failures:
            section.callout(f.error, label=f"{f.recipient} · {f.subject or '(no subject)'}", tone="warn")

    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=3, help="minimum days unanswered to flag (default 3)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered answers and ask the model about every waiting thread again",
    )
    args = parser.parse_args(argv)

    lm = LMStudio()
    lm.ensure_server()
    now = datetime.now(timezone.utc)
    query = f"newer_than:{WINDOW_DAYS}d"

    with Store() as store, store.record_run("tasks.followup_radar") as run:
        last_ok = store.last_successful_run("tasks.followup_radar")
        version = memo_version(
            MEMO_VERSION, prompt_hash("followup_radar/system"), SCHEMA, BODY_LIMIT, model_id(lm)
        )
        memo = Memo(store.conn, "followup_radar", version, fresh=args.fresh)
        with Gmail() as gmail:
            sent_messages = gmail.search(query, folder=gmail.sent(), limit=SENT_LIMIT)
            all_mail_headers = gmail.search(query, limit=ALL_MAIL_LIMIT, headers_only=True)

        own = own_addresses(sent_messages, settings().imap_user)
        candidates = find_unanswered(sent_messages, all_mail_headers, own, now, float(args.days))
        candidates, done = handled.split(store.conn, candidates, key=lambda c: c.sent_message.msg_id)

        classified = [classify(lm, c, memo) for c in candidates]
        waiting = [f for f in classified if f.error is None and f.expects_reply]
        failures = [f for f in classified if f.error is not None]

        report = build_report(
            waiting,
            failures,
            days=args.days,
            sent_truncated=len(sent_messages) >= SENT_LIMIT,
            all_mail_truncated=len(all_mail_headers) >= ALL_MAIL_LIMIT,
            done=done,
        )
        handled.seen(store.conn, "followup_radar", run.id, waiting + [
            FollowUp(thread_id=c.thread_id, recipient=c.recipient, subject=c.sent_message.subject,
                     days_waiting=c.days_waiting, web_url=c.sent_message.web_url, msg_id=c.sent_message.msg_id)
            for c in done
        ])
        add_footer(report, memo, last_ok)
        path = report.write()
        run.report_path = path
        run.summary = {
            "candidates": len(candidates), "waiting": len(waiting), "errors": len(failures),
            "handled_hidden": len(done), **memo.summary()
        }
    print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
