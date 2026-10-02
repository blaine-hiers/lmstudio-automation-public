"""Morning inbox brief: triage the last day's Gmail into act/reply/read/ignore.

    python -m tasks.inbox_brief [--hours 6] [--fresh]

Read-only: opens Gmail over IMAP (`lmsa.sources.gmail`, PEEK-only, nothing
ever marked read), pulls up to 150 messages from the default `newer_than:1d`
window (or `--hours` for a narrower one), and classifies each with
`lmsa.triage.classify`. Act and Reply render as cards (Gmail link, due date
in the meta line); Read as a table; Ignore as a single count. One message
that fails to classify is recorded and skipped, never aborts the run.
Classifications are remembered per message (`lmsa.memo`, shared with
`tasks.reply_drafts`), so a re-run only calls the model for new mail;
`--fresh` ignores that memory and reclassifies everything.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime

from lmsa import Card, LMStudio, Report, Store, Tile, handled
from lmsa.digest import Digest, DigestItem, RunRow
from lmsa.memo import Memo, add_footer
from lmsa.sources.gmail import Gmail, Message
from lmsa.triage import BUCKETS, MEMO_TASK, classify, open_memo

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

MESSAGE_LIMIT = 150
DEFAULT_QUERY = "newer_than:1d"


def _query(hours: int | None, *, now: float | None = None) -> str:
    """Gmail's newer_than: takes only d/m/y units, so an hour window uses
    after:<unix seconds>, which Gmail search also accepts."""
    if not hours:
        return DEFAULT_QUERY
    return f"after:{int((time.time() if now is None else now) - hours * 3600)}"


def fetch_messages(hours: int | None) -> tuple[list[Message], bool]:
    """Read-only fetch of the inbox window, newest first, capped at
    MESSAGE_LIMIT. Returns `(messages, truncated)`; `truncated` is true
    when the result hit the cap -- a sign there may be more mail in the
    window than was fetched."""
    with Gmail() as gmail:
        messages = gmail.search(_query(hours), limit=MESSAGE_LIMIT)
    return messages, len(messages) >= MESSAGE_LIMIT


@dataclass
class Triaged:
    msg: Message
    bucket: str | None
    why: str | None
    action: str | None
    due_date: str | None
    error: str | None = None


def classify_one(msg: Message, lm: LMStudio, memo: Memo | None = None) -> Triaged:
    """Classify one message. Never raises: a failure is recorded on the
    returned Triaged (`error` set, the rest null) so one bad message can't
    abort the whole run."""
    try:
        result = classify(msg, lm, memo=memo)
    except Exception as exc:  # malformed/empty JSON, missing keys, server errors
        return Triaged(msg=msg, bucket=None, why=None, action=None, due_date=None, error=str(exc))
    return Triaged(
        msg=msg,
        bucket=result["bucket"],
        why=result["why"],
        action=result["action"],
        due_date=result["due_date"],
    )


def classify_all(messages: list[Message], lm: LMStudio, memo: Memo | None = None) -> list[Triaged]:
    return [classify_one(msg, lm, memo) for msg in messages]


def summary_counts(triaged: list[Triaged]) -> dict:
    counts = {bucket: sum(1 for t in triaged if t.bucket == bucket) for bucket in BUCKETS}
    counts["failed"] = sum(1 for t in triaged if t.error is not None)
    counts["total"] = len(triaged)
    return counts


def _card(t: Triaged, *, tone: str) -> Card:
    meta = t.msg.from_name or t.msg.from_addr
    if t.due_date:
        meta = f"{meta} · due {t.due_date}"
    body = t.why or ""
    if t.action:
        body = f"{body}\n\n{t.action}" if body else t.action
    return Card(
        title=t.msg.subject or "(no subject)",
        meta=meta,
        body=body,
        tone=tone,
        # msg.web_url is built by our own code from a numeric thread id
        # (lmsa.sources.gmail._web_url), never from message content.
        href=t.msg.web_url or None,
        msg_id=t.msg.msg_id or None,
    )


def build_report(
    triaged: list[Triaged], *, query: str, truncated: bool, done: list[Message] | None = None
) -> Report:
    """`done`: messages in the window already checked off (#61) -- not
    triaged, shown only behind the "N handled hidden" toggle."""
    ok = [t for t in triaged if t.error is None]
    failed = [t for t in triaged if t.error is not None]
    by_bucket = {bucket: [t for t in ok if t.bucket == bucket] for bucket in BUCKETS}
    act, reply, read, ignore = (by_bucket[bucket] for bucket in BUCKETS)

    r = Report(
        "Morning inbox brief",
        kicker="lmsa · Gmail",
        subtitle=f"Triaged {len(triaged)} message(s) matching `{query}`.",
        sticker=f"{len(act)} to act on" if act else "inbox is quiet",
        sticker_tone="hot" if act else "acc",
        source="Gmail IMAP",
    )
    r.tiles([
        Tile("Act", len(act), tone="hot" if act else "muted"),
        Tile("Reply", len(reply), tone="warn" if reply else "muted"),
        Tile("Read", len(read), tone="info"),
        Tile("Ignore", len(ignore), tone="muted"),
        Tile("Failed", len(failed), tone="hot" if failed else "muted"),
    ])

    if truncated:
        r.section("Note", tone="warn").callout(
            f"Hit the {MESSAGE_LIMIT}-message cap; there may be more mail in "
            "this window than is shown here. Re-run with a narrower --hours.",
            label="truncated",
            tone="warn",
        )

    if act:
        r.section("Act", tone="hot").cards([_card(t, tone="hot") for t in act])
    if reply:
        r.section("Reply", tone="warn").cards([_card(t, tone="warn") for t in reply])
    if read:
        r.section("Read", tone="info").table(
            ["From", "Subject", "Why"],
            [[t.msg.from_name or t.msg.from_addr, t.msg.subject, t.why] for t in read],
        )
    if done:
        r.section("Checked off", tone="muted").handled_hidden(handled.hidden_cards(done))
    if failed:
        section = r.section("Failed to classify", tone="hot")
        for t in failed:
            section.callout(t.error, label=t.msg.subject or t.msg.from_addr or "(unknown)", tone="hot")

    return r


def _summary_dict(summary_json: str | None) -> dict:
    try:
        data = json.loads(summary_json) if summary_json else {}
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _open_cards(conn, run_id: int | None) -> list[tuple[str, str, str, str]]:
    """(msg_id, subject, sender, web_url) for the act/reply cards open as of
    this run's report or any later one, minus anything checked off since.
    No model call, no network: `handled_seen` and `handled` are stored rows
    from this and earlier runs.

    `handled_seen` is keyed by message, not (task, run): re-showing a card
    in a *later* run moves its row's `run_id` forward (`lmsa.handled.seen`'s
    `ON CONFLICT ... SET run_id=excluded.run_id`), even when that later run
    goes on to fail after `seen()` already committed. Filtering by exact
    equality would then lose the card the moment a later (possibly failed)
    run touches it, although it's still open -- exactly the run the master
    report falls back to (#104 review round 2). Filtering by `run_id >=
    run_id` instead: a later run can only ever add cards to what this one
    showed, never make one disappear."""
    rows = conn.execute(
        "SELECT s.message_id, s.subject, s.sender, s.web_url FROM handled_seen s "
        "LEFT JOIN handled h ON h.message_id = s.message_id "
        "WHERE s.source_task = 'inbox_brief' AND s.run_id >= ? AND h.message_id IS NULL",
        (run_id,),
    ).fetchall()
    return [(r[0], r[1] or "(no subject)", r[2] or "", r[3] or "") for r in rows]


def _memo_lookup(conn, msg_ids: list[str]) -> dict[str, dict]:
    """`{msg_id: {"bucket", "why", "action", "due_date"}}` from the shared
    "triage" memo (`lmsa.memo`/`lmsa.triage`) -- the model's own answer for
    each message, already computed by a past run; never a fresh call."""
    if not msg_ids:
        return {}
    placeholders = ",".join("?" * len(msg_ids))
    rows = conn.execute(
        f"SELECT item_key, value_json FROM llm_memo WHERE task = ? AND item_key IN ({placeholders})",
        (MEMO_TASK, *msg_ids),
    ).fetchall()
    out = {}
    for key, value_json in rows:
        try:
            value = json.loads(value_json)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            out[key] = value
    return out


def _parse_date(value: str | None) -> date | None:
    try:
        return datetime.fromisoformat(value).date()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_overdue(due_date: object, today: date) -> bool:
    if not isinstance(due_date, str) or not due_date:
        return False
    try:
        return date.fromisoformat(due_date) < today
    except ValueError:
        return False


_ITEM_TONE = {"act": "hot", "reply": "warn"}
_BUCKET_RANK = {"act": 0, "reply": 1}


def digest(run: RunRow, max_items: int) -> Digest:
    """Master report hook (#104): tiles for new mail and the open cards, and the
    still-open act/reply messages as items -- subject, sender, and the
    model's one-line "why" already stored in the triage memo (no new model
    call), linked to the Gmail web URL `lmsa.sources.gmail` derived from the
    message's thread id. Act sorts before reply, each by due date (soonest
    first, no date last). A callout counts overdue replies -- due_date
    earlier than this run's own date."""
    summary = _summary_dict(run.summary_json)

    with Store() as store:
        cards = _open_cards(store.conn, run.id)
        memo = _memo_lookup(store.conn, [c[0] for c in cards])

    today = _parse_date(run.started)
    overdue = 0
    rows = []
    for msg_id, subject, sender, web_url in cards:
        info = memo.get(msg_id) or {}
        bucket = info.get("bucket")
        if bucket not in _BUCKET_RANK:
            continue  # not one of the cards this hook shows (or memo missing/stale)
        due = info.get("due_date")
        if bucket == "reply" and today is not None and _is_overdue(due, today):
            overdue += 1
        rows.append((_BUCKET_RANK[bucket], due or "9999-99-99", subject, sender, web_url, info.get("why") or "",
                     _ITEM_TONE[bucket]))
    rows.sort(key=lambda r: r[:2])

    # Important / Needs reply count the open cards listed below, not the
    # run's frozen summary, so the tiles always agree with the item list
    # (a later run can add open cards; see _open_cards).
    act = sum(1 for r in rows if r[0] == _BUCKET_RANK["act"])
    reply = sum(1 for r in rows if r[0] == _BUCKET_RANK["reply"])
    tiles = [
        Tile("New", summary.get("total", 0), tone="info"),
        Tile("Important", act, tone="hot" if act else "muted"),
        Tile("Needs reply", reply, tone="warn" if reply else "muted"),
    ]

    items = [
        DigestItem(
            title=subject,
            url=web_url or None,
            blurb=f"{sender} · {why}" if why else sender,
            tone=tone,
        )
        for _rank, _due, subject, sender, web_url, why, tone in rows[:max_items]
    ]
    callouts = [f"{overdue} repl{'y is' if overdue == 1 else 'ies are'} overdue"] if overdue else []
    return Digest(tiles=tiles, items=items, callouts=callouts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hours", type=int, default=None,
        help="Look back this many hours instead of the default newer_than:1d window",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered classifications and ask the model about every message again",
    )
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()

    with Store() as db:
        last_ok = db.last_successful_run("tasks.inbox_brief")
        memo = open_memo(db.conn, lm, fresh=args.fresh)
        with db.record_run("tasks.inbox_brief") as run:
            messages, truncated = fetch_messages(args.hours)
            messages, done = handled.split(db.conn, messages)
            triaged = classify_all(messages, lm, memo)
            report = build_report(triaged, query=_query(args.hours), truncated=truncated, done=done)
            carded = [t.msg for t in triaged if t.error is None and t.bucket in ("act", "reply")]
            handled.seen(db.conn, "inbox_brief", run.id, carded + done)
            add_footer(report, memo, last_ok)
            path = report.write(out_dir=lm.cfg.output_dir)
            run.report_path = path
            run.summary = {**summary_counts(triaged), "handled_hidden": len(done), **memo.summary()}
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
