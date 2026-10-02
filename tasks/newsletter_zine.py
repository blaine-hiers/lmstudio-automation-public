"""Weekly newsletter zine: the last 7 days of newsletters, summarized and grouped by topic.

Reads mail via `lmsa.sources.gmail` (read-only). A "newsletter" is a message
from the last 7 days carrying a `List-Id` or `List-Unsubscribe` header --
the same signal a mail client uses to offer an unsubscribe button -- with
receipts excluded via Gmail's own `-category:purchases` search operator (a
documented heuristic, not a second model call). Each newsletter is summarized
once via `LMStudio.extract()` into a topic, three bullets, and a 0-5
"worth the full read" score; messages are then grouped by topic (normalized
for case/whitespace so near-duplicate topics merge) into an Afterhours "zine"
report: a masthead ticker of top headlines, one section per topic, and a
"Worth the full read" callout for anything scoring 4 or higher.

    python -m tasks.newsletter_zine [--fresh]

Summaries are remembered per message (`lmsa.memo`), so a re-run only asks
the model about newsletters it hasn't summarized yet; `--fresh` re-asks for
every one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from lmsa import Card, LMStudio, Report, Store, Tile
from lmsa.memo import Memo, add_footer, cached, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

QUERY = "newer_than:7d -category:purchases"
BODY_LIMIT = 4000
TICKER_LIMIT = 8
WORTH_READING_THRESHOLD = 4
MEMO_VERSION = 1  # bump when summarize_one()'s prompt building or result handling changes

SYSTEM_PROMPT = load_prompt("newsletter_zine/system_prompt")

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "three_bullets": {"type": "array", "items": {"type": "string"}},
        "worth_full_read": {"type": "integer"},
        "why": {"type": "string"},
    },
    "required": ["topic", "three_bullets", "worth_full_read", "why"],
    "additionalProperties": False,
}


@dataclass
class Summary:
    topic: str
    three_bullets: list[str]
    worth_full_read: int
    why: str


@dataclass
class Item:
    message: Message
    summary: Summary | None = None
    error: str | None = None


def is_newsletter(message: Message) -> bool:
    """A newsletter is a message that carries List-Id or List-Unsubscribe --
    the header pair mail clients use to offer an unsubscribe button."""
    return bool(message.list_unsubscribe or message.list_id)


def truncate(text: str, limit: int = BODY_LIMIT) -> str:
    """Bound untrusted email text before it goes to the model."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated, {omitted} more characters omitted ...]"


def normalize_topic(topic: str) -> str:
    """Case/whitespace-insensitive key so e.g. "AI News" and "ai  news" merge."""
    return " ".join(topic.split()).casefold()


def _summary(result: dict) -> Summary:
    """The model's answer as a Summary; raises on a malformed one."""
    score = int(result["worth_full_read"])
    score = max(0, min(5, score))
    return Summary(
        topic=str(result["topic"]).strip() or "Uncategorized",
        three_bullets=[str(b) for b in result["three_bullets"]],
        worth_full_read=score,
        why=str(result["why"]),
    )


def _valid(result: object) -> bool:
    try:
        _summary(result)  # type: ignore[arg-type]
    except Exception:
        return False
    return True


def summarize_one(lm: LMStudio, message: Message, memo: Memo | None = None) -> Item:
    """Summarize one message. Never raises: a failure is recorded on the
    returned Item so one bad message can't abort the whole run. With
    `memo`, a stored answer for this msg_id is reused."""
    prompt = f"Subject: {message.subject}\n\nBody:\n{truncate(message.text)}"
    try:
        result = cached(
            memo,
            message.msg_id,
            lambda: lm.extract(prompt, SCHEMA, system=SYSTEM_PROMPT, name="newsletter_zine"),
            valid=_valid,
        )
        return Item(message=message, summary=_summary(result))
    except Exception as exc:  # malformed/empty JSON, missing keys, server errors
        return Item(message=message, error=str(exc))


def group_by_topic(items: list[Item]) -> dict[str, list[Item]]:
    """Groups successfully-summarized items by normalized topic, preserving
    first-seen order for both the groups and the items within each group.
    The display title is the first-seen (un-normalized) topic spelling."""
    groups: dict[str, list[Item]] = {}
    titles: dict[str, str] = {}
    for item in items:
        if item.summary is None:
            continue
        key = normalize_topic(item.summary.topic)
        if key not in groups:
            groups[key] = []
            titles[key] = item.summary.topic
        groups[key].append(item)
    return {titles[key]: items for key, items in groups.items()}


def _card(item: Item) -> Card:
    summary = item.summary
    assert summary is not None
    body = "\n\n".join(f"- {b}" for b in summary.three_bullets)
    return Card(
        title=item.message.subject or "(no subject)",
        meta=item.message.from_name or item.message.from_addr,
        body=body,
        tags=[f"score {summary.worth_full_read}"],
        href=item.message.web_url or None,
        tone="acc",
    )


def build_report(items: list[Item]) -> Report:
    summarized = [i for i in items if i.summary is not None]
    failed = [i for i in items if i.error is not None]
    grouped = group_by_topic(summarized)
    worth_reading = sorted(
        (i for i in summarized if i.summary.worth_full_read >= WORTH_READING_THRESHOLD),
        key=lambda i: i.summary.worth_full_read,
        reverse=True,
    )
    ticker = [
        f"{i.summary.topic}: {i.message.subject}"
        for i in sorted(summarized, key=lambda i: i.summary.worth_full_read, reverse=True)[:TICKER_LIMIT]
    ]

    r = Report(
        "Newsletter zine",
        kicker="lmsa · weekly digest",
        subtitle="Newsletters from the last 7 days, summarized and grouped by topic.",
        sticker=f"{len(summarized)} newsletters" if summarized else "quiet week",
        sticker_tone="acc",
        source="Gmail IMAP",
        ticker=ticker,
    )
    r.tiles([
        Tile("Newsletters", len(summarized), tone="info"),
        Tile("Topics", len(grouped), tone="acc"),
        Tile("Worth reading", len(worth_reading), tone="violet"),
        Tile("Failed", len(failed), tone="hot" if failed else "muted"),
    ])

    if not summarized and not failed:
        r.section("This week").text("No newsletters found in the last 7 days.")
        return r

    if worth_reading:
        section = r.section("Worth the full read", tone="violet")
        for i in worth_reading:
            section.callout(
                i.summary.why,
                label=f"{i.summary.topic} · score {i.summary.worth_full_read}",
                tone="violet",
            )

    for topic, topic_items in grouped.items():
        r.section(topic).cards([_card(i) for i in topic_items])

    if failed:
        r.section("Extraction failures", tone="hot").callout(
            "\n".join(f"{i.message.subject or '(no subject)'}: {i.error}" for i in failed),
            label="skipped",
            tone="hot",
        )

    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered summaries and ask the model about every newsletter again",
    )
    args = parser.parse_args(argv)

    lm = LMStudio()
    lm.ensure_server()

    with Gmail() as gmail:
        messages = gmail.search(QUERY)
    newsletters = [m for m in messages if is_newsletter(m)]

    with Store() as store:
        last_ok = store.last_successful_run("newsletter_zine")
        version = memo_version(
            MEMO_VERSION, prompt_hash("newsletter_zine/system_prompt"), SCHEMA, BODY_LIMIT, model_id(lm)
        )
        memo = Memo(store.conn, "newsletter_zine", version, fresh=args.fresh)
        items = [summarize_one(lm, m, memo) for m in newsletters]

    with Store() as store, store.record_run("newsletter_zine") as run:
        r = build_report(items)
        add_footer(r, memo, last_ok)
        path = r.write()
        run.report_path = path
        run.summary = {
            "candidates": len(messages),
            "newsletters": len(newsletters),
            "summarized": sum(1 for i in items if i.summary is not None),
            "failed": sum(1 for i in items if i.error is not None),
            **memo.summary(),
        }
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
