"""Incrementally index Gmail messages for semantic search ("ask my inbox").

    python -m tasks.inbox_index [--query "in:inbox"] [--limit 200]

Searches Gmail (read-only IMAP via `lmsa.sources.gmail`), embeds each
message's subject + sender + first `BODY_CHARS` chars of body with
`lmsa.LMStudio.embed`, and upserts the vectors into the store's
`embeddings` table (namespace `gmail`) via `lmsa.vectors`.

Per-message metadata needed later to render answer sources (subject,
from, date) is written to the store's shared `messages` table; the one
field it doesn't carry (the Gmail web link) plus the excerpt actually
embedded (so `tasks.ask` can quote it without re-fetching mail) live in
this task's own `mail_index` table, keyed by the same `msg_id`.

Incremental: a message already embedded under the *current* embed model
is skipped (its metadata is still refreshed -- cheap, and covers a
message whose labels changed). Re-run after switching `LMSA_EMBED_MODEL`
to backfill it under the new model without re-fetching mail.

Read-only over Gmail -- nothing here writes, labels, or deletes a message.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from lmsa import LMStudio, Report, Store, Tile
from lmsa.sources.gmail import Gmail, Message
from lmsa.vectors import embed_and_upsert

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "embed"}

NS = "gmail"
DEFAULT_QUERY = "in:inbox"
DEFAULT_LIMIT = 200
BODY_CHARS = 2000  # chars of message body folded into the embedded/stored text
EMBED_BATCH_SIZE = 25  # messages per embed() call; isolates a bad batch, not just a bad message

_CREATE_MAIL_INDEX = """
CREATE TABLE IF NOT EXISTS mail_index (
    msg_id TEXT PRIMARY KEY,
    web_url TEXT NOT NULL,
    excerpt TEXT NOT NULL
);
"""


@dataclass
class IndexStats:
    fetched: int = 0
    indexed: int = 0
    unchanged: int = 0
    truncated: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embed_text(msg: Message, *, body_chars: int = BODY_CHARS) -> str:
    """subject + sender + first `body_chars` of body -- what gets embedded and stored."""
    body = (msg.text or "")[:body_chars]
    return f"Subject: {msg.subject}\nFrom: {msg.from_name} <{msg.from_addr}>\n\n{body}"


def _embedded_ids(conn: sqlite3.Connection, msg_ids: list[str], model: str) -> set[str]:
    if not msg_ids:
        return set()
    placeholders = ",".join("?" * len(msg_ids))
    rows = conn.execute(
        f"SELECT item_id FROM embeddings WHERE ns=? AND model=? AND item_id IN ({placeholders})",
        (NS, model, *msg_ids),
    ).fetchall()
    return {r[0] for r in rows}


def _upsert_metadata(conn: sqlite3.Connection, msg: Message, excerpt: str) -> None:
    conn.execute(
        "INSERT INTO messages(msg_id, thread_id, date, from_addr, subject, labels_json, seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(msg_id) DO UPDATE SET thread_id=excluded.thread_id, date=excluded.date, "
        "from_addr=excluded.from_addr, subject=excluded.subject, labels_json=excluded.labels_json",
        (
            msg.msg_id,
            msg.thread_id,
            msg.date.isoformat() if msg.date else None,
            msg.from_addr,
            msg.subject,
            json.dumps(msg.labels),
            _now(),
        ),
    )
    conn.execute(
        "INSERT INTO mail_index(msg_id, web_url, excerpt) VALUES (?, ?, ?) "
        "ON CONFLICT(msg_id) DO UPDATE SET web_url=excluded.web_url, excerpt=excluded.excerpt",
        (msg.msg_id, msg.web_url, excerpt),
    )


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_messages(
    conn: sqlite3.Connection,
    lm,
    messages: list[Message],
    *,
    model: str,
) -> IndexStats:
    conn.executescript(_CREATE_MAIL_INDEX)
    stats = IndexStats(fetched=len(messages))

    ids = [m.msg_id for m in messages if m.msg_id]
    already = _embedded_ids(conn, ids, model)

    to_embed: list[tuple[str, str]] = []
    for msg in messages:
        if not msg.msg_id:
            stats.errors.append(("(unknown)", "message missing X-GM-MSGID, skipped"))
            continue
        try:
            excerpt = embed_text(msg)
            _upsert_metadata(conn, msg, excerpt)
            if msg.msg_id in already:
                stats.unchanged += 1
            else:
                to_embed.append((msg.msg_id, excerpt))
                stats.indexed += 1
        except Exception as exc:  # a single malformed message must not abort the run
            stats.errors.append((msg.msg_id, f"{type(exc).__name__}: {exc}"))
    conn.commit()

    # Embedded in batches (fewer round trips to the embed endpoint); if a
    # batch call fails, only that batch's messages are marked failed --
    # already-committed metadata and prior batches' embeddings stand, and
    # later batches still run.
    for batch in _batches(to_embed, EMBED_BATCH_SIZE):
        try:
            embed_and_upsert(conn, lm, ns=NS, model=model, items=batch)
        except Exception as exc:
            for msg_id, _text in batch:
                stats.errors.append((msg_id, f"embed failed: {type(exc).__name__}: {exc}"))
                stats.indexed -= 1

    return stats


def build_report(stats: IndexStats, query: str) -> Report:
    r = Report(
        "Inbox index",
        kicker="lmsa · ask my inbox",
        subtitle=f'Semantic index over Gmail matching "{query}".',
        sticker=f"{stats.indexed} indexed" if stats.indexed else "up to date",
        sticker_tone="acc",
        source="tasks.inbox_index",
    )
    r.tiles([
        Tile("Fetched", stats.fetched, tone="info"),
        Tile("Newly indexed", stats.indexed, tone="acc"),
        Tile("Already indexed", stats.unchanged, tone="muted"),
        Tile("Errors", len(stats.errors), tone="hot" if stats.errors else "muted"),
    ])
    if stats.truncated:
        r.section("Note", tone="warn").callout(
            "Hit the Gmail search limit before fetching every match; raise --limit "
            "or narrow --query to reach older mail on a later run.",
            label="truncated",
            tone="warn",
        )
    if stats.errors:
        section = r.section("Messages that failed to index", tone="hot")
        for msg_id, error in stats.errors:
            section.callout(error, label=msg_id, tone="hot")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Gmail search query")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()
    model = lm.cfg.embed_model

    with Gmail() as gmail:
        messages = gmail.search(args.query, limit=args.limit)
    truncated = len(messages) == args.limit

    with Store() as db:
        with db.record_run("inbox_index") as run:
            stats = index_messages(db.conn, lm, messages, model=model)
            stats.truncated = truncated
            report_path = build_report(stats, args.query).write(out_dir=lm.cfg.output_dir)
            run.summary = {
                "fetched": stats.fetched,
                "indexed": stats.indexed,
                "unchanged": stats.unchanged,
                "errors": len(stats.errors),
                "truncated": stats.truncated,
            }
            run.report_path = report_path

    print(f"{stats.indexed} new message(s) indexed, {len(stats.errors)} error(s) -> {report_path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
