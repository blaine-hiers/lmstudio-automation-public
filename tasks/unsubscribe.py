"""Unsubscribe hit list: rank senders worth unsubscribing from.

Headers-only scan of the last 90 days of received mail (Sent excluded). Per
sender: volume, share unread, whether you've ever sent *to* them, and
whether they offer a `List-Unsubscribe`. Score = volume x unread share,
zeroed for anyone you've written to -- they're not a stranger blasting you.

Report-only: this never follows, POSTs to, or otherwise acts on an
unsubscribe link. The `List-Unsubscribe` header is untrusted input straight
from the sender; it is parsed for its `<...>` entries and rendered as a
plain link via `lmsa.report.safe_href`, nothing more.

    python -m tasks.unsubscribe [--days 90] [--top 25]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from lmsa import Link, Report, Store, Tile
from lmsa.sources.gmail import Gmail, Message

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"]}

DAYS = 90
# Gmail's default search() limit (200) is well under 90 days of ordinary
# mail; raise it for this scan and say so in the report if it's still hit.
SEARCH_LIMIT = 5000
SENT_LIMIT = 5000
TOP_N = 25

_LU_ENTRY_RE = re.compile(r"<([^>]+)>")


def parse_list_unsubscribe(header: str) -> str | None:
    """Best link out of a `List-Unsubscribe` header's `<...>` entries.

    A header can hold several entries (commonly one `https:` and one
    `mailto:`); prefer the `https/http` one since it's what most mail
    clients show as "Unsubscribe". Never interpreted beyond string
    extraction -- nothing here fetches or sends anything.
    """
    if not header:
        return None
    entries = _LU_ENTRY_RE.findall(header)
    https = next((e for e in entries if e.lower().startswith(("http://", "https://"))), None)
    mailto = next((e for e in entries if e.lower().startswith("mailto:")), None)
    return https or mailto


@dataclass
class SenderStats:
    addr: str
    name: str
    count: int = 0
    unread: int = 0
    has_unsubscribe: bool = False
    unsubscribe_href: str | None = None
    ever_sent_to: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def unread_share(self) -> float:
        return self.unread / self.count if self.count else 0.0

    @property
    def score(self) -> float:
        return 0.0 if self.ever_sent_to else self.count * self.unread_share


def _normalize(addr: str) -> str:
    return addr.strip().lower()


def sent_recipients(gmail: Gmail) -> tuple[set[str], bool]:
    """Every address you've ever sent to, from a single Sent-folder scan.

    Returns `(addresses, truncated)`; `truncated` is true if the scan hit
    `SENT_LIMIT`, meaning older sent mail may be missing from the set.
    """
    sent_folder = gmail.sent()
    messages = gmail.search("in:sent", folder=sent_folder, limit=SENT_LIMIT, headers_only=True)
    recipients = {_normalize(addr) for m in messages for addr in m.to if addr}
    return recipients, len(messages) == SENT_LIMIT


def collect_stats(messages: list[Message], sent_to: set[str]) -> tuple[dict[str, SenderStats], int]:
    """Per-sender stats from a headers-only inbound scan.

    One malformed message must never abort the whole scan: a failure while
    processing a message is counted and skipped, not raised.
    """
    stats: dict[str, SenderStats] = {}
    failures = 0
    for msg in messages:
        # Everything about this message is worked out before it touches `stats`,
        # so a failure partway through can never leave a sender half-updated.
        try:
            addr = _normalize(msg.from_addr)
            if not addr:
                failures += 1
                continue
            is_unread = "\\Seen" not in msg.flags
            has_unsubscribe = bool(msg.list_unsubscribe)
            link = parse_list_unsubscribe(msg.list_unsubscribe) if has_unsubscribe else None
            ever_sent_to = addr in sent_to
        except Exception as exc:  # noqa: BLE001 -- one bad message can't sink the scan
            failures += 1
            stats.setdefault(
                "(unknown)", SenderStats(addr="(unknown)", name="(unknown)")
            ).errors.append(str(exc))
            continue

        s = stats.setdefault(addr, SenderStats(addr=addr, name=msg.from_name or addr))
        s.count += 1
        if is_unread:
            s.unread += 1
        if has_unsubscribe:
            s.has_unsubscribe = True
            if s.unsubscribe_href is None:
                s.unsubscribe_href = link
        if ever_sent_to:
            s.ever_sent_to = True
    return stats, failures


def build_report(
    stats: dict[str, SenderStats],
    *,
    days: int,
    scanned: int,
    failures: int,
    inbound_truncated: bool,
    sent_truncated: bool,
    top_n: int,
) -> Report:
    senders = [s for s in stats.values() if s.addr != "(unknown)"]
    hit_list = sorted((s for s in senders if s.score > 0), key=lambda s: s.score, reverse=True)

    r = Report(
        "Unsubscribe hit list",
        kicker=f"lmsa · last {days}d",
        subtitle="Headers-only scan, ranked by volume x unread share. "
        "Nothing here is followed or sent automatically -- links are for you to click.",
        sticker=f"{len(hit_list)} candidates" if hit_list else "inbox is clean",
        sticker_tone="hot" if hit_list else "acc",
        source="Gmail IMAP",
    )
    r.tiles([
        Tile("Messages scanned", scanned, tone="info"),
        Tile("Senders seen", len(senders), tone="muted"),
        Tile("Hit list", len(hit_list), tone="hot" if hit_list else "muted"),
        Tile("Offer unsubscribe", sum(1 for s in senders if s.has_unsubscribe), tone="acc"),
    ])

    if inbound_truncated:
        r.section("Note", tone="warn").callout(
            f"The inbound scan hit its {SEARCH_LIMIT}-message cap; there may be more mail "
            f"in the last {days}d than shown here.",
            label="truncated",
            tone="warn",
        )
    if sent_truncated:
        r.section("Note", tone="warn").callout(
            f"The Sent-folder scan hit its {SENT_LIMIT}-message cap; \"ever sent to\" may "
            "miss older recipients.",
            label="truncated",
            tone="warn",
        )
    if failures:
        r.section("Note", tone="warn").callout(
            f"{failures} message(s) could not be attributed to a sender and were skipped.",
            label="skipped",
            tone="warn",
        )

    if not hit_list:
        r.section("Hit list", tone="acc").text("No sender scored above zero in this window.")
        return r

    top = hit_list[:top_n]
    r.section("Top offenders", tone="hot").bars(
        [(s.name or s.addr, round(s.score, 1)) for s in top]
    )
    r.section("Ranked", tone="hot").table(
        ["Sender", "Volume", "Unread %", "Ever emailed", "Unsubscribe"],
        [
            [
                s.name or s.addr,
                s.count,
                round(s.unread_share * 100, 1),
                "yes" if s.ever_sent_to else "no",
                Link("unsubscribe", s.unsubscribe_href) if s.unsubscribe_href
                else ("offered" if s.has_unsubscribe else "-"),
            ]
            for s in top
        ],
    )
    if len(hit_list) > top_n:
        r.section("Note", tone="warn").callout(
            f"Showing the top {top_n} of {len(hit_list)} scored senders.",
            label="truncated",
            tone="warn",
        )
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--top", type=int, default=TOP_N)
    args = parser.parse_args()

    with Store() as store, store.record_run("tasks.unsubscribe") as run:
        with Gmail() as gmail:
            sent_to, sent_truncated = sent_recipients(gmail)
            messages = gmail.search(
                f"newer_than:{args.days}d -in:sent", limit=SEARCH_LIMIT, headers_only=True
            )

        stats, failures = collect_stats(messages, sent_to)
        inbound_truncated = len(messages) == SEARCH_LIMIT

        r = build_report(
            stats,
            days=args.days,
            scanned=len(messages),
            failures=failures,
            inbound_truncated=inbound_truncated,
            sent_truncated=sent_truncated,
            top_n=args.top,
        )
        path = r.write()
        run.report_path = path
        run.summary = {"scanned": len(messages), "senders": len(stats), "errors": failures}
    print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
