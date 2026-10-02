"""Count today's Gmail messages over IMAP and write a tiny Afterhours report.

Counts only -- no message content leaves this task.

    python -m tasks.gmail_probe
"""

from __future__ import annotations

from lmsa import Report, Tile
from lmsa.sources.gmail import Gmail

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"]}


def main() -> None:
    with Gmail() as gmail:
        count = gmail.count("newer_than:1d")

    r = Report(
        "Gmail probe",
        kicker="lmsa · diagnostics",
        subtitle="Read-only IMAP probe of the last day's mail. Counts only, no content.",
        sticker=f"{count} today",
        source="Gmail IMAP",
    )
    r.tiles([Tile("Messages, last 24h", count, tone="info")])
    print(r.write())


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
