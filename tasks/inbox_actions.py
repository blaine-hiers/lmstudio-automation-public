"""Inbox actions: archive / mark read in Gmail the emails you checked off.

Checking an email off (in a report opened via the dashboard, or in the
dashboard's Inbox panel) only stores it locally (`lmsa.handled`). This task
is the optional second step. Dry-run by default: it writes a report of what
it *would* archive / mark read. `--apply` does it, logging every message
under this run's id *before* the write; `--undo RUN_ID` reverses exactly
what that run did (unarchive / mark unread) and nothing else.

    python -m tasks.inbox_actions --archive-handled               # dry run
    python -m tasks.inbox_actions --archive-handled --apply
    python -m tasks.inbox_actions --mark-read-handled --apply
    python -m tasks.inbox_actions --undo 42

The only Gmail writes are `Gmail.archive` (remove \\Inbox), `Gmail.mark_read`
(set \\Seen) and their undos -- no other label or flag, no delete, no send.

Senders archived 3+ times in 30 days get a Gmail filter suggestion: the
report links `output/gmail-filters.xml` (Gmail Settings -> Filters and
Blocked Addresses -> Import filters). Gmail filters can't be created over
IMAP, so nothing is created silently: importing it is up to you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lmsa import Link, Report, Store, Tile, handled
from lmsa.config import settings

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"]}

LIMIT = 200
FILTER_FILE = "gmail-filters.xml"


def write_filters(out_dir: Path, conn) -> tuple[Path | None, list[tuple[str, int]]]:
    """Write the filter-import file for current suggestions (or remove a
    stale one when there are none). Writes only under output/."""
    suggestions = handled.filter_suggestions(conn, now=handled._now())
    path = out_dir / FILTER_FILE
    if not suggestions:
        if path.exists():
            path.unlink()
        return None, []
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(handled.filter_xml([s for s, _n in suggestions], now=handled._now()), encoding="utf-8")
    return path, suggestions


def written_count(outcomes: dict[str, list[handled.Outcome]]) -> int:
    """How many Gmail writes actually changed something (what --undo covers)."""
    return sum(1 for outs in outcomes.values() for o in outs if o.changed)


def unknown_count(outcomes: dict[str, list[handled.Outcome]]) -> int:
    """How many Gmail writes failed with an unknown outcome (maybe applied;
    not covered by --undo)."""
    return sum(1 for outs in outcomes.values() for o in outs if o.maybe_applied)


def build_report(
    plan: dict[str, list[dict]],
    outcomes: dict[str, list[handled.Outcome]],
    *,
    applied: bool,
    run_id: int | None,
    limit: int,
    total: dict[str, int],
    filters: Path | None,
    suggestions: list[tuple[str, int]],
    gave_up: dict[str, list[dict]] | None = None,
) -> Report:
    verbs = {"archive": ("Archived", "Would archive"), "mark_read": ("Marked read", "Would mark read")}
    written = written_count(outcomes)
    if not applied:
        sticker = "dry run"
    elif written and run_id is not None:
        sticker = f"undo: --undo {run_id}"
    elif unknown_count(outcomes):
        sticker = "outcome unknown -- check Gmail"
    else:
        sticker = "nothing written"
    r = Report(
        f"Inbox actions ({'applied' if applied else 'dry run'})",
        kicker="lmsa · Gmail",
        subtitle="Gmail archive / mark read for emails you checked off.",
        sticker=sticker,
        sticker_tone="hot" if written else "info",
        source="lmsa.handled + Gmail IMAP",
    )
    tiles = []
    for action, rows in plan.items():
        done = [o for o in outcomes.get(action, []) if o.changed] if applied else rows
        tiles.append(Tile(verbs[action][0 if applied else 1], len(done), tone="hot" if done else "muted"))
        failed = [o for o in outcomes.get(action, []) if o.error]
        if applied:
            already = [o for o in outcomes.get(action, []) if o.status == "already"]
            tiles.append(Tile(handled.ALREADY[action].capitalize(), len(already), tone="muted"))
            tiles.append(Tile(f"{action} failures", len(failed), tone="warn" if failed else "muted"))
    if tiles:
        r.tiles(tiles)
    if not plan:
        r.section("Nothing chosen", tone="info").text(
            "Pass --archive-handled and/or --mark-read-handled (add --apply to write)."
        )
    for action, rows in (gave_up or {}).items():
        if rows:
            r.section(f"{verbs[action][0]}: gave up", tone="warn").table(
                ["Subject", "From", "Status"],
                [[row["subject"] or "(no subject)", row["sender"], handled.GAVE_UP] for row in rows],
            )
    for action, rows in plan.items():
        if total.get(action, 0) > len(rows):
            r.section("Note", tone="warn").callout(
                f"Only the first {len(rows)} of {total[action]} handled emails were taken (--limit {limit}); "
                "run again for the rest.",
                label="limit reached", tone="warn",
            )
        title = verbs[action][0 if applied else 1]
        if not rows:
            r.section(title, tone="acc").text("Nothing pending.")
            continue
        by_id = {o.msg_id: o for o in outcomes.get(action, [])}
        body = []
        for row in rows:
            o = by_id.get(row["msg_id"])
            if not applied:
                status = "dry run"
            elif o is None:
                status = "not attempted"
            elif o.status == "done":
                status = "done"
            elif o.status == "already":
                status = f"{handled.ALREADY[action]} -- nothing written"
            else:
                status = ("outcome unknown -- check Gmail; not covered by --undo" if o.maybe_applied
                          else "not applied") + f": {o.error}"
            body.append([row["subject"] or "(no subject)", row["sender"], status])
        r.section(title, tone="hot" if applied else "acc").table(["Subject", "From", "Status"], body)
    if suggestions:
        s = r.section("Gmail filter suggestions", tone="violet")
        s.text(
            "These senders were archived 3+ times in 30 days. Gmail filters can't be created over IMAP: "
            "import the file below in Gmail (Settings -> Filters and Blocked Addresses -> Import filters) "
            "to skip the inbox for them. Nothing is created until you do."
        )
        s.table(["Sender", "Archived (30d)"], [[sender, n] for sender, n in suggestions])
        if filters is not None:
            s.table(["Filter file"], [[Link(filters.name, filters.name)]])
    return r


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-handled", action="store_true", help="Archive (remove \\Inbox) checked-off emails")
    parser.add_argument("--mark-read-handled", action="store_true", help="Mark checked-off emails read")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max emails per action")
    parser.add_argument("--apply", action="store_true", help="[writes] Do it in Gmail (default is dry-run)")
    parser.add_argument("--undo", metavar="RUN_ID", type=int, help="[writes] Undo a previous --apply by its run id")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    from lmsa.sources.gmail import Gmail

    with Store() as store:
        if args.undo is not None:
            with Gmail() as gmail:
                for note in handled.undo(store.conn, gmail, args.undo):
                    print(note)
            return

        actions = [a for a, on in (("archive", args.archive_handled), ("mark_read", args.mark_read_handled)) if on]
        with store.record_run("tasks.inbox_actions") as run:
            total = {a: len(handled.pending(store.conn, a)) for a in actions}
            plan = {a: handled.pending(store.conn, a, limit=args.limit) for a in actions}
            gave_up = {a: handled.gave_up(store.conn, a) for a in actions}
            outcomes: dict[str, list[handled.Outcome]] = {}
            if args.apply and any(plan.values()):
                with Gmail() as gmail:
                    for action, rows in plan.items():
                        outcomes[action] = handled.apply(store.conn, gmail, run.id, action, rows)
            out_dir = settings().output_dir
            filters, suggestions = write_filters(out_dir, store.conn)
            report = build_report(
                plan, outcomes, applied=args.apply, run_id=run.id, limit=args.limit,
                total=total, filters=filters, suggestions=suggestions, gave_up=gave_up,
            )
            path = report.write(out_dir=out_dir)
            run.report_path = path
            run.summary = {
                **{f"{a}_planned": len(rows) for a, rows in plan.items()},
                **{f"{a}_done": sum(1 for o in outs if o.changed) for a, outs in outcomes.items()},
                **{f"{a}_already": sum(1 for o in outs if o.status == "already") for a, outs in outcomes.items()},
                **{f"{a}_failed": sum(1 for o in outs if o.error) for a, outs in outcomes.items()},
                **{f"{a}_gave_up": len(rows) for a, rows in gave_up.items()},
                "filter_suggestions": len(suggestions),
            }
            print(path)
            unknown = unknown_count(outcomes)
            if args.apply and written_count(outcomes):
                print(f"run id for --undo: {run.id}")
            elif args.apply and not unknown:
                print("nothing was written to Gmail; nothing to undo")
            if args.apply and unknown:
                print(f"{unknown} write(s) had an unknown outcome -- check Gmail; not covered by --undo")
            for action, rows in gave_up.items():
                if rows:
                    print(f"{action}: {len(rows)} email(s) {handled.GAVE_UP}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
