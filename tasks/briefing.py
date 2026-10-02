"""Briefings: config-driven topic reports (#52).

Each `config/briefings/<name>.toml` is one report -- feeds, SearXNG queries,
watched pages and API adapters, clustered into stories, ranked against your
interests, summarized with verified quotes, and compared with the last run.

    python -m tasks.briefing --name example   # run one briefing now
    python -m tasks.briefing                  # run every enabled briefing that is due
    python -m tasks.briefing --list           # what's configured, enabled, last run

Scheduling: `config/schedule.toml` has one `[tasks.briefing]` entry. With no
`--name`, the task runs each briefing whose file says `enabled = true` and
whose last successful run is at least its `every_hours` old (default 20),
so each briefing keeps its own cadence without a schedule entry per file.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from lmsa import LMStudio, Store
from lmsa.briefing import config as bconfig
from lmsa.briefing.engine import build_report, run_briefing
from lmsa.briefing.sources import PLUGIN_ERRORS, load_plugins
from lmsa.memo import add_footer

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/briefings/example.toml"], "model": "chat", "searxng": True}

USER_AGENT = "lmsa-briefing/1.0 (personal news briefing)"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_fetcher():
    from lmsa.web import Fetcher

    return Fetcher(USER_AGENT)


def last_ok(conn, name: str) -> datetime | None:
    row = conn.execute(
        "SELECT MAX(started) FROM briefing_runs WHERE briefing=? AND status='ok'", (name,)
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        dt = datetime.fromisoformat(row[0])
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def due(cfg: bconfig.BriefingConfig, last: datetime | None, now: datetime) -> bool:
    """Enabled, and never run or last run at least `every_hours` ago (with
    a 30-minute grace, so a daily briefing doesn't slip a day when the
    dispatcher starts a little earlier than yesterday)."""
    if cfg.enabled is not True:
        return False
    return last is None or now - last >= timedelta(hours=cfg.every_hours) - timedelta(minutes=30)


def list_briefings(conn, *, overrides: dict[str, bool] | None = None) -> list[str]:
    lines = []
    for name in bconfig.names():
        try:
            cfg = bconfig.load(name, overrides=overrides)
        except bconfig.ConfigError as exc:
            lines.append(f"{name:20} INVALID  {exc}")
            continue
        state = {True: "enabled", False: "disabled", None: "new"}[cfg.enabled]
        last = last_ok(conn, name)
        when = f"{last.astimezone():%Y-%m-%d %H:%M}" if last else "never"
        lines.append(f"{name:20} {state:8} every {cfg.every_hours:g} h  last {when}  {cfg.title}")
    if not lines:
        lines.append(f"no briefings in {bconfig.BRIEFINGS_DIR}")
    return lines


def run_one(name: str, *, lm, store, run, fetcher, now: datetime, fresh: bool = False, searxng_factory=None,
           overrides: dict[str, bool] | None = None):
    """Load, run, persist and write one briefing's report. Returns (path, summary)."""
    cfg = bconfig.load(name, overrides=overrides)
    result = run_briefing(cfg, lm=lm, conn=store.conn, run_id=run.id, fetcher=fetcher, now=now,
                          searxng_factory=searxng_factory, fresh=fresh)
    for err in PLUGIN_ERRORS:
        result.notes.append(err)
    report = build_report(result)
    add_footer(report, result.memo, result.last_ok)
    path = report.write(f"briefing-{name}-{report.created:%Y%m%d-%H%M}.html")
    store.conn.execute("UPDATE briefing_runs SET report_path=? WHERE briefing=? AND run_id=?",
                       (str(path), name, run.id))
    store.conn.commit()
    return path, result.summary()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Config-driven topic briefings (config/briefings/*.toml).")
    parser.add_argument("--name", help="briefing to run now (config/briefings/NAME.toml); blank = every enabled briefing that is due")
    parser.add_argument("--list", action="store_true", help="list configured briefings and exit")
    parser.add_argument("--fresh", action="store_true", help="ignore remembered story summaries")
    args = parser.parse_args(argv)

    # Read the local on/off overrides once for this whole invocation (#95
    # review finding 2): a malformed file is printed once, at the top, and
    # says it's why nothing is due -- instead of one stderr warning per
    # briefing file loaded below.
    overrides, bad = bconfig.local_overrides_status()
    if bad:
        print(bconfig.local_overrides_warning(bad))

    if args.list:
        with Store() as store:
            print("\n".join(list_briefings(store.conn, overrides=overrides)))
        return

    now = _now()
    with Store() as store:
        if args.name:
            bconfig.load(args.name, overrides=overrides)  # fail fast on a bad name/config, before loading a model
            names = [args.name]
        else:
            names = []
            for name in bconfig.names():
                try:
                    cfg = bconfig.load(name, overrides=overrides)
                except bconfig.ConfigError as exc:
                    print(f"skipping {name}: {exc}", file=sys.stderr)
                    continue
                if due(cfg, last_ok(store.conn, name), now):
                    names.append(name)
    if not names:
        print("no briefing is enabled and due (see --list)")
        return

    lm = LMStudio()
    lm.ensure_server()
    load_plugins()
    fetcher = make_fetcher()
    failures: list[str] = []
    with Store() as store, store.record_run("briefing") as run:
        summaries: dict[str, dict] = {}
        for name in names:
            try:
                path, summary = run_one(name, lm=lm, store=store, run=run, fetcher=fetcher, now=now,
                                        fresh=args.fresh, overrides=overrides)
            except Exception as exc:  # one broken briefing must not stop the others
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                summaries[name] = {"error": str(exc)[:300]}
                continue
            summaries[name] = summary
            run.report_path = path
            print(path)
        run.summary = {"briefings": summaries, "failed": len(failures)}
        if failures:
            raise RuntimeError("briefing(s) failed: " + "; ".join(failures))


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
