"""Inbox weather report: a weekly climate read on the mailbox.

Headers-only IMAP scans (no bodies, no per-message model calls) over the
last WINDOW_DAYS of Sent and All Mail, turned into: daily volume
(sparkline + calendar heatmap), a weekday x hour activity heatmap (local
time), top senders, median reply time (each Sent message paired with the
most recent inbound message in the same thread before it), and the
unread backlog now vs. a snapshot from about a week ago (kept in the
store's `kv` table across runs, pruned past SNAPSHOT_RETENTION_DAYS so
it doesn't grow forever). A short model-written "forecast" paragraph is
generated from the computed numbers only -- never message content --
and is wrapped so a model failure still lets the numeric report render.

Own-address/alias detection (`own_addresses`) and automated-mail
detection (`_is_automated`) are reused from `tasks.followup_radar`
rather than duplicated here.

    python -m tasks.inbox_weather
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from lmsa import LMStudio, Report, Store, Tile
from lmsa.config import settings
from lmsa.prompts import load as load_prompt
from lmsa.sources.gmail import Gmail, Message
from tasks.followup_radar import _is_automated, own_addresses

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "model": "chat"}

WINDOW_DAYS = 30
# Gmail search caps unless a higher limit is passed; say so in the report
# when either fetch hits its cap rather than silently under-counting.
SENT_LIMIT = 1000
ALL_MAIL_LIMIT = 5000
TOP_SENDERS_LIMIT = 10
BASELINE_TARGET_DAYS = 7
# A snapshot only counts as "about a week ago" inside this window; outside
# it (a 1-day-old snapshot, or a 60-day-old one after a long gap) there's
# no meaningful "last week" to compare against.
BASELINE_MIN_DAYS = 5
BASELINE_MAX_DAYS = 10
SNAPSHOT_RETENTION_DAYS = 60

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = [f"{h:02d}" for h in range(24)]

KV_NS = "tasks.inbox_weather"

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {"forecast": {"type": "string"}},
    "required": ["forecast"],
    "additionalProperties": False,
}

FORECAST_SYSTEM = load_prompt("inbox_weather/forecast_system")


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


# -- fetch + shape -------------------------------------------------------------


def daily_counts(messages: list[Message], now: datetime, days: int) -> dict[date, int]:
    """Message count per calendar day, local time, for the `days` days
    ending today. Messages with no date are skipped (can't place them)."""
    local_today = now.astimezone().date()
    counts = {local_today - timedelta(days=i): 0 for i in range(days)}
    for m in messages:
        if m.date is None:
            continue
        d = m.date.astimezone().date()
        if d in counts:
            counts[d] += 1
    return counts


def weekday_hour_counts(messages: list[Message]) -> list[list[int]]:
    """7x24 matrix (rows Mon..Sun, cols hour 00..23) of message counts,
    local time. Messages with no date are skipped."""
    grid = [[0] * 24 for _ in range(7)]
    for m in messages:
        if m.date is None:
            continue
        local = m.date.astimezone()
        grid[local.weekday()][local.hour] += 1
    return grid


def top_senders(messages: list[Message], own: set[str], limit: int = TOP_SENDERS_LIMIT) -> list[tuple[str, int]]:
    """Top inbound senders by message count, excluding our own address(es)."""
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for m in messages:
        addr = (m.from_addr or "").strip().lower()
        if not addr or addr in own:
            continue
        counts[addr] = counts.get(addr, 0) + 1
        names.setdefault(addr, m.from_name or addr)
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [(names[addr], count) for addr, count in ranked]


def _group_by_thread(messages: list[Message]) -> dict[str, list[Message]]:
    groups: dict[str, list[Message]] = {}
    for m in messages:
        if not m.thread_id:
            continue
        groups.setdefault(m.thread_id, []).append(m)
    return groups


def reply_hours(sent_messages: list[Message], all_headers: list[Message], own: set[str]) -> list[float]:
    """For each Sent message, hours between it and the most recent inbound
    (not-us, not-automated) message in the same thread that came before
    it. A Sent message that opens a thread (no prior inbound message)
    contributes nothing -- there was nothing to reply to."""
    by_thread = _group_by_thread(all_headers)
    gaps: list[float] = []
    for sent in sent_messages:
        if sent.date is None or not sent.thread_id:
            continue
        inbound = [
            m
            for m in by_thread.get(sent.thread_id, [])
            if m.date is not None
            and m.date < sent.date
            and (m.from_addr or "").strip().lower() not in own
            and not _is_automated(m)
        ]
        if not inbound:
            continue
        most_recent = max(inbound, key=lambda m: m.date)
        gaps.append((sent.date - most_recent.date).total_seconds() / 3600)
    return gaps


def _fmt_duration(hours: float) -> str:
    return f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"


def _busiest_slot(grid: list[list[int]]) -> tuple[str, str, int] | None:
    best = None
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value <= 0:
                continue
            if best is None or value > best[2]:
                best = (WEEKDAYS[r], HOURS[c], value)
    return best


# -- unread backlog snapshot (store kv) -----------------------------------------


def load_unread_snapshots(conn) -> list[dict]:
    """Every unread-backlog snapshot recorded by prior runs, oldest first.
    A row that fails to parse (corrupt/foreign kv data) is skipped rather
    than aborting the run."""
    rows = conn.execute(
        "SELECT value_json FROM kv WHERE ns = ? AND key LIKE 'unread:%'", (KV_NS,)
    ).fetchall()
    snapshots = []
    for (value_json,) in rows:
        try:
            data = json.loads(value_json)
            captured_at = datetime.fromisoformat(data["captured_at"])
            snapshots.append({"unread": int(data["unread"]), "captured_at": captured_at})
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    snapshots.sort(key=lambda s: s["captured_at"])
    return snapshots


def closest_to_week_ago(snapshots: list[dict], now: datetime) -> dict | None:
    """The snapshot nearest `now - 7 days`, restricted to snapshots between
    BASELINE_MIN_DAYS and BASELINE_MAX_DAYS old. A snapshot outside that
    window (too recent, or too stale after a long gap between runs) isn't
    "about a week ago" and is treated the same as having no baseline at
    all, rather than being reported under a misleading "7d ago" label."""
    candidates = [
        s
        for s in snapshots
        if BASELINE_MIN_DAYS <= (now - s["captured_at"]).total_seconds() / 86400 <= BASELINE_MAX_DAYS
    ]
    if not candidates:
        return None
    target = now - timedelta(days=BASELINE_TARGET_DAYS)
    return min(candidates, key=lambda s: abs((s["captured_at"] - target).total_seconds()))


def record_unread_snapshot(conn, now: datetime, unread: int) -> None:
    """Insert this run's snapshot, then drop any older than
    SNAPSHOT_RETENTION_DAYS -- keeps the kv table from growing forever
    across months/years of runs."""
    key = f"unread:{now.isoformat()}"
    payload = json.dumps({"unread": unread, "captured_at": now.isoformat()})
    conn.execute(
        "INSERT INTO kv (ns, key, value_json, updated) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(ns, key) DO UPDATE SET value_json=excluded.value_json, updated=excluded.updated",
        (KV_NS, key, payload, now.isoformat()),
    )
    conn.commit()
    _prune_unread_snapshots(conn, now)


def _prune_unread_snapshots(conn, now: datetime) -> None:
    cutoff = now - timedelta(days=SNAPSHOT_RETENTION_DAYS)
    rows = conn.execute(
        "SELECT key, value_json FROM kv WHERE ns = ? AND key LIKE 'unread:%'", (KV_NS,)
    ).fetchall()
    stale_keys = []
    for key, value_json in rows:
        try:
            captured_at = datetime.fromisoformat(json.loads(value_json)["captured_at"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # can't tell its age -- leave it rather than guess
        if captured_at < cutoff:
            stale_keys.append(key)
    if stale_keys:
        conn.executemany("DELETE FROM kv WHERE ns = ? AND key = ?", [(KV_NS, k) for k in stale_keys])
        conn.commit()


# -- forecast --------------------------------------------------------------------


def build_forecast(lm: LMStudio, stats: "Stats") -> str:
    """Ask the model for a short forecast paragraph from computed numbers
    only -- no subject line, sender name, or message body ever goes in
    this prompt."""
    busiest = stats.busiest_slot
    prompt = (
        f"Messages in the last {WINDOW_DAYS} days: {stats.total}\n"
        f"Daily volume, oldest to newest: {stats.daily_values}\n"
        f"Busiest weekday+hour message count: {busiest[2] if busiest else 0}\n"
        f"Top sender's message count: {stats.top_sender_count}\n"
        f"Messages with a computable reply time: {len(stats.reply_gaps_hours)}\n"
        f"Median reply time (hours): {stats.median_reply_hours if stats.median_reply_hours is not None else 'unknown'}\n"
        f"Unread now: {stats.unread_now}\n"
        f"Unread ~{stats.baseline_days_ago if stats.baseline_days_ago is not None else BASELINE_TARGET_DAYS} days ago: "
        f"{stats.baseline_unread if stats.baseline_unread is not None else 'unknown'}\n"
    )
    result = lm.extract(prompt, FORECAST_SCHEMA, system=FORECAST_SYSTEM, name="forecast")
    return str(result["forecast"])


# -- report --------------------------------------------------------------------


@dataclass
class Stats:
    total: int
    daily_counts: dict[date, int]
    grid: list[list[int]]
    senders: list[tuple[str, int]]
    reply_gaps_hours: list[float]
    median_reply_hours: float | None
    unread_now: int
    baseline_unread: int | None
    baseline_captured_at: datetime | None
    baseline_days_ago: int | None

    @property
    def daily_values(self) -> list[int]:
        return [self.daily_counts[d] for d in sorted(self.daily_counts)]

    @property
    def top_sender_count(self) -> int:
        return self.senders[0][1] if self.senders else 0

    @property
    def busiest_slot(self) -> tuple[str, str, int] | None:
        return _busiest_slot(self.grid)


def build_report(
    stats: Stats,
    *,
    forecast: str | None,
    forecast_error: str | None,
    sent_truncated: bool,
    all_mail_truncated: bool,
) -> Report:
    unread_delta = None
    if stats.baseline_unread is not None:
        unread_delta = stats.unread_now - stats.baseline_unread

    r = Report(
        "Inbox weather report",
        kicker="lmsa · Gmail",
        subtitle=f"Last {WINDOW_DAYS} days, headers only ({stats.total} messages scanned).",
        sticker=f"{stats.unread_now} unread now",
        sticker_tone="hot" if stats.unread_now else "acc",
        source="Gmail IMAP (Sent + All Mail headers)",
    )

    tiles = [
        Tile("Messages", stats.total, tone="info", spark=stats.daily_values),
        Tile(
            "Median reply time",
            _fmt_duration(stats.median_reply_hours) if stats.median_reply_hours is not None else "no data",
            tone="acc" if stats.median_reply_hours is not None else "muted",
            note=f"{len(stats.reply_gaps_hours)} replied thread(s)",
        ),
    ]
    if unread_delta is None:
        tiles.append(Tile("Unread now", stats.unread_now, tone="warn", note="no baseline yet"))
    else:
        sign = "+" if unread_delta >= 0 else ""
        tiles.append(
            Tile(
                "Unread now",
                stats.unread_now,
                delta=f"{sign}{unread_delta} vs {stats.baseline_days_ago}d ago",
                tone="hot" if unread_delta > 0 else "acc",
            )
        )
    r.tiles(tiles)

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

    r.section("Daily volume", tone="acc").heatmap(stats.daily_counts)
    r.section("Activity by weekday and hour (local time)", tone="violet").grid_heatmap(
        WEEKDAYS, HOURS, stats.grid
    )

    if stats.senders:
        r.section("Top senders", tone="acc").bars(stats.senders)

    if forecast:
        r.section("Forecast", tone="info").text(forecast)
    elif forecast_error:
        r.section("Forecast", tone="warn").callout(
            f"Forecast unavailable: {forecast_error}", label="model error", tone="warn"
        )

    return r


# -- main --------------------------------------------------------------------


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    query = f"newer_than:{WINDOW_DAYS}d"

    with Gmail() as gmail:
        sent_messages = gmail.search(query, folder=gmail.sent(), limit=SENT_LIMIT, headers_only=True)
        all_headers = gmail.search(query, limit=ALL_MAIL_LIMIT, headers_only=True)
        unread_now = gmail.count("in:inbox is:unread")

    own = own_addresses(sent_messages, settings().imap_user)

    counts = daily_counts(all_headers, now, WINDOW_DAYS)
    grid = weekday_hour_counts(all_headers)
    senders = top_senders(all_headers, own)
    gaps = reply_hours(sent_messages, all_headers, own)
    median_reply = statistics.median(gaps) if gaps else None

    with Store() as db, db.record_run("tasks.inbox_weather") as run:
        previous_snapshots = load_unread_snapshots(db.conn)
        baseline = closest_to_week_ago(previous_snapshots, now)
        record_unread_snapshot(db.conn, now, unread_now)

        stats = Stats(
            total=len(all_headers),
            daily_counts=counts,
            grid=grid,
            senders=senders,
            reply_gaps_hours=gaps,
            median_reply_hours=median_reply,
            unread_now=unread_now,
            baseline_unread=baseline["unread"] if baseline else None,
            baseline_captured_at=baseline["captured_at"] if baseline else None,
            baseline_days_ago=round((now - baseline["captured_at"]).total_seconds() / 86400) if baseline else None,
        )

        try:
            forecast = build_forecast(lm, stats)
            forecast_error = None
        except Exception as exc:  # malformed/empty JSON, missing keys, server errors
            forecast = None
            forecast_error = str(exc)

        report = build_report(
            stats,
            forecast=forecast,
            forecast_error=forecast_error,
            sent_truncated=len(sent_messages) >= SENT_LIMIT,
            all_mail_truncated=len(all_headers) >= ALL_MAIL_LIMIT,
        )
        path = report.write()
        run.report_path = path
        run.summary = {
            "messages": stats.total,
            "median_reply_hours": stats.median_reply_hours,
            "unread_now": stats.unread_now,
            "unread_baseline": stats.baseline_unread,
        }
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
