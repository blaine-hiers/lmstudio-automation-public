"""The error feed behind the dashboard's Errors view, plus log housekeeping.

Three sources are merged into one newest-first list:
  - `dispatch_runs` rows with outcome error / skipped (scheduled runs), with
    the log the dispatcher kept (`log_path`);
  - the dashboard's own finished runs that exited non-zero or were stopped
    (handed in as `LocalRun`s -- they only live in the dashboard's memory);
  - `runs` rows with status 'error'. One that happened inside a *failed*
    scheduled or dashboard run is folded into that run's entry (it adds the
    report link and summary) -- at most one per run, the nearest; every other
    one stands as its own entry (a terminal run, no log). None is dropped.

The dashboard's finished runs are also persisted (`dashboard_runs`, created
here) so its failures survive a restart and the in-memory history cap.

Log paths come only from DB rows or the dashboard's own runs, and every one
is confined lexically to the two log dirs before any filesystem call (a UNC
path would otherwise open an SMB connection). Logs are read bounded and
decoded with errors="replace".

Items a Gmail cursor task gave up on (`RetrySet`, kv key "retry") are listed
with the ids and subjects the reports already show -- never bodies.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Iterable

from .store import Store

DEFAULT_RETENTION_DAYS = 30
MAX_ENTRIES = 200  # rows the Errors list shows (newest first)
SCAN_ROWS = 500  # rows read per source before merging
TAIL_LINES = 500
TAIL_BYTES = 256 * 1024  # enough for ~500 ordinary lines
ALL_BYTES = 4 * 1024 * 1024  # "show all" still stops somewhere
REASON_BYTES = 8 * 1024  # read from a dashboard log's end for its one-line reason
MATCH_SLACK_S = 2.0
OPEN_WINDOW_S = 6 * 3600  # the dispatcher's hard limit: an abandoned row this recent may still be running
DEFAULT_TASK_TIMEOUT_S = 60 * 60  # schedule.toml's default task_timeout_minutes
MAX_TASK_TIMEOUT_S = 330 * 60  # dispatch.HARD_LIMIT_MINUTES: no task outlives it
MAX_GAVE_UP = 500
TRACEBACK_START = "Traceback (most recent call last):"


# ---------------------------------------------------------------------------
# Paths and logs
# ---------------------------------------------------------------------------

def confine(path: object, dirs: Iterable[Path]) -> Path | None:
    """`path` as a Path if it names a file directly or deeper inside one of
    `dirs` -- decided lexically, with no filesystem access -- else None.
    Relative, UNC, device and NUL-containing paths are refused outright."""
    if not isinstance(path, (str, Path)):
        return None
    text = str(path)
    if not text or "\x00" in text or len(text) > 1024:
        return None
    win = PureWindowsPath(text)
    if text.startswith(("\\\\", "//")) or win.drive.startswith(("\\\\", "//")):
        return None  # UNC or \\?\ device path
    if not os.path.isabs(text):
        return None
    full = os.path.normcase(os.path.abspath(text))  # lexical: no I/O
    for d in dirs:
        root = os.path.normcase(os.path.abspath(str(d))).rstrip("\\/") + os.sep
        if full.startswith(root):
            return Path(text)
    return None


def _open_confined(path: object, dirs: Iterable[Path]):
    """Open a confined log for reading, re-checking containment after links
    are resolved (safe now: the lexical check already passed). None if it
    isn't there or escapes."""
    dirs = list(dirs)
    p = confine(path, dirs)
    if p is None:
        return None
    try:
        real = p.resolve()
        if not any(real.is_relative_to(Path(d).resolve()) for d in dirs) or not real.is_file():
            return None
        return open(real, "rb")
    except (OSError, ValueError):
        return None


def traceback_spans(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line indexes, inclusive, of each Python traceback: the
    "Traceback (most recent call last):" header, its indented frames, and
    the exception line after them."""
    spans = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].lstrip().startswith(TRACEBACK_START):
            j = i + 1
            while j < n and lines[j][:1] in (" ", "\t"):
                j += 1
            end = j if j < n and lines[j].strip() else j - 1
            spans.append((i, max(end, i)))
            i = max(end, i) + 1
        else:
            i += 1
    return spans


def read_log(path: object, dirs: Iterable[Path], *, full: bool = False) -> dict | None:
    """The end of a confined log: the last TAIL_LINES lines, or with `full`
    up to ALL_BYTES from the end. None if there's no readable log.

    -> {text, size, truncated, shown_lines, spans}; `truncated` says whether
    anything before the shown part was left out."""
    fh = _open_confined(path, dirs)
    if fh is None:
        return None
    with fh:
        try:
            size = os.fstat(fh.fileno()).st_size
            want = ALL_BYTES if full else TAIL_BYTES
            start = max(0, size - want)
            fh.seek(start)
            data = fh.read(want)
        except OSError:
            return None
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    cut = start > 0
    if cut and lines:
        lines = lines[1:]  # the first line was entered mid-way
    if not full and len(lines) > TAIL_LINES:
        lines, cut = lines[-TAIL_LINES:], True
    return {
        "text": "\n".join(lines),
        "size": size,
        "truncated": cut,
        "shown_lines": len(lines),
        "spans": traceback_spans(lines),
    }


def _last_line(path: object, dirs: Iterable[Path]) -> str | None:
    fh = _open_confined(path, dirs)
    if fh is None:
        return None
    with fh:
        try:
            size = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, size - REASON_BYTES))
            data = fh.read(REASON_BYTES)
        except OSError:
            return None
    lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    return lines[-1][-200:] if lines else None


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def log_retention_days() -> int:
    """`LMSA_LOG_RETENTION_DAYS` (default 30, at least 1)."""
    raw = os.getenv("LMSA_LOG_RETENTION_DAYS", "")
    try:
        days = int(raw) if raw.strip() else DEFAULT_RETENTION_DAYS
    except ValueError:
        days = DEFAULT_RETENTION_DAYS
    return max(1, days)


def prune_logs(dirs: Iterable[Path], days: int, keep: Iterable[str | Path] = (), now: float | None = None) -> int:
    """Delete `*.log` files directly in `dirs` not modified for `days` days
    (at least 1: a live run's log is always fresh) and not named in `keep`
    (logs of live runs). Never raises; -> how many were deleted."""
    cutoff = (now if now is not None else time.time()) - max(1, days) * 86400
    kept = {os.path.normcase(os.path.abspath(str(k))) for k in keep}
    removed = 0
    for d in dirs:
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                if not e.name.endswith(".log") or not e.is_file(follow_symlinks=False):
                    continue
                if os.path.normcase(os.path.abspath(e.path)) in kept:
                    continue
                if e.stat(follow_symlinks=False).st_mtime >= cutoff:
                    continue
                os.unlink(e.path)
                removed += 1
            except OSError:
                continue
    return removed


def live_dispatch_logs(db_path: Path, now: float | None = None) -> list[str]:
    """Logs of scheduled tasks that may still be running: rows still marked
    "abandoned" (the dispatcher's placeholder while a task runs) that started
    within the dispatcher's hard limit."""
    since = datetime.fromtimestamp((now or time.time()) - OPEN_WINDOW_S, timezone.utc).isoformat(timespec="seconds")
    try:
        with Store(db_path) as db:
            if "log_path" not in _columns(db.conn, "dispatch_runs"):
                return []
            rows = db.conn.execute(
                "SELECT log_path FROM dispatch_runs WHERE outcome='error' AND reason LIKE 'abandoned:%' "
                "AND log_path IS NOT NULL AND started >= ?", (since,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# The feed
# ---------------------------------------------------------------------------

@dataclass
class LocalRun:
    """A finished dashboard run, as the dashboard hands it over."""
    id: str
    task: str
    started: float
    finished: float | None
    returncode: int | None
    stopped: bool
    log_path: str | None
    report_path: str | None = None

    @property
    def failed(self) -> bool:
        return self.stopped or (self.returncode is not None and self.returncode != 0)


DASHBOARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS dashboard_runs (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    argv_json TEXT,
    started REAL NOT NULL,
    finished REAL,
    returncode INTEGER,
    stopped INTEGER NOT NULL DEFAULT 0,
    log_path TEXT,
    report_path TEXT
);
CREATE INDEX IF NOT EXISTS dashboard_runs_task ON dashboard_runs(task, finished);
"""


def record_dashboard_run(db_path: Path, run: LocalRun, argv: list[str] | None = None) -> None:
    """Persist one finished dashboard run (upsert by id). Raises sqlite3.Error."""
    with Store(db_path) as db:
        db.conn.executescript(DASHBOARD_SCHEMA)
        db.conn.execute(
            "INSERT OR REPLACE INTO dashboard_runs(id, task, argv_json, started, finished, returncode, stopped, "
            "log_path, report_path) VALUES (?,?,?,?,?,?,?,?,?)",
            (run.id, run.task, json.dumps(argv) if argv is not None else None, run.started, run.finished,
             run.returncode, int(bool(run.stopped)), run.log_path, run.report_path),
        )
        db.conn.commit()


def _dashboard_rows(conn: sqlite3.Connection, where: str, args: tuple = ()) -> list[LocalRun]:
    if not _columns(conn, "dashboard_runs"):
        return []
    out = []
    for rid, task, started, finished, rc, stopped, log, report in conn.execute(
        "SELECT id, task, started, finished, returncode, stopped, log_path, report_path FROM dashboard_runs " + where,
        args,
    ).fetchall():
        try:
            out.append(LocalRun(str(rid), str(task), float(started), None if finished is None else float(finished),
                                None if rc is None else int(rc), bool(stopped), log, report))
        except (TypeError, ValueError):
            continue
    return out


def prune_dashboard_runs(db_path: Path, days: int, now: float | None = None) -> int:
    """Drop persisted dashboard runs as old as the logs `prune_logs` deletes."""
    cutoff = (now if now is not None else time.time()) - max(1, days) * 86400
    try:
        with Store(db_path) as db:
            if not _columns(db.conn, "dashboard_runs"):
                return 0
            n = db.conn.execute("DELETE FROM dashboard_runs WHERE COALESCE(finished, started) < ?", (cutoff,)).rowcount
            db.conn.commit()
            return n
    except sqlite3.Error:
        return 0


@dataclass
class ErrorEntry:
    id: str  # "d<dispatch_runs.id>" | "r<runs.id>" | "x<dashboard run id>"
    task: str
    when: float  # epoch seconds
    source: str  # "scheduled" | "dashboard" | "terminal"
    outcome: str  # "error" | "skipped" | "exit 3" | "stopped"
    reason: str
    log_path: str | None = None  # server side only; never sent to the page
    report_path: str | None = None
    summary: dict | None = None
    # (start, end) epoch window a `runs` row must start in to belong here
    window: tuple[float, float] = field(default=(0.0, 0.0), repr=False)


def _ts(iso: object) -> float | None:
    try:
        dt = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _task_name(task: str) -> str:
    return task[len("tasks."):] if task.startswith("tasks.") else task


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _summary(raw: object) -> dict | None:
    try:
        value = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _summary_reason(summary: dict | None) -> str:
    if summary and isinstance(summary.get("error"), str):
        return summary["error"][:200]
    if summary:
        return ("recorded an error; " + ", ".join(f"{k}: {v}" for k, v in summary.items()))[:200]
    return "recorded an error (no summary)"


def summary_lines(summary: dict | None) -> list[str]:
    return [f"{k}: {v}" for k, v in (summary or {}).items()]


def _window(started: float, finished: float | None, open_s: float, *,
            open_ended: bool = False) -> tuple[float, float]:
    """The span a `runs` row must start in to belong to this run.

    A run with a real result spans [started, finished] (plus slack) -- even
    when `finished == started`: `dispatch_runs` stores whole seconds, so a
    task that fails in under a second (an import error) looks zero-length.
    Only an `open_ended` run (the dispatcher's "abandoned:" placeholder,
    which never got a finish) gets the task's own timeout from its start."""
    if open_ended:
        # the task was killed at its timeout at the latest, so nothing it recorded starts that late
        return started - MATCH_SLACK_S, started + open_s - 1.0
    end = finished if finished is not None and finished > started else started
    return started - MATCH_SLACK_S, end + MATCH_SLACK_S


def collect(db_path: Path, local: Iterable[LocalRun], log_dirs: Iterable[Path],
            limit: int | None = MAX_ENTRIES, now: float | None = None,
            timeouts: dict[str, float] | None = None) -> tuple[list[ErrorEntry], int]:
    """-> (newest `limit` entries, how many there were in all).

    `local` is the dashboard's in-memory finished runs; persisted ones are
    read from `dashboard_runs` too (memory wins on the same id). `timeouts`
    (task -> seconds, from the schedule) bounds the window of a scheduled row
    that never got a finish time."""
    log_dirs = list(log_dirs)
    now = time.time() if now is None else now
    timeouts = timeouts or {}

    def open_s(task: str) -> float:
        t = timeouts.get(task, DEFAULT_TASK_TIMEOUT_S)
        return min(max(float(t), 60.0), MAX_TASK_TIMEOUT_S)

    failures: list[ErrorEntry] = []
    hosts: list[ErrorEntry] = []  # failed runs a `runs` error row may be folded into
    runs_rows: list[tuple] = []
    dash: dict[str, LocalRun] = {}
    try:
        with Store(db_path) as db:
            for r in _dashboard_rows(db.conn, "WHERE finished IS NOT NULL AND (stopped = 1 OR returncode != 0) "
                                     "ORDER BY finished DESC LIMIT ?", (SCAN_ROWS,)):
                dash[r.id] = r
            cols = _columns(db.conn, "dispatch_runs")
            if cols:
                log_col = "log_path" if "log_path" in cols else "NULL"
                for oid, task, started, finished, outcome, reason, log in db.conn.execute(
                    f"SELECT id, task, started, finished, outcome, reason, {log_col} FROM dispatch_runs "
                    "WHERE outcome IN ('error','skipped') ORDER BY id DESC LIMIT ?", (SCAN_ROWS,),
                ).fetchall():
                    t0 = _ts(started)
                    if t0 is None:
                        continue
                    name = _task_name(task)
                    e = ErrorEntry(
                        id=f"d{oid}", task=name, when=t0, source="scheduled",
                        outcome=outcome, reason=reason or outcome, log_path=log,
                        window=_window(t0, _ts(finished), open_s(name),
                                       open_ended=(reason or "").startswith("abandoned:")),
                    )
                    if outcome == "error" and (reason or "").startswith("abandoned:") and now - t0 < OPEN_WINDOW_S:
                        e.outcome = "running?"  # the placeholder row of a task that may still be going
                        e.reason = "no result yet: still running, or the dispatcher was stopped mid-task"
                    failures.append(e)
                    if outcome == "error":  # a skip never ran, so nothing happened inside it
                        hosts.append(e)
            runs_rows = db.conn.execute(
                "SELECT id, task, started, report_path, summary_json FROM runs "
                "WHERE status='error' ORDER BY id DESC LIMIT ?", (SCAN_ROWS,),
            ).fetchall()
    except sqlite3.Error:
        pass

    for r in local:
        dash[r.id] = r
    for r in dash.values():
        if r.finished is None or not r.failed:
            continue
        e = ErrorEntry(
            id=f"x{r.id}", task=r.task, when=r.started, source="dashboard",
            outcome="stopped" if r.stopped else f"exit {r.returncode}",
            reason="stopped from the dashboard" if r.stopped else (
                f"exit {r.returncode}: {_last_line(r.log_path, log_dirs) or 'no output'}"),
            log_path=r.log_path, report_path=r.report_path, window=_window(r.started, r.finished, 0.0),
        )
        failures.append(e)
        hosts.append(e)

    # Fold: each runs error row picks the nearest failed host whose window
    # holds it; each host keeps only the nearest row that picked it. Every
    # row that doesn't fold stands as its own (terminal) entry.
    by_task: dict[str, list[ErrorEntry]] = {}
    for h in hosts:
        by_task.setdefault(h.task, []).append(h)
    rows = []
    for rid, task, started, report, summ in runs_rows:
        t0 = _ts(started)
        if t0 is None:
            continue
        name = _task_name(task)
        inside = [h for h in by_task.get(name, ()) if h.window[0] <= t0 <= h.window[1]]
        host = min(inside, key=lambda h: abs(t0 - h.when), default=None)
        rows.append((rid, name, t0, report, _summary(summ), host))
    winner: dict[int, tuple] = {}
    for row in rows:
        host = row[5]
        if host is not None:
            best = winner.get(id(host))
            if best is None or abs(row[2] - host.when) < abs(best[2] - host.when):
                winner[id(host)] = row
    folded = {id(r) for r in winner.values()}
    for row in rows:
        rid, name, t0, report, summary, host = row
        if id(row) in folded:
            host.report_path = host.report_path or report
            host.summary = host.summary or summary
            continue
        failures.append(ErrorEntry(
            id=f"r{rid}", task=name, when=t0, source="terminal", outcome="error",
            reason=_summary_reason(summary), report_path=report, summary=summary,
        ))

    failures.sort(key=lambda e: e.when, reverse=True)
    return (failures if limit is None else failures[:limit]), len(failures)


def latest_failed(db_path: Path, local: Iterable[LocalRun]) -> set[str]:
    """Tasks whose most recent finished run -- terminal, scheduled or from
    the dashboard -- failed. A skip isn't a run and doesn't count."""
    latest: dict[str, tuple[float, bool]] = {}

    def see(task: str, when: float | None, failed: bool) -> None:
        if when is None:
            return
        name = _task_name(task)
        prev = latest.get(name)
        if prev is None or when > prev[0] or (when == prev[0] and failed):
            latest[name] = (when, failed)

    for r in local:
        if r.finished is not None:
            see(r.task, r.finished, r.failed)
    try:
        with Store(db_path) as db:
            for r in _dashboard_rows(db.conn, "WHERE finished IS NOT NULL AND rowid IN (SELECT rowid FROM "
                                     "dashboard_runs d WHERE d.finished = (SELECT MAX(finished) FROM dashboard_runs x "
                                     "WHERE x.task = d.task))"):
                see(r.task, r.finished, r.failed)
            for task, finished, status in db.conn.execute(
                "SELECT task, finished, status FROM runs WHERE id IN "
                "(SELECT MAX(id) FROM runs WHERE status IN ('ok','error') GROUP BY task)"
            ).fetchall():
                see(task, _ts(finished), status == "error")
            if _columns(db.conn, "dispatch_runs"):
                # A recent "abandoned" placeholder is a task still running: no verdict yet.
                recent = datetime.fromtimestamp(time.time() - OPEN_WINDOW_S, timezone.utc).isoformat(timespec="seconds")
                for task, finished, outcome in db.conn.execute(
                    "SELECT task, finished, outcome FROM dispatch_runs WHERE id IN "
                    "(SELECT MAX(id) FROM dispatch_runs WHERE outcome IN ('ran','error') "
                    "AND NOT (COALESCE(reason, '') LIKE 'abandoned:%' AND started >= ?) GROUP BY task)",
                    (recent,),
                ).fetchall():
                    see(task, _ts(finished), outcome == "error")
    except sqlite3.Error:
        pass
    return {t for t, (_, failed) in latest.items() if failed}


def gave_up_items(db_path: Path, limit: int = MAX_GAVE_UP) -> tuple[list[dict], int]:
    """Items each Gmail cursor task's RetrySet gave up on: task, msg_id,
    subject, attempts and last error -- what the reports show, no bodies."""
    items: list[dict] = []
    try:
        with Store(db_path) as db:
            rows = db.conn.execute("SELECT ns, value_json FROM kv WHERE key='retry' ORDER BY ns").fetchall()
    except sqlite3.Error:
        return [], 0
    for ns, raw in rows:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        for msg_id, v in value.items():
            if not isinstance(v, dict) or not v.get("gave_up"):
                continue
            attempts = v.get("attempts")
            items.append({
                "task": _task_name(str(ns)),
                "msg_id": str(msg_id)[:200],
                "subject": str(v.get("subject") or "")[:300],
                "attempts": attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else None,
                "error": str(v.get("error") or "")[:300],
            })
    return items[:limit], len(items)


_ID_RE = re.compile(r"(d\d{1,12}|r\d{1,12}|x[0-9a-f]{1,32})")


def valid_id(entry_id: str) -> bool:
    return bool(_ID_RE.fullmatch(entry_id))
