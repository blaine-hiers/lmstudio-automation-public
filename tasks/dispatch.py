"""Load-aware dispatcher: the only thing Task Scheduler runs (every 30 min).

    python -m tasks.dispatch             # run whatever is due, if the PC is quiet
    python -m tasks.dispatch --dry-run   # print what would run / why not; loads nothing, records nothing

Each invocation:
  1. Reads `config/schedule.toml` (per-task cadence, deadline, model role).
  2. Works out which tasks are due from its own `dispatch_runs` table -- NOT
     from `runs`, so a task that doesn't record its runs still dispatches
     correctly. A due occurrence is done once it has a ran / error / skipped
     row; `deferred` rows just mean "seen, but the PC was busy".
  3. Skips any occurrence still blocked at its deadline. The deadline clock
     starts when the dispatcher first saw the occurrence due (normally the
     due time itself), so a PC that was off or asleep never "misses" a
     deadline it wasn't awake for -- the task waits for the next quiet slot.
  4. Checks the gate (`lmsa.gate.busy_reasons`). Busy -> every due task is
     deferred with the reason, and retried next time.
     A no-model task with its own lower `idle_minutes` (e.g.
     master_report's 0) is checked against that instead, so input activity
     alone doesn't hold it back.
  5. Runs due tasks grouped by model: no-model tasks, then the whisper
     batch (speech-to-text in the task's own process: every LM Studio model
     the dispatcher loaded is unloaded first, and the batch waits while any
     other LM Studio model is still resident), then the chat batch
     (one load), unload, then the vision batch. Vision-batch tasks get
     `LMSA_CHAT_MODEL=<vision model>` so their text calls use the model that
     is already resident instead of JIT-loading the chat model beside it.
     Each task runs as a subprocess (`python -m tasks.<name>`) with a
     timeout (the whole process tree is killed on timeout), so a crashing or
     hanging task can't take the dispatcher down.
  6. A watchdog re-checks the gate BETWEEN tasks (a task already running is
     not interrupted). If the PC has turned busy it stops, unloads what it
     loaded, and defers the rest. It also stops starting new tasks once the
     run has used its time budget (`max_run_minutes`).
  7. Only models the dispatcher loaded -- or that appeared while one of its
     tasks ran (JIT loads) -- are unloaded (`lmsa.models`).

Before a task starts, its row is written as "error: abandoned", then updated
with the real outcome; if Task Scheduler kills the dispatcher mid-task, the
occurrence stays closed so a later dispatcher can't run it twice beside an
orphan.

Each task's combined stdout and stderr is kept in
`data/dispatch/logs/<task>-<ts>.log`, and its path in the row's `log_path`
(shown by the dashboard's Errors view). The reason text is still the last
stderr line. Logs older than `LMSA_LOG_RETENTION_DAYS` (default 30) are
pruned before each real dispatch.

Reporting: whenever the dispatcher decides skips, it records its own
`Store.record_run("dispatch")` row straight away (before running anything,
so the evening recap -- itself a dispatched task -- sees them), listing every
skip not yet reported. A skip decided after that day's recap already ran is
held back (`reported_at` stays NULL) and carried into the next day's first
dispatch row. When something ran or failed it records another row with the
counts, and rewrites today's `dispatch-YYYYMMDD.html` from the day's
`dispatch_runs` rows. Invocations where everything was simply deferred or
nothing was due only write `dispatch_runs` rows, so the recap isn't flooded
with 48 no-op entries a day.

A lock file (`data/dispatch.lock`, held with an OS file lock that is released
automatically if the process dies) keeps two dispatchers from running at once.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import sqlite3
import subprocess
import sys
import time as _time
import tempfile
import threading
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Callable, Iterable

from lmsa import Report, Store, Tile
from lmsa import gate as gate_mod
from lmsa.config import settings
from lmsa.errors import log_retention_days, prune_logs
from lmsa.gate import GateConfig
from lmsa.models import Lms, ModelError, ModelManager
from lmsa.store import DEFAULT_DB

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "schedule.toml"
TASK_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MODEL_ROLES = ("chat", "vision")  # LM Studio models the dispatcher loads
# Not an LM Studio model: the task loads faster-whisper itself. The dispatcher
# only clears the GPU of LM Studio models for it and checks VRAM (#62).
WHISPER = "whisper"
MODEL_CLASSES = (*MODEL_ROLES, WHISPER)
GROUP_ORDER = (None, WHISPER, "chat", "vision")  # no-model tasks first, then one model at a time
TERMINAL = ("ran", "error", "skipped")
WATCHDOG_SETTLE_S = 3.0  # let nvidia-smi's utilization sample decay after a task's inference
REASON_MAX = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS dispatch_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    due_at TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    outcome TEXT NOT NULL,
    reason TEXT,
    reported_at TEXT,
    log_path TEXT
);
CREATE INDEX IF NOT EXISTS dispatch_runs_task_due ON dispatch_runs(task, due_at);
"""
RECAP_TASKS = ("evening_recap", "tasks.evening_recap")  # names it may record its runs under
HARD_LIMIT_MINUTES = 330  # Task Scheduler kills the dispatcher at 6 h; finish every task well before
ABANDONED = "abandoned: the dispatcher stopped before this task finished"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cadence:
    kind: str  # "daily" | "weekly" | "every"
    at: time = time(0, 0)
    weekday: int = 0  # Monday = 0
    hours: int = 0

    def __str__(self) -> str:
        if self.kind == "daily":
            return f"daily {self.at:%H:%M}"
        if self.kind == "weekly":
            return f"weekly {WEEKDAYS[self.weekday]} {self.at:%H:%M}"
        return f"every {self.hours} hours"


def _parse_hhmm(text: str) -> time:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text.strip())
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"bad time {text!r} (want HH:MM)")
    return time(int(m.group(1)), int(m.group(2)))


def parse_cadence(text: str) -> Cadence:
    """"daily 06:30" | "weekly sun 03:00" | "every 6 hours" (N divides 24)."""
    parts = str(text).strip().lower().split()
    if len(parts) == 2 and parts[0] == "daily":
        return Cadence("daily", at=_parse_hhmm(parts[1]))
    if len(parts) == 3 and parts[0] == "weekly" and parts[1][:3] in WEEKDAYS:
        return Cadence("weekly", at=_parse_hhmm(parts[2]), weekday=WEEKDAYS.index(parts[1][:3]))
    if len(parts) == 3 and parts[0] == "every" and parts[2] in ("hour", "hours") and parts[1].isdigit():
        n = int(parts[1])
        if n < 1 or 24 % n:
            raise ValueError(f"every N hours: N must divide 24 (got {n})")
        return Cadence("every", hours=n)
    raise ValueError(f"bad cadence {text!r} (want 'daily HH:MM', 'weekly <day> HH:MM' or 'every N hours')")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    cadence: Cadence
    deadline: timedelta
    model: str | None  # "chat" | "vision" | "whisper" | None
    timeout: timedelta
    # Per-task input-idle requirement (minutes) overriding the day/overnight
    # one when lower; model "none" tasks only (#103: the 06:45 master email).
    idle_minutes: float | None = None


@dataclass
class Schedule:
    gate: GateConfig
    overnight: tuple[time, time] = (time(1, 0), time(7, 0))
    overnight_idle_minutes: float = 0
    model_ttl_s: int = 900
    stop_server_at_end: bool = False
    max_run_minutes: float = 240  # stop starting tasks after this; Task Scheduler's limit is 6 h
    model_sizes_mb: dict[str, int] = field(default_factory=lambda: {"chat": 8200, "vision": 6400, WHISPER: 2000})
    tasks: list[TaskSpec] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _num(value: object, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def load_schedule(path: Path = CONFIG_PATH, tasks_dir: Path = ROOT / "tasks") -> Schedule:
    """Parse schedule.toml. A bad task entry is reported in `problems` and
    left out; the rest still load. A bad [dispatch] section raises."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    d = data.get("dispatch", {})
    blocked = d.get("blocked_processes", [])
    if not isinstance(blocked, list) or not all(isinstance(b, str) for b in blocked):
        raise ValueError("dispatch.blocked_processes must be a list of strings")
    overnight = d.get("overnight", ["01:00", "07:00"])
    if not (isinstance(overnight, list) and len(overnight) == 2):
        raise ValueError('dispatch.overnight must be ["HH:MM", "HH:MM"]')
    default_timeout = _num(d.get("task_timeout_minutes", 60), "task_timeout_minutes", minimum=1)
    sched = Schedule(
        gate=GateConfig(
            gpu_util_max=int(_num(d.get("gpu_util_max", 30), "gpu_util_max")),
            vram_headroom_mb=int(_num(d.get("vram_headroom_mb", 1024), "vram_headroom_mb")),
            idle_minutes=_num(d.get("idle_minutes", 15), "idle_minutes"),
            blocked_processes=tuple(blocked),
        ),
        overnight=(_parse_hhmm(overnight[0]), _parse_hhmm(overnight[1])),
        overnight_idle_minutes=_num(d.get("overnight_idle_minutes", 0), "overnight_idle_minutes"),
        model_ttl_s=int(_num(d.get("model_ttl_seconds", 900), "model_ttl_seconds", minimum=60)),
        stop_server_at_end=bool(d.get("stop_server_at_end", False)),
        max_run_minutes=_num(d.get("max_run_minutes", 240), "max_run_minutes", minimum=1),
    )
    if sched.max_run_minutes >= HARD_LIMIT_MINUTES:
        raise ValueError(f"dispatch.max_run_minutes must be below {HARD_LIMIT_MINUTES}")
    m = data.get("models", {})
    for role in MODEL_CLASSES:
        if f"{role}_size_mb" in m:
            sched.model_sizes_mb[role] = int(_num(m[f"{role}_size_mb"], f"{role}_size_mb", minimum=1))

    for name, t in (data.get("tasks") or {}).items():
        try:
            if not TASK_NAME_RE.fullmatch(name):
                raise ValueError("not a valid task module name")
            if not (tasks_dir / f"{name}.py").exists():
                raise ValueError(f"no module tasks/{name}.py")
            if not isinstance(t, dict):
                raise ValueError("must be a table")
            if t.get("enabled", True) is False:
                continue
            model = t.get("model", "none")
            if model not in (*MODEL_CLASSES, "none"):
                raise ValueError(f"model must be one of chat, vision, whisper, none (got {model!r})")
            deadline_h = _num(t.get("deadline_hours", 6), "deadline_hours", minimum=0.5)
            timeout_m = _num(t.get("timeout_minutes", default_timeout), "timeout_minutes", minimum=1)
            idle_m = None
            if "idle_minutes" in t:
                if model != "none":
                    raise ValueError('idle_minutes is only allowed on model = "none" tasks')
                idle_m = _num(t["idle_minutes"], "idle_minutes")
            sched.tasks.append(TaskSpec(
                name=name,
                cadence=parse_cadence(t.get("cadence", "")),
                deadline=timedelta(hours=deadline_h),
                model=None if model == "none" else model,
                timeout=timedelta(minutes=timeout_m),
                idle_minutes=idle_m,
            ))
        except (ValueError, TypeError) as exc:
            sched.problems.append(f"tasks.{name}: {exc}")
    return sched


# ---------------------------------------------------------------------------
# Time (all comparisons aware; wall-clock schedule in the local zone)
# ---------------------------------------------------------------------------

def _wall(d: date, t: time, tz: tzinfo | None) -> datetime:
    """The aware instant of wall-clock `d t` in `tz` (None = system local).

    A time skipped by a DST spring-forward resolves to the pre-transition
    offset (i.e. an hour later in the new offset); an ambiguous fall-back
    time resolves to its first occurrence. Either way it's one instant per day.
    """
    naive = datetime.combine(d, t)
    return naive.replace(tzinfo=tz) if tz is not None else naive.astimezone()


def latest_occurrence(c: Cadence, now: datetime, tz: tzinfo | None = None) -> datetime:
    """The most recent scheduled instant <= now (UTC)."""
    today = now.astimezone(tz).date()
    candidates: list[datetime] = []
    for back in range(0, 8):
        d = today - timedelta(days=back)
        if c.kind == "daily":
            candidates.append(_wall(d, c.at, tz))
        elif c.kind == "weekly":
            if d.weekday() == c.weekday:
                candidates.append(_wall(d, c.at, tz))
        else:
            candidates += [_wall(d, time(h), tz) for h in range(0, 24, c.hours)]
        past = [x for x in candidates if x <= now]
        if past:
            return max(past).astimezone(timezone.utc)
    raise AssertionError("unreachable: a weekly cadence recurs within 8 days")


def in_overnight(now: datetime, window: tuple[time, time], tz: tzinfo | None = None) -> bool:
    t = now.astimezone(tz).time()
    start, end = window
    return start <= t < end if start <= end else (t >= start or t < end)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dispatch_runs)")}
    if "reported_at" not in cols:  # table created by an earlier build
        conn.execute("ALTER TABLE dispatch_runs ADD COLUMN reported_at TEXT")
    if "log_path" not in cols:  # table created before task output was kept
        conn.execute("ALTER TABLE dispatch_runs ADD COLUMN log_path TEXT")
    conn.commit()


def record(conn, task: str, due_at: str, outcome: str, reason: str | None,
           started: datetime, finished: datetime | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO dispatch_runs(task, due_at, started, finished, outcome, reason) VALUES (?,?,?,?,?,?)",
        (task, due_at, _iso(started), _iso(finished or started), outcome, (reason or None) and reason[:REASON_MAX]),
    )
    conn.commit()
    return cur.lastrowid


def finish(conn, row_id: int, outcome: str, reason: str | None, finished: datetime) -> None:
    conn.execute(
        "UPDATE dispatch_runs SET outcome=?, reason=?, finished=? WHERE id=?",
        (outcome, (reason or None) and reason[:REASON_MAX], _iso(finished), row_id),
    )
    conn.commit()


def set_log_path(conn, row_id: int, path: str | Path | None) -> None:
    conn.execute("UPDATE dispatch_runs SET log_path=? WHERE id=?", (str(path) if path else None, row_id))
    conn.commit()


def dispatch_log_dir() -> Path:
    """`data/dispatch/logs`, beside the store (so `LMSA_DB` moves it too)."""
    return Path(os.path.abspath(Path(os.getenv("LMSA_DB", str(DEFAULT_DB))).parent / "dispatch" / "logs"))


def unreported_skips(conn) -> list[tuple[int, str, str, str]]:
    """(id, task, due_at, reason) of skipped rows no recap has been handed yet."""
    return conn.execute(
        "SELECT id, task, due_at, reason FROM dispatch_runs "
        "WHERE outcome='skipped' AND reported_at IS NULL ORDER BY id"
    ).fetchall()


def recap_ran_today(conn, now: datetime, tz: tzinfo | None = None) -> bool:
    """Has the evening recap already started today (local)? Then a `runs`
    row written now would never be shown by any recap."""
    today = now.astimezone(tz).date()
    marks = ",".join("?" * len(RECAP_TASKS))
    rows = conn.execute(f"SELECT started FROM runs WHERE task IN ({marks})", RECAP_TASKS).fetchall()
    rows += conn.execute(
        f"SELECT started FROM dispatch_runs WHERE task IN ({marks}) AND outcome IN ('ran','error')", RECAP_TASKS
    ).fetchall()
    for (started,) in rows:
        try:
            if _parse_iso(started).astimezone(tz).date() == today:
                return True
        except (TypeError, ValueError):
            continue
    return False


def mark_reported(conn, ids: list[int], when: datetime) -> None:
    conn.executemany("UPDATE dispatch_runs SET reported_at=? WHERE id=?", [(_iso(when), i) for i in ids])
    conn.commit()


@dataclass
class Pending:
    spec: TaskSpec
    due_at: str  # UTC ISO key
    first_seen: datetime
    last_reason: str | None


def plan(conn, sched: Schedule, now: datetime, tz: tzinfo | None = None):
    """-> (pending, expired, superseded, up_to_date).

    expired / superseded are (task, due_at, reason) triples to record as
    skipped; up_to_date is (task, due_at, outcome) for the report."""
    pending: list[Pending] = []
    expired: list[tuple[str, str, str]] = []
    superseded: list[tuple[str, str, str]] = []
    up_to_date: list[tuple[str, str, str]] = []
    for spec in sched.tasks:
        due = latest_occurrence(spec.cadence, now, tz)
        key = _iso(due)
        rows = conn.execute(
            "SELECT outcome, started, reason FROM dispatch_runs WHERE task=? AND due_at=? ORDER BY id",
            (spec.name, key),
        ).fetchall()
        terminal = [r for r in rows if r[0] in TERMINAL]
        # Older occurrences still waiting (deferred, never finished) are
        # superseded by this one: close them out as skipped.
        for old_due, last in conn.execute(
            "SELECT due_at, (SELECT reason FROM dispatch_runs x WHERE x.task=d.task AND x.due_at=d.due_at "
            "ORDER BY x.id DESC LIMIT 1) FROM dispatch_runs d WHERE task=? AND due_at<? AND outcome='deferred' "
            "AND NOT EXISTS (SELECT 1 FROM dispatch_runs t WHERE t.task=d.task AND t.due_at=d.due_at "
            "AND t.outcome IN ('ran','error','skipped')) GROUP BY due_at",
            (spec.name, key),
        ).fetchall():
            superseded.append((spec.name, old_due, f"superseded by the next scheduled run (last blocked by: {last})"))
        if terminal:
            up_to_date.append((spec.name, key, terminal[-1][0]))
            continue
        deferred = [r for r in rows if r[0] == "deferred"]
        first_seen = _parse_iso(deferred[0][1]) if deferred else now
        last_reason = deferred[-1][2] if deferred else None
        if now >= max(due, first_seen) + spec.deadline:
            expired.append((spec.name, key, f"deadline passed while busy: {last_reason or 'no quiet window'}"))
            continue
        pending.append(Pending(spec, key, first_seen, last_reason))
    return pending, expired, superseded, up_to_date


# ---------------------------------------------------------------------------
# Running tasks
# ---------------------------------------------------------------------------

def task_command(spec: TaskSpec) -> list[str]:
    return [sys.executable, "-m", f"tasks.{spec.name}"]


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and every process it started (a task's own children)."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()  # no-op if taskkill already got it
    except OSError:
        pass


class TaskOutcome(tuple):
    """`(outcome, reason)` -- compares and unpacks like the plain pair -- plus
    `log_path`: where the task's output was kept, or None."""

    log_path: Path | None

    def __new__(cls, outcome: str, reason: str | None, log_path: Path | None = None):
        self = super().__new__(cls, (outcome, reason))
        self.log_path = log_path
        return self


PIPE_DRAIN_S = 5.0  # after exit, how long to wait for stderr EOF (a grandchild may hold the pipe)
_LINE_KEEP = 4096  # chars of a still-open stderr line kept verbatim; more is folded into a flag


class _LastStderrLine:
    """The last non-blank stderr line, fed in chunks -- what
    `communicate()` + `splitlines()` gave, without keeping all of stderr.

    Only the open (unterminated) line is buffered, and past _LINE_KEEP chars
    it is folded down to "has text" plus its last 200 chars, which is all the
    reason ever shows (`[-200:]`)."""

    def __init__(self) -> None:
        import codecs

        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._open = ""  # the unterminated line so far (maybe folded)
        self._open_text = False  # the folded-away part of it had non-blank text
        self.last: str | None = None

    def feed(self, data: bytes, final: bool = False) -> None:
        parts = (self._open + self._dec.decode(data, final)).splitlines(keepends=True)
        tail = ""
        if parts and not final and parts[-1].splitlines()[0] == parts[-1]:
            tail = parts.pop()  # no line break yet: the line stays open
        carried = self._open_text
        for i, part in enumerate(parts):
            line = part.splitlines()[0]
            if line.strip() or (i == 0 and carried):
                self.last = line[-200:]
        if parts:
            self._open_text = False
        self._open = tail
        if len(self._open) > _LINE_KEEP:
            self._open_text = self._open_text or bool(self._open[:-200].strip())
            self._open = self._open[-200:]

    def line(self) -> str | None:
        """The answer so far, counting the open line as the last one."""
        if self._open.strip() or (self._open and self._open_text):
            return self._open[-200:]
        return self.last


class _Tee:
    """Copies the task's stderr into its log (the same open file its stdout
    goes to; stderr passes through a pipe first, so a stderr line can land a
    little after stdout written at the same moment) and tracks the reason line.

    `detach()` stops the log writes and closes the log while the thread
    keeps draining the pipe, so a grandchild still holding stderr never
    blocks on a full pipe. Once the run returns nothing more is written
    to the log from stderr; a grandchild that inherited the task's stdout
    can still append to it until it exits."""

    def __init__(self, pipe, sink):
        self.pipe, self.sink = pipe, sink
        self.reason = _LastStderrLine()
        self.lock = threading.Lock()
        self.detached = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = os.read(self.pipe.fileno(), 65536)
                if not chunk:
                    break
                with self.lock:
                    self.reason.feed(chunk)
                    if not self.detached:
                        try:  # unbuffered, at the file pointer the task's stdout handle shares
                            os.write(self.sink.fileno(), chunk)
                        except (OSError, ValueError):
                            pass
        except (OSError, ValueError):
            pass
        finally:
            with self.lock:
                self.reason.feed(b"", final=True)
            try:
                self.pipe.close()
            except OSError:
                pass
            self.detach()

    def detach(self) -> None:
        with self.lock:
            if not self.detached:
                self.detached = True
                try:
                    self.sink.close()
                except OSError:
                    pass

    def finish(self, wait_s: float) -> str | None:
        """Wait up to `wait_s` for stderr EOF, detach the log, -> reason line."""
        self.thread.join(wait_s)
        self.detach()
        with self.lock:
            return self.reason.line()


def _open_log(path: Path):
    """Open the log for writing. On Windows it's shared for delete/rename, so
    a grandchild that inherited the task's stdout can't pin it (pruning,
    moving). Raises OSError."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        return open(path, "wb")
    import _winapi
    import msvcrt

    handle = _winapi.CreateFile(
        str(path), 0x40000000,  # GENERIC_WRITE
        0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | WRITE | DELETE
        0, 2, 0x80, 0,  # CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL
    )
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_WRONLY)
    except OSError:
        _winapi.CloseHandle(handle)
        raise
    return open(fd, "wb")


def run_task_subprocess(
    spec: TaskSpec, env: dict[str, str] | None = None, timeout_s: float | None = None,
    log_path: Path | None = None,
) -> TaskOutcome:
    """Run `python -m tasks.<name>`; -> ("ran", None) or ("error", reason).

    `env` is layered over the dispatcher's environment; `timeout_s` defaults
    to the task's own timeout. On timeout the whole process tree is killed.
    Combined stdout+stderr go to `log_path` (a discarded temp file when None
    or when it can't be created); the result's `.log_path` says where it went.

    Output goes to a file, not a pipe we wait on: only the task process
    itself is waited for, so a grandchild still holding its handles can't
    stall the dispatcher. stderr alone passes through a pipe (teed into the
    log by a thread) so the reason is still the last stderr line."""
    timeout = spec.timeout.total_seconds() if timeout_s is None else timeout_s
    kept: Path | None = None
    sink = None
    if log_path is not None:
        try:
            sink = _open_log(Path(log_path))
            kept = Path(log_path)
        except OSError:
            sink = None
    if sink is None:
        sink = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            task_command(spec),
            cwd=str(ROOT),
            stdout=sink,
            stderr=subprocess.PIPE,
            env={**os.environ, **(env or {})},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        sink.close()
        return TaskOutcome("error", f"couldn't start: {exc}", kept)
    tee = _Tee(proc.stderr, sink)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            pass
        tee.finish(PIPE_DRAIN_S)
        return TaskOutcome("error", f"timed out after {timeout / 60:g} min (killed with its child processes)", kept)
    last = tee.finish(PIPE_DRAIN_S)
    if proc.returncode == 0:
        return TaskOutcome("ran", None, kept)
    return TaskOutcome("error", f"exit {proc.returncode}: {(last or 'no stderr')[-200:]}", kept)


def _takes_log_path(fn: Callable) -> bool:
    """Does this runner accept `log_path=`? (Injected test runners don't.)"""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "log_path" or p.kind is p.VAR_KEYWORD for p in params)


@dataclass
class Result:
    now: datetime
    overnight: bool
    idle_minutes: float
    busy: list[str] = field(default_factory=list)  # the first gate check
    ran: list[tuple[str, float]] = field(default_factory=list)  # (task, seconds)
    errors: list[tuple[str, str]] = field(default_factory=list)
    deferred: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    would_run: list[tuple[str, str | None]] = field(default_factory=list)  # dry-run: (task, model)
    up_to_date: list[tuple[str, str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    watchdog: str | None = None

    @property
    def eventful(self) -> bool:
        """Worth a `runs` row of its own (skips get theirs from `record_skips`)."""
        return bool(self.ran or self.errors or self.notes)


GateFn = Callable[..., list[str]]


def dispatch(
    sched: Schedule,
    conn,
    *,
    gate_fn: GateFn,
    models,
    model_names: dict[str, str],
    run_task: Callable[..., tuple[str, str | None]] | None = None,
    on_skips: Callable[[Result], None] | None = None,
    now: datetime | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    tz: tzinfo | None = None,
    dry_run: bool = False,
    sleep: Callable[[float], None] = _time.sleep,
    log_dir: Path | None = None,
) -> Result:
    now = now or clock()
    run_task = run_task or run_task_subprocess
    keeps_logs = log_dir is not None and not dry_run and _takes_log_path(run_task)
    overnight = in_overnight(now, sched.overnight, tz)
    idle = sched.overnight_idle_minutes if overnight else sched.gate.idle_minutes
    res = Result(now=now, overnight=overnight, idle_minutes=idle, notes=list(sched.problems))

    def rec(task, due_at, outcome, reason, started=None):
        if not dry_run:
            record(conn, task, due_at, outcome, reason, started or clock(), clock())

    def elapsed_min() -> float:
        return (clock() - now).total_seconds() / 60

    budget_msg = f"time budget used ({sched.max_run_minutes:g} min); the rest wait for the next run"

    def defer(items: Iterable[Pending], reason: str) -> None:
        for p in items:
            res.deferred.append((p.spec.name, reason))
            rec(p.spec.name, p.due_at, "deferred", reason)

    pending, expired, superseded, res.up_to_date = plan(conn, sched, now, tz)
    for task, due_at, reason in [*superseded, *expired]:
        res.skipped.append((task, reason))
        rec(task, due_at, "skipped", reason)
    if on_skips is not None and not dry_run:
        try:  # hand skips to the recap BEFORE running anything (the recap may be in this batch)
            on_skips(res)
        except Exception as exc:
            res.notes.append(f"couldn't record skips: {type(exc).__name__}: {exc}")
    if not pending:
        return res

    def task_idle(p: Pending) -> float:
        """The input-idle requirement for one task: the schedule-wide one,
        or the task's own `idle_minutes` when that is lower."""
        own = p.spec.idle_minutes
        return idle if own is None else min(idle, own)

    # One gate check per distinct idle requirement (just one, unless a
    # pending task sets its own idle_minutes): a busy verdict only defers
    # the tasks it applies to.
    checks: dict[float, list[str]] = {}
    for p in pending:
        if task_idle(p) not in checks:
            checks[task_idle(p)] = gate_fn(need_vram_mb=0, idle_minutes=task_idle(p))
    blocked = [p for p in pending if checks[task_idle(p)]]
    res.busy = checks[task_idle(blocked[0])] if blocked else []
    for p in blocked:
        defer([p], "; ".join(checks[task_idle(p)]))
    pending = [p for p in pending if not checks[task_idle(p)]]
    if not pending:
        return res

    # A task with its own idle_minutes runs first in its group, so an ordinary
    # task's between-task watchdog can't defer it along with the rest. With no
    # such task this is the plain due-time order.
    groups = {role: sorted((p for p in pending if p.spec.model == role),
                           key=lambda p: (p.spec.idle_minutes is None, p.due_at))
              for role in GROUP_ORDER}
    stop: str | None = None
    try:
        for role in GROUP_ORDER:
            items = groups[role]
            if not items:
                continue
            if not stop and elapsed_min() >= sched.max_run_minutes:
                stop = budget_msg
            if stop:
                defer(items, stop)
                continue
            model = model_names.get(role) if role in MODEL_ROLES else None
            if role == WHISPER:
                # Whisper never shares the GPU with an LM Studio model: unload
                # ours, and wait while one we don't own is still resident.
                if not dry_run:
                    res.notes += models.release_all()
                try:
                    resident = sorted(models.snapshot())
                except Exception as exc:  # can't ask `lms`: the VRAM check below still guards
                    resident = []
                    res.notes.append(f"lms ps before the whisper batch: {exc}")
                if resident:
                    defer(items, f"LM Studio model(s) still loaded ({', '.join(resident)}); "
                                 "whisper never shares the GPU with them")
                    continue
                if res.ran or res.errors:
                    sleep(WATCHDOG_SETTLE_S)
                busy = gate_fn(need_vram_mb=sched.model_sizes_mb[WHISPER], idle_minutes=idle)
                if busy:
                    defer(items, "; ".join(busy))
                    continue
                if dry_run:
                    res.would_run += [(p.spec.name, WHISPER) for p in items]
                    continue
            elif model:
                try:
                    need = 0 if models.is_resident(model) else sched.model_sizes_mb[role]
                except Exception as exc:
                    defer(items, f"can't query LM Studio: {exc}")
                    continue
                if res.ran or res.errors:  # this check doubles as the between-groups watchdog
                    sleep(WATCHDOG_SETTLE_S)
                busy = gate_fn(need_vram_mb=need, idle_minutes=idle)
                if busy:  # maybe only VRAM for this model; the next group re-checks for itself
                    defer(items, "; ".join(busy))
                    continue
                if dry_run:
                    res.would_run += [(p.spec.name, model) for p in items]
                    continue
                try:
                    models.ensure(model, sched.model_sizes_mb[role])
                except Exception as exc:
                    defer(items, str(exc))
                    continue
            elif dry_run:
                res.would_run += [(p.spec.name, None) for p in items]
                continue
            # A vision-batch task's text calls (extract/chat) go to the resident
            # vision model rather than JIT-loading the chat model beside it.
            env = {"LMSA_CHAT_MODEL": model} if role == "vision" and model else {}
            try:
                for i, p in enumerate(items):
                    if elapsed_min() >= sched.max_run_minutes:
                        stop = budget_msg
                        defer(items[i:], stop)
                        break
                    if i > 0:  # watchdog; the first task's gate check just ran
                        sleep(WATCHDOG_SETTLE_S)
                        busy = gate_fn(need_vram_mb=0, idle_minutes=task_idle(p))
                        if busy:
                            stop = res.watchdog = "; ".join(busy)
                            defer(items[i:], f"watchdog stopped: {stop}")
                            break
                    # Never let a task outlive Task Scheduler's hard limit on the dispatcher.
                    timeout_s = min(p.spec.timeout.total_seconds(), max(60.0, (HARD_LIMIT_MINUTES - elapsed_min()) * 60))
                    started = clock()
                    row_id = None
                    if not dry_run:  # closed up front: survives the dispatcher being killed mid-task
                        row_id = record(conn, p.spec.name, p.due_at, "error", ABANDONED, started)
                    log_kw: dict = {}
                    if keeps_logs and row_id is not None:
                        log_kw["log_path"] = Path(log_dir) / f"{p.spec.name}-{started.astimezone():%Y%m%d-%H%M%S}.log"
                        set_log_path(conn, row_id, log_kw["log_path"])  # an abandoned row still points at it
                    try:
                        before = models.snapshot()
                    except Exception as exc:
                        before = None
                        res.notes.append(f"lms ps before {p.spec.name}: {exc}")
                    t0 = _time.monotonic()
                    result = None
                    try:
                        result = run_task(p.spec, env, timeout_s, **log_kw)
                        outcome, reason = result
                    except Exception as exc:  # a broken runner must not kill the batch
                        outcome, reason = "error", f"{type(exc).__name__}: {exc}"
                    if before is not None:
                        try:
                            for jit in models.adopt_new(before):
                                res.notes.append(f"{p.spec.name} JIT-loaded {jit}; unloading it with the batch")
                        except Exception as exc:
                            res.notes.append(f"lms ps after {p.spec.name}: {exc}")
                    if row_id is not None:
                        finish(conn, row_id, outcome, reason, clock())
                        if log_kw:  # None if the log file couldn't be created
                            set_log_path(conn, row_id, getattr(result, "log_path", None))
                    if outcome == "ran":
                        res.ran.append((p.spec.name, _time.monotonic() - t0))
                    else:
                        res.errors.append((p.spec.name, reason or "failed"))
            finally:
                # Ours: the batch model plus anything its tasks JIT-loaded.
                res.notes += models.release_all()
    finally:
        if not dry_run:
            res.notes += models.release_all()
    if not dry_run and sched.stop_server_at_end and res.ran:
        try:
            models.stop_server()
        except Exception as exc:
            res.notes.append(f"server not stopped: {exc}")
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _clip(items: list[str], limit: int = REASON_MAX) -> str:
    out = ", ".join(items)
    return out if len(out) <= limit else out[: limit - 20].rsplit(",", 1)[0] + f", ... (+{len(items)} total)"


def summary(res: Result) -> dict:
    """The end-of-run `runs` summary. Skips are reported separately (and
    earlier) by `record_skips`, so they aren't counted twice in the recap."""
    s: dict = {
        "ran": len(res.ran),
        "errors": len(res.errors),
        "deferred": len(res.deferred),
    }
    if res.errors:
        s["failed_tasks"] = _clip([t for t, _ in res.errors])
    if res.watchdog:
        s["watchdog"] = res.watchdog[:REASON_MAX]
    return s


def skip_summary(rows: list[tuple[int, str, str, str]], tz: tzinfo | None = None) -> dict:
    return {
        "skipped": len(rows),
        "skipped_tasks": _clip([f"{task} (due {_local_hm(due, tz)})" for _, task, due, _ in rows]),
    }


def _write_report(conn, res: Result, tz: tzinfo | None = None) -> Path:
    day = res.now.astimezone(tz).date()
    return build_report(conn, res, day, tz).write(name=f"dispatch-{day:%Y%m%d}.html")


def record_skips(db: Store, res: Result, tz: tzinfo | None = None) -> None:
    """Give the evening recap every skip it hasn't been handed yet.

    Writes a finished `runs` row ("dispatch") listing all unreported skips --
    this run's and any carried over -- then marks them reported. If today's
    recap has already run, nobody would see a row written now: new skips
    still get a row (for the log) but stay unreported, and the next day's
    first dispatcher run carries them into a row that day's recap shows."""
    rows = unreported_skips(db.conn)
    if not rows:
        return
    held = recap_ran_today(db.conn, res.now, tz)
    if held and not res.skipped:
        return  # nothing new; carried into tomorrow's first row
    with db.record_run("dispatch") as run:
        run.summary = skip_summary(rows, tz)
        run.report_path = _write_report(db.conn, res, tz)
    if not held:
        mark_reported(db.conn, [r[0] for r in rows], res.now)


def _local_hm(iso: str, tz: tzinfo | None) -> str:
    try:
        return _parse_iso(iso).astimezone(tz).strftime("%a %H:%M")
    except (TypeError, ValueError):
        return str(iso)


def build_report(conn, res: Result, day: date, tz: tzinfo | None = None) -> Report:
    """Today's dispatcher log, rebuilt from every `dispatch_runs` row today."""
    rows = []
    for r in conn.execute(
        "SELECT task, due_at, started, finished, outcome, reason FROM dispatch_runs ORDER BY id"
    ).fetchall():
        try:
            if _parse_iso(r[2]).astimezone(tz).date() == day:
                rows.append(r)
        except (TypeError, ValueError):
            continue
    by = {o: [r for r in rows if r[4] == o] for o in ("ran", "error", "skipped", "deferred")}
    rpt = Report(
        "Dispatcher",
        kicker="lmsa · dispatch",
        subtitle=f"Load-aware dispatch log for {day.isoformat()}.",
        sticker=f"{len(by['skipped'])} skipped" if by["skipped"] else ("errors" if by["error"] else "on schedule"),
        sticker_tone="hot" if by["skipped"] or by["error"] else "acc",
    )
    rpt.tiles([
        Tile("Ran", len(by["ran"]), tone="acc"),
        Tile("Errors", len(by["error"]), tone="hot" if by["error"] else "muted"),
        Tile("Skipped at deadline", len(by["skipped"]), tone="hot" if by["skipped"] else "muted"),
        Tile("Deferrals", len(by["deferred"]), tone="warn" if by["deferred"] else "muted"),
    ])
    latest = "; ".join(res.busy) if res.busy else "quiet"
    rpt.section("Latest check", tone="warn" if res.busy or res.watchdog else "acc").callout(
        f"{res.now.astimezone(tz):%H:%M} ({'overnight' if res.overnight else 'daytime'}): {latest}"
        + (f"\nwatchdog stopped the batch: {res.watchdog}" if res.watchdog else ""),
        label="gate", tone="warn" if res.busy or res.watchdog else "acc",
    )
    if by["skipped"]:
        rpt.section("Skipped at deadline", tone="hot").table(
            ["Task", "Due", "Reason"], [[r[0], _local_hm(r[1], tz), r[5] or ""] for r in by["skipped"]]
        )
    if by["error"]:
        rpt.section("Errors", tone="hot").table(
            ["Task", "Due", "Started", "Reason"],
            [[r[0], _local_hm(r[1], tz), _local_hm(r[2], tz), r[5] or ""] for r in by["error"]],
        )
    if by["ran"]:
        rpt.section("Ran", tone="acc").table(
            ["Task", "Due", "Started", "Finished"],
            [[r[0], _local_hm(r[1], tz), _local_hm(r[2], tz), _local_hm(r[3] or "", tz)] for r in by["ran"]],
        )
    if by["deferred"]:
        agg: dict[str, list] = {}
        for r in by["deferred"]:
            agg.setdefault(r[0], []).append(r)
        rpt.section("Deferrals", tone="warn").table(
            ["Task", "Times deferred", "Latest reason"],
            [[t, len(rs), rs[-1][5] or ""] for t, rs in sorted(agg.items())],
        )
    if res.notes:
        rpt.section("Problems", tone="warn").callout("\n".join(res.notes), label="config / cleanup", tone="warn")
    return rpt


# ---------------------------------------------------------------------------
# Lock + CLI
# ---------------------------------------------------------------------------

class AlreadyRunning(RuntimeError):
    pass


class DispatchLock:
    """Exclusive, non-blocking OS lock on a file; released if the process dies."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "DispatchLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise AlreadyRunning(f"another dispatcher holds {self.path}") from exc
        self._fh = fh
        return self

    def __exit__(self, *exc_info) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None


def print_plan(res: Result) -> None:
    print(f"dispatch dry-run at {res.now.astimezone():%Y-%m-%d %H:%M} "
          f"({'overnight' if res.overnight else 'daytime'}; input-idle threshold {res.idle_minutes:g} min)")
    print("gate: " + ("; ".join(res.busy) if res.busy else "quiet"))
    for task, model in res.would_run:
        print(f"  would run   {task}" + (f"  [load {model}]" if model else ""))
    by_reason: dict[str, list[str]] = {}
    for task, reason in res.deferred:
        by_reason.setdefault(reason, []).append(task)
    for reason, tasks in by_reason.items():
        print(f"  would defer {', '.join(tasks)}")
        print(f"    because: {reason}")
    for task, reason in res.skipped:
        print(f"  would skip  {task}: {reason}")
    if res.up_to_date:
        print("  up to date: " + ", ".join(t for t, _, _ in res.up_to_date))
    for note in res.notes:
        print(f"  problem: {note}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tasks.dispatch", description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan; load nothing, record nothing")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)

    sched = load_schedule(args.config)
    cfg = settings()
    model_names = {"chat": cfg.chat_model, "vision": cfg.vision_model}
    models = ModelManager(
        Lms(),
        free_vram_mb=lambda: gate_mod._read_gpu().free_mb,
        headroom_mb=sched.gate.vram_headroom_mb,
        ttl_s=sched.model_ttl_s,
    )

    def gate_fn(**kw) -> list[str]:
        return gate_mod.busy_reasons(sched.gate, **kw)

    if args.dry_run:
        with Store() as db:
            ensure_table(db.conn)
            res = dispatch(sched, db.conn, gate_fn=gate_fn, models=models, model_names=model_names, dry_run=True)
        print_plan(res)
        for label, probe in (("gpu", lambda: gate_mod._read_gpu()), ("loaded models", lambda: models.lms.loaded())):
            try:
                print(f"{label}: {probe()}")
            except Exception as exc:
                print(f"{label}: can't read ({exc})")
        return

    lock_path = Path(os.getenv("LMSA_DB", str(DEFAULT_DB))).parent / "dispatch.lock"
    try:
        with DispatchLock(lock_path):
            _dispatch_and_record(sched, gate_fn, models, model_names)
    except AlreadyRunning as exc:
        print(exc)


def _dispatch_and_record(sched: Schedule, gate_fn: GateFn, models: ModelManager, model_names: dict[str, str]) -> None:
    log_dir = dispatch_log_dir()
    try:  # we hold the dispatch lock; a live task's log is fresh, and only old ones go
        prune_logs([log_dir], log_retention_days())
    except OSError:
        pass
    with Store() as db:
        ensure_table(db.conn)
        try:
            res = dispatch(
                sched, db.conn, gate_fn=gate_fn, models=models, model_names=model_names,
                on_skips=lambda r: record_skips(db, r), log_dir=log_dir,
            )
        except Exception:
            with db.record_run("dispatch"):
                raise
        if res.eventful:
            with db.record_run("dispatch") as run:
                path = _write_report(db.conn, res)
                run.report_path = path
                run.summary = summary(res)
                print(path)
        else:
            print(f"nothing ran: {len(res.deferred)} deferred, {len(res.skipped)} skipped" + (f" ({'; '.join(res.busy)})" if res.busy else ""))


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
