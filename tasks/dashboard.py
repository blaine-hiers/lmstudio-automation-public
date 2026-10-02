"""Launcher dashboard: run any task by hand from a local Afterhours page.

    python -m tasks.dashboard [--port 8765] [--no-browser]

A stdlib-only web app (`http.server`) bound to 127.0.0.1. One launcher tile
per module in `tasks/`, with that task's options, a live log console, a Stop
button, and links to the reports a run produced. The dispatcher is untouched:
this only imports from it.

This page spawns processes, so every request is checked:
  - the server binds 127.0.0.1 only;
  - `Host` must be 127.0.0.1:<port> or localhost:<port> (DNS rebinding);
  - a POST needs the per-launch token (header `X-LMSA-Token`), a JSON body,
    and an `Origin` that is absent or the dashboard's own;
  - a task name must be in the discovered catalog, and argv is built from the
    curated field table below (a list, never a shell string);
  - `/reports/` serves only files that resolve inside `output/`, sandboxed by
    CSP so a report can't script the dashboard -- the one script a report may
    run is `lmsa.report.HANDLED_JS` (allowed by hash), which fetches a
    separate per-launch token from `/api/handled/token` that only
    `POST /api/handled` accepts (check off an email a report showed, #61);
  - untrusted text (logs, docstrings, file names) is escaped server side and
    set with `textContent` client side.

`/errors` lists failed runs with their logs (`lmsa.errors`); log paths come
only from DB rows or this dashboard's runs and are confined to the two log
dirs before any filesystem call.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hmac
import json
import mimetypes
import os
import re
import secrets
import shlex
import socket
import socketserver
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
import webbrowser
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

from lmsa import capabilities as caps_data
from lmsa import errors as errfeed
from lmsa import handled as handled_store
from lmsa.capabilities_page import render_capabilities_page, render_static_report
from lmsa.config import settings
from lmsa.errors_page import nav_html, render_errors_page
from lmsa.report import COLLAPSE_JS, FONTS, HANDLED_JS_HASH, THEME, collapse_toggle_html, esc
from lmsa.config import DASHBOARD_PORT
from lmsa.store import DEFAULT_DB, Store
from lmsa.task_meta import WRITE_FLAG_NAMES, read_meta
from tasks.ask import SCOPE_CHOICES as ASK_SCOPE_CHOICES
from tasks.dispatch import CONFIG_PATH, ROOT, TASK_NAME_RE, WEEKDAYS, kill_tree, load_schedule

HOST = "127.0.0.1"
DEFAULT_PORT = DASHBOARD_PORT
# Launcher fields a link may pre-fill (`/?task=<name>&<field>=<value>`, #70).
# An explicit allowlist: every other task and field ignores the query string.
PREFILL_FIELDS: dict[str, frozenset[str]] = {"go_deep": frozenset({"--story", "--topic"})}
TOKEN_HEADER = "X-LMSA-Token"
MAX_BODY = 64 * 1024
# Best-effort ceiling for draining a request body we're refusing without
# reading it in full (bad/oversized Content-Length, chunked transfer
# encoding) -- keeps a client that declares or streams a huge body from
# tying up the handler thread just to avoid a Windows RST on close (see
# Handler._drain and do_POST/do_GET below). Real requests never come close.
DRAIN_CAP = MAX_BODY * 4
MAX_TEXT = 2000  # per field value
LOG_CHUNK = 64 * 1024  # bytes returned per poll
SKIP_MODULES = {"__init__", "dashboard", "dashboard_setup"}
MAX_RUN_HISTORY = 200  # finished runs kept in memory (live runs are never dropped)
PRUNE_EVERY_S = 24 * 3600

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

GROUPS = ("Gmail", "Other", "Diagnostics & manual", "Scheduler")
GROUP_TONES = {"Gmail": "info", "Other": "acc", "Diagnostics & manual": "violet", "Scheduler": "warn"}
# The dashboard's tabs, in tab-bar order (#100).
TABS: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("tasks", "Tasks"),
    ("inbox", "Inbox"),
    ("reports", "Reports"),
    ("setup", "Setup"),
)
# Manual-only tasks are listed in config/dashboard.toml ([manual.<task>]).
DASHBOARD_CONFIG_PATH = ROOT / "config" / "dashboard.toml"
DASHBOARD_CONFIG_KEYS = {"manual"}
MANUAL_ENTRY_KEYS = {"note"}
# The "Add to schedule" form offers only these; dashboard_setup.add_to_schedule
# accepts only these (each is a value tasks.dispatch.load_schedule accepts).
CADENCE_KINDS = ("daily", "weekly", "every")
TIME_CHOICES = tuple(f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45))
HOUR_CHOICES = ("1", "2", "3", "4", "6", "8", "12", "24")  # every N hours: N divides 24
DEADLINE_CHOICES = ("0.5", "1", "2", "3", "4", "6", "8", "12", "24", "48", "72")
SCHEDULE_MODEL_CHOICES = ("none", "chat", "vision")
TIMEOUT_CHOICES = ("", "15", "30", "60", "120", "240")  # "" = [dispatch].task_timeout_minutes
# Model role of tasks the schedule doesn't list (it only knows scheduled ones).
MODEL_HINTS = {"ask": "chat", "doc_search": "chat", "shootout": "chat"}


@dataclass(frozen=True)
class Field:
    """One launcher option. `flag` is the argparse flag (or the positional's
    dest for kind "positional"). kinds: positional | int | bool | text.
    `write` marks a flag that changes something outside the report (Gmail,
    files, the store) -- shown hot, and a run using it needs a confirm."""
    flag: str
    kind: str
    label: str = ""
    write: bool = False
    required: bool = False
    default: bool = False  # initial state of a bool checkbox
    choices: tuple[str, ...] = ()  # a text option offered as a dropdown ("" = leave it out)


def _pos(dest: str, label: str) -> Field:
    return Field(dest, "positional", label, required=True)


def _int(flag: str, label: str = "", write: bool = False) -> Field:
    return Field(flag, "int", label, write=write)


def _bool(flag: str, label: str = "", write: bool = False, default: bool = False) -> Field:
    return Field(flag, "bool", label, write=write, default=default)


def _text(flag: str, label: str = "", write: bool = False) -> Field:
    return Field(flag, "text", label, write=write)


# Curated from each task's argparse block. tests/test_dashboard.py fails if a
# flag here isn't one the task's parser actually defines.
FIELDS: dict[str, tuple[Field, ...]] = {
    "ask": (
        _pos("question", "question"), _int("--k", "top k"),
        Field("--scope", "text", "scope", choices=ASK_SCOPE_CHOICES),
        _text("--since", "since (YYYY-MM-DD)"),
    ),
    "doc_search": (_pos("query", "query"), _int("--k", "top k"), _bool("--answer", "cited answer")),
    "attachments": (_int("--limit"),),
    "doc_index": (_int("--limit", "max files"),),
    "file_sorter": (
        _int("--limit", "max files"),
        _bool("--apply", "move the files", write=True),
        _text("--undo", "undo log path", write=True),
    ),
    "followup_radar": (_int("--days", "days unanswered"),),
    "inbox_brief": (_int("--hours", "look back hours"),),
    "inbox_actions": (
        _bool("--archive-handled", "archive checked-off mail"),
        _bool("--mark-read-handled", "mark checked-off mail read"),
        _int("--limit"),
        _bool("--apply", "write to Gmail", write=True),
        _int("--undo", "undo run id", write=True),
    ),
    "inbox_index": (_int("--limit"),),
    "job_tracker": (_int("--limit"),),
    "labeler": (
        _int("--days"),
        _int("--limit"),
        _bool("--apply", "apply labels in Gmail", write=True),
        _int("--undo", "undo run id", write=True),
    ),
    "phish_check": (_int("--days"), _int("--limit")),
    "pretriage": (_int("--limit"),),
    "receipts": (_int("--limit"),),
    "reply_drafts": (
        _int("--hours", "look back hours"),
        _bool("--apply", "save drafts to Gmail", write=True),
        _text("--retry", "retry message id", write=True),
    ),
    "server_logs": (_int("--days"),),
    "shootout": (_int("--limit", "first N prompts"), _int("--max-tokens"), _bool("--no-judge", "skip the judge")),
    "unsubscribe": (_int("--days"), _int("--top")),
    "dispatch": (_bool("--dry-run", "plan only", default=True),),
}


@dataclass
class TaskInfo:
    name: str
    doc: str
    group: str
    cadence: str | None = None  # None = manual only
    model: str | None = None  # "chat" | "vision" | None
    fields: tuple[Field, ...] = ()
    # Every option string of a write flag the task's parser defines (read by
    # ast), on top of the fields' own: extra args naming one need a confirm.
    write_flags: tuple[str, ...] = ()
    derived: bool = False  # fields read from the task's argparse calls (no curated entry)
    new: bool = False  # not in schedule.toml and not marked manual-only
    warnings: tuple[str, ...] = ()  # the module couldn't be fully read (shown on the tile)
    setup: dict | None = None  # the module's SETUP block (literal values only), None if absent
    setup_warnings: tuple[str, ...] = ()
    # tasks.briefing only: config/briefings/*.toml files with no `enabled` key yet (#52)
    briefings_new: tuple[str, ...] = ()

    @property
    def tone(self) -> str:
        return GROUP_TONES[self.group]


def _group(name: str, uses_gmail: bool, manual: set[str]) -> str:
    if name == "dispatch":
        return "Scheduler"
    if name in manual:
        return "Diagnostics & manual"
    if uses_gmail:
        return "Gmail"
    return "Other"


def _group_slug(group: str) -> str:
    """`group` as a section id fragment ("Diagnostics & manual" -> "diagnostics-manual")."""
    return re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-") or "group"


def parse_dashboard_config(data: dict) -> tuple[set[str], list[str]]:
    """config/dashboard.toml (already parsed) -> (manual-only task names,
    problems). Bad entries are left out and listed in problems."""
    problems: list[str] = []
    manual: set[str] = set()
    for key in data:
        if key not in DASHBOARD_CONFIG_KEYS:
            problems.append(f"unknown key {key!r} (known: {', '.join(sorted(DASHBOARD_CONFIG_KEYS))})")
    table = data.get("manual", {})
    if not isinstance(table, dict):
        return manual, [*problems, "manual must be a table of [manual.<task>] entries"]
    for name, entry in table.items():
        if not TASK_NAME_RE.fullmatch(name):
            problems.append(f"manual.{name}: not a valid task module name")
            continue
        if not isinstance(entry, dict):
            problems.append(f"manual.{name}: must be a table")
            continue
        extra = set(entry) - MANUAL_ENTRY_KEYS
        if extra:
            problems.append(f"manual.{name}: unknown key(s) {', '.join(sorted(extra))}")
            continue
        if not isinstance(entry.get("note", ""), str):
            problems.append(f"manual.{name}: note must be a string")
            continue
        manual.add(name)
    return manual, problems


def load_manual(path: Path | None) -> set[str]:
    """Manual-only task names from config/dashboard.toml. A missing or
    unreadable file means none (the Setup page's config check flags a bad one)."""
    if path is None:
        return set()
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return parse_dashboard_config(data)[0]


def listed_in_schedule(schedule_path: Path | None) -> set[str] | None:
    """Every [tasks.<name>] table in schedule.toml, enabled or not -- or None
    when the file can't be read (then no task is flagged NEW)."""
    if schedule_path is None:
        return None
    try:
        tasks = tomllib.loads(Path(schedule_path).read_text(encoding="utf-8")).get("tasks", {})
    except (OSError, ValueError):
        return None
    return set(tasks) if isinstance(tasks, dict) else None


def briefing_files(briefings_dir: Path) -> tuple[list[str], list[str], str | None]:
    """(every briefing name, the ones whose *effective* on/off state is
    still unset, and a warning message if config/local/briefings.toml is
    malformed -- #95) in `briefings_dir` -- read with tomllib only, never
    the engine. An unreadable briefing file is listed but never "new" (fix
    it in the file first). Effective state honours
    config/local/briefings.toml the same way
    `lmsa.briefing.config.effective_enabled` does, so a local override
    clears the NEW badge without touching the shipped file. The local
    overrides file is read once here (not once per briefing), and a
    malformed one is surfaced as the third element instead of only going to
    stderr (#95 review finding 2), so the tile can show it."""
    from lmsa.briefing.config import effective_enabled, local_overrides_path, local_overrides_status, \
        local_overrides_warning, names

    all_names, new = names(briefings_dir), []
    overrides, reason = local_overrides_status(local_overrides_path(briefings_dir))
    warning = local_overrides_warning(reason) if reason else None
    for name in all_names:
        try:
            data = tomllib.loads((briefings_dir / f"{name}.toml").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        b = data.get("briefing")
        if not isinstance(b, dict):
            continue
        shipped = b.get("enabled")
        shipped = shipped if isinstance(shipped, bool) else None
        if effective_enabled(shipped, name, overrides=overrides) is None:
            new.append(name)
    return all_names, new, warning


def derived_fields(args) -> tuple[Field, ...]:
    """Launcher fields from a task's argparse calls (`lmsa.task_meta.Arg`)."""
    return tuple(Field(a.flag, a.kind, a.label, write=a.write, required=a.required) for a in args)


def discover_tasks(tasks_dir: Path, schedule_path: Path | None = CONFIG_PATH,
                   dashboard_path: Path | None = DASHBOARD_CONFIG_PATH) -> list[TaskInfo]:
    """Every runnable module in `tasks_dir`, sorted by group then name.

    Modules are read with `ast` (lmsa.task_meta), never imported. Options
    come from the curated FIELDS entry when the task has one, else from its
    argparse calls. A task neither in schedule.toml nor marked manual-only
    in dashboard.toml is NEW."""
    scheduled: dict[str, tuple[str, str | None]] = {}
    if schedule_path is not None:
        try:
            for spec in load_schedule(schedule_path, tasks_dir).tasks:
                scheduled[spec.name] = (str(spec.cadence), spec.model)
        except (OSError, ValueError):
            pass  # unreadable schedule: every task shows "manual only"
    listed = listed_in_schedule(schedule_path)
    manual = load_manual(dashboard_path)
    tasks = []
    for path in sorted(tasks_dir.glob("*.py")):
        name = path.stem
        if name in SKIP_MODULES or not TASK_NAME_RE.fullmatch(name):
            continue
        meta = read_meta(path)
        setup_model = (meta.setup or {}).get("model")
        hint = MODEL_HINTS.get(name) or (setup_model if setup_model in ("chat", "vision") else None)
        cadence, model = scheduled.get(name, (None, hint))
        curated = FIELDS.get(name)
        fields = curated if curated is not None else derived_fields(meta.args)
        briefings_new: tuple[str, ...] = ()
        briefing_warning: list[str] = []
        if name == "briefing":  # --name choices + onboarding for new briefing files (#52)
            config_dir = Path(schedule_path).parent if schedule_path is not None else tasks_dir.parent / "config"
            all_b, new_b, local_warning = briefing_files(config_dir / "briefings")
            fields = tuple(replace(f, choices=tuple(all_b)) if f.flag == "--name" and f.kind == "text" else f
                           for f in fields)
            briefings_new = tuple(new_b)
            if local_warning:
                briefing_warning = [local_warning]
        tasks.append(TaskInfo(
            name=name,
            doc=meta.doc,
            group=_group(name, meta.uses_gmail, manual),
            cadence=cadence,
            model=model,
            fields=fields,
            write_flags=meta.write_options,
            derived=curated is None and bool(fields),
            new=listed is not None and name not in listed and name not in manual and name != "dispatch",
            warnings=tuple(meta.warnings + (meta.arg_warnings if curated is None else []) + briefing_warning),
            setup=meta.setup,
            setup_warnings=tuple(meta.setup_warnings),
            briefings_new=briefings_new,
        ))
    tasks.sort(key=lambda t: (GROUPS.index(t.group), t.name))
    return tasks


# ---------------------------------------------------------------------------
# Building argv
# ---------------------------------------------------------------------------

class BadRequest(ValueError):
    """The request can't be turned into a run (HTTP 400)."""


class NeedsConfirm(ValueError):
    """The run would write; the client must resend with confirm (HTTP 428)."""


class Conflict(RuntimeError):
    """The task already has a live run (HTTP 409)."""


def split_extra(text: str) -> list[str]:
    """Split free-form extra args, Windows-aware.

    POSIX `shlex.split` eats backslashes (C:\\Users -> C:Users) and
    `posix=False` keeps the quote characters inside the argument, so neither
    fits a Windows path. This is POSIX quoting with the escape character
    turned off: quotes group, backslashes stay literal.
    """
    lex = shlex.shlex(text, posix=True)
    lex.whitespace_split = True
    lex.escape = ""
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError as exc:  # "No closing quotation"
        raise BadRequest(f"extra args: {exc}") from exc


def _names_write_flag(token: str, write_flags: list[str]) -> bool:
    """True if `token` would select a write flag -- argparse accepts any
    unambiguous prefix (`--app` for `--apply`) and `--flag=value`. A short
    write flag (`-a`) also counts bundled (`-xa`) or with an attached value
    (`-a3`), so any short token containing its letter is treated as one."""
    head = token.split("=", 1)[0]
    if token.startswith("--"):
        if len(token) < 3:
            return False
        return any(flag.startswith(head) for flag in write_flags if flag.startswith("--"))
    if token.startswith("-") and len(token) > 1:
        # A single-dash long option (`-purge`) matches like a long one:
        # exactly, by an unambiguous prefix (`-pur`, even `-p`), or `-purge=x`.
        if any(flag.startswith(head) for flag in write_flags if not flag.startswith("--") and len(flag) > 2):
            return True
        shorts = {flag[1] for flag in write_flags if not flag.startswith("--") and len(flag) == 2}
        return any(c in shorts for c in token[1:])
    return False


def build_args(task: TaskInfo, values: dict, extra: str = "", *, confirm: bool = False) -> list[str]:
    """The task's CLI arguments (after `-m tasks.<name>`) from form values.

    Raises BadRequest for anything malformed or unknown, and NeedsConfirm when
    a write flag is in play (a field or the extra args) without `confirm`."""
    if not isinstance(values, dict):
        raise BadRequest("fields must be an object")
    known = {f.flag: f for f in task.fields}
    unknown = set(values) - set(known)
    if unknown:
        raise BadRequest(f"unknown option(s) for {task.name}: {', '.join(sorted(map(str, unknown)))}")
    args: list[str] = []
    positional: list[str] = []
    writes: list[str] = []
    for f in task.fields:
        v = values.get(f.flag)
        if f.kind == "bool":
            if v not in (None, False, True):
                raise BadRequest(f"{f.flag} must be true or false")
            if v:
                args.append(f.flag)
                if f.write:
                    writes.append(f.flag)
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            if f.required:
                raise BadRequest(f"{task.name} needs a {f.label or f.flag}")
            continue
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            raise BadRequest(f"{f.flag} must be text")
        text = str(v).strip()
        if len(text) > MAX_TEXT or "\x00" in text:
            raise BadRequest(f"{f.flag} is too long or contains a NUL")
        if f.kind == "int":
            if not re.fullmatch(r"-?\d{1,9}", text):
                raise BadRequest(f"{f.flag} must be a whole number")
        if f.kind == "positional":
            positional.append(text)
        else:
            args += [f.flag, text]
        if f.write:
            writes.append(f.flag)

    if not isinstance(extra, str) or len(extra) > MAX_TEXT or "\x00" in extra:
        raise BadRequest("extra args must be short text")
    extra_args = split_extra(extra)
    # Fail safe: the standard write names always count in extra args, whatever
    # the parser scan found -- or if the module couldn't be read at all.
    write_flags = [f.flag for f in task.fields if f.write] + list(task.write_flags) + sorted(WRITE_FLAG_NAMES)
    writes += [t for t in extra_args if _names_write_flag(t, write_flags)]
    args += extra_args
    if positional:
        args += ["--", *positional]  # a question starting with "-" stays a question
    if writes and not confirm:
        raise NeedsConfirm(f"this run writes ({', '.join(writes)}); confirm to launch")
    return args


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@dataclass
class Run:
    id: str
    task: str
    argv: list[str]
    log_path: Path
    started: float  # time.time()
    model: str | None
    proc: subprocess.Popen | None = None
    finished: float | None = None
    returncode: int | None = None
    stopped: bool = False
    report: Path | None = None
    pid: int | None = None
    created: int | None = None  # process_identity(pid) at launch -- guards PID reuse
    adopted: bool = False  # started by an earlier dashboard process; exit code unknowable
    overlapped: bool = False  # another run was live at some point during this one
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def live(self) -> bool:
        return not self.done.is_set()


def _utf8_prefix(data: bytes) -> bytes:
    """Drop a multi-byte character cut off at the end of a log chunk; the
    next poll starts at its first byte and reads it whole."""
    for cut in range(0, 4):
        chunk = data[: len(data) - cut] if cut else data
        try:
            chunk.decode("utf-8")
            return chunk
        except UnicodeDecodeError as exc:
            if exc.end != len(chunk):  # invalid in the middle: not a cut
                return data
    return data


STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _win_identity(pid: int) -> int | None:
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # private instance: argtypes stay local
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != STILL_ACTIVE:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        k32.CloseHandle(handle)


def process_identity(pid: object) -> int | None:
    """The creation time of running process `pid` (an opaque int), or None if
    no such process is running. A recorded pid only still names the same
    process when this matches the value recorded at launch (PIDs are reused)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        try:
            return _win_identity(pid)
        except OSError:
            return None
    try:
        os.kill(pid, 0)
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


class _PidProc:
    """Just enough of Popen for `kill_tree`: a process known only by pid (a
    run adopted after a restart). `kill()` re-checks the identity first."""

    def __init__(self, pid: int, created: int | None):
        self.pid, self.created = pid, created

    def kill(self) -> None:
        if self.created is not None and process_identity(self.pid) == self.created:
            try:
                os.kill(self.pid, 9)  # TerminateProcess on Windows
            except OSError:
                pass


class Dashboard:
    """Catalog, runs and status for one dashboard process. Everything that
    touches the machine (LM Studio, `lms`, the gate) is an injectable callable."""

    def __init__(
        self,
        root: Path = ROOT,
        *,
        output_dir: Path | None = None,
        log_dir: Path | None = None,
        db_path: Path | None = None,
        schedule_path: Path | None = CONFIG_PATH,
        dashboard_path: Path | None = None,
        python: str = sys.executable,
        probe_server: Callable[[], bool] | None = None,
        loaded_models: Callable[[], list[str]] | None = None,
        busy: Callable[[], list[str]] | None = None,
        token: str | None = None,
    ):
        self.root = Path(root)
        self.tasks_dir = self.root / "tasks"
        self.output_dir = Path(output_dir or settings().output_dir)
        self.db_path = Path(db_path or os.getenv("LMSA_DB", str(DEFAULT_DB)))
        self.log_dir = Path(log_dir or self.db_path.parent / "dashboard" / "logs")
        self.schedule_path = schedule_path
        self.dashboard_path = Path(dashboard_path) if dashboard_path else self.root / "config" / "dashboard.toml"
        self.python = python
        self.probe_server = probe_server or _probe_server
        self.loaded_models = loaded_models or _loaded_models
        self.busy = busy or (lambda: _busy(schedule_path))
        self.token = token or secrets.token_urlsafe(32)
        # Narrow per-launch token for report pages (#61): accepted only by
        # POST /api/handled, so a report can't launch tasks with it.
        self.handled_token = secrets.token_urlsafe(32)
        self.live_dir = self.log_dir.parent / "live"
        self.dispatch_log_dir = self.db_path.parent / "dispatch" / "logs"
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._pruned_at = 0.0
        self._adopt_orphans()
        self.prune_logs()
        from tasks.dashboard_setup import Setup  # it imports this module's errors

        self.setup = Setup(self)

    # -- catalog ------------------------------------------------------------

    def catalog(self) -> list[TaskInfo]:
        return discover_tasks(self.tasks_dir, self.schedule_path, self.dashboard_path)

    def task(self, name: object) -> TaskInfo:
        """The catalog entry for `name` -- the allowlist every launch goes through."""
        if not isinstance(name, str) or not TASK_NAME_RE.fullmatch(name):
            raise BadRequest("unknown task")
        for t in self.catalog():
            if t.name == name:
                return t
        raise BadRequest("unknown task")

    def last_runs(self) -> dict[str, dict]:
        """Newest `runs` row per task (recorded as `tasks.x` or `x`)."""
        out: dict[str, dict] = {}
        try:
            with Store(self.db_path) as db:
                rows = db.conn.execute(
                    "SELECT id, task, started, finished, status, report_path FROM runs ORDER BY id"
                ).fetchall()
        except sqlite3.Error:
            return out
        for _id, task, started, finished, status, report in rows:
            name = task[len("tasks."):] if task.startswith("tasks.") else task
            out[name] = {"started": started, "finished": finished, "status": status, "report": report}
        return out

    def report_url(self, path: str | Path | None) -> str | None:
        """`/reports/<name>` for a file inside output/, else None.

        `path` comes from the `runs` table, so it is checked lexically first
        (like `resolve_report`): only a path that already sits under output/
        -- relative ones are taken from the repo root, the tasks' cwd -- is
        ever resolved, so a UNC value never reaches the filesystem (SMB)."""
        if not path:
            return None
        text = str(path)
        if "\x00" in text:
            return None
        try:
            if not os.path.isabs(text):
                text = os.path.join(self.root, text)
            full = os.path.normcase(os.path.abspath(text))  # lexical: no I/O
            out = os.path.normcase(os.path.abspath(self.output_dir))
            if not full.startswith(out.rstrip("\\/") + os.sep):
                return None
            p = Path(text).resolve()  # now safe: follows local links only
            rel = p.relative_to(self.output_dir.resolve())
        except (OSError, ValueError):
            return None
        return "/reports/" + quote(rel.as_posix()) if p.is_file() else None

    def resolve_report(self, rel: str) -> Path | None:
        """The file under output/ that `/reports/<rel>` names, or None."""
        rel = unquote(rel)
        if "\x00" in rel:
            return None
        # Refuse absolute, drive and UNC forms before touching the filesystem:
        # `out / "//host/share/x"` is a UNC path, and resolve() on it opens an
        # SMB connection (leaking NTLM) before the containment check can run.
        if ":" in rel or rel.startswith(("/", "\\")) or PureWindowsPath(rel).anchor:
            return None
        out = self.output_dir.resolve()
        try:
            p = (out / (rel or "index.html")).resolve()
        except (OSError, ValueError):
            return None
        if not p.is_relative_to(out) or not p.is_file():
            return None
        return p

    # -- runs ---------------------------------------------------------------
    #
    # Each live run also has a marker file, data/dashboard/live/<task>.json
    # (pid, process creation time, log, started). It keeps "one live run per
    # task" true across dashboard processes, and lets a restarted dashboard
    # adopt runs an earlier one started. A marker only counts while its pid
    # is alive AND has the recorded creation time (PIDs get reused).

    def _marker(self, task: str) -> Path:
        return self.live_dir / f"{task}.json"

    @contextlib.contextmanager
    def _markers_locked(self, timeout: float = 10.0):
        """Hold the cross-process lock that every marker read/write/delete
        runs under. Windows can't replace or delete a file another process
        has open, and a check-then-unlink of a stale marker must not race a
        second dashboard's claim; one lock around all of it rules out both.
        The OS drops the lock if this process dies. Raises TimeoutError
        (an OSError) if another dashboard holds it past `timeout`."""
        self.live_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.live_dir / ".lock", "a+")
        try:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fh.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"another dashboard holds {self.live_dir / '.lock'}")
                    time.sleep(0.02)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):  # closing the handle releases it anyway
                    fh.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    def _live_marker(self, task: str) -> dict | None:
        """The task's marker if it names a running process; a stale or
        unreadable marker is deleted and ignored."""
        path = self._marker(task)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("created") is not None \
                and process_identity(data.get("pid")) == data.get("created"):
            return data
        try:
            path.unlink()
        except OSError:
            pass
        return None

    def _write_marker(self, task: str, data: dict) -> None:
        tmp = self.live_dir / f".{task}.{secrets.token_hex(4)}.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self._marker(task))

    def _claim(self, task: str) -> None:
        """Create the task's marker exclusively, owned by this dashboard until
        the child's pid replaces it -- a second dashboard starting the same
        task at the same moment gets a Conflict, not a second writer.
        Caller holds `_markers_locked`."""
        m = self._live_marker(task)
        if m is not None:
            if not (m.get("placeholder") and m.get("pid") == os.getpid()):
                raise Conflict(f"{task} is already running (started by another dashboard)")
            # Our own placeholder: `self._lock` is held, so no start of ours is in
            # flight -- it is a leftover from a failed start whose cleanup failed.
            with contextlib.suppress(FileNotFoundError):
                self._marker(task).unlink()
        try:
            fd = os.open(self._marker(task), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Conflict(f"{task} is already starting elsewhere") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"task": task, "pid": os.getpid(), "created": process_identity(os.getpid()),
                       "placeholder": True}, f)

    def _release(self, run: Run) -> None:
        """Delete the run's marker, if it is still this run's."""
        path = self._marker(run.task)
        try:
            with self._markers_locked():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("pid") == run.pid and data.get("created") == run.created:
                    path.unlink()
        except (OSError, ValueError, AttributeError):
            pass  # a leftover marker is harmless: it goes stale when the pid dies

    def _other_live_elsewhere(self, task: str) -> bool:
        """Is another task's run live in some other dashboard process?
        Caller holds `_markers_locked`."""
        if not self.live_dir.is_dir():
            return False
        known = {r.task for r in self.runs.values() if r.live}
        for m in self.live_dir.glob("*.json"):
            if m.stem != task and m.stem not in known and self._live_marker(m.stem) is not None:
                return True
        return False

    def _adopt_orphans(self) -> None:
        """Take over runs an earlier dashboard started that are still going:
        they show as running, stream their log and can be stopped."""
        if not self.live_dir.is_dir():
            return
        try:
            with self._markers_locked():
                self._adopt_orphans_locked()
        except OSError:
            pass  # another dashboard is busy with markers; adoption is best-effort

    def _adopt_orphans_locked(self) -> None:
        log_root = os.path.normcase(os.path.abspath(self.log_dir)) + os.sep
        for path in sorted(self.live_dir.glob("*.json")):
            task = path.stem
            if not TASK_NAME_RE.fullmatch(task):
                continue
            m = self._live_marker(task)
            if m is None or m.get("placeholder"):
                continue
            log = str(m.get("log") or "")
            run_id = str(m.get("run_id") or "")
            if not re.fullmatch(r"[0-9a-f]{1,32}", run_id) or run_id in self.runs \
                    or not os.path.normcase(os.path.abspath(log)).startswith(log_root):
                continue
            argv = m.get("argv") if isinstance(m.get("argv"), list) else ["python", "-m", f"tasks.{task}"]
            started = m.get("started") if isinstance(m.get("started"), (int, float)) else time.time()
            run = Run(run_id, task, [str(a) for a in argv], Path(log), float(started), m.get("model"),
                      pid=m["pid"], created=m["created"], adopted=True, overlapped=True)
            self.runs[run_id] = run
            threading.Thread(target=self._watch_adopted, args=(run,), daemon=True).start()

    def _watch_adopted(self, run: Run, interval: float = 0.5) -> None:
        while process_identity(run.pid) == run.created:
            time.sleep(interval)
        run.finished = time.time()  # the exit code went to the dashboard that started it
        run.report = self._find_report(run)
        self._persist(run)
        self._release(run)
        run.done.set()

    def live_runs(self) -> list[dict]:
        return [{"id": r.id, "task": r.task} for r in self.runs.values() if r.live]

    def start(self, name: object, values: dict | None = None, extra: str = "", *, confirm: bool = False) -> tuple[Run, str | None]:
        """Launch a task; -> (run, soft warning or None)."""
        task = self.task(name)
        args = build_args(task, values or {}, extra, confirm=confirm)
        argv = [self.python, "-m", f"tasks.{task.name}", *args]
        with self._lock:
            live = [r for r in self.runs.values() if r.live]
            if any(r.task == task.name for r in live):
                raise Conflict(f"{task.name} is already running")
            with self._markers_locked():
                self._claim(task.name)
                try:
                    warning = None
                    model_live = [r.task for r in live if r.model]
                    if task.model and model_live:
                        warning = (f"{', '.join(model_live)} is also using a model -- 10 GB of VRAM fits one, "
                                   "so this may be slow or fail to load")
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    run_id = secrets.token_hex(6)
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    run = Run(run_id, task.name, argv, self.log_dir / f"{task.name}-{stamp}-{run_id}.log",
                              time.time(), task.model)
                    # Report attribution: overlapping runs make "newest file in
                    # output/" ambiguous, for this run and for every run live now.
                    run.overlapped = bool(live) or self._other_live_elsewhere(task.name)
                    for other in live:
                        other.overlapped = True
                    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                    with open(run.log_path, "wb") as log:
                        try:
                            run.proc = subprocess.Popen(
                                argv, cwd=str(self.root), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            )
                        except (OSError, ValueError) as exc:
                            raise BadRequest(f"couldn't start {task.name}: {exc}") from exc
                    run.pid = run.proc.pid
                    run.created = process_identity(run.pid)
                    try:
                        if run.created is not None:  # None: it already exited; nothing to guard
                            self._write_marker(task.name, {
                                "task": task.name, "run_id": run_id, "pid": run.pid, "created": run.created,
                                "log": str(run.log_path), "started": run.started, "argv": argv, "model": task.model,
                            })
                        else:
                            self._marker(task.name).unlink(missing_ok=True)
                    except BaseException:
                        # Untracked, unmarked and unstoppable otherwise: don't leave it running.
                        kill_tree(run.proc)
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            run.proc.wait(timeout=10)
                        raise
                except BaseException:
                    with contextlib.suppress(OSError):  # never mask the real error
                        self._marker(task.name).unlink(missing_ok=True)
                    raise
            self.runs[run_id] = run
            self._trim_history()
        threading.Thread(target=self._wait, args=(run,), daemon=True).start()
        return run, warning

    def _trim_history(self) -> None:
        """Drop the oldest finished runs past MAX_RUN_HISTORY (their records
        stay in `dashboard_runs` and their logs on disk until pruned). Caller
        holds `self._lock`."""
        done = sorted((r for r in self.runs.values() if not r.live), key=lambda r: r.started)
        for r in done[: max(0, len(self.runs) - MAX_RUN_HISTORY)]:
            del self.runs[r.id]

    def _wait(self, run: Run) -> None:
        assert run.proc is not None
        run.returncode = run.proc.wait()
        run.finished = time.time()
        run.report = self._find_report(run)
        self._persist(run)
        self._release(run)
        run.done.set()

    def _foreign_runs_overlap(self, run: Run) -> bool:
        """Did a `runs` row of another task (the dispatcher, a terminal run)
        start or finish while this run was going?"""
        since = datetime.fromtimestamp(run.started - 1, timezone.utc).isoformat()
        until = datetime.fromtimestamp((run.finished or time.time()) + 1, timezone.utc).isoformat()
        recent = datetime.fromtimestamp(run.started - 12 * 3600, timezone.utc).isoformat()
        try:
            with Store(self.db_path) as db:
                row = db.conn.execute(
                    "SELECT 1 FROM runs WHERE task NOT IN (?, ?) AND ("
                    " (started BETWEEN ? AND ?) OR (finished BETWEEN ? AND ?)"
                    " OR (finished IS NULL AND started >= ?)) LIMIT 1",
                    (run.task, f"tasks.{run.task}", since, until, since, until, recent),
                ).fetchone()
        except sqlite3.Error:
            return True  # can't tell: don't guess
        return row is not None

    def _find_report(self, run: Run) -> Path | None:
        """The report this run produced: its `runs` row's report_path, else
        -- only when that is unambiguous -- the newest file in output/ written
        since it started. Unambiguous means: it exited 0, wasn't stopped, and
        no other run (here, in another dashboard, or in the `runs` table) was
        live at any point while it ran. Otherwise no report is credited."""
        since = datetime.fromtimestamp(run.started - 1, timezone.utc).isoformat()
        try:
            with Store(self.db_path) as db:
                row = db.conn.execute(
                    "SELECT report_path FROM runs WHERE task IN (?, ?) AND started >= ? "
                    "AND report_path IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (run.task, f"tasks.{run.task}", since),
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row and self.report_url(row[0]):
            return Path(row[0])
        if run.returncode != 0 or run.stopped or run.overlapped or self._foreign_runs_overlap(run):
            return None
        try:
            fresh = [p for p in self.output_dir.rglob("*")
                     if p.is_file() and p.name != "index.html" and p.stat().st_mtime >= run.started - 1]
        except OSError:
            return None
        return max(fresh, key=lambda p: p.stat().st_mtime, default=None)

    def stop(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.live:
            if run.proc is not None:
                run.stopped = True
                kill_tree(run.proc)
            elif run.pid is not None and process_identity(run.pid) == run.created:
                run.stopped = True
                kill_tree(_PidProc(run.pid, run.created))  # adopted: same taskkill /T /F by pid
        return run

    def run_state(self, run: Run, offset: int = 0) -> dict:
        text, new_offset = "", offset
        try:
            with open(run.log_path, "rb") as f:
                f.seek(max(offset, 0))
                data = f.read(LOG_CHUNK)
            if run.live or len(data) == LOG_CHUNK:  # more bytes may follow
                data = _utf8_prefix(data)
            new_offset = max(offset, 0) + len(data)
            text = data.decode("utf-8", errors="replace")
        except OSError:
            pass
        end = run.finished or time.time()
        more = new_offset < (run.log_path.stat().st_size if run.log_path.exists() else 0)
        return {
            "id": run.id,
            "task": run.task,
            "command": " ".join(["python", *run.argv[1:]]),
            "started": datetime.fromtimestamp(run.started).strftime("%H:%M:%S"),
            "done": not run.live and not more,
            "returncode": run.returncode,
            "stopped": run.stopped,
            "adopted": run.adopted,
            "exit_unknown": not run.live and run.returncode is None and not run.stopped,
            "duration_s": round(end - run.started, 1),
            "report": self.report_url(run.report),
            "report_name": run.report.name if run.report else None,
            "text": text,
            "offset": new_offset,
        }

    # -- errors -------------------------------------------------------------

    def log_dirs(self) -> tuple[Path, Path]:
        return self.log_dir, self.dispatch_log_dir

    @staticmethod
    def _as_local(r: Run) -> errfeed.LocalRun:
        return errfeed.LocalRun(r.id, r.task, r.started, r.finished, r.returncode, r.stopped,
                                str(r.log_path), str(r.report) if r.report else None)

    def _local_runs(self) -> list[errfeed.LocalRun]:
        return [self._as_local(r) for r in list(self.runs.values()) if not r.live]

    def _persist(self, run: Run) -> None:
        """Record a finished run in `dashboard_runs`, so the Errors view still
        has it after a restart or the history cap. Best-effort."""
        try:
            errfeed.record_dashboard_run(self.db_path, self._as_local(run), run.argv)
        except (sqlite3.Error, OSError):
            pass

    def _timeouts(self) -> dict[str, float]:
        """Per-task timeouts from the schedule (bounds an abandoned row's window)."""
        if self.schedule_path is None:
            return {}
        try:
            return {t.name: t.timeout.total_seconds() for t in load_schedule(self.schedule_path, self.tasks_dir).tasks}
        except (OSError, ValueError):
            return {}

    def prune_logs(self, force: bool = True) -> int:
        """Delete dashboard and dispatch logs older than the retention period,
        never one a live run is writing (ours, another dashboard's, or a
        scheduled task's). Throttled to once a day unless `force`."""
        if not force and time.time() - self._pruned_at < PRUNE_EVERY_S:
            return 0
        self._pruned_at = time.time()
        keep = [str(r.log_path) for r in list(self.runs.values()) if r.live]
        with contextlib.suppress(OSError):
            for m in self.live_dir.glob("*.json"):
                with contextlib.suppress(OSError, ValueError, AttributeError):
                    keep.append(str(json.loads(m.read_text(encoding="utf-8")).get("log") or ""))
        keep += errfeed.live_dispatch_logs(self.db_path)
        days = errfeed.log_retention_days()
        errfeed.prune_dashboard_runs(self.db_path, days)  # the records go with their logs
        return errfeed.prune_logs(self.log_dirs(), days, keep)

    # -- inbox / check-off (#61) ------------------------------------------------

    def set_handled(self, msg_id: object, want: object) -> dict:
        """Check an email off (or back on). Only a message some report showed
        as an email card is accepted (`lmsa.handled.is_known`)."""
        if not isinstance(msg_id, str) or not handled_store.MSG_ID_RE.fullmatch(msg_id):
            raise BadRequest("msg_id must be a Gmail message id")
        if not isinstance(want, bool):
            raise BadRequest("handled must be true or false")
        with Store(self.db_path) as db:
            if not handled_store.is_known(db.conn, msg_id):
                raise BadRequest("unknown message")
            if want:
                handled_store.mark(db.conn, msg_id)
            else:
                handled_store.unmark(db.conn, msg_id)
        return {"msg_id": msg_id, "handled": want}

    def inbox_data(self) -> dict:
        """The Inbox panel: open items from the latest brief / follow-up /
        deadlines runs, recent check-offs (for undo), filter suggestions."""
        try:
            with Store(self.db_path) as db:
                return {
                    "open": handled_store.open_items(db.conn),
                    "handled": handled_store.recent_handled(db.conn, 30),
                    "filters": [{"sender": s, "count": n} for s, n in handled_store.filter_suggestions(db.conn)],
                }
        except sqlite3.Error as exc:
            return {"open": [], "handled": [], "filters": [], "error": f"store unavailable: {exc}"}

    def filters_xml(self) -> str | None:
        """Gmail filter-import XML for the current suggestions, or None."""
        with Store(self.db_path) as db:
            senders = [s for s, _n in handled_store.filter_suggestions(db.conn)]
        return handled_store.filter_xml(senders) if senders else None

    # -- Reports (#105) ------------------------------------------------------
    #
    # The master-report configurator: a master list (add/rename/delete via
    # a structural save, enable/disable via a per-machine local override), a
    # section builder over every bundleable task and briefing config, and
    # preview/test-send. Structural saves (name/title/send_at/llm_intro/
    # max_age_hours/skip_empty/sections) go through tasks.dashboard_setup's
    # `save_master_report`, which reuses the ordinary CONFIG_FILES
    # load->validate->backup->atomic-replace path with master_report.toml
    # on its allowlist; enabled/send are per-machine and go through
    # `set_master_local` into config/local/master_report.toml instead
    # (dashboard_setup.set_briefing_enabled's pattern). `to` is never shown
    # as a real address -- v1 sends only to LMSA_IMAP_USER.

    def _masters_or_error(self) -> tuple[list, str | None]:
        """config/master_report.toml + config/local/master_report.toml,
        scoped to *this* dashboard's own root -- not tasks.master_report's
        module-level default (its own package install location), so a save
        made through this dashboard is what a read through it sees back."""
        from tasks import master_report as mr
        from tasks.dashboard_setup import local_master_report_path

        try:
            return mr.load_config(
                path=self.setup.config_path("master_report"), local_path=local_master_report_path(self.setup),
                tasks_dir=self.tasks_dir, briefings_dir=self.setup.config_dir / "briefings",
                schedule_path=self.setup.config_dir / "schedule.toml",
            ), None
        except mr.ConfigError as exc:
            return [], "; ".join(exc.errors)

    def _find_master(self, name: object):
        masters, error = self._masters_or_error()
        if error:
            raise BadRequest(f"config/master_report.toml is invalid: {error}")
        if not isinstance(name, str):
            raise BadRequest("name must be a string")
        master = next((m for m in masters if m.name == name), None)
        if master is None:
            raise BadRequest("unknown master")
        return master

    @staticmethod
    def _last_status(conn, *, task: str | None = None, briefing: str | None = None) -> str | None:
        """'ok' | 'error' | None (never run yet) -- the latest finished run,
        matching both "x" and "tasks.x" spellings for a task, same as
        tasks.master_report.source_runs."""
        if briefing is not None:
            row = conn.execute(
                "SELECT status FROM briefing_runs WHERE briefing=? AND status IN ('ok', 'error') "
                "ORDER BY started DESC LIMIT 1", (briefing,)).fetchone()
        else:
            row = conn.execute(
                "SELECT status FROM runs WHERE task IN (?, ?) AND status IN ('ok', 'error') "
                "ORDER BY started DESC LIMIT 1", (task, f"tasks.{task}")).fetchone()
        return row[0] if row else None

    def reports_catalog(self) -> list[dict]:
        """Every section a master could bundle -- bundleable tasks plus each
        briefing config -- with its last-run status and whether it's
        already scheduled (a task) or enabled (a briefing), for the section
        builder's checklist and status column.

        `bundleable_tasks()` still lists "briefing" itself (it's a real
        module, and `master_report`'s own NOT_BUNDLEABLE doesn't exclude
        it): that would be a *task*-kind row whose id collides with every
        briefing *section*'s `task == "briefing"`, so checking it can never
        mean anything sane (a bare `{task: "briefing", briefing: None}` is
        rejected by parse_config) and unchecking it would match -- and so
        drop -- every briefing section at once (#105 review). The one
        per-config briefing row below already covers it, so it's excluded
        here."""
        from lmsa.briefing import config as bconfig
        from tasks import master_report as mr

        by_name = {t.name: t for t in self.catalog()}
        briefings_dir = self.setup.config_dir / "briefings"
        out = []
        with Store(self.db_path) as db:
            for name in mr.bundleable_tasks(self.tasks_dir):
                if name == "briefing":
                    continue
                t = by_name.get(name)
                model = t.model if t and t.model in SCHEDULE_MODEL_CHOICES else "none"
                out.append({"kind": "task", "id": name, "label": name, "scheduled": t.cadence is not None if t else False,
                            "model": model, "last_status": self._last_status(db.conn, task=name)})
            for name in bconfig.names(briefings_dir):
                try:
                    enabled = bool(bconfig.load(name, briefings_dir).enabled)
                except Exception:
                    enabled = None
                out.append({"kind": "briefing", "id": name, "label": f"briefing: {name}", "scheduled": enabled,
                            "last_status": self._last_status(db.conn, briefing=name)})
        return out

    def reports_files(self, limit: int = 20) -> list[dict]:
        """The most recently written master-report-*.html files under
        output/ (preview or full-report bodies), newest first."""
        paths = sorted(self.output_dir.rglob("master-report-*.html"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for p in paths:
            url = self.report_url(p)
            if url:
                out.append({"name": p.name, "url": url})
        return out

    def reports_data(self) -> dict:
        """Every configured master, its sections and schedule status, plus
        the section-builder catalog, recent report files, and the base sha
        a structural save must send back (optimistic locking, same as the
        generic config editor's `base`)."""
        from tasks.dashboard_setup import current_sha, redact
        from tasks import master_report as mr

        masters, error = self._masters_or_error()
        now = datetime.now(timezone.utc)
        out = []
        for m in masters:
            to = ["LMSA_IMAP_USER"] if not m.to else [redact(a, a, min_len=1) for a in m.to]
            out.append({
                "name": m.name, "title": m.title, "send_at": m.send_at, "enabled": m.enabled,
                "send": m.send, "llm_intro": m.llm_intro, "max_age_hours": m.max_age_hours,
                "skip_empty": m.skip_empty, "due": mr.due(m, now), "to": to,
                "sections": [{"task": s.task, "title": s.title, "mode": s.mode,
                              "max_items": s.max_items, "briefing": s.briefing} for s in m.sections],
            })
        return {"masters": out, "error": error, "catalog": self.reports_catalog(),
                "files": self.reports_files(), "modes": list(mr.MODES),
                "base": current_sha(self.setup, "master_report")}

    def reports_preview(self, name: object) -> dict:
        """`name`'s email body from the latest data: a dry run (like
        `--preview`) that never sends and never touches the `sends` table,
        with the rendered HTML handed back inline so the caller can drop it
        straight into a sandboxed iframe."""
        from lmsa.email_render import GMAIL_CLIP
        from tasks import master_report as mr

        master = self._find_master(name)
        with Store(self.db_path) as db:
            result = mr.run_master(master, db.conn, send=False, now=datetime.now(timezone.utc))
        html = Path(result["preview"]).read_text(encoding="utf-8")
        size = len(html.encode("utf-8"))
        return {"name": master.name, "html": html, "bytes": size, "clip": GMAIL_CLIP,
                "clipped": size > GMAIL_CLIP, "report": self.report_url(result.get("report_path"))}

    def reports_test_send(self, name: object, *, confirm: bool = False) -> dict:
        """Send `name` right now, for real, bypassing the once-a-day guard
        (`force=True`, tasks.master_report.run_master, #105): a deliberate,
        confirmed action that must go out even if today's scheduled send
        already happened, and must never make that scheduled send think
        today is done."""
        from tasks import master_report as mr

        master = self._find_master(name)
        if not confirm:
            raise NeedsConfirm(f"sends {master.title!r} right now, for real, to LMSA_IMAP_USER; confirm")
        with Store(self.db_path) as db:
            result = mr.run_master(master, db.conn, send=True, force=True, now=datetime.now(timezone.utc))
        if result["status"] == "failed":
            raise BadRequest(f"test send failed: {result.get('error_type', 'unknown error')}")
        return {"name": master.name, "status": result["status"]}

    def failed_tasks(self) -> set[str]:
        """Tasks whose most recent finished run failed (the tile badge)."""
        return errfeed.latest_failed(self.db_path, self._local_runs())

    def _rerun(self, task: str) -> dict | None:
        """A Re-run through /api/run with the tile's initial options, or None
        if the task isn't in the catalog."""
        try:
            t = self.task(task)
        except BadRequest:
            return None
        return {"task": t.name, "fields": {f.flag: True for f in t.fields if f.kind == "bool" and f.default}}

    @staticmethod
    def _when(ts: float) -> tuple[str, str]:
        dt = datetime.fromtimestamp(ts).astimezone()
        return dt.strftime("%b %d %H:%M"), dt.isoformat(timespec="seconds")

    def errors_data(self) -> dict:
        self.prune_logs(force=False)
        entries, total = errfeed.collect(self.db_path, self._local_runs(), self.log_dirs(), timeouts=self._timeouts())
        items, gave_total = errfeed.gave_up_items(self.db_path)
        out = []
        for e in entries:
            label, iso = self._when(e.when)
            out.append({"id": e.id, "task": e.task, "when_label": label, "when": iso,
                        "source": e.source, "outcome": e.outcome, "reason": e.reason})
        return {"entries": out, "total": total, "gave_up": {"items": items, "total": gave_total}}

    def error_detail(self, entry_id: str, full: bool = False) -> dict | None:
        """One failure with its log (tail, or up to the cap with `full`),
        report link and a Re-run spec. The log path never leaves the server."""
        if not errfeed.valid_id(entry_id):
            return None
        entries, _ = errfeed.collect(self.db_path, self._local_runs(), self.log_dirs(), limit=None,
                                     timeouts=self._timeouts())
        e = next((x for x in entries if x.id == entry_id), None)
        if e is None:
            return None
        log = errfeed.read_log(e.log_path, self.log_dirs(), full=full) if e.log_path else None
        if log is not None:
            note = ""
        elif e.source == "terminal":
            note = "ran from a terminal: no log was kept, only its summary"
        elif e.outcome == "skipped":
            note = "skipped: it never ran, so there is no output"
        elif not e.log_path:
            note = "no log was kept for this run"
        else:
            note = f"the log is gone (logs are pruned after {errfeed.log_retention_days()} days)"
        label, iso = self._when(e.when)
        return {
            "id": e.id, "task": e.task, "when_label": label, "when": iso, "source": e.source,
            "outcome": e.outcome, "reason": e.reason, "summary": errfeed.summary_lines(e.summary),
            "report": self.report_url(e.report_path), "log": log, "log_note": note,
            "rerun": self._rerun(e.task),
        }

    # -- status -------------------------------------------------------------

    def status(self) -> dict:
        def safe(fn, fallback):
            try:
                return fn(), None
            except Exception as exc:
                return fallback, f"{type(exc).__name__}: {str(exc)[:160]}"

        server, server_err = safe(self.probe_server, False)
        models, models_err = safe(self.loaded_models, [])
        busy, busy_err = safe(self.busy, [])
        if busy_err:
            busy = [f"gate: can't read ({busy_err})"]
        midnight = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            with Store(self.db_path) as db:
                today = db.conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE started >= ?",
                    (midnight.astimezone(timezone.utc).isoformat(),),
                ).fetchone()[0]
        except sqlite3.Error:
            today = None
        return {
            "server": bool(server),
            "server_error": server_err,
            "models": [str(m) for m in models],
            "models_error": models_err,
            "busy": [str(b) for b in busy],
            "runs_today": today,
            "live": [r.task for r in self.runs.values() if r.live],
        }


def _probe_server() -> bool:
    url = settings().base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 -- configured local URL
            return resp.status == 200
    except OSError:
        return False


def _loaded_models() -> list[str]:
    from lmsa.models import Lms

    return Lms(timeout=10).loaded()


def _busy(schedule_path: Path | None) -> list[str]:
    from lmsa import gate
    from lmsa.gate import GateConfig

    cfg = load_schedule(schedule_path).gate if schedule_path else GateConfig()
    return gate.busy_reasons(cfg)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _field_html(task: str, f: Field, value: str | None = None) -> str:
    fid = esc(f"f-{task}-{f.flag.strip('-')}")
    val = f' value="{esc(value)}"' if value is not None else ""  # pre-filled from a link (#70)
    write = ' data-write="1"' if f.write else ""
    badge = '<span class="badge-writes">writes</span>' if f.write else ""
    label = esc(f.label or f.flag)
    hot = " t-hot writes" if f.write else ""
    if f.kind == "bool":
        checked = " checked" if f.default else ""
        return (
            f'<label class="check{hot}" for="{fid}"><input type="checkbox" id="{fid}" '
            f'data-flag="{esc(f.flag)}"{write}{checked}><span class="flag">{esc(f.flag)}</span>'
            f'<span class="hint">{label}</span>{badge}</label>'
        )
    if f.kind == "positional":
        req = " required" if f.required else ""
        return (
            f'<label class="field wide" for="{fid}"><span class="flag">{label}</span>'
            f'<input type="text" id="{fid}" data-flag="{esc(f.flag)}"{req}{val} '
            f'maxlength="{MAX_TEXT}" placeholder="{"required" if f.required else "optional"}" autocomplete="off"></label>'
        )
    if f.kind == "text" and f.choices:
        opts = f'<option value="">{esc(f.label or "default")}</option>' + "".join(
            f'<option value="{esc(c)}">{esc(c)}</option>' for c in f.choices)
        return (
            f'<label class="field{hot}" for="{fid}"><span class="flag">{esc(f.flag)}{badge}</span>'
            f'<select id="{fid}" data-flag="{esc(f.flag)}"{write}>{opts}</select></label>'
        )
    kind = 'type="number" inputmode="numeric" step="1"' if f.kind == "int" else 'type="text" autocomplete="off"'
    return (
        f'<label class="field{hot}" for="{fid}"><span class="flag">{esc(f.flag)}{badge}</span>'
        f'<input {kind} id="{fid}" data-flag="{esc(f.flag)}"{write}{val} maxlength="{MAX_TEXT}" '
        f'placeholder="{label if f.label else "default"}"></label>'
    )


def _last_html(dash: Dashboard, last: dict | None) -> str:
    if not last:
        return '<span class="dot st-none"></span>never run'
    status = last.get("status") or "?"
    tone = {"ok": "ok", "error": "err", "running": "run"}.get(status, "none")
    when = ""
    try:
        when = datetime.fromisoformat(last["started"]).astimezone().strftime("%b %d %H:%M")
    except (TypeError, ValueError):
        pass
    url = dash.report_url(last.get("report"))
    link = f' · <a href="{esc(url)}" target="_blank" rel="noopener">report</a>' if url else ""
    return f'<span class="dot st-{tone}"></span>{esc(status)} · {esc(when)}{link}'


def _select_html(task: str, name: str, label: str, options, selected: str, show_for: str = "") -> str:
    sid = esc(f"o-{task}-{name}")
    opts = "".join(
        f'<option value="{esc(o)}"{" selected" if o == selected else ""}>'
        f'{esc(o or "default")}</option>' for o in options)
    data_for = f' data-for="{esc(show_for)}"' if show_for else ""
    return (f'<label class="field" for="{sid}"{data_for}><span class="flag">{esc(label)}</span>'
            f'<select id="{sid}" name="{esc(name)}">{opts}</select></label>')


def _onboard_html(t: TaskInfo) -> str:
    """The NEW tile's Add-to-schedule form (every option a valid value)."""
    model = t.model if t.model in SCHEDULE_MODEL_CHOICES else "none"
    fields = "".join((
        _select_html(t.name, "kind", "cadence", CADENCE_KINDS, "daily"),
        _select_html(t.name, "day", "day", WEEKDAYS, "sun", "weekly"),
        _select_html(t.name, "time", "at", TIME_CHOICES, "04:00", "daily weekly"),
        _select_html(t.name, "hours", "every N hours", HOUR_CHOICES, "6", "every"),
        _select_html(t.name, "deadline_hours", "deadline hours", DEADLINE_CHOICES, "12"),
        _select_html(t.name, "model", "model", SCHEDULE_MODEL_CHOICES, model),
        _select_html(t.name, "timeout_minutes", "timeout min", TIMEOUT_CHOICES, ""),
    ))
    return (
        '<details class="onboard" open><summary>new tool &middot; add to schedule</summary>'
        f'<form class="onboard-form" novalidate><div class="fields">{fields}</div>'
        '<div class="launcher-actions">'
        '<button class="btn btn-sm btn-hot" type="button" data-onboard="schedule">add to schedule</button>'
        '<button class="btn btn-ghost btn-sm" type="button" data-onboard="manual">mark manual-only</button>'
        '<span class="launcher-msg" role="status"></span></div></form></details>'
    )


def _briefings_onboard_html(t: TaskInfo) -> str:
    """tasks.briefing's tile: one enable / disable form per briefing file that
    doesn't say `enabled` yet (#52). Same buttons and confirm as NEW tools."""
    rows = "".join(
        f'<form class="onboard-form" novalidate data-briefing="{esc(n)}"><div class="fields">'
        f'<span class="flag">config/briefings/{esc(n)}.toml</span></div><div class="launcher-actions">'
        '<button class="btn btn-sm btn-hot" type="button" data-onboard="briefing-enable">enable</button>'
        '<button class="btn btn-ghost btn-sm" type="button" data-onboard="briefing-disable">keep disabled</button>'
        '<span class="launcher-msg" role="status"></span></div></form>'
        for n in t.briefings_new)
    return f'<details class="onboard" open><summary>new briefings &middot; schedule them</summary>{rows}</details>'


def prefill_values(t: TaskInfo, query: dict[str, list[str]]) -> dict[str, str]:
    """Launcher values from a link's query string (`/?task=go_deep&story=...`,
    #70): only the fields `PREFILL_FIELDS` allows for this task (`story` ->
    `--story`); every other task gets nothing. Write flags, checkboxes and
    dropdowns are never pre-filled, and nothing is run -- the user still
    presses run."""
    out: dict[str, str] = {}
    permitted = PREFILL_FIELDS.get(t.name, frozenset())
    if not permitted:
        return out
    for key, vals in query.items():
        if key == "task" or not vals or not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", key):
            continue
        for f in t.fields:
            if f.flag in (key, "--" + key.replace("_", "-")):
                break
        else:
            continue
        if f.flag not in permitted:
            continue
        v = vals[0].strip()
        if f.write or f.kind not in ("text", "int", "positional") or f.choices or not v or len(v) > MAX_TEXT:
            continue
        if any(ord(c) < 32 or ord(c) == 127 for c in v):
            continue
        if f.kind == "int" and not re.fullmatch(r"-?\d{1,9}", v):
            continue
        out[f.flag] = v
    return out


def _launcher_html(dash: Dashboard, t: TaskInfo, last: dict | None, failed: bool = False,
                   values: dict[str, str] | None = None) -> str:
    cadence = (f'<span class="tag">{esc(t.cadence)}</span>' if t.cadence
               else '<span class="tag t-muted">manual only</span>')
    new = '<span class="tag t-hot">new</span>' if t.new else ""
    if t.briefings_new:
        new += f'<span class="tag t-hot">{len(t.briefings_new)} new briefing{"s" if len(t.briefings_new) != 1 else ""}</span>'
    derived = ('<span class="tag t-muted" title="options read from its argparse calls">auto options</span>'
               if t.derived else "")
    warnings = ("<ul class=\"tile-warnings\">" + "".join(f"<li>{esc(w)}</li>" for w in t.warnings) + "</ul>"
                if t.warnings else "")
    badge = (f'<a class="badge-err" href="/errors?task={quote(t.name)}" title="its last run failed">error</a>'
             if failed else "")
    model = f'<span class="tag t-violet">{esc(t.model)}</span>' if t.model else ""
    values = values or {}
    fields = "".join(_field_html(t.name, f, values.get(f.flag)) for f in t.fields)
    has_write = any(f.write for f in t.fields)
    ribbon = '<div class="ribbon">writes</div>' if has_write else ""
    # Compact by default; Options opens on its own only when the card has
    # something the user is likely to need right now (#102): a required
    # field (e.g. ask's question), a failed last run, or a prefilled value
    # from a deep link. That's not a mere default -- it's authoritative, so
    # `data-must-open` tells OPTIONS_JS to ignore a stale saved "closed" and
    # keep the card open; a plain toggle is still remembered in localStorage
    # and wins once the card is back to a cosmetic default (closed).
    must_open = bool(values) or failed or any(f.required for f in t.fields)
    open_attrs = ' open data-must-open="1"' if must_open else ""
    options = (
        f'<details class="options"{open_attrs} data-options="{esc(t.name)}">'
        f'<summary>Options</summary><div class="fields">{fields}</div></details>'
    ) if t.fields else ""
    return (
        f'<article class="launcher t-{t.tone}" id="task-{esc(t.name)}" data-task="{esc(t.name)}">{ribbon}'
        f'<header class="launcher-head"><h3>{esc(t.name)}</h3>{badge}'
        f'<div class="tags">{new}{cadence}{model}{derived}</div></header>'
        f'<p class="launcher-doc">{esc(t.doc)}</p>{warnings}'
        f'<div class="launcher-last">{_last_html(dash, last)}</div>'
        f'{_onboard_html(t) if t.new else ""}'
        f'{_briefings_onboard_html(t) if t.briefings_new else ""}'
        f'<form class="launcher-form" novalidate>{options}'
        f'<details class="extra"><summary>extra args</summary>'
        f'<input type="text" data-extra maxlength="{MAX_TEXT}" autocomplete="off" '
        f'placeholder="--flag value &quot;quoted arg&quot;" aria-label="extra args for {esc(t.name)}"></details>'
        f'<div class="launcher-actions"><button class="btn" type="submit">run <span aria-hidden="true">&#9656;</span></button>'
        f'<span class="launcher-msg" role="status"></span></div></form></article>'
    )


def _tabbar_html(counts: dict[str, int]) -> str:
    """The `role="tablist"` switch between panels (#100). Order and labels
    come from `TABS`, so a later Reports tab (#105) is one more entry there."""
    tabs = []
    for i, (tid, label) in enumerate(TABS):
        n = counts.get(tid, 0)
        badge = f'<span class="tab-badge" data-tab-badge="{tid}"{"" if n else " hidden"}>{n}</span>'
        tabs.append(
            f'<a href="#{tid}" id="tab-{tid}" class="tab" role="tab" aria-controls="{tid}" '
            f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{0 if i == 0 else -1}">'
            f'{esc(label)}{badge}</a>'
        )
    return '<div class="tabbar" role="tablist" aria-label="Dashboard sections">' + "".join(tabs) + "</div>"


def _task_filters_html(groups: list[str]) -> str:
    chips = "".join(
        f'<button type="button" class="chip-filter" data-group-filter="{esc(g)}" aria-pressed="true">{esc(g)}</button>'
        for g in groups
    )
    return (
        '<div class="task-filters"><div class="chip-filters" role="group" aria-label="filter by group">'
        f'{chips}</div><label class="field task-search" for="task-search"><span class="flag">search</span>'
        '<input type="search" id="task-search" placeholder="task name" autocomplete="off"></label></div>'
    )


def render_page(dash: Dashboard, nonce: str, port: int, query: dict[str, list[str]] | None = None) -> str:
    from tasks.dashboard_setup import SETUP_JS, render_section as render_setup

    tasks = dash.catalog()
    # `?task=<name>&<field>=<value>` pre-fills that task's launcher (#70); never runs it.
    want = ((query or {}).get("task") or [""])[0]
    target = next((t for t in tasks if t.name == want), None)
    prefill = prefill_values(target, query or {}) if target is not None else {}
    last = dash.last_runs()
    failed = dash.failed_tasks()
    present_groups = [g for g in GROUPS if any(t.group == g for t in tasks)]
    sections, i = [], 3
    for group in present_groups:
        members = [t for t in tasks if t.group == group]
        tiles = "".join(_launcher_html(dash, t, last.get(t.name), t.name in failed,
                                       prefill if t is target else None) for t in members)
        gid = f"group-{_group_slug(group)}"
        attention = sum(1 for t in members if t.name in failed)
        sections.append(
            f'<section class="t-{GROUP_TONES[group]}" id="{gid}" data-group="{esc(group)}">'
            f'<h2><span class="idx">{i:02d} //</span>{esc(group)}'
            f'<span class="count">{len(members)}</span>{collapse_toggle_html(gid, attention)}</h2>'
            f'<div class="sec-body" id="body-{gid}"><div class="launchers">{tiles}</div></div></section>'
        )
        i += 1
    stat = lambda key, label, tone: (  # noqa: E731
        f'<div class="tile t-{tone}" data-stat="{key}"><div class="label">{label}</div>'
        f'<div class="value">&mdash;</div><div class="note">&nbsp;</div></div>'
    )
    ticker_items = ["manual override", f"127.0.0.1:{port}", f"{len(tasks)} tasks",
                    "one model fits in 10 gb", "reads before writes"]
    ticker = "".join(f'<span class="ticker-item">{esc(t)}</span>' for t in ticker_items)
    new = [t for t in tasks if t.new]

    # Overview: attention items known synchronously (failed runs, new tools).
    # "Missing setup" is added client side once /api/setup answers (#100).
    attn_items = [
        f'<li><a href="#tasks" data-goto="task-{esc(t.name)}">{esc(t.name)}</a> is new '
        '&mdash; not scheduled or marked manual-only</li>' for t in new
    ] + [
        f'<li><a href="#tasks" data-goto="task-{esc(name)}">{esc(name)}</a> &mdash; last run failed</li>'
        for name in sorted(failed)
    ]
    attn_callout = (
        f'<div class="callout t-hot" id="attn-callout"{"" if attn_items else " hidden"}>'
        '<span class="tag">attention</span>'
        f'<ul class="attn-list" id="attn-list">{"".join(attn_items)}</ul></div>'
    )
    new_banner = ""
    if new:
        links = ", ".join(f'<a href="#tasks" data-goto="task-{esc(t.name)}">{esc(t.name)}</a>' for t in new)
        new_banner = (
            '<div class="callout t-hot new-tools"><span class="tag">new</span>'
            f'<p>{len(new)} new tool{"s" if len(new) != 1 else ""} in tasks/ '
            f'{"aren&#39;t" if len(new) != 1 else "isn&#39;t"} scheduled or marked manual-only: {links}. '
            'Add each to the schedule, or mark it manual-only, from its tile; its setup needs are in Setup.</p></div>'
        )
    prefill_banner = ""
    if prefill:
        prefill_banner = (
            '<div class="callout t-info"><span class="tag">pre-filled</span>'
            f'<p>A link filled in <a href="#task-{esc(target.name)}">{esc(target.name)}</a>: '
            + esc(", ".join(f"{k} {v}" for k, v in prefill.items()))
            + '. Check it and press run -- nothing has started.</p></div>'
        )

    tab_counts = {"overview": len(new) + len(failed), "tasks": len(failed)}
    body_prefill = f' data-prefill-task="{esc(target.name)}"' if prefill and target is not None else ""
    body_attrs = f' data-attn-base="{len(new) + len(failed)}"' + body_prefill

    overview_panel = (
        '<div id="overview" class="tabpanel" role="tabpanel" aria-labelledby="tab-overview">'
        + new_banner + attn_callout +
        '<section class="t-info" id="status-sec"><h2><span class="idx">00 //</span>Status'
        '<button class="btn btn-ghost btn-sm" type="button" id="refresh">refresh</button>'
        + collapse_toggle_html("status-sec") + '</h2>'
        '<div class="sec-body" id="body-status-sec"><div class="tiles">'
        + stat("server", "LM Studio", "acc") + stat("models", "Loaded models", "violet")
        + stat("gate", "Gate", "warn") + stat("today", "Runs today", "info")
        + '</div><div class="callout t-warn busy" id="busy" hidden><span class="tag">busy</span><ul id="busy-list"></ul></div>'
        '</div></section></div>'
    )
    tasks_panel = (
        '<div id="tasks" class="tabpanel" role="tabpanel" aria-labelledby="tab-tasks">'
        + prefill_banner + _task_filters_html(present_groups) + "".join(sections) + '</div>'
    )
    inbox_panel = (
        '<div id="inbox" class="tabpanel" role="tabpanel" aria-labelledby="tab-inbox">'
        + INBOX_SECTION_HTML + '</div>'
    )
    setup_panel = (
        '<div id="setup" class="tabpanel" role="tabpanel" aria-labelledby="tab-setup">'
        + render_setup(dash.setup, 1) + '</div>'
    )
    reports_panel = (
        '<div id="reports" class="tabpanel" role="tabpanel" aria-labelledby="tab-reports">'
        + REPORTS_SECTION_HTML + '</div>'
    )

    console_dock = (
        '<div class="console-dock" id="console-dock"><div class="dock-bar">'
        # Rendered open: TABS_JS collapses it on load (its default), so with
        # JS off the console stays reachable, as it was before the dock.
        '<button class="collapse-toggle dock-toggle" type="button" id="dock-toggle" '
        'aria-expanded="true" aria-controls="dock-body" aria-label="toggle console">'
        '<span class="chevron" aria-hidden="true">&#9662;</span></button>'
        '<span class="dock-label">Console</span><span class="console-meta" id="dock-summary">idle</span></div>'
        '<div class="dock-body" id="dock-body">'
        f'<div class="console" id="console" data-live="{esc(json.dumps(dash.live_runs()))}"><div class="console-bar">'
        '<span class="console-title" id="c-title">idle</span><span class="console-meta" id="c-meta"></span>'
        '<button class="btn btn-ghost btn-sm" type="button" id="c-clear">clear finished</button>'
        '<button class="btn btn-hot btn-sm" type="button" id="c-stop" hidden>stop</button></div>'
        '<div class="console-runs" id="c-runs"></div>'
        '<pre class="console-log" id="c-log" aria-live="polite">// pick a task below and hit run</pre>'
        '<div class="console-foot" id="c-foot"></div></div></div></div>'
    )

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark"><meta name="referrer" content="no-referrer">'
        f'<meta name="lmsa-token" content="{esc(dash.token)}">'
        f"<title>Afterhours control</title>{FONTS}<style>{THEME.read_text(encoding='utf-8')}</style></head>"
        f'<body class="has-dock"{body_attrs}>'
        f'<div class="hazard"></div><div class="ticker"><div class="ticker-track">{ticker}{ticker}</div></div>'
        '<main class="wrap">'
        '<header class="masthead"><div class="sticker t-hot">manual override</div>'
        f'<div class="kicker">lmsa · launcher</div><h1>Afterhours control</h1>{nav_html("launcher")}'
        '<p class="subtitle">Run any automation by hand, watch it live, open what it wrote. '
        'Anything that writes asks twice. Reports: <a href="/reports/index.html" target="_blank" rel="noopener">index hub</a>.</p>'
        '</header>'
        + _tabbar_html(tab_counts)
        + overview_panel + tasks_panel + inbox_panel + reports_panel + setup_panel
        + f"<footer>// afterhours control · 127.0.0.1:{port} · session token rotates every launch</footer>"
        '</main>' + console_dock +
        f'<script nonce="{esc(nonce)}">{PAGE_JS}{SETUP_JS}{INBOX_JS}{REPORTS_JS}{COLLAPSE_JS}{OPTIONS_JS}{TABS_JS}</script></body></html>'
    )


PAGE_JS = r"""
(() => {
  const TOKEN = document.querySelector('meta[name=lmsa-token]').content;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  async function api(url, body) {
    const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LMSA-Token': TOKEN }, body: JSON.stringify(body) };
    const r = await fetch(url, opt);
    const j = await r.json().catch(() => ({ error: r.statusText }));
    return { ok: r.ok, status: r.status, j };
  }
  const setStat = (key, value, note) => { const t = $(`[data-stat="${key}"]`); $('.value', t).textContent = value; $('.note', t).textContent = note || '\u00a0'; };

  async function refresh() {
    ['server', 'models', 'gate', 'today'].forEach(k => setStat(k, '\u2026', 'probing'));
    const { ok, j } = await api('/api/status').catch(() => ({ ok: false, j: {} }));
    if (!ok) { setStat('server', '?', 'status failed'); return; }
    setStat('server', j.server ? 'online' : 'offline', j.server ? 'answering /v1/models' : (j.server_error || 'not answering'));
    setStat('models', j.models.length, j.models_error ? 'lms: ' + j.models_error : (j.models.join(', ') || 'none resident'));
    setStat('gate', j.busy.length ? 'busy' : 'quiet', j.busy.length ? j.busy.length + ' reason(s)' : 'nothing in the way');
    setStat('today', j.runs_today == null ? '?' : j.runs_today, j.live.length ? 'live: ' + j.live.join(', ') : 'none live');
    const list = $('#busy-list'); list.replaceChildren(...j.busy.map(b => el('li', '', b)));
    $('#busy').hidden = !j.busy.length;
  }
  $('#refresh').addEventListener('click', refresh);

  const CLEARED_KEY = 'lmsa-cleared-runs', MAX_CLEARED = 200;
  const readCleared = () => {
    try { const v = JSON.parse(localStorage.getItem(CLEARED_KEY) || '[]'); return Array.isArray(v) ? v : []; }
    catch (e) { return []; }
  };
  const rememberCleared = ids => {
    try { localStorage.setItem(CLEARED_KEY, JSON.stringify([...readCleared(), ...ids].slice(-MAX_CLEARED))); }
    catch (e) {}
  };

  const log = $('#c-log'), runs = {}; let cur = null, timer = null;
  // The console dock's one-line summary (#100): "N running · last: task OK".
  function dockSummary() {
    const bar = $('#dock-summary'); if (!bar) return;
    const chipEls = [...document.querySelectorAll('#c-runs button')];
    const running = chipEls.filter(b => b.classList.contains('live')).length;
    const doneChip = chipEls.find(b => !b.classList.contains('live'));
    const parts = [];
    if (running) parts.push(running + ' running');
    if (doneChip) {
      const outcome = doneChip.classList.contains('ok') ? 'OK' : doneChip.classList.contains('err') ? 'FAILED' : 'unknown';
      parts.push('last: ' + doneChip.textContent + ' ' + outcome);
    }
    bar.textContent = parts.length ? parts.join(' · ') : 'idle';
  }
  function idle() {
    cur = null; clearTimeout(timer);
    $('#c-title').textContent = 'idle'; $('#c-title').className = 'console-title';
    $('#c-meta').textContent = ''; $('#c-stop').hidden = true;
    log.textContent = '// pick a task below and hit run';
    $('#c-foot').replaceChildren();
  }
  function show(id) {
    cur = id; clearTimeout(timer); runs[id].off = 0; log.textContent = '';
    document.querySelectorAll('#c-runs button').forEach(b => b.classList.toggle('on', b.dataset.id === id));
    poll();
  }
  function chip(id, task) {
    const b = el('button', 'chip live', task); b.type = 'button'; b.dataset.id = id;
    b.addEventListener('click', () => show(id)); $('#c-runs').prepend(b); runs[id] = { off: 0, chip: b, task };
    dockSummary();
  }
  // "finished" is asked of the server (not guessed from a chip's last-seen
  // class): a chip nobody has watched to completion is still clearable, and
  // a live run is never removed even if this tab's poll loop is stale.
  $('#c-clear').addEventListener('click', async () => {
    const known = Object.keys(runs);
    if (!known.length) return;
    const { ok, j } = await api('/api/runs').catch(() => ({ ok: false, j: {} }));
    const liveIds = new Set(ok ? j.live.map(r => r.id) : []);
    const finished = known.filter(id => ok ? !liveIds.has(id) : false);  // can't tell: clear nothing
    if (!finished.length) return;
    finished.forEach(id => { runs[id].chip.remove(); delete runs[id]; });
    rememberCleared(finished);
    dockSummary();
    if (finished.includes(cur)) {
      const next = Object.keys(runs)[0];
      if (next) show(next); else idle();
    }
  });
  const RESTARTED = 'the dashboard restarted \u2014 reload the page';
  const outcome = r => r.stopped ? 'stopped' : r.exit_unknown ? 'finished \u2014 exit code unknown (started before restart)' : 'exit ' + r.returncode;
  const tone = r => r.returncode === 0 ? 'ok' : r.exit_unknown ? 'unk' : 'err';
  const hub = () => { const a = el('a', '', 'index hub'); a.href = '/reports/index.html'; a.target = '_blank'; a.rel = 'noopener'; return a; };
  function foot(r) {
    const f = $('#c-foot'); f.replaceChildren();
    if (!r.done) return;
    f.append(el('span', 'tag ' + (tone(r) === 'ok' ? 't-acc' : tone(r) === 'unk' ? 't-muted' : 't-hot'), outcome(r)), el('span', 'console-meta', r.duration_s + 's'));
    if (r.report) { const a = el('a', 'btn btn-ghost btn-sm', 'open report \u25B8 ' + r.report_name); a.href = r.report; a.target = '_blank'; a.rel = 'noopener'; f.append(a); }
    else { const m = el('span', 'console-meta', 'no report credited to this run \u00B7 see the '); m.append(hub()); f.append(m); }
  }
  function gone(id) {
    clearTimeout(timer); $('#c-stop').hidden = true;
    $('#c-title').className = 'console-title'; $('#c-meta').textContent = 'run not found: ' + RESTARTED;
    if (runs[id]) runs[id].chip.className = 'chip err on';
  }
  async function poll() {
    const id = cur; if (!id) return;
    const { ok, status, j } = await api(`/api/runs/${encodeURIComponent(id)}?offset=${runs[id].off}`).catch(() => ({ ok: false, j: {} }));
    if (id !== cur) return;
    if (status === 404) return gone(id);  // a restarted dashboard never knew this id
    if (!ok) { timer = setTimeout(poll, 2000); return; }
    runs[id].off = j.offset;
    if (j.text) {
      const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
      log.append(j.text);
      if (log.textContent.length > 400000) log.textContent = log.textContent.slice(-300000);
      if (stick) log.scrollTop = log.scrollHeight;
    }
    $('#c-title').textContent = j.task; $('#c-title').className = 'console-title' + (j.done ? '' : ' live');
    $('#c-meta').textContent = j.command + ' \u00b7 ' + j.started + ' \u00b7 ' + j.duration_s + 's';
    $('#c-stop').hidden = j.done;
    foot(j);
    if (j.done) {
      runs[id].chip.className = 'chip ' + tone(j) + (id === cur ? ' on' : '');
      const last = $(`[data-task="${CSS.escape(j.task)}"] .launcher-last`);
      const head = $(`[data-task="${CSS.escape(j.task)}"] .launcher-head`);
      if (head && tone(j) !== 'unk') {
        const old = $('.badge-err', head); if (old) old.remove();
        if (tone(j) === 'err') { const b = el('a', 'badge-err', 'error'); b.href = '/errors?task=' + encodeURIComponent(j.task); $('h3', head).after(b); }
      }
      if (last) {
        last.replaceChildren(el('span', 'dot st-' + (tone(j) === 'unk' ? 'none' : tone(j))), outcome(j) + ' \u00b7 just now');
        if (j.report) { const a = el('a', '', 'report'); a.href = j.report; a.target = '_blank'; a.rel = 'noopener'; last.append(' \u00b7 ', a); }
      }
      dockSummary();
    } else timer = setTimeout(poll, 700);
  }
  $('#c-stop').addEventListener('click', async () => {
    if (!cur) return;
    const { ok, status, j } = await api(`/api/runs/${encodeURIComponent(cur)}/stop`, {});
    if (status === 403 || status === 404) $('#c-meta').textContent = RESTARTED;
    else if (!ok) $('#c-meta').textContent = j.error || 'stop failed';
  });

  document.addEventListener('lmsa:run', ev => {
    chip(ev.detail.id, ev.detail.task); show(ev.detail.id);
    if (window.lmsaTabs) window.lmsaTabs.openDock();
  });
  document.querySelectorAll('form.launcher-form').forEach(form => {
    const btn = $('button[type=submit]', form), msg = $('.launcher-msg', form), label = btn.innerHTML;
    let armed = 0;
    const disarm = () => { armed = 0; btn.classList.remove('armed'); btn.innerHTML = label; };
    form.addEventListener('submit', async ev => {
      ev.preventDefault();
      const task = form.closest('[data-task]').dataset.task, fields = {};
      for (const i of form.querySelectorAll('[data-flag]')) {
        const v = i.type === 'checkbox' ? i.checked : i.value.trim();
        if (v === false || v === '') continue;
        fields[i.dataset.flag] = v;
      }
      const miss = [...form.querySelectorAll('[required]')].find(i => !i.value.trim());
      if (miss) { msg.textContent = 'needs a ' + miss.closest('label').textContent; msg.className = 'launcher-msg err'; miss.focus(); return; }
      const confirm = armed > Date.now();
      btn.disabled = true;
      const { ok, status, j } = await api('/api/run', { task, fields, extra: $('[data-extra]', form).value, confirm }).catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
      btn.disabled = false; disarm();
      if (status === 428) {
        armed = Date.now() + 8000; btn.classList.add('armed'); btn.textContent = 'confirm \u2014 this writes';
        msg.textContent = j.error; msg.className = 'launcher-msg hot'; setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100); return;
      }
      if (status === 403) { msg.textContent = RESTARTED; msg.className = 'launcher-msg err'; return; }  // token rotated
      if (!ok) { msg.textContent = j.error || 'failed'; msg.className = 'launcher-msg err'; return; }
      msg.textContent = j.warning ? '\u26A0 ' + j.warning : 'launched'; msg.className = 'launcher-msg' + (j.warning ? ' hot' : ' ok');
      chip(j.id, task); show(j.id);
      if (window.lmsaTabs) window.lmsaTabs.openDock();
    });
  });
  document.querySelectorAll('form.onboard-form').forEach(form => {
    const task = form.closest('[data-task]').dataset.task, msg = $('.launcher-msg', form);
    const kind = form.elements.namedItem('kind');
    const sync = () => form.querySelectorAll('[data-for]').forEach(l => { l.hidden = !l.dataset.for.split(' ').includes(kind.value); });
    if (kind) { kind.addEventListener('change', sync); sync(); }
    form.addEventListener('submit', ev => ev.preventDefault());
    form.querySelectorAll('button[data-onboard]').forEach(btn => {
      const label = btn.textContent; let armed = 0;
      const disarm = () => { armed = 0; btn.classList.remove('armed'); btn.textContent = label; };
      btn.addEventListener('click', async () => {
        const fields = {};
        if (btn.dataset.onboard === 'schedule') form.querySelectorAll('select').forEach(s => { if (!s.closest('[hidden]')) fields[s.name] = s.value; });
        if (form.dataset.briefing) fields.name = form.dataset.briefing;
        const confirm = armed > Date.now();
        btn.disabled = true;
        const { ok, status, j } = await api('/api/tasks/' + btn.dataset.onboard, { task, fields, confirm }).catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
        btn.disabled = false; disarm();
        if (status === 428) {
          armed = Date.now() + 8000; btn.classList.add('armed'); btn.textContent = 'confirm \u2014 ' + label;
          msg.textContent = j.error; msg.className = 'launcher-msg hot'; setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100); return;
        }
        if (status === 403) { msg.textContent = RESTARTED; msg.className = 'launcher-msg err'; return; }
        if (!ok) { msg.textContent = j.error || 'failed'; msg.className = 'launcher-msg err'; return; }
        msg.textContent = (j.message || 'done') + ' \u00b7 reloading'; msg.className = 'launcher-msg ok';
        setTimeout(() => location.reload(), 1500);
      });
    });
  });
  window.lmsaLauncher = { api, el, $, chip, show, RESTARTED };  // for the Setup panel (dashboard_setup.SETUP_JS)
  // re-attach to runs still going (after a reload, or adopted after a restart)
  let live = [];
  try { live = JSON.parse($('#console').dataset.live || '[]'); } catch (e) { live = []; }
  const cleared = new Set(readCleared());
  live = live.filter(r => !cleared.has(r.id));  // a finished run is never live, but never trust a stale race
  live.forEach(r => chip(r.id, r.task));
  const want = (location.hash.match(/^#run-([0-9a-f]+)$/) || [])[1];  // from the Errors page's Re-run
  if (live.length) show((live.find(r => r.id === want) || live[0]).id);
  refresh();
})();
"""

# Remembers each launcher card's Options expander (open/closed) per task in
# localStorage, the same try/catch-wrapped pattern as COLLAPSE_JS (#102). A
# native <details> needs no JS at all to open or close -- this only lets a
# manual toggle override the server-rendered default on later loads. A card
# the server marked `data-must-open` (a required field, a failed last run, or
# a prefilled value) is authoritative, not a mere default: a stale saved "0"
# from before that became true is never applied, so the card a user needs to
# see right now can't be reopened by their own earlier, unrelated close.
OPTIONS_JS = r"""
(() => {
  const key = id => 'lmsa-options-' + id;
  document.querySelectorAll('details[data-options]').forEach(det => {
    const id = det.dataset.options, mustOpen = det.dataset.mustOpen === '1';
    try {
      const saved = localStorage.getItem(key(id));
      if (saved === '1') det.open = true;
      else if (saved === '0' && !mustOpen) det.open = false;
    } catch (e) {}
    det.addEventListener('toggle', () => {
      try { localStorage.setItem(key(id), det.open ? '1' : '0'); } catch (e) {}
    });
  });
})();
"""

# The Inbox panel (#61): filled by INBOX_JS from /api/inbox. Check-offs go to
# /api/handled; the bulk Gmail buttons launch tasks.inbox_actions through
# /api/run, so they get the usual confirm-twice and the live console.
INBOX_SECTION_HTML = (
    '<section class="t-warn" id="inbox-sec"><h2><span class="idx">02b //</span>Inbox'
    '<span class="count" id="inbox-count">&mdash;</span>'
    '<button class="btn btn-ghost btn-sm" type="button" id="inbox-refresh">refresh</button>'
    + collapse_toggle_html("inbox-sec") + '</h2>'
    '<div class="sec-body" id="body-inbox-sec">'
    '<p class="subtitle">Open emails from the latest inbox brief, follow-up radar and deadlines runs. '
    'Checking one off hides it from later reports; Gmail is untouched unless you use a button below '
    '(logged per run, undo with <code>inbox_actions --undo RUN_ID</code>).</p>'
    '<ul class="inbox-list" id="inbox-open"></ul>'
    '<div class="inbox-actions">'
    '<button class="btn btn-sm" type="button" data-inbox-bulk="--archive-handled">also archive in Gmail</button>'
    '<button class="btn btn-sm" type="button" data-inbox-bulk="--mark-read-handled">also mark read</button>'
    '<span class="launcher-msg" id="inbox-msg"></span></div>'
    '<div class="callout t-violet" id="inbox-filters" hidden><span class="tag">gmail filter</span>'
    '<p id="inbox-filters-text"></p><p><a href="/api/inbox/filters.xml" download="mailFilters.xml">'
    'download mailFilters.xml</a> &middot; import it in Gmail: Settings &rarr; Filters and Blocked Addresses '
    '&rarr; Import filters. Nothing is created until you do.</p></div>'
    '<details class="handled-hidden"><summary>recently checked off <span class="hint">&middot; undo</span></summary>'
    '<ul class="inbox-list" id="inbox-done"></ul></details>'
    '</div></section>'
)

INBOX_JS = r"""
(() => {
  const TOKEN = document.querySelector('meta[name=lmsa-token]').content;
  const $ = s => document.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const post = (url, body) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LMSA-Token': TOKEN }, body: JSON.stringify(body) })
    .then(async r => ({ ok: r.ok, status: r.status, j: await r.json().catch(() => ({})) }));
  const msg = $('#inbox-msg');
  function row(item, done) {
    const li = el('li', 'inbox-item' + (done ? ' done' : ''));
    const box = el('input'); box.type = 'checkbox'; box.checked = done;
    box.title = done ? 'uncheck to show it in reports again' : 'check off: hide it from later reports';
    box.addEventListener('change', async () => {
      box.disabled = true;
      const { ok, j } = await post('/api/handled', { msg_id: item.msg_id, handled: box.checked }).catch(() => ({ ok: false, j: {} }));
      if (!ok) { box.checked = !box.checked; box.disabled = false; msg.textContent = j.error || 'could not save'; msg.className = 'launcher-msg err'; return; }
      load();
    });
    const text = el('div');
    let subj = el('span', 'inbox-subject', item.subject);
    if (/^https:\/\/mail\.google\.com\//.test(item.web_url || '')) { const a = el('a', 'inbox-subject', item.subject); a.href = item.web_url; a.target = '_blank'; a.rel = 'noopener'; subj = a; }
    const meta = [item.sender, item.task].concat(done && item.gmail_action ? [item.gmail_action + ' (run ' + item.run_id + ')'] : []).filter(Boolean).join(' · ');
    text.append(subj, el('div', 'inbox-meta', meta));
    li.append(box, text, el('span', 'inbox-meta', done ? String(item.handled_at || '').slice(0, 10) : ''));
    return li;
  }
  async function load() {
    const r = await fetch('/api/inbox').then(r => r.json()).catch(() => null);
    if (!r) { msg.textContent = 'inbox unavailable'; msg.className = 'launcher-msg err'; return; }
    $('#inbox-open').replaceChildren(...(r.open.length ? r.open.map(i => row(i, false)) : [el('li', 'inbox-meta', 'nothing open — inbox zero')]));
    $('#inbox-done').replaceChildren(...r.handled.map(i => row(i, true)));
    $('#inbox-count').textContent = r.open.length;
    if (window.lmsaTabs) window.lmsaTabs.setBadge('inbox', r.open.length);
    $('#inbox-filters').hidden = !r.filters.length;
    $('#inbox-filters-text').textContent = r.filters.length
      ? 'Archived 3+ times in 30 days: ' + r.filters.map(x => x.sender + ' (' + x.count + ')').join(', ') + '. Create a Gmail filter to skip the inbox for them?'
      : '';
  }
  document.querySelectorAll('[data-inbox-bulk]').forEach(btn => {
    const label = btn.textContent; let armed = 0;
    const disarm = () => { armed = 0; btn.classList.remove('armed'); btn.textContent = label; };
    btn.addEventListener('click', async () => {
      const confirm = armed > Date.now();
      const fields = { [btn.dataset.inboxBulk]: true, '--apply': true };
      btn.disabled = true;
      const { ok, status, j } = await post('/api/run', { task: 'inbox_actions', fields, extra: '', confirm }).catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
      btn.disabled = false; disarm();
      if (status === 428) {
        armed = Date.now() + 8000; btn.classList.add('armed'); btn.textContent = 'confirm — this writes to Gmail';
        msg.textContent = j.error; msg.className = 'launcher-msg hot'; setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100); return;
      }
      if (!ok) { msg.textContent = j.error || 'failed'; msg.className = 'launcher-msg err'; return; }
      msg.textContent = 'launched — see the console; its report names the run id to undo'; msg.className = 'launcher-msg ok';
      document.dispatchEvent(new CustomEvent('lmsa:run', { detail: { id: j.id, task: 'inbox_actions' } }));
    });
  });
  $('#inbox-refresh').addEventListener('click', load);
  load();
})();
"""

# The Reports panel (#105): the master-report configurator. A master list
# (add/rename/delete, settings, a section builder over every bundleable task
# and briefing config) that batches into one structural save
# (POST /api/reports/save -> dashboard_setup.save_master_report, validated
# and written through the ordinary CONFIG_FILES path); enabled/send toggle
# immediately through POST /api/reports/local (per-machine,
# config/local/master_report.toml). Preview/test-send are unchanged from
# the read-only version of this tab.
REPORTS_SECTION_HTML = (
    '<section class="t-acc" id="reports-sec"><h2><span class="idx">02c //</span>Reports'
    '<button class="btn btn-ghost btn-sm" type="button" id="reports-refresh">refresh</button>'
    + collapse_toggle_html("reports-sec") + '</h2>'
    '<div class="sec-body" id="body-reports-sec">'
    '<p class="subtitle">Master Report configurator: combine several reports into one scheduled email. '
    '<code>enabled</code>/<code>send</code> apply immediately (per machine, '
    '<code>config/local/master_report.toml</code>); everything else is batched into one '
    '<strong>Save changes</strong> (validated, backed up, atomic). '
    'The recipient is never shown as a real address -- v1 always sends to <code>LMSA_IMAP_USER</code>.</p>'
    '<div class="callout t-hot" id="reports-error" hidden><span class="tag">config error</span>'
    '<p id="reports-error-text"></p></div>'
    '<div class="reports-list" id="reports-list"></div>'
    '<div class="reports-actions">'
    '<button class="btn btn-sm" type="button" id="reports-add">+ add master</button>'
    '<button class="btn btn-sm btn-hot" type="button" id="reports-save">save changes</button>'
    '<span class="launcher-msg" id="reports-save-msg"></span></div>'
    '<h3>Latest report files</h3><ul class="reports-files" id="reports-files"></ul>'
    '</div></section>'
)

REPORTS_JS = r"""
(() => {
  const TOKEN = document.querySelector('meta[name=lmsa-token]').content;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const post = (url, body) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LMSA-Token': TOKEN }, body: JSON.stringify(body) })
    .then(async r => ({ ok: r.ok, status: r.status, j: await r.json().catch(() => ({})) }));
  const list = $('#reports-list');
  if (!list) return;  // no Reports tab on this render
  const errBox = $('#reports-error'), errText = $('#reports-error-text');
  const filesList = $('#reports-files'), saveBtn = $('#reports-save'), addBtn = $('#reports-add');
  const saveMsg = $('#reports-save-msg');
  const fmtBytes = n => n >= 1000 ? (n / 1000).toFixed(1) + ' KB' : n + ' B';
  const MODES = ['full', 'digest', 'headline', 'link'];
  let data = null;  // last GET /api/reports response
  let masters = [];  // working copy: what "save changes" will send

  function matchesCatalog(s, c) {
    // Whether section `s` is the one catalog entry `c` would add/remove --
    // matched on kind + id, not just a name, so a task and a briefing that
    // happen to share a name can never be confused for each other.
    return c.kind === 'briefing' ? (s.task === 'briefing' && s.briefing === c.id) : (s.task === c.id);
  }

  function sectionFor(c) {
    return c.kind === 'briefing' ? { task: 'briefing', mode: 'full', max_items: 5, briefing: c.id }
                                 : { task: c.id, mode: 'full', max_items: 5, briefing: null };
  }

  // -- one-click "add to schedule" / "enable" for a catalog row -----------

  function scheduleButton(c, onDone) {
    const btn = el('button', 'btn btn-sm', c.kind === 'briefing' ? 'enable' : 'add to schedule');
    btn.type = 'button';
    let armed = 0;
    const label = btn.textContent;
    const disarm = () => { armed = 0; btn.classList.remove('armed'); btn.textContent = label; };
    btn.addEventListener('click', async () => {
      const confirm = armed > Date.now();
      const url = c.kind === 'briefing' ? '/api/tasks/briefing-enable' : '/api/tasks/schedule';
      const payload = c.kind === 'briefing'
        ? { fields: { name: c.id }, confirm }
        : { task: c.id, fields: { kind: 'daily', time: '06:00', deadline_hours: '24', model: c.model || 'none', timeout_minutes: '' }, confirm };
      btn.disabled = true;
      const { status, j } = await post(url, payload).catch(() => ({ status: 0, j: { error: 'dashboard unreachable' } }));
      btn.disabled = false;
      if (status === 428) {
        armed = Date.now() + 8000; btn.classList.add('armed'); btn.textContent = 'confirm — ' + label;
        setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100);
        return;
      }
      disarm();
      if (status === 200 || status === 201) onDone(j); else onDone(null, j && j.error);
    });
    return btn;
  }

  // -- section builder: checklist (catalog) + ordered list (m.sections) ---

  function sectionBuilder(m, rerender) {
    const wrap = el('div', 'reports-builder');
    const check = el('div', 'reports-checklist');
    (data.catalog || []).forEach(c => {
      const row = el('label', 'reports-check-row');
      const box = el('input'); box.type = 'checkbox';
      box.checked = m.sections.some(s => matchesCatalog(s, c));
      box.addEventListener('change', () => {
        if (box.checked) m.sections.push(sectionFor(c));
        else m.sections = m.sections.filter(s => !matchesCatalog(s, c));
        rerender();
      });
      row.append(box, el('span', null, c.label));
      if (c.kind === 'task') row.append(el('span', 'chip' + (c.scheduled ? ' ok' : ''), c.scheduled ? 'scheduled' : 'not scheduled'));
      else row.append(el('span', 'chip' + (c.scheduled ? ' ok' : ''), c.scheduled == null ? 'unknown' : c.scheduled ? 'enabled' : 'disabled'));
      row.append(el('span', 'chip' + (c.last_status === 'ok' ? ' ok' : c.last_status === 'error' ? ' err' : ''),
        'last: ' + (c.last_status || 'never run')));
      if (!c.scheduled) {
        const note = el('span', 'launcher-msg');
        const btn = scheduleButton(c, (j, err) => {
          note.textContent = err || (j && j.message) || 'done'; note.className = 'launcher-msg' + (err ? ' err' : ' ok');
        });
        row.append(btn, note);
      }
      check.append(row);
    });
    wrap.append(check);

    const ordered = el('ol', 'reports-sections reports-sections-edit');
    m.sections.forEach((s, i) => {
      const li = el('li', 'reports-section-row');
      li.append(el('span', null, s.task === 'briefing' ? 'briefing: ' + s.briefing : s.task));
      const modeSel = document.createElement('select');
      MODES.forEach(mo => { const o = document.createElement('option'); o.value = mo; o.textContent = mo; if (mo === s.mode) o.selected = true; modeSel.append(o); });
      modeSel.addEventListener('change', () => { s.mode = modeSel.value; });
      const maxInput = document.createElement('input');
      maxInput.type = 'number'; maxInput.min = '1'; maxInput.max = '50'; maxInput.value = String(s.max_items);
      maxInput.className = 'reports-maxitems';
      // max_items only limits digest mode; full shows the whole report, headline/link no items.
      maxInput.hidden = s.mode !== 'digest';
      modeSel.addEventListener('change', () => { maxInput.hidden = modeSel.value !== 'digest'; });
      maxInput.addEventListener('change', () => { s.max_items = parseInt(maxInput.value, 10) || 5; });
      const up = el('button', 'btn btn-ghost btn-sm', '↑'); up.type = 'button'; up.disabled = i === 0;
      const down = el('button', 'btn btn-ghost btn-sm', '↓'); down.type = 'button'; down.disabled = i === m.sections.length - 1;
      up.addEventListener('click', () => { [m.sections[i - 1], m.sections[i]] = [m.sections[i], m.sections[i - 1]]; rerender(); });
      down.addEventListener('click', () => { [m.sections[i + 1], m.sections[i]] = [m.sections[i], m.sections[i + 1]]; rerender(); });
      const remove = el('button', 'btn btn-ghost btn-sm', '×'); remove.type = 'button';
      remove.addEventListener('click', () => { m.sections.splice(i, 1); rerender(); });
      li.append(modeSel, maxInput, up, down, remove);
      ordered.append(li);
    });
    if (!m.sections.length) ordered.append(el('li', null, '(no sections selected yet)'));
    wrap.append(ordered);
    return wrap;
  }

  // -- one master's card ----------------------------------------------------

  function localToggle(label, key, m) {
    // enabled/send are per-machine (config/local/master_report.toml,
    // set_master_local): applied immediately, not batched into "save
    // changes". `send: true` turns on a REAL daily send, so this needs
    // the same armed double-click confirm as "send test now" below, not
    // an automatic confirmed retry (#105 review: a single click must
    // never be enough to turn on real sending) -- applied to `enabled`
    // too, for consistency, even though it's lower stakes. The checkbox's
    // own visual state is reverted the instant it's clicked and only set
    // for real once the write actually succeeds, so it always reflects
    // what's saved, not what's pending.
    const wrap = el('label', 'reports-check-row');
    const box = el('input'); box.type = 'checkbox'; box.checked = !!m[key];
    const msg = el('span', 'launcher-msg');
    const confirmBtn = el('button', 'btn btn-ghost btn-sm', 'confirm'); confirmBtn.type = 'button'; confirmBtn.hidden = true;
    let armed = 0, pending = null;
    function disarm() { armed = 0; pending = null; confirmBtn.hidden = true; }
    async function apply(want) {
      box.disabled = true; confirmBtn.disabled = true;
      const { status, j } = await post('/api/reports/local', { name: m.name, values: { [key]: want }, confirm: true })
        .catch(() => ({ status: 0, j: { error: 'dashboard unreachable' } }));
      box.disabled = false; confirmBtn.disabled = false;
      disarm();
      if (status !== 200) {
        box.checked = !!m[key]; msg.textContent = (j && j.error) || 'failed'; msg.className = 'launcher-msg err'; return;
      }
      m[key] = want; box.checked = want; msg.textContent = (j && j.message) || 'saved'; msg.className = 'launcher-msg ok';
    }
    box.addEventListener('change', () => {
      const want = box.checked;
      box.checked = !!m[key];  // never shows a pending value as if it were saved
      if (armed > Date.now() && pending === want) { apply(want); return; }
      pending = want; armed = Date.now() + 8000;
      confirmBtn.hidden = false; confirmBtn.textContent = `confirm: ${key} = ${want}`;
      msg.textContent = 'click confirm to apply'; msg.className = 'launcher-msg hot';
      setTimeout(() => { if (armed && armed <= Date.now()) { disarm(); msg.textContent = ''; } }, 8100);
    });
    confirmBtn.addEventListener('click', () => { if (pending !== null) apply(pending); });
    wrap.append(box, el('span', null, label), confirmBtn, msg);
    return wrap;
  }

  function card(m, rerenderAll) {
    const c = el('div', 'card reports-card');
    const head = el('div', 'reports-card-head');
    const nameInput = document.createElement('input'); nameInput.value = m.name; nameInput.className = 'reports-name';
    nameInput.placeholder = 'name'; nameInput.addEventListener('input', () => { m.name = nameInput.value; });
    const titleInput = document.createElement('input'); titleInput.value = m.title; titleInput.className = 'reports-title';
    titleInput.placeholder = 'title'; titleInput.addEventListener('input', () => { m.title = titleInput.value; });
    head.append(nameInput, titleInput);
    const del = el('button', 'btn btn-ghost btn-sm', 'delete master'); del.type = 'button';
    del.addEventListener('click', () => { masters = masters.filter(x => x !== m); rerenderAll(); });
    head.append(del);
    c.append(head);

    const settings = el('div', 'reports-settings');
    const sendAt = document.createElement('input'); sendAt.value = m.send_at; sendAt.placeholder = 'HH:MM';
    sendAt.pattern = '^([01]\\d|2[0-3]):[0-5]\\d$'; sendAt.className = 'reports-sendat';
    sendAt.addEventListener('input', () => { m.send_at = sendAt.value; });
    settings.append(el('span', null, 'send_at'), sendAt);
    const llm = el('label', 'reports-check-row'); const llmBox = el('input'); llmBox.type = 'checkbox'; llmBox.checked = !!m.llm_intro;
    llmBox.addEventListener('change', () => { m.llm_intro = llmBox.checked; });
    llm.append(llmBox, el('span', null, 'llm intro'));
    settings.append(llm);
    const skip = el('label', 'reports-check-row'); const skipBox = el('input'); skipBox.type = 'checkbox'; skipBox.checked = !!m.skip_empty;
    skipBox.addEventListener('change', () => { m.skip_empty = skipBox.checked; });
    skip.append(skipBox, el('span', null, 'skip empty sections'));
    settings.append(skip);
    const age = document.createElement('input'); age.type = 'number'; age.min = '1'; age.max = '720'; age.value = String(m.max_age_hours);
    age.className = 'reports-maxage'; age.addEventListener('change', () => { m.max_age_hours = parseFloat(age.value) || 24; });
    settings.append(el('span', null, 'max age (h)'), age);
    c.append(settings);

    const local = el('div', 'reports-local');
    local.append(localToggle('enabled (this machine)', 'enabled', m));
    local.append(localToggle('send (this machine, real send)', 'send', m));
    local.append(el('span', 'reports-meta', 'to: LMSA_IMAP_USER'));
    c.append(local);

    c.append(sectionBuilder(m, rerenderAll));

    const msg = el('span', 'launcher-msg');
    const previewBtn = el('button', 'btn btn-sm', 'preview'); previewBtn.type = 'button';
    const sendBtn = el('button', 'btn btn-sm btn-hot', 'send test now'); sendBtn.type = 'button';
    const actions = el('div', 'reports-actions');
    actions.append(previewBtn, sendBtn, msg);
    c.append(actions);
    const previewBox = el('div', 'reports-preview'); previewBox.hidden = true;
    c.append(previewBox);

    previewBtn.addEventListener('click', async () => {
      previewBtn.disabled = true;
      msg.textContent = 'rendering…'; msg.className = 'launcher-msg';
      const { ok, status, j } = await post('/api/reports/preview', { name: m.name })
        .catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
      previewBtn.disabled = false;
      if (status === 403) { msg.textContent = 'session expired -- reload the page'; msg.className = 'launcher-msg err'; return; }
      if (!ok) { msg.textContent = j.error || 'preview failed (save changes first?)'; msg.className = 'launcher-msg err'; return; }
      msg.textContent = fmtBytes(j.bytes) + ' body'
        + (j.clipped ? ' — over the ' + fmtBytes(j.clip) + ' Gmail clip: Gmail shows "View entire message"' : '');
      msg.className = 'launcher-msg' + (j.clipped ? ' hot' : ' ok');
      previewBox.textContent = '';
      const frame = document.createElement('iframe');
      frame.setAttribute('sandbox', '');  // no allow-scripts, no allow-same-origin: render only
      frame.className = 'reports-frame';
      previewBox.append(frame);
      previewBox.hidden = false;
      frame.srcdoc = j.html;
    });

    let armed = 0;
    const disarm = () => { armed = 0; sendBtn.classList.remove('armed'); sendBtn.textContent = 'send test now'; };
    sendBtn.addEventListener('click', async () => {
      const confirm = armed > Date.now();
      sendBtn.disabled = true;
      const { ok, status, j } = await post('/api/reports/test-send', { name: m.name, confirm })
        .catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
      sendBtn.disabled = false;
      if (status === 428) {
        armed = Date.now() + 8000; sendBtn.classList.add('armed'); sendBtn.textContent = 'confirm — send test now';
        msg.textContent = j.error; msg.className = 'launcher-msg hot';
        setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100);
        return;
      }
      disarm();
      if (status === 403) { msg.textContent = 'session expired -- reload the page'; msg.className = 'launcher-msg err'; return; }
      if (!ok) { msg.textContent = j.error || 'test send failed'; msg.className = 'launcher-msg err'; return; }
      msg.textContent = 'test send: ' + j.status; msg.className = 'launcher-msg ok';
    });
    return c;
  }

  function renderMasters() {
    list.textContent = '';
    masters.forEach(m => list.append(card(m, renderMasters)));
    if (!masters.length) list.append(el('p', 'subtitle', 'no masters yet -- add one below.'));
  }

  function renderFiles() {
    filesList.textContent = '';
    (data.files || []).forEach(f => {
      const li = el('li');
      const a = document.createElement('a'); a.href = f.url; a.target = '_blank'; a.rel = 'noopener'; a.textContent = f.name;
      li.append(a);
      filesList.append(li);
    });
    if (!(data.files || []).length) filesList.append(el('li', null, '(none yet)'));
  }

  async function load() {
    saveBtn.disabled = true; addBtn.disabled = true;  // no in-memory `data`/`masters` to act on until this resolves
    const r = await fetch('/api/reports').then(r => r.json()).catch(() => null);
    if (!r) { errBox.hidden = false; errText.textContent = 'reports tab unavailable'; return; }
    data = r;
    errBox.hidden = !r.error;
    if (r.error) errText.textContent = r.error;
    masters = (r.masters || []).map(m => ({
      // prev_name: this master's name as loaded, so a rename can still be
      // told apart from a delete when the save reconciles local overrides
      // (#105 review) -- never touched again after this load.
      name: m.name, prev_name: m.name, title: m.title, send_at: m.send_at, enabled: m.enabled, send: m.send,
      llm_intro: m.llm_intro, max_age_hours: m.max_age_hours, skip_empty: m.skip_empty,
      sections: m.sections.map(s => ({ task: s.task, mode: s.mode, max_items: s.max_items, briefing: s.briefing })),
    }));
    renderMasters();
    renderFiles();
    saveMsg.textContent = '';
    saveBtn.disabled = false; addBtn.disabled = false;
  }

  addBtn.addEventListener('click', () => {
    masters.push({ name: '', prev_name: null, title: '', send_at: '06:45', enabled: true, send: false,
                  llm_intro: false, max_age_hours: 24, skip_empty: true, sections: [] });
    renderMasters();
  });

  saveBtn.addEventListener('click', async () => {
    if (!data) return;
    saveBtn.disabled = true;
    saveMsg.textContent = 'saving…'; saveMsg.className = 'launcher-msg';
    const { ok, status, j } = await post('/api/reports/save', { masters, base: data.base })
      .catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
    saveBtn.disabled = false;
    if (status === 409) { saveMsg.textContent = j.error + ' (reloading)'; saveMsg.className = 'launcher-msg hot'; load(); return; }
    if (!ok) { saveMsg.textContent = j.error || 'save failed'; saveMsg.className = 'launcher-msg err'; return; }
    saveMsg.textContent = j.message || 'saved'; saveMsg.className = 'launcher-msg ok';
    load();
  });

  $('#reports-refresh').addEventListener('click', load);
  load();
})();
"""

# Tabs + console dock (#100): tablist activation with hash routing and
# localStorage memory, arrow-key navigation, the Tasks group/search filter,
# deep-link reveal-and-scroll, and the dock's open/collapsed state. Runs after
# COLLAPSE_JS, so a group section's collapsed state is already applied before
# a deep link forces it open.
TABS_JS = r"""
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const ids = tabs.map(t => t.id.slice(4));
  const panels = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
  const TAB_KEY = 'lmsa-tab';

  function setBadge(id, n) {
    const b = $(`[data-tab-badge="${id}"]`); if (!b) return;
    b.textContent = String(n); b.hidden = !n;
  }

  function activate(id, { hash = true, focus = false } = {}) {
    if (!panels[id]) id = ids[0];
    for (const i of ids) {
      const on = i === id;
      if (panels[i]) panels[i].hidden = !on;
      const tab = document.getElementById('tab-' + i);
      if (tab) { tab.setAttribute('aria-selected', String(on)); tab.tabIndex = on ? 0 : -1; }
    }
    try { localStorage.setItem(TAB_KEY, id); } catch (e) {}
    if (hash && location.hash.slice(1) !== id) history.replaceState(null, '', '#' + id);
    if (focus) { const tab = document.getElementById('tab-' + id); if (tab) tab.focus(); }
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', ev => { ev.preventDefault(); activate(tab.id.slice(4)); });
    tab.addEventListener('keydown', ev => {
      const cur = ids.indexOf(tab.id.slice(4));
      let next = null;
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') next = ids[(cur + 1) % ids.length];
      else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') next = ids[(cur - 1 + ids.length) % ids.length];
      else if (ev.key === 'Home') next = ids[0];
      else if (ev.key === 'End') next = ids[ids.length - 1];
      if (next) { ev.preventDefault(); activate(next, { focus: true }); }
    });
  });
  window.addEventListener('hashchange', () => {
    const id = location.hash.slice(1);
    if (ids.includes(id)) activate(id, { hash: false });
  });

  // -- Tasks tab: group chips + name search ------------------------------
  const chips = [...document.querySelectorAll('[data-group-filter]')];
  const search = document.getElementById('task-search');
  function applyFilter() {
    const active = new Set(chips.filter(c => c.getAttribute('aria-pressed') === 'true').map(c => c.dataset.groupFilter));
    const q = ((search && search.value) || '').trim().toLowerCase();
    document.querySelectorAll('#tasks section[data-group]').forEach(sec => {
      const groupOn = active.has(sec.dataset.group);
      let any = false;
      sec.querySelectorAll('.launcher').forEach(card => {
        const name = (card.dataset.task || '').toLowerCase();
        const docEl = card.querySelector('.launcher-doc');
        const doc = (docEl ? docEl.textContent : '').toLowerCase();
        const show = groupOn && (!q || name.includes(q) || doc.includes(q));
        card.hidden = !show;
        if (show) any = true;
      });
      sec.hidden = !any;
    });
  }
  chips.forEach(c => c.addEventListener('click', () => {
    c.setAttribute('aria-pressed', c.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    applyFilter();
  }));
  if (search) search.addEventListener('input', applyFilter);

  // -- deep links: un-collapse + scroll to a task's launcher (#70, #100) --
  function revealTask(name) {
    const card = document.getElementById('task-' + name); if (!card) return;
    chips.forEach(c => c.setAttribute('aria-pressed', 'true'));
    if (search) search.value = '';
    applyFilter();
    const body = card.closest('.sec-body');
    if (body && body.hidden) {
      body.hidden = false;
      const sid = body.id.replace(/^body-/, '');
      const toggle = document.querySelector(`[data-collapse="${sid}"]`);
      if (toggle) { toggle.setAttribute('aria-expanded', 'true'); toggle.classList.remove('collapsed'); }
    }
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  document.querySelectorAll('[data-goto]').forEach(a => {
    a.addEventListener('click', ev => {
      ev.preventDefault();
      activate('tasks');
      requestAnimationFrame(() => revealTask(a.dataset.goto));
    });
  });

  // -- console dock --------------------------------------------------------
  const dockToggle = document.getElementById('dock-toggle'), dockBody = document.getElementById('dock-body');
  const DOCK_KEY = 'lmsa-dock-open';
  function setDock(open) {
    if (!dockToggle || !dockBody) return;
    dockToggle.setAttribute('aria-expanded', String(open));
    dockToggle.classList.toggle('collapsed', !open);
    dockBody.hidden = !open;
    try { if (open) localStorage.setItem(DOCK_KEY, '1'); else localStorage.removeItem(DOCK_KEY); } catch (e) {}
  }
  if (dockToggle) {
    let open = false;
    try { open = localStorage.getItem(DOCK_KEY) === '1'; } catch (e) {}
    setDock(open);
    dockToggle.addEventListener('click', () => setDock(dockBody.hidden));
  }
  window.lmsaTabs = { setBadge, activate, openDock: () => setDock(true) };

  // -- initial tab: hash > a deep link's task > remembered > Overview ----
  const prefillTask = document.body.dataset.prefillTask || '';
  let initial = null;
  const hashId = location.hash.slice(1);
  if (ids.includes(hashId)) initial = hashId;
  else if (prefillTask) initial = 'tasks';
  else { try { const stored = localStorage.getItem(TAB_KEY); if (ids.includes(stored)) initial = stored; } catch (e) {} }
  activate(initial || ids[0], { hash: false });
  applyFilter();
  if (prefillTask) revealTask(prefillTask);
})();
"""

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

# Reports stay sandboxed; the only script allowed is lmsa.report.HANDLED_JS,
# by hash (#61). allow-same-origin is what lets it call /api/handled with a
# matching Origin; no other inline script, handler or javascript: URL can run.
REPORT_CSP = (
    "sandbox allow-popups allow-popups-to-escape-sandbox allow-downloads allow-scripts allow-same-origin; "
    f"script-src '{HANDLED_JS_HASH}'; connect-src 'self'; base-uri 'none'; form-action 'none'"
)


def page_csp(nonce: str) -> str:
    return (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        "style-src 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


class Handler(BaseHTTPRequestHandler):
    server: "DashboardServer"
    server_version = "lmsa-dashboard"
    sys_version = ""
    # Socket timeout: a client that stalls sending headers or a POST body
    # (read before the auth checks, see do_POST) can't hold a thread forever.
    timeout = 10

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib signature
        pass  # polling would flood the console

    # -- helpers ------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def _drain(self, limit: int) -> None:
        """Read and discard up to `limit` bytes from the socket so a
        response that closes the connection right after doesn't leave
        unread bytes behind -- Windows turns that into an RST the client
        sees as a connection failure instead of our actual status code.
        Best-effort: stops on EOF/short read or any socket error rather
        than blocking for exactly `limit` bytes.
        """
        remaining = limit
        while remaining > 0:
            try:
                chunk = self.rfile.read(min(remaining, 65536))
            except OSError:
                return
            if not chunk:
                return
            remaining -= len(chunk)

    def _drain_chunked(self, cap: int) -> None:
        """Best-effort dechunk-and-discard for a `Transfer-Encoding:
        chunked` body we're refusing outright (we don't support chunked
        requests): stop as soon as the terminating zero-size chunk is
        seen, or once `cap` total bytes have been read, whichever comes
        first. That means a short, well-formed chunked body (the only
        kind any real client would send here) drains immediately instead
        of waiting out the full cap, while a client streaming an
        unbounded one still can't hold the thread past `cap`.
        """
        read = 0
        while read < cap:
            try:
                line = self.rfile.readline(min(64, cap - read))  # a chunk-size line is tiny
            except OSError:
                return
            read += len(line)
            if not line:
                return  # EOF
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                # Not a chunk-size line -- stop trying to parse chunk framing
                # and fall back to a raw drain for whatever's left of the
                # budget, so a garbled/unterminated body still gets drained
                # up to `cap` instead of leaving most of it unread.
                return self._drain(cap - read)
            if size == 0:
                return  # terminating chunk; trailers/final CRLF aren't worth chasing
            remaining = size + 2  # chunk data plus its trailing CRLF
            while remaining > 0 and read < cap:
                try:
                    chunk = self.rfile.read(min(remaining, cap - read, 65536))
                except OSError:
                    return
                if not chunk:
                    return
                read += len(chunk)
                remaining -= len(chunk)

    def _drain_declared_body(self) -> None:
        """For do_GET/do_HEAD, which never read a body at all otherwise:
        best-effort drain of whatever the client declared via
        Content-Length or `Transfer-Encoding: chunked`, bounded by
        DRAIN_CAP, so a client that sends a body on a GET doesn't leave it
        on the socket for the close that follows to turn into an RST.
        """
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self._drain_chunked(DRAIN_CAP)
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length < 0:  # unreadable or negative: drain to the cap, as do_POST does
            self._drain(DRAIN_CAP)
        elif length > 0:
            self._drain(min(length, DRAIN_CAP))

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 -- stdlib name
        # GET/HEAD carry no body in normal use, but a client that sends one
        # anyway (repro: a GET to a bad Host with a 200 KB body) leaves it
        # unread on the socket unless we drain it first -- same RST-on-close
        # hazard do_POST guards against below.
        self._drain_declared_body()
        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
        url = urlsplit(self.path)
        path = url.path
        dash = self.server.dashboard
        try:
            if path == "/":
                nonce = secrets.token_urlsafe(16)
                csp = (
                    "default-src 'none'; "
                    f"script-src 'nonce-{nonce}'; "
                    "style-src 'unsafe-inline' https://fonts.googleapis.com; "
                    "font-src https://fonts.gstatic.com; img-src data:; connect-src 'self'; "
                    # frame-src 'self' (#105): the Reports tab's email preview
                    # is a `sandbox=""` (no allow-scripts, no allow-same-origin)
                    # iframe filled via .srcdoc, never a src= URL -- nothing
                    # outside this page's own origin is ever framed.
                    "frame-src 'self'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                )
                page = render_page(dash, nonce, self.server.port, parse_qs(url.query)).encode("utf-8")
                return self._send(200, page, "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if path == "/errors":
                nonce = secrets.token_urlsafe(16)
                page = render_errors_page(dash.token, nonce, self.server.port, errfeed.log_retention_days())
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": page_csp(nonce)})
            if path == "/capabilities":
                nonce = secrets.token_urlsafe(16)
                page = render_capabilities_page(caps_data.build(dash), nonce, self.server.port)
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": page_csp(nonce)})
            if path == "/api/errors":
                return self._json(200, dash.errors_data())
            m = re.fullmatch(r"/api/errors/([a-z0-9]{1,40})", path)
            if m:
                full = (parse_qs(url.query).get("all") or ["0"])[0] == "1"
                detail = dash.error_detail(m.group(1), full=full)
                if detail is None:
                    return self._error(404, "no such failure")
                return self._json(200, detail)
            if path == "/api/handled/token":
                # Only same-origin pages (a report served under /reports/, or
                # the dashboard) may fetch it; SOP already stops cross-origin
                # reads, Sec-Fetch-Site refuses them outright where sent.
                site = self.headers.get("Sec-Fetch-Site")
                if site is not None and site != "same-origin":
                    return self._error(403, "cross-origin request refused")
                return self._json(200, {"token": dash.handled_token})
            if path == "/api/inbox":
                return self._json(200, dash.inbox_data())
            if path == "/api/inbox/filters.xml":
                xml = dash.filters_xml()
                if xml is None:
                    return self._error(404, "no filter suggestions")
                return self._send(200, xml.encode("utf-8"), "application/xml; charset=utf-8",
                                  {"Content-Disposition": 'attachment; filename="mailFilters.xml"'})
            if path == "/api/status":
                return self._json(200, dash.status())
            if path == "/api/runs":
                return self._json(200, {"live": dash.live_runs()})
            if path == "/api/setup":
                return self._json(200, dash.setup.status_all())
            if path == "/api/reports":
                return self._json(200, dash.reports_data())
            m = re.fullmatch(r"/api/runs/([0-9a-f]{1,32})", path)
            if m:
                run = dash.runs.get(m.group(1))
                if run is None:
                    return self._error(404, "no such run")
                qs = parse_qs(url.query)
                raw = (qs.get("offset") or ["0"])[0]
                offset = int(raw) if raw.isdigit() else 0
                return self._json(200, dash.run_state(run, offset))
            if path.startswith("/reports/"):
                p = dash.resolve_report(path[len("/reports/"):])
                if p is None:
                    return self._error(404, "no such report")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                if ctype.startswith("text/"):
                    ctype += "; charset=utf-8"
                return self._send(200, p.read_bytes(), ctype, {"Content-Security-Policy": REPORT_CSP})
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_HEAD = do_GET

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 -- stdlib name
        # Read the body before any check below that can return early: leaving
        # bytes unread in the socket when the response then closes the
        # connection makes Windows send RST instead of FIN, which the client
        # can see as a connection failure instead of the intended status code
        # -- rare in isolation but much more likely once concurrent workers
        # (`pytest -n auto`) put the process under load and widen the race.
        #
        # A chunked body has no Content-Length to read by, and a bad or
        # oversized one can't be read in full without letting a client tie
        # up the thread by declaring (or streaming) an unbounded body -- for
        # all three, drain up to DRAIN_CAP of whatever's actually on the
        # socket (best-effort; real requests never come close) and refuse
        # instead of running the request against a body we never read.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self._drain_chunked(DRAIN_CAP)
            return self._error(411, "chunked request bodies are not supported")
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._drain(DRAIN_CAP)
            return self._error(400, "bad Content-Length")
        if length < 0 or length > MAX_BODY:
            # A negative length is as unusable as a non-numeric one -- we
            # don't know how much is actually coming, so drain up to the
            # full cap same as the bad-Content-Length branch above, instead
            # of the 0 bytes `min(length, DRAIN_CAP)` would give here.
            self._drain(DRAIN_CAP if length < 0 else min(length, DRAIN_CAP))
            return self._error(413, "body too large")
        raw = self.rfile.read(length) if length else b""

        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            return self._error(403, "cross-origin request refused")
        token = self.headers.get(TOKEN_HEADER, "").encode("utf-8")
        full = hmac.compare_digest(token, self.server.dashboard.token.encode("utf-8"))
        # The report-page token (#61) is good for /api/handled and nothing else.
        scoped = urlsplit(self.path).path == "/api/handled" and hmac.compare_digest(
            token, self.server.dashboard.handled_token.encode("utf-8"))
        if not (full or scoped):
            return self._error(403, "missing or wrong token")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._error(415, "send application/json")
        try:
            body = json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, "body is not JSON")
        if not isinstance(body, dict):
            return self._error(400, "body must be an object")

        dash = self.server.dashboard
        path = urlsplit(self.path).path
        try:
            if path == "/api/handled":
                return self._json(200, dash.set_handled(body.get("msg_id"), body.get("handled")))
            if path == "/api/run":
                run, warning = dash.start(
                    body.get("task"), body.get("fields") or {}, body.get("extra") or "",
                    confirm=body.get("confirm") is True,
                )
                return self._json(201, {"id": run.id, "warning": warning})
            if path in ("/api/tasks/briefing-enable", "/api/tasks/briefing-disable"):
                fields = body.get("fields") if isinstance(body.get("fields"), dict) else {}
                result = dash.setup.set_briefing_enabled(fields.get("name"), path.endswith("-enable"),
                                                         confirm=body.get("confirm") is True)
                return self._json(200, result)
            if path in ("/api/tasks/schedule", "/api/tasks/manual"):
                confirm = body.get("confirm") is True
                if path.endswith("/schedule"):
                    result = dash.setup.add_to_schedule(body.get("task"), body.get("fields") or {}, confirm=confirm)
                else:
                    result = dash.setup.mark_manual(body.get("task"), confirm=confirm)
                return self._json(200, result)
            if path == "/api/setup/action":
                result = dash.setup.act(body.get("item"), body.get("action"), body.get("fields") or {},
                                        confirm=body.get("confirm") is True)
                return self._json(200, result)
            if path == "/api/reports/preview":
                return self._json(200, dash.reports_preview(body.get("name")))
            if path == "/api/reports/test-send":
                result = dash.reports_test_send(body.get("name"), confirm=body.get("confirm") is True)
                return self._json(200, result)
            if path == "/api/reports/save":
                result = dash.setup.save_master_report(body.get("masters"), body.get("base"))
                return self._json(200, result)
            if path == "/api/reports/local":
                result = dash.setup.set_master_local(body.get("name"), body.get("values"),
                                                     confirm=body.get("confirm") is True)
                return self._json(200, result)
            m = re.fullmatch(r"/api/runs/([0-9a-f]{1,32})/stop", path)
            if m:
                try:
                    run = dash.stop(m.group(1))
                except KeyError:
                    return self._error(404, "no such run")
                return self._json(200, {"id": run.id, "stopped": run.stopped})
            return self._error(404, "not found")
        except NeedsConfirm as exc:
            return self._error(428, str(exc))
        except BadRequest as exc:
            return self._error(400, str(exc))
        except Conflict as exc:
            return self._error(409, str(exc))
        except OSError as exc:  # marker lock busy past its timeout, or the marker couldn't be written
            return self._error(503, f"couldn't start: {exc} -- try again")


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second socket bind the same port; use
    # an exclusive bind instead so nothing can sit on (or under) the dashboard.
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)  # skips HTTPServer's slow getfqdn()
        self.server_name, self.server_port = self.server_address[:2]

    def __init__(self, dashboard: Dashboard, port: int = DEFAULT_PORT, host: str = HOST):
        if host != HOST:
            raise ValueError(f"the dashboard only binds {HOST} (got {host!r})")
        super().__init__((host, port), Handler)
        self.dashboard = dashboard
        self.port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tasks.dashboard", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page in the default browser")
    ap.add_argument("--capabilities", action="store_true",
                     help="write output/capabilities.html (the static capabilities report) and exit")
    args = ap.parse_args(argv)

    if args.capabilities:
        dash = Dashboard()
        path = render_static_report(caps_data.build(dash)).write("capabilities.html", dash.output_dir)
        print(f"Capabilities written to {path}", flush=True)
        return

    try:
        server = DashboardServer(Dashboard(), args.port)
    except OSError as exc:
        raise SystemExit(f"can't listen on 127.0.0.1:{args.port} ({exc}) -- is the dashboard already open?")
    adopted = [r.task for r in server.dashboard.runs.values() if r.adopted]
    print(f"Afterhours control on {server.url}  (Ctrl+C to quit; running tasks keep running "
          "and the next dashboard picks them up)", flush=True)
    if adopted:
        print(f"still running from before: {', '.join(adopted)}", flush=True)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    from lmsa.runner import run

    # Run the importable `tasks.dashboard`, not this `__main__` copy: the Setup
    # panel imports tasks.dashboard, and its BadRequest / NeedsConfirm /
    # Conflict must be the classes the request handler catches (with two
    # copies, a 428 or 400 from a Setup action dropped the connection).
    from tasks import dashboard as _module

    run(_module.main)
