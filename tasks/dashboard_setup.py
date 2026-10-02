"""Setup checklist for the dashboard: what isn't set up yet, and the fix.

`tasks.dashboard` only wires this in (a Setup section, `GET /api/setup`,
`POST /api/setup/action`); everything else lives here.

Each checklist entry is an `Item`: a name, a `status()` that returns
ok / missing / warning with the reason, and `Action`s. Items come from
*providers* -- functions `(setup) -> items` -- so a tool can add its own
entries with `register(provider)` without touching this file (#38).

Rules this module keeps:
  - Secrets (Gmail app password, Slack webhook) arrive only in a token-checked
    POST, go straight to the keyring, and are never echoed: not in a status,
    a response, an error message, or a log line. A failing keyring call is
    reported by exception type with the value redacted from its text.
  - `.env` writes: only keys `.env.example` lists, never a secret key, values
    without newlines/NULs/`${`, existing lines and comments preserved.
  - Config edits: a fixed allowlist of `config/*.toml` names, never a path
    from the request; validated by tomllib *and* the consuming task's own
    loader, backed up, then written atomically.
  - Every subprocess (`lms`, `powershell`) is an argv list with a timeout;
    nothing from a request is put in a command string.
  - New tools (#38): a task's `SETUP` block (read by ast, never imported)
    becomes a checklist item; "add to schedule" / "mark manual-only" append a
    table to schedule.toml / dashboard.toml through the same locked,
    validated, backed-up, atomic save as the config editor.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import secrets as _secrets
import shutil
import subprocess
import threading
import time
import tomllib
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable

from dotenv import dotenv_values

from lmsa import secrets as lmsa_secrets
from lmsa.config import settings
from lmsa.models import Lms
from lmsa.report import collapse_toggle_html, esc
from lmsa.search import (
    SearchError,
    SearchNotJSON,
    SearchUnavailable,
    SearXNG,
    is_connection_error,
    unresolved_executable,
)
from tasks import dashboard as _dash
from tasks.dashboard import BadRequest, Conflict, NeedsConfirm

STATES = ("ok", "missing", "warning")
MAX_VALUE = 2000  # a .env value or a secret
MAX_CONFIG = 60_000  # characters in a config textarea (the POST body cap is 64 KiB)
PROBE_TIMEOUT = 30  # seconds for the whole checklist; each probe has its own shorter one
# Redaction: a stored secret shorter than this isn't scrubbed from messages
# (a one-letter "secret" would shred every sentence); real app passwords and
# webhooks are far longer. Reading the secrets is bounded and cached so a
# slow or hung keyring can't stall the page.
MIN_REDACT_LEN = 8
SECRET_LOOKUP_TIMEOUT = 2.0
SECRET_CACHE_S = 30.0
PS_TIMEOUT = 20
REGISTER_TIMEOUT = 120
LMS_TIMEOUT = 20
OUTPUT_TAIL = 4000

# Keys that are secrets: never written through the plain .env form.
SECRET_ENV_KEYS = frozenset({*lmsa_secrets._ENV_VARS.values(), "LMSTUDIO_API_KEY"})

# config/<name>.toml files the editor may touch -- the whole allowlist.
# master_report (#105): the *tracked* config/master_report.toml only --
# names, titles and sections. Per-machine enabled/send/to overrides live in
# the untracked config/local/master_report.toml, which this allowlist never
# touches (tasks.master_report.parse_config enforces that split on load).
CONFIG_FILES = ("labels", "watchlist", "front_page", "doc_search", "schedule", "dashboard", "master_report")
# Allowlisted files that may be absent (absent = defaults, not a warning).
OPTIONAL_CONFIGS = frozenset({"dashboard"})

SCHEDULED_TASK = r"\lmsa\dispatch"
# Fixed script: no request data is ever interpolated into it.
PS_QUERY_TASK = (
    "$t = Get-ScheduledTask -TaskPath '\\lmsa\\' -TaskName 'dispatch' -ErrorAction SilentlyContinue; "
    "if ($t) { [string]$t.State } else { 'MISSING' }"
)
PIP_COMMAND = r'.venv\Scripts\python -m pip install -e ".[dev]"'
JIT_NOTE = ("Manual step: the JIT idle timeout can't be set through `lms`. In LM Studio's Developer "
            "tab, open the server settings, turn on \"Auto unload unused JIT loaded models\" and set "
            "its max idle TTL, so a model a task JIT-loads doesn't stay resident.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hint:
    """A line under the status. `command` hints get a copy button."""
    text: str
    command: bool = False


@dataclass
class Status:
    state: str  # one of STATES
    reason: str
    hints: list[Hint] = field(default_factory=list)
    # input name -> options, for a "select" input whose options are live
    choices: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state": self.state if self.state in STATES else "warning",
            "reason": self.reason,
            "hints": [{"text": h.text, "command": h.command} for h in self.hints],
            "choices": self.choices,
        }


@dataclass(frozen=True)
class Input:
    """A form input shared by an item's actions.
    kinds: text | secret (write-only password box) | select | textarea | hidden."""
    name: str
    label: str = ""
    kind: str = "text"
    options: tuple[str, ...] = ()  # static options of a select
    placeholder: str = ""


@dataclass(frozen=True)
class Action:
    """A button. `uses` names the item inputs it sends. `confirm` (a prompt)
    makes the server answer 428 until the client resends with confirm."""
    id: str
    label: str
    run: Callable[["Setup", dict], dict]
    uses: tuple[str, ...] = ()
    confirm: str | None = None


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    doc: str
    status: Callable[["Setup"], Status]
    inputs: tuple[Input, ...] = ()
    actions: tuple[Action, ...] = ()
    optional: bool = False
    note: str = ""  # a manual step, shown as text
    tone: str = "info"
    wide: bool = False


Provider = Callable[["Setup"], Iterable[Item]]
PROVIDERS: list[Provider] = []


def register(provider: Provider) -> Provider:
    """Add a checklist provider (usable as a decorator). Its items follow the
    built-in ones; ids must be unique across providers."""
    if provider not in PROVIDERS:
        PROVIDERS.append(provider)
    return provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forms(value: str) -> set[str]:
    """`value` as it can show up inside an error message: raw, repr()'d
    (a backend formatting `{value!r}` doubles backslashes and escapes
    quotes), JSON-escaped, unicode-escaped."""
    out = {value, repr(value), repr(value)[1:-1], ascii(value)[1:-1],
           json.dumps(value)[1:-1], value.encode("unicode_escape").decode("ascii")}
    return {f for f in out if f}


def redact(text: str, *values: str, min_len: int = MIN_REDACT_LEN) -> str:
    """`text` with every form of each value replaced by [redacted]. Values
    shorter than `min_len` are skipped; pass min_len=1 for values that must
    go whatever their length (the secret a request just submitted)."""
    forms = set()
    for v in values:
        if v and len(v) >= min_len:
            forms |= _forms(v)
    for f in sorted(forms, key=len, reverse=True):
        text = text.replace(f, "[redacted]")
    return text


def _read_secrets() -> list[str]:
    """The keyring's secret values -- read directly, not through resolve():
    an env var that shadows the keyring entry must not hide it, or removing
    the env var would expose a value that was never on the redaction list.
    (Env values are read live by known_secrets.)"""
    import keyring

    out = []
    for name in lmsa_secrets._ENV_VARS:
        try:
            value = keyring.get_password(lmsa_secrets.SERVICE, name)
        except Exception:
            continue
        if value:
            out.append(value)
    return out


def _has_control(value: str) -> bool:
    """Control characters and every line/paragraph separator
    (str.splitlines() also splits on \x0b, \x0c, \x1c-\x1e, \x85, U+2028/9)."""
    return any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in value)


def _check_value(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise BadRequest(f"{what} must be text")
    if len(value) > MAX_VALUE or _has_control(value):
        raise BadRequest(f"{what} must be one short line (no newlines or control characters)")
    return value


def _dotenv_quote(value: str) -> str:
    """A single-quoted python-dotenv value: only `\\` and `'` need escaping."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def atomic_write(path: Path, data: bytes, *, attempts: int = 5) -> None:
    """Write via a temp file + os.replace. On Windows the replace fails with
    PermissionError while another program has the target open; retry briefly,
    then give up with the target untouched and the temp file removed."""
    tmp = path.with_name(f".{path.name}.{_secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    try:
        replace_into(tmp, path, attempts=attempts)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def replace_into(tmp: Path, path: Path, *, attempts: int = 5) -> None:
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise Conflict(f"{path.name} is open in another program; nothing was changed -- try again") from None
            time.sleep(0.1 * (i + 1))


def _version_tuple(text: str) -> tuple[int, ...]:
    m = re.match(r"\d+(?:\.\d+)*", text.strip())
    return tuple(int(p) for p in m.group(0).split(".")) if m else ()


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;]*)")


def check_requirement(req: str, version_of: Callable[[str], str]) -> str | None:
    """None if `req` (a PEP 508 string) is satisfied, else the problem.
    Checks presence always, and `>=`, `>`, `==` bounds when given."""
    m = _REQ_RE.match(req)
    if not m:
        return None
    name, spec = m.group(1), m.group(2).strip()
    try:
        have = version_of(name)
    except importlib.metadata.PackageNotFoundError:
        return f"{name} is not installed"
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        cm = re.fullmatch(r"(>=|==|>)\s*([0-9][0-9.]*)", clause)
        if not cm:
            continue
        op, want = cm.group(1), _version_tuple(cm.group(2))
        got = _version_tuple(have)
        ok = {">=": got >= want, ">": got > want, "==": got == want}[op]
        if not ok:
            return f"{name} {have} installed, need {clause}"
    return None


def default_powershell(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class Setup:
    """The checklist for one dashboard. Everything that touches the machine
    (`lms`, PowerShell, Slack) is injectable; secrets go through the keyring
    module, which tests swap for an in-memory backend."""

    def __init__(
        self,
        dash,
        *,
        root: Path | None = None,
        config_dir: Path | None = None,
        backup_dir: Path | None = None,
        lms: Lms | None = None,
        searxng: SearXNG | None = None,
        powershell: Callable[[list[str], float], subprocess.CompletedProcess] | None = None,
        send_slack: Callable[[str], bool] | None = None,
        version_of: Callable[[str], str] = importlib.metadata.version,
    ):
        self.dash = dash
        self.root = Path(root or dash.root)
        self.env_path = self.root / ".env"
        self.env_example = self.root / ".env.example"
        self.config_dir = Path(config_dir or self.root / "config")
        self.backup_dir = Path(backup_dir or dash.db_path.parent / "dashboard" / "config-backups")
        self.lms = lms or Lms(timeout=LMS_TIMEOUT)
        # start_command=None: this runs on every page view, and must never spawn SearXNG (#89).
        self.searxng = searxng or SearXNG(settings().searxng_url, start_command=None)
        self.powershell = powershell or default_powershell
        self.send_slack = send_slack or _send_slack
        self.version_of = version_of
        self._env_lock = threading.Lock()
        # Redaction list: see known_secrets().
        self._secrets_lock = threading.Lock()
        self._secrets_last: list[str] = []   # last keyring read; never discarded, only replaced
        self._secrets_at: float | None = None  # when that read is fresh from; None = stale
        self._secrets_recent: list[str] = []  # values stored since the last fresh read
        self._secrets_gen = 0                  # bumped by every store; older reads aren't fresh
        self._secrets_future: concurrent.futures.Future | None = None
        with self._secrets_lock:
            self._start_read_locked()  # warm up: the first request shouldn't wait on a cold keyring
        # One save at a time per config file: the base check, validation,
        # backup and replace must not interleave (two tabs, one base).
        self.config_locks = {name: threading.Lock() for name in CONFIG_FILES}
        # Keys the real process environment set (not the .env): like
        # load_dotenv(override=False), they keep winning over the file.
        file_vals = self._file_values()
        self._process_env = {k for k in self.env_keys()
                             if os.environ.get(k) and file_vals.get(k) != os.environ.get(k)}

    # -- registry -----------------------------------------------------------

    def items(self) -> list[Item]:
        out: list[Item] = []
        seen: set[str] = set()
        for provider in [builtin_items, tool_items, *PROVIDERS]:
            for item in provider(self):
                if item.id not in seen:
                    seen.add(item.id)
                    out.append(item)
        return out

    def item(self, item_id: object) -> Item:
        for it in self.items():
            if it.id == item_id:
                return it
        raise BadRequest("unknown setup item")

    def status_all(self) -> dict:
        """Every item's status, probed concurrently: one slow probe (a hung
        `lms`, a slow PowerShell) costs its own row, not the page."""
        self.sync_env()
        items = self.items()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(len(items), 1))
        try:
            futures = {it.id: pool.submit(it.status, self) for it in items}
            concurrent.futures.wait(futures.values(), timeout=PROBE_TIMEOUT)
            out = []
            for it in items:
                fut = futures[it.id]
                if not fut.done():
                    st = Status("warning", f"still checking after {PROBE_TIMEOUT}s -- refresh later")
                else:
                    try:
                        st = fut.result()
                    except Exception as exc:  # a broken probe is a warning, not a 500
                        st = Status("warning", f"check failed: {type(exc).__name__}: {self.clean(exc, 200)}")
                out.append({"id": it.id, **st.as_dict()})
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        secrets_now = self.known_secrets()
        for row in out:
            row["reason"] = redact(row["reason"], *secrets_now)
            for h in row["hints"]:
                if not h.get("command"):  # commands are built from model ids, not secrets
                    h["text"] = redact(h["text"], *secrets_now)
        return {"items": out}

    def act(self, item_id: object, action_id: object, fields: object, *, confirm: bool = False) -> dict:
        item = self.item(item_id)
        action = next((a for a in item.actions if a.id == action_id), None)
        if action is None:
            raise BadRequest("unknown setup action")
        if not isinstance(fields, dict):
            raise BadRequest("fields must be an object")
        unknown = set(fields) - set(action.uses)
        if unknown:
            raise BadRequest(f"unknown field(s) for {item.id}.{action.id}")
        kinds = {i.name: i.kind for i in item.inputs}
        values: dict[str, str] = {}
        for name in action.uses:
            v = fields.get(name, "")
            if v is None:
                v = ""
            if not isinstance(v, str):
                raise BadRequest(f"{name} must be text")
            if len(v) > (MAX_CONFIG if kinds.get(name) == "textarea" else MAX_VALUE) or "\x00" in v:
                raise BadRequest(f"{name} is too long or contains a NUL")
            values[name] = v
        submitted = [values[n] for n in action.uses if kinds.get(n) == "secret" and values[n]]
        if action.confirm and not confirm:
            raise NeedsConfirm(action.confirm)
        # Every message leaving here is scrubbed of the submitted secrets AND
        # of every secret resolvable right now (an action that never declared
        # one -- "send test alert" -- can still trip over it).
        try:
            result = action.run(self, values)
        except (BadRequest, NeedsConfirm, Conflict) as exc:
            raise type(exc)(redact(redact(str(exc), *submitted, min_len=1), *self.known_secrets())) from None
        except Exception as exc:
            raise BadRequest(f"{action.label} failed: {type(exc).__name__}: "
                             f"{self.clean(exc, 300, *submitted)}") from None
        secrets_now = self.known_secrets()
        for key in ("message", "output", "warning"):
            if isinstance(result.get(key), str):
                result[key] = redact(redact(result[key], *submitted, min_len=1), *secrets_now)
        return result

    def known_secrets(self) -> list[str]:
        """Every secret value that could show up in a message, for redaction.

        - Env-var secrets are read live on every call (free).
        - Keyring secrets are read on a background thread and cached for
          SECRET_CACHE_S. The thread stores its own result when it finishes,
          so a read slower than SECRET_LOOKUP_TIMEOUT still lands in the
          cache for the next call instead of being thrown away.
        - While no fresh read is available, the last values read and every
          value stored since are still returned: nothing known is dropped.
        - A store (`forget_secrets`) marks the cache stale and bumps a
          generation, so a read that was already running when the secret
          changed is kept as extra values but never counted as fresh."""
        env = [v for v in (os.environ.get(e) for e in lmsa_secrets._ENV_VARS.values()) if v]
        with self._secrets_lock:
            fresh = self._secrets_at is not None and time.monotonic() - self._secrets_at < SECRET_CACHE_S
            known = [*self._secrets_last, *self._secrets_recent]
            fut = self._start_read_locked() if not fresh else None
        if not fresh:
            try:
                known += fut.result(timeout=SECRET_LOOKUP_TIMEOUT)
            except Exception:
                pass  # timed out or the keyring failed: fall back to what's known
        return list(dict.fromkeys(env + known))

    def _start_read_locked(self) -> concurrent.futures.Future:
        """The in-flight keyring read, starting one if none is. Caller holds
        `_secrets_lock`. At most one read is ever outstanding."""
        fut = self._secrets_future
        if fut is None or fut.done():
            fut = concurrent.futures.Future()
            threading.Thread(target=self._read_into_cache, args=(fut, self._secrets_gen),
                             daemon=True, name="setup-secrets").start()
            self._secrets_future = fut
        return fut

    def _read_into_cache(self, fut: concurrent.futures.Future, gen: int) -> None:
        try:
            values = _read_secrets()
        except BaseException as exc:
            fut.set_exception(exc)
            return
        with self._secrets_lock:
            if gen == self._secrets_gen:
                self._secrets_last, self._secrets_at, self._secrets_recent = values, time.monotonic(), []
            else:  # a secret changed mid-read: keep these as extras, not as fresh
                self._secrets_last = list(dict.fromkeys(self._secrets_last + values))
        fut.set_result(values)

    def forget_secrets(self, stored: str | None = None) -> None:
        """A secret was just stored: the cache is stale, and `stored` is
        redacted from now on even before the next keyring read."""
        with self._secrets_lock:
            self._secrets_gen += 1
            self._secrets_at = None
            if stored:
                self._secrets_recent.append(stored)

    def clean(self, text: object, limit: int | None = None, *extra: str, tail: bool = False) -> str:
        """`text` with every known secret redacted, THEN clipped to `limit`
        characters: clipping first could cut a secret so it no longer matches."""
        out = redact(redact(str(text), *extra, min_len=1), *self.known_secrets())
        if limit is not None and len(out) > limit:
            out = out[-limit:] if tail else out[:limit]
        return out

    # -- .env -----------------------------------------------------------------

    def env_keys(self) -> set[str]:
        """Keys `.env.example` lists -- the only ones this page reads or writes."""
        try:
            text = self.env_example.read_text(encoding="utf-8")
        except OSError:
            return set()
        return {m.group(1) for m in re.finditer(r"(?m)^\s*([A-Z][A-Z0-9_]*)\s*=", text)}

    def _file_values(self) -> dict[str, str]:
        try:
            if not self.env_path.is_file():
                return {}
            # interpolate=False: this reads what the file says, and a value
            # can't make it expand ${...} (write_env refuses those anyway).
            return {k: v for k, v in dotenv_values(self.env_path, interpolate=False).items() if v is not None}
        except (OSError, ValueError, UnicodeDecodeError):
            return {}

    def env_value(self, key: str) -> str | None:
        """What a task launched now would see for `key`: the process
        environment when it set the key, else the `.env` file as it is now."""
        if key in self._process_env:
            return os.environ.get(key) or None
        return self._file_values().get(key) or None

    def sync_env(self) -> None:
        """Make this process's environment match the `.env` file for the keys
        `.env.example` lists (tasks the dashboard launches inherit it), so an
        edit counts without restarting the dashboard. Keys the real
        environment set are left alone."""
        with self._env_lock:
            file_vals = self._file_values()
            for key in self.env_keys() - self._process_env:
                value = file_vals.get(key)
                if value:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)

    def write_env(self, key: str, value: str) -> None:
        """Set `key` in the local `.env`: in place if present (every
        occurrence, so no later duplicate wins), else appended. Creates the
        file if missing; other lines and comments are kept byte for byte."""
        allowed = self.env_keys() - SECRET_ENV_KEYS
        if key not in allowed:
            raise BadRequest(f"{key} can't be set here (not a non-secret key in .env.example)")
        _check_value(value, key)
        if "${" in value:
            raise BadRequest(f"{key}: '${{' isn't allowed (the .env loader would expand it)")
        with self._env_lock:
            try:
                # bytes, not read_text: universal newlines would turn CRLF into LF
                text = self.env_path.read_bytes().decode("utf-8") if self.env_path.is_file() else ""
            except (OSError, UnicodeDecodeError) as exc:
                raise BadRequest(f"can't read .env ({type(exc).__name__})") from None
            nl = "\r\n" if "\r\n" in text else "\n"
            line = f"{key}={_dotenv_quote(value)}"
            pattern = re.compile(rf"^[ \t]*(?:export[ \t]+)?{re.escape(key)}[ \t]*=.*$")
            # Split on "\n" only (keeping "\r\n"): str.splitlines() also breaks on
            # \x0b, \x0c, \x85, U+2028 ... which python-dotenv does not.
            lines = re.findall(r"[^\n]*\n|[^\n]+$", text)
            found = False
            for i, old in enumerate(lines):
                body = old.rstrip("\r\n")
                if pattern.match(body):
                    lines[i] = line + old[len(body):]
                    found = True
            if not found:
                if lines and not lines[-1].endswith(("\n", "\r")):
                    lines[-1] += nl
                lines.append(line + nl)
            atomic_write(self.env_path, "".join(lines).encode("utf-8"))
        self.sync_env()

    # -- secrets ------------------------------------------------------------

    def set_secret(self, name: str, value: str) -> str:
        """Store a secret in the keyring. Returns a note when an env var
        still overrides the keyring entry. Never echoes the value."""
        if name not in lmsa_secrets._ENV_VARS:
            raise BadRequest("unknown secret")
        if not value:
            raise BadRequest(f"no {name} value given")
        if len(value) > MAX_VALUE or _has_control(value):
            raise BadRequest(f"the {name} value must be one short line (no control characters)")
        if len(value) < MIN_REDACT_LEN:
            # Shorter values can't be redacted without shredding every message;
            # Gmail app passwords are 16 characters and webhooks far longer.
            raise BadRequest(f"that {name} value is too short to be real (at least {MIN_REDACT_LEN} characters)")
        import keyring

        try:
            keyring.set_password(lmsa_secrets.SERVICE, name, value)
            self.forget_secrets(value)
        except Exception as exc:
            # Type only: a backend's message may quote the value in any encoding.
            raise BadRequest(f"couldn't store it in the keyring ({type(exc).__name__})") from None
        env_var = lmsa_secrets._ENV_VARS[name]
        if os.environ.get(env_var):
            return f"stored in the keyring, but {env_var} is set (environment or .env) and still wins"
        return "stored in the keyring"

    def secret_source(self, name: str) -> str | None:
        """Where secret `name` resolves from ("keyring" / its env var), or None.
        Raises what the keyring raises."""
        env_var = lmsa_secrets._ENV_VARS[name]
        if os.environ.get(env_var):
            return env_var
        return "keyring" if lmsa_secrets.resolve(name) is not None else None

    # -- launcher -------------------------------------------------------------

    def launch(self, task: str, values: dict | None = None) -> dict:
        """Start a task through the dashboard's own launcher, so its log
        streams in the console like any other run."""
        run, warning = self.dash.start(task, values or {})
        return {"message": f"started {task}", "run": {"id": run.id, "task": task}, "warning": warning}

    # -- config files ---------------------------------------------------------

    def config_path(self, name: object) -> Path:
        if name not in CONFIG_FILES:
            raise BadRequest("unknown config file")
        return self.config_dir / f"{name}.toml"

    def add_to_schedule(self, task: object, fields: object, *, confirm: bool = False) -> dict:
        return add_to_schedule(self, task, fields, confirm=confirm)

    def mark_manual(self, task: object, *, confirm: bool = False) -> dict:
        return mark_manual(self, task, confirm=confirm)

    def set_briefing_enabled(self, name: object, enabled: bool, *, confirm: bool = False) -> dict:
        return set_briefing_enabled(self, name, enabled, confirm=confirm)

    def set_master_local(self, name: object, values: object, *, confirm: bool = False) -> dict:
        return set_master_local(self, name, values, confirm=confirm)

    def save_master_report(self, masters: object, base: object) -> dict:
        return save_master_report(self, masters, base)

    def validate_config(self, name: str, path: Path) -> None:
        """tomllib, then the consuming task's own loader. Raises ValueError."""
        tomllib.loads(path.read_text(encoding="utf-8"))
        VALIDATORS[name](self, path)


def _send_slack(text: str) -> bool:
    from lmsa.notify import slack

    return slack(text)


# ---------------------------------------------------------------------------
# Validators: reuse what each consuming task already runs
# ---------------------------------------------------------------------------

def _v_schedule(setup: Setup, path: Path) -> None:
    from tasks.dispatch import load_schedule

    sched = load_schedule(path, setup.root / "tasks")
    if sched.problems:
        raise ValueError("; ".join(sched.problems))


def _v_labels(setup: Setup, path: Path) -> None:
    from tasks.labeler import load_labels

    load_labels(path)


def _v_watchlist(setup: Setup, path: Path) -> None:
    from tasks.price_watch import load_watchlist

    bad = [i.load_error for i in load_watchlist(path) if i.load_error]
    if bad:
        raise ValueError("; ".join(bad))


def _v_front_page(setup: Setup, path: Path) -> None:
    from tasks.front_page import load_config

    load_config(path)


def _v_doc_search(setup: Setup, path: Path) -> None:
    from tasks.doc_index import load_config

    load_config(path)


def _v_master_report(setup: Setup, path: Path) -> None:
    from tasks.master_report import ConfigError, parse_config

    # Validate against this machine's local overrides too: renaming or
    # deleting a master here must not orphan its [master.<name>] override,
    # which would break load_config for the tab and the scheduled run.
    local = local_master_report_path(setup)
    try:
        overrides = local.read_text(encoding="utf-8")
    except FileNotFoundError:
        overrides = None
    try:
        parse_config(path.read_text(encoding="utf-8"), overrides=overrides, **_mr_scope(setup))
    except ConfigError as exc:
        raise ValueError("; ".join(exc.errors)) from None


def _v_dashboard(setup: Setup, path: Path) -> None:
    manual, problems = _dash.parse_dashboard_config(tomllib.loads(path.read_text(encoding="utf-8")))
    # names already matched TASK_NAME_RE, so this stat stays inside tasks/
    problems += [f"manual.{n}: no module tasks/{n}.py" for n in sorted(manual)
                 if not (setup.root / "tasks" / f"{n}.py").exists()]
    if problems:
        raise ValueError("; ".join(problems))


VALIDATORS: dict[str, Callable[[Setup, Path], None]] = {
    "schedule": _v_schedule,
    "dashboard": _v_dashboard,
    "labels": _v_labels,
    "watchlist": _v_watchlist,
    "front_page": _v_front_page,
    "doc_search": _v_doc_search,
    "master_report": _v_master_report,
}


# ---------------------------------------------------------------------------
# Built-in items
# ---------------------------------------------------------------------------

# -- Gmail --

def _gmail_status(s: Setup) -> Status:
    user = s.env_value("LMSA_IMAP_USER")
    try:
        source = s.secret_source("imap")
    except Exception as exc:
        return Status("warning", f"can't read the keyring ({type(exc).__name__})")
    missing = []
    if not user:
        missing.append("no Gmail address (LMSA_IMAP_USER)")
    if not source:
        missing.append("no app password")
    if missing:
        return Status("missing", "; ".join(missing))
    return Status("ok", f"{user} · app password from {source}")


def _gmail_save(s: Setup, v: dict) -> dict:
    address, password = v.get("address", "").strip(), v.get("password", "")
    if not address and not password:
        raise BadRequest("enter the Gmail address, the app password, or both")
    done = []
    if address:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address):
            raise BadRequest("that doesn't look like an email address")
        s.write_env("LMSA_IMAP_USER", address)
        done.append("address saved to .env")
    if password:
        done.append("app password " + s.set_secret("imap", password))
    return {"message": "; ".join(done)}


# -- Slack --

def _slack_status(s: Setup) -> Status:
    try:
        source = s.secret_source("slack")
    except Exception as exc:
        return Status("warning", f"can't read the keyring ({type(exc).__name__})")
    if not source:
        return Status("missing", "optional: no webhook, so alerts only show as Windows toasts")
    return Status("ok", f"webhook from {source}")


def _slack_save(s: Setup, v: dict) -> dict:
    url = v.get("webhook", "").strip()
    if not url:
        raise BadRequest("paste the incoming-webhook URL")
    _check_webhook(url)
    return {"message": "webhook " + s.set_secret("slack", url)}


def _check_webhook(url: str) -> None:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        parts = None
    if (parts is None or parts.scheme != "https" or not parts.hostname or _has_control(url)
            or any(c.isspace() for c in url)):
        raise BadRequest("the webhook must be an https:// URL with no spaces or control characters")


def _slack_test(s: Setup, v: dict) -> dict:
    ok = s.send_slack(":wave: test alert from the lmsa dashboard")
    if not ok:
        raise BadRequest("not sent: no webhook configured, or Slack refused it (see the dashboard's terminal)")
    return {"message": "test alert sent"}


# -- LM Studio --

def _model_names(entry: dict) -> set[str]:
    names = {entry.get("modelKey"), entry.get("path"), entry.get("indexedModelIdentifier")}
    variants = entry.get("variants")
    if isinstance(variants, list):
        names.update(variants)
    return {str(n) for n in names if n}


def _pinned(instances: list[dict]) -> list[str]:
    """Loaded instances with no TTL (`ttlMs` present and null). A missing
    `ttlMs` key means an `lms` that doesn't report it: not flagged."""
    return [str(m.get("identifier") or m.get("modelKey")) for m in instances
            if "ttlMs" in m and m.get("ttlMs") is None]


def _ttl_unknown(instances: list[dict]) -> list[str]:
    """Loaded instances whose TTL `lms ps --json` doesn't report at all --
    the pinned check can't say anything about them."""
    return [str(m.get("identifier") or m.get("modelKey")) for m in instances if "ttlMs" not in m]


def _lmstudio_status(s: Setup) -> Status:
    problems, warnings, hints = [], [], []
    try:
        server = bool(s.dash.probe_server())
    except Exception:
        server = False
    if not server:
        problems.append("server not answering (start LM Studio's server)")
        hints.append(Hint("lms server start", command=True))
    cfg = settings()
    wanted = {"chat": cfg.chat_model, "vision": cfg.vision_model, "embed": cfg.embed_model}
    try:
        on_disk: set[str] = set()
        for entry in s.lms.downloaded():
            on_disk |= _model_names(entry)
        for role, model in wanted.items():
            if model not in on_disk:
                problems.append(f"{role} model {model} isn't downloaded")
                hints.append(Hint(f"lms get {model}", command=True))
    except Exception as exc:
        warnings.append(f"can't list models ({type(exc).__name__}: {s.clean(exc, 160)})")
    pinned: list[str] = []
    try:
        instances = s.lms.instances()
        pinned = _pinned(instances)
        unknown = _ttl_unknown(instances)
        if unknown:
            hints.append(Hint(f"pinned check unknown for {', '.join(unknown)}: this `lms ps --json` "
                              "doesn't report a TTL (ttlMs)"))
    except Exception as exc:
        warnings.append(f"can't list loaded models ({type(exc).__name__}: {s.clean(exc, 160)})")
    if pinned:
        warnings.append(f"loaded without a TTL (pinned): {', '.join(pinned)}")
    choices = {"model": pinned}
    if problems:
        return Status("missing", "; ".join(problems + warnings), hints, choices)
    if warnings:
        return Status("warning", "; ".join(warnings), hints, choices)
    pinned_note = "pinned check unknown" if any("pinned check unknown" in h.text for h in hints) else "nothing pinned"
    return Status("ok", f"server up; chat, vision and embed models downloaded; {pinned_note}", hints, choices)


def _lmstudio_unload(s: Setup, v: dict) -> dict:
    model = v.get("model", "")
    if not model or model.startswith("-"):
        raise BadRequest("pick a pinned model to unload")
    if model not in _pinned(s.lms.instances()):
        raise BadRequest("that model isn't loaded without a TTL (anymore); refresh")
    s.lms.unload(model)
    return {"message": f"unloaded {model}"}


# -- Scheduler --

def _ps(*args: str) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", *args]


def _scheduler_status(s: Setup) -> Status:
    try:
        proc = s.powershell(_ps("-Command", PS_QUERY_TASK), PS_TIMEOUT)
    except subprocess.TimeoutExpired:
        return Status("warning", f"Task Scheduler didn't answer within {PS_TIMEOUT}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return Status("warning", f"can't run PowerShell ({type(exc).__name__})")
    state = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    state = state[0].strip()
    if proc.returncode != 0 or not state:
        err = s.clean((proc.stderr or "").strip(), 200, tail=True)
        return Status("warning", f"can't query Task Scheduler (exit {proc.returncode}) {err}".strip())
    if state == "MISSING":
        return Status("missing", f"{SCHEDULED_TASK} isn't registered: nothing runs on a schedule")
    if state.lower() == "disabled":
        return Status("warning", f"{SCHEDULED_TASK} is registered but disabled")
    return Status("ok", f"{SCHEDULED_TASK} registered ({s.clean(state, 40)})")


def _scheduler_register(s: Setup, v: dict) -> dict:
    script = s.root / "scheduler" / "register-task.ps1"
    if not script.is_file():
        raise BadRequest("scheduler/register-task.ps1 is missing")
    argv = _ps("-ExecutionPolicy", "Bypass", "-File", str(script))
    try:
        proc = s.powershell(argv, REGISTER_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise BadRequest(f"register-task.ps1 didn't finish within {REGISTER_TIMEOUT}s") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise BadRequest(f"can't run PowerShell ({type(exc).__name__})") from None
    output = s.clean(((proc.stdout or "") + (proc.stderr or "")).strip(), OUTPUT_TAIL, tail=True)
    if proc.returncode != 0:
        return {"message": f"register-task.ps1 exited {proc.returncode}", "output": output, "error": True}
    return {"message": f"registered {SCHEDULED_TASK}", "output": output or "(no output)"}


# -- Python env --

def _python_status(s: Setup) -> Status:
    try:
        data = tomllib.loads((s.root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Status("warning", f"can't read pyproject.toml ({type(exc).__name__})")
    project = data.get("project", {})
    reqs = list(project.get("dependencies", []))
    reqs += list(project.get("optional-dependencies", {}).get("dev", []))
    problems = [p for p in (check_requirement(r, s.version_of) for r in reqs if isinstance(r, str)) if p]
    if problems:
        state = "missing" if any("not installed" in p for p in problems) else "warning"
        return Status(state, "; ".join(problems), [Hint(PIP_COMMAND, command=True)])
    return Status("ok", f"{len(reqs)} dependencies from pyproject.toml installed")


# -- Config files --

def _config_status(s: Setup) -> Status:
    bad, absent = [], []
    for name in CONFIG_FILES:
        path = s.config_path(name)
        if not path.is_file():
            if name not in OPTIONAL_CONFIGS:
                absent.append(f"{name}.toml")
            continue
        try:
            s.validate_config(name, path)
        except Exception as exc:
            bad.append(f"{name}.toml: {type(exc).__name__}: {s.clean(exc, 200)}")
    if bad or absent:
        parts = bad + ([f"not found: {', '.join(absent)}"] if absent else [])
        return Status("warning", "; ".join(parts))
    return Status("ok", f"{len(CONFIG_FILES)} files parse and pass their task's checks")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def current_sha(s: Setup, name: str) -> str:
    """The base a caller building on config/<name>.toml's current bytes
    should send back to a `_config_save_locked`-based save (`"new"` if the
    file doesn't exist yet) -- what `_config_load` hands the generic editor,
    exposed for a structured editor (the Reports tab) that never calls it."""
    try:
        return _sha(s.config_path(name).read_bytes())
    except FileNotFoundError:
        return "new"


def _config_load(s: Setup, v: dict) -> dict:
    name = v.get("file", "")
    path = s.config_path(name)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return {"message": f"{name}.toml doesn't exist yet: saving creates it",
                "fill": {"text": "", "base": "new"}}
    except OSError as exc:
        raise BadRequest(f"can't read {name}.toml ({type(exc).__name__})") from None
    return {"message": f"loaded {name}.toml", "fill": {"text": data.decode("utf-8", errors="replace"),
                                                       "base": _sha(data)}}


def _config_save(s: Setup, v: dict) -> dict:
    name, text, base = v.get("file", ""), v.get("text", ""), v.get("base", "")
    path = s.config_path(name)
    if not base:
        raise BadRequest(f"load {name}.toml first")
    # Base check -> validate -> backup -> replace under one lock: two saves
    # from the same base can't both pass the check (the second gets a 409).
    if not s.config_locks[name].acquire(timeout=30):
        raise Conflict(f"another save of {name}.toml is still running; try again")
    try:
        return _config_save_locked(s, name, path, text, base)
    finally:
        s.config_locks[name].release()


def _config_save_locked(s: Setup, name: str, path: Path, text: str, base: str,
                        check: Callable[[Path], None] | None = None,
                        validate: Callable[[Path], None] | None = None) -> dict:
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    if (current is None and base != "new") or (current is not None and _sha(current) != base):
        raise Conflict(f"{name}.toml changed since you loaded it; load it again")
    s.config_dir.mkdir(parents=True, exist_ok=True)
    if current is not None and b"\r\n" in current and "\r" not in text:
        text = text.replace("\n", "\r\n")  # a textarea hands back LF; keep the file's CRLF
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        # A lone UTF-16 surrogate (valid inside a Python str -- and inside a
        # JSON string, whose \uXXXX escapes are exactly UTF-16 code units --
        # but never valid UTF-8) reaches here from any caller that didn't
        # already reject it: a textarea (the generic config editor), or a
        # structured save that builds `text` itself (#105 review). Without
        # this, encode() raises past every caller straight into the request
        # handler, which has no `except` for it -- the connection drops with
        # no response at all instead of a clean 400.
        raise BadRequest(f"not saved -- {name}.toml has text that isn't valid Unicode ({exc})") from None
    tmp = path.with_name(f".{path.name}.{_secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    try:
        try:
            if validate is not None:
                validate(tmp)
            else:
                s.validate_config(name, tmp)
            if check is not None:
                check(tmp)
        except Exception as exc:
            msg = str(exc).replace(str(tmp), f"{name}.toml").replace(tmp.name, f"{name}.toml")
            raise BadRequest(f"not saved -- {name}.toml is invalid: {type(exc).__name__}: {s.clean(msg, 400)}") from None
        backup = None
        if current is not None:
            s.backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = s.backup_dir / f"{name}-{stamp}-{_secrets.token_hex(3)}.toml"
            with open(backup, "xb") as f:  # exclusive: never overwrite another backup
                f.write(current)
        replace_into(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    note = f"; backup: {backup.name}" if backup else ""
    return {"message": f"saved {name}.toml{note}", "fill": {"base": _sha(data)}}


def builtin_items(s: Setup) -> list[Item]:
    return [
        Item(
            "gmail", "Gmail", "IMAP login for every Gmail task: your address and a Google app password.",
            _gmail_status,
            inputs=(Input("address", "gmail address", placeholder="you@gmail.com"),
                    Input("password", "app password", kind="secret", placeholder="write-only")),
            actions=(Action("save", "save", _gmail_save, uses=("address", "password")),
                     Action("test", "test", lambda s, v: s.launch("gmail_probe"))),
            tone="info",
        ),
        Item(
            "slack", "Slack alerts", "Incoming-webhook URL; alerts also post to Slack when set.",
            _slack_status,
            inputs=(Input("webhook", "webhook url", kind="secret", placeholder="https://hooks.slack.com/..."),),
            actions=(Action("save", "save", _slack_save, uses=("webhook",)),
                     Action("test", "send test alert", _slack_test)),
            optional=True, tone="violet",
        ),
        Item(
            "lmstudio", "LM Studio", "Server reachable, the configured chat / vision / embed models downloaded, "
            "nothing pinned in VRAM.",
            _lmstudio_status,
            inputs=(Input("model", "pinned model", kind="select"),),
            actions=(Action("unload", "unload", _lmstudio_unload, uses=("model",),
                            confirm="unloading frees it for everyone using this LM Studio server; confirm"),),
            note=JIT_NOTE, tone="acc",
        ),
        Item(
            "scheduler", "Scheduler", f"The dispatcher registered as {SCHEDULED_TASK} in Task Scheduler.",
            _scheduler_status,
            actions=(Action("register", "register / update", _scheduler_register,
                            confirm="runs scheduler/register-task.ps1 (registers the dispatcher every 30 min); confirm"),
                     Action("dryrun", "dry run", lambda s, v: s.launch("dispatch", {"--dry-run": True}))),
            tone="warn",
        ),
        Item(
            "python", "Python env", "The venv's installed packages match pyproject.toml.",
            _python_status, tone="info",
        ),
        Item(
            "config", "Config files", "Edit config/*.toml. Saving validates with tomllib and the task's own "
            "loader, keeps a backup in data/dashboard/config-backups/, and writes atomically.",
            _config_status,
            inputs=(Input("file", "file", kind="select", options=CONFIG_FILES),
                    Input("text", "contents", kind="textarea"),
                    Input("base", kind="hidden")),
            actions=(Action("load", "load", _config_load, uses=("file",)),
                     Action("save", "save", _config_save, uses=("file", "text", "base"),
                            confirm="overwrites the config file (a backup is kept); confirm")),
            tone="acc", wide=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Web search (SearXNG) (#51)
# ---------------------------------------------------------------------------

SEARXNG_DOCKER_HINT = "docker run -d -p 8888:8080 --name searxng searxng/searxng"
SEARXNG_SETTINGS_HINT = "settings.yml: search:\n    formats:\n      - html\n      - json"


def _searxng_probe(s: "Setup") -> tuple[str, str]:
    """(state, reason) from a live search against ``s.searxng`` -- state is
    one of STATES. Shared by the dedicated card and the generic tool-needs
    check so both reflect the same probe. ``s.searxng`` is built with
    start_command=None (#89), so this never spawns anything itself; when the
    instance just isn't running right now and LMSA_SEARXNG_START is
    configured, that's an expected, ok state -- it starts on the next real
    search. But when the configured command's executable can't even be
    resolved (typo'd path, tool not installed), "starts on demand" would be
    a lie -- report missing, naming the unresolved executable, instead."""
    try:
        s.searxng.search("ping", max_results=1)
    except SearchNotJSON as exc:
        return "warning", f"JSON API not enabled ({exc})"
    except SearchUnavailable as exc:
        start_command = settings().searxng_start
        if is_connection_error(exc.__cause__) and start_command:
            bad_exe = unresolved_executable(start_command)
            if bad_exe:
                return "missing", (
                    f"not reachable, and LMSA_SEARXNG_START's executable can't be found: {bad_exe!r}"
                )
            return "ok", "not running now; starts on demand (LMSA_SEARXNG_START)"
        return "missing", f"not reachable ({exc})"
    except SearchError as exc:
        return "missing", f"not reachable ({exc})"
    return "ok", f"reachable at {s.searxng.base_url}, JSON API enabled"


def _searxng_status(s: "Setup") -> Status:
    state, reason = _searxng_probe(s)
    hints: list[Hint] = []
    if state == "missing":
        hints.append(Hint(SEARXNG_DOCKER_HINT, command=True))
    if state == "warning":
        hints.append(Hint(SEARXNG_SETTINGS_HINT, command=True))
    return Status(state, reason, hints)


def searxng_items(s: "Setup") -> list[Item]:
    return [
        Item(
            "searxng", "Web search (SearXNG)",
            "Self-hosted SearXNG instance (LMSA_SEARXNG_URL) used for open web search "
            "by briefings and go-deep. Needs its JSON API enabled.",
            _searxng_status,
            optional=True, tone="info",
        ),
    ]


register(searxng_items)


# ---------------------------------------------------------------------------
# New tools (#38): SETUP items, add to schedule, mark manual-only
# ---------------------------------------------------------------------------

MODEL_ROLES_SETUP = ("chat", "vision", "embed")
_COMMAND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# How to get a command a tool declares; anything else gets a generic hint.
INSTALL_HINTS = {"gh": "winget install --id GitHub.cli"}
SECRET_CARDS = {"imap": "the Gmail card", "slack": "the Slack alerts card"}


def config_rel(entry: str) -> str | None:
    """`entry` as a path relative to config/ -- or None unless it is lexically
    confined there. Nothing touches the filesystem before this returns: an
    absolute, drive, UNC or `..` path would make a later stat() open an SMB
    connection (leaking NTLM) or read outside the repo."""
    if not isinstance(entry, str) or not entry or "\x00" in entry or ":" in entry or _has_control(entry):
        return None
    norm = entry.replace("\\", "/")
    if norm.startswith("/") or PureWindowsPath(entry).anchor:
        return None
    parts = norm.split("/")
    if parts[0] != "config" or len(parts) < 2 or any(p in ("", ".", "..") for p in parts[1:]):
        return None
    return "/".join(parts[1:])


def _tool_status(s: Setup, t) -> Status:
    """A tool's declared needs. Cheap probes only (env, keyring presence,
    a file stat under config/, shutil.which); never a secret's value."""
    need = t.setup or {}
    missing: list[str] = []
    warn: list[str] = list(t.setup_warnings)
    ok: list[str] = []
    hints: list[Hint] = []
    for name in need.get("secrets", []):
        if name not in lmsa_secrets._ENV_VARS:
            warn.append(f"unknown secret {name!r} (lmsa.secrets knows {', '.join(sorted(lmsa_secrets._ENV_VARS))})")
            continue
        try:
            source = s.secret_source(name)
        except Exception as exc:
            warn.append(f"can't read the keyring ({type(exc).__name__})")
            continue
        if source:
            ok.append(f"secret {name} from {source}")
        else:
            missing.append(f"secret {name} not set")
            where = SECRET_CARDS.get(name)
            hints.append(Hint(f"set it below, or on {where}" if where else "set it below"))
    keys = s.env_keys()
    for key in need.get("env", []):
        if key not in keys:
            warn.append(f"unknown env var {key!r} (not in .env.example)")
            continue
        present = os.environ.get(key) if key in SECRET_ENV_KEYS else s.env_value(key)
        if present:
            ok.append(f"{key} set")  # set or not: never the value
        else:
            missing.append(f"{key} not set")
    for entry in need.get("config", []):
        rel = config_rel(entry)
        if rel is None:
            warn.append(f"config path {entry[:80]!r} isn't a relative path under config/; ignored")
            continue
        path = s.config_dir / rel
        if path.is_file():
            ok.append(f"config/{rel} present")
            continue
        missing.append(f"config/{rel} missing")
        stem = rel[:-5] if rel.endswith(".toml") and "/" not in rel else None
        hints.append(Hint(f"create it in the Config files card ({stem})" if stem in CONFIG_FILES
                          else f"create config/{rel}"))
    for cmd in need.get("commands", []):
        if not _COMMAND_RE.fullmatch(cmd):
            warn.append(f"command {cmd[:40]!r} must be a bare program name; ignored")
            continue
        if shutil.which(cmd):
            ok.append(f"{cmd} on PATH")
        else:
            missing.append(f"{cmd} isn't on PATH")
            hints.append(Hint(INSTALL_HINTS[cmd], command=True) if cmd in INSTALL_HINTS
                         else Hint(f"install {cmd} and make sure it's on PATH, then restart the dashboard"))
    model = need.get("model")
    if model is not None:
        if model in MODEL_ROLES_SETUP:
            cfg = settings()
            ok.append(f"uses the {model} model ({getattr(cfg, model + '_model', '?')})")
        elif model != "none":
            warn.append(f"unknown model role {model!r} (use chat, vision or embed)")
    if need.get("searxng"):
        state, reason = _searxng_probe(s)
        (ok if state == "ok" else warn if state == "warning" else missing).append(f"SearXNG {reason}")
        if state != "ok":
            hints.append(Hint("set up SearXNG on the Web search (SearXNG) card"))
    if missing:
        return Status("missing", "; ".join(missing + warn), hints)
    if warn:
        return Status("warning", "; ".join(warn), hints)
    return Status("ok", "; ".join(ok) or "declares nothing it needs", hints)


def _tool_inputs(s: Setup, t) -> tuple[Input, ...]:
    """Inputs for the needs this page can fix: non-secret env keys that
    .env.example lists, and known secrets (write-only boxes)."""
    need = t.setup or {}
    keys = s.env_keys()
    out = [Input(f"secret-{n}", f"{n} secret", kind="secret", placeholder="write-only")
           for n in need.get("secrets", []) if n in lmsa_secrets._ENV_VARS]
    out += [Input(k, k) for k in need.get("env", []) if k in keys and k not in SECRET_ENV_KEYS]
    return tuple(out)


def _tool_save(s: Setup, v: dict) -> dict:
    done = []
    for name, value in v.items():
        if not value:
            continue
        if name.startswith("secret-"):
            secret = name[len("secret-"):]
            value = value.strip()
            if secret == "slack":
                _check_webhook(value)
            done.append(f"{secret}: " + s.set_secret(secret, value))
        else:
            s.write_env(name, value.strip())
            done.append(f"{name} saved to .env")
    if not done:
        raise BadRequest("nothing to save: fill in at least one box")
    return {"message": "; ".join(done)}


def tool_items(s: Setup) -> list[Item]:
    """One item per task that declares SETUP (or whose SETUP couldn't be read)."""
    catalog = getattr(s.dash, "catalog", None)
    if catalog is None:
        return []
    items = []
    for t in catalog():
        if t.setup is None and not t.setup_warnings:
            continue
        inputs = _tool_inputs(s, t)
        actions = (Action("save", "save", _tool_save, uses=tuple(i.name for i in inputs)),) if inputs else ()
        items.append(Item(
            f"tool-{t.name}", t.name, f"Needs declared by tasks/{t.name}.py (SETUP).",
            lambda s, t=t: _tool_status(s, t), inputs=inputs, actions=actions, tone="violet",
        ))
    return items


def _choice(fields: dict, key: str, choices, what: str) -> str:
    value = fields.get(key)
    if not isinstance(value, str) or value not in choices:
        raise BadRequest(f"{what} must be one of the offered choices")
    return value


_HHMM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
SCHEDULE_FIELDS = {"kind", "day", "time", "hours", "deadline_hours", "model", "timeout_minutes"}


def schedule_block(name: str, fields: dict) -> tuple[str, str]:
    """(cadence, the TOML lines of a [tasks.<name>] table) from the form.
    Every value must be one the form offers (each is one load_schedule accepts)."""
    from tasks.dispatch import parse_cadence

    if not isinstance(fields, dict):
        raise BadRequest("fields must be an object")
    unknown = set(fields) - SCHEDULE_FIELDS
    if unknown:
        raise BadRequest(f"unknown field(s): {', '.join(sorted(map(str, unknown)))[:200]}")
    kind = _choice(fields, "kind", _dash.CADENCE_KINDS, "cadence")
    if kind == "every":
        cadence = f"every {_choice(fields, 'hours', _dash.HOUR_CHOICES, 'every N hours')} hours"
    else:
        at = fields.get("time")
        if not isinstance(at, str) or not _HHMM_RE.fullmatch(at):
            raise BadRequest("time must be HH:MM (24-hour)")
        cadence = f"daily {at}" if kind == "daily" else \
            f"weekly {_choice(fields, 'day', _dash.WEEKDAYS, 'day')} {at}"
    parse_cadence(cadence)  # the dispatcher's own parser has the last word
    deadline = _choice(fields, "deadline_hours", _dash.DEADLINE_CHOICES, "deadline_hours")
    model = _choice(fields, "model", _dash.SCHEDULE_MODEL_CHOICES, "model")
    timeout = _choice({"timeout_minutes": fields.get("timeout_minutes", "")}, "timeout_minutes",
                      _dash.TIMEOUT_CHOICES, "timeout_minutes")
    lines = [f"[tasks.{name}]", f'cadence = "{cadence}"', f"deadline_hours = {deadline}", f'model = "{model}"']
    if timeout:
        lines.append(f"timeout_minutes = {timeout}")
    return cadence, lines


def _append_lines(current: bytes | None, lines: list[str], header: str = "") -> str:
    """`current` (kept byte for byte) plus `lines` as a new block, in the
    file's own newline style."""
    text = current.decode("utf-8") if current is not None else header
    nl = "\r\n" if "\r\n" in text else "\n"
    if text and not text.endswith("\n"):
        text += nl
    if text:
        text += nl
    return text + nl.join(lines) + nl


def _new_task(s: Setup, name: object):
    t = s.dash.task(name)  # the catalog allowlist (BadRequest if unknown)
    if t.name == "dispatch":
        raise BadRequest("the dispatcher can't schedule itself")
    return t


def add_to_schedule(s: Setup, name: object, fields: object, *, confirm: bool = False) -> dict:
    """Append a [tasks.<name>] table to config/schedule.toml."""
    t = _new_task(s, name)
    cadence, lines = schedule_block(t.name, fields)
    if not confirm:
        raise NeedsConfirm(f"adds [tasks.{t.name}] ({cadence}) to config/schedule.toml (a backup is kept); confirm")
    path = s.config_path("schedule")
    lines[0] += "  # added from the dashboard " + datetime.now().strftime("%Y-%m-%d")

    def check(tmp: Path) -> None:
        from tasks.dispatch import load_schedule

        specs = {x.name: x for x in load_schedule(tmp, s.root / "tasks").tasks}
        if t.name not in specs or str(specs[t.name].cadence) != cadence:
            raise ValueError(f"tasks.{t.name} didn't load back as {cadence}")

    return _append_locked(s, "schedule", path, t.name, lines, check, section="tasks")


def mark_manual(s: Setup, name: object, *, confirm: bool = False) -> dict:
    """Append a [manual.<name>] table to config/dashboard.toml."""
    t = _new_task(s, name)
    if not t.new:
        raise Conflict(f"{t.name} isn't a new tool (it's scheduled or already manual-only)")
    if not confirm:
        raise NeedsConfirm(f"marks {t.name} manual-only in config/dashboard.toml; confirm")
    path = s.config_path("dashboard")
    lines = [f"[manual.{t.name}]", f'note = "marked manual-only from the dashboard {datetime.now():%Y-%m-%d}"']

    def check(tmp: Path) -> None:
        if t.name not in _dash.load_manual(tmp):
            raise ValueError(f"manual.{t.name} didn't load back")

    return _append_locked(s, "dashboard", path, t.name, lines, check, section="manual",
                          header=DASHBOARD_HEADER)


_BRIEFING_LOCK = threading.Lock()
_LOCAL_BRIEFINGS_HEADER = (
    "# Local briefing on/off switches (#95) -- gitignored, never the shipped\n"
    "# config/briefings/<name>.toml. The dashboard's Enable/Disable buttons\n"
    "# write this file; hand edits are fine too (name = true|false).\n"
)
# [briefings] header line, allowing a trailing comment (mirrors the shipped
# file's own [briefing] header pattern before #95 moved this off it). The
# match consumes the line's own newline (`\r?\n`) or stops at EOF (`\Z`) --
# never a bare `$`, which sits *before* the `\n` in a CRLF file and leaves a
# stray `\r` the rest of the pattern can't match past.
_BRIEFINGS_HEADER_RE = re.compile(r"^[ \t]*\[briefings\][ \t]*(?:#[^\r\n]*)?(?:\r?\n|\Z)", re.M)
# Any top-level table header -- used to find where the [briefings] section ends.
_SECTION_HEADER_RE = re.compile(r"^[ \t]*\[[^\]\r\n]+\][ \t]*(?:#[^\r\n]*)?(?:\r?\n|\Z)", re.M)


def _upsert_local_briefing_line(current: bytes | None, name: str, enabled: bool) -> str:
    """Set `name = true|false` inside config/local/briefings.toml's
    [briefings] table with a line-level edit, not a whole-file rewrite
    (#95 review finding 3): every other line -- including hand-written
    comments and the other briefings' own lines, in their original order --
    survives byte-for-byte. The file's own header says hand edits are fine,
    so a toggle must not clobber them."""
    value = "true" if enabled else "false"
    new_line = f"{name} = {value}"
    if current is None:
        return f"{_LOCAL_BRIEFINGS_HEADER}\n[briefings]\n{new_line}\n"
    text = current.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    m = _BRIEFINGS_HEADER_RE.search(text)
    if m is None:
        # No [briefings] table yet (a hand-edited file with only comments,
        # say): append one after everything else, untouched.
        sep = "" if not text or text.endswith(("\n", "\r\n")) else nl
        return f"{text}{sep}[briefings]{nl}{new_line}{nl}"
    body_start = m.end()  # already past the header line's own newline (or EOF)
    next_header = _SECTION_HEADER_RE.search(text, body_start)
    body_end = next_header.start() if next_header else len(text)
    body = text[body_start:body_end]
    line_re = re.compile(rf"^[ \t]*{re.escape(name)}[ \t]*=[^\r\n]*", re.M)
    lm = line_re.search(body)
    if lm is not None:
        new_body = body[:lm.start()] + new_line + body[lm.end():]
    else:
        sep = "" if not body or body.endswith(("\n", "\r\n")) else nl
        new_body = f"{body}{sep}{new_line}{nl}"
    head = text[:body_start]
    if head and not head.endswith(("\n", "\r\n")):
        head += nl  # the [briefings] header was the file's last, unterminated line
    return head + new_body + text[body_end:]


def set_briefing_enabled(s: Setup, name: object, enabled: bool, *, confirm: bool = False) -> dict:
    """Write `name = true|false` into config/local/briefings.toml's
    [briefings] table (#95): a gitignored local override that always wins
    over config/briefings/<name>.toml's own `enabled`
    (`lmsa.briefing.config.effective_enabled`). The shipped file is never
    touched. Confirm, backup, atomic write and load-back validation all
    still apply, same as before #95. The write itself is a line-level edit
    (`_upsert_local_briefing_line`) so hand-written comments and the other
    briefings' lines survive untouched."""
    from lmsa.briefing import config as bconfig

    directory = s.config_dir / "briefings"
    if not isinstance(name, str) or not bconfig.NAME_RE.fullmatch(name) or name not in bconfig.names(directory):
        raise BadRequest("unknown briefing")
    value = "true" if enabled else "false"
    path = bconfig.local_overrides_path(directory)
    if not confirm:
        raise NeedsConfirm(f"sets {name} = {value} in config/local/briefings.toml (a backup is kept); confirm")
    if not _BRIEFING_LOCK.acquire(timeout=30):
        raise Conflict("another briefing save is still running; try again")
    try:
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None
        except OSError as exc:
            raise BadRequest(f"config/local/{path.name} can't be read ({type(exc).__name__}); fix it first") from None
        if current is not None:
            # Validate the existing file's shape before touching it -- a
            # malformed local file is refused, never silently rewritten.
            try:
                data = tomllib.loads(current.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise BadRequest(
                    f"config/local/{path.name} can't be read ({type(exc).__name__}); fix it by hand first") from None
            extra = set(data) - {"briefings"}
            if extra:
                raise BadRequest(f"config/local/{path.name}: unknown key {sorted(extra)[0]!r} "
                                 "(only [briefings] is used)")
            table = data.get("briefings", {})
            if not isinstance(table, dict):
                raise BadRequest(f"config/local/{path.name}: briefings must be a table of name = true|false")
            for k, v in table.items():
                if not isinstance(k, str) or not bconfig.NAME_RE.fullmatch(k) or not isinstance(v, bool):
                    raise BadRequest(f"config/local/{path.name}: briefings.{k} must be true or false")
        new_text = _upsert_local_briefing_line(current, name, enabled)

        def validate(tmp: Path) -> None:
            _, problem = bconfig.local_overrides_status(tmp)
            if problem:  # never save a file that would silently drop the owner's switches
                raise ValueError(f"the updated config/local/briefings.toml wouldn't read back cleanly: {problem}")
            cfg = bconfig.load(name, directory, local_path=tmp)
            if cfg.enabled is not enabled:
                raise ValueError(f"briefing.enabled didn't load back as {value}")

        base = _sha(current) if current is not None else "new"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            result = _config_save_locked(s, "briefings-local", path, new_text, base, validate=validate)
        except OSError as exc:
            raise BadRequest(
                f"couldn't write config/local/{path.name} ({type(exc).__name__}); nothing was changed") from None
    finally:
        _BRIEFING_LOCK.release()
    backup = result["message"].partition("; backup: ")[2]
    note = f" (backup: {backup})" if backup else ""
    return {"message": f"config/local/briefings.toml: {name} = {value}{note}", "fill": result["fill"]}


# ---------------------------------------------------------------------------
# Master Report (#105): the Reports tab's local enabled/send switches, and
# the structured save of config/master_report.toml itself.
# ---------------------------------------------------------------------------

def _mr_scope(s: Setup) -> dict:
    """tasks_dir/briefings_dir/schedule_path for tasks.master_report's
    parse_config/load_config, scoped to *this* Setup's own root -- never
    the module's own install-location defaults (#105: this dashboard's
    config may not be the one it's installed alongside, e.g. in tests)."""
    return {"tasks_dir": s.root / "tasks", "briefings_dir": s.config_dir / "briefings",
            "schedule_path": s.config_dir / "schedule.toml"}


_MASTER_LOCAL_LOCK = threading.Lock()
_MASTER_LOCAL_KEYS = ("enabled", "send")  # "to" is never edited from the dashboard: v1 always sends to self.
# [master.<name>] header, allowing a trailing comment -- mirrors _BRIEFINGS_HEADER_RE.
_MASTER_LOCAL_HEADER_RE_TMPL = r"^[ \t]*\[master\.{name}\][ \t]*(?:#[^\r\n]*)?(?:\r?\n|\Z)"
_LOCAL_MASTER_HEADER = (
    "# Per-machine master-report switches (#105) -- gitignored, never the\n"
    "# tracked config/master_report.toml. The Reports tab's enable/disable and\n"
    "# send/dry-run controls write this file; hand edits (enabled/send/to under\n"
    "# [master.<name>]) are fine too.\n"
)


def _upsert_master_local_lines(current: bytes | None, name: str, values: dict[str, bool]) -> str:
    """Set every key in `values` (enabled/send: true|false) inside
    config/local/master_report.toml's [master.<name>] table, as a
    line-level edit: every other line -- other masters' overrides,
    hand-written comments, key order -- survives byte for byte. Creates the
    [master.<name>] table (and the file) if it doesn't exist yet."""
    text = current.decode("utf-8") if current is not None else ""
    nl = "\r\n" if "\r\n" in text else "\n"
    new_lines = [f"{k} = {'true' if v else 'false'}" for k, v in values.items()]
    header_re = re.compile(_MASTER_LOCAL_HEADER_RE_TMPL.format(name=re.escape(name)), re.M)
    m = header_re.search(text)
    if m is None:
        sep = "" if not text or text.endswith(("\n", "\r\n")) else nl
        prefix = text if text else _LOCAL_MASTER_HEADER + nl
        block = f"[master.{name}]{nl}" + "".join(f"{ln}{nl}" for ln in new_lines)
        return f"{prefix}{sep}{block}"
    body_start = m.end()
    next_header = _SECTION_HEADER_RE.search(text, body_start)
    body_end = next_header.start() if next_header else len(text)
    body = text[body_start:body_end]
    for k, v in values.items():
        line_re = re.compile(rf"^[ \t]*{re.escape(k)}[ \t]*=[^\r\n]*", re.M)
        new_line = f"{k} = {'true' if v else 'false'}"
        lm = line_re.search(body)
        if lm is not None:
            body = body[:lm.start()] + new_line + body[lm.end():]
        else:
            sep = "" if not body or body.endswith(("\n", "\r\n")) else nl
            body = f"{body}{sep}{new_line}{nl}"
    head = text[:body_start]
    if head and not head.endswith(("\n", "\r\n")):
        head += nl
    return head + body + text[body_end:]


def local_master_report_path(s: Setup) -> Path:
    return s.config_dir / "local" / "master_report.toml"


def set_master_local(s: Setup, name: object, values: object, *, confirm: bool = False) -> dict:
    """Set enabled/send for one master in config/local/master_report.toml
    (#105): the untracked, per-machine override the tracked file's own
    header says these two belong in. Same shape as `set_briefing_enabled`:
    confirm, a light shape-check of whatever's already there, a line-level
    upsert, then the usual locked/backed-up/atomic save with a load-back
    check -- just against tasks.master_report.load_config's overrides
    instead of lmsa.briefing.config's."""
    from tasks import master_report as mr

    if not isinstance(name, str) or not mr.NAME_RE.fullmatch(name):
        raise BadRequest("unknown master")
    try:
        masters = mr.load_config(path=s.config_path("master_report"), **_mr_scope(s))
    except mr.ConfigError as exc:
        raise BadRequest(f"config/master_report.toml is invalid: {'; '.join(exc.errors)}") from None
    if not any(m.name == name for m in masters):
        raise BadRequest("unknown master")
    if not isinstance(values, dict) or not values:
        raise BadRequest("nothing to set")
    unknown = set(values) - set(_MASTER_LOCAL_KEYS)
    if unknown:
        raise BadRequest(f"only {', '.join(_MASTER_LOCAL_KEYS)} can be set here (not {sorted(unknown)[0]!r})")
    for k, v in values.items():
        if not isinstance(v, bool):
            raise BadRequest(f"{k} must be true or false")
    summary = ", ".join(f"{k} = {v}" for k, v in values.items())
    if not confirm:
        raise NeedsConfirm(f"sets {summary} for {name!r} in config/local/master_report.toml; confirm")
    path = local_master_report_path(s)
    if not _MASTER_LOCAL_LOCK.acquire(timeout=30):
        raise Conflict("another master-report save is still running; try again")
    try:
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None
        except OSError as exc:
            raise BadRequest(f"config/local/{path.name} can't be read ({type(exc).__name__}); fix it first") from None
        if current is not None:
            try:
                data = tomllib.loads(current.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise BadRequest(
                    f"config/local/{path.name} can't be read ({type(exc).__name__}); fix it by hand first") from None
            extra = set(data) - {"master"}
            if extra:
                raise BadRequest(f"config/local/{path.name}: unknown key {sorted(extra)[0]!r} "
                                 "(only [master.<name>] is used)")
            table = data.get("master", {})
            if not isinstance(table, dict):
                raise BadRequest(f"config/local/{path.name}: master must be a table of [master.<name>] tables")
        new_text = _upsert_master_local_lines(current, name, values)

        def validate(tmp: Path) -> None:
            try:
                updated = mr.load_config(path=s.config_path("master_report"), local_path=tmp, **_mr_scope(s))
            except mr.ConfigError as exc:
                raise ValueError("; ".join(exc.errors)) from None
            m = next((x for x in updated if x.name == name), None)
            if m is None:
                raise ValueError(f"master {name!r} didn't load back")
            for k, v in values.items():
                if getattr(m, k) is not v:
                    raise ValueError(f"{k} didn't load back as {v}")

        base = _sha(current) if current is not None else "new"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            result = _config_save_locked(s, "master-report-local", path, new_text, base, validate=validate)
        except OSError as exc:
            raise BadRequest(
                f"couldn't write config/local/{path.name} ({type(exc).__name__}); nothing was changed") from None
    finally:
        _MASTER_LOCAL_LOCK.release()
    backup = result["message"].partition("; backup: ")[2]
    note = f" (backup: {backup})" if backup else ""
    return {"message": f"config/local/master_report.toml: {name}: {summary}{note}", "fill": result["fill"]}


# -- structural save: config/master_report.toml itself ----------------------

_MASTER_FIELD_MAX = 4000  # a sane upper bound before this ever reaches parse_config's own checks
_MASTER_HEADER_RE = re.compile(r"^\[\[master\]\]", re.M)


def _toml_str(value: str) -> str:
    """A TOML basic string literal for `value`: backslash, double-quote and
    every control character are escaped, so whatever the caller puts in
    can never end the string early or inject another key/table -- at worst
    it makes tasks.master_report.parse_config reject the result, which
    never writes anything either way."""
    out = ['"']
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _is_utf8_encodable(value: str) -> bool:
    """False for a lone UTF-16 surrogate (\\ud800-\\udfff unpaired): valid
    inside a Python str, and inside a JSON string (whose \\uXXXX escapes
    are UTF-16 code units, not Unicode scalar values, so JSON itself never
    rejects one), but not encodable as UTF-8 -- and `_config_save_locked`
    has to encode this text eventually."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _str_field(m: dict, where: str, key: str) -> str:
    v = m.get(key)
    if not isinstance(v, str) or len(v) > _MASTER_FIELD_MAX:
        raise BadRequest(f"{where}.{key} must be a short string")
    if not _is_utf8_encodable(v):
        raise BadRequest(f"{where}.{key} contains a character that isn't valid Unicode (a lone surrogate)")
    return v


def _bool_field(m: dict, where: str, key: str) -> bool:
    v = m.get(key)
    if not isinstance(v, bool):
        raise BadRequest(f"{where}.{key} must be true or false")
    return v


def _master_from_json(i: int, m: object) -> dict:
    """Shape-check one `masters[i]` entry from the Reports tab's save
    request -- not a semantic validator (tasks.master_report.parse_config,
    run against the serialized TOML, is the one truth for that); just
    enough to build valid TOML text out of arbitrary JSON without letting a
    huge or wrongly-typed value reach the serializer.

    `prev_name` (optional) is the name this master had when the Reports tab
    loaded it -- null for one added in this session. It's never written to
    TOML; `_serialize_master_report` uses it only to find the master's
    *previous* tracked entry (by old name, across a rename) so enabled/
    send/to can be carried forward, and `save_master_report` uses it to
    tell a rename from a delete when reconciling local overrides (#105
    review)."""
    where = f"masters[{i}]"
    from tasks.master_report import MAX_SECTIONS

    if not isinstance(m, dict):
        raise BadRequest(f"{where}: must be an object")
    prev_name = m.get("prev_name")
    if prev_name is not None and (not isinstance(prev_name, str) or len(prev_name) > _MASTER_FIELD_MAX
                                  or not _is_utf8_encodable(prev_name)):
        raise BadRequest(f"{where}.prev_name must be a string or null")
    name, title, send_at = _str_field(m, where, "name"), _str_field(m, where, "title"), _str_field(m, where, "send_at")
    llm_intro, skip_empty = _bool_field(m, where, "llm_intro"), _bool_field(m, where, "skip_empty")
    max_age_hours = m.get("max_age_hours")
    if not isinstance(max_age_hours, (int, float)) or isinstance(max_age_hours, bool):
        raise BadRequest(f"{where}.max_age_hours must be a number")
    sections = m.get("sections")
    if not isinstance(sections, list) or len(sections) > MAX_SECTIONS:
        raise BadRequest(f"{where}.sections must be a list")
    out_sections = []
    for j, sec in enumerate(sections):
        sw = f"{where}.sections[{j}]"
        if not isinstance(sec, dict):
            raise BadRequest(f"{sw}: must be an object")
        task, mode = _str_field(sec, sw, "task"), _str_field(sec, sw, "mode")
        max_items = sec.get("max_items")
        if not isinstance(max_items, int) or isinstance(max_items, bool):
            raise BadRequest(f"{sw}.max_items must be an integer")
        briefing = sec.get("briefing")
        if briefing is not None and (not isinstance(briefing, str) or len(briefing) > _MASTER_FIELD_MAX
                                     or not _is_utf8_encodable(briefing)):
            raise BadRequest(f"{sw}.briefing must be a string or null")
        out_sections.append({"task": task, "mode": mode, "max_items": max_items, "briefing": briefing})
    return {"name": name, "prev_name": prev_name, "title": title, "send_at": send_at, "llm_intro": llm_intro,
            "max_age_hours": max_age_hours, "skip_empty": skip_empty, "sections": out_sections}


def _extract_master_report_header(current: bytes | None) -> str:
    """Whatever precedes the first `[[master]]` table in the current file
    (the tracked file's whole header comment block) -- kept verbatim on
    save, since the structural editor only ever rewrites the `[[master]]`
    entries themselves. Comments *inside* or *between* master entries are
    not preserved: the save always rewrites those wholesale from the
    edited structure."""
    if not current:
        return ""
    try:
        text = current.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    m = _MASTER_HEADER_RE.search(text)
    return text[:m.start()] if m else ""


def _serialize_master_report(masters: list[dict], current: bytes | None, old_by_name: dict) -> str:
    """`masters` (already shape-checked by `_master_from_json`) as TOML
    text: the original file's header comment block, followed by one
    `[[master]]` table per entry, in the given order.

    `enabled`/`send`/`to` aren't edited by this UI -- but if the master
    already existed (found in `old_by_name` by `prev_name`, or by `name`
    for one that wasn't renamed), whatever the tracked file already said
    for it is carried through unchanged, never silently reset to the
    dataclass defaults by an unrelated structural edit (#105 review). Only
    a genuinely new master (nothing found) is left to default."""
    header = _extract_master_report_header(current)
    blocks = []
    for m in masters:
        old = old_by_name.get(m.get("prev_name") or m["name"])
        lines = ["[[master]]", f"name = {_toml_str(m['name'])}", f"title = {_toml_str(m['title'])}",
                 f"send_at = {_toml_str(m['send_at'])}"]
        if old is not None:
            lines.append(f"enabled = {'true' if old.enabled else 'false'}")
            lines.append(f"send = {'true' if old.send else 'false'}")
            if old.to:
                lines.append("to = [" + ", ".join(_toml_str(a) for a in old.to) + "]")
        lines += [
            f"llm_intro = {'true' if m['llm_intro'] else 'false'}",
            f"max_age_hours = {m['max_age_hours']!r}" if isinstance(m["max_age_hours"], float)
            else f"max_age_hours = {int(m['max_age_hours'])}",
            f"skip_empty = {'true' if m['skip_empty'] else 'false'}",
        ]
        for s in m["sections"]:
            lines += ["", "  [[master.section]]", f"  task = {_toml_str(s['task'])}",
                      f"  mode = {_toml_str(s['mode'])}", f"  max_items = {int(s['max_items'])}"]
            if s["briefing"]:
                lines.append(f"  briefing = {_toml_str(s['briefing'])}")
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks)
    return header + body + ("\n" if body else "")


def _drop_orphaned_master_local_tables(current: bytes | None, keep_names: set) -> str:
    """`current` (config/local/master_report.toml) with every
    `[master.<name>]` table dropped whose name isn't in `keep_names` --
    run before a structural save so a master that's being renamed or
    deleted never leaves its per-machine enabled/send/to override behind
    for `load_config` to choke on later (#105 review: a scheduled run, or
    this tab's own next GET, would otherwise see `masters: []`).

    Deliberately a *drop*, not a rename-in-place, even for a renamed
    master: the two files can't be written as one atomic transaction, and
    only *removing* an override is safe to do first. Removing an override
    can never make the *old* names invalid (a master with no local
    override just gets the tracked defaults), so this write -- done
    before the tracked file changes at all -- has no window in which the
    on-disk pair fails to load; a renamed master simply loses its
    per-machine override and reverts to the tracked defaults, one click to
    reapply. Malformed existing content is left untouched -- this never
    corrects a file it can't confidently parse."""
    if not current:
        return ""
    text = current.decode("utf-8", errors="replace")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return text
    table = data.get("master")
    if not isinstance(table, dict):
        return text
    for name in [n for n in table if isinstance(n, str) and n not in keep_names]:
        header_re = re.compile(_MASTER_LOCAL_HEADER_RE_TMPL.format(name=re.escape(name)), re.M)
        m = header_re.search(text)
        if m is None:
            continue
        next_header = _SECTION_HEADER_RE.search(text, m.end())
        end = next_header.start() if next_header else len(text)
        text = text[:m.start()] + text[end:]
    return text


def _check_master_local_shape(tmp: Path) -> None:
    """tomllib, then just enough shape-checking to refuse a local file this
    module doesn't understand -- shared by `set_master_local`'s own edits
    and `save_master_report`'s orphan cleanup."""
    data = tomllib.loads(tmp.read_text(encoding="utf-8"))
    extra = set(data) - {"master"}
    if extra:
        raise ValueError(f"unknown key {sorted(extra)[0]!r} (only [master.<name>] is used)")
    table = data.get("master", {})
    if not isinstance(table, dict):
        raise ValueError("master must be a table of [master.<name>] tables")


def save_master_report(s: Setup, masters: object, base: object) -> dict:
    """The Reports tab's "save changes": `masters` (a JSON list mirroring
    tasks.master_report.MasterConfig, minus enabled/send/to, which this UI
    never edits) is serialized to TOML and validated with
    tasks.master_report.parse_config *before* the lock on the tracked file
    is even taken, so a structural mistake comes back as that validator's
    own per-field sentences and nothing is written.

    Before that: any `[master.<name>]` table in config/local/master_report
    .toml whose name no longer appears among `masters` is dropped (a
    rename or a delete, #105 review) through its own locked/backed-up/
    atomic write, and the *combined* tracked+local state -- not the
    tracked file alone -- is what gets validated, so a save can never
    leave a pair that `load_config` chokes on."""
    from tasks import master_report as mr

    if not isinstance(masters, list):
        raise BadRequest("masters must be a list")
    if not isinstance(base, str) or not base:
        raise BadRequest("load the report list first")
    parsed = [_master_from_json(i, m) for i, m in enumerate(masters)]
    new_names = {m["name"] for m in parsed}

    path = s.config_path("master_report")
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    try:
        old_by_name = {m.name: m for m in mr.parse_config(current.decode("utf-8"), **_mr_scope(s))} \
            if current is not None else {}
    except (UnicodeDecodeError, mr.ConfigError):
        # The tracked file is already broken -- this save may well be the
        # fix. Nothing to carry forward from a file that doesn't parse.
        old_by_name = {}

    local_path = local_master_report_path(s)
    try:
        old_local = local_path.read_bytes()
    except FileNotFoundError:
        old_local = None
    new_local_text = _drop_orphaned_master_local_tables(old_local, new_names)
    old_local_text = old_local.decode("utf-8", errors="replace") if old_local else ""

    # Validate and check the base BEFORE any write, so a save rejected with
    # a 400 or a 409 never costs a local override (#105 PR review).
    text = _serialize_master_report(parsed, current, old_by_name)
    try:
        mr.parse_config(text, overrides=(new_local_text or None), **_mr_scope(s))
    except mr.ConfigError as exc:
        raise BadRequest("; ".join(exc.errors)) from None
    if not s.config_locks["master_report"].acquire(timeout=30):
        raise Conflict("another save of master_report.toml is still running; try again")
    try:
        try:
            now = path.read_bytes()
        except FileNotFoundError:
            now = None
        if (now is None and base != "new") or (now is not None and _sha(now) != base) or now != current:
            raise Conflict("master_report.toml changed since you loaded it; load it again")
        if new_local_text != old_local_text:
            if not _MASTER_LOCAL_LOCK.acquire(timeout=30):
                raise Conflict("another master-report save is still running; try again")
            try:
                local_base = _sha(old_local) if old_local is not None else "new"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                _config_save_locked(s, "master-report-local", local_path, new_local_text, local_base,
                                    validate=_check_master_local_shape)
            finally:
                _MASTER_LOCAL_LOCK.release()
        return _config_save_locked(s, "master_report", path, text, base)
    finally:
        s.config_locks["master_report"].release()


DASHBOARD_HEADER = ("# Dashboard settings (python -m tasks.dashboard).\n"
                    "# [manual.<task>] marks a task manual-only (see README).\n")


def _append_locked(s: Setup, file: str, path: Path, task: str, lines: list[str],
                   check: Callable[[Path], None], *, section: str, header: str = "") -> dict:
    if not s.config_locks[file].acquire(timeout=30):
        raise Conflict(f"another save of {file}.toml is still running; try again")
    try:
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            if not header:
                raise BadRequest(f"config/{file}.toml doesn't exist") from None
            current = None
        except OSError as exc:
            raise BadRequest(f"can't read {file}.toml ({type(exc).__name__})") from None
        try:
            data = tomllib.loads(current.decode("utf-8")) if current is not None else {}
        except (UnicodeDecodeError, ValueError) as exc:
            raise BadRequest(f"{file}.toml is invalid now ({type(exc).__name__}); fix it in Config files first") from None
        table = data.get(section, {})
        if isinstance(table, dict) and task in table:
            raise Conflict(f"{task} is already in {file}.toml ([{section}.{task}])")
        text = _append_lines(current, lines, header)
        base = _sha(current) if current is not None else "new"
        try:
            return _config_save_locked(s, file, path, text, base, check=check)
        except OSError as exc:
            raise BadRequest(f"couldn't write {file}.toml ({type(exc).__name__}); nothing was changed") from None
    finally:
        s.config_locks[file].release()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _input_html(item: Item, i: Input) -> str:
    iid = esc(f"s-{item.id}-{i.name}")
    name = esc(i.name)
    label = f'<span class="flag">{esc(i.label or i.name)}</span>'
    ph = f' placeholder="{esc(i.placeholder)}"' if i.placeholder else ""
    if i.kind == "hidden":
        return f'<input type="hidden" name="{name}">'
    if i.kind == "select":
        opts = "".join(f'<option value="{esc(o)}">{esc(o)}</option>' for o in i.options)
        live = "" if i.options else " data-live-choices"
        return (f'<label class="field" for="{iid}">{label}<select id="{iid}" name="{name}"{live}>{opts}</select>'
                "</label>")
    if i.kind == "textarea":
        return (f'<label class="field wide" for="{iid}">{label}<textarea id="{iid}" name="{name}" rows="14" '
                f'spellcheck="false" maxlength="{MAX_CONFIG}"></textarea></label>')
    kind = ('type="password" autocomplete="new-password" data-secret' if i.kind == "secret"
            else 'type="text" autocomplete="off"')
    return (f'<label class="field" for="{iid}">{label}<input {kind} id="{iid}" name="{name}" '
            f'maxlength="{MAX_VALUE}"{ph}></label>')


def _item_html(item: Item) -> str:
    optional = '<span class="tag t-muted">optional</span>' if item.optional else ""
    note = f'<p class="setup-note">{esc(item.note)}</p>' if item.note else ""
    buttons = "".join(
        f'<button class="btn btn-sm{" btn-hot" if a.confirm else ""}" type="button" data-action="{esc(a.id)}" '
        f'data-uses="{esc(",".join(a.uses))}">{esc(a.label)}</button>'
        for a in item.actions
    )
    form = ""
    if item.actions:
        fields = "".join(_input_html(item, i) for i in item.inputs)
        fields_html = f'<div class="fields">{fields}</div>' if fields else ""
        form = (f'<form class="setup-form" novalidate autocomplete="off">{fields_html}'
                f'<div class="launcher-actions">{buttons}<span class="launcher-msg" role="status"></span></div>'
                '</form><pre class="setup-output" hidden></pre>')
    wide = " wide" if item.wide else ""
    sid = f"setup-{item.id}"
    return (
        f'<article class="launcher setup-item{wide} t-{esc(item.tone)}" data-item="{esc(item.id)}">'
        f'<header class="launcher-head"><h3>{esc(item.title)}</h3><div class="tags">{optional}'
        '<span class="setup-state st-none" data-state>checking</span></div>'
        f'{collapse_toggle_html(sid)}</header>'
        f'<div class="sec-body" id="body-{esc(sid)}">'
        f'<p class="launcher-doc">{esc(item.doc)}</p>'
        '<p class="setup-reason" data-reason>&hellip;</p><ul class="setup-hints" data-hints></ul>'
        f"{note}{form}</div></article>"
    )


def render_section(setup: Setup, idx: int) -> str:
    items = setup.items()
    tools = [i for i in items if i.id.startswith("tool-")]
    general = [i for i in items if not i.id.startswith("tool-")]
    tool_html = ""
    if tools:
        tool_html = (
            f'<details class="tool-needs" open><summary>tool needs <span class="count">{len(tools)}</span>'
            '<span class="hint">declared by each task&#39;s SETUP</span></summary>'
            f'<div class="setups">{"".join(_item_html(i) for i in tools)}</div></details>'
        )
    return (
        f'<section class="t-warn" id="setup-sec"><h2><span class="idx">{idx:02d} //</span>Setup'
        f'<span class="count">{len(items)}</span>'
        '<button class="btn btn-ghost btn-sm" type="button" id="setup-refresh">recheck</button>'
        + collapse_toggle_html("setup-sec") + '</h2>'
        '<div class="sec-body" id="body-setup-sec">'
        f'<div class="setups">{"".join(_item_html(i) for i in general)}</div>{tool_html}</div></section>'
    )


SETUP_JS = r"""
(() => {
  const L = window.lmsaLauncher; if (!L) return;
  const { api, el, $ } = L;
  const STATE_TEXT = { ok: 'ok', missing: 'missing', warning: 'warning' };
  async function copy(text, btn) {
    try { await navigator.clipboard.writeText(text); btn.textContent = 'copied'; }
    catch (e) { btn.textContent = 'select + copy'; }
    setTimeout(() => { btn.textContent = 'copy'; }, 1500);
  }
  async function refresh() {
    document.querySelectorAll('[data-item] [data-state]').forEach(s => { s.textContent = 'checking'; s.className = 'setup-state st-none'; });
    const { ok, j } = await api('/api/setup').catch(() => ({ ok: false, j: {} }));
    if (!ok) { document.querySelectorAll('[data-item] [data-reason]').forEach(r => { r.textContent = (j && j.error) || 'setup check failed'; }); return; }
    for (const it of j.items) {
      const card = $(`[data-item="${CSS.escape(it.id)}"]`); if (!card) continue;
      const st = $('[data-state]', card); st.textContent = STATE_TEXT[it.state] || it.state; st.className = 'setup-state st-' + it.state;
      $('[data-reason]', card).textContent = it.reason;
      const hints = $('[data-hints]', card);
      hints.replaceChildren(...it.hints.map(h => {
        const li = el('li', h.command ? 'cmd' : '');
        if (h.command) { const b = el('button', 'btn btn-ghost btn-sm', 'copy'); b.type = 'button'; b.addEventListener('click', () => copy(h.text, b)); li.append(el('code', '', h.text), b); }
        else li.textContent = h.text;
        return li;
      }));
      for (const [name, opts] of Object.entries(it.choices || {})) {
        const sel = $(`select[name="${CSS.escape(name)}"][data-live-choices]`, card); if (!sel) continue;
        const keep = sel.value;
        sel.replaceChildren(...(opts.length ? opts : ['']).map(o => { const op = el('option', '', o || '(none)'); op.value = o; return op; }));
        if (opts.includes(keep)) sel.value = keep;
        sel.disabled = !opts.length;
        card.querySelectorAll('button[data-uses]').forEach(b => { if (b.dataset.uses.split(',').includes(name)) b.disabled = !opts.length; });
      }
    }
    // Overview's attention list and the Setup tab's badge (#100): items this
    // check just found missing, reusing the id -> title already on the card.
    const missing = j.items.filter(it => it.state === 'missing');
    const list = $('#attn-list');
    if (list) {
      list.querySelectorAll('.attn-setup').forEach(li => li.remove());
      for (const it of missing) {
        const card = $(`[data-item="${CSS.escape(it.id)}"]`);
        const title = card ? $('h3', card).textContent : it.id;
        const li = el('li', 'attn-setup');
        const a = el('a', '', title + ' needs setup'); a.href = '#setup';
        li.append(a); list.append(li);
      }
      const callout = $('#attn-callout');
      if (callout) callout.hidden = !list.children.length;
    }
    if (window.lmsaTabs) {
      const base = parseInt(document.body.dataset.attnBase || '0', 10);
      window.lmsaTabs.setBadge('overview', base + missing.length);
      window.lmsaTabs.setBadge('setup', missing.length);
    }
  }
  $('#setup-refresh').addEventListener('click', refresh);
  document.querySelectorAll('form.setup-form').forEach(form => {
    const card = form.closest('[data-item]'), msg = $('.launcher-msg', form), out = card.querySelector('.setup-output');
    form.addEventListener('submit', ev => ev.preventDefault());
    form.querySelectorAll('button[data-action]').forEach(btn => {
      const label = btn.textContent; let armed = 0;
      const disarm = () => { armed = 0; btn.classList.remove('armed'); btn.textContent = label; };
      btn.addEventListener('click', async () => {
        const fields = {};
        for (const name of (btn.dataset.uses || '').split(',').filter(Boolean)) {
          const input = form.elements.namedItem(name); if (input) fields[name] = input.value;
        }
        const confirm = armed > Date.now();
        btn.disabled = true;
        const res = await api('/api/setup/action', { item: card.dataset.item, action: btn.dataset.action, fields, confirm }).catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
        btn.disabled = false; disarm();
        form.querySelectorAll('[data-secret]').forEach(i => { i.value = ''; });  // write-only: never kept
        const { ok, status, j } = res;
        if (status === 428) {
          armed = Date.now() + 8000; btn.classList.add('armed'); btn.textContent = 'confirm — ' + label;
          msg.textContent = j.error; msg.className = 'launcher-msg hot'; setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100); return;
        }
        if (status === 403) { msg.textContent = L.RESTARTED; msg.className = 'launcher-msg err'; return; }
        if (!ok) { msg.textContent = j.error || 'failed'; msg.className = 'launcher-msg err'; return; }
        msg.textContent = (j.warning ? '⚠ ' + j.warning + ' · ' : '') + (j.message || 'done');
        msg.className = 'launcher-msg' + (j.error || j.warning ? ' hot' : ' ok');
        if (out) { out.hidden = !j.output; out.textContent = j.output || ''; }
        for (const [name, value] of Object.entries(j.fill || {})) {
          const input = form.elements.namedItem(name); if (input) input.value = value;
        }
        if (j.run) {
          L.chip(j.run.id, j.run.task); L.show(j.run.id);
          if (window.lmsaTabs) window.lmsaTabs.openDock();
        }
        refresh();
      });
    });
  });
  refresh();
})();
"""


# ---------------------------------------------------------------------------
# Voice memos (#62): faster-whisper installed? CUDA available?
# ---------------------------------------------------------------------------

def voice_status(s: Setup) -> Status:
    """Cheap probes only; each degrades to a warning, never an exception."""
    from lmsa import voice

    if not voice.whisper_installed():
        return Status("missing", "faster-whisper isn't installed (the optional [voice] extra)",
                      [Hint(voice.INSTALL_COMMAND, command=True)])
    parts = [f"faster-whisper installed (model {voice.whisper_model()}, device {voice.whisper_device()})"]
    hints: list[Hint] = []
    warn = False
    cuda = voice.cuda_devices()
    if cuda is None:
        warn = True
        parts.append("can't tell whether CUDA is available (CTranslate2 didn't answer)")
    elif cuda == 0:
        warn = True
        parts.append("no CUDA device: transcribes on the CPU (slower)")
    else:
        parts.append(f"CUDA available ({cuda} device{'s' if cuda > 1 else ''})")
    folder = voice.recordings_dir()
    if folder.is_dir():
        parts.append(f"recordings folder {folder} present")
    else:
        warn = True
        parts.append(f"recordings folder {folder} not found")
        hints.append(Hint("create it, or set LMSA_RECORDINGS_DIR"))
    return Status("warning" if warn else "ok", "; ".join(parts), hints)


def voice_items(s: Setup) -> list[Item]:
    return [Item("voice", "Voice memos (faster-whisper)",
                 "Local speech-to-text for tasks.voice_memos / tasks.voice_transcribe.",
                 voice_status, optional=True)]


register(voice_items)
