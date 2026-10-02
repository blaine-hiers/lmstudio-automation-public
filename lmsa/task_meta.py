"""What a task module declares -- its argparse options and its `SETUP` block
-- read without importing it (#38).

Importing a task runs its top-level code, so this only `ast`-parses the
source. Only literal values count: anything else is skipped and reported in
`warnings`, never evaluated. An oversized source or a syntax error gives an
empty result with a warning, so the dashboard still shows the tile.

Conventions a task can use:
  - argparse `add_argument(...)` calls with literal arguments become launcher
    options (flag, `type=int`, `action="store_true"`, `help`, `default`,
    `metavar`, positionals);
  - a flag named --apply/--undo/--retry/--send/--delete/--force, or whose
    help starts with "[writes]", is a write flag: a run using it needs a confirm;
  - `SETUP = {"secrets": [...], "env": [...], "config": [...], "commands": [...],
    "model": "chat", "searxng": True}` lists what the tool needs (all keys optional).
"""

from __future__ import annotations

import ast
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

MAX_SOURCE_BYTES = 512 * 1024
MAX_ARGS = 40
MAX_HELP = 160
MAX_WARNINGS = 12
MAX_SETUP_ITEMS = 20
MAX_SETUP_TEXT = 200
MAX_FULL_DOC = 4000  # the capabilities page shows the whole docstring, but bounded (#67)
WRITE_FLAG_NAMES = frozenset({"--apply", "--undo", "--retry", "--send", "--delete", "--force"})
WRITES_MARKER = "[writes]"
SETUP_LIST_KEYS = ("secrets", "env", "config", "commands")
SETUP_BOOL_KEYS = ("searxng",)
SETUP_KEYS = (*SETUP_LIST_KEYS, "model", *SETUP_BOOL_KEYS)

_OPTION_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_DEST_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_BOOL_ACTIONS = {"store_true", "store_false", "store_const", "count"}
_VALUE_ACTIONS = {None, "store", "append", "extend"}
_MISSING = object()


@dataclass(frozen=True)
class Arg:
    """One argparse argument. `flag` is the long option (else the first
    option string) or, for a positional, its dest. `options` is every option
    string the parser accepts for it."""
    flag: str
    kind: str  # positional | int | bool | text
    label: str = ""
    write: bool = False
    required: bool = False
    options: tuple[str, ...] = ()


@dataclass
class TaskMeta:
    doc: str = ""
    full_doc: str = ""  # the whole docstring (doc is only its first line), capped at MAX_FULL_DOC (#67)
    args: tuple[Arg, ...] = ()
    setup: dict | None = None  # None: the module declares no SETUP
    warnings: list[str] = field(default_factory=list)  # the module couldn't be read at all
    arg_warnings: list[str] = field(default_factory=list)
    setup_warnings: list[str] = field(default_factory=list)
    uses_gmail: bool = False
    # Every literal option string of every add_argument call that names a
    # write flag, found BEFORE the checks that decide what the tile offers:
    # a flag the tile skips (SUPPRESS, a custom action, **kw ...) is still
    # one argparse accepts in extra args, so it still needs a confirm.
    write_options: tuple[str, ...] = ()

    def warn(self, text: str, kind: str = "") -> None:
        bucket = {"arg": self.arg_warnings, "setup": self.setup_warnings}.get(kind, self.warnings)
        if len(bucket) < MAX_WARNINGS:
            bucket.append(text[:240])


_cache: dict[str, tuple[tuple[int, int], TaskMeta]] = {}
_cache_lock = threading.Lock()


def read_meta(path: Path) -> TaskMeta:
    """The module's metadata, cached by (mtime, size)."""
    try:
        st = path.stat()
    except OSError as exc:
        return TaskMeta(warnings=[f"can't read {path.name} ({type(exc).__name__})"])
    key, stamp = str(path), (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    meta = _read(path, st.st_size)
    with _cache_lock:
        if len(_cache) > 1024:
            _cache.clear()
        _cache[key] = (stamp, meta)
    return meta


def _read(path: Path, size: int) -> TaskMeta:
    if size > MAX_SOURCE_BYTES:
        return TaskMeta(warnings=[f"{path.name} is over {MAX_SOURCE_BYTES // 1024} KiB: options and SETUP not read"])
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_SOURCE_BYTES + 1)  # bounded even if it grew since the stat
    except OSError as exc:
        return TaskMeta(warnings=[f"can't read {path.name} ({type(exc).__name__})"])
    if len(data) > MAX_SOURCE_BYTES:
        return TaskMeta(warnings=[f"{path.name} is over {MAX_SOURCE_BYTES // 1024} KiB: options and SETUP not read"])
    source = data.decode("utf-8-sig", errors="replace")  # Python runs files with a BOM; read them too
    return parse_source(source)


def parse_source(source: str) -> TaskMeta:
    meta = TaskMeta(uses_gmail="lmsa.sources.gmail" in source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        meta.warn(f"can't parse the module: SyntaxError at line {exc.lineno}: options and SETUP not read")
        return meta
    except (ValueError, RecursionError, MemoryError) as exc:
        meta.warn(f"can't parse the module ({type(exc).__name__}): options and SETUP not read")
        return meta
    doc = (ast.get_docstring(tree) or "").strip()
    meta.doc = doc.splitlines()[0].strip() if doc else ""
    meta.full_doc = doc[:MAX_FULL_DOC]
    meta.write_options = _write_options(tree)
    meta.args = _args(tree, meta)
    meta.setup = _setup(tree, meta)
    return meta


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _literal(node: ast.AST | None, what: str, meta: TaskMeta, quiet: bool = False) -> object:
    if node is None:
        return _MISSING
    if isinstance(node, ast.Constant):
        return node.value
    if not quiet:
        meta.warn(f"{what} isn't a literal; ignored", "arg")
    return _MISSING


def is_write_option(names: list[str], help_text: object) -> bool:
    """A standard write name, or a literal help starting with the [writes] marker."""
    return any(n in WRITE_FLAG_NAMES for n in names) or (
        isinstance(help_text, str) and help_text.lstrip().lower().startswith(WRITES_MARKER))


def _write_options(tree: ast.AST) -> tuple[str, ...]:
    """Option strings of every write flag in any add_argument call: every
    literal name counts, whatever else the call does (non-literal names,
    **kw, custom actions, SUPPRESS are ignored here, never a reason to skip)."""
    out: dict[str, None] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value[:64] for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("-")]
        help_node = next((k.value for k in node.keywords if k.arg == "help"), None)
        help_text = help_node.value if isinstance(help_node, ast.Constant) else None
        if names and is_write_option(names, help_text):
            out.update(dict.fromkeys(names))
    return tuple(out)


def _args(tree: ast.AST, meta: TaskMeta) -> tuple[Arg, ...]:
    calls = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "add_argument"),
        key=lambda n: (n.lineno, n.col_offset),  # source order; ast.walk is breadth-first
    )
    out: dict[str, Arg] = {}
    for node in calls:
        arg = _one_arg(node, meta)
        if arg is None or arg.flag in out:
            continue
        if len(out) >= MAX_ARGS:
            meta.warn(f"more than {MAX_ARGS} arguments: the rest are left out", "arg")
            break
        out[arg.flag] = arg
    return tuple(out.values())


def _one_arg(node: ast.Call, meta: TaskMeta) -> Arg | None:
    line = getattr(node, "lineno", 0)
    names: list[str] = []
    for a in node.args:
        if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
            meta.warn(f"line {line}: add_argument with a non-literal name; skipped", "arg")
            return None
        names.append(a.value)
    if not names:
        return None
    kws: dict[str, ast.AST] = {}
    for kw in node.keywords:
        if kw.arg is None:
            meta.warn(f"line {line}: add_argument(**...) isn't read; skipped", "arg")
            return None
        kws[kw.arg] = kw.value
    positional = not names[0].startswith("-")
    if positional:
        if len(names) != 1 or not _DEST_RE.fullmatch(names[0]):
            meta.warn(f"line {line}: unusual positional {names[0][:40]!r}; skipped", "arg")
            return None
        flag = names[0]
    else:
        if not all(_OPTION_RE.fullmatch(n) for n in names):
            meta.warn(f"line {line}: unusual option name; skipped", "arg")
            return None
        flag = next((n for n in names if n.startswith("--")), names[0])

    action_node = kws.get("action")
    if isinstance(action_node, ast.Attribute) and action_node.attr == "BooleanOptionalAction":
        action: object = "store_true"
    else:
        action = _literal(action_node, f"{flag}: action=", meta)
        if action is _MISSING:
            if action_node is not None:
                return None  # unknown semantics: don't offer it
            action = None
    if action in ("help", "version"):
        return None
    if action not in _BOOL_ACTIONS and action not in _VALUE_ACTIONS:
        meta.warn(f"{flag}: action={str(action)[:30]!r} isn't supported; skipped", "arg")
        return None

    help_node = kws.get("help")
    if isinstance(help_node, ast.Attribute) and help_node.attr == "SUPPRESS":
        return None  # hidden on purpose
    help_text = _literal(help_node, f"{flag}: help=", meta)
    help_text = help_text if isinstance(help_text, str) else ""
    write = is_write_option(names, help_text)
    label = help_text.strip()
    if label.lower().startswith(WRITES_MARKER):
        label = label[len(WRITES_MARKER):].strip()
    if not label:
        metavar = _literal(kws.get("metavar"), "", meta, quiet=True)
        label = metavar if isinstance(metavar, str) else ""

    if action in _BOOL_ACTIONS:
        kind = "bool"
    elif positional:
        kind = "positional"
    else:
        type_node = kws.get("type")
        kind = "int" if isinstance(type_node, ast.Name) and type_node.id == "int" else "text"
        default = _literal(kws.get("default"), "", meta, quiet=True)
        if isinstance(default, (str, int, float)) and not isinstance(default, bool):
            shown = str(default)[:40]
            label = f"{label} (default {shown})" if label else f"default {shown}"

    required = False
    if positional:
        nargs = _literal(kws.get("nargs"), f"{flag}: nargs=", meta)
        required = nargs not in ("?", "*")
    else:
        req = _literal(kws.get("required"), f"{flag}: required=", meta)
        required = req is True and kind != "bool"
    return Arg(flag, kind, label[:MAX_HELP], write=write, required=required,
               options=tuple(names) if not positional else ())


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

def _setup(tree: ast.Module, meta: TaskMeta) -> dict | None:
    value = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SETUP" for t in stmt.targets):
            if value is not None:
                meta.warn("SETUP is assigned more than once: the last one counts", "setup")
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "SETUP" \
                and stmt.value is not None:
            if value is not None:
                meta.warn("SETUP is assigned more than once: the last one counts", "setup")
            value = stmt.value
    if value is None:
        return None
    if not isinstance(value, ast.Dict):
        meta.warn("SETUP isn't a dict literal; ignored", "setup")
        return {}
    out: dict[str, object] = {}
    for k, v in zip(value.keys, value.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            meta.warn("SETUP has a non-literal key; ignored", "setup")
            continue
        key = k.value
        if key not in SETUP_KEYS:
            meta.warn(f"SETUP key {key[:40]!r} isn't known (use {', '.join(SETUP_KEYS)}); ignored", "setup")
            continue
        try:
            lit = ast.literal_eval(v)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            meta.warn(f"SETUP[{key!r}] isn't a literal; ignored", "setup")
            continue
        if key == "model":
            if not isinstance(lit, str) or len(lit) > MAX_SETUP_TEXT:
                meta.warn("SETUP['model'] must be a string; ignored", "setup")
                continue
            out[key] = lit
            continue
        if key in SETUP_BOOL_KEYS:
            if not isinstance(lit, bool):
                meta.warn(f"SETUP[{key!r}] must be true or false; ignored", "setup")
                continue
            out[key] = lit
            continue
        if isinstance(lit, str):
            lit = [lit]
        if not isinstance(lit, (list, tuple)) or not all(isinstance(x, str) for x in lit):
            meta.warn(f"SETUP[{key!r}] must be a list of strings; ignored", "setup")
            continue
        if len(lit) > MAX_SETUP_ITEMS or any(len(x) > MAX_SETUP_TEXT for x in lit):
            meta.warn(f"SETUP[{key!r}] is too long; ignored", "setup")
            continue
        out[key] = list(dict.fromkeys(lit))
    return out
