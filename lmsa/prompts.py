"""Prompts as files (#49): every task's system/user prompt lives under
``prompts/<task>/<name>.md`` instead of an inline string constant, so a
prompt edit shows up as a plain text diff and can be snapshot-tested.

    from lmsa.prompts import load, prompt_hash

    EXTRACT_SYSTEM = load("deadlines/extract_system")
    version = memo_version(MEMO_VERSION, prompt_hash("deadlines/extract_system"), SCHEMA, ...)

Naming convention (later tickets should follow this):
- `name` is the prompt's path under `prompts/`, without the `.md`
  extension -- e.g. `"travel/extract_system"` reads
  `prompts/travel/extract_system.md`. The first path segment is normally
  the task module's name (`tasks/travel.py` -> `travel/...`); a prompt that
  belongs to a sub-feature rather than a whole task (the shootout judge)
  gets its own segment instead, e.g. `"shootout_judge/system"`, so it can't
  collide with an existing directory's own file format (`prompts/shootout/`
  is `tasks.shootout`'s own prompt-library format and is untouched by this
  module).
- File basenames are the lowercased constant they replace
  (`EXTRACT_SYSTEM` -> `extract_system.md`).
- A prompt file is plain UTF-8 text, one paragraph, no front matter. It may
  end with a trailing newline (normal for a text file); `load()` strips
  exactly one.

Resolved relative to `lmsa.config.ROOT` (the repo root), the same way
`config/*.toml` is -- no packaging step is needed for an editable install,
since `pip install -e .` points straight back at the source tree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import ROOT

PROMPTS_DIR = ROOT / "prompts"


class PromptError(Exception):
    """A prompt file is missing, or a `{var}` in it has no matching argument."""


def _path(name: str) -> Path:
    return PROMPTS_DIR / f"{name}.md"


def _read_text(name: str) -> str:
    path = _path(name)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(f"prompts.load({name!r}): no such prompt file: {path}") from exc


class _StrictVars(dict):
    """`dict` for `str.format_map` that raises `PromptError` naming the
    prompt file, instead of a bare `KeyError`, for a `{placeholder}` with no
    matching keyword argument."""

    def __init__(self, name: str, path: Path, values: dict):
        super().__init__(values)
        self._name = name
        self._path = path

    def __missing__(self, key: str) -> str:
        raise PromptError(f"prompts.load({self._name!r}): missing template var {key!r} for {self._path}")


def load(name: str, /, **vars: object) -> str:
    """The prompt text at `prompts/<name>.md`, UTF-8, one trailing newline
    stripped. With no `**vars`, the file's text is returned verbatim (no
    `{...}` processing at all, so a prompt containing literal braces -- a
    JSON example, code -- is never mistaken for a template). With `**vars`,
    every `{var}` placeholder is substituted via `str.format_map`; a
    placeholder with no matching argument raises `PromptError` naming the
    file and the missing var.
    """
    text = _read_text(name).strip("\n")
    if not vars:
        return text
    return text.format_map(_StrictVars(name, _path(name), vars))


def prompt_hash(name: str) -> str:
    """Short, stable hash of `prompts/<name>.md`'s text (before any `{var}`
    substitution) -- 16 hex chars of its sha256. Fold this into a task's
    `memo_version(...)` so editing the prompt file invalidates that task's
    cached LLM results.

    Hashed as text (universal newlines), not raw bytes, so the same prompt
    checked out with CRLF line endings on one machine and LF on another
    hashes the same -- a line-ending difference alone must never look like
    an edited prompt and invalidate every cached result.
    """
    text = _read_text(name).strip("\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
