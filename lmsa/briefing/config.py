"""Briefing config files: `config/briefings/<name>.toml` -> `BriefingConfig`.

One file per briefing; adding a report is writing a file (see
`config/briefings/example.toml` for every key, documented inline). Every
table and key is checked: an unknown one raises `ConfigError` naming it, so
a typo never silently turns a filter off.

Source entries (`[[sources.<kind>]]`) accept the common keys below plus the
extra keys their adapter declares with `register(..., keys=...)`, so a new
adapter adds its own keys without editing this module.

On/off state (`enabled`) is never edited into the shipped file by the
dashboard (#95): `config/local/briefings.toml` (gitignored) can override it
per briefing, and always wins -- see `effective_enabled`.
"""

from __future__ import annotations

import math
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ROOT
from .. import geo

BRIEFINGS_DIR = ROOT / "config" / "briefings"
# Shared per-source lean/reliability tags (#69), reused across briefings.
SOURCES_FILE = ROOT / "config" / "sources.toml"
# Gitignored local on/off overrides (#95): keeps the dashboard's toggle out
# of the tracked config/briefings/<name>.toml files entirely.
LOCAL_OVERRIDES_PATH = ROOT / "config" / "local" / "briefings.toml"

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
LEANS = ("left", "center-left", "center", "center-right", "right", "state-media", "intl", "unknown")
RELIABILITY = ("high", "mixed", "low", "unknown")

# Keys every source entry may carry, whatever its kind. `lean`/`reliability`
# are the owner's own tags (#69); `group` buckets sources in the report.
# `prefilter = false` exempts the entry's docs from the keyword pre-filter (#128).
COMMON_SOURCE_KEYS = ("name", "lean", "reliability", "group", "limit", "prefilter")
SOURCE_KINDS = ("feeds", "search", "pages", "api")

# [briefing] keys -> (type(s), default). None default = optional, no value.
_BRIEFING_KEYS: dict[str, tuple[tuple[type, ...], Any]] = {
    "title": ((str,), None),
    "description": ((str,), ""),
    "enabled": ((bool,), None),          # None = never reviewed (shown as "new" by --list)
    "every_hours": ((int, float), 20),   # min hours between scheduled runs
    "interests": ((list,), None),
    "keywords": ((list,), []),
    "regions": ((list,), []),
    "prefilter": ((bool,), True),
    "follow_links": ((bool,), False),
    "max_items": ((int,), 300),
    "max_stories": ((int,), 20),
    "max_fetch": ((int,), 20),
    "max_docs_per_story": ((int,), 6),
    "cluster_threshold": ((int, float), 0.82),
    "half_life_hours": ((int, float), 36),
    "memory_days": ((int,), 7),
    "extensions": ((list,), []),
}
_INT_BOUNDS = {
    "max_items": (1, 5000), "max_stories": (1, 200), "max_fetch": (0, 500),
    "max_docs_per_story": (1, 30), "memory_days": (1, 365),
}


class ConfigError(ValueError):
    """A briefing file is missing, unreadable, or fails validation."""


@dataclass
class BriefingConfig:
    name: str
    path: Path
    title: str
    description: str = ""
    enabled: bool | None = None
    every_hours: float = 20.0
    interests: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)  # ISO alpha-2, upper-case
    prefilter: bool = True
    follow_links: bool = False
    max_items: int = 300
    max_stories: int = 20
    max_fetch: int = 20
    max_docs_per_story: int = 6
    cluster_threshold: float = 0.82
    half_life_hours: float = 36.0
    memory_days: int = 7
    extensions: list[str] = field(default_factory=list)
    # kind -> list of validated entry dicts (common keys + adapter keys).
    sources: dict[str, list[dict]] = field(default_factory=dict)

    def entries(self) -> list[tuple[str, dict]]:
        """Every source entry as `(kind, entry)`, in file order per kind."""
        return [(kind, e) for kind in SOURCE_KINDS for e in self.sources.get(kind, [])]


def path_for(name: str, directory: Path | None = None) -> Path:
    if not NAME_RE.fullmatch(name or ""):
        raise ConfigError(f"briefing name {name!r} must be lowercase letters, digits, '-' or '_'")
    return (directory or BRIEFINGS_DIR) / f"{name}.toml"


def names(directory: Path | None = None) -> list[str]:
    """Every briefing file's name (the stem), sorted."""
    d = directory or BRIEFINGS_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.toml") if NAME_RE.fullmatch(p.stem))


def local_overrides_path(directory: Path | None = None) -> Path:
    """Where the local (gitignored) on/off overrides file lives, next to
    `directory` -- e.g. config/briefings -> config/local/briefings.toml.
    With no `directory` this is `LOCAL_OVERRIDES_PATH` (module-level, so
    tests can monkeypatch it directly); with one, it's derived from it the
    same way `directory` already relocates `BRIEFINGS_DIR` for tests, so
    passing the same `directory` two isolated tests use for briefings keeps
    the override file isolated with it, with no extra fixture needed."""
    # The real config/briefings maps to LOCAL_OVERRIDES_PATH itself, so one
    # monkeypatch of it isolates every caller (the dashboard passes the real
    # directory explicitly).
    if directory is None or Path(directory).resolve() == BRIEFINGS_DIR.resolve():
        return LOCAL_OVERRIDES_PATH
    return directory.parent / "local" / "briefings.toml"


def local_overrides_status(path: Path | None = None) -> tuple[dict[str, bool], str | None]:
    """Like `local_overrides`, but hands back the malformed-file reason
    instead of only printing it to stderr, so a call site that a person
    actually looks at (the dashboard's briefing tile, `python -m
    tasks.briefing`) can show it instead of it landing silently in a
    background process's stderr (#95 review finding 2).

    Read this once per top-level operation (one `--list`/run of
    `tasks.briefing`, one dashboard page render) and thread the resulting
    `overrides` dict through via `load(..., overrides=...)` /
    `parse(..., overrides=...)` so a malformed file is reported once, not
    once per briefing (finding 2's "not N times")."""
    p = path or LOCAL_OVERRIDES_PATH
    if not p.is_file():
        return {}, None
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(data, dict):
        return {}, "expected a table"
    extra = set(data) - {"briefings"}
    if extra:
        return {}, f"unknown table/key {sorted(extra)[0]!r} (expected [briefings])"
    table = data.get("briefings", {})
    if not isinstance(table, dict):
        return {}, "briefings must be a table of name = true|false"
    out: dict[str, bool] = {}
    bad: str | None = None
    for key, value in table.items():
        if not isinstance(key, str) or not NAME_RE.fullmatch(key) or not isinstance(value, bool):
            bad = bad or f"briefings.{key!r} must be true or false"
            continue
        out[key] = value
    return out, bad


def local_overrides_warning(reason: str) -> str:
    """The user-facing message for a malformed local overrides file (#95
    review finding 2): shown on the dashboard's briefing tile and printed
    once at the top by `python -m tasks.briefing`, so the file's
    switches-are-ignored consequence is visible, not just a stderr line."""
    return (f"config/local/briefings.toml has a problem: {reason}; "
            "any switch it can't read is ignored (the briefing falls back to its shipped setting)")


def local_overrides(path: Path | None = None) -> dict[str, bool]:
    """`{briefing name: enabled}` from the untracked local overrides file
    (#95): a `[briefings]` table, `<name> = true|false`. This is the file
    the dashboard's Enable/Disable toggle writes -- never the shipped
    config/briefings/<name>.toml.

    Missing file = no overrides. A malformed file is a warning, never a
    crash: one bad local edit must not take out every briefing (the same
    rule `load_source_tags` follows for sources.toml). Prefer
    `local_overrides_status` at a call site that can show the reason to a
    person instead of only stderr."""
    out, reason = local_overrides_status(path)
    if reason is not None:
        print(f"warning: ignoring {path or LOCAL_OVERRIDES_PATH}: {reason}", file=sys.stderr)
    return out


def effective_enabled(shipped: bool | None, name: str, *, overrides: dict[str, bool] | None = None,
                      local_path: Path | None = None) -> bool | None:
    """The one place the on/off state is resolved (#95). Precedence: the
    local override file wins, then the shipped file's own `enabled`, then
    None ("never reviewed", shown as NEW). Every reader -- `BriefingConfig.
    enabled` (via `parse`/`load`), `tasks.briefing`'s `--list`/due
    selection, and the dashboard's NEW badge -- goes through this."""
    if overrides is None:
        overrides = local_overrides(local_path)
    return overrides.get(name, shipped)


def load(name: str, directory: Path | None = None, *, sources_file: Path | None = None,
        local_path: Path | None = None, overrides: dict[str, bool] | None = None) -> BriefingConfig:
    path = path_for(name, directory)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"no briefing file {path}") from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    lp = local_path if local_path is not None else local_overrides_path(directory)
    return parse(name, data, path, sources_file=sources_file, local_path=lp, overrides=overrides)


def load_source_tags(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Shared per-source `lean`/`reliability` tags (#69): `config/sources.toml`
    (`[[sources]] name = "..."`), matched by each `[[sources.<kind>]]`
    entry's resolved `name` (explicit `name`, else its url/query/adapter --
    see `_source_entry`). These are the owner's own tags, set by hand once,
    never scraped.

    Missing file = no tags (everything stays "unknown" unless a briefing
    entry sets it itself). A briefing entry's own `lean`/`reliability`
    always wins over this file -- this file only fills in what the entry
    left unset."""
    p = path or SOURCES_FILE
    if not p.is_file():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{p.name}: {exc}") from exc
    extra = set(data) - {"sources"}
    if extra:
        raise ConfigError(f"{p.name}: unknown table {sorted(extra)[0]!r} (expected [[sources]])")
    entries = data.get("sources", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ConfigError(f"{p.name}: sources must be an array of tables ([[sources]])")
    tags: dict[str, dict[str, str]] = {}
    for i, e in enumerate(entries):
        label = f"sources[{i}]"
        name = e.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{p.name}: {label}.name is required")
        extra_keys = set(e) - {"name", "lean", "reliability"}
        if extra_keys:
            raise ConfigError(f"{p.name}: {label}: unknown key {sorted(extra_keys)[0]!r}")
        lean = e.get("lean", "unknown")
        if lean not in LEANS:
            raise ConfigError(f"{p.name}: {label}.lean must be one of {', '.join(LEANS)}")
        reliability = e.get("reliability", "unknown")
        if reliability not in RELIABILITY:
            raise ConfigError(f"{p.name}: {label}.reliability must be one of {', '.join(RELIABILITY)}")
        tags[name.strip()] = {"lean": lean, "reliability": reliability}
    return tags


def _str_list(value: object, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"briefing.{key} must be a list of non-empty strings")
    return [v.strip() for v in value]


def parse(name: str, data: dict, path: Path | None = None, *, sources_file: Path | None = None,
         local_path: Path | None = None, overrides: dict[str, bool] | None = None) -> BriefingConfig:
    """Validate a parsed TOML document. Raises `ConfigError` naming the bad key."""
    from . import sources as _sources  # registry (adapter-declared keys)

    # Plugin adapters/extensions must be registered before their keys and
    # names are checked -- otherwise every briefing using one is "invalid".
    _sources.load_plugins()
    where = path.name if path else f"{name}.toml"
    extra = set(data) - {"briefing", "sources"}
    if extra:
        raise ConfigError(f"{where}: unknown table/key {sorted(extra)[0]!r} (expected [briefing], [[sources.*]])")
    b = data.get("briefing")
    if not isinstance(b, dict):
        raise ConfigError(f"{where}: missing [briefing] table")
    for key in b:
        if key not in _BRIEFING_KEYS:
            raise ConfigError(f"{where}: unknown key briefing.{key}")
    values: dict[str, Any] = {}
    for key, (types, default) in _BRIEFING_KEYS.items():
        if key not in b:
            values[key] = list(default) if isinstance(default, list) else default
            continue
        v = b[key]
        if isinstance(v, bool) and bool not in types:
            raise ConfigError(f"{where}: briefing.{key} has the wrong type")
        if not isinstance(v, types):
            raise ConfigError(f"{where}: briefing.{key} has the wrong type")
        values[key] = v
    # Local override (#95) wins over the shipped file's own `enabled` --
    # the one place this precedence is applied, so every reader that goes
    # through `parse`/`load` sees the effective state. A caller looping over
    # several briefings in one operation passes `overrides=` (computed once
    # via `local_overrides_status`) so the file is read -- and a malformed
    # one reported -- once per operation, not once per briefing.
    if overrides is not None:
        values["enabled"] = effective_enabled(values["enabled"], name, overrides=overrides)
    else:
        lp = local_path if local_path is not None else local_overrides_path(path.parent if path else None)
        values["enabled"] = effective_enabled(values["enabled"], name, local_path=lp)
    if not isinstance(values["title"], str) or not values["title"].strip():
        raise ConfigError(f"{where}: briefing.title is required")
    if values["interests"] is None:
        raise ConfigError(f"{where}: briefing.interests is required (text descriptions used for ranking)")
    values["interests"] = _str_list(values["interests"], "interests")
    if not values["interests"]:
        raise ConfigError(f"{where}: briefing.interests must not be empty")
    values["keywords"] = _str_list(values["keywords"], "keywords")
    values["extensions"] = _str_list(values["extensions"], "extensions")
    regions = []
    for code in _str_list(values["regions"], "regions"):
        if not re.fullmatch(r"[A-Za-z]{2}", code) or geo.country(code) is None:
            raise ConfigError(f"{where}: briefing.regions: {code!r} is not an ISO 3166 alpha-2 country code")
        regions.append(code.upper())
    values["regions"] = regions
    for key, (lo, hi) in _INT_BOUNDS.items():
        if not lo <= values[key] <= hi:
            raise ConfigError(f"{where}: briefing.{key} must be between {lo} and {hi}")
    ct = float(values["cluster_threshold"])
    if not (math.isfinite(ct) and 0 < ct <= 1):
        raise ConfigError(f"{where}: briefing.cluster_threshold must be in (0, 1]")
    for key in ("every_hours", "half_life_hours"):
        v = float(values[key])
        if not (math.isfinite(v) and v > 0):
            raise ConfigError(f"{where}: briefing.{key} must be a positive number")
        values[key] = v
    values["cluster_threshold"] = ct
    for ext in values["extensions"]:
        if ext not in _sources.EXTENSIONS:
            raise ConfigError(f"{where}: briefing.extensions: no extension named {ext!r}")

    srcs = data.get("sources", {})
    if not isinstance(srcs, dict):
        raise ConfigError(f"{where}: sources must be tables ([[sources.feeds]] ...)")
    try:
        tags = load_source_tags(sources_file)
    except ConfigError as exc:
        # sources.toml is shared by every briefing: one bad line there must
        # cost only the lean/reliability tags, never every briefing's run.
        print(f"warning: ignoring source tags: {exc}", file=sys.stderr)
        tags = {}
    parsed: dict[str, list[dict]] = {}
    for kind, entries in srcs.items():
        if kind not in SOURCE_KINDS:
            raise ConfigError(f"{where}: unknown source kind sources.{kind} (use {', '.join(SOURCE_KINDS)})")
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ConfigError(f"{where}: sources.{kind} must be an array of tables ([[sources.{kind}]])")
        parsed[kind] = [_source_entry(where, kind, i, e, _sources, tags) for i, e in enumerate(entries)]
    if not any(parsed.values()):
        raise ConfigError(f"{where}: no sources: add at least one [[sources.feeds]], [[sources.search]], "
                          "[[sources.pages]] or [[sources.api]] entry")
    return BriefingConfig(name=name, path=path or Path(where), sources=parsed, **values)


def _source_entry(where: str, kind: str, i: int, entry: dict, registry,
                   tags: dict[str, dict[str, str]] | None = None) -> dict:
    label = f"sources.{kind}[{i}]"
    if kind == "api":
        adapter = entry.get("adapter")
        if not isinstance(adapter, str) or not adapter:
            raise ConfigError(f"{where}: {label}.adapter is required (a registered adapter name)")
        spec = registry.ADAPTERS.get(adapter)
        if spec is None or adapter in ("feeds", "search", "pages"):
            known = ", ".join(sorted(n for n in registry.ADAPTERS if n not in ("feeds", "search", "pages"))) or "none"
            raise ConfigError(f"{where}: {label}.adapter: no adapter named {adapter!r} (registered: {known})")
        allowed = {"adapter", *COMMON_SOURCE_KEYS, *spec.keys}
        required = set(spec.required)
    else:
        spec = registry.ADAPTERS.get(kind)
        if spec is None:
            # e.g. [[sources.pages]] before a site-watch adapter is installed:
            # accepted (key-checked only for the common keys + url), and the
            # run records a "not available" note for it.
            allowed, required = None, {"url"}
        else:
            allowed = {*COMMON_SOURCE_KEYS, *spec.keys}
            required = set(spec.required)
    if allowed is not None:
        for key in entry:
            if key not in allowed:
                raise ConfigError(f"{where}: unknown key {label}.{key}")
    for key in sorted(required):
        if key not in entry or entry[key] in ("", None):
            raise ConfigError(f"{where}: {label}.{key} is required")
    for key in ("name", "group"):
        if key in entry and not isinstance(entry[key], str):
            raise ConfigError(f"{where}: {label}.{key} must be a string")
    if "lean" in entry and entry["lean"] not in LEANS:
        raise ConfigError(f"{where}: {label}.lean must be one of {', '.join(LEANS)}")
    if "reliability" in entry and entry["reliability"] not in RELIABILITY:
        raise ConfigError(f"{where}: {label}.reliability must be one of {', '.join(RELIABILITY)}")
    if "prefilter" in entry and not isinstance(entry["prefilter"], bool):
        raise ConfigError(f"{where}: {label}.prefilter must be true or false")
    if "limit" in entry and (isinstance(entry["limit"], bool) or not isinstance(entry["limit"], int)
                             or not 1 <= entry["limit"] <= 1000):
        raise ConfigError(f"{where}: {label}.limit must be an integer 1-1000")
    if "url" in entry and not (isinstance(entry["url"], str) and entry["url"].lower().startswith(("http://", "https://"))):
        raise ConfigError(f"{where}: {label}.url must be an http(s) URL")
    out = dict(entry)
    out.setdefault("name", str(entry.get("name") or entry.get("url") or entry.get("query") or entry.get("adapter") or kind))
    tag = (tags or {}).get(out["name"], {})
    out.setdefault("lean", tag.get("lean", "unknown"))
    out.setdefault("reliability", tag.get("reliability", "unknown"))
    out.setdefault("group", "")
    if spec is not None and spec.validate is not None:
        try:
            spec.validate(out)
        except ConfigError:
            raise
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"{where}: {label}: {exc}") from exc
    return out
