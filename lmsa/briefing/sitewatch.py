"""Site watch core (#53): fetch a page, diff it against the last snapshot,
and ask the model whether the change matches a plain-English rule --
changedetection.io's pattern, but judged and summarized instead of dumped
as a raw diff.

Shared by the `pages` briefing adapter (`lmsa/briefing/adapters/pages.py`,
for `[[sources.pages]]` entries in any briefing) and the standalone
`tasks/site_watch.py` (`config/site_watch.toml`) -- same fetch/diff/judge
core, different config file and report.

    from lmsa.briefing.sitewatch import PageSpec, check_page

    spec = PageSpec(url="https://example.com", rule="only if the price drops",
                     watcher="task:site_watch:my-page")
    result = check_page(fetcher=fetcher, lm=lm, conn=conn, spec=spec, now=now)

`check_page()` never raises for a page-level problem (a bad fetch, a
selector that matches nothing, a model call gone wrong): it's caught and
comes back as `PageResult.error`, so one bad page never takes the run down.

**Watchers.** Snapshots and typed-field history are keyed by `(watcher,
url)`, not just `url` -- two different watchers of the same page (the
standalone task and a briefing's `[[sources.pages]]` entry, or two
briefings) each keep their own snapshot lineage, so each one independently
sees every change it's due, rather than whichever watcher runs first
consuming the diff and leaving the others with nothing to compare against.
`watcher` is a caller-chosen id (`"task:site_watch:<name>"`,
`"briefing:<briefing>:<entry name>"`); an empty one falls back to `url`
itself. Model *judging* is memoized independently of the watcher, in the
shared `llm_memo` table under task `briefing.pages`, keyed by `(url, diff
hash, rule hash, prompt hash)` -- so the same real-world change, judged
under the same rule, costs one model call however many watchers see it,
even though each watcher's own diff (against its own last snapshot) got it
there separately.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..feeds import strip_html
from ..memo import Memo, has_keys, memo_version, model_id
from ..prompts import load as load_prompt
from ..prompts import prompt_hash
from ..web import extract_article

log = logging.getLogger(__name__)

TASK = "briefing.pages"
MEMO_VERSION = "site_watch-1"
SYSTEM_PROMPT = "site_watch/extract_system"
USER_PROMPT = "site_watch/extract_user"

MAX_TEXT_CHARS = 8_000    # page text kept per snapshot / sent to the model
MAX_DIFF_CHARS = 6_000    # unified diff sent to the model
MAX_SNAPSHOTS = 20        # snapshots kept per (watcher, url) -- bounded history
MAX_FIELD_HISTORY = 365   # typed-field readings kept per (watcher, url, field)
MAX_SUMMARY_CHARS = 500

FIELD_TYPES = ("number", "text", "bool")
STATUSES = ("baseline", "unchanged", "changed")

_TRUNCATED_HINT = (
    "Note: the page text and/or diff below was cut short to a length limit -- "
    "judge based on what's shown; don't assume anything about what was cut."
)


@dataclass
class PageSpec:
    """One watched page, however it's configured -- a `[[sources.pages]]`
    briefing entry or a `[[pages]]` row in `config/site_watch.toml`.
    `watcher` identifies *who* is watching this url (see module docstring);
    an empty one falls back to `url`. `fields` maps a name to one of
    `FIELD_TYPES`. `load_error` marks an entry that failed its own
    validation (bad url, duplicate, bad field type) -- `check_page` turns
    it straight into that page's error rather than ever fetching."""

    url: str
    name: str = ""
    rule: str = ""
    selector: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    notify: bool = False
    watcher: str = ""
    load_error: str | None = None

    def __post_init__(self) -> None:
        self.name = self.name or self.url


@dataclass
class PageResult:
    url: str
    name: str
    status: str = "error"  # baseline | unchanged | changed | error
    text: str = ""
    diff: str = ""
    relevant: bool | None = None
    summary: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    truncated: bool = False
    should_notify: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)


# -- selector-based text extraction ------------------------------------------

_TAG_OPEN_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>")


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"|{name}\s*=\s*\'([^\']*)\'|{name}\s*=\s*([^\s">]+)', attrs, re.I)
    if not m:
        return None
    return next(g for g in m.groups() if g is not None)


def _matches_selector(tag: str, attrs: str, selector: str) -> bool:
    if selector.startswith("#"):
        return _attr(attrs, "id") == selector[1:]
    if selector.startswith("."):
        return selector[1:] in (_attr(attrs, "class") or "").split()
    return tag.lower() == selector.lower()


def select_html(html: str, selector: str) -> str | None:
    """Inner HTML of the first element matching a bare tag / `#id` /
    `.class` selector ("keep simple" per the issue). Depth-aware so a
    nested element of the same tag name doesn't close it early. An
    unterminated tag returns the rest of the document (best-effort); no
    match returns None."""
    for m in _TAG_OPEN_RE.finditer(html):
        tag, attrs = m.group(1), m.group(2)
        if not _matches_selector(tag, attrs, selector):
            continue
        start = m.end()
        depth = 1
        boundary_re = re.compile(rf"<{re.escape(tag)}\b[^>]*>|</{re.escape(tag)}\s*>", re.I)
        for bm in boundary_re.finditer(html, start):
            if bm.group(0).startswith("</"):
                depth -= 1
                if depth == 0:
                    return html[start:bm.start()]
            else:
                depth += 1
        return html[start:]
    return None


def extract_text(html: str, url: str, selector: str | None) -> str:
    """Main text of the page: `selector` when given and it matches, else
    `lmsa.web.extract_article`."""
    if selector:
        picked = select_html(html, selector)
        if picked is not None:
            return strip_html(picked)
    return extract_article(html, url).text


# -- snapshot + field history storage ----------------------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_snapshot(conn, watcher: str, url: str) -> dict | None:
    row = conn.execute(
        "SELECT text, text_hash FROM site_watch_snapshots WHERE watcher=? AND url=? "
        "ORDER BY checked_at DESC, rowid DESC LIMIT 1",
        (watcher, url),
    ).fetchone()
    if row is None:
        return None
    return {"text": row[0], "text_hash": row[1]}


def _save_snapshot(conn, watcher: str, url: str, name: str, checked_at: str, text: str, text_hash: str,
                    *, max_snapshots: int = MAX_SNAPSHOTS) -> None:
    conn.execute(
        "INSERT INTO site_watch_snapshots (watcher, url, name, checked_at, text, text_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (watcher, url, name, checked_at, text, text_hash),
    )
    # Bounded history (acceptance criterion): keep only the newest N per (watcher, url).
    conn.execute(
        "DELETE FROM site_watch_snapshots WHERE watcher=? AND url=? AND rowid NOT IN ("
        "SELECT rowid FROM site_watch_snapshots WHERE watcher=? AND url=? "
        "ORDER BY checked_at DESC, rowid DESC LIMIT ?)",
        (watcher, url, watcher, url, max_snapshots),
    )
    conn.commit()


def record_field(conn, watcher: str, url: str, field_name: str, checked_at: str, value: Any,
                  *, max_history: int = MAX_FIELD_HISTORY) -> None:
    """Append one typed-field reading for `(watcher, url, field_name)` (a
    report's sparkline history), bounded to the newest `max_history`."""
    try:
        value_json = json.dumps(value)
    except (TypeError, ValueError):
        return
    conn.execute(
        "INSERT INTO site_watch_fields (watcher, url, field, checked_at, value_json) VALUES (?, ?, ?, ?, ?)",
        (watcher, url, field_name, checked_at, value_json),
    )
    conn.execute(
        "DELETE FROM site_watch_fields WHERE watcher=? AND url=? AND field=? AND rowid NOT IN ("
        "SELECT rowid FROM site_watch_fields WHERE watcher=? AND url=? AND field=? "
        "ORDER BY checked_at DESC, rowid DESC LIMIT ?)",
        (watcher, url, field_name, watcher, url, field_name, max_history),
    )
    conn.commit()


def field_history(conn, watcher: str, url: str, field_name: str, *, limit: int = 30) -> list[float]:
    """Numeric values recorded for `(watcher, url, field_name)`, oldest
    first (a sparkline). Non-numeric readings (text/bool fields) are
    skipped."""
    rows = conn.execute(
        "SELECT value_json FROM site_watch_fields WHERE watcher=? AND url=? AND field=? "
        "ORDER BY checked_at ASC, rowid ASC LIMIT ?",
        (watcher, url, field_name, limit),
    ).fetchall()
    out: list[float] = []
    for (raw,) in rows:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.append(float(value))
    return out


# -- model schema / coercion --------------------------------------------------


def _fields_schema(fields: dict[str, str]) -> dict:
    props = {}
    for name, kind in fields.items():
        if kind == "number":
            props[name] = {"type": ["number", "null"]}
        elif kind == "bool":
            props[name] = {"type": ["boolean", "null"]}
        else:
            props[name] = {"type": ["string", "null"]}
    return {"type": "object", "properties": props, "required": list(fields), "additionalProperties": False}


def build_schema(fields: dict[str, str]) -> dict:
    schema: dict = {
        "type": "object",
        "properties": {"relevant": {"type": "boolean"}, "summary": {"type": "string"}},
        "required": ["relevant", "summary"],
        "additionalProperties": False,
    }
    if fields:
        schema["properties"]["fields"] = _fields_schema(fields)
        schema["required"] = schema["required"] + ["fields"]
    return schema


# Strip everything but digits, sign and the two decimal/thousands separator
# characters -- drops currency symbols/codes ("$", "€", "USD", "kr", ...)
# and any surrounding words.
_NUMERIC_STRIP_RE = re.compile(r"[^\d,.\-+]")


def _parse_number(raw: Any) -> tuple[float | None, str | None]:
    """A best-effort number from a model-extracted field value: plain
    numbers pass through; a string may carry a currency symbol, and
    thousands/decimal separators in either style ("$1,299.99",
    "1.299,99 €", "1 299,99") -- whichever separator appears last (or is
    followed by exactly 1-2 digits, with no other separator) is taken as
    the decimal point, the other as thousands grouping. Returns
    `(value, None)` on success or `(None, a note to disclose)`."""
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, f"expected a number, got boolean {raw!r}"
    if isinstance(raw, (int, float)):
        return float(raw), None
    if not isinstance(raw, str):
        return None, f"expected a number, got {type(raw).__name__}"

    text = raw.strip()
    if not text:
        return None, "empty value"
    cleaned = _NUMERIC_STRIP_RE.sub("", text)
    if not cleaned or not re.search(r"\d", cleaned):
        return None, f"couldn't parse {raw!r} as a number"

    negative = cleaned.startswith("-")
    cleaned = cleaned.lstrip("+-")
    comma_i, dot_i = cleaned.rfind(","), cleaned.rfind(".")

    try:
        if "," in cleaned and "." in cleaned:
            # Both separators present: whichever comes last is the decimal
            # point, the other is thousands grouping.
            if comma_i > dot_i:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Comma only: a single comma followed by 1-2 digits reads as a
            # decimal comma ("12,50"); anything else (a 3-digit group, or
            # more than one comma) reads as thousands grouping ("1,234").
            last_group = cleaned.rsplit(",", 1)[1]
            if cleaned.count(",") == 1 and len(last_group) in (1, 2):
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif cleaned.count(".") > 1:
            # More than one dot with no comma: dots are thousands grouping.
            cleaned = cleaned.replace(".", "")
        value = float(cleaned)
    except (ValueError, IndexError):
        return None, f"couldn't parse {raw!r} as a number"
    return (-value if negative else value), None


def _coerce(value: Any, kind: str) -> tuple[Any, str | None]:
    """`(coerced value, disclosure note)` -- the note is set only when a
    non-null value couldn't be made to fit `kind`."""
    if value is None:
        return None, None
    if kind == "number":
        return _parse_number(value)
    if kind == "bool":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1"):
                return True, None
            if low in ("false", "no", "0"):
                return False, None
            return None, f"couldn't parse {value!r} as a bool"
        return bool(value), None
    return str(value), None


# -- the check -----------------------------------------------------------------


def check_page(*, fetcher: Any, lm: Any, conn: Any, spec: PageSpec, now: datetime,
                fresh: bool = False) -> PageResult:
    """Fetch `spec.url` once, diff it against `spec.watcher`'s last
    snapshot for that url, and (only on a real, non-empty diff) ask the
    model whether it matches `spec.rule`.

    The snapshot only advances past a `"changed"` diff once the model has
    successfully judged it (or the diff turns out empty at the line level,
    a clean `"unchanged"`) -- a failed or invalid model call leaves the
    previous snapshot in place so the next run re-diffs the same change and
    retries, instead of silently losing it. A first-ever check for
    `(watcher, url)` is `"baseline"`: it establishes the snapshot but is
    never judged or alerted on -- there's nothing yet to diff against."""
    checked_at = now.isoformat()
    watcher = spec.watcher or spec.url
    result = PageResult(url=spec.url, name=spec.name)
    try:
        if spec.load_error:
            raise ValueError(spec.load_error)

        resp = fetcher.get(spec.url)
        text = extract_text(resp.text, spec.url, spec.selector)
        text_truncated = len(text) > MAX_TEXT_CHARS
        text = text[:MAX_TEXT_CHARS]
        text_hash = _hash(text)
        result.text = text
        result.truncated = text_truncated

        previous = _last_snapshot(conn, watcher, spec.url)

        if previous is None:
            _save_snapshot(conn, watcher, spec.url, spec.name, checked_at, text, text_hash, max_snapshots=MAX_SNAPSHOTS)
            result.status = "baseline"
            return result

        if previous["text_hash"] == text_hash:
            result.status = "unchanged"
            return result

        diff = "\n".join(difflib.unified_diff(
            previous["text"].splitlines(), text.splitlines(),
            fromfile="previous", tofile="current", lineterm="",
        ))
        if not diff.strip():
            # Hash differs (e.g. whitespace collapsed differently) but the
            # line-level diff is empty: a clean "unchanged" -- advance the
            # snapshot so this trivial difference isn't re-diffed forever.
            _save_snapshot(conn, watcher, spec.url, spec.name, checked_at, text, text_hash, max_snapshots=MAX_SNAPSHOTS)
            result.status = "unchanged"
            return result

        result.status = "changed"
        diff_truncated = len(diff) > MAX_DIFF_CHARS
        if diff_truncated:
            diff = diff[:MAX_DIFF_CHARS]
        result.truncated = result.truncated or diff_truncated
        result.diff = diff

        rule = spec.rule or ""
        schema = build_schema(spec.fields)
        system = load_prompt(SYSTEM_PROMPT)
        version = memo_version(MEMO_VERSION, prompt_hash(SYSTEM_PROMPT), model_id(lm), schema)
        key = "|".join((spec.url, _hash(diff), _hash(rule), prompt_hash(SYSTEM_PROMPT)))
        memo = Memo(conn, TASK, version, fresh=fresh)
        truncated_for_prompt = result.truncated

        def compute() -> dict:
            fields_hint = (
                "Also extract these fields from the page: "
                + ", ".join(f"{n} ({k})" for n, k in spec.fields.items())
                if spec.fields else "No extra fields requested."
            )
            prompt = load_prompt(
                USER_PROMPT,
                rule=rule or "(no rule given: treat any real content change as relevant)",
                fields_hint=fields_hint,
                truncated_hint=_TRUNCATED_HINT if truncated_for_prompt else "",
                diff=diff,
            )
            return lm.extract(prompt, schema, system=system, name="site_watch")

        try:
            answer = memo.cached(key, compute, valid=lambda r: has_keys(r, ("relevant", "summary")))
        except Exception as exc:
            # Judging failed (model error, bad JSON, missing keys): the
            # change is real but unjudged. Do NOT advance the snapshot --
            # the next run re-diffs against the same previous snapshot and
            # retries the judgment, rather than losing the change.
            log.warning("site_watch judge failed for %s", spec.url, exc_info=True)
            result.error = f"change not judged ({type(exc).__name__}: {exc})"
            return result

        result.relevant = bool(answer.get("relevant"))
        result.summary = str(answer.get("summary") or "")[:MAX_SUMMARY_CHARS]
        raw_fields = answer.get("fields") or {}
        coerced: dict[str, Any] = {}
        for name, kind in spec.fields.items():
            value, note = _coerce(raw_fields.get(name), kind)
            coerced[name] = value
            if note:
                result.notes.append(f"field {name!r}: {note}")
        result.fields = coerced
        for name, value in coerced.items():
            if value is not None:
                record_field(conn, watcher, spec.url, name, checked_at, value, max_history=MAX_FIELD_HISTORY)

        result.should_notify = bool(spec.notify and result.relevant)
        # Only now that judging succeeded does the snapshot advance.
        _save_snapshot(conn, watcher, spec.url, spec.name, checked_at, text, text_hash, max_snapshots=MAX_SNAPSHOTS)
        return result
    except Exception as exc:
        # Per-item isolation: a bad fetch, a bad load_error, or an
        # unmatched selector is this page's problem, never the run's.
        log.warning("site_watch check failed for %s", spec.url, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
        return result
