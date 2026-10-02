"""Bookmark tagging: Chrome/Edge bookmarks + emailed-to-self links, tagged,
summarized, searchable (Karakeep/Hoarder pattern, #66).

    python -m tasks.bookmarks [--limit 25]
    python -m tasks.bookmarks --query "that rust async talk" [--k 8]

Reads bookmarks from Chrome's and Edge's `Bookmarks` JSON (default Windows
profile paths, overridable in `config/bookmarks.toml`) and links found in
emails sent to yourself (Gmail, read-only IMAP, incremental via the shared
cursor -- see `lmsa.gmail_cursor`). Every URL seen this way is upserted into
this task's own `bookmarks` table, deduplicated by normalized URL
(`lmsa.web.normalize_url`) across every source and folder it was seen in
(`bookmark_sources`).

Never-checked URLs are fetched through `lmsa.web.Fetcher` (robots.txt
honoured, HTTP cache, SSRF guard), up to `--limit` new fetches per run (the
rest are picked up on a later run -- see the report's "truncated" note).
The article text is extracted (`lmsa.web.extract_article`) and the model
tags it with `{summary, tags[], type}` via `extract()`, then the page
(title + summary + tags) is embedded into the `bookmarks` vectors namespace
(`lmsa.vectors`) for `--query` semantic search, reusing the same
embed-and-cosine-search machinery `tasks.doc_search` uses over `docs`.

A fetch failure is classified, never left as an undifferentiated error:
- **dead**: a definitive failure -- HTTP 404/410, or a DNS resolution
  failure -- surfaced in its own report section.
- **couldn't check**: a robots.txt denial, a timeout, a 5xx, or any other
  transient `FetchError` -- reported separately, not counted as dead.
- **skipped**: a URL the Fetcher's SSRF guard refuses outright (a
  `file:`/`javascript:` scheme, an intranet/private host, an unparseable
  URL) -- never even attempted as a network fetch.

Read-only over the sources: the browser `Bookmarks` files are copy-read
(the browser may hold the original open; it is never opened for write) and
Gmail is read via IMAP PEEK only, same as every other Gmail-cursor task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from lmsa import LMStudio, Link, Report, Store, Tile
from lmsa.gmail_cursor import GmailCursor, Window, add_pending_note
from lmsa.prompts import load as load_prompt
from lmsa.report import safe_href
from lmsa.sources.gmail import Gmail, GmailAuthError
from lmsa.vectors import embed_and_upsert, search
from lmsa.web import (
    BlockedURL,
    FetchError,
    Fetcher,
    HTTPStatusError,
    RobotsDenied,
    TooLarge,
    extract_article,
    normalize_url,
)

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], "config": ["config/bookmarks.toml"], "model": "chat"}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "bookmarks.toml"

NS = "bookmarks"
KV_NS = "bookmarks_mail"
CURSOR_KEY = "since"
USER_AGENT = "lmsa-bookmarks/1.0 (+personal automation, single user)"

DEFAULT_FETCH_LIMIT = 25  # new URLs fetched per run; the rest wait for the next run
DEFAULT_SELF_SENT_LOOKBACK_DAYS = 365  # first-ever run only; later runs use the cursor
SELF_SENT_SEARCH_LIMIT = 200  # self-sent messages scanned per run
CONTENT_CHAR_CAP = 6000  # chars of extracted article text sent to the model
REPORT_ROWS_CAP = 50  # rows shown per report table, so a big backlog can't blow up the HTML

EXTRACT_TYPES = ("article", "reference", "tool", "video", "shopping", "forum", "other")

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "type": {"type": "string", "enum": list(EXTRACT_TYPES)},
    },
    "required": ["summary", "tags", "type"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = load_prompt("bookmarks/extract_system")

# Chrome/Edge JSON: URLs starting with a scheme other than http(s) (file:,
# javascript:, chrome:, ftp:, ...) reach `normalize_url` fine but the
# Fetcher's own guard refuses them as BlockedURL -- classified "skipped".
URL_RE = re.compile(r"""https?://[^\s<>"')\]]+""", re.IGNORECASE)

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookmarks (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'new',
    http_status INTEGER,
    check_error TEXT,
    checked_at TEXT,
    summary TEXT,
    tags_json TEXT,
    type TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bookmark_sources (
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    added_at TEXT,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (url, source, context)
);
CREATE TABLE IF NOT EXISTS bookmarks_mail_applied (
    msg_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


def _now() -> datetime:
    """Current UTC time. Tests monkeypatch this (#34)."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_CREATE_SCHEMA)
    conn.commit()


# -- config -------------------------------------------------------------------


@dataclass
class Config:
    chrome_path: Path | None
    edge_path: Path | None
    fetch_limit: int
    self_sent_lookback_days: int


def default_chrome_path() -> Path | None:
    base = os.getenv("LOCALAPPDATA")
    return Path(base) / "Google" / "Chrome" / "User Data" / "Default" / "Bookmarks" if base else None


def default_edge_path() -> Path | None:
    base = os.getenv("LOCALAPPDATA")
    return Path(base) / "Microsoft" / "Edge" / "User Data" / "Default" / "Bookmarks" if base else None


def _configured_path(raw: object, default: Path | None) -> Path | None:
    if raw is None:
        return default
    if isinstance(raw, str) and not raw.strip():
        return None  # explicit "" opts that browser out
    return Path(str(raw)).expanduser()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config(
        chrome_path=_configured_path(data.get("chrome_path"), default_chrome_path()),
        edge_path=_configured_path(data.get("edge_path"), default_edge_path()),
        fetch_limit=int(data.get("fetch_limit", DEFAULT_FETCH_LIMIT)),
        self_sent_lookback_days=int(data.get("self_sent_lookback_days", DEFAULT_SELF_SENT_LOOKBACK_DAYS)),
    )


# -- shared entry type ----------------------------------------------------------


@dataclass
class BookmarkEntry:
    url: str  # raw, not yet normalized
    title: str
    source: str  # "chrome" | "edge" | "gmail"
    context: str  # folder path, or "self-sent: <subject>"
    added_at: str | None  # ISO datetime, or None


# -- Chrome/Edge Bookmarks JSON -------------------------------------------------


def webkit_to_iso(value: object) -> str | None:
    """Chrome's `date_added`/`date_modified`: microseconds since 1601-01-01
    (Windows FILETIME epoch), as a decimal string. `0`/missing/garbage -> None."""
    try:
        micros = int(value)
    except (TypeError, ValueError):
        return None
    if micros <= 0:
        return None
    try:
        return (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)).isoformat()
    except OverflowError:
        return None


def _walk_bookmark_node(node: object, folder_path: str) -> Iterator[dict]:
    """Depth-first over one Chrome/Edge bookmark tree node, yielding one
    dict per `url`-type leaf. Tolerates a malformed node (wrong type, no
    `children`) by simply yielding nothing further down that branch."""
    if not isinstance(node, dict):
        return
    node_type = node.get("type")
    if node_type == "folder":
        name = str(node.get("name") or "")
        sub_path = f"{folder_path}/{name}" if folder_path else name
        for child in node.get("children") or []:
            yield from _walk_bookmark_node(child, sub_path)
    elif node_type == "url":
        url = node.get("url")
        if isinstance(url, str) and url:
            yield {
                "url": url,
                "title": str(node.get("name") or ""),
                "folder": folder_path,
                "added_at": webkit_to_iso(node.get("date_added")),
            }


def parse_bookmarks_json(data: dict, source: str) -> list[BookmarkEntry]:
    """Every `url` leaf under `data["roots"]`, nested folders flattened into
    a `/`-joined path (e.g. `"Bookmarks bar/Work/Reading"`)."""
    roots = data.get("roots") if isinstance(data, dict) else None
    if not isinstance(roots, dict):
        return []
    entries: list[BookmarkEntry] = []
    for root_name, root_node in roots.items():
        if not isinstance(root_node, dict):
            continue
        top = str(root_node.get("name") or root_name)
        # Walk the root's own children with `top` as the starting path --
        # not `_walk_bookmark_node(root_node, top)`, which would fold the
        # root's own name into the path a second time.
        for child in root_node.get("children") or []:
            for leaf in _walk_bookmark_node(child, top):
                entries.append(BookmarkEntry(
                    url=leaf["url"], title=leaf["title"], source=source,
                    context=leaf["folder"], added_at=leaf["added_at"],
                ))
    return entries


def read_bookmarks_file(path: Path) -> dict | None:
    """Copy-read `path`: the browser may hold its `Bookmarks` file open, so
    this never opens it for write and never keeps a handle on the original
    -- it copies the bytes to a throwaway temp file first and parses that.
    Returns None if the file doesn't exist; raises on a copy/parse failure
    so the caller can record it as an error rather than silently skipping.
    """
    if not path.exists():
        return None
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="lmsa-bookmarks-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(path, tmp_path)
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def load_browser_bookmarks(config: Config) -> tuple[list[BookmarkEntry], list[tuple[str, str]]]:
    """Every bookmark from every configured browser. A missing file (browser
    not installed / no bookmarks yet) is silently skipped; a present-but-
    unreadable/corrupt one is recorded as an error -- either way, one
    browser's problem never stops the other's bookmarks (or Gmail) from
    being processed this run."""
    entries: list[BookmarkEntry] = []
    errors: list[tuple[str, str]] = []
    for source, path in (("chrome", config.chrome_path), ("edge", config.edge_path)):
        if path is None:
            continue
        try:
            data = read_bookmarks_file(path)
            if data is None:
                continue
            entries.extend(parse_bookmarks_json(data, source))
        except Exception as exc:  # noqa: BLE001 - isolate one browser's failure
            errors.append((source, f"{type(exc).__name__}: {exc}"))
    return entries, errors


# -- self-sent links (Gmail) -----------------------------------------------------


def extract_urls(text: str | None) -> list[str]:
    """http(s) URLs found in `text`, in order, de-duplicated, with trailing
    sentence/markup punctuation stripped off each match."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip('.,;:!?)\'"]')
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def self_sent_query(addr: str) -> str:
    return f"from:{addr} to:{addr}"


def _cursor(conn: sqlite3.Connection) -> GmailCursor:
    return GmailCursor(conn, KV_NS, since_key=CURSOR_KEY)


def _is_applied(conn: sqlite3.Connection, msg_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM bookmarks_mail_applied WHERE msg_id=?", (msg_id,)).fetchone()
    return row is not None


def _mark_applied(conn: sqlite3.Connection, msg_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO bookmarks_mail_applied(msg_id, applied_at) VALUES (?, ?)",
        (msg_id, _now_iso()),
    )


def collect_self_sent_entries(
    conn: sqlite3.Connection,
    gmail,
    addr: str,
    *,
    limit: int = SELF_SENT_SEARCH_LIMIT,
    lookback_days: int = DEFAULT_SELF_SENT_LOOKBACK_DAYS,
) -> tuple[list[BookmarkEntry], Window, list[tuple[str, str]]]:
    """Links pulled from messages you sent to yourself, incrementally (the
    shared `lmsa.gmail_cursor` scheme -- see `tasks.packages`). A message
    with no links still gets marked applied (concluded: "no bookmarks
    here"), so it is never rescanned. A message that fails outright (a
    malformed body) is left off the applied ledger, so the cursor holds at
    it and it's retried next run -- it never blocks the cursor from
    advancing past messages that *did* succeed, since `advance()` only
    needs every candidate to be `done`.
    """
    cursor = _cursor(conn)

    def concluded(msg_id: str) -> bool:
        return _is_applied(conn, msg_id)

    cursor.start()
    window = cursor.scan(
        gmail, self_sent_query(addr),
        default_since=_now() - timedelta(days=lookback_days),
        limit=limit, skip=concluded,
    )
    messages = cursor.fetch(gmail, window)

    entries: list[BookmarkEntry] = []
    errors: list[tuple[str, str]] = []
    for msg in messages:
        try:
            urls = extract_urls(msg.text)
            event_date = msg.date or msg.internal_date
            added_at = event_date.isoformat() if event_date else None
            context = f"self-sent: {msg.subject}" if msg.subject else "self-sent"
            for url in urls:
                entries.append(BookmarkEntry(
                    url=url, title=msg.subject or url, source="gmail",
                    context=context, added_at=added_at,
                ))
            _mark_applied(conn, msg.msg_id)
        except Exception as exc:  # noqa: BLE001 - one bad message must not abort the scan
            errors.append((msg.msg_id, f"{type(exc).__name__}: {exc}"))

    cursor.advance(window, done=concluded)
    conn.commit()
    return entries, window, errors


# -- merge into the store ---------------------------------------------------------


def normalize_bookmark_url(raw: str) -> str | None:
    try:
        return normalize_url(raw)
    except (ValueError, UnicodeError):
        return None


@dataclass
class SeenStats:
    new: int = 0
    seen_again: int = 0
    invalid: int = 0


def upsert_seen(conn: sqlite3.Connection, entries: list[BookmarkEntry]) -> SeenStats:
    """Merge `entries` into `bookmarks` (one row per normalized URL,
    deduplicated across every source/folder it showed up in) and
    `bookmark_sources` (one row per distinct (url, source, context), so a
    URL bookmarked in two folders -- or found in two self-sent emails --
    keeps every one of those, not just the first). An unparseable URL
    (rare: a malformed href in a bookmark file) is counted, not fetched,
    and never reaches the `bookmarks` table."""
    now = _now_iso()
    stats = SeenStats()
    for e in entries:
        url = normalize_bookmark_url(e.url)
        if url is None:
            stats.invalid += 1
            continue
        row = conn.execute("SELECT title FROM bookmarks WHERE url=?", (url,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO bookmarks(url, title, status, first_seen, last_seen) "
                "VALUES (?, ?, 'new', ?, ?)",
                (url, e.title or "", now, now),
            )
            stats.new += 1
        else:
            title = row[0] or (e.title or "")
            conn.execute(
                "UPDATE bookmarks SET last_seen=?, title=? WHERE url=?", (now, title, url),
            )
            stats.seen_again += 1
        conn.execute(
            "INSERT OR IGNORE INTO bookmark_sources(url, source, context, added_at, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (url, e.source, e.context or "", e.added_at, now),
        )
    conn.commit()
    return stats


# -- fetch + tag ------------------------------------------------------------------


def pending_urls(conn: sqlite3.Connection, limit: int) -> list[str]:
    """Up to `limit` never-checked URLs, oldest-first-seen -- deterministic
    and resumable across runs, same reasoning as `tasks.doc_index`."""
    rows = conn.execute(
        "SELECT url FROM bookmarks WHERE checked_at IS NULL ORDER BY first_seen, url LIMIT ?", (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def count_pending(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM bookmarks WHERE checked_at IS NULL").fetchone()[0]


def classify_fetch_error(exc: FetchError) -> tuple[str, str]:
    """`(status, reason)`. `status` is one of:
    - "dead": a definitive failure -- HTTP 404/410, or a DNS lookup that
      failed outright.
    - "unchecked": everything the ticket says isn't "dead" -- robots.txt
      denial, any other HTTP error status (incl. 5xx), a body too large to
      read, or a transient network failure (timeout, reset, TLS, ...).
    - "skipped": never attempted as a network fetch at all -- a scheme the
      Fetcher doesn't handle (file:, javascript:, ...), a host resolving to
      a private/intranet address, or an unparseable URL.
    """
    if isinstance(exc, HTTPStatusError):
        if exc.status in (404, 410):
            return "dead", f"HTTP {exc.status}"
        return "unchecked", f"HTTP {exc.status}"
    if isinstance(exc, RobotsDenied):
        return "unchecked", "robots.txt denied this URL"
    if isinstance(exc, TooLarge):
        return "unchecked", "response body too large to read"
    if isinstance(exc, BlockedURL):
        message = str(exc)
        if "cannot resolve" in message:
            return "dead", "DNS resolution failed"
        return "skipped", message
    return "unchecked", f"{type(exc).__name__}: {exc}"


def _mark_checked(
    conn: sqlite3.Connection, url: str, *, status: str, http_status: int | None = None,
    check_error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE bookmarks SET status=?, http_status=?, check_error=?, checked_at=? WHERE url=?",
        (status, http_status, check_error, _now_iso(), url),
    )
    conn.commit()


def _mark_ok(
    conn: sqlite3.Connection, url: str, *, http_status: int, summary: str, tags: list[str],
    btype: str, title: str,
) -> None:
    conn.execute(
        "UPDATE bookmarks SET status='ok', http_status=?, check_error=NULL, checked_at=?, "
        "summary=?, tags_json=?, type=?, title=CASE WHEN title='' THEN ? ELSE title END WHERE url=?",
        (http_status, _now_iso(), summary, json.dumps(tags), btype, title or "", url),
    )
    conn.commit()


@dataclass
class FetchResult:
    url: str
    status: str  # "ok" | "dead" | "unchecked" | "skipped" | "error"
    reason: str | None = None
    tags: list[str] = field(default_factory=list)
    type: str | None = None
    summary: str | None = None
    content_truncated: bool = False


def process_url(lm, fetcher: Fetcher, conn: sqlite3.Connection, url: str) -> FetchResult:
    """Fetch, classify, and (on success) tag + embed one URL. Never raises:
    every failure mode -- a network/policy failure, or the model/embed call
    failing on a page that *did* fetch -- is isolated onto the returned
    result and recorded on the row, so one bad bookmark can't abort the run.
    """
    try:
        resp = fetcher.get(url)
    except FetchError as exc:
        status, reason = classify_fetch_error(exc)
        _mark_checked(conn, url, status=status, http_status=getattr(exc, "status", None), check_error=reason)
        return FetchResult(url=url, status=status, reason=reason)
    except Exception as exc:  # noqa: BLE001 - defensive: an unexpected Fetcher failure
        reason = f"{type(exc).__name__}: {exc}"
        _mark_checked(conn, url, status="unchecked", check_error=reason)
        return FetchResult(url=url, status="unchecked", reason=reason)

    try:
        article = extract_article(resp.text, resp.url)
        text = (article.text or "")[:CONTENT_CHAR_CAP]
        truncated = len(article.text or "") > CONTENT_CHAR_CAP
        prompt = f"Title: {article.title}\n\nURL: {resp.url}\n\n{text}"
        extracted = lm.extract(prompt, EXTRACT_SCHEMA, system=EXTRACT_SYSTEM, name="bookmark")

        summary = str(extracted.get("summary") or "").strip()
        tags = sorted({
            t.strip().lower() for t in (extracted.get("tags") or []) if isinstance(t, str) and t.strip()
        })
        btype = extracted.get("type")
        if btype not in EXTRACT_TYPES:
            btype = "other"

        _mark_ok(conn, url, http_status=resp.status, summary=summary, tags=tags, btype=btype, title=article.title)

        embed_value = "\n".join(part for part in (article.title, summary, " ".join(tags)) if part)
        if embed_value:
            embed_and_upsert(conn, lm, ns=NS, model=lm.cfg.embed_model, items=[(url, embed_value)])

        return FetchResult(
            url=url, status="ok", tags=tags, type=btype, summary=summary, content_truncated=truncated,
        )
    except Exception as exc:  # noqa: BLE001 - fetched fine, tagging/embedding failed
        reason = f"{type(exc).__name__}: {exc}"
        _mark_checked(conn, url, status="error", http_status=resp.status, check_error=reason)
        return FetchResult(url=url, status="error", reason=reason)


# -- semantic search (--query) ----------------------------------------------------


@dataclass
class SearchMatch:
    url: str
    score: float
    title: str
    summary: str
    tags: list[str]
    type: str | None


def build_search_matches(conn: sqlite3.Connection, hits: list[tuple[str, float]]) -> list[SearchMatch]:
    matches = []
    for url, score in hits:
        row = conn.execute(
            "SELECT title, summary, tags_json, type FROM bookmarks WHERE url=?", (url,),
        ).fetchone()
        if row is None:
            matches.append(SearchMatch(url=url, score=score, title="", summary="", tags=[], type=None))
            continue
        title, summary, tags_json, btype = row
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (TypeError, ValueError):
            tags = []
        matches.append(SearchMatch(
            url=url, score=score, title=title or "", summary=summary or "", tags=list(tags), type=btype,
        ))
    return matches


def build_search_report(query: str, matches: list[SearchMatch]) -> Report:
    r = Report(
        "Bookmark search",
        kicker="lmsa · bookmarks",
        subtitle=f'Top {len(matches)} match(es) for "{query}".',
        sticker=f"{len(matches)} match(es)" if matches else "no matches",
        sticker_tone="acc" if matches else "warn",
        source="tasks.bookmarks",
    )
    if not matches:
        r.section("Matches", tone="warn").callout(
            "No indexed bookmarks matched. Has a plain `tasks.bookmarks` run indexed any yet?",
            label="empty", tone="warn",
        )
        return r
    r.section("Matches", tone="acc").table(
        ["Bookmark", "Score", "Type", "Tags", "Summary"],
        [
            [
                Link(m.title or m.url, safe_href(m.url)),
                round(m.score, 4),
                m.type or "—",
                ", ".join(m.tags) or "—",
                m.summary or "—",
            ]
            for m in matches
        ],
    )
    return r


# -- report -------------------------------------------------------------------------


def _dead_links(conn: sqlite3.Connection, limit: int = REPORT_ROWS_CAP) -> tuple[list[tuple], int]:
    total = conn.execute("SELECT COUNT(*) FROM bookmarks WHERE status='dead'").fetchone()[0]
    rows = conn.execute(
        "SELECT url, title, check_error, checked_at FROM bookmarks WHERE status='dead' "
        "ORDER BY checked_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return rows, total


def _unchecked_links(conn: sqlite3.Connection, limit: int = REPORT_ROWS_CAP) -> tuple[list[tuple], int]:
    total = conn.execute("SELECT COUNT(*) FROM bookmarks WHERE status='unchecked'").fetchone()[0]
    rows = conn.execute(
        "SELECT url, title, check_error, checked_at FROM bookmarks WHERE status='unchecked' "
        "ORDER BY checked_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return rows, total


def build_report(
    conn: sqlite3.Connection,
    seen: SeenStats,
    results: list[FetchResult],
    *,
    browser_errors: list[tuple[str, str]],
    gmail_errors: list[tuple[str, str]],
    skipped: list[FetchResult],
    window: Window | None,
    fetch_limit: int,
    fetch_truncated: bool,
) -> Report:
    ok = [r for r in results if r.status == "ok"]
    dead_now = [r for r in results if r.status == "dead"]
    unchecked_now = [r for r in results if r.status == "unchecked"]
    errors_now = [r for r in results if r.status == "error"]

    r = Report(
        "Bookmarks",
        kicker="lmsa · bookmarks",
        subtitle=f"{seen.new} new bookmark(s) seen, {len(results)} fetched and checked this run.",
        sticker=f"{len(dead_now)} dead link(s)" if dead_now else f"{len(ok)} tagged",
        sticker_tone="hot" if dead_now else "acc",
        source="tasks.bookmarks",
    )
    r.tiles([
        Tile("New bookmarks seen", seen.new, tone="info"),
        Tile("Fetched this run", len(results), tone="acc"),
        Tile("Tagged", len(ok), tone="acc"),
        Tile("Dead links", len(dead_now), tone="hot" if dead_now else "muted"),
        Tile("Couldn't check", len(unchecked_now), tone="warn" if unchecked_now else "muted"),
        Tile("Skipped (not fetched)", len(skipped), tone="muted"),
        Tile("Fetch cap this run", fetch_limit, tone="violet"),
    ])

    if fetch_truncated:
        r.section("Note", tone="warn").callout(
            f"More than {fetch_limit} URL(s) were waiting to be checked; the rest will be "
            "picked up on a later run.",
            label="truncated", tone="warn",
        )
    if any(r_.content_truncated for r_ in ok):
        n = sum(1 for r_ in ok if r_.content_truncated)
        r.section("Note", tone="warn").callout(
            f"{n} page(s) had article text longer than {CONTENT_CHAR_CAP} characters; only the "
            "first chunk was sent to the model.",
            label="truncated content", tone="warn",
        )

    add_pending_note(r, window)

    if ok:
        by_tag: dict[str, list[FetchResult]] = {}
        for item in ok:
            for tag in (item.tags or ["(untagged)"]):
                by_tag.setdefault(tag, []).append(item)
        section = r.section("New bookmarks by tag", tone="acc")
        for tag in sorted(by_tag):
            items = by_tag[tag]
            section.text(f"{tag} ({len(items)})")
            section.table(
                ["Bookmark", "Type", "Summary"],
                [[Link(i.url, safe_href(i.url)), i.type or "—", i.summary or "—"] for i in items],
            )
    else:
        r.section("New bookmarks by tag").text("Nothing newly tagged this run.")

    dead_rows, dead_total = _dead_links(conn)
    if dead_rows:
        section = r.section("Dead links", tone="hot")
        if dead_total > len(dead_rows):
            section.text(f"Showing the {len(dead_rows)} most recently checked of {dead_total} total.")
        section.table(
            ["URL", "Title", "Reason", "Checked"],
            [[Link(title or url, safe_href(url)), title or "—", reason or "—", checked or "—"]
             for url, title, reason, checked in dead_rows],
        )

    unchecked_rows, unchecked_total = _unchecked_links(conn)
    if unchecked_rows:
        section = r.section("Couldn't check", tone="warn")
        section.text(
            "Blocked by robots.txt, timed out, or returned a server error -- not counted as dead."
        )
        if unchecked_total > len(unchecked_rows):
            section.text(f"Showing the {len(unchecked_rows)} most recently checked of {unchecked_total} total.")
        section.table(
            ["URL", "Title", "Reason", "Checked"],
            [[Link(title or url, safe_href(url)), title or "—", reason or "—", checked or "—"]
             for url, title, reason, checked in unchecked_rows],
        )

    if skipped:
        r.section("Skipped (not fetched)", tone="muted").table(
            ["URL", "Reason"], [[s.url, s.reason or "—"] for s in skipped[:REPORT_ROWS_CAP]],
        )

    if errors_now:
        r.section("Tagging errors", tone="hot").table(
            ["URL", "Error"], [[e.url, e.reason or "—"] for e in errors_now],
        )

    all_errors = [(f"browser:{name}", msg) for name, msg in browser_errors] + \
                 [(f"gmail:{msg_id}", msg) for msg_id, msg in gmail_errors]
    if all_errors:
        r.section("Source errors", tone="warn").table(["Source", "Error"], [list(e) for e in all_errors])

    return r


# -- main ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="max never-checked URLs fetched this run")
    parser.add_argument("--query", default=None, help="semantic search over already-tagged bookmarks")
    parser.add_argument("--k", type=int, default=8, help="matches to show for --query")
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()
    # Passed explicitly (not relying on load_config's default parameter,
    # which is bound once at import time) so tests can monkeypatch
    # DEFAULT_CONFIG_PATH and have main() actually see the override.
    config = load_config(DEFAULT_CONFIG_PATH)
    fetch_limit = args.limit if args.limit is not None else config.fetch_limit

    with Store() as store:
        conn = store.conn
        ensure_schema(conn)
        with store.record_run("bookmarks") as run:
            if args.query:
                hits = search(conn, lm, ns=NS, model=lm.cfg.embed_model, query=args.query, k=args.k)
                matches = build_search_matches(conn, hits)
                report_path = build_search_report(args.query, matches).write(out_dir=lm.cfg.output_dir)
                run.summary = {"mode": "query", "query": args.query, "matches": len(matches)}
                run.report_path = report_path
                print(f"{len(matches)} match(es) -> {report_path}")
                return

            browser_entries, browser_errors = load_browser_bookmarks(config)

            gmail_entries: list[BookmarkEntry] = []
            gmail_errors: list[tuple[str, str]] = []
            window: Window | None = None
            try:
                with Gmail() as gmail:
                    addr = gmail.cfg.imap_user
                    if addr:
                        gmail_entries, window, gmail_errors = collect_self_sent_entries(
                            conn, gmail, addr, lookback_days=config.self_sent_lookback_days,
                        )
            except GmailAuthError as exc:
                gmail_errors.append(("(auth)", str(exc)))
            except Exception as exc:  # noqa: BLE001 - network/TLS/IMAP trouble: still report the browser side
                gmail_errors.append(("(connect)", f"{type(exc).__name__}: {exc}"))

            seen = upsert_seen(conn, browser_entries + gmail_entries)

            urls = pending_urls(conn, fetch_limit)
            fetch_truncated = count_pending(conn) > len(urls)

            fetcher = Fetcher(USER_AGENT)
            results = [process_url(lm, fetcher, conn, u) for u in urls]
            skipped = [r for r in results if r.status == "skipped"]

            report_path = build_report(
                conn, seen, results,
                browser_errors=browser_errors, gmail_errors=gmail_errors, skipped=skipped,
                window=window, fetch_limit=fetch_limit, fetch_truncated=fetch_truncated,
            ).write(out_dir=lm.cfg.output_dir)

            run.summary = {
                "new_bookmarks": seen.new,
                "seen_again": seen.seen_again,
                "invalid_urls": seen.invalid,
                "fetched": len(results),
                "tagged": sum(1 for r in results if r.status == "ok"),
                "dead": sum(1 for r in results if r.status == "dead"),
                "unchecked": sum(1 for r in results if r.status == "unchecked"),
                "skipped": len(skipped),
                "errors": sum(1 for r in results if r.status == "error"),
                "fetch_truncated": fetch_truncated,
                "pending_gmail": window.pending if window else 0,
            }
            run.report_path = report_path

    print(f"{len(results)} fetched, {sum(1 for r in results if r.status == 'ok')} tagged -> {report_path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
