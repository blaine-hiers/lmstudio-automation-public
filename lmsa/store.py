"""Local SQLite store for run history and task data.

`data/lmsa.sqlite` by default; override the path with the `LMSA_DB` env var.
Schema changes are plain SQL scripts applied in order and tracked with
`PRAGMA user_version`, so opening an up-to-date db is a no-op and opening an
old one just runs the scripts it's missing.

Public API (small and meant to stay that way -- later tasks build on it):

    Store() -- context manager. `.conn` is a stdlib `sqlite3.Connection`,
        opened in WAL mode with the schema below already applied. Callers
        are free to run their own SQL against `.conn` for the `kv`,
        `messages`, and `embeddings` tables.

    Store.record_run(task: str) -- context manager yielding a RunHandle
        (mutable `.summary` dict and `.report_path`). Inserts the run row
        on entry, and on exit marks it 'ok' (or 'error', then re-raises)
        and writes back whatever `.summary` / `.report_path` were set to.

    Store.last_successful_run(task: str) -- the `started` time (ISO 8601
        UTC string) of the latest 'ok' run of `task`, or None. Matches both
        spellings `runs.task` has been written with ("tasks.x" and bare "x").

    pack_vector(vector) / unpack_vector(blob) -- float32 <-> BLOB helpers
        for the `embeddings` table's `vector` column.

Tables:
    runs(id, task, started, finished, status, report_path, summary_json)
    kv(ns, key, value_json, updated)
    messages(msg_id PK, thread_id, date, from_addr, subject, labels_json, seen_at)
    embeddings(ns, item_id, model, dim, vector) PK(ns, item_id, model)
    llm_memo(task, item_key, version, value_json, created, last_seen)
        PK(task, item_key) -- owned by `lmsa.memo`

Named migrations: a feature that needs its own tables adds one entry to
`_NAMED_MIGRATIONS` (name -> idempotent SQL) instead of a new numbered
script, so features developed in parallel never compete for the same
`user_version`. They run after the numbered scripts, in name order, once
each; `schema_migrations` records which have been applied.
"""

from __future__ import annotations

import array
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "lmsa.sqlite"

# Applied in order starting from the db's current `PRAGMA user_version`.
# Each entry is idempotent (IF NOT EXISTS) so a race between two processes
# migrating the same fresh db can't raise "table already exists".
_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        started TEXT NOT NULL,
        finished TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        report_path TEXT,
        summary_json TEXT
    );
    CREATE TABLE IF NOT EXISTS kv (
        ns TEXT NOT NULL,
        key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        updated TEXT NOT NULL,
        PRIMARY KEY (ns, key)
    );
    CREATE TABLE IF NOT EXISTS messages (
        msg_id TEXT PRIMARY KEY,
        thread_id TEXT,
        date TEXT,
        from_addr TEXT,
        subject TEXT,
        labels_json TEXT,
        seen_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS embeddings (
        ns TEXT NOT NULL,
        item_id TEXT NOT NULL,
        model TEXT NOT NULL,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        PRIMARY KEY (ns, item_id, model)
    );
    """,
    # 2: per-item LLM result memory (lmsa.memo, #46).
    """
    CREATE TABLE IF NOT EXISTS llm_memo (
        task TEXT NOT NULL,
        item_key TEXT NOT NULL,
        version TEXT NOT NULL,
        value_json TEXT NOT NULL,
        created TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        PRIMARY KEY (task, item_key)
    );
    CREATE INDEX IF NOT EXISTS llm_memo_last_seen ON llm_memo(last_seen);
    """,
)


# Named, order-independent migrations (see module docstring). Each value
# must be idempotent SQL (CREATE ... IF NOT EXISTS). One entry per feature,
# keyed by a short unique name, e.g. "briefing_stories".
_NAMED_MIGRATIONS: dict[str, str] = {
    # #52 briefing engine (lmsa.briefing): one row per story per briefing
    # (latest state; status as of last_run_id), every article it has seen,
    # and one row per briefing run. Vectors live in `embeddings` under
    # ns='briefings' keyed by the rows' item_id.
    "briefing": """
    CREATE TABLE IF NOT EXISTS briefing_stories (
        briefing TEXT NOT NULL,
        story_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        headline TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        score REAL,
        severity INTEGER,
        countries_json TEXT NOT NULL DEFAULT '[]',
        urls_json TEXT NOT NULL DEFAULT '[]',
        data_json TEXT NOT NULL DEFAULT '{}',
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        first_run_id INTEGER,
        last_run_id INTEGER,
        PRIMARY KEY (briefing, story_id)
    );
    CREATE INDEX IF NOT EXISTS briefing_stories_run ON briefing_stories(briefing, last_run_id);
    CREATE TABLE IF NOT EXISTS briefing_articles (
        briefing TEXT NOT NULL,
        url TEXT NOT NULL,
        item_id TEXT NOT NULL,
        story_id TEXT,
        title TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT '',
        published TEXT,
        lean TEXT NOT NULL DEFAULT 'unknown',
        reliability TEXT NOT NULL DEFAULT 'unknown',
        grp TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL DEFAULT '',
        text_hash TEXT NOT NULL DEFAULT '',
        meta_json TEXT NOT NULL DEFAULT '{}',
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        last_run_id INTEGER,
        PRIMARY KEY (briefing, url)
    );
    CREATE INDEX IF NOT EXISTS briefing_articles_item ON briefing_articles(item_id);
    CREATE TABLE IF NOT EXISTS briefing_runs (
        briefing TEXT NOT NULL,
        run_id INTEGER NOT NULL,
        started TEXT NOT NULL,
        status TEXT NOT NULL,
        report_path TEXT,
        summary_json TEXT,
        PRIMARY KEY (briefing, run_id)
    );
    """,
    # #61 check off emails (lmsa.handled): what was checked off, which email
    # cards reports have shown (validates check-offs, feeds the Inbox panel),
    # and the per-run log of Gmail archive / mark-read writes (for undo).
    "handled": """
    CREATE TABLE IF NOT EXISTS handled (
        message_id TEXT PRIMARY KEY,
        uid INTEGER,
        folder TEXT,
        subject TEXT,
        sender TEXT,
        handled_at TEXT NOT NULL,
        source_task TEXT,
        gmail_action TEXT,
        run_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS handled_seen (
        source_task TEXT NOT NULL,
        message_id TEXT NOT NULL,
        run_id INTEGER,
        uid INTEGER,
        folder TEXT,
        subject TEXT,
        sender TEXT,
        web_url TEXT,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        PRIMARY KEY (source_task, message_id)
    );
    CREATE INDEX IF NOT EXISTS handled_seen_msg ON handled_seen(message_id);
    CREATE TABLE IF NOT EXISTS handled_gmail_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        message_id TEXT NOT NULL,
        action TEXT NOT NULL,
        sender TEXT,
        status TEXT NOT NULL DEFAULT 'intent',
        error TEXT,
        logged_at TEXT NOT NULL,
        applied_at TEXT,
        undone_at TEXT
    );
    """,
    # #58 osint movements: compact per-aircraft samples (tasks.osint_air_sample)
    # and extracted USNI Fleet Tracker ship positions (tasks.osint_movements).
    "osint_air_samples": """
        CREATE TABLE IF NOT EXISTS osint_air_samples (
            sampled_at TEXT NOT NULL,
            hex TEXT NOT NULL,
            actype TEXT,
            callsign TEXT,
            lat_bucket INTEGER,
            lon_bucket INTEGER,
            alt_baro INTEGER
        );
        CREATE INDEX IF NOT EXISTS osint_air_samples_sampled_at ON osint_air_samples(sampled_at);
    """,
    "osint_ship_positions": """
        CREATE TABLE IF NOT EXISTS osint_ship_positions (
            fetched_at TEXT NOT NULL,
            article_url TEXT NOT NULL,
            article_title TEXT,
            region TEXT NOT NULL,
            group_name TEXT NOT NULL,
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS osint_ship_positions_fetched_at ON osint_ship_positions(fetched_at);
    """,
    # Per-country daily metrics (#56 risk map; #59 writes "hazard", #60
    # "markets"). Helpers in `lmsa.geo_metrics`.
    "geo_metrics": """
    CREATE TABLE IF NOT EXISTS geo_metrics (
        date TEXT, country TEXT, metric TEXT, value REAL,
        PRIMARY KEY (date, country, metric)
    );
    CREATE INDEX IF NOT EXISTS geo_metrics_country_metric ON geo_metrics(country, metric, date);
    """,
    # #64 photo library captioning + semantic search (tasks.photo_index):
    # one row per photo under LMSA_PHOTOS_DIR. Read-only over the source
    # files -- this table (and the "photos" embeddings namespace) is the
    # only place any of this is written. GPS is personal location data:
    # stored at full precision here, but tasks.doc_search's report only
    # ever shows a coarse place (country, or lat/lon rounded to 1 decimal).
    "photos": """
        CREATE TABLE IF NOT EXISTS photos (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            caption TEXT,
            tags_json TEXT,
            detected_text TEXT,
            taken_at TEXT,
            gps_lat REAL,
            gps_lon REAL,
            country TEXT,
            indexed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS photos_taken_at ON photos(taken_at);
    """,
    # #70 go deep (tasks.go_deep): one row per research report, so the
    # originating briefing story's card can link to its latest deep report.
    "go_deep": """
    CREATE TABLE IF NOT EXISTS go_deep_reports (
        run_id INTEGER PRIMARY KEY,
        story TEXT NOT NULL DEFAULT '',
        topic TEXT NOT NULL,
        report_path TEXT NOT NULL,
        created TEXT NOT NULL,
        stopped_by TEXT,
        summary_json TEXT
    );
    CREATE INDEX IF NOT EXISTS go_deep_reports_story ON go_deep_reports(story, created);
    """,
    # #53 site watch (lmsa.briefing.sitewatch): shared by the `pages`
    # briefing adapter and tasks.site_watch. Snapshots and typed-field
    # history are keyed by (watcher, url), not just url -- two watchers of
    # the same page (the standalone task, a briefing entry, two briefings)
    # each keep their own snapshot lineage, so each independently sees
    # every change rather than whichever runs first consuming the diff.
    # Bounded history both ways: newest N snapshots, newest N field
    # readings, per (watcher, url[, field]).
    "site_watch": """
        CREATE TABLE IF NOT EXISTS site_watch_snapshots (
            watcher TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            checked_at TEXT NOT NULL,
            text TEXT NOT NULL DEFAULT '',
            text_hash TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS site_watch_snapshots_watcher_url
            ON site_watch_snapshots(watcher, url, checked_at);
        CREATE TABLE IF NOT EXISTS site_watch_fields (
            watcher TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL,
            field TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            value_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS site_watch_fields_watcher_url_field
            ON site_watch_fields(watcher, url, field, checked_at);
    """,
    # #60 prediction markets: one daily snapshot per (source, market_id).
    # Helpers in `lmsa.markets`; `tasks.markets` writes it, the briefing
    # extension `lmsa/briefing/adapters/markets.py` reads it.
    "markets": """
        CREATE TABLE IF NOT EXISTS market_snapshots (
            source TEXT NOT NULL,
            market_id TEXT NOT NULL,
            day TEXT NOT NULL,
            question TEXT NOT NULL,
            url TEXT,
            probability REAL,
            country TEXT,
            category TEXT,
            volume REAL,
            closed INTEGER NOT NULL DEFAULT 0,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (source, market_id, day)
        );
        CREATE INDEX IF NOT EXISTS market_snapshots_day ON market_snapshots(day);
        CREATE INDEX IF NOT EXISTS market_snapshots_country ON market_snapshots(country, day);
    """,
    # #57 defense industry briefing: one row per DoD contract-award paragraph
    # parsed off defense.gov's daily contracts page. The identity of an
    # award is the contract number + announcement day (a corrected /
    # re-published announcement of the same contract on the same day, with
    # wording tweaked, is the same award and must upsert in place). When no
    # contract number could be parsed, `identity_key` falls back to the
    # paragraph hash alone -- distinct paragraph text on that day still gets
    # its own row, since there is nothing else to key it on.
    "dod_contracts": """
    CREATE TABLE IF NOT EXISTS dod_contracts (
        identity_key TEXT NOT NULL,
        contract_number TEXT NOT NULL DEFAULT '',
        day TEXT NOT NULL,
        paragraph_hash TEXT NOT NULL,
        service TEXT NOT NULL DEFAULT '',
        contractor TEXT NOT NULL DEFAULT '',
        location TEXT NOT NULL DEFAULT '',
        program_or_item TEXT NOT NULL DEFAULT '',
        contract_type TEXT NOT NULL DEFAULT '',
        completion_date TEXT NOT NULL DEFAULT '',
        amount_usd REAL,
        amount_note TEXT NOT NULL DEFAULT '',
        is_modification INTEGER NOT NULL DEFAULT 0,
        source_url TEXT NOT NULL DEFAULT '',
        raw_text TEXT NOT NULL DEFAULT '',
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        PRIMARY KEY (day, identity_key)
    );
    CREATE INDEX IF NOT EXISTS dod_contracts_day ON dod_contracts(day);
    CREATE INDEX IF NOT EXISTS dod_contracts_contractor ON dod_contracts(contractor);
    CREATE INDEX IF NOT EXISTS dod_contracts_service ON dod_contracts(service);
    """,
    # #101 lmsa.mailer: one row per (name, local date) a real (non-dry-run)
    # send was attempted for -- `already_sent()` reads it, `send()` writes
    # it. A retry after a failure overwrites the row instead of colliding
    # on the primary key.
    "mailer": """
    CREATE TABLE IF NOT EXISTS sends (
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        message_id TEXT,
        status TEXT NOT NULL,
        error_type TEXT,
        created TEXT NOT NULL,
        PRIMARY KEY (name, date)
    );
    """,
}


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a vector of floats into a float32 BLOB for `embeddings.vector`."""
    return array.array("f", vector).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of `pack_vector`."""
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: object) -> str | None:
    """`json.dumps` that never raises.

    `summary` is caller-supplied and may contain things the encoder
    doesn't know (sets, datetimes, numpy scalars, ...); `default=str`
    covers those. A handful of things `default` can't fix either (e.g.
    non-string dict keys) fall back to an error marker instead of
    propagating -- a run finishing is more important than its summary
    round-tripping exactly.
    """
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception as exc:
        return json.dumps({"error": f"summary not JSON-serialisable: {exc}"})


@dataclass
class RunHandle:
    id: int
    task: str
    summary: dict | None = field(default=None)
    report_path: str | Path | None = field(default=None)


def _set_wal_mode(conn: sqlite3.Connection, *, retries: int = 50, delay: float = 0.05) -> None:
    """Switch to WAL journal mode, retrying on the brief lock contention
    that happens when several processes open a fresh db at once.

    SQLite's own busy_timeout doesn't cover this particular transition
    (it needs no other connection mid-open, not just mid-transaction), so
    "database is locked" here is retried by hand instead.
    """
    for attempt in range(retries):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" not in message and "busy" not in message:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i in range(version, len(_MIGRATIONS)):
        conn.executescript(_MIGRATIONS[i])
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    for name in sorted(_NAMED_MIGRATIONS):
        if name in applied:
            continue
        conn.executescript(_NAMED_MIGRATIONS[name])
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, applied) VALUES (?, ?)", (name, _now())
        )
    conn.commit()


class Store:
    """Opens (and migrates) the sqlite db for the duration of a `with` block."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("LMSA_DB", str(DEFAULT_DB)))
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> "Store":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # `timeout` is sqlite's busy timeout: a concurrent writer waits up to
        # 30s for the lock instead of raising `database is locked`.
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        _set_wal_mode(conn)
        _migrate(conn)
        self.conn = conn
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @contextmanager
    def record_run(self, task: str) -> Iterator[RunHandle]:
        """Insert a `runs` row, yield a handle, record ok/error on exit.

        The block may set `run.summary` (a small dict of headline numbers)
        and `run.report_path`; both are written back when the run finishes,
        whether it succeeded or raised.
        """
        assert self.conn is not None, "Store must be used as a context manager"
        cur = self.conn.execute(
            "INSERT INTO runs(task, started, status) VALUES (?, ?, 'running')",
            (task, _now()),
        )
        self.conn.commit()
        run = RunHandle(id=cur.lastrowid, task=task)
        try:
            yield run
        except Exception:
            # The task's own exception always wins: _finish is best-effort
            # here so a problem finalising the row (there shouldn't be one,
            # now that _finish can't raise on a bad summary) can never mask
            # what the task actually raised.
            try:
                self._finish(run, status="error")
            except Exception:
                pass
            raise
        else:
            self._finish(run, status="ok")

    def last_successful_run(self, task: str) -> str | None:
        """`started` of the newest `status='ok'` row for `task`, or None.

        `runs.task` has been recorded both as "tasks.x" and as bare "x", so
        either spelling of `task` matches rows written with either one.
        """
        assert self.conn is not None, "Store must be used as a context manager"
        bare = task[len("tasks."):] if task.startswith("tasks.") else task
        row = self.conn.execute(
            "SELECT started FROM runs WHERE task IN (?, ?) AND status='ok' ORDER BY started DESC LIMIT 1",
            (bare, f"tasks.{bare}"),
        ).fetchone()
        return row[0] if row else None

    def _finish(self, run: RunHandle, *, status: str) -> None:
        assert self.conn is not None
        self.conn.execute(
            "UPDATE runs SET finished=?, status=?, report_path=?, summary_json=? WHERE id=?",
            (
                _now(),
                status,
                str(run.report_path) if run.report_path is not None else None,
                _safe_json(run.summary),
                run.id,
            ),
        )
        self.conn.commit()
