"""Per-item LLM result memory, so snapshot reports don't re-sweep (#46).

Snapshot tasks (inbox_brief, travel, deadlines, ...) must keep showing every
item in their window, so a "since last run" cursor is the wrong tool. What
they reuse instead is the model's answer for an item they've already seen:
the window stays the same, only new or changed items cost an LLM call.

    from lmsa.memo import Memo, cached, has_keys, memo_version, model_id

    memo = Memo(store.conn, "travel", memo_version(MEMO_VERSION, SYSTEM, SCHEMA, model_id(lm)),
                fresh=args.fresh)
    result = cached(memo, msg.msg_id, lambda: lm.extract(...),
                    valid=lambda r: has_keys(r, SCHEMA["required"]))
    ...
    add_footer(report, memo, store.last_successful_run("tasks.travel"))
    run.summary = {..., **memo.summary()}

Rules:
- `version` is a short hash of everything that shapes the answer (prompt,
  schema, model id, a per-task MEMO_VERSION). A stored row only counts as a
  hit when its version matches, so editing a prompt or switching models
  invalidates old results automatically.
- The item key must change whenever the input the model saw changes
  (a Gmail msg_id; thread + latest msg_id; repo#n@updatedAt; path+size+mtime).
- Only successful, valid results are stored: `compute()` raising (timeouts,
  bad JSON, missing keys) stores nothing, and neither does a result the
  `valid` predicate rejects. Either way the item retries next run.
- The value stored is exactly what `compute()` returned (JSON round-tripped),
  so downstream code can't tell a hit from a fresh call. Callers keep values
  to plain JSON types.
- `fresh=True` ignores stored rows (every lookup is a miss) but still writes
  results, so a `--fresh` run refreshes the memory.
- It's a cache: a sqlite error reading or writing it is logged and treated
  as a miss / skipped write, never allowed to fail the run.
- Pruning is opportunistic: at most once per 24 h (tracked in `kv`), rows
  whose `last_seen` is older than `PRUNE_DAYS` are deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

PRUNE_DAYS = 180  # longer than any wired task's window (travel: 120 days)
PRUNE_EVERY = timedelta(hours=24)
_KV_NS = "memo"
_KV_PRUNED = "last_prune"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def memo_version(*parts: object) -> str:
    """A short, stable hash of everything that affects the model's answer."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def model_id(lm: object, attr: str = "chat_model") -> str:
    """The model id `lm` would use (`lm.cfg.<attr>`), falling back to
    `settings()` for a client without a full config (test fakes)."""
    value = getattr(getattr(lm, "cfg", None), attr, None)
    if isinstance(value, str) and value:
        return value
    from .config import settings

    return str(getattr(settings(), attr, "") or "")


def has_keys(value: object, keys: Iterable[str]) -> bool:
    """True if `value` is a dict carrying every one of `keys` -- the usual
    "is this a real answer, not an empty/partial JSON object" check."""
    return isinstance(value, dict) and all(k in value for k in keys)


class Memo:
    """One task's view of the `llm_memo` table (see module docstring)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        task: str,
        version: str,
        *,
        fresh: bool = False,
        prune_days: int = PRUNE_DAYS,
    ):
        self.conn = conn
        self.task = task
        self.version = version
        self.fresh = fresh
        self.hits = 0
        self.misses = 0
        self.maybe_prune(prune_days)

    # -- lookups ---------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """The stored value for `key`, or None (a miss). Only a row with this
        memo's `version` is a hit; a hit refreshes the row's `last_seen`.
        Always a miss when `fresh`. Does not touch the hit/miss counters --
        `cached()` does."""
        if self.fresh:
            return None
        try:
            row = self.conn.execute(
                "SELECT value_json FROM llm_memo WHERE task=? AND item_key=? AND version=?",
                (self.task, key, self.version),
            ).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            self.conn.execute(
                "UPDATE llm_memo SET last_seen=? WHERE task=? AND item_key=?",
                (_now().isoformat(), self.task, key),
            )
            self.conn.commit()
        except (sqlite3.Error, ValueError):
            log.warning("memo read failed for %s/%s; recomputing", self.task, key, exc_info=True)
            return None
        return value

    def put(self, key: str, value: Any) -> None:
        """Store `value` for `key` under this memo's version (replacing any
        older row). A None value, or one that isn't JSON-serialisable, is
        not stored."""
        if value is None:
            return
        try:
            value_json = json.dumps(value)
        except (TypeError, ValueError):
            log.warning("memo value for %s/%s isn't JSON; not stored", self.task, key)
            return
        now = _now().isoformat()
        try:
            self.conn.execute(
                "INSERT INTO llm_memo(task, item_key, version, value_json, created, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(task, item_key) DO UPDATE SET version=excluded.version, "
                "value_json=excluded.value_json, created=excluded.created, last_seen=excluded.last_seen",
                (self.task, key, self.version, value_json, now, now),
            )
            self.conn.commit()
        except sqlite3.Error:
            log.warning("memo write failed for %s/%s", self.task, key, exc_info=True)

    def cached(
        self, key: str, compute: Callable[[], Any], *, valid: Callable[[Any], bool] | None = None
    ) -> Any:
        """`get(key)`, or on a miss `compute()` and store its result -- only
        if `compute()` returned (didn't raise), the value isn't None, and
        `valid(value)` (when given) is true. Exceptions propagate unchanged.
        An empty key (an item with no stable identity) is never memoised."""
        if not key:
            self.misses += 1
            return compute()
        hit = self.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        value = compute()
        if value is not None and (valid is None or valid(value)):
            self.put(key, value)
            # Hand back the same JSON round-trip a later hit would, so a
            # fresh call and a hit are indistinguishable downstream.
            try:
                return json.loads(json.dumps(value))
            except (TypeError, ValueError):
                return value
        return value

    # -- housekeeping ----------------------------------------------------------

    def prune(self, older_than_days: int = PRUNE_DAYS) -> int:
        """Delete rows (every task's) whose `last_seen` is older than
        `older_than_days`. Returns how many were deleted."""
        cutoff = (_now() - timedelta(days=older_than_days)).isoformat()
        cur = self.conn.execute("DELETE FROM llm_memo WHERE last_seen < ?", (cutoff,))
        self.conn.commit()
        return cur.rowcount

    def maybe_prune(self, older_than_days: int = PRUNE_DAYS) -> None:
        """`prune()` unless it already ran in the last 24 h (tracked in kv)."""
        try:
            row = self.conn.execute(
                "SELECT value_json FROM kv WHERE ns=? AND key=?", (_KV_NS, _KV_PRUNED)
            ).fetchone()
            now = _now()
            if row is not None:
                try:
                    last = datetime.fromisoformat(json.loads(row[0]))
                    if last.tzinfo is not None and now - last < PRUNE_EVERY:
                        return
                except (TypeError, ValueError):
                    pass  # unreadable marker: prune and rewrite it
            self.prune(max(older_than_days, PRUNE_DAYS))
            self.conn.execute(
                "INSERT OR REPLACE INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
                (_KV_NS, _KV_PRUNED, json.dumps(now.isoformat()), now.isoformat()),
            )
            self.conn.commit()
        except sqlite3.Error:
            log.warning("memo prune failed", exc_info=True)

    # -- reporting -------------------------------------------------------------

    def summary(self) -> dict:
        """`{"memo_hits": n, "memo_misses": n}` for a run's summary_json."""
        return {"memo_hits": self.hits, "memo_misses": self.misses}

    def line(self) -> str:
        fresh = " (--fresh)" if self.fresh else ""
        return f"Memory: {self.hits} reused, {self.misses} new LLM calls{fresh}"


def cached(
    memo: Memo | None, key: str, compute: Callable[[], Any], *, valid: Callable[[Any], bool] | None = None
) -> Any:
    """`memo.cached(...)`, or just `compute()` when there's no memo."""
    if memo is None:
        return compute()
    return memo.cached(key, compute, valid=valid)


def footer_line(memo: Memo, last_ok: str | None) -> str:
    """"Memory: 184 reused, 6 new LLM calls · last successful run 2026-09-24 07:00"
    (local time, like the "generated" stamp next to it)."""
    when = "never"
    if last_ok:
        try:
            dt = datetime.fromisoformat(last_ok)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            when = f"{dt.astimezone():%Y-%m-%d %H:%M}"  # local, like the footer's "generated"
        except ValueError:
            when = last_ok
    return f"{memo.line()} · last successful run {when}"


def add_footer(report: Any, memo: Memo, last_ok: str | None) -> None:
    """Put `footer_line()` in the report's footer."""
    report.footer_note(footer_line(memo, last_ok))
