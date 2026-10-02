"""lmsa.memo: per-item LLM result memory (#46) and its store migration."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from lmsa import memo as memo_mod
from lmsa.memo import Memo, cached, footer_line, has_keys, memo_version, model_id
from lmsa.report import Report
from lmsa.store import _MIGRATIONS, Store


@pytest.fixture
def db(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        yield store


class Counter:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


# -- hits, misses, versions ------------------------------------------------------

def test_second_lookup_is_a_hit_and_returns_an_equal_value(db):
    memo = Memo(db.conn, "t", "v1")
    compute = Counter({"bucket": "act", "n": 3, "tags": ["a", "b"], "due": None})

    first = memo.cached("k", compute)
    second = memo.cached("k", compute)

    assert compute.calls == 1
    assert first == second == {"bucket": "act", "n": 3, "tags": ["a", "b"], "due": None}
    assert memo.summary() == {"memo_hits": 1, "memo_misses": 1}


def test_a_fresh_result_and_a_hit_are_indistinguishable(db):
    # A tuple becomes a list through JSON; the fresh call must hand back the
    # same shape a later hit will, or downstream could tell them apart.
    fresh = Memo(db.conn, "t", "v1").cached("k", lambda: {"xs": (1, 2)})
    hit = Memo(db.conn, "t", "v1").cached("k", lambda: pytest.fail("should be a hit"))
    assert fresh == hit == {"xs": [1, 2]}


def test_version_mismatch_is_a_miss_and_replaces_the_row(db):
    Memo(db.conn, "t", "v1").cached("k", lambda: {"answer": 1})
    memo2 = Memo(db.conn, "t", "v2")
    compute = Counter({"answer": 2})

    assert memo2.cached("k", compute) == {"answer": 2}
    assert compute.calls == 1
    # the old version no longer hits either: one row per (task, key)
    assert Memo(db.conn, "t", "v1").get("k") is None
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 1


def test_tasks_are_separate_namespaces(db):
    Memo(db.conn, "a", "v").cached("k", lambda: {"x": 1})
    assert Memo(db.conn, "b", "v").get("k") is None


def test_memo_version_changes_with_prompt_schema_or_model():
    base = memo_version(1, "prompt", {"type": "object"}, "model-a")
    assert base == memo_version(1, "prompt", {"type": "object"}, "model-a")
    assert base != memo_version(1, "prompt!", {"type": "object"}, "model-a")
    assert base != memo_version(1, "prompt", {"type": "array"}, "model-a")
    assert base != memo_version(1, "prompt", {"type": "object"}, "model-b")
    assert base != memo_version(2, "prompt", {"type": "object"}, "model-a")


def test_model_id_reads_cfg_and_falls_back_to_settings(monkeypatch):
    class LM:
        class cfg:
            chat_model = "chat-x"
            vision_model = "vis-y"

    assert model_id(LM()) == "chat-x"
    assert model_id(LM(), "vision_model") == "vis-y"
    monkeypatch.setenv("LMSA_CHAT_MODEL", "from-env")
    from lmsa.config import settings

    assert model_id(object()) == settings().chat_model


# -- failures are never stored ---------------------------------------------------

def test_an_exception_propagates_and_stores_nothing(db):
    memo = Memo(db.conn, "t", "v")
    with pytest.raises(TimeoutError):
        memo.cached("k", Counter(TimeoutError("slow")))
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 0

    retry = Counter({"ok": True})
    assert memo.cached("k", retry) == {"ok": True}
    assert retry.calls == 1  # retried, not served a remembered failure


@pytest.mark.parametrize("value", [None, {}, {"bucket": "act"}])
def test_invalid_or_empty_results_are_returned_but_not_stored(db, value):
    memo = Memo(db.conn, "t", "v")
    valid = lambda r: has_keys(r, ["bucket", "why"])  # noqa: E731

    assert memo.cached("k", lambda: value, valid=valid) == value
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 0


def test_a_non_json_value_is_not_stored(db):
    memo = Memo(db.conn, "t", "v")
    value = {"when": datetime(2026, 1, 1)}
    assert memo.cached("k", lambda: value) is value
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 0


def test_cached_without_a_memo_just_computes():
    compute = Counter({"x": 1})
    assert cached(None, "k", compute) == {"x": 1}
    assert cached(None, "k", compute) == {"x": 1}
    assert compute.calls == 2


# -- fresh ---------------------------------------------------------------------

def test_fresh_ignores_stored_rows_but_still_writes(db):
    Memo(db.conn, "t", "v").cached("k", lambda: {"answer": "old"})
    fresh = Memo(db.conn, "t", "v", fresh=True)
    compute = Counter({"answer": "new"})

    assert fresh.cached("k", compute) == {"answer": "new"}
    assert compute.calls == 1
    assert fresh.summary() == {"memo_hits": 0, "memo_misses": 1}
    assert Memo(db.conn, "t", "v").get("k") == {"answer": "new"}


# -- pruning -------------------------------------------------------------------

def _age_rows(conn, days):
    old = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn.execute("UPDATE llm_memo SET last_seen=?", (old,))
    conn.commit()


def test_prune_drops_rows_not_seen_within_the_window(db):
    memo = Memo(db.conn, "t", "v")
    memo.put("old", {"x": 1})
    _age_rows(db.conn, 200)
    memo.put("new", {"x": 2})

    assert memo.prune(180) == 1
    assert memo.get("old") is None
    assert memo.get("new") == {"x": 2}


def test_a_hit_refreshes_last_seen_so_live_items_survive_pruning(db):
    memo = Memo(db.conn, "t", "v")
    memo.put("k", {"x": 1})
    _age_rows(db.conn, 200)
    assert memo.get("k") == {"x": 1}  # seen again today
    assert memo.prune(180) == 0


def test_maybe_prune_runs_at_most_once_a_day(db, monkeypatch):
    Memo(db.conn, "t", "v").put("k", {"x": 1})  # the first Memo() pruned (nothing) and stamped kv
    _age_rows(db.conn, 400)

    Memo(db.conn, "t", "v")  # within 24 h of the last prune: skipped
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 1

    later = datetime.now(timezone.utc) + timedelta(hours=25)
    monkeypatch.setattr(memo_mod, "_now", lambda: later)
    Memo(db.conn, "t", "v")
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 0


def test_maybe_prune_never_goes_below_the_default_window(db):
    memo = Memo(db.conn, "t", "v")
    memo.put("k", {"x": 1})
    _age_rows(db.conn, 100)
    db.conn.execute("DELETE FROM kv")
    Memo(db.conn, "t", "v", prune_days=30)  # a caller asking for less still keeps 180 days
    assert memo.get("k") == {"x": 1}


# -- fail-soft -----------------------------------------------------------------

def test_a_broken_table_degrades_to_misses_not_errors(db):
    memo = Memo(db.conn, "t", "v")
    db.conn.execute("DROP TABLE llm_memo")
    compute = Counter({"x": 1})
    assert memo.cached("k", compute) == {"x": 1}
    assert memo.cached("k", compute) == {"x": 1}
    assert compute.calls == 2


# -- reporting -----------------------------------------------------------------

def test_footer_line_and_report_footer(db):
    memo = Memo(db.conn, "t", "v")
    memo.cached("a", lambda: {"x": 1})
    memo.cached("a", lambda: {"x": 1})
    line = footer_line(memo, "2026-09-24T07:00:00+00:00")
    local = datetime(2026, 9, 24, 7, 0, tzinfo=timezone.utc).astimezone()
    assert line == f"Memory: 1 reused, 1 new LLM calls · last successful run {local:%Y-%m-%d %H:%M}"
    assert footer_line(memo, None).endswith("last successful run never")

    r = Report("t")
    r.footer_note("<b>" + line)
    html = r.render()
    assert "&lt;b&gt;Memory: 1 reused" in html  # escaped, inside the footer
    assert html.index("Memory:") > html.index("<footer>")


# -- store: migration + last_successful_run ----------------------------------

def test_migration_adds_llm_memo_to_an_existing_v1_db(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_MIGRATIONS[0])
    conn.execute("PRAGMA user_version = 1")
    conn.execute("INSERT INTO runs(task, started, status) VALUES ('tasks.x', '2026-01-01', 'ok')")
    conn.commit()
    conn.close()

    for _ in range(2):  # idempotent: the second open is a no-op
        with Store(path) as store:
            assert store.conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
            cols = {r[1] for r in store.conn.execute("PRAGMA table_info(llm_memo)")}
            assert cols == {"task", "item_key", "version", "value_json", "created", "last_seen"}
            assert store.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_old_code_still_works_against_a_migrated_db(tmp_path, monkeypatch):
    path = tmp_path / "lmsa.sqlite"
    with Store(path) as store:
        Memo(store.conn, "t", "v").put("k", {"x": 1})

    # "Old code" = the store with only migration 1: it must open the newer
    # db without error or downgrade, and record runs as before.
    from lmsa import store as store_mod

    monkeypatch.setattr(store_mod, "_MIGRATIONS", _MIGRATIONS[:1])
    with Store(path) as store:
        with store.record_run("tasks.x") as run:
            run.summary = {"n": 1}
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
        assert store.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 1


def test_last_successful_run_normalises_task_names(db):
    rows = [
        ("tasks.travel", "2026-09-01T00:00:00+00:00", "ok"),
        ("travel", "2026-09-03T00:00:00+00:00", "ok"),
        ("tasks.travel", "2026-09-05T00:00:00+00:00", "error"),
        ("tasks.travel", "2026-09-06T00:00:00+00:00", "running"),
        ("tasks.other", "2026-09-07T00:00:00+00:00", "ok"),
    ]
    db.conn.executemany("INSERT INTO runs(task, started, status) VALUES (?, ?, ?)", rows)
    db.conn.commit()

    assert db.last_successful_run("tasks.travel") == "2026-09-03T00:00:00+00:00"
    assert db.last_successful_run("travel") == "2026-09-03T00:00:00+00:00"
    assert db.last_successful_run("tasks.nope") is None


def test_values_are_stored_as_json_only(db):
    Memo(db.conn, "t", "v").put("k", {"bucket": "read"})
    (value_json,) = db.conn.execute("SELECT value_json FROM llm_memo").fetchone()
    assert json.loads(value_json) == {"bucket": "read"}


def test_an_empty_key_is_never_memoised(db):
    memo = Memo(db.conn, "t", "v")
    compute = Counter({"x": 1})
    memo.cached("", compute)
    memo.cached("", compute)
    assert compute.calls == 2
    assert db.conn.execute("SELECT COUNT(*) FROM llm_memo").fetchone()[0] == 0
