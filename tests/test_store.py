import json
import sqlite3
import threading
from datetime import datetime

import pytest

from lmsa.store import Store, _set_wal_mode, pack_vector, unpack_vector


def test_migrations_create_expected_tables(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"runs", "kv", "messages", "embeddings"} <= tables


def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        version_after_first_open = store.conn.execute("PRAGMA user_version").fetchone()[0]

    # Reopening an already-migrated db must not error and must not re-run
    # the (non-idempotent-looking) CREATE TABLE statements.
    with Store(db) as store:
        version_after_second_open = store.conn.execute("PRAGMA user_version").fetchone()[0]
        count = store.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    assert version_after_first_open == version_after_second_open
    assert count == 0


def test_record_run_marks_ok_and_attaches_summary(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        with store.record_run("healthcheck") as run:
            run.summary = {"models": 3}
            run.report_path = "output/healthcheck.html"

        row = store.conn.execute(
            "SELECT task, status, report_path, summary_json, started, finished FROM runs WHERE id=?",
            (run.id,),
        ).fetchone()

    assert row[0] == "healthcheck"
    assert row[1] == "ok"
    assert row[2] == "output/healthcheck.html"
    assert row[3] == '{"models": 3}'
    assert row[4] is not None and row[5] is not None


def test_record_run_marks_error_and_reraises(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        run_id = None
        with pytest.raises(RuntimeError):
            with store.record_run("flaky") as run:
                run_id = run.id
                raise RuntimeError("boom")

        row = store.conn.execute(
            "SELECT status, finished FROM runs WHERE id=?", (run_id,)
        ).fetchone()

    assert row[0] == "error"
    assert row[1] is not None


def test_record_run_serialises_non_json_native_summary_values(tmp_path):
    # sets and datetimes aren't JSON-native but `json.dumps(..., default=str)`
    # covers them -- the run must still finish 'ok' and keep the summary.
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        with store.record_run("digest") as run:
            run.summary = {"senders": {"alice", "bob"}, "ran_at": datetime(2026, 1, 1)}

        row = store.conn.execute(
            "SELECT status, finished, summary_json FROM runs WHERE id=?", (run.id,)
        ).fetchone()

    assert row[0] == "ok"
    assert row[1] is not None
    stored = json.loads(row[2])
    assert "alice" in stored["senders"] and "bob" in stored["senders"]
    assert stored["ran_at"] == "2026-01-01 00:00:00"


def test_record_run_finishes_ok_when_summary_is_truly_unserialisable(tmp_path):
    # A dict with a non-string key defeats even `default=str` (it only
    # rescues values, not keys). The row must still be finalised instead of
    # staying stuck at status='running'.
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        with store.record_run("digest") as run:
            run.summary = {("bad", "key"): 1}

        row = store.conn.execute(
            "SELECT status, finished, summary_json FROM runs WHERE id=?", (run.id,)
        ).fetchone()

    assert row[0] == "ok"
    assert row[1] is not None
    stored = json.loads(row[2])
    assert "error" in stored


def test_record_run_reraises_original_error_even_with_bad_summary(tmp_path):
    # The task's own exception must win over any trouble finalising the
    # row -- an unserialisable summary must not replace it with a TypeError.
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        run_id = None
        with pytest.raises(RuntimeError, match="boom"):
            with store.record_run("flaky") as run:
                run_id = run.id
                run.summary = {("bad", "key"): 1}
                raise RuntimeError("boom")

        row = store.conn.execute(
            "SELECT status, finished, summary_json FROM runs WHERE id=?", (run_id,)
        ).fetchone()

    assert row[0] == "error"
    assert row[1] is not None
    assert "error" in json.loads(row[2])


def test_set_wal_mode_does_not_retry_non_lock_errors():
    calls = []

    class FakeConn:
        def execute(self, _sql):
            calls.append(1)
            raise sqlite3.OperationalError("unable to open database file")

    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        _set_wal_mode(FakeConn(), retries=50, delay=0.05)
    assert len(calls) == 1


def test_set_wal_mode_retries_only_lock_contention():
    calls = []

    class FakeConn:
        def execute(self, _sql):
            calls.append(1)
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _set_wal_mode(FakeConn(), retries=3, delay=0.001)
    assert len(calls) == 3


def test_embedding_round_trip_is_exact():
    vector = [0.5, -1.25, 3.0, 0.0, 1e-3]
    blob = pack_vector(vector)
    result = unpack_vector(blob)
    assert result == pytest.approx(vector, rel=0, abs=1e-6)


def test_embedding_round_trip_via_table(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    vector = [1.0, 2.5, -3.75]
    with Store(db) as store:
        store.conn.execute(
            "INSERT INTO embeddings(ns, item_id, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
            ("emails", "msg-1", "nomic-embed-text-v1.5", len(vector), pack_vector(vector)),
        )
        store.conn.commit()

        blob = store.conn.execute(
            "SELECT vector FROM embeddings WHERE ns=? AND item_id=? AND model=?",
            ("emails", "msg-1", "nomic-embed-text-v1.5"),
        ).fetchone()[0]

    assert unpack_vector(blob) == pytest.approx(vector, rel=0, abs=1e-6)


def test_concurrent_open_does_not_raise(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    errors: list[BaseException] = []

    def worker(task_name: str) -> None:
        try:
            with Store(db) as store:
                with store.record_run(task_name) as run:
                    run.summary = {"n": 1}
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"task-{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors

    with Store(db) as store:
        count = store.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert count == 8


def test_named_migrations_apply_once_in_name_order(tmp_path, monkeypatch):
    import lmsa.store as store_mod

    monkeypatch.setattr(
        store_mod,
        "_NAMED_MIGRATIONS",
        {
            "b_second": "CREATE TABLE IF NOT EXISTS b_t (x INTEGER); INSERT INTO order_log VALUES ('b');",
            "a_first": "CREATE TABLE IF NOT EXISTS order_log (name TEXT); INSERT INTO order_log VALUES ('a');",
        },
    )
    db = tmp_path / "t.sqlite"
    with Store(db):
        pass
    with Store(db) as s:  # second open must not re-run the scripts
        log = [r[0] for r in s.conn.execute("SELECT name FROM order_log")]
        applied = {r[0] for r in s.conn.execute("SELECT name FROM schema_migrations")}
    assert log == ["a", "b"]
    assert applied == {"a_first", "b_second"}
