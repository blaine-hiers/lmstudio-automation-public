"""lmsa.errors: the Errors feed, confined bounded log reads, pruning."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lmsa import errors as ef
from lmsa.store import Store
from tasks import dispatch as dp

UTC = timezone.utc


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


@pytest.fixture
def dirs(tmp_path):
    dash, disp = tmp_path / "data" / "dashboard" / "logs", tmp_path / "data" / "dispatch" / "logs"
    dash.mkdir(parents=True)
    disp.mkdir(parents=True)
    return dash, disp


@pytest.fixture
def dbp(tmp_path):
    path = tmp_path / "data" / "lmsa.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Store(path) as db:
        dp.ensure_table(db.conn)
    return path


# ---------------------------------------------------------------------------
# confinement + bounded reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    r"\\evil-host\share\x.log", "//evil-host/share/x.log", r"\\?\C:\x.log", r"\\.\pipe\x",
    "relative.log", "", "a\x00b.log", None, 42,
])
def test_confine_refuses_unc_relative_and_junk(dirs, bad):
    assert ef.confine(bad, dirs) is None


def test_confine_is_lexical_and_refuses_escapes(dirs, tmp_path):
    dash, disp = dirs
    assert ef.confine(str(dash / "a.log"), dirs) == dash / "a.log"
    assert ef.confine(str(disp / "sub" / "b.log"), dirs) is not None
    assert ef.confine(str(dash / ".." / ".." / "lmsa.sqlite"), dirs) is None
    assert ef.confine(str(tmp_path / "data" / "dashboard" / "logs-evil" / "a.log"), dirs) is None  # prefix trick
    assert ef.confine(str(dash), dirs) is None  # the dir itself


def test_unc_log_path_never_reaches_the_filesystem(dirs, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("touched the filesystem")

    monkeypatch.setattr(Path, "resolve", boom)
    monkeypatch.setattr("builtins.open", boom)
    monkeypatch.setattr(os, "stat", boom)
    assert ef.read_log(r"\\evil-host\share\x.log", dirs) is None
    assert ef._last_line("//evil-host/share/x.log", dirs) is None


def test_read_log_tails_500_lines_and_show_all_gives_the_rest(dirs):
    dash, _ = dirs
    log = dash / "big.log"
    log.write_text("".join(f"line {i}\n" for i in range(2000)), encoding="utf-8")
    tail = ef.read_log(str(log), dirs)
    assert tail["shown_lines"] == 500 and tail["truncated"]
    assert tail["text"].splitlines()[0] == "line 1500" and tail["text"].endswith("line 1999")
    full = ef.read_log(str(log), dirs, full=True)
    assert full["shown_lines"] == 2000 and not full["truncated"]


def test_read_log_is_bounded_and_tolerates_bad_bytes(dirs, monkeypatch):
    dash, _ = dirs
    monkeypatch.setattr(ef, "TAIL_BYTES", 4096)
    monkeypatch.setattr(ef, "ALL_BYTES", 8192)
    log = dash / "huge.log"
    log.write_bytes(b"\xff\xfe junk \x80\n" * 10 + b"x" * 100_000 + b"\nlast line \xe9\n")
    tail = ef.read_log(str(log), dirs)
    assert tail["size"] > 100_000 and tail["truncated"]
    assert len(tail["text"].encode("utf-8")) <= 4096 * 3  # never more than the bytes read (plus replacement chars)
    assert tail["text"].endswith("last line \ufffd")
    full = ef.read_log(str(log), dirs, full=True)
    assert full["truncated"] and len(full["text"]) <= 8192


def test_read_log_missing_file_is_none(dirs):
    assert ef.read_log(str(dirs[0] / "gone.log"), dirs) is None


def test_symlink_out_of_the_log_dir_is_refused(dirs, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = dirs[0] / "link.log"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("no symlink privilege")
    assert ef.read_log(str(link), dirs) is None


def test_traceback_spans():
    lines = [
        "starting",
        "Traceback (most recent call last):",
        '  File "tasks/x.py", line 3, in <module>',
        "    main()",
        "KeyError: 'total'",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "Traceback (most recent call last):",
        '  File "y.py", line 1',
        "ValueError: bad",
        "after",
    ]
    assert ef.traceback_spans(lines) == [(1, 4), (8, 10)]
    assert ef.traceback_spans(["Traceback (most recent call last):", '  File "a", line 1']) == [(0, 1)]


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------

def _age(path: Path, days: float) -> None:
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_prune_deletes_only_old_logs_not_kept_ones(dirs):
    dash, disp = dirs
    old, fresh, live, other = dash / "old.log", disp / "fresh.log", dash / "live.log", dash / "notes.txt"
    for p in (old, fresh, live, other):
        p.write_text("x", encoding="utf-8")
    for p in (old, live, other):
        _age(p, 40)
    assert ef.prune_logs(dirs, 30, keep=[str(live)]) == 1
    assert not old.exists() and fresh.exists() and live.exists() and other.exists()


def test_prune_never_goes_below_a_day_and_survives_missing_dirs(dirs, tmp_path):
    dash, _ = dirs
    recent = dash / "recent.log"
    recent.write_text("x", encoding="utf-8")
    _age(recent, 0.5)
    assert ef.prune_logs([dash, tmp_path / "nope"], 0) == 0
    assert recent.exists()


@pytest.mark.parametrize("raw,days", [("", 30), ("7", 7), ("0", 1), ("-3", 1), ("abc", 30)])
def test_log_retention_days(monkeypatch, raw, days):
    monkeypatch.setenv("LMSA_LOG_RETENTION_DAYS", raw)
    assert ef.log_retention_days() == days


# ---------------------------------------------------------------------------
# the feed
# ---------------------------------------------------------------------------

def _dispatch_row(db, task, started, finished, outcome, reason, log=None):
    cur = db.conn.execute(
        "INSERT INTO dispatch_runs(task, due_at, started, finished, outcome, reason, log_path) VALUES (?,?,?,?,?,?,?)",
        (task, iso(started), iso(started), iso(finished), outcome, reason, log),
    )
    db.conn.commit()
    return cur.lastrowid


def _runs_row(db, task, started, status, summary=None, report=None, finished=None):
    cur = db.conn.execute(
        "INSERT INTO runs(task, started, finished, status, report_path, summary_json) VALUES (?,?,?,?,?,?)",
        (task, iso(started), iso(finished or started + 1), status, report, json.dumps(summary) if summary else None),
    )
    db.conn.commit()
    return cur.lastrowid


def test_collect_merges_all_three_sources_newest_first(dbp, dirs):
    dash, disp = dirs
    t = time.time() - 3600
    log = disp / "receipts-1.log"
    log.write_text("Traceback (most recent call last):\nKeyError: 'total'\n", encoding="utf-8")
    with Store(dbp) as db:
        d_err = _dispatch_row(db, "receipts", t, t + 60, "error", "exit 1: KeyError: 'total'", str(log))
        inner = _runs_row(db, "tasks.receipts", t + 5, "error", {"error": "KeyError"}, report="output/r.html")
        d_skip = _dispatch_row(db, "labeler", t + 100, t + 100, "skipped", "deadline passed while busy: input")
        _dispatch_row(db, "packages", t + 200, t + 260, "ran", None)
        _runs_row(db, "packages", t + 210, "ok")
        term = _runs_row(db, "tasks.ask", t + 300, "error", {"asked": 1})
    local = [
        ef.LocalRun("abc123", "fakejob", t + 400, t + 410, 3, False, str(dash / "fakejob.log")),
        ef.LocalRun("def456", "brief", t + 500, t + 505, 0, False, str(dash / "brief.log")),
        ef.LocalRun("0a0b0c", "sleeper", t + 600, t + 601, None, True, str(dash / "sleeper.log")),
    ]
    (dash / "fakejob.log").write_text("fakejob start\nValueError: fake\n", encoding="utf-8")
    entries, total = ef.collect(dbp, local, dirs)
    got = [(e.id, e.task, e.source, e.outcome) for e in entries]
    assert got == [
        ("x0a0b0c", "sleeper", "dashboard", "stopped"),
        ("xabc123", "fakejob", "dashboard", "exit 3"),
        (f"r{term}", "ask", "terminal", "error"),
        (f"d{d_skip}", "labeler", "scheduled", "skipped"),
        (f"d{d_err}", "receipts", "scheduled", "error"),
    ]
    assert total == 5
    by = {e.id: e for e in entries}
    assert by[f"d{d_err}"].report_path == "output/r.html"  # the inner runs row was folded in
    assert by[f"d{d_err}"].log_path == str(log)
    assert f"r{inner}" not in by
    assert by["xabc123"].reason == "exit 3: ValueError: fake"
    assert by[f"r{term}"].log_path is None and by[f"r{term}"].summary == {"asked": 1}
    few, total = ef.collect(dbp, local, dirs, limit=2)
    assert len(few) == 2 and total == 5


def test_runs_error_inside_a_successful_run_stands_alone_as_terminal(dbp, dirs):
    t = time.time() - 600
    with Store(dbp) as db:
        _dispatch_row(db, "receipts", t, t + 60, "ran", None, str(dirs[1] / "ok.log"))
        rid = _runs_row(db, "receipts", t + 5, "error", {"error": "partial failure"})
    (e,), _ = ef.collect(dbp, [], dirs)
    assert (e.id, e.source, e.outcome, e.reason, e.log_path) == (f"r{rid}", "terminal", "error", "partial failure", None)


def test_abandoned_row_folds_only_within_the_task_timeout(dbp, dirs):
    # The dispatcher died mid-task a day ago; later terminal failures of the
    # same task must not vanish into that row.
    base = time.time() - 86400
    with Store(dbp) as db:
        d = _dispatch_row(db, "mailer", base, base, "error", dp.ABANDONED)
        inner = _runs_row(db, "tasks.mailer", base + 30, "error", {"error": "inside the scheduled run"})
        one = _runs_row(db, "tasks.mailer", base + 3 * 3600, "error", {"error": "terminal fail ONE"})
        two = _runs_row(db, "tasks.mailer", base + 4 * 3600, "error", {"error": "terminal fail TWO"})
    for timeouts in ({"mailer": 600}, {}, {"mailer": 10**9}):  # schedule, default 60 min, capped at 330 min
        entries, total = ef.collect(dbp, [], dirs, timeouts=timeouts)
        by = {e.id: e for e in entries}
        assert total == 3, timeouts
        assert by[f"d{d}"].summary == {"error": "inside the scheduled run"} and f"r{inner}" not in by
        assert by[f"r{one}"].reason == "terminal fail ONE" and by[f"r{two}"].reason == "terminal fail TWO"


def test_a_host_takes_only_its_nearest_runs_row(dbp, dirs):
    t = time.time() - 3600
    with Store(dbp) as db:
        d = _dispatch_row(db, "receipts", t, t + 600, "error", "exit 1: x")
        near = _runs_row(db, "receipts", t + 2, "error", {"error": "near"})
        far = _runs_row(db, "receipts", t + 300, "error", {"error": "far"})
    entries, total = ef.collect(dbp, [], dirs)
    assert total == 2
    by = {e.id: e for e in entries}
    assert by[f"d{d}"].summary == {"error": "near"} and f"r{near}" not in by
    assert by[f"r{far}"].source == "terminal" and by[f"r{far}"].reason == "far"


def test_a_runs_row_folds_into_the_nearest_of_two_failed_hosts(dbp, dirs):
    t = time.time() - 3600
    with Store(dbp) as db:
        a = _dispatch_row(db, "receipts", t, t + 600, "error", "exit 1: a")
        b = _dispatch_row(db, "receipts", t + 100, t + 200, "error", "exit 1: b")
        _runs_row(db, "receipts", t + 110, "error", {"error": "inner"})
    entries, total = ef.collect(dbp, [], dirs)
    by = {e.id: e for e in entries}
    assert total == 2 and by[f"d{b}"].summary == {"error": "inner"} and by[f"d{a}"].summary is None


def test_dashboard_failures_persist_across_restarts(dbp, dirs):
    dash, _ = dirs
    log = dash / "fakejob-1.log"
    log.write_text('Traceback (most recent call last):\n  File "x"\nValueError: persisted\n', encoding="utf-8")
    t = time.time() - 100
    ef.record_dashboard_run(dbp, ef.LocalRun("abc", "fakejob", t, t + 5, 1, False, str(log)),
                            ["python", "-m", "tasks.fakejob"])
    ef.record_dashboard_run(dbp, ef.LocalRun("def", "fakejob", t + 10, t + 12, 0, False, str(log)))
    unc = "//evil-host/share/x.log"
    ef.record_dashboard_run(dbp, ef.LocalRun("0a1", "brief", t + 20, t + 25, None, True, unc))
    entries, total = ef.collect(dbp, [], dirs)  # nothing in memory: a fresh dashboard
    by = {e.id: e for e in entries}
    assert total == 2 and by["xabc"].source == "dashboard" and by["xabc"].reason == "exit 1: ValueError: persisted"
    assert by["x0a1"].outcome == "stopped" and ef.read_log(by["x0a1"].log_path, dirs) is None  # still confined
    assert ef.latest_failed(dbp, []) == {"brief"}  # fakejob's last run (def) was ok
    # memory wins on the same id
    entries, _ = ef.collect(dbp, [ef.LocalRun("abc", "fakejob", t, t + 5, 0, False, str(log))], dirs)
    assert "xabc" not in {e.id for e in entries}


def test_dashboard_records_are_pruned_with_their_logs(dbp):
    now = time.time()
    ef.record_dashboard_run(dbp, ef.LocalRun("old", "a", now - 40 * 86400, now - 40 * 86400, 1, False, None))
    ef.record_dashboard_run(dbp, ef.LocalRun("new", "a", now - 60, now - 50, 1, False, None))
    assert ef.prune_dashboard_runs(dbp, 30, now) == 1
    entries, total = ef.collect(dbp, [], [])
    assert [e.id for e in entries] == ["xnew"]


def test_an_abandoned_row_that_may_still_be_running_says_so(dbp, dirs):
    t = time.time() - 60
    with Store(dbp) as db:
        _dispatch_row(db, "receipts", t, t, "error", dp.ABANDONED)
        _dispatch_row(db, "labeler", t - 86400, t - 86400, "error", dp.ABANDONED)
    entries, _ = ef.collect(dbp, [], dirs)
    assert [(e.task, e.outcome) for e in entries] == [("receipts", "running?"), ("labeler", "error")]
    assert ef.live_dispatch_logs(dbp) == []  # no log paths on those rows


def test_collect_survives_a_db_without_dispatch_runs_or_log_path(tmp_path, dirs):
    path = tmp_path / "old.sqlite"
    with Store(path) as db:
        _runs_row(db, "ask", time.time(), "error")
    entries, total = ef.collect(path, [], dirs)
    assert total == 1 and entries[0].source == "terminal"
    with Store(path) as db:
        db.conn.execute("CREATE TABLE dispatch_runs (id INTEGER PRIMARY KEY, task TEXT, due_at TEXT, started TEXT, "
                        "finished TEXT, outcome TEXT, reason TEXT)")
        db.conn.execute("INSERT INTO dispatch_runs(task, due_at, started, finished, outcome, reason) "
                        "VALUES ('a', 'x', ?, ?, 'error', 'exit 1: y')", (iso(time.time()),) * 2)
        db.conn.commit()
    entries, total = ef.collect(path, [], dirs)
    assert total == 2 and all(e.log_path is None for e in entries)


def test_latest_failed_uses_the_newest_finished_run_from_any_source(dbp):
    t = time.time() - 3600
    with Store(dbp) as db:
        _runs_row(db, "tasks.ask", t, "error", finished=t + 1)
        _runs_row(db, "tasks.ask", t + 100, "ok", finished=t + 101)  # fixed since
        _runs_row(db, "receipts", t, "ok", finished=t + 1)
        _dispatch_row(db, "receipts", t + 50, t + 60, "error", "exit 1: x")  # then failed on schedule
        _dispatch_row(db, "labeler", t + 50, t + 60, "skipped", "deadline")  # a skip isn't a run
        _dispatch_row(db, "packages", t, t + 5, "error", "exit 1: y")
        _dispatch_row(db, "packages", t + 70, t + 70, "error", dp.ABANDONED)  # running now: no verdict
        _runs_row(db, "running_one", t, "running")
    local = [ef.LocalRun("a1", "fakejob", t + 10, t + 20, 2, False, None),
             ef.LocalRun("a2", "ask", t + 200, t + 210, None, True, None)]
    assert ef.latest_failed(dbp, local) == {"receipts", "packages", "fakejob", "ask"}


def test_gave_up_items_show_ids_and_subjects_only(dbp):
    with Store(dbp) as db:
        db.conn.execute("INSERT INTO kv(ns, key, value_json, updated) VALUES (?,?,?,?)", ("receipts", "retry", json.dumps({
            "m1": {"subject": "Your order <b>#1</b>", "error": "extract failed", "attempts": 3, "gave_up": True},
            "m2": {"subject": "still trying", "error": "x", "attempts": 1, "gave_up": False},
            "m3": "junk",
        }), "now"))
        db.conn.execute("INSERT INTO kv(ns, key, value_json, updated) VALUES ('packages', 'retry', 'not json', 'now')")
        db.conn.execute("INSERT INTO kv(ns, key, value_json, updated) VALUES ('receipts', 'last_run', '\"x\"', 'now')")
        db.conn.commit()
    items, total = ef.gave_up_items(dbp)
    assert total == 1
    assert items == [{"task": "receipts", "msg_id": "m1", "subject": "Your order <b>#1</b>",
                      "attempts": 3, "error": "extract failed"}]


@pytest.mark.parametrize("entry_id,ok", [
    ("d12", True), ("r3", True), ("xabc123", True), ("x", False), ("d", False), ("dabc", False),
    ("../x", False), ("r1/..", False), ("xABC", False),
])
def test_valid_id(entry_id, ok):
    assert ef.valid_id(entry_id) is ok


def test_a_sub_second_scheduled_failure_does_not_swallow_later_failures(dbp, dirs):
    # dispatch_runs stores whole seconds: a task that dies on import finishes
    # "at" its start. That is a real result, not an abandoned placeholder.
    t = time.time() - 7200
    with Store(dbp) as db:
        d = _dispatch_row(db, "mailer", t, t, "error", "exit 1: ModuleNotFoundError: No module named 'x'")
        later = _runs_row(db, "tasks.mailer", t + 20 * 60, "error", {"error": "TERMINAL run 20 min later"})
    entries, total = ef.collect(dbp, [], dirs)
    by = {e.id: e for e in entries}
    assert total == 2
    assert by[f"d{d}"].reason.startswith("exit 1: ModuleNotFoundError") and by[f"d{d}"].summary in (None, {})
    assert by[f"r{later}"].source == "terminal" and by[f"r{later}"].reason == "TERMINAL run 20 min later"
