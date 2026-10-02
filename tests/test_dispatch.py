import json
import os
import sys
import time as _time
import subprocess
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lmsa.gate import GateConfig
from lmsa.models import Lms, ModelError, ModelManager
from lmsa.store import Store
from tasks import dispatch as dp
from tasks.evening_recap import parse_summary

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
CHAT = "qwen/qwen3-14b"
VISION = "qwen/qwen3-vl-8b"
NAMES = {"chat": CHAT, "vision": VISION}


def ny(*args):
    return datetime(*args, tzinfo=NY)


# ---------------------------------------------------------------------------
# cadence + time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("daily 06:30", "daily 06:30"),
    ("Weekly Sunday 3:05", "weekly sun 03:05"),
    ("every 6 hours", "every 6 hours"),
    ("every 1 hour", "every 1 hours"),
])
def test_parse_cadence(text, expected):
    assert str(dp.parse_cadence(text)) == expected


@pytest.mark.parametrize("bad", ["", "daily", "daily 25:00", "daily 6:60", "weekly xyz 01:00", "every 5 hours", "every 0 hours", "hourly"])
def test_parse_cadence_rejects(bad):
    with pytest.raises(ValueError):
        dp.parse_cadence(bad)


def test_daily_latest_occurrence_before_and_after_time():
    c = dp.parse_cadence("daily 06:00")
    assert dp.latest_occurrence(c, ny(2026, 9, 24, 7, 0), NY) == ny(2026, 9, 24, 6, 0)
    assert dp.latest_occurrence(c, ny(2026, 9, 24, 5, 59), NY) == ny(2026, 9, 23, 6, 0)
    assert dp.latest_occurrence(c, ny(2026, 9, 24, 6, 0), NY) == ny(2026, 9, 24, 6, 0)


def test_daily_across_spring_forward_keeps_wall_time():
    c = dp.parse_cadence("daily 06:00")
    before = dp.latest_occurrence(c, ny(2026, 3, 7, 12, 0), NY)
    after = dp.latest_occurrence(c, ny(2026, 3, 8, 12, 0), NY)
    assert before.astimezone(NY).hour == after.astimezone(NY).hour == 6
    assert after - before == timedelta(hours=23)


def test_daily_at_a_nonexistent_spring_forward_time_is_one_instant():
    c = dp.parse_cadence("daily 02:30")  # 02:30 doesn't exist on 2026-03-08 in New York
    occ = dp.latest_occurrence(c, ny(2026, 3, 8, 12, 0), NY)
    assert occ.date() == datetime(2026, 3, 8).date()
    assert occ == dp.latest_occurrence(c, ny(2026, 3, 8, 23, 0), NY)  # stable all day


def test_daily_across_fall_back_is_due_once():
    c = dp.parse_cadence("daily 01:30")  # 01:30 happens twice on 2026-11-01
    first = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT
    second_pass = datetime(2026, 11, 1, 6, 45, tzinfo=UTC)  # 01:45 EST, after the repeat
    assert dp.latest_occurrence(c, first, NY) == first
    assert dp.latest_occurrence(c, second_pass, NY) == first  # same key -> not due twice
    nxt = dp.latest_occurrence(c, datetime(2026, 11, 2, 12, 0, tzinfo=UTC), NY)
    assert nxt - first == timedelta(hours=25)


def test_weekly_and_every():
    wk = dp.parse_cadence("weekly sun 03:00")
    assert dp.latest_occurrence(wk, ny(2026, 9, 24, 12, 0), NY) == ny(2026, 9, 20, 3, 0)  # Thu -> last Sun
    assert dp.latest_occurrence(wk, ny(2026, 9, 20, 2, 0), NY) == ny(2026, 9, 13, 3, 0)
    ev = dp.parse_cadence("every 6 hours")
    assert dp.latest_occurrence(ev, ny(2026, 9, 24, 13, 0), NY) == ny(2026, 9, 24, 12, 0)
    assert dp.latest_occurrence(ev, ny(2026, 9, 24, 0, 30), NY) == ny(2026, 9, 24, 0, 0)


def test_in_overnight_including_wraparound():
    w = (time(1, 0), time(7, 0))
    assert dp.in_overnight(ny(2026, 9, 24, 3, 0), w, NY)
    assert not dp.in_overnight(ny(2026, 9, 24, 7, 0), w, NY)
    wrap = (time(23, 0), time(6, 0))
    assert dp.in_overnight(ny(2026, 9, 24, 23, 30), wrap, NY)
    assert dp.in_overnight(ny(2026, 9, 24, 2, 0), wrap, NY)
    assert not dp.in_overnight(ny(2026, 9, 24, 12, 0), wrap, NY)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_shipped_schedule_loads_cleanly_with_an_empty_denylist():
    sched = dp.load_schedule()
    assert sched.problems == []
    assert sched.gate.blocked_processes == ()
    assert len(sched.tasks) >= 20
    assert '# "steam.exe"' in dp.CONFIG_PATH.read_text(encoding="utf-8")


def test_bad_task_entries_are_isolated(tmp_path):
    (tmp_path / "good.py").write_text("")
    (tmp_path / "bad_model.py").write_text("")
    cfg = tmp_path / "s.toml"
    cfg.write_text(
        '[dispatch]\nblocked_processes = ["Steam.exe"]\n'
        '[tasks.good]\ncadence = "daily 01:00"\nmodel = "chat"\n'
        '[tasks.missing]\ncadence = "daily 01:00"\n'
        '[tasks.bad_model]\ncadence = "daily 01:00"\nmodel = "gpt"\n'
        '[tasks."../evil"]\ncadence = "daily 01:00"\n'
        '[tasks.good2]\ncadence = "nope"\n',
        encoding="utf-8",
    )
    sched = dp.load_schedule(cfg, tasks_dir=tmp_path)
    assert [t.name for t in sched.tasks] == ["good"]
    assert len(sched.problems) == 4
    assert sched.gate.blocked_processes == ("Steam.exe",)


def test_nonfinite_threshold_rejected(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text("[dispatch]\nidle_minutes = nan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dp.load_schedule(cfg, tasks_dir=tmp_path)


# ---------------------------------------------------------------------------
# dispatch loop
# ---------------------------------------------------------------------------

def spec(name, cadence="daily 02:00", model=None, deadline_h=6):
    return dp.TaskSpec(name, dp.parse_cadence(cadence), timedelta(hours=deadline_h), model, timedelta(minutes=5))


def sched_with(*specs):
    return dp.Schedule(gate=GateConfig(), tasks=list(specs), overnight_idle_minutes=0)


class FakeModels:
    def __init__(self, resident=()):
        self.resident = set(resident)
        self.ours: list[str] = []
        self.log: list[tuple[str, str]] = []

    def is_resident(self, m):
        return m in self.resident

    def ensure(self, m, size):
        self.log.append(("load", m))
        self.resident.add(m)
        self.ours.append(m)

    def snapshot(self):
        return set(self.resident)

    def adopt_new(self, before):
        new = sorted(m for m in self.resident if m not in before and m not in self.ours)
        self.ours += new
        return new

    def release_all(self):
        for m in self.ours:
            if m in self.resident:
                self.log.append(("unload", m))
                self.resident.discard(m)
        self.ours = []
        return []

    def stop_server(self):
        self.log.append(("stop", ""))


class Gate:
    """Scripted gate: returns `script[i]` for call i, then the last entry."""

    def __init__(self, *script):
        self.script = list(script) or [[]]
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        return list(self.script[i])


@pytest.fixture
def conn(tmp_path):
    with Store(tmp_path / "t.sqlite") as db:
        dp.ensure_table(db.conn)
        yield db.conn


def rows(conn):
    return conn.execute("SELECT task, due_at, outcome, reason FROM dispatch_runs ORDER BY id").fetchall()


def run(sched, conn, *, now, gate=None, models=None, ran=None, dry_run=False, runner=None,
        clock=None, calls=None, on_skips=None):
    ran = ran if ran is not None else []
    calls = calls if calls is not None else []

    def run_task(s, env=None, timeout_s=None):
        ran.append(s.name)
        calls.append((s.name, dict(env or {}), timeout_s))
        return runner(s) if runner else ("ran", None)

    return dp.dispatch(
        sched, conn, gate_fn=gate or Gate([]), models=models or FakeModels(), model_names=NAMES,
        run_task=run_task, now=now, clock=clock or (lambda: now), tz=NY, dry_run=dry_run,
        sleep=lambda s: None, on_skips=on_skips,
    )


def test_due_task_runs_once_then_is_up_to_date(conn):
    s = sched_with(spec("a"))
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), ran=ran)
    assert ran == ["a"] and res.ran[0][0] == "a"
    res2 = run(s, conn, now=ny(2026, 9, 24, 3, 0), ran=ran)
    assert ran == ["a"] and res2.up_to_date[0][2] == "ran"
    run(s, conn, now=ny(2026, 9, 25, 2, 30), ran=ran)  # next day's occurrence
    assert ran == ["a", "a"]


def test_busy_defers_then_runs_when_quiet(conn):
    s = sched_with(spec("a"))
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=Gate(["fullscreen: a fullscreen game (D3D) is running"]))
    assert res.deferred == [("a", "fullscreen: a fullscreen game (D3D) is running")]
    assert rows(conn)[-1][2] == "deferred"
    ran = []
    run(s, conn, now=ny(2026, 9, 24, 3, 0), ran=ran)
    assert ran == ["a"] and rows(conn)[-1][2] == "ran"


def test_still_busy_at_deadline_is_skipped_with_reason(conn):
    s = sched_with(spec("a", deadline_h=2))
    busy = Gate(["gpu: 90% utilization (limit 30%)"])
    run(s, conn, now=ny(2026, 9, 24, 2, 0), gate=busy)
    run(s, conn, now=ny(2026, 9, 24, 3, 30), gate=busy)
    res = run(s, conn, now=ny(2026, 9, 24, 4, 0), gate=busy)
    assert res.skipped == [("a", "deadline passed while busy: gpu: 90% utilization (limit 30%)")]
    assert rows(conn)[-1][2] == "skipped"
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 5, 0), ran=ran)  # quiet now, but that run is closed
    assert ran == [] and res.up_to_date[0][2] == "skipped"


def test_pc_asleep_past_deadline_still_runs_when_it_wakes(conn):
    s = sched_with(spec("a", deadline_h=2))  # due 02:00, PC off until 09:00
    ran = []
    run(s, conn, now=ny(2026, 9, 24, 9, 0), ran=ran)
    assert ran == ["a"]


def test_deadline_clock_starts_at_first_sighting(conn):
    s = sched_with(spec("a", deadline_h=2))
    busy = Gate(["input: busy"])
    run(s, conn, now=ny(2026, 9, 24, 9, 0), gate=busy)  # woke at 09:00, busy
    res = run(s, conn, now=ny(2026, 9, 24, 10, 30), gate=busy)
    assert res.skipped == [] and res.deferred
    res = run(s, conn, now=ny(2026, 9, 24, 11, 0), gate=busy)
    assert [t for t, _ in res.skipped] == ["a"]


def test_deferred_occurrence_is_superseded_by_the_next(conn):
    s = sched_with(spec("a", cadence="every 6 hours", deadline_h=12))
    run(s, conn, now=ny(2026, 9, 24, 0, 30), gate=Gate(["input: busy"]))
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 6, 30), ran=ran)
    assert ran == ["a"]
    assert res.skipped and "superseded" in res.skipped[0][1] and "input: busy" in res.skipped[0][1]
    res = run(s, conn, now=ny(2026, 9, 24, 7, 0))
    assert res.skipped == []  # closed out exactly once


def test_batches_by_model_with_a_single_load_each(conn):
    s = sched_with(
        spec("v1", model="vision"), spec("c1", model="chat"), spec("n1"),
        spec("c2", model="chat", cadence="daily 01:00"),
    )
    models = FakeModels()
    ran = []
    run(s, conn, now=ny(2026, 9, 24, 2, 30), models=models, ran=ran)
    assert ran == ["n1", "c2", "c1", "v1"]
    assert models.log == [("load", CHAT), ("unload", CHAT), ("load", VISION), ("unload", VISION)]


def test_group_gate_asks_for_vram_only_when_model_not_resident(conn):
    s = sched_with(spec("c1", model="chat"), spec("v1", model="vision"))
    gate = Gate([])
    run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=gate, models=FakeModels(resident={CHAT}))
    needs = [c["need_vram_mb"] for c in gate.calls]
    assert needs == [0, 0, 6400]  # initial check, chat already resident, vision needs loading


def test_low_vram_for_one_model_defers_only_that_group(conn):
    s = sched_with(spec("c1", model="chat"), spec("n1"))
    gate = Gate([], ["gpu: 500 MiB VRAM free, need 9224 MiB"])
    ran = []
    models = FakeModels()
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=gate, models=models, ran=ran)
    assert ran == ["n1"]
    assert res.deferred == [("c1", "gpu: 500 MiB VRAM free, need 9224 MiB")]
    assert models.log == []


def test_model_refusal_defers_group(conn):
    class Refusing(FakeModels):
        def ensure(self, m, size):
            raise ModelError("not loading: too little VRAM")

    s = sched_with(spec("c1", model="chat"))
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), models=Refusing())
    assert res.deferred == [("c1", "not loading: too little VRAM")]


def test_watchdog_abort_unloads_and_defers_the_rest(conn):
    s = sched_with(
        spec("c1", model="chat", cadence="daily 01:00"), spec("c2", model="chat"), spec("v1", model="vision"),
    )
    # initial check quiet, chat group check quiet, watchdog before c2 busy.
    gate = Gate([], [], ["fullscreen: a fullscreen game (D3D) is running"])
    models = FakeModels()
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=gate, models=models, ran=ran)
    assert ran == ["c1"]
    assert models.log == [("load", CHAT), ("unload", CHAT)]  # vision never loaded
    assert res.watchdog.startswith("fullscreen")
    assert {t for t, _ in res.deferred} == {"c2", "v1"}
    assert all("fullscreen" in r for _, r in res.deferred)
    # Next quiet check picks up exactly the deferred ones.
    ran2 = []
    run(s, conn, now=ny(2026, 9, 24, 3, 0), ran=ran2)
    assert ran2 == ["c2", "v1"]


def test_watchdog_abort_with_real_manager_unloads_only_ours(conn):
    """Real ModelManager + fake `lms`: the owner's hand-loaded vision model survives."""
    loaded = [VISION]
    calls = []

    def fake_lms(cmd, **kw):
        args = cmd[1:]
        calls.append(args)
        out = ""
        if args[0] == "ps":
            out = json.dumps([{"identifier": m} for m in loaded])
        elif args[0] == "load":
            loaded.append(args[1])
        elif args[0] == "unload":
            loaded.remove(args[1])
        return subprocess.CompletedProcess(cmd, 0, out, "")

    mm = ModelManager(Lms(exe="lms", runner=fake_lms), free_vram_mb=lambda: 9500, headroom_mb=1024, ttl_s=600)
    s = sched_with(spec("c1", model="chat", cadence="daily 01:00"), spec("c2", model="chat"), spec("v1", model="vision"))
    gate = Gate([], [], ["input: busy"])
    run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=gate, models=mm)
    mutations = [c for c in calls if c[0] in ("load", "unload")]
    assert mutations == [["load", CHAT, "--ttl", "600", "-y"], ["unload", CHAT]]
    assert loaded == [VISION]


def test_user_loaded_model_is_used_and_left_loaded(conn):
    loaded = [CHAT]
    calls = []

    def fake_lms(cmd, **kw):
        calls.append(cmd[1:])
        out = json.dumps([{"identifier": m} for m in loaded]) if cmd[1] == "ps" else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    mm = ModelManager(Lms(exe="lms", runner=fake_lms), free_vram_mb=lambda: 100)
    ran = []
    run(sched_with(spec("c1", model="chat")), conn, now=ny(2026, 9, 24, 2, 30), models=mm, ran=ran)
    assert ran == ["c1"]
    assert [c for c in calls if c[0] != "ps"] == []


def test_crashing_task_is_isolated(conn):
    s = sched_with(spec("a", cadence="daily 01:00"), spec("b"))

    def runner(sp):
        if sp.name == "a":
            raise RuntimeError("runner exploded")
        return "ran", None

    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 2, 30), ran=ran, runner=runner)
    assert ran == ["a", "b"]
    assert res.errors == [("a", "RuntimeError: runner exploded")]
    assert [r[2] for r in rows(conn)] == ["error", "ran"]


def test_dry_run_records_nothing_and_loads_nothing(conn):
    s = sched_with(spec("c1", model="chat"), spec("n1"), spec("old", deadline_h=0.5))
    run(s, conn, now=ny(2026, 9, 24, 2, 5), gate=Gate(["input: busy"]))  # 'old' first seen
    before = rows(conn)
    models = FakeModels()
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 3, 0), models=models, ran=ran, dry_run=True)
    assert rows(conn) == before
    assert ran == [] and models.log == []
    assert res.would_run == [("n1", None), ("c1", CHAT)]
    assert [t for t, _ in res.skipped] == ["old"]


def test_dry_run_prints_busy_reasons(conn, capsys):
    res = run(sched_with(spec("a")), conn, now=ny(2026, 9, 24, 12, 0), gate=Gate(["process: steam is running"]), dry_run=True)
    dp.print_plan(res)
    out = capsys.readouterr().out
    assert "gate: process: steam is running" in out and "would defer a" in out


def test_daytime_uses_idle_threshold_and_overnight_relaxes_it(conn):
    s = sched_with(spec("a", cadence="every 1 hours"))
    s.gate = GateConfig(idle_minutes=15)
    s.overnight_idle_minutes = 0
    g = Gate([])
    run(s, conn, now=ny(2026, 9, 24, 12, 30), gate=g)
    run(s, conn, now=ny(2026, 9, 24, 2, 30), gate=g)
    assert [c["idle_minutes"] for c in g.calls] == [15, 0]


# ---------------------------------------------------------------------------
# lock, report, recap integration
# ---------------------------------------------------------------------------

def test_lock_prevents_a_second_dispatcher(tmp_path):
    path = tmp_path / "dispatch.lock"
    with dp.DispatchLock(path):
        with pytest.raises(dp.AlreadyRunning):
            with dp.DispatchLock(path):
                pass
    with dp.DispatchLock(path):  # released
        pass


def test_report_lists_skips_and_escapes(conn, tmp_path):
    now = ny(2026, 9, 24, 12, 0)
    dp.record(conn, "evil", "2026-09-24T06:00:00+00:00", "skipped", "<script>alert(1)</script>", now)
    dp.record(conn, "a", "2026-09-24T06:00:00+00:00", "ran", None, now)
    dp.record(conn, "b", "2026-09-24T06:00:00+00:00", "deferred", "input: busy", now)
    res = dp.Result(now=now, overnight=False, idle_minutes=15, busy=["input: busy"])
    html = dp.build_report(conn, res, now.date(), NY).render()
    assert "Skipped at deadline" in html and "evil" in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_eventful_run_records_dispatch_run_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(dp, "run_task_subprocess", lambda s, env=None, timeout_s=None: ("ran", None))
    s = sched_with(spec("a", cadence="every 1 hours"))
    dp._dispatch_and_record(s, Gate([]), FakeModels(), NAMES)
    with Store() as db:
        r = db.conn.execute("SELECT task, status, report_path, summary_json FROM runs").fetchall()
    assert len(r) == 1 and r[0][0] == "dispatch" and r[0][1] == "ok"
    assert Path(r[0][2]).exists() and json.loads(r[0][3])["ran"] == 1


def test_quiet_noop_run_records_no_dispatch_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    s = sched_with(spec("a", cadence="every 1 hours"))
    dp._dispatch_and_record(s, Gate(["input: busy"]), FakeModels(), NAMES)  # only deferrals
    with Store() as db:
        assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert db.conn.execute("SELECT outcome FROM dispatch_runs").fetchall() == [("deferred",)]


# ---------------------------------------------------------------------------
# review fixes: one model resident, JIT adoption, skips reaching the recap,
# process-tree timeouts, time budget, abandoned rows
# ---------------------------------------------------------------------------

def test_vision_batch_tasks_use_the_vision_model_for_text_calls(conn):
    s = sched_with(spec("c1", model="chat"), spec("v1", model="vision"), spec("n1"))
    calls = []
    run(s, conn, now=ny(2026, 9, 24, 2, 30), calls=calls)
    envs = {name: env for name, env, _ in calls}
    assert envs == {"n1": {}, "c1": {}, "v1": {"LMSA_CHAT_MODEL": VISION}}


def _fake_lms(loaded, calls):
    def runner(cmd, **kw):
        args = cmd[1:]
        calls.append(args)
        out = ""
        if args[0] == "ps":
            out = json.dumps([{"identifier": m} for m in loaded])
        elif args[0] == "load":
            loaded.append(args[1])
        elif args[0] == "unload":
            loaded.remove(args[1])
        return subprocess.CompletedProcess(cmd, 0, out, "")
    return runner


def test_models_jit_loaded_by_a_task_are_unloaded_with_the_batch(conn):
    """A vision task that still JIT-loads the chat model, and a no-model task that
    JIT-loads an embedding model: both are adopted and unloaded; the owner's
    hand-loaded model is left alone."""
    owner = "owner/hand-loaded"
    embed = "text-embedding-nomic-embed-text-v1.5"
    loaded, calls = [owner], []
    mm = ModelManager(Lms(exe="lms", runner=_fake_lms(loaded, calls)), free_vram_mb=lambda: 9500)

    def runner(sp):
        loaded.append({"v1": CHAT, "n1": embed}[sp.name])  # LM Studio JIT-loads during the task
        return "ran", None

    res = run(sched_with(spec("n1"), spec("v1", model="vision")), conn, now=ny(2026, 9, 24, 2, 30),
              models=mm, runner=runner)
    assert loaded == [owner]
    unloads = [c[1] for c in calls if c[0] == "unload"]
    assert sorted(unloads) == sorted([embed, VISION, CHAT])
    assert owner not in unloads
    assert any("JIT-loaded" in n and CHAT in n for n in res.notes)
    order = [c[:2] for c in calls if c[0] in ("load", "unload")]
    assert order.index(["unload", embed]) < order.index(["load", VISION])


def test_adopt_new_ignores_models_present_before():
    loaded, calls = ["a", "b"], []
    mm = ModelManager(Lms(exe="lms", runner=_fake_lms(loaded, calls)), free_vram_mb=lambda: 0)
    before = mm.snapshot()
    loaded.append("c")
    assert mm.adopt_new(before) == ["c"]
    mm.release_all()
    assert loaded == ["a", "b"]


def test_abandoned_row_is_written_before_the_task_and_closed_after(conn):
    seen = []

    def runner(sp):
        seen.append(rows(conn)[-1][2:])
        return "ran", None

    run(sched_with(spec("a")), conn, now=ny(2026, 9, 24, 2, 30), runner=runner)
    assert seen == [("error", dp.ABANDONED)]
    assert rows(conn)[-1][2:] == ("ran", None)


def test_dispatcher_killed_mid_task_leaves_the_run_closed(conn):
    s = sched_with(spec("a"))

    def killed(sp):
        raise KeyboardInterrupt  # stands in for Task Scheduler terminating the dispatcher

    with pytest.raises(KeyboardInterrupt):
        run(s, conn, now=ny(2026, 9, 24, 2, 30), runner=killed)
    ran = []
    res = run(s, conn, now=ny(2026, 9, 24, 3, 0), ran=ran)
    assert ran == []  # not run a second time beside a possible orphan
    assert res.up_to_date == [("a", rows(conn)[0][1], "error")]
    assert rows(conn)[0][3] == dp.ABANDONED


def test_time_budget_stops_starting_tasks_and_caps_timeouts(conn):
    t = {"now": ny(2026, 9, 24, 2, 30)}
    s = sched_with(
        spec("a", cadence="daily 01:00"), spec("b", cadence="daily 01:30"), spec("c"),
        spec("v", model="vision"),
    )
    s.tasks[1] = dp.TaskSpec("b", s.tasks[1].cadence, timedelta(hours=6), None, timedelta(hours=10))
    s.max_run_minutes = 240

    def runner(sp):
        t["now"] += timedelta(minutes=150)
        return "ran", None

    calls = []
    models = FakeModels()
    res = run(s, conn, now=t["now"], clock=lambda: t["now"], runner=runner, calls=calls, models=models)
    assert [c[0] for c in calls] == ["a", "b"]
    assert calls[1][2] == pytest.approx((dp.HARD_LIMIT_MINUTES - 150) * 60)  # capped below the 6 h kill
    assert {name for name, _ in res.deferred} == {"c", "v"}
    assert all("time budget" in r for _, r in res.deferred)
    assert models.log == []  # vision model never loaded once the budget was gone


def test_budget_config_must_stay_under_the_hard_limit(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text("[dispatch]\nmax_run_minutes = 400\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dp.load_schedule(cfg, tasks_dir=tmp_path)


def _write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_run_task_subprocess_env_and_failure(tmp_path, monkeypatch):
    ok = _write_script(tmp_path, "ok.py", "import os, sys\nsys.exit(0 if os.environ.get('LMSA_CHAT_MODEL') == 'vl' else 3)\n")
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(ok)])
    assert dp.run_task_subprocess(spec("a"), {"LMSA_CHAT_MODEL": "vl"}) == ("ran", None)
    assert dp.run_task_subprocess(spec("a"), {}) == ("error", "exit 3: no stderr")
    bad = _write_script(tmp_path, "bad.py", "import sys\nsys.stderr.write('Traceback\\nValueError: bad\\n')\nsys.exit(1)\n")
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(bad)])
    assert dp.run_task_subprocess(spec("a")) == ("error", "exit 1: ValueError: bad")


def test_run_task_subprocess_invokes_the_task_module():
    assert dp.task_command(spec("healthcheck")) == [sys.executable, "-m", "tasks.healthcheck"]


def _alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@pytest.mark.slow
def test_timeout_kills_the_whole_process_tree(tmp_path, monkeypatch):
    pidfile = tmp_path / "child.pid"
    parent = _write_script(tmp_path, "parent.py", (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(parent)])
    outcome, reason = dp.run_task_subprocess(spec("a"), timeout_s=3)
    assert outcome == "error" and "timed out" in reason and "child processes" in reason
    child = int(pidfile.read_text())
    for _ in range(20):
        if not _alive(child):
            break
        _time.sleep(0.25)
    assert not _alive(child)


def test_skips_are_handed_over_before_any_task_runs(conn):
    order = []
    s = sched_with(spec("late", deadline_h=0.5), spec("a", cadence="every 1 hours"))
    run(s, conn, now=ny(2026, 9, 24, 2, 5), gate=Gate(["input: busy"]))  # 'late' first seen, deferred

    def runner(sp):
        order.append(("run", sp.name))
        return "ran", None

    run(s, conn, now=ny(2026, 9, 24, 2, 45), runner=runner,
        on_skips=lambda r: order.append(("skips", [t for t, _ in r.skipped])))
    assert order == [("skips", ["late"]), ("run", "a")]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    with Store(tmp_path / "r.sqlite") as store:
        dp.ensure_table(store.conn)
        yield store


def _dispatch_rows(db):
    return db.conn.execute("SELECT task, summary_json FROM runs WHERE task='dispatch' ORDER BY id").fetchall()


def _skip(db, task, when):
    dp.record(db.conn, task, dp._iso(when), "skipped", "deadline passed while busy: input: busy", when)


def test_record_skips_before_the_recap_marks_them_reported(db):
    now = ny(2026, 9, 24, 12, 0)
    _skip(db, "inbox_brief", now)
    res = dp.Result(now=now, overnight=False, idle_minutes=15, skipped=[("inbox_brief", "x")])
    dp.record_skips(db, res, NY)
    (_, summ), = _dispatch_rows(db)
    tiles, lines = parse_summary(summ)
    assert [(t.label, t.value) for t in tiles] == [("skipped", 1)]
    assert lines == ["skipped_tasks: inbox_brief (due Thu 12:00)"]
    assert dp.unreported_skips(db.conn) == []
    dp.record_skips(db, res, NY)  # nothing unreported: no second row
    assert len(_dispatch_rows(db)) == 1


def test_skip_after_the_recap_is_carried_to_the_next_day(db):
    db.conn.execute("INSERT INTO runs(task, started, finished, status) VALUES ('evening_recap', ?, ?, 'ok')",
                    (dp._iso(ny(2026, 9, 24, 21, 0)),) * 2)
    late = ny(2026, 9, 24, 23, 30)
    _skip(db, "price_watch", late)
    dp.record_skips(db, dp.Result(now=late, overnight=False, idle_minutes=15, skipped=[("price_watch", "x")]), NY)
    assert len(_dispatch_rows(db)) == 1  # logged tonight...
    assert len(dp.unreported_skips(db.conn)) == 1  # ...but still owed to a recap
    dp.record_skips(db, dp.Result(now=late + timedelta(minutes=20), overnight=True, idle_minutes=0), NY)
    assert len(_dispatch_rows(db)) == 1  # nothing new later that night: no extra row
    dp.record_skips(db, dp.Result(now=ny(2026, 9, 25, 1, 0), overnight=True, idle_minutes=0), NY)
    got = _dispatch_rows(db)
    assert len(got) == 2 and "price_watch" in json.loads(got[1][1])["skipped_tasks"]
    assert dp.unreported_skips(db.conn) == []


def test_recap_run_by_the_dispatcher_counts_even_without_a_runs_row(db):
    dp.record(db.conn, "evening_recap", "2026-09-25T01:00:00+00:00", "ran", None, ny(2026, 9, 24, 21, 0))
    assert dp.recap_ran_today(db.conn, ny(2026, 9, 24, 22, 0), NY)
    assert not dp.recap_ran_today(db.conn, ny(2026, 9, 25, 9, 0), NY)


def test_ensure_table_migrates_an_old_dispatch_runs(tmp_path):
    with Store(tmp_path / "old.sqlite") as store:
        store.conn.execute(
            "CREATE TABLE dispatch_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, "
            "due_at TEXT NOT NULL, started TEXT NOT NULL, finished TEXT, outcome TEXT NOT NULL, reason TEXT)"
        )
        dp.ensure_table(store.conn)
        dp.ensure_table(store.conn)  # idempotent
        assert "reported_at" in {r[1] for r in store.conn.execute("PRAGMA table_info(dispatch_runs)")}


def test_end_of_run_summary_does_not_double_count_skips():
    res = dp.Result(now=ny(2026, 9, 24, 3, 0), overnight=True, idle_minutes=0)
    res.ran = [("a", 1.0)]
    res.errors = [("b", "boom")]
    res.skipped = [("c", "deadline")]
    assert dp.summary(res) == {"ran": 1, "errors": 1, "deferred": 0, "failed_tasks": "b"}


def test_skip_summary_clips_long_lists():
    rows_ = [(i, f"task_{i:03d}", "2026-09-24T06:00:00+00:00", "x") for i in range(100)]
    s = dp.skip_summary(rows_, NY)
    assert s["skipped"] == 100 and len(s["skipped_tasks"]) <= dp.REASON_MAX and "(+100 total)" in s["skipped_tasks"]


# ---------------------------------------------------------------------------
# #37: each task's combined output is kept in a log, its path on the row
# ---------------------------------------------------------------------------

def test_run_task_subprocess_keeps_combined_output_in_the_log(tmp_path, monkeypatch):
    bad = _write_script(tmp_path, "bad.py", (
        "import sys\n"
        "print('scanning 3 messages', flush=True)\n"
        "sys.stderr.write('Traceback (most recent call last):\\n  File \"x.py\", line 1\\nValueError: bad\\n')\n"
        "sys.exit(1)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(bad)])
    log = tmp_path / "logs" / "a-1.log"
    result = dp.run_task_subprocess(spec("a"), log_path=log)
    assert result == ("error", "exit 1: ValueError: bad")  # reason text unchanged
    assert result.log_path == log
    text = log.read_text(encoding="utf-8")
    assert "scanning 3 messages" in text and "Traceback (most recent call last):" in text and "ValueError: bad" in text


def test_reason_is_still_the_last_stderr_line_when_stdout_comes_after(tmp_path, monkeypatch):
    script = _write_script(tmp_path, "late.py", (
        "import sys\n"
        "sys.stderr.write('RuntimeError: the real reason\\n'); sys.stderr.flush()\n"
        "print('cleanup output after the error', flush=True)\n"
        "sys.exit(2)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(script)])
    log = tmp_path / "late.log"
    assert dp.run_task_subprocess(spec("a"), log_path=log) == ("error", "exit 2: RuntimeError: the real reason")
    assert "cleanup output after the error" in log.read_text(encoding="utf-8")


def test_without_a_log_path_nothing_is_kept(tmp_path, monkeypatch):
    ok = _write_script(tmp_path, "ok.py", "print('hi')\n")
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(ok)])
    result = dp.run_task_subprocess(spec("a"))
    assert result == ("ran", None) and result.log_path is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ok.py"]


def test_unwritable_log_path_still_runs_the_task(tmp_path, monkeypatch):
    bad = _write_script(tmp_path, "bad.py", "import sys\nsys.stderr.write('boom\\n')\nsys.exit(5)\n")
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(bad)])
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    result = dp.run_task_subprocess(spec("a"), log_path=tmp_path / "afile" / "a.log")  # parent is a file
    assert result == ("error", "exit 5: boom") and result.log_path is None


def _kill_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


@pytest.mark.slow
def test_grandchild_holding_the_output_does_not_hang_the_dispatcher(tmp_path, monkeypatch):
    # The task exits at once but leaves a grandchild that inherited its stdout
    # and stderr and sleeps for a minute. Only the task itself is waited for.
    pidfile = tmp_path / "gc.pid"
    parent = _write_script(tmp_path, "parent.py", (
        "import subprocess, sys\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "print('spawned', flush=True)\n"
        "sys.stderr.write('ValueError: parent failed\\n'); sys.stderr.flush()\n"
        "sys.exit(4)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(parent)])
    log = tmp_path / "gc.log"
    t0 = _time.monotonic()
    try:
        result = dp.run_task_subprocess(spec("a"), timeout_s=45, log_path=log)
        elapsed = _time.monotonic() - t0
    finally:
        if pidfile.exists():
            _kill_pid(int(pidfile.read_text()))
    assert result == ("error", "exit 4: ValueError: parent failed")
    assert elapsed < dp.PIPE_DRAIN_S + 10  # nowhere near the grandchild's 60 s or the 45 s timeout
    assert "spawned" in log.read_text(encoding="utf-8")


@pytest.mark.slow
def test_timeout_with_file_output_still_kills_the_tree_promptly(tmp_path, monkeypatch):
    pidfile = tmp_path / "child.pid"
    parent = _write_script(tmp_path, "parent.py", (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "print('working', flush=True)\n"
        "time.sleep(120)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(parent)])
    log = tmp_path / "t.log"
    t0 = _time.monotonic()
    result = dp.run_task_subprocess(spec("a"), timeout_s=3, log_path=log)
    elapsed = _time.monotonic() - t0
    assert result == ("error", "timed out after 0.05 min (killed with its child processes)")
    assert elapsed < 3 + dp.PIPE_DRAIN_S + 15
    child = int(pidfile.read_text())
    for _ in range(20):
        if not _alive(child):
            break
        _time.sleep(0.25)
    assert not _alive(child)
    assert "working" in log.read_text(encoding="utf-8")


def test_dispatch_puts_the_log_path_on_the_row_before_and_after_the_task(conn, tmp_path):
    seen = {}

    def runner(s, env=None, timeout_s=None, log_path=None):
        seen["during"] = conn.execute("SELECT outcome, reason, log_path FROM dispatch_runs").fetchall()
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("Traceback (most recent call last):\nKeyError: 'x'\n", encoding="utf-8")
        return dp.TaskOutcome("error", "exit 1: KeyError: 'x'", Path(log_path))

    now = ny(2026, 9, 24, 3, 0)
    dp.dispatch(sched_with(spec("a")), conn, gate_fn=Gate([]), models=FakeModels(), model_names=NAMES,
                run_task=runner, now=now, clock=lambda: now, tz=NY, sleep=lambda s: None, log_dir=tmp_path / "logs")
    (outcome, reason, during_path), = seen["during"]
    assert outcome == "error" and reason == dp.ABANDONED  # an abandoned row already points at its log
    assert Path(during_path).parent == tmp_path / "logs" and Path(during_path).name.startswith("a-2026092")
    (o, r, final), = conn.execute("SELECT outcome, reason, log_path FROM dispatch_runs").fetchall()
    assert (o, r, final) == ("error", "exit 1: KeyError: 'x'", during_path)


def test_runners_without_log_path_still_work_with_a_log_dir(conn, tmp_path):
    now = ny(2026, 9, 24, 3, 0)
    res = dp.dispatch(sched_with(spec("a")), conn, gate_fn=Gate([]), models=FakeModels(), model_names=NAMES,
                      run_task=lambda s, env=None, timeout_s=None: ("ran", None), now=now, clock=lambda: now,
                      tz=NY, sleep=lambda s: None, log_dir=tmp_path / "logs")
    assert [t for t, _ in res.ran] == ["a"]
    assert conn.execute("SELECT outcome, log_path FROM dispatch_runs").fetchall() == [("ran", None)]


def test_failed_log_creation_clears_the_path(conn, tmp_path):
    now = ny(2026, 9, 24, 3, 0)
    dp.dispatch(sched_with(spec("a")), conn, gate_fn=Gate([]), models=FakeModels(), model_names=NAMES,
                run_task=lambda s, env=None, timeout_s=None, log_path=None: dp.TaskOutcome("error", "exit 1: x", None),
                now=now, clock=lambda: now, tz=NY, sleep=lambda s: None, log_dir=tmp_path / "logs")
    assert conn.execute("SELECT log_path FROM dispatch_runs").fetchall() == [(None,)]


def test_ensure_table_adds_log_path_to_an_existing_table(tmp_path):
    with Store(tmp_path / "old.sqlite") as store:
        store.conn.execute(
            "CREATE TABLE dispatch_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, "
            "due_at TEXT NOT NULL, started TEXT NOT NULL, finished TEXT, outcome TEXT NOT NULL, reason TEXT, "
            "reported_at TEXT)"
        )
        store.conn.execute("INSERT INTO dispatch_runs(task, due_at, started, outcome) VALUES ('a','x','y','ran')")
        dp.ensure_table(store.conn)
        dp.ensure_table(store.conn)  # idempotent
        assert "log_path" in {r[1] for r in store.conn.execute("PRAGMA table_info(dispatch_runs)")}
        assert store.conn.execute("SELECT task, log_path FROM dispatch_runs").fetchall() == [("a", None)]


def test_real_dispatch_keeps_a_log_under_the_store_and_prunes_old_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("LMSA_LOG_RETENTION_DAYS", raising=False)
    logs = tmp_path / "dispatch" / "logs"
    logs.mkdir(parents=True)
    old, fresh = logs / "old-20200101-000000.log", logs / "fresh-20260924-000000.log"
    old.write_text("old", encoding="utf-8")
    fresh.write_text("fresh", encoding="utf-8")
    month_ago = _time.time() - 31 * 86400
    os.utime(old, (month_ago, month_ago))
    bad = _write_script(tmp_path, "bad.py", "import sys\nprint('out')\nsys.stderr.write('ValueError: nope\\n')\nsys.exit(1)\n")
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(bad)])
    dp._dispatch_and_record(sched_with(spec("a", cadence="every 1 hours")), Gate([]), FakeModels(), NAMES)
    assert not old.exists() and fresh.exists()
    with Store() as db:
        (outcome, reason, log_path), = db.conn.execute("SELECT outcome, reason, log_path FROM dispatch_runs").fetchall()
    assert (outcome, reason) == ("error", "exit 1: ValueError: nope")
    assert Path(log_path).parent == logs and "ValueError: nope" in Path(log_path).read_text(encoding="utf-8")


# review round 2: the reason matches communicate()'s on awkward stderr; the
# log is final (and deletable) even while an orphaned grandchild lives on

def _old_reason(stderr: bytes, returncode: int) -> str:
    """The reason logic before #37: all of stderr via communicate(), text mode."""
    text = stderr.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return f"exit {returncode}: {(lines[-1] if lines else 'no stderr')[-200:]}"


STDERR_CASES = {
    "big_then_short": b"Error: real reason\n" + b" " * 70000 + b"\n",
    "exact64k": b"y" * 65536 + b"\n",
    "long_line_then_blank": b"x" * 300 + b"END\n\n   \n",
    "unterminated": b"first\nlast without newline",
    "blank_only": b"\n  \n\t\n",
    "crlf": b"one\r\ntwo\r\n",
    "utf8_split": "café ✓ done\n".encode("utf-8"),
    "bad_bytes": b"\xff\xfe broken \x80\n",
    "huge_text_line": b"a" * 200_000 + b"\n" + b" " * 100_000,
    "empty": b"",
}


@pytest.mark.parametrize("name", sorted(STDERR_CASES))
@pytest.mark.parametrize("chunk", [1, 7, 4096, 65536])
def test_last_stderr_line_matches_the_old_communicate_logic(name, chunk):
    data = STDERR_CASES[name]
    if chunk == 1 and len(data) > 5000:
        pytest.skip("byte-at-a-time over a huge input is just slow")
    tracker = dp._LastStderrLine()
    for i in range(0, len(data), chunk):
        tracker.feed(data[i:i + chunk])
    tracker.feed(b"", final=True)
    assert f"exit 1: {(tracker.line() or 'no stderr')[-200:]}" == _old_reason(data, 1)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["big_then_short", "exact64k"])
def test_reason_on_real_subprocess_matches_the_old_logic(tmp_path, monkeypatch, name):
    data = STDERR_CASES[name]
    head, tail = data[:-1], data[-1:]
    script = _write_script(tmp_path, "s.py", (
        "import sys, time\n"
        f"sys.stderr.buffer.write({head!r}); sys.stderr.flush(); time.sleep(0.3)\n"
        f"sys.stderr.buffer.write({tail!r}); sys.stderr.flush()\n"
        "sys.exit(1)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(script)])
    assert dp.run_task_subprocess(spec("a"), log_path=tmp_path / "s.log") == ("error", _old_reason(data, 1))


@pytest.mark.slow
def test_orphaned_grandchild_leaves_a_final_deletable_log(tmp_path, monkeypatch):
    pidfile = tmp_path / "gc.pid"
    parent = _write_script(tmp_path, "parent.py", (
        "import subprocess, sys\n"
        "c = subprocess.Popen([sys.executable, '-c', "
        "'import sys, time; sys.stderr.write(\"gc alive\\\\n\"); sys.stderr.flush(); time.sleep(30); "
        "sys.stderr.write(\"gc late\\\\n\")'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "sys.stderr.write('ValueError: parent failed\\n'); sys.stderr.flush()\n"
        "sys.exit(4)\n"
    ))
    monkeypatch.setattr(dp, "task_command", lambda sp: [sys.executable, str(parent)])
    log = tmp_path / "logs" / "orphan.log"
    try:
        result = dp.run_task_subprocess(spec("a"), timeout_s=45, log_path=log)
        assert result[0] == "error" and result[1].startswith("exit 4: ")
        assert _alive(int(pidfile.read_text()))  # the grandchild really is still holding the handles
        text = log.read_text(encoding="utf-8")
        assert "ValueError: parent failed" in text
        moved = log.with_suffix(".moved")
        log.rename(moved)  # neither our tee nor the grandchild's inherited stdout pins it
        moved.unlink()
        assert not moved.exists()
    finally:
        if pidfile.exists():
            _kill_pid(int(pidfile.read_text()))


# ---------------------------------------------------------------------------
# whisper model class (#62)
# ---------------------------------------------------------------------------

def test_whisper_class_loads_from_schedule(tmp_path):
    (tmp_path / "t.py").write_text("")
    cfg = tmp_path / "s.toml"
    cfg.write_text('[models]\nwhisper_size_mb = 1500\n[tasks.t]\ncadence = "daily 01:00"\nmodel = "whisper"\n',
                   encoding="utf-8")
    sched = dp.load_schedule(cfg, tasks_dir=tmp_path)
    assert sched.problems == [] and sched.tasks[0].model == "whisper"
    assert sched.model_sizes_mb["whisper"] == 1500


def test_shipped_schedule_runs_transcription_in_the_whisper_class():
    by = {t.name: t.model for t in dp.load_schedule().tasks}
    assert by["voice_transcribe"] == "whisper" and by["voice_memos"] == "chat"


def test_whisper_batch_runs_with_no_lm_studio_model_loaded(conn):
    """No-model batch first (its JIT embed model unloaded), then whisper with
    nothing resident and a VRAM check for whisper's size, then chat, then vision."""
    models = FakeModels()
    seen_at_run: dict[str, set] = {}

    def runner(sp):
        seen_at_run[sp.name] = set(models.resident)
        if sp.name == "n1":
            models.resident.add("embed")  # JIT-loaded by the no-model task
        return "ran", None

    gate = Gate([])
    ran, calls = [], []
    s = sched_with(spec("c1", model="chat"), spec("w1", model="whisper"), spec("n1"), spec("v1", model="vision"))
    run(s, conn, now=ny(2026, 9, 24, 2, 30), models=models, ran=ran, runner=runner, gate=gate, calls=calls)
    assert ran == ["n1", "w1", "c1", "v1"]
    assert seen_at_run["w1"] == set()
    assert [c["need_vram_mb"] for c in gate.calls][:2] == [0, 2000]
    assert {n: e for n, e, _ in calls}["w1"] == {}


def test_whisper_waits_while_a_model_it_doesnt_own_is_loaded(conn):
    models = FakeModels(resident={"owner/hand-loaded"})
    ran = []
    res = run(sched_with(spec("w1", model="whisper"), spec("n1")), conn, now=ny(2026, 9, 24, 2, 30),
              models=models, ran=ran)
    assert ran == ["n1"]
    assert res.deferred and res.deferred[0][0] == "w1" and "owner/hand-loaded" in res.deferred[0][1]
    assert models.resident == {"owner/hand-loaded"}  # never unloaded: not ours


def test_whisper_low_vram_defers_only_whisper(conn):
    gate = Gate([], ["gpu: 900 MiB VRAM free, need 3024 MiB"], [])
    ran = []
    res = run(sched_with(spec("w1", model="whisper"), spec("c1", model="chat")), conn,
              now=ny(2026, 9, 24, 2, 30), gate=gate, ran=ran)
    assert ran == ["c1"] and res.deferred == [("w1", "gpu: 900 MiB VRAM free, need 3024 MiB")]


def test_whisper_dry_run(conn):
    res = run(sched_with(spec("w1", model="whisper")), conn, now=ny(2026, 9, 24, 2, 30), dry_run=True)
    assert res.would_run == [("w1", "whisper")] and rows(conn) == []
