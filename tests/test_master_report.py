"""tasks.master_report + lmsa.digest + lmsa.email_render (#103). No test
sends mail, calls a model, or runs a real task: SMTP and the LLM are fakes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lmsa import Tile
from lmsa import mailer as mailer_mod
from lmsa.digest import (FAILED, NOT_RUN, OK, STALE, Digest, DigestItem, RunRow, SectionStatus, normalize,
                         section_status)
from lmsa.email_render import GMAIL_CLIP, EmailSection, render_email
from lmsa.store import Store
from tasks import master_report as mr
from tasks.dispatch import load_schedule

FAKE_PASSWORD = "test-app-pw-7d1e0a55"  # invented
NOW = datetime(2026, 9, 28, 11, 45, tzinfo=timezone.utc)

VALID = """
[[master]]
name = "morning"
title = "Morning Report"
send_at = "06:45"
send = false
llm_intro = false

  [[master.section]]
  task = "inbox_brief"
  mode = "digest"
  max_items = 5

  [[master.section]]
  task = "disasters"
  mode = "headline"

  [[master.section]]
  task = "job_tracker"
  mode = "link"
"""


class _FakeSMTP:
    instances: list["_FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.sent = []
        _FakeSMTP.instances.append(self)

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


class _RefusingSMTP:
    def __init__(self, *a, **k):
        raise AssertionError("a dry run must not open an SMTP connection")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_IMAP_USER", "owner@example.test")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", FAKE_PASSWORD)
    monkeypatch.setattr(mailer_mod, "_smtp_class", _RefusingSMTP)
    _FakeSMTP.instances = []


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _insert(conn, task, hours_ago, status="ok", summary=None, report_path=None):
    conn.execute(
        "INSERT INTO runs(task, started, finished, status, report_path, summary_json) VALUES (?,?,?,?,?,?)",
        (task, _ts(hours_ago), _ts(hours_ago), status, report_path,
         json.dumps(summary) if summary is not None else None),
    )
    conn.commit()


def _no_hooks(task):
    return None


# -- config loader ------------------------------------------------------------


def test_shipped_config_is_valid():
    masters = mr.load_config(local_path=mr.ROOT / "no-such-local.toml")
    m = {x.name: x for x in masters}["morning"]
    assert m.send_at == "06:45" and m.send is False and m.to == []
    assert [(s.task, s.briefing) for s in m.sections] == [
        ("disasters", None), ("followup_radar", None), ("osint_movements", None), ("osint_air_sample", None),
        ("briefing", "defense"), ("briefing", "conflicts"),
    ]


def test_parse_config_valid_text_and_dict_agree():
    import tomllib

    from_text = mr.parse_config(VALID)
    from_dict = mr.parse_config(tomllib.loads(VALID))
    assert from_text == from_dict
    m = from_text[0]
    assert (m.name, m.title, m.enabled, m.skip_empty, m.max_age_hours) == ("morning", "Morning Report", True, True, 24.0)
    assert [(s.task, s.mode, s.max_items) for s in m.sections] == [
        ("inbox_brief", "digest", 5), ("disasters", "headline", 5), ("job_tracker", "link", 5)]


def test_local_overrides_apply():
    m = mr.parse_config(VALID, overrides='[master.morning]\nsend = true\nenabled = false\nto = ["me@example.com"]\n')[0]
    assert (m.send, m.enabled, m.to) == (True, False, ["me@example.com"])


def test_load_config_reads_local_file(tmp_path):
    main, local = tmp_path / "m.toml", tmp_path / "local.toml"
    main.write_text(VALID, encoding="utf-8")
    local.write_text("[master.morning]\nsend = true\n", encoding="utf-8")
    assert mr.load_config(main, local)[0].send is True
    assert mr.load_config(main, tmp_path / "missing.toml")[0].send is False


@pytest.mark.parametrize("patch,needle", [
    (('task = "disasters"', 'task = "no_such_task"'), "unknown task 'no_such_task'"),
    (('mode = "headline"', 'mode = "fancy"'), "unknown mode 'fancy'"),
    (('send_at = "06:45"', 'send_at = "6:45"'), "send_at '6:45'"),
    (('send_at = "06:45"', 'send_at = "24:00"'), "send_at '24:00'"),
    (('send = false', 'send = "yes"'), "send must be bool"),
    (('send = false', 'colour = "red"'), "unknown key 'colour'"),
    (('max_items = 5', 'max_items = 0'), "max_items must be between"),
    (('task = "disasters"', 'task = "dashboard"'), "unknown task 'dashboard'"),
    (('task = "disasters"', 'task = "master_report"'), "unknown task 'master_report'"),
    (('task = "disasters"', 'task = "briefing"'), "needs briefing ="),
    (('mode = "headline"', 'mode = "headline"\n  briefing = "example"'), "only applies to task"),
    (('send_at = "06:45"', 'send_at = "06:45"\nto = ["not an address"]'), "plain email addresses"),
    (('send_at = "06:45"', 'send_at = "06:45"\nmax_age_hours = true'), "max_age_hours must be"),
])
def test_parse_config_rejects_with_a_clear_message(patch, needle):
    bad = VALID.replace(*patch, 1)
    assert bad != VALID
    with pytest.raises(mr.ConfigError) as info:
        mr.parse_config(bad)
    assert any(needle in e for e in info.value.errors), info.value.errors


def test_parse_config_collects_every_error_and_handles_bad_toml():
    with pytest.raises(mr.ConfigError) as info:
        mr.parse_config(VALID.replace('"06:45"', '"x"').replace('"fancy"', '"x"').replace('"headline"', '"x"'))
    assert len(info.value.errors) == 2
    with pytest.raises(mr.ConfigError, match="not valid TOML"):
        mr.parse_config("[[master]\n")
    with pytest.raises(mr.ConfigError, match="at least one"):
        mr.parse_config("")
    with pytest.raises(mr.ConfigError, match="duplicate"):
        mr.parse_config(VALID + VALID)


def test_briefing_section_needs_an_existing_briefing():
    cfg = VALID.replace('task = "disasters"', 'task = "briefing"\n  briefing = "example"')
    assert mr.parse_config(cfg)[0].sections[1].briefing == "example"
    with pytest.raises(mr.ConfigError, match="unknown briefing 'nope'"):
        mr.parse_config(cfg.replace('"example"', '"nope"'))


@pytest.mark.parametrize("local,needle", [
    ("[master.evening]\nsend = true\n", "no master named 'evening'"),
    ("[master.morning]\ntitle = \"x\"\n", "unknown key 'title'"),
    ("[master.morning]\nsend = 1\n", "send must be true or false"),
    ("[other]\n", "unknown table/key 'other'"),
])
def test_local_overrides_are_validated(local, needle):
    with pytest.raises(mr.ConfigError) as info:
        mr.parse_config(VALID, overrides=local)
    assert any(needle in e for e in info.value.errors), info.value.errors


# -- section status -----------------------------------------------------------


def _run(hours_ago, status="ok"):
    return RunRow(id=1, task="tasks.x", started=_ts(hours_ago), status=status)


def test_section_status_ok_stale_failed_not_run():
    assert section_status(_run(2), _run(2), NOW, 24) == SectionStatus(OK, 2.0)
    stale = section_status(_run(31), _run(31), NOW, 24)
    assert stale.kind == STALE and stale.label == "STALE (last good run 31h ago)"
    assert section_status(_run(30), _run(1, "error"), NOW, 24).kind == FAILED
    assert section_status(None, _run(1, "error"), NOW, 24).label == FAILED
    assert section_status(None, None, NOW, 24).label == NOT_RUN


def test_source_runs_ignore_running_rows_so_todays_missing_run_is_stale():
    """inbox_brief hasn't finished today yet: yesterday's run shows STALE, not FAILED."""
    with Store() as store:
        _insert(store.conn, "tasks.inbox_brief", 25.0, summary={"threads": 4})
        _insert(store.conn, "tasks.inbox_brief", 0.1, status="running")
        _insert(store.conn, "disasters", 3.0, summary={"events": 2})  # bare spelling matches too
        _insert(store.conn, "tasks.job_tracker", 30.0)
        _insert(store.conn, "tasks.job_tracker", 5.0, status="error")
        master = mr.parse_config(VALID)[0]
        got = {}
        for s in master.sections:
            ok, latest = mr.source_runs(store.conn, s)
            got[s.task] = section_status(ok, latest, NOW, 24).kind
    assert got == {"inbox_brief": STALE, "disasters": OK, "job_tracker": FAILED}


def test_briefing_section_reads_briefing_runs():
    with Store() as store:
        store.conn.execute("INSERT INTO briefing_runs(briefing, run_id, started, status, summary_json) "
                           "VALUES ('example', 7, ?, 'ok', '{\"stories\": 3}')", (_ts(1),))
        section = mr.SectionConfig(task="briefing", briefing="example")
        ok, latest = mr.source_runs(store.conn, section)
    assert ok.id == 7 and ok.briefing == "example" and latest == ok


# -- digests ------------------------------------------------------------------


def test_fallback_digest_from_summary_json():
    run = RunRow(1, "tasks.disasters", _ts(1), "ok", summary_json=json.dumps(
        {"events": 12, "countries": 4, "note": "text", "nested": {"a": 1}, "ratio": 0.5, "x": 1, "y": 2}))
    d = mr.fallback_digest(run)
    assert [(t.label, t.value) for t in d.tiles][:3] == [("events", 12), ("countries", 4), ("ratio", 0.5)]
    assert d.items == [] and d.callouts == []
    clean, _, _ = normalize(d, 5)
    assert len(clean.tiles) == 4  # contract: at most 4 tiles
    assert mr.fallback_digest(RunRow(1, "x", _ts(1), "ok", summary_json="not json")).is_empty()


def test_hook_is_used_and_a_failing_hook_falls_back():
    run = RunRow(1, "tasks.inbox_brief", _ts(1), "ok", summary_json='{"threads": 9}')
    section = mr.SectionConfig(task="inbox_brief", mode="digest", max_items=2)
    hook = lambda r, n: Digest([Tile("Needs reply", 3)], [DigestItem(f"t{i}") for i in range(5)], ["late"])
    b = mr.build_section(section, run, SectionStatus(OK), lambda t: hook)
    assert [i.title for i in b.email.digest.items] == ["t0", "t1"] and b.email.more == 3

    def boom(r, n):
        raise KeyError("x")
    b = mr.build_section(section, run, SectionStatus(OK), lambda t: boom)
    assert b.email.digest.tiles[0].label == "threads" and "KeyError" in b.email.note


def test_normalize_keeps_only_http_links():
    d, _, _ = normalize(Digest(items=[
        DigestItem("a", url="javascript:alert(1)"), DigestItem("b", url="file:///C:/out/r.html"),
        DigestItem("c", url="https://example.com/x"), DigestItem("d\nsecond line", blurb="x" * 1000)]), 10)
    assert [i.url for i in d.items] == [None, None, "https://example.com/x", None]
    assert d.items[3].title == "d second line" and len(d.items[3].blurb) <= 240


def test_skip_empty_hides_empty_ok_sections_but_never_failures():
    with Store() as store:
        _insert(store.conn, "tasks.inbox_brief", 1.0, summary={"label_only": "x"})  # no numbers -> empty
        _insert(store.conn, "tasks.disasters", 1.0, status="error")
        master = mr.parse_config(VALID)[0]
        titles = [b.email.title for b in mr.collect(master, store.conn, NOW, _no_hooks)]
    assert titles == ["disasters", "job_tracker"]  # job_tracker: NOT RUN, link mode


# -- rendering ----------------------------------------------------------------

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'


def _section(n_items, title=HOSTILE, url="https://example.com/a?b=1&c=2"):
    items = [DigestItem(f"{title} {i}", url=url, blurb="blurb " * 30) for i in range(n_items)]
    return EmailSection(title, SectionStatus(OK), Digest([Tile(title, 1)], items, [title]))


def test_email_escapes_hostile_text_and_is_email_safe():
    email = render_email(HOSTILE, [_section(3)], intro=HOSTILE)
    assert "<script" not in email.html and "<img" not in email.html
    assert "&lt;script&gt;" in email.html
    for banned in ("<style", "<svg", "var(--", "fonts.googleapis", "<link"):
        assert banned not in email.html
    assert 'href="https://example.com/a?b=1&amp;c=2"' in email.html
    assert HOSTILE in email.text  # the plain-text part is text, not markup


def test_email_drops_non_http_links():
    s = _section(1, title="t")
    s.digest.items[0].url = "javascript:alert(1)"
    email = render_email("t", [s])
    assert "javascript:" not in email.html and "href=" not in email.html


def test_email_is_never_trimmed_with_many_items():
    """#119: the email is as long as its content -- every item of every
    section, far past Gmail's clip size (Gmail's "View entire message" is fine)."""
    sections = [_section(50, title=f"sec{j}") for j in range(30)]
    email = render_email("Big", sections)
    assert email.bytes > GMAIL_CLIP and email.clipped
    for j in range(30):
        for i in (0, 25, 49):
            assert f"sec{j} {i}" in email.html and f"sec{j} {i}" in email.text
    assert "more in" not in email.html


def test_modes_headline_and_link():
    d = Digest([Tile("n", 1)], [DigestItem("item-title")], ["a callout"])
    email = render_email("t", [EmailSection("h", SectionStatus(OK), d, mode="headline"),
                               EmailSection("l", SectionStatus(STALE, 30), d, mode="link")])
    assert "item-title" not in email.html
    assert "STALE (last good run 30h ago)" in email.html and "Full report in the attachment" in email.html


# -- run_master: dry run, send once a day --------------------------------------


def _seed(conn):
    _insert(conn, "tasks.inbox_brief", 1.0, summary={"threads": 4})
    _insert(conn, "tasks.disasters", 1.0, summary={"events": 2})


def test_dry_run_writes_preview_and_eml_without_sending(tmp_path):
    master = mr.parse_config(VALID)[0]
    with Store() as store:
        _seed(store.conn)
        result = mr.run_master(master, store.conn, send=False, now=NOW, hook_finder=_no_hooks)
        sends = store.conn.execute("SELECT COUNT(*) FROM sends").fetchone()[0]
    assert result["status"] == "dry_run" and sends == 0
    preview = Path(result["preview"])
    assert preview.parent == tmp_path / "output" / "2026-09-28" / "master-report-morning" and preview.exists()
    assert preview.name.startswith("master-report-morning-") and "<table" in preview.read_text(encoding="utf-8")
    eml = open(result["eml"], "rb").read()
    assert b"Morning Report" in eml and b"master-report-morning-" in eml  # attachment filename
    assert FAKE_PASSWORD.encode() not in eml


def test_real_send_goes_once_per_day(monkeypatch, capsys):
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeSMTP)
    monkeypatch.setattr(mailer_mod, "_today", lambda: NOW.astimezone().date().isoformat())
    master = mr.parse_config(VALID)[0]
    with Store() as store:
        _seed(store.conn)
        first = mr.run_master(master, store.conn, send=True, now=NOW, hook_finder=_no_hooks)
        second = mr.run_master(master, store.conn, send=True, now=NOW, hook_finder=_no_hooks)
    assert first["status"] == "sent" and second["status"] == "already_sent"
    assert len(_FakeSMTP.instances) == 1
    msg = _FakeSMTP.instances[0].sent[0]
    assert msg["To"] == "owner@example.test"  # to = [] -> LMSA_IMAP_USER
    assert "already sent today" in capsys.readouterr().out


def test_force_test_send_bypasses_the_guard_without_touching_the_real_slot(monkeypatch, capsys):
    """#105: "send test now" (force=True) must go out even when today's real
    slot is already claimed, and must never claim that real slot itself --
    a scheduled run later the same day still has to see today as un-sent."""
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeSMTP)
    today = NOW.astimezone().date().isoformat()
    monkeypatch.setattr(mailer_mod, "_today", lambda: today)
    master = mr.parse_config(VALID)[0]
    with Store() as store:
        _seed(store.conn)
        real = mr.run_master(master, store.conn, send=True, now=NOW, hook_finder=_no_hooks)
        assert real["status"] == "sent"
        first = mr.run_master(master, store.conn, send=True, force=True, now=NOW, hook_finder=_no_hooks)
        second = mr.run_master(master, store.conn, send=True, force=True, now=NOW, hook_finder=_no_hooks)
        rows = store.conn.execute("SELECT name, status FROM sends ORDER BY name").fetchall()
    assert first["status"] == "sent" and second["status"] == "sent"
    assert len(_FakeSMTP.instances) == 3  # the real send + two forced test sends
    assert rows == [("master-report-morning", "sent"), ("master-report-morning-test", "sent")]


def test_force_test_send_is_never_blocked_by_the_days_real_send(monkeypatch):
    """A test send must go out even when the real scheduled send already
    claimed today's slot -- the opposite direction of the guard above."""
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeSMTP)
    today = NOW.astimezone().date().isoformat()
    monkeypatch.setattr(mailer_mod, "_today", lambda: today)
    master = mr.parse_config(VALID)[0]
    with Store() as store:
        _seed(store.conn)
        store.conn.execute("INSERT INTO sends(name, date, status, created) VALUES (?, ?, 'sent', ?)",
                           (master.send_name, today, NOW.isoformat()))
        store.conn.commit()
        result = mr.run_master(master, store.conn, send=True, force=True, now=NOW, hook_finder=_no_hooks)
    assert result["status"] == "sent" and len(_FakeSMTP.instances) == 1


def test_mailer_skipped_is_logged_visibly(monkeypatch, capsys):
    """A row stuck at 'sending' (crash mid-send): the mailer skips; say so."""
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeSMTP)
    today = NOW.astimezone().date().isoformat()
    monkeypatch.setattr(mailer_mod, "_today", lambda: today)
    master = mr.parse_config(VALID)[0]
    with Store() as store:
        store.conn.execute("INSERT INTO sends(name, date, status, created) VALUES (?, ?, 'sending', ?)",
                           (master.send_name, today, NOW.isoformat()))
        store.conn.commit()
        result = mr.run_master(master, store.conn, send=True, now=NOW, hook_finder=_no_hooks)
    assert result["status"] == "skipped" and not _FakeSMTP.instances
    assert "mailer skipped the send" in capsys.readouterr().err


class _LM:
    def __init__(self, fail=False):
        self.fail = fail
        self.chats = 0

    def ensure_server(self):
        if self.fail:
            raise ConnectionError("down")

    def chat(self, prompt, system=None, max_tokens=None):
        self.chats += 1
        return "Good morning. <b>Quiet</b> night."


def test_llm_intro_is_used_and_fails_soft(monkeypatch):
    monkeypatch.setattr(mr, "chat_busy_reasons", lambda: [])
    bundled = [mr.Bundled(EmailSection("x", SectionStatus(OK), Digest([Tile("n", 1)])))]
    assert mr.try_intro(_LM(), bundled) == "Good morning. <b>Quiet</b> night."
    assert mr.try_intro(_LM(fail=True), bundled) is None


def test_llm_intro_skips_the_model_when_it_cant_be_had(monkeypatch, capsys):
    bundled = [mr.Bundled(EmailSection("x", SectionStatus(OK), Digest([Tile("n", 1)])))]
    lm = _LM()
    monkeypatch.setattr(mr, "chat_busy_reasons", lambda: ["gpu: free VRAM 900 MB < 9224 MB"])
    assert mr.try_intro(lm, bundled) is None and lm.chats == 0
    assert "intro skipped" in capsys.readouterr().out

    def broken():
        raise OSError("nvidia-smi missing")
    monkeypatch.setattr(mr, "chat_busy_reasons", broken)
    assert mr.try_intro(lm, bundled) is None and lm.chats == 0


def test_main_preview_records_a_run(monkeypatch, tmp_path):
    cfg = tmp_path / "master_report.toml"
    cfg.write_text(VALID.replace("send = false", "send = true"), encoding="utf-8")
    monkeypatch.setattr(mr, "CONFIG_PATH", cfg)
    monkeypatch.setattr(mr, "LOCAL_PATH", tmp_path / "none.toml")
    monkeypatch.setattr(mr, "find_hook", _no_hooks)
    mr.main(["--name", "morning", "--preview"])  # --preview wins over send = true: _RefusingSMTP not hit
    with Store() as store:
        row = store.conn.execute(
            "SELECT status, summary_json, report_path FROM runs WHERE task='tasks.master_report'").fetchone()
    assert row[0] == "ok" and json.loads(row[1])["dry_run"] == 1 and row[2]


def test_due_respects_send_at_and_enabled():
    m = mr.parse_config(VALID)[0]
    local_645 = datetime.now().astimezone().replace(hour=6, minute=45, second=0, microsecond=0)
    assert mr.due(m, local_645) and not mr.due(m, local_645 - timedelta(minutes=1))
    m.enabled = False
    assert not mr.due(m, local_645)


# -- schedule -----------------------------------------------------------------


def test_schedule_sends_at_0645_after_inbox_brief():
    specs = {s.name: s for s in load_schedule().tasks}
    assert str(specs["master_report"].cadence) == "daily 06:45"
    # No model requirement and no input-idle requirement: see the dispatch tests below.
    assert specs["master_report"].model is None and specs["master_report"].idle_minutes == 0
    for task in ("inbox_brief", "disasters", "job_tracker"):
        assert specs[task].cadence.at < specs["master_report"].cadence.at


# -- dispatcher: the 06:45 email isn't held back (#103 review) ------------------

from zoneinfo import ZoneInfo  # noqa: E402

from tasks import dispatch as dp  # noqa: E402

NY = ZoneInfo("America/New_York")


class _InputActiveGate:
    """The user is at the keyboard: busy whenever input-idle is required."""

    def __init__(self):
        self.calls = []

    def __call__(self, need_vram_mb=0, idle_minutes=0):
        self.calls.append(idle_minutes)
        return ["input: active 1 min ago"] if idle_minutes > 0 else []


class _Models:
    def __init__(self, unreachable=False):
        self.unreachable = unreachable
        self.ours = []

    def _lms(self):
        if self.unreachable:
            raise ConnectionError("lms: server not reachable")
        return set()

    def is_resident(self, m):
        return bool(self._lms())

    def ensure(self, m, size):
        self._lms()

    def snapshot(self):
        return self._lms()

    def adopt_new(self, before):
        return []

    def release_all(self):
        return []


def _dispatch(conn, tasks, now, gate, models):
    ran = []

    def run_task(spec, env=None, timeout_s=None):
        ran.append(spec.name)
        return "ran", None

    sched = load_schedule()
    sched.tasks = tasks
    res = dp.dispatch(sched, conn, gate_fn=gate, models=models, model_names={"chat": "c", "vision": "v"},
                      run_task=run_task, now=now, clock=lambda: now, tz=NY, sleep=lambda s: None)
    return res, ran


def _specs(*names):
    by_name = {s.name: s for s in load_schedule().tasks}
    return [by_name[n] for n in names]


def test_dispatch_runs_master_at_0700_while_the_user_is_active():
    now = datetime(2026, 9, 28, 7, 0, tzinfo=NY)  # first tick after 06:45, outside overnight
    with Store() as store:
        dp.ensure_table(store.conn)
        res, ran = _dispatch(store.conn, _specs("master_report", "markets"), now, _InputActiveGate(),
                             _Models())
    assert ran == ["master_report"]
    assert [t for t, _ in res.deferred] == ["markets"]  # an ordinary task still waits for idle


def test_dispatch_runs_master_when_lm_studio_is_unreachable():
    now = datetime(2026, 9, 28, 6, 50, tzinfo=NY)
    with Store() as store:
        dp.ensure_table(store.conn)
        res, ran = _dispatch(store.conn, _specs("master_report", "inbox_brief"), now, _InputActiveGate(),
                             _Models(unreachable=True))
    assert ran == ["master_report"]
    assert res.deferred and res.deferred[0][0] == "inbox_brief" and "LM Studio" in res.deferred[0][1]


def test_watchdog_after_an_earlier_task_does_not_hold_master_back():
    """Overnight catch-up tasks due before 06:45 share master_report's group.
    If the user becomes active mid-batch, the between-task watchdog must not
    defer master_report along with them."""
    now = datetime(2026, 9, 28, 7, 0, tzinfo=NY)
    ran = []

    def gate(need_vram_mb=0, idle_minutes=0):
        return ["input: active 1 min ago"] if ran and idle_minutes > 0 else []

    def run_task(spec, env=None, timeout_s=None):
        ran.append(spec.name)
        return "ran", None

    with Store() as store:
        dp.ensure_table(store.conn)
        sched = load_schedule()
        sched.tasks = _specs("disasters", "risk_map", "master_report")
        res = dp.dispatch(sched, store.conn, gate_fn=gate, models=_Models(),
                          model_names={"chat": "c", "vision": "v"}, run_task=run_task, now=now,
                          clock=lambda: now, tz=NY, sleep=lambda s: None)
    assert ran[0] == "master_report"
    assert "master_report" not in [t for t, _ in res.deferred]


def test_idle_minutes_is_only_for_no_model_tasks(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    cfg = tmp_path / "s.toml"
    cfg.write_text('[tasks.a]\ncadence = "daily 01:00"\nidle_minutes = 0\n'
                   '[tasks.b]\ncadence = "daily 01:00"\nmodel = "chat"\nidle_minutes = 0\n', encoding="utf-8")
    sched = dp.load_schedule(cfg, tasks_dir=tmp_path)
    assert [(t.name, t.idle_minutes) for t in sched.tasks] == [("a", 0.0)]
    assert "idle_minutes is only allowed" in sched.problems[0]


# -- malformed hook digests (#103 review) ---------------------------------------


@pytest.mark.parametrize("bad", [
    Digest(items=["x"]),
    Digest(tiles=[{"label": "n", "value": 1}]),
    Digest(tiles=None),
    Digest(tiles=5, items="abc", callouts=[1, None]),
])
def test_malformed_hook_digest_is_dropped_with_a_note_not_a_crash(bad):
    run = RunRow(1, "tasks.inbox_brief", _ts(1), "ok", summary_json='{"threads": 9}')
    b = mr.build_section(mr.SectionConfig(task="inbox_brief"), run, SectionStatus(OK), lambda t: (lambda r, n: bad))
    assert "malformed" in b.email.note
    render_email("t", [b.email])  # renders fine


def test_hook_digest_normalize_failure_falls_back(monkeypatch):
    run = RunRow(1, "tasks.inbox_brief", _ts(1), "ok", summary_json='{"threads": 9}')

    class Weird:
        def __str__(self):
            raise RuntimeError("no str")
    bad = Digest(items=[DigestItem(Weird())])
    b = mr.build_section(mr.SectionConfig(task="inbox_brief"), run, SectionStatus(OK), lambda t: (lambda r, n: bad))
    assert b.email.digest.tiles[0].label == "threads" and "RuntimeError" in b.email.note


def test_normalize_caps_tile_label_and_value():
    d, _, _ = normalize(Digest([Tile("L" * 500, "X" * 500000), Tile("big", 1e308), Tile("n", 12345)]), 5)
    assert len(d.tiles[0].label) <= 60 and len(d.tiles[0].value) <= 24
    assert d.tiles[1].value == "1e+308" and d.tiles[2].value == 12345


# -- no trimming (#119; replaces the #103 size-budget guarantees) ---------------


def _fat_section(i):
    return EmailSection(f"s{i}", SectionStatus(OK), Digest(
        [Tile("<" * 60, "<" * 24) for _ in range(4)], [], [f"callout {i} " + "<" * 240 for _ in range(5)]))


def test_huge_tile_is_shown_whole():
    email = render_email("t", [EmailSection("a", SectionStatus(OK), Digest([Tile("a", "X" * 500000)]))])
    assert "X" * 500000 in email.html and "X" * 500000 in email.text


def test_many_callouts_tiles_and_the_intro_all_survive():
    sections = [_fat_section(i) for i in range(30)]
    email = render_email("t", sections, intro="intro " + "<" * 1200)
    assert email.clipped
    assert "intro " + "&lt;" * 1200 in email.html and "intro " + "<" * 1200 in email.text
    for i in range(30):
        assert email.html.count(f"callout {i} ") == 5 and email.text.count(f"callout {i} ") == 5


def test_every_section_is_kept_however_many():
    sections = [EmailSection(f"sec-{i}", SectionStatus(OK), Digest()) for i in range(400)]
    email = render_email("T" * 150, sections, subtitle="S" * 150, intro="i" * 1000)
    lines = email.text.splitlines()
    assert all(f"sec-{i}<" in email.html and f"## sec-{i}" in lines for i in range(400))
    assert "i" * 1000 in email.html


# -- config: title, send_at vs schedule (#103 review) ---------------------------


@pytest.mark.parametrize("title", ["Hi\nBcc: x@y.z", "a\rb", "tab\there", "sep\u2028x", "x" * 121, "  "])
def test_title_rejects_control_chars_and_length(title):
    import tomllib

    data = tomllib.loads(VALID)
    data["master"][0]["title"] = title
    with pytest.raises(mr.ConfigError, match="title must be"):
        mr.parse_config(data)


def test_name_with_newline_is_rejected():
    import tomllib

    data = tomllib.loads(VALID)
    data["master"][0]["name"] = "morning\n"
    with pytest.raises(mr.ConfigError, match="name must be"):
        mr.parse_config(data)


def _schedule(tmp_path, entry):
    p = tmp_path / "schedule.toml"
    p.write_text("[dispatch]\n" + entry, encoding="utf-8")
    return p


@pytest.mark.parametrize("entry,send_at,ok", [
    ('[tasks.master_report]\ncadence = "daily 06:45"\n', "06:45", True),
    ('[tasks.master_report]\ncadence = "daily 06:45"\n', "06:30", True),
    ('[tasks.master_report]\ncadence = "daily 06:45"\n', "07:30", False),
    ('[tasks.master_report]\ncadence = "every 1 hours"\n', "07:30", True),
    ('[tasks.master_report]\ncadence = "every 6 hours"\n', "19:00", False),
    ('[tasks.markets]\ncadence = "daily 01:00"\n', "23:00", True),  # not scheduled: manual only
])
def test_send_at_must_not_be_after_the_scheduled_run(tmp_path, entry, send_at, ok):
    cfg = VALID.replace('send_at = "06:45"', f'send_at = "{send_at}"')
    path = _schedule(tmp_path, entry)
    if ok:
        assert mr.parse_config(cfg, schedule_path=path)[0].send_at == send_at
    else:
        with pytest.raises(mr.ConfigError, match="would never be sent"):
            mr.parse_config(cfg, schedule_path=path)
