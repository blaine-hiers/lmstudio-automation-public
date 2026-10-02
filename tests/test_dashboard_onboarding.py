"""New-tool onboarding (#38): options derived from argparse by ast, SETUP
needs on the Setup page, NEW tiles with Add to schedule / mark manual-only.

Hermetic: a tmp root with fake task modules, an in-memory keyring
(conftest.py), fake `lms` / PowerShell; nothing is imported from the fake
tasks and nothing touches the real config, .env or Task Scheduler.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa import task_meta
from lmsa.search import SearXNG
from tasks import dashboard as db
from tasks import dashboard_setup as ds
from tasks.dispatch import load_schedule

from test_dashboard import call, jcall, poll_until_done
from test_dashboard_setup import ENV_EXAMPLE, FakeLms, FakePowerShell, FakeSearXNGFetcher

ROOT = Path(__file__).resolve().parents[1]

SCHEDULE = """\
# schedule comment -- kept byte for byte
[dispatch]
task_timeout_minutes = 60   # trailing comment

[tasks.oldjob]            # an existing entry
cadence = "daily 01:00"
model = "none"
"""

NEWTOOL = '''\
"""Brand-new tool <b>bold</b> for the onboarding tests."""
import argparse
import os
import sys
from pathlib import Path

# Importing this module would leave a marker behind: the dashboard must not.
Path(os.environ.get("ONBOARD_MARKER", "unused-marker")).write_text("imported", encoding="utf-8")

SETUP = {
    "secrets": ["imap", "nope"],
    "env": ["LMSA_IMAP_USER", "NOT_IN_EXAMPLE"],
    "config": ["config/newtool.toml", "//evil-host/share/x.toml", "config/../.env"],
    "commands": ["definitely-not-a-command-38"],
    "model": "chat",
    "surprise": 1,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5, help="how many <i>items</i>")
    p.add_argument("--push", action="store_true", help="[writes] push the results somewhere")
    p.add_argument("--send", metavar="TO")
    p.add_argument("--label", help=os.environ.get("X", "dynamic"))
    print("newtool argv", sys.argv[1:], flush=True)


if __name__ == "__main__":
    main()
'''

PLAIN = '"""Plain helper tool."""\nprint("plain", flush=True)\n'


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "root"
    (r / "tasks").mkdir(parents=True)
    (r / "config").mkdir()
    (r / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (r / "tasks" / "oldjob.py").write_text('"""Old job."""\n', encoding="utf-8")
    (r / "tasks" / "newtool.py").write_text(NEWTOOL, encoding="utf-8")
    (r / "tasks" / "plain.py").write_text(PLAIN, encoding="utf-8")
    (r / "tasks" / "dispatch.py").write_text('"""Fake dispatcher."""\n', encoding="utf-8")
    (r / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    (r / "config" / "schedule.toml").write_bytes(SCHEDULE.replace("\n", "\r\n").encode())
    for key in (line.split("=")[0] for line in ENV_EXAMPLE.splitlines() if "=" in line):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ONBOARD_MARKER", str(tmp_path / "imported.txt"))
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "data" / "lmsa.sqlite"))
    return r


@pytest.fixture
def dash(root, tmp_path):
    d = db.Dashboard(
        root, output_dir=tmp_path / "output", db_path=tmp_path / "data" / "lmsa.sqlite",
        schedule_path=root / "config" / "schedule.toml",
        probe_server=lambda: True, loaded_models=lambda: [], busy=lambda: [], token="test-token",
    )
    d.setup = ds.Setup(d, lms=FakeLms(), searxng=SearXNG("http://fake-searxng.test:8888",
                       fetcher=FakeSearXNGFetcher()), powershell=FakePowerShell(), send_slack=lambda t: True,
                       version_of=lambda name: "99.0")
    yield d
    for run in list(d.runs.values()):
        if run.live:
            d.stop(run.id)
            run.done.wait(15)


@pytest.fixture
def server(dash):
    srv = db.DashboardServer(dash, 0)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def tasks_by_name(dash):
    return {t.name: t for t in dash.catalog()}


def add(srv, task, fields, confirm=True):
    return jcall(srv, "POST", "/api/tasks/schedule", {"task": task, "fields": fields, "confirm": confirm})


DAILY = {"kind": "daily", "time": "04:15", "deadline_hours": "12", "model": "chat", "timeout_minutes": ""}


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------

def test_new_tool_is_flagged_with_derived_options_and_setup_items(server, dash, tmp_path):
    t = tasks_by_name(dash)["newtool"]
    assert t.new and not tasks_by_name(dash)["oldjob"].new
    assert t.derived and t.model == "chat"  # the SETUP model role, for the VRAM warning
    fields = {f.flag: f for f in t.fields}
    assert fields["--limit"].kind == "int" and fields["--limit"].label == "how many <i>items</i> (default 5)"
    assert fields["--push"].kind == "bool" and fields["--push"].write and fields["--push"].label.startswith("push")
    assert fields["--send"].kind == "text" and fields["--send"].write and fields["--send"].label == "TO"
    assert not fields["--limit"].write
    assert any("--label: help=" in w for w in t.warnings)  # non-literal help: reported, not evaluated

    status, _, raw = call(server, "GET", "/")
    page = raw.decode()
    assert status == 200
    # #100: the new-tools banner jumps to the Tasks tab and reveals the tile.
    assert '<span class="tag t-hot">new</span>' in page and 'data-goto="task-newtool"' in page
    assert 'data-onboard="schedule"' in page and 'data-onboard="manual"' in page
    assert "<b>bold</b>" not in page and "&lt;b&gt;bold&lt;/b&gt;" in page
    assert "<i>items</i>" not in page
    assert 'data-item="tool-newtool"' in page
    assert not (tmp_path / "imported.txt").exists()  # read with ast, never imported

    status, body = jcall(server, "GET", "/api/setup")
    item = {i["id"]: i for i in body["items"]}["tool-newtool"]
    assert item["state"] == "missing"
    for needle in ("secret imap not set", "LMSA_IMAP_USER not set", "config/newtool.toml missing",
                   "definitely-not-a-command-38 isn't on PATH", "unknown secret 'nope'",
                   "unknown env var 'NOT_IN_EXAMPLE'", "isn't a relative path under config/",
                   "SETUP key 'surprise' isn't known"):
        assert needle in item["reason"], (needle, item["reason"])


def test_derived_write_flag_needs_a_confirm(server, dash):
    for fields, extra in (({"--push": True}, ""), ({"--send": "x"}, ""), ({}, "--pu"), ({}, "--send=me"),
                          ({}, "--se me")):
        status, body = jcall(server, "POST", "/api/run", {"task": "newtool", "fields": fields, "extra": extra})
        assert status == 428, (fields, extra, body)
    status, body = jcall(server, "POST", "/api/run", {"task": "newtool", "fields": {"--limit": "3"}})
    assert status == 201, body
    _, text = poll_until_done(server, body["id"])
    assert "newtool argv ['--limit', '3']" in text
    status, body = jcall(server, "POST", "/api/run",
                         {"task": "newtool", "fields": {"--push": True}, "confirm": True})
    assert status == 201, body
    _, text = poll_until_done(server, body["id"])
    assert "'--push'" in text


def test_add_to_schedule_appends_a_valid_table_the_dispatcher_loads(server, dash, root, tmp_path):
    path = root / "config" / "schedule.toml"
    before = path.read_bytes()
    fields = {**DAILY, "timeout_minutes": "30"}
    status, body = add(server, "newtool", fields, confirm=False)
    assert status == 428 and "schedule.toml" in body["error"]
    assert path.read_bytes() == before
    status, body = add(server, "newtool", fields)
    assert status == 200, body
    after = path.read_bytes()
    assert after.startswith(before)  # comments and every other byte kept
    added = after[len(before):].decode()
    assert "\r\n" in added and "\n" not in added.replace("\r\n", "")  # the file's CRLF style
    assert '[tasks.newtool]' in added and 'cadence = "daily 04:15"' in added and "timeout_minutes = 30" in added
    spec = {t.name: t for t in load_schedule(path, root / "tasks").tasks}["newtool"]
    assert str(spec.cadence) == "daily 04:15" and spec.model == "chat" and spec.timeout.total_seconds() == 1800
    backups = list((tmp_path / "data" / "dashboard" / "config-backups").glob("schedule-*.toml"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    t = tasks_by_name(dash)["newtool"]
    assert not t.new and t.cadence == "daily 04:15"
    status, body = add(server, "newtool", fields)
    assert status == 409 and "already" in body["error"]
    assert path.read_bytes() == after


@pytest.mark.parametrize("fields", [
    {"kind": "weekly", "day": "sun", "time": "02:30", "deadline_hours": "48", "model": "none"},
    {"kind": "every", "hours": "6", "deadline_hours": "0.5", "model": "vision", "timeout_minutes": "240"},
])
def test_every_cadence_kind_round_trips(server, root, fields):
    status, body = add(server, "plain", fields)
    assert status == 200, body
    sched = load_schedule(root / "config" / "schedule.toml", root / "tasks")
    assert not sched.problems and "plain" in {t.name for t in sched.tasks}


@pytest.mark.parametrize("task,fields,code", [
    ("newtool", {**DAILY, "kind": "hourly"}, 400),
    ("newtool", {**DAILY, "time": "25:00"}, 400),
    ("newtool", {**DAILY, "time": "4:15"}, 400),
    ("newtool", {**DAILY, "deadline_hours": "0.1"}, 400),
    ("newtool", {**DAILY, "deadline_hours": 12}, 400),
    ("newtool", {**DAILY, "model": "embed"}, 400),
    ("newtool", {**DAILY, "timeout_minutes": "5"}, 400),
    ("newtool", {**DAILY, "kind": "weekly", "day": "funday"}, 400),
    ("newtool", {**DAILY, "kind": "every", "hours": "5"}, 400),
    ("newtool", {**DAILY, "evil": "x"}, 400),
    ("newtool", "not a dict", 400),
    ("dispatch", DAILY, 400),
    ("../newtool", DAILY, 400),
    ("nope", DAILY, 400),
    ("oldjob", DAILY, 409),
])
def test_add_to_schedule_refuses_bad_input_and_changes_nothing(server, root, task, fields, code):
    before = (root / "config" / "schedule.toml").read_bytes()
    status, body = add(server, task, fields)
    assert status == code, body
    assert (root / "config" / "schedule.toml").read_bytes() == before


def test_add_validates_the_whole_file_before_replacing_it(server, root):
    path = root / "config" / "schedule.toml"
    path.write_bytes(path.read_bytes() + b'[tasks.plain]\r\ncadence = "hourly"\r\n')  # a bad entry already there
    before = path.read_bytes()
    status, body = add(server, "newtool", DAILY)
    assert status == 400 and "tasks.plain" in body["error"], body  # load_schedule's verdict, before any replace
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_windows_sharing_error_is_a_409_and_the_file_is_kept(server, root, monkeypatch):
    def locked(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(ds.os, "replace", locked)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    before = (root / "config" / "schedule.toml").read_bytes()
    status, body = add(server, "newtool", DAILY)
    assert status == 409 and "in another program" in body["error"]
    assert (root / "config" / "schedule.toml").read_bytes() == before
    assert not [p for p in (root / "config").iterdir() if p.name.endswith(".tmp")]


def test_concurrent_adds_of_two_tools_both_land(dash, root):
    errors = []

    def go(name):
        try:
            dash.setup.add_to_schedule(name, DAILY, confirm=True)
        except Exception as exc:  # noqa: BLE001 -- recorded for the assert
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(n,)) for n in ("newtool", "plain")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert not errors
    names = {t.name for t in load_schedule(root / "config" / "schedule.toml", root / "tasks").tasks}
    assert {"newtool", "plain", "oldjob"} <= names


def test_mark_manual_only_creates_then_appends_dashboard_toml(server, dash, root):
    path = root / "config" / "dashboard.toml"
    assert not path.exists()
    status, body = jcall(server, "POST", "/api/tasks/manual", {"task": "plain"})
    assert status == 428 and not path.exists()
    status, body = jcall(server, "POST", "/api/tasks/manual", {"task": "plain", "confirm": True})
    assert status == 200, body
    t = tasks_by_name(dash)["plain"]
    assert not t.new and t.group == "Diagnostics & manual"
    first = path.read_bytes()
    status, body = jcall(server, "POST", "/api/tasks/manual", {"task": "newtool", "confirm": True})
    assert status == 200, body
    assert path.read_bytes().startswith(first)
    assert db.load_manual(path) == {"plain", "newtool"}
    # neither is NEW any more; a scheduled or already-manual task is refused
    for name in ("plain", "oldjob"):
        status, body = jcall(server, "POST", "/api/tasks/manual", {"task": name, "confirm": True})
        assert status == 409, body
    status, _, raw = call(server, "GET", "/")
    assert 'data-onboard=' not in raw.decode()


def test_dashboard_toml_is_in_the_config_editor_with_its_own_validator(server, dash, root):
    status, body = jcall(server, "POST", "/api/setup/action",
                         {"item": "config", "action": "load", "fields": {"file": "dashboard"}})
    assert status == 200 and body["fill"]["base"] == "new"
    for text, needle in (('[manual.nope]\n', "no module tasks/nope.py"),
                         ('colour = "red"\n', "unknown key"),
                         ('[manual.plain]\nwhy = "x"\n', "unknown key"),
                         ('manual = ["plain"]\n', "table"),
                         ('[manual.plain\n', "TOMLDecodeError")):
        status, body = jcall(server, "POST", "/api/setup/action", {
            "item": "config", "action": "save", "fields": {"file": "dashboard", "text": text, "base": "new"},
            "confirm": True})
        assert status == 400 and needle in body["error"], (text, body)
    assert not (root / "config" / "dashboard.toml").exists()
    status, body = jcall(server, "POST", "/api/setup/action", {
        "item": "config", "action": "save",
        "fields": {"file": "dashboard", "text": '[manual.plain]\nnote = "helper"\n', "base": "new"}, "confirm": True})
    assert status == 200, body
    assert not tasks_by_name(dash)["plain"].new


# ---------------------------------------------------------------------------
# the tool's Setup item
# ---------------------------------------------------------------------------

def test_tool_item_saves_env_and_secret_without_echoing_it(server, dash, root, fake_keyring):
    secret = "S3NT1NEL-onboard-pw-38"
    status, body = jcall(server, "POST", "/api/setup/action", {
        "item": "tool-newtool", "action": "save",
        "fields": {"secret-imap": secret, "LMSA_IMAP_USER": "someone@example.com"}})
    assert status == 200, body
    assert secret not in json.dumps(body)
    assert fake_keyring.store[("lmsa", "imap")] == secret
    (root / "config" / "newtool.toml").write_text("", encoding="utf-8")
    item = {i["id"]: i for i in jcall(server, "GET", "/api/setup")[1]["items"]}["tool-newtool"]
    assert secret not in json.dumps(item) and "someone@example.com" not in item["reason"]
    assert item["state"] == "missing" and "secret imap not set" not in item["reason"]  # still missing: the command
    assert "config/newtool.toml" not in item["reason"] and "LMSA_IMAP_USER not set" not in item["reason"]
    # only declared inputs are accepted: an undeclared env key is refused
    status, _ = jcall(server, "POST", "/api/setup/action", {
        "item": "tool-newtool", "action": "save", "fields": {"LMSA_CHAT_MODEL": "x"}})
    assert status == 400


def test_tool_item_ok_when_everything_is_there(dash, root, monkeypatch, fake_keyring):
    (root / "tasks" / "ready.py").write_text(
        '"""Ready."""\nSETUP = {"secrets": ["imap"], "env": ["LMSA_IMAP_USER"], '
        '"config": ["config/ready.toml"], "commands": ["python"], "model": "embed"}\n', encoding="utf-8")
    (root / "config" / "ready.toml").write_text("", encoding="utf-8")
    fake_keyring.store[("lmsa", "imap")] = "a-long-app-password"
    (root / ".env").write_text("LMSA_IMAP_USER=someone@example.com\n", encoding="utf-8")
    monkeypatch.setattr(ds.shutil, "which", lambda cmd: f"C:/bin/{cmd}.exe")
    item = dash.setup.item("tool-ready")
    st = item.status(dash.setup)
    assert st.state == "ok", st.reason
    assert "a-long-app-password" not in st.reason and "uses the embed model" in st.reason


@pytest.mark.parametrize("entry", [
    "//evil-host/share/x.toml", r"\\evil-host\share\x.toml", "C:/Windows/win.ini", "c:config/x.toml",
    "/etc/passwd", "config/../.env", "config/./x.toml", "config", "config/", "labels.toml",
    "config\\..\\..\\x", "config/sub//x.toml", "config/x\x00.toml",
])
def test_config_paths_are_confined_lexically_before_any_filesystem_call(entry):
    assert ds.config_rel(entry) is None


def test_confined_config_paths_are_accepted():
    assert ds.config_rel("config/labels.toml") == "labels.toml"
    assert ds.config_rel(r"config\sub\x.toml") == "sub/x.toml"


def test_unc_config_entry_never_reaches_the_filesystem(dash, root, monkeypatch):
    (root / "tasks" / "sneaky.py").write_text(
        '"""Sneaky."""\nSETUP = {"config": ["//evil-host/share/x.toml", "config\\\\..\\\\..\\\\y"]}\n',
        encoding="utf-8")
    seen = []
    real = Path.is_file

    def spy(self):
        seen.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "is_file", spy)
    st = dash.setup.item("tool-sneaky").status(dash.setup)
    assert st.state == "warning" and "isn't a relative path under config/" in st.reason
    assert seen == []


# ---------------------------------------------------------------------------
# lmsa.task_meta: ast only, literals only, bounded
# ---------------------------------------------------------------------------

def test_meta_reads_positionals_and_the_writes_marker():
    meta = task_meta.parse_source(
        'import argparse\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("query", help="what to look for")\n'
        'p.add_argument("maybe", nargs="?")\n'
        'p.add_argument("-f", "--force", action="store_true")\n'
        'p.add_argument("--post", help="  [WRITES] posts it")\n'
        'p.add_argument("--quiet", action="store_false", help="less output")\n'
        'p.add_argument("--hidden", help=argparse.SUPPRESS)\n'
        'p.add_argument("--version", action="version", version="1")\n'
        'p.add_argument(*names)\n'
        'p.add_argument("--kw", **opts)\n'
        'p.add_argument("--act", action=Custom)\n'
    )
    args = {a.flag: a for a in meta.args}
    assert list(args) == ["query", "maybe", "--force", "--post", "--quiet"]
    assert args["query"].kind == "positional" and args["query"].required
    assert not args["maybe"].required
    assert args["--force"].write and args["--force"].options == ("-f", "--force")
    assert args["--post"].write and args["--post"].label == "posts it"
    assert not args["--quiet"].write and args["--quiet"].kind == "bool"
    assert len(meta.arg_warnings) == 3  # *names, **opts, action=Custom


def test_short_alias_of_a_write_flag_needs_a_confirm_even_bundled():
    meta = task_meta.parse_source('import argparse\np = argparse.ArgumentParser()\n'
                                  'p.add_argument("-f", "--force", action="store_true")\n'
                                  'p.add_argument("-v", action="store_true")\n')
    t = db.TaskInfo("x", "", "Other", fields=db.derived_fields(meta.args),
                    write_flags=tuple(o for a in meta.args if a.write for o in a.options))
    for extra in ("-f", "-vf", "--fo", "--force"):
        with pytest.raises(db.NeedsConfirm):
            db.build_args(t, {}, extra)
    assert db.build_args(t, {}, "-v") == ["-v"]


def test_meta_setup_takes_literals_only():
    meta = task_meta.parse_source(
        'import os\nSETUP = {"env": [os.getenv("X")], "secrets": "imap", "config": [1], '
        '"commands": ["gh"], **extra}\n')
    assert meta.setup == {"secrets": ["imap"], "commands": ["gh"]}
    assert len(meta.setup_warnings) == 3
    assert task_meta.parse_source("SETUP = make_setup()\n").setup == {}
    assert task_meta.parse_source('"""no setup"""\n').setup is None


def test_meta_survives_syntax_errors_and_oversized_modules(tmp_path, root, dash):
    (root / "tasks" / "broken.py").write_text('"""Broken."""\ndef main(:\n', encoding="utf-8")
    (root / "tasks" / "huge.py").write_text('"""Huge."""\n' + "#" * (task_meta.MAX_SOURCE_BYTES + 10),
                                            encoding="utf-8")
    tasks = tasks_by_name(dash)
    assert tasks["broken"].warnings and "SyntaxError" in tasks["broken"].warnings[0]
    assert tasks["huge"].warnings and "KiB" in tasks["huge"].warnings[0]
    assert tasks["broken"].new and not tasks["broken"].fields
    page = db.render_page(dash, "n", 1)
    assert 'id="task-broken"' in page and "tile-warnings" in page


def test_meta_is_cached_and_refreshed_when_the_file_changes(tmp_path):
    p = tmp_path / "t.py"
    p.write_text('"""One."""\n', encoding="utf-8")
    assert task_meta.read_meta(p).doc == "One."
    assert task_meta.read_meta(p) is task_meta.read_meta(p)
    p.write_text('"""Two, longer."""\n', encoding="utf-8")
    assert task_meta.read_meta(p).doc == "Two, longer."


# ---------------------------------------------------------------------------
# the real repo
# ---------------------------------------------------------------------------

def test_real_catalog_has_no_new_tools_and_manual_set_moved_to_config():
    tasks = {t.name: t for t in db.discover_tasks(ROOT / "tasks")}
    assert not [t.name for t in tasks.values() if t.new]
    assert db.load_manual(ROOT / "config" / "dashboard.toml") == {
        "ask", "doc_search", "shootout", "gmail_probe", "style_preview", "index", "healthcheck",
        "inbox_actions", "go_deep"}
    assert not [t.name for t in tasks.values() if t.warnings]


def test_curated_fields_still_win_over_derived_ones():
    tasks = {t.name: t for t in db.discover_tasks(ROOT / "tasks")}
    for name, fields in db.FIELDS.items():
        assert tasks[name].fields == fields and not tasks[name].derived


def test_real_setup_blocks_are_valid():
    env_keys = ds.Setup.env_keys(SimpleNamespace(env_example=ROOT / ".env.example"))
    tasks = {t.name: t for t in db.discover_tasks(ROOT / "tasks")}
    declared = {n: t.setup for n, t in tasks.items() if t.setup is not None}
    expected = {"labeler", "price_watch", "front_page", "doc_index", "doc_search", "github_digest"}
    gmail = {n for n, t in tasks.items() if t.group == "Gmail" or n == "gmail_probe"}
    assert expected | gmail <= set(declared)
    assert declared["github_digest"]["commands"] == ["gh"]
    for name, setup in declared.items():
        assert not tasks[name].setup_warnings, name
        assert set(setup.get("secrets", [])) <= set(ds.lmsa_secrets._ENV_VARS), name
        assert set(setup.get("env", [])) <= env_keys, name
        for entry in setup.get("config", []):
            rel = ds.config_rel(entry)
            assert rel is not None and (ROOT / "config" / rel).is_file(), (name, entry)
        assert setup.get("model", "chat") in ds.MODEL_ROLES_SETUP, name
    for name in gmail:
        assert declared[name]["secrets"] == ["imap"] and "LMSA_IMAP_USER" in declared[name]["env"]


@pytest.mark.slow
def test_python_dash_m_dashboard_answers_setup_errors_with_json(tmp_path):
    """Under `python -m tasks.dashboard` the page's handler and the Setup
    panel must share one tasks.dashboard module: a NeedsConfirm / BadRequest
    raised by dashboard_setup is then a 428 / 400, not a dropped connection."""
    import http.client
    import os
    import socket
    import subprocess
    import sys
    import time

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "LMSA_DB": str(tmp_path / "lmsa.sqlite"), "LMSA_OUTPUT_DIR": str(tmp_path / "out"),
           "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    proc = subprocess.Popen([sys.executable, "-m", "tasks.dashboard", "--no-browser", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert "Afterhours control on" in proc.stdout.readline().decode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        conn.request("GET", "/")
        page = conn.getresponse().read().decode()
        token = page.split('name="lmsa-token" content="', 1)[1].split('"', 1)[0]
        results = []
        for path, body in (("/api/tasks/schedule", {"task": "no_such_task", "fields": {}}),
                           ("/api/setup/action", {"item": "lmstudio", "action": "unload", "fields": {"model": "x"}})):
            conn.request("POST", path, body=json.dumps(body),
                         headers={"Content-Type": "application/json", db.TOKEN_HEADER: token})
            resp = conn.getresponse()
            results.append((resp.status, json.loads(resp.read())))
        assert results[0][0] == 400 and "unknown task" in results[0][1]["error"]
        assert results[1][0] == 428
    finally:
        proc.kill()
        proc.wait(10)


# ---------------------------------------------------------------------------
# PR #43 review: write flags the tile skips still need a confirm
# ---------------------------------------------------------------------------

def _task_from(source: str) -> db.TaskInfo:
    meta = task_meta.parse_source(source)
    return db.TaskInfo("x", meta.doc, "Other", fields=db.derived_fields(meta.args),
                       write_flags=meta.write_options, warnings=tuple(meta.warnings))


@pytest.mark.parametrize("line,extras", [
    ('p.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)', ["--apply", "--app", "--apply=1"]),
    ('p.add_argument("--send", action="append_const", const=1)', ["--send", "--se"]),
    ('p.add_argument(*names, "--force")', ["--force", "--fo"]),
    ('p.add_argument("--delete", **kw)', ["--delete", "--del=1"]),
    ('p.add_argument("--wipe", action=Custom, help="[writes] wipes it")', ["--wipe", "--wi"]),
    ('p.add_argument("-purge", action="store_true", help="[writes] purges")', ["-purge", "-pur", "-p", "-purge=1"]),
    ('p.add_argument("-z", "--zap", help="[writes] zaps", **kw)', ["-z", "-vz", "--zap", "--za"]),
])
def test_write_flags_the_tile_skips_still_need_a_confirm(line, extras):
    t = _task_from(f'import argparse\np = argparse.ArgumentParser()\n{line}\n'
                   'p.add_argument("--limit", type=int)\np.add_argument("-v", action="store_true")\n')
    for extra in extras:
        with pytest.raises(db.NeedsConfirm):
            db.build_args(t, {}, extra)
        assert db.build_args(t, {}, extra, confirm=True)
    assert db.build_args(t, {}, "--limit 3 -v") == ["--limit", "3", "-v"]


def test_single_dash_long_write_flag_is_matched_like_argparse():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-purge", action="store_true")
    for token in ("-purge", "-pur", "-p"):
        assert p.parse_args([token]).purge  # what argparse accepts ...
        assert db._names_write_flag(token, ["-purge"])  # ... is what we confirm
    assert not db._names_write_flag("-q", ["-purge"])


def test_standard_write_names_need_a_confirm_even_when_the_module_cant_be_parsed():
    t = _task_from('"""Broken."""\ndef main(:\n    p.add_argument("--apply")\n')
    assert t.warnings and not t.fields and not t.write_flags
    for extra in ("--apply", "--app", "--undo=3", "--retry x", "--send", "--delete", "--force", "--forc"):
        with pytest.raises(db.NeedsConfirm):
            db.build_args(t, {}, extra)
    assert db.build_args(t, {}, "--limit 2") == ["--limit", "2"]


def test_a_module_with_a_utf8_bom_is_read(dash, root):
    (root / "tasks" / "bomtool.py").write_bytes(
        b'\xef\xbb\xbf"""BOM tool."""\nimport argparse\nSETUP = {"env": ["LMSA_IMAP_USER"]}\n'
        b'p = argparse.ArgumentParser()\np.add_argument("--apply", action="store_true")\n')
    t = tasks_by_name(dash)["bomtool"]
    assert t.doc == "BOM tool." and not t.warnings
    assert [f.flag for f in t.fields] == ["--apply"] and t.fields[0].write
    assert t.setup == {"env": ["LMSA_IMAP_USER"]} and "--apply" in t.write_flags
    with pytest.raises(db.NeedsConfirm):
        db.build_args(t, {}, "--app")
