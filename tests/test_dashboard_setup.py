"""tasks.dashboard_setup: the Setup checklist (#36).

Hermetic: an in-memory keyring (tests/conftest.py), a fake `lms`, a fake
PowerShell runner, a fake Slack sender, and a tmp root with its own
`.env.example`, `.env`, `config/` and `tasks/`.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values

from lmsa import task_meta
from lmsa.search import SearXNG
from lmsa.web import FetchError, HTTPStatusError, InvalidJSON
from tasks import dashboard as db
from tasks import dashboard_setup as ds

from test_dashboard import call, jcall, poll_until_done

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "S3NT1NEL-app-pw-7f3a9c"
HOOK_SENTINEL = "https://hooks.example.test/services/S3NT1NEL-hook-42"

ENV_EXAMPLE = """\
# Names only.
LMSTUDIO_BASE_URL=
LMSTUDIO_API_KEY=
LMSA_CHAT_MODEL=
LMSA_OUTPUT_DIR=
LMSA_SLACK_WEBHOOK=
LMSA_IMAP_HOST=
LMSA_IMAP_USER=
LMSA_IMAP_PASSWORD=
"""

SCHEDULE = """\
# schedule comment
[dispatch]
[tasks.gmail_probe]
cadence = "daily 01:00"
model = "none"
"""

LABELS = '[[label]]\nname = "Receipts"\ndescription = "Purchase confirmations."\n'

FAKE_PROBE = '''"""Fake gmail probe."""
import os
print("probe user", os.environ.get("LMSA_IMAP_USER"), flush=True)
'''
FAKE_DISPATCH = '''"""Fake dispatcher."""
import sys
print("dispatch argv", sys.argv[1:], flush=True)
'''

DEFAULT_MODELS = ("qwen/qwen3-14b", "qwen/qwen3-vl-8b", "text-embedding-nomic-embed-text-v1.5")


class FakeLms:
    def __init__(self, downloaded=DEFAULT_MODELS, instances=()):
        self._downloaded = [{"modelKey": m, "path": m} for m in downloaded]
        self._instances = list(instances)
        self.unloaded: list[str] = []

    def downloaded(self):
        return self._downloaded

    def instances(self):
        return self._instances

    def unload(self, model):
        self.unloaded.append(model)
        self._instances = [m for m in self._instances if m.get("identifier") != model]


_UNSET = object()


class FakeSearXNGFetcher:
    """`.get_json` stand-in for lmsa.web.Fetcher -- no network. With neither
    argument given, behaves as "not reachable" so tests that never touch
    SearXNG stay hermetic without opting in."""

    def __init__(self, json_result=None, raises=_UNSET):
        self.json_result = json_result
        self.raises = (None if json_result is not None else FetchError("no SearXNG in tests")) \
            if raises is _UNSET else raises

    def get_json(self, url, **kwargs):
        if self.raises is not None:
            raise self.raises
        return self.json_result


def _connection_refused(message="connection refused"):
    """A FetchError shaped like a real connection failure -- an OSError
    cause -- rather than a bare FetchError. #89's review: is_connection_error
    classifies by the cause chain now, not by excluding named subclasses, so
    a test that wants "the instance plain isn't answering" has to build one
    of these instead of a bare ``FetchError("connection refused")``."""
    exc = FetchError(message)
    exc.__cause__ = ConnectionRefusedError(message)
    return exc


class FakePowerShell:
    def __init__(self, state="MISSING", rc=0):
        self.state, self.rc = state, rc
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv, timeout):
        assert isinstance(argv, list) and argv[0] == "powershell"
        self.calls.append((argv, timeout))
        if "-File" in argv:
            self.state = "Ready"
            return SimpleNamespace(returncode=0, stdout="TaskPath TaskName State\n\\lmsa\\ dispatch Ready\n", stderr="")
        return SimpleNamespace(returncode=self.rc, stdout=self.state + "\n", stderr="")


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "root"
    (r / "tasks").mkdir(parents=True)
    (r / "config").mkdir()
    (r / "scheduler").mkdir()
    (r / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (r / "tasks" / "gmail_probe.py").write_text(FAKE_PROBE, encoding="utf-8")
    (r / "tasks" / "dispatch.py").write_text(FAKE_DISPATCH, encoding="utf-8")
    (r / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    (r / "config" / "schedule.toml").write_text(SCHEDULE, encoding="utf-8")
    (r / "config" / "labels.toml").write_text(LABELS, encoding="utf-8")
    (r / "config" / "watchlist.toml").write_text('[[item]]\nname = "x"\nurl = "https://example.com/x"\n',
                                                 encoding="utf-8")
    (r / "config" / "front_page.toml").write_text("[fetch]\ntop_n = 5\n", encoding="utf-8")
    (r / "config" / "doc_search.toml").write_text('roots = ["~/Documents"]\n', encoding="utf-8")
    # master_report's own validator (#105) resolves tasks/briefings/schedule
    # against the *real* repo (tasks.master_report's own module-level
    # defaults), not this fake root -- gmail_probe is a real task there too,
    # and send_at must be no later than the real config/schedule.toml's
    # [tasks.master_report] run (06:45) or parse_config rejects it.
    (r / "config" / "master_report.toml").write_text(
        '[[master]]\nname = "test"\ntitle = "Test Report"\nsend_at = "06:00"\nsend = false\n\n'
        '  [[master.section]]\n  task = "gmail_probe"\n  mode = "link"\n', encoding="utf-8")
    (r / "scheduler" / "register-task.ps1").write_text("# fake\n", encoding="utf-8")
    (r / "pyproject.toml").write_text(
        '[project]\ndependencies = ["keyring>=25", "python-dotenv>=1.0"]\n'
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n', encoding="utf-8")
    # Nothing from the real machine's environment leaks in; monkeypatch
    # restores whatever sync_env() changes.
    for key in {*ds.Setup.env_keys(SimpleNamespace(env_example=ROOT / ".env.example")),
                *(line.split("=")[0] for line in ENV_EXAMPLE.splitlines() if "=" in line)}:
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "data" / "lmsa.sqlite"))
    return r


@pytest.fixture
def fakes():
    return SimpleNamespace(lms=FakeLms(), ps=FakePowerShell(), slack_sent=[], server=True,
                           searxng=SearXNG("http://fake-searxng.test:8888", fetcher=FakeSearXNGFetcher()))


@pytest.fixture
def dash(root, tmp_path, fakes):
    d = db.Dashboard(
        root,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "data" / "lmsa.sqlite",
        schedule_path=root / "config" / "schedule.toml",
        probe_server=lambda: fakes.server,
        loaded_models=lambda: [],
        busy=lambda: [],
        token="test-token",
    )

    def send_slack(text):
        fakes.slack_sent.append(text)
        return True

    d.setup = ds.Setup(d, lms=fakes.lms, searxng=fakes.searxng, powershell=fakes.ps, send_slack=send_slack,
                       version_of=lambda name: "99.0")
    yield d
    for run in list(d.runs.values()):
        if run.live:
            d.stop(run.id)
            run.done.wait(15)


@pytest.fixture
def server(dash):
    srv = db.DashboardServer(dash, 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def statuses(srv) -> dict:
    status, body = jcall(srv, "GET", "/api/setup")
    assert status == 200
    return {i["id"]: i for i in body["items"]}


def act(srv, item, action, fields=None, confirm=False):
    return jcall(srv, "POST", "/api/setup/action",
                 {"item": item, "action": action, "fields": fields or {}, "confirm": confirm})


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------

def test_fresh_machine_lists_everything_missing_with_its_fix(server, dash, fakes):
    fakes.server = False
    dash.setup.lms = FakeLms(downloaded=())

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    dash.setup.version_of = not_installed
    items = statuses(server)
    for key in ("gmail", "slack", "lmstudio", "scheduler", "python"):
        assert items[key]["state"] == "missing", (key, items[key])
    assert "LMSA_IMAP_USER" in items["gmail"]["reason"] and "app password" in items["gmail"]["reason"]
    assert "isn't registered" in items["scheduler"]["reason"]
    commands = [h["text"] for h in items["lmstudio"]["hints"] if h["command"]]
    assert "lms get qwen/qwen3-14b" in commands and "lms server start" in commands
    assert items["python"]["hints"][0]["text"] == ds.PIP_COMMAND
    # and each has its action on the page
    status, _, raw = call(server, "GET", "/")
    page = raw.decode()
    for key, action in (("gmail", "save"), ("slack", "save"), ("scheduler", "register"), ("config", "save")):
        assert f'data-item="{key}"' in page
    assert 'data-action="register"' in page and 'data-action="dryrun"' in page
    assert 'type="password" autocomplete="new-password"' in page


def test_gmail_save_turns_green_without_restart(server, dash, root, fake_keyring):
    assert statuses(server)["gmail"]["state"] == "missing"
    status, body = act(server, "gmail", "save", {"address": "someone@example.com", "password": SENTINEL})
    assert status == 200, body
    assert SENTINEL not in json.dumps(body)
    assert fake_keyring.store[("lmsa", "imap")] == SENTINEL
    assert dotenv_values(root / ".env")["LMSA_IMAP_USER"] == "someone@example.com"
    assert SENTINEL not in (root / ".env").read_text(encoding="utf-8")
    item = statuses(server)["gmail"]
    assert item["state"] == "ok" and "someone@example.com" in item["reason"]
    assert os.environ["LMSA_IMAP_USER"] == "someone@example.com"  # tasks launched now inherit it


def test_gmail_test_button_runs_gmail_probe_through_the_launcher(server, dash):
    act(server, "gmail", "save", {"address": "someone@example.com"})
    status, body = act(server, "gmail", "test")
    assert status == 200 and body["run"]["task"] == "gmail_probe"
    state, text = poll_until_done(server, body["run"]["id"])
    assert state["returncode"] == 0 and "probe user someone@example.com" in text


def test_dry_run_goes_through_the_launcher(server, dash):
    status, body = act(server, "scheduler", "dryrun")
    assert status == 200 and body["run"]["task"] == "dispatch"
    _, text = poll_until_done(server, body["run"]["id"])
    assert "dispatch argv ['--dry-run']" in text


def test_gmail_save_validates(server):
    assert act(server, "gmail", "save", {})[0] == 400
    assert act(server, "gmail", "save", {"address": "not an address"})[0] == 400
    assert act(server, "gmail", "save", {"address": "a@b.co", "bogus": "x"})[0] == 400
    assert act(server, "gmail", "save", {"password": "two\nlines"})[0] == 400


# ---------------------------------------------------------------------------
# secrets never leak
# ---------------------------------------------------------------------------

def test_planted_secrets_never_appear_in_any_response_page_or_log(server, dash, tmp_path, root, capfd, caplog):
    assert act(server, "gmail", "save", {"address": "someone@example.com", "password": SENTINEL})[0] == 200
    assert act(server, "slack", "save", {"webhook": HOOK_SENTINEL})[0] == 200
    bodies = []
    for method, path, body in [
        ("GET", "/", None), ("GET", "/api/setup", None), ("GET", "/api/status", None), ("GET", "/api/runs", None),
        ("POST", "/api/setup/action", {"item": "slack", "action": "test", "fields": {}}),
        ("POST", "/api/setup/action", {"item": "gmail", "action": "save", "fields": {"password": "x\ny"}}),
        ("POST", "/api/setup/action", {"item": "config", "action": "load", "fields": {"file": "labels"}}),
    ]:
        bodies.append(call(server, method, path, body)[2].decode("utf-8", "replace"))
    status, run = act(server, "gmail", "test")
    state, text = poll_until_done(server, run["run"]["id"])
    bodies.append(text)
    bodies.append(json.dumps(state))
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.suffix == ".log" for p in files)  # the run log is among what's scanned
    for p in files:
        bodies.append(p.read_bytes().decode("utf-8", "replace"))
    captured = capfd.readouterr()
    bodies += [captured.out, captured.err, caplog.text]
    for b in bodies:
        assert SENTINEL not in b and "S3NT1NEL-hook" not in b


def test_keyring_errors_that_echo_the_secret_are_redacted(server, dash, monkeypatch, fake_keyring):
    def boom(service, name, value):
        raise RuntimeError(f"backend refused {value!r} for {name}")

    monkeypatch.setattr(fake_keyring, "set_password", boom)
    status, body = act(server, "gmail", "save", {"password": SENTINEL})
    assert status == 400
    assert SENTINEL not in json.dumps(body) and "RuntimeError" in body["error"]


def test_unexpected_action_errors_are_redacted_too(dash, monkeypatch):
    def run(s, v):
        raise OSError(f"weird {v['password']}")

    item = ds.Item("leaky", "Leaky", "", lambda s: ds.Status("ok", ""),
                   inputs=(ds.Input("password", kind="secret"),),
                   actions=(ds.Action("go", "go", run, uses=("password",)),))
    monkeypatch.setattr(ds, "PROVIDERS", [lambda s: [item]])
    with pytest.raises(db.BadRequest) as exc:
        dash.setup.act("leaky", "go", {"password": SENTINEL})
    assert SENTINEL not in str(exc.value) and exc.value.__cause__ is None and exc.value.__suppress_context__


def test_env_var_still_wins_over_the_keyring(dash, monkeypatch):
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", "from-env")
    note = dash.setup.set_secret("imap", "from-keyring")
    assert "LMSA_IMAP_PASSWORD" in note and "from-keyring" not in note


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def test_slack_save_and_test(server, fakes, fake_keyring):
    assert act(server, "slack", "save", {"webhook": "http://insecure.example"})[0] == 400
    assert act(server, "slack", "save", {"webhook": HOOK_SENTINEL})[0] == 200
    assert fake_keyring.store[("lmsa", "slack")] == HOOK_SENTINEL
    assert statuses(server)["slack"]["state"] == "ok"
    status, body = act(server, "slack", "test")
    assert status == 200 and fakes.slack_sent


# ---------------------------------------------------------------------------
# .env writes
# ---------------------------------------------------------------------------

def test_write_env_creates_updates_in_place_and_keeps_comments(dash, root):
    s = dash.setup
    s.write_env("LMSA_IMAP_USER", "a@example.com")
    assert (root / ".env").read_text(encoding="utf-8") == "LMSA_IMAP_USER='a@example.com'\n"
    (root / ".env").write_bytes(
        b"# my comment\r\nLMSA_CHAT_MODEL=foo # keep\r\nexport LMSA_IMAP_USER=old\r\nOTHER=1\r\n"
        b"LMSA_IMAP_USER=dup\r\n# tail")
    s.write_env("LMSA_IMAP_USER", "b@example.com")
    s.write_env("LMSA_IMAP_HOST", "imap.example.com")
    text = (root / ".env").read_bytes().decode()
    assert text == ("# my comment\r\nLMSA_CHAT_MODEL=foo # keep\r\nLMSA_IMAP_USER='b@example.com'\r\nOTHER=1\r\n"
                    "LMSA_IMAP_USER='b@example.com'\r\n# tail\r\nLMSA_IMAP_HOST='imap.example.com'\r\n")
    assert dotenv_values(root / ".env")["LMSA_CHAT_MODEL"] == "foo"


@pytest.mark.parametrize("value", ["a b # c", "it's", r"C:\x\n", 'x"y', "  pad  ", "back\\", "caf\u00e9"])
def test_write_env_round_trips_through_dotenv(dash, root, value):
    dash.setup.write_env("LMSA_CHAT_MODEL", value)
    assert dotenv_values(root / ".env")["LMSA_CHAT_MODEL"] == value
    assert os.environ["LMSA_CHAT_MODEL"] == value


@pytest.mark.parametrize("key,value", [
    ("NOT_LISTED", "x"),            # not in .env.example
    ("LMSA_IMAP_PASSWORD", "x"),    # a secret key
    ("LMSA_SLACK_WEBHOOK", "x"),
    ("LMSTUDIO_API_KEY", "x"),
    ("LMSA_IMAP_USER", "a\nLMSA_IMAP_PASSWORD=x"),
    ("LMSA_IMAP_USER", "a\rb"),
    ("LMSA_IMAP_USER", "a\x00b"),
    ("LMSA_IMAP_USER", "${HOME}"),
    ("LMSA_IMAP_USER", "x" * 3000),
])
def test_write_env_refuses(dash, root, key, value):
    with pytest.raises(db.BadRequest):
        dash.setup.write_env(key, value)
    assert not (root / ".env").exists()


def test_env_edited_by_hand_is_picked_up_on_the_next_status(dash, root):
    (root / ".env").write_text("LMSA_IMAP_USER=hand@example.com\n", encoding="utf-8")
    assert dash.setup.env_value("LMSA_IMAP_USER") == "hand@example.com"
    dash.setup.sync_env()
    assert os.environ["LMSA_IMAP_USER"] == "hand@example.com"
    (root / ".env").write_text("", encoding="utf-8")
    dash.setup.sync_env()
    assert "LMSA_IMAP_USER" not in os.environ


def test_real_process_env_is_not_overridden_by_the_file(root, dash, monkeypatch):
    monkeypatch.setenv("LMSA_IMAP_HOST", "from-process")
    s = ds.Setup(dash, lms=FakeLms(), searxng=SearXNG("http://fake-searxng.test:8888", fetcher=FakeSearXNGFetcher()),
                powershell=FakePowerShell())
    s.write_env("LMSA_IMAP_HOST", "from-file")
    assert os.environ["LMSA_IMAP_HOST"] == "from-process"
    assert s.env_value("LMSA_IMAP_HOST") == "from-process"


def test_env_write_is_atomic_when_the_file_is_locked(dash, root, monkeypatch):
    (root / ".env").write_text("LMSA_IMAP_USER=keep@example.com\n", encoding="utf-8")

    def locked(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(ds.os, "replace", locked)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    with pytest.raises(db.Conflict):
        dash.setup.write_env("LMSA_IMAP_USER", "new@example.com")
    assert (root / ".env").read_text(encoding="utf-8") == "LMSA_IMAP_USER=keep@example.com\n"
    assert [p.name for p in root.iterdir() if p.name.endswith(".tmp")] == []


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def test_lmstudio_ok_and_pinned_warning_with_confirmed_unload(server, dash):
    assert statuses(server)["lmstudio"]["state"] == "ok"
    dash.setup.lms = lms = FakeLms(instances=[
        {"identifier": "qwen/qwen3-14b", "ttlMs": None},          # pinned
        {"identifier": "text-embedding-nomic-embed-text-v1.5", "ttlMs": 900000},
        {"identifier": "old-lms", "modelKey": "old-lms"},          # no ttlMs field: unknown, not flagged
    ])
    item = statuses(server)["lmstudio"]
    assert item["state"] == "warning" and "qwen/qwen3-14b" in item["reason"] and "old-lms" not in item["reason"]
    assert item["choices"] == {"model": ["qwen/qwen3-14b"]}
    status, body = act(server, "lmstudio", "unload", {"model": "qwen/qwen3-14b"})
    assert status == 428 and not lms.unloaded
    for bad in ("text-embedding-nomic-embed-text-v1.5", "--all", "", "old-lms"):
        assert act(server, "lmstudio", "unload", {"model": bad}, confirm=True)[0] == 400
    assert not lms.unloaded
    status, body = act(server, "lmstudio", "unload", {"model": "qwen/qwen3-14b"}, confirm=True)
    assert status == 200 and lms.unloaded == ["qwen/qwen3-14b"]


def test_lmstudio_matches_variants_and_reports_missing_models(dash):
    dash.setup.lms = FakeLms(downloaded=())
    dash.setup.lms._downloaded = [{"modelKey": "other", "variants": ["qwen/qwen3-14b"]}]
    st = ds._lmstudio_status(dash.setup)
    assert st.state == "missing"
    assert "chat model" not in st.reason and "vision model qwen/qwen3-vl-8b" in st.reason
    assert ds.Hint("lms get qwen/qwen3-vl-8b", command=True) in st.hints


def test_lms_failure_is_a_warning_not_a_crash(dash):
    class Broken(FakeLms):
        def downloaded(self):
            raise RuntimeError("lms not found")

    dash.setup.lms = Broken()
    st = ds._lmstudio_status(dash.setup)
    assert st.state == "warning" and "lms not found" in st.reason


def test_lms_wrapper_reads_ls_and_ps_json():
    from lmsa.models import Lms

    outputs = {"ls": '[{"modelKey": "a"}, 3]', "ps": '[{"identifier": "a", "ttlMs": null}]'}

    def runner(argv, **kw):
        assert kw["timeout"] and argv[1] in outputs
        return subprocess.CompletedProcess(argv, 0, outputs[argv[1]], "")

    lms = Lms(exe="lms", runner=runner)
    assert lms.downloaded() == [{"modelKey": "a"}]
    assert lms.instances() == [{"identifier": "a", "ttlMs": None}]
    assert lms.loaded() == ["a"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_register_needs_confirm_and_shows_output(server, fakes):
    assert statuses(server)["scheduler"]["state"] == "missing"
    status, body = act(server, "scheduler", "register")
    assert status == 428 and not any("-File" in a for a, _ in fakes.ps.calls)
    status, body = act(server, "scheduler", "register", confirm=True)
    assert status == 200 and "dispatch Ready" in body["output"]
    argv, timeout = next((a, t) for a, t in fakes.ps.calls if "-File" in a)
    assert argv[:3] == ["powershell", "-NoProfile", "-NonInteractive"]
    assert argv[-1].endswith("register-task.ps1") and timeout == ds.REGISTER_TIMEOUT
    assert statuses(server)["scheduler"]["state"] == "ok"


@pytest.mark.parametrize("ps,state", [
    (FakePowerShell("Ready"), "ok"),
    (FakePowerShell("Disabled"), "warning"),
    (FakePowerShell("MISSING"), "missing"),
    (FakePowerShell("", rc=1), "warning"),
])
def test_scheduler_status(dash, ps, state):
    dash.setup.powershell = ps
    assert ds._scheduler_status(dash.setup).state == state
    argv, timeout = ps.calls[0]
    assert argv[-1] == ds.PS_QUERY_TASK and timeout == ds.PS_TIMEOUT


def test_scheduler_timeout_is_a_warning(dash):
    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    dash.setup.powershell = slow
    st = ds._scheduler_status(dash.setup)
    assert st.state == "warning" and "didn't answer" in st.reason


@pytest.mark.slow
def test_a_slow_probe_does_not_hold_up_the_checklist(dash, monkeypatch):
    release = threading.Event()

    def hung(argv, timeout):
        release.wait(10)
        return SimpleNamespace(returncode=0, stdout="Ready\n", stderr="")

    dash.setup.powershell = hung
    monkeypatch.setattr(ds, "PROBE_TIMEOUT", 0.5)
    t0 = time.monotonic()
    items = {i["id"]: i for i in dash.setup.status_all()["items"]}
    release.set()
    assert time.monotonic() - t0 < 5
    assert items["scheduler"]["state"] == "warning" and "still checking" in items["scheduler"]["reason"]
    assert items["gmail"]["state"] == "missing"


# ---------------------------------------------------------------------------
# Python env
# ---------------------------------------------------------------------------

def test_check_requirement():
    import importlib.metadata as md

    versions = {"keyring": "25.2.1", "numpy": "1.9", "pytest": "8.0"}

    def version_of(name):
        if name not in versions:
            raise md.PackageNotFoundError(name)
        return versions[name]

    assert ds.check_requirement("keyring>=25", version_of) is None
    assert ds.check_requirement("numpy>=1.26", version_of) == "numpy 1.9 installed, need >=1.26"
    assert ds.check_requirement("pytest[extra]>=8 ; python_version>'3'", version_of) is None
    assert ds.check_requirement("imapclient>=3", version_of) == "imapclient is not installed"
    assert ds.check_requirement("pytest~=8.0", version_of) is None  # unknown operator: presence only


def test_python_env_against_the_real_venv():
    dash = SimpleNamespace(root=ROOT, db_path=ROOT / "data" / "x.sqlite")
    s = ds.Setup(dash, lms=FakeLms(), searxng=SearXNG("http://fake-searxng.test:8888", fetcher=FakeSearXNGFetcher()),
                powershell=FakePowerShell())
    assert ds._python_status(s).state == "ok"


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

def load(server, name):
    status, body = act(server, "config", "load", {"file": name})
    assert status == 200, body
    return body["fill"]


def test_config_load_edit_save_with_backup(server, dash, root, tmp_path):
    (root / "config" / "labels.toml").write_bytes(LABELS.replace("\n", "\r\n").encode())
    fill = load(server, "labels")
    assert fill["text"] == LABELS.replace("\n", "\r\n")
    new = LABELS + '\n[[label]]\nname = "Travel"\ndescription = "Trips."\n'
    fields = {"file": "labels", "text": new, "base": fill["base"]}
    assert act(server, "config", "save", fields)[0] == 428
    assert (root / "config" / "labels.toml").read_bytes() == LABELS.replace("\n", "\r\n").encode()
    status, body = act(server, "config", "save", fields, confirm=True)
    assert status == 200, body
    assert (root / "config" / "labels.toml").read_bytes() == new.replace("\n", "\r\n").encode()  # CRLF kept
    backups = list((tmp_path / "data" / "dashboard" / "config-backups").glob("labels-*.toml"))
    assert len(backups) == 1 and backups[0].read_bytes() == LABELS.replace("\n", "\r\n").encode()
    # the base moved on: saving again with the returned base works, a stale one doesn't
    assert act(server, "config", "save", {**fields, "base": body["fill"]["base"]}, confirm=True)[0] == 200
    assert act(server, "config", "save", fields, confirm=True)[0] == 409


@pytest.mark.parametrize("name,text,needle", [
    ("labels", "[[label]\nname=", "TOMLDecodeError"),
    ("labels", '[[label]]\nname = "\\\\Inbox"\n', "LabelConfigError"),
    ("schedule", '[dispatch]\n[tasks.gmail_probe]\ncadence = "hourly"\n', "tasks.gmail_probe"),
    ("schedule", '[dispatch]\n[tasks.nope]\ncadence = "daily 01:00"\n', "no module"),
    ("schedule", '[dispatch]\nmax_run_minutes = 999\n', "max_run_minutes"),
    ("watchlist", '[[item]]\nurl = "https://e.x/a"\n[[item]]\nurl = "https://e.x/a"\n', "duplicate url"),
    ("watchlist", '[[item]]\nurl = "https://e.x/a"\ntarget_price = "cheap"\n', "target_price"),
    ("doc_search", 'roots = 3\n', "TypeError"),
])
def test_invalid_config_is_refused_and_the_file_kept(server, root, name, text, needle):
    before = (root / "config" / f"{name}.toml").read_bytes()
    fill = load(server, name)
    status, body = act(server, "config", "save", {"file": name, "text": text, "base": fill["base"]}, confirm=True)
    assert status == 400 and needle in body["error"], body
    assert ".tmp" not in body["error"]
    assert (root / "config" / f"{name}.toml").read_bytes() == before
    assert not [p for p in (root / "config").iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("name", ["../.env", "labels.toml", "..\\labels", "//host/share/x", "", None, "LABELS"])
def test_config_names_come_from_the_allowlist_only(server, name):
    assert act(server, "config", "load", {"file": name})[0] == 400


def test_config_save_requires_a_load_first(server):
    status, body = act(server, "config", "save", {"file": "labels", "text": LABELS, "base": ""}, confirm=True)
    assert status == 400 and "load" in body["error"]


def test_missing_config_can_be_created(server, root):
    (root / "config" / "watchlist.toml").unlink()
    fill = load(server, "watchlist")
    assert fill == {"text": "", "base": "new"}
    status, _ = act(server, "config", "save", {"file": "watchlist", "text": "", "base": "new"}, confirm=True)
    assert status == 200 and (root / "config" / "watchlist.toml").exists()


def test_config_save_when_the_file_is_locked_changes_nothing(server, root, monkeypatch):
    fill = load(server, "labels")

    def locked(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(ds.os, "replace", locked)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    status, body = act(server, "config", "save", {"file": "labels", "text": LABELS + "\n", "base": fill["base"]},
                       confirm=True)
    assert status == 409 and "in another program" in body["error"]
    assert (root / "config" / "labels.toml").read_text(encoding="utf-8") == LABELS
    assert not [p for p in (root / "config").iterdir() if p.name.endswith(".tmp")]


def test_config_status_flags_an_invalid_file(server, root):
    assert statuses(server)["config"]["state"] == "ok"
    (root / "config" / "labels.toml").write_text('[[label]]\nname = "\\\\Trash"\n', encoding="utf-8")
    item = statuses(server)["config"]
    assert item["state"] == "warning" and "labels.toml" in item["reason"]


def test_config_load_and_save_refuse_a_file_outside_the_allowlist(server):
    """#105 acceptance: a request can't name a path outside CONFIG_FILES --
    not another file in config/, and not one outside it either."""
    for bad in ("../secrets", "master_report/../../pyproject", "not_a_real_file"):
        status, body = act(server, "config", "load", {"file": bad})
        assert status == 400 and "unknown config file" in body["error"]
        status, body = act(server, "config", "save", {"file": bad, "text": "x", "base": "new"}, confirm=True)
        assert status == 400 and "unknown config file" in body["error"]


def test_real_repo_config_files_pass_their_validators():
    dash = SimpleNamespace(root=ROOT, db_path=ROOT / "data" / "x.sqlite")
    s = ds.Setup(dash, lms=FakeLms(), searxng=SearXNG("http://fake-searxng.test:8888", fetcher=FakeSearXNGFetcher()),
                powershell=FakePowerShell())
    assert ds._config_status(s).state == "ok", ds._config_status(s).reason


# ---------------------------------------------------------------------------
# requests and registry
# ---------------------------------------------------------------------------

def test_setup_posts_need_the_token_and_setup_gets_the_host(server, fake_keyring):
    body = {"item": "gmail", "action": "save", "fields": {"password": SENTINEL}}
    status, _ = jcall(server, "POST", "/api/setup/action", body, {db.TOKEN_HEADER: "wrong"})
    assert status == 403 and not fake_keyring.store
    status, _ = jcall(server, "POST", "/api/setup/action", body, {"Origin": "http://evil.example"})
    assert status == 403 and not fake_keyring.store
    assert jcall(server, "GET", "/api/setup", headers={"Host": "evil.example"})[0] == 421


@pytest.mark.parametrize("body", [
    {"item": "nope", "action": "save"},
    {"item": "gmail", "action": "nope"},
    {"item": "gmail", "action": "save", "fields": "x"},
    {"item": "gmail", "action": "save", "fields": {"address": 3}},
])
def test_bad_setup_requests(server, body):
    assert jcall(server, "POST", "/api/setup/action", body)[0] == 400


def test_a_registered_provider_adds_items(server, dash, monkeypatch):
    seen = []

    def provider(setup):
        return [ds.Item("newtool", "New tool", "declared by a tool", lambda s: ds.Status("missing", "needs a thing"),
                        actions=(ds.Action("fix", "fix", lambda s, v: seen.append(1) or {"message": "fixed"}),))]

    monkeypatch.setattr(ds, "PROVIDERS", [])
    ds.register(provider)
    ds.register(provider)  # idempotent
    assert ds.PROVIDERS == [provider]
    assert statuses(server)["newtool"]["state"] == "missing"
    assert b'data-item="newtool"' in call(server, "GET", "/")[2]
    assert act(server, "newtool", "fix")[1]["message"] == "fixed" and seen == [1]


def test_setup_module_is_not_a_launcher_tile(dash):
    assert "dashboard_setup" not in {t.name for t in db.discover_tasks(ROOT / "tasks")}


# ---------------------------------------------------------------------------
# review round 2: leaks, races, separators, unknown TTL
# ---------------------------------------------------------------------------

BAD_HOOK = "https://hooks.example.invalid/services/SECRET" + chr(1) + "XYZ"
ENV_HOOK = "hooks.example.invalid/services/T0/B0/ENVSECRET"


def _no_leak(texts, *needles):
    for t in texts:
        for n in needles:
            assert n not in t, (n, t[:300])


def test_send_test_alert_never_leaks_a_malformed_stored_webhook(server, dash, monkeypatch, fake_keyring,
                                                               capfd, caplog):
    import http.client

    from lmsa import notify

    dash.setup.send_slack = ds._send_slack  # the real notify.slack path
    fake_keyring.set_password("lmsa", "slack", BAD_HOOK)

    def urlopen(req, timeout=None):
        raise http.client.InvalidURL(f"URL can't contain control characters. {req.full_url!r}")

    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    status, body = act(server, "slack", "test")
    assert status == 400
    out = capfd.readouterr()
    _no_leak([json.dumps(body), out.out, out.err, caplog.text,
              json.dumps(statuses(server))], "SECRET", "XYZ")


def test_send_test_alert_never_leaks_an_env_webhook_with_no_scheme(server, dash, monkeypatch, capfd, caplog):
    dash.setup.send_slack = ds._send_slack
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", ENV_HOOK)  # Request() raises "unknown url type: '<url>'"
    status, body = act(server, "slack", "test")
    assert status == 400
    out = capfd.readouterr()
    _no_leak([json.dumps(body), out.out, out.err, caplog.text], "ENVSECRET")


def test_notify_slack_logs_only_the_exception_type(monkeypatch, caplog, fake_keyring):
    from lmsa import notify

    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", ENV_HOOK)
    assert notify.slack("x") is False
    assert "ValueError" in caplog.text and "ENVSECRET" not in caplog.text


@pytest.mark.parametrize("hook", [BAD_HOOK, ENV_HOOK, "http://hooks.example.test/x", "https://",
                                  "https://hooks.example.test/a b", "https://hooks.example.test/a" + chr(0x2028) + "b"])
def test_slack_save_refuses_malformed_webhooks(server, fake_keyring, hook):
    status, body = act(server, "slack", "save", {"webhook": hook})
    assert status == 400 and not fake_keyring.store
    _no_leak([json.dumps(body)], hook.strip())


def test_redact_covers_repr_and_escaped_forms():
    value = "pa" + chr(92) + "ss'wd"
    for text in (f"refused {value!r}", f"refused {value}", json.dumps({"v": value}), f"x {ascii(value)}"):
        assert "ss" not in ds.redact(text, value), text


def test_keyring_failure_reports_the_type_only(server, monkeypatch, fake_keyring):
    value = "pa" + chr(92) + "ss'wd"

    def boom(service, name, v):
        raise RuntimeError(f"backend refused {v!r}")

    monkeypatch.setattr(fake_keyring, "set_password", boom)
    status, body = act(server, "gmail", "save", {"password": value})
    assert status == 400 and "RuntimeError" in body["error"] and "refused" not in body["error"]
    assert "ss" not in body["error"].replace("keyring", "")


def test_act_redacts_every_resolvable_secret_even_undeclared(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", "pa" + chr(92) + "ss'wd")
    dash.setup.forget_secrets()  # stored behind the dashboard's back (like a terminal `lmsa.secrets set`)

    def run(s, v):
        raise ValueError(f"oops {'pa' + chr(92) + 'ss' + chr(39) + 'wd'!r}")

    item = ds.Item("leaky2", "Leaky", "", lambda s: ds.Status("ok", ""),
                   actions=(ds.Action("go", "go", run),))
    monkeypatch.setattr(ds, "PROVIDERS", [lambda s: [item]])
    with pytest.raises(db.BadRequest) as exc:
        dash.setup.act("leaky2", "go", {})
    assert "ss" not in str(exc.value).replace("oops", "")


@pytest.mark.slow
def test_concurrent_config_saves_from_one_base_one_wins(dash, root, monkeypatch):
    real = ds.VALIDATORS["labels"]

    def slow(setup, path):
        time.sleep(0.5)
        real(setup, path)

    monkeypatch.setitem(ds.VALIDATORS, "labels", slow)
    fill = dash.setup.act("config", "load", {"file": "labels"})["fill"]
    results = []

    def save(extra):
        try:
            dash.setup.act("config", "save", {"file": "labels", "text": LABELS + extra, "base": fill["base"]},
                           confirm=True)
            results.append("ok")
        except db.Conflict:
            results.append("409")

    threads = [threading.Thread(target=save, args=(f"# tab {i}\n",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert sorted(results) == ["409", "ok"]
    backups = list((dash.db_path.parent / "dashboard" / "config-backups").glob("labels-*.toml"))
    assert len(backups) == 1


def test_backup_names_are_unique_within_one_microsecond(dash, monkeypatch):
    class Frozen(ds.datetime):
        @classmethod
        def now(cls, tz=None):
            return ds.datetime(2026, 9, 25, 1, 2, 3, 4)

    monkeypatch.setattr(ds, "datetime", Frozen)
    for i in range(3):
        fill = dash.setup.act("config", "load", {"file": "labels"})["fill"]
        dash.setup.act("config", "save", {"file": "labels", "text": LABELS + f"# {i}\n", "base": fill["base"]},
                       confirm=True)
    backups = list((dash.db_path.parent / "dashboard" / "config-backups").glob("labels-*.toml"))
    assert len(backups) == 3


def test_write_env_keeps_lines_with_unicode_separators(dash, root):
    sep_line = "# note" + chr(0x2028) + "still the comment" + chr(0x0c) + chr(0x85) + "\n"
    original = sep_line + "LMSA_IMAP_USER=old@example.com\n"
    (root / ".env").write_bytes(original.encode("utf-8"))
    dash.setup.write_env("LMSA_IMAP_USER", "new@example.com")
    assert (root / ".env").read_bytes().decode("utf-8") == sep_line + "LMSA_IMAP_USER='new@example.com'\n"


@pytest.mark.parametrize("ch", [0x2028, 0x2029, 0x0b, 0x0c, 0x1c, 0x1d, 0x1e, 0x85, 0x09, 0x7f, 0x01])
def test_env_values_with_separators_or_control_chars_are_refused(dash, root, ch):
    with pytest.raises(db.BadRequest):
        dash.setup.write_env("LMSA_IMAP_USER", "a" + chr(ch) + "b@example.com")
    with pytest.raises(db.BadRequest):
        dash.setup.set_secret("imap", "a" + chr(ch) + "b")
    assert not (root / ".env").exists()


def test_pinned_check_is_unknown_when_lms_reports_no_ttl(dash):
    dash.setup.lms = FakeLms(instances=[{"identifier": "qwen/qwen3-14b", "modelKey": "qwen/qwen3-14b"}])
    st = ds._lmstudio_status(dash.setup)
    assert st.state == "ok" and "pinned check unknown" in st.reason and "nothing pinned" not in st.reason
    assert any("ttlMs" in h.text and "qwen/qwen3-14b" in h.text for h in st.hints)
    assert st.choices == {"model": []}
    dash.setup.lms = FakeLms(instances=[{"identifier": "qwen/qwen3-14b", "ttlMs": 900000}])
    assert "nothing pinned" in ds._lmstudio_status(dash.setup).reason


# ---------------------------------------------------------------------------
# PR #40 review: redact before clipping, short secrets, slow keyring
# ---------------------------------------------------------------------------

LONG_SECRET = "xoxb-SUPERSECRETTOKEN-1234567890"


def test_secret_straddling_a_clip_boundary_is_not_leaked_in_part(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", LONG_SECRET)
    dash.setup.forget_secrets()

    def status(_s):
        raise RuntimeError("x" * 190 + " " + LONG_SECRET)

    def bad_labels(_s, _p):
        raise ValueError("bad label " + "y" * 170 + LONG_SECRET)

    monkeypatch.setattr(ds, "PROVIDERS", [lambda s: [ds.Item("extra", "Extra", "", status)]])
    monkeypatch.setitem(ds.VALIDATORS, "labels", bad_labels)
    text = json.dumps(dash.setup.status_all())
    for n in range(8, len(LONG_SECRET) + 1):  # no prefix of the secret survives the clip
        assert LONG_SECRET[:n] not in text


def test_short_secrets_do_not_shred_the_page(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", "e")
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", "lms")
    items = {i["id"]: i for i in dash.setup.status_all()["items"]}
    assert "[redacted]" not in json.dumps(items)
    assert "Gmail" in items["gmail"]["reason"] or "LMSA_IMAP_USER" in items["gmail"]["reason"]


def test_command_hints_are_never_redacted(dash, monkeypatch, fakes, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", "qwen/qwen3-14b")  # a secret equal to a model id
    fakes.lms._downloaded = []  # every configured model missing -> `lms get` hints
    hints = [h for i in dash.setup.status_all()["items"] for h in i["hints"] if h.get("command")]
    assert "lms get qwen/qwen3-14b" in [h["text"] for h in hints]


def test_a_hung_keyring_costs_one_bounded_wait_not_the_page(dash, monkeypatch, fake_keyring):
    monkeypatch.setattr(ds, "SECRET_LOOKUP_TIMEOUT", 0.3)
    release = threading.Event()

    def hung(service, name):
        release.wait(10)
        return None

    monkeypatch.setattr(fake_keyring, "get_password", hung)
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", "https://hooks.example.invalid/services/ENVSECRET")
    try:
        t0 = time.monotonic()
        secrets_now = dash.setup.known_secrets()
        assert time.monotonic() - t0 < 2
        assert "https://hooks.example.invalid/services/ENVSECRET" in secrets_now  # env secrets still redacted
        t0 = time.monotonic()
        dash.setup.act("config", "load", {"file": "labels"})
        dash.setup.known_secrets()
        assert time.monotonic() - t0 < 2  # a second request doesn't stack another wait per call
    finally:
        release.set()


def test_storing_a_secret_refreshes_the_redaction_cache(dash, fake_keyring):
    assert dash.setup.known_secrets() == []
    dash.setup.set_secret("imap", LONG_SECRET)
    assert LONG_SECRET in dash.setup.known_secrets()


# ---------------------------------------------------------------------------
# PR #40 final re-review: the redaction cache never loses a known secret
# ---------------------------------------------------------------------------

def _slow_keyring(monkeypatch, ring, delay):
    real = ring.get_password

    def slow(service, name):
        time.sleep(delay)
        return real(service, name)
    monkeypatch.setattr(ring, "get_password", slow)


@pytest.mark.slow
def test_a_read_slower_than_the_timeout_still_fills_the_cache(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", LONG_SECRET)
    dash.setup.forget_secrets()
    monkeypatch.setattr(ds, "SECRET_LOOKUP_TIMEOUT", 0.2)
    _slow_keyring(monkeypatch, fake_keyring, 0.5)
    dash.setup.known_secrets()  # times out; the read keeps going
    time.sleep(1.5)
    t0 = time.monotonic()
    assert LONG_SECRET in dash.setup.known_secrets()
    assert time.monotonic() - t0 < 0.1  # served from the cache the late read filled


def test_storing_a_secret_keeps_the_others_while_the_keyring_is_slow(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "slack", "https://hooks.example.invalid/services/KEEPME")
    dash.setup.forget_secrets()
    assert "https://hooks.example.invalid/services/KEEPME" in dash.setup.known_secrets()  # warm
    monkeypatch.setattr(ds, "SECRET_LOOKUP_TIMEOUT", 0.2)
    _slow_keyring(monkeypatch, fake_keyring, 1.0)
    dash.setup.set_secret("imap", LONG_SECRET)
    known = dash.setup.known_secrets()
    assert LONG_SECRET in known  # the value just stored
    assert "https://hooks.example.invalid/services/KEEPME" in known  # and the last read isn't dropped


@pytest.mark.slow
def test_a_read_racing_a_store_never_hides_the_new_value(dash, monkeypatch, fake_keyring):
    fake_keyring.set_password("lmsa", "imap", "old-PASS-abcdefgh")
    dash.setup.forget_secrets()
    started, go = threading.Event(), threading.Event()
    real = fake_keyring.get_password

    def gated(service, name):
        value = real(service, name)  # reads the OLD value...
        started.set()
        go.wait(5)  # ...and finishes only after the store
        return value
    monkeypatch.setattr(fake_keyring, "get_password", gated)
    monkeypatch.setattr(ds, "SECRET_LOOKUP_TIMEOUT", 0.1)
    dash.setup.known_secrets()
    started.wait(5)
    monkeypatch.setattr(fake_keyring, "get_password", real)
    dash.setup.set_secret("imap", "rotated-PASS-abcdefgh")
    go.set()
    time.sleep(0.3)
    for _ in range(3):
        assert "rotated-PASS-abcdefgh" in dash.setup.known_secrets()


def test_env_secrets_are_read_live_not_from_the_cache(dash, monkeypatch):
    dash.setup.known_secrets()  # warm the cache
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", "fresh-env-secret-123")
    assert "fresh-env-secret-123" in dash.setup.known_secrets()


def test_short_secrets_are_refused_and_submitted_values_always_redacted(dash, monkeypatch, fake_keyring):
    with pytest.raises(db.BadRequest) as exc:
        dash.setup.set_secret("imap", "pw1234x")
    assert "pw1234x" not in str(exc.value)

    def echo(service, name, value):
        raise db.BadRequest(f"keyring rejected {value}")
    monkeypatch.setattr(dash.setup, "set_secret", lambda name, value: echo(None, name, value))
    with pytest.raises(db.BadRequest) as exc:
        dash.setup.act("gmail", "save", {"address": "me@example.com", "password": "pw1234x"})
    assert "pw1234x" not in str(exc.value)


def test_a_keyring_secret_shadowed_by_an_env_var_is_still_redacted(dash, monkeypatch, fake_keyring):
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", "env-shadow-secret-1")
    fake_keyring.set_password("lmsa", "imap", "keyring-hidden-secret-2")
    dash.setup.forget_secrets()
    assert "keyring-hidden-secret-2" in dash.setup.known_secrets()  # read even while the env var wins
    monkeypatch.delenv("LMSA_IMAP_PASSWORD")
    assert "keyring-hidden-secret-2" in dash.setup.known_secrets()  # still there once it stops winning


# ---------------------------------------------------------------------------
# Web search (SearXNG) setup item (#51)
# ---------------------------------------------------------------------------

SEARXNG_RESULTS = {"results": [{"url": "https://example.org/a", "title": "A", "content": "x", "engine": "g"}]}


def test_searxng_item_ok_when_reachable_and_json_enabled(server, dash):
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(json_result=SEARXNG_RESULTS))
    item = statuses(server)["searxng"]
    assert item["state"] == "ok"
    assert "fake-searxng.test:8888" in item["reason"]


def test_searxng_item_missing_when_not_reachable(server, dash):
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=FetchError("connection refused")))
    item = statuses(server)["searxng"]
    assert item["state"] == "missing"
    docker_hints = [h["text"] for h in item["hints"] if h["command"]]
    assert any("docker run" in h and "searxng/searxng" in h for h in docker_hints)


def test_searxng_item_warning_when_json_api_disabled(server, dash):
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=HTTPStatusError(403, "http://fake/search")))
    item = statuses(server)["searxng"]
    assert item["state"] == "warning"
    hints = [h["text"] for h in item["hints"] if h["command"]]
    assert any("formats" in h and "json" in h for h in hints)


def test_searxng_item_warning_on_html_only_instance():
    """A 200 that isn't JSON at all (Fetcher.get_json raises InvalidJSON) is
    the same 'json-disabled' state as an explicit 403."""
    searx = SearXNG("http://fake-searxng.test:8888",
                    fetcher=FakeSearXNGFetcher(raises=InvalidJSON("not JSON")))
    state, reason = ds._searxng_probe(SimpleNamespace(searxng=searx))
    assert state == "warning"
    assert "JSON API" in reason


def test_searxng_item_ok_on_demand_when_down_but_start_command_set(server, dash, monkeypatch):
    """#89: unreachable + LMSA_SEARXNG_START set is an expected, ok state --
    it isn't "missing", it just hasn't been started yet."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "wsl.exe -d Ubuntu -u root --exec sleep 7200")
    monkeypatch.setattr(shutil, "which", lambda exe: r"C:\Windows\System32\wsl.exe")
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=_connection_refused()),
                                 start_command=None)  # the Setup card itself must never spawn
    item = statuses(server)["searxng"]
    assert item["state"] == "ok"
    assert "starts on demand" in item["reason"]
    assert "LMSA_SEARXNG_START" in item["reason"]


def test_searxng_item_missing_when_start_command_names_an_unresolvable_executable(server, dash, monkeypatch):
    """#89 review: LMSA_SEARXNG_START set but its executable can't be found
    on PATH is a misconfiguration, not "starts on demand" -- it must report
    missing and name the executable that didn't resolve."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "no-such-searxng-starter --flag")
    monkeypatch.setattr(shutil, "which", lambda exe: None)
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=_connection_refused()),
                                 start_command=None)
    item = statuses(server)["searxng"]
    assert item["state"] == "missing"
    assert "no-such-searxng-starter" in item["reason"]


def test_searxng_item_still_missing_on_demand_when_start_command_unset(server, dash, monkeypatch):
    monkeypatch.delenv("LMSA_SEARXNG_START", raising=False)
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=_connection_refused()),
                                 start_command=None)
    item = statuses(server)["searxng"]
    assert item["state"] == "missing"


def test_searxng_item_missing_not_on_demand_for_an_http_status_error(server, dash, monkeypatch):
    """A 500 (say) isn't "not running yet" -- starting SearXNG again wouldn't
    fix it, so it must stay 'missing' even with LMSA_SEARXNG_START set."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "wsl.exe -d Ubuntu -u root --exec sleep 7200")
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=HTTPStatusError(500, "http://fake/search")),
                                 start_command=None)
    item = statuses(server)["searxng"]
    assert item["state"] == "missing"


def test_setup_probe_never_spawns_even_when_start_command_is_set(monkeypatch):
    """The Setup card's own SearXNG is built with start_command=None (#89):
    a page view must never spawn anything, however LMSA_SEARXNG_START and
    the instance's own reachability are set."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "wsl.exe -d Ubuntu -u root --exec sleep 7200")
    monkeypatch.setattr(shutil, "which", lambda exe: r"C:\Windows\System32\wsl.exe")

    def boom(*a, **kw):
        raise AssertionError("Setup's probe must never spawn a process")

    searx = SearXNG("http://fake-searxng.test:8888",
                     fetcher=FakeSearXNGFetcher(raises=_connection_refused()),
                     start_command=None, popen=boom)
    state, reason = ds._searxng_probe(SimpleNamespace(searxng=searx))
    assert state == "ok"
    assert "starts on demand" in reason


def test_setup_builds_its_searxng_with_start_command_none_by_default(dash, monkeypatch):
    """Even with LMSA_SEARXNG_START set in the real environment, Setup's
    default-constructed SearXNG (no explicit searxng= kwarg) must not spawn
    on a page view."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "wsl.exe -d Ubuntu -u root --exec sleep 7200")
    setup = ds.Setup(dash, lms=FakeLms(), powershell=FakePowerShell(), send_slack=lambda text: True,
                      version_of=lambda name: "99.0")
    assert setup.searxng.start_command is None


def test_tool_status_recognises_searxng_need(dash):
    t = SimpleNamespace(setup={"searxng": True}, setup_warnings=[])
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(json_result=SEARXNG_RESULTS))
    status = ds._tool_status(dash.setup, t)
    assert status.state == "ok" and "SearXNG" in status.reason


def test_tool_status_reports_searxng_missing_with_a_hint_to_the_card(dash):
    t = SimpleNamespace(setup={"searxng": True}, setup_warnings=[])
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=FetchError("down")))
    status = ds._tool_status(dash.setup, t)
    assert status.state == "missing"
    assert any("Web search (SearXNG)" in h.text for h in status.hints)


def test_task_meta_parses_searxng_need_end_to_end_into_tool_status(dash):
    """A real module source with SETUP = {"searxng": True} must survive
    lmsa.task_meta's ast-based parse (it was silently dropped as an unknown
    key before "searxng" was added to SETUP_KEYS) and reach _tool_status."""
    source = '"""fake tool."""\nSETUP = {"searxng": True}\n'
    meta = task_meta.parse_source(source)
    assert meta.setup == {"searxng": True}
    assert meta.setup_warnings == []

    t = SimpleNamespace(setup=meta.setup, setup_warnings=meta.setup_warnings)
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(json_result=SEARXNG_RESULTS))
    status = ds._tool_status(dash.setup, t)
    assert status.state == "ok" and "SearXNG" in status.reason


def test_task_meta_rejects_a_non_bool_searxng_value():
    meta = task_meta.parse_source('SETUP = {"searxng": "yes"}\n')
    assert meta.setup == {}
    assert any("must be true or false" in w for w in meta.setup_warnings)


def test_task_meta_searxng_false_parses_but_declares_no_need(dash):
    meta = task_meta.parse_source('SETUP = {"searxng": False}\n')
    assert meta.setup == {"searxng": False}
    t = SimpleNamespace(setup=meta.setup, setup_warnings=meta.setup_warnings)
    # a fetcher that would raise if ever called: proves the False branch never probes
    dash.setup.searxng = SearXNG("http://fake-searxng.test:8888",
                                 fetcher=FakeSearXNGFetcher(raises=AssertionError("should not be called")))
    status = ds._tool_status(dash.setup, t)
    assert "SearXNG" not in status.reason

# -- voice memos (#62): faster-whisper / CUDA probes degrade gracefully -------------

def test_voice_item_is_registered():
    assert ds.voice_items in ds.PROVIDERS
    assert [i.id for i in ds.voice_items(None)] == ["voice"]


@pytest.mark.parametrize("installed,cuda,state,needle", [
    (False, None, "missing", "isn't installed"),
    (True, None, "warning", "can't tell whether CUDA"),
    (True, 0, "warning", "on the CPU"),
    (True, 2, "ok", "CUDA available (2 devices)"),
])
def test_voice_status(monkeypatch, tmp_path, installed, cuda, state, needle):
    from lmsa import voice

    monkeypatch.setattr(voice, "whisper_installed", lambda: installed)
    monkeypatch.setattr(voice, "cuda_devices", lambda: cuda)
    monkeypatch.setenv("LMSA_RECORDINGS_DIR", str(tmp_path))
    st = ds.voice_status(None)
    assert st.state == state and needle in st.reason
    if not installed:
        assert st.hints[0].command and "[voice]" in st.hints[0].text


def test_voice_status_missing_recordings_folder_is_a_warning(monkeypatch, tmp_path):
    from lmsa import voice

    monkeypatch.setattr(voice, "whisper_installed", lambda: True)
    monkeypatch.setattr(voice, "cuda_devices", lambda: 1)
    monkeypatch.setenv("LMSA_RECORDINGS_DIR", str(tmp_path / "nope"))
    st = ds.voice_status(None)
    assert st.state == "warning" and "not found" in st.reason
