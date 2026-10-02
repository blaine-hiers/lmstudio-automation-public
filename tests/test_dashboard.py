"""tasks.dashboard: catalog, argv building, request checks, run lifecycle.

No network and no LM Studio: the server runs in-process on port 0 over a
fake tasks dir (tiny fake task modules run as real subprocesses), with fake
status probes injected.
"""

from __future__ import annotations

import ctypes
import http.client
import io
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa.store import Store
from tasks import dashboard as db

ROOT = Path(__file__).resolve().parents[1]

FAKEJOB = '''\
"""Fake job <script>alert(1)</script> for the dashboard tests."""
import os, sys, time
from pathlib import Path
print("fakejob start", sys.argv[1:], flush=True)
print("caf\\u00e9 \\u2713", flush=True)
out = Path(os.environ["LMSA_OUTPUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
(out / "fakejob-20260925-0101.html").write_text("<p>fake report</p>", encoding="utf-8")
print("fakejob done", flush=True)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
'''

SLEEPER = '''\
"""Sleeps, with a child process of its own."""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"child {child.pid}", flush=True)
time.sleep(120)
'''

SCHEDULE = '''\
[dispatch]
[tasks.fakejob]
cadence = "daily 01:00"
model = "none"
[tasks.sleeper]
cadence = "every 6 hours"
model = "chat"
[tasks.chatty]
cadence = "daily 02:00"
model = "chat"
'''


WRITER = '''"""Writes a report half a second in."""
import os, time
from pathlib import Path
time.sleep(0.5)
(Path(os.environ["LMSA_OUTPUT_DIR"]) / "writer-report.html").write_text("w", encoding="utf-8")
'''


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tasks" / "dashboard.py").write_text('"""the dashboard itself"""', encoding="utf-8")
    (root / "tasks" / "fakejob.py").write_text(FAKEJOB, encoding="utf-8")
    (root / "tasks" / "sleeper.py").write_text(SLEEPER, encoding="utf-8")
    (root / "tasks" / "chatty.py").write_text('"""Chatty."""\nimport time\ntime.sleep(60)\n', encoding="utf-8")
    (root / "tasks" / "mailer.py").write_text(
        '"""Reads mail."""\nfrom lmsa.sources.gmail import Gmail\n', encoding="utf-8"
    )
    (root / "tasks" / "fails.py").write_text(
        '"""Fails after a while, writing nothing."""\nimport sys, time\ntime.sleep(2)\nsys.exit(3)\n',
        encoding="utf-8")
    (root / "tasks" / "writer.py").write_text(WRITER, encoding="utf-8")
    (root / "tasks" / "brief.py").write_text(
        '"""Runs briefly."""\nimport time\nprint("brief", flush=True)\ntime.sleep(1.5)\n',
        encoding="utf-8")
    (root / "schedule.toml").write_text(SCHEDULE, encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "data" / "lmsa.sqlite"))
    return root


@pytest.fixture
def make_dash(fake_root, tmp_path):
    """Dashboards over the same dirs -- a second one stands in for a restart."""
    made = []

    def make(token="test-token"):
        d = db.Dashboard(
            fake_root,
            output_dir=tmp_path / "output",
            db_path=tmp_path / "data" / "lmsa.sqlite",
            schedule_path=fake_root / "schedule.toml",
            probe_server=lambda: True,
            loaded_models=lambda: ["qwen/qwen3-14b"],
            busy=lambda: [],
            token=token,
        )
        made.append(d)
        return d

    yield make
    for d in made:
        for run in list(d.runs.values()):
            if run.live:
                d.stop(run.id)
                run.done.wait(15)


@pytest.fixture
def dash(make_dash):
    return make_dash()


@pytest.fixture
def server(dash):
    srv = db.DashboardServer(dash, 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = {}
    if method == "POST":
        h = {"Content-Type": "application/json", db.TOKEN_HEADER: srv.dashboard.token}
    h.update(headers or {})
    conn.request(method, path, body=data, headers=h)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, resp.headers, raw


def jcall(srv, method, path, body=None, headers=None):
    status, headers_, raw = call(srv, method, path, body, headers)
    return status, json.loads(raw) if raw else None


def poll_until_done(srv, run_id, timeout=30):
    text, offset, deadline = "", 0, time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, state = jcall(srv, "GET", f"/api/runs/{run_id}?offset={offset}")
        assert status == 200
        text += state["text"]
        offset = state["offset"]
        if state["done"]:
            return state, text
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} never finished; log so far: {text!r}")


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_catalog_discovers_modules_with_group_cadence_and_doc(dash):
    tasks = {t.name: t for t in dash.catalog()}
    assert set(tasks) == {"fakejob", "sleeper", "chatty", "mailer", "fails", "writer", "brief"}  # no __init__, no dashboard
    assert tasks["fakejob"].doc.startswith("Fake job <script>")
    assert tasks["fakejob"].cadence == "daily 01:00" and tasks["fakejob"].model is None
    assert tasks["sleeper"].cadence == "every 6 hours" and tasks["sleeper"].model == "chat"
    assert tasks["mailer"].cadence is None  # unscheduled -> "manual only" on the page
    assert tasks["mailer"].group == "Gmail"
    assert tasks["fakejob"].group == "Other"


def test_real_catalog_lists_every_task_module():
    tasks = db.discover_tasks(ROOT / "tasks")
    names = {t.name for t in tasks}
    on_disk = {p.stem for p in (ROOT / "tasks").glob("*.py")} - db.SKIP_MODULES
    assert names == on_disk
    groups = {t.name: t.group for t in tasks}
    assert groups["dispatch"] == "Scheduler"
    assert groups["labeler"] == "Gmail"
    assert groups["healthcheck"] == "Diagnostics & manual"
    assert groups["front_page"] == "Other"
    assert all(t.doc for t in tasks)
    assert {t.name: t.cadence for t in tasks}["ask"] is None


def _parser_flags(source: str) -> set[str]:
    """Every option string / positional dest in a module's add_argument calls."""
    return set(re.findall(r"""add_argument\(\s*["']([^"']+)["']""", source))


def test_curated_fields_name_flags_the_task_parser_accepts():
    for task, fields in db.FIELDS.items():
        source = (ROOT / "tasks" / f"{task}.py").read_text(encoding="utf-8")
        accepted = _parser_flags(source)
        for f in fields:
            assert f.flag in accepted, f"tasks.{task} has no argparse {f.flag!r}"
            if f.kind == "positional":
                assert not f.flag.startswith("-")


def test_flag_check_would_catch_an_invented_flag():
    source = 'p.add_argument("--days", type=int)\np.add_argument(\n    "--apply", action="store_true")\n'
    assert _parser_flags(source) == {"--days", "--apply"}
    assert "--bogus" not in _parser_flags(source)


def test_ask_scope_field_renders_as_a_dropdown_with_every_scope():
    """#68: the dashboard's ask tile offers --scope as a `Field.choices`
    dropdown covering every registered scope plus "all"."""
    from tasks.ask import SCOPE_CHOICES

    scope_field = next(f for f in db.FIELDS["ask"] if f.flag == "--scope")
    assert scope_field.kind == "text"
    assert scope_field.choices == SCOPE_CHOICES
    html = db._field_html("ask", scope_field)
    assert "<select" in html
    for choice in SCOPE_CHOICES:
        assert f'<option value="{choice}">{choice}</option>' in html


def test_write_flags_are_marked():
    writes = {(t, f.flag) for t, fs in db.FIELDS.items() for f in fs if f.write}
    assert writes == {
        ("labeler", "--apply"), ("labeler", "--undo"),
        ("reply_drafts", "--apply"), ("reply_drafts", "--retry"),
        ("file_sorter", "--apply"), ("file_sorter", "--undo"),
        ("inbox_actions", "--apply"), ("inbox_actions", "--undo"),
    }


# ---------------------------------------------------------------------------
# argv building
# ---------------------------------------------------------------------------

def _task(name):
    return db.TaskInfo(name, "", "Other", fields=db.FIELDS[name])


def test_build_args_numbers_bools_and_positional_last():
    args = db.build_args(_task("doc_search"), {"query": "-roof budget", "--k": "5", "--answer": True})
    assert args == ["--k", "5", "--answer", "--", "-roof budget"]
    assert db.build_args(_task("shootout"), {"--no-judge": False, "--max-tokens": 200}) == ["--max-tokens", "200"]


@pytest.mark.parametrize("values,extra", [
    ({"--bogus": "1"}, ""),
    ({"--k": "5; rm -rf"}, ""),
    ({"--k": True}, ""),
    ({"--answer": "yes"}, ""),
    ({"query": "x\x00y"}, ""),
    ({"query": "q"}, 'unclosed "quote'),
])
def test_build_args_rejects_bad_values(values, extra):
    with pytest.raises(db.BadRequest):
        db.build_args(_task("doc_search"), values, extra)


def test_ask_requires_a_question():
    with pytest.raises(db.BadRequest, match="question"):
        db.build_args(_task("ask"), {"question": "   "})
    with pytest.raises(db.BadRequest):
        db.build_args(_task("ask"), {})


@pytest.mark.parametrize("values,extra", [
    ({"--apply": True}, ""),
    ({"--undo": "12"}, ""),
    ({}, "--apply"),
    ({}, "--app"),  # argparse accepts unambiguous prefixes
    ({}, "--undo=12"),
])
def test_write_flags_need_confirm(values, extra):
    with pytest.raises(db.NeedsConfirm):
        db.build_args(_task("labeler"), values, extra)
    assert db.build_args(_task("labeler"), values, extra, confirm=True)


def test_read_only_extra_args_need_no_confirm():
    assert db.build_args(_task("labeler"), {"--days": "3"}, "--limit 5") == ["--days", "3", "--limit", "5"]


def test_split_extra_keeps_windows_paths():
    text = r'--root "C:\Users\me\My Files" --root D:\dl'
    assert db.split_extra(text) == ["--root", r"C:\Users\me\My Files", "--root", r"D:\dl"]
    # The pitfalls split_extra avoids: POSIX mode eats backslashes, and
    # posix=False keeps the quote characters inside the argument.
    assert shlex.split(r"D:\dl") == ["D:dl"]
    assert shlex.split(r'"C:\My Files"', posix=False) == [r'"C:\My Files"']


def test_start_allowlists_the_task_name(dash):
    for bad in ("../evil", "os", "dashboard", "__init__", "tasks.fakejob", "", None, 3, "FakeJob"):
        with pytest.raises(db.BadRequest):
            dash.start(bad)


def test_start_uses_an_argv_list_without_shell(dash, monkeypatch):
    seen = {}

    class FakePopen:
        pid = 1

        def __init__(self, argv, **kw):
            seen["argv"], seen["kw"] = argv, kw

        def wait(self):
            return 0

    monkeypatch.setattr(db.subprocess, "Popen", FakePopen)
    run, _ = dash.start("fakejob", {}, "--x 'a b'")
    run.done.wait(5)
    assert seen["argv"] == [dash.python, "-m", "tasks.fakejob", "--x", "a b"]
    assert not seen["kw"].get("shell")
    assert seen["kw"]["cwd"] == str(dash.root)


# ---------------------------------------------------------------------------
# launcher cards: compact by default, Options behind an expander (#102)
# ---------------------------------------------------------------------------

def test_options_closed_by_default_with_no_required_field_and_no_failure(dash):
    html = db._launcher_html(dash, _task("attachments"), None)
    assert '<details class="options" data-options="attachments">' in html
    assert "open" not in html[: html.index("<summary>Options</summary>")]
    assert "data-must-open" not in html[: html.index("<summary>Options</summary>")]


def test_options_open_by_default_when_a_field_is_required(dash):
    # ask's positional "question" is required.
    html = db._launcher_html(dash, _task("ask"), None)
    assert '<details class="options" open data-must-open="1" data-options="ask">' in html


def test_options_open_by_default_when_the_last_run_failed(dash):
    html = db._launcher_html(dash, _task("attachments"), None, failed=True)
    assert '<details class="options" open data-must-open="1" data-options="attachments">' in html


def test_options_open_by_default_when_a_field_is_prefilled(dash):
    html = db._launcher_html(dash, _task("attachments"), None, values={"--limit": "5"})
    assert '<details class="options" open data-must-open="1" data-options="attachments">' in html
    assert 'value="5"' in html


def test_a_task_with_no_fields_gets_no_options_expander(dash):
    empty = db.TaskInfo("noop", "does nothing", "Other", fields=())
    html = db._launcher_html(dash, empty, None)
    assert 'class="options"' not in html


def test_run_with_default_values_works_from_a_collapsed_options_card(server, monkeypatch):
    # fakejob has no curated FIELDS entry until this test gives it a
    # non-required one, so its card's Options starts collapsed.
    monkeypatch.setitem(db.FIELDS, "fakejob", (db.Field("--x", "text"),))
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    fake = page[page.index('data-task="fakejob"'):]
    assert '<details class="options" data-options="fakejob">' in fake[: fake.index("</form>")]
    status, j = jcall(server, "POST", "/api/run", {"task": "fakejob"})
    assert status == 201
    poll_until_done(server, j["id"])


def test_a_must_open_card_marks_itself_so_a_stale_saved_close_cant_win(dash):
    # Regression: OPTIONS_JS runs after the server has already decided a
    # card must be open (required field / failed run / prefill). A `data-
    # must-open` marker is how it tells a saved "closed" from a session
    # before that became true not to apply -- there's no JS runner in this
    # suite, so this checks the marker the server emits and the guard in
    # OPTIONS_JS's own source that has to respect it.
    must = db._launcher_html(dash, _task("ask"), None)
    tag = must[: must.index("<summary>Options</summary>")]
    assert 'open' in tag and 'data-must-open="1"' in tag

    closed = db._launcher_html(dash, _task("attachments"), None)
    tag = closed[: closed.index("<summary>Options</summary>")]
    assert "data-must-open" not in tag

    # The saved-"0" branch must be conditioned on !mustOpen, so a must-open
    # card's `open` (set from data-must-open, read before the saved value is
    # applied) is never cleared by a stale saved close.
    js = db.OPTIONS_JS
    assert "mustOpen = det.dataset.mustOpen === '1'" in js
    close_branch = re.search(r"else if \(saved === '0'([^)]*)\) det\.open = false;", js)
    assert close_branch and "!mustOpen" in close_branch.group(1)
    assert re.search(r"if \(saved === '1'\) det\.open = true;", js)  # a saved "open" is never vetoed


# ---------------------------------------------------------------------------
# request checks
# ---------------------------------------------------------------------------

def test_page_embeds_token_and_escapes_docstrings(server):
    status, headers, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    assert status == 200
    assert '<meta name="lmsa-token" content="test-token">' in page
    assert "Fake job &lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)" not in page
    csp = headers["Content-Security-Policy"]
    assert "script-src 'nonce-" in csp and "frame-ancestors 'none'" in csp
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert f'<script nonce="{nonce}">' in page
    assert page.count("<script") == 1
    assert 'href="/reports/index.html"' in page


def test_every_section_and_setup_card_gets_a_collapse_toggle(server):
    status, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    assert status == 200
    # Status, Setup and every task-group section on the main page. Console is
    # a dock now (#100), with its own toggle -- checked separately below.
    for sid in ("status-sec", "setup-sec", "group-gmail", "group-other"):
        assert f'data-collapse="{sid}"' in page
        assert f'aria-controls="body-{sid}"' in page
        assert f'id="body-{sid}"' in page
    assert page.count('class="collapse-toggle"') >= 4
    assert page.count('aria-expanded="true"') >= 4
    # Each Setup checklist card gets its own toggle too (gmail is a built-in item).
    assert 'data-collapse="setup-gmail"' in page and 'id="body-setup-gmail"' in page
    # Clear finished lives in the console dock.
    assert '<button class="btn btn-ghost btn-sm" type="button" id="c-clear">clear finished</button>' in page


def test_console_dock_sits_outside_the_tabpanels_and_works_without_js(server):
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    dock = page[page.index('<div class="console-dock"'): page.index("</body>")]
    assert 'id="console-dock"' in dock
    # Server-rendered open so the console is reachable with JS off; TABS_JS
    # collapses it on load by default (setDock from lmsa-dock-open).
    m = re.search(r'<button class="collapse-toggle dock-toggle" type="button" id="dock-toggle" [^>]*>', dock)
    assert m and 'aria-expanded="true"' in m.group(0) and 'aria-controls="dock-body"' in m.group(0)
    assert '<div class="dock-body" id="dock-body">' in dock
    assert "setDock(open)" in page  # the on-load collapse that makes it closed by default with JS
    assert 'id="c-clear"' in dock  # the console itself lives inside the dock body
    # The dock markup comes after </main>, so it isn't nested in any tabpanel.
    assert page.index('<div class="console-dock"') > page.index("</main>")


def test_tablist_covers_overview_tasks_inbox_setup_each_section_once(server):
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    assert 'role="tablist"' in page
    for tab_id, label in (("overview", "Overview"), ("tasks", "Tasks"), ("inbox", "Inbox"), ("setup", "Setup")):
        assert f'id="tab-{tab_id}"' in page and f'role="tab"' in page
        assert f'aria-controls="{tab_id}"' in page
        assert f'id="{tab_id}" class="tabpanel"' in page
    # Every section that used to sit in the single stacked page is now inside
    # exactly one tabpanel: status/setup/inbox/every task group.
    overview = page[page.index('id="overview" class="tabpanel"'): page.index('id="tasks" class="tabpanel"')]
    tasks_panel = page[page.index('id="tasks" class="tabpanel"'): page.index('id="inbox" class="tabpanel"')]
    inbox_panel = page[page.index('id="inbox" class="tabpanel"'): page.index('id="setup" class="tabpanel"')]
    setup_panel = page[page.index('id="setup" class="tabpanel"'):]
    assert 'id="status-sec"' in overview and 'id="status-sec"' not in tasks_panel
    assert 'id="group-gmail"' in tasks_panel and 'id="group-gmail"' not in overview
    assert 'id="inbox-sec"' in inbox_panel and 'id="inbox-sec"' not in tasks_panel
    assert 'id="setup-sec"' in setup_panel and 'id="setup-sec"' not in overview
    # First tab active, rest inactive; no-JS fallback: no panel is server-hidden.
    assert 'id="tab-overview" class="tab" role="tab" aria-controls="overview" aria-selected="true"' in page
    assert 'aria-selected="false"' in page
    assert 'class="tabpanel" role="tabpanel" aria-labelledby="tab-overview"' in page
    for panel_id in ("overview", "tasks", "inbox", "setup"):
        chunk = page[page.index(f'id="{panel_id}" class="tabpanel"'):][:80]
        assert "hidden" not in chunk


def test_prefill_url_selects_the_tasks_tab(server, monkeypatch):
    monkeypatch.setitem(db.FIELDS, "fakejob", (db.Field("--x", "text"),))
    monkeypatch.setitem(db.PREFILL_FIELDS, "fakejob", frozenset({"--x"}))
    status, _, raw = call(server, "GET", "/?task=fakejob&x=hello")
    page = raw.decode("utf-8")
    assert status == 200
    # The body flags a prefill deep link (#70); TABS_JS reads this to force
    # the Tasks tab active on load instead of the remembered/hash tab.
    assert 'data-prefill-task="fakejob"' in page
    assert 'value="hello"' in page
    # The prefilled card's Options opens too (#102), not just the tab -- and
    # it's marked must-open so a stale saved close can't undo that (#102).
    fake = page[page.index('data-task="fakejob"'):]
    assert '<details class="options" open data-must-open="1" data-options="fakejob">' in fake[: fake.index("</form>")]


def test_collapse_toggle_is_a_real_keyboard_accessible_button(server):
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    m = re.search(r'<button class="collapse-toggle" type="button" data-collapse="status-sec" [^>]*>', page)
    assert m and 'aria-expanded="true"' in m.group(0) and 'aria-controls="body-status-sec"' in m.group(0)


def test_group_toggle_shows_an_attention_count_for_a_failed_task(server, tmp_path):
    _seed_scheduled_failure(tmp_path, task="fakejob")  # fakejob is in the "Other" group
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    group = page[page.index('id="group-other"'): page.index("</h2>", page.index('id="group-other"'))]
    assert 'class="collapse-attn"' in group and ">1<" in group


def test_errors_page_sections_get_a_collapse_toggle(server, tmp_path):
    _seed_scheduled_failure(tmp_path)
    _, _, raw = call(server, "GET", "/errors")
    page = raw.decode("utf-8")
    for sid in ("err-failures-sec", "err-detail-sec", "err-gaveup-sec"):
        assert f'data-collapse="{sid}"' in page and f'id="body-{sid}"' in page
    # Collapse wiring runs first, so opening a failure (even from the URL hash on
    # load) can expand a collapsed Detail through its live toggle.
    from lmsa.report import COLLAPSE_JS
    from lmsa.errors_page import ERRORS_JS
    assert page.index(COLLAPSE_JS.strip()[:40]) < page.index(ERRORS_JS.strip()[:40])


@pytest.mark.parametrize("host", ["evil.example:80", "127.0.0.1", "127.0.0.1:1", "rebind.attacker.test"])
def test_foreign_host_header_is_refused(server, host):
    assert call(server, "GET", "/", headers={"Host": host})[0] == 421
    assert call(server, "POST", "/api/run", {"task": "fakejob"}, {"Host": host})[0] == 421


def test_localhost_host_header_is_allowed(server):
    assert call(server, "GET", "/api/status", headers={"Host": f"localhost:{server.port}"})[0] == 200


@pytest.mark.parametrize("headers,expected", [
    ({db.TOKEN_HEADER: ""}, 403),
    ({db.TOKEN_HEADER: "test-token-nope"}, 403),
    ({"Origin": "http://evil.example"}, 403),
    ({"Origin": "null"}, 403),
    ({"Content-Type": "text/plain"}, 415),
])
def test_post_checks(server, headers, expected):
    status, body = jcall(server, "POST", "/api/run", {"task": "fakejob"}, headers)
    assert status == expected
    assert not server.dashboard.runs


def test_post_without_token_header(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("POST", "/api/run", body=b'{"task":"fakejob"}', headers={"Content-Type": "application/json"})
    assert conn.getresponse().status == 403
    conn.close()


def test_post_with_own_origin_is_allowed(server):
    status, body = jcall(server, "POST", "/api/run", {"task": "fakejob"},
                         {"Origin": f"http://127.0.0.1:{server.port}"})
    assert status == 201
    poll_until_done(server, body["id"])


def test_server_refuses_non_loopback_host(dash):
    with pytest.raises(ValueError):
        db.DashboardServer(dash, 0, host="0.0.0.0")


def test_unknown_task_over_http_is_400(server):
    status, body = jcall(server, "POST", "/api/run", {"task": "../../evil"})
    assert status == 400 and not server.dashboard.runs


# ---------------------------------------------------------------------------
# /reports/
# ---------------------------------------------------------------------------

def test_reports_serves_output_files_sandboxed(server, tmp_path):
    (tmp_path / "output" / "r-20260925-0101.html").write_text("<p>hi</p>", encoding="utf-8")
    status, headers, raw = call(server, "GET", "/reports/r-20260925-0101.html")
    assert status == 200 and raw == b"<p>hi</p>"
    assert headers["Content-Security-Policy"].startswith("sandbox")
    assert headers["Content-Type"].startswith("text/html")


@pytest.mark.parametrize("path", [
    "/reports/../secret.txt",
    "/reports/%2e%2e/secret.txt",
    "/reports/..%5csecret.txt",
    "/reports/%2e%2e%5csecret.txt",
    "/reports/..%2fsecret.txt",
    "/reports/" + "%2e%2e/" * 3 + "secret.txt",
    "/reports/C:%5cWindows%5cwin.ini",
    "/reports/%00",
    "/reports/nope.html",
])
def test_reports_refuses_anything_outside_output(server, tmp_path, path):
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    status, _, raw = call(server, "GET", path)
    assert status == 404
    assert b"secret\"" not in raw and raw != b"secret"


@pytest.mark.parametrize("rel", [
    "//192.0.2.1/share/x.html",
    "\\\\192.0.2.1\\share\\x.html",
    "%2f%2f192.0.2.1/share/x.html",
    "%5c%5c192.0.2.1%5cshare%5cx.html",
    "/abs.html",
    "C:x.html",
    "index.html::$DATA",
])
def test_resolve_report_refuses_unc_and_absolute_without_touching_disk(dash, monkeypatch, rel):
    """`out / "//host/share"` is a UNC path; resolve() on it would open an SMB
    connection (NTLM leak) from an unauthenticated cross-site GET."""
    touched = []
    real = Path.resolve

    def spy(self, *a, **kw):
        if "192.0.2.1" in str(self):
            touched.append(str(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "resolve", spy)
    assert dash.resolve_report(rel) is None
    assert touched == []


def test_reports_unc_request_is_refused_fast(server):
    t0 = time.monotonic()
    status, _, _ = call(server, "GET", "/reports//192.0.2.1/share/x.html")
    assert status == 404
    assert time.monotonic() - t0 < 2


def test_report_url_only_for_files_inside_output(dash, tmp_path):
    inside = tmp_path / "output" / "a b.html"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "elsewhere.html"
    outside.write_text("x", encoding="utf-8")
    assert dash.report_url(inside) == "/reports/a%20b.html"
    assert dash.report_url(outside) is None
    assert dash.report_url(None) is None


# ---------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------

def test_run_lifecycle_streams_log_exit_code_and_report(server, tmp_path):
    status, body = jcall(server, "POST", "/api/run", {"task": "fakejob", "extra": "--flag x"})
    assert status == 201 and body["warning"] is None
    state, text = poll_until_done(server, body["id"])
    assert "fakejob start ['--flag', 'x']" in text
    assert "café ✓" in text and "fakejob done" in text
    assert state["returncode"] == 0 and not state["stopped"]
    assert state["report"] == "/reports/fakejob-20260925-0101.html"
    assert call(server, "GET", state["report"])[2] == b"<p>fake report</p>"
    run = server.dashboard.runs[body["id"]]
    assert run.log_path.parent == tmp_path / "data" / "dashboard" / "logs"
    assert run.log_path.name.startswith("fakejob-")


def test_failed_run_reports_its_exit_code(server, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    status, body = jcall(server, "POST", "/api/run", {"task": "fakejob"})
    state, _ = poll_until_done(server, body["id"])
    assert state["returncode"] == 3


def test_report_prefers_the_runs_row(dash, tmp_path):
    row_report = tmp_path / "output" / "from-row.html"
    row_report.write_text("x", encoding="utf-8")
    run = db.Run("abc", "fakejob", [], tmp_path / "x.log", time.time() - 5, None)
    with Store(dash.db_path) as store, store.record_run("tasks.fakejob") as r:
        r.report_path = row_report
    (tmp_path / "output" / "newer.html").write_text("x", encoding="utf-8")
    assert dash._find_report(run) == row_report


def test_log_offsets_resume_mid_character(dash, tmp_path):
    log = tmp_path / "l.log"
    log.write_bytes("ab✓".encode("utf-8")[:-1])  # the check mark is cut off
    run = db.Run("r1", "fakejob", [sys.executable], log, time.time(), None)
    first = dash.run_state(run, 0)
    assert first["text"] == "ab" and first["offset"] == 2
    log.write_bytes("ab✓z".encode("utf-8"))
    run.returncode, run.finished = 0, time.time()
    run.done.set()
    second = dash.run_state(run, first["offset"])
    assert second["text"] == "✓z" and second["done"]


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


@pytest.mark.slow
def test_double_launch_is_409_and_stop_kills_the_tree(server, monkeypatch):
    killed = []
    real_kill = db.kill_tree
    monkeypatch.setattr(db, "kill_tree", lambda proc: (killed.append(proc.pid), real_kill(proc)))

    status, body = jcall(server, "POST", "/api/run", {"task": "sleeper"})
    assert status == 201
    run_id = body["id"]
    assert jcall(server, "POST", "/api/run", {"task": "sleeper"})[0] == 409

    text, deadline = "", time.monotonic() + 20
    while "child " not in text and time.monotonic() < deadline:
        text += jcall(server, "GET", f"/api/runs/{run_id}?offset={len(text.encode())}")[1]["text"]
        time.sleep(0.1)
    child = int(re.search(r"child (\d+)", text).group(1))
    parent = server.dashboard.runs[run_id].proc.pid
    assert _alive(child) and _alive(parent)

    status, body = jcall(server, "POST", f"/api/runs/{run_id}/stop", {})
    assert status == 200 and body["stopped"]
    state, _ = poll_until_done(server, run_id)
    assert state["stopped"] and state["returncode"] != 0
    assert killed == [parent]
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child) and not _alive(parent)

    # once it's stopped, the task can be launched again
    status, body = jcall(server, "POST", "/api/run", {"task": "sleeper"})
    assert status == 201


def test_stop_unknown_run_is_404(server):
    assert jcall(server, "POST", "/api/runs/deadbeef/stop", {})[0] == 404
    assert jcall(server, "GET", "/api/runs/deadbeef")[0] == 404


@pytest.mark.slow
def test_second_model_task_gets_a_soft_warning(dash):
    run, warning = dash.start("sleeper")
    assert warning is None
    run2, warning2 = dash.start("chatty")
    assert run2.live and "sleeper" in warning2
    _, warning3 = dash.start("fakejob")  # no model: no warning
    assert warning3 is None


# ---------------------------------------------------------------------------
# status + last runs
# ---------------------------------------------------------------------------

def test_status_uses_injected_probes(dash):
    with Store(dash.db_path) as store:
        with store.record_run("tasks.fakejob"):
            pass
        store.conn.execute("INSERT INTO runs(task, started, status) VALUES ('old', '2000-01-01T00:00:00+00:00', 'ok')")
        store.conn.commit()
    s = dash.status()
    assert s["server"] is True and s["models"] == ["qwen/qwen3-14b"]
    assert s["busy"] == [] and s["runs_today"] == 1


def test_status_survives_failing_probes(dash):
    def boom():
        raise OSError("nvidia-smi missing")

    dash.probe_server = boom
    dash.loaded_models = boom
    dash.busy = boom
    s = dash.status()
    assert s["server"] is False and s["models"] == [] and "nvidia-smi missing" in s["models_error"]
    assert s["busy"] and "can't read" in s["busy"][0]


def test_last_runs_handles_both_task_name_forms(dash, tmp_path):
    with Store(dash.db_path) as store:
        with store.record_run("tasks.fakejob") as r:
            r.report_path = tmp_path / "output" / "x.html"
        with store.record_run("sleeper"):
            pass
    last = dash.last_runs()
    assert last["fakejob"]["status"] == "ok" and last["fakejob"]["report"].endswith("x.html")
    assert last["sleeper"]["status"] == "ok"


def test_page_shows_last_run_with_report_link(server, tmp_path):
    report = tmp_path / "output" / "fake-20260925-0101.html"
    report.write_text("x", encoding="utf-8")
    with Store(server.dashboard.db_path) as store, store.record_run("tasks.fakejob") as r:
        r.report_path = report
    page = call(server, "GET", "/")[2].decode("utf-8")
    assert 'href="/reports/fake-20260925-0101.html"' in page
    assert "never run" in page  # the other tasks
    assert "manual only" in page  # mailer is unscheduled


def test_status_endpoint(server):
    status, body = jcall(server, "GET", "/api/status")
    assert status == 200 and body["server"] is True and body["live"] == []


def test_dashboard_cmd_launches_the_module():
    cmd = (ROOT / "dashboard.cmd").read_text(encoding="utf-8")
    assert r".venv\Scripts\python" in cmd and "-m tasks.dashboard %*" in cmd


def test_unreadable_db_does_not_break_last_runs(dash, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(db, "Store", broken)
    assert dash.last_runs() == {}


# ---------------------------------------------------------------------------
# reload / restart survival, report attribution, report_url hardening
# ---------------------------------------------------------------------------

def _wait_log(d, run, needle, timeout=20):
    text, deadline = "", time.monotonic() + timeout
    while needle not in text and time.monotonic() < deadline:
        text = d.run_state(run, 0)["text"]
        time.sleep(0.1)
    assert needle in text, text
    return text


def _dead_pid():
    import subprocess as sp

    out = sp.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    return int(out.stdout)


def test_process_identity_guards_pid_reuse():
    me = db.process_identity(os.getpid())
    assert isinstance(me, int) and me == db.process_identity(os.getpid())
    assert db.process_identity(_dead_pid()) is None  # exited
    for bad in (0, -5, True, None, "123"):
        assert db.process_identity(bad) is None


def test_reload_reattaches_live_runs(server):
    status, body = jcall(server, "POST", "/api/run", {"task": "sleeper"})
    assert status == 201
    page = call(server, "GET", "/")[2].decode("utf-8")
    m = re.search(r'id="console" data-live="([^"]*)"', page)
    live = json.loads(m.group(1).replace("&quot;", '"'))
    assert live == [{"id": body["id"], "task": "sleeper"}]
    assert jcall(server, "GET", "/api/runs")[1] == {"live": live}
    assert "status === 404" in db.PAGE_JS and "reload the page" in db.PAGE_JS  # old tab stops polling


@pytest.mark.slow
def test_live_marker_is_written_and_removed(dash):
    run, _ = dash.start("brief")
    marker = json.loads(dash._marker("brief").read_text(encoding="utf-8"))
    assert marker["pid"] == run.pid and marker["created"] == db.process_identity(run.pid)
    assert marker["run_id"] == run.id and marker["log"] == str(run.log_path)
    assert run.done.wait(20)
    assert not dash._marker("brief").exists()


@pytest.mark.slow
def test_restart_conflict_adopt_stream_and_stop(make_dash, monkeypatch):
    a = make_dash()
    b_before = make_dash()  # already running when `a` launches: sees only the marker
    run, _ = a.start("sleeper")
    text = _wait_log(a, run, "child ")
    child = int(re.search(r"child (\d+)", text).group(1))

    with pytest.raises(db.Conflict):
        b_before.start("sleeper")  # marker: pid alive and creation time matches

    b = make_dash(token="new-token")  # a restart: adopts the orphan
    adopted = [r for r in b.runs.values() if r.task == "sleeper"]
    assert len(adopted) == 1 and adopted[0].adopted and adopted[0].live
    orphan = adopted[0]
    assert orphan.id == run.id and b.live_runs() == [{"id": run.id, "task": "sleeper"}]
    assert "child " in b.run_state(orphan)["text"]
    with pytest.raises(db.Conflict):
        b.start("sleeper")

    killed = []
    real_kill = db.kill_tree
    monkeypatch.setattr(db, "kill_tree", lambda proc: (killed.append(proc.pid), real_kill(proc)))
    b.stop(orphan.id)
    assert killed == [run.pid]
    assert orphan.done.wait(20)
    state = b.run_state(orphan)
    assert state["stopped"] and not state["exit_unknown"]
    deadline = time.monotonic() + 10
    while (_alive(child) or _alive(run.pid)) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child) and not _alive(run.pid)
    assert run.done.wait(20)
    assert not b._marker("sleeper").exists()
    b.start("sleeper")  # free again


@pytest.mark.slow
def test_adopted_run_that_finishes_reports_unknown_exit(make_dash):
    a = make_dash()
    run, _ = a.start("brief")
    b = make_dash()
    orphan = b.runs[run.id]
    assert orphan.done.wait(20)
    state = b.run_state(orphan)
    assert state["done"] and state["returncode"] is None and state["exit_unknown"]
    assert "brief" in state["text"]
    assert "exit code unknown (started before restart)" in db.PAGE_JS


def test_stale_markers_are_ignored_and_cleaned(dash):
    dash.live_dir.mkdir(parents=True, exist_ok=True)
    me = os.getpid()
    for task, pid, created in (("brief", _dead_pid(), 1), ("fakejob", me, 12345)):  # dead; live pid, wrong birth
        dash._marker(task).write_text(json.dumps(
            {"task": task, "run_id": "abc", "pid": pid, "created": created, "log": "x"}), encoding="utf-8")
    dash._marker("chatty").write_text("{not json", encoding="utf-8")
    fresh = db.Dashboard(dash.root, output_dir=dash.output_dir, db_path=dash.db_path,
                         schedule_path=dash.schedule_path, busy=lambda: [])
    assert fresh.runs == {}
    assert not any(dash.live_dir.glob("*.json"))
    run, _ = dash.start("fakejob")
    assert run.done.wait(20)


@pytest.mark.slow
def test_failing_task_is_not_credited_with_a_concurrent_report(dash, tmp_path):
    fails, _ = dash.start("fails")
    writer, _ = dash.start("writer")
    assert writer.done.wait(20) and fails.done.wait(20)
    assert (tmp_path / "output" / "writer-report.html").exists()
    assert fails.returncode == 3 and fails.report is None
    assert writer.report is None  # overlapped too: ambiguous, so no credit
    assert dash.run_state(fails)["report"] is None
    assert "no report credited" in db.PAGE_JS and "/reports/index.html" in db.PAGE_JS


@pytest.mark.slow
def test_overlap_is_remembered_after_the_other_run_ends(dash):
    writer, _ = dash.start("writer")
    brief, _ = dash.start("brief")  # writer is live when brief starts, then ends first
    assert writer.done.wait(20) and brief.done.wait(20)
    assert brief.report is None and writer.report is None


@pytest.mark.slow
def test_lone_successful_run_still_gets_the_fallback(dash, tmp_path):
    writer, _ = dash.start("writer")
    assert writer.done.wait(20)
    assert writer.report == tmp_path / "output" / "writer-report.html"


def test_foreign_runs_row_during_the_run_blocks_the_fallback(dash, tmp_path):
    (tmp_path / "output" / "x.html").write_text("x", encoding="utf-8")
    run = db.Run("r", "writer", [], tmp_path / "l.log", time.time() - 5, None, returncode=0)
    run.finished = time.time()
    assert dash._find_report(run) == tmp_path / "output" / "x.html"
    with Store(dash.db_path) as store, store.record_run("tasks.receipts"):
        pass  # e.g. the dispatcher ran another task meanwhile
    assert dash._find_report(run) is None


@pytest.mark.parametrize("path", [
    "\\\\192.0.2.1\\share\\r.html",
    "//192.0.2.1/share/r.html",
    "\\\\?\\UNC\\192.0.2.1\\share\\r.html",
])
def test_report_url_refuses_unc_without_touching_disk(dash, monkeypatch, path):
    touched = []
    real_resolve, real_realpath = Path.resolve, os.path.realpath

    def spy_resolve(self, *a, **kw):
        if "192.0.2.1" in str(self):
            touched.append(str(self))
        return real_resolve(self, *a, **kw)

    def spy_realpath(p, *a, **kw):
        if "192.0.2.1" in str(p):
            touched.append(str(p))
        return real_realpath(p, *a, **kw)

    monkeypatch.setattr(Path, "resolve", spy_resolve)
    monkeypatch.setattr(os.path, "realpath", spy_realpath)
    assert dash.report_url(path) is None
    assert touched == []


def test_report_url_lexical_rules(dash, tmp_path):
    inside = tmp_path / "output" / "r.html"
    inside.write_text("x", encoding="utf-8")
    assert dash.report_url(str(inside)) == "/reports/r.html"
    assert dash.report_url(str(tmp_path / "output" / ".." / "output" / "r.html")) == "/reports/r.html"
    (tmp_path / "outside.html").write_text("x", encoding="utf-8")
    assert dash.report_url(str(tmp_path / "output" / ".." / "outside.html")) is None
    assert dash.report_url(str(tmp_path / "output-evil" / "r.html")) is None  # sibling prefix, not a parent
    rel = os.path.relpath(inside, dash.root)  # relative paths are taken from the repo root
    assert dash.report_url(rel) == "/reports/r.html"


# ---------------------------------------------------------------------------
# marker lock (Windows file sharing: a reader blocks replace/unlink)
# ---------------------------------------------------------------------------

def test_marker_lock_excludes_other_dashboards(make_dash):
    a, b = make_dash(), make_dash(token="t2")
    with b._markers_locked():
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with a._markers_locked(timeout=0.3):
                pass
        assert time.monotonic() - t0 >= 0.3
    with a._markers_locked(timeout=0.3):  # free again once b lets go
        pass


@pytest.mark.slow
def test_start_waits_for_the_marker_lock_then_runs(make_dash):
    a, b = make_dash(), make_dash(token="t2")
    held = threading.Event()
    release = threading.Event()

    def hold():
        with b._markers_locked():
            held.set()
            release.wait(5)
    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    threading.Timer(0.5, release.set).start()
    run, _ = a.start("fakejob")
    assert run.done.wait(20) and run.returncode == 0
    t.join(5)


def test_marker_write_failure_kills_the_child_instead_of_orphaning_it(dash, monkeypatch):
    pids = []
    real_popen = subprocess.Popen

    def popen(*a, **kw):
        p = real_popen(*a, **kw)
        pids.append(p.pid)
        return p
    monkeypatch.setattr(db.subprocess, "Popen", popen)

    def fail(task, data):
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr(dash, "_write_marker", fail)
    with pytest.raises(PermissionError):
        dash.start("sleeper")
    assert pids and db.process_identity(pids[0]) is None  # killed, not orphaned
    assert dash.runs == {}
    assert not dash._marker("sleeper").exists()


def test_failed_cleanup_neither_masks_the_error_nor_blocks_the_task(dash, monkeypatch):
    real_popen, real_unlink = subprocess.Popen, Path.unlink

    def bad_popen(*a, **kw):
        raise OSError("simulated")

    def stuck_unlink(self, *a, **kw):
        if self.suffix == ".json":
            raise PermissionError(32, "being used by another process")
        return real_unlink(self, *a, **kw)
    monkeypatch.setattr(db.subprocess, "Popen", bad_popen)
    monkeypatch.setattr(Path, "unlink", stuck_unlink)
    with pytest.raises(db.BadRequest):  # the real error, not the cleanup's
        dash.start("fakejob")
    assert dash._marker("fakejob").exists()  # the placeholder was left behind

    monkeypatch.setattr(db.subprocess, "Popen", real_popen)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    run, _ = dash.start("fakejob")  # our own leftover placeholder doesn't block us
    assert run.done.wait(20) and run.returncode == 0


@pytest.mark.slow
def test_another_dashboards_placeholder_still_conflicts(make_dash):
    a = make_dash()
    a.live_dir.mkdir(parents=True, exist_ok=True)
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.3)
        a._marker("fakejob").write_text(json.dumps({
            "task": "fakejob", "pid": helper.pid, "created": db.process_identity(helper.pid),
            "placeholder": True}), encoding="utf-8")
        with pytest.raises(db.Conflict):
            a.start("fakejob")
    finally:
        helper.kill()
        helper.wait()


# ---------------------------------------------------------------------------
# #37: Errors view, tile badge, re-run, housekeeping
# ---------------------------------------------------------------------------

CRASHER = '''"""Crashes with a traceback."""
print("crasher starting", flush=True)
raise KeyError("total")
'''


def _seed_scheduled_failure(tmp_path, task="fakejob", text=None):
    """A dispatch_runs error row whose log sits in data/dispatch/logs."""
    from tasks import dispatch as dp

    logs = tmp_path / "data" / "dispatch" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{task}-20260925-010000.log"
    log.write_text(text or "scanning\nTraceback (most recent call last):\n  File \"x\", line 1\nValueError: sched <b>boom</b>\n",
                   encoding="utf-8")
    now = time.time() - 120
    stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(t))  # noqa: E731
    with Store(tmp_path / "data" / "lmsa.sqlite") as store:
        dp.ensure_table(store.conn)
        row = store.conn.execute(
            "INSERT INTO dispatch_runs(task, due_at, started, finished, outcome, reason, log_path) VALUES (?,?,?,?,?,?,?)",
            (task, stamp(now), stamp(now), stamp(now + 5), "error", "exit 1: ValueError: sched <b>boom</b>", str(log)),
        ).lastrowid
        store.conn.commit()
    return f"d{row}", log


@pytest.mark.slow
def test_errors_list_and_detail_for_all_three_sources(server, fake_root, tmp_path):
    (fake_root / "tasks" / "crasher.py").write_text(CRASHER, encoding="utf-8")
    status, j = jcall(server, "POST", "/api/run", {"task": "crasher"})
    assert status == 201
    poll_until_done(server, j["id"])
    sched_id, sched_log = _seed_scheduled_failure(tmp_path)
    with Store(tmp_path / "data" / "lmsa.sqlite") as store:
        term = store.conn.execute(
            "INSERT INTO runs(task, started, finished, status, summary_json) VALUES "
            "('tasks.mailer', '2026-01-02T03:04:05+00:00', '2026-01-02T03:04:09+00:00', 'error', ?)",
            (json.dumps({"error": "IMAP login refused <script>x</script>"}),),
        ).lastrowid
        store.conn.commit()

    status, headers, raw = call(server, "GET", "/api/errors")
    assert status == 200
    assert str(tmp_path) not in raw.decode("utf-8") and "log_path" not in raw.decode("utf-8")  # no paths leave
    data = json.loads(raw)
    rows = {e["id"]: e for e in data["entries"]}
    dash = rows[f"x{j['id']}"]
    assert (dash["task"], dash["source"], dash["outcome"]) == ("crasher", "dashboard", "exit 1")
    assert dash["reason"] == "exit 1: KeyError: 'total'"
    assert (rows[sched_id]["source"], rows[sched_id]["outcome"]) == ("scheduled", "error")
    assert (rows[f"r{term}"]["source"], rows[f"r{term}"]["task"]) == ("terminal", "mailer")
    assert [e["id"] for e in data["entries"]][:2] == [f"x{j['id']}", sched_id]  # newest first

    status, d = jcall(server, "GET", f"/api/errors/x{j['id']}")
    assert status == 200
    lines = d["log"]["text"].splitlines()
    (a, b), = d["log"]["spans"]
    assert lines[a].startswith("Traceback (most recent call last):") and lines[b] == "KeyError: 'total'"
    assert d["rerun"] == {"task": "crasher", "fields": {}}

    status, d = jcall(server, "GET", f"/api/errors/{sched_id}")
    assert status == 200 and "ValueError: sched <b>boom</b>" in d["log"]["text"] and d["log"]["spans"]

    status, d = jcall(server, "GET", f"/api/errors/r{term}")
    assert status == 200 and d["log"] is None and "terminal" in d["log_note"]
    assert d["summary"] == ["error: IMAP login refused <script>x</script>"]


def test_error_detail_show_all(server, tmp_path):
    body = "".join(f"line {i}\n" for i in range(1200)) + "Traceback (most recent call last):\n  File \"x\"\nOSError: disk\n"
    sched_id, _ = _seed_scheduled_failure(tmp_path, text=body)
    _, tail = jcall(server, "GET", f"/api/errors/{sched_id}")
    assert tail["log"]["truncated"] and tail["log"]["shown_lines"] == 500
    _, full = jcall(server, "GET", f"/api/errors/{sched_id}?all=1")
    assert not full["log"]["truncated"] and full["log"]["shown_lines"] == 1203


def test_log_path_outside_the_log_dirs_is_never_opened(server, tmp_path, monkeypatch):
    from tasks import dispatch as dp

    secret = tmp_path / "data" / "lmsa-secret.txt"
    secret.write_text("SECRET", encoding="utf-8")
    with Store(tmp_path / "data" / "lmsa.sqlite") as store:
        dp.ensure_table(store.conn)
        ids = []
        for path in (str(secret), r"\\evil-host\share\x.log", str(tmp_path / "data" / "dispatch" / "logs" / ".." / ".." / "lmsa-secret.txt")):
            ids.append(store.conn.execute(
                "INSERT INTO dispatch_runs(task, due_at, started, finished, outcome, reason, log_path) "
                "VALUES ('fakejob', 'x', '2026-09-25T01:00:00+00:00', '2026-09-25T01:00:05+00:00', 'error', 'exit 1: x', ?)",
                (path,),
            ).lastrowid)
        store.conn.commit()
    for i in ids:
        status, raw_headers, raw = call(server, "GET", f"/api/errors/d{i}")
        assert status == 200 and b"SECRET" not in raw
        assert json.loads(raw)["log"] is None


@pytest.mark.parametrize("path", ["/api/errors/..%2f..%2fx", "/api/errors/d99999", "/api/errors/zzz", "/api/errors/x"])
def test_unknown_error_ids_are_404(server, path):
    status, _ = jcall(server, "GET", path)
    assert status == 404


def test_errors_page_is_a_static_shell_with_csp_and_nav(server, tmp_path):
    _seed_scheduled_failure(tmp_path)
    status, headers, raw = call(server, "GET", "/errors")
    page = raw.decode("utf-8")
    assert status == 200
    nonce = re.search(r"script-src 'nonce-([^']+)'", headers["Content-Security-Policy"]).group(1)
    assert f'<script nonce="{nonce}">' in page
    assert 'content="test-token"' in page and 'href="/errors"' in page
    assert "boom" not in page  # failures arrive as JSON and are set with textContent
    assert ".innerHTML" not in page


def test_tile_badge_marks_tasks_whose_last_run_failed(server, tmp_path):
    _seed_scheduled_failure(tmp_path, task="fakejob")
    with Store(tmp_path / "data" / "lmsa.sqlite") as store:
        store.conn.execute("INSERT INTO runs(task, started, finished, status) VALUES "
                           "('brief', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:01+00:00', 'ok')")
        store.conn.commit()
    _, _, raw = call(server, "GET", "/")
    page = raw.decode("utf-8")
    fake = page[page.index('data-task="fakejob"'):]
    assert 'class="badge-err" href="/errors?task=fakejob"' in fake[: fake.index("</header>")]
    brief = page[page.index('data-task="brief"'):]
    assert "badge-err" not in brief[: brief.index("</header>")]


@pytest.mark.slow
def test_rerun_goes_through_the_launcher_and_its_confirm_rules(server, make_dash):
    status, j = jcall(server, "POST", "/api/run", {"task": "fails"})
    assert status == 201
    poll_until_done(server, j["id"])
    _, d = jcall(server, "GET", f"/api/errors/x{j['id']}")
    assert d["rerun"] == {"task": "fails", "fields": {}}
    status, again = jcall(server, "POST", "/api/run", {**d["rerun"], "extra": "", "confirm": False})
    assert status == 201
    server.dashboard.stop(again["id"])
    assert server.dashboard._rerun("nosuchtask") is None


def test_rerun_of_the_dispatcher_keeps_its_dry_run_default(tmp_path):
    d = db.Dashboard(ROOT, output_dir=tmp_path / "out", db_path=tmp_path / "data" / "lmsa.sqlite",
                     probe_server=lambda: True, loaded_models=lambda: [], busy=lambda: [], token="t")
    assert d._rerun("dispatch") == {"task": "dispatch", "fields": {"--dry-run": True}}
    with pytest.raises(db.NeedsConfirm):
        db.build_args(d.task("labeler"), {"--apply": True})  # the rule a re-run's /api/run call meets


@pytest.mark.slow
def test_run_history_is_capped_but_live_runs_stay(dash, monkeypatch):
    monkeypatch.setattr(db, "MAX_RUN_HISTORY", 2)
    done = []
    for _ in range(3):
        run, _ = dash.start("brief")
        run.done.wait(20)
        done.append(run.id)
    live, _ = dash.start("sleeper")
    assert live.id in dash.runs and len(dash.runs) == 2
    assert done[0] not in dash.runs and done[1] not in dash.runs and done[2] in dash.runs


@pytest.mark.slow
def test_dashboard_prunes_old_logs_but_not_live_ones(make_dash, tmp_path, monkeypatch):
    monkeypatch.delenv("LMSA_LOG_RETENTION_DAYS", raising=False)
    first = make_dash()
    live, _ = first.start("sleeper")
    _wait_log(first, live, "child")
    dash_logs, disp_logs = tmp_path / "data" / "dashboard" / "logs", tmp_path / "data" / "dispatch" / "logs"
    disp_logs.mkdir(parents=True)
    old_dash, old_disp = dash_logs / "old-1.log", disp_logs / "old-2.log"
    for p in (old_dash, old_disp):
        p.write_text("x", encoding="utf-8")
    month_ago = time.time() - 40 * 86400
    for p in (old_dash, old_disp, live.log_path):
        os.utime(p, (month_ago, month_ago))
    make_dash()  # a second dashboard starting up prunes; the first one's live log is in a marker
    assert not old_dash.exists() and not old_disp.exists()
    assert live.log_path.exists()


@pytest.mark.slow
def test_dashboard_failure_survives_a_restart_and_the_history_cap(make_dash, monkeypatch):
    first = make_dash()
    run, _ = first.start("fails")
    assert run.done.wait(20)
    rows = sqlite3.connect(first.db_path).execute(
        "SELECT id, task, argv_json, returncode, stopped, log_path FROM dashboard_runs").fetchall()
    assert rows == [(run.id, "fails", json.dumps(run.argv), 3, 0, str(run.log_path))]
    second = make_dash()  # a restart: nothing in memory
    assert run.id not in second.runs
    data = second.errors_data()
    (e,) = [x for x in data["entries"] if x["id"] == f"x{run.id}"]
    assert (e["source"], e["outcome"]) == ("dashboard", "exit 3")
    assert "fails" in second.failed_tasks()
    detail = second.error_detail(f"x{run.id}")
    assert detail["log"] is not None and detail["rerun"] == {"task": "fails", "fields": {}}
    # and in the same dashboard, after the cap has dropped it from memory
    monkeypatch.setattr(db, "MAX_RUN_HISTORY", 0)
    with first._lock:
        first._trim_history()
    assert run.id not in first.runs
    assert f"x{run.id}" in {x["id"] for x in first.errors_data()["entries"]}


def test_a_stalled_post_body_times_out_instead_of_holding_a_thread(server, monkeypatch):
    # The body is read before the auth checks; a client that never sends it
    # must not hold a handler thread forever.
    import socket
    monkeypatch.setattr(db.Handler, "timeout", 0.5)
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    s.sendall(b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 1000\r\n\r\n" + b"x" * 10)
    start = time.monotonic()
    try:
        s.recv(1024)  # server gives up and closes (or answers) once the timeout hits
    except OSError:
        pass
    s.close()
    assert time.monotonic() - start < 4


def test_handler_has_a_socket_timeout():
    assert db.Handler.timeout is not None and 0 < db.Handler.timeout <= 30


# ---------------------------------------------------------------------------
# Reports tab (#105): GET /api/reports, POST /api/reports/preview, /test-send,
# /api/reports/save, /api/reports/local. Dashboard._masters_or_error (and
# dashboard_setup's save_master_report/set_master_local) scope
# tasks.master_report's own tasks_dir/briefings_dir/schedule_path to *this*
# dashboard's own root, so a hermetic config/master_report.toml under
# fake_root/config is exactly what these tests read, write and validate
# against -- never the real repo's tracked file.
# ---------------------------------------------------------------------------

class _FakeReportsSMTP:
    instances: list["_FakeReportsSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.sent = []
        _FakeReportsSMTP.instances.append(self)

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
        raise AssertionError("must never open an SMTP connection")


REPORTS_MASTER_TOML = (
    '[[master]]\nname = "morning"\ntitle = "Morning Report"\nsend_at = "06:45"\n'
    'llm_intro = false\nmax_age_hours = 24\nskip_empty = true\n\n'
    '  [[master.section]]\n  task = "brief"\n  mode = "digest"\n  max_items = 5\n'
)


@pytest.fixture
def reports_env(fake_root, monkeypatch):
    from lmsa import mailer as mailer_mod

    monkeypatch.setenv("LMSA_IMAP_USER", "owner@example.test")
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", "test-app-pw-not-real")
    _FakeReportsSMTP.instances = []
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeReportsSMTP)
    (fake_root / "config").mkdir(exist_ok=True)
    (fake_root / "config" / "master_report.toml").write_text(REPORTS_MASTER_TOML, encoding="utf-8")
    # A real "briefing" task module -- bundleable_tasks() doesn't exclude
    # it (only master_report's own NOT_BUNDLEABLE names), so it's a real
    # bare task-kind catalog candidate here, same as in the real repo.
    (fake_root / "tasks" / "briefing.py").write_text('"""fake briefing task"""\n', encoding="utf-8")
    return fake_root


def test_reports_data_lists_the_fake_master_config(server, reports_env):
    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["to"] == ["LMSA_IMAP_USER"]  # v1: never a real address, even read-only
    assert morning["sections"] and all("mode" in s for s in morning["sections"])
    assert isinstance(body["base"], str) and body["base"]
    task_ids = {c["id"] for c in body["catalog"] if c["kind"] == "task"}
    assert "brief" in task_ids and "dashboard" not in task_ids  # NOT_BUNDLEABLE


def test_reports_catalog_excludes_the_bare_briefing_task_row(server, reports_env):
    """#105 review finding 5: a bare {kind:'task', id:'briefing'} row
    matches -- and, unchecked, would drop -- every briefing *section*
    (task == 'briefing'); the per-config briefing rows cover it instead."""
    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200
    assert not any(c["kind"] == "task" and c["id"] == "briefing" for c in body["catalog"])


def test_reports_catalog_carries_the_tasks_hinted_model(server, reports_env):
    """#105 review finding 3: "add to schedule" must offer the task's own
    model hint (SETUP={"model": ...}), not a hardcoded 'none' -- inbox_
    brief/job_tracker need "chat" or they'd be scheduled with no model."""
    (reports_env / "tasks" / "chatjob.py").write_text('"""fake chat task"""\nSETUP = {"model": "chat"}\n',
                                                      encoding="utf-8")
    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200
    entry = next(c for c in body["catalog"] if c["kind"] == "task" and c["id"] == "chatjob")
    assert entry["model"] == "chat" and entry["scheduled"] is False
    brief = next(c for c in body["catalog"] if c["kind"] == "task" and c["id"] == "brief")
    assert brief["model"] == "none"  # no SETUP hint: falls back to "none", not left unset
    assert isinstance(body["files"], list)


def test_reports_preview_renders_without_sending_or_a_model(server, reports_env, monkeypatch):
    from lmsa import mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "_smtp_class", _RefusingSMTP)  # a preview must never send
    status, body = jcall(server, "POST", "/api/reports/preview", {"name": "morning"})
    assert status == 200, body
    assert "<html" in body["html"].lower() and body["bytes"] > 0
    assert body["clip"] == 102_000 and body["clipped"] is False  # #119: nothing trimmed, Gmail's clip reported
    with Store(server.dashboard.db_path) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM sends").fetchone()[0] == 0


def test_reports_preview_unknown_master_is_bad_request(server, reports_env):
    status, body = jcall(server, "POST", "/api/reports/preview", {"name": "no-such-master"})
    assert status == 400 and "unknown master" in body["error"]


def test_reports_test_send_needs_the_token_and_a_confirm(server, reports_env):
    status, _ = jcall(server, "POST", "/api/reports/test-send", {"name": "morning", "confirm": True},
                      {db.TOKEN_HEADER: "wrong"})
    assert status == 403 and not _FakeReportsSMTP.instances
    status, body = jcall(server, "POST", "/api/reports/test-send", {"name": "morning"})
    assert status == 428 and not _FakeReportsSMTP.instances


def test_reports_test_send_bypasses_the_daily_guard_without_marking_the_real_slot(server, reports_env):
    first_status, first = jcall(server, "POST", "/api/reports/test-send", {"name": "morning", "confirm": True})
    assert first_status == 200 and first["status"] == "sent" and len(_FakeReportsSMTP.instances) == 1
    # A second test send the same day still goes out for real: "send test
    # now" is never blocked by an earlier test send.
    second_status, second = jcall(server, "POST", "/api/reports/test-send", {"name": "morning", "confirm": True})
    assert second_status == 200 and second["status"] == "sent" and len(_FakeReportsSMTP.instances) == 2
    # Neither test send claimed the scheduled slot: a real scheduled run
    # later must still see today as un-sent.
    with Store(server.dashboard.db_path) as store:
        rows = store.conn.execute("SELECT name, status FROM sends").fetchall()
    assert rows == [("master-report-morning-test", "sent")]


def test_reports_save_round_trips_a_valid_edit_including_section_order(server, reports_env):
    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200
    base = body["base"]
    masters = [{
        "name": "morning", "title": "Renamed Morning Report", "send_at": "07:00",
        "llm_intro": True, "max_age_hours": 12, "skip_empty": False,
        "sections": [
            {"task": "fakejob", "mode": "headline", "max_items": 3, "briefing": None},
            {"task": "brief", "mode": "digest", "max_items": 5, "briefing": None},
        ],
    }]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": base})
    assert status == 200, body

    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["title"] == "Renamed Morning Report" and morning["send_at"] == "07:00"
    assert morning["llm_intro"] is True and morning["max_age_hours"] == 12 and morning["skip_empty"] is False
    assert [s["task"] for s in morning["sections"]] == ["fakejob", "brief"]  # order preserved as sent
    # enabled/send were never part of this save -- still their tracked-file defaults
    assert morning["enabled"] is True and morning["send"] is False
    assert list(server.dashboard.setup.backup_dir.glob("master_report-*.toml"))  # a backup was kept

    # save again with the two sections swapped: the file reflects the new order
    masters[0]["sections"] = list(reversed(masters[0]["sections"]))
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": body["base"]})
    assert status == 200, body
    status, body = jcall(server, "GET", "/api/reports")
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert [s["task"] for s in morning["sections"]] == ["brief", "fakejob"]


def test_reports_save_rejects_an_invalid_edit_without_writing(server, reports_env):
    status, body = jcall(server, "GET", "/api/reports")
    base = body["base"]
    before = server.dashboard.setup.config_path("master_report").read_bytes()
    masters = [{
        "name": "morning", "title": "x", "send_at": "not-a-time", "llm_intro": False,
        "max_age_hours": 24, "skip_empty": True,
        "sections": [{"task": "brief", "mode": "digest", "max_items": 5, "briefing": None}],
    }]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": base})
    assert status == 400 and "send_at" in body["error"]
    assert server.dashboard.setup.config_path("master_report").read_bytes() == before


def test_reports_save_conflicts_on_a_stale_base(server, reports_env):
    status, body = jcall(server, "GET", "/api/reports")
    stale_base = body["base"]

    def master(title):
        return [{"name": "morning", "title": title, "send_at": "06:45", "llm_intro": False,
                "max_age_hours": 24, "skip_empty": True,
                "sections": [{"task": "brief", "mode": "digest", "max_items": 5, "briefing": None}]}]

    status, _ = jcall(server, "POST", "/api/reports/save", {"masters": master("First edit"), "base": stale_base})
    assert status == 200
    # stale_base was the sha *before* that save; the file has since changed
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": master("Second edit"), "base": stale_base})
    assert status == 409 and "changed since you loaded it" in body["error"]


def test_reports_local_enable_send_needs_confirm_and_never_touches_the_tracked_file(server, reports_env):
    tracked_before = server.dashboard.setup.config_path("master_report").read_bytes()
    status, body = jcall(server, "POST", "/api/reports/local", {"name": "morning", "values": {"send": True}})
    assert status == 428 and "config/local/master_report.toml" in body["error"]
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"send": True}, "confirm": True})
    assert status == 200, body

    status, body = jcall(server, "GET", "/api/reports")
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["send"] is True and morning["enabled"] is True  # untouched key keeps its default
    local_text = (reports_env / "config" / "local" / "master_report.toml").read_text(encoding="utf-8")
    assert "[master.morning]" in local_text and "send = true" in local_text
    assert server.dashboard.setup.config_path("master_report").read_bytes() == tracked_before


def test_reports_local_rejects_an_unknown_key_or_master(server, reports_env):
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"to": ["x@example.test"]}, "confirm": True})
    assert status == 400 and "only enabled, send" in body["error"]
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "no-such-master", "values": {"send": True}, "confirm": True})
    assert status == 400 and "unknown master" in body["error"]


def _master_payload(**over):
    m = {"name": "morning", "prev_name": "morning", "title": "Morning Report", "send_at": "06:45",
         "llm_intro": False, "max_age_hours": 24, "skip_empty": True,
         "sections": [{"task": "brief", "mode": "digest", "max_items": 5, "briefing": None}]}
    m.update(over)
    return m


def test_reports_local_endpoint_never_writes_without_confirm(server, reports_env):
    """#105 review finding 1: the endpoint itself is the backstop for the
    UI's armed-confirm flow -- an unconfirmed POST must never write,
    regardless of what the client sends (`send: true` included)."""
    before = server.dashboard.setup.config_path("master_report").read_bytes()
    local_path = reports_env / "config" / "local" / "master_report.toml"
    status, body = jcall(server, "POST", "/api/reports/local", {"name": "morning", "values": {"send": True}})
    assert status == 428
    assert not local_path.exists()
    assert server.dashboard.setup.config_path("master_report").read_bytes() == before
    status, body = jcall(server, "GET", "/api/reports")
    assert next(m for m in body["masters"] if m["name"] == "morning")["send"] is False


def test_reports_save_renaming_a_master_drops_its_orphaned_local_override(server, reports_env):
    """#105 review finding 2: renaming a master must not leave
    config/local/master_report.toml pointing at a name that no longer
    exists -- load_config would then fail entirely (masters: [])."""
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"send": True}, "confirm": True})
    assert status == 200, body
    local_path = reports_env / "config" / "local" / "master_report.toml"
    assert "[master.morning]" in local_path.read_text(encoding="utf-8")

    status, body = jcall(server, "GET", "/api/reports")
    masters = [_master_payload(name="am", prev_name="morning")]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": body["base"]})
    assert status == 200, body

    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None  # load_config still succeeds
    names = {m["name"] for m in body["masters"]}
    assert names == {"am"}
    local_text = local_path.read_text(encoding="utf-8")
    assert "[master.morning]" not in local_text and "[master.am]" not in local_text  # dropped, not renamed


def test_reports_save_rejected_by_validation_or_a_stale_base_keeps_local_overrides(server, reports_env):
    """#105 PR review: a save that comes back 400 or 409 must not have
    already dropped a local override -- it validates and base-checks first."""
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"enabled": False}, "confirm": True})
    assert status == 200, body
    local_path = reports_env / "config" / "local" / "master_report.toml"
    before = local_path.read_bytes()

    status, body = jcall(server, "GET", "/api/reports")
    base = body["base"]
    bad = [_master_payload(name="am", prev_name="morning", send_at="25:99")]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": bad, "base": base})
    assert status == 400, body
    assert local_path.read_bytes() == before

    stale = [_master_payload(name="am", prev_name="morning")]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": stale, "base": "0" * 64})
    assert status == 409, body
    assert local_path.read_bytes() == before


def test_generic_config_editor_refuses_a_rename_that_orphans_a_local_override(server, reports_env):
    """#105 PR review: the Setup tab's raw editor validates master_report.toml
    with this machine's local overrides, so it can't orphan one either."""
    from tasks import dashboard_setup as ds

    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"enabled": False}, "confirm": True})
    assert status == 200, body
    setup = server.dashboard.setup
    path = setup.config_path("master_report")
    current = path.read_bytes()
    renamed = current.decode("utf-8").replace('name = "morning"', 'name = "am"')
    with pytest.raises(ValueError, match="morning"):
        ds._config_save(setup, {"file": "master_report", "text": renamed, "base": ds._sha(current)})
    assert path.read_bytes() == current
    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None


def test_reports_save_deleting_a_master_drops_its_orphaned_local_override(server, reports_env):
    """#105 review finding 2, the delete half: same hazard, no rename.
    (parse_config needs at least one [[master]], so this deletes one of
    two rather than emptying the file.)"""
    two_masters = REPORTS_MASTER_TOML + (
        '\n[[master]]\nname = "evening"\ntitle = "Evening Report"\nsend_at = "20:00"\n'
        'llm_intro = false\nmax_age_hours = 24\nskip_empty = true\n\n'
        '  [[master.section]]\n  task = "brief"\n  mode = "digest"\n  max_items = 5\n'
    )
    server.dashboard.setup.config_path("master_report").write_text(two_masters, encoding="utf-8")
    status, body = jcall(server, "POST", "/api/reports/local",
                         {"name": "morning", "values": {"enabled": False}, "confirm": True})
    assert status == 200, body
    local_path = reports_env / "config" / "local" / "master_report.toml"
    assert "[master.morning]" in local_path.read_text(encoding="utf-8")

    status, body = jcall(server, "GET", "/api/reports")
    evening = next(m for m in body["masters"] if m["name"] == "evening")
    masters = [{
        "name": "evening", "prev_name": "evening", "title": evening["title"], "send_at": evening["send_at"],
        "llm_intro": evening["llm_intro"], "max_age_hours": evening["max_age_hours"],
        "skip_empty": evening["skip_empty"],
        "sections": [{"task": s["task"], "mode": s["mode"], "max_items": s["max_items"], "briefing": s["briefing"]}
                     for s in evening["sections"]],
    }]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": body["base"]})
    assert status == 200, body

    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None
    assert {m["name"] for m in body["masters"]} == {"evening"}
    assert "[master.morning]" not in local_path.read_text(encoding="utf-8")


def test_reports_save_carries_forward_tracked_enabled_send_to(server, reports_env):
    """#105 review finding 4: enabled/send/to aren't edited by the UI, but
    an explicit tracked-file value must survive an unrelated structural
    edit -- not get silently reset to the dataclass defaults."""
    disabled_toml = (
        '[[master]]\nname = "morning"\ntitle = "Morning Report"\nsend_at = "06:45"\n'
        'enabled = false\nsend = false\nto = ["someone@example.test"]\n'
        'llm_intro = false\nmax_age_hours = 24\nskip_empty = true\n\n'
        '  [[master.section]]\n  task = "brief"\n  mode = "digest"\n  max_items = 5\n'
    )
    config_path = server.dashboard.setup.config_path("master_report")
    config_path.write_text(disabled_toml, encoding="utf-8")

    status, body = jcall(server, "GET", "/api/reports")
    assert status == 200 and body["error"] is None
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["enabled"] is False  # sanity: the fixture is set up as intended

    masters = [_master_payload(title="Renamed but still disabled")]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": body["base"]})
    assert status == 200, body

    status, body = jcall(server, "GET", "/api/reports")
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["title"] == "Renamed but still disabled"
    assert morning["enabled"] is False and morning["send"] is False  # not silently re-enabled
    assert "enabled = false" in config_path.read_text(encoding="utf-8")


def test_reports_save_rejects_a_lone_surrogate_with_a_clean_400(server, reports_env):
    """#105 review finding 6: a lone UTF-16 surrogate (valid in a Python
    str and in JSON's \\uXXXX escapes, never valid UTF-8) must come back
    as a clean 400, not crash _config_save_locked and drop the connection."""
    status, body = jcall(server, "GET", "/api/reports")
    masters = [_master_payload(title="bad\ud800title")]
    status, body = jcall(server, "POST", "/api/reports/save", {"masters": masters, "base": body["base"]})
    assert status == 400 and "surrogate" in body["error"]
    # nothing was written
    status, body = jcall(server, "GET", "/api/reports")
    morning = next(m for m in body["masters"] if m["name"] == "morning")
    assert morning["title"] == "Morning Report"


# -- #97: remaining RST-on-close paths (drain before responding) --------------


def test_drain_reads_and_discards_up_to_the_limit():
    stub = SimpleNamespace(rfile=io.BytesIO(b"a" * 100))
    db.Handler._drain(stub, 40)
    assert stub.rfile.read() == b"a" * 60  # the other 60 bytes were left alone


def test_drain_stops_at_eof_without_raising():
    stub = SimpleNamespace(rfile=io.BytesIO(b"ab"))
    db.Handler._drain(stub, 1000)  # more than the socket actually has
    assert stub.rfile.read() == b""


def _handler_stub(headers, body):
    stub = SimpleNamespace(headers=headers, rfile=io.BytesIO(body))
    stub._drain = lambda limit: db.Handler._drain(stub, limit)
    stub._drain_chunked = lambda cap: db.Handler._drain_chunked(stub, cap)
    return stub


def test_drain_declared_body_caps_an_oversized_content_length():
    total = db.DRAIN_CAP + 500
    stub = _handler_stub({"Content-Length": str(total)}, b"z" * total)
    db.Handler._drain_declared_body(stub)
    assert len(stub.rfile.read()) == 500  # only DRAIN_CAP bytes were drained


def test_drain_chunked_stops_at_the_terminating_chunk_well_under_the_cap():
    # "0\r\n" is the terminating chunk; the final CRLF after it (and any
    # trailers) is deliberately left unconsumed -- see _drain_chunked.
    body = b"5\r\nhello\r\n0\r\n\r\ntrailing garbage should be left alone"
    stub = _handler_stub({}, body)
    db.Handler._drain_chunked(stub, db.DRAIN_CAP)
    assert stub.rfile.read() == b"\r\ntrailing garbage should be left alone"


def test_drain_chunked_is_capped_for_an_unterminated_body():
    body = b"z" * (db.DRAIN_CAP + 500)  # never a valid chunk-size line, no terminator
    stub = _handler_stub({}, body)
    db.Handler._drain_chunked(stub, db.DRAIN_CAP)
    # A garbled chunk-size line now falls back to a raw drain of the rest of
    # the budget (see _drain_chunked), so exactly `cap` bytes are consumed --
    # not just the first failed readline -- leaving only the 500-byte excess.
    assert len(stub.rfile.read()) == 500


def test_drain_chunked_garbled_body_never_reads_past_the_cap_even_with_embedded_crlfs():
    # A garbage byte that happens to contain \r or \n early on shouldn't let
    # the fallback drain skip past the cap either.
    body = (b"not hex\r\n garbage" * 20000)[: db.DRAIN_CAP + 777]
    stub = _handler_stub({}, body)
    db.Handler._drain_chunked(stub, db.DRAIN_CAP)
    assert len(stub.rfile.read()) == 777


def test_drain_chunked_never_reads_past_a_small_cap_when_garbage_follows_chunks():
    body = b"a\r\n" + b"x" * 10 + b"\r\n"  # 16 bytes of well-formed framing
    body = body * 5 + b"z" * 200               # 80 bytes of chunks, then garbage
    stub = _handler_stub({}, body)
    db.Handler._drain_chunked(stub, 100)
    assert len(stub.rfile.read()) == len(body) - 100


def test_get_with_an_unreadable_content_length_is_drained_to_the_cap():
    stub = _handler_stub({"Content-Length": "notanumber"}, b"y" * (db.DRAIN_CAP + 300))
    db.Handler._drain_declared_body(stub)
    assert len(stub.rfile.read()) == 300


def test_chunked_post_is_refused_with_411_and_never_runs(server):
    import socket
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    chunked_body = b"7\r\n{\"a\":1}\r\n0\r\n\r\n"
    start = time.monotonic()
    s.sendall(
        b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n" + chunked_body
    )
    resp = s.recv(4096)
    elapsed = time.monotonic() - start
    s.close()
    assert resp.split(b"\r\n", 1)[0].split(b" ", 2)[1] == b"411"
    assert server.dashboard.live_runs() == []  # never ran with `{}`
    # A short, well-formed chunked body must be recognized (terminating
    # chunk) and answered right away, not held for the drain cap/timeout --
    # regression for the hang this fix hit during development.
    assert elapsed < 2, f"a short chunked body took {elapsed:.2f}s to be refused"


def test_oversized_post_body_is_drained_before_the_413(server):
    big = b"x" * (db.MAX_BODY + 1000)  # over MAX_BODY, comfortably under DRAIN_CAP
    status, _headers, _raw = call(server, "POST", "/api/run", body=big)
    assert status == 413


def test_post_with_bad_content_length_drains_then_400(server):
    import socket
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    s.sendall(b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: notanumber\r\n\r\n")
    s.shutdown(socket.SHUT_WR)  # that's the whole request; let the server see EOF instead of hanging
    resp = s.recv(4096)
    s.close()
    assert resp.split(b"\r\n", 1)[0].split(b" ", 2)[1] == b"400"


def test_get_with_a_body_to_a_bad_host_is_drained_and_rejected(server):
    body = b"z" * (200 * 1024)  # the issue's own repro size
    import socket
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    s.sendall(f"GET / HTTP/1.1\r\nHost: evil.example\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    resp = s.recv(4096)
    s.close()
    assert resp.split(b"\r\n", 1)[0].split(b" ", 2)[1] == b"421"


def _raw_post_status(port, header_lines: bytes, body: bytes, *, recv_timeout: float = 5) -> bytes:
    """Send a raw POST over a fresh socket and return the response's status
    line. Raises whatever OSError the client sees if the connection is
    reset/aborted instead of returning a clean response -- the caller
    decides how to treat that."""
    import socket
    s = socket.create_connection(("127.0.0.1", port), timeout=recv_timeout)
    try:
        s.sendall(header_lines + body)
        resp = s.recv(4096)
    finally:
        s.close()
    return resp.split(b"\r\n", 1)[0]


def test_negative_content_length_is_drained_up_to_the_cap_no_rst(server):
    # Review repro (attempt 2): Content-Length: -5 followed by a 300 KB
    # body gave RST/ConnectionAborted in 35 of 60 trials, because the
    # `length < 0` branch drained 0 bytes (`min(length, DRAIN_CAP)` with a
    # negative length is negative, so the old `if length > 0 else 0` guard
    # picked 0). It must drain DRAIN_CAP, same as the bad-Content-Length
    # branch right above it. The body here is bigger than DRAIN_CAP so the
    # drain is always satisfied by data already sent -- no timeout wait.
    body = b"y" * (300 * 1024)
    trials = 30
    start = time.monotonic()
    for i in range(trials):
        try:
            status = _raw_post_status(
                server.port, b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: -5\r\n\r\n", body,
            )
        except OSError as exc:
            pytest.fail(f"trial {i}/{trials}: connection reset/aborted instead of a response ({exc!r})")
        assert status.split(b" ", 2)[1] == b"413", f"trial {i}/{trials}: {status!r}"
    # Sanity: none of the trials sat out a drain/socket timeout to get here.
    assert time.monotonic() - start < trials * 2


def test_garbled_chunked_body_is_drained_up_to_the_cap_no_rst(server):
    # Review repro (attempt 2): Transfer-Encoding: chunked with 264 KB of
    # non-hex bytes and no terminator gave RST in 15 of 150 trials, because
    # _drain_chunked gave up after the first failed chunk-size line instead
    # of falling back to a raw bounded drain. The body here is bigger than
    # DRAIN_CAP so the fallback drain is always satisfied by data already
    # sent -- no timeout wait.
    garbage = b"Z" * (264 * 1024)  # not a hex digit, no CRLF anywhere
    trials = 30
    start = time.monotonic()
    for i in range(trials):
        try:
            status = _raw_post_status(
                server.port,
                b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n",
                garbage,
            )
        except OSError as exc:
            pytest.fail(f"trial {i}/{trials}: connection reset/aborted instead of a response ({exc!r})")
        assert status.split(b" ", 2)[1] == b"411", f"trial {i}/{trials}: {status!r}"
    # Sanity: none of the trials sat out a drain/socket timeout to get here.
    assert time.monotonic() - start < trials * 2
