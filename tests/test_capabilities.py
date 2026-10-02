"""lmsa.capabilities / lmsa.capabilities_page / the dashboard's /capabilities
route (#67): a generated catalogue of everything tasks/ can do, never a
hand-maintained list.

Hermetic like test_dashboard_setup.py: an in-memory keyring (tests/conftest.py),
a fake `lms`, PowerShell and SearXNG, and a tmp root with its own tasks/ and
config/.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa import capabilities as caps_data
from lmsa.capabilities_page import render_capabilities_page, render_static_report
from tasks import dashboard as db
from tasks import dashboard_setup as ds

from test_dashboard import call, jcall
from test_dashboard_setup import FakeLms, FakePowerShell, FakeSearXNGFetcher
from lmsa.search import SearXNG

ROOT = Path(__file__).resolve().parents[1]

CAPABLE = '''"""Fake capable task.

Second line of the docstring -- must show up in full, not just the summary.
Also carries <script>alert(1)</script> to prove untrusted text is escaped.
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="max items")
    ap.add_argument("--apply", action="store_true", help="[writes] do the thing for real")
    ap.parse_args(argv)


SETUP = {"env": ["LMSA_TEST_FLAG"]}

if __name__ == "__main__":
    main()
'''

PLAIN = '''"""A plain task with no SETUP block."""


def main():
    pass


if __name__ == "__main__":
    main()
'''

ENV_EXAMPLE = "LMSA_TEST_FLAG=\n"

BRIEFING_OK = '''\
[briefing]
title = "Morning tech"
description = "Tech headlines and release notes."
interests = ["developer tools"]
unknown_future_key = "ignored, not an error"

[[sources.feeds]]
url = "https://news.ycombinator.com/rss"

[[sources.feeds]]
url = "https://lobste.rs/rss"

[[sources.search]]
query = "changelog release notes"
'''

BRIEFING_BAD = "not valid toml [[["


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "root"
    (r / "tasks").mkdir(parents=True)
    (r / "config").mkdir()
    (r / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (r / "tasks" / "capable.py").write_text(CAPABLE, encoding="utf-8")
    (r / "tasks" / "plain.py").write_text(PLAIN, encoding="utf-8")
    (r / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    (r / "config" / "schedule.toml").write_text("[dispatch]\n", encoding="utf-8")
    monkeypatch.delenv("LMSA_TEST_FLAG", raising=False)
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "data" / "lmsa.sqlite"))
    return r


@pytest.fixture
def fakes():
    return SimpleNamespace(lms=FakeLms(), ps=FakePowerShell(), server=True,
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
    d.setup = ds.Setup(d, lms=fakes.lms, searxng=fakes.searxng, powershell=fakes.ps,
                       send_slack=lambda text: True, version_of=lambda name: "99.0")
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


# ---------------------------------------------------------------------------
# lmsa.capabilities.build
# ---------------------------------------------------------------------------

def test_every_task_module_appears(dash):
    caps = caps_data.build(dash)
    names = {t.name for t in caps.tasks}
    assert names == {"capable", "plain"}


def test_a_new_dummy_module_appears_with_no_code_changes(dash, root):
    (root / "tasks" / "brand_new.py").write_text('"""Just added."""\n', encoding="utf-8")
    caps = caps_data.build(dash)
    assert "brand_new" in {t.name for t in caps.tasks}


def test_full_docstring_not_just_first_line(dash):
    caps = caps_data.build(dash)
    capable = next(t for t in caps.tasks if t.name == "capable")
    assert "Second line of the docstring" in capable.doc
    assert capable.summary == "Fake capable task."


def test_setup_needs_reflect_live_status(dash):
    caps = caps_data.build(dash)
    capable = next(t for t in caps.tasks if t.name == "capable")
    assert capable.setup_state == "missing"
    assert "LMSA_TEST_FLAG" in capable.setup_reason
    plain = next(t for t in caps.tasks if t.name == "plain")
    assert plain.setup_state == "none"


def test_setup_needs_go_ok_once_the_env_var_is_set(dash):
    dash.setup.write_env("LMSA_TEST_FLAG", "1")  # the real path: .env, not the raw process env
    caps = caps_data.build(dash)
    capable = next(t for t in caps.tasks if t.name == "capable")
    assert capable.setup_state == "ok"


def test_write_flag_is_detected(dash):
    caps = caps_data.build(dash)
    capable = next(t for t in caps.tasks if t.name == "capable")
    assert capable.writes is True
    plain = next(t for t in caps.tasks if t.name == "plain")
    assert plain.writes is False


def test_not_ready_summary_from_setup_checklist(dash):
    caps = caps_data.build(dash)
    ids = {n.id for n in caps.not_ready}
    assert "tool-capable" in ids  # LMSA_TEST_FLAG isn't set
    assert "gmail" in ids  # nothing configured on a fresh root


def test_no_briefings_directory_means_no_briefings(dash):
    caps = caps_data.build(dash)
    assert caps.briefings == ()


def test_briefings_are_listed_tolerant_of_unknown_keys(dash, root):
    d = root / "config" / "briefings"
    d.mkdir()
    (d / "morning.toml").write_text(BRIEFING_OK, encoding="utf-8")
    (d / "broken.toml").write_text(BRIEFING_BAD, encoding="utf-8")
    caps = caps_data.build(dash)
    by_name = {b.name: b for b in caps.briefings}
    assert by_name["morning"].title == "Morning tech"
    assert by_name["morning"].sources == 3
    assert "Tech headlines" in by_name["morning"].description
    assert by_name["broken"].warning  # unreadable, but doesn't blow up the page


# ---------------------------------------------------------------------------
# the dashboard route
# ---------------------------------------------------------------------------

def test_capabilities_page_lists_every_task_and_is_escaped(server):
    status, headers, raw = call(server, "GET", "/capabilities")
    page = raw.decode("utf-8")
    assert status == 200
    assert 'data-task="capable"' in page and 'data-task="plain"' in page
    assert "Second line of the docstring" in page
    assert "<script>alert(1)</script>" not in page  # escaped, not raw
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_capabilities_page_csp_nonce_matches_its_one_script(server):
    status, headers, raw = call(server, "GET", "/capabilities")
    page = raw.decode("utf-8")
    nonce = re.search(r"script-src 'nonce-([^']+)'", headers["Content-Security-Policy"]).group(1)
    assert page.count("<script") == 1
    assert f'<script nonce="{nonce}">' in page


def test_capabilities_linked_from_the_dashboard_header(server):
    _, _, raw = call(server, "GET", "/")
    assert 'href="/capabilities"' in raw.decode("utf-8")
    _, _, raw = call(server, "GET", "/capabilities")
    assert 'href="/capabilities"' in raw.decode("utf-8")


def test_capabilities_page_search_box_data_is_lowercased_for_client_filtering(server):
    _, _, raw = call(server, "GET", "/capabilities")
    page = raw.decode("utf-8")
    assert 'id="cap-q"' in page
    m = re.search(r'id="cap-capable"[^>]*data-cap-search="([^"]*)"', page)
    assert m and "capable" in m.group(1) and m.group(1) == m.group(1).lower()


# ---------------------------------------------------------------------------
# the static report (output/capabilities.html)
# ---------------------------------------------------------------------------

def test_static_report_has_no_script_at_all(dash):
    caps = caps_data.build(dash)
    html = render_static_report(caps).render()
    assert "<script" not in html
    assert "capable" in html and "plain" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_capabilities_cli_flag_writes_the_static_report(dash, monkeypatch, tmp_path):
    monkeypatch.setattr(db, "Dashboard", lambda: dash)
    db.main(["--capabilities"])
    out = dash.output_dir / "capabilities.html"
    assert out.is_file()
    assert "capable" in out.read_text(encoding="utf-8")
