"""Every scheduled task records its run (Store.record_run) so evening_recap
and the report index can see it. These drive each task's main() against
fakes (empty inputs) and check the `runs` row: status ok, a report path,
and a summary dict.

Deliberately not recorded: tasks.gmail_probe (a credentials diagnostic),
tasks.style_preview (the component gallery), tasks.index (rebuilds the
index itself), and file_sorter's --undo mode (replays a log).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tasks.file_sorter as file_sorter
import tasks.followup_radar as followup_radar
import tasks.github_digest as github_digest
import tasks.pretriage as pretriage
import tasks.server_logs as server_logs
import tasks.shootout as shootout
import tasks.unsubscribe as unsubscribe
from lmsa.store import Store


class _FakeLM:
    def __init__(self, out_dir: Path):
        self.cfg = SimpleNamespace(output_dir=out_dir, chat_model="fake-chat")

    def ensure_server(self):
        pass

    def models(self):
        return []


class _EmptyGmail:
    def __init__(self, cfg=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self):
        return "[Gmail]/Sent Mail"

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        return []


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    return tmp_path


def _run_row(env: Path, task: str):
    with Store(env / "lmsa.sqlite") as db:
        return db.conn.execute(
            "SELECT status, report_path, summary_json FROM runs WHERE task=?", (task,)
        ).fetchone()


def _assert_recorded(env: Path, task: str) -> dict:
    row = _run_row(env, task)
    assert row is not None, f"{task} did not record a run"
    status, report_path, summary_json = row
    assert status == "ok"
    assert report_path and Path(report_path).exists()
    summary = json.loads(summary_json)
    assert isinstance(summary, dict) and summary
    return summary


def test_followup_radar_records_its_run(env, monkeypatch):
    monkeypatch.setattr(followup_radar, "Gmail", _EmptyGmail)
    monkeypatch.setattr(followup_radar, "LMStudio", lambda: _FakeLM(env / "output"))
    followup_radar.main([])
    assert _assert_recorded(env, "tasks.followup_radar")["waiting"] == 0


def test_unsubscribe_records_its_run(env, monkeypatch):
    monkeypatch.setattr(unsubscribe, "Gmail", _EmptyGmail)
    monkeypatch.setattr(sys, "argv", ["unsubscribe"])
    unsubscribe.main()
    assert _assert_recorded(env, "tasks.unsubscribe")["scanned"] == 0


def test_file_sorter_records_its_run(env, monkeypatch):
    monkeypatch.setattr(file_sorter, "LMStudio", lambda: _FakeLM(env / "output"))
    monkeypatch.setattr(file_sorter, "build_plans", lambda lm, files, memo=None: ([], []))
    monkeypatch.setattr(sys, "argv", ["file_sorter", "--root", str(env)])
    file_sorter.main()
    assert _assert_recorded(env, "tasks.file_sorter")["applied"] is False


def test_github_digest_records_its_run(env, monkeypatch):
    monkeypatch.setattr(github_digest, "LMStudio", lambda: _FakeLM(env / "output"))
    monkeypatch.setattr(github_digest, "gather", lambda now: [])
    monkeypatch.setattr(github_digest, "summarize", lambda lm, data: "Nothing happened.")
    github_digest.main()
    assert _assert_recorded(env, "tasks.github_digest")["repos"] == 0


def test_pretriage_records_its_run(env, monkeypatch):
    monkeypatch.setattr(pretriage, "LMStudio", lambda: _FakeLM(env / "output"))
    monkeypatch.setattr(pretriage, "find_candidates", lambda owner, limit: ([], False))
    monkeypatch.setattr(sys, "argv", ["pretriage"])
    pretriage.main()
    assert _assert_recorded(env, "tasks.pretriage")["candidates"] == 0


def test_server_logs_records_its_run(env, monkeypatch):
    log = env / "server.log"
    log.write_text("", encoding="utf-8")
    monkeypatch.setattr(server_logs, "find_log_files", lambda days=None: [log])
    monkeypatch.setattr(sys, "argv", ["server_logs"])
    server_logs.main()
    assert "files" in _assert_recorded(env, "tasks.server_logs")


def test_shootout_records_its_run(env, monkeypatch):
    monkeypatch.setattr(shootout, "LMStudio", lambda: _FakeLM(env / "output"))
    monkeypatch.setattr(shootout, "load_prompts", lambda: [])
    monkeypatch.setattr(sys, "argv", ["shootout", "--no-judge"])
    shootout.main()
    assert _assert_recorded(env, "tasks.shootout")["attempts"] == 0


@pytest.mark.parametrize(
    "module",
    ["receipts", "file_sorter", "github_digest", "pretriage", "server_logs", "shootout", "gmail_probe", "index"],
)
def test_main_guard_goes_through_the_runner(module):
    # The runner is what sends the failure alert for an unattended run.
    source = Path(f"tasks/{module}.py").read_text(encoding="utf-8")
    guard = source[source.index('if __name__ == "__main__":'):]
    assert "from lmsa.runner import run" in guard
    assert "main()" not in guard
