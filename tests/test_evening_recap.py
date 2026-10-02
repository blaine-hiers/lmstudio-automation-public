import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from lmsa.store import Store
from tasks import evening_recap as er

EST = timezone(timedelta(hours=-5))


def _insert(store, *, task, started, status="ok", report_path=None, summary_json=None):
    store.conn.execute(
        "INSERT INTO runs(task, started, finished, status, report_path, summary_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (task, started, started, status, report_path, summary_json),
    )
    store.conn.commit()


# ---------------------------------------------------------------------------
# local date handling
# ---------------------------------------------------------------------------

def test_local_today_uses_given_now_and_tz():
    now = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)  # 25th 02:00 UTC
    assert er.local_today(now=now, tz=EST) == date(2026, 9, 24)  # 24th 21:00 EST
    assert er.local_today(now=now, tz=timezone.utc) == date(2026, 9, 25)


def test_run_local_date_crosses_midnight_with_tz():
    # 2am UTC on the 25th is still the 24th in EST (-5h).
    assert er.run_local_date("2026-09-25T02:00:00+00:00", tz=EST) == date(2026, 9, 24)
    assert er.run_local_date("2026-09-24T23:30:00+00:00", tz=EST) == date(2026, 9, 24)


def test_todays_runs_filters_by_local_date():
    rows = [
        {"started": "2026-09-24T12:00:00+00:00"},
        {"started": "2026-09-23T12:00:00+00:00"},
        {"started": "2026-09-25T02:00:00+00:00"},  # 24th in EST
    ]
    kept, unparseable = er.todays_runs(rows, date(2026, 9, 24), tz=EST)
    assert [r["started"] for r in kept] == [
        "2026-09-24T12:00:00+00:00",
        "2026-09-25T02:00:00+00:00",
    ]
    assert unparseable == []


def test_run_local_date_returns_none_for_unparseable_timestamp():
    assert er.run_local_date("not-a-timestamp") is None
    assert er.run_local_date(None) is None


def test_todays_runs_separates_unparseable_rows_instead_of_raising():
    rows = [
        {"task": "healthcheck", "started": "2026-09-24T12:00:00+00:00"},
        {"task": "flaky", "started": "garbage"},
        {"task": "flaky2", "started": ""},
    ]
    kept, unparseable = er.todays_runs(rows, date(2026, 9, 24), tz=EST)
    assert [r["task"] for r in kept] == ["healthcheck"]
    assert [r["task"] for r in unparseable] == ["flaky", "flaky2"]


# ---------------------------------------------------------------------------
# defensive summary parsing
# ---------------------------------------------------------------------------

def test_parse_summary_handles_missing_summary():
    assert er.parse_summary(None) == ([], [])
    assert er.parse_summary("") == ([], [])


def test_parse_summary_splits_numbers_scalars_and_skips_nested():
    tiles, lines = er.parse_summary(
        json.dumps({
            "models": 3,
            "chat_s": 1.25,
            "ok": True,
            "note": "fine",
            "missing": None,
            "nested_dict": {"a": 1},
            "nested_list": [1, 2, 3],
        })
    )
    by_label = {t.label: t.value for t in tiles}
    assert by_label == {"models": 3, "chat_s": 1.25}
    assert "ok: True" in lines
    assert "note: fine" in lines
    assert "missing: —" in lines
    assert not any("nested" in line for line in lines)


def test_parse_summary_handles_malformed_json():
    tiles, lines = er.parse_summary("{not valid json")
    assert tiles == []
    assert len(lines) == 1 and "unreadable" in lines[0]


def test_parse_summary_handles_non_object_json():
    tiles, lines = er.parse_summary("[1, 2, 3]")
    assert tiles == []
    assert len(lines) == 1 and "not an object" in lines[0]


def test_parse_summary_routes_non_finite_numbers_to_text():
    tiles, lines = er.parse_summary(json.dumps({"ratio": float("nan")}))
    assert tiles == []
    assert "ratio: nan" in lines


# ---------------------------------------------------------------------------
# report links
# ---------------------------------------------------------------------------

def test_report_link_relative_to_output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    report = out / "healthcheck-20260924-1800.html"
    report.write_text("x", encoding="utf-8")
    assert er.report_link(str(report), out) == "../../healthcheck-20260924-1800.html"


def test_report_link_falls_back_when_outside_output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere" / "old.html"
    assert er.report_link(str(elsewhere), out) == "old.html"


def test_report_link_none_when_no_report_path(tmp_path):
    assert er.report_link(None, tmp_path / "output") is None


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self):
        self.calls: list[str] = []

    def ensure_server(self) -> None:
        pass

    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        return "line one\nline two\nline three"


def test_main_rolls_up_todays_runs_and_isolates_failures(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(er, "LMStudio", FakeLMStudio)

    now = datetime.now().astimezone()
    today_iso = now.replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
    yesterday_iso = (now - timedelta(days=1)).replace(hour=9, minute=0).isoformat()

    with Store(db_path) as store:
        healthcheck_report = out_dir / "healthcheck-x.html"
        _insert(
            store,
            task="healthcheck",
            started=today_iso,
            status="ok",
            report_path=str(healthcheck_report),
            summary_json=json.dumps({"models": 3, "chat_s": 1.5}),
        )
        _insert(
            store,
            task="github_digest",
            started=today_iso,
            status="error",
            report_path=None,
            summary_json=None,
        )
        # Not today -- must be excluded.
        _insert(
            store,
            task="healthcheck",
            started=yesterday_iso,
            status="ok",
            report_path=None,
            summary_json=json.dumps({"models": 99}),
        )

    er.main()

    out = capsys.readouterr().out.strip()
    path = Path(out)
    assert path.exists()
    html = path.read_text(encoding="utf-8")

    assert "Evening recap" in html
    assert "healthcheck" in html and "github_digest" in html
    assert "1 failed" in html
    assert "line one" in html  # model wrap-up
    assert '"value">99' not in html  # yesterday's run excluded

    with Store(db_path) as store:
        row = store.conn.execute(
            "SELECT status, summary_json FROM runs WHERE task='evening_recap'"
        ).fetchone()
    assert row[0] == "ok"
    summary = json.loads(row[1])
    assert summary == {"runs": 2, "tasks": 2, "failed": 1, "unparseable": 0}


def test_main_with_no_runs_today_skips_model_call(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))

    fake = FakeLMStudio()
    monkeypatch.setattr(er, "LMStudio", lambda: fake)

    er.main()

    out = capsys.readouterr().out.strip()
    path = Path(out)
    html = path.read_text(encoding="utf-8")
    assert "No runs recorded today" in html
    assert fake.calls == []


# ---------------------------------------------------------------------------
# regressions: unparseable timestamps, and a failed wrap-up must not sink
# the run -- both raised in review of the first pass.
# ---------------------------------------------------------------------------

class BrokenEnsureServer(FakeLMStudio):
    def ensure_server(self) -> None:
        raise TimeoutError("server did not come up within 30s")

    def chat(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("chat() called after ensure_server() failed")


class BrokenChat(FakeLMStudio):
    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls.append(prompt)
        raise ConnectionError("connection reset")


def test_try_wrapup_returns_error_type_when_ensure_server_fails():
    wrapup, error = er.try_wrapup(BrokenEnsureServer(), {"healthcheck": []})
    assert wrapup is None
    assert error == "TimeoutError"


def test_try_wrapup_returns_error_type_when_chat_fails():
    wrapup, error = er.try_wrapup(BrokenChat(), {"healthcheck": []})
    assert wrapup is None
    assert error == "ConnectionError"


def test_main_records_run_ok_and_notes_wrapup_unavailable_when_chat_fails(
    monkeypatch, tmp_path, capsys
):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(er, "LMStudio", BrokenChat)

    with Store(db_path) as store:
        _insert(
            store,
            task="healthcheck",
            started=datetime.now().astimezone().isoformat(),
            status="ok",
            report_path=None,
            summary_json=json.dumps({"models": 3}),
        )

    er.main()  # must not raise

    out = capsys.readouterr().out.strip()
    path = Path(out)
    html = path.read_text(encoding="utf-8")
    assert "wrap-up unavailable: ConnectionError" in html
    assert "healthcheck" in html

    with Store(db_path) as store:
        row = store.conn.execute(
            "SELECT status, report_path FROM runs WHERE task='evening_recap'"
        ).fetchone()
    assert row[0] == "ok"
    assert row[1] is not None and Path(row[1]).exists()


def test_main_records_run_ok_when_ensure_server_fails(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(er, "LMStudio", BrokenEnsureServer)

    with Store(db_path) as store:
        _insert(
            store,
            task="healthcheck",
            started=datetime.now().astimezone().isoformat(),
            status="ok",
            report_path=None,
            summary_json=json.dumps({"models": 3}),
        )

    er.main()  # must not raise even though ensure_server() blew up

    out = capsys.readouterr().out.strip()
    path = Path(out)
    html = path.read_text(encoding="utf-8")
    assert "wrap-up unavailable: TimeoutError" in html
    assert "healthcheck" in html

    with Store(db_path) as store:
        status = store.conn.execute(
            "SELECT status FROM runs WHERE task='evening_recap'"
        ).fetchone()[0]
    assert status == "ok"


def test_main_excludes_unparseable_timestamp_and_still_records_ok(
    monkeypatch, tmp_path, capsys
):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(er, "LMStudio", FakeLMStudio)

    with Store(db_path) as store:
        _insert(
            store,
            task="healthcheck",
            started=datetime.now().astimezone().isoformat(),
            status="ok",
            report_path=None,
            summary_json=json.dumps({"models": 3}),
        )
        # A hand-edited or corrupted row -- must be excluded and shown,
        # not crash the recap.
        _insert(
            store,
            task="mystery_task",
            started="not-a-timestamp",
            status="ok",
            report_path=None,
            summary_json=None,
        )

    er.main()  # must not raise

    out = capsys.readouterr().out.strip()
    path = Path(out)
    html = path.read_text(encoding="utf-8")
    assert "mystery_task" in html
    assert "unparseable started timestamp" in html

    with Store(db_path) as store:
        row = store.conn.execute(
            "SELECT status, summary_json FROM runs WHERE task='evening_recap'"
        ).fetchone()
    assert row[0] == "ok"
    summary = json.loads(row[1])
    assert summary["unparseable"] == 1
    assert summary["runs"] == 1  # mystery_task excluded from the count
