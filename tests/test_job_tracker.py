"""tasks.job_tracker: key normalisation, the stage lattice's anti-regression
and reopening rules, per-item isolation, and the kanban report.

All fixtures use invented companies/people and example.com addresses; no
real mailbox or LM Studio server is touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import tasks.job_tracker as jt
from lmsa.digest import RunRow
from lmsa.sources.gmail import Message
from lmsa.store import Store
from gmail_fakes import MetaSearchMixin


def _days_ago(n: int) -> str:
    """An ISO date string `n` days before now -- keeps `main()` fixtures
    inside the task's default 90-day lookback window regardless of when
    tests run."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).date().isoformat()


def _msg(msg_id: str, subject: str = "Interview update", day: str = "2026-01-15", **overrides) -> Message:
    fields = dict(
        msg_id=msg_id,
        thread_id=f"t-{msg_id}",
        labels=[],
        flags=[],
        from_name="Riley Recruiter",
        from_addr="riley@example.com",
        to=["me@example.com"],
        reply_to="",
        subject=subject,
        date=datetime.fromisoformat(f"{day}T09:00:00+00:00"),
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text="Thanks for your time today.",
        attachments=[],
        web_url="",
    )
    fields.update(overrides)
    return Message(**fields)


# -- normalize_key ---------------------------------------------------------------


def test_normalize_company_drops_legal_suffix_and_folds_case():
    assert jt.normalize_company("Acme Robotics, Inc.") == "acme robotics"
    assert jt.normalize_company("acme robotics inc") == "acme robotics"
    assert jt.normalize_company("Acme Robotics LLC") == "acme robotics"


def test_normalize_role_drops_seniority_words():
    assert jt.normalize_role("Sr. Backend Engineer") == "backend engineer"
    assert jt.normalize_role("Senior Backend Engineer") == "backend engineer"
    assert jt.normalize_role("Backend Engineer") == "backend engineer"


def test_normalize_key_merges_near_duplicates():
    a = jt.normalize_key("Acme Robotics, Inc.", "Sr. Backend Engineer")
    b = jt.normalize_key("ACME ROBOTICS LLC", "Senior   Backend  Engineer")
    assert a == b


def test_normalize_key_handles_empty_strings():
    assert jt.normalize_key("", "") == "unknown company|unknown role"


# -- classify ----------------------------------------------------------------------


class FakeLM:
    def __init__(self, result=None, *, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append((prompt, schema, system, name))
        if self._error is not None:
            raise self._error
        return self._result


JOB_RESULT = {
    "is_job_related": True,
    "company": "Acme Robotics",
    "role": "Backend Engineer",
    "stage": "interview",
    "next_step": "Onsite panel Friday.",
    "contact": "Riley Recruiter <riley@example.com>",
}

NOT_JOB_RESULT = {
    "is_job_related": False,
    "company": "",
    "role": "",
    "stage": "outreach",
    "next_step": "",
    "contact": "",
}


def test_classify_returns_none_for_non_job_email():
    lm = FakeLM(NOT_JOB_RESULT)
    assert jt.classify(lm, _msg("m1"), system="sys") is None


def test_classify_builds_extraction():
    lm = FakeLM(JOB_RESULT)
    msg = _msg("m1", day="2026-03-01")
    result = jt.classify(lm, msg, system="sys")
    assert isinstance(result, jt.Extraction)
    assert result.company == "Acme Robotics"
    assert result.stage == "interview"
    assert result.email_date == msg.date


def test_classify_isolates_extract_exception():
    lm = FakeLM(error=RuntimeError("server returned empty content"))
    result = jt.classify(lm, _msg("m1"), system="sys")
    assert isinstance(result, jt.ItemError)
    assert "server returned empty content" in result.error


def test_classify_flags_message_with_no_date_without_calling_the_model():
    lm = FakeLM(JOB_RESULT)
    msg = _msg("m1", date=None)
    result = jt.classify(lm, msg, system="sys")
    assert isinstance(result, jt.ItemError)
    assert "date" in result.error
    assert lm.calls == []  # never even prompts the model


def test_classify_truncates_long_body_in_prompt():
    lm = FakeLM(JOB_RESULT)
    long_text = "x" * (jt.BODY_LIMIT + 500)
    msg = _msg("m1", text=long_text)
    jt.classify(lm, msg, system="sys")
    prompt = lm.calls[0][0]
    assert len(prompt) < len(long_text)
    assert "truncated" in prompt


# -- stage lattice: _merge_stage / apply_extraction -------------------------------


def _ex(**overrides) -> jt.Extraction:
    defaults = dict(
        msg_id="m1", company="Acme Robotics", role="Backend Engineer",
        stage="outreach", next_step="", contact="",
        email_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return jt.Extraction(**defaults)


def test_first_email_creates_the_row():
    job = jt.apply_extraction(None, _ex(stage="applied"))
    assert job.stage == "applied"
    assert job.stage_date == "2026-01-01"
    assert job.last_contact_date == "2026-01-01"
    assert job.first_seen == "2026-01-01"
    assert job.key == jt.normalize_key("Acme Robotics", "Backend Engineer")


def test_stage_advances_forward_along_the_lattice():
    job = jt.apply_extraction(None, _ex(stage="applied", email_date=datetime(2026, 1, 1, tzinfo=timezone.utc)))
    job = jt.apply_extraction(job, _ex(stage="screen", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc)))
    assert job.stage == "screen"


def test_stage_never_regresses_when_an_older_email_is_processed_after_a_newer_one():
    # Newer email (interview) processed first, then an older one (applied)
    # arrives late (e.g. a truncated/catch-up run finally sees it).
    job = jt.apply_extraction(
        None, _ex(stage="interview", email_date=datetime(2026, 1, 20, tzinfo=timezone.utc))
    )
    job = jt.apply_extraction(
        job, _ex(stage="applied", email_date=datetime(2026, 1, 5, tzinfo=timezone.utc))
    )
    assert job.stage == "interview"


def test_stage_result_is_order_independent_for_a_fixed_set_of_emails():
    e1 = _ex(stage="applied", email_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    e2 = _ex(stage="screen", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    e3 = _ex(stage="interview", email_date=datetime(2026, 1, 20, tzinfo=timezone.utc))

    forward = None
    for e in (e1, e2, e3):
        forward = jt.apply_extraction(forward, e)

    backward = None
    for e in (e3, e2, e1):
        backward = jt.apply_extraction(backward, e)

    assert forward.stage == backward.stage == "interview"


def test_rejected_is_terminal_and_wins_forward():
    job = jt.apply_extraction(
        None, _ex(stage="interview", email_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    )
    job = jt.apply_extraction(
        job, _ex(stage="rejected", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    assert job.stage == "rejected"


def test_stale_rejection_does_not_override_known_later_progress():
    # An out-of-order "rejected" dated BEFORE a later stage we already know
    # about must not clobber it -- it's stale, contradicted by newer info.
    job = jt.apply_extraction(
        None, _ex(stage="interview", email_date=datetime(2026, 1, 20, tzinfo=timezone.utc))
    )
    job = jt.apply_extraction(
        job, _ex(stage="rejected", email_date=datetime(2026, 1, 5, tzinfo=timezone.utc))
    )
    assert job.stage == "interview"


def test_terminal_row_reopens_on_a_genuinely_later_email():
    job = jt.apply_extraction(
        None, _ex(stage="rejected", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    job = jt.apply_extraction(
        job, _ex(stage="outreach", email_date=datetime(2026, 7, 1, tzinfo=timezone.utc))
    )
    assert job.stage == "outreach"
    assert job.stage_date == "2026-07-01"


def test_terminal_row_does_not_reopen_from_an_older_or_same_day_email():
    job = jt.apply_extraction(
        None, _ex(stage="rejected", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    same_day = jt.apply_extraction(
        job, _ex(stage="outreach", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    older = jt.apply_extraction(
        job, _ex(stage="outreach", email_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
    )
    assert same_day.stage == "rejected"
    assert older.stage == "rejected"


def test_display_fields_track_the_chronologically_freshest_email_even_without_a_stage_change():
    job = jt.apply_extraction(
        None,
        _ex(
            stage="interview", contact="Riley <riley@example.com>", next_step="Wait to hear back",
            email_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    # A later email in the SAME stage (no lattice advance) still updates
    # next_step/contact, since it's the freshest info about this key.
    job = jt.apply_extraction(
        job,
        _ex(
            stage="interview", contact="Jordan <jordan@example.com>", next_step="Onsite moved to Monday",
            email_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
        ),
    )
    assert job.stage == "interview"
    assert job.next_step == "Onsite moved to Monday"
    assert job.contact == "Jordan <jordan@example.com>"
    assert job.last_contact_date == "2026-01-15"


def test_older_email_processed_after_a_newer_one_does_not_overwrite_display_fields():
    job = jt.apply_extraction(
        None,
        _ex(
            stage="interview", contact="Jordan <jordan@example.com>", next_step="Onsite Monday",
            email_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
        ),
    )
    job = jt.apply_extraction(
        job,
        _ex(
            stage="applied", contact="Old Contact", next_step="stale info",
            email_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    assert job.contact == "Jordan <jordan@example.com>"
    assert job.next_step == "Onsite Monday"
    assert job.last_contact_date == "2026-01-15"


def test_missing_company_or_role_falls_back_to_existing_or_placeholder():
    job = jt.apply_extraction(None, _ex(company="", role=""))
    assert job.company == "Unknown company"
    assert job.role == "Unknown role"


def test_apply_extraction_is_idempotent_for_the_same_email():
    # A catch-up run's `before` bound can, in principle, hand this task the
    # same message twice (see find_candidates); re-folding the identical
    # extraction a second time must be a true no-op.
    ex = _ex(
        stage="interview", contact="Riley <riley@example.com>", next_step="Onsite Friday",
        email_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )
    once = jt.apply_extraction(None, ex)
    twice = jt.apply_extraction(once, ex)
    assert twice == once


def test_same_day_terminal_tie_is_decided_by_whichever_is_applied_first():
    first = jt.apply_extraction(
        None, _ex(stage="rejected", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    after_second = jt.apply_extraction(
        first, _ex(stage="closed", email_date=datetime(2026, 1, 10, tzinfo=timezone.utc))
    )
    # The first-applied terminal reading stands; the same-day second one is
    # too late to change it (see the module docstring's reopening section).
    assert after_second.stage == "rejected"


# -- staleness -----------------------------------------------------------------


def test_is_stale_true_after_the_threshold():
    job = jt.JobRecord(
        key="k", company="Acme", role="Eng", stage="applied", stage_date="2026-01-01",
        last_contact_date="2026-01-01", contact="", next_step="", first_seen="2026-01-01",
    )
    now = datetime(2026, 1, 20, tzinfo=timezone.utc)  # 19 days later
    assert jt.is_stale(job, now) is True


def test_is_stale_false_within_the_threshold():
    job = jt.JobRecord(
        key="k", company="Acme", role="Eng", stage="applied", stage_date="2026-01-01",
        last_contact_date="2026-01-01", contact="", next_step="", first_seen="2026-01-01",
    )
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)  # 9 days later
    assert jt.is_stale(job, now) is False


def test_is_stale_false_for_terminal_stages_regardless_of_age():
    job = jt.JobRecord(
        key="k", company="Acme", role="Eng", stage="rejected", stage_date="2026-01-01",
        last_contact_date="2026-01-01", contact="", next_step="", first_seen="2026-01-01",
    )
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert jt.is_stale(job, now) is False


# -- store ---------------------------------------------------------------------


def test_save_and_load_job_roundtrip(tmp_path):
    job = jt.JobRecord(
        key="acme|eng", company="Acme", role="Eng", stage="applied", stage_date="2026-01-01",
        last_contact_date="2026-01-01", contact="c", next_step="n", first_seen="2026-01-01",
    )
    with Store(tmp_path / "lmsa.sqlite") as store:
        jt._ensure_schema(store)
        jt.save_job(store, job)
        store.conn.commit()
        loaded = jt.load_job(store, "acme|eng")
    assert loaded == job


def test_save_job_is_idempotent_on_rerun(tmp_path):
    job = jt.JobRecord(
        key="acme|eng", company="Acme", role="Eng", stage="applied", stage_date="2026-01-01",
        last_contact_date="2026-01-01", contact="c", next_step="n", first_seen="2026-01-01",
    )
    with Store(tmp_path / "lmsa.sqlite") as store:
        jt._ensure_schema(store)
        jt.save_job(store, job)
        jt.save_job(store, job)
        store.conn.commit()
        count = store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 1


def test_is_applied_tracks_marked_msg_ids(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        jt._ensure_schema(store)
        assert jt._is_applied(store, "m1") is False
        jt._mark_applied(store, "m1")
        store.conn.commit()
        assert jt._is_applied(store, "m1") is True
        assert jt._is_applied(store, "m2") is False


def test_mark_applied_is_idempotent_on_rerun(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        jt._ensure_schema(store)
        jt._mark_applied(store, "m1")
        jt._mark_applied(store, "m1")  # same msg_id again -- must not raise
        store.conn.commit()
        count = store.conn.execute("SELECT COUNT(*) FROM job_tracker_applied").fetchone()[0]
    assert count == 1


# -- epoch-second search bounds -------------------------------------------------


# -- build_report ----------------------------------------------------------------


def _job(**overrides) -> jt.JobRecord:
    defaults = dict(
        key="acme|eng", company="Acme Robotics", role="Backend Engineer", stage="interview",
        stage_date="2026-01-10", last_contact_date="2026-01-10", contact="Riley <riley@example.com>",
        next_step="Onsite Friday", first_seen="2026-01-01",
    )
    defaults.update(overrides)
    return jt.JobRecord(**defaults)


def test_build_report_empty_state():
    r = jt.build_report([], [], now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    html = r.render()
    assert "all fresh" in html
    assert "Interview (" not in html


def test_build_report_groups_cards_by_stage():
    jobs = [_job(key="a", stage="interview"), _job(key="b", stage="applied", company="Widget Co")]
    r = jt.build_report(jobs, [], now=datetime(2026, 1, 11, tzinfo=timezone.utc))
    html = r.render()
    assert "Interview (1)" in html
    assert "Applied (1)" in html
    assert "Acme Robotics" in html
    assert "Widget Co" in html


def test_build_report_tags_stale_jobs():
    stale = _job(key="a", last_contact_date="2026-01-01")
    fresh = _job(key="b", last_contact_date="2026-01-19", company="Fresh Co")
    now = datetime(2026, 1, 20, tzinfo=timezone.utc)  # 19d since stale's contact, 1d since fresh's
    r = jt.build_report([stale, fresh], [], now=now)
    html = r.render()
    assert "1 stale" in html
    assert '<span class="tag">stale</span>' in html


def test_build_report_never_tags_terminal_jobs_as_stale():
    old_rejected = _job(key="a", stage="rejected", last_contact_date="2026-01-01")
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    r = jt.build_report([old_rejected], [], now=now)
    html = r.render()
    assert "all fresh" in html
    assert '<span class="tag">stale</span>' not in html


def test_build_report_escapes_untrusted_company_contact_and_next_step():
    job = _job(
        company="<script>alert(1)</script>",
        contact="<img onerror=x>",
        next_step="<b>bold</b> injected",
    )
    r = jt.build_report([job], [], now=datetime(2026, 1, 11, tzinfo=timezone.utc))
    html = r.render()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img onerror=x>" not in html
    assert "<b>bold</b>" not in html


def test_build_report_shows_classification_errors():
    err = jt.ItemError(msg_id="m1", subject="Weird email", error="server timeout")
    r = jt.build_report([], [err], now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    html = r.render()
    assert "Classification errors" in html
    assert "server timeout" in html


def _pending_window(pending: int):
    from lmsa.gmail_cursor import Window
    from lmsa.sources.gmail import MessageMeta

    metas = [MessageMeta(msg_id=str(i), internal_date=None, uid=i) for i in range(pending + 1)]
    return Window(candidates=metas, batch=metas[:1])


def test_build_report_shows_pending_note():
    r = jt.build_report([], [], now=datetime(2026, 1, 1, tzinfo=timezone.utc), window=_pending_window(3))
    assert "3 more matching message(s) are pending" in r.render()


# -- main() end to end -----------------------------------------------------------


class FakeGmail(MetaSearchMixin):
    """Filters by the `after:`/`before:` terms in the query (like real Gmail
    search) and returns newest-first, capped at `limit`."""

    def __init__(self, messages):
        self._messages = messages
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        # Real Gmail search accepts whole-second Unix epoch timestamps for
        # `after:`/`before:` (see `_epoch_bound`); this fake honours the
        # same granularity so it actually exercises the same-day, different
        # time-of-day cases the epoch-second fix is for.
        self.queries.append(query)
        since_epoch = int(query.split("after:")[1].split()[0])
        before_epoch = None
        if "before:" in query:
            before_epoch = int(query.split("before:")[1].split()[0])
        matches = [
            m for m in self._messages
            if m.date.timestamp() >= since_epoch
            and (before_epoch is None or m.date.timestamp() < before_epoch)
        ]
        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]


class FakeLMMain:
    def __init__(self, results: dict[str, dict | Exception]):
        self._results = results

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        for subject, result in self._results.items():
            if subject in prompt:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"no fake extract() result configured for prompt: {prompt[:80]!r}")


def test_main_end_to_end_is_idempotent_across_two_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    messages = [
        _msg("m1", subject="Recruiter outreach", day=_days_ago(5)),
        _msg("m2", subject="Newsletter", day=_days_ago(4)),
    ]
    fake_gmail = FakeGmail(messages)
    monkeypatch.setattr(jt, "Gmail", lambda: fake_gmail)

    fake_lm = FakeLMMain({
        "Recruiter outreach": dict(JOB_RESULT, stage="outreach"),
        "Newsletter": NOT_JOB_RESULT,
    })
    monkeypatch.setattr(jt, "LMStudio", lambda: fake_lm)

    jt.main([])
    jt.main([])  # re-run: cursor moved on, nothing new to fetch, still idempotent

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        company = store.conn.execute("SELECT company FROM jobs").fetchone()[0]
    assert count == 1  # the one job-related email, not duplicated, and the newsletter never got a row
    assert company == "Acme Robotics"

    written = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert written


def test_main_isolates_a_classification_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    messages = [_msg("m1", subject="Broken one", day=_days_ago(3))]
    fake_gmail = FakeGmail(messages)
    monkeypatch.setattr(jt, "Gmail", lambda: fake_gmail)

    fake_lm = FakeLMMain({"Broken one": RuntimeError("model unavailable")})
    monkeypatch.setattr(jt, "LMStudio", lambda: fake_lm)

    jt.main([])  # must not raise

    written = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    [html_path] = written
    html = html_path.read_text(encoding="utf-8")
    assert "model unavailable" in html

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 0


def test_main_stage_does_not_regress_when_an_older_email_lands_in_a_later_run(tmp_path, monkeypatch):
    """A stage must never regress when an older email is finally processed
    (here: retried after a model failure) in a later run than newer ones
    that already landed."""
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    messages = [
        _msg("m1", subject="Application received", day=_days_ago(30)),  # oldest -- fails in run 1
        _msg("m2", subject="Screen scheduled", day=_days_ago(15)),
        _msg("m3", subject="Interview scheduled", day=_days_ago(5)),    # newest
    ]
    fake_gmail = FakeGmail(messages)
    monkeypatch.setattr(jt, "Gmail", lambda: fake_gmail)

    fake_lm = FakeLMMain({
        "Application received": RuntimeError("model timed out"),
        "Screen scheduled": dict(JOB_RESULT, stage="screen"),
        "Interview scheduled": dict(JOB_RESULT, stage="interview"),
    })
    monkeypatch.setattr(jt, "LMStudio", lambda: fake_lm)

    jt.main([])  # run 1: m1 fails (queued for retry); m2, m3 land
    key = jt.normalize_key("Acme Robotics", "Backend Engineer")
    with Store(tmp_path / "lmsa.sqlite") as store:
        assert jt.load_job(store, key).stage == "interview"

    fake_lm._results["Application received"] = dict(JOB_RESULT, stage="applied")
    jt.main([])  # run 2: the retry lands the older "applied" email
    with Store(tmp_path / "lmsa.sqlite") as store:
        job = jt.load_job(store, key)
        assert jt._is_applied(store, "m1")
    assert job.stage == "interview"  # the older "applied" email must not regress it




# == digest() (#104: master report hook) ======================================

T0 = "2026-01-01T00:00:00+00:00"          # previous ok run -- the "last digest"
RUN_STARTED = "2026-01-20T00:00:00+00:00"  # current (latest ok) run


def _setup_jobs_and_runs(monkeypatch, tmp_path, jobs: list[jt.JobRecord]) -> RunRow:
    """A db with two `runs` rows (an earlier ok run at T0, the current one at
    RUN_STARTED) and `jobs` saved, then the RunRow digest() would be called
    with for the later run."""
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    with Store() as store:
        store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                           ("tasks.job_tracker", T0))
        store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                           ("tasks.job_tracker", RUN_STARTED))
        store.conn.commit()
        run_id = store.conn.execute(
            "SELECT id FROM runs WHERE started = ?", (RUN_STARTED,)
        ).fetchone()[0]
        jt._ensure_schema(store)
        for job in jobs:
            jt.save_job(store, job)
        store.conn.commit()
    return RunRow(id=run_id, task="tasks.job_tracker", started=RUN_STARTED, status="ok")


CHANGED = _job(key="a", company="Acme Robotics", role="Backend Engineer", stage="interview",
              stage_date="2026-01-15", last_contact_date="2026-01-15", first_seen="2026-01-01")
STALE = _job(key="b", company="Widget Co", role="PM", stage="applied",
            stage_date="2025-12-01", last_contact_date="2025-12-01", first_seen="2025-11-01")
QUIET = _job(key="c", company="Quiet Co", role="Ops", stage="screen",
            stage_date="2025-12-01", last_contact_date="2026-01-18", first_seen="2025-11-15")
OFFER = _job(key="d", company="Offer Co", role="Eng Lead", stage="offer",
            stage_date="2026-01-19", last_contact_date="2026-01-19", first_seen="2026-01-10")
REJECTED = _job(key="e", company="Rejected Co", role="Analyst", stage="rejected",
               stage_date="2025-11-01", last_contact_date="2025-11-01", first_seen="2025-10-01")

ALL_JOBS = [CHANGED, STALE, QUIET, OFFER, REJECTED]


def test_digest_tiles_match_build_report_counts(monkeypatch, tmp_path):
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, ALL_JOBS)

    d = jt.digest(run, max_items=10)

    assert {t.label: t.value for t in d.tiles} == {
        "Tracked": 5, "Active": 4, "Stale (14d+)": 1, "Offers": 1,
    }


def test_digest_orders_changed_since_last_run_before_stale(monkeypatch, tmp_path):
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, ALL_JOBS)

    d = jt.digest(run, max_items=10)

    # OFFER and CHANGED both moved after T0 (the previous ok run) -- newest
    # stage_date first; QUIET never appears (unchanged and not stale); STALE
    # comes last (unchanged, but 14d+ since last contact).
    assert [i.title for i in d.items] == ["Offer Co — Eng Lead", "Acme Robotics — Backend Engineer",
                                          "Widget Co — PM"]
    assert d.items[2].tone == "warn"  # stale overrides the stage's own tone
    assert d.items[0].tone == "hot"   # STAGE_TONES["offer"]
    assert d.items[0].url is None     # jobs carries no message reference


def test_digest_respects_max_items(monkeypatch, tmp_path):
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, ALL_JOBS)

    d = jt.digest(run, max_items=2)

    assert [i.title for i in d.items] == ["Offer Co — Eng Lead", "Acme Robotics — Backend Engineer"]


def test_digest_catches_a_stage_change_dated_the_same_day_as_the_previous_run(monkeypatch, tmp_path):
    """Regression (#104 review round 2): `stage_date` is a bare date, no
    time-of-day. Reviewer's repro -- previous ok run started
    2026-01-20T08:00Z; the job's `stage_date` is 2026-01-20 (the email
    arrived after that run and was only processed by the next one); the
    next run starts 2026-01-21. A strict `>` comparison against
    `previous_started`'s date never catches this: `stage_date` isn't before
    that date on any later run either, so the row would be dropped for
    good. It must show up as "changed" in this digest."""
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    same_day = _job(key="a", company="Acme Robotics", role="Backend Engineer", stage="screen",
                    stage_date="2026-01-20", last_contact_date="2026-01-20", first_seen="2026-01-20")
    with Store() as store:
        store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                           ("tasks.job_tracker", "2026-01-20T08:00:00+00:00"))
        store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                           ("tasks.job_tracker", "2026-01-21T08:00:00+00:00"))
        store.conn.commit()
        run_id = store.conn.execute(
            "SELECT id FROM runs WHERE started = ?", ("2026-01-21T08:00:00+00:00",)
        ).fetchone()[0]
        jt._ensure_schema(store)
        jt.save_job(store, same_day)
        store.conn.commit()
    run = RunRow(id=run_id, task="tasks.job_tracker", started="2026-01-21T08:00:00+00:00", status="ok")

    d = jt.digest(run, max_items=10)

    assert [i.title for i in d.items] == ["Acme Robotics — Backend Engineer"]


def test_digest_does_not_repeat_a_same_day_change_the_next_calendar_day(monkeypatch, tmp_path):
    """The flip side of the fix above: `>=` must not turn into "changed
    forever". A job whose stage hasn't moved since the run that first
    caught it (as in the test above) must stop being flagged once a later
    ok run's own date has moved past it."""
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    unchanged = _job(key="a", company="Acme Robotics", role="Backend Engineer", stage="screen",
                     stage_date="2026-01-20", last_contact_date="2026-01-22", first_seen="2026-01-20")
    with Store() as store:
        for started in ("2026-01-20T08:00:00+00:00", "2026-01-21T08:00:00+00:00", "2026-01-22T08:00:00+00:00"):
            store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                               ("tasks.job_tracker", started))
        store.conn.commit()
        run_id = store.conn.execute(
            "SELECT id FROM runs WHERE started = ?", ("2026-01-22T08:00:00+00:00",)
        ).fetchone()[0]
        jt._ensure_schema(store)
        jt.save_job(store, unchanged)
        store.conn.commit()
    run = RunRow(id=run_id, task="tasks.job_tracker", started="2026-01-22T08:00:00+00:00", status="ok")

    d = jt.digest(run, max_items=10)

    assert d.items == []  # not stale (2d since contact) and no longer "changed" -- already shown on the 21st


def test_digest_callouts_flag_offers_and_interviews(monkeypatch, tmp_path):
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, ALL_JOBS)

    d = jt.digest(run, max_items=10)

    assert d.callouts == [
        "1 offer(s): Offer Co",
        "1 interview(s) active: Acme Robotics",
    ]


def test_digest_with_no_previous_ok_run_treats_nothing_as_changed(monkeypatch, tmp_path):
    """First-ever digest: there is no prior ok run to compare against, so
    "changed since the last digest" is empty rather than everything --
    items fall back to stale-only."""
    db_path = tmp_path / "lmsa.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    with Store() as store:
        store.conn.execute("INSERT INTO runs (task, started, status) VALUES (?, ?, 'ok')",
                           ("tasks.job_tracker", RUN_STARTED))
        store.conn.commit()
        run_id = store.conn.execute("SELECT id FROM runs").fetchone()[0]
        jt._ensure_schema(store)
        for job in ALL_JOBS:
            jt.save_job(store, job)
        store.conn.commit()
    run = RunRow(id=run_id, task="tasks.job_tracker", started=RUN_STARTED, status="ok")

    d = jt.digest(run, max_items=10)

    assert [i.title for i in d.items] == ["Widget Co — PM"]  # stale only; nothing counts as "changed"


def test_digest_never_calls_the_model_or_gmail(monkeypatch, tmp_path):
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, ALL_JOBS)

    def refuse(*a, **kw):
        raise AssertionError("digest() must not touch the model or Gmail")

    monkeypatch.setattr(jt, "LMStudio", refuse)
    monkeypatch.setattr(jt, "Gmail", refuse)

    d = jt.digest(run, max_items=10)
    assert not d.is_empty()


def test_digest_escapes_hostile_company_name(monkeypatch, tmp_path):
    hostile = _job(key="x", company="<script>alert(1)</script>", role="Eng",
                   stage="interview", stage_date="2026-01-15", last_contact_date="2026-01-15",
                   first_seen="2026-01-01")
    run = _setup_jobs_and_runs(monkeypatch, tmp_path, [hostile])

    d = jt.digest(run, max_items=10)

    assert "<script>" in d.items[0].title  # digest() hands back plain, unescaped text ...
    html = jt.Card(title=d.items[0].title, body=d.items[0].blurb).render()
    assert "<script>" not in html and "&lt;script&gt;" in html  # ... Card.render() escapes it
