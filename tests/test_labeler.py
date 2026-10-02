"""Unit tests for tasks.labeler.

Gmail and LM Studio are both faked -- no live server, no mailbox (see
contracts.md). The write path (`Gmail.add_labels`/`remove_labels`) is faked
here at the task level; its own IMAP-command-level behaviour is covered in
tests/test_gmail_source.py.
"""

from __future__ import annotations

import sys

import pytest

from lmsa.sources.gmail import Message
from lmsa.store import Store
from tasks import labeler


def _msg(
    *,
    msg_id="1001",
    subject="Subj",
    from_addr="sender@example.com",
    labels=(),
    text="body text",
    web_url="https://mail.google.com/mail/u/0/#all/1",
) -> Message:
    return Message(
        msg_id=msg_id,
        thread_id="t1",
        labels=list(labels),
        flags=[],
        from_name="Sender",
        from_addr=from_addr,
        to=["me@example.com"],
        reply_to="",
        subject=subject,
        date=None,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text=text,
        attachments=[],
        web_url=web_url,
    )


class FakeGmail:
    """Stands in for lmsa.sources.gmail.Gmail. Records every write call and
    can be scripted to fail for a specific message id, to exercise
    per-message isolation."""

    def __init__(self, messages=(), fail_add=(), fail_remove=()):
        self.messages = list(messages)
        self.fail_add = set(fail_add)
        self.fail_remove = set(fail_remove)
        self.search_calls: list[tuple] = []
        self.add_calls: list[tuple] = []
        self.remove_calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, limit=200, **kwargs):
        self.search_calls.append((query, limit))
        return self.messages[:limit]

    def add_labels(self, msg_id, labels):
        self.add_calls.append((msg_id, list(labels)))
        if msg_id in self.fail_add:
            raise RuntimeError(f"simulated IMAP failure for {msg_id}")

    def remove_labels(self, msg_id, labels):
        self.remove_calls.append((msg_id, list(labels)))
        if msg_id in self.fail_remove:
            raise RuntimeError(f"simulated IMAP failure for {msg_id}")


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self, replies=None):
        self.calls: list[str] = []
        self._replies = list(replies or [])

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        if self._replies:
            reply = self._replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        return {"labels": []}


LABEL_DEFS = [
    labeler.LabelDef(name="Receipts", description="Purchase confirmations."),
    labeler.LabelDef(name="Travel", description="Flight/hotel confirmations."),
]
VALID_NAMES = {l.name for l in LABEL_DEFS}
SYSTEM = labeler.system_prompt(LABEL_DEFS)


# -- is_unlabeled / config parsing ---------------------------------------------


@pytest.mark.parametrize(
    "labels,expected",
    [
        ([], True),
        (["\\Inbox"], True),
        (["\\Inbox", "\\Important"], True),
        (["CATEGORY_PERSONAL"], True),
        (["\\Inbox", "CATEGORY_UPDATES"], True),
        (["Receipts"], False),
        (["\\Inbox", "Work"], False),
    ],
)
def test_is_unlabeled(labels, expected):
    assert labeler.is_unlabeled(_msg(labels=labels)) is expected


def test_load_labels_parses_config(tmp_path):
    path = tmp_path / "labels.toml"
    path.write_text(
        """
        [[label]]
        name = "Receipts"
        description = "Purchase confirmations."

        [[label]]
        name = "Travel"
        description = "Flight/hotel confirmations."
        """,
        encoding="utf-8",
    )
    defs = labeler.load_labels(path)
    assert [d.name for d in defs] == ["Receipts", "Travel"]
    assert defs[0].description == "Purchase confirmations."


def test_load_labels_missing_file_is_empty(tmp_path):
    assert labeler.load_labels(tmp_path / "nope.toml") == []


def test_load_labels_skips_entries_without_name(tmp_path):
    path = tmp_path / "labels.toml"
    path.write_text('[[label]]\ndescription = "no name"\n', encoding="utf-8")
    assert labeler.load_labels(path) == []


# -- classify --------------------------------------------------------------


def test_classify_proposes_configured_labels():
    lm = FakeLMStudio(replies=[{"labels": ["Receipts"]}])
    p = labeler.classify(lm, _msg(), valid_names=VALID_NAMES, system=SYSTEM)
    assert p.proposed == ["Receipts"]
    assert p.dropped == []
    assert p.error is None


def test_classify_drops_labels_outside_configured_set():
    lm = FakeLMStudio(replies=[{"labels": ["Receipts", "MadeUpLabel"]}])
    p = labeler.classify(lm, _msg(), valid_names=VALID_NAMES, system=SYSTEM)
    assert p.proposed == ["Receipts"]
    assert p.dropped == ["MadeUpLabel"]


def test_classify_skips_labels_already_present_idempotent():
    lm = FakeLMStudio(replies=[{"labels": ["Receipts", "Travel"]}])
    p = labeler.classify(
        lm, _msg(labels=["\\Inbox", "Receipts"]), valid_names=VALID_NAMES, system=SYSTEM
    )
    # Receipts already on the message -- never re-proposed; Travel is new.
    assert p.proposed == ["Travel"]


def test_classify_dedupes_repeated_label_in_one_response():
    lm = FakeLMStudio(replies=[{"labels": ["Receipts", "Receipts"]}])
    p = labeler.classify(lm, _msg(), valid_names=VALID_NAMES, system=SYSTEM)
    assert p.proposed == ["Receipts"]


def test_classify_records_model_error_without_raising():
    lm = FakeLMStudio(replies=[RuntimeError("server down")])
    p = labeler.classify(lm, _msg(), valid_names=VALID_NAMES, system=SYSTEM)
    assert p.error == "server down"
    assert p.proposed == []


# -- find_candidates ---------------------------------------------------------


def test_find_candidates_filters_unlabeled_and_caps_at_limit():
    messages = [
        _msg(msg_id="1", labels=["\\Inbox"]),
        _msg(msg_id="2", labels=["\\Inbox", "Work"]),  # already user-labeled
        _msg(msg_id="3", labels=[]),
    ]
    gmail = FakeGmail(messages=messages)
    candidates, truncated, found = labeler.find_candidates(gmail, days=3, limit=1)
    assert [c.msg_id for c in candidates] == ["1"]
    assert truncated is False
    assert found == 2  # "1" and "3" were unlabeled; --limit kept only one
    assert gmail.search_calls == [("newer_than:3d -in:sent -in:drafts", labeler.SEARCH_LIMIT)]


def test_find_candidates_truncated_when_scan_hits_cap():
    messages = [_msg(msg_id=str(i)) for i in range(labeler.SEARCH_LIMIT)]
    gmail = FakeGmail(messages=messages)
    _candidates, truncated, _found = labeler.find_candidates(gmail, days=3, limit=1000)
    assert truncated is True


# -- store: apply / undo ------------------------------------------------------


def test_apply_proposal_logs_intent_before_write_and_marks_done(tmp_path):
    gmail = FakeGmail()
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            p = labeler.Proposal(
                msg_id="42", subject="S", from_addr="a@example.com",
                web_url="", current_labels=[], proposed=["Receipts", "Travel"],
            )
            n = labeler.apply_proposal(store.conn, gmail, run.id, p)

        assert n == 2
        assert gmail.add_calls == [("42", ["Receipts", "Travel"])]
        rows = store.conn.execute(
            "SELECT msg_id, label, applied_at IS NOT NULL FROM labeler_applications "
            "WHERE run_id = ? ORDER BY label",
            (run.id,),
        ).fetchall()
        assert rows == [("42", "Receipts", 1), ("42", "Travel", 1)]


def test_apply_proposal_leaves_rows_unapplied_on_gmail_failure(tmp_path):
    gmail = FakeGmail(fail_add={"42"})
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            p = labeler.Proposal(
                msg_id="42", subject="S", from_addr="a@example.com",
                web_url="", current_labels=[], proposed=["Receipts"],
            )
            n = labeler.apply_proposal(store.conn, gmail, run.id, p)

        assert n == 0
        assert p.apply_error is not None and "simulated IMAP failure" in p.apply_error
        assert p.error is None  # not a classification error
        assert p.applied == [] and p.maybe_applied is False
        # the write provably didn't happen, so its intent row is gone and a
        # later --undo of this run can't strip a label added afterwards
        row = store.conn.execute(
            "SELECT applied_at FROM labeler_applications WHERE run_id = ?", (run.id,)
        ).fetchone()
        assert row is None


def test_unknown_outcome_failure_keeps_the_intent_row_for_undo(tmp_path):
    from lmsa.sources.gmail import LabelWriteError

    class DroppedGmail(FakeGmail):
        def add_labels(self, msg_id, labels):
            raise LabelWriteError("connection dropped mid-STORE", maybe_applied=True)

    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            p = labeler.Proposal(
                msg_id="43", subject="S", from_addr="a@example.com",
                web_url="", current_labels=[], proposed=["Receipts"],
            )
            labeler.apply_proposal(store.conn, DroppedGmail(), run.id, p)

        assert p.maybe_applied is True
        row = store.conn.execute(
            "SELECT applied_at FROM labeler_applications WHERE run_id = ?", (run.id,)
        ).fetchone()
        assert row is not None and row[0] is None  # undoable, marked unconfirmed


def test_apply_proposal_isolates_one_bad_message_from_the_rest(tmp_path):
    gmail = FakeGmail(fail_add={"bad"})
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            bad = labeler.Proposal(
                msg_id="bad", subject="", from_addr="", web_url="",
                current_labels=[], proposed=["Receipts"],
            )
            good = labeler.Proposal(
                msg_id="good", subject="", from_addr="", web_url="",
                current_labels=[], proposed=["Travel"],
            )
            applied = 0
            for p in (bad, good):
                applied += labeler.apply_proposal(store.conn, gmail, run.id, p)

        assert applied == 1  # only "good" landed
        assert bad.apply_error is not None
        assert good.apply_error is None
        assert good.applied == ["Travel"]
        assert gmail.add_calls == [("bad", ["Receipts"]), ("good", ["Travel"])]


def test_undo_run_removes_only_that_runs_applied_labels(tmp_path, monkeypatch):
    gmail = FakeGmail()
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run1:
            p1 = labeler.Proposal(
                msg_id="1", subject="", from_addr="", web_url="",
                current_labels=[], proposed=["Receipts"],
            )
            labeler.apply_proposal(store.conn, gmail, run1.id, p1)
        with store.record_run("tasks.labeler") as run2:
            p2 = labeler.Proposal(
                msg_id="2", subject="", from_addr="", web_url="",
                current_labels=[], proposed=["Travel"],
            )
            labeler.apply_proposal(store.conn, gmail, run2.id, p2)

        notes = labeler.undo_run(store.conn, run1.id)

        assert gmail.remove_calls == [("1", ["Receipts"])]  # never touches run2's message
        assert any("removed" in n for n in notes)
        undone = store.conn.execute(
            "SELECT msg_id, undone_at IS NOT NULL FROM labeler_applications ORDER BY msg_id"
        ).fetchall()
        assert undone == [("1", 1), ("2", 0)]


def test_undo_run_second_call_is_a_noop_nothing_left_to_undo(tmp_path, monkeypatch):
    gmail = FakeGmail()
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            p = labeler.Proposal(
                msg_id="1", subject="", from_addr="", web_url="",
                current_labels=[], proposed=["Receipts"],
            )
            labeler.apply_proposal(store.conn, gmail, run.id, p)

        labeler.undo_run(store.conn, run.id)
        gmail.remove_calls.clear()
        notes = labeler.undo_run(store.conn, run.id)

    assert gmail.remove_calls == []
    assert "nothing to undo" in notes[0]


def test_undo_run_isolates_per_message_failure(tmp_path, monkeypatch):
    apply_gmail = FakeGmail()  # both apply cleanly; the failure only hits undo
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            for msg_id in ("bad", "good"):
                p = labeler.Proposal(
                    msg_id=msg_id, subject="", from_addr="", web_url="",
                    current_labels=[], proposed=["Receipts"],
                )
                labeler.apply_proposal(store.conn, apply_gmail, run.id, p)

        undo_gmail = FakeGmail(fail_remove={"bad"})
        monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: undo_gmail)
        notes = labeler.undo_run(store.conn, run.id)

        undone = dict(
            store.conn.execute(
                "SELECT msg_id, undone_at IS NOT NULL FROM labeler_applications WHERE run_id = ?",
                (run.id,),
            ).fetchall()
        )
    assert undone == {"bad": 0, "good": 1}
    assert any("error" in n for n in notes)


def test_undo_run_never_removes_a_label_the_user_added_by_hand(tmp_path, monkeypatch):
    """The log only ever contains labels this tool itself applied -- a label
    a person added is never in `labeler_applications` in the first place, so
    undo structurally cannot touch it."""
    gmail = FakeGmail()
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            p = labeler.Proposal(
                msg_id="1", subject="", from_addr="", web_url="",
                current_labels=["UserAdded"], proposed=["Receipts"],
            )
            labeler.apply_proposal(store.conn, gmail, run.id, p)

        labeler.undo_run(store.conn, run.id)

    assert gmail.remove_calls == [("1", ["Receipts"])]  # "UserAdded" never appears


# -- main() end to end ---------------------------------------------------------


def test_main_dry_run_writes_report_and_touches_no_store_rows(tmp_path, monkeypatch):
    gmail = FakeGmail(messages=[_msg(msg_id="1", labels=[])])
    lm = FakeLMStudio(replies=[{"labels": ["Receipts"]}])
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", _write_labels_toml(tmp_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler"])

    labeler.main()

    assert gmail.add_calls == []  # dry run: never writes to Gmail
    with Store(tmp_path / "db.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM labeler_applications").fetchone()[0]
    assert count == 0
    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert "Receipts" in written.read_text(encoding="utf-8")


def test_main_apply_writes_gmail_and_logs_run_for_undo(tmp_path, monkeypatch, capsys):
    gmail = FakeGmail(messages=[_msg(msg_id="1", labels=[])])
    lm = FakeLMStudio(replies=[{"labels": ["Receipts"]}])
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", _write_labels_toml(tmp_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler", "--apply"])

    labeler.main()

    assert gmail.add_calls == [("1", ["Receipts"])]
    out = capsys.readouterr().out
    assert "run id for --undo:" in out

    with Store(tmp_path / "db.sqlite") as store:
        row = store.conn.execute(
            "SELECT msg_id, label, applied_at IS NOT NULL FROM labeler_applications"
        ).fetchone()
    assert row == ("1", "Receipts", 1)


def test_main_undo_end_to_end_removes_the_applied_label(tmp_path, monkeypatch, capsys):
    gmail = FakeGmail(messages=[_msg(msg_id="1", labels=[])])
    lm = FakeLMStudio(replies=[{"labels": ["Receipts"]}])
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", _write_labels_toml(tmp_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler", "--apply"])
    labeler.main()
    run_id = int(capsys.readouterr().out.strip().splitlines()[-1].split()[-1])

    monkeypatch.setattr(sys, "argv", ["labeler", "--undo", str(run_id)])
    labeler.main()

    assert gmail.remove_calls == [("1", ["Receipts"])]


def test_main_no_labels_configured_is_a_no_op(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(labeler, "LABELS_PATH", tmp_path / "nope.toml")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler"])

    labeler.main()

    assert "nothing to do" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def _write_labels_toml(tmp_path):
    path = tmp_path / "labels.toml"
    path.write_text(
        '[[label]]\nname = "Receipts"\ndescription = "Purchase confirmations."\n'
        '[[label]]\nname = "Travel"\ndescription = "Flight/hotel confirmations."\n',
        encoding="utf-8",
    )
    return path


# -- regression: Gmail system labels are never written (review finding 1) --------

SYSTEM_LABELS = ["\\Trash", "\\Spam", "\\Inbox", "CATEGORY_PROMOTIONS", "INBOX", "spam", "[Gmail]/Trash"]


@pytest.mark.parametrize("name", SYSTEM_LABELS)
def test_load_labels_rejects_gmail_system_labels(tmp_path, name):
    path = tmp_path / "labels.toml"
    # TOML literal strings, so a backslash is taken as-is
    path.write_text(
        f"[[label]]\nname = 'Receipts'\ndescription = 'ok'\n[[label]]\nname = '{name}'\ndescription = 'bad'\n",
        encoding="utf-8",
    )
    with pytest.raises(labeler.LabelConfigError) as info:
        labeler.load_labels(path)
    assert repr(name) in str(info.value) and "labels.toml" in str(info.value)


@pytest.mark.parametrize("name", SYSTEM_LABELS)
def test_classify_drops_system_labels_even_if_somehow_valid(name):
    """Belt and suspenders: even a system label that slipped into
    valid_names is dropped, never proposed."""
    lm = FakeLMStudio(replies=[{"labels": [name, "Receipts"]}])
    p = labeler.classify(lm, _msg(), valid_names=VALID_NAMES | {name}, system=SYSTEM)
    assert p.proposed == ["Receipts"]
    assert p.dropped == [name]


def test_main_with_system_label_in_config_fails_before_gmail_or_model(tmp_path, monkeypatch):
    path = tmp_path / "labels.toml"
    path.write_text("[[label]]\nname = '\\Trash'\ndescription = 'junk'\n", encoding="utf-8")
    gmail = FakeGmail(messages=[_msg(msg_id="1")])
    lm = FakeLMStudio(replies=[{"labels": ["\\Trash"]}])
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", path)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler", "--apply"])

    with pytest.raises(labeler.LabelConfigError, match="Trash"):
        labeler.main()
    assert gmail.search_calls == [] and gmail.add_calls == [] and lm.calls == []


# -- regression: undo integrity with the REAL Gmail write path (finding 2) ------


class RecordingIMAPClient:
    """Stands in for imapclient.IMAPClient under the real `Gmail` class,
    recording exact commands. `reselect_fail` makes the readonly re-select
    after a write raise; `store_error` makes the STORE itself raise."""

    def __init__(self, *, reselect_fail=False, store_error=None):
        self.calls: list[tuple] = []
        self.reselect_fail = reselect_fail
        self.store_error = store_error
        self.wrote = False

    def login(self, user, password):
        self.calls.append(("login",))

    def select_folder(self, folder, readonly=False):
        self.calls.append(("select_folder", folder, readonly))
        if readonly and self.wrote and self.reselect_fail:
            raise OSError("socket dropped on re-select")

    def search(self, criteria):
        self.calls.append(("search", criteria))
        return [7]

    def add_gmail_labels(self, uids, labels, silent=False):
        self.calls.append(("add_gmail_labels", list(uids), list(labels)))
        self.wrote = True
        if self.store_error:
            raise self.store_error

    def remove_gmail_labels(self, uids, labels, silent=False):
        self.calls.append(("remove_gmail_labels", list(uids), list(labels)))
        self.wrote = True

    def shutdown(self):
        self.calls.append(("shutdown",))

    def logout(self):
        self.calls.append(("logout",))


def _real_gmail(monkeypatch, client):
    from lmsa.sources import gmail as gmail_mod

    monkeypatch.setattr(gmail_mod.imapclient, "IMAPClient", lambda *a, **k: client)
    monkeypatch.setattr(gmail_mod.secrets, "resolve", lambda name: "invented-pass")
    monkeypatch.setenv("LMSA_IMAP_USER", "bot@example.com")
    return gmail_mod.Gmail


def _proposal(msg_id="123", proposed=("Receipts",), subject="S"):
    return labeler.Proposal(
        msg_id=msg_id, subject=subject, from_addr="a@example.com", web_url="",
        current_labels=[], proposed=list(proposed),
    )


def test_reselect_failure_after_store_still_marks_applied_and_undo_removes_it(tmp_path, monkeypatch):
    client = RecordingIMAPClient(reselect_fail=True)
    RealGmail = _real_gmail(monkeypatch, client)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            with RealGmail() as gmail:
                p = _proposal()
                n = labeler.apply_proposal(store.conn, gmail, run.id, p)
                # a second message in the same run fails cleanly: the session was dropped
                p2 = _proposal(msg_id="456")
                n2 = labeler.apply_proposal(store.conn, gmail, run.id, p2)

        assert n == 1 and p.applied == ["Receipts"] and p.apply_error is None
        assert n2 == 0 and "closed" in p2.apply_error and p2.maybe_applied is False
        # no command ever ran on the dropped (possibly writable) session
        assert client.calls[-1] == ("shutdown",)
        assert [c for c in client.calls if c[0] == "add_gmail_labels"] == [
            ("add_gmail_labels", [7], ["Receipts"])
        ]
        rows = dict(
            store.conn.execute(
                "SELECT msg_id, applied_at IS NOT NULL FROM labeler_applications WHERE run_id = ?",
                (run.id,),
            ).fetchall()
        )
        # 456 was never sent (the session was already dropped), so its intent
        # row is removed rather than left for --undo.
        assert rows == {"123": 1}

        undo_client = RecordingIMAPClient()
        _real_gmail(monkeypatch, undo_client)
        monkeypatch.setattr(labeler, "Gmail", RealGmail)
        notes = labeler.undo_run(store.conn, run.id)

    removes = [c for c in undo_client.calls if c[0] == "remove_gmail_labels"]
    assert removes == [("remove_gmail_labels", [7], ["Receipts"])]
    assert any(n.startswith("removed ['Receipts'] from 123") and "unconfirmed" not in n for n in notes)
    assert not any("456" in n for n in notes)


def test_store_dropped_mid_write_is_maybe_applied_and_undoable(tmp_path, monkeypatch):
    client = RecordingIMAPClient(store_error=OSError("connection reset mid-STORE"), reselect_fail=True)
    RealGmail = _real_gmail(monkeypatch, client)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        with store.record_run("tasks.labeler") as run:
            with RealGmail() as gmail:
                p = _proposal()
                n = labeler.apply_proposal(store.conn, gmail, run.id, p)

        assert n == 0 and p.applied == []
        # the STORE error surfaced, not the re-select error that followed it
        assert "connection reset mid-STORE" in p.apply_error
        assert p.maybe_applied is True

        undo_gmail = FakeGmail()
        monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: undo_gmail)
        notes = labeler.undo_run(store.conn, run.id)

    assert undo_gmail.remove_calls == [("123", ["Receipts"])]
    assert any("unconfirmed" in n for n in notes)


def test_undo_never_writes_a_system_label_from_an_old_log_row(tmp_path, monkeypatch):
    """A log row naming \\Inbox (e.g. written before system labels were
    refused) must not archive mail on undo: the real write path refuses it."""
    client = RecordingIMAPClient()
    RealGmail = _real_gmail(monkeypatch, client)
    monkeypatch.setattr(labeler, "Gmail", RealGmail)
    with Store(tmp_path / "db.sqlite") as store:
        labeler.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO labeler_applications (run_id, msg_id, label, logged_at, applied_at) "
            "VALUES (1, '123', ?, 'x', 'x')",
            ("\\Inbox",),
        )
        notes = labeler.undo_run(store.conn, 1)

    assert not any(c[0] in ("select_folder", "remove_gmail_labels") for c in client.calls)
    assert any("error" in n and "system label" in n for n in notes)


# -- regression: the report only claims what actually landed (finding 3) ------


def test_report_lists_only_successful_applies_and_separates_apply_failures():
    ok = _proposal(msg_id="1", subject="Landed one")
    ok.applied = ["Receipts"]
    bad = _proposal(msg_id="2", subject="Failed one", proposed=["Travel"])
    bad.apply_error = "LabelWriteError: STORE failed"
    bad.maybe_applied = True

    html = labeler.build_report([ok, bad], days=3, truncated=False, applied=True).render()

    applied_section = html.split("Applied labels", 1)[1].split("Apply failures", 1)[0]
    assert "Landed one" in applied_section and "Failed one" not in applied_section
    assert html.index("STORE failed") > html.index("Apply failures")
    assert "may have been applied" in html
    assert html.count("Classification errors") == 1  # the tile only, no section


def test_report_after_only_failed_apply_claims_nothing_applied():
    bad = _proposal(subject="Failed one")
    bad.apply_error = "RuntimeError: boom"

    html = labeler.build_report([bad], days=3, truncated=False, applied=True).render()

    assert "Applied labels" not in html
    assert "Apply failures" in html and "not applied" in html
    assert "nothing labeled" in html


def test_main_apply_failure_is_reported_as_apply_failure(tmp_path, monkeypatch):
    gmail = FakeGmail(messages=[_msg(msg_id="1", subject="Failed one")], fail_add={"1"})
    lm = FakeLMStudio(replies=[{"labels": ["Receipts"]}])
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", _write_labels_toml(tmp_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler", "--apply"])

    labeler.main()

    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Applied labels" not in html
    assert html.index("simulated IMAP failure") > html.index("Apply failures")


# -- regression: --limit cap is disclosed (finding 4) ---------------------------


def test_main_discloses_when_limit_caps_candidates(tmp_path, monkeypatch):
    gmail = FakeGmail(messages=[_msg(msg_id=str(i), labels=[]) for i in range(3)])
    lm = FakeLMStudio()
    monkeypatch.setattr(labeler, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(labeler, "LMStudio", lambda: lm)
    monkeypatch.setattr(labeler, "LABELS_PATH", _write_labels_toml(tmp_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(sys, "argv", ["labeler", "--limit", "1"])

    labeler.main()

    assert len(lm.calls) == 1
    [written] = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert "newest 1 of 3 unlabeled" in written.read_text(encoding="utf-8")


def test_report_no_limit_note_when_not_capped():
    html = labeler.build_report(
        [_proposal()], days=3, truncated=False, applied=False, unlabeled_found=1
    ).render()
    assert "limit reached" not in html
