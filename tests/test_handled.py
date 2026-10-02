"""Check off emails from reports and the dashboard (#61): local state,
report checkboxes, the dashboard endpoints, Gmail archive / mark read /
undo, and the Gmail filter suggestion. No network, no LM Studio."""

from __future__ import annotations

import base64
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest
from imapclient.exceptions import IMAPClientError

from lmsa import Card, Report, Store, handled
from lmsa.report import HANDLED_JS, HANDLED_JS_HASH, HANDLED_OFFLINE_TITLE
from lmsa.sources.gmail import Gmail, LabelWriteError, Message

from test_dashboard import call, dash, fake_root, jcall, make_dash, server  # noqa: F401
from test_gmail_source import FakeIMAPClient, _cfg


def _msg(msg_id="101", subject="Invoice due", sender="billing@example.com", **kw) -> Message:
    base = dict(
        msg_id=msg_id, thread_id="1", labels=[], flags=[], from_name="Billing", from_addr=sender,
        to=["me@example.com"], reply_to="", subject=subject, date=None, list_unsubscribe="",
        list_id="", auth_results="", text="Body.", attachments=[],
        web_url=f"https://mail.google.com/mail/u/0/#all/{msg_id}",
    )
    base.update(kw)
    return Message(**base)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as s:
        yield s


# -- local state ---------------------------------------------------------------


def test_handled_state_round_trip(store):
    a, b = _msg("101"), _msg("102", subject="Lunch")
    assert handled.seen(store.conn, "tasks.inbox_brief", 7, [a, b, _msg("not-a-gmail-id")]) == 2
    assert handled.is_known(store.conn, "101") and not handled.is_known(store.conn, "999")

    assert handled.mark(store.conn, "101")
    assert handled.mark(store.conn, "101")  # idempotent
    assert handled.handled_ids(store.conn) == {"101"}
    row = store.conn.execute(
        "SELECT subject, sender, source_task, folder, gmail_action, run_id FROM handled WHERE message_id='101'"
    ).fetchone()
    assert row == ("Invoice due", "billing@example.com", "inbox_brief", "[Gmail]/All Mail", None, None)

    open_, done = handled.split(store.conn, [a, b])
    assert open_ == [b] and done == [a]

    assert handled.unmark(store.conn, "101")
    assert handled.handled_ids(store.conn) == set()


def test_mark_refuses_a_message_no_report_showed(store):
    assert not handled.mark(store.conn, "555")
    assert not handled.mark(store.conn, "1 OR 1=1")
    assert handled.handled_ids(store.conn) == set()


def test_open_items_come_from_the_latest_run_of_each_task(store):
    handled.seen(store.conn, "inbox_brief", 1, [_msg("1", subject="old run")])
    handled.seen(store.conn, "inbox_brief", 2, [_msg("2", subject="new"), _msg("3", subject="done")])
    handled.seen(store.conn, "deadlines", 5, [_msg("2", subject="new"), _msg("4", subject="deadline")])
    handled.mark(store.conn, "3")
    items = handled.open_items(store.conn)
    assert sorted(i["msg_id"] for i in items) == ["2", "4"]


# -- reports -----------------------------------------------------------------


def test_report_opened_from_disk_has_disabled_checkboxes_and_no_token(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    r = Report("Brief")
    r.section("Act").cards([Card("Pay invoice", msg_id="101")]).handled_hidden(
        handled.hidden_cards([_msg("102", subject="Lunch")]))
    path = r.write(name="brief.html", out_dir=tmp_path)
    page = path.read_text(encoding="utf-8")

    boxes = re.findall(r"<input[^>]*data-msgid[^>]*>", page)
    assert len(boxes) == 2 and all(" disabled" in b for b in boxes)
    assert 'data-msgid="102" disabled checked' in page
    assert f'title="{HANDLED_OFFLINE_TITLE}"' in page
    assert "1 handled hidden" in page
    # the only script is the canonical one, and it carries no token
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", page, re.S)
    assert scripts == [HANDLED_JS]
    assert "X-LMSA-Token" in HANDLED_JS and not re.search(r"[A-Za-z0-9_-]{40,}", HANDLED_JS)
    assert "lmsa-token" not in page


def test_report_without_email_cards_has_no_script():
    r = Report("Plain")
    r.section("x").cards([Card("not an email")])
    assert "<script" not in r.render()


def test_untrusted_text_cannot_fake_a_checkbox():
    r = Report("x")
    r.section("x").cards([Card('<input data-msgid="1">', body='data-msgid="1"')])
    assert "<script" not in r.render()


def test_csp_hash_matches_the_script():
    digest = base64.b64encode(hashlib.sha256(HANDLED_JS.encode()).digest()).decode()
    assert HANDLED_JS_HASH == f"sha256-{digest}"


def test_inbox_brief_hides_handled_messages(tmp_path, monkeypatch):
    from tasks import inbox_brief as ib

    class FakeGmail:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search(self, query, limit=200, **kw):
            return [_msg("1", subject="Invoice due"), _msg("2", subject="Already handled")]

    class FakeLM:
        cfg = type("C", (), {"output_dir": tmp_path})()

        def ensure_server(self):
            pass

    asked = []

    def fake_classify(m, lm, memo=None):
        asked.append(m.msg_id)
        return {"bucket": "act", "why": "pay it", "action": "pay", "due_date": None}

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(ib, "Gmail", lambda cfg=None: FakeGmail())
    monkeypatch.setattr(ib, "LMStudio", lambda: FakeLM())
    monkeypatch.setattr(ib, "classify", fake_classify)
    monkeypatch.setattr("sys.argv", ["inbox_brief"])
    with Store() as s:
        handled.seen(s.conn, "inbox_brief", 1, [_msg("2", subject="Already handled")])
        handled.mark(s.conn, "2")

    ib.main()

    assert asked == ["1"]  # the handled one never reaches the model
    [page] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    html = page.read_text(encoding="utf-8")
    assert "1 handled hidden" in html
    assert 'data-msgid="1" disabled>' in html and 'data-msgid="2" disabled checked' in html
    with Store() as s:
        assert {i["msg_id"] for i in handled.open_items(s.conn)} == {"1"}


def test_followup_and_deadlines_split_by_their_own_keys(store):
    from tasks import followup_radar as fr

    c1 = fr.Candidate("t1", _msg("11", subject="Proposal"), "bob@example.com", 4.0)
    c2 = fr.Candidate("t2", _msg("12", subject="Quote"), "eve@example.com", 5.0)
    handled.seen(store.conn, "followup_radar", 1, [c2.sent_message])
    handled.mark(store.conn, "12")
    open_, done = handled.split(store.conn, [c1, c2], key=lambda c: c.sent_message.msg_id)
    assert open_ == [c1] and done == [c2]
    html = fr.build_report([], [], days=3, done=done).render()
    assert "1 handled hidden" in html and "eve@example.com" in html


# -- dashboard ------------------------------------------------------------------


def _seed(server, *ids):
    with Store(server.dashboard.db_path) as s:
        handled.seen(s.conn, "inbox_brief", 1, [_msg(i) for i in ids])


def _post_handled(server, body, headers):
    h = {"Content-Type": "application/json"}
    h.update(headers)
    return jcall(server, "POST", "/api/handled", body, h)


def test_handled_token_endpoint_and_post(server):
    _seed(server, "101")
    status, j = jcall(server, "GET", "/api/handled/token", headers={"Sec-Fetch-Site": "same-origin"})
    assert status == 200 and j["token"] == server.dashboard.handled_token != server.dashboard.token
    origin = f"http://127.0.0.1:{server.port}"
    status, j = _post_handled(server, {"msg_id": "101", "handled": True},
                              {"X-LMSA-Token": j["token"], "Origin": origin})
    assert status == 200 and j == {"msg_id": "101", "handled": True}
    with Store(server.dashboard.db_path) as s:
        assert handled.handled_ids(s.conn) == {"101"}
    status, j = jcall(server, "GET", "/api/inbox")
    assert status == 200 and j["open"] == [] and j["handled"][0]["msg_id"] == "101"


def test_handled_token_refused_cross_site(server):
    status, _ = jcall(server, "GET", "/api/handled/token", headers={"Sec-Fetch-Site": "cross-site"})
    assert status == 403


@pytest.mark.parametrize("headers", [
    {"X-LMSA-Token": ""},                                    # no token
    {"X-LMSA-Token": "wrong"},                               # wrong token
    {"X-LMSA-Token": "HANDLED", "Origin": "http://evil.example"},  # right token, wrong origin
    {"X-LMSA-Token": "HANDLED", "Origin": "null"},           # opaque (sandboxed/file) origin
])
def test_post_handled_refused_without_token_or_from_another_origin(server, headers):
    _seed(server, "101")
    headers = {k: (server.dashboard.handled_token if v == "HANDLED" else v) for k, v in headers.items()}
    status, _ = _post_handled(server, {"msg_id": "101", "handled": True}, headers)
    assert status == 403
    with Store(server.dashboard.db_path) as s:
        assert handled.handled_ids(s.conn) == set()


def test_post_handled_validates_the_body(server):
    _seed(server, "101")
    tok = {"X-LMSA-Token": server.dashboard.handled_token}
    assert _post_handled(server, {"msg_id": "999", "handled": True}, tok)[0] == 400  # not in the store
    assert _post_handled(server, {"msg_id": "abc", "handled": True}, tok)[0] == 400
    assert _post_handled(server, {"msg_id": "101", "handled": "yes"}, tok)[0] == 400


def test_report_token_cannot_launch_tasks(server):
    status, _ = jcall(server, "POST", "/api/run", {"task": "fakejob"},
                      {"X-LMSA-Token": server.dashboard.handled_token})
    assert status == 403 and not server.dashboard.runs


def test_reports_csp_allows_only_the_check_off_script(server, tmp_path):
    (tmp_path / "output" / "r.html").write_text("<p>hi</p>", encoding="utf-8")
    status, headers, _ = call(server, "GET", "/reports/r.html")
    csp = headers["Content-Security-Policy"]
    assert status == 200 and csp.startswith("sandbox")
    assert f"script-src '{HANDLED_JS_HASH}'" in csp and "unsafe-inline" not in csp
    assert "connect-src 'self'" in csp


def test_dashboard_page_has_the_inbox_panel(server):
    status, _h, raw = call(server, "GET", "/")
    page = raw.decode()
    assert status == 200 and 'id="inbox-sec"' in page and "also archive in Gmail" in page
    assert server.dashboard.handled_token not in page  # the page uses its own token


# -- Gmail archive / mark read / undo -------------------------------------------


class FlagIMAP(FakeIMAPClient):
    """Fake IMAP with per-uid X-GM-LABELS / FLAGS state the writes update."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.state = self.script.setdefault("state", {})

    def fetch(self, messages, data):
        self.calls.append(("fetch", list(messages), list(data)))
        return {u: {b"X-GM-LABELS": tuple(self.state[u]["labels"]), b"FLAGS": tuple(self.state[u]["flags"])}
                for u in messages if u in self.state}

    def _mut(self, name, messages, items, key, add):
        self.calls.append((name, list(messages), list(items), True))
        self._store_outcome()
        for u in messages:
            vals = self.state[u][key]
            for it in items:
                it = it if isinstance(it, bytes) else it.encode()
                (vals.add if add else vals.discard)(it)

    def add_gmail_labels(self, messages, labels, silent=False):
        self._mut("add_gmail_labels", messages, labels, "labels", True)

    def remove_gmail_labels(self, messages, labels, silent=False):
        self._mut("remove_gmail_labels", messages, labels, "labels", False)

    def add_flags(self, messages, flags, silent=False):
        self._mut("add_flags", messages, flags, "flags", True)

    def remove_flags(self, messages, flags, silent=False):
        self._mut("remove_flags", messages, flags, "flags", False)


WRITES = ("add_gmail_labels", "remove_gmail_labels", "add_flags", "remove_flags")


def _install(monkeypatch, script):
    created = {}

    def factory(host, port=None, ssl=True, timeout=None):
        created["client"] = FlagIMAP(host, port, script=script, ssl=ssl)
        return created["client"]

    monkeypatch.setattr("lmsa.sources.gmail.imapclient.IMAPClient", factory)
    monkeypatch.setattr("lmsa.sources.gmail.secrets.resolve", lambda name: "app-password-fake")
    return created


def _uid_state(inbox: bool, seen: bool) -> dict:
    return {"labels": {b"\\Inbox"} if inbox else {b"\\Sent"}, "flags": {b"\\Seen"} if seen else set()}


@pytest.mark.parametrize("method, call_name, arg, before", [
    ("archive", "remove_gmail_labels", ["\\Inbox"], (True, False)),
    ("unarchive", "add_gmail_labels", ["\\Inbox"], (False, False)),
    ("mark_read", "add_flags", [b"\\Seen"], (True, False)),
    ("mark_unread", "remove_flags", [b"\\Seen"], (True, True)),
])
def test_gmail_inbox_actions_touch_only_inbox_or_seen(monkeypatch, method, call_name, arg, before):
    created = _install(monkeypatch, {"search_uids": [77], "state": {77: _uid_state(*before)}})
    with Gmail(_cfg()) as gmail:
        result = getattr(gmail, method)(["555"])
    assert result == {"555": True}
    calls = created["client"].calls
    assert [c for c in calls if c[0] in WRITES] == [(call_name, [77], arg, True)]
    assert ("fetch", [77], ["X-GM-LABELS", "FLAGS"]) in calls  # state read before the write
    assert ("select_folder", "[Gmail]/All Mail", False) in calls
    assert [c for c in calls if c[0] == "select_folder"][-1] == ("select_folder", "[Gmail]/All Mail", True)


@pytest.mark.parametrize("method, before", [
    ("archive", (False, True)),      # already archived (e.g. your own sent mail)
    ("mark_read", (False, True)),    # already read
    ("unarchive", (True, False)),    # already back in the inbox
    ("mark_unread", (True, False)),  # already unread
])
def test_gmail_inbox_actions_skip_no_op_writes(monkeypatch, method, before):
    created = _install(monkeypatch, {"search_uids": [77], "state": {77: _uid_state(*before)}})
    with Gmail(_cfg()) as gmail:
        assert getattr(gmail, method)(["555"]) == {"555": False}
    assert not [c for c in created["client"].calls if c[0] in WRITES]


def test_gmail_archive_isolates_failures_per_message(monkeypatch):
    _install(monkeypatch, {"search_uids": [77], "store_error": IMAPClientError("NO"),
                           "state": {77: _uid_state(True, False)}})
    with Gmail(_cfg()) as gmail:
        result = gmail.archive(["555", "not-an-id"])
    assert isinstance(result["555"], LabelWriteError) and result["555"].maybe_applied is False
    assert isinstance(result["not-an-id"], ValueError)


def test_gmail_inbox_actions_take_no_label_argument():
    import inspect

    for name in ("archive", "unarchive", "mark_read", "mark_unread"):
        assert list(inspect.signature(getattr(Gmail, name)).parameters) == ["self", "msg_ids", "folder"]


class FakeGmail:
    """Mirrors `Gmail`'s inbox actions over per-message {inbox, seen} state:
    True when it changed something, False when already so, or the error."""

    def __init__(self, state=None, fail=None, default=(True, False)):
        self.state = state if state is not None else {}
        self.fail = fail or {}
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []
        self.log: list[str] = []  # the test's view of "what happened, in order"

    def _do(self, name, ids, key, value):
        self.calls.append((name, list(ids)))
        out = {}
        for i in ids:
            if i in self.fail:
                out[i] = self.fail[i]
                continue
            st = self.state.setdefault(i, {"inbox": self.default[0], "seen": self.default[1]})
            out[i] = st[key] != value
            st[key] = value
        return out

    def archive(self, ids):
        return self._do("archive", ids, "inbox", False)

    def unarchive(self, ids):
        return self._do("unarchive", ids, "inbox", True)

    def mark_read(self, ids):
        return self._do("mark_read", ids, "seen", True)

    def mark_unread(self, ids):
        return self._do("mark_unread", ids, "seen", False)


def _handled(store, *ids, sender="news@example.com"):
    handled.seen(store.conn, "inbox_brief", 1, [_msg(i, sender=sender) for i in ids])
    for i in ids:
        handled.mark(store.conn, i)


def _log(store, run_id):
    return store.conn.execute(
        "SELECT message_id, action, status FROM handled_gmail_log WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()


def test_apply_and_undo_restore_exactly_the_prior_state(store):
    # 1 in inbox & unread; 2 already archived & read (like your own sent mail);
    # 3 in inbox but already read.
    _handled(store, "1", "2", "3")
    before = {"1": {"inbox": True, "seen": False}, "2": {"inbox": False, "seen": True},
              "3": {"inbox": True, "seen": True}}
    gmail = FakeGmail(state={k: dict(v) for k, v in before.items()})
    arch = handled.apply(store.conn, gmail, 5, "archive", handled.pending(store.conn, "archive"))
    read = handled.apply(store.conn, gmail, 5, "mark_read", handled.pending(store.conn, "mark_read"))
    assert [o.status for o in arch] == ["done", "already", "done"]
    assert [o.status for o in read] == ["done", "already", "already"]
    # no-op writes are logged as 'already', which undo never touches
    assert _log(store, 5) == [("1", "archive", "done"), ("2", "archive", "already"), ("3", "archive", "done"),
                              ("1", "mark_read", "done"), ("2", "mark_read", "already"),
                              ("3", "mark_read", "already")]

    handled.undo(store.conn, gmail, 5)
    assert gmail.state == before
    assert handled.undo(store.conn, gmail, 5) == ["nothing to undo for run 5"]


def test_log_marks_each_write_right_after_it(store):
    _handled(store, "1", "2", "3")

    class Spy(FakeGmail):
        def archive(self, ids):
            # at the moment of each write: its own intent row exists, earlier ones are marked
            self.log.append(repr(_log(store, 7)))
            return super().archive(ids)

    gmail = Spy(fail={"2": LabelWriteError("refused", maybe_applied=False)})
    handled.apply(store.conn, gmail, 7, "archive", handled.pending(store.conn, "archive"))
    assert gmail.calls == [("archive", ["1"]), ("archive", ["2"]), ("archive", ["3"])]
    assert gmail.log == [
        repr([("1", "archive", "intent")]),
        repr([("1", "archive", "done"), ("2", "archive", "intent")]),
        repr([("1", "archive", "done"), ("2", "archive", "failed"), ("3", "archive", "intent")]),
    ]
    # the refused write is kept, marked failed, and never undone
    assert _log(store, 7) == [("1", "archive", "done"), ("2", "archive", "failed"), ("3", "archive", "done")]
    assert "refused" in store.conn.execute(
        "SELECT error FROM handled_gmail_log WHERE message_id='2'").fetchone()[0]
    gmail.calls.clear()
    handled.undo(store.conn, gmail, 7)
    assert gmail.calls == [("unarchive", ["1"]), ("unarchive", ["3"])]
    # a failed one is still pending for the next run
    assert [r["msg_id"] for r in handled.pending(store.conn, "archive")] == ["1", "2", "3"]


def test_undo_leaves_other_runs_alone(store):
    _handled(store, "1")
    gmail = FakeGmail()
    handled.apply(store.conn, gmail, 42, "archive", handled.pending(store.conn, "archive"))
    handled.apply(store.conn, gmail, 43, "mark_read", handled.pending(store.conn, "mark_read"))
    handled.undo(store.conn, gmail, 42)
    assert gmail.state["1"] == {"inbox": True, "seen": True}
    assert store.conn.execute("SELECT gmail_action FROM handled").fetchone() == ("mark_read",)


def test_unknown_outcome_is_marked_failed_and_not_undone(store):
    _handled(store, "1")
    gmail = FakeGmail(fail={"1": LabelWriteError("dropped", maybe_applied=True)})
    [o] = handled.apply(store.conn, gmail, 9, "archive", handled.pending(store.conn, "archive"))
    assert o.maybe_applied and o.status == "failed"
    assert _log(store, 9) == [("1", "archive", "failed")]
    assert handled.undo(store.conn, gmail, 9) == ["nothing to undo for run 9"]


class _Ctx:
    def __init__(self, gmail, opened):
        self.gmail, self.opened = gmail, opened

    def __call__(self, *a, **k):
        self.opened.append(1)
        return self

    def __enter__(self):
        return self.gmail

    def __exit__(self, *a):
        return False


def test_inbox_actions_cli_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    from tasks import inbox_actions

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    with Store() as s:
        _handled(s, "1", "2")
    gmail, opened = FakeGmail(), []
    monkeypatch.setattr("lmsa.sources.gmail.Gmail", _Ctx(gmail, opened))

    inbox_actions.main(["--archive-handled"])
    assert gmail.calls == [] and opened == []  # dry run never even connects
    [page] = [p for p in (tmp_path / "out").rglob("inbox-actions-*.html")]
    assert "Would archive" in page.read_text(encoding="utf-8")

    inbox_actions.main(["--archive-handled", "--apply"])
    assert gmail.calls == [("archive", ["1"]), ("archive", ["2"])]
    with Store() as s:
        run_id = s.conn.execute("SELECT MAX(run_id) FROM handled_gmail_log").fetchone()[0]
    assert f"run id for --undo: {run_id}" in capsys.readouterr().out
    inbox_actions.main(["--archive-handled", "--apply"])  # idempotent: nothing pending
    assert gmail.calls == [("archive", ["1"]), ("archive", ["2"])]

    inbox_actions.main(["--undo", str(run_id)])
    assert gmail.calls[-2:] == [("unarchive", ["1"]), ("unarchive", ["2"])]
    assert gmail.state == {"1": {"inbox": True, "seen": False}, "2": {"inbox": True, "seen": False}}


def test_inbox_actions_gives_no_undo_hint_when_nothing_was_written(tmp_path, monkeypatch, capsys):
    from tasks import inbox_actions

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    with Store() as s:
        _handled(s, "1")
    gmail = FakeGmail(state={"1": {"inbox": False, "seen": True}})  # already archived
    monkeypatch.setattr("lmsa.sources.gmail.Gmail", _Ctx(gmail, []))

    inbox_actions.main(["--archive-handled", "--apply"])
    out = capsys.readouterr().out
    assert "run id for --undo" not in out and "nothing to undo" in out
    [page] = [p for p in (tmp_path / "out").rglob("inbox-actions-*.html")]
    html = page.read_text(encoding="utf-8")
    assert "--undo" not in html and "already archived" in html and "nothing written" in html
    with Store() as s:
        assert s.conn.execute("SELECT status FROM handled_gmail_log").fetchall() == [("already",)]


def test_already_so_is_settled_and_does_not_starve_the_limit(store):
    # three already-read emails checked off first, then one unread one
    _handled(store, "1", "2", "3", "4")
    gmail = FakeGmail(state={i: {"inbox": True, "seen": i != "4"} for i in "1234"})
    first = handled.apply(store.conn, gmail, 1, "mark_read", handled.pending(store.conn, "mark_read", limit=2))
    assert [(o.msg_id, o.status) for o in first] == [("1", "already"), ("2", "already")]
    second = handled.apply(store.conn, gmail, 2, "mark_read", handled.pending(store.conn, "mark_read", limit=2))
    assert [(o.msg_id, o.status) for o in second] == [("3", "already"), ("4", "done")]
    assert gmail.state["4"]["seen"] is True
    assert handled.pending(store.conn, "mark_read") == []
    # undoing run 2 reverses only the real write, and makes that one pending again
    handled.undo(store.conn, gmail, 2)
    assert gmail.state["3"]["seen"] is True and gmail.state["4"]["seen"] is False
    assert [r["msg_id"] for r in handled.pending(store.conn, "mark_read")] == ["4"]


def test_gives_up_after_three_failures_without_counting_toward_the_limit(store):
    _handled(store, "1", "2")
    gone = LookupError("no message with X-GM-MSGID 1 in '[Gmail]/All Mail'")
    gmail = FakeGmail(fail={"1": gone})
    for run in (1, 2, 3):
        rows = handled.pending(store.conn, "archive", limit=1)
        assert [r["msg_id"] for r in rows] == ["1"]
        [o] = handled.apply(store.conn, gmail, run, "archive", rows)
        assert o.status == "failed"
    # 3 failures: no longer retried, reported instead; "2" gets the slot
    assert [r["msg_id"] for r in handled.gave_up(store.conn, "archive")] == ["1"]
    rows = handled.pending(store.conn, "archive", limit=1)
    assert [r["msg_id"] for r in rows] == ["2"]
    handled.apply(store.conn, gmail, 4, "archive", rows)
    assert handled.pending(store.conn, "archive") == []
    assert store.conn.execute(
        "SELECT COUNT(*) FROM handled_gmail_log WHERE message_id = '1'").fetchone() == (3,)
    # per action: mark_read hasn't failed on it
    assert [r["msg_id"] for r in handled.pending(store.conn, "mark_read")] == ["1", "2"]


def test_inbox_actions_reports_gave_up_emails(tmp_path, monkeypatch, capsys):
    from tasks import inbox_actions

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    with Store() as s:
        _handled(s, "1")
    gmail = FakeGmail(fail={"1": LookupError("gone")})
    monkeypatch.setattr("lmsa.sources.gmail.Gmail", _Ctx(gmail, []))
    for _ in range(4):
        inbox_actions.main(["--archive-handled", "--apply"])
    assert gmail.calls == [("archive", ["1"])] * 3
    out = capsys.readouterr().out
    assert "archive: 1 email(s) gave up after 3 failures -- check Gmail" in out
    page = max((tmp_path / "out").rglob("inbox-actions-*.html"), key=lambda p: p.stat().st_mtime_ns)
    assert "gave up after 3 failures" in page.read_text(encoding="utf-8")


def test_inbox_actions_unknown_outcome_is_not_reported_as_nothing_written(tmp_path, monkeypatch, capsys):
    from tasks import inbox_actions

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    with Store() as s:
        _handled(s, "1", "2")
    gmail = FakeGmail(fail={i: LabelWriteError("dropped", maybe_applied=True) for i in "12"})
    monkeypatch.setattr("lmsa.sources.gmail.Gmail", _Ctx(gmail, []))

    inbox_actions.main(["--archive-handled", "--apply"])
    out = capsys.readouterr().out
    assert "nothing was written" not in out and "run id for --undo" not in out
    assert "2 write(s) had an unknown outcome -- check Gmail; not covered by --undo" in out
    [page] = [p for p in (tmp_path / "out").rglob("inbox-actions-*.html")]
    assert "nothing written" not in page.read_text(encoding="utf-8")
    # undo behaviour unchanged: unknown-outcome rows are never undone
    with Store() as s:
        run_id = s.conn.execute("SELECT MAX(run_id) FROM handled_gmail_log").fetchone()[0]
        assert handled.undo(s.conn, gmail, run_id) == [f"nothing to undo for run {run_id}"]


# -- Gmail filter suggestion ------------------------------------------------------


def test_filter_suggestion_after_three_archives_in_thirty_days(store):
    _handled(store, "1", "2", sender="News@Example.com")
    _handled(store, "3", sender="news@example.com")
    _handled(store, "4", "5", sender="rare@example.com")
    handled.apply(store.conn, FakeGmail(), 1, "archive", handled.pending(store.conn, "archive"))
    assert handled.filter_suggestions(store.conn, own=()) == [("news@example.com", 3)]
    later = datetime.now(timezone.utc) + timedelta(days=31)
    assert handled.filter_suggestions(store.conn, now=later, own=()) == []
    handled.undo(store.conn, FakeGmail(), 1)
    assert handled.filter_suggestions(store.conn, own=()) == []


def test_filter_never_suggests_your_own_address(store, monkeypatch):
    _handled(store, "1", "2", "3", sender="Me@Example.com")
    handled.apply(store.conn, FakeGmail(), 1, "archive", handled.pending(store.conn, "archive"))
    assert handled.filter_suggestions(store.conn, own=["me@example.com"]) == []
    monkeypatch.setenv("LMSA_IMAP_USER", "me@example.com")
    assert handled.filter_suggestions(store.conn) == []  # default guard: LMSA_IMAP_USER


def test_followup_rerecord_keeps_the_recipient_identity(store):
    from types import SimpleNamespace as NS

    for i in ("21", "22", "23"):
        handled.seen(store.conn, "followup_radar", 2, [NS(msg_id=i, subject="s", recipient="bob@x.com")])
        handled.mark(store.conn, i)
        # a later run re-records it with your own From address...
        handled.seen(store.conn, "followup_radar", 3, [NS(msg_id=i, subject="s", from_addr="me@mine.com")])
        handled.unmark(store.conn, i)
        handled.mark(store.conn, i)
    senders = {r[0] for r in store.conn.execute("SELECT sender FROM handled")}
    assert senders == {"to bob@x.com"}
    handled.apply(store.conn, FakeGmail(), 6, "archive", handled.pending(store.conn, "archive"))
    assert handled.filter_suggestions(store.conn, own=()) == []


def test_followup_radar_records_done_items_by_recipient(tmp_path, monkeypatch):
    from tasks import followup_radar as fr

    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    sent = _msg("31", subject="Proposal", sender="me@mine.com", thread_id="t31", to=["bob@x.com"],
                date=now - timedelta(days=5))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    with Store() as s:
        # checked off elsewhere, so the radar run itself is what records it
        handled.seen(s.conn, "inbox_brief", 1, [sent])
        handled.mark(s.conn, "31")

    class G:
        cfg = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def sent(self):
            return "[Gmail]/Sent Mail"

        def search(self, query, folder=None, limit=0, headers_only=False):
            return [sent] if folder else [sent]

    class LM:
        def ensure_server(self):
            pass

    monkeypatch.setattr(fr, "Gmail", lambda *a, **k: G())
    monkeypatch.setattr(fr, "LMStudio", lambda: LM())
    monkeypatch.setattr(fr, "model_id", lambda lm: "m")
    monkeypatch.setattr(fr, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: now)}))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    fr.main([])
    with Store() as s:
        assert s.conn.execute(
            "SELECT sender FROM handled_seen WHERE message_id='31' AND source_task='followup_radar'"
        ).fetchone() == ("to bob@x.com",)


def test_filter_xml_is_an_importable_gmail_filter_file():
    xml = handled.filter_xml(
        ["news@example.com", "bad' onload='x@example.com", "not an address"],
        now=datetime(2026, 9, 25, tzinfo=timezone.utc),
    )
    root = ET.fromstring(xml)
    ns = {"a": "http://www.w3.org/2005/Atom", "apps": "http://schemas.google.com/apps/2006"}
    entries = root.findall("a:entry", ns)
    assert len(entries) == 1
    props = {p.get("name"): p.get("value") for p in entries[0].findall("apps:property", ns)}
    assert props == {"from": "news@example.com", "shouldArchive": "true"}
    assert entries[0].find("a:category", ns).get("term") == "filter"


def test_dashboard_serves_filter_xml_only_when_suggested(server):
    status, _h, _b = call(server, "GET", "/api/inbox/filters.xml")
    assert status == 404
    with Store(server.dashboard.db_path) as s:
        _handled(s, "1", "2", "3")
        handled.apply(s.conn, FakeGmail(), 1, "archive", handled.pending(s.conn, "archive"))
    status, headers, body = call(server, "GET", "/api/inbox/filters.xml")
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    assert b"news@example.com" in body
    status, j = jcall(server, "GET", "/api/inbox")
    assert j["filters"] == [{"sender": "news@example.com", "count": 3}]


def test_pending_limit_below_one_is_refused_not_read_as_no_cap(store):
    _handled(store, "1", "2", "3")
    assert len(handled.pending(store.conn, "archive")) == 3
    assert len(handled.pending(store.conn, "archive", limit=2)) == 2
    for bad in (0, -1):
        with pytest.raises(ValueError):
            handled.pending(store.conn, "archive", limit=bad)


def test_inbox_actions_rejects_limit_zero():
    from tasks import inbox_actions
    with pytest.raises(SystemExit):
        inbox_actions.main(["--archive-handled", "--limit", "0", "--apply"])
