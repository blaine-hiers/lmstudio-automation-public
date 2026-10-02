"""The shared Gmail cursor scheme (lmsa.gmail_cursor), end to end through
every task that uses it: receipts, packages, job_tracker, attachments.

Every scenario is a way mail used to be lost (or processed twice), and
each asserts that every message is processed exactly once:

- UID order != INTERNALDATE order (imported/restored mail: a high UID with
  an old delivery time), and a stale/skewed Date header;
- same-second siblings under a limit of 1;
- mail delivered while a run is busy in its model loop;
- Gmail index lag: a message delivered before the run started that only
  shows up in search on a later run;
- a backlog bigger than the per-run limit, worked through over several runs
  (with a "pending" note in the report);
- per-item failures retried, then given up on (and reported).

The fake mailbox behaves like Gmail: `after:` filters on INTERNALDATE
(strictly, the worst case), UIDs are assigned in arrival order, and
`search_meta`/`fetch_uids` are the metadata-then-body API the cursor uses.
All fixtures are invented (example.com).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Callable

import pytest
from PIL import Image

import tasks.attachments as attachments
import tasks.job_tracker as job_tracker
import tasks.packages as packages
import tasks.receipts as receipts
from lmsa import gmail_cursor
from lmsa.gmail_cursor import MAX_RETRY_ATTEMPTS
from lmsa.sources.gmail import Attachment, Message, MessageMeta


def _jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (20, 20), (30, 200, 30)).save(buf, format="JPEG")
    return buf.getvalue()


JPEG = _jpeg()


def _msg(n: int, internal: datetime, header: datetime | None = None) -> Message:
    return Message(
        msg_id=str(1000 + n), thread_id=f"t{n}", labels=[], flags=[],
        from_name="Shop", from_addr="shop@example.com", to=["me@example.com"], reply_to="",
        subject=f"Order {n} receipt", date=header or internal,
        list_unsubscribe="", list_id="", auth_results="", text=f"Order {n} details.",
        attachments=[Attachment(name=f"order{n}.jpg", mime="image/jpeg", size=len(JPEG))],
        web_url="", internal_date=internal,
    )


class FakeMailbox:
    def __init__(self, messages=()):
        self.messages: list[Message] = []
        self.uids: dict[str, int] = {}
        self.hidden: set[str] = set()  # delivered but not yet in Gmail's search index
        self.queries: list[str] = []
        self.refetched: list[list[str]] = []
        for m in messages:
            self.deliver(m)

    def deliver(self, msg: Message, *, hidden: bool = False) -> None:
        self.messages.append(msg)
        self.uids[msg.msg_id] = len(self.uids) + 1  # UIDs follow arrival, not INTERNALDATE
        if hidden:
            self.hidden.add(msg.msg_id)

    def __call__(self, cfg=None):  # stands in for the Gmail class
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search_meta(self, query, folder="[Gmail]/All Mail"):
        self.queries.append(query)
        lo = int(re.search(r"after:(\d+)", query).group(1))
        return [
            MessageMeta(msg_id=m.msg_id, internal_date=m.internal_date, uid=self.uids[m.msg_id])
            for m in self.messages
            if m.msg_id not in self.hidden and m.internal_date.timestamp() > lo
        ]

    def fetch_uids(self, uids, folder="[Gmail]/All Mail", *, headers_only=False):
        by_uid = {self.uids[m.msg_id]: m for m in self.messages}
        return [by_uid[u] for u in uids]

    def fetch_by_msgids(self, msg_ids, folder="[Gmail]/All Mail", *, headers_only=False):
        self.refetched.append(list(msg_ids))
        return [m for i in msg_ids for m in self.messages if m.msg_id == i]

    def attachment_bytes(self, msg, index):
        return JPEG


class FakeLM:
    """extract()/see() record the Subject they were asked about; `fail`
    subjects raise (forever, or `fail_once`); `on_call` runs inside the
    model call (to simulate mail arriving mid-run)."""

    def __init__(self, result_for: Callable[[str], dict], out_dir):
        self.result_for = result_for
        self.subjects: list[str] = []
        self.fail: set[str] = set()
        self.fail_once: set[str] = set()
        self.on_call: Callable[[], None] | None = None
        self.cfg = type("Cfg", (), {"output_dir": out_dir, "chat_model": "fake", "vision_model": "fake"})()

    def ensure_server(self):
        pass

    def _answer(self, prompt: str) -> dict:
        subject = re.search(r"Subject: (.*)", prompt).group(1).strip()
        self.subjects.append(subject)
        if self.on_call is not None:
            hook, self.on_call = self.on_call, None
            hook()
        if subject in self.fail:
            raise RuntimeError("model timed out")
        if subject in self.fail_once:
            self.fail_once.discard(subject)
            raise RuntimeError("model timed out")
        return self.result_for(subject)

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        return self._answer(prompt)

    def see(self, images, prompt, schema=None, *, system=None, model=None, name="result"):
        return self._answer(prompt)


def _receipt(subject):
    return {"is_receipt": True, "merchant": "Example Shop", "amount": "1.00", "currency": "USD",
            "date": "", "category": "", "recurring": False}


def _shipment(subject):
    n = re.search(r"\d+", subject).group(0)
    return {"carrier": "UPS", "tracking_number": f"TRACK{n}Z9999", "merchant": "Example Shop",
            "status": "shipped", "eta": None}


def _job(subject):
    n = re.search(r"\d+", subject).group(0)
    return {"is_job_related": True, "company": f"Example Co {n}", "role": "Engineer",
            "stage": "applied", "next_step": "", "contact": ""}


@dataclass
class TaskCase:
    name: str
    module: object
    result_for: Callable[[str], dict]

    def run(self, monkeypatch, limit: int) -> None:
        if self.module is packages:
            monkeypatch.setattr(packages, "SEARCH_LIMIT", limit)
            packages.main()
        elif self.module is job_tracker:
            job_tracker.main(["--limit", str(limit)])
        else:
            monkeypatch.setattr("sys.argv", [self.name, "--limit", str(limit)])
            self.module.main()


CASES = [
    TaskCase("receipts", receipts, _receipt),
    TaskCase("packages", packages, _shipment),
    TaskCase("job_tracker", job_tracker, _job),
    TaskCase("attachments", attachments, _receipt),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    return tmp_path


@pytest.fixture
def clock(monkeypatch):
    """The cursor's clock, advancing 10s per reading, so "the run's start"
    and "the end of the run" are distinguishable (with a real clock both
    usually fall in the same whole second and a wrong cursor goes unseen).
    `clock.arrival()` is a delivery time strictly after the last reading."""

    class Clock:
        t = datetime.now(timezone.utc)

        def now(self):
            self.t += timedelta(seconds=10)
            return self.t

        def arrival(self):
            return self.t + timedelta(seconds=5)

    c = Clock()
    monkeypatch.setattr(gmail_cursor, "_now", c.now)
    return c


def _install(case: TaskCase, monkeypatch, env, box: FakeMailbox) -> FakeLM:
    lm = FakeLM(case.result_for, env / "output")
    monkeypatch.setattr(case.module, "Gmail", box)
    monkeypatch.setattr(case.module, "LMStudio", lambda: lm)
    return lm


def _latest_report(env) -> str:
    reports = [p for p in (env / "output").rglob("*.html") if p.name != "index.html"]
    return max(reports, key=lambda p: p.stat().st_mtime_ns).read_text(encoding="utf-8")


def _subjects(*ns: int) -> list[str]:
    return [f"Order {n} receipt" for n in ns]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_uid_order_and_date_header_disagreeing_with_internaldate_lose_nothing(case, env, monkeypatch):
    now = datetime.now(timezone.utc)
    box = FakeMailbox([
        _msg(1, now - timedelta(hours=9)),
        _msg(2, now - timedelta(hours=8)),
        _msg(3, now - timedelta(hours=7), header=now - timedelta(days=2)),  # stale Date header
        _msg(4, now - timedelta(hours=6)),
        _msg(5, now - timedelta(days=5)),  # imported/restored: highest UID, oldest INTERNALDATE
    ])
    lm = _install(case, monkeypatch, env, box)
    for _ in range(4):
        case.run(monkeypatch, limit=2)
    assert sorted(lm.subjects) == _subjects(1, 2, 3, 4, 5)  # every one, exactly once


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_same_second_siblings_are_both_processed_at_limit_one(case, env, monkeypatch):
    when = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=3)
    box = FakeMailbox([_msg(1, when), _msg(2, when), _msg(3, when + timedelta(microseconds=500))])
    lm = _install(case, monkeypatch, env, box)
    for _ in range(4):
        case.run(monkeypatch, limit=1)
    assert sorted(lm.subjects) == _subjects(1, 2, 3)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mail_delivered_during_the_model_loop_is_picked_up_next_run(case, env, monkeypatch, clock):
    now = datetime.now(timezone.utc)
    box = FakeMailbox([_msg(1, now - timedelta(hours=1))])
    lm = _install(case, monkeypatch, env, box)
    lm.on_call = lambda: box.deliver(_msg(2, clock.arrival()))
    case.run(monkeypatch, limit=200)
    case.run(monkeypatch, limit=200)
    case.run(monkeypatch, limit=200)
    assert lm.subjects == _subjects(1, 2)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mail_gmail_indexes_late_is_still_picked_up(case, env, monkeypatch, clock):
    now = datetime.now(timezone.utc)
    box = FakeMailbox([_msg(1, now - timedelta(hours=1))])
    lm = _install(case, monkeypatch, env, box)
    # Delivered a minute *before* the run starts, but not searchable yet.
    box.deliver(_msg(2, clock.t + timedelta(minutes=-1)), hidden=True)
    case.run(monkeypatch, limit=200)
    box.hidden.clear()  # the index catches up
    case.run(monkeypatch, limit=200)
    case.run(monkeypatch, limit=200)
    assert lm.subjects == _subjects(1, 2)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_a_backlog_bigger_than_the_limit_is_worked_through_oldest_first(case, env, monkeypatch, clock):
    now = datetime.now(timezone.utc)
    # Delivered newest-first, so UID order is the reverse of INTERNALDATE order.
    box = FakeMailbox([_msg(n, now - timedelta(hours=n)) for n in range(1, 8)])
    lm = _install(case, monkeypatch, env, box)
    case.run(monkeypatch, limit=3)
    assert lm.subjects == _subjects(7, 6, 5)  # oldest first
    assert "4 more matching message(s) are pending" in _latest_report(env)
    lm.on_call = lambda: box.deliver(_msg(8, clock.arrival()))  # new mail mid-backlog
    for _ in range(3):
        case.run(monkeypatch, limit=3)
    assert sorted(lm.subjects) == _subjects(1, 2, 3, 4, 5, 6, 7, 8)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_failed_item_is_retried_next_run(case, env, monkeypatch):
    now = datetime.now(timezone.utc)
    box = FakeMailbox([_msg(1, now - timedelta(hours=1))])
    lm = _install(case, monkeypatch, env, box)
    lm.fail_once.add("Order 1 receipt")
    case.run(monkeypatch, limit=200)
    assert "Retrying next run" in _latest_report(env)
    case.run(monkeypatch, limit=200)  # the forward cursor is past it; the retry set refetches it
    assert box.refetched == [["1001"]]
    assert lm.subjects == ["Order 1 receipt", "Order 1 receipt"]
    case.run(monkeypatch, limit=200)  # succeeded -- nothing left to retry
    assert box.refetched == [["1001"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_permanently_failing_item_is_given_up_on_and_reported(case, env, monkeypatch):
    now = datetime.now(timezone.utc)
    box = FakeMailbox([_msg(1, now - timedelta(hours=1))])
    lm = _install(case, monkeypatch, env, box)
    lm.fail.add("Order 1 receipt")
    for _ in range(MAX_RETRY_ATTEMPTS):
        case.run(monkeypatch, limit=200)
    assert f"Gave up after {MAX_RETRY_ATTEMPTS} attempts" in _latest_report(env)
    case.run(monkeypatch, limit=200)
    assert lm.subjects.count("Order 1 receipt") == MAX_RETRY_ATTEMPTS  # never tried again
