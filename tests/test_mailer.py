from __future__ import annotations

import smtplib
import sqlite3
import sys
import threading

import pytest

from lmsa import mailer as mailer_mod
from lmsa.mailer import SendResult, _build_message, _claim, already_sent, send
from lmsa.store import Store

FAKE_PASSWORD = "test-app-pw-3f9c2b7a"  # invented; never a real credential


class _FakeSMTP:
    """Records call order; no socket is ever opened."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.calls: list[str] = []
        self.sent_message = None
        self.starttls_error: Exception | None = None
        self.login_error: Exception | None = None
        _FakeSMTP.instances.append(self)

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")
        if self.starttls_error:
            raise self.starttls_error

    def login(self, user, password):
        self.calls.append("login")
        self.login_user = user
        self.login_password = password
        if self.login_error:
            raise self.login_error

    def send_message(self, msg):
        self.calls.append("send_message")
        self.sent_message = msg

    def quit(self):
        self.calls.append("quit")


class _RefusingSMTP:
    """Raises the moment it's constructed -- proves dry-run never gets here."""

    def __init__(self, *a, **k):
        raise AssertionError("dry-run must not open an SMTP connection")


class _StoreFailsAfterFirstUse:
    """A `Store` stand-in that behaves normally for the *first* `with`
    block opened against it (the atomic claim, so `send()` reaches the
    SMTP attempt) and then raises like a locked database for every one
    after that (the post-attempt record write) -- isolates the finding-1
    repro ("Store failing while recording the outcome") from the claim
    step, which would otherwise short-circuit before SMTP is ever touched.
    """

    uses = 0

    def __init__(self, path=None):
        pass

    def __enter__(self):
        _StoreFailsAfterFirstUse.uses += 1
        if _StoreFailsAfterFirstUse.uses == 1:
            self._real = Store()
            return self._real.__enter__()
        raise sqlite3.OperationalError("database is locked")

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_real"):
            return self._real.__exit__(exc_type, exc, tb)
        return False


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_IMAP_USER", "owner@example.test")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_IMAP_PASSWORD", FAKE_PASSWORD)
    _FakeSMTP.instances = []
    _StoreFailsAfterFirstUse.uses = 0
    yield


@pytest.fixture
def fake_smtp(monkeypatch):
    monkeypatch.setattr(mailer_mod, "_smtp_class", _FakeSMTP)
    return _FakeSMTP


# -- MIME structure -----------------------------------------------------


def test_build_message_is_multipart_alternative_without_attachments():
    msg = _build_message("Subj", "<p>hi</p>", "hi", (), sender="a@example.test", to="b@example.test")
    assert msg.get_content_type() == "multipart/alternative"
    parts = list(msg.iter_parts())
    assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]
    assert parts[0].get_content().strip() == "hi"
    assert "<p>hi</p>" in parts[1].get_content()
    assert msg["Subject"] == "Subj"
    assert msg["From"] == "a@example.test"
    assert msg["To"] == "b@example.test"
    assert msg["Message-ID"]


def test_build_message_wraps_in_multipart_mixed_with_attachments():
    msg = _build_message(
        "Subj", "<p>hi</p>", "hi",
        [("report.txt", b"data", "text/plain")],
        sender="a@example.test", to="b@example.test",
    )
    assert msg.get_content_type() == "multipart/mixed"
    top_parts = list(msg.iter_parts())
    assert top_parts[0].get_content_type() == "multipart/alternative"
    attachment = top_parts[1]
    assert attachment.get_content_type() == "text/plain"
    assert attachment.get_filename() == "report.txt"
    assert attachment.get_content() == "data"


def test_message_id_uses_the_sender_domain_and_never_looks_up_the_hostname(monkeypatch):
    import socket

    def _no_dns():
        raise AssertionError("make_msgid() fell back to socket.getfqdn()")

    monkeypatch.setattr(socket, "getfqdn", _no_dns)
    msg = _build_message("Subj", "<p>hi</p>", "hi", (), sender="a@example.test", to="b@example.test")
    assert msg["Message-ID"].rstrip(">").endswith("@example.test")


@pytest.mark.parametrize("subject,to", [("a\nBcc: x@example.test", None), ("ok", "b@example.test\r\nBcc: x@example.test")])
def test_header_injection_is_a_failed_result_not_a_raise(monkeypatch, subject, to):
    monkeypatch.setattr(mailer_mod, "_smtp_class", _RefusingSMTP)
    for dry_run in (True, False):
        result = send(subject, "<p>hi</p>", "hi", to=to, dry_run=dry_run)
        assert result.ok is False
        assert result.error_type == "ValueError"


# -- dry run --------------------------------------------------------------


def test_dry_run_writes_eml_and_opens_no_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(mailer_mod, "_smtp_class", _RefusingSMTP)
    result = send("Subj", "<p>hi</p>", "hi", dry_run=True)
    assert result.ok is True
    assert result.dry_run is True
    assert result.path is not None
    assert result.path.exists()
    content = result.path.read_bytes()
    assert b"Subj" in content
    assert FAKE_PASSWORD.encode() not in content


def test_dry_run_never_touches_sends_table(monkeypatch, tmp_path):
    monkeypatch.setattr(mailer_mod, "_smtp_class", _RefusingSMTP)
    send("Subj", "<p>hi</p>", "hi", dry_run=True, name="morning")
    with Store() as store:
        row = store.conn.execute("SELECT COUNT(*) FROM sends").fetchone()
    assert row[0] == 0


# -- real send: call order, success -----------------------------------


def test_real_send_call_order_and_records_sent(fake_smtp):
    result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")

    assert result.ok is True
    assert result.dry_run is False
    assert result.message_id

    smtp = fake_smtp.instances[0]
    assert smtp.calls == ["ehlo", "starttls", "ehlo", "login", "send_message", "quit"]
    assert smtp.login_password == FAKE_PASSWORD
    assert smtp.sent_message["Subject"] == "Subj"

    with Store() as store:
        row = store.conn.execute(
            "SELECT status, message_id, error_type FROM sends WHERE name='morning'"
        ).fetchone()
    assert row == ("sent", result.message_id, None)


def test_real_send_with_no_name_never_touches_sends_table(fake_smtp):
    send("Subj", "<p>hi</p>", "hi", dry_run=False)
    with Store() as store:
        row = store.conn.execute("SELECT COUNT(*) FROM sends").fetchone()
    assert row[0] == 0


# -- auth failure: no password leak, login never called after starttls raises --


def test_starttls_failure_never_calls_login_and_records_failed(fake_smtp, caplog):
    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.starttls_error = smtplib.SMTPException(f"tls failed for {FAKE_PASSWORD}")
        return smtp

    mailer_mod._smtp_class = _raising_ctor

    with caplog.at_level("WARNING"):
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")

    assert result.ok is False
    assert result.error_type == "SMTPException"

    smtp = _FakeSMTP.instances[0]
    assert smtp.calls == ["ehlo", "starttls", "quit"]
    assert "login" not in smtp.calls

    for record in caplog.records:
        assert FAKE_PASSWORD not in record.getMessage()

    with Store() as store:
        row = store.conn.execute(
            "SELECT status, message_id, error_type FROM sends WHERE name='morning'"
        ).fetchone()
    assert row == ("failed", None, "SMTPException")


def test_login_failure_message_containing_password_never_leaks(fake_smtp, caplog):
    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.login_error = smtplib.SMTPAuthenticationError(535, f"bad creds: {FAKE_PASSWORD}".encode())
        return smtp

    mailer_mod._smtp_class = _raising_ctor

    with caplog.at_level("WARNING"):
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")

    assert result.ok is False
    assert result.error_type == "SMTPAuthenticationError"
    for record in caplog.records:
        assert FAKE_PASSWORD not in record.getMessage()
    assert FAKE_PASSWORD not in str(result)


def test_send_failure_notifies_urgent_without_password(fake_smtp, monkeypatch):
    calls = []
    monkeypatch.setattr(mailer_mod, "notify", lambda title, body, url=None, level="info": calls.append(
        (title, body, level)
    ))

    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.login_error = smtplib.SMTPAuthenticationError(535, f"bad creds: {FAKE_PASSWORD}".encode())
        return smtp

    mailer_mod._smtp_class = _raising_ctor

    send("Subj", "<p>hi</p>", "hi", dry_run=False)

    assert len(calls) == 1
    title, body, level = calls[0]
    assert level == "urgent"
    assert FAKE_PASSWORD not in title
    assert FAKE_PASSWORD not in body


# -- idempotency: already_sent / force / failed rows don't block a retry --


def test_already_sent_true_only_after_a_successful_send(fake_smtp):
    with Store() as store:
        assert already_sent(store.conn, "morning") is False

    send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")

    with Store() as store:
        assert already_sent(store.conn, "morning") is True


def test_second_send_same_day_is_skipped_unless_forced(fake_smtp):
    first = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")
    assert first.skipped is False

    second = send("Subj v2", "<p>hi</p>", "hi", dry_run=False, name="morning")
    assert second.skipped is True
    assert len(_FakeSMTP.instances) == 1  # no second connection attempted

    third = send("Subj v3", "<p>hi</p>", "hi", dry_run=False, name="morning", force=True)
    assert third.skipped is False
    assert len(_FakeSMTP.instances) == 2


def test_failed_send_does_not_block_a_retry_for_the_same_name_and_day(fake_smtp):
    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.login_error = smtplib.SMTPAuthenticationError(535, b"bad creds")
        return smtp

    mailer_mod._smtp_class = _raising_ctor
    first = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")
    assert first.ok is False

    mailer_mod._smtp_class = _FakeSMTP
    second = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")
    assert second.ok is True
    assert second.skipped is False

    with Store() as store:
        rows = store.conn.execute("SELECT status FROM sends WHERE name='morning'").fetchall()
    assert rows == [("sent",)]  # the failed row was overwritten, not duplicated


# -- stage-5 review: send() must never raise, on any failure-path step ----


def test_broken_redact_import_never_raises_and_never_leaks_the_password(fake_smtp, monkeypatch, caplog):
    """Repro: knocking `tasks.dashboard_setup` out of `sys.modules` makes
    `_safe()`'s deferred import fail, which used to happen *inside* the
    `except Exception as exc:` block handling the SMTP error -- so the
    ImportError/ModuleNotFoundError propagated to send()'s caller chained
    onto the original (password-bearing) SMTPAuthenticationError."""
    monkeypatch.setitem(sys.modules, "tasks.dashboard_setup", None)

    notified = []
    monkeypatch.setattr(
        mailer_mod, "notify",
        lambda title, body, url=None, level="info": notified.append((title, body)),
    )

    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.login_error = smtplib.SMTPAuthenticationError(535, f"bad creds: {FAKE_PASSWORD}".encode())
        return smtp

    mailer_mod._smtp_class = _raising_ctor

    with caplog.at_level("WARNING"):
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")  # must not raise

    assert result.ok is False
    assert result.error_type == "SMTPAuthenticationError"
    assert FAKE_PASSWORD not in repr(result)
    for record in caplog.records:
        assert FAKE_PASSWORD not in record.getMessage()
    for title, body in notified:
        assert FAKE_PASSWORD not in title
        assert FAKE_PASSWORD not in body


def test_locked_store_while_recording_a_failure_never_raises_or_leaks(fake_smtp, monkeypatch, caplog):
    """Repro: a `Store` whose `__enter__` raises `sqlite3.OperationalError`
    on the post-failure record write used to propagate straight out of the
    `except` block handling the SMTP error, chaining the original
    (password-bearing) exception onto it."""
    monkeypatch.setattr(mailer_mod, "Store", _StoreFailsAfterFirstUse)
    monkeypatch.setattr(mailer_mod, "notify", lambda *a, **k: None)  # no real toast/Slack side effect

    def _raising_ctor(host, port, timeout=None):
        smtp = _FakeSMTP(host, port, timeout=timeout)
        smtp.login_error = smtplib.SMTPAuthenticationError(535, f"bad creds: {FAKE_PASSWORD}".encode())
        return smtp

    mailer_mod._smtp_class = _raising_ctor

    with caplog.at_level("WARNING"):
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")  # must not raise

    assert result.ok is False
    assert result.error_type == "SMTPAuthenticationError"
    for record in caplog.records:
        assert FAKE_PASSWORD not in record.getMessage()


def test_keyring_failure_resolving_the_password_becomes_a_failed_result(fake_smtp, monkeypatch):
    def _broken_resolve(name):
        raise RuntimeError("no keyring backend")

    monkeypatch.setattr(mailer_mod, "resolve", _broken_resolve)
    monkeypatch.setattr(mailer_mod, "notify", lambda *a, **k: None)  # no real toast/Slack side effect

    result = send("Subj", "<p>hi</p>", "hi", dry_run=False)  # must not raise

    assert result.ok is False
    assert result.error_type == "RuntimeError"
    assert _FakeSMTP.instances == []  # never even got as far as opening a connection


def test_recording_failure_after_a_successful_send_still_reports_success(fake_smtp, monkeypatch, caplog):
    monkeypatch.setattr(mailer_mod, "Store", _StoreFailsAfterFirstUse)

    with caplog.at_level("WARNING"):
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="morning")  # must not raise

    assert result.ok is True
    assert result.message_id
    assert any("record a sent send" in r.getMessage() for r in caplog.records)


# -- stage-5 review: the claim is atomic ------------------------------


def test_claim_reclaims_a_failed_row_but_not_a_sent_or_sending_row():
    with Store() as store:
        conn = store.conn
        assert _claim(conn, "morning", "2026-01-01", force=False) is True  # fresh row
        assert _claim(conn, "morning", "2026-01-01", force=False) is False  # already 'sending'

        conn.execute("UPDATE sends SET status='failed' WHERE name='morning' AND date='2026-01-01'")
        conn.commit()
        assert _claim(conn, "morning", "2026-01-01", force=False) is True  # failed row reclaimed

        conn.execute("UPDATE sends SET status='sent' WHERE name='morning' AND date='2026-01-01'")
        conn.commit()
        assert _claim(conn, "morning", "2026-01-01", force=False) is False  # sent row not reclaimed
        assert _claim(conn, "morning", "2026-01-01", force=True) is True  # force bypasses the check


def test_claim_is_atomic_under_concurrent_real_sends(fake_smtp):
    """Two threads both call `send()` for the same (name, day) at once.
    The atomic INSERT/claim must let exactly one of them through to SMTP;
    the other must see it already claimed and skip, never double-sending."""
    barrier = threading.Barrier(2)
    results: list[SendResult] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        result = send("Subj", "<p>hi</p>", "hi", dry_run=False, name="race")
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    skipped = [r for r in results if r.skipped]
    sent = [r for r in results if r.ok and not r.skipped]
    assert len(skipped) == 1
    assert len(sent) == 1
    assert len(_FakeSMTP.instances) == 1  # only the winner ever touched SMTP

    with Store() as store:
        rows = store.conn.execute("SELECT status FROM sends WHERE name='race'").fetchall()
    assert rows == [("sent",)]
