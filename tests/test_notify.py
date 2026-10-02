import subprocess
from types import SimpleNamespace

import pytest

from lmsa import notify as notify_mod
from lmsa.notify import notify, slack, toast


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_toast_builds_powershell_command_with_title_and_body(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(notify_mod.subprocess, "run", fake_run)

    assert toast("Task failed", "boom happened", url="https://example.test/run/1") is True

    cmd = captured["cmd"]
    assert cmd[0] == "powershell"
    script = cmd[-1]
    assert "'Task failed'" in script
    assert "'boom happened'" in script
    assert "SetAttribute('launch', 'https://example.test/run/1')" in script
    assert "ToastNotificationManager" in script


def test_toast_escapes_embedded_single_quotes(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(notify_mod.subprocess, "run", fake_run)

    toast("it's broken", "body")

    assert "'it''s broken'" in captured["cmd"][-1]


def test_toast_fails_soft_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        notify_mod.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stderr=b"no toast for you"),
    )
    assert toast("t", "b") is False


def test_toast_fails_soft_when_powershell_missing(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("powershell not found")

    monkeypatch.setattr(notify_mod.subprocess, "run", fake_run)
    assert toast("t", "b") is False


def test_toast_skips_off_windows(monkeypatch):
    monkeypatch.setattr(notify_mod.sys, "platform", "linux")
    assert toast("t", "b") is False


def test_slack_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("LMSA_SLACK_WEBHOOK", raising=False)
    called = False

    def fake_urlopen(*a, **k):
        nonlocal called
        called = True
        return _FakeResponse(200)

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)

    assert slack("hello") is False
    assert called is False


def test_slack_posts_payload_shape(monkeypatch):
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", "https://hooks.example.test/services/fake")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["content_type"] = req.get_header("Content-type")
        return _FakeResponse(200)

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)

    assert slack("hello world") is True
    assert captured["url"] == "https://hooks.example.test/services/fake"
    assert captured["body"] == b'{"text": "hello world"}'
    assert captured["content_type"] == "application/json"


def test_slack_reads_the_keyring_and_the_env_var_still_wins(monkeypatch, fake_keyring):
    monkeypatch.delenv("LMSA_SLACK_WEBHOOK", raising=False)
    fake_keyring.set_password("lmsa", "slack", "https://hooks.example.test/services/from-keyring")
    urls = []
    monkeypatch.setattr(notify_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: urls.append(req.full_url) or _FakeResponse(200))
    assert slack("a") is True
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", "https://hooks.example.test/services/from-env")
    assert slack("b") is True
    assert urls == ["https://hooks.example.test/services/from-keyring",
                    "https://hooks.example.test/services/from-env"]


def test_slack_fails_soft_when_the_keyring_is_broken(monkeypatch, fake_keyring):
    monkeypatch.delenv("LMSA_SLACK_WEBHOOK", raising=False)

    def broken(service, name):
        raise RuntimeError("no backend")

    monkeypatch.setattr(fake_keyring, "get_password", broken)
    assert slack("hello") is False


def test_slack_fails_soft_on_http_error(monkeypatch):
    monkeypatch.setenv("LMSA_SLACK_WEBHOOK", "https://hooks.example.test/services/fake")

    def fake_urlopen(*a, **k):
        raise notify_mod.urllib.error.URLError("boom")

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)
    assert slack("hello") is False


def test_notify_fans_out_to_both_channels(monkeypatch):
    calls = []
    monkeypatch.setattr(notify_mod, "toast", lambda *a, **k: calls.append(("toast", a, k)))
    monkeypatch.setattr(notify_mod, "slack", lambda *a, **k: calls.append(("slack", a, k)))

    notify("Title", "Body", url="https://example.test", level="urgent")

    assert calls[0][0] == "toast"
    assert calls[0][1] == ("Title", "Body", "https://example.test")
    assert calls[1][0] == "slack"
    text = calls[1][1][0]
    assert "Title" in text and "Body" in text and "https://example.test" in text
    assert ":rotating_light:" in text


def test_notify_rejects_unknown_level():
    with pytest.raises(ValueError):
        notify("t", "b", level="nope")


def test_toast_timeout_fails_soft(monkeypatch):
    import subprocess
    from lmsa import notify

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=15)

    monkeypatch.setattr(notify.subprocess, "run", hang)
    assert notify.toast("t", "b") is False
