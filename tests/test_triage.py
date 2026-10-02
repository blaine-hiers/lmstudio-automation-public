import pytest

from lmsa import triage
from lmsa.sources.gmail import Message
from lmsa.triage import BUCKETS, SCHEMA, classify


def _message(**overrides) -> Message:
    base = dict(
        msg_id="1",
        thread_id="1",
        labels=[],
        flags=[],
        from_name="Alice Example",
        from_addr="alice@example.com",
        to=["bot@example.com"],
        reply_to="",
        subject="Quarterly numbers",
        date=None,
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text="Body text.",
        attachments=[],
        web_url="https://mail.google.com/mail/u/0/#all/1",
    )
    base.update(overrides)
    return Message(**base)


class FakeLM:
    def __init__(self, result=None, *, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append((prompt, schema, system, name))
        if self._error is not None:
            raise self._error
        return self._result


def test_classify_returns_the_four_documented_fields():
    msg = _message()
    lm = FakeLM({
        "bucket": "act",
        "why": "Invoice due at month end.",
        "action": "Pay the invoice.",
        "due_date": "2026-09-30",
    })

    result = classify(msg, lm)

    assert result == {
        "bucket": "act",
        "why": "Invoice due at month end.",
        "action": "Pay the invoice.",
        "due_date": "2026-09-30",
    }


def test_classify_passes_schema_and_system_prompt():
    msg = _message()
    lm = FakeLM({"bucket": "read", "why": "fyi", "action": "none", "due_date": None})

    classify(msg, lm)

    prompt, schema, system, name = lm.calls[0]
    assert schema == SCHEMA
    assert "untrusted data" in system
    assert name == "triage"
    assert "Quarterly numbers" in prompt
    assert "alice@example.com" in prompt


def test_classify_truncates_long_body_in_prompt():
    long_text = "x" * (triage.BODY_LIMIT + 500)
    msg = _message(text=long_text)
    lm = FakeLM({"bucket": "read", "why": "fyi", "action": "none", "due_date": None})

    classify(msg, lm)

    prompt = lm.calls[0][0]
    assert len(prompt) < len(long_text)
    assert "truncated" in prompt
    # the message itself is untouched
    assert len(msg.text) == triage.BODY_LIMIT + 500


def test_classify_propagates_extract_failure():
    msg = _message()
    lm = FakeLM(error=RuntimeError("server returned empty content"))

    with pytest.raises(RuntimeError, match="server returned empty content"):
        classify(msg, lm)


def test_classify_builds_a_default_client_when_none_given(monkeypatch):
    fake = FakeLM({"bucket": "ignore", "why": "newsletter", "action": "none", "due_date": None})
    monkeypatch.setattr(triage, "LMStudio", lambda: fake)

    result = classify(_message())

    assert result["bucket"] == "ignore"
    assert len(fake.calls) == 1


def test_buckets_match_documented_set():
    assert BUCKETS == ("act", "reply", "read", "ignore")
