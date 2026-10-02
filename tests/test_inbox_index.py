from datetime import datetime, timezone
from types import SimpleNamespace

from lmsa.sources.gmail import Message
from lmsa.store import Store, unpack_vector
from tasks import inbox_index
from tasks.inbox_index import (
    build_report,
    embed_text,
    index_messages,
)

MODEL = "fake-embed"


def _msg(**overrides) -> Message:
    defaults = dict(
        msg_id="m1",
        thread_id="t1",
        labels=["INBOX"],
        flags=[],
        from_name="Alice Example",
        from_addr="alice@example.com",
        to=["bot@example.com"],
        reply_to="",
        subject="Roof budget",
        date=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        list_unsubscribe="",
        list_id="",
        auth_results="",
        text="The roof budget is $500. Let's confirm by Friday.",
        attachments=[],
        web_url="https://mail.google.com/mail/u/0/#all/t1",
    )
    defaults.update(overrides)
    return Message(**defaults)


class FakeLM:
    def __init__(self, fail_on: set[str] | None = None):
        self.cfg = SimpleNamespace(embed_model=MODEL)
        self.calls: list[list[str]] = []
        self.fail_on = fail_on or set()

    def embed(self, texts, *, model=None):
        self.calls.append(list(texts))
        if self.fail_on & set(texts):
            raise RuntimeError("embed endpoint down")
        return [[float(len(t)), float(t.count("a"))] for t in texts]


def test_embed_text_includes_subject_sender_and_truncates_body():
    msg = _msg(text="z" * 5000)
    text = embed_text(msg, body_chars=100)
    assert "Subject: Roof budget" in text
    assert "Alice Example" in text
    assert "alice@example.com" in text
    assert text.count("z") == 100


def test_index_messages_writes_metadata_and_embeddings(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        stats = index_messages(store.conn, lm, [_msg()], model=MODEL)

        assert stats.fetched == 1
        assert stats.indexed == 1
        assert stats.unchanged == 0
        assert stats.errors == []

        row = store.conn.execute(
            "SELECT thread_id, from_addr, subject FROM messages WHERE msg_id='m1'"
        ).fetchone()
        assert row == ("t1", "alice@example.com", "Roof budget")

        web_url, excerpt = store.conn.execute(
            "SELECT web_url, excerpt FROM mail_index WHERE msg_id='m1'"
        ).fetchone()
        assert web_url == "https://mail.google.com/mail/u/0/#all/t1"
        assert "roof budget" in excerpt.lower()

        vec_row = store.conn.execute(
            "SELECT vector FROM embeddings WHERE ns='gmail' AND item_id='m1' AND model=?",
            (MODEL,),
        ).fetchone()
        assert vec_row is not None
        assert len(unpack_vector(vec_row[0])) == 2


def test_index_messages_skips_already_embedded_under_current_model(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        index_messages(store.conn, lm, [_msg()], model=MODEL)
        assert len(lm.calls) == 1

        # Second run with the same message under the same model: metadata is
        # refreshed but no new embed() call happens.
        stats = index_messages(store.conn, lm, [_msg(subject="Roof budget (updated)")], model=MODEL)

        assert stats.indexed == 0
        assert stats.unchanged == 1
        assert len(lm.calls) == 1  # no second embed call
        subject = store.conn.execute("SELECT subject FROM messages WHERE msg_id='m1'").fetchone()[0]
        assert subject == "Roof budget (updated)"


def test_index_messages_reembeds_under_a_new_model(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        index_messages(store.conn, lm, [_msg()], model=MODEL)

        stats = index_messages(store.conn, lm, [_msg()], model="other-embed")

        assert stats.indexed == 1
        assert stats.unchanged == 0


def test_index_messages_skips_message_with_no_msg_id(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        stats = index_messages(store.conn, lm, [_msg(msg_id="")], model=MODEL)

        assert stats.indexed == 0
        assert len(stats.errors) == 1
        assert "X-GM-MSGID" in stats.errors[0][1]
        assert lm.calls == []


def test_index_messages_isolates_a_failing_batch(tmp_path):
    """One message that makes the embed call blow up must not stop other
    messages (in other batches) from being indexed."""
    with Store(tmp_path / "db.sqlite") as store:
        good = _msg(msg_id="good", subject="Good", text="fine content")
        bad_text = embed_text(_msg(msg_id="bad", subject="Bad", text="bad content"))
        lm = FakeLM(fail_on={bad_text})

        stats = index_messages(
            store.conn, lm, [good, _msg(msg_id="bad", subject="Bad", text="bad content")], model=MODEL,
        )

        # both in the same batch (well under EMBED_BATCH_SIZE) -> the batch
        # call raises, so both come back as errors and neither embeds --
        # but metadata for both was still committed.
        assert len(stats.errors) == 2
        assert stats.indexed == 0
        meta_count = store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert meta_count == 2


def test_index_messages_per_item_isolation_across_batches(tmp_path, monkeypatch):
    """A bad batch fails without blocking a later, good batch."""
    monkeypatch.setattr(inbox_index, "EMBED_BATCH_SIZE", 1)
    with Store(tmp_path / "db.sqlite") as store:
        good = _msg(msg_id="good", subject="Good", text="fine content")
        bad = _msg(msg_id="bad", subject="Bad", text="bad content")
        bad_text = embed_text(bad)
        lm = FakeLM(fail_on={bad_text})

        stats = index_messages(store.conn, lm, [bad, good], model=MODEL)

        assert stats.indexed == 1  # "good" made it through
        assert len(stats.errors) == 1
        assert stats.errors[0][0] == "bad"
        vec_row = store.conn.execute(
            "SELECT 1 FROM embeddings WHERE ns='gmail' AND item_id='good' AND model=?", (MODEL,)
        ).fetchone()
        assert vec_row is not None


def test_build_report_shows_truncated_note_and_errors():
    stats = inbox_index.IndexStats(fetched=200, indexed=5, unchanged=195, truncated=True,
                                    errors=[("m9", "boom")])
    html = build_report(stats, "in:inbox").render()
    assert "Hit the Gmail search limit" in html
    assert "boom" in html
    assert "m9" in html


class _FakeGmail:
    def __init__(self, messages):
        self._messages = messages
        self.queries = []
        self.limits = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, limit=200, **kwargs):
        self.queries.append(query)
        self.limits.append(limit)
        return self._messages[:limit]


def test_main_indexes_via_fake_gmail_and_writes_report(monkeypatch, tmp_path):
    fake = _FakeGmail([_msg()])
    monkeypatch.setattr(inbox_index, "Gmail", lambda cfg=None: fake)
    monkeypatch.setattr(inbox_index, "LMStudio", lambda: SimpleNamespace(
        ensure_server=lambda: None,
        cfg=SimpleNamespace(embed_model=MODEL, output_dir=tmp_path),
        embed=lambda texts, model=None: [[1.0, 2.0] for _ in texts],
    ))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["inbox_index.py"])

    inbox_index.main()

    assert fake.queries == ["in:inbox"]
    [written] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    assert "Inbox index" in written.read_text(encoding="utf-8")
