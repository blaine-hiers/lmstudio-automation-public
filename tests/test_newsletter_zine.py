import json

from lmsa.sources.gmail import Message
from lmsa.store import Store
from tasks import newsletter_zine as nz


def _message(
    *,
    subject="Weekly digest",
    from_name="Newsletter",
    from_addr="news@example.com",
    text="Hello, this is the body.",
    list_unsubscribe="<mailto:unsub@example.com>",
    list_id="<digest.example.com>",
    web_url="https://mail.google.com/mail/u/0/#all/1",
    msg_id=None,
) -> Message:
    return Message(
        # Distinct per subject by default: Gmail msg_ids are unique, and the
        # summary memo (#46) is keyed on them.
        msg_id=msg_id if msg_id is not None else f"id-{subject}",
        thread_id="1",
        labels=[],
        flags=[],
        from_name=from_name,
        from_addr=from_addr,
        to=["bot@example.com"],
        reply_to="",
        subject=subject,
        date=None,
        list_unsubscribe=list_unsubscribe,
        list_id=list_id,
        auth_results="",
        text=text,
        attachments=[],
        web_url=web_url,
    )


# ---------------------------------------------------------------------------
# is_newsletter / truncate / normalize_topic
# ---------------------------------------------------------------------------

def test_is_newsletter_requires_list_id_or_list_unsubscribe():
    assert nz.is_newsletter(_message(list_unsubscribe="<mailto:x>", list_id="")) is True
    assert nz.is_newsletter(_message(list_unsubscribe="", list_id="<x>")) is True
    assert nz.is_newsletter(_message(list_unsubscribe="", list_id="")) is False


def test_truncate_leaves_short_text_alone():
    assert nz.truncate("short", limit=100) == "short"


def test_truncate_bounds_long_text_and_notes_omission():
    text = "x" * 5000
    out = nz.truncate(text, limit=4000)
    assert out.startswith("x" * 4000)
    assert "truncated" in out
    assert "1000" in out


def test_normalize_topic_merges_case_and_whitespace():
    assert nz.normalize_topic("AI  News") == nz.normalize_topic("ai news")
    assert nz.normalize_topic("  Politics ") == nz.normalize_topic("politics")
    assert nz.normalize_topic("Tech") != nz.normalize_topic("Politics")


# ---------------------------------------------------------------------------
# summarize_one: per-message isolation and score clamping
# ---------------------------------------------------------------------------

class FakeLMStudio:
    def __init__(self, results=None, fail_on=()):
        self.results = results or {}
        self.fail_on = set(fail_on)
        self.prompts: list[str] = []
        self.ensure_server_calls = 0

    def ensure_server(self):
        self.ensure_server_calls += 1

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.prompts.append(prompt)
        assert system is not None and "untrusted" in system
        subject_line = prompt.splitlines()[0]
        if subject_line in self.fail_on:
            raise RuntimeError("model server error")
        return self.results[subject_line]


def test_summarize_one_isolates_extract_failure():
    lm = FakeLMStudio(fail_on={"Subject: broken"})
    item = nz.summarize_one(lm, _message(subject="broken"))
    assert item.summary is None
    assert item.error == "model server error"


def test_summarize_one_clamps_out_of_range_scores():
    lm = FakeLMStudio(results={
        "Subject: hot": {
            "topic": "AI", "three_bullets": ["a", "b", "c"],
            "worth_full_read": 99, "why": "huge",
        },
        "Subject: cold": {
            "topic": "AI", "three_bullets": ["a", "b", "c"],
            "worth_full_read": -5, "why": "nothing",
        },
    })
    hot = nz.summarize_one(lm, _message(subject="hot"))
    cold = nz.summarize_one(lm, _message(subject="cold"))
    assert hot.summary.worth_full_read == 5
    assert cold.summary.worth_full_read == 0


def test_summarize_one_truncates_body_before_prompting():
    lm = FakeLMStudio(results={
        "Subject: long": {
            "topic": "AI", "three_bullets": ["a", "b", "c"],
            "worth_full_read": 3, "why": "ok",
        },
    })
    nz.summarize_one(lm, _message(subject="long", text="y" * 10_000))
    assert "truncated" in lm.prompts[0]
    assert len(lm.prompts[0]) < 5000


# ---------------------------------------------------------------------------
# group_by_topic
# ---------------------------------------------------------------------------

def _item(topic, subject="s", score=3):
    return nz.Item(
        message=_message(subject=subject),
        summary=nz.Summary(topic=topic, three_bullets=["a", "b", "c"], worth_full_read=score, why="w"),
    )


def test_group_by_topic_merges_near_duplicates_and_keeps_first_seen_title():
    items = [_item("AI News", "one"), _item("ai  news", "two"), _item("Politics", "three")]
    grouped = nz.group_by_topic(items)
    assert list(grouped) == ["AI News", "Politics"]
    assert [i.message.subject for i in grouped["AI News"]] == ["one", "two"]
    assert [i.message.subject for i in grouped["Politics"]] == ["three"]


def test_group_by_topic_skips_failed_items():
    failed = nz.Item(message=_message(subject="bad"), error="boom")
    grouped = nz.group_by_topic([_item("AI"), failed])
    assert sum(len(v) for v in grouped.values()) == 1


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def test_build_report_empty_when_no_newsletters():
    r = nz.build_report([])
    html = r.render()
    assert "No newsletters found" in html


def test_build_report_shows_worth_reading_section_and_cards():
    items = [_item("AI", "great one", score=5), _item("AI", "meh one", score=2)]
    r = nz.build_report(items)
    html = r.render()
    assert "Worth the full read" in html
    assert "great one" in html
    assert "meh one" in html
    assert "score 5" in html


def test_build_report_shows_failures_without_aborting():
    ok = _item("AI", "fine")
    failed = nz.Item(message=_message(subject="broke"), error="timeout")
    r = nz.build_report([ok, failed])
    html = r.render()
    assert "Extraction failures" in html
    assert "broke" in html and "timeout" in html


def test_build_report_ticker_present_for_top_scoring_items():
    items = [_item("AI", "headline story", score=5)]
    r = nz.build_report(items)
    assert any("headline story" in t for t in r.ticker)


# ---------------------------------------------------------------------------
# main(): end-to-end against fakes
# ---------------------------------------------------------------------------

class FakeGmail:
    def __init__(self, messages):
        self.messages = messages
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, *args, **kwargs):
        self.queries.append(query)
        return self.messages


def test_main_filters_receipts_query_and_non_newsletters_then_isolates_failures(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "lmsa.sqlite"
    out_dir = tmp_path / "output"
    monkeypatch.setenv("LMSA_DB", str(db_path))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out_dir))

    messages = [
        _message(subject="Weekly AI roundup", list_unsubscribe="<mailto:x>", list_id=""),
        _message(subject="Politics weekly", list_unsubscribe="", list_id="<pol.example.com>"),
        _message(subject="Your receipt", list_unsubscribe="", list_id=""),  # not a newsletter
        _message(subject="broken newsletter", list_unsubscribe="<mailto:y>", list_id=""),
    ]
    fake_gmail = FakeGmail(messages)
    monkeypatch.setattr(nz, "Gmail", lambda cfg=None: fake_gmail)

    lm = FakeLMStudio(
        results={
            "Subject: Weekly AI roundup": {
                "topic": "AI", "three_bullets": ["a", "b", "c"],
                "worth_full_read": 4, "why": "solid week",
            },
            "Subject: Politics weekly": {
                "topic": "Politics", "three_bullets": ["a", "b", "c"],
                "worth_full_read": 2, "why": "meh",
            },
        },
        fail_on={"Subject: broken newsletter"},
    )
    monkeypatch.setattr(nz, "LMStudio", lambda: lm)

    nz.main([])

    assert fake_gmail.queries == [nz.QUERY]
    assert lm.ensure_server_calls == 1
    # "Your receipt" has neither List-Id nor List-Unsubscribe -- filtered
    # out locally rather than sent to the model.
    assert not any("Your receipt" in p for p in lm.prompts)
    assert len(lm.prompts) == 3  # the 2 successes + the 1 isolated failure

    [written] = [p for p in out_dir.rglob("*.html") if p.name != "index.html"]
    html = written.read_text(encoding="utf-8")
    assert "Weekly AI roundup" in html
    assert "Politics weekly" in html
    assert "broken newsletter" in html  # shown in Extraction failures
    assert "Worth the full read" in html  # score 4 item

    with Store(db_path) as store:
        row = store.conn.execute(
            "SELECT status, summary_json FROM runs WHERE task='newsletter_zine'"
        ).fetchone()
    assert row[0] == "ok"
    summary = json.loads(row[1])
    assert summary == {
        "candidates": 4, "newsletters": 3, "summarized": 2, "failed": 1,
        "memo_hits": 0, "memo_misses": 3,
    }
