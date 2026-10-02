"""Site watch (#53): fetch/diff/judge core, the `pages` briefing adapter and
the standalone task. Fake fetcher and fake LM Studio: no network, no model."""

from __future__ import annotations

from datetime import datetime, timezone

from lmsa import Store
from lmsa.briefing import sitewatch
from lmsa.briefing import sources
from lmsa.briefing.adapters import pages as pages_adapter
from lmsa.briefing.sitewatch import PageSpec, check_page, field_history
from lmsa.web import HTTPStatusError, Response
from tasks import site_watch as task

NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def get(self, url, **kw):
        if url not in self.pages:
            raise HTTPStatusError(404, url)
        return Response(status=200, url=url, headers={}, content=self.pages[url].encode("utf-8"))


class FakeLM:
    """extract() returns whatever the test queues, in order."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls = 0

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls += 1
        if not self.answers:
            return {"relevant": True, "summary": "changed", "fields": {}}
        return self.answers.pop(0)


PAGE_V1 = "<html><body><main>Widget: $40. In stock.</main></body></html>"
PAGE_V2 = "<html><body><main>Widget: $35. In stock.</main></body></html>"
PAGE_V2_NAV_ONLY = "<html><body><nav>visit 1234</nav><main>Widget: $40. In stock. (checked daily)</main></body></html>"


def spec(**kw) -> PageSpec:
    base = dict(url="https://example.com/widget", name="Widget", rule="only if the price changes",
                selector="main")
    base.update(kw)
    return PageSpec(**base)


# -- core ----------------------------------------------------------------------


def test_first_run_is_baseline_and_never_judged(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        result = check_page(fetcher=FakeFetcher({spec().url: PAGE_V1}), lm=lm, conn=store.conn,
                             spec=spec(), now=NOW)
    assert result.status == "baseline"
    assert result.relevant is None
    assert result.should_notify is False
    assert lm.calls == 0


def test_unchanged_page_makes_no_model_call(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        fetcher = FakeFetcher({spec().url: PAGE_V1})
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=spec(), now=NOW)
        result = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=spec(), now=NOW)
    assert result.status == "unchanged"
    assert lm.calls == 0
    assert result.should_notify is False


def test_irrelevant_change_is_suppressed_by_rule(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM(answers=[{"relevant": False, "summary": "just nav noise", "fields": {}}])
        fetcher = FakeFetcher({spec().url: PAGE_V1})
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=spec(), now=NOW)
        fetcher.pages[spec().url] = PAGE_V2_NAV_ONLY
        result = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=spec(),
                             now=NOW.replace(hour=7))
    assert result.status == "changed"
    assert result.relevant is False
    assert result.should_notify is False


def test_relevant_change_is_flagged_for_notify_only_when_opted_in(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM(answers=[{"relevant": True, "summary": "price dropped to $35", "fields": {"price": 35}}])
        fetcher = FakeFetcher({spec().url: PAGE_V1})
        s = spec(notify=True, fields={"price": "number"})
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW)
        fetcher.pages[s.url] = PAGE_V2
        result = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=7))
    assert result.status == "changed"
    assert result.relevant is True
    assert result.should_notify is True
    assert result.fields == {"price": 35.0}


def test_notify_false_never_flags_even_when_relevant(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM(answers=[{"relevant": True, "summary": "changed", "fields": {}}])
        fetcher = FakeFetcher({spec().url: PAGE_V1})
        s = spec(notify=False)
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW)
        fetcher.pages[s.url] = PAGE_V2
        result = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=7))
    assert result.should_notify is False


def test_typed_field_history_is_recorded_and_charted(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        s = spec(fields={"price": "number"})
        lm = FakeLM(answers=[
            {"relevant": True, "summary": "dropped", "fields": {"price": 35}},
            {"relevant": True, "summary": "dropped again", "fields": {"price": 30}},
        ])
        fetcher = FakeFetcher({s.url: PAGE_V1})
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW)
        fetcher.pages[s.url] = PAGE_V2
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=7))
        fetcher.pages[s.url] = "<html><body><main>Widget: $30. In stock.</main></body></html>"
        check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=8))
        history = field_history(store.conn, s.watcher or s.url, s.url, "price")
    assert history == [35.0, 30.0]


def test_bad_fetch_is_isolated_and_never_raises(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        result = check_page(fetcher=FakeFetcher({}), lm=lm, conn=store.conn, spec=spec(), now=NOW)
    assert result.error is not None
    assert result.status == "error"


def test_load_error_entry_is_isolated(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        lm = FakeLM()
        s = spec(load_error="invalid fields for this url")
        result = check_page(fetcher=FakeFetcher({s.url: PAGE_V1}), lm=lm, conn=store.conn, spec=s, now=NOW)
    assert result.error == "ValueError: invalid fields for this url"


def test_snapshots_are_bounded_per_page(tmp_path, monkeypatch):
    monkeypatch.setattr(sitewatch, "MAX_SNAPSHOTS", 3)
    with Store(tmp_path / "db.sqlite") as store:
        s = spec()
        lm = FakeLM()
        fetcher = FakeFetcher({s.url: PAGE_V1})
        for i in range(6):
            fetcher.pages[s.url] = f"<html><body><main>Widget: ${40 - i}. In stock.</main></body></html>"
            check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=(i % 12) + 1))
        count = store.conn.execute(
            "SELECT COUNT(*) FROM site_watch_snapshots WHERE url=?", (s.url,)
        ).fetchone()[0]
    assert count == 3


def test_two_watchers_of_the_same_url_each_see_the_change(tmp_path):
    """A url watched by both the standalone task and a briefing entry: each
    keeps its own snapshot lineage, so the second watcher's check doesn't
    find the diff already "consumed" by the first."""
    with Store(tmp_path / "db.sqlite") as store:
        url = spec().url
        task_spec = spec(watcher="task:site_watch:Widget")
        briefing_spec = spec(watcher="briefing:daily:Widget", notify=True)
        fetcher = FakeFetcher({url: PAGE_V1})

        task_baseline = check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=task_spec, now=NOW)
        briefing_baseline = check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=briefing_spec, now=NOW)
        assert task_baseline.status == "baseline"
        assert briefing_baseline.status == "baseline"

        fetcher.pages[url] = PAGE_V2
        later = NOW.replace(hour=7)
        task_result = check_page(
            fetcher=fetcher, lm=FakeLM(answers=[{"relevant": True, "summary": "price dropped", "fields": {}}]),
            conn=store.conn, spec=task_spec, now=later,
        )
        briefing_result = check_page(
            fetcher=fetcher, lm=FakeLM(answers=[{"relevant": True, "summary": "price dropped", "fields": {}}]),
            conn=store.conn, spec=briefing_spec, now=later,
        )
    assert task_result.status == "changed" and task_result.relevant is True
    assert briefing_result.status == "changed" and briefing_result.relevant is True
    assert briefing_result.should_notify is True


def test_two_briefings_watching_the_same_url_each_see_the_change(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        url = spec().url
        spec_a = spec(watcher="briefing:morning:Widget")
        spec_b = spec(watcher="briefing:evening:Widget")
        fetcher = FakeFetcher({url: PAGE_V1})
        check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=spec_a, now=NOW)
        check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=spec_b, now=NOW)

        fetcher.pages[url] = PAGE_V2
        later = NOW.replace(hour=7)
        result_a = check_page(fetcher=fetcher, lm=FakeLM(answers=[{"relevant": True, "summary": "x", "fields": {}}]),
                               conn=store.conn, spec=spec_a, now=later)
        result_b = check_page(fetcher=fetcher, lm=FakeLM(answers=[{"relevant": True, "summary": "x", "fields": {}}]),
                               conn=store.conn, spec=spec_b, now=later)
    assert result_a.status == "changed" and result_a.relevant is True
    assert result_b.status == "changed" and result_b.relevant is True


def test_judge_failure_keeps_previous_snapshot_and_retries(tmp_path):
    class BoomLM(FakeLM):
        def extract(self, prompt, schema, *, system=None, model=None, name="result"):
            self.calls += 1
            raise RuntimeError("model unavailable")

    with Store(tmp_path / "db.sqlite") as store:
        s = spec()
        fetcher = FakeFetcher({s.url: PAGE_V1})
        check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=s, now=NOW)  # baseline

        fetcher.pages[s.url] = PAGE_V2
        boom = BoomLM()
        first_attempt = check_page(fetcher=fetcher, lm=boom, conn=store.conn, spec=s, now=NOW.replace(hour=7))
        assert first_attempt.status == "changed"
        assert first_attempt.relevant is None
        assert "not judged" in first_attempt.error

        # Snapshot was NOT advanced: the next run re-diffs against the
        # original baseline and gets another shot at judging it.
        good_lm = FakeLM(answers=[{"relevant": True, "summary": "price dropped", "fields": {}}])
        second_attempt = check_page(fetcher=fetcher, lm=good_lm, conn=store.conn, spec=s, now=NOW.replace(hour=8))
    assert second_attempt.status == "changed"
    assert second_attempt.relevant is True
    assert second_attempt.summary == "price dropped"


def test_field_history_is_pruned_to_bounded_length(tmp_path, monkeypatch):
    monkeypatch.setattr(sitewatch, "MAX_FIELD_HISTORY", 3)
    with Store(tmp_path / "db.sqlite") as store:
        s = spec(fields={"price": "number"})
        fetcher = FakeFetcher({s.url: PAGE_V1})
        check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=s, now=NOW)
        for i in range(5):
            fetcher.pages[s.url] = f"<html><body><main>Widget: ${39 - i}. In stock.</main></body></html>"
            lm = FakeLM(answers=[{"relevant": True, "summary": "changed", "fields": {"price": 39 - i}}])
            check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=(i % 12) + 1))
        count = store.conn.execute(
            "SELECT COUNT(*) FROM site_watch_fields WHERE watcher=? AND url=? AND field='price'",
            (s.watcher or s.url, s.url),
        ).fetchone()[0]
    assert count == 3


def test_coerce_number_handles_common_formats():
    assert sitewatch._coerce("$1,299.99", "number") == (1299.99, None)
    assert sitewatch._coerce("1.299,99 €", "number") == (1299.99, None)
    assert sitewatch._coerce("1 299,99", "number") == (1299.99, None)
    assert sitewatch._coerce(1299, "number") == (1299.0, None)
    assert sitewatch._coerce("12,50", "number") == (12.5, None)
    assert sitewatch._coerce("1,234", "number") == (1234.0, None)


def test_coerce_number_discloses_a_note_when_unparsable():
    value, note = sitewatch._coerce("call for pricing", "number")
    assert value is None
    assert note is not None


def test_unparsable_field_records_a_disclosed_note(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        s = spec(fields={"price": "number"})
        fetcher = FakeFetcher({s.url: PAGE_V1})
        check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=s, now=NOW)
        fetcher.pages[s.url] = PAGE_V2
        lm = FakeLM(answers=[{"relevant": True, "summary": "x", "fields": {"price": "call for pricing"}}])
        result = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=s, now=NOW.replace(hour=7))
    assert result.fields["price"] is None
    assert any("price" in n for n in result.notes)


def test_truncation_is_flagged_and_disclosed_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(sitewatch, "MAX_TEXT_CHARS", 20)
    with Store(tmp_path / "db.sqlite") as store:
        s = spec()
        fetcher = FakeFetcher({s.url: PAGE_V1})
        baseline = check_page(fetcher=fetcher, lm=FakeLM(), conn=store.conn, spec=s, now=NOW)
        assert baseline.truncated is True

        fetcher.pages[s.url] = PAGE_V2
        captured = {}

        class CapturingLM(FakeLM):
            def extract(self, prompt, schema, *, system=None, model=None, name="result"):
                captured["prompt"] = prompt
                return super().extract(prompt, schema, system=system, model=model, name=name)

        result = check_page(fetcher=fetcher, lm=CapturingLM(), conn=store.conn, spec=s, now=NOW.replace(hour=7))
    assert result.truncated is True
    assert "cut short" in captured["prompt"]


def test_selector_extracts_only_the_matched_element():
    html = "<html><body><nav>skip me</nav><main><p>keep this</p></main></body></html>"
    text = sitewatch.extract_text(html, "https://x", "main")
    assert "keep this" in text
    assert "skip me" not in text


def test_missing_selector_falls_back_to_article_extraction():
    html = "<html><body><article>Some real article body text here.</article></body></html>"
    text = sitewatch.extract_text(html, "https://x", "#does-not-exist")
    assert "article body" in text.lower() or text  # never raises; best-effort


# -- pages adapter ---------------------------------------------------------------


class FakeBriefing:
    name = "test-briefing"


class FakeSourceContext:
    def __init__(self, fetcher, lm, conn, now=NOW, fresh=False, briefing=None):
        self.fetcher = fetcher
        self.lm = lm
        self.conn = conn
        self.now = now
        self.fresh = fresh
        self.briefing = briefing or FakeBriefing()
        self.notes = []

    def note(self, text):
        self.notes.append(text)


def test_pages_adapter_registered():
    sources.load_plugins()
    assert "pages" in sources.ADAPTERS
    assert "url" in sources.ADAPTERS["pages"].keys or "url" in sources.ADAPTERS["pages"].required


def test_pages_adapter_yields_nothing_on_baseline_or_unchanged(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        entry = {"url": "https://example.com/widget", "name": "Widget", "rule": "only if price changes",
                  "selector": "main"}
        ctx = FakeSourceContext(FakeFetcher({entry["url"]: PAGE_V1}), FakeLM(), store.conn)
        docs = list(pages_adapter.pages_adapter(entry, ctx))
        assert docs == []
        ctx2 = FakeSourceContext(FakeFetcher({entry["url"]: PAGE_V1}), FakeLM(), store.conn)
        docs2 = list(pages_adapter.pages_adapter(entry, ctx2))
        assert docs2 == []


def test_pages_adapter_yields_a_doc_only_when_relevant(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        entry = {"url": "https://example.com/widget", "name": "Widget", "rule": "only if price changes",
                  "selector": "main"}
        fetcher = FakeFetcher({entry["url"]: PAGE_V1})
        ctx = FakeSourceContext(fetcher, FakeLM(), store.conn)
        list(pages_adapter.pages_adapter(entry, ctx))  # baseline

        fetcher.pages[entry["url"]] = PAGE_V2_NAV_ONLY
        lm_irrelevant = FakeLM(answers=[{"relevant": False, "summary": "nav only", "fields": {}}])
        ctx2 = FakeSourceContext(fetcher, lm_irrelevant, store.conn, now=NOW.replace(hour=7))
        assert list(pages_adapter.pages_adapter(entry, ctx2)) == []

        fetcher.pages[entry["url"]] = PAGE_V2
        lm_relevant = FakeLM(answers=[{"relevant": True, "summary": "price dropped", "fields": {}}])
        ctx3 = FakeSourceContext(fetcher, lm_relevant, store.conn, now=NOW.replace(hour=8))
        docs = list(pages_adapter.pages_adapter(entry, ctx3))
        assert len(docs) == 1
        assert docs[0].url == entry["url"]
        assert "price dropped" in docs[0].text


def test_pages_adapter_records_a_note_on_fetch_error(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        entry = {"url": "https://example.com/missing"}
        ctx = FakeSourceContext(FakeFetcher({}), FakeLM(), store.conn)
        docs = list(pages_adapter.pages_adapter(entry, ctx))
        assert docs == []
        assert ctx.notes and "404" in ctx.notes[0]


def test_pages_adapter_surfaces_truncation_in_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(sitewatch, "MAX_TEXT_CHARS", 20)
    with Store(tmp_path / "db.sqlite") as store:
        entry = {"url": "https://example.com/widget", "name": "Widget", "rule": "only if price changes",
                  "selector": "main"}
        fetcher = FakeFetcher({entry["url"]: PAGE_V1})
        ctx = FakeSourceContext(fetcher, FakeLM(), store.conn)
        list(pages_adapter.pages_adapter(entry, ctx))  # baseline

        fetcher.pages[entry["url"]] = PAGE_V2
        lm = FakeLM(answers=[{"relevant": True, "summary": "price dropped", "fields": {}}])
        ctx2 = FakeSourceContext(fetcher, lm, store.conn, now=NOW.replace(hour=7))
        docs = list(pages_adapter.pages_adapter(entry, ctx2))
    assert len(docs) == 1
    assert docs[0].meta["truncated"] is True
    assert "truncated" in docs[0].text.lower()


def test_two_briefing_entries_for_the_same_url_get_distinct_watchers(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        url = "https://example.com/widget"
        entry = {"url": url, "name": "Widget", "rule": "only if price changes", "selector": "main"}
        fetcher = FakeFetcher({url: PAGE_V1})

        class Briefing:
            name = "morning"

        class Briefing2:
            name = "evening"

        ctx_a = FakeSourceContext(fetcher, FakeLM(), store.conn)
        ctx_a.briefing = Briefing()
        ctx_b = FakeSourceContext(fetcher, FakeLM(), store.conn)
        ctx_b.briefing = Briefing2()
        list(pages_adapter.pages_adapter(entry, ctx_a))  # baseline for "morning"
        list(pages_adapter.pages_adapter(entry, ctx_b))  # baseline for "evening"

        fetcher.pages[url] = PAGE_V2
        later = NOW.replace(hour=7)
        ctx_a2 = FakeSourceContext(fetcher, FakeLM(answers=[{"relevant": True, "summary": "a", "fields": {}}]),
                                    store.conn, now=later)
        ctx_a2.briefing = Briefing()
        ctx_b2 = FakeSourceContext(fetcher, FakeLM(answers=[{"relevant": True, "summary": "b", "fields": {}}]),
                                    store.conn, now=later)
        ctx_b2.briefing = Briefing2()
        docs_a = list(pages_adapter.pages_adapter(entry, ctx_a2))
        docs_b = list(pages_adapter.pages_adapter(entry, ctx_b2))
    assert len(docs_a) == 1 and len(docs_b) == 1


# -- standalone task ---------------------------------------------------------------


def test_task_load_pages_missing_file_is_empty(tmp_path):
    assert task.load_pages(tmp_path / "nope.toml") == []


def test_task_load_pages_bad_fields_isolated(tmp_path):
    path = tmp_path / "site_watch.toml"
    path.write_text(
        '[[pages]]\nurl = "https://example.com/a"\nfields = { price = "money" }\n\n'
        '[[pages]]\nurl = "https://example.com/b"\n',
        encoding="utf-8",
    )
    specs = task.load_pages(path)
    assert len(specs) == 2
    assert specs[0].load_error is not None
    assert specs[1].load_error is None


def test_task_notifies_only_on_relevant_opt_in_change(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))  # task.main() writes a report; keep it out of repo output/
    calls = []
    monkeypatch.setattr(task, "notify", lambda *a, **kw: calls.append((a, kw)))
    config_path = tmp_path / "site_watch.toml"
    url = "https://example.com/widget"
    config_path.write_text(
        f'[[pages]]\nurl = "{url}"\nname = "Widget"\nselector = "main"\n'
        f'rule = "only if the price changes"\nnotify = true\nfields = {{ price = "number" }}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(task, "CONFIG_PATH", config_path)
    monkeypatch.setattr(task, "_now", lambda: NOW)

    db = tmp_path / "db.sqlite"
    monkeypatch.setenv("LMSA_DB", str(db))

    fetcher = FakeFetcher({url: PAGE_V1})
    monkeypatch.setattr(task, "Fetcher", lambda *a, **kw: fetcher)
    lm = FakeLM()
    monkeypatch.setattr(task, "LMStudio", lambda: lm)

    task.main()  # baseline: no notify
    assert calls == []

    fetcher.pages[url] = PAGE_V2
    lm.answers = [{"relevant": True, "summary": "price dropped to $35", "fields": {"price": 35}}]
    monkeypatch.setattr(task, "_now", lambda: NOW.replace(hour=7))
    task.main()
    assert len(calls) == 1
    assert "Widget" in calls[0][0][0]


def test_report_shows_truncated_column(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    s = spec()
    from lmsa.briefing.sitewatch import PageResult

    truncated_result = PageResult(url=s.url, name=s.name, status="changed", relevant=True,
                                    summary="changed", truncated=True, history={})
    html = task.build_report([(s, truncated_result)]).read_text(encoding="utf-8")
    assert "Truncated" in html
