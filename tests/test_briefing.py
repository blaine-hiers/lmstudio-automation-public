"""Briefing engine (#52): config validation, pre-filter, clustering/ranking,
quote verification, memo reuse and NEW / UPDATED / ONGOING / dropped states.
Fake sources and a fake LM Studio: no network, no model."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lmsa import Store
from lmsa.briefing import config as bconfig
from lmsa.briefing import engine, sources
from lmsa.briefing.quotes import verify_quote
from lmsa.search import SearchResult
from lmsa.web import HTTPStatusError

FIX = Path(__file__).parent / "fixtures" / "briefing"
NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
BBC = "https://feeds.bbci.co.uk/news/world/rss.xml"
GCAPTAIN = "https://gcaptain.com/feed/"
SEARCH_URL = "https://news.example.org/suez-backs-up"

TOPICS = ["suez", "rotterdam", "hormuz", "freight", "panama", "football", "election"]


class FakeLM:
    """Bag-of-topics embeddings; extract() builds an answer from the prompt
    (one real quote lifted from the text, one fabricated)."""

    def __init__(self):
        self.extract_calls = 0
        self.embedded = 0

    def embed(self, texts, *, model=None):
        self.embedded += len(texts)
        out = []
        for t in texts:
            low = t.lower()
            out.append([float(low.count(w) > 0) for w in TOPICS] + [0.05])
        return out

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.extract_calls += 1
        titles = re.findall(r"^TITLE: (.+)$", prompt, re.M)
        heads = re.findall(r"^\[\d+\] .*? \| .*? \| (\S+)$", prompt, re.M)
        low = prompt.lower()
        real = re.search(r"[“\"]([^“”\"]{20,200})[”\"]", prompt)
        quotes = []
        if real:
            # cite the doc whose block contains the quote
            pos = real.start()
            url = heads[0]
            for m in re.finditer(r"^\[\d+\] .*? \| .*? \| (\S+)$", prompt, re.M):
                if m.start() < pos:
                    url = m.group(1)
            quotes.append({"quote": real.group(1), "source_url": url})
        quotes.append({"quote": "We will never reopen this waterway to anyone.", "source_url": heads[0]})
        countries = [c for w, c in (("suez", "EG"), ("rotterdam", "NL"), ("hormuz", "IR"), ("panama", "PA"))
                     if w in low] + ["XX"]
        return {
            "headline": f"Story: {titles[0]}", "summary": f"{len(titles)} sources. " + " / ".join(titles),
            "highlights": ["one", "two"], "key_quotes": quotes,
            "timeline": [{"date": "2026-09-24", "event": "It happened"}],
            "why_it_matters": "Because shipping.", "countries": countries, "actors": ["Canal Authority"],
            "severity": 7,
        }


class FakeFetcher:
    def __init__(self, pages: dict[str, bytes]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url not in self.pages:
            raise HTTPStatusError(404, url)
        return SimpleNamespace(content=self.pages[url], url=url, status=200, headers={}, from_cache=False)


class FakeSearx:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, **kw):
        self.queries.append((query, kw))
        return list(self.results)


def fixture_pages(bbc: bytes | None = None, gcaptain: bytes | None = None) -> dict[str, bytes]:
    return {
        BBC: bbc if bbc is not None else (FIX / "bbc_world.xml").read_bytes(),
        GCAPTAIN: gcaptain if gcaptain is not None else (FIX / "gcaptain.xml").read_bytes(),
        SEARCH_URL: (FIX / "suez_article.html").read_bytes(),
    }


def search_results():
    return [SearchResult(title="Suez Canal traffic backs up as ship stays stuck", url=SEARCH_URL,
                         snippet="Queues grow at Suez.", engine="bing",
                         published=datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc))]


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    return tmp_path / "out"


def run(cfg, db_path, lm, pages, *, now=NOW, results=None, fresh=False):
    with Store(db_path) as store, store.record_run("briefing") as r:
        res = engine.run_briefing(cfg, lm=lm, conn=store.conn, run_id=r.id, fetcher=FakeFetcher(pages), now=now,
                                  searxng_factory=lambda: FakeSearx(results if results is not None else search_results()),
                                  fresh=fresh)
        report = engine.build_report(res)
        return res, report.render()


def by_word(res, word):
    return next(s for s in res.stories if word in s.headline.lower())


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_example_config_end_to_end_states_and_memo(tmp_path, out_dir):
    cfg = bconfig.load("example")
    db = tmp_path / "db.sqlite"
    lm = FakeLM()

    res1, html1 = run(cfg, db, lm, fixture_pages())
    # 7 feed items + 1 search result; football + election fail the keyword pre-filter.
    assert res1.counts["collected"] == 8
    assert res1.counts["prefiltered_out"] == 2
    # Suez: BBC + gCaptain + the search result (article fetched by follow_links) -> one story.
    suez = by_word(res1, "suez")
    assert len(suez.docs) == 3 and len(suez.sources) == 3
    # Same interest similarity (none of their topics is in the interests): the
    # bigger, more diverse cluster ranks higher. Freight matches an interest.
    order = [s.story_id for s in res1.stories]
    for w in ("rotterdam", "hormuz"):
        assert order.index(suez.story_id) < order.index(by_word(res1, w).story_id)
    assert res1.stories[0] is by_word(res1, "freight") or res1.stories[0] is suez
    assert len(res1.stories) == 4
    assert {s.status for s in res1.stories} == {"new"}
    assert lm.extract_calls == 4
    assert any(d.meta.get("fetched") for d in suez.docs)
    # model output sanitized: severity clamped, bogus country code dropped
    assert suez.extract["severity"] == 3 and suez.countries == ["EG"]
    # quotes: the real one verifies, the fabricated one is flagged
    assert [q["verified"] for q in suez.quotes] == [True, False]
    assert "unverified" in html1 and "NEW" in html1 and "caught up" in html1
    assert "Read time" in html1

    # Second run, same input: no LLM calls, everything ONGOING, nothing dropped.
    res2, _ = run(cfg, db, lm, fixture_pages())
    assert lm.extract_calls == 4
    assert res2.memo.hits == 4 and res2.memo.misses == 0
    assert {s.status for s in res2.stories} == {"ongoing"}
    assert res2.dropped == []
    assert by_word(res2, "suez").story_id == suez.story_id

    # Third run: Rotterdam gone, a new Suez source, a new Panama story.
    bbc = (FIX / "bbc_world.xml").read_text(encoding="utf-8")
    bbc = re.sub(r"<item><title>Rotterdam.*?</item>", "", bbc, flags=re.S)
    bbc = bbc.replace("</channel>", (
        "<item><title>Suez Canal reopens partially as ship shifts</title>"
        "<link>https://www.bbc.co.uk/news/world-suez-5</link><pubDate>Fri, 25 Sep 2026 04:00:00 GMT</pubDate>"
        "<description>The Suez Canal partially reopened on Friday after the ship shifted, the canal authority said.</description></item>"
        "<item><title>Drought limits Panama Canal transits</title>"
        "<link>https://www.bbc.co.uk/news/world-panama-6</link><pubDate>Fri, 25 Sep 2026 03:00:00 GMT</pubDate>"
        "<description>Low water in the Panama Canal cut daily transits, raising shipping costs for container lines.</description></item>"
        "</channel>"))
    res3, html3 = run(cfg, db, lm, fixture_pages(bbc=bbc.encode("utf-8")))
    states = {w: by_word(res3, w).status for w in ("suez", "hormuz", "freight", "panama")}
    assert states == {"suez": "updated", "hormuz": "ongoing", "freight": "ongoing", "panama": "new"}
    assert lm.extract_calls == 6  # only Suez (changed) and Panama (new)
    assert by_word(res3, "suez").story_id == suez.story_id
    assert [d["headline"] for d in res3.dropped] == [by_word(res1, "rotterdam").headline]
    assert "Dropped since last run" in html3 and "UPDATED" in html3

    # Persistence: one row per story / article, vectors in the briefings namespace.
    with Store(db) as store:
        n_stories = store.conn.execute("SELECT COUNT(*) FROM briefing_stories WHERE briefing='example'").fetchone()[0]
        assert n_stories == 5  # 4 from run 1 + Panama
        item = store.conn.execute("SELECT item_id FROM briefing_stories WHERE story_id=?", (suez.story_id,)).fetchone()[0]
        assert item == f"example:story:{suez.story_id}"
        ns = {r[0] for r in store.conn.execute("SELECT item_id FROM embeddings WHERE ns='briefings'")}
        assert item in ns
        art = store.conn.execute("SELECT item_id, lean, reliability FROM briefing_articles WHERE url=?",
                                 ("https://feeds.bbci.co.uk/news/world/rss.xml",)).fetchone()
        assert art is None  # articles are keyed by their own URL, not the feed's
        arts = store.conn.execute("SELECT item_id, lean, reliability FROM briefing_articles "
                                  "WHERE source='BBC World'").fetchall()
        assert arts and all(a[1:] == ("intl", "high") for a in arts)
        assert all(a[0] in ns for a in arts)
        runs = store.conn.execute("SELECT COUNT(*) FROM briefing_runs WHERE briefing='example'").fetchone()[0]
        assert runs == 3


def test_keyword_prefilter_skips_llm_calls(tmp_path, out_dir):
    cfg = bconfig.load("example")
    on = FakeLM()
    run(cfg, tmp_path / "a.sqlite", on, fixture_pages())
    cfg.prefilter = False
    off = FakeLM()
    res, _ = run(cfg, tmp_path / "b.sqlite", off, fixture_pages())
    assert off.extract_calls == on.extract_calls + 2  # football + election stories
    assert off.embedded > on.embedded
    assert res.counts["prefiltered_out"] == 0


def test_prefilter_false_source_keeps_title_only_docs_and_others_are_filtered():
    """#128: a title-only feed can't match keywords before follow_links runs."""
    data = {"briefing": {"title": "T", "interests": ["armed conflict"], "keywords": ["sabotage"]},
            "sources": {"feeds": [
                {"name": "Wire", "url": "https://wire.example/feed", "prefilter": False},
                {"name": "Other", "url": "https://other.example/feed"}]}}
    cfg = bconfig.parse("t", data)
    entries = {e["name"]: (k, e) for k, e in cfg.entries()}
    assert entries["Wire"][1]["prefilter"] is False

    def doc(entry, n):
        d = sources.Doc(url=f"https://{entry}.example/{n}", title="The Wire - September 29, 2026",
                        text="Watch on YouTube")
        return engine.normalize_doc(d, *entries[entry])

    docs = [doc("Wire", 1), doc("Wire", 2), doc("Other", 1)]
    kept, dropped = engine.prefilter(cfg, docs)
    assert [d.meta["entry"] for d in kept] == ["Wire", "Wire"]
    assert dropped == 1


def test_source_status_when_keyword_prefilter_is_off_for_the_source():
    from lmsa.briefing.adapters import theaters
    msg = theaters.source_status({"taken": 10, "error": None}, 0, "feeds: W", [], keyword_filtered=False)
    assert "filtered by keywords" not in msg and "10" in msg


def test_fresh_ignores_memo(tmp_path, out_dir):
    cfg = bconfig.load("example")
    lm = FakeLM()
    run(cfg, tmp_path / "db.sqlite", lm, fixture_pages())
    run(cfg, tmp_path / "db.sqlite", lm, fixture_pages(), fresh=True)
    assert lm.extract_calls == 8


def test_bad_source_is_a_note_not_a_crash(tmp_path, out_dir):
    cfg = bconfig.load("example")
    pages = fixture_pages()
    del pages[GCAPTAIN]
    pages[BBC] = pages[BBC]  # BBC still fine

    class BrokenSearx:
        def search(self, *a, **kw):
            raise RuntimeError("searx down")

    lm = FakeLM()
    with Store(tmp_path / "db.sqlite") as store, store.record_run("briefing") as r:
        res = engine.run_briefing(cfg, lm=lm, conn=store.conn, run_id=r.id, fetcher=FakeFetcher(pages), now=NOW,
                                  searxng_factory=BrokenSearx)
    assert any("gCaptain" in n and "HTTPStatusError" in n for n in res.notes)
    assert any("searx down" in n for n in res.notes)
    assert {"suez", "rotterdam"} <= {w for w in ("suez", "rotterdam") for s in res.stories if w in s.headline.lower()}


def test_extract_failure_isolated_to_one_story(tmp_path, out_dir):
    cfg = bconfig.load("example")

    class Flaky(FakeLM):
        def extract(self, prompt, schema, **kw):
            if "Hormuz" in prompt:
                raise TimeoutError("model timed out")
            return super().extract(prompt, schema, **kw)

    res, html = run(cfg, tmp_path / "db.sqlite", Flaky(), fixture_pages())
    hormuz = next(s for s in res.stories if "hormuz" in s.docs[0].title.lower())
    assert hormuz.error and hormuz.extract is None
    assert len(res.stories) == 4 and "could not be summarized" in html


def test_pages_source_without_adapter_is_reported(tmp_path, out_dir):
    # #53 registers a real "pages" adapter; simulate it being absent (an
    # older engine, or a plugin import failure) without permanently
    # unregistering it for the rest of the suite.
    data = {
        "briefing": {"title": "T", "interests": ["ports"], "keywords": ["port"]},
        "sources": {"pages": [{"url": "https://example.com/notices", "selector": "#main"}]},
    }
    saved = sources.ADAPTERS.pop("pages", None)
    try:
        cfg = bconfig.parse("t", data)
        res, html = run(cfg, tmp_path / "db.sqlite", FakeLM(), {})
        assert any("not available" in n for n in res.notes)
        assert "No stories this run" in html
    finally:
        if saved is not None:
            sources.ADAPTERS["pages"] = saved


def test_registered_api_adapter_and_extension(tmp_path, out_dir):
    from lmsa.report import StoryExtra

    @sources.register("fake_api", keys=("topic",), required=("topic",))
    def fake_api(entry, ctx):
        assert ctx.limit == 6  # the entry's limit + 1, so a cut is detectable
        for i in range(10):  # ignores the limit: the engine caps it
            yield sources.Doc(url=f"https://api.example/{i}", title=f"Suez item {i} {entry['topic']}",
                              text="suez canal", published=NOW)
        yield "not a doc"

    def enrich(stories, ctx):
        for s in stories:
            s.extras["odds"] = 0.42

    def render(story, ctx):
        return [StoryExtra("market odds", f"{story.extras['odds']:.0%}", "javascript:alert(1)")]

    sources.register_extension("odds_test", enrich=enrich, render=render)
    try:
        data = {"briefing": {"title": "T", "interests": ["suez"], "extensions": ["odds_test"]},
                "sources": {"api": [{"adapter": "fake_api", "topic": "x", "limit": 5, "lean": "center"}]}}
        cfg = bconfig.parse("t", data)
        res, html = run(cfg, tmp_path / "db.sqlite", FakeLM(), {})
        assert res.counts["collected"] == 5
        assert any("cut at 5 items by its limit (5)" in n for n in res.notes)
        assert "42%" in html and "javascript:" not in html
        with Store(tmp_path / "db.sqlite") as store:
            data_json = store.conn.execute("SELECT data_json FROM briefing_stories").fetchone()[0]
        assert json.loads(data_json)["extras"] == {"odds": 0.42}
    finally:
        sources.ADAPTERS.pop("fake_api", None)
        sources.EXTENSIONS.pop("odds_test", None)


def test_regions_filter_drops_stories_elsewhere(tmp_path, out_dir):
    cfg = bconfig.load("example")
    cfg.regions = ["EG"]
    res, _ = run(cfg, tmp_path / "db.sqlite", FakeLM(), fixture_pages())
    # untagged stories stay (no country to judge by); tagged ones must be in EG
    assert all(s.countries in ([], ["EG"]) for s in res.stories)
    assert {"suez", "freight"} == {w for w in ("suez", "freight", "hormuz", "rotterdam")
                                   for s in res.stories if w in s.headline.lower()}
    assert any("outside regions" in n for n in res.notes)


def _count_adapter(n_items):
    @sources.register("count_api", keys=("n",))
    def count_api(entry, ctx):
        for i in range(min(n_items, ctx.limit)):  # a well-behaved adapter: honours ctx.limit exactly
            yield sources.Doc(url=f"https://api.example/{entry['name']}/{i}", title=f"item {i}", text="x")
    return count_api


@pytest.mark.parametrize("n_items, limit, max_items, expect_note", [
    (10, 3, 300, "cut at 3 items by its limit (3)"),
    (10, None, 4, "cut at 4 items by max_items (4)"),
    (3, 3, 300, None),   # exactly the limit available: nothing was cut, no note
    (4, None, 4, None),
])
def test_caps_that_bite_are_noted_even_with_a_well_behaved_adapter(n_items, limit, max_items, expect_note):
    _count_adapter(n_items)
    try:
        entry = {"adapter": "count_api", "name": "a"}
        if limit:
            entry["limit"] = limit
        cfg = bconfig.parse("t", {"briefing": {"title": "T", "interests": ["x"], "max_items": max_items},
                                  "sources": {"api": [entry]}})
        ctx = sources.SourceContext(briefing=cfg, fetcher=None, lm=None, conn=None, now=NOW, limit=0)
        docs = engine.collect(cfg, ctx)
        assert len(docs) == min(n_items, limit or max_items, max_items)
        if expect_note:
            assert any(expect_note in n for n in ctx.notes), ctx.notes
        else:
            assert not any("cut at" in n for n in ctx.notes), ctx.notes
    finally:
        sources.ADAPTERS.pop("count_api", None)


def test_max_items_note_names_the_sources_left_out():
    _count_adapter(5)
    try:
        cfg = bconfig.parse("t", {"briefing": {"title": "T", "interests": ["x"], "max_items": 5},
                                  "sources": {"api": [{"adapter": "count_api", "name": "a"},
                                                      {"adapter": "count_api", "name": "b"}]}})
        ctx = sources.SourceContext(briefing=cfg, fetcher=None, lm=None, conn=None, now=NOW, limit=0)
        assert len(engine.collect(cfg, ctx)) == 5
        assert any("max_items (5) reached: 1 later source(s)" in n for n in ctx.notes), ctx.notes
    finally:
        sources.ADAPTERS.pop("count_api", None)


def test_failed_extract_keeps_the_old_fingerprint_so_a_change_still_shows(tmp_path, out_dir):
    cfg = bconfig.load("example")
    db = tmp_path / "db.sqlite"
    lm = FakeLM()
    res1, _ = run(cfg, db, lm, fixture_pages())
    suez = by_word(res1, "suez")
    with Store(db) as store:
        fp1 = store.conn.execute("SELECT fingerprint FROM briefing_stories WHERE story_id=?",
                                 (suez.story_id,)).fetchone()[0]
    # Run 2: a new Suez source arrives while the model is down for Suez.
    bbc = (FIX / "bbc_world.xml").read_text(encoding="utf-8").replace("</channel>", (
        "<item><title>Suez Canal reopens partially as ship shifts</title>"
        "<link>https://www.bbc.co.uk/news/world-suez-5</link><pubDate>Fri, 25 Sep 2026 04:00:00 GMT</pubDate>"
        "<description>The Suez Canal partially reopened, the canal authority said.</description></item></channel>"))
    pages = fixture_pages(bbc=bbc.encode("utf-8"))

    class SuezDown(FakeLM):
        def extract(self, prompt, schema, **kw):
            if "Suez" in prompt:
                raise TimeoutError("model timed out")
            return super().extract(prompt, schema, **kw)

    run(cfg, db, SuezDown(), pages)
    with Store(db) as store:
        fp2, headline = store.conn.execute("SELECT fingerprint, headline FROM briefing_stories WHERE story_id=?",
                                           (suez.story_id,)).fetchone()
    assert fp2 == fp1 and headline == suez.headline  # the last good state is kept
    # Run 3: model back, same docs as run 2 -> the story is UPDATED, not "ongoing".
    res3, _ = run(cfg, db, FakeLM(), pages)
    assert by_word(res3, "suez").status == "updated"


def test_plugin_adapters_register_before_config_validation(tmp_path, monkeypatch):
    """A briefing using an adapter/extension from lmsa/briefing/adapters/ must
    validate even when nothing has called load_plugins() yet."""
    import sys

    from lmsa.briefing import adapters as pkg

    plug = tmp_path / "plugs"
    plug.mkdir()
    (plug / "tmp_plugin_52.py").write_text(
        "from lmsa.briefing.sources import register, register_extension\n"
        "@register('tmp_plugin_52', keys=('query',), required=('query',))\n"
        "def adapter(entry, ctx):\n    return []\n"
        "register_extension('tmp_ext_52')\n", encoding="utf-8")
    (plug / "_skipped.py").write_text("raise RuntimeError('never imported')\n", encoding="utf-8")
    monkeypatch.setattr(pkg, "__path__", [*pkg.__path__, str(plug)])
    monkeypatch.setattr(sources, "_loaded", False)
    monkeypatch.setattr(sources, "PLUGIN_ERRORS", [])
    try:
        d = tmp_path / "briefings"
        d.mkdir()
        (d / "mil.toml").write_text(
            '[briefing]\ntitle = "Mil"\ninterests = ["x"]\nextensions = ["tmp_ext_52"]\n'
            '[[sources.api]]\nadapter = "tmp_plugin_52"\nquery = "q"\n', encoding="utf-8")
        cfg = bconfig.load("mil", d)  # first config parse imports the plugins
        assert cfg.sources["api"][0]["query"] == "q" and cfg.extensions == ["tmp_ext_52"]
        assert sources.PLUGIN_ERRORS == []
    finally:
        sources.ADAPTERS.pop("tmp_plugin_52", None)
        sources.EXTENSIONS.pop("tmp_ext_52", None)
        sys.modules.pop("lmsa.briefing.adapters.tmp_plugin_52", None)


def test_extension_can_badge_sources_and_add_a_coverage_bar(tmp_path, out_dir):
    from lmsa.report import StackedBar, StoryExtra, StorySource

    def render_sources(story, links, ctx):
        by_url = {d.url: d for d in story.docs}
        out = []
        for link in links:
            d = by_url.get(link.href)
            rel = d.meta.get("reliability", "unknown") if d else None
            out.append(StorySource(link.text, link.href, badge=rel, tone="acc" if rel == "high" else "bogus"))
        return out

    def render(story, ctx):
        leans = [d.meta.get("lean", "unknown") for d in story.docs]
        return [StackedBar([(x, leans.count(x), "info") for x in sorted(set(leans))], label="coverage by lean"),
                StackedBar([("broken",)]),  # malformed: dropped, never breaks the report
                StoryExtra("n", str(len(leans))), "not a component"]

    sources.register_extension("balance_test", render=render, render_sources=render_sources, default=True)
    try:
        cfg = bconfig.load("example")
        res, html = run(cfg, tmp_path / "db.sqlite", FakeLM(), fixture_pages())
        assert 'class="stacked"' in html and "coverage by lean" in html
        assert "intl 33%" in html  # Suez: BBC (intl) + gCaptain + search (unknown)
        assert '<span class="tag t-acc">high</span>' in html
        assert '<span class="tag t-muted">unknown</span>' in html  # bad tone fell back to muted
        assert not any("balance_test" in n for n in res.notes)
    finally:
        sources.EXTENSIONS.pop("balance_test", None)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_cluster_threshold_and_rank():
    v = np.asarray([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.98, 0.1]], dtype=np.float32)
    assert engine.cluster(v, 0.9) == [[0, 1], [2, 3]]
    assert engine.cluster(v, 0.9999) == [[0], [1], [2], [3]]
    fresh = engine.recency_decay(NOW, NOW, 36)
    old = engine.recency_decay(datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc), NOW, 36)
    assert fresh == 1.0 and old == pytest.approx(0.5)
    assert engine.story_score(0.8, 4, 3, 1.0) > engine.story_score(0.8, 1, 1, 1.0)
    assert engine.story_score(0.8, 1, 1, 1.0) > engine.story_score(0.8, 1, 1, 0.5)
    assert engine.story_score(0.9, 1, 1, 1.0) > engine.story_score(0.5, 1, 1, 1.0)


def test_quote_verification():
    src = ("The canal authority chairman said: “Traffic will resume within 48 hours once the vessel "
           "is refloated.” Dozens of ships wait.")
    assert verify_quote("Traffic will resume within 48 hours once the vessel is refloated.", src)
    assert verify_quote('"traffic will  resume within 48 hours, once the vessel is refloated"', src)  # fuzzy
    assert not verify_quote("We will never reopen this waterway to anyone.", src)
    assert not verify_quote("Traffic will close for a month while engineers dredge the channel.", src)
    assert not verify_quote("the vessel", src)  # too short to mean anything
    assert not verify_quote("anything at all here", "")
    # an ellipsis marks an omission: the parts must appear in order
    assert verify_quote("Traffic will resume ... once the vessel is refloated", src)
    assert not verify_quote("once the vessel is refloated ... Traffic will resume", src)
    # articles are the only words that may differ
    assert verify_quote("Traffic will resume within 48 hours once a vessel is refloated.", src)


# The reviewer's probes: near-identical strings with the opposite meaning.
MEANING_FLIPS = [
    ("We will withdraw our troops from the region by March.",
     "The minister said: we will not withdraw our troops from the region by March."),
    ("The ceasefire is holding across the whole front line tonight.",
     "Officials said the ceasefire is not holding across the whole front line tonight."),
    ("We did authorize the strike on the convoy.", "We did not authorize the strike on the convoy, he said."),
    ("There is no evidence that the vaccine caused the deaths.",
     "There is evidence that the vaccine caused the deaths, the report found."),
    ("The company expects profits to rise sharply next year.", "The company expects profits to fall sharply next year."),
    ("We will never negotiate with them", "We will negotiate with them"),
    ("Casualties rose to 150 in the city", "Casualties rose to 15 in the city"),
    ("unemployment increased 3.5 percent last month", "unemployment decreased 3.5 percent last month"),
]


@pytest.mark.parametrize("quote, source", MEANING_FLIPS)
def test_meaning_flipped_quotes_are_unverified(quote, source):
    assert not verify_quote(quote, source)


def test_numbers_must_match_exactly():
    src = "Unemployment increased 3.5 percent last month, to 1,200 claims."
    assert verify_quote("unemployment increased 3.5 percent last month", src)
    assert not verify_quote("unemployment increased 3.6 percent last month", src)
    assert verify_quote("increased 3.5 percent last month, to 1,200 claims", src)
    assert not verify_quote("increased 3.5 percent last month, to 1,300 claims", src)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

BASE = {"briefing": {"title": "T", "interests": ["ports"]},
        "sources": {"feeds": [{"url": "https://example.com/feed.xml"}]}}


def _with(**changes):
    data = json.loads(json.dumps(BASE))
    for path, value in changes.items():
        table, key = path.split("__")
        if table == "feed":
            data["sources"]["feeds"][0][key] = value
        else:
            data[table][key] = value
    return data


@pytest.mark.parametrize("data, needle", [
    (_with(briefing__colour="red"), "briefing.colour"),
    (_with(feed__colour="red"), "sources.feeds[0].colour"),
    (_with(briefing__regions=["ZZ"]), "'ZZ'"),
    (_with(briefing__regions=["egypt"]), "'egypt'"),
    (_with(feed__lean="far-out"), "lean"),
    (_with(feed__reliability="great"), "reliability"),
    (_with(feed__url="ftp://x"), "http(s)"),
    (_with(feed__prefilter="no"), "prefilter"),
    (_with(briefing__max_stories=0), "max_stories"),
    (_with(briefing__cluster_threshold=1.5), "cluster_threshold"),
    (_with(briefing__extensions=["nope"]), "nope"),
    ({"briefing": {"title": "T", "interests": ["x"]}}, "no sources"),
    ({"briefing": {"title": "T", "interests": ["x"]}, "sources": {"feeds": []}}, "no sources"),
    ({"briefing": {"title": "T", "interests": ["x"]}, "sources": {"rss": [{"url": "https://x"}]}}, "sources.rss"),
    ({"briefing": {"title": "T"}, "sources": BASE["sources"]}, "interests"),
    ({**BASE, "extra": {}}, "'extra'"),
    ({"briefing": {"title": "T", "interests": ["x"]}, "sources": {"api": [{"adapter": "nope"}]}}, "'nope'"),
    ({"briefing": {"title": "T", "interests": ["x"]}, "sources": {"search": [{"query": "q", "time_range": "decade"}]}},
     "time_range"),
])
def test_config_validation_errors(data, needle):
    with pytest.raises(bconfig.ConfigError) as exc:
        bconfig.parse("t", data)
    assert needle in str(exc.value)


def test_config_defaults_and_example_is_valid():
    cfg = bconfig.parse("t", _with(briefing__regions=["eg"]))
    assert cfg.regions == ["EG"] and cfg.enabled is None and cfg.max_stories == 20
    entry = cfg.sources["feeds"][0]
    assert entry["lean"] == "unknown" and entry["reliability"] == "unknown"
    for name in bconfig.names():
        bconfig.load(name)  # every shipped briefing file validates
    with pytest.raises(bconfig.ConfigError):
        bconfig.path_for("../etc")


# ---------------------------------------------------------------------------
# Local on/off overrides (#95): config/local/briefings.toml never touches
# the shipped config/briefings/<name>.toml.
# ---------------------------------------------------------------------------

def test_local_override_wins_over_shipped_enabled_which_wins_over_new(tmp_path):
    local = tmp_path / "briefings.toml"
    shipped_off = dict(BASE, briefing={**BASE["briefing"], "enabled": False})

    local.write_text("[briefings]\nt = true\n", encoding="utf-8")
    assert bconfig.parse("t", shipped_off, local_path=local).enabled is True   # local beats shipped False
    assert bconfig.parse("t", BASE, local_path=local).enabled is True         # local beats "never reviewed"

    local.write_text("[briefings]\nother = true\n", encoding="utf-8")
    assert bconfig.parse("t", shipped_off, local_path=local).enabled is False  # no local entry: shipped stands
    assert bconfig.parse("t", BASE, local_path=local).enabled is None          # neither: still NEW

    missing = tmp_path / "does-not-exist.toml"
    assert bconfig.parse("t", shipped_off, local_path=missing).enabled is False
    assert bconfig.local_overrides(missing) == {}


def test_malformed_local_override_is_a_warning_never_a_crash(tmp_path, capsys):
    local = tmp_path / "briefings.toml"
    for bad in ("not valid toml [[[", "enabled = true\n", "[briefings]\nt = \"yes\"\n",
                "[briefings]\nBAD NAME = true\n", "[other]\nx = 1\n"):
        local.write_text(bad, encoding="utf-8")
        assert bconfig.local_overrides(local) == {}
        assert "warning" in capsys.readouterr().err
        # a briefing still loads fine -- one bad local file never takes out every briefing
        cfg = bconfig.parse("t", BASE, local_path=local)
        assert cfg.enabled is None and cfg.title == "T"


def test_shipped_configs_still_validate_and_stay_untouched_with_a_local_override(tmp_path, monkeypatch):
    """Acceptance (#95): the shipped-config tests keep passing even when a
    local file enables briefings -- and the shipped file itself never
    changes, so `git status` stays clean."""
    local = tmp_path / "briefings.toml"
    local.write_text("[briefings]\nexample = true\nconflicts = true\n", encoding="utf-8")
    monkeypatch.setattr(bconfig, "LOCAL_OVERRIDES_PATH", local)

    example_before = (bconfig.BRIEFINGS_DIR / "example.toml").read_bytes()
    conflicts_before = (bconfig.BRIEFINGS_DIR / "conflicts.toml").read_bytes()

    cfg = bconfig.load("example")
    assert cfg.enabled is True and cfg.title == "Example briefing"  # ships `enabled = false`, overridden on
    cfg2 = bconfig.load("conflicts")
    assert cfg2.enabled is True and cfg2.extensions == ["theaters", "markets"]

    assert (bconfig.BRIEFINGS_DIR / "example.toml").read_bytes() == example_before
    assert (bconfig.BRIEFINGS_DIR / "conflicts.toml").read_bytes() == conflicts_before
    assert "\nenabled" not in conflicts_before.decode("utf-8")


# ---------------------------------------------------------------------------
# Task: scheduling and --list
# ---------------------------------------------------------------------------

def test_due_respects_enabled_and_every_hours():
    from tasks import briefing as task

    cfg = bconfig.parse("t", BASE)
    assert not task.due(cfg, None, NOW)  # never reviewed: not scheduled
    cfg.enabled = True
    assert task.due(cfg, None, NOW)
    assert not task.due(cfg, datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc), NOW)
    assert task.due(cfg, datetime(2026, 9, 24, 10, 30, tzinfo=timezone.utc), NOW)


def test_list_and_main_with_name(tmp_path, out_dir, monkeypatch, capsys):
    from tasks import briefing as task

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    task.main(["--list"])
    assert "example" in capsys.readouterr().out

    lm = FakeLM()
    lm.ensure_server = lambda: None
    monkeypatch.setattr(task, "LMStudio", lambda: lm)
    monkeypatch.setattr(task, "make_fetcher", lambda: FakeFetcher(fixture_pages()))
    monkeypatch.setattr(task, "_now", lambda: NOW)
    import lmsa.briefing.sources as src
    monkeypatch.setattr(src.SourceContext, "searxng", lambda self: FakeSearx(search_results()))
    task.main(["--name", "example"])
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert Path(printed).name.startswith("briefing-example-") and Path(printed).exists()
    # no --name: example ships disabled, so nothing is due
    task.main([])
    assert "no briefing is enabled and due" in capsys.readouterr().out


def test_list_and_due_honour_a_local_override(tmp_path, out_dir, monkeypatch, capsys):
    """Acceptance (#95): this test's shipped-config behaviour keeps passing
    even with a local override in play, and it flips `example` from the
    shipped `enabled = false` to due, with the shipped file never touched."""
    from tasks import briefing as task

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    local = tmp_path / "local-briefings.toml"
    local.write_text("[briefings]\nexample = true\n", encoding="utf-8")
    monkeypatch.setattr(bconfig, "LOCAL_OVERRIDES_PATH", local)

    task.main(["--list"])
    assert re.search(r"example\s+enabled", capsys.readouterr().out)  # overridden on, not shipped `disabled`

    lm = FakeLM()
    lm.ensure_server = lambda: None
    monkeypatch.setattr(task, "LMStudio", lambda: lm)
    monkeypatch.setattr(task, "make_fetcher", lambda: FakeFetcher(fixture_pages()))
    monkeypatch.setattr(task, "_now", lambda: NOW)
    import lmsa.briefing.sources as src
    monkeypatch.setattr(src.SourceContext, "searxng", lambda self: FakeSearx(search_results()))
    task.main([])  # no --name: the local override is what makes it due
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert Path(printed).name.startswith("briefing-example-") and Path(printed).exists()

    example_text = (bconfig.BRIEFINGS_DIR / "example.toml").read_text(encoding="utf-8")
    assert "\nenabled = false" in example_text  # the shipped file itself never changed


def test_malformed_local_override_is_printed_once_at_the_top_by_main(tmp_path, out_dir, monkeypatch, capsys):
    """(#95 review finding 2) a malformed config/local/briefings.toml is
    printed once, at the top of the run, explaining why nothing is due --
    not once per briefing file loaded along the way."""
    from tasks import briefing as task

    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    local = tmp_path / "local-briefings.toml"
    local.write_text("not valid toml [[[", encoding="utf-8")
    monkeypatch.setattr(bconfig, "LOCAL_OVERRIDES_PATH", local)

    task.main([])
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0].startswith("config/local/briefings.toml has a problem:")
    assert lines[0].endswith("falls back to its shipped setting)")
    # printed exactly once, however many shipped briefing files there are
    assert sum(1 for ln in lines if ln.startswith("config/local/briefings.toml has a problem")) == 1
    assert "no briefing is enabled and due" in out

    capsys.readouterr()
    task.main(["--list"])
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert sum(1 for ln in lines if ln.startswith("config/local/briefings.toml has a problem")) == 1


@pytest.mark.parametrize("quote,source", [
    ("Casualties rose to 1,200 by Friday", "casualties rose to 1200 by friday"),
    ("the U.S. will send more aid", "The US will send more aid"),
    ("aid to the US was cut", "aid to the U.S. was cut"),
    ("we met at the Café Central yesterday", "We met at the Cafe Central yesterday."),
])
def test_quote_formatting_variants_still_verify(quote, source):
    from lmsa.briefing.quotes import verify_quote
    assert verify_quote(quote, source)


def test_quote_decimal_point_still_matters():
    from lmsa.briefing.quotes import verify_quote
    assert not verify_quote("growth of 3.5 percent this year", "growth of 35 percent this year")
    assert not verify_quote("casualties rose to 150 people", "casualties rose to 15 people")


def test_default_searxng_client_uses_the_configured_url(monkeypatch):
    # Real runs pass no searxng_factory; the client must still get a base URL.
    monkeypatch.setenv("LMSA_SEARXNG_URL", "http://127.0.0.1:9999")
    ctx = sources.SourceContext(briefing=None, fetcher=None, lm=None, conn=None, now=NOW, limit=0)
    assert ctx.searxng().base_url == "http://127.0.0.1:9999"
