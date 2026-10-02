"""lmsa.markets (#60): defensive parsing of all three sources, movement
calc, country attribution, geo_metrics feed, and the briefing extension's
embedding-similarity matching + render/section -- no network, no LM Studio."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from lmsa import Store, geo_metrics, markets as mkt
from lmsa.briefing.adapters import markets as mkt_ext
from lmsa.briefing.engine import StoryResult
from lmsa.briefing.sources import SourceContext
from lmsa.report import Report
from lmsa.web import Fetcher, TransportResponse


@pytest.fixture
def conn(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        yield store.conn


class CannedTransport:
    """One canned JSON body per host; robots.txt always allowed."""

    def __init__(self, by_host: dict[str, TransportResponse]):
        self.by_host = by_host
        self.calls: list[str] = []

    def __call__(self, req):
        parts = urlsplit(req.url)
        if parts.path == "/robots.txt":
            return TransportResponse(404, {}, b"")
        self.calls.append(req.url)
        return self.by_host[parts.netloc]


def make_fetcher(by_host: dict[str, object]) -> Fetcher:
    responses = {
        host: (body if isinstance(body, TransportResponse) else
               TransportResponse(200, {"content-type": "application/json"}, json.dumps(body).encode()))
        for host, body in by_host.items()
    }
    return Fetcher("lmsa-markets-test/1.0", per_host_delay=0.0, cache_dir=None,
                    transport=CannedTransport(responses),
                    resolver=lambda host: ["93.184.216.34"], clock=lambda: 0.0, sleep=lambda s: None)


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

def test_fetch_polymarket_parses_canned_response_and_skips_bad_items():
    body = [
        {"id": "m1", "question": "Will Country X hold elections in 2026?", "slug": "x-elections-2026",
         "outcomes": '["Yes", "No"]', "outcomePrices": '["0.62", "0.38"]', "volume": "12345.6",
         "closed": False, "tags": [{"label": "Elections"}, {"label": "Ukraine"}]},
        {"weird": "no id or question here"},  # skipped, not a failure
    ]
    f = make_fetcher({"gamma-api.polymarket.com": body})
    markets = mkt.fetch_polymarket(f, tag="geopolitics", limit=10)
    assert len(markets) == 1
    m = markets[0]
    assert m.source == "polymarket" and m.market_id == "m1"
    assert m.probability == pytest.approx(0.62)
    assert m.url == "https://polymarket.com/event/x-elections-2026"
    assert "Ukraine" in m.tags and m.volume == pytest.approx(12345.6)


def test_fetch_polymarket_empty_result_is_not_an_error():
    f = make_fetcher({"gamma-api.polymarket.com": []})
    assert mkt.fetch_polymarket(f) == []


def test_fetch_polymarket_unrecognized_response_is_disclosed():
    body = [{"foo": "bar"}, {"baz": 1}]
    f = make_fetcher({"gamma-api.polymarket.com": body})
    with pytest.raises(mkt.UnrecognizedResponse):
        mkt.fetch_polymarket(f)


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------

def test_fetch_manifold_parses_canned_response_and_skips_non_binary():
    body = [
        {"id": "n1", "question": "Will Y happen in France?", "probability": 0.41,
         "creatorUsername": "alice", "slug": "will-y-happen", "groupSlugs": ["politics"], "volume": 500},
        {"id": "n2", "question": "Pick the winner", "outcomeType": "FREE_RESPONSE"},  # no probability
        {"nope": True},
    ]
    f = make_fetcher({"api.manifold.markets": body})
    markets = mkt.fetch_manifold(f, query="france")
    assert len(markets) == 1
    m = markets[0]
    assert m.source == "manifold" and m.market_id == "n1"
    assert m.probability == pytest.approx(0.41)
    assert m.url == "https://manifold.markets/alice/will-y-happen"
    assert "politics" in m.tags


def test_fetch_manifold_empty_result_is_not_an_error():
    f = make_fetcher({"api.manifold.markets": []})
    assert mkt.fetch_manifold(f, query="anything") == []


def test_fetch_manifold_unrecognized_response_is_disclosed():
    body = [{"a": 1}, {"b": 2}]
    f = make_fetcher({"api.manifold.markets": body})
    with pytest.raises(mkt.UnrecognizedResponse):
        mkt.fetch_manifold(f, query="x")


# ---------------------------------------------------------------------------
# Metaculus
# ---------------------------------------------------------------------------

def test_fetch_metaculus_parses_canned_response_both_shapes():
    body = {"results": [
        {"id": 101, "title": "Will A happen?", "page_url": "/questions/101/a/",
         "community_prediction": {"full": {"q2": 0.77}}, "categories": ["Politics"]},
        {"id": 202, "page_url": "/questions/202/z/",
         "question": {"title": "Will Z happen?",
                      "aggregations": {"recency_weighted": {"latest": {"centers": [0.33]}}}}},
        {"id": 303},  # no title anywhere -- skipped
    ]}
    f = make_fetcher({"www.metaculus.com": body})
    markets = mkt.fetch_metaculus(f, query="geopolitics")
    assert len(markets) == 2
    by_id = {m.market_id: m for m in markets}
    assert by_id["101"].probability == pytest.approx(0.77)
    assert by_id["101"].url == "https://www.metaculus.com/questions/101/a/"
    assert by_id["202"].probability == pytest.approx(0.33)
    assert by_id["202"].question == "Will Z happen?"


def test_fetch_metaculus_empty_result_is_not_an_error():
    f = make_fetcher({"www.metaculus.com": {"results": []}})
    assert mkt.fetch_metaculus(f) == []


def test_fetch_metaculus_unrecognized_response_is_disclosed():
    f = make_fetcher({"www.metaculus.com": {"results": [{"a": 1}]}})
    with pytest.raises(mkt.UnrecognizedResponse):
        mkt.fetch_metaculus(f)


# ---------------------------------------------------------------------------
# collect(): per-source isolation
# ---------------------------------------------------------------------------

def test_collect_isolates_a_broken_source():
    good_poly = [{"id": "m1", "question": "Will X happen?", "slug": "x",
                  "outcomePrices": '["0.5"]'}]
    f = make_fetcher({
        "gamma-api.polymarket.com": good_poly,
        "api.manifold.markets": [{"a": 1}],  # unrecognized
        "www.metaculus.com": {"results": []},
    })
    cfg = mkt.MarketsConfig(polymarket_tags=[None], manifold_queries=["war"], metaculus_queries=[None])
    markets, notes = mkt.collect(f, cfg)
    assert len(markets) == 1 and markets[0].source == "polymarket"
    assert any("manifold" in n and "unrecognized" in n for n in notes)


# ---------------------------------------------------------------------------
# Country attribution (#60 review: false positives from naive n-gram
# matching -- US-state collisions, common-noun/given-name collisions, and
# person-name adjacency all need to be filtered before trusting a hit)
# ---------------------------------------------------------------------------

# (title, expected alpha-2 or None) -- the five review examples plus
# realistic true positives, ambiguous multi-country titles, and a few more
# state/name/noun collisions in the same families.
TITLE_CASES = [
    # -- the five reported false positives -----------------------------------
    ("Georgia Senate race", None),                              # US state, no cue word
    ("New Mexico governor race", None),                         # multi-word US state veto
    ("Jordan Peterson to speak at a conference", None),         # adjacent given-name/surname
    ("Jordan Peterson announces a run for president", None),    # name-adjacency overrides a real cue word
    ("Turkey prices at Thanksgiving dinner", None),             # ambiguous noun, no cue word
    ("Chad to hold elections", "TD"),                           # ambiguous word + cue word "elections" -> counts
    # -- required true positives ---------------------------------------------
    ("Will Maduro remain president of Venezuela?", "VE"),
    ("Will Russia and Ukraine sign a ceasefire?", None),        # two countries -> ambiguous
    ("Georgia (country) parliamentary election", "GE"),         # explicit + cue word overrides ambiguity
    # -- more of the same collision families ----------------------------------
    ("New Jersey Senate race heats up", None),                  # multi-word US state veto
    ("New York City mayoral race", None),                       # no country collision anyway
    ("Niger coup leaders face sanctions", "NE"),                # ambiguous + two cue words
    ("Niger delta oil production rises", None),                 # ambiguous, no cue word
    ("Guinea pig adoption events this weekend", None),          # ambiguous noun, no cue word
    ("Guinea military junta announces election date", "GN"),    # ambiguous + cue words
    ("New Jersey shore reopens for summer", None),               # multi-word US state veto
    ("Will China invade a neighboring island this decade?", "CN"),  # ambiguous + cue word "invade"
    ("Grandma's fine china collection sells at auction", None),  # common-noun collision, no cue word
    ("Will Ukraine hold elections in 2026?", "UA"),             # unambiguous, no gate needed
    ("Will the championship game go into overtime?", None),     # no mention at all
    ("Turkey's president calls snap election", "TR"),           # ambiguous + two cue words
]


@pytest.mark.parametrize("title,expected", TITLE_CASES, ids=[c[0] for c in TITLE_CASES])
def test_attribute_country_from_title(title, expected):
    assert mkt.attribute_country(title) == expected


def test_attribute_country_from_structured_tags():
    assert mkt.attribute_country("Will the election happen on schedule?", tags=["Ukraine"]) == "UA"


def test_attribute_country_structured_tags_take_priority_over_title():
    # The title alone (no cue word) would resolve to None; a single clean
    # tag hit still wins outright without even looking at the title.
    assert mkt.attribute_country("Turkey prices at Thanksgiving dinner", tags=["Turkey"]) == "TR"


def test_attribute_country_ambiguous_tags_do_not_fall_back_to_title():
    # Two distinct countries in the tags is itself ambiguous -- it must not
    # fall through to the (otherwise unambiguous) title text.
    assert mkt.attribute_country("Will Ukraine hold elections in 2026?",
                                  tags=["Ukraine", "Russia"]) is None


# ---------------------------------------------------------------------------
# Storage: idempotency + per-row isolation
# ---------------------------------------------------------------------------

def test_record_snapshots_same_day_rerun_replaces_not_duplicates(conn):
    m = mkt.Market(source="polymarket", market_id="m1", question="Will Q happen?",
                    url="https://x", probability=0.30)
    w1, e1 = mkt.record_snapshots(conn, [m], "2026-09-25", fetched_at="t1")
    assert (w1, e1) == (1, [])
    m2 = mkt.Market(source="polymarket", market_id="m1", question="Will Q happen?",
                     url="https://x", probability=0.35)
    w2, e2 = mkt.record_snapshots(conn, [m2], "2026-09-25", fetched_at="t2")
    assert (w2, e2) == (1, [])
    count, prob = conn.execute(
        "SELECT COUNT(*), MAX(probability) FROM market_snapshots"
        " WHERE source='polymarket' AND market_id='m1' AND day='2026-09-25'"
    ).fetchone()
    assert count == 1 and prob == pytest.approx(0.35)


def test_record_snapshots_isolates_one_bad_row(conn):
    class FlakyConn:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, params=()):
            if "INSERT OR REPLACE INTO market_snapshots" in sql and params[1] == "bad":
                raise sqlite3.OperationalError("simulated failure")
            return self._real.execute(sql, params)

        def commit(self):
            self._real.commit()

    good = mkt.Market(source="polymarket", market_id="good", question="Will Q happen?",
                       url=None, probability=0.5)
    bad = mkt.Market(source="polymarket", market_id="bad", question="Will R happen?",
                      url=None, probability=0.6)
    written, errors = mkt.record_snapshots(FlakyConn(conn), [good, bad], "2026-09-25", fetched_at="t")
    assert written == 1
    assert len(errors) == 1 and "bad" in errors[0]
    assert conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE market_id='good'").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def test_movement_24h_and_7d_within_tolerance(conn):
    def snap(day, prob):
        mkt.record_snapshots(
            conn, [mkt.Market(source="polymarket", market_id="m1", question="Q", url=None, probability=prob)],
            day, fetched_at="t",
        )

    snap("2026-09-18", 0.50)
    snap("2026-09-24", 0.55)
    snap("2026-09-25", 0.60)
    mv = mkt.movement(conn, "polymarket", "m1", "2026-09-25")
    assert mv.probability == pytest.approx(0.60)
    assert mv.pct_24h == pytest.approx(0.05)
    assert mv.pct_7d == pytest.approx(0.10)


def test_movement_is_none_not_zero_when_no_snapshot_is_close_enough(conn):
    mkt.record_snapshots(
        conn, [mkt.Market(source="polymarket", market_id="m2", question="Q", url=None, probability=0.20)],
        "2026-09-05", fetched_at="t",
    )
    mkt.record_snapshots(
        conn, [mkt.Market(source="polymarket", market_id="m2", question="Q", url=None, probability=0.90)],
        "2026-09-25", fetched_at="t",
    )
    mv = mkt.movement(conn, "polymarket", "m2", "2026-09-25")
    assert mv.probability == pytest.approx(0.90)
    assert mv.pct_24h is None
    assert mv.pct_7d is None


def test_movement_missing_market_is_all_none(conn):
    mv = mkt.movement(conn, "polymarket", "does-not-exist", "2026-09-25")
    assert mv == mkt.MarketMovement(None, None, None)


# ---------------------------------------------------------------------------
# geo_metrics feed
# ---------------------------------------------------------------------------

def test_country_metrics_max_abs_move_and_no_data_is_left_out(conn):
    def snap(day, source, market_id, question, prob, tags=()):
        mkt.record_snapshots(conn, [mkt.Market(source=source, market_id=market_id, question=question,
                                                url=None, probability=prob, tags=tuple(tags))],
                              day, fetched_at="t")

    prev, day = "2026-09-24", "2026-09-25"
    # UA market #1: big move (0.15)
    snap(prev, "polymarket", "ua1", "Ukraine ceasefire talks", 0.40)
    snap(day, "polymarket", "ua1", "Ukraine ceasefire talks", 0.55)
    # UA market #2: small move (0.02) -- max() must pick the bigger one
    snap(prev, "manifold", "ua2", "Ukraine grain deal renewed?", 0.50)
    snap(day, "manifold", "ua2", "Ukraine grain deal renewed?", 0.52)
    # No country attributable -- excluded even though it moved a lot
    snap(prev, "polymarket", "x1", "Will something happen?", 0.10)
    snap(day, "polymarket", "x1", "Will something happen?", 0.90)
    # France: attributable but only today's snapshot -- no computable move
    snap(day, "polymarket", "fr1", "Will France do X?", 0.30)

    rows = mkt.country_metrics(conn, day)
    by_country = {country: value for _, country, metric, value in rows}
    assert by_country == {"UA": pytest.approx(0.15)}
    assert "FR" not in by_country


def test_country_metrics_feed_geo_metrics_record_many(conn):
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="ua1",
                                            question="Ukraine election", url=None, probability=0.40)],
                          "2026-09-24", fetched_at="t")
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="ua1",
                                            question="Ukraine election", url=None, probability=0.55)],
                          "2026-09-25", fetched_at="t")
    rows = mkt.country_metrics(conn, "2026-09-25")
    written = geo_metrics.record_many(conn, rows)
    assert written == 1
    series = geo_metrics.series(conn, "UA", "markets")
    assert series[-1][1] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Briefing extension: matching, render, section
# ---------------------------------------------------------------------------

class FakeLM:
    """`embed(texts, model=None) -> vectors`, deterministic per exact text."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.cfg = SimpleNamespace(embed_model="fake-embed")

    def embed(self, texts, model=None):
        return [self.vectors[t] for t in texts]


class CountingLM(FakeLM):
    """Same as `FakeLM`, but records every `embed()` call's texts."""

    def __init__(self, vectors: dict[str, list[float]]):
        super().__init__(vectors)
        self.calls: list[list[str]] = []

    def embed(self, texts, model=None):
        self.calls.append(list(texts))
        return super().embed(texts, model=model)


def make_ctx(conn, lm) -> SourceContext:
    return SourceContext(briefing=None, fetcher=None, lm=lm, conn=conn,
                          now=datetime(2026, 9, 25, tzinfo=timezone.utc), limit=10)


def test_enrich_matches_above_threshold_and_carries_movement(conn):
    q1, q2 = "Will Ukraine hold elections in 2026?", "Will the Lakers win the championship?"
    mkt.record_snapshots(conn, [
        mkt.Market(source="polymarket", market_id="ua1", question=q1, url="https://polymarket.com/x", probability=0.60),
        mkt.Market(source="manifold", market_id="lk1", question=q2, url="https://manifold.markets/y", probability=0.30),
    ], "2026-09-24", fetched_at="t")
    mkt.record_snapshots(conn, [
        mkt.Market(source="polymarket", market_id="ua1", question=q1, url="https://polymarket.com/x", probability=0.68),
        mkt.Market(source="manifold", market_id="lk1", question=q2, url="https://manifold.markets/y", probability=0.30),
    ], "2026-09-25", fetched_at="t")

    ctx = make_ctx(conn, FakeLM({q1: [1.0, 0.0], q2: [0.0, 1.0]}))
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    mkt_ext.enrich([story], ctx)

    matches = story.extras.get("markets")
    assert matches and len(matches) == 1
    assert matches[0]["source"] == "polymarket"
    assert matches[0]["probability"] == pytest.approx(0.68)
    assert matches[0]["pct_24h"] == pytest.approx(0.08)


def test_enrich_caches_question_embeddings_in_the_vectors_table(conn):
    q = "Will Ukraine hold elections in 2026?"
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="ua1", question=q,
                                            url="https://x", probability=0.6)],
                          "2026-09-25", fetched_at="t")
    lm = CountingLM({q: [1.0, 0.0]})
    ctx = make_ctx(conn, lm)

    mkt_ext.enrich([StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])], ctx)
    assert lm.calls == [[q]]
    assert conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE ns='markets' AND item_id='polymarket:ua1'"
    ).fetchone()[0] == 1

    # A second enrich() call (a different briefing, or a rerun) must reuse
    # the cached vector instead of calling embed() again.
    story2 = StoryResult(story_id="s2", docs=[], centroid=[1.0, 0.0])
    mkt_ext.enrich([story2], ctx)
    assert lm.calls == [[q]]  # still just the one call
    assert story2.extras.get("markets")


def test_enrich_only_embeds_uncached_questions(conn):
    q1, q2 = "Will Ukraine hold elections in 2026?", "Will France do X?"
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="ua1", question=q1,
                                            url=None, probability=0.6)],
                          "2026-09-25", fetched_at="t")
    lm = CountingLM({q1: [1.0, 0.0], q2: [0.0, 1.0]})
    ctx = make_ctx(conn, lm)
    mkt_ext.enrich([StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])], ctx)
    assert lm.calls == [[q1]]

    # A new market with a new question shows up; the already-cached one
    # must not be re-sent to embed().
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="fr1", question=q2,
                                            url=None, probability=0.5)],
                          "2026-09-25", fetched_at="t")
    mkt_ext.enrich([StoryResult(story_id="s2", docs=[], centroid=[0.0, 1.0])], ctx)
    assert lm.calls == [[q1], [q2]]


def test_enrich_below_threshold_is_no_match(conn):
    q = "Will something totally unrelated happen?"
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="z1", question=q,
                                            url=None, probability=0.5)], "2026-09-25", fetched_at="t")
    ctx = make_ctx(conn, FakeLM({q: [0.0, 1.0]}))
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    mkt_ext.enrich([story], ctx)
    assert "markets" not in story.extras


def test_enrich_embedding_failure_is_a_note_not_a_crash(conn):
    mkt.record_snapshots(conn, [mkt.Market(source="polymarket", market_id="z1", question="Q",
                                            url=None, probability=0.5)], "2026-09-25", fetched_at="t")

    class BrokenLM:
        cfg = SimpleNamespace(embed_model="fake-embed")

        def embed(self, texts, model=None):
            raise RuntimeError("model unavailable")

    ctx = make_ctx(conn, BrokenLM())
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    mkt_ext.enrich([story], ctx)  # must not raise
    assert "markets" not in story.extras
    assert any("embedding" in n for n in ctx.notes)


def test_render_labels_odds_not_a_forecast():
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    story.extras["markets"] = [{
        "source": "polymarket", "question": "Will Ukraine hold elections?",
        "url": "https://polymarket.com/x", "probability": 0.68, "pct_24h": 0.08, "pct_7d": 0.15,
        "similarity": 0.9,
    }]
    extras = mkt_ext.render(story, ctx=None)
    assert len(extras) == 1
    assert "not a forecast by this tool" in extras[0].text
    assert extras[0].href == "https://polymarket.com/x"
    assert extras[0].tone == "hot"  # 8pt move >= HOT_MOVE


def test_render_no_markets_is_empty():
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    assert mkt_ext.render(story, ctx=None) == []


def test_section_lists_biggest_movers_and_discloses(conn):
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    story.extras["markets"] = [{
        "source": "polymarket", "question": "Will Ukraine hold elections?",
        "url": "https://polymarket.com/x", "probability": 0.68, "pct_24h": 0.08, "pct_7d": 0.15,
    }]
    report = Report("Test briefing")
    mkt_ext.section(report, [story], ctx=None)
    html = report.render()
    assert "Biggest movers" in html
    assert "not a forecast by this tool" in html
    assert "Ukraine hold elections" in html


def test_section_with_no_movement_data_adds_nothing():
    story = StoryResult(story_id="s1", docs=[], centroid=[1.0, 0.0])
    report = Report("Test briefing")
    mkt_ext.section(report, [story], ctx=None)
    assert "Biggest movers" not in report.render()


def test_extension_is_registered_not_default():
    from lmsa.briefing.sources import EXTENSIONS

    ext = EXTENSIONS["markets"]
    assert ext.default is False
    assert ext.enrich is mkt_ext.enrich and ext.render is mkt_ext.render and ext.section is mkt_ext.section
