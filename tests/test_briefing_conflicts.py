"""Conflict tracker (#55): ReliefWeb + GDELT adapters (canned JSON), the
theaters extension (grouping, ranking, escalation flag, quoted figures, map)
and the shipped config. No network, no model."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from lmsa import Store
from lmsa.briefing import config as bconfig
from lmsa.briefing import engine, sources
from lmsa.briefing.engine import StoryResult
from lmsa.briefing.sources import Doc, SourceContext
from lmsa.web import Fetcher, HTTPStatusError, TransportResponse

sources.load_plugins()
from lmsa.briefing.adapters import gdelt_doc, reliefweb, theaters  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "briefing_conflicts"
NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
RW_SUDAN = "https://reliefweb.int/report/sudan/sudan-el-fasher-flash-update-no-12"
BROKEN_FEED = "https://feeds.example.org/broken.xml"
TOPICS = ["ukraine", "sudan", "gaza"]


def rw_bytes() -> bytes:
    return (FIX / "reliefweb.json").read_bytes()


def gdelt_bytes() -> bytes:
    return (FIX / "gdelt_artlist.json").read_bytes()


def ctx(fetcher=None, limit=100, **kw) -> SourceContext:
    return SourceContext(briefing=SimpleNamespace(name="t"), fetcher=fetcher, lm=None, conn=None, now=NOW,
                         limit=limit, **kw)


# ---------------------------------------------------------------------------
# ReliefWeb
# ---------------------------------------------------------------------------

def test_reliefweb_parses_canned_reports():
    docs = reliefweb.parse_reports(json.loads(rw_bytes()))
    assert [d.url for d in docs] == [RW_SUDAN, "https://reliefweb.int/node/4100002"]  # junk item skipped
    sudan = docs[0]
    assert sudan.meta["countries"] == ["SD", "TD"]  # primary first, iso3 (any case) -> alpha-2
    assert sudan.published == datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
    assert sudan.source == "ReliefWeb: OCHA" and "At least 40 people" in sudan.text
    assert docs[1].meta["countries"] == ["UA"]


def test_reliefweb_request_names_the_app_and_window():
    url = reliefweb.request_url({"query": "conflict", "days": 3}, NOW, 61)
    q = parse_qs(urlsplit(url).query)
    assert url.startswith("https://api.reliefweb.int/v1/reports?")
    assert q["appname"] == ["lmstudio-automation"] and q["limit"] == ["61"]
    assert q["query[value]"] == ["conflict"] and q["filter[value][from]"] == ["2026-09-22T06:00:00+00:00"]
    assert "body" in q["fields[include][]"]


def test_reliefweb_empty_vs_unrecognized():
    assert reliefweb.parse_reports({"data": []}) == []
    for bad in ({}, {"data": "x"}, [], {"error": {"status": 400}}):
        with pytest.raises(reliefweb.UnrecognizedResponse):
            reliefweb.parse_reports(bad)
    with pytest.raises(reliefweb.UnrecognizedResponse, match="none usable"):
        reliefweb.parse_reports({"data": [{"id": "x", "fields": {"title": ""}}]})


def test_reliefweb_config_validation():
    base = {"briefing": {"title": "T", "interests": ["x"]}}
    ok = dict(base, sources={"api": [{"adapter": "reliefweb", "days": 5}]})
    assert bconfig.parse("t", ok).sources["api"][0]["days"] == 5
    with pytest.raises(bconfig.ConfigError, match="days"):
        bconfig.parse("t", dict(base, sources={"api": [{"adapter": "reliefweb", "days": 90}]}))


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

def test_gdelt_parses_canned_artlist():
    docs = gdelt_doc.parse_artlist(gdelt_bytes())
    assert len(docs) == 3  # the javascript: link is dropped
    assert docs[0].published == datetime(2026, 9, 25, 3, 15, tzinfo=timezone.utc)
    assert docs[0].source == "news.example.com" and docs[0].text == ""
    assert docs[2].meta["source_country"] == "Kenya"


@pytest.mark.parametrize("body", [b"", b"  \n", b"{}", b'{"articles": []}'])
def test_gdelt_no_matches_is_empty(body):
    assert gdelt_doc.parse_artlist(body) == []


@pytest.mark.parametrize("body", [
    b"Please limit requests to one every 5 seconds",    # non-empty 200 that is not JSON
    b"[1, 2]",
    b'{"timeline": []}',
    b'{"articles": [{"url": "ftp://x", "title": "t"}, {"title": ""}]}',   # articles, none usable
])
def test_gdelt_unrecognized_response_is_an_error_not_a_quiet_day(body):
    with pytest.raises(gdelt_doc.UnrecognizedResponse):
        gdelt_doc.parse_artlist(body)


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def gdelt_fetcher(statuses: list[int], seen: list[str]) -> tuple[Fetcher, Clock]:
    clk = Clock()

    def transport(req):
        seen.append(req.url)
        status = statuses.pop(0) if statuses else 200
        return TransportResponse(status, {"content-type": "application/json"}, gdelt_bytes() if status == 200 else b"")

    f = Fetcher("test-agent", per_host_delay=2.0, cache_dir=None, respect_robots=False, transport=transport,
                resolver=lambda host: ["93.184.216.34"], clock=clk.clock, sleep=clk.sleep, now=lambda: 0.0)
    return f, clk


def test_gdelt_requests_are_spaced_5s_and_capped_per_run():
    seen: list[str] = []
    f, clk = gdelt_fetcher([], seen)
    c = ctx(f)
    entry = {"name": "g", "query": "theme:ARMEDCONFLICT", "timespan": "24h"}
    for _ in range(gdelt_doc.MAX_QUERIES_PER_RUN + 1):
        list(gdelt_doc.gdelt_doc(entry, c))
    assert len(seen) == gdelt_doc.MAX_QUERIES_PER_RUN
    assert clk.slept and all(s >= gdelt_doc.MIN_DELAY - 1e-9 for s in clk.slept)
    assert f.per_host_delay == 2.0  # restored for every other host
    assert any("at most 4 GDELT queries" in n for n in c.notes)
    q = parse_qs(urlsplit(seen[0]).query)
    assert q["mode"] == ["artlist"] and q["format"] == ["json"] and q["maxrecords"] == ["100"]


def test_gdelt_429_skips_the_rest_of_the_run():
    seen: list[str] = []
    f, _ = gdelt_fetcher([429], seen)
    c = ctx(f)
    entry = {"name": "g", "query": "q"}
    with pytest.raises(RuntimeError, match="429"):
        list(gdelt_doc.gdelt_doc(entry, c))
    assert list(gdelt_doc.gdelt_doc(entry, c)) == []
    assert len(seen) == 1 and any("throttled" in n for n in c.notes)


def test_gdelt_never_lowers_a_slower_fetcher():
    seen: list[str] = []
    f, _ = gdelt_fetcher([], seen)
    f.per_host_delay = 9.0
    list(gdelt_doc.gdelt_doc({"name": "g", "query": "q"}, ctx(f)))
    assert f.per_host_delay == 9.0


# ---------------------------------------------------------------------------
# Theaters: unit
# ---------------------------------------------------------------------------

def story(sid, countries=(), severity=1, sources=("a",), status="new", headline="h", meta_countries=None,
          summary="", actors=(), texts=None):
    texts = texts or [""] * len(sources)
    docs = [Doc(url=f"https://x.example/{sid}/{i}", title=headline, text=texts[i], source=src, published=NOW,
                meta={"norm_url": f"https://x.example/{sid}/{i}", "countries": meta_countries or []})
            for i, src in enumerate(sources)]
    return StoryResult(story_id=sid, docs=docs, centroid=[0.0], status=status, countries=list(countries),
                       extract={"headline": headline, "summary": summary, "severity": severity,
                                "actors": list(actors)})


def test_theater_assignment_by_core_country_then_country_then_alias():
    assert theaters.assign(story("1", ["RU", "UA"]))[0] == "russia-ukraine"
    assert theaters.assign(story("2", ["CN", "US"]))[0] == "pair-cn-us"  # CN without TW is not Taiwan Strait
    assert theaters.assign(story("3", ["TW", "CN"]))[0] == "taiwan-strait"
    assert theaters.assign(story("4", [], meta_countries=["SD", "TD"]))[0] == "sudan"  # structured fallback
    assert theaters.assign(story("5", [], headline="Houthi missile hits tanker"))[0] == "yemen-red-sea"
    assert theaters.assign(story("6", [], headline="Unrest somewhere"))[0] == theaters.UNPLACED
    assert theaters.assign(story("7", ["IL", "IR"]))[0] == "israel-gaza-lebanon"
    assert theaters.assign(story("8", ["IR"]))[1] == "Iran"


def test_theater_ranking_is_stories_x_sources_x_severity():
    stories = [
        story("u1", ["UA"], severity=1, sources=("a", "b")),
        story("u2", ["UA", "RU"], severity=2, sources=("b", "c")),   # UA: 2 stories x 3 sources x 3 = 18
        story("s1", ["SD"], severity=3, sources=("a", "d")),          # SD: 1 x 2 x 4 = 8
        story("m1", ["MM"], severity=0, sources=("a",)),              # MM: 1 x 1 x 1 = 1
        story("n1", [], severity=3, sources=("a", "b", "c", "d", "e")),  # unplaced: always last
    ]
    ranked = theaters.group(stories)
    assert [t.slug for t in ranked] == ["russia-ukraine", "sudan", "country-mm", "unplaced"]
    assert [t.activity for t in ranked[:3]] == [18.0, 8.0, 1.0]
    values, links = theaters.map_values(ranked)
    assert values["UA"] == 18.0 and values["RU"] == 18.0 and links["SD"] == "#theater-sudan"
    assert "unplaced" not in "".join(links.values())


@pytest.mark.parametrize("kind,figure,quote,shown", [
    ("casualties", "40", "At least 40 people were killed", "40"),
    ("displacement", "1200", "some 1,200 people displaced overnight", "1,200"),   # shown as the quote writes it
    ("casualties", "1,200 civilians killed", "1,200 people were killed", "1,200"),  # model wording never shown
    ("casualties", "400", "At least 40 people were killed", None),
    ("displacement", "thousands", "thousands have fled the town", "thousands"),
    ("displacement", "hundreds", "thousands have fled the town", None),
    # ellipsis: a number from an unrelated sentence must not verify
    ("casualties", "40", "The 40 ... people were killed", None),
    ("casualties", "40", "The 40\u2026 people were killed", None),
    # number and casualty word in different sentences / clauses
    ("casualties", "40", "The convoy had 40 trucks. Two people were killed.", None),
    ("casualties", "40", "40 trucks crossed; people were killed later", None),
    # the key noun must be there: trucks are not casualties
    ("casualties", "1,200", "1,200 trucks entered Gaza on Monday", None),
    # another number sits between the figure and the word: it is that one's
    ("casualties", "1,200", "1,200 trucks and 2 people were killed", None),
    ("casualties", "2", "1,200 trucks and 2 people were killed", "2"),
    # right number, wrong kind
    ("casualties", "12,000", "More than 12,000 people have been displaced", None),
])
def test_figure_support(kind, figure, quote, shown):
    assert theaters.figure_support(kind, figure, quote) == shown


# ---------------------------------------------------------------------------
# End to end: config -> engine -> theaters -> report
# ---------------------------------------------------------------------------

class FakeLM:
    def __init__(self):
        self.theater_calls = 0
        self.story_calls = 0

    def embed(self, texts, *, model=None):
        return [[float(w in t.lower()) for w in TOPICS] + [0.05] for t in texts]

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        if name == "theater":
            self.theater_calls += 1
            assert "untrusted data" in prompt and "untrusted data" in system
            theater = re.search(r"^Theater: (.+)$", prompt, re.M).group(1)
            if theater == "Sudan":
                return {"trend": "escalating", "reason": "Shelling of El Fasher intensified.", "figures": [
                    {"kind": "casualties", "figure": "40", "quote": "At least 40 people were killed in El Fasher on Monday",
                     "source_url": RW_SUDAN},
                    {"kind": "displacement", "figure": "12,000",  # right quote, wrong citation: corrected
                     "quote": "More than 12,000 people have been displaced", "source_url": "https://wrong.example/x"},
                    {"kind": "casualties", "figure": "500", "quote": "500 soldiers died in the battle for the city",
                     "source_url": RW_SUDAN},  # invented: dropped
                ]}
            if theater.startswith("Russia"):
                return {"trend": "steady", "reason": "Drone strikes on Kharkiv continue.", "figures": []}
            return {"trend": "sideways", "reason": "Talks resumed\nin Cairo.", "figures": "nope"}
        self.story_calls += 1
        low = prompt.lower()
        countries = [c for w, c in (("ukraine", "UA"), ("russian", "RU"), ("sudan", "SD"), ("gaza", "PS")) if w in low]
        sev = 3 if "sudan" in low else 2 if "ukraine" in low else 1
        title = re.search(r"^TITLE: (.+)$", prompt, re.M).group(1)
        return {"headline": title, "summary": f"About {title}.", "highlights": [], "key_quotes": [],
                "timeline": [], "why_it_matters": "", "countries": countries, "actors": [], "severity": sev}


class FakeFetcher:
    def __init__(self):
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url.startswith(gdelt_doc.ENDPOINT):
            return SimpleNamespace(content=gdelt_bytes(), url=url, status=200, headers={}, from_cache=False)
        if url.startswith(reliefweb.ENDPOINT):
            return SimpleNamespace(content=rw_bytes(), url=url, status=200, headers={}, from_cache=False)
        raise HTTPStatusError(404, url)

    def get_json(self, url, **kw):
        return json.loads(self.get(url).content)


def conflict_cfg():
    return bconfig.parse("conf", {
        "briefing": {"title": "Conflicts", "interests": ["armed conflict"], "prefilter": False,
                     "extensions": ["theaters"]},
        "sources": {
            "feeds": [{"name": "Broken feed", "url": BROKEN_FEED, "group": "think tanks"}],
            "api": [{"adapter": "reliefweb", "name": "ReliefWeb", "lean": "intl", "reliability": "high"},
                    {"adapter": "gdelt_doc", "name": "GDELT", "query": "theme:ARMEDCONFLICT"}],
        },
    })


def run(db, lm):
    with Store(db) as store, store.record_run("briefing") as r:
        res = engine.run_briefing(conflict_cfg(), lm=lm, conn=store.conn, run_id=r.id, fetcher=FakeFetcher(), now=NOW)
        return res, engine.build_report(res).render()


def test_end_to_end_theaters_flags_figures_map(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    lm = FakeLM()
    res, html = run(tmp_path / "db.sqlite", lm)
    assert res.counts["collected"] == 5 and len(res.stories) == 3
    ranked = res.ctx._theaters
    # Sudan 1 story x 2 sources x (1+3) = 8; Ukraine 1 x 2 x 3 = 6; Gaza 1 x 1 x 2 = 2
    assert [t.slug for t in ranked] == ["sudan", "russia-ukraine", "israel-gaza-lebanon"]
    assert [t.activity for t in ranked] == [8.0, 6.0, 2.0]
    # every theater carries a flag; a bad model trend becomes "unclear", reason is one line
    assert [t.trend for t in ranked] == ["escalating", "steady", "unclear"]
    assert ranked[2].reason == "Talks resumed in Cairo."
    # figures: quoted + verified only; the wrong citation is corrected, the invented one dropped
    sudan = ranked[0]
    assert [(f["figure"], f["source_url"]) for f in sudan.figures] == [("40", RW_SUDAN), ("12,000", RW_SUDAN)]
    assert any("1 model-reported figure(s) dropped" in n for n in res.notes)
    assert "500 soldiers" not in html
    # stories carry their theater (persisted) and the card links to it
    for s in res.stories:
        assert s.extras["theaters"]["slug"] in {t.slug for t in ranked}
    assert 'href="#theater-sudan"' in html and 'id="theater-sudan"' in html
    assert 'class="geo-map"' in html and html.count('class="geo-svg"') == 1
    assert "Theaters by activity" in html and "What changed in 24 h: 1 new, 0 updated" in html
    # the failing feed is disclosed, in Notes and in the source table
    assert any("feeds: Broken feed: HTTPStatusError" in n for n in res.notes)
    assert "Sources this run" in html and "fetch failed: HTTPStatusError: HTTP 404" in html
    assert "ok: 2 of 2 kept" in html
    with Store(tmp_path / "db.sqlite") as store:
        extras = [json.loads(r[0])["extras"] for r in store.conn.execute("SELECT data_json FROM briefing_stories")]
    assert all("theaters" in e for e in extras)

    # second run, same input: every theater flag comes from the memo
    before = lm.theater_calls
    res2, _ = run(tmp_path / "db.sqlite", lm)
    assert lm.theater_calls == before and [t.trend for t in res2.ctx._theaters] == ["escalating", "steady", "unclear"]


def test_theater_model_failure_is_flagged_unclear_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))

    class Broken(FakeLM):
        def extract(self, prompt, schema, *, system=None, model=None, name="result"):
            if name == "theater":
                raise RuntimeError("model crashed")
            return super().extract(prompt, schema, system=system, model=model, name=name)

    res, html = run(tmp_path / "db.sqlite", Broken())
    assert all(t.trend == "unclear" and "could not be assessed" in t.reason for t in res.ctx._theaters)
    assert any("3 theater(s) could not be assessed" in n for n in res.notes)
    assert "Theaters by activity" in html


def test_shipped_conflicts_config():
    cfg = bconfig.load("conflicts")
    assert cfg.enabled is None  # ships as NEW: the owner enables it
    assert cfg.extensions == ["theaters", "markets"] and not cfg.regions
    entries = cfg.entries()
    adapters = {e.get("adapter") for k, e in entries if k == "api"}
    assert {"reliefweb", "gdelt_doc"} <= adapters
    groups = {e["group"] for _, e in entries}
    assert {"think tanks", "world", "humanitarian", "gdelt"} <= groups
    names = {e["name"] for _, e in entries}
    assert "Brookings" in names and not {"ISW", "CSIS", "RAND", "Crisis Group"} & names  # #117: unreachable
    assert "S2 Underground" in names and "osint" in groups  # #125
    assert all(e["lean"] != "unknown" or e.get("adapter") == "gdelt_doc" for _, e in entries)
    text = (bconfig.BRIEFINGS_DIR / "conflicts.toml").read_text(encoding="utf-8")
    assert "\nenabled" not in text
    assert sum(1 for _, e in entries if e.get("adapter") == "gdelt_doc") <= gdelt_doc.MAX_QUERIES_PER_RUN


def test_shipped_conflicts_config_still_shipped_new_with_a_local_override(tmp_path, monkeypatch):
    """Acceptance (#95): a local override enabling `conflicts` doesn't
    change this test's other assertions, and the shipped file stays
    exactly as `test_shipped_conflicts_config` found it (no `enabled` line,
    unchanged bytes) -- the override lives only in the gitignored local
    file."""
    before = (bconfig.BRIEFINGS_DIR / "conflicts.toml").read_bytes()
    local = tmp_path / "briefings.toml"
    local.write_text("[briefings]\nconflicts = true\n", encoding="utf-8")
    monkeypatch.setattr(bconfig, "LOCAL_OVERRIDES_PATH", local)

    cfg = bconfig.load("conflicts")
    assert cfg.enabled is True  # overridden on, unlike the shipped-NEW default
    assert cfg.extensions == ["theaters", "markets"] and not cfg.regions
    adapters = {e.get("adapter") for k, e in cfg.entries() if k == "api"}
    assert {"reliefweb", "gdelt_doc"} <= adapters

    assert (bconfig.BRIEFINGS_DIR / "conflicts.toml").read_bytes() == before
    assert "\nenabled" not in before.decode("utf-8")


def test_theaters_past_the_section_cap_link_to_the_ranking_table():
    stories = [story(f"s{i}", [c], severity=1) for i, c in
               enumerate(["SD", "MM", "HT", "SO", "ET", "ML", "CD", "YE", "UA", "PS", "KP", "TW", "AM", "LY"])]
    ranked = theaters.group(stories)
    assert len(ranked) > theaters.MAX_SECTIONS
    _, links = theaters.map_values(ranked)
    last = ranked[-1]
    assert theaters.target(last, len(ranked)) == "#theaters"
    assert links[last.countries[0]] == "#theaters"
    assert theaters.target(ranked[0], 1) == f"#{ranked[0].anchor}"


def test_fallback_theaters_do_not_depend_on_model_order():
    a = theaters.assign(story("1", ["IN", "PK"]))
    b = theaters.assign(story("2", ["PK", "IN"]))
    assert a == b and a[0] == "pair-in-pk" and a[1] == "India\u2013Pakistan" and a[2] == ["IN", "PK"]
    # an external actor next to local countries doesn't pool unrelated stories
    slugs = {theaters.assign(story(c, ["US", c]))[0] for c in ("IR", "VE", "SO")}
    assert slugs == {"country-ir", "country-ve", "country-so"}
    assert theaters.assign(story("x", ["SO", "US"]))[0] == "country-so"
    assert theaters.assign(story("u", ["US"]))[0] == "country-us"          # alone, it is its own theater
    assert theaters.assign(story("rc", ["RU", "CN"]))[0] == "pair-cn-ru"   # only external actors: kept
    # 3+ local countries: the one most docs are about, whatever the order
    texts = ["Ethiopia shelling", "Ethiopian army in Ethiopia", "Somalia border"]
    for order in (["ET", "ER", "SO"], ["SO", "ER", "ET"], ["ER", "SO", "ET"]):
        assert theaters.assign(story("e", order, sources=("a", "b", "c"), texts=texts))[0] == "country-et"


@pytest.mark.parametrize("stat,kept,notes,expect", [
    (None, 0, ["max_items (2) reached: 1 later source(s) were not collected"], "not collected (max_items reached"),
    ({"taken": 0, "error": "not available"}, 0, [], "not available"),
    ({"taken": 0, "error": "HTTPStatusError: HTTP 404 for x"}, 0, [], "fetch failed: HTTPStatusError: HTTP 404"),
    ({"taken": 0, "error": None}, 0, ["api: G: GDELT returned no articles for this query (24h)"],
     "returned 0 items (GDELT returned no articles"),
    ({"taken": 0, "error": None}, 0, [], "returned 0 items"),
    ({"taken": 7, "error": None}, 0, [], "all 7 items filtered by keywords"),
    ({"taken": 7, "error": None}, 3, ["api: G: cut at 7 items by its limit (7); more were available"],
     "ok: 3 of 7 kept; cut at 7"),
])
def test_source_status(stat, kept, notes, expect):
    assert theaters.source_status(stat, kept, "api: G", notes).startswith(expect)


def test_source_table_tells_not_collected_from_filtered_and_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    data = {
        "briefing": {"title": "Conflicts", "interests": ["armed conflict"], "keywords": ["zzzz"],
                     "max_items": 2, "extensions": ["theaters"]},
        "sources": {
            "feeds": [{"name": "Broken feed", "url": BROKEN_FEED}],
            "api": [{"adapter": "reliefweb", "name": "ReliefWeb"},
                    {"adapter": "gdelt_doc", "name": "GDELT", "query": "q"}],
        },
    }
    with Store(tmp_path / "db.sqlite") as store, store.record_run("briefing") as r:
        res = engine.run_briefing(bconfig.parse("conf", data), lm=FakeLM(), conn=store.conn, run_id=r.id,
                                  fetcher=FakeFetcher(), now=NOW)
        html = engine.build_report(res).render()
    assert "fetch failed: HTTPStatusError" in html
    assert "all 2 items filtered by keywords" in html            # ReliefWeb: collected, then filtered
    assert "not collected (max_items reached before this source)" in html  # GDELT never called
    assert "nothing matched" not in html


def test_shipped_config_budget_lets_every_source_kind_collect(monkeypatch):
    """With every source returning a full list, ReliefWeb and GDELT (collected
    last) still contribute: max_items covers the sum of all limits."""
    cfg = bconfig.load("conflicts")
    assert sum(int(e.get("limit", cfg.max_items)) for _, e in cfg.entries()) <= cfg.max_items

    def full(kind):
        def adapter(entry, ctx):
            for i in range(ctx.limit + 50):  # always more than asked for
                yield Doc(url=f"https://{kind}.example/{entry['name'].replace(' ', '-')}/{i}", title=f"item {i}")
        return adapter

    for name in ("feeds", "search", "reliefweb", "gdelt_doc"):
        spec = sources.ADAPTERS[name]
        monkeypatch.setitem(sources.ADAPTERS, name, sources.AdapterSpec(name, full(name), spec.keys,
                                                                         spec.required, spec.validate))
    c = SourceContext(briefing=cfg, fetcher=None, lm=None, conn=None, now=NOW, limit=cfg.max_items)
    docs = engine.collect(cfg, c)
    by_entry: dict[str, int] = {}
    for d in docs:
        by_entry[d.meta["entry"]] = by_entry.get(d.meta["entry"], 0) + 1
    assert by_entry["ReliefWeb"] == 60
    assert by_entry["GDELT: armed conflict"] == 75 and by_entry["GDELT: ceasefires and talks"] == 40
    assert not any("max_items" in n for n in c.notes)
    assert len(c.entry_stats) == len(cfg.entries())


# ---------------------------------------------------------------------------
# Coordinator fix round: magnitudes, false negatives, folded theaters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,figure,quote,shown", [
    # magnitude words are part of the number as quoted
    ("displacement", "2 million", "2 million people have been displaced since April", "2 million"),
    ("displacement", "2,000,000", "2 million people have been displaced", "2 million"),
    ("displacement", "1.5 billion", "1.5 billion people displaced", "1.5 billion"),
    ("displacement", "tens of thousands", "Tens of thousands have fled the city", "Tens of thousands"),
    ("displacement", "thousands", "Tens of thousands have fled the city", "Tens of thousands"),
    ("displacement", "hundreds of thousands", "hundreds of thousands of people displaced", "hundreds of thousands"),
    ("displacement", "hundreds of thousands", "thousands of people displaced", None),
    ("casualties", "2", "2 million people were killed", None),              # 2 is not 2 million
    # (a) a subset clause between the figure and its word
    ("casualties", "40", "At least 40 civilians, including 12 children, were killed", "40"),
    ("casualties", "40", "40 people, among them 3 medics, were killed", "40"),
    ("casualties", "40", "40 people, of whom 9 were soldiers, were killed", "40"),
    # (b) "the toll ... has risen to N"
    ("casualties", "40", "The number of people killed in Tuesday's attack on the village has risen to 40", "40"),
    ("casualties", "40", "The death toll from the strikes on the eastern market has climbed to 40", "40"),
    ("casualties", "40", "People killed in Tuesday's attack on the village near the river were 40", None),
    # (c) spelled-out numbers
    ("casualties", "5", "Five soldiers were killed at the checkpoint", "Five"),
    ("casualties", "25", "twenty-five civilians died", "twenty-five"),
    ("casualties", "250", "two hundred and fifty people were killed", "two hundred and fifty"),
    ("displacement", "3000", "three thousand families fled", "three thousand"),
    ("casualties", "5", "Fifteen soldiers were killed", None),
    # (d) ":" does not end a sentence
    ("casualties", "40", "Killed: 40.", "40"),
])
def test_figure_support_fix_round(kind, figure, quote, shown):
    assert theaters.figure_support(kind, figure, quote) == shown


def slugs(stories):
    return {s.story_id: t.slug for t in theaters.group(stories) for s in t.stories}


def test_syria_stays_in_one_theater():
    got = slugs([story("il", ["IL", "SY"]), story("sy", ["SY"]), story("tr", ["SY", "TR"])])
    assert set(got.values()) == {"israel-gaza-lebanon"}
    # no Israel theater this run: the pair folds into the Syria theater
    got = slugs([story("sy", ["SY"]), story("tr", ["TR", "SY"])])
    assert got == {"sy": "country-sy", "tr": "country-sy"}


def test_somalia_stays_in_one_theater():
    got = slugs([story("1", ["SO"]), story("2", ["US", "SO"]), story("3", ["ET", "SO"]), story("4", ["SO", "KE"])])
    assert set(got.values()) == {"country-so"}


def test_afghanistan_pairs_fold_into_the_active_country():
    got = slugs([story("1", ["AF"]), story("2", ["AF", "PK"]), story("3", ["PK", "AF"])])
    assert set(got.values()) == {"country-af"}


def test_india_pakistan_pair_kept_when_neither_is_active_else_folded():
    assert set(slugs([story("1", ["IN", "PK"]), story("2", ["PK", "IN"])]).values()) == {"pair-in-pk"}
    got = slugs([story("1", ["IN", "PK"]), story("2", ["PK"]), story("3", ["PK"]), story("4", ["IN"])])
    assert got["1"] == "country-pk"  # more stories wins
    assert got["2"] == got["3"] == "country-pk" and got["4"] == "country-in"


def test_russia_only_stories_join_the_active_ukraine_war():
    got = slugs([story("ua", ["UA"]), story("ru", ["RU"]), story("ruus", ["RU", "US"]), story("by", ["BY"])])
    assert set(got.values()) == {"russia-ukraine"}
    # no Ukraine story this run: a Russia-only story is its own theater
    assert slugs([story("ru", ["RU"])]) == {"ru": "country-ru"}
    # an external actor beside a local country never pulls it into that war
    assert slugs([story("ua", ["UA"]), story("so", ["RU", "SO"])])["so"] == "country-so"


def test_source_status_partial_failure():
    status = theaters.source_status({"taken": 5, "error": "RuntimeError: boom"}, 5, "api: G", [])
    assert status == "partial: 5 collected, then failed: RuntimeError: boom"
