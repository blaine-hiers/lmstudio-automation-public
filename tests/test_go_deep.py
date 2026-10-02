"""Go deep (#70): bounded research with fake search / fetch / LM Studio --
caps respected and named, citations resolve to fetched pages, quotes are
checked, partial results on a cap, the story card links both ways, and the
dashboard pre-fills (never runs) the launcher. No network, no model."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from lmsa import Store
from lmsa.briefing.adapters import go_deep as ext
from lmsa.config import DASHBOARD_PORT
from lmsa.report import safe_href
from lmsa.search import SearchResult, SearchUnavailable
from lmsa.web import HTTPStatusError
from tasks import go_deep as gd

NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
VOCAB = ["canal", "insurance", "tanker", "houthi", "freight", "premium", "navy", "port"]

PAGES = {
    "https://a.example/canal": ("Canal traffic falls",
        "Traffic through the canal fell by a third in September. Tanker operators said the insurance "
        "premium for the route has tripled since June. The canal authority reported lower revenue."),
    "https://b.example/insurance": ("War risk insurance soars",
        "War risk insurance premium rates for tanker voyages reached record levels this week. Underwriters in "
        "London said the premium now equals one percent of hull value. Freight rates followed insurance higher."),
    "https://c.example/navy": ("Navy escorts expand",
        "The navy expanded escorts for merchant ships near the port. Officials said escorts cut attacks on "
        "tanker traffic. Freight operators welcomed the navy plan but insurance premium rates stayed high."),
    "https://d.example/port": ("Port congestion grows",
        "Port congestion grows as freight is rerouted around the cape. Shipping lines said the port backlog "
        "will last months. Canal revenue fell as freight moved away from the canal route."),
}
QUERIES = {
    "canal traffic": ["https://a.example/canal", "https://d.example/port"],
    "insurance premiums": ["https://b.example/insurance", "https://a.example/canal"],
    "navy escorts": ["https://c.example/navy", "https://missing.example/404"],
}


def html(title: str, body: str) -> bytes:
    paras = "".join(f"<p>{s.strip()}.</p>" for s in body.split(".") if s.strip())
    return (f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1>{paras}"
            "</article></body></html>").encode("utf-8")


class FakeLM:
    def __init__(self):
        self.calls = {"plan": 0, "section": 0, "synth": 0}
        self.prompts: list[str] = []

    def embed(self, texts, *, model=None):
        return [[float(t.lower().count(w)) for w in VOCAB] + [0.01] for t in texts]

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.prompts.append(prompt)
        if system.startswith("You plan"):
            self.calls["plan"] += 1
            return {"questions": [
                {"question": "How far has canal traffic fallen?", "query": "canal traffic"},
                {"question": "How expensive is tanker insurance now?", "query": "insurance premiums"},
                {"question": "Are navy escorts working?", "query": "navy escorts"},
            ]}
        if system.startswith("You write one section"):
            self.calls["section"] += 1
            nums = [int(n) for n in re.findall(r"^\[(\d+)\] ", prompt, re.M)]
            first = nums[0]
            body = prompt.split(f"[{first}] ", 1)[1].split("\n", 2)[1]
            real = " ".join(body.split()[:8])
            return {"text": f"The sources agree [{first}]. One claim has no source [99]. Mixed [{first}, 77].",
                    "quotes": [{"quote": real, "source": first},
                               {"quote": "We will never pay these premiums again", "source": first}]}
        self.calls["synth"] += 1
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", prompt)})
        return {"executive_summary": f"Costs rose [{cited[0]}]; invented [42].",
                "open_questions": ["How long will escorts last?", "  "]}


class FakeFetcher:
    def __init__(self):
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url not in PAGES:
            raise HTTPStatusError(404, url)
        return SimpleNamespace(content=html(*PAGES[url]), url=url, status=200, headers={}, from_cache=False)


class FakeSearx:
    def __init__(self, fail: set[str] = frozenset()):
        self.queries: list[str] = []
        self.fail = fail

    def search(self, query, **kw):
        self.queries.append(query)
        if query in self.fail:
            raise SearchUnavailable("down")
        return [SearchResult(title=PAGES.get(u, ("x", ""))[0], url=u, snippet="", engine="fake", published=None)
                for u in QUERIES.get(query, [])]


class Clock:
    def __init__(self, step: float = 0.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def go(tmp_path, caps=gd.Caps(), *, lm=None, searx=None, fetcher=None, clock=None, story=None, fresh=False):
    lm, searx, fetcher = lm or FakeLM(), searx or FakeSearx(), fetcher or FakeFetcher()
    with Store(tmp_path / "db.sqlite") as store:
        res = gd.research("Red Sea shipping costs", lm=lm, searx=searx, fetcher=fetcher, conn=store.conn,
                          caps=caps, clock=clock or Clock(), story=story, fresh=fresh)
    return res, lm, searx, fetcher


def test_full_run_citations_resolve_to_fetched_pages_and_quotes_are_checked(tmp_path):
    res, lm, searx, fetcher = go(tmp_path)
    assert lm.calls == {"plan": 1, "section": 3, "synth": 1}
    assert searx.queries == ["canal traffic", "insurance premiums", "navy escorts"]
    assert res.caps_hit == []
    fetched = {s.n: s.url for s in res.sources}
    assert set(fetched.values()) == set(PAGES)  # the 404 is skipped, canal page fetched once
    assert fetcher.calls.count("https://a.example/canal") == 1
    assert any("missing.example" in n and "HTTPStatusError" in n for n in res.notes)
    for q in res.questions:
        assert q.text and q.cited
        assert set(q.cited) <= set(q.shown) <= set(fetched)
        # every surviving [n] is a fetched page the model was shown; invented ones are [?]
        for n in re.findall(r"\[(\d+)\]", q.text):
            assert int(n) in q.shown
        assert "[99]" not in q.text and "[?]" in q.text and q.dropped == 2
        assert [x["verified"] for x in q.quotes] == [True, False]
        assert q.quotes[0]["source"] in q.shown
    assert "[42]" not in res.summary and "[?]" in res.summary and res.summary_dropped == 1
    assert res.open_questions == ["How long will escorts last?"]
    assert res.citations_dropped == 7
    # the model saw the fetched text only as bounded, numbered passages marked untrusted
    section_prompts = [p for p in lm.prompts if "SUB-QUESTION" in p]
    assert all("untrusted" in p and len(p) < gd.SECTION_PASSAGE_CHARS + 500 for p in section_prompts)
    page = gd.build_report(res).render()
    assert "Stopped at a cap" not in page and "Sources" in page
    assert "https://b.example/insurance" in page and "unverified" in page


def test_rerun_reuses_remembered_answers(tmp_path):
    go(tmp_path)
    res, lm, _, _ = go(tmp_path)
    assert lm.calls == {"plan": 0, "section": 0, "synth": 0}
    assert res.memo.hits == 5 and all(q.text for q in res.questions)
    assert res.used["chars"] == 0


def test_page_cap_stops_fetching_and_the_report_says_so(tmp_path):
    res, lm, searx, fetcher = go(tmp_path, gd.Caps(pages=2))
    assert len(fetcher.calls) == 2 and res.caps_hit == ["pages"]
    assert searx.queries == ["canal traffic"]  # no searches once pages are spent
    # later sub-questions are still drafted from what was fetched: a partial result
    assert all(q.text for q in res.questions)
    assert {n for q in res.questions for n in q.cited} <= {1, 2}
    page = gd.build_report(res).render()
    assert "partial" in page and "--max-pages (2)" in page


def test_search_cap(tmp_path):
    res, _, searx, _ = go(tmp_path, gd.Caps(searches=1))
    assert searx.queries == ["canal traffic"] and res.caps_hit == ["searches"]
    assert "--max-searches (1)" in gd.build_report(res).render()


def test_chars_cap_stops_drafting_cleanly(tmp_path):
    res, lm, _, _ = go(tmp_path, gd.Caps(chars=7_000))
    assert "chars" in res.caps_hit
    assert res.used["chars"] <= 7_000
    assert lm.calls["section"] < 3
    skipped = [q for q in res.questions if not q.text]
    assert skipped and all(q.skipped == "stopped: --max-chars" for q in skipped)
    page = gd.build_report(res).render()
    assert "--max-chars (7,000)" in page and "SKIPPED" in page


def test_time_cap_stops_and_keeps_what_was_done(tmp_path):
    res, lm, _, _ = go(tmp_path, gd.Caps(seconds=10), clock=Clock(step=0.5))
    assert "time" in res.caps_hit
    assert any(q.skipped == "stopped: --max-minutes" for q in res.questions)
    assert res.used["seconds"] >= 10
    assert "--max-minutes" in gd.build_report(res).render()


def test_failed_search_and_bad_model_answer_are_isolated(tmp_path):
    class Flaky(FakeLM):
        def extract(self, prompt, schema, **kw):
            if "Are navy escorts" in prompt:
                raise RuntimeError("model fell over")
            return super().extract(prompt, schema, **kw)

    res, _, _, _ = go(tmp_path, lm=Flaky(), searx=FakeSearx(fail={"insurance premiums"}))
    assert any("search failed" in n and "SearchUnavailable" in n for n in res.notes)
    navy = res.questions[2]
    assert navy.error.startswith("RuntimeError") and not navy.text
    assert res.questions[0].text and res.summary


def test_planning_failure_falls_back_to_the_topic(tmp_path):
    class NoPlan(FakeLM):
        def extract(self, prompt, schema, *, system=None, **kw):
            if system.startswith("You plan"):
                return {"questions": []}
            return super().extract(prompt, schema, system=system, **kw)

    res, _, searx, _ = go(tmp_path, lm=NoPlan())
    assert [q.question for q in res.questions] == ["Red Sea shipping costs"]


def test_check_citations_forms():
    text, cited, dropped = gd.check_citations("a [1] b [2, 3] c [4;1] d [x]", {1, 3})
    assert text == "a [1] b [?][3] c [?][1] d [x]" and cited == [1, 3] and dropped == 2
    assert gd.check_citations("none [5]", set()) == ("none [?]", [], 1)


@pytest.mark.parametrize("text,allowed,want,cited,dropped", [
    ("x [1-3] y", {1, 2, 3}, "x [1][2][3] y", [1, 2, 3], 0),
    ("x [1\u20133] y", {1, 2, 3}, "x [1][2][3] y", [1, 2, 3], 0),       # en dash
    ("x [1 - 2, 4] y", {1, 2, 4}, "x [1][2][4] y", [1, 2, 4], 0),
    ("x [1-4] y", {1, 2, 3}, "x [?] y", [], 1),                         # 4 not shown: whole range dropped
    ("x [3-1] y", {1, 2, 3}, "x [?] y", [], 1),                         # backwards
    ("x [1-500] y", set(range(1, 501)), "x [?] y", [], 1),              # absurd range
    ("x [12345] y", {12345}, "x [?] y", [], 1),                         # 5+ digits never a source number
    ("x [00001] y", {1}, "x [?] y", [], 1),
    ("x [2, 1-2] y", {1, 2}, "x [2][1][2] y", [2, 1], 0),
])
def test_citation_ranges_and_long_numbers(text, allowed, want, cited, dropped):
    assert gd.check_citations(text, allowed) == (want, cited, dropped)


class LongLM(FakeLM):
    """Maximal-length questions, to stress the synthesis budget."""

    def extract(self, prompt, schema, *, system=None, **kw):
        if system.startswith("You plan"):
            self.calls["plan"] += 1
            return {"questions": [{"question": f"{w} " + "canal insurance tanker freight " * 12, "query": q}
                                  for w, q in (("A", "canal traffic"), ("B", "insurance premiums"),
                                               ("C", "navy escorts"))]}
        return super().extract(prompt, schema, system=system, **kw)


@pytest.mark.parametrize("chars", range(4_000, 24_001, 1_000))
def test_synthesis_never_overruns_the_chars_cap_or_silently_disappears(tmp_path, chars):
    lm = LongLM()
    with Store(tmp_path / "db.sqlite") as store:
        res = gd.research("T" * 300, lm=lm, searx=FakeSearx(), fetcher=FakeFetcher(), conn=store.conn,
                          caps=gd.Caps(chars=chars), clock=Clock(), fresh=True)
    assert res.used["chars"] <= chars
    if any(q.text for q in res.questions):
        assert res.summary or any(n.startswith("executive summary skipped: chars cap") for n in res.notes)
        if not res.summary:
            assert "chars" in res.caps_hit


class Malformed(FakeLM):
    def __init__(self, section, synth):
        super().__init__()
        self.section, self.synth = section, synth

    def extract(self, prompt, schema, *, system=None, **kw):
        if system.startswith("You write one section") and "Are navy escorts" in prompt:
            self.calls["section"] += 1
            return self.section
        if system.startswith("You write the executive"):
            self.calls["synth"] += 1
            return self.synth
        return super().extract(prompt, schema, system=system, **kw)


@pytest.mark.parametrize("section", [
    ["not", "an", "object"],
    {"text": ["a list"], "quotes": []},
    {"text": None, "quotes": None},
    "just a string",
])
@pytest.mark.parametrize("synth", [
    ["list"],
    {"executive_summary": {"x": 1}, "open_questions": "nope"},
])
def test_malformed_model_answers_become_notes(tmp_path, section, synth):
    res, _, _, _ = go(tmp_path, lm=Malformed(section, synth), fresh=True)
    navy = res.questions[2]
    assert not navy.text and navy.error and any("skipped" in n and "navy" in n for n in res.notes)
    assert res.questions[0].text and res.questions[1].text  # the run went on
    assert not res.summary and any(n.startswith("executive summary failed") for n in res.notes)
    gd.build_report(res).render()


def test_quotes_of_the_wrong_shape_are_ignored_not_fatal(tmp_path):
    lm = Malformed({"text": "Escorts help [1].", "quotes": {"quote": "x", "source": 1}},
                   {"executive_summary": "ok", "open_questions": []})
    res, _, _, _ = go(tmp_path, lm=lm, fresh=True)
    navy = res.questions[2]
    assert navy.text and navy.quotes == [] and any("quotes were a dict" in n for n in res.notes)
    assert res.summary == "ok"


# ---------------------------------------------------------------------------
# Story: seed, links both ways
# ---------------------------------------------------------------------------

def seed_story(conn, out_dir: Path) -> None:
    report = out_dir / "briefing-shipping-20260925-0600.html"
    conn.execute(
        "INSERT INTO briefing_stories(briefing, story_id, item_id, fingerprint, headline, summary, status, score, "
        "severity, countries_json, urls_json, data_json, first_seen, last_seen, first_run_id, last_run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("shipping", "abc123def456", "shipping:story:abc123def456", "fp", "Red Sea costs climb",
         "Insurers raise premiums.", "new", 1.0, 1, "[]",
         json.dumps(["https://a.example/canal", "https://b.example/insurance", "https://c.example/navy"]),
         json.dumps({"extract": {"highlights": ["Premiums tripled"]}}), NOW.isoformat(), NOW.isoformat(), 7, 7))
    conn.execute("INSERT INTO briefing_runs(briefing, run_id, started, status, report_path) VALUES (?,?,?,?,?)",
                 ("shipping", 7, NOW.isoformat(), "ok", str(report)))
    conn.commit()


def test_story_seed_links_back_and_the_card_links_forward(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(out))
    with Store(tmp_path / "db.sqlite") as store:
        seed_story(store.conn, out)
        with pytest.raises(ValueError):
            gd.load_story(store.conn, "shipping:nope")
        with pytest.raises(ValueError):
            gd.load_story(store.conn, "../x:abc")
        story = gd.load_story(store.conn, "shipping:abc123def456")
    assert story["headline"] == "Red Sea costs climb" and story["highlights"] == ["Premiums tripled"]

    fetcher = FakeFetcher()
    res, lm, _, _ = go(tmp_path, story=story, fetcher=fetcher)
    assert fetcher.calls[:2] == story["urls"][:gd.SEED_PAGES]  # the story's own articles first
    assert "Insurers raise premiums." in lm.prompts[0]
    assert [s.seed for s in res.sources[:2]] == [True, True]

    href = gd.story_href(story, out)
    assert href == "../../briefing-shipping-20260925-0600.html#story-abc123def456"
    page = gd.build_report(res, story_href=href).render()
    assert f'href="{href}"' in page and "Origin" in page

    path = out / "go-deep-red-sea-20260925-0600.html"
    path.write_text(page, encoding="utf-8")
    with Store(tmp_path / "db.sqlite") as store:
        gd.record(store.conn, 41, res, path, NOW)
        gd.record(store.conn, 41, res, path, NOW)  # same run: no duplicate row
        assert store.conn.execute("SELECT COUNT(*) FROM go_deep_reports").fetchone()[0] == 1
        s = SimpleNamespace(story_id="abc123def456")
        ctx = SimpleNamespace(briefing=SimpleNamespace(name="shipping"), conn=store.conn)
        extras = ext.render(s, ctx)
    launch, deep = extras
    # absolute, so it also works from a report opened from disk (dashboard running)
    assert safe_href(launch.href) == launch.href
    assert launch.href.startswith(f"http://127.0.0.1:{DASHBOARD_PORT}/?") and launch.text == "open in dashboard"
    q = parse_qs(urlsplit(launch.href).query)
    assert q == {"task": ["go_deep"], "story": ["shipping:abc123def456"]}
    assert urlsplit(launch.href).fragment == "task-go_deep"
    assert deep.href == path.name and deep.label == "deep report"


def test_card_link_without_a_report_or_table():
    import sqlite3

    conn = sqlite3.connect(":memory:")  # no go_deep_reports table: launch link only
    ctx = SimpleNamespace(briefing=SimpleNamespace(name="shipping"), conn=conn)
    extras = ext.render(SimpleNamespace(story_id="s1"), ctx)
    assert [x.label for x in extras] == ["go deep"]
    assert ext.render(SimpleNamespace(story_id="s1"), None) == []


def test_extension_is_default_on_for_every_briefing():
    from lmsa.briefing import sources

    sources.load_plugins()
    assert sources.EXTENSIONS["go_deep"].default is True
    assert "go_deep" in [e.name for e in sources.active_extensions(SimpleNamespace(extensions=[]))]


def test_main_needs_exactly_one_of_story_or_topic():
    with pytest.raises(SystemExit):
        gd.main([])
    with pytest.raises(SystemExit):
        gd.main(["--story", "a:b", "--topic", "x"])
    with pytest.raises(SystemExit):
        gd.main(["--topic", "x", "--max-pages", "0"])


# ---------------------------------------------------------------------------
# Dashboard pre-fill
# ---------------------------------------------------------------------------

def test_dashboard_prefills_go_deep_from_a_story_link():
    from tasks import dashboard as db

    root = Path(__file__).resolve().parents[1]
    t = next(t for t in db.discover_tasks(root / "tasks") if t.name == "go_deep")
    assert t.cadence is None and not t.new  # manual only, listed in config/dashboard.toml
    flags = {f.flag for f in t.fields}
    assert {"--story", "--topic", "--max-pages", "--max-chars", "--max-minutes", "--max-searches"} <= flags

    q = parse_qs("task=go_deep&story=shipping%3Aabc&max_pages=5&max-chars=abc&topic=%3Cscript%3E&bogus=1")
    vals = db.prefill_values(t, q)
    # only the allowlisted --story / --topic; caps are never pre-filled
    assert vals == {"--story": "shipping:abc", "--topic": "<script>"}
    html_ = db._launcher_html(None, t, None, values=vals)
    assert 'value="shipping:abc"' in html_ and 'value="&lt;script&gt;"' in html_ and "<script>" not in html_

    # write flags, checkboxes and control characters are never pre-filled
    # every other task ignores the query string, write-path tasks included
    root_tasks = {t.name: t for t in db.discover_tasks(root / "tasks")}
    for name, query in (("doc_inbox", "task=doc_inbox&inbox=C%3A%5CUsers&limit=5"),
                        ("photo_index", "task=photo_index&root=C%3A%5C&max_new_per_run=9"),
                        ("ask", "task=ask&question=hello&k=3")):
        assert db.prefill_values(root_tasks[name], parse_qs(query)) == {}, name

    writer = db.TaskInfo(name="w", doc="", group="Other", fields=(
        db.Field("--apply", "bool"), db.Field("--undo", "text", write=True), db.Field("--q", "text")))
    assert db.prefill_values(writer, {"apply": ["1"], "undo": ["7"], "q": ["a\nb"]}) == {}


def test_main_builds_searxng_with_the_configured_url():
    # Tests inject a fake client; the real entry point must pass a base URL.
    import inspect
    from tasks import go_deep
    assert "SearXNG(settings().searxng_url)" in inspect.getsource(go_deep.main)
