"""Source lean + blindspot extension (#69): coverage balance, blindspot
threshold/min-sources, config precedence, and unknown handling. No network,
no model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lmsa.briefing import config as bconfig
from lmsa.briefing.adapters import lean
from lmsa.briefing.engine import StoryResult
from lmsa.briefing.sources import Doc
from lmsa.report import Report, StackedBar, StorySource, StoryExtra

NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)


def _doc(source, lean_val="unknown", reliability="unknown", url=None):
    return Doc(url=url or f"https://x.example/{source}", title=source, source=source, published=NOW,
              meta={"lean": lean_val, "reliability": reliability})


def _story(docs, sid="s1", headline="Story"):
    return StoryResult(story_id=sid, docs=docs, centroid=[0.0],
                       extract={"headline": headline, "summary": "", "severity": 0, "actors": []})


# ---------------------------------------------------------------------------
# Balance computation
# ---------------------------------------------------------------------------

def test_coverage_balance_counts_by_distinct_source_not_article_count():
    docs = [_doc("A", "left"), _doc("A", "left", url="https://x.example/A/2"),  # same source twice
            _doc("B", "right")]
    assert lean.coverage_balance(docs) == {"total": 2, "left": 1, "right": 1, "leaning": "left", "share": 0.5}


def test_state_media_intl_and_center_count_toward_total_only():
    docs = [_doc("A", "left"), _doc("B", "left"), _doc("C", "left"), _doc("D", "intl"), _doc("E", "state-media")]
    bal = lean.coverage_balance(docs)
    assert bal["total"] == 5 and bal["left"] == 3 and bal["right"] == 0
    assert bal["leaning"] == "left" and bal["share"] == pytest.approx(3 / 5)


def test_unknown_lean_never_counted():
    assert lean.coverage_balance([_doc("A", "left"), _doc("B", "unknown"), _doc("C", "unknown")])["total"] == 1


def test_no_rated_sources_or_no_side_is_not_leaning():
    assert lean.coverage_balance([_doc("A", "unknown")]) == \
        {"total": 0, "left": 0, "right": 0, "leaning": None, "share": 0.0}
    assert lean.coverage_balance([_doc("A", "center"), _doc("B", "intl")])["leaning"] is None


# ---------------------------------------------------------------------------
# Blindspot threshold / min sources
# ---------------------------------------------------------------------------

def test_blindspot_needs_the_minimum_source_count_even_at_100_percent():
    assert not lean.is_blindspot([_doc("A", "left"), _doc("B", "left")])  # only 2 rated sources


def test_blindspot_at_the_80_percent_edge():
    docs = [_doc("A", "left"), _doc("B", "left"), _doc("C", "left"), _doc("D", "left"), _doc("E", "right")]
    assert lean.is_blindspot(docs)  # 4/5 = 80%, exactly at the threshold


def test_not_blindspot_just_under_threshold():
    docs = [_doc("A", "left"), _doc("B", "left"), _doc("C", "left"), _doc("D", "right")]
    assert not lean.is_blindspot(docs)  # 3/4 = 75%


def test_blindspot_on_the_right_side_too():
    docs = [_doc("A", "right"), _doc("B", "center-right"), _doc("C", "right")]
    assert lean.is_blindspot(docs)


def test_intl_and_state_media_dilute_a_would_be_blindspot():
    # 3 of 3 *side-rated* sources are left, but 2 more rated sources (intl,
    # state-media) count toward the total -> share drops under 80%.
    docs = [_doc("A", "left"), _doc("B", "left"), _doc("C", "left"), _doc("D", "intl"), _doc("E", "state-media")]
    assert not lean.is_blindspot(docs)


# ---------------------------------------------------------------------------
# render(): bar + low-reliability warning, unknown handling
# ---------------------------------------------------------------------------

def test_render_skips_bar_when_every_doc_unknown():
    parts = lean.render(_story([_doc("A"), _doc("B")]), None)
    assert not any(isinstance(p, StackedBar) for p in parts)


def test_render_bar_by_lean_and_low_reliability_warning():
    story = _story([_doc("A", "left", "low"), _doc("B", "right", "low")])
    parts = lean.render(story, None)
    bars = [p for p in parts if isinstance(p, StackedBar)]
    extras = [p for p in parts if isinstance(p, StoryExtra)]
    assert len(bars) == 1
    assert {l: v for l, v, _ in bars[0].segments} == {"left": 1, "right": 1}
    assert len(extras) == 1 and extras[0].tone == "warn"


def test_render_no_warning_when_a_high_or_mixed_source_covers_it():
    story = _story([_doc("A", "left", "low"), _doc("B", "right", "high")])
    assert not any(isinstance(p, StoryExtra) for p in lean.render(story, None))


def test_render_no_warning_with_no_docs():
    assert lean.render(_story([]), None) == []


# ---------------------------------------------------------------------------
# render_sources(): reliability badges
# ---------------------------------------------------------------------------

def test_render_sources_badges_reliability_and_leaves_the_more_link_alone():
    docs = [_doc("A", "left", "high", url="https://x.example/a"),
            _doc("B", "right", "unknown", url="https://x.example/b")]
    story = _story(docs)
    links = [StorySource("A: title", "https://x.example/a"), StorySource("B: title", "https://x.example/b"),
             StorySource("+1 more")]  # href=None, as the engine appends it
    out = lean.render_sources(story, links, None)
    assert out[0].badge == "high" and out[0].tone == "acc"
    assert out[1].badge == "unknown" and out[1].tone == "muted"
    assert out[2] is links[2]  # untouched


# ---------------------------------------------------------------------------
# section(): the Blindspot report section
# ---------------------------------------------------------------------------

def test_section_lists_only_blindspot_stories_linked_to_their_card():
    balanced = _story([_doc("A", "left"), _doc("B", "right"), _doc("C", "center")],
                      sid="balanced", headline="Balanced story")
    blind = _story([_doc("D", "left"), _doc("E", "left"), _doc("F", "left"), _doc("G", "left"), _doc("H", "right")],
                   sid="blind", headline="Blindspot story")
    report = Report("T")
    lean.section(report, [balanced, blind], None)
    html = report.render()
    assert "Blindspot story" in html and "#story-blind" in html
    assert "Balanced story" not in html


def test_section_shows_empty_callout_with_no_blindspots():
    balanced = _story([_doc("A", "left"), _doc("B", "right"), _doc("C", "center")])
    report = Report("T")
    lean.section(report, [balanced], None)
    assert "No blindspots this run." in report.render()


# ---------------------------------------------------------------------------
# config precedence: per-source entry > sources.toml > unknown
# ---------------------------------------------------------------------------

BASE = {"briefing": {"title": "T", "interests": ["x"]}}


def test_source_tags_precedence(tmp_path):
    sources_file = tmp_path / "sources.toml"
    sources_file.write_text('[[sources]]\nname = "Shared Feed"\nlean = "left"\nreliability = "high"\n',
                            encoding="utf-8")
    data = {**BASE, "sources": {"feeds": [
        {"url": "https://a.example/feed.xml", "name": "Shared Feed"},                    # picks up shared tags
        {"url": "https://b.example/feed.xml", "name": "Shared Feed", "lean": "right"},   # entry's own lean wins
        {"url": "https://c.example/feed.xml", "name": "Untagged Feed"},                  # no tag anywhere
    ]}}
    cfg = bconfig.parse("t", data, sources_file=sources_file)
    a, b, c = cfg.sources["feeds"]
    assert a["lean"] == "left" and a["reliability"] == "high"
    assert b["lean"] == "right" and b["reliability"] == "high"  # override is per-field, not all-or-nothing
    assert c["lean"] == "unknown" and c["reliability"] == "unknown"


def test_source_tags_file_missing_is_fine(tmp_path):
    cfg = bconfig.parse("t", {**BASE, "sources": {"feeds": [{"url": "https://a.example/feed.xml"}]}},
                        sources_file=tmp_path / "does-not-exist.toml")
    assert cfg.sources["feeds"][0]["lean"] == "unknown"
    assert cfg.sources["feeds"][0]["reliability"] == "unknown"


def test_source_tags_file_validates_lean(tmp_path):
    sources_file = tmp_path / "sources.toml"
    sources_file.write_text('[[sources]]\nname = "X"\nlean = "sideways"\n', encoding="utf-8")
    with pytest.raises(bconfig.ConfigError, match="lean"):
        bconfig.load_source_tags(sources_file)


def test_bad_source_tags_file_never_breaks_a_briefing(tmp_path, capsys):
    sources_file = tmp_path / "sources.toml"
    sources_file.write_text('[[sources]]\nname = "X"\nlean = "sideways"\n', encoding="utf-8")
    cfg = bconfig.parse("t", {**BASE, "sources": {"feeds": [{"url": "https://a.example/feed.xml"}]}},
                        sources_file=sources_file)
    assert cfg.sources["feeds"][0]["lean"] == "unknown"
    assert "ignoring source tags" in capsys.readouterr().err


def test_source_tags_file_rejects_unknown_key(tmp_path):
    sources_file = tmp_path / "sources.toml"
    sources_file.write_text('[[sources]]\nname = "X"\nbias = "left"\n', encoding="utf-8")
    with pytest.raises(bconfig.ConfigError, match="bias"):
        bconfig.load_source_tags(sources_file)


def test_shipped_sources_toml_is_valid_and_has_entries():
    tags = bconfig.load_source_tags()
    assert tags and all(v["lean"] in bconfig.LEANS and v["reliability"] in bconfig.RELIABILITY for v in tags.values())
