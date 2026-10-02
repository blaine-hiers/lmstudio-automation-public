"""lmsa.email_report + the master report's `full` section mode (#119): a
saved lmsa.report page converted into the email body, whole, for iOS Gmail
and iOS Mail. No test sends mail or calls a model."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from lmsa import Card, Link, Marker, Report, Tile
from lmsa import email_report
from lmsa.digest import OK, Digest, DigestItem, RunRow, SectionStatus
from lmsa.email_render import GMAIL_CLIP, EmailSection, render_email
from lmsa.report import StackedBar, Story, StoryExtra, StoryQuote, StorySource
from lmsa.store import Store
from tasks import master_report as mr

NOW = datetime(2026, 9, 28, 11, 45, tzinfo=timezone.utc)
HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>'
BANNED = ("<style", "<script", "<svg", "<img", "<input", "<details", "<button", "var(--", "fonts.googleapis",
          "<link", "file://", "javascript:")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_IMAP_USER", "owner@example.test")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))


def _every_component() -> Report:
    r = Report("All components", kicker="kicker-text", subtitle="subtitle-text", sticker="sticker-text",
               ticker=["ticker-text"], source="tasks.test")
    r.tiles([Tile("TopTile", 42, delta="+5", tone="hot", note="tile-note", spark=[1, 3, 2, 5]),
             Tile("Second", "$1,284", tone="warn"), Tile("Third", 3.5, tone="info")])
    r.section("Text section").text("First paragraph.\n\nSecond paragraph.")
    r.section("Tiles section").tiles([Tile("InnerTile", 7, tone="violet")])
    r.section("Wide table", tone="warn").table(
        ["Date", "Merchant", "Category", "Amount", "Link"],
        [["2026-09-21", "Hetzner", "infra", 41.20, Link("receipt", "https://example.com/r1")],
         ["2026-09-19", "Local", "misc", 3, Link("local-link-text", "file:///C:/out/r.html")],
         ["2026-09-12", HOSTILE, "api", 212.00, Link("anchor-link-text", "#top")]])
    r.section("Narrow table").table(["Sender", "Volume"], [["alice@example.com", 12], ["bob@example.com", 7]])
    r.section("Bars section").bars([("github.com", 31), ("substack.com", 22)], unit=" msgs")
    r.section("Heatmap section").heatmap({date(2026, 9, 1) + timedelta(days=i): i % 5 for i in range(30)})
    r.section("Grid heatmap section").grid_heatmap(["Mon", "Tue"], ["00", "12"], [[1, 2], [3, 4]])
    r.section("Map section").anchor("geo-anchor").world_map(
        {"US": 3, "JP": 1}, links={"US": "#geo-anchor"},
        markers=[Marker(35.7, 139.7, "marker-label-tokyo", href="https://example.com/q")])
    r.section("Callout section").callout("Callout body text.", label="rule", tone="info")
    r.section("Cards section", tone="hot").cards([
        Card("Card with link", "card-meta", "Card body one.\n\nCard body two.", tags=["tag-a", "tag-b"],
             tone="hot", href="https://example.com/card?a=1&b=2", ribbon="urgent"),
        Card("Card with local link", href="file:///C:/output/x.html", image="data:image/png;base64,AAAA"),
        Card("Card checkoff", "billing", msg_id="179000001"),
        Card(HOSTILE, body=HOSTILE),
    ]).handled_hidden([Card("Hidden handled card", "office", msg_id="179000002", handled=True)])
    r.section("Stories section", tone="info").stories([Story(
        "Story headline", summary="Story summary text.", meta="story-meta", badge="new", tone="info",
        highlights=["Highlight one", "Highlight two"],
        quotes=[StoryQuote("Verified quote.", "https://example.com/a", verified=True),
                StoryQuote("Unverified quote.", "javascript:alert(1)", verified=False)],
        timeline=[("2026-09-17", "Timeline event one"), ("2026-09-25", "Timeline event two")],
        why="Why text here.",
        sources=[StorySource("Source A", "https://example.com/a", badge="high", tone="acc"),
                 Link("Source B", "https://example.com/b")],
        tags=["story-tag"],
        extras=[StoryExtra("market odds", "Odds text", "https://example.com/m")],
        bars=[StackedBar([("left", 1, "info"), ("right", 3, "hot")], label="coverage by lean")],
        anchor="story-1",
    )])
    r.section("Raw section").raw(
        '<nav><a href="javascript:alert(1)">raw-js-link</a><button type="button">raw-button</button></nav>'
        '<div style="position:fixed;top:0">raw-styled-text</div><svg><text>svg-text</text></svg>'
        "<script>var leaked = 1;</script>")
    return r


def _write(r: Report, tmp_path: Path, name: str = "report.html") -> Path:
    p = tmp_path / name
    p.write_text(r.render(), encoding="utf-8")
    return p


def _email(path: Path, title: str = "task"):
    return render_email("Morning", [EmailSection(title, SectionStatus(OK), Digest(), mode="full",
                                                 full=email_report.load(path))])


# -- conversion of every component --------------------------------------------------

EVERY_TEXT = [
    "subtitle-text", "TopTile", "42", "+5", "tile-note", "$1,284", "3.50",
    "Text section", "First paragraph.", "Second paragraph.", "InnerTile",
    "Wide table", "Hetzner", "infra", "41.20", "receipt", "local-link-text", "anchor-link-text",
    "alice@example.com", "12", "bob@example.com",
    "Bars section", "github.com", "31 msgs", "substack.com",
    "Callout section", "RULE", "Callout body text.",
    "Card with link", "card-meta", "Card body one.", "Card body two.", "#tag-a", "URGENT",
    "Card with local link", "Card checkoff", "1 handled (hidden in the report)", "Hidden handled card",
    "Story headline", "NEW", "story-meta", "Story summary text.", "Highlight one", "Highlight two",
    "Verified quote.", "Unverified quote.", "unverified", "Timeline event one", "2026-09-25",
    "Why it matters:", "Why text here.", "Source A", "[high]", "Source B", "#story-tag", "market odds",
    "Odds text", "coverage by lean", "left 25%", "right 75%",
    "raw-js-link", "raw-button", "raw-styled-text",
]


def test_every_component_is_converted(tmp_path):
    email = _email(_write(_every_component(), tmp_path))
    for text in EVERY_TEXT:
        assert text in email.html, text
        assert text in email.text, text
    # Masthead, ticker and footer aren't part of the section.
    for gone in ("kicker-text", "sticker-text", "ticker-text", "generated"):
        assert gone not in email.html
    # Charts are dropped, each leaving a note; nothing inside the SVG leaks as text.
    assert "heatmap omitted" in email.html and "map omitted" in email.html and "chart omitted" in email.html
    assert email.html.count("heatmap omitted") == 2
    assert "marker-label-tokyo" not in email.html and "svg-text" not in email.html
    assert "leaked" not in email.html and "leaked" not in email.text


def test_block_structure(tmp_path):
    full = email_report.load(_write(_every_component(), tmp_path))
    assert full.subtitle == "subtitle-text"
    assert isinstance(full.top[0], email_report.Tiles)
    top = full.top[0].tiles[0]
    assert (top.label, top.value, top.delta, top.note, top.tone) == ("TopTile", "42", "+5", "tile-note", "hot")
    by_title = {s.title: s for s in full.sections}
    wide = by_title["Wide table"].blocks[0]
    assert wide.headers == ["Date", "Merchant", "Category", "Amount", "Link"] and len(wide.rows) == 3
    assert wide.rows[0][3].num and wide.rows[0][4].span.href == "https://example.com/r1"
    assert wide.rows[1][4].span.href is None and wide.rows[2][4].span.href is None  # file:// and #anchor dropped
    bars = by_title["Bars section"].blocks[0]
    assert [(b.name, b.amount, round(b.pct)) for b in bars.rows] == [("github.com", "31 msgs", 100),
                                                                    ("substack.com", "22 msgs", 71)]
    story = by_title["Stories section"].blocks[0]
    assert story.badge == "new" and story.title == "Story headline"
    assert [q.verified for q in story.quotes] == [True, False] and story.quotes[1].href is None
    assert story.stacked[0].segments == [("left 25%", 25, "info"), ("right 75%", 75, "hot")]
    cards = by_title["Cards section"].blocks
    assert [type(b).__name__ for b in cards] == ["CardB"] * 4 + ["Subhead", "CardB"]
    assert cards[0].title.href == "https://example.com/card?a=1&b=2" and cards[1].title.href is None


def test_only_http_links_survive(tmp_path):
    email = _email(_write(_every_component(), tmp_path))
    hrefs = re.findall(r'href="([^"]*)"', email.html)
    assert hrefs and all(h.startswith(("http://", "https://")) for h in hrefs)
    assert 'href="https://example.com/card?a=1&amp;b=2"' in email.html
    assert "https://example.com/r1" in email.text  # plain text carries the links too


def test_email_is_email_safe_and_escapes(tmp_path):
    email = _email(_write(_every_component(), tmp_path))
    for banned in BANNED:
        assert banned not in email.html, banned
    assert "&lt;script&gt;" in email.html
    assert "position:fixed" not in email.html  # no style/attribute from the report passes through
    assert HOSTILE in email.text  # the plain-text part is text, not markup


def test_mobile_layout(tmp_path):
    email = _email(_write(_every_component(), tmp_path))
    h = email.html
    assert '<meta name="color-scheme" content="light dark">' in h
    assert '<meta name="supported-color-schemes" content="light dark">' in h
    assert "max-width:640px" in h and "word-break:break-word" in h
    assert "padding:14px 16px" in h  # 16px side padding
    # Tiles two to a row: three top tiles -> two rows, the second padded with an empty half.
    assert '<td width="50%"></td>' in h
    # No font size below 12px anywhere, body text 14px+.
    assert min(int(x) for x in re.findall(r"font-size:(\d+)px", h)) >= 11
    # Every table cell block that sets a background also sets a text colour.
    for style in re.findall(r'style="([^"]*background-color:[^"]*)"', h):
        if "height:" in style or "font-size:0" in style:
            continue  # spacers and bar segments: no text
        assert "color:" in style.replace("background-color:", ""), style


def test_status_chip_is_on_its_own_line(tmp_path):
    email = _email(_write(_every_component(), tmp_path), title="inbox_brief")
    assert re.search(r'inbox_brief</div><div style="padding-top:4px;"><span[^>]*>OK</span></div>', email.html)


def test_large_report_is_not_trimmed(tmp_path):
    r = Report("Big")
    r.section("Many cards").cards([Card(f"card-{i:04d}", f"meta-{i}", "body " * 20) for i in range(1500)])
    path = _write(r, tmp_path)
    assert path.stat().st_size > 100_000
    email = _email(path)
    assert email.bytes > GMAIL_CLIP and email.clipped
    for i in range(1500):
        assert f"card-{i:04d}" in email.html and f"card-{i:04d}" in email.text


def test_unknown_markup_falls_back_to_text():
    full = email_report.parse('<main><section><h2>S</h2><foo-widget data-x="1">hello <b>bold</b> '
                              '<span>a</span><span>b</span></foo-widget></section></main>')
    assert full.sections[0].blocks == [email_report.Para("hello bold a b")]


def test_missing_or_unparseable_input_is_harmless(tmp_path):
    assert email_report.parse("").is_empty()
    assert email_report.parse("<<<not html>>> </section></div>").sections == []
    with pytest.raises(OSError):
        email_report.load(tmp_path / "nope.html")


# -- the master report's full mode --------------------------------------------------

def _run(report_path, summary=None) -> RunRow:
    return RunRow(1, "tasks.inbox_brief", (NOW - timedelta(hours=1)).isoformat(), "ok",
                  report_path=str(report_path) if report_path else None,
                  summary_json=json.dumps(summary or {"threads": 9}))


def test_full_is_the_default_mode_and_parse_config_accepts_it():
    cfg = ('[[master]]\nname = "m"\nsend_at = "06:00"\n'
           '  [[master.section]]\n  task = "inbox_brief"\n'
           '  [[master.section]]\n  task = "disasters"\n  mode = "full"\n')
    m = mr.parse_config(cfg)[0]
    assert [s.mode for s in m.sections] == ["full", "full"]
    assert "full" in mr.MODES and mr.SectionConfig("x").mode == "full"


def test_full_section_carries_the_whole_report(tmp_path):
    path = _write(_every_component(), tmp_path)
    b = mr.build_section(mr.SectionConfig("inbox_brief"), _run(path), SectionStatus(OK), lambda t: None)
    assert b.email.full is not None and b.email.note is None
    email = render_email("t", [b.email])
    assert "Hidden handled card" in email.html and "Story headline" in email.text


def test_missing_report_file_falls_back_to_the_uncut_digest_with_a_note(tmp_path):
    long_blurb = "word " * 200  # 1000 chars: digest mode would cut it to 240
    hook = lambda r, n: Digest([Tile("Needs reply", 3)],
                               [DigestItem(f"item {i}", blurb=long_blurb) for i in range(n)],
                               ["c" * 500])
    b = mr.build_section(mr.SectionConfig("inbox_brief"), _run(tmp_path / "gone.html"), SectionStatus(OK),
                         lambda t: hook)
    assert b.email.full is None and "unreadable" in b.email.note and "gone.html" in b.email.note
    assert len(b.email.digest.items) == mr.MAX_ITEMS  # full mode asks the hook for everything
    email = render_email("t", [b.email])
    assert "showing its digest" in email.html and "showing its digest" in email.text
    assert long_blurb.strip() in email.html and long_blurb.strip() in email.text
    assert "c" * 500 in email.html
    # No report_path at all: same fallback, its own note.
    b = mr.build_section(mr.SectionConfig("inbox_brief"), _run(None), SectionStatus(OK), lambda t: None)
    assert "saved no report" in b.email.note and b.email.digest.tiles[0].label == "threads"


@pytest.mark.parametrize("page", ["", "<!doctype html><html><head><style>body{", "<html><body><main class=\"wrap\"></main>"])
def test_a_report_that_converts_to_nothing_falls_back_to_the_digest(tmp_path, page):
    path = tmp_path / "empty.html"
    path.write_text(page, encoding="utf-8")
    hook = lambda r, n: Digest(items=[DigestItem("kept item")])
    b = mr.build_section(mr.SectionConfig("inbox_brief"), _run(path), SectionStatus(OK), lambda t: hook)
    assert b.email.full is None and "nothing to show" in b.email.note and "empty.html" in b.email.note
    email = render_email("t", [b.email])
    assert "kept item" in email.html and "kept item" in email.text


def test_a_converter_crash_falls_back_too(tmp_path, monkeypatch):
    path = _write(_every_component(), tmp_path)

    def boom(p):
        raise RecursionError("deep")
    monkeypatch.setattr(email_report, "load", boom)
    b = mr.build_section(mr.SectionConfig("inbox_brief"), _run(path), SectionStatus(OK), lambda t: None)
    assert b.email.full is None and "RecursionError" in b.email.note


def test_digest_mode_is_unchanged(tmp_path):
    path = _write(_every_component(), tmp_path)
    hook = lambda r, n: Digest(items=[DigestItem(f"t{i}", blurb="x" * 1000) for i in range(n + 3)])
    b = mr.build_section(mr.SectionConfig("inbox_brief", mode="digest", max_items=2), _run(path),
                         SectionStatus(OK), lambda t: hook)
    assert b.email.full is None and [i.title for i in b.email.digest.items] == ["t0", "t1"]
    assert len(b.email.digest.items[0].blurb) == 240
    assert "Story headline" not in render_email("t", [b.email]).html


def test_run_master_preview_includes_every_card_story_and_row(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    rpt = _every_component()
    path = rpt.write("inbox.html", out_dir=out)
    cfg = ('[[master]]\nname = "morning"\ntitle = "Morning Report"\nsend_at = "06:00"\n'
           '  [[master.section]]\n  task = "inbox_brief"\n')
    master = mr.parse_config(cfg)[0]
    with Store() as store:
        store.conn.execute(
            "INSERT INTO runs(task, started, finished, status, report_path, summary_json) VALUES (?,?,?,?,?,?)",
            ("tasks.inbox_brief", NOW.isoformat(), NOW.isoformat(), "ok", str(path), "{}"))
        store.conn.commit()
        result = mr.run_master(master, store.conn, send=False, now=NOW, hook_finder=lambda t: None)
    body = Path(result["preview"]).read_text(encoding="utf-8")
    for text in ("Card with link", "Hidden handled card", "Story headline", "Hetzner", "alice@example.com",
                 "github.com"):
        assert text in body
    assert "Also attached as an HTML file" in body and "only in the attachment" not in body
    assert result["bytes"] == len(body.encode("utf-8"))


def test_style_preview_email_is_clean(tmp_path, monkeypatch):
    import tasks.style_preview as style_preview

    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    style_preview.main()
    body = (tmp_path / "style-preview-email.html").read_text(encoding="utf-8")
    for banned in BANNED[:-1]:  # the style guide's raw() demo has no javascript: to begin with
        assert banned not in body, banned
    for text in ("Harbour reopens after a week of closures", "Renew domain before Oct 3", "Team lunch moved",
                 "Deals Weekly", "github.com", "map omitted", "Local model output is a draft"):
        assert text in body
