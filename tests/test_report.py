from datetime import date

import pytest

from lmsa import Card, Report, Tile, strip_think


def test_render_is_standalone_and_escaped(tmp_path):
    r = Report("<b>Inbox</b>", kicker="k", sticker="s")
    r.tiles([Tile("Unread", 1200, delta="+5")])
    s = r.section("Senders")
    s.bars([("a<script>", 3), ("b", 1)])
    s.table(["Name", "Count"], [["x", 1], ["y", 2]])
    s.cards([Card("t", body="<img onerror=x>", tags=["t1"])])
    html = r.render()

    assert "<style>" in html and "--acc" in html
    assert "<b>Inbox</b>" not in html and "&lt;b&gt;Inbox&lt;/b&gt;" in html
    assert "<script>" not in html and "<img onerror" not in html
    assert "1,200" in html
    assert '<th class="num">Count</th>' in html
    assert "width:100.0%" in html and "width:33.3%" in html

    path = r.write(out_dir=tmp_path)
    assert path.exists() and path.name.startswith("b-inbox-b-")


def test_unknown_tone_rejected():
    with pytest.raises(ValueError):
        Tile("x", 1, tone="neon").render()


def test_empty_bars_do_not_divide_by_zero():
    Report("r").section("s").bars([("zero", 0)])


def test_strip_think():
    assert strip_think("<think>\nhmm\n</think>\n\nonline") == "online"


def test_tile_sparkline_renders_svg_polyline():
    html = Tile("Unread", 42, spark=[3, 5, 1, 9]).render()
    assert '<svg class="sparkline"' in html
    assert "<polyline points=" in html


def test_tile_without_spark_omits_svg():
    html = Tile("Unread", 42).render()
    assert "sparkline" not in html


def test_tile_sparkline_needs_at_least_two_points():
    html = Tile("Unread", 42, spark=[5]).render()
    assert "sparkline" not in html


def test_card_ribbon_renders_and_escapes():
    html = Card("t", ribbon="urgent<script>").render()
    assert '<div class="ribbon">urgent&lt;script&gt;</div>' in html


def test_card_without_ribbon_omits_it():
    assert "ribbon" not in Card("t").render()


def test_card_image_renders_data_uri_thumbnail():
    html = Card("t", image="data:image/jpeg;base64,QQ==").render()
    assert '<img class="thumb" src="data:image/jpeg;base64,QQ==" alt="" loading="lazy">' in html


def test_card_without_image_omits_thumb():
    assert "thumb" not in Card("t").render()


def test_card_image_value_is_escaped():
    html = Card("t", image='data:x"><script>evil()</script>').render()
    assert "<script>" not in html
    assert "&quot;" in html or "&gt;" in html


def test_section_heatmap_renders_one_cell_per_day_in_the_week_aligned_range():
    # 2026-09-20/21/22 span a single Sun-Sat week, so the grid pads to 7 cells.
    cells = {date(2026, 9, 20): 1, date(2026, 9, 21): 5, date(2026, 9, 22): 0}
    html = Report("r").section("s").heatmap(cells).render(1)
    assert html.count("<rect") == 7
    assert html.count('class="heat-empty"') == 4  # the 4 days with no data
    assert 'class="heat-4"' in html  # max value gets top intensity
    assert "<title>2026-09-21: 5</title>" in html


def test_section_heatmap_empty_is_safe():
    html = Report("r").section("s").heatmap({}).render(1)
    assert "No data." in html
    assert "<rect" not in html


def test_section_grid_heatmap_shape_and_escaping():
    rows = ["Mon<script>", "Tue"]
    cols = ["00", "12"]
    values = [[1, 2], [3, 4]]
    html = Report("r").section("s").grid_heatmap(rows, cols, values).render(1)
    assert html.count("<rect") == 4
    assert "Mon&lt;script&gt;" in html
    assert "<script>" not in html
    assert 'class="heat-4"' in html  # value 4 is the max


def test_report_ticker_renders_and_escapes():
    html = Report("r", ticker=["a<script>", "b"]).render()
    assert '<div class="ticker">' in html
    assert html.count('<span class="ticker-item">a&lt;script&gt;</span>') == 2  # duplicated for the loop
    assert "<script>" not in html


def test_report_without_ticker_omits_it():
    assert '<div class="ticker">' not in Report("r").render()


def test_heatmaps_tolerate_nan():
    from datetime import date
    s = Report("r").section("s")
    s.heatmap({date(2026, 1, 1): float("nan"), date(2026, 1, 2): 1})
    s.grid_heatmap(["a"], ["x", "y"], [[float("nan"), 1]])
    s.render(1)


def test_card_drops_unsafe_href_schemes():
    from lmsa.report import safe_href

    for bad in ("javascript:alert(1)", " JavaScript:alert(1)", "data:text/html,x", "vbscript:x"):
        assert safe_href(bad) is None
        assert "<a " not in Card("t", href=bad).render()
    for ok in ("https://x.example/a", "http://x.example", "mailto:a@example.com", "report-20260924-1200.html", "../a/b.html", "#frag"):
        assert safe_href(ok) == ok
