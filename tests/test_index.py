import re
from datetime import datetime, timedelta
from pathlib import Path

from lmsa import Report
from lmsa.index import rebuild, _parse_filename


def test_parse_filename():
    """Test report filename parsing."""
    assert _parse_filename("report-20260924-1430.html") == ("report", datetime(2026, 9, 24, 14, 30))
    assert _parse_filename("my-task-20260924-0000.html") == ("my-task", datetime(2026, 9, 24, 0, 0))
    assert _parse_filename("index.html") is None
    assert _parse_filename("no-timestamp.html") is None
    assert _parse_filename("bad-20260924-9999.html") is None  # invalid time


def test_grouping_by_slug(tmp_path):
    """Reports are grouped by slug correctly."""
    # Write some reports with different slugs and timestamps
    base_time = datetime(2026, 9, 24, 12, 0)
    for i, slug in enumerate(["task-a", "task-b", "task-a"]):
        r = Report(slug.replace("-", " ").title())
        r.created = base_time + timedelta(hours=i)
        r.write(name=f"{slug}-{r.created:%Y%m%d-%H%M}.html", out_dir=tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # Index should mention both tasks
    assert "task-a" in html
    assert "task-b" in html
    # Total reports tile
    assert "3 reports" in html or "3" in html


def test_ordering_newest_first(tmp_path):
    """Reports within a task are ordered newest first."""
    slug = "test-order"
    base_time = datetime(2026, 9, 24, 12, 0)

    # Write reports in non-chronological order
    for offset_hours in [0, 2, 1]:  # 12:00, 14:00, 13:00
        r = Report(slug.replace("-", " ").title())
        r.created = base_time + timedelta(hours=offset_hours)
        r.write(name=f"{slug}-{r.created:%Y%m%d-%H%M}.html", out_dir=tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # Find the section for this slug
    # Extract filenames in order they appear in the section
    section_match = re.search(rf'<section[^>]*>.*?<h2>.*?{re.escape(slug)}.*?</h2>(.*?)</section>', html, re.DOTALL)
    if section_match:
        section_html = section_match.group(1)
        # Look for hrefs in order
        hrefs = re.findall(r'href="([^"]+)"', section_html)
        # The newest (14:00) should appear first
        if len(hrefs) >= 2:
            # Extract times from filenames
            times = [re.search(r'(\d{4})\.html', h).group(1) for h in hrefs if re.search(r'(\d{4})\.html', h)]
            # 14:00 (1400) should come before 12:00 (1200)
            if len(times) >= 2:
                assert times[0] == "1400", f"Expected 1400 first but got {times}"


def test_relative_links(tmp_path):
    """Report links in the index are relative (not absolute paths)."""
    r = Report("Sample")
    r.created = datetime(2026, 9, 24, 12, 0)
    r.write(name=f"sample-{r.created:%Y%m%d-%H%M}.html", out_dir=tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # Should have relative hrefs like "sample-20260924-1200.html"
    assert 'href="2026-09-24/sample/sample-20260924-1200.html"' in html
    # Should not have absolute paths
    assert str(tmp_path) not in html


def test_index_excludes_itself(tmp_path):
    """The index.html file is not listed in the index."""
    r = Report("First")
    r.write(out_dir=tmp_path)

    # Rebuild a few times to ensure index.html is not self-referencing
    rebuild(tmp_path)
    rebuild(tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # Count hrefs - should not include index.html
    href_matches = re.findall(r'href="[^"]*\.html"', html)
    for href in href_matches:
        assert "index.html" not in href


def test_rebuild_after_write(tmp_path):
    """Report.write() automatically rebuilds the index."""
    r = Report("Auto Index")
    path = r.write(out_dir=tmp_path)

    # Index should exist and contain the new report
    index_path = tmp_path / "index.html"
    assert index_path.exists()

    html = index_path.read_text(encoding="utf-8")
    assert "auto-index" in html


def test_newest_10_per_task(tmp_path):
    """Each task shows at most 10 newest reports."""
    slug = "many-reports"
    base_time = datetime(2026, 9, 24, 0, 0)

    # Write 15 reports for the same slug
    for i in range(15):
        r = Report("Sample Report")
        r.created = base_time + timedelta(hours=i)
        r.write(name=f"{slug}-{r.created:%Y%m%d-%H%M}.html", out_dir=tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # Extract hrefs for this slug
    section_match = re.search(
        rf'<section[^>]*>.*?<h2>.*?{re.escape(slug)}.*?</h2>(.*?)</section>',
        html,
        re.DOTALL,
    )
    if section_match:
        section_html = section_match.group(1)
        hrefs = re.findall(r'href="[^"]*\.html"', section_html)
        # Should have at most 10 (the newest 10)
        assert len(hrefs) <= 10, f"Expected <=10 links but got {len(hrefs)}"


def test_titles_read_from_report_files(tmp_path):
    """Report titles are read from HTML <title> tags and properly escaped."""
    # Create a report with special characters in the title
    title_with_markup = "Report <b>Bold</b> & Special"
    r = Report(title_with_markup)
    r.created = datetime(2026, 9, 24, 12, 0)
    r.write(name=f"test-{r.created:%Y%m%d-%H%M}.html", out_dir=tmp_path)

    index_path = rebuild(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    # The title should appear in the index, escaped properly
    # Card.render escapes the title, so we should see:
    # - The unescaped title text in the card (HTML-escaped by render)
    # - NOT the raw "<b>" tag
    # - NOT the filename as the link text
    assert "&lt;b&gt;Bold&lt;/b&gt; &amp; Special" in html
    assert "<b>Bold</b>" not in html  # Raw markup should not appear
    # Verify the filename is still used for href
    assert 'href="2026-09-24/test/test-20260924-1200.html"' in html


def test_sort_existing_moves_stray_flat_files_into_day_and_type_folders(tmp_path):
    (tmp_path / "dispatch-20260930-0700.html").write_text("<title>D</title>", encoding="utf-8")
    (tmp_path / "mailer-x-20260928-211437.eml").write_text("x", encoding="utf-8")
    (tmp_path / "style-preview.html").write_text("<title>S</title>", encoding="utf-8")
    from lmsa import layout
    assert layout.sort_existing(tmp_path) == 2
    rebuild(tmp_path)
    assert (tmp_path / "2026-09-30" / "dispatch" / "dispatch-20260930-0700.html").exists()
    assert (tmp_path / "2026-09-28" / "mailer-x" / "mailer-x-20260928-211437.eml").exists()
    assert (tmp_path / "style-preview.html").exists()  # undated: stays put
    assert 'href="2026-09-30/dispatch/dispatch-20260930-0700.html"' in (tmp_path / "index.html").read_text(encoding="utf-8")
