from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import xml.etree.ElementTree as ET

from lmsa import feeds
from lmsa.web import Response

FIX = Path(__file__).parent / "fixtures" / "feeds"


def _read(name: str) -> bytes:
    return (FIX / name).read_bytes()


def test_rss_items_dates_and_html_summaries():
    items = feeds.parse_feed(_read("rss.xml"), source="Harbour")
    assert [i.title for i in items] == ["Port reopens after storm", "Undated notice", "Garbled date"]
    first = items[0]
    assert first.url == "https://news.example.org/port-reopens"
    assert first.summary == "The harbour reopened & ships resumed."
    assert first.published == datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)
    assert first.source == "Harbour"
    assert items[1].published is None  # missing date
    assert items[2].published is None  # unparseable date
    assert items[2].summary == ""


def test_atom_prefers_alternate_link_and_keeps_offset():
    items = feeds.parse_feed(_read("atom.xml"), source="Journal")
    assert items[0].url == "https://journal.example.org/posts/1"
    assert items[0].summary == "Summary with markup ."
    assert items[0].published == datetime(2026, 9, 23, 10, 30, tzinfo=timezone(timedelta(hours=2)))
    assert items[1].url == "https://journal.example.org/posts/2"
    assert items[1].summary == "Content only"
    assert items[1].published is None


def test_rdf_items_use_dc_date_and_rdf_about():
    items = feeds.parse_feed(_read("rdf.xml"), source="Bulletin")
    assert [i.title for i in items] == ["RDF item A", "RDF item B", "RDF item C"]
    assert items[0].published == datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
    assert items[0].summary == "Plain text summary."
    assert items[1].url == "https://bulletin.example.org/b"  # from rdf:about
    assert items[1].published is None
    assert items[2].url == ""  # unsafe scheme dropped


def test_limit_and_str_input():
    items = feeds.parse_feed(_read("rss.xml").decode("utf-8").split("\n", 1)[1], limit=1)
    assert len(items) == 1 and items[0].source == ""


def test_malformed_xml_raises_parse_error():
    with pytest.raises(ET.ParseError):
        feeds.parse_feed("<rss><channel><item>")


def test_fetch_feed_uses_fetcher():
    class FakeFetcher:
        def get(self, url):
            assert url == "https://news.example.org/rss"
            return Response(200, url, {}, _read("rss.xml"))

    items = feeds.fetch_feed(FakeFetcher(), "https://news.example.org/rss", source="H", limit=2)
    assert [i.source for i in items] == ["H", "H"]
