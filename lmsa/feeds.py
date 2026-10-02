"""RSS 2.0 / Atom / RDF (RSS 1.0) feed parsing, shared by every crawling task.

CONTRACT
    parse_feed(xml: str | bytes, source: str = "", limit: int | None = None) -> list[FeedItem]
        Parse one feed document. Raises ``xml.etree.ElementTree.ParseError`` on
        malformed XML (callers isolate per feed). Entries without a title are
        skipped; ``limit`` caps the number of entries *considered* (as
        ``tasks.front_page`` always did). Pass ``bytes`` when you have them so the
        XML encoding declaration is honoured.
    fetch_feed(fetcher, url, *, source="", limit=None) -> list[FeedItem]
        ``fetcher.get(url)`` (an ``lmsa.web.Fetcher``) then ``parse_feed``.
    FeedItem(title, url, published, summary, source)
        ``title`` is the element text, whitespace-trimmed (not tag-stripped);
        ``url`` is "" unless it is an absolute http(s) link (no ``javascript:``);
        ``summary`` is HTML-stripped, entity-unescaped, whitespace-collapsed
        plain text (still untrusted: escape it when rendering);
        ``published`` is a tz-aware datetime or None when missing/unparseable.
    strip_html(text) -> str, is_http_url(url) -> bool, parse_date(text) -> datetime | None
"""

from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lmsa.web import Fetcher

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class FeedItem:
    title: str
    url: str
    published: datetime | None
    summary: str
    source: str


def strip_html(text: str) -> str:
    """Strip tags, unescape entities, collapse whitespace.

    RSS/Atom descriptions are HTML -- including HTML wrapped in a CDATA block,
    which the XML parser has already unwrapped into plain text containing
    literal tags by the time this runs. Run it before truncating or embedding.
    """
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def is_http_url(url: str) -> bool:
    """True only for http(s) links -- guards against `javascript:` and other
    unsafe schemes arriving in feed/API content before it becomes an href."""
    if not url:
        return False
    try:
        return urllib.parse.urlsplit(url).scheme.lower() in ("http", "https")
    except ValueError:
        return False


def parse_date(text: str) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom, dc:date); naive times are taken as UTC."""
    text = (text or "").strip()
    if not text:
        return None
    dt = None
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        pass
    if dt is None:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _child_text(elem: ET.Element, *names: str) -> str:
    for child in elem:
        if _local(child.tag) in names:
            return (child.text or "").strip()
    return ""


def _entry_link(elem: ET.Element) -> str:
    """RSS/RDF: <link>text</link>. Atom: <link href="..." rel="alternate"/>."""
    link = ""
    for child in elem:
        if _local(child.tag) != "link":
            continue
        href = child.get("href")
        if href and (child.get("rel", "alternate") == "alternate" or not link):
            link = href
        elif child.text and not link:
            link = child.text.strip()
    if not link:  # RDF items carry their URL as rdf:about when <link> is absent
        link = next((v for k, v in elem.attrib.items() if _local(k) == "about"), "")
    return link


def parse_feed(xml: str | bytes, source: str = "", limit: int | None = None) -> list[FeedItem]:
    root = ET.fromstring(xml)
    entries = [e for e in root.iter() if _local(e.tag) in ("item", "entry")]
    if limit is not None:
        entries = entries[:limit]
    items: list[FeedItem] = []
    for entry in entries:
        title = _child_text(entry, "title")  # plain text as published; escape when rendering
        if not title:
            continue
        link = _entry_link(entry)
        items.append(FeedItem(
            title=title,
            url=link if is_http_url(link) else "",
            published=parse_date(_child_text(entry, "pubDate", "published", "updated", "date")),
            summary=strip_html(_child_text(entry, "description", "summary", "content")),
            source=source,
        ))
    return items


def fetch_feed(fetcher: "Fetcher", url: str, *, source: str = "", limit: int | None = None) -> list[FeedItem]:
    """Politely fetch (robots, delay, cache, SSRF guard) and parse one feed."""
    return parse_feed(fetcher.get(url).content, source=source, limit=limit)
