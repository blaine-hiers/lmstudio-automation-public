from pathlib import Path
import math
from datetime import datetime, timedelta, timezone

import pytest

from tasks import front_page as fp

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Example Blog</title>
<item>
  <title>RSS Item One</title>
  <link>https://example.com/one</link>
  <description>A description of item one.</description>
  <pubDate>Thu, 24 Sep 2026 06:00:00 GMT</pubDate>
</item>
<item>
  <title>RSS Item Two</title>
  <link>https://example.com/two</link>
  <description>A description of item two.</description>
  <pubDate>Wed, 23 Sep 2026 06:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

ATOM_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Example Atom Feed</title>
<entry>
  <title>Atom Entry One</title>
  <link href="https://example.com/atom-one" rel="alternate"/>
  <summary>An atom summary.</summary>
  <updated>2026-09-24T05:00:00Z</updated>
</entry>
</feed>
"""

RSS_CDATA_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Example Blog</title>
<item>
  <title>CDATA Item</title>
  <link>https://example.com/cdata</link>
  <description><![CDATA[<p>Some <a href="https://example.com">linked</a> text &amp; more.  Extra   spaces.</p>]]></description>
  <pubDate>Thu, 24 Sep 2026 06:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

RSS_UNSAFE_LINK_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Unsafe Link Item</title>
  <link>javascript:alert(1)</link>
  <description>no link should survive</description>
</item>
</channel></rss>
"""


# ---- Hacker News ------------------------------------------------------------

def test_fetch_hn_skips_bad_id_and_non_stories(monkeypatch):
    def fake_get_json(url):
        if url.endswith("/topstories.json"):
            return [1, 2, 3, 4]
        if url.endswith("/item/1.json"):
            return {"id": 1, "type": "story", "title": "Story One", "time": 1758700800, "score": 42}
        if url.endswith("/item/2.json"):
            raise RuntimeError("boom")
        if url.endswith("/item/3.json"):
            return {"id": 3, "type": "job", "title": "A job posting"}
        if url.endswith("/item/4.json"):
            return {"id": 4, "type": "story", "dead": True, "title": "Dead story"}
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, errors = fp.fetch_hn("topstories", 10)

    assert [i.title for i in items] == ["Story One"]
    assert items[0].tag == "HN"
    assert items[0].popularity == 42
    assert errors == ["HN item 2: boom"]


def test_fetch_hn_list_failure_returns_no_items(monkeypatch):
    def fake_get_json(url):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, errors = fp.fetch_hn("topstories", 10)
    assert items == []
    assert "rate limited" in errors[0]


def test_fetch_hn_falls_back_to_discussion_link(monkeypatch):
    def fake_get_json(url):
        if url.endswith("/topstories.json"):
            return [5]
        return {"id": 5, "type": "story", "title": "Ask HN: text post"}

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, _ = fp.fetch_hn("topstories", 10)
    assert items[0].url == "https://news.ycombinator.com/item?id=5"


def test_fetch_hn_falls_back_to_discussion_link_for_unsafe_url_scheme(monkeypatch):
    def fake_get_json(url):
        if url.endswith("/topstories.json"):
            return [6]
        return {"id": 6, "type": "story", "title": "Sketchy url", "url": "javascript:alert(1)"}

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, _ = fp.fetch_hn("topstories", 10)
    assert items[0].url == "https://news.ycombinator.com/item?id=6"


# ---- Reddit -------------------------------------------------------------

def test_fetch_reddit_skips_stickied_and_isolates_a_bad_sub(monkeypatch):
    def fake_get_json(url):
        if "/r/alpha/" in url:
            return {
                "data": {
                    "children": [
                        {"data": {"title": "Stickied", "stickied": True, "permalink": "/x", "score": 1}},
                        {
                            "data": {
                                "title": "Real post",
                                "stickied": False,
                                "permalink": "/r/alpha/comments/1/real-post/",
                                "score": 99,
                                "selftext": "body text",
                                "created_utc": 1758700800,
                            }
                        },
                    ]
                }
            }
        if "/r/beta/" in url:
            raise RuntimeError("403")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, errors = fp.fetch_reddit([{"name": "alpha"}, {"name": "beta"}], 25)

    assert [i.title for i in items] == ["Real post"]
    assert items[0].url == "https://www.reddit.com/r/alpha/comments/1/real-post/"
    assert items[0].popularity == 99
    assert items[0].tag == "r/alpha"
    assert errors == ["r/beta: 403"]


def test_fetch_reddit_drops_unsafe_url_when_no_permalink(monkeypatch):
    def fake_get_json(url):
        return {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "No permalink, unsafe url",
                            "stickied": False,
                            "permalink": "",
                            "url": "javascript:alert(1)",
                            "score": 5,
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    items, _ = fp.fetch_reddit([{"name": "alpha"}], 25)
    assert items[0].url == ""


# ---- RSS / Atom -----------------------------------------------------------

def test_parse_feed_rss():
    items = fp.parse_feed(RSS_XML, "Example Blog", limit=10)
    assert [i.title for i in items] == ["RSS Item One", "RSS Item Two"]
    assert items[0].url == "https://example.com/one"
    assert items[0].snippet == "A description of item one."
    assert items[0].published == datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)
    assert items[0].tag == "Example Blog"
    assert items[0].popularity is None


def test_parse_feed_atom():
    items = fp.parse_feed(ATOM_XML, "Example Atom Feed", limit=10)
    assert len(items) == 1
    assert items[0].title == "Atom Entry One"
    assert items[0].url == "https://example.com/atom-one"
    assert items[0].snippet == "An atom summary."
    assert items[0].published == datetime(2026, 9, 24, 5, 0, tzinfo=timezone.utc)


def test_parse_feed_respects_limit():
    items = fp.parse_feed(RSS_XML, "Example Blog", limit=1)
    assert len(items) == 1


def test_fetch_feeds_isolates_a_bad_feed(monkeypatch):
    def fake_get_text(url):
        if url == "https://good/feed":
            return RSS_XML
        raise RuntimeError("timed out")

    monkeypatch.setattr(fp, "_get_text", fake_get_text)
    items, errors = fp.fetch_feeds(
        [{"name": "Good", "url": "https://good/feed"}, {"name": "Bad", "url": "https://bad/feed"}],
        limit=10,
    )
    assert len(items) == 2
    assert errors == ["Bad: timed out"]


def test_parse_feed_strips_html_from_cdata_description():
    items = fp.parse_feed(RSS_CDATA_XML, "Example Blog", limit=10)
    assert len(items) == 1
    # tags gone, entity unescaped, whitespace collapsed -- and this is what
    # feeds item_text() and the card body, so it never sees raw markup.
    assert items[0].snippet == "Some linked text & more. Extra spaces."


def test_parse_feed_drops_unsafe_link_scheme():
    items = fp.parse_feed(RSS_UNSAFE_LINK_XML, "Example Blog", limit=10)
    assert items[0].url == ""


def test_strip_html_tags_entities_and_whitespace():
    assert fp.strip_html('<p>Hello <a href="x">world</a>!</p>') == "Hello world !"
    assert fp.strip_html("plain &amp; simple") == "plain & simple"
    assert fp.strip_html("a\n\n  b\tc") == "a b c"
    assert fp.strip_html("") == ""
    assert fp.strip_html(None) == ""


# ---- URL safety and normalization -------------------------------------------

def test_is_http_url():
    assert fp._is_http_url("https://example.com") is True
    assert fp._is_http_url("http://example.com") is True
    assert fp._is_http_url("javascript:alert(1)") is False
    assert fp._is_http_url("data:text/html,<script>1</script>") is False
    assert fp._is_http_url("") is False
    assert fp._is_http_url(None) is False


def test_normalize_url_lowercases_and_strips_tracking():
    a = fp.normalize_url("HTTPS://Example.COM/Story/?utm_source=hn&utm_medium=rss&id=5#frag")
    b = fp.normalize_url("https://example.com/Story?id=5")
    assert a == b == "https://example.com/Story?id=5"


def test_normalize_url_non_absolute_returns_empty():
    assert fp.normalize_url("") == ""
    assert fp.normalize_url("javascript:alert(1)") == ""
    assert fp.normalize_url("/relative/path") == ""


# ---- Dedupe -------------------------------------------------------------

def test_dedupe_merges_same_story_across_sources():
    hn = fp.Item(source="hn", tag="HN", title="Same story", url="https://example.com/x?utm_source=hn",
                 popularity=50)
    rss = fp.Item(source="rss", tag="Some Feed", title="Same story", url="https://example.com/x/",
                  snippet="from the feed")
    items = fp.dedupe([hn, rss])

    assert len(items) == 1
    kept = items[0]
    assert kept.tag == "HN · RSS"
    assert kept.popularity == 50  # kept item's own popularity wins
    assert kept.snippet == "from the feed"  # empty on the kept item, filled from the merged one


def test_dedupe_leaves_distinct_urls_alone():
    a = fp.Item(source="hn", tag="HN", title="A", url="https://example.com/a")
    b = fp.Item(source="reddit", tag="r/x", title="B", url="https://example.com/b")
    items = fp.dedupe([a, b])
    assert items == [a, b]
    assert items[0].tag == "HN"
    assert items[1].tag == "r/x"


def test_dedupe_never_merges_items_without_a_usable_url():
    a = fp.Item(source="rss", tag="Feed A", title="No link", url="")
    b = fp.Item(source="rss", tag="Feed B", title="No link", url="javascript:x")
    items = fp.dedupe([a, b])
    assert len(items) == 2


def test_gather_dedupes_across_hn_and_rss(monkeypatch):
    def fake_get_json(url):
        if url.endswith("/topstories.json"):
            return [1]
        return {
            "id": 1, "type": "story", "title": "Cross-posted story",
            "url": "https://example.com/story", "score": 77, "time": 1758700800,
        }

    def fake_get_text(url):
        return """<?xml version="1.0"?>
<rss version="2.0"><channel><item>
<title>Cross-posted story (mirror)</title>
<link>https://example.com/story/</link>
<description>mirrored</description>
</item></channel></rss>"""

    monkeypatch.setattr(fp, "_get_json", fake_get_json)
    monkeypatch.setattr(fp, "_get_text", fake_get_text)

    cfg = {
        "fetch": {"hn_list": "topstories", "hn_limit": 5, "subreddit_limit": 5, "feed_limit": 5},
        "source": {"feeds": [{"name": "Mirror", "url": "https://mirror/feed"}]},
    }
    items, errors = fp.gather(cfg)

    assert len(items) == 1
    assert items[0].tag == "HN · RSS"
    assert errors == []


# ---- Ranking math -----------------------------------------------------------

def test_cosine_similarity():
    assert fp.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert fp.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert fp.cosine([0, 0], [1, 1]) == 0.0


def test_recency_decay():
    assert fp.recency_decay(None, NOW, 36) == 1.0
    assert fp.recency_decay(NOW, NOW, 36) == pytest.approx(1.0)
    assert fp.recency_decay(NOW - timedelta(hours=36), NOW, 36) == pytest.approx(0.5)
    # future-dated item (clock skew) is not penalized beyond "fresh"
    assert fp.recency_decay(NOW + timedelta(hours=5), NOW, 36) == pytest.approx(1.0)
    assert fp.recency_decay(NOW - timedelta(hours=10), NOW, 0) == 1.0


def test_normalize():
    assert fp.normalize([0, 5, 10]) == [0.0, 0.5, 1.0]
    assert fp.normalize([3, 3, 3]) == [0.0, 0.0, 0.0]
    assert fp.normalize([]) == []
    assert fp.normalize([float("nan"), 1, 2]) == [0.0, 0.0, 1.0]


def _item(title="t", popularity=None, published=NOW, snippet=""):
    return fp.Item(source="hn", tag="HN", title=title, url="https://x", snippet=snippet,
                    published=published, popularity=popularity)


def test_rank_assigns_best_matching_interest_and_decays_score():
    interests = [{"name": "a", "description": "a"}, {"name": "b", "description": "b"}]
    interest_vectors = [[1, 0], [0, 1]]
    items = [_item("matches a", published=NOW), _item("matches b, stale", published=NOW - timedelta(hours=100))]
    item_vectors = [[1, 0], [0, 1]]

    fp.rank(items, interests, interest_vectors, item_vectors, NOW, half_life_hours=50)

    assert items[0].interest == "a"
    assert items[0].score == pytest.approx(1.0)
    assert items[1].interest == "b"
    assert items[1].score < 1.0  # decayed for staleness


def test_wildcard_scores_treat_missing_popularity_as_zero_signal():
    items = [_item("hn hot", popularity=100), _item("rss no signal", popularity=None)]
    scores = fp.wildcard_scores(items, NOW, half_life_hours=36)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0


# ---- main() end to end -----------------------------------------------------

class FakeLMStudio:
    def __init__(self):
        self.embed_calls: list[list[str]] = []

    def ensure_server(self):
        pass

    def embed(self, texts, *, model=None):
        self.embed_calls.append(texts)
        # deterministic 2-d vectors: interest/item text containing "alpha"
        # points along x, everything else along y.
        return [[1.0, 0.0] if "alpha" in t.lower() else [0.0, 1.0] for t in texts]


@pytest.fixture
def tmp_config(tmp_path):
    cfg_path = tmp_path / "front_page.toml"
    cfg_path.write_text(
        """
[fetch]
hn_list = "topstories"
hn_limit = 5
subreddit_limit = 5
feed_limit = 5
top_n = 2
wildcard_n = 1
recency_half_life_hours = 36
snippet_chars = 4000

[[interests]]
name = "alpha-topic"
description = "alpha stuff"

[[source.subreddits]]
name = "alpha"

[[source.feeds]]
name = "Example Blog"
url = "https://good/feed"
""",
        encoding="utf-8",
    )
    return cfg_path


def _fake_get_json(url):
    if url.endswith("/topstories.json"):
        return [1]
    if url.endswith("/item/1.json"):
        return {"id": 1, "type": "story", "title": "alpha on HN", "time": 1758700800, "score": 50}
    if "/r/alpha/" in url:
        return {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "beta post on reddit",
                            "stickied": False,
                            "permalink": "/r/alpha/comments/2/",
                            "score": 10,
                            "created_utc": 1758700800,
                        }
                    }
                ]
            }
        }
    raise AssertionError(f"unexpected url {url}")


def test_main_writes_report(monkeypatch, tmp_path, tmp_config, capsys):
    monkeypatch.setattr(fp, "CONFIG_PATH", tmp_config)
    monkeypatch.setattr(fp, "LMStudio", FakeLMStudio)
    monkeypatch.setattr(fp, "_get_json", _fake_get_json)
    monkeypatch.setattr(fp, "_get_text", lambda url: RSS_XML)
    monkeypatch.setattr(fp, "_now", lambda: NOW)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    fp.main()

    out = capsys.readouterr().out.strip()
    path = Path(out)
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "Front page" in html
    assert "alpha on HN" in html  # top pick: matches the alpha interest
    # candidates: 1 HN + 1 reddit + 2 rss = 4; top_n=2 so 2 are left over,
    # wildcard_n=1 keeps the single highest-popularity leftover
    assert "4" in html  # candidate count tile


def test_main_requires_at_least_one_interest(monkeypatch, tmp_path):
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text("[fetch]\ntop_n = 5\n", encoding="utf-8")
    monkeypatch.setattr(fp, "CONFIG_PATH", cfg_path)
    with pytest.raises(RuntimeError, match="interests"):
        fp.main()
