import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa.sources.gmail import Message
from lmsa.store import Store
from lmsa.web import Fetcher, TransportResponse
from tasks import bookmarks as bm
from gmail_fakes import MetaSearchMixin

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


# -- fakes ----------------------------------------------------------------------


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no live server, no network."""

    def __init__(self, replies=None, embed_model="fake-embed"):
        self.cfg = SimpleNamespace(
            embed_model=embed_model, chat_model="fake-chat",
            output_dir=Path(os.getenv("LMSA_OUTPUT_DIR", "output")),
        )
        self.calls: list[str] = []
        self.embed_calls: list[list[str]] = []
        self._replies = list(replies or [])

    def ensure_server(self) -> None:
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls.append(prompt)
        assert system is not None and "untrusted" in system
        return self._replies.pop(0)

    def embed(self, texts, *, model=None):
        self.embed_calls.append(list(texts))
        return [[float(len(t)), float(t.count("a"))] for t in texts]


def tag_reply(summary="A page about widgets.", tags=("widgets", "gadgets"), type_="article"):
    return {"summary": summary, "tags": list(tags), "type": type_}


def make_message(msg_id, subject, text="", date=NOW, from_addr="me@example.com", to=("me@example.com",)):
    return Message(
        msg_id=msg_id, thread_id=f"t-{msg_id}", labels=[], flags=[],
        from_name="Me", from_addr=from_addr, to=list(to), reply_to="",
        subject=subject, date=date, list_unsubscribe="", list_id="", auth_results="",
        text=text, attachments=[], web_url="",
    )


class FakeGmail(MetaSearchMixin):
    """Honours `after:`/`before:` in the query as whole-second epoch bounds,
    same fake shape `tests/test_packages.py` uses."""

    def __init__(self, messages=(), imap_user="me@example.com"):
        self.messages = list(messages)
        self.queries: list[str] = []
        self.cfg = SimpleNamespace(imap_user=imap_user)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, limit=200, **kwargs):
        self.queries.append(query)
        after_epoch = int(query.split("after:")[1].split()[0]) if "after:" in query else None
        matches = [
            m for m in self.messages
            if m.date is not None and (after_epoch is None or m.date.timestamp() >= after_epoch)
        ]
        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]


class FakeTransport:
    """routes: url -> TransportResponse or callable(req) -> TransportResponse.
    Unknown robots.txt -> 404 (allow all); any other unknown URL fails loudly."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls = []

    def __call__(self, req):
        self.calls.append(req)
        route = self.routes.get(req.url)
        if route is None:
            if req.url.endswith("/robots.txt"):
                return TransportResponse(404, {}, b"")
            raise AssertionError(f"unexpected fetch {req.url}")
        return route(req) if callable(route) else route


PUBLIC = {"example.com": ["93.184.216.34"], "dead.example": ["93.184.216.35"]}


def ok(body=b"<html><title>Widgets</title><body>All about widgets.</body></html>", status=200, headers=None):
    return TransportResponse(status, headers or {"Content-Type": "text/html; charset=utf-8"}, body)


def make_fetcher(transport, tmp_path, resolver=None, **kw):
    return Fetcher(
        "lmsa-bookmarks-test/1.0", transport=transport,
        resolver=resolver or (lambda host: PUBLIC.get(host, [])),
        clock=lambda: 0.0, sleep=lambda s: None, now=lambda: 0.0,
        cache_dir=tmp_path / "web", **kw,
    )


# -- Chrome/Edge Bookmarks JSON parsing (nested folders) -------------------------


def chrome_tree(*, bar_children, other_children=()):
    return {
        "roots": {
            "bookmark_bar": {"name": "Bookmarks bar", "type": "folder", "children": bar_children},
            "other": {"name": "Other bookmarks", "type": "folder", "children": list(other_children)},
        }
    }


def url_node(name, url, date_added=None):
    node = {"type": "url", "name": name, "url": url}
    if date_added is not None:
        node["date_added"] = date_added
    return node


def folder_node(name, children):
    return {"type": "folder", "name": name, "children": children}


def test_webkit_to_iso_converts_and_handles_garbage():
    assert bm.webkit_to_iso(0) is None
    assert bm.webkit_to_iso(None) is None
    assert bm.webkit_to_iso("not a number") is None
    iso = bm.webkit_to_iso(13360857000000000)
    assert iso is not None and iso.startswith("2024-")


def test_parse_bookmarks_json_flattens_nested_folders():
    tree = chrome_tree(
        bar_children=[
            url_node("Top level", "https://example.com/top"),
            folder_node("Work", [
                url_node("Widgets", "https://example.com/widgets"),
                folder_node("Sub", [url_node("Deep", "https://example.com/deep")]),
            ]),
        ],
        other_children=[url_node("Other", "https://example.com/other")],
    )
    entries = bm.parse_bookmarks_json(tree, "chrome")
    by_url = {e.url: e for e in entries}
    assert by_url["https://example.com/top"].context == "Bookmarks bar"
    assert by_url["https://example.com/widgets"].context == "Bookmarks bar/Work"
    assert by_url["https://example.com/deep"].context == "Bookmarks bar/Work/Sub"
    assert by_url["https://example.com/other"].context == "Other bookmarks"
    assert all(e.source == "chrome" for e in entries)


def test_parse_bookmarks_json_tolerates_malformed_nodes():
    tree = {"roots": {"bookmark_bar": {"type": "folder", "children": [
        {"type": "url"},  # no url -- skipped
        {"type": "folder"},  # no children -- yields nothing
        "not even a dict",  # skipped
        url_node("Fine", "https://example.com/fine"),
    ]}}}
    entries = bm.parse_bookmarks_json(tree, "chrome")
    assert [e.url for e in entries] == ["https://example.com/fine"]


def test_parse_bookmarks_json_empty_or_missing_roots():
    assert bm.parse_bookmarks_json({}, "chrome") == []
    assert bm.parse_bookmarks_json({"roots": "not a dict"}, "chrome") == []


def test_read_bookmarks_file_missing_returns_none(tmp_path):
    assert bm.read_bookmarks_file(tmp_path / "nope.json") is None


def test_read_bookmarks_file_copy_reads_without_holding_the_original_open(tmp_path):
    src = tmp_path / "Bookmarks"
    src.write_text(json.dumps(chrome_tree(bar_children=[url_node("A", "https://example.com/a")])), encoding="utf-8")
    data = bm.read_bookmarks_file(src)
    assert data["roots"]["bookmark_bar"]["children"][0]["url"] == "https://example.com/a"
    # The source file must still exist and be untouched (never opened for write).
    assert src.exists()


def test_load_browser_bookmarks_records_per_browser_errors_without_stopping(tmp_path, monkeypatch):
    good = tmp_path / "chrome.json"
    good.write_text(json.dumps(chrome_tree(bar_children=[url_node("A", "https://example.com/a")])), encoding="utf-8")
    bad = tmp_path / "edge.json"
    bad.write_text("not json", encoding="utf-8")
    config = bm.Config(chrome_path=good, edge_path=bad, fetch_limit=10, self_sent_lookback_days=30)
    entries, errors = bm.load_browser_bookmarks(config)
    assert [e.url for e in entries] == ["https://example.com/a"]
    assert len(errors) == 1 and errors[0][0] == "edge"


def test_load_config_defaults_and_explicit_opt_out(tmp_path):
    config = bm.load_config(tmp_path / "missing.toml")
    assert config.fetch_limit == bm.DEFAULT_FETCH_LIMIT

    path = tmp_path / "bookmarks.toml"
    path.write_text('edge_path = ""\nfetch_limit = 5\n', encoding="utf-8")
    config = bm.load_config(path)
    assert config.edge_path is None
    assert config.fetch_limit == 5


# -- self-sent link extraction ---------------------------------------------------


def test_extract_urls_dedupes_and_strips_trailing_punctuation():
    text = (
        "Check this out: https://example.com/a?x=1, and also "
        "(http://example.com/b). Same one again: https://example.com/a?x=1."
    )
    assert bm.extract_urls(text) == ["https://example.com/a?x=1", "http://example.com/b"]


def test_extract_urls_handles_none_and_empty():
    assert bm.extract_urls(None) == []
    assert bm.extract_urls("") == []
    assert bm.extract_urls("no links here") == []


def test_collect_self_sent_entries_extracts_links_and_marks_applied(tmp_path):
    msg = make_message("m1", "Read later", text="Interesting: https://example.com/read-later see also")
    gmail = FakeGmail([msg])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        entries, window, errors = bm.collect_self_sent_entries(
            store.conn, gmail, "me@example.com", lookback_days=30,
        )
        assert [e.url for e in entries] == ["https://example.com/read-later"]
        assert entries[0].source == "gmail"
        assert not errors
        assert bm._is_applied(store.conn, "m1")


def test_collect_self_sent_entries_message_with_no_links_still_marked_applied(tmp_path):
    msg = make_message("m1", "No links here", text="nothing to see")
    gmail = FakeGmail([msg])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        entries, _window, _errors = bm.collect_self_sent_entries(
            store.conn, gmail, "me@example.com", lookback_days=30,
        )
        assert entries == []
        assert bm._is_applied(store.conn, "m1")


def test_collect_self_sent_entries_is_incremental_second_run_rescans_nothing(tmp_path, monkeypatch):
    msg = make_message("m1", "Read later", text="https://example.com/a")
    gmail = FakeGmail([msg])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        monkeypatch.setattr(bm, "_now", lambda: NOW)
        first, _w, _e = bm.collect_self_sent_entries(store.conn, gmail, "me@example.com", lookback_days=30)
        assert len(first) == 1

        second, _w2, _e2 = bm.collect_self_sent_entries(store.conn, gmail, "me@example.com", lookback_days=30)
        assert second == []  # already-applied message is skipped by the cursor


# -- merge/dedup into the store ---------------------------------------------------


def test_upsert_seen_dedupes_same_url_across_sources_and_counts_invalid(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        entries = [
            bm.BookmarkEntry(url="https://example.com/a", title="A", source="chrome", context="Bar", added_at=None),
            bm.BookmarkEntry(url="https://example.com/a", title="A again", source="gmail", context="self-sent: hi", added_at=None),
            bm.BookmarkEntry(url="http://[::1", title="bad", source="chrome", context="", added_at=None),
        ]
        stats = bm.upsert_seen(store.conn, entries)
        assert stats.new == 1
        assert stats.invalid == 1
        row = store.conn.execute("SELECT title, status FROM bookmarks WHERE url='https://example.com/a'").fetchone()
        assert row == ("A", "new")
        sources = store.conn.execute(
            "SELECT source FROM bookmark_sources WHERE url='https://example.com/a' ORDER BY source"
        ).fetchall()
        assert [s[0] for s in sources] == ["chrome", "gmail"]


def test_upsert_seen_second_sighting_updates_last_seen_not_status(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        e = bm.BookmarkEntry(url="https://example.com/a", title="A", source="chrome", context="Bar", added_at=None)
        bm.upsert_seen(store.conn, [e])
        store.conn.execute("UPDATE bookmarks SET status='ok', checked_at='x' WHERE url='https://example.com/a'")
        store.conn.commit()
        stats = bm.upsert_seen(store.conn, [e])
        assert stats.new == 0 and stats.seen_again == 1
        status = store.conn.execute("SELECT status FROM bookmarks WHERE url='https://example.com/a'").fetchone()[0]
        assert status == "ok"  # a re-sighting never resets an already-checked bookmark


# -- dead link classification / detection with a fake fetcher --------------------


def test_classify_fetch_error_dead_vs_unchecked_vs_skipped(tmp_path):
    from lmsa.web import BlockedURL, HTTPStatusError, RobotsDenied, TooLarge

    assert bm.classify_fetch_error(HTTPStatusError(404, "u"))[0] == "dead"
    assert bm.classify_fetch_error(HTTPStatusError(410, "u"))[0] == "dead"
    assert bm.classify_fetch_error(HTTPStatusError(500, "u"))[0] == "unchecked"
    assert bm.classify_fetch_error(HTTPStatusError(403, "u"))[0] == "unchecked"
    assert bm.classify_fetch_error(RobotsDenied("no"))[0] == "unchecked"
    assert bm.classify_fetch_error(TooLarge("big"))[0] == "unchecked"
    assert bm.classify_fetch_error(BlockedURL("cannot resolve host.example"))[0] == "dead"
    assert bm.classify_fetch_error(BlockedURL("scheme 'javascript' not allowed"))[0] == "skipped"


def test_process_url_dead_link_404_marks_row_dead_and_never_calls_the_model(tmp_path):
    t = FakeTransport({"https://dead.example/gone": ok(status=404, body=b"nope")})
    fetcher = make_fetcher(t, tmp_path)
    lm = FakeLMStudio(replies=[])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen) VALUES (?, '', 'new', 'x', 'x')",
            ("https://dead.example/gone",),
        )
        result = bm.process_url(lm, fetcher, store.conn, "https://dead.example/gone")
        assert result.status == "dead"
        assert not lm.calls
        row = store.conn.execute(
            "SELECT status, http_status FROM bookmarks WHERE url=?", ("https://dead.example/gone",)
        ).fetchone()
        assert row == ("dead", 404)


def test_process_url_dns_failure_is_dead_not_skipped(tmp_path):
    t = FakeTransport({})
    fetcher = make_fetcher(t, tmp_path, resolver=lambda host: [])
    lm = FakeLMStudio(replies=[])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen) VALUES (?, '', 'new', 'x', 'x')",
            ("https://nowhere.example/x",),
        )
        result = bm.process_url(lm, fetcher, store.conn, "https://nowhere.example/x")
        assert result.status == "dead"


def test_process_url_5xx_is_unchecked_not_dead(tmp_path):
    t = FakeTransport({"https://example.com/oops": ok(status=503, body=b"try later")})
    fetcher = make_fetcher(t, tmp_path)
    lm = FakeLMStudio(replies=[])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        result = bm.process_url(lm, fetcher, store.conn, "https://example.com/oops")
        assert result.status == "unchecked"


def test_process_url_javascript_scheme_is_skipped(tmp_path):
    fetcher = make_fetcher(FakeTransport({}), tmp_path)
    lm = FakeLMStudio(replies=[])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        result = bm.process_url(lm, fetcher, store.conn, "javascript:alert(1)")
        assert result.status == "skipped"
        assert not lm.calls


@pytest.mark.slow
def test_process_url_success_tags_and_embeds(tmp_path):
    t = FakeTransport({"https://example.com/widgets": ok()})
    fetcher = make_fetcher(t, tmp_path)
    lm = FakeLMStudio(replies=[tag_reply()])
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen) VALUES (?, '', 'new', 'x', 'x')",
            ("https://example.com/widgets",),
        )
        result = bm.process_url(lm, fetcher, store.conn, "https://example.com/widgets")
        assert result.status == "ok"
        assert result.tags == ["gadgets", "widgets"]
        assert lm.embed_calls  # embedded into the vectors table
        row = store.conn.execute(
            "SELECT status, summary, type FROM bookmarks WHERE url=?", ("https://example.com/widgets",)
        ).fetchone()
        assert row == ("ok", "A page about widgets.", "article")
        vec = store.conn.execute(
            "SELECT item_id FROM embeddings WHERE ns=? AND item_id=?", (bm.NS, "https://example.com/widgets"),
        ).fetchone()
        assert vec is not None


def test_process_url_model_failure_after_successful_fetch_is_isolated(tmp_path):
    t = FakeTransport({"https://example.com/widgets": ok()})
    fetcher = make_fetcher(t, tmp_path)

    class BrokenLM(FakeLMStudio):
        def extract(self, *a, **kw):
            raise RuntimeError("model exploded")

    lm = BrokenLM()
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen) VALUES (?, '', 'new', 'x', 'x')",
            ("https://example.com/widgets",),
        )
        result = bm.process_url(lm, fetcher, store.conn, "https://example.com/widgets")
        assert result.status == "error"
        row = store.conn.execute("SELECT status FROM bookmarks WHERE url=?", ("https://example.com/widgets",)).fetchone()
        assert row == ("error",)


# -- pending / incremental fetch ---------------------------------------------------


def test_pending_urls_only_never_checked_oldest_first(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.executemany(
            "INSERT INTO bookmarks(url, status, first_seen, last_seen, checked_at) VALUES (?, 'new', ?, ?, ?)",
            [
                ("https://example.com/1", "2026-01-01", "2026-01-01", None),
                ("https://example.com/2", "2026-01-02", "2026-01-02", "2026-01-03"),  # already checked
                ("https://example.com/3", "2026-01-03", "2026-01-03", None),
            ],
        )
        store.conn.commit()
        assert bm.pending_urls(store.conn, 10) == ["https://example.com/1", "https://example.com/3"]
        assert bm.pending_urls(store.conn, 1) == ["https://example.com/1"]
        assert bm.count_pending(store.conn) == 2


def test_process_url_of_a_url_already_checked_is_not_reattempted_by_the_run_loop(tmp_path):
    # `pending_urls` excludes anything with checked_at set -- a second run's
    # candidate list simply won't include an already-fetched URL.
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, status, first_seen, last_seen, checked_at) "
            "VALUES ('https://example.com/done', 'ok', 'x', 'x', 'x')"
        )
        store.conn.commit()
        assert bm.pending_urls(store.conn, 10) == []


# -- search matches ----------------------------------------------------------------


def test_build_search_matches_reads_metadata_by_url(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen, summary, tags_json, type) "
            "VALUES ('https://example.com/a', 'A', 'ok', 'x', 'x', 'about a', '[\"x\"]', 'article')"
        )
        store.conn.commit()
        matches = bm.build_search_matches(store.conn, [("https://example.com/a", 0.9)])
        assert matches[0].title == "A"
        assert matches[0].tags == ["x"]
        assert matches[0].score == 0.9


# -- end to end via main() ---------------------------------------------------------


def test_main_reads_browser_bookmarks_and_self_sent_links_and_tags_new_urls(monkeypatch, tmp_path):
    chrome = tmp_path / "chrome.json"
    chrome.write_text(json.dumps(chrome_tree(bar_children=[url_node("Widgets", "https://example.com/widgets")])), encoding="utf-8")
    config_path = tmp_path / "bookmarks.toml"
    config_path.write_text(
        f'chrome_path = "{chrome.as_posix()}"\nedge_path = ""\nfetch_limit = 10\n', encoding="utf-8",
    )

    msg = make_message("m1", "Read later", text="https://example.com/widgets and https://example.com/read")
    gmail = FakeGmail([msg])
    transport = FakeTransport({
        "https://example.com/widgets": ok(),
        "https://example.com/read": ok(body=b"<title>Reading</title><body>stuff</body>"),
    })

    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(bm, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(bm, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(bm, "LMStudio", lambda: FakeLMStudio(replies=[tag_reply(), tag_reply(tags=("reading",))]))
    monkeypatch.setattr(bm, "Fetcher", lambda user_agent: make_fetcher(transport, tmp_path))
    monkeypatch.setattr(bm.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(limit=None, query=None, k=8))

    bm.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        rows = store.conn.execute("SELECT url, status FROM bookmarks ORDER BY url").fetchall()
        assert rows == [
            ("https://example.com/read", "ok"),
            ("https://example.com/widgets", "ok"),
        ]
        run = store.conn.execute("SELECT status, summary_json FROM runs WHERE task='bookmarks'").fetchone()
        assert run[0] == "ok"
        summary = json.loads(run[1])
        assert summary["tagged"] == 2
        assert summary["new_bookmarks"] == 2  # deduped: widgets seen from both chrome and gmail


def test_main_second_run_does_not_refetch_or_rescan(monkeypatch, tmp_path):
    config_path = tmp_path / "bookmarks.toml"
    config_path.write_text('chrome_path = ""\nedge_path = ""\nfetch_limit = 10\n', encoding="utf-8")

    msg = make_message("m1", "Read later", text="https://example.com/widgets")
    gmail = FakeGmail([msg])
    transport = FakeTransport({"https://example.com/widgets": ok()})

    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(bm, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(bm, "Gmail", lambda cfg=None: gmail)
    monkeypatch.setattr(bm, "LMStudio", lambda: FakeLMStudio(replies=[tag_reply()]))
    monkeypatch.setattr(bm, "Fetcher", lambda user_agent: make_fetcher(transport, tmp_path))
    monkeypatch.setattr(bm.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(limit=None, query=None, k=8))
    bm.main()

    # Second run: no new browser bookmarks, the Gmail message is already
    # applied, and the URL is already checked -- nothing should be fetched
    # or sent to the model again.
    monkeypatch.setattr(bm, "LMStudio", lambda: FakeLMStudio(replies=[]))
    monkeypatch.setattr(bm, "Fetcher", lambda user_agent: make_fetcher(FakeTransport({}), tmp_path))
    bm.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        run = store.conn.execute(
            "SELECT summary_json FROM runs WHERE task='bookmarks' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        summary = json.loads(run[0])
        assert summary["fetched"] == 0
        assert summary["new_bookmarks"] == 0


def test_main_query_mode_skips_the_scan_and_searches_instead(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    with Store(tmp_path / "lmsa.sqlite") as store:
        bm.ensure_schema(store.conn)
        store.conn.execute(
            "INSERT INTO bookmarks(url, title, status, first_seen, last_seen, summary, tags_json, type) "
            "VALUES ('https://example.com/a', 'A', 'ok', 'x', 'x', 'about a', '[]', 'article')"
        )
        from lmsa.vectors import embed_and_upsert
        embed_and_upsert(store.conn, FakeLMStudio(), ns=bm.NS, model="fake-embed", items=[("https://example.com/a", "about a")])

    monkeypatch.setattr(bm, "LMStudio", lambda: FakeLMStudio())
    monkeypatch.setattr(bm.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(limit=None, query="widgets", k=8))
    monkeypatch.setattr(bm, "Gmail", lambda cfg=None: (_ for _ in ()).throw(AssertionError("Gmail must not be touched in --query mode")))

    bm.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        run = store.conn.execute("SELECT summary_json FROM runs WHERE task='bookmarks'").fetchone()
        summary = json.loads(run[0])
        assert summary["mode"] == "query"
        assert summary["matches"] == 1


def test_main_gmail_connect_failure_still_reports_browser_bookmarks(monkeypatch, tmp_path):
    """A network/TLS error while connecting to Gmail must not abort the run:
    browser bookmarks are still processed and the error is reported."""
    chrome = tmp_path / "chrome.json"
    chrome.write_text(json.dumps(chrome_tree(bar_children=[url_node("Widgets", "https://example.com/widgets")])), encoding="utf-8")
    config_path = tmp_path / "bookmarks.toml"
    config_path.write_text(f'chrome_path = "{chrome.as_posix()}"\nedge_path = ""\nfetch_limit = 10\n', encoding="utf-8")

    class Unreachable:
        def __enter__(self):
            raise OSError("getaddrinfo failed")

        def __exit__(self, *exc):
            return False

    transport = FakeTransport({"https://example.com/widgets": ok()})
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr(bm, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(bm, "Gmail", lambda cfg=None: Unreachable())
    monkeypatch.setattr(bm, "LMStudio", lambda: FakeLMStudio(replies=[tag_reply()]))
    monkeypatch.setattr(bm, "Fetcher", lambda user_agent: make_fetcher(transport, tmp_path))
    monkeypatch.setattr(bm.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(limit=None, query=None, k=8))

    bm.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        rows = store.conn.execute("SELECT url, status FROM bookmarks").fetchall()
        assert rows == [("https://example.com/widgets", "ok")]
        run = store.conn.execute("SELECT status FROM runs WHERE task='bookmarks'").fetchone()
        assert run[0] == "ok"
