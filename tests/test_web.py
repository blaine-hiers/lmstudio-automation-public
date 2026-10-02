"""lmsa.web with a fake transport, resolver, clock and sleep -- no network."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lmsa import web
from lmsa.web import (
    BlockedURL, FetchError, Fetcher, HTTPStatusError, RobotsDenied, TooLarge,
    TransportResponse, extract_article, urllib_transport,
)

UA = "lmsa-test/1.0"
FIX = Path(__file__).parent / "fixtures" / "web"
DAY1 = datetime(2026, 9, 24, 12, tzinfo=timezone.utc).timestamp()

PUBLIC = {
    "example.com": ["93.184.216.34"],
    "other.example": ["93.184.216.35"],
    "internal.example": ["10.0.0.5"],
    "mixed.example": ["93.184.216.36", "127.0.0.1"],
    "mapped.example": ["::ffff:127.0.0.1"],
    "metadata.example": ["169.254.169.254"],
}


class FakeTransport:
    """routes: url -> TransportResponse, or callable(req) -> TransportResponse.
    Unknown robots.txt -> 404 (allow all); any other unknown URL fails the test."""

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

    def urls(self):
        return [c.url for c in self.calls]


class FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(round(seconds, 6))
        self.t += seconds


def ok(body=b"hello", headers=None, status=200):
    return TransportResponse(status, headers or {"Content-Type": "text/plain; charset=utf-8"}, body)


def make(transport, tmp_path, **kw):
    ft = kw.pop("ft", None) or FakeTime()
    now = kw.pop("now", lambda: DAY1)
    kw.setdefault("cache_dir", tmp_path / "web")
    return Fetcher(
        UA, transport=transport,
        resolver=lambda host: PUBLIC.get(host, []),
        clock=ft.clock, sleep=ft.sleep, now=now, **kw,
    ), ft


def test_get_returns_response_and_sends_user_agent(tmp_path):
    t = FakeTransport({"https://example.com/a": ok(b"caf\xc3\xa9")})
    f, _ = make(t, tmp_path)
    resp = f.get("https://example.com/a")
    assert (resp.status, resp.url, resp.text, resp.from_cache) == (200, "https://example.com/a", "café", False)
    assert resp.ok
    page = t.calls[-1]
    assert page.headers["User-Agent"] == UA and page.ip == "93.184.216.34"
    assert t.urls() == ["https://example.com/robots.txt", "https://example.com/a"]


def test_get_timeout_override_reaches_the_transport_request(tmp_path):
    """#89: a caller (a bounded readiness probe) can ask for a shorter
    per-call timeout than the Fetcher's own; omitting it keeps the
    constructor's value."""
    t = FakeTransport({"https://example.com/a": ok()})
    f, _ = make(t, tmp_path, timeout=20)
    f.get("https://example.com/a", timeout=5)
    first = next(c for c in t.calls if c.url == "https://example.com/a")
    assert first.timeout == 5
    f.get("https://example.com/a")  # no override this time -> Fetcher's own timeout
    second = t.calls[-1]
    assert second.timeout == 20


def test_robots_disallow_raises_and_never_fetches(tmp_path):
    robots = b"User-agent: *\nDisallow: /private/\n"
    t = FakeTransport({"https://example.com/robots.txt": ok(robots)})
    f, _ = make(t, tmp_path)
    with pytest.raises(RobotsDenied):
        f.get("https://example.com/private/page")
    assert "https://example.com/private/page" not in t.urls()


def test_robots_is_cached_per_host_per_day_on_disk(tmp_path):
    robots = b"User-agent: *\nDisallow: /private/\n"
    day = [DAY1]
    t = FakeTransport({"https://example.com/robots.txt": ok(robots), "https://example.com/a": ok()})
    f, _ = make(t, tmp_path, now=lambda: day[0])
    f.get("https://example.com/a")
    f.get("https://example.com/a")
    assert t.urls().count("https://example.com/robots.txt") == 1

    f2, _ = make(t, tmp_path, now=lambda: day[0])  # new process, same day -> disk copy
    f2.get("https://example.com/a")
    assert t.urls().count("https://example.com/robots.txt") == 1

    day[0] = DAY1 + 86400  # next day -> refetched
    f2.get("https://example.com/a")
    assert t.urls().count("https://example.com/robots.txt") == 2


def test_robots_forbidden_or_unreachable_blocks_host(tmp_path):
    t = FakeTransport({"https://example.com/robots.txt": ok(b"", status=403)})
    f, _ = make(t, tmp_path)
    with pytest.raises(RobotsDenied):
        f.get("https://example.com/a")

    def boom(req):
        raise OSError("connection reset")

    t2 = FakeTransport({"https://other.example/robots.txt": boom})
    f2, _ = make(t2, tmp_path)
    with pytest.raises(RobotsDenied):
        f2.get("https://other.example/a")
    # the 403 is remembered for the day; the network failure is not pinned to disk
    records = [json.loads(p.read_text()) for p in (tmp_path / "web" / "robots").glob("*.json")]
    assert [r["status"] for r in records] == [403]


def test_per_host_delay_is_honoured(tmp_path):
    routes = {f"https://example.com/{n}": ok() for n in "abc"}
    routes["https://other.example/x"] = ok()
    t = FakeTransport(routes)
    f, ft = make(t, tmp_path, per_host_delay=2.0, cache_dir=None)
    f.get("https://example.com/a")  # robots + page: one sleep between them
    assert ft.slept == [2.0]
    ft.t += 0.5
    f.get("https://example.com/b")
    assert ft.slept == [2.0, 1.5]
    f.get("https://other.example/x")  # different host: robots (no wait) + page (wait)
    assert ft.slept == [2.0, 1.5, 2.0]
    ft.t += 10
    f.get("https://example.com/c")  # long enough ago: no sleep
    assert ft.slept == [2.0, 1.5, 2.0]


def test_crawl_delay_raises_the_per_host_delay(tmp_path):
    t = FakeTransport({
        "https://example.com/robots.txt": ok(b"User-agent: *\nCrawl-delay: 7\n"),
        "https://example.com/a": ok(),
    })
    f, ft = make(t, tmp_path, per_host_delay=1.0)
    f.get("https://example.com/a")
    assert ft.slept == [7.0]


def test_304_is_served_from_cache(tmp_path):
    first = ok(b"body v1", {"Content-Type": "text/html", "ETag": '"v1"', "Last-Modified": "Wed, 23 Sep 2026 00:00:00 GMT"})
    seen = []

    def page(req):
        seen.append(dict(req.headers))
        return first if len(seen) == 1 else TransportResponse(304, {}, b"")

    t = FakeTransport({"https://example.com/a": page})
    f, _ = make(t, tmp_path)
    assert f.get("https://example.com/a").from_cache is False
    again = f.get("https://example.com/a")
    assert (again.status, again.content, again.from_cache) == (200, b"body v1", True)
    assert seen[1]["If-None-Match"] == '"v1"'
    assert seen[1]["If-Modified-Since"] == "Wed, 23 Sep 2026 00:00:00 GMT"


def test_max_age_serves_cache_without_network(tmp_path):
    now = [DAY1]
    t = FakeTransport({"https://example.com/a": ok(b"x")})
    f, _ = make(t, tmp_path, now=lambda: now[0])
    f.get("https://example.com/a")
    calls = len(t.calls)
    now[0] += 60
    assert f.get("https://example.com/a", max_age=3600).from_cache is True
    assert len(t.calls) == calls
    now[0] += 7200
    f.get("https://example.com/a", max_age=3600)
    assert len(t.calls) == calls + 1


def test_size_cap_enforced(tmp_path):
    t = FakeTransport({"https://example.com/big": ok(b"x" * 101)})
    f, _ = make(t, tmp_path, max_bytes=100)
    with pytest.raises(TooLarge):
        f.get("https://example.com/big")


def test_default_transport_caps_decompressed_gzip():
    with pytest.raises(TooLarge):
        web._decode_body(gzip.compress(b"a" * 10_000), "gzip", 1000)
    assert web._decode_body(gzip.compress(b"abc"), "gzip", 1000) == b"abc"


@pytest.mark.parametrize("host", ["internal.example", "mixed.example", "mapped.example", "metadata.example"])
def test_private_addresses_refused(tmp_path, host):
    t = FakeTransport()
    f, _ = make(t, tmp_path)
    with pytest.raises(BlockedURL):
        f.get(f"http://{host}/admin")
    assert t.calls == []


def test_ip_literals_and_unresolvable_hosts_refused(tmp_path):
    f, _ = make(FakeTransport(), tmp_path)
    f._resolver = lambda host: [host] if host[0].isdigit() or ":" in host else []
    for url in ("http://127.0.0.1/", "http://[::1]/", "http://192.168.1.1:8080/", "http://nowhere.example/"):
        with pytest.raises(BlockedURL):
            f.get(url)


def test_private_allowed_when_explicit(tmp_path):
    t = FakeTransport({"http://internal.example/ok": ok()})
    f, _ = make(t, tmp_path, allowed_hosts=["internal.example"])
    assert f.get("http://internal.example/ok").status == 200
    f2, _ = make(t, tmp_path, allow_private=True)
    assert f2.get("http://internal.example/ok").status == 200


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "example.com/a"])
def test_non_http_schemes_refused(tmp_path, url):
    t = FakeTransport()
    f, _ = make(t, tmp_path)
    with pytest.raises(BlockedURL):
        f.get(url)
    assert t.calls == []


def test_redirect_followed_and_each_hop_checked(tmp_path):
    t = FakeTransport({
        "https://example.com/old": ok(b"", {"Location": "https://other.example/new"}, status=301),
        "https://other.example/new": ok(b"moved"),
    })
    f, _ = make(t, tmp_path)
    resp = f.get("https://example.com/old")
    assert (resp.url, resp.content) == ("https://other.example/new", b"moved")
    assert "https://other.example/robots.txt" in t.urls()  # robots checked for the new host


def test_redirect_to_disallowed_scheme_refused(tmp_path):
    t = FakeTransport({"https://example.com/r": ok(b"", {"Location": "file:///etc/passwd"}, status=302)})
    f, _ = make(t, tmp_path)
    with pytest.raises(BlockedURL):
        f.get("https://example.com/r")


def test_redirect_to_private_address_refused(tmp_path):
    t = FakeTransport({"https://example.com/r": ok(b"", {"Location": "http://metadata.example/latest"}, status=307)})
    f, _ = make(t, tmp_path)
    with pytest.raises(BlockedURL):
        f.get("https://example.com/r")
    assert "http://metadata.example/latest" not in t.urls()


def test_redirect_loop_bounded(tmp_path):
    t = FakeTransport({"https://example.com/loop": ok(b"", {"Location": "/loop"}, status=302)})
    f, _ = make(t, tmp_path, max_redirects=3)
    with pytest.raises(FetchError, match="redirects"):
        f.get("https://example.com/loop")


def test_http_errors_and_transport_errors(tmp_path):
    def boom(req):
        raise TimeoutError("timed out")

    t = FakeTransport({"https://example.com/404": ok(b"no", status=404), "https://example.com/slow": boom})
    f, _ = make(t, tmp_path)
    with pytest.raises(HTTPStatusError) as exc:
        f.get("https://example.com/404")
    assert exc.value.status == 404
    assert f.get("https://example.com/404", raise_for_status=False).status == 404
    with pytest.raises(FetchError, match="timed out"):
        f.get("https://example.com/slow")


def test_get_json(tmp_path):
    t = FakeTransport({"https://example.com/api": ok(b'{"a": [1, 2]}', {"Content-Type": "application/json"})})
    f, _ = make(t, tmp_path)
    assert f.get_json("https://example.com/api") == {"a": [1, 2]}
    assert t.calls[-1].headers["Accept"] == "application/json"


def test_public_ip_classifier():
    assert web.is_public_ip("93.184.216.34")
    assert web.is_public_ip("2606:4700::1111")
    for addr in ("10.1.2.3", "172.16.0.1", "192.168.0.1", "127.0.0.1", "169.254.169.254",
                 "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:10.0.0.1", "junk"):
        assert not web.is_public_ip(addr), addr


def test_real_transport_is_the_default():
    assert Fetcher(UA)._transport is urllib_transport


# ---- article extraction --------------------------------------------------------

def test_extract_article_from_saved_news_page():
    html = (FIX / "news_article.html").read_bytes()
    art = extract_article(html, "https://gazette.example.org/2026/09/tidal-barrier")
    assert "Tidal barrier upgrade approved" in art.title
    assert "approved a long-planned upgrade" in art.text
    assert "consultation on noise limits" in art.text
    assert "window.tracking" not in art.text
    assert "Unrelated story one" not in art.text
    assert art.byline == "Sam Example"
    assert art.published is not None and art.published.date().isoformat() == "2026-09-21"
    assert art.lang in ("", "en")


def test_extract_article_falls_back_to_tag_stripper(monkeypatch):
    import trafilatura

    monkeypatch.setattr(trafilatura, "bare_extraction", lambda *a, **k: None)
    art = extract_article(
        "<html><head><title>Hi &amp; bye</title><script>var x=1;</script></head>"
        "<body><p>First para.</p><p>Second &lt;para&gt;.</p></body></html>",
        "https://example.com/x",
    )
    assert art.title == "Hi & bye"
    assert art.text == "First para.\nSecond <para>."
    assert art.byline == "" and art.published is None


# ---- review round 2 ------------------------------------------------------------

def test_corrupt_or_truncated_gzip_is_a_fetch_error():
    with pytest.raises(FetchError, match="corrupt"):
        web._decode_body(b"\x1f\x8b\x08\x00" + b"\xff" * 40, "gzip", 1000)
    with pytest.raises(FetchError, match="truncated"):
        web._decode_body(gzip.compress(b"hello world")[:-8], "gzip", 1000)


def test_transport_zlib_error_is_wrapped(tmp_path):
    import zlib

    def bad(req):
        raise zlib.error("invalid stored block lengths")

    f, _ = make(FakeTransport({"https://example.com/z": bad}), tmp_path)
    with pytest.raises(FetchError):
        f.get("https://example.com/z")


@pytest.mark.parametrize("target", ["https://other.example/y", "http://example.com/y", "https://example.com:8443/y"])
def test_caller_headers_dropped_on_origin_change(tmp_path, target):
    t = FakeTransport({
        "https://example.com/x": ok(b"", {"Location": target}, status=302),
        target: ok(b"{}"),
    })
    f, _ = make(t, tmp_path, respect_robots=False)
    f.get("https://example.com/x", headers={"Authorization": "Bearer T", "Cookie": "s=1", "X-Api-Key": "k", "Accept": "a/b"})
    first, hop = t.calls[0].headers, t.calls[-1].headers
    assert first["Authorization"] == "Bearer T" and first["X-Api-Key"] == "k"
    assert t.calls[-1].url == target
    assert {k.lower() for k in hop} == {"user-agent", "accept"}


def test_caller_headers_kept_on_same_origin_redirect(tmp_path):
    t = FakeTransport({
        "https://example.com/x": ok(b"", {"Location": "/y"}, status=301),
        "https://example.com/y": ok(),
    })
    f, _ = make(t, tmp_path, respect_robots=False)
    f.get("https://example.com/x", headers={"Authorization": "Bearer T"})
    assert t.calls[-1].headers["Authorization"] == "Bearer T"


def test_default_resolver_keeps_order_and_prefers_ipv4(monkeypatch):
    import socket

    infos = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::1", 0, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.40", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.40", 0)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: infos)
    assert web._default_resolver("dual.example") == ["93.184.216.40", "93.184.216.34", "2606:2800:220:1::1"]


def test_every_validated_address_is_offered_and_tried_in_turn(tmp_path, monkeypatch):
    import socket

    t = FakeTransport({"https://dual.example/": ok()})
    f, _ = make(t, tmp_path, respect_robots=False)
    f._resolver = lambda host: ["93.184.216.34", "2606:2800:220:1::1"]
    f.get("https://dual.example/")
    assert t.calls[-1].ips == ("93.184.216.34", "2606:2800:220:1::1") and t.calls[-1].ip == "93.184.216.34"

    tried = []

    def fake_connect(addr, timeout):
        tried.append(addr)
        if addr[0] == "93.184.216.34":
            raise OSError("unreachable")
        return "sock"

    monkeypatch.setattr(socket, "create_connection", fake_connect)
    assert web._connect_any(["93.184.216.34", "2606:2800:220:1::1"], 443, 5) == "sock"
    assert tried == [("93.184.216.34", 443), ("2606:2800:220:1::1", 443)]
    tried.clear()
    with pytest.raises(OSError):
        web._connect_any(["93.184.216.34"], 443, 5)


@pytest.mark.parametrize("url,host,port", [
    ("http://[2606:2800:220:1::1]/x", "2606:2800:220:1::1", 80),
    ("https://[2606:2800:220:1::1]:8443/x", "2606:2800:220:1::1", 8443),
    ("https://example.com/x", "example.com", 443),
])
def test_connection_gets_explicit_port_for_ipv6_literals(url, host, port):
    conn = web._connection_for(url, ["2606:2800:220:1::1"], 5)
    assert (conn.host, conn.port) == (host, port)


def test_cache_writes_are_atomic_and_io_errors_are_misses(tmp_path, monkeypatch):
    t = FakeTransport({"https://example.com/a": ok(b"v1", {"ETag": '"1"'})})
    f, _ = make(t, tmp_path, respect_robots=False)
    f.get("https://example.com/a")
    assert not list((tmp_path / "web").glob("*.tmp"))

    meta_path, body_path = f._cache_paths("https://example.com/a")
    body_path.write_bytes(b"half-written")  # body changed but meta not: must not validate
    assert f._cache_load("https://example.com/a") is None
    meta_path.write_text("{not json", encoding="utf-8")
    assert f._cache_load("https://example.com/a") is None

    def locked(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(web, "_atomic_write", locked)
    f2, _ = make(FakeTransport({"https://example.com/b": ok(b"b")}), tmp_path)
    assert f2.get("https://example.com/b").content == b"b"  # robots + page writes fail silently


def test_get_json_on_non_json_raises_fetch_error(tmp_path):
    t = FakeTransport({"https://example.com/j": ok(b"<html>", {"Content-Type": "text/html"})})
    f, _ = make(t, tmp_path, respect_robots=False)
    with pytest.raises(web.InvalidJSON):
        f.get_json("https://example.com/j")
    assert issubclass(web.InvalidJSON, FetchError)


def test_huge_crawl_delay_skips_host_instead_of_stalling(tmp_path):
    t = FakeTransport({"https://example.com/robots.txt": ok(b"User-agent: *\nCrawl-delay: 86400\n")})
    f, ft = make(t, tmp_path)
    with pytest.raises(RobotsDenied, match="Crawl-delay"):
        f.get("https://example.com/a")
    assert ft.slept == [] and "https://example.com/a" not in t.urls()


def test_304_without_cached_copy_is_an_error(tmp_path):
    t = FakeTransport({"https://example.com/n": TransportResponse(304, {}, b"")})
    f, _ = make(t, tmp_path, respect_robots=False, cache_dir=None)
    with pytest.raises(FetchError, match="304"):
        f.get("https://example.com/n")
    # ...unless the caller asked the conditional question themselves
    resp = f.get("https://example.com/n", headers={"If-None-Match": '"x"'})
    assert resp.status == 304 and not resp.from_cache


@pytest.mark.parametrize("addr", ["64:ff9b::7f00:1", "64:ff9b::a00:1", "::127.0.0.1", "::10.0.0.1",
                                  "2002:7f00:1::1", "2002:a00:1::1", "::ffff:169.254.169.254"])
def test_ipv6_forms_embedding_private_ipv4_are_refused(addr):
    assert not web.is_public_ip(addr)


# ---- coordinator fix round (issue #50) -----------------------------------------

def test_malformed_redirect_location_is_a_fetch_error(tmp_path):
    t = FakeTransport({"https://example.com/r": ok(b"", {"Location": "http://[::1/"}, status=302)})
    f, _ = make(t, tmp_path, respect_robots=False)
    with pytest.raises(FetchError):
        f.get("https://example.com/r")


def test_corrupt_robots_cache_record_is_treated_as_a_miss(tmp_path):
    web_dir = tmp_path / "web"
    robots_dir = web_dir / "robots"
    robots_dir.mkdir(parents=True)
    day = datetime.fromtimestamp(DAY1, timezone.utc).date().isoformat()
    for bad_status in ("abc", None):
        key = web._key("https://example.com")
        (robots_dir / f"{key}.json").write_text(json.dumps({"day": day, "status": bad_status, "text": ""}))
        t = FakeTransport({"https://example.com/x": ok()})
        f, _ = make(t, tmp_path, now=lambda: DAY1, cache_dir=web_dir)
        assert f.get("https://example.com/x").status == 200  # re-fetches robots.txt instead of raising


def test_max_age_with_corrupt_fetched_at_is_a_cache_miss(tmp_path):
    t = FakeTransport({"https://example.com/w": ok(b"hi")})
    f, _ = make(t, tmp_path, respect_robots=False)
    f.get("https://example.com/w")
    meta_path, _ = f._cache_paths("https://example.com/w")
    meta = json.loads(meta_path.read_text())
    for bad in ("not-a-number", None):
        meta_path.write_text(json.dumps({**meta, "fetched_at": bad}))
        resp = f.get("https://example.com/w", max_age=1_000_000)
        assert resp.from_cache is False  # falls back to the network instead of raising


def test_per_host_delay_keyed_on_hostname_not_port(tmp_path):
    # http->https redirect hits the same host twice within one get(): the second
    # hop must still honour the delay the first hop just started, even though the
    # scheme/port changed (previously the politeness key included the port, so
    # the second hop looked like an untouched host and skipped the wait).
    t = FakeTransport({
        "http://example.com/": ok(b"", {"Location": "https://example.com/"}, status=301),
        "https://example.com/": ok(),
    })
    f, ft = make(t, tmp_path, respect_robots=False, per_host_delay=5.0, cache_dir=None)
    f.get("http://example.com/")
    assert ft.slept == [5.0]


def test_normalize_url_strips_default_ports_but_keeps_others():
    assert web.normalize_url("http://example.com:80/x") == "http://example.com/x"
    assert web.normalize_url("https://example.com:443/x") == "https://example.com/x"
    assert web.normalize_url("http://example.com:8080/x") == "http://example.com:8080/x"
    assert web.normalize_url("https://example.com:8443/x") == "https://example.com:8443/x"


def test_default_port_and_portless_urls_share_one_cache_and_robots_origin(tmp_path):
    t = FakeTransport({"https://example.com/x": ok()})
    f, _ = make(t, tmp_path)
    f.get("https://example.com/x")
    f.get("https://example.com:443/x")
    assert t.urls().count("https://example.com/robots.txt") == 1


def test_allowed_hosts_are_idna_encoded_at_construction(tmp_path):
    t = FakeTransport({"https://xn--bcher-kva.de/": ok()})
    f, _ = make(t, tmp_path, respect_robots=False, allowed_hosts=["bücher.de"])
    f._resolver = lambda host: ["192.168.1.5"]
    assert f.get("https://bücher.de/").status == 200

    with pytest.raises(ValueError):
        make(FakeTransport(), tmp_path, allowed_hosts=["ä" * 70])


def test_iri_urls_are_percent_encoded(tmp_path):
    target = "https://xn--mnchen-3ya.example/wiki/%C3%9Cbersicht%20neu?q=caf%C3%A9&x=1"
    t = FakeTransport({target: ok()})
    f, _ = make(t, tmp_path, respect_robots=False)
    f._resolver = lambda host: ["93.184.216.34"]
    resp = f.get("https://München.example/wiki/Übersicht neu?q=café&x=1#frag")
    assert t.calls[-1].url == target and resp.url == target
    assert web.normalize_url("https://example.com/a%20b?x=%2F") == "https://example.com/a%20b?x=%2F"
    assert web.normalize_url("http://[::1]:8080/p") == "http://[::1]:8080/p"
