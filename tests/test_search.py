"""lmsa.search: SearXNG JSON API client (#51). No network -- a fake fetcher
stands in for lmsa.web.Fetcher (same seam lmsa.feeds tests use)."""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import sys
import textwrap
import urllib.parse
import zlib
from pathlib import Path

import pytest

import lmsa.search as search_mod
from lmsa.search import (
    SearchError,
    SearchNotJSON,
    SearchRateLimited,
    SearchUnavailable,
    SearXNG,
    command_executable,
    is_connection_error,
    unresolved_executable,
)
from lmsa.web import FetchError, HTTPStatusError, InvalidJSON

ROOT = Path(__file__).resolve().parents[1]


def _connection_refused(message: str = "connection refused") -> FetchError:
    """A FetchError shaped like a real connection failure -- an OSError
    cause -- rather than a bare FetchError with none. #89's review: a bare
    FetchError is no longer classified as connection-level (see
    is_connection_error), so a test simulating "the instance plain isn't
    answering" has to build one of these."""
    exc = FetchError(message)
    exc.__cause__ = ConnectionRefusedError(message)
    return exc


@pytest.fixture(autouse=True)
def _reset_starter_state():
    """The on-demand starter (#89) is module-level by design -- reset it
    around every test so tests can't see each other's spawned processes.
    Terminates any leftover starter (real or faked) instead of just
    dropping the handle, so a bug that lets one survive a test doesn't leave
    an orphaned process running."""
    def _cleanup():
        if search_mod._starter_proc is not None:
            try:
                search_mod._terminate_starter()
            except Exception:
                pass
        search_mod._starter_proc = None
        search_mod._starter_command = None
        search_mod._starter_atexit_registered = False

    _cleanup()
    yield
    _cleanup()


class FakeFetcher:
    """Records the last URL asked for; answers however the test wants."""

    def __init__(self, json_result=None, raises: Exception | None = None):
        self.json_result = json_result
        self.raises = raises
        self.urls: list[str] = []
        self.timeouts: list[float | None] = []  # each call's timeout= kwarg (#89), None if omitted

    def get_json(self, url, **kwargs):
        self.urls.append(url)
        self.timeouts.append(kwargs.get("timeout"))
        if self.raises is not None:
            raise self.raises
        return self.json_result

    def last_query(self) -> dict:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.urls[-1]).query))


class QueueFetcher:
    """get_json works through a list of canned outcomes (an Exception
    instance, or a JSON-shaped value); the last one repeats forever once
    the others are exhausted -- lets a test script "down, then up"."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    def get_json(self, url, **kwargs):
        self.urls.append(url)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakePopen:
    """Stands in for subprocess.Popen: no real process. poll()/terminate()/
    kill()/wait() are the only surface SearXNG's starter code touches."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def die(self):
        """Test helper: simulate the process having exited on its own."""
        self.alive = False


def fake_popen_factory():
    """A popen= callable that records every FakePopen it spawns."""
    spawned: list[FakePopen] = []

    def factory(*args, **kwargs):
        proc = FakePopen(*args, **kwargs)
        spawned.append(proc)
        return proc

    factory.spawned = spawned
    return factory


def fake_clock(values):
    it = iter(values)
    return lambda: next(it)


CANNED_PING = {"results": []}


CANNED = {
    "query": "harbour reopening",
    "results": [
        {"url": "https://news.example.org/a", "title": "A", "content": "snippet a",
         "engine": "google", "publishedDate": "2026-09-20T10:00:00"},
        {"url": "https://news.example.org/b", "title": "B", "content": "snippet b",
         "engines": ["bing", "duckduckgo"]},
        # duplicate of "a" once normalized (trailing default port + fragment)
        {"url": "https://news.example.org:443/a#ignored", "title": "A dup", "content": "x", "engine": "bing"},
        {"url": "javascript:alert(1)", "title": "bad scheme", "content": "x", "engine": "google"},
        "not a dict",
        {"title": "no url at all", "content": "x"},
    ],
}


def test_parses_title_url_snippet_engine_published():
    fetcher = FakeFetcher(json_result=CANNED)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("harbour reopening")
    first = results[0]
    assert first.title == "A" and first.url == "https://news.example.org/a"
    assert first.snippet == "snippet a" and first.engine == "google"
    assert first.published is not None and first.published.year == 2026


def test_engine_falls_back_to_first_of_engines_list():
    fetcher = FakeFetcher(json_result=CANNED)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")
    second = next(r for r in results if r.url.endswith("/b"))
    assert second.engine == "bing"
    assert second.published is None  # no publishedDate on this one


def test_drops_non_http_scheme_and_urlless_entries():
    fetcher = FakeFetcher(json_result=CANNED)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")
    assert all(r.url.startswith("http") for r in results)
    assert not any("bad scheme" in r.title for r in results)
    assert not any(r.title == "no url at all" for r in results)


def test_dedupes_by_normalized_url_keeping_first_occurrence():
    fetcher = FakeFetcher(json_result=CANNED)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")
    urls = [r.url for r in results]
    assert urls.count("https://news.example.org/a") == 1
    assert next(r for r in results if r.url == "https://news.example.org/a").title == "A"  # first wins


def test_max_results_caps_after_dedup():
    fetcher = FakeFetcher(json_result=CANNED)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q", max_results=1)
    assert len(results) == 1


def test_request_includes_format_json_and_optional_params():
    fetcher = FakeFetcher(json_result={"results": []})
    SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search(
        "harbour", categories="news", time_range="week", language="en")
    q = fetcher.last_query()
    assert q["q"] == "harbour" and q["format"] == "json"
    assert q["categories"] == "news" and q["time_range"] == "week" and q["language"] == "en"


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        SearXNG("http://127.0.0.1:8888", fetcher=FakeFetcher()).search("   ")


def test_403_on_format_json_is_search_not_json_with_a_clear_message():
    fetcher = FakeFetcher(raises=HTTPStatusError(403, "http://127.0.0.1:8888/search"))
    with pytest.raises(SearchNotJSON, match="settings.yml"):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_html_only_instance_raises_search_not_json():
    """A 200 whose body isn't JSON -- Fetcher.get_json raises InvalidJSON."""
    fetcher = FakeFetcher(raises=InvalidJSON("http://x/search: not JSON"))
    with pytest.raises(SearchNotJSON, match="HTML-only"):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_unexpected_json_shape_is_search_not_json():
    fetcher = FakeFetcher(json_result={"unexpected": "shape"})
    with pytest.raises(SearchNotJSON):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_429_is_search_rate_limited_not_an_empty_list():
    fetcher = FakeFetcher(raises=HTTPStatusError(429, "http://127.0.0.1:8888/search"))
    with pytest.raises(SearchRateLimited):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_other_http_status_is_search_unavailable():
    fetcher = FakeFetcher(raises=HTTPStatusError(500, "http://127.0.0.1:8888/search"))
    with pytest.raises(SearchUnavailable):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_transport_failure_is_search_unavailable():
    fetcher = FakeFetcher(raises=FetchError("connection refused"))
    with pytest.raises(SearchUnavailable):
        SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")


def test_search_unavailable_and_not_json_are_both_search_errors():
    assert issubclass(SearchUnavailable, SearchError)
    assert issubclass(SearchNotJSON, SearchError)
    assert issubclass(SearchRateLimited, SearchError)


def test_malformed_result_url_is_skipped_not_fatal():
    """One bad result (per-item isolation) doesn't kill the whole search."""
    canned = {"results": [
        {"url": "http://[::not-an-ip", "title": "broken", "content": "x", "engine": "g"},
        {"url": "https://ok.example.org/x", "title": "fine", "content": "x", "engine": "g"},
    ]}
    fetcher = FakeFetcher(json_result=canned)
    results = SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("q")
    assert [r.title for r in results] == ["fine"]


def test_default_fetcher_scopes_the_configured_host_only():
    """No allow_private=True wholesale -- only the configured SearXNG host is
    exempted from the private-address guard, and robots.txt is off only for it."""
    searx = SearXNG("http://127.0.0.1:8888")
    fetcher = searx.fetcher
    assert fetcher.allow_private is False
    assert "127.0.0.1" in fetcher.allowed_hosts
    assert fetcher.respect_robots is False
    assert fetcher.cache_dir is None


def test_lan_hostname_via_injected_resolver_is_treated_as_local():
    """A hostname (not a literal IP) that resolves to a private address --
    via the resolver test seam, no real DNS -- gets the same exemption."""
    searx = SearXNG("http://searxng.lan:8888", resolver=lambda host: ["10.0.0.5"])
    fetcher = searx.fetcher
    assert fetcher.allow_private is False
    assert "searxng.lan" in fetcher.allowed_hosts
    assert fetcher.respect_robots is False


def test_public_base_url_gets_no_exemption_and_keeps_robots_on():
    """A public SearXNG instance is not the operator's own local service:
    no allowed_hosts exemption, allow_private stays False, robots stays on."""
    searx = SearXNG("https://search.example.org", resolver=lambda host: ["93.184.216.34"])
    fetcher = searx.fetcher
    assert fetcher.allow_private is False
    assert fetcher.allowed_hosts == set()
    assert fetcher.respect_robots is True


def test_public_ip_literal_base_url_gets_no_exemption():
    searx = SearXNG("http://93.184.216.34:8888")
    fetcher = searx.fetcher
    assert fetcher.allowed_hosts == set()
    assert fetcher.respect_robots is True


def test_unresolvable_host_is_not_treated_as_local():
    """A resolver that can't resolve the host (or raises) is the safe
    default: not local, so robots stays on and nothing is exempted."""
    def broken(host):
        raise OSError("no DNS")
    searx = SearXNG("http://searxng.example.org", resolver=broken)
    fetcher = searx.fetcher
    assert fetcher.allowed_hosts == set()
    assert fetcher.respect_robots is True


def test_default_resolver_is_used_when_none_given(monkeypatch):
    """No resolver passed -> lmsa.search._resolve_host (socket.getaddrinfo),
    not a real network call for search() itself (that still goes through
    the injected fetcher)."""
    import lmsa.search as search_mod

    monkeypatch.setattr(search_mod, "_resolve_host", lambda host: ["10.1.2.3"])
    searx = SearXNG("http://searxng.internal:8888")
    assert "searxng.internal" in searx.fetcher.allowed_hosts
    assert searx.fetcher.respect_robots is False


def test_categories_list_is_sent_comma_separated():
    fetcher = FakeFetcher(json_result={"results": []})
    SearXNG("http://127.0.0.1:8888", fetcher=fetcher).search("harbour", categories=["news", "science"])
    assert fetcher.last_query()["categories"] == "news,science"


# ---------------------------------------------------------------------------
# On-demand start (#89)
# ---------------------------------------------------------------------------

def test_spawns_on_connection_failure_then_ready_then_retry_succeeds():
    """First attempt fails at the transport level; the starter is spawned;
    the readiness probe succeeds; the original query is retried and works."""
    fetcher = QueueFetcher([_connection_refused(), CANNED_PING, {"results": []}])
    popen = fake_popen_factory()
    searx = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher,
        start_command="fake-start-cmd", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    results = searx.search("harbour reopening")
    assert results == []
    assert len(popen.spawned) == 1
    assert popen.spawned[0].args[0] == "fake-start-cmd" or popen.spawned[0].args[0][0] == "fake-start-cmd"


@pytest.mark.parametrize("status", [403, 429, 500])
def test_no_spawn_on_http_status_errors(status):
    fetcher = FakeFetcher(raises=HTTPStatusError(status, "http://127.0.0.1:8888/search"))
    popen = fake_popen_factory()
    searx = SearXNG("http://127.0.0.1:8888", fetcher=fetcher, start_command="fake-start-cmd", popen=popen)
    with pytest.raises(SearchError):
        searx.search("q")
    assert popen.spawned == []


def test_no_spawn_when_start_command_is_none():
    fetcher = FakeFetcher(raises=FetchError("connection refused"))
    popen = fake_popen_factory()
    searx = SearXNG("http://127.0.0.1:8888", fetcher=fetcher, start_command=None, popen=popen)
    with pytest.raises(SearchUnavailable):
        searx.search("q")
    assert popen.spawned == []


def test_ready_timeout_raises_search_unavailable_naming_command_and_timeout():
    fetcher = FakeFetcher(raises=_connection_refused())  # never comes up
    popen = fake_popen_factory()
    searx = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher,
        start_command="fake-start-cmd", start_timeout=2,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0, 5]),
    )
    with pytest.raises(SearchUnavailable, match="fake-start-cmd") as excinfo:
        searx.search("q")
    assert "2" in str(excinfo.value)
    assert len(popen.spawned) == 1  # spawned once, even though it never became ready


def test_second_search_in_same_process_does_not_spawn_again():
    popen = fake_popen_factory()

    fetcher1 = QueueFetcher([_connection_refused(), CANNED_PING, {"results": []}])
    searx1 = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher1,
        start_command="fake-start-cmd", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    searx1.search("q1")
    assert len(popen.spawned) == 1

    fetcher2 = QueueFetcher([_connection_refused(), CANNED_PING, {"results": []}])
    searx2 = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher2,
        start_command="fake-start-cmd", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    searx2.search("q2")
    assert len(popen.spawned) == 1  # still running -- not spawned a second time


def test_dead_starter_process_is_respawned():
    popen = fake_popen_factory()

    fetcher1 = QueueFetcher([_connection_refused(), CANNED_PING, {"results": []}])
    searx1 = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher1,
        start_command="fake-start-cmd", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    searx1.search("q1")
    assert len(popen.spawned) == 1
    popen.spawned[0].die()  # simulate it having exited on its own

    fetcher2 = QueueFetcher([_connection_refused(), CANNED_PING, {"results": []}])
    searx2 = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher2,
        start_command="fake-start-cmd", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    searx2.search("q2")
    assert len(popen.spawned) == 2  # respawned


def test_atexit_cleanup_terminates_the_starter_process():
    popen = fake_popen_factory()
    search_mod._ensure_starter_running("fake-start-cmd", popen)
    assert len(popen.spawned) == 1
    proc = popen.spawned[0]
    assert not proc.terminated

    search_mod._terminate_starter()

    assert proc.terminated
    assert proc.poll() is not None


def test_atexit_cleanup_is_a_noop_when_nothing_was_spawned():
    search_mod._terminate_starter()  # must not raise


def test_atexit_cleanup_kills_a_process_that_wont_terminate():
    class Stubborn(FakePopen):
        def terminate(self):
            raise OSError("no such process")

    popen = lambda *a, **kw: Stubborn(*a, **kw)
    search_mod._ensure_starter_running("fake-start-cmd", popen)
    search_mod._terminate_starter()
    proc = search_mod._starter_proc
    assert proc.killed


def test_default_construction_path_picks_up_start_command_from_settings(monkeypatch):
    """A real SearXNG(url) call (no start_command kwarg at all) must read
    LMSA_SEARXNG_START from settings() -- the sentinel-default seam every
    production call site (SearXNG(settings().searxng_url)) relies on."""
    monkeypatch.setenv("LMSA_SEARXNG_START", "wsl.exe -d Ubuntu -u root --exec sleep 7200")
    monkeypatch.setenv("LMSA_SEARXNG_START_TIMEOUT", "45")
    searx = SearXNG("http://127.0.0.1:8888")
    assert searx.start_command == "wsl.exe -d Ubuntu -u root --exec sleep 7200"
    assert searx.start_timeout == 45.0


def test_default_construction_path_has_no_start_command_when_env_unset(monkeypatch):
    monkeypatch.delenv("LMSA_SEARXNG_START", raising=False)
    monkeypatch.delenv("LMSA_SEARXNG_START_TIMEOUT", raising=False)  # else a set TIMEOUT breaks this assertion
    searx = SearXNG("http://127.0.0.1:8888")
    assert searx.start_command is None
    assert searx.start_timeout == 60


# ---------------------------------------------------------------------------
# is_connection_error: classify by the cause chain, not by exclusion (#89 review)
# ---------------------------------------------------------------------------

def _wrap(cause: BaseException, message: str = "wrapped") -> FetchError:
    exc = FetchError(message)
    exc.__cause__ = cause
    return exc


@pytest.mark.parametrize("cause", [
    ConnectionRefusedError("refused"),
    socket.timeout("timed out"),
    OSError("no route to host"),
    http.client.RemoteDisconnected("Remote end closed connection"),
])
def test_is_connection_error_true_for_refused_and_timeout(cause):
    assert is_connection_error(_wrap(cause))


@pytest.mark.parametrize("exc", [
    FetchError("HTTP 301 without Location: http://x"),  # 30x without Location, no cause
    FetchError("HTTP 304 without a cached copy: http://x"),  # 304, no cached copy, no cause
    FetchError("more than 5 redirects: http://x"),  # redirect loop, no cause
    _wrap(zlib.error("Error -3 while decompressing")),  # corrupt gzip body
    _wrap(ValueError("bad redirect Location")),  # malformed redirect (a plain ValueError cause)
    HTTPStatusError(500, "http://x"),  # any bare HTTP status error
])
def test_is_connection_error_false_for_up_but_broken_instance(exc):
    """These are all FetchErrors from an instance that answered -- just
    badly -- so starting SearXNG again would only waste a full timeout."""
    assert not is_connection_error(exc)


def test_is_connection_error_false_for_none_and_non_fetch_errors():
    assert not is_connection_error(None)
    assert not is_connection_error(ValueError("not even a FetchError"))


def test_is_connection_error_walks_the_cause_chain():
    """A FetchError wrapping another exception that itself wraps the real
    OSError (double-wrapped) still counts."""
    inner = FetchError("inner")
    inner.__cause__ = ConnectionRefusedError("refused")
    outer = FetchError("outer")
    outer.__cause__ = inner
    assert is_connection_error(outer)


# ---------------------------------------------------------------------------
# Bounded readiness probes (#89 review): each probe gets its own short
# timeout instead of the Fetcher's full one, so the total wait can't
# overshoot start_timeout by almost a whole fetch timeout.
# ---------------------------------------------------------------------------

def test_probe_timeout_is_capped_and_bounded_by_remaining_budget():
    fetcher = FakeFetcher(raises=_connection_refused())  # never comes up
    popen = fake_popen_factory()
    searx = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher,
        start_command="fake-start-cmd", start_timeout=12,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0, 6, 11, 12.5]),
    )
    with pytest.raises(SearchUnavailable):
        searx.search("q")
    # index 0 is the plain (unbounded) search attempt; the three readiness
    # probes that follow are capped at 5s, then bounded by what's left
    assert fetcher.timeouts == [None, 5, 5, 1]
    assert all(t is None or t <= search_mod._PROBE_TIMEOUT_CAP for t in fetcher.timeouts)


# ---------------------------------------------------------------------------
# Don't echo the full command (#89 review): logs and error text name only
# the configured executable, never the rest of LMSA_SEARXNG_START.
# ---------------------------------------------------------------------------

def test_command_executable_takes_only_the_first_token():
    assert command_executable("wsl.exe -d Ubuntu -u root --exec sleep 7200") == "wsl.exe"
    assert command_executable('"C:\\Program Files\\tool.exe" --flag') in ("C:\\Program Files\\tool.exe",)
    assert command_executable("") == ""


def test_ready_timeout_message_names_only_the_executable():
    fetcher = FakeFetcher(raises=_connection_refused())  # never comes up
    popen = fake_popen_factory()
    searx = SearXNG(
        "http://127.0.0.1:8888", fetcher=fetcher,
        start_command="wsl.exe -d Ubuntu -u root --exec sleep 7200 --secret-flag-should-not-leak",
        start_timeout=2, popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0, 5]),
    )
    with pytest.raises(SearchUnavailable) as excinfo:
        searx.search("q")
    message = str(excinfo.value)
    assert "wsl.exe" in message
    assert "LMSA_SEARXNG_START" in message
    assert "--secret-flag-should-not-leak" not in message
    assert "-d Ubuntu" not in message


# ---------------------------------------------------------------------------
# unresolved_executable: name a misconfigured LMSA_SEARXNG_START (#89 review)
# ---------------------------------------------------------------------------

def test_unresolved_executable_is_none_for_a_resolvable_command(monkeypatch):
    monkeypatch.setattr(search_mod.shutil, "which", lambda exe: r"C:\Windows\System32\wsl.exe")
    assert unresolved_executable("wsl.exe -d Ubuntu -u root --exec sleep 7200") is None


def test_unresolved_executable_names_the_executable_when_not_found(monkeypatch):
    monkeypatch.setattr(search_mod.shutil, "which", lambda exe: None)
    assert unresolved_executable("no-such-tool --flag") == "no-such-tool"


def test_unresolved_executable_is_none_for_falsy_command():
    assert unresolved_executable(None) is None
    assert unresolved_executable("") is None


# ---------------------------------------------------------------------------
# LMSA_SEARXNG_START_TIMEOUT validation (#89 review): a finite value >= 0,
# else fall back to 60 -- checked at the settings() layer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-5", "not-a-number", ""])
def test_bad_start_timeout_values_fall_back_to_60(monkeypatch, raw):
    from lmsa.config import settings

    if raw == "":
        monkeypatch.delenv("LMSA_SEARXNG_START_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("LMSA_SEARXNG_START_TIMEOUT", raw)
    assert settings().searxng_start_timeout == 60.0


@pytest.mark.parametrize("raw,expected", [("45", 45.0), ("0", 0.0), ("0.5", 0.5)])
def test_good_start_timeout_values_are_kept(monkeypatch, raw, expected):
    from lmsa.config import settings

    monkeypatch.setenv("LMSA_SEARXNG_START_TIMEOUT", raw)
    assert settings().searxng_start_timeout == expected


# ---------------------------------------------------------------------------
# Hermetic suite guard (#89 review): LMSA_SEARXNG_START already present in
# the process environment *before* a test runs -- e.g. loaded from a real
# .env by lmsa.config's module-level load_dotenv -- must never spawn.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_env_preexisting_before_suite_start_never_spawns(tmp_path):
    """Proves the tests/conftest.py autouse guard, not this module's own
    fixtures: runs a *separate* pytest process with LMSA_SEARXNG_START
    already set in the environment before that process -- and therefore
    before its test collection -- even starts, and confirms the marker
    command it names is never run. The probe file is placed inside tests/
    itself (cleaned up in `finally`) so tests/conftest.py's autouse fixture
    actually applies to it, the same as for any real test."""
    marker = tmp_path / "marker"
    start_cmd = f'{sys.executable} -c "open(r\'{marker}\', \'w\').close()"'

    probe = Path(__file__).parent / f"test_probe_89_{os.getpid()}.py"
    probe.write_text(textwrap.dedent("""
        from lmsa.search import SearXNG
        from lmsa.web import FetchError

        class _AlwaysDown:
            def get_json(self, url, **kwargs):
                exc = FetchError("connection refused")
                exc.__cause__ = ConnectionRefusedError("refused")
                raise exc

        def test_probe():
            searx = SearXNG(
                "http://127.0.0.1:8888", fetcher=_AlwaysDown(),
                start_timeout=1, sleep=lambda s: None,
            )
            try:
                searx.search("q")
            except Exception:
                pass
    """), encoding="utf-8")

    try:
        env = {**os.environ, "LMSA_SEARXNG_START": start_cmd, "LMSA_SEARXNG_START_TIMEOUT": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(probe), "-p", "no:cacheprovider"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not marker.exists()
    finally:
        probe.unlink(missing_ok=True)


def test_spawn_failure_is_a_search_unavailable_not_a_raw_oserror():
    def popen(*args, **kwargs):
        raise FileNotFoundError("nosuch.exe")

    searx = SearXNG(
        "http://127.0.0.1:8888", fetcher=QueueFetcher([_connection_refused()]),
        start_command="nosuch.exe -x", start_timeout=5,
        popen=popen, sleep=lambda s: None, clock=fake_clock([0, 0]),
    )
    with pytest.raises(SearchUnavailable, match="nosuch.exe"):
        searx.search("harbour reopening")
