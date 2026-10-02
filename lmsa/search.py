"""SearXNG client: open web search for briefings and go-deep (#48).

Talks to one SearXNG instance's JSON API -- typically self-hosted on
localhost/LAN (Docker), but ``LMSA_SEARXNG_URL`` could just as well name a
public instance. So the Fetcher this module builds by default only exempts
the configured host from the private-address guard (``allowed_hosts``,
never ``allow_private=True`` for everything) and only turns robots.txt off
for that host (a local SearXNG's robots.txt, when present at all, is
typically a blanket ``Disallow`` that would also block ``/search``, and
honouring it can't protect anything the operator doesn't already run
themselves) -- and it does either **only when the configured host actually
resolves to a loopback/private/link-local address** (checked with
``lmsa.web.is_public_ip``). A public base_url gets neither: robots.txt stays
honoured and nothing is exempted from the SSRF guard.

CONTRACT
    SearXNG(base_url, *, fetcher=None, resolver=None,
            user_agent=DEFAULT_USER_AGENT, timeout=20)
        resolver(host: str) -> list[str] -- test seam for the locality check
            above (and, when given, also passed through to the Fetcher this
            builds); default resolves via socket.getaddrinfo.
        .search(query, *, categories=None, time_range=None, language=None,
                max_results=20) -> list[SearchResult]
            GET {base_url}/search?q=...&format=json[&categories&time_range&language]
            via ``fetcher.get_json`` (an ``lmsa.web.Fetcher`` by default, or
            anything with a compatible ``get_json(url, **kw)`` -- the test seam).
            Results with a non-http(s) url are dropped; the rest are deduped by
            ``lmsa.web.normalize_url`` (first occurrence wins) and capped to
            ``max_results`` *after* dedup.
    SearchResult(title, url, snippet, engine, published: datetime | None)

    Exceptions -- all subclass SearchError, so callers can isolate one query:
        SearchNotJSON      the instance's JSON API isn't enabled: HTTP 403 on
                            format=json, or a 200 response that isn't JSON at
                            all (an HTML-only instance). Message names the fix
                            (``search.formats: [html, json]`` in settings.yml).
        SearchRateLimited   HTTP 429 -- never silently returned as [].
        SearchUnavailable   any other fetch failure (network, timeout, other
                             HTTP status, blocked URL...); wraps the FetchError.

    On-demand start (#89): when ``LMSA_SEARXNG_START`` names a command that
    brings the instance up (e.g. ``wsl.exe -d Ubuntu -u root --exec sleep
    7200``), ``SearXNG`` defaults ``start_command``/``start_timeout`` from
    ``lmsa.config.settings()`` -- every production call site that just does
    ``SearXNG(settings().searxng_url)`` picks this up for free. On a
    connection-level ``SearchUnavailable`` (its cause chain is an ``OSError``
    or ``http.client.HTTPException`` -- see ``is_connection_error``; not an
    HTTP status, not blocked, not robots-denied, not a redirect/304/gzip
    problem on an instance that's actually up) it spawns the command once per
    process (module-level, shared across instances; respawned if the process
    has exited), polls the instance -- each probe capped at
    ``min(_PROBE_TIMEOUT_CAP, time remaining)`` so one slow probe can't blow
    past ``start_timeout`` -- every 0.5s up to ``start_timeout`` seconds, then
    retries the search once. A timeout raises ``SearchUnavailable`` naming
    only the configured executable (never the full command, which may carry
    flags) and the timeout. The spawned process is terminated at interpreter
    exit. Pass ``start_command=None`` to opt a caller out (e.g. a page-view
    probe that must never spawn anything).

    ``unresolved_executable(command) -> str | None`` -- the executable named
    by ``command`` (its first token, split the same way the spawn will) when
    ``shutil.which`` can't find it; ``None`` if it resolves fine or
    ``command`` is falsy. Lets a caller (Setup's probe) tell "not started
    yet, but configured correctly" apart from "configured with a command
    that can't run at all". ``command_executable(command) -> str`` is the
    first-token split alone (used for that, and to keep the rest of a
    command out of logs/errors).
"""

from __future__ import annotations

import atexit
import http.client
import ipaddress
import logging
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from lmsa.config import settings
from lmsa.feeds import is_http_url, parse_date
from lmsa.web import (
    Fetcher,
    FetchError,
    HTTPStatusError,
    InvalidJSON,
    is_public_ip,
    normalize_url,
)

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "lmsa-search/1.0 (+lmstudio-automation)"
MAX_SNIPPET = 2000  # untrusted external text; bound before it reaches a report or a model
_UNSET = object()  # sentinel: "use settings()" vs. an explicit None


class SearchError(Exception):
    """Base class: anything that stopped a search from returning results."""


class SearchNotJSON(SearchError):
    """The SearXNG instance's JSON API isn't enabled (or reachable but not JSON)."""


class SearchRateLimited(SearchError):
    """SearXNG answered HTTP 429."""


class SearchUnavailable(SearchError):
    """Any other fetch failure reaching the SearXNG instance."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str
    published: datetime | None = None


_PROBE_TIMEOUT_CAP = 5.0  # seconds; bounds a single readiness-probe fetch (#89 review)


def is_connection_error(exc: BaseException | None) -> bool:
    """True only when ``exc`` is a ``FetchError`` whose cause chain contains
    an ``OSError`` (``socket.timeout``, ``ConnectionRefusedError``, a wrapped
    ``URLError``'s ``.reason``...) or an ``http.client.HTTPException`` (e.g.
    ``RemoteDisconnected``) -- positive proof the instance plain isn't
    answering. Classified by what the cause *is*, never by ruling out named
    FetchError subclasses: a redirect loop, a 30x without Location, a 304
    without a cached copy, or a corrupt/truncated gzip body are all bare
    FetchErrors with no such cause, and are correctly NOT connection-level --
    the instance answered, just badly, and starting SearXNG again would only
    waste a full timeout. Used to decide whether a ``SearchUnavailable`` is
    worth trying to start SearXNG for."""
    if not isinstance(exc, FetchError):
        return False
    seen: set[int] = set()
    cause = exc.__cause__
    while cause is not None and id(cause) not in seen:
        if isinstance(cause, (OSError, http.client.HTTPException)):
            return True
        seen.add(id(cause))
        cause = cause.__cause__
    return False


# ---------------------------------------------------------------------------
# On-demand start (#89): one starter process per Python process, shared by
# every SearXNG instance. Module-level by design -- see the docstring above.
# ---------------------------------------------------------------------------

_starter_lock = threading.Lock()
_starter_proc: Any = None  # a subprocess.Popen, or the test seam's fake
_starter_command: str | None = None
_starter_atexit_registered = False


def _spawn(command: str, popen: Callable[..., Any]) -> Any:
    """Spawn ``command`` hidden: no console window on Windows, DEVNULL'd
    std streams either way. POSIX splits the command line ourselves (no
    shell); Windows takes the string as-is (CreateProcess wants that, and
    CREATE_NO_WINDOW only exists there)."""
    if os.name == "nt":
        return popen(
            command,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return popen(
        shlex.split(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def command_executable(command: str) -> str:
    """The first token of ``command`` -- the executable ``_spawn`` will
    actually invoke -- split the same way per platform (see ``_spawn``).
    Used to name just the executable in logs and error text instead of
    echoing the whole command (its flags are noise at best, and an operator
    may not want them repeated in every log line)."""
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        parts = command.split()
    return parts[0].strip('"') if parts else ""


def unresolved_executable(command: str | None) -> str | None:
    """The executable named by ``command`` (parsed the same way ``_spawn``
    will) when ``shutil.which`` can't find it on PATH or as a literal path --
    else ``None``. A falsy ``command`` is also ``None`` (nothing to check).
    Lets a caller (Setup's probe) tell "not started yet, but configured
    correctly" apart from "configured with a command that can't run"."""
    if not command:
        return None
    exe = command_executable(command)
    return exe if exe and shutil.which(exe) is None else None


def _ensure_starter_running(command: str, popen: Callable[..., Any]) -> None:
    """Make sure the module-level starter process is alive, spawning (or
    respawning, if it exited) under a lock so concurrent callers in the same
    process don't double-spawn."""
    global _starter_proc, _starter_command, _starter_atexit_registered
    with _starter_lock:
        if _starter_proc is not None and _starter_proc.poll() is None:
            return
        log.info("starting SearXNG on demand (LMSA_SEARXNG_START): %s", command_executable(command))
        try:
            _starter_proc = _spawn(command, popen)
        except (OSError, ValueError) as exc:  # missing executable, bad command line
            raise SearchUnavailable(
                f"could not start SearXNG (LMSA_SEARXNG_START: {command_executable(command)}): "
                f"{type(exc).__name__}"
            ) from exc
        _starter_command = command
        if not _starter_atexit_registered:
            atexit.register(_terminate_starter)
            _starter_atexit_registered = True


def _terminate_starter() -> None:
    """atexit hook: terminate the starter process (if any, and still
    running), waiting briefly before escalating to kill."""
    with _starter_lock:
        proc = _starter_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            log.warning("could not stop the SearXNG starter process", exc_info=True)


def _resolve_host(host: str) -> list[str]:
    """Every address ``host`` resolves to, or [] if it doesn't resolve."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    return list({info[4][0] for info in infos})


def _is_local_host(host: str, resolver: Callable[[str], list[str]]) -> bool:
    """True only when every address ``host`` resolves to is loopback,
    private or link-local -- never true for a public hostname, and never
    true when resolution fails (the safe default: robots stays on, and
    nothing is exempted from the SSRF guard)."""
    try:
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addrs = resolver(host)
        except Exception:
            return False
    if not addrs:
        return False
    return all(not is_public_ip(a) for a in addrs)


class SearXNG:
    """Client for one SearXNG instance's JSON API."""

    def __init__(
        self,
        base_url: str,
        *,
        fetcher: Any | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 20,
        start_command: str | None | object = _UNSET,
        start_timeout: float | object = _UNSET,
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        resolve = resolver or _resolve_host
        local = bool(host) and _is_local_host(host, resolve)
        self.fetcher = fetcher if fetcher is not None else Fetcher(
            user_agent,
            timeout=timeout,
            cache_dir=None,  # search results shouldn't be served stale from disk
            allowed_hosts=[host] if local else (),
            respect_robots=not local,  # off only for this one loopback/private host
            resolver=resolver,  # None keeps the Fetcher's own default resolver
        )
        # Sentinel default so every production call site (just
        # SearXNG(settings().searxng_url)) picks up on-demand start for
        # free, while a caller that must never spawn (the Setup probe)
        # passes start_command=None explicitly.
        self.start_command = settings().searxng_start if start_command is _UNSET else start_command
        self.start_timeout = settings().searxng_start_timeout if start_timeout is _UNSET else start_timeout
        self.popen = popen
        self.sleep = sleep
        self.clock = clock

    def search(
        self,
        query: str,
        *,
        categories: str | list[str] | tuple[str, ...] | None = None,
        time_range: str | None = None,
        language: str | None = None,
        max_results: int = 20,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        try:
            return self._search_once(
                query, categories=categories, time_range=time_range, language=language, max_results=max_results
            )
        except SearchUnavailable as exc:
            if not self.start_command or not is_connection_error(exc.__cause__):
                raise
            self._ensure_ready()
            return self._search_once(
                query, categories=categories, time_range=time_range, language=language, max_results=max_results
            )

    def _ensure_ready(self) -> None:
        """Spawn (or confirm) the starter process, then poll this instance
        every 0.5s until it answers or ``start_timeout`` elapses. Each probe
        gets its own short timeout (``min(_PROBE_TIMEOUT_CAP, remaining))``
        rather than the Fetcher's full ``timeout`` -- otherwise one slow
        probe near the end of the budget can overshoot ``start_timeout`` by
        almost its whole fetch timeout."""
        _ensure_starter_running(self.start_command, self.popen)
        started = self.clock()
        while True:
            remaining = self.start_timeout - (self.clock() - started)
            if remaining <= 0:
                raise SearchUnavailable(
                    f"{self.base_url}: did not answer within {self.start_timeout:g}s of starting "
                    f"it (LMSA_SEARXNG_START: {command_executable(self.start_command)})"
                )
            if self._probe_ready(timeout=max(0.001, min(_PROBE_TIMEOUT_CAP, remaining))):
                return
            self.sleep(0.5)

    def _probe_ready(self, *, timeout: float | None = None) -> bool:
        """A cheap search through the same fetcher; only a connection-level
        failure means "not ready yet" -- any other outcome (success, a
        non-JSON body, a rate limit...) means the instance is up. ``timeout``
        bounds this one probe's fetch (see ``_ensure_ready``)."""
        try:
            self._search_once("ping", max_results=1, timeout=timeout)
        except SearchUnavailable as exc:
            return not is_connection_error(exc.__cause__)
        except SearchError:
            pass
        return True

    def _search_once(
        self,
        query: str,
        *,
        categories: str | list[str] | tuple[str, ...] | None = None,
        time_range: str | None = None,
        language: str | None = None,
        max_results: int = 20,
        timeout: float | None = None,
    ) -> list[SearchResult]:
        params = {"q": query, "format": "json"}
        if categories and not isinstance(categories, str):
            categories = ",".join(categories)  # SearXNG takes a comma-separated list
        if categories:
            params["categories"] = categories
        if time_range:
            params["time_range"] = time_range
        if language:
            params["language"] = language
        url = f"{self.base_url}/search?{urllib.parse.urlencode(params)}"
        data = _fetch_json(self.fetcher, url, self.base_url, timeout=timeout)

        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchNotJSON(f"{self.base_url}: unexpected JSON shape from /search (not a SearXNG response?)")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for raw in data["results"]:
            if not isinstance(raw, dict):
                continue
            try:
                item = _parse_result(raw)
            except Exception:
                log.warning("skipping unparseable SearXNG result", exc_info=True)
                continue
            if item is None:
                continue
            try:
                key = normalize_url(item.url)
            except ValueError:
                continue
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
            if len(results) >= max_results:
                break
        return results


def _fetch_json(fetcher: Any, url: str, base_url: str, *, timeout: float | None = None) -> Any:
    try:
        return fetcher.get_json(url) if timeout is None else fetcher.get_json(url, timeout=timeout)
    except HTTPStatusError as exc:
        if exc.status == 429:
            raise SearchRateLimited(f"{base_url}: rate limited (HTTP 429)") from exc
        if exc.status == 403:
            raise SearchNotJSON(
                f"{base_url}: HTTP 403 on format=json -- enable the JSON API "
                f"(`search.formats: [html, json]` in SearXNG's settings.yml) and restart it"
            ) from exc
        raise SearchUnavailable(f"{base_url}: HTTP {exc.status}") from exc
    except InvalidJSON as exc:
        raise SearchNotJSON(
            f"{base_url}: response to format=json wasn't JSON -- this looks like an HTML-only "
            f"instance; enable `search.formats: [html, json]` in settings.yml and restart it"
        ) from exc
    except FetchError as exc:
        raise SearchUnavailable(f"{base_url}: {type(exc).__name__}: {exc}") from exc


def _parse_result(raw: dict) -> SearchResult | None:
    url = raw.get("url")
    if not isinstance(url, str) or not is_http_url(url):
        return None
    title = str(raw.get("title") or "").strip()
    snippet = str(raw.get("content") or "").strip()[:MAX_SNIPPET]
    engines = raw.get("engines")
    engine = raw.get("engine") or (engines[0] if isinstance(engines, list) and engines else "")
    published = None
    pub_raw = raw.get("publishedDate") or raw.get("published_date")
    if isinstance(pub_raw, str):
        published = parse_date(pub_raw)
    return SearchResult(title=title, url=url, snippet=snippet, engine=str(engine), published=published)
