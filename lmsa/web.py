"""Polite, guarded HTTP fetching shared by every crawling task.

URLs reaching this module come from feeds, search results and scraped pages,
so it is untrusted input by default: only http/https, never private /
loopback / link-local / reserved addresses (checked after DNS resolution, on
every redirect hop, and the connection is pinned to the checked address so a
second DNS answer can't swap it), robots.txt honoured, one request per host
every ``per_host_delay`` seconds, bodies capped at ``max_bytes``.

CONTRACT
    Fetcher(user_agent, per_host_delay=2.0, timeout=20, max_bytes=5_000_000,
            cache_dir=DEFAULT_CACHE_DIR, *, allow_private=False, allowed_hosts=(),
            respect_robots=True, max_redirects=5, max_crawl_delay=60.0,
            transport=None, resolver=None, clock=time.monotonic, sleep=time.sleep,
            now=time.time)
        .get(url, *, headers=None, max_age=None, raise_for_status=True, timeout=None) -> Response
            Follows redirects itself (each hop re-checked). With a cached copy of
            a URL it sends If-None-Match / If-Modified-Since; a 304 returns the
            cached body with ``from_cache=True``. ``max_age`` (seconds): serve a
            cached copy younger than that without touching the network.
            IRI input (non-ASCII / spaces) is normalized first: IDNA host,
            percent-encoded path and query, fragment and userinfo dropped.
            Caller ``headers`` go to the first origin only: on a redirect to
            another scheme/host/port (incl. https->http) only Accept and
            Accept-Language survive, so Authorization/Cookie/API keys never leak.
            A robots.txt Crawl-delay over ``max_crawl_delay`` skips the host
            (RobotsDenied) instead of stalling the run. ``timeout`` overrides
            the Fetcher's own ``timeout`` for this one call (e.g. a bounded
            readiness probe); ``None`` keeps the constructor's value.
        .get_json(url, **get_kwargs) -> Any   (InvalidJSON if the body isn't JSON)
    Response(status, url, headers, content, from_cache)
        ``url`` is the final URL after redirects, ``headers`` has lower-case
        keys, ``content`` is bytes; ``.text`` decodes via the Content-Type
        charset (UTF-8 fallback, errors replaced); ``.json()``; ``.ok``.
    extract_article(html: str | bytes, url: str) -> Article(title, text, byline, published, lang)
        trafilatura when it finds text, else a crude tag-stripper. Text is
        untrusted and unbounded: escape it and truncate before a model sees it.

    Exceptions -- all subclass FetchError, so ``except FetchError`` isolates one item:
        BlockedURL       bad scheme, unresolvable host, or non-public address
        RobotsDenied     robots.txt disallows (never fetched)
        TooLarge         body over max_bytes (Content-Length or while reading)
        HTTPStatusError  status >= 400 with raise_for_status (``.status``, ``.url``)
        InvalidJSON      get_json on a non-JSON body (also a ValueError)
        FetchError       transport failures (timeouts, resets, TLS, corrupt gzip,
                         bad redirect, 304 with no cached copy)

    Cache: ``data/web/`` by default (``cache_dir=None`` disables disk caching).
        ``<sha256(url)>.json`` + ``.body`` per 200 response (keyed by the URL
        that answered 200, and also by the URL asked for when it redirected);
        ``robots/<sha256(scheme://host)>.json``
        holds each host's robots.txt for the UTC day it was fetched. Writes are
        atomic (temp + os.replace; body before meta, meta carries the body's
        sha256), and any cache I/O error is logged and treated as a miss.

    Test seams (no network, no sleeping):
        transport(TransportRequest) -> TransportResponse  -- must NOT follow
            redirects; ``.ips`` are the validated addresses (IPv4 first) to try
            in turn, ``.ip`` the first of them.
        resolver(host: str) -> list[str]                   -- IP strings, in preference order
        clock() -> float (monotonic seconds), sleep(seconds), now() -> float (epoch)

    ``allow_private=True`` turns the address check off; ``allowed_hosts`` exempts
    named hosts only (e.g. a LAN SearXNG). Scheme checks always apply.

    Not thread-safe: use one Fetcher per thread. Proxies (HTTP(S)_PROXY) are ignored.
"""

from __future__ import annotations

import hashlib
import html as _html
import http.client
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import time
import urllib.parse
import urllib.robotparser
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "web"
ROBOTS_MAX_BYTES = 512_000
MAX_CRAWL_DELAY = 60.0  # seconds; a robots.txt asking for more skips the host
_REDIRECTS = {301, 302, 303, 307, 308}

log = logging.getLogger(__name__)


# ---- errors -----------------------------------------------------------------

class FetchError(Exception):
    """Base class: anything that stopped a URL from being fetched."""


class BlockedURL(FetchError):
    """Scheme other than http/https, or a host that resolves to a non-public address."""


class RobotsDenied(FetchError):
    """robots.txt disallows this URL for our user agent."""


class TooLarge(FetchError):
    """Response body exceeds the Fetcher's max_bytes."""


class InvalidJSON(FetchError, ValueError):
    """get_json got a body that is not JSON."""


class HTTPStatusError(FetchError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


# ---- data types -------------------------------------------------------------

@dataclass
class Response:
    status: int
    url: str
    headers: dict[str, str]
    content: bytes
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400

    @property
    def encoding(self) -> str:
        m = re.search(r"charset=[\"']?([\w.:-]+)", self.headers.get("content-type", ""), re.I)
        return m.group(1) if m else "utf-8"

    @property
    def text(self) -> str:
        try:
            return self.content.decode(self.encoding, errors="replace")
        except LookupError:  # unknown charset label
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise HTTPStatusError(self.status, self.url)


@dataclass
class TransportRequest:
    url: str  # already normalized: ASCII, percent-encoded, IDNA host
    headers: dict[str, str]
    timeout: float
    max_bytes: int
    ip: str  # first validated address (pinned; the Host header stays the name)
    ips: tuple[str, ...] = ()  # every validated address, in preference order; try each

    def addresses(self) -> tuple[str, ...]:
        return self.ips or (self.ip,)


@dataclass
class TransportResponse:
    status: int
    headers: dict[str, str]  # lower-case keys; Content-Encoding already undone
    body: bytes


@dataclass
class Article:
    title: str
    text: str
    byline: str = ""
    published: datetime | None = None
    lang: str = ""


@dataclass
class _Robots:
    parser: urllib.robotparser.RobotFileParser
    day: str


# ---- default transport: http.client pinned to the validated IPs ---------------

def _connect_any(ips: Iterable[str], port: int, timeout: float) -> socket.socket:
    """Connect to the first reachable validated address (IPv4 is listed first)."""
    last: OSError | None = None
    for ip in ips:
        try:
            return socket.create_connection((ip, port), timeout)
        except OSError as exc:
            last = exc
    raise last or OSError("no address to connect to")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, ips: Iterable[str], timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._ips = tuple(ips)

    def connect(self) -> None:
        self.sock = _connect_any(self._ips, self.port, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, ips: Iterable[str], timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ips = tuple(ips)

    def connect(self) -> None:
        sock = _connect_any(self._ips, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _connection_for(url: str, ips: Iterable[str], timeout: float) -> http.client.HTTPConnection:
    parts = urllib.parse.urlsplit(url)
    https = parts.scheme == "https"
    cls = _PinnedHTTPSConnection if https else _PinnedHTTPConnection
    # Always pass the port explicitly: with port=None http.client re-splits an
    # IPv6 literal host on ':' and connects to the wrong port.
    return cls(parts.hostname, parts.port or (443 if https else 80), ips, timeout)


def _decode_body(data: bytes, encoding: str, max_bytes: int) -> bytes:
    if encoding in ("gzip", "x-gzip", "deflate"):
        wbits = 16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS
        d = zlib.decompressobj(wbits)
        try:
            out = d.decompress(data, max_bytes + 1)
        except zlib.error as exc:
            raise FetchError(f"corrupt {encoding} body: {exc}") from exc
        if len(out) > max_bytes or d.unconsumed_tail:
            raise TooLarge(f"decompressed body over {max_bytes} bytes")
        if not d.eof:
            raise FetchError(f"truncated {encoding} body")
        return out
    return data


def urllib_transport(req: TransportRequest) -> TransportResponse:
    """Real network transport. Does not follow redirects (the Fetcher does)."""
    parts = urllib.parse.urlsplit(req.url)
    conn = _connection_for(req.url, req.addresses(), req.timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        conn.request("GET", path, headers={"Accept-Encoding": "gzip", **req.headers})
        resp = conn.getresponse()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > req.max_bytes:
            raise TooLarge(f"Content-Length {length} over {req.max_bytes} bytes")
        data = resp.read(req.max_bytes + 1)
        if len(data) > req.max_bytes:
            raise TooLarge(f"body over {req.max_bytes} bytes")
        data = _decode_body(data, headers.pop("content-encoding", "").lower().strip(), req.max_bytes)
        headers.pop("content-length", None)
        return TransportResponse(resp.status, headers, data)
    finally:
        conn.close()


def _default_resolver(host: str) -> list[str]:
    """getaddrinfo order, de-duplicated, IPv4 first (many hosts have no working IPv6 path)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    seen: dict[str, int] = {}
    for family, *_rest, sockaddr in infos:
        seen.setdefault(sockaddr[0], family)
    return [a for a, fam in seen.items() if fam == socket.AF_INET] + [
        a for a, fam in seen.items() if fam != socket.AF_INET
    ]


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT = ipaddress.ip_network("::/96")


def is_public_ip(addr: str) -> bool:
    """True for globally routable unicast addresses.

    IPv6 forms that embed an IPv4 address (mapped ::ffff:a.b.c.d, compatible
    ::a.b.c.d, NAT64 64:ff9b::/96, 6to4 2002::/16, Teredo) are judged by the
    embedded IPv4 as well, so none of them can smuggle in 127/8 or 10/8.
    """
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return is_public_ip(str(ip.ipv4_mapped))
        if ip in _NAT64 or ip in _V4_COMPAT:
            return is_public_ip(str(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)))
        embedded = ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None and not is_public_ip(str(embedded)):
            return False
    return ip.is_global and not ip.is_multicast


_PATH_SAFE = "/:@!$&'()*+,;=-._~%"
_QUERY_SAFE = _PATH_SAFE + "?"


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _encode_host(host: str) -> str:
    """Lowercase, IDNA-encode a non-ASCII host. Raises ValueError (UnicodeError
    is a subclass) for a host that can't be encoded."""
    host = host.strip().lower()
    if host and not host.isascii():
        host = host.encode("idna").decode("ascii")
    return host


def normalize_url(url: str) -> str:
    """ASCII request form of a (possibly IRI) URL: IDNA host, percent-encoded
    path/query, userinfo and fragment dropped, explicit default port (:80 for
    http, :443 for https) stripped so it can't split a cache key or robots
    origin from the portless form of the same URL. Existing %XX escapes are
    kept. Raises ValueError for URLs that can't be parsed or encoded."""
    parts = urllib.parse.urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = _encode_host(parts.hostname or "")
    port = parts.port
    if port is not None and port == _DEFAULT_PORTS.get(scheme):
        port = None
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{port}" if port is not None else "")
    path = urllib.parse.quote(parts.path, safe=_PATH_SAFE)
    query = urllib.parse.quote(parts.query, safe=_QUERY_SAFE)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def _origin(parts: urllib.parse.SplitResult) -> tuple[str, str, int]:
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or (443 if scheme == "https" else 80)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """temp file + os.replace, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# Caller headers that are harmless to repeat on another origin; everything
# else (Authorization, Cookie, API-key headers...) stays with the first origin.
_PORTABLE_HEADERS = {"accept", "accept-language"}
_CONDITIONAL_HEADERS = {"if-none-match", "if-modified-since"}


# ---- Fetcher -----------------------------------------------------------------

class Fetcher:
    def __init__(
        self,
        user_agent: str,
        per_host_delay: float = 2.0,
        timeout: float = 20,
        max_bytes: int = 5_000_000,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        *,
        allow_private: bool = False,
        allowed_hosts: Iterable[str] = (),
        respect_robots: bool = True,
        max_redirects: int = 5,
        max_crawl_delay: float = MAX_CRAWL_DELAY,
        transport: Callable[[TransportRequest], TransportResponse] | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ):
        if not user_agent:
            raise ValueError("Fetcher needs a user_agent that identifies the task")
        self.user_agent = user_agent
        self.per_host_delay = per_host_delay
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.allow_private = allow_private
        self.allowed_hosts = {_encode_host(h) for h in allowed_hosts}
        self.respect_robots = respect_robots
        self.max_redirects = max_redirects
        self.max_crawl_delay = max_crawl_delay
        self._transport = transport or urllib_transport
        self._resolver = resolver or _default_resolver
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, _Robots] = {}

    # -- public --

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        max_age: float | None = None,
        raise_for_status: bool = True,
        timeout: float | None = None,
    ) -> Response:
        url = self._normalize(url)
        if max_age is not None:
            cached = self._cache_load(url)
            if cached and self._now() - cached[0].get("fetched_at", 0) <= max_age:
                return cached[1]
        resp = self._follow(
            url, headers or {}, check_robots=self.respect_robots, max_bytes=self.max_bytes, timeout=timeout
        )
        if resp.url != url and resp.status == 200 and not resp.from_cache:
            self._cache_store(url, resp)  # so max_age also works for redirecting URLs
        if raise_for_status:
            resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kwargs: Any) -> Any:
        headers = {"Accept": "application/json", **(kwargs.pop("headers", None) or {})}
        resp = self.get(url, headers=headers, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise InvalidJSON(f"{resp.url}: not JSON ({exc})") from exc

    # -- guards --

    @staticmethod
    def _normalize(url: str) -> str:
        try:
            return normalize_url(url)
        except (ValueError, UnicodeError) as exc:
            raise BlockedURL(f"unparseable URL {url!r}: {exc}") from None

    def _check_url(self, url: str) -> tuple[urllib.parse.SplitResult, list[str]]:
        """Scheme + address check; returns the parsed URL and the validated IPs."""
        try:
            parts = urllib.parse.urlsplit(url)
            parts.port  # noqa: B018 -- raises ValueError on a bad port
        except ValueError as exc:
            raise BlockedURL(f"unparseable URL {url!r}: {exc}") from None
        if parts.scheme.lower() not in ("http", "https"):
            raise BlockedURL(f"scheme {parts.scheme or '(none)'!r} not allowed: {url}")
        host = (parts.hostname or "").lower()
        if not host:
            raise BlockedURL(f"no host in {url}")
        try:
            addrs = list(self._resolver(host))
        except (OSError, UnicodeError) as exc:
            raise BlockedURL(f"cannot resolve {host}: {exc}") from None
        if not addrs:
            raise BlockedURL(f"cannot resolve {host}")
        if not (self.allow_private or host in self.allowed_hosts):
            bad = [a for a in addrs if not is_public_ip(a)]
            if bad:
                raise BlockedURL(f"{host} resolves to non-public address {bad[0]}")
        return parts, addrs

    def _wait_turn(self, host: str, delay: float) -> None:
        last = self._last_hit.get(host)
        if last is not None:
            remaining = last + delay - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_hit[host] = self._clock()

    # -- request loop --

    def _follow(
        self,
        url: str,
        headers: dict[str, str],
        *,
        check_robots: bool,
        max_bytes: int,
        timeout: float | None = None,
    ) -> Response:
        req_timeout = self.timeout if timeout is None else timeout
        first_origin = None
        caller_conditional = any(k.lower() in _CONDITIONAL_HEADERS for k in headers)
        for _ in range(self.max_redirects + 1):
            parts, ips = self._check_url(url)
            origin = _origin(parts)
            if first_origin is None:
                first_origin = origin
            elif origin != first_origin and headers:
                # Cross-origin (incl. https->http downgrade): credentials stay behind.
                headers = {k: v for k, v in headers.items() if k.lower() in _PORTABLE_HEADERS}
                caller_conditional = False
            # Keyed on hostname alone (not scheme/port) so an http->https
            # redirect on the same host still honours the delay it already owes.
            host = origin[1]
            delay = self.per_host_delay
            if check_robots:
                robots = self._robots_for(parts)
                if not robots.parser.can_fetch(self.user_agent, url):
                    raise RobotsDenied(f"robots.txt disallows {url}")
                crawl_delay = robots.parser.crawl_delay(self.user_agent)
                if crawl_delay:
                    if float(crawl_delay) > self.max_crawl_delay:
                        raise RobotsDenied(
                            f"robots.txt Crawl-delay {crawl_delay}s for {origin[1]} is over the "
                            f"{self.max_crawl_delay:g}s cap; host skipped"
                        )
                    delay = max(delay, float(crawl_delay))

            cached = None if caller_conditional else self._cache_load(url)
            req_headers = {"User-Agent": self.user_agent, **headers}
            if cached:
                meta = cached[0]
                if meta.get("etag"):
                    req_headers["If-None-Match"] = meta["etag"]
                if meta.get("last_modified"):
                    req_headers["If-Modified-Since"] = meta["last_modified"]

            self._wait_turn(host, delay)
            try:
                raw = self._transport(
                    TransportRequest(url, req_headers, req_timeout, max_bytes, ips[0], tuple(ips))
                )
            except FetchError:
                raise
            except (OSError, http.client.HTTPException, ValueError, zlib.error) as exc:
                raise FetchError(f"{url}: {exc}") from exc
            if len(raw.body) > max_bytes:
                raise TooLarge(f"body over {max_bytes} bytes: {url}")
            rheaders = {k.lower(): v for k, v in raw.headers.items()}

            if raw.status in _REDIRECTS:
                location = rheaders.get("location")
                if not location:
                    raise FetchError(f"HTTP {raw.status} without Location: {url}")
                try:
                    joined = urllib.parse.urljoin(url, location.strip())
                except ValueError as exc:
                    raise BlockedURL(f"malformed redirect Location {location!r} from {url}: {exc}") from exc
                url = self._normalize(joined)
                continue
            if raw.status == 304:
                if cached:
                    meta, resp = cached
                    self._cache_touch(url, meta)
                    return resp
                if caller_conditional:  # the caller asked a conditional question
                    return Response(304, url, rheaders, b"", from_cache=False)
                raise FetchError(f"HTTP 304 without a cached copy: {url}")
            resp = Response(raw.status, url, rheaders, raw.body, from_cache=False)
            if raw.status == 200 and not caller_conditional:
                self._cache_store(url, resp)
            return resp
        raise FetchError(f"more than {self.max_redirects} redirects: {url}")

    # -- robots --

    def _robots_for(self, parts: urllib.parse.SplitResult) -> _Robots:
        origin = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
        day = datetime.fromtimestamp(self._now(), timezone.utc).date().isoformat()
        hit = self._robots.get(origin)
        if hit and hit.day == day:
            return hit

        path = self.cache_dir / "robots" / f"{_key(origin)}.json" if self.cache_dir else None
        record = None
        if path:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                record = None
            except (OSError, ValueError) as exc:
                log.warning("robots cache unreadable for %s: %s", origin, exc)
                record = None
            if not isinstance(record, dict) or record.get("day") != day:
                record = None
            elif not isinstance(record.get("status"), int) or isinstance(record.get("status"), bool):
                record = None  # corrupt record (e.g. non-numeric status): treat as a miss

        persist = False
        if record is None:
            try:
                resp = self._follow(origin + "/robots.txt", {}, check_robots=False, max_bytes=ROBOTS_MAX_BYTES)
                record = {"day": day, "status": resp.status, "text": resp.text}
                persist = resp.status < 500
            except FetchError as exc:
                # Unreachable robots.txt: stay conservative and skip the host for
                # this run, but don't pin the failure to disk for the whole day.
                record = {"day": day, "status": 599, "text": "", "error": str(exc)}

        parser = urllib.robotparser.RobotFileParser()
        status = int(record.get("status", 599))
        if status in (401, 403) or status >= 500:
            parser.disallow_all = True
        elif status >= 400:
            parser.allow_all = True
        else:
            parser.parse(str(record.get("text", "")).splitlines())
        parser.modified()
        if persist and path:
            try:
                _atomic_write(path, json.dumps(record).encode("utf-8"))
            except OSError as exc:
                log.warning("robots cache write failed for %s: %s", origin, exc)
        robots = _Robots(parser, day)
        self._robots[origin] = robots
        return robots

    # -- HTTP cache (best effort: any I/O problem is a miss, never an error) --

    def _cache_paths(self, url: str) -> tuple[Path, Path] | None:
        if not self.cache_dir:
            return None
        k = _key(url)
        return self.cache_dir / f"{k}.json", self.cache_dir / f"{k}.body"

    def _cache_load(self, url: str) -> tuple[dict, Response] | None:
        paths = self._cache_paths(url)
        if not paths:
            return None
        try:
            meta = json.loads(paths[0].read_text(encoding="utf-8"))
            body = paths[1].read_bytes()
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("web cache unreadable for %s: %s", url, exc)
            return None
        try:
            if meta.get("url") != url or meta.get("sha256") != hashlib.sha256(body).hexdigest():
                return None  # stale pairing (e.g. interrupted write): never validates
            fetched_at = meta.get("fetched_at", 0)
            if not isinstance(fetched_at, (int, float)) or isinstance(fetched_at, bool):
                return None  # corrupt meta field: treat as a miss
            return meta, Response(int(meta["status"]), meta["final_url"], dict(meta["headers"]), body, True)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _write_meta(self, url: str, meta: dict) -> None:
        paths = self._cache_paths(url)
        if paths:
            _atomic_write(paths[0], json.dumps(meta).encode("utf-8"))

    def _cache_store(self, url: str, resp: Response) -> None:
        paths = self._cache_paths(url)
        if not paths:
            return
        meta = {
            "url": url,
            "final_url": resp.url,
            "status": resp.status,
            "headers": resp.headers,
            "etag": resp.headers.get("etag", ""),
            "last_modified": resp.headers.get("last-modified", ""),
            "sha256": hashlib.sha256(resp.content).hexdigest(),
            "fetched_at": self._now(),
        }
        try:
            _atomic_write(paths[1], resp.content)  # body first ...
            self._write_meta(url, meta)  # ... meta last: it is what makes the entry valid
        except OSError as exc:
            log.warning("web cache write failed for %s: %s", url, exc)

    def _cache_touch(self, url: str, meta: dict) -> None:
        try:
            self._write_meta(url, {**meta, "fetched_at": self._now()})
        except OSError as exc:
            log.warning("web cache touch failed for %s: %s", url, exc)


# ---- article extraction --------------------------------------------------------

_DROP_RE = re.compile(r"<(script|style|noscript|template|svg|head)\b.*?</\1\s*>", re.I | re.S)
_BLOCK_RE = re.compile(r"</?(p|div|br|li|h[1-6]|tr|article|section|blockquote)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.I | re.S)


def _strip_tags(markup: str) -> str:
    text = _DROP_RE.sub(" ", markup)
    text = _BLOCK_RE.sub("\n", text)
    text = _html.unescape(_TAG_RE.sub(" ", text))
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def _to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def extract_article(html: str | bytes, url: str) -> Article:
    """Main text + metadata of one page. Never raises on odd markup."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    doc: dict = {}
    try:
        import trafilatura

        result = trafilatura.bare_extraction(html, url=url, with_metadata=True)
        if result is not None:
            doc = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    except ImportError:
        doc = {}
    except Exception:  # trafilatura on hostile markup: fall back, don't kill the run
        doc = {}

    text = (doc.get("text") or "").strip()
    title = (doc.get("title") or "").strip()
    if not text:
        text = _strip_tags(html)
    if not title:
        m = _TITLE_RE.search(html)
        title = _collapse_ws(_html.unescape(m.group(1))) if m else ""
    return Article(
        title=title,
        text=text,
        byline=(doc.get("author") or "").strip(),
        published=_to_datetime(doc.get("date")),
        lang=(doc.get("language") or "").strip(),
    )


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
