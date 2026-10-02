"""GDELT DOC 2.0 article-list adapter (#55).

    [[sources.api]]
    adapter = "gdelt_doc"
    name = "GDELT: armed conflict"
    query = "theme:ARMEDCONFLICT sourcelang:english"
    timespan = "24h"        # optional: 15min .. 3m (GDELT's own syntax), default 24h

One request per entry to `api.gdeltproject.org/api/v2/doc/doc` with
`mode=artlist&format=json` (free, no key). GDELT throttles hard (HTTP 429,
see #56), so:

- requests to its host are spaced at least `MIN_DELAY` (5 s) apart by
  raising the shared Fetcher's `per_host_delay` for the duration of the call
  (never lowering it);
- at most `MAX_QUERIES_PER_RUN` GDELT entries run per briefing run -- later
  ones are skipped with a note;
- after a 429 the remaining GDELT entries of the run are skipped (noted).

Parsing is defensive: `{}`, an empty body or `{"articles": []}` is a real
"no articles" (noted as such); a non-JSON body, another shape, or a
non-empty article list with nothing usable raises `UnrecognizedResponse`,
so a changed or error response is never mistaken for a quiet day.
Articles carry a title only; `follow_links` fetches their text.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlencode

from ...feeds import is_http_url
from ...web import HTTPStatusError
from ..sources import Doc, SourceContext, register

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_DELAY = 5.0              # seconds between requests to GDELT (its documented limit)
MAX_QUERIES_PER_RUN = 4
MAX_RECORDS = 250            # GDELT's artlist maximum
DEFAULT_TIMESPAN = "24h"
_TIMESPAN_RE = re.compile(r"^\d{1,4}(min|h|hours|d|days|w|weeks|m|months)$")
_STATE_ATTR = "_gdelt_state"


class UnrecognizedResponse(ValueError):
    """A GDELT answer that is not an article list we understand."""


def _validate(entry: dict) -> None:
    if not isinstance(entry.get("query"), str) or not entry["query"].strip():
        raise ValueError("query must be a non-empty string")
    ts = entry.get("timespan", DEFAULT_TIMESPAN)
    if not isinstance(ts, str) or not _TIMESPAN_RE.match(ts):
        raise ValueError("timespan must look like 15min, 24h, 3d, 1w or 3m")


def request_url(entry: dict, limit: int) -> str:
    params = {
        "query": entry["query"],
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max(1, min(limit, MAX_RECORDS))),
        "timespan": entry.get("timespan", DEFAULT_TIMESPAN),
        "sort": "datedesc",
    }
    return f"{ENDPOINT}?{urlencode(params)}"


def _seendate(value: object) -> datetime | None:
    try:
        return datetime.strptime(str(value).strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_artlist(body: bytes | str) -> list[Doc]:
    """A DOC API artlist body -> Docs ([] only for GDELT's "no matches")."""
    text = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body or "")
    text = text.strip().lstrip("﻿")
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        raise UnrecognizedResponse(f"unrecognized GDELT response (not JSON): {text[:120]!r}") from None
    if not isinstance(data, dict):
        raise UnrecognizedResponse("unrecognized GDELT response (not a JSON object)")
    if not data:
        return []
    arts = data.get("articles")
    if not isinstance(arts, list):
        raise UnrecognizedResponse("unrecognized GDELT response (no 'articles' list)")
    docs: list[Doc] = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        url = a.get("url")
        title = " ".join(str(a.get("title") or "").split())
        if not isinstance(url, str) or not is_http_url(url) or not title:
            continue
        domain = str(a.get("domain") or "")[:120]
        docs.append(Doc(url=url, title=title, text="", published=_seendate(a.get("seendate")),
                        source=domain or "GDELT",
                        meta={"domain": domain, "language": str(a.get("language") or "")[:40],
                              "source_country": str(a.get("sourcecountry") or "")[:80], "via": "gdelt"}))
    if arts and not docs:
        raise UnrecognizedResponse(f"unrecognized GDELT response ({len(arts)} articles, none usable)")
    return docs


def _state(ctx: SourceContext) -> dict:
    state = getattr(ctx, _STATE_ATTR, None)
    if state is None:
        state = {"queries": 0, "throttled": False}
        setattr(ctx, _STATE_ATTR, state)
    return state


@register("gdelt_doc", keys=("query", "timespan"), required=("query",), validate=_validate)
def gdelt_doc(entry: dict, ctx: SourceContext) -> Iterable[Doc]:
    label = f"api: {entry.get('name', 'gdelt_doc')}"
    state = _state(ctx)
    if state["throttled"]:
        ctx.note(f"{label}: skipped -- GDELT throttled an earlier query this run (HTTP 429)")
        return
    if state["queries"] >= MAX_QUERIES_PER_RUN:
        ctx.note(f"{label}: skipped -- at most {MAX_QUERIES_PER_RUN} GDELT queries per run")
        return
    state["queries"] += 1
    fetcher = ctx.fetcher
    old = getattr(fetcher, "per_host_delay", None)
    try:
        if old is None or old < MIN_DELAY:
            fetcher.per_host_delay = MIN_DELAY  # GDELT's 1 request / 5 s; never lowered
        resp = fetcher.get(request_url(entry, ctx.limit))
    except HTTPStatusError as exc:
        if exc.status == 429:
            state["throttled"] = True
            raise RuntimeError("GDELT throttled this query (HTTP 429); later GDELT queries this run skipped") from None
        raise
    finally:
        if old is not None:
            fetcher.per_host_delay = old
    docs = parse_artlist(resp.content)
    if not docs:
        ctx.note(f"{label}: GDELT returned no articles for this query ({entry.get('timespan', DEFAULT_TIMESPAN)})")
    yield from docs[:ctx.limit]
