"""ReliefWeb reports API adapter (#55): humanitarian situation reports.

    [[sources.api]]
    adapter = "reliefweb"
    name = "ReliefWeb"
    query = "conflict OR violence OR displacement"   # optional full-text query
    days = 2                                         # optional look-back (1-30, default 2)

Calls `api.reliefweb.int/v1/reports` (free, no key; the API asks every
caller to name itself with `appname`). Each report becomes a `Doc` whose
text is the report body; the report's countries (ISO alpha-3 in the API)
are kept as alpha-2 in `meta["countries"]`, primary country first -- the
theaters extension uses them when the model tags no country.

A 200 answer that is not the documented shape raises `UnrecognizedResponse`
(the engine shows it as a note), so a changed API is reported instead of
looking like a quiet day; `{"data": []}` is a real "no reports".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlencode

from ... import geo
from ...feeds import is_http_url
from ..sources import Doc, SourceContext, register

ENDPOINT = "https://api.reliefweb.int/v1/reports"
APPNAME = "lmstudio-automation"
MAX_LIMIT = 200          # API page size cap we ask for
DEFAULT_DAYS = 2
FIELDS = ("title", "url", "url_alias", "date.created", "date.original", "source.name",
          "source.shortname", "country.iso3", "primary_country.iso3", "body")


class UnrecognizedResponse(ValueError):
    """A ReliefWeb answer that is not a report list we understand."""


def _validate(entry: dict) -> None:
    q = entry.get("query")
    if q is not None and (not isinstance(q, str) or not q.strip()):
        raise ValueError("query must be a non-empty string")
    days = entry.get("days", DEFAULT_DAYS)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
        raise ValueError("days must be an integer from 1 to 30")


def request_url(entry: dict, now: datetime, limit: int) -> str:
    since = now - timedelta(days=int(entry.get("days", DEFAULT_DAYS)))
    params: list[tuple[str, str]] = [
        ("appname", APPNAME),
        ("limit", str(max(1, min(limit, MAX_LIMIT)))),
        ("sort[]", "date.created:desc"),
        ("filter[field]", "date.created"),
        ("filter[value][from]", since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")),
    ]
    params += [("fields[include][]", f) for f in FIELDS]
    if entry.get("query"):
        params.append(("query[value]", entry["query"]))
    return f"{ENDPOINT}?{urlencode(params)}"


def _alpha3_map() -> dict[str, str]:
    return {c.alpha3: c.alpha2 for c in geo.countries() if c.alpha3}


def _date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _countries(fields: dict, a3: dict[str, str]) -> list[str]:
    out: list[str] = []
    primary = fields.get("primary_country")
    raw = [primary] if isinstance(primary, dict) else []
    raw += [c for c in fields.get("country") or [] if isinstance(c, dict)] if isinstance(fields.get("country"), list) else []
    for c in raw:
        code = a3.get(str(c.get("iso3") or "").upper())
        if code and code not in out:
            out.append(code)
    return out


def parse_reports(data: object) -> list[Doc]:
    """The API's JSON -> Docs. `{"data": []}` -> []; anything else that
    yields no usable report raises UnrecognizedResponse."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise UnrecognizedResponse("unrecognized ReliefWeb response (no 'data' list)")
    items = data["data"]
    a3 = _alpha3_map()
    docs: list[Doc] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("fields"), dict):
            continue
        f = item["fields"]
        title = " ".join(str(f.get("title") or "").split())
        url = next((u for u in (f.get("url_alias"), f.get("url")) if isinstance(u, str) and is_http_url(u)), None)
        if url is None and str(item.get("id") or "").isdigit():
            url = f"https://reliefweb.int/node/{item['id']}"
        if not title or not url:
            continue
        date = f.get("date") if isinstance(f.get("date"), dict) else {}
        srcs = [s for s in f.get("source") or [] if isinstance(s, dict)] if isinstance(f.get("source"), list) else []
        publisher = ", ".join(str(s.get("shortname") or s.get("name") or "") for s in srcs[:3] if s)
        docs.append(Doc(
            url=url, title=title, text=str(f.get("body") or ""),
            published=_date(date.get("created")) or _date(date.get("original")),
            source=f"ReliefWeb: {publisher}" if publisher else "ReliefWeb",
            meta={"countries": _countries(f, a3), "publisher": publisher},
        ))
    if items and not docs:
        raise UnrecognizedResponse(f"unrecognized ReliefWeb response ({len(items)} items, none usable)")
    return docs


@register("reliefweb", keys=("query", "days"), validate=_validate)
def reliefweb(entry: dict, ctx: SourceContext) -> Iterable[Doc]:
    data = ctx.fetcher.get_json(request_url(entry, ctx.now, ctx.limit))
    docs = parse_reports(data)
    if not docs:
        ctx.note(f"api: {entry.get('name', 'reliefweb')}: ReliefWeb returned no reports for the last "
                 f"{entry.get('days', DEFAULT_DAYS)} day(s)")
    yield from docs[:ctx.limit]
