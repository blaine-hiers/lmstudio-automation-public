"""`[[sources.pages]]` adapter (#53): site watch inside a briefing.

    [[sources.pages]]
    url = "https://example.com/notices"
    rule = "only if a new notice mentions a service outage or a price change"
    # selector = "main"             # optional: bare tag / #id / .class; default is the full-page article extractor
    # notify = true                  # optional: lmsa.notify alert on a relevant change
    # fields = { price = "number" }  # optional typed fields, extracted + charted each run

Fetch, diff and judge is all done by `lmsa.briefing.sitewatch` (shared with
the standalone `tasks/site_watch.py`, so the same page watched two ways is
judged, and memoized, once -- though each keeps its own snapshot lineage,
identified by `watcher="briefing:<briefing>:<entry name>"`, so one watcher
consuming a diff never starves another watching the same url). This module
only adapts that core to the briefing engine's `Doc` shape: a Doc is
yielded only for a page that actually changed *and* whose diff the model
found relevant to `rule` -- never on the first (baseline) check, never for
an unchanged or irrelevant one.
"""

from __future__ import annotations

from typing import Iterable

from ..sitewatch import PageSpec, check_page
from ..sources import Doc, SourceContext, register

FIELD_TYPES = ("number", "text", "bool")


def _validate_pages(entry: dict) -> None:
    fields = entry.get("fields")
    if fields is not None and (
        not isinstance(fields, dict)
        or not all(isinstance(k, str) and v in FIELD_TYPES for k, v in fields.items())
    ):
        raise ValueError('fields must be a table of name = "number" | "text" | "bool"')
    if "selector" in entry and not isinstance(entry["selector"], str):
        raise ValueError("selector must be a string")
    if "rule" in entry and not isinstance(entry["rule"], str):
        raise ValueError("rule must be a string")
    if "notify" in entry and not isinstance(entry["notify"], bool):
        raise ValueError("notify must be true or false")


@register("pages", keys=("url", "rule", "selector", "fields", "notify"), required=("url",),
          validate=_validate_pages)
def pages_adapter(entry: dict, ctx: SourceContext) -> Iterable[Doc]:
    entry_name = entry.get("name") or entry["url"]
    briefing_name = getattr(ctx.briefing, "name", "") or "?"
    spec = PageSpec(
        url=entry["url"],
        name=entry_name,
        rule=entry.get("rule", ""),
        selector=entry.get("selector"),
        fields=dict(entry.get("fields") or {}),
        notify=bool(entry.get("notify", False)),
        watcher=f"briefing:{briefing_name}:{entry_name}",
    )
    result = check_page(fetcher=ctx.fetcher, lm=ctx.lm, conn=ctx.conn, spec=spec, now=ctx.now, fresh=ctx.fresh)

    if result.error:
        ctx.note(f"pages {spec.name}: {result.error}")
        return
    for note in result.notes:
        ctx.note(f"pages {spec.name}: {note}")
    if result.status != "changed" or not result.relevant:
        return

    if result.should_notify:
        from ...notify import notify

        notify(f"{spec.name}: page changed", result.summary or spec.url, url=spec.url, level="info")

    text = result.summary
    if result.fields:
        text = f"{text}\n" + ", ".join(f"{k}={v}" for k, v in result.fields.items() if v is not None)
    if result.truncated:
        text = f"{text}\n(note: the page text and/or diff was truncated to a length limit)"
    yield Doc(
        url=spec.url,
        title=f"{spec.name}: {result.summary[:120] or 'page changed'}",
        text=text,
        published=ctx.now,
        source=spec.name,
        meta={"kind": "pages", "rule": spec.rule, "fields": result.fields, "truncated": result.truncated},
    )
