"""Watch pages in `config/site_watch.toml` for changes an LLM judges
against a plain-English rule (#53) -- changedetection.io's pattern, but
judged and summarized instead of dumped as a raw diff.

Each page's own `rule` decides what's worth reporting ("only if the price
drops below $50", "ignore nav/footer and date changes"); the model reads a
unified diff against the page's last snapshot and says whether it's
relevant, gives a one-sentence summary, and (optionally) extracts typed
`fields` charted as a sparkline over time. A relevant change alerts via
`lmsa.notify` for pages with `notify = true` -- never on a page's first
(baseline) check, since there's nothing yet to compare against.

The fetch/diff/judge core (`lmsa.briefing.sitewatch`) is shared with the
`pages` briefing adapter, so a page watched both ways is judged once.

    python -m tasks.site_watch
"""

from __future__ import annotations

import tomllib
from datetime import datetime, timezone
from pathlib import Path

from lmsa import LMStudio, Report, Store, Tile
from lmsa.briefing.sitewatch import FIELD_TYPES, PageSpec, check_page, field_history
from lmsa.notify import notify
from lmsa.web import Fetcher

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/site_watch.toml"], "model": "chat"}

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "site_watch.toml"

USER_AGENT = "Mozilla/5.0 (compatible; lmsa-site-watch/1.0)"
HISTORY_POINTS = 30


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


def load_pages(path: Path) -> list[PageSpec]:
    """Parse `config/site_watch.toml`. Missing file -> no pages, not an
    error. Each `[[pages]]` entry is validated independently: a bad
    `fields` table or a duplicate url marks *that* entry with `load_error`
    (surfaced as its own check error) rather than aborting the file --
    same shape as `tasks.price_watch.load_watchlist`."""
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    specs: list[PageSpec] = []
    seen_urls: set[str] = set()
    for entry in data.get("pages", []):
        url = entry.get("url")
        if not url:
            continue
        error = None
        fields = entry.get("fields") or {}
        if not isinstance(fields, dict) or not all(
            isinstance(k, str) and v in FIELD_TYPES for k, v in fields.items()
        ):
            error = f'invalid fields for {url!r}: expected name = "number" | "text" | "bool"'
            fields = {}
        if url in seen_urls:
            error = error or f"duplicate url in site_watch: {url!r} (each page must have a unique url)"
        seen_urls.add(url)
        name = entry.get("name") or url
        specs.append(
            PageSpec(
                url=url,
                name=name,
                rule=str(entry.get("rule") or ""),
                selector=entry.get("selector"),
                fields=fields,
                notify=bool(entry.get("notify", False)),
                watcher=f"task:site_watch:{name}",
                load_error=error,
            )
        )
    return specs


# -- report --------------------------------------------------------------------


def build_report(results: list[tuple[PageSpec, "object"]]) -> Path:
    errors = [(s, r) for s, r in results if r.error]
    changed = [(s, r) for s, r in results if r.status == "changed"]
    relevant = [(s, r) for s, r in results if r.relevant]
    alerts = [(s, r) for s, r in results if r.should_notify]

    rep = Report(
        "Site watch",
        kicker="lmsa · site watch",
        subtitle=f"{len(results)} page(s) watched.",
        sticker=f"{len(alerts)} alert(s)" if alerts else "all quiet",
        sticker_tone="hot" if alerts else "acc",
        source="config/site_watch.toml",
    )
    rep.tiles(
        [
            Tile("Pages watched", len(results), tone="info"),
            Tile("Changed this run", len(changed), tone="warn" if changed else "muted"),
            Tile("Relevant changes", len(relevant), tone="hot" if relevant else "muted"),
            Tile("Fetch/extract errors", len(errors), tone="warn" if errors else "muted"),
        ]
    )

    if relevant:
        sec = rep.section("Relevant changes", tone="hot")
        sec.table(
            ["Page", "Summary", "Fields", "URL"],
            [
                [
                    s.name,
                    r.summary,
                    ", ".join(f"{k}={v}" for k, v in r.fields.items() if v is not None) or "—",
                    s.url,
                ]
                for s, r in relevant
            ],
        )

    watch = rep.section("Watched pages")
    watch.table(
        ["Page", "Status", "Relevant", "Summary", "Truncated", "URL"],
        [
            [
                s.name,
                r.status,
                "yes" if r.relevant else "no" if r.relevant is False else "—",
                r.summary or (f"error: {r.error}" if r.error else "—"),
                "yes" if r.truncated else "—",
                s.url,
            ]
            for s, r in results
        ],
    )

    notes = [(s, n) for s, r in results for n in r.notes]
    if notes:
        rep.section("Notes", tone="warn").callout(
            "\n".join(f"{s.name}: {n}" for s, n in notes),
            label="disclosed",
            tone="warn",
        )

    for s, r in results:
        numeric_fields = [name for name in s.fields if r.history.get(name)]
        if numeric_fields:
            rep.section(f"{s.name}: fields").tiles(
                [
                    Tile(
                        label=name,
                        value=r.fields.get(name) if r.fields.get(name) is not None else "—",
                        spark=r.history.get(name, []),
                        tone="info",
                    )
                    for name in numeric_fields
                ]
            )

    if errors:
        rep.section("Errors", tone="warn").callout(
            "\n".join(f"{s.name}: {r.error}" for s, r in errors),
            label="fetch/extract errors",
            tone="warn",
        )

    return rep.write()


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    fetcher = Fetcher(USER_AGENT)
    pages = load_pages(CONFIG_PATH)

    with Store() as store:
        with store.record_run("tasks.site_watch") as run:
            results = []
            for spec in pages:
                res = check_page(fetcher=fetcher, lm=lm, conn=store.conn, spec=spec, now=now)
                if not res.error:
                    res.history = {
                        name: field_history(store.conn, spec.watcher, spec.url, name, limit=HISTORY_POINTS)
                        for name in spec.fields
                    }
                results.append((spec, res))

            alerts = [(s, r) for s, r in results if r.should_notify]
            for s, r in alerts:
                notify(f"{s.name}: page changed", r.summary or s.url, url=s.url, level="info")

            path = build_report(results)
            run.report_path = path
            run.summary = {
                "pages": len(pages),
                "changed": sum(1 for _, r in results if r.status == "changed"),
                "relevant": sum(1 for _, r in results if r.relevant),
                "alerts": len(alerts),
                "errors": sum(1 for _, r in results if r.error),
            }
            print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
