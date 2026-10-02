"""Watch product pages in `config/watchlist.toml` for a price or stock change.

Fetches each page once, strips it to text, and asks the model to extract
`{price, currency, in_stock}`. History is kept in the store so each run can
report a delta vs the last check and a 30-day sparkline. The report always
shows *current* state (a hot tile/column/callout for every item at or under
its target, every run); notify() alerts only on a *transition* -- a target
price newly hit, or an item newly back in stock -- except that an item's
very first check alerts immediately if it's already under target, since
there's no earlier run for that to have been a transition from.

    python -m tasks.price_watch
"""

from __future__ import annotations

import html
import re
import sqlite3
import tomllib
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lmsa import LMStudio, Report, Store, Tile
from lmsa.notify import notify
from lmsa.prompts import load as load_prompt

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/watchlist.toml"], "model": "chat"}

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = ROOT / "config" / "watchlist.toml"

USER_AGENT = "Mozilla/5.0 (compatible; lmsa-price-watch/1.0)"
FETCH_TIMEOUT = 15
MAX_PAGE_CHARS = 4000
HISTORY_DAYS = 30

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _now() -> datetime:
    """Return current UTC time. Tests can monkeypatch this."""
    return datetime.now(timezone.utc)


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "price": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "in_stock": {"type": ["boolean", "null"]},
    },
    "required": ["price", "currency", "in_stock"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = load_prompt("price_watch/extract_system")


@dataclass
class Item:
    name: str
    url: str
    target_price: float | None = None
    # Set at load time when the entry itself is unusable (bad target_price,
    # duplicate url). check_item() turns this straight into that item's
    # per-run error instead of ever reaching fetch/extract.
    load_error: str | None = None


def _coerce_target_price(raw: object) -> tuple[float | None, str | None]:
    """Accept a TOML number or a numeric string; reject anything else.

    Returns (target_price, error). `bool` is rejected explicitly even
    though it's a subclass of `int` in Python -- `target_price = true` is
    almost certainly a typo for a number, not a real target.
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, f"invalid target_price {raw!r}: expected a number"
    if isinstance(raw, (int, float)):
        return float(raw), None
    if isinstance(raw, str):
        try:
            return float(raw.strip()), None
        except ValueError:
            return None, f"invalid target_price {raw!r}: not a number"
    return None, f"invalid target_price {raw!r}: expected a number"


def load_watchlist(path: Path) -> list[Item]:
    """Parse `config/watchlist.toml`. Missing file -> empty list, not an error.

    Each entry is validated independently: a bad `target_price` or a
    duplicate `url` marks *that* item with `load_error` rather than raising,
    so one bad row in the file doesn't take the rest of the watchlist down.
    A duplicate url is rejected outright (rather than merged) because
    `last_check()` looks history up by url -- two live items sharing one
    url would silently blend each other's price/stock history.
    """
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    items: list[Item] = []
    seen_urls: set[str] = set()
    for entry in data.get("item", []):
        url = entry.get("url")
        if not url:
            continue
        target_price, error = _coerce_target_price(entry.get("target_price"))
        if url in seen_urls:
            error = error or f"duplicate url in watchlist: {url!r} (each item must have a unique url)"
        seen_urls.add(url)
        items.append(
            Item(
                name=entry.get("name") or url,
                url=url,
                target_price=target_price,
                load_error=error,
            )
        )
    return items


def fetch_page(url: str) -> str:
    """One polite GET: real UA, bounded timeout, no retry loop."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read()
    return raw.decode(charset, errors="replace")


def strip_to_text(page_html: str) -> str:
    """Drop scripts/styles/markup, unescape entities, collapse whitespace."""
    no_script = _SCRIPT_STYLE.sub(" ", page_html)
    no_tags = _TAG.sub(" ", no_script)
    return _WS.sub(" ", html.unescape(no_tags)).strip()


def truncate(text: str, limit: int = MAX_PAGE_CHARS) -> str:
    return text[:limit]


def extract_listing(lm: LMStudio, page_text: str) -> dict:
    return lm.extract(page_text, EXTRACT_SCHEMA, system=EXTRACT_SYSTEM, name="listing")


# -- store -------------------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_watch_history (
            url TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            name TEXT NOT NULL,
            price REAL,
            currency TEXT,
            in_stock INTEGER,
            error TEXT
        )
        """
    )
    conn.commit()


def last_check(conn: sqlite3.Connection, url: str) -> dict | None:
    # `rowid` breaks ties when two checks land in the same instant (as in
    # tests, and possible in practice given timestamp resolution) so "last"
    # is always insertion order, not an arbitrary tied row.
    row = conn.execute(
        "SELECT price, currency, in_stock FROM price_watch_history "
        "WHERE url = ? AND error IS NULL ORDER BY checked_at DESC, rowid DESC LIMIT 1",
        (url,),
    ).fetchone()
    if row is None:
        return None
    return {"price": row[0], "currency": row[1], "in_stock": bool(row[2]) if row[2] is not None else None}


def record_check(
    conn: sqlite3.Connection,
    *,
    url: str,
    name: str,
    checked_at: str,
    price: float | None,
    currency: str | None,
    in_stock: bool | None,
    error: str | None,
) -> None:
    conn.execute(
        "INSERT INTO price_watch_history (url, checked_at, name, price, currency, in_stock, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, checked_at, name, price, currency, None if in_stock is None else int(in_stock), error),
    )
    conn.commit()


def _currency_changed(previous_currency: str | None, currency: str | None) -> bool:
    """True only when both sides are known and actually differ.

    A `None` on either side isn't a *known* change -- extraction sometimes
    can't read the currency at all -- so it doesn't block a delta; only an
    unambiguous "$" -> "GBP" kind of switch does.
    """
    if previous_currency is None or currency is None:
        return False
    return previous_currency.strip() != currency.strip()


def price_history(conn: sqlite3.Connection, url: str, since: str) -> list[float]:
    rows = conn.execute(
        "SELECT price FROM price_watch_history "
        "WHERE url = ? AND error IS NULL AND price IS NOT NULL AND checked_at >= ? "
        "ORDER BY checked_at ASC, rowid ASC",
        (url, since),
    ).fetchall()
    return [r[0] for r in rows]


# -- per-item check ------------------------------------------------------------


def check_item(lm: LMStudio, conn: sqlite3.Connection, item: Item, now: datetime) -> dict:
    """Fetch + extract one item, isolating its own failure, and record history.

    `under_target` is *current state*, recomputed every run: true whenever
    today's price is at or under the target, regardless of history. It
    drives the report (hot tile/column, persistent callout) every time.

    `hit_target` / `back_in_stock` are the alert triggers and stay
    transition-only, to keep notify() from firing every run:
    - `back_in_stock` needs a prior (error-free, comparable) check that
      wasn't in stock. A first-ever check just establishes the baseline and
      never alerts on stock.
    - `hit_target` is the one deliberate exception: on an item's first-ever
      check, if it's already under target, it DOES alert once -- the owner
      just added it specifically to be told, and there's no earlier "not
      yet under target" run to have already covered that. The same applies
      whenever the previous check's currency doesn't match this run's: an
      incomparable currency means there's no reliable "already hit" basis,
      so it's treated like a first check rather than risk silently missing
      a real hit.
    - a currency change also blocks the price delta outright (comparing
      30.0 "USD" to 30.0 "GBP" would show a delta of 0 that isn't real).

    Any failure -- the network fetch, the model call, or the comparison
    logic below it (e.g. a bad watchlist entry) -- is caught in one place
    so one item's problem is recorded as *that* item's error and never
    aborts the run for the rest of the watchlist.
    """
    checked_at = now.isoformat()
    result = {
        "name": item.name,
        "url": item.url,
        "target_price": item.target_price,
        "price": None,
        "currency": None,
        "in_stock": None,
        "delta": None,
        "currency_changed": False,
        "under_target": False,
        "hit_target": False,
        "back_in_stock": False,
        "error": None,
    }
    previous = last_check(conn, item.url)
    price = currency = in_stock = None

    try:
        if item.load_error:
            raise ValueError(item.load_error)

        page = fetch_page(item.url)
        text = truncate(strip_to_text(page))
        listing = extract_listing(lm, text)
        price = listing.get("price")
        currency = listing.get("currency")
        in_stock = listing.get("in_stock")
        result.update(price=price, currency=currency, in_stock=in_stock)

        changed = previous is not None and _currency_changed(previous["currency"], currency)

        if previous and previous["price"] is not None and price is not None:
            if changed:
                result["currency_changed"] = True
            else:
                result["delta"] = price - previous["price"]

        if item.target_price is not None and price is not None:
            result["under_target"] = price <= item.target_price
            if result["under_target"]:
                previously_hit = (
                    previous is not None
                    and not changed
                    and previous["price"] is not None
                    and previous["price"] <= item.target_price
                )
                result["hit_target"] = not previously_hit

        if in_stock is True:
            previously_in_stock = previous is not None and previous["in_stock"] is True
            result["back_in_stock"] = previous is not None and not previously_in_stock
    except Exception as exc:
        # Isolation, not laziness: this spans a network fetch, a model
        # call, and the comparison logic above, each with its own failure
        # modes (URLError/OSError/timeout, openai API errors, bad JSON from
        # the model, a malformed watchlist entry). Per the review lesson in
        # this repo, one item's failure must never abort the whole run --
        # so every field derived from this check resets to "no result"
        # rather than possibly carrying a partial computation.
        result["error"] = f"{type(exc).__name__}: {exc}"
        result.update(
            price=None, currency=None, in_stock=None, delta=None,
            currency_changed=False, under_target=False, hit_target=False, back_in_stock=False,
        )
        record_check(
            conn, url=item.url, name=item.name, checked_at=checked_at,
            price=None, currency=None, in_stock=None, error=result["error"],
        )
        return result

    record_check(
        conn, url=item.url, name=item.name, checked_at=checked_at,
        price=price, currency=currency, in_stock=in_stock, error=None,
    )
    return result


# -- report --------------------------------------------------------------------


def _fmt_price(price: float | None, currency: str | None) -> str:
    if price is None:
        return "unknown"
    return f"{currency or ''} {price:,.2f}".strip()


def _fmt_delta(delta: float | None, currency: str | None) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:,.2f} {currency or ''}".strip()


def _stock_note(item: dict) -> str:
    if item["error"]:
        return "error"
    if item["in_stock"] is True:
        return "in stock"
    if item["in_stock"] is False:
        return "out of stock"
    return "stock unknown"


def build_report(results: list[dict], hits: list[dict], backs: list[dict]) -> Path:
    errors = [r for r in results if r["error"]]
    under_target_now = [r for r in results if r["under_target"]]
    alerts = len(hits) + len(backs)

    r = Report(
        "Price & stock watcher",
        kicker="lmsa · watchlist",
        subtitle=f"{len(results)} item(s) tracked.",
        sticker=f"{alerts} alert(s)" if alerts else "all quiet",
        sticker_tone="hot" if alerts else "acc",
        source="config/watchlist.toml",
    )
    r.tiles(
        [
            Tile("Items tracked", len(results), tone="info"),
            Tile("Under target now", len(under_target_now), tone="hot" if under_target_now else "muted"),
            Tile("Back in stock now", sum(1 for i in results if i["in_stock"] is True), tone="info"),
            Tile("Fetch/extract errors", len(errors), tone="warn" if errors else "muted"),
        ]
    )

    # New this run: what actually just transitioned (drives notify()).
    if hits or backs:
        lines = [f"{i['name']}: target price hit ({_fmt_price(i['price'], i['currency'])})" for i in hits]
        lines += [f"{i['name']}: back in stock" for i in backs]
        r.section("New this run", tone="hot").callout("\n".join(lines), label="transition", tone="hot")

    # Current state: every item still at/under target, whether or not it
    # transitioned this run -- shows up every run the condition holds, not
    # just the run it first became true.
    if under_target_now:
        lines = [
            f"{i['name']}: {_fmt_price(i['price'], i['currency'])} (target {_fmt_price(i['target_price'], i['currency'])})"
            for i in under_target_now
        ]
        r.section("At or under target", tone="hot").callout("\n".join(lines), label="under target", tone="hot")

    watch = r.section("Watchlist")
    watch.tiles(
        [
            Tile(
                label=item["name"],
                value=_fmt_price(item["price"], item["currency"]),
                delta=(
                    "currency changed" if item["currency_changed"]
                    else _fmt_delta(item["delta"], item["currency"]) if item["delta"] is not None
                    else None
                ),
                tone="hot" if (item["under_target"] or item["back_in_stock"]) else "warn" if item["error"] else "acc",
                note=_stock_note(item),
                spark=item["history"],
            )
            for item in results
        ]
    )
    watch.table(
        ["Item", "Price", "Target", "Under target", "In stock", "URL"],
        [
            [
                item["name"],
                _fmt_price(item["price"], item["currency"]),
                _fmt_price(item["target_price"], item["currency"]) if item["target_price"] is not None else "—",
                (
                    "yes" if item["under_target"]
                    else "no" if item["price"] is not None and item["target_price"] is not None
                    else "—"
                ),
                "yes" if item["in_stock"] is True else "no" if item["in_stock"] is False else "—",
                item["url"],
            ]
            for item in results
        ],
    )

    if errors:
        r.section("Errors", tone="warn").callout(
            "\n".join(f"{e['name']}: {e['error']}" for e in errors),
            label="fetch/extract errors",
            tone="warn",
        )

    return r.write()


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    since = (now - timedelta(days=HISTORY_DAYS)).isoformat()

    items = load_watchlist(WATCHLIST_PATH)

    with Store() as store:
        ensure_schema(store.conn)
        with store.record_run("tasks.price_watch") as run:
            results = []
            for item in items:
                res = check_item(lm, store.conn, item, now)
                res["history"] = [] if res["error"] else price_history(store.conn, item.url, since)
                results.append(res)

            hits = [r for r in results if r["hit_target"]]
            backs = [r for r in results if r["back_in_stock"]]

            for item in hits:
                notify(
                    f"{item['name']}: target price hit",
                    f"{_fmt_price(item['price'], item['currency'])} (target {_fmt_price(item['target_price'], item['currency'])})",
                    url=item["url"],
                    level="urgent",
                )
            for item in backs:
                notify(f"{item['name']}: back in stock", item["url"], url=item["url"], level="urgent")

            path = build_report(results, hits, backs)
            run.report_path = path
            run.summary = {
                "items": len(items),
                "under_target_now": sum(1 for r in results if r["under_target"]),
                "target_hits": len(hits),
                "back_in_stock": len(backs),
                "errors": sum(1 for r in results if r["error"]),
            }
            print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
