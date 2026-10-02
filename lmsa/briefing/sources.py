"""Briefing source adapters and extensions (#52).

An **adapter** turns one `[[sources.<kind>]]` config entry into `Doc`s:

    from lmsa.briefing.sources import Doc, SourceContext, register

    @register("gdelt_doc", keys=("query", "timespan"), required=("query",))
    def gdelt_doc(entry: dict, ctx: SourceContext):
        data = ctx.fetcher.get_json(...)          # polite lmsa.web.Fetcher
        for art in data.get("articles", [])[: ctx.limit]:
            yield Doc(url=art["url"], title=art["title"], text="", published=None,
                      source=entry["name"], meta={"domain": art.get("domain")})

- The built-in kinds are adapters named `feeds`, `search` and `pages`;
  `[[sources.api]] adapter = "<name>"` routes to any other registered name.
  `pages` (site watch, #53) is not built in: until one is registered, a
  `[[sources.pages]]` entry is accepted and the run records a
  "not available" note for it.
- `keys` are the entry keys the adapter accepts on top of
  `config.COMMON_SOURCE_KEYS` (name, lean, reliability, group, limit);
  `required` must be present; `validate(entry)` may raise ValueError for
  anything subtler. Unknown keys are a config error naming the key.
- An adapter may raise anything: the engine catches every exception per
  entry, records it as a note in the report, and keeps the docs yielded
  before the error. It should yield at most `ctx.limit` docs: the engine
  sets it to the entry's cap **plus one** and keeps only the cap, so an
  extra doc is how the report can say "cut at N, more were available".
- Put new adapters in `lmsa/briefing/adapters/<name>.py`: every module
  there is imported by `load_plugins()`, so adding one needs no edit here.

An **extension** hooks the pipeline after per-story extraction (market
odds #60, coverage balance #69, theaters #55):

    register_extension("markets", enrich=fn(stories, ctx), render=fn(story, ctx),
                       render_sources=fn(story, links, ctx),
                       section=fn(report, stories, ctx), default=False)

`enrich` may set `story.extras["markets"] = <JSON-able value>` (persisted
with the story); `render` returns card parts for one story -- any mix of
`lmsa.report.StoryExtra` (a labelled line) and `lmsa.report.StackedBar`
(e.g. coverage by lean); `render_sources` gets the card's source links as
`lmsa.report.StorySource`s (`links[i]` is `story.docs[i]`; a trailing
"+N more" entry has `href=None`) and returns the list to show (e.g. with
reliability badges), or None to leave it; `section` adds report sections
after the stories. `default=True` runs for
every briefing; otherwise only for briefings listing it in
`[briefing] extensions = [...]`. An extension that raises is recorded as a
note and skipped, never fails the run.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


@dataclass
class Doc:
    """One normalized item from any source. `url` is http(s) (others are
    dropped); `text` is untrusted and may be empty (a feed summary, a search
    snippet) -- the engine can follow the link for the article text.
    `published` is tz-aware or None. `source` is the display name. `meta`
    is free-form; the engine adds `lean`, `reliability`, `group`, `kind`
    and `entry` (the config entry's name) from the entry when not set."""
    url: str
    title: str
    text: str = ""
    published: datetime | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class SourceContext:
    """What an adapter gets besides its entry. `fetcher` is the run's shared
    `lmsa.web.Fetcher` (use it for every request); `searxng()` builds the
    SearXNG client lazily; `lm` is the `LMStudio` (for adapters that
    `extract()` -- use `memo` helpers from `lmsa.memo` with `conn`);
    `conn` is the sqlite connection (an adapter may keep its own tables via
    `_NAMED_MIGRATIONS`); `now` is the run's clock; `limit` is the most docs
    to yield for this entry (the kept cap + 1); `note(text)` adds a line to the report's notes."""
    briefing: Any  # BriefingConfig
    fetcher: Any
    lm: Any
    conn: Any
    now: datetime
    limit: int
    searxng_factory: Callable[[], Any] | None = None
    notes: list[str] = field(default_factory=list)
    fresh: bool = False
    # Filled by engine.collect: cfg.entries() index -> {"taken": docs kept from
    # the entry, "error": "<Exc>: msg" or None}. An entry with no key was never
    # called (max_items reached first). Read by extensions (#55 source table).
    entry_stats: dict = field(default_factory=dict)
    _searxng: Any = None

    def note(self, text: str) -> None:
        self.notes.append(str(text)[:300])

    def searxng(self) -> Any:
        if self._searxng is None:
            if self.searxng_factory is None:
                from ..config import settings
                from ..search import SearXNG

                self._searxng = SearXNG(settings().searxng_url)
            else:
                self._searxng = self.searxng_factory()
        return self._searxng


Adapter = Callable[[dict, SourceContext], Iterable[Doc]]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    fn: Adapter
    keys: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    validate: Callable[[dict], None] | None = None


@dataclass(frozen=True)
class Extension:
    name: str
    enrich: Callable[[list, SourceContext], None] | None = None
    render: Callable[[Any, SourceContext], list] | None = None
    section: Callable[[Any, list, SourceContext], None] | None = None
    default: bool = False
    render_sources: Callable[[Any, list, SourceContext], list | None] | None = None


ADAPTERS: dict[str, AdapterSpec] = {}
EXTENSIONS: dict[str, Extension] = {}
PLUGIN_ERRORS: list[str] = []


def register(name: str, *, keys: Iterable[str] = (), required: Iterable[str] = (),
             validate: Callable[[dict], None] | None = None) -> Callable[[Adapter], Adapter]:
    """Decorator: register `fn(entry, ctx) -> Iterable[Doc]` as adapter `name`.
    Re-registering a name replaces it (module reloads in tests)."""
    def deco(fn: Adapter) -> Adapter:
        ADAPTERS[name] = AdapterSpec(name, fn, tuple(keys), tuple(required), validate)
        return fn
    return deco


def register_extension(name: str, *, enrich=None, render=None, section=None, render_sources=None,
                       default: bool = False) -> Extension:
    ext = Extension(name, enrich, render, section, default, render_sources)
    EXTENSIONS[name] = ext
    return ext


def active_extensions(cfg: Any) -> list[Extension]:
    """Default-on extensions plus the ones this briefing lists, in name order."""
    wanted = set(getattr(cfg, "extensions", []) or [])
    return [e for n, e in sorted(EXTENSIONS.items()) if e.default or n in wanted]


_loaded = False


def load_plugins() -> list[str]:
    """Import every module in `lmsa.briefing.adapters` once (they register
    themselves). A module that fails to import is logged and listed in
    `PLUGIN_ERRORS` (shown in the report), never fatal."""
    global _loaded
    if _loaded:
        return PLUGIN_ERRORS
    _loaded = True
    from . import adapters as pkg

    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{info.name}")
        except Exception as exc:  # one broken plugin must not take the engine down
            log.warning("briefing adapter module %s failed to import", info.name, exc_info=True)
            PLUGIN_ERRORS.append(f"adapter module {info.name}: {type(exc).__name__}: {exc}"[:300])
    return PLUGIN_ERRORS


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------

@register("feeds", keys=("url",), required=("url",))
def feeds_adapter(entry: dict, ctx: SourceContext) -> Iterable[Doc]:
    """`[[sources.feeds]] url = "..."`: RSS / Atom / RDF via lmsa.feeds."""
    from ..feeds import fetch_feed

    for item in fetch_feed(ctx.fetcher, entry["url"], source=entry["name"], limit=ctx.limit):
        yield Doc(url=item.url, title=item.title, text=item.summary, published=item.published,
                  source=entry["name"])


_TIME_RANGES = ("day", "week", "month", "year")


def _validate_search(entry: dict) -> None:
    if not isinstance(entry.get("query"), str) or not entry["query"].strip():
        raise ValueError("query must be a non-empty string")
    tr = entry.get("time_range")
    if tr is not None and tr not in _TIME_RANGES:
        raise ValueError(f"time_range must be one of {', '.join(_TIME_RANGES)}")
    cats = entry.get("categories")
    if cats is not None and not (isinstance(cats, list) and all(isinstance(c, str) for c in cats)):
        raise ValueError("categories must be a list of strings")
    if "language" in entry and not isinstance(entry["language"], str):
        raise ValueError("language must be a string")


@register("search", keys=("query", "time_range", "categories", "language"), required=("query",),
          validate=_validate_search)
def search_adapter(entry: dict, ctx: SourceContext) -> Iterable[Doc]:
    """`[[sources.search]] query = "..."`: SearXNG results (snippet as text;
    `follow_links` fetches the article)."""
    results = ctx.searxng().search(
        entry["query"], categories=entry.get("categories"), time_range=entry.get("time_range"),
        language=entry.get("language"), max_results=ctx.limit,
    )
    for r in results:
        yield Doc(url=r.url, title=r.title, text=r.snippet, published=r.published,
                  source=entry.get("name") or r.engine, meta={"engine": r.engine, "query": entry["query"]})
