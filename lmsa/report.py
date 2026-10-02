"""Build standalone HTML reports in the Afterhours house style.

Every task that emits HTML goes through this module, so all outputs share one
look. The stylesheet is inlined at render time: a report is a single file you
can open, archive, or email without anything beside it.

    r = Report("Inbox digest", kicker="Daily · Gmail", sticker="42 unread")
    r.tiles([Tile("Unread", 42, delta="+5", tone="hot")])
    r.section("Top senders").bars([("alice@x.com", 12), ("bob@y.com", 7)])
    path = r.write()
"""

from __future__ import annotations

import base64
import hashlib
import html
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import settings

THEME = Path(__file__).parent / "theme" / "afterhours.css"
TONES = ("acc", "hot", "warn", "info", "violet", "muted")
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Barlow:wght@400;500;600&family=Bebas+Neue"
    '&family=JetBrains+Mono:wght@400;600;700&display=swap">'
)


def esc(value: object) -> str:
    return html.escape(str(value))


_SAFE_SCHEMES = ("http", "https", "mailto")


def safe_href(url: str | None) -> str | None:
    """Return `url` if it is safe to put in an href, else None.

    Hrefs often come from untrusted places (email, feeds, scraped pages).
    Escaping stops markup injection but not `javascript:` / `data:` URLs,
    so only http(s), mailto and scheme-less relative links pass.
    """
    if not url:
        return None
    url = url.strip()
    scheme, sep, _ = url.partition(":")
    if not sep or "/" in scheme or "?" in scheme or "#" in scheme:
        return url  # relative link (no scheme)
    return url if scheme.strip().lower() in _SAFE_SCHEMES else None


def collapse_toggle_html(section_id: str, attention: int = 0) -> str:
    """A collapse/expand `<button>` for a section or card header on the
    dashboard (launcher/errors/setup pages) -- `COLLAPSE_JS` wires it up,
    reads and writes the collapsed state in `localStorage`. `attention`, when
    truthy, renders a small dot/count next to the toggle so something needing
    a look is still visible even while the section is collapsed."""
    sid = esc(section_id)
    dot = (f'<span class="collapse-attn" title="{int(attention)} need attention">{int(attention)}</span>'
           if attention else "")
    return (
        f'{dot}<button class="collapse-toggle" type="button" data-collapse="{sid}" '
        f'aria-expanded="true" aria-controls="body-{sid}" aria-label="collapse section">'
        '<span class="chevron" aria-hidden="true">&#9662;</span></button>'
    )


# Wires up every collapse_toggle_html() button on the page: remembers the
# collapsed state per section id in localStorage, wrapped in try/catch so a
# blocked or full store only loses the "remember", never breaks the toggle.
COLLAPSE_JS = r"""
(() => {
  const key = id => 'lmsa-collapsed-' + id;
  const apply = (btn, collapsed) => {
    const body = document.getElementById('body-' + btn.dataset.collapse);
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.classList.toggle('collapsed', collapsed);
    if (body) body.hidden = collapsed;
  };
  document.querySelectorAll('[data-collapse]').forEach(btn => {
    const id = btn.dataset.collapse;
    let collapsed = false;
    try { collapsed = localStorage.getItem(key(id)) === '1'; } catch (e) {}
    apply(btn, collapsed);
    btn.addEventListener('click', () => {
      const next = btn.getAttribute('aria-expanded') === 'true';
      apply(btn, next);
      try {
        if (next) localStorage.setItem(key(id), '1');
        else localStorage.removeItem(key(id));
      } catch (e) {}
    });
  });
})();
"""


# -- email check-off (#61) -----------------------------------------------------
#
# A report is a static file: its checkboxes ship disabled, and only this
# script -- when the page is served by the dashboard under /reports/ -- asks
# the dashboard for a per-launch token (never written to disk) and enables
# them. The dashboard serves reports with a CSP that allows exactly this
# script by hash (HANDLED_JS_HASH) and nothing else. Opened from disk, the
# boxes stay disabled with a tooltip saying why.

HANDLED_OFFLINE_TITLE = "open via the dashboard to check off"

HANDLED_JS = r"""
(() => {
  const boxes = Array.from(document.querySelectorAll('input[data-msgid]'));
  if (!boxes.length || !/^https?:$/.test(location.protocol) || !location.pathname.startsWith('/reports/')) return;
  fetch('/api/handled/token', { cache: 'no-store', credentials: 'same-origin' })
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(j => {
      const token = String(j.token || '');
      if (!token) return;
      boxes.forEach(box => {
        const label = box.closest('.handled-check');
        box.disabled = false;
        if (label) label.title = 'check off: hide it from later reports';
        box.addEventListener('change', async () => {
          const want = box.checked;
          box.disabled = true;
          try {
            const r = await fetch('/api/handled', {
              method: 'POST', credentials: 'same-origin',
              headers: { 'Content-Type': 'application/json', 'X-LMSA-Token': token },
              body: JSON.stringify({ msg_id: box.dataset.msgid, handled: want }),
            });
            if (!r.ok) throw new Error(String(r.status));
            const card = box.closest('.card');
            if (card) card.classList.toggle('is-handled', want);
          } catch (e) {
            box.checked = !want;
            if (label) label.title = 'could not save -- is the dashboard still running?';
          }
          box.disabled = false;
        });
      });
    })
    .catch(() => {});
})();
"""

# CSP source for the one inline script reports may run (see tasks.dashboard).
HANDLED_JS_HASH = "sha256-" + base64.b64encode(hashlib.sha256(HANDLED_JS.encode("utf-8")).digest()).decode("ascii")


def handled_checkbox_html(msg_id: str, handled: bool = False) -> str:
    """The check-off box for one email card: disabled until HANDLED_JS
    enables it on a dashboard-served page."""
    checked = " checked" if handled else ""
    return (
        f'<label class="handled-check" title="{esc(HANDLED_OFFLINE_TITLE)}">'
        f'<input type="checkbox" data-msgid="{esc(msg_id)}" disabled{checked}> handled</label>'
    )


def _tone(tone: str) -> str:
    if tone not in TONES:
        raise ValueError(f"unknown tone {tone!r}; expected one of {TONES}")
    return f"t-{tone}"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _paragraphs(text: str) -> str:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return "".join(f"<p>{esc(b)}</p>" for b in blocks)


def _sparkline(values: Sequence[float]) -> str:
    """Tiny inline SVG line chart, scaled to its container by viewBox."""
    values = [float(v) for v in values]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    w, h, pad = 100, 26, 2
    step = w / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{pad + (h - 2 * pad) * (1 - (v - lo) / span):.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg class="sparkline" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'role="img" aria-hidden="true"><polyline points="{points}"></polyline></svg>'
    )


def _heat_level(value: float, lo: float, hi: float) -> int:
    """Bucket a value into a 5-step (0-4) intensity level over [lo, hi]."""
    if hi <= lo:
        return 4 if value > 0 else 0
    level = (value - lo) / (hi - lo) * 4
    return round(level) if math.isfinite(level) else 0


@dataclass
class Tile:
    label: str
    value: object
    delta: str | None = None
    tone: str = "acc"
    note: str | None = None
    spark: Sequence[float] = ()

    def render(self) -> str:
        delta = f'<span class="delta">{esc(self.delta)}</span>' if self.delta else ""
        note = f'<div class="note">{esc(self.note)}</div>' if self.note else ""
        svg = _sparkline(self.spark)
        spark = f'<div class="spark">{svg}</div>' if svg else ""
        return (
            f'<div class="tile {_tone(self.tone)}">'
            f'<div class="label">{esc(self.label)}</div>'
            f'<div class="value">{esc(_fmt(self.value))}{delta}</div>{note}{spark}</div>'
        )


@dataclass
class Link:
    """A `table()` cell rendered as a link. `href` is untrusted input (e.g. an
    email header): it goes through `safe_href` at render time, same as
    `Card.href`, so an unsafe scheme falls back to plain (escaped) text.
    """
    text: str
    href: str | None = None


@dataclass
class Card:
    title: str
    meta: str = ""
    body: str = ""
    tags: Sequence[str] = ()
    tone: str = "acc"
    href: str | None = None
    ribbon: str | None = None
    # A `data:` URI thumbnail, e.g. `f"data:image/jpeg;base64,{b64}"`. This
    # is rendered as-is (not passed through `safe_href`, which is for link
    # schemes) -- only pass an image *your own code* base64-encoded from
    # bytes it decoded/re-rendered itself, never a URL or string taken
    # verbatim from untrusted content (email, a scraped page, model output).
    image: str | None = None
    # Gmail X-GM-MSGID of the email this card is built from (#61): renders a
    # "handled" check-off box wired up by HANDLED_JS. `handled` pre-checks it.
    msg_id: str | None = None
    handled: bool = False

    def render(self) -> str:
        title = esc(self.title)
        href = safe_href(self.href)
        if href:
            title = f'<a href="{esc(href)}">{title}</a>'
        image = f'<img class="thumb" src="{esc(self.image)}" alt="" loading="lazy">' if self.image else ""
        meta = f'<div class="meta">{esc(self.meta)}</div>' if self.meta else ""
        body = f'<div class="body">{_paragraphs(self.body)}</div>' if self.body else ""
        tags = ""
        if self.tags:
            tags = '<div class="tags">' + "".join(
                f'<span class="tag">{esc(t)}</span>' for t in self.tags
            ) + "</div>"
        ribbon = f'<div class="ribbon">{esc(self.ribbon)}</div>' if self.ribbon else ""
        check = handled_checkbox_html(self.msg_id, self.handled) if self.msg_id else ""
        return (
            f'<article class="card {_tone(self.tone)}">{ribbon}{image}'
            f"<h3>{title}</h3>{meta}{body}{tags}{check}</article>"
        )


@dataclass
class StoryQuote:
    """A key quote on a `Story` card. `verified` is False unless the caller
    matched it against the source text: an unverified quote is shown with an
    "unverified" flag, never as if it were checked."""
    text: str
    href: str | None = None
    verified: bool = False


@dataclass
class StoryExtra:
    """One labelled line an extension attaches to a `Story` card (market odds,
    coverage balance, ...). `label`/`text` are escaped; `href` goes through
    `safe_href`."""
    label: str
    text: str
    href: str | None = None
    tone: str = "muted"


@dataclass
class StorySource:
    """A source link on a `Story` card with an optional badge (e.g. a
    reliability tag, #69). Text and badge are escaped; `href` goes through
    `safe_href`."""
    text: str
    href: str | None = None
    badge: str | None = None
    tone: str = "muted"


@dataclass
class StackedBar:
    """A thin horizontal bar split into segments (e.g. coverage by lean,
    #69). `segments` are `(label, value, tone)`; widths are each value's
    share of the total (negative / non-finite values count as 0, an all-zero
    bar renders nothing). Labels are escaped and shown in the legend."""
    segments: Sequence[tuple[str, float, str]]
    label: str = ""

    def render(self) -> str:
        segs = []
        for label, value, tone in self.segments:
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = 0.0
            segs.append((str(label), v if math.isfinite(v) and v > 0 else 0.0, tone))
        total = sum(v for _, v, _ in segs)
        if total <= 0:
            return ""
        parts = "".join(
            f'<span class="seg {_tone(tone)}" style="width:{v / total * 100:.1f}%" title="{esc(label)}"></span>'
            for label, v, tone in segs if v > 0)
        legend = "".join(
            f'<span class="{_tone(tone)}"><i></i>{esc(label)} {v / total * 100:.0f}%</span>'
            for label, v, tone in segs if v > 0)
        head = f'<span class="stacked-label">{esc(self.label)}</span>' if self.label else ""
        return (f'<div class="stacked">{head}<div class="stacked-track">{parts}</div>'
                f'<div class="stacked-legend">{legend}</div></div>')


_ANCHOR_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class Story:
    """A briefing story card (#52): headline, badge (NEW / UPDATED / ...),
    summary, highlights, key quotes (flagged when unverified), timeline, why
    it matters, linked sources, extension lines. Every field is untrusted
    text and is escaped; every href goes through `safe_href`. `anchor` makes
    the card a link target (`#<anchor>`, characters outside A-Z a-z 0-9 _ -
    dropped)."""
    title: str
    summary: str = ""
    meta: str = ""
    badge: str | None = None
    tone: str = "acc"
    highlights: Sequence[str] = ()
    quotes: Sequence[StoryQuote] = ()
    timeline: Sequence[tuple[str, str]] = ()
    why: str = ""
    sources: Sequence[Link | StorySource] = ()
    tags: Sequence[str] = ()
    extras: Sequence[StoryExtra] = ()
    anchor: str | None = None
    bars: Sequence[StackedBar] = ()

    def render(self) -> str:
        anchor = _ANCHOR_RE.sub("", self.anchor or "")
        aid = f' id="{esc(anchor)}"' if anchor else ""
        badge = f'<span class="story-badge">{esc(self.badge)}</span>' if self.badge else ""
        meta = f'<div class="meta">{esc(self.meta)}</div>' if self.meta else ""
        summary = f'<div class="body">{_paragraphs(self.summary)}</div>' if self.summary else ""
        bars = "".join(b.render() for b in self.bars)
        parts = [f"<h3>{badge}{esc(self.title)}</h3>{meta}{bars}{summary}"]
        if self.highlights:
            parts.append('<ul class="story-highlights">' + "".join(
                f"<li>{esc(h)}</li>" for h in self.highlights) + "</ul>")
        if self.quotes:
            items = []
            for q in self.quotes:
                href = safe_href(q.href)
                cite = f' <a class="story-cite" href="{esc(href)}">source</a>' if href else ""
                flag = "" if q.verified else ' <span class="story-unverified">unverified</span>'
                cls = "" if q.verified else ' class="unverified"'
                items.append(f"<blockquote{cls}>{esc(q.text)}{cite}{flag}</blockquote>")
            parts.append('<div class="story-quotes">' + "".join(items) + "</div>")
        if self.timeline:
            parts.append('<ol class="story-timeline">' + "".join(
                f'<li><span class="when">{esc(d)}</span> {esc(e)}</li>' for d, e in self.timeline) + "</ol>")
        if self.why:
            parts.append(f'<p class="story-why"><span class="tag">why it matters</span> {esc(self.why)}</p>')
        if self.extras:
            rows = []
            for x in self.extras:
                href = safe_href(x.href)
                text = f'<a href="{esc(href)}">{esc(x.text)}</a>' if href else esc(x.text)
                rows.append(f'<li class="{_tone(x.tone)}"><span class="tag">{esc(x.label)}</span> {text}</li>')
            parts.append('<ul class="story-extras">' + "".join(rows) + "</ul>")
        if self.sources:
            links = []
            for link in self.sources:
                href = safe_href(link.href)
                item = f'<a href="{esc(href)}">{esc(link.text)}</a>' if href else f"<span>{esc(link.text)}</span>"
                badge_text = getattr(link, "badge", None)
                if badge_text:
                    item += f'<span class="tag {_tone(getattr(link, "tone", "muted"))}">{esc(badge_text)}</span>'
                links.append(f'<span class="story-source">{item}</span>')
            parts.append('<div class="story-sources">' + "".join(links) + "</div>")
        if self.tags:
            parts.append('<div class="tags">' + "".join(f'<span class="tag">{esc(t)}</span>' for t in self.tags) + "</div>")
        return f'<article class="story {_tone(self.tone)}"{aid}>{"".join(parts)}</article>'


@dataclass
class Marker:
    """A point event on `Section.world_map` (an earthquake, a headline's
    dateline). lat/lon are WGS84 degrees, projected with `lmsa.geo.project`;
    `label` is untrusted text (escaped, shown as the tooltip); `href` goes
    through `safe_href`; `size` scales the dot radius (1.0 = default,
    clamped to 0.3-5; a non-numeric or NaN size falls back to 1.0)."""
    lat: float
    lon: float
    label: str = ""
    tone: str = "hot"
    size: float = 1.0
    href: str | None = None


def _geo_num(v: float) -> str:
    return _fmt(int(v)) if float(v).is_integer() else _fmt(float(v))


def _geo_level(value: float, lo: float, hi: float, scale: str) -> str:
    """CSS class for a choropleth value: geo-s0..s4 (sequential over
    [lo, hi]) or geo-dn2..geo-dp2 (diverging, symmetric around 0)."""
    if scale == "diverging":
        m = max(abs(lo), abs(hi))
        step = round(value / m * 2) if m else 0
        return {-2: "geo-dn2", -1: "geo-dn1", 0: "geo-d0", 1: "geo-dp1", 2: "geo-dp2"}[step]
    return f"geo-s{_heat_level(value, lo, hi)}"


def _geo_legend(label: str, lo: float, hi: float, scale: str) -> str:
    if scale == "diverging":
        m = max(abs(lo), abs(hi))
        classes = ("geo-dn2", "geo-dn1", "geo-d0", "geo-dp1", "geo-dp2")
        low, high = -m, m
    else:
        classes = tuple(f"geo-s{i}" for i in range(5))
        low, high = lo, hi
    swatches = "".join(
        f'<svg class="geo-swatch" viewBox="0 0 10 10" aria-hidden="true"><rect class="{c}" width="10" height="10"></rect></svg>'
        for c in classes
    )
    cap = f'<span class="geo-legend-label">{esc(label)}</span>' if label else ""
    return (
        f'<div class="geo-legend">{cap}<span class="geo-legend-val">{esc(_geo_num(low))}</span>'
        f'{swatches}<span class="geo-legend-val">{esc(_geo_num(high))}</span></div>'
    )


def _tiles(tiles: Iterable[Tile]) -> str:
    return '<div class="tiles">' + "".join(t.render() for t in tiles) + "</div>"


class Section:
    """One titled block of a report. Every builder method returns self to chain."""

    def __init__(self, title: str, tone: str = "acc"):
        self.title = title
        self.tone = _tone(tone)
        self._parts: list[str] = []

    def text(self, text: str) -> Section:
        self._parts.append(_paragraphs(text))
        return self

    def tiles(self, tiles: Iterable[Tile]) -> Section:
        self._parts.append(_tiles(tiles))
        return self

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> Section:
        """Numeric cells (int/float) are right-aligned in mono automatically.
        A `Link` cell renders as an anchor (see `Link`); everything else is
        formatted with `_fmt` and escaped.
        """
        rows = [list(r) for r in rows]
        numeric = [
            bool(rows) and all(isinstance(r[i], (int, float)) for r in rows if i < len(r))
            for i in range(len(headers))
        ]
        num = ' class="num"'
        head = "".join(
            f"<th{num if numeric[i] else ''}>{esc(h)}</th>" for i, h in enumerate(headers)
        )

        def cell(v: object) -> str:
            if isinstance(v, Link):
                href = safe_href(v.href)
                text = esc(v.text)
                return f'<a href="{esc(href)}">{text}</a>' if href else text
            return esc(_fmt(v))

        body = "".join(
            "<tr>" + "".join(
                f"<td{num if isinstance(v, (int, float)) and not isinstance(v, bool) else ''}>"
                f"{cell(v)}</td>"
                for v in r
            ) + "</tr>"
            for r in rows
        )
        self._parts.append(
            f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>"
        )
        return self

    def bars(self, items: Iterable[tuple[str, float]], unit: str = "") -> Section:
        items = list(items)
        top = max((v for _, v in items), default=0) or 1
        rows = "".join(
            f'<div class="bar"><span class="name" title="{esc(n)}">{esc(n)}</span>'
            f'<span class="track"><span class="fill" style="width:{max(v / top * 100, 0):.1f}%"></span></span>'
            f'<span class="amt">{esc(_fmt(v))}{esc(unit)}</span></div>'
            for n, v in items
        )
        self._parts.append(f'<div class="bars">{rows}</div>')
        return self

    def heatmap(self, cells: dict[date, float]) -> Section:
        """GitHub-contribution-style calendar: weeks x weekdays, 5-step tone ramp."""
        if not cells:
            self._parts.append('<div class="heatmap-empty">No data.</div>')
            return self
        days = sorted(cells)
        lo, hi = min(cells.values()), max(cells.values())

        def sunday_index(d: date) -> int:
            return (d.weekday() + 1) % 7  # Sunday=0 ... Saturday=6

        start = days[0] - timedelta(days=sunday_index(days[0]))
        end = days[-1] + timedelta(days=6 - sunday_index(days[-1]))
        total_days = (end - start).days + 1
        weeks = total_days // 7

        cell, gap = 11, 3
        step = cell + gap
        label_h = 16
        width = weeks * step
        height = label_h + 7 * step

        rects, months, seen_month = [], [], None
        for i in range(total_days):
            d = start + timedelta(days=i)
            week, row = divmod(i, 7)
            if row == 0:
                month = d.strftime("%b")
                if month != seen_month:
                    months.append(
                        f'<text x="{week * step}" y="{label_h - 5}" class="heat-month">{esc(month)}</text>'
                    )
                    seen_month = month
            v = cells.get(d)
            cls = f"heat-{_heat_level(v, lo, hi)}" if v is not None else "heat-empty"
            title = f"{d.isoformat()}: {esc(_fmt(v))}" if v is not None else d.isoformat()
            rects.append(
                f'<rect class="{cls}" x="{week * step}" y="{label_h + row * step}" '
                f'width="{cell}" height="{cell}" rx="2"><title>{title}</title></rect>'
            )

        svg = (
            f'<svg class="heatmap" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="calendar heatmap">{"".join(months)}{"".join(rects)}</svg>'
        )
        self._parts.append(f'<div class="heatmap-wrap">{svg}</div>')
        return self

    def grid_heatmap(
        self,
        rows: Sequence[str],
        cols: Sequence[str],
        values: Sequence[Sequence[float]],
    ) -> Section:
        """Generic matrix heatmap (e.g. weekday x hour), same 5-step tone ramp."""
        flat = [v for row in values for v in row]
        lo, hi = (min(flat), max(flat)) if flat else (0, 0)

        cell, gap = 20, 2
        step = cell + gap
        label_w, label_h = 64, 20
        width = label_w + len(cols) * step
        height = label_h + len(rows) * step

        col_labels = "".join(
            f'<text x="{label_w + c * step + cell / 2:.1f}" y="{label_h - 6}" '
            f'class="grid-heat-label" text-anchor="middle">{esc(col)}</text>'
            for c, col in enumerate(cols)
        )
        row_labels = "".join(
            f'<text x="{label_w - 8}" y="{label_h + r * step + cell / 2 + 4:.1f}" '
            f'class="grid-heat-label" text-anchor="end">{esc(row)}</text>'
            for r, row in enumerate(rows)
        )
        cells = "".join(
            f'<rect class="heat-{_heat_level(v, lo, hi)}" x="{label_w + c * step}" '
            f'y="{label_h + r * step}" width="{cell}" height="{cell}" rx="2">'
            f'<title>{esc(row)} · {esc(col)}: {esc(_fmt(v))}</title></rect>'
            for r, (row, row_vals) in enumerate(zip(rows, values))
            for c, (col, v) in enumerate(zip(cols, row_vals))
        )
        svg = (
            f'<svg class="grid-heatmap" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="heatmap">{col_labels}{row_labels}{cells}</svg>'
        )
        self._parts.append(f'<div class="heatmap-wrap">{svg}</div>')
        return self

    def world_map(
        self,
        values: Mapping[str, float] | None = None,
        *,
        scale: str = "sequential",
        legend: str | bool = True,
        links: Mapping[str, str] | None = None,
        markers: Iterable[Marker] = (),
        label: str = "world map",
    ) -> Section:
        """Offline world map as inline SVG (Natural Earth 1:110m, Equal Earth).

        values  -- alpha-2 code (any case) -> number; shades the country.
                   "sequential": 5 steps of the section tone over [min, max].
                   "diverging": blue below 0, pink above, symmetric around 0.
        legend  -- False hides it; a string captions it.
        links   -- alpha-2 -> href (e.g. "#country-ua", see `anchor()`);
                   the country becomes a link. Hrefs go through `safe_href`.
        markers -- `Marker` point events, projected from lat/lon.

        Unknown codes and non-finite values/coordinates are skipped and
        counted in a note under the map. Countries too small for the 1:110m
        outlines (Singapore, Malta...) are drawn as a dot at their label
        point. Adds ~77 KB per map to the report (outlines are inlined per map).
        """
        from . import geo

        if scale not in ("sequential", "diverging"):
            raise ValueError(f"unknown scale {scale!r}; expected 'sequential' or 'diverging'")
        unknown: set[str] = set()
        shaded: dict[str, float] = {}
        bad_values = 0
        for code, v in (values or {}).items():
            c = geo.country(code)
            if c is None:
                unknown.add(str(code))
                continue
            try:
                v = float(v)
            except (TypeError, ValueError, OverflowError):  # 10**400 overflows a float
                bad_values += 1
                continue
            if not math.isfinite(v):
                bad_values += 1
                continue
            shaded[c.alpha2] = v
        hrefs: dict[str, str] = {}
        for code, href in (links or {}).items():
            c = geo.country(code)
            if c is None:
                unknown.add(str(code))
            elif (safe := safe_href(href)) is not None:
                hrefs[c.alpha2] = safe
        lo, hi = (min(shaded.values()), max(shaded.values())) if shaded else (0.0, 0.0)

        def shape(cc: str, body: str) -> str:
            href = hrefs.get(cc)
            return f'<a href="{esc(href)}">{body}</a>' if href else body

        def tip(cc: str) -> str:
            name = geo.country_name(cc) or cc
            v = shaded.get(cc)
            return f"<title>{esc(name)}{': ' + esc(_geo_num(v)) if v is not None else ''}</title>"

        paths = geo.world_paths()
        edge = [geo.project(lat, 180) for lat in range(-90, 91, 5)]
        edge += [geo.project(lat, -180) for lat in range(90, -91, -5)]
        parts = ['<path class="geo-sphere" d="M' + "L".join(f"{x:.1f} {y:.1f}" for x, y in edge) + 'z"></path>']
        for cc, d in paths.items():
            cls = "geo-c " + (_geo_level(shaded[cc], lo, hi, scale) if cc in shaded else "geo-none")
            parts.append(shape(cc, f'<path class="{cls}" data-cc="{esc(cc)}" d="{d}">{tip(cc)}</path>'))
        for cc in sorted((set(shaded) | set(hrefs)) - set(paths)):  # too small for 1:110m
            c = geo.country(cc)
            if c is None or c.lat is None or c.lon is None:
                continue
            x, y = geo.project(c.lat, c.lon)
            cls = "geo-c geo-dot " + (_geo_level(shaded[cc], lo, hi, scale) if cc in shaded else "geo-none")
            parts.append(shape(cc, f'<circle class="{cls}" data-cc="{esc(cc)}" cx="{x:.1f}" cy="{y:.1f}" r="2.6">{tip(cc)}</circle>'))
        bad_markers = 0
        for m in markers:
            try:
                lat, lon = float(m.lat), float(m.lon)
            except (TypeError, ValueError, OverflowError):
                bad_markers += 1
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                bad_markers += 1
                continue
            try:
                size = float(1.0 if m.size is None else m.size)
            except OverflowError:  # a huge int: clamp like inf
                size = math.inf if m.size > 0 else -math.inf
            except (TypeError, ValueError):
                size = 1.0
            if math.isnan(size):
                size = 1.0
            x, y = geo.project(lat, lon)
            r = 3.5 * max(0.3, min(size, 5.0))  # 0 / negative -> smallest, inf -> largest
            title = f"<title>{esc(m.label)}</title>" if m.label else ""
            dot = f'<circle class="geo-marker {_tone(m.tone)}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}">{title}</circle>'
            href = safe_href(m.href)
            parts.append(f'<a href="{esc(href)}">{dot}</a>' if href else dot)

        svg = (
            f'<svg class="geo-svg" viewBox="0 0 {geo.MAP_W:g} {geo.MAP_H:g}" role="img" '
            f'aria-label="{esc(label)}">{"".join(parts)}</svg>'
        )
        foot = ""
        if legend is not False and shaded:
            foot += _geo_legend(legend if isinstance(legend, str) else "", lo, hi, scale)
        skipped = []
        if unknown:
            skipped.append(f"{len(unknown)} unknown country code{'s' if len(unknown) != 1 else ''}")
        if bad_values:
            skipped.append(f"{bad_values} non-numeric value{'s' if bad_values != 1 else ''}")
        if bad_markers:
            skipped.append(f"{bad_markers} marker{'s' if bad_markers != 1 else ''} without coordinates")
        if skipped:
            foot += f'<div class="geo-note">ignored: {esc(", ".join(skipped))}</div>'
        self._parts.append(f'<figure class="geo-map">{svg}{foot}</figure>')
        return self

    def anchor(self, anchor_id: str) -> Section:
        """An in-page link target (`href="#<anchor_id>"`), e.g. what a
        `world_map(links=...)` country jumps to. Call it first so the jump
        lands at the top of the section."""
        self._parts.append(f'<span class="anchor" id="{esc(anchor_id)}"></span>')
        return self

    def callout(self, text: str, label: str = "Note", tone: str = "warn") -> Section:
        self._parts.append(
            f'<div class="callout {_tone(tone)}"><span class="tag">{esc(label)}</span>'
            f"{_paragraphs(text)}</div>"
        )
        return self

    def cards(self, cards: Iterable[Card]) -> Section:
        self._parts.append('<div class="cards">' + "".join(c.render() for c in cards) + "</div>")
        return self

    def stories(self, stories: Iterable[Story]) -> Section:
        """Briefing story cards (`Story`), one per row."""
        self._parts.append('<div class="stories">' + "".join(s.render() for s in stories) + "</div>")
        return self

    def handled_hidden(self, cards: Sequence[Card]) -> Section:
        """Collapsed "N handled hidden" block holding cards for emails already
        checked off (#61): hidden by default, a click on the summary shows
        them (no script needed, so it works from disk too). No-op if empty."""
        cards = list(cards)
        if cards:
            n = len(cards)
            self._parts.append(
                f'<details class="handled-hidden"><summary>{n} handled hidden '
                f'<span class="hint">· show</span></summary>'
                '<div class="cards">' + "".join(c.render() for c in cards) + "</div></details>"
            )
        return self

    def raw(self, trusted_html: str) -> Section:
        """Insert HTML verbatim. Never pass model output or email content here."""
        self._parts.append(trusted_html)
        return self

    def render(self, index: int) -> str:
        return (
            f'<section class="{self.tone}"><h2><span class="idx">{index:02d} //</span>'
            f"{esc(self.title)}</h2>{''.join(self._parts)}</section>"
        )


class Report:
    def __init__(
        self,
        title: str,
        *,
        kicker: str = "",
        subtitle: str = "",
        sticker: str | None = None,
        sticker_tone: str = "warn",
        source: str | None = None,
        ticker: Sequence[str] = (),
    ):
        self.title = title
        self.kicker = kicker
        self.subtitle = subtitle
        self.sticker = sticker
        self.sticker_tone = _tone(sticker_tone)
        self.source = source
        self.ticker = list(ticker)
        self.created = datetime.now()
        self._top: list[str] = []
        self._sections: list[Section] = []
        self._footer_notes: list[str] = []

    def tiles(self, tiles: Iterable[Tile]) -> Report:
        self._top.append(_tiles(tiles))
        return self

    def footer_note(self, text: str) -> Report:
        """Append a short plain-text note to the footer line (escaped)."""
        self._footer_notes.append(text)
        return self

    def section(self, title: str, tone: str = "acc") -> Section:
        s = Section(title, tone)
        self._sections.append(s)
        return s

    def render(self) -> str:
        kicker = f'<div class="kicker">{esc(self.kicker)}</div>' if self.kicker else ""
        subtitle = f'<p class="subtitle">{esc(self.subtitle)}</p>' if self.subtitle else ""
        sticker = (
            f'<div class="sticker {self.sticker_tone}">{esc(self.sticker)}</div>' if self.sticker else ""
        )
        sections = "".join(s.render(i) for i, s in enumerate(self._sections, 1))
        # Escaped text can't produce this literal (quotes become &quot;).
        script = f"<script>{HANDLED_JS}</script>" if 'data-msgid="' in sections else ""
        footer = f"// generated {self.created:%Y-%m-%d %H:%M}"
        if self.source:
            footer += f" · {esc(self.source)}"
        for note in self._footer_notes:
            footer += f" · {esc(note)}"
        ticker = ""
        if self.ticker:
            items = "".join(f'<span class="ticker-item">{esc(t)}</span>' for t in self.ticker)
            ticker = f'<div class="ticker"><div class="ticker-track">{items}{items}</div></div>'
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="color-scheme" content="dark"><title>{esc(self.title)}</title>'
            f"{FONTS}<style>{THEME.read_text(encoding='utf-8')}</style></head><body>"
            f'<div class="hazard"></div>{ticker}<main class="wrap">'
            f'<header class="masthead">{sticker}{kicker}<h1>{esc(self.title)}</h1>{subtitle}</header>'
            f"{''.join(self._top)}{sections}<footer>{footer}</footer></main>{script}</body></html>"
        )

    def write(self, name: str | None = None, out_dir: Path | None = None) -> Path:
        """Write to output/<day>/<slug>/<slug>-<timestamp>.html unless a name is given
        (see `lmsa.layout`; undated names such as index.html stay at the top level)."""
        out_dir = out_dir or settings().output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-") or "report"
            name = f"{slug}-{self.created:%Y%m%d-%H%M}.html"
        from . import layout
        path = layout.placed(out_dir, name)
        path.write_text(self.render(), encoding="utf-8")

        # Rebuild the index hub to keep it current (unless writing the index itself).
        if name != "index.html":
            from . import index as _index
            _index.rebuild(out_dir)

        return path
