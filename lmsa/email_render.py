"""Email-safe HTML (and plain text) for a master report (#103, #119).

`lmsa.report.Report.render()` can't be used for an email body: its
`<style>` block, CSS variables, web fonts, JS, animation and SVG are
stripped or broken by Gmail and Apple Mail. This renderer is the opposite,
written for iOS Gmail and iOS Mail first:

  - one column, at most 640px wide, 16px side padding; table layout,
    **inline styles only** (plus `bgcolor`), no `<style>`, no CSS variables,
    no JS, no web fonts, no SVG, no images;
  - the font stack, base size (15px) and word wrapping are set once on the
    wrapper and inherited, so the markup stays lean;
  - tiles two to a row, the status chip on its own line under the section
    title, body text >= 14px, wide tables stacked into one block per row,
    long URLs/words wrap;
  - `color-scheme: light dark` declared and every block sets its own
    background *and* text colour (the Afterhours palette, approximated), so
    a client's dark-mode handling leaves each pair readable;
  - all text is escaped; only http(s) links are emitted (a local report path
    is useless on a phone).

**Nothing is trimmed**: the body is as long as its content (#119). A body
over Gmail's clip size (`GMAIL_CLIP`) still arrives whole; Gmail shows the
start and a "View entire message" link.

A section's `mode`:
  full      the task's whole saved report (`lmsa.email_report.FullReport`);
            without one, its digest (every item) and a note saying why
  digest    headline numbers, callouts and the top items
  headline  numbers and callouts only
  link      one line; the report is in the attachment

    email = render_email("Morning Report", sections, intro="...")
    email.html, email.text
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from .digest import FAILED, NOT_RUN, OK, STALE, Digest, SectionStatus, http_url
from .email_report import (Bars, CardB, Callout, FullReport, Note, Para, Span, Stacked, StoryB, Subhead,
                           Table, TileB, Tiles)
from .report import _fmt

GMAIL_CLIP = 102_000  # bytes of HTML after which Gmail clips a message ("View entire message")
MAX_HEADING = 200  # characters of the email title/subtitle and each section title
STACK_COLUMNS = 3  # a table wider than this is shown one block per row (it wouldn't fit a phone)

# Afterhours, approximated (lmsa/theme/afterhours.css).
BG = "#0b0d12"
SURFACE = "#161b24"
LINE = "#2a3140"
TEXT = "#e8ebf2"
MUTED = "#8b94a8"
TONE_COLORS = {
    "acc": "#1abc9c", "hot": "#ff2e63", "warn": "#f1c40f",
    "info": "#3498db", "violet": "#9b59b6", "muted": "#8b94a8",
}
LINK = TONE_COLORS["info"]
STATUS_COLORS = {OK: "#1abc9c", STALE: "#f1c40f", FAILED: "#ff2e63", NOT_RUN: "#8b94a8"}
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "font-family:Menlo,Consolas,'Courier New',monospace;"
MODES = ("full", "digest", "headline", "link")


@dataclass
class EmailSection:
    """One bundled report as the renderer sees it. `digest` is already
    normalized (`lmsa.digest.normalize`); `more` counts items it already
    left out (e.g. past `max_items`). `mode` is one of `MODES`; `full` is the
    parsed saved report a `full` section shows (None: show the digest)."""
    title: str
    status: SectionStatus
    digest: Digest
    mode: str = "digest"
    more: int = 0
    note: str | None = None  # a short plain-text note, e.g. a failed digest hook
    full: FullReport | None = field(default=None, repr=False)


@dataclass
class RenderedEmail:
    html: str
    text: str

    @property
    def bytes(self) -> int:
        return len(self.html.encode("utf-8"))

    @property
    def clipped(self) -> bool:
        """Gmail will show only the start of it, plus "View entire message"."""
        return self.bytes > GMAIL_CLIP


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _tc(tone: str | None) -> str:
    return TONE_COLORS.get(tone or "acc", TONE_COLORS["acc"])


def _a(text: str, href: str | None) -> str:
    url = http_url(href)
    if url:
        return f'<a href="{_e(url)}" style="color:{LINK};">{_e(text)}</a>'
    return _e(text)


def _box(inner: str, bar: str, bg: str = BG) -> str:
    """A block with its own background, text colour and a coloured left bar."""
    return (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin-top:8px;"><tr><td bgcolor="{bg}" style="background-color:{bg};color:{TEXT};'
            f'border-left:3px solid {bar};padding:8px 12px;">{inner}</td></tr></table>')


def _p(inner: str, style: str = "") -> str:
    return f'<div style="padding-top:6px;{style}">{inner}</div>'


def _small(text: str, color: str = MUTED, extra: str = "") -> str:
    return f'<div style="font-size:13px;color:{color};{extra}">{_e(text)}</div>'


def _chip(status: SectionStatus) -> str:
    color = STATUS_COLORS.get(status.kind, MUTED)
    return (f'<div style="padding-top:4px;"><span style="{MONO}font-size:12px;font-weight:bold;color:{BG};'
            f'background-color:{color};padding:2px 6px;">{_e(status.label)}</span></div>')


# -- blocks: HTML ----------------------------------------------------------------

def _tiles_html(tiles: list[TileB]) -> str:
    if not tiles:
        return ""
    gap = '<td width="8" style="font-size:0;line-height:0;">&nbsp;</td>'

    def cell(t: TileB | None) -> str:
        if t is None:
            return '<td width="50%"></td>'
        delta = f' <span style="font-size:13px;color:{_tc(t.tone)};">{_e(t.delta)}</span>' if t.delta else ""
        note = _small(t.note) if t.note else ""
        return (f'<td width="50%" valign="top" bgcolor="{BG}" style="background-color:{BG};color:{TEXT};'
                f'border-top:3px solid {_tc(t.tone)};padding:8px 10px;">'
                f'<div style="{MONO}font-size:12px;color:{MUTED};">{_e(t.label)}</div>'
                f'<div style="font-size:22px;font-weight:bold;">{_e(t.value)}{delta}</div>{note}</td>')

    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2] + [None]
        rows.append(f"<tr>{cell(pair[0])}{gap}{cell(pair[1])}</tr>")
    spacer = '<tr><td colspan="3" style="height:8px;font-size:0;line-height:0;">&nbsp;</td></tr>'
    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin-top:8px;">{spacer.join(rows)}</table>')


def _table_html(t: Table) -> str:
    if not t.rows and not t.headers:
        return ""
    width = max([len(t.headers)] + [len(r) for r in t.rows])
    cell = f"padding:6px;border-bottom:1px solid {LINE};"
    table = (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
             f'bgcolor="{BG}" style="margin-top:8px;border-collapse:collapse;background-color:{BG};color:{TEXT};font-size:14px;">')
    if width > STACK_COLUMNS:
        # Too wide for a phone: one wrapping line per row, values in column
        # order ("a · b · c"), the column names once on top. Lean on purpose --
        # a report's 150-row table stays a few dozen KB.
        head = (f'<tr><td style="{cell}color:{MUTED};font-size:12px;font-weight:bold;">'
                f'{" · ".join(_e(h) for h in t.headers)}</td></tr>') if t.headers else ""
        body = "".join(
            f'<tr><td style="{cell}"><b>{_a(row[0].span.text, row[0].span.href)}</b>'
            + "".join(f" · {_a(c.span.text or '–', c.span.href)}" for c in row[1:]) + "</td></tr>"
            for row in t.rows
        )
        return f"{table}{head}{body}</table>"
    head = ""
    if t.headers:
        head = "<tr>" + "".join(
            f'<td style="{cell}color:{MUTED};font-size:12px;font-weight:bold;">{_e(h)}</td>' for h in t.headers
        ) + "</tr>"
    body = "".join(
        "<tr>" + "".join(
            f'<td valign="top" style="{cell}{"text-align:right;" if c.num else ""}">'
            f"{_a(c.span.text, c.span.href)}</td>" for c in row
        ) + "</tr>"
        for row in t.rows
    )
    return f"{table}{head}{body}</table>"


def _meter(parts: list[tuple[int, str]], height: int) -> str:
    """A horizontal bar made of table cells: (width %, colour) parts."""
    style = f"height:{height}px;font-size:0;line-height:0;"
    cells = "".join(f'<td width="{w}%" bgcolor="{c}" style="background-color:{c};{style}">&nbsp;</td>'
                    for w, c in parts if w > 0)
    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f"<tr>{cells}</tr></table>") if cells else ""


def _bars_html(b: Bars, tone: str) -> str:
    rows = []
    for r in b.rows:
        fill = max(0, min(100, round(r.pct)))
        rows.append(
            f'<tr><td style="padding:4px 0 2px 0;font-size:14px;">{_e(r.name)}</td>'
            f'<td align="right" style="padding:4px 0 2px 8px;font-size:14px;white-space:nowrap;">{_e(r.amount)}</td></tr>'
            f'<tr><td colspan="2" style="padding:0 0 4px 0;">{_meter([(fill, _tc(tone)), (100 - fill, LINE)], 6)}</td></tr>')
    return (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'bgcolor="{SURFACE}" style="margin-top:8px;background-color:{SURFACE};color:{TEXT};">'
            f'{"".join(rows)}</table>')


def _stacked_html(s: Stacked) -> str:
    label = _small(s.label) if s.label else ""
    meter = _meter([(pct, _tc(tone)) for _, pct, tone in s.segments], 8)
    legend = " &middot; ".join(f'<span style="color:{_tc(tone)};">{_e(text)}</span>'
                               for text, _, tone in s.segments)
    return f'<div style="padding-top:6px;">{label}{meter}<div style="font-size:13px;">{legend}</div></div>'


def _tags_html(tags: list[str]) -> str:
    return _small(" · ".join(f"#{t}" for t in tags), extra="padding-top:4px;") if tags else ""


def _paras_html(paras: list[str]) -> str:
    return "".join(_p(_e(p), "font-size:14px;") for p in paras)


def _card_html(c: CardB) -> str:
    ribbon = (f'<div style="font-size:12px;font-weight:bold;color:{_tc(c.tone)};">{_e(c.ribbon.upper())}</div>'
              if c.ribbon else "")
    title = f'<div style="font-size:15px;font-weight:bold;">{_a(c.title.text, c.title.href)}</div>'
    meta = _small(c.meta) if c.meta else ""
    return _box(ribbon + title + meta + _paras_html(c.paras) + _tags_html(c.tags), _tc(c.tone))


def _story_html(s: StoryB) -> str:
    tone = _tc(s.tone)
    badge = (f'<span style="font-size:11px;font-weight:bold;color:{BG};background-color:{tone};'
             f'padding:1px 5px;">{_e(s.badge.upper())}</span> ' if s.badge else "")
    parts = [f'<div style="font-size:16px;font-weight:bold;">{badge}{_e(s.title)}</div>']
    if s.meta:
        parts.append(_small(s.meta))
    parts += [_stacked_html(b) for b in s.stacked]
    parts.append(_paras_html(s.paras))
    parts += [_p(f"&bull; {_e(h)}", "font-size:14px;") for h in s.highlights]
    for q in s.quotes:
        cite = f" {_a('source', q.href)}" if q.href else ""
        flag = f' <span style="color:{TONE_COLORS["warn"]};font-size:12px;">unverified</span>' if not q.verified else ""
        parts.append(f'<div style="margin-top:6px;padding-left:8px;border-left:2px solid {LINE};font-size:14px;'
                     f'font-style:italic;">&ldquo;{_e(q.text)}&rdquo;{cite}{flag}</div>')
    for when, what in s.timeline:
        parts.append(_p(f'<span style="{MONO}font-size:12px;color:{MUTED};">{_e(when)}</span> {_e(what)}',
                        "font-size:14px;"))
    if s.why:
        parts.append(_p(f'<b style="color:{tone};">Why it matters:</b> {_e(s.why)}', "font-size:14px;"))
    for x in s.extras:
        parts.append(_p(f'<span style="color:{_tc(x.tone)};">{_e(x.label)}:</span> {_a(x.span.text, x.span.href)}',
                        "font-size:14px;"))
    if s.sources:
        links = " &middot; ".join(
            _a(src.span.text, src.span.href)
            + (f' <span style="color:{_tc(src.tone)};font-size:12px;">[{_e(src.badge)}]</span>' if src.badge else "")
            for src in s.sources)
        parts.append(_p(f'<span style="color:{MUTED};">Sources:</span> {links}', "font-size:13px;"))
    parts.append(_tags_html(s.tags))
    return _box("".join(parts), tone)


def _block_html(b, tone: str) -> str:
    if isinstance(b, Tiles):
        return _tiles_html(b.tiles)
    if isinstance(b, Para):
        return _p(_e(b.text))
    if isinstance(b, Note):
        return _p(_e(b.text), f"font-size:13px;color:{MUTED};font-style:italic;")
    if isinstance(b, Subhead):
        return f'<div style="padding-top:12px;font-size:14px;font-weight:bold;color:{MUTED};">{_e(b.text)}</div>'
    if isinstance(b, Table):
        return _table_html(b)
    if isinstance(b, Bars):
        return _bars_html(b, tone)
    if isinstance(b, Stacked):
        return _stacked_html(b)
    if isinstance(b, Callout):
        label = (f'<div style="font-size:12px;font-weight:bold;color:{_tc(b.tone)};">{_e(b.label.upper())}</div>'
                 if b.label else "")
        return _box(label + "".join(f'<div style="font-size:14px;">{_e(p)}</div>' for p in b.paras), _tc(b.tone))
    if isinstance(b, CardB):
        return _card_html(b)
    if isinstance(b, StoryB):
        return _story_html(b)
    return ""


def _full_html(f: FullReport) -> str:
    out = [_small(f.subtitle, extra="padding-top:4px;")] if f.subtitle else []
    out += [_block_html(b, "acc") for b in f.top]
    for sec in f.sections:
        out.append(f'<div style="padding:16px 0 4px 0;border-bottom:1px solid {LINE};font-size:16px;'
                   f'font-weight:bold;color:{_tc(sec.tone)};">{_e(sec.title)}</div>')
        out += [_block_html(b, sec.tone) for b in sec.blocks]
    if f.is_empty():
        out.append(_p("The report is empty.", f"color:{MUTED};"))
    return "".join(out)


# -- blocks: plain text ----------------------------------------------------------

def _t(span: Span) -> str:
    url = http_url(span.href)
    return f"{span.text} <{url}>" if url else span.text


def _block_text(b) -> list[str]:
    if isinstance(b, Tiles):
        return [f"{t.label}: {t.value}" + (f" {t.delta}" if t.delta else "") + (f" ({t.note})" if t.note else "")
                for t in b.tiles]
    if isinstance(b, Para):
        return [b.text]
    if isinstance(b, Note):
        return [f"[{b.text}]"]
    if isinstance(b, Subhead):
        return [f"-- {b.text} --"]
    if isinstance(b, Table):
        out = [" | ".join(b.headers)] if b.headers and not b.rows else []
        for row in b.rows:
            cells = [(f"{b.headers[i]}: " if i < len(b.headers) and b.headers[i] else "") + _t(c.span)
                     for i, c in enumerate(row)]
            out.append("- " + " | ".join(cells))
        return out
    if isinstance(b, Bars):
        return [f"- {r.name}: {r.amount}" for r in b.rows]
    if isinstance(b, Stacked):
        return [(f"{b.label}: " if b.label else "") + " · ".join(text for text, _, _ in b.segments)]
    if isinstance(b, Callout):
        return [f"! {b.label.upper()}: " + " ".join(b.paras) if b.label else "! " + " ".join(b.paras)]
    if isinstance(b, CardB):
        out = [f"* {c}" for c in ([b.ribbon.upper()] if b.ribbon else [])]
        out.append(f"* {_t(b.title)}")
        out += [f"  {x}" for x in ([b.meta] if b.meta else []) + b.paras]
        if b.tags:
            out.append("  " + " · ".join(f"#{t}" for t in b.tags))
        return out
    if isinstance(b, StoryB):
        out = [f"* {'[' + b.badge.upper() + '] ' if b.badge else ''}{b.title}"]
        out += [f"  {b.meta}"] if b.meta else []
        out += [f"  {line}" for s in b.stacked for line in _block_text(s)]
        out += [f"  {p}" for p in b.paras]
        out += [f"  - {h}" for h in b.highlights]
        out += [f'  "{q.text}"' + (f" <{http_url(q.href)}>" if http_url(q.href) else "")
                + ("" if q.verified else " (unverified)") for q in b.quotes]
        out += [f"  {when} {what}" for when, what in b.timeline]
        out += [f"  Why it matters: {b.why}"] if b.why else []
        out += [f"  {x.label}: {_t(x.span)}" for x in b.extras]
        out += [f"  Source: {_t(s.span)}" + (f" [{s.badge}]" if s.badge else "") for s in b.sources]
        if b.tags:
            out.append("  " + " · ".join(f"#{t}" for t in b.tags))
        return out
    return []


def _full_text(f: FullReport) -> list[str]:
    out = [f.subtitle] if f.subtitle else []
    for b in f.top:
        out += _block_text(b)
    for sec in f.sections:
        out += ["", f"### {sec.title}"]
        for b in sec.blocks:
            out += _block_text(b)
    if f.is_empty():
        out.append("The report is empty.")
    return out


# -- digest sections -------------------------------------------------------------

def _digest_tiles(d: Digest) -> list[TileB]:
    return [TileB(t.label, _fmt(t.value), tone=t.tone) for t in d.tiles]


def _item_html(item) -> str:
    title = f'<div style="font-size:15px;font-weight:bold;">{_a(item.title, item.url)}</div>'
    blurb = _p(_e(item.blurb), "font-size:14px;padding-top:2px;") if item.blurb else ""
    return _box(title + blurb, _tc(item.tone))


def _more_text(s: EmailSection) -> str | None:
    return f"+{s.more} more in the task's own report" if s.more > 0 and s.mode in ("digest", "full") else None


def _section_body_html(s: EmailSection) -> str:
    body: list[str] = []
    if s.note:
        body.append(_small(s.note, extra="padding-top:4px;"))
    if s.mode == "link":
        body.append(_p("Full report in the attachment.", f"font-size:14px;color:{MUTED};"))
    elif s.mode == "full" and s.full is not None:
        body.append(_full_html(s.full))
    else:
        d = s.digest
        body.append(_tiles_html(_digest_tiles(d)))
        body += [_box(f'<div style="font-size:14px;">{_e(c)}</div>', TONE_COLORS["warn"]) for c in d.callouts]
        if s.mode in ("digest", "full"):
            body += [_item_html(i) for i in d.items]
        if more := _more_text(s):
            body.append(_small(more, extra="padding-top:6px;"))
    return "".join(body)


def _section_html(s: EmailSection) -> str:
    color = STATUS_COLORS.get(s.status.kind, MUTED)
    return (f'<tr><td bgcolor="{SURFACE}" style="background-color:{SURFACE};color:{TEXT};padding:14px 16px 16px 16px;'
            f'border-top:3px solid {color};"><div style="font-size:18px;font-weight:bold;">'
            f'{_e(s.title[:MAX_HEADING])}</div>{_chip(s.status)}{_section_body_html(s)}</td></tr>'
            f'<tr><td height="10" bgcolor="{BG}" style="background-color:{BG};font-size:0;line-height:0;">&nbsp;</td></tr>')


def _section_text(s: EmailSection) -> list[str]:
    out = ["", f"## {s.title[:MAX_HEADING]}", f"[{s.status.label}]"]
    if s.note:
        out.append(s.note)
    if s.mode == "link":
        return out + ["Full report in the attachment."]
    if s.mode == "full" and s.full is not None:
        return out + _full_text(s.full)
    d = s.digest
    out += [" · ".join(f"{t.label}: {_fmt(t.value)}" for t in d.tiles)] if d.tiles else []
    out += [f"! {c}" for c in d.callouts]
    if s.mode in ("digest", "full"):
        for i in d.items:
            out.append(f"- {i.title}" + (f" — {i.blurb}" if i.blurb else ""))
            if url := http_url(i.url):
                out.append(f"  {url}")
    if more := _more_text(s):
        out.append(more)
    return out


# -- the email -------------------------------------------------------------------

FOOTER = "Also attached as an HTML file, for a desktop browser."


def render_email(title: str, sections: list[EmailSection], *, subtitle: str = "",
                 intro: str | None = None) -> RenderedEmail:
    """HTML + plain-text bodies carrying every section in full: nothing is
    cut to fit a size budget (#119)."""
    title, subtitle = title[:MAX_HEADING], subtitle[:MAX_HEADING]
    head = (f'<tr><td bgcolor="{BG}" style="background-color:{BG};color:{TEXT};border-top:4px solid '
            f'{TONE_COLORS["acc"]};padding:16px 16px 12px 16px;"><div style="font-size:24px;font-weight:bold;">'
            f'{_e(title)}</div>' + (f'<div style="{MONO}font-size:13px;color:{MUTED};">{_e(subtitle)}</div>'
                                    if subtitle else "") + "</td></tr>")
    if intro:
        paras = "".join(f'<div style="padding-bottom:6px;">{_e(p)}</div>' for p in intro.split("\n") if p.strip())
        head += (f'<tr><td bgcolor="{SURFACE}" style="background-color:{SURFACE};color:{TEXT};'
                 f'padding:12px 16px 6px 16px;">{paras}</td></tr>'
                 f'<tr><td height="10" bgcolor="{BG}" style="background-color:{BG};font-size:0;line-height:0;">'
                 "&nbsp;</td></tr>")
    body = "".join(_section_html(s) for s in sections) or (
        f'<tr><td bgcolor="{SURFACE}" style="background-color:{SURFACE};color:{MUTED};padding:12px 16px;">'
        "Nothing to report.</td></tr>")
    foot = (f'<tr><td bgcolor="{BG}" style="background-color:{BG};color:{MUTED};padding:10px 16px 16px 16px;'
            f'{MONO}font-size:12px;">{_e(FOOTER)}</td></tr>')
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark"><meta name="supported-color-schemes" content="light dark">'
        f"<title>{_e(title)}</title></head>"
        f'<body bgcolor="{BG}" style="margin:0;padding:0;background-color:{BG};color:{TEXT};'
        '-webkit-text-size-adjust:100%;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="{BG}"'
        f' style="background-color:{BG};color:{TEXT};"><tr><td align="center">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"'
        f' style="max-width:640px;font-family:{FONT};font-size:15px;line-height:1.45;color:{TEXT};'
        f'word-break:break-word;overflow-wrap:anywhere;">{head}{body}{foot}</table></td></tr></table></body></html>'
    )

    text = [title, "=" * len(title)]
    if subtitle:
        text.append(subtitle)
    if intro:
        text += ["", intro.strip()]
    if not sections:
        text += ["", "Nothing to report."]
    for s in sections:
        text += _section_text(s)
    text += ["", FOOTER]
    return RenderedEmail(page, "\n".join(text) + "\n")
