"""The dashboard's Capabilities page (`/capabilities`, and the static
`output/capabilities.html` report written by `python -m tasks.dashboard
--capabilities`): everything `tasks/` can do, its options, live SETUP status,
schedule, last run and whether it writes -- one entry per task, grouped like
the launcher (#67).

The dashboard page is a static shell filled server-side (every task fact is
already in hand -- nothing here is worth a round trip): the only script is a
client-side search box, nonce'd like every other dashboard page. The static
report has no script at all -- it goes through `lmsa.report` like any other
report file, and never gets a `Card` with a `msg_id` (the one thing that
would add one).
"""

from __future__ import annotations

from lmsa.capabilities import Capabilities, TaskCap
from lmsa.report import FONTS, THEME, Card, Report, Tile, esc

from .errors_page import nav_html

SETUP_TONE = {"ok": "acc", "missing": "hot", "warning": "warn", "none": "muted"}


def _cadence_text(t: TaskCap) -> str:
    if not t.cadence:
        return "manual only"
    return f"{t.cadence}" + (f" · {t.model}" if t.model else "")


def _last_run_text(t: TaskCap) -> str:
    lr = t.last_run
    if not lr:
        return "never run"
    when = str(lr.get("started") or "")[:16].replace("T", " ")
    status = lr.get("status") or "?"
    return f"{status} · {when}".strip(" ·")


def _options_text(t: TaskCap) -> str:
    if not t.args:
        return "no options"
    return ", ".join(a.flag for a in t.args)


# ---------------------------------------------------------------------------
# Static report (output/capabilities.html)
# ---------------------------------------------------------------------------

def render_static_report(caps: Capabilities) -> Report:
    """Build the `Report` for `output/capabilities.html`. Pure function of
    the already-gathered `Capabilities` -- callers write it with `.write()`
    or `.render()`."""
    r = Report(
        "Capabilities",
        kicker="lmsa · capabilities",
        subtitle="Everything this repo can do, generated from its own metadata -- never hand-maintained.",
        sticker=f"{len(caps.tasks)} tasks",
        sticker_tone="acc",
    )
    r.tiles([
        Tile("Tasks", len(caps.tasks), tone="acc"),
        Tile("Not set up yet", len(caps.not_ready), tone="hot" if caps.not_ready else "acc"),
        Tile("Briefings", len(caps.briefings), tone="violet"),
    ])

    if caps.not_ready:
        s = r.section("Not set up yet", tone="hot")
        s.table(("Item", "Reason"), ((n.title, n.reason) for n in caps.not_ready))

    groups: dict[str, list[TaskCap]] = {}
    for t in caps.tasks:
        groups.setdefault(t.group, []).append(t)
    for group, members in groups.items():
        s = r.section(group, tone=members[0].tone)
        cards = []
        for t in sorted(members, key=lambda t: t.name):
            tags = [_cadence_text(t), f"setup: {t.setup_state}"]
            if t.writes:
                tags.append("writes")
            if t.new:
                tags.append("new")
            body = (
                f"{t.doc}\n\noptions: {_options_text(t)}\n"
                f"needs: {t.setup_reason}\n"
                f"last run: {_last_run_text(t)}"
            )
            cards.append(Card(
                title=t.name,
                meta=f"python -m tasks.{t.name}",
                body=body,
                tags=tags,
                tone=t.tone,
                href=t.last_run["report_url"] if t.last_run else None,
            ))
        s.cards(cards)

    if caps.briefings:
        s = r.section("Briefings", tone="violet")
        s.table(
            ("Briefing", "Description", "Sources"),
            ((b.title, b.description or b.warning or "(no description)", b.sources) for b in caps.briefings),
        )
    return r


# ---------------------------------------------------------------------------
# Dashboard page (/capabilities)
# ---------------------------------------------------------------------------

def _task_card_html(t: TaskCap) -> str:
    tags = [f'<span class="tag">{esc(_cadence_text(t))}</span>']
    tags.append(f'<span class="tag t-{SETUP_TONE.get(t.setup_state, "muted")}">setup: {esc(t.setup_state)}</span>')
    if t.writes:
        tags.append('<span class="tag t-hot">writes</span>')
    if t.new:
        tags.append('<span class="tag t-hot">new</span>')
    warnings = ("<ul class=\"tile-warnings\">" + "".join(f"<li>{esc(w)}</li>" for w in t.warnings) + "</ul>"
                if t.warnings else "")
    lr = t.last_run
    last_html = "never run"
    if lr:
        when = esc(str(lr.get("started") or "")[:16].replace("T", " "))
        status = esc(str(lr.get("status") or "?"))
        url = lr.get("report_url")
        link = f' &middot; <a href="{esc(url)}" target="_blank" rel="noopener">report</a>' if url else ""
        last_html = f"{status} &middot; {when}{link}"
    haystack = " ".join((t.name, t.summary, t.doc, t.group, t.setup_state, " ".join(a.flag for a in t.args)))
    return (
        f'<article class="launcher cap-card t-{t.tone}" id="cap-{esc(t.name)}" data-task="{esc(t.name)}" '
        f'data-cap-search="{esc(haystack.lower())}">'
        f'<header class="launcher-head"><h3>{esc(t.name)}</h3><div class="tags">{"".join(tags)}</div></header>'
        f'<p class="cap-doc">{esc(t.doc)}</p>{warnings}'
        f'<dl class="cap-meta">'
        f'<dt>run</dt><dd><code>python -m tasks.{esc(t.name)}</code></dd>'
        f'<dt>options</dt><dd>{esc(_options_text(t))}</dd>'
        f'<dt>needs</dt><dd>{esc(t.setup_reason)}</dd>'
        f'<dt>last run</dt><dd>{last_html}</dd>'
        f'</dl></article>'
    )


def _briefing_card_html(b) -> str:
    return (
        f'<article class="launcher cap-card t-violet" data-cap-search="{esc((b.title + " " + b.description).lower())}">'
        f'<header class="launcher-head"><h3>{esc(b.title)}</h3></header>'
        f'<p class="cap-doc">{esc(b.description or b.warning or "(no description)")}</p>'
        f'<dl class="cap-meta"><dt>sources</dt><dd>{b.sources}</dd>'
        f'<dt>config</dt><dd><code>config/briefings/{esc(b.name)}.toml</code></dd></dl></article>'
    )


def render_capabilities_page(caps: Capabilities, nonce: str, port: int) -> str:
    not_ready_html = ""
    if caps.not_ready:
        rows = "".join(
            f'<li><strong>{esc(n.title)}</strong> &mdash; {esc(n.reason)}</li>' for n in caps.not_ready
        )
        not_ready_html = (
            '<section class="t-hot" id="cap-notready-sec"><h2><span class="idx">01 //</span>Not set up yet'
            f'<span class="count">{len(caps.not_ready)}</span></h2>'
            f'<div class="sec-body"><ul class="cap-notready">{rows}</ul>'
            '<p class="err-note">Fix these from the Setup panel on the launcher.</p></div></section>'
        )

    groups: dict[str, list[TaskCap]] = {}
    for t in caps.tasks:
        groups.setdefault(t.group, []).append(t)
    sections, i = [], 2
    for group, members in groups.items():
        cards = "".join(_task_card_html(t) for t in sorted(members, key=lambda t: t.name))
        gid = f"cap-group-{i}"
        sections.append(
            f'<section class="t-{members[0].tone}" id="{gid}"><h2><span class="idx">{i:02d} //</span>{esc(group)}'
            f'<span class="count">{len(members)}</span></h2>'
            f'<div class="sec-body"><div class="launchers cap-list">{cards}</div></div></section>'
        )
        i += 1

    briefings_html = ""
    if caps.briefings:
        cards = "".join(_briefing_card_html(b) for b in caps.briefings)
        briefings_html = (
            f'<section class="t-violet" id="cap-briefings-sec"><h2><span class="idx">{i:02d} //</span>Briefings'
            f'<span class="count">{len(caps.briefings)}</span></h2>'
            f'<div class="sec-body"><div class="launchers cap-list">{cards}</div></div></section>'
        )

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark"><meta name="referrer" content="no-referrer">'
        f"<title>Afterhours capabilities</title>{FONTS}<style>{THEME.read_text(encoding='utf-8')}</style></head><body>"
        '<div class="hazard"></div>'
        '<main class="wrap">'
        '<header class="masthead"><div class="sticker t-acc">what this can do</div>'
        f'<div class="kicker">lmsa · capabilities</div><h1>Capabilities</h1>{nav_html("capabilities")}'
        f'<p class="subtitle">Every module in tasks/, generated from its own docstring, argparse options and '
        f'SETUP block -- {len(caps.tasks)} tasks, {len(caps.briefings)} briefings. Nothing here is hand-maintained.</p>'
        '<div class="cap-search-wrap"><input type="search" id="cap-q" placeholder="search capabilities&hellip;" '
        'aria-label="search capabilities"><span id="cap-q-count" class="cap-q-count"></span></div>'
        '</header>' + not_ready_html + "".join(sections) + briefings_html +
        f"<footer>// afterhours capabilities · 127.0.0.1:{port}</footer>"
        f'</main><script nonce="{esc(nonce)}">{CAPABILITIES_JS}</script></body></html>'
    )


CAPABILITIES_JS = r"""
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const cards = Array.from(document.querySelectorAll('[data-cap-search]'));
  const q = $('#cap-q'), count = $('#cap-q-count');
  function apply() {
    const needle = q.value.trim().toLowerCase();
    let shown = 0;
    for (const c of cards) {
      const hit = !needle || c.dataset.capSearch.includes(needle);
      c.hidden = !hit;
      if (hit) shown++;
    }
    count.textContent = needle ? `${shown} / ${cards.length}` : '';
    document.querySelectorAll('.cap-list, section').forEach(sec => {
      const visible = sec.querySelectorAll ? Array.from(sec.querySelectorAll('[data-cap-search]')) : [];
      if (!visible.length) return;
      sec.hidden = needle ? visible.every(c => c.hidden) : false;
    });
  }
  q.addEventListener('input', apply);
  const want = new URLSearchParams(location.search).get('q');
  if (want) { q.value = want; apply(); }
})();
"""
