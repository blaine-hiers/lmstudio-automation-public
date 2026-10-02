"""Render every Afterhours component with fake data — the living style guide.

    python -m tasks.style_preview
"""

from __future__ import annotations

import base64
from datetime import date, timedelta
from io import BytesIO

from PIL import Image

from lmsa import Card, Link, Marker, Report, Tile
from lmsa import email_report
from lmsa.digest import OK, Digest, SectionStatus
from lmsa.email_render import EmailSection, render_email
from lmsa.report import StackedBar, Story, StoryExtra, StoryQuote, StorySource


def _sample_thumbnail() -> str:
    """A small invented receipt-ish thumbnail, encoded by Pillow here (so
    it's always a valid image) -- stands in for a real attachment thumbnail.
    `Card.image` wants a data: URI *your own code* encoded from decoded
    bytes, never a string taken verbatim from untrusted content."""
    im = Image.new("RGB", (48, 64), (245, 240, 228))
    for y in range(10, 58, 8):  # a few "printed lines"
        for x in range(8, 40):
            im.putpixel((x, y), (120, 110, 100))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


SAMPLE_THUMBNAIL = _sample_thumbnail()


def main() -> None:
    r = Report(
        "Afterhours style guide",
        kicker="lmsa · house style",
        subtitle="Every component a task can emit, rendered with sample data. "
        "If it isn't on this page, add it to lmsa/report.py before a task hand-rolls it.",
        sticker="v0.1",
        source="tasks.style_preview",
        ticker=["3 renewals due", "unread 42", "tokens/s 38.4", "receipts $1,284"],
    )
    r.tiles([
        Tile("Unread", 42, delta="+5", tone="hot", note="since 07:00",
             spark=[31, 38, 29, 44, 40, 52, 42]),
        Tile("Newsletters", 118, delta="-12", tone="acc",
             spark=[140, 135, 128, 130, 122, 119, 118]),
        Tile("Receipts", "$1,284", tone="warn", note="this month"),
        Tile("Tokens/s", 38.4, tone="info", spark=[22, 30, 27, 35, 31, 40, 38.4]),
    ])
    r.section("Top senders", tone="acc").bars(
        [("github.com", 31), ("substack.com", 22), ("amazon.com", 14), ("linkedin.com", 9), ("chase.com", 4)]
    )
    r.section("Action items", tone="hot").cards([
        Card("Renew domain before Oct 3", "registrar · 2d ago", "Auto-renew is off; card on file expired.",
             tags=["deadline", "billing"], tone="hot", ribbon="urgent"),
        Card("Reply to contractor quote", "mike@example.com · 5h ago", "Asked for a yes/no on the revised scope.",
             tags=["reply"], tone="warn"),
        Card("Flight check-in opens", "airline · today", "Check-in opens 24h before departure.",
             tags=["travel"], tone="info", ribbon="new"),
    ])
    r.section("Email check-off", tone="warn").cards([
        Card("Invoice #4411 due Friday", "billing@example.com · 3h ago", "Checkbox: live when opened via the dashboard.",
             tone="warn", msg_id="1790000000000000001"),
    ]).handled_hidden([
        Card("Team lunch moved", "office@example.com · 1d ago", tone="muted",
             msg_id="1790000000000000002", handled=True),
    ])
    dot_png = SAMPLE_THUMBNAIL
    r.section("Attachment thumbnails", tone="acc").cards([
        Card("Hetzner invoice", "2026-09-12 · $41.20", "Extracted from a PDF attachment's text layer.",
             tags=["pdf", "text-layer"], tone="acc", image=dot_png),
        Card("Receipt.jpg", "2026-09-14 · $18.50", "Extracted from a photographed receipt via the vision model.",
             tags=["image", "vision"], tone="info", image=dot_png),
    ])

    r.section("Receipts", tone="warn").table(
        ["Date", "Merchant", "Category", "Amount"],
        [
            ["2026-09-21", "Hetzner", "infra", 41.20],
            ["2026-09-19", "Amazon", "household", 63.97],
            ["2026-09-12", "Anthropic", "api", 212.00],
        ],
    )
    today = date(2026, 9, 24)
    cells = {
        today - timedelta(days=i): (i * 37) % 11
        for i in range(120)
        if (i * 37) % 11 != 3  # a few gaps to show "no data"
    }
    r.section("Activity", tone="acc").heatmap(cells)

    r.section("Unsubscribe hit list", tone="hot").table(
        ["Sender", "Volume", "Unread %", "Unsubscribe"],
        [
            ["Deals Weekly", 41, 92.7, Link("unsubscribe", "https://example.com/unsub")],
            ["Old Receipts", 6, 0.0, "offered"],
            ["Friend", 3, 33.3, "-"],
        ],
    )

    hours = [f"{h:02d}" for h in range(0, 24, 3)]
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grid = [[(r_i * 3 + c_i * 2) % 9 for c_i in range(len(hours))] for r_i in range(len(weekdays))]
    r.section("Send times", tone="violet").grid_heatmap(weekdays, hours, grid)

    r.section("World map · sequential", tone="acc").world_map(
        {"US": 42, "CN": 35, "RU": 28, "UA": 31, "DE": 12, "GB": 18, "IN": 22, "BR": 9,
         "ZA": 6, "AU": 4, "SG": 15, "JP": 14},
        legend="stories", links={"UA": "#geo-ukraine"},
    )
    r.section("World map · diverging", tone="info").world_map(
        {"US": 1.8, "DE": -0.6, "JP": -1.4, "BR": 2.9, "IN": 0.7, "GB": -0.2, "TR": -3.1,
         "AR": 3.4, "ZA": 1.1, "KR": -0.9},
        scale="diverging", legend="day change %",
    )
    r.section("World map · markers", tone="hot").world_map(
        {"JP": 3, "ID": 2, "CL": 1, "TR": 2},
        legend="quakes",
        markers=[Marker(35.7, 139.7, "M6.1 near Tokyo", size=1.6),
                 Marker(-6.2, 106.8, "M5.4 Java", size=1.2),
                 Marker(-33.4, -70.6, "M4.9 Santiago", tone="warn"),
                 Marker(38.4, 38.3, "M5.0 eastern Türkiye", tone="warn")],
    )
    r.section("Ukraine", tone="violet").anchor("geo-ukraine").text(
        "A country on a map with `links` jumps to a section like this one via `Section.anchor()`."
    )

    r.section("Tones", tone="violet").raw(
        '<div class="tags">'
        + "".join(f'<span class="tag t-{t}">{t}</span>' for t in ("acc", "hot", "warn", "info", "violet", "muted"))
        + '</div><p style="margin-top:14px"><span class="live t-acc">pipeline live</span></p>'
    )
    r.section("Buttons & badges", tone="acc").raw(
        '<div class="launcher-actions">'
        '<button class="btn" type="button">run &#9656;</button>'
        '<button class="btn btn-hot" type="button">stop</button>'
        '<button class="btn armed" type="button">confirm &mdash; this writes</button>'
        '<button class="btn btn-ghost btn-sm" type="button">refresh</button>'
        '<span class="badge-writes">writes</span></div>'
    )
    r.section("Collapsible sections", tone="warn").raw(
        '<p style="margin-bottom:14px">Every section and setup card header gets a toggle; a section with '
        'something needing attention keeps a small dot visible while collapsed.</p>'
        '<div class="launcher-actions">'
        '<span class="collapse-attn" title="2 need attention">2</span>'
        '<button class="collapse-toggle" type="button" aria-expanded="true" aria-label="collapse section">'
        '<span class="chevron" aria-hidden="true">&#9662;</span></button>'
        '<button class="collapse-toggle collapsed" type="button" aria-expanded="false" aria-label="expand section">'
        '<span class="chevron" aria-hidden="true">&#9662;</span></button>'
        "</div>"
    )
    r.section("Launcher tiles", tone="info").raw(
        '<div class="launchers">'
        '<article class="launcher t-info"><div class="ribbon">writes</div>'
        '<header class="launcher-head"><h3>labeler</h3><div class="tags">'
        '<span class="tag">daily 02:20</span><span class="tag t-violet">chat</span></div></header>'
        '<p class="launcher-doc">Auto-labeler: classify recent unlabeled mail into configured Gmail labels.</p>'
        '<div class="launcher-last"><span class="dot st-ok"></span>ok · Sep 24 02:21 · <a href="#">report</a></div>'
        '<form class="launcher-form"><details class="options"><summary>Options</summary><div class="fields">'
        '<label class="field"><span class="flag">--days</span><input type="number" placeholder="default"></label>'
        '<label class="field"><span class="flag">--limit</span><input type="number" placeholder="default"></label>'
        '<label class="check t-hot writes"><input type="checkbox"><span class="flag">--apply</span>'
        '<span class="hint">apply labels in Gmail</span><span class="badge-writes">writes</span></label>'
        '</div></details><details class="extra"><summary>extra args</summary><input type="text" placeholder="--flag value"></details>'
        '<div class="launcher-actions"><button class="btn" type="button">run &#9656;</button>'
        '<span class="launcher-msg ok">launched</span></div></form></article>'
        '<article class="launcher t-violet">'
        '<header class="launcher-head"><h3>ask</h3><div class="tags">'
        '<span class="tag t-muted">manual only</span><span class="tag t-violet">chat</span></div></header>'
        '<p class="launcher-doc">Ask a question of your Gmail inbox (semantic search + Q&amp;A).</p>'
        '<div class="launcher-last"><span class="dot st-err"></span>error · Sep 23 21:04</div>'
        '<form class="launcher-form"><details class="options" open data-must-open="1"><summary>Options</summary><div class="fields">'
        '<label class="field wide"><span class="flag">question</span><input type="text" placeholder="required"></label>'
        '</div></details><div class="launcher-actions"><button class="btn" type="button">run &#9656;</button>'
        '<span class="launcher-msg err">ask needs a question</span></div></form></article>'
        "</div>"
    )
    r.section("Setup checklist", tone="warn").raw(
        '<div class="setups">'
        '<article class="launcher setup-item t-info"><header class="launcher-head"><h3>Gmail</h3><div class="tags">'
        '<span class="setup-state st-missing">missing</span></div></header>'
        '<p class="launcher-doc">IMAP login for every Gmail task.</p>'
        '<p class="setup-reason">no Gmail address (LMSA_IMAP_USER); no app password</p>'
        '<form class="setup-form"><div class="fields">'
        '<label class="field"><span class="flag">gmail address</span><input type="text" placeholder="you@gmail.com"></label>'
        '<label class="field"><span class="flag">app password</span><input type="password" placeholder="write-only"></label>'
        '</div><div class="launcher-actions"><button class="btn btn-sm" type="button">save</button>'
        '<button class="btn btn-sm" type="button">test</button><span class="launcher-msg ok">address saved to .env</span></div>'
        '</form></article>'
        '<article class="launcher setup-item t-acc"><header class="launcher-head"><h3>LM Studio</h3><div class="tags">'
        '<span class="setup-state st-warning">warning</span></div></header>'
        '<p class="setup-reason">loaded without a TTL (pinned): qwen/qwen3-14b</p>'
        '<ul class="setup-hints"><li class="cmd"><code>lms get qwen/qwen3-vl-8b</code>'
        '<button class="btn btn-ghost btn-sm" type="button">copy</button></li></ul>'
        '<p class="setup-note">Manual step: set the JIT idle TTL in LM Studio.</p>'
        '<form class="setup-form"><div class="fields"><label class="field"><span class="flag">pinned model</span>'
        '<select><option>qwen/qwen3-14b</option></select></label></div>'
        '<div class="launcher-actions"><button class="btn btn-sm btn-hot" type="button">unload</button></div></form>'
        '</article>'
        '<article class="launcher setup-item wide t-acc"><header class="launcher-head"><h3>Config files</h3><div class="tags">'
        '<span class="setup-state st-ok">ok</span></div></header>'
        '<form class="setup-form"><div class="fields"><label class="field"><span class="flag">file</span>'
        '<select><option>labels</option></select></label><label class="field wide"><span class="flag">contents</span>'
        '<textarea rows="4" spellcheck="false">[[label]]\nname = "Receipts"</textarea></label></div>'
        '<div class="launcher-actions"><button class="btn btn-sm" type="button">load</button>'
        '<button class="btn btn-sm btn-hot" type="button">save</button></div></form>'
        '<pre class="setup-output">saved labels.toml; backup: labels-20260925-010203-000000.toml</pre></article>'
        "</div>"
    )
    r.section("Dashboard tabs & console dock", tone="acc").raw(
        '<p style="margin-bottom:14px">The launcher page groups its sections into tabs (#100); each label '
        'carries an attention badge. The Tasks tab adds group filter chips and a name search above its '
        'sections. Console is a dock pinned to the bottom of every tab, collapsed to a one-line summary.</p>'
        '<div class="tabbar" role="tablist" aria-label="Dashboard sections">'
        '<a href="#overview" class="tab" role="tab" aria-selected="true" tabindex="0">Overview'
        '<span class="tab-badge">2</span></a>'
        '<a href="#tasks" class="tab" role="tab" aria-selected="false" tabindex="-1">Tasks'
        '<span class="tab-badge">1</span></a>'
        '<a href="#inbox" class="tab" role="tab" aria-selected="false" tabindex="-1">Inbox'
        '<span class="tab-badge" hidden>0</span></a>'
        '<a href="#setup" class="tab" role="tab" aria-selected="false" tabindex="-1">Setup'
        '<span class="tab-badge" hidden>0</span></a>'
        '</div>'
        '<div class="task-filters" style="margin-top:18px"><div class="chip-filters" role="group" aria-label="filter by group">'
        '<button type="button" class="chip-filter" aria-pressed="true">Gmail</button>'
        '<button type="button" class="chip-filter" aria-pressed="true">Other</button>'
        '<button type="button" class="chip-filter" aria-pressed="false">Diagnostics &amp; manual</button>'
        '</div><label class="field task-search"><span class="flag">search</span>'
        '<input type="search" placeholder="task name"></label></div>'
        '<div class="console-dock" style="position:static; margin-top:18px; box-shadow:none">'
        '<div class="dock-bar"><button class="collapse-toggle dock-toggle collapsed" type="button" '
        'aria-expanded="false" aria-label="toggle console"><span class="chevron" aria-hidden="true">&#9662;</span></button>'
        '<span class="dock-label">Console</span><span class="console-meta">2 running &middot; last: inbox_brief OK</span></div>'
        '</div>'
    )
    r.section("Console", tone="acc").raw(
        '<div class="console"><div class="console-bar"><span class="console-title live">healthcheck</span>'
        '<span class="console-meta">python -m tasks.healthcheck · 02:30:04 · 3.2s</span>'
        '<button class="btn btn-ghost btn-sm" type="button">clear finished</button>'
        '<button class="btn btn-hot btn-sm" type="button">stop</button></div>'
        '<div class="console-runs"><button class="chip on" type="button">healthcheck</button>'
        '<button class="chip ok" type="button">index</button><button class="chip err" type="button">ask</button></div>'
        '<pre class="console-log" style="height:auto">[lmsa] probing http://localhost:1234/v1\n'
        "models: 3 loaded\nchat  qwen/qwen3-14b  1.84s\nembed nomic-embed-text  0.12s\n</pre>"
        '<div class="console-foot"><span class="tag t-acc">exit 0</span><span class="console-meta">4.1s</span>'
        '<a class="btn btn-ghost btn-sm" href="#">open report &#9656;</a></div></div>'
    )
    r.section("Callouts", tone="info").callout(
        "Local model output is a draft, not a fact. Anything that moves money or sends mail gets a human look first.",
        label="rule",
        tone="info",
    )
    r.section("Dashboard errors", tone="hot").raw(
        '<nav class="dash-nav"><a href="#">launcher</a><a href="#" aria-current="page">errors</a></nav>'
        '<p style="margin:14px 0"><span class="badge-err">error</span></p>'
        '<div class="err-list">'
        '<button class="err-row on" type="button"><span class="err-when">Sep 24 02:21</span>'
        '<span class="err-task">receipts</span><span class="tag t-warn">scheduled</span>'
        '<span class="tag t-hot">error</span><span class="err-reason">exit 1: KeyError: \'total\'</span></button>'
        '<button class="err-row" type="button"><span class="err-when">Sep 23 21:04</span>'
        '<span class="err-task">ask</span><span class="tag t-acc">dashboard</span>'
        '<span class="tag t-hot">stopped</span><span class="err-reason">stopped from the dashboard</span></button>'
        '</div>'
        '<div class="console err-detail" style="margin-top:14px"><div class="console-bar">'
        '<span class="console-title">receipts</span><span class="console-meta">Sep 24 02:21 · scheduled · error</span>'
        '<button class="btn btn-sm" type="button">re-run &#9656;</button></div>'
        '<p class="err-reason-line">exit 1: KeyError: \'total\'</p>'
        '<pre class="console-log err-log">scanning 12 messages\n'
        '<mark class="tb">Traceback (most recent call last):\n  File "tasks/receipts.py", line 88, in main\n'
        '    amount = row["total"]\nKeyError: \'total\'</mark></pre>'
        '<div class="console-foot"><span class="console-meta">all of 1.2 KB · 1 traceback(s) highlighted</span>'
        '<a class="btn btn-ghost btn-sm" href="#">open report &#9656;</a></div></div>'
    )
    r.section("Briefing stories", tone="info").stories([
        Story(
            "Harbour reopens after a week of closures",
            summary="Port authority lifted the closure order; first ships docked overnight.",
            meta="3 sources · 2 h ago · severity 1",
            badge="new", tone="info",
            highlights=["Closure lasted eight days", "Backlog of 40 ships"],
            quotes=[StoryQuote("We expect normal traffic by Friday.", "https://example.com/a", verified=True),
                    StoryQuote("Nobody was told in advance.", "https://example.com/b", verified=False)],
            timeline=[("2026-09-17", "Closure ordered"), ("2026-09-25", "Closure lifted")],
            why="Shipping delays ripple into prices within two weeks.",
            sources=[StorySource("Harbour Gazette", "https://example.com/a", badge="high", tone="acc"),
                     StorySource("Wire", "https://example.com/b", badge="mixed", tone="warn"),
                     Link("Blog", "https://example.com/c")],
            bars=[StackedBar([("left", 1, "info"), ("center", 2, "muted"), ("right", 1, "hot")],
                             label="coverage by lean")],
            extras=[StoryExtra("market odds", "Port strike by Nov: 12% (+3 in 24 h)", "https://example.com/m")],
            anchor="story-sample",
        ),
    ])
    path = r.write("style-preview.html")
    print(path)
    # The same page as the master report's `full` mode puts it in an email
    # body (#119): every component converted by lmsa.email_report/email_render.
    email = render_email("Afterhours style guide · email", [EmailSection(
        "style_preview", SectionStatus(OK), Digest(), mode="full", full=email_report.load(path))],
        subtitle="every component as lmsa.email_render shows it (iOS Gmail / iOS Mail)")
    email_path = path.with_name("style-preview-email.html")
    email_path.write_text(email.html, encoding="utf-8")
    print(email_path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
