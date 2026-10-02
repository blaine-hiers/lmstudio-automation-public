"""The dashboard's Errors page (`/errors`): a static shell plus the script
that fills it from `/api/errors`. Every string from a log, summary, subject
or reason is set with `textContent`; nothing untrusted goes through HTML.
"""

from __future__ import annotations

from .report import COLLAPSE_JS, FONTS, THEME, collapse_toggle_html, esc


def nav_html(current: str) -> str:
    """The launcher / errors switch shown in both pages' mastheads."""
    links = (("/", "launcher"), ("/errors", "errors"), ("/capabilities", "capabilities"))
    return '<nav class="dash-nav">' + "".join(
        f'<a href="{href}"{" aria-current=page" if name == current else ""}>{name}</a>' for href, name in links
    ) + "</nav>"


def render_errors_page(token: str, nonce: str, port: int, retention_days: int) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark"><meta name="referrer" content="no-referrer">'
        f'<meta name="lmsa-token" content="{esc(token)}">'
        f"<title>Afterhours errors</title>{FONTS}<style>{THEME.read_text(encoding='utf-8')}</style></head><body>"
        '<div class="hazard"></div>'
        '<main class="wrap">'
        '<header class="masthead"><div class="sticker t-hot">what broke</div>'
        f'<div class="kicker">lmsa · errors</div><h1>Errors</h1>{nav_html("errors")}'
        '<p class="subtitle">Every failed run -- scheduled, from this dashboard, or from a terminal -- newest first, '
        f'with its full output. Logs older than {int(retention_days)} days are pruned.</p>'
        '</header>'
        '<section class="t-hot" id="err-failures-sec"><h2><span class="idx">01 //</span>Failures'
        '<span class="count" id="err-count"></span>'
        '<button class="btn btn-ghost btn-sm" type="button" id="err-refresh">refresh</button>'
        + collapse_toggle_html("err-failures-sec") + '</h2>'
        '<div class="sec-body" id="body-err-failures-sec">'
        '<div class="err-filter" id="err-filter" hidden></div>'
        '<p class="err-note" id="err-note"></p>'
        '<div class="err-list" id="err-list"><p class="err-empty">loading&hellip;</p></div>'
        '</div></section>'
        '<section class="t-acc" id="err-detail-sec" hidden><h2><span class="idx">02 //</span>Detail'
        + collapse_toggle_html("err-detail-sec") + '</h2>'
        '<div class="sec-body" id="body-err-detail-sec">'
        '<div class="console err-detail"><div class="console-bar">'
        '<span class="console-title" id="d-title"></span><span class="console-meta" id="d-meta"></span>'
        '<button class="btn btn-sm" type="button" id="d-rerun" hidden>re-run &#9656;</button></div>'
        '<p class="err-reason-line" id="d-reason"></p>'
        '<ul class="err-summary" id="d-summary" hidden></ul>'
        '<pre class="console-log err-log" id="d-log"></pre>'
        '<div class="console-foot" id="d-foot"></div></div></div></section>'
        '<section class="t-warn" id="err-gaveup-sec"><h2><span class="idx">03 //</span>Gave up'
        '<span class="count" id="gu-count"></span>'
        + collapse_toggle_html("err-gaveup-sec") + '</h2>'
        '<div class="sec-body" id="body-err-gaveup-sec">'
        '<p class="err-note">Messages a Gmail task stopped retrying after its last attempt, as its reports list them.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Task</th><th>Message id</th><th>Subject</th>'
        '<th class="num">Attempts</th><th>Last error</th></tr></thead><tbody id="gu-body"></tbody></table></div>'
        '<p class="err-note" id="gu-note"></p></div></section>'
        f"<footer>// afterhours errors · 127.0.0.1:{int(port)}</footer>"
        f'</main><script nonce="{esc(nonce)}">{COLLAPSE_JS}{ERRORS_JS}</script></body></html>'
    )


ERRORS_JS = r"""
(() => {
  const TOKEN = document.querySelector('meta[name=lmsa-token]').content;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  async function api(url, body) {
    const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LMSA-Token': TOKEN }, body: JSON.stringify(body) };
    const r = await fetch(url, opt);
    const j = await r.json().catch(() => ({ error: r.statusText }));
    return { ok: r.ok, status: r.status, j };
  }
  const SRC_TONE = { scheduled: 't-warn', dashboard: 't-acc', terminal: 't-info' };
  const outTone = o => o === 'skipped' || o === 'running?' ? 't-warn' : 't-hot';
  let data = null, cur = null, filter = new URLSearchParams(location.search).get('task') || '';

  function row(e) {
    const b = el('button', 'err-row' + (e.id === cur ? ' on' : '')); b.type = 'button'; b.dataset.id = e.id;
    b.append(el('span', 'err-when', e.when_label), el('span', 'err-task', e.task),
      el('span', 'tag ' + (SRC_TONE[e.source] || 't-muted'), e.source), el('span', 'tag ' + outTone(e.outcome), e.outcome),
      el('span', 'err-reason', e.reason));
    b.addEventListener('click', () => open(e.id, false));
    return b;
  }
  function render() {
    const rows = data.entries.filter(e => !filter || e.task === filter);
    $('#err-count').textContent = rows.length;
    const f = $('#err-filter'); f.hidden = !filter; f.replaceChildren();
    if (filter) {
      const clear = el('button', 'chip on', 'task: ' + filter + '  ✕'); clear.type = 'button';
      clear.addEventListener('click', () => { filter = ''; history.replaceState(null, '', '/errors' + location.hash); render(); });
      f.append(clear);
    }
    $('#err-note').textContent = data.total > data.entries.length ? `showing the newest ${data.entries.length} of ${data.total} failures` : '';
    $('#err-list').replaceChildren(...(rows.length ? rows.map(row) : [el('p', 'err-empty', filter ? 'no failures for ' + filter : 'nothing has failed — enjoy it')]));
    const gu = data.gave_up;
    $('#gu-count').textContent = gu.total;
    $('#gu-body').replaceChildren(...(gu.items.length ? gu.items.map(i => {
      const tr = el('tr'); tr.append(el('td', '', i.task), el('td', 'mono', i.msg_id), el('td', '', i.subject || '(no subject)'),
        el('td', 'num', i.attempts == null ? '?' : String(i.attempts)), el('td', '', i.error)); return tr;
    }) : [(() => { const tr = el('tr'); const td = el('td', 'err-empty', 'nothing given up on'); td.colSpan = 5; tr.append(td); return tr; })()]));
    $('#gu-note').textContent = gu.total > gu.items.length ? `showing ${gu.items.length} of ${gu.total}` : '';
  }
  async function load() {
    const { ok, j } = await api('/api/errors').catch(() => ({ ok: false, j: {} }));
    if (!ok) { $('#err-list').replaceChildren(el('p', 'err-empty', 'couldn’t load errors: ' + (j.error || 'dashboard unreachable'))); return; }
    data = j; render();
    const want = location.hash.slice(1);
    if (want && want !== cur && data.entries.some(e => e.id === want)) open(want, false);
  }
  $('#err-refresh').addEventListener('click', load);

  function renderLog(log) {
    const pre = $('#d-log'), lines = log.text.split('\n'), frag = document.createDocumentFragment();
    let i = 0;
    for (const [a, b] of log.spans) {
      if (a > i) frag.append(lines.slice(i, a).join('\n') + '\n');
      const m = el('mark', 'tb', lines.slice(a, b + 1).join('\n')); frag.append(m, '\n'); i = b + 1;
    }
    if (i < lines.length) frag.append(lines.slice(i).join('\n'));
    pre.replaceChildren(frag); pre.hidden = false;
    pre.scrollTop = pre.scrollHeight;
  }
  const kb = n => n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(1) + ' KB' : (n / 1048576).toFixed(1) + ' MB';

  let armed = 0, detail = null;
  const rerun = $('#d-rerun');
  const disarm = () => { armed = 0; rerun.classList.remove('armed'); rerun.textContent = 're-run ▸'; };
  async function open(id, all) {
    cur = id; disarm();
    document.querySelectorAll('.err-row').forEach(b => b.classList.toggle('on', b.dataset.id === id));
    history.replaceState(null, '', location.pathname + location.search + '#' + id);
    const sec = $('#err-detail-sec'); sec.hidden = false;
    // Picking a failure always shows it: expand Detail if it was collapsed
    // (through its own toggle, so aria-expanded and the remembered state follow).
    const detailToggle = sec.querySelector('[data-collapse]');
    if (detailToggle && detailToggle.getAttribute('aria-expanded') === 'false') detailToggle.click();
    $('#d-title').textContent = '…'; $('#d-meta').textContent = 'loading';
    const { ok, j } = await api(`/api/errors/${encodeURIComponent(id)}${all ? '?all=1' : ''}`).catch(() => ({ ok: false, j: {} }));
    if (id !== cur) return;
    if (!ok) { $('#d-meta').textContent = j.error || 'couldn’t load this failure'; return; }
    detail = j;
    $('#d-title').textContent = j.task;
    $('#d-meta').textContent = [j.when_label, j.source, j.outcome].join(' · ');
    $('#d-reason').textContent = j.reason;
    const sum = $('#d-summary'); sum.replaceChildren(...j.summary.map(s => el('li', '', s))); sum.hidden = !j.summary.length;
    const foot = $('#d-foot'); foot.replaceChildren();
    if (j.log) {
      renderLog(j.log);
      foot.append(el('span', 'console-meta', (j.log.truncated ? `last ${j.log.shown_lines} lines of ` : 'all of ') + kb(j.log.size)
        + (j.log.spans.length ? ` · ${j.log.spans.length} traceback(s) highlighted` : '')));
      if (j.log.truncated && !all) {
        const more = el('button', 'btn btn-ghost btn-sm', 'show all'); more.type = 'button';
        more.addEventListener('click', () => open(id, true)); foot.append(more);
      }
    } else {
      $('#d-log').hidden = true;
      foot.append(el('span', 'console-meta', j.log_note));
    }
    if (j.report) { const a = el('a', 'btn btn-ghost btn-sm', 'open report ▸'); a.href = j.report; a.target = '_blank'; a.rel = 'noopener'; foot.append(a); }
    rerun.hidden = !j.rerun;
    foot.append(el('span', 'launcher-msg', ''));
    sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  rerun.addEventListener('click', async () => {
    if (!detail || !detail.rerun) return;
    const msg = $('#d-foot .launcher-msg'), confirm = armed > Date.now();
    rerun.disabled = true;
    const { ok, status, j } = await api('/api/run', { task: detail.rerun.task, fields: detail.rerun.fields, extra: '', confirm })
      .catch(() => ({ ok: false, j: { error: 'dashboard unreachable' } }));
    rerun.disabled = false; disarm();
    if (status === 428) {
      armed = Date.now() + 8000; rerun.classList.add('armed'); rerun.textContent = 'confirm — this writes';
      msg.textContent = j.error; msg.className = 'launcher-msg hot';
      setTimeout(() => { if (armed && armed <= Date.now()) disarm(); }, 8100); return;
    }
    if (status === 403) { msg.textContent = 'the dashboard restarted — reload the page'; msg.className = 'launcher-msg err'; return; }
    if (!ok) { msg.textContent = j.error || 'failed'; msg.className = 'launcher-msg err'; return; }
    msg.replaceChildren((j.warning ? '⚠ ' + j.warning + ' · ' : 'launched · '));
    const a = el('a', '', 'watch it in the console'); a.href = '/#run-' + j.id; msg.append(a);
    msg.className = 'launcher-msg' + (j.warning ? ' hot' : ' ok');
  });
  load();
})();
"""
