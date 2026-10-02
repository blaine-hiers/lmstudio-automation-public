# lmstudio-automation

Scheduled automations and data pulls that run on a **local** model through
LM Studio's OpenAI-compatible server (`http://localhost:1234/v1`). Results come
out as standalone HTML reports in the **Afterhours** house style.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pytest -n auto         # same suite, spread across CPU cores (pytest-xdist)
.venv\Scripts\python -m tasks.healthcheck      # probes the server, writes output/server-healthcheck-*.html
.venv\Scripts\python -m tasks.style_preview    # the living style guide -> output/style-preview.html
```

LM Studio must be serving (Developer tab → Start Server, or `lms server start`).
`LMStudio.ensure_server()` starts it via the `lms` CLI when a task finds it down.

For day-to-day iteration, skip the handful of tests that spin up real
process trees or timers: `python -m pytest -m "not slow"`. Run the full
`python -m pytest` before merging.

### Gmail credentials

`lmsa.sources.gmail.Gmail` reads mail over IMAP with a
[Gmail app password](https://myaccount.google.com/apppasswords). Reads are
read-only (readonly select + `BODY.PEEK`), so nothing is ever marked as read,
and nothing anywhere sends mail. There are three opt-in write paths,
each used by one task and only when you ask for it (by default every one of
these tasks writes a dry-run report of what it *would* do):

- `Gmail.append_draft` — `tasks.reply_drafts --apply` appends reply drafts
  to your Drafts folder (one IMAP APPEND; it never sends them).
- `Gmail.add_labels` — `tasks.labeler --apply` adds your own Gmail labels,
  logging each one first; `Gmail.remove_labels` runs only under
  `tasks.labeler --undo RUN_ID`, which removes exactly the labels that run
  added. Gmail system labels (Inbox, Spam, Trash, ...) are refused.
- `Gmail.archive` / `Gmail.mark_read` — `tasks.inbox_actions --archive-handled
  --apply` / `--mark-read-handled --apply` archive (remove the `\Inbox` label)
  or mark read the emails you checked off (see *Checking emails off*). Each
  message's current labels / flags are read first: one already archived or
  already read (or your own sent mail, never in the inbox) is skipped and
  reported, not written. Every real write is logged under the run id (intent
  before, done / failed right after); `Gmail.unarchive` / `Gmail.mark_unread`
  run only under `tasks.inbox_actions --undo RUN_ID`, which reverses exactly
  the writes that run made -- and a run that changed nothing has no undo.
  The read-then-write is not atomic (IMAP has no compare-and-set): if you
  archive / read a message in Gmail in the instant between the check and the
  write, the run counts it as its own change and `--undo` would reverse it. A
  message that fails 3 times in a row (e.g. gone from All Mail) is no longer
  retried and is reported as "gave up" -- check it in Gmail. These four methods are
  hard-coded to `\Inbox` / `\Seen` and take no label argument, so no other
  system label is reachable through them.

Set `LMSA_IMAP_USER` to the Gmail address, then give it the app password
either way:

```powershell
python -m lmsa.secrets set imap      # stores it in Windows Credential Manager
# or: put LMSA_IMAP_PASSWORD=... in a local .env (gitignored)
python -m lmsa.secrets check imap    # prints present/missing, never the value
```

Resolution order at runtime is `LMSA_IMAP_PASSWORD` first, then the keyring
entry. Nothing about a vault (Doppler, Bitwarden, ...) is baked into this
repo — wiring one of those into either of the above is up to the environment
running it.

## Layout

| Path | What |
|---|---|
| `lmsa/client.py` | `LMStudio`: `chat`, `extract` (JSON-schema structured output), `embed`, `ensure_server` |
| `lmsa/report.py` | `Report` / `Section` / `Tile` / `Card` → one self-contained HTML file |
| `lmsa/notify.py` | `toast`, `slack`, `notify` — Windows toast + Slack webhook alerts |
| `lmsa/handled.py` | Checked-off emails: `seen` / `split` / `mark` / `unmark`, the Inbox panel's open items, logged Gmail archive / mark-read with undo, Gmail filter XML |
| `tasks/inbox_actions.py` | Archive / mark read in Gmail the emails you checked off (dry-run by default, `--undo RUN_ID`) |
| `lmsa/memo.py` | `Memo`: per-item LLM result memory, so snapshot reports only pay for new or changed items |
| `lmsa/prompts.py` | `load`/`prompt_hash`: task prompts as files under `prompts/<task>/<name>.md`, not inline constants |
| `lmsa/web.py` | `Fetcher`: polite, guarded HTTP (robots.txt, per-host delay, conditional-GET cache in `data/web/`, size cap, SSRF guard); `extract_article` |
| `lmsa/feeds.py` | `parse_feed`: RSS 2.0 / Atom / RDF parsing into `FeedItem`s |
| `lmsa/search.py` | `SearXNG`: JSON-API client for a self-hosted SearXNG instance, deduped `SearchResult`s |
| `lmsa/briefing/` | Briefing engine: config validation, source adapter registry (`register`), pipeline (collect → pre-filter → cluster → rank → extract), quote verification |
| `tasks/briefing.py` | Runs `config/briefings/<name>.toml` briefings (`--name`, `--list`); no `--name` runs every enabled briefing that is due |
| `config/briefings/` | One TOML file per briefing; `example.toml` documents every key |
| `lmsa/briefing/adapters/dod_contracts.py` | `dod_contracts` source adapter + report extension: parses defense.gov's daily contracts page into the `dod_contracts` ledger table; `config/briefings/defense.toml` uses it |
| `tasks/go_deep.py` | Go deep: researches one briefing story (`--story BRIEFING:STORY_ID`) or `--topic` further -- sub-questions, SearXNG + fetched pages, cited sections, capped searches/pages/model input/time |
| `lmsa/briefing/sitewatch.py` | Site watch core (#53): fetch, diff vs. last snapshot, judge the diff against a page's plain-English `rule`, optional typed `fields` -- shared by the `pages` briefing adapter and `tasks/site_watch.py` |
| `tasks/site_watch.py` | Standalone site watch report/alerts from `config/site_watch.toml`; same core as `[[sources.pages]]` |
| `lmsa/briefing/adapters/` | Plug-in adapters and extensions: `reliefweb`, `gdelt_doc` (API sources) and `theaters` (conflict theater grouping, ranking, escalation flags, #55) |
| `lmsa/runner.py` | `run(main)` — wraps a task's `main()`, alerting and re-raising on failure |
| `lmsa/theme/afterhours.css` | The house style. Inlined into every report. |
| `lmsa/geo.py` | `normalize_country` (names, codes, aliases, demonyms → ISO alpha-2), `region_of`, `project` (Equal Earth) |
| `lmsa/geo_data/` | Country table, hand-kept aliases, Natural Earth 1:110m outlines as SVG paths (offline) |
| `lmsa/geo_metrics.py` | `geo_metrics` table helpers: `record`/`record_many`/`series`/`latest_date` per-country daily metrics, `zscore` |
| `scripts/build_geo_data.py` | One-off: regenerates `lmsa/geo_data/*.json` from Natural Earth |
| `lmsa/sources/gmail.py` | `Gmail`: read-only IMAP source (`search`, `folders`, `sent`) |
| `lmsa/secrets.py` | Windows Credential Manager / `.env` secret storage (`python -m lmsa.secrets`) |
| `tasks/` | One module per job, each runnable as `python -m tasks.<name>` |
| `tasks/dashboard.py` | Afterhours control: a local page that launches any task by hand (`dashboard.cmd`) |
| `tasks/dashboard_setup.py` | The dashboard's Setup checklist: status and fix for Gmail, Slack, LM Studio, scheduler, venv, config |
| `lmsa/capabilities.py` | `build(dash)`: everything `tasks/` can do, generated from `lmsa.task_meta`, `config/schedule.toml`, live SETUP status and `config/briefings/*.toml` -- never hand-maintained |
| `lmsa/capabilities_page.py` | Renders `/capabilities` (the dashboard page) and the static `output/capabilities.html` report |
| `tasks/doc_inbox.py` | Document inbox auto-filing: title, date, sender, type, tags, due date per PDF/image/.docx, then a templated archive path (`config/doc_inbox.toml`); dry-run, `--apply`, `--undo RUN_ID` |
| `tasks/bookmarks.py` | Chrome/Edge bookmarks + self-sent Gmail links, fetched, tagged (`summary`/`tags`/`type`) and embedded into the `bookmarks` vectors namespace; dead links vs. "couldn't check"; `--query` semantic search |
| `tasks/risk_map.py` | Geopolitical risk map: per-country composite (conflict from GDELT, hazard, markets) vs a 30-day baseline, sub-scores always shown (`config/risk_map.toml`); `--backfill N` |
| `lmsa/markets.py` | Prediction markets (#60): defensive Polymarket/Manifold/Metaculus parsing, daily snapshot storage, 24h/7d movement, country attribution, `geo_metrics` "markets" feed |
| `tasks/markets.py` | Daily prediction-markets snapshot collector; model "none" (`config/markets.toml`) |
| `lmsa/briefing/adapters/markets.py` | Briefing extension `"markets"` (opt-in): matches market snapshots to stories by embedding cosine similarity, shows odds + 24h/7d movement on the card and a "biggest movers" section -- always labelled market odds, not a forecast |
| `tasks/related_notes.py` | Weekly related/near-duplicate documents from `tasks.doc_index`'s embeddings, grouped by recently changed document; optional memoized "why related" one-liner (`--explain`) |
| `tasks/disasters.py` | Disaster fusion report: GDACS + USGS earthquakes + NASA EONET (+ optional FIRMS) merged and deduped, per-country `hazard` metric for the risk map (`config/disasters.toml`) |
| `lmsa/gate.py` | `busy_reasons()`: is the PC quiet? (fullscreen, GPU, input idle, process denylist) |
| `lmsa/models.py` | `ModelManager`: load/unload models on demand through the `lms` CLI |
| `lmsa/voice.py` | Speech-to-text seam over faster-whisper (optional `[voice]` extra) + Setup probes |
| `tasks/voice_memos.py` | Voice memos: transcribe recordings, then summary / decisions / action items per recording |
| `tasks/master_report.py` | Master Report (#103): one daily email bundling several reports (`config/master_report.toml`); dry-run by default, `--send`, `--preview`, `--name` |
| `lmsa/digest.py` | The digest contract: `RunRow`, `Digest`, `DigestItem`, `normalize`, `section_status` -- how a task appears inside a master report |
| `lmsa/email_render.py` | Email-safe HTML + plain text for the master report (inline styles, tables, http-only links, mobile-first for iOS Gmail/Mail, never trimmed) |
| `lmsa/email_report.py` | Reads a saved `lmsa.report` page back as email blocks (cards, stories, tables, bars, ...; charts dropped with a note) -- the master report's `full` mode |
| `config/schedule.toml` | Per-task cadence, deadline and model for the dispatcher, plus the busy thresholds |
| `scheduler/register-task.ps1` | Registers the dispatcher (every 30 min) under Task Scheduler `\lmsa\` |
| `tasks/osint_air_sample.py` | Samples adsb.lol's public `/v2/mil` feed every dispatcher tick into compact, bucketed rows; model "none" |
| `tasks/osint_movements.py` | Daily OSINT movements report: air traffic by type class/region with a 30-day baseline anomaly check, plus USNI Fleet Tracker ship positions (`config/osint_air.toml`) |
| `lmsa/vectors.py` | `embed_and_upsert` / `search` / `top_k`: generic cosine top-k over the `embeddings` table, shared by `doc_index`/`doc_search` and `photo_index` |
| `tasks/doc_index.py` | Semantic index over `~/Documents` (`.md`/`.txt`/`.pdf`/`.docx`), chunked and embedded into the `docs` namespace |
| `tasks/doc_search.py` | Semantic search over `doc_index`'s chunks (`--answer` for a cited answer) or, with `--photos`, over `photo_index`'s photo library |
| `tasks/photo_index.py` | Immich-style photo captioning: vision model caption/tags/on-image text + EXIF date/GPS per photo under `LMSA_PHOTOS_DIR`, embedded into the `photos` namespace; read-only, capped `--max-new-per-run` |
| `tasks/ask.py` | Cited Q&A over indexed sources; `--scope gmail\|briefings\|docs\|markets\|contracts\|all` (default `gmail`, unchanged), `--since DATE`; a new scope registers with `register_scope` without touching the dispatch code. `markets` finds only snapshots embedded by a briefing that runs the `markets` extension (e.g. `conflicts.toml`, once enabled) -- the scheduled `tasks/markets.py` collector doesn't embed. `contracts` searches the `dod_contracts` ledger's own embeddings (written by the adapter itself, no report extension needed) and cites the defense.gov source URL |
| `output/` | Generated reports (gitignored) |

## Running tasks by hand: the dashboard

```powershell
.\dashboard.cmd                                   # or: .venv\Scripts\python -m tasks.dashboard
.venv\Scripts\python -m tasks.dashboard --port 8765 --no-browser
```

`dashboard.cmd` (pin it to the taskbar or desktop) starts a small local web
page, **Afterhours control**, and opens it in the default browser. It lists
every module in `tasks/` as a launcher tile, grouped Gmail / Other /
Diagnostics & manual / Scheduler, with its first docstring line, its cadence
and model from `config/schedule.toml` ("manual only" when unscheduled; tasks
unscheduled on purpose are listed in `config/dashboard.toml`), and its
last run from the `runs` table with a link to that report.

Each tile has the task's own options (numbers, checkboxes, the `ask` question
or `doc_search` query) plus a free-form **extra args** box (quotes group,
backslashes stay literal, so Windows paths work). **Run** starts
`python -m tasks.<name>` as a subprocess; its output streams into the console
panel and is kept in `data/dashboard/logs/`. When it exits you get the exit
code, the duration and a link to the report it wrote. **Stop** kills the task's
whole process tree. One live run per task; launching a second chat/vision task
while one is running gets a warning (10 GB of VRAM fits one model), not a block.

Flags that write outside the report -- `labeler --apply` / `--undo`,
`reply_drafts --apply` / `--retry`, `file_sorter --apply` / `--undo`,
`doc_inbox --apply` / `--undo`, `inbox_actions --apply` / `--undo` -- are
marked **WRITES** and need a second click to confirm (also when typed into
extra args). The status strip shows whether the LM Studio server answers, the
loaded models (`lms ps`), the gate's busy reasons and today's run count; it
refreshes when you ask. Reports open through the dashboard at `/reports/...`.

### Checking emails off: the Inbox panel

Email cards in the inbox brief, follow-up radar, deadlines, reply drafts and
phish check reports carry a **handled** checkbox. Checking one off stores it
locally (`lmsa.handled`, table `handled`) and later runs of those tasks skip
it -- they don't send it to the model, and show it only behind an
"N handled hidden" toggle (click to show, uncheck to bring it back).

Reports are static files, so the checkbox only works when the report is
opened through the dashboard (`/reports/...`). Opened from disk, the boxes are
disabled with the tooltip "open via the dashboard to check off". On a
dashboard-served report a small inline script fetches a per-launch token from
`/api/handled/token` and posts to `/api/handled`: the token is never written
into a report file, it is good for that one endpoint only (not for launching
tasks), and the usual Host and Origin checks apply. Reports are still served
sandboxed; the CSP allows exactly that one script, by hash. A check-off is
only accepted for a message some report actually showed.

The dashboard's **Inbox** panel lists the open emails from the latest inbox
brief / follow-up radar / deadlines runs with the same check-off (and a
"recently checked off" list to undo one). Its **also archive in Gmail** and
**also mark read** buttons launch `tasks.inbox_actions --apply` for all
checked-off emails not yet archived / marked read -- confirm-twice like any
write, logged per run, undone with `python -m tasks.inbox_actions --undo
RUN_ID` (unchecking an email locally never touches Gmail).

When the same sender has been archived 3+ times in 30 days, the panel offers
a Gmail filter to skip the inbox for them (never for your own address). Gmail filters can't be created over
IMAP, so this is a `mailFilters.xml` download (also written to
`output/gmail-filters.xml` by `inbox_actions`) that you import yourself:
Gmail Settings -> Filters and Blocked Addresses -> Import filters.

### Capabilities

`/capabilities` (linked from the header) is a generated catalogue of
everything `tasks/` can do -- never a hand-maintained list. One entry per
task, grouped like the launcher, built from its whole docstring (not just
the summary line), its parsed argparse options, its `SETUP` needs with their
live status (the same probe the Setup panel runs), its cadence from
`config/schedule.toml`, its last run and report link, and whether it writes.
A "not set up yet" summary at the top comes straight from the Setup
checklist, and a search box filters the list client-side. Briefings from the
briefing engine (`config/briefings/*.toml`) are listed individually with
their description and source count when that directory exists.

`python -m tasks.dashboard --capabilities` writes the same catalogue to
`output/capabilities.html` as a static report (through `lmsa.report`, like
any other report file -- no script) and exits without starting the server.

### Setup checklist

The **Setup** panel (`tasks/dashboard_setup.py`) lists what isn't set up yet,
each item with a live status (ok / missing / warning and why) and the action
that fixes it, so a fresh machine can be set up without a terminal:

- **Gmail** — address (saved to the local `.env`) and app password (saved to
  Windows Credential Manager; the box is write-only and never echoed). **Test**
  runs `tasks.gmail_probe` through the launcher.
- **Slack alerts** (optional) — the incoming-webhook URL, stored in the keyring
  like the app password (`python -m lmsa.secrets set slack` does the same; an
  `LMSA_SLACK_WEBHOOK` env var still wins), plus **Send test alert**.
- **LM Studio** — server reachable, the configured chat / vision / embed models
  downloaded (`lms ls`; a missing one gets a copyable `lms get ...`), and a
  warning for any model loaded without a TTL, with a confirmed **Unload**. The
  JIT idle timeout is a manual step in LM Studio, shown as text. "Pinned" is
  read from the `ttlMs` field of `lms ps --json` (`null` = no TTL); if your
  `lms` doesn't report that field, the check says "pinned check unknown"
  instead of guessing.
- **Scheduler** — is `\lmsa\dispatch` registered? **Register / update** runs
  `scheduler/register-task.ps1` (confirm first) and shows its output; **Dry run**
  runs `tasks.dispatch --dry-run` through the launcher.
- **Python env** — the venv's packages against `pyproject.toml`, with the
  `pip install -e ".[dev]"` command when they drift.
- **Config files** — load, edit and save `config/{labels,watchlist,front_page,
  doc_search,schedule,dashboard}.toml`. A save (confirm first) is refused unless the file
  parses and passes its task's own loader (`load_schedule`, the labeler's
  reserved-label check, ...); the old file is kept in
  `data/dashboard/config-backups/` and the new one written atomically.

`.env` edits only touch keys listed in `.env.example` (never a secret one),
keep every other line and comment, and apply to tasks launched afterwards
without restarting the dashboard. Below the built-in items, **tool needs**
shows one item per task that declares a `SETUP` block (see *Adding a new
tool*). Tools can also add their own checklist items with
`tasks.dashboard_setup.register(provider)`.

The page is local-only: it binds `127.0.0.1`, refuses other `Host` headers and
cross-origin POSTs, needs a per-launch token on every POST, and only launches
modules it found in `tasks/`. The dispatcher doesn't depend on it: Task
Scheduler still runs only `tasks.dispatch`.

### Adding a new tool

1. **Drop a module in `tasks/`** with a `main()`, an argparse parser and the
   usual `if __name__ == "__main__"` guard. The dashboard finds it on the next
   page load and derives the tile's options from its `add_argument(...)`
   calls -- flag, `type=int`, `action="store_true"`, `help`, `default`,
   `metavar`, positionals. It reads the file with `ast` and never imports it,
   so only literal arguments count; anything else is skipped and shown as a
   warning on the tile. (A task with a curated entry in `FIELDS` in
   `tasks/dashboard.py` keeps that instead.)
2. **Optionally declare what it needs and what writes.** A module-level
   literal dict, all keys optional:

   ```python
   SETUP = {
       "secrets": ["imap"],                 # names from lmsa.secrets (imap, slack)
       "env": ["LMSA_IMAP_USER"],           # keys listed in .env.example
       "config": ["config/labels.toml"],    # files under config/
       "commands": ["gh"],                  # programs that must be on PATH
       "model": "chat",                     # chat | vision | embed
       "searxng": True,                     # needs the SearXNG web search instance
   }
   ```

   The Setup page shows each need's status (never a secret's value) and a box
   to fill in a missing secret or `.env` key; unknown names are listed as
   warnings, not errors. A flag named `--apply`, `--undo`, `--retry`,
   `--send`, `--delete` or `--force`, or whose help starts with the marker
   `[writes]` (`help="[writes] post the digest to Slack"`), is a write flag:
   the tile marks it and a run using it -- also typed into extra args,
   abbreviated or as `--flag=value` -- asks for a confirm. In extra args
   the six standard names always do, for every task, even a flag the tile
   doesn't offer or a module the dashboard couldn't read.
3. **Open the dashboard to set its schedule.** A task that isn't in
   `config/schedule.toml` and isn't marked manual-only is flagged **NEW**, with
   a banner at the top. Its tile has an **Add to schedule** form (cadence,
   deadline hours, model, timeout -- only values the dispatcher accepts) that
   appends a `[tasks.<name>]` table to `schedule.toml` after a confirm,
   keeping the rest of the file byte for byte; or **mark manual-only**, which
   appends `[manual.<name>]` to `config/dashboard.toml`. Both go through the
   config editor's save path: validated first (`load_schedule` for the
   schedule), backed up, written atomically.

### Errors

The **errors** link (`/errors`) lists every failed run, newest first: `runs`
rows with status `error` (any task, from anywhere), scheduled runs that errored
or were skipped (`dispatch_runs`, with their reason), and this dashboard's own
runs that exited non-zero or were stopped. Each row says where it ran --
scheduled, dashboard or terminal -- the outcome and a one-line reason; a `runs`
row that happened inside a failed scheduled or dashboard run is folded into
that run (one row per run, the nearest); any other stands on its own.
Click one for its full output: the last 500 lines of its log (**show all** for
more), with Python tracebacks highlighted, the report link if there is one,
and **Re-run**, which goes through the launcher with the tile's default options
and the same confirm rules. A terminal run has no log, only its summary.

A launcher tile whose most recent run failed carries an **error** badge that
opens the list filtered to that task. Below the failures, **Gave up** lists
the messages each Gmail task stopped retrying after three attempts (ids,
subjects and last error, as the reports show them -- never bodies).

Dashboard logs (`data/dashboard/logs/`) and dispatcher logs
(`data/dispatch/logs/`) older than 30 days are pruned -- set
`LMSA_LOG_RETENTION_DAYS` to change that -- and a log a live run is writing is
never touched. The dashboard keeps its last 200 finished runs in memory and
records every finished run in the `dashboard_runs` table, so its failures
(with their logs) are still listed after a restart; those records are pruned
with their logs.

## Prompts as files

Every task's system/user prompt lives under `prompts/<task>/<name>.md`
(plain UTF-8 text, one paragraph) instead of an inline string constant, via
`lmsa.prompts`:

```python
from lmsa.prompts import load, prompt_hash

EXTRACT_SYSTEM = load("travel/extract_system")            # prompts/travel/extract_system.md
version = memo_version(MEMO_VERSION, prompt_hash("travel/extract_system"), SCHEMA, model_id(lm))
```

`load(name, **vars)` reads `prompts/<name>.md` and, if any `**vars` are
given, substitutes `{var}` placeholders (a placeholder with no matching var
raises, naming the file); with no `**vars` the file's text is returned
verbatim, so a prompt containing literal braces is never mistaken for a
template. `prompt_hash(name)` is a short hash of the file's text (line endings normalized) --
fold it into a task's `memo_version(...)` (see below) so editing a prompt
file invalidates that task's cached LLM results. `prompts/shootout/` (the
model shootout's own prompt library, parsed by `tasks.shootout.load_prompts`)
and `prompts/tier_rubric.md` predate this module and are unrelated to it.

## Document inbox

```powershell
.venv\Scripts\python -m tasks.doc_inbox                  # dry run: report of proposed filing
.venv\Scripts\python -m tasks.doc_inbox --apply          # file them; prints the RUN_ID
.venv\Scripts\python -m tasks.doc_inbox --undo RUN_ID    # put that run's files back
```

`tasks.doc_inbox` is a paperless-style filer for `LMSA_DOC_INBOX_DIR`. For each
PDF, image or .docx directly in the inbox (not recursive; files younger than
`min_age_minutes` are left alone) it reads the text layer (pypdf /
python-docx) -- or, for a scanned PDF or a photo, sends the first pages to the
vision model through the same readers `tasks.attachments` uses -- and
extracts `title`, `doc_date`, `correspondent`, `doc_type`, `tags`, `amount`
and `due_date`. Text sent to the model is capped at 6000 characters and the
report marks every document that was cut.

The destination comes from `template` in `config/doc_inbox.toml` (default
`<archive>/<doc_type>/<YYYY>/<YYYY-MM-DD> <correspondent> - <title>.<ext>`,
archive default `<inbox>/Filed`; a relative `archive_dir` is relative to the
inbox, and both are logged as absolute paths). Every component is sanitised (no
separators, `..` or device names) and the result must stay inside the
archive; a missing or malformed date falls back to the file's own date and
is marked in the report.

Safety is `file_sorter`'s: dry-run by default; `--apply` moves by rename only
(never copies, deletes or overwrites; collisions get `-2`, `-3`, ...
re-checked at move time) and logs each move to `data/doc_inbox/<RUN_ID>.jsonl`
before making it, then records whether the move completed (folders are only
created after the resolved destination is confirmed inside the archive, so a
planted junction can't redirect anything). `--undo RUN_ID` acts only on
completed moves and moves that run's files back newest-first,
never overwriting an occupied original path, skipping any file that has
changed or moved since, refusing log entries outside that run's inbox and
archive, and printing everything it couldn't restore. Due dates still ahead
are written to `output/doc_inbox-due-*.ics`, the same calendar format
`tasks.deadlines` writes.

## Related notes

```powershell
.venv\Scripts\python -m tasks.related_notes                       # cosine only
.venv\Scripts\python -m tasks.related_notes --explain              # + one-line "why", memoized
```

`tasks.related_notes` reuses `tasks.doc_index`'s embeddings to surface, for
every document changed in the last `--days` (default 7) days, the other
indexed documents it's related to. Relatedness is document-level: each
document's chunk vectors are mean-pooled into one normalized vector, so the
comparison is one cosine score per document pair, not per chunk pair. The
"changed in the last N days" signal is the file's own mtime, the only
timestamp `doc_index` already keeps per document -- there's no separate
"indexed at" field to add.

Only recently changed documents seed a comparison (an "anchor"); a document
that hasn't changed can be listed as a match but never starts one. Anchors
are capped at `--anchor-cap` (default 25, oldest-changed dropped first) so a
run's cost stays bounded no matter how many documents changed at once; a
capped run says so in the report. For each anchor, the top `--k` (default 5)
matches scoring at or above `--min-score` (default 0.75) are "related";
anything at or above `--dup-score` (default 0.95) is a near-duplicate,
listed once in its own section regardless of which side is the anchor.

`--explain` asks the chat model for one plain-text sentence on why a pair is
related, grounded in each document's first chunk only. The answer is
memoized per unordered document pair (`lmsa.memo`), so the same pair
appearing under two anchors, or as both "related" and a near-duplicate,
costs one model call, not several. Off by default -- the cosine table alone
is a complete report.

## Photo search

```powershell
.venv\Scripts\python -m tasks.photo_index                      # caption new/changed photos under LMSA_PHOTOS_DIR
.venv\Scripts\python -m tasks.doc_search "birthday cake" --photos --k 6
```

`tasks.photo_index` walks `LMSA_PHOTOS_DIR` (default `~/Pictures`), and for
each new or changed photo: reads its EXIF date/GPS with Pillow, downscales it
with the same vision-safe re-encode `tasks.attachments` uses, and asks the
vision model for a caption, search tags, and any legible on-image text. The
result is embedded (namespace `photos`) so `tasks.doc_search --photos` can
find it later. Read-only over the library -- nothing is written next to a
source photo, ever. New/changed photos are capped at `--max-new-per-run`
(default 100) per run so a large or freshly-imported library can't consume
the whole overnight window; the rest wait for the next run. A file Pillow
can't decode (HEIC without a HEIF plugin, a corrupt image) is skipped and
listed, never a crash. A path whose photo is gone from disk has its row and
embedding removed.

GPS is personal location data: the full coordinate is stored, but
`doc_search --photos`'s report only ever shows a coarse place -- the country
name when it can be resolved (`lmsa.geo.nearest_country`: point-in-polygon
against the same offline 1:110m outlines `Section.world_map` draws, snapping
to the nearest coastline within ~50 km when the point falls just outside a
simplified boundary), else the coordinate rounded to 1 decimal degree
(~11 km) -- and never plots it on a map or shows the precise value. A
micro-state too small to have its own 1:110m outline (Singapore, Vatican
City, Monaco, ...) resolves to whatever country's polygon actually contains
or borders that point instead of to itself -- a known limit of an offline,
outline-based lookup. Search result thumbnails are re-read from the source
file and downscaled on the fly
(capped at `PHOTO_THUMB_MAX_DIM`px, JPEG quality `PHOTO_THUMB_QUALITY`),
never the original bytes.

## Memory between runs

Snapshot reports (`inbox_brief`, `reply_drafts`, `labeler`, `phish_check`,
`travel`, `deadlines`, `newsletter_zine`, `followup_radar`, `pretriage`,
`file_sorter`) keep showing everything in their window, but they remember the
model's answer per item in the `llm_memo` table of `data/lmsa.sqlite`
(`lmsa/memo.py`). A re-run only calls the model for new or changed items:

- **Keys** change whenever the model's input does: the Gmail msg_id for mail
  tasks; thread + the latest sent msg_id for `followup_radar`;
  `repo#number@updatedAt` for `pretriage`; path + size + mtime for
  `file_sorter`. `inbox_brief` and `reply_drafts` share one "triage" memory,
  so a message either one classified costs the other nothing.
- **Versions**: each answer is stored under a hash of the prompt, schema,
  model id and a per-task `MEMO_VERSION`. Edit a prompt or switch models and
  every old answer is ignored automatically.
- Only successful, valid answers are stored; a failed or malformed call is
  retried next run. Only the model's structured answer is stored -- never a
  message body or a secret.
- `--fresh` on any of those tasks ignores the memory and asks the model again
  (the new answers are still stored). It isn't a write flag, so the dashboard
  runs it without a confirm.
- Each report's footer says `Memory: N reused, M new LLM calls` and when the
  task last succeeded; `memo_hits` / `memo_misses` go into the run summary.
- Rows not reused for 180 days are pruned, at most once a day, by whichever
  task runs next.

## World maps and country tagging

`lmsa.geo` tags text with countries and `Section.world_map` draws them, all
offline -- no tiles, no network, the map is inline SVG in the report file.

```python
from lmsa import Marker, Report, geo

geo.normalize_country("Britain")   # "GB"; also "UK", "Ukrainian" -> "UA", "DPRK" -> "KP"
#   pass extracted country mentions or UPPERCASE codes: "in", "it", "us" and
#   acronyms like "IT", "AI", "TV" are not countries (use geo.country("IT"))
geo.region_of("GB")                # "Europe"   (subregion_of -> "Northern Europe")

r = Report("World desk")
r.section("Coverage").world_map(
    {"UA": 12, "US": 7, "SG": 3},            # alpha-2 -> value, shaded in the section tone
    legend="stories",                        # caption, or False to hide
    links={"UA": "#country-ua"},             # click a country to jump to a section
    markers=[Marker(50.45, 30.52, "Kyiv", tone="hot")],  # point events, lat/lon
)
r.section("Ukraine").anchor("country-ua").text("...")
```

- `scale="sequential"` shades 5 steps of the section tone over min..max;
  `scale="diverging"` goes blue (below 0) to pink (above 0), symmetric.
- Countries too small for the 1:110m outlines (Singapore, Malta, most
  islands) render as a dot at their label point.
- Unknown codes, non-numeric values and markers without coordinates are
  skipped and counted in a note under the map. Names, labels and hrefs are
  escaped / passed through `safe_href`.
- **Size:** each map adds about 77 KB to the report (the outlines are
  inlined per map; `lmsa/geo_data/world_110m.json` is 63 KB).
- The data is Natural Earth admin-0 (public domain): 1:110m outlines
  projected with Equal Earth and simplified; the country table comes from
  the 1:50m layer. `python scripts/build_geo_data.py --download` rebuilds
  both files; aliases and demonyms live in `lmsa/geo_data/aliases.json` and
  are edited by hand.

## Geopolitical risk map

`python -m tasks.risk_map` (daily, no model) shades a world map by each
country's **composite stress index** -- how far today sits from that
country's own trailing 30-day normal -- and never shows the number without
its parts:

- **Sub-scores**, each a z-score against the previous 30 days:
  `conflict` = mean(z of GDELT conflict-theme coverage volume, minus z of its
  tone), `hazard` and `markets` = z of the `geo_metrics` rows other tasks
  write under those metric names (disasters, prediction markets).
- **Composite** = weighted mean of the sub-scores the country actually has
  (weights in `config/risk_map.toml`). A missing sub-score is shown as `—`
  and its weight is left out; it is never counted as zero. A metric older
  than `max_age_days` is `stale` and left out; a sub-score from an earlier
  day than the report shows its age (`+1.20 (6d old)`).
- **Flat baselines**: the baseline std is floored at the larger of a tiny
  absolute minimum per metric (`[baseline.min_std]`; volume `1e-5`, well
  below GDELT's real share-of-coverage noise at 0.0003-0.01) and
  `min_rel_std` x |mean|. Ordinary day-to-day noise is never flattened, so a
  3x spike on a small-volume country still scores high; an all-zero month
  followed by real coverage scores at the cap instead of blank. |z| is
  clipped at 10 and marked `(capped)`.
- **Report**: one world map (diverging, composite z), a top-movers table with
  every sub-score next to the blend plus the raw volume/tone z, 30-day
  sparklines for the top 10 countries, region tiles with the mean composite
  *and* mean sub-scores, a "Not scored" table giving the reason for every
  other country with data *and* every configured country without any ("not
  fetched yet", "fetch failing since <date>", "no coverage returned"), and a
  coverage/method section listing what was fetched, per-country fetch
  errors, and (separately) countries GDELT returned no coverage for.
- **Baseline building**: a metric needs 30 days of history, with at most
  `max_missing_days` (default 5) of them missing, before it scores; until
  then the report says `baseline building (n/30 days)`, and a longer history
  with more holes is a `gap`. `--backfill 31` fills it in one go (GDELT allows
  up to 90). `--no-fetch` re-scores stored data without calling GDELT.

**Source.** The GDELT DOC 2.0 API (`api.gdeltproject.org/api/v2/doc/doc`,
free, no key), modes `TimelineVol` (share of all monitored coverage) and
`TimelineTone`, with the query `(theme:ARMEDCONFLICT OR ...) sourcecountry:<FIPS>`:
conflict-themed coverage by the country's own press -- so a country's
outlets covering a foreign war count for that country, and Palestine (`PS`
-> FIPS `WE`) is West Bank press only (GDELT's Gaza, `GZ`, is not included).
A non-empty answer GDELT returns that isn't a timeline is a per-country
error, not a quiet day. GDELT uses FIPS 10-4
country codes there (Germany is `GM`, Ukraine `UP`), mapped from ISO alpha-2
by the hand-kept table `lmsa/geo_data/fips10_4.json`. Each country costs two
small JSON requests at GDELT's 1 request / 5 s, so a run queries at most
`max_countries_per_run` countries (default list: 60, ~10 min), least recently
fetched successfully first (a failure retries first until it has failed 3
runs in a row, then once per rotation, so permanent failures can't starve
the rest; an empty answer is a successful fetch with no coverage); a longer list rotates, and each fetch re-reads every day since
the country's newest stored value (at least 8, so the timeline is daily), so
rotation leaves no gaps. Reruns upsert: no duplicate rows.

**`geo_metrics`** (`date, country, metric, value`, one row per UTC day,
ISO alpha-2 country, metric) is shared: write a daily per-country level with
`lmsa.geo_metrics.record(conn, day, "UA", "hazard", 3.2)` (higher = more
stress) and the risk map picks it up.

## Disaster fusion report

`python -m tasks.disasters` (daily, no model) fuses GDACS, USGS earthquakes
and NASA EONET (all free, no key) into one deduped list, plots it on a world
map, and writes the risk map's `hazard` sub-score.

- **Sources**: GDACS RSS (`config/disasters.toml`'s `[gdacs] rss_url`, every
  active alert -- earthquakes, cyclones, floods, volcanoes, droughts,
  wildfires), the USGS earthquake GeoJSON summary feed (`[usgs] feed`, e.g.
  `significant_week`), NASA EONET v3 (`[eonet]`, open events from the last
  few days). NASA FIRMS fire detections are optional: set
  `LMSA_FIRMS_MAP_KEY` (`python -m lmsa.secrets set firms`) or it's skipped
  with a note in the report, never an error.
- **Normalized event**: `type` (earthquake/cyclone/flood/volcano/drought/
  wildfire/other), `severity` (a human label -- GDACS alert level, `M5.6`,
  a detection count), `countries` (every ISO alpha-2 a source actually
  attributes it to -- never guessed for a coordinate-only event like a
  quake at sea or an unattributed fire cluster; `country` is the first,
  for display), `lat`/`lon`, `started` (UTC), `url`.
- **Country resolution**: GDACS's `iso3` code(s) through a plain alpha-3
  table (not `normalize_country`, whose denylist would wrongly reject real
  codes like `ARM`/`GEO`) -- an event listing more than one code (a cyclone
  or flood crossing a border) is attributed to every one. USGS's free-text
  `place`, the part after the last comma, is matched against the 50 US
  states + DC (-> `US`) and Puerto Rico/Guam/the US Virgin Islands (-> their
  own code) *before* `normalize_country` is even tried -- a state postal
  code or full name is never handed to `normalize_country`, because several
  collide with real ISO country codes (`CA`=Canada, `PA`=Panama, `GA`=Gabon,
  ...); only a tail that isn't a US place falls through to a foreign country
  name (`"84km SSW of Cauquenes, Chile"` -> `CL`). EONET and FIRMS give
  coordinates only, so their events never carry a country.
- **Cross-source dedupe**: events of the same type within 100 km and 24 h
  (`[dedupe]`) are merged -- most often a quake GDACS and USGS both report,
  or a wildfire GDACS/EONET/FIRMS all see -- keeping the higher-severity
  record's fields, the union of reporting sources (shown as `gdacs+usgs`)
  and the union of countries. "New since last run" is tracked by every raw
  source-event id folded into a merged event, not the merged event's own id
  -- so a rerun where a different source wins the merge (and the surviving
  id changes) doesn't re-flag the same physical event as new.
- **Hazard metric** (`geo_metrics`, `"hazard"`, per country per UTC day): the
  sum of each event's severity weight, credited in full to every country the
  event is attributed to (not split). Weights: GDACS alert level
  green=1/orange=4/red=10; USGS magnitude 1 (`<5`) to 16 (`>=8`); a flat 2
  per open EONET event; a FIRMS fire cluster at 0.2/detection, capped at 5.
  `config/risk_map.toml`'s `[baseline.min_std] hazard` is tuned to that
  scale (a third of the smallest non-zero day).
- **Report**: one world map (markers by type/severity), an events-by-severity
  table (countries listed in full for a multi-country event), a "new since
  last run" table, a **Sources** table (raw items considered vs. events kept
  per source, flagging a non-empty payload that parsed to zero events as
  "unrecognized response" -- distinct from a genuinely empty feed), and a
  coverage/method section with the exact weight table and any fetch errors.
  A FIRMS fetch failure never leaks `LMSA_FIRMS_MAP_KEY` (it rides in the
  URL, and `lmsa.web`'s exceptions include the URL) -- it's redacted to
  `***` before it reaches stats, the run summary or the report.

## Prediction markets (#60)

`python -m tasks.markets` (daily, model "none") snapshots open geopolitics
markets from three free, keyless public APIs -- Polymarket's Gamma API,
Manifold's search-markets, Metaculus's questions API -- into one row per
market per UTC day (`market_snapshots`, `config/markets.toml` for the
tags/queries per source). Every value shown anywhere is **market odds, not a
forecast by this tool**.

- **Parsing is defensive**: none of the three response shapes could be
  verified live, so each source's parser accepts whatever plausible shape it
  recognizes, silently skips one bad item among many, and only raises
  `UnrecognizedResponse` when a non-empty response contains not one item
  shaped like a market at all -- an ordinarily empty result (no markets match
  right now) is never an error.
- **Country attribution** prefers structured data: an exactly-one-distinct-hit
  among a market's tags wins outright; only with no tag hit is the question
  text scanned, and then a US-state collision ("New Mexico" -> not Mexico,
  "New Jersey" -> not Jersey) is excluded outright, a short list of country
  names that double as a common given name/noun/single-word US state
  ("Turkey", "Chad", "Jordan", "Georgia", ...) needs a geopolitical cue word
  (election, war, president, coup, sanctions, ...) in the same question, and
  a match sitting right next to what looks like a person's name ("Jordan
  Peterson") is dropped. Zero hits or a tie between countries either way
  means no country -- never a guess.
- **The `markets` `geo_metrics` metric** is, per country per day, the largest
  absolute 24h probability move (a 0..1 fraction) across that country's
  attributed, open markets; a country with no computable move that day gets
  no row (never a 0). `tasks.risk_map` folds it in as a sub-score
  (`[baseline.min_std] markets = 0.01` in `config/risk_map.toml`, already
  scaled for this metric's 0..1 range).
- **Matching to briefing stories** happens at briefing render time, not in
  the collector: the briefing extension `"markets"` (opt-in via
  `extensions = ["markets"]` in a `config/briefings/<name>.toml`) embeds each
  snapshotted question with the same embed model the briefing already used
  for its stories -- cached in the `embeddings` table (ns `"markets"`, keyed
  by `<source>:<market_id>` and embed model) so the same question is only
  ever embedded once, not once per briefing per run -- and attaches the
  closest markets above a cosine-similarity threshold to the matching story
  as odds + 24h/7d movement.
- **"Biggest movers (prediction markets)"**: every snapshotted market with a
  computable 24h move, ranked by |24h move|, disclaimer included -- shown as
  its own section in both the risk map (`tasks.risk_map`, straight off
  `market_snapshots`) and any briefing listing the `"markets"` extension;
  "no market data yet" when the collector hasn't run.

## Fetching the web politely

Every task that crawls (feeds, search results, pages) goes through
`lmsa.web.Fetcher` rather than raw `urllib`:

- **robots.txt** is honoured (cached per host per UTC day under
  `data/web/robots/`); a disallowed URL raises `RobotsDenied` and is never
  fetched, and `Crawl-delay` stretches the wait.
- **One request per host every `per_host_delay` seconds** (default 2).
- **HTTP cache** in `data/web/`: the stored ETag / Last-Modified are sent back,
  and a `304` returns the cached body (`from_cache=True`). `get(url, max_age=...)`
  skips the network for a fresh enough copy.
- **Guards**: http/https only; hosts resolving to private, loopback,
  link-local or reserved addresses are refused (`BlockedURL`) -- on every
  redirect hop, with the connection pinned to the checked address -- unless
  the Fetcher is built with `allow_private=True` or the host is in
  `allowed_hosts`. Bodies over `max_bytes` (default 5 MB) raise `TooLarge`.
- Every failure subclasses `FetchError`, so one bad URL is caught and skipped.

`lmsa.feeds.parse_feed` turns RSS 2.0 / Atom / RDF into `FeedItem`s
(HTML-stripped summaries, http(s)-only links, `None` for missing dates), and
`lmsa.web.extract_article` pulls the main text, title, byline, date and
language out of a page with trafilatura (plain tag-stripping as a fallback).
Both return untrusted text: escape it, and truncate it before a model sees it.

## Web search (SearXNG)

Open web search (briefings, go-deep) goes through a self-hosted
[SearXNG](https://docs.searxng.org/) instance rather than a paid search API.
Run one locally:

```powershell
docker run -d -p 8888:8080 --name searxng searxng/searxng
```

Its JSON API is off by default -- turn it on in `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Point the automation at it with `LMSA_SEARXNG_URL` (default
`http://127.0.0.1:8888`), then use it from Python:

```python
from lmsa.search import SearXNG, SearchError

client = SearXNG("http://127.0.0.1:8888")
try:
    results = client.search("harbour reopening", time_range="week", max_results=10)
except SearchError as exc:
    ...  # not reachable, JSON API disabled, or rate-limited -- never a silent []
```

`search()` drops any result whose URL isn't http(s), dedupes the rest by
`lmsa.web.normalize_url`, and caps the list to `max_results` after dedup.
Every failure subclasses `SearchError`: `SearchNotJSON` (JSON API not
enabled -- HTTP 403 on `format=json`, or an HTML-only instance),
`SearchRateLimited` (HTTP 429), `SearchUnavailable` (anything else, wrapping
the underlying `lmsa.web.FetchError`). `LMSA_SEARXNG_URL`'s host is resolved
and checked with `lmsa.web.is_public_ip`: only when it's loopback, private or
link-local (a local or LAN instance -- the operator's own service) does the
client exempt just that host from the SSRF guard (`allowed_hosts`, never
`allow_private=True` for every request) and turn `respect_robots` off for it.
A public `LMSA_SEARXNG_URL` gets neither: robots.txt stays honoured and
nothing is exempted.

The dashboard's Setup checklist has a "Web search (SearXNG)" card that shows
whether the instance is reachable and JSON-enabled, with the `docker run`
command and the `settings.yml` snippet above when it isn't; a task that needs
it declares `SETUP = {"searxng": True, ...}` and gets the same check folded
into its own checklist entry.

### Start on demand

If the machine running SearXNG is also used for other things (gaming, say),
a keep-alive container idling all session is wasted resources. Run SearXNG
inside WSL instead and let `lmsa.search.SearXNG` bring it up only when a
search actually needs it:

1. Install a WSL distro (`wsl --install -d Ubuntu`) and Docker Engine inside
   it -- not Docker Desktop, which brings its own always-on VM. `systemctl`
   needs systemd enabled in the distro, which is the default on current
   Ubuntu WSL images (`[boot] systemd=true` in `/etc/wsl.conf` if it isn't):

   ```bash
   sudo apt install docker.io
   sudo systemctl enable --now docker
   ```

2. Write `settings.yml` (with the JSON format enabled, a real secret key,
   and the rate limiter off -- it's talking to one client on localhost, not
   the public internet) and bind-mount it into the container:

   ```yaml
   # /etc/searxng/settings.yml
   search:
     formats:
       - html
       - json
   server:
     secret_key: "<a random string>"
     limiter: false
   ```

   ```bash
   docker run -d --name searxng --restart unless-stopped \
     -p 127.0.0.1:8888:8080 -v /etc/searxng:/etc/searxng searxng/searxng
   ```

3. Set `LMSA_SEARXNG_START` to a command that starts WSL and holds it open
   for a bounded time -- Docker's `--restart unless-stopped` then brings the
   container back up every time WSL starts:

   ```
   LMSA_SEARXNG_START=wsl.exe -d Ubuntu -u root --exec sleep 7200
   ```

   Quote an executable path that contains spaces (`"C:\Program Files\WSL\wsl.exe" -d Ubuntu ...`).

   The `sleep` is deliberately bounded, not `sleep infinity`: `SearXNG`
   terminates this process at interpreter exit, but a hard-killed Python (a
   crash, a task manager kill) can't run that cleanup -- the sleep is what
   guarantees WSL still idles out on its own. 7200s (2h) matches the
   briefing timeout, comfortably longer than any one run.

On a connection failure with `LMSA_SEARXNG_START` set, `SearXNG.search()`
spawns that command once per process (hidden, no console window), polls the
instance every 0.5s until it answers or `LMSA_SEARXNG_START_TIMEOUT` (default
60s) elapses, then retries the search once. Measured cost: about 9s cold
start for WSL + systemd + Docker + the container to answer, and WSL idles
itself out within about 30s of the last WSL process (the `sleep`) exiting.
The dashboard's Setup card never triggers a start (a page view shouldn't
spawn anything) -- when the instance is down but `LMSA_SEARXNG_START` is
set, it reads "not running now; starts on demand (LMSA_SEARXNG_START)"
instead of "missing".

## OSINT movements

Public data, reported in aggregate only -- never a per-aircraft track list.

- `tasks/osint_air_sample.py` runs every dispatcher tick (model "none", so it
  never waits on the gate) and stores a compact sample per aircraft from
  adsb.lol's free `/v2/mil` endpoint (military-flagged aircraft currently
  tracked): hex, ICAO type code, callsign, lat/lon bucketed to 1 degree, and
  altitude. Kept 90 days.
- `tasks/osint_movements.py` runs daily and aggregates those samples into
  counts by aircraft type class (tanker, ISR, transport, bomber, fighter) and
  region -- both mapping tables live in `config/osint_air.toml` (type codes
  and lat/lon bounding boxes; edit either freely). A (class, region) pair
  whose count today is 2 sigma or more above its 30-day baseline is flagged
  as a notable pattern. Activity density is plotted with `Section.world_map`
  markers, and the model writes one paragraph of narrative over the
  aggregates only.

  Separately, about every 6 days, it checks USNI News's Fleet and Marine
  Tracker RSS for a new article and `extract()`s carrier/ARG/ship positions
  by region from it into a table; the report always shows the latest known
  positions, whether or not USNI was actually checked that run.

  `config/schedule.toml`'s cadence syntax only expresses hourly divisors of
  24, so the sampler runs `"every 1 hours"` -- the tightest it can ask for;
  the dispatcher itself still only ticks every 30 minutes regardless.

## Bookmarks

```powershell
.venv\Scripts\python -m tasks.bookmarks                       # scan, fetch, tag
.venv\Scripts\python -m tasks.bookmarks --limit 10             # cap new fetches this run
.venv\Scripts\python -m tasks.bookmarks --query "that rust talk"  # semantic search, no scan
```

`tasks.bookmarks` reads Chrome's and Edge's `Bookmarks` JSON (default Windows
profile paths, overridable in `config/bookmarks.toml`; a browser's file is
copy-read into a temp file first, since the browser may have it open, and
never opened for write) and links found in emails sent to yourself (Gmail,
read-only IMAP, incremental via the same cursor scheme `tasks.packages`
uses -- see "Memory between runs"). Every URL is deduplicated by normalized
URL across every folder/email it showed up in.

Never-checked URLs are fetched through `lmsa.web.Fetcher`, up to
`fetch_limit` (`config/bookmarks.toml`, default 25) new fetches per run --
the rest wait for the next run, disclosed in the report. A fetched page's
article text (`lmsa.web.extract_article`) is capped at 6000 characters and
tagged by the model with `{summary, tags[], type}`, then embedded (title +
summary + tags) into the `bookmarks` vectors namespace (`lmsa.vectors`, the
same machinery `tasks.doc_search` uses over `docs`) for `--query` semantic
search.

A fetch failure is classified, not lumped into one bucket:

- **dead**: a definitive failure -- HTTP 404/410, or a DNS lookup that failed
  outright.
- **couldn't check**: a robots.txt denial, a timeout, a 5xx, or another
  transient failure -- reported separately, never counted as dead.
- **skipped**: never attempted as a network fetch at all -- a `file:`/
  `javascript:` scheme, an intranet/private host the Fetcher's SSRF guard
  refuses, or an unparseable URL.

Read-only over every source: nothing here writes a bookmark file or a Gmail
message.

## Briefings: make your own report

A briefing is a topic report built from feeds, web searches, watched pages and
API adapters, clustered into stories. Adding one is writing a file -- no code:

1. Copy `config/briefings/example.toml` to `config/briefings/<name>.toml`
   (lowercase letters, digits, `-`, `_`). Every key is documented inline.
2. Set `title`, your `interests` (plain-language descriptions -- stories are
   ranked by embedding similarity to them, and the model says why each one
   matters to you), optional `keywords` and `regions` (ISO alpha-2 codes).
3. List sources: `[[sources.feeds]]` (RSS/Atom), `[[sources.search]]`
   (SearXNG queries, see "Web search"), `[[sources.pages]]` (watched pages)
   and `[[sources.api]] adapter = "..."` (adapters registered in code). Each
   entry may carry `name`, `group`, `limit` and your own `lean` /
   `reliability` tags.
4. Try it: `python -m tasks.briefing --name <name>`; `--list` shows every
   briefing, whether it is enabled and when it last ran.
5. Schedule it: turn it on in `config/local/briefings.toml` (gitignored --
   created the first time you toggle one), e.g. `[briefings]` `<name> =
   true`. The single `[tasks.briefing]` entry in `config/schedule.toml` runs
   every enabled briefing whose last successful run is at least
   `every_hours` (default 20) old -- so each briefing keeps its own cadence
   without its own schedule entry. A briefing with no local override and no
   `enabled` in its own file shows as `new` in `--list`, and on the
   dashboard's `briefing` tile with **enable** / **keep disabled** buttons
   (confirm, then `config/local/briefings.toml` -- never the shipped file --
   is validated, backed up and rewritten). The local override always wins
   over the shipped file's own `enabled`. The tile's `--name` option is a
   dropdown of the briefing files.

   Migrating from an older checkout: if a briefing file already has a
   dashboard-written `enabled = ...  # set from the dashboard` line from
   before #95, it still works (the shipped file's own `enabled` is still
   honoured when there's no local override) -- move it to
   `config/local/briefings.toml` and delete the line from the shipped file
   by hand when convenient; nothing auto-edits the shipped file.

What a run does: collect every source (one failing source is a note in the
report, never a failed run) → **keyword pre-filter** (an article that
mentions none of the keywords -- or, without keywords, none of the longer
words in your interests -- is dropped before any model call) → optionally
fetch the full article for items with only a snippet (`follow_links`,
capped by `max_fetch`) → embed → cluster into stories (`cluster_threshold`)
→ rank by interest similarity × recency (`half_life_hours`) × cluster size
and source diversity → one `extract()` per story (headline, summary,
highlights, key quotes, timeline, why it matters, countries, actors,
severity 0-3). Story answers are remembered by a fingerprint of their
sources, so an unchanged story costs no model call next run.

**Quotes are checked**: each key quote is matched (normalized quotes and
whitespace, case, articles) against the text of the source it cites and
must appear there word for word, in order -- a one-word difference ("will"
vs "will not", "15" vs "150") fails; one that isn't there is shown flagged
**unverified**, never as a real quote.

**What changed**: stories are kept in SQLite (`briefing_stories`,
`briefing_articles`, `briefing_runs`) and matched to earlier runs by shared
URLs (then by similarity), so each card is **NEW**, **UPDATED** (new sources
or a changed summary) or ongoing, and a "Dropped since last run" list shows
what fell out. Story and article embeddings are written to the `briefings`
vectors namespace. The report's header shows stories, sources, new since
last run and an estimated read time; every cap that bites (items, stories,
fetches, text sent to the model) is listed in its Notes section.

Adding a source type or a per-story extension is code, but no engine edit:
drop a module into `lmsa/briefing/adapters/` that calls
`lmsa.briefing.sources.register(...)` or `register_extension(...)` (see the
docstring in `lmsa/briefing/sources.py`).

### DoD contract-award ledger (`config/briefings/defense.toml`)

`lmsa/briefing/adapters/dod_contracts.py` is a `[[sources.api]]` adapter
(`adapter = "dod_contracts"`) that fetches defense.gov's daily contracts
page (discovering the day's page from a listing/RSS link when the
configured URL is the listing itself), splits it into per-award
paragraphs, and parses each into `{contractor, location, amount_usd,
service, program_or_item, contract_type, completion_date,
contract_number}`. Amounts are parsed with regexes only (plain
`$1,234,567`, `$1.2 billion`, and a modification's cumulative/ceiling
total) -- the model is only asked for `contractor`/`location`/
`program_or_item`, memoized per paragraph hash, and never invents a dollar
figure regex can't find (stored as `NULL` with a note instead). Rows go in
the `dod_contracts` table, keyed by contract number + day (the paragraph
hash only when no contract number was found), so a rerun of the same day's
page -- or a corrected re-publication of the same award -- upserts instead
of duplicating. A
non-empty page that yields zero recognized award paragraphs is treated as
an unrecognized response, not a zero-award day, and is disclosed as a
note. Each new-or-changed row is also embedded into the `contracts`
vectors namespace (item id `<day>:<identity_key>`) so `tasks.ask --scope
contracts` can find it; a failed embedding call is noted and never blocks
the ledger write, and an unchanged rerun makes no embedding call at all.

The `dod_contracts` report extension (opt-in -- add it to a briefing's
`extensions`) reads that table directly and adds "today's awards", "notable
awards" (above `NOTABLE_THRESHOLD_USD` or matching a watchlist keyword,
both constants in the adapter module) and "totals by service" / "top
contractors" for the last 7/30/90 days as report sections. `defense.toml`
combines it with Defense News and Breaking Defense RSS feeds (their URLs,
and the adapter's default listing URL, are unverified against the live
site -- check them on the first real run) and ships without `enabled`, so
it shows as **new** until a human reviews and enables it.

### Source lean, reliability and blindspots (#69)

Ground News's pattern: a `lean` (`left` / `center-left` / `center` /
`center-right` / `right` / `state-media` / `intl` / `unknown`) and a
`reliability` (`high` / `mixed` / `low` / `unknown`) on each source, set by
hand -- nothing here scrapes a rating site. Tag a source two ways, either
works together:

- Directly on a `[[sources.*]]` entry in a briefing file (`lean = "..."`,
  `reliability = "..."`).
- Once, in the shared `config/sources.toml`, matched by the entry's `name`
  (its explicit `name`, else its url/query/adapter) -- so the same outlet
  doesn't need tagging in every briefing that uses it.

Precedence: an entry's own `lean`/`reliability` always wins; `config/sources.toml`
fills in whatever the entry left unset; anything neither names is
`"unknown"`. Add your own entries to `config/sources.toml` as you tag more
sources. A malformed `config/sources.toml` only drops the shared tags (with
a warning on stderr); it never stops a briefing from running.

Every story card gets a **coverage by lean** bar (skipped when every source
is `unknown` -- nothing to show) and a reliability badge on each source
link; a card with no `high`/`mixed`-reliability source is flagged. A
**Blindspot** report section lists stories where at least 80% of
lean-rated coverage (state-media, intl and center sources count toward the
total but not toward either side) comes from one of left or right, with at
least 3 distinct rated sources so one outlet's framing alone never counts
(`lmsa/briefing/adapters/lean.py`).

### Conflict tracker (`config/briefings/conflicts.toml`)

A global conflict briefing built on the engine: a think tank
(Brookings; ISW, CSIS, RAND and Crisis Group have no reachable feed, #117), world desks (BBC, Al Jazeera, Guardian, NPR,
DW, France 24, UN News, TASS -- each tagged with a `lean` and `reliability`;
state outlets as `state-media`), ReliefWeb situation reports
(`adapter = "reliefweb"`, `api.reliefweb.int/v1/reports` with
`appname=lmstudio-automation`, free) and GDELT DOC 2.0 article lists
(`adapter = "gdelt_doc"`, free). Reuters/AP have no public RSS; they arrive
via GDELT and two optional SearXNG queries. It ships **without** `enabled`,
so it shows as NEW until you enable it; then the daily `[tasks.briefing]`
run (04:30) includes it.

- **Unverified feeds**: the feed URLs could not be checked when the file was
  written; the ones marked `unverified: check on first run` may have moved.
  The report's **Sources this run** table lists every source with what it
  contributed and why not, if it didn't.
- **GDELT is throttled** (HTTP 429 at more than 1 request / 5 s): the adapter
  spaces its requests at least 5 s apart, runs at most 4 GDELT queries per
  run, skips the rest after a 429, and tells "no articles" apart from an
  unrecognized response (both are noted).
- **Theaters** (`extensions = ["theaters"]`): stories are grouped by the
  countries the model tagged (or the ReliefWeb report's countries) into
  multi-country theaters (Russia–Ukraine, Israel/Gaza/Lebanon, Yemen & Red
  Sea, Sahel, Great Lakes, ...) or one-country theaters -- a story that
  names only a member of a theater already active this run (Russia alone,
  Syria alone) joins it, and a two-country pair joins an active one-country
  theater of either member, so one conflict stays in one theater -- ranked
  by activity =
  stories × distinct sources × (1 + max severity), and shown on one world
  map shaded by activity (click a country to jump to its theater).
- **Per theater**: an escalating / de-escalating / steady / unclear flag
  with a one-line reason (one memo'd model call per theater, top 12), what
  changed in the last 24 h, and casualty or displacement figures **only**
  when the model quotes them and the quote is found word for word in a
  source (with its link), has no ellipsis, and puts the number next to a
  casualty/displacement word in one sentence. The report shows the number
  as the quote writes it plus the quote -- never the model's wording, and
  no figure is ever estimated.
- **Sources this run** says what happened to each source: ok (collected /
  kept), fetch failed, partial (collected some, then failed), returned 0
  items, all items filtered by keywords, or
  not collected because `max_items` was reached (keep `max_items` above the
  sum of the sources' `limit`s -- API sources are collected last).

## Go deep: research a story further

Every briefing story card carries a **go deep** link ("open in dashboard"):
`http://127.0.0.1:8765/?task=go_deep&story=<briefing>:<story_id>` (port from
`lmsa.config.DASHBOARD_PORT`, the dashboard's default), so it works from a
report opened from disk too while the dashboard is running. It opens the
launcher with the `go_deep` tile's `--story` pre-filled; nothing runs until
you press **run**. Only `go_deep`'s `--story` / `--topic` can be pre-filled
by a link (`PREFILL_FIELDS` in `tasks/dashboard.py`); every other task
ignores the query string. From a shell:
`python -m tasks.go_deep --story <briefing>:<story_id>`, or any topic with
`python -m tasks.go_deep --topic "..."`.

The model plans 3-6 sub-questions. For each: one SearXNG search, up to 3
new pages fetched politely (`lmsa.web`, the story's own articles first),
pages chunked and embedded, and only the top passages (about 6,000
characters) sent to the model to draft a section with `[n]` citations. A
last pass writes an executive summary and open questions. Every `[n]` must
name a page the model was shown for that section -- anything else is
replaced with `[?]` and counted; quotes are checked word for word.

Hard caps, each a flag: `--max-searches` (6), `--max-pages` (12),
`--max-chars` sent to the model (60,000) and `--max-minutes` (20). A cap
stops the run cleanly: the report is marked partial and names the cap.
Reports are `output/go-deep-<topic>-YYYYMMDD-HHMM.html`; they link back to
the story's card, and the story's card links to its newest deep report
(`go_deep_reports` table). Manual only (`config/dashboard.toml`).

## Site watch: page diffs judged against a plain-English rule

changedetection.io's pattern, but an LLM judges the diff against a rule and
describes the change instead of dumping the raw diff. Same core
(`lmsa/briefing/sitewatch.py`) two ways:

- **`[[sources.pages]]` in any briefing** — a watched page becomes a story
  candidate only when it actually changes *and* the model finds the change
  relevant to the entry's `rule`; see `config/briefings/example.toml`.
- **`config/site_watch.toml` + `python -m tasks.site_watch`** — a
  standalone report and alerts, for pages that don't belong in a topic
  briefing. Copy the commented block in the file and edit it.

Either way, each page carries: `url` (required), `rule` (plain English --
empty means "any real content change"), `selector` (optional; a bare tag,
`#id` or `.class` -- kept deliberately simple; default is the same
full-page article-text extractor `lmsa.web.extract_article` uses),
`notify` (opt-in `lmsa.notify` alert when a change is judged relevant), and
`fields` (optional typed values extracted every run and charted as a
sparkline, e.g. `fields = { price = "number" }`; each value is `"number"`,
`"text"` or `"bool"`).

What a check does: fetch the page once → extract its text (selector or
`extract_article`) → diff against the *last* snapshot for that
**watcher's** own lineage on that url (the standalone task and each
briefing entry watching the same url keep separate snapshots, so one
watcher seeing a change never leaves another with nothing to diff against).
No previous snapshot → this is the page's baseline: the snapshot is saved
but nothing is judged or alerted (there's nothing yet to compare against).
No diff → unchanged, no model call. A real diff → one `extract()` call
decides `relevant` and a one-sentence `summary`, plus any requested
`fields`; judgments (not snapshots) are memoized per url, diff hash, rule
and prompt, so the same real-world change under the same rule is judged
once however many watchers see it. The snapshot only advances once judging
succeeds (or the diff turns out empty at the line level) -- a failed model
call leaves the previous snapshot in place so the next run retries the same
change instead of losing it. Snapshots are kept bounded (the newest 20 per
watcher/url); typed field history is kept bounded too (the newest 365 per
watcher/url/field) for its sparkline. A `fields` value is parsed
leniently (currency symbols, thousands separators either style, a decimal
comma when unambiguous); one that still can't be parsed is `null` with a
disclosed note, never silently dropped. Whenever the page text or the diff
had to be cut to its length limit, that's disclosed to the model in the
prompt and shown (`Truncated`) in the report / the briefing story.

## Scheduling: the dispatcher

Task Scheduler runs one thing, `python -m tasks.dispatch`, every 30 minutes.
The dispatcher decides what is due from `config/schedule.toml`, runs it only
when the PC is quiet, and loads models on demand.

```powershell
.\scheduler\register-task.ps1 -WhatIf              # preview
.\scheduler\register-task.ps1                      # register \lmsa\dispatch
.venv\Scripts\python -m tasks.dispatch --dry-run   # what would run now, and why not
```

The entry runs every 30 min at BelowNormal priority, on battery too, never
wakes the PC, and never starts a second copy while one is running. It runs as
you, only while you're logged on, through `pythonw.exe` (no console window).

**Busy** means any of:

- a fullscreen game or app, or presentation mode (`SHQueryUserNotificationState`);
- the GPU is in use (`nvidia-smi` utilization over `gpu_util_max`), or free VRAM
  is below the model about to load plus `vram_headroom_mb`;
- keyboard/mouse input in the last `idle_minutes` (daytime). In the `overnight`
  window (01:00-07:00 by default) `overnight_idle_minutes` applies instead, 0 by
  default, so overnight only the other checks count;
- a process in `blocked_processes` is running (image name, case-insensitive,
  `.exe` optional). The list ships empty, so nothing blocks until you add e.g.
  `"steam.exe"`.

A signal that can't be read counts as busy. When busy, due tasks are
**deferred** and retried at the next check. A task still blocked at its
`deadline_hours` is **skipped**; the deadline clock starts when the dispatcher
first sees the run due, so time the PC spent off or asleep doesn't count.

When quiet, due tasks run as subprocesses with a timeout (the task's whole
process tree is killed if it overruns), grouped by model: no-model tasks, then
the whisper class (speech-to-text the task runs itself; see *Voice memos*), then
the chat model (one load), unload, then the vision model. Vision tasks get
`LMSA_CHAT_MODEL` pointed at the vision model, so their text calls don't pull
the chat model in beside it. A watchdog re-checks the gate *between* tasks: a
task that is already running finishes, and if the PC has turned busy by then
the dispatcher stops, unloads its model and defers the rest. It also stops
starting tasks after `max_run_minutes` (240), well inside Task Scheduler's 6 h
limit. Only models the dispatcher loaded, or that appeared while one of its
tasks ran (LM Studio JIT loads, e.g. the embedding model), are unloaded; each
load carries a TTL (`model_ttl_seconds`) so a crash leaves nothing pinned.

Every run, deferral and skip goes into the `dispatch_runs` table. A skip is
reported to the evening recap straight away (a `dispatch` run listing the
skipped tasks), before anything runs; a skip decided after that day's recap is
carried into the next day's recap. When something ran or failed, the
dispatcher records another run with the counts and rewrites
`output/dispatch-YYYYMMDD.html`.

Each task's combined stdout and stderr is kept in
`data/dispatch/logs/<task>-<timestamp>.log`, and the row's `log_path` points at
it, so the dashboard's Errors view can show the full traceback of a scheduled
failure. The reason recorded is still the last line of stderr.

**Set by hand in LM Studio:**

- lower the JIT model TTL from 1 h to about 5 min;
- don't pin models (no "keep loaded"). A model you load by hand is never
  unloaded by the dispatcher, and while it sits in VRAM the other model can't
  load, so its tasks wait.

**Old per-task entries** from the previous `register-task.ps1` aren't removed
for you:

```powershell
Get-ScheduledTask -TaskPath "\lmsa\" | Where-Object TaskName -ne "dispatch" |
    Unregister-ScheduledTask -Confirm:$false
```

## Master Report: one email a day

`python -m tasks.master_report` (scheduled `daily 06:45`) sends one email that
bundles several reports, configured in `config/master_report.toml` (every key
documented inline). Per-machine settings -- the recipient, `enabled`, and
`send = true` -- go in the untracked `config/local/master_report.toml`
(`[master.<name>]`). It **re-runs nothing**: each section is built from the
task's latest successful run in the `runs` table (a briefing section from
`briefing_runs`), with a status chip -- `OK`, `STALE (last good run Nh ago)`
(older than `max_age_hours`, which is also how a run that hasn't finished yet
today shows), `FAILED` (latest run failed), or `NOT RUN`.

A task shows up through an optional `digest(run, max_items) -> Digest` hook in
its module (see `lmsa/digest.py` for the contract: stored data only, never the
model); without one it gets its `summary_json` numbers as tiles. A section's
`mode` is `full` by default (#119): the email carries the task's *whole* latest
report (`lmsa/email_report.py` converts the saved page -- every card, story and
table row; maps/heatmaps/sparklines are left out with a note), falling back to
its digest when the file is gone; `digest`, `headline` and `link` show less.
The body is email-safe HTML laid out for iOS Gmail / iOS Mail, plus a
plain-text part with the same content, and is **never trimmed** -- past ~102 KB
Gmail shows "View entire message". An Afterhours HTML copy is attached for
desktop. The
optional `llm_intro` is the only model call and is best effort: the task is
scheduled with `model = "none"` and `idle_minutes = 0`, so neither LM Studio's
state nor input activity holds the email back, and the intro uses the chat
model only if the GPU and VRAM allow it right then. A `send_at` later than the
scheduled run is a config error, since it could never go out.

Dry-run by default: writes `output/master-report-<name>-<ts>.html` and a `.eml`
preview. A real send (`send = true` or `--send`) goes out at most once per
master per day; a same-day rerun is a no-op, and a send the mailer skips
(already claimed, e.g. stuck mid-send after a crash) prints a warning.

## Voice memos

`tasks/voice_memos.py` turns recordings into notes: audio files (m4a, mp3,
wav, ogg, webm) under `LMSA_RECORDINGS_DIR` (default
`~/Documents/Recordings`, subfolders included) from the last `--days` (7) are
transcribed locally with **faster-whisper**, then the chat model writes a
title, summary, decisions, action items (owner and due date only when
spoken) and open follow-up questions per recording.

```powershell
.venv\Scripts\python -m pip install -e ".[voice]"     # faster-whisper: an optional extra
.venv\Scripts\python -m tasks.voice_memos --phase all  # by hand: transcribe, then summarize
.venv\Scripts\python -m tasks.voice_memos --actions    # also append action items to data/actions.jsonl
```

- **Two phases, never sharing the GPU** (10 GB can't hold whisper and the
  chat model together). `voice_transcribe` (= `--phase transcribe`) is
  scheduled in the dispatcher's `whisper` model class: it runs after the
  no-model batch and before the chat batch, only once no LM Studio model is
  loaded (the dispatcher unloads its own; one you loaded by hand makes it
  wait) and free VRAM covers `[models].whisper_size_mb`. `voice_memos`
  (summarize, the default phase) runs in the chat batch. `--phase all` frees
  whisper before its first LM Studio call.
- `LMSA_WHISPER_MODEL` (default `small`), `LMSA_WHISPER_DEVICE`
  (`auto` = CUDA int8 when CTranslate2 sees a GPU, else CPU int8). A CUDA
  failure at load or at the first transcribe (e.g. missing cuBLAS/cuDNN DLLs)
  rebuilds the model on the CPU, retries that file, stays on the CPU for the
  rest of the run, and the report says so.
- **Memoised by file content** (SHA-256): each recording is transcribed once
  (a renamed file isn't redone) and summarized once per prompt/model version;
  `--fresh` re-summarizes but never re-transcribes. A file modified in the
  last 2 minutes is left for the next run. Files over `--max-mb` (500) are
  not transcribed and are listed (one already transcribed under a higher cap
  is still summarized). At most 40 files are transcribed per run, never-tried
  files first, and no new file is started once `--budget-min` (105, under the
  schedule's 120-minute timeout) is spent -- the report says how many are
  left for the next run. An attempt is recorded *before* each file, so one
  that crashes or hangs the process goes to the back of the queue next run
  and is given up on (and reported) after 2 strikes. An unfinished attempt is
  a strike only if it started with at least half the budget left; one the
  timeout cut short goes first next run instead. `--reset-attempts` forgets
  the strikes. A transcript longer than 24,000
  characters is cut for the model and the report says so.
- Transcripts are written to `output/voice_memos/<name>-<hash>.txt` and `.md`
  (timestamped). The recordings folder is only read.
- `--actions` appends each action item to `data/actions.jsonl` as one JSON
  object per line (`id`, `source`, `recording`, `recorded`, `task`, `owner`,
  `due`, `added`). Ids are `voice:<sha16>:<hash of task, owner and due>`
  (case and spacing ignored, punctuation kept), so reruns (even `--fresh`)
  never duplicate an item and never drop a new one. If another run holds the
  lock, the report is still written and says to rerun with `--actions`. The read
  and the append happen under an exclusive lock file (`actions.jsonl.lock`),
  so concurrent runs can't duplicate either. `--actions` is a `[writes]` flag:
  the dashboard asks for a confirm.
- The dashboard's Setup page has a **Voice memos** card: faster-whisper
  installed? CUDA available? recordings folder present?

