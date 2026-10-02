"""Document inbox auto-filing (paperless-style): title, tags, sender, then file.

Scans `LMSA_DOC_INBOX_DIR` (top level only, not recursive) for PDFs, images
and .docx files, reads each one, and asks the model for `{title, doc_date,
correspondent, doc_type, tags, amount, due_date}`:

    - a PDF's text layer via `pypdf`; a scanned/image-only PDF has its first
      pages rendered and sent to the vision model (`LMStudio.see()`)
    - an image (a phone photo, a scan) goes to the vision model directly
    - a .docx via python-docx

The PDF/image readers are `tasks.attachments`' own (same size and render
guards). Text sent to the model is capped at `TEXT_LIMIT` characters; the
report marks every document that was cut.

Each document gets a proposed destination from the template in
`config/doc_inbox.toml` (default
`<archive>/<doc_type>/<YYYY>/<YYYY-MM-DD> <correspondent> - <title>.<ext>`).
Every path component is sanitised (no separators, no `..`, no device names)
and the result must stay inside the archive root.

Dry-run by default: writes an Afterhours report of current name -> proposed
path; nothing on disk changes. `--apply` moves the files with the same safety
as `tasks.file_sorter`: rename only (never copy, never delete, never
overwrite), collisions get a numeric suffix re-checked at move time, and every
move is logged to `data/doc_inbox/<RUN_ID>.jsonl` *before* it happens.
`--undo RUN_ID` moves that run's files back, refusing to overwrite anything
and listing whatever it couldn't restore. A file the model or the filesystem
chokes on is skipped and listed; the rest still run.

Due dates that are still ahead go into an .ics file next to the report
(`output/doc_inbox-due-*.ics`), the same format `tasks.deadlines` writes.

    python -m tasks.doc_inbox                  # dry run
    python -m tasks.doc_inbox --apply          # file them; prints the RUN_ID
    python -m tasks.doc_inbox --undo 20260925-031000-ab12cd34
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from lmsa import Card, LMStudio, Report, Store, Tile
from lmsa import layout
from lmsa.config import ROOT
from lmsa.ics import Event, make_uid, write_ics
from lmsa.memo import Memo, add_footer, cached, memo_version, model_id
from lmsa.prompts import load, prompt_hash
from tasks.attachments import (
    PDF_RENDER_PAGES,
    PDF_RENDER_SCALE,
    PDF_TEXT_MIN_CHARS,
    VISION_JPEG_QUALITY,
    VISION_MAX_DIM,
    _downscale_jpeg,
    _pdf_text,
    _render_pdf_pages,
    _sniff_kind,
)
from tasks.file_sorter import (
    MAX_SUFFIX,
    _extension,
    _key,
    candidate_files,
    memo_key,
    move_no_clobber,
    sanitize,
    unique_dest,
)

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"env": ["LMSA_DOC_INBOX_DIR"], "config": ["config/doc_inbox.toml"], "model": "vision"}

DEFAULT_CONFIG_PATH = ROOT / "config" / "doc_inbox.toml"
DEFAULT_TEMPLATE = "{doc_type}/{year}/{date} {correspondent} - {title}{ext}"
DEFAULT_MIN_AGE_MINUTES = 10  # leave a file a scanner/download may still be writing
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
DOC_EXTS = {".pdf", ".docx"} | IMAGE_EXTS
MAX_BYTES = 25 * 1024 * 1024  # a larger file is skipped, never read into memory
TEXT_LIMIT = 6000  # characters of document text sent to the model
MAX_TAGS = 8
FIELD_MAX = 60  # characters kept of title/correspondent/doc_type in a path component
MEMO_VERSION = 1  # bump when read_document()/prompt building or result handling changes
TEMPLATE_FIELDS = {"doc_type", "year", "month", "date", "correspondent", "title"}
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "doc_date": {"type": ["string", "null"]},
        "correspondent": {"type": "string"},
        "doc_type": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "amount": {"type": ["number", "null"]},
        "due_date": {"type": ["string", "null"]},
    },
    "required": ["title", "doc_date", "correspondent", "doc_type", "tags", "amount", "due_date"],
    "additionalProperties": False,
}

SYSTEM = load("doc_inbox/system")


def _now() -> datetime:
    """Current UTC time. Tests monkeypatch this."""
    return datetime.now(timezone.utc)


# --- config ------------------------------------------------------------------


@dataclass
class Config:
    inbox: Path
    archive: Path
    template: list[str]  # path components, without the trailing {ext}
    min_age: timedelta


def parse_template(template: str) -> list[str]:
    """Split a destination template into components, validating it.

    Only the fields in `TEMPLATE_FIELDS` may appear, with no format spec or
    conversion; `{ext}` may appear only at the very end (it's always added).
    No absolute paths, no empty/`.`/`..` components.
    """
    template = template.strip()
    if template.endswith("{ext}"):
        template = template[: -len("{ext}")]
    if "{ext}" in template:
        raise ValueError("template: {ext} may only appear at the very end")
    if not template or template.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", template):
        raise ValueError(f"template must be a relative path: {template!r}")
    parts = re.split(r"[\\/]", template)
    for part in parts:
        if part.strip() in ("", ".", ".."):
            raise ValueError(f"template has an empty or '.'/'..' component: {template!r}")
        for _, name, spec, conv in string.Formatter().parse(part):
            if name is None:
                continue
            if name not in TEMPLATE_FIELDS or spec or conv:
                raise ValueError(
                    f"template field {{{name}}} not allowed (use {', '.join(sorted(TEMPLATE_FIELDS))}, ext)"
                )
    return parts


def load_config(path: Path = DEFAULT_CONFIG_PATH, *, inbox: str | None = None) -> Config:
    """Read `config/doc_inbox.toml` (all keys optional) plus `LMSA_DOC_INBOX_DIR`."""
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    inbox_raw = inbox or os.getenv("LMSA_DOC_INBOX_DIR", "")
    if not inbox_raw:
        raise ValueError("LMSA_DOC_INBOX_DIR is not set (or pass --inbox DIR)")
    # Absolute from here on: the run log must name the same files whatever cwd --undo runs from.
    inbox_dir = Path(os.path.abspath(Path(inbox_raw).expanduser()))
    archive_raw = str(data.get("archive_dir") or "")
    # A relative archive_dir is relative to the inbox (never to the cwd).
    archive = Path(os.path.abspath(inbox_dir / Path(archive_raw or "Filed").expanduser()))
    if os.path.normcase(os.path.abspath(archive)) == os.path.normcase(os.path.abspath(inbox_dir)):
        raise ValueError("archive_dir must differ from the inbox (filed documents would be re-filed)")
    minutes = data.get("min_age_minutes", DEFAULT_MIN_AGE_MINUTES)
    if not isinstance(minutes, (int, float)) or minutes < 0:
        raise ValueError(f"min_age_minutes must be a number >= 0, got {minutes!r}")
    return Config(
        inbox=inbox_dir,
        archive=archive,
        template=parse_template(str(data.get("template", DEFAULT_TEMPLATE))),
        min_age=timedelta(minutes=minutes),
    )


def inbox_files(cfg: Config) -> list[Path]:
    """Documents directly in the inbox (file_sorter's filters: no symlinks,
    hidden files or partial downloads), old enough not to be mid-write."""
    return [p for p in candidate_files([cfg.inbox], cfg.min_age) if p.suffix.lower() in DOC_EXTS]


# --- reading + extraction ----------------------------------------------------


def _truncate(text: str, limit: int = TEXT_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated, {omitted} more characters omitted ...]", True


def _docx_text(path: Path) -> str:
    import docx

    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def read_document(lm: LMStudio, path: Path) -> dict:
    """Ask the model about one document. Returns `{"fields", "source", "truncated"}`.

    Raises on anything unreadable (too large, corrupt, encrypted, wrong
    content for its extension); the caller records it and moves on.
    """
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{size} bytes, over the {MAX_BYTES} byte cap")
    header = f"Filename: {path.name}\n"
    if path.suffix.lower() == ".docx":
        text, cut = _truncate(_docx_text(path))
        fields = lm.extract(f"{header}\n{text}", SCHEMA, system=SYSTEM, name="doc_fields")
        return {"fields": fields, "source": "docx-text", "truncated": cut}

    data = path.read_bytes()
    kind = _sniff_kind(data)  # content, not the extension, decides
    if kind == "unknown":
        raise ValueError("not a PDF or a readable image")
    if kind == "pdf":
        text = _pdf_text(data)
        if len(text.strip()) >= PDF_TEXT_MIN_CHARS:
            text, cut = _truncate(text)
            fields = lm.extract(f"{header}\n{text}", SCHEMA, system=SYSTEM, name="doc_fields")
            return {"fields": fields, "source": "pdf-text", "truncated": cut}
        pages = _render_pdf_pages(data, pages=PDF_RENDER_PAGES, scale=PDF_RENDER_SCALE)
        if not pages:
            raise ValueError("PDF has no pages to render")
        source = "pdf-vision"
    else:
        pages = [data]
        source = "image-vision"
    images = [_downscale_jpeg(p, max_dim=VISION_MAX_DIM, quality=VISION_JPEG_QUALITY) for p in pages]
    prompt = f"{header}The attached image(s) are pages of the document. Extract its fields."
    fields = lm.see(images, prompt, SCHEMA, system=SYSTEM, name="doc_fields")
    return {"fields": fields, "source": source, "truncated": False}


def _clean(value: object, fallback: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or fallback


def _strict_date(value: object) -> date | None:
    """A YYYY-MM-DD date, or None for anything else (model output is untrusted)."""
    if not isinstance(value, str) or not _ISO_DATE.match(value.strip()):
        return None
    try:
        d = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return d if 1900 <= d.year <= 2200 else None


def _amount(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) < 1e12 else None  # drops NaN/inf


@dataclass
class DocInfo:
    title: str
    correspondent: str
    doc_type: str
    doc_date: date
    date_from_file: bool  # True when the model gave no usable date (file mtime used)
    tags: list[str] = field(default_factory=list)
    amount: float | None = None
    due_date: date | None = None


def normalize(fields: dict, src: Path) -> DocInfo:
    """Model output -> DocInfo, with fallbacks for anything missing or malformed."""
    if not isinstance(fields, dict):
        raise ValueError(f"model returned {type(fields).__name__}, expected an object")
    doc_date = _strict_date(fields.get("doc_date"))
    from_file = doc_date is None
    if doc_date is None:
        doc_date = datetime.fromtimestamp(src.stat().st_mtime).date()
    raw_tags = fields.get("tags") if isinstance(fields.get("tags"), list) else []
    tags: list[str] = []
    for t in raw_tags:
        t = _clean(t, "")[:40]
        if t and t.lower() not in {x.lower() for x in tags}:
            tags.append(t)
    return DocInfo(
        title=_clean(fields.get("title"), src.stem),
        correspondent=_clean(fields.get("correspondent"), "Unknown"),
        doc_type=_clean(fields.get("doc_type"), "Other"),
        doc_date=doc_date,
        date_from_file=from_file,
        tags=tags[:MAX_TAGS],
        amount=_amount(fields.get("amount")),
        due_date=_strict_date(fields.get("due_date")),
    )


# --- destination -------------------------------------------------------------


def _within(path: Path, root: Path) -> bool:
    a, b = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([a, b]) == b and a != b
    except ValueError:  # different drives
        return False


def plan_dest(cfg: Config, src: Path, info: DocInfo) -> Path:
    """The templated destination, every component sanitised, inside the archive."""
    values = {
        "doc_type": sanitize(info.doc_type, fallback="Other", max_len=FIELD_MAX),
        "correspondent": sanitize(info.correspondent, fallback="Unknown", max_len=FIELD_MAX),
        "title": sanitize(info.title, fallback=src.stem, max_len=FIELD_MAX),
        "year": f"{info.doc_date:%Y}",
        "month": f"{info.doc_date:%m}",
        "date": info.doc_date.isoformat(),
    }
    parts = [sanitize(p.format_map(values), fallback="_", max_len=150) for p in cfg.template]
    dest = cfg.archive.joinpath(*parts[:-1], parts[-1] + _extension(src))
    if not _within(dest, cfg.archive):
        raise ValueError(f"destination escapes the archive: {dest}")
    return dest


# --- planning ----------------------------------------------------------------


@dataclass
class Plan:
    src: Path
    dest: Path
    info: DocInfo
    source: str  # "pdf-text" | "pdf-vision" | "image-vision" | "docx-text"
    truncated: bool = False


@dataclass
class Skip:
    src: Path
    reason: str


def open_memo(conn, lm: LMStudio, *, fresh: bool = False) -> Memo:
    version = memo_version(
        MEMO_VERSION, prompt_hash("doc_inbox/system"), SCHEMA, TEXT_LIMIT, PDF_TEXT_MIN_CHARS, PDF_RENDER_PAGES,
        model_id(lm, "vision_model"), model_id(lm, "chat_model"),
    )
    return Memo(conn, "doc_inbox", version, fresh=fresh)


def _valid(result: object) -> bool:
    return (
        isinstance(result, dict)
        and isinstance(result.get("fields"), dict)
        and all(k in result["fields"] for k in SCHEMA["required"])
    )


def build_plans(
    lm: LMStudio, cfg: Config, files: list[Path], memo: Memo | None = None
) -> tuple[list[Plan], list[Skip]]:
    """Plan each file independently: one bad document never stops the rest."""
    plans: list[Plan] = []
    skips: list[Skip] = []
    taken: set[str] = set()
    for src in files:
        try:
            key = memo_key(src)
            result = cached(memo if key else None, key or "", lambda: read_document(lm, src), valid=_valid)
            if not _valid(result):
                raise ValueError("model answer is missing required fields")
            info = normalize(result["fields"], src)
            dest = unique_dest(plan_dest(cfg, src, info), taken)
        except Exception as e:  # noqa: BLE001 - isolate per file, report it
            skips.append(Skip(src, f"{type(e).__name__}: {e}"))
            continue
        plans.append(Plan(src, dest, info, str(result.get("source", "")), bool(result.get("truncated"))))
    return plans, skips


# --- apply / undo ------------------------------------------------------------


def _runs_dir() -> Path:
    return ROOT / "data" / "doc_inbox"


def new_run_id(now: datetime | None = None) -> str:
    return f"{(now or _now()):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


def run_log_path(run_id: str) -> Path:
    """The log for `run_id`. The id is validated, so it can't name a path elsewhere."""
    if not RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"not a doc_inbox run id: {run_id!r} (expected YYYYMMDD-HHMMSS-xxxxxxxx)")
    return _runs_dir() / f"{run_id}.jsonl"


def _real_within(path: Path, root: Path) -> bool:
    """`path` is inside `root` once symlinks/junctions in the existing part of
    either are resolved (`realpath` is non-strict: a missing tail is kept)."""
    return _within(Path(os.path.realpath(path)), Path(os.path.realpath(root)))


def _log_line(log, record: dict) -> None:
    log.write(json.dumps(record) + "\n")
    log.flush()
    os.fsync(log.fileno())


def apply_plans(cfg: Config, plans: list[Plan], run_id: str) -> tuple[Path | None, list[Plan], list[Skip]]:
    """Move each planned file, logging every move to the run log *before* it happens.

    Mirrors `tasks.file_sorter.apply_plans`: collisions are re-checked at move
    time, a newly taken name gets a fresh suffix, rename only. On top of that:

    - containment is checked on the *resolved* path before any folder is
      created (and again after), so a junction/symlink planted in the archive
      can neither receive the file nor get folders created outside it;
    - every intent line (`{"seq", "src", "dest", "size"}`) is followed by a
      fsynced outcome line, `{"done": seq}` or `{"failed": seq}`, and undo
      acts only on `done` entries;
    - if nothing moved, the log is removed and None is returned.

    Returns the run log (None if nothing moved), the moves done, and skips.
    """
    done: list[Plan] = []
    skips: list[Skip] = []
    log_path: Path | None = None
    log = None
    seq = 0
    unconfirmed = False  # a logged intent whose outcome is unknown: keep the log
    try:
        for p in plans:
            try:
                if not p.src.is_file() or p.src.is_symlink():
                    raise FileNotFoundError(f"source no longer a regular file: {p.src}")
                if not _within(p.dest, cfg.archive):
                    raise ValueError(f"destination escapes the archive: {p.dest}")
                if not _real_within(p.dest, cfg.archive):  # before mkdir creates anything
                    raise ValueError(f"destination folder resolves outside the archive: {p.dest.parent}")
                p.dest.parent.mkdir(parents=True, exist_ok=True)
                if not _real_within(p.dest, cfg.archive):  # re-check: the tree may have changed
                    raise ValueError(f"destination folder resolves outside the archive: {p.dest.parent}")
                size = p.src.stat().st_size
                if log is None:
                    # Before the first move: if the log can't be created, nothing has moved yet.
                    log_path = run_log_path(run_id)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = open(log_path, "x", encoding="utf-8")  # never reuse a run's log
                    _log_line(log, {
                        "run_id": run_id, "inbox": str(cfg.inbox), "archive": str(cfg.archive),
                        "created": _now().isoformat(),
                    })
                for _ in range(MAX_SUFFIX):
                    dest = unique_dest(p.dest, set())
                    seq += 1
                    # Intent first, so a crash right after the rename still leaves a
                    # trace (undo reports it as unconfirmed and never acts on it).
                    _log_line(log, {"seq": seq, "src": str(p.src), "dest": str(dest), "size": size})
                    try:
                        move_no_clobber(p.src, dest)
                    except FileExistsError:
                        _log_line(log, {"failed": seq})
                        continue  # raced: something appeared at dest; pick the next name
                    except BaseException:
                        # Only call it failed when the source provably stayed put; an
                        # interrupt right after the rename must stay visible as
                        # unconfirmed rather than be dropped from undo's report.
                        if p.src.exists():
                            _log_line(log, {"failed": seq})
                        else:
                            unconfirmed = True
                        raise
                    try:
                        _log_line(log, {"done": seq})
                    except OSError:
                        # The rename happened; losing the marker must not turn it into a
                        # "skip" or delete the log. Undo lists it as unconfirmed.
                        unconfirmed = True
                    break
                else:
                    raise FileExistsError(f"no free name near {p.dest}")
            except Exception as e:  # noqa: BLE001 - one locked/vanished file must not stop the rest
                skips.append(Skip(p.src, f"{type(e).__name__}: {e}"))
                continue
            done.append(Plan(p.src, dest, p.info, p.source, p.truncated))
    finally:
        if log is not None:
            log.close()
    if log_path is not None and not done and not unconfirmed:
        # Only failed attempts were logged: nothing to undo, so no log and no undo hint.
        log_path.unlink(missing_ok=True)
        log_path = None
    return log_path, done, skips


def read_run_log(log_path: Path) -> tuple[dict | None, list[dict], list[dict]]:
    """(header, completed moves, unconfirmed intents) of a run log.

    A move counts as completed only once its `{"done": seq}` line exists. An
    intent with no outcome line (a crash between rename and marker) comes back
    separately so undo can report it; one marked failed is dropped. A torn
    line is ignored.
    """
    header, intents, done, failed = None, {}, set(), set()
    for i, line in enumerate(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(m, dict):
            continue
        if i == 0 and "run_id" in m:
            header = m
        elif isinstance(m.get("seq"), int) and "src" in m and "dest" in m:
            intents[m["seq"]] = m
        elif isinstance(m.get("done"), int):
            done.add(m["done"])
        elif isinstance(m.get("failed"), int):
            failed.add(m["failed"])
    completed = [intents[n] for n in sorted(intents) if n in done]
    unconfirmed = [intents[n] for n in sorted(intents) if n not in done and n not in failed]
    return header, completed, unconfirmed


@dataclass
class UndoNote:
    status: str  # "restored" | "skipped" | "refused" | "error"
    path: str
    detail: str = ""


def undo(run_id: str) -> list[UndoNote]:
    """Reverse a previous `--apply`, newest move first.

    Never overwrites: an occupied original path is left alone and reported.
    An entry is only restored when the file is still at its destination with
    the size it was filed at, its destination is inside the run's archive and
    its original path directly inside the run's inbox (a hand-edited log
    can't move files anywhere else).
    """
    log_path = run_log_path(run_id)
    if not log_path.is_file():
        raise FileNotFoundError(f"no run log for {run_id}: {log_path}")
    header, completed, unconfirmed = read_run_log(log_path)
    if not isinstance(header, dict) or header.get("run_id") != run_id:
        raise ValueError(f"run log {log_path} has no valid header")
    inbox, archive = Path(header["inbox"]), Path(header["archive"])
    if not (inbox.is_absolute() and archive.is_absolute()):
        raise ValueError(f"run log {log_path} holds relative paths; refusing to guess the cwd")

    notes: list[UndoNote] = [
        UndoNote("skipped", str(m["dest"]), f"move never confirmed (interrupted?); check by hand: {m['src']}")
        for m in unconfirmed
    ]
    for m in reversed(completed):
        src, dest = Path(m["src"]), Path(m["dest"])
        try:
            if _key(src.parent) != _key(inbox) or not _within(dest, archive):
                notes.append(UndoNote("refused", str(dest), "entry points outside this run's inbox/archive"))
            elif dest.is_symlink() or not dest.is_file():
                notes.append(UndoNote("skipped", str(dest), "not at destination (never moved, or moved since)"))
            elif "size" in m and dest.stat().st_size != m["size"]:
                notes.append(UndoNote("skipped", str(dest), "changed since it was filed"))
            elif os.path.lexists(src):
                notes.append(UndoNote("skipped", str(src), "original path is occupied; not overwriting"))
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                move_no_clobber(dest, src, check_reserved=False)
                notes.append(UndoNote("restored", str(src)))
        except Exception as e:  # noqa: BLE001 - keep undoing the rest
            notes.append(UndoNote("error", str(dest), f"{type(e).__name__}: {e}"))
    return notes


# --- due dates -> .ics -------------------------------------------------------


def due_events(plans: list[Plan], today: date) -> list[Event]:
    """One all-day event per document with a due date today or later."""
    events = []
    for p in plans:
        due = p.info.due_date
        if due is None or due < today:
            continue
        events.append(Event(
            uid=make_uid("doc_inbox", p.info.correspondent, p.info.title, due.isoformat()),
            title=f"Due: {p.info.correspondent} - {p.info.title}",
            start=due,
            all_day=True,
            description=f"Document: {p.dest}",
        ))
    return events


# --- report ------------------------------------------------------------------


def _fmt_amount(a: float | None) -> str:
    return "" if a is None else f"{a:,.2f}"


def render_report(
    plans: list[Plan],
    cfg: Config,
    *,
    applied: bool,
    skips: list[Skip] = (),
    run_id: str | None = None,
    ics_path: Path | None = None,
    output_dir: Path | None = None,
) -> Report:
    verb = "Filed" if applied else "Proposed"
    r = Report(
        f"Document inbox ({'applied' if applied else 'dry run'})",
        kicker="lmsa · doc_inbox",
        subtitle=f"{verb} {len(plans)} document(s) from {cfg.inbox} into {cfg.archive}",
        sticker=f"{len(plans)} document(s)",
        sticker_tone="acc" if applied else "info",
        source="tasks.doc_inbox",
    )
    r.tiles([
        Tile("Documents", len(plans), tone="info"),
        Tile("Read by vision", sum(1 for p in plans if p.source.endswith("vision")), tone="violet"),
        Tile("With due date", sum(1 for p in plans if p.info.due_date), tone="warn"),
        Tile("Skipped", len(skips), tone="warn" if skips else "muted"),
    ])
    if applied and run_id:
        r.section("Undo", tone="muted").callout(
            f"python -m tasks.doc_inbox --undo {run_id}", label="run id", tone="info"
        )
    elif not applied:
        r.section("Dry run", tone="muted").callout(
            "Nothing was moved. Re-run with --apply to file these documents.", label="dry run", tone="info"
        )

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(cfg.archive))
        except ValueError:
            return str(p)

    r.section(f"{verb} filing", tone="acc" if applied else "warn").table(
        ["Current name", "Destination (in archive)", "Type", "From", "Date", "Tags", "Amount", "Due", "Read via"],
        [
            [
                p.src.name, rel(p.dest), p.info.doc_type, p.info.correspondent,
                p.info.doc_date.isoformat() + (" (file date)" if p.info.date_from_file else ""),
                ", ".join(p.info.tags), _fmt_amount(p.info.amount),
                p.info.due_date.isoformat() if p.info.due_date else "",
                p.source + (" (truncated)" if p.truncated else ""),
            ]
            for p in plans
        ],
    )
    cut = [p for p in plans if p.truncated]
    if cut:
        r.section("Truncated input", tone="warn").callout(
            f"{len(cut)} document(s) had more than {TEXT_LIMIT} characters of text; only the first "
            f"{TEXT_LIMIT} were sent to the model: " + ", ".join(p.src.name for p in cut),
            label="capped",
        )
    if ics_path is not None:
        try:
            href = layout.link(ics_path, output_dir or ics_path.parent)
        except ValueError:
            href = ics_path.name
        r.section("Due dates calendar").cards([Card(ics_path.name, meta="upcoming due dates", href=href)])
    if skips:
        r.section("Skipped (left untouched)", tone="warn").table(
            ["File", "Reason"], [[str(s.src), s.reason] for s in skips]
        )
    return r


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", metavar="DIR", help="Inbox folder (default: LMSA_DOC_INBOX_DIR)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N documents")
    parser.add_argument("--apply", action="store_true", help="[writes] Move the documents (default is dry-run)")
    parser.add_argument("--undo", metavar="RUN_ID", help="[writes] Move a previous --apply run's documents back")
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore remembered answers and ask the model about every document again",
    )
    args = parser.parse_args(argv)

    if args.undo:
        notes = undo(args.undo)
        for n in notes:
            print(f"{n.status}: {n.path}" + (f" ({n.detail})" if n.detail else ""))
        restored = sum(1 for n in notes if n.status == "restored")
        print(f"restored {restored}, not restored {len(notes) - restored}")
        return

    cfg = load_config(inbox=args.inbox)
    if not cfg.inbox.is_dir():
        raise FileNotFoundError(f"inbox folder does not exist: {cfg.inbox}")
    files = inbox_files(cfg)
    if args.limit is not None:
        files = files[: args.limit]

    lm = LMStudio()
    lm.ensure_server()
    now = _now()
    output_dir = lm.cfg.output_dir
    with Store() as store, store.record_run("tasks.doc_inbox") as run:
        last_ok = store.last_successful_run("tasks.doc_inbox")
        memo = open_memo(store.conn, lm, fresh=args.fresh)
        plans, skips = build_plans(lm, cfg, files, memo)

        run_id = None
        log_path = None
        if args.apply:
            run_id = new_run_id(now)
            log_path, plans, apply_skips = apply_plans(cfg, plans, run_id)
            skips = skips + apply_skips
            if log_path is None:
                run_id = None

        events = due_events(plans, now.astimezone().date())
        ics_path = write_ics(events, layout.placed(output_dir, f"doc_inbox-due-{now:%Y%m%d-%H%M%S}.ics")) if events else None
        report = render_report(
            plans, cfg, applied=args.apply, skips=skips, run_id=run_id, ics_path=ics_path, output_dir=output_dir,
        )
        add_footer(report, memo, last_ok)
        path = report.write(out_dir=output_dir)
        run.summary = {
            "files": len(files), ("filed" if args.apply else "planned"): len(plans), "skipped": len(skips),
            "applied": args.apply, "due_dates": len(events), **memo.summary(),
        }
        if run_id:
            run.summary["run_id"] = run_id
        run.report_path = path
        print(path)
        if ics_path:
            print(ics_path)
        if args.apply:
            print(f"undo with: python -m tasks.doc_inbox --undo {run_id}" if run_id else "nothing moved; no run log written")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
