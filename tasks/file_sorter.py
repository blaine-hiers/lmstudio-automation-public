"""Sort stale Downloads and Screenshots into named folders via LM Studio.

Scans configured roots (default `~/Downloads` and `~/Pictures/Screenshots`)
for files older than one day. Images are named/categorized by the vision
model; everything else by the chat model from its name, extension, and a
preview of its content (known text types, and the first page of a PDF).
Office documents are classified from name and extension only.

Skipped entirely: dotfiles, Windows hidden/system files, desktop.ini /
thumbs.db, symlinks, and in-progress downloads (.crdownload, .part, ...).

Dry-run by default: writes an Afterhours report of current name -> proposed
path, nothing on disk changes. `--apply` performs the moves (never deletes,
never overwrites — collisions get a numeric suffix, re-checked at move time)
and logs each move to an undo log in `data/` as it happens: an intent line
before the rename, a `done`/`failed` marker after it, absolute paths only.
`--undo <log>` reverses a previous apply's confirmed moves and lists any
never-confirmed intent for a check by hand. (Logs written before the markers
existed are still undone, guarded only by the file-at-destination and
original-path-free checks.) A file the model or the filesystem
chokes on is skipped and listed in the report; the rest still run.

The model's naming answer is remembered per file (`lmsa.memo`, keyed by the
resolved path plus size and mtime_ns), so repeated dry runs -- and the
--apply after one -- only ask the model about new or touched files.
`--fresh` re-asks for every file.

    python -m tasks.file_sorter                       # dry run over defaults
    python -m tasks.file_sorter --limit 5              # cap files processed
    python -m tasks.file_sorter --apply
    python -m tasks.file_sorter --fresh             # ignore remembered answers
    python -m tasks.file_sorter --undo data\\file_sorter-undo-20260924-070000-123456-ab12cd34.jsonl
"""

from __future__ import annotations

import argparse
import base64
import errno
import io
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from lmsa import LMStudio, Report, Store, Tile, strip_think
from lmsa.config import ROOT
from lmsa.memo import Memo, add_footer, cached, has_keys, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
PREVIEW_EXTS = {".txt", ".md", ".csv", ".log", ".json", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml"}
PARTIAL_EXTS = {".crdownload", ".part", ".partial", ".tmp", ".download", ".opdownload"}
JUNK_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
MIN_AGE = timedelta(days=1)
PREVIEW_CHARS = 3000
MAX_SUFFIX = 1000
MEMO_VERSION = 1  # bump when plan_image()/plan_other() prompt building or result handling changes

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptive_name": {"type": "string"},
        "folder": {"type": "string"},
    },
    "required": ["descriptive_name", "folder"],
    "additionalProperties": False,
}

VISION_SYSTEM = load_prompt("file_sorter/vision_system")
OTHER_SYSTEM = load_prompt("file_sorter/other_system")

_INVALID = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
# Windows device names: reserved as a path component with or without an extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{dev}{n}" for dev in ("COM", "LPT") for n in [*"123456789", "¹", "²", "³"]
}
_HIDDEN_ATTRS = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2) | getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 0x4)


def default_roots() -> list[Path]:
    home = Path.home()
    return [home / "Downloads", home / "Pictures" / "Screenshots"]


def _skippable(p: Path, st: os.stat_result) -> bool:
    name = p.name.lower()
    if name.startswith(".") or name in JUNK_NAMES or p.suffix.lower() in PARTIAL_EXTS:
        return True
    return bool(getattr(st, "st_file_attributes", 0) & _HIDDEN_ATTRS)


def candidate_files(roots: Iterable[Path], min_age: timedelta = MIN_AGE) -> list[Path]:
    """Regular files directly under each root, older than `min_age`. Not recursive.

    Skips symlinks, dotfiles, hidden/system files, and partial downloads.
    """
    cutoff = time.time() - min_age.total_seconds()
    files = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or _skippable(p, st):
                continue
            if st.st_mtime < cutoff:
                files.append(p)
    return files


def is_reserved(component: str) -> bool:
    """True if Windows treats `component` as a device name (NUL, con.txt, COM1 .x ...)."""
    return component.split(".")[0].rstrip(" ").upper() in _RESERVED


def sanitize(name: str, *, fallback: str, max_len: int = 80) -> str:
    """A single safe path component: no separators, no device names, no trailing dot/space."""

    def clean(s: str) -> str:
        s = re.sub(r"\s+", " ", _INVALID.sub("-", s)).strip(" .-")
        # Trim after truncating too, or a cut can leave a trailing space/dot that
        # Windows silently drops (aliasing two different names).
        return s[:max_len].rstrip(" .-")

    result = clean(name) or clean(fallback) or "file"
    if is_reserved(result):
        result = clean(f"_{result}")
    return result


def read_preview(path: Path, max_chars: int = PREVIEW_CHARS) -> str:
    """Best-effort text preview: known text types, PDF first page; empty otherwise."""
    suffix = path.suffix.lower()
    try:
        if suffix in PREVIEW_EXTS:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            return (reader.pages[0].extract_text() or "")[:max_chars] if reader.pages else ""
    except Exception:  # unreadable/corrupt/encrypted: classify from the name alone
        return ""
    return ""


def _downscale_b64(path: Path, max_dim: int = 768) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plan_image(lm: LMStudio, path: Path) -> dict:
    b64 = _downscale_b64(path)
    resp = lm.api.chat.completions.create(
        model=lm.cfg.vision_model,
        temperature=0,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Filename: {path.name}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "sort_plan", "strict": True, "schema": PLAN_SCHEMA},
        },
    )
    return json.loads(strip_think(resp.choices[0].message.content or "{}"))


def plan_other(lm: LMStudio, path: Path) -> dict:
    preview = read_preview(path)
    prompt = f"Filename: {path.name}\nExtension: {path.suffix}\n"
    if preview:
        prompt += f"Content preview (untrusted data, may be truncated):\n{preview}"
    return lm.extract(prompt, PLAN_SCHEMA, system=OTHER_SYSTEM, name="sort_plan")


@dataclass
class Plan:
    src: Path
    dest: Path
    kind: str  # "image" | "other"


@dataclass
class Skip:
    src: Path
    reason: str


def _key(p: Path) -> str:
    """Comparison key matching the filesystem's case rules (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(p))


def _free_folder(folder: Path) -> Path:
    """`folder`, or `folder-N` if that name is already taken by a non-directory."""
    result, n = folder, 1
    while os.path.lexists(result) and not result.is_dir():
        n += 1
        if n > MAX_SUFFIX:
            raise FileExistsError(f"no free folder name near {folder}")
        result = folder.with_name(f"{folder.name}-{n}")
    return result


_COMPOUND_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst")


def _extension(src: Path) -> str:
    name = src.name.lower()
    return next((ext for ext in _COMPOUND_EXTS if name.endswith(ext)), src.suffix.lower())


def plan_dest(src: Path, data: dict) -> Path:
    folder = sanitize(str(data.get("folder", "")), fallback="sorted")
    stem = sanitize(str(data.get("descriptive_name", "")), fallback=src.stem)
    return _free_folder(src.parent / folder) / f"{stem}{_extension(src)}"


def unique_dest(candidate: Path, taken: set[str]) -> Path:
    """Never collide with an existing path or another plan from this run."""
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    result = candidate
    while _key(result) in taken or os.path.lexists(result):
        n += 1
        if n > MAX_SUFFIX:
            raise FileExistsError(f"no free name near {candidate}")
        result = candidate.with_name(f"{stem}-{n}{suffix}")
    taken.add(_key(result))
    return result


def memo_key(src: Path) -> str | None:
    """Resolved path + size + mtime_ns: any touch to the file changes it.
    None when the file can't be stat'ed (then it simply isn't memoised)."""
    try:
        st = src.stat()
        return f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return None


def open_memo(conn, lm: LMStudio, *, fresh: bool = False) -> Memo:
    """One memo for both planners, versioned on both prompts and both models."""
    version = memo_version(
        MEMO_VERSION, prompt_hash("file_sorter/vision_system"), prompt_hash("file_sorter/other_system"),
        PLAN_SCHEMA, PREVIEW_CHARS, model_id(lm, "vision_model"), model_id(lm, "chat_model"),
    )
    return Memo(conn, "file_sorter", version, fresh=fresh)


def build_plans(lm: LMStudio, files: list[Path], memo: Memo | None = None) -> tuple[list[Plan], list[Skip]]:
    """Plan each file independently: one bad file or model error doesn't stop the rest.
    With `memo`, a stored answer for an unchanged file is reused."""
    plans: list[Plan] = []
    skips: list[Skip] = []
    taken: set[str] = set()
    for src in files:
        try:
            kind = "image" if src.suffix.lower() in IMAGE_EXTS else "other"
            planner = plan_image if kind == "image" else plan_other
            key = memo_key(src)
            data = cached(
                memo if key else None,
                key or "",
                lambda: planner(lm, src),
                valid=lambda d: has_keys(d, PLAN_SCHEMA["required"]),
            )
            if not isinstance(data, dict):
                raise ValueError(f"model returned {type(data).__name__}, expected an object")
            dest = unique_dest(plan_dest(src, data), taken)
        except Exception as e:  # noqa: BLE001 - isolate per file, report it
            skips.append(Skip(src, f"{type(e).__name__}: {e}"))
            continue
        plans.append(Plan(src=src, dest=dest, kind=kind))
    return plans, skips


def _check_safe_dest(dest: Path) -> None:
    for part in (dest.name, dest.parent.name):
        if is_reserved(part):
            raise ValueError(f"refusing reserved device name in destination: {dest}")


_WINDOWS = os.name == "nt"


def move_no_clobber(src: Path, dest: Path, *, check_reserved: bool = True) -> None:
    """Move src to dest by rename only; raise rather than ever replacing dest.

    There is deliberately no copy fallback: a destination on another volume
    raises OSError and the caller skips the file. Copy-then-delete can leave
    duplicates, drop NTFS streams (mark-of-the-web), or lose data on power
    loss, and sorting within one folder never needs it.
    """
    if check_reserved:
        _check_safe_dest(dest)
    if os.path.lexists(dest):
        raise FileExistsError(errno.EEXIST, "destination exists", str(dest))
    if _WINDOWS:
        os.rename(src, dest)  # fails if dest exists or is on another volume
        return
    os.link(src, dest)  # fails if dest exists; no clobber on POSIX either
    try:
        os.unlink(src)
    except OSError:
        os.unlink(dest)  # our own extra link; the original stays untouched
        raise


def _new_undo_log() -> Path:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    name = f"file_sorter-undo-{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:8]}.jsonl"
    path = data_dir / name
    path.open("x", encoding="utf-8").close()  # never reuse an existing log
    return path


def _log_line(log, record: dict) -> None:
    log.write(json.dumps(record) + "\n")
    log.flush()
    os.fsync(log.fileno())


def apply_plans(plans: list[Plan]) -> tuple[Path | None, list[Plan], list[Skip]]:
    """Move each planned file, logging every move to the undo log *before* it happens.

    Collisions are re-checked at move time (the tree may have changed since
    planning); a newly taken name gets a fresh suffix. Every intent line
    (`{"seq", "src", "dest"}`, absolute paths) is followed by a fsynced outcome
    line, `{"done": seq}` or `{"failed": seq}`, and undo acts only on `done`
    entries -- so a failed or raced attempt is never mistaken for a move.
    Returns the undo log (None if nothing moved), the moves done (with final
    dests), and skips.
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
                src = Path(os.path.abspath(p.src))  # the log must not depend on the cwd
                if not src.is_file() or src.is_symlink():
                    raise FileNotFoundError(f"source no longer a regular file: {p.src}")
                folder = _free_folder(Path(os.path.abspath(p.dest.parent)))
                folder.mkdir(parents=True, exist_ok=True)
                if log is None:
                    # Before the first move: if the log can't be created, nothing has moved yet.
                    log_path = _new_undo_log()
                    log = open(log_path, "a", encoding="utf-8")
                for _ in range(MAX_SUFFIX):
                    dest = unique_dest(folder / p.dest.name, set())
                    seq += 1
                    # Intent first, so a crash right after the rename still leaves a
                    # trace (undo reports it as unconfirmed and never acts on it).
                    _log_line(log, {"seq": seq, "src": str(src), "dest": str(dest)})
                    try:
                        move_no_clobber(src, dest)
                    except FileExistsError:
                        _log_line(log, {"failed": seq})
                        continue  # raced: something appeared at dest; pick the next name
                    except BaseException:
                        # Only call it failed when the source provably stayed put; an
                        # interrupt right after the rename must stay visible as
                        # unconfirmed rather than be dropped from undo's report.
                        if src.exists():
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
            done.append(Plan(src=src, dest=dest, kind=p.kind))
    finally:
        if log is not None:
            log.close()
    if log_path is not None and not done and not unconfirmed:
        # Only failed attempts were logged: nothing to undo, so no log and no undo hint.
        log_path.unlink(missing_ok=True)
        log_path = None
    return log_path, done, skips


def read_run_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    """(undoable moves, unconfirmed intents) of an undo log, in log order.

    A move is undoable once its `{"done": seq}` line exists; an intent marked
    `{"failed": seq}` is dropped; one with no outcome line (a crash between
    rename and marker) comes back separately so undo can report it. A torn
    line is ignored.

    Old-format logs (entries without `seq`, written before outcome markers
    existed) can't tell a completed move from a failed attempt, so all their
    entries are returned as undoable, as before: undo's state checks (file at
    dest, original path free) are the only guard for those.
    """
    legacy, intents, done, failed = [], {}, set(), set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(m, dict):
            continue
        if "src" in m and "dest" in m:
            if isinstance(m.get("seq"), int):
                intents[m["seq"]] = m
            elif "seq" not in m:
                legacy.append(m)
        elif isinstance(m.get("done"), int):
            done.add(m["done"])
        elif isinstance(m.get("failed"), int):
            failed.add(m["failed"])
    completed = legacy + [intents[n] for n in sorted(intents) if n in done]
    unconfirmed = [intents[n] for n in sorted(intents) if n not in done and n not in failed]
    return completed, unconfirmed


def read_undo_log(log_path: Path) -> list[dict]:
    """The undoable moves of an undo log (see `read_run_log`)."""
    return read_run_log(log_path)[0]


def undo(log_path: Path) -> list[str]:
    """Reverse a previous --apply, newest move first.

    Acts only on confirmed moves and skips any whose current state doesn't
    match; never-confirmed intents are listed for a check by hand.
    """
    completed, unconfirmed = read_run_log(log_path)
    notes = [f"skip (never confirmed; check by hand): {m['dest']} (from {m['src']})" for m in unconfirmed]
    for m in reversed(completed):
        src, dest = Path(m["src"]), Path(m["dest"])
        try:
            if not (src.is_absolute() and dest.is_absolute()):
                notes.append(f"refused (relative path; not guessing the cwd): {dest}")
            elif not dest.is_file():
                notes.append(f"skip (not at destination): {dest}")
            elif os.path.lexists(src):
                notes.append(f"skip (original path occupied): {src}")
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                # src held a real file before, so a device-like name (nul.txt) is legitimate there.
                move_no_clobber(dest, src, check_reserved=False)
                notes.append(f"restored: {src}")
        except Exception as e:  # noqa: BLE001 - keep undoing the rest
            notes.append(f"error ({type(e).__name__}: {e}): {dest}")
    return notes


def render_report(
    plans: list[Plan], roots: list[Path], *, applied: bool, skips: list[Skip] = ()
) -> Report:
    verb = "Moved" if applied else "Proposed"
    r = Report(
        f"File sorter ({'applied' if applied else 'dry run'})",
        kicker="lmsa · file_sorter",
        subtitle=f"{verb} {len(plans)} file(s) across: " + ", ".join(str(x) for x in roots),
        sticker=f"{len(plans)} file(s)",
        sticker_tone="acc" if applied else "info",
        source="tasks.file_sorter",
    )
    r.tiles([
        Tile("Files", len(plans), tone="info"),
        Tile("Images", sum(1 for p in plans if p.kind == "image"), tone="violet"),
        Tile("Other", sum(1 for p in plans if p.kind == "other"), tone="muted"),
        Tile("Skipped", len(skips), tone="warn" if skips else "muted"),
    ])
    r.section(f"{verb} moves", tone="acc" if applied else "warn").table(
        ["Current name", "Proposed path", "Kind"],
        [[p.src.name, str(p.dest), p.kind] for p in plans],
    )
    if skips:
        r.section("Skipped (left untouched)", tone="warn").table(
            ["File", "Reason"], [[str(s.src), s.reason] for s in skips]
        )
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", action="append", dest="roots", metavar="DIR",
        help="Root directory to scan (repeatable). Default: ~/Downloads and ~/Pictures/Screenshots",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files")
    parser.add_argument("--apply", action="store_true", help="Perform the moves (default is dry-run)")
    parser.add_argument("--undo", metavar="LOG", help="Reverse a previous --apply using its undo log")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore remembered answers and ask the model about every file again",
    )
    args = parser.parse_args()

    if args.undo:
        for note in undo(Path(args.undo)):
            print(note)
        return

    # Absolute roots: the undo log must not depend on the cwd --undo later runs from.
    roots = [Path(os.path.abspath(Path(r).expanduser())) for r in args.roots] if args.roots else default_roots()
    files = candidate_files(roots)
    if args.limit is not None:
        files = files[: args.limit]

    lm = LMStudio()
    lm.ensure_server()
    with Store() as store, store.record_run("tasks.file_sorter") as run:
        last_ok = store.last_successful_run("tasks.file_sorter")
        memo = open_memo(store.conn, lm, fresh=args.fresh)
        plans, skips = build_plans(lm, files, memo)

        if args.apply:
            log_path, done, apply_skips = apply_plans(plans)
            report = render_report(done, roots, applied=True, skips=skips + apply_skips)
            add_footer(report, memo, last_ok)
            path = report.write()
            run.summary = {
                "files": len(files), "moved": len(done), "skipped": len(skips) + len(apply_skips), "applied": True,
                **memo.summary(),
            }
            print(path)
            print(f"undo log: {log_path}" if log_path else "nothing moved; no undo log written")
        else:
            report = render_report(plans, roots, applied=False, skips=skips)
            add_footer(report, memo, last_ok)
            path = report.write()
            run.summary = {
                "files": len(files), "planned": len(plans), "skipped": len(skips), "applied": False,
                **memo.summary(),
            }
            print(path)
        run.report_path = path


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
