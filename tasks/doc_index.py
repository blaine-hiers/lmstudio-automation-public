"""Index personal documents for semantic search.

    python -m tasks.doc_index [--config config/doc_search.toml] [--limit 20]

Walks the roots in `config/doc_search.toml` (default `~/Documents`,
skipping `GitHub/` and hidden directories), chunks `.md`/`.txt`/`.pdf`/
`.docx` files, embeds each chunk via `lmsa.LMStudio.embed`, and upserts the
vectors into the store's `embeddings` table (namespace `docs`) via
`lmsa.vectors`. Chunk text and file metadata (path, chunk index, mtime,
size) live in this task's own `doc_chunks` table so a later run can tell,
by mtime+size, which files changed and re-embed only those; a file whose
chunks are unchanged but lack embeddings under the *current* embed model
(e.g. the model was swapped) is re-embedded from its stored chunk text
without being re-parsed.

After scanning, any previously-indexed path under a configured root that
no longer exists on disk (deleted or renamed away) has its chunks and
embeddings removed -- checked with `Path.exists()`, never inferred from
"not scanned this run", since `--limit` means plenty of real, unchanged
files go unscanned on any given run.

Read-only over the documents themselves -- nothing here writes to, moves,
or deletes a file in the scanned roots. `--limit` caps how many files a
single run processes (the owner's Documents folder may be large); files
beyond the cap are picked up by a later run once earlier ones are marked
unchanged.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import docx
import pypdf

from lmsa import LMStudio, Report, Tile, Store
from lmsa.vectors import embed_and_upsert

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"config": ["config/doc_search.toml"], "model": "embed"}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "doc_search.toml"
DEFAULT_ROOTS = ("~/Documents",)
DEFAULT_EXCLUDE_DIRS = ("GitHub",)
EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
NS = "docs"
CHUNK_WORDS = 800
CHUNK_OVERLAP_WORDS = 120

_CREATE_DOC_CHUNKS = """
CREATE TABLE IF NOT EXISTS doc_chunks (
    path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    PRIMARY KEY (path, chunk_index)
);
"""


@dataclass
class Config:
    roots: list[Path]
    exclude_dirs: set[str]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    data = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    roots = [Path(r).expanduser() for r in data.get("roots", DEFAULT_ROOTS)]
    exclude_dirs = {d.lower() for d in data.get("exclude_dirs", DEFAULT_EXCLUDE_DIRS)}
    return Config(roots=roots, exclude_dirs=exclude_dirs)


def iter_files(config: Config):
    """Yield candidate files under `config.roots`, deepest-first-stable and
    sorted for a deterministic, resumable order across runs."""
    for root in config.roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d.lower() not in config.exclude_dirs
            )
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.suffix.lower() in EXTENSIONS:
                    yield p


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"unsupported extension: {suffix}")


def chunk_text(text: str, *, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Word-count-based chunks (a cheap stand-in for a tokenizer) with overlap."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    unchanged: int = 0
    reembedded: int = 0
    pruned: int = 0
    chunks_written: int = 0
    truncated: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _embedded_item_ids(conn: sqlite3.Connection, item_ids: list[str], model: str) -> set[str]:
    if not item_ids:
        return set()
    placeholders = ",".join("?" * len(item_ids))
    rows = conn.execute(
        f"SELECT item_id FROM embeddings WHERE ns=? AND model=? AND item_id IN ({placeholders})",
        (NS, model, *item_ids),
    ).fetchall()
    return {r[0] for r in rows}


def _delete_path(conn: sqlite3.Connection, path_str: str) -> None:
    """Remove `doc_chunks` rows for `path_str` and every embedding (any
    model) keyed off them -- stale chunk text invalidates all of them,
    not just the currently configured embed model's."""
    old = conn.execute("SELECT chunk_index FROM doc_chunks WHERE path=?", (path_str,)).fetchall()
    if old:
        conn.executemany(
            "DELETE FROM embeddings WHERE ns=? AND item_id=?",
            [(NS, f"{path_str}#{idx}") for (idx,) in old],
        )
    conn.execute("DELETE FROM doc_chunks WHERE path=?", (path_str,))


def _replace_chunks(
    conn: sqlite3.Connection,
    lm: LMStudio,
    *,
    path_str: str,
    mtime: float,
    size: int,
    chunks: list[str],
    model: str,
) -> None:
    _delete_path(conn, path_str)

    items = [(f"{path_str}#{i}", chunk) for i, chunk in enumerate(chunks)]
    embed_and_upsert(conn, lm, ns=NS, model=model, items=items)

    conn.executemany(
        "INSERT INTO doc_chunks(path, chunk_index, text, mtime, size) VALUES (?, ?, ?, ?, ?)",
        [(path_str, i, chunk, mtime, size) for i, chunk in enumerate(chunks)],
    )
    conn.commit()


def _prune_missing(conn: sqlite3.Connection, config: Config) -> list[str]:
    """Remove chunks + embeddings for previously-indexed paths, under a
    currently configured root, that no longer exist on disk (deleted or
    renamed away). Confirmed with `Path.exists()` -- a path merely absent
    from this run's scan (because `--limit` capped it) is not evidence it
    is gone.
    """
    indexed_paths = [row[0] for row in conn.execute("SELECT DISTINCT path FROM doc_chunks").fetchall()]
    gone = [
        path_str for path_str in indexed_paths
        if any(_is_under(Path(path_str), root) for root in config.roots)
        and not Path(path_str).exists()
    ]
    for path_str in gone:
        _delete_path(conn, path_str)
    if gone:
        conn.commit()
    return gone


def index_documents(
    conn: sqlite3.Connection,
    lm: LMStudio,
    config: Config,
    *,
    limit: int | None = None,
    model: str | None = None,
) -> IndexStats:
    conn.executescript(_CREATE_DOC_CHUNKS)
    model = model or lm.cfg.embed_model
    stats = IndexStats()

    candidates = list(iter_files(config))
    to_process = candidates[:limit] if limit is not None else candidates
    stats.truncated = limit is not None and len(candidates) > limit

    for path in to_process:
        stats.scanned += 1
        path_str = str(path)
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
            existing = conn.execute(
                "SELECT chunk_index, text, mtime, size FROM doc_chunks WHERE path=? ORDER BY chunk_index",
                (path_str,),
            ).fetchall()
            content_changed = not existing or existing[0][2] != mtime or existing[0][3] != size

            if content_changed:
                text = extract_text(path)
                chunks = chunk_text(text)
                _replace_chunks(
                    conn, lm, path_str=path_str, mtime=mtime, size=size, chunks=chunks, model=model,
                )
                stats.indexed += 1
                stats.chunks_written += len(chunks)
                continue

            # Content unchanged: reuse the stored chunk text, but the
            # embed model may have changed since the last run -- embed
            # only the chunks that don't have a vector under `model` yet.
            item_ids = [f"{path_str}#{chunk_index}" for chunk_index, _, _, _ in existing]
            embedded = _embedded_item_ids(conn, item_ids, model)
            missing = [
                (f"{path_str}#{chunk_index}", chunk_text_)
                for chunk_index, chunk_text_, _, _ in existing
                if f"{path_str}#{chunk_index}" not in embedded
            ]
            if missing:
                embed_and_upsert(conn, lm, ns=NS, model=model, items=missing)
                stats.reembedded += 1
            else:
                stats.unchanged += 1
        except Exception as exc:  # a corrupt/unreadable file must not abort the run
            stats.errors.append((path_str, f"{type(exc).__name__}: {exc}"))

    stats.pruned = len(_prune_missing(conn, config))
    return stats


def build_report(stats: IndexStats) -> Report:
    r = Report(
        "Document index",
        kicker="lmsa · doc search",
        subtitle="Local semantic index over configured document roots.",
        sticker=f"{stats.indexed} updated" if stats.indexed else "up to date",
        sticker_tone="acc",
        source="tasks.doc_index",
    )
    r.tiles([
        Tile("Scanned", stats.scanned, tone="info"),
        Tile("Indexed (changed)", stats.indexed, tone="acc"),
        Tile("Re-embedded", stats.reembedded, tone="violet"),
        Tile("Unchanged", stats.unchanged, tone="muted"),
        Tile("Pruned (gone)", stats.pruned, tone="warn" if stats.pruned else "muted"),
        Tile("Chunks written", stats.chunks_written, tone="violet"),
        Tile("Errors", len(stats.errors), tone="hot" if stats.errors else "muted"),
    ])
    if stats.truncated:
        r.section("Note", tone="warn").callout(
            "Hit --limit before scanning every candidate file; the rest will "
            "be picked up on a later run.",
            label="truncated",
            tone="warn",
        )
    if stats.errors:
        section = r.section("Files that failed to index", tone="hot")
        for path_str, error in stats.errors:
            section.callout(error, label=path_str, tone="hot")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None, help="max files to process this run")
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()
    config = load_config(args.config)

    with Store() as db:
        with db.record_run("doc_index") as run:
            stats = index_documents(db.conn, lm, config, limit=args.limit)
            report_path = build_report(stats).write(out_dir=lm.cfg.output_dir)
            run.summary = {
                "scanned": stats.scanned,
                "indexed": stats.indexed,
                "reembedded": stats.reembedded,
                "unchanged": stats.unchanged,
                "pruned": stats.pruned,
                "chunks_written": stats.chunks_written,
                "errors": len(stats.errors),
                "truncated": stats.truncated,
            }
            run.report_path = report_path

    print(f"{stats.indexed} file(s) indexed, {len(stats.errors)} error(s) -> {report_path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
