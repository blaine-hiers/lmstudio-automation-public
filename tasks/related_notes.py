"""Related notes: surface neighbours of recently changed documents.

    python -m tasks.related_notes [--days 7] [--k 5] [--min-score 0.75]
                                   [--dup-score 0.95] [--anchor-cap 25]
                                   [--explain] [--fresh]

Reuses the embeddings `tasks.doc_index` already wrote to the store's
`embeddings` table (namespace `docs`) -- this never calls the embed model
itself. Relatedness is computed **per document**, not per chunk: every
document's chunk vectors (from the currently configured embed model) are
mean-pooled into a single normalized vector, then compared with plain
cosine similarity via one `numpy` matrix multiply per anchor. Chunk-pair
comparisons are deliberately avoided -- they'd multiply the compute cost by
the average chunk count per document and let one long document dominate a
short one just for having more chunks.

An "anchor" is a document whose `doc_chunks.mtime` -- the file's own mtime,
recorded by `doc_index` the same run it embedded the file, and the only
"when did this change" signal the index keeps -- falls within the last
`--days` days. Only anchors seed a comparison; older documents can appear as
a match but never as the document a report groups by. Anchors are capped at
`--anchor-cap` (oldest-changed dropped first) to keep the run's compute
bounded regardless of how many documents changed at once; a capped run says
so in the report.

For each anchor, the top `--k` other documents scoring at or above
`--min-score` (and below `--dup-score`) are listed as "related"; any pair
scoring at or above `--dup-score` is a near-duplicate, listed once
regardless of which side is the anchor (or whether both sides are anchors)
-- pairs are deduplicated on the unordered `{path_a, path_b}` set.

With `--explain`, a one-line "why related" comes from the chat model for
each pair actually shown (grounded in each document's first chunk only,
truncated to `EXCERPT_CHARS`), memoized per unordered pair + both files'
mtimes (`lmsa.memo`) so a rerun -- or the same pair appearing on both sides
of a mutual match -- costs at most one model call. Off by default: the
report is fully useful from cosine scores alone, and a weekly run over
however many documents changed should not have to load the chat model
just to produce a table of matches.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from lmsa import LMStudio, Report, Store, Tile
from lmsa.memo import Memo, add_footer, cached, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.store import unpack_vector

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
# No `config` entry: unlike doc_index/doc_search this reads only the store's
# already-built index (doc_chunks + embeddings), never config/doc_search.toml.
SETUP = {"model": "chat"}

NS = "docs"  # tasks.doc_index's embeddings namespace
DEFAULT_DAYS = 7
DEFAULT_K = 5
RELATED_THRESHOLD = 0.75
DUP_THRESHOLD = 0.95
ANCHOR_CAP = 25
EXCERPT_CHARS = 800  # of each document's first chunk, sent to the model per "why"
WHY_MAX_TOKENS = 80
WHY_CHAR_LIMIT = 300
MEMO_VERSION = 1  # bump when explain_pair()'s prompt building or result handling changes

WHY_SYSTEM = load_prompt("related_notes/why_system")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Related:
    path: str
    score: float
    why: str | None = None
    why_error: str | None = None


@dataclass
class NearDup:
    path_a: str
    path_b: str
    score: float
    why: str | None = None
    why_error: str | None = None


@dataclass
class RelatedNotesResult:
    related: dict[str, list[Related]] = field(default_factory=dict)  # anchor path -> matches
    near_dups: list[NearDup] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)  # anchors actually used (after the cap)
    doc_count: int = 0  # documents with a usable vector
    recent_count: int = 0  # documents changed in the window, before the cap
    recent_without_vector: int = 0  # changed in the window, but not embedded (yet)
    truncated: bool = False  # recent_count > len(anchors): the cap dropped some


def _doc_mtimes(conn: sqlite3.Connection) -> dict[str, float]:
    """One mtime per indexed path (all of a path's chunks share the mtime
    `doc_index` recorded when it last (re)wrote them)."""
    rows = conn.execute("SELECT path, MIN(mtime) FROM doc_chunks GROUP BY path").fetchall()
    return {path: mtime for path, mtime in rows}


def build_doc_vectors(conn: sqlite3.Connection, model: str) -> dict[str, np.ndarray]:
    """Mean-pool each document's chunk vectors (under `model`) into one
    unit-normalized vector per path, so a later dot product between two
    rows is exactly their cosine similarity. A document with no embedded
    chunks under `model` (never indexed, or indexed under a different embed
    model) is absent from the result -- it can be neither an anchor nor a
    match until `doc_index` embeds it under the current model.
    """
    rows = conn.execute(
        "SELECT item_id, vector FROM embeddings WHERE ns=? AND model=?", (NS, model)
    ).fetchall()
    by_path: dict[str, list[np.ndarray]] = {}
    for item_id, blob in rows:
        path, sep, _ = item_id.rpartition("#")
        if not sep:
            continue  # not one of doc_index's "<path>#<chunk_index>" ids
        by_path.setdefault(path, []).append(np.asarray(unpack_vector(blob), dtype=np.float32))

    doc_vectors: dict[str, np.ndarray] = {}
    for path, vecs in by_path.items():
        mean = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            doc_vectors[path] = mean / norm
    return doc_vectors


def recent_anchors(
    mtimes: dict[str, float],
    doc_vectors: dict[str, np.ndarray],
    *,
    now: datetime,
    days: int = DEFAULT_DAYS,
    cap: int = ANCHOR_CAP,
) -> tuple[list[str], int, int]:
    """Paths changed within `days` of `now`, newest first, capped at `cap`.

    Returns `(anchors, recent_count, recent_without_vector)` -- `recent_count`
    is every changed path regardless of whether it has a vector yet (so a
    caller can tell "capped" apart from "not embedded"), never affected by
    the cap.
    """
    cutoff = now.timestamp() - days * 86400
    recent = sorted(
        (path for path, mtime in mtimes.items() if mtime >= cutoff),
        key=lambda p: mtimes[p],
        reverse=True,
    )
    without_vector = sum(1 for p in recent if p not in doc_vectors)
    with_vector = [p for p in recent if p in doc_vectors]
    return with_vector[:cap], len(recent), without_vector


def compute_related(
    doc_vectors: dict[str, np.ndarray],
    anchors: list[str],
    *,
    k: int = DEFAULT_K,
    related_threshold: float = RELATED_THRESHOLD,
    dup_threshold: float = DUP_THRESHOLD,
) -> tuple[dict[str, list[Related]], list[NearDup]]:
    """Top-`k` related documents per anchor (`related_threshold` <= score <
    `dup_threshold`), plus every near-duplicate pair (score >= `dup_threshold`)
    touching an anchor, deduplicated on the unordered pair -- a mutual match
    between two anchors is one row, not two.
    """
    paths = list(doc_vectors.keys())
    if not paths or not anchors:
        return {}, []
    matrix = np.stack([doc_vectors[p] for p in paths])
    index = {p: i for i, p in enumerate(paths)}

    related: dict[str, list[Related]] = {}
    near_dups: list[NearDup] = []
    seen_dup_pairs: set[frozenset[str]] = set()

    for anchor in anchors:
        scores = matrix @ matrix[index[anchor]]
        ranked = sorted(
            ((paths[j], float(scores[j])) for j in range(len(paths)) if paths[j] != anchor),
            key=lambda t: -t[1],
        )
        matches = [
            Related(path=p, score=s)
            for p, s in ranked
            if related_threshold <= s < dup_threshold
        ][:k]
        if matches:
            related[anchor] = matches

        for p, s in ranked:
            if s < dup_threshold:
                continue
            pair = frozenset((anchor, p))
            if pair in seen_dup_pairs:
                continue
            seen_dup_pairs.add(pair)
            a, b = sorted((anchor, p))
            near_dups.append(NearDup(path_a=a, path_b=b, score=s))

    near_dups.sort(key=lambda d: -d.score)
    return related, near_dups


def find_related(
    conn: sqlite3.Connection,
    model: str,
    *,
    now: datetime,
    days: int = DEFAULT_DAYS,
    k: int = DEFAULT_K,
    related_threshold: float = RELATED_THRESHOLD,
    dup_threshold: float = DUP_THRESHOLD,
    anchor_cap: int = ANCHOR_CAP,
) -> RelatedNotesResult:
    mtimes = _doc_mtimes(conn)
    doc_vectors = build_doc_vectors(conn, model)
    anchors, recent_count, without_vector = recent_anchors(
        mtimes, doc_vectors, now=now, days=days, cap=anchor_cap
    )
    related, near_dups = compute_related(
        doc_vectors, anchors, k=k, related_threshold=related_threshold, dup_threshold=dup_threshold
    )
    return RelatedNotesResult(
        related=related,
        near_dups=near_dups,
        anchors=anchors,
        doc_count=len(doc_vectors),
        recent_count=recent_count,
        recent_without_vector=without_vector,
        truncated=len(anchors) < recent_count - without_vector,
    )


# -- optional "why related" (LLM, memo'd) -------------------------------------


def _first_chunk_text(conn: sqlite3.Connection, path: str) -> str | None:
    row = conn.execute(
        "SELECT text FROM doc_chunks WHERE path=? AND chunk_index=0", (path,)
    ).fetchone()
    return row[0] if row else None


def pair_key(path_a: str, mtime_a: float, path_b: str, mtime_b: float) -> str:
    """A key stable under swapping A and B, so a symmetric pair -- shown
    once under each side, or once as "related" and once as a near-duplicate
    -- costs one model call, not two."""
    a, b = sorted([(path_a, mtime_a), (path_b, mtime_b)])
    return f"{a[0]}@{a[1]!r}::{b[0]}@{b[1]!r}"


def _why_prompt(path_a: str, text_a: str, path_b: str, text_b: str) -> str:
    return (
        f"Note A ({path_a}):\n{text_a[:EXCERPT_CHARS]}\n\n"
        f"Note B ({path_b}):\n{text_b[:EXCERPT_CHARS]}"
    )


def explain_pair(
    lm: LMStudio,
    memo: Memo | None,
    path_a: str,
    mtime_a: float,
    path_b: str,
    mtime_b: float,
    text_a: str | None,
    text_b: str | None,
) -> tuple[str | None, str | None]:
    """One plain-text sentence on why the two documents are related, or
    `(None, error)`. Never raises: a bad/missing excerpt or a model failure
    must not stop the rest of the report from rendering."""
    if not text_a or not text_b:
        return None, "excerpt unavailable (re-run doc_index)"
    key = pair_key(path_a, mtime_a, path_b, mtime_b)
    try:
        why = cached(
            memo,
            key,
            lambda: lm.chat(
                _why_prompt(path_a, text_a, path_b, text_b),
                system=WHY_SYSTEM,
                max_tokens=WHY_MAX_TOKENS,
            ),
            valid=lambda r: isinstance(r, str) and bool(r.strip()),
        )
    except Exception as exc:  # a single bad pair must not kill the run
        return None, str(exc)
    if not why or not why.strip():
        return None, "model returned no text"
    return why.strip()[:WHY_CHAR_LIMIT], None


def explain_all(
    conn: sqlite3.Connection,
    lm: LMStudio,
    memo: Memo | None,
    mtimes: dict[str, float],
    result: RelatedNotesResult,
) -> None:
    """Fill in `.why` / `.why_error` on every `Related` and `NearDup` in
    `result`, in place. A pair shared between two lists (a mutual match
    that's also close to the dup threshold on the other side, or shown
    under two anchors) is asked about at most once thanks to `explain_pair`'s
    memo key -- the second lookup is a cache hit.
    """
    texts: dict[str, str | None] = {}

    def text_of(path: str) -> str | None:
        if path not in texts:
            texts[path] = _first_chunk_text(conn, path)
        return texts[path]

    for anchor, matches in result.related.items():
        for m in matches:
            m.why, m.why_error = explain_pair(
                lm, memo, anchor, mtimes.get(anchor, 0.0), m.path, mtimes.get(m.path, 0.0),
                text_of(anchor), text_of(m.path),
            )
    for d in result.near_dups:
        d.why, d.why_error = explain_pair(
            lm, memo, d.path_a, mtimes.get(d.path_a, 0.0), d.path_b, mtimes.get(d.path_b, 0.0),
            text_of(d.path_a), text_of(d.path_b),
        )


# -- report --------------------------------------------------------------------


def build_report(result: RelatedNotesResult) -> Report:
    related_pairs = sum(len(v) for v in result.related.values())
    r = Report(
        "Related notes",
        kicker="lmsa · doc search",
        subtitle=f"Documents changed in the last run's window, matched against {result.doc_count} indexed document(s).",
        sticker=f"{related_pairs} related · {len(result.near_dups)} near-dup" if result.anchors else "no recent changes",
        sticker_tone="acc" if result.anchors else "muted",
        source="tasks.related_notes",
    )
    r.tiles([
        Tile("Anchors", len(result.anchors), tone="info"),
        Tile("Related pairs", related_pairs, tone="acc"),
        Tile("Near-duplicates", len(result.near_dups), tone="warn" if result.near_dups else "muted"),
        Tile("Indexed documents", result.doc_count, tone="muted"),
    ])

    if result.doc_count == 0:
        r.section("No index", tone="warn").callout(
            "No embedded documents found. Has tasks.doc_index been run yet?",
            label="empty", tone="warn",
        )
        return r

    if result.truncated:
        r.section("Note", tone="warn").callout(
            f"{result.recent_count} document(s) changed in the window; only the "
            f"{len(result.anchors)} most recently changed were used as anchors this "
            "run. The rest will be picked up next time.",
            label="anchor cap", tone="warn",
        )
    if result.recent_without_vector:
        r.section("Note", tone="warn").callout(
            f"{result.recent_without_vector} recently changed document(s) have no "
            "embedding under the current model yet -- re-run tasks.doc_index.",
            label="not yet embedded", tone="warn",
        )

    if not result.anchors:
        r.section("No recent changes", tone="muted").callout(
            "No indexed document changed in the window.", label="quiet", tone="muted",
        )
        return r

    for anchor in result.anchors:
        matches = result.related.get(anchor, [])
        section = r.section(anchor, tone="acc")
        if not matches:
            section.callout("No related documents above the threshold.", label="none", tone="muted")
            continue
        headers = ["Related document", "Score"]
        has_why = any(m.why for m in matches)
        if has_why:
            headers.append("Why")
        rows = []
        for m in matches:
            row = [m.path, round(m.score, 4)]
            if has_why:
                row.append(m.why or (m.why_error or ""))
            rows.append(row)
        section.table(headers, rows)

    if result.near_dups:
        section = r.section("Near-duplicates", tone="warn")
        headers = ["Document A", "Document B", "Score"]
        has_why = any(d.why for d in result.near_dups)
        if has_why:
            headers.append("Why")
        rows = []
        for d in result.near_dups:
            row = [d.path_a, d.path_b, round(d.score, 4)]
            if has_why:
                row.append(d.why or (d.why_error or ""))
            rows.append(row)
        section.table(headers, rows)

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="anchor window in days")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="top-k related documents per anchor")
    parser.add_argument("--min-score", type=float, default=RELATED_THRESHOLD, help="related-document cosine threshold")
    parser.add_argument("--dup-score", type=float, default=DUP_THRESHOLD, help="near-duplicate cosine threshold")
    parser.add_argument("--anchor-cap", type=int, default=ANCHOR_CAP, help="max recently changed documents used as anchors")
    parser.add_argument("--explain", action="store_true", help="ask the model for a one-line reason per pair (memo'd)")
    parser.add_argument("--fresh", action="store_true", help="ignore memoized explanations and ask again")
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()

    with Store() as db:
        with db.record_run("related_notes") as run:
            mtimes = _doc_mtimes(db.conn)
            result = find_related(
                db.conn, lm.cfg.embed_model, now=_now(), days=args.days, k=args.k,
                related_threshold=args.min_score, dup_threshold=args.dup_score,
                anchor_cap=args.anchor_cap,
            )

            memo = None
            if args.explain and result.anchors:
                memo = Memo(
                    db.conn, "related_notes",
                    memo_version(MEMO_VERSION, prompt_hash("related_notes/why_system"), model_id(lm)),
                    fresh=args.fresh,
                )
                explain_all(db.conn, lm, memo, mtimes, result)

            report = build_report(result)
            if memo:
                report.footer_note(
                    f"Explanations use up to {EXCERPT_CHARS} characters from each "
                    "document's first chunk."
                )
                add_footer(report, memo, db.last_successful_run("related_notes"))
            report_path = report.write(out_dir=lm.cfg.output_dir)

            run.summary = {
                "anchors": len(result.anchors),
                "recent_count": result.recent_count,
                "related_pairs": sum(len(v) for v in result.related.values()),
                "near_duplicates": len(result.near_dups),
                "doc_count": result.doc_count,
                "truncated": result.truncated,
                "explained": args.explain,
                **(memo.summary() if memo else {}),
            }
            run.report_path = report_path

    print(f"{len(result.anchors)} anchor(s) -> {report_path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
