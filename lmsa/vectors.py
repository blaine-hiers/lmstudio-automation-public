"""Embed + cosine-search over the store's `embeddings` table.

Small, generic API meant to be reused by any task that needs "find the
closest items to this query" (doc search now, inbox Q&A later): callers
own their own metadata table (path, subject, whatever) keyed by the same
`item_id`; this module only ever touches `embeddings`.

    from lmsa import LMStudio, Store
    from lmsa.vectors import embed_and_upsert, search

    lm = LMStudio()
    with Store() as db:
        embed_and_upsert(db.conn, lm, ns="docs", model=lm.cfg.embed_model,
                          items=[("chunk-1", "some text"), ("chunk-2", "more text")])
        hits = search(db.conn, lm, ns="docs", model=lm.cfg.embed_model,
                       query="some text", k=5)
        # hits == [(item_id, cosine_score), ...], best first
"""

from __future__ import annotations

import sqlite3
from typing import Protocol, Sequence

import numpy as np

from .store import pack_vector, unpack_vector


class Embedder(Protocol):
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...


def upsert(
    conn: sqlite3.Connection,
    *,
    ns: str,
    model: str,
    items: Sequence[tuple[str, Sequence[float]]],
) -> None:
    """Write `(item_id, vector)` pairs into `embeddings`, replacing any
    existing row for the same `(ns, item_id, model)`."""
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(ns, item_id, model, dim, vector) VALUES (?, ?, ?, ?, ?)",
        [(ns, item_id, model, len(vector), pack_vector(vector)) for item_id, vector in items],
    )
    conn.commit()


def embed_and_upsert(
    conn: sqlite3.Connection,
    lm: Embedder,
    *,
    ns: str,
    model: str,
    items: Sequence[tuple[str, str]],
) -> None:
    """Embed `(item_id, text)` pairs with `lm.embed` and upsert the result.

    A no-op on an empty `items` (LM Studio's embeddings endpoint doesn't need
    to be called at all).
    """
    if not items:
        return
    ids = [item_id for item_id, _ in items]
    vectors = lm.embed([text for _, text in items], model=model)
    upsert(conn, ns=ns, model=model, items=list(zip(ids, vectors)))


def top_k(
    conn: sqlite3.Connection,
    *,
    ns: str,
    model: str,
    query_vector: Sequence[float],
    k: int = 10,
) -> list[tuple[str, float]]:
    """Cosine top-k over every vector stored for `(ns, model)`.

    Returns `[(item_id, score), ...]`, best (highest cosine similarity)
    first. Loads the whole namespace into memory -- fine for the thousands
    of chunks a personal-document index produces, not meant for millions.
    """
    rows = conn.execute(
        "SELECT item_id, vector FROM embeddings WHERE ns=? AND model=?",
        (ns, model),
    ).fetchall()
    if not rows:
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    item_ids = [item_id for item_id, _ in rows]
    matrix = np.asarray([unpack_vector(blob) for _, blob in rows], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    # Zero-norm rows (shouldn't happen, but a stray all-zero vector must not
    # divide by zero and poison the rest of the scores) score 0 rather than NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (matrix @ query) / (norms * query_norm)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.argsort(-scores)[:k]
    return [(item_ids[i], float(scores[i])) for i in order]


def search(
    conn: sqlite3.Connection,
    lm: Embedder,
    *,
    ns: str,
    model: str,
    query: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Embed `query` with `lm.embed` and run `top_k` against it."""
    (query_vector,) = lm.embed([query], model=model)
    return top_k(conn, ns=ns, model=model, query_vector=query_vector, k=k)
