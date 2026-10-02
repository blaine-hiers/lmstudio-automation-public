"""Prediction markets extension (#60): matches `lmsa.markets` snapshots to
briefing stories by embedding cosine similarity against each story's
centroid, then shows odds + 24h/7d movement on the story card and a
"biggest movers" report section.

Registered as extension **"markets"**, `default=False` -- a briefing opts in
with `extensions = ["markets"]` in `config/briefings/<name>.toml`.

Matching happens here, at briefing render time, not in the collector:
`tasks.markets` (model "none") only fetches and stores raw odds, so this
extension is the only place a market question ever gets embedded, using the
same embed model the briefing already used for its own story centroids
(`lmsa.memo.model_id(ctx.lm, "embed_model")`).

Every value shown is **market odds, not a forecast by this tool**
(`lmsa.markets.ODDS_DISCLAIMER`) -- every card line and the section both say
so explicitly.
"""

from __future__ import annotations

import numpy as np

from ...markets import ODDS_DISCLAIMER, latest_snapshot_day, movement, snapshots_on
from ...memo import model_id
from ...report import Link, StoryExtra
from ...store import unpack_vector
from ...vectors import upsert as vectors_upsert
from ..sources import register_extension

# Cosine similarity floor for a market<->story match. Disclosed here rather
# than tuned per briefing: raising it trades recall for precision uniformly.
MATCH_THRESHOLD = 0.62
MAX_MATCHES_PER_STORY = 3
TOP_MOVERS = 15
# A 24h move at or above this (a probability-fraction, e.g. 0.05 = 5 points)
# is shown "hot" on the card; below it, muted.
HOT_MOVE = 0.05

# `embeddings` namespace for cached market-question vectors (#60 review):
# a question is embedded once per embed model, ever, not once per enrich()
# call -- item_id = "<source>:<market_id>", stable across days since the
# question text for a given market rarely changes between snapshots.
MARKETS_NS = "markets"


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(norms > 0, m / norms, 0.0)
    return np.nan_to_num(u)


def _item_id(source: str, market_id: str) -> str:
    return f"{source}:{market_id}"


def _cached_vectors(conn, item_ids: list[str], model: str) -> dict[str, np.ndarray]:
    if not item_ids:
        return {}
    placeholders = ",".join("?" for _ in item_ids)
    rows = conn.execute(
        f"SELECT item_id, vector FROM embeddings WHERE ns=? AND model=? AND item_id IN ({placeholders})",
        (MARKETS_NS, model, *item_ids),
    ).fetchall()
    return {item_id: np.asarray(unpack_vector(blob), dtype=np.float32) for item_id, blob in rows}


def _embed_snapshots(ctx, day: str):
    """`(snapshots, unit-normalized vectors)`, lockstep and filtered down to
    only the snapshots a vector could be found or made for. A question's
    embedding is cached in the `embeddings` table (ns "markets") keyed by
    `<source>:<market_id>` and the embed model, so the same market is only
    ever embedded once per model, not once per enrich() call -- repeat runs
    (reruns, other briefings, other stories the same run) reuse it for free.
    Nothing to embed, or embedding every still-uncached question fails with
    nothing usable cached from before, returns `(snaps, None)` (recorded as
    a note; matching is simply skipped this run, never a fatal error)."""
    snaps = [s for s in snapshots_on(ctx.conn, day) if s.probability is not None and s.question]
    if not snaps:
        return snaps, None
    embed_model = model_id(ctx.lm, "embed_model")
    item_ids = [_item_id(s.source, s.market_id) for s in snaps]
    cached = _cached_vectors(ctx.conn, item_ids, embed_model)
    missing = [(iid, s.question) for iid, s in zip(item_ids, snaps) if iid not in cached]
    if missing:
        try:
            raw = ctx.lm.embed([q for _, q in missing], model=embed_model)
            new_vecs = np.asarray(raw, dtype=np.float32)
            if new_vecs.ndim != 2 or new_vecs.shape[0] != len(missing):
                raise RuntimeError(f"embedding model returned shape {new_vecs.shape} for {len(missing)} question(s)")
            vectors_upsert(ctx.conn, ns=MARKETS_NS, model=embed_model,
                            items=[(iid, vec.tolist()) for (iid, _), vec in zip(missing, new_vecs)])
            cached.update((iid, vec) for (iid, _), vec in zip(missing, new_vecs))
        except Exception as exc:
            ctx.note(f"markets: embedding {len(missing)} new question(s) failed "
                      f"({type(exc).__name__}: {str(exc)[:150]}); "
                      f"{len(cached)}/{len(snaps)} cached match(es) still used")
    if not cached:
        return snaps, None
    kept = [(s, cached[iid]) for s, iid in zip(snaps, item_ids) if iid in cached]
    if not kept:
        return [], None
    kept_snaps = [s for s, _ in kept]
    kept_vecs = _unit(np.asarray([v for _, v in kept], dtype=np.float32))
    return kept_snaps, kept_vecs


def enrich(stories, ctx) -> None:
    """Sets `story.extras["markets"]` to the (up to `MAX_MATCHES_PER_STORY`)
    markets whose question embedding is closest to the story's centroid,
    above `MATCH_THRESHOLD`. One story's failure (bad centroid, etc.) is
    skipped, never aborts the rest."""
    day = latest_snapshot_day(ctx.conn)
    if day is None or not stories:
        return
    snaps, vecs = _embed_snapshots(ctx, day)
    if vecs is None:
        return
    for story in stories:
        try:
            centroid = np.asarray(story.centroid, dtype=np.float32)
            if centroid.ndim != 1 or centroid.shape[0] != vecs.shape[1]:
                continue
            centroid = _unit(centroid[None, :])[0]
            scores = vecs @ centroid
            order = np.argsort(-scores)
            matches = []
            for i in order[:MAX_MATCHES_PER_STORY]:
                score = float(scores[i])
                if not np.isfinite(score) or score < MATCH_THRESHOLD:
                    break
                snap = snaps[int(i)]
                mv = movement(ctx.conn, snap.source, snap.market_id, day)
                matches.append({
                    "source": snap.source,
                    "question": snap.question,
                    "url": snap.url,
                    "probability": mv.probability,
                    "pct_24h": mv.pct_24h,
                    "pct_7d": mv.pct_7d,
                    "similarity": round(score, 3),
                })
            if matches:
                story.extras["markets"] = matches
        except Exception as exc:
            ctx.note(f"markets: matching story {getattr(story, 'story_id', '?')} failed "
                      f"({type(exc).__name__}: {str(exc)[:150]})")


def _fmt_move(v: float | None) -> str:
    return f"{v * 100:+.0f}pt" if v is not None else "—"


def render(story, ctx) -> list:
    """Card lines for the markets matched in `enrich` -- always labelled as
    odds, never as a forecast."""
    matches = story.extras.get("markets") if isinstance(story.extras, dict) else None
    if not matches:
        return []
    out = []
    for m in matches[:MAX_MATCHES_PER_STORY]:
        prob = m.get("probability")
        if prob is None:
            continue
        pct24 = m.get("pct_24h")
        tone = "hot" if pct24 is not None and abs(pct24) >= HOT_MOVE else "muted"
        source = str(m.get("source") or "market").replace("_", " ").title()
        text = (f"{prob * 100:.0f}% ({_fmt_move(pct24)}/24h, {_fmt_move(m.get('pct_7d'))}/7d) "
                f"— {ODDS_DISCLAIMER}")
        out.append(StoryExtra(label=f"{source} odds", text=text, href=m.get("url"), tone=tone))
    return out


def section(report, stories, ctx) -> None:
    """"Biggest movers": every matched market with a computable 24h move,
    across all of this run's stories, ranked by |24h move|. Nothing to show
    (no matches, or no snapshot day yet) adds no section at all."""
    rows = []
    for story in stories:
        matches = story.extras.get("markets") if isinstance(story.extras, dict) else None
        for m in matches or []:
            if m.get("pct_24h") is None or m.get("probability") is None:
                continue
            rows.append((abs(m["pct_24h"]), story, m))
    if not rows:
        return
    rows.sort(key=lambda r: r[0], reverse=True)
    sec = report.section("Biggest movers (prediction markets)", tone="warn")
    sec.text(ODDS_DISCLAIMER)
    sec.table(
        ["Market", "Story", "Odds", "24h", "7d"],
        [
            (
                Link(m["question"][:80], m.get("url")),
                Link(story.headline[:60], f"#{story.anchor}"),
                f"{m['probability'] * 100:.0f}%",
                _fmt_move(m.get("pct_24h")),
                _fmt_move(m.get("pct_7d")),
            )
            for _, story, m in rows[:TOP_MOVERS]
        ],
    )


register_extension("markets", enrich=enrich, render=render, section=section, default=False)
