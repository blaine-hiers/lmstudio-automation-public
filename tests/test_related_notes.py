import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lmsa.memo import Memo
from lmsa.store import Store, pack_vector
from tasks.related_notes import (
    NearDup,
    Related,
    build_doc_vectors,
    build_report,
    compute_related,
    explain_all,
    explain_pair,
    find_related,
    pair_key,
    recent_anchors,
)

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
MODEL = "fake-embed"


def _seed_doc(conn, path, texts, vectors, mtime, model=MODEL):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS doc_chunks (
            path TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
            mtime REAL NOT NULL, size INTEGER NOT NULL,
            PRIMARY KEY (path, chunk_index)
        );
        """
    )
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        conn.execute(
            "INSERT INTO doc_chunks(path, chunk_index, text, mtime, size) VALUES (?, ?, ?, ?, ?)",
            (path, i, text, mtime, len(text)),
        )
        if vector is not None:
            conn.execute(
                "INSERT INTO embeddings(ns, item_id, model, dim, vector) VALUES ('docs', ?, ?, ?, ?)",
                (f"{path}#{i}", model, len(vector), pack_vector(vector)),
            )
    conn.commit()


# -- build_doc_vectors ---------------------------------------------------------


def test_build_doc_vectors_mean_pools_and_normalizes(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_doc(store.conn, "a.md", ["x", "y"], [[1.0, 0.0], [3.0, 0.0]], mtime=0)
        vectors = build_doc_vectors(store.conn, MODEL)

    assert set(vectors) == {"a.md"}
    assert vectors["a.md"] == pytest.approx(np.array([1.0, 0.0]))


def test_build_doc_vectors_skips_docs_with_no_embedding_under_model(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_doc(store.conn, "a.md", ["x"], [[1.0, 0.0]], mtime=0, model="other-model")
        vectors = build_doc_vectors(store.conn, MODEL)

    assert vectors == {}


# -- recent_anchors --------------------------------------------------------------


def test_recent_anchors_only_within_window_newest_first():
    now = NOW
    mtimes = {
        "new.md": (now - timedelta(days=1)).timestamp(),
        "old.md": (now - timedelta(days=30)).timestamp(),
        "newer.md": (now - timedelta(hours=1)).timestamp(),
    }
    doc_vectors = {p: np.array([1.0, 0.0]) for p in mtimes}

    anchors, recent_count, without_vector = recent_anchors(mtimes, doc_vectors, now=now, days=7)

    assert anchors == ["newer.md", "new.md"]
    assert recent_count == 2
    assert without_vector == 0


def test_recent_anchors_caps_and_counts_recent_without_being_capped():
    now = NOW
    mtimes = {
        f"doc{i}.md": (now - timedelta(hours=i)).timestamp() for i in range(5)
    }
    doc_vectors = {p: np.array([1.0, 0.0]) for p in mtimes}

    anchors, recent_count, without_vector = recent_anchors(mtimes, doc_vectors, now=now, days=7, cap=2)

    assert len(anchors) == 2
    assert recent_count == 5  # uncapped count, so a caller can tell "capped" from "quiet"
    assert anchors == ["doc0.md", "doc1.md"]  # newest first


def test_recent_anchors_excludes_but_counts_docs_with_no_vector_yet():
    now = NOW
    mtimes = {
        "embedded.md": (now - timedelta(hours=1)).timestamp(),
        "pending.md": (now - timedelta(hours=2)).timestamp(),
    }
    doc_vectors = {"embedded.md": np.array([1.0, 0.0])}

    anchors, recent_count, without_vector = recent_anchors(mtimes, doc_vectors, now=now, days=7)

    assert anchors == ["embedded.md"]
    assert recent_count == 2
    assert without_vector == 1


def test_recent_anchors_never_includes_a_document_outside_the_window():
    now = NOW
    mtimes = {"stale.md": (now - timedelta(days=100)).timestamp()}
    doc_vectors = {"stale.md": np.array([1.0, 0.0])}

    anchors, recent_count, without_vector = recent_anchors(mtimes, doc_vectors, now=now, days=7)

    assert anchors == []
    assert recent_count == 0


# -- compute_related -------------------------------------------------------------


def _unit(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _at_angle(degrees: float):
    """A unit vector at `degrees` from [1, 0] -- cos(degrees) is then
    exactly its cosine similarity to the anchor, so thresholds are easy to
    reason about."""
    rad = math.radians(degrees)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)


def test_compute_related_buckets_by_threshold():
    doc_vectors = {
        "anchor.md": _unit([1.0, 0.0]),
        "related.md": _at_angle(25),         # cos(25 deg) ~= 0.906 -> related
        "duplicate.md": _unit([1.0, 0.0]),   # identical -> near-duplicate
        "unrelated.md": _unit([0.0, 1.0]),   # orthogonal -> below related threshold
    }
    related, near_dups = compute_related(doc_vectors, ["anchor.md"], k=5)

    related_paths = {r.path for r in related["anchor.md"]}
    assert related_paths == {"related.md"}
    assert "duplicate.md" not in related_paths
    assert "unrelated.md" not in related_paths
    assert len(near_dups) == 1
    assert {near_dups[0].path_a, near_dups[0].path_b} == {"anchor.md", "duplicate.md"}


def test_compute_related_dedupes_symmetric_near_duplicate_pairs():
    # Both docs are anchors and mutually near-identical: the pair must
    # appear once, not once per direction.
    doc_vectors = {"a.md": _unit([1.0, 0.0]), "b.md": _unit([1.0, 0.001])}
    _, near_dups = compute_related(doc_vectors, ["a.md", "b.md"], k=5)

    assert len(near_dups) == 1
    assert {near_dups[0].path_a, near_dups[0].path_b} == {"a.md", "b.md"}


def test_compute_related_limits_to_top_k():
    doc_vectors = {"anchor.md": _unit([1.0, 0.0])}
    # angles 20, 23, ..., 41 degrees all score >= 0.75 and < 0.95, and
    # decrease monotonically -- more than k of them, to prove truncation
    # keeps the *best* k, not just any k.
    for i in range(8):
        doc_vectors[f"cand{i}.md"] = _at_angle(20 + i * 3)
    related, _ = compute_related(doc_vectors, ["anchor.md"], k=3)

    matches = related["anchor.md"]
    assert len(matches) == 3
    assert [m.path for m in matches] == ["cand0.md", "cand1.md", "cand2.md"]
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)


def test_compute_related_only_recent_docs_seed_a_comparison():
    # An anchor list of one path: a non-anchor document must never become a
    # dict key in `related`, even if it would itself have great matches.
    doc_vectors = {
        "anchor.md": _unit([1.0, 0.0]),
        "not_anchor.md": _unit([0.0, 1.0]),
        "match.md": _unit([0.0, 0.99]),
    }
    related, _ = compute_related(doc_vectors, ["anchor.md"], k=5)

    assert "not_anchor.md" not in related
    assert "anchor.md" in related or related == {}  # anchor.md may have no matches above threshold


def test_find_related_end_to_end(tmp_path):
    now = NOW
    with Store(tmp_path / "db.sqlite") as store:
        _seed_doc(store.conn, "new.md", ["alpha"], [[1.0, 0.0]], mtime=(now - timedelta(hours=1)).timestamp())
        _seed_doc(
            store.conn, "sibling.md", ["alpha too"], [_at_angle(25).tolist()],
            mtime=(now - timedelta(days=60)).timestamp(),
        )
        _seed_doc(store.conn, "unrelated.md", ["beta"], [[0.0, 1.0]], mtime=(now - timedelta(days=60)).timestamp())

        result = find_related(store.conn, MODEL, now=now, days=7)

    assert result.anchors == ["new.md"]
    assert result.doc_count == 3
    assert any(m.path == "sibling.md" for m in result.related.get("new.md", []))
    assert not result.truncated


# -- pair_key / explain_pair -----------------------------------------------------


def test_pair_key_is_symmetric():
    assert pair_key("a.md", 1.0, "b.md", 2.0) == pair_key("b.md", 2.0, "a.md", 1.0)


class FakeLM:
    def __init__(self, text="they discuss the same topic", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def chat(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.text


def test_explain_pair_returns_none_when_excerpt_missing():
    lm = FakeLM()
    why, error = explain_pair(lm, None, "a.md", 1.0, "b.md", 2.0, None, "some text")
    assert why is None
    assert "unavailable" in error
    assert lm.calls == []


def test_explain_pair_never_raises_on_model_failure():
    lm = FakeLM(error=RuntimeError("model down"))
    why, error = explain_pair(lm, None, "a.md", 1.0, "b.md", 2.0, "text a", "text b")
    assert why is None
    assert "model down" in error


def test_explain_pair_memoizes_symmetric_pair(tmp_path):
    lm = FakeLM()
    with Store(tmp_path / "db.sqlite") as store:
        memo = Memo(store.conn, "related_notes", "v1")
        why1, err1 = explain_pair(lm, memo, "a.md", 1.0, "b.md", 2.0, "text a", "text b")
        # Same unordered pair, arguments swapped (as it would be when the
        # pair shows up on the other anchor's side): must be a memo hit.
        why2, err2 = explain_pair(lm, memo, "b.md", 2.0, "a.md", 1.0, "text b", "text a")

    assert why1 == why2 == "they discuss the same topic"
    assert err1 is None and err2 is None
    assert len(lm.calls) == 1


def test_explain_all_fills_in_related_and_near_dup_without_duplicate_calls(tmp_path):
    lm = FakeLM()
    with Store(tmp_path / "db.sqlite") as store:
        _seed_doc(store.conn, "a.md", ["excerpt a"], [[1.0, 0.0]], mtime=1.0)
        _seed_doc(store.conn, "b.md", ["excerpt b"], [[1.0, 0.0]], mtime=2.0)
        memo = Memo(store.conn, "related_notes", "v1")

        from tasks.related_notes import RelatedNotesResult

        result = RelatedNotesResult(
            related={"a.md": [Related(path="b.md", score=0.8)]},
            near_dups=[NearDup(path_a="a.md", path_b="b.md", score=0.96)],
            anchors=["a.md"],
        )
        explain_all(store.conn, lm, memo, {"a.md": 1.0, "b.md": 2.0}, result)

    assert result.related["a.md"][0].why == "they discuss the same topic"
    assert result.near_dups[0].why == "they discuss the same topic"
    # Same unordered pair used for both the "related" entry and the
    # near-duplicate entry -- one model call, not two.
    assert len(lm.calls) == 1


# -- build_report ------------------------------------------------------------------


def test_build_report_no_index_shows_warning():
    from tasks.related_notes import RelatedNotesResult

    html = build_report(RelatedNotesResult()).render()
    assert "doc_index" in html


def test_build_report_no_recent_changes():
    from tasks.related_notes import RelatedNotesResult

    result = RelatedNotesResult(doc_count=5, anchors=[], recent_count=0)
    html = build_report(result).render()
    assert "No indexed document changed" in html


def test_build_report_lists_anchor_sections_and_near_dups():
    from tasks.related_notes import RelatedNotesResult

    result = RelatedNotesResult(
        related={"new.md": [Related(path="old.md", score=0.8123)]},
        near_dups=[NearDup(path_a="a.md", path_b="b.md", score=0.97)],
        anchors=["new.md"],
        doc_count=3,
        recent_count=1,
    )
    html = build_report(result).render()
    assert "new.md" in html
    assert "old.md" in html
    assert "0.81" in html  # the shared table formatter renders floats to 2dp
    assert "Near-duplicates" in html
    assert "a.md" in html and "b.md" in html


def test_build_report_discloses_anchor_cap_truncation():
    from tasks.related_notes import RelatedNotesResult

    result = RelatedNotesResult(
        related={}, near_dups=[], anchors=["a.md"], doc_count=10, recent_count=25, truncated=True,
    )
    html = build_report(result).render()
    assert "anchor cap" in html or "most recently changed" in html
