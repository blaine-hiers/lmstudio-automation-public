import pytest

from lmsa.store import Store, unpack_vector
from lmsa.vectors import embed_and_upsert, search, top_k, upsert


class FakeEmbedder:
    """Deterministic embedder: one dim per call, value = text length."""

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}
        self.calls: list[tuple[list[str], str | None]] = []

    def embed(self, texts, *, model=None):
        self.calls.append((list(texts), model))
        return [self.vectors.get(t, [float(len(t))]) for t in texts]


def test_upsert_writes_and_replaces_rows(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
        rows = store.conn.execute(
            "SELECT item_id, vector FROM embeddings WHERE ns='docs' AND model='m1' ORDER BY item_id"
        ).fetchall()
        assert [r[0] for r in rows] == ["a", "b"]
        assert unpack_vector(rows[0][1]) == pytest.approx([1.0, 0.0])

        # re-upsert "a" with a new vector: replaces, doesn't duplicate.
        upsert(store.conn, ns="docs", model="m1", items=[("a", [2.0, 2.0])])
        rows = store.conn.execute(
            "SELECT item_id, vector FROM embeddings WHERE ns='docs' AND model='m1'"
        ).fetchall()
        assert len(rows) == 2
        replaced = next(unpack_vector(v) for i, v in rows if i == "a")
        assert replaced == pytest.approx([2.0, 2.0])


def test_top_k_orders_by_cosine_similarity_best_first(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[
            ("close", [1.0, 0.0]),
            ("far", [0.0, 1.0]),
            ("opposite", [-1.0, 0.0]),
        ])
        results = top_k(store.conn, ns="docs", model="m1", query_vector=[1.0, 0.0], k=3)

    assert [item_id for item_id, _ in results] == ["close", "far", "opposite"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[-1][1] == pytest.approx(-1.0)


def test_top_k_respects_k_limit(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[(f"i{n}", [float(n), 1.0]) for n in range(10)])
        results = top_k(store.conn, ns="docs", model="m1", query_vector=[5.0, 1.0], k=2)
    assert len(results) == 2


def test_top_k_scopes_by_namespace_and_model(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[("a", [1.0, 0.0])])
        upsert(store.conn, ns="emails", model="m1", items=[("b", [1.0, 0.0])])
        upsert(store.conn, ns="docs", model="m2", items=[("c", [1.0, 0.0])])

        results = top_k(store.conn, ns="docs", model="m1", query_vector=[1.0, 0.0], k=10)

    assert [item_id for item_id, _ in results] == ["a"]


def test_top_k_empty_namespace_returns_empty_list(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        results = top_k(store.conn, ns="docs", model="m1", query_vector=[1.0, 0.0], k=5)
    assert results == []


def test_top_k_zero_query_vector_returns_empty_list_instead_of_raising(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[("a", [1.0, 0.0])])
        results = top_k(store.conn, ns="docs", model="m1", query_vector=[0.0, 0.0], k=5)
    assert results == []


def test_embed_and_upsert_calls_embedder_once_and_stores_vectors(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    lm = FakeEmbedder()
    with Store(db) as store:
        embed_and_upsert(store.conn, lm, ns="docs", model="m1", items=[("a", "hi"), ("b", "hello")])
        rows = store.conn.execute("SELECT item_id FROM embeddings WHERE ns='docs'").fetchall()
    assert {r[0] for r in rows} == {"a", "b"}
    assert len(lm.calls) == 1  # one batched embed() call, not one per item


def test_embed_and_upsert_is_a_noop_for_empty_items(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    lm = FakeEmbedder()
    with Store(db) as store:
        embed_and_upsert(store.conn, lm, ns="docs", model="m1", items=[])
    assert lm.calls == []


def test_search_embeds_query_then_delegates_to_top_k(tmp_path):
    db = tmp_path / "lmsa.sqlite"
    lm = FakeEmbedder(vectors={"query text": [1.0, 0.0]})
    with Store(db) as store:
        upsert(store.conn, ns="docs", model="m1", items=[("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
        results = search(store.conn, lm, ns="docs", model="m1", query="query text", k=1)
    assert results == [("a", pytest.approx(1.0))]
    assert lm.calls == [(["query text"], "m1")]
