from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa.store import Store
from tasks import doc_index
from tasks.doc_index import (
    Config,
    chunk_text,
    index_documents,
    iter_files,
    load_config,
)


class FakeLM:
    def __init__(self):
        self.cfg = SimpleNamespace(embed_model="fake-embed")
        self.calls = []

    def embed(self, texts, *, model=None):
        self.calls.append(list(texts))
        return [[float(len(t)), float(t.count("a"))] for t in texts]


def test_load_config_defaults_when_file_missing(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.roots == [Path("~/Documents").expanduser()]
    assert config.exclude_dirs == {"github"}


def test_load_config_reads_toml(tmp_path):
    path = tmp_path / "doc_search.toml"
    path.write_text('roots = ["~/Notes", "/tmp/other"]\nexclude_dirs = ["Archive"]\n', encoding="utf-8")
    config = load_config(path)
    assert config.roots == [Path("~/Notes").expanduser(), Path("/tmp/other")]
    assert config.exclude_dirs == {"archive"}


def test_iter_files_skips_hidden_and_excluded_dirs_and_wrong_extensions(tmp_path):
    (tmp_path / "notes.md").write_text("keep", encoding="utf-8")
    (tmp_path / "image.png").write_text("skip", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("skip", encoding="utf-8")
    (tmp_path / "GitHub").mkdir()
    (tmp_path / "GitHub" / "readme.md").write_text("skip", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text("keep too", encoding="utf-8")

    config = Config(roots=[tmp_path], exclude_dirs={"github"})
    found = sorted(str(p.relative_to(tmp_path)) for p in iter_files(config))

    assert found == ["notes.md", "sub\\deep.txt"] or found == ["notes.md", "sub/deep.txt"]


def test_iter_files_skips_missing_root(tmp_path):
    config = Config(roots=[tmp_path / "does-not-exist"], exclude_dirs=set())
    assert list(iter_files(config)) == []


def test_chunk_text_splits_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, size=800, overlap=120)
    assert len(chunks) == 2
    # the tail of chunk 1 reappears at the head of chunk 2 (the overlap).
    assert chunks[0].split()[-1] == chunks[1].split()[119]


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("   ") == []


def test_index_documents_writes_chunks_and_embeddings(tmp_path):
    (tmp_path / "a.md").write_text("alpha alpha alpha content", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta content here", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        stats = index_documents(store.conn, lm, config)
        chunk_rows = store.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        embed_rows = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns='docs'").fetchone()[0]

    assert stats.scanned == 2
    assert stats.indexed == 2
    assert stats.unchanged == 0
    assert not stats.errors
    assert chunk_rows == 2  # one chunk per short file
    assert embed_rows == 2


def test_index_documents_never_modifies_source_files(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("do not touch me", encoding="utf-8")
    before = doc.read_text(encoding="utf-8")
    before_mtime = doc.stat().st_mtime

    config = Config(roots=[tmp_path], exclude_dirs=set())
    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, FakeLM(), config)

    assert doc.read_text(encoding="utf-8") == before
    assert doc.stat().st_mtime == before_mtime


def test_index_documents_second_run_skips_unchanged_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        first = index_documents(store.conn, lm, config)
        second = index_documents(store.conn, lm, config)

    assert first.indexed == 1
    assert second.indexed == 0
    assert second.unchanged == 1
    assert len(lm.calls) == 1  # second run never re-embedded


def test_index_documents_reindexes_when_file_changes(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("alpha content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, lm, config)
        doc.write_text("alpha content but now much longer than before", encoding="utf-8")
        second = index_documents(store.conn, lm, config)
        rows = store.conn.execute("SELECT text FROM doc_chunks").fetchall()

    assert second.indexed == 1
    assert "much longer" in rows[0][0]


def test_index_documents_isolates_per_file_errors(tmp_path, monkeypatch):
    (tmp_path / "good.md").write_text("fine content", encoding="utf-8")
    (tmp_path / "bad.md").write_text("broken content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    real_extract = doc_index.extract_text

    def flaky_extract(path):
        if path.name == "bad.md":
            raise ValueError("simulated corrupt file")
        return real_extract(path)

    monkeypatch.setattr(doc_index, "extract_text", flaky_extract)

    with Store(tmp_path / "db.sqlite") as store:
        stats = index_documents(store.conn, lm, config)

    assert stats.scanned == 2
    assert stats.indexed == 1
    assert len(stats.errors) == 1
    assert stats.errors[0][0].endswith("bad.md")
    assert "simulated corrupt file" in stats.errors[0][1]


def test_index_documents_limit_caps_files_and_flags_truncated(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.md").write_text(f"content {i}", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        stats = index_documents(store.conn, lm, config, limit=2)

    assert stats.scanned == 2
    assert stats.truncated is True


def test_build_report_notes_truncation_and_errors():
    stats = doc_index.IndexStats(scanned=2, indexed=1, unchanged=0, chunks_written=1,
                                  truncated=True, errors=[("x.pdf", "boom")])
    html = doc_index.build_report(stats).render()
    assert "truncated" in html
    assert "x.pdf" in html
    assert "boom" in html


def test_index_documents_prunes_deleted_file(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("alpha content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, lm, config)
        doc.unlink()
        stats = index_documents(store.conn, lm, config)
        chunk_rows = store.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        embed_rows = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns='docs'").fetchone()[0]

    assert stats.scanned == 0
    assert stats.pruned == 1
    assert chunk_rows == 0
    assert embed_rows == 0


def test_index_documents_prunes_old_path_and_indexes_new_path_on_rename(tmp_path):
    old_path = tmp_path / "old.md"
    old_path.write_text("alpha content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, lm, config)
        new_path = tmp_path / "new.md"
        old_path.rename(new_path)
        stats = index_documents(store.conn, lm, config)

        paths = {row[0] for row in store.conn.execute("SELECT DISTINCT path FROM doc_chunks").fetchall()}
        item_ids = {row[0] for row in store.conn.execute("SELECT item_id FROM embeddings WHERE ns='docs'").fetchall()}

    assert stats.pruned == 1
    assert stats.indexed == 1
    assert paths == {str(new_path)}
    assert item_ids == {f"{new_path}#0"}


def test_index_documents_shrink_removes_stale_high_index_chunks(tmp_path):
    doc = tmp_path / "a.md"
    long_text = " ".join(f"w{i}" for i in range(1000))  # 2 chunks at size=800/overlap=120
    doc.write_text(long_text, encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, lm, config)
        before = store.conn.execute("SELECT chunk_index FROM doc_chunks ORDER BY chunk_index").fetchall()
        assert [r[0] for r in before] == [0, 1]

        doc.write_text("just a few words now", encoding="utf-8")
        index_documents(store.conn, lm, config)

        after = store.conn.execute("SELECT chunk_index FROM doc_chunks ORDER BY chunk_index").fetchall()
        embed_ids = {row[0] for row in store.conn.execute("SELECT item_id FROM embeddings WHERE ns='docs'").fetchall()}

    assert [r[0] for r in after] == [0]
    assert embed_ids == {f"{doc}#0"}


def test_index_documents_reembeds_on_model_swap_without_reparsing(tmp_path, monkeypatch):
    doc = tmp_path / "a.md"
    doc.write_text("alpha content", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    parse_calls = []
    real_extract = doc_index.extract_text

    def counting_extract(path):
        parse_calls.append(path)
        return real_extract(path)

    monkeypatch.setattr(doc_index, "extract_text", counting_extract)

    with Store(tmp_path / "db.sqlite") as store:
        first = index_documents(store.conn, lm, config, model="model-a")
        second = index_documents(store.conn, lm, config, model="model-b")

        rows = store.conn.execute(
            "SELECT model FROM embeddings WHERE ns='docs' ORDER BY model"
        ).fetchall()
        chunk_rows = store.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]

    assert first.indexed == 1
    assert second.indexed == 0
    assert second.reembedded == 1
    assert len(parse_calls) == 1  # the second run never re-parsed the file
    assert [r[0] for r in rows] == ["model-a", "model-b"]
    assert chunk_rows == 1  # doc_chunks untouched, only a new embedding was added


def test_index_documents_limit_never_prunes_unscanned_existing_files(tmp_path):
    (tmp_path / "f0.md").write_text("content zero", encoding="utf-8")
    (tmp_path / "f1.md").write_text("content one", encoding="utf-8")
    config = Config(roots=[tmp_path], exclude_dirs=set())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        index_documents(store.conn, lm, config)  # both indexed, unlimited
        stats = index_documents(store.conn, lm, config, limit=1)  # only scans one this run

        paths = {row[0] for row in store.conn.execute("SELECT DISTINCT path FROM doc_chunks").fetchall()}

    assert stats.scanned == 1
    assert stats.truncated is True
    assert stats.pruned == 0
    assert paths == {str(tmp_path / "f0.md"), str(tmp_path / "f1.md")}
