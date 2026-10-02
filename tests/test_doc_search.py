import json

from lmsa.store import Store
from tasks.doc_search import (
    Match,
    answer_query,
    build_matches,
    build_photo_matches,
    build_photo_report,
    build_report,
    coarse_place,
    _snippet,
)


def _seed_chunk(conn, path, chunk_index, text):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS doc_chunks (
            path TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
            mtime REAL NOT NULL, size INTEGER NOT NULL,
            PRIMARY KEY (path, chunk_index)
        );
        """
    )
    conn.execute(
        "INSERT INTO doc_chunks(path, chunk_index, text, mtime, size) VALUES (?, ?, ?, 0, 0)",
        (path, chunk_index, text),
    )
    conn.commit()


def test_snippet_centers_on_first_query_hit():
    text = "x" * 200 + " the roof budget was cut " + "y" * 200
    snippet = _snippet(text, "roof budget")
    assert "roof budget" in snippet
    assert snippet.startswith("…")
    assert snippet.endswith("…")


def test_snippet_falls_back_to_head_when_no_hit():
    snippet = _snippet("nothing relevant here", "unrelated query")
    assert snippet == "nothing relevant here"


def test_snippet_handles_missing_chunk_text():
    assert "re-run doc_index" in _snippet(None, "anything")


def test_build_matches_looks_up_chunk_text_and_builds_snippets(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_chunk(store.conn, "C:/docs/a.md", 0, "roof budget details go here")
        matches = build_matches(store.conn, [("C:/docs/a.md#0", 0.9)], "roof budget")

    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, Match)
    assert m.path == "C:/docs/a.md"
    assert m.chunk_index == 0
    assert m.score == 0.9
    assert "roof budget" in m.snippet


def test_build_matches_survives_missing_chunk_row(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.executescript(
            "CREATE TABLE IF NOT EXISTS doc_chunks (path TEXT, chunk_index INTEGER, text TEXT, mtime REAL, size INTEGER, PRIMARY KEY(path, chunk_index));"
        )
        matches = build_matches(store.conn, [("gone.md#3", 0.5)], "query")

    assert matches[0].text is None
    assert "re-run doc_index" in matches[0].snippet


class FakeLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append((prompt, schema, system, name))
        if self.error:
            raise self.error
        return self.result


def test_answer_query_returns_grounded_answer_and_citations():
    matches = [Match(label="a.md#0", path="a.md", chunk_index=0, score=0.9,
                      text="roof budget is $500", snippet="roof budget is $500")]
    lm = FakeLM(result={"answer": "The roof budget is $500.", "citations": ["a.md#0"]})

    answer, citations, error = answer_query(lm, "what is the roof budget", matches)

    assert answer == "The roof budget is $500."
    assert citations == ["a.md#0"]
    assert error is None
    assert len(lm.calls) == 1


def test_answer_query_never_raises_on_model_failure():
    matches = [Match(label="a.md#0", path="a.md", chunk_index=0, score=0.9,
                      text="content", snippet="content")]
    lm = FakeLM(error=RuntimeError("model down"))

    answer, citations, error = answer_query(lm, "question", matches)

    assert answer is None
    assert citations == []
    assert "model down" in error


def test_answer_query_skips_model_call_when_no_chunk_text():
    matches = [Match(label="a.md#0", path="a.md", chunk_index=0, score=0.9, text=None, snippet="n/a")]
    lm = FakeLM(result={"answer": "should not be called", "citations": []})

    answer, citations, error = answer_query(lm, "question", matches)

    assert answer is None
    assert lm.calls == []
    assert error is not None


def test_build_report_no_matches_shows_warning():
    html = build_report("some query", []).render()
    assert "No indexed chunks matched" in html


def test_build_report_lists_matches_and_never_dumps_full_chunk_text():
    long_text = "SECRET " * 500
    matches = [Match(label="a.md#0", path="a.md", chunk_index=0, score=0.87,
                      text=long_text, snippet=_snippet(long_text, "secret"))]
    html = build_report("secret", matches).render()
    assert "a.md" in html
    assert long_text not in html  # only the (short) snippet is rendered


def test_build_report_includes_answer_and_citations():
    html = build_report("q", [], answer="Here is the answer.", citations=["a.md#0"]).render()
    assert "Here is the answer." in html
    assert "a.md#0" in html


def test_build_report_shows_answer_error_when_unanswered():
    html = build_report("q", [], answer_error="model down").render()
    assert "model down" in html


# ---------------------------------------------------------------------------
# --photos
# ---------------------------------------------------------------------------


def _seed_photo(conn, path, **overrides):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS photos (
            path TEXT PRIMARY KEY, content_hash TEXT, mtime REAL, size INTEGER,
            caption TEXT, tags_json TEXT, detected_text TEXT, taken_at TEXT,
            gps_lat REAL, gps_lon REAL, country TEXT, indexed_at TEXT
        );
        """
    )
    fields = dict(
        content_hash="h", mtime=0, size=0, caption="a cat on a couch",
        tags_json=json.dumps(["cat", "couch"]), detected_text="", taken_at="2024-05-06T07:08:09",
        gps_lat=None, gps_lon=None, country=None, indexed_at="now",
    )
    fields.update(overrides)
    conn.execute(
        "INSERT INTO photos (path, content_hash, mtime, size, caption, tags_json, detected_text, "
        "taken_at, gps_lat, gps_lon, country, indexed_at) VALUES "
        "(:path, :content_hash, :mtime, :size, :caption, :tags_json, :detected_text, :taken_at, "
        ":gps_lat, :gps_lon, :country, :indexed_at)",
        {"path": path, **fields},
    )
    conn.commit()


def test_coarse_place_prefers_country_name():
    assert coarse_place("US", 12.3456, -98.7654) == "United States of America"


def test_coarse_place_falls_back_to_rounded_coordinate():
    assert coarse_place(None, 12.3456, -98.7654) == "~12.3, -98.8"


def test_coarse_place_never_leaks_full_precision():
    place = coarse_place(None, 12.3456789, -98.7654321)
    assert "3456789" not in place and "7654321" not in place


def test_coarse_place_unknown_when_nothing_known():
    assert coarse_place(None, None, None) == "unknown location"


def test_build_photo_matches_reads_row_and_builds_coarse_place(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_photo(store.conn, "C:/pics/cat.jpg", country="FR")
        matches = build_photo_matches(store.conn, [("C:/pics/cat.jpg", 0.8)])

    assert len(matches) == 1
    m = matches[0]
    assert m.caption == "a cat on a couch"
    assert m.tags == ["cat", "couch"]
    assert m.place == "France"
    assert m.thumbnail is None  # the source file doesn't exist on disk in this test


def test_build_photo_matches_skips_a_hit_whose_row_is_gone(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_photo(store.conn, "C:/pics/cat.jpg")
        matches = build_photo_matches(store.conn, [("C:/pics/gone.jpg", 0.5)])

    assert matches == []


def test_build_photo_report_no_matches_shows_warning():
    html = build_photo_report("cats", []).render()
    assert "No indexed photos matched" in html


def test_build_photo_report_shows_caption_tags_and_discloses_thumbnail_cap():
    from tasks.doc_search import PhotoMatch, PHOTO_THUMB_MAX_DIM

    matches = [PhotoMatch(
        path="a.jpg", score=0.9, caption="a birthday cake", tags=["cake", "party"],
        detected_text="", taken_at="2024-05-06T07:08:09", place="France", thumbnail=None,
    )]
    html = build_photo_report("birthday cake", matches).render()
    assert "a birthday cake" in html
    assert "France" in html
    assert str(PHOTO_THUMB_MAX_DIM) in html  # cap disclosed in the footer
