"""tasks.photo_index: incremental scanning (mtime/size fast-skip + content-hash
memo), EXIF date/GPS extraction, per-run cap, undecodable-file isolation, and
pruning of deleted photos. All fixtures are tiny in-memory Pillow images; no
real vision model or LM Studio server is touched.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from lmsa.memo import Memo
from lmsa.store import Store
from tasks import photo_index as pi


def _jpeg_bytes(size=(20, 20), color=(30, 60, 90), *, gps=None, taken_at=None) -> bytes:
    im = Image.new("RGB", size, color)
    exif = im.getexif()
    if taken_at:
        exif.get_ifd(pi._IFD_EXIF)[pi._TAG_DATETIME_ORIGINAL] = taken_at
    if gps:
        lat, lat_ref, lon, lon_ref = gps
        exif.get_ifd(pi._IFD_GPS).update({1: lat_ref, 2: lat, 3: lon_ref, 4: lon})
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


CAPTION_RESULT = {"caption": "a blue square", "tags": ["blue", "square"], "detected_text": ""}


class FakeLM:
    def __init__(self, result=None):
        self.result = result or CAPTION_RESULT
        self.cfg = SimpleNamespace(embed_model="fake-embed", vision_model="fake-vision")
        self.see_calls: list[bytes] = []
        self.embed_calls: list[list[str]] = []

    def see(self, images, prompt, schema=None, *, system=None, model=None, name="result"):
        self.see_calls.append(images[0])
        return self.result

    def embed(self, texts, *, model=None):
        self.embed_calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


# ---------------------------------------------------------------------------
# iter_images
# ---------------------------------------------------------------------------


def test_iter_images_finds_supported_extensions_recursively(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg_bytes())
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    (tmp_path / ".hidden.jpg").write_bytes(_jpeg_bytes())
    sub = tmp_path / "2024"
    sub.mkdir()
    (sub / "b.png").write_bytes(_jpeg_bytes())

    found = sorted(p.name for p in pi.iter_images(tmp_path))
    assert found == ["a.jpg", "b.png"]


def test_iter_images_missing_root_yields_nothing(tmp_path):
    assert list(pi.iter_images(tmp_path / "nope")) == []


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------


def test_read_exif_extracts_date_and_gps():
    data = _jpeg_bytes(
        taken_at="2024:05:06 07:08:09",
        gps=((37.0, 30.0, 0.0), "N", (122.0, 15.0, 0.0), "W"),
    )
    taken_at, lat, lon = pi.read_exif(data)
    assert taken_at == "2024-05-06T07:08:09"
    assert lat == pytest.approx(37.5)
    assert lon == pytest.approx(-122.25)


def test_read_exif_missing_exif_returns_none():
    assert pi.read_exif(_jpeg_bytes()) == (None, None, None)


def test_read_exif_never_raises_on_garbage():
    assert pi.read_exif(b"not an image at all") == (None, None, None)


# ---------------------------------------------------------------------------
# index_photos
# ---------------------------------------------------------------------------


def test_index_photos_captions_and_embeds_new_photo(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg_bytes(gps=((37, 0, 0.0), "N", (122, 0, 0.0), "W")))
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        stats = pi.index_photos(store.conn, lm, tmp_path)
        row = store.conn.execute(
            "SELECT caption, tags_json, gps_lat, gps_lon, country FROM photos"
        ).fetchone()
        embed_rows = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns='photos'").fetchone()[0]

    assert stats.scanned == 1
    assert stats.indexed == 1
    assert not stats.errors
    assert row[0] == "a blue square"
    assert json.loads(row[1]) == ["blue", "square"]
    assert row[2] == pytest.approx(37.0)
    assert row[3] == pytest.approx(-122.0)
    assert row[4] == "US"
    assert embed_rows == 1
    assert len(lm.see_calls) == 1


def test_index_photos_never_touches_source_file(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(_jpeg_bytes())
    before = photo.read_bytes()
    before_mtime = photo.stat().st_mtime

    with Store(tmp_path / "db.sqlite") as store:
        pi.index_photos(store.conn, FakeLM(), tmp_path)

    assert photo.read_bytes() == before
    assert photo.stat().st_mtime == before_mtime


def test_index_photos_second_run_skips_unchanged_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg_bytes())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        first = pi.index_photos(store.conn, lm, tmp_path)
        second = pi.index_photos(store.conn, lm, tmp_path)

    assert first.indexed == 1
    assert second.indexed == 0
    assert second.unchanged == 1
    assert len(lm.see_calls) == 1  # no second vision call


def test_index_photos_memo_avoids_recaption_when_content_unchanged(tmp_path):
    """Touching a file's mtime (content identical) forces a re-scan, but the
    memo (keyed by content hash) must still avoid a second vision call."""
    photo = tmp_path / "a.jpg"
    photo.write_bytes(_jpeg_bytes())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        memo = Memo(store.conn, "photo_index", "v1")
        pi.index_photos(store.conn, lm, tmp_path, memo=memo)

        import os
        st = photo.stat()
        os.utime(photo, (st.st_atime + 10, st.st_mtime + 10))

        memo2 = Memo(store.conn, "photo_index", "v1")
        stats = pi.index_photos(store.conn, lm, tmp_path, memo=memo2)

    assert stats.indexed == 1  # the row was rewritten (mtime differs)...
    assert len(lm.see_calls) == 1  # ...but the caption came from the memo, not a new call


def test_index_photos_honours_max_new_per_run(tmp_path):
    (tmp_path / "a.jpg").write_bytes(_jpeg_bytes(color=(1, 1, 1)))
    (tmp_path / "b.jpg").write_bytes(_jpeg_bytes(color=(2, 2, 2)))
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        stats = pi.index_photos(store.conn, lm, tmp_path, max_new_per_run=1)
        count = store.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    assert stats.scanned == 2
    assert stats.indexed == 1
    assert stats.deferred == 1
    assert count == 1
    assert len(lm.see_calls) == 1


def test_index_photos_skips_undecodable_file_without_crashing(tmp_path):
    (tmp_path / "broken.heic").write_bytes(b"not actually an image")
    (tmp_path / "ok.jpg").write_bytes(_jpeg_bytes())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        stats = pi.index_photos(store.conn, lm, tmp_path)

    assert stats.scanned == 2
    assert stats.indexed == 1
    assert len(stats.errors) == 1
    assert stats.errors[0].path.endswith("broken.heic")


def test_index_photos_prunes_deleted_files(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(_jpeg_bytes())
    lm = FakeLM()

    with Store(tmp_path / "db.sqlite") as store:
        pi.index_photos(store.conn, lm, tmp_path)
        photo.unlink()
        stats = pi.index_photos(store.conn, lm, tmp_path)
        remaining = store.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        embed_rows = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns='photos'").fetchone()[0]

    assert stats.pruned == 1
    assert remaining == 0
    assert embed_rows == 0


def test_build_report_shows_deferred_and_errors():
    stats = pi.IndexStats(scanned=3, indexed=1, deferred=1, errors=[pi.Skip("x.heic", "bad")])
    html = pi.build_report(stats, __import__("pathlib").Path("~/Pictures")).render()
    assert "capped" in html
    assert "x.heic" in html
