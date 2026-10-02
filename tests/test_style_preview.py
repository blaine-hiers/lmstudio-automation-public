"""tasks.style_preview: the sample attachment thumbnail is a real image."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

import tasks.style_preview as style_preview


def test_sample_thumbnail_is_a_valid_png():
    prefix = "data:image/png;base64,"
    assert style_preview.SAMPLE_THUMBNAIL.startswith(prefix)
    raw = base64.b64decode(style_preview.SAMPLE_THUMBNAIL[len(prefix):])
    with Image.open(BytesIO(raw)) as im:
        im.verify()
    with Image.open(BytesIO(raw)) as im:
        assert im.format == "PNG" and im.size[0] > 4


def test_style_guide_renders_the_thumbnail(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))
    style_preview.main()
    page = tmp_path / "style-preview.html"
    assert style_preview.SAMPLE_THUMBNAIL in page.read_text(encoding="utf-8")
