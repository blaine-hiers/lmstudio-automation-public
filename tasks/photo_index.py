"""Immich-style smart search for a local photo library, fully offline.

    python -m tasks.photo_index [--root DIR] [--max-new-per-run 100] [--fresh]

Walks `LMSA_PHOTOS_DIR` (default `~/Pictures`), and for every new or changed
image: reads EXIF date/GPS with Pillow, downscales it (reusing
`tasks.attachments`'s vision-safe JPEG re-encode), and asks the local vision
model (`LMStudio.see()`) for a caption, search tags, and any legible text in
the shot. The caption + tags + detected text are embedded (`lmsa.vectors`,
namespace "photos") so `tasks.doc_search --photos` can find a photo by what's
in it. Per-item results are cached in `lmsa.memo` (keyed by content hash) so
a renamed or duplicated file never re-pays for a vision call whose answer is
already known.

Incremental: a file whose path already has a `photos` row with the same
mtime+size is skipped without even being read. New or touched files are
capped at `--max-new-per-run` per run (default `MAX_NEW_PER_RUN`) so an
overnight batch can't be swallowed by a large or freshly-imported library;
the rest are picked up on a later run. A previously-indexed path that no
longer exists on disk has its row and embedding removed.

Read-only over the photo library: nothing here writes, moves, renames, or
creates a sidecar next to a source photo -- everything this task produces
lives in the sqlite store. GPS is personal location data: the full
coordinate is stored, but nothing in this module's own report (there isn't
one showing individual photos) or in `tasks.doc_search`'s renders it as a
precise pin; see that module for the coarse-place display.

A file Pillow can't decode (HEIC without a HEIF plugin, a corrupt image, a
non-image with an image extension) is skipped and counted, never a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lmsa import LMStudio, Report, Store, Tile
from lmsa import geo
from lmsa.memo import Memo, cached as memo_cached, has_keys, memo_version, model_id
from lmsa.prompts import load as load_prompt, prompt_hash
from lmsa.vectors import embed_and_upsert
from tasks.attachments import VISION_JPEG_QUALITY, VISION_MAX_DIM, _downscale_jpeg

# What this tool needs, for the dashboard's Setup page (read by ast, never imported).
SETUP = {"env": ["LMSA_PHOTOS_DIR"], "model": "vision"}

NS = "photos"
DEFAULT_ROOT = "~/Pictures"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"}
MAX_NEW_PER_RUN = 100  # new/changed photos captioned per run; the rest wait for the next one
CAPTION_TEXT_LIMIT = 2000  # chars of caption+tags+detected_text embedded per photo
DETECTED_TEXT_LIMIT = 1500  # chars of on-image text kept (a photographed document can run long)
MEMO_VERSION = 1  # bump when caption() prompt building or result handling changes

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "detected_text": {"type": "string"},
    },
    "required": ["caption", "tags", "detected_text"],
    "additionalProperties": False,
}

CAPTION_SYSTEM = load_prompt("photo_index/caption_system")

# EXIF tag ids (see PIL.ExifTags): kept as literals so this module has no
# hard Pillow-version dependency on the ExifTags.Base/IFD enums.
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME = 0x0132
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4


@dataclass
class PhotoRecord:
    path: str
    content_hash: str
    mtime: float
    size: int
    caption: str
    tags: list[str]
    detected_text: str
    taken_at: str | None
    gps_lat: float | None
    gps_lon: float | None
    country: str | None


@dataclass
class Skip:
    path: str
    reason: str


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    unchanged: int = 0
    deferred: int = 0
    pruned: int = 0
    errors: list[Skip] = field(default_factory=list)


def default_root() -> Path:
    return Path(os.getenv("LMSA_PHOTOS_DIR") or DEFAULT_ROOT).expanduser()


def iter_images(root: Path):
    """Yield candidate image files under `root`, recursively, sorted for a
    deterministic, resumable order across runs. Skips dotfiles/dot-dirs."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not name.startswith(".") and p.suffix.lower() in IMAGE_EXTS:
                yield p


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dms_to_decimal(dms, ref: str | None) -> float | None:
    try:
        deg, minutes, sec = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    value = deg + minutes / 60 + sec / 3600
    if ref in ("S", "W"):
        value = -value
    return value


def read_exif(data: bytes) -> tuple[str | None, float | None, float | None]:
    """(taken_at ISO string or None, lat, lon) from `data`'s EXIF, best-effort.

    Never raises: missing/corrupt EXIF, or an image format with none at all,
    just comes back as (None, None, None).
    """
    taken_at = lat = lon = None
    try:
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            if exif:
                sub = exif.get_ifd(_IFD_EXIF)
                raw_dt = sub.get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
                if raw_dt:
                    try:
                        # EXIF datetimes are "YYYY:MM:DD HH:MM:SS", no timezone.
                        date_part, time_part = str(raw_dt).split(" ", 1)
                        taken_at = f"{date_part.replace(':', '-')}T{time_part}"
                        datetime.fromisoformat(taken_at)  # validate; discard if bogus
                    except (ValueError, TypeError):
                        taken_at = None
                gps = exif.get_ifd(_IFD_GPS)
                if gps:
                    lat = _dms_to_decimal(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF))
                    lon = _dms_to_decimal(gps.get(_GPS_LON), gps.get(_GPS_LON_REF))
                    if lat is not None and not (-90 <= lat <= 90):
                        lat = None
                    if lon is not None and not (-180 <= lon <= 180):
                        lon = None
    except Exception:
        return None, None, None
    return taken_at, lat, lon


def open_memo(conn, lm: LMStudio, *, fresh: bool = False) -> Memo:
    version = memo_version(
        MEMO_VERSION, prompt_hash("photo_index/caption_system"), CAPTION_SCHEMA,
        VISION_MAX_DIM, VISION_JPEG_QUALITY, model_id(lm, "vision_model"),
    )
    return Memo(conn, "photo_index", version, fresh=fresh)


def caption_photo(lm: LMStudio, small_jpeg: bytes) -> dict:
    return lm.see(
        [small_jpeg], "Describe this photo for search.", CAPTION_SCHEMA,
        system=CAPTION_SYSTEM, name="photo_caption",
    )


def _embed_text(caption: str, tags: list[str], detected_text: str) -> str:
    parts = [caption, " ".join(tags), detected_text[:DETECTED_TEXT_LIMIT]]
    text = "\n".join(p for p in parts if p)
    return text[:CAPTION_TEXT_LIMIT]


def _upsert_photo(conn, rec: PhotoRecord) -> None:
    conn.execute(
        """
        INSERT INTO photos (
            path, content_hash, mtime, size, caption, tags_json, detected_text,
            taken_at, gps_lat, gps_lon, country, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            content_hash=excluded.content_hash, mtime=excluded.mtime, size=excluded.size,
            caption=excluded.caption, tags_json=excluded.tags_json,
            detected_text=excluded.detected_text, taken_at=excluded.taken_at,
            gps_lat=excluded.gps_lat, gps_lon=excluded.gps_lon, country=excluded.country,
            indexed_at=excluded.indexed_at
        """,
        (
            rec.path, rec.content_hash, rec.mtime, rec.size, rec.caption,
            json.dumps(rec.tags), rec.detected_text, rec.taken_at, rec.gps_lat, rec.gps_lon,
            rec.country, datetime.now(timezone.utc).isoformat(),
        ),
    )


def _delete_photo(conn, path_str: str) -> None:
    conn.execute("DELETE FROM photos WHERE path=?", (path_str,))
    conn.execute("DELETE FROM embeddings WHERE ns=? AND item_id=?", (NS, path_str))


def _prune_missing(conn, root: Path) -> list[str]:
    """Remove rows + embeddings for previously-indexed paths under `root`
    that no longer exist on disk (deleted/renamed away)."""
    rows = [r[0] for r in conn.execute("SELECT path FROM photos").fetchall()]
    gone = []
    for path_str in rows:
        p = Path(path_str)
        try:
            under_root = p.resolve().is_relative_to(root.resolve())
        except (OSError, ValueError):
            under_root = False
        if under_root and not p.exists():
            gone.append(path_str)
    for path_str in gone:
        _delete_photo(conn, path_str)
    return gone


def index_photos(
    conn,
    lm: LMStudio,
    root: Path,
    *,
    max_new_per_run: int = MAX_NEW_PER_RUN,
    memo: Memo | None = None,
    model: str | None = None,
) -> IndexStats:
    model = model or lm.cfg.embed_model
    stats = IndexStats()
    new_budget = max_new_per_run

    for path in iter_images(root):
        stats.scanned += 1
        path_str = str(path)
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
            existing = conn.execute(
                "SELECT mtime, size FROM photos WHERE path=?", (path_str,)
            ).fetchone()
            if existing is not None and existing[0] == mtime and existing[1] == size:
                stats.unchanged += 1
                continue

            if new_budget <= 0:
                stats.deferred += 1
                continue

            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                im.verify()  # raises for anything Pillow can't actually decode (e.g. HEIC w/o plugin)

            content_hash = _content_hash(data)
            taken_at, lat, lon = read_exif(data)
            country = geo.nearest_country(lat, lon) if lat is not None and lon is not None else None
            small = _downscale_jpeg(data, max_dim=VISION_MAX_DIM, quality=VISION_JPEG_QUALITY)

            def compute() -> dict:
                return caption_photo(lm, small)

            result = memo_cached(
                memo, content_hash, compute,
                valid=lambda d: has_keys(d, CAPTION_SCHEMA["required"]),
            )
            if not isinstance(result, dict):
                raise ValueError(f"model returned {type(result).__name__}, expected an object")

            caption = str(result.get("caption") or "").strip()
            tags = [str(t).strip() for t in (result.get("tags") or []) if str(t).strip()]
            detected_text = str(result.get("detected_text") or "").strip()

            rec = PhotoRecord(
                path=path_str, content_hash=content_hash, mtime=mtime, size=size,
                caption=caption, tags=tags, detected_text=detected_text,
                taken_at=taken_at, gps_lat=lat, gps_lon=lon, country=country,
            )
            _upsert_photo(conn, rec)
            embed_and_upsert(
                conn, lm, ns=NS, model=model,
                items=[(path_str, _embed_text(caption, tags, detected_text))],
            )
            conn.commit()
            new_budget -= 1
            stats.indexed += 1
        except UnidentifiedImageError:
            stats.errors.append(Skip(path_str, "not decodable by Pillow (e.g. HEIC without a HEIF plugin)"))
        except Exception as exc:  # a corrupt/unreadable file must not abort the run
            stats.errors.append(Skip(path_str, f"{type(exc).__name__}: {exc}"))

    stats.pruned = len(_prune_missing(conn, root))
    conn.commit()
    return stats


def build_report(stats: IndexStats, root: Path) -> Report:
    r = Report(
        "Photo index",
        kicker="lmsa · photo search",
        subtitle=f"Local semantic index over {root}.",
        sticker=f"{stats.indexed} captioned" if stats.indexed else "up to date",
        sticker_tone="acc",
        source="tasks.photo_index",
    )
    r.tiles([
        Tile("Scanned", stats.scanned, tone="info"),
        Tile("Captioned (new/changed)", stats.indexed, tone="acc"),
        Tile("Unchanged", stats.unchanged, tone="muted"),
        Tile("Deferred (next run)", stats.deferred, tone="warn" if stats.deferred else "muted"),
        Tile("Pruned (gone)", stats.pruned, tone="warn" if stats.pruned else "muted"),
        Tile("Errors", len(stats.errors), tone="hot" if stats.errors else "muted"),
    ])
    if stats.deferred:
        r.section("Note", tone="warn").callout(
            f"Hit the per-run cap with {stats.deferred} new/changed photo(s) left; "
            "they'll be picked up on a later run.",
            label="capped",
            tone="warn",
        )
    if stats.errors:
        section = r.section("Photos that failed to index", tone="hot")
        for s in stats.errors:
            section.callout(s.reason, label=s.path, tone="hot")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Default: LMSA_PHOTOS_DIR or ~/Pictures")
    parser.add_argument("--max-new-per-run", type=int, default=100)  # keep in sync with MAX_NEW_PER_RUN
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore remembered captions and ask the model about every photo again"
    )
    args = parser.parse_args()

    root = args.root.expanduser() if args.root else default_root()

    lm = LMStudio()
    lm.ensure_server()

    with Store() as db:
        with db.record_run("photo_index") as run:
            memo = open_memo(db.conn, lm, fresh=args.fresh)
            stats = index_photos(db.conn, lm, root, max_new_per_run=args.max_new_per_run, memo=memo)
            report_path = build_report(stats, root).write(out_dir=lm.cfg.output_dir)
            run.summary = {
                "scanned": stats.scanned, "indexed": stats.indexed, "unchanged": stats.unchanged,
                "deferred": stats.deferred, "pruned": stats.pruned, "errors": len(stats.errors),
                **memo.summary(),
            }
            run.report_path = report_path

    print(f"{stats.indexed} photo(s) captioned, {len(stats.errors)} error(s) -> {report_path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
