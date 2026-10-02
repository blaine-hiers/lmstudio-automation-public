"""Where generated files live under `output/`: `output/<YYYY-MM-DD>/<type>/<file>`.

The day and type come from the file name, which every task already stamps as
`<type>-YYYYMMDD[-HHMM[SS]].<ext>` (`dispatch-20260930.html`,
`master-report-morning-20260928-211437.eml`). Files with no date in the name
(`index.html`, `capabilities.html`, `style-preview.html`) stay at the top level.

    placed(out_dir, name)   # where a new file belongs (creates the folder)
    link(path, out_dir)     # href to `path` from a report written via placed()
    sort_existing(out_dir)  # move stray flat files into the layout (run by hand, not on write)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_STAMPED = re.compile(r"^(?P<type>.+?)-(?P<day>\d{8})(?:-\d{4,6})?\.[A-Za-z0-9]+$")


def parse(name: str) -> tuple[str, str] | None:
    """`(day "YYYY-MM-DD", type)` for a stamped file name, else None."""
    m = _STAMPED.match(name)
    if not m:
        return None
    try:
        day = datetime.strptime(m["day"], "%Y%m%d")
    except ValueError:
        return None
    return f"{day:%Y-%m-%d}", m["type"]


def target(out_dir: Path, name: str) -> Path:
    """Where `name` belongs under `out_dir` (no filesystem access)."""
    parsed = parse(name)
    if parsed is None:
        return out_dir / name
    day, kind = parsed
    return out_dir / day / kind / name


def placed(out_dir: Path, name: str) -> Path:
    """`target()`, with its folder created."""
    path = target(out_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def link(path: Path, out_dir: Path) -> str:
    """Href to `path` from a report that sits in `<out_dir>/<day>/<type>/`
    (works file-opened and served under the dashboard's `/reports/`)."""
    try:
        rel = path.resolve().relative_to(out_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name
    return f"../../{rel}"


def sort_existing(out_dir: Path) -> int:
    """Move stamped files sitting directly in `out_dir` into the layout.
    Never overwrites; returns how many were moved."""
    moved = 0
    if not out_dir.is_dir():
        return 0
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or parse(p.name) is None:
            continue
        dest = target(out_dir, p.name)
        if dest.exists():
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.rename(dest)
            moved += 1
        except OSError:
            continue
    return moved


if __name__ == "__main__":
    from .config import settings
    from . import index

    d = settings().output_dir
    print(f"moved {sort_existing(d)} file(s) into {d}")
    index.rebuild(d)
