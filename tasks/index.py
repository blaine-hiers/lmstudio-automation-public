"""Rebuild the report index hub.

    python -m tasks.index
"""

from __future__ import annotations

from lmsa.index import rebuild


def main() -> None:
    path = rebuild()
    print(f"Index written to {path}")


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
