"""`run(main)`: wraps a task's `main()` so an unattended failure gets noticed.

    if __name__ == "__main__":
        from lmsa.runner import run

        run(main)
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

from .notify import notify

log = logging.getLogger(__name__)


def _task_name(main: Callable[[], None]) -> str:
    """The dotted module name, even under `python -m tasks.x` (where __module__ is '__main__')."""
    module = getattr(main, "__module__", None)
    spec = getattr(sys.modules.get(module), "__spec__", None) if module else None
    return getattr(spec, "name", None) or module or getattr(main, "__name__", "task")


def run(main: Callable[[], None]) -> None:
    """Call `main()`; on exception, send an urgent alert then re-raise.

    The re-raise keeps the process exit code non-zero so Task Scheduler
    still sees the run as failed.
    """
    try:
        main()
    except Exception as exc:
        try:
            notify(f"{_task_name(main)} failed", f"{type(exc).__name__}: {exc}", level="urgent")
        except Exception:
            log.warning("failure alert could not be sent", exc_info=True)
        raise
