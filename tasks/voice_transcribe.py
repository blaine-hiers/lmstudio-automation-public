"""Voice memos, phase 1: transcribe new recordings with faster-whisper.

    python -m tasks.voice_transcribe [--days 7]

The same as `python -m tasks.voice_memos --phase transcribe`; it is its own
module so the dispatcher can schedule it in the "whisper" model class (run
with no LM Studio model loaded) ahead of `voice_memos` in the chat batch.
Needs no LM Studio server. See tasks/voice_memos.py.
"""

from __future__ import annotations

import argparse
import sys

from tasks import voice_memos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tasks.voice_transcribe", description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=7,  # literal: the dashboard reads it by ast
                        help="only recordings modified in the last N days (default 7)")
    parser.add_argument("--max-mb", type=float, default=500,  # literal: the dashboard reads it by ast
                        help="skip (and report) recordings bigger than this many MB (default 500)")
    parser.add_argument("--budget-min", type=float, default=105,  # literal: the dashboard reads it by ast
                        help="start no new transcription after this many minutes (default 105; "
                             "keep it under the schedule's timeout)")
    parser.add_argument("--reset-attempts", action="store_true",
                        help="forget failed transcription attempts, so given-up files are tried again")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    voice_memos.main(["--phase", "transcribe", "--days", str(args.days), "--max-mb", f"{args.max_mb:g}",
                      "--budget-min", f"{args.budget_min:g}"] + (["--reset-attempts"] if args.reset_attempts else []))


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
