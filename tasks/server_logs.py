"""Summarize LM Studio's own server logs into an Afterhours analytics report.

Reads ``~/.lmstudio/server-logs/**/*.log`` — LM Studio's request/response
log, not anything lmsa writes — and reports request counts, per-model token
and latency stats, model loads, plugin connections, errors, and a per-day
breakdown, aggregated across however many rotated log files fall in the
scanned window.

These logs can contain the owner's prompts and model replies verbatim. This
module never surfaces that text: request bodies are parsed only far enough
to pull out a model name and an item count, "Generated prediction" bodies
only for model/usage/finish_reason, and nothing else from either JSON block
is kept or printed. The report shows metadata only.

    python -m tasks.server_logs [--days N]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

from lmsa import Report, Store, Tile

DEFAULT_LOG_DIR = Path.home() / ".lmstudio" / "server-logs"
DEFAULT_DAYS = 7

_REQUEST_RE = re.compile(r"Received request: (?P<method>GET|POST) to (?P<endpoint>\S+)(?P<body> with body)?")
_MODEL_TAG_RE = re.compile(r"^\[[\d\- :]+\]\[[A-Z]+\]\[(?P<model>[\w.\-]+/[\w.\-]+)\] ")
_GEN_TIMING_RE = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms / +(\d+) tokens")
_LOAD_MODEL_RE = re.compile(r"load_model: loading model '(?P<path>[^']+)'")
_SEND_ERROR_RE = re.compile(r"E srv\s+send_error: task id = \d+, error: (?P<msg>.*)")
_PLUGIN_ERROR_RE = re.compile(r"\[ERROR\]\[Plugin\((?P<plugin>[^)]+)\)\] stderr: (?P<msg>.*)")
_PLUGIN_CLIENT_RE = re.compile(
    r"\[Client=plugin:installed:(?P<plugin>[\w/.\-]+)\] Client (?P<action>created|disconnected)\."
)
# Any bracketed "[YYYY-MM-DD HH:MM:SS]" line starts a new log entry. A JSON
# body block's own lines are always indented (pretty-printed, indent=2), so
# this can never legitimately appear *inside* a still-open block — seeing
# one means the previous block was abandoned without closing.
_ENTRY_START_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")
_LOG_FILENAME_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\.(?P<rot>\d+)\.log$")


@dataclass
class ChatStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reasons: Counter = field(default_factory=Counter)
    gen_ms: float = 0.0
    gen_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> float:
        return self.gen_tokens / (self.gen_ms / 1000) if self.gen_ms else 0.0


@dataclass
class EmbedStats:
    requests: int = 0
    items: int = 0


@dataclass
class DayStats:
    requests: int = 0
    chat_requests: int = 0
    embed_requests: int = 0
    errors: int = 0
    chat_by_model: Counter = field(default_factory=Counter)


@dataclass
class LogStats:
    source: Path
    lines_read: int = 0
    files_scanned: int = 0
    endpoints: Counter = field(default_factory=Counter)
    chat_by_model: dict[str, ChatStats] = field(default_factory=lambda: defaultdict(ChatStats))
    embed_by_model: dict[str, EmbedStats] = field(default_factory=lambda: defaultdict(EmbedStats))
    model_loads: Counter = field(default_factory=Counter)
    plugin_connects: Counter = field(default_factory=Counter)
    plugin_disconnects: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    error_samples: dict[str, str] = field(default_factory=dict)
    # Entries whose JSON body never closed. `malformed_blocks` counts ones
    # abandoned mid-file (a later entry started first — logged and skipped,
    # not a sign of anything still being written). `truncated_tail`/
    # `truncated_files` are for the one genuine case: the block ran off the
    # actual end of a file, i.e. that file was still being written.
    malformed_blocks: int = 0
    truncated_tail: bool = False
    truncated_files: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    daily: dict[str, DayStats] = field(default_factory=lambda: defaultdict(DayStats))


def _classify_error(msg: str) -> str:
    if "context size" in msg:
        return "context_overflow"
    if "WebSocket" in msg:
        return "websocket"
    return "server_error"


def _json_block(lines: list[str], start: int) -> tuple[dict | None, int]:
    """Parse a pretty-printed (``indent=2``) JSON object starting at ``lines[start]``.

    The object's top-level closing brace is the only line that is exactly
    ``}`` with no indentation, so that's the terminator we look for.

    Returns ``(None, index_of_first_unconsumed_line)`` in two distinct cases,
    which the caller tells apart by comparing that index to ``len(lines)``:

    - a later log entry started before this block closed: the returned
      index points *at* that entry's own line, unconsumed, so the caller
      resumes scanning normally from there;
    - the block ran off the real end of the file: the returned index is
      ``len(lines)``, meaning the file was still being written.
    """
    first = lines[start]
    brace = first.rfind("{")
    buf = [first[brace:]]
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if _ENTRY_START_RE.match(line):
            return None, i  # abandoned: a new entry began, this line unconsumed
        buf.append(line)
        if line == "}":
            try:
                return json.loads("\n".join(buf)), i + 1
            except json.JSONDecodeError:
                return None, i + 1
        i += 1
    return None, i  # ran off the real end of the file


def _consume_json_block(lines: list[str], start: int, stats: LogStats) -> tuple[dict | None, int]:
    """`_json_block` plus bookkeeping: classify a failure as malformed vs. mid-write."""
    body, i = _json_block(lines, start)
    if body is None:
        stats.malformed_blocks += 1
        if i >= len(lines):
            stats.truncated_tail = True
    return body, i


def parse_log(path: Path) -> LogStats:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stats = LogStats(source=path, lines_read=len(lines), files_scanned=1)
    current_model: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]

        tag = _MODEL_TAG_RE.match(line)
        if tag:
            current_model = tag.group("model")

        req = _REQUEST_RE.search(line)
        if req:
            stats.endpoints[f'{req.group("method")} {req.group("endpoint")}'] += 1
            if req.group("body"):
                body, i = _consume_json_block(lines, i, stats)
                if body is not None and req.group("endpoint") == "/v1/embeddings":
                    model = body.get("model", "unknown")
                    items = body.get("input")
                    count = len(items) if isinstance(items, list) else (1 if items else 0)
                    es = stats.embed_by_model[model]
                    es.requests += 1
                    es.items += count
                # Chat/completions bodies carry the prompt in `messages`; the
                # model name is read back off the (safe) prediction JSON
                # instead, so the body is discarded once consumed.
                continue
            i += 1
            continue

        if "Generated prediction:" in line:
            body, i = _consume_json_block(lines, i, stats)
            if body is not None:
                model = body.get("model") or current_model or "unknown"
                usage = body.get("usage") or {}
                cs = stats.chat_by_model[model]
                cs.requests += 1
                cs.prompt_tokens += usage.get("prompt_tokens", 0)
                cs.completion_tokens += usage.get("completion_tokens", 0)
                for choice in body.get("choices", []):
                    cs.finish_reasons[choice.get("finish_reason", "unknown")] += 1
            continue

        err = _SEND_ERROR_RE.search(line)
        if err:
            cat = _classify_error(err.group("msg"))
            stats.errors[cat] += 1
            stats.error_samples.setdefault(cat, err.group("msg").replace("\n", " ")[:120])
            i += 1
            continue

        perr = _PLUGIN_ERROR_RE.search(line)
        if perr:
            cat = f'plugin: {perr.group("plugin")}'
            stats.errors[cat] += 1
            stats.error_samples.setdefault(cat, perr.group("msg").replace("\n", " ")[:120])
            i += 1
            continue

        client = _PLUGIN_CLIENT_RE.search(line)
        if client:
            counter = stats.plugin_connects if client.group("action") == "created" else stats.plugin_disconnects
            counter[client.group("plugin")] += 1
            i += 1
            continue

        load = _LOAD_MODEL_RE.search(line)
        if load:
            stats.model_loads[Path(load.group("path")).name] += 1
            i += 1
            continue

        gen = _GEN_TIMING_RE.search(line)
        if gen and current_model:
            cs = stats.chat_by_model[current_model]
            cs.gen_ms += float(gen.group(1))
            cs.gen_tokens += int(gen.group(2))

        i += 1

    return stats


def _sort_key(path: Path) -> tuple[str, int]:
    m = _LOG_FILENAME_RE.search(path.name)
    if m:
        return (m.group("date"), int(m.group("rot")))
    # Doesn't match LM Studio's own "YYYY-MM-DD.N.log" convention: fall back
    # to mtime so an oddly named file still sorts in and doesn't error the scan.
    return (datetime.fromtimestamp(path.stat().st_mtime).date().isoformat(), 0)


def _day_key(path: Path) -> str:
    return _sort_key(path)[0]


def find_log_files(log_dir: Path = DEFAULT_LOG_DIR, days: int | None = None) -> list[Path]:
    """All ``*.log`` files under ``log_dir``, oldest first.

    Rotated files for the same day (``YYYY-MM-DD.1.log``, ``.2.log``, ...)
    are all included. With `days`, only files dated within that many
    calendar days of the most recent file present are kept.
    """
    files = sorted((p for p in log_dir.glob("**/*.log") if p.is_file()), key=_sort_key)
    if days is not None and files:
        latest_day = date.fromisoformat(_sort_key(files[-1])[0])
        cutoff = (latest_day - timedelta(days=days - 1)).isoformat()
        files = [p for p in files if _sort_key(p)[0] >= cutoff]
    return files


def find_latest_log(log_dir: Path = DEFAULT_LOG_DIR) -> Path | None:
    files = find_log_files(log_dir)
    return files[-1] if files else None


def aggregate_logs(paths: Sequence[Path]) -> LogStats:
    """Parse and merge several log files into one `LogStats`.

    A file that can't be read (permissions, disappeared mid-scan, etc.) is
    recorded in `unreadable_files` and skipped — it never aborts the run.
    """
    combined = LogStats(source=paths[-1] if paths else DEFAULT_LOG_DIR)
    for path in paths:
        try:
            stats = parse_log(path)
        except OSError as exc:
            combined.unreadable_files.append(f"{path.name}: {exc}")
            continue

        combined.files_scanned += stats.files_scanned
        combined.lines_read += stats.lines_read
        combined.endpoints.update(stats.endpoints)
        combined.model_loads.update(stats.model_loads)
        combined.plugin_connects.update(stats.plugin_connects)
        combined.plugin_disconnects.update(stats.plugin_disconnects)
        combined.errors.update(stats.errors)
        for cat, sample in stats.error_samples.items():
            combined.error_samples.setdefault(cat, sample)
        combined.malformed_blocks += stats.malformed_blocks
        if stats.truncated_tail:
            combined.truncated_tail = True
            combined.truncated_files.append(path.name)

        for model, cs in stats.chat_by_model.items():
            acc = combined.chat_by_model[model]
            acc.requests += cs.requests
            acc.prompt_tokens += cs.prompt_tokens
            acc.completion_tokens += cs.completion_tokens
            acc.gen_ms += cs.gen_ms
            acc.gen_tokens += cs.gen_tokens
            acc.finish_reasons.update(cs.finish_reasons)

        for model, es in stats.embed_by_model.items():
            acc = combined.embed_by_model[model]
            acc.requests += es.requests
            acc.items += es.items

        day = combined.daily[_day_key(path)]
        day.requests += sum(stats.endpoints.values())
        day.chat_requests += sum(cs.requests for cs in stats.chat_by_model.values())
        day.embed_requests += sum(es.requests for es in stats.embed_by_model.values())
        day.errors += sum(stats.errors.values())
        for model, cs in stats.chat_by_model.items():
            day.chat_by_model[model] += cs.requests

    return combined


def build_report(stats: LogStats) -> Report:
    total_requests = sum(stats.endpoints.values())
    n_chat = sum(s.requests for s in stats.chat_by_model.values())
    n_embed = sum(s.requests for s in stats.embed_by_model.values())
    n_errors = sum(stats.errors.values())
    total_tokens = sum(s.total_tokens for s in stats.chat_by_model.values())
    files_scanned = stats.files_scanned or 1

    r = Report(
        "LM Studio server log analytics",
        kicker="lmsa · diagnostics",
        subtitle=f"Metadata-only summary of {files_scanned} file(s), {stats.lines_read:,} lines.",
        sticker="clean" if n_errors == 0 else f"{n_errors} errors",
        sticker_tone="acc" if n_errors == 0 else "hot",
        source=str(stats.source),
    )
    r.tiles([
        Tile("Requests", total_requests, tone="info"),
        Tile("Chat completions", n_chat, tone="acc"),
        Tile("Embedding calls", n_embed, tone="violet"),
        Tile("Chat tokens", total_tokens, tone="muted"),
        Tile("Errors", n_errors, tone="hot" if n_errors else "acc"),
    ])

    r.section("Requests by endpoint", tone="info").table(
        ["Endpoint", "Count"], sorted(stats.endpoints.items(), key=lambda kv: -kv[1])
    )

    if stats.daily:
        days = sorted(stats.daily)
        r.section("Requests per day", tone="info").table(
            ["Day", "Requests", "Chat", "Embeddings", "Errors"],
            [
                [d, stats.daily[d].requests, stats.daily[d].chat_requests, stats.daily[d].embed_requests, stats.daily[d].errors]
                for d in days
            ],
        )
        r.section("Daily request volume", tone="info").bars([(d, stats.daily[d].requests) for d in days])
        model_day_rows = [
            [d, model, count] for d in days for model, count in stats.daily[d].chat_by_model.most_common()
        ]
        if model_day_rows:
            r.section("Chat completions per model per day", tone="acc").table(
                ["Day", "Model", "Requests"], model_day_rows
            )

    if stats.chat_by_model:
        rows = [
            [
                model,
                cs.requests,
                cs.prompt_tokens,
                cs.completion_tokens,
                round(cs.tokens_per_sec, 1),
                cs.requests - cs.finish_reasons.get("stop", 0),
            ]
            for model, cs in sorted(stats.chat_by_model.items(), key=lambda kv: -kv[1].requests)
        ]
        r.section("Chat completions by model", tone="acc").table(
            ["Model", "Requests", "Prompt tok", "Completion tok", "Gen tok/s", "Non-stop finish"], rows
        )

    if stats.embed_by_model:
        rows = [[model, es.requests, es.items] for model, es in stats.embed_by_model.items()]
        r.section("Embeddings by model", tone="violet").table(["Model", "Requests", "Items embedded"], rows)

    if stats.model_loads:
        r.section("Model loads (JIT swaps)", tone="warn").table(
            ["Model file", "Loads"], sorted(stats.model_loads.items(), key=lambda kv: -kv[1])
        )

    if stats.plugin_connects or stats.plugin_disconnects:
        plugins = sorted(set(stats.plugin_connects) | set(stats.plugin_disconnects))
        rows = [[p, stats.plugin_connects.get(p, 0), stats.plugin_disconnects.get(p, 0)] for p in plugins]
        r.section("Plugin connections", tone="info").table(["Plugin", "Connected", "Disconnected"], rows)

    errs = r.section("Errors", tone="hot" if n_errors else "acc")
    if stats.errors:
        rows = [[cat, count, stats.error_samples.get(cat, "")] for cat, count in stats.errors.most_common()]
        errs.table(["Type", "Count", "Sample"], rows)
    else:
        errs.callout("No errors found in this window.", label="clean", tone="acc")

    notes = []
    if stats.malformed_blocks:
        plural = "y" if stats.malformed_blocks == 1 else "ies"
        notes.append(
            f"{stats.malformed_blocks} log entr{plural} had an unterminated JSON body and "
            "were skipped (not counted as a completion)."
        )
    if stats.truncated_files:
        notes.append("Mid-write tail skipped in: " + ", ".join(stats.truncated_files) + ".")
    elif stats.truncated_tail:
        notes.append("The last entry in the log was still being written when this ran, so it was skipped.")
    if stats.unreadable_files:
        notes.append("Could not read: " + ", ".join(stats.unreadable_files) + ".")
    if notes:
        section = r.section("Notes", tone="muted")
        for note in notes:
            section.callout(note, label="note", tone="info")

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LM Studio server logs.")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS, help=f"most recent calendar days of logs to scan (default: {DEFAULT_DAYS})"
    )
    args = parser.parse_args()

    files = find_log_files(days=args.days)
    if not files:
        raise SystemExit(f"No LM Studio server logs found under {DEFAULT_LOG_DIR}")
    with Store() as store, store.record_run("tasks.server_logs") as run:
        stats = aggregate_logs(files)
        path = build_report(stats).write()
        run.report_path = path
        run.summary = {
            "files": stats.files_scanned,
            "lines": stats.lines_read,
            "chat_models": len(stats.chat_by_model),
            "embed_models": len(stats.embed_by_model),
        }
    print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
