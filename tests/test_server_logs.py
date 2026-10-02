import json
from pathlib import Path

from tasks.server_logs import aggregate_logs, build_report, find_latest_log, find_log_files, parse_log

FIXTURE = Path(__file__).parent / "fixtures" / "server_logs" / "sample.log"

# Every placeholder value planted in the fixture's prompt/response/embedding
# text. None of these may ever reach the rendered report.
_PLACEHOLDERS = [
    "PLACEHOLDER_PROMPT_TEXT_ONE",
    "PLACEHOLDER_RESPONSE_TEXT_ONE",
    "PLACEHOLDER_PROMPT_TEXT_TWO",
    "PLACEHOLDER_RESPONSE_TEXT_TWO",
    "PLACEHOLDER_EMBED_TEXT_ONE",
    "PLACEHOLDER_EMBED_TEXT_TWO",
    "PLACEHOLDER_EMBED_TEXT_THREE",
    "PLACEHOLDER_PROMPT_TEXT_MID_WRITE",
]


def test_endpoint_counts():
    stats = parse_log(FIXTURE)
    assert stats.endpoints["GET /v1/models"] == 1
    assert stats.endpoints["POST /v1/chat/completions"] == 3  # incl. the mid-write one
    assert stats.endpoints["POST /v1/embeddings"] == 1


def test_chat_stats_by_model():
    stats = parse_log(FIXTURE)
    assert set(stats.chat_by_model) == {"qwen/qwen3-14b", "qwen/qwen3-vl-8b"}

    a = stats.chat_by_model["qwen/qwen3-14b"]
    assert a.requests == 1
    assert a.prompt_tokens == 20
    assert a.completion_tokens == 6
    assert a.total_tokens == 26
    assert a.finish_reasons["stop"] == 1
    assert a.tokens_per_sec == 50.0  # 10 tokens / 0.2s

    b = stats.chat_by_model["qwen/qwen3-vl-8b"]
    assert b.requests == 1
    assert b.finish_reasons["length"] == 1
    assert b.tokens_per_sec == 100.0  # 50 tokens / 0.5s


def test_embed_stats():
    stats = parse_log(FIXTURE)
    es = stats.embed_by_model["text-embedding-nomic-embed-text-v1.5"]
    assert es.requests == 1
    assert es.items == 3


def test_model_loads_and_plugin_connections():
    stats = parse_log(FIXTURE)
    assert stats.model_loads["Qwen3-VL-8B-Instruct-Q4_K_M.gguf"] == 2
    assert stats.plugin_connects["sample/plugin-a"] == 1
    assert stats.plugin_connects["sample/plugin-b"] == 1
    assert stats.plugin_disconnects["sample/plugin-a"] == 1
    assert "sample/plugin-b" not in stats.plugin_disconnects


def test_errors_are_classified():
    stats = parse_log(FIXTURE)
    assert stats.errors["context_overflow"] == 1
    assert stats.errors["plugin: sample/plugin-a"] == 1


def test_mid_write_tail_is_skipped_not_crashed():
    """The last request in the fixture never closes its JSON body (simulating
    a log file caught mid-write). Parsing must finish cleanly rather than
    raise, and must flag that the tail was incomplete."""
    stats = parse_log(FIXTURE)
    assert stats.truncated_tail is True
    # The dangling request must not have been double-counted as a completion.
    assert sum(cs.requests for cs in stats.chat_by_model.values()) == 2


def test_report_never_contains_prompt_or_response_text():
    stats = parse_log(FIXTURE)
    html = build_report(stats).render()
    for placeholder in _PLACEHOLDERS:
        assert placeholder not in html


def test_report_contains_expected_metadata():
    stats = parse_log(FIXTURE)
    html = build_report(stats).render()
    assert "qwen/qwen3-14b" in html
    assert "qwen/qwen3-vl-8b" in html
    assert "text-embedding-nomic-embed-text-v1.5" in html
    assert "context_overflow" in html
    assert "Qwen3-VL-8B-Instruct-Q4_K_M.gguf" in html


def test_find_latest_log_picks_newest_mtime(tmp_path):
    import time

    month_dir = tmp_path / "2026-01"
    month_dir.mkdir()
    older = month_dir / "2026-01-04.1.log"
    newer = month_dir / "2026-01-05.1.log"
    older.write_text("old", encoding="utf-8")
    time.sleep(0.01)
    newer.write_text("new", encoding="utf-8")

    assert find_latest_log(tmp_path) == newer


def test_find_latest_log_returns_none_when_empty(tmp_path):
    assert find_latest_log(tmp_path) is None


# ---------------------------------------------------------------------------
# Synthetic-entry helpers for the regression tests below. Built with
# json.dumps(..., indent=2) rather than hand-typed indentation, so the
# generated blocks match LM Studio's own pretty-printing exactly (in
# particular: only the true top-level "}" is flush-left). All text values
# are invented placeholders, never anything resembling a real prompt/reply.
# ---------------------------------------------------------------------------


def _json_lines(prefix: str, obj: dict) -> list[str]:
    dumped = json.dumps(obj, indent=2).splitlines()
    return [prefix + dumped[0]] + dumped[1:]


def _chat_entry(ts: str, model: str, prompt_tokens: int, completion_tokens: int, *, finish: str = "stop", tag: str = "X") -> list[str]:
    request_body = {
        "messages": [{"role": "user", "content": f"PLACEHOLDER_PROMPT_{tag}"}],
        "model": model,
        "max_tokens": 200,
        "temperature": 0.2,
    }
    prediction = {
        "id": f"chatcmpl-{tag}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"PLACEHOLDER_RESPONSE_{tag}",
                    "reasoning_content": "",
                    "tool_calls": [],
                },
                "logprobs": None,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "stats": {},
        "system_fingerprint": model,
    }
    return [
        *_json_lines(f"[{ts}][DEBUG] Received request: POST to /v1/chat/completions with body ", request_body),
        f"[{ts}][INFO][{model}] Running chat completion on conversation with 1 messages.",
        *_json_lines(f"[{ts}][INFO][{model}] Generated prediction: ", prediction),
    ]


def test_two_copies_of_fixture_do_not_lose_the_second_copy(tmp_path):
    """Repro from review: concatenating two copies of a log whose first copy
    ends mid-request used to make the whole scan `break` on the bogus
    unterminated block, silently dropping every line of the second copy."""
    original = FIXTURE.read_text(encoding="utf-8")
    doubled_path = tmp_path / "doubled.log"
    doubled_path.write_text(original + "\n" + original, encoding="utf-8")

    single = parse_log(FIXTURE)
    doubled = parse_log(doubled_path)

    assert doubled.chat_by_model["qwen/qwen3-14b"].requests == 2 * single.chat_by_model["qwen/qwen3-14b"].requests
    assert doubled.chat_by_model["qwen/qwen3-vl-8b"].requests == 2 * single.chat_by_model["qwen/qwen3-vl-8b"].requests
    assert (
        doubled.embed_by_model["text-embedding-nomic-embed-text-v1.5"].items
        == 2 * single.embed_by_model["text-embedding-nomic-embed-text-v1.5"].items
    )
    assert sum(doubled.errors.values()) == 2 * sum(single.errors.values())
    # Only the second copy's dangling entry is genuinely at EOF: the first
    # copy's dangling entry is abandoned (malformed) when copy two's first
    # line starts, but the file itself doesn't end there.
    assert doubled.truncated_tail is True
    assert doubled.malformed_blocks == 2


def test_malformed_block_mid_file_does_not_swallow_later_entries(tmp_path):
    """An unterminated JSON body that is NOT at the end of the file (a later
    bracketed entry starts before it ever closes) must be skipped on its
    own, without discarding everything that follows it."""
    ts1, ts2 = "2026-02-01 10:00:00", "2026-02-01 10:05:00"
    full = _chat_entry(ts1, "model-a", 10, 5, tag="A")
    # Cut the request body off before its closing brace, so it never
    # terminates — the very next line is a fresh bracketed log entry.
    truncated = full[:3]
    entry2 = _chat_entry(ts2, "model-b", 20, 8, tag="B")

    log_path = tmp_path / "2026-02-01.1.log"
    log_path.write_text("\n".join(truncated + entry2), encoding="utf-8")

    stats = parse_log(log_path)
    assert stats.malformed_blocks == 1
    assert stats.truncated_tail is False
    assert stats.chat_by_model["model-a"].requests == 0
    assert stats.chat_by_model["model-b"].requests == 1
    assert stats.chat_by_model["model-b"].prompt_tokens == 20


def test_find_log_files_merges_rotated_files_within_a_day(tmp_path):
    month_dir = tmp_path / "2026-02"
    month_dir.mkdir()
    day1_a = month_dir / "2026-02-01.1.log"
    day1_b = month_dir / "2026-02-01.2.log"
    day2_a = month_dir / "2026-02-02.1.log"

    day1_a.write_text("\n".join(_chat_entry("2026-02-01 08:00:00", "model-a", 10, 5, tag="D1A")), encoding="utf-8")
    day1_b.write_text("\n".join(_chat_entry("2026-02-01 20:00:00", "model-a", 10, 5, tag="D1B")), encoding="utf-8")
    day2_a.write_text("\n".join(_chat_entry("2026-02-02 08:00:00", "model-b", 10, 5, tag="D2A")), encoding="utf-8")

    files = find_log_files(tmp_path)
    assert [p.name for p in files] == ["2026-02-01.1.log", "2026-02-01.2.log", "2026-02-02.1.log"]

    stats = aggregate_logs(files)
    assert stats.files_scanned == 3
    assert stats.chat_by_model["model-a"].requests == 2  # merged across both day-1 rotation files
    assert stats.chat_by_model["model-b"].requests == 1
    assert stats.daily["2026-02-01"].chat_requests == 2
    assert stats.daily["2026-02-02"].chat_requests == 1
    assert stats.daily["2026-02-01"].chat_by_model["model-a"] == 2


def test_find_log_files_days_filter(tmp_path):
    month_dir = tmp_path / "2026-02"
    month_dir.mkdir()
    for d in ("2026-02-01", "2026-02-05", "2026-02-08"):
        (month_dir / f"{d}.1.log").write_text("placeholder log content", encoding="utf-8")

    # latest file is 2026-02-08; days=5 -> cutoff is 2026-02-04, so the
    # 2026-02-01 file (older than the window) is excluded.
    files = find_log_files(tmp_path, days=5)
    assert [p.name for p in files] == ["2026-02-05.1.log", "2026-02-08.1.log"]


def test_aggregate_logs_skips_unreadable_file(tmp_path):
    """One file that can't be read must not abort the whole scan."""
    good = tmp_path / "2026-02-10.1.log"
    good.write_text("\n".join(_chat_entry("2026-02-10 08:00:00", "model-a", 10, 5, tag="G")), encoding="utf-8")
    missing = tmp_path / "2026-02-11.1.log"  # never created

    stats = aggregate_logs([good, missing])
    assert stats.files_scanned == 1
    assert stats.chat_by_model["model-a"].requests == 1
    assert any("2026-02-11.1.log" in name for name in stats.unreadable_files)


def test_aggregate_report_shows_daily_breakdown_without_leaking_placeholders(tmp_path):
    month_dir = tmp_path / "2026-02"
    month_dir.mkdir()
    (month_dir / "2026-02-01.1.log").write_text(
        "\n".join(_chat_entry("2026-02-01 08:00:00", "model-a", 10, 5, tag="P1")), encoding="utf-8"
    )
    (month_dir / "2026-02-02.1.log").write_text(
        "\n".join(_chat_entry("2026-02-02 08:00:00", "model-a", 10, 5, tag="P2")), encoding="utf-8"
    )

    stats = aggregate_logs(find_log_files(tmp_path))
    html = build_report(stats).render()

    assert "2026-02-01" in html and "2026-02-02" in html
    assert "model-a" in html
    assert "PLACEHOLDER_PROMPT_P1" not in html
    assert "PLACEHOLDER_RESPONSE_P1" not in html
