"""tasks.voice_memos / tasks.voice_transcribe with a fake transcriber and a
fake LM: no faster-whisper, no LM Studio, no network."""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa import voice
from lmsa.store import Store
from tasks import voice_memos as vm
from tasks import voice_transcribe

NOW = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)

ANSWER = {
    "title": "Launch sync",
    "summary": "We agreed to ship on Friday.",
    "decisions": ["Ship on Friday"],
    "action_items": [{"task": "Send the release notes", "owner": "Sam", "due": "Thursday"},
                     {"task": "Book the demo room", "owner": "", "due": ""}],
    "follow_up_questions": ["Who tells support?"],
}


class FakeTranscriber:
    def __init__(self, log, fail=()):
        self.log, self.fail = log, set(fail)
        log.append(("load",))

    def transcribe(self, path):
        self.log.append(("transcribe", Path(path).name))
        if Path(path).name in self.fail:
            raise RuntimeError("corrupt audio")
        return {"text": f"words from {Path(path).name}", "language": "en", "duration_s": 75.0,
                "segments": [{"start": 0.0, "end": 75.0, "text": f"words from {Path(path).name}"}],
                "model": "fake"}

    def close(self):
        self.log.append(("close",))


class FakeLM:
    def __init__(self, log, answer=ANSWER, fail=False):
        self.log, self.answer, self.fail = log, answer, fail
        self.prompts: list[str] = []

    def ensure_server(self):
        self.log.append(("ensure_server",))

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.log.append(("extract",))
        self.prompts.append(prompt)
        assert system and "untrusted" in system
        if self.fail:
            raise TimeoutError("server timed out")
        return json.loads(json.dumps(self.answer))


def _audio(root: Path, name: str, data: bytes = b"RIFFdata", age=timedelta(hours=1)) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    ts = (NOW - age).timestamp()
    os.utime(p, (ts, ts))
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    rec = tmp_path / "Recordings"
    rec.mkdir()
    monkeypatch.setenv("LMSA_RECORDINGS_DIR", str(rec))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "data" / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "out"))
    return rec


def _run(argv, log, *, fail_files=(), lm=None):
    vm.main(argv, make_transcriber=lambda: FakeTranscriber(log, fail_files), lm=lm or FakeLM(log), now=NOW)


# -- scan ------------------------------------------------------------------------

def test_scan_filters_by_extension_window_and_settling(env):
    _audio(env, "a.m4a")
    _audio(env, "sub/b.MP3")
    _audio(env, "notes.txt")
    _audio(env, "old.wav", age=timedelta(days=30))
    _audio(env, "live.ogg", age=timedelta(seconds=30))
    found = vm.scan(env, days=7, now=NOW)
    assert sorted(r.name for r in found.recordings) == ["a.m4a", "b.MP3"]
    assert found.settling == ["live.ogg"]
    assert all(len(r.sha256) == 64 for r in found.recordings)


def test_scan_caps_files_and_reports_missing_dir(env, tmp_path):
    for i in range(3):
        _audio(env, f"r{i}.wav", data=bytes([i]), age=timedelta(hours=i + 1))
    found = vm.scan(env, days=7, now=NOW, max_files=2)
    assert [r.name for r in found.recordings] == ["r0.wav", "r1.wav"] and found.capped == 1
    assert vm.scan(tmp_path / "nope", days=7, now=NOW).missing_dir


# -- phase separation ---------------------------------------------------------------

def test_transcribe_phase_never_touches_lm_studio(env):
    _audio(env, "a.m4a")
    log = []

    class NoLM:
        def __getattr__(self, name):
            raise AssertionError(f"LM used in the transcribe phase: {name}")

    _run(["--phase", "transcribe"], log, lm=NoLM())
    assert log == [("load",), ("transcribe", "a.m4a"), ("close",)]
    out = Path(os.environ["LMSA_OUTPUT_DIR"]) / "voice_memos"
    assert len(list(out.glob("a-*.txt"))) == 1 and len(list(out.glob("a-*.md"))) == 1


def test_summarize_phase_never_loads_whisper_and_waits_for_transcripts(env):
    _audio(env, "a.m4a")
    log = []

    def no_whisper():
        raise AssertionError("whisper loaded in the summarize phase")

    lm = FakeLM(log)
    vm.main([], make_transcriber=no_whisper, lm=lm, now=NOW)  # default phase: summarize
    assert ("extract",) not in log  # nothing transcribed yet: nothing to summarize
    with Store() as db:
        row = db.conn.execute("SELECT summary_json FROM runs WHERE task='tasks.voice_memos'").fetchone()
    assert json.loads(row[0])["awaiting_transcription"] == 1


def test_phase_all_closes_whisper_before_the_first_lm_call(env):
    _audio(env, "a.m4a")
    _audio(env, "b.wav", data=b"other")
    log = []
    _run(["--phase", "all"], log)
    close = log.index(("close",))
    assert log.index(("ensure_server",)) > close
    assert all(i > close for i, e in enumerate(log) if e == ("extract",))
    assert log.count(("extract",)) == 2


# -- memo reuse ---------------------------------------------------------------------

def test_memo_reuse_transcribes_and_summarizes_each_file_once(env):
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "transcribe"], log)
    _run(["--phase", "transcribe"], log)
    assert log.count(("transcribe", "a.m4a")) == 1
    assert log.count(("load",)) == 1  # nothing to do: whisper isn't even loaded
    _run([], log)
    _run([], log)
    assert log.count(("extract",)) == 1
    # a renamed copy has the same content hash: reused, not redone
    (env / "a.m4a").rename(env / "renamed.m4a")
    _run(["--phase", "all"], log)
    assert log.count(("transcribe", "renamed.m4a")) == 0 and log.count(("extract",)) == 1
    _run(["--fresh"], log)  # --fresh re-summarizes, never re-transcribes
    assert log.count(("extract",)) == 2 and log.count(("load",)) == 1


def test_one_bad_file_is_recorded_and_skipped(env):
    _audio(env, "good.m4a")
    _audio(env, "bad.m4a", data=b"junk")
    log = []
    _run(["--phase", "all"], log, fail_files={"bad.m4a"})
    assert ("close",) in log and log.count(("extract",)) == 1
    html = next(Path(os.environ["LMSA_OUTPUT_DIR"]).rglob("voice-memos-*.html")).read_text(encoding="utf-8")
    assert "corrupt audio" in html and "Send the release notes" in html
    _run(["--phase", "transcribe"], log)  # retried next run (nothing stored for it)
    assert log.count(("transcribe", "bad.m4a")) == 2


def test_missing_faster_whisper_is_an_error_per_file_not_a_crash(env):
    _audio(env, "a.m4a")

    def unavailable():
        raise ImportError("faster-whisper isn't installed")

    vm.main(["--phase", "transcribe"], make_transcriber=unavailable, lm=FakeLM([]), now=NOW)
    with Store() as db:
        row = db.conn.execute("SELECT status, summary_json FROM runs WHERE task='tasks.voice_transcribe'").fetchone()
    assert json.loads(row[1])["errors"] == 1


def test_llm_failure_is_isolated_and_not_memoised(env):
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "transcribe"], log)
    _run([], log, lm=FakeLM(log, fail=True))
    _run([], log)
    assert log.count(("extract",)) == 2  # the failed answer wasn't stored


# -- action item extraction schema ---------------------------------------------------

def test_schema_is_strict_and_requires_owner_and_due():
    item = vm.SCHEMA["properties"]["action_items"]["items"]
    assert item["required"] == ["task", "owner", "due"] and item["additionalProperties"] is False
    assert vm.SCHEMA["additionalProperties"] is False
    assert set(vm.SCHEMA["required"]) == set(vm.SCHEMA["properties"])


def test_valid_and_clean_summary():
    assert vm.valid_summary(ANSWER)
    assert not vm.valid_summary({**ANSWER, "action_items": "x"})
    assert not vm.valid_summary({k: v for k, v in ANSWER.items() if k != "decisions"})
    cleaned = vm.clean_summary({**ANSWER, "action_items": [{"task": "  ", "owner": "a", "due": ""},
                                                            {"task": " Do it ", "owner": None, "due": " Fri "}]})
    assert cleaned["action_items"] == [{"task": "Do it", "owner": "", "due": "Fri"}]


def test_actions_jsonl_is_opt_in_and_never_duplicated(env):
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "all"], log)
    path = vm.actions_path()
    assert not path.exists()
    _run(["--actions"], log)
    _run(["--actions"], log)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["task"] == "Send the release notes" and lines[0]["owner"] == "Sam" and lines[0]["due"] == "Thursday"
    assert lines[1]["owner"] == "" and lines[1]["source"] == "voice_memos"
    assert lines[0]["id"].startswith("voice:") and lines[0]["id"] != lines[1]["id"]


def test_long_transcript_is_truncated_and_disclosed(env, monkeypatch):
    monkeypatch.setattr(vm, "MAX_TRANSCRIPT_CHARS", 10)
    _audio(env, "a.m4a")
    log = []
    lm = FakeLM(log)
    _run(["--phase", "all"], log, lm=lm)
    assert "transcript truncated" in lm.prompts[0]
    html = next(Path(os.environ["LMSA_OUTPUT_DIR"]).rglob("voice-memos-*.html")).read_text(encoding="utf-8")
    assert "were cut to" in html


def test_truncate_helper():
    assert vm.truncate("abc", 5) == ("abc", 0)
    text, cut = vm.truncate("x" * 12, 5)
    assert cut == 7 and text.startswith("xxxxx") and "7 more characters omitted" in text


def test_model_output_is_escaped_in_the_report(env):
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "all"], log, lm=FakeLM(log, answer={**ANSWER, "summary": "<script>alert(1)</script>"}))
    html = next(Path(os.environ["LMSA_OUTPUT_DIR"]).rglob("voice-memos-*.html")).read_text(encoding="utf-8")
    assert "<script>alert(1)" not in html and "&lt;script&gt;" in html


def test_voice_transcribe_module_runs_phase_one(monkeypatch):
    seen = []
    monkeypatch.setattr(vm, "main", lambda argv: seen.append(argv))
    voice_transcribe.main(["--days", "3"])
    assert seen == [["--phase", "transcribe", "--days", "3", "--max-mb", "500", "--budget-min", "105"]]
    voice_transcribe.main(["--budget-min", "50", "--reset-attempts"])
    assert seen[-1][-3:] == ["--budget-min", "50", "--reset-attempts"]


# -- lmsa.voice probes degrade gracefully --------------------------------------------

def test_probes_without_faster_whisper(monkeypatch):
    import builtins
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert voice.whisper_installed() is False
    real_import = builtins.__import__

    def no_ct2(name, *a, **kw):
        if name == "ctranslate2":
            raise ImportError("nope")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_ct2)
    assert voice.cuda_devices() is None


def test_env_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_WHISPER_DEVICE", "tpu")
    assert voice.whisper_device() == "auto"
    monkeypatch.setenv("LMSA_WHISPER_MODEL", "medium")
    assert voice.whisper_model() == "medium"
    monkeypatch.setenv("LMSA_RECORDINGS_DIR", str(tmp_path))
    assert voice.recordings_dir() == tmp_path.resolve()


# -- review round 2 ------------------------------------------------------------------

class _FakeWhisperModule:
    """Stands in for `faster_whisper`: CUDA loads fine but the first transcribe
    fails the way a missing cuBLAS DLL does on Windows."""

    def __init__(self, cuda_error="Library cublas64_12.dll is not found or cannot be loaded"):
        self.built: list[str] = []
        self.calls: list[tuple[str, str]] = []
        mod = self

        class WhisperModel:
            def __init__(self, name, device, compute_type):
                mod.built.append(device)
                self.device = device

            def transcribe(self, path, vad_filter=True):
                mod.calls.append((self.device, Path(path).name))
                if self.device == "cuda" and cuda_error:
                    raise RuntimeError(cuda_error)
                seg = SimpleNamespace(start=0.0, end=1.0, text=" hello ")
                return iter([seg]), SimpleNamespace(language="en", duration=1.0)

        self.WhisperModel = WhisperModel


def test_cuda_error_at_first_transcribe_falls_back_to_cpu_for_the_rest(monkeypatch, tmp_path):
    fake = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    t = voice.FasterWhisperTranscriber("small", device="cuda")
    assert t.transcribe(tmp_path / "a.wav")["text"] == "hello"
    assert t.transcribe(tmp_path / "b.wav")["model"].endswith("(cpu)")
    assert fake.built == ["cuda", "cpu"]
    assert fake.calls == [("cuda", "a.wav"), ("cpu", "a.wav"), ("cpu", "b.wav")]
    assert "cublas" in t.fallback and "CPU" in t.fallback


def test_non_cuda_transcribe_error_is_not_a_fallback(monkeypatch, tmp_path):
    fake = _FakeWhisperModule(cuda_error="Invalid data found when processing input")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    t = voice.FasterWhisperTranscriber("small", device="cuda")
    with pytest.raises(RuntimeError, match="Invalid data"):
        t.transcribe(tmp_path / "a.wav")
    assert fake.built == ["cuda"] and t.fallback is None


def test_cuda_fallback_is_disclosed_in_the_report(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", _FakeWhisperModule())
    _audio(env, "a.m4a")
    vm.main(["--phase", "all"], make_transcriber=lambda: voice.FasterWhisperTranscriber("small", device="cuda"),
            lm=FakeLM([]), now=NOW)
    html = _report_html()
    assert "CUDA failed while transcribing" in html and "(cpu)" in html


def _report_html() -> str:
    return next(Path(os.environ["LMSA_OUTPUT_DIR"]).rglob("voice-memos-*.html")).read_text(encoding="utf-8")


def test_over_size_cap_is_skipped_and_reported(env):
    _audio(env, "big.wav", data=b"x" * 3000)
    _audio(env, "small.wav", data=b"x" * 10)
    found = vm.scan(env, days=7, now=NOW, max_mb=0.001)
    assert [r.name for r in found.recordings] == ["small.wav"] and [n for n, _ in found.too_big] == ["big.wav"]
    log = []
    _run(["--phase", "all", "--max-mb", "0.001"], log)
    assert ("transcribe", "big.wav") not in log
    html = _report_html()
    assert "Too big to transcribe" in html and "big.wav" in html


class _Crash(BaseException):
    """Stands in for the process dying mid-file (native crash / timeout kill)."""


def test_a_file_that_kills_the_process_goes_last_then_is_given_up(env):
    _audio(env, "killer.wav", data=b"k", age=timedelta(minutes=10))  # newest
    _audio(env, "older.wav", data=b"o", age=timedelta(hours=5))

    class Killer(FakeTranscriber):
        def transcribe(self, path):
            if Path(path).name == "killer.wav":
                self.log.append(("transcribe", "killer.wav"))
                raise _Crash()
            return super().transcribe(path)

    log = []
    with pytest.raises(_Crash):  # run 1: dies on the newest file; older.wav never reached
        vm.main(["--phase", "transcribe"], make_transcriber=lambda: Killer(log), now=NOW)
    assert ("transcribe", "older.wav") not in log
    log.clear()
    with pytest.raises(_Crash):  # run 2: the never-attempted file goes first
        vm.main(["--phase", "transcribe"], make_transcriber=lambda: Killer(log), now=NOW)
    assert [e for e in log if e[0] == "transcribe"] == [("transcribe", "older.wav"), ("transcribe", "killer.wav")]
    log.clear()
    vm.main(["--phase", "transcribe"], make_transcriber=lambda: Killer(log), now=NOW)  # run 3: given up
    assert not [e for e in log if e[0] == "transcribe"]
    _run([], log)
    html = _report_html()
    assert "no longer retried" in html and "killer.wav" in html and "gave up after 2 failed attempts" in html


def test_max_files_prefers_never_attempted(env, tmp_path):
    for i, name in enumerate(["old_failed.wav", "new1.wav", "new2.wav"]):
        _audio(env, name, data=name.encode(), age=timedelta(hours=i + 1))
    with Store() as db:
        found = vm.scan(env, days=7, now=NOW)
        attempts = vm.attempt_memo(db.conn)
        failed = next(r for r in found.recordings if r.name == "old_failed.wav")
        attempts.put(failed.key, {"attempts": 1})
        log = []
        res = vm.transcribe_all(found.recordings, vm.transcript_memo(db.conn), lambda: FakeTranscriber(log),
                                tmp_path / "t", attempts=attempts, max_files=2)
    assert res.transcribed == ["new1.wav", "new2.wav"] and res.waiting == 1


def test_append_actions_concurrent_writers_never_duplicate(tmp_path):
    path = tmp_path / "actions.jsonl"
    recs = [{"id": f"voice:x:{i}", "task": "t"} for i in range(50)]
    barrier = threading.Barrier(8)
    added = []

    def go():
        barrier.wait()
        added.append(vm.append_actions(path, recs))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == 50 == len(set(ids)) and sum(added) == 50
    assert not (tmp_path / "actions.jsonl.lock").exists()


def test_append_actions_respects_another_process_lock_and_takes_over_a_stale_one(tmp_path):
    path = tmp_path / "actions.jsonl"
    lock = tmp_path / "actions.jsonl.lock"
    lock.write_text("123")  # held by another (live) process
    with pytest.raises(TimeoutError):
        with vm._file_lock(path, timeout=0.2, sleep=lambda s: None):
            pass
    old = datetime.now().timestamp() - vm.LOCK_STALE_S - 5
    os.utime(lock, (old, old))  # its owner died long ago
    assert vm.append_actions(path, [{"id": "a", "task": "t"}]) == 1 and not lock.exists()


def test_append_actions_closes_a_torn_last_line(tmp_path):
    path = tmp_path / "actions.jsonl"
    path.write_text('{"id": "voice:a:0", "task": "x"}\n{"id": "voice:tr', encoding="utf-8")
    assert vm.append_actions(path, [{"id": "voice:b:0", "task": "y"}]) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == '{"id": "voice:tr' and json.loads(lines[2])["id"] == "voice:b:0"


def test_action_ids_are_content_based(env):
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "transcribe"], log)
    _run(["--actions"], log)
    changed = {**ANSWER, "action_items": [{"task": "Order pizza", "owner": "", "due": ""},
                                          {"task": "send  the RELEASE notes", "owner": "sam", "due": "thursday"}]}
    _run(["--actions", "--fresh"], log, lm=FakeLM(log, answer=changed))
    lines = [json.loads(x) for x in vm.actions_path().read_text(encoding="utf-8").splitlines()]
    # the same item re-cased/re-spaced keeps its id; a new task at a used position is still added
    assert [x["task"] for x in lines] == ["Send the release notes", "Book the demo room", "Order pizza"]
    assert vm.action_id("ab" * 32, "Send the release notes") == vm.action_id("ab" * 32, " send  the RELEASE notes ")


def test_actions_flag_is_a_write_flag_for_the_dashboard():
    from lmsa.task_meta import read_meta

    meta = read_meta(Path(vm.__file__))
    assert {a.flag: a.write for a in meta.args}["--actions"] is True


# -- review round 3 ------------------------------------------------------------------

class _Clock:
    """A fake monotonic clock (seconds) that the fakes advance."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _SlowTranscriber(FakeTranscriber):
    """Each file takes `minutes` on `clock`; `kill_at` "dies" mid-file (the
    dispatcher's timeout killing the process); `load_minutes` is spent loading."""

    def __init__(self, log, clock, minutes, kill_at=None, load_minutes=0):
        super().__init__(log)
        self.clock, self.minutes, self.kill_at = clock, minutes, kill_at
        clock.t += load_minutes * 60

    def transcribe(self, path):
        if Path(path).name == self.kill_at:
            self.log.append(("transcribe", Path(path).name))
            raise _Crash()
        self.clock.t += self.minutes * 60
        return super().transcribe(path)


def _key(env, name):
    return next(r for r in vm.scan(env, days=7, now=NOW).recordings if r.name == name).key


def test_time_budget_stops_starting_files_and_discloses_the_rest(env):
    for i in range(4):
        _audio(env, f"f{i}.wav", data=f"f{i}".encode(), age=timedelta(hours=i + 1))
    log, clock = [], _Clock()
    vm.main(["--phase", "all", "--budget-min", "105"], make_transcriber=lambda: _SlowTranscriber(log, clock, 40),
            lm=FakeLM(log), now=NOW, clock=clock)
    # files start at 0, 40 and 80 minutes; at 120 the 105-minute budget is spent
    assert [e[1] for e in log if e[0] == "transcribe"] == ["f0.wav", "f1.wav", "f2.wav"]
    assert "1 file(s) left for the next run" in _report_html()
    with Store() as db:
        assert vm.attempts_of(vm.attempt_memo(db.conn), _key(env, "f3.wav")) == 0  # never started: no strike


def test_a_file_the_timeout_cuts_short_takes_no_strike_and_goes_first(env):
    _audio(env, "b.wav", data=b"b", age=timedelta(hours=1))
    _audio(env, "c.wav", data=b"c", age=timedelta(hours=2))
    for _ in range(3):  # 60 of 105 minutes gone before b.wav starts; the timeout kills it; three runs running
        log, clock = [], _Clock()
        with pytest.raises(_Crash):
            vm.main(["--phase", "transcribe"], now=NOW, clock=clock,
                    make_transcriber=lambda: _SlowTranscriber(log, clock, 1, "b.wav", load_minutes=60))
        assert [e[1] for e in log if e[0] == "transcribe"] == ["b.wav"]  # interrupted: first again
    with Store() as db:
        attempts = vm.attempt_memo(db.conn)
        assert vm.resolve_pending(attempts, _key(env, "b.wav")) is True
        assert vm.attempts_of(attempts, _key(env, "b.wav")) == 0  # never a strike: not given up
    log, clock = [], _Clock()
    vm.main(["--phase", "transcribe"], make_transcriber=lambda: _SlowTranscriber(log, clock, 1), now=NOW, clock=clock)
    assert [e[1] for e in log if e[0] == "transcribe"] == ["b.wav", "c.wav"]


def test_an_unfinished_attempt_with_most_of_the_budget_left_is_a_strike(env):
    _audio(env, "hang.wav", data=b"h")
    with Store() as db:
        attempts, key = vm.attempt_memo(db.conn), _key(env, "hang.wav")
        attempts.put(key, {"attempts": 1, "pending": {"fair": True}})
        assert vm.resolve_pending(attempts, key) is False and vm.attempts_of(attempts, key) == 2
        attempts.put(key, {"attempts": 1, "pending": {"fair": False}})
        assert vm.resolve_pending(attempts, key) is True and vm.attempts_of(attempts, key) == 1


def test_reset_attempts_retries_a_given_up_file(env):
    _audio(env, "bad.wav", data=b"b")
    log = []
    for _ in range(3):
        _run(["--phase", "transcribe"], log, fail_files={"bad.wav"})
    assert [e for e in log if e[0] == "transcribe"] == [("transcribe", "bad.wav")] * 2  # given up after 2
    log.clear()
    _run(["--phase", "transcribe", "--reset-attempts"], log)
    assert ("transcribe", "bad.wav") in log


def test_action_ids_keep_owner_due_and_punctuation_apart(tmp_path):
    sha = "ab" * 32
    for a, b in [("Send $100 to Bob", "Send 100 to Bob"), ("Review C++ code", "Review C code"),
                 ("Call Bob.", "Call Bob")]:
        assert vm.action_id(sha, a) != vm.action_id(sha, b)
    assert vm.action_id(sha, "Review the draft", "Alice") != vm.action_id(sha, "Review the draft", "Bob")
    assert vm.action_id(sha, "Review the draft", "", "Friday") != vm.action_id(sha, "Review the draft", "", "")
    assert (vm.action_id(sha, "Review  the Draft", "ALICE", " friday")
            == vm.action_id(sha, "review the draft", "Alice", "Friday"))
    item = vm.Item(vm.Recording(Path("a.wav"), 1, NOW, sha))
    item.summary = {"action_items": [{"task": "Review the draft", "owner": "Alice", "due": ""},
                                     {"task": "Review the draft", "owner": "Bob", "due": "Friday"}]}
    assert vm.append_actions(tmp_path / "actions.jsonl", vm.action_records([item], NOW)) == 2


def test_stale_lock_takeover_is_won_by_one_taker_only(tmp_path):
    lock = tmp_path / "actions.jsonl.lock"
    lock.write_text("dead")
    old = datetime.now().timestamp() - vm.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    assert vm._take_over_stale(lock) is True and not lock.exists()
    assert vm._take_over_stale(lock) is False  # the second taker finds nothing to take
    # B saw the stale lock, then A replaced it with a live one before B acted:
    # B's rename moves A's live lock, sees it is fresh, and puts it back.
    lock.write_text("A")
    first_look = iter([datetime.now().timestamp() + 10_000])

    def b_clock():  # stale on B's first look only
        return next(first_look, datetime.now().timestamp())

    assert vm._take_over_stale(lock, clock=b_clock) is False
    assert lock.read_text() == "A"
    assert [p.name for p in tmp_path.iterdir()] == ["actions.jsonl.lock"]  # no .stale leftovers


def test_busy_actions_lock_still_writes_the_report(env, monkeypatch):
    monkeypatch.setattr(vm, "LOCK_TIMEOUT_S", 0)
    _audio(env, "a.m4a")
    log = []
    _run(["--phase", "transcribe"], log)
    vm.actions_path().parent.mkdir(parents=True, exist_ok=True)
    vm.actions_path().with_name("actions.jsonl.lock").write_text("123")  # a live run holds it
    _run(["--actions"], log)
    html = _report_html()
    assert "Launch sync" in html and "not appended" in html and "rerun with --actions" in html
    assert not vm.actions_path().exists()


def test_cuda_model_is_freed_before_the_cpu_rebuild(monkeypatch, tmp_path):
    import gc
    import weakref

    live: dict[str, weakref.ref] = {}
    cuda_alive_at_cpu_build = []

    class WhisperModel:  # the segments generator holds the model, like the real one
        def __init__(self, name, device, compute_type):
            self.device = device
            if device == "cpu":
                gc.collect()
                cuda_alive_at_cpu_build.append(live["cuda"]() is not None)
            live[device] = weakref.ref(self)

        def transcribe(self, path, vad_filter=True):
            model = self

            def gen():
                if model.device == "cuda":
                    raise RuntimeError("CUDA failed with error out of memory")
                yield SimpleNamespace(start=0.0, end=1.0, text="hello")

            return gen(), SimpleNamespace(language="en", duration=1.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=WhisperModel))
    t = voice.FasterWhisperTranscriber("small", device="cuda")
    assert t.transcribe(tmp_path / "a.wav")["text"] == "hello" and t.device == "cpu"
    assert cuda_alive_at_cpu_build == [False]


def test_summarize_ignores_the_size_cap_for_files_already_transcribed(env):
    _audio(env, "big.wav", data=b"x" * 3000, age=timedelta(hours=1))
    _audio(env, "untranscribed.wav", data=b"y" * 3000, age=timedelta(hours=2))
    log = []
    vm.main(["--phase", "transcribe", "--max-mb", "1"], make_transcriber=lambda: FakeTranscriber(log), now=NOW)
    _audio(env, "untranscribed.wav", data=b"z" * 3000, age=timedelta(hours=2))  # new content: no transcript
    log.clear()
    _run(["--max-mb", "0.001"], log)  # a lower cap than the transcribe run's
    assert log.count(("extract",)) == 1  # big.wav was summarized
    html = _report_html()
    assert "Too big to transcribe" in html and "untranscribed.wav" in html
    assert "big.wav (0 MB)" not in html
