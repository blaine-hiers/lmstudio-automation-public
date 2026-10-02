import pytest

from lmsa import ChatResult
from tasks.shootout import (
    PROMPTS_DIR,
    Attempt,
    build_report,
    is_embedding_model,
    judge,
    load_prompts,
    run,
)


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no server, no network.

    `chat_errors`/`extract_errors` map a model id to an exception instance to
    raise instead of answering, for exercising run()'s per-attempt isolation.
    `extract_missing_score` marks models whose judge reply omits the `score`
    key (as a real malformed judge response would).
    """

    def __init__(
        self,
        replies: dict[str, str],
        scores: dict[str, tuple[int, str]] | None = None,
        chat_errors: dict[str, Exception] | None = None,
        extract_errors: dict[str, Exception] | None = None,
        extract_missing_score: set[str] | None = None,
    ):
        self.replies = replies
        self.scores = scores or {}
        self.chat_errors = chat_errors or {}
        self.extract_errors = extract_errors or {}
        self.extract_missing_score = extract_missing_score or set()
        self.chat_calls: list[tuple[str, str]] = []
        self.extract_calls: list[tuple[str, str]] = []

    def chat_with_usage(self, prompt, *, model=None, max_tokens=None, **_):
        self.chat_calls.append((model, prompt))
        if model in self.chat_errors:
            raise self.chat_errors[model]
        text = self.replies.get(model, "default reply")
        return ChatResult(text=text, completion_tokens=20, prompt_tokens=10, elapsed_s=2.0)

    def extract(self, prompt, schema, *, model=None, name="result", **_):
        self.extract_calls.append((model, prompt))
        if model in self.extract_errors:
            raise self.extract_errors[model]
        if model in self.extract_missing_score:
            return {"reasoning": "no score field in this reply"}
        score, reasoning = self.scores.get(model, (4, "solid answer"))
        return {"score": score, "reasoning": reasoning}


def test_load_prompts_parses_front_matter(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\nid: greet\ncategory: chit-chat\nrubric: 5 if friendly\n---\nSay hi.\n",
        encoding="utf-8",
    )
    prompts = load_prompts(tmp_path)
    assert len(prompts) == 1
    p = prompts[0]
    assert p.id == "greet"
    assert p.category == "chit-chat"
    assert p.rubric == "5 if friendly"
    assert p.text == "Say hi."


def test_load_prompts_missing_front_matter_raises(tmp_path):
    (tmp_path / "bad.md").write_text("Just a prompt, no front matter.", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(tmp_path)


def test_load_prompts_defaults_id_to_filename(tmp_path):
    (tmp_path / "no-id.md").write_text("---\ncategory: x\n---\nBody.", encoding="utf-8")
    prompts = load_prompts(tmp_path)
    assert prompts[0].id == "no-id"
    assert prompts[0].rubric == ""


def test_real_prompt_files_parse_and_have_rubrics():
    prompts = load_prompts(PROMPTS_DIR)
    assert len(prompts) == 5
    ids = {p.id for p in prompts}
    assert len(ids) == 5, "prompt ids must be unique"
    for p in prompts:
        assert p.rubric, f"{p.id} is missing a rubric"
        assert p.text, f"{p.id} has no prompt body"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("text-embedding-nomic-embed-text-v1.5", True),
        ("nomic-embed-text-v1.5", True),
        ("qwen/qwen3-14b", False),
        ("qwen/qwen3-vl-8b", False),
    ],
)
def test_is_embedding_model(model_id, expected):
    assert is_embedding_model(model_id) is expected


def test_run_calls_chat_for_every_model_and_prompt():
    fake = FakeLMStudio(replies={"m1": "reply one", "m2": "reply two"})
    prompts = load_prompts(PROMPTS_DIR)[:2]
    attempts = run(fake, prompts, ["m1", "m2"], max_tokens=50, judge_pass=False, judge_model="m1")

    assert len(attempts) == 4
    assert len(fake.chat_calls) == 4
    assert fake.extract_calls == []
    assert all(a.score is None for a in attempts)
    assert all(a.completion_tokens == 20 and a.elapsed_s == 2.0 for a in attempts)


def test_run_adds_judge_scores_when_enabled():
    fake = FakeLMStudio(
        replies={"m1": "reply one"},
        scores={"judge-model": (5, "excellent")},
    )
    prompts = load_prompts(PROMPTS_DIR)[:1]
    attempts = run(
        fake, prompts, ["m1"], max_tokens=50, judge_pass=True, judge_model="judge-model"
    )

    assert len(attempts) == 1
    assert attempts[0].score == 5
    assert attempts[0].reasoning == "excellent"
    assert fake.extract_calls[0][0] == "judge-model"


def test_judge_returns_score_and_reasoning():
    fake = FakeLMStudio(replies={}, scores={"judge-model": (2, "weak")})
    prompt = load_prompts(PROMPTS_DIR)[0]
    score, reasoning = judge(fake, prompt, "some answer", judge_model="judge-model")
    assert score == 2
    assert reasoning == "weak"


def test_tokens_per_s_guards_zero_elapsed():
    a = Attempt(model="m", prompt_id="p", text="x", completion_tokens=10, elapsed_s=0.0)
    assert a.tokens_per_s == 0.0


def test_build_report_escapes_model_output_and_lists_models():
    attempts = [
        Attempt(
            model="m1",
            prompt_id="extract-order",
            text="<script>alert(1)</script>",
            completion_tokens=20,
            elapsed_s=2.0,
            score=4,
            reasoning="good",
        ),
    ]
    report = build_report(attempts, ["m1"], judge_pass=True)
    html = report.render()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "m1" in html
    assert "10.0 tok/s" in html
    assert "avg score 4.0/5" in html
    assert "score 4/5" in html


def test_build_report_handles_no_attempts():
    report = build_report([], [], judge_pass=False)
    html = report.render()
    assert "Model shootout" in html


def test_run_isolates_a_model_that_raises_mid_run():
    fake = FakeLMStudio(
        replies={"m1": "reply one"},
        chat_errors={"m2": RuntimeError("model failed to load")},
    )
    prompts = load_prompts(PROMPTS_DIR)[:1]
    attempts = run(fake, prompts, ["m1", "m2"], max_tokens=50, judge_pass=False, judge_model="m1")

    # both attempts are still recorded - the failing model didn't sink the run
    assert len(attempts) == 2
    by_model = {a.model: a for a in attempts}
    assert by_model["m1"].error is None
    assert by_model["m1"].text == "reply one"
    assert by_model["m2"].error == "RuntimeError: model failed to load"
    assert by_model["m2"].text == ""
    assert by_model["m2"].completion_tokens == 0
    assert by_model["m2"].elapsed_s == 0.0


def test_run_isolates_a_judge_failure_without_losing_the_answer():
    fake = FakeLMStudio(
        replies={"m1": "reply one"},
        extract_errors={"judge-model": RuntimeError("judge server error")},
    )
    prompts = load_prompts(PROMPTS_DIR)[:1]
    attempts = run(
        fake, prompts, ["m1"], max_tokens=50, judge_pass=True, judge_model="judge-model"
    )

    assert len(attempts) == 1
    a = attempts[0]
    assert a.error is None
    assert a.text == "reply one"  # the answer survives even though judging failed
    assert a.score is None
    assert a.judge_error == "RuntimeError: judge server error"


def test_run_records_judge_error_when_score_is_missing():
    fake = FakeLMStudio(
        replies={"m1": "reply one"},
        extract_missing_score={"judge-model"},
    )
    prompts = load_prompts(PROMPTS_DIR)[:1]
    attempts = run(
        fake, prompts, ["m1"], max_tokens=50, judge_pass=True, judge_model="judge-model"
    )

    a = attempts[0]
    assert a.text == "reply one"
    assert a.score is None
    assert a.judge_error is not None
    assert "KeyError" in a.judge_error


def test_build_report_shows_chat_failures_as_hot_cards_tagged_error():
    attempts = [
        Attempt(
            model="m1",
            prompt_id="p1",
            text="",
            completion_tokens=0,
            elapsed_s=0.0,
            error="RuntimeError: boom",
        ),
    ]
    report = build_report(attempts, ["m1"], judge_pass=False)
    html = report.render()

    assert "boom" in html
    assert '<span class="tag">error</span>' in html
    assert 't-hot' in html


def test_build_report_shows_judge_failures_as_hot_cards_tagged_error():
    attempts = [
        Attempt(
            model="m1",
            prompt_id="p1",
            text="a real answer",
            completion_tokens=20,
            elapsed_s=2.0,
            judge_error="RuntimeError: judge exploded",
        ),
    ]
    report = build_report(attempts, ["m1"], judge_pass=True)
    html = report.render()

    assert "a real answer" in html  # the answer is still shown, just unscored
    assert "judge error" in html
    assert '<span class="tag">error</span>' in html


def test_build_report_tile_reports_when_every_attempt_for_a_model_failed():
    attempts = [
        Attempt(model="m1", prompt_id="p1", text="", completion_tokens=0, elapsed_s=0.0, error="boom"),
        Attempt(model="m1", prompt_id="p2", text="", completion_tokens=0, elapsed_s=0.0, error="boom again"),
    ]
    report = build_report(attempts, ["m1"], judge_pass=False)
    html = report.render()

    assert "no data" in html
    assert "2 failed" in html


def test_build_report_excludes_failed_attempts_from_model_averages():
    attempts = [
        Attempt(model="m1", prompt_id="p1", text="ok", completion_tokens=20, elapsed_s=2.0),
        Attempt(model="m1", prompt_id="p2", text="", completion_tokens=0, elapsed_s=0.0, error="boom"),
    ]
    report = build_report(attempts, ["m1"], judge_pass=False)
    html = report.render()

    # average tok/s comes from the one successful attempt (10 tok/s), not diluted by the failure
    assert "10.0 tok/s" in html
    assert "1 failed" in html
