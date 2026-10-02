"""Run a fixed prompt set against every loaded chat model and report the results.

    python -m tasks.shootout
    python -m tasks.shootout --limit 2 --max-tokens 200 --no-judge

Prompts live in prompts/shootout/*.md: a small front-matter block (id,
category, rubric) followed by the prompt text. Every loaded model that
doesn't look like an embedding model runs every prompt; the optional judge
pass asks LMSA_CHAT_MODEL to score each answer 1-5 against that prompt's
rubric, via a JSON-schema `extract()` call so the score is never
regex-parsed out of free text.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from lmsa import Card, LMStudio, Report, Store, Tile
from lmsa.prompts import load as load_prompt

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "shootout"

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?\n)---\s*\n?(.*)\Z", re.DOTALL)

JUDGE_SYSTEM = load_prompt("shootout_judge/system")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class Prompt:
    id: str
    category: str
    rubric: str
    text: str


def load_prompts(directory: Path = PROMPTS_DIR) -> list[Prompt]:
    """Parse every prompts/shootout/*.md file into a Prompt, sorted by filename."""
    prompts = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = _FRONT_MATTER.match(raw)
        if not m:
            raise ValueError(f"{path} is missing a leading --- front matter block")
        meta: dict[str, str] = {}
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        prompts.append(
            Prompt(
                id=meta.get("id", path.stem),
                category=meta.get("category", ""),
                rubric=meta.get("rubric", ""),
                text=m.group(2).strip(),
            )
        )
    return prompts


def is_embedding_model(model_id: str) -> bool:
    return "embed" in model_id.lower()


@dataclass
class Attempt:
    model: str
    prompt_id: str
    text: str
    completion_tokens: int
    elapsed_s: float
    score: int | None = None
    reasoning: str = ""
    error: str | None = None
    judge_error: str | None = None

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def judge(lm: LMStudio, prompt: Prompt, answer: str, *, judge_model: str) -> tuple[int, str]:
    result = lm.extract(
        f"Rubric: {prompt.rubric}\n\nAnswer to score:\n{answer}",
        JUDGE_SCHEMA,
        system=JUDGE_SYSTEM,
        model=judge_model,
        name="score",
    )
    return int(result["score"]), str(result.get("reasoning", ""))


def run(
    lm: LMStudio,
    prompts: list[Prompt],
    models: list[str],
    *,
    max_tokens: int,
    judge_pass: bool,
    judge_model: str,
) -> list[Attempt]:
    """Run every (prompt, model) pair, isolating failures per attempt.

    A model that can't load, times out, or errors mid-run loses only its own
    attempt — every other (prompt, model) pair still completes and the run
    still produces a report. Same for the judge pass: a judge failure (bad
    server response, or a reply missing/mangling the `score` field) leaves
    the model's answer intact and just records no score.
    """
    attempts = []
    for prompt in prompts:
        for model in models:
            try:
                result = lm.chat_with_usage(prompt.text, model=model, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 - one bad model must not sink the run
                attempts.append(
                    Attempt(
                        model=model,
                        prompt_id=prompt.id,
                        text="",
                        completion_tokens=0,
                        elapsed_s=0.0,
                        error=_describe(exc),
                    )
                )
                continue

            attempt = Attempt(
                model=model,
                prompt_id=prompt.id,
                text=result.text,
                completion_tokens=result.completion_tokens,
                elapsed_s=result.elapsed_s,
            )
            if judge_pass and prompt.rubric:
                try:
                    attempt.score, attempt.reasoning = judge(
                        lm, prompt, result.text, judge_model=judge_model
                    )
                except Exception as exc:  # noqa: BLE001 - a bad judge reply must not lose the answer
                    attempt.judge_error = _describe(exc)
            attempts.append(attempt)
    return attempts


def _model_stats(attempts: list[Attempt], model: str) -> tuple[float, float | None, int]:
    """Average tok/s and score for `model`, excluding failed chat calls; plus its failure count."""
    rows = [a for a in attempts if a.model == model]
    ok = [a for a in rows if a.error is None]
    avg_tok_s = sum(a.tokens_per_s for a in ok) / len(ok) if ok else 0.0
    scored = [a.score for a in rows if a.score is not None]
    avg_score = sum(scored) / len(scored) if scored else None
    return avg_tok_s, avg_score, len(rows) - len(ok)


def build_report(attempts: list[Attempt], models: list[str], *, judge_pass: bool) -> Report:
    r = Report(
        "Model shootout",
        kicker="lmsa · benchmarks",
        subtitle=f"{len(models)} model(s) x each prompt in prompts/shootout/.",
        source="tasks.shootout",
    )

    tiles = []
    for model in models:
        avg_tok_s, avg_score, failed = _model_stats(attempts, model)
        total = len([a for a in attempts if a.model == model])
        if total and failed == total:
            tiles.append(Tile(model, "no data", tone="hot", note=f"{failed} failed"))
            continue
        notes = []
        if avg_score is not None:
            notes.append(f"avg score {avg_score:.1f}/5")
        if failed:
            notes.append(f"{failed} failed")
        tiles.append(
            Tile(
                model,
                f"{avg_tok_s:.1f} tok/s",
                tone="hot" if failed else "acc",
                note=" · ".join(notes) or None,
            )
        )
    r.tiles(tiles)

    r.section("Speed", tone="info").bars(
        [(model, _model_stats(attempts, model)[0]) for model in models],
        unit=" tok/s",
    )

    for prompt_id in dict.fromkeys(a.prompt_id for a in attempts):
        cards = []
        for a in (a for a in attempts if a.prompt_id == prompt_id):
            failed = a.error is not None
            if failed:
                meta = "failed"
                body = a.error or ""
            else:
                meta = f"{a.tokens_per_s:.1f} tok/s · {a.elapsed_s:.2f}s"
                if a.score is not None:
                    meta += f" · score {a.score}/5"
                elif a.judge_error:
                    meta += " · judge error"
                body = a.text
            has_error = failed or a.judge_error is not None
            cards.append(
                Card(
                    a.model,
                    meta=meta,
                    body=body,
                    tags=["error"] if has_error else (),
                    tone="hot" if has_error else "acc",
                )
            )
        r.section(prompt_id).cards(cards)

    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N prompts")
    parser.add_argument("--max-tokens", type=int, default=400, help="cap completion length per call")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge pass")
    args = parser.parse_args()

    lm = LMStudio()
    lm.ensure_server()

    prompts = load_prompts()
    if args.limit is not None:
        prompts = prompts[: args.limit]

    with Store() as store, store.record_run("tasks.shootout") as handle:
        models = [m for m in lm.models() if not is_embedding_model(m)]

        attempts = run(
            lm,
            prompts,
            models,
            max_tokens=args.max_tokens,
            judge_pass=not args.no_judge,
            judge_model=lm.cfg.chat_model,
        )

        path = build_report(attempts, models, judge_pass=not args.no_judge).write()
        handle.report_path = path
        handle.summary = {"prompts": len(prompts), "models": len(models), "attempts": len(attempts)}
    print(path)


if __name__ == "__main__":
    # Aliased: this module has its own `run()` (the shootout loop).
    from lmsa.runner import run as run_task

    run_task(main)
