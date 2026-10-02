"""Probe the LM Studio server end to end and write an Afterhours report.

    python -m tasks.healthcheck
"""

from __future__ import annotations

import time

from lmsa import LMStudio, Report, Store, Tile


def main() -> None:
    lm = LMStudio()
    lm.ensure_server()
    cfg = lm.cfg

    with Store() as store, store.record_run("healthcheck") as run:
        models = lm.models()

        t0 = time.perf_counter()
        reply = lm.chat("Reply with exactly one word: online. /no_think", max_tokens=200)
        chat_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        vec = lm.embed(["healthcheck"])[0]
        embed_s = time.perf_counter() - t0

        ok = "online" in reply.lower()
        r = Report(
            "Server healthcheck",
            kicker="lmsa · diagnostics",
            subtitle=f"End-to-end probe of the LM Studio server at {cfg.base_url}.",
            sticker="all green" if ok else "check chat",
            sticker_tone="acc" if ok else "hot",
            source=cfg.base_url,
        )
        r.tiles([
            Tile("Models loaded", len(models), tone="info"),
            Tile("Chat latency", f"{chat_s:.2f}s", tone="acc" if chat_s < 10 else "warn"),
            Tile("Embed latency", f"{embed_s:.2f}s", tone="violet"),
            Tile("Embed dims", len(vec), tone="muted"),
        ])
        r.section("Models", tone="info").table(
            ["Model", "Role"],
            [
                [
                    m,
                    "chat" if m == cfg.chat_model
                    else "embed" if m == cfg.embed_model
                    else "vision" if m == cfg.vision_model
                    else "",
                ]
                for m in models
            ],
        )
        r.section("Chat probe", tone="acc" if ok else "hot").callout(
            f"{cfg.chat_model} replied: {reply!r}",
            label="pass" if ok else "unexpected reply",
            tone="acc" if ok else "hot",
        )
        path = r.write()
        run.report_path = path
        run.summary = {
            "models": len(models),
            "chat_s": round(chat_s, 2),
            "embed_s": round(embed_s, 2),
            "ok": ok,
        }
        print(path)


if __name__ == "__main__":
    from lmsa.runner import run

    run(main)
