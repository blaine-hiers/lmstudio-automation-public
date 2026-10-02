"""Thin wrapper over LM Studio's OpenAI-compatible server.

Keeps model defaults, timeouts, and the quirks of local models (Qwen3's
<think> blocks) in one place so tasks only say what they want.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, OpenAI

from .config import Settings, settings

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_image_mime(data: bytes) -> str:
    """Guess an image's mime type from its own magic bytes -- `see()`'s
    caller already decoded/re-encoded these (never trust an untrusted
    attachment's declared Content-Type), so this is just for the data URL.
    Defaults to PNG, the one format `Pillow.save()` never fails to write."""
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    return "image/png"


@dataclass
class ChatResult:
    """A chat reply plus the timing/usage a benchmark needs.

    `chat()` is the common case (just the text); tasks that need latency or
    token counts (e.g. a model shootout) call `chat_with_usage()` instead.
    """

    text: str
    completion_tokens: int
    prompt_tokens: int
    elapsed_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _lms() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    fallback = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    return str(fallback) if fallback.exists() else None


class LMStudio:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings()
        self.api = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout,
            max_retries=2,
        )

    def models(self) -> list[str]:
        return [m.id for m in self.api.models.list().data]

    def ensure_server(self, wait: float = 30) -> None:
        """Start the LM Studio server via the `lms` CLI if it isn't answering.

        Scheduled tasks call this first: after a reboot the app may be closed.
        """
        try:
            self.models()
            return
        except APIConnectionError:
            lms = _lms()
            if not lms:
                raise RuntimeError("LM Studio server is down and the `lms` CLI was not found")
            subprocess.run([lms, "server", "start"], check=True, capture_output=True)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                self.models()
                return
            except APIConnectionError:
                time.sleep(1)
        raise RuntimeError(f"LM Studio server did not come up within {wait:.0f}s")

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat_with_usage(
            prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens
        ).text

    def chat_with_usage(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResult:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        t0 = time.monotonic()
        resp = self.api.chat.completions.create(
            model=model or self.cfg.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_s = time.monotonic() - t0
        usage = resp.usage
        return ChatResult(
            text=strip_think(resp.choices[0].message.content or ""),
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            elapsed_s=elapsed_s,
        )

    def extract(
        self,
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
        model: str | None = None,
        name: str = "result",
    ) -> dict:
        """Structured output: the server constrains generation to `schema`."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        resp = self.api.chat.completions.create(
            model=model or self.cfg.chat_model,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        return json.loads(strip_think(resp.choices[0].message.content or "{}"))

    def see(
        self,
        images: list[bytes],
        prompt: str,
        schema: dict | None = None,
        *,
        system: str | None = None,
        model: str | None = None,
        name: str = "result",
    ) -> dict | str:
        """Vision call: `prompt` plus one or more already-encoded images.

        `images` are raw PNG/JPEG/GIF bytes -- callers downscale/encode
        first (e.g. via Pillow) so the request stays small and fast; this
        method only base64s them into `image_url` data URLs, sniffing each
        one's own mime from its magic bytes. Uses `settings().vision_model`
        unless `model` overrides it. With `schema`, behaves like
        `extract()` (constrained JSON, temperature 0, returns a dict);
        without one, behaves like `chat()` (free text, returns a str).
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_sniff_image_mime(img)};base64,{b64}"},
            })
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content})
        vision_model = model or self.cfg.vision_model

        if schema is not None:
            resp = self.api.chat.completions.create(
                model=vision_model,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                },
            )
            return json.loads(strip_think(resp.choices[0].message.content or "{}"))

        resp = self.api.chat.completions.create(
            model=vision_model, messages=messages, temperature=0.2
        )
        return strip_think(resp.choices[0].message.content or "")

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        resp = self.api.embeddings.create(model=model or self.cfg.embed_model, input=texts)
        return [d.embedding for d in resp.data]
