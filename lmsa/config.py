"""Settings, read from the environment (and a local .env if one exists)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# tasks.dashboard's default --port; reports that link into the dashboard
# (e.g. a story card's "go deep" link, #70) build absolute URLs from it.
DASHBOARD_PORT = 8765


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    vision_model: str
    output_dir: Path
    timeout: float
    # Fields with defaults stay last (dataclass ordering).
    imap_host: str
    imap_port: int
    imap_user: str
    slack_webhook: str | None = None
    # lmsa.mailer (#101): same Gmail app password as imap_* above, over
    # SMTP+STARTTLS instead of IMAP. No separate secret.
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    searxng_url: str = "http://127.0.0.1:8888"
    # #89: optional command that brings a self-hosted SearXNG up on demand
    # (e.g. `wsl.exe -d Ubuntu -u root --exec sleep 7200`); unset keeps the
    # old always-assumed-running behaviour.
    searxng_start: str | None = None
    searxng_start_timeout: float = 60


_DEFAULT_SEARXNG_START_TIMEOUT = 60.0


def _searxng_start_timeout(raw: str | None) -> float:
    """A finite, non-negative number of seconds from ``raw`` -- missing,
    non-numeric, negative, NaN or infinite all fall back to the default
    (60s) instead of hanging a readiness poll forever (NaN/inf) or crashing
    every ``settings()`` call (a non-numeric string used to raise ValueError
    here uncaught)."""
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = float("nan")
        if math.isfinite(value) and value >= 0:
            return value
    return _DEFAULT_SEARXNG_START_TIMEOUT


def settings() -> Settings:
    return Settings(
        base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        # LM Studio ignores the key unless auth is turned on; the client just needs a value.
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        chat_model=os.getenv("LMSA_CHAT_MODEL", "qwen/qwen3-14b"),
        embed_model=os.getenv("LMSA_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        vision_model=os.getenv("LMSA_VISION_MODEL", "qwen/qwen3-vl-8b"),
        output_dir=Path(os.getenv("LMSA_OUTPUT_DIR", str(ROOT / "output"))),
        timeout=float(os.getenv("LMSA_TIMEOUT", "300")),
        slack_webhook=os.getenv("LMSA_SLACK_WEBHOOK") or None,
        # IMAP password is never kept here — see lmsa.secrets.resolve("imap").
        imap_host=os.getenv("LMSA_IMAP_HOST", "imap.gmail.com"),
        imap_port=int(os.getenv("LMSA_IMAP_PORT", "993")),
        imap_user=os.getenv("LMSA_IMAP_USER", ""),
        smtp_host=os.getenv("LMSA_SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("LMSA_SMTP_PORT", "587")),
        searxng_url=os.getenv("LMSA_SEARXNG_URL", "http://127.0.0.1:8888"),
        searxng_start=os.getenv("LMSA_SEARXNG_START") or None,
        searxng_start_timeout=_searxng_start_timeout(os.getenv("LMSA_SEARXNG_START_TIMEOUT")),
    )
