"""Explicit model lifecycle through the LM Studio `lms` CLI.

    mm = ModelManager(Lms(), free_vram_mb=..., headroom_mb=1024, ttl_s=900)
    mm.ensure("qwen/qwen3-14b", size_mb=8200)   # load (or reuse) before a batch
    ...
    mm.release_all()                              # unload what WE loaded

Rules:
  - Only models this manager loaded are ever unloaded. A model the owner
    loaded by hand (or that was already resident) is reused if it's the one
    needed, and otherwise left strictly alone. A model that appears WHILE a
    dispatched task runs (LM Studio JIT-loading it for that task) is adopted
    via `snapshot()` / `adopt_new()` and unloaded with ours.
  - One dispatcher-loaded model at a time: loading the next one unloads the
    previous one we loaded first.
  - Refuse to load (ModelError) when free VRAM is below size + headroom.
  - Every load carries a TTL, so a dispatcher that crashes mid-batch leaves
    nothing pinned: LM Studio unloads it after `ttl_s` idle seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence

LMS_TIMEOUT_S = 300  # a cold load of a 14B model can take a while


class ModelError(RuntimeError):
    """A model couldn't be loaded (or the `lms` CLI failed)."""


def default_lms_path() -> str:
    exe = Path(os.path.expanduser("~")) / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms")
    return str(exe) if exe.exists() else "lms"


Runner = Callable[..., subprocess.CompletedProcess]


class Lms:
    """Thin wrapper over the `lms` CLI. `runner` is injectable for tests."""

    def __init__(self, exe: str | None = None, runner: Runner = subprocess.run, timeout: float = LMS_TIMEOUT_S):
        self.exe = exe or default_lms_path()
        self.runner = runner
        self.timeout = timeout

    def _run(self, args: Sequence[str]) -> str:
        try:
            proc = self.runner(
                [self.exe, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ModelError(f"lms {' '.join(args)}: {type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-300:]
            raise ModelError(f"lms {' '.join(args)} exited {proc.returncode}: {detail}")
        return proc.stdout or ""

    def _json_list(self, args: Sequence[str]) -> list[dict]:
        out = self._run(args)
        try:
            data = json.loads(out or "[]")
        except json.JSONDecodeError as exc:
            raise ModelError(f"lms {' '.join(args)}: unreadable output ({exc})") from exc
        if not isinstance(data, list):
            raise ModelError(f"lms {' '.join(args)}: expected a list")
        return [m for m in data if isinstance(m, dict)]

    def instances(self) -> list[dict]:
        """Every loaded model instance as `lms ps --json` reports it
        (`identifier`, `modelKey`, `ttlMs` -- null when it has no TTL, ...)."""
        return self._json_list(["ps", "--json"])

    def loaded(self) -> list[str]:
        """Identifiers of every model currently loaded (`lms ps --json`)."""
        return [str(m.get("identifier") or m.get("modelKey")) for m in self.instances()]

    def downloaded(self) -> list[dict]:
        """Every model on disk as `lms ls --json` reports it (`modelKey`,
        `path`, `variants`, ...)."""
        return self._json_list(["ls", "--json"])

    def load(self, model: str, ttl_s: int) -> None:
        self._run(["load", model, "--ttl", str(int(ttl_s)), "-y"])

    def unload(self, model: str) -> None:
        self._run(["unload", model])

    def server_stop(self) -> None:
        self._run(["server", "stop"])


class ModelManager:
    def __init__(
        self,
        lms: Lms,
        *,
        free_vram_mb: Callable[[], int],
        headroom_mb: int = 1024,
        ttl_s: int = 900,
    ):
        self.lms = lms
        self.free_vram_mb = free_vram_mb
        self.headroom_mb = headroom_mb
        self.ttl_s = ttl_s
        self.ours: list[str] = []  # models this manager loaded, oldest first

    def is_resident(self, model: str) -> bool:
        return model in self.lms.loaded()

    def ensure(self, model: str, size_mb: int) -> bool:
        """Make `model` resident. Returns True if we loaded it, False if it
        was already loaded (by us or by hand). Raises ModelError on refusal."""
        if self.is_resident(model):
            return False
        # One of ours at a time: free our previous model before the next.
        for other in list(self.ours):
            self.release(other)
        try:
            free = int(self.free_vram_mb())
        except Exception as exc:
            raise ModelError(f"can't read free VRAM before loading {model}: {exc}") from exc
        want = size_mb + self.headroom_mb
        if free < want:
            raise ModelError(
                f"not loading {model}: {free} MiB VRAM free, need {want} MiB "
                f"({size_mb} model + {self.headroom_mb} headroom)"
            )
        self.lms.load(model, self.ttl_s)
        self.ours.append(model)
        return True

    def snapshot(self) -> set[str]:
        """What's loaded right now; pass to `adopt_new` after a task runs."""
        return set(self.lms.loaded())

    def adopt_new(self, before: set[str]) -> list[str]:
        """Claim every model that appeared since `before` -- a JIT load the
        task triggered (a second chat model, an embedding model) -- so it is
        unloaded with the rest of ours. Models loaded before stay untouched."""
        new = [m for m in self.lms.loaded() if m not in before and m not in self.ours]
        self.ours.extend(new)
        return new

    def release(self, model: str) -> None:
        """Unload `model` only if this manager loaded it."""
        if model not in self.ours:
            return
        self.ours.remove(model)
        if model in self.lms.loaded():  # its TTL may already have expired
            self.lms.unload(model)

    def release_all(self) -> list[str]:
        """Unload every model we loaded; returns errors instead of raising."""
        errors = []
        for model in list(self.ours):
            try:
                self.release(model)
            except Exception as exc:
                errors.append(f"unload {model}: {exc}")
        return errors

    def stop_server(self) -> None:
        """`lms server stop`, but only when nothing the owner loaded is resident."""
        if self.lms.loaded():
            raise ModelError("not stopping the server: models are still loaded")
        self.lms.server_stop()
