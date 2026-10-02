"""Is the PC quiet enough to run automations? `busy_reasons()` answers.

    reasons = busy_reasons(GateConfig(...), need_vram_mb=8200)
    if not reasons: ...  # quiet: go ahead

Each signal lives behind its own small function and is fail-safe: a signal
that can't be read counts as BUSY and names why, so a broken probe can only
make the dispatcher more cautious, never less.

Signals (the owner's definition of busy -- ANY of them):
  - fullscreen: `SHQueryUserNotificationState` reports a D3D fullscreen app,
    "busy" (a fullscreen app/game), or presentation mode.
  - GPU: `nvidia-smi` utilization over the threshold, or free VRAM below the
    VRAM the next model needs plus headroom.
  - input: `GetLastInputInfo` shows fewer than N minutes since the last
    keyboard/mouse input (N = 0 disables this check).
  - processes: a process from the configured denylist is running (matched on
    the image name, case-insensitive, `.exe` optional). An empty denylist
    never blocks and never even lists processes.

The `_read_*` functions touch the OS; tests monkeypatch them.
"""

from __future__ import annotations

import csv
import ctypes
import io
import subprocess
import sys
from dataclasses import dataclass, field

NVIDIA_SMI_TIMEOUT_S = 10
TASKLIST_TIMEOUT_S = 15

# SHQueryUserNotificationState values (shellapi.h).
QUNS_NOT_PRESENT = 1
QUNS_BUSY = 2
QUNS_RUNNING_D3D_FULL_SCREEN = 3
QUNS_PRESENTATION_MODE = 4
QUNS_ACCEPTS_NOTIFICATIONS = 5
QUNS_QUIET_TIME = 6
QUNS_APP = 7

_BUSY_STATES = {
    QUNS_BUSY: "a fullscreen app is running",
    QUNS_RUNNING_D3D_FULL_SCREEN: "a fullscreen game (D3D) is running",
    QUNS_PRESENTATION_MODE: "presentation mode is on",
}


@dataclass(frozen=True)
class GateConfig:
    gpu_util_max: int = 30  # percent
    vram_headroom_mb: int = 1024
    idle_minutes: float = 15  # 0 disables the input-idle check
    blocked_processes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GpuState:
    util_pct: int
    free_mb: int
    total_mb: int


# ---------------------------------------------------------------------------
# OS readers (monkeypatched in tests)
# ---------------------------------------------------------------------------

def _read_notification_state() -> int:
    state = ctypes.c_int(0)
    hr = ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state))
    if hr != 0:
        raise OSError(f"SHQueryUserNotificationState failed (HRESULT {hr & 0xFFFFFFFF:#010x})")
    return state.value


def _read_gpu() -> GpuState:
    """First GPU's utilization and free/total VRAM, via `nvidia-smi`."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    return parse_nvidia_smi(out)


def parse_nvidia_smi(out: str) -> GpuState:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("nvidia-smi printed nothing")
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 3:
        raise ValueError(f"unexpected nvidia-smi line: {lines[0][:80]!r}")
    util, free, total = (int(p) for p in parts)  # "[N/A]" raises -> unreadable
    return GpuState(util_pct=util, free_mb=free, total_mb=total)


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def _read_idle_seconds() -> float:
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        raise OSError("GetLastInputInfo failed")
    now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
    return ((now - info.dwTime) & 0xFFFFFFFF) / 1000.0  # both wrap at 2**32 ms


def _read_process_names() -> list[str]:
    out = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        timeout=TASKLIST_TIMEOUT_S,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
    return [row[0] for row in csv.reader(io.StringIO(out)) if row]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _unreadable(signal: str, exc: BaseException) -> list[str]:
    return [f"{signal}: can't read ({type(exc).__name__}: {str(exc)[:120]}) - treating as busy"]


def fullscreen_reasons() -> list[str]:
    try:
        state = _read_notification_state()
    except Exception as exc:
        return _unreadable("fullscreen", exc)
    reason = _BUSY_STATES.get(state)
    return [f"fullscreen: {reason}"] if reason else []


def gpu_reasons(cfg: GateConfig, need_vram_mb: int = 0) -> list[str]:
    try:
        gpu = _read_gpu()
    except Exception as exc:
        return _unreadable("gpu", exc)
    reasons = []
    if gpu.util_pct > cfg.gpu_util_max:
        reasons.append(f"gpu: {gpu.util_pct}% utilization (limit {cfg.gpu_util_max}%)")
    if need_vram_mb > 0:
        want = need_vram_mb + cfg.vram_headroom_mb
        if gpu.free_mb < want:
            reasons.append(
                f"gpu: {gpu.free_mb} MiB VRAM free, need {want} MiB "
                f"({need_vram_mb} model + {cfg.vram_headroom_mb} headroom)"
            )
    return reasons


def idle_reasons(idle_minutes: float) -> list[str]:
    if idle_minutes <= 0:
        return []
    try:
        idle_s = _read_idle_seconds()
    except Exception as exc:
        return _unreadable("input", exc)
    if idle_s < idle_minutes * 60:
        return [f"input: last keyboard/mouse input {idle_s / 60:.1f} min ago (need {idle_minutes:g} min idle)"]
    return []


def normalize_process_name(name: str) -> str:
    name = name.strip().lower()
    return name[:-4] if name.endswith(".exe") else name


def process_reasons(blocked: tuple[str, ...] | list[str]) -> list[str]:
    wanted = {normalize_process_name(b) for b in blocked if b and b.strip()}
    if not wanted:
        return []  # empty denylist: never blocks, never lists processes
    try:
        running = {normalize_process_name(n) for n in _read_process_names()}
    except Exception as exc:
        return _unreadable("processes", exc)
    hits = sorted(wanted & running)
    return [f"process: {name} is running" for name in hits]


def busy_reasons(
    cfg: GateConfig,
    *,
    need_vram_mb: int = 0,
    idle_minutes: float | None = None,
) -> list[str]:
    """Every reason the PC is busy right now; an empty list means quiet.

    `need_vram_mb` is the size of the model about to be loaded (0 when none
    is, or it's already resident). `idle_minutes` overrides the configured
    input-idle threshold (the dispatcher relaxes it overnight).
    """
    idle = cfg.idle_minutes if idle_minutes is None else idle_minutes
    reasons: list[str] = []
    reasons += fullscreen_reasons()
    reasons += gpu_reasons(cfg, need_vram_mb)
    reasons += idle_reasons(idle)
    reasons += process_reasons(cfg.blocked_processes)
    return reasons


if __name__ == "__main__":  # quick manual probe: python -m lmsa.gate
    found = busy_reasons(GateConfig())
    print("\n".join(found) if found else "quiet")
    sys.exit(1 if found else 0)
