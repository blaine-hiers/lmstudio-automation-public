"""Speech-to-text for voice memos (#62): a small transcriber seam over
faster-whisper, plus cheap probes the dashboard's Setup page uses.

faster-whisper (and CTranslate2 under it) is an optional extra:
`pip install -e ".[voice]"`. Nothing here imports it at module load -- only
`FasterWhisperTranscriber()` does -- so the rest of the repo, and the tests,
never need it. Tests pass their own object with the same two methods:

    class Transcriber(Protocol):
        def transcribe(self, path: Path) -> dict: ...   # see TRANSCRIPT_KEYS
        def close(self) -> None: ...                    # free the model (VRAM)

A transcript is plain JSON: `{"text", "language", "duration_s", "segments":
[{"start", "end", "text"}], "model"}`.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Protocol

AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".ogg", ".webm")
DEFAULT_MODEL = "small"
DEVICES = ("auto", "cuda", "cpu")
TRANSCRIPT_KEYS = ("text", "language", "duration_s", "segments")
INSTALL_COMMAND = r'.venv\Scripts\python -m pip install -e ".[voice]"'


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> dict: ...

    def close(self) -> None: ...


def recordings_dir() -> Path:
    """`LMSA_RECORDINGS_DIR`, default `~/Documents/Recordings`, absolute."""
    raw = os.getenv("LMSA_RECORDINGS_DIR") or str(Path.home() / "Documents" / "Recordings")
    return Path(os.path.abspath(os.path.expanduser(raw)))


def whisper_model() -> str:
    """faster-whisper model size/name (`LMSA_WHISPER_MODEL`, default small)."""
    return (os.getenv("LMSA_WHISPER_MODEL") or DEFAULT_MODEL).strip()


def whisper_device() -> str:
    """`LMSA_WHISPER_DEVICE`: auto (CUDA if present, else CPU) | cuda | cpu."""
    value = (os.getenv("LMSA_WHISPER_DEVICE") or "auto").strip().lower()
    return value if value in DEVICES else "auto"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash (hex) -- the memo key, so a renamed file isn't redone."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Probes (dashboard Setup): cheap, never raise
# ---------------------------------------------------------------------------

def whisper_installed() -> bool:
    """Is faster-whisper importable? (find_spec: nothing heavy is loaded.)"""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def cuda_devices() -> int | None:
    """CUDA devices CTranslate2 can use; None if CTranslate2 isn't installed
    or can't be asked (then nobody can tell without it)."""
    try:
        import ctranslate2  # installed with faster-whisper
    except Exception:
        return None
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# faster-whisper
# ---------------------------------------------------------------------------

_CUDA_MARKERS = ("cuda", "cublas", "cudnn", "cudart", "cufft", "nvrtc", "gpu", "out of memory")


def looks_like_cuda_error(exc: BaseException) -> bool:
    """Does this error come from the GPU stack (a missing cuBLAS/cuDNN DLL,
    which on Windows usually surfaces only at the first transcribe, or
    running out of VRAM) rather than from the audio?"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _CUDA_MARKERS)


class FasterWhisperTranscriber:
    """faster-whisper on CUDA (int8_float16) when available, else CPU (int8).

    `device="auto"` picks CUDA if CTranslate2 sees a device. It falls back to
    CPU if loading on CUDA fails, or if a transcribe fails with a CUDA-looking
    error (missing cuDNN/cuBLAS DLLs usually fail only then): the model is
    rebuilt on CPU, that file is retried, and the rest of the run stays on
    CPU. `fallback` says why, for the report.
    Call `close()` when done so the VRAM is free before a chat model loads.
    Raises ImportError (with the install command) if the extra isn't there."""

    def __init__(self, model: str | None = None, device: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(f"faster-whisper isn't installed: {INSTALL_COMMAND}") from exc
        self.model_name = model or whisper_model()
        want = device or whisper_device()
        if want == "auto":
            want = "cuda" if (cuda_devices() or 0) > 0 else "cpu"
        self.fallback: str | None = None
        self._whisper_model = WhisperModel
        try:
            self._model = WhisperModel(self.model_name, device=want,
                                       compute_type="int8_float16" if want == "cuda" else "int8")
            self.device = want
        except Exception as exc:
            if want != "cuda":
                raise
            self._to_cpu(f"CUDA load failed ({type(exc).__name__}: {str(exc)[:160]})")

    def _to_cpu(self, why: str) -> None:
        self._model = None
        gc.collect()
        self._model = self._whisper_model(self.model_name, device="cpu", compute_type="int8")
        self.device = "cpu"
        self.fallback = f"{why}; transcribed on the CPU instead (slower)"

    def transcribe(self, path: Path) -> dict:
        try:
            return self._transcribe(path)
        except Exception as exc:
            if self.device != "cuda" or not looks_like_cuda_error(exc):
                raise
            why = f"CUDA failed while transcribing ({type(exc).__name__}: {str(exc)[:160]})"
        # Rebuild on the CPU only after leaving the except block: until then the
        # traceback's frames (and a half-run segments generator) keep the CUDA
        # model -- and its VRAM -- alive.
        self._to_cpu(why)
        return self._transcribe(path)  # the same file, once, on the CPU

    def _transcribe(self, path: Path) -> dict:
        segments, info = self._model.transcribe(str(path), vad_filter=True)
        segs = [{"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip()}
                for s in segments]  # a generator: decoding happens here
        return {
            "text": " ".join(s["text"] for s in segs if s["text"]),
            "language": str(getattr(info, "language", "") or ""),
            "duration_s": round(float(getattr(info, "duration", 0.0) or 0.0), 1),
            "segments": segs,
            "model": f"faster-whisper {self.model_name} ({self.device})",
        }

    def close(self) -> None:
        self._model = None
        gc.collect()  # CTranslate2 frees its VRAM when the model object goes
