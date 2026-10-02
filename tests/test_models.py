"""lmsa.models against a fake `lms` -- never the real CLI."""

import json
import subprocess

import pytest

from lmsa.models import Lms, ModelError, ModelManager

CHAT = "qwen/qwen3-14b"
VISION = "qwen/qwen3-vl-8b"


class FakeLms:
    """A fake `lms` executable: tracks loaded models, records every call."""

    def __init__(self, loaded=(), fail=None):
        self.loaded = list(loaded)
        self.calls: list[list[str]] = []
        self.fail = fail or set()

    def __call__(self, cmd, **kwargs):
        args = cmd[1:]
        self.calls.append(args)
        if args[0] in self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        out = ""
        if args[:1] == ["ps"]:
            out = json.dumps([{"identifier": m, "modelKey": m} for m in self.loaded])
        elif args[0] == "load":
            self.loaded.append(args[1])
        elif args[0] == "unload":
            self.loaded.remove(args[1])
        return subprocess.CompletedProcess(cmd, 0, out, "")

    def mutations(self):
        return [c for c in self.calls if c[0] in ("load", "unload", "server")]


def _mm(fake, free=9500):
    return ModelManager(Lms(exe="lms", runner=fake), free_vram_mb=lambda: free, headroom_mb=1024, ttl_s=900)


def test_load_with_ttl_then_unload():
    fake = FakeLms()
    mm = _mm(fake)
    assert mm.ensure(CHAT, 8200) is True
    mm.release(CHAT)
    assert fake.mutations() == [["load", CHAT, "--ttl", "900", "-y"], ["unload", CHAT]]
    assert fake.loaded == []


def test_user_loaded_model_is_reused_and_never_unloaded():
    fake = FakeLms(loaded=[CHAT])
    mm = _mm(fake, free=500)  # low VRAM doesn't matter: nothing to load
    assert mm.ensure(CHAT, 8200) is False
    mm.release(CHAT)
    assert mm.release_all() == []
    assert fake.mutations() == []
    assert fake.loaded == [CHAT]


def test_one_model_at_a_time_unloads_ours_first():
    fake = FakeLms()
    mm = _mm(fake)
    mm.ensure(CHAT, 8200)
    mm.ensure(VISION, 6400)
    assert fake.mutations() == [
        ["load", CHAT, "--ttl", "900", "-y"],
        ["unload", CHAT],
        ["load", VISION, "--ttl", "900", "-y"],
    ]
    mm.release_all()
    assert fake.loaded == []


def test_loading_ours_never_unloads_a_hand_loaded_other_model():
    fake = FakeLms(loaded=[VISION])
    mm = _mm(fake, free=9500)
    mm.ensure(CHAT, 8200)
    mm.release_all()
    assert ["unload", VISION] not in fake.calls
    assert fake.loaded == [VISION]


def test_refuses_to_load_when_vram_too_low():
    fake = FakeLms()
    mm = _mm(fake, free=8000)
    with pytest.raises(ModelError, match="8000 MiB VRAM free, need 9224"):
        mm.ensure(CHAT, 8200)
    assert fake.mutations() == []
    assert mm.ours == []


def test_unreadable_vram_refuses():
    fake = FakeLms()

    def boom():
        raise OSError("nvidia-smi missing")

    mm = ModelManager(Lms(exe="lms", runner=fake), free_vram_mb=boom)
    with pytest.raises(ModelError, match="can't read free VRAM"):
        mm.ensure(CHAT, 8200)
    assert fake.mutations() == []


def test_release_skips_unload_when_ttl_already_expired():
    fake = FakeLms()
    mm = _mm(fake)
    mm.ensure(CHAT, 8200)
    fake.loaded.clear()  # LM Studio's TTL unloaded it
    mm.release(CHAT)
    assert ["unload", CHAT] not in fake.calls


def test_lms_failure_is_model_error():
    fake = FakeLms(fail={"load"})
    mm = _mm(fake)
    with pytest.raises(ModelError, match="exited 1: boom"):
        mm.ensure(CHAT, 8200)
    assert mm.ours == []


def test_lms_missing_executable_is_model_error():
    def runner(cmd, **kw):
        raise FileNotFoundError("no lms")

    with pytest.raises(ModelError):
        Lms(exe="lms", runner=runner).loaded()


def test_unreadable_ps_output_is_model_error():
    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "not json", "")

    with pytest.raises(ModelError, match="unreadable"):
        Lms(exe="lms", runner=runner).loaded()


def test_stop_server_refuses_while_models_loaded():
    fake = FakeLms(loaded=[CHAT])
    with pytest.raises(ModelError):
        _mm(fake).stop_server()
    assert fake.mutations() == []
    fake.loaded.clear()
    _mm(fake).stop_server()
    assert fake.mutations() == [["server", "stop"]]
