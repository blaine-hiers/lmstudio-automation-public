import pytest

from lmsa import gate
from lmsa.gate import GateConfig, GpuState


def _boom(*a, **k):
    raise OSError("probe exploded")


@pytest.fixture
def quiet(monkeypatch):
    """Every reader reports a quiet PC; tests override one signal at a time."""
    monkeypatch.setattr(gate, "_read_notification_state", lambda: gate.QUNS_ACCEPTS_NOTIFICATIONS)
    monkeypatch.setattr(gate, "_read_gpu", lambda: GpuState(util_pct=2, free_mb=9500, total_mb=10240))
    monkeypatch.setattr(gate, "_read_idle_seconds", lambda: 3600.0)
    monkeypatch.setattr(gate, "_read_process_names", lambda: ["explorer.exe", "Code.exe"])
    return monkeypatch


CFG = GateConfig(gpu_util_max=30, vram_headroom_mb=1024, idle_minutes=15)


def test_all_quiet(quiet):
    assert gate.busy_reasons(CFG, need_vram_mb=8200) == []


@pytest.mark.parametrize("state", [gate.QUNS_RUNNING_D3D_FULL_SCREEN, gate.QUNS_BUSY, gate.QUNS_PRESENTATION_MODE])
def test_fullscreen_states_are_busy(quiet, state):
    quiet.setattr(gate, "_read_notification_state", lambda: state)
    reasons = gate.busy_reasons(CFG)
    assert len(reasons) == 1 and reasons[0].startswith("fullscreen:")


@pytest.mark.parametrize("state", [gate.QUNS_ACCEPTS_NOTIFICATIONS, gate.QUNS_QUIET_TIME, gate.QUNS_NOT_PRESENT])
def test_other_notification_states_are_quiet(quiet, state):
    quiet.setattr(gate, "_read_notification_state", lambda: state)
    assert gate.fullscreen_reasons() == []


def test_gpu_utilization_over_threshold_is_busy(quiet):
    quiet.setattr(gate, "_read_gpu", lambda: GpuState(util_pct=85, free_mb=9500, total_mb=10240))
    assert gate.busy_reasons(CFG) == ["gpu: 85% utilization (limit 30%)"]


def test_low_free_vram_is_busy_only_when_a_model_is_needed(quiet):
    quiet.setattr(gate, "_read_gpu", lambda: GpuState(util_pct=0, free_mb=400, total_mb=10240))
    reasons = gate.busy_reasons(CFG, need_vram_mb=8200)
    assert len(reasons) == 1 and "400 MiB VRAM free, need 9224 MiB" in reasons[0]
    assert gate.busy_reasons(CFG, need_vram_mb=0) == []  # model already resident / none needed


def test_vram_boundary_is_inclusive(quiet):
    quiet.setattr(gate, "_read_gpu", lambda: GpuState(util_pct=0, free_mb=9224, total_mb=10240))
    assert gate.busy_reasons(CFG, need_vram_mb=8200) == []


def test_recent_input_is_busy(quiet):
    quiet.setattr(gate, "_read_idle_seconds", lambda: 120.0)
    reasons = gate.busy_reasons(CFG)
    assert len(reasons) == 1 and reasons[0].startswith("input: last keyboard/mouse input 2.0 min ago")


def test_idle_threshold_boundary_and_override(quiet):
    quiet.setattr(gate, "_read_idle_seconds", lambda: 15 * 60.0)
    assert gate.busy_reasons(CFG) == []  # exactly N minutes idle is enough
    quiet.setattr(gate, "_read_idle_seconds", lambda: 60.0)
    assert gate.busy_reasons(CFG, idle_minutes=0.5) == []  # overnight override


def test_zero_idle_minutes_never_reads_input(quiet):
    quiet.setattr(gate, "_read_idle_seconds", _boom)
    assert gate.idle_reasons(0) == []


def test_denylisted_process_is_busy_case_and_exe_insensitive(quiet):
    quiet.setattr(gate, "_read_process_names", lambda: ["explorer.exe", "steam.exe", "OBS64.EXE"])
    cfg = GateConfig(blocked_processes=("Steam.EXE", "obs64", "notrunning.exe"))
    assert gate.process_reasons(cfg.blocked_processes) == ["process: obs64 is running", "process: steam is running"]
    assert gate.busy_reasons(cfg) == ["process: obs64 is running", "process: steam is running"]


def test_denylist_does_not_match_substrings(quiet):
    quiet.setattr(gate, "_read_process_names", lambda: ["steamwebhelper.exe"])
    assert gate.process_reasons(["steam.exe"]) == []


def test_empty_denylist_never_blocks_and_never_lists_processes(quiet):
    quiet.setattr(gate, "_read_process_names", _boom)  # would be "unreadable" if called
    assert gate.process_reasons([]) == []
    assert gate.process_reasons(["", "  "]) == []
    assert gate.busy_reasons(GateConfig(blocked_processes=())) == []


@pytest.mark.parametrize("reader,prefix", [
    ("_read_notification_state", "fullscreen: can't read"),
    ("_read_gpu", "gpu: can't read"),
    ("_read_idle_seconds", "input: can't read"),
    ("_read_process_names", "processes: can't read"),
])
def test_unreadable_signal_counts_as_busy(quiet, reader, prefix):
    quiet.setattr(gate, reader, _boom)
    cfg = GateConfig(blocked_processes=("steam.exe",))
    reasons = gate.busy_reasons(cfg)
    assert len(reasons) == 1 and reasons[0].startswith(prefix) and "treating as busy" in reasons[0]


def test_parse_nvidia_smi():
    assert gate.parse_nvidia_smi("12, 2048, 10240\n") == GpuState(12, 2048, 10240)
    assert gate.parse_nvidia_smi("3, 9000, 10240\n40, 100, 8192\n") == GpuState(3, 9000, 10240)  # first GPU
    for bad in ("", "[N/A], 100, 10240", "1, 2"):
        with pytest.raises(ValueError):
            gate.parse_nvidia_smi(bad)


def test_unparseable_nvidia_smi_is_busy(quiet, monkeypatch):
    class P:
        stdout = "[N/A], [N/A], [N/A]\n"
    monkeypatch.setattr(gate, "_read_gpu", lambda: gate.parse_nvidia_smi(P.stdout))
    assert gate.gpu_reasons(CFG)[0].startswith("gpu: can't read")
