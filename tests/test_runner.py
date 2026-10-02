import pytest

from lmsa import runner as runner_mod
from lmsa.runner import run


def test_run_calls_main_and_returns_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(runner_mod, "notify", lambda *a, **k: calls.append((a, k)))

    def main():
        calls.append("ran")

    run(main)

    assert calls == ["ran"]


def test_run_alerts_and_reraises_on_failure(monkeypatch):
    alerts = []
    monkeypatch.setattr(runner_mod, "notify", lambda *a, **k: alerts.append((a, k)))

    def main():
        raise ValueError("boom")

    main.__module__ = "tasks.fake_task"

    with pytest.raises(ValueError, match="boom"):
        run(main)

    assert len(alerts) == 1
    args, kwargs = alerts[0]
    assert args[0] == "tasks.fake_task failed"
    assert "ValueError: boom" in args[1]
    assert kwargs == {"level": "urgent"}


def test_run_keeps_original_exception_when_alert_fails(monkeypatch):
    import pytest
    from lmsa import runner

    def broken_notify(*a, **k):
        raise RuntimeError("alert channel down")

    monkeypatch.setattr(runner, "notify", broken_notify)

    def main():
        raise ValueError("task broke")

    with pytest.raises(ValueError, match="task broke"):
        runner.run(main)


def test_task_name_resolves_dunder_main_via_spec(monkeypatch):
    import sys
    import types
    from lmsa import runner

    fake_main = types.ModuleType("__main__")
    fake_main.__spec__ = types.SimpleNamespace(name="tasks.fake_task")
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    def main():
        pass

    main.__module__ = "__main__"
    assert runner._task_name(main) == "tasks.fake_task"
