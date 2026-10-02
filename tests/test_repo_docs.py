"""Repo docs/config stay in step with the code: .env.example names every env
var read, the README describes the Gmail write paths, deps are listed once."""

from __future__ import annotations

import re
from pathlib import Path

from lmsa import secrets

ROOT = Path(__file__).resolve().parents[1]


def test_every_env_var_read_is_declared_in_env_example():
    read: set[str] = set(secrets._ENV_VARS.values())
    for path in list((ROOT / "lmsa").rglob("*.py")) + list((ROOT / "tasks").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        read |= set(re.findall(r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z0-9_]+)["']""", source))
        read |= set(re.findall(r"""os\.environ\[\s*["']([A-Z0-9_]+)["']\s*\]""", source))
    declared = set(re.findall(r"^([A-Z0-9_]+)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
    assert "LMSA_FRONT_PAGE_CONFIG" in read  # the scan actually sees task-level reads
    assert read - declared == set()


def test_readme_describes_the_two_opt_in_gmail_write_paths():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "never writes labels, drafts" not in readme
    for name in ("append_draft", "add_labels", "remove_labels", "--apply"):
        assert name in readme


def test_pyproject_lists_each_dependency_once():
    import tomllib

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    names = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in deps]
    assert len(names) == len(set(names)), names
