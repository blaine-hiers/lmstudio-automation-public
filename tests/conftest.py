"""Suite-wide fixtures.

Every test runs against an in-memory keyring, so nothing reads or writes the
real Windows Credential Manager (`lmsa.secrets.resolve` falls back to it).
"""

from __future__ import annotations

import keyring
import pytest
from keyring.backend import KeyringBackend

import lmsa.search as search_mod
from lmsa.briefing import config as bconfig


class MemoryKeyring(KeyringBackend):
    priority = 1  # never auto-selected; installed explicitly below

    def __init__(self):
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


@pytest.fixture(autouse=True)
def fake_keyring():
    before = keyring.get_keyring()
    ring = MemoryKeyring()
    keyring.set_keyring(ring)
    try:
        yield ring
    finally:
        keyring.set_keyring(before)


class _UnfakedSearXNGSpawn(AssertionError):
    """Raised in place of a real ``subprocess.Popen`` call from
    ``lmsa.search``'s on-demand SearXNG starter (#89) when a test builds a
    ``SearXNG`` without passing its own ``popen=`` fake."""


def _forbidden_popen(*args, **kwargs):
    raise _UnfakedSearXNGSpawn(
        "lmsa.search tried to spawn a real process via subprocess.Popen -- "
        "pass popen=<fake> to SearXNG(...) instead of exercising the real default."
    )


@pytest.fixture(autouse=True)
def _no_real_searxng_start(monkeypatch):
    """Hermetic guard for #89 (see review of the first attempt): once the
    repo owner sets ``LMSA_SEARXNG_START`` in the real ``.env``,
    ``lmsa.config``'s module-level ``load_dotenv`` puts it in
    ``os.environ`` for the whole process -- and every ``SearXNG`` built in a
    test without an explicit ``start_command`` picks it up via the sentinel
    default, so a connection-level failure (very common against a fake/dead
    instance in tests) would spawn the *real* command. Two layers:

    1. Scrub both env vars before every test, however they got into
       ``os.environ`` (a real ``.env``, an exported shell var, or a leftover
       from another test) -- ``load_dotenv`` already ran at import time, so
       deleting from ``os.environ`` here is what actually takes it away.
    2. Replace the default ``popen`` (real ``subprocess.Popen``, captured as
       a keyword-only default on ``SearXNG.__init__`` at import time --
       patching ``subprocess.Popen`` itself would NOT reach it) with one
       that fails the test loudly instead of spawning, so any spawn that
       slips through some other way (a test explicitly re-setting the env
       var, a future call site) can't actually launch a process.
    """
    monkeypatch.delenv("LMSA_SEARXNG_START", raising=False)
    monkeypatch.delenv("LMSA_SEARXNG_START_TIMEOUT", raising=False)
    init = search_mod.SearXNG.__init__
    monkeypatch.setattr(init, "__kwdefaults__", {**init.__kwdefaults__, "popen": _forbidden_popen})


@pytest.fixture(autouse=True)
def _no_real_local_briefing_overrides(monkeypatch, tmp_path):
    """Hermetic guard for #95 (see review of the first attempt): several
    tests call `bconfig.load`/`bconfig.parse` with no `directory`/`path`,
    which resolves the local overrides file to the module-level
    `LOCAL_OVERRIDES_PATH` -- the *real* `config/local/briefings.toml` in
    the owner's own checkout. Without this, a real local file that enables
    `conflicts`/`example`/etc. makes `test_shipped_conflicts_config` and
    `test_list_and_main_with_name` fail (and the latter would even run the
    real briefings via `task.main([])`).

    Point it at a path under this test's own `tmp_path` that never exists,
    so every test starts with "no local overrides" unless it explicitly
    monkeypatches `LOCAL_OVERRIDES_PATH` (or passes its own `local_path`/
    `overrides`) to something else."""
    monkeypatch.setattr(bconfig, "LOCAL_OVERRIDES_PATH", tmp_path / "no-such-local-briefings-override.toml")
