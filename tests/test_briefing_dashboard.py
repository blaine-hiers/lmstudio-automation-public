"""Briefing onboarding in the dashboard (#52, #95): the `briefing` tile
offers `--name` as a dropdown of config/briefings/*.toml, and each briefing
whose *effective* on/off state is still unset is NEW, with enable /
keep-disabled actions. Those actions write config/local/briefings.toml (a
gitignored local override) through the config-save path (confirm,
validated, backed up, atomic) -- the shipped config/briefings/<name>.toml
is never touched."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmsa.briefing import config as bconfig
from tasks import dashboard as db

from test_dashboard import call, jcall
from test_dashboard_onboarding import dash, root, server, tasks_by_name  # noqa: F401 (fixtures)

REPO = Path(__file__).resolve().parents[1]

BRIEFING = """\
# a comment that must survive
[briefing]   # the table
title = "{title}"
interests = ["ports"]
{extra}
[[sources.feeds]]
url = "https://example.com/feed.xml"
"""


@pytest.fixture
def bdir(root):  # noqa: F811
    (root / "tasks" / "briefing.py").write_text((REPO / "tasks" / "briefing.py").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    d = root / "config" / "briefings"
    d.mkdir()
    (d / "alpha.toml").write_bytes(BRIEFING.format(title="Alpha", extra="").replace("\n", "\r\n").encode())
    (d / "bravo.toml").write_text(BRIEFING.format(title="Bravo", extra="enabled = true"), encoding="utf-8")
    (d / "charlie.toml").write_text(BRIEFING.format(title="Charlie", extra=""), encoding="utf-8")
    (d / "broken.toml").write_text(BRIEFING.format(title="Broken", extra="colour = 1"), encoding="utf-8")
    return d


def post(srv, action, name, confirm=True):
    return jcall(srv, "POST", f"/api/tasks/briefing-{action}", {"task": "briefing", "fields": {"name": name},
                                                               "confirm": confirm})


def test_tile_offers_briefing_names_and_flags_new_files(server, dash, bdir):  # noqa: F811
    t = tasks_by_name(dash)["briefing"]
    name_field = next(f for f in t.fields if f.flag == "--name")
    assert name_field.choices == ("alpha", "bravo", "broken", "charlie")
    assert t.briefings_new == ("alpha", "broken", "charlie")
    status, _, raw = call(server, "GET", "/")
    page = raw.decode()
    assert status == 200
    assert '<select id="f-briefing-name" data-flag="--name">' in page and '<option value="alpha">' in page
    assert 'data-briefing="alpha"' in page and 'data-onboard="briefing-enable"' in page
    assert 'data-onboard="briefing-disable"' in page and "3 new briefings" in page


def test_enable_writes_the_local_file_and_leaves_the_shipped_one_alone(server, dash, bdir, root, tmp_path):  # noqa: F811, E501
    path = bdir / "alpha.toml"
    local = root / "config" / "local" / "briefings.toml"
    before = path.read_bytes()
    assert not local.exists()
    status, body = post(server, "enable", "alpha", confirm=False)
    assert status == 428 and "alpha = true" in body["error"] and "config/local/briefings.toml" in body["error"]
    assert path.read_bytes() == before and not local.exists()
    status, body = post(server, "enable", "alpha")
    assert status == 200, body
    # the shipped file is byte-for-byte untouched (git status stays clean)
    assert path.read_bytes() == before
    after = local.read_text(encoding="utf-8")
    assert "[briefings]" in after and "alpha = true" in after
    assert bconfig.load("alpha", bdir).enabled is True
    # no prior local file existed, so the first write makes no backup
    backups = list((tmp_path / "data" / "dashboard" / "config-backups").glob("briefings-local-*.toml"))
    assert backups == []
    assert "alpha" not in tasks_by_name(dash)["briefing"].briefings_new
    # toggling again is not an error: the local override always wins, so
    # re-asserting the same state is just a (backed-up) rewrite
    status, body = post(server, "enable", "alpha")
    assert status == 200, body
    backups = list((tmp_path / "data" / "dashboard" / "config-backups").glob("briefings-local-*.toml"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == after


def test_disable_and_refusals(server, dash, bdir, root):  # noqa: F811
    status, body = post(server, "disable", "charlie")
    assert status == 200, body
    assert bconfig.load("charlie", bdir).enabled is False
    # bravo already ships `enabled = true`: overriding it locally still
    # works (the local file always wins, there's no "already set" error)
    status, body = post(server, "enable", "bravo")
    assert status == 200, body
    assert bconfig.load("bravo", bdir).enabled is True
    for name, code in (("../bravo", 400), ("nope", 400), (None, 400)):
        status, body = post(server, "enable", name)
        assert status == code, (name, body)
    broken = (bdir / "broken.toml").read_bytes()
    status, body = post(server, "enable", "broken")
    assert status == 400 and "invalid" in body["error"] and "colour" in body["error"]
    assert (bdir / "broken.toml").read_bytes() == broken  # never replaced with an invalid file
    local = root / "config" / "local" / "briefings.toml"
    assert "broken" not in (local.read_text(encoding="utf-8") if local.exists() else "")
    status, _, _ = call(server, "POST", "/api/tasks/briefing-enable", {"fields": {"name": "alpha"}, "confirm": True},
                        headers={db.TOKEN_HEADER: "wrong"})
    assert status == 403  # same token check as every other write


def test_local_file_holds_more_than_one_briefing(server, dash, bdir, root):  # noqa: F811
    post(server, "enable", "alpha")
    post(server, "disable", "charlie")
    local = root / "config" / "local" / "briefings.toml"
    cfg_alpha, cfg_charlie = bconfig.load("alpha", bdir), bconfig.load("charlie", bdir)
    assert cfg_alpha.enabled is True and cfg_charlie.enabled is False
    text = local.read_text(encoding="utf-8")
    assert "alpha = true" in text and "charlie = false" in text


def test_malformed_local_file_is_refused_on_write_but_never_corrupts_the_shipped_file(server, dash, bdir, root):  # noqa: F811, E501
    local = root / "config" / "local" / "briefings.toml"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("not valid toml [[[", encoding="utf-8")
    status, body = post(server, "enable", "alpha")
    assert status == 400 and "can't be read" in body["error"]
    assert local.read_text(encoding="utf-8") == "not valid toml [[["  # left alone, not overwritten
    assert (bdir / "alpha.toml").exists()  # and the shipped file was never touched either


def test_malformed_local_file_warns_on_the_briefing_tile(server, dash, bdir, root):  # noqa: F811
    """(#95 review finding 2) a malformed config/local/briefings.toml is
    surfaced on the dashboard's briefing tile, not just stderr."""
    local = root / "config" / "local" / "briefings.toml"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("not valid toml [[[", encoding="utf-8")
    t = tasks_by_name(dash)["briefing"]
    assert any("config/local/briefings.toml has a problem" in w and "is ignored" in w for w in t.warnings)
    status, _, raw = call(server, "GET", "/")
    assert status == 200
    assert "config/local/briefings.toml has a problem" in raw.decode()


def test_toggle_preserves_hand_written_comments_and_order(server, dash, bdir, root):  # noqa: F811
    """(#95 review finding 3) the old whole-file rewrite dropped every hand
    edit; a toggle must now touch only the one line it means to change."""
    local = root / "config" / "local" / "briefings.toml"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(
        "# hand-written note about alpha -- must survive a toggle\n"
        "[briefings]\n"
        "alpha = true   # keep me on\n"
        "# charlie is off until Q2\n"
        "charlie = false\n",
        encoding="utf-8",
    )
    status, body = post(server, "disable", "alpha")
    assert status == 200, body
    text = local.read_text(encoding="utf-8")
    assert "# hand-written note about alpha -- must survive a toggle" in text
    assert "# charlie is off until Q2" in text
    assert "charlie = false" in text
    assert "alpha = false" in text
    assert text.index("alpha") < text.index("charlie")  # order preserved
    assert bconfig.load("alpha", bdir).enabled is False
    assert bconfig.load("charlie", bdir).enabled is False


def test_mkdir_error_when_config_local_is_a_file_is_a_bad_request(server, dash, bdir, root):  # noqa: F811
    """(#95 review finding 4) `path.parent.mkdir` must be inside the same
    try/except OSError as the rest of the write, so the owner gets the
    usual "nothing was changed" instead of a 500."""
    local_as_file = root / "config" / "local"
    local_as_file.write_bytes(b"oops, this is a file, not a directory")
    status, body = post(server, "enable", "alpha")
    assert status == 400 and "nothing was changed" in body["error"]
    assert local_as_file.read_bytes() == b"oops, this is a file, not a directory"  # untouched
    assert (bdir / "alpha.toml").exists()


def test_concurrent_toggles_of_different_briefings_all_survive(server, dash, bdir, root):  # noqa: F811
    """Several threads toggling different briefings at once must not lose
    each other's writes -- the global _BRIEFING_LOCK serializes the
    read-modify-write so nothing is clobbered."""
    import threading

    names = ["alpha", "bravo", "charlie"]
    results = {}

    def toggle(n):
        results[n] = post(server, "enable", n)

    threads = [threading.Thread(target=toggle, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for n in names:
        assert results[n][0] == 200, results[n]
        assert bconfig.load(n, bdir).enabled is True

    local = root / "config" / "local" / "briefings.toml"
    text = local.read_text(encoding="utf-8")
    for n in names:
        assert f"{n} = true" in text


def test_concurrent_external_edit_between_read_and_write_is_caught_by_sha_check(
        server, dash, bdir, root, monkeypatch):  # noqa: F811
    """(#95 review finding 5) a cross-process edit landing between our read
    and our write must be caught by the base-sha check in
    `_config_save_locked`, not silently overwritten."""
    from tasks import dashboard_setup as ds

    local = root / "config" / "local" / "briefings.toml"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("[briefings]\nalpha = false\n", encoding="utf-8")

    real_upsert = ds._upsert_local_briefing_line
    racing_text = "[briefings]\nalpha = false\ncharlie = true\n"

    def racing(current, name, enabled):
        # Simulate another process writing to the file after we already
        # read `current` but before our own write lands.
        local.write_text(racing_text, encoding="utf-8")
        return real_upsert(current, name, enabled)

    monkeypatch.setattr(ds, "_upsert_local_briefing_line", racing)
    status, body = post(server, "enable", "alpha")
    assert status == 409 and "changed since" in body["error"]
    assert local.read_text(encoding="utf-8") == racing_text  # the race's write is untouched


def test_toggle_refuses_to_save_a_local_file_that_would_not_read_back(server, dash, bdir, root):  # noqa: F811
    # A quoted key the line-level edit doesn't recognise would gain a duplicate
    # entry. bravo ships enabled = true, so a load-back check alone would pass;
    # the save must still be refused and the file left untouched.
    local = root / "config" / "local" / "briefings.toml"
    local.parent.mkdir(parents=True, exist_ok=True)
    original = b'[briefings]\n"bravo" = false\n'
    local.write_bytes(original)
    status, body = post(server, "enable", "bravo")
    assert status == 400, body
    assert "read back" in body["error"]
    assert local.read_bytes() == original


def test_real_briefings_dir_uses_the_isolated_local_path():
    # conftest points LOCAL_OVERRIDES_PATH at tmp; the dashboard passes the
    # real config/briefings explicitly and must land on the same path.
    assert bconfig.local_overrides_path(bconfig.BRIEFINGS_DIR) == bconfig.LOCAL_OVERRIDES_PATH
