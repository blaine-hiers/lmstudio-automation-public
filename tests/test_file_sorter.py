import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tasks.file_sorter as fs
from tasks.file_sorter import (
    Plan,
    apply_plans,
    build_plans,
    candidate_files,
    is_reserved,
    move_no_clobber,
    plan_dest,
    read_preview,
    read_undo_log,
    render_report,
    sanitize,
    undo,
    unique_dest,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows filesystem semantics")


def _fake_completion(**kwargs) -> SimpleNamespace:
    content = json.dumps({"descriptive_name": "sunset-hike", "folder": "Photos"})
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLMStudio:
    """Stands in for lmsa.LMStudio: no server, no network.

    `plans` maps a filename to the plan dict to return, or to an exception to raise.
    """

    def __init__(self, other_plan=None, plans=None):
        self.cfg = SimpleNamespace(vision_model="fake-vision", chat_model="fake-chat")
        self._other_plan = other_plan or {"descriptive_name": "note", "folder": "Notes"}
        self._plans = plans or {}
        self.prompts: list[str] = []
        # only .api.chat.completions.create is used, by plan_image()
        self.api = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_completion))
        )

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.prompts.append(prompt)
        filename = prompt.split("\n")[0][len("Filename: "):]
        result = self._plans.get(filename, self._other_plan)
        if isinstance(result, Exception):
            raise result
        return result if not isinstance(result, dict) else dict(result)


def _age(path: Path, days: float = 2) -> None:
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def _files(root: Path, *names: str) -> list[Path]:
    out = []
    for n in names:
        p = root / n
        p.write_text(n)
        _age(p)
        out.append(p)
    return out


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr("tasks.file_sorter.ROOT", repo)
    return repo


# --- candidate_files -------------------------------------------------------


def test_candidate_files_filters_by_age(tmp_path):
    old = tmp_path / "old.txt"
    old.write_text("x")
    _age(old, 2)
    new = tmp_path / "new.txt"
    new.write_text("x")
    _age(new, 0.1)
    (tmp_path / "sub").mkdir()

    found = candidate_files([tmp_path])
    assert found == [old]


def test_candidate_files_skips_partial_downloads_and_junk(tmp_path):
    keep = _files(tmp_path, "report.pdf")
    _files(
        tmp_path,
        "x.crdownload", "y.part", "z.tmp", "w.download", "v.partial",
        "desktop.ini", "Thumbs.db", ".hidden-dotfile",
    )
    assert candidate_files([tmp_path]) == keep


@windows_only
def test_candidate_files_skips_hidden_and_system_attributes(tmp_path):
    keep = _files(tmp_path, "visible.txt")
    hidden, system = _files(tmp_path, "hidden.txt", "system.txt")
    subprocess.run(["attrib", "+h", str(hidden)], check=True)
    subprocess.run(["attrib", "+s", str(system)], check=True)
    try:
        assert candidate_files([tmp_path]) == keep
    finally:
        subprocess.run(["attrib", "-h", str(hidden)], check=True)
        subprocess.run(["attrib", "-s", str(system)], check=True)


# --- sanitize / reserved names ----------------------------------------------


def test_sanitize_strips_invalid_chars_and_falls_back():
    assert sanitize("a/b:c*d?e<f>g|h", fallback="x") == "a-b-c-d-e-f-g-h"
    assert sanitize("   ", fallback="fallback-name") == "fallback-name"
    assert sanitize("..", fallback="fb") == "fb"


@pytest.mark.parametrize(
    "name", ["CON", "nul", "Prn", "AUX", "COM1", "com9", "LPT1", "lpt9", "NUL.txt", "con.tar.gz", "COM1 ", "AUX .x"]
)
def test_sanitize_never_returns_a_reserved_device_name(name):
    out = sanitize(name, fallback="fb")
    assert not is_reserved(out)
    assert out.startswith("_")


def test_sanitize_reserved_fallback_is_also_renamed():
    assert not is_reserved(sanitize("", fallback="NUL"))


def test_sanitize_strips_trailing_space_and_dot_after_truncation():
    # A cut at 80 chars lands right after a space / dot.
    assert sanitize("z" * 79 + " q", fallback="x") == "z" * 79
    assert sanitize("z" * 79 + ".q", fallback="x") == "z" * 79
    out = sanitize("ok" + " " * 10 + "z" * 69 + " " + "q" * 20, fallback="x")
    assert not out.endswith((" ", "."))
    assert len(out) <= 80


def test_plan_dest_uses_sanitized_folder_and_name(tmp_path):
    src = tmp_path / "IMG_0001.PNG"
    src.touch()
    dest = plan_dest(src, {"descriptive_name": "sunset hike", "folder": "Photos/2026"})
    assert dest == tmp_path / "Photos-2026" / "sunset hike.png"


def test_plan_dest_renames_reserved_folder_and_file(tmp_path):
    src = tmp_path / "a.txt"
    dest = plan_dest(src, {"descriptive_name": "CON", "folder": "NUL"})
    assert dest == tmp_path / "_NUL" / "_CON.txt"


def test_plan_dest_avoids_folder_name_taken_by_a_file(tmp_path):
    (tmp_path / "Notes").write_text("I am a file")
    dest = plan_dest(tmp_path / "a.txt", {"descriptive_name": "a", "folder": "Notes"})
    assert dest == tmp_path / "Notes-2" / "a.txt"


@windows_only
def test_apply_reserved_names_never_lose_the_source(tmp_path, data_root):
    srcs = _files(tmp_path, "a.txt", "b.txt")
    lm = FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "CON", "folder": "NUL"},
        "b.txt": {"descriptive_name": "x", "folder": "COM1"},
    })
    plans, skips = build_plans(lm, srcs)
    assert not skips
    log_path, done, apply_skips = apply_plans(plans)
    assert not apply_skips
    for p in done:
        assert p.dest.read_text() in {"a.txt", "b.txt"}
    assert sorted(p.dest.read_text() for p in done) == ["a.txt", "b.txt"]


@windows_only
def test_move_no_clobber_refuses_reserved_destination(tmp_path):
    src = _files(tmp_path, "a.txt")[0]
    with pytest.raises(ValueError):
        move_no_clobber(src, tmp_path / "NUL")
    with pytest.raises(ValueError):
        move_no_clobber(src, tmp_path / "con.txt")
    assert src.read_text() == "a.txt"


# --- unique_dest --------------------------------------------------------------


def test_unique_dest_suffixes_on_collision(tmp_path):
    candidate = tmp_path / "note.txt"
    candidate.write_text("existing")
    taken: set[str] = set()

    first = unique_dest(candidate, taken)
    assert first == tmp_path / "note-2.txt"

    second = unique_dest(candidate, taken)
    assert second == tmp_path / "note-3.txt"


@windows_only
def test_case_insensitive_plans_do_not_collide(tmp_path, data_root):
    srcs = _files(tmp_path, "a.txt", "b.txt")
    lm = FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "Same", "folder": "F"},
        "b.txt": {"descriptive_name": "same", "folder": "f"},
    })
    plans, _ = build_plans(lm, srcs)
    _, done, skips = apply_plans(plans)
    assert not skips
    assert sorted(p.dest.read_text() for p in done) == ["a.txt", "b.txt"]


# --- build_plans --------------------------------------------------------------


def test_build_plans_routes_images_and_other_files(tmp_path):
    from PIL import Image

    img = tmp_path / "shot.png"
    Image.new("RGB", (4, 4), color="red").save(img)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    lm = FakeLMStudio()
    plans, skips = build_plans(lm, [img, txt])

    assert skips == []
    by_src = {p.src: p for p in plans}
    assert by_src[img].kind == "image"
    assert by_src[img].dest == tmp_path / "Photos" / "sunset-hike.png"
    assert by_src[txt].kind == "other"
    assert by_src[txt].dest == tmp_path / "Notes" / "note.txt"


def test_build_plans_isolates_model_errors(tmp_path):
    a, b = _files(tmp_path, "a.txt", "b.txt")
    lm = FakeLMStudio(plans={"a.txt": RuntimeError("model timeout")})
    plans, skips = build_plans(lm, [a, b])
    assert [p.src for p in plans] == [b]
    assert [s.src for s in skips] == [a]
    assert "model timeout" in skips[0].reason


def test_build_plans_isolates_corrupt_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"notanimage")
    (txt,) = _files(tmp_path, "a.txt")
    plans, skips = build_plans(FakeLMStudio(), [bad, txt])
    assert [p.src for p in plans] == [txt]
    assert [s.src for s in skips] == [bad]
    assert "UnidentifiedImageError" in skips[0].reason


def test_build_plans_isolates_non_object_reply(tmp_path):
    a, b = _files(tmp_path, "a.txt", "b.txt")
    plans, skips = build_plans(FakeLMStudio(plans={"a.txt": ["list"]}), [a, b])
    assert [p.src for p in plans] == [b]
    assert [s.src for s in skips] == [a]


# --- PDF preview ------------------------------------------------------------


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_read_preview_extracts_pdf_first_page(tmp_path):
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(_minimal_pdf("Invoice 4471 from Acme"))
    assert "Invoice 4471" in read_preview(pdf)


def test_pdf_text_reaches_the_model_and_corrupt_pdf_is_isolated(tmp_path):
    good = tmp_path / "good.pdf"
    good.write_bytes(_minimal_pdf("Quarterly statement"))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-garbage")
    lm = FakeLMStudio()
    plans, skips = build_plans(lm, [good, bad])
    assert skips == []  # a corrupt PDF still gets a name/extension-only plan
    assert len(plans) == 2
    assert "Quarterly statement" in lm.prompts[0]
    assert "Content preview" not in lm.prompts[1]


# --- apply / undo -------------------------------------------------------------


def test_apply_then_undo_round_trips(tmp_path, data_root):
    src = tmp_path / "notes.txt"
    src.write_text("hello")
    dest = tmp_path / "Notes" / "note.txt"
    plans = [Plan(src=src, dest=dest, kind="other")]

    log_path, done, skips = apply_plans(plans)

    assert skips == []
    assert not src.exists()
    assert dest.read_text() == "hello"
    assert read_undo_log(log_path) == [{"seq": 1, "src": str(src), "dest": str(dest)}]

    notes = undo(log_path)

    assert src.read_text() == "hello"
    assert not dest.exists()
    assert notes == [f"restored: {src}"]
    # A second undo is a no-op.
    assert undo(log_path) == [f"skip (not at destination): {dest}"]


def test_apply_never_overwrites_a_dest_created_after_planning(tmp_path, data_root):
    (src,) = _files(tmp_path, "a.txt")
    plans, _ = build_plans(FakeLMStudio(plans={"a.txt": {"descriptive_name": "a", "folder": "F"}}), [src])
    (tmp_path / "F").mkdir()
    (tmp_path / "F" / "a.txt").write_text("PRECIOUS")

    _, done, skips = apply_plans(plans)

    assert skips == []
    assert (tmp_path / "F" / "a.txt").read_text() == "PRECIOUS"
    assert done[0].dest == tmp_path / "F" / "a-2.txt"
    assert done[0].dest.read_text() == "a.txt"


def test_move_no_clobber_raises_instead_of_replacing(tmp_path):
    src, dest = _files(tmp_path, "a.txt", "b.txt")
    with pytest.raises(FileExistsError):
        move_no_clobber(src, dest)
    assert src.read_text() == "a.txt" and dest.read_text() == "b.txt"


def test_move_no_clobber_skips_cross_device_instead_of_copying(tmp_path, monkeypatch):
    src = _files(tmp_path, "a.txt")[0]
    dest = tmp_path / "out.txt"

    def fake_cross_device(*a, **k):
        e = OSError(18, "cross-device")
        e.errno = 18  # EXDEV
        if sys.platform == "win32":
            e.winerror = 17
        raise e

    monkeypatch.setattr(os, "rename", fake_cross_device)
    monkeypatch.setattr(os, "link", fake_cross_device, raising=False)
    with pytest.raises(OSError):
        move_no_clobber(src, dest)
    assert src.read_text() == "a.txt" and not dest.exists()


def test_posix_link_path_removes_its_own_link_when_unlink_fails(tmp_path, monkeypatch):
    src = _files(tmp_path, "a.txt")[0]
    dest = tmp_path / "out.txt"
    made = []

    def fake_link(s, d):
        Path(d).write_bytes(Path(s).read_bytes())  # stand-in for a hard link
        made.append(Path(d))

    real_unlink = os.unlink

    def fake_unlink(p):
        if Path(p) == src:
            raise PermissionError("locked")
        real_unlink(p)

    monkeypatch.setattr(fs, "_WINDOWS", False)
    monkeypatch.setattr(fs.os, "link", fake_link, raising=False)
    monkeypatch.setattr(fs.os, "unlink", fake_unlink)
    with pytest.raises(PermissionError):
        move_no_clobber(src, dest)
    assert made == [dest] and not dest.exists() and src.read_text() == "a.txt"


def test_undo_restores_a_device_like_original_name(tmp_path, data_root):
    if sys.platform != "win32":
        pytest.skip("device-like names are ordinary files only on Windows 11")
    src = tmp_path / "nul.txt"
    try:
        src.write_text("keep me")
        assert src.read_text() == "keep me"
    except (OSError, AssertionError):
        pytest.skip("this Windows build treats nul.txt as a device")
    dest = tmp_path / "F" / "note.txt"
    dest.parent.mkdir()
    log_path, done, skips = apply_plans([fs.Plan(src=src, dest=dest, kind="other")])
    assert done and not src.exists()
    assert undo(log_path) == [f"restored: {src}"]
    assert src.read_text() == "keep me"


def test_compound_archive_extensions_are_kept(tmp_path):
    src = tmp_path / "backup.tar.gz"
    src.write_text("x")
    assert fs.plan_dest(src, {"descriptive_name": "Backup", "folder": "Archives"}).name == "Backup.tar.gz"


def test_no_move_happens_when_the_undo_log_cannot_be_created(tmp_path, data_root, monkeypatch):
    srcs = _files(tmp_path, "a.txt")
    plans = [fs.Plan(src=srcs[0], dest=tmp_path / "F" / "a.txt", kind="other")]

    def no_log():
        raise PermissionError("data/ not writable")

    monkeypatch.setattr(fs, "_new_undo_log", no_log)
    log_path, done, skips = apply_plans(plans)
    assert log_path is None and done == [] and [s.src for s in skips] == srcs
    assert srcs[0].exists() and not (tmp_path / "F" / "a.txt").exists()


def test_apply_isolates_vanished_file(tmp_path, data_root):
    a, b = _files(tmp_path, "a.txt", "b.txt")
    plans, _ = build_plans(FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "a", "folder": "F"},
        "b.txt": {"descriptive_name": "b", "folder": "F"},
    }), [a, b])
    a.unlink()
    log_path, done, skips = apply_plans(plans)
    assert [s.src for s in skips] == [a]
    assert [p.src for p in done] == [b]
    assert read_undo_log(log_path) == [{"seq": 1, "src": str(b), "dest": str(tmp_path / "F" / "b.txt")}]


def test_apply_isolates_locked_file(tmp_path, data_root):
    srcs = _files(tmp_path, "a.txt", "b.txt", "c.txt")
    lm = FakeLMStudio(plans={n: {"descriptive_name": n[0], "folder": "F"} for n in ["a.txt", "b.txt", "c.txt"]})
    plans, _ = build_plans(lm, srcs)
    with open(srcs[1]):  # an open handle blocks rename on Windows
        log_path, done, skips = apply_plans(plans)
    assert [s.src for s in skips] == [srcs[1]]
    assert [p.src for p in done] == [srcs[0], srcs[2]]
    # 2 confirmed moves; the locked file's intent is marked failed, so undo never sees it.
    assert len(read_undo_log(log_path)) == 2
    assert {"failed": 2} in [json.loads(line) for line in log_path.read_text().splitlines()]
    assert srcs[1].read_text() == "b.txt"


def test_apply_handles_folder_name_equal_to_existing_file(tmp_path, data_root):
    a, b, zzz = _files(tmp_path, "a.txt", "b.txt", "zzz")
    lm = FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "a", "folder": "zzz"},
        "b.txt": {"descriptive_name": "b", "folder": "Q"},
        "zzz": {"descriptive_name": "z", "folder": "Q"},
    })
    plans, skips = build_plans(lm, [a, b, zzz])
    assert skips == []
    _, done, apply_skips = apply_plans(plans)
    assert apply_skips == []
    assert (tmp_path / "zzz-2" / "a.txt").read_text() == "a.txt"
    assert (tmp_path / "Q" / "z").read_text() == "zzz"


def test_apply_folder_turned_into_file_after_planning(tmp_path, data_root):
    a, b = _files(tmp_path, "a.txt", "b.txt")
    plans, _ = build_plans(FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "a", "folder": "F"},
        "b.txt": {"descriptive_name": "b", "folder": "G"},
    }), [a, b])
    (tmp_path / "F").write_text("a file squatting on the folder name")
    _, done, skips = apply_plans(plans)
    assert skips == []
    assert (tmp_path / "F").read_text() == "a file squatting on the folder name"
    assert (tmp_path / "F-2" / "a.txt").read_text() == "a.txt"
    assert (tmp_path / "G" / "b.txt").read_text() == "b.txt"


def test_undo_log_survives_a_crash_mid_apply(tmp_path, data_root, monkeypatch):
    srcs = _files(tmp_path, "a.txt", "b.txt", "c.txt")
    lm = FakeLMStudio(plans={n: {"descriptive_name": n[0], "folder": "F"} for n in ["a.txt", "b.txt", "c.txt"]})
    plans, _ = build_plans(lm, srcs)

    real_move = fs.move_no_clobber
    calls = {"n": 0}

    def crash_on_second(src, dest):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # simulated process death mid-run
        real_move(src, dest)

    monkeypatch.setattr(fs, "move_no_clobber", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        apply_plans(plans)

    logs = list((data_root / "data").glob("file_sorter-undo-*.jsonl"))
    assert len(logs) == 1
    # The 2nd file's intent was logged before its move crashed; the source stayed put,
    # so it is marked failed and undo only sees the confirmed first move.
    assert read_undo_log(logs[0]) == [{"seq": 1, "src": str(srcs[0]), "dest": str(tmp_path / "F" / "a.txt")}]
    assert {"failed": 2} in [json.loads(line) for line in logs[0].read_text().splitlines()]
    monkeypatch.setattr(fs, "move_no_clobber", real_move)
    assert undo(logs[0]) == [f"restored: {srcs[0]}"]
    assert srcs[0].read_text() == "a.txt"


def test_read_undo_log_ignores_torn_last_line(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"src": "a", "dest": "b"}) + "\n" + '{"src": "c", "de')
    assert read_undo_log(log) == [{"src": "a", "dest": "b"}]


def test_two_applies_within_one_second_get_distinct_logs(tmp_path, data_root):
    (a,) = _files(tmp_path, "a.txt")
    l1, _, _ = apply_plans(build_plans(FakeLMStudio(), [a])[0])
    (b,) = _files(tmp_path, "b.txt")
    l2, _, _ = apply_plans(build_plans(FakeLMStudio(), [b])[0])
    assert l1 != l2
    assert len(read_undo_log(l1)) == 1 and len(read_undo_log(l2)) == 1


def test_undo_log_is_never_reused(tmp_path, data_root, monkeypatch):
    class FixedUUID:
        hex = "0" * 32

    monkeypatch.setattr(fs.uuid, "uuid4", lambda: FixedUUID())

    class FixedDT:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime(2026, 1, 1)

    monkeypatch.setattr(fs, "datetime", FixedDT)
    (a,) = _files(tmp_path, "a.txt")
    apply_plans(build_plans(FakeLMStudio(), [a])[0])
    (b,) = _files(tmp_path, "b.txt")
    with pytest.raises(FileExistsError):
        fs._new_undo_log()


def test_apply_with_nothing_to_move_writes_no_log(tmp_path, data_root):
    assert apply_plans([]) == (None, [], [])


def test_undo_skips_when_state_no_longer_matches(tmp_path):
    src = tmp_path / "notes.txt"
    dest = tmp_path / "Notes" / "note.txt"
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(json.dumps({"src": str(src), "dest": str(dest)}) + "\n")

    notes = undo(log_path)

    assert notes == [f"skip (not at destination): {dest}"]


def test_undo_never_overwrites_a_reoccupied_original(tmp_path, data_root):
    (a,) = _files(tmp_path, "a.txt")
    log_path, done, _ = apply_plans(build_plans(FakeLMStudio(), [a])[0])
    a.write_text("new file at old path")
    assert undo(log_path) == [f"skip (original path occupied): {a}"]
    assert a.read_text() == "new file at old path"
    assert done[0].dest.read_text() == "a.txt"


@windows_only
def test_undo_isolates_locked_entry(tmp_path, data_root):
    a, b = _files(tmp_path, "a.txt", "b.txt")
    lm = FakeLMStudio(plans={
        "a.txt": {"descriptive_name": "a", "folder": "F"},
        "b.txt": {"descriptive_name": "b", "folder": "F"},
    })
    log_path, done, _ = apply_plans(build_plans(lm, [a, b])[0])
    with open(done[1].dest):
        notes = undo(log_path)
    assert notes[0].startswith("error (")
    assert notes[1] == f"restored: {a}"
    assert a.read_text() == "a.txt"
    assert done[1].dest.read_text() == "b.txt"


def test_dry_run_writes_nothing(tmp_path):
    (a,) = _files(tmp_path, "a.txt")
    before = sorted(tmp_path.rglob("*"))
    mtime = a.stat().st_mtime
    build_plans(FakeLMStudio(), candidate_files([tmp_path]))
    assert sorted(tmp_path.rglob("*")) == before
    assert a.stat().st_mtime == mtime


# --- report -------------------------------------------------------------------


def test_render_report_lists_current_proposed_and_skipped(tmp_path):
    src = tmp_path / "shot.png"
    dest = tmp_path / "Photos" / "sunset-hike.png"
    plans = [Plan(src=src, dest=dest, kind="image")]

    html = render_report(
        plans, [tmp_path], applied=False, skips=[fs.Skip(tmp_path / "bad<b>.png", "boom <script>")]
    ).render()

    assert "shot.png" in html
    assert "sunset-hike.png" in html
    assert "Proposed moves" in html
    assert "Skipped" in html
    assert "<script>" not in html


# --- #87: undo acts only on confirmed moves ---------------------------------


def _log_records(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_undo_of_a_run_whose_move_failed_leaves_a_later_runs_move_alone(tmp_path, data_root, monkeypatch):
    x, y = _files(tmp_path, "x.png.txt", "y.txt")
    folder = tmp_path / "F"
    plan_x = Plan(src=x, dest=folder / "x.txt", kind="other")
    plan_y = Plan(src=y, dest=folder / "y.txt", kind="other")

    real_move = fs.move_no_clobber

    def locked_x(src, dest, **kw):
        if src == x:
            raise PermissionError("locked by another process")
        real_move(src, dest, **kw)

    monkeypatch.setattr(fs, "move_no_clobber", locked_x)
    log_a, done_a, skips_a = apply_plans([plan_x, plan_y])  # run A: x fails, y moves
    monkeypatch.setattr(fs, "move_no_clobber", real_move)
    assert [s.src for s in skips_a] == [x] and [p.src for p in done_a] == [y]

    log_b, done_b, _ = apply_plans([plan_x])  # run B: x now lands at the same dest
    assert done_b[0].dest == folder / "x.txt"

    notes = undo(log_a)

    assert notes == [f"restored: {y}"]
    assert (folder / "x.txt").read_text() == "x.png.txt"  # B's move untouched
    assert not x.exists()
    assert undo(log_b) == [f"restored: {x}"]


def test_race_lost_attempt_is_marked_failed_and_undo_never_moves_the_foreign_file(tmp_path, data_root, monkeypatch):
    (a,) = _files(tmp_path, "a.txt")
    folder = tmp_path / "F"
    folder.mkdir()
    real_move = fs.move_no_clobber
    calls = {"n": 0}

    def racing(src, dest, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            dest.write_text("FOREIGN")  # another process wins the name between check and rename
        real_move(src, dest, **kw)

    monkeypatch.setattr(fs, "move_no_clobber", racing)
    log_path, done, skips = apply_plans([Plan(src=a, dest=folder / "a.txt", kind="other")])
    monkeypatch.setattr(fs, "move_no_clobber", real_move)
    assert skips == [] and done[0].dest == folder / "a-2.txt"

    records = _log_records(log_path)
    first = next(r for r in records if r.get("dest") == str(folder / "a.txt"))
    assert {"failed": first["seq"]} in records
    assert [m["dest"] for m in read_undo_log(log_path)] == [str(folder / "a-2.txt")]

    # The user already put the file back by hand; the original path is free again.
    real_move(folder / "a-2.txt", a)
    notes = undo(log_path)
    assert notes == [f"skip (not at destination): {folder / 'a-2.txt'}"]
    assert (folder / "a.txt").read_text() == "FOREIGN"
    assert a.read_text() == "a.txt"


def test_relative_root_undoes_from_another_cwd(tmp_path, data_root, monkeypatch):
    root = tmp_path / "dl"
    root.mkdir()
    _files(root, "a.txt")
    monkeypatch.chdir(tmp_path)
    files = candidate_files([Path("dl")])
    assert files == [Path("dl") / "a.txt"]
    plans, _ = build_plans(FakeLMStudio(), files)
    log_path, done, _ = apply_plans(plans)
    assert all(Path(m["src"]).is_absolute() and Path(m["dest"]).is_absolute() for m in read_undo_log(log_path))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert undo(log_path) == [f"restored: {root / 'a.txt'}"]
    assert (root / "a.txt").read_text() == "a.txt"


def test_undo_refuses_relative_paths_in_a_log(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(json.dumps({"seq": 1, "src": "a.txt", "dest": "F/a.txt"}) + "\n" + json.dumps({"done": 1}) + "\n")
    notes = undo(log_path)
    assert len(notes) == 1 and notes[0].startswith("refused (relative path")


def test_old_format_log_still_undoes(tmp_path):
    src = tmp_path / "a.txt"
    dest = tmp_path / "F" / "a.txt"
    dest.parent.mkdir()
    dest.write_text("a")
    log_path = tmp_path / "file_sorter-undo-old.jsonl"
    log_path.write_text(json.dumps({"src": str(src), "dest": str(dest)}) + "\n")
    assert undo(log_path) == [f"restored: {src}"]
    assert src.read_text() == "a" and not dest.exists()


def test_intent_and_done_markers_are_logged(tmp_path, data_root):
    (a,) = _files(tmp_path, "a.txt")
    log_path, done, _ = apply_plans([Plan(src=a, dest=tmp_path / "F" / "a.txt", kind="other")])
    assert _log_records(log_path) == [
        {"seq": 1, "src": str(a), "dest": str(tmp_path / "F" / "a.txt")},
        {"done": 1},
    ]


def test_only_failed_attempts_write_no_log(tmp_path, data_root, monkeypatch):
    (a,) = _files(tmp_path, "a.txt")

    def locked(src, dest, **kw):
        raise PermissionError("locked")

    monkeypatch.setattr(fs, "move_no_clobber", locked)
    log_path, done, skips = apply_plans([Plan(src=a, dest=tmp_path / "F" / "a.txt", kind="other")])
    assert log_path is None and done == [] and [s.src for s in skips] == [a]
    assert list((data_root / "data").glob("file_sorter-undo-*.jsonl")) == []


def test_lost_done_marker_still_counts_as_moved_and_keeps_the_log(tmp_path, data_root, monkeypatch):
    (a,) = _files(tmp_path, "a.txt")
    real_line = fs._log_line

    def no_done(log, record):
        if "done" in record:
            raise OSError("disk full")
        real_line(log, record)

    monkeypatch.setattr(fs, "_log_line", no_done)
    log_path, done, skips = apply_plans([Plan(src=a, dest=tmp_path / "F" / "a.txt", kind="other")])
    assert skips == [] and [p.src for p in done] == [a]
    assert log_path is not None and log_path.exists()
    notes = undo(log_path)
    assert len(notes) == 1 and notes[0].startswith("skip (never confirmed; check by hand)")
    assert (tmp_path / "F" / "a.txt").exists()  # never acted on


def test_interrupt_after_the_rename_stays_unconfirmed(tmp_path, data_root, monkeypatch):
    (a,) = _files(tmp_path, "a.txt")
    real_move = fs.move_no_clobber

    def move_then_die(src, dest, **kw):
        real_move(src, dest, **kw)
        raise KeyboardInterrupt

    monkeypatch.setattr(fs, "move_no_clobber", move_then_die)
    with pytest.raises(KeyboardInterrupt):
        apply_plans([Plan(src=a, dest=tmp_path / "F" / "a.txt", kind="other")])
    (log_path,) = (data_root / "data").glob("file_sorter-undo-*.jsonl")
    assert not any("failed" in r for r in _log_records(log_path))
    notes = undo(log_path)
    assert len(notes) == 1 and "never confirmed" in notes[0]
