"""tasks.doc_inbox: extraction schema, sanitised destinations, collisions,
apply + undo round trip, and a dry run that writes nothing. No LM Studio."""

import json
import os
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import tasks.doc_inbox as di
from lmsa import Store
from tasks.doc_inbox import (
    SCHEMA,
    Config,
    DocInfo,
    apply_plans,
    build_plans,
    due_events,
    load_config,
    normalize,
    parse_template,
    plan_dest,
    read_document,
    run_log_path,
    undo,
)


def fields(**over):
    base = {
        "title": "Electricity bill",
        "doc_date": "2026-03-04",
        "correspondent": "City Power",
        "doc_type": "Bill",
        "tags": ["utilities", "power"],
        "amount": 81.5,
        "due_date": "2026-10-15",
    }
    base.update(over)
    return base


class FakeLM:
    """Stands in for lmsa.LMStudio. `answers` maps a filename to fields or an exception."""

    def __init__(self, answers=None, default=None, output_dir=None):
        self.cfg = SimpleNamespace(vision_model="fake-vision", chat_model="fake-chat", output_dir=output_dir)
        self.answers = answers or {}
        self.default = default or fields()
        self.extract_calls: list[str] = []
        self.see_calls: list[tuple[int, str]] = []

    def _answer(self, prompt):
        name = prompt.split("\n")[0][len("Filename: "):]
        result = self.answers.get(name, self.default)
        if isinstance(result, Exception):
            raise result
        return dict(result)

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        assert schema is SCHEMA
        self.extract_calls.append(prompt)
        return self._answer(prompt)

    def see(self, images, prompt, schema=None, *, system=None, model=None, name="result"):
        assert schema is SCHEMA and images and all(isinstance(i, bytes) for i in images)
        self.see_calls.append((len(images), prompt))
        return self._answer(prompt)

    def ensure_server(self):
        pass


def _png(color="white") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return buf.getvalue()


def _blank_pdf() -> bytes:
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


def _docx(path: Path, text: str) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph(text)
    d.save(str(path))


@pytest.fixture
def cfg(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return Config(
        inbox=inbox,
        archive=inbox / "Filed",
        template=parse_template(di.DEFAULT_TEMPLATE),
        min_age=timedelta(0),
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(di, "ROOT", repo)
    return repo


def _drop(cfg, *names, content=None):
    out = []
    for n in names:
        p = cfg.inbox / n
        p.write_bytes(content if content is not None else _png())
        t = datetime.now().timestamp() - 3600  # older than any min_age the tests use
        os.utime(p, (t, t))
        out.append(p)
    return out


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --- extraction schema -------------------------------------------------------


def test_schema_is_strict_and_requires_every_field():
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"]) == {
        "title", "doc_date", "correspondent", "doc_type", "tags", "amount", "due_date",
    }
    assert SCHEMA["properties"]["tags"]["type"] == "array"


def test_normalize_keeps_good_fields(tmp_path):
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    info = normalize(fields(), src)
    assert info == DocInfo(
        title="Electricity bill", correspondent="City Power", doc_type="Bill",
        doc_date=date(2026, 3, 4), date_from_file=False, tags=["utilities", "power"],
        amount=81.5, due_date=date(2026, 10, 15),
    )


def test_normalize_falls_back_on_malformed_model_output(tmp_path):
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    t = datetime(2025, 7, 1, 12, 0).timestamp()
    os.utime(src, (t, t))
    info = normalize(
        fields(title="  ", correspondent=None, doc_type="", doc_date="March 4th", due_date="2026-13-40",
               amount=float("nan"), tags=["a", "A", " ", "b" * 100, *[f"t{i}" for i in range(20)]]),
        src,
    )
    assert info.title == "scan" and info.correspondent == "Unknown" and info.doc_type == "Other"
    assert info.doc_date == date(2025, 7, 1) and info.date_from_file
    assert info.due_date is None and info.amount is None
    assert info.tags[0] == "a" and len(info.tags) == di.MAX_TAGS and len(info.tags[1]) == 40
    assert normalize(fields(amount=True), src).amount is None
    with pytest.raises(ValueError):
        normalize(["not", "a", "dict"], src)


def test_read_document_routes_each_kind(tmp_path, monkeypatch):
    lm = FakeLM()
    img = tmp_path / "photo.jpg"
    img.write_bytes(_png())  # PNG bytes behind a .jpg name: content decides, not the extension
    assert read_document(lm, img)["source"] == "image-vision"

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(_blank_pdf())  # no text layer -> rendered pages to the vision model
    out = read_document(lm, scan)
    assert out["source"] == "pdf-vision" and lm.see_calls[-1][0] == 1

    doc = tmp_path / "letter.docx"
    _docx(doc, "Dear customer, your policy renews on 2026-11-01.")
    out = read_document(lm, doc)
    assert out["source"] == "docx-text" and "policy renews" in lm.extract_calls[-1]
    assert out["fields"]["title"] == "Electricity bill" and out["truncated"] is False


def test_read_document_truncates_long_text_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(di, "_pdf_text", lambda data: "word " * 5000)
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(_blank_pdf())
    lm = FakeLM()
    out = read_document(lm, pdf)
    assert out["source"] == "pdf-text" and out["truncated"] is True
    assert len(lm.extract_calls[-1]) < di.TEXT_LIMIT + 200 and "truncated" in lm.extract_calls[-1]


def test_read_document_rejects_mislabelled_and_oversized_files(tmp_path, monkeypatch):
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"MZ\x90\x00 not a pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        read_document(FakeLM(), fake)
    monkeypatch.setattr(di, "MAX_BYTES", 10)
    big = tmp_path / "big.png"
    big.write_bytes(_png())
    with pytest.raises(ValueError, match="cap"):
        read_document(FakeLM(), big)


# --- filename sanitisation / template ---------------------------------------


def test_default_destination(cfg):
    src = cfg.inbox / "scan0001.PDF"
    src.write_bytes(b"x")
    dest = plan_dest(cfg, src, normalize(fields(), src))
    assert dest == cfg.archive / "Bill" / "2026" / "2026-03-04 City Power - Electricity bill.pdf"


@pytest.mark.parametrize("hostile", [
    {"doc_type": "../../..", "correspondent": "..\\..\\Windows", "title": "/etc/passwd"},
    {"doc_type": "CON", "correspondent": "NUL", "title": "COM1"},
    {"doc_type": "a/b\\c", "correspondent": 'x:*?"<>|y', "title": "t\x00\x1f."},
    {"doc_type": "C:\\Users", "correspondent": "\\\\server\\share", "title": " . "},
])
def test_hostile_fields_stay_inside_the_archive(cfg, hostile):
    src = cfg.inbox / "in.pdf"
    src.write_bytes(b"x")
    dest = plan_dest(cfg, src, normalize(fields(**hostile), src))
    rel = dest.relative_to(cfg.archive)
    assert len(rel.parts) == 3 and dest.suffix == ".pdf"
    for part in rel.parts:
        assert part not in ("", ".", "..") and not any(c in part for c in '\\/:*?"<>|\x00')
        assert part == part.rstrip(" .")
    from tasks.file_sorter import is_reserved

    assert not any(is_reserved(p) for p in rel.parts)


def test_long_fields_are_truncated(cfg):
    src = cfg.inbox / "in.pdf"
    src.write_bytes(b"x")
    dest = plan_dest(cfg, src, normalize(fields(title="x" * 500, correspondent="y" * 500), src))
    assert len(dest.name) <= 150 + len(".pdf")


@pytest.mark.parametrize("bad", [
    "{doc_type}/../{title}", "/abs/{title}", "C:/x/{title}", "{doc_type}//{title}",
    "{secret}/{title}", "{title!r}", "{year:>10}", "{ext}/{title}", "",
])
def test_template_validation_rejects_unsafe_templates(bad):
    with pytest.raises(ValueError):
        parse_template(bad)


def test_template_without_ext_still_keeps_extension(cfg):
    cfg.template = parse_template("{correspondent}/{year}-{month} {title}")
    src = cfg.inbox / "in.docx"
    src.write_bytes(b"x")
    assert plan_dest(cfg, src, normalize(fields(), src)) == cfg.archive / "City Power" / "2026-03 Electricity bill.docx"


def test_load_config(tmp_path, monkeypatch):
    toml = tmp_path / "doc_inbox.toml"
    toml.write_text('template = "{doc_type}/{title}{ext}"\nmin_age_minutes = 0\n', encoding="utf-8")
    monkeypatch.setenv("LMSA_DOC_INBOX_DIR", str(tmp_path / "in"))
    c = load_config(toml)
    assert c.inbox == tmp_path / "in" and c.archive == tmp_path / "in" / "Filed"
    assert c.template == ["{doc_type}", "{title}"] and c.min_age == timedelta(0)
    toml.write_text(f'archive_dir = "{(tmp_path / "in").as_posix()}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        load_config(toml)
    monkeypatch.delenv("LMSA_DOC_INBOX_DIR")
    with pytest.raises(ValueError, match="LMSA_DOC_INBOX_DIR"):
        load_config(tmp_path / "missing.toml")


def test_shipped_config_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSA_DOC_INBOX_DIR", str(tmp_path))
    c = load_config()
    assert c.template == parse_template(di.DEFAULT_TEMPLATE)


# --- planning + collisions ---------------------------------------------------


def test_one_bad_document_does_not_stop_the_rest(cfg):
    files = _drop(cfg, "a.png", "b.png", "c.png")
    lm = FakeLM(answers={"b.png": RuntimeError("model fell over")})
    plans, skips = build_plans(lm, cfg, files)
    assert [p.src.name for p in plans] == ["a.png", "c.png"]
    assert len(skips) == 1 and "model fell over" in skips[0].reason


def test_collisions_within_a_run_and_on_disk_get_suffixes(cfg):
    files = _drop(cfg, "a.png", "b.png", "c.png")
    existing = cfg.archive / "Bill" / "2026" / "2026-03-04 City Power - Electricity bill.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already filed")
    plans, skips = build_plans(FakeLM(), cfg, files)
    assert not skips
    assert [p.dest.name for p in plans] == [
        "2026-03-04 City Power - Electricity bill-2.png",
        "2026-03-04 City Power - Electricity bill-3.png",
        "2026-03-04 City Power - Electricity bill-4.png",
    ]


def test_memo_skips_the_model_for_unchanged_files(cfg, tmp_path):
    files = _drop(cfg, "a.png")
    lm = FakeLM()
    with Store(tmp_path / "db.sqlite") as store:
        memo = di.open_memo(store.conn, lm)
        build_plans(lm, cfg, files, memo)
        build_plans(lm, cfg, files, memo)
    assert len(lm.see_calls) == 1


# --- apply + undo ------------------------------------------------------------


def test_apply_then_undo_round_trip(cfg, data_root):
    files = _drop(cfg, "a.png", "b.png")
    files[1].write_bytes(_png("black"))
    before = _tree(cfg.inbox)
    plans, _ = build_plans(FakeLM(), cfg, files)
    run_id = di.new_run_id()
    log, done, skips = apply_plans(cfg, plans, run_id)
    assert not skips and len(done) == 2 and log == run_log_path(run_id)
    assert log.is_relative_to(data_root / "data" / "doc_inbox")
    assert not files[0].exists() and not files[1].exists()
    assert all(p.dest.is_file() for p in done)
    header = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert header["run_id"] == run_id and header["archive"] == str(cfg.archive)

    notes = undo(run_id)
    assert [n.status for n in notes] == ["restored", "restored"]
    assert {k: v for k, v in _tree(cfg.inbox).items() if not k.startswith("Filed")} == before


def test_apply_never_overwrites_a_file_that_appears_after_planning(cfg, data_root):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    plans[0].dest.parent.mkdir(parents=True)
    plans[0].dest.write_bytes(b"raced in")
    _, done, skips = apply_plans(cfg, plans, di.new_run_id())
    assert not skips and done[0].dest.name.endswith("-2.png")
    assert plans[0].dest.read_bytes() == b"raced in"


def test_apply_skips_a_vanished_source_and_writes_no_log_when_nothing_moved(cfg, data_root):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    files[0].unlink()
    log, done, skips = apply_plans(cfg, plans, di.new_run_id())
    assert log is None and not done and len(skips) == 1
    assert not (data_root / "data" / "doc_inbox").exists() or not any((data_root / "data" / "doc_inbox").iterdir())


def test_undo_refuses_to_overwrite_and_reports_what_it_could_not_restore(cfg, data_root):
    files = _drop(cfg, "a.png", "b.png", "c.png")
    plans, _ = build_plans(FakeLM(answers={
        "a.png": fields(title="A"), "b.png": fields(title="B"), "c.png": fields(title="C"),
    }), cfg, files)
    run_id = di.new_run_id()
    _, done, _ = apply_plans(cfg, plans, run_id)
    by_name = {p.src.name: p for p in done}
    files[0].write_bytes(b"new file with the old name")  # a.png: original path occupied
    by_name["b.png"].dest.write_bytes(b"edited after filing")  # b.png: changed since
    notes = undo(run_id)
    status = {Path(n.path).name: (n.status, n.detail) for n in notes}
    assert status["c.png"][0] == "restored"
    assert status["a.png"] == ("skipped", "original path is occupied; not overwriting")
    assert status[by_name["b.png"].dest.name] == ("skipped", "changed since it was filed")
    assert files[0].read_bytes() == b"new file with the old name"
    assert by_name["a.png"].dest.is_file()  # left where it was


def test_undo_refuses_log_entries_outside_the_runs_inbox_and_archive(cfg, data_root, tmp_path):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    run_id = di.new_run_id()
    log, _, _ = apply_plans(cfg, plans, run_id)
    victim = tmp_path / "elsewhere" / "important.txt"
    victim.parent.mkdir()
    victim.write_text("keep me")
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 99, "src": str(tmp_path / "elsewhere" / "moved.txt"), "dest": str(victim),
                            "size": 7}) + "\n")
        f.write(json.dumps({"done": 99}) + "\n")
    notes = undo(run_id)
    assert notes[0].status == "refused" and victim.read_text() == "keep me"
    assert notes[1].status == "restored"


@pytest.mark.parametrize("bad", ["../../etc/passwd", "20260925-031000", "x" * 30, ""])
def test_undo_rejects_a_malformed_run_id(bad, data_root):
    with pytest.raises(ValueError, match="run id"):
        undo(bad)


def test_undo_unknown_run_id(data_root):
    with pytest.raises(FileNotFoundError):
        undo("20260925-031000-abcdef12")


# --- review round 2: cwd independence, completion markers, junctions --------


def test_relative_inbox_is_logged_absolute_and_undo_works_from_another_cwd(tmp_path, data_root, monkeypatch):
    (tmp_path / "inbox").mkdir()
    toml = tmp_path / "doc_inbox.toml"
    toml.write_text('archive_dir = "Archive"\nmin_age_minutes = 0\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(toml, inbox="inbox")
    assert cfg.inbox == tmp_path / "inbox" and cfg.inbox.is_absolute()
    assert cfg.archive == tmp_path / "inbox" / "Archive"  # relative archive_dir: relative to the inbox
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    run_id = di.new_run_id()
    log, done, _ = apply_plans(cfg, plans, run_id)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert all(Path(v).is_absolute() for m in lines for k, v in m.items() if k in ("inbox", "archive", "src", "dest"))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert [n.status for n in undo(run_id)] == ["restored"]
    assert (tmp_path / "inbox" / "a.png").is_file()


def test_failed_move_in_run_a_is_never_undone_by_undo_a(cfg, data_root, monkeypatch):
    """Repro: run A's move of a.png fails after its intent is logged; run B then
    files a.png to the same destination. undo(A) must not reverse B's move."""
    files = _drop(cfg, "a.png", "b.png")
    lm = FakeLM(answers={"a.png": fields(title="A"), "b.png": fields(title="B")})
    plans_a, _ = build_plans(lm, cfg, files)
    real_move = di.move_no_clobber

    def flaky(src, dest, **kw):
        if Path(src).name == "a.png":
            raise PermissionError(13, "locked by another process", str(src))
        return real_move(src, dest, **kw)

    monkeypatch.setattr(di, "move_no_clobber", flaky)
    run_a = di.new_run_id()
    log_a, done_a, skips_a = apply_plans(cfg, plans_a, run_a)
    assert [p.src.name for p in done_a] == ["b.png"] and "locked" in skips_a[0].reason
    monkeypatch.setattr(di, "move_no_clobber", real_move)

    plans_b, _ = build_plans(lm, cfg, [files[0]])
    run_b = di.new_run_id()
    _, done_b, _ = apply_plans(cfg, plans_b, run_b)
    assert done_b[0].dest == plans_a[0].dest  # same destination A had logged an intent for

    notes = undo(run_a)
    assert [(n.status, Path(n.path).name) for n in notes] == [("restored", "b.png")]
    assert done_b[0].dest.is_file() and not files[0].exists()  # B's move left alone
    assert [n.status for n in undo(run_b)] == ["restored"]


def test_race_retry_marks_the_lost_attempt_failed(cfg, data_root, monkeypatch):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    real_move = di.move_no_clobber
    raced = plans[0].dest

    def racing(src, dest, **kw):
        if Path(dest) == raced and not raced.exists():
            raced.write_bytes(b"someone else's file")  # appears between unique_dest and the rename
            raise FileExistsError(17, "destination exists", str(dest))
        return real_move(src, dest, **kw)

    monkeypatch.setattr(di, "move_no_clobber", racing)
    run_id = di.new_run_id()
    log, done, _ = apply_plans(cfg, plans, run_id)
    assert done[0].dest.name.endswith("-2.png")
    header, completed, unconfirmed = di.read_run_log(log)
    assert [Path(m["dest"]).name for m in completed] == [done[0].dest.name] and not unconfirmed
    files[0].write_bytes(b"new a.png")  # original path now occupied too
    notes = undo(run_id)
    assert raced.read_bytes() == b"someone else's file"
    assert [n.status for n in notes] == ["skipped"]


def test_nothing_moved_leaves_no_log_and_no_undo_hint(cfg, data_root, tmp_path, monkeypatch, capsys):
    _drop(cfg, "a.png")

    def locked(src, dest, **kw):
        raise PermissionError(13, "locked", str(src))

    monkeypatch.setattr(di, "move_no_clobber", locked)
    _run_main(monkeypatch, tmp_path, cfg, ["--apply"])
    assert "nothing moved; no run log written" in capsys.readouterr().out
    runs = data_root / "data" / "doc_inbox"
    assert not runs.exists() or not any(runs.iterdir())


def test_unconfirmed_intent_is_reported_not_acted_on(cfg, data_root):
    files = _drop(cfg, "a.png", "b.png")
    plans, _ = build_plans(FakeLM(answers={"a.png": fields(title="A"), "b.png": fields(title="B")}), cfg, files)
    run_id = di.new_run_id()
    log, done, _ = apply_plans(cfg, plans, run_id)
    # Simulate a crash between the rename and its done marker for b.png.
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join(x for x in lines if x != json.dumps({"done": 2})) + "\n", encoding="utf-8")
    notes = undo(run_id)
    assert sorted(n.status for n in notes) == ["restored", "skipped"]
    unconfirmed = next(n for n in notes if n.status == "skipped")
    assert "never confirmed" in unconfirmed.detail and "b.png" in unconfirmed.detail
    assert done[1].dest.is_file()


def _link_dir(link: Path, target: Path) -> None:
    """A directory junction on Windows (no admin needed), a symlink elsewhere."""
    import subprocess
    import sys

    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        os.symlink(target, link, target_is_directory=True)


def test_junction_in_archive_gets_nothing_created_or_moved_outside(cfg, data_root, tmp_path):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    outside = tmp_path / "outside"
    outside.mkdir()
    cfg.archive.mkdir()
    try:
        _link_dir(cfg.archive / "Bill", outside)  # the doc_type folder is a planted junction
    except (OSError, Exception) as e:  # noqa: BLE001
        pytest.skip(f"can't create a directory link here: {e}")
    log, done, skips = apply_plans(cfg, plans, di.new_run_id())
    assert not done and log is None and "outside the archive" in skips[0].reason
    assert list(outside.iterdir()) == []  # no "2026" folder created through the junction
    assert files[0].is_file()


def test_containment_checked_before_mkdir(cfg, data_root, tmp_path, monkeypatch):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    real = os.path.realpath
    elsewhere = str(tmp_path / "elsewhere")

    def fake_realpath(p, *a, **kw):  # pretend archive/Bill resolves elsewhere
        r = real(p, *a, **kw)
        bill = real(cfg.archive / "Bill")
        return elsewhere + r[len(bill):] if os.path.normcase(r).startswith(os.path.normcase(bill)) else r

    monkeypatch.setattr(di.os.path, "realpath", fake_realpath)
    _, done, skips = apply_plans(cfg, plans, di.new_run_id())
    assert not done and "outside the archive" in skips[0].reason
    assert not (cfg.archive / "Bill").exists()  # refused before any folder was created


# --- due dates ---------------------------------------------------------------


def test_due_events_only_future_due_dates(cfg):
    files = _drop(cfg, "a.png", "b.png", "c.png")
    lm = FakeLM(answers={
        "a.png": fields(title="A", due_date="2026-10-15"),
        "b.png": fields(title="B", due_date="2026-01-01"),
        "c.png": fields(title="C", due_date=None),
    })
    plans, _ = build_plans(lm, cfg, files)
    events = due_events(plans, date(2026, 9, 25))
    assert [(e.title, e.start) for e in events] == [("Due: City Power - A", date(2026, 10, 15))]
    assert events[0].all_day


# --- main: dry run writes nothing -------------------------------------------


def _run_main(monkeypatch, tmp_path, cfg, argv):
    out = tmp_path / "out"
    lm = FakeLM(output_dir=out)
    monkeypatch.setattr(di, "LMStudio", lambda: lm)
    monkeypatch.setattr(di, "_now", lambda: datetime(2026, 9, 25, 3, 20, tzinfo=timezone.utc))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("LMSA_DOC_INBOX_DIR", str(cfg.inbox))
    monkeypatch.setattr(di, "DEFAULT_CONFIG_PATH", tmp_path / "cfg.toml")
    (tmp_path / "cfg.toml").write_text("min_age_minutes = 0\n", encoding="utf-8")
    di.main(argv)
    return out


def test_dry_run_writes_nothing_outside_output(cfg, data_root, tmp_path, monkeypatch, capsys):
    _drop(cfg, "a.png", "b.png")
    before = _tree(cfg.inbox)
    out = _run_main(monkeypatch, tmp_path, cfg, [])
    assert _tree(cfg.inbox) == before
    assert not cfg.archive.exists()
    assert not (data_root / "data").exists()
    html = next(p for p in out.rglob("*.html") if p.name != "index.html").read_text(encoding="utf-8")
    assert "dry run" in html and "Electricity bill" in html
    assert list(out.rglob("doc_inbox-due-*.ics"))  # output/ only


def test_main_apply_then_undo(cfg, data_root, tmp_path, monkeypatch, capsys):
    _drop(cfg, "a.png")
    before = _tree(cfg.inbox)
    _run_main(monkeypatch, tmp_path, cfg, ["--apply"])
    printed = capsys.readouterr().out
    run_id = printed.strip().splitlines()[-1].rsplit(" ", 1)[-1]
    assert di.RUN_ID_RE.match(run_id) and not (cfg.inbox / "a.png").exists()
    di.main(["--undo", run_id])
    assert "restored 1, not restored 0" in capsys.readouterr().out
    assert {k: v for k, v in _tree(cfg.inbox).items() if not k.startswith("Filed")} == before


def test_report_escapes_model_output(cfg, data_root, tmp_path, monkeypatch):
    _drop(cfg, "a.png")
    monkeypatch.setattr(FakeLM, "_answer", lambda self, prompt: fields(title="<script>alert(1)</script>"))
    out = _run_main(monkeypatch, tmp_path, cfg, [])
    html = next(p for p in out.rglob("*.html") if p.name != "index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)" not in html


def test_lost_done_marker_keeps_the_move_and_its_log(cfg, data_root, monkeypatch):
    """The rename happened but the done marker couldn't be written (disk full):
    the move counts as done, the log survives, undo lists it as unconfirmed."""
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    real_log_line = di._log_line

    def no_space_for_done(log, record):
        if "done" in record:
            raise OSError(28, "No space left on device")
        return real_log_line(log, record)

    monkeypatch.setattr(di, "_log_line", no_space_for_done)
    log, done, skips = apply_plans(cfg, plans, di.new_run_id())
    assert [p.src.name for p in done] == ["a.png"] and not skips
    assert log is not None and log.is_file()
    _, completed, unconfirmed = di.read_run_log(log)
    assert not completed and [Path(m["src"]).name for m in unconfirmed] == ["a.png"]


def test_interrupt_after_rename_stays_unconfirmed(cfg, data_root, monkeypatch):
    files = _drop(cfg, "a.png")
    plans, _ = build_plans(FakeLM(), cfg, files)
    real_move = di.move_no_clobber

    def moved_then_interrupted(src, dest, **kw):
        real_move(src, dest, **kw)
        raise KeyboardInterrupt

    monkeypatch.setattr(di, "move_no_clobber", moved_then_interrupted)
    run_id = di.new_run_id()
    with pytest.raises(KeyboardInterrupt):
        apply_plans(cfg, plans, run_id)
    log = di.run_log_path(run_id)
    _, completed, unconfirmed = di.read_run_log(log)
    assert not completed and [Path(m["src"]).name for m in unconfirmed] == ["a.png"]
