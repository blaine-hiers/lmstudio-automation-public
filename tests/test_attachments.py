"""tasks.attachments: MIME sniffing (never trusting a declared Content-Type),
PDF text-layer vs. rendered-page vs. image vision branches, per-item
isolation (a corrupt PDF or a failed fetch can't abort the run), and the
shared `receipts` table upsert.

All fixtures are invented (example.com addresses, tiny generated images/PDFs
in memory); no real mailbox or LM Studio server is touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfWriter

import tasks.attachments as attachments
import tasks.receipts as receipts_task
from lmsa.sources.gmail import Attachment, Message
from lmsa.store import Store
from gmail_fakes import MetaSearchMixin


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).date().isoformat()


def _png_bytes(size=(20, 20), color=(200, 30, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(20, 20), color=(30, 200, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _blank_pdf_bytes(width=200, height=200) -> bytes:
    """A structurally valid PDF with a blank page -- no text layer."""
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=height)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


CORRUPT_PDF_BYTES = b"%PDF-1.4 this looks like a pdf but pypdf cannot parse it at all"


def _msg(msg_id: str, subject="Your receipt", day="2026-01-15", attachments=(), **overrides) -> Message:
    fields = dict(
        msg_id=msg_id, thread_id=f"t-{msg_id}", labels=[], flags=[],
        from_name="Billing", from_addr="billing@example.com", to=["me@example.com"],
        reply_to="", subject=subject, date=datetime.fromisoformat(f"{day}T09:00:00+00:00"),
        list_unsubscribe="", list_id="", auth_results="", text="Thanks for your purchase.",
        attachments=list(attachments), web_url="",
    )
    fields.update(overrides)
    return Message(**fields)


class _FakeLMStudio:
    """`extract()`/`see()` results are looked up by a substring of the
    prompt (the attachment name, which both prompt-builders include)."""

    def __init__(self, extract_results=None, see_results=None):
        self._extract_results = extract_results or {}
        self._see_results = see_results or {}
        self.see_calls: list[tuple[int, str]] = []
        self.cfg = type("Cfg", (), {"output_dir": None, "chat_model": "fake", "vision_model": "fake-vision"})()

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        for key, result in self._extract_results.items():
            if key in prompt:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"no fake extract() result for prompt: {prompt[:120]!r}")

    def see(self, images, prompt, schema=None, *, system=None, model=None, name="result"):
        self.see_calls.append((len(images), prompt))
        for key, result in self._see_results.items():
            if key in prompt:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"no fake see() result for prompt: {prompt[:120]!r}")


RECEIPT_RESULT = {
    "is_receipt": True, "merchant": "Acme Cloud", "amount": "19.99", "currency": "usd",
    "date": "2026-01-15", "category": "software", "recurring": False,
}
NOT_A_RECEIPT = {
    "is_receipt": False, "merchant": "", "amount": "0", "currency": "", "date": "",
    "category": "", "recurring": False,
}
SYS = ("sys-image", "sys-pdf")


# ---------------------------------------------------------------------------
# _sniff_kind -- never trusts a declared Content-Type, only the bytes
# ---------------------------------------------------------------------------


def test_sniff_kind_pdf():
    assert attachments._sniff_kind(_blank_pdf_bytes()) == "pdf"


def test_sniff_kind_image():
    assert attachments._sniff_kind(_png_bytes()) == "image"
    assert attachments._sniff_kind(_jpeg_bytes()) == "image"


def test_sniff_kind_unknown_for_garbage_or_other_mime():
    assert attachments._sniff_kind(b"PK\x03\x04 this is actually a zip file") == "unknown"
    assert attachments._sniff_kind(b"") == "unknown"


# ---------------------------------------------------------------------------
# image/PDF-render helpers (real Pillow/pypdf/pypdfium2, no mocking)
# ---------------------------------------------------------------------------


def test_downscale_jpeg_reencodes_and_shrinks():
    big = _png_bytes(size=(800, 800))
    small = attachments._downscale_jpeg(big, max_dim=100, quality=70)
    with Image.open(BytesIO(small)) as im:
        assert im.format == "JPEG"
        assert max(im.size) <= 100


def test_render_pdf_pages_returns_png_bytes():
    pages = attachments._render_pdf_pages(_blank_pdf_bytes(), pages=2, scale=1.0)
    assert len(pages) == 1  # the fixture PDF has exactly one page
    with Image.open(BytesIO(pages[0])) as im:
        assert im.format == "PNG"


def test_pdf_text_is_empty_for_a_blank_page():
    assert attachments._pdf_text(_blank_pdf_bytes()).strip() == ""


def test_render_pdf_pages_rejects_an_oversized_mediabox():
    # A 435-byte PDF that declares a 14400x14400pt page: at the default
    # PDF_RENDER_SCALE this would ask pypdfium2 to allocate a
    # multi-gigabyte bitmap if the page-area budget didn't stop it first.
    huge = _blank_pdf_bytes(width=14400, height=14400)
    with pytest.raises(attachments.PdfPageTooLargeError, match="too large to render"):
        attachments._render_pdf_pages(huge, pages=attachments.PDF_RENDER_PAGES, scale=attachments.PDF_RENDER_SCALE)


def test_render_pdf_pages_normal_page_is_well_under_the_budget():
    # A normal (Letter-ish) page at the real scale must not trip the guard.
    pages = attachments._render_pdf_pages(
        _blank_pdf_bytes(width=612, height=792), pages=2, scale=attachments.PDF_RENDER_SCALE
    )
    assert len(pages) == 1


# ---------------------------------------------------------------------------
# extract_attachment -- the three branches, plus per-item error isolation
# ---------------------------------------------------------------------------


def test_extract_attachment_returns_none_for_unknown_kind():
    lm = _FakeLMStudio()
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "notes.zip", b"PK\x03\x04 not a pdf or image", system_image=SYS[0], system_pdf=SYS[1]
    )
    assert result is None
    assert lm.see_calls == []  # never even tried to classify it


def test_extract_attachment_image_path_uses_see_and_builds_a_thumbnail():
    lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "receipt.jpg", _jpeg_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.ExtractedAttachment)
    assert result.source == "image-vision"
    assert result.receipt.merchant == "Acme Cloud"
    assert result.receipt.amount_minor == 1999
    assert result.receipt.currency == "USD"
    assert result.thumbnail.startswith("data:image/jpeg;base64,")
    assert lm.see_calls[0][0] == 1  # exactly one image sent


def test_extract_attachment_pdf_text_layer_path_uses_extract_and_has_no_thumbnail(monkeypatch):
    monkeypatch.setattr(
        attachments, "_pdf_text", lambda data: "Invoice total: $19.99, billed by Acme Cloud. Thanks for your purchase!"
    )
    lm = _FakeLMStudio(extract_results={"invoice.pdf": RECEIPT_RESULT})
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "invoice.pdf", _blank_pdf_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.ExtractedAttachment)
    assert result.source == "pdf-text"
    assert result.thumbnail is None
    assert lm.see_calls == []


def test_extract_attachment_pdf_without_text_layer_falls_back_to_vision():
    lm = _FakeLMStudio(see_results={"invoice.pdf": RECEIPT_RESULT})
    msg = _msg("m1")
    # the blank-page fixture PDF genuinely has no text layer -- no monkeypatching needed
    result = attachments.extract_attachment(
        lm, msg, "invoice.pdf", _blank_pdf_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.ExtractedAttachment)
    assert result.source == "pdf-vision"
    assert result.thumbnail.startswith("data:image/jpeg;base64,")


def test_extract_attachment_huge_pdf_page_becomes_an_error_not_a_crash():
    lm = _FakeLMStudio()
    msg = _msg("m1")
    huge = _blank_pdf_bytes(width=14400, height=14400)  # no text layer -- falls to the render path
    result = attachments.extract_attachment(
        lm, msg, "huge.pdf", huge, system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.AttachmentError)
    assert "too large to render" in result.error
    assert lm.see_calls == []  # never got as far as a vision call


def test_extract_attachment_returns_none_when_not_a_receipt():
    lm = _FakeLMStudio(see_results={"photo.jpg": NOT_A_RECEIPT})
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "photo.jpg", _jpeg_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert result is None


def test_extract_attachment_corrupt_pdf_becomes_an_error_not_a_crash():
    lm = _FakeLMStudio()
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "broken.pdf", CORRUPT_PDF_BYTES, system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.AttachmentError)
    assert result.attachment == "broken.pdf"


def test_extract_attachment_bad_amount_becomes_an_error_not_a_crash():
    bad_amount = dict(RECEIPT_RESULT, amount="not a number")
    lm = _FakeLMStudio(see_results={"receipt.jpg": bad_amount})
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "receipt.jpg", _jpeg_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.AttachmentError)


def test_extract_attachment_model_exception_becomes_an_error_not_a_crash():
    lm = _FakeLMStudio(see_results={"receipt.jpg": RuntimeError("model server down")})
    msg = _msg("m1")
    result = attachments.extract_attachment(
        lm, msg, "receipt.jpg", _jpeg_bytes(), system_image=SYS[0], system_pdf=SYS[1]
    )
    assert isinstance(result, attachments.AttachmentError)
    assert "model server down" in result.error


# ---------------------------------------------------------------------------
# process_message -- per-message attachment loop
# ---------------------------------------------------------------------------


class _FakeGmailAttachments:
    def __init__(self, payloads: dict):
        self._payloads = payloads
        self.calls: list[tuple[str, int]] = []

    def attachment_bytes(self, msg, index):
        self.calls.append((msg.msg_id, index))
        result = self._payloads[(msg.msg_id, index)]
        if isinstance(result, Exception):
            raise result
        return result


def test_process_message_stops_at_first_receipt_and_skips_the_rest():
    msg = _msg("m1", attachments=[
        Attachment(name="junk.zip", mime="application/zip", size=10),
        Attachment(name="receipt.jpg", mime="image/jpeg", size=10),
        Attachment(name="second.jpg", mime="image/jpeg", size=10),
    ])
    gmail = _FakeGmailAttachments({
        ("m1", 0): b"PK\x03\x04 not an image or pdf",
        ("m1", 1): _jpeg_bytes(),
        ("m1", 2): _jpeg_bytes(),
    })
    lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})

    hit, errors = attachments.process_message(gmail, lm, msg, system_image=SYS[0], system_pdf=SYS[1])

    assert hit is not None
    assert hit.attachment_name == "receipt.jpg"
    assert ("m1", 2) not in gmail.calls  # never tried once a hit was found


def test_process_message_skips_oversized_attachment_without_fetching():
    from lmsa.sources.gmail import ATTACHMENT_MAX_BYTES
    msg = _msg("m1", attachments=[Attachment(name="huge.pdf", mime="application/pdf", size=ATTACHMENT_MAX_BYTES + 1)])
    gmail = _FakeGmailAttachments({})  # no payload configured -- must never be called
    lm = _FakeLMStudio()

    hit, errors = attachments.process_message(gmail, lm, msg, system_image=SYS[0], system_pdf=SYS[1])

    assert hit is None
    assert gmail.calls == []
    assert len(errors) == 1
    assert "over the" in errors[0].error


def test_process_message_records_fetch_failure_and_continues():
    from lmsa.sources.gmail import AttachmentNotFoundError

    msg = _msg("m1", attachments=[
        Attachment(name="gone.pdf", mime="application/pdf", size=10),
        Attachment(name="receipt.jpg", mime="image/jpeg", size=10),
    ])
    gmail = _FakeGmailAttachments({
        ("m1", 0): AttachmentNotFoundError("message not found"),
        ("m1", 1): _jpeg_bytes(),
    })
    lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})

    hit, errors = attachments.process_message(gmail, lm, msg, system_image=SYS[0], system_pdf=SYS[1])

    assert hit is not None
    assert hit.attachment_name == "receipt.jpg"
    assert len(errors) == 1
    assert errors[0].attachment == "gone.pdf"


def test_process_message_returns_none_when_no_attachment_is_a_receipt():
    msg = _msg("m1", attachments=[Attachment(name="photo.jpg", mime="image/jpeg", size=10)])
    gmail = _FakeGmailAttachments({("m1", 0): _jpeg_bytes()})
    lm = _FakeLMStudio(see_results={"photo.jpg": NOT_A_RECEIPT})

    hit, errors = attachments.process_message(gmail, lm, msg, system_image=SYS[0], system_pdf=SYS[1])

    assert hit is None
    assert errors == []


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def test_build_report_renders_thumbnail_and_totals():
    receipt = attachments.ExtractedReceipt(
        msg_id="m1", merchant="Acme Cloud", amount_minor=1999, currency="USD",
        purchase_date="2026-01-15", category="software", recurring_hint=False, subject="Your receipt",
    )
    hit = attachments.ExtractedAttachment(
        receipt=receipt, source="image-vision", attachment_name="receipt.jpg",
        thumbnail="data:image/jpeg;base64,QQ==",
    )
    html = attachments.build_report([hit], errors=[]).render()
    assert "19.99 USD" in html
    assert '<img class="thumb" src="data:image/jpeg;base64,QQ=="' in html
    assert "Acme Cloud" in html


def _pending_window(pending: int):
    from lmsa.gmail_cursor import Window
    from lmsa.sources.gmail import MessageMeta

    metas = [MessageMeta(msg_id=str(i), internal_date=None, uid=i) for i in range(pending + 1)]
    return Window(candidates=metas, batch=metas[:1])


def test_build_report_lists_errors_and_notes_pending():
    err = attachments.AttachmentError(msg_id="m1", subject="s", attachment="broken.pdf", error="boom")
    html = attachments.build_report([], errors=[err], window=_pending_window(2)).render()
    assert "boom" in html
    assert "2 more matching message(s) are pending" in html


# ---------------------------------------------------------------------------
# main() -- end to end against fakes
# ---------------------------------------------------------------------------


class _FakeGmail(MetaSearchMixin):
    """Combines search() (epoch-second window filtering, matching the real
    Gmail after:/before: semantics tasks.attachments now relies on -- see
    contracts.md's cursor lesson) with attachment_bytes() (by (msg_id, index))."""

    def __init__(self, messages, payloads):
        self._messages = messages
        self._payloads = payloads
        self.search_queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.search_queries.append(query)
        after_ts = int(query.split("after:")[1].split()[0])
        before_ts = int(query.split("before:")[1].split()[0]) if "before:" in query else None
        matches = [
            m for m in self._messages
            if int(m.date.timestamp()) >= after_ts
            and (before_ts is None or int(m.date.timestamp()) < before_ts)
        ]
        matches.sort(key=lambda m: m.date, reverse=True)
        return matches[:limit]

    def attachment_bytes(self, msg, index):
        result = self._payloads[(msg.msg_id, index)]
        if isinstance(result, Exception):
            raise result
        return result


def test_main_end_to_end_is_idempotent_across_two_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    msg = _msg(
        "m1", subject="Your receipt", day=_days_ago(5),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    fake_gmail = _FakeGmail([msg], {("m1", 0): _jpeg_bytes()})
    monkeypatch.setattr(attachments, "Gmail", lambda cfg=None: fake_gmail)

    fake_lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(attachments, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["attachments"])

    attachments.main()
    attachments.main()  # re-run: cursor has moved on, nothing new to fetch

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        merchant = store.conn.execute("SELECT merchant FROM receipts").fetchone()[0]
    assert count == 1
    assert merchant == "Acme Cloud"

    written = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert written


def test_main_skips_messages_with_no_attachments(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    msg = _msg("m1", subject="Your receipt", day=_days_ago(5), attachments=[])
    fake_gmail = _FakeGmail([msg], {})
    monkeypatch.setattr(attachments, "Gmail", lambda cfg=None: fake_gmail)

    fake_lm = _FakeLMStudio()
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(attachments, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["attachments"])

    attachments.main()  # must not blow up or call the (unconfigured) fake LM

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# ledger-overwrite guard -- the attachment path must never clobber a row
# tasks.receipts already wrote for the same msg_id
# ---------------------------------------------------------------------------


def test_main_never_overwrites_a_row_tasks_receipts_already_wrote(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    # Simulate tasks.receipts having already landed a row for this message
    # from its body, before the attachment scan ever runs (order A).
    body_row = attachments.ExtractedReceipt(
        msg_id="m1", merchant="Body Merchant", amount_minor=500, currency="USD",
        purchase_date="2026-01-10", category="body", recurring_hint=False, subject="Your receipt",
    )
    with Store(tmp_path / "lmsa.sqlite") as store:
        attachments._ensure_schema(store)
        attachments.upsert_receipt(store, body_row)
        store.conn.commit()

    msg = _msg(
        "m1", subject="Your receipt", day=_days_ago(5),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    fake_gmail = _FakeGmail([msg], {("m1", 0): _jpeg_bytes()})
    monkeypatch.setattr(attachments, "Gmail", lambda cfg=None: fake_gmail)

    # Would say $19.99 / Acme Cloud if it ever ran -- it must not run at all.
    fake_lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(attachments, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["attachments"])

    attachments.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        count = store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        merchant, amount = store.conn.execute(
            "SELECT merchant, amount_minor FROM receipts WHERE msg_id='m1'"
        ).fetchone()
    assert count == 1
    assert merchant == "Body Merchant"  # unchanged -- the attachment's data never landed
    assert amount == 500
    assert fake_lm.see_calls == []  # never even ran the vision call for an already-ledgered message

    reports = [p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"]
    assert reports
    assert "already in ledger from body" in reports[0].read_text(encoding="utf-8")


def test_receipts_upsert_never_overwrites_an_attachment_row(tmp_path):
    # Order B (attachments first, tasks.receipts second): first writer owns
    # the shared row, so the attachment's read survives.
    db = tmp_path / "lmsa.sqlite"
    from_attachment = attachments.ExtractedReceipt(
        msg_id="m1", merchant="Attachment Merchant", amount_minor=1999, currency="USD",
        purchase_date="2026-01-15", category="software", recurring_hint=False, subject="Your receipt",
    )
    from_body = attachments.ExtractedReceipt(
        msg_id="m1", merchant="Body Merchant", amount_minor=500, currency="USD",
        purchase_date="2026-01-10", category="body", recurring_hint=False, subject="Your receipt",
    )
    with Store(db) as store:
        attachments._ensure_schema(store)
        assert attachments.upsert_receipt(store, from_attachment) is True  # attachments.py runs first
        assert receipts_task.upsert_receipt(store, from_body) is False     # tasks.receipts runs second
        store.conn.commit()
        merchant = store.conn.execute("SELECT merchant FROM receipts WHERE msg_id='m1'").fetchone()[0]
    assert merchant == "Attachment Merchant"


# ---------------------------------------------------------------------------
# epoch-second cursor -- no same-day mail lost, boundary re-see is harmless
# ---------------------------------------------------------------------------


def test_epoch_cursor_catches_same_day_mail_without_reprocessing_the_boundary(tmp_path, monkeypatch):
    """Regression test for contracts.md's cursor lesson: Gmail's after:/before:
    are day-granularity with a plain date, so a catch-up boundary set to the
    *date* of the oldest message would silently lose earlier same-day mail.
    With the shared scheme (epoch bounds, oldest unprocessed first, cursor
    never past an unprocessed message), both same-day messages get processed
    across the two runs -- and nothing is sent to the vision model twice.
    """
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))

    day = _days_ago(2)
    morning = _msg(
        "morning", subject="Your receipt", day=day,
        date=datetime.fromisoformat(f"{day}T08:00:00+00:00"),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    evening = _msg(
        "evening", subject="Your receipt", day=day,
        date=datetime.fromisoformat(f"{day}T18:00:00+00:00"),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    fake_gmail = _FakeGmail([morning, evening], {
        ("morning", 0): _jpeg_bytes(),
        ("evening", 0): _jpeg_bytes(),
    })
    monkeypatch.setattr(attachments, "Gmail", lambda cfg=None: fake_gmail)

    fake_lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(attachments, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["attachments", "--limit", "1"])

    attachments.main()  # limit=1: processes the oldest ("morning") only
    with Store(tmp_path / "lmsa.sqlite") as store:
        assert store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 1
    assert len(fake_lm.see_calls) == 1

    attachments.main()  # the cursor held at "evening", which is processed now
    with Store(tmp_path / "lmsa.sqlite") as store:
        msg_ids = {row[0] for row in store.conn.execute("SELECT msg_id FROM receipts").fetchall()}
    assert msg_ids == {"morning", "evening"}  # same-day mail wasn't lost
    assert len(fake_lm.see_calls) == 2  # but "evening" wasn't re-sent to the vision model


# ---------------------------------------------------------------------------
# shared cursor scheme -- retry set, applied ledger, legacy migration
# ---------------------------------------------------------------------------


class _RetryableFakeGmail(_FakeGmail):
    """Adds `fetch_by_msgids` (the retry path's refetch by X-GM-MSGID)."""

    def __init__(self, messages, payloads):
        super().__init__(messages, payloads)
        self.refetched: list[list[str]] = []

    def fetch_by_msgids(self, msg_ids, folder="[Gmail]/All Mail", *, headers_only=False):
        self.refetched.append(list(msg_ids))
        return [m for i in msg_ids for m in self._messages if m.msg_id == i]


def _run(monkeypatch, tmp_path, fake_gmail, fake_lm):
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    fake_lm.cfg.output_dir = tmp_path / "output"
    monkeypatch.setattr(attachments, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(attachments, "LMStudio", lambda: fake_lm)
    monkeypatch.setattr("sys.argv", ["attachments"])
    attachments.main()


def test_transient_failure_is_not_marked_processed_and_is_retried_next_run(tmp_path, monkeypatch):
    msg = _msg(
        "m1", subject="Your receipt", day=_days_ago(3),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    payloads = {("m1", 0): OSError("connection reset mid-fetch")}
    fake_gmail = _RetryableFakeGmail([msg], payloads)
    fake_lm = _FakeLMStudio(see_results={"receipt.jpg": RECEIPT_RESULT})

    _run(monkeypatch, tmp_path, fake_gmail, fake_lm)  # run 1: fetch fails
    with Store(tmp_path / "lmsa.sqlite") as store:
        assert not attachments._is_applied(store, "m1")
        assert store.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 0

    payloads[("m1", 0)] = _jpeg_bytes()  # the transient problem clears
    _run(monkeypatch, tmp_path, fake_gmail, fake_lm)  # run 2: cursor moved past m1, retry refetches it
    assert fake_gmail.refetched == [["m1"]]
    with Store(tmp_path / "lmsa.sqlite") as store:
        assert attachments._is_applied(store, "m1")
        assert store.conn.execute("SELECT merchant FROM receipts WHERE msg_id='m1'").fetchone()[0] == "Acme Cloud"


def test_permanent_skip_is_concluded_not_retried(tmp_path, monkeypatch):
    huge = attachments.ATTACHMENT_MAX_BYTES + 1
    msg = _msg(
        "m1", subject="Your receipt", day=_days_ago(3),
        attachments=[Attachment(name="huge.pdf", mime="application/pdf", size=huge)],
    )
    fake_gmail = _RetryableFakeGmail([msg], {})
    _run(monkeypatch, tmp_path, fake_gmail, _FakeLMStudio())
    _run(monkeypatch, tmp_path, fake_gmail, _FakeLMStudio())
    assert fake_gmail.refetched == []  # nothing queued for retry
    with Store(tmp_path / "lmsa.sqlite") as store:
        assert attachments._is_applied(store, "m1")


def test_gives_up_after_max_attempts_and_says_so(tmp_path, monkeypatch):
    msg = _msg(
        "m1", subject="Flaky receipt", day=_days_ago(3),
        attachments=[Attachment(name="receipt.jpg", mime="image/jpeg", size=10)],
    )
    fake_gmail = _RetryableFakeGmail([msg], {("m1", 0): OSError("still down")})
    for _ in range(3):
        _run(monkeypatch, tmp_path, fake_gmail, _FakeLMStudio())
    latest = max(
        (p for p in (tmp_path / "output").rglob("*.html") if p.name != "index.html"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    assert "Gave up after 3 attempts" in latest.read_text(encoding="utf-8")
    _run(monkeypatch, tmp_path, fake_gmail, _FakeLMStudio())
    assert fake_gmail.refetched == [["m1"], ["m1"]]  # runs 2 and 3 only -- never after giving up


def test_legacy_processed_msg_ids_migrate_into_the_applied_ledger(tmp_path):
    import json as _json

    with Store(tmp_path / "lmsa.sqlite") as store:
        store.conn.execute(
            "INSERT INTO kv(ns, key, value_json, updated) VALUES (?, ?, ?, ?)",
            (attachments.KV_NS, "processed_msg_ids", _json.dumps(["a", "b"]), "now"),
        )
        store.conn.commit()
        attachments._ensure_applied_schema(store)
        assert attachments._is_applied(store, "a") and attachments._is_applied(store, "b")
        assert attachments._kv_get(store, "processed_msg_ids") is None
