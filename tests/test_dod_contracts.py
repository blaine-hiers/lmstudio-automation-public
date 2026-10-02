"""DoD contract-award ledger adapter (#57): parser, amount parsing, ledger
idempotency and the opt-in report extension. No network, no LM Studio."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmsa import Store
from lmsa.briefing.adapters import dod_contracts as dc
from lmsa.briefing.sources import SourceContext
from lmsa.report import Report
from lmsa.web import FetchError

FIX = Path(__file__).parent / "fixtures" / "briefing" / "dod_contracts_page.html"
PAGE_URL = "https://www.defense.gov/News/Contracts/"
NOW = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url not in self.pages:
            raise FetchError(f"no fixture for {url}")
        return SimpleNamespace(text=self.pages[url])


class FakeLM:
    """extract() lifts contractor/location off the leading '<name>, <city>,
    <state>,' pattern -- good enough to prove the memoized model call path
    without needing a real model. embed() returns a fixed vector regardless
    of input, good enough to prove vectors get written under the right
    ns/model/id without a real embedding model."""

    def __init__(self):
        self.cfg = SimpleNamespace(chat_model="fake-model", embed_model="fake-embed-model")
        self.calls = 0
        self.embed_calls = 0

    def extract(self, prompt, schema, *, system=None, model=None, name="result"):
        self.calls += 1
        # the paragraph is embedded verbatim after the "Untrusted paragraph" marker
        text = prompt.split("\n\n", 1)[-1]
        parts = [p.strip() for p in text.split(",")]
        contractor = parts[0] if parts else ""
        location = ", ".join(parts[1:3]) if len(parts) > 2 else ""
        return {"contractor": contractor, "location": location, "program_or_item": "widget"}

    def embed(self, texts, *, model=None):
        self.embed_calls += len(texts)
        return [[1.0, 0.0] for _ in texts]


def make_ctx(conn, fetcher, lm, *, now=NOW, limit=100, url=None, fresh=False):
    entry = {"name": "DoD contract awards", "adapter": "dod_contracts"}
    if url:
        entry["url"] = url
    ctx = SourceContext(briefing=SimpleNamespace(name="defense"), fetcher=fetcher, lm=lm, conn=conn,
                        now=now, limit=limit, fresh=fresh)
    return entry, ctx


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_split_paragraphs_and_grouping_on_saved_fixture():
    html = FIX.read_text(encoding="utf-8")
    paragraphs = dc.split_paragraphs(html)
    assert len(paragraphs) == 10  # h1 isn't a <p>; 10 <p> blocks are

    grouped = dc.group_by_service(paragraphs)
    awards = [(svc, p) for svc, p in grouped if dc.is_award_paragraph(p)]
    assert len(awards) == 5
    services = [svc for svc, _ in awards]
    assert services == ["ARMY", "ARMY", "NAVY", "AIR FORCE", "DEFENSE LOGISTICS AGENCY"]
    # the closing boilerplate paragraph is grouped but never counted as an award
    assert not dc.is_award_paragraph("This is closing boilerplate about press inquiries.")


def test_multi_contractor_and_modification_paragraph():
    html = FIX.read_text(encoding="utf-8")
    paragraphs = dc.split_paragraphs(html)
    mod = next(p for p in paragraphs if "General Dynamics" in p)
    assert dc.is_modification(mod)
    assert "BAE Systems" in mod  # multi-contractor award, both named in one paragraph
    amount, note = dc.parse_amount(mod)
    assert amount == 1_340_000_000.0  # cumulative total wins over the $210.5M modification figure
    assert note == "cumulative total"
    assert dc.parse_contract_number(mod) == "W56HZV-22-D-0010"  # not the modification number P00012


@pytest.mark.parametrize("text,expected,expected_note", [
    ("was awarded a $1,234,567 firm-fixed-price contract", 1_234_567.0, ""),
    ("is awarded a $1.2 billion cost-plus contract", 1_200_000_000.0, ""),
    ("modification bringing the total cumulative face value to $12,345,678,901 for sustainment",
     12_345_678_901.0, "cumulative total"),
    ("a maximum $45,000,000 ceiling requirements contract", 45_000_000.0, "ceiling"),
    ("was awarded a contract for services with no dollar figure stated", None, "no amount found by regex"),
    # realistic IDIQ wording, obligated figure before the ceiling in the text
    ("Fiscal 2026 funds in the amount of $500,000 are being obligated at time of award. "
     "The maximum dollar value of this multiple-award contract is $950,000,000.",
     950_000_000.0, "ceiling"),
    # same wording, order reversed -- the ceiling must still win regardless
    # of which figure the text states first
    ("The maximum dollar value of this multiple-award contract is $950,000,000. "
     "Fiscal 2026 funds in the amount of $500,000 are being obligated at time of award.",
     950_000_000.0, "ceiling"),
    # only an obligated figure is stated anywhere -- no ceiling/maximum to
    # prefer, so the fallback figure is used but flagged, not silently
    # presented as the contract's award value
    ("Fiscal 2026 funds in the amount of $500,000 are being obligated at time of award for this task order.",
     500_000.0, "obligated amount, not a stated ceiling/maximum"),
])
def test_parse_amount_formats(text, expected, expected_note):
    amount, note = dc.parse_amount(text)
    assert amount == expected
    assert note == expected_note


def test_parse_contract_number_missing_is_empty_not_invented():
    text = "Boeing Co. was awarded a $12,750,000 contract. No parenthetical here."
    assert dc.parse_contract_number(text) == ""


def test_unrecognized_nonempty_page_is_disclosed_not_zero_awards():
    """A page with <p> content but nothing matching an award verb + '$' is a
    parse failure to disclose, distinct from a legitimate zero-award day."""
    html = "<html><body><p>Site maintenance notice.</p><p>Please check back later.</p></body></html>"
    paragraphs = dc.split_paragraphs(html)
    assert paragraphs  # non-empty page
    awards = [p for p in paragraphs if dc.is_award_paragraph(p)]
    assert awards == []


# ---------------------------------------------------------------------------
# Adapter + ledger idempotency
# ---------------------------------------------------------------------------

def test_adapter_records_awards_and_yields_no_docs(tmp_path):
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        docs = list(dc.dod_contracts_adapter(entry, ctx))
        assert docs == []  # side-effect adapter: nothing feeds the story pipeline
        assert lm.calls == 5  # one model call per award paragraph

        rows = store.conn.execute("SELECT contract_number, day, amount_usd, service FROM dod_contracts").fetchall()
        assert len(rows) == 5
        by_service = sorted(r[3] for r in rows)
        assert by_service == ["AIR FORCE", "ARMY", "ARMY", "DEFENSE LOGISTICS AGENCY", "NAVY"]
        assert any(r[0] == "" for r in rows)  # the Air Force paragraph had no parenthetical


def test_award_embeds_into_contracts_namespace(tmp_path):
    """#68: each recorded award gets a vector in the `contracts` namespace,
    item id `<day>:<identity_key>`, under the same embedding model
    `lm.cfg.embed_model` the rest of the app uses."""
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))

        row = store.conn.execute(
            "SELECT ns, model FROM embeddings WHERE item_id = ?", ("2026-09-24:W31P4Q-26-C-0045",),
        ).fetchone()
        assert row == ("contracts", "fake-embed-model")

        count = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns = 'contracts'").fetchone()[0]
        assert count == 5  # one vector per recorded award


def test_republication_still_one_vector(tmp_path):
    """A corrected re-publication of the same day's award (same identity
    key) must overwrite its existing vector, not add a second one."""
    html = FIX.read_text(encoding="utf-8")
    corrected = html.replace("M270 rocket launcher upgrades.", "M270A2 rocket launcher upgrades.")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))

        fetcher2 = FakeFetcher({PAGE_URL: corrected})
        lm2 = FakeLM()
        entry2, ctx2 = make_ctx(store.conn, fetcher2, lm2, now=NOW)
        list(dc.dod_contracts_adapter(entry2, ctx2))

        count = store.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE ns = 'contracts' AND item_id = ?",
            ("2026-09-24:W31P4Q-26-C-0045",),
        ).fetchone()[0]
        assert count == 1


def test_rerun_same_day_unchanged_text_does_not_reembed(tmp_path):
    """A rerun of the same day's unchanged page must not re-embed rows that
    haven't changed."""
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))
        assert lm.embed_calls == 5

        fetcher2 = FakeFetcher({PAGE_URL: html})
        lm2 = FakeLM()
        entry2, ctx2 = make_ctx(store.conn, fetcher2, lm2, now=NOW)
        list(dc.dod_contracts_adapter(entry2, ctx2))
        assert lm2.embed_calls == 0  # nothing changed -- no embedding calls at all


def test_embed_failure_does_not_block_ledger_write(tmp_path):
    """Embedding is best-effort: a failed embed() call must not prevent the
    ledger row from being written, and must be disclosed as a note."""
    html = FIX.read_text(encoding="utf-8")

    class FailingEmbedLM(FakeLM):
        def embed(self, texts, *, model=None):
            self.embed_calls += len(texts)
            raise RuntimeError("embedding service unavailable")

    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FailingEmbedLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        docs = list(dc.dod_contracts_adapter(entry, ctx))
        assert docs == []

        rows = store.conn.execute("SELECT COUNT(*) FROM dod_contracts").fetchone()[0]
        assert rows == 5  # every ledger row still written despite embedding failures

        count = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns = 'contracts'").fetchone()[0]
        assert count == 0  # nothing embedded, but that's the only casualty

        assert any("embedding failed" in n for n in ctx.notes)

        # the next run of the same page, with embedding back, catches up
        lm2 = FakeLM()
        entry, ctx2 = make_ctx(store.conn, fetcher, lm2)
        list(dc.dod_contracts_adapter(entry, ctx2))
        count = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE ns = 'contracts'").fetchone()[0]
        assert count == 5
        assert store.conn.execute("SELECT COUNT(*) FROM dod_contracts").fetchone()[0] == 5


def test_ledger_idempotent_across_reruns_same_day(tmp_path):
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))
        first_seen = {r[0]: r[1] for r in store.conn.execute(
            "SELECT contract_number || '/' || paragraph_hash, first_seen FROM dod_contracts")}

        # rerun same day, same page: no new rows, first_seen preserved, model
        # calls are memoized by paragraph hash (no new extract() calls)
        fetcher2 = FakeFetcher({PAGE_URL: html})
        lm2 = FakeLM()
        entry2, ctx2 = make_ctx(store.conn, fetcher2, lm2, now=NOW)
        list(dc.dod_contracts_adapter(entry2, ctx2))

        rows = store.conn.execute("SELECT contract_number || '/' || paragraph_hash, first_seen FROM dod_contracts").fetchall()
        assert len(rows) == 5
        assert dict(rows) == first_seen
        assert lm2.calls == 0


def test_corrected_republication_upserts_not_duplicates(tmp_path):
    """A same-day re-publication of the same award with a word or two
    corrected must update the existing ledger row, not add a second one
    that doubles every total -- the #57 review's repro turned $58.4M into
    $116.9M this way. Identity is (day, contract_number), not the paragraph
    text/hash, so a reworded but otherwise-identical announcement upserts."""
    html = FIX.read_text(encoding="utf-8")
    corrected = html.replace(
        "M270 rocket launcher upgrades.", "M270A2 rocket launcher upgrades.")
    assert corrected != html  # sanity: the word replacement actually landed
    assert dc.paragraph_hash(corrected) != dc.paragraph_hash(html)

    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))

        # re-publication of the same day's page, one word corrected in the
        # Lockheed paragraph; contract number and day are unchanged
        fetcher2 = FakeFetcher({PAGE_URL: corrected})
        lm2 = FakeLM()
        entry2, ctx2 = make_ctx(store.conn, fetcher2, lm2, now=NOW)
        list(dc.dod_contracts_adapter(entry2, ctx2))

        contract_rows = store.conn.execute(
            "SELECT COUNT(*), SUM(amount_usd), MAX(program_or_item) FROM dod_contracts "
            "WHERE contract_number = ?", ("W31P4Q-26-C-0045",),
        ).fetchone()
        assert contract_rows[0] == 1  # still one row for this contract
        assert contract_rows[1] == 58_432_100.0  # total did not double

        total_rows = store.conn.execute("SELECT COUNT(*) FROM dod_contracts").fetchone()[0]
        assert total_rows == 5  # still 5 distinct awards, not 6

        # the window totals used by the report must reflect the same,
        # un-doubled figure
        report = Report("Defense industry")
        dc.render_ledger_section(report, [], ctx2)
        html_out = report.render() if hasattr(report, "render") else str(report)
        assert "116,864,200" not in html_out  # the doubled figure must not appear
        assert "58,432,100" in html_out


def test_bad_paragraph_never_kills_the_run(tmp_path, monkeypatch):
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})

        class FlakyLM(FakeLM):
            def extract(self, *a, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom")
                return super().extract(*a, **kw)

        lm = FlakyLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        docs = list(dc.dod_contracts_adapter(entry, ctx))
        assert docs == []
        rows = store.conn.execute("SELECT COUNT(*) FROM dod_contracts").fetchone()[0]
        assert rows == 4  # 5 awards, one paragraph's model call failed and was skipped
        assert any("skipped" in n for n in ctx.notes)


def test_entry_limit_caps_awards_recorded(tmp_path):
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        # engine sets ctx.limit = kept cap + 1; cap=2 here
        entry, ctx = make_ctx(store.conn, fetcher, lm, limit=3)
        list(dc.dod_contracts_adapter(entry, ctx))
        rows = store.conn.execute("SELECT COUNT(*) FROM dod_contracts").fetchone()[0]
        assert rows == 2
        assert any("cut at 2" in n for n in ctx.notes)


# ---------------------------------------------------------------------------
# Discovery via a listing/RSS page
# ---------------------------------------------------------------------------

def test_discovers_day_page_from_listing_links(tmp_path):
    listing = (
        '<html><body><ul>'
        '<li><a href="/News/Contracts/Contract/Article/1000001/">Sept. 23, 2026</a></li>'
        '<li><a href="/News/Contracts/Contract/Article/1000002/">Sept. 24, 2026</a></li>'
        '</ul></body></html>'
    )
    day_url = "https://www.defense.gov/News/Contracts/Contract/Article/1000002/"
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: listing, day_url: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm, now=NOW)
        list(dc.dod_contracts_adapter(entry, ctx))
        assert day_url in fetcher.calls
        rows = store.conn.execute("SELECT COUNT(*) FROM dod_contracts WHERE source_url = ?", (day_url,)).fetchone()[0]
        assert rows == 5


def test_fetch_failure_is_a_note_not_a_crash(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({})  # no pages at all -> FetchError
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        docs = list(dc.dod_contracts_adapter(entry, ctx))
        assert docs == []
        assert any("could not fetch" in n for n in ctx.notes)


# ---------------------------------------------------------------------------
# Report extension
# ---------------------------------------------------------------------------

def test_render_ledger_section_reads_the_table(tmp_path):
    html = FIX.read_text(encoding="utf-8")
    with Store(tmp_path / "lmsa.sqlite") as store:
        fetcher = FakeFetcher({PAGE_URL: html})
        lm = FakeLM()
        entry, ctx = make_ctx(store.conn, fetcher, lm)
        list(dc.dod_contracts_adapter(entry, ctx))

        report = Report("Defense industry")
        dc.render_ledger_section(report, [], ctx)
        html_out = report.render() if hasattr(report, "render") else str(report)
        assert "DoD contract awards" in html_out


def test_render_ledger_section_empty_day_is_not_an_error(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        entry, ctx = make_ctx(store.conn, FakeFetcher({}), FakeLM())
        report = Report("Defense industry")
        dc.render_ledger_section(report, [], ctx)  # no rows for today: must not raise
