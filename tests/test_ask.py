from datetime import date

from lmsa.store import Store
from lmsa.vectors import upsert as vectors_upsert
from tasks import ask
from tasks.ask import (
    AskContext,
    Match,
    NOT_FOUND,
    ScopeHit,
    answer_query,
    build_matches,
    build_report,
)


def _seed_message(conn, msg_id, *, subject, from_addr, date, web_url, excerpt):
    conn.execute(
        "INSERT INTO messages(msg_id, thread_id, date, from_addr, subject, labels_json, seen_at) "
        "VALUES (?, ?, ?, ?, ?, '[]', ?)",
        (msg_id, f"{msg_id}-thread", date, from_addr, subject, "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mail_index "
        "(msg_id TEXT PRIMARY KEY, web_url TEXT NOT NULL, excerpt TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO mail_index(msg_id, web_url, excerpt) VALUES (?, ?, ?)",
        (msg_id, web_url, excerpt),
    )
    conn.commit()


def test_build_matches_joins_messages_and_mail_index(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_message(
            store.conn, "m1", subject="Roof budget", from_addr="alice@example.com",
            date="2026-09-01T12:00:00+00:00", web_url="https://mail.google.com/x", excerpt="Budget is $500.",
        )
        matches = build_matches(store.conn, [("m1", 0.9)])

    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, Match)
    assert m.number == 1
    assert m.subject == "Roof budget"
    assert m.from_addr == "alice@example.com"
    assert m.web_url == "https://mail.google.com/x"
    assert m.excerpt == "Budget is $500."
    assert m.score == 0.9


def test_build_matches_numbers_sequentially_in_hit_order(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_message(store.conn, "m1", subject="A", from_addr="a@example.com",
                       date=None, web_url="", excerpt="text a")
        _seed_message(store.conn, "m2", subject="B", from_addr="b@example.com",
                       date=None, web_url="", excerpt="text b")
        matches = build_matches(store.conn, [("m2", 0.8), ("m1", 0.5)])

    assert [m.number for m in matches] == [1, 2]
    assert [m.msg_id for m in matches] == ["m2", "m1"]


def test_build_matches_survives_missing_metadata_row(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.execute(
            "CREATE TABLE IF NOT EXISTS mail_index "
            "(msg_id TEXT PRIMARY KEY, web_url TEXT NOT NULL, excerpt TEXT NOT NULL)"
        )
        matches = build_matches(store.conn, [("gone", 0.5)])

    assert matches[0].excerpt is None
    assert "re-run inbox_index" in matches[0].subject


class FakeLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append((prompt, schema, system, name))
        if self.error:
            raise self.error
        return self.result


def _match(number=1, excerpt="Budget is $500.", subject="Roof budget", from_addr="alice@example.com"):
    return Match(number=number, msg_id=f"m{number}", subject=subject, from_addr=from_addr,
                 date="2026-09-01T12:00:00+00:00", web_url="https://mail.google.com/x",
                 score=0.9, excerpt=excerpt)


def test_answer_query_returns_grounded_answer_and_citations():
    matches = [_match()]
    lm = FakeLM(result={"answer": "The roof budget is $500.", "citations": [1]})

    answer, citations, error = answer_query(lm, "what is the roof budget", matches)

    assert answer == "The roof budget is $500."
    assert citations == [1]
    assert error is None
    assert len(lm.calls) == 1


def test_answer_query_never_raises_on_model_failure():
    matches = [_match()]
    lm = FakeLM(error=RuntimeError("model down"))

    answer, citations, error = answer_query(lm, "question", matches)

    assert answer is None
    assert citations == []
    assert "model down" in error


def test_answer_query_says_not_found_when_no_usable_excerpts():
    matches = [_match(excerpt=None)]
    lm = FakeLM(result={"answer": "should not be called", "citations": []})

    answer, citations, error = answer_query(lm, "question", matches)

    assert answer == NOT_FOUND
    assert citations == []
    assert error is None
    assert lm.calls == []  # no wasted model call when there's nothing to ground on


def test_build_report_no_matches_shows_warning():
    html = build_report("some query", []).render()
    assert "No indexed messages matched" in html


def test_build_report_shows_answer_and_citations_and_source_cards():
    matches = [_match()]
    html = build_report(
        "what is the roof budget", matches, answer="The roof budget is $500.", citations=[1],
    ).render()
    assert "The roof budget is $500." in html
    assert "[1]" in html
    assert "Roof budget" in html
    assert "https://mail.google.com/x" in html


def test_build_report_never_dumps_full_excerpt_only_in_source_card_meta():
    long_excerpt = "SECRET " * 500
    matches = [_match(excerpt=long_excerpt)]
    html = build_report("secret", matches).render()
    # the excerpt itself is model input, not rendered verbatim into the report
    assert long_excerpt not in html


def test_build_report_shows_answer_error_when_unanswered():
    html = build_report("q", [], answer_error="model down").render()
    assert "model down" in html


def test_build_report_escapes_untrusted_subject_and_drops_unsafe_href():
    matches = [_match(subject="<script>alert(1)</script>")]
    matches[0].web_url = "javascript:alert(1)"
    html = build_report("q", matches).render()
    assert "<script>alert(1)</script>" not in html
    assert "javascript:alert(1)" not in html


# ---------------------------------------------------------------------------
# Multi-scope registry (#68)
# ---------------------------------------------------------------------------

class FakeEmbedLM:
    """A `.embed()` stand-in for `lmsa.vectors.search` -- always returns the
    same fixed vector, whatever the query text, so cosine similarity against
    seeded embeddings is deterministic."""

    def __init__(self, vector):
        self.vector = vector

    def embed(self, texts, *, model=None):
        return [self.vector for _ in texts]


def test_since_none_leaves_gmail_matches_unfiltered():
    """#68: the --since filter is a no-op when --since is omitted (the CLI
    default) -- this is what keeps `--scope gmail` byte-identical to the
    pre-#68 behaviour, since `build_matches`/`answer_query`/`build_report`
    above are untouched by this ticket."""
    matches = [_match(number=1), _match(number=2, subject="B")]
    assert ask._apply_since(matches, None) == matches
    assert ask._apply_since(matches, None) is matches


def test_apply_since_drops_older_gmail_matches():
    old = _match(number=1, subject="old")
    old.date = "2026-01-01T00:00:00+00:00"
    new = _match(number=2, subject="new")
    new.date = "2026-09-01T00:00:00+00:00"
    kept = ask._apply_since([old, new], date(2026, 6, 1))
    assert [m.subject for m in kept] == ["new"]


def test_scope_choices_cover_the_registered_scopes_plus_all():
    assert set(ask.registered_scopes()) == {"gmail", "briefings", "docs", "markets", "contracts"}
    assert set(ask.SCOPE_CHOICES) == set(ask.registered_scopes()) | {"all"}


def test_register_scope_lets_a_new_scope_join_without_touching_dispatch():
    """A future scope only needs to call `register_scope` -- `--scope all`
    picks it up with no change to `_search_all` or any other dispatch
    code."""
    calls = []

    def fake_scope(ctx, query, k):
        calls.append((query, k))
        return [ScopeHit(number=0, scope="widgets", title="A widget",
                          meta="widgets", href="https://example.com/w", score=0.99, excerpt="widget text")]

    ask.register_scope("widgets", fake_scope)
    try:
        ctx = AskContext(conn=None, lm=None, embed_model="m", since=None, output_dir=None)
        hits = ask._search_all(ctx, "q", 5)
    finally:
        del ask._SCOPES["widgets"]
    assert calls == [("q", 5)]
    assert any(h.scope == "widgets" and h.score == 0.99 for h in hits)


def _seed_briefing_story(conn, *, briefing, story_id, headline, summary, run_id, last_seen, report_path):
    item_id = f"{briefing}:story:{story_id}"
    conn.execute(
        "INSERT INTO briefing_stories(briefing, story_id, item_id, fingerprint, headline, summary, status, "
        "countries_json, urls_json, data_json, first_seen, last_seen, first_run_id, last_run_id) "
        "VALUES (?, ?, ?, 'fp', ?, ?, 'new', '[]', '[]', '{}', ?, ?, ?, ?)",
        (briefing, story_id, item_id, headline, summary, last_seen, last_seen, run_id, run_id),
    )
    conn.execute(
        "INSERT INTO briefing_runs(briefing, run_id, started, status, report_path) VALUES (?, ?, ?, 'ok', ?)",
        (briefing, run_id, last_seen, report_path),
    )
    conn.commit()
    return item_id


def _seed_briefing_article(conn, *, briefing, url, title, source, published, text, last_seen):
    item_id = f"{briefing}:article:{abs(hash(url)) % (10 ** 12)}"
    conn.execute(
        "INSERT INTO briefing_articles(briefing, url, item_id, title, source, published, text, "
        "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (briefing, url, item_id, title, source, published, text, last_seen, last_seen),
    )
    conn.commit()
    return item_id


def test_search_briefings_story_hit_links_to_report_anchor(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    report_file = output_dir / "briefing-example-20260901-0000.html"
    report_file.write_text("<html></html>", encoding="utf-8")

    with Store(tmp_path / "db.sqlite") as store:
        item_id = _seed_briefing_story(
            store.conn, briefing="example", story_id="abc123", headline="Port strike widens",
            summary="Dockworkers expand the strike to two more ports.", run_id=1,
            last_seen="2026-09-01T00:00:00+00:00", report_path=str(report_file),
        )
        vectors_upsert(store.conn, ns="briefings", model="m", items=[(item_id, [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=output_dir)
        hits = ask._search_briefings(ctx, "port strike", 5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.scope == "briefings"
    assert hit.title == "Port strike widens"
    assert hit.href == f"/reports/{report_file.name}#story-abc123"
    assert "Dockworkers" in hit.excerpt


def test_search_briefings_article_hit_links_to_its_own_url(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        item_id = _seed_briefing_article(
            store.conn, briefing="example", url="https://news.example/a", title="Port strike widens",
            source="Example Wire", published="2026-09-01T00:00:00+00:00",
            text="Dockworkers expand the strike.", last_seen="2026-09-01T00:00:00+00:00",
        )
        vectors_upsert(store.conn, ns="briefings", model="m", items=[(item_id, [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_briefings(ctx, "port strike", 5)

    assert len(hits) == 1
    assert hits[0].href == "https://news.example/a"
    assert hits[0].excerpt == "Dockworkers expand the strike."


def test_search_briefings_since_drops_older_articles(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        item_id = _seed_briefing_article(
            store.conn, briefing="example", url="https://news.example/old", title="Old story",
            source="Example Wire", published="2026-01-01T00:00:00+00:00",
            text="Old text.", last_seen="2026-01-01T00:00:00+00:00",
        )
        vectors_upsert(store.conn, ns="briefings", model="m", items=[(item_id, [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=date(2026, 6, 1), output_dir=tmp_path)
        hits = ask._search_briefings(ctx, "old story", 5)

    assert hits == []


def test_search_docs_returns_chunk_snippet_with_no_href(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.execute(
            "CREATE TABLE IF NOT EXISTS doc_chunks (path TEXT, chunk_index INTEGER, text TEXT, "
            "mtime REAL, size INTEGER, PRIMARY KEY(path, chunk_index))"
        )
        store.conn.execute(
            "INSERT INTO doc_chunks(path, chunk_index, text, mtime, size) VALUES (?, 0, ?, 0, 100)",
            (r"C:\docs\roof-budget.md", "The roof budget is $500."),
        )
        store.conn.commit()
        vectors_upsert(store.conn, ns="docs", model="m", items=[(r"C:\docs\roof-budget.md#0", [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_docs(ctx, "roof budget", 5)

    assert len(hits) == 1
    assert hits[0].href is None
    assert hits[0].excerpt == "The roof budget is $500."
    assert hits[0].title == "roof-budget.md"


def test_search_markets_returns_question_with_market_url(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.execute(
            "INSERT INTO market_snapshots(source, market_id, day, question, url, probability, fetched_at) "
            "VALUES ('polymarket', 'm1', '2026-09-01', 'Will the strike end by October?', "
            "'https://polymarket.com/m1', 0.42, '2026-09-01T00:00:00+00:00')"
        )
        store.conn.commit()
        vectors_upsert(store.conn, ns="markets", model="m", items=[("polymarket:m1", [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_markets(ctx, "strike odds", 5)

    assert len(hits) == 1
    assert hits[0].href == "https://polymarket.com/m1"
    assert "42%" in hits[0].meta
    assert hits[0].excerpt == "Will the strike end by October?"


def _seed_award(conn, *, day, identity_key, contractor, program_or_item, service, location,
                 amount_usd, contract_number, source_url):
    conn.execute(
        "INSERT INTO dod_contracts (identity_key, contract_number, day, paragraph_hash, service, "
        "contractor, location, program_or_item, amount_usd, source_url, first_seen, last_seen) "
        "VALUES (?, ?, ?, 'ph', ?, ?, ?, ?, ?, ?, ?, ?)",
        (identity_key, contract_number, day, service, contractor, location, program_or_item,
         amount_usd, source_url, f"{day}T00:00:00+00:00", f"{day}T00:00:00+00:00"),
    )
    conn.commit()


def test_search_contracts_returns_contractor_and_amount_with_source_url(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_award(
            store.conn, day="2026-09-20", identity_key="N00024-26-C-0001", contractor="Acme Defense LLC",
            program_or_item="submarine sonar upgrade", service="NAVY", location="Groton, Connecticut",
            amount_usd=45_000_000.0, contract_number="N00024-26-C-0001",
            source_url="https://www.defense.gov/News/Contracts/Contract/Article/1",
        )
        item_id = "2026-09-20:N00024-26-C-0001"
        vectors_upsert(store.conn, ns="contracts", model="m", items=[(item_id, [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_contracts(ctx, "submarine contract", 5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.scope == "contracts"
    assert hit.href == "https://www.defense.gov/News/Contracts/Contract/Article/1"
    assert "Acme Defense LLC" in hit.title
    assert "$45,000,000" in hit.title
    assert "NAVY" in hit.meta


def test_search_contracts_since_filters_on_ledger_day(tmp_path):
    with Store(tmp_path / "db.sqlite") as store:
        _seed_award(
            store.conn, day="2026-01-01", identity_key="old-1", contractor="Old Corp",
            program_or_item="widgets", service="ARMY", location="Nowhere",
            amount_usd=1_000.0, contract_number="old-1", source_url="https://www.defense.gov/x",
        )
        item_id = "2026-01-01:old-1"
        vectors_upsert(store.conn, ns="contracts", model="m", items=[(item_id, [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=date(2026, 6, 1), output_dir=tmp_path)
        hits = ask._search_contracts(ctx, "widgets", 5)

    assert hits == []


def test_search_all_merges_scopes_and_ranks_by_score(tmp_path):
    """Mixed-namespace ranking (#68 acceptance criteria): a strong gmail
    match should outrank a weak briefings match in the merged "all" list."""
    with Store(tmp_path / "db.sqlite") as store:
        store.conn.execute(
            "INSERT INTO messages(msg_id, thread_id, date, from_addr, subject, labels_json, seen_at) "
            "VALUES ('m1', 't1', '2026-09-01T00:00:00+00:00', 'a@example.com', 'Roof budget', '[]', "
            "'2026-09-01T00:00:00+00:00')"
        )
        store.conn.execute(
            "CREATE TABLE IF NOT EXISTS mail_index (msg_id TEXT PRIMARY KEY, web_url TEXT NOT NULL, "
            "excerpt TEXT NOT NULL)"
        )
        store.conn.execute(
            "INSERT INTO mail_index(msg_id, web_url, excerpt) VALUES ('m1', 'https://mail.example/m1', "
            "'Budget is $500.')"
        )
        story_item = _seed_briefing_story(
            store.conn, briefing="example", story_id="s1", headline="Unrelated story",
            summary="Something else entirely.", run_id=1, last_seen="2026-09-01T00:00:00+00:00",
            report_path=str(tmp_path / "missing.html"),
        )
        store.conn.commit()
        # gmail vector points the same way as the query; the briefing vector
        # is nearly orthogonal, so it must score (and rank) lower.
        vectors_upsert(store.conn, ns="gmail", model="m", items=[("m1", [1.0, 0.0])])
        vectors_upsert(store.conn, ns="briefings", model="m", items=[(story_item, [0.01, 1.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_all(ctx, "roof budget", 5)

    assert [h.scope for h in hits[:1]] == ["gmail"]
    assert hits[0].score > hits[-1].score


def test_search_all_includes_contracts(tmp_path):
    """#68 acceptance criteria: `--scope all` must reach the `contracts`
    namespace too, not just gmail/briefings/docs/markets."""
    with Store(tmp_path / "db.sqlite") as store:
        _seed_award(
            store.conn, day="2026-09-20", identity_key="N00024-26-C-0001", contractor="Acme Defense LLC",
            program_or_item="submarine sonar upgrade", service="NAVY", location="Groton, Connecticut",
            amount_usd=45_000_000.0, contract_number="N00024-26-C-0001",
            source_url="https://www.defense.gov/News/Contracts/Contract/Article/1",
        )
        vectors_upsert(store.conn, ns="contracts", model="m",
                        items=[("2026-09-20:N00024-26-C-0001", [1.0, 0.0])])

        ctx = AskContext(conn=store.conn, lm=FakeEmbedLM([1.0, 0.0]), embed_model="m",
                          since=None, output_dir=tmp_path)
        hits = ask._search_all(ctx, "submarine contract", 5)

    assert any(h.scope == "contracts" for h in hits)


def test_answer_hits_returns_grounded_answer_and_citations():
    hits = [ScopeHit(number=1, scope="briefings", title="Port strike widens", meta="example · story",
                      href="/reports/x.html#story-1", score=0.9, excerpt="Dockworkers expand the strike.")]
    lm = FakeLM(result={"answer": "The strike is widening.", "citations": [1]})

    answer, citations, error = ask.answer_hits(lm, "what's happening with the strike", hits)

    assert answer == "The strike is widening."
    assert citations == [1]
    assert error is None


def test_answer_hits_says_not_found_with_no_usable_excerpts():
    hits = [ScopeHit(number=1, scope="docs", title="x", meta="x", href=None, score=0.1, excerpt=None)]
    lm = FakeLM(result={"answer": "should not be called", "citations": []})

    answer, citations, error = ask.answer_hits(lm, "q", hits)

    assert answer == NOT_FOUND
    assert citations == []
    assert lm.calls == []


def test_build_scope_report_shows_citations_and_source_urls():
    hits = [ScopeHit(number=1, scope="briefings", title="Port strike widens", meta="example · story",
                      href="/reports/x.html#story-1", score=0.9, excerpt="text")]
    html = ask.build_scope_report(
        "what's happening", "briefings", hits, answer="The strike is widening.", citations=[1],
    ).render()
    assert "The strike is widening." in html
    assert "[1]" in html
    assert "/reports/x.html#story-1" in html


def test_build_scope_report_escapes_untrusted_title_and_drops_unsafe_href():
    hits = [ScopeHit(number=1, scope="docs", title="<script>alert(1)</script>", meta="m",
                      href="javascript:alert(1)", score=0.5, excerpt="text")]
    html = ask.build_scope_report("q", "docs", hits).render()
    assert "<script>alert(1)</script>" not in html
    assert "javascript:alert(1)" not in html


def test_build_scope_report_no_hits_shows_scope_specific_warning():
    html = ask.build_scope_report("q", "markets", []).render()
    assert "markets" in html
    assert "No indexed items matched" in html
