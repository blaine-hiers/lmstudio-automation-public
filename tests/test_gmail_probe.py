"""tasks.gmail_probe must count via IMAP SEARCH only -- never fetch messages
just to report a number (that both wastes a round trip and, pre-fix, capped
the reported count at search()'s default limit of 200)."""

import tasks.gmail_probe as probe


class _FakeGmail:
    def __init__(self, cfg=None):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def count(self, query, folder="[Gmail]/All Mail"):
        self.queries.append(query)
        return 437  # deliberately > the old search() default limit of 200

    def search(self, *args, **kwargs):
        raise AssertionError("gmail_probe must count via SEARCH, not fetch messages")


def test_probe_counts_via_search_and_reports_full_total(monkeypatch, tmp_path):
    fake = _FakeGmail()
    monkeypatch.setattr(probe, "Gmail", lambda cfg=None: fake)
    monkeypatch.setenv("LMSA_IMAP_USER", "bot@example.com")
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path))

    probe.main()

    assert fake.queries == ["newer_than:1d"]
    [written] = [p for p in tmp_path.rglob("*.html") if p.name != "index.html"]
    assert "437" in written.read_text(encoding="utf-8")
