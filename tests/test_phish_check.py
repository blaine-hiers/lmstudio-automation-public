from pathlib import Path
from datetime import datetime, timezone

import pytest

from lmsa.sources.gmail import Message
from tasks import phish_check as pc


def _msg(**overrides) -> Message:
    defaults = dict(
        msg_id="1",
        thread_id="1",
        labels=[],
        flags=[],
        from_name="Alice Example",
        from_addr="alice@example.com",
        to=["me@example.com"],
        reply_to="",
        subject="Hello",
        date=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        list_unsubscribe="",
        list_id="",
        auth_results="mx.google.com; spf=pass smtp.mailfrom=alice@example.com; "
                      "dkim=pass header.i=@example.com; dmarc=pass header.from=example.com",
        text="Just a normal message.",
        attachments=[],
        web_url="https://mail.google.com/mail/u/0/#all/1",
    )
    defaults.update(overrides)
    return Message(**defaults)


# ---- registrable_domain ------------------------------------------------------

def test_registrable_domain_from_email_address():
    assert pc.registrable_domain("user@mail.example.com") == "example.com"


def test_registrable_domain_two_label_suffix():
    assert pc.registrable_domain("shop.example.co.uk") == "example.co.uk"


def test_registrable_domain_plain_generic_suffix_keeps_last_two():
    assert pc.registrable_domain("sub.mail.example.com") == "example.com"


def test_registrable_domain_decodes_punycode_and_normalizes():
    # xn--mnchen-3ya.de is the punycode form of muenchen.de.
    assert pc.registrable_domain("xn--mnchen-3ya.de") == "münchen.de"


def test_registrable_domain_handles_empty_and_single_label():
    assert pc.registrable_domain("") == ""
    assert pc.registrable_domain("localhost") == "localhost"


# ---- edit_distance ------------------------------------------------------------

def test_edit_distance_identical_is_zero():
    assert pc.edit_distance("example.com", "example.com") == 0


def test_edit_distance_one_substitution():
    assert pc.edit_distance("paypal.com", "paypa1.com") == 1


def test_edit_distance_against_empty():
    assert pc.edit_distance("", "abc") == 3
    assert pc.edit_distance("abc", "") == 3


# ---- Authentication-Results parsing --------------------------------------------

def test_parse_auth_results_extracts_all_three():
    header = "mx.google.com; spf=softfail smtp.mailfrom=x; dkim=fail header.i=@x; dmarc=none"
    assert pc.parse_auth_results(header) == {"spf": "softfail", "dkim": "fail", "dmarc": "none"}


def test_parse_auth_results_worst_verdict_wins_on_repeat_except_dkim():
    # spf/dmarc: worst wins. dkim: any valid signature wins (mailing lists
    # break the original signature and add their own).
    assert pc.parse_auth_results("mx.google.com; spf=pass; spf=fail")["spf"] == "fail"
    assert pc.parse_auth_results("mx.google.com; dmarc=fail; dmarc=pass")["dmarc"] == "fail"
    header = "mx.google.com; dkim=fail header.i=@a; dkim=pass header.i=@b"
    assert pc.parse_auth_results(header)["dkim"] == "pass"
    header = "mx.google.com; dkim=neutral header.i=@a; dkim=fail header.i=@b"
    assert pc.parse_auth_results(header)["dkim"] == "fail"


def test_parse_auth_results_empty_header():
    assert pc.parse_auth_results("") == {}


def test_authserv_id_extracted():
    assert pc.authserv_id("mx.google.com; spf=pass") == "mx.google.com"
    assert pc.authserv_id("") == ""


# ---- auth_signals ---------------------------------------------------------------

def test_auth_signals_all_pass_is_clean():
    msg = _msg(auth_results="mx.google.com; spf=pass; dkim=pass; dmarc=pass")
    assert pc.auth_signals(msg) == []


def test_auth_signals_flags_softfail_none_and_temperror():
    msg = _msg(auth_results="mx.google.com; spf=softfail; dkim=none; dmarc=temperror")
    tags = {s.tag for s in pc.auth_signals(msg)}
    assert tags == {"auth:spf=softfail", "auth:dkim=none", "auth:dmarc=temperror"}


def test_auth_signals_missing_method_is_not_flagged_as_none():
    # dkim is simply absent from the header -- must not be treated as dkim=none.
    msg = _msg(auth_results="mx.google.com; spf=pass; dmarc=pass")
    assert pc.auth_signals(msg) == []


# ---- brand_mismatch_signals -----------------------------------------------------

def test_brand_mismatch_flagged_when_domain_does_not_match():
    msg = _msg(from_name="PayPal Support", from_addr="support@paypa1-security.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {"brand-mismatch:paypal"}


def test_brand_mismatch_not_flagged_when_domain_matches():
    msg = _msg(from_name="PayPal Support", from_addr="service@paypal.com")
    assert pc.brand_mismatch_signals(msg) == []


def test_brand_mismatch_uses_word_boundaries():
    # "irs" must not match inside "First" -- a bare substring check would.
    msg = _msg(from_name="First National Notices", from_addr="notices@example.com")
    assert pc.brand_mismatch_signals(msg) == []


# ---- lookalike_signals ------------------------------------------------------------

def test_lookalike_flags_close_but_not_exact_domain():
    known = {"example.com"}
    signals = pc.lookalike_signals("examp1e.com", known, tag_prefix="lookalike")
    assert len(signals) == 1
    assert signals[0].tag == "lookalike:examp1e.com"


def test_lookalike_exact_match_is_not_flagged():
    known = {"example.com"}
    assert pc.lookalike_signals("example.com", known, tag_prefix="lookalike") == []


def test_lookalike_too_far_is_not_flagged():
    known = {"example.com"}
    assert pc.lookalike_signals("totallydifferent.org", known, tag_prefix="lookalike") == []


def test_lookalike_empty_domain_is_not_flagged():
    assert pc.lookalike_signals("", {"example.com"}, tag_prefix="lookalike") == []


# ---- reply_to_signal ---------------------------------------------------------------

def test_reply_to_mismatch_flagged():
    msg = _msg(from_addr="alice@example.com", reply_to="reply@other-domain.com")
    tags = {s.tag for s in pc.reply_to_signal(msg)}
    assert tags == {"reply-to-mismatch"}


def test_reply_to_same_domain_not_flagged():
    msg = _msg(from_addr="alice@example.com", reply_to="billing@example.com")
    assert pc.reply_to_signal(msg) == []


def test_reply_to_mismatch_on_a_later_address_is_flagged():
    # Only the *second* Reply-To address is foreign -- a reply goes to both.
    msg = _msg(
        from_addr="ceo@acme.com", reply_to="ceo@acme.com",
        reply_to_all=("ceo@acme.com", "wire@evil-payments.example"),
    )
    signals = pc.reply_to_signal(msg)
    assert [s.tag for s in signals] == ["reply-to-mismatch"]
    assert "evil-payments.example" in signals[0].detail


def test_reply_to_all_same_domain_not_flagged():
    msg = _msg(
        from_addr="ceo@acme.com", reply_to="ceo@acme.com",
        reply_to_all=("ceo@acme.com", "billing@acme.com"),
    )
    assert pc.reply_to_signal(msg) == []


def test_reply_to_absent_not_flagged():
    msg = _msg(from_addr="alice@example.com", reply_to="")
    assert pc.reply_to_signal(msg) == []


# ---- link_signals -----------------------------------------------------------------

def test_link_signals_finds_urls_and_flags_lookalike_domain():
    text = "Please visit http://paypa1.com/verify now, or http://example.com for help."
    signals, urls = pc.link_signals(text, {"paypal.com"})
    assert urls == ["http://paypa1.com/verify", "http://example.com"]
    assert [s.tag for s in signals] == ["link-homograph:paypa1.com", "link-domain:paypa1.com"]


def test_link_signals_no_urls():
    signals, urls = pc.link_signals("No links here.", {"example.com"})
    assert signals == []
    assert urls == []


def test_link_signals_dedupes_repeated_domain():
    text = "http://examp1e.com/a http://examp1e.com/b"
    signals, urls = pc.link_signals(text, {"example.com"})
    assert len(urls) == 2
    # one signal per check per distinct domain, not per URL
    assert [s.tag for s in signals] == ["link-homograph:examp1e.com", "link-domain:examp1e.com"]


# ---- analyze (per-item isolation) --------------------------------------------------

class _FakeLM:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append(prompt)
        if self._error is not None:
            raise self._error
        return self._result


def test_analyze_combines_heuristics_and_model_score():
    msg = _msg(
        from_name="PayPal Support",
        from_addr="support@paypa1-security.com",
        subject="Your account will be suspended",
        text="Act now or lose access.",
    )
    lm = _FakeLM(result={"urgency_or_money_pressure": 5, "reason": "urgent threat"})
    finding = pc.analyze(lm, msg, {"paypal.com"}, system="system prompt")
    assert finding.urgency == 5
    assert finding.reason == "urgent threat"
    assert finding.model_error is None
    assert finding.flagged is True
    assert any(s.tag == "brand-mismatch:paypal" for s in finding.signals)
    # untrusted message content must reach the model only via a single user prompt
    assert len(lm.calls) == 1
    assert "suspended" in lm.calls[0]


def test_analyze_isolates_model_failure_without_aborting():
    msg = _msg()  # clean message: no heuristic signals
    lm = _FakeLM(error=RuntimeError("server exploded"))
    finding = pc.analyze(lm, msg, set(), system="system prompt")
    assert finding.model_error == "server exploded"
    assert finding.urgency is None
    assert finding.signals == []
    assert finding.flagged is False


def test_analyze_never_raises_even_if_a_heuristic_breaks(monkeypatch):
    def boom(message):
        raise ValueError("boom")

    monkeypatch.setattr(pc, "auth_signals", boom)
    msg = _msg()
    lm = _FakeLM(result={"urgency_or_money_pressure": 0, "reason": "fine"})
    finding = pc.analyze(lm, msg, set(), system="system prompt")
    assert finding.flagged is True
    assert any(s.tag == "analysis-error" for s in finding.signals)


# ---- known_domains_from_sent (one scan) --------------------------------------------

class _FakeGmail:
    def __init__(self, sent_messages=None, inbox_messages=None, sent_error=None):
        self.sent_messages = sent_messages if sent_messages is not None else []
        self.inbox_messages = inbox_messages if inbox_messages is not None else []
        self.sent_error = sent_error
        self.search_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def sent(self):
        if self.sent_error:
            raise RuntimeError(self.sent_error)
        return "[Gmail]/Sent Mail"

    def search(self, query, folder="[Gmail]/All Mail", limit=200, *, headers_only=False):
        self.search_calls.append((query, folder, limit, headers_only))
        if folder == "[Gmail]/Sent Mail":
            return self.sent_messages
        return self.inbox_messages


def test_known_domains_from_sent_scans_once_and_extracts_recipient_domains():
    fake = _FakeGmail(sent_messages=[
        _msg(to=["bob@example.com", "carol@example.org"]),
        _msg(to=["dave@example.com"]),
    ])
    scan = pc.known_domains_from_sent(fake)
    assert scan.domains == {"example.com", "example.org"}
    assert scan.errors == []
    assert scan.capped is False
    assert len(fake.search_calls) == 1  # exactly one scan of Sent
    assert fake.search_calls[0][1] == "[Gmail]/Sent Mail"
    assert fake.search_calls[0][3] is True  # headers_only


def test_known_domains_from_sent_degrades_on_failure_without_raising():
    fake = _FakeGmail(sent_error="no such folder")
    scan = pc.known_domains_from_sent(fake)
    assert scan.domains == set()
    assert len(scan.errors) == 1
    assert "no such folder" in scan.errors[0]


# ---- build_report / rendering -----------------------------------------------------

def test_build_report_renders_flagged_card_with_tags_and_escapes_untrusted_text():
    msg = _msg(
        subject='<script>alert(1)</script>',
        from_name="PayPal Support",
        from_addr="support@paypa1-security.com",
        text="Click http://paypa1-security.com/verify now.",
    )
    finding = pc.Finding(
        message=msg,
        signals=[pc.Signal("brand-mismatch:paypal", "display name mismatch")],
        urls=["http://paypa1-security.com/verify"],
        urgency=4,
        reason="urgency language",
    )
    report = pc.build_report(
        [finding], days=3, known_count=1, sent_errors=[], truncated=False,
    )
    html = report.render()

    # untrusted subject must be escaped, never executed as markup
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    # the suspicious link is shown as text, never as a clickable href
    assert 'href="http://paypa1-security.com/verify"' not in html
    assert "http://paypa1-security.com/verify" in html
    assert "brand-mismatch:paypal" in html
    assert "urgency:4" in html
    # the report's own Gmail deep link is still a real clickable href
    assert f'href="{msg.web_url}"' in html


def test_build_report_no_flags_renders_clean_state():
    report = pc.build_report([], days=3, known_count=5, sent_errors=[], truncated=False)
    html = report.render()
    assert "nothing flagged" in html
    assert "No messages triggered" in html


def test_build_report_surfaces_truncation_and_sent_errors():
    report = pc.build_report(
        [], days=3, known_count=0, sent_errors=["could not scan Sent folder: boom"],
        truncated=True,
    )
    html = report.render()
    assert "truncated" in html.lower()
    assert "could not scan Sent folder: boom" in html


# ---- main() end-to-end --------------------------------------------------------------

class _FakeLMStudio:
    def __init__(self, cfg=None):
        self.calls = []

    def ensure_server(self):
        pass

    def extract(self, prompt, schema, *, system=None, name="result"):
        self.calls.append(prompt)
        if "suspended" in prompt:
            return {"urgency_or_money_pressure": 5, "reason": "urgent account threat"}
        return {"urgency_or_money_pressure": 0, "reason": "routine"}


def test_main_writes_report_and_flags_suspicious_message(monkeypatch, tmp_path, capsys):
    sent = [_msg(to=["alice@example.com"])]
    inbox = [
        _msg(
            msg_id="10", thread_id="10",
            subject="Your account will be suspended",
            from_name="PayPal Support",
            from_addr="support@paypa1-security.com",
            reply_to="reply@somewhere-else.com",
            auth_results="mx.google.com; spf=softfail; dkim=fail; dmarc=none",
            text="Act now: http://paypa1-security.com/verify",
            web_url="https://mail.google.com/mail/u/0/#all/10",
        ),
        _msg(
            msg_id="11", thread_id="11",
            subject="Weekly newsletter",
            from_name="Alice Example",
            from_addr="alice@example.com",
            text="Nothing interesting.",
        ),
    ]
    fake_gmail = _FakeGmail(sent_messages=sent, inbox_messages=inbox)

    monkeypatch.setattr(pc, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(pc, "LMStudio", _FakeLMStudio)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr("sys.argv", ["phish_check"])

    pc.main()

    out = capsys.readouterr().out.strip()
    written = Path(out)
    assert written.exists()
    html = written.read_text(encoding="utf-8")
    assert "Phish check" in html
    assert "1 flagged" in html
    assert "brand-mismatch:paypal" in html
    assert "reply-to-mismatch" in html
    assert "auth:spf=softfail" in html
    # the clean newsletter message must not appear as a flagged card
    assert "Weekly newsletter" not in html

    inbox_query = [c for c in fake_gmail.search_calls if c[1] == "[Gmail]/All Mail"]
    assert inbox_query == [("newer_than:3d in:inbox", "[Gmail]/All Mail", pc.DEFAULT_LIMIT, False)]


def test_main_truncated_when_inbox_search_hits_limit(monkeypatch, tmp_path, capsys):
    inbox = [_msg(msg_id=str(i), thread_id=str(i)) for i in range(3)]
    fake_gmail = _FakeGmail(sent_messages=[], inbox_messages=inbox)

    monkeypatch.setattr(pc, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(pc, "LMStudio", _FakeLMStudio)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr("sys.argv", ["phish_check", "--limit", "3"])

    pc.main()

    out = capsys.readouterr().out.strip()
    written = Path(out)
    html = written.read_text(encoding="utf-8")
    assert "truncated" in html.lower()


# =====================================================================================
# Regression tests for the attempt-1 adversarial review findings.
# =====================================================================================

# ---- finding 1: auth-results parsed per resinfo, comments stripped, worst wins -------

def test_auth_comment_cannot_smuggle_a_passing_dmarc():
    header = (
        'mx.google.com; spf=pass (google.com: domain of "dmarc=pass"@evil.com '
        'designates 192.0.2.1 as permitted sender) smtp.mailfrom="dmarc=pass"@evil.com; '
        "dkim=pass header.i=@evil.com; dmarc=fail (p=REJECT) header.from=paypal.com"
    )
    assert pc.parse_auth_results(header)["dmarc"] == "fail"
    tags = {s.tag for s in pc.auth_signals(_msg(auth_results=header))}
    assert "auth:dmarc=fail" in tags


def test_auth_quoted_property_with_semicolon_cannot_start_a_segment():
    header = (
        'mx.google.com; spf=pass smtp.mailfrom="x;dmarc=pass"@evil.com; '
        "dmarc=fail header.from=paypal.com"
    )
    assert pc.parse_auth_results(header)["dmarc"] == "fail"


def test_auth_nested_comments_are_stripped():
    header = "mx.google.com; spf=pass (a (nested; dmarc=pass) b) smtp.mailfrom=x; dmarc=none"
    assert pc.parse_auth_results(header) == {"spf": "pass", "dmarc": "none"}


def test_auth_method_only_counts_at_segment_start():
    header = "mx.google.com; spf=pass smtp.mailfrom=dmarc=pass; dmarc=fail"
    assert pc.parse_auth_results(header)["dmarc"] == "fail"


def test_auth_unterminated_comment_is_flagged_unparseable():
    header = "mx.google.com; spf=pass (unterminated; dmarc=fail"
    tags = {s.tag for s in pc.auth_signals(_msg(auth_results=header))}
    assert tags == {"auth:unparseable"}


# ---- finding 2: only the trusted authserv-id is believed ---------------------------

def test_auth_untrusted_authserv_id_is_flagged_and_results_ignored():
    header = "mx.evil.example; spf=pass; dkim=pass; dmarc=pass"
    tags = {s.tag for s in pc.auth_signals(_msg(auth_results=header))}
    assert tags == {"auth:untrusted-authserv"}
    # a verdict from an untrusted server is not believed either way
    header = "mx.evil.example; dmarc=fail"
    tags = {s.tag for s in pc.auth_signals(_msg(auth_results=header))}
    assert tags == {"auth:untrusted-authserv"}


def test_auth_missing_authserv_id_is_untrusted():
    tags = {s.tag for s in pc.auth_signals(_msg(auth_results="dmarc=pass; spf=pass"))}
    assert tags == {"auth:untrusted-authserv"}


def test_auth_trusted_id_is_a_module_constant():
    assert pc.TRUSTED_AUTHSERV_ID == "mx.google.com"
    assert pc.authserv_id("MX.Google.com; spf=pass") == "MX.Google.com"
    assert pc.auth_signals(_msg(auth_results="MX.Google.com; spf=pass")) == []


# ---- finding 3: url_domain resolves the host a browser would visit -----------------

@pytest.mark.parametrize("url", [
    "https://paypa1.com?@paypal.com",
    "https://paypa1.com#@paypal.com",
    "https://paypa1.com\\@paypal.com",
    "https://paypa1.com/@paypal.com",
])
def test_url_domain_userinfo_tricks_resolve_to_real_host(url):
    assert pc.url_domain(url) == "paypa1.com"


def test_url_domain_lowercases_and_decodes_idna():
    assert pc.url_domain("https://WWW.Example.COM/x") == "example.com"
    ace = "xn--" + "pаypal".encode("punycode").decode("ascii")
    assert pc.url_domain(f"https://{ace}.com/") == "pаypal.com"


def test_url_domain_invalid_url_returns_empty():
    assert pc.url_domain("https://[bad/") == ""


@pytest.mark.parametrize("text,expected", [
    ("go to https://paypa1.com/login, now", "https://paypa1.com/login"),
    ("go to https://paypa1.com/login; now", "https://paypa1.com/login"),
    ("go to https://paypa1.com/login! now", "https://paypa1.com/login"),
    ("(see https://paypa1.com/login)", "https://paypa1.com/login"),
    ("[https://paypa1.com/login]", "https://paypa1.com/login"),
    ("end https://paypa1.com/login.", "https://paypa1.com/login"),
    ("wiki https://en.example.org/Foo_(bar)", "https://en.example.org/Foo_(bar)"),
])
def test_extract_urls_strips_trailing_punctuation_with_balanced_parens(text, expected):
    assert pc.extract_urls(text) == [expected]


def test_link_signals_flag_userinfo_trick_url():
    signals, _ = pc.link_signals("Log in: https://paypa1.com?@paypal.com", set(pc.BRAND_DOMAINS))
    assert "link-homograph:paypa1.com" in {s.tag for s in signals}


# ---- finding 4: homographs ----------------------------------------------------------

def test_skeleton_folds_cyrillic_greek_and_ascii_tricks():
    assert pc.skeleton("pаypаl.com") == "paypal.com"   # Cyrillic a
    assert pc.skeleton("οutlook.com") == "outlook.com"        # Greek omicron
    assert pc.skeleton("rnicrosoft.com") == "microsoft.com"
    assert pc.skeleton("vvellsfargo.com") == "wellsfargo.com"
    assert pc.skeleton("g00gle.com") == "google.com"
    assert pc.skeleton("paypa1.com") == "paypal.com"


@pytest.mark.parametrize("domain", [
    "pаypаl.com",                  # Cyrillic a -- mixed script
    "аррӏе.com",    # all-Cyrillic "apple"
    "rnicrosoft.com",
    "g00gle.com",
])
def test_homograph_sender_domain_is_flagged(domain):
    signals = pc.homograph_signals(domain, set(pc.BRAND_DOMAINS), tag_prefix="homograph")
    assert [s.tag for s in signals] == [f"homograph:{domain}"]


def test_homograph_mixed_script_label_flagged_even_without_known_twin():
    domain = "exаmple-shop.com"   # Cyrillic a, not a known domain
    signals = pc.homograph_signals(domain, set(), tag_prefix="homograph")
    assert len(signals) == 1
    assert "mixes scripts" in signals[0].detail


def test_fullwidth_domain_is_suspicious_not_the_real_one():
    raw = "ｐａｙｐａｌ.com"   # full-width "paypal"
    domain = pc.registrable_domain(f"support@{raw}")
    assert domain != "paypal.com"
    signals = pc.homograph_signals(domain, set(pc.BRAND_DOMAINS), tag_prefix="homograph")
    assert len(signals) == 1
    assert "full-width" in signals[0].detail


def test_fullwidth_punycode_domain_is_not_folded_to_the_brand():
    # IDNA nameprep would fold this to paypal.com; per-label punycode must not.
    label = "ｐａｙｐａｌ"
    ace = "xn--" + label.encode("punycode").decode("ascii")
    domain = pc.registrable_domain(f"{ace}.com")
    assert domain == f"{label}.com"
    assert pc.homograph_signals(domain, set(pc.BRAND_DOMAINS), tag_prefix="homograph")


def test_homograph_real_domain_is_not_flagged():
    assert pc.homograph_signals("paypal.com", set(pc.BRAND_DOMAINS), tag_prefix="homograph") == []
    assert pc.homograph_signals("münchen.de", set(), tag_prefix="homograph") == []


def test_analyze_flags_homograph_sender():
    msg = _msg(from_name="Accounts", from_addr="service@pаypаl.com")
    lm = _FakeLM(result={"urgency_or_money_pressure": 0, "reason": "x"})
    finding = pc.analyze(lm, msg, set(pc.BRAND_DOMAINS), system="s")
    assert "homograph:pаypаl.com" in {s.tag for s in finding.signals}


@pytest.mark.parametrize("name", [
    "PаyPаl",                         # Cyrillic a
    "ＰａｙＰａｌ",      # full-width
    "PayPa1 Jane",                               # digit swap, even with a person-ish word
])
def test_brand_mismatch_catches_disguised_display_names(name):
    msg = _msg(from_name=name, from_addr="notice@example.net")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {"brand-mismatch:paypal"}


# ---- finding 5: messages whose model call failed stay visible ------------------------

def test_report_lists_unflagged_messages_whose_model_call_failed():
    ok = pc.Finding(message=_msg(subject="Routine note"), urgency=0, reason="fine")
    failed = pc.Finding(
        message=_msg(msg_id="2", subject="<b>Quarterly numbers</b>", from_addr="bob@example.org"),
        model_error="timed out after 60s",
    )
    html = pc.build_report(
        [ok, failed], days=3, known_count=1, sent_errors=[], truncated=False,
    ).render()
    assert "Not scored by the model" in html
    assert "&lt;b&gt;Quarterly numbers&lt;/b&gt;" in html
    assert "bob@example.org" in html
    assert "timed out after 60s" in html
    assert "Routine note" not in html


def test_model_error_message_is_bounded():
    lm = _FakeLM(error=RuntimeError("x" * 5000))
    finding = pc.analyze(lm, _msg(), set(), system="s")
    assert len(finding.model_error) <= pc.ERROR_LIMIT + 3


# ---- finding 6: urgency must be a real int in 0..5 -------------------------------------

@pytest.mark.parametrize("value", [True, False, 6, -1, 3.5, "4", None, 99])
def test_invalid_urgency_is_a_model_error(value):
    lm = _FakeLM(result={"urgency_or_money_pressure": value, "reason": "r"})
    finding = pc.analyze(lm, _msg(), set(), system="s")
    assert finding.urgency is None
    assert finding.model_error and "urgency" in finding.model_error
    assert finding.flagged is False


def test_valid_urgency_bounds_accepted():
    for value in (0, 5):
        lm = _FakeLM(result={"urgency_or_money_pressure": value, "reason": "r"})
        assert pc.analyze(lm, _msg(), set(), system="s").urgency == value


def test_non_dict_model_result_is_a_model_error():
    finding = pc.analyze(_FakeLM(result=["nope"]), _msg(), set(), system="s")
    assert finding.model_error and finding.urgency is None


# ---- finding 7: Sent-folder scan cap is disclosed -----------------------------------------

def test_sent_scan_cap_is_reported(monkeypatch):
    monkeypatch.setattr(pc, "SENT_SCAN_LIMIT", 2)
    fake = _FakeGmail(sent_messages=[_msg(to=["a@example.com"]), _msg(to=["b@example.org"])])
    scan = pc.known_domains_from_sent(fake)
    assert scan.capped is True
    assert fake.search_calls[0][2] == 2
    html = pc.build_report(
        [], days=3, known_count=2, sent_errors=[], truncated=False, sent_capped=scan.capped,
    ).render()
    assert "Sent-folder scan capped" in html
    assert "stopped at 2 messages" in html


def test_main_discloses_sent_cap(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pc, "SENT_SCAN_LIMIT", 1)
    fake_gmail = _FakeGmail(sent_messages=[_msg(to=["a@example.com"])], inbox_messages=[_msg()])
    monkeypatch.setattr(pc, "Gmail", lambda cfg=None: fake_gmail)
    monkeypatch.setattr(pc, "LMStudio", _FakeLMStudio)
    monkeypatch.setenv("LMSA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))
    monkeypatch.setattr("sys.argv", ["phish_check"])
    pc.main()
    out = capsys.readouterr().out.strip()
    html = Path(out).read_text(encoding="utf-8")
    assert "Sent-folder scan capped" in html


# ---- finding 8: false positives ------------------------------------------------------------

def test_person_named_like_a_brand_is_not_a_brand_mismatch():
    msg = _msg(from_name="Chase Smith", from_addr="chase.smith@gmail.com")
    assert pc.brand_mismatch_signals(msg) == []


@pytest.mark.parametrize("name", ["Jane Doe via DocuSign", "Apple Pie Bakery", "Ups and Downs Club"])
def test_brand_word_inside_a_non_brand_name_is_not_flagged(name):
    assert pc.brand_mismatch_signals(_msg(from_name=name, from_addr="x@example.com")) == []


@pytest.mark.parametrize("name,tag", [
    ("Chase", "brand-mismatch:chase"),
    ("Chase Bank Alerts", "brand-mismatch:chase"),
    ("PayPal Support", "brand-mismatch:paypal"),
    ("service@paypal.com", "brand-mismatch:paypal"),
])
def test_brand_presented_as_identity_is_flagged(name, tag):
    tags = {s.tag for s in pc.brand_mismatch_signals(_msg(from_name=name, from_addr="x@example.com"))}
    assert tags == {tag}


@pytest.mark.parametrize("name,addr", [
    ("DocuSign", "dse@eumail.docusign.net"),
    ("Facebook", "notification@facebookmail.com"),
    ("PayPal", "service@paypal.co.uk"),
    ("Amazon", "shipment-tracking@amazon.de"),
    ("Chase", "no.reply.alerts@chase.com"),
])
def test_brand_alias_domains_are_not_mismatches(name, addr):
    assert pc.brand_mismatch_signals(_msg(from_name=name, from_addr=addr)) == []


@pytest.mark.parametrize("domain", ["paypal.co.uk", "docusign.net", "facebookmail.com", "amazon.de"])
def test_brand_alias_domains_are_not_lookalikes(domain):
    known = set(pc.BRAND_DOMAINS)
    assert pc.lookalike_signals(domain, known, tag_prefix="lookalike") == []
    assert pc.homograph_signals(domain, known, tag_prefix="homograph") == []


@pytest.mark.parametrize("domain", ["ubs.com", "upc.com", "usa.gov", "usbs.gov"])
def test_short_labels_are_not_lookalikes(domain):
    known = {"ups.com", "usps.gov"}
    assert pc.lookalike_signals(domain, known, tag_prefix="lookalike") == []


def test_lookalike_budget_scales_with_label_length():
    assert pc.lookalike_max_distance(4) == 0
    assert pc.lookalike_max_distance(5) == 1
    assert pc.lookalike_max_distance(7) == 1
    assert pc.lookalike_max_distance(8) == 2
    known = {"chase.com"}
    assert pc.lookalike_signals("chasr.com", known, tag_prefix="l")          # 1 edit, label 5
    assert pc.lookalike_signals("chxsr.com", known, tag_prefix="l") == []    # 2 edits, label 5
    assert pc.lookalike_signals("microsofft.com", {"microsoft.com"}, tag_prefix="l")
    assert pc.lookalike_signals("micrasofft.com", {"microsoft.com"}, tag_prefix="l")  # 2 edits, label 9


def test_lookalike_requires_similar_length():
    assert pc.lookalike_signals("abcdefghij.com", {"abcdefgh.com"}, tag_prefix="l")        # +2
    assert pc.lookalike_signals("abcdefghijk.com", {"abcdefgh.com"}, tag_prefix="l") == []  # +3


@pytest.mark.parametrize("host,expected", [
    ("shop.example.com.tr", "example.com.tr"),
    ("mail.example.co.za", "example.co.za"),
    ("portal.example.gov.br", "example.gov.br"),
    ("example.co.uk", "example.co.uk"),
    ("com.tr", ""),
    ("co.uk", ""),
    ("paypal.me", "paypal.me"),
])
def test_registrable_domain_two_level_suffixes(host, expected):
    assert pc.registrable_domain(host) == expected


def test_bare_suffix_never_enters_known_set():
    fake = _FakeGmail(sent_messages=[_msg(to=["a@com.tr", "b@shop.example.com.tr"])])
    scan = pc.known_domains_from_sent(fake)
    assert scan.domains == {"example.com.tr"}
    # an unrelated com.tr domain is not a "lookalike" of the bare suffix
    assert pc.lookalike_signals("other.com.tr", scan.domains, tag_prefix="l") == []


def test_uts46_alternate_dots_and_ignorables_do_not_hide_the_host():
    from tasks.phish_check import url_domain

    assert url_domain("http://paypa1。com/login") == "paypa1.com"
    assert url_domain("http://pay­pa1.com/login") == "paypa1.com"
    assert url_domain("http://pay​pa1.com/login") == "paypa1.com"


def test_link_domains_are_capped_and_the_cap_is_a_signal():
    from tasks.phish_check import MAX_LINK_DOMAINS, link_signals

    text = " ".join(f"https://site{i}.org/x" for i in range(MAX_LINK_DOMAINS + 10))
    signals, urls = link_signals(text, set())
    assert len(urls) == MAX_LINK_DOMAINS + 10
    assert [s.tag for s in signals if s.tag == "link-cap"] == ["link-cap"]


def test_one_passing_dkim_signature_is_enough():
    from tasks.phish_check import parse_auth_results

    header = "mx.google.com; dkim=pass header.i=@list.example.org; dkim=fail header.i=@origin.example.com; spf=pass; dmarc=pass"
    assert parse_auth_results(header)["dkim"] == "pass"
    # worst-wins still applies to the other methods
    assert parse_auth_results("mx.google.com; spf=pass; spf=fail")["spf"] == "fail"


# =====================================================================================
# Regression tests for #30: brand display-name bypasses and Japanese mixed-script FPs.
# =====================================================================================

# ---- brand display-name bypasses: one extra product/service word ---------------------

@pytest.mark.parametrize("name,tag", [
    ("Amazon Prime", "brand-mismatch:amazon"),
    ("Microsoft 365", "brand-mismatch:microsoft"),
    ("Microsoft Office", "brand-mismatch:microsoft"),
    ("Microsoft Outlook", "brand-mismatch:microsoft"),
    ("Apple Pay", "brand-mismatch:apple"),
    ("Google Workspace", "brand-mismatch:google"),
    ("PayPal Refunds", "brand-mismatch:paypal"),
    ("DocuSign Electronic Signature", "brand-mismatch:docusign"),
])
def test_brand_plus_product_word_is_flagged(name, tag):
    msg = _msg(from_name=name, from_addr="a@evil.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tag in tags


def test_two_brands_in_one_name_flags_both():
    msg = _msg(from_name="PayPal via DocuSign", from_addr="a@evil.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {"brand-mismatch:paypal", "brand-mismatch:docusign"}


# ---- brand display-name bypasses: word-boundary tricks (join/split) ------------------

@pytest.mark.parametrize("name,tag", [
    ("WellsFargo", "brand-mismatch:wells fargo"),
    ("Pay Pal", "brand-mismatch:paypal"),
    ("P a y P a l", "brand-mismatch:paypal"),
])
def test_brand_split_or_joined_across_words_is_flagged(name, tag):
    msg = _msg(from_name=name, from_addr="a@evil.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {tag}


def test_brand_in_name_word_boundary_join_alone_is_not_disguised():
    # A plain word-count mismatch (join/split) is honest spelling, not a
    # disguise -- it only presents because there's no leftover word left
    # unexplained, same as an exact one-word match would.
    presents, disguised = pc._brand_in_name("wellsfargo", "wells fargo")
    assert (presents, disguised) == (True, False)
    presents, disguised = pc._brand_in_name("pay pal", "paypal")
    assert (presents, disguised) == (True, False)


def test_brand_in_name_spaced_out_letters_is_disguised_even_without_leftover():
    presents, disguised = pc._brand_in_name("p a y p a l", "paypal")
    assert (presents, disguised) == (True, True)


# ---- brand display-name bypasses: capital I standing in for lowercase l --------------

def test_capital_i_for_lowercase_l_is_flagged():
    msg = _msg(from_name="PayPaI", from_addr="a@evil.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {"brand-mismatch:paypal"}


def test_capital_i_for_lowercase_l_counts_as_disguised():
    presents, disguised = pc._brand_in_name("PayPaI", "paypal")
    assert (presents, disguised) == (True, True)
    # a plain lowercase "i" is not the same trick and must not match "paypal"
    assert pc._brand_in_name("paypai", "paypal") == (False, False)


# A blanket "map every capital I to l" pass would corrupt any brand whose
# *real* spelling contains a capital I -- these all genuinely contain one
# and must still match on their own plain spelling, undisguised.
@pytest.mark.parametrize("name,tag", [
    ("IRS", "brand-mismatch:irs"),
    ("IRS Notice", "brand-mismatch:irs"),
    ("IRS Refunds", "brand-mismatch:irs"),
    ("LinkedIn", "brand-mismatch:linkedin"),
    ("LinkedIn Notifications", "brand-mismatch:linkedin"),
    ("Instagram", "brand-mismatch:instagram"),
    ("INSTAGRAM", "brand-mismatch:instagram"),
    ("NETFLIX", "brand-mismatch:netflix"),
    ("MICROSOFT", "brand-mismatch:microsoft"),
])
def test_brand_with_a_genuine_capital_i_still_matches_undisguised(name, tag):
    msg = _msg(from_name=name, from_addr="a@evil.com")
    tags = {s.tag for s in pc.brand_mismatch_signals(msg)}
    assert tags == {tag}
    presents, disguised = pc._brand_in_name(name, tag.split(":", 1)[1])
    assert (presents, disguised) == (True, False)


# ---- brand display-name bypasses: still no false positives on real names -------------

@pytest.mark.parametrize("name", [
    # People whose name happens to contain a brand word plus an innocuous
    # word -- must not be excused into flagging just because a brand word
    # is technically present somewhere.
    "Chase Smith", "Apple Valley Dental", "Apple Pie Bakery",
    "Ups and Downs Club", "Amazon Smith", "Chase Bank Robertson",
    "Pay Palmer",
    # Word-count-mismatch (split/join) matches with a genuine leftover word
    # must still fail the leftover check, not bypass it.
    "Chas E Smith", "Chas E. Walker", "Fed Ex Smith", "Micro Soft Jones",
    "Drop Box Repair Shop", "Face Book Club",
    # A bare second brand name (no "via" relay wording) must not excuse a
    # leftover, and a single word-fragment of a multi-word brand
    # ("wells", "of", "america") is never a *complete* other name.
    "Jane Doe via DocuSign", "Chase Wells", "Wells Chase", "Chase America",
    "Apple Of America", "Chase Apple",
    # A brand's own product word must not excuse an unrelated brand.
    "Chase Pay", "Chase Office",
])
def test_brand_word_bypass_fix_does_not_reintroduce_false_positives(name):
    msg = _msg(from_name=name, from_addr="a@example.com")
    assert pc.brand_mismatch_signals(msg) == []


# ---- Japanese mixed-script false positives --------------------------------------------

@pytest.mark.parametrize("domain", [
    "コーヒー.jp",  # コーヒー.jp -- katakana + prolonged sound mark
    "佐々木.jp",         # 佐々木.jp -- kanji + iteration mark
])
def test_japanese_prolonged_sound_mark_and_iteration_mark_are_not_mixed_script(domain):
    assert pc.homograph_signals(domain, set(), tag_prefix="homograph") == []


def test_genuine_cross_script_mixing_still_flagged():
    # Latin "pay" + Cyrillic "ал" in one label must still be caught.
    domain = "payал.com"
    signals = pc.homograph_signals(domain, set(), tag_prefix="homograph")
    assert len(signals) == 1
    assert "mixes scripts" in signals[0].detail
