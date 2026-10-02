"""DoD contract-award ledger (#57): defense.gov's daily contracts page,
split into per-award paragraphs and parsed into a ledger table, plus an
opt-in report extension that renders it.

    [[sources.api]]
    adapter = "dod_contracts"
    # optional override of the day's contracts page or listing/RSS feed
    # (defaults to DEFAULT_LISTING_URL below -- UNVERIFIED live, see
    # config/briefings/defense.toml):
    # url = "https://www.defense.gov/News/Contracts/"

This is a *side-effect* adapter, like #56's geo_metrics: it writes straight
to its own `dod_contracts` table via `ctx.conn` and yields no `Doc`s to the
story pipeline. Two reasons: every award on a page shares that page's URL,
which would collide in the engine's per-URL dedup (only the first award
would survive); and the ledger's 7/30/90-day totals need history the
current run's docs alone can't provide. `ctx.limit` (the entry's cap + 1,
same as every adapter) still bounds how many award paragraphs are written
per run, and a cap that bit is disclosed via `ctx.note()`.

The `dod_contracts` report extension (opt-in -- list it in a briefing's
`extensions`, it is not `default=True`) reads the table directly and adds
"today's awards" / "totals by service" / "top contractors (7/30/90d)" /
"notable" sections after the stories.

Defense.gov's real page structure cannot be verified from here (no live
fetch in this environment): parsing is deliberately defensive -- a
non-empty page that yields zero recognized award paragraphs is treated as
an *unrecognized response* (disclosed as a note), never as "zero awards
today". Amounts are parsed with regexes only; the model never fills in a
dollar figure it can't find (`parse_amount` returns `None` and a note
instead of guessing).
"""

from __future__ import annotations

import hashlib
import html as _html
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta
from typing import Iterable
from urllib.parse import urljoin

from ...memo import Memo, memo_version, model_id
from ...prompts import load as load_prompt
from ...prompts import prompt_hash
from ...report import Report
from ...vectors import embed_and_upsert
from ...web import FetchError
from ..sources import Doc, SourceContext, register, register_extension

log = logging.getLogger(__name__)

TASK = "briefing.dod_contracts"
MEMO_VERSION = "dod-contracts-1"
SYSTEM_PROMPT = "briefing_defense/extract_system"
USER_PROMPT = "briefing_defense/extract_user"

# Vectors namespace for #68's `tasks.ask --scope contracts`: one vector per
# ledger row, item id `<day>:<identity_key>` (`contract_item_id`) so a
# re-published award (same day, same identity_key) overwrites its vector
# instead of adding a new one.
VECTOR_NS = "contracts"

# UNVERIFIED live (no network in this environment) -- check this URL still
# resolves to the contracts listing on the first real run; if defense.gov's
# layout moved, `[[sources.api]] url = "..."` overrides it per-briefing.
DEFAULT_LISTING_URL = "https://www.defense.gov/News/Contracts/"

MAX_PARAGRAPH_CHARS = 2_000  # sent to the model, per award paragraph

# Configurable in code (no per-extension config table in briefing TOML):
NOTABLE_THRESHOLD_USD = 100_000_000.0
WATCHLIST_KEYWORDS = (
    "hypersonic", "nuclear", "submarine", "classified", "space force",
    "missile defense", "cyber", "artificial intelligence", "directed energy",
)
WINDOWS_DAYS = (7, 30, 90)
TOP_N_CONTRACTORS = 10

SERVICES = (
    "ARMY", "NAVY", "AIR FORCE", "SPACE FORCE", "MARINE CORPS",
    "DEFENSE LOGISTICS AGENCY", "DEFENSE HEALTH AGENCY",
    "MISSILE DEFENSE AGENCY", "DEFENSE INFORMATION SYSTEMS AGENCY",
    "DEFENSE THREAT REDUCTION AGENCY", "U.S. SPECIAL OPERATIONS COMMAND",
    "U.S. TRANSPORTATION COMMAND", "WASHINGTON HEADQUARTERS SERVICES",
    "DEFENSE COMMISSARY AGENCY", "NATIONAL GEOSPATIAL-INTELLIGENCE AGENCY",
    "DEFENSE COUNTERINTELLIGENCE AND SECURITY AGENCY",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_P_SPLIT_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
_HEADING_RE = re.compile(r"^[A-Z][A-Z .,&/-]{2,80}$")
_AWARD_VERB_RE = re.compile(
    r"\b(?:has been awarded|was awarded|is awarded|have been awarded|is being awarded)\b", re.I)
_MODIFICATION_RE = re.compile(r"\bmodification\b", re.I)

_CONTRACT_NUM_CANDIDATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9./-]{4,29}$")

_CUMULATIVE_RE = re.compile(
    r"(?:bringing the total (?:cumulative )?(?:face )?value(?: of (?:this|the) contract)? to|"
    r"cumulative face value of|total cumulative face value of)\s*\$([\d,]+(?:\.\d+)?)", re.I)
# "ceiling"/"maximum" figure, either order -- defense.gov writes both
# "$45,000,000 ceiling" and "the maximum dollar value ... is $950,000,000".
# The prefix form's gap is bounded to one sentence (no '.' or '$' inside it)
# so an unrelated "maximum" earlier in the paragraph can't reach across to a
# later dollar figure.
_CEILING_POSTFIX_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*(?:ceiling|maximum)", re.I)
_CEILING_PREFIX_RE = re.compile(
    r"(?:maximum|ceiling)(?:\s+dollar)?(?:\s+value)?[^.$]{0,80}?\$([\d,]+(?:\.\d+)?)", re.I)
_WORD_MULT_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)\b", re.I)
_PLAIN_AMOUNT_RE = re.compile(r"\$(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_MULTIPLIERS = {"thousand": 1_000.0, "million": 1_000_000.0, "billion": 1_000_000_000.0}
_OBLIGATED_RE = re.compile(r"\bobligat\w*\b", re.I)

_TYPE_KEYWORDS = (
    "firm-fixed-price", "cost-plus-fixed-fee", "cost-plus-incentive-fee",
    "cost-plus-award-fee", "indefinite-delivery/indefinite-quantity",
    "time-and-materials", "cost-reimbursement", "fixed-price-incentive",
    "labor-hour",
)
_COMPLETION_RE = re.compile(
    r"(?:is expected to be completed|work will be completed|"
    r"with an? (?:estimated )?completion date of|completion date is)"
    r"\s*(?:by|on|is)?\s*([A-Z][a-z]+\.?\s+\d{1,2},\s*\d{4})", re.I)

_DAY_LINK_RE = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*>\s*([A-Z][a-z]+\.?\s+\d{1,2},\s*\d{4})\s*</a>', re.I)


# ---------------------------------------------------------------------------
# Parsing (deterministic, no model)
# ---------------------------------------------------------------------------

def _strip(text: str) -> str:
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", text))).strip()


def split_paragraphs(page_html: str) -> list[str]:
    """Plain-text paragraphs in document order. A page with no `<p>` tags at
    all yields nothing -- the caller discloses that as an unrecognized page,
    not zero awards."""
    return [t for t in (_strip(raw) for raw in _P_SPLIT_RE.findall(page_html)) if t]


def is_heading(paragraph: str) -> bool:
    upper = paragraph.strip().rstrip(":").upper()
    if not upper or _AWARD_VERB_RE.search(paragraph):
        return False
    return upper in SERVICES or (len(upper) <= 60 and bool(_HEADING_RE.match(upper)))


def is_award_paragraph(paragraph: str) -> bool:
    return "$" in paragraph and bool(_AWARD_VERB_RE.search(paragraph))


def group_by_service(paragraphs: Iterable[str]) -> list[tuple[str, str]]:
    """`[(service, paragraph)]` for every non-heading paragraph; `service`
    is the nearest preceding ALL-CAPS heading (defense.gov's service
    sections), or "" for text before the first one."""
    out: list[tuple[str, str]] = []
    current = ""
    for p in paragraphs:
        if is_heading(p):
            current = p.strip().rstrip(":").upper()
            continue
        out.append((current, p))
    return out


def parse_amount(paragraph: str) -> tuple[float | None, str]:
    """Deterministic amount parse. Priority, searched anywhere in the
    paragraph regardless of word order: a stated cumulative/running total
    (the honest ledger figure for a modification) > a ceiling/maximum/face
    value figure (an IDIQ's real size) > a "$X billion/million/thousand"
    word-multiplier > the first plain "$1,234,567"-style figure. A figure
    obligated "at time of award" is routinely far smaller than an IDIQ's
    ceiling and must never outrank a ceiling/maximum/cumulative figure just
    because it happens to appear first in the text.

    Returns `(amount, note)`; `amount` is `None` when nothing matches -- the
    model never fills this in. When only a plain/word-multiplier figure
    could be found and it sits in an "obligated" clause, `note` says so
    (non-empty) instead of silently presenting funds-obligated-so-far as if
    it were the contract's award value."""
    m = _CUMULATIVE_RE.search(paragraph)
    if m:
        return _to_float(m.group(1)), "cumulative total"
    m = _CEILING_POSTFIX_RE.search(paragraph) or _CEILING_PREFIX_RE.search(paragraph)
    if m:
        return _to_float(m.group(1)), "ceiling"
    m = _WORD_MULT_RE.search(paragraph)
    if m:
        val = _to_float(m.group(1))
        mult = _MULTIPLIERS[m.group(2).lower()]
        amount = val * mult if val is not None else None
        return amount, _obligated_note(paragraph, m.start())
    m = _PLAIN_AMOUNT_RE.search(paragraph)
    if m:
        return _to_float(m.group(1)), _obligated_note(paragraph, m.start())
    return None, "no amount found by regex"


def _obligated_note(paragraph: str, match_pos: int) -> str:
    """Flag a fallback figure (no cumulative/ceiling/maximum wording found
    anywhere) that sits in the same sentence as "obligat[ed/ion]" -- that is
    funds committed so far, not necessarily the contract's full award/face
    value, so it must not be presented as an unqualified amount."""
    start = paragraph.rfind(".", 0, match_pos) + 1
    end = paragraph.find(".", match_pos)
    sentence = paragraph[start:] if end == -1 else paragraph[start:end]
    if _OBLIGATED_RE.search(sentence):
        return "obligated amount, not a stated ceiling/maximum"
    return ""


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_contract_number(paragraph: str) -> str:
    """The last parenthesized token that looks like a contract number
    (alnum, at least one letter and one digit, no spaces) -- defense.gov
    conventionally puts it last."""
    for cand in reversed(re.findall(r"\(([^()]{3,40})\)", paragraph)):
        c = cand.strip().rstrip(".")
        if (_CONTRACT_NUM_CANDIDATE_RE.match(c) and any(ch.isdigit() for ch in c)
                and any(ch.isalpha() for ch in c)):
            return c
    return ""


def parse_contract_type(paragraph: str) -> str:
    low = paragraph.lower()
    for kw in _TYPE_KEYWORDS:
        if kw in low:
            return kw
    return ""


def parse_completion_date(paragraph: str) -> str:
    m = _COMPLETION_RE.search(paragraph)
    return m.group(1) if m else ""


def is_modification(paragraph: str) -> bool:
    return bool(_MODIFICATION_RE.search(paragraph))


def find_day_links(page_html: str, base_url: str) -> list[tuple[str, str]]:
    """`[(absolute_url, date_text)]` for every listing link whose text looks
    like a date (defense.gov's contracts listing links to one page per
    release day)."""
    return [(urljoin(base_url, href), date_text) for href, date_text in _DAY_LINK_RE.findall(page_html)]


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def _parse_link_date(text: str) -> date | None:
    cleaned = text.strip().rstrip(".").replace("Sept", "Sep")
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", cleaned)
    if not m:
        return None
    month = _MONTHS.get(m.group(1)[:3].lower())
    if month is None:
        return None
    try:
        return date(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def pick_day_link(links: list[tuple[str, str]], today: date) -> str | None:
    """The link dated `today`, else the most recently dated link (defense.gov
    publishes on business days, so "today" may not have a page yet)."""
    dated = [(url, _parse_link_date(text)) for url, text in links]
    dated = [(url, d) for url, d in dated if d is not None]
    for url, d in dated:
        if d == today:
            return url
    if not dated:
        return None
    return max(dated, key=lambda pair: pair[1])[0]


# ---------------------------------------------------------------------------
# Model-assisted extraction (memoized by paragraph hash)
# ---------------------------------------------------------------------------

EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "contractor": {"type": "string"},
        "location": {"type": "string"},
        "program_or_item": {"type": "string"},
    },
    "required": ["contractor", "location", "program_or_item"],
    "additionalProperties": False,
}


def paragraph_hash(paragraph: str) -> str:
    return hashlib.sha256(paragraph.encode("utf-8")).hexdigest()[:16]


def _extract_fields(paragraph: str, ctx: SourceContext, memo: Memo) -> dict:
    """contractor/location/program_or_item -- the fields a regex can't
    reliably pull out of free text. Memoized by paragraph hash so a rerun
    of the same day's page (same paragraph text) makes no new model calls."""
    key = paragraph_hash(paragraph)

    def compute() -> dict:
        system = load_prompt(SYSTEM_PROMPT)
        user = load_prompt(USER_PROMPT, paragraph=paragraph[:MAX_PARAGRAPH_CHARS])
        return ctx.lm.extract(user, EXTRACT_SCHEMA, system=system, name="dod_contract")

    result = memo.cached(
        key, compute,
        valid=lambda r: isinstance(r, dict) and all(k in r for k in EXTRACT_SCHEMA["required"]),
    )
    if not isinstance(result, dict):
        return {"contractor": "", "location": "", "program_or_item": ""}
    return {k: str(result.get(k) or "")[:200] for k in ("contractor", "location", "program_or_item")}


def extract_award(paragraph: str, service: str, ctx: SourceContext, memo: Memo) -> dict:
    """Combine deterministic (regex) fields with the model-filled ones into
    one award dict, ready for `record_award`."""
    amount_usd, amount_note = parse_amount(paragraph)
    fields = _extract_fields(paragraph, ctx, memo)
    return {
        "service": service,
        "contractor": fields["contractor"],
        "location": fields["location"],
        "program_or_item": fields["program_or_item"],
        "contract_type": parse_contract_type(paragraph),
        "completion_date": parse_completion_date(paragraph),
        "contract_number": parse_contract_number(paragraph),
        "amount_usd": amount_usd,
        "amount_note": amount_note,
        "is_modification": is_modification(paragraph),
    }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def award_identity_key(contract_number: str, paragraph: str) -> str:
    """The identity of an award within one day: the contract number when
    one was parsed (so a corrected / re-published announcement of the same
    contract -- wording tweaked, contract number unchanged -- upserts the
    same row instead of duplicating it), else the paragraph hash alone (the
    only thing left to key a numberless award on)."""
    return contract_number or paragraph_hash(paragraph)


def contract_item_id(day: str, identity_key: str) -> str:
    """Vectors item id for one ledger row (`<day>:<identity_key>`, #68) --
    a re-published award (same day, same `award_identity_key`) overwrites
    its vector in place instead of adding a new one."""
    return f"{day}:{identity_key}"


def _award_embed_text(award: dict) -> str:
    """Contractor, program/item, service, location, amount and contract
    number, joined into one string for `tasks.ask --scope contracts` to
    match a query against. Empty when nothing usable was parsed at all."""
    amount = award.get("amount_usd")
    amount_text = f"${amount:,.0f}" if isinstance(amount, (int, float)) else ""
    parts = [
        award.get("contractor") or "",
        award.get("program_or_item") or "",
        award.get("service") or "",
        award.get("location") or "",
        amount_text,
        award.get("contract_number") or "",
    ]
    return " · ".join(p for p in parts if p)


def _embed_award(ctx: SourceContext, day: str, identity_key: str, award: dict) -> None:
    """Embed one ledger row into the `contracts` vectors namespace. Never
    raises: the ledger row this is called for has already been committed,
    so a failed embedding call (model down, bad response) is noted and
    otherwise ignored -- it must never look like the award wasn't
    recorded."""
    text = _award_embed_text(award)
    if not text:
        return
    item_id = contract_item_id(day, identity_key)
    try:
        embed_and_upsert(ctx.conn, ctx.lm, ns=VECTOR_NS, model=ctx.lm.cfg.embed_model, items=[(item_id, text)])
    except Exception as exc:  # noqa: BLE001 - embedding is best-effort, ledger row already committed
        log.warning("dod_contracts: embedding failed for %s", item_id, exc_info=True)
        ctx.note(f"dod_contracts: embedding failed for {item_id} ({type(exc).__name__}); ledger row kept")


def record_award(conn: sqlite3.Connection, day: str, paragraph: str, award: dict, *,
                  source_url: str, now: datetime, ctx: SourceContext | None = None) -> None:
    """Upsert one ledger row, keyed on (day, identity_key) -- see
    `award_identity_key`. A rerun of the same day's page, even with a word
    or two of wording changed by a corrected republication, updates the row
    in place (refreshing `last_seen` and the parsed fields) instead of
    duplicating it and double-counting the amount; `first_seen` is
    preserved across reruns.

    When `ctx` is given, the award is also embedded into the `contracts`
    vectors namespace (#68) -- but only when the row is new or its
    paragraph text changed since it was last recorded, or it has no vector
    yet (an earlier embed failed) -- so a same-text rerun of an already
    embedded award makes no embedding call at all."""
    now_iso = now.isoformat()
    identity_key = award_identity_key(award["contract_number"], paragraph)
    new_hash = paragraph_hash(paragraph)
    existing = conn.execute(
        "SELECT paragraph_hash FROM dod_contracts WHERE day=? AND identity_key=?", (day, identity_key),
    ).fetchone()
    changed = existing is None or existing[0] != new_hash
    conn.execute(
        "INSERT INTO dod_contracts (identity_key, contract_number, day, paragraph_hash, service, "
        "contractor, location, program_or_item, contract_type, completion_date, amount_usd, "
        "amount_note, is_modification, source_url, raw_text, first_seen, last_seen) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(day, identity_key) DO UPDATE SET "
        "contract_number=excluded.contract_number, paragraph_hash=excluded.paragraph_hash, "
        "service=excluded.service, contractor=excluded.contractor, location=excluded.location, "
        "program_or_item=excluded.program_or_item, contract_type=excluded.contract_type, "
        "completion_date=excluded.completion_date, amount_usd=excluded.amount_usd, "
        "amount_note=excluded.amount_note, is_modification=excluded.is_modification, "
        "source_url=excluded.source_url, raw_text=excluded.raw_text, last_seen=excluded.last_seen",
        (identity_key, award["contract_number"], day, new_hash, award["service"],
         award["contractor"], award["location"], award["program_or_item"], award["contract_type"],
         award["completion_date"], award["amount_usd"], award["amount_note"],
         int(bool(award["is_modification"])), source_url, paragraph[:4000], now_iso, now_iso),
    )
    conn.commit()
    if ctx is not None and (changed or not _has_vector(ctx, day, identity_key)):
        _embed_award(ctx, day, identity_key, award)


def _has_vector(ctx: SourceContext, day: str, identity_key: str) -> bool:
    """Whether this award already has a vector under the current embed model
    -- so an award whose embedding failed on an earlier run is retried."""
    return ctx.conn.execute(
        "SELECT 1 FROM embeddings WHERE ns=? AND item_id=? AND model=?",
        (VECTOR_NS, contract_item_id(day, identity_key), ctx.lm.cfg.embed_model),
    ).fetchone() is not None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def _fetch_page(ctx: SourceContext, url: str) -> str | None:
    try:
        return ctx.fetcher.get(url).text
    except FetchError as exc:
        ctx.note(f"dod_contracts: could not fetch {url}: {type(exc).__name__}")
        return None


@register("dod_contracts", keys=("url",))
def dod_contracts_adapter(entry: dict, ctx: SourceContext) -> list[Doc]:
    """Fetch the day's contracts page (discovering it from the listing/RSS
    if the configured URL is the listing itself), parse it into award
    paragraphs, and write each one to the `dod_contracts` ledger. Writes
    directly via `ctx.conn` and yields no `Doc`s -- see the module
    docstring for why."""
    listing_url = entry.get("url") or DEFAULT_LISTING_URL
    page_url = listing_url
    page_html = _fetch_page(ctx, listing_url)
    if page_html is None:
        return []

    paragraphs = split_paragraphs(page_html)
    grouped = group_by_service(paragraphs)
    awards = [(svc, p) for svc, p in grouped if is_award_paragraph(p)]

    if not awards:
        # The configured URL may be a listing/RSS page that links to the
        # actual day's release rather than containing it.
        links = find_day_links(page_html, listing_url)
        target = pick_day_link(links, ctx.now.date())
        if target is not None:
            day_html = _fetch_page(ctx, target)
            if day_html is not None:
                page_url, page_html = target, day_html
                paragraphs = split_paragraphs(page_html)
                grouped = group_by_service(paragraphs)
                awards = [(svc, p) for svc, p in grouped if is_award_paragraph(p)]

    if not paragraphs:
        ctx.note(f"dod_contracts: {page_url} had no readable content")
        return []
    if not awards:
        ctx.note(f"dod_contracts: {page_url} returned a page but no award paragraphs were "
                  "recognized -- treating this as an unrecognized response, not as zero awards today")
        return []

    day = ctx.now.strftime("%Y-%m-%d")
    version = memo_version(MEMO_VERSION, prompt_hash(SYSTEM_PROMPT), prompt_hash(USER_PROMPT),
                            EXTRACT_SCHEMA, model_id(ctx.lm))
    memo = Memo(ctx.conn, TASK, version, fresh=ctx.fresh)

    cap = max(ctx.limit - 1, 0)  # ctx.limit is the kept cap + 1, same as every adapter
    kept = 0
    for service, paragraph in awards:
        if kept >= cap:
            ctx.note(f"dod_contracts: cut at {cap} award(s) for this run; more were on the page")
            break
        try:
            award = extract_award(paragraph, service, ctx, memo)
            record_award(ctx.conn, day, paragraph, award, source_url=page_url, now=ctx.now, ctx=ctx)
        except Exception as exc:  # one bad paragraph never kills the run
            log.warning("dod_contracts: one award paragraph failed", exc_info=True)
            ctx.note(f"dod_contracts: one award paragraph failed ({type(exc).__name__}); skipped")
            continue
        kept += 1
    ctx.note(f"dod_contracts: recorded {kept} award(s) from {page_url}")
    return []


# ---------------------------------------------------------------------------
# Report extension (opt-in)
# ---------------------------------------------------------------------------

def _window_totals(rows: list[tuple[str, str, float | None]]) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Sum `rows` (one row per `(day, identity_key)`, per the ledger's
    primary key -- see `award_identity_key`) by service and by contractor.
    Safe from double-counting a re-published announcement: the ledger
    upserts same-day, same-contract rows in place instead of duplicating
    them, so each award contributes exactly once per day it was ever
    reported on."""
    by_service: dict[str, float] = {}
    by_contractor: dict[str, float] = {}
    for service, contractor, amount in rows:
        if amount is None:
            continue
        by_service[service or "(unspecified)"] = by_service.get(service or "(unspecified)", 0.0) + amount
        if contractor:
            by_contractor[contractor] = by_contractor.get(contractor, 0.0) + amount
    services = sorted(by_service.items(), key=lambda kv: -kv[1])
    contractors = sorted(by_contractor.items(), key=lambda kv: -kv[1])[:TOP_N_CONTRACTORS]
    return services, contractors


def _money(value: float | None) -> str:
    if value is None:
        return "unk."
    return f"${value:,.0f}"


_CORP_SUFFIX_RE = re.compile(r"\b(?:Inc\.?|Corp\.?|Corporation|L\.?P\.?|LLC|Ltd\.?|Co\.?|Company)\b")


def _is_joint_award(contractor: str) -> bool:
    """True when `contractor` names more than one company (defense.gov
    sometimes credits one award to a teaming arrangement of two firms). The
    ledger keeps a joint award under one composite contractor string (no
    schema change), so the report must label it rather than let a reader
    assume one company received the whole amount alone."""
    return len(_CORP_SUFFIX_RE.findall(contractor)) > 1


def _contractor_label(contractor: str) -> str:
    name = contractor or "(unnamed)"
    if contractor and _is_joint_award(contractor):
        name += " (joint award, split unknown)"
    return name


def _is_notable(row: dict) -> bool:
    amount = row["amount_usd"]
    if amount is not None and amount >= NOTABLE_THRESHOLD_USD:
        return True
    text = f"{row['program_or_item']} {row['contractor']}".lower()
    return any(kw in text for kw in WATCHLIST_KEYWORDS)


_TODAY_COLUMNS = (
    "contract_number", "day", "service", "contractor", "location", "program_or_item",
    "contract_type", "completion_date", "amount_usd", "amount_note", "is_modification",
)


def render_ledger_section(report: Report, stories: list, ctx: SourceContext) -> None:
    conn = getattr(ctx, "conn", None)
    if conn is None:
        return
    try:
        today = ctx.now.strftime("%Y-%m-%d")
        rows = conn.execute(
            f"SELECT {', '.join(_TODAY_COLUMNS)} FROM dod_contracts WHERE day = ? "
            "ORDER BY (amount_usd IS NULL), amount_usd DESC", (today,),
        ).fetchall()
    except sqlite3.Error:
        log.warning("dod_contracts: ledger read failed", exc_info=True)
        return
    today_rows = [dict(zip(_TODAY_COLUMNS, r)) for r in rows]

    sec = report.section("DoD contract awards", tone="info")
    if not today_rows:
        sec.callout("No awards recorded for today (either none published yet, or the page "
                    "wasn't recognized -- see Notes).", label="ledger", tone="muted")
    else:
        sec.table(
            ["Amount", "Contractor", "Service", "Program / item", "Location", "Contract #"],
            [(_money(r["amount_usd"]) + (" (mod.)" if r["is_modification"] else ""),
              _contractor_label(r["contractor"]), r["service"] or "(unspecified)",
              r["program_or_item"], r["location"], r["contract_number"] or "(none found)")
             for r in today_rows],
        )

    notable = [r for r in today_rows if _is_notable(r)]
    if notable:
        note_sec = report.section("Notable awards", tone="hot")
        note_sec.table(
            ["Amount", "Contractor", "Program / item", "Why notable"],
            [(_money(r["amount_usd"]), _contractor_label(r["contractor"]), r["program_or_item"],
              "above threshold" if (r["amount_usd"] or 0) >= NOTABLE_THRESHOLD_USD else "watchlist keyword")
             for r in notable],
        )

    for days in WINDOWS_DAYS:
        since = (ctx.now - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            window_rows = conn.execute(
                "SELECT service, contractor, amount_usd FROM dod_contracts WHERE day BETWEEN ? AND ?",
                (since, today),
            ).fetchall()
        except sqlite3.Error:
            continue
        services, contractors = _window_totals(window_rows)
        win_sec = report.section(f"Totals, last {days}d", tone="acc")
        if not services:
            win_sec.callout(f"No ledger rows in the last {days} days.", label="ledger", tone="muted")
            continue
        win_sec.table(["Service", "Total"], [(s, _money(v)) for s, v in services])
        if contractors:
            win_sec.table(["Top contractors", "Total"], [(_contractor_label(c), _money(v)) for c, v in contractors])


register_extension("dod_contracts", section=render_ledger_section, default=False)
