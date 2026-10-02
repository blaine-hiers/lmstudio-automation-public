"""Read-only Gmail source over IMAP.

Scripts can't use the LM Studio email plugin's tools -- plugin tools only run
inside LM Studio chats, not over the `/v1` API these tasks call. So tasks that
need Gmail read it directly over IMAP with a Gmail app password, the same way
the plugin does.

    with Gmail() as gmail:
        messages = gmail.search("newer_than:1d category:purchases")

Wire-level IMAP handling (mailbox name quoting and modified UTF-7 encoding,
literal/continuation handling, Gmail's X-GM-MSGID/X-GM-THRID/X-GM-LABELS
extensions) is delegated to `imapclient`, a maintained third-party library --
a hand-rolled `imaplib` response parser kept getting these details wrong.
MIME parsing of the fetched raw message stays stdlib `email`.

By default this module writes nothing: the mailbox is opened readonly, and
every fetch uses `BODY.PEEK[...]` so nothing is ever marked read. There are
a few narrow, explicit write paths, each used by one task:

- `append_draft` (reply drafts, #17): one IMAP APPEND to the Drafts folder
  (resolved by its `\\Drafts` special-use flag) with the `\\Draft` flag set.
  It never selects that folder.
- `add_labels`/`remove_labels` (auto-labeler, #16): select the folder
  non-readonly only for that one call, issue Gmail's `STORE +/-X-GM-LABELS`
  extension, and re-select readonly before returning (if that re-select
  fails, the session is dropped rather than left writable). Gmail system
  labels (`is_reserved_label`: \\Trash, \\Spam, \\Inbox, CATEGORY_*,
  [Gmail]/...) are refused before any network call.

- `archive`/`unarchive` and `mark_read`/`mark_unread` (inbox actions, #61):
  same scoped writable select, hard-coded to exactly one change each --
  STORE -/+X-GM-LABELS (\\Inbox) or STORE -/+FLAGS (\\Seen). They take no
  label or flag argument, so no other system label is reachable. Used only
  by `tasks.inbox_actions` (dry-run by default, logged per run, undoable).

None of these ever issues EXPUNGE, COPY, MOVE, or delete, and there is no
SMTP anywhere in this module: nothing here sends mail.
"""

from __future__ import annotations

import html as htmllib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header
from email.message import Message as EmailMessage
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from typing import Any

import imapclient
from imapclient.exceptions import IMAPClientAbortError, IMAPClientError
from imapclient.imap_utf7 import decode as decode_utf7

from .. import secrets
from ..config import Settings, settings

DEFAULT_FOLDER = "[Gmail]/All Mail"
TEXT_CAP = 20_000  # characters; keeps a message body well inside model context
SENT_FLAG = rb"\Sent"  # RFC 6154 special-use flag; not exported by imapclient's __all__
DRAFTS_FLAG = rb"\Drafts"  # same story as SENT_FLAG -- not exported by imapclient's __all__
ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB; callers fetching bigger get AttachmentTooLargeError

log = logging.getLogger(__name__)

# Gmail system labels/mailboxes that the label write path must never touch:
# adding \Trash or \Spam moves mail there, removing \Inbox archives it, etc.
_RESERVED_LABEL_NAMES = frozenset(
    {"INBOX", "SPAM", "TRASH", "DRAFT", "DRAFTS", "SENT", "STARRED", "IMPORTANT", "UNREAD", "CHAT"}
)
_RESERVED_LABEL_PREFIXES = ("CATEGORY_", "[GMAIL]/", "[GOOGLE MAIL]/")


def is_reserved_label(label: str) -> bool:
    """True for a Gmail system label/mailbox (anything starting with a
    backslash, CATEGORY_*, INBOX/SPAM/TRASH/..., or under [Gmail]/) --
    never a user label this module may add or remove. Case-insensitive."""
    name = (label or "").strip()
    upper = name.upper()
    return (
        not name
        or name.startswith("\\")
        or upper in _RESERVED_LABEL_NAMES
        or upper in ("[GMAIL]", "[GOOGLE MAIL]")
        or upper.startswith(_RESERVED_LABEL_PREFIXES)
    )


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")


def _shutdown(client: imapclient.IMAPClient) -> None:
    try:
        client.shutdown()
    except Exception:
        pass


class GmailAuthError(RuntimeError):
    """Missing or invalid IMAP credentials."""


class LabelWriteError(RuntimeError):
    """A label STORE failed. `maybe_applied` is True when the outcome is
    unknown (e.g. the connection dropped mid-STORE, so the server may have
    applied it) and False when the server explicitly refused it."""

    def __init__(self, message: str, *, maybe_applied: bool):
        super().__init__(message)
        self.maybe_applied = maybe_applied


class AttachmentTooLargeError(ValueError):
    """The attachment exceeds `attachment_bytes`'s `max_size` cap."""


class AttachmentNotFoundError(RuntimeError):
    """The message, or the attachment at the given index, could not be refetched."""


@dataclass(frozen=True)
class Attachment:
    name: str
    mime: str
    size: int


@dataclass(frozen=True)
class MessageMeta:
    """What `Gmail.search_meta` returns per match: enough to order and pick
    messages without fetching any bodies. `uid` is only valid for the
    folder and session it came from (see `Gmail.fetch_uids`)."""

    msg_id: str
    internal_date: datetime | None
    uid: int


@dataclass(frozen=True)
class Message:
    msg_id: str
    thread_id: str
    labels: list[str]
    flags: list[str]
    from_name: str
    from_addr: str
    to: list[str]
    reply_to: str
    subject: str
    date: datetime | None
    list_unsubscribe: str
    list_id: str
    auth_results: str
    text: str
    attachments: list[Attachment]
    web_url: str
    # Added additively for #17 (reply drafts, which thread via In-Reply-To /
    # References): default "" / () so existing keyword-constructed Messages
    # (tests, other tasks) don't need updating.
    message_id: str = ""
    references: str = ""
    in_reply_to: str = ""
    # Every address on Reply-To (a header can legally carry more than one),
    # in header order. `reply_to` above stays a single address for backward
    # compatibility with existing callers; this is additive for #17, which
    # needs to see (and draft to) all of them.
    reply_to_all: tuple[str, ...] = ()
    # Gmail's INTERNALDATE (when the message was delivered to the mailbox),
    # aware. Additive, default None. This -- not the sender-controlled Date
    # header in `date`, which can be skewed, stale, or forged -- is what
    # Gmail's `after:`/`before:` search operators filter on, so every
    # since-last-run cursor orders and bounds by it (see `lmsa.gmail_cursor`).
    internal_date: datetime | None = None


def _decode_header_value(value: object) -> str:
    """Decode a header to text: RFC 2047 encoded words, and raw 8-bit UTF-8
    (SMTPUTF8 mail), which compat32 hands back as an `email.header.Header`
    with an unknown-8bit charset. Tolerates None and bare str."""
    if not value:
        return ""
    try:
        parts = decode_header(value)
    except (ValueError, LookupError, TypeError):
        return str(value)
    out = []
    for chunk, charset in parts:
        if isinstance(chunk, str):
            out.append(chunk)
        elif charset in (None, "unknown-8bit"):
            try:
                out.append(chunk.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(chunk.decode("latin-1"))
        else:
            try:
                out.append(chunk.decode(charset, errors="replace"))
            except LookupError:
                out.append(chunk.decode("utf-8", errors="replace"))
    return "".join(out)


def _unfold_header(value: str) -> str:
    """Collapse RFC 5322 header folding (a CRLF/LF followed by leading
    whitespace, used to wrap a long header across physical lines) into a
    single line.

    `email`'s compat32 policy hands header values back with that folding
    still embedded (verified against a real folded `References` header:
    `msg.get("References")` returns `"<a@x>\\r\\n <b@x>"`, literal CRLF and
    all). That's harmless for display, but `email.message.EmailMessage`
    raises "Header values may not contain linefeed" if you try to set a new
    header to a string containing one -- which is exactly what building a
    reply's In-Reply-To/References from the original's does. Only applied
    to those identifier headers, not general text like Subject/From, so
    this doesn't change anything about how the rest of the module decodes
    headers.
    """
    return " ".join(value.split())


def _decode_part(part: EmailMessage) -> str:
    payload = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return payload.decode("utf-8", errors="replace")


def _html_to_text(raw_html: str) -> str:
    stripped = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = _TAG_RE.sub(" ", stripped)
    text = htmllib.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _extract_text(msg: EmailMessage) -> str:
    plain: str | None = None
    html_body: str | None = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if (part.get("Content-Disposition") or "").lower().startswith("attachment"):
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain" and plain is None:
                plain = _decode_part(part)
            elif ctype == "text/html" and html_body is None:
                html_body = _decode_part(part)
    elif msg.get_content_type() == "text/html":
        html_body = _decode_part(msg)
    else:
        plain = _decode_part(msg)
    text = plain if plain is not None else _html_to_text(html_body or "")
    return text[:TEXT_CAP].strip()


def _attachment_parts(msg: EmailMessage) -> list[EmailMessage]:
    """The MIME parts `_extract_attachments`/`attachment_bytes` agree count as
    attachments, in a stable order -- shared so `Message.attachments[i]`
    always lines up with part `i` here, whether the caller is summarising
    (`_extract_attachments`) or fetching one part's bytes later
    (`Gmail.attachment_bytes`)."""
    if not msg.is_multipart():
        return []
    parts = []
    for part in msg.walk():
        filename = part.get_filename()
        disposition = (part.get("Content-Disposition") or "").lower()
        if not filename and "attachment" not in disposition:
            continue
        parts.append(part)
    return parts


def _extract_attachments(msg: EmailMessage) -> list[Attachment]:
    out = []
    for part in _attachment_parts(msg):
        payload = part.get_payload(decode=True) or b""
        out.append(
            Attachment(
                name=_decode_header_value(part.get_filename()) or "attachment",
                mime=part.get_content_type(),
                size=len(payload),
            )
        )
    return out


def _decode_labels(raw: Any) -> list[str]:
    """X-GM-LABELS tokens are IMAP quoted strings, still in modified UTF-7."""
    # imapclient's parser turns an all-digit atom into int and a bare NIL into None.
    return [
        "NIL" if label is None else str(label) if isinstance(label, int) else decode_utf7(label)
        for label in (raw or ())
    ]


def _decode_flags(raw: Any) -> list[str]:
    return [f.decode("ascii") if isinstance(f, bytes) else str(f) for f in (raw or ())]


def _aware_date(header_value: str | None) -> datetime | None:
    if not header_value:
        return None
    try:
        date = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if date.tzinfo is None:
        # e.g. a "-0000" (unknown) offset: stdlib returns this naive. Treat as UTC
        # rather than silently comparing naive/aware datetimes wrong later.
        date = date.replace(tzinfo=timezone.utc)
    return date


def _aware_internal_date(value: Any) -> datetime | None:
    """INTERNALDATE as an aware datetime.

    `Gmail` turns imapclient's `normalise_times` off at connect, so
    INTERNALDATE arrives aware, carrying the server's own offset (imapclient
    4.1 `parse_to_datetime(normalise=False)`). With normalising left on,
    imapclient converts to naive local time using the *current* UTC offset,
    and re-localising that with the message date's offset is an hour off
    across a DST change -- enough to skip or stall a cursor. A naive value
    here can only mean the server sent no zone at all; RFC 3501 requires
    one, so it is taken as UTC.
    """
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _web_url(thread_id: int | None) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{thread_id:x}" if thread_id else ""


def _address_list(values: list) -> list[str]:
    """Addresses from raw address-list header values.

    ASCII values are parsed raw, so an RFC 2047 encoded word stays inside a
    display name and can never smuggle in an extra address. Raw 8-bit
    (SMTPUTF8) values -- which compat32 returns as `Header` objects or
    non-ASCII str -- are decoded to text first, or the address comes back
    full of U+FFFD.
    """
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.isascii():
            value = _decode_header_value(value)
        out.extend(addr for _name, addr in getaddresses([value]) if addr)
    return out


def _parse_message(uid: int, item: dict, *, headers_only: bool) -> Message:
    msg_id = item.get(b"X-GM-MSGID")
    thread_id = item.get(b"X-GM-THRID")
    body_key = b"BODY[HEADER]" if headers_only else b"BODY[]"
    raw = item.get(body_key) or b""
    parsed = message_from_bytes(raw)

    from_name, from_addr = parseaddr(_decode_header_value(parsed.get("From")))
    to = _address_list(parsed.get_all("To", []) or [])
    # A Reply-To header can legally carry more than one address (RFC 5322
    # allows an address-list there, same as To/Cc).
    reply_to_all = tuple(_address_list(parsed.get_all("Reply-To", []) or []))
    reply_addr = reply_to_all[0] if reply_to_all else ""

    return Message(
        msg_id=str(msg_id) if msg_id is not None else "",
        thread_id=str(thread_id) if thread_id is not None else "",
        labels=_decode_labels(item.get(b"X-GM-LABELS")),
        flags=_decode_flags(item.get(b"FLAGS")),
        from_name=from_name,
        from_addr=from_addr,
        to=to,
        reply_to=reply_addr,
        subject=_decode_header_value(parsed.get("Subject")),
        date=_aware_date(parsed.get("Date")),
        list_unsubscribe=_decode_header_value(parsed.get("List-Unsubscribe")),
        list_id=_decode_header_value(parsed.get("List-Id")),
        auth_results=_decode_header_value(parsed.get("Authentication-Results")),
        text="" if headers_only else _extract_text(parsed),
        attachments=[] if headers_only else _extract_attachments(parsed),
        web_url=_web_url(thread_id),
        message_id=_unfold_header(_decode_header_value(parsed.get("Message-ID"))),
        references=_unfold_header(_decode_header_value(parsed.get("References"))),
        in_reply_to=_unfold_header(_decode_header_value(parsed.get("In-Reply-To"))),
        reply_to_all=reply_to_all,
        internal_date=_aware_internal_date(item.get(b"INTERNALDATE")),
    )


class Gmail:
    """Readonly IMAP session. Use as a context manager.

    Raises `GmailAuthError` immediately if `LMSA_IMAP_USER` or the password
    (env var, then OS keyring -- see `lmsa.secrets`) is missing, before ever
    touching the network.
    """

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings()
        self._client: imapclient.IMAPClient | None = None
        self._dropped_reason: str | None = None

    def __enter__(self) -> "Gmail":
        password = secrets.resolve("imap")
        if not self.cfg.imap_user or not password:
            raise GmailAuthError(
                "Gmail IMAP credentials missing: set LMSA_IMAP_USER and either "
                "LMSA_IMAP_PASSWORD or `python -m lmsa.secrets set imap`."
            )
        # A timeout keeps an unattended task from hanging on a dead connection.
        client = imapclient.IMAPClient(
            self.cfg.imap_host, port=self.cfg.imap_port, ssl=True, timeout=self.cfg.timeout
        )
        try:
            client.login(self.cfg.imap_user, password)
        except IMAPClientError:
            _shutdown(client)
            # Never let the password reach an exception message.
            raise GmailAuthError(f"Gmail IMAP login failed for {self.cfg.imap_user!r}") from None
        except BaseException:
            _shutdown(client)
            raise
        # Aware INTERNALDATE with the server's own offset -- see
        # `_aware_internal_date` for why normalising to local time is wrong.
        client.normalise_times = False
        self._client = client
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._client is not None:
            try:
                self._client.logout()
            except Exception:
                pass
            self._client = None

    def _require_client(self) -> imapclient.IMAPClient:
        if self._client is None:
            if self._dropped_reason:
                raise RuntimeError(f"Gmail session was closed: {self._dropped_reason}; open a new one")
            raise RuntimeError("Gmail must be used as a context manager: `with Gmail() as gmail:`")
        return self._client

    def _select(self, folder: str) -> None:
        self._require_client().select_folder(folder, readonly=True)

    def search(
        self,
        query: str,
        folder: str = DEFAULT_FOLDER,
        limit: int = 200,
        *,
        headers_only: bool = False,
    ) -> list[Message]:
        """Gmail search syntax (`newer_than:1d category:purchases`), newest first."""
        client = self._require_client()
        self._select(folder)
        uids = sorted(client.gmail_search(query), reverse=True)
        if limit:
            uids = uids[:limit]
        if not uids:
            return []
        return self._fetch_messages(uids, headers_only=headers_only)

    META_CHUNK = 500  # UIDs per metadata FETCH in `search_meta`

    def search_meta(self, query: str, folder: str = DEFAULT_FOLDER) -> list[MessageMeta]:
        """Every message matching `query` (no limit), as `MessageMeta` --
        X-GM-MSGID + INTERNALDATE only, fetched in chunks, never a body.
        Read-only. Order is unspecified; callers sort by `internal_date`.
        This is what lets a cursor pick the *oldest* unprocessed messages
        instead of trusting UID order (which imported/restored mail breaks).
        """
        client = self._require_client()
        self._select(folder)
        uids = sorted(client.gmail_search(query))
        out: list[MessageMeta] = []
        for i in range(0, len(uids), self.META_CHUNK):
            chunk = uids[i:i + self.META_CHUNK]
            fetched = client.fetch(chunk, ["X-GM-MSGID", "INTERNALDATE"])
            for uid in chunk:
                item = fetched.get(uid)
                if item is None or item.get(b"X-GM-MSGID") is None:
                    continue
                out.append(MessageMeta(
                    msg_id=str(item[b"X-GM-MSGID"]),
                    internal_date=_aware_internal_date(item.get(b"INTERNALDATE")),
                    uid=uid,
                ))
        return out

    def fetch_uids(
        self, uids: list[int], folder: str = DEFAULT_FOLDER, *, headers_only: bool = False
    ) -> list[Message]:
        """Full messages for `uids` from `search_meta` (same folder, same
        session), in the order given. Read-only (readonly select, PEEK)."""
        self._select(folder)
        if not uids:
            return []
        return self._fetch_messages(list(uids), headers_only=headers_only)

    def _fetch_messages(self, uids: list[int], *, headers_only: bool) -> list[Message]:
        """Fetch + parse `uids` (in the currently selected folder), in order."""
        body_item = "BODY.PEEK[HEADER]" if headers_only else "BODY.PEEK[]"
        fetched = self._require_client().fetch(
            uids, ["X-GM-MSGID", "X-GM-THRID", "X-GM-LABELS", "FLAGS", "INTERNALDATE", body_item]
        )
        return [
            _parse_message(uid, fetched[uid], headers_only=headers_only)
            for uid in uids
            if uid in fetched
        ]

    def fetch_by_msgids(
        self,
        msg_ids: list[str],
        folder: str = DEFAULT_FOLDER,
        *,
        headers_only: bool = False,
    ) -> list[Message]:
        """Refetch specific messages by their stable `X-GM-MSGID`
        (`Message.msg_id`), in the order given. Read-only (readonly select,
        PEEK). A msg_id that no longer exists in `folder` (deleted, or not a
        valid X-GM-MSGID) is simply absent from the result -- callers compare
        against what they asked for. Used to retry items a task failed on.
        """
        client = self._require_client()
        self._select(folder)
        uids: list[int] = []
        for msg_id in msg_ids:
            try:
                gm_msgid = int(msg_id)
            except (TypeError, ValueError):
                continue
            found = client.search(["X-GM-MSGID", gm_msgid])
            if found:
                uids.append(found[0])
        if not uids:
            return []
        return self._fetch_messages(uids, headers_only=headers_only)

    def count(self, query: str, folder: str = DEFAULT_FOLDER) -> int:
        """Count of matching messages via SEARCH alone -- no fetch at all."""
        client = self._require_client()
        self._select(folder)
        return len(client.gmail_search(query))

    def attachment_bytes(
        self,
        msg: Message,
        index: int,
        *,
        folder: str = DEFAULT_FOLDER,
        max_size: int = ATTACHMENT_MAX_BYTES,
    ) -> bytes:
        """Decoded bytes of `msg.attachments[index]`. Still read-only (PEEK).

        `Message` doesn't carry the raw MIME (only the summary `Attachment`
        list from `search()`), so this refetches the message by its
        `X-GM-MSGID` -- Gmail's own stable id, not the mutable UID -- and
        walks it with the same `_attachment_parts` ordering `search()` used
        to build `msg.attachments`, so `index` lines up. Size-capped against
        both the declared size (`msg.attachments[index].size`, cheap, before
        any network fetch) and the actual decoded payload (a lying
        Content-Length can't be trusted): either over `max_size` raises
        `AttachmentTooLargeError` rather than fetching or returning it.
        Raises `AttachmentNotFoundError` if the message or that many parts
        no longer exist (e.g. it was deleted between `search()` and here).
        """
        if index < 0 or index >= len(msg.attachments):
            raise IndexError(
                f"attachment index {index} out of range for {len(msg.attachments)} attachment(s)"
            )
        declared = msg.attachments[index]
        if declared.size > max_size:
            raise AttachmentTooLargeError(
                f"{declared.name!r} is {declared.size} bytes, over the {max_size} byte cap"
            )
        if not msg.msg_id:
            raise AttachmentNotFoundError("message has no X-GM-MSGID to refetch by")

        client = self._require_client()
        self._select(folder)
        uids = client.search(["X-GM-MSGID", msg.msg_id])
        if not uids:
            raise AttachmentNotFoundError(f"message {msg.msg_id} not found in {folder!r}")
        uid = uids[0]
        fetched = client.fetch([uid], ["BODY.PEEK[]"])
        raw = fetched.get(uid, {}).get(b"BODY[]")
        if not raw:
            raise AttachmentNotFoundError(f"message {msg.msg_id} fetch returned no body")

        parts = _attachment_parts(message_from_bytes(raw))
        if index >= len(parts):
            raise AttachmentNotFoundError(
                f"message {msg.msg_id} now has {len(parts)} attachment part(s), "
                f"expected at least {index + 1}"
            )
        payload = parts[index].get_payload(decode=True) or b""
        if len(payload) > max_size:
            raise AttachmentTooLargeError(
                f"{declared.name!r} is {len(payload)} bytes, over the {max_size} byte cap"
            )
        return payload

    def folders(self) -> list[str]:
        return [name for _flags, _delim, name in self._require_client().list_folders()]

    def sent(self) -> str:
        """The Sent folder, resolved by its `\\Sent` special-use flag."""
        folder = self._require_client().find_special_folder(SENT_FLAG)
        if not folder:
            raise RuntimeError(r"no folder advertises the \Sent special-use flag")
        return folder

    def append_draft(self, message_bytes: bytes, folder: str | None = None) -> None:
        """Append a message to the Drafts folder with the `\\Draft` flag set.

        One of this module's two write paths (see the module docstring) --
        deliberately narrow:
        resolves Drafts by its `\\Drafts` special-use flag (never selects it,
        so this can't be mistaken for a write to whatever's currently
        selected) and issues exactly one IMAP APPEND. No STORE, EXPUNGE,
        COPY, or MOVE, and no SMTP -- this never sends anything.
        """
        client = self._require_client()
        client.append(folder or self.drafts_folder(), message_bytes, flags=[imapclient.DRAFT])

    def drafts_folder(self) -> str:
        """The Drafts folder, by its `\\Drafts` special-use flag (a LIST, no write)."""
        folder = self._require_client().find_special_folder(DRAFTS_FLAG)
        if not folder:
            raise RuntimeError(r"no folder advertises the \Drafts special-use flag")
        return folder

    def _drop_client(self, reason: str) -> None:
        """Abandon the session so nothing else ever runs on it (e.g. after a
        failed readonly re-select left it possibly writable-selected).
        Later calls raise a clear RuntimeError instead."""
        client, self._client = self._client, None
        self._dropped_reason = reason
        if client is not None:
            _shutdown(client)

    def _restore_readonly(self, folder: str) -> None:
        """Re-select `folder` readonly after a write. Never raises: a failure
        here must not mask the write's own outcome (or error), so it is
        logged and the session is dropped instead of left writable."""
        try:
            self._select(folder)
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("readonly re-select of %r failed; dropping IMAP session: %s", folder, exc)
            self._drop_client(f"readonly re-select after a label write failed ({type(exc).__name__}: {exc})")

    def _apply_labels(
        self, msg_id: str, labels: list[str], folder: str, *, remove: bool
    ) -> None:
        """Shared body of `add_labels`/`remove_labels` -- see their docstrings.

        Never called with an empty `labels` list (both public methods guard
        that themselves), so this never issues a no-op STORE.
        """
        # Validate everything before touching the network.
        reserved = [label for label in labels if is_reserved_label(label)]
        if reserved:
            raise ValueError(f"refusing to write Gmail system label(s): {reserved!r}")
        try:
            gm_msgid = int(msg_id)
        except (TypeError, ValueError):
            raise ValueError(f"not a valid X-GM-MSGID: {msg_id!r}") from None

        client = self._require_client()
        try:
            # Non-readonly select, scoped to this one call: the only place in
            # this class that ever selects a folder writable.
            client.select_folder(folder, readonly=False)
            uids = list(client.search(["X-GM-MSGID", gm_msgid]))
            if not uids:
                raise LookupError(f"no message with X-GM-MSGID {msg_id} in {folder!r}")
            try:
                if remove:
                    client.remove_gmail_labels(uids, labels, silent=True)
                else:
                    client.add_gmail_labels(uids, labels, silent=True)
            except Exception as exc:
                # A tagged NO/BAD reply means the server refused the STORE.
                # Anything else (socket error, abort) may have dropped the
                # connection after the server acted, so the outcome is unknown.
                refused = isinstance(exc, IMAPClientError) and not isinstance(exc, IMAPClientAbortError)
                raise LabelWriteError(
                    f"STORE {'-' if remove else '+'}X-GM-LABELS failed: {type(exc).__name__}: {exc}",
                    maybe_applied=not refused,
                ) from exc
            # STORE returned: this is the success point. Nothing after it can
            # raise (the re-select below never does).
        finally:
            # Always restore readonly, even if resolution/STORE raised, so a
            # caller reusing this session never inherits a writable mailbox.
            self._restore_readonly(folder)

    def add_labels(self, msg_id: str, labels: list[str], folder: str = DEFAULT_FOLDER) -> None:
        """Add Gmail `labels` to the single message identified by `msg_id`
        (its stable `X-GM-MSGID`, as returned on `Message.msg_id` -- never a
        UID, which can change across sessions).

        A label that doesn't already exist in the account is created by
        Gmail itself the first time it's applied; this method does not
        create labels ahead of time. Adding a label the message already has
        is a no-op (Gmail's `STORE +X-GM-LABELS` is idempotent).
        """
        if not labels:
            return
        self._apply_labels(msg_id, list(labels), folder, remove=False)

    def remove_labels(self, msg_id: str, labels: list[str], folder: str = DEFAULT_FOLDER) -> None:
        """Remove Gmail `labels` from the single message identified by
        `msg_id` (its stable `X-GM-MSGID`). Removing a label the message
        doesn't have is a no-op (Gmail's `STORE -X-GM-LABELS` is idempotent),
        which is what makes an undo safe to run even if a label was already
        removed by hand in the meantime.
        """
        if not labels:
            return
        self._apply_labels(msg_id, list(labels), folder, remove=True)

    # -- inbox actions (lmstudio-automation#61) --------------------------------
    #
    # Four fixed writes, each hard-coded to exactly one thing: removing or
    # re-adding the \Inbox label (archive / unarchive), and setting or
    # clearing the \Seen flag (mark_read / mark_unread). None takes a label or
    # flag argument, so no caller can reach any other system label through
    # them. Each runs per message (one failure never touches the others) with
    # the same writable-select-then-restore-readonly scoping as
    # `_apply_labels`, and first FETCHes the message's current X-GM-LABELS /
    # FLAGS: when the write would change nothing (already archived, already
    # read, a sent message that was never in the inbox) no STORE is sent and
    # the result is False, so the caller logs nothing an undo could reverse.
    # `lmsa.handled` logs every real write per run and undoes a run by its id.

    INBOX_LABEL = "\\Inbox"

    # op -> (what to inspect, whether the op wants it present afterwards)
    _INBOX_OPS = {
        "archive": ("inbox", False),
        "unarchive": ("inbox", True),
        "mark_read": ("seen", True),
        "mark_unread": ("seen", False),
    }

    @staticmethod
    def _current_state(item: dict, what: str) -> bool:
        if what == "inbox":
            labels = _decode_labels(item.get(b"X-GM-LABELS"))
            return any(label.lower() == "\\inbox" for label in labels)
        flags = _decode_flags(item.get(b"FLAGS"))
        return any(flag.lower() == "\\seen" for flag in flags)

    def _write_one(self, msg_id: str, folder: str, op: str) -> bool:
        """Apply `op` to one message. True: the STORE was sent (the state
        changed). False: already in the target state, nothing written.

        The read (FETCH) and the write (STORE) are two IMAP commands, not one
        atomic step: if the state changes elsewhere in between (e.g. you
        archive the message in the Gmail app in that instant), the STORE is
        still sent and reported as a change, so a later `--undo` would
        reverse your own action too. The window is one round trip per
        message; IMAP offers no compare-and-set to close it."""
        what, want = self._INBOX_OPS[op]
        try:
            gm_msgid = int(msg_id)
        except (TypeError, ValueError):
            raise ValueError(f"not a valid X-GM-MSGID: {msg_id!r}") from None
        client = self._require_client()
        try:
            client.select_folder(folder, readonly=False)
            uids = list(client.search(["X-GM-MSGID", gm_msgid]))
            if not uids:
                raise LookupError(f"no message with X-GM-MSGID {msg_id} in {folder!r}")
            fetched = client.fetch(uids, ["X-GM-LABELS", "FLAGS"])
            items = [fetched.get(uid) for uid in uids]
            if any(item is None for item in items):
                raise LookupError(f"could not read the current state of X-GM-MSGID {msg_id}")
            todo = [uid for uid, item in zip(uids, items) if self._current_state(item, what) != want]
            if not todo:
                return False
            try:
                if op == "archive":
                    client.remove_gmail_labels(todo, [self.INBOX_LABEL], silent=True)
                elif op == "unarchive":
                    client.add_gmail_labels(todo, [self.INBOX_LABEL], silent=True)
                elif op == "mark_read":
                    client.add_flags(todo, [imapclient.SEEN], silent=True)
                else:  # mark_unread
                    client.remove_flags(todo, [imapclient.SEEN], silent=True)
            except Exception as exc:
                refused = isinstance(exc, IMAPClientError) and not isinstance(exc, IMAPClientAbortError)
                raise LabelWriteError(
                    f"{op} STORE failed: {type(exc).__name__}: {exc}", maybe_applied=not refused
                ) from exc
            return True
        finally:
            self._restore_readonly(folder)

    def _write_many(self, msg_ids: list[str], folder: str, op: str) -> dict[str, bool | Exception]:
        """Run `op` per message: {msg_id: True (changed) | False (already
        so, nothing written) | the exception}."""
        results: dict[str, bool | Exception] = {}
        for msg_id in msg_ids:
            try:
                results[msg_id] = self._write_one(msg_id, folder, op)
            except Exception as exc:  # noqa: BLE001 - isolate per message
                results[msg_id] = exc
        return results

    def archive(self, msg_ids: list[str], folder: str = DEFAULT_FOLDER) -> dict[str, bool | Exception]:
        """Archive: remove the `\\Inbox` label (Gmail's STORE -X-GM-LABELS)
        from each message that has it, by stable X-GM-MSGID. Nothing is
        moved, trashed or expunged; the message stays in All Mail. Undo with
        `unarchive`. See `_write_many` for the result."""
        return self._write_many(list(msg_ids), folder, "archive")

    def unarchive(self, msg_ids: list[str], folder: str = DEFAULT_FOLDER) -> dict[str, bool | Exception]:
        """Undo of `archive`: add `\\Inbox` back (where it isn't already)."""
        return self._write_many(list(msg_ids), folder, "unarchive")

    def mark_read(self, msg_ids: list[str], folder: str = DEFAULT_FOLDER) -> dict[str, bool | Exception]:
        """Set the `\\Seen` flag where it isn't set. Undo with `mark_unread`."""
        return self._write_many(list(msg_ids), folder, "mark_read")

    def mark_unread(self, msg_ids: list[str], folder: str = DEFAULT_FOLDER) -> dict[str, bool | Exception]:
        """Undo of `mark_read`: clear the `\\Seen` flag where it is set."""
        return self._write_many(list(msg_ids), folder, "mark_unread")
