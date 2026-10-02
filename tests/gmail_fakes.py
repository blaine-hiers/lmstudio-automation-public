"""Test helper: give a fake Gmail with a `search(query, limit=...)` the
metadata API the shared cursor (lmsa.gmail_cursor) uses.

`search_meta` runs the fake's own `search` with no limit and reports each
match's msg_id + INTERNALDATE (falling back to the Date header, as a test
fixture without an internal_date means "delivered when dated");
`fetch_uids` hands the matching messages back by those uids, and
`fetch_by_msgids` (the retry path) looks messages up by msg_id.
"""

from __future__ import annotations

from lmsa.sources.gmail import MessageMeta


class MetaSearchMixin:
    def search_meta(self, query, folder="[Gmail]/All Mail"):
        messages = self.search(query, limit=10**9)
        self._meta_by_uid = {uid: m for uid, m in enumerate(messages, start=1)}
        return [
            MessageMeta(msg_id=m.msg_id, internal_date=m.internal_date or m.date, uid=uid)
            for uid, m in self._meta_by_uid.items()
        ]

    def fetch_uids(self, uids, folder="[Gmail]/All Mail", *, headers_only=False):
        return [self._meta_by_uid[u] for u in uids if u in self._meta_by_uid]

    def fetch_by_msgids(self, msg_ids, folder="[Gmail]/All Mail", *, headers_only=False):
        everything = {m.msg_id: m for m in self.search("after:0", limit=10**9)}
        return [everything[i] for i in msg_ids if i in everything]
