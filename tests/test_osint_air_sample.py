import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lmsa.store import Store
from lmsa.web import FetchError, InvalidJSON
from tasks import osint_air_sample as sampler

FIXTURE = Path(__file__).parent / "fixtures" / "osint" / "adsb_mil.json"
PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))
NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


class FakeFetcher:
    """Stands in for lmsa.web.Fetcher: no live server, no network."""

    def __init__(self, json_result=None, raises: Exception | None = None):
        self.json_result = json_result
        self.raises = raises
        self.calls: list[str] = []

    def get_json(self, url: str):
        self.calls.append(url)
        if self.raises is not None:
            raise self.raises
        return self.json_result


# -- bucketing / sampling ------------------------------------------------------


def test_sample_from_aircraft_buckets_and_normalizes():
    s = sampler.sample_from_aircraft(
        {"hex": "AE1460", "t": "c17", "flight": " REACH123 ", "lat": 51.48, "lon": -0.45, "alt_baro": 33000}, NOW,
    )
    assert s.hex == "ae1460"
    assert s.actype == "C17"
    assert s.callsign == "REACH123"
    assert s.lat_bucket == 51
    assert s.lon_bucket == -1  # floor(-0.45) == -1, not round-toward-zero
    assert s.alt_baro == 33000


def test_sample_from_aircraft_ground_altitude_is_zero():
    s = sampler.sample_from_aircraft({"hex": "abc", "lat": 1, "lon": 1, "alt_baro": "ground"}, NOW)
    assert s.alt_baro == 0


def test_sample_from_aircraft_skips_missing_hex_or_position():
    assert sampler.sample_from_aircraft({"hex": "", "lat": 1, "lon": 1}, NOW) is None
    assert sampler.sample_from_aircraft({"hex": "abc", "lat": "n/a", "lon": 1}, NOW) is None
    assert sampler.sample_from_aircraft({"hex": "abc", "lat": 1, "lon": None}, NOW) is None


def test_sample_from_aircraft_no_type_or_callsign_is_none_not_empty_string():
    s = sampler.sample_from_aircraft({"hex": "9a0001", "t": "", "flight": "", "lat": 1, "lon": 1}, NOW)
    assert s.actype is None
    assert s.callsign is None


def test_sample_from_aircraft_never_raises_on_garbage():
    assert sampler.sample_from_aircraft({"hex": ["not", "a", "string"]}, NOW) is None
    assert sampler.sample_from_aircraft({}, NOW) is None


# -- run_sample: fetch + store + prune ------------------------------------------


def test_run_sample_stores_valid_and_skips_bad_records(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        summary = sampler.run_sample(store.conn, NOW, fetcher=FakeFetcher(json_result=PAYLOAD))
        assert summary["error"] is None
        assert summary["aircraft_seen"] == 6
        assert summary["stored"] == 4  # unknown-pos (bad lat) and "" hex are skipped
        assert summary["skipped"] == 2

        rows = store.conn.execute(
            "SELECT hex, actype, callsign, lat_bucket, lon_bucket, alt_baro FROM osint_air_samples ORDER BY hex"
        ).fetchall()
        assert ("43c123", "KC135", "TANKR01", 51, 1, 28000) in rows
        assert ("7c1234", "F16", "VIPER22", 35, 129, 0) in rows
        assert ("9a0001", None, None, 34, -119, 5000) in rows
        assert ("ae1460", "C17", "REACH123", 51, -1, 33000) in rows


def test_run_sample_uses_the_adsb_mil_url(tmp_path):
    fetcher = FakeFetcher(json_result=PAYLOAD)
    with Store(tmp_path / "lmsa.sqlite") as store:
        sampler.run_sample(store.conn, NOW, fetcher=fetcher)
    assert fetcher.calls == [sampler.ADSB_MIL_URL]


def test_run_sample_isolates_fetch_error(tmp_path):
    fetcher = FakeFetcher(raises=FetchError("network down"))
    with Store(tmp_path / "lmsa.sqlite") as store:
        summary = sampler.run_sample(store.conn, NOW, fetcher=fetcher)
        assert summary["stored"] == 0
        assert summary["aircraft_seen"] == 0
        assert "network down" in summary["error"]


def test_run_sample_isolates_invalid_json(tmp_path):
    fetcher = FakeFetcher(raises=InvalidJSON("not json"))
    with Store(tmp_path / "lmsa.sqlite") as store:
        summary = sampler.run_sample(store.conn, NOW, fetcher=fetcher)
        assert summary["stored"] == 0
        assert summary["error"] is not None


def test_run_sample_isolates_an_unexpected_payload_shape(tmp_path):
    """`get_json` succeeded but returned something that isn't the expected
    {"ac": [...]} object (e.g. a compromised or changed endpoint) -- must
    not raise, just see zero aircraft."""
    fetcher = FakeFetcher(json_result=["not", "a", "dict"])
    with Store(tmp_path / "lmsa.sqlite") as store:
        summary = sampler.run_sample(store.conn, NOW, fetcher=fetcher)
        assert summary["error"] is None
        assert summary["aircraft_seen"] == 0
        assert summary["stored"] == 0


def test_run_sample_is_idempotent_safe_called_twice_back_to_back(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        sampler.run_sample(store.conn, NOW, fetcher=FakeFetcher(json_result=PAYLOAD))
        sampler.run_sample(store.conn, NOW, fetcher=FakeFetcher(json_result=PAYLOAD))
        count = store.conn.execute("SELECT COUNT(*) FROM osint_air_samples").fetchone()[0]
        assert count == 8  # each call is its own timestamped sample, no crash, no data loss


# -- retention ------------------------------------------------------------------


def test_prune_old_samples_removes_rows_past_retention(tmp_path):
    with Store(tmp_path / "lmsa.sqlite") as store:
        old = (NOW - timedelta(days=sampler.RETENTION_DAYS + 1)).isoformat()
        recent = (NOW - timedelta(days=1)).isoformat()
        store.conn.execute(
            "INSERT INTO osint_air_samples (sampled_at, hex, actype, callsign, lat_bucket, lon_bucket, alt_baro) "
            "VALUES (?, 'old01', 'C17', 'X', 1, 1, 100)",
            (old,),
        )
        store.conn.execute(
            "INSERT INTO osint_air_samples (sampled_at, hex, actype, callsign, lat_bucket, lon_bucket, alt_baro) "
            "VALUES (?, 'new01', 'C17', 'X', 1, 1, 100)",
            (recent,),
        )
        store.conn.commit()
        pruned = sampler.prune_old_samples(store.conn, NOW)
        assert pruned == 1
        remaining = [r[0] for r in store.conn.execute("SELECT hex FROM osint_air_samples").fetchall()]
        assert remaining == ["new01"]


# -- main() end to end ----------------------------------------------------------


def test_main_writes_run_row_and_never_blocks_on_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(sampler, "_now", lambda: NOW)
    monkeypatch.setattr(sampler, "default_fetcher", lambda: FakeFetcher(json_result=PAYLOAD))
    monkeypatch.setenv("LMSA_DB", str(tmp_path / "lmsa.sqlite"))

    sampler.main()

    with Store(tmp_path / "lmsa.sqlite") as store:
        row = store.conn.execute("SELECT task, status, summary_json FROM runs").fetchone()
        assert row[0] == "tasks.osint_air_sample"
        assert row[1] == "ok"
        summary = json.loads(row[2])
        assert summary["stored"] == 4
