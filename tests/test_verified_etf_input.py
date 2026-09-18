"""Reject stale, mismatched or incomplete snapshots before any strategy runs."""

import hashlib
import json

import duckdb
import pandas as pd
import pytest

from qkquant.data import verified
from qkquant.data.storage import DuckStore
from qkquant.data.verified import VerifiedInputError, open_verified_etf_input


CODES = """159773 159840 159841 159842 159843 159845 159847 159848 159849
159851 159852 159855 159856 159857 159858 159859 159861 159862 159863 159864
159865 159867 159869 159870 159872 159873 159875 159877 159883 159885 159886
159887 159888 159889 159890 159891 159895 159896 159898 159899 159901 159902
159903 159905 159906 159907 159908 159909 159910 159912 159913 159915 159916
159918 159919 159922 159923 159925 159928 159929 159930 159931 159933 159935
159936 159938 159939 159940 159943 159944 159945 159948 159949 159952 159956
159957 159958 159959 159961 159964 159965 159966 159967 159968 159970 159971
159973 159974 159975 159976 159977 159981 159982 159991 159992 159993 159994
159995 159996 159997 159998""".split()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(root, **values):
    path = root / "snapshot_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(values)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def mutate_database(root, sql):
    path = root / "verified_etf_snapshot.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute(sql)
        rows = con.execute("SELECT count(*) FROM daily_bars WHERE adjust='qfq'").fetchone()[0]
        minimum = con.execute("SELECT min(n) FROM (SELECT count(*) n FROM daily_bars WHERE adjust='qfq' GROUP BY trade_date)").fetchone()[0]
    update_manifest(root, database_sha256=digest(path), daily_rows=rows, minimum_daily_coverage=minimum)


@pytest.fixture
def snapshot(tmp_path):
    dates = pd.date_range("2024-01-02", periods=3)
    accepted = [code for code in CODES if code != "159925"]
    frames = []
    for adjust, price in (("", 10.0), ("qfq", 9.0)):
        frames.append(pd.DataFrame([
            dict(code=code, trade_date=day, open=price, high=price+.1, low=price-.1,
                 close=price, volume=1000, amount=10000, pct_chg=0, turnover=0, adjust=adjust)
            for code in accepted for day in dates
        ]))
    database = tmp_path / "verified_etf_snapshot.duckdb"
    with DuckStore(database) as store:
        store.upsert_daily(pd.concat(frames))
        store.upsert_calendar([day.date() for day in dates])
    freeze = tmp_path / "freeze_manifest.json"
    freeze.write_text(json.dumps(dict(universe=dict(codes=CODES, count=101),
                                     latest_completed_date_cap="2024-01-04")), encoding="utf-8")
    calendar = tmp_path / "trade_calendar.csv"
    pd.DataFrame({"trade_date": dates}).to_csv(calendar, index=False)
    manifest = dict(status="ready_for_frozen_research", price_adjustment_verified=True,
                    synthetic_gap_prices_created=False, duplicate_or_single_source_missing_bars_allowed=False,
                    availability_protocol="v2_common_source_gaps_explicitly_unavailable",
                    database="Z:/not-the-local-copy/old.duckdb", database_sha256=digest(database),
                    frozen_manifest="Z:/absent/freeze.json", frozen_manifest_sha256=digest(freeze),
                    calendar_source_file="Z:/absent/calendar.csv", calendar_source_sha256=digest(calendar),
                    codes_requested=CODES, codes_accepted=accepted, accepted_count=100, rejected_count=1,
                    start="2024-01-01", end="2024-01-04", daily_rows=300, minimum_daily_coverage=100)
    (tmp_path / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_portable_readonly_snapshot_keeps_fixed_universe_and_native_prices(snapshot):
    with open_verified_etf_input(snapshot) as data:
        assert data.panel["close"].shape == (3, 101)
        assert data.panel["close"]["159925"].isna().all()
        assert data.panel["close"]["159773"].iloc[0] == 9
        assert data.raw_panel["close"]["159773"].iloc[0] == 10
        assert data.metadata["excluded_codes"] == ["159925"]
        assert data.coverage["coverage_fraction"].iloc[0] == 100 / 101
        assert data.store._read_only
        with pytest.raises(duckdb.InvalidInputException):
            data.store.conn.execute("DELETE FROM daily_bars")
    assert data.store._conn is None


def test_default_missing_snapshot_does_not_fall_back_to_legacy_db(monkeypatch, tmp_path):
    monkeypatch.setattr(verified, "DEFAULT_SNAPSHOT_DIR", tmp_path / "missing")
    with pytest.raises(VerifiedInputError, match="Cannot verify ETF snapshot"):
        with open_verified_etf_input():
            pytest.fail("Missing snapshot must not reach the strategy")


@pytest.mark.parametrize("filename", ["verified_etf_snapshot.duckdb", "freeze_manifest.json", "trade_calendar.csv"])
def test_modified_companion_file_is_rejected(snapshot, filename):
    path = snapshot / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(VerifiedInputError, match="SHA256 mismatch"):
        with open_verified_etf_input(snapshot):
            pass


def test_unverified_snapshot_is_rejected(snapshot):
    update_manifest(snapshot, price_adjustment_verified=False)
    with pytest.raises(VerifiedInputError, match="audit"):
        with open_verified_etf_input(snapshot):
            pass


def test_different_universe_cannot_hide_behind_new_manifest_hash(snapshot):
    path = snapshot / "freeze_manifest.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    freeze["universe"]["codes"][0] = "510300"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    update_manifest(snapshot, frozen_manifest_sha256=digest(path), codes_requested=freeze["universe"]["codes"])
    with pytest.raises(VerifiedInputError, match="fixed 101-code"):
        with open_verified_etf_input(snapshot):
            pass


def test_quarantined_code_cannot_be_reintroduced(snapshot):
    accepted = [code for code in CODES if code != "159773"]
    update_manifest(snapshot, codes_accepted=accepted)
    with pytest.raises(VerifiedInputError, match="quarantined 159925"):
        with open_verified_etf_input(snapshot):
            pass


def test_low_coverage_uses_original_101_as_denominator(snapshot):
    selected = ",".join(f"'{code}'" for code in CODES[:5])
    mutate_database(snapshot, f"DELETE FROM daily_bars WHERE code IN ({selected}) AND trade_date='2024-01-03'")
    with pytest.raises(VerifiedInputError, match="below 95%.*2024-01-03"):
        with open_verified_etf_input(snapshot):
            pass


def test_joint_source_gap_remains_nan_without_shortening_calendar(snapshot):
    mutate_database(snapshot, "DELETE FROM daily_bars WHERE code='159773' AND trade_date='2024-01-03'")
    with open_verified_etf_input(snapshot) as data:
        assert len(data.panel["close"]) == 3
        assert pd.isna(data.panel["close"].loc["2024-01-03", "159773"])
        assert data.coverage["codes"].tolist() == [100, 99, 100]


def test_entire_missing_session_is_not_removed_from_time_axis(snapshot):
    mutate_database(snapshot, "DELETE FROM daily_bars WHERE trade_date='2024-01-03'")
    with pytest.raises(VerifiedInputError, match="below 95%.*2024-01-03"):
        with open_verified_etf_input(snapshot):
            pass


def test_database_calendar_cannot_omit_session(snapshot):
    mutate_database(snapshot, "DELETE FROM trade_calendar WHERE trade_date='2024-01-03'")
    with pytest.raises(VerifiedInputError, match="calendar differs"):
        with open_verified_etf_input(snapshot):
            pass


@pytest.mark.parametrize("sql, message", [
    ("UPDATE daily_bars SET adjust='hfq' WHERE adjust='qfq'", "native.*qfq row counts"),
    ("DELETE FROM daily_bars WHERE adjust='' AND code='159773' AND trade_date='2024-01-03'", "native.*qfq row counts"),
    ("UPDATE daily_bars SET amount=amount+1 WHERE adjust=''", "native volume/amount"),
    ("UPDATE daily_bars SET open=0 WHERE adjust='qfq'", "Invalid or inconsistent open"),
    ("UPDATE daily_bars SET high=1 WHERE adjust='qfq'", "OHLC ordering"),
])
def test_matching_hash_does_not_replace_content_validation(snapshot, sql, message):
    mutate_database(snapshot, sql)
    with pytest.raises(VerifiedInputError, match=message):
        with open_verified_etf_input(snapshot):
            pass
