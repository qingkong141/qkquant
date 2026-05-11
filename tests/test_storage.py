"""DuckStore 基本读写测试。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from qkquant.data.storage import DuckStore


def test_schema_created(tmp_path):
    store = DuckStore(tmp_path / "x.duckdb")
    s = store.stats()
    assert s["bars"] == 0
    assert s["instruments"] == 0
    store.close()


def test_upsert_and_load_daily(tmp_store: DuckStore):
    s = tmp_store.stats()
    assert s["codes"] == 5
    assert s["bars"] >= 5 * 200

    df = tmp_store.load_daily(codes=["000001"], start="2023-01-01", end="2023-12-31")
    assert not df.empty
    assert df["code"].nunique() == 1
    assert set(df.columns) >= {"code", "trade_date", "open", "high", "low", "close", "volume"}


def test_incremental_max_date(tmp_store: DuckStore):
    mx = tmp_store.get_max_trade_date("000001", "hfq")
    assert mx is not None
    assert isinstance(mx, date)


def test_index_constituents_roundtrip(tmp_store: DuckStore):
    codes = tmp_store.load_index_constituents("000300")
    assert len(codes) == 5
    assert "000001" in codes


def test_instruments_st_filter(tmp_store: DuckStore):
    df = tmp_store.load_instruments()
    assert len(df) == 5
    assert df["is_st"].sum() == 0


def test_upsert_idempotent(tmp_store: DuckStore):
    before = tmp_store.stats()["bars"]
    existing = tmp_store.load_daily()
    tmp_store.upsert_daily(existing)
    after = tmp_store.stats()["bars"]
    assert before == after, "re-upsert should not duplicate rows"
