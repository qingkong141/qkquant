"""fetcher 中纯函数/辅助工具的单测（不走网络）。"""

from __future__ import annotations

import pandas as pd
import pytest

from qkquant.data.fetcher import (
    DataFetcher,
    bs_adjust,
    from_bs_code,
    to_bs_code,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("000001", "sz.000001"),
        ("300750", "sz.300750"),
        ("600000", "sh.600000"),
        ("601318", "sh.601318"),
        ("688001", "sh.688001"),
        ("900001", "sh.900001"),
        ("430047", "bj.430047"),
        ("830799", "bj.830799"),
        (1, "sz.000001"),
    ],
)
def test_to_bs_code(raw, expected):
    assert to_bs_code(raw) == expected


def test_to_bs_code_rejects_invalid():
    with pytest.raises(ValueError):
        to_bs_code("123456")


@pytest.mark.parametrize(
    "bs,expected",
    [
        ("sh.600000", "600000"),
        ("sz.000001", "000001"),
        ("bj.430047", "430047"),
        ("000001", "000001"),  # 已经是纯代码
    ],
)
def test_from_bs_code(bs, expected):
    assert from_bs_code(bs) == expected


@pytest.mark.parametrize(
    "adjust,expected",
    [
        ("hfq", "1"),
        ("qfq", "2"),
        ("", "3"),
        ("none", "3"),
        ("HFQ", "1"),  # 大小写不敏感
    ],
)
def test_bs_adjust(adjust, expected):
    assert bs_adjust(adjust) == expected


def test_bs_adjust_rejects_unknown():
    with pytest.raises(ValueError):
        bs_adjust("foobar")


def test_auto_fallback_uses_baostock_when_akshare_raises(monkeypatch):
    """source=auto：akshare 抛异常时应回退到 baostock。"""
    fetcher = DataFetcher(source="auto")
    call_log: list[str] = []

    def fake_ak(code, start, end, adjust):
        call_log.append("ak")
        raise RuntimeError("simulated akshare proxy failure")

    def fake_bs(code, start, end, adjust):
        call_log.append("bs")
        return pd.DataFrame(
            {
                "code": [code],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
                "pct_chg": [2.0],
                "turnover": [0.5],
                "adjust": ["hfq"],
            }
        )

    monkeypatch.setattr(fetcher, "_fetch_daily_ak", fake_ak)
    monkeypatch.setattr(fetcher, "_fetch_daily_bs", fake_bs)

    df = fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31", adjust="hfq")
    assert call_log == ["ak", "bs"]
    assert not df.empty
    assert df["code"].iloc[0] == "000001"


def test_auto_fallback_uses_baostock_when_akshare_empty(monkeypatch):
    fetcher = DataFetcher(source="auto")
    call_log: list[str] = []

    def fake_ak(code, start, end, adjust):
        call_log.append("ak")
        return pd.DataFrame()

    def fake_bs(code, start, end, adjust):
        call_log.append("bs")
        return pd.DataFrame(
            {
                "code": [code],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
                "pct_chg": [2.0],
                "turnover": [0.5],
                "adjust": ["hfq"],
            }
        )

    monkeypatch.setattr(fetcher, "_fetch_daily_ak", fake_ak)
    monkeypatch.setattr(fetcher, "_fetch_daily_bs", fake_bs)

    df = fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31", adjust="hfq")
    assert call_log == ["ak", "bs"]
    assert len(df) == 1


def test_akshare_only_does_not_fallback(monkeypatch):
    """source=akshare：akshare 失败应原样抛出，不回退。"""
    fetcher = DataFetcher(source="akshare")

    def fake_ak(code, start, end, adjust):
        raise RuntimeError("simulated akshare failure")

    def fake_bs(code, start, end, adjust):
        raise AssertionError("baostock should NOT be called under source=akshare")

    monkeypatch.setattr(fetcher, "_fetch_daily_ak", fake_ak)
    monkeypatch.setattr(fetcher, "_fetch_daily_bs", fake_bs)

    with pytest.raises(RuntimeError, match="simulated akshare failure"):
        fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31", adjust="hfq")


def test_baostock_only_skips_akshare(monkeypatch):
    fetcher = DataFetcher(source="baostock")
    calls: list[str] = []

    def fake_ak(*args, **kwargs):
        calls.append("ak")
        raise AssertionError("akshare should not be called under source=baostock")

    def fake_bs(code, start, end, adjust):
        calls.append("bs")
        return pd.DataFrame({"code": [code], "close": [1.0]})

    monkeypatch.setattr(fetcher, "_fetch_daily_ak", fake_ak)
    monkeypatch.setattr(fetcher, "_fetch_daily_bs", fake_bs)

    fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31", adjust="hfq")
    assert calls == ["bs"]


def test_bulk_update_daily_parallel_akshare_writes_rows(tmp_path, monkeypatch):
    from qkquant.data.storage import DuckStore

    store = DuckStore(tmp_path / "parallel.duckdb")
    fetcher = DataFetcher(store=store, source="akshare")
    calls: list[str] = []

    def fake_fetch_daily(code, start, end, adjust):
        calls.append(code)
        return pd.DataFrame(
            {
                "code": [code],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
                "pct_chg": [2.0],
                "turnover": [0.5],
                "adjust": ["hfq"],
            }
        )

    monkeypatch.setattr(fetcher, "fetch_daily", fake_fetch_daily)

    summary = fetcher.bulk_update_daily(
        ["000001", "000002"],
        start="2024-01-01",
        end="2024-01-02",
        adjust="hfq",
        incremental=False,
        jobs=2,
    )

    assert sorted(calls) == ["000001", "000002"]
    assert summary["jobs"] == 2
    assert summary["total_rows"] == 2
    assert store.stats()["bars"] == 2
    store.close()
