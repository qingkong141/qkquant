"""pytest 公用 fixtures：临时 DuckDB + 合成行情数据。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qkquant.data.storage import DAILY_COLUMNS, DuckStore
from qkquant.logger import setup_logger

setup_logger(level="WARNING")


def _trading_days(start: date, n: int) -> list[date]:
    """合成的"交易日"：跳过周六周日。"""
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _synth_price(seed: int, n: int, drift: float = 0.0005, vol: float = 0.015) -> np.ndarray:
    """几何布朗运动合成价格。"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=vol, size=n)
    price = 10.0 * np.exp(np.cumsum(returns))
    return price


def _make_bars(code: str, dates: list[date], seed: int) -> pd.DataFrame:
    n = len(dates)
    close = _synth_price(seed, n)
    open_ = close * (1 + np.random.default_rng(seed + 1).normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.default_rng(seed + 2).normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.default_rng(seed + 3).normal(0, 0.005, n)))
    volume = np.random.default_rng(seed + 4).integers(10_000_000, 50_000_000, n).astype(float)
    pct_chg = np.concatenate([[0.0], np.diff(close) / close[:-1] * 100])
    df = pd.DataFrame(
        {
            "code": code,
            "trade_date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "pct_chg": pct_chg,
            "turnover": np.random.default_rng(seed + 5).uniform(0.5, 3.0, n),
            "adjust": "hfq",
        }
    )
    return df[DAILY_COLUMNS]


@pytest.fixture
def synth_bars() -> pd.DataFrame:
    dates = _trading_days(date(2023, 1, 3), 250)
    frames = [
        _make_bars("000001", dates, seed=1),
        _make_bars("000002", dates, seed=2),
        _make_bars("600000", dates, seed=3),
        _make_bars("600036", dates, seed=4),
        _make_bars("601318", dates, seed=5),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def tmp_store(tmp_path: Path, synth_bars: pd.DataFrame) -> DuckStore:
    db_path = tmp_path / "test.duckdb"
    store = DuckStore(db_path)
    store.upsert_daily(synth_bars)
    store.upsert_index_constituents("000300", synth_bars["code"].unique().tolist())
    inst_df = pd.DataFrame(
        {
            "code": synth_bars["code"].unique(),
            "name": ["TestA", "TestB", "TestC", "TestD", "TestE"],
            "is_st": [False] * 5,
        }
    )
    store.upsert_instruments(inst_df)
    yield store
    store.close()
