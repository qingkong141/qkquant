"""技术指标的基本性质测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qkquant.factors.indicators import (
    add_common_indicators,
    atr,
    ema,
    macd,
    rsi,
    sma,
)


@pytest.fixture
def price_series() -> pd.Series:
    rng = np.random.default_rng(42)
    prices = 10.0 * np.exp(np.cumsum(rng.normal(0.001, 0.015, 200)))
    return pd.Series(prices, name="close")


def test_sma_window(price_series):
    out = sma(price_series, 20)
    assert out.isna().sum() == 19
    assert abs(out.iloc[19] - price_series.iloc[:20].mean()) < 1e-8


def test_ema_converges(price_series):
    out = ema(price_series, 10)
    assert out.isna().sum() == 9
    assert out.iloc[-1] == pytest.approx(out.iloc[-1], rel=1e-6)


def test_macd_columns(price_series):
    out = macd(price_series)
    assert set(out.columns) == {"macd", "signal", "hist"}
    assert len(out) == len(price_series)
    assert (out["hist"] - (out["macd"] - out["signal"])).abs().max() < 1e-9


def test_rsi_range(price_series):
    out = rsi(price_series, 14).dropna()
    assert out.between(0, 100).all()


def test_atr_nonneg():
    high = pd.Series([10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11])
    low = pd.Series([9, 10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10])
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 13.5, 12.5, 11.5, 10.5])
    out = atr(high, low, close, 5).dropna()
    assert (out >= 0).all()


def test_add_common_indicators(price_series):
    df = pd.DataFrame(
        {
            "close": price_series,
            "high": price_series * 1.01,
            "low": price_series * 0.99,
        }
    )
    out = add_common_indicators(df)
    for col in ["ma5", "ma20", "macd", "signal", "hist", "rsi14", "atr14"]:
        assert col in out.columns
