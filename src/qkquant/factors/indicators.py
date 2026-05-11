"""纯 pandas 实现的常用技术指标：MA / EMA / MACD / RSI / ATR。

选择自己实现而非 pandas-ta 的原因：pandas-ta 已停止维护，
且与 numpy 2.x / 新版 pandas 常有冲突，自实现几十行代码更稳。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移动平均（span=window，等价于通行的 EMA 定义）"""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """返回 DataFrame 含列 macd / signal / hist。"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}, index=close.index
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder 平滑 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)  # 避开 0/0 的 NaN，返回中性值


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """平均真实波幅。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def add_common_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
    ma_windows: tuple[int, ...] = (5, 10, 20, 60),
) -> pd.DataFrame:
    """在原 DataFrame 上追加常用指标列。"""
    out = df.copy()
    for w in ma_windows:
        out[f"ma{w}"] = sma(out[price_col], w)
    macd_df = macd(out[price_col])
    out = out.join(macd_df)
    out["rsi14"] = rsi(out[price_col], 14)
    if {"high", "low", "close"}.issubset(out.columns):
        out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    return out


__all__ = ["sma", "ema", "macd", "rsi", "atr", "add_common_indicators"]
