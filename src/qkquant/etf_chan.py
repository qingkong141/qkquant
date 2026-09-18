"""Deterministic Chan-inspired structure filter for ETF daily bars.

This is intentionally a simplified, testable approximation. It does not claim
to implement every discretionary rule of classical Chan theory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChanSignal:
    state: str
    allowed: bool
    last_price: float
    last_low: float | None
    previous_low: float | None
    last_high: float | None
    previous_high: float | None
    ma20: float | None
    divergence: str | None
    divergence_strength: float | None
    reason: str
    previous_high_date: str | None = None
    breakout_high_date: str | None = None
    pullback_date: str | None = None
    confirmation_date: str | None = None
    signal_age_bars: int | None = None
    distance_to_pullback: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fractals(values: pd.Series, order: int = 2) -> tuple[pd.Series, pd.Series]:
    window = order * 2 + 1
    highs = values.where(values == values.rolling(window, center=True).max()).dropna()
    lows = values.where(values == values.rolling(window, center=True).min()).dropna()
    return highs, lows


def classify_chan_structure(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    fractal_order: int = 2,
    tolerance: float = 0.015,
) -> ChanSignal:
    """Classify a daily ETF series as second/third-buy candidate or no entry.

    second_buy: latest confirmed swing low is above the prior swing low, price
    is above a rising MA20, and price has reclaimed the latest swing high.

    third_buy: ordered old high -> breakout high -> pullback low, on the
    pullback confirmation bar only. third_buy_active is continuation, not entry.
    """
    if not isinstance(fractal_order, int) or fractal_order < 1:
        raise ValueError("fractal_order must be a positive integer")
    frame = pd.concat(
        {
            "close": pd.to_numeric(close, errors="coerce"),
            "high": pd.to_numeric(high if high is not None else close, errors="coerce"),
            "low": pd.to_numeric(low if low is not None else close, errors="coerce"),
        },
        axis=1,
    )
    if not frame.empty and frame.iloc[-1].isna().any():
        return ChanSignal("missing_latest_bar", False, np.nan, None, None, None, None,
                          None, None, None, "latest_bar_must_be_complete")
    frame = frame.dropna().tail(200)
    if len(frame) < 30:
        return ChanSignal("insufficient_data", False, np.nan, None, None, None, None, None, None, None, "need_at_least_30_bars")

    swing_highs, _ = _fractals(frame["high"], fractal_order)
    _, swing_lows = _fractals(frame["low"], fractal_order)
    price = float(frame["close"].iloc[-1])
    ma20_series = frame["close"].rolling(20).mean()
    ma20 = float(ma20_series.iloc[-1])
    ma20_rising = bool(ma20_series.iloc[-1] > ma20_series.iloc[-6])
    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    macd_hist = 2 * (diff - dea)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return ChanSignal("insufficient_fractals", False, price, None, None, None, None, ma20, None, None, "need_two_highs_and_two_lows")

    last_high, previous_high = float(swing_highs.iloc[-1]), float(swing_highs.iloc[-2])
    last_low, previous_low = float(swing_lows.iloc[-1]), float(swing_lows.iloc[-2])
    last_low_mom = float(macd_hist.loc[swing_lows.index[-1]])
    previous_low_mom = float(macd_hist.loc[swing_lows.index[-2]])
    last_high_mom = float(macd_hist.loc[swing_highs.index[-1]])
    previous_high_mom = float(macd_hist.loc[swing_highs.index[-2]])
    bottom_divergence = last_low < previous_low * (1 - tolerance) and last_low_mom > previous_low_mom
    top_divergence = last_high > previous_high * (1 + tolerance) and last_high_mom < previous_high_mom
    divergence = "bottom" if bottom_divergence else "top" if top_divergence else None
    divergence_strength = None
    if bottom_divergence:
        divergence_strength = (last_low_mom - previous_low_mom) / max(abs(previous_low_mom), 1e-9)
    elif top_divergence:
        divergence_strength = (previous_high_mom - last_high_mom) / max(abs(previous_high_mom), 1e-9)

    if bottom_divergence and price > ma20:
        return ChanSignal("first_buy_divergence", True, price, last_low, previous_low, last_high, previous_high, ma20, divergence, divergence_strength, "lower_price_low_with_stronger_macd_and_ma20_reclaim")
    if top_divergence:
        return ChanSignal("top_divergence", False, price, last_low, previous_low, last_high, previous_high, ma20, divergence, divergence_strength, "higher_price_high_with_weaker_macd")

    # A confirmed breakout followed by a pullback that remains above the old high.
    breakout_level = previous_high
    third_buy = (
        last_high > breakout_level * (1 + tolerance)
        and last_low >= breakout_level * (1 - tolerance)
        and price > last_low
        and price > ma20
    )
    if third_buy:
        old_date, breakout_date = swing_highs.index[-2], swing_highs.index[-1]
        pullback_date = swing_lows.index[-1]
        confirmation_pos = frame.index.get_loc(pullback_date) + fractal_order
        age = len(frame) - 1 - confirmation_pos
        details = dict(
            previous_high_date=str(old_date), breakout_high_date=str(breakout_date),
            pullback_date=str(pullback_date), confirmation_date=str(frame.index[confirmation_pos]),
            signal_age_bars=age, distance_to_pullback=price / last_low - 1,
        )
        if not old_date < breakout_date < pullback_date:
            return ChanSignal("third_buy_invalid_order", False, price, last_low, previous_low,
                              last_high, previous_high, ma20, divergence, divergence_strength,
                              "require_old_high_then_breakout_high_then_pullback_low", **details)
        fresh = age == 0
        return ChanSignal("third_buy" if fresh else "third_buy_active", fresh, price,
                          last_low, previous_low, last_high, previous_high, ma20,
                          divergence, divergence_strength,
                          "ordered_pullback_just_confirmed" if fresh else "old_pullback_no_new_entry",
                          **details)

    # A higher low followed by renewed strength is the mechanical second-buy proxy.
    second_buy = (
        last_low > previous_low * (1 + tolerance)
        and price > last_high
        and price > ma20
        and ma20_rising
    )
    if second_buy:
        return ChanSignal("second_buy", True, price, last_low, previous_low, last_high, previous_high, ma20, divergence, divergence_strength, "higher_low_and_reclaimed_swing_high")

    if price < last_low * (1 - tolerance):
        state, reason = "structure_broken", "price_below_latest_swing_low"
    elif last_low > previous_low:
        state, reason = "second_buy_watch", "higher_low_but_reclaim_not_confirmed"
    else:
        state, reason = "no_entry", "no_confirmed_second_or_third_buy"
    return ChanSignal(state, False, price, last_low, previous_low, last_high, previous_high, ma20, divergence, divergence_strength, reason)


def chan_selection_allowed(signal: ChanSignal, entry_states: tuple[str, ...], holding: bool = False) -> bool:
    """An old third-buy may retain a holding but must never open a new one."""
    if signal.state == "third_buy_active":
        return holding and "third_buy" in entry_states
    if signal.state == "third_buy_invalid_order":
        return False
    return signal.allowed and signal.state in entry_states


__all__ = ["ChanSignal", "chan_selection_allowed", "classify_chan_structure"]
