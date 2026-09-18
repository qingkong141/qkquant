"""Validate a fixed ETF data snapshot before it can be admitted to research."""

from __future__ import annotations

import numpy as np
import pandas as pd
from qkquant.data.storage import DAILY_COLUMNS

PRICE_TOLERANCE = 0.0005001
OHLC = ["open", "high", "low", "close"]


class SnapshotValidationError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(f"{audit['code']}: {audit['reason']}")


def _date_strings(dates):
    return [str(day.date()) for day in dates]


def _reject(audit, reason, source=None, dates=None):
    audit.update(status="rejected", reason=reason)
    if source is not None:
        audit["source"] = source
    if dates is not None:
        audit["anomaly_dates"] = _date_strings(dates)
    raise SnapshotValidationError(audit)


def audit_etf(code, sina_raw, tx_raw, tx_qfq, factors, calendar, start, end):
    """Return verified qfq/raw bars and audit evidence, or reject the entire ETF.

    Sina supplies actual volume and turnover value; Tencent supplies qfq prices.
    The supported Sina ETF factor convention is ``raw / s - u``, with f == 1.
    Factor rows become effective on their own date, never on an earlier bar.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    audit = dict(code=str(code).zfill(6), status="checking", rows=0,
                 requested_start=str(start.date()), requested_end=str(end.date()),
                 price_tolerance=PRICE_TOLERANCE, availability_protocol="v2")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce")).normalize()
    if start > end or calendar.hasnans or calendar.has_duplicates:
        _reject(audit, "invalid_calendar_or_date_range")
    expected = calendar[(calendar >= start) & (calendar <= end)].sort_values()
    if expected.empty:
        _reject(audit, "empty_calendar")
    audit["expected_rows"] = len(expected)
    audit["source_rows"] = {}

    frames = {}
    for source, original in (("sina_raw", sina_raw), ("tx_raw", tx_raw), ("tx_qfq", tx_qfq)):
        required = {"trade_date", "volume", *OHLC}
        if source == "sina_raw":
            required.add("amount")
        if original is None or not required.issubset(original.columns):
            _reject(audit, "missing_bar_columns", source)
        frame = original.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        if frame.trade_date.isna().any():
            _reject(audit, "invalid_bar_date", source)
        frame = frame[frame.trade_date.between(start, end)].set_index("trade_date").sort_index()
        audit["source_rows"][source] = len(frame)
        if frame.index.has_duplicates:
            _reject(audit, "duplicate_bar_dates", source, frame.index[frame.index.duplicated()])
        extra = frame.index.difference(expected)
        if len(extra):
            _reject(audit, "bars_outside_trading_calendar", source, extra)
        frame[OHLC] = frame[OHLC].apply(pd.to_numeric, errors="coerce").astype(float)
        bad_price = (~np.isfinite(frame[OHLC]) | (frame[OHLC] <= 0)).any(axis=1)
        bad_bounds = ((frame.high < frame[["open", "close", "low"]].max(axis=1))
                      | (frame.low > frame[["open", "close", "high"]].min(axis=1)))
        if bad_price.any():
            _reject(audit, "invalid_ohlc", source, frame.index[bad_price])
        if bad_bounds.any():
            _reject(audit, "inconsistent_ohlc_bounds", source, frame.index[bad_bounds])
        frames[source] = frame

    observed = frames["sina_raw"].index
    for source in ("tx_raw", "tx_qfq"):
        if not observed.equals(frames[source].index):
            _reject(audit, "source_date_mismatch", source,
                    observed.symmetric_difference(frames[source].index))
    if observed.empty:
        _reject(audit, "no_common_actual_bars")
    unavailable = expected.difference(observed)
    audit.update(calendar_missing_dates=_date_strings(unavailable),
                 calendar_missing_reason="unknown_unavailable" if len(unavailable) else None,
                 calendar_missing=[dict(date=str(day.date()), reason="unknown_unavailable")
                                   for day in unavailable])

    if factors is None or factors.empty or not {"d", "f", "s", "u"}.issubset(factors.columns):
        _reject(audit, "missing_adjustment_factors", "sina_factors")
    factors = factors[["d", "f", "s", "u"]].copy()
    factors["d"] = pd.to_datetime(factors.d, errors="coerce").dt.normalize()
    if factors.d.isna().any():
        _reject(audit, "invalid_factor_date", "sina_factors")
    factors = factors.sort_values("d")
    if factors.d.duplicated().any():
        _reject(audit, "duplicate_factor_dates", "sina_factors", factors.loc[factors.d.duplicated(), "d"])
    if (factors.d > end).any():
        _reject(audit, "factor_after_frozen_end", "sina_factors", factors.loc[factors.d > end, "d"])
    factors[["f", "s", "u"]] = factors[["f", "s", "u"]].apply(pd.to_numeric, errors="coerce").astype(float)
    unsupported = ((~np.isfinite(factors[["f", "s", "u"]])).any(axis=1)
                   | (factors.f != 1) | (factors.s <= 0) | (factors.u < 0))
    if unsupported.any():
        _reject(audit, "unsupported_adjustment_factor", "sina_factors", factors.loc[unsupported, "d"])
    if factors.d.iloc[0] > observed[0]:
        _reject(audit, "factor_does_not_cover_first_bar", "sina_factors", observed[:1])
    if factors.s.iloc[-1] != 1 or factors.u.iloc[-1] != 0:
        _reject(audit, "qfq_not_anchored_at_frozen_end", "sina_factors", factors.d.iloc[-1:])
    factors["s_ratio_to_previous"] = factors.s / factors.s.shift(1)
    audit["factor_events"] = [dict(d=str(row.d.date()), f=float(row.f), s=float(row.s),
                                  u=float(row.u), s_ratio_to_previous=(
                                      float(row.s_ratio_to_previous) if pd.notna(row.s_ratio_to_previous) else None))
                              for row in factors.itertuples(index=False)]
    applied = pd.merge_asof(pd.DataFrame({"trade_date": observed}), factors,
                            left_on="trade_date", right_on="d", direction="backward").set_index("trade_date")
    raw = frames["sina_raw"]
    reconstructed = raw[OHLC].div(applied.s, axis=0).sub(applied.u, axis=0)
    raw_difference = (raw[OHLC] - frames["tx_raw"][OHLC]).abs()
    qfq_difference = (reconstructed - frames["tx_qfq"][OHLC]).abs()
    audit.update(max_raw_price_difference=float(raw_difference.to_numpy().max()),
                 max_qfq_price_difference=float(qfq_difference.to_numpy().max()))
    for source, difference in (("tx_raw", raw_difference), ("tx_qfq", qfq_difference)):
        bad = (difference > PRICE_TOLERANCE).any(axis=1)
        if bad.any():
            _reject(audit, "cross_source_price_mismatch", source, difference.index[bad])

    raw["volume"] = pd.to_numeric(raw.volume, errors="coerce").astype(float)
    raw["amount"] = pd.to_numeric(raw.amount, errors="coerce").astype(float)
    invalid_volume = ~np.isfinite(raw.volume) | (raw.volume < 0)
    invalid_amount = np.isinf(raw.amount) | (raw.amount < 0)
    if invalid_volume.any():
        _reject(audit, "invalid_volume", "sina_raw", raw.index[invalid_volume])
    if invalid_amount.any():
        _reject(audit, "invalid_amount", "sina_raw", raw.index[invalid_amount])
    invalid_zero_volume = (raw.volume == 0) & (raw.amount > 0)
    if invalid_zero_volume.any():
        _reject(audit, "positive_amount_with_zero_volume", "sina_raw", raw.index[invalid_zero_volume])
    traded = (raw.volume > 0) & (raw.amount > 0)
    audit["amount_rounding_cny"] = .5
    # Sina reports whole yuan: use money bounds so a 100-share trade can round
    # by half a yuan without incorrectly widening the underlying price tolerance.
    minimum_amount, maximum_amount = raw.low * raw.volume, raw.high * raw.volume
    epsilon = np.maximum(1e-9, np.finfo(float).eps * maximum_amount * 4)
    invalid_units = traded & ((raw.amount < minimum_amount - .5 - epsilon)
                             | (raw.amount > maximum_amount + .5 + epsilon))
    if invalid_units.any():
        _reject(audit, "amount_volume_units_inconsistent_with_ohlc", "sina_raw", raw.index[invalid_units])
    tx_shares = pd.to_numeric(frames["tx_raw"].volume, errors="coerce").astype(float) * 100
    comparable = np.isfinite(tx_shares) & (tx_shares >= 0)
    ratios = tx_shares[comparable & (raw.volume > 0)] / raw.volume[comparable & (raw.volume > 0)]
    differences = (tx_shares[comparable] - raw.volume[comparable]).abs()
    audit["volume_comparison"] = dict(
        informational_only=True,
        market_scope="frozen_shenzhen_etf" if audit["code"].startswith("15") else "other_market_not_validated",
        unit_assumption="Tencent raw volume * 100 vs Sina shares; no cross-source volume rejection",
        median_ratio=float(ratios.median()) if len(ratios) else None,
        max_abs_difference=float(differences.max()) if len(differences) else None)
    audit.update(zero_amount_dates=_date_strings(raw.index[raw.amount == 0]),
                 missing_amount_dates=_date_strings(raw.index[raw.amount.isna()]),
                 untradable_dates=_date_strings(raw.index[raw.amount.isna() | (raw.amount == 0)]))

    def daily(prices, adjust):
        result = prices[OHLC].copy()
        result.index.name = "trade_date"
        result["code"] = audit["code"]
        result["volume"], result["amount"] = raw.volume, raw.amount
        result["pct_chg"] = result.close.reindex(expected).pct_change(fill_method=None).reindex(observed) * 100
        result["turnover"], result["adjust"] = None, adjust
        return result.reset_index()[DAILY_COLUMNS]

    audit.update(status="accepted", rows=len(observed), start=str(observed[0].date()),
                 end=str(observed[-1].date()), adjustment_formula="raw / s - u")
    return daily(frames["tx_qfq"], "qfq"), daily(raw, ""), audit
