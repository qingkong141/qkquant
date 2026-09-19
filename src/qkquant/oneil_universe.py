"""Historical-cohort selection and daily-bar checks for O'Neil research."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAIN_PREFIXES = ("sh.600", "sh.601", "sh.603", "sh.605", "sz.000", "sz.001", "sz.002", "sz.003")


def frozen_holdings_cohort(holdings: pd.DataFrame, historical: pd.DataFrame, day):
    """Use an already public holding list, retaining suspended historical members."""
    day = pd.Timestamp(day).normalize()
    rows = holdings.copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    if rows.code.duplicated().any() or historical.code.duplicated().any():
        raise ValueError("duplicate universe code")
    published = pd.to_datetime(rows.publication_date, errors="raise")
    asof = pd.to_datetime(rows.holdings_asof, errors="raise")
    if published.isna().any() or asof.isna().any() or (published >= day).any() or (asof > published).any():
        raise ValueError("holding list was not available before cohort inception")
    rows["bs_code"] = rows.code.map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    rows["cohort_status"] = np.select(
        [~rows.bs_code.str.startswith(MAIN_PREFIXES), ~rows.bs_code.isin(historical.code)],
        ["outside_main_board", "not_in_historical_listing"], default="selected")
    # Do not use today's name, ST flag or survival status to select this cohort.
    selected = rows[rows.cohort_status == "selected"].sort_values("code").reset_index(drop=True)
    return selected, rows


def checked_baostock_bars(raw: pd.DataFrame, adjusted: pd.DataFrame, code: str):
    """Keep native shares/CNY turnover and hfq prices; quarantine broken bars.

    No synthetic suspension or post-delisting candles are inserted. A bar with
    unknown historical ST/trading status cannot enter or join the RS ranking.
    """
    raw, adjusted = raw.copy(), adjusted.copy()
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST"}
    for frame in (raw, adjusted):
        if required - set(frame.columns):
            raise ValueError(f"missing bar fields: {sorted(required - set(frame.columns))}")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.date.isna().any() or frame.date.duplicated().any():
            raise ValueError("missing or duplicate bar date")
        if not frame.code.eq(("sh." if code.startswith("6") else "sz.") + code).all():
            raise ValueError("bar code mismatch")
    if set(raw.date) != set(adjusted.date):
        raise ValueError("raw and adjusted calendars differ")
    fields = ["open", "high", "low", "close"]
    pair = raw.merge(adjusted[["date", *fields]], on="date", suffixes=("_raw", ""), validate="one_to_one")
    for field in [*fields, *(f"{f}_raw" for f in fields), "volume", "amount"]:
        pair[field] = pd.to_numeric(pair[field], errors="coerce")
    trading = pair.tradestatus.astype(str).eq("1")
    state_known = pair.tradestatus.astype(str).isin(["0", "1"]) & pair.isST.astype(str).isin(["0", "1"])
    ohlc = pair[[*fields, *(f"{f}_raw" for f in fields)]]
    valid = np.isfinite(ohlc).all(axis=1) & ohlc.gt(0).all(axis=1)
    for suffix in ("", "_raw"):
        valid &= (pair[f"low{suffix}"] <= pair[f"open{suffix}"]) & (pair[f"open{suffix}"] <= pair[f"high{suffix}"])
        valid &= (pair[f"low{suffix}"] <= pair[f"close{suffix}"]) & (pair[f"close{suffix}"] <= pair[f"high{suffix}"])
    # BaoStock applies one multiplicative factor to all four prices each day.
    scale = pair.close / pair.close_raw
    adjustment_ok = np.column_stack([
        np.isclose(pair[field], pair[f"{field}_raw"] * scale, rtol=1e-7, atol=1e-6)
        for field in fields]).all(axis=1)
    valid &= adjustment_ok
    turnover_ok = (np.isfinite(pair[["volume", "amount"]]).all(axis=1)
                   & pair.volume.gt(0) & pair.amount.gt(0)
                   & pair.amount.ge(pair.volume * pair.low_raw * .98)
                   & pair.amount.le(pair.volume * pair.high_raw * 1.02))
    broken = trading & (~valid | ~turnover_ok | ~state_known)
    unavailable = ~trading | broken
    pair.loc[unavailable, [*fields, *(f"{f}_raw" for f in fields)]] = np.nan
    pair.loc[unavailable, ["volume", "amount"]] = np.nan
    result = pd.DataFrame(dict(code=code, trade_date=pair.date, **{f: pair[f] for f in fields},
                               **{f"raw_{f}": pair[f"{f}_raw"] for f in fields},
                               volume=pair.volume, amount=pair.amount,
                               eligible=trading & ~broken & state_known & pair.isST.astype(str).eq("0"),
                               historical_is_st=pair.isST, historical_trade_status=pair.tradestatus))
    audit = dict(code=code, rows=len(result), trading_rows=int(trading.sum()),
                 suspended_rows=int((~trading & state_known).sum()),
                 unknown_status_rows=int((~state_known).sum()),
                 inconsistent_adjustment_rows=int((trading & ~adjustment_ok).sum()),
                 st_rows=int(pair.isST.astype(str).eq("1").sum()),
                 first_reported_date=pair.date.min(), last_reported_date=pair.date.max(),
                 last_valid_trading_date=pair.loc[~unavailable, "date"].max(),
                 quarantined_rows=int(broken.sum()),
                 quarantined_dates=pair.loc[broken, "date"].dt.strftime("%Y-%m-%d").tolist())
    return result.sort_values("trade_date").reset_index(drop=True), audit
