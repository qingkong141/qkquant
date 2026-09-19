"""Historical cohort, native turnover and point-in-time eligibility checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qkquant.oneil import OneilConfig, prepare_prices
from qkquant.oneil_universe import checked_baostock_bars, frozen_holdings_cohort


def holdings(codes):
    return pd.DataFrame(dict(code=codes, publication_date="2024-02-28",
                             holdings_asof="2023-12-31"))


def test_frozen_cohort_keeps_historical_suspensions_st_and_later_delistings():
    rows = holdings([600002, 408, 600001, 300001, 688001, 600003])
    historical = pd.DataFrame({
        "code": ["sh.600001", "sh.600002", "sz.000408", "sz.300001", "sh.688001"],
        "code_name": ["*ST historical member", "delisted today", "suspended", "growth", "STAR"],
        "tradeStatus": ["1", "1", "0", "1", "1"],
        "isST": ["1", "0", "0", "0", "0"],
        "currently_delisted": [False, True, False, False, False],
    })
    original_rows, original_history = rows.copy(deep=True), historical.copy(deep=True)

    selected, audit = frozen_holdings_cohort(rows, historical, "2024-03-01")

    assert selected.code.tolist() == ["000408", "600001", "600002"]
    assert selected.bs_code.tolist() == ["sz.000408", "sh.600001", "sh.600002"]
    statuses = audit.set_index("code").cohort_status.to_dict()
    assert statuses["300001"] == statuses["688001"] == "outside_main_board"
    assert statuses["600003"] == "not_in_historical_listing"
    pd.testing.assert_frame_equal(rows, original_rows)
    pd.testing.assert_frame_equal(historical, original_history)


@pytest.mark.parametrize("publication,asof", [
    ("2024-03-01", "2023-12-31"),
    ("2024-03-02", "2023-12-31"),
    ("2024-02-28", "2024-02-29"),
    (None, "2023-12-31"),
    ("2024-02-28", None),
])
def test_holdings_must_be_public_before_inception(publication, asof):
    rows = holdings(["600001"])
    rows["publication_date"], rows["holdings_asof"] = publication, asof
    with pytest.raises(ValueError, match="available before cohort inception"):
        frozen_holdings_cohort(rows, pd.DataFrame({"code": ["sh.600001"]}), "2024-03-01")


def test_index_codes_cannot_supply_stock_listing_evidence():
    rows = holdings(["000001", "000300", "399001", "510300"])
    indices_and_fund = pd.DataFrame({"code": ["sh.000001", "sh.000300", "sz.399001", "sh.510300"]})

    selected, audit = frozen_holdings_cohort(rows, indices_and_fund, "2024-03-01")
    assert selected.empty
    assert audit.set_index("code").loc["000001", "bs_code"] == "sz.000001"

    historical = pd.concat([indices_and_fund, pd.DataFrame({"code": ["sz.000001"]})])
    selected, _ = frozen_holdings_cohort(rows, historical, "2024-03-01")
    assert selected.code.tolist() == ["000001"]


@pytest.mark.parametrize("duplicate_side", ["holdings", "historical"])
def test_duplicate_members_cannot_inflate_the_cohort(duplicate_side):
    rows = holdings(["000001"])
    historical = pd.DataFrame({"code": ["sz.000001"]})
    if duplicate_side == "holdings":
        # Numeric and padded representations identify the same security.
        rows = holdings([1, "000001"])
    else:
        historical = pd.concat([historical, historical], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate universe code"):
        frozen_holdings_cohort(rows, historical, "2024-03-01")


def baostock_pair(periods=3):
    raw = pd.DataFrame({
        "date": pd.bdate_range("2024-03-01", periods=periods).strftime("%Y-%m-%d"),
        "code": "sh.600001", "open": "10", "high": "10.2", "low": "9.8", "close": "10",
        "volume": "1000", "amount": "10000", "tradestatus": "1", "isST": "0",
    })
    adjusted = raw.copy(deep=True)
    for field in ("open", "high", "low", "close"):
        adjusted[field] = pd.to_numeric(raw[field]) * 10
    # Adjusted quote metadata must not replace the native share/CNY turnover.
    adjusted["volume"], adjusted["amount"] = "10", "1000000"
    return raw, adjusted


def test_native_turnover_uses_raw_prices_and_dates_not_hfq_scale_or_row_order():
    raw, adjusted = baostock_pair()
    adjusted.loc[1, ["open", "high", "low", "close"]] = [200, 204, 196, 200]
    adjusted = adjusted.iloc[::-1].reset_index(drop=True)
    raw_before, adjusted_before = raw.copy(deep=True), adjusted.copy(deep=True)

    bars, audit = checked_baostock_bars(raw, adjusted, "600001")

    assert bars.close.tolist() == [100, 200, 100]
    assert bars.raw_close.tolist() == [10, 10, 10]
    assert bars.volume.tolist() == [1000, 1000, 1000]
    assert bars.amount.tolist() == [10000, 10000, 10000]
    assert bars.eligible.all()
    assert audit["quarantined_rows"] == 0
    assert bars.trade_date.is_monotonic_increasing
    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(adjusted, adjusted_before)


@pytest.mark.parametrize("missing_side", ["raw", "adjusted"])
def test_raw_and_adjusted_calendar_mismatch_is_rejected(missing_side):
    raw, adjusted = baostock_pair()
    if missing_side == "raw":
        raw = raw.iloc[:-1]
    else:
        adjusted = adjusted.iloc[:-1]
    with pytest.raises(ValueError, match="calendars differ"):
        checked_baostock_bars(raw, adjusted, "600001")


@pytest.mark.parametrize("bad_side", ["raw", "adjusted"])
def test_quote_code_mismatch_and_duplicate_dates_are_rejected(bad_side):
    raw, adjusted = baostock_pair()
    frame = raw if bad_side == "raw" else adjusted
    frame.loc[0, "code"] = "sh.000001"
    with pytest.raises(ValueError, match="bar code mismatch"):
        checked_baostock_bars(raw, adjusted, "600001")
    frame.loc[0, "code"] = "sh.600001"
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicate bar date"):
        checked_baostock_bars(raw, adjusted, "600001")


def test_suspension_cannot_trade_despite_vendor_carrying_nonzero_ohlc():
    raw, adjusted = baostock_pair()
    raw.loc[1, "tradestatus"] = "0"

    bars, audit = checked_baostock_bars(raw, adjusted, "600001")

    assert len(bars) == 3  # Preserve the reported date rather than manufacturing a bar.
    assert bars.eligible.tolist() == [True, False, True]
    price_and_turnover = ["open", "high", "low", "close", "raw_open", "raw_high",
                          "raw_low", "raw_close", "volume", "amount"]
    assert bars.loc[1, price_and_turnover].isna().all()
    assert audit["suspended_rows"] == 1
    assert audit["quarantined_rows"] == 0


def test_historical_st_is_excluded_only_on_its_flagged_date():
    raw, adjusted = baostock_pair()
    raw.loc[1, "isST"] = "1"
    bars, audit = checked_baostock_bars(raw, adjusted, "600001")
    assert bars.eligible.tolist() == [True, False, True]
    assert bars.historical_is_st.tolist() == ["0", "1", "0"]
    assert audit["st_rows"] == 1
    assert audit["quarantined_rows"] == 0


def test_adjusted_close_inside_daily_range_still_needs_consistent_price_basis():
    raw, adjusted = baostock_pair()
    # This passes ordinary OHLC bounds but uses a different factor for close.
    adjusted.loc[1, "close"] = 101
    bars, audit = checked_baostock_bars(raw, adjusted, "600001")
    assert bars.eligible.tolist() == [True, False, True]
    assert bars.loc[1, ["open", "close", "raw_close", "volume"]].isna().all()
    assert audit["inconsistent_adjustment_rows"] == 1


@pytest.mark.parametrize("missing_st", [None, np.nan, "", "unknown"])
def test_unknown_historical_st_value_never_enters_the_tradable_set(missing_st):
    raw, adjusted = baostock_pair()
    raw.loc[1, "isST"] = missing_st
    bars, audit = checked_baostock_bars(raw, adjusted, "600001")
    assert bars.eligible.tolist() == [True, False, True]
    assert audit["quarantined_dates"] == [raw.loc[1, "date"]]


def test_missing_historical_st_column_fails_closed_with_a_clear_contract():
    raw, adjusted = baostock_pair()
    raw = raw.drop(columns="isST")
    try:
        bars, _ = checked_baostock_bars(raw, adjusted, "600001")
    except ValueError as error:
        assert "isST" in str(error)
    else:
        assert not bars.eligible.any()


@pytest.mark.parametrize("raw_changes,adjusted_changes", [
    ({"volume": "10"}, {}),  # Lots cannot silently pass as native shares.
    ({"amount": "1"}, {}),  # Ten-thousand-CNY units cannot pass as native CNY.
    ({"volume": "0", "amount": "0"}, {}),
    ({"volume": "-1000"}, {}),
    ({"close": "11"}, {}),
    ({}, {"close": 110}),
    ({"amount": "not-a-number"}, {}),
    ({"volume": "inf", "amount": "inf"}, {}),
])
def test_anomalous_price_or_turnover_is_quarantined_without_poisoning_other_dates(
        raw_changes, adjusted_changes):
    raw, adjusted = baostock_pair()
    for field, value in raw_changes.items():
        raw.loc[1, field] = value
    for field, value in adjusted_changes.items():
        adjusted.loc[1, field] = value
    bars, audit = checked_baostock_bars(raw, adjusted, "600001")
    assert bars.eligible.tolist() == [True, False, True]
    assert bars.loc[1, ["close", "raw_close", "volume", "amount"]].isna().all()
    assert bars.loc[[0, 2], "volume"].tolist() == [1000, 1000]
    assert audit["quarantined_rows"] == 1
    assert audit["quarantined_dates"] == [raw.loc[1, "date"]]


def breakout_market():
    paths = [np.r_[np.geomspace(start, 20, 160), np.full(25, 20), 20.8, 20.85, 21, 21.5]
             for start in (5, 10, 15)]
    days = pd.bdate_range("2023-01-02", periods=len(paths[0]))
    volume = np.full(len(days), 10_000_000.)
    volume[185] *= 2
    bars = pd.concat([
        pd.DataFrame(dict(code=code, trade_date=days, open=close, close=close,
                          high=close * 1.005, low=close * .995, volume=volume,
                          amount=volume * close, raw_open=close, raw_close=close,
                          raw_high=close * 1.005, raw_low=close * .995))
        for code, close in zip(("600001", "600002", "600003"), paths, strict=True)
    ], ignore_index=True)
    benchmark = pd.DataFrame(dict(trade_date=days, raw_close=np.linspace(100, 200, len(days))))
    return bars, benchmark, replace(OneilConfig(), rs_days=60), days[185]


def assert_prepared_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], pd.DataFrame):
            pd.testing.assert_frame_equal(left[key], right[key], obj=key)
        else:
            pd.testing.assert_series_equal(left[key], right[key], obj=key)


def test_omitting_eligible_is_identical_to_explicit_all_true():
    bars, benchmark, cfg, day = breakout_market()
    legacy = prepare_prices(bars, benchmark, cfg)
    explicit = prepare_prices(bars.assign(eligible=True), benchmark, cfg)
    assert legacy["technical"].loc[day, "600001"]
    assert_prepared_equal(legacy, explicit)


def test_ineligible_leader_does_not_dilute_rs_ranking_or_trigger_a_breakout():
    bars, benchmark, cfg, day = breakout_market()
    baseline = prepare_prices(bars, benchmark, cfg)
    assert baseline["technical"].loc[day, "600001"]
    assert not baseline["technical"].loc[day, "600002"]
    assert baseline["rs"].loc[day, "600002"] == pytest.approx(2 / 3)

    bars["eligible"] = True
    bars.loc[(bars.code == "600001") & (bars.trade_date == day), "eligible"] = False
    masked = prepare_prices(bars, benchmark, cfg)

    assert pd.isna(masked["rs"].loc[day, "600001"])
    assert not masked["technical"].loc[day, "600001"]
    assert masked["rs"].loc[day, "600002"] == 1
    assert masked["rs"].loc[day, "600003"] == .5
    assert masked["technical"].loc[day, "600002"]
    pd.testing.assert_frame_equal(masked["rs"].loc[:day].iloc[:-1],
                                  baseline["rs"].loc[:day].iloc[:-1])


def test_future_eligibility_cannot_rewrite_prefix_indicators_or_signals():
    bars, benchmark, cfg, cutoff = breakout_market()
    bars["eligible"] = True
    bars.loc[(bars.trade_date > cutoff) & (bars.code != "600003"), "eligible"] = False
    full = prepare_prices(bars, benchmark, cfg)
    prefix = prepare_prices(bars[bars.trade_date <= cutoff],
                            benchmark[benchmark.trade_date <= cutoff], cfg)
    assert prefix["technical"].loc[cutoff, "600001"]
    assert_prepared_equal({key: value.loc[:cutoff] for key, value in full.items()}, prefix)
