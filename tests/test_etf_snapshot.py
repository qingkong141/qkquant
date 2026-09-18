import numpy as np
import pandas as pd
import pytest
from qkquant.data.etf_snapshot import SnapshotValidationError, audit_etf
from qkquant.data.storage import DAILY_COLUMNS
from qkquant.etf_signal import EtfSignalConfig
from qkquant.etf_signal_diagnostics import event_outcome


def inputs():
    dates = pd.bdate_range("2026-01-05", periods=4)
    close = np.array([20., 21., 10.5, 11.])
    sina = pd.DataFrame(dict(trade_date=dates, open=close, high=close + .2,
                             low=close - .2, close=close, volume=[100., 200., 300., 400.],
                             amount=[2000., 4200., 3150., 4400.]))
    tx_raw = sina.drop(columns="amount").copy()
    tx_raw["volume"] = 999.
    qfq = tx_raw.copy()
    qfq.loc[:1, ["open", "high", "low", "close"]] /= 2
    factors = pd.DataFrame(dict(d=[dates[0], dates[2]], f=[1., 1.], s=[2., 1.], u=[0., 0.]))
    return dict(code="159919", sina_raw=sina, tx_raw=tx_raw, tx_qfq=qfq,
                factors=factors, calendar=dates, start=dates[0], end=dates[-1])


def test_split_divides_s_and_only_uses_effective_factor():
    args = inputs()
    qfq, raw, audit = audit_etf(**args)
    assert qfq.columns.tolist() == raw.columns.tolist() == DAILY_COLUMNS
    assert qfq.close.tolist() == [10., 10.5, 10.5, 11.]
    assert raw.close.tolist() == [20., 21., 10.5, 11.]
    assert qfq.volume.tolist() == raw.volume.tolist() == args["sina_raw"].volume.tolist()
    assert qfq.amount.tolist() == raw.amount.tolist() == args["sina_raw"].amount.tolist()
    assert set(qfq.adjust) == {"qfq"} and set(raw.adjust) == {""}
    assert qfq.pct_chg.iloc[2] == pytest.approx(0.)
    assert raw.pct_chg.iloc[2] == pytest.approx(-50.)
    assert pd.isna(qfq.pct_chg.iloc[0]) and qfq.turnover.isna().all()
    assert audit["status"] == "accepted" and audit["rows"] == 4
    assert audit["max_qfq_price_difference"] == 0
    assert audit["factor_events"][1]["s_ratio_to_previous"] == .5
    assert audit["untradable_dates"] == []


def test_cash_dividend_u_is_subtracted_in_price_units_after_division():
    args = inputs()
    args["factors"].loc[0, "u"] = .12
    args["tx_qfq"].loc[:1, ["open", "high", "low", "close"]] -= .12
    qfq, _, audit = audit_etf(**args)
    assert qfq.close.tolist() == [9.88, 10.38, 10.5, 11.]
    assert audit["max_qfq_price_difference"] == 0
    args["tx_qfq"].loc[:1, ["open", "high", "low", "close"]] += .108
    with pytest.raises(SnapshotValidationError) as error:
        audit_etf(**args)
    assert error.value.audit["source"] == "tx_qfq"


def test_zero_and_missing_amount_are_retained_and_flagged():
    args = inputs()
    args["sina_raw"].loc[1, "amount"] = 0
    args["sina_raw"].loc[2, "amount"] = np.nan
    qfq, raw, audit = audit_etf(**args)
    assert len(raw) == len(qfq) == 4
    assert raw.amount.iloc[1] == 0 and pd.isna(qfq.amount.iloc[2])
    assert audit["zero_amount_dates"] == ["2026-01-06"]
    assert audit["missing_amount_dates"] == ["2026-01-07"]
    assert audit["untradable_dates"] == ["2026-01-06", "2026-01-07"]


@pytest.mark.parametrize("source", ["sina_raw", "tx_raw", "tx_qfq"])
def test_missing_session_rejects_entire_etf(source):
    args = inputs()
    args[source] = args[source].drop(index=1)
    with pytest.raises(SnapshotValidationError) as error:
        audit_etf(**args)
    assert error.value.audit["status"] == "rejected"
    assert error.value.audit["reason"] == "source_date_mismatch"
    assert error.value.audit["anomaly_dates"] == ["2026-01-06"]


@pytest.mark.parametrize("source", ["sina_raw", "tx_raw", "tx_qfq"])
def test_duplicate_session_is_rejected(source):
    args = inputs()
    args[source] = pd.concat([args[source], args[source].iloc[[1]]], ignore_index=True)
    with pytest.raises(SnapshotValidationError, match="duplicate_bar_dates"):
        audit_etf(**args)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_nonpositive_or_nonfinite_ohlc_is_rejected(value):
    args = inputs()
    args["tx_qfq"].loc[1, "close"] = value
    with pytest.raises(SnapshotValidationError, match="invalid_ohlc"):
        audit_etf(**args)


def test_impossible_ohlc_and_missing_amount_column_are_rejected():
    args = inputs()
    args["tx_raw"].loc[1, "high"] = 1
    with pytest.raises(SnapshotValidationError, match="inconsistent_ohlc_bounds"):
        audit_etf(**args)
    args = inputs()
    args["sina_raw"] = args["sina_raw"].drop(columns="amount")
    with pytest.raises(SnapshotValidationError, match="missing_bar_columns"):
        audit_etf(**args)


def test_nullable_missing_ohlc_is_rejected_and_missing_amount_is_retained():
    args = inputs()
    args["sina_raw"]["amount"] = args["sina_raw"].amount.astype("Float64")
    args["sina_raw"].loc[1, "amount"] = pd.NA
    qfq, _, audit = audit_etf(**args)
    assert pd.isna(qfq.amount.iloc[1])
    assert audit["missing_amount_dates"] == ["2026-01-06"]
    args["tx_qfq"]["close"] = args["tx_qfq"].close.astype("Float64")
    args["tx_qfq"].loc[1, "close"] = pd.NA
    with pytest.raises(SnapshotValidationError, match="invalid_ohlc"):
        audit_etf(**args)


@pytest.mark.parametrize("source", ["tx_raw", "tx_qfq"])
def test_half_tick_tolerance_is_fixed(source):
    args = inputs()
    args[source][["open", "high", "low", "close"]] += .0005
    audit_etf(**args)
    args[source][["open", "high", "low", "close"]] += .000001
    with pytest.raises(SnapshotValidationError, match="cross_source_price_mismatch") as error:
        audit_etf(**args)
    assert error.value.audit["source"] == source


@pytest.mark.parametrize("column,value", [("f", 2), ("s", 0), ("s", np.nan), ("u", -1)])
def test_unsupported_factors_are_rejected(column, value):
    args = inputs()
    args["factors"].loc[0, column] = value
    with pytest.raises(SnapshotValidationError, match="unsupported_adjustment_factor"):
        audit_etf(**args)


def test_missing_factors_are_never_treated_as_identity():
    args = inputs()
    args["factors"] = args["factors"].iloc[:0]
    with pytest.raises(SnapshotValidationError, match="missing_adjustment_factors"):
        audit_etf(**args)
    args = inputs()
    args["factors"] = args["factors"].iloc[1:]
    with pytest.raises(SnapshotValidationError, match="factor_does_not_cover_first_bar"):
        audit_etf(**args)


def test_future_bars_are_clipped_but_future_factors_are_rejected():
    args = inputs()
    for source in ("sina_raw", "tx_raw", "tx_qfq"):
        extra = args[source].iloc[[-1]].copy()
        extra["trade_date"] = pd.Timestamp("2026-01-09")
        extra[["open", "high", "low", "close"]] = -1
        args[source] = pd.concat([args[source], extra], ignore_index=True)
    qfq, _, audit = audit_etf(**args)
    assert len(qfq) == audit["rows"] == 4
    args["factors"].loc[2] = [pd.Timestamp("2026-01-09"), 1., 1., 0.]
    with pytest.raises(SnapshotValidationError, match="factor_after_frozen_end"):
        audit_etf(**args)


@pytest.mark.parametrize("column,value", [("s", 2.), ("u", .01)])
def test_qfq_must_be_anchored_at_frozen_end(column, value):
    args = inputs()
    args["factors"].loc[1, column] = value
    with pytest.raises(SnapshotValidationError, match="qfq_not_anchored_at_frozen_end"):
        audit_etf(**args)


def test_sina_amount_volume_units_are_verified_in_raw_price_space():
    args = inputs()
    args["sina_raw"]["volume"] /= 100
    with pytest.raises(SnapshotValidationError, match="amount_volume_units_inconsistent_with_ohlc"):
        audit_etf(**args)
    args = inputs()
    args["sina_raw"].loc[1, "volume"] = 0
    with pytest.raises(SnapshotValidationError, match="positive_amount_with_zero_volume"):
        audit_etf(**args)
    args["sina_raw"].loc[1, "amount"] = 0
    audit_etf(**args)


def test_amount_volume_average_allows_only_fixed_rounding_tolerance():
    args = inputs()
    args["sina_raw"].loc[0, "amount"] = args["sina_raw"].high.iloc[0] * 100 + .5
    _, _, audit = audit_etf(**args)
    assert audit["amount_rounding_cny"] == .5
    args["sina_raw"].loc[0, "amount"] += .00001
    with pytest.raises(SnapshotValidationError, match="amount_volume_units_inconsistent_with_ohlc"):
        audit_etf(**args)


@pytest.mark.parametrize("price,amount", [(1.849, 185.), (4.602, 460.), (2.145, 215.)])
def test_one_lot_trade_allows_whole_yuan_rounding(price, amount):
    args = inputs()
    for source in ("sina_raw", "tx_raw", "tx_qfq"):
        args[source].loc[2, ["open", "high", "low", "close"]] = price
    args["sina_raw"].loc[2, ["volume", "amount"]] = [100., amount]
    _, _, audit = audit_etf(**args)
    assert audit["status"] == "accepted"
    assert audit["amount_rounding_cny"] == .5


@pytest.mark.parametrize("multiplier", [.01, .1, 10., 100.])
def test_amount_rounding_does_not_accept_wrong_money_units(multiplier):
    args = inputs()
    args["sina_raw"]["amount"] *= multiplier
    with pytest.raises(SnapshotValidationError, match="amount_volume_units_inconsistent_with_ohlc"):
        audit_etf(**args)


def test_tencent_volume_comparison_is_informational_only():
    args = inputs()
    args["tx_raw"]["volume"] = args["sina_raw"].volume / 100
    _, _, audit = audit_etf(**args)
    assert audit["volume_comparison"]["median_ratio"] == 1.
    assert audit["volume_comparison"]["max_abs_difference"] == 0.
    args["tx_raw"]["volume"] *= 1.01
    _, _, audit = audit_etf(**args)
    assert audit["volume_comparison"]["median_ratio"] == pytest.approx(1.01)
    assert audit["status"] == "accepted"
    args["code"] = "510300"
    _, _, audit = audit_etf(**args)
    assert audit["volume_comparison"]["market_scope"] == "other_market_not_validated"


def test_common_missing_session_is_preserved_in_audit_and_not_bridged_in_returns():
    args = inputs()
    for source in ("sina_raw", "tx_raw", "tx_qfq"):
        args[source] = args[source].drop(index=1)
    qfq, raw, audit = audit_etf(**args)
    assert audit["availability_protocol"] == "v2"
    assert audit["expected_rows"] == 4 and audit["rows"] == len(raw) == len(qfq) == 3
    assert audit["calendar_missing_dates"] == ["2026-01-06"]
    assert audit["calendar_missing_reason"] == "unknown_unavailable"
    assert audit["calendar_missing"] == [dict(date="2026-01-06", reason="unknown_unavailable")]
    assert qfq.trade_date.tolist() == args["calendar"].delete(1).tolist()
    assert pd.isna(qfq.pct_chg.iloc[1]) and pd.isna(raw.pct_chg.iloc[1])
    assert qfq.pct_chg.iloc[2] == pytest.approx((11 / 10.5 - 1) * 100)


def test_all_missing_sessions_are_rejected():
    args = inputs()
    for source in ("sina_raw", "tx_raw", "tx_qfq"):
        args[source] = args[source].iloc[:0]
    with pytest.raises(SnapshotValidationError, match="no_common_actual_bars"):
        audit_etf(**args)


def test_common_missing_days_do_not_compress_horizons_or_hide_mature_missing_exits():
    dates = pd.bdate_range("2026-01-05", periods=25)
    close = np.arange(25, dtype=float) + 10
    sina = pd.DataFrame(dict(trade_date=dates, open=close, high=close + .2,
                             low=close - .2, close=close, volume=100., amount=100 * close))
    factors = pd.DataFrame(dict(d=[dates[0]], f=[1.], s=[1.], u=[0.]))
    observed = sina.drop(index=[5, 24])
    qfq, _, audit = audit_etf("159919", observed, observed, observed, factors,
                             dates, dates[0], dates[-1])
    assert audit["expected_rows"] == 25 and audit["rows"] == 23
    panel = {field: qfq.pivot(index="trade_date", columns="code", values=field).reindex(dates)
             for field in ("open", "high", "low", "close", "amount")}
    assert len(panel["close"]) == 25
    assert panel["close"].iloc[[5, 24]].isna().all().all()
    cfg = EtfSignalConfig()
    # Signal t=0 must keep its missing holding-path day, never skip to a later exit.
    assert event_outcome(panel, "159919", 0, 10, cfg)["status"] == "missing"
    result = event_outcome(panel, "159919", 6, 10, cfg)
    assert result["status"] == "valid"
    assert result["gross"] == pytest.approx(close[17] / close[7] - 1)
    assert event_outcome(panel, "159919", 4, 10, cfg)["reason"] == "entry_price_or_amount_missing"
    mature = event_outcome(panel, "159919", 13, 10, cfg)
    assert mature["status"] == "missing" and mature["reason"] == "exit_price_amount_or_path_missing"
    assert event_outcome(panel, "159919", 14, 10, cfg)["status"] == "pending"
