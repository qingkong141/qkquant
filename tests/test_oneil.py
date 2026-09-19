"""Chronology, financial-vintage and execution regression tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qkquant.oneil import OneilConfig, build_signals, financial_state, prepare_prices, run_backtest
from qkquant.oneil_data import normalize_turnover


def financial_rows():
    rows = []
    for year, eps in zip(range(2021, 2025), (1, 1.4, 1.96, 2.744), strict=True):
        rows.append(dict(code="600001", kind="annual", report_date=f"{year}-12-31",
                         notice_date=f"{year + 1}-03-20", update_date=f"{year + 1}-03-20",
                         eps=eps, revenue=100, roe=20))
    for year, eps, revenue in ((2024, .5, 100), (2025, .8, 150)):
        rows.append(dict(code="600001", kind="quarter", report_date=f"{year}-03-31",
                         notice_date=f"{year}-04-25", update_date=f"{year}-04-25",
                         eps=eps, revenue=revenue, roe=np.nan))
    return pd.DataFrame(rows)


def test_financial_publication_day_and_later_revision():
    rows = financial_rows()
    cfg = OneilConfig()
    assert not financial_state(rows, "2025-04-25", cfg)["fundamental_ok"]
    assert financial_state(rows, "2025-04-26", cfg)["fundamental_ok"]
    rows.loc[rows.report_date == "2024-03-31", "update_date"] = "2025-05-01"
    assert financial_state(rows, "2025-04-26", cfg)["reason"] == "revision_not_available"
    assert financial_state(rows, "2025-04-26", cfg, "notice")["fundamental_ok"]
    assert financial_state(rows, "2025-05-02", cfg)["fundamental_ok"]


def test_missing_annual_year_and_negative_base_fail_closed():
    rows = financial_rows()
    rows.loc[rows.report_date == "2021-12-31", "report_date"] = "2020-12-31"
    assert financial_state(rows, "2025-04-26", OneilConfig())["reason"] == "missing_comparison_period"
    rows = financial_rows()
    rows.loc[rows.report_date == "2024-03-31", "eps"] = -.1
    assert not financial_state(rows, "2025-04-26", OneilConfig())["fundamental_ok"]


def test_new_announced_report_does_not_fall_back_to_old_passing_report():
    rows = financial_rows()
    rows.loc[len(rows)] = ["600001", "quarter", "2025-06-30", "2025-08-20", "2026-08-20", .9, 160, np.nan]
    # No prior-year Q2 means the new report cannot be evaluated; Q1 must not be reused.
    assert not financial_state(rows, "2025-08-21", OneilConfig())["fundamental_ok"]


def test_flat_base_uses_prior_bars_and_is_prefix_invariant():
    close = np.r_[np.geomspace(5, 20, 160), np.full(25, 20), 20.8, 50]
    days = pd.bdate_range("2023-01-02", periods=len(close))
    volume = np.full(len(close), 10_000_000.)
    volume[-2] *= 2
    bars = pd.DataFrame(dict(code="600001", trade_date=days, open=close, close=close,
                             high=close * 1.005, low=close * .995, volume=volume, amount=volume * close,
                             raw_open=close, raw_close=close, raw_high=close * 1.005, raw_low=close * .995))
    benchmark = pd.DataFrame(dict(trade_date=days, raw_close=np.linspace(100, 200, len(close))))
    cfg = replace(OneilConfig(), rs_days=60)
    full = prepare_prices(bars, benchmark, cfg)
    prefix = prepare_prices(bars.iloc[:-1], benchmark.iloc[:-1], cfg)
    assert full["technical"].iloc[-2, 0]
    assert full["pivot"].iloc[-2, 0] == pytest.approx(20.1)
    for key in ("technical", "pivot", "rs", "avg_volume"):
        pd.testing.assert_frame_equal(full[key].iloc[:-1], prefix[key])


def prepared(prices, lows=None):
    days = pd.bdate_range("2025-05-01", periods=len(prices))
    def frame(values):
        return pd.DataFrame({"600001": values}, index=days, dtype=float)
    close = frame(prices)
    low = frame(lows if lows is not None else np.asarray(prices) * .999)
    return {"open": close.copy(), "close": close, "high": close * 1.001, "low": low,
            "raw_open": close.copy(), "raw_close": close.copy(), "raw_high": close * 1.001, "raw_low": low.copy(),
            "raw_previous": close.ffill().shift(1), "valuation": close.ffill(), "volume": frame([100] * len(days)),
            "avg_volume": frame([100] * len(days)), "ma50": frame([8] * len(days))}


def signals(p):
    return pd.DataFrame([dict(signal_date=p["close"].index[0], code="600001", pivot=9.99,
                              rs=1, qualified=True)])


def simulate(p, cfg=None):
    cfg = cfg or replace(OneilConfig(), slippage=0, commission=0, minimum_commission=0)
    return run_backtest(p, signals(p), cfg, start=str(p["close"].index[0].date()))


def test_entry_next_session_and_buy_day_stop_cannot_sell_same_day():
    p = prepared([10, 10, 9.1, 9.5], [9.9, 9, 9, 9.4])
    result = simulate(p)
    buy, sell = result["trades"].iloc[0], result["trades"].iloc[1]
    assert buy.date == p["close"].index[1]
    assert sell.date == p["close"].index[2]
    assert sell.price_adjusted == pytest.approx(9.1)
    assert sell.reason == "stop_loss"
    assert "t_plus_one_stop_deferred" in result["rejections"].reason.tolist()
    assert result["equity"].equity.iloc[0] == 100_000


def test_stop_gap_and_blocked_sale_remain_pending():
    p = prepared([10, 10, 9, 8.1, 8.2], [9.9, 9.9, 8.9, 8, 8.1])
    result = simulate(p)
    sells = result["trades"].query("side == 'SELL'")
    assert len(sells) == 1
    assert sells.iloc[0].date == p["close"].index[4]
    assert sells.iloc[0].price_adjusted == pytest.approx(8.2)
    assert sells.iloc[0].reason == "stop_loss"


def test_eight_week_hold_does_not_override_protective_stop():
    p = prepared([10, 10, 11, 12.1, 11, 10, 9.5], [9.9, 9.9, 10.9, 12, 10.9, 9.9, 9.1])
    result = simulate(p)
    sells = result["trades"].query("side == 'SELL'")
    assert len(sells) == 1
    assert sells.iloc[0].eight_week_hold
    assert sells.iloc[0].reason == "stop_loss"
    assert sells.iloc[0].price_adjusted == pytest.approx(9.2)


def test_ordinary_profit_sells_after_close_confirmation():
    p = prepared([10, 10, 11, 12.1, 12.2])
    cfg = replace(OneilConfig(), slippage=0, commission=0, minimum_commission=0, fast_rise_days=1)
    result = simulate(p, cfg)
    sell = result["trades"].query("side == 'SELL'").iloc[0]
    assert sell.date == p["close"].index[4]
    assert sell.reason == "take_profit"


def test_missing_bar_carries_valuation_without_fabricated_sale():
    p = prepared([10, 10, np.nan, 10])
    result = simulate(p)
    curve = result["equity"]
    assert curve.equity.iloc[2] == pytest.approx(curve.equity.iloc[1])
    assert result["metrics"]["open_positions"] == 1
    assert (curve.cash >= 0).all()


def test_gap_outside_buy_zone_cancels_instead_of_chasing():
    p = prepared([10, 10.6, 11])
    result = simulate(p)
    assert result["trades"].empty
    assert result["rejections"].reason.tolist() == ["outside_buy_zone"]


def test_unknown_financial_policy_is_rejected():
    with pytest.raises(ValueError, match="financial policy"):
        financial_state(financial_rows(), "2025-04-26", OneilConfig(), "latest")


def test_mixed_lots_and_shares_cannot_create_a_hundredfold_volume_spike():
    bars = pd.DataFrame({"volume": [100, 10000, 100], "raw_volume": [100, 100, 100],
                         "amount": [100000, 100000, 1000], "raw_low": [9, 9, 9], "raw_high": [11, 11, 11]})
    normalized, bad = normalize_turnover(bars)
    assert normalized.volume.iloc[:2].tolist() == [10000, 10000]
    assert normalized.volume.iloc[2:].isna().all()
    assert bad == 1
    assert bars.volume.tolist() == [100, 10000, 100]


def test_complete_growth_breakout_pipeline_can_trade_with_available_reports():
    close = np.r_[np.geomspace(5, 20, 160), np.full(25, 20), 20.8, 20.85]
    days = pd.bdate_range(end="2025-04-29", periods=len(close))
    volume = np.full(len(close), 10_000_000.)
    volume[-2] *= 2
    bars = pd.DataFrame(dict(code="600001", trade_date=days, open=close, close=close,
                             high=close * 1.005, low=close * .995, volume=volume, amount=volume * close,
                             raw_open=close, raw_close=close, raw_high=close * 1.005, raw_low=close * .995))
    index = pd.DataFrame(dict(trade_date=days, raw_close=np.linspace(100, 200, len(close))))
    cfg = replace(OneilConfig(), rs_days=60)
    panel = prepare_prices(bars, index, cfg)
    events = build_signals(panel, financial_rows(), cfg)
    assert events.qualified.sum() == 1
    result = run_backtest(panel, events, cfg, start="2025-04-24")
    assert result["metrics"]["buy_count"] == 1
    assert result["trades"].iloc[0].date == pd.Timestamp("2025-04-29")
