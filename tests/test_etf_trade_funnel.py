import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_portfolio_backtest import run_etf_portfolio_backtest
from qkquant.etf_signal import EtfSignalConfig


spec = importlib.util.spec_from_file_location(
    "trade_funnel", Path(__file__).parents[1] / "scripts" / "diagnose_etf_trade_funnel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_score_ranks_in_original_prefilter_pool():
    # X has huge factors but fails liquidity, so it must not alter A/B ranks.
    values = dict(mom60=[.1, .1, 10], mom120=[.1, .2, 10], risk_mom=[1, 2, 100],
                  trend=[1, 2, 100], downside=[.1, .2, .001],
                  amount20=[1e8, 1e8, 1], close=[10, 10, 10], ma120=[9, 9, 9])
    prepared = {key: pd.DataFrame([row], columns=["A", "B", "X"]) for key, row in values.items()}
    frame = module.factor_gate(prepared, 0, EtfSignalConfig())
    assert frame.loc["A", "score"] == pytest.approx(.50)
    assert frame.loc["B", "score"] == pytest.approx(.80)
    assert not frame.loc["A", "score_ok"]
    assert pd.isna(frame.loc["X", "score"])


def test_funnel_conditions_overlap_and_are_counted_once():
    events = pd.DataFrame([{key: True for key, _ in module.STAGES} for _ in range(3)])
    events["date"] = ["2025-01-01", "2025-01-01", "2025-01-02"]
    events["code"] = ["A", "B", "A"]
    events.loc[0, ["score_ok", "scheduled", "opened"]] = False
    events.loc[1, ["scheduled", "opened"]] = False
    funnel = module.summarize_funnel(events).set_index("stage")
    assert funnel.loc["fresh", "events"] == 3
    assert funnel.loc["fresh", "dates"] == 2
    assert funnel.loc["score_ok", "removed_at_step"] == 1
    assert funnel.loc["scheduled", "removed_at_step"] == 1
    assert funnel.loc["opened", "events"] == 1


@pytest.fixture
def market(monkeypatch):
    from qkquant import etf_portfolio_backtest as engine
    index = pd.bdate_range("2024-01-01", periods=140)
    values = 10 * np.exp(np.arange(140) * .003 + np.sin(np.arange(140)) * .001)
    close = pd.DataFrame({"ETF": values}, index=index)
    panel = {field: close.copy() for field in ("open", "high", "low", "close")}
    panel["amount"] = close * 1e8
    instruments = pd.DataFrame([{"code": "ETF", "name": "ETF", "lot_size": 100}])
    store = SimpleNamespace(load_instruments=lambda codes: instruments)
    fresh_positions = {121, 130, 139}

    def classify(series, *args):
        fresh = len(series) - 1 in fresh_positions
        return SimpleNamespace(state="third_buy" if fresh else "third_buy_active", allowed=fresh,
            previous_high_date=None, breakout_high_date=None, pullback_date=None,
            confirmation_date=str(series.index[-1]), signal_age_bars=0 if fresh else 1,
            distance_to_pullback=.01)

    monkeypatch.setattr(engine, "classify_chan_structure", classify)
    monkeypatch.setattr(module, "classify_chan_structure", classify)
    result = run_etf_portfolio_backtest(store, panel=panel, config=EtfSignalConfig(),
                                      risk_config=DrawdownConfig(), rebalance_days=10)
    curves = pd.DataFrame({key: result[key] for key in ("equity", "cash", "exposure")})
    trades = pd.DataFrame(result["trades"])
    data = SimpleNamespace(panel=panel, store=store, codes=["ETF"], metadata={"excluded_codes": []})
    return data, curves, trades


def test_off_schedule_signal_is_not_an_actual_trade_and_last_open_is_pending(market):
    data, curves, trades = market
    events, days, _ = module.collect_funnel(data, curves, trades)
    assert events.position.tolist() == [121, 130, 139]
    assert events.daily_eligible.all()
    assert events.scheduled.tolist() == [False, True, False]
    assert events.opened.tolist() == [False, True, False]
    assert events.iloc[-1].next_open_status == "not_observed"
    assert len(days) == 20


def test_reference_equity_must_reconcile_not_silently_use_wrong_account(market):
    data, curves, trades = market
    curves.iloc[5, curves.columns.get_loc("cash")] += 10
    with pytest.raises(ValueError, match="reconcile reference equity"):
        module.collect_funnel(data, curves, trades)


def test_future_prices_do_not_change_recorded_signal_prefix(market):
    data, curves, trades = market
    before, _, _ = module.collect_funnel(data, curves, trades)
    # Only the unopened ending prices change; keep the pre-change comparison.
    short = SimpleNamespace(panel={key: value.iloc[:133] for key, value in data.panel.items()},
                            store=data.store, codes=data.codes, metadata=data.metadata)
    after, _, _ = module.collect_funnel(short, curves.iloc[:13], trades[trades.date <= str(data.panel["close"].index[132].date())])
    pd.testing.assert_frame_equal(before[before.position <= 132].reset_index(drop=True), after.reset_index(drop=True))


def test_empty_slice_still_reports_zero_without_invented_events():
    events = pd.DataFrame(columns=[key for key, _ in module.STAGES] + ["date", "code"])
    for key, _ in module.STAGES:
        events[key] = events[key].astype(bool)
    assert module.summarize_funnel(events).events.sum() == 0
