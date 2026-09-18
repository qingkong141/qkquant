from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qkquant import etf_portfolio_backtest as engine
from qkquant.etf_drawdown import DrawdownConfig, DrawdownState
from qkquant.etf_signal import EtfSignalConfig


@pytest.fixture
def market():
    dates = pd.bdate_range("2024-01-01", periods=150)
    values = 10 * np.exp(np.arange(150) * .003 + np.sin(np.arange(150)) * .001)
    close = pd.DataFrame({"ETF": values, "QUARANTINED": np.nan}, index=dates)
    panel = {key: close.copy() for key in ("open", "high", "low", "close")}
    panel["amount"] = close * 1e8
    instruments = pd.DataFrame([{"code": "ETF", "name": "ETF", "lot_size": 100}])
    store = SimpleNamespace(load_instruments=lambda codes: instruments)
    cfg = EtfSignalConfig(chan_filter_enabled=False, score_threshold=0)
    return panel, store, cfg, instruments.set_index("code")


def test_supplied_panel_preserves_missing_session_and_quarantined_column(market, monkeypatch):
    panel, store, cfg, _ = market
    missing_day = panel["close"].index[125]
    for value in panel.values():
        value.loc[missing_day] = np.nan
    monkeypatch.setattr(engine, "load_panel", lambda *args, **kwargs: pytest.fail("must use supplied panel"))
    prepared = engine.prepare_inputs(panel)
    assert list(prepared["close"].columns) == ["ETF", "QUARANTINED"]
    assert prepared["close"].loc[missing_day].isna().all()
    assert prepared["ret"].iloc[126].isna().all()
    result = engine.run_etf_portfolio_backtest(store, panel=panel, config=cfg, rebalance_days=10)
    pd.testing.assert_index_equal(result["equity"].index, panel["close"].index[120:])
    assert result["equity"].loc[missing_day] == pytest.approx(result["equity"].iloc[4])
    assert not any(t["date"] == str(missing_day.date()) for t in result["trades"])
    assert result["final_state"]["position"] == 149


def test_last_close_decision_is_saved_but_only_executes_on_following_session(market):
    panel, store, cfg, _ = market
    panel["open"].iloc[131, 0] *= 1.10
    short_panel = {key: value.iloc[:131] for key, value in panel.items()}
    short = engine.run_etf_portfolio_backtest(store, panel=short_panel, config=cfg, rebalance_days=10)
    pending = short["final_state"]["pending"]
    final_date = str(panel["close"].index[130].date())
    assert pending is not None and pending[3] == final_date
    assert short["last_decision"]["scheduled"]
    assert not any(t["signal_date"] == final_date for t in short["trades"])
    longer = engine.run_etf_portfolio_backtest(
        store, panel={key: value.iloc[:132] for key, value in panel.items()}, config=cfg, rebalance_days=10,
    )
    following = [t for t in longer["trades"] if t["signal_date"] == final_date]
    assert following and all(t["date"] == str(panel["close"].index[131].date()) for t in following)
    pd.testing.assert_series_equal(short["equity"], longer["equity"].iloc[:-1])
    assert "not actual shares" in short["simulation_basis"]


def test_recorded_account_close_uses_identical_decision_and_risk_state(market):
    panel, store, cfg, instruments = market
    risk = DrawdownConfig()
    earlier = engine.run_etf_portfolio_backtest(
        store, panel={key: value.iloc[:130] for key, value in panel.items()}, config=cfg,
        rebalance_days=10, risk_config=risk,
    )["final_state"]
    state = DrawdownState(**earlier["drawdown"])
    invested = sum(qty * panel["close"].iloc[130][code] for code, qty in earlier["holdings"].items())
    daily = engine.close_decision(
        engine.prepare_inputs(panel), instruments, 130, earlier["holdings"],
        earlier["cash"] + invested, invested, state, cfg, risk,
    )
    backtest = engine.run_etf_portfolio_backtest(
        store, panel={key: value.iloc[:131] for key, value in panel.items()}, config=cfg,
        rebalance_days=10, risk_config=risk,
    )
    assert daily == backtest["last_decision"]
    assert asdict(state) == backtest["final_state"]["drawdown"]
    assert daily["exposure"] <= .40


def test_scheduled_rebalance_cannot_override_risk_only_sale(market):
    panel, store, cfg, _ = market
    for key in ("open", "high", "low", "close"):
        panel[key].iloc[130:, 0] *= .85
    result = engine.run_etf_portfolio_backtest(
        store, panel=panel, config=cfg, risk_config=DrawdownConfig(), rebalance_days=10,
    )
    actions = [t for t in result["trades"] if t["signal_date"] == str(panel["close"].index[130].date())]
    assert actions
    assert all(t["side"] == "SELL" and t["reason"] == "risk_reduce" for t in actions)


def test_non_scheduled_day_explicitly_reports_no_action(market):
    panel, _, cfg, instruments = market
    decision = engine.close_decision(
        engine.prepare_inputs(panel), instruments, 121, {}, cfg.capital, 0,
        DrawdownState(cfg.capital, cfg.capital), cfg, DrawdownConfig(),
    )
    assert not decision["scheduled"]
    assert decision["pending"] is None and decision["action"] == "no_action"
    assert decision["regime"] == "risk_on" and decision["exposure"] == .40


def test_prepare_rejects_misaligned_field_instead_of_compressing_sessions(market):
    panel, _, _, _ = market
    panel["open"] = panel["open"].iloc[1:]
    with pytest.raises(ValueError, match="match close dates and codes"):
        engine.prepare_inputs(panel)
