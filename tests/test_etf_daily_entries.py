from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qkquant import etf_portfolio_backtest as engine
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_signal import EtfSignalConfig


@pytest.fixture
def market(monkeypatch):
    index = pd.bdate_range('2024-01-01', periods=140)
    values = 10 * np.exp(np.arange(140) * .003)
    close = pd.DataFrame({'A': values, 'B': values * 2}, index=index)
    panel = {field: close.copy() for field in ('open', 'high', 'low', 'close')}
    panel['amount'] = close * 1e8
    monkeypatch.setattr(engine, 'load_panel', lambda *args, **kwargs: panel)
    store = SimpleNamespace(load_etf_codes=lambda category: ['A', 'B'],
        load_instruments=lambda codes: pd.DataFrame([{'code': code, 'name': code, 'lot_size': 100} for code in codes]))
    cfg = EtfSignalConfig(score_threshold=0, corr_limit=1, risk_on_breadth=0)
    events = {'A': 121, 'B': 124}
    def classifier(close, *args):
        fresh = len(close) - 1 == events[close.name]
        return SimpleNamespace(state='third_buy' if fresh else 'third_buy_active', allowed=fresh)
    monkeypatch.setattr(engine, 'classify_chan_structure', classifier)
    return panel, store, cfg


def run(market, daily=True):
    _, store, cfg = market
    return engine.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10,
        daily_entries=daily, risk_config=DrawdownConfig())


def test_daily_entry_catches_off_cycle_signal_next_open(market):
    panel, _, _ = market
    assert not run(market, False)['trades']
    result = run(market)
    first = result['trades'][0]
    assert first['side'] == 'BUY' and first['code'] == 'A'
    assert first['signal_date'] == str(panel['close'].index[121].date())
    assert first['date'] == str(panel['close'].index[122].date())
    assert first['reason'] == 'daily_entry'
    assert result['cash'].min() >= 0


def test_daily_entry_preserves_existing_positions_without_topups(market):
    panel, _, _ = market
    result = run(market)
    before_rebalance = [t for t in result['trades'] if t['date'] < str(panel['close'].index[131].date())]
    assert [(t['code'], t['side']) for t in before_rebalance] == [('A', 'BUY'), ('B', 'BUY')]
    assert result['exposure'].loc[panel['close'].index[125]] < .40


def test_daily_risk_reduction_has_priority_over_new_entry(market):
    panel, _, _ = market
    for field in ('open', 'high', 'low', 'close'):
        panel[field].loc[panel[field].index[124]:, 'A'] *= .3
    result = run(market)
    day = str(panel['close'].index[125].date())
    actions = [t for t in result['trades'] if t['date'] == day]
    assert actions and all(t['side'] == 'SELL' and t['reason'] == 'risk_reduce' for t in actions)


def test_future_data_cannot_change_daily_entry_history(market):
    panel, _, _ = market
    before = run(market)
    for field in ('open', 'high', 'low', 'close'):
        panel[field].iloc[127:] *= .5
    after = run(market)
    pd.testing.assert_series_equal(before['equity'].iloc[:7], after['equity'].iloc[:7])


def test_daily_entries_require_fresh_signal_filter(market):
    _, store, _ = market
    with pytest.raises(ValueError, match='fresh third_buy'):
        engine.run_etf_portfolio_backtest(store, config=EtfSignalConfig(chan_filter_enabled=False), daily_entries=True)


def test_full_slot_cannot_be_replaced_between_rebalances(market):
    panel, store, cfg = market
    result = engine.run_etf_portfolio_backtest(store, config=replace(cfg, max_positions=1),
        rebalance_days=10, daily_entries=True, risk_config=DrawdownConfig())
    assert not any(t['code'] == 'B' and t['date'] < str(panel['close'].index[131].date()) for t in result['trades'])


def test_daily_candidate_is_checked_against_held_correlation(market):
    _, store, cfg = market
    result = engine.run_etf_portfolio_backtest(store, config=replace(cfg, corr_limit=.8),
        rebalance_days=10, daily_entries=True, risk_config=DrawdownConfig())
    assert not any(t['code'] == 'B' and t['reason'] == 'daily_entry' for t in result['trades'])
