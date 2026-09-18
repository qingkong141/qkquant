from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qkquant import etf_portfolio_backtest as module
from qkquant.etf_drawdown import DrawdownConfig, DrawdownState
from qkquant.etf_signal import EtfSignalConfig


def test_reduction_recovery_and_lifetime_halt():
    cfg = DrawdownConfig()
    state = DrawdownState(100, 100)
    assert state.update(95, 1, False, True, cfg)
    assert state.multiplier == .75
    state.update(95, 20, True, True, cfg)
    assert state.multiplier == .75
    state.update(95, 21, True, False, cfg)
    assert state.multiplier == .75
    state.update(95, 22, True, True, cfg)
    assert state.multiplier == 1
    assert state.peak == 100
    assert state.anchor == 95
    state.update(89, 23, False, True, cfg)
    assert state.halted and state.multiplier == 0
    state.update(100, 100, True, True, cfg)
    assert state.halted and state.multiplier == 0


@pytest.mark.parametrize('equity,multiplier,halted', [(96, .75, False), (94, .5, False), (92, 0, False), (90, 0, True)])
def test_exact_thresholds(equity, multiplier, halted):
    state = DrawdownState(100, 100)
    state.update(equity, 1, False, True, DrawdownConfig())
    assert state.multiplier == multiplier
    assert state.halted == halted


@pytest.mark.parametrize('kwargs', [{'max_exposure': 0}, {'reduce_at': .07}, {'recovery_days': 0}, {'max_exposure': float('nan')}])
def test_invalid_risk_config(kwargs):
    with pytest.raises(ValueError):
        DrawdownConfig(**kwargs)


@pytest.fixture
def setup_engine(monkeypatch):
    dates = pd.bdate_range('2024-01-01', periods=190)
    price = 10 * np.exp(np.arange(190) * .003 + np.sin(np.arange(190)) * .001)
    close = pd.DataFrame({'ETF': price}, index=dates)
    panel = {key: close.copy() for key in ('open', 'high', 'low', 'close')}
    panel['amount'] = close * 1e8
    monkeypatch.setattr(module, 'load_panel', lambda *args, **kwargs: panel)
    store = SimpleNamespace(
        load_etf_codes=lambda category: ['ETF'],
        load_instruments=lambda codes: pd.DataFrame([{'code': 'ETF', 'name': 'ETF', 'lot_size': 100}]),
    )
    cfg = EtfSignalConfig(chan_filter_enabled=False, score_threshold=0)
    return panel, store, cfg


def test_daily_crash_reduces_before_scheduled_rebalance(setup_engine):
    panel, store, cfg = setup_engine
    # Buy at 121; crash at 123, next scheduled signal is 130.
    for field in ('open', 'high', 'low', 'close'):
        panel[field].iloc[123:] *= .77
    result = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10, risk_config=DrawdownConfig())
    sells = [t for t in result['trades'] if t['side'] == 'SELL']
    assert sells[0]['date'] == str(panel['close'].index[124].date())
    assert sells[0]['reason'] == 'risk_reduce'
    assert sells[0]['signal_date'] == str(panel['close'].index[123].date())
    assert all(t['date'] > t['signal_date'] for t in result['trades'])


def test_missing_mark_does_not_erase_holdings(setup_engine):
    panel, store, cfg = setup_engine
    normal = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10)
    panel['close'].iloc[125] = np.nan
    actual = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10)
    assert actual['equity'].iloc[5] == pytest.approx(normal['equity'].iloc[4])


def test_proportional_commission_fully_reserved(setup_engine):
    _, store, _ = setup_engine
    cfg = EtfSignalConfig(chan_filter_enabled=False, score_threshold=0, commission_rate=.10, commission_min=.2)
    result = module.run_etf_portfolio_backtest(store, config=cfg)
    assert result['cash'].min() >= 0


def test_zero_turnover_bar_cannot_fill(setup_engine):
    panel, store, cfg = setup_engine
    blocked = panel['close'].index[121]
    panel['amount'].loc[blocked] = 0
    result = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10)
    assert not any(t['date'] == str(blocked.date()) for t in result['trades'])
    assert result['equity'].loc[blocked] == cfg.capital


def test_blocked_risk_sell_retries(setup_engine):
    panel, store, cfg = setup_engine
    for field in ('open', 'high', 'low', 'close'):
        panel[field].iloc[123:] *= .77
    panel['open'].iloc[124] = np.nan
    result = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10, risk_config=DrawdownConfig())
    sells = [t for t in result['trades'] if t['side'] == 'SELL']
    assert sells[0]['date'] == str(panel['close'].index[125].date())


def test_future_price_changes_do_not_change_past(setup_engine):
    panel, store, cfg = setup_engine
    expected = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10, risk_config=DrawdownConfig())
    for field in ('open', 'high', 'low', 'close'):
        panel[field].iloc[151:] *= .7
    actual = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10, risk_config=DrawdownConfig())
    pd.testing.assert_series_equal(expected['equity'].iloc[:31], actual['equity'].iloc[:31])


def test_gap_can_breach_drawdown_target(setup_engine):
    panel, store, cfg = setup_engine
    for field in ('open', 'high', 'low', 'close'):
        panel[field].iloc[123:] *= .5
    result = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10, risk_config=DrawdownConfig())
    assert result['strategy']['max_drawdown'] < -.1
    assert result['risk_history'][-1]['halted']


def test_insufficient_history_has_clear_error(setup_engine):
    panel, store, cfg = setup_engine
    for key in panel:
        panel[key] = panel[key].iloc[:120]
    with pytest.raises(ValueError, match='122'):
        module.run_etf_portfolio_backtest(store, config=cfg)
