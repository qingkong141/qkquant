from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qkquant import etf_portfolio_backtest as engine
from qkquant import etf_signal as planner
from qkquant.etf_chan import chan_selection_allowed, classify_chan_structure
from qkquant.etf_signal import EtfSignalConfig


def bars(low36=10.9):
    anchors = {0: 10, 10: 10.4, 16: 10.9, 20: 10.2, 24: 10.8, 28: 10.4,
               32: 11.3, 36: low36, 40: 12.7, 44: 11.5, 48: 12.8, 52: 13.4}
    close = pd.Series(np.interp(np.arange(53), list(anchors), list(anchors.values())),
                      index=pd.bdate_range('2025-01-01', periods=53))
    return close


def classify(close):
    return classify_chan_structure(close, close + .02, close - .02)


def test_pullback_requires_two_future_bars_before_confirmation():
    close = bars()
    assert classify(close.iloc[:45]).state != 'third_buy'
    assert classify(close.iloc[:46]).state != 'third_buy'
    fresh = classify(close.iloc[:47])
    assert fresh.state == 'third_buy' and fresh.allowed
    assert fresh.pullback_date == str(close.index[44])
    assert fresh.confirmation_date == str(close.index[46])
    assert fresh.signal_age_bars == 0


def test_old_structure_cannot_reenter_but_can_retain_holding():
    old = classify(bars().iloc[:49])
    assert old.state == 'third_buy_active'
    assert not old.allowed and old.signal_age_bars == 2
    assert not chan_selection_allowed(old, ('third_buy',))
    assert chan_selection_allowed(old, ('third_buy',), holding=True)
    assert not chan_selection_allowed(old, ('second_buy',), holding=True)


def test_low_before_breakout_is_rejected():
    signal = classify(bars(low36=11.2).iloc[:43])
    assert signal.state == 'third_buy_invalid_order'
    assert not signal.allowed
    assert not chan_selection_allowed(signal, ('third_buy_invalid_order',), holding=True)


def test_missing_current_bar_does_not_reemit_old_confirmation():
    close = bars().iloc[:48].copy()
    close.iloc[-1] = np.nan
    assert classify(close).state == 'missing_latest_bar'
    assert not classify(close).allowed


def test_future_changes_cannot_rewrite_prefix_signal():
    close = bars()
    expected = classify(close.iloc[:47]).to_dict()
    close.iloc[47:] *= .5
    assert classify(close.iloc[:47]).to_dict() == expected


def test_engine_does_not_sell_or_add_only_because_signal_aged(monkeypatch):
    index = pd.bdate_range('2024-01-01', periods=151)
    close = pd.DataFrame({'ETF': 10 * np.exp(np.arange(151) * .003)}, index=index)
    panel = {key: close.copy() for key in ('open', 'high', 'low', 'close')}
    panel['amount'] = close * 1e8
    monkeypatch.setattr(engine, 'load_panel', lambda *args, **kwargs: panel)
    fresh = classify(bars().iloc[:47])
    old = replace(fresh, state='third_buy_active', allowed=False, signal_age_bars=10)
    monkeypatch.setattr(engine, 'classify_chan_structure', lambda c, *args: fresh if len(c) == 121 else old)
    store = SimpleNamespace(load_etf_codes=lambda category: ['ETF'],
        load_instruments=lambda codes: pd.DataFrame([{'code': 'ETF', 'name': 'ETF', 'lot_size': 100}]))
    cfg = EtfSignalConfig(score_threshold=0, slippage_pct=0, commission_rate=0, commission_min=0)
    result = engine.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10)
    assert len(result['trades']) == 1
    assert result['trades'][0]['side'] == 'BUY'
    assert result['exposure'].iloc[-1] > 0
    # Without the first fresh event an old state cannot create a position.
    monkeypatch.setattr(engine, 'classify_chan_structure', lambda *args: old)
    empty = engine.run_etf_portfolio_backtest(store, config=cfg, rebalance_days=10)
    assert not empty['trades']


def test_plan_rejects_old_entry_and_does_not_add_to_old_holding(monkeypatch, tmp_path):
    index = pd.bdate_range('2024-01-01', periods=151)
    close = pd.DataFrame({'159001': 10 * np.exp(np.arange(151) * .003)}, index=index)
    panel = {key: close.copy() for key in ('open', 'high', 'low', 'close')}
    panel['amount'] = close * 1e8
    monkeypatch.setattr(planner, 'load_panel', lambda *args, **kwargs: panel)
    old = replace(classify(bars().iloc[:47]), state='third_buy_active', allowed=False, signal_age_bars=10)
    monkeypatch.setattr(planner, 'classify_chan_structure', lambda *args, **kwargs: old)
    store = SimpleNamespace(load_etf_codes=lambda *args: ['159001'],
        load_instruments=lambda codes: pd.DataFrame([{'code': '159001', 'name': 'ETF', 'lot_size': 100}]))
    cfg = EtfSignalConfig(min_cross_section=1, score_threshold=0)
    assert not planner.build_etf_plan(store, config=cfg)['selected']
    holdings = tmp_path / 'holdings.yaml'
    holdings.write_text('positions:\n  - code: "159001"\n    qty: 100\n', encoding='utf-8')
    plan = planner.build_etf_plan(store, holdings_path=holdings, config=cfg)
    assert len(plan['selected']) == 1
    assert plan['actions'][0]['action'] == 'HOLD'
    assert plan['actions'][0]['target_qty'] == 100
