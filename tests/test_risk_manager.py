"""RiskManager / RiskConfig 测试。"""

from __future__ import annotations

from types import SimpleNamespace

from qkquant.risk import RiskConfig, RiskManager


def test_empty_config_no_rules():
    mgr = RiskManager(RiskConfig())
    assert not mgr.enabled
    assert mgr.rules == []


def test_config_builds_expected_rules():
    cfg = RiskConfig(
        blacklist={"enabled": True, "codes": ["x"]},
        trailing_stop={"enabled": True, "threshold": 0.1},
        portfolio_drawdown={"enabled": True, "threshold": 0.2},
    )
    mgr = RiskManager(cfg)
    names = [r.name for r in mgr.rules]
    assert "blacklist" in names
    assert "trailing_stop" in names
    assert "portfolio_drawdown" in names
    assert "position_stop" not in names


def test_config_any_enabled_flag():
    assert not RiskConfig().any_enabled()
    assert RiskConfig(blacklist={"enabled": True}).any_enabled()


def test_check_buy_short_circuits_on_first_deny():
    """若多条规则命中，返回第一条的原因。"""
    cfg = RiskConfig(
        blacklist={"enabled": True, "codes": ["x"]},
        concentration={"enabled": True, "max_weight": 0.01},
    )
    mgr = RiskManager(cfg)

    # 造一个 FakeStrategy
    data = SimpleNamespace(_name="x", close=SimpleNamespace(__getitem__=lambda _self, _i: 10.0))
    strat = SimpleNamespace(
        getposition=lambda _d: SimpleNamespace(size=0),
        broker=SimpleNamespace(getvalue=lambda: 1_000_000),
    )
    dec = mgr.check_buy(strat, data, size=100)
    assert not dec.allowed
    # 应该是第一条 (blacklist) 的原因
    assert "blacklist" in dec.reason


def test_forced_exits_dedupes_across_rules():
    """若多条规则都标记同一只股票强平，只保留第一个。"""
    cfg = RiskConfig(
        position_stop={"enabled": True, "threshold": 0.05},
        trailing_stop={"enabled": True, "threshold": 0.05},
    )
    mgr = RiskManager(cfg)

    # 手动设状态：position_stop 和 trailing_stop 都会判 x 该平仓
    from qkquant.risk.rules import PositionStopLossRule, TrailingStopRule

    ps = next(r for r in mgr.rules if isinstance(r, PositionStopLossRule))
    ts = next(r for r in mgr.rules if isinstance(r, TrailingStopRule))
    ps._entry_price["x"] = 10.0
    ts._peak_price["x"] = 10.0

    class FakeClose:
        def __getitem__(self, _i):
            return 8.0  # 比入场 / 峰值跌 20%

    data = SimpleNamespace(_name="x", close=FakeClose())
    strat = SimpleNamespace(
        datas=[data],
        getposition=lambda _d: SimpleNamespace(size=100),
    )

    exits = mgr.forced_exits(strat)
    assert len(exits) == 1  # 去重
    assert exits[0][0] == "x"
