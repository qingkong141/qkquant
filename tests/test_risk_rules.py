"""风控规则单元测试。

为了不拉起 backtrader，手写 FakeStrategy / FakeData / FakeOrder 提供规则所需的最小接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from qkquant.risk.rules import (
    BlacklistRule,
    ConcentrationRule,
    PortfolioDrawdownRule,
    PositionStopLossRule,
    TrailingStopRule,
)


# ---------------- fakes ----------------


class FakeCloseLine:
    """模拟 backtrader 的 data.close，支持 close[0]、close[-n]。"""

    def __init__(self, series: list[float]) -> None:
        self._series = series

    def __getitem__(self, idx: int) -> float:
        # backtrader 语义：close[0] 是当前，close[-1] 是上一根
        return self._series[len(self._series) - 1 + idx]


@dataclass
class FakeData:
    _name: str
    closes: list[float]

    @property
    def close(self) -> FakeCloseLine:
        return FakeCloseLine(self.closes)


@dataclass
class FakePosition:
    size: int = 0


@dataclass
class FakeStrategy:
    datas: list[FakeData]
    positions: dict[str, FakePosition] = field(default_factory=dict)
    portfolio_value: float = 1_000_000.0

    def getposition(self, data: FakeData) -> FakePosition:
        return self.positions.setdefault(data._name, FakePosition())

    @property
    def broker(self):
        return SimpleNamespace(getvalue=lambda: self.portfolio_value)


def _make_order(code: str, price: float, is_buy: bool, completed: bool = True):
    """伪造一个 backtrader 风格的 order 对象。"""
    # order.status 和 order.Completed 需要相等才算成交
    Completed = 4
    order = SimpleNamespace(
        data=SimpleNamespace(_name=code),
        status=Completed if completed else 1,
        Completed=Completed,
        executed=SimpleNamespace(price=price),
        isbuy=(lambda: is_buy),
    )
    return order


# ---------------- BlacklistRule ----------------


def test_blacklist_denies_listed_code():
    rule = BlacklistRule(codes=["000001.SZ"])
    data = FakeData("000001.SZ", [10.0])
    strat = FakeStrategy(datas=[data])

    dec = rule.check_buy(strat, data, size=100)
    assert not dec.allowed
    assert "blacklist" in dec.reason


def test_blacklist_allows_unlisted_code():
    rule = BlacklistRule(codes=["000001.SZ"])
    data = FakeData("600000.SH", [10.0])
    strat = FakeStrategy(datas=[data])
    assert rule.check_buy(strat, data, size=100).allowed


# ---------------- ConcentrationRule ----------------


def test_concentration_denies_when_new_weight_exceeds():
    rule = ConcentrationRule(max_weight=0.15)
    data = FakeData("x", [10.0])
    # 买 20000 股 * 10 = 200000，占 1_000_000 的 20% > 15%
    strat = FakeStrategy(datas=[data], portfolio_value=1_000_000.0)
    dec = rule.check_buy(strat, data, size=20_000)
    assert not dec.allowed
    assert "concentration" in dec.reason


def test_concentration_allows_when_under():
    rule = ConcentrationRule(max_weight=0.15)
    data = FakeData("x", [10.0])
    # 买 10000 股 * 10 = 100000，占 10%
    strat = FakeStrategy(datas=[data], portfolio_value=1_000_000.0)
    assert rule.check_buy(strat, data, size=10_000).allowed


# ---------------- PositionStopLossRule ----------------


def test_position_stop_triggers_after_drop():
    rule = PositionStopLossRule(threshold=0.10)
    data = FakeData("x", [10.0])
    strat = FakeStrategy(datas=[data])

    # 买入 @ 10，持仓 100 股
    order = _make_order("x", price=10.0, is_buy=True)
    strat.positions["x"] = FakePosition(size=100)
    rule.on_order_filled(strat, order)

    # 当前价 9.5，跌 5%，不触发
    data.closes = [9.5]
    rule.on_bar(strat)
    assert rule.forced_exits(strat) == []

    # 当前价 8.9，跌 11%，触发
    data.closes = [8.9]
    rule.on_bar(strat)
    exits = rule.forced_exits(strat)
    assert len(exits) == 1 and exits[0][0] == "x"


def test_position_stop_clears_after_exit():
    rule = PositionStopLossRule(threshold=0.10)
    data = FakeData("x", [10.0])
    strat = FakeStrategy(datas=[data])
    strat.positions["x"] = FakePosition(size=100)

    buy = _make_order("x", price=10.0, is_buy=True)
    rule.on_order_filled(strat, buy)

    # 平仓
    strat.positions["x"] = FakePosition(size=0)
    sell = _make_order("x", price=8.0, is_buy=False)
    rule.on_order_filled(strat, sell)

    assert "x" not in rule._entry_price


# ---------------- TrailingStopRule ----------------


def test_trailing_stop_tracks_peak():
    rule = TrailingStopRule(threshold=0.12)
    data = FakeData("x", [10.0])
    strat = FakeStrategy(datas=[data])

    # 买入 @ 10
    strat.positions["x"] = FakePosition(size=100)
    rule.on_order_filled(strat, _make_order("x", 10.0, is_buy=True))

    # 涨到 15
    data.closes = [15.0]
    rule.on_bar(strat)
    assert rule._peak_price["x"] == 15.0

    # 回到 13.5，从峰值回撤 10%，不触发
    data.closes = [13.5]
    rule.on_bar(strat)
    assert rule.forced_exits(strat) == []
    # 峰值应保持 15
    assert rule._peak_price["x"] == 15.0

    # 回撤到 13（-13.3% from 15），触发
    data.closes = [13.0]
    rule.on_bar(strat)
    exits = rule.forced_exits(strat)
    assert exits and exits[0][0] == "x"


# ---------------- PortfolioDrawdownRule ----------------


def test_portfolio_drawdown_triggers_full_liquidation():
    rule = PortfolioDrawdownRule(threshold=0.20, cooldown_days=3)
    data_a = FakeData("a", [10.0])
    data_b = FakeData("b", [20.0])
    strat = FakeStrategy(datas=[data_a, data_b], portfolio_value=1_000_000.0)

    # 持有 a 和 b
    strat.positions["a"] = FakePosition(size=100)
    strat.positions["b"] = FakePosition(size=100)

    # 推高到 1_200_000
    strat.portfolio_value = 1_200_000
    rule.on_bar(strat)
    assert rule._peak_value == 1_200_000

    # 跌到 950_000，回撤 ~20.8%，触发
    strat.portfolio_value = 950_000
    rule.on_bar(strat)
    exits = rule.forced_exits(strat)
    codes = {c for c, _ in exits}
    assert codes == {"a", "b"}

    # 冷却期内禁买
    dec = rule.check_buy(strat, data_a, size=100)
    assert not dec.allowed


def test_portfolio_drawdown_cooldown_countdown():
    rule = PortfolioDrawdownRule(threshold=0.20, cooldown_days=2)
    data = FakeData("a", [10.0])
    strat = FakeStrategy(datas=[data], portfolio_value=1_000_000.0)
    strat.positions["a"] = FakePosition(size=100)

    # 达到峰值再回撤
    rule.on_bar(strat)  # peak = 1_000_000
    strat.portfolio_value = 750_000   # -25%，远超 20% 阈值
    rule.on_bar(strat)  # 触发，cooldown = 2
    assert rule._triggered is True

    # 再过 1 天（cooldown 从 2 -> 1）
    rule.on_bar(strat)
    # 再过 1 天（cooldown 从 1 -> 0，且 triggered 清除）
    rule.on_bar(strat)
    assert rule._triggered is False
    # 冷却结束后，买入应该再允许
    assert rule.check_buy(strat, data, size=100).allowed
