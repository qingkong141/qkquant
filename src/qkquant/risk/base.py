"""风控规则抽象与决策对象。

设计要点：
- RiskRule 有三类 hook：check_buy / check_sell / forced_exits
- 规则允许持有自己的状态（入场价、持仓峰值、组合峰值等），状态通过 on_order_filled
  / on_bar 更新
- RiskDecision 统一"允许/否决 + 原因"返回值，方便 rejections 日志记录
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import backtrader as bt


@dataclass
class RiskDecision:
    """风控检查结果。"""

    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> "RiskDecision":
        return cls(allowed=True, reason="")

    @classmethod
    def deny(cls, reason: str) -> "RiskDecision":
        return cls(allowed=False, reason=reason)


class RiskRule:
    """风控规则基类。子类只需覆盖关心的 hook。

    注意：rule 可以持有状态（如入场价、峰值等），每次回测会 new 一个 RiskManager，
    其中的 rule 也都是新实例，所以状态不会跨回测污染。
    """

    name: str = "base"

    # ---- 下单前检查 ----

    def check_buy(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        return RiskDecision.allow()

    def check_sell(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        return RiskDecision.allow()

    # ---- 每日开盘前强平扫描 ----

    def forced_exits(self, strat: "bt.Strategy") -> list[tuple[str, str]]:
        """返回 (code, reason) 列表，当日应强制卖出的持仓。"""
        return []

    # ---- 状态更新 hook ----

    def on_order_filled(self, strat: "bt.Strategy", order: "bt.Order") -> None:
        """订单成交后被调用，用于更新入场价/持仓峰值等状态。"""
        pass

    def on_bar(self, strat: "bt.Strategy") -> None:
        """每根 K 线开始时调用，用于更新组合峰值等随时间变化的状态。"""
        pass


__all__ = ["RiskDecision", "RiskRule"]
