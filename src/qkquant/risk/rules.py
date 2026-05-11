"""5 条具体风控规则。

- BlacklistRule          买前否决：代码在黑名单里
- ConcentrationRule      买前否决：若成交后单票市值占比超阈值
- PositionStopLossRule   强平：从入场价下跌超阈值
- TrailingStopRule       强平：从持仓期间最高价回撤超阈值
- PortfolioDrawdownRule  强平：组合净值从峰值回撤超阈值（全平仓 + 冷却期禁买）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qkquant.risk.base import RiskDecision, RiskRule

if TYPE_CHECKING:
    import backtrader as bt


# ---------------- 买前否决类 ----------------


class BlacklistRule(RiskRule):
    """代码级黑名单（ST / 停牌 / 人工拉黑）。"""

    name = "blacklist"

    def __init__(self, codes: list[str]) -> None:
        self.codes: set[str] = set(codes)

    def check_buy(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        if data._name in self.codes:
            return RiskDecision.deny(f"blacklist:{data._name}")
        return RiskDecision.allow()


class ConcentrationRule(RiskRule):
    """单票市值占比上限。

    检查 "成交后的预估仓位市值 / 组合总市值"，若超过 max_weight 则否决。
    """

    name = "concentration"

    def __init__(self, max_weight: float) -> None:
        self.max_weight = max_weight

    def check_buy(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        try:
            price = float(data.close[0])
        except Exception:
            return RiskDecision.allow()
        pos = strat.getposition(data)
        total_value = float(strat.broker.getvalue())
        if total_value <= 0:
            return RiskDecision.allow()
        new_market_value = (pos.size + size) * price
        weight = new_market_value / total_value
        if weight > self.max_weight:
            return RiskDecision.deny(
                f"concentration:{weight:.1%}>{self.max_weight:.0%}"
            )
        return RiskDecision.allow()


# ---------------- 强平类 ----------------


class _EntryTracker(RiskRule):
    """维护 code → 入场价 / 持仓期间最高价 的基础 mixin 风格。

    由 PositionStopLossRule / TrailingStopRule 继承。
    """

    def __init__(self) -> None:
        self._entry_price: dict[str, float] = {}
        self._peak_price: dict[str, float] = {}

    def on_order_filled(self, strat: "bt.Strategy", order: "bt.Order") -> None:
        if order.status != order.Completed:
            return
        code = order.data._name
        price = float(order.executed.price)
        # 买入：新建或补仓 → 用最新成交均价作为"入场价"（简化：不做 VWAP）
        if order.isbuy():
            pos = strat.getposition(order.data)
            if pos.size > 0 and code not in self._entry_price:
                self._entry_price[code] = price
                self._peak_price[code] = price
            elif pos.size > 0:
                # 加仓：加权平均；这里简化为保留原入场价
                pass
        else:  # 卖出且平仓
            pos = strat.getposition(order.data)
            if pos.size <= 0:
                self._entry_price.pop(code, None)
                self._peak_price.pop(code, None)

    def on_bar(self, strat: "bt.Strategy") -> None:
        for data in strat.datas:
            code = data._name
            pos = strat.getposition(data)
            if pos.size <= 0:
                continue
            try:
                close = float(data.close[0])
            except Exception:
                continue
            if code not in self._peak_price:
                # 防御：没追到入场事件时用当前 close 初始化
                self._peak_price[code] = close
                self._entry_price.setdefault(code, close)
            else:
                self._peak_price[code] = max(self._peak_price[code], close)


class PositionStopLossRule(_EntryTracker):
    """单票从入场价跌 threshold 强平。"""

    name = "position_stop"

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forced_exits(self, strat: "bt.Strategy") -> list[tuple[str, str]]:
        exits: list[tuple[str, str]] = []
        for data in strat.datas:
            code = data._name
            pos = strat.getposition(data)
            if pos.size <= 0:
                continue
            entry = self._entry_price.get(code)
            if entry is None or entry <= 0:
                continue
            try:
                close = float(data.close[0])
            except Exception:
                continue
            if close <= entry * (1 - self.threshold):
                exits.append((code, f"position_stop:-{self.threshold:.0%}"))
        return exits


class TrailingStopRule(_EntryTracker):
    """单票从持仓期间最高价回撤 threshold 强平。"""

    name = "trailing_stop"

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forced_exits(self, strat: "bt.Strategy") -> list[tuple[str, str]]:
        exits: list[tuple[str, str]] = []
        for data in strat.datas:
            code = data._name
            pos = strat.getposition(data)
            if pos.size <= 0:
                continue
            peak = self._peak_price.get(code)
            if peak is None or peak <= 0:
                continue
            try:
                close = float(data.close[0])
            except Exception:
                continue
            if close <= peak * (1 - self.threshold):
                exits.append((code, f"trailing_stop:-{self.threshold:.0%}"))
        return exits


class PortfolioDrawdownRule(RiskRule):
    """组合净值从历史峰值回撤超阈值 → 全平仓 + 冷却期内禁买。

    冷却结束前，check_buy 会否决所有买入。
    """

    name = "portfolio_drawdown"

    def __init__(self, threshold: float, cooldown_days: int = 20) -> None:
        self.threshold = threshold
        self.cooldown_days = cooldown_days
        self._peak_value: float = 0.0
        self._cooldown_remaining: int = 0
        self._triggered: bool = False

    def on_bar(self, strat: "bt.Strategy") -> None:
        try:
            value = float(strat.broker.getvalue())
        except Exception:
            return
        self._peak_value = max(self._peak_value, value)
        # 每日减少冷却天数
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        # 检查是否触发
        if self._peak_value > 0:
            drawdown = 1 - value / self._peak_value
            # 容忍极小浮点误差：0.19999999 >= 0.20 应视为触发
            if drawdown + 1e-9 >= self.threshold and not self._triggered:
                self._triggered = True
                self._cooldown_remaining = self.cooldown_days
        # 冷却结束后重置"已触发"，允许下一轮熔断
        if self._triggered and self._cooldown_remaining == 0:
            self._triggered = False

    def forced_exits(self, strat: "bt.Strategy") -> list[tuple[str, str]]:
        if not self._triggered:
            return []
        exits = []
        for data in strat.datas:
            pos = strat.getposition(data)
            if pos.size > 0:
                exits.append((data._name, f"portfolio_drawdown:-{self.threshold:.0%}"))
        return exits

    def check_buy(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        if self._triggered or self._cooldown_remaining > 0:
            return RiskDecision.deny(
                f"portfolio_cooldown:{self._cooldown_remaining}d_remaining"
            )
        return RiskDecision.allow()


__all__ = [
    "BlacklistRule",
    "ConcentrationRule",
    "PortfolioDrawdownRule",
    "PositionStopLossRule",
    "TrailingStopRule",
]
