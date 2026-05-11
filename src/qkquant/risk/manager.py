"""RiskManager：根据 RiskConfig 组装规则，提供统一的检查/强平/状态更新接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qkquant.risk.base import RiskDecision, RiskRule
from qkquant.risk.config import RiskConfig
from qkquant.risk.rules import (
    BlacklistRule,
    ConcentrationRule,
    PortfolioDrawdownRule,
    PositionStopLossRule,
    TrailingStopRule,
)

if TYPE_CHECKING:
    import backtrader as bt


class RiskManager:
    """风控规则聚合器。"""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.rules: list[RiskRule] = self._build_rules(self.config)

    @staticmethod
    def _build_rules(cfg: RiskConfig) -> list[RiskRule]:
        rules: list[RiskRule] = []
        if cfg.blacklist.enabled:
            rules.append(BlacklistRule(cfg.blacklist.codes))
        if cfg.concentration.enabled:
            rules.append(ConcentrationRule(cfg.concentration.max_weight))
        if cfg.position_stop.enabled:
            rules.append(PositionStopLossRule(cfg.position_stop.threshold))
        if cfg.trailing_stop.enabled:
            rules.append(TrailingStopRule(cfg.trailing_stop.threshold))
        if cfg.portfolio_drawdown.enabled:
            rules.append(
                PortfolioDrawdownRule(
                    threshold=cfg.portfolio_drawdown.threshold,
                    cooldown_days=cfg.portfolio_drawdown.cooldown_days,
                )
            )
        return rules

    # ---- 聚合 hook ----

    def check_buy(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        for rule in self.rules:
            d = rule.check_buy(strat, data, size)
            if not d.allowed:
                return d
        return RiskDecision.allow()

    def check_sell(self, strat: "bt.Strategy", data, size: int) -> RiskDecision:
        for rule in self.rules:
            d = rule.check_sell(strat, data, size)
            if not d.allowed:
                return d
        return RiskDecision.allow()

    def forced_exits(self, strat: "bt.Strategy") -> list[tuple[str, str]]:
        """返回 (code, reason) 列表。同一代码被多条规则命中时取第一条。"""
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for rule in self.rules:
            for code, reason in rule.forced_exits(strat):
                if code in seen:
                    continue
                seen.add(code)
                out.append((code, reason))
        return out

    def on_order_filled(self, strat: "bt.Strategy", order: "bt.Order") -> None:
        for rule in self.rules:
            rule.on_order_filled(strat, order)

    def on_bar(self, strat: "bt.Strategy") -> None:
        for rule in self.rules:
            rule.on_bar(strat)

    @property
    def enabled(self) -> bool:
        return len(self.rules) > 0


__all__ = ["RiskManager"]
