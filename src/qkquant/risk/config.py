"""风控配置（pydantic）。各策略 yaml 的 `risk:` 段映射到这里。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BlacklistConfig(BaseModel):
    enabled: bool = False
    codes: list[str] = Field(default_factory=list)


class PositionStopLossConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.10  # 从入场价跌 10%


class TrailingStopConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.12  # 从持仓期间最高点回撤 12%


class ConcentrationConfig(BaseModel):
    enabled: bool = False
    max_weight: float = 0.15  # 单票最大市值占比 15%


class PortfolioDrawdownConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.20  # 组合从峰值回撤 20% 触发全平熔断
    cooldown_days: int = 20  # 熔断触发后多少个交易日禁止买入


class RiskConfig(BaseModel):
    """风控总配置。每条规则默认 disabled，按需打开。"""

    blacklist: BlacklistConfig = Field(default_factory=BlacklistConfig)
    position_stop: PositionStopLossConfig = Field(default_factory=PositionStopLossConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    concentration: ConcentrationConfig = Field(default_factory=ConcentrationConfig)
    portfolio_drawdown: PortfolioDrawdownConfig = Field(default_factory=PortfolioDrawdownConfig)

    def any_enabled(self) -> bool:
        return any(
            getattr(self, name).enabled
            for name in (
                "blacklist",
                "position_stop",
                "trailing_stop",
                "concentration",
                "portfolio_drawdown",
            )
        )


__all__ = [
    "BlacklistConfig",
    "ConcentrationConfig",
    "PortfolioDrawdownConfig",
    "PositionStopLossConfig",
    "RiskConfig",
    "TrailingStopConfig",
]
