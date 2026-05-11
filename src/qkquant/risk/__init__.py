"""风控层：策略无关的组合层规则集。

用法：
    from qkquant.risk import RiskConfig, RiskManager
    cfg = RiskConfig(trailing_stop={"enabled": True, "threshold": 0.12})
    manager = RiskManager(cfg)
    # 传给 BacktestEngine 或注入 strategy
"""

from qkquant.risk.base import RiskDecision, RiskRule
from qkquant.risk.config import (
    BlacklistConfig,
    ConcentrationConfig,
    PortfolioDrawdownConfig,
    PositionStopLossConfig,
    RiskConfig,
    TrailingStopConfig,
)
from qkquant.risk.manager import RiskManager
from qkquant.risk.rules import (
    BlacklistRule,
    ConcentrationRule,
    PortfolioDrawdownRule,
    PositionStopLossRule,
    TrailingStopRule,
)

__all__ = [
    "BlacklistConfig",
    "BlacklistRule",
    "ConcentrationConfig",
    "ConcentrationRule",
    "PortfolioDrawdownConfig",
    "PortfolioDrawdownRule",
    "PositionStopLossConfig",
    "PositionStopLossRule",
    "RiskConfig",
    "RiskDecision",
    "RiskManager",
    "RiskRule",
    "TrailingStopConfig",
    "TrailingStopRule",
]
