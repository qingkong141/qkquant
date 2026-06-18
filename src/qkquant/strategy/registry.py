"""策略注册表：CLI 从这里根据 name 找到 backtrader 策略类和默认配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

import yaml

from qkquant.backtest.engine import BtStrategyBase
from qkquant.config import PROJECT_ROOT
from qkquant.risk import RiskConfig
from qkquant.strategy.chan import Chan2BuyStrategy
from qkquant.strategy.momentum import MomentumStrategy
from qkquant.strategy.momentum_breakout import MomentumBreakoutStrategy


@dataclass
class StrategyInfo:
    name: str
    cls: Type[BtStrategyBase]
    description: str
    config_path: Path | None = None


_STRAT_DIR = PROJECT_ROOT / "config" / "strategies"

_REGISTRY: dict[str, StrategyInfo] = {
    "momentum": StrategyInfo(
        name="动量策略",
        cls=MomentumStrategy,
        description="绝对动量 + 趋势过滤 + 分批止盈 + 行业中性",
        config_path=_STRAT_DIR / "momentum.yaml",
    ),
    "momentum_breakout": StrategyInfo(
        name="动量突破",
        cls=MomentumBreakoutStrategy,
        description="强动量+紧贴峰值+创10日新高，追真正的突破",
        config_path=_STRAT_DIR / "momentum_breakout.yaml",
    ),
    "chan_2buy": StrategyInfo(
        name="缠论二买",
        cls=Chan2BuyStrategy,
        description="一买确认反转后，回踩不破前低入场",
        config_path=_STRAT_DIR / "chan_2buy.yaml",
    ),
    "resonance": StrategyInfo(
        name="动量共振",
        cls=MomentumStrategy,
        description="动量策略 + 动量突破 共振：两策略同时看中才买入",
        config_path=_STRAT_DIR / "resonance.yaml",
    ),
}


def get_strategy(name: str) -> StrategyInfo:
    # 先按 key 查找，再按显示名查找
    if name in _REGISTRY:
        return _REGISTRY[name]
    for info in _REGISTRY.values():
        if info.name == name:
            return info
    raise KeyError(f"unknown strategy: {name}. Available: {[s.name for s in _REGISTRY.values()]}")


def list_strategies() -> list[StrategyInfo]:
    return list(_REGISTRY.values())


def load_strategy_config(info: StrategyInfo) -> dict:
    if info.config_path and info.config_path.exists():
        with info.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_risk_config(strategy_cfg: dict | None) -> RiskConfig:
    """从策略 yaml 字典中解析 `risk:` 段，没有则返回空 RiskConfig（全部 disabled）。"""
    if not strategy_cfg:
        return RiskConfig()
    risk_section = strategy_cfg.get("risk") or {}
    return RiskConfig(**risk_section)


__all__ = [
    "StrategyInfo",
    "get_strategy",
    "list_strategies",
    "load_risk_config",
    "load_strategy_config",
]
