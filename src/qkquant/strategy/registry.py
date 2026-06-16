"""策略注册表：CLI 从这里根据 name 找到 backtrader 策略类和默认配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

import yaml

from qkquant.backtest.engine import BtStrategyBase
from qkquant.config import PROJECT_ROOT
from qkquant.risk import RiskConfig
from qkquant.strategy.ma_boll import MaBollStrategy
from qkquant.strategy.momentum import MomentumStrategy


@dataclass
class StrategyInfo:
    name: str
    cls: Type[BtStrategyBase]
    description: str
    config_path: Path | None = None


_STRAT_DIR = PROJECT_ROOT / "config" / "strategies"

# 生产入口只保留这两个策略。旧策略源码和配置作为历史实验保留，
# 但不再注册到 CLI / scan 的可用策略列表中。
_REGISTRY: dict[str, StrategyInfo] = {
    "momentum": StrategyInfo(
        name="momentum",
        cls=MomentumStrategy,
        description="绝对动量 + 追踪止损（每日扫描，上限 10 只）",
        config_path=_STRAT_DIR / "momentum.yaml",
    ),
    "ma_boll": StrategyInfo(
        name="ma_boll",
        cls=MaBollStrategy,
        description="双均线 + 布林带（金叉+趋势确认+不追高+中轨止损）",
        config_path=_STRAT_DIR / "ma_boll.yaml",
    ),
}


def get_strategy(name: str) -> StrategyInfo:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


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
