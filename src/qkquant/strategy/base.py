"""策略抽象基类与通用数据结构。

这一层不依赖 backtrader——它是框架级的契约，
回测引擎和未来的实盘引擎共用一份 StrategyBase 定义。

本次 MVP 中 backtrader 内置的 bt.Strategy 被直接继承使用；
这里的 StrategyBase 主要作为未来自研事件引擎/实盘的契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Bar:
    code: str
    dt: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None
    pct_chg: float | None = None
    turnover: float | None = None


@dataclass
class Signal:
    code: str
    dt: date
    type: SignalType
    price: float | None = None          # None 表示市价/下一开盘价
    target_weight: float | None = None  # 0.1 代表目标仓位 10%
    reason: str = ""


@dataclass
class Context:
    """策略运行时上下文：账户状态、可交易股票池、当前时间等。"""

    current_date: date
    cash: float
    total_value: float
    positions: dict[str, int] = field(default_factory=dict)   # code -> qty
    universe: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class StrategyBase(ABC):
    """所有策略继承此类。回测与实盘共用同一个接口契约。"""

    name: str = "base"
    params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = {**self.__class__.params, **(params or {})}

    def on_start(self, ctx: Context) -> None:
        """回测/实盘启动时调用一次。"""

    def on_end(self, ctx: Context) -> None:
        """回测/实盘结束时调用一次。"""

    @abstractmethod
    def on_bar(self, bar: Bar, ctx: Context) -> list[Signal]:
        """每根 K 线回调一次，返回要下发的信号。"""


__all__ = ["Bar", "Context", "Signal", "SignalType", "StrategyBase"]
