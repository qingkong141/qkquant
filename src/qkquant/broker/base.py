"""下单通道抽象接口（未来接入 easytrader / miniQMT / 模拟盘）。

本次仅定义契约，不提供任何实现——保证将来切换通道时，
策略层和调度层代码零改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str
    code: str
    side: OrderSide
    price: float
    qty: int
    filled_qty: int
    status: OrderStatus
    created_at: datetime
    reason: str = ""


@dataclass
class Position:
    code: str
    qty: int
    available_qty: int        # T+1 下可用余额
    avg_price: float
    last_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class AccountSnapshot:
    cash: float
    available_cash: float
    total_value: float
    market_value: float
    positions: list[Position]
    updated_at: datetime


class BrokerBase(ABC):
    """交易通道抽象。未来每种通道（easytrader / miniQMT / Simulator）继承并实现。"""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def buy(self, code: str, price: float, qty: int) -> Order: ...

    @abstractmethod
    def sell(self, code: str, price: float, qty: int) -> Order: ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool: ...

    @abstractmethod
    def query_orders(self) -> list[Order]: ...

    @abstractmethod
    def query_positions(self) -> list[Position]: ...

    @abstractmethod
    def query_account(self) -> AccountSnapshot: ...


__all__ = [
    "AccountSnapshot",
    "BrokerBase",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Position",
]
