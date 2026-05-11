"""双均线突破策略。

逻辑：
- 买入：5 日均线上穿 20 日均线，且 仓位未满 max_positions，且 今日非涨停
- 卖出：5 日均线下穿 20 日均线，或 收盘跌破 20 日均线 (1 - stop_loss_pct)
- 仓位：最多同时持有 max_positions 只，等权分配。
- 每只候选股的下单金额 = 当前账户总市值 / max_positions。
"""

from __future__ import annotations

from collections import OrderedDict

import backtrader as bt

from qkquant.backtest.engine import BtStrategyBase


class MaBreakoutStrategy(BtStrategyBase):
    """双均线突破：适合做日频 MVP 演示。"""

    params = (
        ("fast", 5),
        ("slow", 20),
        ("stop_loss_pct", 0.05),
        ("max_positions", 10),
        ("min_price", 1.0),       # 低于此价格跳过（防 1 分以下极端）
    )

    def __init__(self) -> None:
        super().__init__()
        self.inds: dict[str, dict] = {}
        for data in self.datas:
            self.inds[data._name] = {
                "fast": bt.indicators.SMA(data.close, period=self.p.fast),
                "slow": bt.indicators.SMA(data.close, period=self.p.slow),
                "cross": bt.indicators.CrossOver(
                    bt.indicators.SMA(data.close, period=self.p.fast),
                    bt.indicators.SMA(data.close, period=self.p.slow),
                ),
            }
        self._trade_log: list[dict] = []

    # ---- 辅助 ----

    def _current_positions(self) -> list[str]:
        held = []
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                held.append(data._name)
        return held

    def _target_position_value(self) -> float:
        total = self.broker.getvalue()
        return total / self.p.max_positions

    # ---- 主循环 ----

    def next(self) -> None:
        # 0. 风控强平（若启用 RiskManager）
        self.apply_forced_exits()

        held = set(self._current_positions())
        today = self._today()

        # 1. 先卖出
        for data in self.datas:
            code = data._name
            if code not in held:
                continue
            ind = self.inds[code]
            close = float(data.close[0])
            slow_val = float(ind["slow"][0]) if len(ind["slow"]) > 0 else 0.0
            cross = float(ind["cross"][0]) if len(ind["cross"]) > 0 else 0.0

            sell_reason = None
            if cross < 0:
                sell_reason = "ma_cross_down"
            elif slow_val > 0 and close < slow_val * (1 - self.p.stop_loss_pct):
                sell_reason = "stop_loss"

            if sell_reason:
                pos = self.getposition(data)
                order = self.safe_sell(data, pos.size, reason=sell_reason)
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "SELL",
                            "price": close,
                            "qty": pos.size,
                            "reason": sell_reason,
                        }
                    )

        # 2. 再买入（仓位有空位才买）
        held = set(self._current_positions())
        slots = self.p.max_positions - len(held)
        if slots <= 0:
            return

        candidates: list[tuple[str, float]] = []
        for data in self.datas:
            code = data._name
            if code in held:
                continue
            ind = self.inds[code]
            if len(ind["cross"]) == 0:
                continue
            cross = float(ind["cross"][0])
            close = float(data.close[0])
            if cross > 0 and close >= self.p.min_price:
                # 用 (fast - slow) 的相对强度作为排序键,优先买强势股
                strength = float(ind["fast"][0]) / max(float(ind["slow"][0]), 1e-6) - 1
                candidates.append((code, strength))

        candidates.sort(key=lambda x: x[1], reverse=True)

        target_value = self._target_position_value()
        data_by_code = {d._name: d for d in self.datas}
        for code, _ in candidates[:slots]:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue
            qty = int((target_value / close) // 100) * 100    # A股按手(100股)下单
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="ma_cross_up")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "ma_cross_up",
                    }
                )


__all__ = ["MaBreakoutStrategy"]
