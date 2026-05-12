"""动量突破策略：强化追高版本。

vs momentum 的差别：
- 入场门槛 +3% → +5%（更强动量才考虑）
- 距 20 日高点 10% → 3%（必须紧贴峰值）
- 新增"当日创 10 日新高收盘"硬条件（必须正在突破）
- 出场逻辑保留 momentum 的反转退出（mom < -3%）

设计意图：在 A股市场上"追真正的趋势"——回测显示等回调反而失败，
那就反向加重追高。
"""

from __future__ import annotations

import backtrader as bt  # noqa: F401  预留

from qkquant.backtest.engine import BtStrategyBase


class MomentumBreakoutStrategy(BtStrategyBase):
    params = (
        ("mom_window", 20),
        ("entry_threshold", 0.05),         # +5%
        ("exit_threshold", -0.03),         # mom 反转 -3% 退出
        ("drawdown_from_peak", 0.03),      # 距 20日高点不超过 3%
        ("breakout_window", 10),           # 必须创 10日新高
        ("max_positions", 8),
        ("min_price", 1.0),
        ("min_float_cap", 2_000_000_000),  # 盘口：最小流通市值 20 亿
        ("max_float_cap", 0),              # 盘口：最大流通市值上限（0=不限）
        ("min_amount", 5_000_000),         # 盘口：最小成交额 500 万
        ("market_caps", None),             # 内部：由引擎注入
    )

    def __init__(self) -> None:
        super().__init__()
        self._trade_log: list[dict] = []
        self._mc: dict[str, dict] = self.p.market_caps or {}

    def _window_return(self, data) -> float:
        w = self.p.mom_window
        if len(data.close) <= w:
            return float("-inf")
        start = float(data.close[-w])
        end = float(data.close[0])
        return end / start - 1.0 if start > 0 else float("-inf")

    def _window_high(self, data) -> float:
        w = self.p.mom_window
        if len(data.high) <= w:
            return 0.0
        return max(float(data.high[-i]) for i in range(w))

    def _is_close_breakout(self, data) -> bool:
        """今天收盘是否突破过去 breakout_window 日的最高收盘？"""
        w = self.p.breakout_window
        if len(data.close) <= w:
            return False
        today_close = float(data.close[0])
        prev_max = max(float(data.close[-i]) for i in range(1, w + 1))
        return today_close >= prev_max

    def _current_positions(self) -> list[str]:
        return [d._name for d in self.datas if self.getposition(d).size > 0]

    def next(self) -> None:
        self.apply_forced_exits()
        today = self._today()

        # 1. 出场（动量反转）
        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            close = float(data.close[0])
            mom = self._window_return(data)
            if mom < self.p.exit_threshold:
                order = self.safe_sell(data, pos.size, reason="momentum_exit")
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "SELL",
                            "price": close,
                            "qty": pos.size,
                            "reason": "momentum_exit",
                        }
                    )

        # 2. 入场
        held = set(self._current_positions())
        slots = self.p.max_positions - len(held)
        if slots <= 0:
            return

        candidates: list[tuple[str, float]] = []
        for data in self.datas:
            code = data._name
            if code in held:
                continue
            close = float(data.close[0])
            if close < self.p.min_price:
                continue
            mom = self._window_return(data)
            if mom < self.p.entry_threshold:
                continue
            win_high = self._window_high(data)
            if win_high <= 0:
                continue
            # 必须紧贴 20 日高点
            if close < win_high * (1 - self.p.drawdown_from_peak):
                continue
            # 必须创 N 日新高
            if not self._is_close_breakout(data):
                continue

            # 盘口：市值过滤
            if self._mc:
                mc = self._mc.get(code)
                if mc is not None:
                    float_cap = mc.get("float_cap", 0) or 0
                    if self.p.min_float_cap > 0 and float_cap < self.p.min_float_cap:
                        continue
                    if self.p.max_float_cap > 0 and float_cap > self.p.max_float_cap:
                        continue

            # 盘口：最小成交额
            if self.p.min_amount > 0:
                vol = float(data.volume[0])
                amount = vol * close
                if amount < self.p.min_amount:
                    continue

            candidates.append((code, mom))

        candidates.sort(key=lambda x: -x[1])
        target_value = self.broker.getvalue() / self.p.max_positions
        data_by_code = {d._name: d for d in self.datas}

        for code, _ in candidates[:slots]:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue
            qty = int((target_value / close) // 100) * 100
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="breakout_entry")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "breakout_entry",
                    }
                )


__all__ = ["MomentumBreakoutStrategy"]
