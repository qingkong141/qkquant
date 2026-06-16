"""动量 + 动量反转信号的策略（Trend Following）。

逻辑：
- 每日扫描。
- 买入条件：过去 mom_window 日收益率 > entry_threshold（默认 +3%），且 当前价在 mom_window 日内最高点的 drawdown_from_peak 以内（默认 10%），
  且 当前未满仓。
- 卖出条件：过去 mom_window 日收益率 < exit_threshold（默认 -3%）触发动量反转退出。
- 最多持有 max_positions 只，等权。

注意：追踪止损（trailing stop）已下放到 qkquant.risk.TrailingStopRule，
通过策略 yaml 的 risk.trailing_stop 段启用。本策略代码只保留"动量信号"相关逻辑。
"""

from __future__ import annotations

import backtrader as bt  # noqa: F401  预留未来接 indicator 用

from qkquant.backtest.engine import BtStrategyBase


class MomentumStrategy(BtStrategyBase):
    """绝对动量 + 动量反转退出；追踪止损由风控层提供。"""

    params = (
        ("mom_window", 20),
        ("entry_threshold", 0.03),     # 过去 20 日 +3% 才考虑买
        ("exit_threshold", -0.03),     # 过去 20 日 -3% 触发卖
        ("drawdown_from_peak", 0.10),  # 从 20 日最高点的最大折价
        ("max_positions", 5),
        ("min_price", 1.0),
        ("min_float_cap", 2_000_000_000),   # 盘口：最小流通市值 20 亿（过滤微盘庄股）
        ("max_float_cap", 0),               # 盘口：最大流通市值上限（0=不限）
        ("min_amount", 5_000_000),          # 盘口：最小成交额 500 万（过滤无流动性票）
        ("market_caps", None),              # 内部：由引擎注入 {code: {total_cap, float_cap}}
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

    def _current_positions(self) -> list[str]:
        return [d._name for d in self.datas if self.getposition(d).size > 0]

    def next(self) -> None:
        # 0. 风控强平（trailing stop 等都在这里跑）
        self.apply_forced_exits()

        today = self._today()

        # 1. 策略自己的动量反转退出（信号层，不是风控层）
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

        # 2. 再跑买入
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
            if win_high > 0 and close < win_high * (1 - self.p.drawdown_from_peak):
                continue

            # ---- 盘口限制 ----

            # 1. 市值过滤：太小不碰（庄股/流动性差），太大不碰（盘子太重难拉）
            if self._mc:
                mc = self._mc.get(code)
                if mc is not None:
                    float_cap = mc.get("float_cap", 0) or 0
                    if self.p.min_float_cap > 0 and float_cap < self.p.min_float_cap:
                        continue
                    if self.p.max_float_cap > 0 and float_cap > self.p.max_float_cap:
                        continue

            # 2. 最小成交额：流动性不足不参与
            if self.p.min_amount > 0:
                vol = float(data.volume[0])
                amount = vol * close
                if amount < self.p.min_amount:
                    continue

            # ADX 趋势强度过滤：震荡市不买入
            if self.p.adx_threshold > 0 and not self._adx_ok(data):
                continue
            # 冷却期过滤：卖出后 N 天内不重新买入
            if self._in_cooldown(code, today):
                continue

            candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
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
            order = self.safe_buy(data, qty, reason="momentum_entry")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "momentum_entry",
                    }
                )


__all__ = ["MomentumStrategy"]
