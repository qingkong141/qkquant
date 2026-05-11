"""双均线 + 布林带 (MA + BOLL) 组合策略。

设计意图：
- ma_breakout 的死穴是震荡市 whipsaw，BOLL 提供波动率维度的额外信息
- 入场要求 close 在 BOLL 中轨之上 → 趋势已经确认
- 入场要求 close 距上轨有 buffer → 避免在带状极端追高
- 出场用 BOLL 中轨做止损，比固定百分比更适应波动率

入场（同时满足）：
- MA(fast) 上穿 MA(slow) 金叉
- close > BOLL 中轨（趋势完好）
- close ≤ BOLL 上轨 × (1 - upper_buffer)（未触及上轨）
- close ≥ min_price

出场（任一）：
- MA 死叉
- close < BOLL 中轨 × (1 - mid_break_pct)
"""

from __future__ import annotations

import backtrader as bt

from qkquant.backtest.engine import BtStrategyBase


class MaBollStrategy(BtStrategyBase):
    params = (
        ("fast", 5),
        ("slow", 20),
        ("boll_period", 20),
        ("boll_dev", 2.0),
        ("upper_buffer", 0.03),       # 距上轨需有 3% 缓冲
        ("mid_break_pct", 0.05),      # 跌破中轨 5% 强出
        ("max_positions", 10),
        ("min_price", 1.0),
    )

    def __init__(self) -> None:
        super().__init__()
        self.inds: dict[str, dict] = {}
        for data in self.datas:
            fast = bt.indicators.SMA(data.close, period=self.p.fast)
            slow = bt.indicators.SMA(data.close, period=self.p.slow)
            boll = bt.indicators.BollingerBands(
                data.close, period=self.p.boll_period, devfactor=self.p.boll_dev
            )
            self.inds[data._name] = {
                "fast": fast,
                "slow": slow,
                "cross": bt.indicators.CrossOver(fast, slow),
                "boll": boll,
            }
        self._trade_log: list[dict] = []

    def _current_positions(self) -> list[str]:
        return [d._name for d in self.datas if self.getposition(d).size > 0]

    def _target_position_value(self) -> float:
        return self.broker.getvalue() / self.p.max_positions

    def next(self) -> None:
        self.apply_forced_exits()

        held = set(self._current_positions())
        today = self._today()

        # 1. 出场
        for data in self.datas:
            code = data._name
            if code not in held:
                continue
            ind = self.inds[code]
            close = float(data.close[0])
            cross = float(ind["cross"][0]) if len(ind["cross"]) > 0 else 0.0
            boll_mid = float(ind["boll"].mid[0]) if len(ind["boll"].mid) > 0 else 0.0

            sell_reason = None
            if cross < 0:
                sell_reason = "ma_cross_down"
            elif boll_mid > 0 and close < boll_mid * (1 - self.p.mid_break_pct):
                sell_reason = "below_boll_mid"

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
            ind = self.inds[code]
            if len(ind["cross"]) == 0 or len(ind["boll"].mid) == 0:
                continue
            cross = float(ind["cross"][0])
            close = float(data.close[0])
            if cross <= 0 or close < self.p.min_price:
                continue

            mid = float(ind["boll"].mid[0])
            upper = float(ind["boll"].top[0])
            if mid <= 0 or upper <= 0:
                continue

            # 趋势完好 + 不在上轨极端
            if close <= mid:
                continue
            if close > upper * (1 - self.p.upper_buffer):
                self._rejections.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "reason": "near_upper_band",
                    }
                )
                continue

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
            qty = int((target_value / close) // 100) * 100
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="ma_boll_entry")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "ma_boll_entry",
                    }
                )


__all__ = ["MaBollStrategy"]
