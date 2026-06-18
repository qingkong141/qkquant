"""缠论二买策略：一买确认反转后，回踩不破前低时入场。

一买 = 跌破中枢 + 底背驰（空头衰竭）
二买 = 一买后反弹 >5% + 回踩不破一买低点 → 入场
出场 = 跌破一买低点（反转失败）或 移动止盈
"""

from __future__ import annotations

import backtrader as bt
import numpy as np

from qkquant.backtest.engine import BtStrategyBase


class ChanStrategy(BtStrategyBase):
    """缠论二买：等一买确认反转后，回踩企稳入场。"""

    params = (
        ("max_positions", 5),
        ("min_price", 1.0),
        ("lookback", 100),           # 中枢计算回看K线数
        ("swing_min_bars", 4),       # 笔的最小K线数
        ("rebound_pct", 0.05),       # 一买后至少反弹 5% 才找二买
        ("stop_at_first_buy", True), # 跌破一买低点止损
        ("take_profit_pct", 0.15),   # 分批止盈
    )

    def __init__(self) -> None:
        super().__init__()
        self._trade_log: list[dict] = []
        self._entry_price: dict[str, float] = {}
        self._first_buy_low: dict[str, float] = {}  # 一买低点（止损参考）
        self._cache: dict[str, dict] = {}           # 缓存每只票的中枢/笔

    def _compute_chan(self, data) -> dict:
        """计算中枢、笔、底背驰，返回最近的一买和二买位置。

        每 10 根 K 线重算一次（缓存优化）。
        """
        code = data._name
        n = len(data)
        cached = self._cache.get(code, {})
        if cached.get("computed_at", 0) >= n - 5:
            return cached

        lookback = min(n, self.p.lookback)
        if lookback < 40:
            return {}

        highs = np.array([float(data.high[-i]) for i in range(lookback - 1, -1, -1)])
        lows = np.array([float(data.low[-i]) for i in range(lookback - 1, -1, -1)])
        closes = np.array([float(data.close[-i]) for i in range(lookback - 1, -1, -1)])

        # 笔
        mb = self.p.swing_min_bars
        raw_swings = []
        for i in range(mb, len(highs) - mb):
            if all(highs[i] >= highs[i-j] for j in range(1, mb+1)) and all(highs[i] > highs[i+j] for j in range(1, mb+1)):
                raw_swings.append({"type": "high", "price": float(highs[i]), "idx": i})
            if all(lows[i] <= lows[i-j] for j in range(1, mb+1)) and all(lows[i] < lows[i+j] for j in range(1, mb+1)):
                raw_swings.append({"type": "low", "price": float(lows[i]), "idx": i})

        swings = []
        for s in sorted(raw_swings, key=lambda x: x["idx"]):
            if not swings or s["type"] != swings[-1]["type"]:
                swings.append(s)

        if len(swings) < 6:
            self._cache[code] = {"computed_at": n}
            return self._cache[code]

        # 中枢：最后 3 笔的重叠
        last_highs = [s["price"] for s in swings[-6:] if s["type"] == "high"][:3]
        last_lows = [s["price"] for s in swings[-6:] if s["type"] == "low"][:3]
        if len(last_highs) < 3 or len(last_lows) < 3:
            self._cache[code] = {"computed_at": n}
            return self._cache[code]

        zg = min(last_highs)
        zd = max(last_lows)
        has_zs = zg > zd

        # 底背驰检测
        bottom_div_idx = None
        if has_zs:
            low_swings = [s for s in swings if s["type"] == "low"]
            for i in range(1, len(low_swings)):
                prev = low_swings[i-1]
                curr = low_swings[i]
                if curr["price"] >= prev["price"]:
                    continue
                # MACD 面积
                if len(closes) > curr["idx"]:
                    macd_fast = pd_ema(closes, 12) - pd_ema(closes, 26)
                    signal = pd_ema(macd_fast, 9)
                    hist = macd_fast - signal
                    pa = np.sum(np.abs(hist[max(0, prev["idx"]-10):prev["idx"]+1]))
                    ca = np.sum(np.abs(hist[max(0, curr["idx"]-10):curr["idx"]+1]))
                    if ca < pa * 0.7:
                        bottom_div_idx = curr["idx"]
                        break

        # 一买：跌破中枢 + 底背驰
        first_buy_idx = None
        first_buy_low = None
        if has_zs and bottom_div_idx is not None:
            fb_swings = [s for s in swings if s["type"] == "low" and s["idx"] == bottom_div_idx]
            if fb_swings and fb_swings[0]["price"] < zd:
                first_buy_idx = bottom_div_idx
                first_buy_low = fb_swings[0]["price"]

        # 二买：一买后反弹 >5% + 回踩不破一买低点
        second_buy_idx = None
        if first_buy_idx is not None:
            post_fb = closes[first_buy_idx:]
            if len(post_fb) > 10:
                peak = np.max(post_fb[5:])
                if peak > first_buy_low * (1 + self.p.rebound_pct):
                    peak_pos = first_buy_idx + 5 + np.argmax(post_fb[5:])
                    after_peak = closes[peak_pos:]
                    for j in range(2, min(len(after_peak), 60)):
                        idx = peak_pos + j
                        if idx < len(closes) - 2:
                            if closes[idx] < closes[idx-1] and closes[idx] < closes[idx+1]:
                                if closes[idx] > first_buy_low * 1.01:
                                    second_buy_idx = idx
                                    break

        self._cache[code] = {
            "computed_at": n,
            "has_zs": has_zs, "zg": zg, "zd": zd,
            "first_buy_idx": first_buy_idx, "first_buy_low": first_buy_low,
            "second_buy_idx": second_buy_idx,
        }
        return self._cache[code]


def pd_ema(series: np.ndarray, window: int) -> np.ndarray:
    """简版 EMA"""
    alpha = 2.0 / (window + 1)
    out = np.zeros_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out


# ── 策略逻辑 ────────────────────────────────────────────────────────

class Chan2BuyStrategy(ChanStrategy):
    """缠论二买策略：一买确认反转后，回踩企稳入场。"""

    def _current_held(self):
        return [d._name for d in self.datas if self.getposition(d).size > 0]

    def next(self) -> None:
        self.apply_forced_exits()
        today = self._today()

        # 更新持仓追踪
        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size <= 0:
                self._first_buy_low.pop(code, None)
                self._entry_price.pop(code, None)
                continue

        # 出场
        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            close = float(data.close[0])
            entry = self._entry_price.get(code, close)

            # 分批止盈
            if self.p.take_profit_pct > 0 and entry > 0:
                gain = close / entry - 1.0
                if gain >= self.p.take_profit_pct and pos.size > 100:
                    half = (pos.size // 200) * 100
                    if half > 0:
                        order = self.safe_sell(data, half, reason="take_profit")
                        if order is not None:
                            self._trade_log.append({
                                "date": today, "code": code, "side": "SELL",
                                "price": close, "qty": half,
                                "reason": f"take_profit:{gain:+.1%}",
                            })

            # 跌破一买低点 → 全平
            fb_low = self._first_buy_low.get(code)
            if fb_low and close < fb_low:
                order = self.safe_sell(data, pos.size, reason="below_first_buy")
                if order is not None:
                    self._trade_log.append({
                        "date": today, "code": code, "side": "SELL",
                        "price": close, "qty": pos.size,
                        "reason": "below_first_buy",
                    })

        # 入场
        held = set(self._current_held())
        slots = self.p.max_positions - len(held)
        if slots <= 0:
            return

        candidates = []
        for data in self.datas:
            code = data._name
            if code in held:
                continue
            close = float(data.close[0])
            if close < self.p.min_price:
                continue

            chan = self._compute_chan(data)
            second_idx = chan.get("second_buy_idx")
            if second_idx is None:
                continue
            # 二买必须在最近 10 根K线内
            actual_lb = min(len(data), self.p.lookback)
            bars_ago = actual_lb - 1 - second_idx
            if bars_ago > 10:
                continue

            first_low = chan.get("first_buy_low", 0)
            score = close / max(first_low, 0.01)
            candidates.append((code, score, first_low))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_value = self.broker.getvalue() / self.p.max_positions
        data_by_code = {d._name: d for d in self.datas}

        for code, _, fb_low in candidates[:slots]:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue
            qty = int((target_value / close) // 100) * 100
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="chan_2buy")
            if order is not None:
                self._entry_price[code] = close
                self._first_buy_low[code] = fb_low
                self._trade_log.append({
                    "date": today, "code": code, "side": "BUY",
                    "price": close, "qty": qty, "reason": "chan_2buy",
                })

    def notify_order(self, order: bt.Order) -> None:
        super().notify_order(order)
        if order.status == order.Completed and order.issell():
            code = order.data._name
            price = float(order.executed.price)
            qty = int(order.executed.size)
            today = self._today()
            for t in self._trade_log:
                if t["date"] == today and t["code"] == code and t["side"] == "SELL":
                    break
            else:
                self._trade_log.append({
                    "date": today, "code": code, "side": "SELL",
                    "price": price, "qty": qty, "reason": "forced_exit",
                })


__all__ = ["Chan2BuyStrategy"]
