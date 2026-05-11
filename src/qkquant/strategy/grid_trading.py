"""网格交易策略：围绕基准价按网格分档调仓（高抛低吸）。

核心逻辑：
- 先确定基准价：优先使用 fixed_base_price；否则使用 close 的 base_window 均线。
- 计算当前价相对基准价偏离，映射到网格层数。
- 网格越高，目标仓位越低；网格越低，目标仓位越高。
- 按目标仓位与当前仓位差值下单（A股按 100 股一手）。
"""

from __future__ import annotations

import math

from qkquant.backtest.engine import BtStrategyBase


class GridTradingStrategy(BtStrategyBase):
    params = (
        ("grid_pct", 0.03),               # 单格间距：3%
        ("max_grid_levels", 5),           # 最多向上/向下各 5 格
        ("base_window", 20),              # 动态基准价均线窗口
        ("fixed_base_price", 0.0),        # >0 时固定基准价
        ("base_position_pct", 0.50),      # 基准价附近目标仓位（相对单票预算）
        ("position_step_pct", 0.10),      # 每上/下一格仓位变化 10%
        ("min_position_pct", 0.00),       # 单票最低目标仓位
        ("max_position_pct", 1.00),       # 单票最高目标仓位
        ("max_positions", 10),            # 单票预算 = 总资产 / max_positions
        ("min_price", 1.0),
        ("trend_window", 60),             # 趋势过滤均线窗口
        ("bear_position_cap", 0.20),      # 跌破趋势线时的仓位上限
        ("min_trade_value", 3000.0),      # 最小调仓金额，过滤噪音换手
        ("bull_position_boost", 0.15),    # 站上趋势线时额外加仓（提高漂移收益）
        ("rebalance_on_level_change_only", True),  # 仅网格层变化时调仓，减少碎单
    )

    def __init__(self) -> None:
        super().__init__()
        self._trade_log: list[dict] = []
        self._last_level: dict[str, int] = {}

    def _base_price(self, data) -> float:
        fixed = float(self.p.fixed_base_price)
        if fixed > 0:
            return fixed
        w = int(self.p.base_window)
        if w <= 0 or len(data.close) < w:
            return 0.0
        closes = [float(data.close[-i]) for i in range(w)]
        return sum(closes) / w

    def _grid_level(self, close: float, base: float) -> int:
        if base <= 0:
            return 0
        grid = max(float(self.p.grid_pct), 1e-6)
        raw = (close / base - 1.0) / grid
        level = math.floor(raw)
        max_lv = int(self.p.max_grid_levels)
        return max(-max_lv, min(max_lv, level))

    def _target_position_pct(self, level: int) -> float:
        # 价格越高(level 越大)目标仓位越低；价格越低(level 越小)目标仓位越高
        target = float(self.p.base_position_pct) - level * float(self.p.position_step_pct)
        return max(float(self.p.min_position_pct), min(float(self.p.max_position_pct), target))

    def _target_shares(self, close: float, target_pct: float) -> int:
        if close <= 0:
            return 0
        slot_value = self.broker.getvalue() / max(int(self.p.max_positions), 1)
        target_value = slot_value * target_pct
        return int((target_value / close) // 100) * 100

    def _trend_ma(self, data) -> float:
        w = int(self.p.trend_window)
        if w <= 0 or len(data.close) < w:
            return 0.0
        closes = [float(data.close[-i]) for i in range(w)]
        return sum(closes) / w

    def next(self) -> None:
        self.apply_forced_exits()
        today = self._today()

        for data in self.datas:
            code = data._name
            close = float(data.close[0])
            if close < float(self.p.min_price):
                continue

            base = self._base_price(data)
            if base <= 0:
                continue

            level = self._grid_level(close, base)
            target_pct = self._target_position_pct(level)

            # 趋势过滤：弱势阶段限制最高仓位，减少单边下跌中不断补仓
            trend_ma = self._trend_ma(data)
            if trend_ma > 0 and close < trend_ma:
                target_pct = min(target_pct, float(self.p.bear_position_cap))
            elif trend_ma > 0 and close >= trend_ma:
                target_pct = min(
                    float(self.p.max_position_pct),
                    target_pct + float(self.p.bull_position_boost),
                )

            # 仅在网格层发生变化时才重平衡，避免因为净值浮动导致的频繁小额调仓
            if bool(self.p.rebalance_on_level_change_only):
                prev_level = self._last_level.get(code)
                if prev_level is not None and prev_level == level:
                    continue
                self._last_level[code] = level

            target_qty = self._target_shares(close, target_pct)
            curr_qty = int(self.getposition(data).size)
            delta = target_qty - curr_qty
            trade_value = abs(delta) * close
            if trade_value < float(self.p.min_trade_value):
                continue

            if delta >= 100:
                order = self.safe_buy(data, delta, reason=f"grid_buy_l{level}")
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "BUY",
                            "price": close,
                            "qty": delta,
                            "reason": f"grid_buy_l{level}",
                            "grid_level": level,
                            "base_price": base,
                            "target_pct": target_pct,
                        }
                    )
            elif delta <= -100:
                sell_qty = abs(delta)
                order = self.safe_sell(data, sell_qty, reason=f"grid_sell_l{level}")
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "SELL",
                            "price": close,
                            "qty": sell_qty,
                            "reason": f"grid_sell_l{level}",
                            "grid_level": level,
                            "base_price": base,
                            "target_pct": target_pct,
                        }
                    )


__all__ = ["GridTradingStrategy"]
