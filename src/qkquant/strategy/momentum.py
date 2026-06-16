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
from qkquant.factors.indicators import atr


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
        ("take_profit_pct", 0.15),          # 分批止盈：盈利超此比例卖一半；0=关闭
        ("time_stop_days", 60),             # 时间止损：持仓超此天数强制清仓；0=关闭
        ("atr_weight", False),              # ATR 波动率倒数加权仓位；高波票少配
        ("vol_confirm", True),              # 量价确认：放量突破才买入
        ("vol_ratio_min", 0.8),             # 当日量必须 > 20日均量×此倍率
        ("min_up_ratio", 0.50),             # 连涨比率：mom_window 内上涨天数占比下限
    )

    def __init__(self) -> None:
        super().__init__()
        self._trade_log: list[dict] = []
        self._mc: dict[str, dict] = self.p.market_caps or {}
        self._entry_price: dict[str, float] = {}  # 每笔持仓的入场均价
        self._held_since: dict[str, int] = {}     # 持仓至今的 bar 数

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

        # ---- 更新持仓追踪 ----
        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size <= 0:
                self._held_since.pop(code, None)
                self._entry_price.pop(code, None)
                continue
            self._held_since[code] = self._held_since.get(code, 0) + 1

        # ---- 1. 出场逻辑 ----

        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            close = float(data.close[0])
            entry = self._entry_price.get(code, close)

            # 1a. 分批止盈：盈利超 take_profit_pct → 卖一半
            if self.p.take_profit_pct > 0 and entry > 0:
                gain = close / entry - 1.0
                if gain >= self.p.take_profit_pct and pos.size > 100:
                    half = (pos.size // 200) * 100  # 取整百股
                    if half > 0:
                        order = self.safe_sell(data, half, reason="take_profit")
                        if order is not None:
                            self._trade_log.append({
                                "date": today, "code": code, "side": "SELL",
                                "price": close, "qty": half,
                                "reason": f"take_profit:{gain:+.1%}",
                            })

            # 1b. 时间止损：持仓超 N 个交易日 → 全平
            if self.p.time_stop_days > 0:
                held_bars = self._held_since.get(code, 0)
                if held_bars >= self.p.time_stop_days:
                    order = self.safe_sell(data, pos.size, reason="time_stop")
                    if order is not None:
                        self._trade_log.append({
                            "date": today, "code": code, "side": "SELL",
                            "price": close, "qty": pos.size,
                            "reason": f"time_stop:{held_bars}d",
                        })
                    continue  # 已清仓，跳过后续检查

            # 1c. 动量反转退出
            mom = self._window_return(data)
            if mom < self.p.exit_threshold:
                order = self.safe_sell(data, pos.size, reason="momentum_exit")
                if order is not None:
                    self._trade_log.append({
                        "date": today, "code": code, "side": "SELL",
                        "price": close, "qty": pos.size,
                        "reason": "momentum_exit",
                    })

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
            # PE 估值过滤：太贵不买
            max_pe = self.p.max_pe
            if max_pe > 0:
                eps_map: dict[str, float] = self.p.eps_map or {}
                eps = eps_map.get(code, 0)
                if eps > 0 and close / eps > max_pe:
                    continue

            # 量价确认：今日量必须 > 20日均量×vol_ratio_min（缩量突破不可靠）
            if self.p.vol_confirm:
                w = self.p.mom_window
                if len(data.volume) > w:
                    vols = [float(data.volume[-i]) for i in range(1, w + 1)]
                    avg_vol = sum(vols) / len(vols)
                    today_vol = float(data.volume[0])
                    if avg_vol > 0 and today_vol / avg_vol < self.p.vol_ratio_min:
                        continue

            # 连涨比率：上涨天数/总天数 < min_up_ratio → 个别暴涨撑动量，不靠谱
            min_ur = self.p.min_up_ratio
            if min_ur > 0:
                w = self.p.mom_window
                if len(data.close) > w:
                    up_days = sum(
                        1 for i in range(1, w + 1)
                        if float(data.close[-i]) > float(data.close[-i - 1])
                    )
                    if up_days / w < min_ur:
                        continue

            candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_value = self.broker.getvalue() / self.p.max_positions
        data_by_code = {d._name: d for d in self.datas}

        # 行业中性化：每个行业最多 max_per_industry 只
        ind_map: dict[str, str] = self.p.industry_map or {}
        max_per_ind = self.p.max_per_industry
        ind_count: dict[str, int] = {}
        selected: list[str] = []
        for code, _ in candidates:
            if len(selected) >= slots:
                break
            if max_per_ind > 0:
                ind = ind_map.get(code, "")
                if ind and ind_count.get(ind, 0) >= max_per_ind:
                    continue
                ind_count[ind] = ind_count.get(ind, 0) + 1
            selected.append(code)

        for code in selected:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue

            # 基础仓位：等权
            base_qty = int((target_value / close) // 100) * 100
            if base_qty <= 0:
                continue

            # ATR 波动率倒数加权：高波少配、低波多配
            if self.p.atr_weight:
                try:
                    high_s = data.high.get(size=15)
                    low_s = data.low.get(size=15)
                    close_s = data.close.get(size=15)
                    import pandas as pd
                    atr_val = float(atr(
                        pd.Series([float(x) for x in high_s]),
                        pd.Series([float(x) for x in low_s]),
                        pd.Series([float(x) for x in close_s]),
                    ).iloc[-1])
                except Exception:
                    atr_val = 0
                if atr_val > 0 and close > 0:
                    atr_ratio = atr_val / close  # ATR 占价格百分比
                    # 基准 ATR 比 3%，高于此 → 减仓，低于此 → 加仓
                    base_ratio = 0.03
                    weight = min(max(base_ratio / max(atr_ratio, 0.005), 0.5), 1.5)
                    qty = int(base_qty * weight // 100) * 100
                else:
                    qty = base_qty
            else:
                qty = base_qty

            if qty <= 0:
                continue

            order = self.safe_buy(data, qty, reason="momentum_entry")
            if order is not None:
                self._entry_price[code] = close
                self._held_since[code] = 0
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
