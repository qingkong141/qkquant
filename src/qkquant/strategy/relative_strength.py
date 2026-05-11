"""相对强度选股策略（Relative Strength / 横截面动量）。

逻辑：
- 每 rebalance_days 个交易日做一次调仓。
- 以过去 lookback 日收益率为选股信号。
- 选出当前非停牌、非涨停的 top_k 只，等权持有。
- 调仓日：卖掉不在新名单里的，买入新名单里还没持有的。
- 其他日子不动作。

对比 ma_breakout：后者是单票逐个看信号，本策略是横截面排序，更像量化公募。
"""

from __future__ import annotations

import backtrader as bt

from qkquant.backtest.engine import BtStrategyBase


class RelativeStrengthStrategy(BtStrategyBase):
    """横截面动量选股。"""

    params = (
        ("lookback", 60),           # 过去 N 日收益率
        ("rebalance_days", 20),     # 每 N 日调仓
        ("top_k", 10),              # 持有 top K
        ("min_price", 1.0),
    )

    def __init__(self) -> None:
        super().__init__()
        self._bar_count = 0
        self._trade_log: list[dict] = []

    def _momentum(self, data) -> float:
        """过去 lookback 日收益率。"""
        lb = self.p.lookback
        if len(data.close) <= lb:
            return float("-inf")
        try:
            start_price = float(data.close[-lb])
            end_price = float(data.close[0])
            if start_price <= 0:
                return float("-inf")
            return end_price / start_price - 1.0
        except IndexError:
            return float("-inf")

    def _should_rebalance(self) -> bool:
        self._bar_count += 1
        return self._bar_count % self.p.rebalance_days == 1

    def next(self) -> None:
        # 风控强平（若启用 RiskManager）。注意：放在 rebalance 检查之外，
        # 让风控每日都能跑（即使今日不是调仓日也需要扫强平信号）。
        self.apply_forced_exits()

        if not self._should_rebalance():
            return

        today = self._today()

        scored: list[tuple[str, float, object]] = []
        for data in self.datas:
            if float(data.close[0]) < self.p.min_price:
                continue
            mom = self._momentum(data)
            if mom == float("-inf"):
                continue
            scored.append((data._name, mom, data))

        scored.sort(key=lambda x: x[1], reverse=True)
        target_codes = {code for code, _, _ in scored[: self.p.top_k]}
        data_by_code = {d._name: d for d in self.datas}

        # 1. 卖出：不在新名单里的当前持仓
        for data in self.datas:
            code = data._name
            pos = self.getposition(data)
            if pos.size > 0 and code not in target_codes:
                order = self.safe_sell(data, pos.size, reason="rs_rebalance_out")
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "SELL",
                            "price": float(data.close[0]),
                            "qty": pos.size,
                            "reason": "rebalance_out",
                        }
                    )

        # 2. 买入：在新名单里但还没持仓的
        held = {d._name for d in self.datas if self.getposition(d).size > 0}
        buy_list = [c for c in target_codes if c not in held]
        if not buy_list:
            return

        target_value = self.broker.getvalue() / self.p.top_k
        for code in buy_list:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue
            qty = int((target_value / close) // 100) * 100
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="rs_rebalance_in")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "rebalance_in",
                    }
                )


__all__ = ["RelativeStrengthStrategy"]
