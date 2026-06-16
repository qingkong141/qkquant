"""基于 backtrader 的回测引擎。

A 股特殊规则在这一层实现，策略代码只关心信号逻辑：
- T+1：当日买入，次日才可卖出（在 BtStrategyBase.safe_sell 里拦截）
- 涨停禁买、跌停禁卖：基于 pct_chg 列在下单前拦截
- 佣金/印花税/滑点：通过 CommissionInfo + set_slippage_perc
- ST 过滤：由上层 universe 筛选时完成，这里不重复处理
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from qkquant.backtest.datafeed import PandasDataExt, build_feeds
from qkquant.config import get_settings
from qkquant.data.storage import DuckStore
from qkquant.factors.indicators import adx as compute_adx
from qkquant.logger import logger
from qkquant.risk import RiskConfig, RiskManager

LIMIT_UP_THRESHOLD = 9.8     # pct_chg >= 9.8 视为涨停（保守，覆盖主板）
LIMIT_DOWN_THRESHOLD = -9.8  # pct_chg <= -9.8 视为跌停


class AShareCommission(bt.CommInfoBase):
    """A 股交易费用：佣金双边万 2.5（最低 5 元）+ 卖出印花税千 1。"""

    params = (
        ("commission_rate", 0.00025),
        ("commission_min", 5.0),
        ("stamp_tax", 0.001),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission_rate, self.p.commission_min)
        if size < 0:  # 卖出
            commission += turnover * self.p.stamp_tax
        return commission


class BtStrategyBase(bt.Strategy):
    """A 股策略基类：所有用户策略继承本类，免费获得 T+1 / 涨跌停 / 风控 / 日志 能力。

    风控：可选注入一个 RiskManager（见 qkquant.risk）。若启用，在 safe_buy / safe_sell
    里会经过一层 check_buy / check_sell 否决，并可通过 self.apply_forced_exits() 在 next()
    开头扫描强平。
    """

    params = (
        ("risk_manager", None),
        ("adx_threshold", 0),        # ADX 趋势强度过滤：<N 不买入；0=关闭
        ("cooldown_days", 0),        # 卖出后 N 天内不重新买入同一只票；0=关闭
        ("adx_market_filter", False), # True=仅在 HS300<MA60 时空头发力 ADX，多头时关闭
        ("hs300_trend", None),       # 内部：引擎注入 {date: is_above_ma60}
    )

    def __init__(self) -> None:
        self._buy_dates: dict[str, date] = {}       # code -> 最近一次买入成交日期
        self._sell_dates: dict[str, date] = {}      # code -> 最近一次卖出成交日期（冷却期用）
        self._order_reasons: dict[int, str] = {}    # order.ref -> reason
        self._rejections: list[dict] = []           # 被规则拦截的订单
        self.risk: RiskManager | None = self.p.risk_manager

    # -------- A 股规则辅助 --------

    @staticmethod
    def _data_code(data) -> str:
        return data._name

    def _today(self) -> date:
        return self.datas[0].datetime.date(0)

    def _is_limit_up(self, data: PandasDataExt) -> bool:
        try:
            return float(data.pct_chg[0]) >= LIMIT_UP_THRESHOLD
        except Exception:
            return False

    def _is_limit_down(self, data: PandasDataExt) -> bool:
        try:
            return float(data.pct_chg[0]) <= LIMIT_DOWN_THRESHOLD
        except Exception:
            return False

    def _can_sell(self, data: PandasDataExt) -> bool:
        code = self._data_code(data)
        today = self._today()
        bought = self._buy_dates.get(code)
        if bought is not None and bought >= today:
            return False
        return not self._is_limit_down(data)

    def _can_buy(self, data: PandasDataExt) -> bool:
        return not self._is_limit_up(data)

    # -------- ADX 趋势强度过滤 --------

    def _adx_ok(self, data: PandasDataExt) -> bool:
        """ADX >= adx_threshold 才允许买入；阈值 ≤0 表示关闭过滤。

        若 adx_market_filter=True：仅在 HS300 空头排列（<MA60）时才启用 ADX 过滤，
        多头时跳过（趋势市中动量策略不需要 ADX 保护）。
        """
        threshold = self.p.adx_threshold
        if threshold <= 0:
            return True
        # 市场多头时关闭 ADX 过滤
        if self.p.adx_market_filter:
            trend = self.p.hs300_trend or {}
            today = self._today()
            if trend.get(today, True):  # 默认 True=多头，不拦截
                return True
        lookback = 42
        if len(data) < lookback:
            return True
        try:
            closes = pd.Series([float(data.close[-i]) for i in range(lookback - 1, -1, -1)])
            highs = pd.Series([float(data.high[-i]) for i in range(lookback - 1, -1, -1)])
            lows = pd.Series([float(data.low[-i]) for i in range(lookback - 1, -1, -1)])
            adx_df = compute_adx(highs, lows, closes)
            if adx_df.empty:
                return True
            val = adx_df["adx"].iloc[-1]
            if pd.isna(val):
                return True
            return float(val) >= threshold
        except Exception:
            return True  # 计算失败不阻塞

    # -------- 冷却期过滤 --------

    def _in_cooldown(self, code: str, today: date) -> bool:
        """卖出后 cooldown_days 天内禁止重新买入。"""
        if self.p.cooldown_days <= 0:
            return False
        last_sell = self._sell_dates.get(code)
        if last_sell is None:
            return False
        return today <= last_sell + timedelta(days=self.p.cooldown_days)

    # -------- 安全下单 --------

    def safe_buy(self, data: PandasDataExt, size: int, reason: str = "") -> bt.Order | None:
        if size <= 0:
            return None
        if not self._can_buy(data):
            self._rejections.append(
                {
                    "date": self._today(),
                    "code": self._data_code(data),
                    "side": "BUY",
                    "reason": "limit_up",
                }
            )
            return None
        if self.risk is not None:
            dec = self.risk.check_buy(self, data, size)
            if not dec.allowed:
                self._rejections.append(
                    {
                        "date": self._today(),
                        "code": self._data_code(data),
                        "side": "BUY",
                        "reason": f"risk:{dec.reason}",
                    }
                )
                return None
        order = self.buy(data=data, size=size)
        if order is not None:
            self._order_reasons[order.ref] = f"BUY:{reason}"
        return order

    def safe_sell(self, data: PandasDataExt, size: int, reason: str = "") -> bt.Order | None:
        if size <= 0:
            return None
        if not self._can_sell(data):
            self._rejections.append(
                {
                    "date": self._today(),
                    "code": self._data_code(data),
                    "side": "SELL",
                    "reason": "t1_or_limit_down",
                }
            )
            return None
        if self.risk is not None:
            dec = self.risk.check_sell(self, data, size)
            if not dec.allowed:
                self._rejections.append(
                    {
                        "date": self._today(),
                        "code": self._data_code(data),
                        "side": "SELL",
                        "reason": f"risk:{dec.reason}",
                    }
                )
                return None
        order = self.sell(data=data, size=size)
        if order is not None:
            self._order_reasons[order.ref] = f"SELL:{reason}"
        return order

    # -------- 风控 hook（子类在 next() 开头调用一次）--------

    def apply_forced_exits(self) -> None:
        """扫描风控强平；由子策略在 next() 开头调用。

        1. 调 risk.on_bar 更新组合峰值等
        2. 调 risk.forced_exits 获取应平仓列表，逐个触发 safe_sell
        """
        if self.risk is None:
            return
        self.risk.on_bar(self)
        exits = self.risk.forced_exits(self)
        if not exits:
            return
        data_by_code = {d._name: d for d in self.datas}
        for code, reason in exits:
            data = data_by_code.get(code)
            if data is None:
                continue
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            self.safe_sell(data, pos.size, reason=reason)

    # -------- backtrader 事件 --------

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed,):
            code = order.data._name
            today = self._today()
            if order.isbuy():
                self._buy_dates[code] = today
            else:
                self._sell_dates[code] = today
            if self.risk is not None:
                self.risk.on_order_filled(self, order)
            logger.debug(
                f"{self._today()} {code} {'BUY' if order.isbuy() else 'SELL'} "
                f"filled qty={order.executed.size} price={order.executed.price:.3f}"
            )
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            logger.debug(
                f"{self._today()} {order.data._name} order {order.Status[order.status]}"
            )


class BacktestEngine:
    """封装 backtrader 的 Cerebro：构建 feeds、注册策略、注册分析器、跑。"""

    def __init__(
        self,
        store: DuckStore,
        strategy_cls: type[BtStrategyBase],
        strategy_params: dict | None = None,
        initial_capital: float | None = None,
        slippage_pct: float | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.store = store
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params or {}
        cfg = get_settings().backtest
        self.initial_capital = initial_capital or cfg.initial_capital
        self.slippage_pct = slippage_pct if slippage_pct is not None else cfg.slippage_pct
        self.cfg = cfg
        self.risk_config = risk_config

    def run(
        self,
        codes: list[str],
        start: date | str,
        end: date | str,
        adjust: str = "qfq",
    ) -> dict:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(self.initial_capital)
        cerebro.broker.set_coc(False)           # 次开盘成交,而非今收盘
        cerebro.broker.set_coo(False)
        cerebro.broker.set_slippage_perc(self.slippage_pct, slip_open=True, slip_limit=True)

        comm = AShareCommission(
            commission_rate=self.cfg.commission_rate,
            commission_min=self.cfg.commission_min,
            stamp_tax=self.cfg.stamp_tax,
        )
        cerebro.broker.addcommissioninfo(comm)

        min_bars = max(
            30,
            self.strategy_params.get("slow", 20),
            self.strategy_params.get("trend_period", 0),
            self.strategy_params.get("boll_period", 0),
            self.strategy_params.get("mom_window", 0),
            self.strategy_params.get("lookback", 0),
            self.strategy_params.get("breakout_window", 0),
        ) + 1
        feeds = build_feeds(self.store, codes, start=start, end=end, adjust=adjust, min_bars=min_bars)
        if not feeds:
            raise RuntimeError(
                f"no feeds built for {len(codes)} codes in [{start}, {end}]; "
                "run `qkquant update-data` first"
            )
        for code, feed in feeds:
            cerebro.adddata(feed, name=code)

        strategy_kwargs = dict(self.strategy_params)
        # 注入市值数据（仅策略声明 market_caps 参数时传入）
        feed_codes = [c for c, _ in feeds]
        market_caps = self.store.load_market_caps(feed_codes)
        strategy_param_names = set(getattr(self.strategy_cls.params, "_getkeys", lambda: [])())
        if market_caps and "market_caps" in strategy_param_names:
            strategy_kwargs["market_caps"] = market_caps
        # HS300 趋势（仅在 adx_market_filter=True 时需要）
        if strategy_kwargs.get("adx_market_filter"):
            hs300_codes = self.store.load_index_constituents("000300")
            if hs300_codes:
                hs300_df = self.store.load_daily(
                    codes=hs300_codes, start=start, end=end, adjust=adjust
                )
                if not hs300_df.empty:
                    daily_close = hs300_df.groupby("trade_date")["close"].mean()
                    ma60 = daily_close.rolling(60, min_periods=1).mean()
                    strategy_kwargs["hs300_trend"] = (
                        (daily_close > ma60).to_dict()
                    )
        if self.risk_config is not None and self.risk_config.any_enabled():
            strategy_kwargs["risk_manager"] = RiskManager(self.risk_config)
        cerebro.addstrategy(self.strategy_cls, **strategy_kwargs)

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(
            bt.analyzers.SharpeRatio,
            _name="sharpe",
            timeframe=bt.TimeFrame.Days,
            riskfreerate=0.02,
            annualize=True,
        )
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addobserver(bt.observers.Broker)
        cerebro.addobserver(bt.observers.Trades)

        logger.info(
            f"backtest start: codes={len(feeds)}, period=[{start}, {end}], "
            f"capital={self.initial_capital}"
        )
        t0 = datetime.now()
        results = cerebro.run()
        elapsed = (datetime.now() - t0).total_seconds()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        logger.info(
            f"backtest done in {elapsed:.1f}s; final_value={final_value:.2f} "
            f"(pnl={final_value - self.initial_capital:+.2f})"
        )

        return {
            "strategy": strat,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "elapsed_seconds": elapsed,
            "analyzers": {
                "timereturn": strat.analyzers.timereturn.get_analysis(),
                "drawdown": strat.analyzers.drawdown.get_analysis(),
                "sharpe": strat.analyzers.sharpe.get_analysis(),
                "trades": strat.analyzers.trades.get_analysis(),
                "returns": strat.analyzers.returns.get_analysis(),
            },
            "rejections": getattr(strat, "_rejections", []),
            "period": {"start": str(start), "end": str(end)},
            "codes": [c for c, _ in feeds],
        }


__all__ = ["BacktestEngine", "BtStrategyBase", "AShareCommission"]
