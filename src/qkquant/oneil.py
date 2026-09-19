"""O'Neil-only, causal daily research model (no Chan or ETF-score dependency).

The account holds fractional adjusted-price units. Raw prices check next-open
gaps/limits; this is an adjusted-price experiment, not a stock-lot execution
simulator. Historical financial vintages and corporate-action cash ledgers are
not available in the pilot snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OneilConfig:
    base_days: int = 25
    max_base_depth: float = .15
    prior_trend_days: int = 60
    prior_gain: float = .20
    volume_multiple: float = 1.4
    rs_days: int = 252
    rs_percentile: float = .80
    min_amount: float = 100_000_000
    quarterly_growth: float = .25
    annual_cagr: float = .25
    min_roe: float = 17
    buy_zone: float = .05
    stop_loss: float = .08
    take_profit: float = .20
    fast_rise_days: int = 15
    hold_days: int = 40
    risk_per_trade: float = .005
    max_weight: float = .15
    max_positions: int = 5
    capital: float = 100_000
    commission: float = .00025
    minimum_commission: float = 5
    slippage: float = .002

    def __post_init__(self):
        if not (0 < self.stop_loss < 1 and 0 < self.risk_per_trade < 1
                and 0 < self.max_weight <= 1 and self.capital > 0):
            raise ValueError("invalid capital/risk parameters")
        if min(self.base_days, self.prior_trend_days, self.rs_days, self.max_positions) < 1:
            raise ValueError("lookbacks and max_positions must be positive")
        if min(self.commission, self.minimum_commission, self.slippage) < 0:
            raise ValueError("costs must be nonnegative")


def financial_state(rows: pd.DataFrame, day, cfg: OneilConfig, policy="updated") -> dict:
    """Require current-quarter EPS/sales and four consecutive annual reports.

    Date-only publications become usable the following calendar day. Latest
    announced reports cannot be silently replaced with a stale passing report.
    """
    if policy not in ("updated", "notice"):
        raise ValueError("financial policy must be updated or notice")
    day = pd.Timestamp(day).normalize()
    if rows.empty:
        return {"fundamental_ok": False, "reason": "missing_financials"}
    rows = rows.copy()
    for key in ("report_date", "notice_date", "update_date"):
        rows[key] = pd.to_datetime(rows[key]).dt.normalize()
    known = rows[(rows.notice_date < day) & (rows.report_date < day)]
    quarter = known[known.kind == "quarter"].sort_values("report_date")
    annual = known[known.kind == "annual"].sort_values("report_date")
    if quarter.empty or len(annual) < 4:
        return {"fundamental_ok": False, "reason": "insufficient_history"}
    q = quarter.iloc[-1]
    previous_date = q.report_date - pd.DateOffset(years=1)
    previous = quarter[quarter.report_date == previous_date]
    years = annual.tail(4)
    if previous.empty or years.report_date.dt.year.diff().dropna().ne(1).any():
        return {"fundamental_ok": False, "reason": "missing_comparison_period"}
    if (day - q.report_date).days > 200 or (day - years.report_date.iloc[-1]).days > 550:
        return {"fundamental_ok": False, "reason": "stale_financials"}
    required = pd.concat([quarter.tail(1), previous.tail(1), years])
    if policy == "updated" and (required.update_date.isna().any()
                                or (required.update_date >= day).any()):
        return {"fundamental_ok": False, "reason": "revision_not_available"}
    p = previous.iloc[-1]
    eps = pd.to_numeric(years.eps, errors="coerce").to_numpy()
    values = [q.eps, p.eps, q.revenue, p.revenue, years.roe.iloc[-1], *eps]
    if not np.isfinite(np.asarray(values, dtype=float)).all() or min(p.eps, p.revenue, *eps) <= 0:
        return {"fundamental_ok": False, "reason": "invalid_or_nonpositive_financials"}
    eps_growth, sales_growth = q.eps / p.eps - 1, q.revenue / p.revenue - 1
    cagr = (eps[-1] / eps[0]) ** (1 / 3) - 1
    checks = {"quarter_eps_ok": bool(eps_growth >= cfg.quarterly_growth),
              "quarter_sales_ok": bool(sales_growth >= cfg.quarterly_growth),
              "annual_growth_ok": bool(cagr >= cfg.annual_cagr and np.all(np.diff(eps) > 0)),
              "roe_ok": bool(years.roe.iloc[-1] >= cfg.min_roe)}
    passed = all(checks.values())
    return {"fundamental_ok": bool(passed), "reason": "pass" if passed else "growth_filter",
            **checks,
            "quarter_eps_growth": eps_growth, "quarter_sales_growth": sales_growth,
            "annual_eps_cagr": cagr, "annual_roe": years.roe.iloc[-1],
            "report_date": q.report_date, "latest_required_notice": required.notice_date.max(),
            "latest_required_update": required.update_date.max()}


def prepare_prices(bars: pd.DataFrame, benchmark: pd.DataFrame, cfg: OneilConfig) -> dict:
    if bars.duplicated(["trade_date", "code"]).any():
        raise ValueError("duplicate stock/date bars")
    if benchmark.trade_date.duplicated().any():
        raise ValueError("duplicate benchmark dates")
    calendar = pd.DatetimeIndex(benchmark.trade_date).sort_values()
    fields = ["open", "high", "low", "close", "volume", "amount",
              "raw_open", "raw_high", "raw_low", "raw_close"]
    panel = {f: bars.pivot(index="trade_date", columns="code", values=f).reindex(calendar)
             for f in fields}
    close, high, low, volume = (panel[k] for k in ("close", "high", "low", "volume"))
    if ((close <= 0) | (high < close) | (low > close) | (low <= 0)).any().any():
        raise ValueError("invalid adjusted OHLC")
    pivot = high.shift(1).rolling(cfg.base_days, min_periods=cfg.base_days).max()
    bottom = low.shift(1).rolling(cfg.base_days, min_periods=cfg.base_days).min()
    depth = 1 - bottom / pivot
    prior = close.shift(cfg.base_days) / close.shift(cfg.base_days + cfg.prior_trend_days) - 1
    eligible = (bars.pivot(index="trade_date", columns="code", values="eligible").reindex(
        index=calendar, columns=close.columns).astype("boolean").fillna(False).astype(bool)
        if "eligible" in bars else pd.DataFrame(True, index=calendar, columns=close.columns))
    rs = close.pct_change(cfg.rs_days, fill_method=None).where(eligible).rank(axis=1, pct=True)
    avg_volume = volume.shift(1).rolling(50, min_periods=50).mean()
    amount = panel["amount"].shift(1).rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    base_ok = ((depth <= cfg.max_base_depth) & (depth > 0) & (prior >= cfg.prior_gain))
    breakout = ((close > pivot) & (close.shift(1) <= pivot)
                & (close <= pivot * (1 + cfg.buy_zone)))
    technical = (base_ok & breakout & (volume >= avg_volume * cfg.volume_multiple)
                 & (rs >= cfg.rs_percentile) & (amount >= cfg.min_amount) & (close > ma50) & eligible)
    index = benchmark.set_index("trade_date").raw_close.reindex(calendar)
    ma60 = index.rolling(60, min_periods=60).mean()
    market_ok = ((index > ma60) & (ma60 > ma60.shift(5))).fillna(False)
    return {**panel, "pivot": pivot, "rs": rs, "ma50": ma50, "avg_volume": avg_volume,
            "technical": technical, "market_ok": market_ok, "benchmark": index,
            "valuation": close.ffill(), "raw_previous": panel["raw_close"].ffill().shift(1)}


def build_signals(prepared: dict, financials: pd.DataFrame, cfg: OneilConfig, policy="updated") -> pd.DataFrame:
    if financials.duplicated(["code", "kind", "report_date"]).any():
        raise ValueError("snapshot must contain one financial vintage per report")
    by_code = dict(tuple(financials.groupby("code")))
    technical = prepared["technical"]
    rows = []
    for i, j in zip(*np.where(technical.to_numpy()), strict=True):
        day, code = technical.index[i], technical.columns[j]
        state = financial_state(by_code.get(code, financials.iloc[:0]), day, cfg, policy)
        market = bool(prepared["market_ok"].iloc[i])
        rows.append({"signal_date": day, "code": code, "pivot": prepared["pivot"].iloc[i, j],
                     "rs": prepared["rs"].iloc[i, j], "market_ok": market, **state,
                     "qualified": market and state["fundamental_ok"]})
    columns = ["signal_date", "code", "pivot", "rs", "market_ok", "fundamental_ok", "reason", "qualified"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def performance(equity: pd.Series) -> dict:
    returns = equity.pct_change().dropna()
    years = max(len(returns) / 252, 1 / 252)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    std = returns.std()
    return {"total_return": float(total), "annualized_return": float((1 + total) ** (1 / years) - 1),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "sharpe": float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0}


def run_backtest(prepared: dict, signals: pd.DataFrame, cfg: OneilConfig,
                 start="2024-07-01", end=None) -> dict:
    """Close decisions, following-session entries; intraday stops only after T+1.

    A stopped/blocked sale remains pending. Mark missing bars at last known
    adjusted close. Eight-week holding never overrides a protective stop.
    """
    calendar = prepared["close"].index
    active = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end or calendar[-1]))]
    if len(active) < 2:
        raise ValueError("backtest needs at least two benchmark sessions")
    events = {day: group.sort_values(["rs", "code"], ascending=[False, True])
              for day, group in signals[signals.qualified.astype(bool)].groupby("signal_date")}
    cash, holdings, trades, rejects, equity = cfg.capital, {}, [], [], []

    def commission(notional):
        return max(cfg.minimum_commission, notional * cfg.commission)

    for day in active:
        i = calendar.get_loc(day)
        previous_day = calendar[i - 1] if i else None
        row = {key: value.iloc[i] for key, value in prepared.items() if isinstance(value, pd.DataFrame)}
        opened_today = set()
        sold_today = set()

        def reject(code, side, reason, day=day):
            rejects.append({"date": day, "code": code, "side": side, "reason": reason})

        def executable(code, side, raw_price=None, row=row):
            price = row["raw_open"][code] if raw_price is None else raw_price
            prev = row["raw_previous"][code]
            if not np.isfinite([price, prev, row["open"][code]]).all() or min(price, prev, row["open"][code]) <= 0:
                return False
            if not np.isfinite(row["volume"][code]) or row["volume"][code] <= 0:
                return False
            # Conservative main-board opening/stop-price limit approximation.
            bound = round(prev * (1.1 if side == "BUY" else .9), 2)
            return price < bound - .001 if side == "BUY" else price > bound + .001

        def sell(code, price, reason, day=day, i=i, sold_today=sold_today):
            nonlocal cash
            position = holdings.pop(code)
            proceeds = position["units"] * price
            tax = .0005 if day >= pd.Timestamp("2023-08-28") else .001
            fee = commission(proceeds) + proceeds * (tax + .00001)
            cash += proceeds - fee
            trades.append({"date": day, "code": code, "side": "SELL", "reason": reason,
                           "price_adjusted": price, "units": position["units"], "notional": proceeds,
                           "fees": fee, "pnl": proceeds - fee - position["cost"],
                           "signal_date": position["signal_date"], "entry_date": position["entry_date"],
                           "holding_sessions": i - position["entry_i"], "eight_week_hold": position["hold8"]})
            sold_today.add(code)

        # Decisions made at yesterday's close execute before today's entries.
        for code, position in list(holdings.items()):
            reason = position.get("pending_exit")
            if reason:
                if executable(code, "SELL"):
                    sell(code, row["open"][code] * (1 - cfg.slippage), reason)
                else:
                    reject(code, "SELL", "untradable_pending_exit")

        # A signal belongs only to the immediately following benchmark session.
        if day != active[0] and previous_day in events:
            for signal in events[previous_day].itertuples(index=False):
                code = signal.code
                if code in holdings or code in sold_today:
                    continue
                if len(holdings) >= cfg.max_positions:
                    reject(code, "BUY", "position_limit")
                    continue
                if not executable(code, "BUY"):
                    reject(code, "BUY", "missing_bar_or_limit_up")
                    continue
                price = row["open"][code] * (1 + cfg.slippage)
                if not signal.pivot <= price <= signal.pivot * (1 + cfg.buy_zone):
                    reject(code, "BUY", "outside_buy_zone")
                    continue
                # Only opening prices or yesterday's valuation may size an order.
                opening_equity = cash + sum(p["units"] * (
                    row["open"][c] if np.isfinite(row["open"][c])
                    else prepared["valuation"].iloc[i - 1][c]) for c, p in holdings.items())
                budget = opening_equity * min(cfg.max_weight, cfg.risk_per_trade / cfg.stop_loss)
                notional = min(budget, (cash - cfg.minimum_commission) / 1.00001,
                               cash / (1 + cfg.commission + .00001))
                if notional <= 0 or notional < cfg.minimum_commission:
                    reject(code, "BUY", "insufficient_cash")
                    continue
                fee = commission(notional) + notional * .00001
                cash -= notional + fee
                holdings[code] = {"units": notional / price, "entry_price": price,
                                  "cost": notional + fee, "entry_date": day, "entry_i": i,
                                  "signal_date": previous_day, "signal_i": i - 1,
                                  "pivot": signal.pivot, "hold8": False}
                opened_today.add(code)
                trades.append({"date": day, "code": code, "side": "BUY", "reason": "flat_base_breakout",
                               "price_adjusted": price, "units": notional / price, "notional": notional,
                               "fees": fee, "signal_date": previous_day})

        for code, position in list(holdings.items()):
            stop = position["entry_price"] * (1 - cfg.stop_loss)
            low, close = row["low"][code], row["close"][code]
            if np.isfinite(low) and low <= stop:
                position["pending_exit"] = "stop_loss"
                if code in opened_today:
                    reject(code, "SELL", "t_plus_one_stop_deferred")
                else:
                    price = min(row["open"][code], stop)
                    raw_price = price * row["raw_open"][code] / row["open"][code]
                    if executable(code, "SELL", raw_price):
                        sell(code, price * (1 - cfg.slippage), "stop_loss")
                    else:
                        reject(code, "SELL", "untradable_stop")
                continue
            if not np.isfinite(close) or position.get("pending_exit"):
                continue
            age = i - position["signal_i"]
            if age < cfg.fast_rise_days and close >= position["pivot"] * (1 + cfg.take_profit):
                position["hold8"] = True
            hold_active = position["hold8"] and age < cfg.hold_days
            distribution = (close < row["ma50"][code]
                            and row["volume"][code] >= row["avg_volume"][code] * cfg.volume_multiple)
            if distribution:
                position["pending_exit"] = "heavy_volume_ma50_break"
            elif not hold_active and close >= position["entry_price"] * (1 + cfg.take_profit):
                position["pending_exit"] = "take_profit"

        invested = sum(p["units"] * row["valuation"][code] for code, p in holdings.items())
        if cash < -1e-7:
            raise AssertionError("cash must not become negative")
        equity.append({"date": day, "equity": cash + invested, "cash": cash,
                       "exposure": invested / (cash + invested), "positions": len(holdings)})

    curve = pd.DataFrame(equity).set_index("date")
    executions = pd.DataFrame(trades, columns=["date", "code", "side", "reason", "price_adjusted",
                                              "units", "notional", "fees", "pnl", "signal_date",
                                              "entry_date", "holding_sessions", "eight_week_hold"])
    closed = executions[executions.side == "SELL"]
    stats = {**performance(curve.equity), "buy_count": int((executions.side == "BUY").sum()),
             "closed_trades": len(closed), "open_positions": len(holdings),
             "win_rate": float((closed.pnl > 0).mean()) if len(closed) else None,
             "average_exposure": float(curve.exposure.mean()), "fees": float(executions.fees.sum()),
             "start": str(active[0].date()), "end": str(active[-1].date())}
    return {"metrics": stats, "equity": curve, "trades": executions,
            "rejections": pd.DataFrame(rejects, columns=["date", "code", "side", "reason"]),
            "holdings": holdings, "config": asdict(cfg)}
