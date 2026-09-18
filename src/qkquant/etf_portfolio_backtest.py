"""Portfolio backtest for the ETF score + risk regime + Chan third-buy strategy."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.etf_chan import classify_chan_structure
from qkquant.etf_signal import EtfSignalConfig, theme_key
from qkquant.factors.library import trend_quality_60d
from qkquant.factors.pipeline import load_panel


def _metrics(equity: pd.Series) -> dict:
    returns = equity.pct_change(fill_method=None).dropna()
    years = max(len(returns) / 252, 1 / 252)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annual = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    drawdown = equity / equity.cummax() - 1
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    return {"total_return": total, "annualized_return": annual, "max_drawdown": float(drawdown.min()), "annualized_volatility": volatility, "sharpe": sharpe}


def run_etf_portfolio_backtest(
    store: DuckStore,
    start: str = "2022-01-01",
    end: str | None = None,
    category: str = "equity",
    rebalance_days: int = 5,
    config: EtfSignalConfig | None = None,
) -> dict:
    cfg = config or EtfSignalConfig()
    codes = store.load_etf_codes(category)
    panel = load_panel(store, codes=codes, start=start, end=end)
    open_, high, low, close, amount = (panel[key] for key in ("open", "high", "low", "close", "amount"))
    ret = close.pct_change(fill_method=None)
    mom60, mom120 = close.pct_change(60, fill_method=None), close.pct_change(120, fill_method=None)
    risk_mom = mom60 / ret.rolling(60, min_periods=60).std().replace(0, np.nan)
    trend = trend_quality_60d(panel)
    downside = ret.clip(upper=0).rolling(20, min_periods=20).std()
    ma120 = close.rolling(120, min_periods=120).mean()
    amount20 = amount.rolling(20, min_periods=20).mean()
    instruments = store.load_instruments(codes).set_index("code")

    cash = float(cfg.capital)
    holdings: dict[str, int] = {}
    pending: tuple[list[str], float] | None = None
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    exposure_rows: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    turnover = 0.0
    start_pos = 120

    for pos in range(start_pos, len(close)):
        day = close.index[pos]
        if pending is not None:
            selected, exposure = pending
            valid_open = open_.iloc[pos]
            mark = close.iloc[pos - 1]
            equity_before = cash + sum(qty * float(mark.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(mark.get(code, np.nan)))
            target_weight = exposure / len(selected) if selected else 0.0
            targets: dict[str, int] = {}
            for code in selected:
                price = valid_open.get(code)
                if pd.isna(price):
                    continue
                lot = int(instruments.loc[code].get("lot_size") or 100)
                targets[code] = int(equity_before * target_weight / float(price) / lot) * lot

            # Sell first so proceeds are available for buys.
            for code in sorted(set(holdings) | set(targets)):
                current, target = holdings.get(code, 0), targets.get(code, 0)
                if target >= current:
                    continue
                raw = valid_open.get(code)
                if pd.isna(raw):
                    continue
                qty = current - target
                price = float(raw) * (1 - cfg.slippage_pct)
                value = qty * price
                commission = max(value * cfg.commission_rate, cfg.commission_min)
                cash += value - commission
                turnover += value
                trades.append({"date": str(day.date()), "code": code, "side": "SELL", "qty": qty, "price": price, "commission": commission})
                if target:
                    holdings[code] = target
                else:
                    holdings.pop(code, None)
            for code in selected:
                current, target = holdings.get(code, 0), targets.get(code, 0)
                if target <= current:
                    continue
                raw = valid_open.get(code)
                if pd.isna(raw):
                    continue
                lot = int(instruments.loc[code].get("lot_size") or 100)
                price = float(raw) * (1 + cfg.slippage_pct)
                affordable = int(max(cash - cfg.commission_min, 0) / price / lot) * lot
                qty = min(target - current, affordable)
                if qty <= 0:
                    continue
                value = qty * price
                commission = max(value * cfg.commission_rate, cfg.commission_min)
                cash -= value + commission
                turnover += value
                holdings[code] = current + qty
                trades.append({"date": str(day.date()), "code": code, "side": "BUY", "qty": qty, "price": price, "commission": commission})
            pending = None

        marks = close.iloc[pos]
        equity = cash + sum(qty * float(marks.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(marks.get(code, np.nan)))
        equity_rows.append((day, equity))
        invested = sum(qty * float(marks.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(marks.get(code, np.nan)))
        exposure_rows.append((day, invested / equity if equity > 0 else 0.0))

        if pos >= len(close) - 1 or (pos - start_pos) % rebalance_days:
            continue
        breadth_mask = close.iloc[pos].notna() & ma120.iloc[pos].notna()
        breadth = float((close.iloc[pos][breadth_mask] > ma120.iloc[pos][breadth_mask]).mean())
        exposure = 1.0 if breadth >= cfg.risk_on_breadth else cfg.neutral_exposure if breadth >= cfg.risk_off_breadth else 0.0
        frame = pd.DataFrame({
            "mom60": mom60.iloc[pos], "mom120": mom120.iloc[pos], "risk_mom": risk_mom.iloc[pos],
            "trend": trend.iloc[pos], "downside": downside.iloc[pos], "amount20": amount20.iloc[pos],
            "above_ma120": close.iloc[pos] > ma120.iloc[pos],
        }).dropna()
        frame = frame[(frame.mom60 > 0) & (frame.mom120 > 0) & frame.above_ma120 & (frame.amount20 >= cfg.min_amount_20d)].copy()
        frame["score"] = 0.35 * frame.risk_mom.rank(pct=True) + 0.25 * frame.mom120.rank(pct=True) + 0.20 * frame.trend.rank(pct=True) + 0.20 * (1 - frame.downside.rank(pct=True))
        frame = frame[frame.score >= cfg.score_threshold].sort_values("score", ascending=False)
        selected: list[str] = []
        themes: set[str] = set()
        corr_data = ret.iloc[max(0, pos - cfg.corr_window + 1) : pos + 1]
        if exposure:
            for code in frame.index:
                chan = classify_chan_structure(close[code].iloc[: pos + 1], high[code].iloc[: pos + 1], low[code].iloc[: pos + 1], cfg.chan_fractal_order, cfg.chan_tolerance)
                if cfg.chan_filter_enabled and chan.state not in cfg.chan_entry_states:
                    continue
                info = instruments.loc[code]
                theme = theme_key(str(info.get("name", code)), info.get("benchmark_code"))
                if theme in themes:
                    continue
                if selected:
                    correlations = corr_data[selected + [code]].corr()[code].drop(code)
                    if len(correlations) and correlations.max() > cfg.corr_limit:
                        continue
                selected.append(code)
                themes.add(theme)
                if len(selected) >= cfg.max_positions:
                    break
        pending = (selected, exposure)

    equity_curve = pd.Series(dict(equity_rows), dtype=float).sort_index()
    exposure_curve = pd.Series(dict(exposure_rows), dtype=float).sort_index()
    benchmark_returns = ret.loc[equity_curve.index].mean(axis=1, skipna=True).fillna(0)
    benchmark = cfg.capital * (1 + benchmark_returns).cumprod()
    return {
        "start": str(equity_curve.index[0].date()), "end": str(equity_curve.index[-1].date()),
        "config": asdict(cfg), "strategy": _metrics(equity_curve), "benchmark": _metrics(benchmark),
        "trade_count": len(trades), "turnover_ratio": float(turnover / equity_curve.mean()),
        "commission_total": float(sum(t["commission"] for t in trades)),
        "equity": equity_curve, "benchmark_equity": benchmark, "exposure": exposure_curve,
        "trades": trades, "attribution": _attribution(equity_curve, exposure_curve, trades),
    }


def _attribution(equity: pd.Series, exposure: pd.Series, trades: list[dict]) -> dict:
    year_end = equity.resample("YE").last()
    previous = pd.concat([pd.Series([equity.iloc[0]], index=[equity.index[0] - pd.Timedelta(days=1)]), year_end])
    yearly = {str(idx.year): float(value) for idx, value in previous.pct_change(fill_method=None).dropna().items()}
    drawdown = equity / equity.cummax() - 1
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()

    books: dict[str, dict] = {}
    realized: dict[str, float] = {}
    round_trips: list[dict] = []
    for trade in trades:
        code, qty = trade["code"], int(trade["qty"])
        book = books.setdefault(code, {"qty": 0, "cost": 0.0, "entry_date": trade["date"]})
        if trade["side"] == "BUY":
            if book["qty"] == 0:
                book["entry_date"] = trade["date"]
            book["qty"] += qty
            book["cost"] += qty * trade["price"] + trade["commission"]
            continue
        if book["qty"] <= 0:
            continue
        sold = min(qty, book["qty"])
        allocated_cost = book["cost"] * sold / book["qty"]
        proceeds = sold * trade["price"] - trade["commission"]
        pnl = proceeds - allocated_cost
        realized[code] = realized.get(code, 0.0) + pnl
        round_trips.append({"code": code, "entry_date": book["entry_date"], "exit_date": trade["date"], "pnl": pnl})
        book["qty"] -= sold
        book["cost"] -= allocated_cost
        if book["qty"] == 0:
            book["cost"] = 0.0
    best = sorted(realized.items(), key=lambda item: item[1], reverse=True)[:5]
    worst = sorted(realized.items(), key=lambda item: item[1])[:5]
    wins = [row for row in round_trips if row["pnl"] > 0]
    return {
        "yearly_returns": yearly,
        "drawdown_peak": str(peak.date()), "drawdown_trough": str(trough.date()),
        "average_exposure": float(exposure.mean()), "cash_day_ratio": float((exposure < 0.05).mean()),
        "realized_pnl": float(sum(realized.values())), "round_trip_count": len(round_trips),
        "round_trip_win_rate": float(len(wins) / len(round_trips)) if round_trips else None,
        "best_codes": best, "worst_codes": worst,
    }


def format_portfolio_backtest(result: dict) -> str:
    s, b = result["strategy"], result["benchmark"]
    a = result["attribution"]
    lines = [
        f"ETF 组合回测 | {result['start']} ~ {result['end']}",
        "指标             策略        ETF等权基准",
        f"累计收益      {s['total_return']:>9.2%}    {b['total_return']:>9.2%}",
        f"年化收益      {s['annualized_return']:>9.2%}    {b['annualized_return']:>9.2%}",
        f"最大回撤      {s['max_drawdown']:>9.2%}    {b['max_drawdown']:>9.2%}",
        f"年化波动      {s['annualized_volatility']:>9.2%}    {b['annualized_volatility']:>9.2%}",
        f"夏普比率      {s['sharpe']:>9.2f}    {b['sharpe']:>9.2f}",
        f"成交次数: {result['trade_count']} | 总佣金: {result['commission_total']:.2f} | 累计换手: {result['turnover_ratio']:.2f} 倍",
        f"平均仓位: {a['average_exposure']:.1%} | 近空仓天数: {a['cash_day_ratio']:.1%} | 已实现胜率: {a['round_trip_win_rate']:.1%}" if a["round_trip_win_rate"] is not None else "无已实现交易",
        f"最深回撤区间: {a['drawdown_peak']} -> {a['drawdown_trough']}",
        "年度收益: " + " | ".join(f"{year} {value:.2%}" for year, value in a["yearly_returns"].items()),
        "盈利贡献: " + " | ".join(f"{code} {pnl:.0f}" for code, pnl in a["best_codes"]),
        "亏损贡献: " + " | ".join(f"{code} {pnl:.0f}" for code, pnl in a["worst_codes"]),
    ]
    return "\n".join(lines)


__all__ = ["format_portfolio_backtest", "run_etf_portfolio_backtest"]
