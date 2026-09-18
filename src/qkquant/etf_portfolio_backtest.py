"""Portfolio backtest for the ETF score + risk regime + Chan third-buy strategy."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.etf_drawdown import DrawdownConfig, DrawdownState
from qkquant.etf_chan import chan_selection_allowed, classify_chan_structure
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


def prepare_inputs(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute causal signals without filling or removing missing sessions/codes."""
    close = panel["close"]
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("panel dates must be unique and increasing")
    if not close.columns.is_unique:
        raise ValueError("panel codes must be unique")
    for key in ("open", "high", "low", "amount"):
        if not panel[key].index.equals(close.index) or not panel[key].columns.equals(close.columns):
            raise ValueError(f"panel {key} must match close dates and codes")
    ret = close.pct_change(fill_method=None)
    mom60 = close.pct_change(60, fill_method=None)
    return {
        **panel,
        "ret": ret,
        "mom60": mom60,
        "mom120": close.pct_change(120, fill_method=None),
        "risk_mom": mom60 / ret.rolling(60, min_periods=60).std().replace(0, np.nan),
        "trend": trend_quality_60d(panel),
        "downside": ret.clip(upper=0).rolling(20, min_periods=20).std(),
        "ma120": close.rolling(120, min_periods=120).mean(),
        "amount20": panel["amount"].rolling(20, min_periods=20).mean(),
        "valuation": close.ffill(),
    }


def close_decision(
    prepared: dict[str, pd.DataFrame],
    instruments: pd.DataFrame,
    pos: int,
    holdings: dict[str, int],
    equity: float,
    invested: float,
    state: DrawdownState,
    cfg: EtfSignalConfig,
    risk_cfg: DrawdownConfig | None = None,
    rebalance_days: int = 10,
    rebalance_offset: int = 0,
    daily_entries: bool = False,
) -> dict:
    """Update close risk state and decide targets for the next session only.

    ``state`` is updated in place. Daily callers supply recorded account equity
    and invested value in raw currency; adjusted prices are used only for signals.
    The returned pending tuple is (codes, exposure, reduce_only, date, entry_only).
    """
    if rebalance_days < 1 or not 0 <= rebalance_offset < rebalance_days:
        raise ValueError("require rebalance_days >= 1 and 0 <= rebalance_offset < rebalance_days")
    if daily_entries and (not cfg.chan_filter_enabled or cfg.chan_entry_states != ("third_buy",)):
        raise ValueError("daily_entries research requires fresh third_buy filtering")
    close, high, low = (prepared[key] for key in ("close", "high", "low"))
    if pos < 120 or pos >= len(close):
        raise ValueError("decision position must follow 120 warmup bars and exist in panel")
    if not np.isfinite(equity) or equity <= 0 or not np.isfinite(invested) or invested < 0:
        raise ValueError("account equity must be positive and invested value nonnegative")
    ma120, ret = prepared["ma120"], prepared["ret"]
    day = close.index[pos]
    scheduled = (pos - 120) % rebalance_days == rebalance_offset
    breadth_mask = close.iloc[pos].notna() & ma120.iloc[pos].notna()
    breadth = float((close.iloc[pos][breadth_mask] > ma120.iloc[pos][breadth_mask]).mean())
    regime = "risk_on" if breadth >= cfg.risk_on_breadth else "neutral" if breadth >= cfg.risk_off_breadth else "risk_off"
    market_exposure = 1.0 if regime == "risk_on" else cfg.neutral_exposure if regime == "neutral" else 0.0
    decision = {"pending": None, "continuing": set(), "scheduled": scheduled,
                "date": str(day.date()), "position": pos, "breadth": breadth,
                "regime": regime, "exposure_cap": 1.0, "exposure": market_exposure,
                "risk_row": None, "action": "no_action"}
    if risk_cfg:
        reduced = state.update(equity, pos, scheduled, breadth >= cfg.risk_on_breadth, risk_cfg)
        cap = risk_cfg.max_exposure * state.multiplier
        decision["exposure_cap"] = cap
        decision["exposure"] *= cap
        decision["risk_row"] = {"date": day, "multiplier": state.multiplier,
                                "exposure_cap": cap, "drawdown": 1 - equity / state.peak,
                                "halted": state.halted}
        # Risk sales override a scheduled rebalance too; never replace this with
        # a selection containing new buys. Retry blocked reductions on every day.
        if holdings and (reduced or invested / equity > cap + 0.005):
            decision["exposure"] = min(cap, invested / equity)
            decision["pending"] = (list(holdings), decision["exposure"], True, str(day.date()), False)
            decision["action"] = "risk_reduce"
            return decision
    if not scheduled and (not daily_entries or len(holdings) >= cfg.max_positions):
        return decision
    exposure = decision["exposure"]
    frame = pd.DataFrame({
        key: prepared[key].iloc[pos]
        for key in ("mom60", "mom120", "risk_mom", "trend", "downside", "amount20")
    })
    frame["above_ma120"] = close.iloc[pos] > ma120.iloc[pos]
    frame = frame.dropna()
    frame = frame[(frame.mom60 > 0) & (frame.mom120 > 0) & frame.above_ma120 & (frame.amount20 >= cfg.min_amount_20d)].copy()
    frame["score"] = 0.35 * frame.risk_mom.rank(pct=True) + 0.25 * frame.mom120.rank(pct=True) + 0.20 * frame.trend.rank(pct=True) + 0.20 * (1 - frame.downside.rank(pct=True))
    frame = frame[frame.score >= cfg.score_threshold].sort_values("score", ascending=False)
    selected: list[str] = list(holdings) if not scheduled else []
    continuing: set[str] = set()
    themes = {theme_key(str(instruments.loc[code].get("name", code)), instruments.loc[code].get("benchmark_code")) for code in selected}
    corr_data = ret.iloc[max(0, pos - cfg.corr_window + 1): pos + 1]
    if exposure:
        for code in frame.index:
            if not scheduled and code in holdings:
                continue
            chan = classify_chan_structure(close[code].iloc[: pos + 1], high[code].iloc[: pos + 1], low[code].iloc[: pos + 1], cfg.chan_fractal_order, cfg.chan_tolerance)
            if cfg.chan_filter_enabled and not chan_selection_allowed(chan, cfg.chan_entry_states, holdings.get(code, 0) > 0):
                continue
            if cfg.chan_filter_enabled and chan.state == "third_buy_active":
                continuing.add(code)
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
    decision["continuing"] = continuing
    if scheduled or set(selected) - set(holdings):
        decision["pending"] = (selected, exposure, False, str(day.date()), not scheduled)
        decision["action"] = "rebalance" if scheduled else "daily_entry"
    return decision


def run_etf_portfolio_backtest(
    store: DuckStore,
    start: str = "2022-01-01",
    end: str | None = None,
    category: str = "equity",
    rebalance_days: int = 5,
    config: EtfSignalConfig | None = None,
    rebalance_offset: int = 0,
    risk_config: DrawdownConfig | None = None,
    daily_entries: bool = False,
    panel: dict[str, pd.DataFrame] | None = None,
) -> dict:
    if rebalance_days < 1 or not 0 <= rebalance_offset < rebalance_days:
        raise ValueError("require rebalance_days >= 1 and 0 <= rebalance_offset < rebalance_days")
    cfg = config or EtfSignalConfig()
    if daily_entries and (not cfg.chan_filter_enabled or cfg.chan_entry_states != ("third_buy",)):
        raise ValueError("daily_entries research requires fresh third_buy filtering")
    if (cfg.capital <= 0 or cfg.max_positions < 1 or cfg.commission_rate < 0
            or cfg.commission_min < 0 or not 0 <= cfg.slippage_pct < 1):
        raise ValueError("invalid capital, positions or execution costs")
    if panel is None:
        panel = load_panel(store, codes=store.load_etf_codes(category), start=start, end=end)
    prepared = prepare_inputs(panel)
    open_, close, amount, ret, valuation = (prepared[key] for key in ("open", "close", "amount", "ret", "valuation"))
    codes = list(close.columns)
    instruments = store.load_instruments(codes).set_index("code")
    missing_instruments = close.columns[close.notna().any()].difference(instruments.index)
    if len(missing_instruments):
        raise ValueError(f"missing instrument metadata for {list(missing_instruments)}")
    if len(close) < 122:
        raise ValueError("at least 122 daily bars required including 120 warmup bars")

    cash = float(cfg.capital)
    holdings: dict[str, int] = {}
    continuing: set[str] = set()
    pending: tuple[list[str], float, bool, str, bool] | None = None
    state = DrawdownState(cfg.capital, cfg.capital)
    risk_rows: list[dict] = []
    cash_rows: list[tuple[pd.Timestamp, float]] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    exposure_rows: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    turnover = 0.0
    start_pos = 120

    for pos in range(start_pos, len(close)):
        day = close.index[pos]
        if pending is not None:
            selected, exposure, reduce_only, signal_day, entry_only = pending
            valid_open = open_.iloc[pos]
            mark = valuation.iloc[pos - 1]
            equity_before = cash + sum(qty * float(mark.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(mark.get(code, np.nan)))
            target_weight = exposure / len(selected) if selected else 0.0
            if entry_only:
                new_codes = set(selected) - set(holdings)
                open_value = sum(qty * float(valid_open[code] if pd.notna(valid_open.get(code)) and valid_open[code] > 0 else mark[code]) for code, qty in holdings.items())
                remaining = max(0.0, equity_before * exposure - open_value)
                target_weight = min(exposure / cfg.max_positions, remaining / equity_before / len(new_codes)) if new_codes else 0.0
            targets: dict[str, int] = {}
            for code in selected:
                if entry_only and code in holdings:
                    targets[code] = holdings[code]
                    continue
                price = valid_open.get(code)
                if pd.isna(price) or price <= 0:
                    targets[code] = holdings.get(code, 0)
                    continue
                lot = int(instruments.loc[code].get("lot_size") or 100)
                targets[code] = int(equity_before * target_weight / float(price) / lot) * lot
                if reduce_only or code in continuing:
                    targets[code] = min(targets[code], holdings.get(code, 0))

            def can_trade(code):
                raw = valid_open.get(code)
                # Do not fabricate executions on missing/zero-turnover bars.
                return pd.notna(raw) and raw > 0 and amount.iloc[pos].get(code, 0) > 0

            # Sell first so proceeds are available for buys.
            for code in sorted(set(holdings) | set(targets)):
                current, target = holdings.get(code, 0), targets.get(code, 0)
                if target >= current:
                    continue
                raw = valid_open.get(code)
                if not can_trade(code):
                    continue
                qty = current - target
                price = float(raw) * (1 - cfg.slippage_pct)
                value = qty * price
                commission = max(value * cfg.commission_rate, cfg.commission_min)
                cash += value - commission
                turnover += value
                trades.append({"date": str(day.date()), "signal_date": signal_day, "reason": "risk_reduce" if reduce_only else "rebalance", "code": code, "side": "SELL", "qty": qty, "price": price, "commission": commission})
                if target:
                    holdings[code] = target
                else:
                    holdings.pop(code, None)
            for code in selected:
                current, target = holdings.get(code, 0), targets.get(code, 0)
                if target <= current:
                    continue
                raw = valid_open.get(code)
                if not can_trade(code):
                    continue
                lot = int(instruments.loc[code].get("lot_size") or 100)
                price = float(raw) * (1 + cfg.slippage_pct)
                affordable_value = min(max(cash - cfg.commission_min, 0), cash / (1 + cfg.commission_rate))
                affordable = int(max(affordable_value, 0) / price / lot) * lot
                qty = min(target - current, affordable)
                if qty <= 0:
                    continue
                value = qty * price
                commission = max(value * cfg.commission_rate, cfg.commission_min)
                cash -= value + commission
                turnover += value
                holdings[code] = current + qty
                trades.append({"date": str(day.date()), "signal_date": signal_day, "reason": "daily_entry" if entry_only else "rebalance", "code": code, "side": "BUY", "qty": qty, "price": price, "commission": commission})
            pending = None

        marks = valuation.iloc[pos]
        equity = cash + sum(qty * float(marks.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(marks.get(code, np.nan)))
        equity_rows.append((day, equity))
        invested = sum(qty * float(marks.get(code, np.nan)) for code, qty in holdings.items() if pd.notna(marks.get(code, np.nan)))
        exposure_rows.append((day, invested / equity if equity > 0 else 0.0))
        cash_rows.append((day, cash))
        last_decision = close_decision(
            prepared, instruments, pos, holdings, equity, invested, state, cfg,
            risk_config, rebalance_days, rebalance_offset, daily_entries,
        )
        pending = last_decision["pending"]
        continuing = last_decision["continuing"]
        if last_decision["risk_row"] is not None:
            risk_rows.append(last_decision["risk_row"])

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
        "cash": pd.Series(dict(cash_rows), dtype=float), "risk_history": risk_rows,
        "risk_config": asdict(risk_config) if risk_config else None,
        "rebalance_days": rebalance_days, "rebalance_offset": rebalance_offset,
        "daily_entries": daily_entries,
        "simulation_basis": "qfq research units; not actual shares or a dividend/cash ledger",
        "final_state": {"cash": cash, "holdings": dict(holdings), "drawdown": asdict(state),
                        "pending": pending, "continuing": sorted(continuing),
                        "date": str(day.date()), "position": pos,
                        "equity": equity, "invested": invested},
        "last_decision": last_decision,
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
        f"调仓周期: {result.get('rebalance_days', '?')} 个交易日 | 每日回撤控制: {'启用' if result.get('risk_config') else '关闭'}",
        f"非调仓日新信号补空位: {'启用（研究模式）' if result.get('daily_entries') else '关闭'}",
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
    if result.get("risk_config"):
        lines.append(f"历史回撤10%目标: {'通过' if s['max_drawdown'] >= -0.10 else '未通过'}（样本内研究，不保证未来）")
    lines.append("执行边界: 日线近似；未完整模拟涨跌停、容量及信号计划的限价/跳空取消条件。")
    return "\n".join(lines)


__all__ = ["close_decision", "prepare_inputs", "format_portfolio_backtest", "run_etf_portfolio_backtest"]
