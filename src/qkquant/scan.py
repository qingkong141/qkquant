"""每日信号扫描：跑历史回测到信号日，提取最后一天的信号。

核心思路：
- 不重写策略逻辑，复用 BacktestEngine。
- 跑过去 ~6 个月历史到 as_of，策略 _trade_log 已记录每笔模拟交易。
- 过滤 date == as_of 的条目即为今日信号。

注意事项：
- BUY 信号：可直接采纳（如果该票你还没持有）。
- SELL 信号：分两类——
  1) 你 positions.yaml 里真实持有的 → 优先关注
  2) 策略模拟仓里持有但你没有的 → 仅供参考
- relative_strength 是周期性调仓（每 20 个 bar），非调仓日没有信号是正常的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from qkquant.backtest.engine import BacktestEngine
from qkquant.data.storage import DuckStore
from qkquant.logger import logger
from qkquant.strategy.registry import (
    get_strategy,
    load_risk_config,
    load_strategy_config,
)

WARMUP_DAYS = 180


@dataclass
class StrategySignals:
    strategy: str
    as_of: date
    buys: list[dict] = field(default_factory=list)
    sells: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)


def latest_data_date(store: DuckStore) -> Optional[date]:
    row = store.conn.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()
    return row[0] if row and row[0] else None


def scan_strategy(
    store: DuckStore,
    strategy_name: str,
    codes: list[str],
    as_of: date,
    capital: float = 1_000_000.0,
) -> StrategySignals:
    info = get_strategy(strategy_name)
    cfg = load_strategy_config(info)
    params = (cfg.get("params") or {}) if cfg else {}
    risk_cfg = load_risk_config(cfg)

    start = as_of - timedelta(days=WARMUP_DAYS + 30)
    engine = BacktestEngine(
        store=store,
        strategy_cls=info.cls,
        strategy_params=params,
        initial_capital=capital,
        risk_config=risk_cfg,
    )
    result = engine.run(codes=codes, start=start, end=as_of)
    strat = result["strategy"]
    trade_log = getattr(strat, "_trade_log", [])
    rejections = result.get("rejections", [])

    buys = [t for t in trade_log if t["date"] == as_of and t["side"] == "BUY"]
    sells = [t for t in trade_log if t["date"] == as_of and t["side"] == "SELL"]
    rejs = [r for r in rejections if r["date"] == as_of]

    return StrategySignals(
        strategy=strategy_name,
        as_of=as_of,
        buys=buys,
        sells=sells,
        rejections=rejs,
    )


def scan_all(
    store: DuckStore,
    strategies: list[str],
    codes: list[str],
    as_of: Optional[date] = None,
) -> list[StrategySignals]:
    if as_of is None:
        as_of = latest_data_date(store)
        if as_of is None:
            raise RuntimeError("no data in db; run `qkquant update-data` first")
    out: list[StrategySignals] = []
    for name in strategies:
        logger.info(f"scanning {name} as of {as_of} ...")
        out.append(scan_strategy(store, name, codes, as_of))
    return out


def load_holdings(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pos_list = cfg.get("positions") or []
    return {p["code"]: p for p in pos_list if isinstance(p, dict) and "code" in p}


def format_signals(
    signals: list[StrategySignals],
    holdings: dict[str, dict],
    name_map: Optional[dict[str, str]] = None,
) -> str:
    name_map = name_map or {}

    def label(code: str) -> str:
        nm = name_map.get(code)
        return f"{code} {nm}" if nm else code

    out: list[str] = []
    as_of = signals[0].as_of if signals else "-"
    out.append(f"# qkquant 信号 | {as_of}")
    out.append("")

    # ---- BUY 汇总 ----
    buy_by_code: dict[str, list[str]] = {}
    for s in signals:
        for b in s.buys:
            buy_by_code.setdefault(b["code"], []).append(
                f"{s.strategy}({b.get('reason', '?')})"
            )
    out.append("## [BUY] 建议买入")
    if buy_by_code:
        for code, sources in sorted(buy_by_code.items(), key=lambda x: -len(x[1])):
            tag = "  [!! 你已持有]" if code in holdings else ""
            multi = "  [** 多策略共振]" if len(sources) >= 2 else ""
            out.append(f"  - {label(code)}{multi}{tag}  ←  {' / '.join(sources)}")
    else:
        out.append("  (无)")
    out.append("")

    # ---- SELL：你的持仓 vs 模拟仓 ----
    held_sells: list[str] = []
    sim_sells: list[str] = []
    for s in signals:
        for x in s.sells:
            line = f"  - {label(x['code'])}  ←  {s.strategy}({x.get('reason', '?')})"
            if x["code"] in holdings:
                qty = holdings[x["code"]].get("qty", "?")
                cost = holdings[x["code"]].get("cost", "?")
                held_sells.append(
                    f"  - {label(x['code'])} (你持有 {qty} 股, 成本 {cost})"
                    f"  ←  {s.strategy}({x.get('reason', '?')})"
                )
            else:
                sim_sells.append(line)

    out.append("## [SELL] 建议卖出")
    if held_sells:
        out.append("**【你的真实持仓】**")
        out.extend(held_sells)
    if sim_sells:
        if held_sells:
            out.append("")
        out.append("**【策略模拟仓 — 你不持有，仅供参考】**")
        out.extend(sim_sells)
    if not held_sells and not sim_sells:
        out.append("  (无)")
    out.append("")

    # ---- 拦截 ----
    rej_lines: list[str] = []
    for s in signals:
        for r in s.rejections:
            rej_lines.append(
                f"  - {s.strategy}: {r['side']} {label(r['code'])} → {r.get('reason', '?')}"
            )
    out.append("## [REJECTED] 被拦截订单（涨跌停 / T+1 / 风控）")
    if rej_lines:
        out.extend(rej_lines)
    else:
        out.append("  (无)")
    out.append("")

    out.append("---")
    out.append("注意: 信号基于收盘价。明日开盘下单可能有滑点；")
    out.append("      涨停一字板买不进，跌停一字板卖不出，请预判。")
    out.append("注意: relative_strength 每 20 个 bar 才调仓一次，非调仓日无信号是正常的。")

    return "\n".join(out)


# =====================================================================
# 裸信号扫描（--raw）：对每只股票直接算策略入场条件，
# 不依赖 BacktestEngine、不受模拟仓状态/风控熔断影响。
# =====================================================================


def _raw_ma_breakout(df: pd.DataFrame, params: dict) -> dict:
    """对单只股票算 ma_breakout 当前信号。"""
    fast = int(params.get("fast", 5))
    slow = int(params.get("slow", 20))
    stop_loss_pct = float(params.get("stop_loss_pct", 0.05))
    min_price = float(params.get("min_price", 1.0))

    if len(df) < slow + 1:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}

    closes = df["close"].to_numpy()
    today_close = float(closes[-1])
    today_fast = float(closes[-fast:].mean())
    today_slow = float(closes[-slow:].mean())
    yest_fast = float(closes[-fast - 1 : -1].mean())
    yest_slow = float(closes[-slow - 1 : -1].mean())

    cross_up = (today_fast > today_slow) and (yest_fast <= yest_slow)
    cross_down = (today_fast < today_slow) and (yest_fast >= yest_slow)

    buy = cross_up and today_close >= min_price
    sell_stop = today_slow > 0 and today_close < today_slow * (1 - stop_loss_pct)
    sell = cross_down or sell_stop

    if sell:
        sell_reason = "ma_cross_down" if cross_down else "stop_loss"
    else:
        sell_reason = None

    return {
        "buy": buy,
        "sell": sell,
        "buy_reason": "ma_cross_up" if buy else None,
        "sell_reason": sell_reason,
        "score": today_fast / max(today_slow, 1e-6) - 1.0,
        "metrics": {
            "close": today_close,
            "fast_ma": today_fast,
            "slow_ma": today_slow,
            "fast_over_slow": today_fast / max(today_slow, 1e-6) - 1.0,
        },
    }


def _raw_ma_boll(df: pd.DataFrame, params: dict) -> dict:
    """ma_breakout + 布林带过滤的裸信号。"""
    fast = int(params.get("fast", 5))
    slow = int(params.get("slow", 20))
    boll_period = int(params.get("boll_period", 20))
    boll_dev = float(params.get("boll_dev", 2.0))
    upper_buffer = float(params.get("upper_buffer", 0.03))
    mid_break_pct = float(params.get("mid_break_pct", 0.05))
    min_price = float(params.get("min_price", 1.0))

    if len(df) < max(slow, boll_period) + 1:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}

    closes = df["close"].to_numpy()
    today_close = float(closes[-1])
    today_fast = float(closes[-fast:].mean())
    today_slow = float(closes[-slow:].mean())
    yest_fast = float(closes[-fast - 1 : -1].mean())
    yest_slow = float(closes[-slow - 1 : -1].mean())

    cross_up = (today_fast > today_slow) and (yest_fast <= yest_slow)
    cross_down = (today_fast < today_slow) and (yest_fast >= yest_slow)

    boll_window = closes[-boll_period:]
    boll_mid = float(boll_window.mean())
    boll_std = float(boll_window.std(ddof=0))
    boll_upper = boll_mid + boll_dev * boll_std
    boll_lower = boll_mid - boll_dev * boll_std

    above_mid = today_close > boll_mid
    not_at_upper = today_close <= boll_upper * (1 - upper_buffer)

    buy = (
        cross_up
        and today_close >= min_price
        and above_mid
        and not_at_upper
    )
    sell_break_mid = boll_mid > 0 and today_close < boll_mid * (1 - mid_break_pct)
    sell = cross_down or sell_break_mid

    sell_reason = None
    if sell:
        sell_reason = "ma_cross_down" if cross_down else "below_boll_mid"

    return {
        "buy": buy,
        "sell": sell,
        "buy_reason": "ma_boll_entry" if buy else None,
        "sell_reason": sell_reason,
        "score": today_fast / max(today_slow, 1e-6) - 1.0,
        "metrics": {
            "close": today_close,
            "fast_ma": today_fast,
            "slow_ma": today_slow,
            "boll_mid": boll_mid,
            "boll_upper": boll_upper,
            "boll_lower": boll_lower,
            "band_position": (today_close - boll_mid) / (boll_upper - boll_mid)
            if boll_upper > boll_mid
            else 0.0,
        },
    }


def _raw_momentum(df: pd.DataFrame, params: dict) -> dict:
    mom_window = int(params.get("mom_window", 20))
    entry_threshold = float(params.get("entry_threshold", 0.03))
    exit_threshold = float(params.get("exit_threshold", -0.03))
    drawdown_from_peak = float(params.get("drawdown_from_peak", 0.10))
    min_price = float(params.get("min_price", 1.0))

    if len(df) < mom_window + 1:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    today_close = float(closes[-1])
    start_close = float(closes[-mom_window - 1])
    if start_close <= 0:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}
    mom = today_close / start_close - 1.0
    win_high = float(highs[-mom_window:].max())
    drawdown = (today_close / win_high - 1.0) if win_high > 0 else 0.0

    near_high = (win_high > 0) and (today_close >= win_high * (1 - drawdown_from_peak))
    buy = (today_close >= min_price) and (mom > entry_threshold) and near_high
    sell = mom < exit_threshold

    return {
        "buy": buy,
        "sell": sell,
        "buy_reason": "momentum_entry" if buy else None,
        "sell_reason": "momentum_exit" if sell else None,
        "score": mom,
        "metrics": {
            "close": today_close,
            "mom_20d": mom,
            "win_high": win_high,
            "drawdown_from_high": drawdown,
        },
    }


def _raw_momentum_breakout(df: pd.DataFrame, params: dict) -> dict:
    mom_window = int(params.get("mom_window", 20))
    entry_threshold = float(params.get("entry_threshold", 0.05))
    exit_threshold = float(params.get("exit_threshold", -0.03))
    drawdown_from_peak = float(params.get("drawdown_from_peak", 0.03))
    breakout_window = int(params.get("breakout_window", 10))
    min_price = float(params.get("min_price", 1.0))

    needed = max(mom_window, breakout_window) + 1
    if len(df) < needed:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    today_close = float(closes[-1])
    if today_close < min_price:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}

    start_close = float(closes[-mom_window - 1])
    if start_close <= 0:
        return {"buy": False, "sell": False, "score": 0.0, "metrics": {}}
    mom = today_close / start_close - 1.0
    win_high = float(highs[-mom_window:].max())
    drawdown = (today_close / win_high - 1.0) if win_high > 0 else 0.0

    near_peak = (win_high > 0) and (today_close >= win_high * (1 - drawdown_from_peak))

    prev_close_max = float(closes[-breakout_window - 1 : -1].max())
    is_breakout = today_close >= prev_close_max

    buy = (
        (today_close >= min_price)
        and (mom > entry_threshold)
        and near_peak
        and is_breakout
    )
    sell = mom < exit_threshold

    return {
        "buy": buy,
        "sell": sell,
        "buy_reason": "breakout_entry" if buy else None,
        "sell_reason": "momentum_exit" if sell else None,
        "score": mom,
        "metrics": {
            "close": today_close,
            "mom_20d": mom,
            "win_high_20d": win_high,
            "drawdown_from_high": drawdown,
            "prev_close_max_10d": prev_close_max,
            "is_new_high": int(is_breakout),
        },
    }


def _raw_relative_strength(
    price_data: dict[str, pd.DataFrame], params: dict
) -> dict[str, dict]:
    lookback = int(params.get("lookback", 60))
    top_k = int(params.get("top_k", 10))
    min_price = float(params.get("min_price", 1.0))

    scored: list[tuple[str, float, float]] = []
    for code, df in price_data.items():
        if len(df) < lookback + 1:
            continue
        closes = df["close"].to_numpy()
        today_close = float(closes[-1])
        if today_close < min_price:
            continue
        start = float(closes[-lookback - 1])
        if start <= 0:
            continue
        mom = today_close / start - 1.0
        scored.append((code, mom, today_close))

    scored.sort(key=lambda x: -x[1])
    top_codes = {c for c, _, _ in scored[:top_k]}

    out: dict[str, dict] = {}
    for rank, (code, mom, close) in enumerate(scored, start=1):
        out[code] = {
            "buy": code in top_codes,
            "sell": False,
            "buy_reason": f"rs_top{top_k}" if code in top_codes else None,
            "sell_reason": None,
            "score": mom,
            "metrics": {
                "close": close,
                "lookback_return": mom,
                "rank": rank,
                "total_scored": len(scored),
            },
        }
    return out


def scan_raw(
    store: DuckStore,
    strategies: list[str],
    codes: list[str],
    holdings: dict[str, dict],
    as_of: date,
    history_days: int = 120,
) -> dict[str, dict]:
    """裸信号扫描：对每只股票直接算策略入场/出场条件。

    Returns:
        { strategy_name: { "buys": [...], "sells": [...] } }
    """
    start = as_of - timedelta(days=history_days * 2)
    df_all = store.load_daily(codes=codes, start=start, end=as_of, adjust="hfq")
    if df_all.empty:
        raise RuntimeError(f"no daily data in [{start}, {as_of}]")
    price_data = {
        code: g.sort_values("trade_date").reset_index(drop=True)
        for code, g in df_all.groupby("code")
    }

    results: dict[str, dict] = {name: {"buys": [], "sells": []} for name in strategies}

    for name in strategies:
        info = get_strategy(name)
        cfg = load_strategy_config(info)
        params = (cfg.get("params") or {}) if cfg else {}

        if name == "ma_boll":
            cap = int(params.get("max_positions", 10))
            for code, df in price_data.items():
                sig = _raw_ma_boll(df, params)
                row = {"code": code, **sig}
                if sig.get("buy"):
                    results[name]["buys"].append(row)
                if sig.get("sell") and code in holdings:
                    results[name]["sells"].append(row)
            results[name]["buys"].sort(key=lambda x: -x["score"])
            results[name]["buys"] = results[name]["buys"][:cap]

        elif name == "ma_breakout":
            cap = int(params.get("max_positions", 10))
            for code, df in price_data.items():
                sig = _raw_ma_breakout(df, params)
                row = {"code": code, **sig}
                if sig.get("buy"):
                    results[name]["buys"].append(row)
                if sig.get("sell") and code in holdings:
                    results[name]["sells"].append(row)
            results[name]["buys"].sort(key=lambda x: -x["score"])
            results[name]["buys"] = results[name]["buys"][:cap]

        elif name == "momentum":
            cap = int(params.get("max_positions", 10))
            for code, df in price_data.items():
                sig = _raw_momentum(df, params)
                row = {"code": code, **sig}
                if sig.get("buy"):
                    results[name]["buys"].append(row)
                if sig.get("sell") and code in holdings:
                    results[name]["sells"].append(row)
            results[name]["buys"].sort(key=lambda x: -x["score"])
            results[name]["buys"] = results[name]["buys"][:cap]

        elif name == "momentum_breakout":
            cap = int(params.get("max_positions", 8))
            for code, df in price_data.items():
                sig = _raw_momentum_breakout(df, params)
                row = {"code": code, **sig}
                if sig.get("buy"):
                    results[name]["buys"].append(row)
                if sig.get("sell") and code in holdings:
                    results[name]["sells"].append(row)
            results[name]["buys"].sort(key=lambda x: -x["score"])
            results[name]["buys"] = results[name]["buys"][:cap]

        elif name == "relative_strength":
            sigs = _raw_relative_strength(price_data, params)
            top_k = int(params.get("top_k", 10))
            for code, sig in sigs.items():
                row = {"code": code, **sig}
                if sig.get("buy"):
                    results[name]["buys"].append(row)
                if code in holdings and not sig.get("buy"):
                    row["sell"] = True
                    row["sell_reason"] = f"fell_out_of_top{top_k}"
                    results[name]["sells"].append(row)
            results[name]["buys"].sort(key=lambda x: -x["score"])

    return results


def _fmt_metrics(m: dict) -> str:
    parts: list[str] = []
    for k, v in m.items():
        if isinstance(v, float):
            if k.endswith("_return") or k.endswith("_high") or k == "drawdown_from_high" or k == "fast_over_slow" or k.startswith("mom_"):
                parts.append(f"{k}={v:+.2%}" if abs(v) < 10 else f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


def format_raw_signals(
    results: dict[str, dict],
    holdings: dict[str, dict],
    name_map: Optional[dict[str, str]] = None,
    as_of: Optional[date] = None,
) -> str:
    name_map = name_map or {}

    def label(code: str) -> str:
        nm = name_map.get(code)
        return f"{code} {nm}" if nm else code

    titles = {
        "ma_breakout": "ma_breakout (双均线突破: fast/slow 金叉)",
        "ma_boll": "ma_boll (双均线 + 布林带: 金叉+中轨上方+不近上轨)",
        "momentum": "momentum (绝对动量: 20日累计涨幅+近高点过滤)",
        "momentum_breakout": "momentum_breakout (动量突破: 强动量+紧贴峰值+创新高)",
        "relative_strength": "relative_strength (横截面: 60日涨幅排名)",
    }

    out: list[str] = []
    out.append(f"# qkquant 裸信号扫描 | {as_of or '-'}")
    out.append("> 不跑回测、不受模拟仓状态和风控熔断影响；纯今日条件判断")
    out.append("")

    for name in ["ma_breakout", "ma_boll", "momentum", "momentum_breakout", "relative_strength"]:
        if name not in results:
            continue
        r = results[name]
        out.append(f"## {titles.get(name, name)}")
        out.append("")

        out.append(f"### [BUY] 满足入场条件 ({len(r['buys'])} 只)")
        if r["buys"]:
            for s in r["buys"]:
                code = s["code"]
                tag = "  [!! 你已持有]" if code in holdings else ""
                out.append(f"  - {label(code)}{tag}")
                out.append(f"      {_fmt_metrics(s.get('metrics', {}))}")
        else:
            out.append("  (无)")
        out.append("")

        out.append(f"### [SELL] 你持仓中触发出场 ({len(r['sells'])} 只)")
        if r["sells"]:
            for s in r["sells"]:
                code = s["code"]
                qty = holdings[code].get("qty", "?") if code in holdings else "?"
                cost = holdings[code].get("cost", "?") if code in holdings else "?"
                out.append(
                    f"  - {label(code)} (持仓 {qty} 股, 成本 {cost})  ←  {s.get('sell_reason')}"
                )
                out.append(f"      {_fmt_metrics(s.get('metrics', {}))}")
        else:
            out.append("  (无)")
        out.append("")

    out.append("---")
    out.append("注意: 裸信号忽略风控熔断、组合冷却期、T+1 / 涨跌停拦截。下单前请自行确认。")
    out.append("注意: 多策略共振（同一只票被多个策略 BUY）通常更可靠。")
    return "\n".join(out)


__all__ = [
    "StrategySignals",
    "scan_strategy",
    "scan_all",
    "scan_raw",
    "load_holdings",
    "format_signals",
    "format_raw_signals",
    "latest_data_date",
]
