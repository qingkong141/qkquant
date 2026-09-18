"""ETF close-to-next-open selection and rebalance plan."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qkquant.data.storage import DuckStore
from qkquant.etf_chan import classify_chan_structure
from qkquant.factors.library import trend_quality_60d
from qkquant.factors.pipeline import load_panel


@dataclass(frozen=True)
class EtfSignalConfig:
    min_cross_section: int = 20
    score_threshold: float = 0.65
    min_amount_20d: float = 50_000_000
    corr_limit: float = 0.80
    corr_window: int = 60
    max_positions: int = 3
    capital: float = 100_000
    commission_rate: float = 0.00005
    commission_min: float = 0.2
    slippage_pct: float = 0.002
    max_cost_rate: float = 0.005
    rebalance_band: float = 0.05
    min_trade_value: float = 1_000
    risk_on_breadth: float = 0.55
    risk_off_breadth: float = 0.40
    neutral_exposure: float = 0.50
    price_buffer: float = 0.01
    max_open_gap: float = 0.02
    chan_filter_enabled: bool = True
    chan_entry_states: tuple[str, ...] = ("third_buy",)
    chan_fractal_order: int = 2
    chan_tolerance: float = 0.015


def theme_key(name: str, benchmark_code: str | None = None) -> str:
    if benchmark_code and str(benchmark_code).strip():
        return str(benchmark_code).strip()
    text = re.sub(r"[\s\-_/]", "", str(name).upper())
    prefix = text.split("ETF", 1)[0]
    return prefix or text


def load_holdings(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("positions", []) if isinstance(raw, dict) else raw
    return {
        str(row.get("code", "")).strip().zfill(6): dict(row)
        for row in rows or []
        if str(row.get("code", "")).strip()
    }


def estimated_cost(value: float, stamp_tax: float, cfg: EtfSignalConfig) -> float:
    if value <= 0:
        return 0.0
    return max(value * cfg.commission_rate, cfg.commission_min) + value * (
        cfg.slippage_pct + stamp_tax
    )


def _research_date(
    close: pd.DataFrame, requested: date | str | None, min_cross_section: int
) -> pd.Timestamp:
    if requested:
        result = pd.to_datetime(requested)
        if result not in close.index or close.loc[result].notna().sum() < min_cross_section:
            raise RuntimeError(
                f"requested date {result.date()} has fewer than {min_cross_section} ETF bars"
            )
        return result
    coverage = close.notna().sum(axis=1)
    valid = coverage[coverage >= min_cross_section]
    if valid.empty:
        raise RuntimeError(f"no date has at least {min_cross_section} ETF bars")
    return valid.index.max()


def build_etf_plan(
    store: DuckStore,
    category: str = "equity",
    as_of: date | str | None = None,
    holdings_path: Path | None = None,
    config: EtfSignalConfig | None = None,
    adjust: str = "qfq",
) -> dict:
    cfg = config or EtfSignalConfig()
    codes = store.load_etf_codes(category)
    panel = load_panel(store, codes=codes, start="2022-01-01", end=as_of, adjust=adjust)
    close, amount = panel["close"], panel["amount"]
    signal_date = _research_date(close, as_of, cfg.min_cross_section)
    ret = close.pct_change(fill_method=None)
    mom60 = close.pct_change(60, fill_method=None)
    mom120 = close.pct_change(120, fill_method=None)
    risk_mom = mom60 / ret.rolling(60, min_periods=60).std().replace(0, np.nan)
    trend = trend_quality_60d(panel)
    downside = ret.clip(upper=0).rolling(20, min_periods=20).std()
    ma120 = close.rolling(120, min_periods=120).mean()
    amount20 = amount.rolling(20, min_periods=20).mean()

    breadth_mask = close.loc[signal_date].notna() & ma120.loc[signal_date].notna()
    breadth = float(
        (close.loc[signal_date, breadth_mask] > ma120.loc[signal_date, breadth_mask]).mean()
    )
    if breadth >= cfg.risk_on_breadth:
        regime, exposure = "risk_on", 1.0
    elif breadth >= cfg.risk_off_breadth:
        regime, exposure = "neutral", cfg.neutral_exposure
    else:
        regime, exposure = "risk_off", 0.0

    frame = pd.DataFrame(
        {
            "close": close.loc[signal_date], "mom60": mom60.loc[signal_date],
            "mom120": mom120.loc[signal_date], "risk_mom": risk_mom.loc[signal_date],
            "trend": trend.loc[signal_date], "downside": downside.loc[signal_date],
            "amount20": amount20.loc[signal_date],
            "above_ma120": close.loc[signal_date] > ma120.loc[signal_date],
        }
    ).dropna()
    frame = frame[
        (frame.mom60 > 0) & (frame.mom120 > 0) & frame.above_ma120
        & (frame.amount20 >= cfg.min_amount_20d)
    ].copy()
    frame["score"] = (
        0.35 * frame.risk_mom.rank(pct=True)
        + 0.25 * frame.mom120.rank(pct=True)
        + 0.20 * frame.trend.rank(pct=True)
        + 0.20 * (1 - frame.downside.rank(pct=True))
    )
    frame = frame[frame.score >= cfg.score_threshold].sort_values("score", ascending=False)
    inst = store.load_instruments(frame.index.tolist()).set_index("code") if len(frame) else pd.DataFrame()

    selected: list[str] = []
    rejected: list[dict] = []
    themes: set[str] = set()
    corr_data = ret.loc[:signal_date].tail(cfg.corr_window)
    chan_signals: dict[str, dict] = {}
    for code in frame.index:
        info = inst.loc[code]
        name = str(info.get("name", code))
        chan = classify_chan_structure(
            close[code].loc[:signal_date],
            panel["high"][code].loc[:signal_date],
            panel["low"][code].loc[:signal_date],
            fractal_order=cfg.chan_fractal_order,
            tolerance=cfg.chan_tolerance,
        )
        chan_signals[code] = chan.to_dict()
        if cfg.chan_filter_enabled and chan.state not in cfg.chan_entry_states:
            rejected.append({"code": code, "reason": f"chan:{chan.state}", "chan": chan.to_dict()})
            continue
        theme = theme_key(name, info.get("benchmark_code"))
        if theme in themes:
            rejected.append({"code": code, "reason": f"duplicate_theme:{theme}"})
            continue
        max_corr = 0.0
        if selected:
            corr = corr_data[selected + [code]].corr()[code].drop(code)
            max_corr = float(corr.max()) if len(corr) else 0.0
        if max_corr > cfg.corr_limit:
            rejected.append({"code": code, "reason": f"correlation:{max_corr:.3f}"})
            continue
        selected.append(code)
        themes.add(theme)
        if len(selected) == cfg.max_positions:
            break
    if exposure == 0:
        selected = []

    all_holdings = load_holdings(holdings_path)
    etf_codes = set(store.load_etf_codes())
    holdings = {code: row for code, row in all_holdings.items() if code in etf_codes}
    ignored = sorted(set(all_holdings) - etf_codes)
    target_weight = exposure / len(selected) if selected else 0.0
    chosen: list[dict] = []
    actions: list[dict] = []

    for code in selected:
        row, info = frame.loc[code], inst.loc[code]
        price = float(row.close)
        lot = int(info.get("lot_size") or 100)
        target_qty = math.floor(cfg.capital * target_weight / price / lot) * lot
        current_qty = int(holdings.get(code, {}).get("qty", 0) or 0)
        delta = target_qty - current_qty
        value = abs(delta) * price
        cost = estimated_cost(value, 0.0, cfg)
        cost_rate = cost / value if value else 0.0
        drift = abs(target_weight - current_qty * price / cfg.capital)
        action, reason = "HOLD", "within_rebalance_band"
        if delta and value >= cfg.min_trade_value and drift >= cfg.rebalance_band:
            action, reason = (("BUY" if delta > 0 else "SELL"), "rebalance") if cost_rate <= cfg.max_cost_rate else ("HOLD", "cost_rate_too_high")
        chosen.append({
            "code": code, "name": str(info.get("name", code)), "score": float(row.score),
            "mom60": float(row.mom60), "mom120": float(row.mom120),
            "target_weight": target_weight,
            "chan_state": chan_signals[code]["state"],
            "divergence": chan_signals[code]["divergence"],
            "divergence_strength": chan_signals[code]["divergence_strength"],
        })
        actions.append(_action(code, str(info.get("name", code)), action, abs(delta), current_qty, target_qty, price, cost, cost_rate, reason, cfg))

    for code, holding in holdings.items():
        if code in selected:
            continue
        bars = store.load_daily([code], end=signal_date.date(), adjust=adjust)
        if bars.empty:
            continue
        price, qty = float(bars.iloc[-1].close), int(holding.get("qty", 0) or 0)
        info = store.load_instruments([code]).iloc[0]
        value = price * qty
        cost = estimated_cost(value, float(info.get("stamp_tax_rate") or 0), cfg)
        actions.append(_action(code, str(info.get("name", code)), "SELL", qty, qty, 0, price, cost, cost / value if value else 0, "not_selected_or_risk_off", cfg))

    return {
        "as_of": str(signal_date.date()), "execution_session": "next_trading_day_open",
        "regime": regime, "market_breadth": breadth, "target_exposure": exposure,
        "config": asdict(cfg), "selected": chosen, "actions": actions,
        "rejected": rejected, "ignored_non_etf_holdings": ignored,
    }


def _action(code: str, name: str, action: str, qty: int, current_qty: int, target_qty: int, price: float, cost: float, cost_rate: float, reason: str, cfg: EtfSignalConfig) -> dict:
    direction = 1 if action == "BUY" else -1
    limit_price = round(price * (1 + direction * cfg.price_buffer), 3) if action in {"BUY", "SELL"} else None
    return {
        "code": code, "name": name, "action": action, "qty": qty,
        "current_qty": current_qty, "target_qty": target_qty, "reference_close": price,
        "limit_price": limit_price, "estimated_cost": round(cost, 2),
        "estimated_cost_rate": cost_rate, "reason": reason,
        "execution": f"next open; cancel if absolute gap > {cfg.max_open_gap:.1%}",
    }


def format_etf_plan(plan: dict) -> str:
    lines = [f"# ETF 次日开盘计划 | {plan['as_of']}", ""]
    lines.append(f"状态: **{plan['regime']}** | 宽度: {plan['market_breadth']:.1%} | 目标仓位: {plan['target_exposure']:.0%}")
    lines += ["", "## 入选", "", "| ETF | 得分 | 60日 | 120日 | 缠论状态 | 背离 | 目标权重 |", "|---|---:|---:|---:|---|---|---:|"]
    for row in plan["selected"]:
        lines.append(f"| {row['code']} {row['name']} | {row['score']:.3f} | {row['mom60']:.2%} | {row['mom120']:.2%} | {row['chan_state']} | {row['divergence'] or '-'} | {row['target_weight']:.1%} |")
    if not plan["selected"]:
        lines.append("| 无 | - | - | - | - | - | 0% |")
    lines += ["", "## 次日动作", "", "| 动作 | ETF | 数量 | 限价参考 | 预估成本 | 原因 |", "|---|---|---:|---:|---:|---|"]
    for row in plan["actions"]:
        limit_text = "-" if row["limit_price"] is None else f"{row['limit_price']:.3f}"
        lines.append(f"| {row['action']} | {row['code']} {row['name']} | {row['qty']} | {limit_text} | {row['estimated_cost']:.2f} | {row['reason']} |")
    lines += ["", "仅使用收盘信息；次交易日开盘执行；开盘缺口超阈值取消。"]
    if plan["ignored_non_etf_holdings"]:
        lines.append("已忽略非 ETF 旧持仓: " + ", ".join(plan["ignored_non_etf_holdings"]))
    return "\n".join(lines)


def save_etf_plan(plan: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"etf_plan_{plan['as_of']}"
    md, js = output_dir / f"{stem}.md", output_dir / f"{stem}.json"
    md.write_text(format_etf_plan(plan), encoding="utf-8")
    js.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return md, js


__all__ = ["EtfSignalConfig", "build_etf_plan", "estimated_cost", "format_etf_plan", "save_etf_plan", "theme_key"]
