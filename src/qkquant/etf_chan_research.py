"""Walk-forward research helpers for the ETF Chan-inspired filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.etf_chan import classify_chan_structure
from qkquant.factors.pipeline import load_panel


@dataclass(frozen=True)
class ChanResearchConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    sample_step: int = 5
    min_history: int = 120
    min_amount_20d: float = 50_000_000
    fractal_order: int = 2
    tolerance: float = 0.015


def _summary(samples: pd.DataFrame, horizons: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    states = ["all", *sorted(samples["state"].dropna().unique())] if len(samples) else ["all"]
    for state in states:
        group = samples if state == "all" else samples[samples["state"] == state]
        for horizon in horizons:
            values = group[f"return_{horizon}d"].dropna()
            adverse = group.loc[values.index, f"mae_{horizon}d"].dropna()
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append({
                "state": state,
                "horizon": horizon,
                "n": int(len(values)),
                "mean_return": float(values.mean()) if len(values) else None,
                "median_return": float(values.median()) if len(values) else None,
                "win_rate": float((values > 0).mean()) if len(values) else None,
                "t_stat": float(values.mean() / (std / np.sqrt(len(values)))) if len(values) > 1 and std > 0 else None,
                "mean_mae": float(adverse.mean()) if len(adverse) else None,
            })
    return rows


def evaluate_chan_signals(
    store: DuckStore,
    category: str = "equity",
    start: str = "2022-01-01",
    end: str | None = None,
    config: ChanResearchConfig | None = None,
) -> dict:
    cfg = config or ChanResearchConfig()
    codes = store.load_etf_codes(category)
    panel = load_panel(store, codes=codes, start=start, end=end)
    close, high, low, amount = (panel[key] for key in ("close", "high", "low", "amount"))
    amount20 = amount.rolling(20, min_periods=20).mean()
    records: list[dict] = []

    last_horizon = max(cfg.horizons)
    for position in range(cfg.min_history, len(close) - last_horizon, cfg.sample_step):
        signal_date = close.index[position]
        for code in close.columns:
            if pd.isna(close.iloc[position][code]) or amount20.iloc[position][code] < cfg.min_amount_20d:
                continue
            signal = classify_chan_structure(
                close[code].iloc[: position + 1],
                high[code].iloc[: position + 1],
                low[code].iloc[: position + 1],
                fractal_order=cfg.fractal_order,
                tolerance=cfg.tolerance,
            )
            entry = float(close.iloc[position][code])
            row = {
                "date": str(signal_date.date()), "code": code, "state": signal.state,
                "allowed": signal.allowed, "divergence": signal.divergence,
                "divergence_strength": signal.divergence_strength,
            }
            for horizon in cfg.horizons:
                future_close = close[code].iloc[position + horizon]
                future_lows = low[code].iloc[position + 1 : position + horizon + 1]
                row[f"return_{horizon}d"] = float(future_close / entry - 1) if pd.notna(future_close) else np.nan
                row[f"mae_{horizon}d"] = float(future_lows.min() / entry - 1) if future_lows.notna().any() else np.nan
            records.append(row)

    samples = pd.DataFrame(records)
    allowed = samples[samples["allowed"]].copy() if len(samples) else samples.copy()
    vetoed_top = samples[samples["state"] == "top_divergence"].copy() if len(samples) else samples.copy()
    return {
        "start": str(close.index[cfg.min_history].date()) if len(close) > cfg.min_history else None,
        "end": str(close.index[-last_horizon - 1].date()) if len(close) > last_horizon else None,
        "category": category,
        "config": cfg.__dict__,
        "sample_count": int(len(samples)),
        "allowed_count": int(len(allowed)),
        "summary_baseline": _summary(samples, cfg.horizons),
        "summary_allowed": _summary(allowed, cfg.horizons),
        "summary_top_divergence": _summary(vetoed_top, cfg.horizons),
        "samples": samples,
    }


def format_chan_research(result: dict) -> str:
    lines = [
        f"ETF 缠论信号检验 | {result['start']} ~ {result['end']}",
        f"全部样本: {result['sample_count']} | 入场信号: {result['allowed_count']}",
        "",
        "全体可交易样本基准:",
    ]
    for row in result["summary_baseline"]:
        if row["state"] != "all":
            continue
        mean = "n/a" if row["mean_return"] is None else f"{row['mean_return']:.2%}"
        win = "n/a" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        lines.append(f"{row['horizon']:>3}d N={row['n']:<6} 平均={mean:>8} 胜率={win:>7}")
    lines += ["", "入场信号统计:", "状态                 周期      N      平均收益      胜率      t值      平均MAE"]
    for row in result["summary_allowed"]:
        mean = "n/a" if row["mean_return"] is None else f"{row['mean_return']:.2%}"
        win = "n/a" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        t_stat = "n/a" if row["t_stat"] is None else f"{row['t_stat']:.2f}"
        mae = "n/a" if row["mean_mae"] is None else f"{row['mean_mae']:.2%}"
        lines.append(f"{row['state']:<20} {row['horizon']:>3}d {row['n']:>6} {mean:>12} {win:>9} {t_stat:>8} {mae:>11}")
    lines += ["", "顶背离（应弱于市场）统计:"]
    for row in result["summary_top_divergence"]:
        if row["state"] != "all":
            continue
        mean = "n/a" if row["mean_return"] is None else f"{row['mean_return']:.2%}"
        win = "n/a" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        lines.append(f"{row['horizon']:>3}d N={row['n']:<6} 平均={mean:>8} 胜率={win:>7}")
    return "\n".join(lines)


__all__ = ["ChanResearchConfig", "evaluate_chan_signals", "format_chan_research"]
