"""Watchlist / 持仓跟踪：累计涨跌 + 相对 HS300 等权基准的 alpha。

主要供 daily_scan 调用，每天产生一份 track_YYYY-MM-DD.md 推送到微信。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from qkquant.config import get_settings
from qkquant.data.storage import DuckStore


def _compute_hs300_ew_return(
    store: DuckStore, start: date, end: date, adjust: str | None = None,
) -> float:
    if adjust is None:
        adjust = get_settings().data.fetcher.adjust
    """HS300 等权累计收益（从 start 到 end）。

    用 HS300 成分股的简单平均代表"市场基准"。不严格等同于 HS300 指数（市值加权），
    但对个人选股的"超额收益"评估足够。
    """
    codes = store.load_index_constituents("000300")
    if not codes:
        return 0.0
    df = store.load_daily(codes=codes, start=start, end=end, adjust=adjust)
    if df.empty:
        return 0.0
    rets: list[float] = []
    for _, g in df.groupby("code"):
        if len(g) < 2:
            continue
        g_sorted = g.sort_values("trade_date")
        start_close = float(g_sorted["close"].iloc[0])
        end_close = float(g_sorted["close"].iloc[-1])
        if start_close > 0:
            rets.append(end_close / start_close - 1.0)
    return sum(rets) / len(rets) if rets else 0.0


def track_holdings(
    store: DuckStore,
    holdings: dict[str, dict],
    as_of: Optional[date] = None,
) -> dict:
    """对每只持仓计算从 bought_at 到 as_of 的累计收益 + 基准 + alpha。"""
    if not holdings:
        return {"rows": [], "summary": {}}

    if as_of is None:
        row = store.conn.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()
        as_of = row[0] if row and row[0] else None
    if as_of is None:
        return {"rows": [], "summary": {}}

    codes = list(holdings.keys())
    inst = store.load_instruments(codes)
    name_map = dict(zip(inst["code"], inst["name"])) if not inst.empty else {}

    # 按 bought_at 分组缓存基准（避免重复跑全 HS300 计算）
    benchmark_cache: dict[date, float] = {}

    rows: list[dict] = []
    for code, hold in holdings.items():
        cost = float(hold.get("cost", 0) or 0)
        qty = int(hold.get("qty", 0) or 0)
        bought_at_raw = hold.get("bought_at")
        if not bought_at_raw or cost <= 0:
            continue
        bought_at = pd.to_datetime(bought_at_raw).date()

        df = store.load_daily(codes=[code], start=bought_at, end=as_of, adjust=get_settings().data.fetcher.adjust)
        if df.empty:
            continue
        current = float(df.sort_values("trade_date").iloc[-1]["close"])
        ret_pct = (current / cost - 1.0) * 100 if cost > 0 else 0.0
        days = (as_of - bought_at).days

        if bought_at not in benchmark_cache:
            benchmark_cache[bought_at] = _compute_hs300_ew_return(store, bought_at, as_of)
        hs_ret_pct = benchmark_cache[bought_at] * 100

        rows.append(
            {
                "code": code,
                "name": name_map.get(code, ""),
                "qty": qty,
                "cost": cost,
                "current": current,
                "ret_pct": ret_pct,
                "days": days,
                "bought_at": str(bought_at),
                "hs300_ew_pct": hs_ret_pct,
                "alpha_pct": ret_pct - hs_ret_pct,
                "pnl": (current - cost) * qty,
                "note": hold.get("note", ""),
            }
        )

    if not rows:
        return {"rows": [], "summary": {}}

    rows.sort(key=lambda x: -x["ret_pct"])

    summary = {
        "as_of": str(as_of),
        "n_holdings": len(rows),
        "avg_ret_pct": sum(r["ret_pct"] for r in rows) / len(rows),
        "avg_hs300_pct": sum(r["hs300_ew_pct"] for r in rows) / len(rows),
        "avg_alpha_pct": sum(r["alpha_pct"] for r in rows) / len(rows),
        "total_pnl": sum(r["pnl"] for r in rows),
        "best": (rows[0]["code"], rows[0]["ret_pct"]),
        "worst": (rows[-1]["code"], rows[-1]["ret_pct"]),
        "win_count": sum(1 for r in rows if r["ret_pct"] > 0),
        "lose_count": sum(1 for r in rows if r["ret_pct"] < 0),
    }
    return {"rows": rows, "summary": summary}


def format_track_report(track: dict) -> str:
    if not track.get("rows"):
        return "# Watchlist 跟踪\n\n(无有效持仓 - 检查 config/positions.yaml)"

    s = track["summary"]
    out: list[str] = []
    out.append(f"# Watchlist 跟踪 | {s['as_of']}")
    out.append("")
    out.append(
        f"持仓 {s['n_holdings']} 只 | 累计盈亏 {s['total_pnl']:+.0f} 元 | "
        f"赢 {s['win_count']} 输 {s['lose_count']}"
    )
    out.append(
        f"平均收益 {s['avg_ret_pct']:+.2f}% | "
        f"HS300等权 {s['avg_hs300_pct']:+.2f}% | "
        f"alpha {s['avg_alpha_pct']:+.2f}%"
    )
    out.append("")
    out.append("## 个股明细 (按收益降序)")
    out.append("")
    out.append("| 票 | 成本 | 当前 | 收益 | 持有 | HS300基准 | alpha |")
    out.append("|---|---|---|---|---|---|---|")
    for r in track["rows"]:
        label = f"{r['code']} {r['name']}" if r["name"] else r["code"]
        out.append(
            f"| {label} | {r['cost']:.2f} | {r['current']:.2f} | "
            f"**{r['ret_pct']:+.2f}%** | {r['days']}d | "
            f"{r['hs300_ew_pct']:+.2f}% | {r['alpha_pct']:+.2f}% |"
        )
    out.append("")
    best_code, best_ret = s["best"]
    worst_code, worst_ret = s["worst"]
    out.append(f"- 最优: {best_code} ({best_ret:+.2f}%)")
    out.append(f"- 最差: {worst_code} ({worst_ret:+.2f}%)")
    out.append("")
    out.append("---")
    out.append("注: alpha = 个股收益 - HS300等权累计 (>0 表示跑赢市场)")
    out.append("注: 此为 watchlist 模拟跟踪, 非真实交易")
    return "\n".join(out)


__all__ = [
    "track_holdings",
    "format_track_report",
]
