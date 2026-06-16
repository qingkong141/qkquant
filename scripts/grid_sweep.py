"""grid_trading 参数扫描。

目标：
- 在统一时间段和股票池上批量扫描网格参数
- 输出收益/年化/MDD/夏普/交易数
- 保存完整结果到 reports/grid_sweep_*.csv，便于后续筛选
"""

from __future__ import annotations

import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from qkquant.backtest.engine import BacktestEngine
from qkquant.data.storage import DuckStore
from qkquant.logger import setup_logger
from qkquant.strategy.registry import get_strategy, load_risk_config, load_strategy_config


def calc_annual_pct(timereturn: dict) -> float:
    s = pd.Series(dict(timereturn)).sort_index()
    if s.empty:
        return 0.0
    eq = (1 + s).cumprod()
    days = (pd.to_datetime(s.index[-1]) - pd.to_datetime(s.index[0])).days
    if days <= 0:
        return 0.0
    annual = float(eq.iloc[-1]) ** (365.25 / days) - 1.0
    return annual * 100.0


def run_one(
    store: DuckStore,
    codes: list[str],
    start: str,
    end: str,
    params_override: dict,
    capital: float = 30_000.0,
) -> dict:
    info = get_strategy("grid_trading")
    cfg = load_strategy_config(info)
    params = dict((cfg.get("params") or {}) if cfg else {})
    params.update(params_override)
    risk_cfg = load_risk_config(cfg)

    engine = BacktestEngine(
        store=store,
        strategy_cls=info.cls,
        strategy_params=params,
        initial_capital=capital,
        risk_config=risk_cfg,
    )
    result = engine.run(codes=codes, start=start, end=end)
    final = float(result["final_value"])
    ret_pct = (final - capital) / capital * 100.0

    dd = result["analyzers"]["drawdown"]
    mdd_pct = float(dd.get("max", {}).get("drawdown", 0.0))
    sharpe_raw = result["analyzers"]["sharpe"].get("sharperatio", None)
    sharpe = float(sharpe_raw) if sharpe_raw is not None else 0.0
    annual_pct = calc_annual_pct(result["analyzers"]["timereturn"])
    n_orders = len(getattr(result["strategy"], "_trade_log", []))

    return {
        "ret_pct": ret_pct,
        "annual_pct": annual_pct,
        "mdd_pct": mdd_pct,
        "sharpe": sharpe,
        "n_orders": n_orders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="grid_trading 参数扫描")
    parser.add_argument("--quick", action="store_true", help="快速模式（较少参数组合）")
    parser.add_argument("--start", default="2023-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-05-07", help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    setup_logger("WARNING")
    store = DuckStore()
    codes = store.load_index_constituents("000300")
    inst = store.load_instruments(codes)
    if not inst.empty and "is_st" in inst.columns:
        st_codes = set(inst[inst["is_st"]]["code"].tolist())
        codes = [c for c in codes if c not in st_codes]

    start = args.start
    end = args.end
    print(f"universe: {len(codes)} HS300 codes (ST excluded)")
    print(f"period:   {start} ~ {end}")

    # 先用较小网格做快速搜索，后续可按最优区间细化
    if args.quick:
        grid_pct_grid = [0.02, 0.03]
        base_window_grid = [10, 20]
        bear_cap_grid = [0.10, 0.20]
        bull_boost_grid = [0.10, 0.20]
        min_trade_value_grid = [2000, 3000]
    else:
        grid_pct_grid = [0.02, 0.03, 0.04]
        base_window_grid = [10, 20, 30]
        bear_cap_grid = [0.10, 0.20, 0.30]
        bull_boost_grid = [0.10, 0.15, 0.20]
        min_trade_value_grid = [2000, 3000]

    combos = list(
        itertools.product(
            grid_pct_grid,
            base_window_grid,
            bear_cap_grid,
            bull_boost_grid,
            min_trade_value_grid,
        )
    )
    print(f"total combos: {len(combos)}\n")

    results: list[dict] = []
    for i, (grid_pct, base_window, bear_cap, bull_boost, min_trade_value) in enumerate(combos, 1):
        params = {
            "grid_pct": grid_pct,
            "base_window": base_window,
            "bear_position_cap": bear_cap,
            "bull_position_boost": bull_boost,
            "min_trade_value": min_trade_value,
        }
        print(f"[{i:>3}/{len(combos)}] running {params} ...", flush=True)
        m = run_one(store, codes, start, end, params)
        # 一个简洁综合分：夏普优先，回撤惩罚
        score = (m["sharpe"] if np.isfinite(m["sharpe"]) else -9.0) - 0.02 * m["mdd_pct"]
        results.append(
            {
                **params,
                **m,
                "score": score,
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"grid_sweep_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "grid_pct",
                "base_window",
                "bear_position_cap",
                "bull_position_boost",
                "min_trade_value",
                "ret_pct",
                "annual_pct",
                "mdd_pct",
                "sharpe",
                "n_orders",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 96)
    print("Top 15 by score (sharpe - 0.02*mdd):")
    print(
        f"{'rank':<5} {'params':<58} {'累计':>8} {'年化':>8} {'MDD':>8} {'夏普':>8} {'交易':>6}"
    )
    print("-" * 96)
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    for rank, r in enumerate(ranked[:15], 1):
        ptxt = (
            f"gp={r['grid_pct']:.2f}, bw={int(r['base_window'])}, "
            f"bear={r['bear_position_cap']:.2f}, bull={r['bull_position_boost']:.2f}, "
            f"mtv={int(r['min_trade_value'])}"
        )
        print(
            f"{rank:<5} {ptxt:<58} {r['ret_pct']:>+7.2f}% {r['annual_pct']:>+7.2f}% "
            f"{r['mdd_pct']:>7.2f}% {r['sharpe']:>+7.3f} {int(r['n_orders']):>6}"
        )

    print(f"\nfull results saved: {out_csv}")
    store.close()


if __name__ == "__main__":
    main()
