"""组合策略测试: 把多个独立策略按权重合成, 看相关性和组合优势。

方法:
- 分别跑 momentum / momentum_breakout / ma_boll, 提取每日收益序列
- 计算两两相关性
- 合成多个权重组合的"虚拟组合"日收益: combined_r = sum(w_i * r_i)
- 输出每个组合的累计/年化/MDD/夏普

注意:
- 这模拟"每天按目标权重 rebalance"的组合, 不是简单按初始资金分配
- 相关性低的策略组合后通常 MDD 改善, 夏普可能优于单一最强策略
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from qkquant.backtest.engine import BacktestEngine
from qkquant.data.storage import DuckStore
from qkquant.strategy.registry import (
    get_strategy,
    load_risk_config,
    load_strategy_config,
)


def run_strategy(
    store: DuckStore,
    name: str,
    codes: list[str],
    start: str,
    end: str,
    capital: float = 30_000.0,
) -> pd.Series:
    """跑一个策略, 返回日收益 Series。"""
    info = get_strategy(name)
    cfg = load_strategy_config(info)
    params = (cfg.get("params") or {}) if cfg else {}
    risk_cfg = load_risk_config(cfg)

    engine = BacktestEngine(
        store=store,
        strategy_cls=info.cls,
        strategy_params=params,
        initial_capital=capital,
        risk_config=risk_cfg,
    )
    result = engine.run(codes=codes, start=start, end=end)

    tr = result["analyzers"]["timereturn"]  # OrderedDict {date: daily_return}
    s = pd.Series(dict(tr)).sort_index()
    s.index = pd.to_datetime(s.index)
    s.name = name
    return s


def metrics(daily_rets: pd.Series, risk_free_annual: float = 0.02) -> dict:
    """从日收益序列算指标。"""
    eq = (1 + daily_rets).cumprod()
    final_ret = eq.iloc[-1] - 1
    days = (daily_rets.index[-1] - daily_rets.index[0]).days
    annual = (eq.iloc[-1]) ** (365.25 / max(days, 1)) - 1

    rolling_max = eq.cummax()
    dd = eq / rolling_max - 1
    mdd = float(dd.min())

    daily_excess = daily_rets - risk_free_annual / 252
    sharpe = (
        float(daily_excess.mean() / daily_excess.std() * np.sqrt(252))
        if daily_excess.std() > 0
        else 0.0
    )
    return {
        "ret_pct": final_ret * 100,
        "annual_pct": annual * 100,
        "mdd_pct": mdd * 100,
        "sharpe": sharpe,
    }


def main() -> None:
    store = DuckStore()
    codes = store.load_index_constituents("000300")
    inst = store.load_instruments(codes)
    if not inst.empty and "is_st" in inst.columns:
        st_codes = set(inst[inst["is_st"]]["code"].tolist())
        codes = [c for c in codes if c not in st_codes]

    print(f"universe: {len(codes)} HS300 codes")
    print(f"period:   2023-01-01 ~ 2026-05-07\n")

    strategies = ["momentum", "momentum_breakout", "ma_boll"]
    series_list: list[pd.Series] = []
    for name in strategies:
        print(f"running {name} ...", flush=True)
        series_list.append(run_strategy(store, name, codes, "2023-01-01", "2026-05-07"))

    df = pd.concat(series_list, axis=1).fillna(0.0)
    print()

    # 单策略指标
    print("=" * 80)
    print("单策略表现:")
    print(
        f"  {'name':<22} {'累计':>9}  {'年化':>8}  {'MDD':>8}  {'夏普':>8}"
    )
    for name in strategies:
        m = metrics(df[name])
        print(
            f"  {name:<22} {m['ret_pct']:>+8.2f}%  {m['annual_pct']:>+7.2f}%  "
            f"{m['mdd_pct']:>+7.2f}%  {m['sharpe']:>+7.3f}"
        )

    print()
    print("=" * 80)
    print("日收益相关性:")
    corr = df.corr().round(3)
    print(corr.to_string())

    print()
    print("=" * 80)
    print("组合方案 (每日按目标权重 rebalance):")
    print(
        f"  {'组合':<46} {'累计':>9}  {'年化':>8}  {'MDD':>8}  {'夏普':>8}"
    )
    combos = [
        ("momentum 100%",                          {"momentum": 1.0}),
        ("momentum_breakout 100%",                 {"momentum_breakout": 1.0}),
        ("ma_boll 100%",                           {"ma_boll": 1.0}),
        ("70% momentum + 30% ma_boll",             {"momentum": 0.7, "ma_boll": 0.3}),
        ("50% momentum + 50% ma_boll",             {"momentum": 0.5, "ma_boll": 0.5}),
        ("70% momentum_breakout + 30% ma_boll",    {"momentum_breakout": 0.7, "ma_boll": 0.3}),
        ("50% momentum_breakout + 50% ma_boll",    {"momentum_breakout": 0.5, "ma_boll": 0.5}),
        ("33/33/33 三策略等权",                     {"momentum": 1/3, "momentum_breakout": 1/3, "ma_boll": 1/3}),
        ("50% momentum + 50% momentum_breakout",   {"momentum": 0.5, "momentum_breakout": 0.5}),
    ]
    for label, w in combos:
        weights = pd.Series(w).reindex(df.columns).fillna(0)
        combined_daily = (df * weights).sum(axis=1)
        m = metrics(combined_daily)
        print(
            f"  {label:<46} {m['ret_pct']:>+8.2f}%  {m['annual_pct']:>+7.2f}%  "
            f"{m['mdd_pct']:>+7.2f}%  {m['sharpe']:>+7.3f}"
        )

    store.close()


if __name__ == "__main__":
    main()
