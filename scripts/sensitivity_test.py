"""成本敏感性测试：在不同 (佣金, 滑点) 组合下跑同一批策略。

5 个场景 × 3 策略 = 15 次回测。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qkquant.backtest.engine import BacktestEngine
from qkquant.data.storage import DuckStore
from qkquant.strategy.registry import (
    get_strategy,
    load_risk_config,
    load_strategy_config,
)


def run_one(
    store: DuckStore,
    strategy_name: str,
    codes: list[str],
    start: str,
    end: str,
    commission: float,
    slippage: float,
    capital: float = 30_000.0,
) -> dict:
    info = get_strategy(strategy_name)
    cfg = load_strategy_config(info)
    params = (cfg.get("params") or {}) if cfg else {}
    risk_cfg = load_risk_config(cfg)

    engine = BacktestEngine(
        store=store,
        strategy_cls=info.cls,
        strategy_params=params,
        initial_capital=capital,
        slippage_pct=slippage,
        risk_config=risk_cfg,
    )
    engine.cfg.commission_rate = commission  # 运行时覆盖

    result = engine.run(codes=codes, start=start, end=end)
    final = result["final_value"]
    ret_pct = (final - capital) / capital * 100
    n_orders = len(getattr(result["strategy"], "_trade_log", []))
    dd = result["analyzers"]["drawdown"]
    mdd_pct = float(getattr(dd.get("max", {}), "drawdown", 0.0))
    return {
        "final": final,
        "ret_pct": ret_pct,
        "n_orders": n_orders,
        "mdd_pct": mdd_pct,
    }


def main() -> None:
    store = DuckStore()
    codes = store.load_index_constituents("000300")
    inst = store.load_instruments(codes)
    if not inst.empty and "is_st" in inst.columns:
        st_codes = set(inst[inst["is_st"]]["code"].tolist())
        codes = [c for c in codes if c not in st_codes]

    print(f"universe: {len(codes)} HS300 codes (ST excluded)")
    print(f"period:   2023-01-01 ~ 2026-05-07\n")

    scenarios = [
        # (label, commission_rate, slippage_pct)
        ("baseline (万 2.5 + 滑 0.2%)",   0.00025, 0.002),
        ("低佣 (万 1.5 + 滑 0.2%)",       0.00015, 0.002),
        ("高滑点 (万 2.5 + 滑 0.5%)",     0.00025, 0.005),
        ("现实 (万 1.5 + 滑 0.5%)",       0.00015, 0.005),
        ("最坏 (万 2.5 + 滑 1.0%)",       0.00025, 0.010),
    ]

    strategies = ["momentum", "momentum_breakout", "ma_breakout"]

    matrix: dict[str, dict[str, dict]] = {}
    for label, comm, slip in scenarios:
        matrix[label] = {}
        for s in strategies:
            print(f"running [{label}] x [{s}] ...", flush=True)
            matrix[label][s] = run_one(
                store, s, codes, "2023-01-01", "2026-05-07", comm, slip
            )

    print("\n" + "=" * 90)
    print(f"{'场景':<32} | " + " | ".join(f"{s:^20}" for s in strategies))
    print("-" * 100)
    for label in [s[0] for s in scenarios]:
        cells = []
        for st in strategies:
            r = matrix[label][st]
            cells.append(f"{r['ret_pct']:+6.2f}% (MDD {r['mdd_pct']:.1f}%, {r['n_orders']:>2}笔)")
        print(f"{label:<32} | " + " | ".join(f"{c:<20}" for c in cells))

    print("\n表格列含义：累计收益率（最大回撤，下单次数）")
    store.close()


if __name__ == "__main__":
    main()
