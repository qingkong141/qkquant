"""ma_boll 参数扫描。

网格: boll_dev × upper_buffer, 12 个组合
- boll_dev:      1.5 / 2.0 / 2.5
- upper_buffer:  0.01 / 0.03 / 0.05 / 0.10

输出三个表格:
  1) 按夏普排序的 ranking
  2) 夏普 heatmap (dev × buffer)
  3) 累计收益 heatmap

注意: 12 次回测找到的"最优"有 overfitting 风险, 看相邻参数稳定性更可靠。
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
    params_override: dict | None = None,
    capital: float = 100_000.0,
) -> dict:
    info = get_strategy(strategy_name)
    cfg = load_strategy_config(info)
    params = dict((cfg.get("params") or {}) if cfg else {})
    if params_override:
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
    final = result["final_value"]
    ret_pct = (final - capital) / capital * 100
    n_orders = len(getattr(result["strategy"], "_trade_log", []))

    dd = result["analyzers"]["drawdown"]
    mdd_pct = float(dd.get("max", {}).get("drawdown", 0.0))
    sharpe_raw = result["analyzers"]["sharpe"].get("sharperatio", None)
    sharpe = float(sharpe_raw) if sharpe_raw is not None else 0.0

    return {
        "ret_pct": ret_pct,
        "mdd_pct": mdd_pct,
        "sharpe": sharpe,
        "n_orders": n_orders,
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

    boll_dev_grid = [1.5, 2.0, 2.5]
    upper_buffer_grid = [0.01, 0.03, 0.05, 0.10]

    print("baselines:")
    base_breakout = run_one(store, "ma_breakout", codes, "2023-01-01", "2026-05-07")
    base_boll = run_one(store, "ma_boll", codes, "2023-01-01", "2026-05-07")
    print(
        f"  ma_breakout                : {base_breakout['ret_pct']:+7.2f}% "
        f"(sharpe {base_breakout['sharpe']:+.3f}, MDD {base_breakout['mdd_pct']:.2f}%)"
    )
    print(
        f"  ma_boll (dev=2.0, buf=0.03): {base_boll['ret_pct']:+7.2f}% "
        f"(sharpe {base_boll['sharpe']:+.3f}, MDD {base_boll['mdd_pct']:.2f}%)"
    )
    print()

    results: list[dict] = []
    for dev in boll_dev_grid:
        for buf in upper_buffer_grid:
            label = f"dev={dev:.1f}, buf={buf:.2f}"
            print(f"running {label} ...", flush=True)
            r = run_one(
                store,
                "ma_boll",
                codes,
                "2023-01-01",
                "2026-05-07",
                params_override={"boll_dev": dev, "upper_buffer": buf},
            )
            results.append({"label": label, "dev": dev, "buf": buf, **r})

    print("\n" + "=" * 80)
    print("Sorted by 夏普 (descending):")
    print(
        f"{'rank':<5} {'params':<22} {'累计':>10} {'夏普':>10} {'MDD':>8} {'交易数':>8}"
    )
    print("-" * 70)
    for i, r in enumerate(sorted(results, key=lambda x: -x["sharpe"]), 1):
        print(
            f"{i:<5} {r['label']:<22} {r['ret_pct']:>+8.2f}% "
            f"{r['sharpe']:>+10.3f} {r['mdd_pct']:>7.2f}% {r['n_orders']:>8}"
        )

    print("\n" + "=" * 80)
    print("夏普 heatmap (rows=boll_dev, cols=upper_buffer):")
    header = "         " + "".join(f"{f'buf={b:.2f}':>14}" for b in upper_buffer_grid)
    print(header)
    for dev in boll_dev_grid:
        line = f"dev={dev:.1f}".rjust(9)
        for buf in upper_buffer_grid:
            r = next(x for x in results if x["dev"] == dev and x["buf"] == buf)
            line += f"{r['sharpe']:>+14.3f}"
        print(line)

    print("\n" + "=" * 80)
    print("累计收益 heatmap (rows=boll_dev, cols=upper_buffer):")
    print(header)
    for dev in boll_dev_grid:
        line = f"dev={dev:.1f}".rjust(9)
        for buf in upper_buffer_grid:
            r = next(x for x in results if x["dev"] == dev and x["buf"] == buf)
            line += f"{r['ret_pct']:>+13.2f}%"
        print(line)

    store.close()


if __name__ == "__main__":
    main()
