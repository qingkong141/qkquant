"""端到端：relative_strength 和 momentum 策略在合成数据上能跑通。

以及验证 compare 报告生成。
"""

from __future__ import annotations

from qkquant.backtest.engine import BacktestEngine
from qkquant.backtest.report import (
    calc_metrics,
    generate_compare_report,
)
from qkquant.strategy.ma_breakout import MaBreakoutStrategy
from qkquant.strategy.momentum import MomentumStrategy
from qkquant.strategy.relative_strength import RelativeStrengthStrategy


def _run(store, strategy_cls, params):
    engine = BacktestEngine(
        store=store,
        strategy_cls=strategy_cls,
        strategy_params=params,
        initial_capital=100_000.0,
    )
    codes = store.load_index_constituents("000300")
    return engine.run(codes=codes, start="2023-01-01", end="2024-12-31", adjust="hfq")


def test_relative_strength_runs(tmp_store):
    result = _run(
        tmp_store,
        RelativeStrengthStrategy,
        {"lookback": 30, "rebalance_days": 10, "top_k": 3},
    )
    assert result["final_value"] > 0
    m = calc_metrics(result)
    assert m["initial_capital"] == 100_000.0


def test_momentum_runs(tmp_store):
    result = _run(
        tmp_store,
        MomentumStrategy,
        {
            "mom_window": 20,
            "entry_threshold": 0.0,
            "exit_threshold": -0.05,
            "max_positions": 3,
        },
    )
    assert result["final_value"] > 0
    m = calc_metrics(result)
    assert m["initial_capital"] == 100_000.0


def test_compare_report(tmp_store, tmp_path):
    ma = _run(tmp_store, MaBreakoutStrategy, {"fast": 5, "slow": 20, "max_positions": 3})
    rs = _run(
        tmp_store,
        RelativeStrengthStrategy,
        {"lookback": 30, "rebalance_days": 10, "top_k": 3},
    )
    out_dir = generate_compare_report(
        {"ma_breakout": ma, "relative_strength": rs}, report_root=tmp_path
    )
    assert (out_dir / "comparison.md").exists()
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "equity_overlay.png").exists()
    assert (out_dir / "equities.csv").exists()
