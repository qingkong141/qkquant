"""端到端：合成 5 只股票 1 年日线，跑 ma_breakout 回测。

这条测试不访问网络，只验证框架组装得对、回测能正常完成且核心指标可计算。
"""

from __future__ import annotations

from qkquant.backtest.engine import BacktestEngine
from qkquant.backtest.report import generate_report
from qkquant.strategy.ma_breakout import MaBreakoutStrategy


def test_end_to_end_backtest(tmp_store, tmp_path):
    engine = BacktestEngine(
        store=tmp_store,
        strategy_cls=MaBreakoutStrategy,
        strategy_params={"fast": 5, "slow": 20, "max_positions": 3},
        initial_capital=100_000.0,
    )
    codes = tmp_store.load_index_constituents("000300")
    result = engine.run(codes=codes, start="2023-01-01", end="2024-12-31", adjust="hfq")

    assert result["final_value"] > 0
    assert result["codes"], "at least one feed should be loaded"
    assert "timereturn" in result["analyzers"]
    assert "drawdown" in result["analyzers"]

    out_dir = generate_report(result, strategy_name="ma_breakout_test", report_root=tmp_path)
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "equity.csv").exists()
    # equity png may fail to render on headless without Agg? we configured Agg — should work.
    assert (out_dir / "equity.png").exists()


def test_safe_sell_respects_t1(tmp_store):
    """验证 T+1：当 _buy_dates 记录为今天时不允许卖。"""
    from datetime import date

    from qkquant.backtest.engine import BtStrategyBase

    class Dummy:
        def __init__(self, code, pct):
            self._name = code
            self.pct_chg = [pct]

    s = BtStrategyBase.__new__(BtStrategyBase)
    s._buy_dates = {"X": date.today()}
    s._rejections = []
    s._order_reasons = {}
    s.datas = [Dummy("X", 0.0)]

    # patch _today -> today
    s._today = lambda: date.today()
    assert not s._can_sell(Dummy("X", 0.0)), "T+1 should block same-day sell"
