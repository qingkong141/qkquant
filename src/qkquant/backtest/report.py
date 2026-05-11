"""回测绩效报告：净值曲线图、trades.csv、summary.md。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qkquant.config import get_settings
from qkquant.logger import logger

# 设置中文字体（Windows 常见）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def calc_equity_curve(timereturn: dict, initial_capital: float) -> pd.Series:
    """由 backtrader TimeReturn analyzer 的结果还原净值曲线。"""
    if not timereturn:
        return pd.Series(dtype=float)
    s = pd.Series(timereturn).sort_index()
    s.index = pd.to_datetime(s.index)
    equity = (1 + s).cumprod() * initial_capital
    return equity


# 向后兼容旧别名
_calc_equity_curve = calc_equity_curve


def calc_metrics(result: dict) -> dict:
    initial = result["initial_capital"]
    final = result["final_value"]
    timereturn = result["analyzers"]["timereturn"]
    equity = calc_equity_curve(timereturn, initial)

    total_return = final / initial - 1
    n_days = len(equity)
    ann_return = (final / initial) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0.0

    dd = result["analyzers"]["drawdown"]
    max_dd = dd.get("max", {}).get("drawdown", 0.0) / 100.0 if dd else 0.0

    sharpe = result["analyzers"]["sharpe"]
    sharpe_ratio = sharpe.get("sharperatio") if sharpe else None

    trades = result["analyzers"]["trades"]
    total_trades = trades.get("total", {}).get("closed", 0) if trades else 0
    won = trades.get("won", {}).get("total", 0) if trades else 0
    lost = trades.get("lost", {}).get("total", 0) if trades else 0
    win_rate = (won / total_trades) if total_trades else 0.0
    avg_win = trades.get("won", {}).get("pnl", {}).get("average", 0.0) if trades else 0.0
    avg_loss = abs(trades.get("lost", {}).get("pnl", {}).get("average", 0.0)) if trades else 0.0
    profit_factor = (avg_win * won) / (avg_loss * lost) if (lost and avg_loss) else None

    return {
        "initial_capital": initial,
        "final_value": final,
        "total_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "equity_curve": equity,
    }


def _plot_equity(equity: pd.Series, initial: float, out_path: Path, title: str) -> None:
    if equity.empty:
        logger.warning("empty equity curve; skipping plot")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, linewidth=1.4, label="净值")
    ax1.axhline(initial, linestyle="--", linewidth=0.8, alpha=0.5, label="初始资金")
    ax1.set_title(title)
    ax1.set_ylabel("账户总市值")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _extract_trades(strat) -> pd.DataFrame:
    """从策略对象的 trades analyzer 提不出每笔明细；
    我们改用 strategy._trades 回放：这里简化为从 notify_trade 累积的列表。
    backtrader 默认没有把每笔 trade 做成 DataFrame，我们遍历 cerebro 的 broker 历史不稳定，
    退而求其次：读取 observer 或直接从 analyzers.trades 提取聚合数据并写一份简化 CSV。"""
    # 如果策略作者主动把交易写入 self._trade_log，则优先用它
    trade_log = getattr(strat, "_trade_log", None)
    if trade_log:
        return pd.DataFrame(trade_log)
    return pd.DataFrame()


def generate_report(
    result: dict,
    strategy_name: str,
    report_root: Path | None = None,
) -> Path:
    """写入 reports/<strategy>_<ts>/ 目录并返回该目录路径。"""
    cfg = get_settings().backtest
    root = report_root or cfg.report_root_abs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"{strategy_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = calc_metrics(result)
    equity = metrics.pop("equity_curve")

    equity.to_csv(out_dir / "equity.csv", header=["total_value"], index_label="date")
    _plot_equity(
        equity,
        result["initial_capital"],
        out_dir / "equity.png",
        title=f"{strategy_name} 回测 {result['period']['start']} ~ {result['period']['end']}",
    )

    trades_df = _extract_trades(result["strategy"])
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)

    rejections = result.get("rejections", [])
    if rejections:
        pd.DataFrame(rejections).to_csv(out_dir / "rejections.csv", index=False)

    def _fmt_pct(x):
        return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "n/a"

    def _fmt_num(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"

    summary = (
        f"# {strategy_name} 回测报告\n\n"
        f"- 回测区间: {result['period']['start']} ~ {result['period']['end']}\n"
        f"- 股票池大小: {len(result['codes'])}\n"
        f"- 耗时: {result['elapsed_seconds']:.2f}s\n\n"
        f"## 核心指标\n\n"
        f"| 指标 | 值 |\n|---|---|\n"
        f"| 初始资金 | {metrics['initial_capital']:.2f} |\n"
        f"| 最终资金 | {metrics['final_value']:.2f} |\n"
        f"| 累计收益 | {_fmt_pct(metrics['total_return'])} |\n"
        f"| 年化收益 | {_fmt_pct(metrics['annualized_return'])} |\n"
        f"| 最大回撤 | {_fmt_pct(metrics['max_drawdown'])} |\n"
        f"| 夏普比率 | {_fmt_num(metrics['sharpe_ratio'])} |\n"
        f"| 总交易数 | {metrics['total_trades']} |\n"
        f"| 胜率 | {_fmt_pct(metrics['win_rate'])} |\n"
        f"| 平均盈利 | {_fmt_num(metrics['avg_win'])} |\n"
        f"| 平均亏损 | {_fmt_num(metrics['avg_loss'])} |\n"
        f"| 盈亏比 | {_fmt_num(metrics['profit_factor'])} |\n\n"
        f"## 订单拦截\n\n"
        f"- 被 T+1 / 涨跌停规则拦截的订单数: {len(rejections)}\n"
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    logger.info(f"report written to {out_dir}")
    return out_dir


def print_summary(result: dict) -> None:
    """在终端打印核心指标。"""
    metrics = calc_metrics(result)
    equity = metrics.pop("equity_curve")  # noqa: F841

    def pct(x):
        return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "n/a"

    def num(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"

    print("=" * 56)
    print(f" 回测区间 : {result['period']['start']} ~ {result['period']['end']}")
    print(f" 股票池   : {len(result['codes'])}")
    print(f" 初始资金 : {metrics['initial_capital']:.2f}")
    print(f" 最终资金 : {metrics['final_value']:.2f}")
    print(f" 累计收益 : {pct(metrics['total_return'])}")
    print(f" 年化收益 : {pct(metrics['annualized_return'])}")
    print(f" 最大回撤 : {pct(metrics['max_drawdown'])}")
    print(f" 夏普比率 : {num(metrics['sharpe_ratio'])}")
    print(f" 总交易数 : {metrics['total_trades']}")
    print(f" 胜率     : {pct(metrics['win_rate'])}")
    print(f" 盈亏比   : {num(metrics['profit_factor'])}")
    print("=" * 56)


def generate_compare_report(
    results: dict[str, dict],
    report_root: Path | None = None,
) -> Path:
    """跨策略对比报告：叠加净值图 + 指标对比表。

    ``results`` 形如 {strategy_name: engine.run(...) 的返回值}。
    """
    if not results:
        raise ValueError("results is empty")
    cfg = get_settings().backtest
    root = report_root or cfg.report_root_abs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    equities: dict[str, pd.Series] = {}
    initial = next(iter(results.values()))["initial_capital"]

    for name, result in results.items():
        m = calc_metrics(result)
        equities[name] = m.pop("equity_curve")
        rows.append(
            {
                "strategy": name,
                "final_value": m["final_value"],
                "total_return": m["total_return"],
                "annualized_return": m["annualized_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe_ratio": m["sharpe_ratio"],
                "total_trades": m["total_trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
            }
        )

    # 1. 指标对比 CSV + MD
    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(out_dir / "comparison.csv")

    def _pct(x):
        return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "n/a"

    def _num(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"

    md_rows = ["# 策略对比报告", ""]
    period = next(iter(results.values()))["period"]
    md_rows.append(f"- 回测区间: {period['start']} ~ {period['end']}")
    md_rows.append(f"- 初始资金: {initial:.2f}")
    md_rows.append(f"- 策略数: {len(results)}")
    md_rows.append("")
    md_rows.append("| 策略 | 最终资金 | 累计 | 年化 | 最大回撤 | 夏普 | 交易数 | 胜率 | 盈亏比 |")
    md_rows.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md_rows.append(
            "| {s} | {fv:.0f} | {tr} | {ar} | {dd} | {sh} | {nt} | {wr} | {pf} |".format(
                s=r["strategy"],
                fv=r["final_value"],
                tr=_pct(r["total_return"]),
                ar=_pct(r["annualized_return"]),
                dd=_pct(r["max_drawdown"]),
                sh=_num(r["sharpe_ratio"]),
                nt=r["total_trades"],
                wr=_pct(r["win_rate"]),
                pf=_num(r["profit_factor"]),
            )
        )
    (out_dir / "comparison.md").write_text("\n".join(md_rows), encoding="utf-8")

    # 2. 叠加净值图
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, eq in equities.items():
        if eq.empty:
            continue
        ax.plot(eq.index, eq.values, linewidth=1.3, label=name)
    ax.axhline(initial, linestyle="--", linewidth=0.8, alpha=0.5, label="初始资金")
    ax.set_title(f"策略净值对比 {period['start']} ~ {period['end']}")
    ax.set_xlabel("日期")
    ax.set_ylabel("账户总市值")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_dir / "equity_overlay.png", dpi=120)
    plt.close(fig)

    # 3. 合并每日净值到同一个 CSV 方便后续分析
    if equities:
        eq_df = pd.concat(equities, axis=1)
        eq_df.to_csv(out_dir / "equities.csv", index_label="date")

    logger.info(f"compare report written to {out_dir}")
    return out_dir


def print_compare(results: dict[str, dict]) -> None:
    """终端打印对比摘要表。"""
    def _pct(x):
        return f"{x*100:+7.2f}%" if isinstance(x, (int, float)) else "    n/a"

    def _num(x):
        return f"{x:7.2f}" if isinstance(x, (int, float)) else "    n/a"

    print("=" * 90)
    print(
        f"{'strategy':<20} {'final':>10} {'total':>9} {'annual':>9} "
        f"{'maxDD':>9} {'sharpe':>8} {'trades':>7} {'winrate':>9}"
    )
    print("-" * 90)
    for name, result in results.items():
        m = calc_metrics(result)
        print(
            f"{name:<20} {m['final_value']:>10.0f} {_pct(m['total_return'])} "
            f"{_pct(m['annualized_return'])} {_pct(m['max_drawdown'])} "
            f"{_num(m['sharpe_ratio'])} {m['total_trades']:>7} "
            f"{_pct(m['win_rate'])}"
        )
    print("=" * 90)


__all__ = [
    "calc_equity_curve",
    "calc_metrics",
    "generate_compare_report",
    "generate_report",
    "print_compare",
    "print_summary",
]
