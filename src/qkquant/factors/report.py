"""因子评估报告：IC 统计表、分组净值图、多空曲线、markdown 汇总。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qkquant.config import get_settings
from qkquant.factors.library import FactorSpec
from qkquant.logger import logger

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------- 单因子报告 ----------------


def _plot_quintile_equity(
    quintile_daily: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    equity = (1 + quintile_daily.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(equity.columns)))
    for i, col in enumerate(equity.columns):
        ax.plot(equity.index, equity[col], label=col, linewidth=1.3, color=colors[i])
    ax.axhline(1.0, linestyle="--", linewidth=0.7, alpha=0.5, color="grey")
    ax.set_title(title)
    ax.set_ylabel("分组累计净值（初始 = 1）")
    ax.legend(loc="best", ncol=len(equity.columns))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_long_short(
    ls_daily: pd.Series,
    out_path: Path,
    title: str,
) -> None:
    equity = (1 + ls_daily.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, linewidth=1.4, color="steelblue")
    ax.axhline(1.0, linestyle="--", linewidth=0.7, alpha=0.5, color="grey")
    ax.set_title(title)
    ax.set_ylabel("多空净值（Q_top - Q_bottom）")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_ic_by_horizon(
    ic_stats: pd.DataFrame,  # index=horizon, columns include ic_mean, ic_ir
    out_path: Path,
    title: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ic_stats))
    ax1.bar(x, ic_stats["ic_mean"].values, alpha=0.6, label="IC mean", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h}d" for h in ic_stats.index])
    ax1.set_ylabel("IC mean", color="steelblue")
    ax1.axhline(0, linestyle="--", linewidth=0.7, alpha=0.5, color="grey")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, ic_stats["ic_ir"].values, "o-", color="darkorange", label="ICIR")
    ax2.set_ylabel("ICIR", color="darkorange")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fmt_pct(x) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:+.2f}%"


def _fmt_num(x, digits: int = 4) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


def generate_factor_report(
    spec: FactorSpec,
    ic_stats_by_horizon: dict[int, dict],
    ic_series_by_horizon: dict[int, pd.Series],
    primary_horizon: int,
    quintile_daily: pd.DataFrame,
    quintile_stats: pd.DataFrame,
    long_short: pd.Series,
    period: tuple[str, str],
    report_root: Path | None = None,
) -> Path:
    """写入 reports/factors/<name>_<ts>/ 并返回路径。"""
    cfg = get_settings().backtest
    root = report_root or (cfg.report_root_abs / "factors")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"{spec.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    ic_stats_df = pd.DataFrame(ic_stats_by_horizon).T
    ic_stats_df.index.name = "horizon"
    ic_stats_df.to_csv(out_dir / "ic_stats.csv")

    ic_ts = pd.DataFrame(
        {f"ic_{h}d": ic_series_by_horizon[h] for h in sorted(ic_series_by_horizon)}
    )
    ic_ts.to_csv(out_dir / "ic_timeseries.csv", index_label="date")

    quintile_daily.to_csv(out_dir / "quintile_daily_returns.csv", index_label="date")
    quintile_stats.to_csv(out_dir / "quintile_stats.csv", index=False)

    # --- 图 ---
    _plot_quintile_equity(
        quintile_daily,
        out_dir / "quintile_equity.png",
        title=f"{spec.name} 分组累计净值（forward={primary_horizon}d）",
    )
    _plot_long_short(
        long_short,
        out_dir / "long_short.png",
        title=f"{spec.name} 多空组合净值（Q_top - Q_bottom, forward={primary_horizon}d）",
    )
    _plot_ic_by_horizon(
        ic_stats_df,
        out_dir / "ic_by_horizon.png",
        title=f"{spec.name} IC across forward horizons",
    )

    # --- summary.md ---
    # 方向判断：用 primary horizon 的 IC mean 符号 vs 预期方向
    primary_ic_mean = ic_stats_by_horizon[primary_horizon]["ic_mean"]
    sign_match = (
        (primary_ic_mean > 0 and spec.expected_sign > 0)
        or (primary_ic_mean < 0 and spec.expected_sign < 0)
    )
    sign_mark = "✅" if sign_match else "⚠️"

    lines = []
    lines.append(f"# 因子评估报告：{spec.name}")
    lines.append("")
    lines.append(f"- 描述：{spec.description}")
    lines.append(f"- 预期方向：{'+1 (越大越好)' if spec.expected_sign > 0 else '-1 (越小越好)'}")
    lines.append(f"- 回测区间：{period[0]} ~ {period[1]}")
    lines.append(f"- Primary forward：{primary_horizon}d")
    lines.append(f"- 方向一致性：{sign_mark} (primary IC mean = {_fmt_num(primary_ic_mean)})")
    lines.append("")

    lines.append("## IC 统计（不同预测期）")
    lines.append("")
    lines.append("| Forward | IC 均值 | IC std | ICIR | t-stat | IC>0 比例 | 样本数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for h in sorted(ic_stats_by_horizon):
        s = ic_stats_by_horizon[h]
        lines.append(
            f"| {h}d | {_fmt_num(s['ic_mean'])} | {_fmt_num(s['ic_std'])} | "
            f"{_fmt_num(s['ic_ir'])} | {_fmt_num(s['t_stat'], 2)} | "
            f"{_fmt_pct(s['positive_ratio'])} | {s['n_samples']} |"
        )
    lines.append("")

    lines.append(f"## 分组回测（forward = {primary_horizon}d）")
    lines.append("")
    lines.append("| 分组 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 天数 |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in quintile_stats.iterrows():
        lines.append(
            f"| {row['group']} | {_fmt_pct(row['cum_return'])} | {_fmt_pct(row['ann_return'])} | "
            f"{_fmt_pct(row['ann_vol'])} | {_fmt_num(row['sharpe'], 2)} | {int(row['n_days'])} |"
        )

    ls_cum = float((1 + long_short.fillna(0)).prod() - 1)
    ls_sharpe = (
        float(long_short.mean() / long_short.std(ddof=1) * np.sqrt(252))
        if long_short.std(ddof=1) > 0
        else np.nan
    )
    lines.append("")
    lines.append(f"**多空组合（Q_top - Q_bottom）**：累计 {_fmt_pct(ls_cum)}，Sharpe {_fmt_num(ls_sharpe, 2)}")
    lines.append("")

    lines.append("## 图表")
    lines.append("")
    lines.append("- `ic_by_horizon.png`: 不同 forward 的 IC 均值与 ICIR")
    lines.append(f"- `quintile_equity.png`: 5 组累计净值（primary forward={primary_horizon}d）")
    lines.append("- `long_short.png`: 多空组合净值")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"factor report written to {out_dir}")
    return out_dir


# ---------------- 多因子对比报告 ----------------


def generate_factor_comparison_report(
    results: dict[str, dict],  # factor_name -> {ic_stats_by_horizon, ic_stats_df, period, primary_horizon, spec}
    report_root: Path | None = None,
) -> Path:
    """横向对比多个因子。输入来自多次 single-factor 评估后的聚合。"""
    cfg = get_settings().backtest
    root = report_root or (cfg.report_root_abs / "factors")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / f"all_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 聚合到一张表：因子 × (horizon, metric)
    rows = []
    for fname, r in results.items():
        spec: FactorSpec = r["spec"]
        for h, s in r["ic_stats_by_horizon"].items():
            rows.append({
                "factor": fname,
                "expected_sign": spec.expected_sign,
                "horizon": h,
                "ic_mean": s["ic_mean"],
                "ic_std": s["ic_std"],
                "ic_ir": s["ic_ir"],
                "t_stat": s["t_stat"],
                "positive_ratio": s["positive_ratio"],
                "n_samples": s["n_samples"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ic_stats.csv", index=False)

    # 宽表便于阅读：每个因子一行，列是不同 horizon 的 ic_mean / ic_ir
    pivot_mean = df.pivot(index="factor", columns="horizon", values="ic_mean")
    pivot_ir = df.pivot(index="factor", columns="horizon", values="ic_ir")
    pivot_mean.columns = [f"IC_{c}d" for c in pivot_mean.columns]
    pivot_ir.columns = [f"ICIR_{c}d" for c in pivot_ir.columns]
    wide = pd.concat([pivot_mean, pivot_ir], axis=1)
    wide.to_csv(out_dir / "ic_stats_wide.csv")

    # summary.md
    lines = []
    lines.append("# 多因子 IC 横向对比")
    lines.append("")
    period = next(iter(results.values()))["period"]
    lines.append(f"- 回测区间：{period[0]} ~ {period[1]}")
    lines.append(f"- 因子数：{len(results)}")
    horizons = sorted(df["horizon"].unique())
    lines.append(f"- Forward windows：{', '.join(str(h) + 'd' for h in horizons)}")
    lines.append("")

    lines.append("## IC 均值（越大越好，符号应与预期方向一致）")
    lines.append("")
    lines.append("| 因子 | 预期 | " + " | ".join(f"IC_{h}d" for h in horizons) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(horizons)) + "|")
    for fname in results:
        spec = results[fname]["spec"]
        cells = []
        for h in horizons:
            s = results[fname]["ic_stats_by_horizon"][h]
            cells.append(_fmt_num(s["ic_mean"]))
        lines.append(f"| {fname} | {'+1' if spec.expected_sign > 0 else '-1'} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## ICIR（|值|>0.3 通常视为有效，>0.5 较强）")
    lines.append("")
    lines.append("| 因子 | " + " | ".join(f"ICIR_{h}d" for h in horizons) + " |")
    lines.append("|---|" + "|".join(["---"] * len(horizons)) + "|")
    for fname in results:
        cells = []
        for h in horizons:
            s = results[fname]["ic_stats_by_horizon"][h]
            cells.append(_fmt_num(s["ic_ir"]))
        lines.append(f"| {fname} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 方向一致性检查")
    lines.append("")
    lines.append("| 因子 | 预期 | primary IC | 方向 |")
    lines.append("|---|---|---|---|")
    for fname in results:
        spec = results[fname]["spec"]
        ph = results[fname]["primary_horizon"]
        ic_mean = results[fname]["ic_stats_by_horizon"][ph]["ic_mean"]
        match = (
            (ic_mean > 0 and spec.expected_sign > 0)
            or (ic_mean < 0 and spec.expected_sign < 0)
        )
        lines.append(
            f"| {fname} | {'+1' if spec.expected_sign > 0 else '-1'} | "
            f"{_fmt_num(ic_mean)} | {'✅' if match else '⚠️'} |"
        )
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"factor comparison report written to {out_dir}")
    return out_dir


__all__ = [
    "generate_factor_comparison_report",
    "generate_factor_report",
]
