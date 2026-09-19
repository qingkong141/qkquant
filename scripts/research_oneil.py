"""Run the isolated O'Neil experiment; public downloads require --fetch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qkquant.oneil import OneilConfig, build_signals, performance, prepare_prices, run_backtest
from qkquant.oneil_data import fetch_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download/cache public financials and raw prices")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/research/oneil_20260919")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/oneil_20260919")
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()
    if args.fetch:
        fetch_snapshot(ROOT / "data/daily.duckdb", args.snapshot)
    bars, financials, benchmark, manifest = load_snapshot(args.snapshot)
    cfg = OneilConfig(capital=args.capital)
    prepared = prepare_prices(bars, benchmark, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    runs, summaries = {}, []
    for policy in ("updated", "notice"):
        signals = build_signals(prepared, financials, cfg, policy)
        result = run_backtest(prepared, signals, cfg, args.start, args.end)
        window = signals[signals.signal_date.between(result["metrics"]["start"], result["metrics"]["end"])]
        runs[policy] = result
        summaries.append({"financial_policy": policy, **result["metrics"],
                          "technical_events": len(window), "qualified_signals": int(window.qualified.sum())})
        folder = args.output / policy
        folder.mkdir(exist_ok=True)
        signals.to_csv(folder / "signals.csv", index=False)
        result["equity"].to_csv(folder / "equity.csv")
        result["trades"].to_csv(folder / "trades.csv", index=False)
        result["rejections"].to_csv(folder / "rejections.csv", index=False)
        window.groupby(["reason", "market_ok"], dropna=False).size().rename("events").to_csv(folder / "funnel.csv")
        checks = [c for c in ("quarter_eps_ok", "quarter_sales_ok", "annual_growth_ok", "roe_ok") if c in window]
        window[checks].sum().rename("passing_events").to_csv(folder / "financial_checks.csv")
        pd.DataFrame.from_dict(result["holdings"], orient="index").to_csv(folder / "open_positions.csv")
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    dates = runs["updated"]["equity"].index
    index = prepared["benchmark"].reindex(dates)
    benchmark_metrics = performance(index)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.snapshot / name for name in ("bars.csv.gz", "financials.csv", "benchmark.csv", "manifest.json"))}
    audit = {"config": runs["updated"]["config"], "input_sha256": hashes, "snapshot": manifest,
             "benchmark": benchmark_metrics, "results": summaries,
             "account_model": "fractional_adjusted_price_units", "real_execution_validated": False}
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, run in runs.items():
        ax.plot(run["equity"].index, run["equity"].equity / cfg.capital,
                label="Conservative revision dates" if policy == "updated" else "Notice dates (exploratory)")
    ax.plot(index.index, index / index.iloc[0], label="CSI 300 price index", color="gray", alpha=.7)
    ax.set(title="O'Neil flat-base research: fixed rules, no Chan", ylabel="Normalized equity")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "equity.png", dpi=150)
    plt.close(fig)
    table = ["|财报日期口径|累计收益|年化收益|最大回撤|买入次数|平均仓位|合格信号|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        label = "公告与更新日取较晚（主结果）" if row["financial_policy"] == "updated" else "仅公告日（探索性，存在修订前视风险）"
        table.append(f"|{label}|{row['total_return']:.2%}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|"
                     f"{row['buy_count']}|{row['average_exposure']:.2%}|{row['qualified_signals']}|")
    diagnosis = ""
    if not summaries[0]["buy_count"]:
        diagnosis = "主结果没有交易，表中0%只表示资金未参与交易，不能用来评价策略的赚钱能力或风险控制效果。"
    funnels = []
    for policy in runs:
        funnel = pd.read_csv(args.output / policy / "funnel.csv")
        details = "；".join(f"{r.reason}、市场过滤{'通过' if r.market_ok else '未通过'}：{r.events}次"
                           for r in funnel.itertuples(index=False))
        funnels.append(f"- {policy}：{details}。")
    report = f"""# 纯欧奈尔日线研究试跑

区间：{dates[0].date()}—{dates[-1].date()}。初始资金：{cfg.capital:,.0f}元。
输入：{len(manifest['codes'])}只现存主板样本；财报覆盖{manifest['financial_codes']}只；数据截止{manifest['end']}。
成交量统一使用腾讯日线手数×100转换为股；{manifest.get('quarantined_turnover_rows', 0)}条成交额/成交量异常记录不参与量价信号计算。

{chr(10).join(table)}

{diagnosis}

技术候选的过滤原因：

{chr(10).join(funnels)}

同期沪深300价格指数收益{benchmark_metrics['total_return']:.2%}，最大回撤{benchmark_metrics['max_drawdown']:.2%}。
指数未扣投资成本，也不是同仓位组合。低仓位不能简单解释为选股产生了超额收益。

## 固定规则

- 当前季度EPS与营收同比均≥25%；四个连续年度EPS为正、逐年增长，三年复合增速≥25%；最近年报ROE≥17%。
- 252日股价相对强度前20%；此前20日日均成交额≥1亿元。
- 25日平底整理，深度≤15%，整理前60日上涨≥20%；收盘突破此前平台高点，量能≥此前50日均量1.4倍；买价距突破点不超过5%。
- 沪深300收盘在MA60以上，MA60高于五日前；只限制新开仓。
- 收盘确认后次交易日开盘执行；8%保护止损，买入当日止损延至次日；一般20%止盈，15个交易日内从突破点上涨20%触发40个交易日持有规则。
- 放量跌破MA50与保护止损优先于持有规则；入场条件消失不自动卖出。
- 单笔计划风险0.5%、单股上限15%、最多5只；因此默认初始每笔仓位实际约6.25%。
- 单边滑点0.2%，佣金万2.5、最低5元，双边过户费万0.1；印花税使用2023-08-28前后税率。费率是研究假设。

## 解读限制

主结果按公告日与更新日较晚者的下一自然日才允许使用财报；这能避免把本次记录的后续修订直接提前使用，
但可能让历史可用信息严重不足，并不能恢复原始历史财报。仅公告日结果不能作为可信样本外收益。
零交易或少量交易不能证明策略无效，也不能用放松参数制造成绩。

{chr(10).join('- ' + s for s in manifest['limitations'])}
- 当前数据不是最新行情；没有补齐研究日至数据截止日之间的区间。
- 账户使用可分割的复权价格单位，未实现100股整数手、真实公司行动现金/股份/红利税账本；不应当作实盘成交复现。
- 原始价格只用于下一开盘及保护止损价格的涨跌停近似检查；缺价时不交易，持仓使用最近有效价格估值。队列、成交容量与历史ST状态未完整建模。
- 平底形态、市场确认均是预先固定的机械代理；未量化新产品、机构持仓等CAN SLIM定性条件，不能称为完整原版CAN SLIM。
- 本次是固定参数初步试跑，未经独立样本外检验，未调整任何ETF生产配置。

## 文件与复现

`comparison.csv`保存对照，`audit.json`保存配置与输入哈希；各日期口径目录保存信号、过滤原因、逐笔交易、拒单、未平仓与净值。

```powershell
.\\.venv\\Scripts\\python.exe scripts\\research_oneil.py
# 需要重新取公共数据时显式增加 --fetch（缓存已存在的数据版本）
```

规则来源：[IBD平底形态](https://get.investors.com/wp-content/uploads/2024/08/IBDD-How-to-buy-Stocks-infographic.pdf)、
[IBD投资规则](https://shop.investors.com/images/promotional/20-Rules_102808.pdf)。
印花税依据：[财政部、税务总局2023年第39号公告](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)。
"""
    (args.output / "summary.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Report: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
