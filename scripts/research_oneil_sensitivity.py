"""Four fixed O'Neil entry/filter scenarios; diagnose activity, not optimize returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from qkquant.oneil import OneilConfig, build_signals, prepare_prices, run_backtest
from qkquant.oneil_data import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/research/oneil_20260919")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/oneil_sensitivity_20260919")
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    bars, financials, benchmark, manifest = load_snapshot(args.snapshot)
    execution_cfg = OneilConfig()
    financial_changes = dict(quarterly_growth=.20, annual_cagr=.20, min_roe=15)
    entry_changes = dict(volume_multiple=1.2, rs_percentile=.70, min_amount=50_000_000, prior_gain=.15)
    scenarios = {
        "baseline": ("原参数", execution_cfg),
        "financial_moderate": ("仅放宽成长", replace(execution_cfg, **financial_changes)),
        "entry_moderate": ("仅放宽入场", replace(execution_cfg, **entry_changes)),
        "combined": ("两组同时放宽", replace(execution_cfg, **financial_changes, **entry_changes)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary, check_rows, ablations = [], [], []
    checks = ["quarter_eps_ok", "quarter_sales_ok", "annual_growth_ok", "roe_ok"]
    for name, (label, cfg) in scenarios.items():
        prepared = prepare_prices(bars, benchmark, cfg)
        for policy in ("updated", "notice"):
            signals = build_signals(prepared, financials, cfg, policy)
            # volume_multiple also controls MA50 exits. Keep execution rules fixed
            # so entry sensitivity cannot silently alter exit timing.
            result = run_backtest(prepared, signals, execution_cfg, args.start, args.end)
            metrics = result["metrics"]
            window = signals[signals.signal_date.between(metrics["start"], metrics["end"])]
            summary.append({"scenario": name, "label": label, "policy": policy,
                            "technical_events": len(window), "market_events": int(window.market_ok.sum()),
                            "qualified_signals": int(window.qualified.sum()), **metrics})
            folder = args.output / name / policy
            folder.mkdir(parents=True, exist_ok=True)
            window.to_csv(folder / "signals.csv", index=False)
            for key in ("trades", "rejections"):
                result[key].to_csv(folder / f"{key}.csv", index=False)
            result["equity"].to_csv(folder / "equity.csv")
            pd.DataFrame.from_dict(result["holdings"], orient="index").to_csv(folder / "open_positions.csv")
            window.groupby(["reason", "market_ok"], dropna=False).size().rename("events").to_csv(folder / "funnel.csv")
            # Missing/nonpositive inputs remain failures even in this diagnostic.
            valid = window.reindex(columns=checks).eq(True)
            for check in checks:
                check_rows.append({"scenario": name, "policy": policy, "check": check,
                                   "passing_events": int(valid[check].sum()),
                                   "passing_market_events": int((valid[check] & window.market_ok).sum())})
                if name == "baseline" and policy == "notice":
                    all_inputs_valid = window.reason.isin(["pass", "growth_filter"])
                    passed = all_inputs_valid & window.market_ok & valid.drop(columns=check).all(axis=1)
                    ablations.append({"omitted_check": check, "qualified_signals": int(passed.sum())})

    comparison = pd.DataFrame(summary)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    pd.DataFrame(check_rows).to_csv(args.output / "financial_checks.csv", index=False)
    pd.DataFrame(ablations).to_csv(args.output / "omit_one_check_diagnostic.csv", index=False)
    audit = {
        "purpose": "Fixed sensitivity scenarios; no return-based parameter selection or out-of-sample claim",
        "input_manifest": manifest,
        "input_sha256": {name: hashlib.sha256((args.snapshot / name).read_bytes()).hexdigest()
                         for name in ("bars.csv.gz", "financials.csv", "benchmark.csv")},
        "signal_configs": {name: asdict(cfg) for name, (_, cfg) in scenarios.items()},
        "execution_config": asdict(execution_cfg), "start": args.start, "end": args.end,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    table = ["|对照组|技术候选|市场通过|保守口径合格信号/买入|公告口径合格信号/买入|",
             "|---|---:|---:|---:|---:|"]
    for name, (label, _) in scenarios.items():
        pair = comparison[comparison.scenario == name].set_index("policy")
        conservative, exploratory = pair.loc["updated"], pair.loc["notice"]
        table.append(f"|{label}|{conservative.technical_events}|{conservative.market_events}|"
                     f"{conservative.qualified_signals}/{conservative.buy_count}|"
                     f"{exploratory.qualified_signals}/{exploratory.buy_count}|")
    report = f"""# 欧奈尔零成交：固定参数敏感性诊断

区间 {comparison.start.iloc[0]}—{comparison.end.iloc[0]}；{len(manifest['codes'])} 只现存主板样本。
保守口径（updated）要求财报公告日与更新日均早于信号日；公告口径（notice）忽略更新日，
使用了本次下载的修订后数值，有前视风险，只能诊断筛选是否太严。

{chr(10).join(table)}

## 实验约定

四组在运行前固定，无收益排名、网格搜索或最佳参数选择。原研究默认参数保持不变。

|参数|原值|放宽组值|
|---|---:|---:|
|季度 EPS 与营收同比门槛|25%|20%|
|三年年度 EPS CAGR|25%|20%|
|ROE|17%|15%|
|入场放量倍数|1.4|1.2|
|252 日涨幅排名|前 20%|前 30%|
|20 日日均成交额|1 亿元|5000 万元|
|平台前 60 日涨幅|20%|15%|

“仅放宽成长”改变前三项，“仅放宽入场”改变后四项，“两组同时放宽”改变全部七项。
仍要求四年 EPS 为正且逐年增长，25 日平台深度不超过 15%，大盘趋势通过，买入区间不超过 5%。
8% 止损、20% 一般止盈、持有规则、仓位、费用及 T+1 保持原设置。
卖出时的放量门槛固定为 1.4 倍，避免修改入场参数同时修改退出规则。
放宽值只是本次敏感性假设，不是欧奈尔原规则或已经验证的 A 股适用参数。

## 如何解读

- 保守口径被财报版本日期拦截时，降低增长门槛不能恢复当时缺失的历史信息。
- 公告口径有成交，只说明这些假设下信号和执行链路能产生交易；不能据此认定有收益优势。
- 技术候选数、市场通过数、合格信号数和买入次数不是同一指标；已持仓时不重复买入，次日价格超出买区等仍会取消买入。
- 当前仍只是 {len(manifest['codes'])} 只现存主板股票的样本，存在选择与幸存者偏差；未覆盖其他欧奈尔底部形态。
- 模型使用可分割复权价格单位，缺少真实公司行动账本、整数手与完整成交约束。
- 本轮已观察数据，不能把同一区间叫作独立样本外验证；须先补齐历史财报版本和样本，再预先固定规则验证。

各组目录保存信号、拒单、逐笔成交、净值和期末持仓。`comparison.csv` 包含收益等研究指标，
公告口径收益有前视风险，不能用于选优。`financial_checks.csv` 的通过数为独立计数，不能相加。
`omit_one_check_diagnostic.csv` 在原技术候选与市场条件不变时逐项移除成长约束，
只统计诊断信号，不运行交易，也不放行缺失或非正财务输入。
`audit.json` 保存各组完整参数、固定执行参数和输入哈希。

复现：`.\\.venv\\Scripts\\python.exe scripts\\research_oneil_sensitivity.py`

原规则参考：[MarketSmith CAN SLIM](https://www.marketsmith.hk/can-slim-overview/?lang=en)、
[IBD 平底形态](https://shop.investors.com/images/promotional/flat-b-b_112408.pdf)。
"""
    (args.output / "summary.md").write_text(report, encoding="utf-8")
    print(comparison[["scenario", "policy", "technical_events", "market_events", "qualified_signals", "buy_count"]].to_string(index=False))
    print(f"Report: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
