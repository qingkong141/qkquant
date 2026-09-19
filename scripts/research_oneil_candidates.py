"""Audit every original technical candidate using reviewed necessary conditions.

Sparse failure evidence can reject entries; only the prior complete case audit
can approve one. Unresolved candidates remain distinct from financial failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from qkquant.oneil import OneilConfig, prepare_prices, run_backtest
from qkquant.oneil_data import load_snapshot
from qkquant.oneil_pit import documented_financial_failure, financial_state_asof

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/research/oneil_candidate_audit_20260919"
PRIOR = ROOT / "reports/oneil_eps_followup_20260919"
OUTPUT = ROOT / "reports/oneil_candidate_audit_20260919"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path, expected, hashes):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    hashes[str(path.relative_to(ROOT))] = actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, help="Additional reviewed evidence.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    hashes, decisions = {}, []
    evidence_paths = [EVIDENCE / group / "evidence.json" for group in ("group_a", "group_b", "group_c")]
    if args.supplement:
        evidence_paths.append(args.supplement.resolve())
    for evidence_path in evidence_paths:
        folder = evidence_path.parent
        payload = read_json(evidence_path)
        for source in payload["sources"]:
            verify(folder / source["source_file"], source["source_sha256"], hashes)
        for item in payload["decisions"]:
            row = {**item, "evidence_file": str((folder / "evidence.json").relative_to(ROOT))}
            if row.get("source_file"):
                verify(folder / row["source_file"], row["source_sha256"], hashes)
            if row.get("latest_annual_source_file"):
                verify(folder / row["latest_annual_source_file"], row["latest_annual_source_sha256"], hashes)
            decisions.append(row)
        for path in folder.rglob("*.json"):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    prior_audit = read_json(PRIOR / "audit.json")
    for path, digest in prior_audit["source_sha256"].items():
        verify(ROOT / path, digest, hashes)
    financials = pd.DataFrame(read_json(PRIOR / "financial_versions_with_eps_basis.json"))
    actions = pd.DataFrame(read_json(PRIOR / "eps_actions.json"))
    prior_scope = {(c["code"], str(pd.Timestamp(c["signal_date"]).date())) for c in prior_audit["cases"]}
    cfg = OneilConfig()
    if asdict(cfg) != prior_audit["config"]:
        raise ValueError("candidate audit requires the unchanged original configuration")
    snapshot = ROOT / "data/research/oneil_20260919"
    bars, _, benchmark, manifest = load_snapshot(snapshot)
    prepared = prepare_prices(bars, benchmark, cfg)
    technical = prepared["technical"].loc["2024-07-01":"2026-07-17"]
    rows = []
    for i, j in zip(*np.where(technical.to_numpy()), strict=True):
        day, code = technical.index[i], technical.columns[j]
        date = str(day.date())
        candidates = [d for d in decisions if d["code"] == code and date in d["signal_dates"]]
        vetoes = [d for d in candidates if documented_financial_failure(d, code, day, cfg)]
        status, reason, source = "unresolved", "no_verified_financial_decision", ""
        if vetoes:
            status, reason, source = "verified_fail", vetoes[0]["check"], vetoes[0]["source_url"]
        elif (code, date) in prior_scope:
            state = financial_state_asof(financials[financials.code == code], actions, day, cfg)
            if state["fundamental_ok"]:
                status, reason = "verified_pass", "prior_document_and_eps_basis_audit"
            elif state["reason"] == "growth_filter":
                status, reason = "verified_fail", "prior_document_growth_failure"
            source = str((PRIOR / "audit.json").relative_to(ROOT))
        elif candidates:
            reason = candidates[-1].get("notes", "evidence_not_sufficient_for_veto")
            source = candidates[-1].get("source_url", "")
        market = bool(prepared["market_ok"].loc[day])
        qualified = market and status == "verified_pass"
        # A market rejection settles this entry even if the financial data remain
        # unresolved; it must not be counted as a verified financial failure.
        entry = ("market_rejected" if not market else "qualified" if qualified else
                 "financial_rejected" if status == "verified_fail" else "unresolved")
        rows.append(dict(code=code, signal_date=day, pivot=prepared["pivot"].loc[day, code],
                         rs=prepared["rs"].loc[day, code], market_ok=market,
                         financial_status=status, reason=reason, source=source,
                         entry_status=entry, qualified=qualified))
    signals = pd.DataFrame(rows)
    baseline = pd.read_csv(ROOT / "reports/oneil_20260919/updated/signals.csv", dtype={"code": str},
                           parse_dates=["signal_date"])
    baseline = baseline[baseline.signal_date.between("2024-07-01", "2026-07-17")]
    if set(zip(signals.code, signals.signal_date, strict=True)) != set(zip(baseline.code, baseline.signal_date, strict=True)):
        raise AssertionError("original technical candidate set changed")
    replay = run_backtest(prepared, signals, cfg, start="2024-07-01", end="2026-07-17")
    signals.to_csv(output / "candidate_decisions.csv", index=False)
    signals[signals.entry_status == "unresolved"].to_csv(output / "unresolved_entries.csv", index=False)
    replay["trades"].to_csv(output / "verified_entries_trades.csv", index=False)
    replay["equity"].to_csv(output / "verified_entries_equity.csv")
    replay["rejections"].to_csv(output / "execution_rejections.csv", index=False)
    financial_counts = signals.financial_status.value_counts().to_dict()
    entry_counts = signals.entry_status.value_counts().to_dict()
    for path in [PRIOR / n for n in ("audit.json", "financial_versions_with_eps_basis.json", "eps_actions.json")]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [snapshot / n for n in ("bars.csv.gz", "benchmark.csv", "manifest.json")]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(config=asdict(cfg), universe_size=len(manifest["codes"]), candidate_companies=signals.code.nunique(),
                 technical_events=len(signals), financial_status_counts=financial_counts, entry_status_counts=entry_counts,
                 verified_entries_replay_metrics=replay["metrics"], input_sha256=hashes, evidence=decisions,
                 complete_historical_financial_database=False, expanded_stock_universe=False,
                 all_entry_decisions_resolved=not signals.entry_status.eq("unresolved").any(),
                 limitations=manifest["limitations"])
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    labels = {"verified_fail": "原文证实不达标", "verified_pass": "已核验通过", "unresolved": "财务仍待核验"}
    entry_labels = {"market_rejected": "市场条件不通过", "financial_rejected": "财务条件不通过",
                    "qualified": "合格入场信号", "unresolved": "待核验，未模拟买入"}
    lines = ["|股票|信号日期|财务核验|最终入场判断|", "|---|---|---|---|"]
    for row in signals.itertuples():
        lines.append(f"|{row.code}|{row.signal_date.date()}|{labels[row.financial_status]}|{entry_labels[row.entry_status]}|")
    sells = replay["trades"].query("side == 'SELL'")
    supplement_note = ("新增一季度原报证据：2025-08-15所用季度EPS同比约19.6%，即使计入披露舍入误差也低于25%，"
                       "已独立拒绝该信号；2025-10-15仍受年度可比缺口影响。" if args.supplement else "")
    reproduce = r".\.venv\Scripts\python.exe scripts\research_oneil_candidates.py"
    if args.supplement:
        reproduce += f' --supplement "{args.supplement.as_posix()}"'
    reproduce += f' --output "{output.as_posix()}"'
    report = f"""# 欧奈尔原候选：逐项财报证据核验

保持全部原参数，对原246只股票产生的 **{signals.code.nunique()}家公司、{len(signals)}次技术信号**逐一登记。
本轮扩大了候选财报的核验范围，没有扩大股票池，也没有建立完整的历史财报数据库。

## 入场判断

- 市场条件不通过：{entry_counts.get('market_rejected', 0)}次。
- 市场通过、原报证明至少一项财务条件不通过：{entry_counts.get('financial_rejected', 0)}次。
- 市场和财务均已核验通过：{entry_counts.get('qualified', 0)}次。
- 市场通过但财务证据仍不足：{entry_counts.get('unresolved', 0)}次，保留为待核验。

财务层单独计数：已证实失败{financial_counts.get('verified_fail', 0)}次、通过{financial_counts.get('verified_pass', 0)}次、
仍待核验{financial_counts.get('unresolved', 0)}次。市场拒绝不冒充财务核验完成。

## 如何判定

ROE低于17%、所需年度EPS非正，或可比年度EPS未逐年增长，任何一项就足以否定入场。
本次以信号日前原公告核实这些必要条件；不要求为已经能明确排除的候选补满所有季度。
这只证明特定信号不合格，不代表该公司所有财务历史已经恢复。通过候选则继续要求完整成长检查。
证据记录公告日、报告期、页码、哈希、检索范围和指标单位；较晚的公告不能用于较早的信号。

新华保险601336涉及新保险/新金融工具准则切换，原年报不同年度列存在不可比说明。
旧准则EPS下降不能替代信号时所需可比序列的失败证明，因此保持待核验。
{supplement_note}
金融类采集器还发现证券、保险营收字段与银行一样是`OPERATE_INCOME`，已补齐映射；旧快照保留。
字段存在不等于会计口径已验证，修复不会自动解除该公司的待核验状态。

## 已核验入场的执行回放

产生{replay['metrics']['buy_count']}笔买入、{len(sells)}笔平仓，已实现盈亏{sells.pnl.sum():.2f}元，
本金100,000元，含原模型费用和滑点。逐笔见`verified_entries_trades.csv`。
待核验信号没有被证明不合格，因此这仍是已核验入场的有限回放，不能当作完整策略收益或采用依据。
原复权价格、小数单位持仓、现存股票样本及历史成分偏差等限制仍然存在。

## 逐信号结果

{chr(10).join(lines)}

## 复现与证据

运行`{reproduce}`。
此前的`research_oneil_pit.py`和`research_oneil_eps.py`输出为已核验案例依赖。
`candidate_decisions.csv`保存逐信号状态，`unresolved_entries.csv`列出尚影响成交判断的缺口，
`audit.json`保存固定配置、源文件哈希及原公告链接。
原PDF、公告目录和更正检索位于`data/research/oneil_candidate_audit_20260919/group_a`、`group_b`、`group_c`。
已检查接口分页计数与唯一公告ID；查询结果中的未见更正仍不等于穷尽所有可能的历史修订。
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"Companies: {signals.code.nunique()}, technical events: {len(signals)}")
    print(f"Financial: {financial_counts}; entry decisions: {entry_counts}")
    print(f"Verified entries: {replay['metrics']['buy_count']}; closed PnL: {sells.pnl.sum():.2f}")
    print(f"Report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
