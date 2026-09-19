"""Verify the four-company document pilot; do not run a return backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from qkquant.oneil import OneilConfig, financial_state
from qkquant.oneil_pit import select_financial_versions

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/research/oneil_pit_pilot_20260919")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/oneil_pit_pilot_20260919")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, hashes = [], {}
    for folder, filename in (("002028", "records.json"), ("600926", "records.json"), ("correction", "versions.json")):
        for source in read_json(args.evidence / folder / filename):
            row = dict(source)
            row["published_at"] = row.get("published_at", row.get("notice_date"))
            row["version_id"] = row.get("version_id", row.get("announcement_id", row.get("metadata_announcement_id")))
            file = Path(row["source_file"])
            if not file.is_absolute():
                file = args.evidence / folder / file
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != row["source_sha256"]:
                raise ValueError(f"source hash mismatch: {file}")
            hashes[str(file.relative_to(args.evidence))] = digest
            records.append(row)

    # Preserve Dongpeng's raw share bases; do not fabricate comparable prior EPS
    # by dividing a current EPS by the rounded, published growth rate.
    dongpeng = read_json(args.evidence / "605499/evidence.json")
    documents = {d["id"]: d for d in dongpeng["documents"]}
    for doc in documents.values():
        file = args.evidence / "605499" / doc["local_pdf"]
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != doc["sha256"]:
            raise ValueError(f"source hash mismatch: {file}")
        hashes[str(file.relative_to(args.evidence))] = digest
    grouped = {}
    for fact in dongpeng["facts"]:
        metric = fact["metric"]
        field = {"basic_eps": "eps", "basic_eps_original": "eps", "revenue": "revenue",
                 "annual_basic_eps_original_basis": "eps", "annual_basic_eps_comparative": "eps",
                 "weighted_average_roe": "roe"}.get(metric)
        if field is None:
            continue
        doc = documents[fact["document_id"]]
        kind = "annual" if "annual" in fact["document_id"] else "quarter"
        key = kind, fact["report_period_end"], fact["document_id"]
        row = grouped.setdefault(key, dict(code="605499", kind=kind, report_date=key[1],
                                            published_at=fact["publication_date"], eps=None, revenue=None, roe=None,
                                            source_url=fact["source_url"], source_sha256=doc["sha256"],
                                            version_id=fact["document_id"], source_pages=[],
                                            eps_comparability="raw reported basis; cross-period alignment not approved"))
        row[field] = fact["value"]
        row["source_pages"].append(fact["pdf_page_one_based"])
    records.extend(grouped.values())
    frame = pd.DataFrame(records)
    # Also hash the correction notice, which establishes the changed version's date.
    for source in read_json(args.evidence / "correction/sources.json"):
        file = args.evidence / "correction" / source["filename"]
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"source hash mismatch: {file}")
        hashes[str(file.relative_to(args.evidence))] = digest
    frame.to_json(args.output / "financial_versions.json", orient="records", force_ascii=False, indent=2)

    boundaries = []
    for day, expected in (("2026-04-27", None), ("2026-04-28", 1.94),
                          ("2026-05-07", 1.94), ("2026-05-08", 1.38)):
        chosen = select_financial_versions(frame[frame.code == "301607"], day)
        actual = None if chosen.empty else chosen.eps.item()
        if actual != expected:
            raise AssertionError((day, actual, expected))
        boundaries.append(dict(day=day, expected_eps=expected, actual_eps=actual, passed=True))

    cfg, assessments = OneilConfig(), {}
    for code, day in (("600926", "2024-07-29"), ("002028", "2026-04-22")):
        selected = select_financial_versions(frame[frame.code == code], day)
        selected.to_csv(args.output / f"{code}_asof_{day}.csv", index=False)
        # Both dates now refer to verified publication, never vendor UPDATE_DATE.
        inputs = selected.assign(notice_date=selected.published_at, update_date=selected.published_at)
        assessments[code] = financial_state(inputs, day, cfg)
    select_financial_versions(frame[frame.code == "605499"], "2024-12-12").to_csv(
        args.output / "605499_raw_asof_2024-12-12.csv", index=False)
    facts = dongpeng["facts"]
    annual = sorted((f for f in facts if f["metric"] == "annual_basic_eps_original_basis"),
                    key=lambda f: f["report_period_end"])
    used = annual + [f for f in facts if f["metric"] in (
        "basic_eps_yoy_as_disclosed", "revenue_yoy_as_disclosed", "weighted_average_roe")]
    if not all(pd.Timestamp(f["publication_date"]) < pd.Timestamp("2024-12-12") for f in used):
        raise AssertionError("future document in Dongpeng threshold diagnostic")
    eps = [f["value"] for f in annual]
    cagr = (eps[-1] / eps[0]) ** (1 / 3) - 1
    eps_growth = next(f["value"] / 100 for f in facts if f["metric"] == "basic_eps_yoy_as_disclosed")
    revenue_growth = next(f["value"] / 100 for f in facts if f["metric"] == "revenue_yoy_as_disclosed")
    roe = next(f["value"] for f in facts if f["metric"] == "weighted_average_roe")
    assessments["605499"] = dict(
        quarter_eps_growth_as_disclosed=eps_growth, quarter_sales_growth_as_disclosed=revenue_growth,
        annual_eps_cagr_on_common_pre_bonus_basis=cagr, annual_roe=roe,
        document_thresholds_pass=bool(eps_growth >= cfg.quarterly_growth and revenue_growth >= cfg.quarterly_growth
                                     and cagr >= cfg.annual_cagr and roe >= cfg.min_roe
                                     and min(eps) > 0 and all(b > a for a, b in zip(eps[:-1], eps[1:], strict=True))),
        complete_machine_pit_verified=False)
    audit = dict(config=asdict(cfg), source_sha256=hashes, records=len(frame), assessments=assessments,
                 correction_boundaries=boundaries, full_universe_pit_verified=False, returns_retested=False)
    audit["evidence_json_sha256"] = {str(p.relative_to(args.evidence)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(args.evidence.rglob("*.json"))}
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(boundaries).to_csv(args.output / "correction_boundaries.csv", index=False)

    report = f"""# 历史财报原公告核验：四家公司试点

已存证并校验 {len(hashes)} 份原始 PDF 的哈希，整理 {len(frame)} 条带发布日期和来源的财务版本记录。
目标是核验历史数据路径和指标口径，未扩大投资池，也未重跑收益或修改成长/入场参数。

|案例|核验结果|对上一轮结果的影响|
|---|---|---|
|600926 杭州银行|四年年报和两个同比季度可恢复；银行营收字段应为 OPERATE_INCOME|缺失字段可以修复，但仍不满足原成长门槛|
|605499 东鹏饮料|原报与快照存在季度EPS及年度EPS股本口径差异|不能继续解释为必须降低CAGR门槛才可通过|
|002028 思源电气|四年EPS、ROE及第四季度营收已核实|第四季度EPS尚未直接核验，原试跑成交不能视为已通过本次财务验收|
|301607 富特科技|同一期年报EPS从1.94更正到1.38，版本日期可明确区分|更正前后查询验证通过；创业板样本仅用于数据测试|

## 东鹏饮料：修正前轮调参解释

截至2024-12-12已公开的原报告：单季度可比EPS同比 {eps_growth:.2%}、营收同比 {revenue_growth:.2%}；
2020—2023年同一转增前口径年度EPS依次为 {', '.join(str(x) for x in eps)} 元，均为正且逐年增长，
三年复合增速 {cagr:.2%}；2023年加权平均ROE {roe:.2f}%。这些值达到原四项财务阈值。

旧快照2023年度EPS为3.9225，约等于原值5.0993除以1.3，前三年未作同样处理，导致原计算CAGR约20.25%。
旧快照2023Q3 EPS为0.4136，而原报告当时披露1.3687；2024Q3 EPS为1.8784，已考虑转增后的比较口径。
不能直接把后两个原EPS相除，也没有从舍入后的78.40%反推一个精确比较EPS。
本次依据公司直接披露的可比同比作门槛诊断；未把原EPS混合导入原有financial_state。
2020年EPS来自信号日前发布的后续年报比较列，未声称这是其首次披露版本。

## 杭州银行：字段修复与成长判断分开

2024Q1营收9,760,688,000元，2023Q1为9,430,545,000元；对应EPS为0.82、0.67元。
季度EPS增长 {assessments['600926']['quarter_eps_growth']:.2%}、营收增长 {assessments['600926']['quarter_sales_growth']:.2%}，
年度CAGR {assessments['600926']['annual_eps_cagr']:.2%}、ROE {assessments['600926']['annual_roe']:.2f}%。
部分值不达原阈值，所以补齐后仍不合格。采集器已修复银行季度营收字段，旧快照没有覆盖重写。

## 思源电气：未验证的第四季度EPS保持缺失

2026-04-22信号之前最新季度是2025Q4，而不是4月25日才公告的2026Q1。
已核四年年度EPS为1.59、2.02、2.64、4.04元，2025年ROE22.63%；2024/2025Q4营收分别为
5,050,801,337.10元、7,712,054,219.53元。核验的年报分季度表没有列出基本EPS。
原供应商季度EPS0.72和1.23的计算方法仍待核实，本次保留为空，不以累计EPS相减补值。
因此本次严格原文输入拒绝该财务状态，不能把之前1笔模拟成交视为完全验证后的信号。

## 更正日期验证

富特科技原年报公告2026-04-27，EPS1.94；更正年报公告2026-05-07，EPS1.38。
更正年报封面仍写4月27日，实际版本时间必须看官方公告元数据和更正公告。
保守按公告日期之后一天使用：4月27日无值、4月28日1.94、5月7日仍1.94、5月8日1.38，均验证通过。

## 适用范围与后续数据路线

这是小样本存证及选择器验收，不是完整历史财报数据库。没有证明已穷尽更正链，也没有自动验证网页真实性。
每期多版本按经过核验的发布日期选择；供应商UPDATE_DATE仅可保留为元数据，不能替代真实更正日期。
新增选择器已测试未来版本隔离、缺值不回退和同日冲突拒绝，尚未替换原回测数据入口。

建议下一步优先验收结构化来源的历史版本、单季度EPS和送转股可比口径，再扩展到22家原候选公司。
公开原公告可用于复核；单靠批量抓PDF仍难保证所有公司Q4 EPS和全部修订链完整。
暂不据旧快照挑选最佳参数，不由本次四家公司推断全样本收益。

证据目录：`{args.evidence.resolve()}`。各子目录保留原PDF、官方查询响应、页码、哈希及关键页面核验结果。
`financial_versions.json`保存原文版本，605499仍保留原股本口径，不能当作已经统一口径的回测输入。
`audit.json`保存字段诊断和原固定配置，`correction_boundaries.csv`保存真实更正边界测试。

复现：`.\\.venv\\Scripts\\python.exe scripts\\research_oneil_pit.py`
"""
    (args.output / "summary.md").write_text(report, encoding="utf-8")
    print(f"Verified {len(hashes)} source PDFs; {len(frame)} financial versions; four correction boundaries passed.")
    print(f"Report: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
