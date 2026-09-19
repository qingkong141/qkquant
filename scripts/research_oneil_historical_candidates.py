"""Apply reviewed necessary-condition failures to the frozen historical cohort.

This audit can reject a financial candidate, but never certifies a financial
pass or simulates trades. Unreviewed evidence remains explicitly unresolved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from qkquant.oneil import OneilConfig, prepare_prices
from qkquant.oneil_pit import documented_financial_failure

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/research/oneil_historical_cohort_20260919"
OUTPUT = ROOT / "reports/oneil_historical_candidates_20260919"
EVIDENCE = [ROOT / "data/research" / folder / group / "evidence.json"
            for folder in ("oneil_candidate_audit_20260919", "oneil_historical_candidate_audit_20260919")
            for group in ("group_a", "group_b", "group_c")]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path, expected, hashes):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    hashes[str(path.relative_to(ROOT))] = actual


def main():
    cfg = OneilConfig()
    manifest = read_json(SNAPSHOT / "manifest.json")
    if manifest["config"] != asdict(cfg):
        raise ValueError("historical candidate audit requires unchanged original parameters")
    codes = manifest["selected_codes"]
    expected_sources = {f"{code}_{adjust}.json" for code in codes for adjust in ("3", "1")}
    if len(set(codes)) != len(codes) or manifest["selected_count"] != len(codes):
        raise ValueError("invalid frozen cohort membership")
    if set(manifest["source_sha256"]) != expected_sources:
        raise ValueError("frozen cohort source inventory is incomplete")
    benchmark_path = ROOT / "data/research/oneil_20260919/benchmark.csv"
    if str(benchmark_path.relative_to(ROOT)) not in manifest["input_sha256"]:
        raise ValueError("frozen benchmark hash is missing")
    hashes = {}
    bars_path = SNAPSHOT / "bars.csv.gz"
    verify(bars_path, manifest["bars_sha256"], hashes)
    for path, expected in manifest["input_sha256"].items():
        verify(ROOT / path, expected, hashes)
    for name, expected in manifest["source_sha256"].items():
        verify(SNAPSHOT / "sources" / name, expected, hashes)
    hashes[str((SNAPSHOT / "manifest.json").relative_to(ROOT))] = hashlib.sha256(
        (SNAPSHOT / "manifest.json").read_bytes()).hexdigest()

    decisions = []
    for path in EVIDENCE:
        payload = read_json(path)
        for record_path in path.parent.rglob("*.json"):
            hashes[str(record_path.relative_to(ROOT))] = hashlib.sha256(record_path.read_bytes()).hexdigest()
        for source in payload["sources"]:
            verify(path.parent / source["source_file"], source["source_sha256"], hashes)
        for decision in payload["decisions"]:
            for prefix in ("", "latest_annual_"):
                if decision.get(prefix + "source_file"):
                    verify(path.parent / decision[prefix + "source_file"],
                           decision[prefix + "source_sha256"], hashes)
            decisions.append({**decision, "evidence_file": str(path.relative_to(ROOT))})

    bars = pd.read_csv(bars_path, dtype={"code": str}, parse_dates=["trade_date"], float_precision="round_trip")
    if not bars.code.isin(codes).all():
        raise ValueError("price code outside frozen cohort")
    benchmark = pd.read_csv(benchmark_path, parse_dates=["trade_date"])
    prepared = prepare_prices(bars, benchmark, cfg)
    technical = prepared["technical"].loc[manifest["inception"]:manifest["price_end"]]
    if not technical.columns.isin(codes).all():
        raise ValueError("technical code outside frozen cohort")
    rows = []
    for i, j in zip(*np.where(technical.to_numpy()), strict=True):
        day, code = technical.index[i], technical.columns[j]
        vetoes = [d for d in decisions if documented_financial_failure(d, code, day, cfg)]
        market = bool(prepared["market_ok"].loc[day])
        veto = vetoes[0] if vetoes else {}
        pending = [d for d in decisions if d["code"] == code and str(day.date()) in d["signal_dates"]
                   and d.get("decision") == "unresolved" and pd.Timestamp(d.get("published_at")) < day]
        reviewed = veto or (pending[-1] if pending else {})
        rows.append(dict(code=code, signal_date=day, market_ok=market,
                         financial_status="verified_fail" if veto else "unresolved",
                         entry_status="market_rejected" if not market else "financial_rejected" if veto else "unresolved",
                         reason=veto.get("check", reviewed.get("notes", "no_verified_necessary_condition_failure")),
                         source=reviewed.get("source_url", ""), evidence_file=reviewed.get("evidence_file", ""),
                         qualified=False))
    signals = pd.DataFrame(rows, columns=["code", "signal_date", "market_ok", "financial_status",
                                         "entry_status", "reason", "source", "evidence_file", "qualified"])
    if (len(signals) != manifest["technical_events"]
            or int(signals.market_ok.sum()) != manifest["market_events"]):
        raise ValueError("frozen historical screening results changed")
    entry_counts = signals.entry_status.value_counts().to_dict()
    financial_counts = signals.financial_status.value_counts().to_dict()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUTPUT / "candidate_decisions.csv", index=False)
    signals[signals.entry_status == "unresolved"].to_csv(OUTPUT / "unresolved_entries.csv", index=False)
    audit = dict(config=asdict(cfg), universe_size=manifest["selected_count"],
                 technical_events=len(signals), candidate_companies=signals.code.nunique(),
                 entry_status_counts=entry_counts, financial_status_counts=financial_counts,
                 input_sha256=hashes, reviewed_evidence=decisions,
                 financial_passes_certified=False, returns_tested=False,
                 complete_historical_financial_database=False)
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = f"""# 欧奈尔：历史固定样本的必要财务条件核验

对回测前已公开的固定{manifest['selected_count']}只主板持仓样本，重新使用封存价格和全部原参数计算，
得到{len(signals)}次技术候选、{signals.code.nunique()}家公司。
样本、价格来源及每日相对强度分母与原246只试验不同，不能把两组候选数量直接解释成参数效果。

- 市场条件不通过：{entry_counts.get('market_rejected', 0)}次。
- 市场通过、原财报证明至少一项必要财务条件不通过：{entry_counts.get('financial_rejected', 0)}次。
- 市场通过、仍需补充财务核验：{entry_counts.get('unresolved', 0)}次。

本脚本只接收可复核的失败证据，不能证明整体财务通过，全部qualified=false。
这些数字不代表零成交的完整回测结果；本轮不模拟交易或计算收益。
每条证据限定公司和信号日期，并检查原公告时间、最新适用报告、口径与原门槛。
未核验不冒充失败，较晚公布的报表不用于较早的信号。

逐信号结果见candidate_decisions.csv，仍影响入场判断的待办见unresolved_entries.csv；
audit.json保留固定参数、原报告出处与输入哈希。
文档最初按不完整下载缓存安排核验优先级，此处只统计完整冻结样本实际产生的信号。
完整财报版本库、合格候选的全项证明、退市结算及整手/分红执行模型仍未完成。

复现：`.\\.venv\\Scripts\\python.exe scripts\\research_oneil_universe.py`，
再运行`.\\.venv\\Scripts\\python.exe scripts\\research_oneil_historical_candidates.py`。
"""
    (OUTPUT / "summary.md").write_text(report, encoding="utf-8")
    print(f"Historical cohort: {len(signals)} technical; {entry_counts}")


if __name__ == "__main__":
    main()
