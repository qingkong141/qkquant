"""Replay only audited O'Neil cases with comparable, document-backed EPS.

Run research_oneil_pit.py first. This is not a full-universe PIT backtest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from qkquant.oneil import OneilConfig, prepare_prices, run_backtest
from qkquant.oneil_data import load_snapshot
from qkquant.oneil_pit import financial_state_asof, prepare_financials_asof

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/research/oneil_pit_pilot_20260919"
EVIDENCE = ROOT / "data/research/oneil_eps_followup_20260919"
PRIOR = ROOT / "reports/oneil_pit_pilot_20260919"
OUTPUT = ROOT / "reports/oneil_eps_followup_20260919"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path, expected, hashes):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    hashes[str(path.relative_to(ROOT))] = actual


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    prior_audit = read_json(PRIOR / "audit.json")
    for relative, digest in prior_audit["source_sha256"].items():
        verify(PILOT / relative, digest, hashes)
    for relative, digest in prior_audit["evidence_json_sha256"].items():
        verify(PILOT / relative, digest, hashes)
    versions = pd.DataFrame(read_json(PRIOR / "financial_versions.json"))
    versions = versions[versions.code.isin(["605499", "600926", "002028"])].copy()
    # Basis declarations are specific to these audited reports, not a provider-wide rule.
    action_id = "605499_bonus_20241009"
    versions["eps_applied_actions"] = [
        [action_id] if r.code == "605499" and r.report_date == "2024-09-30" else []
        for r in versions.itertuples()
    ]
    precision = versions.code.map({"605499": .0001, "600926": .01, "002028": .01})
    versions["eps_lower"] = versions.eps - precision / 2
    versions["eps_upper"] = versions.eps + precision / 2
    versions["eps_value_status"] = "reported_rounded"
    dongpeng = read_json(EVIDENCE / "605499/evidence.json")
    verify(Path(dongpeng["prior_evidence_file"]), dongpeng["prior_evidence_sha256"], hashes)
    for source in dongpeng["sources"]:
        verify(EVIDENCE / "605499" / source["source_file"], source["source_sha256"], hashes)
    event = dongpeng["corporate_action"]
    source = next(s for s in dongpeng["sources"] if s["id"] == event["source_id"])
    actions = pd.DataFrame([dict(code="605499", action_id=action_id, action_type="bonus_shares",
                                 factor=float(event["share_multiplier_exact"]),
                                 published_at=event["published_at"], effective_date=event["economic_effective_date"],
                                 source_url=source["source_url"], source_sha256=source["source_sha256"])])

    # Sieyuan's conditional intervals remain diagnostic; do not substitute them
    # for directly disclosed EPS or silently change eligibility when inputs change.
    sieyuan_path = EVIDENCE / "002028/evidence.json"
    sieyuan = read_json(sieyuan_path)
    for item in read_json(EVIDENCE / "002028/sources.json"):
        verify(EVIDENCE / "002028" / item["source_file"], item["source_sha256"], hashes)
    for item in read_json(EVIDENCE / "methods/sources.json")["sources"]:
        verify(EVIDENCE / "methods" / item["local_file"], item["sha256"], hashes)
    cfg = OneilConfig()
    bars, _, benchmark, manifest = load_snapshot(ROOT / "data/research/oneil_20260919")
    prepared = prepare_prices(bars, benchmark, cfg)
    cases = []
    for code, date in (("605499", "2024-12-12"), ("600926", "2024-07-29"), ("002028", "2026-04-22")):
        day = pd.Timestamp(date)
        rows = versions[versions.code == code]
        selected = prepare_financials_asof(rows, actions, day)
        selected.to_json(OUTPUT / f"{code}_asof_{date}.json", orient="records", date_format="iso",
                         force_ascii=False, indent=2)
        state = financial_state_asof(rows, actions, day, cfg)
        technical, market = bool(prepared["technical"].loc[day, code]), bool(prepared["market_ok"].loc[day])
        cases.append(dict(code=code, signal_date=day, technical_ok=technical, market_ok=market,
                          pivot=prepared["pivot"].loc[day, code], rs=prepared["rs"].loc[day, code],
                          **state, qualified=technical and market and state["fundamental_ok"]))
    signals = pd.DataFrame(cases)
    replay = run_backtest(prepared, signals, cfg, start="2024-07-01", end="2026-07-17")
    signals.to_csv(OUTPUT / "audited_cases.csv", index=False)
    replay["trades"].to_csv(OUTPUT / "partial_replay_trades.csv", index=False)
    replay["equity"].to_csv(OUTPUT / "partial_replay_equity.csv")
    replay["rejections"].to_csv(OUTPUT / "partial_replay_rejections.csv", index=False)
    versions.to_json(OUTPUT / "financial_versions_with_eps_basis.json", orient="records", force_ascii=False, indent=2)
    actions.to_json(OUTPUT / "eps_actions.json", orient="records", force_ascii=False, indent=2)
    for path in (PRIOR / "financial_versions.json", PRIOR / "audit.json", EVIDENCE / "605499/evidence.json"):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in EVIDENCE.rglob("*.json"):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("bars.csv.gz", "benchmark.csv", "manifest.json"):
        path = ROOT / "data/research/oneil_20260919" / name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = dict(config=asdict(cfg), cases=cases, partial_replay_metrics=replay["metrics"],
                 source_sha256=hashes, universe_size=len(manifest["codes"]), evaluated_cases=len(cases),
                 full_universe_pit_verified=False, full_strategy_returns_retested=False,
                 selection="Three manually audited company/date cases only; other signals are unassessed.",
                 sieyuan_interval_evidence=sieyuan,
                 snapshot_limitations=manifest["limitations"])
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    dp, bank, sy = cases
    sells = replay["trades"].query("side == 'SELL'")
    pnl = float(sells.pnl.sum())
    report = f"""# 欧奈尔：EPS 口径修复与已核验信号局部回放

原成长和入场参数保持不变。东鹏饮料在统一 EPS 股数基础后通过原规则，局部回放产生
{replay['metrics']['buy_count']} 笔买入、{len(sells)} 笔平仓，已实现盈亏 {pnl:.2f} 元（含模型费用和滑点）。
这说明旧数据口径会导致漏掉交易，不能说明策略已被证明有效。

## 本次实际核验范围

相对强度仍用原快照的 {len(manifest['codes'])} 只股票计算；只评估以下三个公司/日期。
其余信号尚未核验，不能当作不满足条件。本报告不是全量财报修复后的收益回测。

|公司与评估日|财务结论|原技术条件|市场条件|入场资格|
|---|---|---|---|---|
|东鹏饮料 605499 / 2024-12-12|通过|{dp['technical_ok']}|{dp['market_ok']}|{dp['qualified']}|
|杭州银行 600926 / 2024-07-29|成长/ROE不达标|{bank['technical_ok']}|{bank['market_ok']}|{bank['qualified']}|
|思源电气 002028 / 2026-04-22|Q4 EPS直接披露值仍缺失；区间证据单独保留|{sy['technical_ok']}|{sy['market_ok']}|{sy['qualified']}|

## 东鹏饮料的可比 EPS

2024-09-26实施公告确认每10股转增3股，2024-10-09除权，股数倍数为1.3。
2020—2023年报和2023Q3的原EPS统一除以1.3；2024Q3原报已使用新基础，不能再除一次。
记录同时保留原值、来源版本、股数基础、调整因子和舍入区间；只有当时已公开且已生效的事件才参与换算。

季度EPS同比点估计 {dp['quarter_eps_growth']:.4%}，按原报四位小数的精度计算区间为
[{dp['quarter_eps_growth_lower']:.4%}, {dp['quarter_eps_growth_upper']:.4%}]，下界也高于25%。
原报同比78.40%与该区间在披露精度下相容；点估计本身四舍五入为78.41%，不冒充完全相等。
季度营收同比 {dp['quarter_sales_growth']:.2%}；年度EPS三年CAGR {dp['annual_eps_cagr']:.2%}，
保守下界 {dp['annual_eps_cagr_lower']:.2%}；ROE {dp['annual_roe']:.2f}%。原四项门槛均满足。

股东会日期在Q3脚注与事件专门公告之间存在差异，证据中保留此差异；比例和除权日以实施公告为依据。
检索未发现影响本次EPS的数值更正，但未声称穷尽全部历史修订链。

## 局部执行回放

本金仍为100,000元，单笔默认预算6.25%，次日开盘买入、8%止损等执行规则均未变。
逐笔日期、价格、费用和退出理由见 `partial_replay_trades.csv`。
仍使用原项目的复权价格/小数单位研究模型，尚未实现100股整手、完整分红现金账和全历史成分。
单笔亏损既不能证明策略无效，也不能因恢复一笔成交就判断值得采用。

## 思源与后续覆盖

第四季度不能默认以年度EPS减前三季度EPS。已归档原利润、股数与方法核查，
区间推导与直接披露值明确分开；本次回放不以未完全验证的推导替换原报缺值。
在普通股利润和股数调整假设成立时，2024Q4 EPS约为0.71717—0.71841，2025Q4约为1.22596—1.22748，
两位小数分别稳定显示0.72/1.23。但尚缺完整的普通股EPS分子和分母调节核对，
证据标记为 `conditional`、`eligible_for_trading=false`。满足条件的区间推导不等于公司直接披露值。
见 `data/research/oneil_eps_followup_20260919/002028` 与 `methods/eps_method_review.md`。
杭州银行字段修复后仍不达成长条件。全量22家原候选（放宽入场组为69家）的财报修订链与历史股票池仍待补齐。

复现：先运行 `scripts/research_oneil_pit.py`，再运行 `scripts/research_oneil_eps.py`。
`audit.json` 保存原配置、输入哈希、三个日期的判定和局部回放统计；旧快照与旧报告保留。

原文：[东鹏权益分派实施公告]({source['source_url']})、
[2024Q3](https://static.cninfo.com.cn/finalpage/2024-10-30/1221560302.PDF)、
[2023Q3](https://static.cninfo.com.cn/finalpage/2023-10-28/1218189869.PDF)。
"""
    (OUTPUT / "summary.md").write_text(report, encoding="utf-8")
    print(signals[["code", "signal_date", "fundamental_ok", "technical_ok", "market_ok", "qualified"]].to_string(index=False))
    print(replay["trades"][["code", "date", "side", "reason", "pnl"]].to_string(index=False))
    print(f"Report: {OUTPUT / 'summary.md'}")


if __name__ == "__main__":
    main()
