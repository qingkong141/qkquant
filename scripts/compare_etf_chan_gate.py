"""Compare the existing Chan rule switch with all other portfolio rules fixed."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import validate_verified_etf as validation
from qkquant.data import verified
from qkquant.etf_portfolio_backtest import run_etf_portfolio_backtest


def comparison_configs(manifest: dict) -> tuple[dict, object, dict]:
    costs, risk, portfolio = validation.frozen_configs(manifest)
    cases = {}
    for cost, cfg in costs.items():
        cases[f"chan_on_{cost}"] = cfg
        cases[f"chan_off_{cost}"] = replace(cfg, chan_filter_enabled=False)
    return cases, risk, portfolio


def verify_reference(data, reference: Path, revisions: dict) -> dict:
    meta = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or meta["source_revision"] != revisions:
        raise ValueError("Reference must be complete and use identical strategy sources.")
    for key in ("database_sha256", "snapshot_manifest_sha256", "freeze_manifest_sha256", "calendar_sha256"):
        if meta["snapshot"][key] != data.metadata[key]:
            raise ValueError(f"Reference snapshot differs: {key}")
    costs, risk, portfolio = validation.frozen_configs(data.freeze_manifest)
    configs = json.loads(json.dumps({name: asdict(cfg) for name, cfg in costs.items()}))
    if (meta["scenarios"] != configs or meta["drawdown_config"] != asdict(risk)
            or meta["portfolio"] != portfolio):
        raise ValueError("Reference parameters differ.")
    if meta["verified_input_source_sha256"] != validation.file_sha256(Path(verified.__file__)):
        raise ValueError("Verified data loader differs from reference.")
    paths = [reference / "metadata.json"] + [
        reference / f"{cost}_{kind}.csv"
        for cost in costs for kind in ("equity", "trades", "risk")
    ]
    return {path.name: validation.file_sha256(path) for path in paths}


def compare_reference(output: Path, name: str, reference: Path) -> None:
    cost = name.removeprefix("chan_on_")
    for kind in ("equity", "trades", "risk"):
        actual = pd.read_csv(output / f"{name}_{kind}.csv")
        expected = pd.read_csv(reference / f"{cost}_{kind}.csv")
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-9)


def summarize(result: dict, extension_start: str) -> list[dict]:
    rows = validation.summarize_result(result, extension_start)
    risk = pd.DataFrame(result["risk_history"]).set_index("date")
    risk.index = pd.to_datetime(risk.index)
    for row in rows:
        sample = risk.loc[row["start"]:row["end"]]
        trades = [t for t in result["trades"] if row["start"] <= t["date"] <= row["end"]]
        halted = sample.loc[sample.halted]
        row.update(
            gross_traded_value=sum(t["qty"] * t["price"] for t in trades),
            exposure_reduced_days=int((sample.multiplier < 1).sum()),
            halted_days=len(halted),
            first_halted_date=str(halted.index[0].date()) if len(halted) else "",
            risk_reduce_fills=sum(t["reason"] == "risk_reduce" for t in trades),
        )
    return rows


def paired_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cost, sample), pair in summary.groupby(["cost", "sample"], sort=False):
        pair = pair.set_index("chan_enabled")
        if len(pair) != 2 or set(pair.index) != {False, True}:
            raise ValueError("Every cost/sample must contain both Chan configurations.")
        on, off = pair.loc[True], pair.loc[False]
        rows.append({"cost": cost, "sample": sample,
                     **{f"off_minus_on_{field}": off[field] - on[field] for field in (
                         "total_return", "max_drawdown", "average_exposure", "trade_count", "commission_total")}})
    return pd.DataFrame(rows)


def write_report(output: Path, summary: pd.DataFrame) -> None:
    lines = ["# 三买规则开关：固定参数组合对照", "",
             "唯一策略配置差异是 `chan_filter_enabled`。关闭时同时移除入选/续持资格和旧三买不加仓限制，因此这是整套三买规则的消融，不是单独入场效果。",
             "每组连续运行一次，再切年度和历史延伸区间；不重置现金、持仓、回撤峰值或调仓相位。", "",
             "| 三买规则 | 成本 | 区间 | 收益 | 区间最大回撤 | 历史峰值回撤 | 平均仓位 | 成交笔数 | 风险降档天数 | 停机天数 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {'开启' if row['chan_enabled'] else '关闭'} | {row['cost']} | {row['sample']} | "
                     f"{row['total_return']:.2%} | {row['max_drawdown']:.2%} | "
                     f"{row['lifetime_peak_drawdown_in_slice']:.2%} | {row['average_exposure']:.2%} | "
                     f"{row['trade_count']} | {row['exposure_reduced_days']} | {row['halted_days']} |")
    lines += ["", "解释边界：", "",
              "- 开启组必须复现已有普通/压力基线的全部净值、成交和风控记录，才接受此次比较。",
              "- 10 个交易日调仓、相位 0、仅调仓日选股；40% 最大目标仓位及 4/6/8/10% 回撤规则保持一致。",
              "- 关闭开关后实际仓位、风险状态、持仓路径会变化；收益差不是同风险暴露下的纯信号收益。",
              "- 压力组最低佣金 5 元、单边滑点 0.4%；普通组最低佣金 0.2 元、单边滑点 0.2%。两组佣金率均为 0.005%。",
              "- 压力成本会改变现金、整数份额和风控路径，差值不等同于对同一交易表简单扣费。",
              "- 2026-07-22 起仅为历史延伸切片，不是未见过的样本外；没有根据结果扫描参数或另选相位。",
              "- 固定 101 只 ETF 中 100 只有通过核验的行情，159925 为缺失列；未重建历史可投资样本池。",
              "- 复权价格上的研究份额尚未还原真实分红现金和份额折算；日线执行未完整模拟涨跌停、容量和跳空取消。",
              "- 回撤 10% 是历史曲线检查，不是未来保证；停止交易后低波动也不能当成稳定盈利证据。",
              "- 本次不更改默认策略、不启动账户、不进行真实下单。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_comparison(snapshot: Path, reference: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Comparison outputs are immutable; choose a new output directory.")
    with verified.open_verified_etf_input(snapshot) as data:
        cases, risk, portfolio = comparison_configs(data.freeze_manifest)
        revisions = validation.source_revision(data.freeze_manifest)
        reference_hashes = verify_reference(data, reference, revisions)
        script_paths = [Path(__file__), Path(validation.__file__), Path(verified.__file__)]
        runner_hashes = {str(path): validation.file_sha256(path) for path in script_paths}
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
            "snapshot": data.metadata, "source_revision": revisions, "runner_sha256": runner_hashes,
            "reference": str(reference.resolve()), "reference_sha256": reference_hashes,
            "scenarios": {name: asdict(cfg) for name, cfg in cases.items()},
            "drawdown_config": asdict(risk), "portfolio": portfolio,
            "historical_extension_start": data.freeze_manifest["historical_extension_start"],
            "only_strategy_parameter_changed": "chan_filter_enabled",
            "scope": "entry/retention eligibility plus active-third-buy no-topup constraint",
            "parameter_search_performed": False, "independent_oos": False,
            "prospective_returns": False, "defaults_changed": False,
            "slice_policy": "Continuous account per case; slices inherit holdings, phase and lifetime risk state.",
            "baseline_reproduction": {},
        }
        output.mkdir(parents=True, exist_ok=False)
        # Written before the first portfolio run; this protocol is never rewritten.
        (output / "protocol.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        try:
            for name, cfg in cases.items():
                print(f"Running fixed comparison: {name}", flush=True)
                result = run_etf_portfolio_backtest(
                    data.store, panel=data.panel, config=cfg, risk_config=risk, **portfolio)
                result["benchmark_equity"] = result["benchmark_equity"] / result["benchmark_equity"].iloc[0] * cfg.capital
                validation.save_result(output, name, result)
                if cfg.chan_filter_enabled:
                    compare_reference(output, name, reference)
                    meta["baseline_reproduction"][name] = "matched_curves_trades_risk"
                rows.extend({"scenario": name, "cost": name.rsplit("_", 1)[-1],
                             "chan_enabled": cfg.chan_filter_enabled, **row}
                            for row in summarize(result, meta["historical_extension_start"]))
            if validation.source_revision(data.freeze_manifest) != revisions:
                raise ValueError("Strategy source changed during comparison.")
            if {str(path): validation.file_sha256(path) for path in script_paths} != runner_hashes:
                raise ValueError("Research or input source changed during comparison.")
            if verify_reference(data, reference, revisions) != reference_hashes:
                raise ValueError("Reference output changed during comparison.")
            summary = pd.DataFrame(rows)
            summary.to_csv(output / "summary.csv", index=False)
            paired_summary(summary).to_csv(output / "paired_comparison.csv", index=False)
            write_report(output, summary)
            meta["status"] = "complete"
            meta["historical_drawdown_target_met"] = {
                row["scenario"]: row["max_drawdown"] >= -.1 for row in rows if row["sample"] == "full"}
            print(summary.loc[summary["sample"].isin(["full", "historical_extension"]),
                              ["scenario", "sample", "total_return", "max_drawdown", "average_exposure",
                               "trade_count", "halted_days"]].to_string(index=False))
        except Exception as exc:
            meta.update(status="failed", failure=str(exc))
            raise
        finally:
            (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_comparison(args.snapshot, args.reference, args.output)


if __name__ == "__main__":
    main()
