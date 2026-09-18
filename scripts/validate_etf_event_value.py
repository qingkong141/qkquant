"""Frozen event study on verified ETF data; never changes the portfolio strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qkquant.config import PROJECT_ROOT
from qkquant.data.verified import open_verified_etf_input
from qkquant.etf_signal import EtfSignalConfig
from qkquant.etf_signal_diagnostics import collect_diagnostics


PROTOCOL = dict(main_horizon=10, secondary_horizon=20, spacing_bars=20,
                budget=100_000 * .4 / 3, stress_min_commission=5, stress_slippage=.004)
GROUP_LABELS = {
    "chan": "全部合格三买", "chan_off_schedule": "非调仓日三买",
    "chan_scheduled": "调仓日三买", "pullback": "普通突破回踩",
    "trend": "MA20 收复", "pullback_without_chan": "回踩成立但三买未通过",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spaced_events(events: pd.DataFrame) -> pd.DataFrame:
    """Use only signal identity, before outcomes, phases or evaluation slices."""
    last, indices = {}, []
    for row in events.sort_values(["position", "code"]).itertuples():
        if row.position - last.get(row.code, -1000) > PROTOCOL["spacing_bars"]:
            indices.append(row.Index)
            last[row.code] = row.position
    return events.loc[indices].copy()


def group_samples(events: pd.DataFrame) -> dict:
    masks = {name: events[name] for name in ("chan", "pullback", "trend")}
    masks["pullback_without_chan"] = events.pullback & ~events.chan
    result = {}
    for group, mask in masks.items():
        raw = events.loc[mask].sort_values(["position", "code"])
        for sample, selected in (("all", raw), ("spaced", spaced_events(raw))):
            result[group, sample] = selected
            if group == "chan":
                # Split the already selected parent; never re-space each phase.
                result["chan_off_schedule", sample] = selected.loc[~selected.scheduled]
                result["chan_scheduled", sample] = selected.loc[selected.scheduled]
    return result


def leave_one_date_bounds(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if len(values) < 2:
        return np.nan, np.nan
    leave_one_out = (values.sum() - values) / (len(values) - 1)
    return float(leave_one_out.min()), float(leave_one_out.max())


def period_slices(events: pd.DataFrame):
    yield "full", events
    yield "development", events.loc[events.date < "2026-07-22"]
    yield "historical_extension", events.loc[events.date >= "2026-07-22"]
    for year in (2024, 2025, 2026):
        yield str(year), events.loc[events.date.str.startswith(str(year))]


def event_statistics(events: pd.DataFrame, cost: str, horizon: int) -> tuple[dict, pd.DataFrame]:
    prefix = f"{cost}_"
    status = events[f"{prefix}status_{horizon}"]
    net, edge = f"{prefix}net_{horizon}", f"{prefix}edge_{horizon}"
    valid = events.loc[status.eq("valid")]
    if valid[net].isna().any() or (~status.eq("valid") & events[net].notna()).any():
        raise ValueError("execution status and outcome disagree")
    daily = valid.groupby("date").agg(
        events=("code", "size"), mean_net=(net, "mean"), mean_edge=(edge, "mean"))
    net_min, net_max = leave_one_date_bounds(daily.mean_net)
    edge_min, edge_max = leave_one_date_bounds(daily.mean_edge)
    values = valid[net]
    return dict(
        events=len(events), valid=len(valid), dates=int(valid.date.nunique()), etfs=int(valid.code.nunique()),
        pending=int(status.eq("pending").sum()), missing=int(status.eq("missing").sum()),
        unexecutable=int(status.eq("unexecutable").sum()),
        mean_gross=valid[f"{prefix}gross_{horizon}"].mean(), mean_net=values.mean(),
        median_net=values.median(), positive_fraction=(values > 0).mean() if len(values) else np.nan,
        worst_net=values.min(), best_net=values.max(),
        date_equal_net=daily.mean_net.mean(), mean_edge=valid[edge].mean(),
        date_equal_edge=daily.mean_edge.mean(), mean_mae=valid[f"{prefix}mae_{horizon}"].mean(),
        worst_mae=valid[f"{prefix}mae_{horizon}"].min(),
        leave_one_date_net_min=net_min, leave_one_date_net_max=net_max,
        leave_one_date_edge_min=edge_min, leave_one_date_edge_max=edge_max,
    ), daily


def summarize_samples(samples: dict):
    summary, dates, paired = [], [], []
    for (group, sample), selected in samples.items():
        for period, piece in period_slices(selected):
            for horizon in (10, 20):
                shared = piece.loc[piece[f"base_status_{horizon}"].eq("valid")
                                   & piece[f"stress_status_{horizon}"].eq("valid")]
                paired.append(dict(group=group, sample=sample, period=period, horizon=horizon,
                    events=len(shared), base_mean_net=shared[f"base_net_{horizon}"].mean(),
                    stress_mean_net=shared[f"stress_net_{horizon}"].mean(),
                    mean_cost_difference=(shared[f"stress_net_{horizon}"] - shared[f"base_net_{horizon}"]).mean()))
                for cost in ("base", "stress"):
                    metrics, daily = event_statistics(piece, cost, horizon)
                    summary.append(dict(group=group, sample=sample, period=period, horizon=horizon, cost=cost, **metrics))
                    if period == "full":
                        dates.append(daily.reset_index().assign(group=group, sample=sample, horizon=horizon, cost=cost))
    return pd.DataFrame(summary), pd.concat(dates, ignore_index=True), pd.DataFrame(paired)


def verify_reference(data, funnel: Path):
    manifest = data.freeze_manifest
    if manifest["config"] != json.loads(json.dumps(asdict(EtfSignalConfig()))):
        raise ValueError("signal parameters differ from the frozen configuration")
    if manifest["event_protocol"] != PROTOCOL:
        raise ValueError("event protocol differs from the predeclared version")
    meta = json.loads((funnel / "metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or meta["snapshot"]["database_sha256"] != data.metadata["database_sha256"]:
        raise ValueError("funnel reference is incomplete or used different market data")
    for name, digest in meta["strategy_source_sha256"].items():
        path = (PROJECT_ROOT / name).resolve()
        if not path.is_relative_to(PROJECT_ROOT.resolve()) or sha256(path) != digest:
            raise ValueError(f"funnel rule source changed: {name}")
    return meta


def reconcile_identity(events: pd.DataFrame, funnel_events: pd.DataFrame):
    if events.duplicated(["date", "code"]).any():
        raise ValueError("duplicate common-pool identities")
    expected = funnel_events.loc[funnel_events.daily_eligible]
    actual = events.loc[events.chan]
    fields = ["date", "code", "position"]
    if set(map(tuple, expected[fields].to_numpy())) != set(map(tuple, actual[fields].to_numpy())):
        raise ValueError("eligible third-buy identities differ from the completed funnel")


def percent(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def report_table(frame: pd.DataFrame) -> list[str]:
    lines = ["| 组别 | 持有期 | 成本 | 有效/全部 | 不同日期 | 平均净收益 | 中位数 | 日期等权净收益 | 同日参考差值（日等权） |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {GROUP_LABELS[row.group]} | {row.horizon} | {row.cost} | {row.valid}/{row.events} | {row.dates} | "
                     f"{percent(row.mean_net)} | {percent(row.median_net)} | {percent(row.date_equal_net)} | {percent(row.date_equal_edge)} |")
    return lines


def write_report(output: Path, summary: pd.DataFrame, events: pd.DataFrame):
    primary = summary.loc[(summary["sample"] == "spaced") & (summary.period == "full")]
    chan = primary.loc[primary.group.isin(["chan", "chan_off_schedule", "chan_scheduled"])]
    off = chan.loc[chan.group == "chan_off_schedule"]
    lines = ["# 核验数据上的固定入场事件验证", "",
        "本轮固定原候选池、三买定义与成本；10 个交易日为主观察期，20 日仅作辅助。没有修改调仓频率、策略参数或账户。",
        "事件收益按每笔独立预算计算，不能累加为组合净值，不能据此推断账户最大回撤。", "",
        f"共同候选共 {len(events)} 个 ETF × 日期；其中合格三买 {int(events.chan.sum())} 个。",
        "先对完整三买组按同 ETF 间隔大于 20 个交易日选首个事件，再拆分调仓日/非调仓日；结果不参与去重。", "",
        "## 三买主表：去重后", "", *report_table(chan), "",
        "## 既定对照：去重后", "",
        *report_table(primary.loc[~primary.group.isin(["chan", "chan_off_schedule", "chan_scheduled"]) & (primary.horizon == 10)]), "",
        "## 非调仓日三买的日期集中度敏感性", "",
        "下表每次删除一个完整信号日，再计算剩余日期等权平均。它是敏感性检查，不是置信区间；少于两天不计算。", "",
        "| 持有期 | 成本 | 净收益留一日期范围 | 同日参考差值留一日期范围 |",
        "|---|---|---:|---:|"]
    for row in off.itertuples():
        lines.append(f"| {row.horizon} | {row.cost} | {percent(row.leave_one_date_net_min)}～{percent(row.leave_one_date_net_max)} | "
                     f"{percent(row.leave_one_date_edge_min)}～{percent(row.leave_one_date_edge_max)} |")
    lines += ["", "口径和限制：", "",
        "- 收盘信号后 t+1 开盘买入，t+11/t+21 开盘卖出；每笔预算 100000×40%÷3，100 份取整。",
        "- base：佣金 0.005%、最低 0.2 元、单边滑点 0.2%；stress：最低佣金 5 元、单边滑点 0.4%。双方都扣买卖成本，剩余预算保留为现金。",
        "- 所有信号保留 valid/pending/missing/unexecutable 状态；未成熟、缺失和无法执行不混入有效收益均值。共同可评估成本配对结果见 paired_costs.csv。",
        "- 同日参考池为当天全部原始合格候选（不限三买），同成本、同持有期的有效收益均值；不随组别去重。参考含事件自身，会使差值向零收缩，也不是可投资基准或因果比较。",
        "- 平均净收益为事件等权；日期等权先平均当日 ETF，再平均日期，降低同一日期多个相关 ETF 的重复权重。不同日期的持有窗口仍可能重叠。",
        "- MAE 是持有路径最低价及退出开盘价相对入场开盘的不利波动，不含成本，不是账户回撤。未完整模拟涨跌停、容量、限价与跳空撤单。",
        "- 复权价格和整数份额是研究口径，未还原真实份额折算、分红现金到账；不能直接用作实际交易数量。",
        "- 已知的每日基线未持仓不代表改变入场频率后仍有现金和持仓名额，本轮没有运行每日入场组合。",
        "- 所有年份与历史延伸期从完整历史去重后的身份切片；没有切换起始日或重新去重。2026-07-22 起仍是历史延伸，不是真实前瞻。",
        "- 年度、完整原始样本、20 日辅助对照和所有失败状态见 summary.csv；不根据表现选择赢家。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(snapshot: Path, funnel: Path, output: Path):
    if output.exists():
        raise ValueError("event-study reports are immutable; select a new output directory")
    with open_verified_etf_input(snapshot) as data:
        reference = verify_reference(data, funnel)
        sources = dict(reference["strategy_source_sha256"])
        sources["src/qkquant/data/verified.py"] = sha256(PROJECT_ROOT / "src/qkquant/data/verified.py")
        sources["scripts/validate_etf_event_value.py"] = sha256(Path(__file__))
        meta = dict(created_at=datetime.now(timezone.utc).isoformat(), status="running",
            protocol=PROTOCOL, strategy_parameters_changed=False, portfolio_backtest_performed=False,
            independent_oos=False, prospective=False, snapshot=data.metadata,
            source_sha256=sources, funnel=str(funnel.resolve()),
            funnel_hashes={name: sha256(funnel / name) for name in ("metadata.json", "fresh_events.csv")},
            group_definitions=GROUP_LABELS, primary_group="chan_off_schedule", primary_sample="spaced",
            selection_policy="Space each parent signal group before phase/year/outcome filters; phase split of chan inherits chan spacing.",
            aggregation_policy="Event means and date-equal means; reference is same-date whole common pool including self.",
            sensitivity="Leave one signal date out; descriptive range, not an interval estimate.")
        output.mkdir(parents=True, exist_ok=False)
        metadata_path = output / "metadata.json"
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            events = collect_diagnostics(data.panel, EtfSignalConfig(), eligible_dates=data.coverage.eligible,
                progress=lambda p, n, count: print(f"Events: {p}/{n} sessions, {count} common-pool observations", flush=True))
            events["scheduled"] = (events.position - 120) % 10 == 0
            reference_events = pd.read_csv(funnel / "fresh_events.csv", dtype={"code": str})
            reconcile_identity(events, reference_events)
            samples = group_samples(events)
            summary, date_aggregates, paired = summarize_samples(samples)
            events.to_csv(output / "all_candidates.csv", index=False)
            pd.concat([frame.assign(group=group, sample=sample) for (group, sample), frame in samples.items()],
                      ignore_index=True).to_csv(output / "group_events.csv", index=False)
            summary.to_csv(output / "summary.csv", index=False)
            date_aggregates.to_csv(output / "date_aggregates.csv", index=False)
            paired.to_csv(output / "paired_costs.csv", index=False)
            for name, digest in sources.items():
                if sha256(PROJECT_ROOT / name) != digest:
                    raise ValueError(f"source changed during the event study: {name}")
            meta.update(status="complete", common_pool_events=len(events),
                        third_buy_events=int(events.chan.sum()),
                        spaced_third_buy_events=len(samples["chan", "spaced"]),
                        spaced_off_schedule_events=len(samples["chan_off_schedule", "spaced"]))
            write_report(output, summary, events)
            print(summary.loc[(summary.group == "chan_off_schedule") & (summary["sample"] == "spaced")
                              & (summary.period == "full"),
                ["horizon", "cost", "events", "valid", "dates", "mean_net", "median_net", "date_equal_net", "date_equal_edge"]].to_string(index=False))
        except Exception as exc:
            meta.update(status="failed", failure=str(exc))
            raise
        finally:
            metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.funnel, args.output)


if __name__ == "__main__":
    main()
