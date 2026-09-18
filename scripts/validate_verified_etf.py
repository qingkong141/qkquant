"""Validate the frozen ETF portfolio with verified data, without parameter search."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qkquant.config import PROJECT_ROOT
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_signal import EtfSignalConfig


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_configs(manifest: dict) -> tuple[dict, DrawdownConfig, dict]:
    """Require the declared defaults; stress costs were fixed before this run."""
    cfg, risk = EtfSignalConfig(), DrawdownConfig()
    if json.loads(json.dumps(asdict(cfg))) != manifest.get("config"):
        raise ValueError("Signal defaults differ from the frozen parameters.")
    if asdict(risk) != manifest.get("drawdown_config"):
        raise ValueError("Drawdown defaults differ from the frozen parameters.")
    portfolio = dict(rebalance_days=10, rebalance_offset=0, daily_entries=False)
    if portfolio != manifest.get("portfolio"):
        raise ValueError("Portfolio protocol differs from the frozen parameters.")
    protocol = manifest.get("event_protocol", {})
    if protocol.get("stress_min_commission") != 5 or protocol.get("stress_slippage") != .004:
        raise ValueError("Stress costs differ from the predeclared protocol.")
    return {"base": cfg, "stress": replace(cfg, commission_min=5, slippage_pct=.004)}, risk, portfolio


def slice_metrics(equity: pd.Series, capital: float, start=None, end=None) -> dict:
    """Retain the return on the first slice day and the inherited account peak."""
    sample = equity.loc[start:end]
    if sample.empty:
        return {}
    previous = equity.loc[equity.index < sample.index[0]]
    opening = float(previous.iloc[-1]) if len(previous) else float(capital)
    values = np.r_[opening, sample.to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1
    local_dd = values / np.maximum.accumulate(values) - 1
    lifetime_values = np.r_[capital, equity.to_numpy(dtype=float)]
    lifetime_dd = pd.Series(
        (lifetime_values / np.maximum.accumulate(lifetime_values) - 1)[1:],
        index=equity.index,
    ).loc[sample.index]
    ratio = float(values[-1] / opening)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return {
        "start": str(sample.index[0].date()), "end": str(sample.index[-1].date()),
        "sessions": len(sample), "opening_equity": opening, "closing_equity": float(values[-1]),
        "total_return": ratio - 1,
        "annualized_return": ratio ** (252 / len(sample)) - 1,
        "max_drawdown": float(local_dd.min()),
        "lifetime_peak_drawdown_in_slice": float(lifetime_dd.min()),
        "annualized_volatility": std * np.sqrt(252),
        "sharpe": float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0,
    }


def summarize_result(result: dict, extension_start: str) -> list[dict]:
    equity = result["equity"]
    capital = result["config"]["capital"]
    extension = pd.Timestamp(extension_start)
    ranges = [("full", None, None),
              ("development", None, extension - pd.Timedelta(days=1)),
              ("historical_extension", extension, None)]
    ranges += [(str(year), f"{year}-01-01", f"{year}-12-31") for year in sorted(set(equity.index.year))]
    rows = []
    for label, start, end in ranges:
        metrics = slice_metrics(equity, capital, start, end)
        if not metrics:
            continue
        dates = equity.loc[start:end].index
        trades = [t for t in result["trades"] if metrics["start"] <= t["date"] <= metrics["end"]]
        benchmark = slice_metrics(result["benchmark_equity"], capital, start, end)
        rows.append({
            "sample": label, **metrics,
            "benchmark_total_return": benchmark["total_return"],
            "benchmark_max_drawdown": benchmark["max_drawdown"],
            "average_exposure": float(result["exposure"].loc[dates].mean()),
            "near_cash_day_fraction": float((result["exposure"].loc[dates] < .05).mean()),
            "trade_count": len(trades),
            "commission_total": float(sum(t["commission"] for t in trades)),
        })
    return rows


def save_result(output: Path, name: str, result: dict) -> None:
    curve = pd.DataFrame({key: result[key] for key in ("equity", "benchmark_equity", "cash", "exposure")})
    curve.index.name = "trade_date"
    curve.to_csv(output / f"{name}_equity.csv")
    pd.DataFrame(result["trades"], columns=["date", "signal_date", "reason", "code", "side", "qty", "price", "commission"]).to_csv(
        output / f"{name}_trades.csv", index=False)
    pd.DataFrame(result["risk_history"]).to_csv(output / f"{name}_risk.csv", index=False)


def source_revision(manifest: dict) -> dict:
    current = {}
    for relative, frozen_hash in manifest["source_sha256"].items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid frozen source: {relative}")
        digest = file_sha256(path)
        current[relative] = {"frozen_sha256": frozen_hash, "current_sha256": digest, "changed": digest != frozen_hash}
    return current


def write_report(output: Path, summary: pd.DataFrame, meta: dict) -> None:
    names = {"full": "完整连续组合", "development": "历史开发区间", "historical_extension": "历史延伸区间"}
    lines = ["# 固定参数 ETF 组合验证", "",
        "本报告使用同一条连续账户曲线切片，不在年初或历史延伸开始时重新建仓、重置回撤或调仓相位。",
        "参数沿用冻结清单；压力情景只采用预先约定的最低佣金 5 元和单边滑点 0.4%。", "",
        "| 成本 | 区间 | 日期 | 收益 | 区间最大回撤 | 历史峰值回撤 | 平均仓位 | 成交次数 |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['scenario']} | {names.get(row['sample'], row['sample'])} | {row['start']}～{row['end']} | "
            f"{row['total_return']:.2%} | {row['max_drawdown']:.2%} | {row['lifetime_peak_drawdown_in_slice']:.2%} | "
            f"{row['average_exposure']:.2%} | {row['trade_count']} |")
    lines += ["", "研究边界：", "",
        "- 复权价格上的整数份额/现金是研究账本；尚未逐笔还原原始价格、现金分红、拆分份额和实际到账日期，不能当作真实可交易账务。",
        "- 日线近似未完整模拟涨跌停、容量、限价与跳空取消；报告中的仓位上限是目标预算，价格波动可能使实际仓位略超预算。",
        "- 历史延伸区间从 2026-07-22 开始，仍是冻结前历史；不是独立样本外，也不是真实前瞻。",
        "- 固定 101 只 ETF 的 100 只有核验通过的行情；159925 保留为缺失列。未重建当时可知的上市/退市样本。",
        "- 完整交易日历保留缺失日期；缺失价格只在持仓估值时沿用上一已知价，不虚构成交。",
        "- 等权 ETF 参考组合在策略首个记录日收盘归一至初始资金；未施加策略的仓位、成本和可交易限制，不能直接视为可投资比较基准。",
        "- 最大回撤 10% 只检查本次历史研究曲线；即使通过，也不保证未来。短切片年化值仅存于 CSV，不用于选择策略。", "",
        f"本次运行：`{meta['created_at']}`。源码相对旧冻结清单的差异保存在 `metadata.json`，没有覆盖旧冻结记录。",
        f"实现修订说明：{meta['implementation_note']}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-note", required=True,
                        help="Explain the reviewed implementation revision relative to the old freeze; no parameter selection.")
    args = parser.parse_args()
    run_validation(args.snapshot, args.output, args.implementation_note)


def run_validation(snapshot_dir: Path, output: Path, implementation_note: str) -> dict:
    from qkquant.data import verified
    from qkquant.etf_portfolio_backtest import run_etf_portfolio_backtest

    if not implementation_note.strip():
        raise ValueError("An implementation revision note is required.")
    if output.exists():
        raise ValueError("Validation outputs are immutable; choose a new output directory.")
    with verified.open_verified_etf_input(snapshot_dir) as data:
        scenarios, risk, portfolio = frozen_configs(data.freeze_manifest)
        revisions = source_revision(data.freeze_manifest)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
            "implementation_note": implementation_note, "frozen_parameters_unchanged": True,
            "parameter_search_performed": False, "independent_oos": False,
            "prospective_returns": False, "execution_accounting": "qfq_research_units_not_native_cash_dividends",
            "snapshot": data.metadata, "source_revision": revisions,
            "runner_sha256": file_sha256(Path(__file__)),
            "verified_input_source_sha256": file_sha256(Path(verified.__file__)),
            "scenarios": {name: asdict(cfg) for name, cfg in scenarios.items()},
            "drawdown_config": asdict(risk), "portfolio": portfolio,
            "historical_extension_start": data.freeze_manifest["historical_extension_start"],
            "slice_policy": "One continuous portfolio per cost scenario; no state, warmup or phase reset at slice boundaries.",
        }
        output.mkdir(parents=True, exist_ok=False)
        metadata_path = output / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        data.coverage.to_csv(output / "date_coverage.csv")
        rows = []
        try:
            for name, cfg in scenarios.items():
                print(f"Running fixed portfolio: {name}", flush=True)
                result = run_etf_portfolio_backtest(
                    data.store, panel=data.panel, config=cfg, risk_config=risk, **portfolio)
                # Both series start at the first recorded close, before any strategy fill.
                result["benchmark_equity"] = result["benchmark_equity"] / result["benchmark_equity"].iloc[0] * cfg.capital
                save_result(output, name, result)
                rows.extend({"scenario": name, **row} for row in summarize_result(
                    result, data.freeze_manifest["historical_extension_start"]))
            if source_revision(data.freeze_manifest) != revisions:
                raise ValueError("Source files changed while the validation was running.")
            summary = pd.DataFrame(rows)
            summary.to_csv(output / "summary.csv", index=False)
            metadata["status"] = "complete"
            metadata["historical_drawdown_target_met"] = {
                row["scenario"]: row["max_drawdown"] >= -.1
                for row in rows if row["sample"] == "full"
            }
            write_report(output, summary, metadata)
            print(summary.loc[summary["sample"].isin(["full", "historical_extension"]),
                  ["scenario", "sample", "total_return", "max_drawdown", "trade_count"]].to_string(index=False))
        except Exception as exc:
            metadata.update(status="failed", failure=str(exc))
            raise
        finally:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    main()
