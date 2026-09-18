"""Explain the existing ETF portfolio's sparse entries without changing any rule."""

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
from qkquant.etf_chan import classify_chan_structure
from qkquant.etf_drawdown import DrawdownConfig, DrawdownState
from qkquant.etf_portfolio_backtest import close_decision, prepare_inputs
from qkquant.etf_signal import EtfSignalConfig, theme_key


STAGES = [
    ("fresh", "三买刚确认"),
    ("complete", "因子数据完整"),
    ("mom60_ok", "60 日动量为正"),
    ("mom120_ok", "120 日动量为正"),
    ("above_ma120", "价格高于 MA120"),
    ("amount_ok", "20 日平均成交额至少 5000 万"),
    ("score_ok", "原候选池内综合得分至少 0.65"),
    ("market_ok", "市场宽度允许持仓"),
    ("scheduled", "恰逢固定 10 日调仓日"),
    ("risk_buy_ok", "账户允许买入且无优先减仓"),
    ("unheld", "当日尚未持有该标的"),
    ("selected", "通过主题、相关性和持仓上限筛选"),
    ("opened", "下一交易日实际建立新仓"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factor_gate(prepared: dict, pos: int, cfg: EtfSignalConfig) -> pd.DataFrame:
    """Rank in the engine's prefiltered universe, never within fresh signals."""
    fields = ("mom60", "mom120", "risk_mom", "trend", "downside", "amount20")
    frame = pd.DataFrame({key: prepared[key].iloc[pos] for key in fields})
    frame["complete"] = frame[list(fields)].notna().all(axis=1)
    frame["mom60_ok"] = frame.mom60 > 0
    frame["mom120_ok"] = frame.mom120 > 0
    frame["above_ma120"] = prepared["close"].iloc[pos] > prepared["ma120"].iloc[pos]
    frame["amount_ok"] = frame.amount20 >= cfg.min_amount_20d
    frame["prefilter"] = frame[["complete", "mom60_ok", "mom120_ok", "above_ma120", "amount_ok"]].all(axis=1)
    pool = frame.loc[frame.prefilter]
    score = (.35 * pool.risk_mom.rank(pct=True) + .25 * pool.mom120.rank(pct=True)
             + .20 * pool.trend.rank(pct=True) + .20 * (1 - pool.downside.rank(pct=True)))
    frame["score"] = score.reindex(frame.index)
    frame["score_ok"] = frame.score >= cfg.score_threshold
    return frame


def summarize_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """Cumulative, order-dependent attrition; not independent causal effects."""
    mask = pd.Series(True, index=events.index)
    rows = []
    previous = len(events)
    for key, label in STAGES:
        mask &= events[key]
        n = int(mask.sum())
        rows.append(dict(stage=key, label=label, events=n,
                         dates=int(events.loc[mask, "date"].nunique()),
                         codes=int(events.loc[mask, "code"].nunique()),
                         removed_at_step=previous - n))
        previous = n
    return pd.DataFrame(rows)


def load_reference(directory: Path, data, cfg: EtfSignalConfig):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("reference portfolio validation is not complete")
    if metadata["snapshot"]["database_sha256"] != data.metadata["database_sha256"]:
        raise ValueError("reference portfolio used different market data")
    if metadata["scenarios"]["base"] != json.loads(json.dumps(asdict(cfg))):
        raise ValueError("reference portfolio signal parameters differ")
    if metadata["drawdown_config"] != asdict(DrawdownConfig()):
        raise ValueError("reference portfolio risk parameters differ")
    if metadata["portfolio"] != dict(rebalance_days=10, rebalance_offset=0, daily_entries=False):
        raise ValueError("reference portfolio schedule differs")
    for relative, revision in metadata["source_revision"].items():
        if sha256(PROJECT_ROOT / relative) != revision["current_sha256"]:
            raise ValueError(f"reference source changed: {relative}")
    if sha256(PROJECT_ROOT / "src/qkquant/data/verified.py") != metadata["verified_input_source_sha256"]:
        raise ValueError("verified input code differs from the reference run")
    curves = pd.read_csv(directory / "base_equity.csv", index_col="trade_date", parse_dates=True)
    expected = data.panel["close"].index[120:]
    if not curves.index.equals(expected):
        raise ValueError("reference curve is not on the same complete trading calendar")
    trades = pd.read_csv(directory / "base_trades.csv", dtype={"code": str})
    return metadata, curves, trades


def collect_funnel(data, curves: pd.DataFrame, trades: pd.DataFrame, progress=None):
    cfg, risk = EtfSignalConfig(), DrawdownConfig()
    prepared = prepare_inputs(data.panel)
    close, high, low = (prepared[key] for key in ("close", "high", "low"))
    instruments = data.store.load_instruments(data.codes).set_index("code")
    state = DrawdownState(cfg.capital, cfg.capital)
    holdings, rows, day_rows, states = {}, [], [], []
    grouped_trades = {day: group.to_dict("records") for day, group in trades.groupby("date", sort=False)}
    observed_openings = set()
    for pos in range(120, len(close)):
        day = close.index[pos]
        day_text = str(day.date())
        for trade in grouped_trades.get(day_text, []):
            code, quantity = trade["code"], int(trade["qty"])
            if trade["side"] == "BUY" and not holdings.get(code, 0):
                observed_openings.add((trade["signal_date"], code))
            holdings[code] = holdings.get(code, 0) + (quantity if trade["side"] == "BUY" else -quantity)
            if holdings[code] < 0:
                raise ValueError("reference trade history creates a short position")
            if not holdings[code]:
                holdings.pop(code)
        account = curves.loc[day]
        invested = sum(qty * float(prepared["valuation"].loc[day, code]) for code, qty in holdings.items())
        if not np.isclose(account.cash + invested, account.equity, rtol=1e-12, atol=1e-7):
            raise ValueError(f"holdings cannot reconcile reference equity on {day_text}")
        decision = close_decision(prepared, instruments, pos, holdings, account.equity,
                                  invested, state, cfg, risk, rebalance_days=10)
        frame = factor_gate(prepared, pos, cfg)
        selected = decision["pending"][0] if decision["pending"] else []
        score_codes = frame.loc[frame.score_ok].sort_values("score", ascending=False).index.tolist()
        fresh_count = eligible_count = 0
        for code in data.codes:
            if code in data.metadata["excluded_codes"]:
                continue
            signal = classify_chan_structure(close[code].iloc[:pos + 1], high[code].iloc[:pos + 1],
                                             low[code].iloc[:pos + 1], cfg.chan_fractal_order, cfg.chan_tolerance)
            states.append(dict(date=day_text, code=code, state=signal.state))
            if signal.state != "third_buy":
                continue
            fresh_count += 1
            row = dict(date=day_text, code=code, position=pos, phase=(pos - 120) % 10,
                       name=str(instruments.loc[code, "name"]), fresh=True, **frame.loc[code].to_dict())
            row.update(market_ok=decision["breadth"] >= cfg.risk_off_breadth,
                       breadth=decision["breadth"], scheduled=decision["scheduled"],
                       risk_buy_ok=decision["action"] != "risk_reduce" and state.multiplier > 0,
                       unheld=holdings.get(code, 0) == 0,
                       selected=code in selected and decision["action"] == "rebalance",
                       held_qty=holdings.get(code, 0), decision_action=decision["action"],
                       exposure_cap=decision["exposure_cap"],
                       previous_high_date=signal.previous_high_date, breakout_high_date=signal.breakout_high_date,
                       pullback_date=signal.pullback_date, confirmation_date=signal.confirmation_date,
                       signal_age_bars=signal.signal_age_bars, distance_to_pullback=signal.distance_to_pullback,
                       missing_prices_in_last_200_calendar_days=int(close[code].iloc[max(0, pos - 199):pos + 1].isna().sum()))
            row["daily_eligible"] = bool(row["prefilter"] and row["score_ok"] and row["market_ok"])
            eligible_count += row["daily_eligible"]
            row["selection_rejection"] = ""
            if (row["daily_eligible"] and row["scheduled"] and row["risk_buy_ok"]
                    and row["unheld"] and not row["selected"]):
                ahead = [other for other in selected if score_codes.index(other) < score_codes.index(code)]
                theme = theme_key(str(instruments.loc[code, "name"]), instruments.loc[code].get("benchmark_code"))
                if len(ahead) >= cfg.max_positions:
                    row["selection_rejection"] = "position_limit"
                elif any(theme == theme_key(str(instruments.loc[other, "name"]), instruments.loc[other].get("benchmark_code")) for other in ahead):
                    row["selection_rejection"] = "duplicate_theme"
                elif ahead and prepared["ret"].iloc[max(0, pos - cfg.corr_window + 1):pos + 1][ahead + [code]].corr()[code].drop(code).max() > cfg.corr_limit:
                    row["selection_rejection"] = "correlation"
                else:
                    raise ValueError(f"selection does not reconcile with original rule: {day_text} {code}")
            if pos + 1 == len(close):
                row["next_open_status"] = "not_observed"
            else:
                price, amount = data.panel["open"].iloc[pos + 1][code], data.panel["amount"].iloc[pos + 1][code]
                row["next_open_status"] = "tradable" if np.isfinite(price) and price > 0 and np.isfinite(amount) and amount > 0 else "missing_or_untradable"
            rows.append(row)
        day_rows.append(dict(date=day_text, position=pos, phase=(pos - 120) % 10,
                             scheduled=decision["scheduled"], breadth=decision["breadth"],
                             action=decision["action"], risk_multiplier=state.multiplier,
                             fresh_signals=fresh_count, daily_eligible_fresh=eligible_count,
                             prefiltered_count=int(frame.prefilter.sum()), score_pass_count=int(frame.score_ok.sum()),
                             selected="|".join(selected), equity=account.equity, cash=account.cash,
                             exposure=invested / account.equity))
        if progress and (pos - 120) % 50 == 0:
            progress(pos - 120, len(close) - 120, len(rows))
    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError("no fresh third-buy signals found; inspect classification_states.csv")
    events["opened"] = [(row.date, row.code) in observed_openings for row in events.itertuples()]
    if set(zip(events.loc[events.opened, "date"], events.loc[events.opened, "code"])) != observed_openings:
        raise ValueError("some actual new positions were not explained by a fresh signal")
    mask = events[[key for key, _ in STAGES[:-1]]].all(axis=1)
    if (events.opened & ~mask).any():
        raise ValueError("an actual opening fails the diagnostic funnel")
    events["first_blocker"] = "opened"
    for index, row in events.iterrows():
        for key, _ in STAGES:
            if not row[key]:
                events.loc[index, "first_blocker"] = key
                break
    return events, pd.DataFrame(day_rows), pd.DataFrame(states)


def write_report(output: Path, events: pd.DataFrame, days: pd.DataFrame, funnel: pd.DataFrame):
    qualified = events[events.daily_eligible]
    offset = qualified[~qualified.scheduled]
    extensions = events[events.date >= "2026-07-22"]
    lines = ["# 三买为什么只有三次开仓", "",
             "本轮只解释已冻结策略的实际交易路径，不改阈值、不切换调仓偏移、不计算替代策略收益。",
             f"区间 {days.date.min()}～{days.date.max()}；{len(days)} 个收盘日，{int(days.scheduled.sum())} 个固定调仓日。",
             "计数单位为 ETF × 信号日；同一 ETF 不同确认日分别计数，不代表独立统计样本。", "",
             "| 累积条件 | 剩余事件 | 独立日期 | ETF 数 | 本层减少 |",
             "|---|---:|---:|---:|---:|"]
    for row in funnel.itertuples():
        lines.append(f"| {row.label} | {row.events} | {row.dates} | {row.codes} | {row.removed_at_step} |")
    lines += ["", f"符合因子、评分及市场环境的三买共有 {len(qualified)} 个，其中 {len(offset)} 个出现在非调仓日。",
              "fresh 三买只在回踩分型刚确认日允许建仓。后续 third_buy_active 仅能延续已有持仓，不能等到下个调仓日补买。",
              f"历史延伸期共有 {len(extensions)} 个原始新三买，{int(extensions.daily_eligible.sum())} 个通过日常条件，{int(extensions.opened.sum())} 个实际开仓。", "",
              "各层减少量依赖表中展示顺序；实际代码先检查风险/调仓日，再筛因子和评分。条件会重叠，不能把每层减少当作互相独立的因果贡献。",
              "横截面评分严格在原策略的完整因子、正动量、均线、流动性候选池内排名，没有在三买子集重排。", "",
              "边界与下一步：", "",
              "- 非调仓日候选数只说明信号和日历存在错位，不说明这些信号买入后会赚钱。未按结果选择新偏移或新阈值。",
              "- 不通过成交额门槛可能是可交易性保护；不通过评分可能是缺乏趋势强度。不能为了增加笔数直接删除过滤条件。",
              "- 当前检测是简化的局部高低点与 MA20 结构，不是完整缠论中枢定义；局部三买不表示处于全局低位。",
              "- Chan 分类内部对历史缺价 dropna，以有效 K 线确认分型。fresh_events.csv 单列最近 200 日缺价数量；完整日历、调仓相位没有压缩。",
              "- 原始分类包含彼此重叠的同主题 ETF 和重复时期；本轮没有计算新信号的未来收益，不构成盈利验证或真实前瞻。",
              "- 决策账户沿用上次已保存的连续组合，每日现金、持仓市值逐日对账；最后一日之后的开盘仍记为未观察。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("diagnostic outputs are immutable; choose a new directory")
    with open_verified_etf_input(args.snapshot) as data:
        reference, curves, trades = load_reference(args.reference, data, EtfSignalConfig())
        sources = {name: sha256(PROJECT_ROOT / name) for name in reference["source_revision"]}
        meta = dict(created_at=datetime.now(timezone.utc).isoformat(), status="running",
                    strategy_parameters_unchanged=True, strategy_sources_unchanged=True,
                    alternative_strategy_returns_computed=False, snapshot=data.metadata,
                    reference=str(args.reference.resolve()),
                    reference_hashes={name: sha256(args.reference / name) for name in ("metadata.json", "base_trades.csv", "base_equity.csv")},
                    strategy_source_sha256=sources, diagnostic_source_sha256=sha256(Path(__file__)),
                    counting_unit="ETF x signal date, no future outcome filtering or event deduplication")
        args.output.mkdir(parents=True, exist_ok=False)
        metadata_path = args.output / "metadata.json"
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            events, days, states = collect_funnel(data, curves, trades,
                progress=lambda position, total, count: print(f"Classified {position}/{total} dates; fresh signals={count}", flush=True))
            funnel = summarize_funnel(events)
            events.to_csv(args.output / "fresh_events.csv", index=False)
            days.to_csv(args.output / "daily_funnel.csv", index=False)
            states.to_csv(args.output / "classification_states.csv", index=False)
            states.groupby("state").size().rename("observations").to_csv(args.output / "state_counts.csv")
            funnel.to_csv(args.output / "funnel.csv", index=False)
            events[events.daily_eligible].groupby("phase").agg(events=("code", "size"), dates=("date", "nunique")).reindex(range(10), fill_value=0).to_csv(args.output / "phase_counts.csv")
            slices = []
            for label, subset in [("full", events), ("historical_extension", events[events.date >= "2026-07-22"]),
                                  *[(str(year), events[events.date.str.startswith(str(year))]) for year in (2024, 2025, 2026)]]:
                slices.append(summarize_funnel(subset).assign(sample=label))
            pd.concat(slices).to_csv(args.output / "funnel_slices.csv", index=False)
            if sources != {name: sha256(PROJECT_ROOT / name) for name in sources}:
                raise ValueError("strategy sources changed during diagnosis")
            meta.update(status="complete", fresh_events=len(events), actual_new_positions=int(events.opened.sum()),
                        daily_eligible_events=int(events.daily_eligible.sum()),
                        off_schedule_eligible_events=int((events.daily_eligible & ~events.scheduled).sum()))
            write_report(args.output, events, days, funnel)
            print(funnel[["stage", "events", "dates", "removed_at_step"]].to_string(index=False), flush=True)
        except Exception as exc:
            meta.update(status="failed", failure=str(exc))
            raise
        finally:
            metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
