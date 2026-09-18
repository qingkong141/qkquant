"""Reproducible, fixed-candidate ETF ablation study; does not change live settings."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.etf_portfolio_backtest import _metrics, run_etf_portfolio_backtest
from qkquant.etf_signal import EtfSignalConfig


def main():
    output = Path("reports/strategy_research_20260918")
    output.mkdir(parents=True, exist_ok=True)
    base = EtfSignalConfig()
    # Small explicit hypothesis set for retrospective research; no grid search.
    candidates = [
        ("baseline_third_buy_5d", 5, base),
        ("third_buy_10d", 10, base),
        ("third_buy_20d", 20, base),
        ("all_buy_states_10d", 10, replace(base, chan_entry_states=("third_buy", "second_buy", "first_buy_divergence"))),
        ("third_buy_20d_stress", 20, replace(base, slippage_pct=0.004, commission_min=5.0)),
        ("no_chan_5d", 5, replace(base, chan_filter_enabled=False)),
        ("no_chan_10d", 10, replace(base, chan_filter_enabled=False)),
        ("no_chan_20d", 20, replace(base, chan_filter_enabled=False)),
        ("no_breadth_5d", 5, replace(base, risk_on_breadth=0.0)),
        ("no_chan_no_breadth_10d", 10, replace(base, chan_filter_enabled=False, risk_on_breadth=0.0)),
    ]
    rows, details = [], {}
    with DuckStore(read_only=True) as store:
        coverage = store.conn.execute("""
            SELECT i.etf_category, min(d.trade_date) AS first_date,
                max(d.trade_date) AS last_date, count(DISTINCT d.code) AS codes,
                count(*) AS bars
            FROM daily_bars d JOIN instruments i USING(code)
            WHERE i.instrument_type='etf' AND d.adjust='qfq'
            GROUP BY 1
        """).df()
        coverage.to_csv(output / "coverage.csv", index=False)
        print(coverage.to_string(index=False), flush=True)
        dates = store.conn.execute("""
            SELECT d.trade_date, count(*) AS codes
            FROM daily_bars d JOIN instruments i USING(code)
            WHERE i.etf_category='equity' AND d.adjust='qfq'
            GROUP BY 1 ORDER BY 1
        """).df()
        dates.to_csv(output / "daily_coverage.csv", index=False)
        # Avoid an incomplete trailing cross-section; report the full coverage too.
        end = str(dates.loc[dates.codes >= dates.codes.max() * .9, "trade_date"].iloc[-1].date())
        for name, days, cfg in candidates:
            print(f"Running {name} through {end}", flush=True)
            result = run_etf_portfolio_backtest(store, start="2024-01-01", end=end, rebalance_days=days, config=cfg)
            pd.DataFrame({"equity": result["equity"], "benchmark": result["benchmark_equity"],
                          "exposure": result["exposure"]}).to_csv(output / f"{name}_equity.csv")
            pd.DataFrame(result["trades"]).to_csv(output / f"{name}_trades.csv", index=False)
            # Chronological slices are retrospective diagnostics, NOT untouched OOS.
            for period, lower, upper in [("full", "2022", "2027"),
                                         ("early", "2022", "2025"),
                                         ("late", "2025", "2027")]:
                equity = result["equity"]
                mask = (equity.index >= lower) & (equity.index < upper)
                positions = [i for i, valid in enumerate(mask) if valid]
                if len(positions) < 2:
                    continue
                sample = equity.iloc[max(0, positions[0] - 1):positions[-1] + 1]
                rows.append({"candidate": name, "period": period, **_metrics(sample),
                             "average_exposure": float(result["exposure"].loc[mask].mean())})
            details[name] = {"config": asdict(cfg), "rebalance_days": days,
                             "start": result["start"], "end": result["end"],
                             "trade_count": result["trade_count"],
                             "commission_total": result["commission_total"],
                             "turnover_ratio": result["turnover_ratio"],
                             "attribution": result["attribution"]}
            pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
            (output / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
            print(name, result["strategy"], flush=True)

        # Check every scheduling phase: an attractive result may depend on the
        # arbitrary first rebalance date rather than on the holding interval.
        phase_rows = []
        for days in (5, 10, 20):
            for offset in range(days):
                result = run_etf_portfolio_backtest(
                    store, start="2024-01-01", end=end, rebalance_days=days,
                    config=base, rebalance_offset=offset,
                )
                phase_rows.append({"rebalance_days": days, "offset": offset,
                                   **result["strategy"]})
            pd.DataFrame(phase_rows).to_csv(output / "schedule_phases.csv", index=False)
            print(f"Completed all {days} scheduling phases", flush=True)


if __name__ == "__main__":
    main()
