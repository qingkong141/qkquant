"""ma_boll structural experiment.

This script does not change the production MaBollStrategy. It defines an
experimental variant to test:
- cross_lookback: allow entries shortly after a fresh MA cross
- max_positions: tune position count for a 100k account
- trend_filter: require close > MA60 to avoid weak trends
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qkquant.backtest.engine import BacktestEngine, BtStrategyBase
from qkquant.backtest.report import calc_metrics
from qkquant.data.storage import DuckStore
from qkquant.strategy.registry import get_strategy, load_risk_config, load_strategy_config


class MaBollExperimentStrategy(BtStrategyBase):
    params = (
        ("fast", 5),
        ("slow", 20),
        ("boll_period", 20),
        ("boll_dev", 2.0),
        ("upper_buffer", 0.03),
        ("mid_break_pct", 0.05),
        ("max_positions", 10),
        ("min_price", 1.0),
        ("cross_lookback", 1),
        ("trend_filter", False),
        ("trend_period", 60),
    )

    def __init__(self) -> None:
        super().__init__()
        self.inds: dict[str, dict] = {}
        for data in self.datas:
            fast = bt.indicators.SMA(data.close, period=self.p.fast)
            slow = bt.indicators.SMA(data.close, period=self.p.slow)
            boll = bt.indicators.BollingerBands(
                data.close, period=self.p.boll_period, devfactor=self.p.boll_dev
            )
            self.inds[data._name] = {
                "fast": fast,
                "slow": slow,
                "cross": bt.indicators.CrossOver(fast, slow),
                "boll": boll,
                "trend": bt.indicators.SMA(data.close, period=self.p.trend_period),
            }
        self._trade_log: list[dict] = []

    def _current_positions(self) -> list[str]:
        return [d._name for d in self.datas if self.getposition(d).size > 0]

    def _target_position_value(self) -> float:
        return self.broker.getvalue() / self.p.max_positions

    def _recent_cross_up(self, cross) -> bool:
        lookback = max(1, int(self.p.cross_lookback))
        available = min(lookback, len(cross))
        return any(float(cross[-i]) > 0 for i in range(available))

    def next(self) -> None:
        self.apply_forced_exits()

        held = set(self._current_positions())
        today = self._today()

        for data in self.datas:
            code = data._name
            if code not in held:
                continue
            ind = self.inds[code]
            close = float(data.close[0])
            cross = float(ind["cross"][0]) if len(ind["cross"]) > 0 else 0.0
            boll_mid = float(ind["boll"].mid[0]) if len(ind["boll"].mid) > 0 else 0.0

            sell_reason = None
            if cross < 0:
                sell_reason = "ma_cross_down"
            elif boll_mid > 0 and close < boll_mid * (1 - self.p.mid_break_pct):
                sell_reason = "below_boll_mid"

            if sell_reason:
                pos = self.getposition(data)
                order = self.safe_sell(data, pos.size, reason=sell_reason)
                if order is not None:
                    self._trade_log.append(
                        {
                            "date": today,
                            "code": code,
                            "side": "SELL",
                            "price": close,
                            "qty": pos.size,
                            "reason": sell_reason,
                        }
                    )

        held = set(self._current_positions())
        slots = self.p.max_positions - len(held)
        if slots <= 0:
            return

        candidates: list[tuple[str, float]] = []
        for data in self.datas:
            code = data._name
            if code in held:
                continue
            ind = self.inds[code]
            if len(ind["cross"]) == 0 or len(ind["boll"].mid) == 0:
                continue

            close = float(data.close[0])
            fast = float(ind["fast"][0])
            slow = float(ind["slow"][0])
            if close < self.p.min_price or fast <= slow:
                continue
            if not self._recent_cross_up(ind["cross"]):
                continue

            mid = float(ind["boll"].mid[0])
            upper = float(ind["boll"].top[0])
            if mid <= 0 or upper <= 0 or close <= mid:
                continue
            if close > upper * (1 - self.p.upper_buffer):
                self._rejections.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "reason": "near_upper_band",
                    }
                )
                continue
            if self.p.trend_filter:
                trend = float(ind["trend"][0]) if len(ind["trend"]) > 0 else 0.0
                if trend <= 0 or close <= trend:
                    continue

            band_pos = (close - mid) / max(upper - mid, 1e-6)
            trend_strength = fast / max(slow, 1e-6) - 1.0
            # Prefer confirmed trends that are not already pinned near the upper band.
            score = trend_strength - max(band_pos - 0.75, 0.0) * 0.05
            candidates.append((code, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        target_value = self._target_position_value()
        data_by_code = {d._name: d for d in self.datas}
        for code, _ in candidates[:slots]:
            data = data_by_code[code]
            close = float(data.close[0])
            if close <= 0:
                continue
            qty = int((target_value / close) // 100) * 100
            if qty <= 0:
                continue
            order = self.safe_buy(data, qty, reason="ma_boll_experiment_entry")
            if order is not None:
                self._trade_log.append(
                    {
                        "date": today,
                        "code": code,
                        "side": "BUY",
                        "price": close,
                        "qty": qty,
                        "reason": "ma_boll_experiment_entry",
                    }
                )


def run_one(
    store: DuckStore,
    codes: list[str],
    start: str,
    end: str,
    params: dict,
    capital: float = 30_000.0,
) -> dict:
    info = get_strategy("ma_boll")
    cfg = load_strategy_config(info)
    risk_cfg = load_risk_config(cfg)
    engine = BacktestEngine(
        store=store,
        strategy_cls=MaBollExperimentStrategy,
        strategy_params=params,
        initial_capital=capital,
        risk_config=risk_cfg,
    )
    result = engine.run(codes=codes, start=start, end=end)
    metrics = calc_metrics(result)
    return {
        "final": result["final_value"],
        "return_pct": metrics["total_return"] * 100,
        "annual_pct": metrics["annualized_return"] * 100,
        "mdd_pct": metrics["max_drawdown"] * 100,
        "sharpe": metrics["sharpe_ratio"] or 0.0,
        "trades": metrics["total_trades"],
        "winrate_pct": metrics["win_rate"] * 100,
        "profit_factor": metrics["profit_factor"] or 0.0,
    }


def main() -> None:
    start = "2025-05-12"
    end = "2026-05-12"
    capital = 30_000.0

    store = DuckStore()
    codes = store.load_index_constituents("000300")
    inst = store.load_instruments(codes)
    if not inst.empty and "is_st" in inst.columns:
        st_codes = set(inst[inst["is_st"]]["code"].tolist())
        codes = [c for c in codes if c not in st_codes]

    cfg = load_strategy_config(get_strategy("ma_boll"))
    base_params = dict(cfg.get("params") or {})

    experiments: list[dict] = []
    for cross_lookback in [1, 2, 3, 5]:
        for max_positions in [5, 8, 10]:
            for trend_filter in [False, True]:
                params = {
                    **base_params,
                    "cross_lookback": cross_lookback,
                    "max_positions": max_positions,
                    "trend_filter": trend_filter,
                    "trend_period": 60,
                }
                label = (
                    f"cross={cross_lookback}, "
                    f"pos={max_positions}, "
                    f"trend={'on' if trend_filter else 'off'}"
                )
                print(f"running {label} ...", flush=True)
                row = {
                    "label": label,
                    "cross_lookback": cross_lookback,
                    "max_positions": max_positions,
                    "trend_filter": trend_filter,
                    **run_one(store, codes, start, end, params, capital),
                }
                experiments.append(row)

    store.close()

    df = pd.DataFrame(experiments).sort_values(
        ["sharpe", "return_pct"], ascending=[False, False]
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports") / f"ma_boll_experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "# ma_boll structural experiment",
        "",
        f"- Period: {start} ~ {end}",
        f"- Universe: HS300 ({len(codes)} codes, ST excluded)",
        f"- Capital: {capital:.0f}",
        "",
        "| rank | label | return | annual | MDD | Sharpe | trades | winrate | PF |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(df.head(12).to_dict("records"), 1):
        md_lines.append(
            f"| {rank} | {row['label']} | {row['return_pct']:+.2f}% | "
            f"{row['annual_pct']:+.2f}% | {row['mdd_pct']:.2f}% | "
            f"{row['sharpe']:.2f} | {int(row['trades'])} | "
            f"{row['winrate_pct']:.2f}% | {row['profit_factor']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("\nTop 12 by Sharpe:")
    print(
        df.head(12)[
            [
                "label",
                "return_pct",
                "annual_pct",
                "mdd_pct",
                "sharpe",
                "trades",
                "winrate_pct",
                "profit_factor",
            ]
        ].to_string(index=False)
    )
    print(f"\nreport: {out_dir}")


if __name__ == "__main__":
    main()
