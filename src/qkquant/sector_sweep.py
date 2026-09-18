"""Experimental intraday sector-strength sweep, using completed minute bars only.

This is a research approximation of an observed strategy, not a recovered rule or
an order-execution strategy. CSV columns: time, code, name, sector, close, prev_close.
Supply the full eligible universe at every minute; multiple sector rows per code
are allowed. Signals generated from a completed minute are actionable no earlier
than the following minute.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SweepConfig:
    min_sector_members: int = 5
    min_sector_breadth: float = 0.60
    min_sector_leaders: int = 2
    leader_gain: float = 0.07
    min_sector_near_limit: int = 1
    near_limit_gain: float = 0.095
    min_stock_gain: float = 0.08
    min_price: float = 3.0
    max_price: float = 12.0


REQUIRED = {"time", "code", "name", "sector", "close", "prev_close"}


def scan_minute_bars(bars: pd.DataFrame, config: SweepConfig | None = None) -> pd.DataFrame:
    """Return each code's first qualifying signal and its qualifying sector.

    The data must contain every eligible stock at every completed minute. Missing
    members can inflate sector breadth; this function cannot infer omitted rows.
    """
    config = config or SweepConfig()
    missing = REQUIRED - set(bars.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    data = bars.copy()
    data["time"] = pd.to_datetime(data["time"], errors="raise")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["close"] = pd.to_numeric(data["close"], errors="raise")
    data["prev_close"] = pd.to_numeric(data["prev_close"], errors="raise")
    if data[["time", "code", "sector"]].isna().any().any():
        raise ValueError("time, code and sector must not be null")
    if (data["prev_close"] <= 0).any() or (data["close"] <= 0).any():
        raise ValueError("close and prev_close must be positive")
    if data.duplicated(["time", "code", "sector"]).any():
        raise ValueError("duplicate time/code/sector rows")
    data = data[data["code"].str.match(r"^(000|001|002|003|600|601|603|605)\d{3}$")]
    data["gain"] = data["close"] / data["prev_close"] - 1.0

    signals: list[dict] = []
    seen: set[str] = set()
    for minute, frame in data.groupby("time", sort=True):
        sector_stats = (
            frame.groupby("sector")
            .agg(
                members=("code", "nunique"),
                breadth=("gain", lambda x: (x > 0).mean()),
                leaders=("gain", lambda x: (x >= config.leader_gain).sum()),
                near_limit=("gain", lambda x: (x >= config.near_limit_gain).sum()),
            )
            .reset_index()
        )
        strong = sector_stats[
            (sector_stats["members"] >= config.min_sector_members)
            & (sector_stats["breadth"] >= config.min_sector_breadth)
            & (sector_stats["leaders"] >= config.min_sector_leaders)
            & (sector_stats["near_limit"] >= config.min_sector_near_limit)
        ]
        if strong.empty:
            continue
        eligible = frame.merge(strong, on="sector", how="inner")
        eligible = eligible[
            (eligible["gain"] >= config.min_stock_gain)
            & (eligible["close"].between(config.min_price, config.max_price))
            & (~eligible["code"].isin(seen))
        ]
        # A multi-sector stock appears once, attributed to its strongest sector.
        eligible = eligible.sort_values(
            ["near_limit", "leaders", "breadth", "code"],
            ascending=[False, False, False, True],
        ).drop_duplicates("code")
        for row in eligible.itertuples(index=False):
            signals.append(
                {
                    "time": minute,
                    "code": row.code,
                    "name": row.name,
                    "sector": row.sector,
                    "price": row.close,
                    "gain": row.gain,
                    "sector_members": row.members,
                    "sector_breadth": row.breadth,
                    "sector_leaders": row.leaders,
                    "sector_near_limit": row.near_limit,
                }
            )
            seen.add(row.code)
    return pd.DataFrame(signals, columns=[
        "time", "code", "name", "sector", "price", "gain", "sector_members",
        "sector_breadth", "sector_leaders", "sector_near_limit",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="full-universe minute CSV")
    parser.add_argument("output", type=Path, help="candidate CSV to write")
    args = parser.parse_args()
    bars = pd.read_csv(args.input, dtype={"code": str})
    signals = scan_minute_bars(bars)
    signals.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(signals)} research signals written to {args.output}")
    print("Parameters:", asdict(SweepConfig()))


if __name__ == "__main__":
    main()
