"""Fetch dated Sina trade prints for the seven observed orders (research only).

These are trade records, not a full order book or verified exchange-level ticks.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ORDERS = ("000753", "000592", "002383", "002218", "002617", "002753", "600644")
ORDER_MINUTES = ("10:30", "10:14", "09:42", "09:39", "09:33", "09:32")
BASE = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"


def fetch_code(session: requests.Session, code: str, day: str) -> tuple[pd.DataFrame, int]:
    symbol = ("sh" if code.startswith("6") else "sz") + code
    params = {
        "symbol": symbol,
        "num": "60",
        "page": "1",
        "sort": "ticktime",
        "asc": "1",
        "volume": "0",
        "amount": "0",
        "type": "0",
        "day": day,
    }
    headers = {
        "Referer": f"https://vip.stock.finance.sina.com.cn/quotes_service/view/cn_bill.php?symbol={symbol}",
        "User-Agent": "Mozilla/5.0",
    }
    count_response = session.get(BASE + "CN_Bill.GetBillListCount", params=params, headers=headers, timeout=15)
    count_response.raise_for_status()
    expected = int(count_response.json())
    rows: list[dict] = []
    for page in range(1, math.ceil(expected / 60) + 1):
        params["page"] = str(page)
        response = session.get(BASE + "CN_Bill.GetBillList", params=params, headers=headers, timeout=15)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"unexpected response for {code} page {page}")
        rows.extend(batch)
    if len(rows) < expected:
        raise ValueError(f"{code}: expected {expected} records, received {len(rows)}")
    # On a live trading day, the final page may grow between count and page reads.
    # Ascending order keeps earlier pages stable; retain the initial snapshot size.
    frame = pd.DataFrame(rows[:expected])
    frame.insert(0, "date", day)
    frame.insert(1, "code", code)
    return frame, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-09-16")
    parser.add_argument("--output-dir", type=Path, default=Path("data/research"))
    parser.add_argument("--all-minute-controls", action="store_true")
    parser.add_argument("--first-hour-candidates", action="store_true")
    args = parser.parse_args()
    if args.all_minute_controls and args.first_hour_candidates:
        parser.error("choose one candidate-selection mode")
    codes = ORDERS
    if args.all_minute_controls or args.first_hour_candidates:
        minute_path = args.output_dir / f"minutes_{args.date.replace('-', '')}_project_universe.csv.gz"
        minute = pd.read_csv(minute_path, dtype={"code": str})
        minute["code"] = minute["code"].str.zfill(6)
        if args.first_hour_candidates:
            time_mask = minute["time"].str[11:16].between("09:30", "10:30")
        else:
            time_mask = minute["time"].str[11:16].isin(ORDER_MINUTES)
        minute = minute[
            time_mask
            & minute["code"].str.match(r"^(000|001|002|003|600|601|603|605)\d{3}$")
            & minute["close"].between(3, 12)
            & (minute["prev_close"] > 0)
            & (minute["close"] / minute["prev_close"] >= 1.04)
        ]
        selected = set(minute["code"])
        if args.all_minute_controls:
            selected |= set(ORDERS)
        codes = tuple(sorted(selected))
    session = requests.Session()
    session.trust_env = False
    frames = []
    counts = {}
    for code in codes:
        frame, count = fetch_code(session, code, args.date)
        frames.append(frame)
        counts[code] = count
        print(f"{code}: {count} trade records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.first_hour_candidates:
        suffix = "first_hour_candidates"
    else:
        suffix = "minute_controls" if args.all_minute_controls else "observed_7"
    stem = f"ticks_{args.date.replace('-', '')}_{suffix}"
    path = args.output_dir / f"{stem}.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")
    manifest = args.output_dir / f"{stem}.manifest.json"
    manifest.write_text(
        pd.Series({
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "source": BASE,
            "date_requested": args.date,
            "record_counts": counts,
            "trade_prints_not_order_book": True,
        }).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
