"""Fetch Tencent minute closes for sector-sweep research, with coverage audit.

Default: seven observed codes. Use --all-project-universe for the repository's
instrument list (which may be stale). This does not fetch concept membership.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import duckdb
import requests

OBSERVED = ["000753", "000592", "002383", "002218", "002617", "002753", "600644"]
URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"


def fetch_one(code: str, day: str) -> tuple[str, list[dict], str | None]:
    symbol = ("sh" if code.startswith("6") else "sz") + code
    for attempt in range(3):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(URL, params={"code": symbol}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            item = payload["data"][symbol]
            if item["data"]["date"] != day.replace("-", ""):
                return code, [], f"source date {item['data']['date']} != {day}"
            previous = float(item["qt"][symbol][4])
            if previous <= 0:
                return code, [], "invalid previous close"
            rows = []
            for point in item["data"]["data"]:
                hhmm, close, volume, amount = point.split()[:4]
                if not ("0930" <= hhmm <= "1500"):
                    continue
                rows.append({
                    "time": f"{day} {hhmm[:2]}:{hhmm[2:]}:00",
                    "code": code,
                    "close": float(close),
                    "prev_close": previous,
                    "cumulative_volume": float(volume),
                    "cumulative_amount": float(amount),
                })
            if not rows:
                return code, [], "no regular-session minute rows"
            return code, rows, None
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            if attempt == 2:
                return code, [], f"{type(exc).__name__}: {exc}"
            time.sleep(attempt + 1)
    return code, [], "unreachable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-09-16")
    parser.add_argument("--all-project-universe", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    datetime.strptime(args.date, "%Y-%m-%d")
    codes = OBSERVED
    if args.all_project_universe:
        connection = duckdb.connect("data/daily.duckdb", read_only=True)
        codes = [row[0] for row in connection.execute(
            "SELECT DISTINCT code FROM instruments WHERE instrument_type = 'stock'"
        ).fetchall()]
        connection.close()
    codes = sorted(set(codes))

    output_dir = Path("data/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "project_universe" if args.all_project_universe else "observed_7"
    output = output_dir / f"minutes_{args.date.replace('-', '')}_{suffix}.csv.gz"
    failures: dict[str, str] = {}
    counts: dict[str, int] = {}
    fieldnames = ["time", "code", "close", "prev_close", "cumulative_volume", "cumulative_amount"]
    with gzip.open(output, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_one, code, args.date): code for code in codes}
            for index, future in enumerate(as_completed(futures), 1):
                code, rows, error = future.result()
                if error:
                    failures[code] = error
                else:
                    writer.writerows(rows)
                    counts[code] = len(rows)
                if index % 250 == 0:
                    print(f"fetched {index}/{len(codes)} codes", flush=True)

    manifest = {
        "date": args.date,
        "source": URL,
        "scope": suffix,
        "requested_codes": len(codes),
        "successful_codes": len(counts),
        "minute_rows": sum(counts.values()),
        "rows_per_code": counts,
        "failures": failures,
        "concept_membership_included": False,
        "complete_universe": False,
    }
    manifest_path = output.with_suffix("").with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {manifest['minute_rows']} rows from {len(counts)}/{len(codes)} codes: {output}")
    print(f"coverage manifest: {manifest_path}")


if __name__ == "__main__":
    main()
