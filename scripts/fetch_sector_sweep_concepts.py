"""Fetch a current snapshot of THS concept membership for research replay.

Membership is a post-close snapshot, not a historical point-in-time feed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = "https://q.10jqka.com.cn/gn/detail/code/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_page(board_code: str, page: int = 1) -> tuple[set[str], int, str | None]:
    url = f"{ROOT}{board_code}/" + (f"page/{page}/" if page > 1 else "")
    for attempt in range(3):
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content.decode("gbk", errors="replace"), "lxml")
            codes = {
                cell.get_text(strip=True)
                for cell in soup.select("table.m-table tbody tr td:nth-child(2) a")
                if re.fullmatch(r"\d{6}", cell.get_text(strip=True))
            }
            page_info = soup.select_one(".page_info")
            pages = int(page_info.get_text(strip=True).split("/")[-1]) if page_info else 1
            if not codes:
                return set(), pages, "no constituent codes"
            return codes, pages, None
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                return set(), 0, f"{type(exc).__name__}: {exc}"
            time.sleep(attempt + 1)
    return set(), 0, "unreachable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-codes", help="comma-separated concept codes; omit for all")
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{ROOT}307822/", headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content.decode("gbk", errors="replace"), "lxml")
    boards = {
        match.group(1): link.get_text(" ", strip=True)
        for link in soup.select("div.cate_inner a[href]")
        if (match := re.search(r"/code/(\d{6})/", link["href"]))
    }
    if len(boards) < 300:
        raise RuntimeError(f"concept list unexpectedly small: {len(boards)}")
    if args.board_codes:
        requested = set(args.board_codes.split(","))
        unknown = requested - boards.keys()
        if unknown:
            raise ValueError(f"unknown concept codes: {sorted(unknown)}")
        boards = {code: name for code, name in boards.items() if code in requested}

    memberships: dict[str, set[str]] = {code: set() for code in boards}
    failures: dict[str, str] = {}
    pages: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_page, code): code for code in boards}
        for index, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            codes, count, error = future.result()
            if error:
                failures[f"{code}/1"] = error
            else:
                memberships[code].update(codes)
                pages[code] = count
            if index % 100 == 0:
                print(f"first pages: {index}/{len(boards)}", flush=True)

    remaining = [(code, page) for code, count in pages.items() for page in range(2, count + 1)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_page, code, page): (code, page) for code, page in remaining}
        for index, future in enumerate(as_completed(futures), 1):
            code, page = futures[future]
            codes, _, error = future.result()
            if error:
                failures[f"{code}/{page}"] = error
            else:
                memberships[code].update(codes)
            if index % 250 == 0:
                print(f"remaining pages: {index}/{len(remaining)}", flush=True)

    output_dir = Path("data/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "selected" if args.board_codes else "all"
    output = output_dir / f"concept_membership_20260916_{suffix}.csv.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["code", "sector", "board_code"])
        for board_code, codes in memberships.items():
            for code in sorted(codes):
                writer.writerow([code, boards[board_code], board_code])
    manifest = {
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "source": ROOT,
        "board_count": len(boards),
        "boards_with_members": sum(bool(x) for x in memberships.values()),
        "requested_pages": len(boards) + len(remaining),
        "failed_pages": failures,
        "membership_rows": sum(map(len, memberships.values())),
        "point_in_time_membership": False,
    }
    manifest_path = output_dir / f"concept_membership_20260916_{suffix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {manifest['membership_rows']} memberships, failures {len(failures)}: {output}")


if __name__ == "__main__":
    main()
