"""Freeze a public historical ETF-holdings cohort and screen unchanged O'Neil rules.

Downloads are isolated, resumable and sequential because BaoStock has a global
session. Financials and execution are not inferred from price-only signals.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import socket
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qkquant.oneil import OneilConfig, prepare_prices
from qkquant.oneil_universe import checked_baostock_bars, frozen_holdings_cohort

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/research/oneil_universe_probe_20260919"
SNAPSHOT = ROOT / "data/research/oneil_historical_cohort_20260919"
OUTPUT = ROOT / "reports/oneil_historical_cohort_20260919"
START, INCEPTION, END = "2023-06-19", "2024-07-01", "2026-07-17"
FIELDS = "date,code,open,high,low,close,volume,amount,tradestatus,isST"


class BaoStockConnectionError(ConnectionError):
    def __init__(self, error_code, message):
        self.error_code = error_code
        super().__init__(message)


def check_response(response, operation):
    if response.error_code != "0":
        message = f"BaoStock {operation}: {response.error_code} {response.error_msg}"
        if response.error_code in {"10002007", "10001001"}:
            raise BaoStockConnectionError(response.error_code, message)
        raise ValueError(message)


def close_baostock_socket():
    from baostock.common import context

    connection = getattr(context, "default_socket", None)
    context.default_socket = None
    if connection is not None:
        connection.close()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bar_request(code, adjust):
    return dict(code=("sh." if code.startswith("6") else "sz.") + code, fields=FIELDS,
                start_date=START, end_date=END, frequency="d", adjustflag=adjust)


def read_cached_bars(code, adjust, snapshot):
    """Read a validated cache entry without importing or calling BaoStock."""
    path = snapshot / "sources" / f"{code}_{adjust}.json"
    payload = read_json(path)
    if payload.get("request") != bar_request(code, adjust):
        raise ValueError(f"cached request mismatch: {path}")
    if payload.get("error_code") != "0":
        raise ValueError(f"unsuccessful cached response: {path}")
    fields = FIELDS.split(",")
    records = payload.get("records")
    if (payload.get("fields") != fields or not isinstance(records, list)
            or any(not isinstance(row, dict) or set(row) != set(fields) for row in records)):
        raise ValueError(f"cached fields mismatch: {path}")
    frame = pd.DataFrame(records, columns=fields)
    dates = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if (not frame.code.eq(bar_request(code, adjust)["code"]).all() or dates.isna().any()
            or (dates < pd.Timestamp(START)).any() or (dates > pd.Timestamp(END)).any()):
        raise ValueError(f"cached code or date outside request: {path}")
    return frame


def fetch_bars(code, adjust, snapshot):
    path = snapshot / "sources" / f"{code}_{adjust}.json"
    if path.exists():
        return read_cached_bars(code, adjust, snapshot)
    import baostock as bs

    request = bar_request(code, adjust)
    rs = bs.query_history_k_data_plus(**request)
    check_response(rs, f"{code}/{adjust}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    check_response(rs, f"{code}/{adjust} after reading rows")
    if rs.fields != FIELDS.split(","):
        raise ValueError(f"unexpected BaoStock fields: {code}/{adjust}")
    frame = pd.DataFrame(rows, columns=rs.fields)
    save_json(path, dict(provider="BaoStock", api="query_history_k_data_plus", request=request,
                         retrieved_at=datetime.now(UTC).isoformat(), fields=rs.fields,
                         error_code=rs.error_code, records=frame.to_dict("records")))
    return frame


def fetch_cohort(cohort, snapshot):
    """Retry only observed connection failures, with one live session at a time."""
    import baostock as bs

    socket.setdefaulttimeout(30)
    connected = False
    try:
        for number, code in enumerate(cohort.code, 1):
            for adjust in ("3", "1"):
                if (snapshot / "sources" / f"{code}_{adjust}.json").exists():
                    read_cached_bars(code, adjust, snapshot)
                    continue
                for attempt in range(1, 4):
                    try:
                        if not connected:
                            check_response(bs.login(), "login")
                            connected = True
                        fetch_bars(code, adjust, snapshot)
                        break
                    except BaoStockConnectionError as exc:
                        connected = False
                        # A failed receive can leave unread bytes: never send
                        # logout over that connection before replacing it.
                        close_baostock_socket()
                        action = "stopping" if attempt == 3 else "reconnecting"
                        print(f"BaoStock {code}/{adjust}: connection error {exc.error_code}; "
                              f"attempt {attempt}/3, {action}", flush=True)
                        if attempt == 3:
                            raise
                        time.sleep(1)
            if number % 10 == 0 or number == len(cohort):
                print(f"Cached {number}/{len(cohort)} historical stocks", flush=True)
            time.sleep(.05)
    finally:
        try:
            if connected:
                bs.logout()
        finally:
            close_baostock_socket()


def validate_completed_snapshot(snapshot, manifest, plan, config, input_hashes):
    """Reject changed frozen inputs before any snapshot or report is written."""
    if manifest.get("config") != config:
        raise ValueError("completed snapshot configuration changed")
    if manifest.get("input_sha256") != input_hashes:
        raise ValueError("completed snapshot input hashes changed")
    if any(manifest.get(key) != value for key, value in plan.items() if key != "input_sha256"):
        raise ValueError("completed snapshot selection changed")
    expected = {f"{code}_{adjust}.json": (code, adjust)
                for code in plan["selected_codes"] for adjust in ("3", "1")}
    recorded = manifest.get("source_sha256", {})
    actual = {p.name for p in (snapshot / "sources").glob("*.json")}
    if set(recorded) != actual or actual != set(expected):
        raise ValueError("completed snapshot source inventory changed")
    for name, expected_hash in recorded.items():
        if digest(snapshot / "sources" / name) != expected_hash:
            raise ValueError(f"completed snapshot source hash changed: {name}")
        read_cached_bars(*expected[name], snapshot)
    if digest(snapshot / "bars.csv.gz") != manifest.get("bars_sha256"):
        raise ValueError("completed snapshot bars hash changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    snapshot, output = args.snapshot, args.output
    if output.resolve() == snapshot.resolve() or snapshot.resolve() in output.resolve().parents:
        raise ValueError("report output must be outside the snapshot")
    manifest_path = snapshot / "manifest.json"
    completed = read_json(manifest_path) if manifest_path.exists() else None
    if completed is not None and args.fetch:
        raise ValueError("completed snapshot is immutable; use offline mode or a new directory")
    holdings_path = PROBE / "official/holdings.csv"
    listing_path = PROBE / "baostock/all_stock_20240701.json"
    listing = read_json(listing_path)
    if listing["request"]["day"] != INCEPTION or listing["error_code"] != "0":
        raise ValueError("wrong historical listing date or failed listing response")
    holdings = pd.read_csv(holdings_path, dtype={"code": str})
    cohort, selection = frozen_holdings_cohort(holdings, pd.DataFrame(listing["records"]), INCEPTION)
    selection_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (
        holdings_path, listing_path, PROBE / "official/510500_2023_annual.pdf")}
    plan = dict(inception=INCEPTION, price_start=START, price_end=END, selected_codes=cohort.code.tolist(),
                selected_count=len(cohort), input_sha256=selection_hashes.copy(),
                selection="All main-board stocks in the 2023 year-end 510500 ETF complete holdings, publicly disclosed before inception, also present in the inception-date listing. No current survival/ST filter.")
    plan_path = snapshot / "frozen_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("frozen selection inputs changed")
    benchmark_path = ROOT / "data/research/oneil_20260919/benchmark.csv"
    input_hashes = {**selection_hashes, str(benchmark_path.relative_to(ROOT)): digest(benchmark_path)}
    cfg = OneilConfig()
    if completed is not None:
        if not plan_path.exists():
            raise ValueError("completed snapshot frozen plan is missing")
        validate_completed_snapshot(snapshot, completed, plan, asdict(cfg), input_hashes)
        for name, frame in (("cohort_selection.csv", selection), ("frozen_cohort.csv", cohort)):
            if (snapshot / name).read_text(encoding="utf-8") != frame.to_csv(index=False).replace("\r\n", "\n"):
                raise ValueError(f"completed snapshot cohort changed: {name}")
    else:
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "sources").mkdir(exist_ok=True)
        selection.to_csv(snapshot / "cohort_selection.csv", index=False)
        cohort.to_csv(snapshot / "frozen_cohort.csv", index=False)
        save_json(plan_path, plan)
    output.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_cohort(cohort, snapshot)
    frames, audits, failures = [], [], []
    for code in cohort.code:
        # A missing/invalid cache aborts sealing, so an unfinished download can
        # still resume. Empty successful histories remain explicit coverage gaps.
        raw, adjusted = (read_cached_bars(code, a, snapshot) for a in ("3", "1"))
        try:
            if raw.empty or adjusted.empty:
                failures.append(dict(code=code, reason="empty_provider_history"))
                continue
            frame, audit = checked_baostock_bars(raw, adjusted, code)
            frames.append(frame)
            audits.append(audit)
        except ValueError as exc:
            failures.append(dict(code=code, reason=str(exc)))
    if not frames:
        raise ValueError("no historical prices available")
    bars = pd.concat(frames, ignore_index=True)
    if completed is not None:
        with gzip.open(snapshot / "bars.csv.gz", "rt", encoding="utf-8") as source:
            if source.read() != bars.to_csv(index=False, lineterminator="\n"):
                raise ValueError("completed snapshot normalized bars changed")
    benchmark = pd.read_csv(benchmark_path, parse_dates=["trade_date"])
    prepared = prepare_prices(bars, benchmark, cfg)
    technical = prepared["technical"].loc[INCEPTION:END]
    candidates = []
    for i, j in zip(*np.where(technical.to_numpy()), strict=True):
        day, code = technical.index[i], technical.columns[j]
        candidates.append(dict(code=code, signal_date=day, market_ok=bool(prepared["market_ok"].loc[day]),
                               pivot=prepared["pivot"].loc[day, code], rs=prepared["rs"].loc[day, code],
                               financial_status="not_yet_audited", qualified=False))
    signals = pd.DataFrame(candidates, columns=["code", "signal_date", "market_ok", "pivot", "rs", "financial_status", "qualified"])
    signals.to_csv(output / "technical_candidates.csv", index=False)
    if completed is None:
        bars.to_csv(snapshot / "bars.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    stock_coverage = pd.DataFrame(audits)
    stock_coverage.to_csv(output / "stock_coverage.csv", index=False)
    early_ending = stock_coverage[stock_coverage.last_reported_date < pd.Timestamp(END)]
    pd.DataFrame(failures, columns=["code", "reason"]).to_csv(output / "missing_histories.csv", index=False)
    calendar = pd.DatetimeIndex(benchmark.trade_date)
    coverage = bars.groupby("trade_date").agg(reported_codes=("code", "nunique"), eligible_codes=("eligible", "sum"))
    coverage = coverage.reindex(calendar).fillna(0).astype(int)
    coverage["rs_ranked_codes"] = prepared["rs"].notna().sum(axis=1)
    coverage.to_csv(output / "date_coverage.csv")
    no_rs_days = int(coverage.loc[INCEPTION:END, "rs_ranked_codes"].eq(0).sum())
    source_hashes = {p.name: digest(p) for p in sorted((snapshot / "sources").glob("*.json"))}
    manifest = dict(plan, input_sha256=input_hashes, config=asdict(cfg), provider="BaoStock", source_sha256=source_hashes,
                    bars_sha256=digest(snapshot / "bars.csv.gz"), histories_available=len(frames),
                    empty_or_failed_histories=failures, technical_events=len(signals),
                    candidate_companies=signals.code.nunique(), market_events=int(signals.market_ok.sum()),
                    returns_tested=False, historical_financials_verified=False,
                    missing_histories_removed_from_frozen_cohort=False,
                    historical_status_source="Provider daily isST and tradestatus; not an exhaustive official announcement audit.",
                    adjustment="Provider hfq prices; raw volume in shares and amount in CNY. No corporate-action cash ledger.",
                    limitations=["ETF-disclosed fixed holdings are not the full market or exact historical index membership.",
                                 "Membership after inception is fixed: later IPOs are not added.",
                                 "Missing/delisted history remains visible; RS ranks use available eligible observations.",
                                 "Daily isST=0 does not establish ordinary 10% price limits; special trading regimes are not implemented.",
                                 "No financial acceptance or portfolio profitability is inferred from technical candidates."])
    if completed is None:
        save_json(manifest_path, manifest)
    save_json(output / "audit.json", manifest)
    ending_rows = "; ".join(f"{r.code}: {r.last_reported_date.date()}" for r in early_ending.itertuples())
    report = f"""# 欧奈尔：历史固定股票池扩样

从2024-07-01前已经公开的南方中证500ETF 2023年完整持仓中选取主板股票，
并与2024-07-01历史证券表核对，固定为 **{len(cohort)}只**。这是已公开ETF持仓样本，不是精确指数成分或全市场。
不按今天的存续状态或ST名称筛选；停牌成员保留。原246只快照保留供对照。

价格预热从{START}开始，筛选区间{INCEPTION}—{END}。取得{len(frames)}只历史行情，
{len(failures)}只为空或未通过核验（名单仍保留）。每日采用提供商历史isST与交易状态限制技术候选及相对强度排名。
所有原成长/技术参数保持不变，但样本和价格来源改变，技术信号应重新计算。
有{len(early_ending)}只的提供商行情在请求结束日前终止（{ending_rows or '无'}），仍保留在冻结名单；
本轮未据此单独认定退市或其他终止原因。回测窗口中有{no_rs_days}个日期尚无股票满足RS回看长度，候选自然为空。

得到 **{len(signals)}次技术候选、{signals.code.nunique()}家公司**，其中市场条件通过{int(signals.market_ok.sum())}次。
这些不是合格买入：新信号财报尚未逐期核验，`qualified`统一为false，本轮不输出收益回测。

逐股见`stock_coverage.csv`，缺口见`missing_histories.csv`，逐日覆盖见`date_coverage.csv`，
候选及财报待办见`technical_candidates.csv`。原始raw/hfq响应、请求、抓取时间及哈希保存在独立快照。
逐日覆盖另外记录实际参与RS排名的股票数，历史长度不足与当天不可用的股票不计入该分母。
退市后的价格不延长、不补造；已有持仓如何结算退市价值尚未实现，因此当前仅做候选筛选。

数据限制：历史交易状态与复权依赖提供商记录，尚未逐家公司动作原公告复核；
缺失证券会影响有效相对强度分母；样本为固定ETF持仓，仍有风格选择偏差且不纳入后续IPO。
历史isST=0不等于适用普通主板10%涨跌停规则，退市整理等特殊交易制度尚未实现；本轮没有交易执行。

持仓来源：[南方基金官方2023年年报](https://www.nffund.com/main/files/2024/03/29/234344061328.pdf)，
2023-12-31持仓，2024-03-30送出。完整574只明细的市值已与分项汇总核对，主板及上市日筛选另存。

复现：`.\\.venv\\Scripts\\python.exe scripts\\research_oneil_universe.py`（使用已缓存响应）。
首次采集加`--fetch`；未封存的下载可以续传，离线缺缓存则停止、不自动联网。
完成快照先核验冻结输入、参数及源文件哈希，复现只更新报告，不改写快照。旧数据库未写入。
"""
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"Selected {len(cohort)}; histories {len(frames)}; missing {len(failures)}; technical {len(signals)}; market {int(signals.market_ok.sum())}", flush=True)


if __name__ == "__main__":
    main()
