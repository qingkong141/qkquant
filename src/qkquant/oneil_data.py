"""Public-data snapshot for the isolated O'Neil research experiment."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import requests

DATA_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
QUARTER_URL = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/lrbAjaxNew"
PRICE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def normalize_turnover(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace mixed local lots/shares with one source; quarantine bad turnover."""
    bars = bars.copy()
    bars["local_volume"] = bars["volume"]
    bars["volume"] = bars.raw_volume * 100
    bad = ((bars.amount < bars.volume * bars.raw_low * .98)
           | (bars.amount > bars.volume * bars.raw_high * 1.02)
           | (bars.amount <= 0) | (bars.volume <= 0))
    bars.loc[bad, ["volume", "amount"]] = float("nan")
    return bars, int(bad.sum())


def _get(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def fetch_snapshot(database: str | Path, output: str | Path) -> dict:
    """Freeze the existing qfq stock cohort; never modify the source database.

    Cached payloads preserve the downloaded financial vintage. They are NOT
    historical as-published vintages, even when NOTICE_DATE exists.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "source"
    cache.mkdir(exist_ok=True)
    with duckdb.connect(str(database), read_only=True) as con:
        bars = con.execute("""
            SELECT b.* FROM daily_bars b JOIN instruments i USING(code)
            WHERE b.adjust='qfq' AND i.instrument_type='stock'
              AND NOT COALESCE(i.is_st,FALSE)
            ORDER BY b.code,b.trade_date
        """).fetch_df()
    # Main-board pilot avoids treating 10% and 20% price-limit boards alike.
    bars = bars[bars.code.str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]
    counts = bars.groupby("trade_date").code.nunique()
    end = counts[counts >= bars.code.nunique() * .95].index.max()
    bars = bars[bars.trade_date <= end].copy()
    codes = sorted(bars.code.unique())
    start = bars.trade_date.min().strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")

    def cached(name, url, params, allow_empty=False):
        path = cache / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        payload = _get(url, params)
        if (not payload.get("data") and not (payload.get("result") or {}).get("data")
                and not (allow_empty and payload.get("data") == [])):
            raise ValueError(f"empty provider response: {name}")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def prices(code):
        symbol = ("sh" if code.startswith("6") or code == "000300" else "sz") + code
        payload = cached(f"raw_{symbol}_{start}_{end_text}", PRICE_URL,
                         {"param": f"{symbol},day,{start},{end_text},1500,"})
        rows = payload["data"][symbol]["day"]
        frame = pd.DataFrame([r[:6] for r in rows],
                             columns=["trade_date", "raw_open", "raw_close", "raw_high", "raw_low", "raw_volume"])
        frame["code"] = code
        frame["trade_date"] = pd.to_datetime(frame.trade_date)
        for field in frame.columns.difference(["code", "trade_date"]):
            frame[field] = pd.to_numeric(frame[field], errors="raise")
        return frame

    # The provider silently caps each quarterly response at five reports.
    # Fetch single-quarter EPS, not differences of cumulative EPS (which need
    # not be additive when share counts change).
    dates = pd.date_range("2022-03-31", end, freq="QE").strftime("%Y-%m-%d").tolist()

    def financials(code):
        annual_payload = cached(f"annual_{code}_{end_text}", DATA_URL, {
            "reportName": "RPT_F10_FINANCE_MAINFINADATA", "columns": "ALL",
            "filter": f'(SECURITY_CODE="{code}")(REPORT_DATE>=\'2019-01-01\')(REPORT_DATE<=\'{end_text}\')',
            "pageSize": 100, "pageNumber": 1, "sortColumns": "REPORT_DATE", "sortTypes": "-1",
        })
        main = annual_payload["result"]["data"]
        org_type = main[0]["ORG_TYPE"]
        company_type = {"银行": "3", "证券": "1", "保险": "2", "通用": "4"}.get(org_type)
        if company_type is None:
            raise ValueError(f"unsupported financial company type: {code} {org_type}")
        symbol = ("SH" if code.startswith("6") else "SZ") + code
        quarters = []
        for offset in range(0, len(dates), 5):
            batch = dates[offset:offset + 5]
            quarters.extend(cached(f"quarter_{code}_{batch[0]}_{batch[-1]}", QUARTER_URL, {
                "companyType": company_type, "reportDateType": "0", "reportType": "2",
                "code": symbol, "dates": ",".join(batch),
            }, allow_empty=True)["data"])
        rows = []
        for kind, source_rows in (("annual", main), ("quarter", quarters)):
            for row in source_rows:
                period = row["REPORT_DATE"][:10]
                if kind == "annual" and not period.endswith("12-31"):
                    continue
                rows.append({
                    "code": code, "kind": kind, "report_date": period,
                    "notice_date": row["NOTICE_DATE"], "update_date": row.get("UPDATE_DATE"),
                    "eps": row.get("EPSJB") if kind == "annual" else row.get("BASIC_EPS"),
                    "revenue": row.get("TOTALOPERATEREVE") if kind == "annual" else row.get(
                        "OPERATE_INCOME" if org_type in ("银行", "证券", "保险") else "TOTAL_OPERATE_INCOME"),
                    "roe": row.get("ROEJQ") if kind == "annual" else None,
                })
        return pd.DataFrame(rows)

    frames, fundamentals, failures = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fn, code): (kind, code)
                for kind, fn in (("price", prices), ("financial", financials)) for code in codes}
        for n, future in enumerate(as_completed(jobs), 1):
            kind, code = jobs[future]
            try:
                (frames if kind == "price" else fundamentals).append(future.result())
            except Exception as exc:
                failures.append({"code": code, "kind": kind, "error": str(exc)})
            if n % 50 == 0:
                print(f"snapshot {n}/{len(jobs)}, failures={len(failures)}", flush=True)
    if not frames or not fundamentals:
        raise ValueError("snapshot requires both historical prices and financial statements")
    raw = pd.concat(frames, ignore_index=True)
    bars = bars.merge(raw, on=["code", "trade_date"], how="left", validate="one_to_one")
    # The old database mixes AkShare lots and BaoStock shares. Tencent's daily
    # stock volume is in lots; normalize the entire history, not just new rows.
    bars, bad_turnover_count = normalize_turnover(bars)
    bars.to_csv(output / "bars.csv.gz", index=False)
    pd.concat(fundamentals, ignore_index=True).to_csv(output / "financials.csv", index=False)
    benchmark = prices("000300")
    benchmark.to_csv(output / "benchmark.csv", index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "start": start, "end": end_text,
        "codes": codes, "rows": len(bars), "financial_codes": len(fundamentals),
        "missing_raw_rows": int(bars.raw_close.isna().sum()), "failures": failures,
        "quarantined_turnover_rows": bad_turnover_count,
        "volume_unit": "shares: Tencent daily lots multiplied by 100",
        "price_source": PRICE_URL, "financial_sources": [DATA_URL, QUARTER_URL],
        "local_adjusted_source": str(Path(database).resolve()),
        "historical_constituents_verified": False, "historical_financial_vintages_verified": False,
        "limitations": ["现存主板股票样本，存在幸存者/成分选择偏差；不是历史全市场。",
                        "财报是本次下载版本；公告日与更新日都保留，不能证明历史原始版本。",
                        "本地前复权序列未完成跨源复权核验；原始价格来自腾讯。"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_snapshot(path: str | Path):
    path = Path(path)
    return (
        pd.read_csv(path / "bars.csv.gz", dtype={"code": str}, parse_dates=["trade_date"]),
        pd.read_csv(path / "financials.csv", dtype={"code": str},
                    parse_dates=["report_date", "notice_date", "update_date"]),
        pd.read_csv(path / "benchmark.csv", parse_dates=["trade_date"]),
        json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    )
