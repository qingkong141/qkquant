"""数据抓取封装，支持 akshare / baostock 双源 + auto 回退。

为何需要两个数据源：
- akshare 走 HTTPS，受公司/VPN 代理影响大，容易被拦；数据覆盖最全。
- baostock 走 TCP:8081，几乎不经过 HTTPS 代理，在代理环境下更稳；
  缺点是部分接口数据滞后 1-2 日。

三种模式：
- source="akshare": 仅用 akshare
- source="baostock": 仅用 baostock
- source="auto"(默认): akshare 优先，失败自动回退 baostock
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Literal

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from qkquant.config import get_settings
from qkquant.data.storage import DuckStore
from qkquant.logger import logger

Source = Literal["akshare", "sina", "baostock", "auto"]

_CN_TO_EN_DAILY = {
    "日期": "trade_date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}

_BS_ADJUST_MAP = {"hfq": "1", "qfq": "2", "": "3", "none": "3"}

DAILY_OUT_COLS = [
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "turnover",
    "adjust",
]


class FetchError(RuntimeError):
    pass


def to_bs_code(code: str) -> str:
    """6 位代码 → baostock 带前缀格式。

    - 6xxxxx / 9xxxxx → sh.xxxxxx
    - 0xxxxx / 3xxxxx → sz.xxxxxx
    - 4xxxxx / 8xxxxx → bj.xxxxxx (北交所)
    """
    c = str(code).zfill(6)
    first = c[0]
    if first in ("6", "9"):
        return f"sh.{c}"
    if first in ("0", "3"):
        return f"sz.{c}"
    if first in ("4", "8"):
        return f"bj.{c}"
    raise ValueError(f"unrecognized A-share code: {code}")


def from_bs_code(bs_code: str) -> str:
    """baostock 带前缀代码 → 6 位纯代码。"""
    if "." in bs_code:
        return bs_code.split(".")[-1]
    return bs_code


def bs_adjust(adjust: str) -> str:
    key = (adjust or "").lower()
    if key not in _BS_ADJUST_MAP:
        raise ValueError(
            f"unknown adjust '{adjust}'; expected one of hfq/qfq/'' (or 'none')"
        )
    return _BS_ADJUST_MAP[key]


class DataFetcher:
    """两源数据抓取器；支持 auto 回退。"""

    def __init__(
        self,
        store: DuckStore | None = None,
        source: Source = "auto",
    ) -> None:
        self.cfg = get_settings().data
        self.store = store
        self.source: Source = source
        self._bs_logged_in = False

    @staticmethod
    def _classify_etf(name: str) -> str:
        text = str(name).upper()
        if any(x in text for x in ("货币", "现金", "添利")):
            return "money"
        if any(x in text for x in ("黄金", "豆粕", "有色", "商品")):
            return "commodity"
        if any(x in text for x in ("债", "国开", "国债")):
            return "bond"
        if any(x in text for x in ("QDII", "纳指", "标普", "日经", "德国", "法国", "恒生", "港股")):
            return "cross_border"
        return "equity"

    @staticmethod
    def _etf_defaults(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["instrument_type"] = "etf"
        out["is_st"] = False
        out["etf_category"] = out["name"].map(DataFetcher._classify_etf)
        out["exchange"] = out["code"].map(lambda c: "SSE" if str(c).startswith("5") else "SZSE")
        out["settlement_days"] = out["etf_category"].map(
            lambda c: 0 if c in {"bond", "commodity", "money", "cross_border"} else 1
        )
        out["lot_size"] = 100
        out["price_tick"] = 0.001
        out["limit_pct"] = 0.10
        out["stamp_tax_rate"] = 0.0
        return out

    # -------- baostock 会话管理 --------

    def _bs_login(self) -> None:
        if self._bs_logged_in:
            return
        import baostock as bs

        rs = bs.login()
        if getattr(rs, "error_code", "0") != "0":
            raise FetchError(f"baostock login failed: {rs.error_msg}")
        self._bs_logged_in = True
        logger.info("baostock login ok")

    def _bs_logout(self) -> None:
        if not self._bs_logged_in:
            return
        import baostock as bs

        bs.logout()
        self._bs_logged_in = False

    def __enter__(self) -> DataFetcher:
        return self

    def __exit__(self, *exc) -> None:
        self._bs_logout()

    def close(self) -> None:
        self._bs_logout()

    # -------- 股票基本信息 --------

    def fetch_a_share_universe(self) -> pd.DataFrame:
        if self.source == "baostock":
            return self._fetch_universe_bs()
        try:
            return self._fetch_universe_ak()
        except Exception as e:
            if self.source == "akshare":
                raise
            logger.warning(f"akshare universe failed, fallback to baostock: {e}")
            return self._fetch_universe_bs()

    def _fetch_universe_ak(self) -> pd.DataFrame:
        import akshare as ak

        @retry(
            stop=stop_after_attempt(self.cfg.fetcher.retry_times),
            wait=wait_fixed(self.cfg.fetcher.retry_wait_seconds),
            retry=retry_if_exception_type((FetchError, ConnectionError, TimeoutError)),
            reraise=True,
        )
        def _call() -> pd.DataFrame:
            try:
                df = ak.stock_info_a_code_name()
            except Exception as e:
                raise FetchError(f"stock_info_a_code_name failed: {e}") from e
            if df is None or df.empty:
                raise FetchError("stock_info_a_code_name returned empty")
            return df

        df = _call()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["is_st"] = df["name"].str.contains(r"ST", na=False, regex=True)
        return df[["code", "name", "is_st"]].reset_index(drop=True)

    def _fetch_universe_bs(self) -> pd.DataFrame:
        """baostock 获取全市场 A 股基本信息。

        优先用 query_stock_basic（不依赖交易日，返回所有已上市证券含类型字段），
        失败回退到 query_all_stock(today) 的轻量模式。
        """
        import baostock as bs

        self._bs_login()
        rs = bs.query_stock_basic()
        if getattr(rs, "error_code", "0") != "0":
            raise FetchError(f"query_stock_basic failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            raise FetchError("baostock query_stock_basic returned empty")
        df = pd.DataFrame(rows, columns=rs.fields)
        # baostock: type 1=股票, 2=指数, 3=其他, 4=可转债, 5=ETF; status 1=上市, 0=退市
        df = df[(df["type"] == "1") & (df["status"] == "1")].copy()
        keep = df["code"].str.match(r"^(sh|sz|bj)\.(6|9|0|3|4|8)\d{5}$")
        df = df[keep].copy()
        df["code"] = df["code"].apply(from_bs_code)
        df.rename(columns={"code_name": "name"}, inplace=True)
        df["is_st"] = df["name"].str.contains(r"ST", na=False, regex=True)
        return df[["code", "name", "is_st"]].reset_index(drop=True)

    # -------- ETF 基本信息 --------

    def fetch_etf_universe(self) -> pd.DataFrame:
        if self.source == "baostock":
            return self._fetch_etf_universe_bs()
        if self.source == "sina":
            return self._fetch_etf_universe_sina()
        try:
            return self._fetch_etf_universe_ak()
        except Exception as e:
            if self.source == "akshare":
                raise
            logger.warning(f"akshare ETF universe failed, fallback to sina: {e}")
            try:
                return self._fetch_etf_universe_sina()
            except Exception as sina_error:
                logger.warning(f"sina ETF universe failed, fallback to baostock: {sina_error}")
                return self._fetch_etf_universe_bs()

    def _fetch_etf_universe_ak(self) -> pd.DataFrame:
        import akshare as ak

        raw = ak.fund_etf_spot_em()
        if raw is None or raw.empty:
            raise FetchError("fund_etf_spot_em returned empty")
        code_col = next((c for c in ("代码", "基金代码", "code") if c in raw.columns), None)
        name_col = next((c for c in ("名称", "基金简称", "name") if c in raw.columns), None)
        if not code_col or not name_col:
            raise FetchError(f"ETF list missing code/name columns: {list(raw.columns)}")
        df = pd.DataFrame({
            "code": raw[code_col].astype(str).str.extract(r"(\d{6})", expand=False),
            "name": raw[name_col].astype(str),
        }).dropna(subset=["code"])
        return self._etf_defaults(df).reset_index(drop=True)

    def _fetch_etf_universe_sina(self) -> pd.DataFrame:
        import akshare as ak

        raw = ak.fund_etf_category_sina(symbol="ETF基金")
        if raw is None or raw.empty:
            raise FetchError("fund_etf_category_sina returned empty")
        df = pd.DataFrame(
            {
                "code": raw["代码"].astype(str).str.extract(r"(\d{6})", expand=False),
                "name": raw["名称"].astype(str),
            }
        ).dropna(subset=["code"])
        return self._etf_defaults(df).reset_index(drop=True)

    def _fetch_etf_universe_bs(self) -> pd.DataFrame:
        import baostock as bs

        self._bs_login()
        rs = bs.query_stock_basic()
        if getattr(rs, "error_code", "0") != "0":
            raise FetchError(f"query_stock_basic failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=rs.fields)
        raw = raw[(raw["type"] == "5") & (raw["status"] == "1")].copy()
        if raw.empty:
            raise FetchError("baostock ETF universe returned empty")
        df = pd.DataFrame({
            "code": raw["code"].map(from_bs_code),
            "name": raw["code_name"],
            "list_date": pd.to_datetime(raw.get("ipoDate"), errors="coerce"),
        })
        return self._etf_defaults(df).reset_index(drop=True)

    # -------- 沪深 300 成分股 --------

    def fetch_hs300_constituents(self) -> list[str]:
        if self.source == "baostock":
            return self._fetch_hs300_bs()
        try:
            return self._fetch_hs300_ak()
        except Exception as e:
            if self.source == "akshare":
                raise
            logger.warning(f"akshare HS300 failed, fallback to baostock: {e}")
            return self._fetch_hs300_bs()

    def _fetch_hs300_ak(self) -> list[str]:
        import akshare as ak

        attempts: list[tuple[str, callable]] = [
            ("index_stock_cons_csindex(000300)", lambda: ak.index_stock_cons_csindex(symbol="000300")),
            ("index_stock_cons_sina(000300)", lambda: ak.index_stock_cons_sina(symbol="000300")),
        ]
        last_err: Exception | None = None
        for name, call in attempts:
            try:
                df = call()
                if df is None or df.empty:
                    raise FetchError(f"{name} empty")
                code_col = next(
                    (c for c in ("成分券代码", "code", "品种代码", "证券代码") if c in df.columns),
                    None,
                )
                if code_col is None:
                    raise FetchError(f"{name} missing code col; got {list(df.columns)}")
                codes = df[code_col].astype(str).str.zfill(6).tolist()
                logger.info(f"fetched {len(codes)} HS300 via {name}")
                return codes
            except Exception as e:
                last_err = e
                logger.warning(f"HS300 fetch via {name} failed: {e}")
        raise FetchError(f"all akshare HS300 sources failed; last: {last_err}")

    def _fetch_hs300_bs(self) -> list[str]:
        import baostock as bs

        self._bs_login()
        rs = bs.query_hs300_stocks()
        if getattr(rs, "error_code", "0") != "0":
            raise FetchError(f"query_hs300_stocks failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            raise FetchError("baostock query_hs300_stocks returned empty")
        df = pd.DataFrame(rows, columns=rs.fields)
        codes = df["code"].apply(from_bs_code).tolist()
        logger.info(f"fetched {len(codes)} HS300 via baostock")
        return codes

    # -------- 日线 --------

    def fetch_daily(
        self,
        code: str,
        start: date | str,
        end: date | str,
        adjust: str | None = None,
    ) -> pd.DataFrame:
        adj = adjust or self.cfg.fetcher.adjust
        is_etf = False
        if self.store is not None:
            inst = self.store.load_instruments([str(code).zfill(6)])
            is_etf = not inst.empty and inst.iloc[0].get("instrument_type") == "etf"
        if is_etf and self.source == "sina":
            return self._fetch_etf_daily_sina(code, start, end, adj)
        if is_etf and self.source != "baostock":
            try:
                return self._fetch_etf_daily_ak(code, start, end, adj)
            except Exception as e:
                if self.source == "akshare":
                    raise
                logger.debug(f"akshare ETF daily({code}) failed: {e}; fallback sina")
                try:
                    return self._fetch_etf_daily_sina(code, start, end, adj)
                except Exception as sina_error:
                    logger.debug(f"sina ETF daily({code}) failed: {sina_error}; fallback baostock")
                    return self._fetch_daily_bs(code, start, end, adj)
        if self.source == "baostock":
            return self._fetch_daily_bs(code, start, end, adj)
        try:
            df = self._fetch_daily_ak(code, start, end, adj)
            if df is None or df.empty:
                if self.source == "auto":
                    return self._fetch_daily_bs(code, start, end, adj)
            return df
        except Exception as e:
            if self.source == "akshare":
                raise
            logger.debug(f"akshare fetch_daily({code}) failed: {e}; fallback baostock")
            return self._fetch_daily_bs(code, start, end, adj)

    def _fetch_etf_daily_ak(self, code: str, start, end, adjust: str) -> pd.DataFrame:
        import akshare as ak

        raw = ak.fund_etf_hist_em(
            symbol=str(code).zfill(6),
            period="daily",
            start_date=pd.to_datetime(start).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end).strftime("%Y%m%d"),
            adjust=adjust,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.rename(columns=_CN_TO_EN_DAILY).copy()
        df["code"] = str(code).zfill(6)
        df["adjust"] = adjust
        for col in ("turnover", "amount", "pct_chg"):
            if col not in df.columns:
                df[col] = None
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df[DAILY_OUT_COLS]

    def _fetch_etf_daily_sina(self, code: str, start, end, adjust: str) -> pd.DataFrame:
        """Sina ETF daily bars. Sina does not expose an adjustment selector."""
        import akshare as ak

        pure_code = str(code).zfill(6)
        prefix = "sh" if pure_code.startswith("5") else "sz"
        raw = ak.fund_etf_hist_sina(symbol=f"{prefix}{pure_code}")
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        raw = raw[(raw["date"] >= start_ts) & (raw["date"] <= end_ts)]
        if raw.empty:
            return pd.DataFrame()
        close = pd.to_numeric(raw["close"], errors="coerce")
        df = pd.DataFrame(
            {
                "code": pure_code,
                "trade_date": raw["date"],
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": close,
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
                "amount": pd.to_numeric(raw["amount"], errors="coerce"),
                "pct_chg": close.pct_change(fill_method=None) * 100,
                "turnover": None,
                "adjust": adjust,
            }
        )
        return df.dropna(subset=["close"])[DAILY_OUT_COLS].reset_index(drop=True)

    def _fetch_daily_ak(
        self, code: str, start, end, adjust: str
    ) -> pd.DataFrame:
        import akshare as ak

        start_s = pd.to_datetime(start).strftime("%Y%m%d")
        end_s = pd.to_datetime(end).strftime("%Y%m%d")

        @retry(
            stop=stop_after_attempt(self.cfg.fetcher.retry_times),
            wait=wait_fixed(self.cfg.fetcher.retry_wait_seconds),
            retry=retry_if_exception_type((FetchError, ConnectionError, TimeoutError)),
            reraise=True,
        )
        def _call() -> pd.DataFrame:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_s,
                    end_date=end_s,
                    adjust=adjust,
                )
            except Exception as e:
                raise FetchError(f"stock_zh_a_hist({code}) failed: {e}") from e
            return df if df is not None else pd.DataFrame()

        raw = _call()
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.rename(columns=_CN_TO_EN_DAILY).copy()
        keep = [c for c in _CN_TO_EN_DAILY.values() if c in df.columns]
        df = df[keep].copy()
        df["code"] = code
        df["adjust"] = adjust
        for col in ("turnover", "amount", "pct_chg"):
            if col not in df.columns:
                df[col] = None
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df[DAILY_OUT_COLS]

    def _fetch_daily_bs(
        self, code: str, start, end, adjust: str
    ) -> pd.DataFrame:
        import baostock as bs

        self._bs_login()
        bs_code = to_bs_code(code)
        start_s = pd.to_datetime(start).strftime("%Y-%m-%d")
        end_s = pd.to_datetime(end).strftime("%Y-%m-%d")
        fields = "date,code,open,high,low,close,volume,amount,pctChg,turn,tradestatus"

        @retry(
            stop=stop_after_attempt(self.cfg.fetcher.retry_times),
            wait=wait_fixed(self.cfg.fetcher.retry_wait_seconds),
            retry=retry_if_exception_type((FetchError, ConnectionError, TimeoutError)),
            reraise=True,
        )
        def _call() -> pd.DataFrame:
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_s,
                end_date=end_s,
                frequency="d",
                adjustflag=bs_adjust(adjust),
            )
            if getattr(rs, "error_code", "0") != "0":
                raise FetchError(f"baostock k_data({bs_code}) err: {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=rs.fields)

        raw = _call()
        if raw.empty:
            return pd.DataFrame()

        # 过滤停牌日(tradestatus != 1)
        raw = raw[raw["tradestatus"] == "1"].copy()
        if raw.empty:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "code": code,
                "trade_date": pd.to_datetime(raw["date"]),
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
                "amount": pd.to_numeric(raw["amount"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["pctChg"], errors="coerce"),
                "turnover": pd.to_numeric(raw["turn"], errors="coerce"),
                "adjust": adjust,
            }
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df[DAILY_OUT_COLS]

    # -------- 批量 --------

    def bulk_update_daily(
        self,
        codes: list[str],
        start: date | str,
        end: date | str,
        adjust: str | None = None,
        incremental: bool = True,
        jobs: int = 1,
    ) -> dict:
        if self.store is None:
            raise RuntimeError("bulk_update_daily requires DuckStore")

        adj = adjust or self.cfg.fetcher.adjust
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        jobs = max(1, int(jobs))

        if self.source in ("baostock", "auto"):
            try:
                self._bs_login()
            except Exception as e:
                if self.source == "baostock":
                    raise
                logger.warning(f"baostock preemptive login failed (auto still OK): {e}")

        tasks: list[tuple[str, date]] = []
        skipped = 0
        for code in codes:
            real_start = start_d
            if incremental:
                last = self.store.get_max_trade_date(code, adj)
                if last is not None:
                    real_start = max(start_d, last + pd.Timedelta(days=1).to_pytimedelta())
                if real_start > end_d:
                    skipped += 1
                    continue
            tasks.append((code, real_start))

        total_rows = 0
        failed: list[str] = []

        # Only akshare is parallelized. Baostock uses a process-global session and
        # is kept sequential to avoid corrupting the login/query state.
        effective_jobs = jobs if self.source == "akshare" else 1

        if effective_jobs == 1:
            for code, real_start in tqdm(tasks, desc="update_daily", unit="code"):
                try:
                    df = self.fetch_daily(code, real_start, end_d, adjust=adj)
                    if not df.empty:
                        n = self.store.upsert_daily(df)
                        total_rows += n
                except Exception as e:
                    logger.warning(f"fetch_daily({code}) failed: {e}")
                    failed.append(code)
                time.sleep(self.cfg.fetcher.request_sleep_ms / 1000.0)
        else:
            def _fetch_one(item: tuple[str, date]) -> tuple[str, pd.DataFrame]:
                code, real_start = item
                df = self.fetch_daily(code, real_start, end_d, adjust=adj)
                time.sleep(self.cfg.fetcher.request_sleep_ms / 1000.0)
                return code, df

            with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
                future_map = {executor.submit(_fetch_one, item): item[0] for item in tasks}
                for future in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc="update_daily",
                    unit="code",
                ):
                    code = future_map[future]
                    try:
                        _, df = future.result()
                        if not df.empty:
                            n = self.store.upsert_daily(df)
                            total_rows += n
                    except Exception as e:
                        logger.warning(f"fetch_daily({code}) failed: {e}")
                        failed.append(code)

        return {
            "total_rows": total_rows,
            "failed": failed,
            "skipped_up_to_date": skipped,
            "codes": len(codes),
            "source": self.source,
            "jobs": effective_jobs,
        }


    def fetch_eps_ttm(self, codes: list[str]) -> pd.DataFrame:
        """获取 TTM EPS（基于 baostock query_profit_data）。

        只查当前最可能已发布的最新年季度，每只票最多尝试 2 个季度的查询。
        预计 ~30 秒/只，300 只约 2.5 小时。建议挂机运行。
        """
        import baostock as bs

        self._bs_login()
        rows: list[dict] = []
        # 根据当前日期推断最新可用季报
        today = date.today()
        if today.month <= 4:
            candidates = [(today.year - 1, 3)]  # 年报还没出完
        elif today.month <= 8:
            candidates = [(today.year, 1), (today.year - 1, 4)]  # Q1 或去年 Q4
        else:
            candidates = [(today.year, 2), (today.year, 1)]  # Q2 或 Q1

        for code in tqdm(codes, desc="eps_ttm", unit="code"):
            try:
                bs_code = to_bs_code(code)
                for yr, qtr in candidates:
                    rs = bs.query_profit_data(bs_code, year=yr, quarter=qtr)
                    if getattr(rs, "error_code", "0") != "0":
                        continue
                    row_data = None
                    while rs.next():
                        rd = rs.get_row_data()
                        if rd and len(rd) > 7:
                            row_data = rd
                    if row_data is None:
                        continue
                    eps = float(row_data[7]) if row_data[7] and row_data[7] != "" else 0.0
                    if eps > 0:
                        rows.append({
                            "code": code,
                            "pub_date": row_data[1],
                            "stat_date": row_data[2],
                            "eps_ttm": eps,
                            "total_share": float(row_data[9] or 0),
                            "liqa_share": float(row_data[10] or 0),
                        })
                        break  # 取到了，跳出季度循环
            except Exception as e:
                logger.warning(f"eps_ttm({code}) failed: {e}")
        return pd.DataFrame(rows)

    def fetch_market_caps(self, codes: list[str]) -> pd.DataFrame:
        """获取总市值和流通市值。"""
        import akshare as ak

        rows: list[dict] = []
        for code in tqdm(codes, desc="market_cap", unit="code"):
            try:
                info = ak.stock_individual_info_em(symbol=code)
                info_dict = dict(zip(info["item"], info["value"]))
                total_cap = float(info_dict.get("总市值", 0) or 0)
                float_cap = float(info_dict.get("流通市值", 0) or 0)
                if float_cap > 0:
                    rows.append(
                        {"code": code, "total_cap": total_cap, "float_cap": float_cap}
                    )
            except Exception as e:
                logger.warning(f"market_cap({code}) failed: {e}")
            time.sleep(self.cfg.fetcher.request_sleep_ms / 1000.0)
        return pd.DataFrame(rows)


__all__ = [
    "DAILY_OUT_COLS",
    "DataFetcher",
    "FetchError",
    "Source",
    "bs_adjust",
    "from_bs_code",
    "to_bs_code",
]
