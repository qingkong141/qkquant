"""DuckDB 本地行情存储：日线、交易日历、股票基本信息、指数成分股。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from qkquant.config import get_settings
from qkquant.data.board import is_cn_main_board
from qkquant.logger import logger

DAILY_COLUMNS = [
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

INSTRUMENT_COLUMNS = [
    "code", "name", "is_st", "instrument_type", "etf_category", "exchange",
    "list_date", "delist_date", "settlement_days", "lot_size", "price_tick",
    "limit_pct", "stamp_tax_rate", "benchmark_code", "management_fee",
]


class DuckStore:
    """DuckDB 封装。线程不安全；每个线程应持有独立实例。"""

    def __init__(self, path: str | Path | None = None, read_only: bool = False) -> None:
        cfg = get_settings().data
        self.path: Path = Path(path) if path else cfg.duckdb_abs_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._read_only = read_only
        if not read_only:
            self._init_schema()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.path), read_only=self._read_only)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _init_schema(self) -> None:
        con = self.conn
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT NOT NULL,
                trade_date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                pct_chg DOUBLE,
                turnover DOUBLE,
                adjust TEXT NOT NULL,
                PRIMARY KEY (code, trade_date, adjust)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                code TEXT PRIMARY KEY,
                name TEXT,
                is_st BOOLEAN DEFAULT FALSE,
                total_cap DOUBLE,
                float_cap DOUBLE,
                updated_at TIMESTAMP
            )
            """
        )
        # 兼容旧表，补上可能缺失的市值列
        for col, dtype in [("total_cap", "DOUBLE"), ("float_cap", "DOUBLE")]:
            try:
                con.execute(f"ALTER TABLE instruments ADD COLUMN {col} {dtype}")
            except Exception:
                pass
        instrument_columns = {
            "instrument_type": "TEXT DEFAULT 'stock'",
            "etf_category": "TEXT",
            "exchange": "TEXT",
            "list_date": "DATE",
            "delist_date": "DATE",
            "settlement_days": "INTEGER DEFAULT 1",
            "lot_size": "INTEGER DEFAULT 100",
            "price_tick": "DOUBLE DEFAULT 0.01",
            "limit_pct": "DOUBLE DEFAULT 0.10",
            "stamp_tax_rate": "DOUBLE DEFAULT 0.001",
            "benchmark_code": "TEXT",
            "management_fee": "DOUBLE",
        }
        for col, dtype in instrument_columns.items():
            try:
                con.execute(f"ALTER TABLE instruments ADD COLUMN {col} {dtype}")
            except Exception:
                pass
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_calendar (
                trade_date DATE PRIMARY KEY
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS index_constituents (
                index_code TEXT NOT NULL,
                code TEXT NOT NULL,
                updated_at TIMESTAMP,
                PRIMARY KEY (index_code, code)
            )
            """
        )

    # ---------- daily bars ----------

    def upsert_daily(self, df: pd.DataFrame) -> int:
        """插入/更新日线数据。df 必须包含 DAILY_COLUMNS。返回写入行数。"""
        if df is None or df.empty:
            return 0
        missing = set(DAILY_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"daily dataframe missing columns: {missing}")
        df = df[DAILY_COLUMNS].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        con = self.conn
        con.register("_tmp_daily", df)
        con.execute(
            """
            INSERT OR REPLACE INTO daily_bars
            SELECT code, trade_date, open, high, low, close,
                   volume, amount, pct_chg, turnover, adjust
            FROM _tmp_daily
            """
        )
        con.unregister("_tmp_daily")
        return len(df)

    def get_max_trade_date(self, code: str, adjust: str) -> date | None:
        row = self.conn.execute(
            "SELECT MAX(trade_date) FROM daily_bars WHERE code = ? AND adjust = ?",
            [code, adjust],
        ).fetchone()
        return row[0] if row and row[0] else None

    def load_daily(
        self,
        codes: list[str] | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """按代码/日期范围读取日线。空返回空 DataFrame。"""
        sql = "SELECT * FROM daily_bars WHERE adjust = ?"
        params: list = [adjust]
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            sql += f" AND code IN ({placeholders})"
            params.extend(codes)
        if start:
            sql += " AND trade_date >= ?"
            params.append(pd.to_datetime(start).date())
        if end:
            sql += " AND trade_date <= ?"
            params.append(pd.to_datetime(end).date())
        sql += " ORDER BY code, trade_date"
        df = self.conn.execute(sql, params).fetch_df()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    # ---------- instruments ----------

    def upsert_instruments(self, df: pd.DataFrame) -> int:
        """Upsert stock or ETF master data; missing optional fields keep defaults."""
        if df is None or df.empty:
            return 0
        df = df.copy()
        defaults = {
            "name": "", "is_st": False, "instrument_type": "stock",
            "etf_category": None, "exchange": None, "list_date": None,
            "delist_date": None, "settlement_days": 1, "lot_size": 100,
            "price_tick": 0.01, "limit_pct": 0.10, "stamp_tax_rate": 0.001,
            "benchmark_code": None, "management_fee": None,
        }
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value
        df = df[INSTRUMENT_COLUMNS].copy()
        for col in ("list_date", "delist_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df["updated_at"] = pd.Timestamp.now()
        con = self.conn
        con.register("_tmp_inst", df)
        con.execute(
            """
            INSERT INTO instruments (
                code, name, is_st, instrument_type, etf_category, exchange,
                list_date, delist_date, settlement_days, lot_size, price_tick,
                limit_pct, stamp_tax_rate, benchmark_code, management_fee, updated_at
            )
            SELECT code, name, is_st, instrument_type, etf_category, exchange,
                   list_date, delist_date, settlement_days, lot_size, price_tick,
                   limit_pct, stamp_tax_rate, benchmark_code, management_fee, updated_at
            FROM _tmp_inst
            ON CONFLICT (code) DO UPDATE SET
                name = EXCLUDED.name,
                is_st = EXCLUDED.is_st,
                instrument_type = EXCLUDED.instrument_type,
                etf_category = EXCLUDED.etf_category,
                exchange = EXCLUDED.exchange,
                list_date = EXCLUDED.list_date,
                delist_date = EXCLUDED.delist_date,
                settlement_days = EXCLUDED.settlement_days,
                lot_size = EXCLUDED.lot_size,
                price_tick = EXCLUDED.price_tick,
                limit_pct = EXCLUDED.limit_pct,
                stamp_tax_rate = EXCLUDED.stamp_tax_rate,
                benchmark_code = EXCLUDED.benchmark_code,
                management_fee = EXCLUDED.management_fee,
                updated_at = EXCLUDED.updated_at
            """
        )
        con.unregister("_tmp_inst")
        return len(df)

    def load_instruments(self, codes: list[str] | None = None) -> pd.DataFrame:
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            return self.conn.execute(
                f"SELECT * FROM instruments WHERE code IN ({placeholders})", codes
            ).fetch_df()
        return self.conn.execute("SELECT * FROM instruments").fetch_df()

    def load_etf_codes(self, category: str | None = None) -> list[str]:
        sql = "SELECT code FROM instruments WHERE instrument_type = 'etf'"
        params: list = []
        if category:
            sql += " AND etf_category = ?"
            params.append(category)
        sql += " ORDER BY code"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def upsert_market_caps(self, df: pd.DataFrame) -> int:
        """df 需含 code / total_cap / float_cap 列。"""
        if df is None or df.empty:
            return 0
        df = df[["code", "total_cap", "float_cap"]].copy()
        df["updated_at"] = pd.Timestamp.now()
        con = self.conn
        con.register("_tmp_mc", df)
        con.execute(
            """
            INSERT INTO instruments (code, total_cap, float_cap, updated_at)
            SELECT code, total_cap, float_cap, updated_at FROM _tmp_mc
            ON CONFLICT (code) DO UPDATE SET
                total_cap = EXCLUDED.total_cap,
                float_cap = EXCLUDED.float_cap,
                updated_at = EXCLUDED.updated_at
            """
        )
        con.unregister("_tmp_mc")
        return len(df)

    # ---------- EPS TTM (估值因子) ----------

    def upsert_eps_ttm(self, df: pd.DataFrame) -> int:
        """df 需含 code / pub_date / eps_ttm / total_share / liqa_share。"""
        if df is None or df.empty:
            return 0
        con = self.conn
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS eps_ttm (
                code TEXT NOT NULL,
                pub_date DATE NOT NULL,
                stat_date DATE NOT NULL,
                eps_ttm DOUBLE,
                total_share DOUBLE,
                liqa_share DOUBLE,
                PRIMARY KEY (code, stat_date)
            )
            """
        )
        df = df.copy()
        for col in ["pub_date", "stat_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        con.register("_tmp_eps", df)
        con.execute(
            """
            INSERT OR REPLACE INTO eps_ttm
            SELECT code, pub_date, stat_date, eps_ttm, total_share, liqa_share
            FROM _tmp_eps
            """
        )
        con.unregister("_tmp_eps")
        return len(df)

    def load_latest_eps(self, codes: list[str]) -> dict[str, float]:
        """返回 {code: eps_ttm}，取每个 code 的最新一条记录。"""
        if not codes:
            return {}
        placeholders = ",".join(["?"] * len(codes))
        rows = self.conn.execute(
            f"""
            SELECT code, eps_ttm FROM (
                SELECT code, eps_ttm,
                       ROW_NUMBER() OVER (PARTITION BY code ORDER BY stat_date DESC) AS rn
                FROM eps_ttm
                WHERE code IN ({placeholders})
            ) WHERE rn = 1
            """,
            codes,
        ).fetchall()
        return {r[0]: r[1] for r in rows if r[1] is not None and r[1] > 0}

    def load_market_caps(self, codes: list[str] | None = None) -> dict[str, dict]:
        """返回 {code: {total_cap, float_cap}}，市值未填的标的不会出现在结果里。"""
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            rows = self.conn.execute(
                f"SELECT code, total_cap, float_cap FROM instruments "
                f"WHERE code IN ({placeholders}) AND float_cap IS NOT NULL",
                codes,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT code, total_cap, float_cap FROM instruments WHERE float_cap IS NOT NULL"
            ).fetchall()
        return {r[0]: {"total_cap": r[1], "float_cap": r[2]} for r in rows}

    # ---------- calendar ----------

    def upsert_calendar(self, dates: list[date]) -> int:
        if not dates:
            return 0
        df = pd.DataFrame({"trade_date": [pd.to_datetime(d).date() for d in dates]})
        con = self.conn
        con.register("_tmp_cal", df)
        con.execute("INSERT OR REPLACE INTO trade_calendar SELECT trade_date FROM _tmp_cal")
        con.unregister("_tmp_cal")
        return len(df)

    def load_calendar(
        self, start: date | str | None = None, end: date | str | None = None
    ) -> list[date]:
        sql = "SELECT trade_date FROM trade_calendar WHERE 1=1"
        params: list = []
        if start:
            sql += " AND trade_date >= ?"
            params.append(pd.to_datetime(start).date())
        if end:
            sql += " AND trade_date <= ?"
            params.append(pd.to_datetime(end).date())
        sql += " ORDER BY trade_date"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    # ---------- index constituents ----------

    def upsert_index_constituents(self, index_code: str, codes: list[str]) -> int:
        if not codes:
            return 0
        df = pd.DataFrame(
            {
                "index_code": index_code,
                "code": codes,
                "updated_at": pd.Timestamp.now(),
            }
        )
        con = self.conn
        con.execute("DELETE FROM index_constituents WHERE index_code = ?", [index_code])
        con.register("_tmp_ic", df)
        con.execute(
            "INSERT INTO index_constituents SELECT index_code, code, updated_at FROM _tmp_ic"
        )
        con.unregister("_tmp_ic")
        return len(df)

    def load_index_constituents(self, index_code: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT code FROM index_constituents WHERE index_code = ? ORDER BY code",
            [index_code],
        ).fetchall()
        return [r[0] for r in rows]

    def load_main_board_codes(self) -> list[str]:
        """从 instruments 中取非 ST 且代码符合沪深主板规则的标的。"""
        df = self.conn.execute(
            "SELECT code FROM instruments WHERE COALESCE(is_st, FALSE) = FALSE ORDER BY code"
        ).fetch_df()
        if df.empty:
            return []
        out: list[str] = []
        for raw in df["code"].astype(str):
            c = raw.strip().zfill(6)
            if is_cn_main_board(c):
                out.append(c)
        return sorted(set(out))

    def stats(self) -> dict:
        con = self.conn
        n_bars = con.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
        n_codes = con.execute("SELECT COUNT(DISTINCT code) FROM daily_bars").fetchone()[0]
        n_inst = con.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
        n_cal = con.execute("SELECT COUNT(*) FROM trade_calendar").fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM daily_bars"
        ).fetchone()
        return {
            "bars": n_bars,
            "codes": n_codes,
            "instruments": n_inst,
            "calendar_days": n_cal,
            "date_min": date_range[0],
            "date_max": date_range[1],
        }


@contextmanager
def open_store(path: str | Path | None = None) -> Iterator[DuckStore]:
    store = DuckStore(path)
    try:
        yield store
    finally:
        store.close()


__all__ = ["DuckStore", "open_store", "DAILY_COLUMNS", "INSTRUMENT_COLUMNS"]
