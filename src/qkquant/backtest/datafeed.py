"""把 DuckDB 日线转成 backtrader 数据 feed。

扩展 PandasData，额外携带 pct_chg 列，
供策略在回测中判断涨停/跌停。
"""

from __future__ import annotations

from datetime import date

import backtrader as bt
import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.logger import logger


class PandasDataExt(bt.feeds.PandasData):
    """扩展 PandasData，增加 pct_chg 字段。"""

    lines = ("pct_chg",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("pct_chg", "pct_chg"),
    )


def build_feeds(
    store: DuckStore,
    codes: list[str],
    start: date | str,
    end: date | str,
    adjust: str = "qfq",
    min_bars: int = 30,
) -> list[tuple[str, PandasDataExt]]:
    """为每只股票构造一个 bt feed；数据过少的股票会被跳过。"""
    df = store.load_daily(codes=codes, start=start, end=end, adjust=adjust)
    if df.empty:
        logger.warning(
            f"no data loaded for codes={len(codes)} in [{start}, {end}]; did you run update-data?"
        )
        return []

    feeds: list[tuple[str, PandasDataExt]] = []
    for code, group in df.groupby("code"):
        g = group.sort_values("trade_date").copy()
        if len(g) < min_bars:
            logger.debug(f"skip {code}: only {len(g)} bars < {min_bars}")
            continue
        g = g.set_index("trade_date")
        g.index = pd.to_datetime(g.index)
        g = g[["open", "high", "low", "close", "volume", "pct_chg"]].astype(float)
        g["pct_chg"] = g["pct_chg"].fillna(0.0)
        feed = PandasDataExt(dataname=g, name=code, plot=False)
        feeds.append((code, feed))

    logger.info(f"built {len(feeds)} feeds from DuckDB")
    return feeds


__all__ = ["PandasDataExt", "build_feeds"]
