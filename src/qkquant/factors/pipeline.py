"""因子面板构建 + 因子值与 forward returns 计算。

核心数据结构：
- Panel: dict[field_name, DataFrame(date × code)]，宽表形态，方便截面运算。
- forward_returns: dict[horizon, DataFrame(date × code)]，day t 处的值 = [t, t+N] 的收益率。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.factors.library import FACTOR_REGISTRY, Panel


# 长表字段 → 宽表字段名（大部分同名）
PANEL_FIELDS = ("open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover")


def load_panel(
    store: DuckStore,
    codes: list[str] | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    adjust: str = "hfq",
    fields: tuple[str, ...] = PANEL_FIELDS,
) -> Panel:
    """从 DuckDB 读取长表日线，转成 {field: DataFrame(date × code)} 的宽表 dict。

    日期索引经过排序；缺失的 (date, code) 组合保留为 NaN。
    """
    df = store.load_daily(codes=codes, start=start, end=end, adjust=adjust)
    if df.empty:
        raise RuntimeError(
            f"no daily bars in [{start}, {end}] for {len(codes) if codes else 'all'} codes"
        )
    df = df.sort_values(["trade_date", "code"]).copy()

    panel: Panel = {}
    for f in fields:
        if f not in df.columns:
            continue
        wide = df.pivot(index="trade_date", columns="code", values=f)
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        panel[f] = wide
    return panel


def compute_factor_values(panel: Panel, factor_name: str) -> pd.DataFrame:
    """计算单个因子。"""
    spec = FACTOR_REGISTRY[factor_name]
    missing = [f for f in spec.required_fields if f not in panel]
    if missing:
        raise ValueError(f"factor {factor_name} requires {missing} not in panel")
    return spec.func(panel)


def compute_forward_returns(
    close: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[int, pd.DataFrame]:
    """计算 N-day forward return：day t 处 = close[t+N] / close[t] - 1。

    返回 {N: DataFrame(date × code)}。最后 N 行会是 NaN。
    """
    out: dict[int, pd.DataFrame] = {}
    for n in horizons:
        out[n] = close.pct_change(n, fill_method=None).shift(-n)
    return out


def align_valid(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    tradeable_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 factor 和 forward return 对齐，保留两边都有值的位置。

    可选 tradeable_mask（date × code, bool）用于排除不可交易（停牌/涨跌停）的样本。
    """
    common_idx = factor.index.intersection(forward.index)
    common_cols = factor.columns.intersection(forward.columns)
    f = factor.loc[common_idx, common_cols]
    r = forward.loc[common_idx, common_cols]

    valid = f.notna() & r.notna()
    if tradeable_mask is not None:
        tm = tradeable_mask.reindex(index=common_idx, columns=common_cols).fillna(False)
        valid = valid & tm

    f = f.where(valid)
    r = r.where(valid)
    return f, r


def build_tradeable_mask(panel: Panel, limit_pct: float = 9.8) -> pd.DataFrame:
    """构建"次日可交易"mask：排除今日涨停/跌停/停牌。

    - 停牌：pct_chg 或 close 为 NaN
    - 涨跌停：|pct_chg| >= limit_pct
    返回 DataFrame(date × code, bool)，True = 因子信号在当日可转化为持仓。
    """
    close = panel["close"]
    if "pct_chg" in panel:
        pct = panel["pct_chg"]
        tradeable = close.notna() & pct.notna() & (pct.abs() < limit_pct)
    else:
        tradeable = close.notna()
    return tradeable


__all__ = [
    "PANEL_FIELDS",
    "align_valid",
    "build_tradeable_mask",
    "compute_factor_values",
    "compute_forward_returns",
    "load_panel",
]
