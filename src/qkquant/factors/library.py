"""单因子函数库。

每个因子函数接受一个 `Panel`（dict of wide DataFrame: date × code）,
返回一个 wide DataFrame（date × code）的因子值。

约定：
- 因子值在同一截面（同一行）内可比即可，绝对尺度不重要（IC 测试用 rank）
- NaN 表示"该股当天无信号"（数据不足 / 非交易日）
- 每个因子附带 `expected_sign`：+1 表示"因子越大，未来收益越高"；-1 相反。
  用于 summary 判断方向是否符合预期（实际方向由 IC 决定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Panel: {field_name: DataFrame(index=date, columns=code)}
Panel = dict[str, pd.DataFrame]


@dataclass
class FactorSpec:
    name: str
    func: Callable[[Panel], pd.DataFrame]
    expected_sign: int          # +1 / -1，用于 summary 方向判断
    description: str
    required_fields: tuple[str, ...] = ("close",)


# ---------------- 因子实现 ----------------


def _pct_change(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """过去 window 日累计收益率。close.pct_change(window) 会自动对齐。"""
    return close.pct_change(window, fill_method=None)


def mom_20d(panel: Panel) -> pd.DataFrame:
    return _pct_change(panel["close"], 20)


def mom_60d(panel: Panel) -> pd.DataFrame:
    return _pct_change(panel["close"], 60)


def mom_120d(panel: Panel) -> pd.DataFrame:
    return _pct_change(panel["close"], 120)


def reversal_5d(panel: Panel) -> pd.DataFrame:
    """过去 5 日收益。预期为反转因子：IC 应为负。"""
    return _pct_change(panel["close"], 5)


def vol_20d(panel: Panel) -> pd.DataFrame:
    """过去 20 日日收益率的标准差。预期：低波动溢价，IC 为负。"""
    ret = panel["close"].pct_change(fill_method=None)
    return ret.rolling(window=20, min_periods=20).std()


def turnover_20d(panel: Panel) -> pd.DataFrame:
    """过去 20 日平均换手率。预期：低换手溢价，IC 为负。"""
    return panel["turnover"].rolling(window=20, min_periods=20).mean()


def amihud_20d(panel: Panel) -> pd.DataFrame:
    """Amihud 非流动性：mean(|pct_chg| / amount)。
    高值=交易成本高（流动性差）=被忽视股=预期溢价，IC 为正。
    amount 单位在 akshare/baostock 里是元，乘 1e8 归一化到"亿元"避免除出超小数。
    """
    pct_abs = panel["pct_chg"].abs()
    amount_yi = panel["amount"] / 1e8
    ratio = pct_abs / amount_yi.replace(0, np.nan)
    return ratio.rolling(window=20, min_periods=20).mean()


def rsi_14(panel: Panel) -> pd.DataFrame:
    """14 日 RSI（Wilder 平滑）。预期：高 RSI 超买反转，IC 为负。"""
    close = panel["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out


# ---------------- 注册表 ----------------


FACTOR_REGISTRY: dict[str, FactorSpec] = {
    "mom_20d": FactorSpec(
        name="mom_20d",
        func=mom_20d,
        expected_sign=+1,
        description="过去 20 日累计收益（短期动量）",
    ),
    "mom_60d": FactorSpec(
        name="mom_60d",
        func=mom_60d,
        expected_sign=+1,
        description="过去 60 日累计收益（中期动量）",
    ),
    "mom_120d": FactorSpec(
        name="mom_120d",
        func=mom_120d,
        expected_sign=+1,
        description="过去 120 日累计收益（长期动量）",
    ),
    "reversal_5d": FactorSpec(
        name="reversal_5d",
        func=reversal_5d,
        expected_sign=-1,
        description="过去 5 日累计收益（短期反转）",
    ),
    "vol_20d": FactorSpec(
        name="vol_20d",
        func=vol_20d,
        expected_sign=-1,
        description="过去 20 日波动率（低波异象）",
    ),
    "turnover_20d": FactorSpec(
        name="turnover_20d",
        func=turnover_20d,
        expected_sign=-1,
        description="过去 20 日平均换手率（低关注度溢价）",
        required_fields=("turnover",),
    ),
    "amihud_20d": FactorSpec(
        name="amihud_20d",
        func=amihud_20d,
        expected_sign=+1,
        description="Amihud 非流动性（流动性溢价）",
        required_fields=("pct_chg", "amount"),
    ),
    "rsi_14": FactorSpec(
        name="rsi_14",
        func=rsi_14,
        expected_sign=-1,
        description="14 日 RSI（超买超卖反转）",
    ),
}


def get_factor(name: str) -> FactorSpec:
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"unknown factor: {name}. Available: {list(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


def list_factor_names() -> list[str]:
    return list(FACTOR_REGISTRY.keys())


__all__ = [
    "FACTOR_REGISTRY",
    "FactorSpec",
    "Panel",
    "get_factor",
    "list_factor_names",
    # 单因子函数直接暴露,方便外部直接调用
    "mom_20d",
    "mom_60d",
    "mom_120d",
    "reversal_5d",
    "vol_20d",
    "turnover_20d",
    "amihud_20d",
    "rsi_14",
]
