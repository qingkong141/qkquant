"""IC 测试与分组回测。

两个核心函数：
- compute_ic_stats: Spearman Rank IC 及其统计量
- compute_quintile_returns: staggered-portfolio 分组回测

输入都是 align_valid 后的 (factor, forward_returns) 宽表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- IC ----------------


def _row_rank_ic(f_row: pd.Series, r_row: pd.Series, min_n: int = 20) -> float:
    """单截面 Spearman 相关系数。"""
    mask = f_row.notna() & r_row.notna()
    if mask.sum() < min_n:
        return np.nan
    f = f_row[mask]
    r = r_row[mask]
    # 用 pandas rank 再 pearson 等价于 spearman，对 ties 用 average
    fr = f.rank()
    rr = r.rank()
    # 去均值除以 std 再点乘
    fr_c = fr - fr.mean()
    rr_c = rr - rr.mean()
    denom = fr_c.std(ddof=0) * rr_c.std(ddof=0) * len(fr)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((fr_c * rr_c).sum() / denom)


def compute_ic_series(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_cross_section: int = 20,
) -> pd.Series:
    """逐日 Rank IC 序列（index=date, value=IC）。"""
    # vectorized Spearman: rank along rows, then correlate row-by-row
    f_rank = factor.rank(axis=1)
    r_rank = forward_returns.rank(axis=1)

    valid = factor.notna() & forward_returns.notna()
    n = valid.sum(axis=1)

    # 向量化 pearson correlation on ranks
    f_mean = f_rank.mean(axis=1)
    r_mean = r_rank.mean(axis=1)
    f_dev = f_rank.sub(f_mean, axis=0)
    r_dev = r_rank.sub(r_mean, axis=0)

    numer = (f_dev * r_dev).sum(axis=1, min_count=1)
    denom = np.sqrt(
        (f_dev.pow(2)).sum(axis=1, min_count=1) * (r_dev.pow(2)).sum(axis=1, min_count=1)
    )
    ic = numer / denom.replace(0, np.nan)
    ic = ic.where(n >= min_cross_section)
    ic.name = "ic"
    return ic


def summarize_ic(ic: pd.Series) -> dict:
    """IC 序列的统计量。"""
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "t_stat": np.nan,
            "positive_ratio": np.nan,
            "n_samples": 0,
        }
    mean = float(ic_clean.mean())
    std = float(ic_clean.std(ddof=1))
    n = int(len(ic_clean))
    ir = mean / std if std > 0 else np.nan
    # t-stat: 假设 IC 序列近似 iid，t = mean / (std / sqrt(n))
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    positive = float((ic_clean > 0).mean())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": float(ir) if np.isfinite(ir) else np.nan,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "positive_ratio": positive,
        "n_samples": n,
    }


# ---------------- 分组回测 ----------------


def assign_quintiles(factor: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """把每行（每个截面）的因子值映射到 0..n_groups-1 的分组标签。

    NaN 保留为 NaN。截面内有效样本数不足 n_groups 时整行为 NaN。
    """
    # 每行 rank，pct=True 给出 (0, 1] 的百分位
    ranks = factor.rank(axis=1, pct=True, method="average")
    # 映射到 [0, n_groups-1]。处理 pct=1.0 的边界：clip
    bins = np.ceil(ranks * n_groups).clip(lower=1, upper=n_groups) - 1
    # 截面有效数不足时整行设 NaN
    counts = factor.notna().sum(axis=1)
    bins = bins.where(counts.ge(n_groups), np.nan)
    return bins


def compute_quintile_daily_returns(
    factor: pd.DataFrame,
    daily_returns: pd.DataFrame,
    holding_period: int = 5,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Staggered-portfolio 分组日度收益。

    在每个"形成日" s 对因子排序得到 n_groups 组，每组等权持有 holding_period 日。
    任意日 t 的 q 组日度收益 = 在 [t-holding_period, t-1] 形成并仍有效的各子组合当日
    等权平均收益 的再平均（等价于 1/holding_period × 各子组合日收益之和）。

    Args:
        factor: date × code，因子值（NaN = 不进任何组）
        daily_returns: date × code，每日收益率（常用 close.pct_change()，day t 处 = t-1→t）
        holding_period: 持有期 N
        n_groups: 分组数

    Returns:
        DataFrame(index=date, columns=0..n_groups-1)，各组日度收益。
    """
    # 对齐
    common_idx = factor.index.intersection(daily_returns.index)
    common_cols = factor.columns.intersection(daily_returns.columns)
    factor = factor.loc[common_idx, common_cols]
    daily_returns = daily_returns.loc[common_idx, common_cols]

    quintile = assign_quintiles(factor, n_groups=n_groups)

    # 每个组每个形成日的组内股票数，用于等权归一
    # indicator_q[s, c] = 1 if quintile[s, c] == q
    result = pd.DataFrame(0.0, index=common_idx, columns=range(n_groups))

    for q in range(n_groups):
        ind = (quintile == q).astype(float)  # date × code
        size_q = ind.sum(axis=1)              # date
        # 等权子组合：每只股票权重 = 1 / size_q。size_q=0 的日期整行权重为 0
        weight = ind.div(size_q.replace(0, np.nan), axis=0).fillna(0.0)
        # 子组合形成于 s，贡献 day t = s+1..s+holding_period。
        # 等价：active_weight[t, c] = sum_{k=1..N} weight[t-k, c]，再 /N 得平均。
        active = weight.shift(1).rolling(window=holding_period, min_periods=1).sum() / holding_period
        port_ret = (active * daily_returns).sum(axis=1, min_count=1)
        result[q] = port_ret

    result.columns = [f"q{i + 1}" for i in range(n_groups)]
    return result


def summarize_quintile(
    quintile_daily: pd.DataFrame,
    annualize_factor: float = 252.0,
) -> pd.DataFrame:
    """分组收益统计：年化收益、年化波动、Sharpe、累计收益。"""
    rows = []
    for col in quintile_daily.columns:
        r = quintile_daily[col].dropna()
        if r.empty:
            rows.append(
                {"group": col, "cum_return": np.nan, "ann_return": np.nan,
                 "ann_vol": np.nan, "sharpe": np.nan, "n_days": 0}
            )
            continue
        cum = float((1 + r).prod() - 1)
        ann = float((1 + r.mean()) ** annualize_factor - 1)
        vol = float(r.std(ddof=1) * np.sqrt(annualize_factor))
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(annualize_factor)) if r.std() > 0 else np.nan
        rows.append({
            "group": col,
            "cum_return": cum,
            "ann_return": ann,
            "ann_vol": vol,
            "sharpe": sharpe,
            "n_days": int(len(r)),
        })
    return pd.DataFrame(rows)


def long_short_series(quintile_daily: pd.DataFrame) -> pd.Series:
    """多空组合日收益：Q_top - Q_bottom。"""
    top = quintile_daily.columns[-1]
    bot = quintile_daily.columns[0]
    s = quintile_daily[top] - quintile_daily[bot]
    s.name = "long_short"
    return s


__all__ = [
    "assign_quintiles",
    "compute_ic_series",
    "compute_quintile_daily_returns",
    "long_short_series",
    "summarize_ic",
    "summarize_quintile",
]
