"""IC 与分组回测的正确性测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qkquant.factors.ic import (
    assign_quintiles,
    compute_ic_series,
    compute_quintile_daily_returns,
    long_short_series,
    summarize_ic,
    summarize_quintile,
)


def _make_panel(n_days: int = 60, n_codes: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-03", periods=n_days, freq="B")
    codes = [f"c{i:03d}" for i in range(n_codes)]
    factor = pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=dates, columns=codes)
    return factor, dates, codes


def test_perfect_ic_is_one():
    """如果 forward_return == factor，IC 应该 == 1。"""
    factor, _, _ = _make_panel()
    forward = factor.copy()
    ic = compute_ic_series(factor, forward)
    # 第一行可能 NaN（截面为 factor 自身，rank 完全一致），Spearman = 1
    assert ic.dropna().mean() > 0.99


def test_perfect_negative_ic():
    factor, _, _ = _make_panel()
    forward = -factor
    ic = compute_ic_series(factor, forward)
    assert ic.dropna().mean() < -0.99


def test_zero_ic_under_independence():
    """因子与 forward return 独立 → IC 应接近 0。"""
    factor, _, _ = _make_panel(n_days=500, n_codes=100, seed=1)
    rng = np.random.default_rng(42)
    forward = pd.DataFrame(rng.normal(size=factor.shape), index=factor.index, columns=factor.columns)
    ic = compute_ic_series(factor, forward)
    stats = summarize_ic(ic)
    assert abs(stats["ic_mean"]) < 0.05
    # t-stat 不显著
    assert abs(stats["t_stat"]) < 3


def test_summarize_ic_fields():
    factor, _, _ = _make_panel()
    ic = compute_ic_series(factor, factor)
    s = summarize_ic(ic)
    assert set(s.keys()) >= {"ic_mean", "ic_std", "ic_ir", "t_stat", "positive_ratio", "n_samples"}


def test_assign_quintiles_distribution():
    factor, dates, codes = _make_panel(n_days=10, n_codes=50)
    q = assign_quintiles(factor, n_groups=5)
    # 每一行每组应该大约 10 只
    for _, row in q.iterrows():
        counts = row.value_counts()
        assert set(counts.index.astype(int)) == {0, 1, 2, 3, 4}
        # 均衡性：每组应在 [8, 12] 之间
        assert counts.min() >= 8
        assert counts.max() <= 12


def test_quintile_sufficient_samples():
    """截面样本不足 n_groups 时整行为 NaN。"""
    factor, _, _ = _make_panel(n_days=5, n_codes=3)
    q = assign_quintiles(factor, n_groups=5)
    assert q.isna().all().all()


def test_quintile_daily_returns_monotonic_when_factor_predicts_return():
    """构造：forward return 直接正比于 factor → 分组收益应单调上升。"""
    factor, dates, codes = _make_panel(n_days=120, n_codes=50, seed=0)
    # 让每日收益 = 0.01 × factor_rank（完美预测）
    rng = np.random.default_rng(2)
    # daily_returns[t, c] = 0.005 * factor[t-1, c] + noise
    noise = pd.DataFrame(
        rng.normal(scale=0.001, size=factor.shape), index=factor.index, columns=factor.columns
    )
    daily_ret = 0.005 * factor.shift(1) + noise

    qd = compute_quintile_daily_returns(factor, daily_ret, holding_period=1, n_groups=5)
    means = qd.mean()
    # 均值应基本单调：q5 > q1
    assert means.iloc[-1] > means.iloc[0], f"means={means.to_dict()}"


def test_long_short_series():
    factor, _, _ = _make_panel()
    qd = compute_quintile_daily_returns(factor, factor.shift(1).fillna(0), holding_period=1)
    ls = long_short_series(qd)
    assert ls.name == "long_short"
    assert len(ls) == len(qd)


def test_summarize_quintile_shape():
    factor, _, _ = _make_panel(n_days=100, n_codes=30)
    qd = compute_quintile_daily_returns(factor, factor.shift(1), holding_period=5)
    stats = summarize_quintile(qd)
    assert set(stats.columns) >= {"group", "cum_return", "ann_return", "ann_vol", "sharpe"}
    assert len(stats) == 5


def test_staggered_holding_period_smoothing():
    """holding_period=5 时单日权重被摊在 5 天内。验证：
    在只有一个非零 formation 日的极简情形下，该组合日收益会出现在未来 5 天内，
    每天的贡献 ≈ 1/5 的单日收益。"""
    dates = pd.date_range("2023-01-03", periods=30, freq="B")
    codes = ["a", "b", "c"]
    factor = pd.DataFrame(np.nan, index=dates, columns=codes)
    factor.iloc[0] = [0.0, 0.5, 1.0]  # 只在第 0 天给出因子值
    daily_ret = pd.DataFrame(0.01, index=dates, columns=codes)  # 固定 1%

    qd = compute_quintile_daily_returns(
        factor, daily_ret, holding_period=5, n_groups=3
    )
    # 第 0 天用作 formation，贡献到 1..5 天
    # 每组只有 1 只股票，权重 = 1；active = 1 / 5 = 0.2；daily_ret = 0.01；组合日收益 = 0.002
    for t in range(1, 6):
        for q in ["q1", "q2", "q3"]:
            assert abs(qd[q].iloc[t] - 0.002) < 1e-9
    # 第 6 天起应回到 0
    for t in range(6, 15):
        for q in ["q1", "q2", "q3"]:
            assert abs(qd[q].iloc[t]) < 1e-9


def test_min_cross_section_filter():
    """截面样本数不足时，IC 应为 NaN。"""
    dates = pd.date_range("2023-01-03", periods=10, freq="B")
    codes = ["a", "b", "c"]
    factor = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 3)), index=dates, columns=codes)
    forward = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 3)), index=dates, columns=codes)
    ic = compute_ic_series(factor, forward, min_cross_section=20)
    assert ic.isna().all()
