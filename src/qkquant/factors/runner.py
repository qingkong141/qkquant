"""因子评估主流程：读数据 → 算因子 → IC + 分组 → 出报告。

被 CLI 和测试共用。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from qkquant.data.storage import DuckStore
from qkquant.factors.ic import (
    compute_ic_series,
    compute_quintile_daily_returns,
    long_short_series,
    summarize_ic,
    summarize_quintile,
)
from qkquant.factors.library import FACTOR_REGISTRY, FactorSpec
from qkquant.factors.pipeline import (
    align_valid,
    build_tradeable_mask,
    compute_factor_values,
    compute_forward_returns,
    load_panel,
)
from qkquant.factors.report import (
    generate_factor_comparison_report,
    generate_factor_report,
)
from qkquant.logger import logger

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)


def evaluate_factor(
    store: DuckStore,
    factor_name: str,
    codes: list[str],
    start: date | str,
    end: date | str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int = 5,
    n_groups: int = 5,
    write_report: bool = True,
    report_root: Path | None = None,
) -> dict:
    """评估单个因子，返回聚合结果 dict。"""
    spec: FactorSpec = FACTOR_REGISTRY[factor_name]
    logger.info(f"evaluate factor {factor_name}: loading panel ...")
    panel = load_panel(store, codes=codes, start=start, end=end)
    close = panel["close"]

    factor = compute_factor_values(panel, factor_name)
    tradeable = build_tradeable_mask(panel)
    fwd = compute_forward_returns(close, horizons=horizons)

    ic_stats_by_horizon: dict[int, dict] = {}
    ic_series_by_horizon: dict[int, pd.Series] = {}
    for h in horizons:
        f_aligned, r_aligned = align_valid(factor, fwd[h], tradeable_mask=tradeable)
        ic_series = compute_ic_series(f_aligned, r_aligned)
        ic_series_by_horizon[h] = ic_series
        ic_stats_by_horizon[h] = summarize_ic(ic_series)

    # 分组回测用 primary horizon
    f_aligned, _ = align_valid(factor, fwd[primary_horizon], tradeable_mask=tradeable)
    daily_ret = close.pct_change(fill_method=None)
    quintile_daily = compute_quintile_daily_returns(
        f_aligned, daily_ret, holding_period=primary_horizon, n_groups=n_groups
    )
    quintile_stats = summarize_quintile(quintile_daily)
    ls = long_short_series(quintile_daily)

    # 若因子"预期方向"为 -1（小值好），把多空方向反过来使多空曲线更直观
    if spec.expected_sign < 0:
        ls = -ls
        ls.name = "long_short_neg"

    result = {
        "spec": spec,
        "period": (str(start), str(end)),
        "primary_horizon": primary_horizon,
        "ic_stats_by_horizon": ic_stats_by_horizon,
        "ic_series_by_horizon": ic_series_by_horizon,
        "quintile_daily": quintile_daily,
        "quintile_stats": quintile_stats,
        "long_short": ls,
    }

    if write_report:
        out = generate_factor_report(
            spec=spec,
            ic_stats_by_horizon=ic_stats_by_horizon,
            ic_series_by_horizon=ic_series_by_horizon,
            primary_horizon=primary_horizon,
            quintile_daily=quintile_daily,
            quintile_stats=quintile_stats,
            long_short=ls,
            period=(str(start), str(end)),
            report_root=report_root,
        )
        result["report_dir"] = out

    return result


def evaluate_all_factors(
    store: DuckStore,
    codes: list[str],
    start: date | str,
    end: date | str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int = 5,
    factor_names: list[str] | None = None,
    report_root: Path | None = None,
) -> Path:
    """批量评估所有因子，产出横向对比报告并返回目录。"""
    names = factor_names or list(FACTOR_REGISTRY.keys())
    results: dict[str, dict] = {}
    for name in names:
        logger.info(f"--- factor {name} ---")
        results[name] = evaluate_factor(
            store=store,
            factor_name=name,
            codes=codes,
            start=start,
            end=end,
            horizons=horizons,
            primary_horizon=primary_horizon,
            write_report=False,
        )
    return generate_factor_comparison_report(results, report_root=report_root)


__all__ = [
    "DEFAULT_HORIZONS",
    "evaluate_all_factors",
    "evaluate_factor",
]
