"""因子函数库的正确性/形状测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qkquant.factors.library import FACTOR_REGISTRY, list_factor_names
from qkquant.factors.pipeline import (
    build_tradeable_mask,
    compute_factor_values,
    compute_forward_returns,
    load_panel,
)


def test_registry_has_expected_factors():
    names = set(list_factor_names())
    expected = {
        "mom_20d", "mom_60d", "mom_120d", "reversal_5d",
        "vol_20d", "turnover_20d", "amihud_20d", "rsi_14",
    }
    assert expected.issubset(names), f"missing factors: {expected - names}"


def test_load_panel_shapes(tmp_store):
    panel = load_panel(tmp_store)
    assert "close" in panel and "volume" in panel and "pct_chg" in panel
    close = panel["close"]
    assert close.shape[0] >= 200
    assert close.shape[1] == 5  # 5 只合成票


def test_factor_shape_matches_close(tmp_store):
    panel = load_panel(tmp_store)
    close = panel["close"]
    for name in list_factor_names():
        f = compute_factor_values(panel, name)
        assert f.shape == close.shape, f"{name} shape mismatch"
        assert (f.index == close.index).all(), f"{name} index mismatch"
        assert (f.columns == close.columns).all(), f"{name} columns mismatch"


def test_mom_20d_matches_manual(tmp_store):
    panel = load_panel(tmp_store)
    f = compute_factor_values(panel, "mom_20d")
    # 手工比对某一只股票某个日期：应等于 close.pct_change(20)
    code = panel["close"].columns[0]
    expected = panel["close"][code].pct_change(20)
    pd.testing.assert_series_equal(f[code], expected, check_names=False)


def test_rsi_range(tmp_store):
    panel = load_panel(tmp_store)
    f = compute_factor_values(panel, "rsi_14")
    valid = f.dropna(how="all")
    v = valid.values
    v = v[~np.isnan(v)]
    assert (v >= 0).all() and (v <= 100).all(), "RSI 应在 [0, 100]"


def test_forward_returns_shape_and_alignment(tmp_store):
    panel = load_panel(tmp_store)
    close = panel["close"]
    fwd = compute_forward_returns(close, horizons=(1, 5, 20))
    assert set(fwd.keys()) == {1, 5, 20}
    for h, df in fwd.items():
        assert df.shape == close.shape
        # 最后 h 行应全是 NaN
        assert df.iloc[-h:].isna().all().all(), f"last {h} rows of fwd[{h}] should be NaN"


def test_forward_return_values(tmp_store):
    panel = load_panel(tmp_store)
    close = panel["close"]
    fwd = compute_forward_returns(close, horizons=(5,))[5]
    code = close.columns[0]
    # fwd.iloc[t] = close.iloc[t+5] / close.iloc[t] - 1
    idx = 50
    expected = close[code].iloc[idx + 5] / close[code].iloc[idx] - 1
    assert abs(float(fwd[code].iloc[idx]) - expected) < 1e-12


def test_tradeable_mask_excludes_limit(tmp_store):
    panel = load_panel(tmp_store)
    mask = build_tradeable_mask(panel, limit_pct=9.8)
    # 合成数据波动不会真到 ±10%，mask 应几乎全 True
    assert mask.sum().sum() > 0
    # 人为注入一个涨停
    panel["pct_chg"].iloc[10, 0] = 10.5
    mask2 = build_tradeable_mask(panel, limit_pct=9.8)
    assert not mask2.iloc[10, 0]


def test_all_factors_have_expected_sign():
    for spec in FACTOR_REGISTRY.values():
        assert spec.expected_sign in (+1, -1), f"{spec.name} sign invalid"
