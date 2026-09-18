from __future__ import annotations

import pandas as pd
import pytest

from qkquant.etf_portfolio_backtest import _attribution, _metrics


def test_portfolio_metrics_for_monotonic_equity():
    equity = pd.Series([100.0, 101.0, 102.0, 104.0], index=pd.date_range("2025-01-01", periods=4))
    result = _metrics(equity)
    assert result["total_return"] == pytest.approx(0.04)
    assert result["max_drawdown"] == 0
    assert result["sharpe"] > 0


def test_portfolio_metrics_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0], index=pd.date_range("2025-01-01", periods=3))
    assert _metrics(equity)["max_drawdown"] == -0.25


def test_attribution_reports_year_and_exposure():
    index = pd.to_datetime(["2024-01-02", "2024-12-31", "2025-12-31"])
    equity = pd.Series([100.0, 110.0, 121.0], index=index)
    exposure = pd.Series([0.0, 0.5, 1.0], index=index)
    result = _attribution(equity, exposure, [])
    assert result["average_exposure"] == 0.5
    assert result["cash_day_ratio"] == 1 / 3
    assert result["yearly_returns"]["2025"] == pytest.approx(0.1)
