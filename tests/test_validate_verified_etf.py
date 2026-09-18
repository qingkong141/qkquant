"""Checks for chronological return attribution in the frozen portfolio report."""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_verified_etf.py"
SPEC = importlib.util.spec_from_file_location("validate_verified_etf", SCRIPT)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_slice_keeps_first_day_pnl_and_inherited_peak():
    equity = pd.Series([120.0, 110.0, 115.0], index=pd.bdate_range("2025-01-01", periods=3))
    metrics = validation.slice_metrics(equity, 100, start="2025-01-02")
    assert metrics["opening_equity"] == 120
    assert metrics["total_return"] == pytest.approx(115 / 120 - 1)
    assert metrics["max_drawdown"] == pytest.approx(110 / 120 - 1)
    assert metrics["lifetime_peak_drawdown_in_slice"] == pytest.approx(110 / 120 - 1)
    # A one-day later slice must still disclose the prior account high.
    later = validation.slice_metrics(equity, 100, start="2025-01-03")
    assert later["max_drawdown"] == 0
    assert later["lifetime_peak_drawdown_in_slice"] == pytest.approx(115 / 120 - 1)


def test_first_record_uses_starting_capital_not_first_close():
    equity = pd.Series([95.0, 98.0], index=pd.bdate_range("2025-01-01", periods=2))
    metrics = validation.slice_metrics(equity, 100)
    assert metrics["total_return"] == pytest.approx(-.02)
    assert metrics["max_drawdown"] == pytest.approx(-.05)
    assert metrics["sessions"] == 2
    assert validation.slice_metrics(equity, 100, start="2026-01-01") == {}


def test_frozen_cost_scenarios_do_not_modify_other_parameters():
    import json
    manifest = {
        "config": json.loads(json.dumps(asdict(validation.EtfSignalConfig()))),
        "drawdown_config": asdict(validation.DrawdownConfig()),
        "portfolio": {"rebalance_days": 10, "rebalance_offset": 0, "daily_entries": False},
        "event_protocol": {"stress_min_commission": 5, "stress_slippage": .004},
    }
    scenarios, risk, portfolio = validation.frozen_configs(manifest)
    base, stress = asdict(scenarios["base"]), asdict(scenarios["stress"])
    assert {k for k in base if base[k] != stress[k]} == {"commission_min", "slippage_pct"}
    assert risk.max_exposure == .4
    assert portfolio["rebalance_days"] == 10
    manifest["config"]["score_threshold"] = .64
    with pytest.raises(ValueError, match="frozen parameters"):
        validation.frozen_configs(manifest)


def test_extension_is_a_slice_of_the_continuous_account():
    dates = pd.bdate_range("2026-07-20", periods=4)
    equity = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)
    result = {"equity": equity, "benchmark_equity": equity,
              "exposure": pd.Series(.4, index=dates), "config": {"capital": 100},
              "trades": [{"date": "2026-07-21", "commission": .2},
                         {"date": "2026-07-22", "commission": .3}]}
    rows = {row["sample"]: row for row in validation.summarize_result(result, "2026-07-22")}
    assert rows["historical_extension"]["opening_equity"] == 110
    assert rows["historical_extension"]["total_return"] == pytest.approx(-.01)
    assert rows["historical_extension"]["max_drawdown"] == pytest.approx(-.1)
    assert rows["historical_extension"]["trade_count"] == 1
    assert rows["historical_extension"]["commission_total"] == .3
    assert ((1 + rows["development"]["total_return"]) * (1 + rows["historical_extension"]["total_return"])) == pytest.approx(1 + rows["full"]["total_return"])
