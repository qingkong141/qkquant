from __future__ import annotations

import pandas as pd

from qkquant.etf_signal import EtfSignalConfig, build_etf_plan, estimated_cost, theme_key


def test_theme_key_deduplicates_provider_variants():
    assert theme_key("证券ETF鹏华") == theme_key("证券ETF天弘") == "证券"
    assert theme_key("任意ETF", "000300") == "000300"


def test_estimated_cost_includes_minimum_commission_and_slippage():
    cfg = EtfSignalConfig(commission_rate=0.00025, commission_min=5, slippage_pct=0.002)
    assert estimated_cost(1_000, 0, cfg) == 7
    assert estimated_cost(100_000, 0, cfg) == 225


def test_default_broker_commission_is_half_per_ten_thousand():
    cfg = EtfSignalConfig(slippage_pct=0)
    assert cfg.commission_rate == 0.00005
    assert cfg.commission_min == 0.2
    assert estimated_cost(1_000, 0, cfg) == 0.2
    assert estimated_cost(100_000, 0, cfg) == 5.0


def test_default_chan_entry_is_third_buy_only():
    assert EtfSignalConfig().chan_entry_states == ("third_buy",)


def test_build_etf_plan_has_risk_and_execution_fields(tmp_store):
    codes = tmp_store.load_index_constituents("000300")
    tmp_store.upsert_instruments(
        pd.DataFrame(
            {
                "code": codes,
                "name": [f"主题{i}ETF测试" for i in range(len(codes))],
                "instrument_type": "etf",
                "etf_category": "equity",
            }
        )
    )
    cfg = EtfSignalConfig(
        min_cross_section=5,
        score_threshold=0,
        min_amount_20d=0,
        corr_limit=1,
        risk_on_breadth=0,
        max_positions=3,
        chan_filter_enabled=False,
    )
    plan = build_etf_plan(tmp_store, config=cfg)
    assert plan["execution_session"] == "next_trading_day_open"
    assert plan["regime"] == "risk_on"
    assert 0 <= plan["market_breadth"] <= 1
    assert all(row["action"] in {"BUY", "SELL", "HOLD"} for row in plan["actions"])
