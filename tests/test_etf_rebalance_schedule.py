from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qkquant import etf_portfolio_backtest as module
from qkquant.etf_signal import EtfSignalConfig


@pytest.mark.parametrize("days,offset", [(0, 0), (5, -1), (5, 5)])
def test_invalid_schedule(days, offset):
    with pytest.raises(ValueError, match="rebalance_days"):
        module.run_etf_portfolio_backtest(None, rebalance_days=days, rebalance_offset=offset)


def test_offset_changes_signal_day_but_execution_remains_next_open(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=145)
    close = pd.DataFrame({"ETF": 10 * np.exp(np.arange(145) * .003 + np.sin(np.arange(145)) * .001)}, index=index)
    panel = {key: close.copy() for key in ("open", "high", "low", "close")}
    panel["amount"] = close * 100_000_000
    monkeypatch.setattr(module, "load_panel", lambda *args, **kwargs: panel)
    store = SimpleNamespace(
        load_etf_codes=lambda category: ["ETF"],
        load_instruments=lambda codes: pd.DataFrame([{"code": "ETF", "name": "ETF", "lot_size": 100}]),
    )
    cfg = EtfSignalConfig(chan_filter_enabled=False, score_threshold=0)
    for offset in (0, 2, 4):
        result = module.run_etf_portfolio_backtest(store, config=cfg, rebalance_offset=offset)
        assert result["trades"][0]["date"] == str(index[121 + offset].date())
        assert result["trades"][0]["side"] == "BUY"
        assert (result["exposure"].iloc[:offset + 1] == 0).all()
