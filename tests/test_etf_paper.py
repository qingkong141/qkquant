import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qkquant.etf_paper import create_paper_account, record_paper_close


@pytest.fixture
def account_data(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=145)
    close = pd.DataFrame({"ETF": 10 + np.arange(145) / 100}, index=dates)
    panel = {key: close.copy() for key in ("open", "high", "low", "close")}
    panel["amount"] = close * 1e8
    # Deliberately different price bases: account valuation MUST use raw.
    raw = {key: value * 2 for key, value in panel.items()}
    instruments = pd.DataFrame([{"code": "ETF", "name": "ETF", "lot_size": 100}])
    data = SimpleNamespace(
        panel=panel, raw_panel=raw, codes=["ETF"], metadata={"database_sha256": "test"},
        store=SimpleNamespace(load_instruments=lambda codes: instruments),
    )
    path = tmp_path / "paper.json"
    create_paper_account(path)
    return path, data


def test_persistent_peak_drawdown_and_raw_valuation(account_data):
    path, data = account_data
    dates = data.panel["close"].index
    first = record_paper_close(path, data, str(dates[120].date()))
    assert first["equity"] == 100_000 and first["exposure_cap"] == .4
    qty = 1_000
    price = float(data.raw_panel["close"].iloc[121, 0])
    second = record_paper_close(path, data, str(dates[121].date()),
                               {"cash": 94_000 - qty * price, "holdings": {"ETF": qty}})
    assert second["equity"] == 94_000
    assert second["drawdown_state"]["peak"] == 100_000
    assert second["exposure_cap"] == .2
    assert second["targets"][0]["reason"] == "risk_reduce"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["records"]) == 2
    assert saved["records"][-1]["drawdown_state"] == second["drawdown_state"]


def test_repeat_date_idempotent_and_conflicting_snapshot_rejected(account_data):
    path, data = account_data
    day = str(data.panel["close"].index[120].date())
    record_paper_close(path, data, day)
    before = path.read_bytes()
    record_paper_close(path, data, day)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwritten"):
        record_paper_close(path, data, day, {"cash": 1, "holdings": {}})
    assert path.read_bytes() == before
    assert not path.with_suffix(".json.lock").exists()


def test_next_close_requires_explicit_snapshot_and_no_skipped_session(account_data):
    path, data = account_data
    dates = data.panel["close"].index
    record_paper_close(path, data, str(dates[120].date()))
    with pytest.raises(ValueError, match="explicit"):
        record_paper_close(path, data, str(dates[121].date()))
    with pytest.raises(ValueError, match="sequence"):
        record_paper_close(path, data, str(dates[122].date()), {"cash": 100_000, "holdings": {}})
    assert len(json.loads(path.read_text())["records"]) == 1


def test_missing_raw_price_cannot_silently_erase_position_value(account_data):
    path, data = account_data
    dates = data.panel["close"].index
    record_paper_close(path, data, str(dates[120].date()))
    data.raw_panel["close"].iloc[121, 0] = np.nan
    with pytest.raises(ValueError, match="cannot be valued"):
        record_paper_close(path, data, str(dates[121].date()), {"cash": 90_000, "holdings": {"ETF": 100}})


def test_account_cannot_be_reset_or_use_another_strategy(account_data):
    path, data = account_data
    with pytest.raises(FileExistsError):
        create_paper_account(path)
    content = json.loads(path.read_text())
    content["risk_config"]["max_exposure"] = 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="fixed strategy"):
        record_paper_close(path, data, str(data.panel["close"].index[120].date()))


def test_writer_lock_prevents_lost_updates(account_data):
    path, data = account_data
    path.with_suffix(".json.lock").write_text("another writer")
    with pytest.raises(FileExistsError):
        record_paper_close(path, data, str(data.panel["close"].index[120].date()))
    assert path.with_suffix(".json.lock").read_text() == "another writer"


def test_lifetime_halt_survives_process_reloads_and_recovery(account_data):
    path, data = account_data
    dates = data.panel["close"].index
    record_paper_close(path, data, str(dates[120].date()))
    stopped = record_paper_close(path, data, str(dates[121].date()), {"cash": 89_000, "holdings": {}})
    assert stopped["drawdown_state"]["halted"]
    recovered = record_paper_close(path, data, str(dates[122].date()), {"cash": 100_000, "holdings": {}})
    assert recovered["drawdown_state"]["halted"] and recovered["exposure_cap"] == 0


def test_reduction_retains_budget_and_separate_quantity_cap(account_data):
    path, data = account_data
    dates = data.panel["close"].index
    record_paper_close(path, data, str(dates[120].date()))
    price = float(data.raw_panel["close"].iloc[121, 0])
    record = record_paper_close(path, data, str(dates[121].date()),
                               {"cash": 95_000 - 1_000 * price, "holdings": {"ETF": 1_000}})
    target = record["targets"][0]
    assert target["target_weight"] == pytest.approx(target["current_weight"])
    assert target["max_qty"] == 1_000
    assert target["allow_increase"] is False


def test_continuing_signal_keeps_budget_but_cannot_top_up_after_open_gap(account_data, monkeypatch):
    from qkquant import etf_portfolio_backtest
    path, data = account_data
    dates = data.panel["close"].index
    monkeypatch.setattr(etf_portfolio_backtest, "classify_chan_structure",
                        lambda *args: SimpleNamespace(state="third_buy_active", allowed=False))
    record_paper_close(path, data, str(dates[120].date()))
    for pos in range(121, 131):
        price = float(data.raw_panel["close"].iloc[pos, 0])
        record = record_paper_close(path, data, str(dates[pos].date()),
                                   {"cash": 100_000 - 100 * price, "holdings": {"ETF": 100}})
    target = record["targets"][0]
    assert target["target_weight"] == .4
    assert target["current_weight"] < .04
    assert target["max_qty"] == 100 and target["allow_increase"] is False


def test_malformed_holdings_rejected_without_mutation(account_data):
    path, data = account_data
    before = path.read_bytes()
    with pytest.raises(ValueError, match="holdings object"):
        record_paper_close(path, data, str(data.panel["close"].index[120].date()),
                           {"cash": 100_000, "holdings": []})
    assert path.read_bytes() == before


def test_source_revision_change_blocks_continuing_account(account_data, monkeypatch):
    from qkquant import etf_paper
    path, data = account_data
    monkeypatch.setattr(etf_paper, "_rule_revision", lambda: {"different": "version"})
    with pytest.raises(ValueError, match="source changed"):
        record_paper_close(path, data, str(data.panel["close"].index[120].date()))


def test_plan_cli_uses_verified_input_and_never_legacy_holdings(account_data, monkeypatch):
    from contextlib import contextmanager
    from qkquant import cli
    from qkquant.data import verified
    path, data = account_data
    monkeypatch.setattr(cli, "setup_logger", lambda **kwargs: None)

    @contextmanager
    def source(snapshot_dir):
        assert snapshot_dir == "audit-copy"
        yield data

    monkeypatch.setattr(verified, "open_verified_etf_input", source)
    day = str(data.panel["close"].index[120].date())
    result = CliRunner().invoke(cli.app, ["etf-plan", "--account", str(path),
                                       "--snapshot-dir", "audit-copy", "--as-of", day])
    assert result.exit_code == 0, result.output
    assert len(json.loads(path.read_text())["records"]) == 1
    assert "40%" in result.output


def test_backtest_cli_requires_verified_snapshot_before_running(monkeypatch):
    from qkquant import cli
    from qkquant.data import verified
    monkeypatch.setattr(cli, "setup_logger", lambda **kwargs: None)

    def rejected(snapshot_dir):
        raise verified.VerifiedInputError("untrusted snapshot; no fallback")

    monkeypatch.setattr(verified, "open_verified_etf_input", rejected)
    result = CliRunner().invoke(cli.app, ["etf-backtest"])
    assert result.exit_code == 2
    assert "no fallback" in result.output
