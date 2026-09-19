"""Completed research snapshots must reproduce offline without changing evidence."""

import importlib.util
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/research_oneil_universe.py"
    spec = importlib.util.spec_from_file_location("oneil_universe_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PROBE", tmp_path / "probe")
    snapshot, output = tmp_path / "snapshot", tmp_path / "report"
    (snapshot / "sources").mkdir(parents=True)
    (module.PROBE / "official").mkdir(parents=True)
    (module.PROBE / "baostock").mkdir()
    holdings = pd.DataFrame([dict(code="600001", publication_date="2024-03-30",
                                  holdings_asof="2023-12-31", market_value=1000)])
    holdings.to_csv(module.PROBE / "official/holdings.csv", index=False)
    (module.PROBE / "official/510500_2023_annual.pdf").write_bytes(b"fixture source document")
    module.save_json(module.PROBE / "baostock/all_stock_20240701.json", {
        "request": {"day": module.INCEPTION}, "error_code": "0",
        "records": [{"code": "sh.600001", "tradeStatus": "1", "code_name": "fixture"}]})
    days = pd.bdate_range(module.START, periods=300)
    benchmark_path = tmp_path / "data/research/oneil_20260919/benchmark.csv"
    benchmark_path.parent.mkdir(parents=True)
    pd.DataFrame(dict(trade_date=days, raw_close=range(100, 400))).to_csv(benchmark_path, index=False)
    for adjust, multiplier in (("3", 1), ("1", 2)):
        records = [dict(date=str(day.date()), code="sh.600001", open=str(10 * multiplier),
                        high=str(10.1 * multiplier), low=str(9.9 * multiplier), close=str(10 * multiplier),
                        volume="10000000", amount="100000000", tradestatus="1", isST="0") for day in days]
        module.save_json(snapshot / "sources" / f"600001_{adjust}.json", {
            "provider": "BaoStock", "api": "query_history_k_data_plus",
            "request": module.bar_request("600001", adjust), "fields": module.FIELDS.split(","),
            "error_code": "0", "records": records})

    def forbid_network(*args, **kwargs):
        raise AssertionError("offline execution attempted a BaoStock call")

    monkeypatch.setitem(sys.modules, "baostock", SimpleNamespace(
        login=forbid_network, logout=forbid_network, query_history_k_data_plus=forbid_network))
    monkeypatch.setitem(sys.modules, "baostock.common", SimpleNamespace(
        context=SimpleNamespace(default_socket=None)))

    def run(*extra, snapshot_path=snapshot, output_path=output):
        monkeypatch.setattr(sys, "argv", [str(path), "--snapshot", str(snapshot_path),
                                         "--output", str(output_path), *extra])
        module.main()

    return SimpleNamespace(module=module, snapshot=snapshot, output=output, run=run,
                           benchmark=benchmark_path, monkeypatch=monkeypatch)


def fingerprint(folder):
    return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


def test_completed_offline_run_changes_only_reports_and_does_not_share_plan_hashes(runner):
    runner.run()
    before = fingerprint(runner.snapshot)
    plan = runner.module.read_json(runner.snapshot / "frozen_plan.json")
    manifest = runner.module.read_json(runner.snapshot / "manifest.json")
    assert len(plan["input_sha256"]) == 3
    assert len(manifest["input_sha256"]) == 4
    (runner.output / "technical_candidates.csv").write_text("replace this report", encoding="utf-8")
    runner.run()
    assert fingerprint(runner.snapshot) == before
    assert "financial_status" in (runner.output / "technical_candidates.csv").read_text(encoding="utf-8")
    assert not manifest["returns_tested"]


@pytest.mark.parametrize("changed", ["source", "benchmark", "holdings", "config", "bars", "cohort", "normalization"])
def test_changed_completed_evidence_is_rejected_before_any_output_write(runner, changed):
    runner.run()
    if changed == "source":
        path = runner.snapshot / "sources/600001_3.json"
        payload = runner.module.read_json(path)
        payload["records"][0]["amount"] = "100000001"
        runner.module.save_json(path, payload)
    elif changed == "benchmark":
        runner.benchmark.write_bytes(runner.benchmark.read_bytes() + b"\n")
    elif changed == "holdings":
        path = runner.module.PROBE / "official/holdings.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif changed == "config":
        cfg = replace(runner.module.OneilConfig(), rs_days=200)
        runner.monkeypatch.setattr(runner.module, "OneilConfig", lambda: cfg)
    elif changed == "bars":
        path = runner.snapshot / "bars.csv.gz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif changed == "cohort":
        (runner.snapshot / "frozen_cohort.csv").write_text("changed", encoding="utf-8")
    else:
        original = runner.module.checked_baostock_bars

        def changed_normalization(*args):
            bars, audit = original(*args)
            bars["eligible"] = ~bars.eligible
            return bars, audit

        runner.monkeypatch.setattr(runner.module, "checked_baostock_bars", changed_normalization)
    before_snapshot, before_report = fingerprint(runner.snapshot), fingerprint(runner.output)
    with pytest.raises(ValueError, match="changed"):
        runner.run()
    assert fingerprint(runner.snapshot) == before_snapshot
    assert fingerprint(runner.output) == before_report


@pytest.mark.parametrize("changed", ["request", "error", "fields", "record_fields", "code", "early", "late", "missing_date"])
def test_cache_read_rejects_unusable_or_out_of_scope_responses(runner, changed):
    path = runner.snapshot / "sources/600001_3.json"
    payload = runner.module.read_json(path)
    if changed == "request":
        payload["request"]["adjustflag"] = "1"
    elif changed == "error":
        payload["error_code"] = "10002007"
    elif changed == "fields":
        payload["fields"].remove("isST")
    elif changed == "record_fields":
        del payload["records"][0]["isST"]
    elif changed == "code":
        payload["records"][0]["code"] = "sh.600002"
    else:
        payload["records"][0]["date"] = {"early": "2023-06-18", "late": "2026-07-18", "missing_date": None}[changed]
    runner.module.save_json(path, payload)
    with pytest.raises(ValueError, match="cached"):
        runner.module.read_cached_bars("600001", "3", runner.snapshot)


def test_missing_cache_never_connects_or_seals_an_unfinished_snapshot(runner):
    (runner.snapshot / "sources/600001_1.json").unlink()
    with pytest.raises(FileNotFoundError):
        runner.run()
    assert not (runner.snapshot / "manifest.json").exists()


def test_explicit_fetch_resumes_only_the_missing_response(runner):
    path = runner.snapshot / "sources/600001_1.json"
    missing = runner.module.read_json(path)
    path.unlink()
    existing = (runner.snapshot / "sources/600001_3.json").read_bytes()
    calls = []

    class Response:
        error_code, error_msg = "0", "success"
        fields = missing["fields"]

        def __init__(self):
            self.position = 0

        def next(self):
            return self.position < len(missing["records"])

        def get_row_data(self):
            row = missing["records"][self.position]
            self.position += 1
            return [row[field] for field in self.fields]

    def query(**request):
        calls.append(request)
        return Response()

    runner.monkeypatch.setitem(sys.modules, "baostock", SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0"), logout=lambda: None,
        query_history_k_data_plus=query))
    runner.run("--fetch")
    assert calls == [runner.module.bar_request("600001", "1")]
    assert (runner.snapshot / "sources/600001_3.json").read_bytes() == existing
    assert (runner.snapshot / "manifest.json").exists()


def mock_download(runner, responses, login_codes=()):
    """A fresh mock socket per login exposes reconnect ordering and concurrency."""
    path = runner.snapshot / "sources/600001_1.json"
    payload = runner.module.read_json(path)
    path.unlink()
    context = sys.modules["baostock.common"].context
    events = []
    replies, logins = iter(responses), iter(login_codes)

    class Connection:
        def close(self):
            events.append("close")

    class Response:
        error_code, error_msg = "0", "success"

        def __init__(self, outcome):
            self.fields = payload["fields"].copy()
            self.position = 0
            if outcome == "fields":
                self.fields.remove("isST")
            elif outcome == "date":
                payload["records"][0]["date"] = "2026-07-18"
            elif outcome == "data":
                payload["records"][0]["date"] = payload["records"][1]["date"]
            elif outcome != "success":
                self.error_code = outcome

        def next(self):
            return self.position < len(payload["records"])

        def get_row_data(self):
            row = payload["records"][self.position]
            self.position += 1
            return [row[field] for field in self.fields]

    def login():
        assert context.default_socket is None, "old socket survived reconnect"
        events.append("login")
        context.default_socket = Connection()
        return SimpleNamespace(error_code=next(logins, "0"), error_msg="mock login")

    def query(**request):
        assert request == runner.module.bar_request("600001", "1")
        assert context.default_socket is not None
        events.append("query")
        return Response(next(replies))

    runner.monkeypatch.setitem(sys.modules, "baostock", SimpleNamespace(
        login=login, logout=lambda: events.append("logout"), query_history_k_data_plus=query))
    runner.monkeypatch.setattr(runner.module.time, "sleep", lambda _: None)
    return events, context


@pytest.mark.parametrize("error_code", ["10002007", "10001001"])
def test_connection_failure_closes_socket_before_relogin_and_preserves_cache(runner, capsys, error_code):
    existing = fingerprint(runner.snapshot / "sources")["600001_3.json"]
    events, context = mock_download(runner, [error_code, "success"])
    runner.run("--fetch")
    assert events == ["login", "query", "close", "login", "query", "logout", "close"]
    assert context.default_socket is None
    assert fingerprint(runner.snapshot / "sources")["600001_3.json"] == existing
    assert (runner.snapshot / "manifest.json").exists()
    assert f"connection error {error_code}; attempt 1/3, reconnecting" in capsys.readouterr().out


def test_login_connection_failure_uses_the_same_bounded_recovery(runner):
    events, context = mock_download(runner, ["success"], login_codes=["10002007", "0"])
    runner.run("--fetch")
    assert events == ["login", "close", "login", "query", "logout", "close"]
    assert context.default_socket is None


def test_connection_retry_exhaustion_never_logs_out_over_failed_socket(runner, capsys):
    events, context = mock_download(runner, ["10002007"] * 3)
    with pytest.raises(runner.module.BaoStockConnectionError) as caught:
        runner.run("--fetch")
    assert caught.value.error_code == "10002007"
    assert events == ["login", "query", "close"] * 3
    assert context.default_socket is None
    assert not (runner.snapshot / "sources/600001_1.json").exists()
    assert not (runner.snapshot / "manifest.json").exists()
    assert "attempt 3/3, stopping" in capsys.readouterr().out


@pytest.mark.parametrize("outcome", ["10004004", "fields", "date", "data"])
def test_non_connection_errors_are_not_retried(runner, outcome):
    events, context = mock_download(runner, [outcome])
    with pytest.raises(ValueError):
        runner.run("--fetch")
    assert events == ["login", "query", "logout", "close"]
    assert context.default_socket is None
    assert not (runner.snapshot / "manifest.json").exists()


def test_invalid_cached_error_code_never_triggers_network_retry(runner):
    path = runner.snapshot / "sources/600001_3.json"
    payload = runner.module.read_json(path)
    payload["error_code"] = "10002007"
    runner.module.save_json(path, payload)
    before = fingerprint(runner.snapshot / "sources")
    with pytest.raises(ValueError, match="unsuccessful cached response"):
        runner.run("--fetch")
    assert fingerprint(runner.snapshot / "sources") == before


def test_first_seal_gzip_is_deterministic_and_completed_fetch_is_rejected(runner):
    other = runner.snapshot.parent / "second_snapshot"
    shutil.copytree(runner.snapshot, other)
    runner.monkeypatch.setattr("gzip.time.time", lambda: 1000000000)
    runner.run()
    runner.monkeypatch.setattr("gzip.time.time", lambda: 1000000001)
    runner.run(snapshot_path=other)
    assert (runner.snapshot / "bars.csv.gz").read_bytes() == (other / "bars.csv.gz").read_bytes()
    before = fingerprint(runner.snapshot)
    with pytest.raises(ValueError, match="immutable"):
        runner.run("--fetch")
    assert fingerprint(runner.snapshot) == before


def test_report_destination_cannot_write_inside_snapshot(runner):
    before = fingerprint(runner.snapshot)
    with pytest.raises(ValueError, match="outside the snapshot"):
        runner.run(output_path=runner.snapshot / "reports")
    assert fingerprint(runner.snapshot) == before
