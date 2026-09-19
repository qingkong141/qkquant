"""The historical report may veto actual sealed-cohort signals, never approve them."""

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def save_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def audit_runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/research_oneil_historical_candidates.py"
    spec = importlib.util.spec_from_file_location("historical_candidates_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot, output = tmp_path / "snapshot", tmp_path / "report"
    (snapshot / "sources").mkdir(parents=True)
    evidence_file = tmp_path / "evidence.json"
    pdf = tmp_path / "annual.pdf"
    pdf.write_bytes(b"reviewed original report fixture")
    source = dict(source_file=pdf.name, source_sha256=digest(pdf))
    decision = dict(code="600001", signal_dates=["2025-08-20"], decision="verified_fail",
                    latest_annual_verified=True, check="annual_roe", report_date="2024-12-31",
                    published_at="2025-04-20", value=8.15, units="percentage_points",
                    source_url="https://example.test/original-annual.pdf", pdf_page=9,
                    threshold=17, operator=">=", **source)
    payload = dict(sources=[source], decisions=[decision])
    save_json(evidence_file, payload)
    dates = pd.date_range("2025-08-19", periods=4)
    codes = ["600001", "600002"]
    bars_path = snapshot / "bars.csv.gz"
    bars = pd.DataFrame([dict(trade_date=day, code=code, close=10.0)
                         for day in dates for code in codes])
    bars.to_csv(bars_path, index=False)
    benchmark = tmp_path / "data/research/oneil_20260919/benchmark.csv"
    benchmark.parent.mkdir(parents=True)
    pd.DataFrame(dict(trade_date=dates, raw_close=[100, 101, 102, 103])).to_csv(benchmark, index=False)
    source_hashes = {}
    for code in codes:
        for adjust in ("3", "1"):
            cache = snapshot / "sources" / f"{code}_{adjust}.json"
            save_json(cache, dict(code=code, adjustflag=adjust, records=[]))
            source_hashes[cache.name] = digest(cache)
    manifest = dict(config=asdict(module.OneilConfig()), bars_sha256=digest(bars_path),
                    input_sha256={str(benchmark.relative_to(tmp_path)): digest(benchmark)},
                    source_sha256=source_hashes, selected_codes=codes, selected_count=2,
                    inception="2025-08-20", price_end="2025-08-21", technical_events=2, market_events=1)
    save_json(snapshot / "manifest.json", manifest)
    # True events outside the sealed screening dates must not enter the report.
    prepared = dict(technical=pd.DataFrame([[True, True], [True, False],
                                            [False, True], [True, True]], index=dates, columns=codes),
                    market_ok=pd.Series([True, True, False, True], index=dates))
    calls = []

    def prepare(input_bars, input_benchmark, cfg):
        calls.append((input_bars, input_benchmark, cfg))
        return prepared

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SNAPSHOT", snapshot)
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(module, "EVIDENCE", [evidence_file])
    monkeypatch.setattr(module, "prepare_prices", prepare)
    return SimpleNamespace(module=module, snapshot=snapshot, output=output, evidence=evidence_file,
                           payload=payload, manifest=manifest, bars=bars_path, benchmark=benchmark,
                           pdf=pdf, prepared=prepared, calls=calls)


def results(runner):
    runner.module.main()
    signals = pd.read_csv(runner.output / "candidate_decisions.csv", dtype={"code": str})
    audit = json.loads((runner.output / "audit.json").read_text(encoding="utf-8"))
    return signals, audit


def test_only_actual_in_range_signals_are_counted_and_market_failure_is_separate(audit_runner):
    r = audit_runner
    r.payload["decisions"].append({**r.payload["decisions"][0], "code": "600999"})
    save_json(r.evidence, r.payload)
    signals, audit = results(r)
    assert list(zip(signals.code, signals.signal_date, strict=True)) == [
        ("600001", "2025-08-20"), ("600002", "2025-08-21")]
    assert signals.financial_status.tolist() == ["verified_fail", "unresolved"]
    assert signals.entry_status.tolist() == ["financial_rejected", "market_rejected"]
    assert not signals.qualified.any()
    assert audit["technical_events"] == audit["candidate_companies"] == 2
    assert audit["entry_status_counts"] == {"financial_rejected": 1, "market_rejected": 1}
    assert audit["financial_status_counts"] == {"verified_fail": 1, "unresolved": 1}
    assert not audit["financial_passes_certified"] and not audit["returns_tested"]
    assert asdict(r.calls[0][2]) == asdict(r.module.OneilConfig())
    assert pd.read_csv(r.output / "unresolved_entries.csv").empty


@pytest.mark.parametrize("changes", [
    {"published_at": "2025-08-20"},
    {"published_at": "2025-08-21"},
    {"signal_dates": ["2025-08-19"]},
    {"code": "600002"},
    {"latest_annual_verified": False},
    {"decision": "unresolved"},
    {"decision": "verified_pass", "value": 30},
    {"value": 18, "threshold": 20},
])
def test_unavailable_or_nonfailing_evidence_never_certifies_or_rejects(audit_runner, changes):
    r = audit_runner
    r.payload["decisions"][0].update(changes)
    save_json(r.evidence, r.payload)
    signals, audit = results(r)
    assert signals.financial_status.tolist() == ["unresolved", "unresolved"]
    assert signals.entry_status.tolist() == ["unresolved", "market_rejected"]
    assert not signals.qualified.any()
    pending = pd.read_csv(r.output / "unresolved_entries.csv", dtype={"code": str})
    assert pending.code.tolist() == ["600001"]
    assert audit["financial_status_counts"] == {"unresolved": 2}


def test_missing_financial_review_stays_unresolved(audit_runner):
    r = audit_runner
    r.payload["decisions"] = []
    save_json(r.evidence, r.payload)
    signals, _ = results(r)
    assert signals.financial_status.eq("unresolved").all()
    assert not signals.qualified.any()


@pytest.mark.parametrize("which", ["bars", "benchmark", "raw", "source_pdf", "decision_hash"])
def test_source_changes_are_rejected_before_screening_or_output(audit_runner, which):
    r = audit_runner
    if which == "decision_hash":
        r.payload["decisions"][0]["source_sha256"] = "0" * 64
        save_json(r.evidence, r.payload)
    else:
        file = {"bars": r.bars, "benchmark": r.benchmark,
                "raw": r.snapshot / "sources/600001_3.json", "source_pdf": r.pdf}[which]
        file.write_bytes(file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        r.module.main()
    assert not r.calls
    assert not r.output.exists()


def test_changed_original_parameters_are_rejected_before_screening(audit_runner):
    r = audit_runner
    r.manifest["config"]["min_roe"] = 10
    save_json(r.snapshot / "manifest.json", r.manifest)
    with pytest.raises(ValueError, match="original parameters"):
        r.module.main()
    assert not r.calls and not r.output.exists()


@pytest.mark.parametrize("field", ["technical_events", "market_events"])
def test_recomputed_event_count_must_match_the_sealed_screen(audit_runner, field):
    r = audit_runner
    r.manifest[field] += 1
    save_json(r.snapshot / "manifest.json", r.manifest)
    with pytest.raises(ValueError, match="screening results changed"):
        r.module.main()
    assert not r.output.exists()


def test_incomplete_frozen_source_set_cannot_be_called_a_complete_screen(audit_runner):
    r = audit_runner
    del r.manifest["source_sha256"]["600002_1.json"]
    (r.snapshot / "sources/600002_1.json").unlink()
    save_json(r.snapshot / "manifest.json", r.manifest)
    with pytest.raises(ValueError):
        r.module.main()
    assert not r.calls and not r.output.exists()


@pytest.mark.parametrize("change", ["count", "duplicate"])
def test_frozen_membership_is_consistent_before_reporting(audit_runner, change):
    r = audit_runner
    if change == "count":
        r.manifest["selected_count"] = 406
    else:
        r.manifest["selected_codes"] = ["600001", "600001"]
    save_json(r.snapshot / "manifest.json", r.manifest)
    with pytest.raises(ValueError):
        r.module.main()
    assert not r.output.exists()


def test_benchmark_cannot_bypass_hash_verification_by_omission(audit_runner):
    r = audit_runner
    r.manifest["input_sha256"] = {}
    save_json(r.snapshot / "manifest.json", r.manifest)
    with pytest.raises(ValueError):
        r.module.main()
    assert not r.calls and not r.output.exists()


@pytest.mark.parametrize("change", ["bars", "technical"])
def test_companies_outside_the_frozen_cohort_cannot_enter_statistics(audit_runner, change):
    r = audit_runner
    if change == "bars":
        bars = pd.read_csv(r.bars, dtype={"code": str})
        bars.loc[bars.code == "600002", "code"] = "600999"
        bars.to_csv(r.bars, index=False)
        r.manifest["bars_sha256"] = digest(r.bars)
        save_json(r.snapshot / "manifest.json", r.manifest)
    else:
        r.prepared["technical"] = r.prepared["technical"].rename(columns={"600002": "600999"})
    with pytest.raises(ValueError):
        r.module.main()
    assert not r.output.exists()
