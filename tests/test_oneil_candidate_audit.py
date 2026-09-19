"""Sparse original-report evidence may veto, but must never approve, a signal."""

import pytest

from qkquant.oneil import OneilConfig
from qkquant.oneil_pit import documented_financial_failure


def evidence():
    return dict(code="600001", signal_dates=["2025-08-20"], decision="verified_fail",
                latest_annual_verified=True, check="annual_roe", report_date="2024-12-31",
                published_at="2025-04-20", value=12.65, units="percentage_points", source_url="https://example.test/annual.pdf",
                source_file="annual.pdf", source_sha256="reviewed-file-hash", pdf_page=8)


def test_original_roe_veto_uses_unchanged_config_not_evidence_threshold():
    row = evidence()
    assert documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row.update(value=18, threshold=20)
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


@pytest.mark.parametrize("changes", [
    {"published_at": "2025-08-20"}, {"published_at": "2025-08-21"},
    {"latest_annual_verified": False}, {"source_sha256": ""},
    {"decision": "unresolved"}, {"code": "600002"},
    {"report_date": "2023-12-31"}, {"value": float("nan")},
    {"signal_dates": ["2025-08-21"]}, {"units": "fraction"}, {"value": None},
    {"latest_annual_published_at": "2025-07-20"},
])
def test_unavailable_or_out_of_scope_veto_cannot_reject(changes):
    row = {**evidence(), **changes}
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_eps_decline_must_be_comparable_and_within_required_four_years():
    row = {**evidence(), "check": "annual_eps_increasing", "inputs": dict(
        previous_report_date="2022-12-31", current_report_date="2023-12-31",
        previous_eps=.57, current_eps=.48, same_share_basis=True, same_accounting_basis=True)}
    assert documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row["inputs"]["same_share_basis"] = False
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row["inputs"].update(same_share_basis=True, previous_report_date="2020-12-31", current_report_date="2021-12-31")
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_old_source_can_prove_an_eps_decline_only_with_current_annual_evidence():
    row = {**evidence(), "check": "annual_eps_increasing", "report_date": "2022-12-31",
           "published_at": "2023-04-20", "signal_dates": ["2024-08-30"],
           "latest_annual_report_date": "2023-12-31", "latest_annual_published_at": "2024-04-20",
           "inputs": dict(previous_report_date="2020-12-31", current_report_date="2021-12-31",
                          previous_eps=1.03, current_eps=1.01, same_share_basis=True, same_accounting_basis=True)}
    cfg = OneilConfig()
    assert not documented_financial_failure(row, "600001", "2024-08-30", cfg)
    row.update(latest_annual_source_url="https://example.test/new.pdf", latest_annual_source_file="new.pdf",
               latest_annual_source_sha256="another-reviewed-hash")
    assert documented_financial_failure(row, "600001", "2024-08-30", cfg)
    row["inputs"].update(previous_report_date="2022-12-31", current_report_date="2023-12-31")
    assert not documented_financial_failure(row, "600001", "2024-08-30", cfg)
    row["inputs"].update(previous_report_date="2020-12-31", current_report_date="2021-12-31")
    row["inputs"]["same_accounting_basis"] = False
    assert not documented_financial_failure(row, "600001", "2024-08-30", cfg)


def test_nonpositive_eps_must_belong_to_required_history():
    row = {**evidence(), "check": "annual_eps_positive", "value": -.01,
           "inputs": {"eps_report_date": "2022-12-31"}}
    assert documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row["inputs"]["eps_report_date"] = "2020-12-31"
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_missing_inputs_do_not_count_as_verified_financial_failures():
    for check in ("annual_eps_positive", "annual_eps_increasing"):
        row = {**evidence(), "check": check, "value": -1}
        assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_old_source_cannot_document_future_negative_eps():
    row = {**evidence(), "check": "annual_eps_positive", "value": -1,
           "report_date": "2022-12-31", "published_at": "2023-04-20",
           "latest_annual_report_date": "2024-12-31", "latest_annual_published_at": "2025-04-20",
           "latest_annual_source_url": "https://example.test/latest.pdf",
           "latest_annual_source_file": "latest.pdf", "latest_annual_source_sha256": "latest-hash",
           "inputs": {"eps_report_date": "2024-12-31"}}
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_quarter_eps_growth_rejection_uses_pessimistic_rounding_bound():
    row = {**evidence(), "check": "quarter_eps_growth", "report_date": "2025-03-31",
           "published_at": "2025-04-30", "latest_quarter_verified": True,
           "inputs": dict(previous_report_date="2024-03-31", previous_eps=1.58, current_eps=1.89,
                          previous_eps_lower=1.575, current_eps_upper=1.895,
                          same_share_basis=True, same_accounting_basis=True)}
    assert documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row["inputs"].update(current_eps=1.97, current_eps_upper=1.975)
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    row["inputs"].update(current_eps=1.89, current_eps_upper=1.895, previous_report_date="2024-06-30")
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


@pytest.mark.parametrize("changes", [
    {"latest_quarter_report_date": "2025-06-30"},
    {"latest_quarter_published_at": "2025-07-20"},
    {"latest_quarter_report_date": None},
    {"published_at": "2025-08-20"},
    {"latest_quarter_verified": False},
])
def test_quarter_failure_cannot_use_a_stale_or_unavailable_version(changes):
    row = {**evidence(), "check": "quarter_eps_growth", "report_date": "2025-03-31",
           "published_at": "2025-04-30", "latest_quarter_verified": True,
           "inputs": dict(previous_report_date="2024-03-31", previous_eps=1.58, current_eps=1.89,
                          previous_eps_lower=1.575, current_eps_upper=1.895,
                          same_share_basis=True, same_accounting_basis=True), **changes}
    assert not documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())


def test_annual_cagr_failure_requires_comparable_available_endpoints_and_rounding_margin():
    row = {**evidence(), "check": "annual_eps_cagr", "inputs": dict(
        previous_report_date="2021-12-31", current_report_date="2024-12-31",
        previous_eps=1.3165, current_eps=2.5302, previous_eps_lower=1.31645, current_eps_upper=2.53025,
        previous_eps_published_at="2024-04-20", current_eps_published_at="2025-04-20",
        same_share_basis=True, same_accounting_basis=True)}
    assert documented_financial_failure(row, "600001", "2025-08-20", OneilConfig())
    for change in ({"same_share_basis": False}, {"same_accounting_basis": False},
                   {"previous_report_date": "2022-12-31"}, {"current_report_date": "2023-12-31"},
                   {"previous_eps_published_at": "2025-08-20"}, {"current_eps_published_at": "2025-08-20"},
                   {"previous_eps_lower": 0}, {"current_eps_upper": float("nan")},
                   {"current_eps": 2.5712, "current_eps_upper": 2.5713}):
        changed = {**row, "inputs": {**row["inputs"], **change}}
        assert not documented_financial_failure(changed, "600001", "2025-08-20", OneilConfig())
