"""Historical document versions must not leak later financial corrections."""

import numpy as np
import pandas as pd
import pytest

from qkquant.oneil import OneilConfig
from qkquant.oneil_pit import (
    financial_state_asof,
    prepare_financials_asof,
    select_financial_versions,
)


def versions():
    # Real correction values; source evidence is archived by the pilot runner.
    return pd.DataFrame([
        dict(code="301607", kind="annual", report_date="2025-12-31", published_at="2026-04-27",
             eps=1.94, version_id="1225175090", source_url="https://static.cninfo.com.cn/finalpage/2026-04-27/1225175090.PDF"),
        dict(code="301607", kind="annual", report_date="2025-12-31", published_at="2026-05-07",
             eps=1.38, version_id="1225283007", source_url="https://static.cninfo.com.cn/finalpage/2026-05-07/1225283007.PDF"),
    ])


def test_actual_correction_is_available_only_after_its_publication_date():
    rows = versions()
    assert select_financial_versions(rows, "2026-04-27").empty
    assert select_financial_versions(rows, "2026-04-28").eps.item() == 1.94
    assert select_financial_versions(rows, "2026-05-07").eps.item() == 1.94
    assert select_financial_versions(rows, "2026-05-08").eps.item() == 1.38


def test_future_version_and_provider_refresh_do_not_change_past_values():
    rows = versions()
    rows["update_date"] = "2030-01-01"
    pd.testing.assert_frame_equal(select_financial_versions(rows, "2026-05-01"),
                                  select_financial_versions(rows.iloc[:1], "2026-05-01"))


def test_new_version_with_missing_value_does_not_silently_reuse_old_value():
    rows = versions()
    rows.loc[1, "eps"] = np.nan
    chosen = select_financial_versions(rows, "2026-05-08")
    assert chosen.version_id.item() == "1225283007"
    assert chosen.eps.isna().all()


def test_same_day_conflict_is_rejected_instead_of_selecting_arbitrarily():
    rows = versions()
    rows.loc[1, "published_at"] = rows.loc[0, "published_at"]
    with pytest.raises(ValueError, match="ambiguous same-day"):
        select_financial_versions(rows, "2026-05-08")


def test_future_conflict_cannot_block_a_valid_past_query():
    rows = versions()
    conflict = rows.iloc[[1]].assign(version_id="conflicting-future-record", eps=.5)
    rows = pd.concat([rows, conflict], ignore_index=True)
    assert select_financial_versions(rows, "2026-04-28").eps.item() == 1.94
    with pytest.raises(ValueError, match="ambiguous same-day"):
        select_financial_versions(rows, "2026-05-08")


@pytest.mark.parametrize("column", ["published_at", "source_url", "version_id"])
def test_undocumented_version_is_rejected(column):
    rows = versions()
    rows.loc[0, column] = None
    with pytest.raises(ValueError, match="missing"):
        select_financial_versions(rows, "2026-05-08")


def bonus_actions():
    return pd.DataFrame([dict(code="605499", action_id="bonus2024", action_type="bonus_shares",
                              factor=1.3, published_at="2024-09-26", effective_date="2024-10-09",
                              source_url="https://static.cninfo.com.cn/finalpage/2024-09-26/1221291027.PDF")])


def reviewed_rows():
    # Original reported precision; current Q3 already uses the new share basis.
    rows = []
    for year, eps in zip(range(2020, 2024), (2.2557, 3.1120, 3.6012, 5.0993), strict=True):
        rows.append(dict(kind="annual", report_date=f"{year}-12-31", published_at=f"{year + 1}-04-20",
                         eps=eps, revenue=np.nan, roe=35.82, eps_applied_actions=[]))
    for year, eps, sales in ((2023, 1.3687, 3180727441.02), (2024, 1.8784, 4684993650.16)):
        rows.append(dict(kind="quarter", report_date=f"{year}-09-30", published_at=f"{year}-10-30",
                         eps=eps, revenue=sales, roe=np.nan,
                         eps_applied_actions=["bonus2024"] if year == 2024 else []))
    result = pd.DataFrame(rows)
    result["code"] = "605499"
    result["version_id"] = result.report_date + result.kind
    result["source_url"] = "https://example.test/reviewed-report"
    result["eps_lower"] = result.eps - .00005
    result["eps_upper"] = result.eps + .00005
    return result


def test_bonus_is_not_applied_before_effective_or_publication_date():
    rows, actions = reviewed_rows(), bonus_actions()
    before = prepare_financials_asof(rows, actions, "2024-10-08")
    after = prepare_financials_asof(rows, actions, "2024-10-09")
    assert before.eps.iloc[0] == pytest.approx(2.2557)
    assert after.eps.iloc[0] == pytest.approx(2.2557 / 1.3)
    actions["published_at"] = "2024-10-09"
    assert prepare_financials_asof(rows, actions, "2024-10-09").eps.iloc[0] == pytest.approx(2.2557)


def test_comparable_eps_passes_original_rules_without_double_adjustment():
    rows, actions = reviewed_rows(), bonus_actions()
    selected = prepare_financials_asof(rows, actions, "2024-12-12")
    current = selected[selected.report_date == "2024-09-30"].iloc[0]
    assert current.eps == 1.8784
    assert current.eps_adjustment_factor == 1
    state = financial_state_asof(rows, actions, "2024-12-12", OneilConfig())
    assert state["fundamental_ok"]
    assert state["quarter_eps_growth_lower"] == pytest.approx(.784003652968)
    assert state["annual_eps_cagr"] == pytest.approx(.312430781838)


def test_uncertain_growth_cannot_pass_on_point_estimate():
    rows, actions = reviewed_rows(), bonus_actions()
    current = rows.report_date == "2024-09-30"
    rows.loc[current, ["eps", "eps_lower", "eps_upper"]] = [1.4, 1.2, 1.6]
    state = financial_state_asof(rows, actions, "2024-12-12", OneilConfig())
    assert state["quarter_eps_growth"] > .25
    assert state["quarter_eps_growth_lower"] < .25
    assert not state["fundamental_ok"]
    assert state["reason"] == "eps_threshold_not_proven_by_bounds"


def test_missing_eps_basis_or_future_basis_cannot_be_guessed():
    rows = reviewed_rows()
    rows.at[0, "eps_applied_actions"] = None
    with pytest.raises(ValueError, match="declare"):
        prepare_financials_asof(rows, bonus_actions(), "2024-12-12")
    rows.at[0, "eps_applied_actions"] = ["future_action"]
    with pytest.raises(ValueError, match="unavailable action"):
        prepare_financials_asof(rows, bonus_actions(), "2024-12-12")


def test_ordinary_issuance_is_not_a_retrospective_eps_factor():
    actions = bonus_actions().assign(action_type="option_exercise")
    with pytest.raises(ValueError, match="equity-neutral"):
        prepare_financials_asof(reviewed_rows(), actions, "2024-12-12")


def test_derived_point_must_be_inside_its_interval():
    rows = reviewed_rows()
    rows.loc[rows.report_date == "2024-09-30", "eps_upper"] = 1.8
    state = financial_state_asof(rows, bonus_actions(), "2024-12-12", OneilConfig())
    assert state["reason"] == "invalid_eps_bounds"
    assert not state["fundamental_ok"]


def test_financial_state_rejects_mixed_companies():
    rows = reviewed_rows()
    rows.loc[0, "code"] = "600926"
    with pytest.raises(ValueError, match="one company"):
        financial_state_asof(rows, bonus_actions(), "2024-12-12", OneilConfig())


def test_empty_actions_and_no_incorporated_events_are_valid():
    rows = reviewed_rows()
    rows["eps_applied_actions"] = [[] for _ in range(len(rows))]
    empty = pd.DataFrame(columns=bonus_actions().columns)
    selected = prepare_financials_asof(rows, empty, "2024-12-12")
    assert selected.eps_adjustment_factor.eq(1).all()


def test_old_publication_cannot_claim_to_include_a_future_bonus():
    rows = reviewed_rows()
    rows.at[0, "eps_applied_actions"] = ["bonus2024"]
    with pytest.raises(ValueError, match="predates"):
        prepare_financials_asof(rows, bonus_actions(), "2024-12-12")
