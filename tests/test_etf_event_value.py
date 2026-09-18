import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "event_value", Path(__file__).parents[1] / "scripts" / "validate_etf_event_value.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_events(positions=(120, 121, 141)):
    rows = []
    for index, pos in enumerate(positions):
        row = dict(date=str(pd.Timestamp("2025-01-01").date() + pd.Timedelta(days=index)),
                   position=pos, code="A", chan=True, pullback=True, trend=False,
                   scheduled=(pos - 120) % 10 == 0)
        for cost in ("base", "stress"):
            for horizon in (10, 20):
                row.update({f"{cost}_status_{horizon}": "valid", f"{cost}_net_{horizon}": .01,
                            f"{cost}_edge_{horizon}": .002, f"{cost}_mae_{horizon}": -.02,
                            f"{cost}_gross_{horizon}": .015})
        rows.append(row)
    return pd.DataFrame(rows)


def test_parent_spacing_precedes_schedule_split_and_is_strictly_more_than_twenty():
    events = sample_events((120, 121, 140, 141))
    samples = module.group_samples(events)
    assert samples["chan", "spaced"].position.tolist() == [120, 141]
    assert samples["chan_off_schedule", "spaced"].position.tolist() == [141]
    assert samples["chan_scheduled", "spaced"].position.tolist() == [120]
    assert samples["chan_off_schedule", "all"].position.tolist() == [121, 141]


def test_outcomes_cannot_change_spaced_identities():
    events = sample_events()
    original = module.group_samples(events)["chan", "spaced"].index
    events.loc[0, "base_status_10"] = "missing"
    events.loc[0, "base_net_10"] = np.nan
    events.loc[2, "base_net_10"] = 100
    assert module.group_samples(events)["chan", "spaced"].index.equals(original)


def test_date_equal_mean_does_not_overweight_same_day_etfs():
    events = sample_events()
    events["date"] = ["2025-01-01", "2025-01-01", "2025-01-02"]
    events["code"] = ["A", "B", "A"]
    events["base_net_10"] = [.1, .1, -.1]
    events["base_edge_10"] = [.02, .04, -.03]
    stats, dates = module.event_statistics(events, "base", 10)
    assert stats["mean_net"] == pytest.approx(.1 / 3)
    assert stats["date_equal_net"] == pytest.approx(0)
    assert stats["date_equal_edge"] == pytest.approx(0)
    assert stats["leave_one_date_net_min"] == pytest.approx(-.1)
    assert stats["leave_one_date_net_max"] == pytest.approx(.1)
    assert len(dates) == 2


def test_single_date_has_no_leave_one_date_sensitivity_range():
    low, high = module.leave_one_date_bounds(pd.Series([.01]))
    assert np.isnan(low) and np.isnan(high)


def test_all_execution_statuses_retained_and_cost_comparison_is_paired():
    events = sample_events((120, 141, 162, 183))
    for index, status in enumerate(("valid", "pending", "missing", "unexecutable")):
        events.loc[index, "stress_status_10"] = status
        if index:
            events.loc[index, "stress_net_10"] = np.nan
    summary, _, pairs = module.summarize_samples(module.group_samples(events))
    row = summary.loc[(summary.group == "chan") & (summary["sample"] == "all")
                      & (summary.period == "full") & (summary.cost == "stress") & (summary.horizon == 10)].iloc[0]
    assert row.events == 4 and row.valid == 1
    assert row.pending == row.missing == row.unexecutable == 1
    pair = pairs.loc[(pairs.group == "chan") & (pairs["sample"] == "all")
                     & (pairs.period == "full") & (pairs.horizon == 10)].iloc[0]
    assert pair.events == 1


def test_extension_split_does_not_restart_spacing():
    events = sample_events((120, 130, 141))
    events["date"] = ["2026-07-10", "2026-07-22", "2026-08-06"]
    selected = module.group_samples(events)["chan", "spaced"]
    extension = dict(module.period_slices(selected))["historical_extension"]
    assert extension.position.tolist() == [141]


def test_funnel_identity_disagreement_is_rejected():
    events = sample_events()
    reference = events.assign(daily_eligible=True)
    module.reconcile_identity(events, reference)
    with pytest.raises(ValueError, match="identities differ"):
        module.reconcile_identity(events.iloc[:-1], reference)


def test_splitting_groups_does_not_recompute_same_date_reference_edge():
    events = sample_events()
    events.loc[1, "base_edge_10"] = -.023
    subgroup = module.group_samples(events)["chan_off_schedule", "all"]
    assert subgroup.loc[1, "base_edge_10"] == -.023
