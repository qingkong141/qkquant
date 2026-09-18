from __future__ import annotations

import pandas as pd

from qkquant.etf_chan_research import _summary


def test_summary_groups_states_and_computes_win_rate():
    samples = pd.DataFrame({
        "state": ["second_buy", "second_buy", "third_buy"],
        "return_5d": [0.10, -0.05, 0.02],
        "mae_5d": [-0.02, -0.08, -0.01],
    })
    rows = _summary(samples, (5,))
    all_row = next(row for row in rows if row["state"] == "all")
    second = next(row for row in rows if row["state"] == "second_buy")
    assert all_row["n"] == 3
    assert all_row["win_rate"] == 2 / 3
    assert second["n"] == 2
    assert second["mean_mae"] == -0.05
