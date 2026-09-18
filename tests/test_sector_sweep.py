from __future__ import annotations

import pandas as pd
import pytest

from qkquant.sector_sweep import scan_minute_bars


def minute(time: str, gains: list[float], sector: str = "power") -> list[dict]:
    return [
        {
            "time": time,
            "code": f"00200{i}",
            "name": f"stock{i}",
            "sector": sector,
            "close": 10 * (1 + gain),
            "prev_close": 10,
        }
        for i, gain in enumerate(gains)
    ]


def test_finds_first_qualifying_minute_without_future_data():
    rows = minute("2026-09-16 09:31", [0.09, 0.08, 0.01, 0.01, -0.01])
    rows += minute("2026-09-16 09:32", [0.10, 0.08, 0.01, 0.01, -0.01])
    result = scan_minute_bars(pd.DataFrame(rows))
    assert set(result.code) == {"002000", "002001"}
    assert set(result.time) == {pd.Timestamp("2026-09-16 09:32")}


def test_extra_candidates_are_not_silently_capped_at_seven():
    rows = minute("2026-09-16 09:32", [0.10] * 8 + [0.01])
    result = scan_minute_bars(pd.DataFrame(rows))
    assert len(result) == 8


def test_weak_sector_and_non_main_board_are_excluded():
    rows = minute("2026-09-16 09:32", [0.10, 0.01, 0.01, -0.01, -0.01])
    rows += minute("2026-09-16 09:32", [0.10, 0.08, 0.01, 0.01, -0.01], "other")
    for row in rows[5:]:
        row["code"] = "3" + row["code"][1:]
    assert scan_minute_bars(pd.DataFrame(rows)).empty


def test_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="missing columns"):
        scan_minute_bars(pd.DataFrame({"code": ["000753"]}))
