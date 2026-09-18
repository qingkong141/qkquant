from __future__ import annotations

import numpy as np
import pandas as pd

from qkquant.etf_chan import classify_chan_structure


def test_chan_requires_enough_bars():
    result = classify_chan_structure(pd.Series(range(20), dtype=float))
    assert result.state == "insufficient_data"
    assert result.allowed is False


def test_chan_output_contains_divergence_fields():
    x = np.arange(120)
    close = pd.Series(100 + 0.05 * x + 3 * np.sin(x / 4), dtype=float)
    result = classify_chan_structure(close, close + 0.5, close - 0.5)
    payload = result.to_dict()
    assert "divergence" in payload
    assert "divergence_strength" in payload
    assert result.state in {
        "first_buy_divergence", "top_divergence", "second_buy", "third_buy",
        "second_buy_watch", "structure_broken", "no_entry", "insufficient_fractals",
    }


def test_top_divergence_vetoes_entry_when_detected():
    x = np.arange(160)
    amplitude = np.linspace(8, 2, len(x))
    close = pd.Series(100 + 0.08 * x + amplitude * np.sin(x / 5), dtype=float)
    result = classify_chan_structure(close, close + 0.2, close - 0.2, tolerance=0.001)
    if result.divergence == "top":
        assert result.state == "top_divergence"
        assert result.allowed is False
