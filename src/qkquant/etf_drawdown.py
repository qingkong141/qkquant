"""Daily close-based risk budget for the ETF research backtest."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownConfig:
    max_exposure: float = 0.40
    reduce_at: float = 0.04
    halve_at: float = 0.06
    exit_at: float = 0.08
    hard_stop: float = 0.10
    recovery_days: int = 20

    def __post_init__(self):
        values = (self.max_exposure, self.reduce_at, self.halve_at, self.exit_at, self.hard_stop)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("risk parameters must be finite")
        if not 0 < self.max_exposure <= 1:
            raise ValueError("max_exposure must be in (0, 1]")
        if not 0 < self.reduce_at < self.halve_at < self.exit_at < self.hard_stop < 1:
            raise ValueError("require 0 < reduce < halve < exit < hard_stop < 1")
        if not isinstance(self.recovery_days, int) or self.recovery_days < 1:
            raise ValueError("recovery_days must be a positive integer")


@dataclass
class DrawdownState:
    peak: float
    anchor: float
    multiplier: float = 1.0
    last_change: int = -1
    halted: bool = False

    def update(self, equity, position, scheduled, market_ok, cfg):
        self.peak = max(self.peak, equity)
        self.anchor = max(self.anchor, equity)
        total_dd = 1 - equity / self.peak
        control_dd = 1 - equity / self.anchor
        previous = self.multiplier
        if total_dd >= cfg.hard_stop - 1e-12:
            self.halted = True
        if self.halted:
            self.multiplier = 0.0
        else:
            target = (0.0 if control_dd >= cfg.exit_at - 1e-12 else
                      0.5 if control_dd >= cfg.halve_at - 1e-12 else
                      0.75 if control_dd >= cfg.reduce_at - 1e-12 else 1.0)
            if target < self.multiplier:
                self.multiplier = target
            elif (scheduled and market_ok and self.multiplier < 1
                  and position - self.last_change >= cfg.recovery_days):
                self.multiplier = min(1.0, self.multiplier + 0.25)
                # Reset only the recovery controller. Lifetime drawdown never resets.
                self.anchor = equity
        if previous != self.multiplier:
            self.last_change = position
        return self.multiplier < previous
