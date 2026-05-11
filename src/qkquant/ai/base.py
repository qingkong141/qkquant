"""AI analysis contracts.

The AI layer is intentionally read-only: it explains existing scan signals and
never creates or mutates trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol


@dataclass(slots=True)
class AiCandidate:
    code: str
    name: str | None = None
    strategies: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    is_holding: bool = False

    @property
    def label(self) -> str:
        return f"{self.code} {self.name}" if self.name else self.code

    @property
    def resonance(self) -> int:
        return len(self.strategies)

    @property
    def best_score(self) -> float:
        return max(self.scores.values(), default=0.0)


@dataclass(slots=True)
class AiHoldingRisk:
    code: str
    name: str | None = None
    qty: Any = "?"
    cost: Any = "?"
    strategies: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.code} {self.name}" if self.name else self.code


@dataclass(slots=True)
class AiRequest:
    as_of: date
    strategies: list[str]
    candidates: list[AiCandidate] = field(default_factory=list)
    holding_risks: list[AiHoldingRisk] = field(default_factory=list)
    holdings_count: int = 0
    universe_size: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AiResponse:
    markdown: str
    provider: str
    model: str = ""
    ok: bool = True
    error: str | None = None


class AiProvider(Protocol):
    name: str
    model: str

    def analyze(self, request: AiRequest) -> AiResponse:
        """Return a markdown analysis for an existing scan result."""


__all__ = [
    "AiCandidate",
    "AiHoldingRisk",
    "AiProvider",
    "AiRequest",
    "AiResponse",
]
