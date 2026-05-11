"""AI analysis orchestration for scan results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from qkquant.ai.base import (
    AiCandidate,
    AiHoldingRisk,
    AiProvider,
    AiRequest,
    AiResponse,
)
from qkquant.ai.providers import NoopAiProvider, OpenAICompatibleProvider
from qkquant.config import PROJECT_ROOT


@dataclass(slots=True)
class AiRuntimeConfig:
    enabled: bool = False
    provider: str = "noop"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "QKQUANT_AI_API_KEY"
    timeout_seconds: int = 30
    max_candidates: int = 8


DEFAULT_AI_CONFIG_PATH = PROJECT_ROOT / "config" / "ai.yaml"
LOCAL_AI_CONFIG_PATH = PROJECT_ROOT / "config" / "ai.local.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "ai" in raw and isinstance(raw["ai"], dict):
        raw = raw["ai"]
    return raw


def load_ai_config(
    path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> AiRuntimeConfig:
    config_path = Path(path) if path else DEFAULT_AI_CONFIG_PATH
    raw = _load_yaml(config_path)
    if local_path is None and path is None:
        local_config_path = LOCAL_AI_CONFIG_PATH
    elif local_path is None:
        local_config_path = config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")
    else:
        local_config_path = Path(local_path)
    raw.update(_load_yaml(local_config_path))
    allowed = set(AiRuntimeConfig.__dataclass_fields__)
    data = {k: v for k, v in raw.items() if k in allowed}
    return AiRuntimeConfig(**data)


def make_ai_provider(config: AiRuntimeConfig) -> AiProvider:
    provider = config.provider.strip().lower()
    if provider == "noop":
        return NoopAiProvider()
    if provider in {"openai_compatible", "openai-compatible", "openai"}:
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=config.api_key or os.getenv(config.api_key_env),
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    return NoopAiProvider()


def _name(code: str, name_map: dict[str, str] | None) -> str | None:
    return (name_map or {}).get(code)


def build_raw_ai_request(
    results: dict[str, dict],
    holdings: dict[str, dict],
    *,
    name_map: dict[str, str] | None = None,
    as_of: date,
    strategies: list[str] | None = None,
    max_candidates: int = 8,
    universe_size: int | None = None,
) -> AiRequest:
    by_code: dict[str, AiCandidate] = {}
    risks_by_code: dict[str, AiHoldingRisk] = {}

    for strategy, result in results.items():
        for row in result.get("buys", []):
            code = row["code"]
            candidate = by_code.setdefault(
                code,
                AiCandidate(
                    code=code,
                    name=_name(code, name_map),
                    is_holding=code in holdings,
                ),
            )
            candidate.strategies.append(strategy)
            if row.get("buy_reason"):
                candidate.reasons[strategy] = row["buy_reason"]
            candidate.scores[strategy] = float(row.get("score") or 0.0)
            candidate.metrics[strategy] = row.get("metrics") or {}

        for row in result.get("sells", []):
            code = row["code"]
            holding = holdings.get(code, {})
            risk = risks_by_code.setdefault(
                code,
                AiHoldingRisk(
                    code=code,
                    name=_name(code, name_map),
                    qty=holding.get("qty", "?"),
                    cost=holding.get("cost", "?"),
                ),
            )
            risk.strategies.append(strategy)
            if row.get("sell_reason"):
                risk.reasons[strategy] = row["sell_reason"]
            risk.metrics[strategy] = row.get("metrics") or {}

    candidates = sorted(
        by_code.values(),
        key=lambda c: (-c.resonance, -c.best_score, c.code),
    )[:max_candidates]
    holding_risks = sorted(
        risks_by_code.values(),
        key=lambda r: (-len(r.strategies), r.code),
    )

    return AiRequest(
        as_of=as_of,
        strategies=strategies or list(results),
        candidates=candidates,
        holding_risks=holding_risks,
        holdings_count=len(holdings),
        universe_size=universe_size,
        notes=[
            "scan_raw ignores simulated portfolio state, risk cooldown, T+1, and limit-up/limit-down rejections",
            "signals are based on close prices; next open may have slippage",
            "multi-strategy resonance is usually more reliable than single-strategy hits",
        ],
    )


def analyze_raw_signals(
    results: dict[str, dict],
    holdings: dict[str, dict],
    *,
    name_map: dict[str, str] | None = None,
    as_of: date,
    strategies: list[str] | None = None,
    universe_size: int | None = None,
    config: AiRuntimeConfig | None = None,
    provider: AiProvider | None = None,
) -> AiResponse:
    cfg = config or AiRuntimeConfig()
    request = build_raw_ai_request(
        results,
        holdings,
        name_map=name_map,
        as_of=as_of,
        strategies=strategies,
        max_candidates=max(1, cfg.max_candidates),
        universe_size=universe_size,
    )
    active_provider = provider or make_ai_provider(cfg)
    try:
        return active_provider.analyze(request)
    except Exception as exc:
        provider_name = getattr(active_provider, "name", "unknown")
        model = getattr(active_provider, "model", "")
        return AiResponse(
            markdown="",
            provider=provider_name,
            model=model,
            ok=False,
            error=str(exc),
        )


def format_ai_section(response: AiResponse) -> str:
    if response.ok and response.markdown.strip():
        return f"## AI 分析\n\n{response.markdown.strip()}"
    detail = f": {response.error}" if response.error else ""
    return f"## AI 分析\n\nAI 分析不可用{detail}。原始扫描结果不受影响。"


__all__ = [
    "AiRuntimeConfig",
    "analyze_raw_signals",
    "build_raw_ai_request",
    "format_ai_section",
    "load_ai_config",
    "make_ai_provider",
]
