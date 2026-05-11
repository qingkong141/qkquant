from __future__ import annotations

from datetime import date

from qkquant.ai import (
    AiRuntimeConfig,
    analyze_raw_signals,
    build_raw_ai_request,
    format_ai_section,
    load_ai_config,
)
from qkquant.ai.base import AiRequest


def _sample_results() -> dict[str, dict]:
    return {
        "momentum": {
            "buys": [
                {
                    "code": "000001",
                    "buy_reason": "momentum_entry",
                    "score": 0.08,
                    "metrics": {"close": 12.3, "mom_20d": 0.08},
                },
                {
                    "code": "000002",
                    "buy_reason": "momentum_entry",
                    "score": 0.04,
                    "metrics": {"close": 9.8, "mom_20d": 0.04},
                },
            ],
            "sells": [
                {
                    "code": "600000",
                    "sell_reason": "momentum_exit",
                    "score": -0.04,
                    "metrics": {"close": 10.0, "mom_20d": -0.04},
                }
            ],
        },
        "ma_boll": {
            "buys": [
                {
                    "code": "000001",
                    "buy_reason": "ma_boll_entry",
                    "score": 0.03,
                    "metrics": {"close": 12.3, "fast_over_slow": 0.03},
                }
            ],
            "sells": [],
        },
    }


def test_build_raw_ai_request_aggregates_resonance_and_risks():
    holdings = {"600000": {"qty": 100, "cost": 10.5}}
    request = build_raw_ai_request(
        _sample_results(),
        holdings,
        name_map={"000001": "平安银行", "600000": "浦发银行"},
        as_of=date(2026, 5, 11),
        strategies=["momentum", "ma_boll"],
        max_candidates=8,
        universe_size=300,
    )

    assert request.as_of == date(2026, 5, 11)
    assert request.universe_size == 300
    assert request.candidates[0].code == "000001"
    assert request.candidates[0].resonance == 2
    assert request.candidates[0].name == "平安银行"
    assert request.holding_risks[0].code == "600000"
    assert request.holding_risks[0].qty == 100


def test_noop_provider_returns_ai_markdown_section():
    response = analyze_raw_signals(
        _sample_results(),
        {"600000": {"qty": 100, "cost": 10.5}},
        as_of=date(2026, 5, 11),
        config=AiRuntimeConfig(provider="noop", max_candidates=1),
    )
    section = format_ai_section(response)

    assert response.ok
    assert "## AI 分析" in section
    assert "今日结论" in section
    assert "000001" in section
    assert "000002" not in section


def test_provider_failure_degrades_to_unavailable_section():
    class BrokenProvider:
        name = "broken"
        model = "test"

        def analyze(self, request: AiRequest):  # noqa: ANN202
            raise RuntimeError("boom")

    response = analyze_raw_signals(
        _sample_results(),
        {},
        as_of=date(2026, 5, 11),
        provider=BrokenProvider(),
    )
    section = format_ai_section(response)

    assert not response.ok
    assert "AI 分析不可用" in section
    assert "boom" in section


def test_load_ai_config_reads_non_secret_values(tmp_path):
    path = tmp_path / "ai.yaml"
    path.write_text(
        "\n".join(
            [
                "enabled: true",
                "provider: openai_compatible",
                "model: deepseek-chat",
                "base_url: https://api.deepseek.com/v1",
                "api_key_env: QKQUANT_DEEPSEEK_API_KEY",
                "timeout_seconds: 10",
                "max_candidates: 5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_ai_config(path)

    assert cfg.enabled is True
    assert cfg.provider == "openai_compatible"
    assert cfg.model == "deepseek-chat"
    assert cfg.api_key_env == "QKQUANT_DEEPSEEK_API_KEY"
    assert cfg.max_candidates == 5


def test_load_ai_config_merges_local_secret_override(tmp_path):
    path = tmp_path / "ai.yaml"
    local_path = tmp_path / "ai.local.yaml"
    path.write_text(
        "\n".join(
            [
                "enabled: true",
                "provider: openai_compatible",
                "model: deepseek-chat",
                "base_url: https://api.deepseek.com/v1",
                "api_key_env: QKQUANT_DEEPSEEK_API_KEY",
            ]
        ),
        encoding="utf-8",
    )
    local_path.write_text('api_key: "local-secret"\nmax_candidates: 3\n', encoding="utf-8")

    cfg = load_ai_config(path, local_path=local_path)

    assert cfg.api_key == "local-secret"
    assert cfg.api_key_env == "QKQUANT_DEEPSEEK_API_KEY"
    assert cfg.max_candidates == 3
