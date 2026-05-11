"""Deterministic no-network AI provider.

This keeps `qkquant scan --raw --ai` useful before any real model API is
configured, and gives tests a stable provider.
"""

from __future__ import annotations

from qkquant.ai.base import AiRequest, AiResponse


class NoopAiProvider:
    name = "noop"
    model = "rule_based_summary"

    def analyze(self, request: AiRequest) -> AiResponse:
        lines: list[str] = []
        candidates = request.candidates
        risks = request.holding_risks
        multi = [c for c in candidates if c.resonance >= 2]

        lines.append("### 今日结论")
        if candidates:
            lines.append(
                f"- 今日共有 {len(candidates)} 只候选进入 AI 摘要范围，"
                f"其中 {len(multi)} 只出现多策略共振。"
            )
        else:
            lines.append("- 今日没有 BUY 候选，优先观察持仓风险和市场环境。")
        if risks:
            lines.append(f"- 你的持仓/watchlist 中有 {len(risks)} 只触发出场或风险提示。")
        lines.append("")

        lines.append("### 重点候选")
        if candidates:
            for c in candidates[:5]:
                source = " / ".join(c.strategies) if c.strategies else "unknown"
                tag = "，已在持仓/watchlist" if c.is_holding else ""
                lines.append(f"- {c.label}: {c.resonance} 策略命中（{source}）{tag}。")
        else:
            lines.append("- 无。")
        lines.append("")

        lines.append("### 风险提示")
        if risks:
            for r in risks[:5]:
                source = " / ".join(r.strategies) if r.strategies else "unknown"
                reason = " / ".join(v for v in r.reasons.values() if v) or "risk"
                lines.append(f"- {r.label}: {source} 触发 {reason}，先处理持仓风险。")
        else:
            lines.append("- 未发现持仓/watchlist 出场信号，但裸信号不包含风控熔断和 T+1 约束。")
        lines.append("- BUY 信号基于收盘条件，次日开盘可能出现滑点或涨停买不进。")
        lines.append("")

        lines.append("### 明日执行清单")
        lines.append("- 先看持仓 SELL，再看多策略共振 BUY。")
        lines.append("- 开盘大幅高开、接近一字板或流动性异常时，不按收盘信号机械追入。")
        lines.append("- 单策略命中的候选只适合观察，除非价格和风险回报仍然合理。")

        return AiResponse(
            markdown="\n".join(lines),
            provider=self.name,
            model=self.model,
        )


__all__ = ["NoopAiProvider"]
