"""Prompt assembly for AI scan analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Any

from qkquant.ai.base import AiRequest

SYSTEM_PROMPT = """你是 qkquant 的 A 股量化信号分析助手。

边界:
- 只解释已有策略扫描结果、排序候选和提示风险。
- 不生成新的 BUY/SELL 信号，不改变仓位，不替代交易决策。
- 不编造新闻、财报、资金流或盘口信息；输入里没有的信息必须明确说没有。
- 输出要简洁、纪律化，强调滑点、涨跌停、追高和持仓先看卖出信号。
"""


def _json_default(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def request_to_payload(request: AiRequest) -> dict[str, Any]:
    return asdict(request)


def build_user_prompt(request: AiRequest) -> str:
    payload = request_to_payload(request)
    data = json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2)
    return f"""请根据下面的结构化扫描结果，输出 Markdown 分析。

固定使用 4 段:
1. 今日结论
2. 重点候选
3. 风险提示
4. 明日执行清单

扫描结果:
```json
{data}
```
"""


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "request_to_payload"]
