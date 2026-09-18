"""Manual paper-account close records using the portfolio backtest's decision rule.

No orders are sent or fills invented. Each new close after initialization needs
an explicit cash/position snapshot, including any dividends or share changes.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qkquant.etf_drawdown import DrawdownConfig, DrawdownState
from qkquant.etf_portfolio_backtest import close_decision, prepare_inputs
from qkquant.etf_signal import EtfSignalConfig


def _rule_revision() -> dict[str, str]:
    root = Path(__file__).parent
    files = ("etf_portfolio_backtest.py", "etf_chan.py", "etf_drawdown.py",
             "etf_signal.py", "etf_paper.py", "factors/library.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def create_paper_account(path: Path, capital: float = 100_000) -> dict:
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be finite and positive")
    account = {
        "schema": 1, "kind": "manual_paper", "capital": capital,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(EtfSignalConfig(capital=capital)),
        "risk_config": asdict(DrawdownConfig()),
        "rebalance_days": 10, "rebalance_offset": 0, "daily_entries": False,
        "rule_revision": _rule_revision(),
        "records": [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(account, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return account


def _snapshot_values(snapshot: dict, codes: list[str]) -> tuple[float, dict[str, int]]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("holdings"), dict):
        raise ValueError("snapshot requires cash and a holdings object mapping codes to integer quantities")
    cash = float(snapshot["cash"])
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be finite and nonnegative")
    holdings = {}
    for code, qty in snapshot["holdings"].items():
        if code not in codes or isinstance(qty, bool) or not isinstance(qty, int) or qty < 0:
            raise ValueError(f"invalid paper holding: {code}")
        if qty:
            holdings[code] = qty
    return cash, holdings


def record_paper_close(path: Path, data, as_of: str, snapshot: dict | None = None) -> dict:
    """Append one close atomically. Repeated identical dates are read-only/idempotent."""
    path = Path(path)
    lock = path.with_suffix(path.suffix + ".lock")
    # An interrupted writer leaves an explicit lock for inspection, never a
    # partially replaced account file. Concurrent planners must not lose records.
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        account = json.loads(path.read_text(encoding="utf-8"))
        count = len(account["records"])
        result = _record_close(account, data, as_of, snapshot)
        if len(account["records"]) == count:
            return result
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(account, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return result
    finally:
        lock.unlink()


def _record_close(account: dict, data, as_of: str, snapshot: dict | None) -> dict:
    if account.get("schema") != 1 or account.get("kind") != "manual_paper":
        raise ValueError("expected a manual paper account; real account files are unsupported")
    if account.get("rule_revision") != _rule_revision():
        raise ValueError("decision source changed; preserve this account and review a new version")
    cfg_data = dict(account["config"])
    cfg_data["chan_entry_states"] = tuple(cfg_data["chan_entry_states"])
    cfg = EtfSignalConfig(**cfg_data)
    if cfg != EtfSignalConfig(capital=account["capital"]) or account["risk_config"] != asdict(DrawdownConfig()):
        raise ValueError("paper account must use the fixed strategy/risk configuration")
    if (account["rebalance_days"], account["rebalance_offset"], account["daily_entries"]) != (10, 0, False):
        raise ValueError("paper account must use the fixed rebalance schedule")
    day = pd.Timestamp(as_of)
    close = data.panel["close"]
    if day not in close.index:
        raise ValueError("requested close is absent from the verified trading calendar")
    pos = int(close.index.get_loc(day))
    if pos < 120:
        raise ValueError("120 trading sessions of indicator warmup required")
    if account.get("codes", data.codes) != data.codes:
        raise ValueError("paper account universe changed; reconcile before continuing")
    records = account["records"]
    previous = records[-1] if records else None
    if previous and day == pd.Timestamp(previous["as_of"]):
        if snapshot is not None:
            cash, holdings = _snapshot_values(snapshot, data.codes)
            if cash != previous["cash"] or holdings != previous["holdings"]:
                raise ValueError("an archived close cannot be overwritten")
        return previous
    if previous:
        prior = pd.Timestamp(previous["as_of"])
        if prior not in close.index or int(close.index.get_loc(prior)) != previous["position"]:
            raise ValueError("historical trading calendar changed")
        if pos != previous["position"] + 1:
            raise ValueError("record every trading close in sequence; skipped dates reset risk history")
        if snapshot is None:
            raise ValueError("supply an explicit cash/holdings snapshot; no fills are assumed")
    elif snapshot is None:
        snapshot = {"cash": account["capital"], "holdings": {}}
    cash, holdings = _snapshot_values(snapshot, data.codes)
    if not previous and (cash != account["capital"] or holdings):
        raise ValueError("a new paper account starts with its initial cash and no positions")
    raw = data.raw_panel["close"].loc[day]
    if any(pd.isna(raw.get(code)) or raw[code] <= 0 for code in holdings):
        raise ValueError("held ETF lacks a valid raw closing price; account cannot be valued")
    invested = sum(qty * float(raw[code]) for code, qty in holdings.items())
    equity = cash + invested
    if equity <= 0:
        raise ValueError("paper equity must be positive")
    state = DrawdownState(**previous["drawdown_state"]) if previous else DrawdownState(equity, equity)
    instruments = data.store.load_instruments(data.codes).set_index("code")
    decision = close_decision(
        prepare_inputs({key: frame.loc[:day] for key, frame in data.panel.items()}),
        instruments, pos, holdings, equity, invested, state, cfg, DrawdownConfig(),
        rebalance_days=10, rebalance_offset=0, daily_entries=False,
    )
    targets = []
    pending = decision["pending"]
    if pending:
        selected, exposure, reduce_only, _, _ = pending
        weight = exposure / len(selected) if selected else 0.0
        for code in sorted(set(holdings) | set(selected)):
            price = raw.get(code)
            if pd.isna(price) or price <= 0:
                targets.append({"code": code, "status": "missing_raw_close", "target_weight": None})
                continue
            current_weight = holdings.get(code, 0) * float(price) / equity
            target_weight = weight if code in selected else 0.0
            allow_increase = not reduce_only and code not in decision["continuing"]
            targets.append({
                "code": code, "status": "target_only", "current_qty": holdings.get(code, 0),
                "reference_raw_close": float(price), "current_weight": current_weight,
                "target_weight": target_weight,
                "allow_increase": allow_increase,
                "max_qty": None if allow_increase else holdings.get(code, 0),
                "reason": "risk_reduce" if reduce_only else "scheduled_rebalance",
            })
    record = {
        "as_of": str(day.date()), "position": pos,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "cash": cash, "holdings": holdings, "equity": equity, "invested": invested,
        "drawdown_state": asdict(state), "drawdown": 1 - equity / state.peak,
        "scheduled": bool(decision["scheduled"]),
        "exposure_cap": DrawdownConfig().max_exposure * state.multiplier,
        "targets": targets, "execution_session": "next_trading_day_open",
        "source": data.metadata,
        "accounting": "manual_cash_and_positions_raw_close; no_automatic_fills",
    }
    account["codes"] = data.codes
    records.append(record)
    return record


def format_paper_plan(record: dict) -> str:
    lines = [
        f"ETF 人工模拟账户计划 | {record['as_of']}",
        f"净值 {record['equity']:.2f} | 历史回撤 {record['drawdown']:.2%} | 风险仓位上限 {record['exposure_cap']:.0%}",
        f"调仓日: {'是' if record['scheduled'] else '否'} | 账户暂停: {'是' if record['drawdown_state']['halted'] else '否'}",
        "目标供下一交易日开盘核对；参考价为未复权收盘价。没有自动成交或真实下单。",
    ]
    for row in record["targets"]:
        weight = row["target_weight"]
        text = "行情缺失，禁止据此下单" if weight is None else f"目标 {weight:.2%} ({row['reason']})"
        if row.get("max_qty") is not None:
            text += f"；份额最多 {row['max_qty']}，禁止补仓"
        lines.append(f"{row['code']}: {text}")
    if not record["targets"]:
        lines.append("本日无调仓目标；下一收盘仍需记录现金和持仓以检查回撤。")
    return "\n".join(lines)
