"""信号绩效分析：解析 scan_raw 报告，模拟'完全按推送操作'的每笔交易收益。

对每条 BUY 信号买入，逐日推进策略出场条件直到 sell=True，计算收益率。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qkquant.config import get_settings, PROJECT_ROOT
from qkquant.data.storage import DuckStore
from qkquant.factors.indicators import adx as compute_adx
from qkquant.logger import logger
from qkquant.scan import (
    _raw_momentum,
    _raw_ma_boll,
    _raw_ma_breakout,
    _raw_momentum_breakout,
)
from qkquant.strategy.registry import get_strategy, load_strategy_config

# ── 策略 → raw 函数映射 ────────────────────────────────────────────
RAW_FUNC_MAP = {
    "momentum": _raw_momentum,
    "ma_boll": _raw_ma_boll,
    "ma_breakout": _raw_ma_breakout,
    "momentum_breakout": _raw_momentum_breakout,
}


# ── 数据结构 ────────────────────────────────────────────────────────


@dataclass
class SignalEntry:
    strategy: str
    code: str
    name: str
    buy_date: date
    buy_price: float


@dataclass
class Trade:
    strategy: str
    code: str
    name: str
    buy_date: date
    buy_price: float
    sell_date: Optional[date] = None
    sell_price: Optional[float] = None
    sell_reason: Optional[str] = None
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None
    is_open: bool = False


# ── 报告解析 ────────────────────────────────────────────────────────

_HEADER_DATE = re.compile(r"^# qkquant 裸信号扫描 \| (\d{4}-\d{2}-\d{2})")
_STRATEGY_HDR = re.compile(r"^## (\S+) ")
_BUY_SECTION = re.compile(r"^### \[BUY\]")
_SELL_OR_REJ_SECTION = re.compile(r"^### \[(SELL|REJECTED)\]")
_STOCK_LINE = re.compile(r"^\s{2}-\s(\d{6})\s+(.+?)(?:\s{2}\[.*?\])?\s*$")
_CLOSE_RE = re.compile(r"close=([\d.]+)")


def parse_scan_reports(reports_dir: Path) -> dict[str, list[SignalEntry]]:
    """解析所有 scan_*_raw.md，按策略分组返回 SignalEntry 列表。"""
    signals: dict[str, list[SignalEntry]] = {}

    for fpath in sorted(reports_dir.glob("scan_*_raw.md")):
        # 从文件名提取日期（更可靠）
        m_file = re.search(r"scan_(\d{4}-\d{2}-\d{2})_raw\.md", fpath.name)
        if not m_file:
            continue
        report_date = date.fromisoformat(m_file.group(1))

        text = fpath.read_text(encoding="utf-8")
        current_strategy: Optional[str] = None
        in_buy = False
        pending_code: Optional[str] = None
        pending_name: str = ""

        for line in text.splitlines():
            # 策略标题
            m = _STRATEGY_HDR.match(line)
            if m:
                sname = m.group(1)
                if sname in RAW_FUNC_MAP:
                    current_strategy = sname
                    in_buy = False
                else:
                    current_strategy = None
                continue

            # BUY 子节
            if _BUY_SECTION.match(line):
                in_buy = True
                pending_code = None
                continue

            # SELL / REJECTED 子节结束 BUY 上下文
            if _SELL_OR_REJ_SECTION.match(line):
                in_buy = False
                pending_code = None
                continue

            # 页脚（--- 之后不解析）
            if line.strip() == "---":
                in_buy = False
                pending_code = None
                continue

            if not in_buy or current_strategy is None:
                continue

            # 股票行
            m_s = _STOCK_LINE.match(line)
            if m_s:
                pending_code = m_s.group(1)
                pending_name = m_s.group(2).strip()
                continue

            # 指标行（close= 出现时才消费 pending_code）
            if pending_code is not None:
                m_c = _CLOSE_RE.search(line)
                if m_c:
                    entry = SignalEntry(
                        strategy=current_strategy,
                        code=pending_code,
                        name=pending_name,
                        buy_date=report_date,
                        buy_price=float(m_c.group(1)),
                    )
                    signals.setdefault(current_strategy, []).append(entry)
                pending_code = None

    # 按 (buy_date, code) 排序
    for sname in signals:
        signals[sname].sort(key=lambda e: (e.buy_date, e.code))

    return signals


# ── 策略参数加载 ─────────────────────────────────────────────────────


def _load_params(strategy: str) -> dict:
    """加载策略参数，兼容已注册和未注册的策略。"""
    try:
        info = get_strategy(strategy)
        cfg = load_strategy_config(info)
        return (cfg.get("params") or {}) if cfg else {}
    except KeyError:
        config_path = PROJECT_ROOT / "config" / "strategies" / f"{strategy}.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("params") or {}
        return {}


# ── 出场模拟 ────────────────────────────────────────────────────────


def _simulate_exit(
    strategy: str,
    code: str,
    df: pd.DataFrame,
    params: dict,
) -> tuple[Optional[date], Optional[float], Optional[str]]:
    """从 df（已截取到买入日及之后）逐日推进，直到 sell=True。

    Returns (sell_date, sell_price, sell_reason) 或 (None, None, None) 表示仍持有。
    """
    raw_func = RAW_FUNC_MAP[strategy]
    needed = _min_bars(strategy, params)
    # 注入 code 供市值过滤（momentum / momentum_breakout）
    if strategy in ("momentum", "momentum_breakout"):
        params = {**params, "_code": code}

    # 从买入次日开始检查（T+1）
    for i in range(1, len(df)):
        hist = df.iloc[: i + 1].reset_index(drop=True)
        if len(hist) < needed:
            continue

        result = raw_func(hist, params)
        if result.get("sell"):
            sell_date = pd.Timestamp(df.iloc[i]["trade_date"]).date()
            sell_price = float(df.iloc[i]["close"])
            return sell_date, sell_price, result.get("sell_reason")

    return None, None, None


def _min_bars(strategy: str, params: dict) -> int:
    """策略所需最小数据条数。"""
    if strategy == "momentum":
        return int(params.get("mom_window", 20)) + 1
    elif strategy == "momentum_breakout":
        return max(int(params.get("mom_window", 20)), int(params.get("breakout_window", 10))) + 1
    elif strategy == "ma_breakout":
        slow = int(params.get("slow", 20))
        trend = int(params.get("trend_period", 60)) if params.get("trend_filter", True) else 0
        return max(slow, trend) + 1
    elif strategy == "ma_boll":
        slow = int(params.get("slow", 20))
        boll = int(params.get("boll_period", 20))
        return max(slow, boll) + 1
    return 30


# ── 主模拟逻辑 ──────────────────────────────────────────────────────


def simulate_trades(
    signals: dict[str, list[SignalEntry]],
    store: DuckStore,
    adjust: str,
) -> list[Trade]:
    trades: list[Trade] = []

    all_dates: set[date] = set()
    for entries in signals.values():
        for e in entries:
            all_dates.add(e.buy_date)
    if not all_dates:
        return trades

    global_start = min(all_dates)
    latest_data = store.conn.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()
    global_end = latest_data[0] if latest_data and latest_data[0] else date.today()
    # 留出指标计算余量
    buffer_start = global_start - pd.Timedelta(days=120)

    # 加载市值数据（momentum 系列策略需要）
    all_codes: set[str] = set()
    for entries in signals.values():
        for e in entries:
            all_codes.add(e.code)
    market_caps = store.load_market_caps(list(all_codes)) if all_codes else {}

    for strategy, entries in signals.items():
        if strategy not in RAW_FUNC_MAP:
            logger.warning(f"skip unsupported strategy: {strategy}")
            continue

        params = _load_params(strategy)
        if market_caps and strategy in ("momentum", "momentum_breakout"):
            params["market_caps"] = market_caps
        cooldown_days = int(params.get("cooldown_days", 0))
        cooldown_until: dict[str, date] = {}  # code → 冷却截止日

        codes = list({e.code for e in entries})
        logger.info(f"[{strategy}] loading {len(codes)} codes from {buffer_start} to {global_end} ...")
        df_all = store.load_daily(codes=codes, start=buffer_start, end=global_end, adjust=adjust)
        if df_all.empty:
            logger.warning(f"[{strategy}] no price data; skip")
            continue

        price_data: dict[str, pd.DataFrame] = {}
        for c, g in df_all.groupby("code"):
            price_data[c] = g.sort_values("trade_date").reset_index(drop=True)

        # 按 code 分组，仓位状态机
        entries_by_code: dict[str, list[SignalEntry]] = {}
        for e in entries:
            entries_by_code.setdefault(e.code, []).append(e)

        for code, code_entries in entries_by_code.items():
            if code not in price_data:
                logger.warning(f"[{strategy}] {code} no price data; skip")
                continue

            df_code = price_data[code]
            in_position = False

            for entry in code_entries:
                if in_position:
                    continue  # 持仓中跳过后续买入信号

                # 定位买入日在 df 中的位置
                entry_dt = pd.Timestamp(entry.buy_date)
                mask = df_code["trade_date"] == entry_dt
                if not mask.any():
                    logger.warning(f"[{strategy}] {code} buy date {entry.buy_date} not in data; skip")
                    continue

                idx = int(mask.idxmax())
                # 用 DB 实际收盘价代替报告里的价格，保证买卖同复权口径
                actual_buy_price = float(df_code.iloc[idx]["close"])

                # ADX 趋势强度事后筛查：若买入日 ADX 低于阈值，说明当日震荡，跳过
                adx_threshold = float(params.get("adx_threshold", 0))
                if adx_threshold > 0:
                    df_to_entry = df_code.iloc[: idx + 1].reset_index(drop=True)
                    try:
                        adx_df = compute_adx(
                            df_to_entry["high"], df_to_entry["low"], df_to_entry["close"]
                        )
                        if not adx_df.empty:
                            adx_val = adx_df["adx"].iloc[-1]
                            if not pd.isna(adx_val) and adx_val < adx_threshold:
                                logger.info(
                                    f"[{strategy}] {code} ADX={adx_val:.1f}<{adx_threshold} on {entry.buy_date}; skip"
                                )
                                continue
                    except Exception:
                        pass  # ADX 计算失败不阻塞

                # 冷却期过滤：卖出后 N 天内不重新买入
                if cooldown_days > 0 and code in cooldown_until:
                    if entry.buy_date <= cooldown_until[code]:
                        logger.info(
                            f"[{strategy}] {code} cooldown until {cooldown_until[code]}; skip {entry.buy_date}"
                        )
                        continue
                    else:
                        del cooldown_until[code]

                df_from_entry = df_code.iloc[idx:].reset_index(drop=True)

                sell_date, sell_price, sell_reason = _simulate_exit(
                    strategy, code, df_from_entry, params
                )

                if sell_date is not None:
                    ret = (sell_price / actual_buy_price - 1.0) * 100
                    days = (sell_date - entry.buy_date).days
                    trades.append(Trade(
                        strategy=strategy, code=code, name=entry.name,
                        buy_date=entry.buy_date, buy_price=actual_buy_price,
                        sell_date=sell_date, sell_price=sell_price,
                        sell_reason=sell_reason, return_pct=ret,
                        holding_days=days, is_open=False,
                    ))
                    in_position = False
                    # 设置冷却期：卖出后 N 天内不得重新买入
                    if cooldown_days > 0:
                        cooldown_until[code] = sell_date + timedelta(days=cooldown_days)
                else:
                    last_close = float(df_code.iloc[-1]["close"])
                    last_date = pd.Timestamp(df_code.iloc[-1]["trade_date"]).date()
                    ret = (last_close / actual_buy_price - 1.0) * 100
                    days = (last_date - entry.buy_date).days
                    trades.append(Trade(
                        strategy=strategy, code=code, name=entry.name,
                        buy_date=entry.buy_date, buy_price=actual_buy_price,
                        sell_date=last_date, sell_price=last_close,
                        sell_reason="(still open)", return_pct=ret,
                        holding_days=days, is_open=True,
                    ))
                    in_position = True

    trades.sort(key=lambda t: (t.strategy, t.buy_date, t.code))
    return trades


# ── 报告输出 ────────────────────────────────────────────────────────


def generate_report(trades: list[Trade], adjust: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# Signal Performance Report | {date.today()}")
    lines.append(f"Generated: {now}  |  Adjust: {adjust}")
    lines.append("")
    lines.append('> 模拟"完全按推送操作"：对每条扫描报告的买入信号按收盘价入场，')
    lines.append('> 逐日推进策略出场条件判定，触发即卖出。未触发卖出的标记为持仓中。')
    lines.append("")

    if not trades:
        lines.append("(无交易记录)")
        return "\n".join(lines)

    # ── 按策略汇总 ──
    lines.append("## 策略汇总")
    lines.append("")
    lines.append("| 策略 | 交易 | 已完成 | 胜 | 负 | 胜率 | 平均收益 | 总收益 |")
    lines.append("|------|-----:|------:|---:|---|-----:|--------:|------:|")

    by_strategy: dict[str, list[Trade]] = {}
    for t in trades:
        by_strategy.setdefault(t.strategy, []).append(t)

    total_wins = 0
    total_losses = 0
    total_completed = 0
    sum_ret_completed = 0.0

    for sname in sorted(by_strategy):
        st = by_strategy[sname]
        completed = [t for t in st if not t.is_open]
        wins = [t for t in completed if (t.return_pct or 0) > 0]
        losses = [t for t in completed if (t.return_pct or 0) < 0]
        n = len(st)
        nc = len(completed)
        wr = f"{len(wins) / nc * 100:.0f}%" if nc > 0 else "-"
        avg_ret = sum(t.return_pct or 0 for t in completed) / nc if nc > 0 else 0.0
        total_ret = sum(t.return_pct or 0 for t in completed)

        lines.append(
            f"| {sname} | {n} | {nc} | {len(wins)} | {len(losses)} | "
            f"{wr} | {avg_ret:+.2f}% | {total_ret:+.2f}% |"
        )

        total_wins += len(wins)
        total_losses += len(losses)
        total_completed += nc
        sum_ret_completed += total_ret

    lines.append("")

    # ── 每笔明细 ──
    lines.append("## 每笔明细")
    lines.append("")
    lines.append("| # | 策略 | 代码 | 名称 | 买入日 | 买入价 | 卖出日 | 卖出价 | 收益 | 持日 | 原因 |")
    lines.append("|---|------|------|------|--------|--------|--------|--------|------|-----|------|")

    for i, t in enumerate(trades, 1):
        open_tag = " 🔓" if t.is_open else ""
        ret_str = f"**{t.return_pct:+.2f}%**{open_tag}" if t.return_pct is not None else "-"
        lines.append(
            f"| {i} | {t.strategy} | {t.code} | {t.name} | "
            f"{t.buy_date} | {t.buy_price:.2f} | "
            f"{t.sell_date or '-'} | {t.sell_price:.2f} | "
            f"{ret_str} | {t.holding_days or '-'}d | {t.sell_reason or '-'} |"
        )
    lines.append("")

    # ── 持仓中 ──
    open_trades = [t for t in trades if t.is_open]
    if open_trades:
        lines.append("## 持仓中 (未触发卖出)")
        lines.append("")
        lines.append("| 策略 | 代码 | 名称 | 买入日 | 买入价 | 最新价 | 未实现收益 | 持日 |")
        lines.append("|------|------|------|--------|--------|--------|----------:|-----|")
        for t in open_trades:
            lines.append(
                f"| {t.strategy} | {t.code} | {t.name} | {t.buy_date} | "
                f"{t.buy_price:.2f} | {t.sell_price:.2f} | "
                f"**{t.return_pct:+.2f}%** | {t.holding_days}d |"
            )
        lines.append("")

    # ── 总汇总 ──
    all_completed = [t for t in trades if not t.is_open]
    lines.append("## 总汇总")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 总交易 | {len(trades)} |")
    lines.append(f"| 已完成 | {total_completed} |")
    lines.append(f"| 持仓中 | {len(open_trades)} |")
    lines.append(f"| 胜 | {total_wins} |")
    lines.append(f"| 负 | {total_losses} |")
    all_wr = f"{total_wins / total_completed * 100:.1f}%" if total_completed > 0 else "-"
    lines.append(f"| 胜率 | {all_wr} |")
    all_avg = sum_ret_completed / total_completed if total_completed > 0 else 0.0
    lines.append(f"| 平均收益 (已完成) | {all_avg:+.2f}% |")
    lines.append(f"| 总收益 (已完成) | {sum_ret_completed:+.2f}% |")
    if all_completed:
        avg_hold = sum(t.holding_days or 0 for t in all_completed) / len(all_completed)
        lines.append(f"| 平均持日 | {avg_hold:.0f}d |")
    lines.append("")
    lines.append("---")
    lines.append("注意: 收益基于当日收盘价，未扣除佣金/滑点/印花税。")
    lines.append("注意: 信号可能重叠，同一股票跨策略独立核算。")

    return "\n".join(lines)


# ── 入口 ─────────────────────────────────────────────────────────────


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    adjust = get_settings().data.fetcher.adjust

    print(f"parsing scan reports from {reports_dir} ...")
    signals = parse_scan_reports(reports_dir)

    total_entries = sum(len(v) for v in signals.values())
    print(f"found {total_entries} BUY entries across {len(signals)} strategies:")
    for sname, entries in sorted(signals.items()):
        print(f"  {sname}: {len(entries)} entries  ({entries[0].buy_date} ~ {entries[-1].buy_date})")

    print(f"\nsimulating trades (adjust={adjust}) ...")
    store = DuckStore()
    try:
        trades = simulate_trades(signals, store, adjust)
    finally:
        store.close()

    print(f"\ngenerated {len(trades)} trades:")
    completed = [t for t in trades if not t.is_open]
    open_pos = [t for t in trades if t.is_open]
    print(f"  completed: {len(completed)}")
    print(f"  open:      {len(open_pos)}")

    report = generate_report(trades, adjust)
    out_path = reports_dir / f"signal_performance_{date.today()}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nreport saved to: {out_path}")

    # 打印终端摘要
    print("\n" + "=" * 70)
    for sname in sorted({t.strategy for t in trades}):
        st = [t for t in trades if t.strategy == sname]
        comp = [t for t in st if not t.is_open]
        wins = sum(1 for t in comp if (t.return_pct or 0) > 0)
        total_ret = sum(t.return_pct or 0 for t in comp)
        avg_ret = total_ret / len(comp) if comp else 0
        print(f"{sname:<22s}  {len(comp):>2d} closed  "
              f"win {wins}/{len(comp)}  "
              f"avg {avg_ret:+.2f}%  total {total_ret:+.2f}%")


if __name__ == "__main__":
    main()
