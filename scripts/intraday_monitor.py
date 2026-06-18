"""盘中 5 分钟线监控 —— 东方财富 + 新浪双源，盯持仓 + 日线候选池。

启动: python scripts/intraday_monitor.py
停止: Ctrl+C
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from pathlib import Path

import requests
import yaml

# ── 配置 ──
SCAN_INTERVAL = 300
TRADING_START = "09:30"
TRADING_END = "15:00"
_UT = "fa5fd1943c7b386f172d6893dbf10df7"
PROJECT = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════
# 数据源（东方财富主力 + 新浪备用）
# ══════════════════════════════════════════════════════════════════════

def em_secid(code: str) -> str:
    c = str(code).zfill(6)
    return f"1.{c}" if c[0] in "69" else f"0.{c}"


def fetch_quote_em(code: str) -> dict | None:
    try:
        r = requests.get("https://push2.eastmoney.com/api/qt/stock/get", params={
            "secid": em_secid(code),
            "fields": "f43,f44,f45,f46,f47,f48,f170,f58",
            "ut": _UT,
        }, timeout=5)
        d = r.json().get("data", {})
        if not d: return None
        return {
            "price": (d.get("f43") or 0) / 1000,
            "high": (d.get("f44") or 0) / 1000,
            "low": (d.get("f45") or 0) / 1000,
            "open": (d.get("f46") or 0) / 1000,
            "vol": d.get("f47", 0),
            "amount": d.get("f48", 0),
            "pct_chg": (d.get("f170") or 0) / 100,
            "name": d.get("f58", ""),
        }
    except Exception:
        return None


def fetch_quote_sina(code: str) -> dict | None:
    """新浪备用：只返回基础价格。"""
    c = str(code).zfill(6)
    prefix = "sh" if c[0] in "69" else "sz"
    try:
        r = requests.get(f"https://hq.sinajs.cn/list={prefix}{c}",
                         headers={"Referer": "https://finance.sina.com.cn"}, timeout=5)
        parts = r.text.split('"')[1].split(",")
        return {
            "name": parts[0],
            "open": float(parts[1]),
            "price": float(parts[3]),
            "high": float(parts[4]),
            "low": float(parts[5]),
            "vol": int(parts[8]),
            "pct_chg": (float(parts[3]) / float(parts[2]) - 1) * 100,
        }
    except Exception:
        return None


def fetch_quote(code: str) -> dict | None:
    return fetch_quote_em(code) or fetch_quote_sina(code)


def fetch_5min(code: str, limit: int = 10) -> list[dict]:
    try:
        r = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
            "secid": em_secid(code), "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "5", "fqt": "1", "end": "20500101", "lmt": str(limit), "ut": _UT,
        }, timeout=5)
        klines = r.json().get("data", {}).get("klines", []) or []
        out = []
        for k in klines:
            p = k.split(",")
            out.append({
                "time": p[0], "open": float(p[1]), "close": float(p[2]),
                "high": float(p[3]), "low": float(p[4]), "vol": float(p[5]),
            })
        return out
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════
# 股票池
# ══════════════════════════════════════════════════════════════════════

def load_holdings() -> dict:
    p = PROJECT / "config" / "positions.yaml"
    if not p.exists(): return {}
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {x["code"]: x for x in cfg.get("positions", []) if isinstance(x, dict) and "code" in x}


def load_candidates() -> list[str]:
    """从今天的 scan_raw 报告解析 BUY 列表。"""
    today = date.today().isoformat()
    report = PROJECT / "reports" / f"scan_{today}_raw.md"
    if not report.exists():
        return []
    text = report.read_text(encoding="utf-8")
    codes: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s{2}-\s(\d{6})\s", line)
        if m and "[BUY]" in text[max(0, text.rfind("###", 0, text.find(line))):text.find(line)].split("\n")[-3]:
            codes.append(m.group(1))
    # 简化：直接匹配股票行并查它是否在 BUY 区
    in_buy = False
    for line in text.splitlines():
        if "### [BUY]" in line:
            in_buy = True
        elif line.startswith("### [SELL]") or line.startswith("### [REJECTED]"):
            in_buy = False
        elif in_buy:
            m = re.match(r"^\s{2}-\s(\d{6})\s", line)
            if m and m.group(1) not in codes:
                codes.append(m.group(1))
    return codes


holdings = load_holdings()
candidates = load_candidates()
# 去重合并：持仓 + 今日候选
watch_set: dict[str, str] = {}  # code -> source
for c in holdings:
    watch_set[c] = "holding"
for c in candidates:
    if c not in watch_set:
        watch_set[c] = "candidate"

print(f"[POOL] holdings={len(holdings)}  candidates={len(candidates)}  total={len(watch_set)}")


# ══════════════════════════════════════════════════════════════════════
# 信号检测
# ══════════════════════════════════════════════════════════════════════

def in_trading_hours() -> bool:
    now = datetime.now().strftime("%H:%M")
    return TRADING_START <= now <= TRADING_END


def check_signals() -> list[dict]:
    """返回信号列表，每个信号带 type/level/code/msg"""
    signals: list[dict] = []

    for code, source in watch_set.items():
        q = fetch_quote(code)
        if not q:
            continue
        name = q.get("name", code)
        tag = "[H]" if source == "holding" else "[C]"  # H=持仓 C=候选

        # ---- 持仓专用：风险信号 ----
        if source == "holding":
            # 从日内高点回落超 5%
            if q["high"] > 0 and q["price"] < q["high"] * 0.95:
                drop = round((q["price"] / q["high"] - 1) * 100, 1)
                signals.append({
                    "type": "risk", "code": code,
                    "msg": f"{tag} {code} {name} 从高点回落{drop}% H={q['high']:.2f} now={q['price']:.2f}"
                })
            # 跌超 3%
            if q["pct_chg"] < -3:
                signals.append({
                    "type": "risk", "code": code,
                    "msg": f"{tag} {code} {name} 跌幅{q['pct_chg']:+.2f}% price={q['price']:.2f}"
                })

        # ---- 候选池专用：入场信号 ----
        if source == "candidate":
            bars = fetch_5min(code, 10)
            if len(bars) < 6:
                continue
            latest = bars[-1]
            prev = bars[:-1]

            # 突破日内新高 (前 9 根的最高点)
            intra_high = max(b["high"] for b in prev)
            if intra_high > 0 and latest["close"] > intra_high * 1.02:
                pct = round((latest["close"] / intra_high - 1) * 100, 1)
                signals.append({
                    "type": "buy", "code": code,
                    "msg": f"{tag} [BUY] {code} {name} 突破日内新高 +{pct}% close={latest['close']:.2f}"
                })

            # 放量 + 涨幅确认
            avg_vol = sum(b["vol"] for b in prev[-5:]) / 5
            if avg_vol > 0 and latest["vol"] > avg_vol * 1.5:
                bar_chg = (latest["close"] / latest["open"] - 1) * 100
                if bar_chg > 1:  # 放量上涨
                    signals.append({
                        "type": "buy", "code": code,
                        "msg": f"{tag} [BUY] {code} {name} 放量上涨 vol={latest['vol']/avg_vol:.1f}x bar={bar_chg:+.1f}%"
                    })

    return signals


# ══════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[MONITOR] {len(watch_set)} codes  interval={SCAN_INTERVAL}s")
    print(f"[MONITOR] hours={TRADING_START}-{TRADING_END}")
    print(f"[MONITOR] push=OFF (测试模式)")

    while True:
        now = datetime.now()
        if not in_trading_hours():
            print(f"\r[zzz] {now.strftime('%H:%M')} ...", end="")
            time.sleep(60)
            continue

        t0 = time.time()
        print(f"\r[...] {now.strftime('%H:%M:%S')} scanning {len(watch_set)} codes ...", end="")
        try:
            sigs = check_signals()
            if sigs:
                elapsed = time.time() - t0
                print(f"\n[{'='*50}]")
                for s in sigs:
                    print(f"  {s['msg']}")
                print(f"[{'='*50}]  {len(sigs)} signals  {elapsed:.1f}s")
        except Exception as e:
            print(f"\n[ERR] {e}")

        elapsed = time.time() - t0
        sleep_time = max(10, SCAN_INTERVAL - elapsed)
        time.sleep(sleep_time)
