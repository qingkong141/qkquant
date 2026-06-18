"""单次盘中扫描测试 — 不推送，只打印结果"""
import requests, yaml, json
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
UT = "fa5fd1943c7b386f172d6893dbf10df7"

# 加载持仓
p = PROJECT / "config" / "positions.yaml"
holdings = {}
if p.exists():
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    holdings = {x["code"]: x for x in cfg.get("positions", []) if isinstance(x, dict) and "code" in x}

def secid(c):
    return f"1.{c}" if str(c).zfill(6)[0] in "69" else f"0.{c}"

def quote(code):
    try:
        r = requests.get("https://push2.eastmoney.com/api/qt/stock/get", params={
            "secid": secid(code), "fields": "f43,f44,f45,f46,f47,f170,f58", "ut": UT
        }, timeout=5)
        d = r.json().get("data", {})
        if not d: return None
        return {"price": (d.get("f43") or 0)/1000, "high": (d.get("f44") or 0)/1000,
                "low": (d.get("f45") or 0)/1000, "open": (d.get("f46") or 0)/1000,
                "chg": (d.get("f170") or 0)/100, "name": d.get("f58", "")}
    except: return None

def five_min(code, n=5):
    try:
        r = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
            "secid": secid(code), "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "5", "fqt": "1", "end": "20500101", "lmt": str(n), "ut": UT
        }, timeout=5)
        klines = r.json().get("data", {}).get("klines", []) or []
        return [dict(zip(["time","open","close","high","low","vol"],
                         [p[0]] + [float(x) for x in p[1:]])) for k in klines if (p := k.split(","))]
    except: return []

print(f"=== 盘中扫描 {datetime.now().strftime('%H:%M:%S')} ===\n")

for code in list(holdings.keys()):
    q = quote(code)
    if q:
        print(f"{code} {q['name']:6s} price={q['price']:>8.2f} chg={q['chg']:>+6.2f}%  "
              f"O={q['open']:.2f} H={q['high']:.2f} L={q['low']:.2f}")

print()

for code in list(holdings.keys())[:3]:
    bars = five_min(code, 8)
    if len(bars) >= 3:
        first = bars[0]; last = bars[-1]
        chg = (last["close"]/first["open"]-1)*100
        vol_sum = sum(b["vol"] for b in bars)
        print(f"{code} 5min: {len(bars)}bars  {first['time'][-8:-3]}->{last['time'][-8:-3]}  "
              f"chg={chg:+.2f}%  vol={vol_sum/10000:.0f}万")
