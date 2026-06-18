"""缠论中枢识别 + 三类买卖点 Demo。

核心概念：
  中枢 = 三段以上线段重叠的横盘区间（多空拉锯带）
  一买 = 跌破中枢后趋势衰竭反转（抄底）
  二买 = 回踩不破一买低点（确认）
  三买 = 突破中枢后回踩不跌回中枢（追涨 — 跟你的 momentum 最像）

用日线 K 线简化实现。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qkquant.data.storage import DuckStore
from qkquant.factors.indicators import macd


# ═══════════════════════════════════════════════════════════════════
# 1. 找笔（线段）—— 价格方向的转折点
# ═══════════════════════════════════════════════════════════════════

def find_swings(high: np.ndarray, low: np.ndarray, min_bars: int = 4) -> list[dict]:
    """找局部高低点转折（简化版"笔"）。

    返回 [{idx, type: 'high'|'low', price}]
    min_bars: 一个转折至少需要 N 根 K 线才被确认
    """
    n = len(high)
    swings: list[dict] = []

    # 找局部极值点
    for i in range(min_bars, n - min_bars):
        # 局部高点
        if all(high[i] >= high[i - j] for j in range(1, min_bars + 1)) and \
           all(high[i] > high[i + j] for j in range(1, min_bars + 1)):
            swings.append({"idx": i, "type": "high", "price": float(high[i])})
        # 局部低点
        if all(low[i] <= low[i - j] for j in range(1, min_bars + 1)) and \
           all(low[i] < low[i + j] for j in range(1, min_bars + 1)):
            swings.append({"idx": i, "type": "low", "price": float(low[i])})

    # 去重：相同类型相邻的只保留最极端的
    filtered = []
    for s in swings:
        if not filtered or s["type"] != filtered[-1]["type"]:
            filtered.append(s)
        elif s["type"] == "high" and s["price"] > filtered[-1]["price"]:
            filtered[-1] = s
        elif s["type"] == "low" and s["price"] < filtered[-1]["price"]:
            filtered[-1] = s

    return sorted(filtered, key=lambda x: x["idx"])


# ═══════════════════════════════════════════════════════════════════
# 2. 中枢识别
# ═══════════════════════════════════════════════════════════════════

def find_zhongshu(swings: list[dict]) -> list[dict]:
    """从笔序列中识别中枢。

    中枢 = 连续 3 笔以上的重叠区间。
    取前三笔的高低点做交集，Z_high = min(高1,高2,高3), Z_low = max(低1,低2,低3)

    返回 [{start_idx, end_idx, high, low, swings_included}]
    """
    if len(swings) < 6:
        return []

    zhongshu_list = []
    i = 0

    while i <= len(swings) - 6:
        sub = swings[i:i + 6]  # 3 笔 = 6 个转折点（高→低→高→低→高→低）
        # 前三笔的交集
        highs = [s["price"] for s in sub if s["type"] == "high"][:3]
        lows = [s["price"] for s in sub if s["type"] == "low"][:3]
        if len(highs) < 3 or len(lows) < 3:
            i += 1
            continue

        zg = min(highs)  # 中枢上沿
        zd = max(lows)   # 中枢下沿

        if zg > zd:  # 有重叠 = 有效中枢
            zhongshu_list.append({
                "start_idx": sub[0]["idx"],
                "end_idx": sub[-1]["idx"],
                "high": zg,
                "low": zd,
                "mid": (zg + zd) / 2,
            })
            i += 2  # 跳过一笔
        else:
            i += 1

    # 合并相邻重叠的中枢
    merged = []
    for zs in zhongshu_list:
        if merged and zs["low"] <= merged[-1]["high"] and zs["high"] >= merged[-1]["low"]:
            merged[-1]["high"] = min(merged[-1]["high"], zs["high"])
            merged[-1]["low"] = max(merged[-1]["low"], zs["low"])
            merged[-1]["end_idx"] = zs["end_idx"]
        else:
            merged.append(zs)

    return merged


# ═══════════════════════════════════════════════════════════════════
# 3. 背驰检测（顶部 + 底部）
# ═══════════════════════════════════════════════════════════════════

def check_divergence(closes: np.ndarray, swings: list[dict]) -> dict:
    """检测顶部和底部背驰。

    顶背驰: 价格新高，但 MACD 柱面积 < 前一波 → 多头衰竭，要跌
    底背驰: 价格新低，但 MACD 柱面积 < 前一波 → 空头衰竭，要涨

    Returns {"top": [idx,...], "bottom": [idx,...]}
    """
    if len(swings) < 4:
        return {"top": [], "bottom": []}

    macd_df = macd(pd.Series(closes))
    hist = macd_df["hist"].values

    top_div, bottom_div = [], []

    for peak_type, cmp_fn in [("high", lambda a, b: a > b), ("low", lambda a, b: a < b)]:
        type_swings = [s for s in swings if s["type"] == peak_type]
        for i in range(1, len(type_swings)):
            prev = type_swings[i - 1]
            curr = type_swings[i]
            # 价格极端化
            if not cmp_fn(curr["price"], prev["price"]):
                continue
            # MACD 柱面积衰减
            area_prev = np.sum(np.abs(hist[max(0, prev["idx"] - 10):prev["idx"] + 1]))
            area_curr = np.sum(np.abs(hist[max(0, curr["idx"] - 10):curr["idx"] + 1]))
            if area_curr < area_prev * 0.7:
                if peak_type == "high":
                    top_div.append(curr["idx"])
                else:
                    bottom_div.append(curr["idx"])

    return {"top": top_div, "bottom": bottom_div}


# ═══════════════════════════════════════════════════════════════════
# 4. 三类买卖点
# ═══════════════════════════════════════════════════════════════════

def find_all_buy_points(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    swings: list[dict], zhongshu_list: list[dict], divergences: dict,
) -> list[dict]:
    """找一买、二买、三买。"""

    buys = []
    bottom_div = set(divergences.get("bottom", []))

    # ── 一买: 跌破最后一个中枢 + 底背驰 ──
    if zhongshu_list and bottom_div:
        last_zs = zhongshu_list[-1]
        low_swings = [s for s in swings if s["type"] == "low"]
        for s in low_swings:
            if s["price"] < last_zs["low"] and s["idx"] in bottom_div:
                # 找背驰点的最低收盘价位置（精确定位）
                nearby_lows = closes[max(0, s["idx"] - 3):s["idx"] + 4]
                exact_idx = int(np.argmin(nearby_lows)) + max(0, s["idx"] - 3)
                buys.append({
                    "idx": exact_idx, "type": "一买",
                    "price": float(closes[exact_idx]),
                    "reason": f"跌破中枢{last_zs['low']:.2f}后底背驰，空头衰竭反转",
                })
                break  # 只取最近一个

    # ── 二买: 一买之后回落，不破一买低点 ──
    if buys:
        first_buy = buys[0]
        for i in range(first_buy["idx"] + 10, min(first_buy["idx"] + 60, len(closes))):
            # 找一买之后的局部低点
            if i < len(closes) - 4:
                seg = lows[i - 3:i + 4]
                if lows[i] == seg.min() and closes[i] > first_buy["price"] * 1.01:
                    # 确认不是阴跌（之前有反弹）
                    prev_range = closes[first_buy["idx"]:i]
                    if prev_range.max() > first_buy["price"] * 1.05:
                        buys.append({
                            "idx": i, "type": "二买",
                            "price": float(closes[i]),
                            "reason": f"一买{first_buy['price']:.2f}后回踩确认，未破前低",
                        })
                        break

    # ── 三买: 突破中枢 + 回踩不破中枢上沿 ──
    for zs in reversed(zhongshu_list):
        zs_end = zs["end_idx"]
        for i in range(zs_end, len(closes) - 10):
            if closes[i] > zs["high"]:
                # 找突破后的回踩点
                post_peak = i
                for j in range(i + 1, min(i + 15, len(closes))):
                    if closes[j] > closes[post_peak]:
                        post_peak = j
                for j in range(post_peak + 1, min(post_peak + 15, len(closes))):
                    if closes[j] > zs["high"] * 0.97 and closes[j] < zs["high"] * 1.03:
                        buys.append({
                            "idx": j, "type": "三买",
                            "price": float(closes[j]),
                            "reason": f"突破中枢{zs['high']:.2f}后回踩确认",
                        })
                        break
                break
        break

    return sorted(buys, key=lambda x: x["idx"])


# ═══════════════════════════════════════════════════════════════════
# 5. 运行 Demo
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    store = DuckStore()

    # 选几只票演示
    for code in ["600183", "000338", "002384", "688008", "000002"]:
        df = store.load_daily(codes=[code], start="2025-06-01", end="2026-06-13", adjust="qfq")
        if df.empty:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # 缠论分析
        swings = find_swings(highs, lows)
        zhongshu = find_zhongshu(swings)
        divergences = check_divergence(closes, swings)
        buy_points = find_all_buy_points(closes, highs, lows, swings, zhongshu, divergences)

        # 输出
        inst = store.load_instruments([code])
        name = inst.iloc[0]["name"] if not inst.empty else code

        print(f"\n{'='*60}")
        print(f"  {code} {name}")
        print(f"  K线: {len(df)} 根  笔: {len(swings)} 个  中枢: {len(zhongshu)} 个")
        print(f"  顶背驰: {len(divergences['top'])} 个  底背驰: {len(divergences['bottom'])} 个")

        # 最近的中枢
        if zhongshu:
            last_zs = zhongshu[-1]
            zs_s = str(df.iloc[last_zs["start_idx"]]["trade_date"])[:10]
            zs_e = str(df.iloc[last_zs["end_idx"]]["trade_date"])[:10]
            print(f"  最后一个中枢: {zs_s}~{zs_e}")
            print(f"    区间: [{last_zs['low']:.2f} ~ {last_zs['high']:.2f}]")
            current = closes[-1]
            pos = "上方" if current > last_zs["high"] else ("下方" if current < last_zs["low"] else "内部")
            print(f"    当前价={current:.2f}  位置={pos}")

        # 买卖点
        if buy_points:
            print(f"  缠论买点:")
            for bp in buy_points[-5:]:
                dt = str(df.iloc[bp["idx"]]["trade_date"])[:10]
                ret_to_now = (closes[-1] / bp["price"] - 1) * 100
                print(f"    {dt} {bp['type']} price={bp['price']:.2f}  →至今{ret_to_now:+.1f}%  ({bp['reason']})")
        else:
            print(f"  缠论买点: 无")

    store.close()
