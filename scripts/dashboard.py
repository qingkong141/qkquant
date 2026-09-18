"""qkquant Streamlit 可视化面板。

启动: streamlit run scripts/dashboard.py
"""

from __future__ import annotations

raise SystemExit(
    "旧个股看板已停用。当前 ETF 入口：etf-backtest / etf-plan；使用说明见 docs/ETF_WORKFLOW.md。"
)

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qkquant.config import PROJECT_ROOT, get_settings
from qkquant.data.storage import DuckStore
from qkquant.scan import load_holdings
from qkquant.strategy.registry import get_strategy, list_strategies, load_strategy_config, load_risk_config

st.set_page_config(page_title="qkquant Dashboard", page_icon="📊", layout="wide")
st.title("📊 qkquant 量化交易面板")

# ── 侧边栏 ──────────────────────────────────────────────────────────

page = st.sidebar.radio("导航", ["📈 策略回测", "💼 持仓跟踪", "📋 信号历史"])

# ── 公共数据 ────────────────────────────────────────────────────────

@st.cache_resource
def get_store():
    return DuckStore(read_only=True)  # 只读，不锁库，定时任务可并行写

store = get_store()
adjust = get_settings().data.fetcher.adjust

# ── 策略列表 ────────────────────────────────────────────────────────

def get_available_strategies():
    return [s.name for s in list_strategies()]

# ── 加载持仓 ────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_positions():
    p = PROJECT_ROOT / "config" / "positions.yaml"
    return load_holdings(p)

holdings = load_positions()

# ══════════════════════════════════════════════════════════════════════
# 页面 1: 策略回测
# ══════════════════════════════════════════════════════════════════════

if page == "📈 策略回测":
    st.header("策略回测")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        strategy_name = st.selectbox("策略", get_available_strategies(), index=0)
    with col2:
        capital = st.number_input("初始资金", value=50_000, step=10_000, min_value=10_000)
    with col3:
        start_date = st.date_input("开始日期", date(2023, 1, 1))
    with col4:
        end_date = st.date_input("结束日期", date.today())

    # 初始化 session state
    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None

    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        with st.spinner("回测中..."):
            info = get_strategy(strategy_name)
            cfg = load_strategy_config(info)
            params = (cfg.get("params") or {}) if cfg else {}
            risk_cfg = load_risk_config(cfg)

            codes = store.load_index_constituents("000300")
            inst = store.load_instruments(codes)
            if not inst.empty and "is_st" in inst.columns:
                st_codes = set(inst[inst["is_st"]]["code"].tolist())
                codes = [c for c in codes if c not in st_codes]

            from qkquant.backtest.engine import BacktestEngine

            engine = BacktestEngine(
                store=store,
                strategy_cls=info.cls,
                strategy_params=params,
                initial_capital=capital,
                risk_config=risk_cfg,
            )
            result = engine.run(codes=codes, start=start_date, end=end_date)
            strat = result["strategy"]

            final = result["final_value"]
            ret = (final - capital) / capital * 100
            dd = result["analyzers"]["drawdown"]
            mdd = float((dd.get("max") or {}).get("drawdown", 0))
            trade_log = getattr(strat, "_trade_log", [])
            buys = sum(1 for t in trade_log if t["side"] == "BUY")
            sells = sum(1 for t in trade_log if t["side"] == "SELL")
            annual = ((final / capital) ** (1 / max((end_date - start_date).days / 365, 0.5)) - 1) * 100

            # ── 指标卡 ──
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("最终值", f"¥{final:,.0f}")
            c2.metric("总收益", f"{ret:+.2f}%")
            c3.metric("年化", f"{annual:+.1f}%")
            c4.metric("最大回撤", f"{mdd:.1f}%")
            c5.metric("买入", buys)
            c6.metric("卖出", sells)

            st.divider()

            # ── 净值曲线 ──
            daily_rets = result["analyzers"]["timereturn"]
            if daily_rets:
                df_eq = pd.DataFrame(
                    {"date": list(daily_rets.keys()), "daily_ret": list(daily_rets.values())}
                )
                df_eq["date"] = pd.to_datetime(df_eq["date"])
                df_eq = df_eq.sort_values("date")
                df_eq["equity"] = (1 + df_eq["daily_ret"]).cumprod() * capital

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_eq["date"], y=df_eq["equity"],
                    name="策略净值", line=dict(color="#1f77b4", width=2),
                ))
                # 回撤
                df_eq["peak"] = df_eq["equity"].cummax()
                df_eq["drawdown"] = (df_eq["equity"] / df_eq["peak"] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=df_eq["date"], y=df_eq["drawdown"],
                    name="回撤 %", yaxis="y2",
                    line=dict(color="red", width=1), fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
                ))
                # 买卖点
                if trade_log:
                    buy_dates = [t["date"] for t in trade_log if t["side"] == "BUY"]
                    sell_dates = [t["date"] for t in trade_log if t["side"] == "SELL"]
                    buy_vals = [
                        df_eq[df_eq["date"] == pd.Timestamp(d)]["equity"].values[0]
                        for d in buy_dates if d in df_eq["date"].values
                    ]
                    sell_vals = [
                        df_eq[df_eq["date"] == pd.Timestamp(d)]["equity"].values[0]
                        for d in sell_dates if d in df_eq["date"].values
                    ]
                    fig.add_trace(go.Scatter(
                        x=buy_dates, y=buy_vals, mode="markers",
                        name="买入", marker=dict(symbol="triangle-up", size=10, color="green"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=sell_dates, y=sell_vals, mode="markers",
                        name="卖出", marker=dict(symbol="triangle-down", size=10, color="red"),
                    ))

                fig.update_layout(
                    title=f"{strategy_name} 净值曲线",
                    xaxis_title="日期", yaxis_title="净值 (¥)",
                    yaxis2=dict(title="回撤 %", overlaying="y", side="right"),
                    hovermode="x unified",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── 月度收益热力图 ──
            if daily_rets:
                df_eq["year"] = df_eq["date"].dt.year
                df_eq["month"] = df_eq["date"].dt.month
                monthly = df_eq.groupby(["year", "month"])["daily_ret"].apply(
                    lambda x: (1 + x).prod() - 1
                ).reset_index()
                monthly_pivot = monthly.pivot(index="year", columns="month", values="daily_ret")
                monthly_pivot = monthly_pivot * 100

                fig2 = go.Figure(data=go.Heatmap(
                    z=monthly_pivot.values,
                    x=["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"][:monthly_pivot.shape[1]],
                    y=monthly_pivot.index.astype(str),
                    text=[[f"{v:+.1f}%" if pd.notna(v) else "" for v in row] for row in monthly_pivot.values],
                    texttemplate="%{text}",
                    colorscale="RdYlGn", zmid=0,
                ))
                fig2.update_layout(title="月度收益热力图 (%)", height=250)
                st.plotly_chart(fig2, use_container_width=True)

            # ── 交易明细 ──
            if trade_log:
                st.subheader("交易明细")
                df_trades = pd.DataFrame(trade_log)
                df_trades["value"] = df_trades["price"] * df_trades["qty"]
                st.dataframe(
                    df_trades.sort_values(["date", "side"], ascending=[True, False]),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "date": "日期", "code": "代码", "side": "方向",
                        "price": st.column_config.NumberColumn("价格", format="%.2f"),
                        "qty": "数量", "value": st.column_config.NumberColumn("金额", format="%.0f"),
                        "reason": "原因",
                    },
                )

            # ── AI 分析 ──
            st.divider()
            if st.button("🤖 AI 分析策略表现", key="ai_btn"):
                with st.spinner("DeepSeek 分析中..."):
                    try:
                        from qkquant.ai.analyzer import load_ai_config, make_ai_provider
                        from qkquant.ai.base import AiRequest, AiCandidate
                        cfg = load_ai_config()
                        provider = make_ai_provider(cfg)
                        # 取最近 10 笔买入作为候选
                        recent_buys = [t for t in trade_log if t["side"] == "BUY"][-10:]
                        buy_codes = list(set(t["code"] for t in recent_buys))
                        name_map = dict(zip(inst["code"], inst["name"])) if not inst.empty else {}
                        candidates = [
                            AiCandidate(
                                code=c, name=name_map.get(c, ""),
                                strategies=[strategy_name],
                                scores={strategy_name: 1.0},
                                reasons={strategy_name: "backtest_entry"},
                            ) for c in buy_codes
                        ]
                        req = AiRequest(
                            as_of=end_date, strategies=[strategy_name],
                            candidates=candidates, notes=[
                                f"回测周期 {start_date} ~ {end_date}",
                                f"总收益 {ret:+.2f}% 最大回撤 {mdd:.1f}% 年化 {annual:+.1f}%",
                                f"买入 {buys} 笔 卖出 {sells} 笔",
                            ],
                        )
                        resp = provider.analyze(req)
                        if resp.ok:
                            st.markdown(resp.markdown)
                        else:
                            st.warning(f"AI: {resp.error}")
                    except Exception as ex:
                        st.warning(f"AI 不可用: {ex}")

            # ── 风控日志 ──
            if trade_log:
                strat_reasons = {"momentum_exit", "breakout_entry"}
                forced = [t for t in trade_log
                    if t["side"] == "SELL"
                    and not any(r in str(t.get("reason", "")) for r in ["momentum_exit", "take_profit", "time_stop", "ma_cross", "below_boll"])]
                if forced:
                    with st.expander(f"🛡️ 风控止损明细 ({len(forced)} 笔)", expanded=False):
                        # 配对买卖计算回撤
                        buys_dict = {}
                        for t in trade_log:
                            if t["side"] == "BUY":
                                buys_dict.setdefault(t["code"], []).append(t)
                        lines = []
                        for s in forced:
                            code = s["code"]
                            entry = None
                            code_buys = buys_dict.get(code, [])
                            for b in reversed(code_buys):
                                if b["date"] <= s["date"]:
                                    entry = b
                                    break
                            if entry:
                                dd = (s["price"] / entry["price"] - 1) * 100
                                lines.append(
                                    f"{s['date']} {code} 入场 {entry['price']:.2f}({entry['date']}) "
                                    f"→ 出场 {s['price']:.2f} 回撤 {dd:+.1f}%"
                                )
                            else:
                                lines.append(f"{s['date']} {code} 出场 {s['price']:.2f}")
                        st.text("\n".join(lines))

            # 保存回测结果到 session_state，供股票走势图复用
            import copy
            st.session_state.bt_result = {
                "trade_log": copy.deepcopy(trade_log),
                "start_date": start_date,
                "end_date": end_date,
                "name_map": dict(zip(inst["code"], inst["name"])) if not inst.empty else {},
                "strategy_name": strategy_name,
                "final": final, "ret": ret, "mdd": mdd, "annual": annual,
                "buys": buys, "sells": sells,
            }
            # 净值数据（plotly 图不方便序列化，存原始数据）
            if daily_rets:
                df_eq["date"] = df_eq["date"].astype(str)
                st.session_state.bt_result["equity_data"] = df_eq.to_dict("list")
                if trade_log:
                    st.session_state.bt_result["buy_dates"] = [
                        str(d) for d in buy_dates if d in df_eq["date"].values
                    ]
                    st.session_state.bt_result["buy_vals"] = buy_vals
                    st.session_state.bt_result["sell_dates"] = [
                        str(d) for d in sell_dates if d in df_eq["date"].values
                    ]
                    st.session_state.bt_result["sell_vals"] = sell_vals



# ══════════════════════════════════════════════════════════════════════
# 页面 2: 持仓跟踪
# ══════════════════════════════════════════════════════════════════════

elif page == "💼 持仓跟踪":
    st.header("持仓跟踪")

    if not holdings:
        st.warning("暂无持仓数据（config/positions.yaml 为空）")
    else:
        # 加载持仓明细
        codes = list(holdings.keys())
        inst = store.load_instruments(codes)
        name_map = dict(zip(inst["code"], inst["name"])) if not inst.empty else {}

        rows = []
        as_of = store.conn.execute("SELECT MAX(trade_date) FROM daily_bars").fetchone()[0]
        as_of = as_of or date.today()

        # 加载行业
        import csv
        ind_path = PROJECT_ROOT / "data" / "industries.csv"
        ind_map = {}
        if ind_path.exists():
            with ind_path.open("r", encoding="utf-8-sig") as f:
                ind_map = {row["code"]: row["industry"] for row in csv.DictReader(f)}

        for code, h in holdings.items():
            cost = float(h.get("cost", 0))
            qty = int(h.get("qty", 0))
            bought = h.get("bought_at", "")

            df_p = store.load_daily(codes=[code], start=bought, end=as_of, adjust=adjust)
            if df_p.empty:
                current = cost
            else:
                current = float(df_p.sort_values("trade_date").iloc[-1]["close"])

            ret_pct = (current / cost - 1) * 100 if cost > 0 else 0
            pnl = (current - cost) * qty
            days = (as_of - pd.Timestamp(bought).date()).days if bought else 0
            rows.append({
                "code": code,
                "name": name_map.get(code, ""),
                "qty": qty,
                "cost": cost,
                "current": current,
                "ret_pct": ret_pct,
                "pnl": pnl,
                "days": days,
                "industry": ind_map.get(code, "未知"),
                "note": h.get("note", ""),
            })

        df_pos = pd.DataFrame(rows).sort_values("pnl", ascending=False)
        total_pnl = df_pos["pnl"].sum()
        total_value = (df_pos["current"] * df_pos["qty"]).sum()
        n_win = (df_pos["ret_pct"] > 0).sum()
        n_lose = (df_pos["ret_pct"] < 0).sum()

        # ── 指标卡 ──
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("持仓数", len(rows))
        c2.metric("总市值", f"¥{total_value:,.0f}")
        c3.metric("累计盈亏", f"¥{total_pnl:+,.0f}")
        c4.metric("盈利数", n_win)
        c5.metric("亏损数", n_lose)

        st.divider()

        col_l, col_r = st.columns([1, 1])

        with col_l:
            # ── 行业分布 ──
            ind_counts = df_pos.groupby("industry")["current"].apply(lambda x: (x * df_pos.loc[x.index, "qty"]).sum())
            ind_counts = ind_counts.sort_values(ascending=True)
            fig_ind = go.Figure(go.Bar(
                x=ind_counts.values,
                y=ind_counts.index,
                orientation="h",
                text=[f"¥{v:,.0f}" for v in ind_counts.values],
                textposition="outside",
            ))
            fig_ind.update_layout(title="行业分布（市值）", height=300, xaxis_title="")
            st.plotly_chart(fig_ind, use_container_width=True)

        with col_r:
            # ── 个股权重 ──
            if total_value > 0:
                df_pos["weight"] = (df_pos["current"] * df_pos["qty"]) / total_value * 100
                fig_w = go.Figure(go.Pie(
                    labels=df_pos["name"] + " " + df_pos["code"],
                    values=df_pos["weight"],
                    textinfo="label+percent",
                ))
                fig_w.update_layout(title="个股权重", height=300)
                st.plotly_chart(fig_w, use_container_width=True)

        st.divider()

        # ── 持仓明细表 ──
        st.subheader("持仓明细")
        st.dataframe(
            df_pos,
            use_container_width=True, hide_index=True,
            column_config={
                "code": "代码", "name": "名称", "qty": "股数",
                "cost": st.column_config.NumberColumn("成本", format="%.2f"),
                "current": st.column_config.NumberColumn("现价", format="%.2f"),
                "ret_pct": st.column_config.NumberColumn("收益", format="%+.2f%%"),
                "pnl": st.column_config.NumberColumn("盈亏", format="¥%+,.0f"),
                "days": "持日", "industry": "行业", "note": "备注",
            },
            column_order=["code", "name", "qty", "cost", "current", "ret_pct", "pnl", "days", "industry", "note"],
        )

        # ── P&L 柱状图 ──
        fig_pnl = px.bar(
            df_pos.sort_values("pnl"),
            x="pnl", y="name", orientation="h",
            color="ret_pct", color_continuous_scale=["red", "gray", "green"],
            title="个股权益分布",
            text_auto=True,
        )
        fig_pnl.update_traces(texttemplate="¥%{value:+,.0f}", textposition="outside")
        fig_pnl.update_layout(height=40 + 30 * len(df_pos), xaxis_title="累计盈亏 (¥)")
        st.plotly_chart(fig_pnl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# 页面 3: 信号历史
# ══════════════════════════════════════════════════════════════════════

elif page == "📋 信号历史":
    st.header("信号历史")

    import re
    from dataclasses import dataclass

    reports_dir = PROJECT_ROOT / "reports"
    report_files = sorted(reports_dir.glob("scan_*_raw.md"))

    if not report_files:
        st.warning("暂无历史信号报告（reports/scan_*_raw.md）")
    else:
        # 解析所有报告
        _STRATEGY_HDR = re.compile(r"^## (\S+) ")
        _BUY_SECTION = re.compile(r"^### \[BUY\]")
        _STOCK_LINE = re.compile(r"^\s{2}-\s(\d{6})\s+(.+?)(?:\s{2}\[.*?\])?\s*$")
        _CLOSE_RE = re.compile(r"close=([\d.]+)")
        _SELL_SECTION = re.compile(r"^### \[SELL\]")
        _SELL_STOCK = re.compile(r"^\s{2}-\s(\d{6})\s+(.+?)\s+\(持仓.*?←\s+(\S+)")

        records: list[dict] = []
        for fpath in report_files:
            m = re.search(r"scan_(\d{4}-\d{2}-\d{2})_raw\.md", fpath.name)
            if not m:
                continue
            rdate = m.group(1)
            text = fpath.read_text(encoding="utf-8")
            strat = None
            in_buy = False
            pending = None
            for line in text.splitlines():
                hm = _STRATEGY_HDR.match(line)
                if hm:
                    strat = hm.group(1)
                    in_buy = False; continue
                if _BUY_SECTION.match(line):
                    in_buy = True; pending = None; continue
                if _SELL_SECTION.match(line):
                    in_buy = False; continue
                if line.strip() == "---":
                    in_buy = False; continue
                if in_buy and strat:
                    sm = _STOCK_LINE.match(line)
                    if sm:
                        pending = (sm.group(1), sm.group(2).strip())
                        continue
                    if pending:
                        cm = _CLOSE_RE.search(line)
                        if cm:
                            records.append({
                                "日期": rdate, "策略": strat, "方向": "BUY",
                                "代码": pending[0], "名称": pending[1],
                                "价格": float(cm.group(1)),
                            })
                            pending = None

        df_sig = pd.DataFrame(records)
        if df_sig.empty:
            st.warning("未解析到信号")
        else:
            # ── 信号统计卡 ──
            c1, c2, c3 = st.columns(3)
            c1.metric("信号报告数", len(report_files))
            c2.metric("总信号数", len(df_sig))
            c3.metric("覆盖股票数", df_sig["代码"].nunique())

            st.divider()

            # ── 每日信号数 ──
            daily = df_sig.groupby("日期").size().reset_index(name="count")
            fig_d = px.bar(daily, x="日期", y="count", title="每日 BUY 信号数", color="count")
            st.plotly_chart(fig_d, use_container_width=True)

            # ── 策略分布 ──
            col1, col2 = st.columns(2)
            with col1:
                strat_cnt = df_sig.groupby("策略").size().reset_index(name="count")
                fig_s = px.pie(strat_cnt, values="count", names="策略", title="策略信号分布")
                st.plotly_chart(fig_s, use_container_width=True)
            with col2:
                top_codes = df_sig["代码"].value_counts().head(10).reset_index()
                top_codes.columns = ["代码", "次数"]
                fig_c = px.bar(top_codes, x="代码", y="次数", title="被选中最多的股票", color="次数")
                st.plotly_chart(fig_c, use_container_width=True)

            # ── 信号明细 ──
            st.subheader("信号明细")
            st.dataframe(
                df_sig.sort_values(["日期", "策略"]),
                use_container_width=True, hide_index=True,
                column_config={
                    "日期": "日期", "策略": "策略", "方向": "方向",
                    "代码": "代码", "名称": "名称",
                    "价格": st.column_config.NumberColumn("价格", format="%.2f"),
                },
            )


# ══════════════════════════════════════════════════════════════════════
# 个股买卖点走势（所有页面共用，回测跑一次后切换股票不重新跑）
# ══════════════════════════════════════════════════════════════════════

bt = st.session_state.get("bt_result")
if bt and bt.get("trade_log"):
    st.divider()
    st.subheader("📈 个股买卖点走势")
    traded_codes = sorted(set(t["code"] for t in bt["trade_log"]))
    code_pick = st.selectbox("选择股票", traded_codes,
        format_func=lambda c: f"{c} {bt['name_map'].get(c, '')}",
        key="stock_chart_global")

    if code_pick:
        code_trades = [t for t in bt["trade_log"] if t["code"] == code_pick]
        df_stock = store.load_daily(
            codes=[code_pick],
            start=pd.Timestamp(bt["start_date"]) - timedelta(days=30),
            end=pd.Timestamp(bt["end_date"]) + timedelta(days=1),
            adjust=adjust,
        )
        if not df_stock.empty:
            df_stock = df_stock.sort_values("trade_date")
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=df_stock["trade_date"], y=df_stock["close"],
                name="收盘价", line=dict(color="#1f77b4", width=1.5),
            ))
            for t in code_trades:
                dt = pd.Timestamp(t["date"])
                qty_str = f" x{t['qty']}" if t.get('qty') else ""
                if t["side"] == "BUY":
                    color, sym, label = "green", "triangle-up", f"B {t['price']:.2f}{qty_str}"
                else:
                    color, sym, label = "red", "triangle-down", f"S {t['price']:.2f} {t.get('reason','')}"
                fig_s.add_trace(go.Scatter(
                    x=[dt], y=[t["price"]],
                    mode="markers+text",
                    marker=dict(symbol=sym, size=14, color=color),
                    text=[label], textposition="top center",
                    showlegend=False,
                ))
            buys_list = [t for t in code_trades if t["side"] == "BUY"]
            sells_list = [t for t in code_trades if t["side"] == "SELL"]
            for b in buys_list:
                bd = pd.Timestamp(b["date"])
                later = [s for s in sells_list if pd.Timestamp(s["date"]) >= bd]
                sd = pd.Timestamp(later[0]["date"]) if later else df_stock["trade_date"].max()
                fig_s.add_vrect(x0=bd, x1=sd, fillcolor="green", opacity=0.06, line_width=0)
            fig_s.update_layout(
                title=f"{code_pick} {bt['name_map'].get(code_pick,'')} 买卖点",
                xaxis_title="日期", yaxis_title="价格",
                height=400, hovermode="x unified",
            )
            st.plotly_chart(fig_s, use_container_width=True)


# ── 底部 ────────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption(f"数据: {adjust} 复权 | 最新: {store.conn.execute('SELECT MAX(trade_date) FROM daily_bars').fetchone()[0]}")
