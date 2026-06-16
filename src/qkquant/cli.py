"""qkquant 命令行入口（typer）。

子命令：
- update-data    拉/增量更新日线到本地 DuckDB
- backtest       跑回测并生成报告
- list-strategies 列出内置策略
- stats          查看本地数据库摘要
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from qkquant.config import get_settings
from qkquant.data.board import is_cn_main_board
from qkquant.data.fetcher import DataFetcher
from qkquant.data.storage import DuckStore
from qkquant.logger import logger, setup_logger
from qkquant.strategy.registry import (
    get_strategy,
    list_strategies,
    load_risk_config,
    load_strategy_config,
)

app = typer.Typer(
    help="qkquant - A股量化交易 MVP CLI",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _default_start() -> str:
    return (date.today() - timedelta(days=3 * 365)).isoformat()


def _today_str() -> str:
    return date.today().isoformat()


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
) -> None:
    setup_logger(level=log_level)


# ---------------- update-data ----------------


@app.command("update-data")
def update_data(
    since: str = typer.Option(_default_start(), "--since", help="起始日期 YYYY-MM-DD"),
    until: str = typer.Option(_today_str(), "--until", help="结束日期 YYYY-MM-DD"),
    universe: str = typer.Option(
        "hs300",
        "--universe",
        help="股票池: hs300 / main_board / all / custom（搭配 --codes 使用）",
    ),
    codes: Optional[str] = typer.Option(
        None, "--codes", help="逗号分隔的自定义代码列表（universe=custom 时用）"
    ),
    adjust: str = typer.Option("hfq", "--adjust", help="复权方式 hfq/qfq/''"),
    full: bool = typer.Option(False, "--full", help="全量重拉（默认增量）"),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="只拉前 N 只，用于快速测试"
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="数据源: akshare(最全但依赖 HTTPS) / baostock(走 TCP,代理友好) / auto(akshare 优先失败回退)",
    ),
    jobs: int = typer.Option(
        1,
        "--jobs",
        min=1,
        help="并发抓取线程数；仅 source=akshare 时启用，baostock/auto 保持串行",
    ),
    recent_days: Optional[int] = typer.Option(
        None,
        "--recent-days",
        min=1,
        help="只更新最近 N 个自然日，用于每日增量任务；不影响 --full 的语义",
    ),
) -> None:
    """拉取/增量更新日线到本地 DuckDB。"""
    if source not in ("akshare", "baostock", "auto"):
        raise typer.BadParameter(f"unknown --source: {source}")
    store = DuckStore()
    fetcher = DataFetcher(store=store, source=source)  # type: ignore[arg-type]

    # 1. 股票池
    if universe == "custom":
        if not codes:
            raise typer.BadParameter("--universe custom 需要配合 --codes")
        target_codes = [c.strip() for c in codes.split(",") if c.strip()]
    elif universe == "hs300":
        logger.info("fetching HS300 constituents ...")
        target_codes = fetcher.fetch_hs300_constituents()
        store.upsert_index_constituents("000300", target_codes)
    elif universe == "all":
        logger.info("fetching full A-share universe ...")
        df = fetcher.fetch_a_share_universe()
        store.upsert_instruments(df)
        target_codes = df[~df["is_st"]]["code"].tolist()
    elif universe == "main_board":
        logger.info("fetching main-board A-share universe ...")
        df = fetcher.fetch_a_share_universe()
        store.upsert_instruments(df)
        df = df[~df["is_st"]].copy()
        df["code"] = df["code"].astype(str).str.zfill(6)
        target_codes = [c for c in df["code"].tolist() if is_cn_main_board(c)]
        logger.info(f"main_board filtered: {len(target_codes)} codes")
    else:
        raise typer.BadParameter(f"unknown universe: {universe}")

    if limit:
        target_codes = target_codes[:limit]

    # 2. 更新股票基本信息（顺带做 ST 过滤）
    try:
        inst_df = fetcher.fetch_a_share_universe()
        store.upsert_instruments(inst_df)
        st_set = set(inst_df[inst_df["is_st"]]["code"].tolist())
        before = len(target_codes)
        target_codes = [c for c in target_codes if c not in st_set]
        logger.info(f"filtered ST: {before} -> {len(target_codes)}")
    except Exception as e:
        logger.warning(f"instrument fetch failed (continue without ST filter): {e}")

    # 3. 拉日线
    effective_since = since
    if recent_days and not full:
        until_d = pd.to_datetime(until).date()
        recent_start = until_d - timedelta(days=recent_days - 1)
        configured_start = pd.to_datetime(since).date()
        effective_since = max(configured_start, recent_start).isoformat()
    console.print(
        f"[bold]updating {len(target_codes)} codes[/bold] [{effective_since} -> {until}] "
        f"adjust={adjust} incremental={not full} jobs={jobs if source == 'akshare' else 1}"
    )
    summary = fetcher.bulk_update_daily(
        codes=target_codes,
        start=effective_since,
        end=until,
        adjust=adjust,
        incremental=not full,
        jobs=jobs,
    )
    fetcher.close()
    # 4. 更新市值
    if len(target_codes) <= 500:
        console.print("[dim]fetching market caps ...[/dim]")
        try:
            mc_fetcher = DataFetcher(store=store, source=source)
            mc_df = mc_fetcher.fetch_market_caps(target_codes)
            mc_fetcher.close()
            if not mc_df.empty:
                n_mc = store.upsert_market_caps(mc_df)
                console.print(f"[dim]market caps updated: {n_mc} codes[/dim]")
        except Exception as e:
            logger.warning(f"market cap fetch failed (non-critical): {e}")
    stats = store.stats()
    store.close()

    console.print("[bold green]done[/bold green]")
    console.print(summary)
    console.print(f"db stats: {stats}")


# ---------------- backtest ----------------


@app.command("backtest")
def backtest(
    strategy: str = typer.Argument(..., help="策略名，例如 momentum"),
    start: str = typer.Option("2023-01-01", "--start", help="回测开始日期"),
    end: str = typer.Option(_today_str(), "--end", help="回测结束日期"),
    capital: float = typer.Option(30_000.0, "--capital", help="初始资金"),
    universe: Optional[str] = typer.Option(
        None,
        "--universe",
        help="hs300 / main_board / custom；省略则读策略 yaml universe.source，再无则 hs300",
    ),
    codes: Optional[str] = typer.Option(None, "--codes", help="自定义代码列表"),
    limit: Optional[int] = typer.Option(None, "--limit", help="只取前 N 只（调试用）"),
    no_risk: bool = typer.Option(
        False, "--no-risk", help="关闭 yaml 里的风控配置（用于 A/B 对比）"
    ),
) -> None:
    """跑回测并生成报告。"""
    from qkquant.backtest.engine import BacktestEngine
    from qkquant.backtest.report import generate_report, print_summary
    from qkquant.risk import RiskConfig

    info = get_strategy(strategy)
    cfg = load_strategy_config(info)

    store = DuckStore()

    cfg_univ = (cfg.get("universe") or {}).get("source")
    eff_univ = (universe or cfg_univ or "hs300")
    if isinstance(eff_univ, str):
        eff_univ = eff_univ.strip().lower()

    target_codes = _resolve_universe(store, eff_univ, codes)

    if limit:
        target_codes = target_codes[:limit]

    params = (cfg.get("params") or {}) if cfg else {}
    risk_cfg = RiskConfig() if no_risk else load_risk_config(cfg)
    if risk_cfg.any_enabled():
        console.print(
            f"[dim]risk rules enabled: "
            f"{[n for n in ('blacklist','concentration','position_stop','trailing_stop','portfolio_drawdown') if getattr(risk_cfg, n).enabled]}[/dim]"
        )
    engine = BacktestEngine(
        store=store,
        strategy_cls=info.cls,
        strategy_params=params,
        initial_capital=capital,
        risk_config=risk_cfg,
    )
    result = engine.run(codes=target_codes, start=start, end=end)

    print_summary(result)
    tag = strategy if risk_cfg.any_enabled() else f"{strategy}_norisk"
    out_dir = generate_report(result, strategy_name=tag)
    console.print(f"[bold green]report:[/bold green] {out_dir}")
    store.close()


# ---------------- compare ----------------


@app.command("compare")
def compare_cmd(
    strategies: list[str] = typer.Argument(
        ..., help="要对比的策略名列表，例如: momentum ma_boll"
    ),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option(_today_str(), "--end"),
    capital: float = typer.Option(30_000.0, "--capital"),
    universe: str = typer.Option(
        "hs300", "--universe", help="hs300 / main_board / custom"
    ),
    codes: Optional[str] = typer.Option(None, "--codes"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    no_risk: bool = typer.Option(
        False, "--no-risk", help="关闭所有策略的 yaml 风控配置"
    ),
) -> None:
    """同一股票池 / 时间段上跑多个策略并输出横向对比。"""
    from qkquant.backtest.engine import BacktestEngine
    from qkquant.backtest.report import generate_compare_report, print_compare
    from qkquant.risk import RiskConfig

    store = DuckStore()

    target_codes = _resolve_universe(store, universe.strip().lower(), codes)

    if limit:
        target_codes = target_codes[:limit]

    results: dict[str, dict] = {}
    for name in strategies:
        info = get_strategy(name)
        cfg = load_strategy_config(info)
        params = (cfg.get("params") or {}) if cfg else {}
        risk_cfg = RiskConfig() if no_risk else load_risk_config(cfg)
        label = name if risk_cfg.any_enabled() else f"{name}_norisk"
        console.print(f"[cyan]running {label} ...[/cyan]")
        engine = BacktestEngine(
            store=store,
            strategy_cls=info.cls,
            strategy_params=params,
            initial_capital=capital,
            risk_config=risk_cfg,
        )
        results[label] = engine.run(codes=target_codes, start=start, end=end)

    store.close()

    print_compare(results)
    out_dir = generate_compare_report(results)
    console.print(f"[bold green]compare report:[/bold green] {out_dir}")


# ---------------- factor-test ----------------


@app.command("factor-test")
def factor_test_cmd(
    factor: str = typer.Argument(..., help="因子名，例如 mom_20d；运行 list-factors 查看全部"),
    start: str = typer.Option("2022-01-04", "--start", help="回测开始日期"),
    end: str = typer.Option(_today_str(), "--end", help="回测结束日期"),
    universe: str = typer.Option(
        "hs300", "--universe", help="股票池 hs300 / main_board / custom"
    ),
    codes: Optional[str] = typer.Option(None, "--codes", help="自定义代码列表"),
    primary_forward: int = typer.Option(5, "--primary-forward", help="分组回测持有期（日）"),
    horizons: str = typer.Option("1,5,10,20", "--horizons", help="IC 评估的 forward 日列表"),
    n_groups: int = typer.Option(5, "--n-groups", help="分组数"),
) -> None:
    """单因子评估：IC 统计 + 分组回测 + 多空曲线。"""
    from qkquant.factors.library import FACTOR_REGISTRY
    from qkquant.factors.runner import evaluate_factor

    if factor not in FACTOR_REGISTRY:
        raise typer.BadParameter(
            f"unknown factor: {factor}. Available: {list(FACTOR_REGISTRY)}"
        )

    horizon_tuple = tuple(int(x) for x in horizons.split(",") if x.strip())
    if primary_forward not in horizon_tuple:
        horizon_tuple = tuple(sorted({*horizon_tuple, primary_forward}))

    store = DuckStore()
    target_codes = _resolve_universe(store, universe, codes)

    result = evaluate_factor(
        store=store,
        factor_name=factor,
        codes=target_codes,
        start=start,
        end=end,
        horizons=horizon_tuple,
        primary_horizon=primary_forward,
        n_groups=n_groups,
        write_report=True,
    )
    store.close()

    # 终端概要
    spec = result["spec"]
    console.print(f"\n[bold]{spec.name}[/bold] - {spec.description}")
    console.print(f"预期方向: {'+1 越大越好' if spec.expected_sign > 0 else '-1 越小越好'}")
    table = Table(title="IC 统计")
    table.add_column("horizon", style="cyan")
    table.add_column("IC mean", justify="right")
    table.add_column("ICIR", justify="right")
    table.add_column("t-stat", justify="right")
    table.add_column("IC>0", justify="right")
    table.add_column("N", justify="right")
    for h in sorted(result["ic_stats_by_horizon"]):
        s = result["ic_stats_by_horizon"][h]
        table.add_row(
            f"{h}d",
            f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "n/a",
            f"{s['ic_ir']:+.3f}" if pd.notna(s["ic_ir"]) else "n/a",
            f"{s['t_stat']:+.2f}" if pd.notna(s["t_stat"]) else "n/a",
            f"{s['positive_ratio']:.1%}" if pd.notna(s["positive_ratio"]) else "n/a",
            str(s["n_samples"]),
        )
    console.print(table)
    console.print(f"[bold green]report:[/bold green] {result['report_dir']}")


@app.command("factor-test-all")
def factor_test_all_cmd(
    start: str = typer.Option("2022-01-04", "--start"),
    end: str = typer.Option(_today_str(), "--end"),
    universe: str = typer.Option(
        "hs300", "--universe", help="hs300 / main_board / custom"
    ),
    codes: Optional[str] = typer.Option(None, "--codes"),
    primary_forward: int = typer.Option(5, "--primary-forward"),
    horizons: str = typer.Option("1,5,10,20", "--horizons"),
) -> None:
    """批量评估所有注册因子，产出横向对比报告。"""
    from qkquant.factors.runner import evaluate_all_factors

    horizon_tuple = tuple(int(x) for x in horizons.split(",") if x.strip())
    if primary_forward not in horizon_tuple:
        horizon_tuple = tuple(sorted({*horizon_tuple, primary_forward}))

    store = DuckStore()
    target_codes = _resolve_universe(store, universe, codes)

    out_dir = evaluate_all_factors(
        store=store,
        codes=target_codes,
        start=start,
        end=end,
        horizons=horizon_tuple,
        primary_horizon=primary_forward,
    )
    store.close()
    console.print(f"[bold green]compare report:[/bold green] {out_dir}")


@app.command("list-factors")
def list_factors_cmd() -> None:
    """列出所有内置因子。"""
    from qkquant.factors.library import FACTOR_REGISTRY

    table = Table(title="内置因子")
    table.add_column("name", style="cyan")
    table.add_column("expected", justify="center")
    table.add_column("description")
    for name, spec in FACTOR_REGISTRY.items():
        table.add_row(
            name,
            "+1" if spec.expected_sign > 0 else "-1",
            spec.description,
        )
    console.print(table)


def _resolve_universe(store: DuckStore, universe: str, codes: Optional[str]) -> list[str]:
    """股票池解析的小工具（被 backtest / compare / factor-test 复用）。"""
    u = universe.strip().lower()
    if u == "custom":
        if not codes:
            raise typer.BadParameter("--universe custom 需要 --codes")
        target_codes = [c.strip() for c in codes.split(",") if c.strip()]
    elif u == "main_board":
        target_codes = store.load_main_board_codes()
        if not target_codes:
            console.print(
                "[yellow]本地无主板的非 ST 标的（instruments 为空或未同步）。"
                "请先运行 `qkquant update-data --universe main_board` 或 `--universe all`。[/yellow]"
            )
            raise typer.Exit(code=2)
    elif u == "hs300":
        target_codes = store.load_index_constituents("000300")
        if not target_codes:
            console.print(
                "[yellow]请先运行 `qkquant update-data --universe hs300`[/yellow]"
            )
            raise typer.Exit(code=2)
    else:
        raise typer.BadParameter(
            f"unknown universe: {universe}. Use hs300 / main_board / custom"
        )

    inst = store.load_instruments(target_codes)
    if not inst.empty and "is_st" in inst.columns:
        st_codes = set(inst[inst["is_st"]]["code"].tolist())
        target_codes = [c for c in target_codes if c not in st_codes]
    return target_codes


# ---------------- scan ----------------


@app.command("scan")
def scan_cmd(
    strategies: str = typer.Option(
        "momentum,ma_boll",
        "--strategies",
        help="逗号分隔的策略名；默认只跑 momentum,ma_boll",
    ),
    universe: str = typer.Option(
        "hs300", "--universe", help="股票池 hs300 / main_board / custom"
    ),
    codes: Optional[str] = typer.Option(None, "--codes", help="自定义代码"),
    as_of: Optional[str] = typer.Option(
        None, "--as-of", help="信号日 YYYY-MM-DD（默认数据库最新日）"
    ),
    holdings_file: Optional[str] = typer.Option(
        None, "--holdings", help="持仓 yaml 路径，默认 config/positions.yaml"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="股票池只取前 N 只（调试）"),
    save: bool = typer.Option(
        True, "--save/--no-save", help="保存到 reports/scan_YYYY-MM-DD.md"
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="裸信号模式：不跑回测，直接对每只股票算今日入场条件（忽略模拟仓和风控熔断）",
    ),
    ai: bool = typer.Option(
        False,
        "--ai/--no-ai",
        help="追加 AI 分析（仅解释裸信号，不改变策略结果）",
    ),
    push: bool = typer.Option(
        False, "--push", help="把信号推送到 config/notify.yaml 配置的通道"
    ),
) -> None:
    """每日信号扫描：跑历史回测到 as_of，输出今天的 BUY/SELL 名单。"""
    from qkquant.config import PROJECT_ROOT
    from qkquant.scan import (
        format_raw_signals,
        format_signals,
        latest_data_date,
        load_holdings,
        scan_all,
        scan_raw,
    )

    store = DuckStore()

    target_codes = _resolve_universe(store, universe, codes)
    if limit:
        target_codes = target_codes[:limit]

    strat_list = [s.strip() for s in strategies.split(",") if s.strip()]
    as_of_d = pd.to_datetime(as_of).date() if as_of else latest_data_date(store)
    if as_of_d is None:
        raise typer.Exit("no data in db, run `qkquant update-data` first")

    mode_label = "raw" if raw else "sim"
    console.print(
        f"[cyan]scanning {len(strat_list)} strategies on {len(target_codes)} codes "
        f"as of {as_of_d} (mode={mode_label}) ...[/cyan]"
    )

    holdings_path = (
        Path(holdings_file)
        if holdings_file
        else PROJECT_ROOT / "config" / "positions.yaml"
    )
    holdings = load_holdings(holdings_path)
    if holdings:
        console.print(f"[dim]loaded {len(holdings)} holdings from {holdings_path}[/dim]")

    inst = store.load_instruments(target_codes)
    name_map = dict(zip(inst["code"], inst["name"])) if not inst.empty else {}

    if raw:
        results = scan_raw(store, strat_list, target_codes, holdings, as_of=as_of_d)
        text = format_raw_signals(results, holdings, name_map=name_map, as_of=as_of_d)
    else:
        signals = scan_all(store, strat_list, target_codes, as_of=as_of_d)
        text = format_signals(signals, holdings, name_map=name_map)

    if ai:
        if raw:
            from qkquant.ai import analyze_raw_signals, format_ai_section, load_ai_config

            ai_cfg = load_ai_config(PROJECT_ROOT / "config" / "ai.yaml")
            ai_response = analyze_raw_signals(
                results,
                holdings,
                name_map=name_map,
                as_of=as_of_d,
                strategies=strat_list,
                universe_size=len(target_codes),
                config=ai_cfg,
            )
            text = f"{text}\n\n{format_ai_section(ai_response)}"
        else:
            text = (
                f"{text}\n\n## AI 分析\n\n"
                "AI 分析当前仅支持 `--raw` 裸信号模式；原始扫描结果不受影响。"
            )

    if save:
        out_dir = PROJECT_ROOT / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_raw" if raw else ""
        out_path = out_dir / f"scan_{as_of_d}{suffix}.md"
        out_path.write_text(text, encoding="utf-8")
        console.print(f"[bold green]saved:[/bold green] {out_path}")

    if push:
        from qkquant.notify import load_notifiers, push_all

        notify_path = PROJECT_ROOT / "config" / "notify.yaml"
        notifiers = load_notifiers(notify_path)
        if not notifiers:
            console.print(
                f"[yellow]no notifier enabled in {notify_path}; skip push[/yellow]"
            )
        else:
            title = f"qkquant 信号 {as_of_d} ({mode_label})"
            n_ok = push_all(notifiers, title, text)
            console.print(
                f"[green]pushed to {n_ok}/{len(notifiers)} channel(s)[/green]"
            )

    # 直接走 stdout 避免 Windows GBK 控制台与 rich 的编码冲突
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(text)

    store.close()


# ---------------- track ----------------


@app.command("track")
def track_cmd(
    holdings_file: Optional[str] = typer.Option(
        None, "--holdings", help="持仓 yaml, 默认 config/positions.yaml"
    ),
    as_of: Optional[str] = typer.Option(
        None, "--as-of", help="跟踪截止日 YYYY-MM-DD（默认数据库最新日）"
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="保存到 reports/track_YYYY-MM-DD.md"
    ),
    push: bool = typer.Option(
        False, "--push", help="推送到 config/notify.yaml 配置的通道"
    ),
) -> None:
    """跟踪 watchlist/持仓: 累计涨跌 + HS300等权基准 + alpha。"""
    from qkquant.config import PROJECT_ROOT
    from qkquant.scan import latest_data_date, load_holdings
    from qkquant.watchlist import format_track_report, track_holdings

    store = DuckStore()
    holdings_path = (
        Path(holdings_file)
        if holdings_file
        else PROJECT_ROOT / "config" / "positions.yaml"
    )
    holdings = load_holdings(holdings_path)
    if not holdings:
        console.print(f"[yellow]no holdings in {holdings_path}[/yellow]")
        store.close()
        raise typer.Exit()

    as_of_d = pd.to_datetime(as_of).date() if as_of else latest_data_date(store)
    if as_of_d is None:
        console.print("[red]no data in db, run `qkquant update-data` first[/red]")
        store.close()
        raise typer.Exit(code=1)

    track = track_holdings(store, holdings, as_of=as_of_d)
    text = format_track_report(track)

    if save:
        out_dir = PROJECT_ROOT / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"track_{as_of_d}.md"
        out_path.write_text(text, encoding="utf-8")
        console.print(f"[bold green]saved:[/bold green] {out_path}")

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(text)

    if push:
        from qkquant.notify import load_notifiers, push_all

        notify_path = PROJECT_ROOT / "config" / "notify.yaml"
        notifiers = load_notifiers(notify_path)
        if not notifiers:
            console.print(
                f"[yellow]no notifier enabled in {notify_path}; skip push[/yellow]"
            )
        else:
            title = f"qkquant Watchlist {as_of_d}"
            n_ok = push_all(notifiers, title, text)
            console.print(
                f"[green]pushed to {n_ok}/{len(notifiers)} channel(s)[/green]"
            )

    store.close()


# ---------------- list-strategies ----------------


@app.command("list-strategies")
def list_strategies_cmd() -> None:
    """列出所有内置策略。"""
    table = Table(title="内置策略")
    table.add_column("name", style="cyan")
    table.add_column("description")
    table.add_column("config")
    for s in list_strategies():
        cfg = str(s.config_path.relative_to(Path.cwd())) if s.config_path and s.config_path.exists() else "-"
        table.add_row(s.name, s.description, cfg)
    console.print(table)


# ---------------- stats ----------------


@app.command("stats")
def stats_cmd() -> None:
    """查看本地 DuckDB 摘要。"""
    store = DuckStore()
    s = store.stats()
    store.close()
    console.print(s)


if __name__ == "__main__":
    app()
