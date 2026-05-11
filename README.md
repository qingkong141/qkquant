# qkquant

A 股量化交易 MVP。从**回测研究**走到**半自动信号推送**，不触碰真实资金。

技术栈：Python 3.12 · `uv` · DuckDB · akshare/baostock · backtrader · pydantic · loguru · typer · ServerChan

## 功能现状

- [x] 行情数据层：akshare / baostock 双源 → 本地 DuckDB（支持增量、auto 回退）
- [x] 交易日历、股票基本信息、指数成分股（沪深 300）
- [x] 技术指标：SMA / EMA / MACD / RSI / ATR / BOLL / KDJ（纯 pandas / backtrader 实现）
- [x] 回测引擎：基于 backtrader，内建 A 股规则
  - T+1（当日买入次日方可卖出）
  - 涨跌停禁买/禁卖（基于 `pct_chg` 拦截）
  - ST 过滤（基于股票池筛选）
  - 交易费用：佣金万 2.5（最低 5 元）+ 印花税千 1 + 滑点 0.2%
- [x] **5 个生产级策略**（见下方"策略一览"）
- [x] 风控模块（`risk/`）：trailing_stop / concentration / portfolio_drawdown / blacklist
- [x] 因子库（`factors/`）：mom_20d/60d/120d, reversal_5d, vol_20d, rsi_14, turnover, amihud + IC 测试 / 分组回测
- [x] **每日信号扫描（`scan` 命令）**：4 策略对每只票算今日入场/出场信号，支持 watchlist 持仓追踪
- [x] **推送通道（`notify.py`）**：ServerChan / 企微机器人 / 飞书机器人，自动推到私人微信
- [x] **Daily 调度脚本**（`scripts/daily_scan.ps1`）：Windows 任务计划程序定时跑
- [x] 绩效报告：净值曲线图、回撤图、summary.md、trades.csv、横向对比报告
- [x] CLI：`update-data` / `backtest` / `compare` / `scan` / `factor-test` / `list-strategies` / `stats`
- [x] 下单通道抽象接口 `BrokerBase`（暂无实现，预留给未来 easytrader / miniQMT）

## 快速上手

### 1. 安装

前置要求：Windows / macOS / Linux，已安装 `uv`（如未安装：`pip install uv`）。

```powershell
cd d:\lsl\qkquant
uv sync --extra dev
```

`uv` 会自动下载 Python 3.12 并安装所有依赖到 `.venv`。

### 2. 首次拉数据（沪深 300 近 3 年日线）

```powershell
uv run qkquant update-data --universe hs300 --since 2022-01-01
```

首次全量大约 5-15 分钟（受 akshare 限频影响）。数据写入 `data/daily.duckdb`。

增量更新（只拉新日期）：

```powershell
uv run qkquant update-data --universe hs300

# 每日任务推荐：只更新最近几天，并发走 akshare，避免 baostock 逐票慢查
uv run qkquant update-data --universe hs300 --source akshare --jobs 8 --recent-days 10
```

#### 数据源选择（代理环境必看）

`--source` 参数可选 `akshare` / `baostock` / `auto`（默认）：

| 场景 | 推荐 |
|---|---|
| 网络通畅、想拿最新/最全 | `--source akshare` |
| 公司/校园 VPN / 代理卡住 HTTPS | `--source baostock`（走 TCP:8081，绕开代理） |
| 不确定、想要一次到位 | `--source auto`（akshare 优先，失败自动回退 baostock） |

```powershell
uv run qkquant update-data --universe hs300 --since 2022-01-01 --source baostock
```

baostock 约滞后 1-2 个交易日，首次拉全市场基本信息约 60-90 秒（5500+ 条逐条流式）。

`--jobs` 只在 `--source akshare` 下启用并发；`auto` / `baostock` 会保持串行，避免 baostock 全局会话并发不稳定。
`--recent-days N` 适合每日增量，只把本次请求窗口收窄到最近 N 个自然日。

查看本地库摘要：

```powershell
uv run qkquant stats
```

### 3. 跑回测

```powershell
# 单策略
uv run qkquant backtest momentum --start 2023-01-01 --end 2025-12-31 --capital 100000

# 横向对比多策略（推荐）
uv run qkquant compare ma_breakout ma_boll momentum momentum_breakout --start 2023-01-01 --end 2025-12-31
```

终端打印核心指标，`reports/<策略>_<时间戳>/` 下生成：

- `summary.md` / `comparison.md` - 报告
- `equity.png` / `equity_overlay.png` - 净值曲线
- `equity.csv` / `equities.csv` - 每日净值
- `trades.csv` - 交易流水
- `rejections.csv` - 被规则拦截的订单

### 4. 每日信号扫描（半自动半实盘）

```powershell
# 跑一次裸信号扫描（不依赖回测状态，直接看今天哪些票满足入场条件）
uv run qkquant scan --raw

# 加 --push 推送到 config/notify.yaml 配置的微信/飞书/企微
uv run qkquant scan --raw --push

# 加 --ai 追加 AI 分析（只解读信号，不生成买卖信号）
uv run qkquant scan --raw --ai
```

输出到 `reports/scan_YYYY-MM-DD_raw.md`，区分"BUY 候选" vs "你的 watchlist 中触发出场"。

AI 分析配置在 `config/ai.yaml`。默认 `provider: noop`，不联网也能生成规则化摘要；
如需接 DeepSeek / OpenAI / 通义等 OpenAI-compatible API，配置 `provider` / `model` /
`base_url`，并把密钥放到 `api_key_env` 指定的环境变量中，不要写进 yaml。

定时调度（Windows 任务计划程序，每个交易日 15:30 自动跑）：

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$pwd\scripts\daily_scan.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At "15:30"
Register-ScheduledTask -TaskName "qkquant_daily_scan" -Action $action -Trigger $trigger
```

### 5. 跑测试

```powershell
uv run pytest
```

14 个单测 + 端到端合成回测，1-2 秒内跑完。

## 策略一览

| 策略 | 核心逻辑 | 累计收益 | 夏普 | MDD | 备注 |
|---|---|---|---|---|---|
| momentum | 20日累计涨幅 > +3% & 距高点 < 10% | **+21.62%** | **0.52** | 10.32% | 单跑最优 |
| momentum_breakout | momentum 强化（紧贴峰值 + 创新高） | +20.47% | 0.44 | 9.53% | 抗滑点最强 |
| **ma_boll** | 双均线 + 布林带（金叉 + 中轨上方 + 不近上轨） | +16.40% | 0.40 | **9.30%** | 与 momentum 相关性 0.438（低相关） |
| ma_breakout | 朴素双均线（5/20） | +7.75% | 0.09 | 6.34% | 教学用 / 基准 |
| relative_strength | 横截面 60 日涨幅排名前 10 | - | - | - | 月频调仓 |

回测区间：HS300 / 2023-01-01 ~ 2026-05-07 / 初始资金 10 万 / 默认风控 + 现实成本

## 目录结构

```
qkquant/
├── pyproject.toml
├── config/
│   ├── settings.yaml            # 全局配置
│   ├── notify.yaml              # 推送通道配置 (gitignore)
│   ├── positions.yaml           # 个人持仓/watchlist (gitignore)
│   └── strategies/
│       ├── ma_breakout.yaml
│       ├── ma_boll.yaml
│       ├── momentum.yaml
│       ├── momentum_breakout.yaml
│       └── relative_strength.yaml
├── data/                        # DuckDB 本地库 (gitignore)
├── logs/                        # 日志 (gitignore)
├── reports/                     # 回测/扫描报告 (gitignore)
├── scripts/
│   ├── daily_scan.ps1           # Windows 每日调度脚本
│   ├── sensitivity_test.py      # 成本敏感性测试
│   ├── ma_boll_sweep.py         # BOLL 参数扫描
│   └── combo_test.py            # 策略组合相关性测试
├── src/qkquant/
│   ├── config.py / logger.py / cli.py
│   ├── scan.py                  # 每日信号扫描 (回测模式 + 裸信号模式)
│   ├── notify.py                # 推送通道 (ServerChan / 企微 / 飞书)
│   ├── data/                    # DuckDB + akshare/baostock fetcher
│   ├── factors/                 # 因子库 + IC/分组回测 runner
│   ├── strategy/                # 5 个生产策略 + base + registry
│   ├── risk/                    # 风控规则 (trailing/concentration/blacklist 等)
│   ├── backtest/                # backtrader 引擎 + 报告生成
│   └── broker/                  # 下单通道抽象 (实现待补)
└── tests/
```

## A 股回测规则速查

| 规则 | 实现位置 |
|---|---|
| 后复权价格喂策略 | `update-data --adjust hfq` 默认 |
| T+1 | `engine.BtStrategyBase.safe_sell` |
| 涨停禁买 | `engine.BtStrategyBase.safe_buy` |
| 跌停禁卖 | `engine.BtStrategyBase.safe_sell` |
| ST 过滤 | `cli.update_data` + `cli.backtest` 自动排除 |
| 佣金/印花税 | `engine.AShareCommission` |
| 滑点 | `broker.set_slippage_perc(0.002)` |

## 实验日志（回测验证过的真知识）

记录策略研发过程中通过回测**证伪**或**证实**的核心 insight，避免未来重复踩坑。
回测条件统一：HS300 / 2023-01-01 ~ 2026-05-07 / 10 万初始资金 / 默认风控。

### ❌ 失败实验（已删除）

| 策略 | 假说 | 累计 | 夏普 | 失败原因 |
|---|---|---|---|---|
| momentum_pullback | 等回调到 MA5 再买（避免追高） | -13.67% | -0.49 | A 股趋势性弱，回调时趋势已结束；机构同步出货 |
| turtle | 海龟 System 1（20 日突破 + ATR + 金字塔加仓） | -23.32% | -1.00 | 仅多头 + 震荡市；海龟核心战场是期货双向 |
| ma_breakout_v2 | 放量金叉过滤 | +1.76% | -0.18 | HS300 流动性极好，量能信号是噪音 |
| ma_breakout_quiet | 缩量金叉（反向假说） | -7.08% | -1.03 | 缩量金叉同样是噪音 |
| ma_kdj | MA20 上方 + KD 金叉 + J<60 | +0.23% | -0.30 | KDJ 反转指标在大盘股完全失效 |
| momentum_boll_combo | 70% momentum + 30% ma_boll 组合 | +10.74% | 0.19 | 虚拟组合需 200k+ 才能实现，100k 资金内 ma_boll tier 仅 3 只过度集中 |

### 💡 通用经验（**比策略本身更重要**）

1. **A 股 HS300 上有效信号仅来自趋势 + 波动率**（动量、BOLL 系）；反转/超买超卖（KDJ、量能反向）在大盘股全是噪音
2. **教科书"放量金叉"和"等回调买入"在 A 股反而无效**，因为机构主导市场会反向利用散户教科书
3. **滑点 ≫ 佣金**对策略的影响：5 万账户跟券商谈佣金率没意义（吃 5 元最低）；高频策略在最坏滑点下直接由正变负
4. **虚拟组合 alpha 有容量门槛**：加权回测的 sharpe 改善在小资金账户内不可复现（需要 200k+）
5. **回测海龟在 A 股 HS300 上完全失效**：跌 23%；验证"被神化的策略也要回测"
6. **参数过拟合迹象**：相邻参数稳定性优于绝对最优；BOLL 的 dev=2.0 整行正夏普 = 真信号，dev=2.5 单点尖峰 = overfit
7. **多策略共振信号通常更可靠**：但只在低相关策略间组合才有 diversification 价值（momentum × ma_boll 相关性 0.438 ✓ vs momentum × momentum_breakout 相关性 0.784 ✗）

### 🛠️ 辅助实验脚本

- `scripts/sensitivity_test.py` — 成本敏感性（5 场景 × 3 策略）
- `scripts/ma_boll_sweep.py` — BOLL 参数网格扫描（dev × upper_buffer）
- `scripts/combo_test.py` — 策略组合相关性 + 加权日收益模拟

## 后续路线

### 已完成（本轮）

- ✅ 风控层（`src/qkquant/risk/`）：trailing_stop / concentration / portfolio_drawdown / blacklist
- ✅ 通知（`src/qkquant/notify.py`）：ServerChan / 企微机器人 / 飞书机器人
- ✅ 调度脚本（`scripts/daily_scan.ps1`）：Windows 任务计划程序
- ✅ 因子框架（`src/qkquant/factors/`）：IC 测试 + 分组回测
- ✅ 5 个生产策略 + 6 个失败实验的回测证据

### 待做

1. **实盘通道**：
   - `broker/easytrader_impl.py`：接同花顺 / 通达信客户端，适合小资金
   - `broker/simulator.py`：本地模拟盘（真实行情，虚拟账户）
   - `broker/qmt_impl.py`：等资金凑够 50 万后切 miniQMT (`xtquant`)
2. **财报黑名单过滤**：用 akshare 拉财报数据，BUY 信号触发时过滤"营收/净利暴雷"的票
3. **业绩预告窗口规避**：财报披露前后 N 天禁止开新仓
4. **小盘股 / 中证 1000 适配**：当前所有结论仅在 HS300 验证；小盘股池可能有不同的有效信号（比如 KDJ/量能在小盘可能 work）
5. **真实 Diversification 实现**：账户拆分跑两份策略，或扩大 max_positions 让 combo 满血
6. **研究平台**：Jupyter notebook 模板做策略原型设计

## 常识性风险提示

- akshare 是社区项目，接口偶尔失效。本仓库已内置 baostock 作二源冗余（`--source auto` 默认启用回退）；生产场景可再叠加 tushare pro。
- backtrader 日频够用，但分钟/高频场景建议切 `vnpy` 或 `qlib`。
- **所有回测结果都是对过去的拟合**，**实盘前务必至少 1 个月模拟盘 + 小资金验证**。
- 即使未来接入 easytrader，A 股客户端每日盘前需要人工登录一次（验证码 / U 盾）。这是客户端层面的限制，无法绕过。

## 许可

MIT
