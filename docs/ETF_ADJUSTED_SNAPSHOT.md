# 固定ETF池复权行情快照

本工具为冻结的101只ETF收集2024-01-01至冻结清单日期上限的公开行情，建立独立DuckDB，原`data/daily.duckdb`不变。只处理数据，不计算策略收益，也不修改信号规则。

## 来源与校验

- 腾讯按年分段读取`day`与`qfqday`，防止单次640根限制截断历史。六字段响应的最后一列是成交量，不当成成交额。
- 新浪保留原始OHLC、份数、元，以及ETF特有`d/f/s/u`复权记录。支持`f=1`时`前复权价=原始价/s-u`，按事件生效日向后匹配；末段应`s=1,u=0`。
- 每根腾讯原始价格与新浪原始价格、腾讯前复权价格与新浪重建价格分别对比，误差上限0.0005001。份额折算与现金分红分别处理，不把基金净值当交易价格。
- 三份价格序列日期集合必须相同。共同缺行情的日子保留为不可用，不人工填值；单源日期差异仍拒绝。完整交易日历独立保留，持有期不压缩。
- 新浪真实成交量/金额不随价格复权。金额整元舍入容许0.5元，不等于放宽OHLC核验。
- 每份公开响应文本保存URL、请求参数、抓取时间和SHA256；每只ETF另保存核验结果。共同缺失和异常数据都披露，不依据未来收益决定纳入。

官方API实现参考：[AKShare腾讯日线源码](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_hist_tx.py)、[新浪ETF源码](https://github.com/akfamily/akshare/blob/main/akshare/fund/fund_etf_sina.py)。实际接入以归档响应字段及测试为准，不能直接套旧版SDK的列名。

## 使用

运行前提供原冻结清单和完整交易日历。每次输出目录必须新建；完成的快照禁止覆盖。

```powershell
.\.venv\Scripts\python.exe scripts\build_etf_snapshot.py --freeze-manifest <冻结清单.json> --calendar <交易日历.csv> --output <新快照目录>
```

`--download-only`只归档数据；`--offline --audit-only`使用已归档响应核验，不入库；`--offline`核验并建立独立快照。重用的响应先检查来源、参数及文件哈希，不会无声换源。原库若已改变，须明确建立新数据版本，而不能冒用旧冻结清单。

本轮交付的研究输入：

```powershell
.\.venv\Scripts\python.exe scripts\research_signal_value.py --freeze-manifest data\verified_etf_20260918\freeze_manifest.json --database data\verified_etf_20260918\verified_etf_snapshot.duckdb --evaluation-start 2026-07-22 --output reports\etf_extension_v1
```

`snapshot_manifest.json`和`etf_audit.json`说明已通过哪些检查；`official_action_results.json`说明代表性基金公告抽查及额外历史测试的结果。原研究入口的`price_adjustment_verified=False`表示入口本身不重复审计价格，不能代替读取本快照审计凭据。

## 边界

这是公开来源逐日交叉核验，不是交易所逐笔数据认证。两家供应商的最终底层数据是否独立未获证明；基金公告只做代表性抽查。固定池的上市/退市时点资料仍不完整，未消除幸存者偏差。前复权行情也不等于精确的分红现金账务；账户收益和最大回撤仍须另外验证。
