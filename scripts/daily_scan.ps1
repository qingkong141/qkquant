# qkquant 每日扫描脚本（供 Windows 任务计划程序调度）
#
# 流程：
#   1) 切到项目根目录
#   2) 增量更新 HS300 日线（默认 auto：优先 akshare，失败再 baostock）
#      勿固定只用 baostock：其日线常滞后 1～2 个交易日，会出现「14 号推送标题仍是 13 号」
#   3) 跑 scan --raw 并推送到已启用通道
# 日志：logs/daily_scan_YYYYMMDD.log
#
# 调度建议：每个交易日 16:30 触发（A股 15:00 收盘，留 1.5 小时给数据源算复权因子）
# 注意：15:30 太早，baostock 当日复权因子可能还没更新好，会拉到脏数据

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Today = Get-Date -Format "yyyyMMdd"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir "daily_scan_$Today.log"

$Qkquant = Join-Path $ProjectRoot ".venv\Scripts\qkquant.exe"
if (-not (Test-Path $Qkquant)) {
    "[$(Get-Date)] qkquant.exe not found at $Qkquant" | Tee-Object -FilePath $LogFile -Append
    exit 1
}

"[$(Get-Date)] === daily_scan start ===" | Tee-Object -FilePath $LogFile -Append

# 1) 增量拉今日数据
"[$(Get-Date)] step 1: update-data --source auto --recent-days 10 ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant update-data --universe hs300 --source auto --recent-days 10 *>> $LogFile
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date)] update-data failed (exit $LASTEXITCODE), continue scan anyway" | Tee-Object -FilePath $LogFile -Append
}

# 2) 跑裸信号扫描并推送 + 自动跟单更新持仓
"[$(Get-Date)] step 2: scan --raw --push --auto-position ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant scan --raw --push --auto-position *>> $LogFile

# 3) 跟踪 watchlist 累计涨跌 + alpha
"[$(Get-Date)] step 3: track --push ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant track --push *>> $LogFile

"[$(Get-Date)] === daily_scan done (exit $LASTEXITCODE) ===" | Tee-Object -FilePath $LogFile -Append
exit $LASTEXITCODE
