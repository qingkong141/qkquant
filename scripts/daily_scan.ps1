# qkquant 每日扫描脚本（供 Windows 任务计划程序调度）
#
# 流程：
#   1) 切到项目根目录
#   2) 增量更新 HS300 日线（baostock 更稳，akshare/东财接口易受网络影响）
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
"[$(Get-Date)] step 1: update-data --source baostock --recent-days 10 ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant update-data --universe hs300 --source baostock --recent-days 10 *>> $LogFile
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date)] update-data failed (exit $LASTEXITCODE), continue scan anyway" | Tee-Object -FilePath $LogFile -Append
}

# 2) 跑裸信号扫描并推送
"[$(Get-Date)] step 2: scan --raw --push ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant scan --raw --push *>> $LogFile

# 3) 跟踪 watchlist 累计涨跌 + alpha
"[$(Get-Date)] step 3: track --push ..." | Tee-Object -FilePath $LogFile -Append
& $Qkquant track --push *>> $LogFile

"[$(Get-Date)] === daily_scan done (exit $LASTEXITCODE) ===" | Tee-Object -FilePath $LogFile -Append
exit $LASTEXITCODE
