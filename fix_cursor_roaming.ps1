Write-Host "======================================"
Write-Host "  Fix: Move Cursor Roaming to D drive"
Write-Host "======================================"

$src = "C:\Users\Lenovo\AppData\Roaming\Cursor"
$dst = "D:\CursorData\Roaming"

# Step 1: Kill all Cursor and related processes
Write-Host ""
Write-Host "[1] Closing ALL Cursor processes..."
Get-Process | Where-Object { $_.Name -match "cursor" -or $_.Name -match "Cursor" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
# Kill any remaining node processes from Cursor
Get-Process -Name "node" -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $p = $_.Path
        if ($p -and $p -match "cursor") {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}
Start-Sleep -Seconds 3
Write-Host "Processes killed."

# Step 2: Check source
$item = Get-Item $src -ErrorAction SilentlyContinue
if (-not $item) {
    Write-Host "Source not found: $src"
    Read-Host "Press Enter to exit"
    exit
}
if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Write-Host "Already a junction link! Nothing to do."
    Read-Host "Press Enter to exit"
    exit
}

# Step 3: D:\CursorData\Roaming already has a copy, verify and replace
if (Test-Path $dst) {
    Write-Host ""
    Write-Host "[2] D:\CursorData\Roaming already exists, refreshing copy..."
    Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "[3] Copying $src -> $dst ..."
Copy-Item -Path $src -Destination $dst -Recurse -Force -ErrorAction Stop

$s1 = (Get-ChildItem $src -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$s2 = (Get-ChildItem $dst -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$gb = [math]::Round($s2/1GB, 2)

if ([math]::Abs($s1 - $s2) -lt 4096) {
    Write-Host "Size verified OK (${gb} GB)"
    
    Write-Host "[4] Deleting old folder..."
    Remove-Item $src -Recurse -Force -ErrorAction Stop
    
    if (-not (Test-Path $src)) {
        Write-Host "[5] Creating junction link..."
        cmd /c mklink /J $src $dst
        Write-Host ""
        Write-Host "SUCCESS! Cursor Roaming moved to D drive."
    } else {
        Write-Host "ERROR: Could not delete $src. Some files still locked."
        Write-Host "Try closing ALL programs and run this script again."
    }
} else {
    Write-Host "ERROR: Size mismatch (src=$s1 dst=$s2). Aborted."
}

Write-Host ""
$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$free = [math]::Round($disk.FreeSpace/1GB, 1)
Write-Host "C: Free: ${free} GB"
Write-Host ""
Read-Host "Press Enter to exit"
