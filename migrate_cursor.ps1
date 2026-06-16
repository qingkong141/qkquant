Write-Host ""
Write-Host "======================================"
Write-Host "  Cursor Migration Script"
Write-Host "  Move Cursor data from C to D drive"
Write-Host "======================================"
Write-Host ""

# Step 1: Kill Cursor processes
Write-Host "[Step 1] Closing Cursor..."
Get-Process -Name "Cursor" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -match "cursor"
} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host "Cursor closed."

function Move-WithJunction($src, $dst, $label) {
    Write-Host ""
    Write-Host "======================================"
    Write-Host "[$label] $src -> $dst"
    Write-Host "======================================"

    if (-not (Test-Path $src)) {
        Write-Host "SKIP: Source not found."
        return
    }

    $item = Get-Item $src -ErrorAction SilentlyContinue
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Write-Host "SKIP: Already a junction link."
        return
    }

    if (Test-Path $dst) {
        Write-Host "SKIP: Destination already exists."
        return
    }

    Write-Host "Copying..."
    Copy-Item -Path $src -Destination $dst -Recurse -Force -ErrorAction Stop

    $s1 = (Get-ChildItem $src -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $s2 = (Get-ChildItem $dst -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $gb = [math]::Round($s2/1GB, 2)

    if ([math]::Abs($s1 - $s2) -lt 1024) {
        Write-Host "Size verified OK (${gb} GB)"
        Remove-Item $src -Recurse -Force -ErrorAction Stop
        Write-Host "Old folder deleted."
        cmd /c mklink /J $src $dst
        Write-Host "Junction link created! DONE!"
    } else {
        Write-Host "ERROR: Size mismatch! Keeping source."
    }
}

Move-WithJunction "C:\Users\Lenovo\AppData\Roaming\Cursor" "D:\CursorData\Roaming" "Step 2: Cursor Roaming (3.08 GB)"

Move-WithJunction "C:\Users\Lenovo\.cursor" "D:\CursorData\.cursor" "Step 3: .cursor config (0.86 GB)"

Move-WithJunction "C:\Users\Lenovo\AppData\Local\Programs\cursor" "D:\CursorData\Programs" "Step 4: Cursor Program (0.76 GB)"

Write-Host ""
Write-Host "======================================"
Write-Host "  All Done!"
Write-Host "======================================"
$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$free = [math]::Round($disk.FreeSpace/1GB, 1)
$usage = [math]::Round(($disk.Size - $disk.FreeSpace)/$disk.Size*100, 1)
Write-Host "C: Free: ${free} GB | Usage: ${usage}%"
Write-Host ""
Write-Host "Now you can reopen Cursor, everything works as before."
Write-Host ""
Read-Host "Press Enter to exit"
