# NOVA - one command to run the demo with both windows side by side.
#
#   .\run_demo.ps1              live
#   .\run_demo.ps1 -Record      live, and write demo.mp4
#
# The CARLA 3D view and the NOVA dashboard are separate OS windows - CARLA's
# belongs to the simulator process, so they cannot be merged. This lays them
# out for you: simulator on the left, dashboard on the right. Record the
# DESKTOP, not a single window.
param(
    [switch]$Record,
    [string]$Town = "Town01",
    [double]$VTarget = 6.0,
    [int]$Vehicles = 10,
    [int]$Walkers = 3,
    # A DIFFERENT ROUTE EVERY RUN, unless you ask for one you have seen.
    #
    # nova_drive.py defaults to seed 42, and this script never overrode it,
    # so every demo started from the same spawn point and met the same
    # traffic in the same order. Fine for debugging, poor for a demo - and
    # it meant one unlucky spawn was unlucky on every single run.
    #
    #   .\run_demo.ps1              a new route each time
    #   .\run_demo.ps1 -Seed 3      a run verified clean on 2026-09-09
    #
    # VERIFIED SEEDS: 1, 3, 11, 23 all pull away immediately and drive with
    # zero collisions. Seed 7 spawns somewhere the planner will not leave and
    # needs ~12 s of StuckRecovery reversing before it gets going - it does
    # recover, but it is not what you want on stage. If a run starts badly,
    # press Q and start again, or pass one of the verified seeds.
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 100000)
)

# --- window positioning ------------------------------------------------
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win {
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int t, bool r);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
}
"@

function Move-AppWindow([string]$namePart, [int]$x, [int]$y, [int]$w, [int]$h) {
    foreach ($try in 1..25) {
        $p = Get-Process | Where-Object {
            $_.MainWindowTitle -and $_.MainWindowTitle -like "*$namePart*"
        } | Select-Object -First 1
        if ($p) { [Win]::MoveWindow($p.MainWindowHandle, $x, $y, $w, $h, $true) | Out-Null; return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

Write-Host "stopping anything already running..." -ForegroundColor DarkGray
Get-Process python,CarlaUE4-Win64-Shipping,CarlaUE4 -ErrorAction SilentlyContinue |
    Stop-Process -Force
Start-Sleep -Seconds 8

Write-Host "starting CARLA..." -ForegroundColor Cyan
Start-Process -FilePath "C:\CARLA_0.9.16\CarlaUE4.exe" -ArgumentList @(
    "-quality-level=Low", "-windowed", "-ResX=860", "-ResY=900", "-dx11",
    "-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=1536")

$ready = $false
foreach ($i in 1..60) {
    Start-Sleep -Seconds 3
    & "$PSScriptRoot\.venv\Scripts\python.exe" -c @"
import carla, sys
try:
    c = carla.Client('127.0.0.1', 2000); c.set_timeout(5.0)
    c.get_world().get_map().name
    sys.exit(0)
except Exception:
    sys.exit(1)
"@ 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
}
if (-not $ready) { Write-Host "CARLA did not start - run it again" -ForegroundColor Red; exit 1 }
Write-Host "CARLA up" -ForegroundColor Green

Move-AppWindow "CarlaUE4" 0 0 860 900 | Out-Null

Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" scripts\reset_carla.py 2>&1 | Select-Object -Last 1

Write-Host "seed $Seed  (re-run this exact demo with -Seed $Seed)" -ForegroundColor DarkGray
$novaArgs = @("-u", "scripts\nova_drive.py", "--town", $Town, "--ground-truth",
              "--no-cameras", "--v-target", "$VTarget", "--seed", "$Seed",
              "--n-vehicles", "$Vehicles", "--n-walkers", "$Walkers")
if ($Record) {
    $novaArgs += @("--record", "demo.mp4")
    Write-Host "recording to demo.mp4" -ForegroundColor Yellow
}

Write-Host "starting NOVA..." -ForegroundColor Cyan
Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" `
    -ArgumentList $novaArgs -WorkingDirectory $PSScriptRoot

# The dashboard window is created by OpenCV once the first frame renders.
if (Move-AppWindow "NOVA" 870 0 1040 900) {
    Write-Host "both windows placed - CARLA left, NOVA right" -ForegroundColor Green
} else {
    Write-Host "NOVA window not found yet; drag it into place" -ForegroundColor Yellow
}
Write-Host "`npress Q in the NOVA window to stop" -ForegroundColor DarkGray
