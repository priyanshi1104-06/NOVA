# NOVA - launch the CARLA server with this project's settings.
#
#   .\start_carla.ps1              -> Town01, low quality (day-to-day)
#   .\start_carla.ps1 Town07       -> village roads
#   .\start_carla.ps1 Town10HD     -> dense urban
#   .\start_carla.ps1 Town01 Epic  -> demo/recording quality
#
# Always launch the SERVER from here and your scripts from a second window.

param(
    [string]$Map = "Town01",
    # Force the DirectX 11 renderer.
    #
    # WHY: on this machine CARLA's D3D12 path dies with
    #     LowLevelFatalError: CommandList->Close() failed
    #     D3D12CommandList.cpp:144, error E_INVALIDARG
    # It killed the server during load_world() and during scenario spawning,
    # which for hours looked like a VRAM problem or a bug in our own code. It
    # is neither - UE4 4.26's DX12 renderer is fragile on older NVIDIA drivers
    # (this machine is on 561.00). DX11 is the mature path and costs nothing
    # here: CARLA uses no ray tracing and we run at Low quality anyway.
    #
    # Pass -UseDX12 only after updating the GPU driver, to test.
    [switch]$UseDX12,
    [string]$Quality = "Low",
    [int]$PoolSizeMB = 3072,
    # Render with no spectator window (-RenderOffScreen). OPT-IN, OFF BY
    # DEFAULT.
    #
    # The idea is sound: the spectator window is a FOURTH full scene render
    # every tick on top of the three camera sensors, plus a swapchain present,
    # and nothing in the demo comes from it - the recording writes the HUD
    # canvas, not that window.
    #
    # But it is NOT the default, because making it the default produced a
    # server that never answered on port 2000, and a launch that boots is
    # worth more than a launch that is fast. Try it with -Offscreen when you
    # have time to check; do not make it the default without a run that both
    # boots and reports a better [budget] line.
    [switch]$Offscreen
)

$exe = "C:\CARLA_0.9.16\CarlaUE4.exe"
if (-not (Test-Path $exe)) { throw "CARLA not found at $exe" }

# THE MAP ARGUMENT ON THE COMMAND LINE DOES NOTHING. Measured, three ways:
#
#   /Game/Carla/Maps/Town05  -> loaded Carla/Maps/Town10HD_Opt
#   Town05                   -> loaded Carla/Maps/Town10HD_Opt
#   -map=Town05              -> loaded Carla/Maps/Town10HD_Opt
#
# CarlaUE4.exe is a stub that re-launches CarlaUE4-Win64-Shipping.exe with the
# project name as argv[1], so the map lands in argv[2] where UE4 never looks
# for a URL. The server always boots GameDefaultMap from
# CarlaUE4\Config\DefaultEngine.ini, which ships as Town10HD_Opt.
#
# This mattered more than it looks. Scenario.__init__ in step_a_traffic.py
# skips load_world() when the requested town is already loaded - a guard put
# there deliberately, because load_world() rebuilds every render resource in
# the process. That guard NEVER FIRED: the loaded map was always
# Town10HD_Opt, never the requested town, so every single run paid for a full
# map transition it was written to avoid.
#
# So load the map here, once per server, over the RPC port instead.

# r.Streaming.PoolSize caps UE4's texture cache. Left alone, UE4 expands to
# fill all 6 GB of VRAM and leaves nothing for YOLO's CUDA context. This is
# NOT a quality setting - worst case you get mild texture pop-in.
#
# RAISED 1024 -> 3072. A pool that is too SMALL is not free: UE4 evicts and
# re-streams textures every frame trying to fit the visible set into it, and
# the render thread stalls waiting. On a 6 GB card 1 GB was miserly, and the
# symptom - a low frame rate that the client-side timings do not account for
# - matches. 3 GB still leaves ~3 GB for YOLO's CUDA context, which needs
# well under 1 GB for yolov8n.
# If VRAM ever runs short with the full YOLO path, lower this first.
$poolArg = "-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=$PoolSizeMB"

Write-Host "starting CARLA  map=$Map  quality=$Quality  pool=${PoolSizeMB}MB  renderer=$(if ($UseDX12) {'DX12'} else {'DX11'})  display=$(if ($Offscreen) {'offscreen'} else {'window'})" -ForegroundColor Cyan
Write-Host "(close this window or Ctrl+C to stop the server)`n"

$rhiArg = if ($UseDX12) { "-dx12" } else { "-dx11" }

$displayArgs = if ($Offscreen) {
    @("-RenderOffScreen")
} else {
    @("-windowed", "-ResX=800", "-ResY=600")
}

# Build the whole argument list FIRST, then pass it as one variable.
# Do not inline `@(...) + $displayArgs + @(...)` into the -ArgumentList
# parameter: PowerShell binds the first array to the parameter and then treats
# the following `+` as a positional argument, and Start-Process fails with
# "A positional parameter cannot be found that accepts argument '+'".
$carlaArgs = @("-quality-level=$Quality")
$carlaArgs += $displayArgs
$carlaArgs += $poolArg
$carlaArgs += $rhiArg
# -log makes UE4 write Saved\Logs\CarlaUE4.log. Shipping builds write
# NOTHING without it, which is why two days of crashes left an empty log
# directory and the only evidence was the minidumps in Saved\Crashes.
$carlaArgs += "-log"

Start-Process -FilePath $exe -ArgumentList $carlaArgs

# Wait for the RPC port, then load the map the caller actually asked for.
Write-Host "waiting for the server to accept connections..." -ForegroundColor DarkGray
$loaded = $null
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 2
    $loaded = & "$PSScriptRoot\.venv\Scripts\python.exe" -c @"
import sys
import carla
try:
    c = carla.Client('127.0.0.1', 2000); c.set_timeout(5.0)
    if '$Map' in c.get_world().get_map().name:
        print(c.get_world().get_map().name)
    else:
        print(c.load_world('$Map').get_map().name)
except Exception:
    sys.exit(1)
"@ 2>$null
    if ($LASTEXITCODE -eq 0 -and $loaded) { break }
}

if (-not $loaded) {
    Write-Host "server never answered on port 2000 - check the CARLA window" -ForegroundColor Red
} else {
    Write-Host "map loaded: $loaded" -ForegroundColor Green
}
