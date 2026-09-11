# NOVA 2.0 - NIGHT demo.  Run it with:   .\night.ps1
#
# Lit arterial road, Gurgaon. Every value here was measured - see CLAUDE.md.
# Note how much differs from day.ps1; these are NOT interchangeable:
#
#   --horizon 0.50  this clip's horizon really is near the middle. Using the
#                   daytime 0.72 here gives ZERO detections - every box lands
#                   above the assumed horizon, range goes infinite, and
#                   max_range discards the lot.
#   --every 5       the night clip costs 177 ms/frame against the day's 76,
#                   so it needs a bigger frame step to play at real time.
#
# Q quits. --loop restarts at the end so it cannot run out mid-sentence.
param(
    [double]$Scale = 1.8,      # 1.8 roughly fills a 1080p screen
    [switch]$Fullscreen,
    [string]$Video = "C:\Users\Admin\Videos\DASHCAM NIGHT.mp4"
)

$nova = @(
    "scripts\nova_video.py", "--video", $Video,
    "--weights", "best.pt", "--conf", "0.25",
    "--fov", "100", "--horizon", "0.5",
    "--ego-speed", "12.8", "--every", "5", "--loop"
)
if ($Fullscreen) { $nova += "--fullscreen" } else { $nova += @("--scale", "$Scale") }

Write-Host "NOVA 2.0 - night.  Q in the window to stop." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" @nova
