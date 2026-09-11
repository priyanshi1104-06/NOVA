# NOVA 2.0 - DAYTIME demo.  Run it with:   .\day.ps1
#
# Delhi toll plaza, dense mixed traffic. Every value here was measured, not
# guessed - see CLAUDE.md. The two that matter:
#
#   --horizon 0.72  this clip's horizon sits at 72% of frame height, not the
#                   middle. Range is fy*h/(v-cy) and cy IS the horizon, so
#                   assuming centre makes everything look 4x too close.
#   --conf 0.25     below NOVA's 0.35 default on purpose. Overall recall is
#                   0.358, and a MISSED object is what causes a collision;
#                   a mislabelled one usually is not.
#
# Q quits. --loop restarts at the end so it cannot run out mid-sentence.
param(
    [double]$Scale = 1.8,      # 1.8 roughly fills a 1080p screen
    [switch]$Fullscreen,
    [string]$Video = "C:\Users\Admin\Videos\DASHCAM DAYTIME.mp4"
)

$nova = @(
    "scripts\nova_video.py", "--video", $Video,
    "--weights", "best.pt", "--conf", "0.25",
    "--fov", "100", "--horizon", "0.72",
    "--ego-speed", "9.7", "--every", "2", "--loop"
)
if ($Fullscreen) { $nova += "--fullscreen" } else { $nova += @("--scale", "$Scale") }

Write-Host "NOVA 2.0 - daytime.  Q in the window to stop." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" @nova
