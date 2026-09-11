# NOVA — autonomous path planning for Indian roads

SIH 2026, problem statement 26037 (MathWorks). Hackathon 10–11 Sept 2026.
Owner: Priyanshi. One demo: a car driving itself in CARLA through chaotic
traffic, from camera input, with a live dashboard.

## How to run

    # window 1 — the simulator
    .\start_carla.ps1 Town01

    # window 2 — always reset first
    .venv\Scripts\activate
    python scripts\reset_carla.py
    python scripts\nova_drive.py                 # full: vision + driving + HUD
    python scripts\nova_drive.py --ground-truth  # same, minus YOLO

## Two settings that are load-bearing on this machine — do not change back

TOWN01, NOT TOWN05. The large maps crash the CARLA render thread on this
hardware, in well under a minute, every time. Measured, 30 vehicles + 15
walkers, watched to 180 s:

    Town01  survived 180 s        Town03  crashed after   5 s
                                  Town05  crashed after  30 s

Not VRAM (Town05 idles at 1713 MiB of 6141). Not the renderer — identical
under D3D11 and D3D12. Not the Intel/NVIDIA present path — Town05 crashes
with `-RenderOffScreen` too, no swapchain at all. The crash is always
`RenderThread`, always the same call chain, and the fault address is
different every run (0x0, 0xffffffffffffffff, 0x3f800000 — that last one is
the bit pattern of the float 1.0f). A jump target read out of float data is
corrupted render memory. Root cause unknown and below our code; Town01 is a
workaround, not a fix.

THE SIM OVERLAY DEFAULTS TO OFF. Full-resolution cameras plus the debug
overlay crash the render thread even on Town01. See the note in
nova_drive.py — dropping either side clears it, and the HUD already draws
the same data.

A SLOW CLIENT LOOP KILLS THE CLIENT. Separate failure, separate symptom:
the SERVER stays up and the CLIENT dies, ~40 s in, always inside
`world.tick()`:

    Fatal Python error: Aborted
      File "scripts\nova_drive.py", line 208 in main

No traceback, no exit code, empty stderr - the abort is in C++, so nothing
Python-side reports it. Reproduce on demand with `--debug-sleep 0.12`.

It is LATENCY, not the detector and not the GPU. Measured:

    --ground-truth, fast loop                      survived 330 s
    --ground-truth + --debug-sleep 0.12            ABORTED at  40 s
    --weights (cuda)                               ABORTED at  35 s
    --weights --device cpu                         ABORTED at  35 s
    --weights --plan-every 4                       ABORTED at  35 s
    --weights --cam-width 416                      ABORTED at  40 s
    ultralytics + ByteTrack, 600 frames, no carla  clean

The last two lines rule out the obvious suspects: not YOLO (the detector
alone is fine), not VRAM (cpu aborts identically to cuda), not camera size
or detection rate (halving either changes nothing).

BUT DO NOT TRUST "slow loop" AS THE MECHANISM. It does not fit everything:
a --ground-truth run measured at 3.4 FPS (~290 ms/frame) SURVIVED 330 s,
which the theory says should have aborted at 40 s. The --debug-sleep result
is real and reproducible; the explanation for it is not settled.

WORSE, AND THE THING TO KNOW BEFORE TRUSTING ANY OF THIS: the results are
not reproducible across hours. The same code that ran 330 s clean at ~21:00
aborted at 25 s at ~23:30, and Town01 - which had survived eleven runs -
started crashing the SERVER again at 30 s. Reverting every change made no
difference. No WHEA errors and no display-driver resets in the event log, so
it is not obviously failing hardware, but nvidia-smi reports power.draw of
590 W against a 45 W limit, which is a nonsense value.

PRACTICAL CONSEQUENCE: treat a good run as something to CAPTURE, not
something to rely on. Record every clean run. Before a demo, reboot, close
everything else, and budget time for several attempts.

Consequence for the scenario suite: Town01 is a single-lane grid, so
`force_lane_change` cut-ins are not meaningful there. Build scenarios around
junctions and pedestrian crossings instead.

DIAGNOSING THE NEXT ONE. `start_carla.ps1` now passes `-log`, so the server
writes `%LOCALAPPDATA%\CarlaUE4\Saved\Logs\CarlaUE4.log`. Shipping builds
write NOTHING without that flag, which is why the first two days of crashes
left an empty log directory. Minidumps are in `..\Saved\Crashes\` — the
crashing thread is the one with `<IsCrashed>true</IsCrashed>`, and its name
tells you immediately whether to suspect our code (GameThread) or not
(RenderThread/RHIThread).

`reset_carla.py` before every run. A script that died without cleanup leaves
its actors behind AND leaves the server in synchronous mode with nobody
ticking it — CARLA then looks frozen and later scripts hang on connect.

## Architecture — the one rule

Each stage knows only the SHAPE of its input, never its source.

    carla_bridge.py / perception.py  -> List[Track]
    prediction.py                    -> List[Prediction]
    riskmap.py                       -> grid[t, y, x]
    planner.py                       -> Plan(steer, accel)
    pipeline.py                      -> runs prediction -> risk -> planner
    hud.py                           -> draws everything

All types are in `types.py`. That file is the contract; changing it means
changing everything. Swapping simulator ground truth for YOLO was a one-line
change precisely because both emit `List[Track]`.

`carla_bridge.py` is the ONLY file that may `import carla`. Keep it that way —
it is what makes "portable to real hardware" a true claim.

## Coordinate frame — cause of the worst class of bug here

NOVA:  +x forward, +y LEFT,  radians, ego at origin
CARLA: +x forward, +y RIGHT, degrees, world coordinates

Convert in `carla_bridge.py` only. Mirroring y REQUIRES negating yaw too.
Symptom of getting it wrong: the car steers into the thing it is avoiding,
which looks exactly like a planner bug and is not one.

## Constants that were measured, not guessed — do not "clean these up"

- `planner.pos_res = 1.5` — closed-set resolution. Sweep result:
  1.0 -> 105 ms / 2.29 m clearance; 1.5 -> 83 ms / 4.18 m; 2.0 -> 36 ms but
  the avoidance manoeuvre DISAPPEARS. 1.5 is the knee.
- `planner.w_goal = 1.0` — raising it makes the planner fast by making it
  stop searching. w_goal 4 gives 1.4 ms and drives in a straight line.
- `planner.w_speed = 0.35` — REQUIRED. Zero speed means zero future risk, so
  without a cost on being slow, "stop forever" is mathematically optimal.
- `planner.w_jerk = 2.5` — without it the steering oscillates every replan.
- `riskmap.offroad_cost = 5.0`, saturating. It was `(|y|-6)**2 * 3`, which hit
  577 while real traffic risk peaks near 1.5, so the planner ignored vehicles.
- `prediction` turn-rate decay (tau = 1.0) and the tyre clamp. Without them a
  1 rad/s estimate held for 3 s rotates an object 172 degrees and throws its
  predicted path 30 m off the road.

General rule: if a change makes the planner much faster, check that it has not
simply stopped avoiding things. Measure clearance, not just latency.

## Design decisions to preserve

- The risk grid is 3D `[t, y, x]`, not 2D. Flattening time makes the car
  refuse gaps that are empty by the time it arrives.
- There are NO behaviour modes. No `if market: slow_down()`. Different road
  types work because the risk field changes, not the planner. Do not add
  road-type branches — that claim is checkable by reading the file.
- `hud.LaneAnalyzer` fits curves to the SEGMENTED DRIVABLE AREA, not painted
  lane lines. That is the Indian-roads adaptation. Do not "fix" it to detect
  lane markings.
- `prediction.py` is heuristic by design, with `LSTMPredictor` as a drop-in
  slot behind the same interface. Never describe the LSTM as trained unless
  weights actually exist.

## Hardware limits

RTX 4050 laptop, 6141 MiB VRAM. CARLA and YOLO share it.
- Launch via `start_carla.ps1`; it caps UE4's texture pool so CUDA has room.
- YOLOv8n only. Cameras 640x480 — YOLO resizes to 640 anyway.
- Sim runs ~3.3x real time bare; planning every 2nd tick keeps it near 1.0x.

## Verified performance (measured on this machine)

    prediction   0.51 ms
    risk map     4.65 ms
    planner     58.45 ms
    total       63.61 ms  ->  15.7 Hz     (target was 10 Hz)

`python test_modules.py` reproduces this without CARLA. Run it after any
change to modules 2, 3 or 4 — it catches regressions in seconds.

## STATE AS OF 2026-09-06 — READ THIS FIRST

THE ONE BLOCKER: ATTACHING ANY CAMERA SENSOR WEDGES THE SERVER.

Reproduce in 60 s, with none of NOVA's code involved:
`scratchpad/sensor_test.py` and `camsize.py` spawn one car and one camera,
then tick. Result, every time:

    no camera at all          server runs 6000+ ticks, car drives 18-21 km/h
    camera 64x64              server wedges ~6 ticks after it attaches
    camera 160x120 .. 640x480 identical

The server does NOT crash. It stays alive, keeps listening on 2000/2001/2002,
and stops answering RPC - `world.tick()` then times out. So this is a HANG,
not a crash, and every "crash" reported earlier in this file's history was
actually this timeout being misread.

RULED OUT BY MEASUREMENT, do not re-test these:
  - VRAM            Town01 idles at 1713 MiB of 6141; 64x64 camera fails too
  - system RAM      8.4 GB free, bIsOOM=0 in all 16 crash dumps
  - disk            NVMe SSD, 181 GB free
  - CARLA install   re-extracted from the same zip into a fresh tree: identical
  - map             Town01 and Town10HD_Opt both, sizes 1.7 GB and 4.9 GB
  - renderer        D3D11 and D3D12 both
  - window/present  -RenderOffScreen (no swapchain at all) fails too
  - sync mode       asynchronous + wait_for_tick() fails at the same tick 6
  - GPU driver      two versions, plus a clean reinstall by the user
  - HAGS            already disabled (HwSchMode=1)
  - Windows GPU pref  both exes already forced to the RTX (GpuPreference=2)

NOT YET TRIED, in rough order of promise:
  1. The Intel iGPU instead of the RTX (Settings > Display > Graphics >
     per-app > Power saving). Removes the NVIDIA path entirely. Costs CUDA,
     so YOLO would fall back to CPU.
  2. A different machine. Nothing about NOVA's code is implicated.
  3. CARLA 0.9.15 - a different build might not hit whatever this is.

WHAT WORKS RIGHT NOW, TODAY, and is worth recording:

    python scripts\nova_drive.py --town Town01 --ground-truth --no-cameras

The car drives (measured 18-21 km/h), 33-56 FPS, planner 8-12 ms, risk map
and predicted futures live. What is missing is the camera panel, YOLO, and
the segmentation corridor fit - `--no-cameras` seeds a blank image and an
all-ones drivable mask, so those three HUD panels show placeholder data and
look broken. If demoing this way, blank those panels rather than show them.

DO IN PARALLEL, NEEDS NO CARLA: TASK 4, the IDD fine-tune on Colab. Free GPU
hours, and it is what makes "trained on Indian roads" true. `test_modules.py`
still proves modules 2-4 without CARLA in seconds.

## FOUR REAL BUGS WERE FIXED TODAY — they were never environment issues

1. STEERING SIGN WAS INVERTED. step_c_drive.to_control() converted the
   planner's radians to CARLA's [-1, 1] without negating. NOVA has +y LEFT
   and positive steer turns left; CARLA's steer is -1 for LEFT. The car
   therefore turned the OPPOSITE way to every decision the planner made and
   drove off the road on an empty street. This is exactly the trap the
   coordinate-frame section above warns about.

2. A MISSED CAMERA FRAME SKIPPED PLANNING. `if bgr is None: continue` jumped
   over pipe.step() and to_control(), so `control` kept its initial
   VehicleControl() with throttle 0.0. Whenever sensors stalled the car sat
   still for the whole run while the HUD drew a valid plan. THIS was the
   "car never moves" bug, not the planner and not tyre friction.

3. observe() MADE ~126 RPC ROUND TRIPS PER FRAME. get_transform() twice plus
   get_velocity() for every actor. Now one world.get_snapshot() plus one
   get_actors(): 2 calls. CARLA's client aborts under heavy RPC load with an
   uncaught msgpack exception (carla#3618) - SIGABRT, no Python traceback.

4. THE COLLISION COUNTER COUNTED CONTACT FRAMES. CARLA fires a collision
   event every frame contact persists, so one car resting against a wall
   reported 10117 "collisions". Now one impact = one count, 1 s refractory.

## The IDD dataset, as prepared on 2026-09-07

Downloaded and extracted to
`C:\Users\Admin\Downloads\idd-detection\IDD_Detection` (Pascal VOC, 24 GB,
41,857 annotated frames). `scripts/idd_to_yolo.py` turned it into
`C:\NOVA\idd_yolo` -> `idd_yolo.zip`, ready for Colab.

TWO CHOICES IN THAT CONVERTER, BOTH LOAD-BEARING:

FRONT CAMERAS ONLY (`--cameras`, defaults to frontFar, frontNear,
highquality_16k). IDD ships seven cameras; only 24,312 of the 41,857 frames
face forward. NOVA sees one forward camera. A car shot from a 90-degree side
camera shares almost no appearance with the same car ahead of a dashcam, and
yolov8n has no capacity to spend on a viewpoint it will never see. Unfiltered,
over half of a capped sample comes from side and rear views.

IDD'S OWN SPLIT, NOT A RANDOM ONE. Measured here: train.txt lists 438
sequences, val.txt lists 119, and they share ZERO. So the official split is
already leak-free, and using it makes val mAP comparable to published IDD
numbers. This is not a formality - IDD frames come from continuous drives, so
a per-frame random split puts near-identical neighbouring frames on both
sides and returns a flattering, meaningless number.

Result: 10,000 train / 2,000 val, 729 MB, 640 px long side.

    class             train boxes   val boxes
    motorcycle             36353        6731
    rider                  34783        6811
    person                 32376        4648
    car                    28520        6732
    autorickshaw           11920        2210
    truck                   9281        2000
    vehicle fallback        7046        1594
    bus                     5660        1380
    animal                  1510         317
    bicycle                  938         120

BICYCLE AND ANIMAL ARE THIN. Report their AP honestly rather than quoting
only the mean - 938 training boxes will not produce a strong bicycle
detector, and a judge who reads the per-class table will notice before you
do. Everything else has thousands of instances.

The ten class NAMES are the entire contract with `nova/perception.py`, which
does `NAME_TO_CLASS[names[cls].lower()]`. All ten map, and all ten resolve to
a `ClassProfile`; verified, not assumed. Rename one in the converter and NOVA
silently drops that class - the detection still fires, `.get()` returns None,
the box vanishes with no error.

## best.pt - trained 2026-09-08, and what its numbers actually say

TRAINED LOCALLY, NOT ON COLAB. `scripts/train_idd.py` on the 4050: 40 epochs,
10k images, 3.1 hours, batch 16 at 640. Colab was abandoned after two failed
transfers - a Drive FOLDER upload delivered 6,793 of 10,000 images with a
mismatched label count, which the notebook's own consistency check caught. If
Colab is ever needed again, upload the ZIP; one 735 MB file is far more
reliable than 12,000 small ones.

Measured on IDD's official val split, 2000 images / 32,542 instances:

    class              AP50   AP50-95      P       R
    autorickshaw      0.600     0.417  0.776   0.526
    bus               0.590     0.436  0.774   0.516
    car               0.582     0.386  0.765   0.516
    motorcycle        0.561     0.314  0.745   0.509
    truck             0.503     0.349  0.661   0.454
    rider             0.467     0.255  0.737   0.406
    person            0.375     0.193  0.719   0.305
    bicycle           0.236     0.138  0.674   0.200
    animal            0.141     0.069  0.420   0.120
    vehicle fallback  0.031     0.015  0.382   0.027
    OVERALL           0.409     0.257  0.665   0.358

Inference 1.9 ms/image - far under the 12 ms budget.

AUTORICKSHAW IS THE BEST CLASS IN THE MODEL, ahead of car. The class that
exists in no Western dataset is the one it detects most reliably. That is the
whole argument for fine-tuning, and it is a measurement, not a claim.

VEHICLE FALLBACK AT 0.031 IS NOT A TRAINING FAILURE. It is IDD's catch-all
for "vehicle-like but unnameable" - carts, tractors, oddities - so it has no
consistent appearance to learn. It alone pulls the mean down: EXCLUDING it,
mAP50 is 0.451. Quote both numbers and explain the difference; quoting only
0.451 is dishonest and quoting only 0.409 undersells the other nine.

BICYCLE AND ANIMAL CAME IN WEAK EXACTLY AS THE BOX COUNTS PREDICTED (938 and
1,510 training instances). Predicted before training, confirmed after.

RECALL, NOT mAP, IS THE SAFETY NUMBER. Overall recall is 0.358: a missed
object is the failure that causes a collision, a mislabelled one usually is
not. NOVA defaults to `--conf 0.35`; use `--conf 0.25` for demos, trading
precision for detections that would otherwise never reach the risk map. There
is latency budget to spare for it.

## NOVA 2.0 on real dashcam video - settings that were measured

Two clips in `C:\Users\Admin\Videos\`, both Delhi/Gurgaon, both with speed
and GPS burned into the frame. Verified end to end, whole clip, no crash:

    DASHCAM DAYTIME.mp4  1916x1038  58 s  toll plaza, dense mixed traffic
    DASHCAM NIGHT.mp4    1918x980   50 s  lit arterial road

    day    --fov 100 --horizon 0.72 --ego-speed 9.7  --every 3
    night  --fov 100 --horizon 0.50 --ego-speed 12.8 --every 5

--horizon IS THE ONE THAT MATTERS AND IT IS NEW. Range is
`fy * cam_height / (v - cy)`, so cy IS the horizon row. `from_fov()` puts it
at the image centre, which is right for CARLA and wrong for a dashcam aimed
slightly up. The daytime clip's horizon sits at 0.72 of frame height - mostly
sky above it. Measured effect on the same frames:

    horizon 0.50   nearest 2.4 m   median  4.1 m   (everything looks lethal)
    horizon 0.62   nearest 3.3 m   median  7.3 m
    horizon 0.72   nearest 4.8 m   median 21.4 m   (matches the scene)

Set it wrong the other way and you get NO detections at all: at 0.72 on the
night clip every box lands above the assumed horizon, range goes infinite,
and `max_range` discards the lot. Find it by looking at one frame and finding
the row where the road disappears.

EGO SPEED FOLLOWS THE PLANNER NOW, and this was worth 3x the frame rate.
Video has no odometry, so --ego-speed used to be held constant - meaning NOVA
closed forever on traffic that never receded and sat in EMERGENCY for 80-87%
of every clip. Emergency re-searches are expensive, so it was also SLOW.
Integrating `plan.accel` instead:

    fixed speed     EMERGENCY 87% of frames   plan 130 ms   5.3 FPS
    follows plan    NOMINAL most of the time  plan   8 ms  17.2 FPS

`--fixed-speed` restores the old behaviour. NOTE WHEN DEMOING: the HUD's
SPEED is then NOVA's COMMANDED speed, not the vehicle's - the dashcam's own
burned-in number is the real one, and the two will differ whenever NOVA
decides to brake. Say so before a judge spots it; "NOVA wants 5 km/h here and
the human was doing 62" is a good answer, being caught out is not.

PLAYBACK IS PACED TO THE SOURCE FRAME RATE unless you pass --fast. Without
pacing, playback speed is whatever the hardware manages - measured 0.38x
through dense night traffic and over 2x on an empty stretch, changing as the
scene changes. --every still needs setting per clip because pacing can only
slow things down: day runs 76 ms/frame, night 177 ms/frame.

TWO HUD BUGS FIXED HERE, both only visible on video:
  - The green corridor was drawn from the all-ones stub whenever the mask was
    absent, covering the very vehicles the demo is about. Now gated on
    `cameras_ok`, like the three inset panels already were.
  - NOTHING clipped the predicted futures or the planned path to the risk
    panel - only the route had a bounds check - so lines for distant agents
    drew straight across the camera view and the status bar. All three now go
    through `cv2.clipLine`, which trims to the rectangle instead of dropping
    the segment, so a partly-visible line still runs to the edge.

## The bus that rear-ended us every run - fixed 2026-09-09

SYMPTOM: identical spawn every run, a bus arrives from behind, COLLISIONS
reads 1 before the route is 3% done.

CAUSE 1, the spawn geometry. `EGO_CLEARANCE_M = 25.0` in step_a_traffic was a
RADIAL distance, so a bus 26 m directly behind in the ego's own lane passed
exactly the same test as one 26 m sideways on another street. Those are not
equivalent risks. AHEAD is safe - the ego closes slowly and the planner sees
it for seconds. BEHIND is not - the ego starts at 0 km/h, Traffic Manager
launches its vehicles at speed, the whole closing velocity belongs to the
other party, and NOVA has no actuator that influences a driver behind it.
Now: 25 m all round, PLUS a 70 m x 6 m clear corridor to the rear.

CAUSE 2, the counter blamed us for it. Being rear-ended while stationary is
not an at-fault collision, and counting it measures Traffic Manager's driving
rather than NOVA's. Rear impacts are now counted SEPARATELY and shown as
"0  (+1 rear)" - reported, never hidden. The exception is reversing: if
StuckRecovery is backing up when we hit something behind, that is ours.

CAUSE 3, the seed. run_demo.ps1 never passed --seed, so nova_drive's default
42 gave the same spawn and the same traffic every single run - one unlucky
spawn was unlucky forever. It now passes a random seed and prints it, so any
run can be reproduced with -Seed N.

A SECOND BUG SURFACED WHILE VERIFYING, and it was worse. On seed 7 the ego
moved for one frame then reported accel=+0.00 thr=0.00 str=-1.00 for 2,230
consecutive frames at route 0.0%. StuckRecovery never fired because it only
watched for "asking to go forward and not moving" - and NOVA was not asking.
A planner that settles on accel = 0 is exactly as stuck as one pushing
against a pole. `StuckRecovery.update()` now also fires when the car is
stationary for no reason, guarded by `blocked` (a red light, or a hazard
within 8 m) so it never reverses when stopping is correct.

MEASURED AFTER THE FIX, 45 s per seed, Town01, 10 vehicles + 3 walkers:

    seed  1   rolling by frame 20   route 29.2%   0 at-fault   0 rear
    seed  3   rolling by frame 20   route 44.0%   0 at-fault   0 rear
    seed 11   rolling by frame 20   route 19.1%   0 at-fault   0 rear
    seed 23   rolling by frame 20   route 39.7%   0 at-fault   0 rear
    seed  7   recovers after ~12 s of reversing, then 0 collisions over 100 s
              and completes a whole route before starting a second

Seed 7's slow start is a SPAWN problem, not a planner one: the route's first
goal lands beside and then behind the car (+4.7,+9.3 then -1.4,+5.7), which
it cannot drive to forwards. `pick_clear_spawn` checks there is road ahead but
not that the ROUTE leaves in a drivable direction. Worth fixing if there is
time; recovery makes it survivable meanwhile.

## Status - checked against disk 2026-09-09, hackathon is TOMORROW

DONE AND VERIFIED BY RUNNING IT:
  - modules 2, 3, 4 - `test_modules.py`, 69.4 ms/frame, 14.4 Hz, 6.22 m
    clearance
  - module 5, the NetworkX route - `nova/route.py` exists and drives; earlier
    versions of this file listed it as pending, which was wrong
  - CARLA closed-loop: `--ground-truth --no-cameras`, 0 off-route re-plans,
    0 at-fault collisions, 20-24 km/h
  - module 1 perception + module 6 HUD - both run every frame of NOVA 2.0
  - best.pt, trained locally, per-class numbers above
  - NOVA 2.0 on both dashcam clips, whole clip, no crash

NOT DONE. In the order they are worth doing with one day left:

  1. A RECORDED FALLBACK OF THE CARLA RUN. `runs/gt_demo.mp4` is 44 BYTES -
     recording has silently failed at least once. This file's own warning is
     that a clean CARLA run is something to CAPTURE, not rely on, and right
     now nothing is captured. Do this first, and PLAY THE FILE afterwards.
  2. The metrics table (TASK 2). Cheap, and it is what makes the report
     defensible rather than anecdotal.
  3. Scenario suite (TASK 3) - junctions and pedestrian crossings, NOT lane
     changes; Town01 is a single-lane grid.
  4. Module 7, the FastAPI + Supabase dashboard. Nothing exists. This is a
     large build for very little demo value when two live windows already
     show everything - drop it unless items 1-3 are finished and there is
     real time spare.

The video demo scales to a projector with `--scale 1.8` or `--fullscreen`.
The HUD renders at a fixed 1000x470 and every font size in hud.py is a fixed
pixel value, so the finished canvas is upscaled rather than rendered larger -
rendering larger would give a big canvas with the same tiny text.

## Conventions

- Comments explain WHY, not what. Several record bugs that were actually hit —
  keep those; they are the reason the constant is what it is.
- Every script cleans up in a `finally` block. Order matters: restore
  asynchronous mode BEFORE destroying actors, or the server crashes.
- Variables referenced in a `finally` are declared before the `try`, so a
  setup failure reports its real cause instead of UnboundLocalError.
