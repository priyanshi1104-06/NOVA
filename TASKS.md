# NOVA — remaining work, in priority order

Each task below is written so it can be pasted into Claude Code as-is.
Read CLAUDE.md first — it holds the architecture rules and the constants
that must not be changed.

Do these IN ORDER. Each ends with something observable; if you cannot see it
working, do not move to the next one.

---

## TASK 0 — get the existing demo running (do this before writing anything)

Not a coding task. Blocking everything else.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install ultralytics
    python -c "import torch; print(torch.cuda.is_available())"   # want True

    python scripts\reset_carla.py
    python scripts\nova_drive.py --ground-truth    # dashboard, no YOLO
    python scripts\nova_drive.py                   # full, from camera

DONE WHEN: the car drives itself and the dashboard shows the green corridor,
detection boxes, risk panel and live metrics.

Likely tuning needed: the perspective trapezoid `src` in `hud.LaneAnalyzer`.
If the green corridor splays out or pinches in on a straight road, adjust
those four points. Nothing else is wrong.

---

## TASK 1 — Module 5: global route  ->  nova/route.py

WHY: right now `RouteFollower` in step_c_drive.py just asks the map for a
waypoint 25 m ahead. It cannot navigate from A to B, so "global route layer"
is not yet true and the demo cannot show a journey across town.

BUILD: a `GlobalRoute` class in `nova/route.py`.

    class GlobalRoute:
        def __init__(self, carla_map, start_location, end_location,
                     resolution=2.0)
        def goal(self, ego_location) -> carla.Location
        def remaining_distance(self, ego_location) -> float
        def is_complete(self, ego_location) -> bool
        def polyline(self) -> list          # for drawing on the HUD

Use CARLA's `agents.navigation.global_route_planner.GlobalRoutePlanner`,
which is NetworkX-based, so the "NetworkX graph search" claim is honest.
It lives in `C:\CARLA_0.9.16\PythonAPI\carla\agents\` — add that to sys.path.

REQUIREMENTS:
- Sticky goal: return the same target until the ego is within ~10 m of it,
  THEN advance. Recomputing every frame makes the goal jump between branches
  at junctions and the planner swings the wheel back and forth.
- Look ahead ~25 m along the route, not to the final destination — the local
  planner only searches 3 seconds.
- Pick start and end from `map.get_spawn_points()` far apart, seeded.

WIRE IN: replace `RouteFollower` in nova_drive.py. Draw `polyline()` on the
HUD's risk panel in a dim colour so the intended route is visible.

DONE WHEN: the car drives from one side of Town05 to the other and the
dashboard shows remaining distance counting down.

---

## TASK 2 — Module 7: metrics  ->  nova/metrics.py

WHY: "it drove and didn't crash" is not a result. A table is.

BUILD: a `RunMetrics` class recording per-frame and summarising per-run.

    class RunMetrics:
        def __init__(self, run_id, scenario, seed, perception_mode)
        def update(self, ego_state, tracks, plan, dt, collided: bool)
        def summary(self) -> dict
        def to_json(self, path)

MUST COMPUTE:
- collisions, and collisions per km
- distance travelled, duration, whether the goal was reached
- planning latency: median, p95, max
- minimum time-to-collision across the run
    TTC = distance_to_object / closing_speed, only for objects AHEAD with
    closing_speed > 0. Ignore objects moving away or it is meaningless.
- peak lateral acceleration = v * yaw_rate  (the comfort metric)
- mean speed, and time spent below 1 m/s (a proxy for freezing)

WIRE IN: call `update()` each planning cycle in nova_drive.py; write the JSON
in the finally block.

DONE WHEN: a run produces a JSON file with all fields populated and sane.

---

## TASK 3 — scenario suite  ->  scripts/run_scenarios.py

WHY: the success-rate metric needs repeated runs, not one lucky take. This is
also what produces the results table for the pitch.

BUILD: a runner that executes the same scenario across N seeds, headless-ish,
and aggregates.

    python scripts\run_scenarios.py --seeds 10 --scenario cutin

SCENARIOS (each a spawn configuration plus a scripted event):
- `cruise`   — normal dense traffic, no scripted event
- `cutin`    — a motorcycle forced to change lane into the ego's path
- `crossing` — a pedestrian walked into the road ahead at a set distance
- `braking`  — the lead vehicle brakes hard
- `dense`    — double the vehicle count, tight following

Force events with the Traffic Manager: `force_lane_change(vehicle, bool)` for
the cut-in, and `controller.go_to_location()` for the pedestrian.

MUST ALSO: run each scenario with NOVA and with CARLA's stock autopilot, and
report both. The autopilot has perfect information and hand-written rules —
if NOVA holds up against it from a camera, that is the strong claim.

OUTPUT: a markdown table plus a CSV.

    | scenario | perception | runs | collisions | success | p95 ms | min TTC |

DONE WHEN: `results/summary.md` exists with 10 seeds per scenario.

---

## TASK 4 — IDD fine-tune  ->  notebooks/train_idd.ipynb

WHY: this is what makes "trained on Indian roads" true, and adds the
autorickshaw and animal classes no Western dataset has.

BUILD: a Google Colab notebook (free T4 GPU).
1. Download IDD Detection from idd.insaan.iiit.ac.in (registration required)
2. Convert its Pascal-VOC XML annotations to YOLO txt format
3. Class list: car, bus, truck, motorcycle, bicycle, autorickshaw, person,
   rider, animal, vehicle fallback
4. Train `yolov8n.pt`, imgsz 640, ~50 epochs, batch 16
5. Export `best.pt`, report mAP50 and per-class AP

USE IT: `python scripts\nova_drive.py --weights best.pt`
`NAME_TO_CLASS` in perception.py already maps the IDD names, so nothing else
changes.

NOTE: run this on Colab IN PARALLEL with other work — it is hours of GPU time
and needs none of your machine.

---

## TASK 5 — recording and the results dashboard

RECORD (do this the first time you get a clean run, and after every change):

    python scripts\nova_drive.py --record results\run_01.mp4

A recording is insurance against a dead projector on stage. Not optional.

DASHBOARD (lowest priority — the HUD already shows live metrics):
- `nova/logging_supabase.py` — push each run summary to a Supabase table
- `scripts/serve_metrics.py` — small FastAPI app serving the results table

Only build this if TASKS 1–4 are done. It adds a line to the tech stack but
no marks that the HUD is not already earning.

---

## Order of value, if time runs short

1. TASK 0 — without it there is no demo
2. TASK 3 — numbers are what separate a demo from a result
3. TASK 1 — makes the "global route" claim true
4. TASK 5 recording — cheap, and saves you if hardware fails
5. TASK 4 — makes the "trained on IDD" claim true
6. TASK 2 metrics — partly covered already by the HUD
7. TASK 5 dashboard — least value per hour spent
