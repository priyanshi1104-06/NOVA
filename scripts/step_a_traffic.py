"""
NOVA - STEP A: CARLA up, synchronous, with Indian-style chaotic traffic.

WHAT THIS PROVES
----------------
Run this and you should see a car driving itself through dense, badly-behaved
traffic, with the camera following it. It is still CARLA's stock autopilot
driving - NOVA is not involved yet. That is deliberate: this step verifies the
simulator, the client connection, synchronous mode and the traffic
configuration in isolation, BEFORE we add our own code to the picture.

Debug one thing at a time. If you wire NOVA in now and the car crashes, you
will not know whether it was your planner or your spawn logic.

HOW TO RUN
----------
Terminal 1:
    cd C:\\CARLA_0.9.16
    .\\CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

Terminal 2:
    cd C:\\NOVA
    .venv\\Scripts\\activate
    pip install carla==0.9.16
    python scripts\\step_a_traffic.py

Press Ctrl+C to stop. Cleanup is automatic and it MATTERS - see cleanup() below.

THE TWO IDEAS IN THIS FILE WORTH UNDERSTANDING
----------------------------------------------
1. SYNCHRONOUS MODE. By default CARLA runs free and your script sees whatever
   frame happens to be current - so a slow planning cycle means the car has
   already driven several metres past where it decided to turn. In synchronous
   mode the server does NOTHING until your script calls world.tick(). A slow
   frame becomes slow motion instead of a crash, and identical seeds produce
   identical runs, which is the only reason your metrics table means anything.

2. "INDIAN" IS BEHAVIOUR, NOT MESHES. CARLA has no auto-rickshaws. What makes
   Indian traffic hard is not vehicle shape - it is that agents do not hold
   lanes, do not signal, tailgate, and cross at unmarked points. Every one of
   those is a Traffic Manager parameter, set in configure_indian_traffic().
   That function IS your "Indian road conditions" claim; be ready to walk a
   judge through it line by line.
"""

import argparse
import random
import sys
import time

try:
    import carla
except ImportError:
    sys.exit(
        "Could not import carla.\n"
        "  Activate the venv, then:  pip install carla==0.9.16\n"
        "  Check with:               python -c \"import carla; print(carla.__file__)\""
    )


# ----------------------------------------------------------------------
# THE INDIAN TRAFFIC RECIPE - this is the important part of the file
# ----------------------------------------------------------------------
# Each entry: (TrafficManager method name, value, why it matters).
# Applied per-vehicle. Wrapped in getattr() so that if a method is missing in
# your CARLA build the script warns and continues instead of dying - useful
# because TM gained several of these across versions.
INDIAN_TM_PROFILE = [
    # No lane discipline. Biggest visual difference between European and
    # Indian traffic: vehicles sit wherever there is room.
    ("keep_right_rule_percentage",        0.0,   "ignore keep-left/right"),
    # Unsignalled lane changes in BOTH directions - this generates the cut-ins
    # the predictor exists to handle.
    # TUNED DOWN from 60. At 60, combined with a 0.5 m following distance,
    # traffic gridlocked at junctions and the whole town sat at 0 km/h. Chaos
    # that stops moving is not chaos, it is a car park.
    ("random_left_lanechange_percentage", 40.0,  "sudden cut-ins, left"),
    ("random_right_lanechange_percentage", 40.0, "sudden cut-ins, right"),
    # Tailgating. 0.5 m was deadlock-inducing; 1.2 m still looks aggressive
    # (a normal TM default is 3-5 m) but leaves room to resolve conflicts.
    ("distance_to_leading_vehicle",       1.2,   "tailgating"),
    # Signals treated as advisory.
    ("ignore_lights_percentage",          20.0,  "jumping lights"),
    ("ignore_signs_percentage",           40.0,  "rolling stop signs"),
    # Assertive near pedestrians. NOW ZERO - see the crash note in
    # spawn_walkers(). Background traffic running pedestrians over was
    # crashing the CARLA server. Our own ego still has to avoid them, which
    # is the thing we are actually demonstrating; background cars mowing
    # people down demonstrates nothing and killed the server every ~60 s.
    ("ignore_walkers_percentage",         0.0,   "yield to VRUs"),
]

# Fleet composition as FRACTIONS of --n-vehicles, not absolute counts.
#
# THE BUG THIS FIXES: the original used absolute counts summing to 47. Ask for
# 30 vehicles and the list was truncated to its first 30 entries - every one a
# two-wheeler - so ZERO cars spawned. The class census in step_b_observe.py is
# what caught it, which is exactly why that census line is in there.
#
# ~50% two-wheelers and bicycles is what makes footage read as Indian rather
# than European, and it is the hardest mix for a planner: small, fast,
# laterally agile agents that fit through gaps a car cannot.
TWO_WHEELER_BPS = [
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.yamaha.yzf",
    "vehicle.vespa.zx125",
]
BICYCLE_BPS = [
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
]
FRAC_TWO_WHEELER = 0.40     # motorcycles / scooters
FRAC_BICYCLE = 0.12         # pedal cycles
# remainder fills with 4-wheeled vehicles


def configure_indian_traffic(tm, vehicle, rng):
    """Apply the chaos profile to one vehicle."""
    for method_name, value, _why in INDIAN_TM_PROFILE:
        fn = getattr(tm, method_name, None)
        if fn is None:
            print(f"  [warn] TrafficManager has no {method_name}() - skipped")
            continue
        try:
            fn(vehicle, value)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] {method_name} failed: {exc}")

    # Big per-vehicle speed spread. Uniform speeds look like a motorway in
    # Germany; wildly mixed speeds look like a road in Ahmedabad, and they
    # force the planner to deal with overtaking and closing gaps.
    # Negative = faster than the posted limit.
    tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-40.0, 30.0))


class Scenario:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.vehicles = []
        self.walkers = []
        self.controllers = []
        self.ego = None
        self.original_settings = None

        self.client = carla.Client(args.host, args.port)
        # 20 s: loading a map is slow on a mechanical drive and the default
        # 5 s timeout throws a confusing RuntimeError mid-load.
        self.client.set_timeout(20.0)
        print(f"connected to CARLA {self.client.get_server_version()}")

        # Only reload the map if we are not already on it.
        #
        # load_world() tears the entire world down and rebuilds it. Doing that
        # on every script run is slow and is a known way to destabilise the
        # server - especially if a previous run left it in synchronous mode.
        # Reusing the loaded map makes runs faster AND removes a crash vector.
        current = self.client.get_world().get_map().name
        if args.town not in current:
            print(f"loading {args.town} (was {current})")
            self.world = self.client.load_world(args.town)
        else:
            print(f"reusing loaded map {current}")
            self.world = self.client.get_world()
        self.map = self.world.get_map()

    # ------------------------------------------------------------------
    def setup_sync(self):
        """Synchronous mode + fixed timestep. See module docstring, idea 1."""
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # Fixed timestep. Must be <= 0.1 or CARLA's own physics gets unstable,
        # so 0.1 (10 Hz) is the LARGEST legal step and 0.05 (20 Hz) was the
        # original.
        #
        # WHY THIS IS NOW A KNOB. In synchronous mode the world advances one
        # step per client world.tick(), so the wall-clock speed of the sim is
        # (client FPS x fixed_delta_seconds). At 3.4 client FPS a 0.05 s step
        # runs the world at 0.17x real time and everything looks frozen -
        # which is exactly what "the car is not moving" turned out to be.
        # Doubling the step to 0.1 doubles the world's wall-clock speed AND
        # halves the number of camera renders the server has to do, because
        # the sensors fire once per tick.
        dt = float(getattr(self.args, "sim_dt", 0.05))
        settings.fixed_delta_seconds = min(max(dt, 0.01), 0.1)
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(self.args.tm_port)
        # The TM must ALSO be synchronous or it will step at its own rate and
        # the traffic will visibly stutter relative to the world.
        self.tm.set_synchronous_mode(True)
        # Reproducibility. This is what lets a judge pick a seed and lets you
        # run the same scenario 10 times for the success-rate metric.
        self.tm.set_random_device_seed(self.args.seed)
        print(f"synchronous mode on, "
              f"{1.0 / settings.fixed_delta_seconds:.0f} Hz "
              f"(dt={settings.fixed_delta_seconds:.3f}s), seed={self.args.seed}")

    # ------------------------------------------------------------------
    def spawn_ego(self):
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "ego")
        # SPAWN SOMEWHERE THE CAR CAN ACTUALLY DRIVE AWAY FROM.
        #
        # This used to take the first spawn point that accepted an actor,
        # which only proves the point was empty at that instant - not that
        # there is any road ahead. Starting nose-first in a junction meant the
        # first route goal was already across a kerb and the ego drove into it
        # within a couple of seconds, which read as a planner bug.
        #
        # Now: not in a junction, road continues 12 m ahead, nobody parked in
        # that space. Falls back to the old behaviour if nothing qualifies.
        from nova.route import pick_clear_spawn

        preferred = pick_clear_spawn(self.world, self.map, self.rng)
        spawn_points = self.map.get_spawn_points()
        self.rng.shuffle(spawn_points)
        if preferred is not None:
            spawn_points.insert(0, preferred)
        for sp in spawn_points:
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is not None:
                self.ego = actor
                break
        if self.ego is None:
            raise RuntimeError("could not spawn ego - all spawn points blocked")
        # Stock autopilot FOR NOW. In Step C this line is replaced by NOVA.
        self.ego.set_autopilot(True, self.args.tm_port)
        configure_indian_traffic(self.tm, self.ego, self.rng)

        # TICK ONCE before anyone reads the ego's transform.
        #
        # In synchronous mode a freshly spawned actor reports location
        # (0, 0, 0) until the world advances. Without this tick,
        # spawn_traffic() sorted spawn points by distance from the MAP ORIGIN
        # instead of from the car - so traffic clustered wherever the origin
        # happens to be and the ego drove alone. The symptom was
        # "furthest 0 m from ego" and step_b printing "0 tracks in range".
        self.world.tick()

        print(f"ego spawned: {self.ego.type_id} id={self.ego.id} "
              f"at {self.ego.get_transform().location}")

    def spawn_traffic(self):
        """Spawn the fleet AROUND THE EGO, in the right proportions.

        Two things here matter more than they look.

        1. PROPORTIONAL MIX. See the FRAC_* constants - absolute counts break
           silently when you change --n-vehicles.

        2. SPAWN NEAR THE EGO. CARLA's spawn points cover the entire town, so
           scattering 30 vehicles across Town01 leaves the ego driving alone
           and step_b prints "0 tracks in range". We sort spawn points by
           distance from the ego and use the nearest ones, which puts the
           traffic where the planner will actually meet it. For the demo this
           is not cheating - it is choosing where the scenario takes place.
        """
        library = self.world.get_blueprint_library()
        n = self.args.n_vehicles

        def pick(ids):
            out = []
            for bp_id in ids:
                found = library.filter(bp_id)
                if found:
                    out.append(found[0])
                else:
                    print(f"  [warn] blueprint {bp_id} not in this build")
            return out

        two_bps = pick(TWO_WHEELER_BPS)
        cyc_bps = pick(BICYCLE_BPS)
        car_bps = [b for b in library.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) == 4]

        n_two = int(n * FRAC_TWO_WHEELER) if two_bps else 0
        n_cyc = int(n * FRAC_BICYCLE) if cyc_bps else 0
        n_car = n - n_two - n_cyc

        wanted = ([self.rng.choice(two_bps) for _ in range(n_two)] +
                  [self.rng.choice(cyc_bps) for _ in range(n_cyc)] +
                  [self.rng.choice(car_bps) for _ in range(n_car)])
        self.rng.shuffle(wanted)

        # Nearest spawn points first, but NOT ON TOP OF THE EGO.
        #
        # THE BUG THIS FIXES: this used to be sorted-by-distance then [1:],
        # which drops exactly ONE spawn point - the ego's own. Every other
        # vehicle therefore spawned at the 2nd, 3rd, 4th... nearest spawn
        # point, and on a small map those are metres away, often in the ego's
        # own lane directly ahead. The dashboard showed MIN GAP 3.5 m on the
        # very first frame and COLLISIONS 1 within seconds, and after that the
        # ego sat at 0 km/h with PLANNER: NOMINAL - because a car parked on
        # its bumper makes every forward motion expensive, so braking really
        # is the cheapest plan. The car was not broken; it was boxed in at
        # t=0.
        #
        # EGO_CLEARANCE_M is the demo's most important spawn parameter. Too
        # small and the ego starts trapped; too large and the traffic is over
        # the horizon and the first thirty seconds are empty road.
        # AHEAD AND BEHIND ARE NOT THE SAME RISK, and treating them the same
        # is what put a bus into the ego's boot on every single run.
        #
        # This was one radial number. A vehicle 26 m AHEAD is fine: the ego
        # closes on it slowly and the planner sees it coming for seconds. A
        # vehicle 26 m BEHIND IN THE SAME LANE is not: the ego starts from
        # 0 km/h while Traffic Manager launches its cars at speed, so the
        # whole closing velocity is the other party's, and NOVA has no
        # actuator that influences a driver behind it. It got rear-ended
        # before the route was 3% done, and the counter blamed NOVA for it.
        #
        # So: keep 25 m all round, and additionally clear a long corridor
        # behind. BEHIND_LATERAL_M is deliberately narrow - a bus one street
        # over is not a threat, and widening it strips out most of the map's
        # usable spawn points.
        EGO_CLEARANCE_M = 25.0
        BEHIND_CLEARANCE_M = 70.0
        BEHIND_LATERAL_M = 6.0

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()

        def usable(sp):
            dx = sp.location.x - ego_loc.x
            dy = sp.location.y - ego_loc.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= EGO_CLEARANCE_M:
                return False
            along = dx * fwd.x + dy * fwd.y          # +ve = in front of us
            lateral = abs(-dx * fwd.y + dy * fwd.x)  # off our centre line
            if along < 0 and lateral < BEHIND_LATERAL_M \
                    and dist < BEHIND_CLEARANCE_M:
                return False                          # the bus case
            return True

        spawn_points = [sp for sp in self.map.get_spawn_points() if usable(sp)]
        spawn_points.sort(key=lambda sp: sp.location.distance(ego_loc))
        if len(spawn_points) < len(wanted):
            print(f"  [warn] only {len(spawn_points)} spawn points outside "
                  f"{EGO_CLEARANCE_M:.0f} m of the ego; fleet will be smaller")

        for bp, sp in zip(wanted, spawn_points):
            if bp.has_attribute("color"):
                bp.set_attribute(
                    "color", self.rng.choice(
                        bp.get_attribute("color").recommended_values))
            v = self.world.try_spawn_actor(bp, sp)
            if v is None:
                continue                      # spawn point occupied, skip
            v.set_autopilot(True, self.args.tm_port)
            configure_indian_traffic(self.tm, v, self.rng)
            self.vehicles.append(v)

        two_wheel = sum(1 for v in self.vehicles
                        if int(v.attributes.get("number_of_wheels", 4)) == 2)
        if self.vehicles:
            far = max(v.get_transform().location.distance(ego_loc)
                      for v in self.vehicles)
            print(f"spawned {len(self.vehicles)} vehicles "
                  f"({two_wheel} two-wheelers, "
                  f"{100*two_wheel//len(self.vehicles)}%), "
                  f"furthest {far:.0f} m from ego")

    def spawn_walkers(self):
        """Pedestrians with AI controllers.

        NOTE the ordering below - it is fiddly and the usual source of
        'walkers spawn but never move'. You must spawn the walker, tick the
        world so it exists server-side, THEN attach and start the controller.
        """
        library = self.world.get_blueprint_library()
        walker_bps = library.filter("walker.pedestrian.*")

        # Same reasoning as spawn_traffic: oversample random navigation
        # points and keep the ones nearest the ego, so pedestrians appear
        # where the car will actually encounter them.
        ego_loc = self.ego.get_transform().location
        candidates = []
        for _ in range(self.args.n_walkers * 8):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                candidates.append(loc)
        candidates.sort(key=lambda l: l.distance(ego_loc))
        batch_locations = candidates[: self.args.n_walkers]

        for loc in batch_locations:
            bp = self.rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                # ============================================================
                # KEEP THIS, BUT THE CRASH STORY BELOW IS WRONG.
                #
                # 2026-09-05: all 14 minidumps in Saved\Crashes were read.
                # Not one crashes on the GameThread, and none shows a pure
                # virtual call. Every one is RenderThread/RHIThread with the
                # same call chain, under both D3D11 and D3D12, and it
                # reproduces with ZERO walkers and zero vehicles. So the
                # account below is not what was happening. Invincible walkers
                # are still the right call - a walker killed mid-run is a
                # dead pawn under a live controller either way - but do not
                # cite this as the fix for the server crash. That was the
                # large maps plus render-thread load; see CLAUDE.md.
                #
                # The original note, kept because the reasoning is sound even
                # though the evidence was misattributed:
                #
                # With is_invincible="false" (which is what CARLA's own
                # generate_traffic.py ships), a walker that gets hit by a
                # vehicle is KILLED server-side. Its attached
                # controller.ai.walker keeps ticking and dereferences the
                # dead pawn, which surfaces as a GameThread null-vtable call:
                #     EXCEPTION_ACCESS_VIOLATION 0x0000000000000000
                #     "Pure virtual function being called"
                # Identical callstack hash under BOTH D3D11 and D3D12 and
                # under two different NVIDIA drivers - which is how we knew
                # it was never a graphics problem. Always 30-100 s in, i.e.
                # however long it took the first car to hit someone.
                #
                # Invincible walkers still collide, still get pushed, still
                # register on our collision sensor. Nothing about the demo
                # is weaker; the server just stops dying.
                # ============================================================
                bp.set_attribute("is_invincible", "true")
            w = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if w is not None:
                self.walkers.append(w)

        self.world.tick()          # walkers must exist before controllers attach

        ctrl_bp = library.find("controller.ai.walker")
        for w in self.walkers:
            c = self.world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            if c is None:
                continue
            self.controllers.append(c)

        self.world.tick()

        for c in self.controllers:
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            # Mixed walking speeds; a few move fast enough to be a real hazard.
            c.set_max_speed(self.rng.uniform(0.8, 2.2))

        print(f"spawned {len(self.walkers)} pedestrians "
              f"({len(self.controllers)} with controllers)")

    # ------------------------------------------------------------------
    def follow_with_spectator(self):
        """Park the spectator camera behind the ego so you can see it drive."""
        tf = self.ego.get_transform()
        fwd = tf.get_forward_vector()
        cam = carla.Transform(
            tf.location + carla.Location(x=-7 * fwd.x, y=-7 * fwd.y, z=4.0),
            carla.Rotation(pitch=-15, yaw=tf.rotation.yaw),
        )
        self.world.get_spectator().set_transform(cam)

    def run(self):
        print("\nrunning - Ctrl+C to stop\n")
        frame = 0
        t0 = time.perf_counter()
        try:
            while True:
                self.world.tick()
                self.follow_with_spectator()
                frame += 1
                if frame % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    v = self.ego.get_velocity()
                    speed = 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5
                    print(f"  frame {frame:5d}   sim {frame*0.05:6.1f}s   "
                          f"real {elapsed:6.1f}s   "
                          f"ego {speed:5.1f} km/h")
        except KeyboardInterrupt:
            print("\nstopping")

    # ------------------------------------------------------------------
    def cleanup(self):
        """Shut down safely. ORDER MATTERS - this crashed the server twice.

        The rule: RESTORE ASYNCHRONOUS MODE BEFORE DESTROYING ANYTHING.

        In synchronous mode the server does nothing until the client calls
        world.tick(). Ctrl+C stops the ticking, so a DestroyActor batch issued
        afterwards is handed to a server that will never process it - and
        tearing down ~100 actors in that state takes CARLA down with it.
        Going asynchronous first lets the server process the destruction on
        its own clock. CARLA's own generate_traffic.py does it in this order
        for the same reason.

        Second rule: walker controllers must be STOPPED before their walkers
        are destroyed, or the AI controller ticks against a dead actor.

        Every step is individually wrapped - a failure in one must not skip
        the rest, or you leave the server in a half-configured state that
        makes the NEXT run fail with an unrelated-looking error.
        """
        print("cleaning up...")

        # 1. ASYNC FIRST. Everything else depends on this.
        try:
            if self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
            else:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] could not restore async mode: {exc}")

        try:
            self.tm.set_synchronous_mode(False)
        except Exception:                              # noqa: BLE001
            pass

        # Give the server a moment to actually leave sync mode before we
        # start throwing destruction commands at it.
        time.sleep(0.3)

        # 2. Stop walker AI before destroying the walkers it drives.
        for c in self.controllers:
            try:
                c.stop()
            except Exception:                          # noqa: BLE001
                pass

        # 3. Destroy in small batches. One giant batch is more likely to
        #    trip the server than several modest ones.
        actors = self.controllers + self.walkers + self.vehicles
        if self.ego is not None:
            actors.append(self.ego)

        destroyed = 0
        for i in range(0, len(actors), 20):
            chunk = actors[i:i + 20]
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(a) for a in chunk])
                destroyed += len(chunk)
                time.sleep(0.1)
            except Exception as exc:                   # noqa: BLE001
                print(f"  [warn] batch destroy failed: {exc}")

        time.sleep(0.5)
        print(f"destroyed {destroyed} actors, async mode restored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    # Town01 is the lightest map. Move to Town07 (village) / Town10HD (dense
    # urban) / Town04 (highway) once it runs - it is a one-word change.
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--n-vehicles", type=int, default=60)
    ap.add_argument("--n-walkers", type=int, default=40)
    # Let a judge choose this number. A scripted demo breaks; this one does not.
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scn = Scenario(args)
    try:
        scn.setup_sync()
        scn.spawn_ego()
        scn.spawn_traffic()
        scn.spawn_walkers()
        scn.run()
    finally:
        scn.cleanup()


if __name__ == "__main__":
    main()
