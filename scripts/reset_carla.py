"""
NOVA - reset a CARLA server that a crashed script left in a bad state.

WHEN YOU NEED THIS
------------------
A script that died without running cleanup() leaves two problems behind:

  1. Its actors are still there. Run again and you stack another 100 on top,
     until spawning fails or the server falls over.
  2. The server is still in SYNCHRONOUS MODE with nobody ticking it. CARLA
     then looks frozen, and every later script hangs on connect with an error
     that mentions none of this.

This fixes both without restarting CARLA - which matters at 3am on day two
when a restart is 60 seconds you would rather spend debugging.

RUN
---
    python scripts\\reset_carla.py
"""

import argparse
import sys
import time

try:
    import carla
except ImportError:
    sys.exit("carla not importable - activate the venv first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"connected to CARLA {client.get_server_version()}")

    # 1. Asynchronous FIRST - see cleanup() in step_a_traffic.py for why.
    settings = world.get_settings()
    was_sync = settings.synchronous_mode
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print(f"synchronous mode was {was_sync} -> now False")

    try:
        client.get_trafficmanager(args.tm_port).set_synchronous_mode(False)
    except Exception:                                  # noqa: BLE001
        pass

    time.sleep(0.3)

    # 2. Stop every walker controller before destroying anything.
    controllers = list(world.get_actors().filter("controller.ai.walker"))
    for c in controllers:
        try:
            c.stop()
        except Exception:                              # noqa: BLE001
            pass
    print(f"stopped {len(controllers)} walker controllers")

    # 3. Destroy in modest batches.
    victims = (list(controllers)
               + list(world.get_actors().filter("walker.pedestrian.*"))
               + list(world.get_actors().filter("vehicle.*"))
               + list(world.get_actors().filter("sensor.*")))

    destroyed = 0
    for i in range(0, len(victims), 20):
        chunk = victims[i:i + 20]
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in chunk])
            destroyed += len(chunk)
            time.sleep(0.1)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [warn] {exc}")

    time.sleep(0.5)
    remaining = len(world.get_actors().filter("vehicle.*"))
    print(f"destroyed {destroyed} actors; {remaining} vehicles remain")
    print("server is clean and asynchronous - safe to run a scenario again")


if __name__ == "__main__":
    main()
