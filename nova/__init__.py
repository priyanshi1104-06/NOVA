"""NOVA - autonomous path planning for Indian roads.

The import surface, in dependency order:

    types       the contract - every stage below speaks only these types
    perception  camera pixels          -> List[Track]
    carla_bridge  simulator state      -> List[Track]   (only file importing carla)
    prediction  List[Track]            -> List[Prediction]
    riskmap     List[Prediction]       -> grid[t, y, x]
    planner     grid                   -> Plan(steer, accel)
    pipeline    predict -> risk -> plan, in one call
    hud         draws all of it

Deliberately empty of re-exports. `from nova.planner import HybridAStarPlanner`
names the module it came from, which is the point of splitting them; a flat
`from nova import *` would hide exactly the layering this project is pitched on.

This file exists so `nova` is a real package rather than a namespace package.
Both import fine, but a namespace package silently absorbs any other directory
named `nova` that happens to be on sys.path - and the scripts prepend two
directories to sys.path before importing.
"""
