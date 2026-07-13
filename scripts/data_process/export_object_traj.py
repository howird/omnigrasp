"""
Headless export of a 3D pose trajectory for any object mesh, from a waypoints
JSON file (the same sidecar format authored/loaded by
`scripts/vis/create_object_traj_viser.py`).

Output: {"pos": (T,3), "rot": (T,4) xyzw, "fps": FPS}, consumed by
`phc/utils/traj_generator_3d.py`'s `real_traj` mode.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.getcwd())

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tyro

from phc.utils.object_traj_waypoints import interpolate, load_waypoints


@dataclass
class Config:
    waypoints_path: Path
    """Waypoints JSON to densify, e.g. data/presets/wide_range_stick.waypoints.json."""
    output_path: Path = Path("data/custom_object_traj.pkl")
    """Where to save the densified {pos, rot, fps} trajectory."""
    fps: float = 30.0
    """Sample rate for densification."""


def export(cfg: Config) -> None:
    waypoints = load_waypoints(cfg.waypoints_path)
    assert len(waypoints) >= 2, f"Need at least 2 waypoints in {cfg.waypoints_path}"

    n_frames = max(2, round((waypoints[-1].time - waypoints[0].time) * cfg.fps) + 1)
    t = np.linspace(waypoints[0].time, waypoints[-1].time, n_frames)
    pos, rot = interpolate(waypoints, t)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pos": pos, "rot": rot, "fps": cfg.fps}, cfg.output_path, compress=3)
    print(f"Saved {n_frames} frames to {cfg.output_path}")
    print(
        f"  pos range: x=[{pos[:,0].min():.2f},{pos[:,0].max():.2f}] "
        f"y=[{pos[:,1].min():.2f},{pos[:,1].max():.2f}] "
        f"z=[{pos[:,2].min():.2f},{pos[:,2].max():.2f}]"
    )


if __name__ == "__main__":
    export(tyro.cli(Config))
