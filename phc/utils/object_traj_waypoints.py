"""Shared waypoint data model + I/O for authoring 3D pose trajectories for any
object mesh.

Used by `scripts/vis/create_object_traj_viser.py` (interactive authoring) and
`scripts/data_process/export_object_traj.py` (headless export) so both produce
identical `{"pos", "rot", "fps"}` trajectories from the same waypoint format.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import RotationSpline
from scipy.interpolate import PchipInterpolator


@dataclass
class Waypoint:
    time: float
    position: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]  # viser convention


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    return wxyz[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    return xyzw[..., [3, 0, 1, 2]]


def interpolate(waypoints: list[Waypoint], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos (len(t),3), rot xyzw (len(t),4)) via PCHIP position / RotationSpline rotation."""
    times = np.array([wp.time for wp in waypoints])
    positions = np.array([wp.position for wp in waypoints])
    xyzw = wxyz_to_xyzw(np.array([wp.wxyz for wp in waypoints]))

    t_clipped = np.clip(t, times[0], times[-1])

    pos_interp = PchipInterpolator(times, positions, axis=0)
    pos = pos_interp(t_clipped).astype(np.float32)

    rot_spline = RotationSpline(times, sRot.from_quat(xyzw))
    rot = rot_spline(t_clipped).as_quat().astype(np.float32)
    return pos, rot


def load_waypoints(sidecar_path: Path) -> list[Waypoint]:
    if not sidecar_path.exists():
        return []
    raw = json.loads(sidecar_path.read_text())
    return sorted(
        (
            Waypoint(
                time=w["time"],
                position=tuple(w["position"]),
                wxyz=tuple(w["wxyz"]),
            )
            for w in raw
        ),
        key=lambda wp: wp.time,
    )


def save_waypoints(sidecar_path: Path, waypoints: list[Waypoint]) -> None:
    sidecar_path.write_text(json.dumps([asdict(wp) for wp in waypoints], indent=2))


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Loads a {"pos","rot"[,"fps"]} dict or a bare (T,3) position-only array (old format).

    Returns (pos (T,3) float32, rot xyzw (T,4) float32, source_fps) — rotation
    defaults to identity and source_fps to None when the file predates rotation
    or fps metadata support.
    """
    raw = joblib.load(path)
    if isinstance(raw, dict):
        pos = np.asarray(raw["pos"], dtype=np.float32)
        rot = np.asarray(raw["rot"], dtype=np.float32)
        source_fps = float(raw["fps"]) if "fps" in raw else None
    else:
        pos = np.asarray(raw, dtype=np.float32)
        rot = np.zeros((pos.shape[0], 4), dtype=np.float32)
        rot[:, 3] = 1.0  # identity, xyzw
        source_fps = None
    return pos, rot, source_fps
