"""Shared helpers for synthesizing an OmniGrasp inference pkl entry for a new
object: mesh loading + BPS shape encoding, a synthetic object trajectory, a
synthetic hand grasp, and the on-disk entry-dict assembly.

Used by `scripts/data_process/create_object_pkl.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import torch
import trimesh
from typing_extensions import assert_never

from phc.utils.point_utils import normalize_points_byshapenet

TABLE_STL: Path = Path("phc/data/assets/mesh/grab/table.stl")
TABLE_OUT_OF_FRAME_POS: tuple[float, float, float] = (0.0, -5.0, 0.73)


def trim(arr: np.ndarray, t: int) -> np.ndarray:
    """Trim or tile an array to exactly t frames along axis 0."""
    if len(arr) >= t:
        return arr[:t]
    reps = (t // len(arr)) + 1
    return np.tile(arr, (reps,) + (1,) * (arr.ndim - 1))[:t]


def trim_torch(arr: torch.Tensor, t: int) -> torch.Tensor:
    """Trim or tile a torch tensor to exactly t frames along dim 0."""
    if arr.shape[0] >= t:
        return arr[:t]
    reps = (t // arr.shape[0]) + 1
    return arr.repeat((reps,) + (1,) * (arr.dim() - 1))[:t]


def load_mesh_vertices(mesh_path: Path, mesh_scale: float) -> torch.Tensor:
    """Load a mesh (.stl or .obj, via trimesh) and return its scaled vertices."""
    mesh = trimesh.load_mesh(str(mesh_path))
    vertices = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32))
    return vertices * mesh_scale


def compute_bps_encoding(bps_basis: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """BPS shape code: for each basis point, the distance to the nearest surface point."""
    points_norm = normalize_points_byshapenet(vertices)  # (1, 2048, 3)
    return torch.cdist(bps_basis, points_norm).min(dim=-1).values


def make_table_pose(t: int) -> np.ndarray:
    """Table placed out of frame, identity orientation, tiled over t frames. (t, 7)"""
    pos = np.tile(np.array(TABLE_OUT_OF_FRAME_POS, dtype=np.float32), (t, 1))
    quat = np.zeros((t, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    return np.concatenate([pos, quat], axis=1)


# ── Object trajectory ────────────────────────────────────────────────────────


@dataclass
class LinearApproach:
    """Fixed 2-point linear interpolation, identity rotation throughout."""
    start: tuple[float, float, float] = (0.0, -0.3, 0.881)
    end: tuple[float, float, float] = (0.0, 0.3, 0.881)


@dataclass
class RandomSmoothApproach:
    """Randomized smooth trajectory via PCHIP through waypoints, plus a random yaw sweep."""
    seed: int = 42
    start: tuple[float, float, float] = (0.05, -0.5, 0.85)
    n_waypoints: int = 6
    x_range: tuple[float, float] = (-0.3, 0.3)
    y_range: tuple[float, float] = (-0.7, -0.3)
    z_range: tuple[float, float] = (0.75, 1.15)
    yaw_sweep_range: tuple[float, float] = (-1.5707963, 1.5707963)  # relative to a random start yaw


ObjectTrajectory = Union[LinearApproach, RandomSmoothApproach]  # runtime alias, not an annotation -- `|` needs 3.10


def generate_object_trajectory(traj: ObjectTrajectory, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos (t,3), quat xyzw (t,4)) for the object over t frames."""
    if isinstance(traj, LinearApproach):
        alpha = np.linspace(0, 1, t, dtype=np.float32)[:, None]
        start = np.array(traj.start, dtype=np.float32)
        end = np.array(traj.end, dtype=np.float32)
        pos = (1 - alpha) * start + alpha * end
        quat = np.zeros((t, 4), dtype=np.float32)
        quat[:, 3] = 1.0
        return pos, quat
    elif isinstance(traj, RandomSmoothApproach):
        from scipy.interpolate import PchipInterpolator
        from scipy.spatial.transform import Rotation as sRot

        rng = np.random.default_rng(seed=traj.seed)
        t_wp = np.linspace(0, 1, traj.n_waypoints)
        waypoints = np.zeros((traj.n_waypoints, 3), dtype=np.float64)
        waypoints[0] = traj.start
        for i in range(1, traj.n_waypoints):
            waypoints[i, 0] = rng.uniform(*traj.x_range)
            waypoints[i, 1] = rng.uniform(*traj.y_range)
            waypoints[i, 2] = rng.uniform(*traj.z_range)
        t_dense = np.linspace(0, 1, t)
        pos = PchipInterpolator(t_wp, waypoints, axis=0)(t_dense).astype(np.float32)

        yaw_start = rng.uniform(-np.pi, np.pi)
        yaw_end = yaw_start + rng.uniform(*traj.yaw_sweep_range)
        yaws = np.linspace(yaw_start, yaw_end, t)
        quat = sRot.from_euler("z", yaws).as_quat().astype(np.float32)
        return pos, quat
    else:
        assert_never(traj)


# ── Hand grasp ────────────────────────────────────────────────────────────────


@dataclass
class AbsoluteShaftHeight:
    """Hands follow the object's (x, y) each frame but sit at a fixed world z,
    independent of the object's z position."""
    height: float = 0.85


@dataclass
class RelativeShaftOffset:
    """Hands follow the object's full (x, y, z) each frame, offset by a fixed z delta."""
    z_offset: float = 0.05


HandGraspZMode = Union[AbsoluteShaftHeight, RelativeShaftOffset]  # runtime alias, not an annotation -- `|` needs 3.10


@dataclass
class HandGraspParams:
    z_mode: HandGraspZMode
    jitter_std: float = 0.02
    x_offset: float = 0.06
    seed: int = 42


def generate_hand_grasp(
    params: HandGraspParams, object_pos: np.ndarray, t: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hand_trans (t,32,3), hand_rot (t,32,4) identity) for 16 left + 16 right joints."""
    rng = np.random.default_rng(seed=params.seed)
    hand_l_local = rng.standard_normal((16, 3)).astype(np.float32) * params.jitter_std
    hand_r_local = rng.standard_normal((16, 3)).astype(np.float32) * params.jitter_std
    hand_l_local[:, 0] -= params.x_offset
    hand_r_local[:, 0] += params.x_offset

    hand_trans = np.zeros((t, 32, 3), dtype=np.float32)
    if isinstance(params.z_mode, AbsoluteShaftHeight):
        hand_l_local[:, 2] += params.z_mode.height
        hand_r_local[:, 2] += params.z_mode.height
        for i in range(t):
            xy_offset = np.array([object_pos[i, 0], object_pos[i, 1], 0.0], dtype=np.float32)
            hand_trans[i, :16] = hand_l_local + xy_offset
            hand_trans[i, 16:] = hand_r_local + xy_offset
    elif isinstance(params.z_mode, RelativeShaftOffset):
        hand_l_local[:, 2] += params.z_mode.z_offset
        hand_r_local[:, 2] += params.z_mode.z_offset
        for i in range(t):
            hand_trans[i, :16] = hand_l_local + object_pos[i]
            hand_trans[i, 16:] = hand_r_local + object_pos[i]
    else:
        assert_never(params.z_mode)

    hand_rot = np.zeros((t, 32, 4), dtype=np.float32)
    hand_rot[..., 3] = 1.0
    return hand_trans, hand_rot


# ── Entry assembly ────────────────────────────────────────────────────────────


@dataclass
class GraspEntry:
    """Assembled fields for one pkl entry, ready for save_grasp_entry."""
    entry_key: str
    pose_aa: np.ndarray
    trans_orig: np.ndarray
    root_trans_offset: torch.Tensor
    pose_quat_global: np.ndarray
    pose_quat: np.ndarray
    beta: Any
    gender: Any
    v_template: Any
    fps: int
    obj_pose: np.ndarray
    obj_info_path: str
    hand_trans: np.ndarray
    hand_rot: np.ndarray
    contact_info: np.ndarray
    object_code: torch.Tensor
    bps_basis: torch.Tensor


def save_grasp_entry(entry: GraspEntry, out_path: Path) -> None:
    """Assemble the on-disk dict shape and joblib.dump it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pose_aa": entry.pose_aa,
        "trans_orig": entry.trans_orig,
        "root_trans_offset": entry.root_trans_offset,
        "pose_quat_global": entry.pose_quat_global,
        "pose_quat": entry.pose_quat,
        "beta": entry.beta,
        "gender": entry.gender,
        "v_template": entry.v_template,
        "fps": entry.fps,
        "obj_data": {
            "obj_pose": entry.obj_pose,
            "obj_info": [entry.obj_info_path, str(TABLE_STL)],
            "hand_trans": entry.hand_trans,
            "hand_rot": entry.hand_rot,
            "contact_info": entry.contact_info,
            "object_code": entry.object_code,
            "bps_basis": entry.bps_basis,
        },
    }
    joblib.dump({entry.entry_key: payload}, out_path, compress=3)
    print(f"Saved: {out_path}")
