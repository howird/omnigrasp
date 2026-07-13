"""
Render OmniGrasp RL policy rollout as a full-body skeleton + bowl video.

Reads the per-frame state npz files produced by the modified base_task.py
auto_record mode (body_pos: T×52×3, obj_state: T×13) and renders them
with Open3D OffscreenRenderer.

Joint ordering (SMPLH_MUJOCO_NAMES, 52 joints):
  0  Pelvis
  1  L_Hip   2  L_Knee   3  L_Ankle   4  L_Toe
  5  R_Hip   6  R_Knee   7  R_Ankle   8  R_Toe
  9  Torso  10  Spine   11  Chest    12  Neck   13  Head
 14  L_Thorax 15  L_Shoulder 16  L_Elbow 17  L_Wrist
 18-20  L_Index1-3   21-23  L_Middle1-3  24-26  L_Pinky1-3
 27-29  L_Ring1-3    30-32  L_Thumb1-3
 33  R_Thorax 34  R_Shoulder 35  R_Elbow 36  R_Wrist
 37-39  R_Index1-3   40-42  R_Middle1-3  43-45  R_Pinky1-3
 46-48  R_Ring1-3    49-51  R_Thumb1-3
"""

from __future__ import annotations
import sys, os

sys.path.append(os.getcwd())

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm


# ── Skeleton connectivity ─────────────────────────────────────────────────────
BONES: list[tuple[int, int]] = [
    # Spine + head
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    # Left leg
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Right leg
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Left arm
    (11, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    # Left fingers
    (17, 18),
    (18, 19),
    (19, 20),
    (17, 21),
    (21, 22),
    (22, 23),
    (17, 24),
    (24, 25),
    (25, 26),
    (17, 27),
    (27, 28),
    (28, 29),
    (17, 30),
    (30, 31),
    (31, 32),
    # Right arm
    (11, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    # Right fingers
    (36, 37),
    (37, 38),
    (38, 39),
    (36, 40),
    (40, 41),
    (41, 42),
    (36, 43),
    (43, 44),
    (44, 45),
    (36, 46),
    (46, 47),
    (47, 48),
    (36, 49),
    (49, 50),
    (50, 51),
]

BODY_COLOR = np.array([0.25, 0.55, 0.95])  # blue-ish body
LEFT_COLOR = np.array([0.90, 0.30, 0.20])  # red  = left side
RIGHT_COLOR = np.array([0.20, 0.80, 0.30])  # green = right side

# Joint index sets
LEFT_JOINTS = set(range(1, 33))  # left leg + arm + fingers
RIGHT_JOINTS = set(range(5, 9)) | set(range(33, 52))  # right leg + arm + fingers


def joint_color(idx: int) -> np.ndarray:
    if idx in LEFT_JOINTS:
        return LEFT_COLOR
    if idx in RIGHT_JOINTS:
        return RIGHT_COLOR
    return BODY_COLOR


def bone_color(j0: int, j1: int) -> np.ndarray:
    """Use left/right color when both endpoints are on the same side."""
    if j0 in LEFT_JOINTS and j1 in LEFT_JOINTS:
        return LEFT_COLOR * 0.75
    if j0 in RIGHT_JOINTS and j1 in RIGHT_JOINTS:
        return RIGHT_COLOR * 0.75
    return BODY_COLOR * 0.75


# ── Geometry helpers ──────────────────────────────────────────────────────────
def sphere_at(
    pos: np.ndarray, radius: float, color: np.ndarray
) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
    s.translate(pos.astype(float))
    s.paint_uniform_color(color.tolist())
    s.compute_vertex_normals()
    return s


def cylinder_between(
    p0: np.ndarray, p1: np.ndarray, radius: float, color: np.ndarray
) -> Optional[o3d.geometry.TriangleMesh]:
    p0, p1 = p0.astype(float), p1.astype(float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    mid = (p0 + p1) * 0.5
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=8, split=1
    )
    axis = vec / length
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, axis)
    dot = np.dot(z, axis)
    if np.linalg.norm(cross) < 1e-6:
        R = np.eye(3) if dot > 0 else -np.eye(3)
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        R = sRot.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()
    cyl.rotate(R, center=np.zeros(3))
    cyl.translate(mid)
    cyl.paint_uniform_color(color.tolist())
    cyl.compute_vertex_normals()
    return cyl


def load_mesh(path: str, scale: float = 1.0) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(path)
    if scale != 1.0:
        m.scale(scale, center=np.zeros(3))
    m.compute_vertex_normals()
    return m


def apply_pose(
    tmpl: o3d.geometry.TriangleMesh, pos: np.ndarray, quat_xyzw: np.ndarray
) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(tmpl)
    m.rotate(sRot.from_quat(quat_xyzw).as_matrix(), center=np.zeros(3))
    m.translate(pos.astype(float))
    return m


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    states_file: Path = Path(
        "output/renderings/omnigrasp_neurips_grab-2026-06-19-02:16:13_states.npz"
    )
    """Per-frame state npz from the OmniGrasp rollout."""
    obj_mesh: Path = Path("phc/data/assets/mesh/grab/bowl.stl")
    """Object mesh (STL or OBJ) to render."""
    obj_scale: float = 1.0
    """Scale factor applied to obj_mesh vertices (e.g. 0.001 for mm→m)."""
    table_stl: Path = Path("phc/data/assets/mesh/grab/table.stl")
    table_pos: tuple[float, float, float] = (0.0, -0.39, 0.0)
    """Table world position. Use (0, -5, 0.73) to move table out of frame."""
    out_video: Path = Path("output/renderings/policy_skeleton.mp4")
    width: int = 1280
    height: int = 720
    fps: int = 30
    joint_radius: float = 0.015
    bone_radius: float = 0.007
    cam_smooth: int = 8


def main(cfg: Config) -> None:
    print(f"Loading {cfg.states_file}...")
    d = np.load(cfg.states_file)
    body_pos = d["body_pos"]  # (T, 52, 3)
    obj_state = d["obj_state"]  # (T, 13): pos(3) + quat(4) + vel(3) + angvel(3)
    T = body_pos.shape[0]
    print(f"  {T} frames, {body_pos.shape[1]} joints")

    print("Loading object meshes...")
    obj_tmpl = load_mesh(str(cfg.obj_mesh), scale=cfg.obj_scale)
    table_tmpl = load_mesh(str(cfg.table_stl))

    table_pos_arr = np.array(cfg.table_pos, dtype=float)
    table_quat_arr = np.array([0.0, 0.0, 0.0, 1.0])  # identity xyzw
    table_mesh = apply_pose(table_tmpl, table_pos_arr, table_quat_arr)
    table_mesh.paint_uniform_color([0.58, 0.43, 0.28])
    table_mesh.compute_vertex_normals()

    # ── Renderer setup ────────────────────────────────────────────────────────
    print("Setting up renderer...")
    renderer = rendering.OffscreenRenderer(cfg.width, cfg.height)
    renderer.scene.set_background([0.85, 0.87, 0.90, 1.0])

    mat_body = rendering.MaterialRecord()
    mat_body.shader = "defaultLit"
    mat_obj = rendering.MaterialRecord()
    mat_obj.shader = "defaultLit"
    mat_table = rendering.MaterialRecord()
    mat_table.shader = "defaultLit"
    mat_gnd = rendering.MaterialRecord()
    mat_gnd.shader = "defaultLit"

    # Ground plane
    ground = o3d.geometry.TriangleMesh.create_box(40, 40, 0.02)
    ground.translate([-20, -20, -0.02])
    ground.paint_uniform_color([0.40, 0.40, 0.40])
    ground.compute_vertex_normals()
    renderer.scene.add_geometry("ground", ground, mat_gnd)

    # Static table
    renderer.scene.add_geometry("table", table_mesh, mat_table)

    # Lighting
    renderer.scene.scene.set_sun_light([-0.4, -0.7, -1.0], [1.6, 1.5, 1.4], 90000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_indirect_light_intensity(30000)

    # Pre-register geometry names
    joint_names = [f"j{i}" for i in range(52)]
    bone_names = [f"b{i}" for i in range(len(BONES))]
    bowl_name = "bowl"

    # Camera smooth tracking
    cam_hist: list[np.ndarray] = []

    def smooth_target(pt: np.ndarray) -> np.ndarray:
        cam_hist.append(pt.copy())
        if len(cam_hist) > cfg.cam_smooth:
            cam_hist.pop(0)
        return np.mean(cam_hist, axis=0)

    # Fixed camera offset relative to humanoid root (behind & above)
    CAM_OFFSET = np.array([1.8, -1.8, 1.1])

    # ── Render loop ────────────────────────────────────────────────────────────
    cfg.out_video.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {T} frames → {cfg.out_video}")
    writer = imageio.get_writer(str(cfg.out_video), fps=cfg.fps, macro_block_size=None)

    for i in tqdm(range(T)):
        bp = body_pos[i]  # (52, 3)
        os_ = obj_state[i]  # (13,)
        bowl_pos = os_[:3]
        bowl_quat = os_[3:7]  # xyzw in Isaac Gym

        # ── Object ───────────────────────────────────────────────────────────
        bowl_m = apply_pose(obj_tmpl, bowl_pos, bowl_quat)
        bowl_m.paint_uniform_color([0.92, 0.76, 0.42])
        bowl_m.compute_vertex_normals()
        renderer.scene.remove_geometry(bowl_name)
        renderer.scene.add_geometry(bowl_name, bowl_m, mat_obj)

        # ── Joints ────────────────────────────────────────────────────────────
        for j in range(52):
            color = joint_color(j)
            renderer.scene.remove_geometry(joint_names[j])
            renderer.scene.add_geometry(
                joint_names[j], sphere_at(bp[j], cfg.joint_radius, color), mat_body
            )

        # ── Bones ─────────────────────────────────────────────────────────────
        for b, (j0, j1) in enumerate(BONES):
            cyl = cylinder_between(bp[j0], bp[j1], cfg.bone_radius, bone_color(j0, j1))
            renderer.scene.remove_geometry(bone_names[b])
            if cyl is not None:
                renderer.scene.add_geometry(bone_names[b], cyl, mat_body)

        # ── Follow camera: tracks midpoint of pelvis + bowl ──────────────────
        pelvis = bp[0]
        focus = 0.6 * pelvis + 0.4 * bowl_pos  # weighted toward humanoid
        target = smooth_target(focus + np.array([0, 0, 0.2]))
        eye = pelvis + CAM_OFFSET
        renderer.setup_camera(52.0, target, eye, np.array([0.0, 0.0, 1.0]))

        img = renderer.render_to_image()
        writer.append_data(np.asarray(img))

    writer.close()
    print(f"\nSaved: {cfg.out_video}")


if __name__ == "__main__":
    main(tyro.cli(Config))
