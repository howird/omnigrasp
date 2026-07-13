"""
Render a GRAB grasping sequence to video using Open3D OffscreenRenderer.

The grab_sample.pkl body pose (pose_aa) is static, so we drop the body mesh
and instead render the full 32-joint hand skeleton as spheres + bones, plus
the animated bowl. This clearly shows the grasping motion.

Joint ordering (from convert_grab_smplx.py mujoco_hand_joints):
  L: 0=L_Wrist, 1-3=L_Index1-3, 4-6=L_Middle1-3, 7-9=L_Pinky1-3,
     10-12=L_Ring1-3, 13-15=L_Thumb1-3
  R: 16=R_Wrist, 17-19=R_Index1-3, 20-22=R_Middle1-3, 23-25=R_Pinky1-3,
     26-28=R_Ring1-3, 29-31=R_Thumb1-3
"""
import sys, os
sys.path.append(os.getcwd())

import joblib, numpy as np, torch, imageio
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as sRot

# ── Config ───────────────────────────────────────────────────────────────────
MOTION_FILE = "sample_data/grab_sample.pkl"
OBJECT_KEY  = "bowl"
OUT_VIDEO   = f"output/renderings/grab_{OBJECT_KEY}_skeleton.mp4"
W, H        = 1280, 720
FPS         = 30

os.makedirs("output/renderings", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {MOTION_FILE} [{OBJECT_KEY}]...")
data  = joblib.load(MOTION_FILE)
entry = data[OBJECT_KEY]

obj_data   = entry["obj_data"]
obj_pose   = obj_data["obj_pose"]     # (T, 14)
obj_info   = obj_data["obj_info"]     # [bowl.stl, table.stl]
hand_trans = obj_data["hand_trans"]   # (T, 32, 3)

T = obj_pose.shape[0]

# ── Hand skeleton connectivity ────────────────────────────────────────────────
# Each finger: Wrist → Finger1 → Finger2 → Finger3
# L_Wrist=0; fingers start at 1 (Index), 4 (Middle), 7 (Pinky), 10 (Ring), 13 (Thumb)
# R_Wrist=16; same offsets +16

def finger_chain(wrist, start):
    return [(wrist, start), (start, start+1), (start+1, start+2)]

L_BONES = (finger_chain(0,  1) + finger_chain(0,  4) + finger_chain(0,  7) +
           finger_chain(0, 10) + finger_chain(0, 13))
R_BONES = (finger_chain(16, 17) + finger_chain(16, 20) + finger_chain(16, 23) +
           finger_chain(16, 26) + finger_chain(16, 29))

ALL_BONES = L_BONES + R_BONES

L_COLOR = np.array([0.90, 0.25, 0.20])  # red  = left
R_COLOR = np.array([0.20, 0.80, 0.25])  # green = right
JOINT_RADIUS = 0.015
BONE_RADIUS  = 0.008

# ── Object meshes ─────────────────────────────────────────────────────────────
def load_stl(path):
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    return m

print("Loading object meshes...")
obj_mesh_tmpl   = load_stl(obj_info[0])
table_mesh_tmpl = load_stl(obj_info[1])

def apply_pose(tmpl, pos, quat_xyzw):
    m = o3d.geometry.TriangleMesh(tmpl)
    m.rotate(sRot.from_quat(quat_xyzw).as_matrix(), center=np.zeros(3))
    m.translate(np.array(pos, dtype=float))
    return m

# ── Geometry helpers ──────────────────────────────────────────────────────────
def sphere_at(pos, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
    s.translate(np.array(pos, dtype=float))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s

def bone_between(p0, p1, radius, color):
    """Cylinder connecting two 3-D points."""
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    mid = (p0 + p1) * 0.5
    length = np.linalg.norm(p1 - p0)
    if length < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                     height=length,
                                                     resolution=8,
                                                     split=1)
    # Default cylinder is along Z; rotate to align with (p1-p0)
    axis = (p1 - p0) / length
    z    = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, axis)
    dot   = np.dot(z, axis)
    if np.linalg.norm(cross) < 1e-6:
        R = np.eye(3) if dot > 0 else -np.eye(3)
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        R = sRot.from_rotvec(cross / np.linalg.norm(cross) * angle).as_matrix()
    cyl.rotate(R, center=np.zeros(3))
    cyl.translate(mid)
    cyl.paint_uniform_color(list(color))
    cyl.compute_vertex_normals()
    return cyl

# ── Renderer setup ────────────────────────────────────────────────────────────
print("Setting up renderer...")
renderer = rendering.OffscreenRenderer(W, H)
renderer.scene.set_background([0.06, 0.06, 0.09, 1.0])

mat_obj   = rendering.MaterialRecord(); mat_obj.shader   = "defaultLit"
mat_table = rendering.MaterialRecord(); mat_table.shader = "defaultLit"
mat_gnd   = rendering.MaterialRecord(); mat_gnd.shader   = "defaultLit"
mat_hand  = rendering.MaterialRecord(); mat_hand.shader  = "defaultLit"

ground = o3d.geometry.TriangleMesh.create_box(30, 30, 0.02)
ground.translate([-15, -15, -0.02])
ground.paint_uniform_color([0.42, 0.42, 0.42])
ground.compute_vertex_normals()
renderer.scene.add_geometry("ground", ground, mat_gnd)

renderer.scene.scene.set_sun_light([-0.4, -0.7, -1.0], [1.6, 1.5, 1.4], 90000)
renderer.scene.scene.enable_sun_light(True)
renderer.scene.scene.set_indirect_light_intensity(30000)

# ── Camera: close-up on the action (hands + bowl) ─────────────────────────────
# Scene centroid ~(0, -0.39, 0.89). Camera from front-right, looking in.
CAM_EYE = np.array([1.8, -1.8, 1.4])
CAM_UP  = np.array([0.0, 0.0, 1.0])
SMOOTH  = 6
cam_hist = []

def smooth_lookat(hist, pt):
    hist.append(pt.copy())
    if len(hist) > SMOOTH:
        hist.pop(0)
    return np.mean(hist, axis=0)

# Pre-build geometry name lists so we can remove them each frame
JOINT_NAMES = [f"j{i}" for i in range(32)]
BONE_NAMES  = [f"b{i}" for i in range(len(ALL_BONES))]

# ── Render loop ───────────────────────────────────────────────────────────────
print(f"Rendering {T} frames → {OUT_VIDEO}")
writer = imageio.get_writer(OUT_VIDEO, fps=FPS, macro_block_size=None)

for i in tqdm(range(T)):
    ht = hand_trans[i]   # (32, 3)

    # ── Bowl ─────────────────────────────────────────────────────────────────
    o_pos  = obj_pose[i, :3]
    o_quat = obj_pose[i, 3:7]
    obj_m  = apply_pose(obj_mesh_tmpl, o_pos, o_quat)
    obj_m.paint_uniform_color([0.92, 0.76, 0.42])
    obj_m.compute_vertex_normals()
    renderer.scene.remove_geometry("obj")
    renderer.scene.add_geometry("obj", obj_m, mat_obj)

    # ── Table ─────────────────────────────────────────────────────────────────
    t_pos  = obj_pose[i, 7:10]
    t_quat = obj_pose[i, 10:14]
    tbl_m  = apply_pose(table_mesh_tmpl, t_pos, t_quat)
    tbl_m.paint_uniform_color([0.58, 0.43, 0.28])
    tbl_m.compute_vertex_normals()
    renderer.scene.remove_geometry("table")
    renderer.scene.add_geometry("table", tbl_m, mat_table)

    # ── Hand joints (spheres) ────────────────────────────────────────────────
    for j in range(32):
        color = L_COLOR if j < 16 else R_COLOR
        renderer.scene.remove_geometry(JOINT_NAMES[j])
        renderer.scene.add_geometry(JOINT_NAMES[j],
            sphere_at(ht[j], JOINT_RADIUS, color), mat_hand)

    # ── Bones (cylinders) ────────────────────────────────────────────────────
    for b, (j0, j1) in enumerate(ALL_BONES):
        color = L_COLOR * 0.7 if j0 < 16 else R_COLOR * 0.7
        cyl = bone_between(ht[j0], ht[j1], BONE_RADIUS, color)
        renderer.scene.remove_geometry(BONE_NAMES[b])
        if cyl is not None:
            renderer.scene.add_geometry(BONE_NAMES[b], cyl, mat_hand)

    # ── Camera tracks midpoint of wrists + bowl ───────────────────────────────
    lw = ht[0]; rw = ht[16]; bowl = np.array(o_pos)
    focus = 0.35 * lw + 0.35 * rw + 0.30 * bowl
    focus[0] = 0.0
    target = smooth_lookat(cam_hist, focus)
    renderer.setup_camera(52.0, target, CAM_EYE, CAM_UP)

    img = renderer.render_to_image()
    writer.append_data(np.asarray(img))

writer.close()
print(f"\nSaved: {OUT_VIDEO}")
