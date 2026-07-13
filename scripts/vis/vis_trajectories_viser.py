"""Visualize recorded robot and object trajectories with viser.

Supports two input formats:
  - .npz  from base_task.record_frame_headless(): body_pos (T,J,3), obj_state (T,7)
  - .pkl  AMASS/GRAB training data loaded via MotionLibSMPLObj
"""

from dataclasses import dataclass
from pathlib import Path
import os
import time
import numpy as np
import torch
import viser
import tyro
from typing import List, Tuple


@dataclass
class Config:
    motion_file: Path = Path("output/recordings/states.npz")
    """Path to .npz recording or .pkl AMASS/GRAB motion file."""
    port: int = 8080
    """Viser server port."""
    fps: float = 30.0
    """Playback frames per second."""


@dataclass
class TrajectoryData:
    rb_pos: np.ndarray  # (T, J, 3) rigid body positions
    obj_pos: np.ndarray  # (T, N_obj, 3) object positions
    obj_rot: np.ndarray  # (T, N_obj, 4) object rotations (xyzw)
    obj_names: List[str]
    parent_indices: List[int]
    seq_name: str


def _build_skeleton() -> List[int]:
    """Return parent indices for the SMPL-X humanoid skeleton."""
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
    from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": False,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smplx",
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "fix_height": False,
        "sim": "mujoco",
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gender_beta = np.zeros(21)
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(gender_beta[None, 1:]).float(),
        gender=gender_beta[0:1],
        objs_info=None,
    )
    os.makedirs("/tmp/smpl", exist_ok=True)
    xml_path = "/tmp/smpl/vis_robot.xml"
    smpl_robot.write_xml(xml_path)
    sk_tree = SkeletonTree.from_mjcf(xml_path)
    return sk_tree.parent_indices.tolist()


def load_npz(path: Path) -> TrajectoryData:
    """Load from base_task.record_frame_headless() output."""
    data = np.load(path)
    rb_pos = data["body_pos"]  # (T, J, 3)
    obj_state = data["obj_state"]  # (T, 13) pos+quat+vel or (T, N_obj*7)

    # _obj_states from Isaac Gym are (13,): pos(3)+quat(4)+linvel(3)+angvel(3)
    # Trim to pos+quat only if the last dim is 13.
    if obj_state.shape[-1] == 13:
        obj_state = obj_state[..., :7]  # drop velocity fields → (T, 7)
    if obj_state.ndim == 1 or obj_state.shape[-1] == 7:
        obj_state = obj_state.reshape(obj_state.shape[0], -1, 7)
    obj_pos = obj_state[..., :3]  # (T, N_obj, 3)
    obj_rot = obj_state[..., 3:]  # (T, N_obj, 4)

    N_obj = obj_pos.shape[1]
    parent_indices = _build_skeleton()

    return TrajectoryData(
        rb_pos=rb_pos,
        obj_pos=obj_pos,
        obj_rot=obj_rot,
        obj_names=[f"object_{i}" for i in range(N_obj)],
        parent_indices=parent_indices,
        seq_name=path.stem,
    )


def load_pkl(path: Path) -> TrajectoryData:
    """Load from AMASS/GRAB pkl via MotionLibSMPLObj."""
    import joblib
    from easydict import EasyDict
    from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
    from phc.utils.motion_lib_smpl_obj import MotionLibSMPLObj
    from phc.utils.motion_lib_base import FixHeightMode
    from phc.utils.flags import flags

    flags.im_eval = True
    device = torch.device("cpu")

    raw_data = joblib.load(path)
    seq_key = list(raw_data.keys())[0]
    seq_data = raw_data[seq_key]

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": False,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smplx",
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "fix_height": False,
        "sim": "mujoco",
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    v_template = seq_data.get("v_template")
    if v_template is not None:
        smpl_robot.load_from_skeleton(v_template=torch.from_numpy(v_template).float())
    else:
        gender_beta = np.zeros(21)
        smpl_robot.load_from_skeleton(
            betas=torch.from_numpy(gender_beta[None, 1:]).float(),
            gender=gender_beta[0:1],
            objs_info=None,
        )

    os.makedirs("/tmp/smpl", exist_ok=True)
    xml_path = "/tmp/smpl/vis_robot.xml"
    smpl_robot.write_xml(xml_path)
    sk_tree = SkeletonTree.from_mjcf(xml_path)

    motion_lib = MotionLibSMPLObj(
        EasyDict(
            {
                "motion_file": str(path),
                "device": device,
                "fix_height": FixHeightMode.no_fix,
                "min_length": -1,
                "max_length": -1,
                "im_eval": True,
                "multi_thread": False,
                "smpl_type": "smplx",
                "randomrize_heading": False,
            }
        )
    )
    motion_lib.load_motions(
        skeleton_trees=[sk_tree],
        gender_betas=[torch.zeros(21)],
        limb_weights=[np.zeros(10)],
        random_sample=False,
        start_idx=0,
    )

    motion_id = 0
    loaded_key = motion_lib.curr_motion_keys[0]
    motion_len = motion_lib.get_motion_length(motion_id).item()
    dt = 1.0 / 30.0
    T = int(motion_len * 30)

    print(f"Pre-computing {T} frames...")
    all_rb_pos, all_obj_pos, all_obj_rot = [], [], []
    for t_idx in range(T):
        res = motion_lib.get_motion_state(
            torch.tensor([motion_id]),
            torch.tensor([t_idx * dt]),
        )
        all_rb_pos.append(res["rg_pos"][0].numpy())
        all_obj_pos.append(res["o_rb_pos"][0].numpy())
        all_obj_rot.append(res["o_rb_rot"][0].numpy())

    obj_info: List[str] = raw_data[seq_key]["obj_data"]["obj_info"]
    obj_names = [p.split("/")[-1].split(".")[0] for p in obj_info]

    return TrajectoryData(
        rb_pos=np.stack(all_rb_pos),
        obj_pos=np.stack(all_obj_pos),
        obj_rot=np.stack(all_obj_rot),
        obj_names=obj_names,
        parent_indices=sk_tree.parent_indices.tolist(),
        seq_name=loaded_key,
    )


def make_bone_points(
    frame_rb_pos: np.ndarray, bone_pairs: List[Tuple[int, int]]
) -> np.ndarray:
    starts = frame_rb_pos[[p for p, _ in bone_pairs]]
    ends = frame_rb_pos[[c for _, c in bone_pairs]]
    return np.stack([starts, ends], axis=1)


def main(cfg: Config) -> None:
    suffix = cfg.motion_file.suffix.lower()
    if suffix == ".npz":
        traj = load_npz(cfg.motion_file)
    elif suffix == ".pkl":
        traj = load_pkl(cfg.motion_file)
    else:
        raise ValueError(f"Unsupported file format: {suffix} (expected .npz or .pkl)")

    T, J, _ = traj.rb_pos.shape
    N_obj = traj.obj_pos.shape[1]
    bone_pairs = [
        (traj.parent_indices[i], i) for i in range(J) if traj.parent_indices[i] >= 0
    ]

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=6, height=6, cell_size=0.5)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=T - 1, step=1, initial_value=0
        )
        play_btn = server.gui.add_button("Play / Pause")
    server.gui.add_markdown(
        f"**Sequence:** `{traj.seq_name}`  \n"
        f"**Frames:** {T} @ {cfg.fps:.0f} fps  \n"
        f"**Objects:** {', '.join(traj.obj_names)}"
    )

    playing = [True]

    @play_btn.on_click
    def _(_) -> None:
        playing[0] = not playing[0]

    joint_handle = server.scene.add_point_cloud(
        "/robot/joints",
        points=traj.rb_pos[0],
        colors=np.tile([50, 150, 255], (J, 1)).astype(np.uint8),
        point_size=0.025,
    )
    bone_handle = server.scene.add_line_segments(
        "/robot/bones",
        points=make_bone_points(traj.rb_pos[0], bone_pairs),
        colors=np.array([200, 200, 200], dtype=np.uint8),
        line_width=2.0,
    )

    _colors = [(255, 140, 0), (80, 200, 80), (180, 80, 220)]
    obj_handles = [
        server.scene.add_icosphere(
            f"/objects/{name}",
            radius=0.05,
            color=_colors[i % len(_colors)],
            position=tuple(traj.obj_pos[0, i].tolist()),
        )
        for i, name in enumerate(traj.obj_names)
    ]

    print(f"Open http://localhost:{cfg.port} in your browser")

    t_idx = 0
    last_update = time.time()
    frame_dt = 1.0 / cfg.fps

    while True:
        now = time.time()
        if playing[0] and (now - last_update) >= frame_dt:
            t_idx = (t_idx + 1) % T
            frame_slider.value = t_idx
            last_update = now
        else:
            t_idx = int(frame_slider.value)

        joint_handle.points = traj.rb_pos[t_idx]
        bone_handle.points = make_bone_points(traj.rb_pos[t_idx], bone_pairs)
        for i, h in enumerate(obj_handles):
            h.position = tuple(traj.obj_pos[t_idx, i].tolist())

        time.sleep(0.005)


if __name__ == "__main__":
    main(tyro.cli(Config))
