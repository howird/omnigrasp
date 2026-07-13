"""
Synthesize an OmniGrasp inference pkl entry for a known object.

Borrows body motion from a donor entry in a GRAB-format pkl (default: the
bowl entry in sample_data/grab_sample.pkl), computes the BPS shape encoding
for the object's mesh using the shared GRAB basis, and constructs a
synthetic object trajectory + hand grasp. Object-specific defaults (mesh,
trajectory strategy, grasp parameters, output path) come from
`phc.utils.object_assets.OBJECT_RECIPES` -- adding a new object means adding
a recipe there, not a new script.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.getcwd())

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tyro

from phc.utils.object_assets import OBJECT_RECIPES, ObjectName
from phc.utils.grasp_pkl import (
    GraspEntry,
    compute_bps_encoding,
    generate_hand_grasp,
    generate_object_trajectory,
    load_mesh_vertices,
    make_table_pose,
    save_grasp_entry,
    trim,
    trim_torch,
)


@dataclass
class Config:
    object_name: ObjectName
    """Which object to synthesize a pkl for; selects defaults from OBJECT_RECIPES."""
    output_path: Path | None = None
    """Where to save the pkl. Defaults to the recipe's output_path if not given."""
    contact_onset_frame: int = 45
    """First frame (of num_frames) where contact_info switches from 0 (approach) to 1 (contact)."""
    grab_pkl: Path = Path("sample_data/grab_sample.pkl")
    """GRAB-format pkl to borrow donor body motion from."""
    donor_key: str = "bowl"
    """Entry key within grab_pkl to borrow body motion from."""
    num_frames: int = 300
    fps: int = 30


def main(cfg: Config) -> None:
    recipe = OBJECT_RECIPES[cfg.object_name]

    print(f"Loading {cfg.grab_pkl}...")
    grab = joblib.load(cfg.grab_pkl)
    donor = grab[cfg.donor_key]

    print(f"Computing BPS encoding for {cfg.object_name}...")
    bps_basis = donor["obj_data"]["bps_basis"]  # (1, 512, 3)
    vertices = load_mesh_vertices(recipe.asset.mesh_path, recipe.asset.mesh_scale)
    print(f"  Vertices: {vertices.shape[0]}, bbox z: [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}] m")
    object_code = compute_bps_encoding(bps_basis, vertices)
    print(f"  object_code shape: {tuple(object_code.shape)}, range: [{object_code.min():.3f}, {object_code.max():.3f}]")

    t = cfg.num_frames
    obj_pos, obj_quat = generate_object_trajectory(recipe.trajectory, t)
    hand_trans, hand_rot = generate_hand_grasp(recipe.grasp, obj_pos, t)
    obj_pose = np.concatenate([obj_pos, obj_quat, make_table_pose(t)], axis=1)  # (t, 14)

    contact_info = np.zeros(t, dtype=np.int64)
    contact_info[cfg.contact_onset_frame:] = 1

    entry = GraspEntry(
        entry_key=recipe.entry_key,
        pose_aa=trim(donor["pose_aa"], t),
        trans_orig=trim(donor["trans_orig"], t),
        root_trans_offset=trim_torch(donor["root_trans_offset"], t),
        pose_quat_global=trim(donor["pose_quat_global"], t),
        pose_quat=trim(donor["pose_quat"], t),
        beta=donor["beta"],
        gender=donor["gender"],
        v_template=donor["v_template"],
        fps=cfg.fps,
        obj_pose=obj_pose,
        obj_info_path=recipe.asset.obj_info_path,
        hand_trans=hand_trans,
        hand_rot=hand_rot,
        contact_info=contact_info,
        object_code=object_code,
        bps_basis=bps_basis,
    )
    save_grasp_entry(entry, cfg.output_path or recipe.output_path)


if __name__ == "__main__":
    main(tyro.cli(Config))
